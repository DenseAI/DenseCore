/**
 * @file scheduler.cpp
 * @brief Scheduler implementation for continuous batching inference
 *
 * Implements the Scheduler class defined in scheduler.h.
 * Responsible for managing request queues, scheduling prefill/decode batches,
 * and handling preemption based on memory pressure.
 */

#include "scheduler.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <unordered_set>

#include "densecore/utils/logging.h"

namespace densecore {

namespace {

bool EqualsIgnoreCase(const char* lhs, const char* rhs) {
    if (!lhs || !rhs) {
        return false;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        const unsigned char lhs_ch = static_cast<unsigned char>(*lhs);
        const unsigned char rhs_ch = static_cast<unsigned char>(*rhs);
        if (std::tolower(lhs_ch) != std::tolower(rhs_ch)) {
            return false;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

bool ParseEnvBool(const char* value, bool default_value) {
    if (!value || *value == '\0') {
        return default_value;
    }

    if (EqualsIgnoreCase(value, "1") || EqualsIgnoreCase(value, "true") || EqualsIgnoreCase(value, "yes") ||
        EqualsIgnoreCase(value, "on")) {
        return true;
    }

    if (EqualsIgnoreCase(value, "0") || EqualsIgnoreCase(value, "false") || EqualsIgnoreCase(value, "no") ||
        EqualsIgnoreCase(value, "off")) {
        return false;
    }

    return default_value;
}

int ParseEnvInt(const char* value, int default_value) {
    if (!value || *value == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }

    if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }

    return static_cast<int>(parsed);
}

size_t ScheduledSeqCount(const SchedulerOutput& output) {
    return output.prefill_seq_ids.size() + output.decode_seq_ids.size();
}

}  // namespace

// ============================================================================
// Scheduler Implementation
// ============================================================================

Scheduler::Scheduler(BlockManager* block_manager, const SchedulerConfig& config)
    : block_manager_(block_manager), config_(config) {
    decode_homogeneous_batch_n_past_ =
        ParseEnvBool(std::getenv("DENSECORE_SCHED_DECODE_HOMOGENEOUS_N_PAST"), /*default_value=*/false);
    config_.enable_mixed_prefill_decode =
        ParseEnvBool(std::getenv("DENSECORE_SCHED_ENABLE_MIXED_PREFILL_DECODE"), config_.enable_mixed_prefill_decode);
    config_.max_mixed_prefill_tokens = std::max(
        1, ParseEnvInt(std::getenv("DENSECORE_SCHED_MAX_MIXED_PREFILL_TOKENS"), config_.max_mixed_prefill_tokens));
}

int Scheduler::AddRequest(int request_id, int prompt_len, int max_output_len, int priority,
                          const std::vector<int>* prefix_tokens, bool allow_chunked_prefill) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Fast reject impossible requests:
    // when chunking is disabled, prompt must fit in a single scheduling budget.
    if (!allow_chunked_prefill && prompt_len > config_.max_num_batched_tokens) {
        LOG_WARN("Scheduler rejected request ", request_id, " (prompt_len=", prompt_len,
                 " exceeds max_num_batched_tokens=", config_.max_num_batched_tokens, " with chunked prefill disabled)");
        return -1;
    }

    SequenceGroup group;
    group.request_id = request_id;
    group.sequence_ids = {next_seq_id_++};
    group.prompt_len = prompt_len;
    group.max_output_len = max_output_len;
    group.priority = priority;
    group.arrival_time = std::chrono::steady_clock::now();
    group.num_tokens_to_process = prompt_len;

    // Check for prefix cache hit with multi-stage collision verification
    if (prefix_tokens && !prefix_tokens->empty()) {
        uint64_t hash = BlockManager::ComputeTokenHash(prefix_tokens->data(), prompt_len);
        int cached_block = block_manager_->FindCachedBlockWithVerification(hash, prefix_tokens->data(), prompt_len);
        if (cached_block >= 0) {
            group.shared_prefix_len = prompt_len;
            group.shared_block_ids.push_back(cached_block);
        }
    }

    waiting_queue_.push_back(group);
    seq_status_[group.sequence_ids[0]] = SequenceStatus::WAITING;

    // Track for smart preemption
    int seq_id = group.sequence_ids[0];
    seq_priority_[seq_id] = priority;
    seq_arrival_[seq_id] = group.arrival_time;
    seq_generated_tokens_[seq_id] = 0;
    seq_context_len_[seq_id] = group.shared_prefix_len;
    seq_prompt_len_[seq_id] = prompt_len;
    seq_allow_chunked_prefill_[seq_id] = allow_chunked_prefill;

    return seq_id;
}

void Scheduler::RemoveRequest(int seq_id, bool finished) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Update status (for tracking)
    if (seq_status_.find(seq_id) != seq_status_.end()) {
        seq_status_[seq_id] = finished ? SequenceStatus::FINISHED : SequenceStatus::CANCELLED;
    }

    CleanupSequenceState(seq_id, /*erase_from_waiting_queue=*/true, /*free_block_manager_blocks=*/false);
}

void Scheduler::CleanupSequenceState(int seq_id, bool erase_from_waiting_queue, bool free_block_manager_blocks) {
    // Free resources
    FreeSequence(seq_id);

    if (free_block_manager_blocks) {
        auto seq_blocks_it = seq_block_ids_.find(seq_id);
        if (seq_blocks_it != seq_block_ids_.end() && !seq_blocks_it->second.empty()) {
            block_manager_->Free(seq_blocks_it->second);
        }
    }

    if (erase_from_waiting_queue) {
        waiting_queue_.erase(std::remove_if(waiting_queue_.begin(), waiting_queue_.end(),
                                            [seq_id](const SequenceGroup& g) {
                                                return std::find(g.sequence_ids.begin(), g.sequence_ids.end(),
                                                                 seq_id) != g.sequence_ids.end();
                                            }),
                             waiting_queue_.end());
    }

    // Remove from running/swapped sets
    running_seqs_.erase(seq_id);
    swapped_seqs_.erase(seq_id);

    // Clean up tracking maps
    seq_priority_.erase(seq_id);
    seq_arrival_.erase(seq_id);
    seq_generated_tokens_.erase(seq_id);
    seq_context_len_.erase(seq_id);
    seq_prompt_len_.erase(seq_id);
    seq_allow_chunked_prefill_.erase(seq_id);
    seq_predicted_experts_.erase(seq_id);
    seq_block_ids_.erase(seq_id);
}

bool Scheduler::HasRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !waiting_queue_.empty() || !running_seqs_.empty();
}

void Scheduler::FreeSequence(int seq_id) {
    // Placeholder for releasing sequence-specific resources
    // In a full implementation, this might release pre-allocated blocks
    // if not managed by BlockManager globally.
    // Since BlockManager handles global blocks, we just ensure
    // internal bookkeeping is clean.
    (void)seq_id;
}

SchedulerOutput Scheduler::Schedule() {
    std::lock_guard<std::mutex> lock(mutex_);

    SchedulerOutput output;

    // 1. Check memory pressure
    float memory_usage = GetMemoryUsage();

    if (memory_usage > config_.watermark_high) {
        // Need to preempt some sequences
        PreemptSequences(output);
    }

    // 2. Try to swap in sequences if memory is available
    ScheduleSwapped(output);

    // 3. Phase isolation policy: preserve the current decode-first isolated path
    // by default, but optionally admit a bounded prefill chunk into a decode
    // iteration when the decode context bucket is homogeneous.
    if (config_.isolate_prefill_decode) {
        const bool has_waiting = !waiting_queue_.empty();
        const bool has_running = !running_seqs_.empty();

        bool schedule_prefill_first = false;
        if (has_waiting) {
            if (!has_running) {
                schedule_prefill_first = true;
            } else if (consecutive_decode_batches_ >= config_.max_consecutive_decode_batches) {
                schedule_prefill_first = true;
            }
        }

        if (schedule_prefill_first) {
            ScheduleWaiting(output);
            if (!output.prefill_seq_ids.empty()) {
                consecutive_decode_batches_ = 0;
                return output;
            }
        }

        ScheduleRunning(output);
        if (!output.decode_seq_ids.empty()) {
            bool mixed_prefill_admitted = false;
            if (config_.enable_mixed_prefill_decode && !waiting_queue_.empty()) {
                int decode_context_bucket = -1;
                bool homogeneous_decode_context = true;
                for (int seq_id : output.decode_seq_ids) {
                    const int seq_context = GetSequenceContextLen(seq_id);
                    if (decode_context_bucket < 0) {
                        decode_context_bucket = seq_context;
                    } else if (decode_context_bucket != seq_context) {
                        homogeneous_decode_context = false;
                        break;
                    }
                }

                if (homogeneous_decode_context && decode_context_bucket >= 0) {
                    const size_t prefill_before = output.prefill_seq_ids.size();
                    const int remaining_tokens = std::max(0, config_.max_num_batched_tokens - output.total_tokens);
                    const int mixed_prefill_cap = std::min(config_.max_mixed_prefill_tokens,
                                                           std::min(config_.max_prefill_tokens, remaining_tokens));
                    if (mixed_prefill_cap > 0) {
                        output.batch_context_len = decode_context_bucket;
                        ScheduleWaiting(output, mixed_prefill_cap);
                        mixed_prefill_admitted = output.prefill_seq_ids.size() > prefill_before;
                    }
                }
            }

            if (mixed_prefill_admitted) {
                consecutive_decode_batches_ = 0;
            } else {
                consecutive_decode_batches_++;
            }
            return output;
        }

        // Decode had nothing schedulable, try prefill as fallback.
        ScheduleWaiting(output);
        if (!output.prefill_seq_ids.empty()) {
            consecutive_decode_batches_ = 0;
        }
        return output;
    }

    // Legacy mixed-mode scheduling (for compatibility/debug).
    ScheduleRunning(output);
    ScheduleWaiting(output);

    // std::cerr << "[DEBUG] Scheduler::Schedule exit. Empty? " <<
    // output.IsEmpty() << std::endl;
    return output;
}

int Scheduler::ForkSequence(int parent_seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    int new_seq_id = next_seq_id_++;
    seq_status_[new_seq_id] = seq_status_[parent_seq_id];

    // ==========================================================================
    // Copy-on-Write Implementation (Patent Compliance)
    // ==========================================================================
    // When forking a sequence (e.g., for beam search or parallel sampling),
    // we share the parent's KV cache blocks by incrementing their ref_count.
    // This avoids immediate copying. When either sequence modifies a shared
    // block, CopyOnWrite() must be called to create a private copy.
    // ==========================================================================
    auto parent_blocks_it = seq_block_ids_.find(parent_seq_id);
    if (parent_blocks_it != seq_block_ids_.end()) {
        const auto& parent_blocks = parent_blocks_it->second;
        std::vector<int> forked_blocks;
        forked_blocks.reserve(parent_blocks.size());

        for (int block_id : parent_blocks) {
            // Fork increases ref_count, enabling CoW semantics
            int forked_id = block_manager_->Fork(block_id);
            if (forked_id >= 0) {
                forked_blocks.push_back(forked_id);
            }
        }

        // Child sequence shares all parent blocks (with incremented ref_count)
        seq_block_ids_[new_seq_id] = forked_blocks;
    }

    // Copy other metadata from parent
    auto priority_it = seq_priority_.find(parent_seq_id);
    if (priority_it != seq_priority_.end()) {
        seq_priority_[new_seq_id] = priority_it->second;
    }

    auto arrival_it = seq_arrival_.find(parent_seq_id);
    if (arrival_it != seq_arrival_.end()) {
        seq_arrival_[new_seq_id] = arrival_it->second;
    }

    seq_generated_tokens_[new_seq_id] = seq_generated_tokens_[parent_seq_id];
    seq_context_len_[new_seq_id] = GetSequenceContextLen(parent_seq_id);

    return new_seq_id;
}

SequenceStatus Scheduler::GetStatus(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_status_.find(seq_id);
    if (it == seq_status_.end()) return SequenceStatus::FINISHED;
    return it->second;
}

void Scheduler::UpdateProgress(int seq_id, int tokens_generated) {
    std::lock_guard<std::mutex> lock(mutex_);
    seq_generated_tokens_[seq_id] += tokens_generated;
    seq_context_len_[seq_id] += tokens_generated;
}

// =============================================================================
// EnsureBlockWritable - Copy-on-Write Trigger (Patent Compliance)
// =============================================================================
// This function must be called before modifying any KV cache block that may be
// shared with another sequence (e.g., after ForkSequence for beam search).
// If the block's ref_count > 1, CopyOnWrite() allocates a new private block.
// =============================================================================
int Scheduler::EnsureBlockWritable(int seq_id, int block_index,
                                   const std::function<void(int src, int dst)>& copy_callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_block_ids_.find(seq_id);
    if (it == seq_block_ids_.end()) {
        return -1;  // Sequence not found
    }

    auto& blocks = it->second;
    if (block_index < 0 || block_index >= static_cast<int>(blocks.size())) {
        return -1;  // Invalid block index
    }

    int old_block_id = blocks[block_index];

    // Check if block is shared and needs CoW
    if (block_manager_->IsShared(old_block_id)) {
        // Pass the callback to block manager to perform the actual copy during the transaction
        int new_block_id = block_manager_->CopyOnWrite(old_block_id, copy_callback);
        if (new_block_id >= 0 && new_block_id != old_block_id) {
            // Update block table to use the new private copy
            blocks[block_index] = new_block_id;
            return new_block_id;
        }
    }

    // Block was not shared, or CopyOnWrite returned the same block
    return old_block_id;
}

// =============================================================================
// SetPredictedExperts - MoE Expert Prediction (Patent Claim 4)
// =============================================================================
// After MoE routing selects experts for a token, call this to record them.
// The scheduler uses this for expert-coverage-aware batching: requests using
// similar experts are grouped together to maximize L3 cache utilization.
// =============================================================================
void Scheduler::SetPredictedExperts(int seq_id, const std::vector<int>& experts) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store in the dedicated map for all sequences (running or waiting)
    seq_predicted_experts_[seq_id] = experts;

    // Also update the waiting queue's SequenceGroup if the sequence is waiting
    for (auto& group : waiting_queue_) {
        for (int sid : group.sequence_ids) {
            if (sid == seq_id) {
                group.predicted_experts = experts;
                return;
            }
        }
    }
}

Scheduler::Stats Scheduler::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats stats;
    stats.waiting_count = waiting_queue_.size();
    stats.running_count = running_seqs_.size();
    stats.swapped_count = swapped_seqs_.size();
    stats.memory_usage = GetMemoryUsage();
    return stats;
}

float Scheduler::GetMemoryUsage() const {
    if (block_manager_->num_blocks == 0) return 1.0f;  // Treat as full to trigger preemption/safety check
    int used = block_manager_->GetUsedBlockCount();
    int total = block_manager_->num_blocks;
    return (float)used / total;
}

void Scheduler::PreemptSequences(SchedulerOutput& output) {
    while (GetMemoryUsage() > config_.watermark_low && !running_seqs_.empty()) {
        int victim = SelectPreemptionVictim();
        if (victim < 0) break;

        running_seqs_.erase(victim);
        swapped_seqs_.insert(victim);
        seq_status_[victim] = SequenceStatus::SWAPPED;
        output.preempted_seq_ids.push_back(victim);
        seq_block_ids_.erase(victim);
    }
}

int Scheduler::SelectPreemptionVictim() {
    if (running_seqs_.empty()) return -1;

    int best_victim = -1;
    int best_priority = -1;
    int best_progress = INT_MAX;
    std::chrono::steady_clock::time_point best_arrival;

    for (int seq_id : running_seqs_) {
        int priority = GetSequencePriority(seq_id);
        int progress = GetSequenceProgress(seq_id);
        auto arrival = GetSequenceArrival(seq_id);

        bool is_better_victim = false;

        if (config_.enable_priority) {
            if (priority > best_priority) {
                is_better_victim = true;
            } else if (priority == best_priority) {
                if (progress < best_progress) {
                    is_better_victim = true;
                } else if (progress == best_progress) {
                    if (best_victim < 0 || arrival > best_arrival) {
                        is_better_victim = true;
                    }
                }
            }
        } else {
            if (progress < best_progress) {
                is_better_victim = true;
            } else if (progress == best_progress) {
                if (best_victim < 0 || arrival > best_arrival) {
                    is_better_victim = true;
                }
            }
        }

        if (is_better_victim) {
            best_victim = seq_id;
            best_priority = priority;
            best_progress = progress;
            best_arrival = arrival;
        }
    }

    return best_victim;
}

int Scheduler::GetSequencePriority(int seq_id) const {
    auto it = seq_priority_.find(seq_id);
    return (it != seq_priority_.end()) ? it->second : 100;
}

int Scheduler::GetSequenceProgress(int seq_id) const {
    auto it = seq_generated_tokens_.find(seq_id);
    return (it != seq_generated_tokens_.end()) ? it->second : 0;
}

int Scheduler::GetSequenceContextLen(int seq_id) const {
    auto it = seq_context_len_.find(seq_id);
    return (it != seq_context_len_.end()) ? it->second : 0;
}

std::chrono::steady_clock::time_point Scheduler::GetSequenceArrival(int seq_id) const {
    auto it = seq_arrival_.find(seq_id);
    return (it != seq_arrival_.end()) ? it->second : std::chrono::steady_clock::now();
}

void Scheduler::ScheduleRunning(SchedulerOutput& output) {
    int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;
    std::unordered_set<int> active_experts;

    std::vector<int> running_ids(running_seqs_.begin(), running_seqs_.end());
    std::sort(running_ids.begin(), running_ids.end(), [this](int a, int b) {
        int pri_a = GetSequencePriority(a);
        int pri_b = GetSequencePriority(b);
        if (pri_a != pri_b) return pri_a < pri_b;
        return GetSequenceArrival(a) < GetSequenceArrival(b);
    });

    int target_context_len = -1;

    for (int seq_id : running_ids) {
        if (ScheduledSeqCount(output) >= static_cast<size_t>(config_.max_num_seqs)) {
            break;
        }
        if (tokens_budget <= 0) {
            break;
        }

        const int seq_context = GetSequenceContextLen(seq_id);
        if (decode_homogeneous_batch_n_past_ && target_context_len >= 0 && seq_context != target_context_len) {
            continue;
        }

        if (config_.enable_moe_clustering) {
            auto it = seq_predicted_experts_.find(seq_id);
            if (it != seq_predicted_experts_.end() && !it->second.empty()) {
                int new_experts = 0;
                for (int e : it->second) {
                    if (active_experts.find(e) == active_experts.end()) {
                        new_experts++;
                    }
                }

                if (static_cast<int>(active_experts.size()) + new_experts > config_.max_active_experts) {
                    float rand_val = static_cast<float>(seq_id % 100) / 100.0f;
                    if (rand_val < config_.moe_batch_strictness) {
                        continue;
                    }
                }

                for (int e : it->second) {
                    active_experts.insert(e);
                }
            }
        }

        output.decode_seq_ids.push_back(seq_id);
        output.num_decode_tokens++;
        output.total_tokens++;
        if (target_context_len < 0) {
            target_context_len = seq_context;
        }
        output.batch_context_len = target_context_len;
        tokens_budget--;

        if (tokens_budget <= 0) break;
    }
}

void Scheduler::ScheduleWaiting(SchedulerOutput& output, int prefill_token_cap) {
    int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;

    std::vector<SequenceGroup> sorted_queue(waiting_queue_.begin(), waiting_queue_.end());
    std::sort(sorted_queue.begin(), sorted_queue.end(), [](const SequenceGroup& a, const SequenceGroup& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.arrival_time < b.arrival_time;
    });

    std::deque<SequenceGroup> still_waiting;

    // Track active experts for MoE-aware batching
    std::unordered_set<int> active_experts;
    const bool extending_decode_batch = !output.decode_seq_ids.empty();
    int target_context_len = output.batch_context_len;

    for (auto& group : sorted_queue) {
        if (ScheduledSeqCount(output) >= static_cast<size_t>(config_.max_num_seqs)) {
            still_waiting.push_back(group);
            continue;
        }

        const int seq_id = group.sequence_ids[0];
        const int remaining = group.num_tokens_to_process;
        if (remaining <= 0) {
            running_seqs_.insert(seq_id);
            seq_status_[seq_id] = SequenceStatus::RUNNING;
            continue;
        }

        auto chunk_policy_it = seq_allow_chunked_prefill_.find(seq_id);
        const bool allow_chunk = (chunk_policy_it == seq_allow_chunked_prefill_.end()) ? true : chunk_policy_it->second;
        const bool can_chunk = config_.enable_chunked_prefill && allow_chunk;

        if (!can_chunk && remaining > config_.max_num_batched_tokens) {
            LOG_WARN("Scheduler cancelling seq ", seq_id, " (remaining prefill=", remaining,
                     " exceeds max_num_batched_tokens=", config_.max_num_batched_tokens,
                     " with chunked prefill disabled)");
            seq_status_[seq_id] = SequenceStatus::CANCELLED;
            CleanupSequenceState(seq_id, /*erase_from_waiting_queue=*/false, /*free_block_manager_blocks=*/true);
            continue;
        }

        int prefill_budget = tokens_budget;
        if (prefill_token_cap >= 0) {
            prefill_budget = std::min(prefill_budget, prefill_token_cap);
        }
        if (can_chunk) {
            prefill_budget = std::min(prefill_budget, config_.max_prefill_tokens);
        }
        if (!can_chunk && remaining > prefill_budget) {
            still_waiting.push_back(group);
            continue;
        }
        const int tokens_needed = can_chunk ? std::min(remaining, prefill_budget) : remaining;
        if (tokens_needed <= 0) {
            still_waiting.push_back(group);
            continue;
        }

        const int group_context = GetSequenceContextLen(seq_id);
        // Mixed prefill+decode batches are only safe when the waiting prefill
        // group already lives in the same retained-history bucket as the decode
        // rows. The worker always lays out prefill rows before decode rows, and
        // graph-wide KV retention is still derived from batch.n_past[0], so
        // admitting a mismatched prefill group would corrupt decode history.
        if (config_.enforce_homogeneous_batch_n_past && target_context_len >= 0 &&
            group_context != target_context_len) {
            still_waiting.push_back(group);
            continue;
        }

        // =====================================================================
        // MoE Expert Coverage Control
        // =====================================================================
        // If MoE clustering is enabled and this request has predicted experts,
        // check if adding it would exceed the max_active_experts threshold.
        // With strictness=1.0, always defer; with strictness=0.0, always admit.
        // =====================================================================
        if (config_.enable_moe_clustering && !group.predicted_experts.empty()) {
            int new_experts = 0;
            for (int e : group.predicted_experts) {
                if (active_experts.find(e) == active_experts.end()) {
                    new_experts++;
                }
            }

            if (static_cast<int>(active_experts.size()) + new_experts > config_.max_active_experts) {
                // Would exceed expert budget - use strictness to decide
                // strictness=1.0 means always defer, strictness=0.0 means always admit
                // We use a simple threshold: defer if strictness > random
                // For determinism, we use strictness as probability threshold
                float rand_val = static_cast<float>(group.request_id % 100) / 100.0f;
                if (rand_val < config_.moe_batch_strictness) {
                    still_waiting.push_back(group);
                    continue;
                }
            }

            // Track this request's experts as active
            for (int e : group.predicted_experts) {
                active_experts.insert(e);
            }
        }

        auto& seq_blocks = seq_block_ids_[seq_id];
        if (seq_blocks.empty() && !group.shared_block_ids.empty()) {
            seq_blocks = group.shared_block_ids;
        }

        const int required_context = group_context + tokens_needed;
        const int required_blocks = SequenceBlockTable::BlocksNeeded(required_context);
        const int blocks_to_allocate = std::max(0, required_blocks - static_cast<int>(seq_blocks.size()));

        std::vector<int> new_blocks;
        if (blocks_to_allocate > 0) {
            if (block_manager_->GetFreeBlockCount() < blocks_to_allocate) {
                LOG_DEBUG("Scheduler: Not enough blocks for seq ", seq_id, " (needed ", blocks_to_allocate, ", free ",
                          block_manager_->GetFreeBlockCount(), ")");
                still_waiting.push_back(group);
                continue;
            }

            new_blocks = block_manager_->Allocate(blocks_to_allocate);
            if ((int)new_blocks.size() != blocks_to_allocate) {
                LOG_DEBUG("Scheduler: Allocate failed for seq ", seq_id);
                still_waiting.push_back(group);
                continue;
            }
            seq_blocks.insert(seq_blocks.end(), new_blocks.begin(), new_blocks.end());
        }

        LOG_DEBUG("Scheduler: Scheduled seq ", seq_id, " (prefill)");
        output.prefill_seq_ids.push_back(seq_id);
        output.prefill_chunk_info.push_back({seq_id, tokens_needed});
        output.num_prefill_tokens += tokens_needed;
        output.total_tokens += tokens_needed;
        if (target_context_len < 0) {
            target_context_len = group_context;
        }
        output.batch_context_len = target_context_len;
        if (!new_blocks.empty()) {
            output.new_block_allocations.push_back({seq_id, new_blocks});
        }

        // Expose prefix cache hit information to worker
        if (group.shared_prefix_len > 0 && !group.shared_block_ids.empty()) {
            PrefixCacheInfo cache_info;
            cache_info.seq_id = seq_id;
            cache_info.cached_tokens = group.shared_prefix_len;
            cache_info.cached_block_ids = group.shared_block_ids;
            output.prefix_cache_hits.push_back(cache_info);
            LOG_DEBUG("Scheduler: Prefix cache hit for seq ", seq_id, " (", group.shared_prefix_len, " tokens cached)");
            // Emit prefix cache hit once; avoid duplicate append in worker.
            group.shared_prefix_len = 0;
            group.shared_block_ids.clear();
        }

        seq_generated_tokens_[seq_id] = 0;
        seq_priority_[seq_id] = group.priority;
        seq_arrival_[seq_id] = group.arrival_time;

        tokens_budget -= tokens_needed;
        if (prefill_token_cap >= 0) {
            prefill_token_cap = std::max(0, prefill_token_cap - tokens_needed);
        }

        if (remaining > tokens_needed) {
            group.num_tokens_to_process = remaining - tokens_needed;
            seq_status_[seq_id] = SequenceStatus::WAITING;
            still_waiting.push_back(group);
        } else {
            group.num_tokens_to_process = 0;
            running_seqs_.insert(seq_id);
            seq_status_[seq_id] = SequenceStatus::RUNNING;
        }
    }

    waiting_queue_ = std::move(still_waiting);
}

void Scheduler::ScheduleSwapped(SchedulerOutput& output) {
    if (swapped_seqs_.empty()) return;

    int available_slots = config_.max_num_seqs - static_cast<int>(running_seqs_.size());
    if (available_slots <= 0) return;

    std::vector<int> swapped_ids(swapped_seqs_.begin(), swapped_seqs_.end());
    std::sort(swapped_ids.begin(), swapped_ids.end(), [this](int a, int b) {
        int pri_a = GetSequencePriority(a);
        int pri_b = GetSequencePriority(b);
        if (pri_a != pri_b) return pri_a < pri_b;
        return GetSequenceArrival(a) < GetSequenceArrival(b);
    });

    for (int seq_id : swapped_ids) {
        if (available_slots <= 0) break;

        int prompt_len = 0;
        auto prompt_it = seq_prompt_len_.find(seq_id);
        if (prompt_it != seq_prompt_len_.end()) {
            prompt_len = prompt_it->second;
        }
        int generated = GetSequenceProgress(seq_id);
        int total_tokens = prompt_len + generated;
        if (total_tokens <= 0) total_tokens = 1;

        int blocks_needed = SequenceBlockTable::BlocksNeeded(total_tokens);
        if (block_manager_->GetFreeBlockCount() < blocks_needed) {
            continue;
        }

        std::vector<int> new_blocks = block_manager_->Allocate(blocks_needed);
        if (new_blocks.empty()) {
            continue;
        }

        output.swap_in_seq_ids.push_back(seq_id);
        output.new_block_allocations.push_back({seq_id, new_blocks});
        seq_block_ids_[seq_id] = new_blocks;

        swapped_seqs_.erase(seq_id);
        running_seqs_.insert(seq_id);
        seq_status_[seq_id] = SequenceStatus::RUNNING;
        available_slots--;
    }
}

// ============================================================================
// Factory Function for Default Scheduler Configuration
// ============================================================================

SchedulerConfig CreateInteractiveConfig() {
    SchedulerConfig config;
    config.max_num_seqs = 4;
    config.max_num_batched_tokens = 512;
    config.max_model_len = 4096;
    config.enable_priority = false;
    config.enable_chunked_prefill = false;
    return config;
}

SchedulerConfig CreateThroughputConfig() {
    SchedulerConfig config;
    config.max_num_seqs = 256;
    config.max_num_batched_tokens = 4096;
    config.max_model_len = 8192;
    config.enable_priority = true;
    config.priority_preempt_threshold = 5;
    config.enable_chunked_prefill = true;
    config.max_prefill_tokens = 1024;
    config.enable_mixed_prefill_decode = true;
    config.max_mixed_prefill_tokens = 128;
    return config;
}

SchedulerConfig CreateMemoryEfficientConfig() {
    SchedulerConfig config;
    config.max_num_seqs = 8;
    config.max_num_batched_tokens = 256;
    config.max_model_len = 2048;
    config.enable_priority = false;
    config.watermark_high = 0.7f;
    config.watermark_low = 0.5f;
    config.enable_chunked_prefill = true;
    config.max_prefill_tokens = 128;
    return config;
}

// ============================================================================
// Helper Functions
// ============================================================================

bool ScheduleStep(Scheduler& scheduler, SchedulerOutput& output) {
    output = scheduler.Schedule();
    return !output.IsEmpty();
}

void PrintSchedulerDebugInfo(const Scheduler& scheduler) {
    auto stats = scheduler.GetStats();
    std::cout << "[Scheduler] Waiting: " << stats.waiting_count << " Running: " << stats.running_count
              << " Swapped: " << stats.swapped_count << " Mem: " << stats.memory_usage << std::endl;
}

const char* SequenceStatusToString(SequenceStatus status) {
    switch (status) {
    case SequenceStatus::WAITING: return "WAITING";
    case SequenceStatus::RUNNING: return "RUNNING";
    case SequenceStatus::SWAPPED: return "SWAPPED";
    case SequenceStatus::FINISHED: return "FINISHED";
    case SequenceStatus::CANCELLED: return "CANCELLED";
    default: return "UNKNOWN";
    }
}

void PrintSchedulerOutput(const SchedulerOutput& output) {
    std::cout << "[SchedulerOutput] Total: " << output.total_tokens << " Prefill: " << output.num_prefill_tokens
              << " Decode: " << output.num_decode_tokens << std::endl;
}

}  // namespace densecore
