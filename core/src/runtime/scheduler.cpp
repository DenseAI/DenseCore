/**
 * @file scheduler.cpp
 * @brief Scheduler implementation for continuous batching inference
 *
 * Implements the Scheduler class defined in densecore/runtime/scheduler.h.
 * Responsible for managing request queues, scheduling prefill/decode batches,
 * and handling preemption based on memory pressure.
 */

#include "runtime/scheduler_internal.h"

#include "densecore/utils/logging.h"

namespace densecore {

// ============================================================================
// Scheduler Implementation
// ============================================================================

Scheduler::Scheduler(BlockManager* block_manager, const SchedulerConfig& config)
    : block_manager_(block_manager), config_(config) {
    config_.enable_chunked_prefill = scheduler_internal::ParseEnvBool(
        std::getenv("DENSECORE_SCHED_ENABLE_CHUNKED_PREFILL"), config_.enable_chunked_prefill);
    config_.max_prefill_tokens =
        std::max(1, scheduler_internal::ParseEnvInt(std::getenv("DENSECORE_SCHED_MAX_PREFILL_TOKENS"),
                                                    config_.max_prefill_tokens));
    decode_homogeneous_batch_n_past_ =
        scheduler_internal::ParseEnvBool(std::getenv("DENSECORE_SCHED_DECODE_HOMOGENEOUS_N_PAST"),
                                         /*default_value=*/false);
    config_.enable_mixed_prefill_decode = scheduler_internal::ParseEnvBool(
        std::getenv("DENSECORE_SCHED_ENABLE_MIXED_PREFILL_DECODE"), config_.enable_mixed_prefill_decode);
    config_.max_mixed_prefill_tokens =
        std::max(1, scheduler_internal::ParseEnvInt(std::getenv("DENSECORE_SCHED_MAX_MIXED_PREFILL_TOKENS"),
                                                    config_.max_mixed_prefill_tokens));
    config_.enable_moe_clustering = scheduler_internal::ParseEnvBool(
        std::getenv("DENSECORE_SCHED_ENABLE_MOE_CLUSTERING"), config_.enable_moe_clustering);
    config_.max_active_experts =
        std::max(1, scheduler_internal::ParseEnvInt(std::getenv("DENSECORE_SCHED_MAX_ACTIVE_EXPERTS"),
                                                    config_.max_active_experts));
    config_.moe_batch_strictness =
        std::clamp(scheduler_internal::ParseEnvFloat(std::getenv("DENSECORE_SCHED_MOE_BATCH_STRICTNESS"),
                                                     config_.moe_batch_strictness),
                   0.0f, 1.0f);
    config_.max_prefill_tokens = std::min(config_.max_prefill_tokens, std::max(1, config_.max_num_batched_tokens));
}

int Scheduler::AddRequest(int request_id, int prompt_len, int max_output_len, int priority,
                          const std::vector<int>* prefix_tokens, bool allow_chunked_prefill,
                          bool require_hybrid_ssm_prefix_snapshot) {
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

    // Check for reusable full-block prefix hits.
    // Prefix reuse always leaves at least one token to execute so prompt-end
    // logits are still computed by the normal prefill path.
    if (prefix_tokens && !prefix_tokens->empty()) {
        auto match = block_manager_->FindLongestCachedPrefixWithVerification(prefix_tokens->data(), prompt_len,
                                                                             require_hybrid_ssm_prefix_snapshot);
        if (match.cached_tokens > 0 && !match.cached_block_ids.empty()) {
            group.shared_prefix_len = match.cached_tokens;
            group.shared_block_ids = std::move(match.cached_block_ids);
            group.num_tokens_to_process = std::max(0, prompt_len - group.shared_prefix_len);
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

            if (mixed_prefill_admitted && waiting_queue_.empty()) {
                consecutive_decode_batches_ = 0;
            } else {
                // Preserve the decode streak while any waiting prompt is still
                // blocked so the isolated-prefill fairness fallback can fire.
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

void Scheduler::UpdateProgressBatch(const std::vector<std::pair<int, int>>& progress_updates) {
    if (progress_updates.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& update : progress_updates) {
        if (update.first < 0 || update.second <= 0) {
            continue;
        }
        seq_generated_tokens_[update.first] += update.second;
        seq_context_len_[update.first] += update.second;
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
    config.enable_moe_clustering = true;
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
