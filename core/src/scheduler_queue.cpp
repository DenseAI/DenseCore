#include "scheduler_internal.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_set>

#include "densecore/utils/logging.h"

namespace densecore {

int Scheduler::EnsureBlockWritable(int seq_id, int block_index,
                                   const std::function<void(int src, int dst)>& copy_callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_block_ids_.find(seq_id);
    if (it == seq_block_ids_.end()) {
        return -1;
    }

    auto& blocks = it->second;
    if (block_index < 0 || block_index >= static_cast<int>(blocks.size())) {
        return -1;
    }

    int old_block_id = blocks[block_index];
    if (block_manager_->IsShared(old_block_id)) {
        int new_block_id = block_manager_->CopyOnWrite(old_block_id, copy_callback);
        if (new_block_id >= 0 && new_block_id != old_block_id) {
            blocks[block_index] = new_block_id;
            return new_block_id;
        }
    }

    return old_block_id;
}

void Scheduler::SetPredictedExperts(int seq_id, const std::vector<int>& experts) {
    std::lock_guard<std::mutex> lock(mutex_);
    seq_predicted_experts_[seq_id] = experts;

    for (auto& group : waiting_queue_) {
        for (int sid : group.sequence_ids) {
            if (sid == seq_id) {
                group.predicted_experts = experts;
                return;
            }
        }
    }
}

void Scheduler::ScheduleRunning(SchedulerOutput& output) {
    int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;
    std::unordered_set<int> active_experts;

    std::vector<int> running_ids(running_seqs_.begin(), running_seqs_.end());
    std::sort(running_ids.begin(), running_ids.end(), [this](int a, int b) {
        const int pri_a = GetSequencePriority(a);
        const int pri_b = GetSequencePriority(b);
        if (pri_a != pri_b) {
            return pri_a < pri_b;
        }
        return GetSequenceArrival(a) < GetSequenceArrival(b);
    });

    int target_context_len = -1;
    std::vector<uint8_t> scheduled(running_ids.size(), 0);
    while (true) {
        int best_index = -1;
        int best_priority = std::numeric_limits<int>::max();
        int best_overlap = -1;
        int best_new_experts = std::numeric_limits<int>::max();
        int best_expert_count = std::numeric_limits<int>::max();
        auto best_arrival = std::chrono::steady_clock::time_point::max();

        for (size_t idx = 0; idx < running_ids.size(); ++idx) {
            if (scheduled[idx]) {
                continue;
            }

            const int seq_id = running_ids[idx];
            const int seq_context = GetSequenceContextLen(seq_id);
            if (decode_homogeneous_batch_n_past_ && target_context_len >= 0 && seq_context != target_context_len) {
                continue;
            }

            auto experts_it = seq_predicted_experts_.find(seq_id);
            const std::vector<int>* experts =
                (experts_it != seq_predicted_experts_.end()) ? &experts_it->second : nullptr;
            if (experts && scheduler_internal::ShouldDeferForMoEBudget(seq_id, *experts, active_experts, config_)) {
                continue;
            }

            const int priority = GetSequencePriority(seq_id);
            const int overlap = experts ? scheduler_internal::CountExpertOverlap(*experts, active_experts) : 0;
            const int new_experts = experts ? scheduler_internal::CountNewExperts(*experts, active_experts) : 0;
            const int expert_count = experts ? static_cast<int>(experts->size()) : 0;
            const auto arrival = GetSequenceArrival(seq_id);

            bool better = false;
            if (best_index < 0 || priority < best_priority) {
                better = true;
            } else if (priority == best_priority) {
                if (!active_experts.empty() && overlap != best_overlap) {
                    better = overlap > best_overlap;
                } else if (!active_experts.empty() && new_experts != best_new_experts) {
                    better = new_experts < best_new_experts;
                } else if (!active_experts.empty() && expert_count != best_expert_count) {
                    better = expert_count < best_expert_count;
                } else if (arrival < best_arrival) {
                    better = true;
                }
            }

            if (better) {
                best_index = static_cast<int>(idx);
                best_priority = priority;
                best_overlap = overlap;
                best_new_experts = new_experts;
                best_expert_count = expert_count;
                best_arrival = arrival;
            }
        }

        if (best_index < 0 || scheduler_internal::ScheduledSeqCount(output) >= static_cast<size_t>(config_.max_num_seqs) ||
            tokens_budget <= 0) {
            break;
        }

        const int seq_id = running_ids[static_cast<size_t>(best_index)];
        scheduled[static_cast<size_t>(best_index)] = 1;

        const int seq_context = GetSequenceContextLen(seq_id);
        auto it = seq_predicted_experts_.find(seq_id);
        if (it != seq_predicted_experts_.end()) {
            for (int expert_id : it->second) {
                active_experts.insert(expert_id);
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
    }
}

void Scheduler::ScheduleWaiting(SchedulerOutput& output, int prefill_token_cap) {
    int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;

    std::vector<SequenceGroup> sorted_queue(waiting_queue_.begin(), waiting_queue_.end());
    std::sort(sorted_queue.begin(), sorted_queue.end(), [](const SequenceGroup& a, const SequenceGroup& b) {
        if (a.priority != b.priority) {
            return a.priority < b.priority;
        }
        return a.arrival_time < b.arrival_time;
    });

    if (config_.enable_moe_clustering && sorted_queue.size() > 1) {
        std::vector<SequenceGroup> pending = sorted_queue;
        std::vector<SequenceGroup> clustered;
        clustered.reserve(sorted_queue.size());
        std::unordered_set<int> preview_active_experts;

        while (!pending.empty()) {
            size_t best_index = 0;
            int best_priority = pending[0].priority;
            int best_overlap = -1;
            int best_new_experts = std::numeric_limits<int>::max();
            int best_expert_count = std::numeric_limits<int>::max();
            auto best_arrival = pending[0].arrival_time;

            for (size_t idx = 0; idx < pending.size(); ++idx) {
                const SequenceGroup& group = pending[idx];
                const int overlap = scheduler_internal::CountExpertOverlap(group.predicted_experts, preview_active_experts);
                const int new_experts = scheduler_internal::CountNewExperts(group.predicted_experts, preview_active_experts);
                const int expert_count = static_cast<int>(group.predicted_experts.size());

                bool better = false;
                if (group.priority < best_priority) {
                    better = true;
                } else if (group.priority == best_priority) {
                    if (!preview_active_experts.empty() && overlap != best_overlap) {
                        better = overlap > best_overlap;
                    } else if (!preview_active_experts.empty() && new_experts != best_new_experts) {
                        better = new_experts < best_new_experts;
                    } else if (!preview_active_experts.empty() && expert_count != best_expert_count) {
                        better = expert_count < best_expert_count;
                    } else if (group.arrival_time < best_arrival) {
                        better = true;
                    }
                }

                if (better) {
                    best_index = idx;
                    best_priority = group.priority;
                    best_overlap = overlap;
                    best_new_experts = new_experts;
                    best_expert_count = expert_count;
                    best_arrival = group.arrival_time;
                }
            }

            for (int expert_id : pending[best_index].predicted_experts) {
                preview_active_experts.insert(expert_id);
            }
            clustered.push_back(std::move(pending[best_index]));
            pending.erase(pending.begin() + static_cast<ptrdiff_t>(best_index));
        }

        sorted_queue = std::move(clustered);
    }

    std::deque<SequenceGroup> still_waiting;
    std::unordered_set<int> active_experts;
    int target_context_len = output.batch_context_len;

    for (auto& group : sorted_queue) {
        if (scheduler_internal::ScheduledSeqCount(output) >= static_cast<size_t>(config_.max_num_seqs)) {
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
        if (config_.enforce_homogeneous_batch_n_past && target_context_len >= 0 && group_context != target_context_len) {
            still_waiting.push_back(group);
            continue;
        }

        if (config_.enable_moe_clustering && !group.predicted_experts.empty()) {
            if (scheduler_internal::ShouldDeferForMoEBudget(group.request_id, group.predicted_experts, active_experts,
                                                            config_)) {
                still_waiting.push_back(group);
                continue;
            }

            for (int expert_id : group.predicted_experts) {
                active_experts.insert(expert_id);
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
            if (static_cast<int>(new_blocks.size()) != blocks_to_allocate) {
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

        if (group.shared_prefix_len > 0 && !group.shared_block_ids.empty()) {
            PrefixCacheInfo cache_info;
            cache_info.seq_id = seq_id;
            cache_info.cached_tokens = group.shared_prefix_len;
            cache_info.cached_block_ids = group.shared_block_ids;
            output.prefix_cache_hits.push_back(cache_info);
            LOG_DEBUG("Scheduler: Prefix cache hit for seq ", seq_id, " (", group.shared_prefix_len, " tokens cached)");
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
    if (swapped_seqs_.empty()) {
        return;
    }

    int available_slots = config_.max_num_seqs - static_cast<int>(running_seqs_.size());
    if (available_slots <= 0) {
        return;
    }

    std::vector<int> swapped_ids(swapped_seqs_.begin(), swapped_seqs_.end());
    std::sort(swapped_ids.begin(), swapped_ids.end(), [this](int a, int b) {
        const int pri_a = GetSequencePriority(a);
        const int pri_b = GetSequencePriority(b);
        if (pri_a != pri_b) {
            return pri_a < pri_b;
        }
        return GetSequenceArrival(a) < GetSequenceArrival(b);
    });

    for (int seq_id : swapped_ids) {
        if (available_slots <= 0) {
            break;
        }

        int prompt_len = 0;
        auto prompt_it = seq_prompt_len_.find(seq_id);
        if (prompt_it != seq_prompt_len_.end()) {
            prompt_len = prompt_it->second;
        }
        int generated = GetSequenceProgress(seq_id);
        int total_tokens = prompt_len + generated;
        if (total_tokens <= 0) {
            total_tokens = 1;
        }

        const int blocks_needed = SequenceBlockTable::BlocksNeeded(total_tokens);
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

}  // namespace densecore
