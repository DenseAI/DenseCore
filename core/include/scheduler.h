/**
 * @file scheduler.h
 * @brief Advanced scheduler for continuous batching
 *
 * Implements vLLM-style iteration-level scheduling:
 * - Separate prefill and decode batches
 * - Preemption support for priority requests
 * - Memory-aware scheduling
 */

#ifndef DENSECORE_SCHEDULER_H
#define DENSECORE_SCHEDULER_H

#include <algorithm>
#include <chrono>
#include <climits>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache.h"

namespace densecore {

/**
 * Sequence state in the scheduler
 */
enum class SequenceStatus {
    WAITING,    // In queue, not started
    RUNNING,    // Currently being processed
    SWAPPED,    // Preempted and swapped to CPU
    FINISHED,   // Completed generation
    CANCELLED,  // User cancelled
};

/**
 * Sequence group - represents one request that may have multiple sequences
 * (e.g., beam search, parallel sampling)
 */
struct SequenceGroup {
    int request_id;
    std::vector<int> sequence_ids;
    int prompt_len;
    int max_output_len;
    int priority;  // Lower = higher priority
    std::chrono::steady_clock::time_point arrival_time;

    // Prefix sharing
    int shared_prefix_len = 0;
    std::vector<int> shared_block_ids;

    // Computed at each iteration
    int num_running_seqs = 0;
    int num_tokens_to_process = 0;

    // Progress tracking (for smart preemption)
    int generated_tokens = 0;  // Tokens generated so far

    // MoE optimization: predicted or last-used experts for this request
    std::vector<int> predicted_experts;

    bool IsPrefill() const { return num_tokens_to_process > 1; }
};

/**
 * Prefix cache hit information for a sequence
 */
struct PrefixCacheInfo {
    int seq_id;                         // Sequence ID
    int cached_tokens;                  // Number of tokens already in cache
    std::vector<int> cached_block_ids;  // Block IDs containing cached KV
};

/**
 * Prefill chunk metadata for one sequence in this iteration.
 */
struct PrefillChunkInfo {
    int seq_id = -1;
    int chunk_tokens = 0;
};

/**
 * Scheduler output for one iteration
 */
struct SchedulerOutput {
    // Sequences to process in this iteration
    std::vector<int> prefill_seq_ids;    // Full prefill
    std::vector<int> decode_seq_ids;     // Single token decode
    std::vector<int> preempted_seq_ids;  // Preempted this round
    std::vector<int> swap_in_seq_ids;    // Swapped-in this round

    // Block allocations
    std::vector<std::pair<int, std::vector<int>>> new_block_allocations;  // (seq_id, block_ids)
    std::vector<std::pair<int, std::vector<int>>> freed_blocks;           // (seq_id, block_ids)

    // Prefix cache hits (seq_id -> cached token count and blocks)
    std::vector<PrefixCacheInfo> prefix_cache_hits;
    // Per-sequence chunk size for prefill scheduling
    std::vector<PrefillChunkInfo> prefill_chunk_info;

    // Batching info
    int total_tokens = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    // Dominant context length (n_past) for the selected batch.
    // -1 means unset/empty.
    int batch_context_len = -1;

    bool IsEmpty() const { return prefill_seq_ids.empty() && decode_seq_ids.empty(); }
};

/**
 * Scheduler configuration
 */
struct SchedulerConfig {
    int max_num_seqs = 256;             // Max concurrent sequences
    int max_num_batched_tokens = 2048;  // Max tokens per iteration
    int max_model_len = 4096;           // Max sequence length

    // Priority scheduling
    bool enable_priority = true;
    int priority_preempt_threshold = 10;  // Priority diff to trigger preemption

    // Memory management
    float watermark_high = 0.9f;  // Start preemption
    float watermark_low = 0.8f;   // Stop preemption

    // Chunked prefill
    bool enable_chunked_prefill = true;
    int max_prefill_tokens = 512;  // Max prefill tokens per iteration

    // Batch-shape safety: enforce single n_past bucket per iteration.
    bool enforce_homogeneous_batch_n_past = true;
    // If true, do not mix prefill/decode in the same scheduler output.
    bool isolate_prefill_decode = true;
    // Fairness guard: allow prefill admission after N decode-only iterations.
    int max_consecutive_decode_batches = 8;

    // ==========================================================================
    // MoE Optimization (Expert Locality-Aware Batching)
    // ==========================================================================
    bool enable_moe_clustering = false;  // Group requests by expert affinity
    int max_active_experts = 8;          // Max experts per batch (L3 cache fit)
    float moe_batch_strictness = 0.5f;   // 0.0=FIFO, 1.0=strict expert locality
};

/**
 * Advanced scheduler for continuous batching
 */
class Scheduler {
public:
    explicit Scheduler(BlockManager* block_manager, const SchedulerConfig& config = SchedulerConfig());

    /**
     * Add a new request to the scheduler
     */
    int AddRequest(int request_id, int prompt_len, int max_output_len, int priority = 100,
                   const std::vector<int>* prefix_tokens = nullptr, bool allow_chunked_prefill = true);

    /**
     * Remove a request (cancel or complete)
     */
    void RemoveRequest(int seq_id, bool finished = true);

    /**
     * Check if scheduler has any pending or active requests
     */
    bool HasRequests() const;

    /**
     * Schedule next iteration
     * Main scheduling algorithm
     */
    SchedulerOutput Schedule();

    /**
     * Fork a sequence for beam search / parallel sampling
     * Uses Copy-on-Write: child shares parent's blocks with incremented ref_count
     */
    int ForkSequence(int parent_seq_id);

    /**
     * Get sequence status
     */
    SequenceStatus GetStatus(int seq_id) const;

    /**
     * Ensure a block is writable (Copy-on-Write trigger)
     * Call before modifying any block that might be shared.
     * Returns the (possibly new) block ID that is safe to write.
     */
    int EnsureBlockWritable(int seq_id, int block_index, const std::function<void(int src, int dst)>& copy_callback);

    /**
     * Set predicted experts for a sequence (MoE Expert Prediction)
     * Call after routing to record which experts were selected,
     * enabling expert-coverage-aware batching on next iteration.
     */
    void SetPredictedExperts(int seq_id, const std::vector<int>& experts);

    /**
     * Update sequence progress (call after each token generation)
     */
    void UpdateProgress(int seq_id, int tokens_generated = 1);

    /**
     * Get scheduler stats
     */
    struct Stats {
        int waiting_count;
        int running_count;
        int swapped_count;
        float memory_usage;
    };

    Stats GetStats() const;

private:
    float GetMemoryUsage() const;

    void PreemptSequences(SchedulerOutput& output);

    /**
     * Centralized sequence cleanup path shared by RemoveRequest and internal
     * cancellation branches in ScheduleWaiting.
     */
    void CleanupSequenceState(int seq_id, bool erase_from_waiting_queue, bool free_block_manager_blocks);

    /**
     * Free sequence resources (blocks and metadata)
     */
    void FreeSequence(int seq_id);

    /**
     * Select the best victim for preemption using smart criteria
     */
    int SelectPreemptionVictim();

    int GetSequencePriority(int seq_id) const;
    int GetSequenceProgress(int seq_id) const;
    int GetSequenceContextLen(int seq_id) const;
    std::chrono::steady_clock::time_point GetSequenceArrival(int seq_id) const;

    void ScheduleRunning(SchedulerOutput& output);
    void ScheduleWaiting(SchedulerOutput& output);
    void ScheduleSwapped(SchedulerOutput& output);

    BlockManager* block_manager_;
    SchedulerConfig config_;
    mutable std::mutex mutex_;

    std::deque<SequenceGroup> waiting_queue_;
    std::unordered_set<int> running_seqs_;
    std::unordered_set<int> swapped_seqs_;
    std::unordered_map<int, SequenceStatus> seq_status_;

    int next_seq_id_ = 1;

    // Progress tracking for smart preemption
    std::unordered_map<int, int> seq_generated_tokens_;                           // seq_id -> tokens generated
    std::unordered_map<int, int> seq_context_len_;                                // seq_id -> current context length
    std::unordered_map<int, int> seq_priority_;                                   // seq_id -> priority
    std::unordered_map<int, std::chrono::steady_clock::time_point> seq_arrival_;  // seq_id -> arrival time
    std::unordered_map<int, int> seq_prompt_len_;                                 // seq_id -> prompt length
    std::unordered_map<int, bool> seq_allow_chunked_prefill_;                     // seq_id -> chunking policy

    // Phase-isolation fairness bookkeeping
    int consecutive_decode_batches_ = 0;
    // Decode n_past bucketing policy.
    // Defaults to false so decode can batch heterogeneous context lengths.
    bool decode_homogeneous_batch_n_past_ = false;

    // ==========================================================================
    // Copy-on-Write Support (Patent Compliance)
    // ==========================================================================
    // Block table per sequence: maps seq_id -> list of block IDs owned by that seq
    // When ForkSequence is called, these blocks get their ref_count incremented
    // via BlockManager::Fork(). When the sequence writes to a shared block,
    // CopyOnWrite() must be called first.
    // ==========================================================================
    std::unordered_map<int, std::vector<int>> seq_block_ids_;  // seq_id -> block_ids

    // ==========================================================================
    // MoE Expert Prediction Support (Patent Claim 4)
    // ==========================================================================
    // Tracks which experts each sequence used, enabling expert-coverage-aware
    // batching. Updated after MoE routing via SetPredictedExperts().
    // ==========================================================================
    std::unordered_map<int, std::vector<int>> seq_predicted_experts_;  // seq_id -> expert_ids
};

// Configuration factories
SchedulerConfig CreateInteractiveConfig();
SchedulerConfig CreateThroughputConfig();
SchedulerConfig CreateMemoryEfficientConfig();

// Helper functions (exposed for testing/utils)
bool ScheduleStep(Scheduler& scheduler, SchedulerOutput& output);
void PrintSchedulerDebugInfo(const Scheduler& scheduler);
const char* SequenceStatusToString(SequenceStatus status);
void PrintSchedulerOutput(const SchedulerOutput& output);

}  // namespace densecore

#endif  // DENSECORE_SCHEDULER_H
