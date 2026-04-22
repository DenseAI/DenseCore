/**
 * @file profiler.h
 * @brief Thread-safe MoE Expert Profiler for usage statistics
 *
 * Provides lock-free expert hit tracking with EMA (Exponential Moving Average)
 * for identifying "hot" experts. Designed for minimal overhead on inference hot path.
 *
 * Usage:
 *   ExpertProfiler profiler(n_experts);
 *   profiler.RecordHit(expert_id);  // Call during MoE routing
 *   auto hot = profiler.GetHotExperts(8);  // Get top-8 hot experts
 */

#ifndef DENSECORE_MOE_PROFILER_H
#define DENSECORE_MOE_PROFILER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace densecore {
namespace moe {

/**
 * @brief Statistics for a single expert
 *
 * All fields are atomic for lock-free concurrent updates.
 * Memory layout is cache-line friendly (64-byte padding optional).
 */
struct ExpertStats {
    std::atomic<uint64_t> hit_count{0};      ///< Total cumulative hits
    std::atomic<float> load_ema{0.0f};       ///< Exponential moving average (α=0.1)
    std::atomic<uint64_t> last_accessed{0};  ///< Monotonic timestamp (nanoseconds)

    ExpertStats() = default;

    // Non-copyable but movable
    ExpertStats(const ExpertStats& other)
        : hit_count(other.hit_count.load(std::memory_order_relaxed)),
          load_ema(other.load_ema.load(std::memory_order_relaxed)),
          last_accessed(other.last_accessed.load(std::memory_order_relaxed)) {}

    ExpertStats& operator=(const ExpertStats& other) {
        if (this != &other) {
            hit_count.store(other.hit_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            load_ema.store(other.load_ema.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_accessed.store(other.last_accessed.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    void Reset() {
        hit_count.store(0, std::memory_order_relaxed);
        load_ema.store(0.0f, std::memory_order_relaxed);
        last_accessed.store(0, std::memory_order_relaxed);
    }
};

/**
 * @brief Thread-safe MoE Expert Profiler
 *
 * Tracks expert usage statistics with lock-free recording and EMA-based
 * "hotness" tracking. Designed for integration with MoE routing kernels.
 *
 * Thread Safety:
 * - RecordHit(): Lock-free, safe to call from any thread
 * - GetHotExperts(): Acquires internal mutex for consistent snapshot
 * - Reset(): Acquires internal mutex
 *
 * Performance:
 * - RecordHit() uses atomic fetch_add and compare-exchange
 * - Typical overhead: ~10-20ns per call on modern CPUs
 */
class ExpertProfiler {
public:
    /// Default EMA smoothing factor (α=0.1 means ~10 samples to reach 63% of new value)
    static constexpr float kDefaultEmaAlpha = 0.1f;

    /// Default decay interval for EMA updates (1.0 = update on every hit)
    static constexpr float kDefaultDecayRate = 1.0f;

    /**
     * @brief Construct a profiler for n_experts
     * @param n_experts Number of experts to track (must be > 0)
     * @param ema_alpha EMA smoothing factor (0.0-1.0, higher = more weight to recent)
     */
    explicit ExpertProfiler(int n_experts, float ema_alpha = kDefaultEmaAlpha);

    /**
     * @brief Record a hit for an expert (lock-free)
     *
     * Updates hit_count, load_ema, and last_accessed atomically.
     * Safe to call concurrently from multiple threads.
     *
     * @param expert_id Expert index (0 to n_experts-1)
     */
    void RecordHit(int expert_id);

    /**
     * @brief Record multiple hits in batch
     *
     * Slightly more efficient than calling RecordHit() in a loop.
     *
     * @param expert_ids Array of expert IDs
     * @param count Number of IDs in array
     */
    void RecordHitBatch(const int* expert_ids, int count);

    /**
     * @brief Get the top-k hottest experts by EMA
     *
     * Returns expert IDs sorted by load_ema in descending order.
     * Uses partial_sort for O(n + k log k) complexity.
     *
     * @param top_k Number of hot experts to return
     * @return Vector of expert IDs (may be < top_k if n_experts < top_k)
     */
    std::vector<int> GetHotExperts(int top_k) const;

    /**
     * @brief Get all expert statistics (for debugging/monitoring)
     * @return Copy of all ExpertStats
     */
    std::vector<ExpertStats> GetAllStats() const;

    /**
     * @brief Get statistics for a specific expert
     * @param expert_id Expert index
     * @return Copy of ExpertStats for that expert
     */
    ExpertStats GetExpertStats(int expert_id) const;

    /**
     * @brief Get total hit count across all experts (O(1))
     */
    uint64_t GetTotalHits() const;

    /**
     * @brief Get hit count for a specific expert
     * @param expert_id Expert index
     * @return Hit count for that expert
     */
    uint64_t GetHitCount(int expert_id) const;

    /**
     * @brief Get current EMA load for a specific expert
     * @param expert_id Expert index
     * @return EMA load value
     */
    float GetEmaLoad(int expert_id) const;

    /**
     * @brief Reset all statistics to zero
     */
    void Reset();

    /**
     * @brief Get number of experts being tracked
     */
    int GetNumExperts() const { return n_experts_; }

    /**
     * @brief Apply time-based decay to EMA values
     *
     * Call periodically to decay stale experts. Each call multiplies
     * all EMA values by (1 - kEmaAlpha).
     */
    void ApplyDecay();

    // =========================================================================
    // Sticky Routing: Expert to NUMA Node Mapping
    // =========================================================================

    /**
     * @brief Set the NUMA node where an expert's weights reside
     *
     * Called after RebalanceExperts() migrates weights to a specific node.
     * Thread-safe (lock-free atomic store).
     *
     * @param expert_id Expert index (0 to n_experts-1)
     * @param numa_node NUMA node ID (-1 means unassigned/any)
     */
    void SetExpertNumaNode(int expert_id, int numa_node);

    /**
     * @brief Get the NUMA node where an expert's weights reside
     *
     * Used by DispatchExpertFFN to route computation to the correct node.
     *
     * @param expert_id Expert index
     * @return NUMA node ID, or -1 if unassigned
     */
    int GetExpertNumaNode(int expert_id) const;

private:
    int n_experts_;
    float ema_alpha_;  ///< Configurable EMA smoothing factor
    std::vector<ExpertStats> stats_;
    std::atomic<uint64_t> total_hits_{0};                  ///< O(1) total hit counter
    mutable std::mutex mutex_;                             ///< For GetHotExperts consistency
    std::unique_ptr<std::atomic<int>[]> expert_numa_map_;  ///< NUMA node per expert (-1 = unassigned)

    /// Get current timestamp in nanoseconds
    static uint64_t NowNanos();

    /// Update EMA with new hit (lock-free)
    void UpdateEma(ExpertStats& stat);
};

/**
 * @brief Global singleton profiler (optional convenience)
 *
 * Use for simple integration. For multi-model scenarios,
 * create separate ExpertProfiler instances.
 */
ExpertProfiler& GetGlobalExpertProfiler();

/**
 * @brief Initialize global profiler with n_experts
 *
 * Must be called before GetGlobalExpertProfiler() if using global instance.
 * Thread-safe (uses call_once).
 */
void InitGlobalExpertProfiler(int n_experts);

}  // namespace moe
}  // namespace densecore

#endif  // DENSECORE_MOE_PROFILER_H
