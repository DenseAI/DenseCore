/**
 * @file profiler.cpp
 * @brief MoE Expert Profiler implementation
 *
 * Implements lock-free expert hit tracking with EMA-based hotness scoring.
 */

#include "moe/profiler.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace densecore {
namespace moe {

// =============================================================================
// ExpertProfiler Implementation
// =============================================================================

ExpertProfiler::ExpertProfiler(int n_experts, float ema_alpha) : n_experts_(n_experts), ema_alpha_(ema_alpha) {
    if (n_experts <= 0) {
        throw std::invalid_argument("ExpertProfiler: n_experts must be > 0");
    }
    stats_.resize(n_experts);
    // Initialize NUMA mapping to -1 (unassigned)
    // Using unique_ptr<atomic[]> avoids std::vector resize issues with non-movable atomics
    expert_numa_map_ = std::make_unique<std::atomic<int>[]>(n_experts);
    for (int i = 0; i < n_experts; i++) {
        expert_numa_map_[i].store(-1, std::memory_order_relaxed);
    }
}

uint64_t ExpertProfiler::NowNanos() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

void ExpertProfiler::UpdateEma(ExpertStats& stat) {
    // Lock-free EMA update using compare-exchange loop
    float old_ema = stat.load_ema.load(std::memory_order_relaxed);
    float new_ema;
    do {
        // EMA formula: new_ema = α * 1.0 + (1 - α) * old_ema
        // Where 1.0 represents the current "hit" signal
        new_ema = ema_alpha_ * 1.0f + (1.0f - ema_alpha_) * old_ema;
    } while (
        !stat.load_ema.compare_exchange_weak(old_ema, new_ema, std::memory_order_relaxed, std::memory_order_relaxed));
}

void ExpertProfiler::RecordHit(int expert_id) {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return;  // Silently ignore out-of-bounds (hot path, no exceptions)
    }

    ExpertStats& stat = stats_[expert_id];

    // Atomic increment of hit count
    stat.hit_count.fetch_add(1, std::memory_order_relaxed);
    total_hits_.fetch_add(1, std::memory_order_relaxed);  // O(1) total hits

    // Update EMA (lock-free)
    UpdateEma(stat);

    // Update timestamp
    stat.last_accessed.store(NowNanos(), std::memory_order_relaxed);
}

void ExpertProfiler::RecordHitBatch(const int* expert_ids, int count) {
    uint64_t now = NowNanos();

    for (int i = 0; i < count; i++) {
        int expert_id = expert_ids[i];
        if (expert_id < 0 || expert_id >= n_experts_) {
            continue;
        }

        ExpertStats& stat = stats_[expert_id];
        stat.hit_count.fetch_add(1, std::memory_order_relaxed);
        total_hits_.fetch_add(1, std::memory_order_relaxed);  // O(1) total hits
        UpdateEma(stat);
        stat.last_accessed.store(now, std::memory_order_relaxed);
    }
}

std::vector<int> ExpertProfiler::GetHotExperts(int top_k) const {
    if (top_k <= 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Build (ema, expert_id) pairs for sorting
    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_experts_);

    for (int i = 0; i < n_experts_; i++) {
        float ema = stats_[i].load_ema.load(std::memory_order_relaxed);
        scored.emplace_back(ema, i);
    }

    // Partial sort to get top-k (O(n + k log k))
    int k = std::min(top_k, n_experts_);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract expert IDs
    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < k; i++) {
        result.push_back(scored[i].second);
    }

    return result;
}

std::vector<ExpertStats> ExpertProfiler::GetAllStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;  // Returns copy (ExpertStats has copy ctor)
}

ExpertStats ExpertProfiler::GetExpertStats(int expert_id) const {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return ExpertStats{};
    }
    return stats_[expert_id];
}

uint64_t ExpertProfiler::GetTotalHits() const {
    return total_hits_.load(std::memory_order_relaxed);
}

uint64_t ExpertProfiler::GetHitCount(int expert_id) const {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return 0;
    }
    return stats_[expert_id].hit_count.load(std::memory_order_relaxed);
}

float ExpertProfiler::GetEmaLoad(int expert_id) const {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return 0.0f;
    }
    return stats_[expert_id].load_ema.load(std::memory_order_relaxed);
}

void ExpertProfiler::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& stat : stats_) {
        stat.Reset();
    }
    total_hits_.store(0, std::memory_order_relaxed);
}

void ExpertProfiler::ApplyDecay() {
    const float decay_factor = 1.0f - ema_alpha_;
    for (int i = 0; i < n_experts_; i++) {
        ExpertStats& stat = stats_[i];

        // Decay EMA using compare-exchange loop
        float old_ema = stat.load_ema.load(std::memory_order_relaxed);
        float new_ema;
        do {
            new_ema = decay_factor * old_ema;
        } while (!stat.load_ema.compare_exchange_weak(old_ema, new_ema, std::memory_order_relaxed,
                                                      std::memory_order_relaxed));
    }
}

// =============================================================================
// Sticky Routing: Expert to NUMA Node Mapping
// =============================================================================

void ExpertProfiler::SetExpertNumaNode(int expert_id, int numa_node) {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return;  // Silently ignore invalid IDs (hot path safety)
    }
    expert_numa_map_[expert_id].store(numa_node, std::memory_order_relaxed);
}

int ExpertProfiler::GetExpertNumaNode(int expert_id) const {
    if (expert_id < 0 || expert_id >= n_experts_) {
        return -1;  // Invalid -> unassigned
    }
    return expert_numa_map_[expert_id].load(std::memory_order_relaxed);
}

// =============================================================================
// Global Singleton
// =============================================================================

namespace {
std::unique_ptr<ExpertProfiler> g_global_profiler;
std::once_flag g_profiler_init_flag;
int g_profiler_n_experts = 0;
}  // namespace

void InitGlobalExpertProfiler(int n_experts) {
    g_profiler_n_experts = n_experts;
    std::call_once(g_profiler_init_flag,
                   []() { g_global_profiler = std::make_unique<ExpertProfiler>(g_profiler_n_experts); });
}

ExpertProfiler& GetGlobalExpertProfiler() {
    if (!g_global_profiler) {
        throw std::runtime_error("Global ExpertProfiler not initialized. Call InitGlobalExpertProfiler() first.");
    }
    return *g_global_profiler;
}

}  // namespace moe
}  // namespace densecore
