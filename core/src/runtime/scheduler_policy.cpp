#include "runtime/scheduler_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>

namespace densecore::scheduler_internal {

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

uint64_t MixRequestKey(uint64_t value) {
    // SplitMix64-derived mixer to avoid pathological low-ID bias such as
    // request_key=0 always mapping to the most strict branch.
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

std::size_t ScheduledSeqCount(const SchedulerOutput& output) {
    return output.prefill_seq_ids.size() + output.decode_seq_ids.size();
}

int CountExpertOverlap(const std::vector<int>& experts, const std::unordered_set<int>& active_experts) {
    int overlap = 0;
    for (int expert_id : experts) {
        if (active_experts.find(expert_id) != active_experts.end()) {
            ++overlap;
        }
    }
    return overlap;
}

int CountNewExperts(const std::vector<int>& experts, const std::unordered_set<int>& active_experts) {
    int new_experts = 0;
    for (int expert_id : experts) {
        if (active_experts.find(expert_id) == active_experts.end()) {
            ++new_experts;
        }
    }
    return new_experts;
}

float ComputeMoEDeferScore(int request_key) {
    const uint64_t mixed = MixRequestKey(static_cast<uint64_t>(static_cast<uint32_t>(request_key)));
    constexpr double kScale = 1.0 / static_cast<double>(UINT64_C(0xFFFFFFFFFFFFFFFF));
    return static_cast<float>(static_cast<double>(mixed) * kScale);
}

bool ShouldDeferForMoEBudget(int request_key, const std::vector<int>& experts,
                             const std::unordered_set<int>& active_experts, const SchedulerConfig& config) {
    if (!config.enable_moe_clustering || experts.empty()) {
        return false;
    }
    if (active_experts.empty()) {
        // Never starve the first runnable request in an iteration. Locality
        // optimization starts only after a seed request establishes the active
        // expert set for that batch.
        return false;
    }

    const int new_experts = CountNewExperts(experts, active_experts);
    if (static_cast<int>(active_experts.size()) + new_experts <= config.max_active_experts) {
        return false;
    }

    const float strictness = std::clamp(config.moe_batch_strictness, 0.0f, 1.0f);
    if (strictness <= 0.0f) {
        return false;
    }
    if (strictness >= 1.0f) {
        return true;
    }
    return ComputeMoEDeferScore(request_key) < strictness;
}

}  // namespace densecore::scheduler_internal
