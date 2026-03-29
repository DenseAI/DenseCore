#include "scheduler_internal.h"

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

}  // namespace

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

float ParseEnvFloat(const char* value, float default_value) {
    if (!value || *value == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0') {
        return default_value;
    }
    return parsed;
}

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

bool ShouldDeferForMoEBudget(int request_key, const std::vector<int>& experts,
                             const std::unordered_set<int>& active_experts, const SchedulerConfig& config) {
    if (!config.enable_moe_clustering || experts.empty()) {
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
    const float rand_val = static_cast<float>(request_key % 100) / 100.0f;
    return rand_val < strictness;
}

}  // namespace densecore::scheduler_internal
