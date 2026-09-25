#include "llm/attention/diagnostics.h"
#include "densecore/runtime/inference.h"
#include "runtime/runtime_env.h"
#include <atomic>
#include <cstring>
namespace densecore::llm::graph::detail {
bool IsDebugInferenceStatsEnabled();
}
using densecore::llm::graph::detail::IsDebugInferenceStatsEnabled;

bool IsDebugPagedAttentionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDebugPagedAttentionEagerReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ShouldRunPagedAttentionReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_LAYER", -1);
    static const int target_token =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool ShouldRunPagedAttentionEagerReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionEagerReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_LAYER", -1);
    static const int target_token =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool IsAttentionDecodeProfilingEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_PROFILE_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDecodeAttentionPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_LOG_DECODE_ATTENTION_PATH");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        return IsAttentionDecodeProfilingEnabled() || IsDebugInferenceStatsEnabled();
    }();
    return enabled;
}
