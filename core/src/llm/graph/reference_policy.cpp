#include "densecore/exceptions.h"
#include "densecore/runtime/dtype_utils.h"
#include "densecore/runtime/optimization_bridge.h"
#include "densecore/simd/simd_ops.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/support_internal.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/work_context.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace densecore::llm::graph::detail {
bool IsDebugInferenceStatsEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_INFERENCE_STATS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Dispatch path labels for instrumentation


bool IsDebugSSMQkvReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SSM_QKV_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDebugSSMProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SSM_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAttentionProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugFinalProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugHiddenSnapshotEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugFfnProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDebugAttentionCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAddRmsNormReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ShouldRunAddRmsNormReferenceProbe(int layer_idx) {
    if (!IsDebugAddRmsNormReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }
    int configured_layer = -1;
    if (const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_LAYER")) {
        if (env[0] != '\0') {
            configured_layer = std::atoi(env);
        }
    }
    return configured_layer < 0 || configured_layer == layer_idx;
}

bool IsDebugAttentionPostReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_ATTN_POST_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugKvRoundTripEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_KV_ROUNDTRIP");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ShouldRunKvRoundTripProbe(int layer, bool is_k) {
    if (!IsDebugKvRoundTripEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_KV_ROUNDTRIP_LAYER", -1);
    static const int target_kind =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_KV_ROUNDTRIP_KIND", -1);  // -1 both, 0 V, 1 K
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_KV_ROUNDTRIP_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_kind >= 0 && target_kind != (is_k ? 1 : 0)) {
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

bool ShouldRunAttentionProjectionReferenceProbe(int layer) {
    if (!IsDebugAttentionProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_MAX_CALLS", 8)};

    if (target_layer >= 0 && layer != target_layer) {
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

bool ShouldRunFinalProjectionReferenceProbe() {
    if (!IsDebugFinalProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE_MAX_CALLS", 1)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool ShouldRunHiddenSnapshotProbe(int layer, const char* stage) {
    if (!IsDebugHiddenSnapshotEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_LAYER", -2);
    if (target_layer >= -1 && layer != target_layer) {
        return false;
    }

    static const std::string target_stage = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_STAGE");
        return (env && env[0] != '\0') ? std::string(env) : std::string();
    }();
    if (!target_stage.empty() && stage && target_stage != stage) {
        return false;
    }

    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_MAX_CALLS", 4)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool ShouldRunFfnProjectionReferenceProbe(int layer) {
    if (!IsDebugFfnProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_MAX_CALLS", 8)};
    if (target_layer >= 0 && layer != target_layer) {
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

static bool IsDebugRmsNormReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ShouldRunRmsNormReferenceProbe(int layer) {
    if (!IsDebugRmsNormReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_RMSNORM_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_RMSNORM_REFERENCE_MAX_CALLS", 8)};
    if (target_layer >= 0 && layer != target_layer) {
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
}  // namespace densecore::llm::graph::detail

using namespace densecore::llm::graph::detail;

struct ggml_tensor* MaybeAttachGemma4SharedKVProbe(struct ggml_context* ctx_c, struct ggml_tensor* tensor,
                                                   const char* action, const char* kind, int layer_idx,
                                                   int source_layer) {
    const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
    const bool enabled = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    if (!enabled || !ctx_c || !tensor) {
        return tensor;
    }
    auto* ud = AllocateGemma4KVSummaryUserData(ctx_c);
    if (!ud) {
        return tensor;
    }
    ud->layer_idx = layer_idx;
    ud->source_layer = source_layer;
    ud->action = action;
    ud->kind = kind;
    return ggml_map_custom1(ctx_c, tensor, cb_gemma4_kv_summary_probe, 1, ud);
}

bool ShouldRunSharedScalarGateReferenceProbe(int layer_idx) {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (!enabled) return false;
    if (IsCurrentGraphBuildNoAlloc()) return false;
    static const int target_layer = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    static std::atomic<int> remaining_budget{[]() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_MAX_CALLS");
        if (!env || env[0] == '\0') return 4;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env || parsed <= 0) ? 4 : static_cast<int>(parsed);
    }()};
    if (target_layer >= 0 && layer_idx != target_layer) return false;
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

#ifdef DENSECORE_TEST_BUILD
namespace {
thread_local AttentionCaptureState attention_capture;
}

AttentionCaptureState ExchangeAttentionCaptureForTest(AttentionCaptureState next) {
    const auto previous = attention_capture;
    attention_capture = next;
    return previous;
}

void* AttentionCaptureUserDataForTest() {
    return attention_capture.output;
}

bool ShouldCaptureAttentionForTest(int layer) {
    return attention_capture.layer == layer;
}

void cb_test_capture_attention_tensor(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)nth;
    if (!dst || !src || !dst->data || !src->data) return;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    auto* output = static_cast<std::vector<float>*>(userdata);
    if (ith != 0 || !output || src->type != GGML_TYPE_F32) return;
    const int elems = static_cast<int>(ggml_nelements(src));
    output->resize(static_cast<size_t>(elems));
    std::memcpy(output->data(), src->data, static_cast<size_t>(elems) * sizeof(float));
}
#endif
