#include "runtime/worker_internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_set>

#include "densecore/arm_runtime.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/exceptions.h"
#include "densecore/models/model_descriptor.h"
#include "ggml.h"
#include "models/model_inference_policy.h"
#include "runtime/runtime_env.h"

#ifndef DENSECORE_DEFAULT_PRECOMPUTED_ROPE
#define DENSECORE_DEFAULT_PRECOMPUTED_ROPE 1
#endif

#ifndef DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
#define DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM 0
#endif

#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

// Defined in inference.cpp (GGML custom paged decode callback).
void cb_paged_attention_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);

namespace {

int ParsePositiveEnvIntOrDefault(const char* name, int default_value) {
    return densecore::env::ParsePositiveEnvInt(name, default_value);
}

bool IsWideSimdLevel(densecore::simd::SimdLevel simd_level) {
    switch (simd_level) {
    case densecore::simd::SimdLevel::AMX:
    case densecore::simd::SimdLevel::AVX512:
    case densecore::simd::SimdLevel::SVE:
    case densecore::simd::SimdLevel::SVE2: return true;
    default: return false;
    }
}

int CapThreadsToAvailableCores(int physical_core_count, int base_threads) {
    int cap = physical_core_count > 0 ? physical_core_count : base_threads;
    if (cap <= 0) {
        cap = 1;
    }
    if (base_threads > 0) {
        cap = std::min(cap, base_threads);
    }
    return std::max(1, cap);
}

bool IsQwen36HybridSsmSingleRequest(const TransformerModel* model, int num_seqs) {
    if (num_seqs != 1 || !model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    return descriptor.variant == ModelVariant::QWEN36;
}

int ResolveQwen36WideSimdSingleDecodeFallbackThreads(int cap) {
    if (cap <= 8) {
        return cap;
    }
    if (cap <= 12) {
        return 8;
    }
    if (cap <= 16) {
        return 10;
    }
    return 12;
}

bool GraphContainsPagedDecodeCustomOp(const struct ggml_cgraph* graph) {
    if (!graph) return false;
    struct CustomOpParamsView {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(CustomOpParamsView) <= GGML_MAX_OP_PARAMS, "Custom op params view too large");
    const int n_nodes = ggml_graph_n_nodes(const_cast<struct ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(const_cast<struct ggml_cgraph*>(graph), i);
        if (!node || node->op != GGML_OP_CUSTOM) continue;
        CustomOpParamsView params{};
        std::memcpy(&params, node->op_params, sizeof(params));
        if (params.fun == cb_paged_attention_decode) {
            return true;
        }
    }
    return false;
}

bool CompiledWithArmSveForSummary() {
#if defined(__ARM_FEATURE_SVE)
    return true;
#else
    return false;
#endif
}

bool CompiledWithArmSve2ForSummary() {
#if defined(__ARM_FEATURE_SVE2)
    return true;
#else
    return false;
#endif
}

bool ResolvePagedDecodeHeadDims(const TransformerModel* model, int* n_head_out, int* n_head_kv_out, int* head_dim_q_out,
                                int* head_dim_kv_out) {
    if (!model || !n_head_out || !n_head_kv_out || !head_dim_q_out || !head_dim_kv_out) {
        return false;
    }
    if (!model->gemma4_layer_n_head_kv.empty()) {
        for (size_t i = 0; i < model->gemma4_layer_n_head_kv.size(); ++i) {
            const uint32_t layer_n_head_kv = model->gemma4_layer_n_head_kv[i];
            if (layer_n_head_kv > 0 && layer_n_head_kv != model->hparams.n_head_kv) {
                return false;
            }
        }
    }
    const int n_head = model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    if (n_head <= 0 || n_head_kv <= 0) {
        return false;
    }

    const int head_dim_q = model->hparams.n_embd / n_head;
    const int head_dim_kv =
        (model->hparams.n_embd_head_k > 0) ? model->hparams.n_embd_head_k : (model->hparams.n_embd / n_head);
    if (head_dim_q <= 0 || head_dim_kv <= 0) {
        return false;
    }

    *n_head_out = n_head;
    *n_head_kv_out = n_head_kv;
    *head_dim_q_out = head_dim_q;
    *head_dim_kv_out = head_dim_kv;
    return true;
}

size_t LongestPrefixSuffixMatch(const std::string& text, const std::string& pattern) {
    if (text.empty() || pattern.empty()) return 0;
    const size_t max_len = std::min(text.size(), pattern.size() - 1);
    for (size_t len = max_len; len > 0; --len) {
        if (text.compare(text.size() - len, len, pattern, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

size_t LongestTagCarry(const std::string& text, const std::string& open, const std::string& close) {
    return std::max(LongestPrefixSuffixMatch(text, open), LongestPrefixSuffixMatch(text, close));
}

}  // namespace

bool IsDebugGraphLoggingEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_GRAPH", false);
    return enabled;
}

bool IsVerboseTokenTraceEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_VERBOSE_TOKEN_TRACE", false);
    return enabled;
}

bool IsReasoningTagSuppressionEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_SUPPRESS_REASONING_TAGS", true);
    return enabled;
}

bool IsDirectCallbackEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DIRECT_CALLBACK", false);
    return enabled;
}

bool ShouldUseDirectResultCallbacks(bool benchmark_fast_path) {
    return IsDirectCallbackEnabled() || (benchmark_fast_path && IsBenchmarkDirectCallbackEnabled());
}

void EmitRequestResult(EngineState* state, Request* req, const std::string& token, int token_id, bool finished,
                       bool error, bool use_direct_callback) {
    if (!req || (!req->callback && !req->token_result_callback)) {
        return;
    }

    if (use_direct_callback) {
        if (req->callback) {
            req->callback(token.c_str(), finished ? 1 : 0, req->user_data);
        } else if (req->token_result_callback) {
            TokenResult result;
            result.token_id = token_id;
            result.text = token.c_str();
            result.is_finished = finished ? 1 : (error ? 1 : 0);
            req->token_result_callback(&result, req->user_data);
        }
        return;
    }

    if (!state) {
        return;
    }

    PushResultEvent(state, req->id, token, token_id, finished, error, req->callback, req->token_result_callback,
                    req->user_data);
}

bool IsSingleRequestFastPathEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_SINGLE_REQUEST_FAST_PATH", true);
    return enabled;
}

bool ShouldBypassSingleRequestFastPathForLongHybridSSM(const TransformerModel* model, const Request* req) {
    if (!model || !req || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    if (densecore::env::ParseNonZeroEnv("DENSECORE_DISABLE_SINGLE_REQ_FAST_PATH_FOR_LONG_HYBRID_SSM", false)) {
        return true;
    }
    return req->n_past > 0 || static_cast<int>(req->tokens.size()) > BLOCK_SIZE;
}

bool IsBenchmarkFastPathEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_BENCH_MODE");
        const bool on = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        if (on) {
            std::cerr << "[DenseCore] DENSECORE_BENCH_MODE enabled; benchmark fast-path bypasses normal scheduler "
                         "behavior and can mask serving-path stalls."
                      << std::endl;
        }
        return on;
    }();
    return enabled;
}

bool IsBenchmarkDirectCallbackEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_BENCH_DIRECT_CALLBACK", false);
    return enabled;
}

bool IsBenchmarkDecodeBatchFastPathEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_BENCH_DECODE_BATCH_FAST_PATH", false);
    return enabled;
}

int BenchmarkFastPathMaxBatch() {
    static const int max_batch = []() {
        return densecore::env::ParsePositiveEnvInt("DENSECORE_BENCH_FAST_PATH_MAX_BATCH", 8);
    }();
    return max_batch;
}

bool AllowDecodeThreadsOverBase() {
    static const bool enabled = []() {
        if (const char* env = std::getenv("DENSECORE_ALLOW_EXCEED_BASE_THREADS"); env && env[0] != '\0') {
            return densecore::env::ParseNonZeroEnv("DENSECORE_ALLOW_EXCEED_BASE_THREADS", false);
        }
        return densecore::env::ParseNonZeroEnv("DENSECORE_ALLOW_DECODE_THREADS_OVER_BASE", false);
    }();
    return enabled;
}

bool IsDecodeBatchPerfLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_LOG_DECODE_BATCH_TPS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDecodeProfileEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_PROFILE_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsBatchedDecodeCorrectnessCheckEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_CHECK_BATCHED_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsBatchedDecodeCorrectnessAbortEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_CHECK_BATCHED_DECODE_ABORT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

float BatchedDecodeCorrectnessTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_CHECK_BATCHED_DECODE_TOL");
        if (!env || env[0] == '\0') {
            return 1e-3f;
        }
        char* end = nullptr;
        const float v = std::strtof(env, &end);
        if (end == env || !std::isfinite(v) || v <= 0.0f) {
            return 1e-3f;
        }
        return v;
    }();
    return tol;
}

float DecodeGraphCacheRegressionTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_CHECK_DECODE_GRAPH_CACHE_TOL");
        if (!env || env[0] == '\0') {
            return BatchedDecodeCorrectnessTolerance();
        }
        char* end = nullptr;
        const float v = std::strtof(env, &end);
        if (end == env || !std::isfinite(v) || v <= 0.0f) {
            return BatchedDecodeCorrectnessTolerance();
        }
        return v;
    }();
    return tol;
}

size_t BatchedDecodeCorrectnessContextBytes() {
    static const size_t bytes = []() {
        const char* env = std::getenv("DENSECORE_CHECK_BATCHED_DECODE_CTX_MB");
        unsigned long long mb = 512ULL;
        if (env && env[0] != '\0') {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0ULL) {
                mb = parsed;
            }
        }
        return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
    }();
    return bytes;
}

ArmComputeAffinityPolicy ResolveArmComputeAffinityPolicy() {
    ArmComputeAffinityPolicy policy;
#if defined(__linux__) && defined(__aarch64__) && !defined(__ANDROID__)
    const densecore::arm_runtime::CoreClusters clusters = densecore::arm_runtime::DetectCoreClustersLinux();
    if (clusters.big_cores.empty() && clusters.little_cores.empty()) {
        return policy;
    }

    std::string mode = densecore::env::AsciiLowerCopy(std::getenv("DENSECORE_ARM_COMPUTE_CLUSTER"));
    if (mode.empty()) {
        mode = "auto";
    }

    auto use_all_cores = [&]() {
        policy.core_ids = clusters.big_cores;
        policy.core_ids.insert(policy.core_ids.end(), clusters.little_cores.begin(), clusters.little_cores.end());
        std::sort(policy.core_ids.begin(), policy.core_ids.end());
        policy.core_ids.erase(std::unique(policy.core_ids.begin(), policy.core_ids.end()), policy.core_ids.end());
        policy.label = "all";
    };

    if (mode == "all") {
        use_all_cores();
    } else if (mode == "little") {
        policy.core_ids = clusters.little_cores;
        policy.label = "little";
    } else if (mode == "big") {
        policy.core_ids = clusters.big_cores;
        policy.label = "big";
    } else {
        if (!clusters.big_cores.empty() && !clusters.little_cores.empty()) {
            policy.core_ids = clusters.big_cores;
            policy.label = "big";
        } else {
            use_all_cores();
        }
    }

    if (policy.core_ids.empty()) {
        use_all_cores();
    }
#endif
    return policy;
}

bool IsDecodeGraphCacheEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DECODE_GRAPH_CACHE", true);
    return enabled;
}

bool IsDecodeGraphCacheSafeForModel(const TransformerModel* model) {
    return model != nullptr;
}

bool DoesDecodeGraphCacheRequireRuntimeRebind(const TransformerModel* model) {
    return model && model->arch_flags.is_hybrid_ssm;
}

bool IsBatchedPagedDecodeEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_ENABLE_BATCHED_PAGED_DECODE", false);
    return enabled;
}

bool IsPagedDecodeGloballyDisabled() {
    static const bool disabled = []() {
        if (densecore::env::ParseNonZeroEnv("DENSECORE_FORCE_PAGED_DECODE", false)) {
            return false;
        }
        return densecore::env::ParseRuntimeToggleMode("DENSECORE_PAGED_ATTN_DECODE_MODE",
                                                      densecore::env::RuntimeToggleMode::Auto) ==
               densecore::env::RuntimeToggleMode::Off;
    }();
    return disabled;
}

bool IsForcePagedDecodeEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_FORCE_PAGED_DECODE", false);
    return enabled;
}

bool IsPagedDecodeModeForcedOn() {
    static const bool forced_on = []() {
        if (IsForcePagedDecodeEnabled()) {
            return true;
        }
        if (densecore::env::ParseRuntimeToggleMode("DENSECORE_PAGED_ATTN_DECODE_MODE",
                                                   densecore::env::RuntimeToggleMode::Auto) ==
            densecore::env::RuntimeToggleMode::On) {
            return true;
        }
        return densecore::env::ParseNonZeroEnv("DENSECORE_ENABLE_PAGED_ATTN_DECODE", false);
    }();
    return forced_on;
}

bool IsFlashAttentionForcedForCacheKey() {
    static const bool forced = densecore::env::ParseNonZeroEnv("DENSECORE_FORCE_FLASH_ATTN", false);
    return forced;
}

bool IsFlashAttentionDisabledForCacheKey() {
    static const bool disabled = []() {
        if (densecore::env::ParseRuntimeToggleMode("DENSECORE_FLASH_ATTN_MODE",
                                                   densecore::env::RuntimeToggleMode::Auto) ==
            densecore::env::RuntimeToggleMode::Off) {
            return true;
        }
        return densecore::env::ParseNonZeroEnv("DENSECORE_DISABLE_FLASH_ATTN", false);
    }();
    return disabled;
}

bool IsPrecomputedRoPEEnabledForCacheKey() {
    static const bool enabled = []() {
        const densecore::env::RuntimeToggleMode mode = densecore::env::ParseRuntimeToggleMode(
            "DENSECORE_ROPE_PRECOMPUTED_MODE", DENSECORE_DEFAULT_PRECOMPUTED_ROPE
                                                   ? densecore::env::RuntimeToggleMode::On
                                                   : densecore::env::RuntimeToggleMode::Off);
        return mode != densecore::env::RuntimeToggleMode::Off;
    }();
    return enabled;
}

bool IsFusedResidualRMSNormEnabledForCacheKey() {
    static const bool enabled = []() {
        const densecore::env::RuntimeToggleMode mode = densecore::env::ParseRuntimeToggleMode(
            "DENSECORE_FUSED_RESIDUAL_RMSNORM_MODE", DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
                                                         ? densecore::env::RuntimeToggleMode::On
                                                         : densecore::env::RuntimeToggleMode::Off);
        return mode != densecore::env::RuntimeToggleMode::Off;
    }();
    return enabled;
}

bool IsFusedQKVEnabledForCacheKey() {
    static const bool enabled = []() {
        const densecore::env::RuntimeToggleMode mode = densecore::env::ParseRuntimeToggleMode(
            "DENSECORE_FUSED_QKV_MODE", DENSECORE_DEFAULT_FUSED_QKV ? densecore::env::RuntimeToggleMode::On
                                                                    : densecore::env::RuntimeToggleMode::Off);
        return mode != densecore::env::RuntimeToggleMode::Off;
    }();
    return enabled;
}

bool IsDecodeGraphCacheDebugValidationEnabled() {
#ifndef NDEBUG
    static const bool enabled = []() {
        return densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_DECODE_GRAPH_CACHE", true);
    }();
    return enabled;
#else
    return false;
#endif
}

bool IsDecodeGraphCacheRuntimeScanEnabled() {
#ifndef NDEBUG
    static const bool enabled = []() {
        return densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_DECODE_GRAPH_CACHE_RUNTIME_SCAN", false);
    }();
    return enabled;
#else
    return false;
#endif
}

bool IsDecodeGraphCacheRegressionEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_CHECK_DECODE_GRAPH_CACHE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

int DecodeGraphCacheRegressionSteps() {
    static const int steps = ParsePositiveEnvIntOrDefault("DENSECORE_CHECK_DECODE_GRAPH_CACHE_STEPS", 200);
    return steps;
}

uint64_t BuildDecodeGraphFeatureFlags(const TransformerModel* model) {
    uint64_t feature_flags = 0;
    if (IsForcePagedDecodeEnabled()) feature_flags |= (1ull << 0);
    if (IsPagedDecodeModeForcedOn()) feature_flags |= (1ull << 1);
    if (IsBatchedPagedDecodeEnabled()) feature_flags |= (1ull << 2);
    if (IsPagedDecodeGloballyDisabled()) feature_flags |= (1ull << 3);
    if (IsFlashAttentionForcedForCacheKey()) feature_flags |= (1ull << 4);
    if (IsFlashAttentionDisabledForCacheKey()) feature_flags |= (1ull << 5);
    if (model && model->arch_flags.requires_q_norm) feature_flags |= (1ull << 6);
    if (model && model->arch_flags.requires_k_norm) feature_flags |= (1ull << 7);
    if (IsPrecomputedRoPEEnabledForCacheKey()) feature_flags |= (1ull << 8);
    if (IsFusedResidualRMSNormEnabledForCacheKey()) feature_flags |= (1ull << 9);
    if (IsFusedQKVEnabledForCacheKey()) feature_flags |= (1ull << 10);
    return feature_flags;
}

bool VerifyCachedDecodeGraphPagedOpOnBuild(const struct ggml_cgraph* graph, int batch_size) {
    if (batch_size <= 1) return true;
    if (GraphContainsPagedDecodeCustomOp(graph)) return true;
#ifndef NDEBUG
    if (IsDecodeGraphCacheDebugValidationEnabled()) {
        std::cerr << "[DecodeGraphCache][DEBUG] cached graph missing paged decode custom op on cache creation (bs="
                  << batch_size << ")" << std::endl;
        throw densecore::InvalidArgumentException(
            "Decode graph cache invariant failed: expected paged decode custom op for batched decode layout.");
    }
    if (IsDebugGraphLoggingEnabled()) {
        std::cerr << "[DecodeGraphCache][DEBUG] skip cache insert: missing paged decode custom op (bs=" << batch_size
                  << ")" << std::endl;
    }
#else
    if (IsDebugGraphLoggingEnabled()) {
        std::cerr << "[DecodeGraphCache][DEBUG] skip cache insert: missing paged decode custom op (bs=" << batch_size
                  << ")" << std::endl;
    }
#endif
    return false;
}

void DebugVerifyCachedDecodeGraphReuseState(const struct ggml_cgraph* graph, int batch_size, bool reused_graph,
                                            bool verified_paged_decode_op) {
#ifndef NDEBUG
    if (batch_size <= 1) return;
    if (!verified_paged_decode_op) {
        std::cerr << "[DecodeGraphCache][DEBUG] cached entry missing paged decode verification bit (bs=" << batch_size
                  << ", reused=" << (reused_graph ? 1 : 0) << ")" << std::endl;
        throw densecore::InvalidArgumentException(
            "Decode graph cache invariant failed: cached entry missing paged decode verification state.");
    }
    if (!IsDecodeGraphCacheRuntimeScanEnabled()) return;
    if (GraphContainsPagedDecodeCustomOp(graph)) return;
    std::cerr << "[DecodeGraphCache][DEBUG] runtime scan mismatch: cached graph missing paged decode custom op (bs="
              << batch_size << ", reused=" << (reused_graph ? 1 : 0) << ")" << std::endl;
    throw densecore::InvalidArgumentException(
        "Decode graph cache invariant failed: runtime scan expected paged decode custom op.");
#else
    (void)graph;
    (void)batch_size;
    (void)reused_graph;
    (void)verified_paged_decode_op;
#endif
}

int DecodeGraphCacheMaxBatch() {
    static const int max_batch = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_MAX_BATCH", 4);
    return max_batch;
}

int DecodeGraphCacheLruSize() {
    static const int lru = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_LRU_SIZE", 8);
    return lru;
}

size_t DecodeGraphCacheCtxBytes() {
    static const size_t bytes = []() {
        const int mb = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_CTX_MB", 96);
        return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
    }();
    return bytes;
}

bool IsDebugDecodeThreadsEnabled() {
    static const bool enabled = []() {
        return densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_DECODE_THREADS", false);
    }();
    return enabled;
}

bool IsDecodeRuntimeStatsLoggingEnabled() {
    static const bool enabled = []() {
        if (densecore::env::ParseNonZeroEnv("DENSECORE_LOG_DECODE_RUNTIME_STATS", false)) {
            return true;
        }
        return IsDecodeProfileEnabled();
    }();
    return enabled;
}

int DecodeRuntimeStatsInterval() {
    static const int interval = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_STATS_INTERVAL", 64);
    return interval;
}

int DecodeThreadsBatchOverride(int batch_size) {
    switch (batch_size) {
    case 1: return ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_THREADS_BATCH1", 0);
    case 2: return ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_THREADS_BATCH2", 0);
    case 3: return ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_THREADS_BATCH3", 0);
    case 4: return ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_THREADS_BATCH4", 0);
    default: return 0;
    }
}

bool UseLegacyDecodeThreadPolicy() {
    static const bool legacy = []() {
        const std::string mode = densecore::env::AsciiLowerCopy(std::getenv("DENSECORE_DECODE_THREAD_POLICY"));
        return mode == "legacy" || mode == "old";
    }();
    return legacy;
}

bool UseLegacyDecodeGraphCachePolicy() {
    static const bool legacy = []() {
        const std::string mode = densecore::env::AsciiLowerCopy(std::getenv("DENSECORE_DECODE_GRAPH_CACHE_POLICY"));
        return mode == "legacy" || mode == "old";
    }();
    return legacy;
}

bool IsQwen36SingleDecodeCacheCandidate(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    return descriptor.variant == ModelVariant::QWEN36;
}

int ResolveLegacyDecodeThreads(int num_seqs, int physical_core_count, int base_threads) {
    int available_threads = physical_core_count > 0 ? physical_core_count : base_threads;
    if (available_threads <= 0) {
        available_threads = 1;
    }

    int active_threads = 1;
    if (available_threads >= 16) {
        active_threads = std::max(4, available_threads / 2);
    } else if (available_threads >= 8) {
        active_threads = std::max(2, available_threads - 2);
    } else {
        active_threads = std::max(1, available_threads);
    }

    if (num_seqs >= 4) {
        const int decode_base = active_threads;
        const float scale = std::min(1.0f, static_cast<float>(num_seqs) / 8.0f);
        active_threads = decode_base + static_cast<int>(scale * static_cast<float>(available_threads - decode_base));
    }

    return std::max(1, active_threads);
}

PrefillThreadPolicySelection ResolvePrefillThreadPolicySelection(const TransformerModel* model, int num_seqs,
                                                                 int prompt_token_count, int physical_core_count,
                                                                 int base_threads,
                                                                 densecore::simd::SimdLevel simd_level) {
    PrefillThreadPolicySelection selection;
    selection.threads = CapThreadsToAvailableCores(physical_core_count, base_threads);
    selection.label = "prefill_base";

    if (IsQwen36HybridSsmSingleRequest(model, num_seqs) && IsWideSimdLevel(simd_level)) {
        if ((simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2) &&
            physical_core_count >= 16) {
            selection.label = "prefill_qwen36_single_c4a_full_core";
            return selection;
        }
        if (prompt_token_count > 0 && prompt_token_count < 64) {
            selection.threads = std::min(selection.threads, 8);
            selection.label = "prefill_qwen36_single_short_prompt";
        } else if (prompt_token_count > 0 && prompt_token_count < 128) {
            selection.threads = std::min(selection.threads, 12);
            selection.label = "prefill_qwen36_single_medium_prompt";
        } else {
            selection.label = "prefill_qwen36_single_long_prompt";
        }
    }
    return selection;
}

int ResolveAutoDecodeThreadsForBatchWithSimd(int num_seqs, int physical_core_count, int base_threads,
                                             densecore::simd::SimdLevel simd_level) {
    int cap = CapThreadsToAvailableCores(physical_core_count, base_threads);

    const int min_threads = std::min(cap, std::max(1, num_seqs));
    if (cap <= 4) {
        return cap;
    }

    int threads_per_seq = 4;
    switch (simd_level) {
    case densecore::simd::SimdLevel::AMX:
    case densecore::simd::SimdLevel::AVX512:
    case densecore::simd::SimdLevel::SVE:
    case densecore::simd::SimdLevel::SVE2: threads_per_seq = 8; break;
    case densecore::simd::SimdLevel::NEON: threads_per_seq = 6; break;
    default: break;
    }

    const int primary_batch = std::min(std::max(1, num_seqs), 8);
    int target = threads_per_seq * primary_batch;
    if (num_seqs > 8) {
        const int spill_threads_per_seq = std::max(1, threads_per_seq / 2);
        target += (num_seqs - 8) * spill_threads_per_seq;
    }

    return std::max(min_threads, std::min(cap, target));
}

DecodeThreadPolicySelection ResolveDecodeThreadPolicySelection(const TransformerModel* model, int num_seqs,
                                                               int physical_core_count, int base_threads,
                                                               densecore::simd::SimdLevel simd_level) {
    DecodeThreadPolicySelection selection;
    selection.threads =
        ResolveAutoDecodeThreadsForBatchWithSimd(num_seqs, physical_core_count, base_threads, simd_level);
    selection.label = "decode_batch_auto";

    if (!IsQwen36HybridSsmSingleRequest(model, num_seqs)) {
        return selection;
    }
    if (!IsWideSimdLevel(simd_level)) {
        return selection;
    }

    const int cap = CapThreadsToAvailableCores(physical_core_count, base_threads);
    const int env_override = densecore::env::ParsePositiveEnvInt("DENSECORE_QWEN36_SINGLE_DECODE_THREADS", 0);
    if (env_override > 0) {
        selection.threads = std::max(1, std::min(cap, env_override));
        selection.label =
            (model->hparams.n_experts > 0) ? "decode_qwen36_a3b_env_override" : "decode_qwen36_single_env_override";
        return selection;
    }

    if (model->hparams.n_experts > 0) {
        const bool arm_c4a_wide_simd =
            densecore::simd::IsArmFamily(simd_level) &&
            (simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2);
        if (arm_c4a_wide_simd && physical_core_count >= 16 && cap >= 16) {
            selection.threads = 16;
            selection.label = "decode_qwen36_a3b_c4a_moe_16";
            return selection;
        }
        if (arm_c4a_wide_simd && cap >= 12) {
            selection.threads = 12;
            selection.label = "decode_qwen36_a3b_arm_safe_cap";
            return selection;
        }
        selection.threads = std::max(1, std::min(cap, 8));
        selection.label = "decode_qwen36_a3b_safe_cap";
        return selection;
    }

    selection.threads = ResolveQwen36WideSimdSingleDecodeFallbackThreads(cap);
    selection.label = "decode_qwen36_dense27_c4_sweet_spot";
    return selection;
}

int ResolveAutoDecodeThreadsForBatch(int num_seqs, int physical_core_count, int base_threads) {
    return ResolveAutoDecodeThreadsForBatchWithSimd(num_seqs, physical_core_count, base_threads,
                                                    densecore::simd::DetectSimdLevel());
}

bool IsStablePagedDecodeTopologyForCache(const TransformerModel* model, const PagedKVCache* cache,
                                         const BatchSpec& batch) {
    if (!model || !cache || model->arch_flags.is_glm_dsa || !densecore::models::SupportsPagedDecodeAttention(model)) {
        return false;
    }
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    if (!IsDecodeOnlyBatchLayout(batch, n_tokens_in_batch)) {
        return false;
    }

    int n_head = 0;
    int n_head_kv = 0;
    int head_dim_q = 0;
    int head_dim_kv = 0;
    if (!ResolvePagedDecodeHeadDims(model, &n_head, &n_head_kv, &head_dim_q, &head_dim_kv)) {
        return false;
    }
    if (!IsPagedDecodeCandidate(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv)) {
        return false;
    }

    if (batch.num_seqs > 1) {
        return true;
    }
    return IsPagedDecodeModeAlwaysOn() || IsQwen36SingleDecodeCacheCandidate(model);
}

DecodeWorkerStats& GetDecodeWorkerStats() {
    static DecodeWorkerStats stats;
    return stats;
}

void MaybeLogDecodeRuntimeStats() {
    if (!IsDecodeRuntimeStatsLoggingEnabled()) {
        return;
    }

    DecodeWorkerStats& worker_stats = GetDecodeWorkerStats();
    const uint64_t decode_batches = worker_stats.decode_batches.load(std::memory_order_relaxed);
    const uint64_t interval = static_cast<uint64_t>(std::max(1, DecodeRuntimeStatsInterval()));
    static std::atomic<uint64_t> next_report{interval};
    uint64_t expected = next_report.load(std::memory_order_relaxed);
    while (decode_batches >= expected) {
        if (next_report.compare_exchange_weak(expected, expected + interval, std::memory_order_relaxed)) {
            const DecodeRuntimeStatsSnapshot runtime = GetDecodeRuntimeStatsSnapshot();
            const uint64_t graph_attempts = worker_stats.graph_cache_attempts.load(std::memory_order_relaxed);
            const uint64_t graph_hits = worker_stats.graph_cache_hits.load(std::memory_order_relaxed);
            const uint64_t graph_builds = worker_stats.graph_cache_builds.load(std::memory_order_relaxed);
            const uint64_t shared_quant_total = runtime.shared_quant_total;
            const double paged_hit_rate =
                runtime.path_total > 0 ? (100.0 * static_cast<double>(runtime.path_paged) / runtime.path_total) : 0.0;
            const double graph_hit_rate =
                graph_attempts > 0 ? (100.0 * static_cast<double>(graph_hits) / graph_attempts) : 0.0;
            const double shared_quant_hit_rate =
                shared_quant_total > 0 ? (100.0 * static_cast<double>(runtime.shared_quant_reused) / shared_quant_total)
                                       : 0.0;
            const densecore::CpuBackend::MoERuntimeStatsSnapshot moe_stats =
                densecore::GetCpuBackend().GetMoERuntimeStatsSnapshot();
            const KVRuntimeStatsSnapshot kv_stats = GetKVRuntimeStatsSnapshot();
            const double moe_avg_unique_experts =
                moe_stats.batches > 0 ? static_cast<double>(moe_stats.total_active_experts) / moe_stats.batches : 0.0;
            const double moe_local_hot_ratio =
                moe_stats.total_active_experts > 0
                    ? (100.0 * static_cast<double>(moe_stats.total_local_hot_experts) / moe_stats.total_active_experts)
                    : 0.0;
            const double moe_step_reuse =
                moe_stats.total_reuse_union > 0
                    ? (100.0 * static_cast<double>(moe_stats.total_reuse_intersection) / moe_stats.total_reuse_union)
                    : 0.0;
            const double moe_concentration =
                moe_stats.total_assignments > 0
                    ? (100.0 * static_cast<double>(moe_stats.total_max_expert_batch) / moe_stats.total_assignments)
                    : 0.0;
            const double moe_avg_cached_experts =
                moe_stats.batches > 0 ? static_cast<double>(moe_stats.total_cached_experts) / moe_stats.batches : 0.0;
            const double moe_avg_dequant_experts =
                moe_stats.batches > 0 ? static_cast<double>(moe_stats.total_dequantized_experts) / moe_stats.batches
                                      : 0.0;

            std::cerr << "[DecodeRuntimeStats] batches=" << decode_batches << " paged_hit_rate=" << paged_hit_rate
                      << "% (" << runtime.path_paged << "/" << runtime.path_total << ")"
                      << " graph_cache_hit_rate=" << graph_hit_rate << "% (" << graph_hits << "/" << graph_attempts
                      << ", builds=" << graph_builds << ")" << " shared_quant_hit_rate=" << shared_quant_hit_rate
                      << "% (" << runtime.shared_quant_reused << "/" << shared_quant_total
                      << ", tls=" << runtime.shared_quant_tls << ")"
                      << " decode_threads[b1=" << worker_stats.last_threads_by_batch[1].load(std::memory_order_relaxed)
                      << ",b2=" << worker_stats.last_threads_by_batch[2].load(std::memory_order_relaxed)
                      << ",b3=" << worker_stats.last_threads_by_batch[3].load(std::memory_order_relaxed)
                      << ",b4=" << worker_stats.last_threads_by_batch[4].load(std::memory_order_relaxed) << "]"
                      << " moe[avg_unique=" << moe_avg_unique_experts << ",local_hot=" << moe_local_hot_ratio
                      << "%,reuse=" << moe_step_reuse << "%,concentration=" << moe_concentration
                      << "%,avg_cached=" << moe_avg_cached_experts << ",avg_dequant=" << moe_avg_dequant_experts;
            if (moe_stats.total_prefetch_calls > 0 || moe_stats.total_dequantized_experts > 0) {
                std::cerr << ",prefetch_calls=" << moe_stats.total_prefetch_calls << ",prefetch_mb="
                          << (static_cast<double>(moe_stats.total_prefetch_bytes) / (1024.0 * 1024.0))
                          << ",dequant_experts=" << moe_stats.total_dequantized_experts << ",dequant_mb="
                          << (static_cast<double>(moe_stats.total_dequantized_bytes) / (1024.0 * 1024.0));
            }
            std::cerr << "]" << " kv_bulk[reads=" << kv_stats.bulk_read_calls << "/" << kv_stats.bulk_read_slots
                      << ",writes=" << kv_stats.bulk_write_calls << "/" << kv_stats.bulk_write_slots << "]";

            bool wrote_reason = false;
            for (std::size_t i = 1; i < runtime.paged_fallback_reasons.size(); ++i) {
                const uint64_t count = runtime.paged_fallback_reasons[i];
                if (count == 0) {
                    continue;
                }
                std::cerr << (wrote_reason ? "," : " paged_fallbacks=");
                std::cerr << GetDecodePagedFallbackReasonName(i) << ":" << count;
                wrote_reason = true;
            }
            if (!wrote_reason) {
                std::cerr << " paged_fallbacks=none";
            }

            bool wrote_dispatch = false;
            for (std::size_t weight_idx = 0; weight_idx < kHybridSSMDispatchWeightCount; ++weight_idx) {
                for (std::size_t path_idx = 0; path_idx < kHybridSSMDispatchPathCount; ++path_idx) {
                    const std::size_t flat_idx = weight_idx * kHybridSSMDispatchPathCount + path_idx;
                    const uint64_t count = runtime.hybrid_ssm_dispatch_counts[flat_idx];
                    if (count == 0) {
                        continue;
                    }
                    std::cerr << (wrote_dispatch ? "," : " hybrid_ssm_dispatch=");
                    std::cerr << GetHybridSSMDispatchWeightName(weight_idx) << ":"
                              << GetHybridSSMDispatchPathName(path_idx) << ":" << count;
                    wrote_dispatch = true;
                }
            }
            if (!wrote_dispatch) {
                std::cerr << " hybrid_ssm_dispatch=none";
            }

            std::cerr << " graph_cache_skips[disabled="
                      << worker_stats.graph_cache_skip_disabled.load(std::memory_order_relaxed)
                      << ",unstable=" << worker_stats.graph_cache_skip_unstable.load(std::memory_order_relaxed)
                      << ",lora=" << worker_stats.graph_cache_skip_lora.load(std::memory_order_relaxed)
                      << ",backend=" << worker_stats.graph_cache_skip_backend.load(std::memory_order_relaxed)
                      << ",uncacheable="
                      << worker_stats.graph_cache_rejected_uncacheable.load(std::memory_order_relaxed) << "]";

            bool wrote_variant_bucket = false;
            for (std::size_t variant_idx = 0; variant_idx < kDecodeGraphCacheTrackedVariants; ++variant_idx) {
                for (std::size_t batch_idx = 1; batch_idx < kDecodeGraphCacheTrackedBatches; ++batch_idx) {
                    const auto& bucket = worker_stats.graph_cache_by_variant_batch[variant_idx][batch_idx];
                    const uint64_t attempts = bucket.attempts.load(std::memory_order_relaxed);
                    const uint64_t hits = bucket.hits.load(std::memory_order_relaxed);
                    const uint64_t builds = bucket.builds.load(std::memory_order_relaxed);
                    const uint64_t rejected = bucket.rejected_uncacheable.load(std::memory_order_relaxed);
                    if (attempts == 0 && hits == 0 && builds == 0 && rejected == 0) {
                        continue;
                    }
                    std::cerr << (wrote_variant_bucket ? "," : " graph_cache_by_model_batch=");
                    std::cerr << densecore::models::ModelVariantName(static_cast<ModelVariant>(variant_idx)) << ":b"
                              << batch_idx << "{attempts=" << attempts << ",hits=" << hits << ",builds=" << builds
                              << ",rejected_uncacheable=" << rejected << "}";
                    wrote_variant_bucket = true;
                }
            }
            if (!wrote_variant_bucket) {
                std::cerr << " graph_cache_by_model_batch=none";
            }
            std::cerr << std::endl;
            return;
        }
    }
}

void EnsureRequestHybridSSMRuntimeState(TransformerModel* model, Request* req) {
    if (!model || !req || !model->arch_flags.is_hybrid_ssm) {
        return;
    }

    const int conv_channels = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
    const int head_dim = model->ssm_inner_size / model->ssm_time_step_rank;
    const size_t n_ssm_layers = model->ssm_layer_states.size();
    const auto reinit_all = [&]() {
        req->ssm_runtime_states.resize(n_ssm_layers);
        for (auto& state : req->ssm_runtime_states) {
            state.Init(conv_channels, model->ssm_conv_kernel, model->ssm_time_step_rank, head_dim,
                       model->ssm_state_size);
        }
    };
    const size_t expected_conv =
        TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(conv_channels, model->ssm_conv_kernel);
    const size_t expected_ssm = TransformerModel::SSMSequenceRuntimeState::ExpectedStateElements(
        model->ssm_time_step_rank, head_dim, model->ssm_state_size);

    if (req->ssm_runtime_states.size() != n_ssm_layers) {
        reinit_all();
        return;
    }

    for (auto& state : req->ssm_runtime_states) {
        if (state.conv_state.size() != expected_conv || state.ssm_state.size() != expected_ssm ||
            !state.MatchesShape(conv_channels, model->ssm_conv_kernel, model->ssm_time_step_rank, head_dim,
                                model->ssm_state_size)) {
            reinit_all();
            return;
        }
        state.Reset();
    }
}

void SuppressTaggedBlock(std::string* token_text, bool* in_block, std::string* pending, const char* open_tag,
                         const char* close_tag) {
    if (!token_text || !in_block || !pending || !open_tag || !close_tag) {
        return;
    }

    const std::string open(open_tag);
    const std::string close(close_tag);
    std::string current;
    current.reserve(pending->size() + token_text->size());
    current.append(*pending);
    current.append(*token_text);
    pending->clear();
    if (current.empty()) {
        token_text->clear();
        return;
    }

    std::string out;
    out.reserve(current.size());

    size_t pos = 0;
    while (pos < current.size()) {
        if (*in_block) {
            const size_t close_pos = current.find(close, pos);
            if (close_pos == std::string::npos) {
                const size_t carry = LongestPrefixSuffixMatch(current, close);
                if (carry > 0) {
                    pending->assign(current, current.size() - carry, carry);
                }
                token_text->clear();
                return;
            }
            pos = close_pos + close.size();
            *in_block = false;
            continue;
        }

        const size_t open_pos = current.find(open, pos);
        const size_t close_pos = current.find(close, pos);

        if (close_pos != std::string::npos && (open_pos == std::string::npos || close_pos < open_pos)) {
            out.append(current, pos, close_pos - pos);
            pos = close_pos + close.size();
            continue;
        }

        if (open_pos == std::string::npos) {
            out.append(current, pos, std::string::npos);
            break;
        }

        out.append(current, pos, open_pos - pos);
        const size_t block_close_pos = current.find(close, open_pos + open.size());
        if (block_close_pos == std::string::npos) {
            *in_block = true;
            break;
        }
        pos = block_close_pos + close.size();
    }

    *token_text = std::move(out);
    if (!*in_block) {
        const size_t carry = LongestTagCarry(*token_text, open, close);
        if (carry > 0) {
            pending->assign(*token_text, token_text->size() - carry, carry);
            token_text->erase(token_text->size() - carry);
        }
    }
}

bool IsStopTokenId(const TransformerModel* model, int token_id) {
    if (!model) return false;
    if (token_id == model->eos_token_id) {
        return true;
    }
    for (int32_t stop_id : model->stop_token_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }
    return false;
}

bool ShouldTerminateRepetitiveLoop(const TransformerModel* model, const Request* req) {
    if (!model || !req) return false;
    if (model->arch != ModelArch::QWEN3 && model->arch != ModelArch::QWEN35) {
        return false;
    }

    const auto descriptor = densecore::models::DescribeModel(model);
    const bool is_qwen36 = descriptor.variant == ModelVariant::QWEN36;
    if (is_qwen36 && req->generated_count < 32) {
        return false;
    }

    const auto& history = req->token_history;
    const size_t n = history.size();
    const size_t same_suffix_limit = is_qwen36 ? 16 : 8;
    const size_t alternating_window = is_qwen36 ? 24 : 12;
    if (n < std::min(same_suffix_limit, alternating_window)) {
        return false;
    }

    const int latest = history.back();
    size_t same_suffix = 1;
    while (same_suffix < n && history[n - 1 - same_suffix] == latest) {
        ++same_suffix;
    }
    if (same_suffix >= same_suffix_limit) {
        return true;
    }

    if (n >= alternating_window) {
        const int a = history[n - 1];
        const int b = history[n - 2];
        if (a != b) {
            bool alternating = true;
            for (size_t i = 0; i < alternating_window; ++i) {
                const int expected = (i % 2 == 0) ? a : b;
                if (history[n - 1 - i] != expected) {
                    alternating = false;
                    break;
                }
            }
            if (alternating) {
                return true;
            }
        }
    }

    return false;
}

size_t Utf8ValidPrefixLength(const std::string& s) {
    const size_t n = s.size();
    size_t i = 0;
    size_t valid = 0;

    while (i < n) {
        const uint8_t c0 = static_cast<uint8_t>(s[i]);
        if (c0 <= 0x7F) {
            ++i;
            valid = i;
            continue;
        }

        size_t need = 0;
        if ((c0 & 0xE0) == 0xC0) {
            if (c0 < 0xC2) {
                ++i;
                valid = i;
                continue;
            }
            need = 2;
        } else if ((c0 & 0xF0) == 0xE0) {
            need = 3;
        } else if ((c0 & 0xF8) == 0xF0) {
            if (c0 > 0xF4) {
                ++i;
                valid = i;
                continue;
            }
            need = 4;
        } else {
            ++i;
            valid = i;
            continue;
        }

        if (i + need > n) {
            break;
        }

        bool ok = true;
        for (size_t j = 1; j < need; ++j) {
            const uint8_t cx = static_cast<uint8_t>(s[i + j]);
            if ((cx & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            ++i;
            valid = i;
            continue;
        }

        if (need == 3) {
            const uint8_t c1 = static_cast<uint8_t>(s[i + 1]);
            if ((c0 == 0xE0 && c1 < 0xA0) || (c0 == 0xED && c1 >= 0xA0)) {
                ++i;
                valid = i;
                continue;
            }
        } else if (need == 4) {
            const uint8_t c1 = static_cast<uint8_t>(s[i + 1]);
            if ((c0 == 0xF0 && c1 < 0x90) || (c0 == 0xF4 && c1 >= 0x90)) {
                ++i;
                valid = i;
                continue;
            }
        }

        i += need;
        valid = i;
    }

    return valid;
}

int DecodeVisibleProgressTimeoutMs() {
    static const int timeout_ms =
        densecore::env::ParsePositiveEnvInt("DENSECORE_DECODE_VISIBLE_PROGRESS_TIMEOUT_MS", 5000);
    return timeout_ms;
}

int DecodeVisibleProgressMaxSilentSteps() {
    static const int max_steps =
        densecore::env::ParsePositiveEnvInt("DENSECORE_DECODE_VISIBLE_PROGRESS_MAX_STEPS", 256);
    return max_steps;
}

const char* DecodeFinishCauseName(DecodeFinishCause cause) {
    switch (cause) {
    case DecodeFinishCause::StopToken: return "stop_token";
    case DecodeFinishCause::StopSequence: return "stop_sequence";
    case DecodeFinishCause::MaxTokens: return "max_tokens";
    case DecodeFinishCause::LoopGuard: return "loop_guard";
    case DecodeFinishCause::DecodeVisibleProgressTimeout: return "decode_visible_progress_timeout";
    case DecodeFinishCause::SchedulerEmptyBatchStall: return "scheduler_empty_batch_stall";
    case DecodeFinishCause::SchedulerUnschedulable: return "scheduler_unschedulable";
    case DecodeFinishCause::BatchBuildStall: return "batch_build_stall";
    case DecodeFinishCause::RequestCanceled: return "request_canceled";
    case DecodeFinishCause::OutOfMemory: return "out_of_memory";
    case DecodeFinishCause::MissingTokens: return "missing_tokens";
    case DecodeFinishCause::SchedulerRejected: return "scheduler_rejected";
    case DecodeFinishCause::Unknown:
    default: return "unknown";
    }
}

const char* DecodeSilentFinishReasonName(DecodeSilentFinishReason reason) {
    switch (reason) {
    case DecodeSilentFinishReason::ReasoningSuppressedOnly: return "reasoning_suppressed_only";
    case DecodeSilentFinishReason::Utf8PendingOnly: return "utf8_pending_only";
    case DecodeSilentFinishReason::None:
    default: return "none";
    }
}

void NoteDecodeSampleProgress(Request* req, std::chrono::steady_clock::time_point now, int token_id) {
    if (!req) {
        return;
    }
    req->last_sampled_token_time = now;
    req->decode_no_output_steps++;
    req->sampled_token_count++;
    if (req->first_sampled_token_id < 0) {
        req->first_sampled_token_id = token_id;
    }
    if (req->last_external_emit_time == std::chrono::steady_clock::time_point()) {
        req->last_external_emit_time = now;
    }
}

void NoteSuppressedToken(Request* req) {
    if (!req) {
        return;
    }
    req->suppressed_token_count++;
}

void NoteVisibleEmitProgress(Request* req, std::chrono::steady_clock::time_point now, int token_id) {
    if (!req) {
        return;
    }
    req->last_external_emit_time = now;
    req->decode_no_output_steps = 0;
    req->visible_emitted_token_count++;
    if (req->first_visible_token_id < 0) {
        req->first_visible_token_id = token_id;
    }
}

void FinalizeDecodeSilentFinishReason(Request* req) {
    if (!req) {
        return;
    }
    req->decode_silent_finish_reason = DecodeSilentFinishReason::None;
    if (req->visible_emitted_token_count != 0) {
        return;
    }
    if (req->suppressed_token_count != 0) {
        req->decode_silent_finish_reason = DecodeSilentFinishReason::ReasoningSuppressedOnly;
        return;
    }
    if (!req->utf8_pending.empty()) {
        req->decode_silent_finish_reason = DecodeSilentFinishReason::Utf8PendingOnly;
    }
}

void LogRequestDecodeSummary(const Request* req, const TransformerModel* model) {
    if (!req || !model) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::QWEN36) {
        return;
    }
    const auto ns_to_ms = [](uint64_t ns) { return static_cast<double>(ns) / 1000000.0; };
    const auto point_to_ms = [](std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end) -> double {
        if (start == std::chrono::steady_clock::time_point() || end == std::chrono::steady_clock::time_point() ||
            end < start) {
            return 0.0;
        }
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
    };
    const int prompt_tokens =
        req->prompt_token_count > 0 ? req->prompt_token_count : std::max(0, req->n_past - req->generated_count);
    const uint64_t steady_visible_tokens =
        req->visible_emitted_token_count > 0 ? (req->visible_emitted_token_count - 1) : 0;
    const double prefill_ttft_ms = point_to_ms(req->start_time, req->first_token_time);
    const double decode_visible_ms =
        steady_visible_tokens > 0 ? point_to_ms(req->first_token_time, req->last_external_emit_time) : 0.0;
    const double steady_visible_tok_s =
        (decode_visible_ms > 0.0) ? (static_cast<double>(steady_visible_tokens) / (decode_visible_ms / 1000.0)) : 0.0;
    const DecodeRuntimeStatsSnapshot runtime = GetDecodeRuntimeStatsSnapshot();
    const KVRuntimeStatsSnapshot kv_stats = GetKVRuntimeStatsSnapshot();
    const double paged_hit_rate =
        runtime.path_total > 0 ? (100.0 * static_cast<double>(runtime.path_paged) / runtime.path_total) : 0.0;
    const densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
    const int physical_core_count = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
    const bool sve_runtime_detected = densecore::simd::HasArmSveOrBetter(simd_level);
    const bool sve_compiled_enabled = CompiledWithArmSveForSummary();
    const bool sve2_compiled_enabled = CompiledWithArmSve2ForSummary();
    std::cerr
        << "[Qwen36DecodeSummary] req=" << req->id
        << " finish_cause=" << DecodeFinishCauseName(req->decode_finish_cause)
        << " silent_reason=" << DecodeSilentFinishReasonName(req->decode_silent_finish_reason)
        << " prompt_tokens=" << prompt_tokens << " sampled_tokens=" << req->sampled_token_count
        << " visible_tokens=" << req->visible_emitted_token_count << " steady_visible_tokens=" << steady_visible_tokens
        << " long_form_visible=" << (req->visible_emitted_token_count >= 64 ? 1 : 0)
        << " nonempty_visible_output=" << (req->visible_emitted_token_count > 0 ? 1 : 0)
        << " suppressed_tokens=" << req->suppressed_token_count << " prefill_ttft_ms=" << prefill_ttft_ms
        << " decode_visible_ms=" << decode_visible_ms << " steady_visible_tok_s=" << steady_visible_tok_s
        << " active_threads=" << req->active_thread_count << " prefill_threads=" << req->prefill_thread_count
        << " prefill_policy=" << (req->prefill_thread_policy.empty() ? "none" : req->prefill_thread_policy.c_str())
        << " decode_threads=" << req->decode_thread_count
        << " decode_policy=" << (req->decode_thread_policy.empty() ? "none" : req->decode_thread_policy.c_str())
        << " simd_level=" << densecore::simd::SimdLevelName(simd_level) << " physical_cores=" << physical_core_count
        << " scheduler_wait_ms=" << ns_to_ms(req->scheduler_wait_ns)
        << " batch_build_ms=" << ns_to_ms(req->batch_build_ns) << " graph_build_ms=" << ns_to_ms(req->graph_build_ns)
        << " graph_rebind_ms=" << ns_to_ms(req->graph_rebind_ns)
        << " graph_execute_ms=" << ns_to_ms(req->graph_execute_ns) << " attention_ms=" << ns_to_ms(req->attention_ns)
        << " paged_attention_ms=" << ns_to_ms(req->paged_attention_ns)
        << " standard_attention_ms=" << ns_to_ms(req->standard_attention_ns)
        << " portable_flash_attention_ms=" << ns_to_ms(req->portable_flash_attention_ns)
        << " native_flash_attention_ms=" << ns_to_ms(req->native_flash_attention_ns)
        << " hal_attention_ms=" << ns_to_ms(req->hal_attention_ns)
        << " attention_repack_ms=" << ns_to_ms(req->attention_repack_ns)
        << " moe_forward_ms=" << ns_to_ms(req->moe_forward_ns)
        << " shared_expert_ms=" << ns_to_ms(req->shared_expert_ns)
        << " quant_matmul_ms=" << ns_to_ms(req->quant_matmul_ns) << " ssm_conv1d_ms=" << ns_to_ms(req->ssm_conv1d_ns)
        << " ssm_delta_ms=" << ns_to_ms(req->ssm_delta_ns) << " kv_update_ms=" << ns_to_ms(req->kv_update_ns)
        << " sample_ms=" << ns_to_ms(req->sample_ns) << " graph_cache_hits=" << req->graph_cache_hit_count
        << " graph_cache_misses=" << req->graph_cache_miss_count << " paged_hit_rate=" << paged_hit_rate
        << " paged_path_hits=" << runtime.path_paged << " decode_path_total=" << runtime.path_total
        << " prefill_chunk_tokens_effective=" << req->prefill_chunk_tokens_effective
        << " q4k_true_batched_used=" << req->q4k_true_batched_used
        << " arm_batched_quant_used=" << req->arm_batched_quant_used
        << " sve_runtime_detected=" << (sve_runtime_detected ? 1 : 0)
        << " sve_compiled_enabled=" << (sve_compiled_enabled ? 1 : 0)
        << " sve2_compiled_enabled=" << (sve2_compiled_enabled ? 1 : 0)
        << " attention_path_paged=" << req->attention_path_paged
        << " attention_path_standard=" << req->attention_path_standard
        << " attention_path_portable_flash=" << req->attention_path_portable_flash
        << " attention_path_native_flash=" << req->attention_path_native_flash
        << " attention_path_hal=" << req->attention_path_hal << " moe_task_count=" << req->moe_task_count
        << " selected_expert_count=" << req->selected_expert_count << " ssm_conv1d_calls=" << req->ssm_conv1d_calls
        << " ssm_delta_calls=" << req->ssm_delta_calls << " shared_quant_reused=" << runtime.shared_quant_reused
        << " shared_quant_total=" << runtime.shared_quant_total
        << " kv_single_slot_read_count=" << kv_stats.single_slot_read_count
        << " kv_single_slot_write_count=" << kv_stats.single_slot_write_count
        << " kv_bulk_read_count=" << kv_stats.bulk_read_count << " kv_bulk_write_count=" << kv_stats.bulk_write_count
        << " kv_slot_fallback_count=" << kv_stats.slot_fallback_count
        << " kv_hot_path_alloc_count=" << kv_stats.hot_path_alloc_count << " kv_bulk_reads=" << kv_stats.bulk_read_calls
        << " kv_bulk_read_slots=" << kv_stats.bulk_read_slots << " kv_bulk_writes=" << kv_stats.bulk_write_calls
        << " kv_bulk_write_slots=" << kv_stats.bulk_write_slots
        << " kv_slot_read_fallbacks=" << kv_stats.slot_read_fallback_calls
        << " kv_slot_write_fallbacks=" << kv_stats.slot_write_fallback_calls
        << " kv_scratch_grows=" << kv_stats.scratch_buffer_grows
        << " first_sampled_token_id=" << req->first_sampled_token_id
        << " first_visible_token_id=" << req->first_visible_token_id << " generated_count=" << req->generated_count
        << " n_past=" << req->n_past << std::endl;
}

bool HasDecodeVisibleProgressStalled(const Request* req, std::chrono::steady_clock::time_point now) {
    if (!req || req->finished || req->last_sampled_token_time == std::chrono::steady_clock::time_point()) {
        return false;
    }
    const auto last_emit = (req->last_external_emit_time == std::chrono::steady_clock::time_point())
                               ? req->last_sampled_token_time
                               : req->last_external_emit_time;
    const auto silent_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_emit).count();
    return req->decode_no_output_steps >= static_cast<uint64_t>(DecodeVisibleProgressMaxSilentSteps()) ||
           silent_ms >= DecodeVisibleProgressTimeoutMs();
}
