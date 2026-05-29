#include "runtime/worker_internal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

#include "densecore/arm_runtime.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/exceptions.h"
#include "densecore/models/model_descriptor.h"
#include "ggml.h"
#include "kernels/q4k_repacked_gemv.h"
#include "models/model_inference_policy.h"
#include "runtime/kernel_admission.h"
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

bool CompiledWithX86Avx512() {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
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

int EffectiveWorkerThreadCap(int physical_core_count, int base_threads) {
    int cap = CapThreadsToAvailableCores(physical_core_count, base_threads);
    if (physical_core_count > 0) {
        cap = std::max(cap, physical_core_count);
    }
    if (base_threads > 0) {
        cap = std::max(cap, base_threads);
    }
    return std::max(1, std::min(cap, 256));
}

bool IsBenchmarkOrServerPerfProfile() {
    if (densecore::env::ParseTruthyEnv("DENSECORE_BENCH_MODE", false) ||
        densecore::env::ParseTruthyEnv("DENSECORE_BENCH_RESPECT_THREADS", false)) {
        return true;
    }
    const std::string profile = densecore::env::AsciiLowerCopy(std::getenv("DENSECORE_BENCHMARK_PROFILE"));
    return profile == "single-e2e" || profile == "go-server" || profile == "native-runtime";
}

const char* RuntimeToggleModeSummaryName(densecore::env::RuntimeToggleMode mode) {
    switch (mode) {
    case densecore::env::RuntimeToggleMode::Off: return "off";
    case densecore::env::RuntimeToggleMode::Auto: return "auto";
    case densecore::env::RuntimeToggleMode::On: return "on";
    }
    return "off";
}

densecore::env::RuntimeToggleMode ParseSummaryRuntimeToggleFailClosed(const char* name,
                                                                      densecore::env::RuntimeToggleMode default_mode) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_mode;
    }
    return densecore::env::ParseRuntimeToggleModeValue(value, densecore::env::RuntimeToggleMode::Off);
}

int EffectiveCloudWorkerCap(int physical_core_count, int base_threads, bool benchmark_or_server_perf_profile) {
    if (!benchmark_or_server_perf_profile || base_threads < 16) {
        return CapThreadsToAvailableCores(physical_core_count, base_threads);
    }
    return EffectiveWorkerThreadCap(physical_core_count, base_threads);
}

bool IsQwen35HybridSsmSingleRequest(const TransformerModel* model, int num_seqs) {
    if (num_seqs != 1 || !model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    return descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36;
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
    const int n_head = model->hparams.n_head;
    int n_head_kv = model->hparams.n_head_kv;
    if (n_head <= 0) {
        return false;
    }
    if (!model->gemma4_layer_n_head_kv.empty()) {
        for (uint32_t layer_n_head_kv : model->gemma4_layer_n_head_kv) {
            if (layer_n_head_kv == 0) {
                return false;
            }
            const int layer_heads = static_cast<int>(layer_n_head_kv);
            if ((n_head % layer_heads) != 0) {
                return false;
            }
            if (n_head_kv <= 0) {
                n_head_kv = layer_heads;
            }
        }
    }
    if (n_head_kv <= 0 || (n_head % n_head_kv) != 0) {
        return false;
    }

    const auto descriptor = densecore::models::DescribeModel(model);
    if (model->arch_flags.is_hybrid_ssm && descriptor.variant == ModelVariant::QWEN35) {
        for (const TransformerLayer& layer : model->layers) {
            const ggml_tensor* wq = layer.Get(model_keys::kAttnQWeight);
            const ggml_tensor* wk = layer.Get(model_keys::kAttnKWeight);
            if (!wq || !wk || wq->ne[1] <= 0 || wk->ne[1] <= 0) {
                continue;
            }
            int head_dim_q = 0;
            const int64_t q_out = wq->ne[1];
            if (q_out % (2LL * n_head) == 0) {
                head_dim_q = static_cast<int>(q_out / (2LL * n_head));
            } else if (q_out % n_head == 0) {
                head_dim_q = static_cast<int>(q_out / n_head);
            }
            const int head_dim_kv = (wk->ne[1] % n_head_kv) == 0 ? static_cast<int>(wk->ne[1] / n_head_kv) : 0;
            if (head_dim_q > 0 && head_dim_kv > 0) {
                *n_head_out = n_head;
                *n_head_kv_out = n_head_kv;
                *head_dim_q_out = head_dim_q;
                *head_dim_kv_out = head_dim_kv;
                return true;
            }
        }
    }
    const bool qwen36_hybrid_ssm = model->arch_flags.is_hybrid_ssm && descriptor.variant == ModelVariant::QWEN36;
    const int head_dim_q = (qwen36_hybrid_ssm && model->hparams.n_embd_head_k > 0) ? model->hparams.n_embd_head_k
                                                                                   : (model->hparams.n_embd / n_head);
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

bool IsQwenHybridSSMSingleDecodeCacheCandidate(const TransformerModel* model);

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

void EmitRequestResult(EngineState* state, Request* req, const std::string& token, int token_id, bool finished,
                       bool error, bool use_direct_callback) {
    if (!req || (!req->callback && !req->callback_ex && !req->token_result_callback)) {
        return;
    }

    if (use_direct_callback) {
        if (req->callback_ex) {
            req->callback_ex(token.data(), static_cast<int>(token.size()), token_id, finished ? 1 : (error ? 1 : 0),
                             req->user_data);
        } else if (req->callback) {
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

    PushResultEvent(state, req->id, token, token_id, finished, error, req->callback, req->callback_ex,
                    req->token_result_callback, req->user_data);
}

bool ShouldBypassSingleRequestFastPathForLongHybridSSM(const TransformerModel* model, const Request* req) {
    if (!model || !req || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    return req->is_prefill && static_cast<int>(req->tokens.size()) > BLOCK_SIZE;
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
    if (!model) {
        return false;
    }
    // Gemma4 graph reuse is safe only when every decode attention layer uses
    // paged attention. Mixed sliding-paged + full standard attention embeds
    // n_past-dependent graph shapes and drifts when reused across tokens.
    if (model->arch_flags.is_gemma4) {
        return densecore::models::SupportsPagedDecodeAttention(model);
    }
    // Qwen3.5/Qwen3.6 hybrid-SSM single-token decode has stable paged-attention
    // topology and the SSM custom-op runtime state is rebound on every cache reuse.
    if (IsQwenHybridSSMSingleDecodeCacheCandidate(model)) {
        return true;
    }
    // Other hybrid-SSM graphs still contain request-local state pointers until
    // their runtime rebind coverage is qualified.
    if (model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    return true;
}

bool DoesDecodeGraphCacheRequireRuntimeRebind(const TransformerModel* model) {
    return model && model->arch_flags.is_hybrid_ssm;
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
    feature_flags |= (1ull << 1);  // paged decode is the maintained decode path.
    feature_flags |= (1ull << 2);  // batched paged decode is admitted by model/shape checks.
    if (model && model->arch_flags.requires_q_norm) feature_flags |= (1ull << 6);
    if (model && model->arch_flags.requires_k_norm) feature_flags |= (1ull << 7);
    if (DENSECORE_DEFAULT_PRECOMPUTED_ROPE != 0) feature_flags |= (1ull << 8);
    if (DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM != 0) feature_flags |= (1ull << 9);
    if (DENSECORE_DEFAULT_FUSED_QKV != 0) feature_flags |= (1ull << 10);
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
    static const int max_batch = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_MAX_BATCH", 16);
    return max_batch;
}

int DecodeGraphCacheLruSize() {
    static const int lru = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_LRU_SIZE", 32);
    return lru;
}

size_t DecodeGraphCacheCtxBytes() {
    static const size_t bytes = []() {
        const int mb = ParsePositiveEnvIntOrDefault("DENSECORE_DECODE_GRAPH_CACHE_CTX_MB", 192);
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

bool IsQwenHybridSSMSingleDecodeCacheCandidate(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    return descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36;
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

    if (IsQwen35HybridSsmSingleRequest(model, num_seqs) && IsWideSimdLevel(simd_level)) {
        const auto descriptor = densecore::models::DescribeModel(model);
        const bool is_qwen35 = descriptor.variant == ModelVariant::QWEN35;
        const bool is_qwen36 = descriptor.variant == ModelVariant::QWEN36;
        const int effective_cap =
            EffectiveCloudWorkerCap(physical_core_count, base_threads, IsBenchmarkOrServerPerfProfile());
        if (base_threads >= 16 && effective_cap >= 16 && (is_qwen35 || is_qwen36)) {
            selection.threads = 16;
        }
        if ((simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2) &&
            effective_cap >= 16) {
            selection.label = is_qwen35 ? "prefill_qwen35_single_long_prompt" : "prefill_qwen36_single_long_prompt";
            return selection;
        }
        if (prompt_token_count > 0 && prompt_token_count < 64) {
            selection.threads = std::min(selection.threads, 8);
            selection.label = is_qwen35 ? "prefill_qwen35_single_short_prompt" : "prefill_qwen36_single_short_prompt";
        } else if (prompt_token_count > 0 && prompt_token_count < 128) {
            selection.threads = std::min(selection.threads, 12);
            selection.label = is_qwen35 ? "prefill_qwen35_single_short_prompt" : "prefill_qwen36_single_short_prompt";
        } else {
            selection.label = is_qwen35 ? "prefill_qwen35_single_long_prompt" : "prefill_qwen36_single_long_prompt";
        }
    }
    if (model && model->arch_flags.is_gemma4 && num_seqs == 1 && model->hparams.n_experts > 0 &&
        IsWideSimdLevel(simd_level)) {
        const int cap = CapThreadsToAvailableCores(physical_core_count, base_threads);
        if (cap >= 16) {
            selection.threads = 16;
            selection.label = densecore::simd::IsArmFamily(simd_level) ? "prefill_gemma4_a4b_c4a_moe_16"
                                                                       : "prefill_gemma4_a4b_c4_moe_16";
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
        // Single-sequence decode on 16+ core machines: use all cores.
        // MoE GEMV is memory-bandwidth bound — more threads = more parallel
        // expert tile work. Multi-seq keeps 8 to avoid per-seq contention.
        threads_per_seq = (cap >= 16 && num_seqs == 1) ? cap : 8;
        break;
    case densecore::simd::SimdLevel::SVE:
    case densecore::simd::SimdLevel::SVE2: threads_per_seq = (cap >= 16 && num_seqs == 1) ? cap : 8; break;
    case densecore::simd::SimdLevel::NEON: threads_per_seq = (cap >= 16 && num_seqs == 1) ? cap : 6; break;
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

    const bool gemma4_single_request = model && model->arch_flags.is_gemma4 && num_seqs == 1;
    if (gemma4_single_request && IsWideSimdLevel(simd_level)) {
        const int cap = CapThreadsToAvailableCores(physical_core_count, base_threads);
        const bool arm_c4a_wide_simd =
            densecore::simd::IsArmFamily(simd_level) &&
            (simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2);
        if (arm_c4a_wide_simd && physical_core_count >= 16 && cap >= 16) {
            selection.threads = 16;
            selection.label =
                (model->hparams.n_experts > 0) ? "decode_gemma4_a4b_c4a_moe_16" : "decode_gemma4_dense_c4a_16";
            return selection;
        }
        if (arm_c4a_wide_simd && cap >= 12) {
            selection.threads = 12;
            selection.label =
                (model->hparams.n_experts > 0) ? "decode_gemma4_a4b_arm_safe_cap" : "decode_gemma4_dense_arm_12";
            return selection;
        }
        if (!densecore::simd::IsArmFamily(simd_level) && model->hparams.n_experts > 0 && cap >= 16) {
            selection.threads = 16;
            selection.label = "decode_gemma4_a4b_c4_moe_16";
            return selection;
        }
        selection.threads = std::max(1, std::min(cap, 8));
        selection.label =
            (model->hparams.n_experts > 0) ? "decode_gemma4_a4b_safe_cap" : "decode_gemma4_dense_safe_cap";
        return selection;
    }

    if (!IsQwen35HybridSsmSingleRequest(model, num_seqs)) {
        return selection;
    }
    if (!IsWideSimdLevel(simd_level)) {
        return selection;
    }

    const int cap = EffectiveCloudWorkerCap(physical_core_count, base_threads, IsBenchmarkOrServerPerfProfile());
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts <= 0 &&
        !densecore::simd::IsArmFamily(simd_level) && cap >= 16) {
        selection.threads = 16;
        selection.label = "decode_qwen35_dense_c4_16";
        return selection;
    }
    if (descriptor.variant == ModelVariant::QWEN36 && model->hparams.n_experts <= 0 &&
        !densecore::simd::IsArmFamily(simd_level) && cap >= 16) {
        selection.threads = 16;
        selection.label = "decode_qwen36_dense_c4_16";
        return selection;
    }
    if (model->hparams.n_experts > 0) {
        const int qwen_moe_cap = EffectiveWorkerThreadCap(physical_core_count, base_threads);
        const bool qwen35_a3b = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
        const bool qwen36_a3b = descriptor.variant == ModelVariant::QWEN36 && model->hparams.n_experts > 0;
        if (!qwen35_a3b && !qwen36_a3b) {
            return selection;
        }
        const bool arm_c4a_wide_simd =
            densecore::simd::IsArmFamily(simd_level) &&
            (simd_level == densecore::simd::SimdLevel::SVE || simd_level == densecore::simd::SimdLevel::SVE2);
        if (arm_c4a_wide_simd && qwen_moe_cap >= 16) {
            selection.threads = 16;
            selection.label = qwen35_a3b ? "decode_qwen35_a3b_c4a_moe_16" : "decode_qwen36_a3b_c4a_moe_16";
            return selection;
        }
        if (arm_c4a_wide_simd && qwen_moe_cap >= 12) {
            selection.threads = 12;
            selection.label = qwen35_a3b ? "decode_qwen35_a3b_arm_safe_cap" : "decode_qwen36_a3b_arm_safe_cap";
            return selection;
        }
        if (!densecore::simd::IsArmFamily(simd_level) && qwen_moe_cap >= 16) {
            selection.threads = 16;
            selection.label = qwen35_a3b ? "decode_qwen35_a3b_c4_moe_16" : "decode_qwen36_a3b_c4_moe_16";
            return selection;
        }
        selection.threads = std::max(1, std::min(qwen_moe_cap, 8));
        selection.label = qwen35_a3b ? "decode_qwen35_a3b_safe_cap" : "decode_qwen36_a3b_safe_cap";
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
    static std::atomic<int> debug_budget{
        densecore::env::ParsePositiveEnvInt("DENSECORE_DEBUG_DECODE_GRAPH_CACHE_STABILITY_MAX", 0)};
    const auto debug_fail = [&](const char* reason, int n_tokens_in_batch = -1, int n_head = 0, int n_head_kv = 0,
                                int head_dim_q = 0, int head_dim_kv = 0) {
        int remaining = debug_budget.load(std::memory_order_relaxed);
        while (remaining > 0 &&
               !debug_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {}
        if (remaining > 0) {
            std::cerr << "[DecodeGraphCacheStability] stable=0 reason=" << (reason ? reason : "unknown")
                      << " num_seqs=" << batch.num_seqs << " tokens=" << batch.tokens.size()
                      << " n_tokens_in_batch=" << n_tokens_in_batch << " n_head=" << n_head
                      << " n_head_kv=" << n_head_kv << " head_dim_q=" << head_dim_q << " head_dim_kv=" << head_dim_kv
                      << " has_cache=" << (cache ? 1 : 0);
            if (cache) {
                std::cerr << " max_blocks=" << cache->max_blocks
                          << " cache_type=" << static_cast<int>(cache->cache_type);
            }
            if (!batch.seq_id.empty()) {
                std::cerr << " seq0=" << batch.seq_id[0];
            }
            if (!batch.pos.empty()) {
                std::cerr << " pos0=" << batch.pos[0];
            }
            if (!batch.n_past.empty()) {
                std::cerr << " n_past0=" << batch.n_past[0];
            }
            if (!batch.block_tables.empty()) {
                std::cerr << " block_table0_size=" << batch.block_tables[0].size();
                if (!batch.block_tables[0].empty()) {
                    std::cerr << " block0=" << batch.block_tables[0][0];
                }
            }
            std::cerr << std::endl;
        }
        return false;
    };

    if (!model) {
        return debug_fail("missing_model");
    }
    if (!cache) {
        return debug_fail("missing_cache");
    }
    if (model->arch_flags.is_glm_dsa) {
        return debug_fail("glm_dsa");
    }
    if (!densecore::models::SupportsPagedDecodeAttention(model)) {
        return debug_fail("unsupported_model");
    }
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    if (!IsDecodeOnlyBatchLayout(batch, n_tokens_in_batch)) {
        return debug_fail("non_decode_layout", n_tokens_in_batch);
    }

    int n_head = 0;
    int n_head_kv = 0;
    int head_dim_q = 0;
    int head_dim_kv = 0;
    if (!ResolvePagedDecodeHeadDims(model, &n_head, &n_head_kv, &head_dim_q, &head_dim_kv)) {
        return debug_fail("head_dim_resolve_failed", n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv);
    }
    // Gemma4 carries fixed heterogeneous attention shapes; the actual per-layer
    // paged-decode graph is deterministic, but the generic topology candidate
    // helper only needs a shape-compatible head tuple to validate sequence and
    // block-table invariants.
    const int candidate_head_dim_kv = model->arch_flags.is_gemma4 ? head_dim_q : head_dim_kv;
    if (!IsPagedDecodeCandidate(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q,
                                candidate_head_dim_kv)) {
        return debug_fail("paged_candidate_failed", n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv);
    }

    if (batch.num_seqs > 1) {
        return true;
    }
    return IsPagedDecodeModeAlwaysOn() || IsQwenHybridSSMSingleDecodeCacheCandidate(model);
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
                      << ",b4=" << worker_stats.last_threads_by_batch[4].load(std::memory_order_relaxed)
                      << ",b8=" << worker_stats.last_threads_by_batch[8].load(std::memory_order_relaxed) << "]"
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
                      << ",layout=" << worker_stats.graph_cache_skip_layout.load(std::memory_order_relaxed)
                      << ",max_batch=" << worker_stats.graph_cache_skip_max_batch.load(std::memory_order_relaxed)
                      << ",unstable=" << worker_stats.graph_cache_skip_unstable.load(std::memory_order_relaxed)
                      << ",lora=" << worker_stats.graph_cache_skip_lora.load(std::memory_order_relaxed)
                      << ",model=" << worker_stats.graph_cache_skip_model.load(std::memory_order_relaxed)
                      << ",backend=" << worker_stats.graph_cache_skip_backend.load(std::memory_order_relaxed)
                      << ",key_uncacheable="
                      << worker_stats.graph_cache_skip_uncacheable.load(std::memory_order_relaxed) << ",build_failure="
                      << worker_stats.graph_cache_skip_build_failure.load(std::memory_order_relaxed)
                      << ",rebind_failure="
                      << worker_stats.graph_cache_skip_rebind_failure.load(std::memory_order_relaxed) << ",uncacheable="
                      << worker_stats.graph_cache_rejected_uncacheable.load(std::memory_order_relaxed) << "]";
            std::cerr << " prefill_arena_reuse[hit="
                      << worker_stats.prefill_arena_reuse_hit.load(std::memory_order_relaxed)
                      << ",miss=" << worker_stats.prefill_arena_reuse_miss.load(std::memory_order_relaxed) << "]";
            std::cerr << " prefill_graph_reuse[hit="
                      << worker_stats.prefill_graph_reuse_hit.load(std::memory_order_relaxed)
                      << ",skip_model=" << worker_stats.prefill_graph_reuse_skip_model.load(std::memory_order_relaxed)
                      << "]";

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
    if (descriptor.variant != ModelVariant::QWEN35 && descriptor.variant != ModelVariant::QWEN36 &&
        descriptor.variant != ModelVariant::GEMMA4) {
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
    const double prefill_tok_s = (prefill_ttft_ms > 0.0 && prompt_tokens > 0)
                                     ? (static_cast<double>(prompt_tokens) / (prefill_ttft_ms / 1000.0))
                                     : 0.0;
    const double decode_visible_ms =
        steady_visible_tokens > 0 ? point_to_ms(req->first_token_time, req->last_external_emit_time) : 0.0;
    const double steady_visible_tok_s =
        (decode_visible_ms > 0.0) ? (static_cast<double>(steady_visible_tokens) / (decode_visible_ms / 1000.0)) : 0.0;
    const DecodeRuntimeStatsSnapshot runtime = GetDecodeRuntimeStatsSnapshot();
    const KVRuntimeStatsSnapshot kv_stats = GetKVRuntimeStatsSnapshot();
    const double paged_runtime_hit_rate =
        runtime.path_total > 0 ? (100.0 * static_cast<double>(runtime.path_paged) / runtime.path_total) : 0.0;
    const int attention_path_total = req->attention_path_paged + req->attention_path_standard +
                                     req->attention_path_portable_flash + req->attention_path_native_flash +
                                     req->attention_path_hal;
    const double paged_hit_rate = attention_path_total > 0
                                      ? (100.0 * static_cast<double>(req->attention_path_paged) / attention_path_total)
                                      : paged_runtime_hit_rate;
    const densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
    const int physical_core_count = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
    const bool sve_runtime_detected = densecore::simd::HasArmSveOrBetter(simd_level);
    const bool sve_compiled_enabled = CompiledWithArmSveForSummary();
    const bool sve2_compiled_enabled = CompiledWithArmSve2ForSummary();
    const bool x86_avx512_compiled_enabled = CompiledWithX86Avx512();
    const char* summary_tag = "[Qwen36DecodeSummary]";
    if (descriptor.variant == ModelVariant::QWEN35) {
        summary_tag = "[Qwen35DecodeSummary]";
    } else if (descriptor.variant == ModelVariant::GEMMA4) {
        summary_tag = "[Gemma4DecodeSummary]";
    }
    const auto weight_hist_string = [](const std::array<uint64_t, kMatmulWeightTypeHistCount>& hist) {
        static constexpr const char* labels[kMatmulWeightTypeHistCount] = {"q4_k", "q5_k", "q6_k", "q8_0",
                                                                           "f16",  "f32",  "other"};
        std::ostringstream oss;
        for (std::size_t i = 0; i < kMatmulWeightTypeHistCount; ++i) {
            if (i != 0) oss << ",";
            oss << labels[i] << ":" << hist[i];
        }
        return oss.str();
    };
    const auto quant_input_hist_string = [](const std::array<uint64_t, kMatmulQuantInputTypeHistCount>& hist) {
        static constexpr const char* labels[kMatmulQuantInputTypeHistCount] = {"q8_k", "q8_0", "none", "other"};
        std::ostringstream oss;
        for (std::size_t i = 0; i < kMatmulQuantInputTypeHistCount; ++i) {
            if (i != 0) oss << ",";
            oss << labels[i] << ":" << hist[i];
        }
        return oss.str();
    };
    const auto path_hist_string = [](const std::array<uint64_t, kMatmulPathHistCount>& hist) {
        static constexpr const char* labels[kMatmulPathHistCount] = {
            "ggml_mul_mat", "ggml_mul_mat_id", "custom_gemv", "custom_batched_gemv",
            "moe_native",   "ssm_projection",  "other"};
        std::ostringstream oss;
        for (std::size_t i = 0; i < kMatmulPathHistCount; ++i) {
            if (i != 0) oss << ",";
            oss << labels[i] << ":" << hist[i];
        }
        return oss.str();
    };
    const auto shape_census_string = [](const std::vector<MatmulShapeCensusEntry>& entries) {
        if (entries.empty()) {
            return std::string("none");
        }
        auto safe = [](std::string value) {
            for (char& ch : value) {
                if (std::isspace(static_cast<unsigned char>(ch)) || ch == ';' || ch == ':') {
                    ch = '_';
                }
            }
            return value;
        };
        std::ostringstream oss;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (i != 0) oss << ";";
            oss << safe(entry.model_family.empty() ? "unknown" : entry.model_family) << ":" << safe(entry.phase) << ":"
                << safe(entry.dispatch_path) << ":" << safe(entry.weight_type) << ":" << safe(entry.shape_bucket);
            if (!entry.op_type.empty()) {
                oss << ":op=" << safe(entry.op_type);
            }
            if (!entry.weight_class.empty()) {
                oss << ":class=" << safe(entry.weight_class);
            }
            if (entry.wall_ns != 0) {
                oss << ":ms=" << (static_cast<double>(entry.wall_ns) / 1.0e6);
            }
            if (entry.calls != 0) {
                oss << ":calls=" << entry.calls;
            }
            if (entry.active_threads != 0) {
                oss << ":threads=" << entry.active_threads;
            }
            if (entry.contiguous_or_copy_input != 0) {
                oss << ":copy_in=" << entry.contiguous_or_copy_input;
            }
            oss << ":ops=" << entry.ops << ":w=" << safe(entry.left_name) << ":x=" << safe(entry.right_name);
        }
        return oss.str();
    };
    const auto top_slow_string = [](const std::vector<MatmulDispatchCensusEntry>& entries) {
        if (entries.empty()) {
            return std::string("none");
        }
        std::ostringstream oss;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (i != 0) oss << ";";
            oss << (entry.model_family.empty() ? "unknown" : entry.model_family) << ":" << entry.phase << ":"
                << entry.dispatch_path << ":" << entry.weight_type << ":" << entry.shape_bucket
                << ":ms=" << (static_cast<double>(entry.wall_ns) / 1.0e6) << ",ops=" << entry.ops;
        }
        return oss.str();
    };
    const std::string gemv_weight_hist = weight_hist_string(req->gemv_custom_weight_type_hist);
    const std::string gemv_quant_input_hist = quant_input_hist_string(req->gemv_custom_quant_input_type_hist);
    const std::string moe_weight_hist = weight_hist_string(req->moe_expert_matmul_weight_type_hist);
    const std::string ssm_weight_hist = weight_hist_string(req->qwen36_ssm_projection_weight_type_hist);
    const std::string matmul_top_slow = top_slow_string(req->matmul_dispatch_top_slow_entries);
    const std::string decode_matmul_weight_hist = weight_hist_string(req->decode_matmul_weight_type_hist);
    const std::string decode_matmul_path_hist = path_hist_string(req->decode_matmul_path_hist);
    const std::string decode_matmul_top_shapes = shape_census_string(req->decode_matmul_top_shapes);
    const std::string prefill_matmul_weight_hist = weight_hist_string(req->prefill_matmul_weight_type_hist);
    const std::string prefill_matmul_path_hist = path_hist_string(req->prefill_matmul_path_hist);
    const std::string q6k_gemv_weight_shapes = shape_census_string(req->q6k_gemv_weight_shapes);
    const std::string native_moe_graph_top_slow_nodes = shape_census_string(req->native_moe_graph_top_slow_nodes);
    const std::string decode_graph_top_slow_nodes = shape_census_string(req->decode_graph_top_slow_nodes);
    std::ostringstream decode_graph_node_hist;
    decode_graph_node_hist << "custom:count=" << req->decode_graph_node_custom_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_custom_ns)
                           << ",mul_mat:count=" << req->decode_graph_node_mul_mat_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_mul_mat_ns)
                           << ",mul_mat_id:count=" << req->decode_graph_node_mul_mat_id_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_mul_mat_id_ns)
                           << ",norm:count=" << req->decode_graph_node_norm_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_norm_ns)
                           << ",view_copy:count=" << req->decode_graph_node_view_copy_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_view_copy_ns)
                           << ",elementwise:count=" << req->decode_graph_node_elementwise_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_elementwise_ns)
                           << ",attention:count=" << req->decode_graph_node_attention_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_attention_ns)
                           << ",other:count=" << req->decode_graph_node_other_count
                           << ":ms=" << ns_to_ms(req->decode_graph_node_other_ns);
    const std::string qwen35_moe_w1w3_hist = weight_hist_string(req->qwen35_moe_w1w3_weight_type_hist);
    const std::string qwen35_moe_w2_hist = weight_hist_string(req->qwen35_moe_w2_weight_type_hist);
    const std::string qwen36_prefill_top_slow_ops = shape_census_string(req->qwen36_prefill_top_slow_ops);
    const std::string gemma4_prefill_top_slow_ops = shape_census_string(req->gemma4_prefill_top_slow_ops);
    const auto native_moe_fast_decode_config = ParseSummaryRuntimeToggleFailClosed(
        "DENSECORE_NATIVE_MOE_FAST_DECODE", densecore::env::RuntimeToggleMode::Auto);
    const bool native_moe_fast_w2_q5k_used = req->native_moe_fast_w2_q5k_used_ops > 0;
    const bool native_moe_fast_w2_q5k_rejected = req->native_moe_fast_w2_q5k_rejected_ops > 0;
    const char* native_moe_fast_mode_effective = "auto_discovery_only";
    if (native_moe_fast_decode_config == densecore::env::RuntimeToggleMode::Off) {
        native_moe_fast_mode_effective = "off";
    } else if (req->native_moe_fast_decode_used_ops > 0) {
        native_moe_fast_mode_effective = "auto_used_native_graph_w2_q5k";
    } else if (req->native_moe_fast_decode_rejected_ops > 0) {
        native_moe_fast_mode_effective = "auto_rejected_noop";
    } else if (native_moe_fast_decode_config == densecore::env::RuntimeToggleMode::On) {
        native_moe_fast_mode_effective = "on_noop";
    }
    const char* native_moe_fast_w2_q5k_effective_state = "candidate";
    if (native_moe_fast_w2_q5k_used) {
        native_moe_fast_w2_q5k_effective_state = "used";
    } else if (native_moe_fast_w2_q5k_rejected) {
        native_moe_fast_w2_q5k_effective_state = "rejected";
    } else if (steady_visible_tokens == 0 || req->qwen35_moe_forward_calls == 0) {
        native_moe_fast_w2_q5k_effective_state = "off";
    }
    const uint64_t native_moe_fast_w1w3_seen_ops =
        req->native_moe_fallback_w1w3_ops + req->native_moe_fast_w1w3_used_ops;
    const uint64_t native_moe_fast_w2_seen_ops = req->native_moe_fallback_w2_ops + req->native_moe_fast_w2_used_ops;
    const uint64_t native_moe_fast_w2_q5k_seen_ops =
        std::max(req->native_moe_fast_w2_q5k_candidate_ops,
                 req->native_moe_fast_w2_q5k_used_ops + req->native_moe_fast_w2_q5k_rejected_ops);
    const uint64_t native_moe_fast_decode_seen_ops =
        std::max(req->native_moe_fast_decode_candidate_ops,
                 req->native_moe_fast_decode_used_ops + req->native_moe_fast_decode_rejected_ops);
    const char* qwen35_native_moe_down_exec_path =
        native_moe_fast_w2_q5k_used ? "custom_op" : (req->native_moe_fallback_w2_ops > 0 ? "ggml_mul_mat_id" : "none");
    const char* native_graph_moe_down_q5k_applicability = "unsupported";
    if (native_moe_fast_w2_q5k_used) {
        native_graph_moe_down_q5k_applicability = "used";
    } else if (native_moe_fast_w2_q5k_rejected) {
        native_graph_moe_down_q5k_applicability = "rejected";
    } else if (steady_visible_tokens > 0 && req->qwen35_moe_forward_calls > 0) {
        native_graph_moe_down_q5k_applicability = "active";
    }
    const int moe_w2_fast_path_wrong_boundary = req->native_moe_fast_w2_q5k_used_ops == 0 &&
                                                        req->native_moe_fallback_w2_ops == 0 &&
                                                        steady_visible_tokens > 0 && req->qwen35_moe_forward_calls > 0
                                                    ? 1
                                                    : 0;
    const bool qwen35_moe_descriptor = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
    const int qwen35_moe_instrumentation_missing =
        qwen35_moe_descriptor && req->qwen35_moe_forward_calls == 0 ? 1 : req->qwen35_moe_instrumentation_missing;
    const bool native_moe_expected =
        (descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36) &&
        model->hparams.n_experts > 0 &&
        (req->qwen35_moe_path == "native_graph" || req->native_moe_graph_ns > 0 ||
         !req->native_moe_graph_node_hist.empty());
    const int native_moe_timing_missing =
        native_moe_expected && req->native_moe_graph_ns == 0 ? 1 : req->native_moe_timing_missing;
    const char* q6k_effective_state = "unused";
    if (req->q6k_gemv_used_ops != 0) {
        q6k_effective_state = "used";
    } else if (req->q6k_gemv_rejected_ops != 0) {
        q6k_effective_state = "rejected";
    }
    const std::string q6k_last_reject_reason =
        req->q6k_gemv_last_reject_reason.empty()
            ? (req->q6k_gemv_seen_ops > 0 && req->q6k_gemv_candidate_ops == 0 ? "unknown_pre_candidate" : "none")
            : req->q6k_gemv_last_reject_reason;
    const char* q4k_applicability = "active";
    if (req->q4k_repacked_gemv_effective_state == "disabled") {
        q4k_applicability = "disabled";
    } else if (req->q4k_repacked_gemv_used_ops != 0 || req->q4k_repacked_gemv_used != 0) {
        q4k_applicability = "active";
    } else if (req->q4k_repacked_gemv_rejected_ops != 0) {
        q4k_applicability = "rejected";
    } else if (req->decode_matmul_created_ops == 0) {
        q4k_applicability = "no_decode_gemv_seen";
    } else if (req->q4k_repacked_gemv_seen_ops == 0 && req->decode_matmul_weight_type_hist[0] == 0) {
        q4k_applicability = "no_q4k_seen";
    }
    auto resolve_effective_attention_path = [&]() {
        const bool gemma4_prefill_flash_seen =
            descriptor.variant == ModelVariant::GEMMA4 && req->flash_attention_headseq_prefill_calls > 0;
        if (gemma4_prefill_flash_seen) {
            return req->flash_attention_reference_calls > 0 ? "portable_cpu_flash_reference" : "portable_cpu_flash";
        }
        if (req->attention_path_paged > 0) {
            return "paged_decode_attention";
        }
        if (req->attention_path_native_flash > 0) {
            return "native_flash";
        }
        if (req->attention_path_portable_flash > 0) {
            return "portable_cpu_flash";
        }
        if (req->attention_path_standard > 0) {
            return "standard_attention";
        }
        if (req->attention_path_hal > 0) {
            return "hal";
        }
        return "none";
    };
    const char* effective_attention_path = resolve_effective_attention_path();
    if (descriptor.variant == ModelVariant::GEMMA4) {
        std::cerr << "[Gemma4PrefillSummary]" << " req=" << req->id << " prefill_ttft_ms=" << prefill_ttft_ms
                  << " prompt_tokens=" << prompt_tokens << " prefill_tok_s=" << prefill_tok_s
                  << " graph_build_ms=" << ns_to_ms(req->graph_build_ns)
                  << " graph_execute_ms=" << ns_to_ms(req->graph_execute_ns)
                  << " portable_flash_attention_ms=" << ns_to_ms(req->portable_flash_attention_ns)
                  << " hal_attention_ms=" << ns_to_ms(req->hal_attention_ns)
                  << " standard_attention_ms=" << ns_to_ms(req->standard_attention_ns)
                  << " gemma4_prefill_total_ms=" << ns_to_ms(req->gemma4_prefill_total_ns)
                  << " gemma4_prefill_graph_build_ms=" << ns_to_ms(req->gemma4_prefill_graph_build_ns)
                  << " gemma4_prefill_graph_execute_ms=" << ns_to_ms(req->gemma4_prefill_graph_execute_ns)
                  << " gemma4_prefill_attention_ms=" << ns_to_ms(req->gemma4_prefill_attention_ns)
                  << " gemma4_prefill_moe_or_mlp_ms=" << ns_to_ms(req->gemma4_prefill_moe_or_mlp_ns)
                  << " gemma4_prefill_mul_mat_id_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_id_ns)
                  << " gemma4_prefill_mul_mat_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_ns)
                  << " gemma4_prefill_flash_attention_ms=" << ns_to_ms(req->gemma4_prefill_flash_attention_ns)
                  << " gemma4_dense_prefill_native_used_ops=" << req->gemma4_dense_prefill_native_used_ops
                  << " gemma4_dense_prefill_native_ms=" << ns_to_ms(req->gemma4_dense_prefill_native_ns)
                  << " gemma4_dense_prefill_replaced_ggml_mul_mat_ops="
                  << req->gemma4_dense_prefill_replaced_ggml_mul_mat_ops
                  << " gemma4_fast_gelu_used=" << req->gemma4_fast_gelu_used
                  << " gemma4_fast_gelu_ms=" << ns_to_ms(req->gemma4_fast_gelu_ns)
                  << " gemma4_native_moe_prefill_gate_up_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_gate_up_ns)
                  << " gemma4_native_moe_prefill_down_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_down_ns)
                  << " gemma4_native_moe_prefill_total_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_total_ns)
                  << " gemma4_prefill_top_slow_ops=" << gemma4_prefill_top_slow_ops
                  << " moe_forward_ms=" << ns_to_ms(req->moe_forward_ns)
                  << " moe_route_ms=" << ns_to_ms(req->moe_route_ns)
                  << " moe_expert_ms=" << ns_to_ms(req->moe_expert_ns)
                  << " shared_expert_ms=" << ns_to_ms(req->shared_expert_ns)
                  << " kv_update_ms=" << ns_to_ms(req->kv_update_ns) << " active_threads=" << req->active_thread_count
                  << " simd_level=" << densecore::simd::SimdLevelName(simd_level)
                  << " compute_flash_attention_reference_used=" << (req->flash_attention_reference_calls > 0 ? 1 : 0)
                  << " x86_avx512_compiled_enabled=" << (x86_avx512_compiled_enabled ? 1 : 0)
                  << " effective_attention_path=" << effective_attention_path
                  << " flash_attention_headseq_prefill_calls=" << req->flash_attention_headseq_prefill_calls
                  << " flash_attention_native_decode_calls=" << req->flash_attention_native_decode_calls
                  << " flash_attention_non_avx512_tiled_calls=" << req->flash_attention_non_avx512_tiled_calls
                  << " flash_attention_avx512_tiled_calls=" << req->flash_attention_avx512_tiled_calls
                  << " flash_attention_last_nth=" << req->flash_attention_last_nth
                  << " flash_attention_last_active_threads=" << req->flash_attention_last_active_threads << std::endl;
    }
    if (req->native_moe_fast_decode_candidate_ops > 0 && req->native_moe_fast_decode_used_ops == 0) {
        std::cerr << summary_tag << "[WARN] native_moe_fast_decode_candidate_without_use" << " req=" << req->id
                  << " candidates=" << req->native_moe_fast_decode_candidate_ops
                  << " rejected=" << req->native_moe_fast_decode_rejected_ops << " reason="
                  << (req->native_moe_fast_decode_last_reject_reason.empty()
                          ? "none"
                          : req->native_moe_fast_decode_last_reject_reason.c_str())
                  << std::endl;
    }
    if (req->q6k_gemv_seen_ops > 0 && req->q6k_gemv_candidate_ops == 0) {
        std::cerr << summary_tag << "[WARN] q6k_gemv_seen_without_candidate" << " req=" << req->id
                  << " seen=" << req->q6k_gemv_seen_ops << " rejected=" << req->q6k_gemv_rejected_ops
                  << " reason=" << q6k_last_reject_reason << std::endl;
    }
    if (req->graph_ctx_requested_mb > 0 && req->graph_ctx_available_mb > 0 &&
        req->graph_ctx_requested_mb + req->graph_ctx_safety_margin_mb > req->graph_ctx_available_mb) {
        std::cerr << summary_tag << "[WARN] graph_ctx_request_exceeds_safe_available" << " req=" << req->id
                  << " requested_mb=" << req->graph_ctx_requested_mb << " available_mb=" << req->graph_ctx_available_mb
                  << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << " fail_reason="
                  << (req->graph_ctx_fail_reason.empty() ? "none" : req->graph_ctx_fail_reason.c_str()) << std::endl;
    }
    std::cerr
        << summary_tag << " req=" << req->id << " finish_cause=" << DecodeFinishCauseName(req->decode_finish_cause)
        << " silent_reason=" << DecodeSilentFinishReasonName(req->decode_silent_finish_reason)
        << " prompt_tokens=" << prompt_tokens << " sampled_tokens=" << req->sampled_token_count
        << " visible_tokens=" << req->visible_emitted_token_count << " steady_visible_tokens=" << steady_visible_tokens
        << " long_form_visible=" << (req->visible_emitted_token_count >= 64 ? 1 : 0)
        << " nonempty_visible_output=" << (req->visible_emitted_token_count > 0 ? 1 : 0)
        << " token_id_submit_used=" << (req->token_id_submit_used ? 1 : 0)
        << " callback_mode=" << (req->callback_mode.empty() ? "unknown" : req->callback_mode.c_str())
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
        << " graph_execute_ms=" << ns_to_ms(req->graph_execute_ns)
        << " decode_graph_execute_ms=" << ns_to_ms(req->decode_graph_execute_ns)
        << " decode_attention_ms=" << ns_to_ms(req->decode_attention_ns)
        << " decode_paged_attention_ms=" << ns_to_ms(req->decode_paged_attention_ns)
        << " decode_native_moe_graph_ms=" << ns_to_ms(req->decode_native_moe_graph_ns)
        << " decode_moe_route_ms=" << ns_to_ms(req->decode_moe_route_ns)
        << " decode_moe_w1w3_ms=" << ns_to_ms(req->decode_moe_w1w3_ns)
        << " decode_moe_w2_ms=" << ns_to_ms(req->decode_moe_w2_ns)
        << " decode_moe_reduce_ms=" << ns_to_ms(req->decode_moe_reduce_ns)
        << " decode_ssm_qkv_ms=" << ns_to_ms(req->decode_ssm_qkv_wall_ns)
        << " decode_ssm_out_ms=" << ns_to_ms(req->decode_ssm_out_wall_ns)
        << " decode_ssm_delta_wall_ms=" << ns_to_ms(req->decode_ssm_delta_wall_ns)
        << " decode_ssm_conv1d_ms=" << ns_to_ms(req->decode_ssm_conv1d_ns)
        << " decode_ssm_delta_ms=" << ns_to_ms(req->decode_ssm_delta_ns)
        << " decode_sample_ms=" << ns_to_ms(req->decode_sample_ns)
        << " decode_graph_node_measured_ms=" << ns_to_ms(req->decode_graph_node_measured_ns)
        << " decode_graph_node_hist=" << decode_graph_node_hist.str()
        << " decode_graph_top_slow_nodes=" << decode_graph_top_slow_nodes
        << " attention_ms=" << ns_to_ms(req->attention_ns)
        << " paged_attention_ms=" << ns_to_ms(req->paged_attention_ns)
        << " standard_attention_ms=" << ns_to_ms(req->standard_attention_ns)
        << " portable_flash_attention_ms=" << ns_to_ms(req->portable_flash_attention_ns)
        << " native_flash_attention_ms=" << ns_to_ms(req->native_flash_attention_ns)
        << " hal_attention_ms=" << ns_to_ms(req->hal_attention_ns)
        << " attention_repack_ms=" << ns_to_ms(req->attention_repack_ns)
        << " moe_forward_ms=" << ns_to_ms(req->moe_forward_ns) << " moe_route_ms=" << ns_to_ms(req->moe_route_ns)
        << " moe_reorder_ms=" << ns_to_ms(req->moe_reorder_ns) << " moe_expert_ms=" << ns_to_ms(req->moe_expert_ns)
        << " moe_reduce_ms=" << ns_to_ms(req->moe_reduce_ns) << " moe_w1w3_ms=" << ns_to_ms(req->moe_w1w3_ns)
        << " moe_w2_ms=" << ns_to_ms(req->moe_w2_ns) << " moe_rowblock_used=" << req->moe_rowblock_used
        << " moe_rowblock_ms=" << ns_to_ms(req->moe_rowblock_ns)
        << " moe_rowblock_w1w3_ms=" << ns_to_ms(req->moe_rowblock_w1w3_ns)
        << " moe_rowblock_w2_ms=" << ns_to_ms(req->moe_rowblock_w2_ns)
        << " shared_expert_ms=" << ns_to_ms(req->shared_expert_ns)
        << " quant_matmul_ms=" << ns_to_ms(req->quant_matmul_ns) << " ssm_qkv_ms=" << ns_to_ms(req->ssm_qkv_wall_ns)
        << " ssm_gate_ms=" << ns_to_ms(req->ssm_gate_wall_ns)
        << " ssm_delta_wall_ms=" << ns_to_ms(req->ssm_delta_wall_ns) << " ssm_out_ms=" << ns_to_ms(req->ssm_out_wall_ns)
        << " ssm_conv1d_ms=" << ns_to_ms(req->ssm_conv1d_ns) << " ssm_delta_ms=" << ns_to_ms(req->ssm_delta_ns)
        << " kv_update_ms=" << ns_to_ms(req->kv_update_ns) << " sample_ms=" << ns_to_ms(req->sample_ns)
        << " kleidiai_compiled_enabled=" << req->kleidiai_compiled_enabled
        << " kleidiai_candidate_ops=" << req->kleidiai_candidate_ops
        << " kleidiai_allowed_ops=" << req->kleidiai_allowed_ops
        << " kleidiai_rejected_ops=" << req->kleidiai_rejected_ops << " kleidiai_last_reject_reason="
        << densecore::runtime::KernelAdmissionRejectReasonName(
               static_cast<densecore::runtime::KernelAdmissionRejectReason>(req->kleidiai_last_reject_reason))
        << " graph_cache_hits=" << req->graph_cache_hit_count << " graph_cache_misses=" << req->graph_cache_miss_count
        << " graph_cache_skips=" << req->graph_cache_skip_count << " graph_cache_last_skip_reason="
        << (req->graph_cache_last_skip_reason.empty() ? "none" : req->graph_cache_last_skip_reason.c_str())
        << " prefix_cache_allowed=" << (req->prefix_cache_allowed ? 1 : 0)
        << " prefix_cache_hit=" << (req->prefix_cache_hit ? 1 : 0)
        << " prefix_cache_skipped_tokens=" << req->prefix_cache_skipped_tokens
        << " prefix_cache_hit_blocks=" << req->prefix_cache_hit_blocks
        << " prefix_cache_registered_blocks=" << req->prefix_cache_registered_blocks
        << " prefix_cache_extended_blocks=" << req->prefix_cache_extended_blocks
        << " hybrid_ssm_snapshot_restore_attempted=" << (req->hybrid_ssm_snapshot_restore_attempted ? 1 : 0)
        << " hybrid_ssm_snapshot_restore_applied=" << (req->hybrid_ssm_snapshot_restore_applied ? 1 : 0)
        << " prefix_cache_skip_reason="
        << (req->prefix_cache_skip_reason.empty() ? "none" : req->prefix_cache_skip_reason.c_str())
        << " paged_hit_rate=" << paged_hit_rate << " paged_path_hits=" << req->attention_path_paged
        << " decode_path_total=" << attention_path_total << " paged_runtime_hit_rate=" << paged_runtime_hit_rate
        << " paged_runtime_path_hits=" << runtime.path_paged << " decode_runtime_path_total=" << runtime.path_total
        << " prefill_chunk_tokens_effective=" << req->prefill_chunk_tokens_effective
        << " graph_ctx_requested_mb=" << req->graph_ctx_requested_mb
        << " graph_ctx_available_mb=" << req->graph_ctx_available_mb
        << " graph_ctx_safety_margin_mb=" << req->graph_ctx_safety_margin_mb
        << " graph_ctx_downgraded_chunk_tokens=" << req->graph_ctx_downgraded_chunk_tokens << " graph_ctx_fail_reason="
        << (req->graph_ctx_fail_reason.empty() ? "none" : req->graph_ctx_fail_reason.c_str())
        << " q4k_true_batched_used=" << req->q4k_true_batched_used
        << " qwen36_prefill_q4k_batched_mode=" << req->qwen36_prefill_q4k_batched_mode
        << " qwen36_prefill_q4k_batched_used=" << req->qwen36_prefill_q4k_batched_used
        << " qwen36_prefill_q4k_batched_probe_pass=" << req->qwen36_prefill_q4k_batched_probe_pass
        << " qwen36_prefill_q4k_batched_reject_reason="
        << (req->qwen36_prefill_q4k_batched_reject_reason.empty()
                ? "none"
                : req->qwen36_prefill_q4k_batched_reject_reason.c_str())
        << " qwen36_prefill_q4k_batched_max_abs_error=" << req->qwen36_prefill_q4k_batched_max_abs_error
        << " qwen36_prefill_q4k_probe_participants=" << req->qwen36_prefill_q4k_probe_participants
        << " qwen36_prefill_q4k_probe_failures=" << req->qwen36_prefill_q4k_probe_failures
        << " qwen36_prefill_q4k_admission_downgraded=" << req->qwen36_prefill_q4k_admission_downgraded
        << " qwen36_ssm_q8_prefill_amx_mode=" << req->qwen36_ssm_q8_prefill_amx_mode
        << " qwen36_ssm_q8_prefill_amx_prepared=" << req->qwen36_ssm_q8_prefill_amx_prepared
        << " qwen36_ssm_q8_prefill_amx_used=" << req->qwen36_ssm_q8_prefill_amx_used
        << " qwen36_ssm_q8_prefill_amx_reject_reason="
        << (req->qwen36_ssm_q8_prefill_amx_reject_reason.empty() ? "none"
                                                                 : req->qwen36_ssm_q8_prefill_amx_reject_reason.c_str())
        << " qwen36_ssm_q8_prefill_amx_prepared_projection_counts="
        << (req->qwen36_ssm_q8_prefill_amx_prepared_projection_counts.empty()
                ? "none"
                : req->qwen36_ssm_q8_prefill_amx_prepared_projection_counts.c_str())
        << " qwen36_ssm_q8_prefill_amx_projection_counts="
        << (req->qwen36_ssm_q8_prefill_amx_projection_counts.empty()
                ? "none"
                : req->qwen36_ssm_q8_prefill_amx_projection_counts.c_str())
        << " qwen36_ssm_projection_weight_type_hist=" << ssm_weight_hist
        << " qwen36_ssm_q8_prefill_amx_candidate_ops=" << req->qwen36_ssm_q8_prefill_amx_candidate_ops
        << " qwen36_ssm_q8_prefill_amx_used_ops=" << req->qwen36_ssm_q8_prefill_amx_used_ops
        << " qwen36_ssm_q8_prefill_amx_rejected_ops=" << req->qwen36_ssm_q8_prefill_amx_rejected_ops
        << " qwen36_ssm_q8_decode_used_original_q8_path=" << req->qwen36_ssm_q8_decode_used_original_q8_path
        << " qwen36_ssm_projection_quant_preserved=" << req->qwen36_ssm_projection_quant_preserved
        << " qwen36_ssm_projection_dequantized_count=" << req->qwen36_ssm_projection_dequantized_count
        << " qwen36_ssm_projection_actual_types="
        << (req->qwen36_ssm_projection_actual_types.empty() ? "none" : req->qwen36_ssm_projection_actual_types.c_str())
        << " q4k_repacked_gemv_used=" << req->q4k_repacked_gemv_used
        << " q4k_repacked_gemv_seen_ops=" << req->q4k_repacked_gemv_seen_ops
        << " q4k_repacked_gemv_candidate_ops=" << req->q4k_repacked_gemv_candidate_ops
        << " q4k_repacked_gemv_used_ops=" << req->q4k_repacked_gemv_used_ops
        << " q4k_repacked_gemv_rejected_ops=" << req->q4k_repacked_gemv_rejected_ops
        << " q4k_repacked_gemv_cache_hits=" << req->q4k_repacked_gemv_cache_hits
        << " q4k_repacked_gemv_cache_waited_hits=" << req->q4k_repacked_gemv_cache_waited_hits
        << " q4k_repacked_gemv_cache_misses=" << req->q4k_repacked_gemv_cache_misses
        << " q4k_repacked_gemv_cache_evictions=" << req->q4k_repacked_gemv_cache_evictions
        << " q4k_repacked_gemv_cache_evicted_bytes=" << req->q4k_repacked_gemv_cache_evicted_bytes
        << " q4k_repacked_gemv_repack_bytes=" << req->q4k_repacked_gemv_repack_bytes
        << " q4k_repacked_gemv_probe_ms=" << (static_cast<double>(req->q4k_repacked_gemv_probe_ns) / 1.0e6)
        << " q4k_repacked_gemv_resident_bytes=" << req->q4k_repacked_gemv_resident_bytes
        << " q4k_repacked_gemv_cache_limit_bytes=" << densecore::kernels::Q4KRepackedGemvCacheLimitBytes()
        << " q4k_repacked_gemv_cache_limit_source="
        << (densecore::kernels::Q4KRepackedGemvManualCacheLimitConfigured() ? "manual" : "auto")
        << " q4k_repacked_gemv_distinct_weights_seen=" << req->q4k_repacked_gemv_distinct_weights_seen
        << " q4k_repacked_gemv_repeated_repack_count=" << req->q4k_repacked_gemv_repeated_repack_count
        << " q4k_repacked_gemv_last_reject_reason="
        << (req->q4k_repacked_gemv_last_reject_reason_text.empty()
                ? "none"
                : req->q4k_repacked_gemv_last_reject_reason_text.c_str())
        << " q4k_repacked_gemv_primary_disable_reason="
        << (req->q4k_repacked_gemv_primary_disable_reason_text.empty()
                ? "none"
                : req->q4k_repacked_gemv_primary_disable_reason_text.c_str())
        << " q4k_repacked_gemv_cache_thrash_detected=" << req->q4k_repacked_gemv_cache_thrash_detected
        << " q4k_repacked_gemv_effective_state="
        << (req->q4k_repacked_gemv_effective_state.empty() ? "unused" : req->q4k_repacked_gemv_effective_state.c_str())
        << " q4k_repacked_gemv_applicability=" << q4k_applicability
        << " gemv_custom_total_ops=" << req->gemv_custom_total_ops
        << " gemv_custom_decode_ops=" << req->gemv_custom_decode_ops
        << " gemv_custom_prefill_ops=" << req->gemv_custom_prefill_ops
        << " gemv_custom_q4k_seen_ops=" << req->gemv_custom_q4k_seen_ops
        << " gemv_custom_non_q4k_ops=" << req->gemv_custom_non_q4k_ops
        << " gemv_custom_quant_input_null_ops=" << req->gemv_custom_quant_input_null_ops
        << " gemv_custom_shape_reject_ops=" << req->gemv_custom_shape_reject_ops
        << " gemv_custom_phase_unknown_ops=" << req->gemv_custom_phase_unknown_ops
        << " gemv_custom_force_reference_ops=" << req->gemv_custom_force_reference_ops
        << " gemv_custom_dynamic_lora_ops=" << req->gemv_custom_dynamic_lora_ops
        << " gemv_custom_weight_type_hist=" << gemv_weight_hist
        << " gemv_custom_quant_input_type_hist=" << gemv_quant_input_hist
        << " gemv_custom_tasks_effective=" << req->gemv_custom_tasks_effective << " gemv_custom_tasks_cap_reason="
        << (req->gemv_custom_tasks_cap_reason.empty() ? "none" : req->gemv_custom_tasks_cap_reason.c_str())
        << " decode_matmul_created_ops=" << req->decode_matmul_created_ops
        << " decode_matmul_weight_type_hist=" << decode_matmul_weight_hist
        << " decode_matmul_path_hist=" << decode_matmul_path_hist
        << " decode_matmul_top_shapes=" << decode_matmul_top_shapes
        << " prefill_matmul_weight_type_hist=" << prefill_matmul_weight_hist
        << " prefill_matmul_path_hist=" << prefill_matmul_path_hist << " q6k_gemv_seen_ops=" << req->q6k_gemv_seen_ops
        << " q6k_gemv_candidate_ops=" << req->q6k_gemv_candidate_ops << " q6k_gemv_used_ops=" << req->q6k_gemv_used_ops
        << " q6k_gemv_rejected_ops=" << req->q6k_gemv_rejected_ops
        << " q6k_gemv_reject_quant_input_null_ops=" << req->q6k_gemv_reject_quant_input_null_ops
        << " q6k_gemv_reject_unsupported_quant_input_ops=" << req->q6k_gemv_reject_unsupported_quant_input_ops
        << " q6k_gemv_reject_shape_ops=" << req->q6k_gemv_reject_shape_ops
        << " q6k_gemv_reject_phase_ops=" << req->q6k_gemv_reject_phase_ops
        << " q6k_gemv_reject_kernel_unavailable_ops=" << req->q6k_gemv_reject_kernel_unavailable_ops
        << " q6k_gemv_last_reject_reason=" << q6k_last_reject_reason << " q6k_gemv_effective_phase="
        << (req->q6k_gemv_effective_phase.empty() ? "none" : req->q6k_gemv_effective_phase.c_str())
        << " q6k_gemv_graph_phase=" << (req->q6k_gemv_graph_phase.empty() ? "none" : req->q6k_gemv_graph_phase.c_str())
        << " q6k_gemv_callback_phase="
        << (req->q6k_gemv_callback_phase.empty() ? "none" : req->q6k_gemv_callback_phase.c_str())
        << " q6k_gemv_weight_shapes=" << q6k_gemv_weight_shapes
        << " q6k_gemv_total_ms=" << ns_to_ms(req->q6k_gemv_total_ns)
        << " q6k_gemv_effective_state=" << q6k_effective_state
        << " q4k_copied_gemv_experiment_used=" << req->q4k_copied_gemv_experiment_used
        << " q4k_copied_gemv_experiment_cache_hits=" << req->q4k_copied_gemv_experiment_cache_hits
        << " q4k_copied_gemv_experiment_cache_misses=" << req->q4k_copied_gemv_experiment_cache_misses
        << " q4k_copied_gemv_experiment_reject_reason="
        << (req->q4k_copied_gemv_experiment_reject_reason.empty()
                ? "none"
                : req->q4k_copied_gemv_experiment_reject_reason.c_str())
        << " qact_cache_hits=" << req->qact_cache_hits << " qact_cache_misses=" << req->qact_cache_misses
        << " qact_cache_reused_bytes=" << req->qact_cache_reused_bytes
        << " moe_small_decode_parallel_candidate_ops=" << req->moe_small_decode_parallel_candidate_ops
        << " moe_small_decode_parallel_used_ops=" << req->moe_small_decode_parallel_used_ops
        << " moe_small_decode_parallel_rejected_ops=" << req->moe_small_decode_parallel_rejected_ops
        << " moe_small_decode_parallel_last_reject_reason="
        << (req->moe_small_decode_parallel_last_reject_reason.empty()
                ? "none"
                : req->moe_small_decode_parallel_last_reject_reason.c_str())
        << " moe_expert_matmul_weight_type_hist=" << moe_weight_hist
        << " moe_q4k_repacked_candidate_ops=" << req->moe_q4k_repacked_candidate_ops
        << " moe_q4k_repacked_used_ops=" << req->moe_q4k_repacked_used_ops
        << " moe_q4k_repacked_rejected_ops=" << req->moe_q4k_repacked_rejected_ops
        << " moe_q4k_repacked_last_reject_reason="
        << (req->moe_q4k_repacked_last_reject_reason.empty() ? "none"
                                                             : req->moe_q4k_repacked_last_reject_reason.c_str())
        << " moe_q5k_repacked_candidate_ops=" << req->moe_q5k_repacked_candidate_ops
        << " moe_q5k_repacked_used_ops=" << req->moe_q5k_repacked_used_ops
        << " moe_q5k_repacked_rejected_ops=" << req->moe_q5k_repacked_rejected_ops
        << " moe_q5k_repacked_last_reject_reason="
        << (req->moe_q5k_repacked_last_reject_reason.empty() ? "none"
                                                             : req->moe_q5k_repacked_last_reject_reason.c_str())
        << " gemma4_moe_prefill_quant_batch_candidate_ops=" << req->gemma4_moe_prefill_quant_batch_candidate_ops
        << " gemma4_moe_prefill_quant_batch_used_ops=" << req->gemma4_moe_prefill_quant_batch_used_ops
        << " gemma4_moe_prefill_quant_batch_rejected_ops=" << req->gemma4_moe_prefill_quant_batch_rejected_ops
        << " gemma4_moe_prefill_quant_batch_last_reject_reason="
        << (req->gemma4_moe_prefill_quant_batch_last_reject_reason.empty()
                ? "none"
                : req->gemma4_moe_prefill_quant_batch_last_reject_reason.c_str())
        << " gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops="
        << req->gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops
        << " gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops="
        << req->gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops
        << " gemma4_moe_prefill_quant_batch_gate_up_used=" << req->gemma4_moe_prefill_quant_batch_gate_up_used
        << " gemma4_moe_prefill_quant_batch_down_used=" << req->gemma4_moe_prefill_quant_batch_down_used
        << " gemma4_native_moe_prefill_candidate_layers=" << req->gemma4_native_moe_prefill_candidate_layers
        << " gemma4_native_moe_prefill_used_layers=" << req->gemma4_native_moe_prefill_used_layers
        << " gemma4_native_moe_prefill_rejected_layers=" << req->gemma4_native_moe_prefill_rejected_layers
        << " gemma4_native_moe_prefill_last_reject_reason="
        << (req->gemma4_native_moe_prefill_last_reject_reason.empty()
                ? "none"
                : req->gemma4_native_moe_prefill_last_reject_reason.c_str())
        << " gemma4_native_moe_prefill_gate_up_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_gate_up_ns)
        << " gemma4_native_moe_prefill_down_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_down_ns)
        << " gemma4_native_moe_prefill_total_ms=" << ns_to_ms(req->gemma4_native_moe_prefill_total_ns)
        << " gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops="
        << req->gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops
        << " gemma4_native_moe_prefill_duplicate_work_detected="
        << req->gemma4_native_moe_prefill_duplicate_work_detected
        << " gemma4_dense_prefill_native_candidate_ops=" << req->gemma4_dense_prefill_native_candidate_ops
        << " gemma4_dense_prefill_native_used_ops=" << req->gemma4_dense_prefill_native_used_ops
        << " gemma4_dense_prefill_native_rejected_ops=" << req->gemma4_dense_prefill_native_rejected_ops
        << " gemma4_dense_prefill_native_last_reject_reason="
        << (req->gemma4_dense_prefill_native_last_reject_reason.empty()
                ? "none"
                : req->gemma4_dense_prefill_native_last_reject_reason.c_str())
        << " gemma4_dense_prefill_native_q4k_ops=" << req->gemma4_dense_prefill_native_q4k_ops
        << " gemma4_dense_prefill_native_q8_0_ops=" << req->gemma4_dense_prefill_native_q8_0_ops
        << " gemma4_dense_prefill_native_ms=" << ns_to_ms(req->gemma4_dense_prefill_native_ns)
        << " gemma4_dense_prefill_replaced_ggml_mul_mat_ops=" << req->gemma4_dense_prefill_replaced_ggml_mul_mat_ops
        << " gemma4_dense_prefill_duplicate_work_detected=" << req->gemma4_dense_prefill_duplicate_work_detected
        << " gemma4_fast_gelu_enabled=" << req->gemma4_fast_gelu_enabled
        << " gemma4_fast_gelu_used=" << req->gemma4_fast_gelu_used
        << " gemma4_fast_gelu_ms=" << ns_to_ms(req->gemma4_fast_gelu_ns)
        << " gemma4_native_moe_prefill_gate_up_fast_gelu_ms="
        << ns_to_ms(req->gemma4_native_moe_prefill_gate_up_fast_gelu_ns)
        << " gemma4_decode_native_candidate_ops=" << req->gemma4_decode_native_candidate_ops
        << " gemma4_decode_native_used_ops=" << req->gemma4_decode_native_used_ops
        << " gemma4_decode_native_rejected_ops=" << req->gemma4_decode_native_rejected_ops
        << " gemma4_decode_native_last_reject_reason="
        << (req->gemma4_decode_native_last_reject_reason.empty() ? "none"
                                                                 : req->gemma4_decode_native_last_reject_reason.c_str())
        << " gemma4_decode_native_moe_used_ops=" << req->gemma4_decode_native_moe_used_ops
        << " gemma4_decode_native_dense_used_ops=" << req->gemma4_decode_native_dense_used_ops
        << " gemma4_decode_native_lm_head_used_ops=" << req->gemma4_decode_native_lm_head_used_ops
        << " gemma4_decode_native_ms=" << ns_to_ms(req->gemma4_decode_native_ns)
        << " gemma4_decode_replaced_ggml_mul_mat_ops=" << req->gemma4_decode_replaced_ggml_mul_mat_ops
        << " gemma4_decode_replaced_ggml_mul_mat_id_ops=" << req->gemma4_decode_replaced_ggml_mul_mat_id_ops
        << " gemma4_decode_duplicate_work_detected=" << req->gemma4_decode_duplicate_work_detected
        << " gemma4_native_int4_gemv_candidate_ops=" << req->gemma4_native_int4_gemv_candidate_ops
        << " gemma4_native_int4_gemv_used_ops=" << req->gemma4_native_int4_gemv_used_ops
        << " gemma4_native_int4_gemv_ms=" << ns_to_ms(req->gemma4_native_int4_gemv_ns)
        << " gemma4_native_int4_repacked_weight_count=" << req->gemma4_native_int4_repacked_weight_count
        << " gemma4_native_int4_repacked_bytes=" << req->gemma4_native_int4_repacked_bytes
        << " gemma4_native_fused_gateup_used_ops=" << req->gemma4_native_fused_gateup_used_ops
        << " ggml_delegated_quant_gemv_ops=" << req->ggml_delegated_quant_gemv_ops
        << " gemma4_native_paged_attention_candidate_ops=" << req->gemma4_native_paged_attention_candidate_ops
        << " gemma4_native_paged_attention_used_ops=" << req->gemma4_native_paged_attention_used_ops
        << " gemma4_native_paged_attention_ms=" << ns_to_ms(req->gemma4_native_paged_attention_ns)
        << " gemma4_ggml_attention_fallback_ops=" << req->gemma4_ggml_attention_fallback_ops
        << " gemma4_paged_attention_cache_type=" << req->gemma4_paged_attention_cache_type
        << " gemma4_paged_attention_context_len=" << req->gemma4_paged_attention_context_len
        << " gemma4_paged_attention_head_range=" << req->gemma4_paged_attention_head_range
        << " gemma4_ggml_mul_mat_prefill_remaining_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_ns)
        << " gemma4_ggml_mul_mat_id_prefill_remaining_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_id_ns)
        << " gemma4_ggml_mul_mat_decode_remaining_ms=" << ns_to_ms(req->decode_graph_node_mul_mat_ns)
        << " gemma4_ggml_mul_mat_id_decode_remaining_ms=" << ns_to_ms(req->decode_graph_node_mul_mat_id_ns)
        << " gemma4_native_prefill_replaced_ggml_ops="
        << (req->gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops +
            req->gemma4_dense_prefill_replaced_ggml_mul_mat_ops)
        << " gemma4_native_decode_replaced_ggml_ops="
        << (req->gemma4_decode_replaced_ggml_mul_mat_ops + req->gemma4_decode_replaced_ggml_mul_mat_id_ops)
        << " native_moe_graph_ms=" << ns_to_ms(req->native_moe_graph_ns) << " native_moe_graph_node_hist="
        << (req->native_moe_graph_node_hist.empty() ? "none" : req->native_moe_graph_node_hist.c_str())
        << " native_moe_graph_top_slow_nodes=" << native_moe_graph_top_slow_nodes
        << " native_moe_timing_missing=" << native_moe_timing_missing
        << " native_moe_fast_decode_config=" << RuntimeToggleModeSummaryName(native_moe_fast_decode_config)
        << " native_moe_fast_mode_effective=" << native_moe_fast_mode_effective
        << " native_moe_fast_decode_seen_ops=" << native_moe_fast_decode_seen_ops
        << " native_moe_fast_w1w3_seen_ops=" << native_moe_fast_w1w3_seen_ops
        << " native_moe_fast_w2_seen_ops=" << native_moe_fast_w2_seen_ops
        << " native_moe_fast_w2_q5k_seen_ops=" << native_moe_fast_w2_q5k_seen_ops
        << " native_moe_fast_decode_candidate_ops=" << req->native_moe_fast_decode_candidate_ops
        << " native_moe_fast_decode_used_ops=" << req->native_moe_fast_decode_used_ops
        << " native_moe_fast_decode_rejected_ops=" << req->native_moe_fast_decode_rejected_ops
        << " native_moe_fast_decode_last_reject_reason="
        << (req->native_moe_fast_decode_last_reject_reason.empty()
                ? "none"
                : req->native_moe_fast_decode_last_reject_reason.c_str())
        << " native_moe_fast_decode_w1w3_used_ops=" << req->native_moe_fast_decode_w1w3_used_ops
        << " native_moe_fast_decode_w2_used_ops=" << req->native_moe_fast_decode_w2_used_ops
        << " native_moe_fast_decode_ms=" << ns_to_ms(req->native_moe_fast_decode_ns)
        << " native_moe_fallback_w1w3_ms=" << ns_to_ms(req->native_moe_fallback_w1w3_ns)
        << " native_moe_fallback_w2_ms=" << ns_to_ms(req->native_moe_fallback_w2_ns)
        << " native_moe_fast_w1w3_ms=" << ns_to_ms(req->native_moe_fast_w1w3_ns)
        << " native_moe_fast_w2_ms=" << ns_to_ms(req->native_moe_fast_w2_ns)
        << " native_moe_fast_reduce_ms=" << ns_to_ms(req->native_moe_fast_reduce_ns)
        << " native_moe_fast_total_ms=" << ns_to_ms(req->native_moe_fast_total_ns)
        << " native_moe_fast_w1w3_used_ops=" << req->native_moe_fast_w1w3_used_ops
        << " native_moe_fast_w2_used_ops=" << req->native_moe_fast_w2_used_ops
        << " native_moe_fallback_w1w3_ops=" << req->native_moe_fallback_w1w3_ops
        << " native_moe_fallback_w2_ops=" << req->native_moe_fallback_w2_ops
        << " native_moe_fallback_ops=" << req->native_moe_fallback_ops
        << " native_moe_fast_replaced_fallback_ops=" << req->native_moe_fast_replaced_fallback_ops
        << " native_moe_fast_duplicate_work_detected=" << req->native_moe_fast_duplicate_work_detected
        << " native_moe_fast_missing_w2=" << req->native_moe_fast_missing_w2
        << " native_moe_fast_partial_expert_coverage=" << req->native_moe_fast_partial_expert_coverage
        << " native_moe_fast_covered_experts=" << req->native_moe_fast_covered_experts
        << " native_moe_selected_experts=" << req->native_moe_selected_experts
        << " native_moe_fast_w2_q5k_candidate_ops=" << req->native_moe_fast_w2_q5k_candidate_ops
        << " native_moe_fast_w2_q5k_used_ops=" << req->native_moe_fast_w2_q5k_used_ops
        << " native_moe_fast_w2_q5k_rejected_ops=" << req->native_moe_fast_w2_q5k_rejected_ops
        << " native_moe_fast_w2_q5k_last_reject_reason="
        << (req->native_moe_fast_w2_q5k_last_reject_reason.empty()
                ? "none"
                : req->native_moe_fast_w2_q5k_last_reject_reason.c_str())
        << " native_moe_fast_w2_q5k_ms=" << ns_to_ms(req->native_moe_fast_w2_q5k_ns)
        << " native_moe_fast_w2_q5k_config=builtin"
        << " native_moe_fast_w2_q5k_effective_state=" << native_moe_fast_w2_q5k_effective_state
        << " qwen35_native_moe_down_exec_path=" << qwen35_native_moe_down_exec_path
        << " qwen35_native_moe_down_q5k_config=builtin"
        << " qwen35_native_moe_down_q5k_seen_ops=" << native_moe_fast_w2_q5k_seen_ops
        << " qwen35_native_moe_down_q5k_candidate_ops=" << req->native_moe_fast_w2_q5k_candidate_ops
        << " qwen35_native_moe_down_q5k_used_ops=" << req->native_moe_fast_w2_q5k_used_ops
        << " qwen35_native_moe_down_q5k_rejected_ops=" << req->native_moe_fast_w2_q5k_rejected_ops
        << " qwen35_native_moe_down_q5k_last_reject_reason="
        << (req->native_moe_fast_w2_q5k_last_reject_reason.empty()
                ? "none"
                : req->native_moe_fast_w2_q5k_last_reject_reason.c_str())
        << " qwen35_native_moe_down_q5k_ms=" << ns_to_ms(req->native_moe_fast_w2_q5k_ns)
        << " qwen35_native_moe_down_q5k_replaced_fallback_ops=" << req->native_moe_fast_replaced_fallback_ops
        << " qwen35_native_moe_down_q5k_duplicate_work_detected=" << req->native_moe_fast_duplicate_work_detected
        << " native_graph_moe_down_q5k_applicability=" << native_graph_moe_down_q5k_applicability
        << " moe_w2_fast_path_wrong_boundary=" << moe_w2_fast_path_wrong_boundary
        << " qwen35_moe_path=" << (req->qwen35_moe_path.empty() ? "none" : req->qwen35_moe_path.c_str())
        << " qwen35_moe_layers_seen=" << req->qwen35_moe_layers_seen
        << " qwen35_moe_forward_calls=" << req->qwen35_moe_forward_calls
        << " qwen35_moe_w1w3_weight_type_hist=" << qwen35_moe_w1w3_hist
        << " qwen35_moe_w2_weight_type_hist=" << qwen35_moe_w2_hist
        << " qwen_native_moe_w1w3_weight_type_hist=" << qwen35_moe_w1w3_hist
        << " qwen_native_moe_w2_weight_type_hist=" << qwen35_moe_w2_hist
        << " qwen35_moe_route_ms=" << ns_to_ms(req->moe_route_ns)
        << " qwen35_moe_w1w3_ms=" << ns_to_ms(req->moe_w1w3_ns) << " qwen35_moe_w2_ms=" << ns_to_ms(req->moe_w2_ns)
        << " qwen35_moe_reduce_ms=" << ns_to_ms(req->moe_reduce_ns)
        << " qwen35_moe_selected_expert_count=" << req->qwen35_moe_selected_expert_count
        << " qwen35_moe_top_k=" << req->qwen35_moe_top_k
        << " qwen35_moe_instrumentation_missing=" << qwen35_moe_instrumentation_missing
        << " qwen36_moe_route_ms=" << ns_to_ms(req->moe_route_ns)
        << " qwen36_moe_w1w3_ms=" << ns_to_ms(req->moe_w1w3_ns) << " qwen36_moe_w2_ms=" << ns_to_ms(req->moe_w2_ns)
        << " qwen36_moe_reduce_ms=" << ns_to_ms(req->moe_reduce_ns)
        << " moe_selected_expert_count=" << req->moe_selected_expert_count << " moe_top_k=" << req->moe_top_k
        << " moe_expert_parallel_tasks=" << req->moe_expert_parallel_tasks
        << " matmul_dispatch_top_slow=" << matmul_top_slow
        << " qwen_target_ggml_compute_ops=" << req->qwen_target_ggml_compute_ops
        << " qwen_target_ggml_matmul_ops=" << req->qwen_target_ggml_matmul_ops
        << " qwen_target_ggml_matmul_id_ops=" << req->qwen_target_ggml_matmul_id_ops
        << " qwen_target_ggml_quant_vecdot_ops=" << req->qwen_target_ggml_quant_vecdot_ops
        << " qwen_target_ggml_quantize_kv_ops=" << req->qwen_target_ggml_quantize_kv_ops
        << " qwen_target_ggml_attention_ops=" << req->qwen_target_ggml_attention_ops
        << " qwen_target_ggml_compute_last_reason="
        << (req->qwen_target_ggml_compute_last_reason.empty() ? "none"
                                                              : req->qwen_target_ggml_compute_last_reason.c_str())
        << " qwen_target_ggml_compute_last_op="
        << (req->qwen_target_ggml_compute_last_op.empty() ? "none" : req->qwen_target_ggml_compute_last_op.c_str())
        << " qwen_target_ggml_compute_target="
        << (req->qwen_target_ggml_compute_target.empty() ? "none" : req->qwen_target_ggml_compute_target.c_str())
        << " qwen36_prefill_total_ms=" << ns_to_ms(req->qwen36_prefill_total_ns)
        << " qwen36_prefill_ssm_projection_ms=" << ns_to_ms(req->qwen36_prefill_ssm_projection_ns)
        << " qwen36_prefill_ssm_delta_state_ms=" << ns_to_ms(req->qwen36_prefill_ssm_delta_state_ns)
        << " qwen36_prefill_attention_ms=" << ns_to_ms(req->qwen36_prefill_attention_ns)
        << " qwen36_prefill_mlp_or_moe_ms=" << ns_to_ms(req->qwen36_prefill_mlp_or_moe_ns)
        << " qwen36_prefill_native_moe_fast_candidate_ops=" << req->qwen36_prefill_native_moe_fast_candidate_ops
        << " qwen36_prefill_native_moe_fast_used_ops=" << req->qwen36_prefill_native_moe_fast_used_ops
        << " qwen36_prefill_native_moe_fast_rejected_ops=" << req->qwen36_prefill_native_moe_fast_rejected_ops
        << " qwen36_prefill_native_moe_fast_last_reject_reason="
        << (req->qwen36_prefill_native_moe_fast_last_reject_reason.empty()
                ? "none"
                : req->qwen36_prefill_native_moe_fast_last_reject_reason.c_str())
        << " qwen36_prefill_mlp_or_moe_ms_before_fastpath=" << ns_to_ms(req->qwen36_prefill_mlp_or_moe_ns)
        << " qwen36_prefill_mlp_or_moe_ms_after_fastpath=" << ns_to_ms(req->qwen36_prefill_mlp_or_moe_ns)
        << " qwen36_prefill_graph_build_ms=" << ns_to_ms(req->qwen36_prefill_graph_build_ns)
        << " qwen36_prefill_graph_execute_ms=" << ns_to_ms(req->qwen36_prefill_graph_execute_ns)
        << " qwen36_prefill_top_slow_ops=" << qwen36_prefill_top_slow_ops
        << " gemma4_prefill_total_ms=" << ns_to_ms(req->gemma4_prefill_total_ns)
        << " gemma4_prefill_graph_build_ms=" << ns_to_ms(req->gemma4_prefill_graph_build_ns)
        << " gemma4_prefill_graph_execute_ms=" << ns_to_ms(req->gemma4_prefill_graph_execute_ns)
        << " gemma4_prefill_attention_ms=" << ns_to_ms(req->gemma4_prefill_attention_ns)
        << " gemma4_prefill_moe_or_mlp_ms=" << ns_to_ms(req->gemma4_prefill_moe_or_mlp_ns)
        << " gemma4_prefill_mul_mat_id_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_id_ns)
        << " gemma4_prefill_mul_mat_ms=" << ns_to_ms(req->gemma4_prefill_mul_mat_ns)
        << " gemma4_prefill_flash_attention_ms=" << ns_to_ms(req->gemma4_prefill_flash_attention_ns)
        << " gemma4_prefill_top_slow_ops=" << gemma4_prefill_top_slow_ops
        << " paged_attn_decode_head_tile_effective=" << req->paged_attn_decode_head_tile_effective
        << " moe_decode_scratch_reused=" << req->moe_decode_scratch_reused
        << " moe_decode_allocations_avoided=" << req->moe_decode_allocations_avoided
        << " arm_batched_quant_used=" << req->arm_batched_quant_used
        << " sve_runtime_detected=" << (sve_runtime_detected ? 1 : 0)
        << " sve_compiled_enabled=" << (sve_compiled_enabled ? 1 : 0)
        << " sve2_compiled_enabled=" << (sve2_compiled_enabled ? 1 : 0)
        << " x86_avx512_compiled_enabled=" << (x86_avx512_compiled_enabled ? 1 : 0)
        << " attention_path_paged=" << req->attention_path_paged
        << " attention_path_standard=" << req->attention_path_standard
        << " attention_path_portable_flash=" << req->attention_path_portable_flash
        << " attention_path_native_flash=" << req->attention_path_native_flash
        << " attention_path_hal=" << req->attention_path_hal << " effective_attention_path=" << effective_attention_path
        << " flash_attention_headseq_prefill_calls=" << req->flash_attention_headseq_prefill_calls
        << " flash_attention_native_decode_calls=" << req->flash_attention_native_decode_calls
        << " flash_attention_reference_calls=" << req->flash_attention_reference_calls
        << " flash_attention_non_avx512_tiled_calls=" << req->flash_attention_non_avx512_tiled_calls
        << " flash_attention_avx512_tiled_calls=" << req->flash_attention_avx512_tiled_calls
        << " flash_attention_last_nth=" << req->flash_attention_last_nth
        << " flash_attention_last_active_threads=" << req->flash_attention_last_active_threads
        << " compute_flash_attention_reference_used=" << (req->flash_attention_reference_calls > 0 ? 1 : 0)
        << " moe_task_count=" << req->moe_task_count << " moe_rowblock_tasks=" << req->moe_rowblock_tasks
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
