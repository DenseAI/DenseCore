#include "inference.h"

#include "densecore/inference_types_internal.h"          // Shared internal types
#include "densecore/models/transformer_graph_builder.h"  // Strategy Pattern for graph building
#include "flash_attention.h"
#include "ggml-cpu.h"           // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"               // Required for ggml_tensor definition
#include "hardware_topology.h"  // For compute thread affinity
#include "matmul_backend.h"
#include "models/model_inference_policy.h"
#include "densecore/models/model_graph_capabilities.h"
#include "optimization_bridge.h"  // Runtime SIMD dispatch

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "dtype_utils.h"  // For GgmlTypeToDType
#include "kernels/hwy/hwy_kernels.h"
#include "kv_cache.h"  // Added for KV cache
#include "memory_pool.h"
#include "moe/moe_routing.h"
#include "quantization/int4_types.h"  // For TensorInt4
#include "qwen35_ssm_math.h"
#include "scheduler.h"
#include "simd_ops.h"

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif

#ifndef DENSECORE_DEFAULT_PRECOMPUTED_ROPE
#define DENSECORE_DEFAULT_PRECOMPUTED_ROPE 1
#endif

#ifndef DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
#define DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM 0
#endif

#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

namespace {
constexpr const char* kGemma4RouterScaleKey = "gemma4.router.scale";
constexpr const char* kGemma4RouterPerExpertScaleKey = "gemma4.router.per_expert_scale";
constexpr const char* kGemma4PreMoeNormKey = "gemma4.pre_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostSharedNormKey = "gemma4.post_feedforward_layernorm_1.weight";
constexpr const char* kGemma4PostMoeNormKey = "gemma4.post_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostFfnNormKey = "gemma4.post_feedforward_layernorm.weight";
std::atomic<uint64_t> g_moe_graph_wiring_debug_counter{0};

enum class MoEWiringReasonCode : int {
    Wired = 0,
    ModelHasNoMoE = 1,
    LayerFlagFalse = 2,
    MissingMoeGate = 3,
    NoExperts = 4,
    DenseReplaceGate = 5,
    LayerFlagMismatch = 6,
};

bool IsMoEWiringDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_MOE_WIRING_DEBUG");
        if (!env || env[0] == '\0') {
            env = std::getenv("DENSECORE_DEBUG_MOE_WIRING");
        }
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static struct ggml_tensor* BuildAttentionMaskTensor(struct ggml_context* ctx, int n_total_tokens, int n_queries,
                                                    int n_past, int sliding_window, int n_padded = -1) {
    if (!ctx || n_total_tokens <= 0 || n_queries <= 0) {
        return nullptr;
    }
    if (n_padded < n_queries) {
        n_padded = n_queries;
    }

    struct ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_total_tokens, n_padded, 1, 1);
    if (!mask || !mask->data) {
        return mask;
    }

    int sink_tokens = 0;
    if (const char* env = std::getenv("DENSECORE_KV_SINK_TOKENS"); env && env[0] != '\0') {
        sink_tokens = std::max(0, std::atoi(env));
    } else if (const char* env = std::getenv("DENSECORE_SINK_TOKENS"); env && env[0] != '\0') {
        sink_tokens = std::max(0, std::atoi(env));
    }
    const int sink_kept = std::clamp(sink_tokens, 0, n_past);
    const int tail_start = (sliding_window >= 0) ? std::max(sink_kept, n_past - std::max(0, sliding_window)) : n_past;
    const int history_kept = (sliding_window >= 0) ? (sink_kept + std::max(0, n_past - tail_start)) : n_past;
    auto map_retained_history_index = [&](int retained_index) {
        if (retained_index < sink_kept) {
            return retained_index;
        }
        return tail_start + (retained_index - sink_kept);
    };

    float* mask_data = reinterpret_cast<float*>(mask->data);
    for (int q = 0; q < n_padded; ++q) {
        const bool padded_query = q >= n_queries;
        const int query_pos = n_past + q;
        for (int k = 0; k < n_total_tokens; ++k) {
            const int idx = k + q * n_total_tokens;
            if (padded_query) {
                mask_data[idx] = 0.0f;
                continue;
            }

            const int key_pos = (k < history_kept) ? map_retained_history_index(k) : (n_past + (k - history_kept));
            bool allow = key_pos <= query_pos;
            if (allow && sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                allow = false;
            }
            mask_data[idx] = allow ? 0.0f : -INFINITY;
        }
    }

    return mask;
}

static bool ShouldBuildExplicitStandardAttentionMask(int n_queries, int sliding_window) {
    // Decode with a sliding-window still needs an explicit mask even when N == 1.
    // Otherwise the standard fallback path attends to stale history that Gemma4
    // sliding layers are supposed to exclude.
    return n_queries > 1 || sliding_window >= 0;
}

static const InferenceConfig& ResolveInferenceConfig(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->config) {
        return *batch->deps->config;
    }
    return InferenceConfig::Instance();
}

static densecore::HardwareTopology& ResolveHardwareTopology(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->hardware_topology) {
        return *batch->deps->hardware_topology;
    }
    return densecore::HardwareTopology::GetInstance();
}

static densecore::BackendRegistry& ResolveBackendRegistry(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->backend_registry) {
        return *batch->deps->backend_registry;
    }
    return densecore::BackendRegistry::Instance();
}

static densecore::DeviceType ResolvePreferredDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredMatmulDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_matmul_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredAttentionDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_attention_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredNormDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_norm_device;
    }
    return densecore::DeviceType::CPU;
}

static bool IsMixedRoutingEnabled(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->mixed_operation_routing;
    }
    return false;
}

static bool IsVerboseGraphBuildLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_VERBOSE_GRAPH_BUILD");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugInferenceStatsEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_INFERENCE_STATS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugGemvSelectionEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMV_SELECTION");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulDispatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_DISPATCH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Dispatch path labels for instrumentation
static const char* MatmulWeightTypeLabel(ggml_type wtype, bool is_packed_int4, bool is_packed_fp8) {
    if (is_packed_int4) return "PACKED_INT4";
    if (is_packed_fp8) return "PACKED_FP8";
    if (ggml_is_quantized(wtype)) return "GGML_QUANT";
    if (wtype == GGML_TYPE_F32) return "FLOAT_F32";
    if (wtype == GGML_TYPE_F16) return "FLOAT_F16";
    if (wtype == GGML_TYPE_BF16) return "FLOAT_BF16";
    return "UNKNOWN";
}

static const char* DetectedISATier() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

static void LogMatmulDispatch(const char* weight_name, const char* weight_type_label, int M, int N, int K,
                              const char* path_label, const char* fallback_reason = nullptr) {
    if (!IsDebugMatmulDispatchEnabled()) return;
    if (fallback_reason) {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s fallback=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label,
                fallback_reason);
    } else {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label);
    }
}

static bool IsHybridSSMQkvWeightName(const char* weight_name) {
    if (!weight_name || weight_name[0] == '\0') {
        return false;
    }
    return std::strstr(weight_name, "attn_qkv.weight") != nullptr ||
           std::strstr(weight_name, "linear_attn.in_proj_qkv.weight") != nullptr;
}

static bool IsDebugSSMQkvReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_QKV_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugSSMProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAttentionProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void ValidateAttentionProjectionShape3D(const struct ggml_tensor* tensor, const char* tensor_name, int layer_idx,
                                               int ne0, int ne1, int ne2, int projected_dim, int n_heads) {
    if (!tensor) {
        throw densecore::InvalidArgumentException("Missing attention projection tensor for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx));
    }
    if (projected_dim <= 0 || n_heads <= 0 || ne2 <= 0) {
        throw densecore::InvalidArgumentException(
            "Invalid attention reshape parameters for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": projected_dim=" + std::to_string(projected_dim) +
            " n_heads=" + std::to_string(n_heads) + " n_tokens=" + std::to_string(ne2));
    }
    if ((projected_dim % n_heads) != 0) {
        throw densecore::InvalidArgumentException("Attention projection dimension is not divisible by head count for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx) + ": dim=" + std::to_string(projected_dim) +
                                                  " n_heads=" + std::to_string(n_heads));
    }
    const int64_t expected = static_cast<int64_t>(ne0) * static_cast<int64_t>(ne1) * static_cast<int64_t>(ne2);
    const int64_t actual = ggml_nelements(tensor);
    if (actual != expected) {
        throw densecore::InvalidArgumentException(
            "Attention reshape contract mismatch for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": tensor=[" + std::to_string(tensor->ne[0]) + "," +
            std::to_string(tensor->ne[1]) + "," + std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) +
            "] projected_dim=" + std::to_string(projected_dim) + " n_heads=" + std::to_string(n_heads) + " target=[" +
            std::to_string(ne0) + "," + std::to_string(ne1) + "," + std::to_string(ne2) +
            "] actual_nelements=" + std::to_string(actual) + " expected_nelements=" + std::to_string(expected));
    }
}

static bool IsDebugFinalProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static int GetDebugDisableFastAttentionFromLayer() {
    static const int layer = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_DISABLE_FAST_ATTN_FROM_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    return layer;
}

static bool IsDebugHiddenSnapshotEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugGemma4SharedKVEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugFfnProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugSSMCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ShouldForcePlainGgmlForHybridSSMQkv_Internal() {
    // Explicit user override: always respected
    const char* env = std::getenv("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML");
    if (env && env[0] != '\0') {
        return std::strcmp(env, "0") != 0;
    }
    // Keep the optimized DenseCore paths enabled by default. The legacy
    // conservative default routed hybrid SSM QKV back to plain GGML, which is
    // slower on C4A-class ARM CPUs and can regress Qwen3.5 numerics. Retain an
    // env escape hatch for diagnostics.
    const char* conservative = std::getenv("DENSECORE_HYBRID_SSM_QKV_FORCE_CONSERVATIVE");
    if (conservative && conservative[0] != '\0') {
        return std::strcmp(conservative, "0") != 0;
    }
    return false;
}

static std::atomic<bool> g_hybrid_ssm_qkv_force_ggml_cached{false};
static bool g_hybrid_ssm_qkv_force_ggml_val = false;
static std::mutex g_hybrid_ssm_qkv_force_ggml_mutex;

static bool ShouldForcePlainGgmlForHybridSSMQkv() {
    if (!g_hybrid_ssm_qkv_force_ggml_cached.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_hybrid_ssm_qkv_force_ggml_mutex);
        if (!g_hybrid_ssm_qkv_force_ggml_cached.load(std::memory_order_relaxed)) {
            g_hybrid_ssm_qkv_force_ggml_val = ShouldForcePlainGgmlForHybridSSMQkv_Internal();
            g_hybrid_ssm_qkv_force_ggml_cached.store(true, std::memory_order_release);
        }
    }
    return g_hybrid_ssm_qkv_force_ggml_val;
}

static void LogHybridSSMQkvDispatch(const char* weight_name, ggml_type weight_type, int M, int N, int K,
                                    const char* chosen_path, bool used_batched_quant_nrc, bool used_native_q4k_vecdot,
                                    bool used_direct_int4_fastpath) {
    if (!IsDebugMatmulDispatchEnabled() || !IsHybridSSMQkvWeightName(weight_name)) {
        return;
    }
    fprintf(stderr,
            "[SSM_QKV_DISPATCH] w=%s ggml_type=%s M=%d N=%d K=%d path=%s batched_quant_nrc=%d native_q4k_vecdot=%d "
            "direct_int4_fastpath=%d\n",
            weight_name ? weight_name : "(unnamed)", ggml_type_name(weight_type), M, N, K,
            chosen_path ? chosen_path : "unknown", used_batched_quant_nrc ? 1 : 0, used_native_q4k_vecdot ? 1 : 0,
            used_direct_int4_fastpath ? 1 : 0);
}

// Env-tunable thresholds declared here, defined after ParsePositiveEnvInt.
static int GetBatchedMinM();
static int GetBatchedMinN();
static int GetBatchedMinK();

static void LogMatmulPathOnce(const char* path) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::atomic<bool> logged_ggml_mul_mat{false};
    static std::atomic<bool> logged_gemv_batched_quant_nrc{false};
    std::atomic<bool>* once_flag = nullptr;

    if (std::strcmp(path, "ggml_mul_mat") == 0) {
        once_flag = &logged_ggml_mul_mat;
    } else if (std::strcmp(path, "gemv_batched_quant_nrc") == 0) {
        once_flag = &logged_gemv_batched_quant_nrc;
    } else {
        return;
    }

    bool expected = false;
    if (once_flag->compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        fprintf(stderr, "[DenseCore][MatmulPath] %s\n", path);
    }
}

[[maybe_unused]] static void LogMatmulValidationOnce(const char* path, bool ok, float max_abs_diff) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::mutex mu;
    static std::unordered_map<std::string, bool> logged;
    std::lock_guard<std::mutex> lock(mu);
    if (logged[path]) {
        return;
    }
    logged[path] = true;
    fprintf(stderr, "[DenseCore][MatmulValidate] %s status=%s max_abs_diff=%.8f\n", path, ok ? "pass" : "fail",
            static_cast<double>(max_abs_diff));
}

enum class RuntimeToggleMode { Off = 0, Auto = 1, On = 2 };

[[maybe_unused]] static RuntimeToggleMode GetArmQ4KNativeVecDotMode();

static bool ShouldUseArmNativeQ4KVecDotValidated(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                                 const char* weight_name, const void* sample_row_ptr,
                                                 const void* sample_quant_input, const float* sample_input_f32,
                                                 int N) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (weight_type != GGML_TYPE_Q4_K) {
        return type_traits_cpu && type_traits_cpu->vec_dot;
    }
    if (!type_traits_cpu || !type_traits_cpu->vec_dot || !sample_row_ptr || !sample_quant_input || !sample_input_f32 ||
        N <= 0) {
        return false;
    }

    const RuntimeToggleMode mode = GetArmQ4KNativeVecDotMode();
    if (mode == RuntimeToggleMode::Off) {
        return false;
    }
    if (mode == RuntimeToggleMode::On) {
        return true;
    }

    const bool is_hybrid_ssm_qkv = IsHybridSSMQkvWeightName(weight_name);
    if (is_hybrid_ssm_qkv && ShouldForcePlainGgmlForHybridSSMQkv()) {
        return false;  // Always fail-closed for hybrid SSM QKV on ARM unless specifically opted out
    }

    // Multi-sample validation: test multiple representative weight rows
    static std::atomic<int> state{0};
    int current = state.load(std::memory_order_acquire);
    if (current != 0) {
        return current == 1;
    }

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    current = state.load(std::memory_order_relaxed);
    if (current == 0) {
        const auto* type_traits = ggml_get_type_traits(weight_type);
        bool ok = false;
        float global_max_abs_diff = 0.0f;
        if (type_traits && type_traits->to_float) {
            thread_local std::vector<float> dequant_buffer;
            dequant_buffer.resize(static_cast<size_t>(N));
            const size_t row_stride = ggml_row_size(weight_type, N);

            // Validate at least kMinValidationRows representative rows.
            // Rows are sampled at the base pointer (row 0) and at evenly
            // spaced offsets to detect position-dependent accuracy loss.
            static constexpr int kMinValidationRows = 4;
            ok = true;
            for (int sample = 0; sample < kMinValidationRows; ++sample) {
                const void* row_ptr =
                    reinterpret_cast<const char*>(sample_row_ptr) + static_cast<size_t>(sample) * row_stride;

                float native_sum = 0.0f;
                type_traits_cpu->vec_dot(N, &native_sum, 0, row_ptr, 0, sample_quant_input, 0, 1);

                type_traits->to_float(row_ptr, dequant_buffer.data(), N);
                float ref_sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    ref_sum += dequant_buffer[static_cast<size_t>(i)] * sample_input_f32[i];
                }

                const float diff = std::fabs(native_sum - ref_sum);
                global_max_abs_diff = std::max(global_max_abs_diff, diff);

                // Tighter tolerance than before (was 5e-4 relative, now 2e-4).
                // Fail closed: any single row exceeding tolerance disables the path.
                const float tol = std::max(1e-3f, 2e-4f * std::fabs(ref_sum));
                if (!std::isfinite(native_sum) || diff > tol) {
                    ok = false;
                    break;
                }
            }
        }

        state.store(ok ? 1 : 2, std::memory_order_release);
        LogMatmulValidationOnce("arm_q4k_native_vecdot", ok, global_max_abs_diff);
        current = ok ? 1 : 2;
    }

    return current == 1;
#else
    (void)weight_type;
    (void)weight_name;
    (void)sample_row_ptr;
    (void)sample_quant_input;
    (void)sample_input_f32;
    (void)N;
    return type_traits_cpu && type_traits_cpu->vec_dot;
#endif
}


static bool IsCustomGemvDisabled() {
    static const bool disabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_DISABLE_CUSTOM_GEMV");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const char* mode_env = std::getenv("DENSECORE_CUSTOM_GEMV_MODE");
        if (mode_env && mode_env[0] != '\0') {
            std::string mode(mode_env);
            for (char& c : mode) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (mode == "on" || mode == "1" || mode == "true" || mode == "force") {
                return false;
            }
            if (mode == "off" || mode == "0" || mode == "false") {
                return true;
            }
        }

        // Auto mode: keep custom GEMV enabled unless architecture-specific
        // logic in smart_mul_mat selects native GGML GEMV for better throughput.
        return false;
    }();
    return disabled;
}

static bool IsPackedInt4CustomDisabled() {
    static const bool disabled = []() {
        const char* env = std::getenv("DENSECORE_DISABLE_PACKED_INT4_CUSTOM");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return disabled;
}

// Legacy opt-in (kept for backward compat, but batched quant is now always-on by default)
static bool IsSmallBatchQuantGemmEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_SMALL_BATCH_GEMV_QUANT");
        if (!env || env[0] == '\0') {
            return true;
        }
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0 ||
            std::strcmp(env, "off") == 0 || std::strcmp(env, "OFF") == 0 || std::strcmp(env, "no") == 0 ||
            std::strcmp(env, "NO") == 0) {
            return false;
        }
        return true;
    }();
    return enabled;
}

// Primary opt-out: DENSECORE_DISABLE_BATCHED_QUANT=1 disables the shared-quant nrc=M fast path.
// Batched quant is ON by default for all quant weights with M>=2.
static bool IsBatchedQuantDisabled() {
    static const bool disabled = []() {
        const char* env = std::getenv("DENSECORE_DISABLE_BATCHED_QUANT");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        // Honor legacy env as well
        return !IsSmallBatchQuantGemmEnabled();
    }();
    return disabled;
}

// ARM small-batch quantized matmul policy.
// Default to the DenseCore batched path on ARM and keep an opt-out for
// diagnostics. Falling back to generic GGML on this model family was both
// slower and numerically unstable on real prefill traffic.
static bool IsArmBatchedQuantSafeByDefault() {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const bool enabled = []() {
        const char* disable_env = std::getenv("DENSECORE_ARM_DISABLE_BATCHED_QUANT");
        if (disable_env && disable_env[0] != '\0' && std::strcmp(disable_env, "0") != 0) {
            return false;
        }
        const char* enable_env = std::getenv("DENSECORE_ARM_ENABLE_BATCHED_QUANT");
        if (enable_env && enable_env[0] != '\0') {
            return std::strcmp(enable_env, "0") != 0;
        }
        return true;
    }();
    return enabled;
#else
    return true;
#endif
}

// Enable nrc-batched quant path by default and rely on per-type runtime checks
// (vec_dot nrows >= M) before dispatching to the fast path.
// DENSECORE_ENABLE_QUANT_NRC_BATCH=0 can be used to force-disable it.
static bool IsQuantNrcBatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_ENABLE_QUANT_NRC_BATCH");
        if (env && env[0] != '\0') {
            return std::strcmp(env, "0") != 0;
        }
        return true;
    }();
    return enabled;
}

// Deprecated path: keep disabled until a correctness/perf-positive
// implementation is available.
static bool IsQ4KTrueBatchedKernelEnabled() {
    return false;
}

static bool IsQ4KTrueBatchedAvx2Enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_ENABLE_Q4K_BATCHED_AVX2");
        if (!env || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static inline void SpinPause(int spin_count) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if ((spin_count & 0x3F) != 0) {
        _mm_pause();
        return;
    }
#elif defined(__aarch64__)
    if ((spin_count & 0x3F) != 0) {
        asm volatile("yield");
        return;
    }
#endif
    std::this_thread::yield();
}

static uint64_t ComputeGemvBatchedQuantStamp(const BatchSpec* batch, int M, int slot_id, const void* src_data_ptr,
                                             const void* weight_data_ptr) {
    // FNV-1a over stable per-op metadata to avoid stale-buffer reuse when
    // slot_id repeats across different batched inputs.
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ull;
    };

    mix(static_cast<uint64_t>(static_cast<uint32_t>(slot_id)));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(M)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_data_ptr)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_data_ptr)));
    if (batch) {
        mix(static_cast<uint64_t>(batch->tokens.size()));
        const int n_pos = std::min(M, static_cast<int>(batch->pos.size()));
        for (int i = 0; i < n_pos; ++i) {
            mix(static_cast<uint64_t>(static_cast<uint32_t>(batch->pos[static_cast<size_t>(i)])));
        }
    }

    // Zero is the reset sentinel for stamps.
    return hash == 0 ? 1 : hash;
}

enum class DecodePagedAttentionMode { Off = 0, Auto = 1, On = 2 };

struct DecodePagedAttentionPolicy {
    DecodePagedAttentionMode mode = DecodePagedAttentionMode::Auto;
    int min_context_tokens = 256;
    int min_batched_context_tokens = 256;
    int min_head_dim = 64;
    int min_heads = 8;
    bool allow_quantized_auto = false;
    bool debug_log = false;
};

static int ParsePositiveEnvInt(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static int ParseIntEnv(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        return default_value;
    }
    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

struct KVRetentionPolicy {
    int sliding_window = -1;
    int sink_tokens = 0;
    bool enabled = false;
};

struct KVRetentionSpan {
    int history_kept = 0;
    int sink_kept = 0;
    int tail_start = 0;
};

static KVRetentionPolicy LoadKVRetentionPolicy() {
    KVRetentionPolicy policy;
    policy.sliding_window =
        ParseIntEnv("DENSECORE_KV_SLIDING_WINDOW", ParseIntEnv("DENSECORE_SLIDING_WINDOW_SIZE", -1));
    policy.sink_tokens = ParseIntEnv("DENSECORE_KV_SINK_TOKENS", ParseIntEnv("DENSECORE_SINK_TOKENS", 0));

    if (policy.sliding_window < 0) {
        policy.sliding_window = -1;
    }
    if (policy.sink_tokens < 0) {
        policy.sink_tokens = 0;
    }
    policy.enabled = (policy.sliding_window >= 0);
    return policy;
}

static const KVRetentionPolicy& GetKVRetentionPolicy() {
    static const KVRetentionPolicy policy = LoadKVRetentionPolicy();
    return policy;
}

static KVRetentionSpan ComputeKVRetentionSpan(int n_past, const KVRetentionPolicy& policy) {
    KVRetentionSpan span;
    if (n_past <= 0) {
        return span;
    }

    if (!policy.enabled || policy.sliding_window < 0) {
        span.history_kept = n_past;
        // No sliding-window retention: preserved history must remain in
        // identity order [0, n_past) for all downstream KV readers.
        span.sink_kept = n_past;
        span.tail_start = n_past;
        return span;
    }

    span.sink_kept = std::clamp(policy.sink_tokens, 0, n_past);
    span.tail_start = std::max(span.sink_kept, n_past - std::max(0, policy.sliding_window));
    span.history_kept = span.sink_kept + std::max(0, n_past - span.tail_start);
    return span;
}

static int MapRetainedHistoryIndex(const KVRetentionSpan& span, int retained_index) {
    if (retained_index < span.sink_kept) {
        return retained_index;
    }
    return span.tail_start + (retained_index - span.sink_kept);
}

// Env-tunable thresholds for batched GEMM/GEMV routing
[[maybe_unused]] static int GetBatchedMinM() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_M", 2);
    return val;
}

[[maybe_unused]] static int GetBatchedMinN() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_N", 32);
    return val;
}

[[maybe_unused]] static int GetBatchedMinK() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_K", 32);
    return val;
}

[[maybe_unused]] static RuntimeToggleMode GetArmQ4KNativeVecDotMode() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // Legacy env: ALLOW=0 → disable (Off), ALLOW=1/non-zero → enable (On).
    // NOTE(fix): prior code had On/Off swapped here, causing ALLOW=0 to
    // *enable* the native path. This was the primary root cause of
    // "finite but wrong" ARM outputs on hybrid-SSM QKV projections.
    const char* legacy = std::getenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT");
    if (legacy && legacy[0] != '\0') {
        if (std::strcmp(legacy, "0") == 0 || std::strcmp(legacy, "false") == 0 || std::strcmp(legacy, "FALSE") == 0) {
            return RuntimeToggleMode::Off;
        }
        return RuntimeToggleMode::On;
    }

    const char* env = std::getenv("DENSECORE_ARM_Q4K_NATIVE_VECDOT_MODE");
    if (!env || env[0] == '\0') {
        return RuntimeToggleMode::Auto;
    }
    if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0) {
        return RuntimeToggleMode::Off;
    }
    if (std::strcmp(env, "on") == 0 || std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "TRUE") == 0 || std::strcmp(env, "force") == 0) {
        return RuntimeToggleMode::On;
    }
    return RuntimeToggleMode::Auto;
#else
    return RuntimeToggleMode::On;
#endif
}

[[maybe_unused]] static RuntimeToggleMode GetArmInt4DirectFastPathMode() {
#if defined(__aarch64__) || defined(_M_ARM64)
    const char* env = std::getenv("DENSECORE_ARM_INT4_DIRECT_FASTPATH_MODE");
    if (!env || env[0] == '\0') {
        return RuntimeToggleMode::Auto;
    }
    if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0) {
        return RuntimeToggleMode::Off;
    }
    if (std::strcmp(env, "on") == 0 || std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
        std::strcmp(env, "TRUE") == 0 || std::strcmp(env, "force") == 0) {
        return RuntimeToggleMode::On;
    }
    return RuntimeToggleMode::Auto;
#else
    return RuntimeToggleMode::On;
#endif
}

static bool ParseTruthyEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0 ||
           std::strcmp(env, "yes") == 0 || std::strcmp(env, "YES") == 0 || std::strcmp(env, "on") == 0 ||
           std::strcmp(env, "ON") == 0;
}

static std::string AsciiLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static RuntimeToggleMode ParseRuntimeToggleMode(const char* name, RuntimeToggleMode default_mode) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_mode;
    }

    const std::string mode = AsciiLower(env);
    if (mode == "off" || mode == "0" || mode == "false" || mode == "no") {
        return RuntimeToggleMode::Off;
    }
    if (mode == "on" || mode == "1" || mode == "true" || mode == "yes" || mode == "force") {
        return RuntimeToggleMode::On;
    }
    if (mode == "auto") {
        return RuntimeToggleMode::Auto;
    }
    return default_mode;
}

static int ResolveTaskCount(const BatchSpec* batch, int work_items) {
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    if (work_items > 0) {
        n_tasks = std::min(n_tasks, work_items);
    }
    return std::max(1, n_tasks);
}

static densecore::simd::SimdLevel GetRuntimeSimdLevel() {
    static const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return level;
}

static DecodePagedAttentionMode ParseDecodePagedAttentionMode() {
    if (ParseTruthyEnv("DENSECORE_FORCE_PAGED_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    const char* mode_env = std::getenv("DENSECORE_PAGED_ATTN_DECODE_MODE");
    if (mode_env && mode_env[0] != '\0') {
        const std::string mode = AsciiLower(mode_env);
        if (mode == "off" || mode == "0" || mode == "false") {
            return DecodePagedAttentionMode::Off;
        }
        if (mode == "on" || mode == "1" || mode == "true" || mode == "force") {
            return DecodePagedAttentionMode::On;
        }
        if (mode == "auto") {
            return DecodePagedAttentionMode::Auto;
        }
    }

    // Backward-compatible env fallback.
    if (ParseTruthyEnv("DENSECORE_ENABLE_PAGED_ATTN_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    // Default: always use paged decode attention when candidate conditions are met.
    // The materialized KV path (ggml_pad) is strictly worse for memory bandwidth.
    return DecodePagedAttentionMode::On;
}

static DecodePagedAttentionPolicy LoadDecodePagedAttentionPolicy() {
    DecodePagedAttentionPolicy policy;
    policy.mode = ParseDecodePagedAttentionMode();
    const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
    const bool has_avx2_or_better = densecore::simd::HasX86Avx2OrBetter(simd);
    const bool has_avx512_or_better = densecore::simd::HasX86Avx512OrBetter(simd);
    const bool is_arm = densecore::simd::IsArmFamily(simd);
    const int default_min_context = has_avx2_or_better ? 128 : (is_arm ? 64 : 256);
    const int default_min_batched_context = (has_avx2_or_better && !has_avx512_or_better) ? 64 : default_min_context;
    const int legacy_min_context = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_CONTEXT", default_min_context);
    policy.min_context_tokens = ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_CONTEXT", legacy_min_context);
    const int legacy_min_batched_context =
        ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_BATCH_CONTEXT", default_min_batched_context);
    policy.min_batched_context_tokens =
        ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", legacy_min_batched_context);
    policy.min_head_dim = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEAD_DIM", 64);
    policy.min_heads = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEADS", 8);
    // Quantized KV cache (Q8_0, Q4_0) must use paged decode — the materialized
    // fallback path re-allocs the full KV history every step, costing far more
    // memory bandwidth than the paged path saves on dequant overhead.
    const bool legacy_allow_q8 = ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8", true);
    policy.allow_quantized_auto = ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED", legacy_allow_q8);
    policy.debug_log = ParseTruthyEnv("DENSECORE_DEBUG_PAGED_ATTN_DECODE", false);
    return policy;
}

bool IsPagedDecodeModeAlwaysOnImpl() {
    return ParseDecodePagedAttentionMode() == DecodePagedAttentionMode::On;
}

struct DecodeContextSummary {
    int min_context = 0;
    int max_context = 0;
    int avg_context = 0;
    bool valid = false;
};

[[maybe_unused]] static bool IsBatchedPagedDecodeEnabled() {
    static const bool enabled = ParseTruthyEnv("DENSECORE_ENABLE_BATCHED_PAGED_DECODE", false);
    return enabled;
}

bool IsDecodeOnlyBatchLayoutImpl(const BatchSpec& batch, int n_tokens_in_batch) {
    if (n_tokens_in_batch <= 0) {
        return false;
    }
    if (batch.num_seqs != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.tokens.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.seq_id.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.pos.size()) != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.block_tables.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.n_past.size()) != n_tokens_in_batch) {
        return false;
    }

    std::vector<uint8_t> seen(static_cast<size_t>(n_tokens_in_batch), 0);
    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch) {
            return false;
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(seq_idx)] = 1;
    }

    return true;
}

static DecodeContextSummary SummarizeDecodeContext(const BatchSpec& batch, int n_tokens_in_batch) {
    DecodeContextSummary summary;
    if (!IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch)) {
        return summary;
    }

    summary.min_context = std::numeric_limits<int>::max();
    summary.max_context = 0;
    int64_t sum_context = 0;
    std::vector<uint8_t> seen(static_cast<size_t>(batch.num_seqs), 0);

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
            return DecodeContextSummary{};
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return DecodeContextSummary{};
        }
        seen[static_cast<size_t>(seq_idx)] = 1;

        const int pos_i = batch.pos[static_cast<size_t>(i)];
        if (pos_i < 0) {
            return DecodeContextSummary{};
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodeContextSummary{};
        }
        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodeContextSummary{};
        }

        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (n_past_i < 0) {
            return DecodeContextSummary{};
        }

        const KVRetentionSpan retained = ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const int max_context_i = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len_i = std::max(1, std::min(retained.history_kept + 1, max_context_i));
        summary.min_context = std::min(summary.min_context, context_len_i);
        summary.max_context = std::max(summary.max_context, context_len_i);
        sum_context += context_len_i;
    }

    summary.avg_context = static_cast<int>((sum_context + n_tokens_in_batch - 1) / n_tokens_in_batch);
    summary.valid = true;
    return summary;
}

enum class DecodePagedFallbackReason : std::size_t {
    None = 0,
    NoCache = 1,
    GlmDsa = 2,
    NonDecodeOnlyLayout = 3,
    InvalidHeadConfig = 4,
    HeadDimMismatch = 5,
    InvalidSeqMapping = 6,
    MissingBlockTable = 7,
    InvalidBlockTableIndex = 8,
    InvalidBlockId = 9,
    InvalidContextSummary = 10,
    PolicyOff = 11,
    AutoQuantizedDisabled = 12,
    AutoMinHeads = 13,
    AutoMinHeadDim = 14,
    AutoContextShort = 15,
};

static_assert(static_cast<std::size_t>(DecodePagedFallbackReason::AutoContextShort) + 1 ==
                  kDecodePagedFallbackReasonCount,
              "Decode paged fallback count mismatch");

static const char* DecodePagedFallbackReasonName(DecodePagedFallbackReason reason) {
    switch (reason) {
    case DecodePagedFallbackReason::None: return "none";
    case DecodePagedFallbackReason::NoCache: return "no_cache";
    case DecodePagedFallbackReason::GlmDsa: return "glm_dsa";
    case DecodePagedFallbackReason::NonDecodeOnlyLayout: return "non_decode_layout";
    case DecodePagedFallbackReason::InvalidHeadConfig: return "invalid_head_config";
    case DecodePagedFallbackReason::HeadDimMismatch: return "head_dim_mismatch";
    case DecodePagedFallbackReason::InvalidSeqMapping: return "invalid_seq_mapping";
    case DecodePagedFallbackReason::MissingBlockTable: return "missing_block_table";
    case DecodePagedFallbackReason::InvalidBlockTableIndex: return "invalid_block_index";
    case DecodePagedFallbackReason::InvalidBlockId: return "invalid_block_id";
    case DecodePagedFallbackReason::InvalidContextSummary: return "invalid_context";
    case DecodePagedFallbackReason::PolicyOff: return "policy_off";
    case DecodePagedFallbackReason::AutoQuantizedDisabled: return "auto_quant_disabled";
    case DecodePagedFallbackReason::AutoMinHeads: return "auto_min_heads";
    case DecodePagedFallbackReason::AutoMinHeadDim: return "auto_min_head_dim";
    case DecodePagedFallbackReason::AutoContextShort: return "auto_context_short";
    default: return "unknown";
    }
}

const char* GetDecodePagedFallbackReasonNameImpl(std::size_t index) {
    if (index >= kDecodePagedFallbackReasonCount) {
        return "unknown";
    }
    return DecodePagedFallbackReasonName(static_cast<DecodePagedFallbackReason>(index));
}

static DecodePagedFallbackReason DiagnosePagedDecodeCandidateFailure(const PagedKVCache* cache, const BatchSpec& batch,
                                                                     int n_tokens_in_batch, int n_head, int n_head_kv,
                                                                     int head_dim_q, int head_dim_kv) {
    if (!cache) {
        return DecodePagedFallbackReason::NoCache;
    }
    if (!IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch)) {
        return DecodePagedFallbackReason::NonDecodeOnlyLayout;
    }
    if (n_head_kv <= 0 || n_head <= 0) {
        return DecodePagedFallbackReason::InvalidHeadConfig;
    }
    if (n_head % n_head_kv != 0) {
        return DecodePagedFallbackReason::InvalidHeadConfig;
    }
    if (head_dim_q != head_dim_kv) {
        return DecodePagedFallbackReason::HeadDimMismatch;
    }

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch) {
            return DecodePagedFallbackReason::InvalidSeqMapping;
        }
        const int pos_i = batch.pos[static_cast<size_t>(i)];
        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (pos_i < 0 || n_past_i < 0) {
            return DecodePagedFallbackReason::InvalidSeqMapping;
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodePagedFallbackReason::MissingBlockTable;
        }

        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodePagedFallbackReason::InvalidBlockTableIndex;
        }
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= cache->max_blocks) {
            return DecodePagedFallbackReason::InvalidBlockId;
        }
    }

    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        return DecodePagedFallbackReason::InvalidContextSummary;
    }

    return DecodePagedFallbackReason::None;
}

bool IsPagedDecodeCandidateImpl(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                int n_head_kv, int head_dim_q, int head_dim_kv) {
    return DiagnosePagedDecodeCandidateFailure(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q,
                                               head_dim_kv) == DecodePagedFallbackReason::None;
}

struct DecodePagedDecision {
    bool candidate = false;
    bool requested = false;
    DecodePagedFallbackReason reason = DecodePagedFallbackReason::None;
};

struct BasePagedDecodeExecutionDecision {
    DecodePagedDecision paged_decode_decision{};
    bool paged_decode_supported = false;
    bool decode_only_batch = false;
    bool force_batched_decode_path = false;
    bool use_paged_decode_attention = false;
    int n_past_val = 0;
};

#ifdef DENSECORE_TEST_BUILD
static std::atomic<int> g_test_force_flash_attention_disabled{0};
#endif

static DecodePagedDecision EvaluatePagedDecodeDecision(const DecodePagedAttentionPolicy& policy,
                                                       const TransformerModel* model, const PagedKVCache* cache,
                                                       const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                                       int n_head_kv, int head_dim_q, int head_dim_kv) {
    DecodePagedDecision decision;
    if (model && model->arch_flags.is_glm_dsa) {
        decision.reason = DecodePagedFallbackReason::GlmDsa;
        return decision;
    }

    decision.reason = DiagnosePagedDecodeCandidateFailure(cache, batch, n_tokens_in_batch, n_head, n_head_kv,
                                                          head_dim_q, head_dim_kv);
    if (decision.reason != DecodePagedFallbackReason::None) {
        return decision;
    }
    decision.candidate = true;

    if (policy.mode == DecodePagedAttentionMode::Off) {
        decision.reason = DecodePagedFallbackReason::PolicyOff;
        return decision;
    }

    if (policy.mode == DecodePagedAttentionMode::On) {
        decision.requested = true;
        return decision;
    }

    // Auto mode: enable only when context is long enough to amortize callback/setup overhead.
    if (!policy.allow_quantized_auto && ggml_is_quantized(cache->cache_type)) {
        decision.reason = DecodePagedFallbackReason::AutoQuantizedDisabled;
        return decision;
    }
    if (n_head < policy.min_heads) {
        decision.reason = DecodePagedFallbackReason::AutoMinHeads;
        return decision;
    }
    if (head_dim_q < policy.min_head_dim) {
        decision.reason = DecodePagedFallbackReason::AutoMinHeadDim;
        return decision;
    }
    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        decision.reason = DecodePagedFallbackReason::InvalidContextSummary;
        return decision;
    }
    // Use the shortest context in the decode micro-batch as representative.
    // This prevents Auto mode from enabling paged decode only because a subset
    // of sequences has long history.
    const int context_len = context_summary.min_context;
    const int effective_min_context =
        (n_tokens_in_batch > 1) ? policy.min_batched_context_tokens : policy.min_context_tokens;
    if (context_len < effective_min_context) {
        decision.reason = DecodePagedFallbackReason::AutoContextShort;
        return decision;
    }

    decision.requested = true;
    return decision;
}

static BasePagedDecodeExecutionDecision
ResolveBasePagedDecodeExecutionDecision(const DecodePagedAttentionPolicy& policy, const TransformerModel* model,
                                        const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch,
                                        int n_head, int n_head_kv, int head_dim_q, int head_dim_kv) {
    BasePagedDecodeExecutionDecision decision;
    const bool use_cache = (cache != nullptr);
    if (use_cache && batch.num_seqs > 0 && !batch.n_past.empty()) {
        for (int n_past_i : batch.n_past) {
            if (model && model->arch_flags.is_gemma4) {
                decision.n_past_val = std::max(decision.n_past_val, std::max(0, n_past_i));
            } else {
                const KVRetentionPolicy& retention_policy = GetKVRetentionPolicy();
                decision.n_past_val =
                    std::max(decision.n_past_val, ComputeKVRetentionSpan(n_past_i, retention_policy).history_kept);
            }
        }
    }

    decision.paged_decode_decision = EvaluatePagedDecodeDecision(policy, model, cache, batch, n_tokens_in_batch, n_head,
                                                                 n_head_kv, head_dim_q, head_dim_kv);
    decision.paged_decode_supported = densecore::models::SupportsPagedDecodeAttention(model);
    decision.decode_only_batch = IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch);
    // For decode-only N>1 this preserves a policy-approved paged decode
    // choice across later local dispatch branches. It must not upgrade a
    // batch from mere shape/cache candidacy into paged decode when policy
    // explicitly disabled the path or auto-mode declined it.
    decision.force_batched_decode_path = decision.paged_decode_supported && use_cache && decision.decode_only_batch &&
                                         n_tokens_in_batch > 1 && decision.paged_decode_decision.requested;
    decision.use_paged_decode_attention = decision.paged_decode_decision.requested && decision.paged_decode_supported;
    if (decision.force_batched_decode_path) {
        decision.use_paged_decode_attention = true;
    }
    return decision;
}

static int ResolveAttentionQueryBasePosition(const BatchSpec& batch) {
    if (batch.n_past.empty()) {
        return 0;
    }
    int max_n_past = 0;
    for (int n_past_i : batch.n_past) {
        max_n_past = std::max(max_n_past, n_past_i);
    }
    return max_n_past;
}

static bool IsFlashAttentionDisabled() {
#ifdef DENSECORE_TEST_BUILD
    if (g_test_force_flash_attention_disabled.load(std::memory_order_relaxed) != 0) {
        return true;
    }
#endif
    static const bool disabled = []() {
        if (ParseRuntimeToggleMode("DENSECORE_FLASH_ATTN_MODE", RuntimeToggleMode::Auto) == RuntimeToggleMode::Off) {
            return true;
        }
        const char* legacy_env = std::getenv("DENSECORE_DISABLE_FLASH_ATTN");
        return legacy_env && legacy_env[0] != '\0' && std::strcmp(legacy_env, "0") != 0;
    }();
    return disabled;
}

static bool IsPagedAttentionHwyEnabled() {
    static const bool enabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_PAGED_ATTN_USE_HWY");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const RuntimeToggleMode mode = ParseRuntimeToggleMode("DENSECORE_PAGED_ATTN_HWY_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        // Auto mode: use Highway on SIMD-capable hosts and scalar on true
        // scalar-only CPUs.
        return GetRuntimeSimdLevel() != densecore::simd::SimdLevel::NONE;
    }();
    return enabled;
}

static bool IsDebugPagedAttentionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugPagedAttentionEagerReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAttentionCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAddRmsNormReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ShouldRunAddRmsNormReferenceProbe(int layer_idx) {
    if (!IsDebugAddRmsNormReferenceEnabled()) {
        return false;
    }
    int configured_layer = -1;
    if (const char* env = std::getenv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_LAYER")) {
        if (env[0] != '\0') {
            configured_layer = std::atoi(env);
        }
    }
    return configured_layer < 0 || configured_layer == layer_idx;
}

static bool IsDebugAttentionPostReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_POST_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugKvRoundTripEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_KV_ROUNDTRIP");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return 0.0f;
    }
    return model->gemma4_attention_logit_softcapping > 0.0f ? model->gemma4_attention_logit_softcapping : 50.0f;
}

static bool IsGemma4SharedKVSourceLayer(const TransformerModel* model, int layer_idx) {
    if (!model || !model->arch_flags.is_gemma4 || layer_idx < 0) {
        return false;
    }
    for (size_t i = 0; i < model->gemma4_layer_kv_source.size(); ++i) {
        if (static_cast<int>(i) > layer_idx && model->gemma4_layer_kv_source[i] == layer_idx) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunKvRoundTripProbe(int layer, bool is_k) {
    if (!IsDebugKvRoundTripEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_KV_ROUNDTRIP_LAYER", -1);
    static const int target_kind = ParseIntEnv("DENSECORE_DEBUG_KV_ROUNDTRIP_KIND", -1);  // -1 both, 0 V, 1 K
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_KV_ROUNDTRIP_MAX_CALLS", 1)};

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

static bool ShouldRunPagedAttentionReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_MAX_CALLS", 1)};

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

static bool ShouldRunPagedAttentionEagerReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionEagerReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_MAX_CALLS", 1)};

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

static bool ShouldRunAttentionCoreReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionCoreReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_MAX_CALLS", 1)};

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

static bool ShouldRunAttentionPostReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionPostReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_POST_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_ATTN_POST_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_POST_REFERENCE_MAX_CALLS", 1)};

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

static bool ShouldRunAttentionProjectionReferenceProbe(int layer) {
    if (!IsDebugAttentionProjectionReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_MAX_CALLS", 8)};

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

static bool ShouldRunFinalProjectionReferenceProbe() {
    if (!IsDebugFinalProjectionReferenceEnabled()) {
        return false;
    }

    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE_MAX_CALLS", 1)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunHiddenSnapshotProbe(int layer, const char* stage) {
    if (!IsDebugHiddenSnapshotEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_LAYER", -2);
    if (target_layer >= -1 && layer != target_layer) {
        return false;
    }

    static const std::string target_stage = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_STAGE");
        return (env && env[0] != '\0') ? std::string(env) : std::string();
    }();
    if (!target_stage.empty() && stage && target_stage != stage) {
        return false;
    }

    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_MAX_CALLS", 4)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunFfnProjectionReferenceProbe(int layer) {
    if (!IsDebugFfnProjectionReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_MAX_CALLS", 8)};
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
        const char* env = std::getenv("DENSECORE_DEBUG_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ShouldRunRmsNormReferenceProbe(int layer) {
    if (!IsDebugRmsNormReferenceEnabled()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_RMSNORM_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_RMSNORM_REFERENCE_MAX_CALLS", 8)};
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

static void DebugLogTensorFiniteStats(const char* tag, const struct ggml_tensor* tensor) {
    if (!tag || !tensor || !tensor->data || tensor->type != GGML_TYPE_F32) {
        return;
    }
    const float* data = reinterpret_cast<const float*>(tensor->data);
    const int n = ggml_nelements(tensor);
    if (n <= 0) {
        return;
    }

    int nan_ct = 0;
    int inf_ct = 0;
    int zero_ct = 0;
    int finite_ct = 0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++nan_ct;
            continue;
        }
        if (!std::isfinite(v)) {
            ++inf_ct;
            continue;
        }
        if (v == 0.0f) {
            ++zero_ct;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        ++finite_ct;
    }

    if (!std::isfinite(mn)) mn = 0.0f;
    if (!std::isfinite(mx)) mx = 0.0f;
    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
    std::fprintf(
        stderr,
        "[%s] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n", tag,
        static_cast<int>(tensor->type), static_cast<long>(tensor->ne[0]), static_cast<long>(tensor->ne[1]),
        static_cast<long>(tensor->ne[2]), n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
}

static void ComputeMatmulReferenceF32(const struct ggml_tensor* weight, const struct ggml_tensor* input,
                                      std::vector<float>* out) {
    constexpr int kMatmulReferenceMaxDequant = 16384;
    constexpr size_t kMatmulReferenceMaxQuantRowBytes = 1u << 16;
    if (!weight || !input || !out || !weight->data || !input->data || input->type != GGML_TYPE_F32) {
        out->clear();
        return;
    }
    const int M = static_cast<int>(input->ne[1]);
    if (M <= 0) {
        out->clear();
        return;
    }

    const int input_dim = static_cast<int>(input->ne[0]);
    const int ne0 = static_cast<int>(weight->ne[0]);
    const int ne1 = static_cast<int>(weight->ne[1]);
    const char* weight_base = reinterpret_cast<const char*>(weight->data);
    const char* input_base = reinterpret_cast<const char*>(input->data);

    if (weight->type == GGML_TYPE_F32 && input_dim == ne1 && ne0 > 0 && ne1 > 0) {
        out->assign(static_cast<size_t>(ne0) * static_cast<size_t>(M), 0.0f);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            for (int out_idx = 0; out_idx < ne0; ++out_idx) {
                float sum = 0.0f;
                for (int in_idx = 0; in_idx < ne1; ++in_idx) {
                    const char* weight_row =
                        weight_base + static_cast<size_t>(in_idx) * static_cast<size_t>(weight->nb[1]);
                    const float w =
                        *reinterpret_cast<const float*>(weight_row + static_cast<size_t>(out_idx) * weight->nb[0]);
                    const float x =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(in_idx) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(ne0) + static_cast<size_t>(out_idx)] = sum;
            }
        }
        return;
    }

    const int N = ne0;
    const int K = ne1;
    if (N <= 0 || K <= 0 || input_dim != N) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
    if (weight->type == GGML_TYPE_F32) {
        for (int k = 0; k < K; ++k) {
            const char* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                float sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const float w = *reinterpret_cast<const float*>(row_ptr + static_cast<size_t>(i) * weight->nb[0]);
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
            }
        }
        return;
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        if (input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
            quant_row_size <= kMatmulReferenceMaxQuantRowBytes) {
            std::vector<uint8_t> quant_rows(quant_row_size * static_cast<size_t>(M));
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                std::vector<float> gathered_input(static_cast<size_t>(N), 0.0f);
                for (int i = 0; i < N; ++i) {
                    gathered_input[static_cast<size_t>(i)] =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                }
                input_type_traits->from_float(gathered_input.data(),
                                              quant_rows.data() + quant_row_size * static_cast<size_t>(m),
                                              static_cast<int64_t>(N));
            }

            out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
            for (int k = 0; k < K; ++k) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
                for (int m = 0; m < M; ++m) {
                    float sum = 0.0f;
                    const void* q_ptr = quant_rows.data() + quant_row_size * static_cast<size_t>(m);
                    type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                    (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
                }
            }
            return;
        }
    }

    const auto* type_traits = ggml_get_type_traits(weight->type);
    if (!type_traits || !type_traits->to_float || N > kMatmulReferenceMaxDequant) {
        out->clear();
        return;
    }

    std::vector<float> dequant_row(static_cast<size_t>(N), 0.0f);
    for (int k = 0; k < K; ++k) {
        const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                sum += dequant_row[static_cast<size_t>(i)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
        }
    }
}

static bool IsDecodeProfileEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_PROFILE_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDecodeAttentionPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_LOG_DECODE_ATTENTION_PATH");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        return IsDecodeProfileEnabled() || IsDebugInferenceStatsEnabled();
    }();
    return enabled;
}

enum class DecodeAttentionPathKind : uint8_t {
    Paged = 0,
    Hal = 1,
    PortableCpuFlash = 2,
    NativeFlash = 3,
    Standard = 4,
};

struct DecodeAttentionPathCounters {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> paged{0};
    std::atomic<uint64_t> hal{0};
    std::atomic<uint64_t> portable_cpu_flash{0};
    std::atomic<uint64_t> native_flash{0};
    std::atomic<uint64_t> standard{0};
};

static DecodeAttentionPathCounters& GetDecodeAttentionPathCounters() {
    static DecodeAttentionPathCounters counters;
    return counters;
}

struct DecodeSharedQuantCounters {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> reused{0};
    std::atomic<uint64_t> tls{0};
};

static DecodeSharedQuantCounters& GetDecodeSharedQuantCounters() {
    static DecodeSharedQuantCounters counters;
    return counters;
}

static std::array<std::atomic<uint64_t>, kDecodePagedFallbackReasonCount>& GetDecodePagedFallbackCounters() {
    static std::array<std::atomic<uint64_t>, kDecodePagedFallbackReasonCount> counters{};
    return counters;
}

static void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int layer, int N) {
    if (N <= 0 || N > 4 || reason == DecodePagedFallbackReason::None) {
        return;
    }
    GetDecodePagedFallbackCounters()[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

static void RecordSharedQuantReuse(bool reused_shared_buffer) {
    DecodeSharedQuantCounters& counters = GetDecodeSharedQuantCounters();
    counters.total.fetch_add(1, std::memory_order_relaxed);
    if (reused_shared_buffer) {
        counters.reused.fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.tls.fetch_add(1, std::memory_order_relaxed);
    }
}

DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshotImpl() {
    DecodeRuntimeStatsSnapshot snapshot;

    const DecodeAttentionPathCounters& path = GetDecodeAttentionPathCounters();
    snapshot.path_total = path.total.load(std::memory_order_relaxed);
    snapshot.path_paged = path.paged.load(std::memory_order_relaxed);
    snapshot.path_hal = path.hal.load(std::memory_order_relaxed);
    snapshot.path_portable_cpu_flash = path.portable_cpu_flash.load(std::memory_order_relaxed);
    snapshot.path_native_flash = path.native_flash.load(std::memory_order_relaxed);
    snapshot.path_standard = path.standard.load(std::memory_order_relaxed);

    const auto& fallback = GetDecodePagedFallbackCounters();
    for (std::size_t i = 0; i < snapshot.paged_fallback_reasons.size(); ++i) {
        snapshot.paged_fallback_reasons[i] = fallback[i].load(std::memory_order_relaxed);
    }

    const DecodeSharedQuantCounters& quant = GetDecodeSharedQuantCounters();
    snapshot.shared_quant_total = quant.total.load(std::memory_order_relaxed);
    snapshot.shared_quant_reused = quant.reused.load(std::memory_order_relaxed);
    snapshot.shared_quant_tls = quant.tls.load(std::memory_order_relaxed);

    return snapshot;
}

static const char* DecodeAttentionPathName(DecodeAttentionPathKind kind) {
    switch (kind) {
    case DecodeAttentionPathKind::Paged: return "paged_decode";
    case DecodeAttentionPathKind::Hal: return "hal_flash";
    case DecodeAttentionPathKind::PortableCpuFlash: return "cpu_flash_hal";
    case DecodeAttentionPathKind::NativeFlash: return "flash";
    case DecodeAttentionPathKind::Standard: return "standard";
    default: return "unknown";
    }
}

static void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int layer, int N, int n_past_val, int n_head,
                                      int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                                      bool paged_candidate, bool paged_selected, bool portable_supported,
                                      bool offset_safe) {
    if (N <= 0) {
        return;
    }

    DecodeAttentionPathCounters& counters = GetDecodeAttentionPathCounters();
    const uint64_t total = counters.total.fetch_add(1, std::memory_order_relaxed) + 1;
    switch (kind) {
    case DecodeAttentionPathKind::Paged: counters.paged.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::Hal: counters.hal.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::PortableCpuFlash:
        counters.portable_cpu_flash.fetch_add(1, std::memory_order_relaxed);
        break;
    case DecodeAttentionPathKind::NativeFlash: counters.native_flash.fetch_add(1, std::memory_order_relaxed); break;
    case DecodeAttentionPathKind::Standard: counters.standard.fetch_add(1, std::memory_order_relaxed); break;
    }

    if (IsDebugInferenceStatsEnabled() && total <= 8) {
        std::cerr << "[DecodeAttentionDecision] path=" << DecodeAttentionPathName(kind) << " N=" << N
                  << " n_past=" << n_past_val << " n_head=" << n_head << " n_head_kv=" << n_head_kv
                  << " preferred_device=" << densecore::DeviceTypeName(preferred_device)
                  << " native_layout=" << (native_layout ? "1" : "0")
                  << " paged_candidate=" << (paged_candidate ? "1" : "0")
                  << " paged_selected=" << (paged_selected ? "1" : "0")
                  << " portable_supported=" << (portable_supported ? "1" : "0")
                  << " offset_safe=" << (offset_safe ? "1" : "0") << std::endl;
    }

    if (IsDecodeAttentionPathLoggingEnabled() && total % 100 == 0) {
        std::cerr << "[DecodeAttentionPathStats] total=" << total
                  << " paged=" << counters.paged.load(std::memory_order_relaxed)
                  << " hal=" << counters.hal.load(std::memory_order_relaxed)
                  << " cpu_flash=" << counters.portable_cpu_flash.load(std::memory_order_relaxed)
                  << " flash=" << counters.native_flash.load(std::memory_order_relaxed)
                  << " standard=" << counters.standard.load(std::memory_order_relaxed) << std::endl;
    }
}

static bool IsForceSafeGqaDecodeEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_FORCE_SAFE_GQA_DECODE");
        if (!env || env[0] == '\0') return true;
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsPrefillAttentionSkipContEnabled() {
    static const bool enabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_PREFILL_ATTN_SKIP_CONT");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        // Auto mode: keep OFF on AVX2 where regression was observed, enable on
        // AVX-512/AMX class x86 where memory bandwidth pressure is lower.
        return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel());
    }();
    return enabled;
}

static bool IsFlashAttentionForced() {
    static const bool forced = []() { return ParseTruthyEnv("DENSECORE_FORCE_FLASH_ATTN", false); }();
    return forced;
}

static bool IsFlashAttentionIsaSupported() {
    static const bool supported = []() { return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel()); }();
    return supported;
}

static bool IsPortableCpuFlashAttentionEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_PORTABLE_FLASH_ATTN_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }
        // Auto mode: enable portable flash attention on all platforms.
        // The prior ARM correctness issue was caused by K/V tensors not being
        // registered in the GGML graph's src[] dependency chain, which meant
        // their ggml_cont ops were never executed during graph compute.
        return true;
    }();
    return enabled;
}

static bool IsInt4SingleThreadingLayerEnabled() {
    static const bool enabled = []() {
        const char* legacy = std::getenv("DENSECORE_INT4_USE_BACKEND_THREADPOOL");
        if (legacy && legacy[0] != '\0') {
            return std::strcmp(legacy, "0") == 0;
        }
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_INT4_THREADING_MODE",
            DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsPrecomputedRoPEEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_ROPE_PRECOMPUTED_MODE",
                                   DENSECORE_DEFAULT_PRECOMPUTED_ROPE ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsFusedResidualRmsNormEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_FUSED_RESIDUAL_RMSNORM_MODE",
            DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsFusedQKVEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_FUSED_QKV_MODE", DENSECORE_DEFAULT_FUSED_QKV ? RuntimeToggleMode::Auto : RuntimeToggleMode::Off);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
        return densecore::simd::HasX86Avx2OrBetter(simd) || densecore::simd::IsArmFamily(simd);
    }();
    return enabled;
}
}  // namespace

bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch) {
    return IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch);
}

bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                            int n_head_kv, int head_dim_q, int head_dim_kv) {
    return IsPagedDecodeCandidateImpl(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv);
}

bool IsPagedDecodeModeAlwaysOn() {
    return IsPagedDecodeModeAlwaysOnImpl();
}

DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshot() {
    return GetDecodeRuntimeStatsSnapshotImpl();
}

const char* GetDecodePagedFallbackReasonName(std::size_t index) {
    return GetDecodePagedFallbackReasonNameImpl(index);
}

// ============================================================================
// InferenceContext Implementation ("Rebuild Graph, Reuse Memory" Strategy)
// ============================================================================

void InferenceContext::Init(size_t buffer_size) {
    if (initialized) {
        return;  // Already initialized
    }

    // Allocate aligned buffer for GGML context.
    // Avoid std::vector::resize() here because it zero-fills the entire
    // region. For multi-GB hybrid-SSM graph contexts that turns startup into
    // a giant memset before any real work begins.
    constexpr size_t kAlignment = 64;
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(buffer_size, kAlignment);
#else
    if (posix_memalign(&ptr, kAlignment, buffer_size) != 0) {
        ptr = nullptr;
    }
#endif
    if (!ptr) {
        throw densecore::OutOfMemoryException("InferenceContext: aligned allocation failed");
    }
    compute_buffer = ptr;
    compute_buffer_size = buffer_size;

    struct ggml_init_params params = {
        .mem_size = buffer_size,
        .mem_buffer = compute_buffer,
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);

    if (ctx_compute) {
        initialized = true;
        std::cerr << "[InferenceContext] Initialized with " << (buffer_size / (1024 * 1024)) << " MB persistent buffer"
                  << std::endl;
    } else {
        throw densecore::OutOfMemoryException("InferenceContext: ggml_init failed");
    }
}

void InferenceContext::Reset() {
    if (!initialized || !compute_buffer || compute_buffer_size == 0) {
        return;
    }

    // GGML doesn't expose a public ggml_reset_pool() API.
    // Workaround: Free and re-init with the SAME memory buffer.
    // This is effectively O(1) since:
    //   - No malloc/free syscalls (buffer is reused)
    //   - ggml_init just sets up internal allocator state
    if (ctx_compute) {
        ggml_free(ctx_compute);
    }

    struct ggml_init_params params = {
        .mem_size = compute_buffer_size,
        .mem_buffer = compute_buffer,
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);
    if (!ctx_compute) {
        throw densecore::OutOfMemoryException("InferenceContext::Reset: ggml_init failed");
    }
}

void InferenceContext::Free() {
    if (ctx_compute) {
        ggml_free(ctx_compute);
        ctx_compute = nullptr;
    }
    if (compute_buffer) {
#if defined(_WIN32)
        _aligned_free(compute_buffer);
#else
        std::free(compute_buffer);
#endif
        compute_buffer = nullptr;
    }
    compute_buffer_size = 0;
    initialized = false;
}

// Forward declaration for explicit work context
struct InferenceWorkContext;
static thread_local InferenceWorkContext* tls_work_ctx = nullptr;
static std::atomic<const BatchSpec*> g_shared_batch{nullptr};


// ============================================================================
// NEW: Robust KV Cache Update and Gather UserData
// ============================================================================
// This replaces the fragile cb_kv_manage approach that relied on ggml_pad
// assumptions. The new approach explicitly:
//   1. Writes current K/V to the PagedKVCache
//   2. Reads history from the cache into destination tensor
//   3. Appends current K/V to destination tensor
// ============================================================================

struct KVUpdateGatherUserData {
    PagedKVCache* cache;               // KV cache instance
    const BatchSpec* batch;            // Batch specification with block tables
    int layer;                         // Current transformer layer
    int head_dim;                      // Dimension per head
    int n_head_kv;                     // Number of KV heads
    int N;                             // Current batch size (new tokens)
    int n_past;                        // Number of past/history tokens
    bool is_k;                         // True for K tensor, false for V tensor
    bool read_only_shared_kv = false;  // Shared Gemma4 layers reuse cache without writing
    struct ggml_tensor* src_tensor;    // Pointer to Kcur/Vcur tensor (data accessed at runtime)
};

// Pool size for KVUpdateGatherUserData
static constexpr int kMaxKVUpdateGatherSlots = 256;

KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k);

// ============================================================================
// Multi-LoRA Batching Callback
// ============================================================================
// Applies per-request LoRA adapters during inference graph execution.
// Uses thread-local batch context to access the adapter-to-token mapping.
// The tensor name (e.g., "blk.0.attn_q") identifies which layer weights to use.
// ============================================================================

static const BatchSpec* GetCurrentBatch();

/**
 * @brief GGML callback for Multi-LoRA application during inference.
 *
 * 각 LoRA 가능 레이어(QKV, Output Projection, FFN)에서 호출됩니다.
 * Gather-Compute-Scatter 패턴으로 배치 내 각 토큰에 해당하는 LoRA 어댑터를 적용합니다.
 *
 * Time Complexity: O(N * R * D) where N=tokens, R=rank, D=hidden_dim
 * Space Complexity: O(N * max(R, D)) for scratch buffers (thread-local)
 *
 * @param dst Output tensor (base projection result, LoRA delta added in-place)
 * @param src0 First input (projection output tensor, same as dst)
 * @param src1 Second input (pre-projection hidden states, used for Gather)
 * @param ith Thread index (only thread 0 executes to avoid races)
 * @param nth Total threads
 * @param userdata Unused (uses GetCurrentBatch() for batch context)
 */
void cb_apply_multi_lora(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                         int ith, int nth, void* userdata) {
    (void)nth;
    (void)userdata;

    // Single-threaded execution: LoRA application is already parallelized internally
    if (ith != 0) return;

    if (!dst || !src0 || !dst->data || !src0->data) return;

    // Get current batch from explicit work context
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    // NOTE: ggml_map_custom2() allocates dst with ggml_dup_tensor(a), which only
    // duplicates metadata (shape/type) and does not copy payload. We must copy
    // base projection output (src0) into dst first, then apply LoRA deltas.
    if (dst->data != src0->data) {
        if (src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->ne[0] == dst->ne[0] &&
            src0->ne[1] == dst->ne[1]) {
            const size_t row_bytes = static_cast<size_t>(src0->ne[0]) * sizeof(float);
            for (int64_t row = 0; row < src0->ne[1]; ++row) {
                const auto* src_row = reinterpret_cast<const uint8_t*>(src0->data) + row * src0->nb[1];
                auto* dst_row = reinterpret_cast<uint8_t*>(dst->data) + row * dst->nb[1];
                memcpy(dst_row, src_row, row_bytes);
            }
        } else {
            memcpy(dst->data, src0->data, std::min(ggml_nbytes(dst), ggml_nbytes(src0)));
        }
    }

    // Early exit if no LoRA adapters in this batch
    if (batch->lora_map.empty()) return;

    // Validate tensor data - src1 is the pre-projection hidden states
    if (!src1 || !src1->data) return;

    // Extract layer name from tensor name (e.g., "blk.0.attn_q" -> "blk.0.attn_q")
    // The tensor name is set by apply_lora lambda in BuildTransformerGraph
    const char* layer_name = dst->name;
    if (!layer_name || layer_name[0] == '\0') return;

    // Convert GGML tensors to DenseCore Tensor format
    // src1 = pre-projection hidden states (input to LoRA)
    densecore::Tensor t_input;
    t_input.data = const_cast<void*>(src1->data);
    t_input.dtype = densecore::GgmlTypeToDType(src1->type);
    t_input.ndim = 2;
    t_input.shape[0] = src1->ne[1];  // tokens (GGML: ne[1] = rows after mul_mat)
    t_input.shape[1] = src1->ne[0];  // hidden_dim
    t_input.stride[0] = src1->nb[1];
    t_input.stride[1] = src1->nb[0];

    // dst = projection output (LoRA delta added in-place)
    densecore::Tensor t_output;
    t_output.data = dst->data;
    t_output.dtype = densecore::GgmlTypeToDType(dst->type);
    t_output.ndim = 2;
    t_output.shape[0] = dst->ne[1];  // tokens
    t_output.shape[1] = dst->ne[0];  // output_dim
    t_output.stride[0] = dst->nb[1];
    t_output.stride[1] = dst->nb[0];

    // Dispatch to CpuBackend (NUMA-aware, parallelized internally)
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ApplyMultiLoRA(t_input, std::string(layer_name), batch->lora_map, &t_output);
}

// ============================================================================
// Fused Add + RMSNorm Callback (AVX-512 Optimized)
// ============================================================================
// Combines residual connection (x += residual) and RMSNorm in a single pass
// to reduce memory bandwidth by loading/storing data once instead of twice.
// ============================================================================

/**
 * User data for fused Add+RMSNorm operation
 */
struct AddRMSNormUserData {
    const float* residual;                ///< Residual tensor data [n_embd, N]
    const float* rms_weight;              ///< RMSNorm weight [n_embd]
    int n_embd;                           ///< Embedding dimension
    int n_tokens;                         ///< Number of tokens
    float eps;                            ///< RMSNorm epsilon
    ptrdiff_t residual_row_stride = 0;    ///< Residual row stride in float elements
    std::vector<float> owned_rms_weight;  ///< Optional canonicalized RMS weight storage
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

// Thread-local pool for AddRMSNorm user data
static constexpr int kMaxAddRMSNormSlots = 256;

AddRMSNormUserData* GetAddRMSNormUserData();

// =============================================================================
// Parallel GEMV User Data + Buffers
// =============================================================================
static constexpr int kMaxGemvUserDataSlots = 2048;
static constexpr size_t kMaxQuantInputBufferSize = 65536;  // 64KB for large N
static constexpr int kMaxSmallBatchColsHard = 16;
static constexpr size_t kMaxDequantBufferSize = 16384;
static constexpr int kMaxPagedAttentionUserDataSlots = 256;

/**
 * User data for parallel GEMV operation
 */
struct GemvUserData {
    struct ggml_tensor* weight_tensor;  // Weight tensor (data accessed at runtime)
    int N;                              // Input dimension
    int K;                              // Output dimension
    ggml_type weight_type;              // Tensor type (F32, Q4_K, Q8_0, etc.)
    ggml_type input_quant_type;         // Quantization type for input (Q8_K, Q8_0, or F32)
    bool force_reference_scalar = false;
    int slot_id = -1;
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
};

struct GemvBatchedUserData {
    struct ggml_tensor* weight_tensor = nullptr;  // Weight tensor (data accessed at runtime)
    int N = 0;                                    // Input dimension
    int K = 0;                                    // Output dimension
    int M = 0;                                    // Number of input columns (tokens)
    ggml_type weight_type = GGML_TYPE_F32;
    bool force_reference_scalar = false;
    int slot_id = -1;
    ggml_type input_quant_type = GGML_TYPE_F32;
    size_t quant_row_stride = 0;  // Pre-computed aligned row stride for quantized input
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
};

struct PagedAttentionUserData {
    PagedKVCache* cache = nullptr;
    int layer = 0;
    int head_dim = 0;
    int v_head_dim = 0;
    int n_head = 0;
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    std::atomic<uint64_t> epoch_started{0};
    std::atomic<uint64_t> epoch_done{0};
    std::atomic<int> kv_writers_done{0};
    std::atomic<int> shared_block_ptrs_ready{0};
    std::vector<const void*>* shared_k_block_ptrs = nullptr;
    std::vector<const void*>* shared_v_block_ptrs = nullptr;
};

inline int ResolvePagedAttentionDecodeHeadTile(int n_head, int n_tokens, int n_tasks) {
    const int configured_head_tile = std::max(1, ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", 8));
    if (n_head <= 0 || n_tokens <= 0 || n_tasks <= 0) {
        return configured_head_tile;
    }

    // For latency-sensitive decode (batch 1-4), expose enough head tiles to keep
    // CPU workers occupied. The historical fixed tile=8 leaves batch=1 with only
    // four tasks on 32-head Qwen models, which strands cores during decode.
    if (n_tokens <= 4) {
        const int target_tiles_per_token = std::max(1, (n_tasks + n_tokens - 1) / n_tokens);
        const int adaptive_head_tile = std::max(1, (n_head + target_tiles_per_token - 1) / target_tiles_per_token);
        return std::min(configured_head_tile, adaptive_head_tile);
    }
    return configured_head_tile;
}

/**
 * Custom callback for fused Add + RMSNorm
 *
 * Input tensor (src): Current tensor to add residual to and normalize
 * Output tensor (dst): Result of (src + residual) normalized with RMSNorm
 *
 * Uses AVX-512 fused kernel for optimal memory bandwidth utilization.
 */
void cb_residual_rmsnorm_fused(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                               void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->rms_weight) return;
    if (!src || !dst || !src->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (src->nb[0] != static_cast<int64_t>(sizeof(float)) || dst->nb[0] != static_cast<int64_t>(sizeof(float))) return;

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    const float eps = ud->eps;
    const bool has_residual = ud->residual != nullptr;
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t residual_row_stride = ud->residual_row_stride > 0 ? ud->residual_row_stride : n_embd;
    const bool run_reference_probe = ShouldRunAddRmsNormReferenceProbe(ud->layer_idx);

    // Partition work across tokens
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens) return;

    // Process assigned tokens
    for (int t = t_start; t < t_end; t++) {
        const float* x_ptr = reinterpret_cast<const float*>(src->data) + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* out_ptr = reinterpret_cast<float*>(dst->data) + static_cast<ptrdiff_t>(t) * dst_row_stride;
        const float* res_ptr =
            has_residual ? (ud->residual + static_cast<ptrdiff_t>(t) * residual_row_stride) : nullptr;

        if (has_residual) {
            // Use unified AddRMSNorm dispatcher (Runtime AVX512/AVX2/Scalar)
            densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);
        } else {
            densecore::simd::RMSNorm(x_ptr, ud->rms_weight, out_ptr, static_cast<size_t>(n_embd), eps);
        }

        if (run_reference_probe) {
            static std::atomic<int> emitted{0};
            const int max_calls = ParsePositiveEnvInt("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_MAX_CALLS", 8);
            const int prior = emitted.load(std::memory_order_relaxed);
            if (prior < max_calls && emitted.fetch_add(1, std::memory_order_relaxed) < max_calls) {
                std::vector<float> summed(static_cast<size_t>(n_embd));
                double sum_sq = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    float val = x_ptr[i];
                    if (res_ptr) {
                        val += res_ptr[i];
                    }
                    summed[static_cast<size_t>(i)] = val;
                    sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }

                const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);
                float max_abs_diff = 0.0f;
                int first_bad_idx = -1;
                float first_actual = 0.0f;
                float first_ref = 0.0f;
                bool actual_nonfinite = false;
                bool ref_nonfinite = false;
                for (int i = 0; i < n_embd; ++i) {
                    const float ref = summed[static_cast<size_t>(i)] * inv_rms * ud->rms_weight[i];
                    const float actual = out_ptr[i];
                    actual_nonfinite = actual_nonfinite || !std::isfinite(actual);
                    ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
                    const float diff = std::fabs(actual - ref);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        first_bad_idx = i;
                        first_actual = actual;
                        first_ref = ref;
                    }
                }
                const int seq_id = ud->token_seq_ids ? ud->token_seq_ids[t] : -1;
                fprintf(stderr,
                        "[ADD_RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d has_residual=%d "
                        "max_abs_diff=%.9g first_idx=%d actual=%.9g ref=%.9g actual_nonfinite=%d "
                        "ref_nonfinite=%d\n",
                        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", t,
                        seq_id, has_residual ? 1 : 0, max_abs_diff, first_bad_idx, first_actual, first_ref,
                        actual_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0);
            }
        }
    }
}

// ============================================================================
// Fused SiLU×Mul Callback (SwiGLU FFN Optimization)
// ============================================================================
// Computes: out = silu(gate) * up in a single pass
// Saves memory by avoiding intermediate silu(gate) tensor allocation.
// ============================================================================

/**
 * Custom callback for fused SiLU×Mul (SwiGLU FFN)
 *
 * Signature compatible with ggml_map_custom2:
 *   void (*)(struct ggml_tensor *dst, const struct ggml_tensor *a,
 *            const struct ggml_tensor *b, int ith, int nth, void *userdata)
 */
void cb_silu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;  // Not needed since we use a/b directly

    // a = gate tensor (w1 output, SiLU input)
    // b = up tensor (w3 output)
    // dst = output tensor

    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;

    const float* gate = reinterpret_cast<const float*>(a->data);
    const float* up = reinterpret_cast<const float*>(b->data);
    float* out = reinterpret_cast<float*>(dst->data);

    // Total elements (flattened)
    const size_t size = ggml_nelements(a);

    // Use parallel SiLU×Mul from simd_ops.h
    densecore::simd::SiLUMulParallel(out, gate, up, size, ith, nth);
}

void cb_gelu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;
    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;

    const float* gate = reinterpret_cast<const float*>(a->data);
    const float* up = reinterpret_cast<const float*>(b->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const size_t size = ggml_nelements(a);

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    for (size_t i = begin; i < end; ++i) {
        const float x = gate[i];
        const float x3 = x * x * x;
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
        out[i] = gelu * up[i];
    }
}

void cb_gelu_tanh_unary(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!src || !dst || !src->data || !dst->data) return;

    const float* in = reinterpret_cast<const float*>(src->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const size_t size = ggml_nelements(src);

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    for (size_t i = begin; i < end; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        out[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
    }
}

struct SharedScalarGateApplyUserData {
    const struct ggml_tensor* gate_logits_scalar = nullptr;
};

static inline SharedScalarGateApplyUserData* GetSharedScalarGateApplyUserData() {
    static thread_local SharedScalarGateApplyUserData pool[64];
    static thread_local int index = 0;
    if (index >= 64) {
        index = 0;
    }
    return &pool[index++];
}

static void cb_apply_shared_scalar_gate(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                        void* userdata) {
    auto* ud = static_cast<SharedScalarGateApplyUserData*>(userdata);
    if (!dst || !src || !ud || !ud->gate_logits_scalar || !dst->data || !src->data || !ud->gate_logits_scalar->data) {
        return;
    }

    const int hidden = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int scalar_elems = static_cast<int>(ggml_nelements(ud->gate_logits_scalar));
    if (hidden <= 0 || tokens <= 0 || scalar_elems < tokens) {
        return;
    }

    const auto stable_sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };

    const float* src_base = reinterpret_cast<const float*>(src->data);
    float* dst_base = reinterpret_cast<float*>(dst->data);
    const float* gate_base = reinterpret_cast<const float*>(ud->gate_logits_scalar->data);
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t gate_row_stride = static_cast<ptrdiff_t>(ud->gate_logits_scalar->nb[1] / sizeof(float));

    const int token_begin = (tokens * ith) / nth;
    const int token_end = (tokens * (ith + 1)) / nth;
    for (int t = token_begin; t < token_end; ++t) {
        const float gate = stable_sigmoid(gate_base[static_cast<ptrdiff_t>(t) * gate_row_stride]);
        const float* src_row = src_base + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* dst_row = dst_base + static_cast<ptrdiff_t>(t) * dst_row_stride;
        densecore::simd::ScaleF32(dst_row, src_row, gate, static_cast<size_t>(hidden));
    }
}

// ============================================================================
// Fused QKV Projection Callback (Tensor-Level Parallelism)
// ============================================================================

// Computes Q, K, V projections in a single pass with intra-operator parallelism
// across the output dimension (dim_q + dim_k + dim_v). This enables
// multi-thread utilization during decode when batch_size=1.
// ============================================================================

/**
 * User data for fused Q/K/V projection operation
 */
struct QKVUserData {
    const float* w_q;  ///< Q weight [dim_q, n_embd] (row-major)
    const float* w_k;  ///< K weight [dim_k, n_embd] (row-major)
    const float* w_v;  ///< V weight [dim_v, n_embd] (row-major)
    int n_embd;        ///< Input embedding dimension
    int dim_q;         ///< Q output dimension (n_head * head_dim)
    int dim_k;         ///< K output dimension (n_head_kv * head_dim)
    int dim_v;         ///< V output dimension (n_head_kv * head_dim)
};

// ============================================================================
// SSM (Mamba2) Callback UserData Structs
// ============================================================================
struct SSMConv1DUserData {
    float* conv_state;
    const float* weight;
    int channels;
    int kernel_size;
    int layer_idx = -1;
    int ssm_ordinal = -1;
    const int* token_seq_ids = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
};

struct ProjectionReferenceUserData {
    const struct ggml_tensor* weight_tensor = nullptr;
    const struct ggml_tensor* input_tensor = nullptr;
    const uint8_t* int4_packed = nullptr;
    const float* int4_scales = nullptr;
    const float* int4_zeros = nullptr;
    int int4_group_size = 0;
    int int4_k = 0;
    int int4_n = 0;
    const uint8_t* fp8_packed = nullptr;
    TransformerModel::FP8Format fp8_format = TransformerModel::FP8Format::E4M3FN;
    int fp8_k = 0;
    int fp8_n = 0;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct SharedScalarGateReferenceUserData {
    const struct ggml_tensor* shared_ffn_pre_gate = nullptr;
    const struct ggml_tensor* shared_gate_logits_scalar = nullptr;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct RmsNormReferenceUserData {
    const struct ggml_tensor* input_tensor = nullptr;
    const struct ggml_tensor* norm_weight = nullptr;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct HiddenSnapshotUserData {
    int layer_idx = -1;
    int token_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct Gemma4KVSummaryUserData {
    int layer_idx = -1;
    int source_layer = -1;
    const char* action = nullptr;
    const char* kind = nullptr;
};

static Gemma4KVSummaryUserData* AllocateGemma4KVSummaryUserData(struct ggml_context* ctx_c);
static void cb_gemma4_kv_summary_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* userdata);

static struct ggml_tensor* MaybeAttachGemma4SharedKVProbe(struct ggml_context* ctx_c, struct ggml_tensor* tensor,
                                                          const char* action, const char* kind, int layer_idx,
                                                          int source_layer) {
    const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
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

static void ComputeMatmulReferenceInt4BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->int4_packed || !ud->int4_scales || !ud->int4_zeros || ud->int4_group_size <= 0 || ud->int4_k <= 0 ||
        ud->int4_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->int4_k;
    const int N = ud->int4_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K || (K % ud->int4_group_size) != 0 ||
        (ud->int4_group_size & 1) != 0) {
        out->clear();
        return;
    }

    const int num_groups = K / ud->int4_group_size;
    const int packed_k = (K + 1) / 2;
    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int m = 0; m < M; ++m) {
        const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const float scale = ud->int4_scales[n * num_groups + g];
                const float zero = ud->int4_zeros[n * num_groups + g];
                const int k_start = g * ud->int4_group_size;
                const uint8_t* w_packed = ud->int4_packed + static_cast<size_t>(n) * packed_k +
                                          static_cast<size_t>(g) * (ud->int4_group_size / 2);
                for (int k = 0; k < ud->int4_group_size; ++k) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    const uint8_t packed_byte = w_packed[byte_idx];
                    int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                                 : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    if (q & 0x08) {
                        q = static_cast<int8_t>(q | static_cast<int8_t>(0xF0));
                    }
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k_start + k) *
                                                                                  ud->input_tensor->nb[0]);
                    sum += x * (scale * (static_cast<float>(q) - zero));
                }
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static void ComputeMatmulReferenceFp8BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->fp8_packed || ud->fp8_k <= 0 || ud->fp8_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->fp8_k;
    const int N = ud->fp8_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    std::vector<float> dequant_row(static_cast<size_t>(K), 0.0f);
    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int n = 0; n < N; ++n) {
        const uint8_t* packed_row = ud->fp8_packed + static_cast<size_t>(n) * static_cast<size_t>(K);
        if (ud->fp8_format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        } else {
            densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        }
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float x =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k) * ud->input_tensor->nb[0]);
                sum += dequant_row[static_cast<size_t>(k)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static bool ShouldRunSharedScalarGateReferenceProbe(int layer_idx) {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (!enabled) return false;
    static const int target_layer = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    static std::atomic<int> remaining_budget{[]() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_MAX_CALLS");
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

struct AttentionCoreReferenceUserData {
    const struct ggml_tensor* value_tensor = nullptr;
    const struct ggml_tensor* gate_tensor = nullptr;
    int layer_idx = -1;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim_q = 0;
    int head_dim_k = 0;
    int head_dim_v = 0;
    int n_past = 0;
    int sliding_window = -1;
    float attention_scale = 0.0f;
    float logit_softcap = 0.0f;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

#ifdef DENSECORE_TEST_BUILD
static std::atomic<int> g_test_capture_attention_layer{-1};
static std::vector<float>* g_test_capture_attention_out = nullptr;

static void cb_test_capture_attention_tensor(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                             void* userdata) {
    (void)nth;
    (void)userdata;
    if (!dst || !src || !dst->data || !src->data) return;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    if (ith != 0 || !g_test_capture_attention_out || src->type != GGML_TYPE_F32) {
        return;
    }
    const int elems = static_cast<int>(ggml_nelements(src));
    g_test_capture_attention_out->resize(static_cast<size_t>(elems));
    std::memcpy(g_test_capture_attention_out->data(), src->data, static_cast<size_t>(elems) * sizeof(float));
}
#endif


// ============================================================================
// SSM Delta Callback Helper logic
// ============================================================================

struct GLMDSAPackUserData {
    int n_heads;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int v_head_dim;
};

// Keep these implementation chunks in the same translation unit so GGML custom
// callbacks, thread-local pools, and anonymous-namespace helpers preserve their
// original linkage and runtime behavior while still splitting this file by role.
#include "inference_runtime.inl"

#include "inference_matmul.inl"

#include "inference_model_ops.inl"

static struct ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache,
                                                           struct ggml_context* ctx_c, const BatchSpec& batch,
                                                           bool embedding_mode, struct ggml_cgraph* gf,
                                                           struct ggml_tensor** out_embd, struct ggml_tensor** out_pos);

namespace {

const char* GraphExecutionRouteName(densecore::TransformerGraphExecutionRoute route) {
    switch (route) {
    case densecore::TransformerGraphExecutionRoute::RegistryBuilder: return "RegistryBuilder";
    case densecore::TransformerGraphExecutionRoute::InlineDenseAttention: return "InlineDenseAttention";
    case densecore::TransformerGraphExecutionRoute::InlineHybridSSM: return "InlineHybridSSM";
    case densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV: return "InlineSlidingWindowSharedKV";
    case densecore::TransformerGraphExecutionRoute::Reject:
    default: return "Reject";
    }
}

densecore::models::GraphBuilderSupport MakeDenseCoreInlineGraphSupport() {
    densecore::models::GraphBuilderSupport support{};
    support.builder_name = "DenseCoreInlineTransformerGraph";
    support.supported_families = {densecore::models::GraphFamily::DecoderDenseAttention,
                                  densecore::models::GraphFamily::DecoderHybridSSM,
                                  densecore::models::GraphFamily::DecoderSlidingWindowSharedKV};
    support.supports_sliding_window_attention = true;
    support.supports_shared_kv_source = true;
    support.supports_hybrid_ssm_mixer = true;
    support.supports_moe = true;
    support.supports_q_norm = true;
    support.supports_k_norm = true;
    support.supports_v_norm = true;
    support.supports_per_layer_kv_head_variability = true;
    support.supports_special_attention_mask = true;
    support.supports_special_residual_scaling = true;
    return support;
}

class InlineDenseDecoderRegistryBuilder : public densecore::TransformerGraphBuilder {
public:
    struct ggml_tensor* Build(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx,
                              const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                              struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) override {
        return BuildTransformerGraphInlineImpl(model, cache, ctx, batch, embedding_mode, gf, out_embd, out_pos);
    }

    const char* Name() const override { return "densecore_inline_dense_decoder"; }
};

struct InlineDenseDecoderRegistryBuilderRegistrar {
    InlineDenseDecoderRegistryBuilderRegistrar() {
        densecore::TransformerGraphRegistry::Instance().RegisterExact(
            densecore::kDenseDecoderGenericBuilderKey, "densecore_inline_dense_decoder",
            densecore::models::MakeDenseDecoderGenericSupport("densecore_inline_dense_decoder"),
            []() { return std::make_unique<InlineDenseDecoderRegistryBuilder>(); });
    }
};

static InlineDenseDecoderRegistryBuilderRegistrar g_inline_dense_decoder_registry_builder_registrar;

}  // namespace

densecore::TransformerGraphExecutionPlan
densecore::ResolveTransformerGraphExecutionPlan(const TransformerModel* model) {
    TransformerGraphExecutionPlan plan{};
    plan.resolution = models::ResolveGraphFamily(model);

    if (!model) {
        plan.route = TransformerGraphExecutionRoute::Reject;
        plan.debug_reason = "model is null";
        return plan;
    }

    std::string registry_debug;
    const auto descriptor =
        TransformerGraphRegistry::Instance().ResolveBuilderDescriptor(plan.resolution, &registry_debug);
    plan.debug_reason = "registry admission: " + registry_debug;
    if (descriptor) {
        plan.route = TransformerGraphExecutionRoute::RegistryBuilder;
        plan.registry_builder_key = descriptor->key;
        plan.selected_builder_name = descriptor->display_name;
        return plan;
    }

    const auto inline_admission = models::AdmitGraphBuilder(plan.resolution, MakeDenseCoreInlineGraphSupport());
    plan.debug_reason += " | inline admission: " + inline_admission.Summary();
    if (!inline_admission.admitted) {
        plan.route = TransformerGraphExecutionRoute::Reject;
        return plan;
    }

    switch (inline_admission.admitted_family) {
    case models::GraphFamily::DecoderDenseAttention:
        plan.route = TransformerGraphExecutionRoute::InlineDenseAttention;
        break;
    case models::GraphFamily::DecoderHybridSSM: plan.route = TransformerGraphExecutionRoute::InlineHybridSSM; break;
    case models::GraphFamily::DecoderSlidingWindowSharedKV:
        plan.route = TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV;
        break;
    case models::GraphFamily::EncoderDecoder:
    case models::GraphFamily::MultimodalProjectedDecoder:
    case models::GraphFamily::UNKNOWN:
    default:
        plan.route = TransformerGraphExecutionRoute::Reject;
        plan.debug_reason += " | unsupported inline family";
        break;
    }

    return plan;
}

std::unique_ptr<densecore::TransformerGraphBuilder>
densecore::InstantiateRegistryBuilderForExecutionPlan(const TransformerGraphExecutionPlan& plan,
                                                      std::string* error_reason) {
    if (plan.route != TransformerGraphExecutionRoute::RegistryBuilder) {
        if (error_reason) {
            *error_reason = "execution plan route is not RegistryBuilder";
        }
        return nullptr;
    }
    if (plan.registry_builder_key.empty()) {
        if (error_reason) {
            *error_reason = "registry builder route is missing an exact builder key";
        }
        return nullptr;
    }

    auto builder = TransformerGraphRegistry::Instance().GetBuilderExact(plan.registry_builder_key);
    if (!builder && error_reason) {
        *error_reason = "exact registry builder key '" + plan.registry_builder_key + "' is not available for execution";
    }
    return builder;
}

struct ggml_tensor* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx_c,
                                          const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                                          struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) {
    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(model);

    if (IsVerboseGraphBuildLoggingEnabled()) {
        std::cerr << "[BuildTransformerGraph] Resolved capabilities: "
                  << densecore::models::FormatModelGraphCapabilities(plan.resolution.capabilities) << std::endl;
        std::cerr << "[BuildTransformerGraph] Selected graph family: "
                  << densecore::models::FormatGraphFamilyResolution(plan.resolution) << std::endl;
        if (!plan.debug_reason.empty()) {
            std::cerr << "[BuildTransformerGraph] Dispatch detail: " << plan.debug_reason << std::endl;
        }
        std::cerr << "[BuildTransformerGraph] Execution route: " << GraphExecutionRouteName(plan.route)
                  << (plan.selected_builder_name.empty() ? "" : " via ")
                  << (plan.selected_builder_name.empty() ? "" : plan.selected_builder_name.c_str()) << std::endl;
    }

    switch (plan.route) {
    case densecore::TransformerGraphExecutionRoute::RegistryBuilder: {
        std::string execution_error;
        auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &execution_error);
        if (!builder) {
            throw densecore::GraphBuildException("BuildTransformerGraph dispatch selected registry builder key '" +
                                                 plan.registry_builder_key +
                                                 "' but execution failed closed: " + execution_error);
        }
        return builder->Build(model, cache, ctx_c, batch, embedding_mode, gf, out_embd, out_pos);
    }
    case densecore::TransformerGraphExecutionRoute::InlineDenseAttention:
    case densecore::TransformerGraphExecutionRoute::InlineHybridSSM:
    case densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV:
        return BuildTransformerGraphInlineImpl(model, cache, ctx_c, batch, embedding_mode, gf, out_embd, out_pos);
    case densecore::TransformerGraphExecutionRoute::Reject:
    default: throw densecore::GraphBuildException("BuildTransformerGraph fail-closed: " + plan.debug_reason);
    }
}

static struct ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache,
                                                           struct ggml_context* ctx_c, const BatchSpec& batch,
                                                           bool embedding_mode, struct ggml_cgraph* gf,
                                                           struct ggml_tensor** out_embd,
                                                           struct ggml_tensor** out_pos) {
    // ENSURE: ctx_c must be initialized with sufficient memory (e.g. 128MB+)
    // to hold the compute graph nodes, especially for deep models like Qwen.
    // This initialization happens in worker.cpp (InitGraphCache or temp
    // context).

    // N is batch size
    const int N = batch.tokens.size();
    const int n_embd = model->hparams.n_embd;
    const int n_head = model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    const int n_layer = model->hparams.n_layer;
    const int n_ctx = model->hparams.n_ctx;
    const DecodePagedAttentionPolicy decode_paged_policy = LoadDecodePagedAttentionPolicy();
    (void)n_embd;
    (void)n_head_kv;
    (void)n_ctx;

    // =========================================================================
    // 1. Token Embedding Lookup
    // =========================================================================
    struct ggml_tensor* embd_inp = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
    ggml_set_name(embd_inp, "embd_inp");
    if (embd_inp->data) {
        memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
    }
    if (out_embd) *out_embd = embd_inp;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, embd_inp);
    if (const float embedding_scale = densecore::models::ResolveInputEmbeddingScale(model); embedding_scale != 1.0f) {
        cur = ggml_scale(ctx_c, cur, embedding_scale);
    }

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check embedding output
        auto cb_check_embd = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* ud) {
            if (ith != 0) return;
            static int cb_ct = 0;
            const bool emit = (cb_ct < 5);
            if (emit && src && src->data) {
                const float* d = (const float*)src->data;
                int n = ggml_nelements(src);
                int zero_ct = 0;
                float mn = d[0], mx = d[0];
                for (int i = 0; i < n; i++) {
                    if (d[i] == 0.0f)
                        zero_ct++;
                    else {
                        if (d[i] < mn) mn = d[i];
                        if (d[i] > mx) mx = d[i];
                    }
                }
                fprintf(stderr, "[EMBED #%d] shape=[%ld,%ld] total=%d zero=%d min=%.6f max=%.6f\n", cb_ct,
                        (long)src->ne[0], (long)src->ne[1], n, zero_ct, mn, mx);
                cb_ct++;
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_embd, 1, nullptr);
    }

    // Position tensor for RoPE
    const int pos_ids_per_token = PositionIdsPerToken(model);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, static_cast<int64_t>(N) * pos_ids_per_token);
    ggml_set_name(pos, "pos");
    if (pos->data) {
        PopulatePositionTensor(model, batch, pos);
    }
    if (out_pos) *out_pos = pos;

    const bool decode_only_batch_layout = IsDecodeOnlyBatchLayout(batch, N);
    const int debug_query_base_pos = ResolveAttentionQueryBasePosition(batch);
    const bool gemma4_decode_special_transforms_disabled =
        model->arch_flags.is_gemma4 && densecore::models::IsGemma4DecodeSpecialTransformDisabled() &&
        debug_query_base_pos > 0;

    auto requires_gemma_rms_weight_offset = [&]() -> bool {
        return densecore::models::RequiresUnitOffsetRmsNorm(model);
    };

    auto effective_rms_weight = [&](struct ggml_tensor * norm_weight, const char* debug_name) -> struct ggml_tensor* {
        if (!norm_weight) {
            return norm_weight;
        }
        if (requires_gemma_rms_weight_offset()) {
            struct ggml_tensor* one = ggml_new_f32(ctx_c, 1.0f);
            struct ggml_tensor* shifted = ggml_add1(ctx_c, norm_weight, one);
            if (debug_name) {
                ggml_set_name(shifted, debug_name);
            }
            return shifted;
        }
        if (debug_name) {
            ggml_set_name(norm_weight, debug_name);
        }
        return norm_weight;
    };

    auto bind_add_rmsnorm_weight = [&](AddRMSNormUserData* ud, struct ggml_tensor* norm_weight,
                                       const char* weight_name) -> void {
        if (!ud || !norm_weight) {
            throw densecore::InvalidArgumentException("Missing RMSNorm weight tensor");
        }
        if (norm_weight->type != GGML_TYPE_F32 || !norm_weight->data) {
            throw densecore::InvalidArgumentException(std::string("RMSNorm weight ") + weight_name +
                                                      " must be materialized FP32");
        }
        const int64_t n_weight = ggml_nelements(norm_weight);
        if (n_weight <= 0) {
            throw densecore::InvalidArgumentException(std::string("RMSNorm weight ") + weight_name +
                                                      " has invalid element count");
        }
        const float* raw_weight = reinterpret_cast<const float*>(norm_weight->data);
        if (requires_gemma_rms_weight_offset()) {
            ud->owned_rms_weight.resize(static_cast<size_t>(n_weight));
            for (int64_t i = 0; i < n_weight; ++i) {
                ud->owned_rms_weight[static_cast<size_t>(i)] = raw_weight[i] + 1.0f;
            }
            ud->rms_weight = ud->owned_rms_weight.data();
        } else {
            ud->owned_rms_weight.clear();
            ud->rms_weight = raw_weight;
        }
    };

    auto apply_weighted_rms_norm = [&](struct ggml_tensor * src, struct ggml_tensor * norm_weight,
                                       const char* debug_name, int debug_layer_idx = -1) -> struct ggml_tensor* {
        if (!src || !norm_weight) {
            return src;
        }
        struct ggml_tensor* effective_norm_weight = effective_rms_weight(norm_weight, nullptr);

        const bool force_cpu_norm =
            IsMixedRoutingEnabled(&batch) && ResolvePreferredNormDevice(&batch) == densecore::DeviceType::CPU &&
            ResolvePreferredDevice(&batch) != densecore::DeviceType::CPU && src->type == GGML_TYPE_F32 &&
            effective_norm_weight->type == GGML_TYPE_F32 && src->nb[0] == static_cast<int64_t>(sizeof(float));

        if (!force_cpu_norm) {
            struct ggml_tensor* out = ggml_rms_norm(ctx_c, src, model->hparams.f_norm_rms_eps);
            out = ggml_mul(ctx_c, out, effective_norm_weight);
            if (debug_name) {
                ggml_set_name(out, debug_name);
            }
            if (debug_layer_idx >= 0 && ShouldRunRmsNormReferenceProbe(debug_layer_idx)) {
                auto* rms_ud = GetRmsNormReferenceUserData();
                rms_ud->input_tensor = src;
                rms_ud->norm_weight = effective_norm_weight;
                rms_ud->layer_idx = debug_layer_idx;
                rms_ud->token_seq_ids = batch.seq_id.data();
                rms_ud->stage = "rms_norm";
                rms_ud->var_name = debug_name ? debug_name : "rms_norm";
                out = ggml_map_custom1(ctx_c, out, cb_rmsnorm_reference_probe, 1, rms_ud);
            }
            return out;
        }

        AddRMSNormUserData* ud = GetAddRMSNormUserData();
        ud->residual = nullptr;
        bind_add_rmsnorm_weight(ud, norm_weight, debug_name ? debug_name : "rms_norm");
        ud->n_embd = static_cast<int>(src->ne[0]);
        ud->n_tokens = static_cast<int>(src->ne[1]);
        ud->eps = model->hparams.f_norm_rms_eps;
        ud->residual_row_stride = 0;
        ud->layer_idx = -1;
        ud->token_seq_ids = batch.seq_id.data();
        ud->stage = "rms_norm";
        ud->var_name = debug_name ? debug_name : "rms_norm";
        const int n_tasks = ResolveTaskCount(&batch, std::max(1, ud->n_tokens));
        struct ggml_tensor* out = ggml_map_custom1(ctx_c, src, cb_residual_rmsnorm_fused, n_tasks, ud);
        if (debug_name) {
            ggml_set_name(out, debug_name);
        }
        return out;
    };

    struct ggml_tensor* gemma4_per_layer_inputs = nullptr;
    if (model->arch_flags.is_gemma4 && model->gemma4_hidden_size_per_layer_input > 0 &&
        model->gemma4_per_layer_model_projection && model->gemma4_per_layer_projection_norm &&
        model->gemma4_per_layer_token_embeddings) {
        const int hidden_per_layer = model->gemma4_hidden_size_per_layer_input;
        const int n_layer_i = static_cast<int>(model->hparams.n_layer);

        struct ggml_tensor* token_inputs = ggml_get_rows(ctx_c, model->gemma4_per_layer_token_embeddings, embd_inp);
        token_inputs = ggml_scale(ctx_c, token_inputs, std::sqrt(static_cast<float>(hidden_per_layer)));
        token_inputs = ggml_reshape_3d(ctx_c, token_inputs, hidden_per_layer, n_layer_i, N);

        struct ggml_tensor* projected_inputs =
            smart_mul_mat(ctx_c, model->gemma4_per_layer_model_projection, cur, model);
        projected_inputs =
            ggml_scale(ctx_c, projected_inputs, 1.0f / std::sqrt(static_cast<float>(model->hparams.n_embd)));
        projected_inputs = ggml_reshape_3d(ctx_c, projected_inputs, hidden_per_layer, n_layer_i, N);

        struct ggml_tensor* projected_inputs_2d =
            ggml_reshape_2d(ctx_c, projected_inputs, hidden_per_layer, n_layer_i * N);
        projected_inputs_2d = apply_weighted_rms_norm(projected_inputs_2d, model->gemma4_per_layer_projection_norm,
                                                      "gemma4_per_layer_projection_norm");
        gemma4_per_layer_inputs = ggml_reshape_3d(ctx_c, projected_inputs_2d, hidden_per_layer, n_layer_i, N);
        gemma4_per_layer_inputs = ggml_add(ctx_c, gemma4_per_layer_inputs, token_inputs);
        gemma4_per_layer_inputs = ggml_scale(ctx_c, gemma4_per_layer_inputs, std::pow(2.0f, -0.5f));
    }

    // Reuse causal mask tensor across layers for the same forward pass.
    // Prefill builds N>1 attention repeatedly; hoisting avoids O(L * N * K)
    // mask materialization and graph-build overhead.
    struct ggml_tensor* shared_prefill_flash_mask = nullptr;
    int shared_prefill_mask_n_total = -1;
    int shared_prefill_mask_n_padded = -1;
    int shared_prefill_mask_n = -1;
    int shared_prefill_mask_n_past = -1;
    int shared_prefill_mask_sliding_window = -1;
    if (N > 1 && !decode_only_batch_layout) {
        int prefill_n_past = 0;
        if (cache && batch.num_seqs > 0 && !batch.n_past.empty()) {
            prefill_n_past = *std::max_element(batch.n_past.begin(), batch.n_past.end());
        }
        const int prefill_n_total = prefill_n_past + N;
        const int prefill_n_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);

        shared_prefill_flash_mask = ggml_new_tensor_4d(ctx_c, GGML_TYPE_F32, prefill_n_total, prefill_n_padded, 1, 1);
        float* mask_data = reinterpret_cast<float*>(shared_prefill_flash_mask->data);
        for (int q = 0; q < prefill_n_padded; ++q) {
            for (int k = 0; k < prefill_n_total; ++k) {
                const int query_pos = prefill_n_past + q;
                const int key_pos = k;
                const int idx = k + q * prefill_n_total;
                mask_data[idx] = (q < N && key_pos <= query_pos) ? 0.0f : -INFINITY;
            }
        }

        shared_prefill_mask_n_total = prefill_n_total;
        shared_prefill_mask_n_padded = prefill_n_padded;
        shared_prefill_mask_n = N;
        shared_prefill_mask_n_past = prefill_n_past;
        shared_prefill_mask_sliding_window = -1;
    }

    // =========================================================================
    // 2. Transformer Layers
    // =========================================================================
    int ssm_ordinal_counter = 0;  // Counts SSM layers for state indexing
    std::array<uint64_t, 7> moe_wiring_reason_counts{};
    const bool moe_wiring_debug = IsMoEWiringDebugEnabled();
    if (moe_wiring_debug) {
        const auto resolution = densecore::models::ResolveGraphFamily(model);
        std::fprintf(stderr,
                     "[MOE_WIRING_MODEL] arch=%d variant=%d preferred_family=%s fail_closed=%d capabilities=%s "
                     "n_layer=%d n_experts=%u top_k=%u moe_first_k_dense_replace=%d N=%d decode_only=%d\n",
                     static_cast<int>(resolution.capabilities.arch),
                     static_cast<int>(resolution.capabilities.variant),
                     densecore::models::GraphFamilyName(resolution.preferred_family), resolution.fail_closed ? 1 : 0,
                     densecore::models::FormatModelGraphCapabilities(resolution.capabilities).c_str(), n_layer,
                     model->hparams.n_experts, model->hparams.n_experts_used, model->moe_first_k_dense_replace, N,
                     decode_only_batch_layout ? 1 : 0);
    }
    for (int il = 0; il < n_layer; ++il) {
        auto& layer = model->layers[il];
        auto* attn_norm = layer.Get(model_keys::kAttnNorm);
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        auto* wv = layer.Get(model_keys::kAttnVWeight);
        auto* wo = layer.Get(model_keys::kAttnOWeight);
        auto* q_a = layer.Get(model_keys::kAttnQAProj);
        auto* q_a_norm = layer.Get(model_keys::kAttnQANorm);
        auto* q_b = layer.Get(model_keys::kAttnQBProj);
        auto* kv_a = layer.Get(model_keys::kAttnKvAProj);
        auto* kv_a_norm = layer.Get(model_keys::kAttnKvANorm);
        auto* kv_b = layer.Get(model_keys::kAttnKvBProj);
        auto* indexer_wq_b = layer.Get(model_keys::kIndexerWqB);
        auto* indexer_wk = layer.Get(model_keys::kIndexerWk);
        auto* indexer_k_norm = layer.Get(model_keys::kIndexerKNorm);
        auto* indexer_weights_proj = layer.Get(model_keys::kIndexerWeightsProj);
        auto* bq = layer.Get(model_keys::kAttnQBias);
        auto* bk = layer.Get(model_keys::kAttnKBias);
        auto* bv = layer.Get(model_keys::kAttnVBias);
        auto* bo = layer.Get(model_keys::kAttnOBias);
        auto* q_norm = layer.Get(model_keys::kAttnQNorm);
        auto* k_norm = layer.Get(model_keys::kAttnKNorm);
        auto* rope_freqs = layer.Get(model_keys::kAttnRopeFreqs);
        auto* ffn_norm = layer.Get(model_keys::kFfnNorm);
        auto* ffn_gate = layer.Get(model_keys::kFfnGate);
        auto* ffn_up = layer.Get(model_keys::kFfnUp);
        auto* ffn_down = layer.Get(model_keys::kFfnDown);
        auto* ffn_shared_gate = layer.Get(model_keys::kFfnSharedGate);
        auto* moe_gate = layer.Get(model_keys::kMoeGate);
        const int layer_num_experts = static_cast<int>(layer.NumExperts());
        const int dense_replace_cutoff = std::max(0, model->moe_first_k_dense_replace);
        const bool dense_replace_gate = il < dense_replace_cutoff;
        const bool model_has_moe = model->hparams.n_experts > 0;
        const bool has_moe_gate = moe_gate != nullptr;
        const bool has_experts = layer_num_experts > 0;
        const bool layer_moe_candidate = has_moe_gate && has_experts && !dense_replace_gate;
        if (moe_wiring_debug) {
            MoEWiringReasonCode reason = MoEWiringReasonCode::Wired;
            if (!layer.is_moe) {
                if (!model_has_moe) {
                    reason = MoEWiringReasonCode::ModelHasNoMoE;
                } else if (!has_moe_gate) {
                    reason = MoEWiringReasonCode::MissingMoeGate;
                } else if (!has_experts) {
                    reason = MoEWiringReasonCode::NoExperts;
                } else if (dense_replace_gate) {
                    reason = MoEWiringReasonCode::DenseReplaceGate;
                } else if (layer_moe_candidate) {
                    reason = MoEWiringReasonCode::LayerFlagMismatch;
                } else {
                    reason = MoEWiringReasonCode::LayerFlagFalse;
                }
            }
            const int reason_code = static_cast<int>(reason);
            if (reason_code >= 0 && reason_code < static_cast<int>(moe_wiring_reason_counts.size())) {
                moe_wiring_reason_counts[static_cast<size_t>(reason_code)]++;
            }
            std::fprintf(stderr,
                         "[MOE_WIRING_LAYER] layer=%d is_moe_layer=%d has_moe_gate=%d num_experts=%d "
                         "dense_replace_gate=%d model_has_moe=%d attempted=%d reason_code=%d\n",
                         il, layer.is_moe ? 1 : 0, has_moe_gate ? 1 : 0, layer_num_experts, dense_replace_gate ? 1 : 0,
                         model_has_moe ? 1 : 0, layer.is_moe ? 1 : 0, reason_code);
        }

        struct ggml_tensor* inpL = cur;
        if (ShouldRunHiddenSnapshotProbe(il, "layer_input")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "layer_input";
                hidden_ud->var_name = "inpL";
                inpL = ggml_map_custom1(ctx_c, inpL, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        // Attention Norm
        if (!attn_norm) {
            throw densecore::InvalidArgumentException("Missing attention_norm weight in TransformerLayer");
        }
        cur = apply_weighted_rms_norm(cur, attn_norm, "attn_norm", il);

        // SSM / Attention layer dispatch
        const bool is_ssm_layer = model->IsHybridSSMLayer(il);
        struct ggml_tensor* attn_out = nullptr;
        struct ggml_tensor* attn_post_residual = nullptr;

        if (is_ssm_layer) {
            // =================================================================
            // SSM/Mamba2 Layer Forward Path
            // =================================================================
            auto* attn_qkv = layer.Get(model_keys::kAttnQkvWeight);
            auto* ssm_conv1d_w = layer.Get(model_keys::kSSMConv1d);
            auto* ssm_a = layer.Get(model_keys::kSSMA);
            auto* ssm_alpha_w = layer.Get(model_keys::kSSMAlpha);
            auto* ssm_beta_w = layer.Get(model_keys::kSSMBeta);
            auto* ssm_dt_bias_t = layer.Get(model_keys::kSSMDtBias);
            auto* ssm_norm_w = layer.Get(model_keys::kSSMNorm);
            auto* attn_gate_w = layer.Get(model_keys::kAttnGate);
            auto* ssm_out_w = layer.Get(model_keys::kSSMOut);

            if (!attn_qkv || !ssm_conv1d_w || !ssm_a || !ssm_alpha_w || !ssm_beta_w || !ssm_dt_bias_t || !ssm_norm_w ||
                !attn_gate_w || !ssm_out_w) {
                throw densecore::InvalidArgumentException("Missing SSM weights in layer " + std::to_string(il));
            }

            const int ssm_ordinal = ssm_ordinal_counter++;
            auto& ssm_rt = model->ssm_layer_states[ssm_ordinal];

            const int d_inner = model->ssm_inner_size;
            const int num_v_heads = model->ssm_time_step_rank;
            const int head_dim_v = d_inner / num_v_heads;
            const int head_dim_k = model->ssm_state_size;
            const int n_groups = model->ssm_group_count;
            const int conv_channels = d_inner + 2 * n_groups * head_dim_k;
            const int conv_kernel = model->ssm_conv_kernel;
            const size_t expected_conv_weights = static_cast<size_t>(conv_channels) * static_cast<size_t>(conv_kernel);
            const size_t expected_head_by_embd = static_cast<size_t>(num_v_heads) * static_cast<size_t>(n_embd);
            const size_t expected_per_head = static_cast<size_t>(num_v_heads);
            const size_t expected_norm_shared = static_cast<size_t>(head_dim_v);
            const size_t expected_norm_full = static_cast<size_t>(d_inner);

            if (ssm_rt.conv1d_f32.size() != expected_conv_weights || ssm_rt.alpha_f32.size() != expected_head_by_embd ||
                ssm_rt.beta_f32.size() != expected_head_by_embd || ssm_rt.dt_bias_f32.size() != expected_per_head ||
                ssm_rt.a_log_f32.size() != expected_per_head ||
                (ssm_rt.norm_layout == Qwen35SSMNormLayout::SHARED_HEAD_DIM &&
                 ssm_rt.norm_f32.size() != expected_norm_shared) ||
                (ssm_rt.norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER &&
                 ssm_rt.norm_f32.size() != expected_norm_full) ||
                ssm_rt.norm_layout == Qwen35SSMNormLayout::INVALID) {
                throw densecore::InvalidArgumentException("Missing canonical hybrid SSM runtime weights in layer " +
                                                          std::to_string(il));
            }

            if (IsSSMNonFiniteDebugEnabled()) {
                auto cb_check_ssm_input = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                             void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    const int* seq_ids = (GetCurrentWorkContext() && GetCurrentWorkContext()->batch)
                                             ? GetCurrentWorkContext()->batch->seq_id.data()
                                             : nullptr;
                    CheckSSMFiniteTensor(layer_idx, seq_ids, src, "ssm_input", "attn_norm_out");
                    if (dst->data && src->data) {
                        std::memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int ssm_input_layers[128] = {};
                for (int li = 0; li < 128; ++li) ssm_input_layers[li] = li;
                cur = ggml_map_custom1(ctx_c, cur, cb_check_ssm_input, 1, &ssm_input_layers[il]);
            }

            // 1. qkv_mixed projection: normed input [n_embd, N] → [conv_channels, N]
            struct ggml_tensor* qkv_mixed = smart_mul_mat(ctx_c, attn_qkv, cur, model);
            if (IsDebugSSMQkvReferenceEnabled() || IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* qkv_ref_ud = GetProjectionReferenceUserData();
                qkv_ref_ud->weight_tensor = attn_qkv;
                qkv_ref_ud->input_tensor = cur;
                qkv_ref_ud->layer_idx = il;
                qkv_ref_ud->token_seq_ids = batch.seq_id.data();
                qkv_ref_ud->stage = "qkv_proj";
                qkv_ref_ud->var_name = "qkv_mixed";
                qkv_mixed = ggml_map_custom1(ctx_c, qkv_mixed, cb_projection_reference_probe, 1, qkv_ref_ud);
            }
            if (IsSSMNonFiniteDebugEnabled()) {
                auto cb_check_ssm_qkv = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                           void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    const int* seq_ids = (GetCurrentWorkContext() && GetCurrentWorkContext()->batch)
                                             ? GetCurrentWorkContext()->batch->seq_id.data()
                                             : nullptr;
                    CheckSSMFiniteTensor(layer_idx, seq_ids, src, "qkv_proj", "qkv_mixed");
                    if (dst->data && src->data) {
                        std::memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int ssm_qkv_layers[128] = {};
                for (int li = 0; li < 128; ++li) ssm_qkv_layers[li] = li;
                qkv_mixed = ggml_map_custom1(ctx_c, qkv_mixed, cb_check_ssm_qkv, 1, &ssm_qkv_layers[il]);
            }

            // 2. Conv1D: updates conv_state ring buffer, outputs [conv_channels, N]
            SSMConv1DUserData* conv_ud = GetSSMConv1DUserData();
            conv_ud->conv_state = nullptr;
            conv_ud->weight = ssm_rt.conv1d_f32.data();
            conv_ud->channels = conv_channels;
            conv_ud->kernel_size = conv_kernel;
            conv_ud->layer_idx = il;
            conv_ud->ssm_ordinal = ssm_ordinal;
            conv_ud->token_seq_ids = batch.seq_id.data();
            conv_ud->runtime_states = &batch.hybrid_ssm_runtime_states;
            struct ggml_tensor* qkv_conv = ggml_map_custom1(ctx_c, qkv_mixed, cb_ssm_conv1d, 1, conv_ud);

            // 3. z projection and recurrent Qwen3.5 delta-net block.
            struct ggml_tensor* z = smart_mul_mat(ctx_c, attn_gate_w, cur, model);
            if (IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* z_ref_ud = GetProjectionReferenceUserData();
                z_ref_ud->weight_tensor = attn_gate_w;
                z_ref_ud->input_tensor = cur;
                z_ref_ud->layer_idx = il;
                z_ref_ud->token_seq_ids = batch.seq_id.data();
                z_ref_ud->stage = "gate_proj";
                z_ref_ud->var_name = "z";
                z = ggml_map_custom1(ctx_c, z, cb_projection_reference_probe, 1, z_ref_ud);
            }
            SSMQwen35DeltaUserData* scan_ud = GetSSMQwen35DeltaUserData();
            scan_ud->alpha_weight = ssm_rt.alpha_f32.data();
            scan_ud->beta_weight = ssm_rt.beta_f32.data();
            scan_ud->dt_bias = ssm_rt.dt_bias_f32.data();
            scan_ud->a_log = ssm_rt.a_log_f32.data();
            scan_ud->norm_weight = ssm_rt.norm_f32.data();
            scan_ud->ssm_state = nullptr;
            scan_ud->n_embd = n_embd;
            scan_ud->d_inner = d_inner;
            scan_ud->n_heads = num_v_heads;
            scan_ud->head_dim_v = head_dim_v;
            scan_ud->head_dim_k = head_dim_k;
            scan_ud->n_groups = n_groups;
            scan_ud->norm_layout = ssm_rt.norm_layout;
            scan_ud->norm_eps = model->hparams.f_norm_rms_eps;
            scan_ud->layer_idx = il;
            scan_ud->ssm_ordinal = ssm_ordinal;
            scan_ud->token_seq_ids = batch.seq_id.data();
            scan_ud->runtime_states = &batch.hybrid_ssm_runtime_states;
            scan_ud->z_tensor = z;
            scan_ud->input_tensor = cur;
            if (gf) {
                ggml_build_forward_expand(gf, z);
                ggml_build_forward_expand(gf, cur);
            }
            struct ggml_tensor* y_scratch = ggml_map_custom1(ctx_c, qkv_conv, cb_ssm_qwen35_delta_qkv_only, 1, scan_ud);
            struct ggml_tensor* y = ggml_cont(ctx_c, ggml_view_2d(ctx_c, y_scratch, d_inner, N, y_scratch->nb[1], 0));

            // 4. Output projection: [d_inner, N] → [n_embd, N]
            cur = smart_mul_mat(ctx_c, ssm_out_w, y, model);
            if (IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* out_ref_ud = GetProjectionReferenceUserData();
                out_ref_ud->weight_tensor = ssm_out_w;
                out_ref_ud->input_tensor = y;
                out_ref_ud->layer_idx = il;
                out_ref_ud->token_seq_ids = batch.seq_id.data();
                out_ref_ud->stage = "ssm_out_proj";
                out_ref_ud->var_name = "ssm_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, out_ref_ud);
            }

            // Residual connection
            attn_out = cur;
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
        } else {
            // =================================================================
            // Attention Layer Forward Path
            // =================================================================

            // Q/K/V Projections (using smart dispatcher for Parallel GEMV)
            const bool use_glm_dsa_mla =
                model->arch_flags.is_glm_dsa && q_a && q_a_norm && q_b && kv_a && kv_a_norm && kv_b;
            const bool gemma4_value_from_key = !use_glm_dsa_mla && !wv && wk && model->arch_flags.is_gemma4;
            if (!use_glm_dsa_mla && (!wq || !wk || !wv)) {
                if (gemma4_value_from_key) {
                    // Gemma4 alternative attention omits V projection on some layers.
                    // Those blocks use the raw K projection as the input to v_norm.
                    wv = wk;
                }
            }
            if (!use_glm_dsa_mla && (!wq || !wk || !wv)) {
                const std::string hint =
                    model->arch_flags.is_hybrid_ssm
                        ? " (hybrid SSM model: layer may be misclassified — check layer_types in GGUF)"
                        : "";
                throw densecore::InvalidArgumentException("Missing Q/K/V weights in TransformerLayer " +
                                                          std::to_string(il) + hint);
            }
            struct ggml_tensor* Qcur = nullptr;
            struct ggml_tensor* Kcur = nullptr;
            struct ggml_tensor* Vcur = nullptr;
            struct ggml_tensor* glm_q_resid = nullptr;
            struct ggml_tensor* glm_index_query = nullptr;
            struct ggml_tensor* glm_index_weights = nullptr;
            struct ggml_tensor* glm_index_key = nullptr;
            const bool use_glm_dsa_sparse = use_glm_dsa_mla && cache && cache->has_index_cache &&
                                            cache->index_head_dim == model->glm_index_head_dim &&
                                            decode_only_batch_layout && indexer_wq_b && indexer_wk && indexer_k_norm &&
                                            indexer_weights_proj && model->glm_index_n_heads > 0 &&
                                            model->glm_index_head_dim > 0;

            // Optional fused QKV projection (single pass over input per token).
            // Falls back to per-projection matmul for non-F32/quantized weights.
            const bool fused_qkv_supported = !use_glm_dsa_mla && IsFusedQKVEnabled() && cur->type == GGML_TYPE_F32 &&
                                             wq->type == GGML_TYPE_F32 && wk->type == GGML_TYPE_F32 &&
                                             wv->type == GGML_TYPE_F32 && wq->data && wk->data && wv->data &&
                                             wq->ne[0] == cur->ne[0] && wk->ne[0] == cur->ne[0] &&
                                             wv->ne[0] == cur->ne[0] && wq->ne[1] > 0 && wk->ne[1] > 0 && wv->ne[1] > 0;

            if (use_glm_dsa_mla) {
                static bool logged_glm_dsa_path = false;
                if (!logged_glm_dsa_path) {
                    std::cerr << "[DenseCore] GLM-5 DSA path enabled"
                              << (use_glm_dsa_sparse ? " with sparse indexer cache."
                                                     : " without sparse indexer cache; using dense attention.")
                              << std::endl;
                    logged_glm_dsa_path = true;
                }
                const int qk_nope_head_dim = model->glm_qk_nope_head_dim;
                const int qk_rope_head_dim = model->glm_qk_rope_head_dim;
                const int v_head_dim = model->glm_v_head_dim;
                const int kv_lora_rank = model->glm_kv_lora_rank;
                const int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
                const int kv_proj_head_dim = qk_nope_head_dim + v_head_dim;

                if (qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 || v_head_dim <= 0 || kv_lora_rank <= 0) {
                    throw densecore::InvalidArgumentException("Incomplete GLM-5 DSA metadata for MLA projection path");
                }

                glm_q_resid = smart_mul_mat(ctx_c, q_a, cur, model);
                glm_q_resid = apply_weighted_rms_norm(glm_q_resid, q_a_norm, "glm_q_a_norm", il);
                struct ggml_tensor* q_raw = smart_mul_mat(ctx_c, q_b, glm_q_resid, model);

                struct ggml_tensor* kv_a_cur = smart_mul_mat(ctx_c, kv_a, cur, model);
                const int64_t kv_a_dim = kv_a_cur->ne[0];
                if (kv_a_dim < static_cast<int64_t>(kv_lora_rank + qk_rope_head_dim)) {
                    throw densecore::InvalidArgumentException(
                        "GLM-5 kv_a projection is smaller than kv_lora_rank + rope dim");
                }

                struct ggml_tensor* kv_comp = ggml_view_2d(ctx_c, kv_a_cur, kv_lora_rank, N, kv_a_cur->nb[1], 0);
                struct ggml_tensor* k_rope =
                    ggml_view_2d(ctx_c, kv_a_cur, qk_rope_head_dim, N, kv_a_cur->nb[1],
                                 static_cast<size_t>(kv_lora_rank) * ggml_element_size(kv_a_cur));
                kv_comp = ggml_cont(ctx_c, kv_comp);
                k_rope = ggml_cont(ctx_c, k_rope);
                kv_comp = apply_weighted_rms_norm(kv_comp, kv_a_norm, "glm_kv_a_norm", il);
                struct ggml_tensor* kv_raw = smart_mul_mat(ctx_c, kv_b, kv_comp, model);

                if (q_raw->ne[0] != static_cast<int64_t>(q_head_dim * n_head)) {
                    throw densecore::InvalidArgumentException("GLM-5 q_b projection shape mismatch");
                }
                if (kv_raw->ne[0] != static_cast<int64_t>(kv_proj_head_dim * n_head_kv)) {
                    throw densecore::InvalidArgumentException("GLM-5 kv_b projection shape mismatch");
                }

                GLMDSAPackUserData* q_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* k_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* v_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                if (!q_pack_ud || !k_pack_ud || !v_pack_ud) {
                    throw densecore::OutOfMemoryException("Failed to allocate GLM-5 DSA packing userdata");
                }
                *q_pack_ud = {n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *k_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *v_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};

                struct ggml_tensor* q_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head, N);
                struct ggml_tensor* k_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head_kv, N);
                struct ggml_tensor* v_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, v_head_dim * n_head_kv, N);

                // Repack [nope|rope] into [rope|nope] so the existing partial-RoPE path
                // can operate on the leading rope dimensions.
                Qcur = ggml_map_custom2(ctx_c, q_packed, q_raw, cb_pack_glm_dsa_q, 1, q_pack_ud);
                Kcur = ggml_map_custom3(ctx_c, k_packed, kv_raw, k_rope, cb_pack_glm_dsa_k, 1, k_pack_ud);
                Vcur = ggml_map_custom2(ctx_c, v_packed, kv_raw, cb_pack_glm_dsa_v, 1, v_pack_ud);

                if (use_glm_dsa_sparse) {
                    glm_index_query = smart_mul_mat(ctx_c, indexer_wq_b, glm_q_resid, model);
                    glm_index_weights = smart_mul_mat(ctx_c, indexer_weights_proj, cur, model);
                    glm_index_weights =
                        ggml_scale(ctx_c, glm_index_weights, 1.0f / std::sqrt((float)model->glm_index_n_heads));
                    glm_index_key = smart_mul_mat(ctx_c, indexer_wk, cur, model);
                    glm_index_key = apply_weighted_rms_norm(glm_index_key, indexer_k_norm, "glm_index_k_norm", il);
                }
            } else if (fused_qkv_supported) {
                const int dim_q_fused = static_cast<int>(wq->ne[1]);
                const int dim_k_fused = static_cast<int>(wk->ne[1]);
                const int dim_v_fused = static_cast<int>(wv->ne[1]);
                const int n_embd_fused = static_cast<int>(cur->ne[0]);
                const int merged_dim = dim_q_fused + dim_k_fused + dim_v_fused;

                struct ggml_tensor* qkv_input = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx_c, cur);
                struct ggml_tensor* qkv_merged = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, merged_dim, N);

                QKVUserData* qkv_ud = GetQKVUserData();
                qkv_ud->w_q = reinterpret_cast<const float*>(wq->data);
                qkv_ud->w_k = reinterpret_cast<const float*>(wk->data);
                qkv_ud->w_v = reinterpret_cast<const float*>(wv->data);
                qkv_ud->n_embd = n_embd_fused;
                qkv_ud->dim_q = dim_q_fused;
                qkv_ud->dim_k = dim_k_fused;
                qkv_ud->dim_v = dim_v_fused;

                const int n_tasks = ResolveTaskCount(&batch, std::max(1, merged_dim));
                qkv_merged = ggml_map_custom2(ctx_c, qkv_merged, qkv_input, cb_compute_qkv_map2, n_tasks, qkv_ud);

                const size_t k_offset = static_cast<size_t>(dim_q_fused) * sizeof(float);
                const size_t v_offset = static_cast<size_t>(dim_q_fused + dim_k_fused) * sizeof(float);
                Qcur = ggml_view_2d(ctx_c, qkv_merged, dim_q_fused, N, qkv_merged->nb[1], 0);
                Kcur = ggml_view_2d(ctx_c, qkv_merged, dim_k_fused, N, qkv_merged->nb[1], k_offset);
                Vcur = ggml_view_2d(ctx_c, qkv_merged, dim_v_fused, N, qkv_merged->nb[1], v_offset);
            } else {
                Qcur = smart_mul_mat(ctx_c, wq, cur, model);
                Kcur = smart_mul_mat(ctx_c, wk, cur, model);
                Vcur = smart_mul_mat(ctx_c, wv, cur, model);
                if (ShouldRunAttentionProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* q_ref_ud = GetProjectionReferenceUserData();
                    q_ref_ud->weight_tensor = wq;
                    q_ref_ud->input_tensor = cur;
                    q_ref_ud->layer_idx = il;
                    q_ref_ud->token_seq_ids = batch.seq_id.data();
                    q_ref_ud->stage = "attn_q_proj";
                    q_ref_ud->var_name = "Qcur";
                    Qcur = ggml_map_custom1(ctx_c, Qcur, cb_projection_reference_probe, 1, q_ref_ud);

                    ProjectionReferenceUserData* k_ref_ud = GetProjectionReferenceUserData();
                    k_ref_ud->weight_tensor = wk;
                    k_ref_ud->input_tensor = cur;
                    k_ref_ud->layer_idx = il;
                    k_ref_ud->token_seq_ids = batch.seq_id.data();
                    k_ref_ud->stage = "attn_k_proj";
                    k_ref_ud->var_name = "Kcur";
                    Kcur = ggml_map_custom1(ctx_c, Kcur, cb_projection_reference_probe, 1, k_ref_ud);

                    ProjectionReferenceUserData* v_ref_ud = GetProjectionReferenceUserData();
                    v_ref_ud->weight_tensor = wv;
                    v_ref_ud->input_tensor = cur;
                    v_ref_ud->layer_idx = il;
                    v_ref_ud->token_seq_ids = batch.seq_id.data();
                    v_ref_ud->stage = "attn_v_proj";
                    v_ref_ud->var_name = "Vcur";
                    Vcur = ggml_map_custom1(ctx_c, Vcur, cb_projection_reference_probe, 1, v_ref_ud);
                }
            }

            // Apply Multi-LoRA
            char name_buf[64];
            auto apply_lora = [&](struct ggml_tensor* dst, struct ggml_tensor* src, const char* suffix) {
                if (batch.lora_map.empty()) {
                    return dst;
                }
                if (!ggml_is_contiguous(dst)) {
                    dst = ggml_cont(ctx_c, dst);
                }
                snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", il, suffix);
                ggml_set_name(dst, name_buf);
                return ggml_map_custom2(ctx_c, dst, src, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            };

            Qcur = apply_lora(Qcur, cur, "attn_q");
            Kcur = apply_lora(Kcur, cur, "attn_k");
            Vcur = apply_lora(Vcur, cur, "attn_v");

            // Add Bias if present (for Qwen2 and some other models)
            if (bq) {
                if (Qcur->ne[0] == bq->ne[0]) {
                    Qcur = ggml_add(ctx_c, Qcur, bq);
                } else {
                    // SKIP BIAS (Safe fallback)
                    // std::cerr << "Skipping BQ mismatch L" << il << std::endl;
                }
            }
            if (bk) {
                if (Kcur->ne[0] == bk->ne[0]) {
                    Kcur = ggml_add(ctx_c, Kcur, bk);
                } else {
                    // SKIP BIAS on mismatch to avoid graph complexity/hangs
                    // std::cerr << "Skipping BK mismatch" << std::endl;
                }
            }
            if (bv) {
                if (Vcur->ne[0] == bv->ne[0]) {
                    Vcur = ggml_add(ctx_c, Vcur, bv);
                } else {
                    // SKIP BIAS
                }
            }

            // ggml_reshape_3d requires contiguous inputs. Fused QKV views can be
            // strided, so materialize contiguous tensors before reshape.
            if (!ggml_is_contiguous(Qcur)) {
                Qcur = ggml_cont(ctx_c, Qcur);
            }
            if (!ggml_is_contiguous(Kcur)) {
                Kcur = ggml_cont(ctx_c, Kcur);
            }
            if (!ggml_is_contiguous(Vcur)) {
                Vcur = ggml_cont(ctx_c, Vcur);
            }

            // Dynamically infer head dimensions from the actual projected tensors
            int dim_q = Qcur->ne[0];
            int dim_k = Kcur->ne[0];
            int dim_v = Vcur->ne[0];

            int n_head_kv = densecore::models::ResolveLayerKVHeadCount(model, il);
            const bool use_runtime_kv_dims = densecore::models::UseRuntimeKVHeadDims(model);
            int head_dim_kv = (!use_runtime_kv_dims && model->hparams.n_embd_head_k > 0)
                                  ? model->hparams.n_embd_head_k
                                  : (n_head_kv > 0 ? (dim_k / n_head_kv) : 0);
            int head_dim_v = (!use_runtime_kv_dims && model->hparams.n_embd_head_v > 0)
                                 ? model->hparams.n_embd_head_v
                                 : (n_head_kv > 0 ? (dim_v / n_head_kv) : 0);
            struct ggml_tensor* attn_gate = nullptr;

            // Qwen3.5 hybrid attention packs [q | gate] per-head, not as two
            // contiguous global halves. Mirror llama.cpp's strided per-head views.
            if (model->arch_flags.is_hybrid_ssm && dim_q > head_dim_kv * n_head) {
                const int64_t q_attn_dim = static_cast<int64_t>(head_dim_kv) * n_head;
                const int64_t q_full_per_head = dim_q / n_head;
                const int64_t q_attn_per_head = head_dim_kv;
                if (q_full_per_head >= q_attn_per_head * 2) {
                    struct ggml_tensor* Qcur_full = Qcur;
                    const size_t elem = ggml_element_size(Qcur_full);
                    struct ggml_tensor* Qcur_head = ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N,
                                                                 elem * q_full_per_head, Qcur_full->nb[1], 0);
                    struct ggml_tensor* gate_head =
                        ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N, elem * q_full_per_head,
                                     Qcur_full->nb[1], elem * q_attn_per_head);
                    Qcur = ggml_cont_2d(ctx_c, Qcur_head, q_attn_dim, N);
                    attn_gate = ggml_cont_2d(ctx_c, gate_head, q_attn_dim, N);
                } else {
                    attn_gate = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1],
                                                              q_attn_dim * ggml_element_size(Qcur)));
                    Qcur = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1], 0));
                }
                dim_q = static_cast<int>(q_attn_dim);
            }

            int head_dim_q = dim_q / n_head;

            const bool gemma4_shared_kv_layer =
                model->arch_flags.is_gemma4 && densecore::models::Gemma4KVSourceLayer(model, il) != il;
            bool k_done = false;
            bool v_done = false;

            // Reshape Q (Standard)
            ValidateAttentionProjectionShape3D(Qcur, "Qcur", il, head_dim_q, n_head, N, dim_q, n_head);
            Qcur = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);

            // Reshape K/V
            if (!k_done) {
                ValidateAttentionProjectionShape3D(Kcur, "Kcur", il, head_dim_kv, n_head_kv, N, dim_k, n_head_kv);
                Kcur = ggml_reshape_3d(ctx_c, Kcur, head_dim_kv, n_head_kv, N);
            }
            if (!v_done) {
                ValidateAttentionProjectionShape3D(Vcur, "Vcur", il, head_dim_v, n_head_kv, N, dim_v, n_head_kv);
                Vcur = ggml_reshape_3d(ctx_c, Vcur, head_dim_v, n_head_kv, N);
            }

            // =========================================================================
            // Per-Head QK-Norm (Architecture-flag based for Qwen3/Qwen2.5)
            // =========================================================================
            // Qwen3 requires RMS normalization applied per-head, not over the entire
            // embedding dimension. We reshape to [head_dim, n_heads * n_tokens] so that
            // ggml_rms_norm normalizes each head_dim vector independently.
            //
            // Using arch_flags for explicit requirement checking instead of implicit
            // null pointer guards. The tensor null check is kept for safety.
            //
            // Flow:
            //   1. Reshape Q from [head_dim, n_head, N] to [head_dim, n_head * N]
            //   2. Apply ggml_rms_norm (normalizes over ne[0] = head_dim)
            //   3. Multiply by weight [head_dim] (broadcasts across all head*token)
            //   4. Reshape back to [head_dim, n_head, N]
            // =========================================================================
            // Q Normalization (required for Qwen3, optional for others)
            if (densecore::models::ShouldApplyQNorm(model, q_norm)) {
                const int64_t q_n_tokens = Qcur->ne[2];  // N (batch size)
                struct ggml_tensor* q_norm_effective = effective_rms_weight(q_norm, nullptr);
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int qnorm_dbg = 0;
                    if (qnorm_dbg < 2 && q_norm->data && q_norm->type == GGML_TYPE_F32) {
                        const float* qn_raw = reinterpret_cast<const float*>(q_norm->data);
                        fprintf(stderr,
                                "[QNORM_L3_RAW #%d] q_norm_raw[0]=%.6f q_norm_raw[1]=%.6f q_norm_raw[2]=%.6f "
                                "q_norm_raw[3]=%.6f\n",
                                qnorm_dbg, qn_raw[0], qn_raw[1], qn_raw[2], qn_raw[3]);
                    }
                    if (qnorm_dbg < 2 && q_norm_effective->data && q_norm_effective->type == GGML_TYPE_F32) {
                        const float* qn = reinterpret_cast<const float*>(q_norm_effective->data);
                        fprintf(
                            stderr,
                            "[QNORM_L3_EFFECTIVE #%d] q_norm[0]=%.6f q_norm[1]=%.6f q_norm[2]=%.6f q_norm[3]=%.6f\n",
                            qnorm_dbg, qn[0], qn[1], qn[2], qn[3]);
                        qnorm_dbg++;
                    }
                }

                if (head_dim_q == q_norm_effective->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head * n_tokens] for per-head norm
                    struct ggml_tensor* Q_2d = ggml_reshape_2d(ctx_c, Qcur, head_dim_q, n_head * q_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    if (Q_2d->type != GGML_TYPE_F32) {
                        fprintf(stderr, "CRITICAL: Layer %d Q_2d type is %d! Tensor name: %s\n", il, Q_2d->type,
                                Q_2d->name);
                    }
                    Q_2d = ggml_rms_norm(ctx_c, Q_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    Q_2d = ggml_mul(ctx_c, Q_2d, q_norm_effective);

                    // Reshape back to original 3D: [head_dim, n_head, n_tokens]
                    Qcur = ggml_reshape_3d(ctx_c, Q_2d, head_dim_q, n_head, q_n_tokens);
                } else if (il == 0) {
                    static bool logged_q_mismatch = false;
                    if (!logged_q_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_q_norm dimension mismatch! "
                                  << "Qcur->ne[0]=" << Qcur->ne[0] << " vs norm->ne[0]=" << q_norm->ne[0]
                                  << ". Skipping Q normalization." << std::endl;
                        logged_q_mismatch = true;
                    }
                }
            }

            // K Normalization (required for Qwen3, optional for others)
            if (densecore::models::ShouldApplyKNorm(model, k_norm, gemma4_shared_kv_layer)) {
                const int64_t k_n_tokens = Kcur->ne[2];  // N (batch size)
                struct ggml_tensor* k_norm_effective = effective_rms_weight(k_norm, nullptr);
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int knorm_dbg = 0;
                    if (knorm_dbg < 2 && k_norm->data && k_norm->type == GGML_TYPE_F32) {
                        const float* kn_raw = reinterpret_cast<const float*>(k_norm->data);
                        fprintf(stderr,
                                "[KNORM_L3_RAW #%d] k_norm_raw[0]=%.6f k_norm_raw[1]=%.6f k_norm_raw[2]=%.6f "
                                "k_norm_raw[3]=%.6f\n",
                                knorm_dbg, kn_raw[0], kn_raw[1], kn_raw[2], kn_raw[3]);
                    }
                    if (knorm_dbg < 2 && k_norm_effective->data && k_norm_effective->type == GGML_TYPE_F32) {
                        const float* kn = reinterpret_cast<const float*>(k_norm_effective->data);
                        fprintf(
                            stderr,
                            "[KNORM_L3_EFFECTIVE #%d] k_norm[0]=%.6f k_norm[1]=%.6f k_norm[2]=%.6f k_norm[3]=%.6f\n",
                            knorm_dbg, kn[0], kn[1], kn[2], kn[3]);
                        knorm_dbg++;
                    }
                }

                if (head_dim_kv == k_norm_effective->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head_kv * n_tokens] for per-head norm
                    struct ggml_tensor* K_2d = ggml_reshape_2d(ctx_c, Kcur, head_dim_kv, n_head_kv * k_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    if (K_2d->type != GGML_TYPE_F32) {
                        fprintf(stderr, "CRITICAL: Layer %d K_2d type is %d! Tensor name: %s\n", il, K_2d->type,
                                K_2d->name);
                    }
                    K_2d = ggml_rms_norm(ctx_c, K_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    K_2d = ggml_mul(ctx_c, K_2d, k_norm_effective);

                    // Reshape back to original 3D: [head_dim, n_head_kv, n_tokens]
                    Kcur = ggml_reshape_3d(ctx_c, K_2d, head_dim_kv, n_head_kv, k_n_tokens);
                } else if (il == 0) {
                    static bool logged_k_mismatch = false;
                    if (!logged_k_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_k_norm dimension mismatch! "
                                  << "Kcur->ne[0]=" << Kcur->ne[0] << " vs norm->ne[0]=" << k_norm->ne[0]
                                  << ". Skipping K normalization." << std::endl;
                        logged_k_mismatch = true;
                    }
                }
            }

            if (densecore::models::ShouldApplyVNorm(model, Vcur, gemma4_shared_kv_layer)) {
                struct ggml_tensor* v_norm = layer.Get(model_keys::kAttnVNorm);
                const int64_t v_n_tokens = Vcur->ne[2];
                if (head_dim_v > 0 && n_head_kv > 0) {
                    struct ggml_tensor* V_2d = ggml_reshape_2d(ctx_c, Vcur, head_dim_v, n_head_kv * v_n_tokens);
                    V_2d = ggml_rms_norm(ctx_c, V_2d, model->hparams.f_norm_rms_eps);
                    if (v_norm && v_norm->ne[0] == head_dim_v) {
                        V_2d = ggml_mul(ctx_c, V_2d, v_norm);
                    }
                    Vcur = ggml_reshape_3d(ctx_c, V_2d, head_dim_v, n_head_kv, v_n_tokens);
                }
            }

            // Apply RoPE
            // Use n_rot from model params if specified (e.g. for partial RoPE or
            // specific dim) Fallback to full head_dim_q if n_rot is 0
            int rope_dim = model->hparams.n_rot;
            float rope_freq_base = model->hparams.rope_freq_base;
            bool use_gemma4_proportional_rope = false;
            struct ggml_tensor* rope_freq_factors = nullptr;
            if (model->arch_flags.is_gemma4) {
                const bool is_sliding_layer = densecore::models::IsGemma4SlidingLayer(model, il);
                if (is_sliding_layer) {
                    rope_dim = model->gemma4_rope_dim_swa > 0 ? model->gemma4_rope_dim_swa : rope_dim;
                } else {
                    // Gemma4 full attention: proportional RoPE via rope_freqs.weight.
                    // The freq_factors tensor has [1.0×64, 1e30×192] for 256 pairs:
                    // pairs 0-63 (freq=1.0) get normal rotation, pairs 64-255 (freq=1e30)
                    // get effectively zero rotation. The GGUF weights are permuted to
                    // match this NEOX-mode pairing convention (same as llama.cpp).
                    use_gemma4_proportional_rope = !densecore::models::IsGemma4FullRopeFreqsDisabled();
                    rope_freq_factors = use_gemma4_proportional_rope ? rope_freqs : nullptr;
                    rope_dim = use_gemma4_proportional_rope ? model->gemma4_rope_dim_full : 0;
                    if (rope_dim <= 0) {
                        rope_dim = static_cast<int>(std::lround(static_cast<float>(head_dim_q) *
                                                                model->gemma4_full_attention_partial_rotary_factor));
                        if (rope_dim <= 0) rope_dim = head_dim_q;
                    }
                    // RoPE pairs operate on an even count of dimensions.
                    rope_dim &= ~1;
                }
                rope_freq_base = is_sliding_layer && model->gemma4_rope_freq_base_swa > 0.0f
                                     ? model->gemma4_rope_freq_base_swa
                                     : model->gemma4_rope_freq_base_full;
            }
            if (rope_dim <= 0) {
                rope_dim = head_dim_q;
            }

            // Ensure rope_dim is valid (<= head_dim)
            if (rope_dim > head_dim_q) rope_dim = head_dim_q;

            // SKIP RoPE if we detected anomaly and sliced (k_done)
            // Layer 0 anomaly (80 dim) is unsafe for 128-dim RoPE/Kernel which expects
            // 128
            bool skip_rope = k_done;
            if (skip_rope) {
                // std::cerr << "[DenseCore] Skipping RoPE for anomalous Layer " << il <<
                // std::endl;
            }

            if (!skip_rope) {
                const int rope_mode = model->arch_flags.is_gemma4 ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
                struct ggml_tensor* Q_rope_fast = nullptr;
                struct ggml_tensor* K_rope_fast = nullptr;
                const bool can_use_precomputed_rope =
                    IsPrecomputedRoPEEnabled() &&
                    (!model->arch_flags.is_gemma4 ||
                     (use_gemma4_proportional_rope && rope_freq_factors == nullptr && !gemma4_shared_kv_layer));
                if (can_use_precomputed_rope) {
                    Q_rope_fast = ggml_rope_precomputed_table(ctx_c, Qcur, pos, model, rope_dim, &batch);
                    K_rope_fast = ggml_rope_precomputed_table(ctx_c, Kcur, pos, model, rope_dim, &batch);
                }
                const bool use_mrope = model->hparams.rope_sections[0] > 0 && model->hparams.rope_sections[1] > 0;
                if (gemma4_shared_kv_layer) {
                    if (Q_rope_fast) {
                        Qcur = Q_rope_fast;
                    } else if (use_mrope) {
                        const int rope_mrope_mode = ModelMRoPEMode(model);
                        int rope_sections[GGML_MROPE_SECTIONS] = {
                            model->hparams.rope_sections[0],
                            model->hparams.rope_sections[1],
                            model->hparams.rope_sections[2],
                            model->hparams.rope_sections[3],
                        };
                        Qcur =
                            ggml_rope_multi(ctx_c, Qcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    } else {
                        Qcur = ggml_rope_ext(ctx_c, Qcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    }
                } else if (Q_rope_fast && K_rope_fast) {
                    Qcur = Q_rope_fast;
                    Kcur = K_rope_fast;
                } else {
                    if (use_mrope) {
                        const int rope_mrope_mode = ModelMRoPEMode(model);
                        int rope_sections[GGML_MROPE_SECTIONS] = {
                            model->hparams.rope_sections[0],
                            model->hparams.rope_sections[1],
                            model->hparams.rope_sections[2],
                            model->hparams.rope_sections[3],
                        };
                        Qcur =
                            ggml_rope_multi(ctx_c, Qcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur =
                            ggml_rope_multi(ctx_c, Kcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    } else {
                        // Fallback to standard GGML RoPE when precomputed path is
                        // unavailable for this tensor/layout.
                        Qcur = ggml_rope_ext(ctx_c, Qcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur = ggml_rope_ext(ctx_c, Kcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    }
                }
            }

            if (model->arch_flags.is_gemma4 && N == 1 && IsGemma4SharedKVSourceLayer(model, il)) {
                Kcur = MaybeAttachGemma4SharedKVProbe(ctx_c, Kcur, "pre-cache-write", "K", il, il);
                Vcur = MaybeAttachGemma4SharedKVProbe(ctx_c, Vcur, "pre-cache-write", "V", il, il);
            }

            const bool use_explicit_attention_scale = model->hparams.f_attention_scale > 0.0f;
            if (use_explicit_attention_scale) {
                Qcur = ggml_scale(ctx_c, Qcur, model->hparams.f_attention_scale);
            }

            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                const int index_n_heads = model->glm_index_n_heads;
                const int index_head_dim = model->glm_index_head_dim;
                const int index_rope_dim = std::min(model->glm_qk_rope_head_dim, index_head_dim);

                glm_index_query = ggml_cont(ctx_c, glm_index_query);
                glm_index_weights = ggml_cont(ctx_c, glm_index_weights);
                glm_index_key = ggml_cont(ctx_c, glm_index_key);

                glm_index_query = ggml_reshape_3d(ctx_c, glm_index_query, index_head_dim, index_n_heads, N);
                if (index_rope_dim > 0) {
                    glm_index_query = ggml_rope_ext(ctx_c, glm_index_query, pos, nullptr, index_rope_dim, 0, n_ctx,
                                                    model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                                    1.0f, 0.0f, 0.0f);
                    struct ggml_tensor* index_k_3d = ggml_reshape_3d(ctx_c, glm_index_key, index_head_dim, 1, N);
                    index_k_3d = ggml_rope_ext(ctx_c, index_k_3d, pos, nullptr, index_rope_dim, 0, n_ctx,
                                               model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                               1.0f, 0.0f, 0.0f);
                    glm_index_key = ggml_reshape_2d(ctx_c, index_k_3d, index_head_dim, N);
                }
            }

            struct ggml_tensor* KQV = nullptr;
            struct ggml_tensor* attn_ref_k = nullptr;
            struct ggml_tensor* attn_ref_v = nullptr;
            int attn_ref_n_past = 0;
            bool attn_core_reference_eligible = false;
            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                PagedAttentionUserData* ud = GetPagedAttentionUserData();
                ud->cache = cache;
                ud->layer = il;
                ud->head_dim = head_dim_q;
                ud->v_head_dim = head_dim_v;
                ud->n_head = n_head;
                ud->index_n_heads = model->glm_index_n_heads;
                ud->index_head_dim = model->glm_index_head_dim;
                ud->index_topk = model->glm_index_topk;
                ud->epoch_started.store(0, std::memory_order_relaxed);
                ud->epoch_done.store(0, std::memory_order_relaxed);
                ud->kv_writers_done.store(0, std::memory_order_relaxed);

                struct ggml_tensor* q_sparse = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
                struct ggml_tensor* k_sparse = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
                struct ggml_tensor* v_sparse = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);
                struct ggml_tensor* index_q_sparse =
                    ggml_is_contiguous(glm_index_query) ? glm_index_query : ggml_cont(ctx_c, glm_index_query);
                struct ggml_tensor* index_w_sparse =
                    ggml_is_contiguous(glm_index_weights) ? glm_index_weights : ggml_cont(ctx_c, glm_index_weights);
                struct ggml_tensor* index_k_sparse =
                    ggml_is_contiguous(glm_index_key) ? glm_index_key : ggml_cont(ctx_c, glm_index_key);

                KQV = ggml_glm_dsa_attention(ctx_c, q_sparse, k_sparse, v_sparse, index_q_sparse, index_w_sparse,
                                             index_k_sparse, ud);
            }

            // =========================================================================
            // KV CACHE INTEGRATION (Universal Paged Attention)
            // =========================================================================
            if (!KQV) {
                struct ggml_tensor* K_all = Kcur;  // Default to current K
                struct ggml_tensor* V_all = Vcur;  // Default to current V

                const bool use_cache = (cache != nullptr);
                const BasePagedDecodeExecutionDecision base_paged_decode = ResolveBasePagedDecodeExecutionDecision(
                    decode_paged_policy, model, cache, batch, N, n_head, n_head_kv, head_dim_q, head_dim_kv);
                const int n_past_val = base_paged_decode.n_past_val;
                const int attn_query_base_pos = debug_query_base_pos;
                attn_ref_n_past = attn_query_base_pos;
                const int n_total_tokens = n_past_val + N;
                const DecodePagedDecision paged_decode_decision = base_paged_decode.paged_decode_decision;
                const bool paged_decode_candidate = paged_decode_decision.candidate;
                const bool requested_paged_decode_attention = paged_decode_decision.requested;
                const bool decode_only_batch = base_paged_decode.decode_only_batch;
                const int kv_cache_layer = densecore::models::Gemma4KVSourceLayer(model, il);
                const bool gemma4_shared_kv_source_layer = IsGemma4SharedKVSourceLayer(model, il);
                const bool gemma4_shared_kv_explicit_state_disabled =
                    model->arch_flags.is_gemma4 && densecore::models::IsGemma4SharedKVExplicitStateDisabled();

                // Safety override: GGML's generic decode matmul path can become numerically
                // unstable for GQA decode (N=1, n_head != n_head_kv) on some CPU kernels.
                // Force the custom paged decode attention path for correctness in this case.
                const bool paged_decode_supported = base_paged_decode.paged_decode_supported;
                const bool disable_paged_decode_for_model = !paged_decode_supported;
                const bool force_safe_gqa_decode =
                    paged_decode_supported && IsForceSafeGqaDecodeEnabled() && use_cache && paged_decode_candidate &&
                    N == 1 && n_past_val > 0 && n_head_kv > 0 && n_head > n_head_kv && (n_head % n_head_kv == 0) &&
                    (head_dim_q == head_dim_kv) && batch.num_seqs == 1 && !batch.seq_id.empty();
                const bool force_batched_decode_path = base_paged_decode.force_batched_decode_path;
                bool use_paged_decode_attention = base_paged_decode.use_paged_decode_attention;
                if (disable_paged_decode_for_model && requested_paged_decode_attention) {
                    static bool logged_gemma4_paged_decode_disable = false;
                    if (!logged_gemma4_paged_decode_disable) {
                        std::cerr << "[DenseCore] Disabling paged decode attention for Gemma4; "
                                     "falling back to portable attention until the paged path "
                                     "supports Gemma4 softcap/sliding semantics."
                                  << std::endl;
                        logged_gemma4_paged_decode_disable = true;
                    }
                }
                if (force_batched_decode_path) {
                    const bool was_enabled = use_paged_decode_attention;
                    use_paged_decode_attention = true;
                    if (!was_enabled) {
                        static bool logged_force_batched_decode = false;
                        if (!logged_force_batched_decode) {
                            std::cerr
                                << "[DenseCore] Forcing paged decode attention for decode-only batched scheduling "
                                << "for sequence-isolated correctness " << "(N=" << N << ")" << std::endl;
                            logged_force_batched_decode = true;
                        }
                    }
                }
                if (decode_paged_policy.debug_log && il == 0) {
                    const char* mode = "auto";
                    if (decode_paged_policy.mode == DecodePagedAttentionMode::Off) mode = "off";
                    if (decode_paged_policy.mode == DecodePagedAttentionMode::On) mode = "on";
                    const int cache_type = cache ? static_cast<int>(cache->cache_type) : -1;
                    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, N);
                    std::cerr << "[PagedDecode] mode=" << mode << " candidate=" << (paged_decode_candidate ? "1" : "0")
                              << " supported=" << (paged_decode_supported ? "1" : "0")
                              << " use=" << (use_paged_decode_attention ? "1" : "0") << " N=" << N
                              << " n_past=" << n_past_val << " n_head=" << n_head << " n_head_kv=" << n_head_kv
                              << " head_dim_q=" << head_dim_q << " cache_type=" << cache_type
                              << " ctx_min=" << (context_summary.valid ? context_summary.min_context : -1)
                              << " ctx_avg=" << (context_summary.valid ? context_summary.avg_context : -1)
                              << " ctx_max=" << (context_summary.valid ? context_summary.max_context : -1)
                              << " reason=" << DecodePagedFallbackReasonName(paged_decode_decision.reason) << std::endl;
                    if (!paged_decode_supported && decode_only_batch && N > 1 && paged_decode_candidate) {
                        std::cerr << "[DecodeAttentionPath] N=" << N << " path=standard reason=model_veto_paged_decode"
                                  << std::endl;
                    }
                }

                if (use_cache && !use_paged_decode_attention) {
                    // Only need fancy logic if we have history.
                    // If n_past = 0 (Prefill), K_all == Kcur is mostly fine,
                    // BUT we still need to WRITE to cache.
                    // The 'ggml_pad' trick updates cache as side effect.
                    // So we act always if use_cache is true.
                    struct ggml_tensor* cache_k_src = Kcur;
                    struct ggml_tensor* cache_v_src = Vcur;
                    if (model->arch_flags.is_gemma4 && N == 1) {
                        // Single-token Gemma4 decode can reach this path with
                        // non-dense 3D views. Materialize the current K/V slice
                        // before cache write/gather so the source-layer publish
                        // path does not silently write zeros for the appended token.
                        cache_k_src = ggml_cont(ctx_c, Kcur);
                        cache_v_src = ggml_cont(ctx_c, Vcur);
                    }

                    KVCacheUserData* k_ud = GetKVCacheUserData(il, true);
                    *k_ud = {cache, kv_cache_layer, head_dim_kv, true};  // batch accessed via GetCurrentBatch()
                    k_ud->read_only_shared_kv = gemma4_shared_kv_layer;
                    k_ud->force_full_history = gemma4_shared_kv_source_layer;
                    KVCacheUserData* v_ud = GetKVCacheUserData(il, false);
                    *v_ud = {cache, kv_cache_layer, head_dim_v, false};  // batch accessed via GetCurrentBatch()
                    v_ud->read_only_shared_kv = gemma4_shared_kv_layer;
                    v_ud->force_full_history = gemma4_shared_kv_source_layer;

                    int kv_tasks = ResolveInferenceConfig(&batch).num_threads;
                    if (kv_tasks <= 0) {
                        kv_tasks = std::thread::hardware_concurrency();
                        if (kv_tasks <= 0) kv_tasks = 4;
                    }
                    int physical_cores = ResolveHardwareTopology(&batch).GetPhysicalCoreCount();
                    if (physical_cores > 0) {
                        kv_tasks = std::min(kv_tasks, physical_cores);
                    }
                    kv_tasks = std::max(1, kv_tasks);
                    kv_tasks = std::min(kv_tasks, std::max(1, n_head_kv));
                    {
                        // Escape hatch for platform-specific troubleshooting.
                        const char* env = std::getenv("DENSECORE_KV_CALLBACK_SINGLE_THREAD");
                        const bool force_single = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                        if (force_single) {
                            kv_tasks = 1;
                        }
                    }

                    if (gemma4_shared_kv_layer) {
                        if (gemma4_shared_kv_explicit_state_disabled) {
                            K_all = ggml_kv_update_and_gather(ctx_c, Kcur, n_total_tokens, kv_tasks, k_ud);
                            V_all = ggml_kv_update_and_gather(ctx_c, Vcur, n_total_tokens, kv_tasks, v_ud);
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, K_all, "cache-read", "K", il, kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, V_all, "cache-read", "V", il, kv_cache_layer);
                        } else {
                            const Gemma4SharedKVState* shared_kv_state = GetGemma4SharedKVState(kv_cache_layer);
                            if (!shared_kv_state) {
                                throw densecore::InvalidArgumentException(
                                    "Gemma4 shared-KV layer is missing per-forward shared_kv_states for its source layer");
                            }
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->k, "explicit-read", "K",
                                                                   il, kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->v, "explicit-read", "V",
                                                                   il, kv_cache_layer);
                        }
                    } else if (n_past_val == 0) {
                        // Prefill first chunk fast path: avoid materializing an
                        // equivalent [history | current] tensor when history is empty.
                        K_all = ggml_map_custom1(ctx_c, cache_k_src, cb_kv_write_only, kv_tasks, k_ud);
                        V_all = ggml_map_custom1(ctx_c, cache_v_src, cb_kv_write_only, kv_tasks, v_ud);
                    } else {
                        // Prefill later-chunk fast path: write current tokens to KV cache,
                        // gather retained history directly into the final [head_dim, n_head_kv, n_total]
                        // tensor, and append current tokens without routing through ggml_pad.
                        K_all = ggml_kv_update_and_gather(ctx_c, cache_k_src, n_total_tokens, kv_tasks, k_ud);
                        V_all = ggml_kv_update_and_gather(ctx_c, cache_v_src, n_total_tokens, kv_tasks, v_ud);
                    }
                }

                if (model->arch_flags.is_gemma4) {
                    if (gemma4_shared_kv_layer) {
                        if (!use_cache && !gemma4_shared_kv_explicit_state_disabled) {
                            const Gemma4SharedKVState* shared_kv_state = GetGemma4SharedKVState(kv_cache_layer);
                            if (!shared_kv_state) {
                                throw densecore::InvalidArgumentException(
                                    "Gemma4 shared-KV layer is missing per-forward shared_kv_states for its source layer");
                            }
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->k, "explicit-read", "K", il,
                                                                   kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->v, "explicit-read", "V", il,
                                                                   kv_cache_layer);
                        }
                    } else if (gemma4_shared_kv_source_layer) {
                        if (!gemma4_shared_kv_explicit_state_disabled) {
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, K_all, "publish", "K", il, il);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, V_all, "publish", "V", il, il);
                            SetGemma4SharedKVState(il, K_all, V_all);
                        }
                    }
                }

                // =========================================================================
                // NEW: Robust KV Cache Integration (Replaces ggml_pad approach)
                // =========================================================================
                // This approach explicitly:
                //   1. Allocates destination tensors with full size [head_dim, n_head_kv,
                //   n_total]
                //   2. Uses cb_kv_update_and_gather to write cache, read history, append
                //   current
                //   3. Does NOT rely on ggml_pad padding behavior which was causing
                //   NaN/hangs
                // =========================================================================
                if (false) {  // DISABLED: New approach has graph dependency bugs
                    // Step 1: Explicitly allocate K_all and V_all with full context size
                    // Shape: [head_dim_kv, n_head_kv, n_total_tokens]
                    struct ggml_tensor* K_all_tensor =
                        ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
                    struct ggml_tensor* V_all_tensor =
                        ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
                    ggml_set_name(K_all_tensor, "K_all");
                    ggml_set_name(V_all_tensor, "V_all");

                    // Step 2: Force Kcur/Vcur to be contiguous before passing data pointers
                    // This ensures src_data pointer is valid for memcpy in callback
                    struct ggml_tensor* Kcur_contig = ggml_cont(ctx_c, Kcur);
                    struct ggml_tensor* Vcur_contig = ggml_cont(ctx_c, Vcur);

                    // Step 3: Setup userdata for K with src_tensor pointer
                    // NOTE: src_tensor is set below after ggml_cont
                    //       The tensor pointer is stable; data is populated at graph
                    //       execution
                    KVUpdateGatherUserData* k_gather_ud = GetKVUpdateGatherUserData(il, true);
                    k_gather_ud->cache = cache;
                    k_gather_ud->batch = &batch;
                    k_gather_ud->layer = kv_cache_layer;
                    k_gather_ud->head_dim = head_dim_kv;
                    k_gather_ud->n_head_kv = n_head_kv;
                    k_gather_ud->N = N;
                    k_gather_ud->n_past = n_past_val;
                    k_gather_ud->is_k = true;
                    k_gather_ud->src_tensor = nullptr;  // Set below

                    // Step 4: Setup userdata for V
                    KVUpdateGatherUserData* v_gather_ud = GetKVUpdateGatherUserData(il, false);
                    v_gather_ud->cache = cache;
                    v_gather_ud->batch = &batch;
                    v_gather_ud->layer = kv_cache_layer;
                    v_gather_ud->head_dim = head_dim_kv;
                    v_gather_ud->n_head_kv = n_head_kv;
                    v_gather_ud->N = N;
                    v_gather_ud->n_past = n_past_val;
                    v_gather_ud->is_k = false;
                    v_gather_ud->src_tensor = nullptr;  // Set below

                    // Step 5: Create graph nodes that will execute the callbacks
                    // The src_data will be populated from the contiguous tensor's data
                    // pointer when the graph is executed (data is allocated by this point)
                    //
                    // WORKAROUND: ggml_map_custom1 passes its input tensor as 'src'.
                    // We need to pass BOTH the destination and source data.
                    // Solution: Store Kcur_contig as 'src' input, K_all_tensor->data is
                    // 'dst'
                    //
                    // The callback signature is: cb(dst, src, ith, nth, userdata)
                    // We set src_data = src->data in the callback if it's nullptr

                    // For K: Map from Kcur_contig, output shape matches K_all_tensor
                    // We need a custom callback wrapper that sets src_data from src tensor
                    // For now, we'll pass the contiguous tensor and handle in callback

                    // Actually, ggml_map_custom1(ctx, a, cb, n_tasks, userdata) creates:
                    //   result tensor with same shape as 'a'
                    //   callback receives: cb(result, a, ith, nth, userdata)
                    //
                    // So 'a' becomes 'src', and result is 'dst'
                    // We need result to have shape [head_dim_kv, n_head_kv, n_total_tokens]
                    // This means we should pass K_all_tensor as 'a', not Kcur!
                    //
                    // But then we need to access Kcur data via userdata.
                    // Since Kcur_contig->data is available at graph execution time,
                    // we can store its pointer now and it will be valid.

                    // CRITICAL FIX: The tensor data pointers are only valid AFTER
                    // ggml_backend allocates memory. At graph construction time, data may
                    // be nullptr. We need to access the data through the tensor pointer in
                    // the callback.

                    // Store tensor pointers in userdata (not raw data pointers)
                    // This requires modifying the struct to take ggml_tensor* instead of
                    // float* For now, we'll use a simpler workaround: pass Kcur_contig as
                    // input, create output tensor of correct size via ggml_new_tensor, then
                    // use ggml_cpy

                    // SIMPLER APPROACH: Use ggml_map_custom1 on K_all_tensor, pass Kcur as
                    // extra userdata Since Kcur_contig is built into the graph, its data
                    // pointer is stable
                    k_gather_ud->src_tensor = Kcur_contig;
                    v_gather_ud->src_tensor = Vcur_contig;

                    // Create the combined tensors via callback
                    K_all = ggml_map_custom1(ctx_c, K_all_tensor, cb_kv_update_and_gather, 1, k_gather_ud);
                    V_all = ggml_map_custom1(ctx_c, V_all_tensor, cb_kv_update_and_gather, 1, v_gather_ud);

                    // Mark as dependent on Kcur_contig and Vcur_contig for proper execution
                    // order
                    ggml_build_forward_expand(gf, Kcur_contig);
                    ggml_build_forward_expand(gf, Vcur_contig);
                }

                // After projection and reshape:

                // Q: [head_dim_q, n_head, N]
                // K_all: [head_dim_kv, n_head_kv, n_past + N]
                // V_all: [head_dim_kv, n_head_kv, n_past + N]

                // =========================================================================
                // GQA (Grouped Query Attention): LOGICAL BROADCASTING
                // =========================================================================
                // For models like Qwen3 where n_head != n_head_kv (e.g., 32 Q heads, 4 KV
                // heads):
                //
                // OLD APPROACH (REMOVED - caused segfaults and was inefficient):
                //   Used ggml_repeat to physically expand K/V from n_head_kv to n_head.
                //   This allocated 8x more memory and caused OOM/crashes.
                //
                // NEW APPROACH (Logical Broadcasting):
                //   Keep K/V at their original [head_dim, n_head_kv, seq] shape.
                //   The attention kernel computes: kv_head = query_head / (n_head /
                //   n_head_kv) This is zero-copy and memory-efficient.
                //
                // Both ggml_flash_attn_ext and our custom FlashAttentionGQA support this.
                // =========================================================================
                struct ggml_tensor* K = K_all;
                struct ggml_tensor* V = V_all;
                attn_ref_k = K;
                attn_ref_v = V;
                attn_core_reference_eligible = true;

                // Compute GQA repetition factor for attention dispatch
                const int n_rep = (n_head_kv > 0) ? (n_head / n_head_kv) : 1;
                (void)n_rep;  // Used in attention mask/kernel setup

                // =========================================================================
                // ATTENTION (llama.cpp style - corrected tensor layouts)
                // =========================================================================
                // Tensor shapes at this point:
                //   Q: [head_dim_q, n_head, N]
                //   K: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //   V: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //
                // For GQA: The attention kernel handles broadcasting internally.
                // Query heads [0, n_rep) all attend to KV head 0, etc.
                // =========================================================================

                // =========================================================================
                // ATTENTION DISPATCH (Runtime selection based on CPU capabilities)
                // - AVX-512+: Use Flash Attention (ggml_flash_attn_ext) for efficiency
                // - Other: Use standard Q*K^T -> softmax -> V for compatibility
                // =========================================================================
                // Note: Flash Attention still requires AVX-512-class x86 support for
                // correctness/perf in this path.
                const float fast_attn_logit_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
                const int fast_attn_sliding_window =
                    (model->arch_flags.is_gemma4 && densecore::models::IsGemma4SlidingLayer(model, il) &&
                     model->gemma4_sliding_window > 0)
                        ? model->gemma4_sliding_window
                        : -1;
                const bool fast_attn_requires_extended_semantics = fast_attn_logit_softcap > 0.0f;
                const uint32_t fast_attn_semantic_flags =
                    fast_attn_requires_extended_semantics ? kFastAttentionSemanticLogitSoftcap : 0u;
                const bool flash_attn_head_layout_supported =
                    !model->arch_flags.is_glm_dsa && (n_head_kv > 0) && (n_head % n_head_kv == 0) && (n_head % 8 == 0);
                const bool flash_attn_runtime_supported = flash_attn_head_layout_supported &&
                                                          densecore::OpsRegistry::IsInitialized() &&
                                                          IsFlashAttentionIsaSupported();
                const bool flash_attn_forced = IsFlashAttentionForced();
                const bool use_flash_attention = !fast_attn_requires_extended_semantics &&
                                                 !IsFlashAttentionDisabled() && flash_attn_runtime_supported;
                const densecore::DeviceType preferred_attention_device = ResolvePreferredAttentionDevice(&batch);
                // Non-CPU HAL backends do not currently carry the chunked-prefill query
                // offset into causal masking, so they remain limited to zero-offset
                // prefill or decode (N == 1).
                const bool hal_attention_offset_safe = (n_past_val == 0) || (N == 1);
                const bool use_hal_attention_dispatch = !fast_attn_requires_extended_semantics &&
                                                        preferred_attention_device != densecore::DeviceType::CPU &&
                                                        !use_paged_decode_attention && hal_attention_offset_safe &&
                                                        !model->arch_flags.is_glm_dsa;
                const bool portable_cpu_flash_attention_supported =
                    flash_attn_head_layout_supported && densecore::OpsRegistry::IsInitialized() &&
                    (IsPortableCpuFlashAttentionEnabled() || flash_attn_forced);
                const int debug_disable_fast_attn_from_layer = GetDebugDisableFastAttentionFromLayer();
                const bool debug_disable_fast_attn_for_layer =
                    debug_disable_fast_attn_from_layer >= 0 && il >= debug_disable_fast_attn_from_layer;
                const bool prefer_portable_cpu_flash_safe_decode =
                    force_safe_gqa_decode && preferred_attention_device == densecore::DeviceType::CPU &&
                    hal_attention_offset_safe && portable_cpu_flash_attention_supported;
                if (!use_paged_decode_attention && force_safe_gqa_decode && !prefer_portable_cpu_flash_safe_decode) {
                    use_paged_decode_attention = true;
                    static bool logged_force_safe_decode = false;
                    if (!logged_force_safe_decode) {
                        std::cerr << "[DenseCore] Forcing paged decode attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_force_safe_decode = true;
                    }
                } else if (!use_paged_decode_attention && prefer_portable_cpu_flash_safe_decode) {
                    static bool logged_safe_decode_flash = false;
                    if (!logged_safe_decode_flash) {
                        std::cerr << "[DenseCore] Using portable CPU flash attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_safe_decode_flash = true;
                    }
                }
                const bool use_portable_cpu_flash_attention =
                    !IsFlashAttentionDisabled() &&
                    (preferred_attention_device == densecore::DeviceType::CPU ||
                     fast_attn_requires_extended_semantics) &&
                    !use_paged_decode_attention && portable_cpu_flash_attention_supported &&
                    !debug_disable_fast_attn_for_layer;
                const bool use_portable_cpu_flash_native_decode_layout =
                    use_portable_cpu_flash_attention && N == 1 && head_dim_q == head_dim_kv &&
                    head_dim_q == head_dim_v && ggml_is_contiguous(Qcur) && ggml_is_contiguous(K) &&
                    ggml_is_contiguous(V) && !model->arch_flags.is_gemma4;
                if (flash_attn_forced && !flash_attn_runtime_supported && il == 0 &&
                    IsVerboseGraphBuildLoggingEnabled()) {
                    std::cerr
                        << "[DenseCore] DENSECORE_FORCE_FLASH_ATTN requested but native ggml flash is unavailable; "
                           "falling back to DenseCore portable flash attention or standard attention."
                        << std::endl;
                }

                if (decode_paged_policy.debug_log && il == 0 && decode_only_batch && N > 1) {
                    const char* path =
                        use_paged_decode_attention
                            ? "paged_decode"
                            : (use_hal_attention_dispatch         ? "hal_flash"
                               : use_portable_cpu_flash_attention ? "cpu_flash_hal"
                                                                  : (use_flash_attention ? "flash" : "standard"));
                    std::cerr << "[DecodeAttentionPath] N=" << N << " path=" << path << std::endl;
                }

                if (!use_paged_decode_attention) {
                    RecordDecodePagedFallbackReason(paged_decode_decision.reason, il, N);
                }

#ifndef NDEBUG
                if (model->arch_flags.is_gemma4 && use_paged_decode_attention) {
                    throw densecore::InvalidArgumentException(
                        "Decode attention invariant failed: Gemma4 must not execute paged decode attention.");
                }
#endif

                const DecodeAttentionPathKind attention_path_kind =
                    use_paged_decode_attention ? DecodeAttentionPathKind::Paged
                                               : (use_hal_attention_dispatch ? DecodeAttentionPathKind::Hal
                                                  : use_portable_cpu_flash_attention
                                                      ? DecodeAttentionPathKind::PortableCpuFlash
                                                      : (use_flash_attention ? DecodeAttentionPathKind::NativeFlash
                                                                             : DecodeAttentionPathKind::Standard));
                if (n_past_val > 0 || decode_only_batch || N == 1) {
                    RecordDecodeAttentionPath(attention_path_kind, il, N, n_past_val, n_head, n_head_kv,
                                              preferred_attention_device, use_portable_cpu_flash_native_decode_layout,
                                              paged_decode_candidate, use_paged_decode_attention,
                                              portable_cpu_flash_attention_supported, hal_attention_offset_safe);
                }

                if (use_paged_decode_attention) {
                    // -----------------------------------------------------------------------
                    // DECODE PAGED ATTENTION FAST PATH (decode-only batches, one token/seq)
                    // -----------------------------------------------------------------------
                    // Avoids materializing [n_total] K/V tensors each token.
                    // Writes current K/V to paged cache and reads history directly from cache.
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q_decode = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
                    struct ggml_tensor* K_decode = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
                    struct ggml_tensor* V_decode = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);

                    PagedAttentionUserData* ud = GetPagedAttentionUserData();
                    ud->cache = cache;
                    ud->layer = kv_cache_layer;
                    ud->head_dim = head_dim_q;
                    ud->v_head_dim = head_dim_v;
                    ud->n_head = n_head;
                    ud->epoch_started.store(0, std::memory_order_relaxed);
                    ud->epoch_done.store(0, std::memory_order_relaxed);
                    ud->kv_writers_done.store(0, std::memory_order_relaxed);
                    KQV = ggml_paged_attention_decode(ctx_c, Q_decode, K_decode, V_decode, ud);
                } else if (use_hal_attention_dispatch) {
                    // -----------------------------------------------------------------------
                    // HAL ATTENTION PATH (per-op-class mixed routing)
                    // -----------------------------------------------------------------------
                    // Dispatches attention to the preferred attention device while the rest
                    // of the graph can remain on a different primary backend.
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
                    struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                    struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
                    Q_hal = ggml_cont(ctx_c, Q_hal);
                    K_hal = ggml_cont(ctx_c, K_hal);
                    V_hal = ggml_cont(ctx_c, V_hal);

                    const float scale = model->arch_flags.is_gemma4
                                            ? 1.0f
                                            : (use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf((float)head_dim_q));
                    // For decode (N==1), causal=false is correct because K already contains
                    // only historical + current keys (no future positions).
                    const bool hal_causal = (N > 1);
                    KQV = ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv,
                                                   fast_attn_sliding_window, fast_attn_logit_softcap,
                                                   fast_attn_semantic_flags, il, preferred_attention_device);

                    // Convert [head_dim, N, n_head] -> [head_dim, n_head, N]
                    KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                } else if (use_portable_cpu_flash_attention) {
                    // -----------------------------------------------------------------------
                    // PORTABLE CPU FLASH ATTENTION PATH
                    // -----------------------------------------------------------------------
                    // On ARM/Apple CPU runtimes, route through DenseCore's backend-agnostic
                    // FlashAttention op instead of forcing the materialized standard path.
                    // -----------------------------------------------------------------------
                    const float scale = model->arch_flags.is_gemma4
                                            ? 1.0f
                                            : (use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf((float)head_dim_q));
                    const bool hal_causal = (N > 1);
                    const int flash_q_start_offset = attn_query_base_pos;
                    if (use_portable_cpu_flash_native_decode_layout) {
                        KQV = ggml_flash_attention_hal(
                            ctx_c, Qcur, K, V, scale, false, n_head_kv, fast_attn_sliding_window,
                            fast_attn_logit_softcap, fast_attn_semantic_flags, il, densecore::DeviceType::CPU,
                            HalAttentionTensorLayout::GgmlDimHeadSeq, flash_q_start_offset, 0);
                    } else {
                        struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
                        struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                        struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
                        Q_hal = ggml_cont(ctx_c, Q_hal);
                        K_hal = ggml_cont(ctx_c, K_hal);
                        V_hal = ggml_cont(ctx_c, V_hal);

                        KQV = ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv,
                                                       fast_attn_sliding_window, fast_attn_logit_softcap,
                                                       fast_attn_semantic_flags, il, densecore::DeviceType::CPU,
                                                       HalAttentionTensorLayout::HeadSeq, flash_q_start_offset, 0);

                        // Convert [head_dim, N, n_head] -> [head_dim, n_head, N]
                        KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                    }
                } else if (use_flash_attention) {
                    // -----------------------------------------------------------------------
                    // FLASH ATTENTION PATH (AVX-512 only)
                    // -----------------------------------------------------------------------
                    // ggml_flash_attn_ext natively supports GQA - it handles K/V with fewer
                    // heads than Q. The kernel internally computes: kv_head = query_head /
                    // n_rep
                    //
                    // Shapes: Q: [head_dim, N, n_head], K/V: [head_dim, n_total, n_head_kv]
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

                    // K/V: [head_dim, n_head_kv, n_total] -> [head_dim, n_total, n_head_kv]
                    struct ggml_tensor* K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                    struct ggml_tensor* V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

                    // Create mask [n_total, N_padded, 1, 1] as required by
                    // ggml_flash_attn_ext 0.0f = can attend, -INFINITY = cannot attend
                    // (masked)
                    int N_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);
                    struct ggml_tensor* KQ_mask = nullptr;
                    if (shared_prefill_flash_mask && shared_prefill_mask_n_total == n_total_tokens &&
                        shared_prefill_mask_n_padded == N_padded && shared_prefill_mask_n == N &&
                        shared_prefill_mask_n_past == n_past_val &&
                        shared_prefill_mask_sliding_window == fast_attn_sliding_window) {
                        KQ_mask = shared_prefill_flash_mask;
                    } else {
                        KQ_mask = BuildAttentionMaskTensor(ctx_c, n_total_tokens, N, attn_query_base_pos,
                                                           fast_attn_sliding_window, N_padded);

                        if (N > 1 && !decode_only_batch) {
                            shared_prefill_flash_mask = KQ_mask;
                            shared_prefill_mask_n_total = n_total_tokens;
                            shared_prefill_mask_n_padded = N_padded;
                            shared_prefill_mask_n = N;
                            shared_prefill_mask_n_past = n_past_val;
                            shared_prefill_mask_sliding_window = fast_attn_sliding_window;
                        }
                    }

                    // Ensure contiguity for Flash Attention
                    Q = ggml_cont(ctx_c, Q);
                    K_fa = ggml_cont(ctx_c, K_fa);
                    V_fa = ggml_cont(ctx_c, V_fa);

                    // Scale factor: Gemma4 uses scaling=1.0 per HF reference
                    // (q_norm/k_norm handle normalization, no 1/sqrt(d) needed)
                    const float scale = model->arch_flags.is_gemma4
                                            ? 1.0f
                                            : (use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf((float)head_dim_q));

                    // Flash Attention: fused Q*K^T, scale, mask, softmax, *V
                    // Result: [head_dim, N, n_head]
                    KQV = ggml_flash_attn_ext(ctx_c, Q, K_fa, V_fa, KQ_mask, scale, 0.0f, 0.0f);

                    // Permute to [head_dim, n_head, N] for projection
                    KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                } else {
                    // -----------------------------------------------------------------------
                    // STANDARD ATTENTION PATH (AVX2/Fallback) - Tiled GQA Implementation
                    // -----------------------------------------------------------------------
                    // For GQA models (n_head != n_head_kv), we use a TILED approach:
                    //   - Iterate over KV heads (h_kv = 0 to n_head_kv)
                    //   - For each KV head, process n_rep query heads together
                    //   - Use ggml_view to slice tensors without copying (O(1) memory)
                    //
                    // This avoids the massive memory bloat of ggml_repeat while maintaining
                    // correctness on all hardware (AVX2, SSE, etc.)
                    // -----------------------------------------------------------------------

                    // Scale factor: Gemma4 uses scaling=1.0 per HF reference
                    // (q_norm/k_norm handle normalization, no 1/sqrt(d) needed)
                    const float scale = model->arch_flags.is_gemma4
                                            ? 1.0f
                                            : (use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf((float)head_dim_q));

                    // =================================================================
                    // UNIFIED ATTENTION PATH (GQA + MHA)
                    // =================================================================
                    // GGML's mul_mat natively supports GQA broadcasting:
                    // when K has n_head_kv heads and Q has n_head heads (where
                    // n_head % n_head_kv == 0), mul_mat broadcasts K across
                    // query head groups automatically. This matches llama.cpp.
                    // =================================================================
                    {
                        const bool skip_attn_cont = (N > 1) && IsPrefillAttentionSkipContEnabled();
                        // Q: [head_dim, N, n_head]
                        struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

                        // K/V: [head_dim, n_total, n_head_kv]
                        struct ggml_tensor* K_att = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                        struct ggml_tensor* V_att = ggml_permute(ctx_c, V, 0, 2, 1, 3);

                        // ggml_mul_mat requires lhs (a) to be non-transposed. Always
                        // materialize K_att.
                        K_att = ggml_cont(ctx_c, K_att);
                        if (!skip_attn_cont) {
                            // Fully contiguous fallback path.
                            Q = ggml_cont(ctx_c, Q);
                            V_att = ggml_cont(ctx_c, V_att);
                        }

                        // Q @ K^T -> [n_total, N, n_head] (broadcasts n_head_kv → n_head)
                        struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_att, Q);

                        // Scale
                        KQ = ggml_scale(ctx_c, KQ, scale);

                        const float gemma4_attention_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
                        if (model->arch_flags.is_gemma4 && gemma4_attention_softcap > 0.0f) {
                            const float inv_softcap = 1.0f / gemma4_attention_softcap;
                            KQ = ggml_scale(ctx_c, KQ, inv_softcap);
                            KQ = ggml_tanh(ctx_c, KQ);
                            KQ = ggml_scale(ctx_c, KQ, gemma4_attention_softcap);
                        }

                        if (ShouldBuildExplicitStandardAttentionMask(N, fast_attn_sliding_window)) {
                            if (fast_attn_sliding_window >= 0) {
                                struct ggml_tensor* KQ_mask = BuildAttentionMaskTensor(
                                    ctx_c, n_total_tokens, N, attn_query_base_pos, fast_attn_sliding_window);
                                KQ = ggml_add(ctx_c, KQ, KQ_mask);
                            } else {
                                KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
                            }
                        }

                        // Softmax
                        KQ = ggml_soft_max(ctx_c, KQ);

                        // KQ @ V -> [head_dim, N, n_head]
                        // V needs transpose: [n_total, head_dim, n_head_kv]
                        struct ggml_tensor* V_t = ggml_permute(ctx_c, V_att, 1, 0, 2, 3);
                        // ggml_repeat on the CPU path requires a dense float source.
                        // Materialize the permuted V tensor before any GQA expansion.
                        V_t = ggml_cont(ctx_c, V_t);
                        if (n_head_kv > 0 && n_head > n_head_kv && (n_head % n_head_kv) == 0) {
                            struct ggml_tensor* V_repeat_shape =
                                ggml_new_tensor_3d(ctx_c, V_t->type, V_t->ne[0], V_t->ne[1], n_head);
                            V_t = ggml_repeat(ctx_c, V_t, V_repeat_shape);
                        }
                        // ggml_mul_mat requires lhs (a) to be non-transposed.
                        V_t = ggml_cont(ctx_c, V_t);
                        KQV = ggml_mul_mat(ctx_c, V_t, KQ);

                        // Permute to [head_dim, n_head, N] for projection
                        KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                    }
                }
            }

            // Must be contiguous before reshape
            struct ggml_tensor* KQV_merged = ggml_cont(ctx_c, KQV);
            if (attn_core_reference_eligible && attn_ref_k && attn_ref_v && IsDebugAttentionCoreReferenceEnabled()) {
                AttentionCoreReferenceUserData* attn_ref_ud = GetAttentionCoreReferenceUserData();
                const int attn_ref_sliding_window =
                    (model->arch_flags.is_gemma4 && densecore::models::IsGemma4SlidingLayer(model, il) &&
                     model->gemma4_sliding_window > 0)
                        ? model->gemma4_sliding_window
                        : -1;
                const float attn_ref_scale =
                    model->arch_flags.is_gemma4
                        ? 1.0f
                        : (use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf((float)head_dim_q));
                const float attn_ref_logit_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
                attn_ref_ud->value_tensor = attn_ref_v;
                attn_ref_ud->layer_idx = il;
                attn_ref_ud->n_head = n_head;
                attn_ref_ud->n_head_kv = n_head_kv;
                attn_ref_ud->head_dim_q = head_dim_q;
                attn_ref_ud->head_dim_k = head_dim_kv;
                attn_ref_ud->head_dim_v = head_dim_v;
                attn_ref_ud->n_past = attn_ref_n_past;
                attn_ref_ud->sliding_window = attn_ref_sliding_window;
                attn_ref_ud->attention_scale = attn_ref_scale;
                attn_ref_ud->logit_softcap = attn_ref_logit_softcap;
                attn_ref_ud->token_seq_ids = batch.seq_id.data();
                attn_ref_ud->stage = "attn_core";
                attn_ref_ud->var_name = "KQV_merged";
                KQV_merged = ggml_map_custom3(ctx_c, KQV_merged, Qcur, attn_ref_k, cb_attention_core_reference_probe, 1,
                                              attn_ref_ud);
            }
#ifdef DENSECORE_TEST_BUILD
            if (g_test_capture_attention_layer.load(std::memory_order_relaxed) == il) {
                KQV_merged = ggml_map_custom1(ctx_c, KQV_merged, cb_test_capture_attention_tensor, 1, nullptr);
            }
#endif
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_kqv = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[KQV%d #%d] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f "
                                "max=%.6f mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (int)src->type, (long)src->ne[0], (long)src->ne[1],
                                (long)src->ne[2], n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                KQV_merged = ggml_map_custom1(ctx_c, KQV_merged, cb_check_kqv, 1, &dbg_layers[il]);
            }

            const int attn_out_head_dim = static_cast<int>(KQV_merged->ne[0]);
            cur = ggml_reshape_2d(ctx_c, KQV_merged, attn_out_head_dim * n_head, N);
            if (attn_gate) {
                attn_gate = ggml_sigmoid(ctx_c, attn_gate);
                cur = ggml_mul(ctx_c, cur, attn_gate);
            }
            if (IsDebugAttentionPostReferenceEnabled()) {
                AttentionCoreReferenceUserData* post_ref_ud = GetAttentionCoreReferenceUserData();
                post_ref_ud->gate_tensor = attn_gate;
                post_ref_ud->layer_idx = il;
                post_ref_ud->n_head = n_head;
                post_ref_ud->head_dim_v = head_dim_v;
                post_ref_ud->token_seq_ids = batch.seq_id.data();
                post_ref_ud->stage = "attn_post";
                post_ref_ud->var_name = "cur_input_to_wo";
                cur = ggml_map_custom2(ctx_c, cur, KQV_merged, cb_attention_post_reference_probe, 1, post_ref_ud);
            }

            // Output Projection (using smart dispatcher for Parallel GEMV)
            struct ggml_tensor* cur_input_to_wo = cur;
            if (!wo) {
                throw densecore::InvalidArgumentException("Missing attn_output weight in TransformerLayer");
            }
            cur = smart_mul_mat(ctx_c, wo, cur, model);
            if (ShouldRunAttentionProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* o_ref_ud = GetProjectionReferenceUserData();
                o_ref_ud->weight_tensor = wo;
                o_ref_ud->input_tensor = cur_input_to_wo;
                o_ref_ud->layer_idx = il;
                o_ref_ud->token_seq_ids = batch.seq_id.data();
                o_ref_ud->stage = "attn_o_proj";
                o_ref_ud->var_name = "attn_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, o_ref_ud);
            }

            // Apply Multi-LoRA to Output Projection
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, cur_input_to_wo, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
            if (bo) cur = ggml_add(ctx_c, cur, bo);
            if (auto* post_attn_norm = model->layers[il].Get(model_keys::kPostAttnNorm);
                post_attn_norm && (model->arch_flags.is_gemma4 || post_attn_norm != ffn_norm)) {
                cur = apply_weighted_rms_norm(cur, post_attn_norm, "post_attention_norm", il);
            }

            // Residual Connection
            attn_out = cur;
            if (ShouldRunHiddenSnapshotProbe(il, "attn_out_pre_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "attn_out_pre_residual";
                    hidden_ud->var_name = "attn_out";
                    attn_out = ggml_map_custom1(ctx_c, attn_out, cb_hidden_snapshot_probe, 1, hidden_ud);
                    cur = attn_out;
                }
            }
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
            if (ShouldRunHiddenSnapshotProbe(il, "after_attn_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "after_attn_residual";
                    hidden_ud->var_name = "attn_post_residual";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_attn = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                        void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[ATTN%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f "
                                "mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct,
                                inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                cur = ggml_map_custom1(ctx_c, cur, cb_check_attn, 1, &dbg_layers[il]);
            }
            // Usually output projection expects n_embd input.
            // wo: [n_embd, n_embd] (or [n_embd, n_head*head_dim])
            // The standard transformer expects concatenation of all heads to be
            // n_embd. If n_head * head_dim_kv != n_embd, we have a mismatch. Qwen3
            // has n_head=16, head_dim_kv=128 => 2048 != 1024. This implies wo expects
            // 2048 input!

            // KQV = ggml_reshape_2d(ctx_c, KQV, n_head * head_dim_kv, N);

            // Output projection
            // cur = ggml_mul_mat(ctx_c, model->layers[il].wo, KQV);
            // if (model->layers[il].bo)
            //   cur = ggml_add(ctx_c, cur, model->layers[il].bo);

            // Residual connection
            // cur = ggml_add(ctx_c, cur, inpL);

        }  // end else (attention path)

        // =========================================================================
        // FFN (shared between SSM and attention layers)
        // =========================================================================
        struct ggml_tensor* inpFF = attn_post_residual;
        if (!ffn_norm) {
            throw densecore::InvalidArgumentException("Missing ffn_norm weight in TransformerLayer");
        }

        bool used_fused_pre_ffn_norm = false;
        if (IsFusedResidualRmsNormEnabled() && attn_out && inpL && attn_out->type == GGML_TYPE_F32 &&
            inpL->type == GGML_TYPE_F32 && ffn_norm->type == GGML_TYPE_F32 && attn_out->data && inpL->data &&
            ffn_norm->data) {
            AddRMSNormUserData* fused_ud = GetAddRMSNormUserData();
            fused_ud->residual = reinterpret_cast<const float*>(inpL->data);
            bind_add_rmsnorm_weight(fused_ud, ffn_norm, "ffn_norm");
            fused_ud->n_embd = static_cast<int>(attn_out->ne[0]);
            fused_ud->n_tokens = static_cast<int>(attn_out->ne[1]);
            fused_ud->eps = model->hparams.f_norm_rms_eps;
            fused_ud->residual_row_stride = static_cast<ptrdiff_t>(inpL->nb[1] / sizeof(float));
            fused_ud->layer_idx = il;
            fused_ud->token_seq_ids = batch.seq_id.data();
            fused_ud->stage = "pre_ffn_add_rmsnorm";
            fused_ud->var_name = "ffn_norm";

            const int n_tasks = ResolveTaskCount(&batch, std::max<int>(1, fused_ud->n_tokens));
            cur = ggml_map_custom1(ctx_c, attn_out, cb_residual_rmsnorm_fused, n_tasks, fused_ud);
            used_fused_pre_ffn_norm = true;
        }

        if (!used_fused_pre_ffn_norm) {
            cur = apply_weighted_rms_norm(cur, ffn_norm, "ffn_norm", il);
        }

        if (model->layers[il].is_moe) {
            // =====================================================================
            // MOE PATH
            // =====================================================================
            const bool is_gemma4_moe = densecore::models::IsGemma4MoEModel(model, &model->layers[il]);
            struct ggml_tensor* gemma_router_scale = model->layers[il].Get(kGemma4RouterScaleKey);
            struct ggml_tensor* gemma_pre_moe_norm = model->layers[il].Get(kGemma4PreMoeNormKey);
            struct ggml_tensor* gemma_post_shared_norm = model->layers[il].Get(kGemma4PostSharedNormKey);
            struct ggml_tensor* gemma_post_moe_norm = model->layers[il].Get(kGemma4PostMoeNormKey);
            struct ggml_tensor* shared_input = cur;
            struct ggml_tensor* routed_input = cur;
            if (is_gemma4_moe && gemma_pre_moe_norm) {
                routed_input =
                    apply_weighted_rms_norm(inpFF, gemma_pre_moe_norm, "gemma4_pre_feedforward_layernorm_2", il);
            }
            struct ggml_tensor* router_input = cur;
            if (is_gemma4_moe) {
                router_input = ggml_rms_norm(ctx_c, inpFF, model->hparams.f_norm_rms_eps);
                ggml_set_name(router_input, "gemma4_router_rms_norm");
                if (gemma_router_scale) {
                    struct ggml_tensor* router_scale = ggml_repeat(ctx_c, gemma_router_scale, router_input);
                    router_input = ggml_mul(ctx_c, router_input, router_scale);
                }
                router_input =
                    ggml_scale(ctx_c, router_input, 1.0f / std::sqrt(static_cast<float>(model->hparams.n_embd)));
            }

            // 1. Router: gate_logits = moe_gate * input
            if (!moe_gate) {
                throw densecore::InvalidArgumentException("Missing moe_gate weight in TransformerLayer");
            }
            struct ggml_tensor* gate_logits = smart_mul_mat(ctx_c, moe_gate, router_input, model);
            {
                char gate_name[64];
                std::snprintf(gate_name, sizeof(gate_name), "blk.%d.moe_gate_logits", il);
                ggml_set_name(gate_logits, gate_name);
            }
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* gate_ref_ud = GetProjectionReferenceUserData();
                gate_ref_ud->weight_tensor = moe_gate;
                gate_ref_ud->input_tensor = router_input;
                gate_ref_ud->layer_idx = il;
                gate_ref_ud->token_seq_ids = batch.seq_id.data();
                gate_ref_ud->stage = "moe_router_proj";
                gate_ref_ud->var_name = "gate_logits";
                if (const auto it = model->int4_weight_bindings.find(moe_gate);
                    it != model->int4_weight_bindings.end()) {
                    gate_ref_ud->int4_packed = reinterpret_cast<const uint8_t*>(it->second.packed->data);
                    gate_ref_ud->int4_scales = reinterpret_cast<const float*>(it->second.scales->data);
                    gate_ref_ud->int4_zeros = reinterpret_cast<const float*>(it->second.zeros->data);
                    gate_ref_ud->int4_group_size = it->second.group_size;
                    gate_ref_ud->int4_k = static_cast<int>(it->second.k);
                    gate_ref_ud->int4_n = static_cast<int>(it->second.n);
                } else if (const auto it = model->fp8_weight_bindings.find(moe_gate);
                           it != model->fp8_weight_bindings.end()) {
                    gate_ref_ud->fp8_packed = reinterpret_cast<const uint8_t*>(it->second.packed->data);
                    gate_ref_ud->fp8_format = it->second.format;
                    gate_ref_ud->fp8_k = static_cast<int>(it->second.k);
                    gate_ref_ud->fp8_n = static_cast<int>(it->second.n);
                }
                gate_logits = ggml_map_custom1(ctx_c, gate_logits, cb_projection_reference_probe, 1, gate_ref_ud);
            }

            // 2. Dispatch
            MoEUserData* moe_ud = AllocateMoEUserData(ctx_c);
            if (moe_ud) {
                moe_ud->model = model;
                moe_ud->layer = &model->layers[il];
                moe_ud->layer_idx = il;
                moe_ud->k = model->hparams.n_experts_used;
                moe_ud->batch = &batch;
                moe_ud->scheduler = batch.scheduler;
                // Resolve preferred device first, then guarantee CPU fallback for MoE.
                densecore::BackendRegistry& registry = ResolveBackendRegistry(&batch);
                densecore::ComputeBackend* preferred_backend = registry.Get(ResolvePreferredDevice(&batch));
                moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(preferred_backend);
                if (!moe_ud->backend) {
                    moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(registry.Get(densecore::DeviceType::CPU));
                }
                if (!moe_ud->backend) {
                    moe_ud->backend = &densecore::GetTelemetryCpuBackend();
                }
                if (moe_ud->backend) {
                    const densecore::CpuBackend::ExpertWeights* registered_experts = nullptr;
                    int registered_count = 0;
                    if (!moe_ud->backend->GetRegisteredExpertsView(moe_ud->layer, &registered_experts,
                                                                   &registered_count) ||
                        !registered_experts || registered_count <= 0) {
                        auto experts = BuildExpertWeights(moe_ud->layer, model);
                        const int n_experts = static_cast<int>(experts.size());
                        moe_ud->backend->InitMoEProfiler(moe_ud->layer, n_experts);
                        moe_ud->backend->RegisterMoEExperts(moe_ud->layer, experts);
                        moe_ud->backend->GetRegisteredExpertsView(moe_ud->layer, &registered_experts,
                                                                  &registered_count);
                    }
                    EnsureMoERebalanceThread(moe_ud->backend);
                    if (registered_experts && registered_count > 0) {
                        moe_ud->experts = registered_experts;
                        moe_ud->n_experts = registered_count;
                        moe_ud->experts_registered = true;
                    }
                }
            }

            // Use map_custom2: src0=cur, src1=gate_logits
            // Use 1 task (single thread dispatch, threaded inside backend)
            g_moe_graph_wiring_debug_counter.fetch_add(1, std::memory_order_relaxed);
            cur = ggml_map_custom2(ctx_c, routed_input, gate_logits, cb_moe_forward, 1, moe_ud);
            {
                char moe_name[64];
                std::snprintf(moe_name, sizeof(moe_name), "blk.%d.moe_forward", il);
                ggml_set_name(cur, moe_name);
            }

            // Shared-expert branch runs in parallel with routed experts.
            if (model->moe_n_shared_experts > 0 && ffn_gate && ffn_up && ffn_down) {
                if (!is_gemma4_moe && !ffn_shared_gate) {
                    throw densecore::InvalidArgumentException("Missing shared expert gate weight in TransformerLayer");
                }
                struct ggml_tensor* shared_gate = smart_mul_mat(ctx_c, ffn_gate, shared_input, model);
                struct ggml_tensor* shared_up = smart_mul_mat(ctx_c, ffn_up, shared_input, model);
                if (ShouldRunFfnProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* shared_gate_ref_ud = GetProjectionReferenceUserData();
                    shared_gate_ref_ud->weight_tensor = ffn_gate;
                    shared_gate_ref_ud->input_tensor = shared_input;
                    shared_gate_ref_ud->layer_idx = il;
                    shared_gate_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_gate_ref_ud->stage = "shared_expert_gate_proj";
                    shared_gate_ref_ud->var_name = "shared_gate";
                    shared_gate =
                        ggml_map_custom1(ctx_c, shared_gate, cb_projection_reference_probe, 1, shared_gate_ref_ud);

                    ProjectionReferenceUserData* shared_up_ref_ud = GetProjectionReferenceUserData();
                    shared_up_ref_ud->weight_tensor = ffn_up;
                    shared_up_ref_ud->input_tensor = shared_input;
                    shared_up_ref_ud->layer_idx = il;
                    shared_up_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_up_ref_ud->stage = "shared_expert_up_proj";
                    shared_up_ref_ud->var_name = "shared_up";
                    shared_up = ggml_map_custom1(ctx_c, shared_up, cb_projection_reference_probe, 1, shared_up_ref_ud);
                }
                struct ggml_tensor* shared_ffn = nullptr;
                if (is_gemma4_moe) {
                    // Match llama.cpp Gemma4 path: use native GEGLU so strided matmul
                    // outputs do not rely on custom flat-memory assumptions.
                    shared_ffn = ggml_geglu_split(ctx_c, shared_gate, shared_up);
                } else {
                    shared_ffn =
                        ggml_map_custom2(ctx_c, shared_gate, shared_up, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
                }
                struct ggml_tensor* shared_ffn_input = shared_ffn;
                shared_ffn = smart_mul_mat(ctx_c, ffn_down, shared_ffn, model);
                if (ShouldRunFfnProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* shared_down_ref_ud = GetProjectionReferenceUserData();
                    shared_down_ref_ud->weight_tensor = ffn_down;
                    shared_down_ref_ud->input_tensor = shared_ffn_input;
                    shared_down_ref_ud->layer_idx = il;
                    shared_down_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_down_ref_ud->stage = "shared_expert_down_proj";
                    shared_down_ref_ud->var_name = "shared_ffn";
                    shared_ffn =
                        ggml_map_custom1(ctx_c, shared_ffn, cb_projection_reference_probe, 1, shared_down_ref_ud);
                }
                if (is_gemma4_moe) {
                    if (gemma_post_shared_norm) {
                        shared_ffn = apply_weighted_rms_norm(shared_ffn, gemma_post_shared_norm,
                                                             "gemma4_post_feedforward_layernorm_1", il);
                    }
                    if (gemma_post_moe_norm) {
                        cur = apply_weighted_rms_norm(cur, gemma_post_moe_norm, "gemma4_post_feedforward_layernorm_2",
                                                      il);
                    }
                } else {
                    struct ggml_tensor* shared_gate_logits = smart_mul_mat(ctx_c, ffn_shared_gate, shared_input, model);
                    struct ggml_tensor* shared_gate_logits_scalar = shared_gate_logits;
                    if (ShouldRunFfnProjectionReferenceProbe(il)) {
                        ProjectionReferenceUserData* shared_scalar_gate_ref_ud = GetProjectionReferenceUserData();
                        shared_scalar_gate_ref_ud->weight_tensor = ffn_shared_gate;
                        shared_scalar_gate_ref_ud->input_tensor = shared_input;
                        shared_scalar_gate_ref_ud->layer_idx = il;
                        shared_scalar_gate_ref_ud->token_seq_ids = batch.seq_id.data();
                        shared_scalar_gate_ref_ud->stage = "shared_expert_scalar_gate_proj";
                        shared_scalar_gate_ref_ud->var_name = "shared_gate_logits";
                        if (const auto it = model->int4_weight_bindings.find(ffn_shared_gate);
                            it != model->int4_weight_bindings.end()) {
                            shared_scalar_gate_ref_ud->int4_packed =
                                reinterpret_cast<const uint8_t*>(it->second.packed->data);
                            shared_scalar_gate_ref_ud->int4_scales =
                                reinterpret_cast<const float*>(it->second.scales->data);
                            shared_scalar_gate_ref_ud->int4_zeros =
                                reinterpret_cast<const float*>(it->second.zeros->data);
                            shared_scalar_gate_ref_ud->int4_group_size = it->second.group_size;
                            shared_scalar_gate_ref_ud->int4_k = static_cast<int>(it->second.k);
                            shared_scalar_gate_ref_ud->int4_n = static_cast<int>(it->second.n);
                        } else if (const auto it = model->fp8_weight_bindings.find(ffn_shared_gate);
                                   it != model->fp8_weight_bindings.end()) {
                            shared_scalar_gate_ref_ud->fp8_packed =
                                reinterpret_cast<const uint8_t*>(it->second.packed->data);
                            shared_scalar_gate_ref_ud->fp8_format = it->second.format;
                            shared_scalar_gate_ref_ud->fp8_k = static_cast<int>(it->second.k);
                            shared_scalar_gate_ref_ud->fp8_n = static_cast<int>(it->second.n);
                        }
                        shared_gate_logits = ggml_map_custom1(ctx_c, shared_gate_logits, cb_projection_reference_probe,
                                                              1, shared_scalar_gate_ref_ud);
                    }
                    struct ggml_tensor* shared_ffn_pre_scalar_gate = shared_ffn;
                    SharedScalarGateApplyUserData* shared_gate_apply_ud = GetSharedScalarGateApplyUserData();
                    shared_gate_apply_ud->gate_logits_scalar = shared_gate_logits_scalar;
                    const int shared_gate_tasks = ResolveTaskCount(
                        &batch, std::max<int>(1, static_cast<int>(std::max<int64_t>(1, shared_ffn->ne[1]))));
                    shared_ffn = ggml_map_custom1(ctx_c, shared_ffn, cb_apply_shared_scalar_gate, shared_gate_tasks,
                                                  shared_gate_apply_ud);
                    if (ShouldRunSharedScalarGateReferenceProbe(il)) {
                        SharedScalarGateReferenceUserData* shared_gate_runtime_ud =
                            GetSharedScalarGateReferenceUserData();
                        shared_gate_runtime_ud->shared_ffn_pre_gate = shared_ffn_pre_scalar_gate;
                        shared_gate_runtime_ud->shared_gate_logits_scalar = shared_gate_logits_scalar;
                        shared_gate_runtime_ud->layer_idx = il;
                        shared_gate_runtime_ud->token_seq_ids = batch.seq_id.data();
                        shared_gate_runtime_ud->stage = "shared_expert_branch";
                        shared_gate_runtime_ud->var_name = "shared_ffn_after_scalar_gate";
                        shared_ffn = ggml_map_custom1(ctx_c, shared_ffn, cb_shared_scalar_gate_reference_probe, 1,
                                                      shared_gate_runtime_ud);
                    }
                }
                cur = ggml_add(ctx_c, cur, shared_ffn);
            }
        } else {
            // =====================================================================
            // DENSE PATH
            // =====================================================================
            // SwiGLU FFN (using smart dispatcher for INT4 support)
            if (!ffn_gate || !ffn_up || !ffn_down) {
                throw densecore::InvalidArgumentException("Missing FFN weights in TransformerLayer");
            }
            struct ggml_tensor* w1 = smart_mul_mat(ctx_c, ffn_gate, cur, model);
            struct ggml_tensor* w3 = smart_mul_mat(ctx_c, ffn_up, cur, model);
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* w1_ref_ud = GetProjectionReferenceUserData();
                w1_ref_ud->weight_tensor = ffn_gate;
                w1_ref_ud->input_tensor = cur;
                w1_ref_ud->layer_idx = il;
                w1_ref_ud->token_seq_ids = batch.seq_id.data();
                w1_ref_ud->stage = "ffn_gate_proj";
                w1_ref_ud->var_name = "w1";
                w1 = ggml_map_custom1(ctx_c, w1, cb_projection_reference_probe, 1, w1_ref_ud);

                ProjectionReferenceUserData* w3_ref_ud = GetProjectionReferenceUserData();
                w3_ref_ud->weight_tensor = ffn_up;
                w3_ref_ud->input_tensor = cur;
                w3_ref_ud->layer_idx = il;
                w3_ref_ud->token_seq_ids = batch.seq_id.data();
                w3_ref_ud->stage = "ffn_up_proj";
                w3_ref_ud->var_name = "w3";
                w3 = ggml_map_custom1(ctx_c, w3, cb_projection_reference_probe, 1, w3_ref_ud);
            }

            // Apply Multi-LoRA [FFN Gate/Up]
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate", il);
                ggml_set_name(w1, name_buf);
                w1 = ggml_map_custom2(ctx_c, w1, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());

                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up", il);
                ggml_set_name(w3, name_buf);
                w3 = ggml_map_custom2(ctx_c, w3, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }

            // Fused SiLU×Mul: silu(w1) * w3 in single pass (avoids intermediate tensor)
            // This saves ~50% memory bandwidth in FFN forward pass.
            if (model->arch_flags.is_gemma4) {
                // Match llama.cpp Gemma4 path: use native GEGLU so strided matmul
                // outputs do not rely on custom flat-memory assumptions.
                cur = ggml_geglu_split(ctx_c, w1, w3);
            } else {
                cur = ggml_map_custom2(ctx_c, w1, w3, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
            }
            ggml_set_name(cur, "ffn_silu_mul_fused");

            // Apply Multi-LoRA [FFN Down]
            struct ggml_tensor* ffn_input = cur;
            cur = smart_mul_mat(ctx_c, ffn_down, cur, model);
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* down_ref_ud = GetProjectionReferenceUserData();
                down_ref_ud->weight_tensor = ffn_down;
                down_ref_ud->input_tensor = ffn_input;
                down_ref_ud->layer_idx = il;
                down_ref_ud->token_seq_ids = batch.seq_id.data();
                down_ref_ud->stage = "ffn_down_proj";
                down_ref_ud->var_name = "ffn_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, down_ref_ud);
            }
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, ffn_input, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
        }

        if (auto* gemma_post_ffn_norm = model->layers[il].Get(kGemma4PostFfnNormKey)) {
            cur = apply_weighted_rms_norm(cur, gemma_post_ffn_norm, "gemma4_post_feedforward_layernorm", il);
        }

        // Residual connection
        cur = ggml_add(ctx_c, cur, inpFF);
        if (ShouldRunHiddenSnapshotProbe(il, "after_ffn_residual")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "after_ffn_residual";
                hidden_ud->var_name = "layer_output";
                cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        if (model->arch_flags.is_gemma4 && !gemma4_decode_special_transforms_disabled &&
            !densecore::models::IsGemma4PerLayerInputDisabled() && gemma4_per_layer_inputs &&
            model->gemma4_hidden_size_per_layer_input > 0) {
            auto* per_layer_gate = model->layers[il].Get(model_keys::kGemma4PerLayerInputGate);
            auto* per_layer_proj = model->layers[il].Get(model_keys::kGemma4PerLayerProjection);
            auto* post_per_layer_norm = model->layers[il].Get(model_keys::kGemma4PostPerLayerInputNorm);
            if (per_layer_gate && per_layer_proj && post_per_layer_norm) {
                const int hidden_per_layer = model->gemma4_hidden_size_per_layer_input;
                struct ggml_tensor* per_layer_input =
                    ggml_view_2d(ctx_c, gemma4_per_layer_inputs, hidden_per_layer, N, gemma4_per_layer_inputs->nb[2],
                                 static_cast<size_t>(il) * gemma4_per_layer_inputs->nb[1]);
                struct ggml_tensor* per_layer_delta = smart_mul_mat(ctx_c, per_layer_gate, cur, model);
                per_layer_delta = ggml_gelu(ctx_c, per_layer_delta);
                per_layer_delta = ggml_mul(ctx_c, per_layer_delta, per_layer_input);
                per_layer_delta = smart_mul_mat(ctx_c, per_layer_proj, per_layer_delta, model);
                per_layer_delta = apply_weighted_rms_norm(per_layer_delta, post_per_layer_norm,
                                                          "gemma4_post_per_layer_input_norm", il);
                cur = ggml_add(ctx_c, cur, per_layer_delta);
                if (ShouldRunHiddenSnapshotProbe(il, "after_per_layer_input")) {
                    auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                    if (hidden_ud) {
                        hidden_ud->layer_idx = il;
                        hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                        hidden_ud->token_seq_ids = batch.seq_id.data();
                        hidden_ud->stage = "after_per_layer_input";
                        hidden_ud->var_name = "layer_output";
                        cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                    }
                }
            }
        }
        if (model->arch_flags.is_gemma4 && !gemma4_decode_special_transforms_disabled &&
            !densecore::models::IsGemma4LayerOutputScaleDisabled()) {
            if (auto* layer_output_scale = model->layers[il].Get(model_keys::kGemma4LayerOutputScale)) {
                // Gemma4 checkpoint scalar: scale the full layer output (same as llama.cpp).
                // This matches the training-time behavior where subsequent layers see
                // the scaled hidden state as their input.
                struct ggml_tensor* scale = ggml_repeat(ctx_c, layer_output_scale, cur);
                cur = ggml_mul(ctx_c, cur, scale);
                if (ShouldRunHiddenSnapshotProbe(il, "after_layer_output_scale")) {
                    auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                    if (hidden_ud) {
                        hidden_ud->layer_idx = il;
                        hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                        hidden_ud->token_seq_ids = batch.seq_id.data();
                        hidden_ud->stage = "after_layer_output_scale";
                        hidden_ud->var_name = "layer_output";
                        cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                    }
                }
            }
        }
        if (IsDebugInferenceStatsEnabled()) {
            auto cb_check_layer = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                     void* ud) {
                (void)nth;
                if (ith != 0) return;
                const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                if (layer_idx < 0 || layer_idx >= 128) return;

                static int cb_ct[128] = {0};
                const float* d = reinterpret_cast<const float*>(src->data);
                const int n = ggml_nelements(src);
                if (!d || n <= 0) {
                    if (dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                    return;
                }

                int zero_ct = 0;
                int nan_ct = 0;
                int inf_ct = 0;
                float mn = std::numeric_limits<float>::infinity();
                float mx = -std::numeric_limits<float>::infinity();
                double sum = 0.0;
                double sum_sq = 0.0;
                int finite_ct = 0;
                for (int i = 0; i < n; i++) {
                    const float v = d[i];
                    if (std::isnan(v)) {
                        nan_ct++;
                        continue;
                    }
                    if (!std::isfinite(v)) {
                        inf_ct++;
                        continue;
                    }
                    if (v == 0.0f) zero_ct++;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    finite_ct++;
                }

                // Keep early-layer shape/stats visibility while always surfacing NaN/Inf.
                const bool emit_regular = (cb_ct[layer_idx] < 2);
                const bool emit_anomaly = (nan_ct > 0 || inf_ct > 0);
                if (emit_regular || emit_anomaly) {
                    if (!std::isfinite(mn)) mn = 0.0f;
                    if (!std::isfinite(mx)) mx = 0.0f;
                    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                    fprintf(stderr,
                            "[LAYER%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f "
                            "rms=%.6f\n",
                            layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct, inf_ct,
                            mn, mx, mean, rms);
                }

                if (dst->data && src->data) {
                    memcpy(dst->data, src->data, ggml_nbytes(src));
                }
                cb_ct[layer_idx]++;
            };
            static int layers[128] = {};
            for (int li = 0; li < 128; ++li) layers[li] = li;
            void* layer_ud = static_cast<void*>(&layers[il]);
            cur = ggml_map_custom1(ctx_c, cur, cb_check_layer, 1, layer_ud);
        }
    }
    if (moe_wiring_debug) {
        std::fprintf(stderr,
                     "[MOE_WIRING_SUMMARY] reason_counts={wired:%llu,model_no_moe:%llu,layer_flag_false:%llu,"
                     "missing_moe_gate:%llu,no_experts:%llu,dense_replace_gate:%llu,layer_flag_mismatch:%llu}\n",
                     static_cast<unsigned long long>(moe_wiring_reason_counts[0]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[1]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[2]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[3]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[4]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[5]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[6]));
    }

    // =========================================================================
    // 3. Final Layer Norm and LM Head
    // =========================================================================
    if (IsDebugInferenceStatsEnabled()) {
        // TEMP: check raw hidden BEFORE output_norm
        auto cb_pre_norm = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* ud) {
            if (ith != 0) return;
            static int ct = 0;
            if (ct < 2 && src && src->data && src->nb[0] == sizeof(float)) {
                const int ne0 = (int)src->ne[0], ne1 = (int)src->ne[1];
                const float* d = (const float*)src->data;
                fprintf(stderr, "[PRE_NORM #%d] shape=[%d,%d]\n", ct, ne0, ne1);
                for (int ti : {0, ne1 - 1}) {
                    if (ti < 0 || ti >= ne1) continue;
                    const float* row = d + ti * ne0;
                    float mn = row[0], mx = row[0];
                    double sq = 0.0;
                    for (int i = 0; i < ne0; i++) {
                        if (row[i] < mn) mn = row[i];
                        if (row[i] > mx) mx = row[i];
                        sq += (double)row[i] * row[i];
                    }
                    fprintf(stderr, "  [PRE_NORM_TOK %d] min=%.4f max=%.4f rms=%.4f first4=[%.3f,%.3f,%.3f,%.3f]\n", ti,
                            mn, mx, std::sqrt(sq / ne0), row[0], row[1], row[2], row[3]);
                }
                ct++;
            }
            if (dst && src && dst->data && src->data) memcpy(dst->data, src->data, ggml_nbytes(src));
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_pre_norm, 1, nullptr);
    }
    if (ShouldRunHiddenSnapshotProbe(static_cast<int>(model->layers.size()), "pre_output_norm_hidden")) {
        auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
        if (hidden_ud) {
            hidden_ud->layer_idx = static_cast<int>(model->layers.size());
            hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
            hidden_ud->token_seq_ids = batch.seq_id.data();
            hidden_ud->stage = "pre_output_norm_hidden";
            hidden_ud->var_name = "cur";
            cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
        }
    }
    cur = apply_weighted_rms_norm(cur, model->output_norm, "output_norm", static_cast<int>(model->layers.size()));

    if (embedding_mode) {
        return cur;
    }

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check hidden state before lm_head
        auto cb_check_hidden = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* ud) {
            if (ith != 0) return;
            static int cb_ct = 0;
            const bool emit = (cb_ct < 5);
            if (emit && src && src->data) {
                const float* d = (const float*)src->data;
                int n = ggml_nelements(src);
                int zero_ct = 0;
                float mn = d[0], mx = d[0];
                double sum_sq = 0.0;
                for (int i = 0; i < n; i++) {
                    if (d[i] == 0.0f)
                        zero_ct++;
                    else {
                        if (d[i] < mn) mn = d[i];
                        if (d[i] > mx) mx = d[i];
                    }
                    sum_sq += (double)d[i] * d[i];
                }
                double rms_all = std::sqrt(sum_sq / n);
                fprintf(stderr, "[HIDDEN #%d] shape=[%ld,%ld] total=%d zero=%d min=%.6f max=%.6f rms=%.6f\n", cb_ct,
                        (long)src->ne[0], (long)src->ne[1], n, zero_ct, mn, mx, rms_all);
                // Per-token stats: show first, last-1, and last token
                if (src->nb[0] == sizeof(float) && src->ne[1] > 1) {
                    const int n_embd = (int)src->ne[0];
                    const int n_tok = (int)src->ne[1];
                    for (int ti : {0, n_tok - 2, n_tok - 1}) {
                        if (ti < 0 || ti >= n_tok) continue;
                        const float* row = d + ti * n_embd;
                        float rmn = row[0], rmx = row[0];
                        double rsq = 0.0;
                        double sum = 0.0;
                        int pos_cnt = 0;
                        for (int j = 0; j < n_embd; j++) {
                            if (row[j] < rmn) rmn = row[j];
                            if (row[j] > rmx) rmx = row[j];
                            rsq += (double)row[j] * row[j];
                            sum += row[j];
                            if (row[j] > 0) pos_cnt++;
                        }
                        fprintf(stderr,
                                "  [HIDDEN_TOK %d] min=%.6f max=%.6f rms=%.6f mean=%.6f pos=%d/%d "
                                "first8=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n",
                                ti, rmn, rmx, std::sqrt(rsq / n_embd), sum / n_embd, pos_cnt, n_embd, row[0], row[1],
                                row[2], row[3], row[4], row[5], row[6], row[7]);
                    }
                }
                cb_ct++;
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_hidden, 1, nullptr);
    }
    if (ShouldRunHiddenSnapshotProbe(static_cast<int>(model->layers.size()), "pre_lm_head_hidden")) {
        auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
        if (hidden_ud) {
            hidden_ud->layer_idx = static_cast<int>(model->layers.size());
            hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
            hidden_ud->token_seq_ids = batch.seq_id.data();
            hidden_ud->stage = "pre_lm_head_hidden";
            hidden_ud->var_name = "cur";
            cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
        }
    }

    // LM Head projection: [n_embd, N] -> [n_vocab, N]
    struct ggml_tensor* cur_input_to_lm_head = cur;
    cur = smart_mul_mat(ctx_c, model->output, cur, model);
    const bool debug_lm_head = IsDebugInferenceStatsEnabled() || std::getenv("DENSECORE_DEBUG_LM_HEAD_TOP") != nullptr;
    if (debug_lm_head) {
        auto cb_check_lm_head_out = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* ud) {
            (void)nth;
            (void)ud;
            if (ith != 0) return;
            static int cb_lm_ct = 0;
            if (src && src->data) {
                const int n_vocab = (int)src->ne[0];
                const int n_tok = (int)src->ne[1];
                const ptrdiff_t row_stride = (ptrdiff_t)(src->nb[1] / sizeof(float));
                const float* d = (const float*)src->data;
                const bool debug_stats = IsDebugInferenceStatsEnabled();
                const char* top_env = std::getenv("DENSECORE_DEBUG_LM_HEAD_TOP");
                int top_n = 0;
                if (top_env && *top_env) {
                    char* end = nullptr;
                    long parsed = std::strtol(top_env, &end, 10);
                    top_n =
                        (end == top_env || (end && *end != '\0') || parsed <= 0) ? 8 : (int)std::min<long>(parsed, 32);
                }
                if (debug_stats && cb_lm_ct < 3) {
                    for (int ti : {0, n_tok - 1}) {
                        if (ti < 0 || ti >= n_tok) continue;
                        const float* row = d + (ptrdiff_t)ti * row_stride;
                        float mn = row[0], mx = row[0];
                        int pos_ct = 0;
                        for (int i = 0; i < n_vocab; i++) {
                            if (row[i] < mn) mn = row[i];
                            if (row[i] > mx) mx = row[i];
                            if (row[i] > 0) pos_ct++;
                        }
                        fprintf(stderr, "[LM_HEAD_OUT #%d] tok=%d/%d vocab=%d min=%.4f max=%.4f pos_ct=%d stride=%ld\n",
                                cb_lm_ct, ti, n_tok, n_vocab, mn, mx, pos_ct, (long)row_stride);
                    }
                    cb_lm_ct++;
                }
                if (top_n > 0 && n_tok > 0) {
                    const float* row = d + (ptrdiff_t)(n_tok - 1) * row_stride;
                    std::vector<std::pair<float, int>> top;
                    top.reserve((size_t)top_n);
                    auto worse_first = [](const auto& a, const auto& b) {
                        if (a.first == b.first) return a.second < b.second;
                        return a.first > b.first;
                    };
                    for (int i = 0; i < n_vocab; ++i) {
                        const float v = row[i];
                        if (!std::isfinite(v)) continue;
                        if ((int)top.size() < top_n) {
                            top.emplace_back(v, i);
                            std::push_heap(top.begin(), top.end(), worse_first);
                        } else if (v > top.front().first || (v == top.front().first && i < top.front().second)) {
                            std::pop_heap(top.begin(), top.end(), worse_first);
                            top.back() = {v, i};
                            std::push_heap(top.begin(), top.end(), worse_first);
                        }
                    }
                    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
                        if (a.first == b.first) return a.second < b.second;
                        return a.first > b.first;
                    });
                    fprintf(stderr, "[LM_HEAD_TOP] tok=%d/%d top_n=%d\n", n_tok - 1, n_tok, top_n);
                    for (const auto& [score, token_id] : top) {
                        fprintf(stderr, "  [TOP] token=%d logit=%.6f\n", token_id, score);
                    }
                    const char* watch_env = std::getenv("DENSECORE_DEBUG_LM_HEAD_WATCH_IDS");
                    if (watch_env && *watch_env) {
                        std::string spec(watch_env);
                        size_t start = 0;
                        while (start < spec.size()) {
                            size_t end = spec.find(',', start);
                            if (end == std::string::npos) end = spec.size();
                            std::string piece = spec.substr(start, end - start);
                            char* parse_end = nullptr;
                            long parsed = std::strtol(piece.c_str(), &parse_end, 10);
                            if (parse_end != piece.c_str() && (!parse_end || *parse_end == '\0') && parsed >= 0 &&
                                parsed < n_vocab) {
                                fprintf(stderr, "  [WATCH] token=%ld logit=%.6f\n", parsed, row[parsed]);
                            }
                            start = end + 1;
                        }
                    }
                }
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_lm_head_out, 1, nullptr);
    }
    if (ShouldRunFinalProjectionReferenceProbe()) {
        ProjectionReferenceUserData* final_ref_ud = GetProjectionReferenceUserData();
        final_ref_ud->weight_tensor = model->output;
        final_ref_ud->input_tensor = cur_input_to_lm_head;
        final_ref_ud->layer_idx = static_cast<int>(model->layers.size());
        final_ref_ud->token_seq_ids = batch.seq_id.data();
        final_ref_ud->stage = "lm_head";
        final_ref_ud->var_name = "logits";
        cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, final_ref_ud);
    }
    if (model->arch_flags.is_gemma4 && model->gemma4_final_logit_softcapping > 0.0f &&
        !densecore::models::IsGemma4FinalLogitSoftcapDisabled()) {
        const float inv_softcap = 1.0f / model->gemma4_final_logit_softcapping;
        cur = ggml_scale(ctx_c, cur, inv_softcap);
        cur = ggml_tanh(ctx_c, cur);
        cur = ggml_scale(ctx_c, cur, model->gemma4_final_logit_softcapping);
    }
    ggml_set_name(cur, "output");

    // Add to graph
    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }

    return cur;
}

bool RebindHybridSSMDecodeGraphRuntimeState(GgmlGraphHandle* graph, const BatchSpec& batch) {
    if (!graph) {
        return false;
    }
    if (batch.seq_id.empty() || batch.hybrid_ssm_runtime_states.empty()) {
        return false;
    }

    struct Custom1ParamsView {
        ggml_custom1_op_t fun;
        int n_tasks;
        void* userdata;
    };
    struct Custom3ParamsView {
        ggml_custom3_op_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(Custom1ParamsView) <= GGML_MAX_OP_PARAMS, "Custom1ParamsView too large");
    static_assert(sizeof(Custom3ParamsView) <= GGML_MAX_OP_PARAMS, "Custom3ParamsView too large");

    int conv_rebinds = 0;
    int delta_rebinds = 0;
    const int* seq_ids = batch.seq_id.data();
    const auto* runtime_states = &batch.hybrid_ssm_runtime_states;

    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }

        if (node->op == GGML_OP_MAP_CUSTOM1) {
            Custom1ParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_conv1d) {
                auto* ud = static_cast<SSMConv1DUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                conv_rebinds++;
            }
            continue;
        }

        if (node->op == GGML_OP_MAP_CUSTOM3) {
            Custom3ParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_qwen35_delta) {
                auto* ud = static_cast<SSMQwen35DeltaUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                delta_rebinds++;
            }
        }
    }

    return conv_rebinds > 0 && delta_rebinds > 0 && conv_rebinds == delta_rebinds;
}

#include "inference_sampling.inl"

#ifdef DENSECORE_TEST_BUILD
namespace densecore {
namespace testing {
namespace {
class ScopedAttentionCaptureGuard {
public:
    ScopedAttentionCaptureGuard(std::vector<float>* out, int layer)
        : previous_out_(g_test_capture_attention_out),
          previous_layer_(g_test_capture_attention_layer.load(std::memory_order_relaxed)) {
        g_test_capture_attention_out = out;
        g_test_capture_attention_layer.store(layer, std::memory_order_relaxed);
    }

    ~ScopedAttentionCaptureGuard() {
        g_test_capture_attention_layer.store(previous_layer_, std::memory_order_relaxed);
        g_test_capture_attention_out = previous_out_;
    }

private:
    std::vector<float>* previous_out_;
    int previous_layer_;
};
}  // namespace

bool ShouldUsePagedDecodeAttentionForBatchTest(const TransformerModel* model, const PagedKVCache* cache,
                                               const BatchSpec& batch) {
    if (!model) {
        return false;
    }
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(model->hparams.n_head);
    const int n_head_kv = static_cast<int>(model->hparams.n_head_kv);
    if (n_head <= 0 || n_head_kv <= 0) {
        return false;
    }
    const int head_dim_q = static_cast<int>(model->hparams.n_embd) / n_head;
    const int head_dim_kv =
        model->hparams.n_embd_head_k > 0 ? static_cast<int>(model->hparams.n_embd_head_k) : head_dim_q;
    const BasePagedDecodeExecutionDecision decision =
        ::ResolveBasePagedDecodeExecutionDecision(::LoadDecodePagedAttentionPolicy(), model, cache, batch,
                                                  n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv);
    return decision.use_paged_decode_attention;
}
std::vector<float> ComputeStandardAttentionOutputForTest(const std::vector<float>& q, const std::vector<float>& k,
                                                         const std::vector<float>& v, int n_head, int n_head_kv,
                                                         int head_dim_q, int head_dim_k, int head_dim_v, int n_queries,
                                                         int n_total_tokens, int n_past, int sliding_window,
                                                         float scale, float logit_softcap,
                                                         bool retained_history_layout) {
    if (n_head <= 0 || n_head_kv <= 0 || head_dim_q <= 0 || head_dim_k <= 0 || head_dim_v <= 0 || n_queries <= 0 ||
        n_total_tokens <= 0 || (n_head % n_head_kv) != 0) {
        return {};
    }
    if (static_cast<int>(q.size()) != n_queries * n_head * head_dim_q ||
        static_cast<int>(k.size()) != n_total_tokens * n_head_kv * head_dim_k ||
        static_cast<int>(v.size()) != n_total_tokens * n_head_kv * head_dim_v) {
        return {};
    }

    const bool use_explicit_mask = ::ShouldBuildExplicitStandardAttentionMask(n_queries, sliding_window);
    const int n_rep = n_head / n_head_kv;
    std::vector<float> out(static_cast<size_t>(n_queries) * static_cast<size_t>(n_head) * head_dim_v, 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_total_tokens), -std::numeric_limits<float>::infinity());

    KVRetentionPolicy retention_policy;
    retention_policy.enabled = (sliding_window >= 0);
    retention_policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    retention_policy.sink_tokens = GetKVRetentionPolicy().sink_tokens;
    const KVRetentionSpan retained = ::ComputeKVRetentionSpan(n_past, retention_policy);

    for (int token_idx = 0; token_idx < n_queries; ++token_idx) {
        const int query_pos = n_past + token_idx;
        for (int h = 0; h < n_head; ++h) {
            const int kv_head = h / n_rep;
            const float* q_head = q.data() + (static_cast<size_t>(token_idx) * n_head + h) * head_dim_q;
            float* out_head = out.data() + (static_cast<size_t>(token_idx) * n_head + h) * head_dim_v;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k_idx = 0; k_idx < n_total_tokens; ++k_idx) {
                bool masked = false;
                if (use_explicit_mask) {
                    const int key_pos =
                        retained_history_layout
                            ? ((k_idx < retained.history_kept) ? ::MapRetainedHistoryIndex(retained, k_idx)
                                                               : (n_past + (k_idx - retained.history_kept)))
                            : k_idx;
                    if (sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                        masked = true;
                    }
                    if (n_queries > 1 && key_pos > query_pos) {
                        masked = true;
                    }
                }
                if (masked) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const float* k_head = k.data() + (static_cast<size_t>(k_idx) * n_head_kv + kv_head) * head_dim_k;
                float dot = 0.0f;
                for (int d = 0; d < head_dim_q; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                float score = dot * scale;
                if (logit_softcap > 0.0f && std::isfinite(score)) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
                scores[static_cast<size_t>(k_idx)] = score;
                max_score = std::max(max_score, score);
            }

            if (!std::isfinite(max_score)) {
                continue;
            }

            float denom = 0.0f;
            for (int k_idx = 0; k_idx < n_total_tokens; ++k_idx) {
                const float score = scores[static_cast<size_t>(k_idx)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const float weight = std::exp(score - max_score);
                denom += weight;
                const float* v_head = v.data() + (static_cast<size_t>(k_idx) * n_head_kv + kv_head) * head_dim_v;
                for (int d = 0; d < head_dim_v; ++d) {
                    out_head[d] += weight * v_head[d];
                }
            }

            if (denom > 0.0f) {
                const float inv = 1.0f / denom;
                for (int d = 0; d < head_dim_v; ++d) {
                    out_head[d] *= inv;
                }
            }
        }
    }

    return out;
}
static std::vector<float> ExecuteTransformerGraphForTestImpl(TransformerModel* model, PagedKVCache* cache,
                                                             const BatchSpec& batch, int num_threads,
                                                             bool embedding_mode) {
    if (!model) {
        return {};
    }

    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> work_ctx(CreateInferenceWorkContext(),
                                                                                    DestroyInferenceWorkContext);
    if (!work_ctx) {
        return {};
    }

    SetCurrentWorkContext(work_ctx.get());
    ResetInferenceWorkContext(work_ctx.get());
    SetCurrentBatch(&batch);

    struct ggml_init_params params = {
        /*.mem_size   =*/32ull * 1024ull * 1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        SetCurrentWorkContext(nullptr);
        return {};
    }

    std::vector<float> logits;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
    struct ggml_tensor* output = BuildTransformerGraph(model, cache, ctx, batch, embedding_mode, gf, nullptr, nullptr);
    if (output) {
        if (ggml_graph_n_nodes(gf) == 0) {
            ggml_build_forward_expand(gf, output);
        }
        ggml_graph_compute_with_ctx(ctx, gf, std::max(1, num_threads));
        if (output->data && output->type == GGML_TYPE_F32) {
            const int rows = static_cast<int>(output->ne[0]);
            const int cols = std::max(1, static_cast<int>(output->ne[1]));
            const ptrdiff_t row_stride = static_cast<ptrdiff_t>(output->nb[1] / sizeof(float));
            const float* src = reinterpret_cast<const float*>(output->data);
            logits.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            for (int col = 0; col < cols; ++col) {
                std::memcpy(logits.data() + static_cast<size_t>(col) * static_cast<size_t>(rows),
                            src + static_cast<ptrdiff_t>(col) * row_stride, static_cast<size_t>(rows) * sizeof(float));
            }
        }
    }

    ggml_free(ctx);
    ResetInferenceWorkContext(work_ctx.get());
    SetCurrentWorkContext(nullptr);
    return logits;
}
std::vector<float> ExecuteTransformerGraphForTest(TransformerModel* model, PagedKVCache* cache, const BatchSpec& batch,
                                                  int num_threads) {
    return ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/false);
}
std::vector<float> ExecuteTransformerGraphEmbeddingsForTest(TransformerModel* model, PagedKVCache* cache,
                                                            const BatchSpec& batch, int num_threads) {
    return ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/true);
}
std::vector<float> ExecuteTransformerAttentionForTest(TransformerModel* model, PagedKVCache* cache,
                                                      const BatchSpec& batch, int target_layer, int num_threads) {
    std::vector<float> captured;
    ScopedAttentionCaptureGuard capture_guard(&captured, target_layer);
    (void)ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/false);
    return captured;
}
void SetFlashAttentionDisabledForTest(bool disabled) {
    g_test_force_flash_attention_disabled.store(disabled ? 1 : 0, std::memory_order_relaxed);
}
void ComputeKVRetentionSpanForTest(int n_past, int sliding_window, int sink_tokens, int* history_kept, int* sink_kept,
                                   int* tail_start) {
    KVRetentionPolicy policy;
    policy.enabled = (sliding_window >= 0);
    policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    policy.sink_tokens = std::max(0, sink_tokens);

    const KVRetentionSpan span = ::ComputeKVRetentionSpan(n_past, policy);
    if (history_kept) {
        *history_kept = span.history_kept;
    }
    if (sink_kept) {
        *sink_kept = span.sink_kept;
    }
    if (tail_start) {
        *tail_start = span.tail_start;
    }
}
int MapRetainedHistoryIndexForTest(int n_past, int sliding_window, int sink_tokens, int retained_index) {
    KVRetentionPolicy policy;
    policy.enabled = (sliding_window >= 0);
    policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    policy.sink_tokens = std::max(0, sink_tokens);
    const KVRetentionSpan span = ::ComputeKVRetentionSpan(n_past, policy);
    return ::MapRetainedHistoryIndex(span, retained_index);
}
int GetArmQ4KNativeVecDotModeTest() {
    return static_cast<int>(::GetArmQ4KNativeVecDotMode());
}
void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}
uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}
void ResetHybridSSMQkvForceGgmlCache() {
    std::lock_guard<std::mutex> lock(::g_hybrid_ssm_qkv_force_ggml_mutex);
    ::g_hybrid_ssm_qkv_force_ggml_cached.store(false, std::memory_order_release);
}
bool ShouldForcePlainGgmlForHybridSSMQkvTest() {
    return ::ShouldForcePlainGgmlForHybridSSMQkv();
}
bool ShouldUseArmNativeQ4KVecDotValidatedTest(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                              const void* sample_row_ptr, const void* sample_quant_input,
                                              const float* sample_input_f32, int N) {
    return ::ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, nullptr, sample_row_ptr,
                                                  sample_quant_input, sample_input_f32, N);
}
std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeightsForTest(const TransformerLayer* layer,
                                                                            const TransformerModel* model) {
    return ::BuildExpertWeights(layer, model);
}
bool RouteMoESoftmaxTopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoESoftmaxTopK(gate_logits, &ud, routing);
}
bool RouteMoEGroupedSigmoidForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                   const TransformerLayer* layer, int top_k,
                                   densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoEGroupedSigmoid(gate_logits, &ud, routing);
}
bool RouteMoEGemma4TopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                               const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoEGemma4TopK(gate_logits, &ud, routing);
}
std::vector<float> ApplySharedScalarGateForTest(const std::vector<float>& shared_ffn_pre_gate,
                                                const std::vector<float>& shared_gate_logits_scalar, int tokens,
                                                int hidden_dim) {
    if (tokens <= 0 || hidden_dim <= 0 ||
        static_cast<int>(shared_ffn_pre_gate.size()) != tokens * hidden_dim ||
        static_cast<int>(shared_gate_logits_scalar.size()) != tokens) {
        return {};
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return {};
    }

    struct ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, tokens);
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, tokens);
    struct ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, tokens);
    if (!src || !dst || !gate || !src->data || !dst->data || !gate->data) {
        ggml_free(ctx);
        return {};
    }

    std::memcpy(src->data, shared_ffn_pre_gate.data(), shared_ffn_pre_gate.size() * sizeof(float));
    std::memcpy(gate->data, shared_gate_logits_scalar.data(), shared_gate_logits_scalar.size() * sizeof(float));
    SharedScalarGateApplyUserData ud{};
    ud.gate_logits_scalar = gate;
    ::cb_apply_shared_scalar_gate(dst, src, 0, 1, &ud);

    std::vector<float> out(static_cast<size_t>(tokens * hidden_dim), 0.0f);
    std::memcpy(out.data(), dst->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}
std::vector<float> ComputeSharedExpertMergedOutputForTest(const std::vector<float>& moe_input,
                                                          const std::vector<float>& routed_output,
                                                          const std::vector<float>& gate_weight,
                                                          const std::vector<float>& up_weight,
                                                          const std::vector<float>& down_weight,
                                                          const std::vector<float>& shared_gate_logits_scalar,
                                                          int tokens, int hidden_dim, int intermediate_dim) {
    if (tokens <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 ||
        static_cast<int>(moe_input.size()) != tokens * hidden_dim ||
        static_cast<int>(routed_output.size()) != tokens * hidden_dim ||
        static_cast<int>(gate_weight.size()) != intermediate_dim * hidden_dim ||
        static_cast<int>(up_weight.size()) != intermediate_dim * hidden_dim ||
        static_cast<int>(down_weight.size()) != hidden_dim * intermediate_dim ||
        static_cast<int>(shared_gate_logits_scalar.size()) != tokens) {
        return {};
    }

    auto matmul_trans_b = [](const std::vector<float>& input, const std::vector<float>& weight, int M, int K,
                             int N) {
        std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += input[static_cast<size_t>(m * K + k)] * weight[static_cast<size_t>(n * K + k)];
                }
                out[static_cast<size_t>(m * N + n)] = sum;
            }
        }
        return out;
    };

    const std::vector<float> shared_gate = matmul_trans_b(moe_input, gate_weight, tokens, hidden_dim, intermediate_dim);
    const std::vector<float> shared_up = matmul_trans_b(moe_input, up_weight, tokens, hidden_dim, intermediate_dim);
    std::vector<float> shared_ffn_pre_gate(static_cast<size_t>(tokens * intermediate_dim), 0.0f);
    for (size_t i = 0; i < shared_ffn_pre_gate.size(); ++i) {
        const float g = shared_gate[i];
        shared_ffn_pre_gate[i] = (g / (1.0f + std::exp(-g))) * shared_up[i];
    }
    const std::vector<float> gated = ApplySharedScalarGateForTest(shared_ffn_pre_gate, shared_gate_logits_scalar,
                                                                  tokens, intermediate_dim);
    const std::vector<float> shared_down = matmul_trans_b(gated, down_weight, tokens, intermediate_dim, hidden_dim);
    std::vector<float> merged = routed_output;
    for (size_t i = 0; i < merged.size(); ++i) {
        merged[i] += shared_down[i];
    }
    return merged;
}
void CbSsmQwen35DeltaTest(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                          const struct ggml_tensor* c, int ith, int nth, void* userdata) {
    ::cb_ssm_qwen35_delta(dst, a, b, c, ith, nth, userdata);
}
struct ggml_tensor* SmartMulMatTest(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                    TransformerModel* model) {
    return ::smart_mul_mat(ctx, weight, input, model);
}
}  // namespace testing
}  // namespace densecore
#endif

void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}

uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}

void ResetMoECallbackEntryCounter() {
    ::ResetMoECallbackEntryCount();
}

uint64_t GetMoECallbackEntryCounter() {
    return ::GetMoECallbackEntryCount();
}

uint64_t GetMoECallbackMissingUserdataCounter() {
    return ::GetMoECallbackMissingUserdataCount();
}

uint64_t GetMoECallbackMissingBackendCounter() {
    return ::GetMoECallbackMissingBackendCount();
}

uint64_t GetMoECallbackMissingExpertsCounter() {
    return ::GetMoECallbackMissingExpertsCount();
}

uint64_t GetMoECallbackRoutingFailureCounter() {
    return ::GetMoECallbackRoutingFailureCount();
}

uint64_t GetMoECallbackEmptyRoutingCounter() {
    return ::GetMoECallbackEmptyRoutingCount();
}

uint64_t GetMoECallbackFailClosedCounter() {
    return ::GetMoECallbackFailClosedCount();
}

bool ConsumeMoEStrictFailure(std::string* message) {
    return ::ConsumeMoEStrictFailureState(message);
}

void ResetMoEStrictFailure() {
    ::ResetMoEStrictFailureState();
}

namespace densecore::testing {
namespace {
std::vector<float> ApplyWeightedRmsNormVectorForTest(const std::vector<float>& src, const std::vector<float>& weight,
                                                     float eps) {
    if (src.empty()) {
        return {};
    }
    std::vector<float> out(src.size(), 0.0f);
    float sum_sq = 0.0f;
    for (float v : src) {
        sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(src.size()) + eps);
    for (size_t i = 0; i < src.size(); ++i) {
        const float w = i < weight.size() ? weight[i] : 1.0f;
        out[i] = src[i] * inv_rms * w;
    }
    return out;
}
}  // namespace

Gemma4MoEBranchInputsSnapshot ComputeGemma4MoEBranchInputsForTest(const std::vector<float>& attn_post_residual,
                                                                  const std::vector<float>& inp_ff,
                                                                  const std::vector<float>& ffn_norm_weight,
                                                                  const std::vector<float>& pre_moe_norm_weight,
                                                                  float eps) {
    Gemma4MoEBranchInputsSnapshot snapshot;
    snapshot.shared_input = ApplyWeightedRmsNormVectorForTest(attn_post_residual, ffn_norm_weight, eps);
    snapshot.routed_input =
        pre_moe_norm_weight.empty() ? snapshot.shared_input
                                    : ApplyWeightedRmsNormVectorForTest(inp_ff, pre_moe_norm_weight, eps);
    return snapshot;
}
}  // namespace densecore::testing
