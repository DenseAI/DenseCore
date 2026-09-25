#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/models/lfm2_shortconv_math.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/scheduler.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/graph/construction_ops.h"
#include "llm/moe/exec.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/work_context.h"
#include "llm/ssm/internal.h"
#include "runtime/inference_types_internal.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace densecore::llm::graph::detail;
using densecore::env::ParseBoolEnv;
using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using namespace densecore::llm::runtime;

bool IsDebugSSMCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SSM_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
// ============================================================================
// SSM (Mamba2) Callback Functions
// ============================================================================

static inline float SoftplusStable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float SigmoidStable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static inline void DotPairProducts(const float* lhs0, const float* lhs1, const float* rhs, int n, float* out0,
                                   float* out1) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 x = _mm512_loadu_ps(rhs + i);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_loadu_ps(lhs0 + i), x));
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_loadu_ps(lhs1 + i), x));
    }
    sum0 = _mm512_reduce_add_ps(acc0);
    sum1 = _mm512_reduce_add_ps(acc1);
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(rhs + i);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(lhs0 + i), x));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(lhs1 + i), x));
    }
    alignas(32) float lanes0[8];
    alignas(32) float lanes1[8];
    _mm256_store_ps(lanes0, acc0);
    _mm256_store_ps(lanes1, acc1);
    for (int lane = 0; lane < 8; ++lane) {
        sum0 += lanes0[lane];
        sum1 += lanes1[lane];
    }
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t x = vld1q_f32(rhs + i);
        acc0 = vfmaq_f32(acc0, vld1q_f32(lhs0 + i), x);
        acc1 = vfmaq_f32(acc1, vld1q_f32(lhs1 + i), x);
    }
    sum0 = vaddvq_f32(acc0);
    sum1 = vaddvq_f32(acc1);
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#else
    for (int i = 0; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#endif
    *out0 = sum0;
    *out1 = sum1;
}

static inline void QKNormProducts(const float* q_head, const float* k_head, int n, float* q_sum_sq, float* k_sum_sq,
                                  float* qk_raw_dot) {
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float qk_sum = 0.0f;
#if defined(__AVX512F__)
    __m512 q_acc = _mm512_setzero_ps();
    __m512 k_acc = _mm512_setzero_ps();
    __m512 qk_acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 qv = _mm512_loadu_ps(q_head + i);
        const __m512 kv = _mm512_loadu_ps(k_head + i);
        q_acc = _mm512_add_ps(q_acc, _mm512_mul_ps(qv, qv));
        k_acc = _mm512_add_ps(k_acc, _mm512_mul_ps(kv, kv));
        qk_acc = _mm512_add_ps(qk_acc, _mm512_mul_ps(qv, kv));
    }
    q_sum = _mm512_reduce_add_ps(q_acc);
    k_sum = _mm512_reduce_add_ps(k_acc);
    qk_sum = _mm512_reduce_add_ps(qk_acc);
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#elif defined(__AVX2__)
    __m256 q_acc = _mm256_setzero_ps();
    __m256 k_acc = _mm256_setzero_ps();
    __m256 qk_acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q_head + i);
        const __m256 kv = _mm256_loadu_ps(k_head + i);
        q_acc = _mm256_add_ps(q_acc, _mm256_mul_ps(qv, qv));
        k_acc = _mm256_add_ps(k_acc, _mm256_mul_ps(kv, kv));
        qk_acc = _mm256_add_ps(qk_acc, _mm256_mul_ps(qv, kv));
    }
    alignas(32) float q_lanes[8];
    alignas(32) float k_lanes[8];
    alignas(32) float qk_lanes[8];
    _mm256_store_ps(q_lanes, q_acc);
    _mm256_store_ps(k_lanes, k_acc);
    _mm256_store_ps(qk_lanes, qk_acc);
    for (int lane = 0; lane < 8; ++lane) {
        q_sum += q_lanes[lane];
        k_sum += k_lanes[lane];
        qk_sum += qk_lanes[lane];
    }
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t q_acc = vdupq_n_f32(0.0f);
    float32x4_t k_acc = vdupq_n_f32(0.0f);
    float32x4_t qk_acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t qv = vld1q_f32(q_head + i);
        const float32x4_t kv = vld1q_f32(k_head + i);
        q_acc = vfmaq_f32(q_acc, qv, qv);
        k_acc = vfmaq_f32(k_acc, kv, kv);
        qk_acc = vfmaq_f32(qk_acc, qv, kv);
    }
    q_sum = vaddvq_f32(q_acc);
    k_sum = vaddvq_f32(k_acc);
    qk_sum = vaddvq_f32(qk_acc);
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#endif
    *q_sum_sq = q_sum;
    *k_sum_sq = k_sum;
    *qk_raw_dot = qk_sum;
}

bool IsSSMNonFiniteDebugEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_SSM_NONFINITE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}


static void AbortOnFirstSSMNonFinite(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                     const char* var_name, int elem_idx, float value,
                                     const Qwen35SSMDebugScalars& scalars) {
    static std::atomic<bool> fired{false};
    if (!IsSSMNonFiniteDebugEnabled()) return;
    if (fired.exchange(true, std::memory_order_relaxed)) return;

    std::fprintf(
        stderr,
        "[DenseCore][SSM_NONFINITE] layer=%d token=%d seq=%d head=%d stage=%s var=%s elem=%d value=%g "
        "alpha=%g beta=%g softplus=%g exp_a_log=%g g=%g decay=%g beta_gate=%g q_sum_sq=%g k_sum_sq=%g rms=%g\n",
        layer_idx, token_idx, seq_idx, head_idx, stage ? stage : "<unknown>", var_name ? var_name : "<unnamed>",
        elem_idx, static_cast<double>(value), static_cast<double>(scalars.alpha), static_cast<double>(scalars.beta),
        static_cast<double>(scalars.softplus_alpha), static_cast<double>(scalars.exp_a_log),
        static_cast<double>(scalars.g), static_cast<double>(scalars.decay), static_cast<double>(scalars.beta_gate),
        static_cast<double>(scalars.q_sum_sq), static_cast<double>(scalars.k_sum_sq), static_cast<double>(scalars.rms));
    std::fflush(stderr);
    std::abort();
}

static void CheckSSMFiniteScalar(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                 const char* var_name, float value, const Qwen35SSMDebugScalars& scalars) {
    if (!IsSSMNonFiniteDebugEnabled() || std::isfinite(value)) return;
    AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, head_idx, stage, var_name, -1, value, scalars);
}

void CheckSSMFiniteVector(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                          const char* var_name, const float* values, int n, const Qwen35SSMDebugScalars& scalars) {
    if (!IsSSMNonFiniteDebugEnabled() || !values || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, head_idx, stage, var_name, i, values[i], scalars);
        }
    }
}

void CheckSSMFiniteTensor(int layer_idx, const int* token_seq_ids, const struct ggml_tensor* tensor, const char* stage,
                          const char* var_name) {
    if (!IsSSMNonFiniteDebugEnabled() || !tensor || !tensor->data) return;
    const auto* values = reinterpret_cast<const float*>(tensor->data);
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = std::max<int64_t>(1, tensor->ne[1]);
    const int64_t total = ggml_nelements(tensor);
    if (!values || ne0 <= 0 || ne1 <= 0 || total <= 0) return;
    const Qwen35SSMDebugScalars scalars{};
    for (int64_t idx = 0; idx < total; ++idx) {
        if (std::isfinite(values[idx])) continue;
        const int token_idx = static_cast<int>(idx / ne0);
        const int seq_idx = (token_seq_ids && token_idx >= 0 && token_idx < ne1) ? token_seq_ids[token_idx] : -1;
        const int elem_idx = static_cast<int>(idx % ne0);
        AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, -1, stage, var_name, elem_idx, values[idx], scalars);
    }
}

static bool IsDebugSSMDeltaStatsEnabled() {
    static const bool enabled = densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_SSM_DELTA_STATS", false);
    return enabled;
}

static int DebugSSMDeltaStatsLayer() {
    static const int layer = densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_SSM_DELTA_STATS_LAYER", -1);
    return layer;
}

static int DebugSSMDeltaStatsToken() {
    static const int token = densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_SSM_DELTA_STATS_TOKEN", -1);
    return token;
}

static bool ShouldLogSSMDeltaStats(int layer_idx, int token_idx) {
    if (!IsDebugSSMDeltaStatsEnabled()) {
        return false;
    }
    const int target_layer = DebugSSMDeltaStatsLayer();
    if (target_layer >= 0 && target_layer != layer_idx) {
        return false;
    }
    const int target_token = DebugSSMDeltaStatsToken();
    return target_token < 0 || target_token == token_idx;
}

static bool ConsumeSSMDeltaStatsBudget() {
    static std::atomic<int> remaining{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_SSM_DELTA_STATS_MAX_CALLS", 16)};
    int current = remaining.load(std::memory_order_relaxed);
    while (current > 0) {
        if (remaining.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

[[noreturn]] void FatalQwen35SSMRuntimeError(int layer_idx, int token_idx, int seq_idx, int head_idx,
                                             const char* message) {
    std::fprintf(stderr, "[DenseCore][Qwen35SSM] FATAL layer=%d token=%d seq=%d head=%d: %s\n", layer_idx, token_idx,
                 seq_idx, head_idx, message ? message : "unknown error");
#if !defined(NDEBUG)
    std::abort();
#else
    std::terminate();
#endif
}


void RunSSMConv1DReference(const float* conv_state, const float* input, const float* weight, float* output,
                           int channels, int kernel_size) {
    if (!conv_state || !input || !weight || !output || channels <= 0 || kernel_size <= 0) return;
    const int hist = kernel_size - 1;
    for (int ch = 0; ch < channels; ++ch) {
        float sum = 0.0f;
        const float* state_row = conv_state + static_cast<size_t>(ch) * hist;
        const float* weight_row = weight + static_cast<size_t>(ch) * kernel_size;
        for (int k = 0; k < hist; ++k) {
            sum += state_row[k] * weight_row[k];
        }
        sum += input[ch] * weight_row[hist];
        output[ch] = sum;
    }
}

void LogSSMCoreReferenceDiff(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                             const char* var_name, const float* actual, const float* reference, int n) {
    if (!IsDebugSSMCoreReferenceEnabled() || !actual || !reference || n <= 0) return;
    float max_abs_diff = 0.0f;
    int max_idx = -1;
    int first_bad_idx = -1;
    bool actual_nonfinite = false;
    bool ref_nonfinite = false;
    for (int i = 0; i < n; ++i) {
        const bool a_fin = std::isfinite(actual[i]);
        const bool r_fin = std::isfinite(reference[i]);
        if (!a_fin || !r_fin) {
            if (first_bad_idx < 0) {
                first_bad_idx = i;
                actual_nonfinite = !a_fin;
                ref_nonfinite = !r_fin;
            }
            continue;
        }
        const float diff = std::fabs(actual[i] - reference[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }
    if (first_bad_idx < 0 && max_idx < 0) return;
    std::fprintf(stderr,
                 "[DenseCore][SSM_CORE_REF] layer=%d token=%d seq=%d head=%d stage=%s var=%s n=%d first_bad=%d "
                 "actual_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_idx=%d",
                 layer_idx, token_idx, seq_idx, head_idx, stage ? stage : "unknown", var_name ? var_name : "unknown", n,
                 first_bad_idx, actual_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx);
    if (first_bad_idx >= 0) {
        std::fprintf(stderr, " actual=%g ref=%g\n", static_cast<double>(actual[first_bad_idx]),
                     static_cast<double>(reference[first_bad_idx]));
    } else {
        std::fprintf(stderr, " actual=%g ref=%g\n", static_cast<double>(actual[max_idx]),
                     static_cast<double>(reference[max_idx]));
    }
}

static bool IsTraceQwen35SSMRuntimeContractEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_SSM_TRACE_RUNTIME_CONTRACT");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static int TraceQwen35SSMMaxHeads() {
    static const int value = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_SSM_TRACE_MAX_HEADS");
        if (!env || env[0] == '\0') {
            return 32;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0') || parsed <= 0) {
            return 32;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

static int TraceQwen35SSMMaxTokens() {
    static const int value = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_SSM_TRACE_MAX_TOKENS");
        if (!env || env[0] == '\0') {
            return 1024;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0') || parsed <= 0) {
            return 1024;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

static uint64_t HashQwen35SSMFloatSpan(const float* data, size_t count) {
    if (!data) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ull;
    hash ^= static_cast<uint64_t>(count);
    hash *= 1099511628211ull;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ull;
    }
    return hash;
}

static void LogQwen35SSMRuntimeContract(const char* mode, int layer_idx, int token_idx, int seq_idx, int head_idx,
                                        int src_k_head, int heads_per_group, uintptr_t token_state_base_ptr,
                                        size_t token_state_elems, size_t state_offset_bytes, size_t state_span_bytes,
                                        ptrdiff_t input_stride, ptrdiff_t output_stride, bool overlap_prev,
                                        bool grouped_src_reused, uint64_t q_hash, uint64_t k_hash, uint64_t v_hash,
                                        uint64_t state_in_hash, uint64_t state_out_hash, uint64_t y_hash) {
    if (!IsTraceQwen35SSMRuntimeContractEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "[SSM_RUNTIME_CONTRACT] mode=%s layer=%d token=%d seq=%d head=%d src_k_head=%d heads_per_group=%d "
                 "base_ptr=0x%llx total_elems=%zu state_off_bytes=%zu state_span_bytes=%zu input_stride=%lld "
                 "output_stride=%lld overlap_prev=%d grouped_src_reused=%d q_hash=0x%llx k_hash=0x%llx v_hash=0x%llx "
                 "state_in=0x%llx state_out=0x%llx y_hash=0x%llx\n",
                 mode ? mode : "unknown", layer_idx, token_idx, seq_idx, head_idx, src_k_head, heads_per_group,
                 static_cast<unsigned long long>(token_state_base_ptr), token_state_elems, state_offset_bytes,
                 state_span_bytes, static_cast<long long>(input_stride), static_cast<long long>(output_stride),
                 overlap_prev ? 1 : 0, grouped_src_reused ? 1 : 0, static_cast<unsigned long long>(q_hash),
                 static_cast<unsigned long long>(k_hash), static_cast<unsigned long long>(v_hash),
                 static_cast<unsigned long long>(state_in_hash), static_cast<unsigned long long>(state_out_hash),
                 static_cast<unsigned long long>(y_hash));
}

static void LogQwen35SSMLayerAggregate(const char* mode, int layer_idx, int token_idx, int seq_idx,
                                       uint64_t aggregate_hash) {
    if (!IsTraceQwen35SSMRuntimeContractEnabled()) {
        return;
    }
    std::fprintf(stderr, "[SSM_LAYER_AGG] mode=%s layer=%d token=%d seq=%d aggregate_state_hash=0x%llx\n",
                 mode ? mode : "unknown", layer_idx, token_idx, seq_idx,
                 static_cast<unsigned long long>(aggregate_hash));
}

static bool RunQwen35ReferenceHeadStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head,
                                       Qwen35SSMHeadStepStats* stats) {
    if (!cfg.input_t || !cfg.q_head || !cfg.k_head || !cfg.v_head || !cfg.z_head || !cfg.alpha_row || !cfg.beta_row ||
        !state_kv || !y_head || cfg.n_embd <= 0 || cfg.head_dim_k <= 0 || cfg.head_dim_v <= 0) {
        return false;
    }

    std::vector<float> q_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> k_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> delta(static_cast<size_t>(cfg.head_dim_v), 0.0f);

    float alpha = cfg.dt_bias;
    float beta = 0.0f;
    for (int i = 0; i < cfg.n_embd; ++i) {
        alpha += cfg.alpha_row[i] * cfg.input_t[i];
        beta += cfg.beta_row[i] * cfg.input_t[i];
    }

    const float softplus_alpha = SoftplusStable(alpha);
    const float ssm_a = (cfg.a_log_prescaled && cfg.a_log <= 0.0f) ? cfg.a_log : -std::exp(cfg.a_log);
    const float exp_a_log = -ssm_a;
    const float g = ssm_a * softplus_alpha;
    const float decay = std::exp(g);
    const float beta_gate = SigmoidStable(beta);

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }
    const float q_inv_norm = 1.0f / std::max(std::sqrt(q_sum_sq), cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::max(std::sqrt(k_sum_sq), cfg.norm_eps);
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
        k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
    }

    const size_t state_elems = static_cast<size_t>(cfg.head_dim_k) * static_cast<size_t>(cfg.head_dim_v);
    for (size_t i = 0; i < state_elems; ++i) {
        state_kv[i] *= decay;
    }
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float kv_mem = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
        }
        delta[static_cast<size_t>(v)] = (cfg.v_head[v] - kv_mem) * beta_gate;
    }
    for (int k = 0; k < cfg.head_dim_k; ++k) {
        float* row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        const float kval = k_norm[static_cast<size_t>(k)];
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            row[v] += kval * delta[static_cast<size_t>(v)];
        }
    }

    float sum_sq = 0.0f;
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float sum = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
        }
        y_head[v] = sum / std::sqrt(static_cast<float>(cfg.head_dim_k));
        sum_sq += y_head[v] * y_head[v];
    }
    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        const float norm_w = cfg.norm_weight ? cfg.norm_weight[v] : 1.0f;
        y_head[v] = (y_head[v] / rms) * norm_w * (cfg.z_head[v] * SigmoidStable(cfg.z_head[v]));
    }

    if (stats) {
        stats->alpha = alpha;
        stats->beta = beta;
        stats->softplus_alpha = softplus_alpha;
        stats->exp_a_log = exp_a_log;
        stats->g = g;
        stats->decay = decay;
        stats->beta_gate = beta_gate;
        stats->q_sum_sq = q_sum_sq;
        stats->k_sum_sq = k_sum_sq;
        stats->rms = rms;
    }
    return true;
}


void cb_ssm_qwen35_delta_qkv_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                  void* userdata) {
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    if (!dst || !src || !ud || !ud->z_tensor || !ud->input_tensor) {
        return;
    }
    cb_ssm_qwen35_delta(dst, ud->z_tensor, src, ud->input_tensor, ith, nth, userdata);
}

void cb_ssm_qwen35_delta_z_qkv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                               int ith, int nth, void* userdata) {
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    if (!dst || !a || !b || !ud || !ud->input_tensor) {
        return;
    }
    cb_ssm_qwen35_delta(dst, a, b, ud->input_tensor, ith, nth, userdata);
}

void cb_ssm_qwen35_delta_z_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata) {
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    if (!dst || !src || !ud || !ud->qkv_tensor || !ud->input_tensor) {
        return;
    }
    cb_ssm_qwen35_delta(dst, src, ud->qkv_tensor, ud->input_tensor, ith, nth, userdata);
}

void cb_ssm_qwen35_delta_z_qkv_alpha_beta(struct ggml_tensor* dst, const struct ggml_tensor* z,
                                          const struct ggml_tensor* qkv, const struct ggml_tensor* alpha_beta, int ith,
                                          int nth, void* userdata) {
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    if (!dst || !z || !qkv || !alpha_beta || !ud || !ud->input_tensor) {
        return;
    }
    cb_ssm_qwen35_delta(dst, z, qkv, ud->input_tensor, ith, nth, userdata);
}

void cb_ssm_alpha_beta_qk_project_map3(struct ggml_tensor* dst, const struct ggml_tensor* placeholder,
                                       const struct ggml_tensor* input_src, const struct ggml_tensor* qkv, int ith,
                                       int nth, void* userdata) {
    (void)placeholder;
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    if (!ud || !dst || !input_src || !qkv || !dst->data || !input_src->data || !qkv->data || !ud->alpha_weight ||
        !ud->beta_weight || !ud->dt_bias || ud->n_embd <= 0 || ud->n_heads <= 0 || ud->n_groups <= 0 ||
        ud->head_dim_k <= 0 || nth <= 0 || ith < 0 || ith >= nth) {
        return;
    }
    const int out_rows = 2 * ud->n_heads + 3 * ud->n_groups;
    if (input_src->type != GGML_TYPE_F32 || qkv->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        static_cast<int>(input_src->ne[0]) != ud->n_embd || static_cast<int>(dst->ne[0]) != out_rows ||
        input_src->ne[1] != dst->ne[1] || qkv->ne[1] != dst->ne[1]) {
        return;
    }

    const int N = static_cast<int>(dst->ne[1]);
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(input_src->nb[1] / sizeof(float));
    const ptrdiff_t qkv_stride = static_cast<ptrdiff_t>(qkv->nb[1] / sizeof(float));
    const ptrdiff_t out_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const float* input = reinterpret_cast<const float*>(input_src->data);
    const float* qkv_data = reinterpret_cast<const float*>(qkv->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const int qk_total = ud->n_groups * ud->head_dim_k;
    const int alpha_beta_work = N * ud->n_heads;
    const int qk_work = N * ud->n_groups;
    const int total = alpha_beta_work + qk_work;
    const int per_task = (total + nth - 1) / nth;
    const int begin = ith * per_task;
    const int end = std::min(total, begin + per_task);

    for (int idx = begin; idx < end; ++idx) {
        if (idx < alpha_beta_work) {
            const int t = idx / ud->n_heads;
            const int h = idx - t * ud->n_heads;
            const float* input_t = input + static_cast<ptrdiff_t>(t) * input_stride;
            const float* alpha_row = ud->alpha_weight + static_cast<size_t>(h) * static_cast<size_t>(ud->n_embd);
            const float* beta_row = ud->beta_weight + static_cast<size_t>(h) * static_cast<size_t>(ud->n_embd);
            float alpha_dot = 0.0f;
            float beta_dot = 0.0f;
            DotPairProducts(alpha_row, beta_row, input_t, ud->n_embd, &alpha_dot, &beta_dot);
            const float alpha = ud->dt_bias[h] + alpha_dot;
            const float beta = beta_dot;
            float* out_t = out + static_cast<ptrdiff_t>(t) * out_stride;
            out_t[h] = alpha;
            out_t[ud->n_heads + h] = beta;
            continue;
        }

        const int local = idx - alpha_beta_work;
        const int t = local / ud->n_groups;
        const int h = local - t * ud->n_groups;
        const float* row = qkv_data + static_cast<ptrdiff_t>(t) * qkv_stride;
        const float* q_head = row + static_cast<size_t>(h) * ud->head_dim_k;
        const float* k_head = row + static_cast<size_t>(qk_total) + static_cast<size_t>(h) * ud->head_dim_k;
        float q_sum_sq = 0.0f;
        float k_sum_sq = 0.0f;
        float qk_raw_dot = 0.0f;
        QKNormProducts(q_head, k_head, ud->head_dim_k, &q_sum_sq, &k_sum_sq, &qk_raw_dot);
        const float q_inv_norm = 1.0f / std::max(std::sqrt(q_sum_sq), ud->norm_eps);
        const float k_inv_norm = 1.0f / std::max(std::sqrt(k_sum_sq), ud->norm_eps);
        float* out_t = out + static_cast<ptrdiff_t>(t) * out_stride + 2 * ud->n_heads;
        out_t[h] = q_inv_norm;
        out_t[ud->n_groups + h] = k_inv_norm;
        out_t[2 * ud->n_groups + h] = qk_raw_dot * q_inv_norm * k_inv_norm;
    }
}

// Qwen3.5 recurrent delta-net callback.
//
// Inputs:
//   a = z projection                   [d_inner, N]
//   b = convolved qkv_mixed            [conv_channels, N]
//   c = normalized layer input         [n_embd, N]
//
// Output:
//   dst = scratch buffer whose first d_inner rows contain the recurrent
//         linear-attention output after gated RMSNorm
void cb_ssm_qwen35_delta(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                         const struct ggml_tensor* c, int ith, int nth, void* userdata) {
    const int task_count = std::max(1, nth);
    if (ith < 0 || ith >= task_count) {
        return;
    }
    const auto profile_begin =
        IsQwen36ProfilingEnabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ith == 0 && IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> ssm_call_count{0};
        int ssm_id = ssm_call_count.fetch_add(1);
        fprintf(stderr, "[DBG] cb_ssm_delta #%d a=[%lld,%lld]\n", ssm_id, (long long)a->ne[0], (long long)a->ne[1]);
    }
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    const float* z_proj = reinterpret_cast<const float*>(a->data);
    const float* qkv_conv = reinterpret_cast<const float*>(b->data);
    const float* input = reinterpret_cast<const float*>(c->data);
    float* y_out = reinterpret_cast<float*>(dst->data);
    if (!ud || !qkv_conv || !z_proj || !input || !y_out) return;

    const int N = static_cast<int>(a->ne[1]);
    const ptrdiff_t z_stride = static_cast<ptrdiff_t>(a->nb[1] / sizeof(float));
    const ptrdiff_t qkv_stride = static_cast<ptrdiff_t>(b->nb[1] / sizeof(float));
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(c->nb[1] / sizeof(float));
    const ptrdiff_t out_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const float* alpha_beta = (ud->alpha_beta_tensor && ud->alpha_beta_tensor->data)
                                  ? reinterpret_cast<const float*>(ud->alpha_beta_tensor->data)
                                  : nullptr;
    const ptrdiff_t alpha_beta_stride =
        alpha_beta ? static_cast<ptrdiff_t>(ud->alpha_beta_tensor->nb[1] / sizeof(float)) : 0;
    static const bool ssm_debug = (std::getenv("SSM_DEBUG") != nullptr);
    const int num_k_heads = ud->n_groups;
    const int num_v_heads = ud->n_heads;
    const int head_k_dim = ud->head_dim_k;
    const int head_v_dim = ud->head_dim_v;
    const int qk_total = head_k_dim * num_k_heads;
    const int state_stride = head_k_dim * head_v_dim;
    const int heads_per_group = (num_k_heads > 0) ? std::max(1, num_v_heads / num_k_heads) : 1;
    // The per-layer aggregate contract trace hashes every head of a token, so it
    // only means anything when one task owns all heads. Collapse to a single task
    // while it is enabled instead of letting the trace silently go quiet, the
    // same way the conv1d callback serializes for its debug modes.
    const bool serialize_for_contract_trace = IsTraceQwen35SSMRuntimeContractEnabled();
    if (serialize_for_contract_trace && ith != 0) {
        return;
    }
    const int effective_task_count = serialize_for_contract_trace ? 1 : task_count;
    const int effective_task_idx = serialize_for_contract_trace ? 0 : ith;
    const int head_begin = (num_v_heads * effective_task_idx) / effective_task_count;
    const int head_end = (num_v_heads * (effective_task_idx + 1)) / effective_task_count;
    const auto resolve_src_k_head = [&](int v_head_idx) -> int {
        if (num_k_heads == num_v_heads) {
            return v_head_idx;
        }
        if (ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL ||
            ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL ||
            ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN38_OFFICIAL) {
            return v_head_idx % num_k_heads;
        }
        return std::min(num_k_heads - 1, v_head_idx / heads_per_group);
    };

    // =========================================================================
    // LAYOUT ASSERTIONS & RUNTIME CONTRACT CHECK
    // =========================================================================
    if (!ud || !a || !b || !c || !dst || !a->data || !b->data || !c->data || !dst->data) {
        FatalQwen35SSMRuntimeError(ud ? ud->layer_idx : -1, 0, -1, -1, "SSM delta: null tensor or data");
    }

    const int expected_conv_channels = ud->d_inner + 2 * qk_total;

    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || c->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[DenseCore][Qwen35SSM] FATAL DTYPE layer=%d a=%d b=%d c=%d dst=%d\n", ud->layer_idx,
                     static_cast<int>(a->type), static_cast<int>(b->type), static_cast<int>(c->type),
                     static_cast<int>(dst->type));
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM delta inputs/outputs must be F32");
    }

    if (b->nb[0] != sizeof(float) || c->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1,
                                   "SSM delta: non-contiguous element stride detected in b, c, or dst");
    }

    if (b->ne[0] != expected_conv_channels) {
        fprintf(stderr, "[DenseCore][Qwen35SSM] FATAL LAYOUT MISMATCH layer=%d: qkv_conv ne[0]=%lld expected=%d\n",
                ud->layer_idx, (long long)b->ne[0], expected_conv_channels);
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1,
                                   "SSM QKV layout mismatch: ne[0] != d_inner + 2*n_groups*head_dim_k");
    }
    if (a->ne[0] != ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM z tensor shape mismatch");
    }
    if (c->ne[0] != ud->n_embd) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM input tensor shape mismatch");
    }
    // ggml_map_custom3() allocates the output tensor with the same shape as its
    // first source tensor `a`, which is the z projection [d_inner, N].
    if (dst->ne[0] != ud->d_inner || dst->ne[1] != a->ne[1]) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM output tensor shape mismatch");
    }
    if (ud->d_inner < num_v_heads * head_v_dim) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM output tensor is smaller than head layout");
    }

    if (qkv_stride < expected_conv_channels) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM QKV row stride is smaller than element count");
    }
    if (z_stride < ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM z row stride is smaller than element count");
    }
    if (input_stride < ud->n_embd) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM input row stride is smaller than element count");
    }
    if (out_stride < ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM output row stride is smaller than element count");
    }


    const bool ssm_nonfinite_debug = IsSSMNonFiniteDebugEnabled();
    const bool debug_core_ref = IsDebugSSMCoreReferenceEnabled();
    const bool trace_contract_enabled = IsTraceQwen35SSMRuntimeContractEnabled();
    const bool collect_step_debug = ssm_nonfinite_debug || ssm_debug;
    const bool can_use_fast_default_head_step =
        (ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL ||
         ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL ||
         ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN38_OFFICIAL) &&
        !debug_core_ref && !trace_contract_enabled && !collect_step_debug;

    // Reuse scratch storage per thread for optional diagnostics. The default
    // serving path keeps these null so the head step does not copy debug spans
    // or collect per-stage timings for every prefill token/head.
    static thread_local std::vector<float> q_norm_buf;
    static thread_local std::vector<float> k_norm_buf;
    static thread_local std::vector<float> kv_mem_buf;
    static thread_local std::vector<float> delta_buf;
    static thread_local std::vector<float> y_pre_norm_buf;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    float* kv_mem = nullptr;
    float* delta = nullptr;
    float* y_pre_norm = nullptr;
    if (collect_step_debug) {
        q_norm_buf.resize(static_cast<size_t>(head_k_dim));
        k_norm_buf.resize(static_cast<size_t>(head_k_dim));
        kv_mem_buf.resize(static_cast<size_t>(head_v_dim));
        delta_buf.resize(static_cast<size_t>(head_v_dim));
        y_pre_norm_buf.resize(static_cast<size_t>(head_v_dim));
        q_norm = q_norm_buf.data();
        k_norm = k_norm_buf.data();
        kv_mem = kv_mem_buf.data();
        delta = delta_buf.data();
        y_pre_norm = y_pre_norm_buf.data();
    }
    std::vector<float> q_expanded;
    std::vector<float> k_expanded;
    std::vector<float> ref_state;
    std::vector<float> ref_y_head;
    if (debug_core_ref) {
        q_expanded.resize(static_cast<size_t>(num_v_heads) * head_k_dim, 0.0f);
        k_expanded.resize(static_cast<size_t>(num_v_heads) * head_k_dim, 0.0f);
        ref_state.resize(static_cast<size_t>(state_stride), 0.0f);
        ref_y_head.resize(static_cast<size_t>(head_v_dim), 0.0f);
    }

    const Qwen35SSMQkvProjectionContract projection_contract =
        ResolveQwen35SSMQkvProjectionContract(ud->projection_profile);
    const bool qkv_direct_canonical = projection_contract.raw_layout == Qwen35SSMQkvRawLayout::QKV &&
                                      projection_contract.activation == Qwen35SSMQkvActivationPlacement::NONE;
    static thread_local std::vector<float> canonical_qkv_row;
    if (!qkv_direct_canonical) {
        canonical_qkv_row.resize(static_cast<size_t>(expected_conv_channels));
    }

    for (int t = 0; t < N; ++t) {
        float* const y = y_out + static_cast<ptrdiff_t>(t) * out_stride;
        if (ith == 0) {
            const int covered = num_v_heads * head_v_dim;
            if (covered < ud->d_inner) {
                std::fill(y + covered, y + ud->d_inner, 0.0f);
            }
        }
        const float* input_t = input + static_cast<ptrdiff_t>(t) * input_stride;
        const float* z_t = z_proj + static_cast<ptrdiff_t>(t) * z_stride;
        const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;

        const float* q_base;
        const float* k_base;
        const float* v_base;

        const float* qkv_t = qkv_conv + static_cast<ptrdiff_t>(t) * qkv_stride;
        if (qkv_direct_canonical) {
            q_base = qkv_t;
            k_base = qkv_t + qk_total;
            v_base = qkv_t + 2 * qk_total;
        } else {
            if (!Qwen35CanonicalizeProjectedQkvRow(qkv_t, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                                   projection_contract, &canonical_qkv_row)) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, -1,
                                           "failed to canonicalize hybrid SSM projected QKV row");
            }
            q_base = canonical_qkv_row.data();
            k_base = canonical_qkv_row.data() + qk_total;
            v_base = canonical_qkv_row.data() + 2 * qk_total;
        }
        if (debug_core_ref) {
            for (int h = 0; h < num_v_heads; ++h) {
                const int src_k_head = resolve_src_k_head(h);
                std::memcpy(q_expanded.data() + static_cast<size_t>(h) * head_k_dim,
                            q_base + static_cast<size_t>(src_k_head) * head_k_dim,
                            static_cast<size_t>(head_k_dim) * sizeof(float));
                std::memcpy(k_expanded.data() + static_cast<size_t>(h) * head_k_dim,
                            k_base + static_cast<size_t>(src_k_head) * head_k_dim,
                            static_cast<size_t>(head_k_dim) * sizeof(float));
            }
        }

        float* token_ssm_state_base = ud->ssm_state;
        size_t token_ssm_state_elems = 0;
        if (ud->runtime_states && ud->token_seq_ids && ud->ssm_ordinal >= 0) {
            if (seq_idx >= 0 && seq_idx < static_cast<int>(ud->runtime_states->size())) {
                auto* seq_states = (*ud->runtime_states)[static_cast<size_t>(seq_idx)];
                if (seq_states && ud->ssm_ordinal < static_cast<int>(seq_states->size())) {
                    auto& seq_state = (*seq_states)[static_cast<size_t>(ud->ssm_ordinal)].ssm_state;
                    const size_t required = static_cast<size_t>(ud->n_heads) * static_cast<size_t>(state_stride);
                    if (seq_state.size() >= required) {
                        token_ssm_state_base = seq_state.data();
                        token_ssm_state_elems = seq_state.size();
                    } else {
                        token_ssm_state_base = nullptr;
                    }
                }
            }
        } else if (token_ssm_state_base) {
            token_ssm_state_elems = static_cast<size_t>(ud->n_heads) * static_cast<size_t>(state_stride);
        }

        uint64_t layer_aggregate_hash = 1469598103934665603ull;
        const bool trace_layer_aggregate = trace_contract_enabled && t < TraceQwen35SSMMaxTokens();
        for (int h = head_begin; h < head_end; ++h) {
            const int src_k_head = resolve_src_k_head(h);

            // Boundary check for QKV head slicing within the logical row
            if (src_k_head < 0 || src_k_head >= num_k_heads) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: K head index out of range");
            }
            if (h < 0 || h >= num_v_heads) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: V head index out of range");
            }

            const float* q_head = q_base + static_cast<size_t>(src_k_head) * head_k_dim;
            const float* k_head = k_base + static_cast<size_t>(src_k_head) * head_k_dim;
            const float* v_head = v_base + static_cast<size_t>(h) * head_v_dim;

            // Validate that slice pointers are within the expected bounds of the row
            const float* row_end = q_base + expected_conv_channels;
            if (q_head + head_k_dim > k_base) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: Q head slice out of bounds");
            }
            if (k_head + head_k_dim > v_base) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: K head slice out of bounds");
            }
            if (v_head + head_v_dim > row_end) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: V head slice out of bounds");
            }

            const float* alpha_row = ud->alpha_weight + static_cast<size_t>(h) * ud->n_embd;
            const float* beta_row = ud->beta_weight + static_cast<size_t>(h) * ud->n_embd;
            if (!token_ssm_state_base) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: missing recurrent state");
            }
            const size_t state_offset = static_cast<size_t>(h) * static_cast<size_t>(state_stride);
            const size_t state_elems = static_cast<size_t>(state_stride);
            if (state_offset + state_elems > token_ssm_state_elems) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h,
                                           "SSM delta: head state span exceeds resolved runtime state");
            }
            float* state = token_ssm_state_base + state_offset;
            float* y_head = y + static_cast<size_t>(h) * head_v_dim;
            const bool trace_this_head =
                trace_contract_enabled && t < TraceQwen35SSMMaxTokens() && h < TraceQwen35SSMMaxHeads();
            uint64_t q_hash = 0;
            uint64_t k_hash = 0;
            uint64_t v_hash = 0;
            uint64_t state_before_hash = 0;
            if (trace_this_head) {
                q_hash = HashQwen35SSMFloatSpan(q_head, static_cast<size_t>(head_k_dim));
                k_hash = HashQwen35SSMFloatSpan(k_head, static_cast<size_t>(head_k_dim));
                v_hash = HashQwen35SSMFloatSpan(v_head, static_cast<size_t>(head_v_dim));
                state_before_hash = HashQwen35SSMFloatSpan(state, state_elems);
            }
            const bool overlap_prev =
                h > 0 && state_offset < (static_cast<size_t>(h - 1) * static_cast<size_t>(state_stride) + state_elems);
            const bool grouped_src_reused = (num_k_heads != num_v_heads);
            const float* norm_weight_head = ud->norm_weight;
            if (ud->norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER) {
                norm_weight_head += static_cast<size_t>(h) * head_v_dim;
            } else if (ud->norm_layout != Qwen35SSMNormLayout::SHARED_HEAD_DIM) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "invalid hybrid SSM norm layout");
            }

            Qwen35SSMHeadStepConfig cfg{};
            cfg.input_t = input_t;
            cfg.q_head = q_head;
            cfg.k_head = k_head;
            cfg.v_head = v_head;
            cfg.z_head = z_t + static_cast<size_t>(h) * head_v_dim;
            cfg.alpha_row = alpha_row;
            cfg.beta_row = beta_row;
            cfg.norm_weight = norm_weight_head;
            cfg.n_embd = ud->n_embd;
            cfg.head_dim_k = head_k_dim;
            cfg.head_dim_v = head_v_dim;
            cfg.dt_bias = ud->dt_bias[h];
            cfg.a_log = ud->a_log[h];
            cfg.norm_eps = ud->norm_eps;
            cfg.a_log_prescaled = (ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL ||
                                   ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL ||
                                   ud->projection_profile == Qwen35SSMQkvProjectionProfile::QWEN38_OFFICIAL);
            cfg.use_fast_silu = ud->fast_silu_gate;
            if (alpha_beta && alpha_beta_stride >= 2 * num_v_heads) {
                const float* alpha_beta_t = alpha_beta + static_cast<ptrdiff_t>(t) * alpha_beta_stride;
                cfg.has_precomputed_alpha_beta = true;
                cfg.precomputed_alpha = alpha_beta_t[h];
                cfg.precomputed_beta = alpha_beta_t[num_v_heads + h];
            }
            if (alpha_beta && alpha_beta_stride >= 2 * num_v_heads + 3 * num_k_heads) {
                const float* qk_norm_t = alpha_beta + static_cast<ptrdiff_t>(t) * alpha_beta_stride + 2 * num_v_heads;
                cfg.has_precomputed_qk_norm = true;
                cfg.precomputed_q_inv_norm = qk_norm_t[src_k_head];
                cfg.precomputed_k_inv_norm = qk_norm_t[num_k_heads + src_k_head];
                cfg.precomputed_qk_dot = qk_norm_t[2 * num_k_heads + src_k_head];
            }

            Qwen35SSMHeadStepStats ref_stats{};
            const float* q_ref_head = q_head;
            const float* k_ref_head = k_head;
            if (debug_core_ref) {
                q_ref_head = q_expanded.data() + static_cast<size_t>(h) * head_k_dim;
                k_ref_head = k_expanded.data() + static_cast<size_t>(h) * head_k_dim;
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_mapping", "q_head", q_head, q_ref_head,
                                        head_k_dim);
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_mapping", "k_head", k_head, k_ref_head,
                                        head_k_dim);
                std::memcpy(ref_state.data(), state, static_cast<size_t>(state_stride) * sizeof(float));
                Qwen35SSMHeadStepConfig ref_cfg = cfg;
                ref_cfg.q_head = q_ref_head;
                ref_cfg.k_head = k_ref_head;
                if (!RunQwen35ReferenceHeadStep(ref_cfg, ref_state.data(), ref_y_head.data(), &ref_stats)) {
                    FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h,
                                               "RunQwen35ReferenceHeadStep rejected runtime inputs");
                }
            }

            Qwen35SSMHeadStepDebugBuffers step_debug{};
            Qwen35SSMHeadStepTrace step_trace{};
            const bool collect_step_stats = ssm_nonfinite_debug || ssm_debug;
            Qwen35SSMHeadStepStats step_stats{};
            Qwen35SSMHeadStepStats* step_stats_ptr = collect_step_stats ? &step_stats : nullptr;
            Qwen35SSMHeadStepDebugBuffers* step_debug_ptr = nullptr;
            if (collect_step_debug) {
                step_debug.q_norm = q_norm;
                step_debug.k_norm = k_norm;
                step_debug.kv_mem = kv_mem;
                step_debug.delta = delta;
                step_debug.y_pre_norm = y_pre_norm;
                step_debug_ptr = &step_debug;
            }
            bool step_ok = false;
            if (can_use_fast_default_head_step && !collect_step_stats && !step_debug_ptr) {
                // All callback workers share the same fast-default eligibility; wall
                // telemetry is emitted once below from ith==0.
                step_ok = Qwen35RunGatedDeltaHeadStepFastDefault(cfg, state, y_head);
            } else {
                step_ok = Qwen35RunGatedDeltaHeadStep(cfg, state, y_head, step_stats_ptr, step_debug_ptr);
                if (trace_this_head || trace_layer_aggregate) {
                    if (state_before_hash == 0) {
                        state_before_hash = HashQwen35SSMFloatSpan(state, state_elems);
                    }
                    step_trace.state_in_hash = state_before_hash;
                    step_trace.state_out_hash = HashQwen35SSMFloatSpan(state, state_elems);
                    step_trace.y_hash = HashQwen35SSMFloatSpan(y_head, static_cast<size_t>(head_v_dim));
                }
            }
            if (!step_ok) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h,
                                           "Qwen35RunGatedDeltaHeadStep rejected runtime inputs");
            }
            if (trace_this_head) {
                LogQwen35SSMRuntimeContract("normal", ud->layer_idx, t, seq_idx, h, src_k_head, heads_per_group,
                                            reinterpret_cast<uintptr_t>(token_ssm_state_base), token_ssm_state_elems,
                                            state_offset * sizeof(float), state_elems * sizeof(float), input_stride,
                                            out_stride, overlap_prev, grouped_src_reused, q_hash, k_hash, v_hash,
                                            state_before_hash, step_trace.state_out_hash, step_trace.y_hash);
            }
            if (trace_layer_aggregate) {
                layer_aggregate_hash ^= step_trace.state_out_hash;
                layer_aggregate_hash *= 1099511628211ull;
            }
            if (debug_core_ref) {
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_reference", "state", state,
                                        ref_state.data(), state_stride);
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_reference", "y_head", y_head,
                                        ref_y_head.data(), head_v_dim);
            }

            if (ssm_nonfinite_debug) {
                Qwen35SSMDebugScalars dbg{};
                dbg.alpha = step_stats.alpha;
                dbg.beta = step_stats.beta;
                dbg.softplus_alpha = step_stats.softplus_alpha;
                dbg.exp_a_log = step_stats.exp_a_log;
                dbg.g = step_stats.g;
                dbg.decay = step_stats.decay;
                dbg.beta_gate = step_stats.beta_gate;
                dbg.q_sum_sq = step_stats.q_sum_sq;
                dbg.k_sum_sq = step_stats.k_sum_sq;
                dbg.rms = step_stats.rms;
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "alpha_projection", "alpha", step_stats.alpha, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "beta_projection", "beta", step_stats.beta, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "softplus_alpha", step_stats.softplus_alpha,
                                     dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "exp_a_log", step_stats.exp_a_log, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "g", step_stats.g, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "decay", step_stats.decay, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "qk_norm", "q_sum_sq", step_stats.q_sum_sq, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "qk_norm", "k_sum_sq", step_stats.k_sum_sq, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "qk_norm", "q_norm", q_norm, head_k_dim, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "qk_norm", "k_norm", k_norm, head_k_dim, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "kv_mem", "kv_mem", kv_mem, head_v_dim, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "delta", "delta", delta, head_v_dim, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "state_update", "state", state, state_stride, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "y_pre_norm", "y_head", y_pre_norm, head_v_dim, dbg);
                CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "y_post_norm", "rms", step_stats.rms, dbg);
                CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "y_post_norm", "y_head", y_head, head_v_dim, dbg);
            }

            if (ssm_debug && t == 0 && h == 0) {
                fprintf(stderr,
                        "[QWEN35_SSM] alpha=%.4f g=%.4f decay=%.4f beta=%.4f q0=%.4f k0=%.4f v0=%.4f z0=%.4f y0=%.4f\n",
                        step_stats.alpha, step_stats.g, step_stats.decay, step_stats.beta_gate, q_norm[0], k_norm[0],
                        v_head[0], cfg.z_head[0], y_head[0]);
            }
        }

        if (head_begin < head_end && ShouldLogSSMDeltaStats(ud->layer_idx, t) && ConsumeSSMDeltaStatsBudget()) {
            double sum = 0.0;
            double sum_sq = 0.0;
            double z_sum_sq = 0.0;
            double q_sum_sq = 0.0;
            double k_sum_sq = 0.0;
            double v_sum_sq = 0.0;
            double input_sum_sq = 0.0;
            float max_abs = 0.0f;
            float z_max_abs = 0.0f;
            float q_max_abs = 0.0f;
            float k_max_abs = 0.0f;
            float v_max_abs = 0.0f;
            float input_max_abs = 0.0f;
            int finite_count = 0;
            int q_finite_count = 0;
            int k_finite_count = 0;
            int v_finite_count = 0;
            int input_finite_count = 0;
            int first_bad_idx = -1;
            const int elem_begin = head_begin * head_v_dim;
            const int elem_end = head_end * head_v_dim;
            for (int i = elem_begin; i < elem_end; ++i) {
                const float v = y[i];
                if (!std::isfinite(v)) {
                    if (first_bad_idx < 0) {
                        first_bad_idx = i;
                    }
                    continue;
                }
                ++finite_count;
                sum += static_cast<double>(v);
                sum_sq += static_cast<double>(v) * static_cast<double>(v);
                max_abs = std::max(max_abs, std::fabs(v));
                const float z_v = z_t[i];
                if (std::isfinite(z_v)) {
                    z_sum_sq += static_cast<double>(z_v) * static_cast<double>(z_v);
                    z_max_abs = std::max(z_max_abs, std::fabs(z_v));
                }
            }
            for (int h = head_begin; h < head_end; ++h) {
                const int src_k_head = resolve_src_k_head(h);
                const float* q_head = q_base + static_cast<size_t>(src_k_head) * head_k_dim;
                const float* k_head = k_base + static_cast<size_t>(src_k_head) * head_k_dim;
                const float* v_head = v_base + static_cast<size_t>(h) * head_v_dim;
                for (int i = 0; i < head_k_dim; ++i) {
                    const float q_v = q_head[i];
                    const float k_v = k_head[i];
                    if (std::isfinite(q_v)) {
                        ++q_finite_count;
                        q_sum_sq += static_cast<double>(q_v) * static_cast<double>(q_v);
                        q_max_abs = std::max(q_max_abs, std::fabs(q_v));
                    }
                    if (std::isfinite(k_v)) {
                        ++k_finite_count;
                        k_sum_sq += static_cast<double>(k_v) * static_cast<double>(k_v);
                        k_max_abs = std::max(k_max_abs, std::fabs(k_v));
                    }
                }
                for (int i = 0; i < head_v_dim; ++i) {
                    const float v_v = v_head[i];
                    if (std::isfinite(v_v)) {
                        ++v_finite_count;
                        v_sum_sq += static_cast<double>(v_v) * static_cast<double>(v_v);
                        v_max_abs = std::max(v_max_abs, std::fabs(v_v));
                    }
                }
            }
            if (ith == 0) {
                for (int i = 0; i < ud->n_embd; ++i) {
                    const float input_v = input_t[i];
                    if (std::isfinite(input_v)) {
                        ++input_finite_count;
                        input_sum_sq += static_cast<double>(input_v) * static_cast<double>(input_v);
                        input_max_abs = std::max(input_max_abs, std::fabs(input_v));
                    }
                }
            }
            const double denom = static_cast<double>(std::max(1, finite_count));
            const double mean = sum / denom;
            const double rms = std::sqrt(sum_sq / denom);
            const double z_rms = std::sqrt(z_sum_sq / denom);
            const double q_rms = std::sqrt(q_sum_sq / static_cast<double>(std::max(1, q_finite_count)));
            const double k_rms = std::sqrt(k_sum_sq / static_cast<double>(std::max(1, k_finite_count)));
            const double v_rms = std::sqrt(v_sum_sq / static_cast<double>(std::max(1, v_finite_count)));
            const double input_rms = std::sqrt(input_sum_sq / static_cast<double>(std::max(1, input_finite_count)));
            const int seq_idx_for_log = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
            std::fprintf(stderr,
                         "[SSM_DELTA_STATS] layer=%d token=%d seq=%d ith=%d nth=%d heads=[%d,%d) elems=[%d,%d) "
                         "finite=%d first_bad_idx=%d max_abs=%.8g rms=%.8g mean=%.8g first=%.8g last=%.8g "
                         "z_max_abs=%.8g z_rms=%.8g q_max_abs=%.8g q_rms=%.8g k_max_abs=%.8g k_rms=%.8g "
                         "v_max_abs=%.8g v_rms=%.8g input_max_abs=%.8g input_rms=%.8g input_finite=%d\n",
                         ud->layer_idx, t, seq_idx_for_log, ith, task_count, head_begin, head_end, elem_begin, elem_end,
                         finite_count, first_bad_idx, static_cast<double>(max_abs), rms, mean,
                         static_cast<double>(y[elem_begin]), static_cast<double>(y[elem_end - 1]),
                         static_cast<double>(z_max_abs), z_rms, static_cast<double>(q_max_abs), q_rms,
                         static_cast<double>(k_max_abs), k_rms, static_cast<double>(v_max_abs), v_rms,
                         static_cast<double>(input_max_abs), input_rms, input_finite_count);
        }

        if (trace_layer_aggregate) {
            LogQwen35SSMLayerAggregate("normal", ud->layer_idx, t, seq_idx, layer_aggregate_hash);
        }
    }
    if (ud && ud->profile && ith == 0) {
        ud->profile->ssm_delta_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (IsQwen36ProfilingEnabled() && ud && ud->profile) {
        const auto profile_end = std::chrono::steady_clock::now();
        const uint64_t elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(profile_end - profile_begin).count());
        AddQwen36ProfileNs(ud->profile->ssm_delta_ns, elapsed_ns);
        if (ith == 0) {
            AddQwen36ProfileNs(ud->profile->ssm_delta_wall_ns, elapsed_ns);
            if (can_use_fast_default_head_step) {
                ud->profile->ssm_delta_fast_default_used_ops.fetch_add(1, std::memory_order_relaxed);
                AddQwen36ProfileNs(ud->profile->ssm_delta_fast_default_wall_ns, elapsed_ns);
            }
        }
    }
}

void cb_ssm_qwen35_delta_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (!dst) {
        return;
    }
    cb_ssm_qwen35_delta(dst, dst->src[0], dst->src[1], dst->src[2], ith, nth, userdata);
}
