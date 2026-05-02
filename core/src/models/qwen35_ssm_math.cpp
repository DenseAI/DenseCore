#include "densecore/models/qwen35_ssm_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define DENSECORE_NEON_SSM 1
#endif

namespace {

inline uint64_t Fnv1aInit() {
    return 1469598103934665603ull;
}

inline void Fnv1aMixU32(uint64_t* hash, uint32_t value) {
    if (!hash) {
        return;
    }
    *hash ^= static_cast<uint64_t>(value);
    *hash *= 1099511628211ull;
}

inline uint64_t HashFloatSpan(const float* data, size_t count) {
    if (!data) {
        return 0;
    }
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU32(&hash, static_cast<uint32_t>(count & 0xffffffffu));
    Fnv1aMixU32(&hash, static_cast<uint32_t>((count >> 32) & 0xffffffffu));
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        Fnv1aMixU32(&hash, bits);
    }
    return hash;
}

inline float SoftplusStable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float SigmoidStable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

inline float SiluStable(float x) {
    return x * SigmoidStable(x);
}

inline float ResolveSSMA(float stored, bool prefer_materialized) {
    if (prefer_materialized && stored <= 0.0f) {
        return stored;
    }
    return -std::exp(stored);
}

inline bool IsVectorShape(const int64_t ne[4], int expected) {
    return ne && expected > 0 && ne[0] == expected && ne[1] == 1 && ne[2] == 1 && ne[3] == 1;
}

inline float DotSquares(const float* values, int n) {
    float sum = 0.0f;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(values + i);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(v, v));
    }
    sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
        sum += values[i] * values[i];
    }
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(values + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    for (float lane : lanes) {
        sum += lane;
    }
    for (; i < n; ++i) {
        sum += values[i] * values[i];
    }
#elif defined(DENSECORE_NEON_SSM)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(values + i);
        acc = vmlaq_f32(acc, v, v);
    }
    sum = vaddvq_f32(acc);
    for (; i < n; ++i) {
        sum += values[i] * values[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        sum += values[i] * values[i];
    }
#endif
    return sum;
}

inline void ScaleCopy(float* dst, const float* src, float scale, int n) {
#if defined(__AVX512F__)
    const __m512 vscale = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), vscale));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#elif defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vscale));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#elif defined(DENSECORE_NEON_SSM)
    const float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), vscale));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#endif
}

inline void ScaleInPlace(float* values, float scale, size_t n) {
#if defined(__AVX512F__)
    const __m512 vscale = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(values + i, _mm512_mul_ps(_mm512_loadu_ps(values + i), vscale));
    }
    for (; i < n; ++i) {
        values[i] *= scale;
    }
#elif defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(values + i, _mm256_mul_ps(_mm256_loadu_ps(values + i), vscale));
    }
    for (; i < n; ++i) {
        values[i] *= scale;
    }
#elif defined(DENSECORE_NEON_SSM)
    const float32x4_t vscale = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(values + i, vmulq_f32(vld1q_f32(values + i), vscale));
    }
    for (; i < n; ++i) {
        values[i] *= scale;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        values[i] *= scale;
    }
#endif
}

inline void AccumulateScaled(float* dst, const float* src, float scale, int n) {
#if defined(__AVX512F__)
    const __m512 vscale = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 d = _mm512_loadu_ps(dst + i);
        const __m512 s = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, _mm512_add_ps(d, _mm512_mul_ps(s, vscale)));
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#elif defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 d = _mm256_loadu_ps(dst + i);
        const __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, _mm256_mul_ps(s, vscale)));
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#elif defined(DENSECORE_NEON_SSM)
    const float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t d = vld1q_f32(dst + i);
        const float32x4_t s = vld1q_f32(src + i);
        vst1q_f32(dst + i, vmlaq_f32(d, s, vscale));
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#endif
}

inline float SumSquares(const float* values, int n) {
    return DotSquares(values, n);
}

bool IsQwen36SSMDebugTimingEnabled() {
    const char* env = std::getenv("DENSECORE_QWEN36_SSM_DEBUG_TIMING");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

inline void ApplyRmsNormGate(float* y_head, const float* norm_weight, const float* z_head, float inv_rms, int n) {
#if defined(__AVX512F__)
    const __m512 v_inv_rms = _mm512_set1_ps(inv_rms);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 y = _mm512_mul_ps(_mm512_loadu_ps(y_head + i), v_inv_rms);
        const __m512 z = _mm512_loadu_ps(z_head + i);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 zero = _mm512_setzero_ps();
        const __mmask16 pos_mask = _mm512_cmp_ps_mask(z, zero, _CMP_GE_OQ);
        const __m512 exp_arg = _mm512_mask_blend_ps(pos_mask, z, _mm512_sub_ps(zero, z));
        alignas(64) float exp_buf[16];
        _mm512_store_ps(exp_buf, exp_arg);
        for (float& v : exp_buf) {
            v = std::exp(v);
        }
        __m512 exp_v = _mm512_load_ps(exp_buf);
        const __m512 sigmoid_pos = _mm512_div_ps(one, _mm512_add_ps(one, exp_v));
        const __m512 sigmoid_neg = _mm512_div_ps(exp_v, _mm512_add_ps(one, exp_v));
        const __m512 silu = _mm512_mul_ps(z, _mm512_mask_blend_ps(pos_mask, sigmoid_neg, sigmoid_pos));
        const __m512 norm = norm_weight ? _mm512_loadu_ps(norm_weight + i) : one;
        _mm512_storeu_ps(y_head + i, _mm512_mul_ps(_mm512_mul_ps(y, norm), silu));
    }
    for (; i < n; ++i) {
        const float norm = norm_weight ? norm_weight[i] : 1.0f;
        y_head[i] = (y_head[i] * inv_rms) * norm * SiluStable(z_head[i]);
    }
#elif defined(__AVX2__)
    const __m256 v_inv_rms = _mm256_set1_ps(inv_rms);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 y = _mm256_mul_ps(_mm256_loadu_ps(y_head + i), v_inv_rms);
        alignas(32) float z_buf[8];
        _mm256_store_ps(z_buf, _mm256_loadu_ps(z_head + i));
        alignas(32) float silu_buf[8];
        for (int lane = 0; lane < 8; ++lane) {
            silu_buf[lane] = SiluStable(z_buf[lane]);
        }
        const __m256 silu = _mm256_load_ps(silu_buf);
        const __m256 norm = norm_weight ? _mm256_loadu_ps(norm_weight + i) : _mm256_set1_ps(1.0f);
        _mm256_storeu_ps(y_head + i, _mm256_mul_ps(_mm256_mul_ps(y, norm), silu));
    }
    for (; i < n; ++i) {
        const float norm = norm_weight ? norm_weight[i] : 1.0f;
        y_head[i] = (y_head[i] * inv_rms) * norm * SiluStable(z_head[i]);
    }
#elif defined(DENSECORE_NEON_SSM)
    const float32x4_t v_inv_rms = vdupq_n_f32(inv_rms);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        alignas(16) float z_buf[4];
        vst1q_f32(z_buf, vld1q_f32(z_head + i));
        alignas(16) float silu_buf[4];
        for (int lane = 0; lane < 4; ++lane) {
            silu_buf[lane] = SiluStable(z_buf[lane]);
        }
        const float32x4_t y = vmulq_f32(vld1q_f32(y_head + i), v_inv_rms);
        const float32x4_t silu = vld1q_f32(silu_buf);
        const float32x4_t norm = norm_weight ? vld1q_f32(norm_weight + i) : vdupq_n_f32(1.0f);
        vst1q_f32(y_head + i, vmulq_f32(vmulq_f32(y, norm), silu));
    }
    for (; i < n; ++i) {
        const float norm = norm_weight ? norm_weight[i] : 1.0f;
        y_head[i] = (y_head[i] * inv_rms) * norm * SiluStable(z_head[i]);
    }
#else
    for (int i = 0; i < n; ++i) {
        const float norm = norm_weight ? norm_weight[i] : 1.0f;
        y_head[i] = (y_head[i] * inv_rms) * norm * SiluStable(z_head[i]);
    }
#endif
}

}  // namespace

bool Qwen35CanonicalizeHeadByEmbd(const float* raw, const int64_t ne[4], int n_embd, int n_heads,
                                  std::vector<float>* out) {
    if (!raw || !ne || !out || n_embd <= 0 || n_heads <= 0 || ne[2] != 1 || ne[3] != 1) {
        return false;
    }
    out->assign(static_cast<size_t>(n_embd) * static_cast<size_t>(n_heads), 0.0f);
    if (ne[0] == n_embd && ne[1] == n_heads) {
        std::copy(raw, raw + out->size(), out->begin());
        return true;
    }
    if (ne[0] == n_heads && ne[1] == n_embd) {
        for (int h = 0; h < n_heads; ++h) {
            for (int i = 0; i < n_embd; ++i) {
                (*out)[static_cast<size_t>(h) * n_embd + i] = raw[static_cast<size_t>(i) * n_heads + h];
            }
        }
        return true;
    }
    out->clear();
    return false;
}

bool Qwen35CanonicalizeFusedBA(const float* raw, const int64_t ne[4], int n_embd, int n_v_heads, int n_groups,
                               std::vector<float>* beta_out, std::vector<float>* alpha_out) {
    if (!raw || !ne || !beta_out || !alpha_out || n_embd <= 0 || n_v_heads <= 0 || n_groups <= 0 || ne[2] != 1 ||
        ne[3] != 1 || (n_v_heads % n_groups) != 0) {
        return false;
    }

    const int heads_per_group = n_v_heads / n_groups;
    const int fused_width = 2 * n_v_heads;
    beta_out->assign(static_cast<size_t>(n_embd) * static_cast<size_t>(n_v_heads), 0.0f);
    alpha_out->assign(static_cast<size_t>(n_embd) * static_cast<size_t>(n_v_heads), 0.0f);

    auto beta_slot = [&](int head_idx) { return beta_out->data() + static_cast<size_t>(head_idx) * n_embd; };
    auto alpha_slot = [&](int head_idx) { return alpha_out->data() + static_cast<size_t>(head_idx) * n_embd; };
    const bool swap_qwen36_fused_ba = []() {
        const char* env = std::getenv("DENSECORE_QWEN36_SWAP_FUSED_BA_ORDER");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();

    if (ne[0] == n_embd && ne[1] == fused_width) {
        for (int group = 0; group < n_groups; ++group) {
            const int group_base = group * heads_per_group;
            const int fused_base = group * (2 * heads_per_group);
            for (int local_head = 0; local_head < heads_per_group; ++local_head) {
                const int head_idx = group_base + local_head;
                const int first_col = fused_base + local_head;
                const int second_col = fused_base + heads_per_group + local_head;
                const int beta_col = swap_qwen36_fused_ba ? second_col : first_col;
                const int alpha_col = swap_qwen36_fused_ba ? first_col : second_col;
                std::copy(raw + static_cast<size_t>(beta_col) * n_embd,
                          raw + static_cast<size_t>(beta_col + 1) * n_embd, beta_slot(head_idx));
                std::copy(raw + static_cast<size_t>(alpha_col) * n_embd,
                          raw + static_cast<size_t>(alpha_col + 1) * n_embd, alpha_slot(head_idx));
            }
        }
        return true;
    }

    if (ne[0] == fused_width && ne[1] == n_embd) {
        for (int group = 0; group < n_groups; ++group) {
            const int group_base = group * heads_per_group;
            const int fused_base = group * (2 * heads_per_group);
            for (int local_head = 0; local_head < heads_per_group; ++local_head) {
                const int head_idx = group_base + local_head;
                const int first_row = fused_base + local_head;
                const int second_row = fused_base + heads_per_group + local_head;
                const int beta_row = swap_qwen36_fused_ba ? second_row : first_row;
                const int alpha_row = swap_qwen36_fused_ba ? first_row : second_row;
                for (int embd_idx = 0; embd_idx < n_embd; ++embd_idx) {
                    beta_slot(head_idx)[embd_idx] = raw[static_cast<size_t>(embd_idx) * fused_width + beta_row];
                    alpha_slot(head_idx)[embd_idx] = raw[static_cast<size_t>(embd_idx) * fused_width + alpha_row];
                }
            }
        }
        return true;
    }

    beta_out->clear();
    alpha_out->clear();
    return false;
}

bool Qwen35CanonicalizePerHeadVector(const float* raw, const int64_t ne[4], int n_heads, std::vector<float>* out) {
    if (!raw || !out || !IsVectorShape(ne, n_heads)) {
        return false;
    }
    out->assign(raw, raw + n_heads);
    return true;
}

void Qwen35ReorderVHeadsGroupedToTiled(std::vector<float>* values, int row_width, int num_k_heads, int num_v_heads) {
    if (!values || row_width <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0) {
        return;
    }
    const int num_v_per_k = num_v_heads / num_k_heads;
    if (num_v_per_k <= 1 || values->size() != static_cast<size_t>(row_width) * static_cast<size_t>(num_v_heads)) {
        return;
    }

    std::vector<float> reordered(values->size(), 0.0f);
    for (int tiled_idx = 0; tiled_idx < num_v_heads; ++tiled_idx) {
        const int src_group = tiled_idx % num_k_heads;
        const int src_local = tiled_idx / num_k_heads;
        const int src_idx = src_group * num_v_per_k + src_local;
        std::memcpy(reordered.data() + static_cast<size_t>(tiled_idx) * row_width,
                    values->data() + static_cast<size_t>(src_idx) * row_width,
                    static_cast<size_t>(row_width) * sizeof(float));
    }
    values->swap(reordered);
}

Qwen35SSMNormLayout Qwen35CanonicalizeNorm(const float* raw, const int64_t ne[4], int head_dim_v, int d_inner,
                                           std::vector<float>* out) {
    if (!raw || !out) {
        return Qwen35SSMNormLayout::INVALID;
    }
    if (IsVectorShape(ne, head_dim_v)) {
        out->assign(raw, raw + head_dim_v);
        return Qwen35SSMNormLayout::SHARED_HEAD_DIM;
    }
    if (IsVectorShape(ne, d_inner)) {
        out->assign(raw, raw + d_inner);
        return Qwen35SSMNormLayout::FLATTENED_D_INNER;
    }
    out->clear();
    return Qwen35SSMNormLayout::INVALID;
}

Qwen35SSMQkvProjectionContract ResolveQwen35SSMQkvProjectionContract(Qwen35SSMQkvProjectionProfile profile) {
    Qwen35SSMQkvProjectionContract contract;
    contract.profile = profile;
    switch (profile) {
    case Qwen35SSMQkvProjectionProfile::QWEN35_LEGACY:
        contract.raw_layout = Qwen35SSMQkvRawLayout::QKV;
        contract.activation = Qwen35SSMQkvActivationPlacement::FULL_ROW_SILU;
        contract.share_qk_by_group = true;
        contract.dense_value_heads = true;
        break;
    case Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL:
        contract.raw_layout = Qwen35SSMQkvRawLayout::QKV;
        contract.activation = Qwen35SSMQkvActivationPlacement::NONE;
        contract.share_qk_by_group = true;
        contract.dense_value_heads = true;
        break;
    }
    return contract;
}

bool Qwen35CanonicalizeProjectedQkvRow(const float* raw_row, int num_k_heads, int num_v_heads, int head_dim_k,
                                       int head_dim_v, const Qwen35SSMQkvProjectionContract& contract,
                                       std::vector<float>* canonical_qkv) {
    if (!raw_row || !canonical_qkv || num_k_heads <= 0 || num_v_heads <= 0 || head_dim_k <= 0 || head_dim_v <= 0) {
        return false;
    }

    const int qk_total = num_k_heads * head_dim_k;
    const int value_total = num_v_heads * head_dim_v;
    const int conv_channels = qk_total * 2 + value_total;
    if (conv_channels <= 0) {
        return false;
    }

    canonical_qkv->assign(static_cast<size_t>(conv_channels), 0.0f);
    auto apply_activation = [&](float value) {
        switch (contract.activation) {
        case Qwen35SSMQkvActivationPlacement::NONE: return value;
        case Qwen35SSMQkvActivationPlacement::FULL_ROW_SILU: return SiluStable(value);
        }
        return value;
    };
    auto copy_span = [&](int dst_offset, int src_offset, int count) {
        for (int i = 0; i < count; ++i) {
            (*canonical_qkv)[static_cast<size_t>(dst_offset + i)] = apply_activation(raw_row[src_offset + i]);
        }
    };

    switch (contract.raw_layout) {
    case Qwen35SSMQkvRawLayout::QKV:
        copy_span(/*dst_offset=*/0, /*src_offset=*/0, qk_total);
        copy_span(/*dst_offset=*/qk_total, /*src_offset=*/qk_total, qk_total);
        copy_span(/*dst_offset=*/2 * qk_total, /*src_offset=*/2 * qk_total, value_total);
        break;
    }

    return true;
}

size_t Qwen35SSMHeadStateElements(int head_dim_k, int head_dim_v) {
    if (head_dim_k <= 0 || head_dim_v <= 0) {
        return 0;
    }
    return static_cast<size_t>(head_dim_k) * static_cast<size_t>(head_dim_v);
}

bool Qwen35RunGatedDeltaHeadStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head,
                                 Qwen35SSMHeadStepStats* stats, Qwen35SSMHeadStepDebugBuffers* debug) {
    if (!cfg.input_t || !cfg.q_head || !cfg.k_head || !cfg.v_head || !cfg.z_head || !cfg.alpha_row || !cfg.beta_row ||
        !state_kv || !y_head || cfg.n_embd <= 0 || cfg.head_dim_k <= 0 || cfg.head_dim_v <= 0) {
        return false;
    }

    // thread_local reuse buffers — eliminates ~1,536 heap allocs/token
    // (24 SSM layers × ~64 heads × 3 vectors per call)
    static thread_local std::vector<float> q_norm;
    static thread_local std::vector<float> k_norm;
    static thread_local std::vector<float> kv_mem;
    static thread_local std::vector<float> delta;
    q_norm.resize(static_cast<size_t>(cfg.head_dim_k));
    k_norm.resize(static_cast<size_t>(cfg.head_dim_k));
    kv_mem.resize(static_cast<size_t>(cfg.head_dim_v));
    delta.resize(static_cast<size_t>(cfg.head_dim_v));
    const bool log_timing = IsQwen36SSMDebugTimingEnabled();
    const bool collect_timing = stats != nullptr || log_timing;
    const auto ms_since = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };

    const auto alpha_beta_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    float alpha = cfg.dt_bias;
    float beta = 0.0f;
    for (int i = 0; i < cfg.n_embd; ++i) {
        alpha += cfg.alpha_row[i] * cfg.input_t[i];
        beta += cfg.beta_row[i] * cfg.input_t[i];
    }
    const double alpha_beta_dot_ms = collect_timing ? ms_since(alpha_beta_begin) : 0.0;

    const float softplus_alpha = SoftplusStable(alpha);
    const float ssm_a = ResolveSSMA(cfg.a_log, cfg.a_log_prescaled);
    const float exp_a_log = -ssm_a;
    const float g = ssm_a * softplus_alpha;
    const float decay = std::exp(g);
    const float beta_gate = SigmoidStable(beta);

    const auto norm_begin = collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const float q_sum_sq = DotSquares(cfg.q_head, cfg.head_dim_k);
    const float k_sum_sq = DotSquares(cfg.k_head, cfg.head_dim_k);
    const float q_inv_norm = 1.0f / std::max(std::sqrt(q_sum_sq), cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::max(std::sqrt(k_sum_sq), cfg.norm_eps);
    ScaleCopy(q_norm.data(), cfg.q_head, q_inv_norm, cfg.head_dim_k);
    ScaleCopy(k_norm.data(), cfg.k_head, k_inv_norm, cfg.head_dim_k);
    if (debug && debug->q_norm) {
        for (int i = 0; i < cfg.head_dim_k; ++i) {
            debug->q_norm[i] = q_norm[static_cast<size_t>(i)];
        }
    }
    if (debug && debug->k_norm) {
        for (int i = 0; i < cfg.head_dim_k; ++i) {
            debug->k_norm[i] = k_norm[static_cast<size_t>(i)];
        }
    }
    const double norm_ms = collect_timing ? ms_since(norm_begin) : 0.0;

    const size_t state_elems = static_cast<size_t>(cfg.head_dim_k) * static_cast<size_t>(cfg.head_dim_v);

    const auto decay_state_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ScaleInPlace(state_kv, decay, state_elems);
    const double decay_state_ms = collect_timing ? ms_since(decay_state_begin) : 0.0;

    const auto kv_mem_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ScaleCopy(kv_mem.data(), state_kv, k_norm[0], cfg.head_dim_v);
    for (int k = 1; k < cfg.head_dim_k; ++k) {
        const float* state_row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        AccumulateScaled(kv_mem.data(), state_row, k_norm[static_cast<size_t>(k)], cfg.head_dim_v);
    }
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        delta[static_cast<size_t>(v)] = (cfg.v_head[v] - kv_mem[static_cast<size_t>(v)]) * beta_gate;
    }
    if (debug && debug->kv_mem) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->kv_mem[v] = kv_mem[static_cast<size_t>(v)];
        }
    }
    if (debug && debug->delta) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->delta[v] = delta[static_cast<size_t>(v)];
        }
    }
    const double kv_mem_ms = collect_timing ? ms_since(kv_mem_begin) : 0.0;

    // State update: state_kv[k, v] += k_norm[k] * delta[v]
    const auto state_update_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (int k = 0; k < cfg.head_dim_k; ++k) {
        const float k_val = k_norm[static_cast<size_t>(k)];
        float* state_row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        AccumulateScaled(state_row, delta.data(), k_val, cfg.head_dim_v);
    }
    const double state_update_ms = collect_timing ? ms_since(state_update_begin) : 0.0;

    const auto output_accum_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ScaleCopy(y_head, state_kv, q_norm[0], cfg.head_dim_v);
    for (int k = 1; k < cfg.head_dim_k; ++k) {
        const float* state_row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        AccumulateScaled(y_head, state_row, q_norm[static_cast<size_t>(k)], cfg.head_dim_v);
    }
    // llama.cpp's fused GDN applies the attention scale after the state-query
    // dot product. Keeping it here preserves q/k l2_norm parity while matching
    // the fused output contract.
    ScaleInPlace(y_head, 1.0f / std::sqrt(static_cast<float>(cfg.head_dim_v)), cfg.head_dim_v);
    if (debug && debug->y_pre_norm) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->y_pre_norm[v] = y_head[v];
        }
    }
    const double output_accum_ms = collect_timing ? ms_since(output_accum_begin) : 0.0;

    const auto rms_gate_begin =
        collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const float sum_sq = SumSquares(y_head, cfg.head_dim_v);
    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    const float inv_rms = 1.0f / rms;
    ApplyRmsNormGate(y_head, cfg.norm_weight, cfg.z_head, inv_rms, cfg.head_dim_v);
    const double rms_gate_ms = collect_timing ? ms_since(rms_gate_begin) : 0.0;

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
        stats->alpha_beta_dot_ms = alpha_beta_dot_ms;
        stats->norm_ms = norm_ms;
        stats->decay_state_ms = decay_state_ms;
        stats->kv_mem_ms = kv_mem_ms;
        stats->state_update_ms = state_update_ms;
        stats->output_accum_ms = output_accum_ms;
        stats->rms_gate_ms = rms_gate_ms;
    }

    if (log_timing) {
        std::fprintf(stderr,
                     "[Qwen36SSMTiming] n_embd=%d head_k=%d head_v=%d alpha_beta_dot_ms=%.3f norm_ms=%.3f "
                     "decay_state_ms=%.3f kv_mem_ms=%.3f state_update_ms=%.3f output_accum_ms=%.3f "
                     "rms_gate_ms=%.3f total_ms=%.3f\n",
                     cfg.n_embd, cfg.head_dim_k, cfg.head_dim_v, alpha_beta_dot_ms, norm_ms, decay_state_ms, kv_mem_ms,
                     state_update_ms, output_accum_ms, rms_gate_ms,
                     alpha_beta_dot_ms + norm_ms + decay_state_ms + kv_mem_ms + state_update_ms + output_accum_ms +
                         rms_gate_ms);
    }

    return true;
}

bool Qwen35RunGatedDeltaHeadStepWithWriteback(const Qwen35SSMHeadStepConfig& cfg, const float* state_in_kv,
                                              float* state_out_kv, float* y_head, Qwen35SSMHeadStepStats* stats,
                                              Qwen35SSMHeadStepDebugBuffers* debug) {
    if (!state_in_kv || !state_out_kv || !y_head) {
        return false;
    }

    const size_t state_elems = Qwen35SSMHeadStateElements(cfg.head_dim_k, cfg.head_dim_v);
    if (state_elems == 0) {
        return false;
    }

    static thread_local std::vector<float> state_shadow;
    state_shadow.resize(state_elems);
    std::memcpy(state_shadow.data(), state_in_kv, state_elems * sizeof(float));

    if (!Qwen35RunGatedDeltaHeadStep(cfg, state_shadow.data(), y_head, stats, debug)) {
        return false;
    }

    std::memcpy(state_out_kv, state_shadow.data(), state_elems * sizeof(float));
    return true;
}

bool Qwen35RunGatedDeltaHeadStepReferenceSafe(const Qwen35SSMHeadStepConfig& cfg, const float* state_in_kv,
                                              float* state_out_kv, float* y_head, Qwen35SSMHeadStepStats* stats,
                                              Qwen35SSMHeadStepDebugBuffers* debug, Qwen35SSMHeadStepTrace* trace) {
    if (!state_in_kv || !state_out_kv || !y_head) {
        return false;
    }

    const size_t state_elems = Qwen35SSMHeadStateElements(cfg.head_dim_k, cfg.head_dim_v);
    if (state_elems == 0) {
        return false;
    }

    static thread_local std::vector<float> state_shadow;
    static thread_local std::vector<float> state_zeroed_writeback;
    state_shadow.resize(state_elems);
    state_zeroed_writeback.resize(state_elems);

    std::memcpy(state_shadow.data(), state_in_kv, state_elems * sizeof(float));
    if (trace) {
        trace->state_in_hash = HashFloatSpan(state_shadow.data(), state_elems);
    }

    if (!Qwen35RunGatedDeltaHeadStep(cfg, state_shadow.data(), y_head, stats, debug)) {
        return false;
    }

    std::fill(state_zeroed_writeback.begin(), state_zeroed_writeback.end(), 0.0f);
    std::memcpy(state_zeroed_writeback.data(), state_shadow.data(), state_elems * sizeof(float));
    std::memcpy(state_out_kv, state_zeroed_writeback.data(), state_elems * sizeof(float));

    if (trace) {
        trace->state_out_hash = HashFloatSpan(state_out_kv, state_elems);
        trace->y_hash = HashFloatSpan(y_head, static_cast<size_t>(cfg.head_dim_v));
    }
    return true;
}
