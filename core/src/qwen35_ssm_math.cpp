#include "qwen35_ssm_math.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define DENSECORE_NEON_SSM 1
#endif

namespace {

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

inline bool IsVectorShape(const int64_t ne[4], int expected) {
    return ne && expected > 0 && ne[0] == expected && ne[1] == 1 && ne[2] == 1 && ne[3] == 1;
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

    if (ne[0] == n_embd && ne[1] == fused_width) {
        for (int group = 0; group < n_groups; ++group) {
            const int group_base = group * heads_per_group;
            const int fused_base = group * (2 * heads_per_group);
            for (int local_head = 0; local_head < heads_per_group; ++local_head) {
                const int head_idx = group_base + local_head;
                const int beta_col = fused_base + local_head;
                const int alpha_col = fused_base + heads_per_group + local_head;
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
                const int beta_row = fused_base + local_head;
                const int alpha_row = fused_base + heads_per_group + local_head;
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
    static thread_local std::vector<float> delta;
    q_norm.resize(static_cast<size_t>(cfg.head_dim_k));
    k_norm.resize(static_cast<size_t>(cfg.head_dim_k));
    delta.resize(static_cast<size_t>(cfg.head_dim_v));

    float alpha = cfg.dt_bias;
    float beta = 0.0f;
    for (int i = 0; i < cfg.n_embd; ++i) {
        alpha += cfg.alpha_row[i] * cfg.input_t[i];
        beta += cfg.beta_row[i] * cfg.input_t[i];
    }

    const float softplus_alpha = SoftplusStable(alpha);
    const float exp_a_log = std::exp(cfg.a_log);
    const float g = -exp_a_log * softplus_alpha;
    const float decay = std::exp(g);
    const float beta_gate = SigmoidStable(beta);

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
#if defined(DENSECORE_NEON_SSM)
    {
        float32x4_t qss = vdupq_n_f32(0.0f);
        float32x4_t kss = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= cfg.head_dim_k; i += 4) {
            float32x4_t qv = vld1q_f32(cfg.q_head + i);
            float32x4_t kv = vld1q_f32(cfg.k_head + i);
            qss = vmlaq_f32(qss, qv, qv);
            kss = vmlaq_f32(kss, kv, kv);
        }
        q_sum_sq = vaddvq_f32(qss);
        k_sum_sq = vaddvq_f32(kss);
        for (; i < cfg.head_dim_k; ++i) {
            q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
            k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
        }
    }
#else
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }
#endif
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim_k));
    const float q_inv_norm = q_scale / std::sqrt(q_sum_sq + cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + cfg.norm_eps);
#if defined(DENSECORE_NEON_SSM)
    {
        const float32x4_t vqn = vdupq_n_f32(q_inv_norm);
        const float32x4_t vkn = vdupq_n_f32(k_inv_norm);
        int i = 0;
        for (; i + 4 <= cfg.head_dim_k; i += 4) {
            vst1q_f32(q_norm.data() + i, vmulq_f32(vld1q_f32(cfg.q_head + i), vqn));
            vst1q_f32(k_norm.data() + i, vmulq_f32(vld1q_f32(cfg.k_head + i), vkn));
        }
        for (; i < cfg.head_dim_k; ++i) {
            q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
            k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
        }
    }
#else
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
        k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
    }
#endif
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

    const size_t state_elems = static_cast<size_t>(cfg.head_dim_k) * static_cast<size_t>(cfg.head_dim_v);

    // State decay: state_kv *= decay
#if defined(DENSECORE_NEON_SSM)
    {
        const float32x4_t v_decay = vdupq_n_f32(decay);
        size_t idx = 0;
        for (; idx + 4 <= state_elems; idx += 4) {
            vst1q_f32(state_kv + idx, vmulq_f32(vld1q_f32(state_kv + idx), v_decay));
        }
        for (; idx < state_elems; ++idx) {
            state_kv[idx] *= decay;
        }
    }
#else
    for (size_t idx = 0; idx < state_elems; ++idx) {
        state_kv[idx] *= decay;
    }
#endif

    // kv_mem[v] = sum_k(state_kv[k,v] * k_norm[k])
    // delta[v] = (v_head[v] - kv_mem[v]) * beta_gate
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float kv_mem = 0.0f;
#if defined(DENSECORE_NEON_SSM)
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= cfg.head_dim_k; k += 4) {
                // Load 4 k_norm values
                float32x4_t kn = vld1q_f32(k_norm.data() + k);
                // Load 4 state values (column-major: state_kv[k * head_dim_v + v])
                float s0 = state_kv[static_cast<size_t>(k + 0) * cfg.head_dim_v + v];
                float s1 = state_kv[static_cast<size_t>(k + 1) * cfg.head_dim_v + v];
                float s2 = state_kv[static_cast<size_t>(k + 2) * cfg.head_dim_v + v];
                float s3 = state_kv[static_cast<size_t>(k + 3) * cfg.head_dim_v + v];
                float32x4_t sv = {s0, s1, s2, s3};
                acc = vmlaq_f32(acc, sv, kn);
            }
            kv_mem = vaddvq_f32(acc);
            for (; k < cfg.head_dim_k; ++k) {
                kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
            }
        }
#else
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
        }
#endif
        if (debug && debug->kv_mem) {
            debug->kv_mem[v] = kv_mem;
        }
        delta[static_cast<size_t>(v)] = (cfg.v_head[v] - kv_mem) * beta_gate;
    }
    if (debug && debug->delta) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->delta[v] = delta[static_cast<size_t>(v)];
        }
    }

    // State update: state_kv[k, v] += k_norm[k] * delta[v]
    for (int k = 0; k < cfg.head_dim_k; ++k) {
        const float k_val = k_norm[static_cast<size_t>(k)];
        float* state_row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
#if defined(DENSECORE_NEON_SSM)
        {
            const float32x4_t v_kval = vdupq_n_f32(k_val);
            int v = 0;
            for (; v + 4 <= cfg.head_dim_v; v += 4) {
                float32x4_t sr = vld1q_f32(state_row + v);
                float32x4_t dl = vld1q_f32(delta.data() + v);
                vst1q_f32(state_row + v, vmlaq_f32(sr, v_kval, dl));
            }
            for (; v < cfg.head_dim_v; ++v) {
                state_row[v] += k_val * delta[static_cast<size_t>(v)];
            }
        }
#else
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            state_row[v] += k_val * delta[static_cast<size_t>(v)];
        }
#endif
    }

    // Output: y_head[v] = sum_k(state_kv[k,v] * q_norm[k])
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float sum = 0.0f;
#if defined(DENSECORE_NEON_SSM)
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= cfg.head_dim_k; k += 4) {
                float32x4_t qn = vld1q_f32(q_norm.data() + k);
                float s0 = state_kv[static_cast<size_t>(k + 0) * cfg.head_dim_v + v];
                float s1 = state_kv[static_cast<size_t>(k + 1) * cfg.head_dim_v + v];
                float s2 = state_kv[static_cast<size_t>(k + 2) * cfg.head_dim_v + v];
                float s3 = state_kv[static_cast<size_t>(k + 3) * cfg.head_dim_v + v];
                float32x4_t sv = {s0, s1, s2, s3};
                acc = vmlaq_f32(acc, sv, qn);
            }
            sum = vaddvq_f32(acc);
            for (; k < cfg.head_dim_k; ++k) {
                sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
            }
        }
#else
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
        }
#endif
        y_head[v] = sum;
    }
    if (debug && debug->y_pre_norm) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->y_pre_norm[v] = y_head[v];
        }
    }

    float sum_sq = 0.0f;
#if defined(DENSECORE_NEON_SSM)
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int v = 0;
        for (; v + 4 <= cfg.head_dim_v; v += 4) {
            float32x4_t yv = vld1q_f32(y_head + v);
            acc = vmlaq_f32(acc, yv, yv);
        }
        sum_sq = vaddvq_f32(acc);
        for (; v < cfg.head_dim_v; ++v) {
            sum_sq += y_head[v] * y_head[v];
        }
    }
#else
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        sum_sq += y_head[v] * y_head[v];
    }
#endif
    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    const float inv_rms = 1.0f / rms;
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        const float norm_w = cfg.norm_weight ? cfg.norm_weight[v] : 1.0f;
        y_head[v] = (y_head[v] * inv_rms) * norm_w * SiluStable(cfg.z_head[v]);
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
