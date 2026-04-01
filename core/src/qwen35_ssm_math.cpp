#include "qwen35_ssm_math.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

    std::vector<float> q_norm(static_cast<size_t>(cfg.head_dim_k));
    std::vector<float> k_norm(static_cast<size_t>(cfg.head_dim_k));
    std::vector<float> delta(static_cast<size_t>(cfg.head_dim_v));

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
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim_k));
    const float q_inv_norm = q_scale / std::sqrt(q_sum_sq + cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + cfg.norm_eps);
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
        k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
    }
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
    for (size_t idx = 0; idx < state_elems; ++idx) {
        state_kv[idx] *= decay;
    }

    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float kv_mem = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
        }
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

    for (int k = 0; k < cfg.head_dim_k; ++k) {
        const float k_val = k_norm[static_cast<size_t>(k)];
        float* state_row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            state_row[v] += k_val * delta[static_cast<size_t>(v)];
        }
    }

    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float sum = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
        }
        y_head[v] = sum;
    }
    if (debug && debug->y_pre_norm) {
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            debug->y_pre_norm[v] = y_head[v];
        }
    }

    float sum_sq = 0.0f;
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        sum_sq += y_head[v] * y_head[v];
    }
    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        const float norm_w = cfg.norm_weight ? cfg.norm_weight[v] : 1.0f;
        y_head[v] = (y_head[v] / rms) * norm_w * SiluStable(cfg.z_head[v]);
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
