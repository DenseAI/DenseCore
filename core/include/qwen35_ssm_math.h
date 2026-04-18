#ifndef DENSECORE_QWEN35_SSM_MATH_H
#define DENSECORE_QWEN35_SSM_MATH_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct Qwen35SSMHeadStepConfig {
    const float* input_t = nullptr;
    const float* q_head = nullptr;
    const float* k_head = nullptr;
    const float* v_head = nullptr;
    const float* z_head = nullptr;
    const float* alpha_row = nullptr;
    const float* beta_row = nullptr;
    const float* norm_weight = nullptr;
    int n_embd = 0;
    int head_dim_k = 0;
    int head_dim_v = 0;
    float dt_bias = 0.0f;
    float a_log = 0.0f;
    float norm_eps = 1e-6f;
};

struct Qwen35SSMHeadStepStats {
    float alpha = 0.0f;
    float beta = 0.0f;
    float softplus_alpha = 0.0f;
    float exp_a_log = 0.0f;
    float g = 0.0f;
    float decay = 0.0f;
    float beta_gate = 0.0f;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    float rms = 0.0f;
};

struct Qwen35SSMHeadStepDebugBuffers {
    float* q_norm = nullptr;      // [head_dim_k]
    float* k_norm = nullptr;      // [head_dim_k]
    float* kv_mem = nullptr;      // [head_dim_v]
    float* delta = nullptr;       // [head_dim_v]
    float* y_pre_norm = nullptr;  // [head_dim_v]
};

enum class Qwen35SSMNormLayout {
    INVALID = 0,
    SHARED_HEAD_DIM,
    FLATTENED_D_INNER,
};

bool Qwen35CanonicalizeHeadByEmbd(const float* raw, const int64_t ne[4], int n_embd, int n_heads,
                                  std::vector<float>* out);
bool Qwen35CanonicalizeFusedBA(const float* raw, const int64_t ne[4], int n_embd, int n_v_heads, int n_groups,
                               std::vector<float>* beta_out, std::vector<float>* alpha_out);
bool Qwen35CanonicalizePerHeadVector(const float* raw, const int64_t ne[4], int n_heads, std::vector<float>* out);
Qwen35SSMNormLayout Qwen35CanonicalizeNorm(const float* raw, const int64_t ne[4], int head_dim_v, int d_inner,
                                           std::vector<float>* out);

// Runs one Qwen3.5 gated-delta recurrent step for a single head.
//
// `state_kv` uses the canonical reference layout [K, V], flattened as
// `state[k * head_dim_v + v]`. The function updates state in place and writes
// the gated RMSNorm output to `y_head`.
bool Qwen35RunGatedDeltaHeadStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head,
                                 Qwen35SSMHeadStepStats* stats = nullptr,
                                 Qwen35SSMHeadStepDebugBuffers* debug = nullptr);

#endif  // DENSECORE_QWEN35_SSM_MATH_H
