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
    // GGUFs in this family may expose either raw A_log or converter-materialized
    // ssm_a = -exp(A_log). The step code resolves the stored form before use.
    float a_log = 0.0f;
    float norm_eps = 1e-6f;
    bool a_log_prescaled = false;
    bool norm_weight_uses_unit_offset = false;
    bool has_precomputed_alpha_beta = false;
    float precomputed_alpha = 0.0f;
    float precomputed_beta = 0.0f;
    bool has_precomputed_qk_norm = false;
    float precomputed_q_inv_norm = 0.0f;
    float precomputed_k_inv_norm = 0.0f;
    bool use_fast_silu = false;
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
    double alpha_beta_dot_ms = 0.0;
    double norm_ms = 0.0;
    double decay_state_ms = 0.0;
    double kv_mem_ms = 0.0;
    double state_update_ms = 0.0;
    double output_accum_ms = 0.0;
    double rms_gate_ms = 0.0;
};

struct Qwen35SSMHeadStepDebugBuffers {
    float* q_norm = nullptr;      // [head_dim_k]
    float* k_norm = nullptr;      // [head_dim_k]
    float* kv_mem = nullptr;      // [head_dim_v]
    float* delta = nullptr;       // [head_dim_v]
    float* y_pre_norm = nullptr;  // [head_dim_v]
};

struct Qwen35SSMHeadStepTrace {
    uint64_t state_in_hash = 0;
    uint64_t state_out_hash = 0;
    uint64_t y_hash = 0;
};

enum class Qwen35SSMQkvRawLayout {
    QKV = 0,
};

enum class Qwen35SSMQkvActivationPlacement {
    NONE = 0,
    FULL_ROW_SILU,
};

enum class Qwen35SSMQkvProjectionProfile {
    QWEN35_LEGACY = 0,
    QWEN35_OFFICIAL,
    QWEN36_OFFICIAL,
};

struct Qwen35SSMQkvProjectionContract {
    Qwen35SSMQkvProjectionProfile profile = Qwen35SSMQkvProjectionProfile::QWEN35_LEGACY;
    Qwen35SSMQkvRawLayout raw_layout = Qwen35SSMQkvRawLayout::QKV;
    Qwen35SSMQkvActivationPlacement activation = Qwen35SSMQkvActivationPlacement::FULL_ROW_SILU;
    bool share_qk_by_group = true;
    bool dense_value_heads = true;
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
void Qwen35ReorderVHeadsGroupedToTiled(std::vector<float>* values, int row_width, int num_k_heads, int num_v_heads);
Qwen35SSMQkvProjectionContract ResolveQwen35SSMQkvProjectionContract(Qwen35SSMQkvProjectionProfile profile);
bool Qwen35CanonicalizeProjectedQkvRow(const float* raw_row, int num_k_heads, int num_v_heads, int head_dim_k,
                                       int head_dim_v, const Qwen35SSMQkvProjectionContract& contract,
                                       std::vector<float>* canonical_qkv);

size_t Qwen35SSMHeadStateElements(int head_dim_k, int head_dim_v);

// Runs one Qwen3.5 gated-delta recurrent step for a single head.
//
// `state_kv` uses the canonical reference layout [K, V], flattened as
// `state[k * head_dim_v + v]`. The function updates state in place and writes
// the gated RMSNorm output to `y_head`.
bool Qwen35RunGatedDeltaHeadStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head,
                                 Qwen35SSMHeadStepStats* stats = nullptr,
                                 Qwen35SSMHeadStepDebugBuffers* debug = nullptr);

// Default serving fast path for the same math as Qwen35RunGatedDeltaHeadStep.
// It avoids materializing q_norm/k_norm scratch when diagnostics are disabled.
bool Qwen35RunGatedDeltaHeadStepFastDefault(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head);

#endif  // DENSECORE_QWEN35_SSM_MATH_H
