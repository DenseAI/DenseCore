/**
 * @file inference_types_internal.h
 * @brief Internal types shared between inference.cpp and graph builders
 *
 * NOTE: This header is for internal use only. It exposes implementation
 * details that should not be part of the public API.
 */

#ifndef DENSECORE_INFERENCE_TYPES_INTERNAL_H
#define DENSECORE_INFERENCE_TYPES_INTERNAL_H

#include "densecore/backend/cpu_backend.h"
#include "densecore/memory/kv_cache.h"
#include "densecore/models/model_types.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/runtime/scheduler.h"

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct BatchSpec;
struct TransformerModel;
struct Qwen36ProfileCounters;

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif

// ============================================================================
// KV Cache User Data
// ============================================================================
// Used by KV cache callbacks during graph execution.
// Stored in thread-local pools to avoid allocation overhead.
// ============================================================================
struct KVCacheUserData {
    PagedKVCache* cache;
    int layer;
    int head_dim_kv;                   // Store dynamically detected head dim (NOTE: matches inference.cpp naming)
    bool is_k;                         // True if processing K, false if processing V
    bool read_only_shared_kv = false;  // Gemma4 shared layers reuse source KV cache without writing
    bool force_full_history = false;   // Gemma4 correctness path keeps full source history for shared-KV semantics
};

struct Gemma4SharedKVState {
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
    int source_layer = -1;
};

// Thread-local pool access (defined in inference.cpp)
KVCacheUserData* GetKVCacheUserData(int layer, bool is_k);

// ============================================================================
// MoE User Data
// ============================================================================
// Used by MoE forward callbacks during graph execution.
// ============================================================================
struct MoEUserData {
    const TransformerModel* model;
    const TransformerLayer* layer;
    int layer_idx = -1;
    int k;  // Number of experts to use (top-k)
    densecore::CpuBackend* backend;
    const densecore::CpuBackend::ExpertWeights* experts = nullptr;
    int n_experts = 0;
    bool experts_registered = false;
    bool test_force_empty_routing = false;
    Qwen36ProfileCounters* profile = nullptr;
};

// Allocate MoE user data from graph context
MoEUserData* AllocateMoEUserData(struct ggml_context* ctx);

// ============================================================================
// Callback Function Declarations
// ============================================================================
// These callbacks are used by ggml_map_custom* during graph execution.
// ============================================================================

// Multi-LoRA application callback
void cb_apply_multi_lora(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                         int ith, int nth, void* userdata);

// Fused SiLU×Mul callback (SwiGLU FFN)
void cb_silu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata);

// KV cache management callback
void cb_kv_manage(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata);

// MoE forward callback
void cb_moe_forward(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                    int nth, void* userdata);

// ============================================================================
// Qwen3.5 SSM User Data
// ============================================================================
struct SSMQwen35DeltaUserData {
    const float* alpha_weight;  // [n_heads, n_embd]
    const float* beta_weight;   // [n_heads, n_embd]
    const float* dt_bias;       // [n_heads]
    const float* a_log;         // [n_heads]
    const float* norm_weight;   // [head_dim_v] or [d_inner]
    float* ssm_state;           // canonical [n_heads][head_dim_k][head_dim_v]
    int n_embd;
    int d_inner;
    int n_heads;
    int head_dim_v;
    int head_dim_k;
    int n_groups;
    Qwen35SSMNormLayout norm_layout = Qwen35SSMNormLayout::INVALID;
    float norm_eps;
    int layer_idx = -1;
    int ssm_ordinal = -1;
    Qwen35SSMQkvProjectionProfile projection_profile = Qwen35SSMQkvProjectionProfile::QWEN35_LEGACY;
    const int* token_seq_ids = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
    const ggml_tensor* z_tensor = nullptr;
    const ggml_tensor* qkv_tensor = nullptr;
    const ggml_tensor* input_tensor = nullptr;
    const ggml_tensor* alpha_beta_tensor = nullptr;
    Qwen36ProfileCounters* profile = nullptr;
    bool fast_silu_gate = false;
};

// SSM delta recurrent callback
void cb_ssm_conv1d_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_qkv_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                  void* userdata);
void cb_ssm_qwen35_delta_z_qkv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                               int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_z_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata);
void cb_ssm_qwen35_delta(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                         const struct ggml_tensor* c, int ith, int nth, void* userdata);
void cb_ssm_qwen35_delta_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);

// Smart matrix multiplication dispatcher
struct ggml_tensor* smart_mul_mat(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                  TransformerModel* model);

#endif  // DENSECORE_INFERENCE_TYPES_INTERNAL_H
