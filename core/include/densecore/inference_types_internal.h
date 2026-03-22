/**
 * @file inference_types_internal.h
 * @brief Internal types shared between inference.cpp and graph builders
 *
 * NOTE: This header is for internal use only. It exposes implementation
 * details that should not be part of the public API.
 */

#ifndef DENSECORE_INFERENCE_TYPES_INTERNAL_H
#define DENSECORE_INFERENCE_TYPES_INTERNAL_H

#include "kv_cache.h"
#include "model_types.h"
#include "scheduler.h"

// Forward declarations
struct ggml_context;
struct BatchSpec;

namespace densecore {
class CpuBackend;
}

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
    int head_dim_kv;  // Store dynamically detected head dim (NOTE: matches inference.cpp naming)
    bool is_k;        // True if processing K, false if processing V
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
    int k;  // Number of experts to use (top-k)
    densecore::CpuBackend* backend;
    const BatchSpec* batch;
    densecore::Scheduler* scheduler;
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

// Smart matrix multiplication dispatcher
struct ggml_tensor* smart_mul_mat(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                  TransformerModel* model);

#endif  // DENSECORE_INFERENCE_TYPES_INTERNAL_H
