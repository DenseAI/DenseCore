#pragma once

#include "densecore/models/model_types.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
struct InferenceWorkContext;
struct Qwen36ProfileCounters;
namespace densecore::kernels {
struct Q4KRepackedGemvWeight;
}

// Graph-owned callback arguments shared by assembly and execution. Pools and
// executable callback implementations stay in their owning translation unit.
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

struct QKVUserData {
    const float* w_q;  ///< Q weight [dim_q, n_embd] (row-major)
    const float* w_k;  ///< K weight [dim_k, n_embd] (row-major)
    const float* w_v;  ///< V weight [dim_v, n_embd] (row-major)
    int n_embd;        ///< Input embedding dimension
    int dim_q;         ///< Q output dimension (n_head * head_dim)
    int dim_k;         ///< K output dimension (n_head_kv * head_dim)
    int dim_v;         ///< V output dimension (n_head_kv * head_dim)
};

struct BatchSpec;
struct PagedKVCache;
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
inline constexpr int kMaxKVUpdateGatherSlots = 256;
inline constexpr int kMaxAddRMSNormSlots = 256;
