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
    const int* token_ids = nullptr;
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
