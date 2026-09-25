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
struct SSMConv1DUserData {
    float* conv_state;
    const float* weight;
    int channels;
    int kernel_size;
    bool apply_silu = false;
    int layer_idx = -1;
    int ssm_ordinal = -1;
    const int* token_seq_ids = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
    Qwen36ProfileCounters* profile = nullptr;
};

struct LFM2ShortConvUserData {
    const float* conv_weight = nullptr;  // [channels * kernel], layout [channel * kernel + tap]
    int channels = 0;
    int kernel = 0;
    int conv_ordinal = -1;  // index among short-conv layers (state slot)
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const int* token_positions = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
    Qwen36ProfileCounters* profile = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight> decode_in_proj_q4k_packed;
    const void* decode_in_proj_weight_data = nullptr;
    int decode_in_proj_rows = 0;
    int decode_in_proj_cols = 0;
    std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight> decode_out_proj_q4k_packed;
    const void* decode_out_proj_weight_data = nullptr;
    int decode_out_proj_rows = 0;
    int decode_out_proj_cols = 0;
    std::vector<float> decode_bcx;
    std::vector<uint8_t> decode_input_q8;
    std::vector<float> decode_y;
    std::vector<uint8_t> decode_y_q8;
    std::atomic<uint64_t> decode_in_proj_ready_stamp{0};
    std::atomic<int> decode_in_proj_done{0};
    std::atomic<uint64_t> decode_y_stamp{0};
    std::atomic<uint64_t> decode_out_proj_stamp{0};
};

struct GLMDSAPackUserData {
    int n_heads;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int v_head_dim;
};
