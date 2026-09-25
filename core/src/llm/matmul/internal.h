#ifndef DENSECORE_LLM_MATMUL_INTERNAL_H
#define DENSECORE_LLM_MATMUL_INTERNAL_H

#include "densecore/models/model_types.h"
#include <ggml.h>

// Private GGML payload layout shared only by construction and execution.
// These fields are copied into op_params, never allocated per callback.
namespace densecore::llm::matmul {
struct HalMatmulOpData {
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
};

struct HalMatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalMatmulOpData data;
};

struct FP8MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    int K = 0;
    int N = 0;
    TransformerModel::FP8Format format = TransformerModel::FP8Format::E4M3FN;
};

struct FP8MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    FP8MatmulOpData data;
};

struct Int4MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    const float* scales = nullptr;
    const float* zeros = nullptr;
    int K = 0;
    int N = 0;
    int group_size = 0;
    bool is_gemma4 = false;
};

struct Int4MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    Int4MatmulOpData data;
};

}  // namespace densecore::llm::matmul

// Keep callback identities for worker node timing and existing GGML graphs.
void cb_matmul_hal_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_matmul_int4_custom(ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_matmul_fp8_custom(ggml_tensor* dst, int ith, int nth, void* userdata);

#endif
