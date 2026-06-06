#pragma once

#include "densecore/backend/cpu_backend.h"

#include "ggml.h"

#include <vector>

struct InferenceWorkContext;

namespace densecore {

using ::InferenceWorkContext;

struct QuantizedProjectionInputCache {
    const float* source = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    ggml_type type = GGML_TYPE_COUNT;
    size_t row_bytes = 0;
    std::vector<uint8_t> bytes;
};

struct MoEExecutionTraceContext {
    int layer_idx = -1;
    int seq_id = -1;
    int token_idx = -1;
    int decode_step = -1;
    int n_past = -1;
    int expert_id = -1;
};

struct MoEProjectionRuntimeContext {
    CpuBackend* backend = nullptr;
    int numa_node = 0;
    const Tensor* original_input = nullptr;
    QuantizedProjectionInputCache* input_projection_cache = nullptr;
    QuantizedProjectionInputCache* down_projection_cache = nullptr;
    const CpuBackend::ExpertWeights* expert = nullptr;
    const MoEExecutionTraceContext* trace_ctx = nullptr;
    InferenceWorkContext* gemma4_quant_prefill_ctx = nullptr;
    bool safe_reference_mode = false;
    bool ggml_quantized_vecdot_safe = false;
    bool force_gemma4_quant_prefill_fast_path = false;
    bool gemma4_quant_prefill_batch_safe = false;
    bool enable_inner_parallel = false;
};

}  // namespace densecore
