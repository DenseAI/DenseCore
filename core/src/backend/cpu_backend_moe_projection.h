#pragma once

#include "densecore/backend/cpu_backend.h"

#include "ggml.h"
#include "backend/cpu_execution_options.h"

#include <vector>



namespace densecore {



struct QuantizedProjectionInputCache {
    const float* source = nullptr;
    const uint8_t* external_bytes = nullptr;
    size_t external_size = 0;
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
    const CpuExecutionTelemetry* gemma4_quant_prefill_ctx = nullptr;
    bool safe_reference_mode = false;
    bool ggml_quantized_vecdot_safe = false;
    bool gemma4_quant_prefill_batch_safe = false;
    bool record_gemma4_quant_prefill_batch = false;
    bool prefer_q4k_repacked_prefill = false;
    bool allow_q4k_repacked_decode_fused_swiglu = true;
    bool enable_inner_parallel = false;
};

}  // namespace densecore
