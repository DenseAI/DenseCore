#pragma once

#include "kernels/q4k_repacked_gemv.h"

#include <cstdint>
#include <memory>

namespace densecore {

class CpuBackend;

namespace internal {

bool RunQ4KRawBatchedQuantizedProjection(CpuBackend* backend, const void* weight_data, const uint8_t* quantized_input,
                                         size_t quantized_input_row_bytes, float* output_data, int64_t rows,
                                         int64_t output_cols, int64_t input_cols, int numa_node, bool allow_parallel);

bool RunQ4KRawBatchedDenseGemm(CpuBackend* backend, const void* weight_data, const float* input_data,
                               float* output_data, int64_t rows, int64_t output_cols, int64_t input_cols, int numa_node,
                               bool allow_parallel);

bool RunQ4KRepackedDenseGemm(CpuBackend* backend, const std::shared_ptr<kernels::Q4KRepackedGemvWeight>& packed,
                             const float* input_data, float* output_data, int64_t rows, int64_t output_cols,
                             int64_t input_cols, int numa_node, bool allow_parallel);

}  // namespace internal
}  // namespace densecore
