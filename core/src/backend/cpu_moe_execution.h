#pragma once
#include "backend/cpu_backend_moe_projection.h"
#include "backend/cpu_execution_options.h"
#include "densecore/backend/cpu_backend.h"
namespace densecore {
bool RunQ5KRepackedMoEFusedSwiGLURawProjection(const CpuExecutionOptions& options, CpuBackend* backend,
                                               const void* gate_weight_ptr, const void* up_weight_ptr,
                                               const float* input_data, const uint8_t* qinput_data,
                                               size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                               int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ4KRepackedMoEProjection(const CpuExecutionOptions& options, CpuBackend* backend, const void* weight_ptr,
                                 const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                                 int64_t cols, int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ4KRepackedMoEFusedSwiGLUProjection(const CpuExecutionOptions& options, CpuBackend* backend,
                                            const void* gate_weight_ptr, const void* up_weight_ptr,
                                            const float* input_data, const uint8_t* qinput_data,
                                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                            int64_t input_cols, int numa_node, bool allow_parallel);
bool RunMoEQ4KRawBatchedProjection(const CpuExecutionOptions& options, CpuBackend* backend, const void* weight_ptr,
                                   const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                   int64_t N, int64_t K, int numa_node, bool allow_parallel);
namespace internal {
bool RunQ4KRawBatchedQuantizedProjection(const CpuExecutionOptions& options, CpuBackend* backend,
                                         const void* weight_data, const uint8_t* quantized_input,
                                         size_t quantized_input_row_bytes, float* output_data, int64_t rows,
                                         int64_t output_cols, int64_t input_cols, int numa_node, bool allow_parallel);
}
bool RunMoEQ4KRawBatchedWeightedScatterProjection(const CpuExecutionOptions& options, CpuBackend* backend,
                                                  const void* weight_ptr, const uint8_t* qinput_data,
                                                  size_t qinput_row_bytes, const int* token_indices,
                                                  const float* token_weights, float* output_data,
                                                  int64_t output_row_stride, int64_t M, int64_t N, int64_t K,
                                                  int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedProjection(const CpuExecutionOptions& options, CpuBackend* backend, int ggml_type_id,
                                      const void* weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
                                      float* out_data, int64_t M, int64_t N, int64_t K, int numa_node,
                                      bool allow_parallel);
bool RunMoEKQuantRawBatchedFusedSwiGLU(const CpuExecutionOptions& options, CpuBackend* backend, int ggml_type_id,
                                       const void* gate_weight_ptr, const void* up_weight_ptr,
                                       const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                       int64_t N, int64_t K, int numa_node, bool allow_parallel);
bool RunMoEQ4KRawBatchedFusedSwiGLUToQ8(const CpuExecutionOptions& options, CpuBackend* backend,
                                        const void* gate_weight_ptr, const void* up_weight_ptr,
                                        const uint8_t* qinput_data, size_t qinput_row_bytes, uint8_t* qoutput_data,
                                        size_t qoutput_row_bytes, int64_t M, int64_t N, int64_t K, int numa_node,
                                        bool allow_parallel);
namespace testing {
bool RunMoEQ4KQ8KBatchedRowPairParityForTest(const CpuExecutionOptions& options, const void* gate_weight_row,
                                             const void* up_weight_row, const uint8_t* qinput_data,
                                             size_t qinput_row_bytes, int M, int K, bool* specialized_pair_used);
}  // namespace testing
namespace testing {
bool RunGgmlQuantizedProjectionForTest(const CpuExecutionOptions& options, CpuBackend* backend, const void* weight_ptr,
                                       int ggml_type_id, const Tensor& input, Tensor* output, int64_t N, int64_t K);
}  // namespace testing
namespace testing {
bool RunGgmlQuantizedFusedSwiGLUProjectionForTest(const CpuExecutionOptions& options, CpuBackend* backend,
                                                  const void* gate_weight_ptr, int gate_ggml_type_id,
                                                  const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                                  Tensor* output, int64_t N, int64_t K, bool use_gelu_activation,
                                                  QuantizedProjectionInputCache* input_cache);
}  // namespace testing
namespace testing {
bool RunGgmlQuantizedFusedGEGLUProjectionForTest(const CpuExecutionOptions& options, CpuBackend* backend,
                                                 const void* gate_weight_ptr, int gate_ggml_type_id,
                                                 const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                                 Tensor* output, int64_t N, int64_t K);
}  // namespace testing
}  // namespace densecore

namespace densecore {
bool RunQ6KRepackedMoEProjectionCached(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                       size_t qinput_row_bytes, float* out_data, int64_t rows, int64_t cols,
                                       int64_t input_cols, int numa_node, bool allow_parallel,
                                       std::shared_ptr<void>* packed_cache);
}
