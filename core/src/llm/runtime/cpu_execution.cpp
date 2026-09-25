#include "llm/runtime/cpu_execution.h"
#include "densecore/runtime/inference.h"

namespace densecore::llm::runtime {
CpuExecutionTelemetry MakeCpuExecutionTelemetry(InferenceWorkContext* context) {
    CpuExecutionTelemetry sink;
    sink.data = context;
    sink.MoEExpertMatmulWeightType = [](void* data, ggml_type weight_type) {
        ::RecordMoEExpertMatmulWeightType(static_cast<InferenceWorkContext*>(data), weight_type);
    };
    sink.MoEQ4KRepackedDecision = [](void* data, bool candidate, bool used, const char* reject_reason) {
        ::RecordMoEQ4KRepackedDecision(static_cast<InferenceWorkContext*>(data), candidate, used, reject_reason);
    };
    sink.MoEQ5KRepackedDecision = [](void* data, bool candidate, bool used, const char* reject_reason) {
        ::RecordMoEQ5KRepackedDecision(static_cast<InferenceWorkContext*>(data), candidate, used, reject_reason);
    };
    sink.MoEKQuantRawBatchedUse = [](void* data, ggml_type weight_type, uint64_t wall_ns, bool qwen_native_w2_q5k) {
        ::RecordMoEKQuantRawBatchedUse(static_cast<InferenceWorkContext*>(data), weight_type, wall_ns,
                                       qwen_native_w2_q5k);
    };
    sink.QwenNativeMoEQ4GateUpRowPairUse = [](void* data, uint64_t wall_ns) {
        ::RecordQwenNativeMoEQ4GateUpRowPairUse(static_cast<InferenceWorkContext*>(data), wall_ns);
    };
    sink.Gemma4MoEPrefillQuantBatchDecision = [](void* data, bool candidate, bool used, const char* reject_reason,
                                                 bool gate_up_used, bool down_used) {
        ::RecordGemma4MoEPrefillQuantBatchDecision(static_cast<InferenceWorkContext*>(data), candidate, used,
                                                   reject_reason, gate_up_used, down_used);
    };
    sink.Gemma4NativeFusedGateUpUsed = [](void* data) {
        ::RecordGemma4NativeFusedGateUpUsed(static_cast<InferenceWorkContext*>(data));
    };
    sink.MatmulDispatchCensus = [](void* data, CpuExecutionPhase phase, const char* dispatch_path,
                                   ggml_type weight_type, int64_t m, int64_t n, int64_t k, uint64_t wall_ns) {
        ::RecordMatmulDispatchCensus(static_cast<InferenceWorkContext*>(data),
                                     static_cast<InferenceExecutionPhase>(phase), dispatch_path, weight_type, m, n, k,
                                     wall_ns);
    };
    sink.MoESmallDecodeParallelDecision = [](void* data, bool candidate, bool used, const char* reject_reason,
                                             int selected_expert_count, int top_k, int task_count) {
        ::RecordMoESmallDecodeParallelDecision(static_cast<InferenceWorkContext*>(data), candidate, used, reject_reason,
                                               selected_expert_count, top_k, task_count);
    };
    return sink;
}
CpuExecutionOptions MakeCpuExecutionOptions(InferenceWorkContext* context, const CpuExecutionTelemetry* telemetry) {
    auto options = ResolveCpuExecutionOptions(context);
    options.telemetry = context ? telemetry : nullptr;
    return options;
}
}  // namespace densecore::llm::runtime

#include "backend/cpu_backend_moe_projection.h"
#include "backend/cpu_backend_q4k_dense.h"
#include "backend/cpu_moe_execution.h"

namespace densecore {
bool RunQ5KRepackedMoEFusedSwiGLURawProjection(CpuBackend* backend, const void* gate_weight_ptr,
                                               const void* up_weight_ptr, const float* input_data,
                                               const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                               int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                               bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunQ5KRepackedMoEFusedSwiGLURawProjection(options, backend, gate_weight_ptr, up_weight_ptr, input_data,
                                                     qinput_data, qinput_row_bytes, output_data, rows, cols, input_cols,
                                                     numa_node, allow_parallel);
}
bool RunQ4KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunQ4KRepackedMoEProjection(options, backend, weight_ptr, qinput_data, qinput_row_bytes, output_data, rows,
                                       cols, input_cols, numa_node, allow_parallel);
}
bool RunQ4KRepackedMoEFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                            const float* input_data, const uint8_t* qinput_data,
                                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                            int64_t input_cols, int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunQ4KRepackedMoEFusedSwiGLUProjection(options, backend, gate_weight_ptr, up_weight_ptr, input_data,
                                                  qinput_data, qinput_row_bytes, output_data, rows, cols, input_cols,
                                                  numa_node, allow_parallel);
}
bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEQ4KRawBatchedProjection(options, backend, weight_ptr, qinput_data, qinput_row_bytes, out_data, M, N, K,
                                         numa_node, allow_parallel);
}
bool internal::RunQ4KRawBatchedQuantizedProjection(CpuBackend* backend, const void* weight_data,
                                                   const uint8_t* quantized_input, size_t quantized_input_row_bytes,
                                                   float* output_data, int64_t rows, int64_t output_cols,
                                                   int64_t input_cols, int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return internal::RunQ4KRawBatchedQuantizedProjection(options, backend, weight_data, quantized_input,
                                                         quantized_input_row_bytes, output_data, rows, output_cols,
                                                         input_cols, numa_node, allow_parallel);
}
bool RunMoEQ4KRawBatchedWeightedScatterProjection(CpuBackend* backend, const void* weight_ptr,
                                                  const uint8_t* qinput_data, size_t qinput_row_bytes,
                                                  const int* token_indices, const float* token_weights,
                                                  float* output_data, int64_t output_row_stride, int64_t M, int64_t N,
                                                  int64_t K, int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEQ4KRawBatchedWeightedScatterProjection(options, backend, weight_ptr, qinput_data, qinput_row_bytes,
                                                        token_indices, token_weights, output_data, output_row_stride, M,
                                                        N, K, numa_node, allow_parallel);
}
bool RunMoEKQuantRawBatchedProjection(CpuBackend* backend, int ggml_type_id, const void* weight_ptr,
                                      const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                      int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEKQuantRawBatchedProjection(options, backend, ggml_type_id, weight_ptr, qinput_data, qinput_row_bytes,
                                            out_data, M, N, K, numa_node, allow_parallel);
}
bool RunMoEKQuantRawBatchedFusedSwiGLU(CpuBackend* backend, int ggml_type_id, const void* gate_weight_ptr,
                                       const void* up_weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
                                       float* out_data, int64_t M, int64_t N, int64_t K, int numa_node,
                                       bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEKQuantRawBatchedFusedSwiGLU(options, backend, ggml_type_id, gate_weight_ptr, up_weight_ptr,
                                             qinput_data, qinput_row_bytes, out_data, M, N, K, numa_node,
                                             allow_parallel);
}
bool RunMoEQ4KRawBatchedFusedSwiGLUToQ8(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                        const uint8_t* qinput_data, size_t qinput_row_bytes, uint8_t* qoutput_data,
                                        size_t qoutput_row_bytes, int64_t M, int64_t N, int64_t K, int numa_node,
                                        bool allow_parallel) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEQ4KRawBatchedFusedSwiGLUToQ8(options, backend, gate_weight_ptr, up_weight_ptr, qinput_data,
                                              qinput_row_bytes, qoutput_data, qoutput_row_bytes, M, N, K, numa_node,
                                              allow_parallel);
}
namespace testing {
bool RunMoEQ4KQ8KBatchedRowPairParityForTest(const void* gate_weight_row, const void* up_weight_row,
                                             const uint8_t* qinput_data, size_t qinput_row_bytes, int M, int K,
                                             bool* specialized_pair_used) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunMoEQ4KQ8KBatchedRowPairParityForTest(options, gate_weight_row, up_weight_row, qinput_data,
                                                   qinput_row_bytes, M, K, specialized_pair_used);
}
}  // namespace testing
void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const ExpertWeights& expert, const Tensor& w1,
                                   const Tensor& w2, const Tensor& w3, Tensor* output) {

    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}
void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const ExpertWeights& expert, const Tensor& w1, const Tensor& w2, const Tensor& w3,
                                   Tensor* output) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    DispatchExpertFFN(options, layer_key, expert_id, input, expert, w1, w2, w3, output);
}
void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2,
                                   const Tensor& w3, Tensor* output) {

    ExpertWeights expert{};
    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}
void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const Tensor& w1, const Tensor& w2, const Tensor& w3, Tensor* output) {

    ExpertWeights expert{};
    DispatchExpertFFN(layer_key, expert_id, input, expert, w1, w2, w3, output);
}
void CpuBackend::ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {

    ForwardMoE(nullptr, nullptr, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}
void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {

    ForwardMoE(nullptr, layer_key, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()),
               output);
}
void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {

    ForwardMoE(nullptr, layer_key, layer_idx, batch, input, routing, experts.data(), static_cast<int>(experts.size()),
               output);
}
void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output) {

    ForwardMoE(nullptr, layer_key, -1, nullptr, input, routing, experts, num_experts, output);
}
void CpuBackend::ForwardMoE(const TransformerModel* model, const TransformerLayer* layer_key, int layer_idx,
                            const BatchSpec* batch, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output, MoEForwardProfile* profile) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    static const std::vector<int> empty;
    const CpuBatchTraceView trace{batch ? batch->seq_id : empty, batch ? batch->n_past : empty};
    ForwardMoE(options, model, layer_key, layer_idx, batch ? &trace : nullptr, input, routing, experts, num_experts,
               output, profile);
}
void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing, const ExpertWeights* experts,
                            int num_experts, Tensor* output) {

    ForwardMoE(nullptr, layer_key, layer_idx, batch, input, routing, experts, num_experts, output);
}
namespace testing {
bool RunGgmlQuantizedProjectionForTest(CpuBackend* backend, const void* weight_ptr, int ggml_type_id,
                                       const Tensor& input, Tensor* output, int64_t N, int64_t K) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunGgmlQuantizedProjectionForTest(options, backend, weight_ptr, ggml_type_id, input, output, N, K);
}
}  // namespace testing
namespace testing {
bool RunGgmlQuantizedFusedSwiGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                  int gate_ggml_type_id, const void* up_weight_ptr, int up_ggml_type_id,
                                                  const Tensor& input, Tensor* output, int64_t N, int64_t K,
                                                  bool use_gelu_activation,
                                                  QuantizedProjectionInputCache* input_cache) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunGgmlQuantizedFusedSwiGLUProjectionForTest(options, backend, gate_weight_ptr, gate_ggml_type_id,
                                                        up_weight_ptr, up_ggml_type_id, input, output, N, K,
                                                        use_gelu_activation, input_cache);
}
}  // namespace testing
namespace testing {
bool RunGgmlQuantizedFusedGEGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                 int gate_ggml_type_id, const void* up_weight_ptr, int up_ggml_type_id,
                                                 const Tensor& input, Tensor* output, int64_t N, int64_t K) {
    auto sink = llm::runtime::MakeCpuExecutionTelemetry(GetCurrentWorkContext());
    const auto options = llm::runtime::MakeCpuExecutionOptions(GetCurrentWorkContext(), &sink);
    return RunGgmlQuantizedFusedGEGLUProjectionForTest(options, backend, gate_weight_ptr, gate_ggml_type_id,
                                                       up_weight_ptr, up_ggml_type_id, input, output, N, K);
}
}  // namespace testing
}  // namespace densecore

namespace densecore {
bool CpuBackend::MatMulGgmlQuantizedTransB(const Tensor& input, const GgmlQuantizedMatrixView& weight, Tensor* output,
                                           int numa_node) {
    return MatMulGgmlQuantizedTransBWithPath(input, weight, output, numa_node) != GgmlQuantizedMatMulPath::Rejected;
}
CpuBackend::GgmlQuantizedMatMulPath CpuBackend::MatMulGgmlQuantizedTransBWithPath(const Tensor& input,
                                                                                  const GgmlQuantizedMatrixView& weight,
                                                                                  Tensor* output, int numa_node) {
    return MatMulGgmlQuantizedTransBWithPath(llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), input,
                                             weight, output, numa_node);
}
}  // namespace densecore
