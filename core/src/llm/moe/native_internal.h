#pragma once
#include "densecore/runtime/inference.h"
#include "llm/moe/native.h"

// Gemma builders share allocation/NUMA operations, not the Qwen workspace layout.
namespace densecore::llm::graph::detail {
struct Qwen35SharedQ8RowsUserData;
Qwen35SharedQ8RowsUserData* AllocateQwen35SharedQ8RowsUserData(ggml_context* ctx, const ggml_tensor* src,
                                                               int64_t max_assignments = 0,
                                                               int64_t direct_output_rows = 0);
void SetNativeMoECallbackRequestedTaskCount(Qwen35SharedQ8RowsUserData* ud, int requested_task_count);
void SetNativeMoENumaContext(Qwen35SharedQ8RowsUserData* ud, densecore::CpuBackend* backend,
                             const TransformerLayer* layer, ModelVariant model_variant);
bool NativeMoEHasVerifiedNumaPlacement(densecore::CpuBackend* backend, const TransformerLayer* layer);
void RecordNativeMoENumaDispatch(int node);
void cb_gemma4_native_moe_down_weighted_sum(struct ggml_tensor* dst, int ith, int nth, void* userdata);
int ResolveNativeMoEGraphCallbackTaskCount(const TransformerModel* model, const BatchSpec* batch,
                                           InferenceExecutionPhase phase, int64_t n_tokens, int top_k);
ggml_tensor* UseCpuRepackAliasIfAvailable(TransformerModel* model, ggml_tensor* tensor);
int64_t NativeMoEFastPathMaxDirectTokens(const TransformerModel* model);
bool Qwen35NativeMoEReferenceDotQXK(ggml_type weight_type, const void* weight_row, const uint8_t* qrow, int64_t cols,
                                    float* out_value);
ggml_tensor* BuildMoETopKWeightsFromLogits(struct ggml_context* ctx, ggml_tensor* logits, ggml_tensor* selected,
                                           const char* name);

}  // namespace densecore::llm::graph::detail
