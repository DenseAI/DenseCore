#ifndef DENSECORE_LLM_MATMUL_GRAPH_OPS_H
#define DENSECORE_LLM_MATMUL_GRAPH_OPS_H

#include "densecore/models/model_types.h"
#include <ggml.h>

ggml_tensor* ggml_mul_mat_hal(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                              densecore::DeviceType preferred_device);
ggml_tensor* ggml_mul_mat_fp8(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                              const TransformerModel::FP8WeightBinding& binding);

ggml_tensor* ggml_mul_mat_int4(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                               const TransformerModel::Int4WeightBinding& binding, bool is_gemma4 = false);
struct BatchSpec;
int ResolveTaskCount(const BatchSpec* batch, int work_items);

struct GemvUserData;
struct GemvBatchedUserData;
ggml_tensor* ggml_mul_mat_gemv(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input, GemvUserData* userdata);
ggml_tensor* ggml_mul_mat_gemv_batched(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                                       GemvBatchedUserData* userdata, bool use_map3_dependencies = false);
bool ShouldUseQwenHybridSSMQ8RepackedBatched(bool qwen_hybrid_ssm_q8_prefill, int tokens);
bool ShouldUseQwenHybridSSMQ8DirectBatched(bool qwen_hybrid_ssm_q8_prefill, int tokens);
#endif
