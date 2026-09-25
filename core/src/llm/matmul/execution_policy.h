#pragma once
#include "densecore/runtime/inference.h"
#include "densecore/simd/simd_ops.h"
#include <ggml-cpu.h>

uint64_t ComputeGemvBatchedQuantStamp(const BatchSpec* batch, int M, int slot_id, const void* src_data_ptr,
                                      const void* weight_data_ptr);
bool ShouldUseArmNativeQ4KVecDotValidated(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                          const char* weight_name, const void* sample_row_ptr,
                                          const void* sample_quant_input, const float* sample_input_f32, int N);

bool ShouldUseArmQwen36LargeQ8Prefill(bool qwen36_ssm_prefill, int tokens, int reduction, int rows);
bool ShouldUseBatchedDecodeRowMajor(ModelVariant variant, InferenceExecutionPhase phase, int tokens,
                                    int explicit_override);
int ResolveQuantBatchedTileCols(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k);
bool IsQ4KTrueBatchedKernelEnabled();
bool ResolveQ4KTrueBatchedKernelEnabledPolicy(densecore::simd::SimdLevel level, bool compiled_with_sve);
densecore::simd::SimdLevel GetRuntimeSimdLevel();

struct GemvUserData;
enum class GemvCustomTaskCapReason : int;
bool IsSemanticDecodeProjectionGemv(const GemvUserData* userdata, int N, int K);
int ResolveGemvCustomOpTaskCount(int N, int K, int requested_threads, int physical_cores,
                                 bool semantic_decode_projection, GemvCustomTaskCapReason* cap_reason);
