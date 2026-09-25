#ifndef DENSECORE_LLM_MATMUL_DIAGNOSTICS_H
#define DENSECORE_LLM_MATMUL_DIAGNOSTICS_H
#include <cstddef>
#include <cstdint>
#include <ggml.h>
bool IsDebugMatmulPathLoggingEnabled();
void LogMatmulValidationOnce(const char* path, bool ok, float max_abs_diff);
bool IsDebugMatmulDispatchEnabled();
uint64_t GetHybridSSMDispatchCounter(size_t index);
void LogHybridSSMQkvDispatch(const char* weight_name, ggml_type weight_type, int M, int N, int K,
                             const char* chosen_path, bool used_batched_quant_nrc, bool used_native_q4k_vecdot,
                             bool used_direct_int4_fastpath);
void LogMatmulPathOnce(const char* path);
const char* MatmulWeightTypeLabel(ggml_type wtype, bool is_packed_int4, bool is_packed_fp8);
const char* DetectedISATier();
void LogMatmulDispatch(const char* weight_name, const char* weight_type_label, int M, int N, int K,
                       const char* path_label, const char* fallback_reason = nullptr);
bool IsDebugGemvSelectionEnabled();
#endif
