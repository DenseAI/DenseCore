#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(DENSECORE_BUILD_SHARED)
#define DENSECORE_DECISION_API __declspec(dllexport)
#else
#define DENSECORE_DECISION_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define DENSECORE_DECISION_API __attribute__((visibility("default")))
#else
#define DENSECORE_DECISION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* CPU Laya/Julia GGUF runtime. One handle serializes inference; returned strings live
 * until Free. Load and Predict never throw across the ABI. */
typedef struct DenseCoreDecisionHandle DenseCoreDecisionHandle;
DENSECORE_DECISION_API DenseCoreDecisionHandle* DenseCoreDecisionLoad(const char* path, int threads, char* error,
                                                                      size_t error_capacity);
enum DenseCoreDecisionPrecision { DENSECORE_DECISION_FP32 = 0, DENSECORE_DECISION_Q8_0 = 1 };
/* Q8_0 quantizes encoder projections only and is an experimental quality
 * candidate. Norms, attention softmax, scorer and action heads remain FP32.
 * Existing Load always selects FP32. Invalid selectors fail before opening. */
DENSECORE_DECISION_API DenseCoreDecisionHandle* DenseCoreDecisionLoadEx(const char* path, int threads, int precision,
                                                                        char* error, size_t error_capacity);
DENSECORE_DECISION_API void DenseCoreDecisionFree(DenseCoreDecisionHandle* handle);
DENSECORE_DECISION_API const char* DenseCoreDecisionMetadata(DenseCoreDecisionHandle* handle);
DENSECORE_DECISION_API const char* DenseCoreDecisionTokenizerJSON(DenseCoreDecisionHandle* handle);
/* Returns 1 on success, 0 on error. Output capacities must be marker_count and 2.
 * Token IDs include the caller's CLS/SEP, markers index option MASK tokens.
 * Logits are uncalibrated; callers apply GGUF temperatures after inference. */
DENSECORE_DECISION_API int DenseCoreDecisionPredict(DenseCoreDecisionHandle* handle, const int32_t* tokens,
                                                    size_t token_count, int question_type, const int32_t* markers,
                                                    size_t marker_count, float* logits, float* act_logits, char* error,
                                                    size_t error_capacity);
typedef struct DenseCoreDecisionInput {
    const int32_t* tokens;
    size_t token_count;
    int question_type;
    const int32_t* markers;
    size_t marker_count;
} DenseCoreDecisionInput;
typedef struct DenseCoreDecisionOutput {
    float* logits;
    float* act_logits;
} DenseCoreDecisionOutput;
/* Independent requests, grouped internally by exact token count. Batch size is
 * 1..64. Each output has marker_count logits and 2 action logits. All arguments
 * are validated before execution and outputs are written only on full success.
 * No padding or encoder state is shared across requests. The model context
 * limit is architecture-specific; a separate 2 GiB graph workspace limit may
 * reject individually oversized contexts before token-buffer access. */
DENSECORE_DECISION_API int DenseCoreDecisionPredictBatch(DenseCoreDecisionHandle* handle,
                                                         const DenseCoreDecisionInput* inputs, size_t batch_size,
                                                         DenseCoreDecisionOutput* outputs, char* error,
                                                         size_t error_capacity);
#ifdef __cplusplus
}
#endif
