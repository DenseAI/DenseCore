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
/* CPU Laya GGUF runtime. One handle serializes inference; returned strings live
 * until Free. Load and Predict never throw across the ABI. */
typedef struct DenseCoreDecisionHandle DenseCoreDecisionHandle;
DENSECORE_DECISION_API DenseCoreDecisionHandle* DenseCoreDecisionLoad(const char* path, int threads, char* error,
                                                                      size_t error_capacity);
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
#ifdef __cplusplus
}
#endif
