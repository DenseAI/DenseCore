#ifndef DENSECORE_LLM_RUNTIME_WORK_CONTEXT_H
#define DENSECORE_LLM_RUNTIME_WORK_CONTEXT_H

#include "densecore/runtime/inference.h"

// Policy provenance remains the caller's context. A callback binding supplies
// identity/telemetry without enabling a fast path on previously unbound threads.
InferenceWorkContext* GetDispatchWorkContext();
void SetCallbackWorkContext(InferenceWorkContext* context, InferenceWorkContext* dispatch_context);
struct GemvUserData;
struct GemvBatchedUserData;
GemvUserData* GetGemvUserData();
GemvBatchedUserData* GetGemvBatchedUserData();
void BindInferenceWorkContextBatch(InferenceWorkContext* context, const BatchSpec* batch);
const BatchSpec* GetInferenceWorkContextBatch(const InferenceWorkContext* context);
void SetInferenceWorkContextPhase(InferenceWorkContext* context, InferenceExecutionPhase phase);

struct Qwen36ProfileCounters;
Qwen36ProfileCounters* GetInferenceWorkContextProfile(InferenceWorkContext* context);

// GGML invokes callbacks on its own threads. Bind their graph-owned context
// for the callback duration and restore any outer context on return.
class ScopedInferenceWorkContext {
public:
    explicit ScopedInferenceWorkContext(InferenceWorkContext* context)
        : previous_(GetCurrentWorkContext()), previous_dispatch_(GetDispatchWorkContext()) {
        if (context) SetCallbackWorkContext(context, previous_dispatch_);
    }
    ~ScopedInferenceWorkContext() { SetCallbackWorkContext(previous_, previous_dispatch_); }
    ScopedInferenceWorkContext(const ScopedInferenceWorkContext&) = delete;
    ScopedInferenceWorkContext& operator=(const ScopedInferenceWorkContext&) = delete;

private:
    InferenceWorkContext* previous_;
    InferenceWorkContext* previous_dispatch_;
};

#endif
