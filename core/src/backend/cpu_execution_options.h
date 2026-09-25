#pragma once

#include "ggml.h"
#include <cstdint>
#include <thread>
#include <vector>

namespace densecore {
enum class CpuExecutionPhase { Unknown, Prefill, Decode };

// The caller owns observer data and keeps it alive until synchronous execution
// (including joined CPU tasks) returns. Recording happens at operator boundaries.
struct CpuExecutionTelemetry {
    void* data = nullptr;
    void (*MoEExpertMatmulWeightType)(void*, ggml_type weight_type) = nullptr;
    void (*MoEQ4KRepackedDecision)(void*, bool candidate, bool used, const char* reject_reason) = nullptr;
    void (*MoEQ5KRepackedDecision)(void*, bool candidate, bool used, const char* reject_reason) = nullptr;
    void (*MoEKQuantRawBatchedUse)(void*, ggml_type weight_type, uint64_t wall_ns, bool qwen_native_w2_q5k) = nullptr;
    void (*QwenNativeMoEQ4GateUpRowPairUse)(void*, uint64_t wall_ns) = nullptr;
    void (*Gemma4MoEPrefillQuantBatchDecision)(void*, bool candidate, bool used, const char* reject_reason,
                                               bool gate_up_used, bool down_used) = nullptr;
    void (*Gemma4NativeFusedGateUpUsed)(void*) = nullptr;
    void (*MatmulDispatchCensus)(void*, CpuExecutionPhase phase, const char* dispatch_path, ggml_type weight_type,
                                 int64_t m, int64_t n, int64_t k, uint64_t wall_ns) = nullptr;
    void (*MoESmallDecodeParallelDecision)(void*, bool candidate, bool used, const char* reject_reason,
                                           int selected_expert_count, int top_k, int task_count) = nullptr;
};

inline void RecordMoEExpertMatmulWeightType(const CpuExecutionTelemetry* sink, ggml_type weight_type) {
    if (sink && sink->MoEExpertMatmulWeightType) sink->MoEExpertMatmulWeightType(sink->data, weight_type);
}
inline void RecordMoEQ4KRepackedDecision(const CpuExecutionTelemetry* sink, bool candidate, bool used,
                                         const char* reject_reason) {
    if (sink && sink->MoEQ4KRepackedDecision) sink->MoEQ4KRepackedDecision(sink->data, candidate, used, reject_reason);
}
inline void RecordMoEQ5KRepackedDecision(const CpuExecutionTelemetry* sink, bool candidate, bool used,
                                         const char* reject_reason) {
    if (sink && sink->MoEQ5KRepackedDecision) sink->MoEQ5KRepackedDecision(sink->data, candidate, used, reject_reason);
}
inline void RecordMoEKQuantRawBatchedUse(const CpuExecutionTelemetry* sink, ggml_type weight_type, uint64_t wall_ns,
                                         bool qwen_native_w2_q5k) {
    if (sink && sink->MoEKQuantRawBatchedUse)
        sink->MoEKQuantRawBatchedUse(sink->data, weight_type, wall_ns, qwen_native_w2_q5k);
}
inline void RecordQwenNativeMoEQ4GateUpRowPairUse(const CpuExecutionTelemetry* sink, uint64_t wall_ns) {
    if (sink && sink->QwenNativeMoEQ4GateUpRowPairUse) sink->QwenNativeMoEQ4GateUpRowPairUse(sink->data, wall_ns);
}
inline void RecordGemma4MoEPrefillQuantBatchDecision(const CpuExecutionTelemetry* sink, bool candidate, bool used,
                                                     const char* reject_reason, bool gate_up_used, bool down_used) {
    if (sink && sink->Gemma4MoEPrefillQuantBatchDecision)
        sink->Gemma4MoEPrefillQuantBatchDecision(sink->data, candidate, used, reject_reason, gate_up_used, down_used);
}
inline void RecordGemma4NativeFusedGateUpUsed(const CpuExecutionTelemetry* sink) {
    if (sink && sink->Gemma4NativeFusedGateUpUsed) sink->Gemma4NativeFusedGateUpUsed(sink->data);
}
inline void RecordMatmulDispatchCensus(const CpuExecutionTelemetry* sink, CpuExecutionPhase phase,
                                       const char* dispatch_path, ggml_type weight_type, int64_t m, int64_t n,
                                       int64_t k, uint64_t wall_ns) {
    if (sink && sink->MatmulDispatchCensus)
        sink->MatmulDispatchCensus(sink->data, phase, dispatch_path, weight_type, m, n, k, wall_ns);
}
inline void RecordMoESmallDecodeParallelDecision(const CpuExecutionTelemetry* sink, bool candidate, bool used,
                                                 const char* reject_reason, int selected_expert_count, int top_k,
                                                 int task_count) {
    if (sink && sink->MoESmallDecodeParallelDecision)
        sink->MoESmallDecodeParallelDecision(sink->data, candidate, used, reject_reason, selected_expert_count, top_k,
                                             task_count);
}

struct CpuExecutionOptions {
    CpuExecutionPhase phase = CpuExecutionPhase::Unknown;
    bool has_work_context = false;
    bool is_lfm2 = false;
    const CpuExecutionTelemetry* telemetry = nullptr;
    // Preserve the pre-refactor phase/model policy on helper threads. Explicit
    // numerical callers can leave this unset to apply their options everywhere.
    bool thread_scoped_dispatch = false;
    std::thread::id dispatch_thread{};

    bool HasWorkContext() const {
        return has_work_context && (!thread_scoped_dispatch || dispatch_thread == std::this_thread::get_id());
    }
    CpuExecutionPhase EffectivePhase() const {
        return !thread_scoped_dispatch || HasWorkContext() ? phase : CpuExecutionPhase::Unknown;
    }
    bool IsLfm2() const { return is_lfm2 && (!thread_scoped_dispatch || HasWorkContext()); }
};

struct CpuBatchTraceView {
    const std::vector<int>& seq_id;
    const std::vector<int>& n_past;
};
}  // namespace densecore
