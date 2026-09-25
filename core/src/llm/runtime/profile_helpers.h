#pragma once
#include <atomic>
#include <cstddef>
#include "densecore/runtime/inference.h"
namespace densecore::runtime { struct KernelAdmissionDecision; }
void RecordKleidiAIAdmissionDecision(InferenceWorkContext*, const densecore::runtime::KernelAdmissionDecision&);

inline std::size_t MatmulWeightTypeHistIndex(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return 0;
        case GGML_TYPE_Q5_K:
            return 1;
        case GGML_TYPE_Q6_K:
            return 2;
        case GGML_TYPE_Q8_0:
            return 3;
        case GGML_TYPE_F16:
            return 4;
        case GGML_TYPE_F32:
            return 5;
        default:
            return 6;
    }
}

inline std::size_t MatmulQuantInputTypeHistIndex(ggml_type type, bool has_quant_input) {
    if (!has_quant_input) {
        return 2;
    }
    switch (type) {
        case GGML_TYPE_Q8_K:
            return 0;
        case GGML_TYPE_Q8_0:
            return 1;
        default:
            return 3;
    }
}

inline const char* MatmulPhaseName(InferenceExecutionPhase phase) {
    switch (phase) {
        case InferenceExecutionPhase::Prefill:
            return "prefill";
        case InferenceExecutionPhase::Decode:
            return "decode";
        case InferenceExecutionPhase::Unknown:
        default:
            return "unknown";
    }
}

inline void AddQwen36ProfileNs(std::atomic<uint64_t>& counter, uint64_t value) {
    if (!IsQwen36ProfilingEnabled() || value == 0) {
        return;
    }
    counter.fetch_add(value, std::memory_order_relaxed);
}

void SetQwen36ProfileMax(std::atomic<int>& counter, int value);
void MarkQwen36ProfileFlag(std::atomic<int>& counter);
