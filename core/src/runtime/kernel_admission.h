#ifndef DENSECORE_RUNTIME_KERNEL_ADMISSION_H
#define DENSECORE_RUNTIME_KERNEL_ADMISSION_H

#include <cstdint>
#include <cstring>

#include "densecore/models/model_types.h"
#include "ggml.h"

namespace densecore::runtime {

enum class KernelAdmissionRejectReason : int {
    None = 0,
    BackendNotCompiled,
    UnsupportedTensorType,
    UnsupportedShape,
    QualityGateUnpromoted,
    ModelUnsupported,
};

enum class KernelOpKind : int {
    Unknown = 0,
    AttentionQKV,
    AttentionOutput,
    SsmGate,
    SsmOutput,
    Router,
    FfnGate,
    FfnUp,
    FfnDown,
    FfnGateUp,
    LmHead,
    Embedding,
};

struct KernelAdmissionDescriptor {
    ModelVariant model_variant = ModelVariant::UNKNOWN;
    bool is_gemma4 = false;
    bool is_hybrid_ssm = false;
    bool is_moe = false;
    bool is_lm_head = false;
    const char* weight_name = nullptr;
    ggml_type weight_type = GGML_TYPE_COUNT;
    ggml_type input_type = GGML_TYPE_COUNT;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
};

struct KernelAdmissionDecision {
    bool candidate = false;
    bool allowed = false;
    KernelOpKind op_kind = KernelOpKind::Unknown;
    KernelAdmissionRejectReason reject_reason = KernelAdmissionRejectReason::None;
};

inline const char* KernelAdmissionRejectReasonName(KernelAdmissionRejectReason reason) {
    switch (reason) {
    case KernelAdmissionRejectReason::None: return "none";
    case KernelAdmissionRejectReason::BackendNotCompiled: return "backend_not_compiled";
    case KernelAdmissionRejectReason::UnsupportedTensorType: return "unsupported_tensor_type";
    case KernelAdmissionRejectReason::UnsupportedShape: return "unsupported_shape";
    case KernelAdmissionRejectReason::QualityGateUnpromoted: return "quality_gate_unpromoted";
    case KernelAdmissionRejectReason::ModelUnsupported: return "model_unsupported";
    default: return "unknown";
    }
}

inline const char* KernelOpKindName(KernelOpKind kind) {
    switch (kind) {
    case KernelOpKind::AttentionQKV: return "attention_qkv";
    case KernelOpKind::AttentionOutput: return "attention_output";
    case KernelOpKind::SsmGate: return "ssm_gate";
    case KernelOpKind::SsmOutput: return "ssm_output";
    case KernelOpKind::Router: return "router";
    case KernelOpKind::FfnGate: return "ffn_gate";
    case KernelOpKind::FfnUp: return "ffn_up";
    case KernelOpKind::FfnDown: return "ffn_down";
    case KernelOpKind::FfnGateUp: return "ffn_gate_up";
    case KernelOpKind::LmHead: return "lm_head";
    case KernelOpKind::Embedding: return "embedding";
    case KernelOpKind::Unknown:
    default: return "unknown";
    }
}

inline bool KleidiAICompiledEnabled() {
#if defined(DENSECORE_GGML_CPU_KLEIDIAI) && DENSECORE_GGML_CPU_KLEIDIAI
    return true;
#else
    return false;
#endif
}

inline bool KleidiAITensorTypeCandidate(ggml_type type) {
    // ggml's KleidiAI integration currently advertises Q4_0/Q8_0 kernels. Q4_K,
    // Q5_K, and MoE packed layouts must stay on DenseCore/ggml reference paths
    // until they have their own explicit parity gate.
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0;
}

inline KernelOpKind ClassifyKernelOpKind(const KernelAdmissionDescriptor& desc) {
    if (desc.is_lm_head) {
        return KernelOpKind::LmHead;
    }
    const char* name = desc.weight_name ? desc.weight_name : "";
    if (std::strstr(name, "tok_embeddings")) return KernelOpKind::Embedding;
    if (std::strstr(name, "attn_qkv") || std::strstr(name, "qkv")) return KernelOpKind::AttentionQKV;
    if (std::strstr(name, "attn_output") || std::strstr(name, "o_proj") || std::strstr(name, ".wo")) {
        return KernelOpKind::AttentionOutput;
    }
    if (std::strstr(name, "attn_gate")) return KernelOpKind::SsmGate;
    if (std::strstr(name, "ssm_out")) return KernelOpKind::SsmOutput;
    if (std::strstr(name, "ffn_gate_inp") || std::strstr(name, "moe_gate") || std::strstr(name, "router")) {
        return KernelOpKind::Router;
    }
    if (std::strstr(name, "ffn_gate_up")) return KernelOpKind::FfnGateUp;
    if (std::strstr(name, "ffn_gate")) return KernelOpKind::FfnGate;
    if (std::strstr(name, "ffn_up")) return KernelOpKind::FfnUp;
    if (std::strstr(name, "ffn_down")) return KernelOpKind::FfnDown;
    return KernelOpKind::Unknown;
}

inline KernelAdmissionDecision EvaluateKleidiAIAdmission(const KernelAdmissionDescriptor& desc) {
    KernelAdmissionDecision decision;
    decision.op_kind = ClassifyKernelOpKind(desc);

    if (desc.input_type != GGML_TYPE_F32 || !KleidiAITensorTypeCandidate(desc.weight_type)) {
        decision.reject_reason = KernelAdmissionRejectReason::UnsupportedTensorType;
        return decision;
    }
    decision.candidate = true;

    if (desc.m <= 1 || desc.n <= 0 || desc.k <= 0) {
        decision.reject_reason = KernelAdmissionRejectReason::UnsupportedShape;
        return decision;
    }
    if (!KleidiAICompiledEnabled()) {
        decision.reject_reason = KernelAdmissionRejectReason::BackendNotCompiled;
        return decision;
    }
    if (!(desc.model_variant == ModelVariant::QWEN35 || desc.model_variant == ModelVariant::QWEN36 ||
          desc.model_variant == ModelVariant::QWEN38 ||
          desc.model_variant == ModelVariant::GEMMA4 || desc.is_gemma4 || desc.is_hybrid_ssm || desc.is_moe)) {
        decision.reject_reason = KernelAdmissionRejectReason::ModelUnsupported;
        return decision;
    }

    // Quality-first default: no model/op/shape tuple is admitted until parity
    // probes and real Go-server QA promote it. This prevents the C4A Gemma4
    // "--" regression class from becoming an implicit backend choice.
    decision.reject_reason = KernelAdmissionRejectReason::QualityGateUnpromoted;
    return decision;
}

}  // namespace densecore::runtime

#endif  // DENSECORE_RUNTIME_KERNEL_ADMISSION_H
