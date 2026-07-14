#ifndef DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H
#define DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H

#include <array>
#include <cstdint>
#include <cstring>

#include "densecore/models/model_types.h"

namespace densecore::runtime {

enum class GgmlComputeOp : int {
    Matmul = 0,
    MatmulId,
    QuantVecDot,
    QuantizeKv,
    Attention,
};

enum class DenseCoreMatmulPhase : int {
    Unknown = 0,
    Prefill,
    Decode,
};

enum class DenseCoreLayerRole : int {
    Unknown = 0,
    DenseProjection,
    HybridSsmQkv,
    HybridSsmGate,
    HybridSsmOut,
    LmHead,
    MoeGate,
    MoeUp,
    MoeDown,
};

enum class DenseCoreSemanticOp : int {
    Unknown = 0,
    Matmul,
    HybridSsmMixer,
    MoeRouter,
    MoeExpertDispatch,
    Lfm2ShortConvMixer,
    LmHead,
};

enum class DenseCoreTensorRole : int {
    Unknown = 0,
    DenseProjection,
    HybridSSMQkv,
    HybridSSMGate,
    SSMOut,
    MoERouter,
    MoEGateUp,
    MoEDown,
    LmHead,
    ShortConvIn,
    ShortConvOut,
};

enum class DenseCoreKernelFamily : int {
    None = 0,
    DenseCoreInt4Hwy,
    DenseCoreFp8,
    DenseCoreF32Gemv,
    DenseCoreF32SmallBatch,
    DenseCoreQuantGemv,
    DenseCoreQ4KBatched,
    DenseCoreQwenMoeDirect,
    DenseCoreHal,
    TemporaryReferenceGgml,
};

enum class DenseCoreHostBackend : int {
    Unknown = 0,
    GenericCpu,
    X86Amx,
    ArmSve2,
    GgmlCpuRepack,
    Hal,
};

enum class DenseCoreFallbackPolicyKind : int {
    CompatibilityFallback = 0,
    FallbackFreeTarget,
};

struct HostKernelCapabilities {
    bool x86_amx = false;
    bool arm_sve2 = false;
    bool q4k_true_batched = false;
    bool q5k_8x8 = false;
    bool ggml_repack = false;
};

struct QwenHotPathPlan {
    ModelVariant variant = ModelVariant::UNKNOWN;
    bool target_model = false;
    bool qwen35_0_8b = false;
    bool dense_lane = false;
    bool moe_lane = false;
    bool hybrid_ssm_lane = false;
};

struct TargetFastPathPlan {
    ModelVariant variant = ModelVariant::UNKNOWN;
    bool target_model = false;
    bool qwen_target = false;
    bool qwen35_0_8b = false;
    bool gemma4_target = false;
    bool lfm2_target = false;
    bool dense_lane = false;
    bool moe_lane = false;
    bool hybrid_ssm_lane = false;
    bool lfm2_shortconv_lane = false;
};

struct DenseCoreMatmulPlan {
    QwenHotPathPlan qwen;
    TargetFastPathPlan target;
    ModelVariant variant = ModelVariant::UNKNOWN;
    ggml_type weight_type = GGML_TYPE_COUNT;
    ggml_type input_type = GGML_TYPE_COUNT;
    DenseCoreMatmulPhase phase = DenseCoreMatmulPhase::Unknown;
    DenseCoreLayerRole role = DenseCoreLayerRole::Unknown;
    DenseCoreKernelFamily kernel = DenseCoreKernelFamily::None;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    bool compatible = false;
    bool quantized_weight = false;
    bool target_qwen_hot_path = false;
    bool target_fallback_free_path = false;
};

struct KernelResolution {
    DenseCoreMatmulPlan matmul_plan;
    DenseCoreSemanticOp semantic_op = DenseCoreSemanticOp::Unknown;
    DenseCoreTensorRole tensor_role = DenseCoreTensorRole::Unknown;
    DenseCoreKernelFamily selected_kernel = DenseCoreKernelFamily::None;
    DenseCoreHostBackend host_backend = DenseCoreHostBackend::Unknown;
    DenseCoreFallbackPolicyKind fallback_policy = DenseCoreFallbackPolicyKind::CompatibilityFallback;
    std::array<const char*, 4> rejected{};
    int rejected_count = 0;
};

inline const char* GgmlComputeOpName(GgmlComputeOp op) {
    switch (op) {
    case GgmlComputeOp::Matmul: return "ggml_mul_mat";
    case GgmlComputeOp::MatmulId: return "ggml_mul_mat_id";
    case GgmlComputeOp::QuantVecDot: return "ggml_quant_vecdot";
    case GgmlComputeOp::QuantizeKv: return "ggml_quantize_kv";
    case GgmlComputeOp::Attention: return "ggml_attention";
    default: return "unknown";
    }
}

inline const char* DenseCoreSemanticOpName(DenseCoreSemanticOp op) {
    switch (op) {
    case DenseCoreSemanticOp::Matmul: return "matmul";
    case DenseCoreSemanticOp::HybridSsmMixer: return "hybrid_ssm_mixer";
    case DenseCoreSemanticOp::MoeRouter: return "moe_router";
    case DenseCoreSemanticOp::MoeExpertDispatch: return "moe_expert_dispatch";
    case DenseCoreSemanticOp::Lfm2ShortConvMixer: return "lfm2_shortconv_mixer";
    case DenseCoreSemanticOp::LmHead: return "lm_head";
    default: return "unknown";
    }
}

inline const char* DenseCoreTensorRoleName(DenseCoreTensorRole role) {
    switch (role) {
    case DenseCoreTensorRole::DenseProjection: return "dense_projection";
    case DenseCoreTensorRole::HybridSSMQkv: return "hybrid_ssm_qkv";
    case DenseCoreTensorRole::HybridSSMGate: return "hybrid_ssm_gate";
    case DenseCoreTensorRole::SSMOut: return "ssm_out";
    case DenseCoreTensorRole::MoERouter: return "moe_router";
    case DenseCoreTensorRole::MoEGateUp: return "moe_gate_up";
    case DenseCoreTensorRole::MoEDown: return "moe_down";
    case DenseCoreTensorRole::LmHead: return "lm_head";
    case DenseCoreTensorRole::ShortConvIn: return "shortconv_in";
    case DenseCoreTensorRole::ShortConvOut: return "shortconv_out";
    default: return "unknown";
    }
}

inline bool IsHybridSsmTensorRole(DenseCoreTensorRole role) {
    return role == DenseCoreTensorRole::HybridSSMQkv || role == DenseCoreTensorRole::HybridSSMGate ||
           role == DenseCoreTensorRole::SSMOut;
}

inline bool IsShortConvTensorRole(DenseCoreTensorRole role) {
    return role == DenseCoreTensorRole::ShortConvIn || role == DenseCoreTensorRole::ShortConvOut;
}

inline bool IsMoeExpertTensorRole(DenseCoreTensorRole role) {
    return role == DenseCoreTensorRole::MoEGateUp || role == DenseCoreTensorRole::MoEDown;
}

inline int HybridSsmProjectionKind(DenseCoreTensorRole role) {
    switch (role) {
    case DenseCoreTensorRole::HybridSSMQkv: return 1;
    case DenseCoreTensorRole::HybridSSMGate: return 2;
    case DenseCoreTensorRole::SSMOut: return 3;
    default: return 0;
    }
}

inline bool IsQwenTargetVariant(ModelVariant variant) {
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36;
}

inline bool IsDenseCoreFallbackFreeTargetVariant(ModelVariant variant) {
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36 || variant == ModelVariant::GEMMA4 ||
           variant == ModelVariant::LFM2MOE;
}

inline bool IsQwen35Point8B(const TransformerModel* model) {
    return model && model->variant == ModelVariant::QWEN35 && model->hparams.n_experts == 0 &&
           model->arch_flags.is_hybrid_ssm && model->hparams.n_embd == 1024 && model->hparams.n_layer == 24;
}

inline QwenHotPathPlan ResolveQwenHotPathPlan(const TransformerModel* model) {
    QwenHotPathPlan plan;
    if (!model) {
        return plan;
    }
    plan.variant = model->variant;
    plan.target_model = IsQwenTargetVariant(model->variant);
    plan.qwen35_0_8b = IsQwen35Point8B(model);
    plan.dense_lane = plan.target_model && model->hparams.n_experts == 0 && !model->arch_flags.is_hybrid_ssm;
    plan.moe_lane = plan.target_model && model->hparams.n_experts > 0;
    plan.hybrid_ssm_lane = plan.target_model && model->arch_flags.is_hybrid_ssm;
    return plan;
}

inline TargetFastPathPlan ResolveTargetFastPathPlan(const TransformerModel* model) {
    TargetFastPathPlan plan;
    if (!model) {
        return plan;
    }
    plan.variant = model->variant;
    plan.qwen_target =
        IsQwenTargetVariant(model->variant) || (model->arch == ModelArch::QWEN35 && model->arch_flags.is_hybrid_ssm);
    plan.qwen35_0_8b = IsQwen35Point8B(model);
    plan.gemma4_target = model->variant == ModelVariant::GEMMA4 || model->arch_flags.is_gemma4;
    plan.lfm2_target = model->variant == ModelVariant::LFM2MOE || model->arch_flags.is_lfm2_shortconv;
    plan.target_model = IsDenseCoreFallbackFreeTargetVariant(model->variant) || plan.qwen_target ||
                        plan.gemma4_target || plan.lfm2_target;
    plan.dense_lane = plan.target_model && model->hparams.n_experts == 0 && !model->arch_flags.is_hybrid_ssm &&
                      !model->arch_flags.is_lfm2_shortconv;
    plan.moe_lane = plan.target_model && model->hparams.n_experts > 0;
    plan.hybrid_ssm_lane = plan.qwen_target && model->arch_flags.is_hybrid_ssm;
    plan.lfm2_shortconv_lane = plan.lfm2_target && model->arch_flags.is_lfm2_shortconv;
    return plan;
}

inline const char* QwenHotPathTargetLabel(const QwenHotPathPlan& plan) {
    if (!plan.target_model) {
        return "non_qwen_target";
    }
    if (plan.variant == ModelVariant::QWEN36) {
        return "qwen36_35b_a3b";
    }
    if (plan.moe_lane) {
        return "qwen35_35b_a3b";
    }
    if (plan.qwen35_0_8b) {
        return "qwen35_0.8b_dense";
    }
    return "qwen35_9b_dense";
}

inline const char* TargetFastPathLabel(const TargetFastPathPlan& plan) {
    if (!plan.target_model) {
        return "compatibility_model";
    }
    if (plan.qwen_target) {
        QwenHotPathPlan qwen;
        qwen.variant = plan.variant;
        qwen.target_model = true;
        qwen.qwen35_0_8b = plan.qwen35_0_8b;
        qwen.dense_lane = plan.dense_lane;
        qwen.moe_lane = plan.moe_lane;
        qwen.hybrid_ssm_lane = plan.hybrid_ssm_lane;
        return QwenHotPathTargetLabel(qwen);
    }
    if (plan.gemma4_target) {
        return "gemma4_26b_a4b";
    }
    if (plan.lfm2_target) {
        return "lfm2_8b_a1b";
    }
    return "fallback_free_target";
}

inline const char* DenseCoreKernelFamilyName(DenseCoreKernelFamily family) {
    switch (family) {
    case DenseCoreKernelFamily::DenseCoreInt4Hwy: return "densecore_int4_hwy";
    case DenseCoreKernelFamily::DenseCoreFp8: return "densecore_fp8";
    case DenseCoreKernelFamily::DenseCoreF32Gemv: return "densecore_f32_gemv";
    case DenseCoreKernelFamily::DenseCoreF32SmallBatch: return "densecore_f32_small_batch";
    case DenseCoreKernelFamily::DenseCoreQuantGemv: return "densecore_quant_gemv";
    case DenseCoreKernelFamily::DenseCoreQ4KBatched: return "densecore_q4k_batched";
    case DenseCoreKernelFamily::DenseCoreQwenMoeDirect: return "densecore_qwen_moe_direct";
    case DenseCoreKernelFamily::DenseCoreHal: return "densecore_hal";
    case DenseCoreKernelFamily::TemporaryReferenceGgml: return "temporary_reference_ggml";
    default: return "none";
    }
}

inline const char* DenseCoreHostBackendName(DenseCoreHostBackend backend) {
    switch (backend) {
    case DenseCoreHostBackend::GenericCpu: return "generic_cpu";
    case DenseCoreHostBackend::X86Amx: return "x86_amx";
    case DenseCoreHostBackend::ArmSve2: return "arm_sve2";
    case DenseCoreHostBackend::GgmlCpuRepack: return "ggml_cpu_repack";
    case DenseCoreHostBackend::Hal: return "hal";
    default: return "unknown";
    }
}

inline const char* DenseCoreFallbackPolicyKindName(DenseCoreFallbackPolicyKind policy) {
    switch (policy) {
    case DenseCoreFallbackPolicyKind::FallbackFreeTarget: return "fallback_free_target";
    case DenseCoreFallbackPolicyKind::CompatibilityFallback: return "compatibility_fallback";
    }
    return "unknown";
}

inline DenseCoreLayerRole ResolveDenseCoreLayerRole(const TransformerModel* model, const char* weight_name,
                                                    bool is_lm_head) {
    if (is_lm_head) {
        return DenseCoreLayerRole::LmHead;
    }
    if (!weight_name) {
        return DenseCoreLayerRole::Unknown;
    }
    if (std::strstr(weight_name, "ffn_gate_inp") || std::strstr(weight_name, "moe_gate")) {
        return DenseCoreLayerRole::MoeGate;
    }
    if (std::strstr(weight_name, "ffn_gate_exps")) return DenseCoreLayerRole::MoeGate;
    if (std::strstr(weight_name, "ffn_up_exps")) return DenseCoreLayerRole::MoeUp;
    if (std::strstr(weight_name, "ffn_down_exps")) return DenseCoreLayerRole::MoeDown;
    if (model && model->arch_flags.is_hybrid_ssm) {
        if (std::strstr(weight_name, "attn_qkv")) return DenseCoreLayerRole::HybridSsmQkv;
        if (std::strstr(weight_name, "attn_gate")) return DenseCoreLayerRole::HybridSsmGate;
        if (std::strstr(weight_name, "ssm_out")) return DenseCoreLayerRole::HybridSsmOut;
    }
    return DenseCoreLayerRole::DenseProjection;
}

inline DenseCoreTensorRole TensorRoleFromLayerRole(DenseCoreLayerRole role) {
    switch (role) {
    case DenseCoreLayerRole::DenseProjection: return DenseCoreTensorRole::DenseProjection;
    case DenseCoreLayerRole::HybridSsmQkv: return DenseCoreTensorRole::HybridSSMQkv;
    case DenseCoreLayerRole::HybridSsmGate: return DenseCoreTensorRole::HybridSSMGate;
    case DenseCoreLayerRole::HybridSsmOut: return DenseCoreTensorRole::SSMOut;
    case DenseCoreLayerRole::LmHead: return DenseCoreTensorRole::LmHead;
    case DenseCoreLayerRole::MoeGate:
    case DenseCoreLayerRole::MoeUp: return DenseCoreTensorRole::MoEGateUp;
    case DenseCoreLayerRole::MoeDown: return DenseCoreTensorRole::MoEDown;
    default: return DenseCoreTensorRole::Unknown;
    }
}

inline DenseCoreTensorRole ResolveDenseCoreTensorRole(const TransformerModel* model, const char* weight_name,
                                                      bool is_lm_head) {
    if (is_lm_head) {
        return DenseCoreTensorRole::LmHead;
    }
    if (weight_name) {
        if (std::strstr(weight_name, "shortconv.in") || std::strstr(weight_name, "conv_L_cache") ||
            std::strstr(weight_name, "short_conv_in") || std::strstr(weight_name, "shortconv_in_proj")) {
            return DenseCoreTensorRole::ShortConvIn;
        }
        if (std::strstr(weight_name, "shortconv.out") || std::strstr(weight_name, "short_conv_out") ||
            std::strstr(weight_name, "shortconv_out_proj")) {
            return DenseCoreTensorRole::ShortConvOut;
        }
        if (std::strstr(weight_name, "ffn_gate_inp") || std::strstr(weight_name, "moe_gate")) {
            return DenseCoreTensorRole::MoERouter;
        }
        if (std::strstr(weight_name, "ffn_gate_up_exps") || std::strstr(weight_name, "experts.gate_up_proj")) {
            return DenseCoreTensorRole::MoEGateUp;
        }
    }
    return TensorRoleFromLayerRole(ResolveDenseCoreLayerRole(model, weight_name, is_lm_head));
}

inline DenseCoreSemanticOp ResolveDenseCoreSemanticOp(const TransformerModel* model, DenseCoreTensorRole role) {
    switch (role) {
    case DenseCoreTensorRole::HybridSSMQkv:
    case DenseCoreTensorRole::HybridSSMGate:
    case DenseCoreTensorRole::SSMOut: return DenseCoreSemanticOp::HybridSsmMixer;
    case DenseCoreTensorRole::MoERouter: return DenseCoreSemanticOp::MoeRouter;
    case DenseCoreTensorRole::MoEGateUp:
    case DenseCoreTensorRole::MoEDown: return DenseCoreSemanticOp::MoeExpertDispatch;
    case DenseCoreTensorRole::ShortConvIn:
    case DenseCoreTensorRole::ShortConvOut: return DenseCoreSemanticOp::Lfm2ShortConvMixer;
    case DenseCoreTensorRole::LmHead: return DenseCoreSemanticOp::LmHead;
    case DenseCoreTensorRole::DenseProjection:
        if (model && model->arch_flags.is_lfm2_shortconv) {
            return DenseCoreSemanticOp::Lfm2ShortConvMixer;
        }
        return DenseCoreSemanticOp::Matmul;
    default: return DenseCoreSemanticOp::Unknown;
    }
}

inline DenseCoreKernelFamily ResolveMaintainedKernelFamilyForTensorRole(ggml_type raw_type, DenseCoreMatmulPhase phase,
                                                                        DenseCoreTensorRole role) {
    if (role == DenseCoreTensorRole::MoEGateUp || role == DenseCoreTensorRole::MoEDown) {
        return DenseCoreKernelFamily::DenseCoreQwenMoeDirect;
    }
    if (raw_type == GGML_TYPE_F32) {
        return phase == DenseCoreMatmulPhase::Decode ? DenseCoreKernelFamily::DenseCoreF32Gemv
                                                     : DenseCoreKernelFamily::DenseCoreF32SmallBatch;
    }
    if (raw_type == GGML_TYPE_Q4_K && phase == DenseCoreMatmulPhase::Prefill) {
        return DenseCoreKernelFamily::DenseCoreQ4KBatched;
    }
    if (ggml_is_quantized(raw_type)) {
        return DenseCoreKernelFamily::DenseCoreQuantGemv;
    }
    return DenseCoreKernelFamily::TemporaryReferenceGgml;
}

inline HostKernelCapabilities CompileTimeHostKernelCapabilities(bool q4k_true_batched = false,
                                                                bool ggml_repack = false) {
    HostKernelCapabilities caps;
#if defined(__x86_64__) || defined(_M_X64)
    caps.x86_amx = true;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    caps.arm_sve2 = true;
#endif
    caps.q4k_true_batched = q4k_true_batched;
    caps.q5k_8x8 = caps.x86_amx || caps.arm_sve2;
    caps.ggml_repack = ggml_repack;
    return caps;
}

inline DenseCoreMatmulPlan ResolveDenseCoreMatmulPlan(const TransformerModel* model, ggml_type weight_type,
                                                      ggml_type input_type, int64_t m, int64_t n, int64_t k,
                                                      DenseCoreMatmulPhase phase, const char* weight_name,
                                                      bool is_lm_head, bool compatible) {
    DenseCoreMatmulPlan plan;
    plan.qwen = ResolveQwenHotPathPlan(model);
    plan.target = ResolveTargetFastPathPlan(model);
    plan.variant = model ? model->variant : ModelVariant::UNKNOWN;
    plan.weight_type = weight_type;
    plan.input_type = input_type;
    plan.phase = phase;
    plan.role = ResolveDenseCoreLayerRole(model, weight_name, is_lm_head);
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.compatible = compatible;
    plan.quantized_weight = ggml_is_quantized(weight_type);
    plan.target_qwen_hot_path = plan.qwen.target_model;
    plan.target_fallback_free_path = plan.target.target_model;

    if (!compatible || input_type != GGML_TYPE_F32) {
        plan.kernel = DenseCoreKernelFamily::TemporaryReferenceGgml;
    } else if (weight_type == GGML_TYPE_F32 && m <= 1) {
        plan.kernel = DenseCoreKernelFamily::DenseCoreF32Gemv;
    } else if (weight_type == GGML_TYPE_F32 && m > 1 && m <= 8) {
        plan.kernel = DenseCoreKernelFamily::DenseCoreF32SmallBatch;
    } else if (weight_type == GGML_TYPE_Q4_K && m > 1) {
        plan.kernel = DenseCoreKernelFamily::DenseCoreQ4KBatched;
    } else if (plan.quantized_weight && m <= 1) {
        plan.kernel = DenseCoreKernelFamily::DenseCoreQuantGemv;
    } else if (plan.quantized_weight) {
        plan.kernel = DenseCoreKernelFamily::DenseCoreQuantGemv;
    }
    return plan;
}

inline DenseCoreHostBackend ResolveDenseCoreHostBackend(DenseCoreKernelFamily kernel,
                                                        const HostKernelCapabilities& caps) {
    if (kernel == DenseCoreKernelFamily::DenseCoreHal) {
        return DenseCoreHostBackend::Hal;
    }
    if (caps.ggml_repack && kernel == DenseCoreKernelFamily::TemporaryReferenceGgml) {
        return DenseCoreHostBackend::GgmlCpuRepack;
    }
    if (caps.arm_sve2 &&
        (kernel == DenseCoreKernelFamily::DenseCoreQ4KBatched || kernel == DenseCoreKernelFamily::DenseCoreQuantGemv)) {
        return DenseCoreHostBackend::ArmSve2;
    }
    if (caps.x86_amx &&
        (kernel == DenseCoreKernelFamily::DenseCoreQ4KBatched || kernel == DenseCoreKernelFamily::DenseCoreQuantGemv ||
         kernel == DenseCoreKernelFamily::DenseCoreF32Gemv)) {
        return DenseCoreHostBackend::X86Amx;
    }
    return DenseCoreHostBackend::GenericCpu;
}

inline KernelResolution ResolveKernelResolution(
    const TransformerModel* model, ggml_type weight_type, ggml_type input_type, int64_t m, int64_t n, int64_t k,
    DenseCoreMatmulPhase phase, const char* weight_name, bool is_lm_head, bool compatible,
    HostKernelCapabilities caps = HostKernelCapabilities{},
    DenseCoreSemanticOp semantic_override = DenseCoreSemanticOp::Unknown,
    DenseCoreTensorRole role_override = DenseCoreTensorRole::Unknown,
    DenseCoreKernelFamily kernel_override = DenseCoreKernelFamily::None,
    DenseCoreFallbackPolicyKind fallback_policy_override = DenseCoreFallbackPolicyKind::CompatibilityFallback,
    bool has_fallback_policy_override = false) {
    KernelResolution resolution;
    resolution.matmul_plan =
        ResolveDenseCoreMatmulPlan(model, weight_type, input_type, m, n, k, phase, weight_name, is_lm_head, compatible);
    resolution.tensor_role = role_override != DenseCoreTensorRole::Unknown
                                 ? role_override
                                 : ResolveDenseCoreTensorRole(model, weight_name, is_lm_head);
    resolution.semantic_op = semantic_override != DenseCoreSemanticOp::Unknown
                                 ? semantic_override
                                 : ResolveDenseCoreSemanticOp(model, resolution.tensor_role);
    resolution.selected_kernel =
        kernel_override != DenseCoreKernelFamily::None
            ? kernel_override
            : (role_override != DenseCoreTensorRole::Unknown
                   ? ResolveMaintainedKernelFamilyForTensorRole(weight_type, phase, resolution.tensor_role)
                   : resolution.matmul_plan.kernel);
    resolution.fallback_policy =
        has_fallback_policy_override
            ? fallback_policy_override
            : (resolution.matmul_plan.target_fallback_free_path ? DenseCoreFallbackPolicyKind::FallbackFreeTarget
                                                                : DenseCoreFallbackPolicyKind::CompatibilityFallback);
    resolution.host_backend = ResolveDenseCoreHostBackend(resolution.selected_kernel, caps);

    auto reject = [&resolution](const char* reason) {
        if (resolution.rejected_count < static_cast<int>(resolution.rejected.size())) {
            resolution.rejected[static_cast<std::size_t>(resolution.rejected_count)] = reason;
        }
        ++resolution.rejected_count;
    };
    if (!compatible) {
        reject("incompatible_shape");
    }
    if (input_type != GGML_TYPE_F32) {
        reject("unsupported_input_type");
    }
    if (weight_type == GGML_TYPE_Q4_K && m > 1 && !caps.q4k_true_batched) {
        reject("q4k_true_batched_unavailable");
    }
    if (resolution.selected_kernel == DenseCoreKernelFamily::None) {
        reject("no_selected_kernel");
    }
    return resolution;
}

inline bool KernelResolutionTargetsQ4KBatchedPrefill(const KernelResolution& resolution) {
    return resolution.selected_kernel == DenseCoreKernelFamily::DenseCoreQ4KBatched &&
           resolution.matmul_plan.phase == DenseCoreMatmulPhase::Prefill &&
           resolution.matmul_plan.weight_type == GGML_TYPE_Q4_K && resolution.matmul_plan.input_type == GGML_TYPE_F32 &&
           resolution.matmul_plan.m > 1;
}

inline bool KernelResolutionSelectsQ4KBatchedPrefill(const KernelResolution& resolution) {
    return KernelResolutionTargetsQ4KBatchedPrefill(resolution) && resolution.matmul_plan.compatible;
}

inline bool IsExplicitTemporaryReferenceFallback(const char* reason) {
    if (!reason || reason[0] == '\0') {
        return false;
    }
    return std::strncmp(reason, "temporary_reference_", 20) == 0 ||
           std::strncmp(reason, "reference_parity_", 17) == 0 || std::strcmp(reason, "gguf_scaffold") == 0 ||
           std::strcmp(reason, "test_reference") == 0;
}

struct DenseCoreFallbackPolicy {
    static constexpr bool kEnvOverrideAllowed = false;

    static bool AllowsGgmlCompute(const QwenHotPathPlan& plan, const char* reason) {
        (void)reason;
        if (!plan.target_model) {
            return true;
        }
        return false;
    }

    static bool AllowsGgmlCompute(const TargetFastPathPlan& plan, const char* reason) {
        (void)reason;
        if (!plan.target_model) {
            return true;
        }
        return false;
    }
};

inline bool ShouldRejectQwenGgmlCompute(const QwenHotPathPlan& plan, const char* reason) {
    return !DenseCoreFallbackPolicy::AllowsGgmlCompute(plan, reason);
}

inline bool ShouldRejectTargetGgmlCompute(const TargetFastPathPlan& plan, const char* reason) {
    return !DenseCoreFallbackPolicy::AllowsGgmlCompute(plan, reason);
}

}  // namespace densecore::runtime

#endif  // DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H
