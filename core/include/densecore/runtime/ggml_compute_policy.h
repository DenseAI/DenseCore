#ifndef DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H
#define DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H

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

struct QwenHotPathPlan {
    ModelVariant variant = ModelVariant::UNKNOWN;
    bool target_model = false;
    bool dense_lane = false;
    bool moe_lane = false;
    bool hybrid_ssm_lane = false;
};

struct DenseCoreMatmulPlan {
    QwenHotPathPlan qwen;
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

inline bool IsQwenTargetVariant(ModelVariant variant) {
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36;
}

inline QwenHotPathPlan ResolveQwenHotPathPlan(const TransformerModel* model) {
    QwenHotPathPlan plan;
    if (!model) {
        return plan;
    }
    plan.variant = model->variant;
    plan.target_model = IsQwenTargetVariant(model->variant);
    plan.dense_lane = plan.target_model && model->hparams.n_experts == 0 && !model->arch_flags.is_hybrid_ssm;
    plan.moe_lane = plan.target_model && model->hparams.n_experts > 0;
    plan.hybrid_ssm_lane = plan.target_model && model->arch_flags.is_hybrid_ssm;
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
    return "qwen35_9b_dense";
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

inline DenseCoreLayerRole ResolveDenseCoreLayerRole(const TransformerModel* model, const char* weight_name,
                                                    bool is_lm_head) {
    if (is_lm_head) {
        return DenseCoreLayerRole::LmHead;
    }
    if (!weight_name) {
        return DenseCoreLayerRole::Unknown;
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

inline DenseCoreMatmulPlan ResolveDenseCoreMatmulPlan(const TransformerModel* model, ggml_type weight_type,
                                                      ggml_type input_type, int64_t m, int64_t n, int64_t k,
                                                      DenseCoreMatmulPhase phase, const char* weight_name,
                                                      bool is_lm_head, bool compatible) {
    DenseCoreMatmulPlan plan;
    plan.qwen = ResolveQwenHotPathPlan(model);
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
};

inline bool ShouldRejectQwenGgmlCompute(const QwenHotPathPlan& plan, const char* reason) {
    return !DenseCoreFallbackPolicy::AllowsGgmlCompute(plan, reason);
}

}  // namespace densecore::runtime

#endif  // DENSECORE_PUBLIC_RUNTIME_GGML_COMPUTE_POLICY_H
