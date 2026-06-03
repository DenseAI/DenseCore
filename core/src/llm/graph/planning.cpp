#include "llm/graph/planning.h"

#include "densecore/models/transformer_graph_builder.h"

namespace {

densecore::models::GraphBuilderSupport MakeDenseCoreInlineGraphSupport() {
    densecore::models::GraphBuilderSupport support{};
    support.builder_name = "DenseCoreInlineTransformerGraph";
    support.supported_families = {densecore::models::GraphFamily::DecoderDenseAttention,
                                  densecore::models::GraphFamily::DecoderHybridSSM,
                                  densecore::models::GraphFamily::DecoderSlidingWindowSharedKV};
    support.supports_sliding_window_attention = true;
    support.supports_shared_kv_source = true;
    support.supports_hybrid_ssm_mixer = true;
    support.supports_lfm2_shortconv_mixer = true;
    support.supports_moe = true;
    support.supports_q_norm = true;
    support.supports_k_norm = true;
    support.supports_v_norm = true;
    support.supports_per_layer_kv_head_variability = true;
    support.supports_special_attention_mask = true;
    support.supports_special_residual_scaling = true;
    support.supports_shared_dense_ffn = true;
    support.supports_moe_down_scale_sidecar = true;
    support.supports_ffn_post_norms = true;
    support.supported_moe_routers = {densecore::models::DecoderMoERouter::SoftmaxTopK,
                                     densecore::models::DecoderMoERouter::Gemma4SoftmaxTopK,
                                     densecore::models::DecoderMoERouter::GroupedSigmoidTopK};
    support.supported_ffn_activations = {densecore::models::DecoderActivation::Silu,
                                         densecore::models::DecoderActivation::GeluPytorchTanh};
    support.supported_rope_kinds = {densecore::models::DecoderRopeKind::Standard,
                                    densecore::models::DecoderRopeKind::Neox,
                                    densecore::models::DecoderRopeKind::Proportional};
    support.supported_prefill_logits_policies = {
        densecore::models::DecoderPrefillLogitsPolicy::FullSequence,
        densecore::models::DecoderPrefillLogitsPolicy::LastTokenEnvOptIn,
        densecore::models::DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn,
        densecore::models::DecoderPrefillLogitsPolicy::LastTokenForMoE,
    };
    support.supported_semantic_ops = {
        densecore::models::DecoderSemanticOpKind::AttentionNorm,
        densecore::models::DecoderSemanticOpKind::AttentionProjection,
        densecore::models::DecoderSemanticOpKind::HybridSSMMixer,
        densecore::models::DecoderSemanticOpKind::SharedKVRead,
        densecore::models::DecoderSemanticOpKind::SharedKVPublish,
        densecore::models::DecoderSemanticOpKind::AttentionCore,
        densecore::models::DecoderSemanticOpKind::AttentionOutputProjection,
        densecore::models::DecoderSemanticOpKind::FfnNorm,
        densecore::models::DecoderSemanticOpKind::MoERouter,
        densecore::models::DecoderSemanticOpKind::MoEExpertDispatch,
        densecore::models::DecoderSemanticOpKind::SharedDenseFfn,
        densecore::models::DecoderSemanticOpKind::DenseFfn,
        densecore::models::DecoderSemanticOpKind::FfnPostNorm,
        densecore::models::DecoderSemanticOpKind::ResidualAdd,
    };
    return support;
}

}  // namespace

densecore::TransformerGraphExecutionPlan densecore::llm::graph::ResolveExecutionPlan(const TransformerModel* model) {
    TransformerGraphExecutionPlan plan{};
    plan.resolution = models::ResolveGraphFamily(model);

    if (!model) {
        plan.route = TransformerGraphExecutionRoute::Reject;
        plan.debug_reason = "model is null";
        return plan;
    }

    std::string registry_debug;
    const auto descriptor =
        TransformerGraphRegistry::Instance().ResolveBuilderDescriptor(plan.resolution, &registry_debug);
    plan.debug_reason = "registry admission: " + registry_debug;
    if (descriptor) {
        plan.route = TransformerGraphExecutionRoute::RegistryBuilder;
        plan.registry_builder_key = descriptor->key;
        plan.selected_builder_name = descriptor->display_name;
        return plan;
    }

    const auto inline_admission = models::AdmitGraphBuilder(plan.resolution, MakeDenseCoreInlineGraphSupport());
    plan.debug_reason += " | inline admission: " + inline_admission.Summary();
    if (!inline_admission.admitted) {
        plan.route = TransformerGraphExecutionRoute::Reject;
        return plan;
    }

    switch (inline_admission.admitted_family) {
    case models::GraphFamily::DecoderDenseAttention:
        plan.route = TransformerGraphExecutionRoute::InlineDenseAttention;
        break;
    case models::GraphFamily::DecoderHybridSSM: plan.route = TransformerGraphExecutionRoute::InlineHybridSSM; break;
    case models::GraphFamily::DecoderSlidingWindowSharedKV:
        plan.route = TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV;
        break;
    case models::GraphFamily::EncoderDecoder:
    case models::GraphFamily::MultimodalProjectedDecoder:
    case models::GraphFamily::UNKNOWN:
    default:
        plan.route = TransformerGraphExecutionRoute::Reject;
        plan.debug_reason += " | unsupported inline family";
        break;
    }

    return plan;
}

std::unique_ptr<densecore::TransformerGraphBuilder>
densecore::llm::graph::InstantiateRegistryBuilder(const TransformerGraphExecutionPlan& plan,
                                                  std::string* error_reason) {
    if (plan.route != TransformerGraphExecutionRoute::RegistryBuilder) {
        if (error_reason) {
            *error_reason = "execution plan route is not RegistryBuilder";
        }
        return nullptr;
    }
    if (plan.registry_builder_key.empty()) {
        if (error_reason) {
            *error_reason = "registry builder route is missing an exact builder key";
        }
        return nullptr;
    }

    auto builder = TransformerGraphRegistry::Instance().GetBuilderExact(plan.registry_builder_key);
    if (!builder && error_reason) {
        *error_reason = "exact registry builder key '" + plan.registry_builder_key + "' is not available for execution";
    }
    return builder;
}

densecore::TransformerGraphExecutionPlan
densecore::ResolveTransformerGraphExecutionPlan(const TransformerModel* model) {
    return densecore::llm::graph::ResolveExecutionPlan(model);
}

std::unique_ptr<densecore::TransformerGraphBuilder>
densecore::InstantiateRegistryBuilderForExecutionPlan(const TransformerGraphExecutionPlan& plan,
                                                      std::string* error_reason) {
    return densecore::llm::graph::InstantiateRegistryBuilder(plan, error_reason);
}
