#include "densecore/models/decoder_model_spec.h"

#include <algorithm>
#include <sstream>

#include "models/model_inference_policy.h"

namespace densecore::models {
namespace {

constexpr const char* kGemma4PreMoeNormKey = "gemma4.pre_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostSharedNormKey = "gemma4.post_feedforward_layernorm_1.weight";
constexpr const char* kGemma4PostMoeNormKey = "gemma4.post_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostFfnNormKey = "gemma4.post_feedforward_layernorm.weight";

bool LayerHasSharedDenseBranch(const TransformerModel* model, const TransformerLayer& layer, bool is_gemma4_moe) {
    if (!layer.Get(model_keys::kFfnGate) || !layer.Get(model_keys::kFfnUp) || !layer.Get(model_keys::kFfnDown)) {
        return false;
    }
    if (model && model->moe_n_shared_experts > 0) {
        return true;
    }
    return is_gemma4_moe;
}

bool IsHybridSSMLayer(const TransformerModel* model, int layer_idx) {
    if (!model || !model->arch_flags.is_hybrid_ssm || layer_idx < 0) {
        return false;
    }
    if (layer_idx < static_cast<int>(model->hybrid_layer_is_ssm.size())) {
        return model->hybrid_layer_is_ssm[static_cast<size_t>(layer_idx)] != 0;
    }
    return model->ssm_full_attn_interval > 0 && ((layer_idx + 1) % model->ssm_full_attn_interval) != 0;
}

DecoderMoERouter ResolveMoERouter(const TransformerModel* model, const TransformerLayer& layer, bool is_gemma4_moe) {
    if (!layer.is_moe) {
        return DecoderMoERouter::None;
    }
    if (is_gemma4_moe) {
        return DecoderMoERouter::Gemma4SoftmaxTopK;
    }
    if (model && (model->arch_flags.is_glm_moe || model->arch_flags.is_glm_dsa ||
                  (model->variant == ModelVariant::QWEN36 && model->moe_n_group > 1 && model->moe_topk_group > 0))) {
        return DecoderMoERouter::GroupedSigmoidTopK;
    }
    return DecoderMoERouter::SoftmaxTopK;
}

DecoderRopeKind ResolveRopeKind(const TransformerModel* model, int layer_idx) {
    if (!model || !model->arch_flags.is_gemma4) {
        return DecoderRopeKind::Standard;
    }
    if (IsGemma4SlidingLayer(model, layer_idx)) {
        return DecoderRopeKind::Neox;
    }
    return DecoderRopeKind::Proportional;
}

DecoderPrefillLogitsPolicy ResolvePrefillLogitsPolicy(const TransformerModel* model) {
    if (!model) {
        return DecoderPrefillLogitsPolicy::FullSequence;
    }
    if (model->variant == ModelVariant::QWEN35) {
        return DecoderPrefillLogitsPolicy::LastTokenEnvOptIn;
    }
    if (model->variant == ModelVariant::QWEN36) {
        return DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn;
    }
    if (model->arch_flags.is_gemma4 && model->hparams.n_experts > 0) {
        return DecoderPrefillLogitsPolicy::LastTokenForMoE;
    }
    return DecoderPrefillLogitsPolicy::FullSequence;
}

void AppendLayerSemanticOps(DecoderLayerSpec* layer_spec) {
    if (!layer_spec) {
        return;
    }
    const int layer_idx = layer_spec->layer_index;
    auto add = [&](DecoderSemanticOpKind kind) { layer_spec->semantic_ops.push_back({kind, layer_idx}); };

    add(DecoderSemanticOpKind::AttentionNorm);
    if (layer_spec->attention.has_hybrid_ssm_mixer) {
        add(DecoderSemanticOpKind::HybridSSMMixer);
    } else {
        add(DecoderSemanticOpKind::AttentionProjection);
        if (layer_spec->attention.reads_shared_kv) {
            add(DecoderSemanticOpKind::SharedKVRead);
        }
        if (layer_spec->attention.publishes_shared_kv) {
            add(DecoderSemanticOpKind::SharedKVPublish);
        }
        add(DecoderSemanticOpKind::AttentionCore);
        add(DecoderSemanticOpKind::AttentionOutputProjection);
    }
    add(DecoderSemanticOpKind::FfnNorm);
    if (layer_spec->ffn.is_moe) {
        add(DecoderSemanticOpKind::MoERouter);
        add(DecoderSemanticOpKind::MoEExpertDispatch);
        if (layer_spec->ffn.has_shared_dense_branch) {
            add(DecoderSemanticOpKind::SharedDenseFfn);
        }
    } else {
        add(DecoderSemanticOpKind::DenseFfn);
    }
    if (layer_spec->ffn.has_post_ffn_norm || layer_spec->ffn.has_post_moe_norm ||
        layer_spec->ffn.has_post_shared_norm) {
        add(DecoderSemanticOpKind::FfnPostNorm);
    }
    add(DecoderSemanticOpKind::ResidualAdd);
}

void AddSpecialization(DecoderModelSpec* spec, DecoderSpecializationKind kind, int layer_idx = -1) {
    if (!spec) {
        return;
    }
    const auto exists = std::any_of(spec->specializations.begin(), spec->specializations.end(),
                                    [kind, layer_idx](const DecoderSpecialization& specialization) {
                                        return specialization.kind == kind && specialization.layer_index == layer_idx;
                                    });
    if (!exists) {
        spec->specializations.push_back({kind, layer_idx});
    }
}

DecoderRuntimeTopology ResolveRuntimeTopology(const DecoderModelSpec& spec) {
    if (spec.has_sliding_window_attention || spec.has_shared_kv) {
        return spec.has_moe ? DecoderRuntimeTopology::SlidingSharedKVMoE : DecoderRuntimeTopology::SlidingSharedKV;
    }
    if (spec.has_hybrid_ssm_mixer) {
        return spec.has_moe ? DecoderRuntimeTopology::HybridSSMMoE : DecoderRuntimeTopology::HybridSSM;
    }
    if (spec.has_moe) {
        return DecoderRuntimeTopology::DenseAttentionMoE;
    }
    return DecoderRuntimeTopology::DenseAttention;
}

void AppendModelSpecializations(DecoderModelSpec* spec, const TransformerModel* model) {
    if (!spec) {
        return;
    }
    if (spec->output.prefill_logits_policy != DecoderPrefillLogitsPolicy::FullSequence) {
        AddSpecialization(spec, DecoderSpecializationKind::PrefillLastLogits);
    }
    for (const DecoderLayerSpec& layer : spec->layers) {
        const int layer_idx = layer.layer_index;
        const DecoderAttentionSpec& attention = layer.attention;
        const DecoderFfnSpec& ffn = layer.ffn;
        if (attention.has_hybrid_ssm_mixer) {
            AddSpecialization(spec, DecoderSpecializationKind::HybridSSMMixer, layer_idx);
        }
        if (attention.is_sliding_window) {
            AddSpecialization(spec, DecoderSpecializationKind::SlidingWindowAttention, layer_idx);
        }
        if (attention.reads_shared_kv || attention.publishes_shared_kv) {
            AddSpecialization(spec, DecoderSpecializationKind::SharedKV, layer_idx);
        }
        if (model && model->hparams.n_head_kv > 0 && attention.kv_head_count > 0 &&
            attention.kv_head_count != static_cast<int>(model->hparams.n_head_kv)) {
            AddSpecialization(spec, DecoderSpecializationKind::PerLayerKVHeads, layer_idx);
        }
        if (attention.requires_q_norm) {
            AddSpecialization(spec, DecoderSpecializationKind::QNorm, layer_idx);
        }
        if (attention.requires_k_norm) {
            AddSpecialization(spec, DecoderSpecializationKind::KNorm, layer_idx);
        }
        if (attention.requires_v_norm) {
            AddSpecialization(spec, DecoderSpecializationKind::VNorm, layer_idx);
        }
        if (attention.logit_softcap > 0.0f) {
            AddSpecialization(spec, DecoderSpecializationKind::AttentionLogitSoftcap, layer_idx);
        }
        if (ffn.is_moe) {
            AddSpecialization(spec, DecoderSpecializationKind::MoE, layer_idx);
        }
        if (ffn.router == DecoderMoERouter::GroupedSigmoidTopK) {
            AddSpecialization(spec, DecoderSpecializationKind::GroupedMoERouter, layer_idx);
        }
        if (ffn.router == DecoderMoERouter::Gemma4SoftmaxTopK) {
            AddSpecialization(spec, DecoderSpecializationKind::Gemma4MoERouter, layer_idx);
        }
        if (ffn.has_shared_dense_branch) {
            AddSpecialization(spec, DecoderSpecializationKind::SharedDenseFfn, layer_idx);
        }
        if (ffn.has_down_scale_sidecar) {
            AddSpecialization(spec, DecoderSpecializationKind::MoEDownScaleSidecar, layer_idx);
        }
        if (ffn.has_post_shared_norm || ffn.has_post_moe_norm || ffn.has_post_ffn_norm) {
            AddSpecialization(spec, DecoderSpecializationKind::FfnPostNorms, layer_idx);
        }
    }
    spec->runtime_topology = ResolveRuntimeTopology(*spec);
}

}  // namespace

DecoderModelSpec BuildDecoderModelSpec(const TransformerModel* model) {
    DecoderModelSpec spec{};
    if (!model) {
        return spec;
    }

    spec.arch = model->arch;
    spec.variant = model->variant;
    spec.has_moe = model->hparams.n_experts > 0;
    spec.has_hybrid_ssm_mixer = model->arch_flags.is_hybrid_ssm;
    spec.output.prefill_logits_policy = ResolvePrefillLogitsPolicy(model);
    spec.layers.reserve(model->layers.size());

    for (size_t i = 0; i < model->layers.size(); ++i) {
        const int layer_idx = static_cast<int>(i);
        const TransformerLayer& layer = model->layers[i];
        const bool is_gemma4_moe = IsGemma4MoEModel(model, &layer);
        DecoderLayerSpec layer_spec{};
        layer_spec.layer_index = layer_idx;

        auto& attn = layer_spec.attention;
        attn.has_hybrid_ssm_mixer = IsHybridSSMLayer(model, layer_idx);
        attn.is_sliding_window = IsGemma4SlidingLayer(model, layer_idx);
        attn.kv_source_layer = Gemma4KVSourceLayer(model, layer_idx);
        attn.reads_shared_kv =
            model->arch_flags.is_gemma4 && attn.kv_source_layer >= 0 && attn.kv_source_layer != layer_idx;
        attn.publishes_shared_kv = model->arch_flags.is_gemma4 && attn.kv_source_layer == layer_idx &&
                                   layer_idx < static_cast<int>(model->gemma4_layer_kv_source.size());
        spec.has_hybrid_ssm_mixer = spec.has_hybrid_ssm_mixer || attn.has_hybrid_ssm_mixer;
        spec.has_sliding_window_attention = spec.has_sliding_window_attention || attn.is_sliding_window;
        spec.has_shared_kv = spec.has_shared_kv || attn.reads_shared_kv;
        attn.requires_q_norm = ShouldApplyQNorm(model, layer.Get(model_keys::kAttnQNorm));
        attn.requires_k_norm = ShouldApplyKNorm(model, layer.Get(model_keys::kAttnKNorm), attn.reads_shared_kv);
        attn.requires_v_norm = ShouldApplyVNorm(model, layer.Get(model_keys::kAttnVWeight), attn.reads_shared_kv);
        attn.kv_head_count = ResolveLayerKVHeadCount(model, layer_idx);
        attn.sliding_window = attn.is_sliding_window ? model->gemma4_sliding_window : -1;
        attn.attention_scale = model->hparams.f_attention_scale;
        attn.logit_softcap = model->arch_flags.is_gemma4 ? model->gemma4_attention_logit_softcapping : 0.0f;
        attn.rope_kind = ResolveRopeKind(model, layer_idx);
        attn.rope_dim = model->arch_flags.is_gemma4
                            ? (attn.is_sliding_window ? model->gemma4_rope_dim_swa : model->gemma4_rope_dim_full)
                            : static_cast<int>(model->hparams.n_rot);

        auto& ffn = layer_spec.ffn;
        ffn.is_moe = layer.is_moe;
        ffn.num_experts = static_cast<int>(layer.NumExperts());
        ffn.top_k = static_cast<int>(model->hparams.n_experts_used);
        ffn.activation = model->arch_flags.is_gemma4 ? DecoderActivation::GeluPytorchTanh : DecoderActivation::Silu;
        ffn.router = ResolveMoERouter(model, layer, is_gemma4_moe);
        ffn.has_shared_dense_branch = LayerHasSharedDenseBranch(model, layer, is_gemma4_moe);
        ffn.has_pre_moe_norm = layer.Get(kGemma4PreMoeNormKey) != nullptr;
        ffn.has_post_shared_norm = layer.Get(kGemma4PostSharedNormKey) != nullptr;
        ffn.has_post_moe_norm = layer.Get(kGemma4PostMoeNormKey) != nullptr;
        ffn.has_post_ffn_norm = layer.Get(kGemma4PostFfnNormKey) != nullptr;
        for (size_t expert = 0; expert < layer.NumExperts(); ++expert) {
            if (layer.GetExpert(expert, model_keys::kGemma4PackedDownScale)) {
                ffn.has_down_scale_sidecar = true;
                break;
            }
        }
        AppendLayerSemanticOps(&layer_spec);

        spec.layers.push_back(layer_spec);
    }

    if (spec.layers.empty()) {
        for (size_t i = 0; i < model->gemma4_layer_is_sliding.size(); ++i) {
            spec.has_sliding_window_attention =
                spec.has_sliding_window_attention || IsGemma4SlidingLayer(model, static_cast<int>(i));
        }
        for (size_t i = 0; i < model->gemma4_layer_kv_source.size(); ++i) {
            spec.has_shared_kv =
                spec.has_shared_kv || Gemma4KVSourceLayer(model, static_cast<int>(i)) != static_cast<int>(i);
        }
    }

    AppendModelSpecializations(&spec, model);
    return spec;
}

const DecoderModelSpec* GetDecoderModelSpec(const TransformerModel* model) {
    if (!model || !model->decoder_spec) {
        return nullptr;
    }
    return model->decoder_spec.get();
}

const DecoderLayerSpec* GetDecoderLayerSpec(const DecoderModelSpec* spec, int layer_idx) {
    if (!spec || layer_idx < 0 || layer_idx >= static_cast<int>(spec->layers.size())) {
        return nullptr;
    }
    return &spec->layers[static_cast<size_t>(layer_idx)];
}

const DecoderLayerSpec* ResolveDecoderLayerSpecForLayer(const TransformerModel* model, const TransformerLayer* layer) {
    const DecoderModelSpec* spec = GetDecoderModelSpec(model);
    if (!spec || !model || !layer) {
        return nullptr;
    }
    for (size_t i = 0; i < model->layers.size(); ++i) {
        if (&model->layers[i] == layer) {
            return GetDecoderLayerSpec(spec, static_cast<int>(i));
        }
    }
    return nullptr;
}

std::shared_ptr<const DecoderModelSpec> MakeDecoderModelSpec(const TransformerModel* model) {
    return std::make_shared<const DecoderModelSpec>(BuildDecoderModelSpec(model));
}

DecoderPrefillRuntimePolicy DefaultDecoderPrefillRuntimePolicy() {
    return {};
}

bool ShouldUsePrefillLastLogitsOnly(const DecoderModelSpec* spec, int num_seqs, int n_tokens,
                                    DecoderPrefillRuntimePolicy policy) {
    if (!spec || num_seqs != 1 || n_tokens <= 1) {
        return false;
    }
    switch (spec->output.prefill_logits_policy) {
    case DecoderPrefillLogitsPolicy::LastTokenEnvOptIn:
        return policy.qwen35_prefill_last_logits_only;
    case DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn:
        return policy.qwen36_prefill_last_logits_only;
    case DecoderPrefillLogitsPolicy::LastTokenForMoE: return true;
    case DecoderPrefillLogitsPolicy::FullSequence:
    default: return false;
    }
}

const char* DecoderActivationName(DecoderActivation activation) {
    switch (activation) {
    case DecoderActivation::Silu: return "silu";
    case DecoderActivation::GeluPytorchTanh: return "gelu_pytorch_tanh";
    }
    return "unknown";
}

const char* DecoderMoERouterName(DecoderMoERouter router) {
    switch (router) {
    case DecoderMoERouter::None: return "none";
    case DecoderMoERouter::SoftmaxTopK: return "softmax_top_k";
    case DecoderMoERouter::Gemma4SoftmaxTopK: return "gemma4_softmax_top_k";
    case DecoderMoERouter::GroupedSigmoidTopK: return "grouped_sigmoid_top_k";
    }
    return "unknown";
}

const char* DecoderRopeKindName(DecoderRopeKind kind) {
    switch (kind) {
    case DecoderRopeKind::Standard: return "standard";
    case DecoderRopeKind::Neox: return "neox";
    case DecoderRopeKind::Proportional: return "proportional";
    }
    return "unknown";
}

const char* DecoderPrefillLogitsPolicyName(DecoderPrefillLogitsPolicy policy) {
    switch (policy) {
    case DecoderPrefillLogitsPolicy::FullSequence: return "full_sequence";
    case DecoderPrefillLogitsPolicy::LastTokenEnvOptIn: return "last_token_env_opt_in";
    case DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn: return "last_token_env_default_on";
    case DecoderPrefillLogitsPolicy::LastTokenForMoE: return "last_token_for_moe";
    }
    return "unknown";
}

const char* DecoderRuntimeTopologyName(DecoderRuntimeTopology topology) {
    switch (topology) {
    case DecoderRuntimeTopology::Unknown: return "unknown";
    case DecoderRuntimeTopology::DenseAttention: return "dense_attention";
    case DecoderRuntimeTopology::DenseAttentionMoE: return "dense_attention_moe";
    case DecoderRuntimeTopology::HybridSSM: return "hybrid_ssm";
    case DecoderRuntimeTopology::HybridSSMMoE: return "hybrid_ssm_moe";
    case DecoderRuntimeTopology::SlidingSharedKV: return "sliding_shared_kv";
    case DecoderRuntimeTopology::SlidingSharedKVMoE: return "sliding_shared_kv_moe";
    }
    return "unknown";
}

const char* DecoderSemanticOpKindName(DecoderSemanticOpKind kind) {
    switch (kind) {
    case DecoderSemanticOpKind::AttentionNorm: return "attention_norm";
    case DecoderSemanticOpKind::AttentionProjection: return "attention_projection";
    case DecoderSemanticOpKind::HybridSSMMixer: return "hybrid_ssm_mixer";
    case DecoderSemanticOpKind::SharedKVRead: return "shared_kv_read";
    case DecoderSemanticOpKind::SharedKVPublish: return "shared_kv_publish";
    case DecoderSemanticOpKind::AttentionCore: return "attention_core";
    case DecoderSemanticOpKind::AttentionOutputProjection: return "attention_output_projection";
    case DecoderSemanticOpKind::FfnNorm: return "ffn_norm";
    case DecoderSemanticOpKind::MoERouter: return "moe_router";
    case DecoderSemanticOpKind::MoEExpertDispatch: return "moe_expert_dispatch";
    case DecoderSemanticOpKind::SharedDenseFfn: return "shared_dense_ffn";
    case DecoderSemanticOpKind::DenseFfn: return "dense_ffn";
    case DecoderSemanticOpKind::FfnPostNorm: return "ffn_post_norm";
    case DecoderSemanticOpKind::ResidualAdd: return "residual_add";
    }
    return "unknown";
}

const char* DecoderSpecializationKindName(DecoderSpecializationKind kind) {
    switch (kind) {
    case DecoderSpecializationKind::HybridSSMMixer: return "hybrid_ssm_mixer";
    case DecoderSpecializationKind::SlidingWindowAttention: return "sliding_window_attention";
    case DecoderSpecializationKind::SharedKV: return "shared_kv";
    case DecoderSpecializationKind::PerLayerKVHeads: return "per_layer_kv_heads";
    case DecoderSpecializationKind::QNorm: return "q_norm";
    case DecoderSpecializationKind::KNorm: return "k_norm";
    case DecoderSpecializationKind::VNorm: return "v_norm";
    case DecoderSpecializationKind::AttentionLogitSoftcap: return "attention_logit_softcap";
    case DecoderSpecializationKind::MoE: return "moe";
    case DecoderSpecializationKind::GroupedMoERouter: return "grouped_moe_router";
    case DecoderSpecializationKind::Gemma4MoERouter: return "gemma4_moe_router";
    case DecoderSpecializationKind::SharedDenseFfn: return "shared_dense_ffn";
    case DecoderSpecializationKind::MoEDownScaleSidecar: return "moe_down_scale_sidecar";
    case DecoderSpecializationKind::FfnPostNorms: return "ffn_post_norms";
    case DecoderSpecializationKind::PrefillLastLogits: return "prefill_last_logits";
    }
    return "unknown";
}

bool DecoderModelSpecHasSpecialization(const DecoderModelSpec& spec, DecoderSpecializationKind kind) {
    return std::any_of(spec.specializations.begin(), spec.specializations.end(),
                       [kind](const DecoderSpecialization& specialization) {
                           return specialization.kind == kind;
                       });
}

std::string FormatDecoderLayerSpec(const DecoderLayerSpec& layer) {
    std::ostringstream oss;
    oss << "layer=" << layer.layer_index << " attention={hybrid_ssm="
        << (layer.attention.has_hybrid_ssm_mixer ? "true" : "false")
        << ", sliding=" << (layer.attention.is_sliding_window ? "true" : "false")
        << ", shared_kv_read=" << (layer.attention.reads_shared_kv ? "true" : "false")
        << ", shared_kv_publish=" << (layer.attention.publishes_shared_kv ? "true" : "false")
        << ", kv_source=" << layer.attention.kv_source_layer << ", kv_heads=" << layer.attention.kv_head_count
        << ", rope=" << DecoderRopeKindName(layer.attention.rope_kind)
        << ", rope_dim=" << layer.attention.rope_dim << ", softcap=" << layer.attention.logit_softcap
        << "} ffn={moe=" << (layer.ffn.is_moe ? "true" : "false")
        << ", router=" << DecoderMoERouterName(layer.ffn.router)
        << ", activation=" << DecoderActivationName(layer.ffn.activation)
        << ", experts=" << layer.ffn.num_experts << ", top_k=" << layer.ffn.top_k
        << ", shared_dense=" << (layer.ffn.has_shared_dense_branch ? "true" : "false")
        << ", down_scale_sidecar=" << (layer.ffn.has_down_scale_sidecar ? "true" : "false")
        << ", post_norms="
        << ((layer.ffn.has_post_shared_norm || layer.ffn.has_post_moe_norm || layer.ffn.has_post_ffn_norm) ? "true"
                                                                                                           : "false")
        << "} ops=[";
    for (size_t i = 0; i < layer.semantic_ops.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << DecoderSemanticOpKindName(layer.semantic_ops[i].kind);
    }
    oss << "]";
    return oss.str();
}

std::string FormatDecoderSpecializations(const DecoderModelSpec& spec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < spec.specializations.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        const DecoderSpecialization& specialization = spec.specializations[i];
        oss << DecoderSpecializationKindName(specialization.kind);
        if (specialization.layer_index >= 0) {
            oss << "@layer" << specialization.layer_index;
        }
    }
    oss << "]";
    return oss.str();
}

std::string FormatDecoderModelSpec(const DecoderModelSpec& spec) {
    std::ostringstream oss;
    oss << "DecoderModelSpec{arch=" << static_cast<int>(spec.arch)
        << ", variant=" << static_cast<int>(spec.variant) << ", layers=" << spec.layers.size()
        << ", topology=" << DecoderRuntimeTopologyName(spec.runtime_topology)
        << ", moe=" << (spec.has_moe ? "true" : "false")
        << ", hybrid_ssm=" << (spec.has_hybrid_ssm_mixer ? "true" : "false")
        << ", sliding=" << (spec.has_sliding_window_attention ? "true" : "false")
        << ", shared_kv=" << (spec.has_shared_kv ? "true" : "false")
        << ", prefill_logits=" << DecoderPrefillLogitsPolicyName(spec.output.prefill_logits_policy)
        << ", specializations=" << FormatDecoderSpecializations(spec) << "}";
    for (const DecoderLayerSpec& layer : spec.layers) {
        oss << "\n  " << FormatDecoderLayerSpec(layer);
    }
    return oss.str();
}

}  // namespace densecore::models
