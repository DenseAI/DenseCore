#include "densecore/models/decoder_model_spec.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

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

bool ParseBoolEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "False") != 0;
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
        ffn.activation = is_gemma4_moe ? DecoderActivation::GeluPytorchTanh : DecoderActivation::Silu;
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

bool ShouldUsePrefillLastLogitsOnly(const DecoderModelSpec* spec, int num_seqs, int n_tokens) {
    if (!spec || num_seqs != 1 || n_tokens <= 1) {
        return false;
    }
    switch (spec->output.prefill_logits_policy) {
    case DecoderPrefillLogitsPolicy::LastTokenEnvOptIn:
        return ParseBoolEnv("DENSECORE_QWEN35_PREFILL_LAST_LOGITS_ONLY", false);
    case DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn:
        return ParseBoolEnv("DENSECORE_QWEN36_PREFILL_LAST_LOGITS_ONLY", true);
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

}  // namespace densecore::models
