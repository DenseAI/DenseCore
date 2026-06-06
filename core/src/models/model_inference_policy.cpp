#include "models/model_inference_policy.h"

#include <cmath>
#include <iostream>

namespace densecore::models {

float ResolveInputEmbeddingScale(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return 1.0f;
    }
    return std::sqrt(static_cast<float>(model->hparams.n_embd));
}

bool RequiresUnitOffsetRmsNorm(const TransformerModel* model) {
    return model && model->arch_flags.uses_unit_offset_rms_norm;
}

bool SupportsPagedDecodeAttention(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4) {
        return true;
    }
    return true;
}

float SanitizeAttentionLogitSoftcapForLoad(const TransformerModel* model, float gguf_softcap) {
    if (model && model->arch_flags.is_gemma4 && gguf_softcap <= 0.0f) {
        return 50.0f;
    }
    return gguf_softcap;
}

bool IsGemma4SlidingLayer(const TransformerModel* model, int layer_idx) {
    return model && model->arch_flags.is_gemma4 && layer_idx >= 0 &&
           layer_idx < static_cast<int>(model->gemma4_layer_is_sliding.size()) &&
           model->gemma4_layer_is_sliding[static_cast<size_t>(layer_idx)] != 0;
}

int Gemma4KVSourceLayer(const TransformerModel* model, int layer_idx) {
    if (!model || !model->arch_flags.is_gemma4 || layer_idx < 0) {
        return layer_idx;
    }
    if (layer_idx >= static_cast<int>(model->gemma4_layer_kv_source.size())) {
        return layer_idx;
    }
    const int source = model->gemma4_layer_kv_source[static_cast<size_t>(layer_idx)];
    return source >= 0 ? source : layer_idx;
}

bool IsGemma4PerLayerInputDisabled() {
    return false;
}

bool IsGemma4LayerOutputScaleDisabled() {
    return false;
}

bool IsGemma4SharedKVExplicitStateDisabled() {
    return false;
}

bool IsGemma4DecodeSpecialTransformDisabled() {
    return false;
}

bool IsGemma4VNormDisabled() {
    // llama.cpp Gemma4 applies an unweighted RMS norm to V after reshape and
    // before RoPE. Keep that as the maintained semantic path.
    return false;
}

bool IsGemma4FullRopeFreqsDisabled() {
    return false;
}

bool IsGemma4QKNormDisabled() {
    return false;
}

bool IsGemma4FinalLogitSoftcapDisabled() {
    return false;
}

bool IsGemma4MoEModel(const TransformerModel* model, const TransformerLayer* layer) {
    if (!model || model->arch != ModelArch::GEMMA || model->hparams.n_experts == 0) {
        return false;
    }
    return !layer || layer->Get("gemma4.router.scale") || layer->Get("gemma4.router.per_expert_scale") || layer->is_moe;
}

int ResolveLayerKVHeadCount(const TransformerModel* model, int layer_idx) {
    if (!model) {
        return 0;
    }
    if (!model->gemma4_layer_n_head_kv.empty() && layer_idx >= 0 &&
        layer_idx < static_cast<int>(model->gemma4_layer_n_head_kv.size())) {
        return static_cast<int>(model->gemma4_layer_n_head_kv[static_cast<size_t>(layer_idx)]);
    }
    return model->hparams.n_head_kv;
}

bool UseRuntimeKVHeadDims(const TransformerModel* model) {
    return model && (model->arch_flags.is_gemma4 || !model->gemma4_layer_n_head_kv.empty());
}

bool ShouldApplyQNorm(const TransformerModel* model, const ggml_tensor* q_norm) {
    return model && model->arch_flags.requires_q_norm && q_norm &&
           !(model->arch_flags.is_gemma4 && IsGemma4QKNormDisabled());
}

bool ShouldApplyKNorm(const TransformerModel* model, const ggml_tensor* k_norm, bool gemma4_shared_kv_layer) {
    return model && model->arch_flags.requires_k_norm && k_norm && !gemma4_shared_kv_layer &&
           !(model->arch_flags.is_gemma4 && IsGemma4QKNormDisabled());
}

bool ShouldApplyVNorm(const TransformerModel* model, const ggml_tensor* v_tensor, bool gemma4_shared_kv_layer) {
    return model && model->arch_flags.is_gemma4 && v_tensor && !gemma4_shared_kv_layer && !IsGemma4VNormDisabled();
}

}  // namespace densecore::models
