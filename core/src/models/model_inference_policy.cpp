#include "models/model_inference_policy.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace densecore::models {
namespace {

bool ParseBoolEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "False") != 0;
}

bool IsGemma4ForceDenseBaselineEnabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_FORCE_DENSE_BASELINE", false);
}

bool UseGemma4BenchQualityFallback() {
    const char* bench = std::getenv("DENSECORE_BENCH_MODE");
    if (!bench || bench[0] == '\0' || std::strcmp(bench, "0") == 0) {
        return false;
    }
    return ParseBoolEnv("DENSECORE_GEMMA4_BENCH_QUALITY_FALLBACK", false);
}

bool IsGemma4SharedKVDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_SHARED_KV", false);
}

bool IsGemma4SlidingWindowDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_SLIDING_WINDOW", false);
}

bool IsGemma4PagedDecodeEnabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_ENABLE_PAGED_DECODE", true);
}

}  // namespace

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
    if (IsGemma4ForceDenseBaselineEnabled()) {
        return false;
    }
    return IsGemma4PagedDecodeEnabled();
}

float SanitizeAttentionLogitSoftcapForLoad(const TransformerModel* model, float gguf_softcap) {
    if (model && model->arch_flags.is_gemma4) {
        // llama.cpp's Gemma4 loader does not enable attention logit softcapping;
        // only final logits are softcapped. Keep the runtime contract aligned
        // with that reference path even if older GGUF metadata exposes a
        // generic attention_logit_cap key.
        return 0.0f;
    }
    return gguf_softcap;
}

bool IsGemma4SlidingLayer(const TransformerModel* model, int layer_idx) {
    if (model && model->arch_flags.is_gemma4 &&
        (IsGemma4ForceDenseBaselineEnabled() || IsGemma4SlidingWindowDisabled() || UseGemma4BenchQualityFallback())) {
        return false;
    }
    return model && model->arch_flags.is_gemma4 && layer_idx >= 0 &&
           layer_idx < static_cast<int>(model->gemma4_layer_is_sliding.size()) &&
           model->gemma4_layer_is_sliding[static_cast<size_t>(layer_idx)] != 0;
}

int Gemma4KVSourceLayer(const TransformerModel* model, int layer_idx) {
    if (model && model->arch_flags.is_gemma4 && (IsGemma4ForceDenseBaselineEnabled() || IsGemma4SharedKVDisabled())) {
        return layer_idx;
    }
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
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_PER_LAYER_INPUT", false);
}

bool IsGemma4LayerOutputScaleDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE", false);
}

bool IsGemma4SharedKVExplicitStateDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_SHARED_KV_EXPLICIT_STATE", false);
}

bool IsGemma4DecodeSpecialTransformDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_SPECIAL_DECODE_TRANSFORMS", false) || UseGemma4BenchQualityFallback();
}

bool IsGemma4VNormDisabled() {
    // llama.cpp Gemma4 applies an unweighted RMS norm to V after reshape and
    // before RoPE. Keep that as the maintained semantic path.
    return false;
}

bool IsGemma4FullRopeFreqsDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_FULL_ROPE_FREQS", false);
}

bool IsGemma4QKNormDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_QK_NORM", false);
}

bool IsGemma4FinalLogitSoftcapDisabled() {
    return ParseBoolEnv("DENSECORE_GEMMA4_DISABLE_FINAL_LOGIT_SOFTCAP", false);
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
