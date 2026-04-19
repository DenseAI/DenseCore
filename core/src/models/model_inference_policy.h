#ifndef DENSECORE_MODELS_MODEL_INFERENCE_POLICY_H
#define DENSECORE_MODELS_MODEL_INFERENCE_POLICY_H

#include "model_types.h"

struct ggml_tensor;

namespace densecore::models {

float ResolveInputEmbeddingScale(const TransformerModel* model);
bool RequiresUnitOffsetRmsNorm(const TransformerModel* model);
bool SupportsPagedDecodeAttention(const TransformerModel* model);
float SanitizeAttentionLogitSoftcapForLoad(const TransformerModel* model, float gguf_softcap);
bool IsGemma4SlidingLayer(const TransformerModel* model, int layer_idx);
int Gemma4KVSourceLayer(const TransformerModel* model, int layer_idx);
bool IsGemma4PerLayerInputDisabled();
bool IsGemma4LayerOutputScaleDisabled();
bool IsGemma4SharedKVExplicitStateDisabled();
bool IsGemma4DecodeSpecialTransformDisabled();
bool IsGemma4VNormDisabled();
bool IsGemma4FullRopeFreqsDisabled();
bool IsGemma4QKNormDisabled();
bool IsGemma4FinalLogitSoftcapDisabled();
bool IsGemma4MoEModel(const TransformerModel* model, const TransformerLayer* layer = nullptr);
int ResolveLayerKVHeadCount(const TransformerModel* model, int layer_idx);
bool UseRuntimeKVHeadDims(const TransformerModel* model);
bool ShouldApplyQNorm(const TransformerModel* model, const ggml_tensor* q_norm);
bool ShouldApplyKNorm(const TransformerModel* model, const ggml_tensor* k_norm, bool gemma4_shared_kv_layer);
bool ShouldApplyVNorm(const TransformerModel* model, const ggml_tensor* v_tensor, bool gemma4_shared_kv_layer);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_INFERENCE_POLICY_H
