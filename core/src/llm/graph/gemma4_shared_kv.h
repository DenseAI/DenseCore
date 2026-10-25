#pragma once

#include "densecore/exceptions.h"
#include "models/model_inference_policy.h"
#include "runtime/inference_types_internal.h"

#include <sstream>

namespace densecore::llm::graph::detail {

// Shared layers consume the already normalized/rotated source tensors. They
// neither own K/V projections nor apply normalization or RoPE to them again.
inline const Gemma4SharedKVState& RequireGemma4SharedKVSource(
    const TransformerModel* model, int layer, int source, const Gemma4SharedKVState* state, int query_tokens) {
    const auto reject = [&](const char* reason) -> void {
        std::ostringstream message;
        message << "Gemma4 shared-KV " << reason << " layer=" << layer << " source=" << source;
        throw densecore::InvalidArgumentException(message.str());
    };
    if (!model || !model->arch_flags.is_gemma4 || layer < 0 ||
        layer >= static_cast<int>(model->hparams.n_layer) || source < 0 || source >= layer || query_tokens <= 0) {
        reject("source index is invalid");
    }
    if (model->gemma4_layer_is_sliding.size() <= static_cast<size_t>(layer) ||
        model->gemma4_layer_kv_source.size() <= static_cast<size_t>(layer) ||
        model->gemma4_layer_kv_source[source] != source || model->gemma4_layer_kv_source[layer] != source ||
        densecore::models::IsGemma4SlidingLayer(model, source) != densecore::models::IsGemma4SlidingLayer(model, layer)) {
        reject("source attention type or mapping is invalid");
    }
    if (!state || state->source_layer != source || !state->k || !state->v) {
        reject("source tensors have not been published");
    }
    const bool sliding = densecore::models::IsGemma4SlidingLayer(model, layer);
    const int heads = densecore::models::ResolveLayerKVHeadCount(model, layer);
    int key_dim = sliding ? model->gemma4_key_length_swa : model->gemma4_key_length_full;
    int value_dim = sliding ? model->gemma4_value_length_swa : model->gemma4_value_length_full;
    if (key_dim <= 0) key_dim = model->hparams.n_embd_head_k;
    if (value_dim <= 0) value_dim = model->hparams.n_embd_head_v;
    if (state->k->type != GGML_TYPE_F32 || state->v->type != GGML_TYPE_F32) {
        reject("source tensors must contain F32 projected states");
    }
    if (heads <= 0 || key_dim <= 0 || value_dim <= 0 || state->k->ne[0] != key_dim ||
        state->v->ne[0] != value_dim || state->k->ne[1] != heads || state->v->ne[1] != heads ||
        state->k->ne[2] < query_tokens || state->v->ne[2] != state->k->ne[2] ||
        state->k->ne[3] != 1 || state->v->ne[3] != 1) {
        reject("source tensor shape is inconsistent");
    }
    return *state;
}

}  // namespace densecore::llm::graph::detail
