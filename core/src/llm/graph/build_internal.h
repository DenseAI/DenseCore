#pragma once

#include "densecore/runtime/inference.h"

struct GraphAssemblyOverrides {
    ggml_tensor* input_embeddings = nullptr;
    ggml_tensor** pre_output_norm_hidden = nullptr;
    ggml_tensor** normalized_hidden = nullptr;
};

// Internal dense/hybrid assembly seam. Dispatch owns admission and registry
// identity; assembly still shares callback construction helpers with inference.
ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache, ggml_context* context,
                                             const BatchSpec& batch, bool embedding_mode, ggml_cgraph* graph,
                                             ggml_tensor** out_tokens, ggml_tensor** out_positions,
                                             const GraphAssemblyOverrides* overrides = nullptr);
