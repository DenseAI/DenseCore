#pragma once

#include "densecore/runtime/inference.h"

// Internal dense/hybrid assembly seam. Dispatch owns admission and registry
// identity; assembly still shares callback construction helpers with inference.
ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache, ggml_context* context,
                                             const BatchSpec& batch, bool embedding_mode, ggml_cgraph* graph,
                                             ggml_tensor** out_tokens, ggml_tensor** out_positions);
