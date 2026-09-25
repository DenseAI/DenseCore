#pragma once
#include "densecore/runtime/inference.h"

namespace densecore::llm::models::bert {
ggml_tensor* BuildEmbeddingGraph(TransformerModel* model, ggml_context* context, const BatchSpec& batch,
                                 ggml_cgraph* graph, ggml_tensor** out_tokens, ggml_tensor** out_positions);
}  // namespace densecore::llm::models::bert
