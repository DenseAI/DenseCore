#pragma once

#include "densecore/runtime/inference.h"
#include "llm/speculative/ssm_checkpoints.h"

struct GraphAssemblyOverrides {
    ggml_tensor* input_embeddings = nullptr;
    bool retain_all_logits = false;
    // Stable per-SSM ordinal bindings, or null for every ordinary graph.
    const std::vector<densecore::llm::speculative::SSMCheckpointView*>* ssm_checkpoints = nullptr;
    // NextN consumes the target's output-normalized hidden rows, before any
    // last-token/per-sequence compaction. See pinned qwen35 target h_nextn.
    ggml_tensor** full_normalized_hidden = nullptr;
    ggml_tensor** normalized_hidden = nullptr;
};

// Internal dense/hybrid assembly seam. Dispatch owns admission and registry
// identity; assembly still shares callback construction helpers with inference.
ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache, ggml_context* context,
                                             const BatchSpec& batch, bool embedding_mode, ggml_cgraph* graph,
                                             ggml_tensor** out_tokens, ggml_tensor** out_positions,
                                             const GraphAssemblyOverrides* overrides = nullptr);
