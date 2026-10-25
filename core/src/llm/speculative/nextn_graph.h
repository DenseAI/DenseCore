#pragma once
#include "densecore/runtime/inference.h"

namespace densecore::llm::speculative {
// A non-owning executable one-layer view: all weights remain owned by source.
// Keep this view alive until every graph built from it has finished executing.
// Its KV cache is independent of the target trunk and must not be prefix-cached.
class NextNGraph {
public:
    explicit NextNGraph(TransformerModel& source);
    TransformerModel& Model() { return model_; }
    ggml_tensor* Build(ggml_context* ctx, PagedKVCache* cache, const BatchSpec& batch,
                       ggml_tensor* target_hidden, ggml_cgraph* graph,
                       ggml_tensor** next_hidden = nullptr);
private:
    TransformerModel& source_;
    TransformerModel model_;
    ggml_tensor* eh_proj_ = nullptr;
    ggml_tensor* enorm_ = nullptr;
    ggml_tensor* hnorm_ = nullptr;
};
}  // namespace densecore::llm::speculative
