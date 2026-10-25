#pragma once
#include "densecore/runtime/inference.h"

namespace densecore::llm::speculative {
// A non-owning executable one-layer view: all weights remain owned by source.
// Keep this view alive until every graph built from it has finished executing.
// Its KV cache is independent of the target trunk and must not be prefix-cached.
class NextNGraph {
public:
    // Conservative per-active-head bound for borrowed expert descriptors/maps
    // and exact small BF16 router-to-F32
    // buffers; excludes borrowed packed expert bytes/global repack cache.
    // Account in the shared pre-KV MTP reservation, never a second lease here.
    static uint64_t RequiredExpertMetadataBytes(const TransformerModel& source);
    explicit NextNGraph(TransformerModel& source);
    ~NextNGraph() = default;
    NextNGraph(const NextNGraph&) = delete;
    NextNGraph& operator=(const NextNGraph&) = delete;
    TransformerModel& Model() { return model_; }
    ggml_tensor* Build(ggml_context* ctx, PagedKVCache* cache, const BatchSpec& batch,
                       ggml_tensor* target_hidden, ggml_cgraph* graph,
                       ggml_tensor** next_hidden = nullptr);
private:
    TransformerModel& source_;
    // FP32 conversion is restricted to small BF16 router/scalar-gate weights.
    // Buffers outlive model_'s borrowed GGML tensor headers.
    std::vector<float> router_fp32_;
    std::vector<float> shared_gate_fp32_;
    TransformerModel model_;
    ggml_tensor* eh_proj_ = nullptr;
    ggml_tensor* enorm_ = nullptr;
    ggml_tensor* hnorm_ = nullptr;
};
}  // namespace densecore::llm::speculative
