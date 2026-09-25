#pragma once

struct TransformerModel;
struct TransformerLayer;
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
namespace densecore {
class CpuBackend;
namespace models {
struct DecoderLayerSpec;
}
}  // namespace densecore

// Native expert graph construction. Workspace layout and callbacks remain private.
namespace densecore::llm::graph::detail {
ggml_tensor* TryBuildGemma4NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k, densecore::CpuBackend* numa_backend);

ggml_tensor* TryBuildQwen35NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k,
                                          const densecore::models::DecoderLayerSpec* layer_spec,
                                          densecore::CpuBackend* numa_backend);
}  // namespace densecore::llm::graph::detail
