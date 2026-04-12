/**
 * @file model_graph_capabilities.h
 * @brief Graph-family resolution and admission for DenseCore model graphs
 *
 * llm_universal is no longer a sufficient abstraction because modern decoder
 * families now diverge in graph semantics, not only in tensor names or weight
 * shapes. Qwen3.5 hybrid SSM mixers and Gemma4 sliding/shared-KV behavior are
 * the current proof points, but the same split will keep growing.
 *
 * DenseCore now resolves model identity into explicit graph capabilities, then
 * selects a graph family plus an inspectable fallback chain. Builders admit or
 * reject those capabilities explicitly, so unsupported semantics fail closed
 * instead of silently entering a numerically wrong generic path.
 */

#ifndef DENSECORE_MODELS_MODEL_GRAPH_CAPABILITIES_H
#define DENSECORE_MODELS_MODEL_GRAPH_CAPABILITIES_H

#include <cstdint>
#include <string>
#include <vector>

#include "model_types.h"

namespace densecore::models {

enum class GraphTopology : uint8_t {
    UNKNOWN = 0,
    DECODER_ONLY,
    ENCODER_DECODER,
    MULTIMODAL_PROJECTED_DECODER,
};

enum class GraphFamily : uint8_t {
    UNKNOWN = 0,
    DecoderDenseAttention,
    DecoderSlidingWindowSharedKV,
    DecoderHybridSSM,
    EncoderDecoder,
    MultimodalProjectedDecoder,
};

struct ModelGraphCapabilities {
    ModelArch arch = ModelArch::UNKNOWN;
    ModelVariant variant = ModelVariant::UNKNOWN;
    GraphTopology topology = GraphTopology::UNKNOWN;
    bool has_dense_attention = false;
    bool has_sliding_window_attention = false;
    bool has_shared_kv_source = false;
    bool has_hybrid_ssm_mixer = false;
    bool has_moe = false;
    bool requires_q_norm = false;
    bool requires_k_norm = false;
    bool requires_v_norm = false;
    bool has_per_layer_kv_head_variability = false;
    bool requires_special_attention_mask = false;
    bool requires_special_residual_scaling = false;
    bool has_multimodal_projection = false;
};

struct GraphFamilyResolution {
    ModelGraphCapabilities capabilities{};
    GraphFamily preferred_family = GraphFamily::UNKNOWN;
    std::vector<GraphFamily> fallback_chain;
    bool fail_closed = true;
};

struct GraphBuilderSupport {
    const char* builder_name = "unknown";
    std::vector<GraphFamily> supported_families;
    bool supports_sliding_window_attention = false;
    bool supports_shared_kv_source = false;
    bool supports_hybrid_ssm_mixer = false;
    bool supports_moe = false;
    bool supports_q_norm = false;
    bool supports_k_norm = false;
    bool supports_v_norm = false;
    bool supports_per_layer_kv_head_variability = false;
    bool supports_special_attention_mask = false;
    bool supports_special_residual_scaling = false;
    bool supports_multimodal_projection = false;
};

struct GraphAdmissionResult {
    bool admitted = false;
    GraphFamily admitted_family = GraphFamily::UNKNOWN;
    std::vector<std::string> rejection_reasons;

    std::string Summary() const;
};

ModelGraphCapabilities ResolveModelGraphCapabilities(const TransformerModel* model);
GraphFamilyResolution ResolveGraphFamily(const TransformerModel* model);
GraphAdmissionResult AdmitGraphBuilder(const GraphFamilyResolution& resolution, const GraphBuilderSupport& support);
GraphBuilderSupport MakeDenseDecoderGenericSupport(const char* builder_name);

const char* GraphTopologyName(GraphTopology topology);
const char* GraphFamilyName(GraphFamily family);
std::string FormatModelGraphCapabilities(const ModelGraphCapabilities& capabilities);
std::string FormatGraphFamilyResolution(const GraphFamilyResolution& resolution);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_GRAPH_CAPABILITIES_H
