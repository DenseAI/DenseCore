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

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_types.h"

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
    DecoderRuntimeTopology decoder_runtime_topology = DecoderRuntimeTopology::Unknown;
    bool has_dense_attention = false;
    bool has_sliding_window_attention = false;
    bool has_shared_kv_source = false;
    bool has_hybrid_ssm_mixer = false;
    bool has_lfm2_shortconv_mixer = false;
    bool has_moe = false;
    bool requires_q_norm = false;
    bool requires_k_norm = false;
    bool requires_v_norm = false;
    bool has_per_layer_kv_head_variability = false;
    bool requires_special_attention_mask = false;
    bool requires_special_residual_scaling = false;
    bool has_multimodal_projection = false;
    bool requires_shared_dense_ffn = false;
    bool requires_moe_down_scale_sidecar = false;
    bool requires_ffn_post_norms = false;
    bool requires_fallback_free_fast_path = false;
    bool requires_native_moe_fast_path = false;
    bool requires_decode_graph_runtime_rebind = false;
    int64_t native_moe_max_direct_tokens = 0;
    std::vector<DecoderMoERouter> required_moe_routers;
    std::vector<DecoderActivation> required_ffn_activations;
    std::vector<DecoderRopeKind> required_rope_kinds;
    std::vector<DecoderPrefillLogitsPolicy> required_prefill_logits_policies;
    std::vector<DecoderSemanticOpKind> required_semantic_ops;
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
    bool supports_lfm2_shortconv_mixer = false;
    bool supports_moe = false;
    bool supports_q_norm = false;
    bool supports_k_norm = false;
    bool supports_v_norm = false;
    bool supports_per_layer_kv_head_variability = false;
    bool supports_special_attention_mask = false;
    bool supports_special_residual_scaling = false;
    bool supports_multimodal_projection = false;
    bool supports_shared_dense_ffn = false;
    bool supports_moe_down_scale_sidecar = false;
    bool supports_ffn_post_norms = false;
    std::vector<DecoderMoERouter> supported_moe_routers;
    std::vector<DecoderActivation> supported_ffn_activations;
    std::vector<DecoderRopeKind> supported_rope_kinds;
    std::vector<DecoderPrefillLogitsPolicy> supported_prefill_logits_policies;
    std::vector<DecoderSemanticOpKind> supported_semantic_ops;
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
