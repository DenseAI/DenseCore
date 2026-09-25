#include "densecore/models/model_graph_capabilities.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/model_execution_contract.h"

namespace densecore::models {
namespace {

template <typename T, typename Predicate> bool AnyLayer(const std::vector<T>& values, Predicate&& predicate) {
    return std::any_of(values.begin(), values.end(), std::forward<Predicate>(predicate));
}

bool HasPerLayerKvHeadVariabilityFallback(const TransformerModel* model) {
    if (!model || !model->layers.empty() || model->gemma4_layer_n_head_kv.empty()) {
        return false;
    }
    return AnyLayer(model->gemma4_layer_n_head_kv,
                    [model](int32_t n_head_kv) { return n_head_kv > 0 && n_head_kv != model->hparams.n_head_kv; });
}

bool AnyLayerSpec(const DecoderModelSpec* spec, bool (*predicate)(const DecoderLayerSpec&)) {
    return spec && AnyLayer(spec->layers, predicate);
}

bool LayerReadsSharedKv(const DecoderLayerSpec& layer) {
    return layer.attention.reads_shared_kv;
}

bool LayerUsesSlidingWindow(const DecoderLayerSpec& layer) {
    return layer.attention.is_sliding_window;
}

bool LayerUsesHybridSsm(const DecoderLayerSpec& layer) {
    return layer.attention.has_hybrid_ssm_mixer;
}

bool LayerUsesMoe(const DecoderLayerSpec& layer) {
    return layer.ffn.is_moe;
}

bool LayerRequiresQNorm(const DecoderLayerSpec& layer) {
    return layer.attention.requires_q_norm;
}

bool LayerRequiresKNorm(const DecoderLayerSpec& layer) {
    return layer.attention.requires_k_norm;
}

bool LayerRequiresVNorm(const DecoderLayerSpec& layer) {
    return layer.attention.requires_v_norm;
}

bool LayerHasPerLayerKvHeadCount(const TransformerModel* model, const DecoderLayerSpec& layer) {
    return model && layer.attention.kv_head_count > 0 && layer.attention.kv_head_count != model->hparams.n_head_kv;
}

bool LayerRequiresSpecialResidualScaling(const DecoderLayerSpec& layer) {
    return layer.ffn.has_shared_dense_branch || layer.ffn.has_pre_moe_norm || layer.ffn.has_post_shared_norm ||
           layer.ffn.has_post_moe_norm || layer.ffn.has_post_ffn_norm;
}

DecoderRuntimeTopology InferDecoderRuntimeTopologyFromCapabilities(const ModelGraphCapabilities& capabilities) {
    if (capabilities.has_sliding_window_attention || capabilities.has_shared_kv_source) {
        return capabilities.has_moe ? DecoderRuntimeTopology::SlidingSharedKVMoE
                                    : DecoderRuntimeTopology::SlidingSharedKV;
    }
    if (capabilities.has_hybrid_ssm_mixer || capabilities.has_lfm2_shortconv_mixer) {
        return capabilities.has_moe ? DecoderRuntimeTopology::HybridSSMMoE : DecoderRuntimeTopology::HybridSSM;
    }
    if (capabilities.has_moe) {
        return DecoderRuntimeTopology::DenseAttentionMoE;
    }
    return DecoderRuntimeTopology::DenseAttention;
}

std::vector<GraphFamily> CandidateFamilies(const GraphFamilyResolution& resolution) {
    std::vector<GraphFamily> candidates;
    candidates.reserve(1 + resolution.fallback_chain.size());
    if (resolution.preferred_family != GraphFamily::UNKNOWN) {
        candidates.push_back(resolution.preferred_family);
    }
    for (GraphFamily family : resolution.fallback_chain) {
        if (family != GraphFamily::UNKNOWN &&
            std::find(candidates.begin(), candidates.end(), family) == candidates.end()) {
            candidates.push_back(family);
        }
    }
    return candidates;
}

bool SupportsFamily(GraphFamily family, const GraphBuilderSupport& support) {
    return std::find(support.supported_families.begin(), support.supported_families.end(), family) !=
           support.supported_families.end();
}

void AddReason(bool condition, const std::string& reason, std::vector<std::string>* out) {
    if (condition && out) {
        out->push_back(reason);
    }
}

template <typename T> void AddUnique(std::vector<T>* values, T value) {
    if (!values) {
        return;
    }
    if (std::find(values->begin(), values->end(), value) == values->end()) {
        values->push_back(value);
    }
}

template <typename T> bool ContainsAll(const std::vector<T>& supported, const std::vector<T>& required) {
    for (const T value : required) {
        if (std::find(supported.begin(), supported.end(), value) == supported.end()) {
            return false;
        }
    }
    return true;
}

template <typename T, typename NameFn> std::string JoinNames(const std::vector<T>& values, NameFn name_fn) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << name_fn(values[i]);
    }
    oss << "]";
    return oss.str();
}

}  // namespace

const char* GraphTopologyName(GraphTopology topology) {
    switch (topology) {
    case GraphTopology::DECODER_ONLY: return "decoder_only";
    case GraphTopology::ENCODER_DECODER: return "encoder_decoder";
    case GraphTopology::MULTIMODAL_PROJECTED_DECODER: return "multimodal_projected_decoder";
    case GraphTopology::UNKNOWN:
    default: return "unknown";
    }
}

const char* GraphFamilyName(GraphFamily family) {
    switch (family) {
    case GraphFamily::DecoderDenseAttention: return "DecoderDenseAttention";
    case GraphFamily::DecoderSlidingWindowSharedKV: return "DecoderSlidingWindowSharedKV";
    case GraphFamily::DecoderHybridSSM: return "DecoderHybridSSM";
    case GraphFamily::EncoderDecoder: return "EncoderDecoder";
    case GraphFamily::MultimodalProjectedDecoder: return "MultimodalProjectedDecoder";
    case GraphFamily::UNKNOWN:
    default: return "UNKNOWN";
    }
}

std::string GraphAdmissionResult::Summary() const {
    if (admitted) {
        return std::string("admitted for family ") + GraphFamilyName(admitted_family);
    }
    std::ostringstream oss;
    for (size_t i = 0; i < rejection_reasons.size(); ++i) {
        if (i != 0) {
            oss << "; ";
        }
        oss << rejection_reasons[i];
    }
    return oss.str();
}

ModelGraphCapabilities ResolveModelGraphCapabilities(const TransformerModel* model) {
    ModelGraphCapabilities capabilities{};
    if (!model) {
        return capabilities;
    }

    const auto& descriptor = DescribeModel(model);
    ModelArchFlags effective_flags = descriptor.default_flags;
    effective_flags.requires_q_norm = effective_flags.requires_q_norm || model->arch_flags.requires_q_norm;
    effective_flags.requires_k_norm = effective_flags.requires_k_norm || model->arch_flags.requires_k_norm;
    effective_flags.is_gemma4 = effective_flags.is_gemma4 || model->arch_flags.is_gemma4;
    effective_flags.is_lfm2_shortconv = effective_flags.is_lfm2_shortconv || model->arch_flags.is_lfm2_shortconv;
    std::shared_ptr<const DecoderModelSpec> scratch_spec;
    const DecoderModelSpec* decoder_spec = GetDecoderModelSpec(model);
    if (!decoder_spec) {
        scratch_spec = MakeDecoderModelSpec(model);
        decoder_spec = scratch_spec.get();
    }

    capabilities.arch = model->arch;
    capabilities.variant = descriptor.variant;
    switch (model->arch) {
    case ModelArch::WHISPER: capabilities.topology = GraphTopology::ENCODER_DECODER; break;
    case ModelArch::LLAVA:
    case ModelArch::QWEN_VL:
        // TODO: Promote this from a coarse topology marker to explicit projector
        // metadata once TransformerModel exposes the projection semantics needed
        // for admission. Until then, fail closed instead of pretending these are
        // plain decoder-only checkpoints.
        capabilities.topology = GraphTopology::MULTIMODAL_PROJECTED_DECODER;
        break;
    case ModelArch::VIT:
    case ModelArch::CLIP_VISION:
    case ModelArch::SIGLIP:
    case ModelArch::UNKNOWN: capabilities.topology = GraphTopology::UNKNOWN; break;
    default: capabilities.topology = GraphTopology::DECODER_ONLY; break;
    }
    capabilities.has_dense_attention = capabilities.topology == GraphTopology::DECODER_ONLY;
    capabilities.has_sliding_window_attention = decoder_spec && (decoder_spec->has_sliding_window_attention ||
                                                                 AnyLayerSpec(decoder_spec, LayerUsesSlidingWindow));
    capabilities.has_shared_kv_source =
        decoder_spec && (decoder_spec->has_shared_kv || AnyLayerSpec(decoder_spec, LayerReadsSharedKv));
    capabilities.has_hybrid_ssm_mixer =
        decoder_spec && (decoder_spec->has_hybrid_ssm_mixer || AnyLayerSpec(decoder_spec, LayerUsesHybridSsm));
    capabilities.has_lfm2_shortconv_mixer =
        capabilities.topology == GraphTopology::DECODER_ONLY && effective_flags.is_lfm2_shortconv;
    capabilities.has_moe = decoder_spec && (decoder_spec->has_moe || AnyLayerSpec(decoder_spec, LayerUsesMoe));
    capabilities.requires_q_norm = AnyLayerSpec(decoder_spec, LayerRequiresQNorm) ||
                                   (decoder_spec && decoder_spec->layers.empty() && effective_flags.requires_q_norm);
    capabilities.requires_k_norm = AnyLayerSpec(decoder_spec, LayerRequiresKNorm) ||
                                   (decoder_spec && decoder_spec->layers.empty() && effective_flags.requires_k_norm);
    capabilities.requires_v_norm = AnyLayerSpec(decoder_spec, LayerRequiresVNorm) ||
                                   (decoder_spec && decoder_spec->layers.empty() && effective_flags.is_gemma4);
    capabilities.has_per_layer_kv_head_variability =
        (decoder_spec &&
         AnyLayer(decoder_spec->layers,
                  [model](const DecoderLayerSpec& layer) { return LayerHasPerLayerKvHeadCount(model, layer); })) ||
        HasPerLayerKvHeadVariabilityFallback(model);
    capabilities.decoder_runtime_topology =
        decoder_spec ? decoder_spec->runtime_topology : DecoderRuntimeTopology::Unknown;
    capabilities.requires_special_attention_mask = capabilities.has_sliding_window_attention;
    capabilities.requires_special_residual_scaling =
        (decoder_spec && AnyLayerSpec(decoder_spec, LayerRequiresSpecialResidualScaling)) ||
        (decoder_spec && decoder_spec->layers.empty() && effective_flags.is_gemma4);
    if (decoder_spec) {
        AddUnique(&capabilities.required_prefill_logits_policies, decoder_spec->output.prefill_logits_policy);
        for (const DecoderLayerSpec& layer : decoder_spec->layers) {
            AddUnique(&capabilities.required_rope_kinds, layer.attention.rope_kind);
            for (const DecoderSemanticOp& op : layer.semantic_ops) {
                AddUnique(&capabilities.required_semantic_ops, op.kind);
            }
            if (layer.ffn.is_moe) {
                AddUnique(&capabilities.required_moe_routers, layer.ffn.router);
                AddUnique(&capabilities.required_ffn_activations, layer.ffn.activation);
            } else {
                AddUnique(&capabilities.required_ffn_activations, layer.ffn.activation);
            }
            capabilities.requires_shared_dense_ffn =
                capabilities.requires_shared_dense_ffn || layer.ffn.has_shared_dense_branch;
            capabilities.requires_moe_down_scale_sidecar =
                capabilities.requires_moe_down_scale_sidecar || layer.ffn.has_down_scale_sidecar;
            capabilities.requires_ffn_post_norms = capabilities.requires_ffn_post_norms ||
                                                   layer.ffn.has_post_shared_norm || layer.ffn.has_post_moe_norm ||
                                                   layer.ffn.has_post_ffn_norm;
        }
    }

    // Multimodal decoder projection is reserved for an explicit future load
    // path that exposes projector semantics in TransformerModel metadata.
    capabilities.has_multimodal_projection = capabilities.topology == GraphTopology::MULTIMODAL_PROJECTED_DECODER;

    const auto execution_contract = BuildModelExecutionContract(model);
    capabilities.requires_fallback_free_fast_path =
        ModelExecutionContractRequiresFallbackFreeFastPath(execution_contract);
    capabilities.requires_native_moe_fast_path = ModelExecutionContractRequiresNativeMoEFastPath(execution_contract);
    capabilities.requires_decode_graph_runtime_rebind =
        ModelExecutionContractRequiresDecodeGraphRuntimeRebind(execution_contract);
    capabilities.native_moe_max_direct_tokens = ModelExecutionContractNativeMoEMaxDirectTokens(execution_contract);

    return capabilities;
}

GraphFamilyResolution ResolveGraphFamily(const TransformerModel* model) {
    GraphFamilyResolution resolution{};
    resolution.capabilities = ResolveModelGraphCapabilities(model);

    const auto& capabilities = resolution.capabilities;
    switch (capabilities.topology) {
    case GraphTopology::ENCODER_DECODER:
        resolution.preferred_family = GraphFamily::EncoderDecoder;
        resolution.fail_closed = true;
        return resolution;
    case GraphTopology::MULTIMODAL_PROJECTED_DECODER:
        resolution.preferred_family = GraphFamily::MultimodalProjectedDecoder;
        resolution.fail_closed = true;
        return resolution;
    case GraphTopology::DECODER_ONLY: break;
    case GraphTopology::UNKNOWN:
    default:
        resolution.preferred_family = GraphFamily::UNKNOWN;
        resolution.fail_closed = true;
        return resolution;
    }

    const DecoderRuntimeTopology decoder_topology =
        capabilities.has_lfm2_shortconv_mixer ||
                capabilities.decoder_runtime_topology == DecoderRuntimeTopology::Unknown
            ? InferDecoderRuntimeTopologyFromCapabilities(capabilities)
            : capabilities.decoder_runtime_topology;
    switch (decoder_topology) {
    case DecoderRuntimeTopology::HybridSSM:
    case DecoderRuntimeTopology::HybridSSMMoE:
        resolution.preferred_family = GraphFamily::DecoderHybridSSM;
        resolution.fallback_chain = {GraphFamily::DecoderDenseAttention};
        resolution.fail_closed = true;
        return resolution;
    case DecoderRuntimeTopology::SlidingSharedKV:
    case DecoderRuntimeTopology::SlidingSharedKVMoE:
        resolution.preferred_family = GraphFamily::DecoderSlidingWindowSharedKV;
        resolution.fallback_chain = {GraphFamily::DecoderDenseAttention};
        resolution.fail_closed = true;
        return resolution;
    case DecoderRuntimeTopology::DenseAttention:
    case DecoderRuntimeTopology::DenseAttentionMoE:
    case DecoderRuntimeTopology::Unknown:
    default: break;
    }

    if (capabilities.has_hybrid_ssm_mixer || capabilities.has_lfm2_shortconv_mixer) {
        resolution.preferred_family = GraphFamily::DecoderHybridSSM;
        resolution.fallback_chain = {GraphFamily::DecoderDenseAttention};
        resolution.fail_closed = true;
        return resolution;
    }

    if (capabilities.has_sliding_window_attention || capabilities.has_shared_kv_source ||
        capabilities.has_per_layer_kv_head_variability || capabilities.requires_v_norm ||
        capabilities.requires_special_attention_mask || capabilities.requires_special_residual_scaling) {
        resolution.preferred_family = GraphFamily::DecoderSlidingWindowSharedKV;
        resolution.fallback_chain = {GraphFamily::DecoderDenseAttention};
        resolution.fail_closed = true;
        return resolution;
    }

    resolution.preferred_family = GraphFamily::DecoderDenseAttention;
    resolution.fail_closed = true;
    return resolution;
}

GraphAdmissionResult AdmitGraphBuilder(const GraphFamilyResolution& resolution, const GraphBuilderSupport& support) {
    GraphAdmissionResult result{};
    const auto candidates = CandidateFamilies(resolution);
    const auto& capabilities = resolution.capabilities;

    for (GraphFamily family : candidates) {
        std::vector<std::string> reasons;
        AddReason(!SupportsFamily(family, support),
                  std::string(support.builder_name) + " does not implement graph family " + GraphFamilyName(family),
                  &reasons);
        AddReason(capabilities.has_sliding_window_attention && !support.supports_sliding_window_attention,
                  "requires sliding-window attention semantics", &reasons);
        AddReason(capabilities.has_shared_kv_source && !support.supports_shared_kv_source,
                  "requires shared-KV source semantics", &reasons);
        AddReason(capabilities.has_hybrid_ssm_mixer && !support.supports_hybrid_ssm_mixer,
                  "requires hybrid SSM mixer semantics", &reasons);
        AddReason(capabilities.has_lfm2_shortconv_mixer && !support.supports_lfm2_shortconv_mixer,
                  "requires LFM2 short-conv mixer semantics", &reasons);
        AddReason(capabilities.has_moe && !support.supports_moe, "requires MoE routing semantics", &reasons);
        AddReason(capabilities.requires_q_norm && !support.supports_q_norm, "requires Q norm before attention",
                  &reasons);
        AddReason(capabilities.requires_k_norm && !support.supports_k_norm, "requires K norm before attention",
                  &reasons);
        AddReason(capabilities.requires_v_norm && !support.supports_v_norm, "requires V norm support", &reasons);
        AddReason(capabilities.has_per_layer_kv_head_variability && !support.supports_per_layer_kv_head_variability,
                  "requires per-layer KV head variability", &reasons);
        AddReason(capabilities.requires_special_attention_mask && !support.supports_special_attention_mask,
                  "requires special attention mask semantics", &reasons);
        AddReason(capabilities.requires_special_residual_scaling && !support.supports_special_residual_scaling,
                  "requires special residual/input/output scaling", &reasons);
        AddReason(capabilities.has_multimodal_projection && !support.supports_multimodal_projection,
                  "requires multimodal projection semantics", &reasons);
        AddReason(capabilities.requires_shared_dense_ffn && !support.supports_shared_dense_ffn,
                  "requires shared dense FFN branch semantics", &reasons);
        AddReason(capabilities.requires_moe_down_scale_sidecar && !support.supports_moe_down_scale_sidecar,
                  "requires MoE down-projection scale sidecar semantics", &reasons);
        AddReason(capabilities.requires_ffn_post_norms && !support.supports_ffn_post_norms,
                  "requires FFN post-norm ordering semantics", &reasons);
        AddReason(!ContainsAll(support.supported_moe_routers, capabilities.required_moe_routers),
                  "requires MoE router kinds " + JoinNames(capabilities.required_moe_routers, DecoderMoERouterName),
                  &reasons);
        AddReason(!ContainsAll(support.supported_ffn_activations, capabilities.required_ffn_activations),
                  "requires FFN activation kinds " +
                      JoinNames(capabilities.required_ffn_activations, DecoderActivationName),
                  &reasons);
        AddReason(!ContainsAll(support.supported_rope_kinds, capabilities.required_rope_kinds),
                  "requires RoPE kinds " + JoinNames(capabilities.required_rope_kinds, DecoderRopeKindName), &reasons);
        AddReason(
            !ContainsAll(support.supported_prefill_logits_policies, capabilities.required_prefill_logits_policies),
            "requires prefill logits policies " +
                JoinNames(capabilities.required_prefill_logits_policies, DecoderPrefillLogitsPolicyName),
            &reasons);
        AddReason(!ContainsAll(support.supported_semantic_ops, capabilities.required_semantic_ops),
                  "requires semantic ops " + JoinNames(capabilities.required_semantic_ops, DecoderSemanticOpKindName),
                  &reasons);

        if (reasons.empty()) {
            result.admitted = true;
            result.admitted_family = family;
            return result;
        }

        std::ostringstream oss;
        oss << "family " << GraphFamilyName(family) << " rejected: ";
        for (size_t i = 0; i < reasons.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << reasons[i];
        }
        result.rejection_reasons.push_back(oss.str());
    }

    return result;
}

GraphBuilderSupport MakeDenseDecoderGenericSupport(const char* builder_name) {
    GraphBuilderSupport support{};
    support.builder_name = builder_name;
    support.supported_families = {GraphFamily::DecoderDenseAttention};
    support.supported_ffn_activations = {DecoderActivation::Silu};
    support.supported_rope_kinds = {DecoderRopeKind::Standard};
    support.supported_prefill_logits_policies = {DecoderPrefillLogitsPolicy::FullSequence};
    support.supported_semantic_ops = {
        DecoderSemanticOpKind::AttentionNorm, DecoderSemanticOpKind::AttentionProjection,
        DecoderSemanticOpKind::AttentionCore, DecoderSemanticOpKind::AttentionOutputProjection,
        DecoderSemanticOpKind::FfnNorm,       DecoderSemanticOpKind::DenseFfn,
        DecoderSemanticOpKind::ResidualAdd,
    };
    return support;
}

std::string FormatModelGraphCapabilities(const ModelGraphCapabilities& capabilities) {
    std::ostringstream oss;
    oss << "arch=" << static_cast<int>(capabilities.arch) << ", variant=" << static_cast<int>(capabilities.variant)
        << ", topology=" << GraphTopologyName(capabilities.topology)
        << ", decoder_topology=" << DecoderRuntimeTopologyName(capabilities.decoder_runtime_topology)
        << ", dense_attention=" << (capabilities.has_dense_attention ? "true" : "false")
        << ", sliding_window=" << (capabilities.has_sliding_window_attention ? "true" : "false")
        << ", shared_kv=" << (capabilities.has_shared_kv_source ? "true" : "false")
        << ", hybrid_ssm=" << (capabilities.has_hybrid_ssm_mixer ? "true" : "false")
        << ", lfm2_shortconv=" << (capabilities.has_lfm2_shortconv_mixer ? "true" : "false")
        << ", moe=" << (capabilities.has_moe ? "true" : "false")
        << ", q_norm=" << (capabilities.requires_q_norm ? "true" : "false")
        << ", k_norm=" << (capabilities.requires_k_norm ? "true" : "false")
        << ", v_norm=" << (capabilities.requires_v_norm ? "true" : "false")
        << ", per_layer_kv_heads=" << (capabilities.has_per_layer_kv_head_variability ? "true" : "false")
        << ", special_mask=" << (capabilities.requires_special_attention_mask ? "true" : "false")
        << ", special_scaling=" << (capabilities.requires_special_residual_scaling ? "true" : "false")
        << ", multimodal_projection=" << (capabilities.has_multimodal_projection ? "true" : "false")
        << ", shared_dense_ffn=" << (capabilities.requires_shared_dense_ffn ? "true" : "false")
        << ", moe_down_scale_sidecar=" << (capabilities.requires_moe_down_scale_sidecar ? "true" : "false")
        << ", ffn_post_norms=" << (capabilities.requires_ffn_post_norms ? "true" : "false")
        << ", fallback_free_fast_path=" << (capabilities.requires_fallback_free_fast_path ? "true" : "false")
        << ", native_moe_fast_path=" << (capabilities.requires_native_moe_fast_path ? "true" : "false")
        << ", decode_graph_runtime_rebind=" << (capabilities.requires_decode_graph_runtime_rebind ? "true" : "false")
        << ", native_moe_max_direct_tokens=" << capabilities.native_moe_max_direct_tokens
        << ", moe_routers=" << JoinNames(capabilities.required_moe_routers, DecoderMoERouterName)
        << ", ffn_activations=" << JoinNames(capabilities.required_ffn_activations, DecoderActivationName)
        << ", rope_kinds=" << JoinNames(capabilities.required_rope_kinds, DecoderRopeKindName)
        << ", prefill_logits_policies="
        << JoinNames(capabilities.required_prefill_logits_policies, DecoderPrefillLogitsPolicyName)
        << ", semantic_ops=" << JoinNames(capabilities.required_semantic_ops, DecoderSemanticOpKindName);
    return oss.str();
}

std::string FormatGraphFamilyResolution(const GraphFamilyResolution& resolution) {
    std::ostringstream oss;
    oss << "preferred_family=" << GraphFamilyName(resolution.preferred_family) << ", fallback_chain=[";
    for (size_t i = 0; i < resolution.fallback_chain.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << GraphFamilyName(resolution.fallback_chain[i]);
    }
    oss << "], fail_closed=" << (resolution.fail_closed ? "true" : "false");
    return oss.str();
}

}  // namespace densecore::models
