#include "densecore/models/model_execution_contract.h"

#include <algorithm>
#include <memory>
#include <sstream>

namespace densecore::models {
namespace {

constexpr int64_t kDefaultNativeMoEFastPathMaxDirectTokens = 4096;

bool HasTensor(const TransformerLayer& layer, const char* key) {
    return layer.Get(key) != nullptr;
}

bool HasAnyTensor(const TransformerLayer& layer, std::initializer_list<const char*> keys) {
    return std::any_of(keys.begin(), keys.end(), [&layer](const char* key) { return HasTensor(layer, key); });
}

bool HasAnyExpertTensor(const TransformerLayer& layer, const char* key) {
    for (std::size_t i = 0; i < layer.NumExperts(); ++i) {
        if (layer.GetExpert(i, key)) {
            return true;
        }
    }
    return false;
}

ExecutionMoEExpertLayoutKind ResolveMoEExpertLayout(const TransformerLayer& layer, const DecoderLayerSpec* spec) {
    const bool is_moe = (spec && spec->ffn.is_moe) || layer.is_moe;
    if (!is_moe) {
        return ExecutionMoEExpertLayoutKind::None;
    }

    const bool has_gemma4_down_scale = HasAnyExpertTensor(layer, model_keys::kGemma4PackedDownScale);
    const bool has_packed_gate_up = HasAnyTensor(layer, {"ffn_gate_up_exps.weight", "ffn_gate_up_exps",
                                                        "experts.gate_up_proj.weight"});
    const bool has_packed_down =
        HasAnyTensor(layer, {"ffn_down_exps.weight", "ffn_down_exps", "experts.down_proj.weight"});
    if (has_packed_gate_up && has_packed_down && has_gemma4_down_scale) {
        return ExecutionMoEExpertLayoutKind::Gemma4PackedWithDownScale;
    }
    if (has_packed_gate_up && has_packed_down) {
        return ExecutionMoEExpertLayoutKind::PackedGateUpDown;
    }
    if (HasAnyTensor(layer, {"ffn_gate_exps.weight", "ffn_gate_exps"}) &&
        HasAnyTensor(layer, {"ffn_up_exps.weight", "ffn_up_exps"}) && has_packed_down) {
        return ExecutionMoEExpertLayoutKind::PackedSeparateGateUpDown;
    }
    if (layer.NumExperts() > 0) {
        return ExecutionMoEExpertLayoutKind::SeparateExpertMaps;
    }
    return ExecutionMoEExpertLayoutKind::Unknown;
}

ExecutionTensorOwnership ResolveTensorOwnership(const TransformerModel* model, const ModelExecutionLayerContract& layer,
                                                const TransformerLayer* transformer_layer) {
    if (!model || !transformer_layer) {
        return ExecutionTensorOwnership::None;
    }
    if (layer.has_hybrid_ssm_mixer && layer.ssm_ordinal >= 0 &&
        layer.ssm_ordinal < static_cast<int>(model->ssm_layer_states.size())) {
        return ExecutionTensorOwnership::LoaderCanonicalBuffer;
    }
    if (layer.has_lfm2_shortconv_mixer && layer.conv_ordinal >= 0 &&
        layer.conv_ordinal < static_cast<int>(model->lfm2_conv_weight_f32.size()) &&
        !model->lfm2_conv_weight_f32[static_cast<std::size_t>(layer.conv_ordinal)].empty()) {
        return ExecutionTensorOwnership::LoaderCanonicalBuffer;
    }
    if (layer.has_lfm2_shortconv_mixer && transformer_layer->Get(model_keys::kShortConvConv)) {
        return ExecutionTensorOwnership::RawBoundTensor;
    }
    if (layer.has_moe) {
        return ExecutionTensorOwnership::RuntimeOwnedView;
    }
    return ExecutionTensorOwnership::None;
}

ExecutionRuntimeStateShape HybridSSMRuntimeStateShape(const TransformerModel* model) {
    ExecutionRuntimeStateShape shape{};
    shape.kind = ExecutionRuntimeStateKind::HybridSSM;
    if (!model) {
        return shape;
    }
    shape.conv_channels = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
    shape.kernel_size = model->ssm_conv_kernel;
    shape.n_heads = model->ssm_time_step_rank;
    shape.head_dim = model->ssm_time_step_rank > 0 ? model->ssm_inner_size / model->ssm_time_step_rank : 0;
    shape.state_size = model->ssm_state_size;
    shape.expected_conv_state_elements =
        TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(shape.conv_channels, shape.kernel_size);
    shape.expected_ssm_state_elements = TransformerModel::SSMSequenceRuntimeState::ExpectedStateElements(
        shape.n_heads, shape.head_dim, shape.state_size);
    return shape;
}

ExecutionRuntimeStateShape LFM2ShortConvRuntimeStateShape(const TransformerModel* model) {
    ExecutionRuntimeStateShape shape{};
    shape.kind = ExecutionRuntimeStateKind::LFM2ShortConv;
    if (!model) {
        return shape;
    }
    shape.conv_channels = static_cast<int>(model->hparams.n_embd);
    shape.kernel_size = model->lfm2_conv_kernel;
    shape.expected_conv_state_elements =
        TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(shape.conv_channels, shape.kernel_size);
    shape.expected_ssm_state_elements = 0;
    return shape;
}

void AddRebindDescriptor(ModelExecutionContract* contract, ModelExecutionLayerContract* layer,
                         ExecutionCustomOpRebindKind op_kind, ExecutionRuntimeStateKind state_kind, int state_ordinal) {
    if (!contract || !layer) {
        return;
    }
    ExecutionCustomOpRebindDescriptor descriptor{};
    descriptor.layer_index = layer->layer_index;
    descriptor.state_ordinal = state_ordinal;
    descriptor.state_kind = state_kind;
    descriptor.op_kind = op_kind;
    layer->rebind_descriptors.push_back(descriptor);
    contract->rebind_descriptors.push_back(descriptor);
}

void AddRejection(ModelExecutionContract* contract, const char* reason) {
    if (!contract || !reason) {
        return;
    }
    contract->valid = false;
    contract->rejection_reasons.push_back(reason);
}

void AddForbiddenFastPath(ModelExecutionContract* contract, const char* reason) {
    if (!contract || !reason) {
        return;
    }
    contract->forbidden_fast_path_reasons.push_back(reason);
}

void AddRequiredFastPathCounter(ModelExecutionContract* contract, const char* counter) {
    if (!contract || !counter) {
        return;
    }
    contract->required_fast_path_counters.push_back(counter);
}

bool IsQwenHybridSSMContract(const ModelExecutionContract& contract) {
    return contract.has_hybrid_ssm_mixer &&
           (contract.variant == ModelVariant::QWEN35 || contract.variant == ModelVariant::QWEN36);
}

bool IsLFM2ShortConvMoEContract(const ModelExecutionContract& contract) {
    return contract.variant == ModelVariant::LFM2MOE && contract.has_lfm2_shortconv_mixer && contract.has_moe;
}

bool IsGemma4MoEContract(const ModelExecutionContract& contract) {
    return contract.variant == ModelVariant::GEMMA4 && contract.has_moe;
}

ExecutionFastPathClass ResolveFastPathClass(const TransformerModel* model, const ModelExecutionContract& contract) {
    if (!model) {
        return ExecutionFastPathClass::None;
    }
    if (IsQwenHybridSSMContract(contract)) {
        return contract.has_moe ? ExecutionFastPathClass::QwenHybridSSMMoE
                                : ExecutionFastPathClass::QwenHybridSSMDense;
    }
    if ((contract.variant == ModelVariant::QWEN35 || contract.variant == ModelVariant::QWEN36) &&
        !contract.has_moe && !contract.has_hybrid_ssm_mixer) {
        return ExecutionFastPathClass::QwenDense;
    }
    if (IsGemma4MoEContract(contract)) {
        return ExecutionFastPathClass::Gemma4MoE;
    }
    if (IsLFM2ShortConvMoEContract(contract)) {
        return ExecutionFastPathClass::LFM2ShortConvMoE;
    }
    return ExecutionFastPathClass::None;
}

}  // namespace

ModelExecutionContract BuildModelExecutionContract(const TransformerModel* model) {
    ModelExecutionContract contract{};
    if (!model) {
        contract.valid = false;
        contract.decode_graph_cache_static_safe = false;
        contract.rejection_reasons.push_back("null_model");
        return contract;
    }

    const DecoderModelSpec* spec = GetDecoderModelSpec(model);
    std::shared_ptr<const DecoderModelSpec> scratch_spec;
    if (!spec) {
        scratch_spec = MakeDecoderModelSpec(model);
        spec = scratch_spec.get();
    }

    contract.arch = model->arch;
    contract.variant = model->variant;
    contract.decoder_runtime_topology = spec ? spec->runtime_topology : DecoderRuntimeTopology::Unknown;
    const bool synthetic_stateful_model =
        model->layers.empty() && (model->arch_flags.is_hybrid_ssm || model->arch_flags.is_lfm2_shortconv);
    const std::size_t layer_count = synthetic_stateful_model ? 1 : model->layers.size();
    contract.layers.reserve(layer_count);

    int ssm_ordinal = 0;
    int conv_ordinal = 0;
    for (std::size_t i = 0; i < layer_count; ++i) {
        const int layer_idx = static_cast<int>(i);
        const TransformerLayer* transformer_layer = i < model->layers.size() ? &model->layers[i] : nullptr;
        const DecoderLayerSpec* layer_spec = GetDecoderLayerSpec(spec, layer_idx);

        ModelExecutionLayerContract layer{};
        layer.layer_index = layer_idx;
        layer.has_lfm2_shortconv_mixer =
            synthetic_stateful_model ? model->arch_flags.is_lfm2_shortconv : model->IsLFM2ConvLayer(layer_idx);
        layer.has_hybrid_ssm_mixer =
            !layer.has_lfm2_shortconv_mixer &&
            (synthetic_stateful_model ? model->arch_flags.is_hybrid_ssm
                                      : (model->IsHybridSSMLayer(layer_idx) ||
                                         (layer_spec && layer_spec->attention.has_hybrid_ssm_mixer)));
        layer.has_moe = (transformer_layer && transformer_layer->is_moe) || (layer_spec && layer_spec->ffn.is_moe);

        if (layer.has_hybrid_ssm_mixer) {
            layer.ssm_ordinal = ssm_ordinal++;
            layer.runtime_state_shape = HybridSSMRuntimeStateShape(model);
            if (layer.ssm_ordinal >= 0 && layer.ssm_ordinal < static_cast<int>(model->ssm_layer_states.size())) {
                layer.ssm_norm_layout = model->ssm_layer_states[static_cast<std::size_t>(layer.ssm_ordinal)].norm_layout;
            }
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::HybridSSMConv1D,
                                ExecutionRuntimeStateKind::HybridSSM, layer.ssm_ordinal);
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::HybridSSMDelta,
                                ExecutionRuntimeStateKind::HybridSSM, layer.ssm_ordinal);
        }

        if (layer.has_lfm2_shortconv_mixer) {
            const int resolved_ordinal = model->LFM2ConvOrdinal(layer_idx);
            layer.conv_ordinal = resolved_ordinal >= 0 ? resolved_ordinal : conv_ordinal;
            conv_ordinal = std::max(conv_ordinal, layer.conv_ordinal + 1);
            layer.runtime_state_shape = LFM2ShortConvRuntimeStateShape(model);
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::LFM2ShortConv,
                                ExecutionRuntimeStateKind::LFM2ShortConv, layer.conv_ordinal);
        }

        if (layer.has_moe) {
            layer.moe_router = layer_spec ? layer_spec->ffn.router : DecoderMoERouter::None;
            layer.moe_top_k = layer_spec ? layer_spec->ffn.top_k : static_cast<int>(model->hparams.n_experts_used);
            layer.moe_num_experts =
                layer_spec ? layer_spec->ffn.num_experts
                           : static_cast<int>(transformer_layer ? transformer_layer->NumExperts() : 0);
            layer.moe_expert_layout =
                transformer_layer ? ResolveMoEExpertLayout(*transformer_layer, layer_spec)
                                  : ExecutionMoEExpertLayoutKind::Unknown;
            layer.has_packed_moe_sidecar =
                (layer_spec && layer_spec->ffn.has_down_scale_sidecar) ||
                (transformer_layer && HasAnyExpertTensor(*transformer_layer, model_keys::kGemma4PackedDownScale));
            if (layer.moe_expert_layout == ExecutionMoEExpertLayoutKind::Unknown) {
                AddRejection(&contract, "moe_expert_layout_unknown");
            }
        }

        layer.tensor_ownership = ResolveTensorOwnership(model, layer, transformer_layer);
        contract.has_hybrid_ssm_mixer = contract.has_hybrid_ssm_mixer || layer.has_hybrid_ssm_mixer;
        contract.has_lfm2_shortconv_mixer = contract.has_lfm2_shortconv_mixer || layer.has_lfm2_shortconv_mixer;
        contract.has_moe = contract.has_moe || layer.has_moe;
        contract.layers.push_back(std::move(layer));
    }

    contract.has_moe = contract.has_moe || model->hparams.n_experts > 0;
    contract.has_stateful_custom_ops = contract.has_hybrid_ssm_mixer || contract.has_lfm2_shortconv_mixer;
    contract.fast_path_class = ResolveFastPathClass(model, contract);
    contract.requires_fallback_free_fast_path = contract.fast_path_class != ExecutionFastPathClass::None;
    contract.requires_native_moe_fast_path =
        contract.fast_path_class == ExecutionFastPathClass::QwenHybridSSMMoE ||
        contract.fast_path_class == ExecutionFastPathClass::LFM2ShortConvMoE;
    if (contract.requires_fallback_free_fast_path) {
        AddRequiredFastPathCounter(&contract, "target_no_ggml_path");
    }
    if (contract.requires_native_moe_fast_path) {
        AddRequiredFastPathCounter(&contract, "native_moe_fast_w1w3_used_ops");
        AddRequiredFastPathCounter(&contract, "native_moe_fast_w2_used_ops");
    }
    if (contract.has_hybrid_ssm_mixer) {
        AddRequiredFastPathCounter(&contract, "ssm_conv1d_calls");
        AddRequiredFastPathCounter(&contract, "ssm_delta_calls");
    }
    if (contract.fast_path_class == ExecutionFastPathClass::LFM2ShortConvMoE) {
        AddRequiredFastPathCounter(&contract, "lfm2_shortconv_sequence_fast_used_ops");
    }
    if (contract.fast_path_class == ExecutionFastPathClass::Gemma4MoE) {
        AddRequiredFastPathCounter(&contract, "gemma4_prefill_maintained_fast_ops");
        AddRequiredFastPathCounter(&contract, "gemma4_decode_maintained_fast_ops");
        AddForbiddenFastPath(&contract, "gemma4_arm_native_moe_prefill_quality_failed");
    }
    if (contract.requires_native_moe_fast_path) {
        contract.native_moe_max_direct_tokens = kDefaultNativeMoEFastPathMaxDirectTokens;
    }
    contract.requires_decode_graph_runtime_rebind = !contract.rebind_descriptors.empty();
    if (contract.has_lfm2_shortconv_mixer) {
        contract.decode_graph_cache_static_safe = true;
    } else if (contract.has_hybrid_ssm_mixer && !IsQwenHybridSSMContract(contract)) {
        contract.decode_graph_cache_static_safe = false;
    } else {
        contract.decode_graph_cache_static_safe = true;
    }

    if (contract.has_hybrid_ssm_mixer && model->ssm_time_step_rank <= 0) {
        AddRejection(&contract, "invalid_hybrid_ssm_time_step_rank");
        contract.decode_graph_cache_static_safe = false;
    }
    if (contract.has_lfm2_shortconv_mixer && model->lfm2_conv_kernel <= 0) {
        AddRejection(&contract, "invalid_lfm2_conv_kernel");
        contract.decode_graph_cache_static_safe = false;
    }
    if (!contract.valid) {
        contract.decode_graph_cache_static_safe = false;
    }
    return contract;
}

bool ModelExecutionContractAllowsDecodeGraphCache(const ModelExecutionContract& contract) {
    return contract.valid && contract.decode_graph_cache_static_safe;
}

bool ModelExecutionContractRequiresDecodeGraphRuntimeRebind(const ModelExecutionContract& contract) {
    return contract.valid && contract.requires_decode_graph_runtime_rebind;
}

bool ModelExecutionContractRequiresFallbackFreeFastPath(const ModelExecutionContract& contract) {
    return contract.valid && contract.requires_fallback_free_fast_path;
}

bool ModelExecutionContractRequiresNativeMoEFastPath(const ModelExecutionContract& contract) {
    return contract.valid && contract.requires_native_moe_fast_path;
}

int64_t ModelExecutionContractNativeMoEMaxDirectTokens(const ModelExecutionContract& contract) {
    return contract.valid ? contract.native_moe_max_direct_tokens : 0;
}

const char* ExecutionRuntimeStateKindName(ExecutionRuntimeStateKind kind) {
    switch (kind) {
    case ExecutionRuntimeStateKind::None: return "none";
    case ExecutionRuntimeStateKind::HybridSSM: return "hybrid_ssm";
    case ExecutionRuntimeStateKind::LFM2ShortConv: return "lfm2_shortconv";
    }
    return "unknown";
}

const char* ExecutionTensorOwnershipName(ExecutionTensorOwnership ownership) {
    switch (ownership) {
    case ExecutionTensorOwnership::None: return "none";
    case ExecutionTensorOwnership::RawBoundTensor: return "raw_bound_tensor";
    case ExecutionTensorOwnership::LoaderCanonicalBuffer: return "loader_canonical_buffer";
    case ExecutionTensorOwnership::RuntimeOwnedView: return "runtime_owned_view";
    }
    return "unknown";
}

const char* ExecutionMoEExpertLayoutKindName(ExecutionMoEExpertLayoutKind kind) {
    switch (kind) {
    case ExecutionMoEExpertLayoutKind::None: return "none";
    case ExecutionMoEExpertLayoutKind::SeparateExpertMaps: return "separate_expert_maps";
    case ExecutionMoEExpertLayoutKind::PackedGateUpDown: return "packed_gate_up_down";
    case ExecutionMoEExpertLayoutKind::PackedSeparateGateUpDown: return "packed_separate_gate_up_down";
    case ExecutionMoEExpertLayoutKind::Gemma4PackedWithDownScale: return "gemma4_packed_with_down_scale";
    case ExecutionMoEExpertLayoutKind::Unknown: return "unknown";
    }
    return "unknown";
}

const char* ExecutionCustomOpRebindKindName(ExecutionCustomOpRebindKind kind) {
    switch (kind) {
    case ExecutionCustomOpRebindKind::None: return "none";
    case ExecutionCustomOpRebindKind::HybridSSMConv1D: return "hybrid_ssm_conv1d";
    case ExecutionCustomOpRebindKind::HybridSSMDelta: return "hybrid_ssm_delta";
    case ExecutionCustomOpRebindKind::LFM2ShortConv: return "lfm2_shortconv";
    }
    return "unknown";
}

const char* ExecutionFastPathClassName(ExecutionFastPathClass kind) {
    switch (kind) {
    case ExecutionFastPathClass::None: return "none";
    case ExecutionFastPathClass::QwenDense: return "qwen_dense";
    case ExecutionFastPathClass::QwenHybridSSMDense: return "qwen_hybrid_ssm_dense";
    case ExecutionFastPathClass::QwenHybridSSMMoE: return "qwen_hybrid_ssm_moe";
    case ExecutionFastPathClass::Gemma4MoE: return "gemma4_moe";
    case ExecutionFastPathClass::LFM2ShortConvMoE: return "lfm2_shortconv_moe";
    }
    return "unknown";
}

std::string FormatModelExecutionContract(const ModelExecutionContract& contract) {
    std::ostringstream oss;
    oss << "ModelExecutionContract{variant=" << static_cast<int>(contract.variant)
        << ",topology=" << DecoderRuntimeTopologyName(contract.decoder_runtime_topology)
        << ",valid=" << (contract.valid ? "true" : "false")
        << ",has_moe=" << (contract.has_moe ? "true" : "false")
        << ",has_hybrid_ssm=" << (contract.has_hybrid_ssm_mixer ? "true" : "false")
        << ",has_lfm2_shortconv=" << (contract.has_lfm2_shortconv_mixer ? "true" : "false")
        << ",fast_path_class=" << ExecutionFastPathClassName(contract.fast_path_class)
        << ",requires_fallback_free_fast_path="
        << (contract.requires_fallback_free_fast_path ? "true" : "false")
        << ",requires_native_moe_fast_path=" << (contract.requires_native_moe_fast_path ? "true" : "false")
        << ",native_moe_max_direct_tokens=" << contract.native_moe_max_direct_tokens
        << ",decode_cache_static_safe=" << (contract.decode_graph_cache_static_safe ? "true" : "false")
        << ",requires_rebind=" << (contract.requires_decode_graph_runtime_rebind ? "true" : "false")
        << ",required_fast_path_counters=[";
    for (std::size_t i = 0; i < contract.required_fast_path_counters.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << contract.required_fast_path_counters[i];
    }
    oss << "]"
        << ",forbidden_fast_paths=[";
    for (std::size_t i = 0; i < contract.forbidden_fast_path_reasons.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << contract.forbidden_fast_path_reasons[i];
    }
    oss << "]"
        << ",layers=[";
    for (std::size_t i = 0; i < contract.layers.size(); ++i) {
        const auto& layer = contract.layers[i];
        if (i != 0) {
            oss << ";";
        }
        oss << "{idx=" << layer.layer_index << ",ssm=" << (layer.has_hybrid_ssm_mixer ? "true" : "false")
            << ",shortconv=" << (layer.has_lfm2_shortconv_mixer ? "true" : "false")
            << ",moe=" << (layer.has_moe ? "true" : "false") << ",ssm_ordinal=" << layer.ssm_ordinal
            << ",conv_ordinal=" << layer.conv_ordinal
            << ",state=" << ExecutionRuntimeStateKindName(layer.runtime_state_shape.kind)
            << ",router=" << DecoderMoERouterName(layer.moe_router)
            << ",expert_layout=" << ExecutionMoEExpertLayoutKindName(layer.moe_expert_layout)
            << ",ownership=" << ExecutionTensorOwnershipName(layer.tensor_ownership) << "}";
    }
    oss << "],rebinds=[";
    for (std::size_t i = 0; i < contract.rebind_descriptors.size(); ++i) {
        const auto& descriptor = contract.rebind_descriptors[i];
        if (i != 0) {
            oss << ",";
        }
        oss << ExecutionCustomOpRebindKindName(descriptor.op_kind) << "@layer" << descriptor.layer_index;
    }
    oss << "]}";
    return oss.str();
}

}  // namespace densecore::models
