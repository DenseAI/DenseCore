#include "densecore/models/model_execution_contract.h"

#include <algorithm>
#include <cstring>
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
    const bool has_packed_gate_up =
        HasAnyTensor(layer, {"ffn_gate_up_exps.weight", "ffn_gate_up_exps", "experts.gate_up_proj.weight"});
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

void AddRequiredFastPathCounter(ModelExecutionContract* contract, ExecutionFastPathCounterKind counter) {
    if (!contract || counter == ExecutionFastPathCounterKind::Unknown) {
        return;
    }
    contract->required_fast_path_counter_kinds.push_back(counter);
    contract->required_fast_path_counters.push_back(ExecutionFastPathCounterKindName(counter));
}

const char* SafeGgmlTypeName(ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT ? ggml_type_name(type) : "unknown";
}

std::string ShapeLabel(int64_t m, int64_t n, int64_t k) {
    std::ostringstream oss;
    oss << "m" << (m > 0 ? std::to_string(m) : "?") << "xn" << (n > 0 ? std::to_string(n) : "?") << "xk"
        << (k > 0 ? std::to_string(k) : "?");
    return oss.str();
}

std::string MissingKernelReason(const ModelSemanticGraphNode& node,
                                const densecore::runtime::KernelResolution& resolution) {
    std::ostringstream oss;
    oss << "missing_kernel:" << densecore::runtime::DenseCoreSemanticOpName(node.semantic_op) << "/"
        << SafeGgmlTypeName(node.raw_gguf_type) << "/" << ShapeLabel(node.m, node.n, node.k) << "/"
        << densecore::runtime::DenseCoreHostBackendName(resolution.host_backend);
    return oss.str();
}

bool IsFallbackFreeTargetModel(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    if (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36 ||
        model->variant == ModelVariant::QWEN38 ||
        model->variant == ModelVariant::GEMMA4 || model->variant == ModelVariant::LFM2MOE) {
        return true;
    }
    if ((model->arch == ModelArch::QWEN35 && model->arch_flags.is_hybrid_ssm) || model->arch_flags.is_gemma4 ||
        model->arch_flags.is_lfm2_shortconv) {
        return true;
    }
    return false;
}

ExecutionQuantLayoutKind ResolveCanonicalLayout(const TransformerModel* model, const ggml_tensor* tensor,
                                                densecore::runtime::DenseCoreTensorRole role) {
    if (!tensor) {
        return ExecutionQuantLayoutKind::Unknown;
    }
    if (model && model->int4_weight_bindings.find(tensor) != model->int4_weight_bindings.end()) {
        return ExecutionQuantLayoutKind::PackedDenseCore;
    }
    if (role == densecore::runtime::DenseCoreTensorRole::ShortConvIn ||
        role == densecore::runtime::DenseCoreTensorRole::ShortConvOut) {
        return ExecutionQuantLayoutKind::LoaderCanonical;
    }
    return ExecutionQuantLayoutKind::RawGGUF;
}

bool TensorNameMatchesKey(const char* tensor_name, const std::string& tensor_key) {
    if (!tensor_name || !tensor_name[0] || tensor_key.empty()) {
        return false;
    }
    return std::strcmp(tensor_name, tensor_key.c_str()) == 0 || std::strstr(tensor_name, tensor_key.c_str()) != nullptr;
}

densecore::runtime::DenseCoreTensorRole FindLoaderTensorRole(const TransformerModel* model, const ggml_tensor* tensor) {
    using densecore::runtime::DenseCoreTensorRole;
    if (!model || !tensor) {
        return DenseCoreTensorRole::Unknown;
    }
    DenseCoreTensorRole role = model->GetTensorRole(tensor);
    if (role != DenseCoreTensorRole::Unknown) {
        return role;
    }
    for (const TransformerLayer& layer : model->layers) {
        role = layer.GetTensorRole(tensor);
        if (role != DenseCoreTensorRole::Unknown) {
            return role;
        }
    }
    return DenseCoreTensorRole::Unknown;
}

void FinalizeTensorRequirement(const TransformerModel* model, ModelTensorExecutionRequirement* requirement) {
    if (!requirement) {
        return;
    }
    requirement->prefill_kernel = densecore::runtime::ResolveMaintainedKernelFamilyForTensorRole(
        requirement->raw_gguf_type, densecore::runtime::DenseCoreMatmulPhase::Prefill, requirement->tensor_role);
    requirement->decode_kernel = densecore::runtime::ResolveMaintainedKernelFamilyForTensorRole(
        requirement->raw_gguf_type, densecore::runtime::DenseCoreMatmulPhase::Decode, requirement->tensor_role);
    const bool role_has_maintained_target_contract =
        requirement->semantic_op == densecore::runtime::DenseCoreSemanticOp::HybridSsmMixer ||
        requirement->semantic_op == densecore::runtime::DenseCoreSemanticOp::MoeRouter ||
        requirement->semantic_op == densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch ||
        requirement->semantic_op == densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer ||
        requirement->semantic_op == densecore::runtime::DenseCoreSemanticOp::LmHead;
    requirement->fallback_policy = IsFallbackFreeTargetModel(model) && role_has_maintained_target_contract
                                       ? densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget
                                       : densecore::runtime::DenseCoreFallbackPolicyKind::CompatibilityFallback;
}

ModelTensorExecutionRequirement MakeTensorRequirement(const TransformerModel* model, int layer_index,
                                                      const std::string& tensor_key, const ggml_tensor* tensor,
                                                      densecore::runtime::DenseCoreTensorRole role) {
    ModelTensorExecutionRequirement requirement{};
    requirement.layer_index = layer_index;
    requirement.tensor_key = tensor_key;
    const auto loader_role = FindLoaderTensorRole(model, tensor);
    requirement.tensor_role = loader_role != densecore::runtime::DenseCoreTensorRole::Unknown ? loader_role : role;
    requirement.semantic_op = densecore::runtime::ResolveDenseCoreSemanticOp(model, requirement.tensor_role);
    requirement.raw_gguf_type = tensor ? tensor->type : GGML_TYPE_COUNT;
    if (tensor) {
        requirement.tensor_cols = tensor->ne[0];
        requirement.tensor_rows = tensor->ne[1];
    }
    requirement.canonical_layout = ResolveCanonicalLayout(model, tensor, requirement.tensor_role);
    requirement.repacked_layout = ExecutionQuantLayoutKind::Unknown;
    if (model && tensor) {
        if (model->prepared_weights.qwen36_ssm_q8_prefill_amx_aliases.find(tensor) != model->prepared_weights.qwen36_ssm_q8_prefill_amx_aliases.end()) {
            requirement.repacked_layout = ExecutionQuantLayoutKind::AmxPrefillAlias;
        } else if (model->prepared_weights.cpu_repack_aliases.find(tensor) != model->prepared_weights.cpu_repack_aliases.end() ||
                   model->prepared_weights.cpu_decode_repack_aliases.find(tensor) != model->prepared_weights.cpu_decode_repack_aliases.end()) {
            requirement.repacked_layout = ExecutionQuantLayoutKind::CpuRepacked;
        }
    }
    FinalizeTensorRequirement(model, &requirement);
    return requirement;
}

void AddTensorRequirement(ModelExecutionContract* contract, ModelExecutionLayerContract* layer,
                          const TransformerModel* model, const TransformerLayer* transformer_layer,
                          const char* tensor_key, densecore::runtime::DenseCoreTensorRole role,
                          const ggml_tensor* explicit_tensor = nullptr) {
    if (!contract || !layer || !tensor_key) {
        return;
    }
    const ggml_tensor* tensor =
        explicit_tensor ? explicit_tensor : (transformer_layer ? transformer_layer->Get(tensor_key) : nullptr);
    ModelTensorExecutionRequirement requirement =
        MakeTensorRequirement(model, layer->layer_index, tensor_key, tensor, role);
    layer->tensor_requirements.push_back(requirement);
    contract->tensor_requirements.push_back(std::move(requirement));
}

bool IsQwenHybridSSMContract(const ModelExecutionContract& contract) {
    return contract.has_hybrid_ssm_mixer &&
           (contract.variant == ModelVariant::QWEN35 || contract.variant == ModelVariant::QWEN36 ||
            contract.variant == ModelVariant::QWEN38);
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
        return contract.has_moe ? ExecutionFastPathClass::QwenHybridSSMMoE : ExecutionFastPathClass::QwenHybridSSMDense;
    }
    if ((contract.variant == ModelVariant::QWEN35 || contract.variant == ModelVariant::QWEN36 ||
         contract.variant == ModelVariant::QWEN38) &&
        !contract.has_moe &&
        !contract.has_hybrid_ssm_mixer) {
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
            (synthetic_stateful_model
                 ? model->arch_flags.is_hybrid_ssm
                 : (model->IsHybridSSMLayer(layer_idx) || (layer_spec && layer_spec->attention.has_hybrid_ssm_mixer)));
        layer.has_moe = (transformer_layer && transformer_layer->is_moe) || (layer_spec && layer_spec->ffn.is_moe);

        if (layer.has_hybrid_ssm_mixer) {
            layer.ssm_ordinal = ssm_ordinal++;
            layer.runtime_state_shape = HybridSSMRuntimeStateShape(model);
            if (layer.ssm_ordinal >= 0 && layer.ssm_ordinal < static_cast<int>(model->ssm_layer_states.size())) {
                layer.ssm_norm_layout =
                    model->ssm_layer_states[static_cast<std::size_t>(layer.ssm_ordinal)].norm_layout;
            }
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::HybridSSMConv1D,
                                ExecutionRuntimeStateKind::HybridSSM, layer.ssm_ordinal);
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::HybridSSMDelta,
                                ExecutionRuntimeStateKind::HybridSSM, layer.ssm_ordinal);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kAttnQkvWeight,
                                 densecore::runtime::DenseCoreTensorRole::HybridSSMQkv);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kAttnGate,
                                 densecore::runtime::DenseCoreTensorRole::HybridSSMGate);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kSSMOut,
                                 densecore::runtime::DenseCoreTensorRole::SSMOut);
        }

        if (layer.has_lfm2_shortconv_mixer) {
            const int resolved_ordinal = model->LFM2ConvOrdinal(layer_idx);
            layer.conv_ordinal = resolved_ordinal >= 0 ? resolved_ordinal : conv_ordinal;
            conv_ordinal = std::max(conv_ordinal, layer.conv_ordinal + 1);
            layer.runtime_state_shape = LFM2ShortConvRuntimeStateShape(model);
            AddRebindDescriptor(&contract, &layer, ExecutionCustomOpRebindKind::LFM2ShortConv,
                                ExecutionRuntimeStateKind::LFM2ShortConv, layer.conv_ordinal);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kShortConvInProj,
                                 densecore::runtime::DenseCoreTensorRole::ShortConvIn);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kShortConvOutProj,
                                 densecore::runtime::DenseCoreTensorRole::ShortConvOut);
        }

        if (layer.has_moe) {
            layer.moe_router = layer_spec ? layer_spec->ffn.router : DecoderMoERouter::None;
            layer.moe_top_k = layer_spec ? layer_spec->ffn.top_k : static_cast<int>(model->hparams.n_experts_used);
            layer.moe_num_experts = layer_spec
                                        ? layer_spec->ffn.num_experts
                                        : static_cast<int>(transformer_layer ? transformer_layer->NumExperts() : 0);
            layer.moe_expert_layout = transformer_layer ? ResolveMoEExpertLayout(*transformer_layer, layer_spec)
                                                        : ExecutionMoEExpertLayoutKind::Unknown;
            layer.has_packed_moe_sidecar =
                (layer_spec && layer_spec->ffn.has_down_scale_sidecar) ||
                (transformer_layer && HasAnyExpertTensor(*transformer_layer, model_keys::kGemma4PackedDownScale));
            if (layer.moe_expert_layout == ExecutionMoEExpertLayoutKind::Unknown) {
                AddRejection(&contract, "moe_expert_layout_unknown");
            }
            AddTensorRequirement(&contract, &layer, model, transformer_layer, model_keys::kMoeGate,
                                 densecore::runtime::DenseCoreTensorRole::MoERouter);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, "ffn_gate_up_exps.weight",
                                 densecore::runtime::DenseCoreTensorRole::MoEGateUp);
            AddTensorRequirement(&contract, &layer, model, transformer_layer, "ffn_down_exps.weight",
                                 densecore::runtime::DenseCoreTensorRole::MoEDown);
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
    contract.requires_native_moe_fast_path = contract.fast_path_class == ExecutionFastPathClass::QwenHybridSSMMoE ||
                                             contract.fast_path_class == ExecutionFastPathClass::LFM2ShortConvMoE;
    if (contract.requires_fallback_free_fast_path) {
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::TargetNoGgmlPath);
    }
    if (contract.requires_native_moe_fast_path) {
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::NativeMoeFastW1W3UsedOps);
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::NativeMoeFastW2UsedOps);
    }
    if (contract.has_hybrid_ssm_mixer) {
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::SsmConv1DCalls);
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::SsmDeltaCalls);
    }
    if (contract.fast_path_class == ExecutionFastPathClass::LFM2ShortConvMoE) {
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::Lfm2ShortConvSequenceFastUsedOps);
    }
    if (contract.fast_path_class == ExecutionFastPathClass::Gemma4MoE) {
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::Gemma4PrefillMaintainedFastOps);
        AddRequiredFastPathCounter(&contract, ExecutionFastPathCounterKind::Gemma4DecodeMaintainedFastOps);
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
    contract.semantic_graph_nodes = BuildModelSemanticGraph(contract);
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

std::vector<ModelSemanticGraphNode> BuildModelSemanticGraph(const ModelExecutionContract& contract) {
    std::vector<ModelSemanticGraphNode> nodes;
    nodes.reserve(contract.tensor_requirements.size() * 2);
    for (const auto& requirement : contract.tensor_requirements) {
        if (requirement.raw_gguf_type == GGML_TYPE_COUNT) {
            continue;
        }
        for (const auto phase :
             {densecore::runtime::DenseCoreMatmulPhase::Prefill, densecore::runtime::DenseCoreMatmulPhase::Decode}) {
            ModelSemanticGraphNode node{};
            node.layer_index = requirement.layer_index;
            node.tensor_key = requirement.tensor_key;
            node.semantic_op = requirement.semantic_op;
            node.tensor_role = requirement.tensor_role;
            node.raw_gguf_type = requirement.raw_gguf_type;
            node.canonical_layout = requirement.canonical_layout;
            node.phase = phase;
            node.required_kernel = ModelTensorExecutionRequirementKernelForPhase(requirement, phase);
            node.fallback_policy = requirement.fallback_policy;
            node.m = phase == densecore::runtime::DenseCoreMatmulPhase::Prefill ? 2 : 1;
            node.n = requirement.tensor_rows;
            node.k = requirement.tensor_cols;
            nodes.push_back(std::move(node));
        }
    }
    return nodes;
}

KernelCoverageValidationResult ValidateFallbackFreeKernelCoverage(const TransformerModel* model,
                                                                  const ModelExecutionContract& contract,
                                                                  densecore::runtime::HostKernelCapabilities caps) {
    KernelCoverageValidationResult result{};
    result.required = ModelExecutionContractRequiresFallbackFreeFastPath(contract);
    const std::vector<ModelSemanticGraphNode> nodes =
        contract.semantic_graph_nodes.empty() ? BuildModelSemanticGraph(contract) : contract.semantic_graph_nodes;
    for (const auto& node : nodes) {
        if (node.fallback_policy != densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget) {
            continue;
        }
        ++result.graph_node_count;
        const auto resolution = densecore::runtime::ResolveKernelResolution(
            model, node.raw_gguf_type, GGML_TYPE_F32, node.m, node.n, node.k, node.phase, node.tensor_key.c_str(),
            node.tensor_role == densecore::runtime::DenseCoreTensorRole::LmHead, /*compatible=*/true, caps,
            node.semantic_op, node.tensor_role, node.required_kernel, node.fallback_policy,
            /*has_fallback_policy_override=*/true);
        const bool covered =
            resolution.selected_kernel != densecore::runtime::DenseCoreKernelFamily::None &&
            resolution.selected_kernel != densecore::runtime::DenseCoreKernelFamily::TemporaryReferenceGgml &&
            resolution.rejected_count == 0 &&
            resolution.fallback_policy == densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget;
        if (covered) {
            ++result.covered_node_count;
        } else {
            result.missing_kernel_reasons.push_back(MissingKernelReason(node, resolution));
        }
    }
    result.fallback_free = result.missing_kernel_reasons.empty();
    return result;
}

std::string FormatKernelCoverageValidationResult(const KernelCoverageValidationResult& result) {
    std::ostringstream oss;
    oss << "KernelCoverage{required=" << (result.required ? "true" : "false")
        << ",fallback_free=" << (result.fallback_free ? "true" : "false") << ",covered=" << result.covered_node_count
        << "/" << result.graph_node_count << ",missing=[";
    for (std::size_t i = 0; i < result.missing_kernel_reasons.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << result.missing_kernel_reasons[i];
    }
    oss << "]}";
    return oss.str();
}

ModelTensorExecutionRequirement ResolveModelTensorExecutionRequirement(const TransformerModel* model,
                                                                       const char* tensor_name, bool is_lm_head,
                                                                       ggml_type raw_type) {
    return ResolveModelTensorExecutionRequirement(model, nullptr, tensor_name, is_lm_head, raw_type);
}

ModelTensorExecutionRequirement ResolveModelTensorExecutionRequirement(const TransformerModel* model,
                                                                       const ggml_tensor* tensor,
                                                                       const char* tensor_name, bool is_lm_head,
                                                                       ggml_type raw_type) {
    const auto loader_role = FindLoaderTensorRole(model, tensor);
    const auto role = loader_role != densecore::runtime::DenseCoreTensorRole::Unknown
                          ? loader_role
                          : densecore::runtime::ResolveDenseCoreTensorRole(model, tensor_name, is_lm_head);
    ModelTensorExecutionRequirement requirement{};
    requirement.layer_index = -1;
    requirement.tensor_key = tensor_name ? tensor_name : "";
    requirement.tensor_role = role;
    requirement.semantic_op = densecore::runtime::ResolveDenseCoreSemanticOp(model, role);
    requirement.raw_gguf_type = raw_type;
    if (tensor) {
        requirement.tensor_cols = tensor->ne[0];
        requirement.tensor_rows = tensor->ne[1];
    }
    requirement.canonical_layout = ExecutionQuantLayoutKind::RawGGUF;
    requirement.repacked_layout = ExecutionQuantLayoutKind::Unknown;
    FinalizeTensorRequirement(model, &requirement);
    return requirement;
}

const ModelTensorExecutionRequirement*
FindModelTensorExecutionRequirement(const ModelExecutionContract& contract, const char* tensor_name,
                                    densecore::runtime::DenseCoreMatmulPhase phase) {
    for (const auto& requirement : contract.tensor_requirements) {
        if (!TensorNameMatchesKey(tensor_name, requirement.tensor_key)) {
            continue;
        }
        if (phase == densecore::runtime::DenseCoreMatmulPhase::Prefill &&
            requirement.prefill_kernel == densecore::runtime::DenseCoreKernelFamily::None) {
            continue;
        }
        if (phase == densecore::runtime::DenseCoreMatmulPhase::Decode &&
            requirement.decode_kernel == densecore::runtime::DenseCoreKernelFamily::None) {
            continue;
        }
        return &requirement;
    }
    return nullptr;
}

densecore::runtime::DenseCoreKernelFamily
ModelTensorExecutionRequirementKernelForPhase(const ModelTensorExecutionRequirement& requirement,
                                              densecore::runtime::DenseCoreMatmulPhase phase) {
    switch (phase) {
    case densecore::runtime::DenseCoreMatmulPhase::Prefill: return requirement.prefill_kernel;
    case densecore::runtime::DenseCoreMatmulPhase::Decode: return requirement.decode_kernel;
    case densecore::runtime::DenseCoreMatmulPhase::Unknown:
    default: return densecore::runtime::DenseCoreKernelFamily::None;
    }
}

densecore::runtime::KernelResolution
ResolveModelTensorKernelResolution(const TransformerModel* model, const ModelTensorExecutionRequirement& requirement,
                                   ggml_type weight_type, ggml_type input_type, int64_t m, int64_t n, int64_t k,
                                   densecore::runtime::DenseCoreMatmulPhase phase, const char* weight_name,
                                   bool is_lm_head, bool compatible, densecore::runtime::HostKernelCapabilities caps) {
    return densecore::runtime::ResolveKernelResolution(
        model, weight_type, input_type, m, n, k, phase, weight_name, is_lm_head, compatible, caps,
        requirement.semantic_op, requirement.tensor_role,
        ModelTensorExecutionRequirementKernelForPhase(requirement, phase), requirement.fallback_policy,
        /*has_fallback_policy_override=*/true);
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

const char* ExecutionQuantLayoutKindName(ExecutionQuantLayoutKind kind) {
    switch (kind) {
    case ExecutionQuantLayoutKind::RawGGUF: return "raw_gguf";
    case ExecutionQuantLayoutKind::LoaderCanonical: return "loader_canonical";
    case ExecutionQuantLayoutKind::CpuRepacked: return "cpu_repacked";
    case ExecutionQuantLayoutKind::AmxPrefillAlias: return "amx_prefill_alias";
    case ExecutionQuantLayoutKind::PackedDenseCore: return "packed_densecore";
    case ExecutionQuantLayoutKind::Unknown: return "unknown";
    }
    return "unknown";
}

const char* ExecutionFastPathCounterKindName(ExecutionFastPathCounterKind kind) {
    switch (kind) {
    case ExecutionFastPathCounterKind::TargetNoGgmlPath: return "target_no_ggml_path";
    case ExecutionFastPathCounterKind::NativeMoeFastW1W3UsedOps: return "native_moe_fast_w1w3_used_ops";
    case ExecutionFastPathCounterKind::NativeMoeFastW2UsedOps: return "native_moe_fast_w2_used_ops";
    case ExecutionFastPathCounterKind::SsmConv1DCalls: return "ssm_conv1d_calls";
    case ExecutionFastPathCounterKind::SsmDeltaCalls: return "ssm_delta_calls";
    case ExecutionFastPathCounterKind::Lfm2ShortConvSequenceFastUsedOps: return "lfm2_shortconv_sequence_fast_used_ops";
    case ExecutionFastPathCounterKind::Gemma4PrefillMaintainedFastOps: return "gemma4_prefill_maintained_fast_ops";
    case ExecutionFastPathCounterKind::Gemma4DecodeMaintainedFastOps: return "gemma4_decode_maintained_fast_ops";
    case ExecutionFastPathCounterKind::Unknown:
    default: return "unknown";
    }
}

ExecutionFastPathCounterKind ExecutionFastPathCounterKindFromName(const std::string& name) {
    if (name == "target_no_ggml_path") {
        return ExecutionFastPathCounterKind::TargetNoGgmlPath;
    }
    if (name == "native_moe_fast_w1w3_used_ops") {
        return ExecutionFastPathCounterKind::NativeMoeFastW1W3UsedOps;
    }
    if (name == "native_moe_fast_w2_used_ops") {
        return ExecutionFastPathCounterKind::NativeMoeFastW2UsedOps;
    }
    if (name == "ssm_conv1d_calls") {
        return ExecutionFastPathCounterKind::SsmConv1DCalls;
    }
    if (name == "ssm_delta_calls") {
        return ExecutionFastPathCounterKind::SsmDeltaCalls;
    }
    if (name == "lfm2_shortconv_sequence_fast_used_ops") {
        return ExecutionFastPathCounterKind::Lfm2ShortConvSequenceFastUsedOps;
    }
    if (name == "gemma4_prefill_maintained_fast_ops") {
        return ExecutionFastPathCounterKind::Gemma4PrefillMaintainedFastOps;
    }
    if (name == "gemma4_decode_maintained_fast_ops") {
        return ExecutionFastPathCounterKind::Gemma4DecodeMaintainedFastOps;
    }
    return ExecutionFastPathCounterKind::Unknown;
}

std::vector<ExecutionFastPathCounterKind>
ModelExecutionContractRequiredFastPathCounterKinds(const ModelExecutionContract& contract) {
    if (!contract.required_fast_path_counter_kinds.empty()) {
        return contract.required_fast_path_counter_kinds;
    }

    std::vector<ExecutionFastPathCounterKind> kinds;
    kinds.reserve(contract.required_fast_path_counters.size());
    for (const std::string& counter : contract.required_fast_path_counters) {
        const auto kind = ExecutionFastPathCounterKindFromName(counter);
        if (kind != ExecutionFastPathCounterKind::Unknown) {
            kinds.push_back(kind);
        }
    }
    return kinds;
}

std::string FormatModelExecutionContractRequiredFastPathCounters(const ModelExecutionContract& contract) {
    const std::vector<ExecutionFastPathCounterKind> kinds =
        ModelExecutionContractRequiredFastPathCounterKinds(contract);
    if (kinds.empty() && contract.required_fast_path_counters.empty()) {
        return "none";
    }

    std::ostringstream oss;
    bool any = false;
    for (const auto kind : kinds) {
        if (any) {
            oss << ",";
        }
        oss << ExecutionFastPathCounterKindName(kind);
        any = true;
    }
    for (const std::string& counter : contract.required_fast_path_counters) {
        if (ExecutionFastPathCounterKindFromName(counter) != ExecutionFastPathCounterKind::Unknown) {
            continue;
        }
        if (any) {
            oss << ",";
        }
        oss << counter;
        any = true;
    }
    return any ? oss.str() : "none";
}

std::string FormatModelExecutionContract(const ModelExecutionContract& contract) {
    std::ostringstream oss;
    const std::string required_fast_path_counters = FormatModelExecutionContractRequiredFastPathCounters(contract);
    oss << "ModelExecutionContract{variant=" << static_cast<int>(contract.variant)
        << ",topology=" << DecoderRuntimeTopologyName(contract.decoder_runtime_topology)
        << ",valid=" << (contract.valid ? "true" : "false") << ",has_moe=" << (contract.has_moe ? "true" : "false")
        << ",has_hybrid_ssm=" << (contract.has_hybrid_ssm_mixer ? "true" : "false")
        << ",has_lfm2_shortconv=" << (contract.has_lfm2_shortconv_mixer ? "true" : "false")
        << ",fast_path_class=" << ExecutionFastPathClassName(contract.fast_path_class)
        << ",requires_fallback_free_fast_path=" << (contract.requires_fallback_free_fast_path ? "true" : "false")
        << ",requires_native_moe_fast_path=" << (contract.requires_native_moe_fast_path ? "true" : "false")
        << ",native_moe_max_direct_tokens=" << contract.native_moe_max_direct_tokens
        << ",decode_cache_static_safe=" << (contract.decode_graph_cache_static_safe ? "true" : "false")
        << ",requires_rebind=" << (contract.requires_decode_graph_runtime_rebind ? "true" : "false")
        << ",required_fast_path_counters=[";
    if (required_fast_path_counters != "none") {
        oss << required_fast_path_counters;
    }
    oss << "]" << ",forbidden_fast_paths=[";
    for (std::size_t i = 0; i < contract.forbidden_fast_path_reasons.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << contract.forbidden_fast_path_reasons[i];
    }
    oss << "]" << ",layers=[";
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
            << ",ownership=" << ExecutionTensorOwnershipName(layer.tensor_ownership)
            << ",tensor_requirements=" << layer.tensor_requirements.size() << "}";
    }
    oss << "],rebinds=[";
    for (std::size_t i = 0; i < contract.rebind_descriptors.size(); ++i) {
        const auto& descriptor = contract.rebind_descriptors[i];
        if (i != 0) {
            oss << ",";
        }
        oss << ExecutionCustomOpRebindKindName(descriptor.op_kind) << "@layer" << descriptor.layer_index;
    }
    oss << "],tensor_requirements=[";
    for (std::size_t i = 0; i < contract.tensor_requirements.size(); ++i) {
        const auto& requirement = contract.tensor_requirements[i];
        if (i != 0) {
            oss << ",";
        }
        oss << requirement.tensor_key << ":" << densecore::runtime::DenseCoreTensorRoleName(requirement.tensor_role)
            << ":prefill=" << densecore::runtime::DenseCoreKernelFamilyName(requirement.prefill_kernel)
            << ":decode=" << densecore::runtime::DenseCoreKernelFamilyName(requirement.decode_kernel)
            << ":shape=" << ShapeLabel(0, requirement.tensor_rows, requirement.tensor_cols)
            << ":layout=" << ExecutionQuantLayoutKindName(requirement.canonical_layout)
            << ":fallback=" << densecore::runtime::DenseCoreFallbackPolicyKindName(requirement.fallback_policy);
    }
    oss << "],semantic_graph_nodes=" << contract.semantic_graph_nodes.size() << "}";
    return oss.str();
}

}  // namespace densecore::models
