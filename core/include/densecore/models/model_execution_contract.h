#ifndef DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H
#define DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_types.h"
#include "densecore/runtime/ggml_compute_policy.h"

namespace densecore::models {

enum class ExecutionRuntimeStateKind : uint8_t {
    None = 0,
    HybridSSM,
    LFM2ShortConv,
};

enum class ExecutionTensorOwnership : uint8_t {
    None = 0,
    RawBoundTensor,
    LoaderCanonicalBuffer,
    RuntimeOwnedView,
};

enum class ExecutionMoEExpertLayoutKind : uint8_t {
    None = 0,
    SeparateExpertMaps,
    PackedGateUpDown,
    PackedSeparateGateUpDown,
    Gemma4PackedWithDownScale,
    Unknown,
};

enum class ExecutionCustomOpRebindKind : uint8_t {
    None = 0,
    HybridSSMConv1D,
    HybridSSMDelta,
    LFM2ShortConv,
};

enum class ExecutionFastPathClass : uint8_t {
    None = 0,
    QwenDense,
    QwenHybridSSMDense,
    QwenHybridSSMMoE,
    Gemma4MoE,
    LFM2ShortConvMoE,
};

enum class ExecutionQuantLayoutKind : uint8_t {
    Unknown = 0,
    RawGGUF,
    LoaderCanonical,
    CpuRepacked,
    AmxPrefillAlias,
    PackedDenseCore,
};

struct ExecutionRuntimeStateShape {
    ExecutionRuntimeStateKind kind = ExecutionRuntimeStateKind::None;
    int conv_channels = 0;
    int kernel_size = 0;
    int n_heads = 0;
    int head_dim = 0;
    int state_size = 0;
    std::size_t expected_conv_state_elements = 0;
    std::size_t expected_ssm_state_elements = 0;
};

struct ModelTensorExecutionRequirement {
    int layer_index = -1;
    std::string tensor_key;
    densecore::runtime::DenseCoreSemanticOp semantic_op = densecore::runtime::DenseCoreSemanticOp::Unknown;
    densecore::runtime::DenseCoreTensorRole tensor_role = densecore::runtime::DenseCoreTensorRole::Unknown;
    ggml_type raw_gguf_type = GGML_TYPE_COUNT;
    ExecutionQuantLayoutKind canonical_layout = ExecutionQuantLayoutKind::Unknown;
    ExecutionQuantLayoutKind repacked_layout = ExecutionQuantLayoutKind::Unknown;
    densecore::runtime::DenseCoreKernelFamily prefill_kernel = densecore::runtime::DenseCoreKernelFamily::None;
    densecore::runtime::DenseCoreKernelFamily decode_kernel = densecore::runtime::DenseCoreKernelFamily::None;
    densecore::runtime::DenseCoreFallbackPolicyKind fallback_policy =
        densecore::runtime::DenseCoreFallbackPolicyKind::CompatibilityFallback;
};

struct ExecutionCustomOpRebindDescriptor {
    int layer_index = -1;
    int state_ordinal = -1;
    ExecutionRuntimeStateKind state_kind = ExecutionRuntimeStateKind::None;
    ExecutionCustomOpRebindKind op_kind = ExecutionCustomOpRebindKind::None;
};

struct ModelExecutionLayerContract {
    int layer_index = -1;
    bool has_hybrid_ssm_mixer = false;
    bool has_lfm2_shortconv_mixer = false;
    bool has_moe = false;
    int ssm_ordinal = -1;
    int conv_ordinal = -1;
    ExecutionRuntimeStateShape runtime_state_shape{};
    Qwen35SSMNormLayout ssm_norm_layout = Qwen35SSMNormLayout::INVALID;
    DecoderMoERouter moe_router = DecoderMoERouter::None;
    int moe_top_k = 0;
    int moe_num_experts = 0;
    ExecutionMoEExpertLayoutKind moe_expert_layout = ExecutionMoEExpertLayoutKind::None;
    bool has_packed_moe_sidecar = false;
    ExecutionTensorOwnership tensor_ownership = ExecutionTensorOwnership::None;
    std::vector<ModelTensorExecutionRequirement> tensor_requirements;
    std::vector<ExecutionCustomOpRebindDescriptor> rebind_descriptors;
};

struct ModelExecutionContract {
    ModelArch arch = ModelArch::UNKNOWN;
    ModelVariant variant = ModelVariant::UNKNOWN;
    DecoderRuntimeTopology decoder_runtime_topology = DecoderRuntimeTopology::Unknown;
    bool valid = true;
    bool has_moe = false;
    bool has_hybrid_ssm_mixer = false;
    bool has_lfm2_shortconv_mixer = false;
    bool has_stateful_custom_ops = false;
    bool requires_fallback_free_fast_path = false;
    ExecutionFastPathClass fast_path_class = ExecutionFastPathClass::None;
    bool requires_native_moe_fast_path = false;
    int64_t native_moe_max_direct_tokens = 0;
    bool decode_graph_cache_static_safe = true;
    bool requires_decode_graph_runtime_rebind = false;
    std::vector<ModelExecutionLayerContract> layers;
    std::vector<ModelTensorExecutionRequirement> tensor_requirements;
    std::vector<ExecutionCustomOpRebindDescriptor> rebind_descriptors;
    std::vector<std::string> rejection_reasons;
    std::vector<std::string> required_fast_path_counters;
    std::vector<std::string> forbidden_fast_path_reasons;
};

ModelExecutionContract BuildModelExecutionContract(const TransformerModel* model);
bool ModelExecutionContractAllowsDecodeGraphCache(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresDecodeGraphRuntimeRebind(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresFallbackFreeFastPath(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresNativeMoEFastPath(const ModelExecutionContract& contract);
int64_t ModelExecutionContractNativeMoEMaxDirectTokens(const ModelExecutionContract& contract);
ModelTensorExecutionRequirement ResolveModelTensorExecutionRequirement(const TransformerModel* model,
                                                                      const char* tensor_name, bool is_lm_head,
                                                                      ggml_type raw_type);
const ModelTensorExecutionRequirement* FindModelTensorExecutionRequirement(
    const ModelExecutionContract& contract, const char* tensor_name,
    densecore::runtime::DenseCoreMatmulPhase phase = densecore::runtime::DenseCoreMatmulPhase::Unknown);

const char* ExecutionRuntimeStateKindName(ExecutionRuntimeStateKind kind);
const char* ExecutionTensorOwnershipName(ExecutionTensorOwnership ownership);
const char* ExecutionMoEExpertLayoutKindName(ExecutionMoEExpertLayoutKind kind);
const char* ExecutionCustomOpRebindKindName(ExecutionCustomOpRebindKind kind);
const char* ExecutionFastPathClassName(ExecutionFastPathClass kind);
const char* ExecutionQuantLayoutKindName(ExecutionQuantLayoutKind kind);
std::string FormatModelExecutionContract(const ModelExecutionContract& contract);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H
