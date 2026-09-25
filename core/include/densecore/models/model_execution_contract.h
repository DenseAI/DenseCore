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

enum class ExecutionFastPathCounterKind : uint8_t {
    Unknown = 0,
    TargetNoGgmlPath,
    NativeMoeFastW1W3UsedOps,
    NativeMoeFastW2UsedOps,
    SsmConv1DCalls,
    SsmDeltaCalls,
    Lfm2ShortConvSequenceFastUsedOps,
    Gemma4PrefillMaintainedFastOps,
    Gemma4DecodeMaintainedFastOps,
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
    int64_t tensor_cols = 0;
    int64_t tensor_rows = 0;
    ExecutionQuantLayoutKind canonical_layout = ExecutionQuantLayoutKind::Unknown;
    ExecutionQuantLayoutKind repacked_layout = ExecutionQuantLayoutKind::Unknown;
    densecore::runtime::DenseCoreKernelFamily prefill_kernel = densecore::runtime::DenseCoreKernelFamily::None;
    densecore::runtime::DenseCoreKernelFamily decode_kernel = densecore::runtime::DenseCoreKernelFamily::None;
    densecore::runtime::DenseCoreFallbackPolicyKind fallback_policy =
        densecore::runtime::DenseCoreFallbackPolicyKind::CompatibilityFallback;
};

struct ModelSemanticGraphNode {
    int layer_index = -1;
    std::string tensor_key;
    densecore::runtime::DenseCoreSemanticOp semantic_op = densecore::runtime::DenseCoreSemanticOp::Unknown;
    densecore::runtime::DenseCoreTensorRole tensor_role = densecore::runtime::DenseCoreTensorRole::Unknown;
    ggml_type raw_gguf_type = GGML_TYPE_COUNT;
    ExecutionQuantLayoutKind canonical_layout = ExecutionQuantLayoutKind::Unknown;
    densecore::runtime::DenseCoreMatmulPhase phase = densecore::runtime::DenseCoreMatmulPhase::Unknown;
    densecore::runtime::DenseCoreKernelFamily required_kernel = densecore::runtime::DenseCoreKernelFamily::None;
    densecore::runtime::DenseCoreFallbackPolicyKind fallback_policy =
        densecore::runtime::DenseCoreFallbackPolicyKind::CompatibilityFallback;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
};

struct KernelCoverageValidationResult {
    bool required = false;
    bool fallback_free = true;
    std::size_t graph_node_count = 0;
    std::size_t covered_node_count = 0;
    std::vector<std::string> missing_kernel_reasons;
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
    std::vector<ModelSemanticGraphNode> semantic_graph_nodes;
    std::vector<ExecutionCustomOpRebindDescriptor> rebind_descriptors;
    std::vector<std::string> rejection_reasons;
    std::vector<ExecutionFastPathCounterKind> required_fast_path_counter_kinds;
    std::vector<std::string> required_fast_path_counters;
    std::vector<std::string> forbidden_fast_path_reasons;
};

ModelExecutionContract BuildModelExecutionContract(const TransformerModel* model);
bool ModelExecutionContractAllowsDecodeGraphCache(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresDecodeGraphRuntimeRebind(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresFallbackFreeFastPath(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresNativeMoEFastPath(const ModelExecutionContract& contract);
int64_t ModelExecutionContractNativeMoEMaxDirectTokens(const ModelExecutionContract& contract);
std::vector<ModelSemanticGraphNode> BuildModelSemanticGraph(const ModelExecutionContract& contract);
KernelCoverageValidationResult ValidateFallbackFreeKernelCoverage(
    const TransformerModel* model, const ModelExecutionContract& contract,
    densecore::runtime::HostKernelCapabilities caps = densecore::runtime::HostKernelCapabilities{});
std::string FormatKernelCoverageValidationResult(const KernelCoverageValidationResult& result);
ModelTensorExecutionRequirement ResolveModelTensorExecutionRequirement(const TransformerModel* model,
                                                                       const char* tensor_name, bool is_lm_head,
                                                                       ggml_type raw_type);
ModelTensorExecutionRequirement ResolveModelTensorExecutionRequirement(const TransformerModel* model,
                                                                       const ggml_tensor* tensor,
                                                                       const char* tensor_name, bool is_lm_head,
                                                                       ggml_type raw_type);
const ModelTensorExecutionRequirement* FindModelTensorExecutionRequirement(
    const ModelExecutionContract& contract, const char* tensor_name,
    densecore::runtime::DenseCoreMatmulPhase phase = densecore::runtime::DenseCoreMatmulPhase::Unknown);
densecore::runtime::DenseCoreKernelFamily
ModelTensorExecutionRequirementKernelForPhase(const ModelTensorExecutionRequirement& requirement,
                                              densecore::runtime::DenseCoreMatmulPhase phase);
densecore::runtime::KernelResolution ResolveModelTensorKernelResolution(
    const TransformerModel* model, const ModelTensorExecutionRequirement& requirement, ggml_type weight_type,
    ggml_type input_type, int64_t m, int64_t n, int64_t k, densecore::runtime::DenseCoreMatmulPhase phase,
    const char* weight_name, bool is_lm_head, bool compatible,
    densecore::runtime::HostKernelCapabilities caps = densecore::runtime::HostKernelCapabilities{});

const char* ExecutionRuntimeStateKindName(ExecutionRuntimeStateKind kind);
const char* ExecutionTensorOwnershipName(ExecutionTensorOwnership ownership);
const char* ExecutionMoEExpertLayoutKindName(ExecutionMoEExpertLayoutKind kind);
const char* ExecutionCustomOpRebindKindName(ExecutionCustomOpRebindKind kind);
const char* ExecutionFastPathClassName(ExecutionFastPathClass kind);
const char* ExecutionQuantLayoutKindName(ExecutionQuantLayoutKind kind);
const char* ExecutionFastPathCounterKindName(ExecutionFastPathCounterKind kind);
ExecutionFastPathCounterKind ExecutionFastPathCounterKindFromName(const std::string& name);
std::vector<ExecutionFastPathCounterKind>
ModelExecutionContractRequiredFastPathCounterKinds(const ModelExecutionContract& contract);
std::string FormatModelExecutionContractRequiredFastPathCounters(const ModelExecutionContract& contract);
std::string FormatModelExecutionContract(const ModelExecutionContract& contract);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H
