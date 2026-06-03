#ifndef DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H
#define DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_types.h"

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
    bool requires_native_moe_fast_path = false;
    int64_t native_moe_max_direct_tokens = 0;
    bool decode_graph_cache_static_safe = true;
    bool requires_decode_graph_runtime_rebind = false;
    std::vector<ModelExecutionLayerContract> layers;
    std::vector<ExecutionCustomOpRebindDescriptor> rebind_descriptors;
    std::vector<std::string> rejection_reasons;
};

ModelExecutionContract BuildModelExecutionContract(const TransformerModel* model);
bool ModelExecutionContractAllowsDecodeGraphCache(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresDecodeGraphRuntimeRebind(const ModelExecutionContract& contract);
bool ModelExecutionContractRequiresNativeMoEFastPath(const ModelExecutionContract& contract);
int64_t ModelExecutionContractNativeMoEMaxDirectTokens(const ModelExecutionContract& contract);

const char* ExecutionRuntimeStateKindName(ExecutionRuntimeStateKind kind);
const char* ExecutionTensorOwnershipName(ExecutionTensorOwnership ownership);
const char* ExecutionMoEExpertLayoutKindName(ExecutionMoEExpertLayoutKind kind);
const char* ExecutionCustomOpRebindKindName(ExecutionCustomOpRebindKind kind);
std::string FormatModelExecutionContract(const ModelExecutionContract& contract);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_MODEL_EXECUTION_CONTRACT_H
