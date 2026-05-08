#ifndef DENSECORE_MODELS_DECODER_MODEL_SPEC_H
#define DENSECORE_MODELS_DECODER_MODEL_SPEC_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "densecore/models/model_types.h"

namespace densecore::models {

enum class DecoderActivation : uint8_t {
    Silu = 0,
    GeluPytorchTanh,
};

enum class DecoderMoERouter : uint8_t {
    None = 0,
    SoftmaxTopK,
    Gemma4SoftmaxTopK,
    GroupedSigmoidTopK,
};

enum class DecoderRopeKind : uint8_t {
    Standard = 0,
    Neox,
    Proportional,
};

enum class DecoderPrefillLogitsPolicy : uint8_t {
    FullSequence = 0,
    LastTokenEnvOptIn,
    LastTokenEnvDefaultOn,
    LastTokenForMoE,
};

enum class DecoderRuntimeTopology : uint8_t {
    Unknown = 0,
    DenseAttention,
    DenseAttentionMoE,
    HybridSSM,
    HybridSSMMoE,
    SlidingSharedKV,
    SlidingSharedKVMoE,
};

enum class DecoderSemanticOpKind : uint8_t {
    AttentionNorm = 0,
    AttentionProjection,
    HybridSSMMixer,
    SharedKVRead,
    SharedKVPublish,
    AttentionCore,
    AttentionOutputProjection,
    FfnNorm,
    MoERouter,
    MoEExpertDispatch,
    SharedDenseFfn,
    DenseFfn,
    FfnPostNorm,
    ResidualAdd,
};

enum class DecoderSpecializationKind : uint8_t {
    HybridSSMMixer = 0,
    SlidingWindowAttention,
    SharedKV,
    PerLayerKVHeads,
    QNorm,
    KNorm,
    VNorm,
    AttentionLogitSoftcap,
    MoE,
    GroupedMoERouter,
    Gemma4MoERouter,
    SharedDenseFfn,
    MoEDownScaleSidecar,
    FfnPostNorms,
    PrefillLastLogits,
};

struct DecoderSemanticOp {
    DecoderSemanticOpKind kind = DecoderSemanticOpKind::AttentionNorm;
    int layer_index = -1;
};

struct DecoderSpecialization {
    DecoderSpecializationKind kind = DecoderSpecializationKind::HybridSSMMixer;
    int layer_index = -1;
};

struct DecoderAttentionSpec {
    bool has_dense_attention = true;
    bool has_hybrid_ssm_mixer = false;
    bool is_sliding_window = false;
    bool reads_shared_kv = false;
    bool publishes_shared_kv = false;
    bool requires_q_norm = false;
    bool requires_k_norm = false;
    bool requires_v_norm = false;
    int kv_source_layer = -1;
    int kv_head_count = 0;
    int sliding_window = -1;
    int rope_dim = 0;
    float attention_scale = 0.0f;
    float logit_softcap = 0.0f;
    DecoderRopeKind rope_kind = DecoderRopeKind::Standard;
};

struct DecoderFfnSpec {
    bool is_moe = false;
    bool has_shared_dense_branch = false;
    bool has_pre_moe_norm = false;
    bool has_post_shared_norm = false;
    bool has_post_moe_norm = false;
    bool has_post_ffn_norm = false;
    bool has_down_scale_sidecar = false;
    int top_k = 0;
    int num_experts = 0;
    DecoderActivation activation = DecoderActivation::Silu;
    DecoderMoERouter router = DecoderMoERouter::None;
};

struct DecoderLayerSpec {
    int layer_index = -1;
    DecoderAttentionSpec attention;
    DecoderFfnSpec ffn;
    std::vector<DecoderSemanticOp> semantic_ops;
};

struct DecoderOutputSpec {
    DecoderPrefillLogitsPolicy prefill_logits_policy = DecoderPrefillLogitsPolicy::FullSequence;
};

struct DecoderModelSpec {
    ModelArch arch = ModelArch::UNKNOWN;
    ModelVariant variant = ModelVariant::UNKNOWN;
    DecoderRuntimeTopology runtime_topology = DecoderRuntimeTopology::Unknown;
    bool has_moe = false;
    bool has_hybrid_ssm_mixer = false;
    bool has_sliding_window_attention = false;
    bool has_shared_kv = false;
    DecoderOutputSpec output;
    std::vector<DecoderLayerSpec> layers;
    std::vector<DecoderSpecialization> specializations;
};

struct DecoderPrefillRuntimePolicy {
    bool qwen35_prefill_last_logits_only = false;
    bool qwen36_prefill_last_logits_only = true;
};

DecoderModelSpec BuildDecoderModelSpec(const TransformerModel* model);
const DecoderModelSpec* GetDecoderModelSpec(const TransformerModel* model);
const DecoderLayerSpec* GetDecoderLayerSpec(const DecoderModelSpec* spec, int layer_idx);
const DecoderLayerSpec* ResolveDecoderLayerSpecForLayer(const TransformerModel* model, const TransformerLayer* layer);
std::shared_ptr<const DecoderModelSpec> MakeDecoderModelSpec(const TransformerModel* model);
DecoderPrefillRuntimePolicy DefaultDecoderPrefillRuntimePolicy();
bool ShouldUsePrefillLastLogitsOnly(const DecoderModelSpec* spec, int num_seqs, int n_tokens,
                                    DecoderPrefillRuntimePolicy policy = DefaultDecoderPrefillRuntimePolicy());

const char* DecoderActivationName(DecoderActivation activation);
const char* DecoderMoERouterName(DecoderMoERouter router);
const char* DecoderRopeKindName(DecoderRopeKind kind);
const char* DecoderPrefillLogitsPolicyName(DecoderPrefillLogitsPolicy policy);
const char* DecoderRuntimeTopologyName(DecoderRuntimeTopology topology);
const char* DecoderSemanticOpKindName(DecoderSemanticOpKind kind);
const char* DecoderSpecializationKindName(DecoderSpecializationKind kind);
bool DecoderModelSpecHasSpecialization(const DecoderModelSpec& spec, DecoderSpecializationKind kind);
std::string FormatDecoderLayerSpec(const DecoderLayerSpec& layer);
std::string FormatDecoderSpecializations(const DecoderModelSpec& spec);
std::string FormatDecoderModelSpec(const DecoderModelSpec& spec);

}  // namespace densecore::models

#endif  // DENSECORE_MODELS_DECODER_MODEL_SPEC_H
