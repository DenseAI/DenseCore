#include <gtest/gtest.h>

#include <ggml.h>

#include <algorithm>
#include <string>
#include <vector>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_execution_contract.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/runtime/inference.h"

namespace {

ggml_tensor* NewTensor2D(ggml_context* ctx, int rows, int cols) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
}

ggml_tensor* NewTypedTensor2D(ggml_context* ctx, ggml_type type, int rows, int cols) {
    return ggml_new_tensor_2d(ctx, type, rows, cols);
}

bool HasOp(const densecore::models::DecoderLayerSpec& layer, densecore::models::DecoderSemanticOpKind kind) {
    return std::any_of(layer.semantic_ops.begin(), layer.semantic_ops.end(),
                       [kind](const densecore::models::DecoderSemanticOp& op) { return op.kind == kind; });
}

const densecore::models::ModelTensorExecutionRequirement* FindRequirement(
    const densecore::models::ModelExecutionContract& contract, const char* key) {
    return densecore::models::FindModelTensorExecutionRequirement(contract, key);
}

}  // namespace

TEST(DecoderModelSpec, Gemma4MoEResolvesSemanticLayerContract) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.arch_flags.requires_q_norm = true;
    model.arch_flags.requires_k_norm = true;
    model.hparams.n_embd = 8;
    model.hparams.n_layer = 2;
    model.hparams.n_experts = 4;
    model.hparams.n_experts_used = 2;
    model.hparams.f_attention_scale = 0.25f;
    model.gemma4_attention_logit_softcapping = 50.0f;
    model.gemma4_sliding_window = 1024;
    model.gemma4_layer_is_sliding = {1, 0};
    model.gemma4_layer_kv_source = {0, 0};
    model.gemma4_layer_n_head_kv = {4, 2};
    model.gemma4_rope_dim_swa = 8;
    model.gemma4_rope_dim_full = 16;
    model.layers.resize(2);

    auto& layer0 = model.layers[0];
    layer0.is_moe = false;
    layer0.Set(model_keys::kMoeGate, NewTensor2D(ctx, 8, 4));
    layer0.Set(model_keys::kFfnGate, NewTensor2D(ctx, 8, 16));
    layer0.Set(model_keys::kFfnUp, NewTensor2D(ctx, 8, 16));
    layer0.Set(model_keys::kFfnDown, NewTensor2D(ctx, 16, 8));
    layer0.Set(model_keys::kAttnQNorm, NewTensor2D(ctx, 1, 8));
    layer0.Set(model_keys::kAttnKNorm, NewTensor2D(ctx, 1, 8));
    layer0.SetExpert(0, model_keys::kGemma4PackedDownScale, NewTensor2D(ctx, 1, 1));
    layer0.Set("gemma4.router.scale", NewTensor2D(ctx, 1, 1));
    layer0.Set("gemma4.pre_feedforward_layernorm_2.weight", NewTensor2D(ctx, 1, 8));
    layer0.Set("gemma4.post_feedforward_layernorm_1.weight", NewTensor2D(ctx, 1, 8));
    layer0.Set("gemma4.post_feedforward_layernorm_2.weight", NewTensor2D(ctx, 1, 8));

    auto& layer1 = model.layers[1];
    layer1.Set(model_keys::kAttnQNorm, NewTensor2D(ctx, 1, 8));
    layer1.Set(model_keys::kAttnKNorm, NewTensor2D(ctx, 1, 8));

    const auto spec = densecore::models::BuildDecoderModelSpec(&model);
    ASSERT_EQ(spec.layers.size(), 2u);
    EXPECT_TRUE(spec.has_moe);
    EXPECT_TRUE(spec.has_shared_kv);
    EXPECT_TRUE(spec.has_sliding_window_attention);
    EXPECT_EQ(spec.runtime_topology, densecore::models::DecoderRuntimeTopology::SlidingSharedKVMoE);
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::SlidingWindowAttention));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::SharedKV));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::Gemma4MoERouter));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::MoEDownScaleSidecar));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::AttentionLogitSoftcap));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::PrefillLastLogits));
    EXPECT_EQ(spec.output.prefill_logits_policy, densecore::models::DecoderPrefillLogitsPolicy::LastTokenForMoE);
    EXPECT_TRUE(densecore::models::ShouldUsePrefillLastLogitsOnly(&spec, /*num_seqs=*/1, /*n_tokens=*/8));
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&spec, /*num_seqs=*/2, /*n_tokens=*/8));

    const auto& moe = spec.layers[0];
    EXPECT_TRUE(moe.ffn.is_moe);
    EXPECT_TRUE(moe.ffn.has_shared_dense_branch);
    EXPECT_TRUE(moe.ffn.has_down_scale_sidecar);
    EXPECT_EQ(moe.ffn.top_k, 2);
    EXPECT_EQ(moe.ffn.activation, densecore::models::DecoderActivation::GeluPytorchTanh);
    EXPECT_EQ(moe.ffn.router, densecore::models::DecoderMoERouter::Gemma4SoftmaxTopK);
    EXPECT_TRUE(moe.attention.is_sliding_window);
    EXPECT_TRUE(moe.attention.publishes_shared_kv);
    EXPECT_FALSE(moe.attention.reads_shared_kv);
    EXPECT_EQ(moe.attention.kv_head_count, 4);
    EXPECT_EQ(moe.attention.rope_kind, densecore::models::DecoderRopeKind::Neox);
    EXPECT_FLOAT_EQ(moe.attention.logit_softcap, 50.0f);
    EXPECT_TRUE(HasOp(moe, densecore::models::DecoderSemanticOpKind::MoERouter));
    EXPECT_TRUE(HasOp(moe, densecore::models::DecoderSemanticOpKind::MoEExpertDispatch));
    EXPECT_TRUE(HasOp(moe, densecore::models::DecoderSemanticOpKind::SharedDenseFfn));
    EXPECT_TRUE(HasOp(moe, densecore::models::DecoderSemanticOpKind::SharedKVPublish));

    const auto& shared = spec.layers[1];
    EXPECT_FALSE(shared.ffn.is_moe);
    EXPECT_EQ(shared.ffn.activation, densecore::models::DecoderActivation::GeluPytorchTanh);
    EXPECT_TRUE(shared.attention.reads_shared_kv);
    EXPECT_FALSE(shared.attention.requires_k_norm);
    EXPECT_EQ(shared.attention.kv_source_layer, 0);
    EXPECT_EQ(shared.attention.kv_head_count, 2);
    EXPECT_EQ(shared.attention.rope_kind, densecore::models::DecoderRopeKind::Proportional);
    EXPECT_TRUE(HasOp(shared, densecore::models::DecoderSemanticOpKind::SharedKVRead));
    EXPECT_TRUE(HasOp(shared, densecore::models::DecoderSemanticOpKind::DenseFfn));

    const std::string formatted = densecore::models::FormatDecoderModelSpec(spec);
    EXPECT_NE(formatted.find("topology=sliding_shared_kv_moe"), std::string::npos);
    EXPECT_NE(formatted.find("prefill_logits=last_token_for_moe"), std::string::npos);
    EXPECT_NE(formatted.find("specializations=[prefill_last_logits"), std::string::npos);
    EXPECT_NE(formatted.find("gemma4_moe_router@layer0"), std::string::npos);
    EXPECT_NE(formatted.find("router=gemma4_softmax_top_k"), std::string::npos);
    EXPECT_NE(formatted.find("activation=gelu_pytorch_tanh"), std::string::npos);
    EXPECT_NE(formatted.find("shared_kv_read=true"), std::string::npos);
    EXPECT_NE(formatted.find("ops=[attention_norm,attention_projection"), std::string::npos);

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    EXPECT_TRUE(contract.has_moe);
    EXPECT_FALSE(contract.has_stateful_custom_ops);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::Gemma4MoE);
    EXPECT_TRUE(contract.requires_fallback_free_fast_path);
    EXPECT_FALSE(contract.requires_native_moe_fast_path);
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresFallbackFreeFastPath(contract));
    EXPECT_FALSE(densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract));
    ASSERT_EQ(contract.required_fast_path_counters.size(), 3u);
    EXPECT_EQ(contract.required_fast_path_counters[0], "target_no_ggml_path");
    EXPECT_EQ(contract.required_fast_path_counters[1], "gemma4_prefill_maintained_fast_ops");
    EXPECT_EQ(contract.required_fast_path_counters[2], "gemma4_decode_maintained_fast_ops");
    ASSERT_EQ(contract.forbidden_fast_path_reasons.size(), 1u);
    EXPECT_EQ(contract.forbidden_fast_path_reasons[0], "gemma4_arm_native_moe_prefill_quality_failed");
    ASSERT_NE(FindRequirement(contract, "moe_gate.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "moe_gate.weight")->tensor_role,
              densecore::runtime::DenseCoreTensorRole::MoERouter);
    EXPECT_EQ(FindRequirement(contract, "moe_gate.weight")->fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    ASSERT_NE(FindRequirement(contract, "ffn_gate_up_exps.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "ffn_gate_up_exps.weight")->semantic_op,
              densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);

    const std::string contract_formatted = densecore::models::FormatModelExecutionContract(contract);
    EXPECT_NE(contract_formatted.find("fast_path_class=gemma4_moe"), std::string::npos);
    EXPECT_NE(contract_formatted.find("requires_fallback_free_fast_path=true"), std::string::npos);
    EXPECT_NE(contract_formatted.find("requires_native_moe_fast_path=false"), std::string::npos);
    EXPECT_NE(contract_formatted.find("required_fast_path_counters=[target_no_ggml_path,"
                                      "gemma4_prefill_maintained_fast_ops,gemma4_decode_maintained_fast_ops]"),
              std::string::npos);
    EXPECT_NE(contract_formatted.find("forbidden_fast_paths=[gemma4_arm_native_moe_prefill_quality_failed]"),
              std::string::npos);
    EXPECT_NE(contract_formatted.find("moe_gate.weight:moe_router"), std::string::npos);

    ggml_free(ctx);
}

TEST(DecoderModelSpec, Qwen36HybridMoEResolvesReusableSoftmaxContract) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 2;
    model.hparams.n_experts = 256;
    model.hparams.n_experts_used = 8;
    model.hparams.n_head_kv = 4;
    model.hybrid_layer_is_ssm = {1, 0};
    model.layers.resize(2);
    model.layers[0].is_moe = true;
    model.layers[0].experts.resize(256);

    const auto spec = densecore::models::BuildDecoderModelSpec(&model);
    ASSERT_EQ(spec.layers.size(), 2u);
    EXPECT_EQ(spec.runtime_topology, densecore::models::DecoderRuntimeTopology::HybridSSMMoE);
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::HybridSSMMixer));
    EXPECT_TRUE(
        densecore::models::DecoderModelSpecHasSpecialization(spec, densecore::models::DecoderSpecializationKind::MoE));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::PrefillLastLogits));
    EXPECT_TRUE(spec.has_hybrid_ssm_mixer);
    EXPECT_TRUE(spec.layers[0].attention.has_hybrid_ssm_mixer);
    EXPECT_FALSE(spec.layers[1].attention.has_hybrid_ssm_mixer);
    EXPECT_EQ(spec.layers[0].ffn.router, densecore::models::DecoderMoERouter::SoftmaxTopK);
    EXPECT_EQ(spec.layers[0].ffn.activation, densecore::models::DecoderActivation::Silu);
    EXPECT_EQ(spec.layers[0].ffn.num_experts, 256);
    EXPECT_EQ(spec.layers[0].ffn.top_k, 8);
    EXPECT_EQ(spec.output.prefill_logits_policy, densecore::models::DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn);
    EXPECT_TRUE(HasOp(spec.layers[0], densecore::models::DecoderSemanticOpKind::HybridSSMMixer));
    EXPECT_FALSE(HasOp(spec.layers[0], densecore::models::DecoderSemanticOpKind::AttentionCore));
}

TEST(ModelExecutionContract, Qwen36HybridSSMMoEDeclaresLayerStateAndRebindContract) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 2;
    model.hparams.n_embd = 2048;
    model.hparams.n_experts = 256;
    model.hparams.n_experts_used = 8;
    model.ssm_inner_size = 2048;
    model.ssm_group_count = 16;
    model.ssm_state_size = 128;
    model.ssm_time_step_rank = 16;
    model.ssm_conv_kernel = 4;
    model.hybrid_layer_is_ssm = {1, 0};
    model.layers.resize(2);
    model.layers[0].is_moe = true;
    model.layers[0].experts.resize(256);

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    EXPECT_EQ(contract.decoder_runtime_topology, densecore::models::DecoderRuntimeTopology::HybridSSMMoE);
    EXPECT_TRUE(contract.has_hybrid_ssm_mixer);
    EXPECT_TRUE(contract.has_moe);
    EXPECT_TRUE(contract.has_stateful_custom_ops);
    EXPECT_TRUE(contract.requires_fallback_free_fast_path);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::QwenHybridSSMMoE);
    EXPECT_TRUE(contract.requires_native_moe_fast_path);
    EXPECT_NE(std::find(contract.required_fast_path_counters.begin(), contract.required_fast_path_counters.end(),
                        "native_moe_fast_w1w3_used_ops"),
              contract.required_fast_path_counters.end());
    EXPECT_NE(std::find(contract.required_fast_path_counters.begin(), contract.required_fast_path_counters.end(),
                        "ssm_delta_calls"),
              contract.required_fast_path_counters.end());
    EXPECT_GE(contract.native_moe_max_direct_tokens, 4096);
    EXPECT_TRUE(densecore::models::ModelExecutionContractAllowsDecodeGraphCache(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresDecodeGraphRuntimeRebind(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresFallbackFreeFastPath(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract));
    EXPECT_GE(densecore::models::ModelExecutionContractNativeMoEMaxDirectTokens(contract), 4096);
    ASSERT_NE(FindRequirement(contract, "attn_qkv.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "attn_qkv.weight")->tensor_role,
              densecore::runtime::DenseCoreTensorRole::HybridSSMQkv);
    EXPECT_EQ(FindRequirement(contract, "attn_qkv.weight")->semantic_op,
              densecore::runtime::DenseCoreSemanticOp::HybridSsmMixer);
    EXPECT_EQ(FindRequirement(contract, "attn_qkv.weight")->fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    ASSERT_NE(FindRequirement(contract, "ssm_out.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "ssm_out.weight")->tensor_role,
              densecore::runtime::DenseCoreTensorRole::SSMOut);
    ASSERT_EQ(contract.layers.size(), 2u);
    ASSERT_EQ(contract.rebind_descriptors.size(), 2u);

    const auto& layer = contract.layers[0];
    EXPECT_TRUE(layer.has_hybrid_ssm_mixer);
    EXPECT_TRUE(layer.has_moe);
    EXPECT_EQ(layer.ssm_ordinal, 0);
    EXPECT_EQ(layer.runtime_state_shape.kind, densecore::models::ExecutionRuntimeStateKind::HybridSSM);
    EXPECT_EQ(layer.runtime_state_shape.conv_channels, 2048 + 2 * 16 * 128);
    EXPECT_EQ(layer.runtime_state_shape.kernel_size, 4);
    EXPECT_EQ(layer.runtime_state_shape.n_heads, 16);
    EXPECT_EQ(layer.runtime_state_shape.head_dim, 128);
    EXPECT_EQ(layer.runtime_state_shape.state_size, 128);
    EXPECT_EQ(layer.moe_router, densecore::models::DecoderMoERouter::SoftmaxTopK);
    EXPECT_EQ(layer.moe_top_k, 8);
    EXPECT_EQ(layer.moe_num_experts, 256);
    EXPECT_EQ(layer.moe_expert_layout, densecore::models::ExecutionMoEExpertLayoutKind::SeparateExpertMaps);
    EXPECT_EQ(layer.rebind_descriptors.size(), 2u);
    EXPECT_EQ(layer.rebind_descriptors[0].op_kind, densecore::models::ExecutionCustomOpRebindKind::HybridSSMConv1D);
    EXPECT_EQ(layer.rebind_descriptors[1].op_kind, densecore::models::ExecutionCustomOpRebindKind::HybridSSMDelta);

    const std::string formatted = densecore::models::FormatModelExecutionContract(contract);
    EXPECT_NE(formatted.find("requires_rebind=true"), std::string::npos);
    EXPECT_NE(formatted.find("fast_path_class=qwen_hybrid_ssm_moe"), std::string::npos);
    EXPECT_NE(formatted.find("requires_fallback_free_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("requires_native_moe_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("attn_qkv.weight:hybrid_ssm_qkv"), std::string::npos);
    EXPECT_NE(formatted.find("native_moe_max_direct_tokens=4096"), std::string::npos);
    EXPECT_NE(formatted.find("hybrid_ssm_conv1d@layer0"), std::string::npos);
    EXPECT_NE(formatted.find("hybrid_ssm_delta@layer0"), std::string::npos);
}

TEST(ModelExecutionContract, QwenDenseDeclaresFallbackFreeFastPathWithoutNativeMoERequirement) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = false;
    model.hparams.n_layer = 2;
    model.hparams.n_embd = 4096;
    model.hparams.n_experts = 0;
    model.layers.resize(2);

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    EXPECT_FALSE(contract.has_moe);
    EXPECT_FALSE(contract.has_hybrid_ssm_mixer);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::QwenDense);
    EXPECT_TRUE(contract.requires_fallback_free_fast_path);
    EXPECT_FALSE(contract.requires_native_moe_fast_path);
    ASSERT_EQ(contract.required_fast_path_counters.size(), 1u);
    EXPECT_EQ(contract.required_fast_path_counters[0], "target_no_ggml_path");
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresFallbackFreeFastPath(contract));
    EXPECT_FALSE(densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract));

    const std::string formatted = densecore::models::FormatModelExecutionContract(contract);
    EXPECT_NE(formatted.find("fast_path_class=qwen_dense"), std::string::npos);
    EXPECT_NE(formatted.find("requires_fallback_free_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("requires_native_moe_fast_path=false"), std::string::npos);
}

TEST(ModelExecutionContract, QwenHybridSSMDenseDeclaresFallbackFreeRebindWithoutNativeMoERequirement) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 2;
    model.hparams.n_embd = 2048;
    model.hparams.n_experts = 0;
    model.ssm_inner_size = 2048;
    model.ssm_group_count = 16;
    model.ssm_state_size = 128;
    model.ssm_time_step_rank = 16;
    model.ssm_conv_kernel = 4;
    model.hybrid_layer_is_ssm = {1, 0};
    model.layers.resize(2);

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    EXPECT_EQ(contract.decoder_runtime_topology, densecore::models::DecoderRuntimeTopology::HybridSSM);
    EXPECT_TRUE(contract.has_hybrid_ssm_mixer);
    EXPECT_FALSE(contract.has_moe);
    EXPECT_TRUE(contract.has_stateful_custom_ops);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::QwenHybridSSMDense);
    EXPECT_TRUE(contract.requires_fallback_free_fast_path);
    EXPECT_FALSE(contract.requires_native_moe_fast_path);
    EXPECT_NE(std::find(contract.required_fast_path_counters.begin(), contract.required_fast_path_counters.end(),
                        "target_no_ggml_path"),
              contract.required_fast_path_counters.end());
    EXPECT_NE(std::find(contract.required_fast_path_counters.begin(), contract.required_fast_path_counters.end(),
                        "ssm_conv1d_calls"),
              contract.required_fast_path_counters.end());
    EXPECT_EQ(contract.native_moe_max_direct_tokens, 0);
    EXPECT_TRUE(densecore::models::ModelExecutionContractAllowsDecodeGraphCache(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresDecodeGraphRuntimeRebind(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresFallbackFreeFastPath(contract));
    EXPECT_FALSE(densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract));
    ASSERT_EQ(contract.rebind_descriptors.size(), 2u);

    const std::string formatted = densecore::models::FormatModelExecutionContract(contract);
    EXPECT_NE(formatted.find("requires_rebind=true"), std::string::npos);
    EXPECT_NE(formatted.find("fast_path_class=qwen_hybrid_ssm_dense"), std::string::npos);
    EXPECT_NE(formatted.find("requires_fallback_free_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("requires_native_moe_fast_path=false"), std::string::npos);
    EXPECT_NE(formatted.find("native_moe_max_direct_tokens=0"), std::string::npos);
    EXPECT_NE(formatted.find("hybrid_ssm_conv1d@layer0"), std::string::npos);
    EXPECT_NE(formatted.find("hybrid_ssm_delta@layer0"), std::string::npos);
}

TEST(DecoderModelSpec, Qwen36GroupedMetadataStillUsesLlamaCppSoftmaxRouter) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 1;
    model.hparams.n_experts = 256;
    model.hparams.n_experts_used = 8;
    model.moe_n_group = 8;
    model.moe_topk_group = 4;
    model.layers.resize(1);
    model.layers[0].is_moe = true;

    const auto spec = densecore::models::BuildDecoderModelSpec(&model);
    ASSERT_EQ(spec.layers.size(), 1u);
    EXPECT_EQ(spec.runtime_topology, densecore::models::DecoderRuntimeTopology::HybridSSMMoE);
    EXPECT_FALSE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::GroupedMoERouter));
    EXPECT_EQ(spec.layers[0].ffn.router, densecore::models::DecoderMoERouter::SoftmaxTopK);
}

TEST(DecoderModelSpec, QwenPrefillLastLogitsPolicyDefaultsOnForQwen35AndQwen36) {
    TransformerModel qwen35{};
    qwen35.arch = ModelArch::QWEN35;
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.hparams.n_layer = 1;
    qwen35.layers.resize(1);

    const auto qwen35_spec = densecore::models::BuildDecoderModelSpec(&qwen35);
    EXPECT_EQ(qwen35_spec.runtime_topology, densecore::models::DecoderRuntimeTopology::DenseAttention);
    EXPECT_EQ(qwen35_spec.output.prefill_logits_policy,
              densecore::models::DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn);
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        qwen35_spec, densecore::models::DecoderSpecializationKind::PrefillLastLogits));
    EXPECT_TRUE(densecore::models::ShouldUsePrefillLastLogitsOnly(&qwen35_spec, /*num_seqs=*/1, /*n_tokens=*/8));
    densecore::models::DecoderPrefillRuntimePolicy qwen35_opt_out_policy =
        densecore::models::DefaultDecoderPrefillRuntimePolicy();
    qwen35_opt_out_policy.qwen35_prefill_last_logits_only = false;
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&qwen35_spec, /*num_seqs=*/1, /*n_tokens=*/8,
                                                                   qwen35_opt_out_policy));

    TransformerModel qwen36{};
    qwen36.arch = ModelArch::QWEN35;
    qwen36.variant = ModelVariant::QWEN36;
    qwen36.hparams.n_layer = 1;
    qwen36.layers.resize(1);

    const auto qwen36_spec = densecore::models::BuildDecoderModelSpec(&qwen36);
    EXPECT_EQ(qwen36_spec.runtime_topology, densecore::models::DecoderRuntimeTopology::DenseAttention);
    EXPECT_EQ(qwen36_spec.output.prefill_logits_policy,
              densecore::models::DecoderPrefillLogitsPolicy::LastTokenEnvDefaultOn);
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        qwen36_spec, densecore::models::DecoderSpecializationKind::PrefillLastLogits));
    EXPECT_TRUE(densecore::models::ShouldUsePrefillLastLogitsOnly(&qwen36_spec, /*num_seqs=*/1, /*n_tokens=*/8));

    densecore::models::DecoderPrefillRuntimePolicy opt_out_policy =
        densecore::models::DefaultDecoderPrefillRuntimePolicy();
    opt_out_policy.qwen36_prefill_last_logits_only = false;
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&qwen36_spec, /*num_seqs=*/1, /*n_tokens=*/8,
                                                                   opt_out_policy));
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&qwen36_spec, /*num_seqs=*/2, /*n_tokens=*/8));
}

TEST(DecoderModelSpec, LFM2ShortConvMoEPrefillUsesLastTokenLogitsWithoutEnvGate) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_layer = 2;
    model.hparams.n_experts = 64;
    model.hparams.n_experts_used = 6;
    model.layers.resize(2);
    model.layers[0].is_moe = true;
    model.layers[0].experts.resize(64);
    model.layers[1].is_moe = true;
    model.layers[1].experts.resize(64);

    const auto spec = densecore::models::BuildDecoderModelSpec(&model);
    ASSERT_EQ(spec.layers.size(), 2u);
    EXPECT_EQ(spec.runtime_topology, densecore::models::DecoderRuntimeTopology::DenseAttentionMoE);
    EXPECT_EQ(spec.output.prefill_logits_policy, densecore::models::DecoderPrefillLogitsPolicy::LastTokenForMoE);
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::PrefillLastLogits));
    EXPECT_TRUE(densecore::models::DecoderModelSpecHasSpecialization(
        spec, densecore::models::DecoderSpecializationKind::GroupedMoERouter));
    EXPECT_EQ(spec.layers[0].ffn.router, densecore::models::DecoderMoERouter::GroupedSigmoidTopK);
    EXPECT_EQ(spec.layers[0].ffn.top_k, 6);

    densecore::models::DecoderPrefillRuntimePolicy policy =
        densecore::models::DefaultDecoderPrefillRuntimePolicy();
    policy.qwen35_prefill_last_logits_only = false;
    policy.qwen36_prefill_last_logits_only = false;
    EXPECT_TRUE(densecore::models::ShouldUsePrefillLastLogitsOnly(&spec, /*num_seqs=*/1, /*n_tokens=*/8, policy));
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&spec, /*num_seqs=*/1, /*n_tokens=*/1, policy));
    EXPECT_FALSE(densecore::models::ShouldUsePrefillLastLogitsOnly(&spec, /*num_seqs=*/2, /*n_tokens=*/8, policy));

    const std::string formatted = densecore::models::FormatDecoderModelSpec(spec);
    EXPECT_NE(formatted.find("prefill_logits=last_token_for_moe"), std::string::npos);

    const auto capabilities = densecore::models::ResolveModelGraphCapabilities(&model);
    EXPECT_TRUE(capabilities.has_lfm2_shortconv_mixer);
    EXPECT_TRUE(capabilities.has_moe);
    EXPECT_EQ(capabilities.decoder_runtime_topology, densecore::models::DecoderRuntimeTopology::DenseAttentionMoE);

    const auto graph_resolution = densecore::models::ResolveGraphFamily(&model);
    EXPECT_EQ(graph_resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
}

TEST(ModelExecutionContract, LFM2ShortConvDeclaresConvOrdinalsAndRebindContract) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_layer = 3;
    model.hparams.n_embd = 8;
    model.hparams.n_experts = 64;
    model.hparams.n_experts_used = 6;
    model.lfm2_conv_kernel = 3;
    model.lfm2_layer_is_conv = {1, 0, 1};
    model.lfm2_conv_weight_f32 = {
        std::vector<float>(static_cast<size_t>(8 * 3), 0.1f),
        std::vector<float>(static_cast<size_t>(8 * 3), 0.2f),
    };
    model.layers.resize(3);
    for (auto& layer : model.layers) {
        layer.is_moe = true;
        layer.experts.resize(64);
    }

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    EXPECT_TRUE(contract.has_lfm2_shortconv_mixer);
    EXPECT_TRUE(contract.has_moe);
    EXPECT_TRUE(contract.has_stateful_custom_ops);
    EXPECT_TRUE(contract.requires_fallback_free_fast_path);
    EXPECT_EQ(contract.fast_path_class, densecore::models::ExecutionFastPathClass::LFM2ShortConvMoE);
    EXPECT_TRUE(contract.requires_native_moe_fast_path);
    EXPECT_NE(std::find(contract.required_fast_path_counters.begin(), contract.required_fast_path_counters.end(),
                        "lfm2_shortconv_sequence_fast_used_ops"),
              contract.required_fast_path_counters.end());
    EXPECT_GE(contract.native_moe_max_direct_tokens, 4096);
    EXPECT_TRUE(densecore::models::ModelExecutionContractAllowsDecodeGraphCache(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresDecodeGraphRuntimeRebind(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresFallbackFreeFastPath(contract));
    EXPECT_TRUE(densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract));
    EXPECT_GE(densecore::models::ModelExecutionContractNativeMoEMaxDirectTokens(contract), 4096);
    ASSERT_EQ(contract.layers.size(), 3u);
    ASSERT_EQ(contract.rebind_descriptors.size(), 2u);
    ASSERT_NE(FindRequirement(contract, "shortconv_in_proj.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "shortconv_in_proj.weight")->semantic_op,
              densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer);
    EXPECT_EQ(FindRequirement(contract, "shortconv_in_proj.weight")->fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    ASSERT_NE(FindRequirement(contract, "ffn_down_exps.weight"), nullptr);
    EXPECT_EQ(FindRequirement(contract, "ffn_down_exps.weight")->decode_kernel,
              densecore::runtime::DenseCoreKernelFamily::DenseCoreQwenMoeDirect);

    EXPECT_TRUE(contract.layers[0].has_lfm2_shortconv_mixer);
    EXPECT_EQ(contract.layers[0].conv_ordinal, 0);
    EXPECT_EQ(contract.layers[0].runtime_state_shape.kind, densecore::models::ExecutionRuntimeStateKind::LFM2ShortConv);
    EXPECT_EQ(contract.layers[0].runtime_state_shape.conv_channels, 8);
    EXPECT_EQ(contract.layers[0].runtime_state_shape.kernel_size, 3);
    EXPECT_EQ(contract.layers[0].runtime_state_shape.expected_conv_state_elements, 16u);
    EXPECT_EQ(contract.layers[0].tensor_ownership, densecore::models::ExecutionTensorOwnership::LoaderCanonicalBuffer);
    EXPECT_FALSE(contract.layers[1].has_lfm2_shortconv_mixer);
    EXPECT_TRUE(contract.layers[2].has_lfm2_shortconv_mixer);
    EXPECT_EQ(contract.layers[2].conv_ordinal, 1);
    EXPECT_EQ(contract.layers[2].tensor_ownership, densecore::models::ExecutionTensorOwnership::LoaderCanonicalBuffer);
    EXPECT_EQ(contract.layers[0].moe_router, densecore::models::DecoderMoERouter::GroupedSigmoidTopK);

    const std::string formatted = densecore::models::FormatModelExecutionContract(contract);
    EXPECT_NE(formatted.find("fast_path_class=lfm2_shortconv_moe"), std::string::npos);
    EXPECT_NE(formatted.find("requires_fallback_free_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("requires_native_moe_fast_path=true"), std::string::npos);
    EXPECT_NE(formatted.find("native_moe_max_direct_tokens=4096"), std::string::npos);
    EXPECT_NE(formatted.find("lfm2_shortconv@layer0"), std::string::npos);
    EXPECT_NE(formatted.find("lfm2_shortconv@layer2"), std::string::npos);
    EXPECT_NE(formatted.find("shortconv_in_proj.weight:shortconv_in"), std::string::npos);
}

TEST(ModelExecutionContract, SemanticGraphCoverageValidatesLoadedLFM2KernelMatrix) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_layer = 1;
    model.hparams.n_embd = 256;
    model.hparams.n_experts = 4;
    model.hparams.n_experts_used = 2;
    model.lfm2_conv_kernel = 3;
    model.lfm2_layer_is_conv = {1};
    model.lfm2_conv_weight_f32 = {std::vector<float>(static_cast<size_t>(256 * 3), 0.1f)};
    model.layers.resize(1);
    auto& layer = model.layers[0];
    layer.is_moe = true;
    layer.Set(model_keys::kShortConvInProj, NewTypedTensor2D(ctx, GGML_TYPE_Q4_K, 256, 64));
    layer.Set(model_keys::kShortConvOutProj, NewTypedTensor2D(ctx, GGML_TYPE_Q4_K, 256, 64));
    layer.Set(model_keys::kMoeGate, NewTypedTensor2D(ctx, GGML_TYPE_F32, 256, 4));
    layer.Set("ffn_gate_up_exps.weight", NewTypedTensor2D(ctx, GGML_TYPE_Q5_K, 256, 512));
    layer.Set("ffn_down_exps.weight", NewTypedTensor2D(ctx, GGML_TYPE_Q4_K, 512, 256));

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);
    ASSERT_FALSE(contract.semantic_graph_nodes.empty());
    EXPECT_EQ(contract.semantic_graph_nodes.size(), 10u);

    densecore::runtime::HostKernelCapabilities caps{};
    caps.arm_sve2 = true;
    caps.q4k_true_batched = true;
    const auto coverage = densecore::models::ValidateFallbackFreeKernelCoverage(&model, contract, caps);
    EXPECT_TRUE(coverage.required);
    EXPECT_TRUE(coverage.fallback_free) << densecore::models::FormatKernelCoverageValidationResult(coverage);
    EXPECT_EQ(coverage.covered_node_count, contract.semantic_graph_nodes.size());
    EXPECT_EQ(coverage.graph_node_count, contract.semantic_graph_nodes.size());
    EXPECT_TRUE(coverage.missing_kernel_reasons.empty());
    EXPECT_NE(densecore::models::FormatModelExecutionContract(contract).find("semantic_graph_nodes=10"),
              std::string::npos);
    EXPECT_NE(densecore::models::FormatKernelCoverageValidationResult(coverage).find("covered=10/10"),
              std::string::npos);

    ggml_free(ctx);
}

TEST(ModelExecutionContract, SemanticGraphCoverageRejectsUnsupportedLoadedTargetTensor) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_layer = 1;
    model.hparams.n_embd = 256;
    model.hparams.n_experts = 4;
    model.hparams.n_experts_used = 2;
    model.lfm2_conv_kernel = 3;
    model.lfm2_layer_is_conv = {1};
    model.lfm2_conv_weight_f32 = {std::vector<float>(static_cast<size_t>(256 * 3), 0.1f)};
    model.layers.resize(1);
    auto& layer = model.layers[0];
    layer.is_moe = true;
    layer.Set(model_keys::kShortConvInProj, NewTypedTensor2D(ctx, GGML_TYPE_F16, 256, 64));
    layer.Set(model_keys::kShortConvOutProj, NewTypedTensor2D(ctx, GGML_TYPE_Q4_K, 256, 64));
    layer.Set(model_keys::kMoeGate, NewTypedTensor2D(ctx, GGML_TYPE_F32, 256, 4));
    layer.Set("ffn_gate_up_exps.weight", NewTypedTensor2D(ctx, GGML_TYPE_Q5_K, 256, 512));
    layer.Set("ffn_down_exps.weight", NewTypedTensor2D(ctx, GGML_TYPE_Q4_K, 512, 256));

    const auto contract = densecore::models::BuildModelExecutionContract(&model);
    ASSERT_TRUE(contract.valid) << densecore::models::FormatModelExecutionContract(contract);

    densecore::runtime::HostKernelCapabilities caps{};
    caps.arm_sve2 = true;
    caps.q4k_true_batched = true;
    const auto coverage = densecore::models::ValidateFallbackFreeKernelCoverage(&model, contract, caps);
    EXPECT_TRUE(coverage.required);
    EXPECT_FALSE(coverage.fallback_free);
    ASSERT_FALSE(coverage.missing_kernel_reasons.empty());
    EXPECT_NE(coverage.missing_kernel_reasons.front().find("missing_kernel:lfm2_shortconv_mixer/f16/"),
              std::string::npos);

    ggml_free(ctx);
}
