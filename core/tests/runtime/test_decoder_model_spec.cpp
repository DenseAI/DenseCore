#include <gtest/gtest.h>

#include <ggml.h>

#include <algorithm>
#include <vector>

#include "densecore/models/decoder_model_spec.h"

namespace {

ggml_tensor* NewTensor2D(ggml_context* ctx, int rows, int cols) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
}

bool HasOp(const densecore::models::DecoderLayerSpec& layer, densecore::models::DecoderSemanticOpKind kind) {
    return std::any_of(layer.semantic_ops.begin(), layer.semantic_ops.end(),
                       [kind](const densecore::models::DecoderSemanticOp& op) { return op.kind == kind; });
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
    layer0.is_moe = true;
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
    EXPECT_TRUE(shared.attention.reads_shared_kv);
    EXPECT_FALSE(shared.attention.requires_k_norm);
    EXPECT_EQ(shared.attention.kv_source_layer, 0);
    EXPECT_EQ(shared.attention.kv_head_count, 2);
    EXPECT_EQ(shared.attention.rope_kind, densecore::models::DecoderRopeKind::Proportional);
    EXPECT_TRUE(HasOp(shared, densecore::models::DecoderSemanticOpKind::SharedKVRead));
    EXPECT_TRUE(HasOp(shared, densecore::models::DecoderSemanticOpKind::DenseFfn));

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

TEST(DecoderModelSpec, Qwen36GroupedRouterIsDeclaredBySpecNotRuntimeBranch) {
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
    EXPECT_EQ(spec.layers[0].ffn.router, densecore::models::DecoderMoERouter::GroupedSigmoidTopK);
}
