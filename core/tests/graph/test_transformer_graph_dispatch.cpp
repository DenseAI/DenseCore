#include <gtest/gtest.h>

#include "densecore/models/transformer_graph_builder.h"
#include "densecore/runtime/inference.h"
#include "densecore/models/model_types.h"

namespace {

TEST(BuildTransformerGraphDispatchTest, DenseDecoderSelectsRegistryBuilderRoute) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderDenseAttention);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);
    EXPECT_EQ(plan.registry_builder_key, densecore::kDenseDecoderGenericBuilderKey);
    EXPECT_EQ(plan.selected_builder_name, "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, Qwen35SelectsInlineHybridSsmRoute) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.arch_flags.requires_q_norm = true;
    model.arch_flags.requires_k_norm = true;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineHybridSSM);
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, Gemma4SelectsInlineSlidingSharedKvRoute) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_is_sliding = {1, 0};
    model.gemma4_layer_kv_source = {0, 0};
    model.gemma4_layer_n_head_kv = {4, 2};
    model.hparams.n_head_kv = 4;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderSlidingWindowSharedKV);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV);
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, UnsupportedMultimodalProjectionFailsClosed) {
    TransformerModel model{};
    model.arch = ModelArch::LLAVA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::MultimodalProjectedDecoder);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::Reject);
    EXPECT_NE(plan.debug_reason.find("does not implement graph family MultimodalProjectedDecoder"), std::string::npos);
}

TEST(BuildTransformerGraphDispatchTest, Qwen3DoesNotUseLegacyArchOnlyRegistryFallback) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN3;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderDenseAttention);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineDenseAttention);
    EXPECT_TRUE(plan.registry_builder_key.empty());
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, ExactRegistryKeyIsPresentAndLlamaCompatibilityKeyIsNotUsed) {
    auto exact = densecore::TransformerGraphRegistry::Instance().GetBuilderExact(densecore::kDenseDecoderGenericBuilderKey);
    auto legacy = densecore::TransformerGraphRegistry::Instance().GetBuilderExact("llama");
    auto exact_descriptor =
        densecore::TransformerGraphRegistry::Instance().GetBuilderDescriptorExact(densecore::kDenseDecoderGenericBuilderKey);
    auto legacy_descriptor = densecore::TransformerGraphRegistry::Instance().GetBuilderDescriptorExact("llama");

    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(std::string(exact->Name()), "densecore_inline_dense_decoder");
    ASSERT_TRUE(exact_descriptor.has_value());
    EXPECT_EQ(exact_descriptor->display_name, "densecore_inline_dense_decoder");
    EXPECT_EQ(legacy, nullptr);
    EXPECT_FALSE(legacy_descriptor.has_value());
}

TEST(BuildTransformerGraphDispatchTest, ExactBuilderIdentityDoesNotDependOnFallbackOrder) {
    auto builder = densecore::TransformerGraphRegistry::Instance().GetBuilderExact(densecore::kDenseDecoderGenericBuilderKey);
    ASSERT_NE(builder, nullptr);
    EXPECT_EQ(std::string(builder->Name()), "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, UnknownTopologyFailsClosed) {
    TransformerModel model{};
    model.arch = ModelArch::VIT;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.capabilities.topology, densecore::models::GraphTopology::UNKNOWN);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::Reject);
}

TEST(BuildTransformerGraphDispatchTest, ExecutionHelperInstantiatesExactBuilderFromPlan) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);
    ASSERT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);

    std::string error_reason;
    auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &error_reason);

    ASSERT_NE(builder, nullptr);
    EXPECT_TRUE(error_reason.empty());
    EXPECT_EQ(std::string(builder->Name()), "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, ExecutionHelperFailsClosedWhenExactKeyIsUnavailable) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);
    ASSERT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);
    plan.registry_builder_key = "llama";

    std::string error_reason;
    auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &error_reason);

    EXPECT_EQ(builder, nullptr);
    EXPECT_NE(error_reason.find("exact registry builder key 'llama' is not available for execution"),
              std::string::npos);
}

}  // namespace
