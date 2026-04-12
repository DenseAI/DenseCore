#include <gtest/gtest.h>

#include "densecore/models/model_descriptor.h"
#include "model_types.h"

TEST(ModelDescriptorTest, ResolveGemma4AliasPreservesRuntimeFlags) {
    const auto resolved = densecore::models::ResolveModelDescriptor("gemma4_text");

    ASSERT_TRUE(resolved.known);
    EXPECT_EQ(resolved.arch, ModelArch::GEMMA);
    EXPECT_EQ(resolved.variant, ModelVariant::GEMMA4);
    EXPECT_TRUE(resolved.arch_flags.is_gemma4);
    EXPECT_TRUE(resolved.arch_flags.requires_q_norm);
    EXPECT_TRUE(resolved.arch_flags.requires_k_norm);
}

TEST(ModelDescriptorTest, ResolveQwen35AliasMarksHybridSSM) {
    const auto resolved = densecore::models::ResolveModelDescriptor("qwen3_5_moe_text");

    ASSERT_TRUE(resolved.known);
    EXPECT_EQ(resolved.arch, ModelArch::QWEN35);
    EXPECT_EQ(resolved.variant, ModelVariant::QWEN35);
    EXPECT_TRUE(resolved.arch_flags.is_hybrid_ssm);
    EXPECT_TRUE(resolved.arch_flags.requires_q_norm);
    EXPECT_TRUE(resolved.arch_flags.requires_k_norm);
}

TEST(ModelDescriptorTest, ManualModelFallsBackToArchAndFlags) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    const auto& descriptor = densecore::models::DescribeModel(&model);

    EXPECT_EQ(descriptor.variant, ModelVariant::GEMMA4);
    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::TURN_TAGS);
    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::GEMMA_SENTENCEPIECE);
}
