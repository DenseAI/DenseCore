#include <gtest/gtest.h>

#include "densecore/models/model_descriptor.h"
#include "model_types.h"
#include "models/model_inference_policy.h"

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

TEST(ModelDescriptorTest, TokenizerMetadataOverridesDescriptorFamily) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN3;
    model.tokenizer_type = "qwen35";

    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::QWEN35_UNICODE_BPE);
}

TEST(ModelDescriptorTest, ChatTemplateMetadataOverridesPromptFamily) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.tokenizer_type = "llama";
    model.chat_template = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n";

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::TURN_TAGS);
}

TEST(ModelDescriptorTest, Gemma4TextIgnoresAttentionLogitSoftcapMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 50.0f), 0.0f);
    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 0.0f), 0.0f);
}

TEST(ModelDescriptorTest, NonGemmaSoftcapMetadataIsUnchanged) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 50.0f), 50.0f);
}
