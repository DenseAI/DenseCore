#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/model_types.h"
#include "models/model_inference_policy.h"
#include "models/model_prompt_templates.h"

TEST(ModelDescriptorTest, ResolveGemma4AliasPreservesRuntimeFlags) {
    const auto resolved = densecore::models::ResolveModelDescriptor("gemma4_text");

    ASSERT_TRUE(resolved.known);
    EXPECT_EQ(resolved.arch, ModelArch::GEMMA);
    EXPECT_EQ(resolved.variant, ModelVariant::GEMMA4);
    EXPECT_TRUE(resolved.arch_flags.is_gemma4);
    EXPECT_TRUE(resolved.arch_flags.requires_q_norm);
    EXPECT_TRUE(resolved.arch_flags.requires_k_norm);
}

TEST(ModelDescriptorTest, ResolveGemma4FromMetadataHintsUpgradesAmbiguousGemmaArch) {
    densecore::models::ModelDetectionHints hints;
    hints.has_gemma4_metadata = true;

    bool used_hint_upgrade = false;
    const auto resolved = densecore::models::ResolveModelDescriptorWithHints("gemma", hints, &used_hint_upgrade);

    ASSERT_TRUE(resolved.known);
    EXPECT_TRUE(used_hint_upgrade);
    EXPECT_EQ(resolved.arch, ModelArch::GEMMA);
    EXPECT_EQ(resolved.variant, ModelVariant::GEMMA4);
    EXPECT_TRUE(resolved.arch_flags.is_gemma4);
}

TEST(ModelDescriptorTest, PlainGemmaHintsDoNotMisclassifyModel) {
    densecore::models::ModelDetectionHints hints;

    bool used_hint_upgrade = true;
    const auto resolved = densecore::models::ResolveModelDescriptorWithHints("gemma", hints, &used_hint_upgrade);

    ASSERT_TRUE(resolved.known);
    EXPECT_FALSE(used_hint_upgrade);
    EXPECT_EQ(resolved.variant, ModelVariant::GEMMA);
    EXPECT_FALSE(resolved.arch_flags.is_gemma4);
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

TEST(ModelDescriptorTest, Qwen35NoThinkingBlocklistIncludesNoThinkDirectives) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.token_to_id["<think>"] = 101;
    model.token_to_id["/no_think"] = 102;
    model.token_to_id["/nothink"] = 103;

    Request req{};
    densecore::models::ConfigureQwenReasoningTokenBlocklistForModel(&model, &req);

    EXPECT_NE(std::find(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), 101),
              req.disallowed_token_ids.end());
    EXPECT_NE(std::find(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), 102),
              req.disallowed_token_ids.end());
    EXPECT_NE(std::find(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), 103),
              req.disallowed_token_ids.end());
}

TEST(ModelDescriptorTest, ResolveQwen36AliasReusesHybridSsmRuntimeWithVariantOverride) {
    const auto resolved = densecore::models::ResolveModelDescriptor("Qwen3.6-35B-A3B");

    ASSERT_TRUE(resolved.known);
    EXPECT_EQ(resolved.arch, ModelArch::QWEN35);
    EXPECT_EQ(resolved.variant, ModelVariant::QWEN36);
    EXPECT_TRUE(resolved.arch_flags.is_hybrid_ssm);
    EXPECT_TRUE(resolved.arch_flags.requires_q_norm);
    EXPECT_TRUE(resolved.arch_flags.requires_k_norm);
}

TEST(ModelDescriptorTest, ResolveQwen3NextAliasReusesHybridSsmRuntime) {
    const auto resolved = densecore::models::ResolveModelDescriptor("qwen3next");

    ASSERT_TRUE(resolved.known);
    EXPECT_EQ(resolved.arch, ModelArch::QWEN35);
    EXPECT_EQ(resolved.variant, ModelVariant::QWEN3NEXT);
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

TEST(ModelDescriptorTest, Qwen36TokenizerMetadataUsesQwenByteBpeFamily) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.tokenizer_type = "qwen3.6";

    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::QWEN_BYTE_BPE);
}

TEST(ModelDescriptorTest, Qwen36TokenizerAliasIsRecognizedWithoutCompatibilityWarning) {
    EXPECT_TRUE(densecore::models::IsKnownTokenizerModel("qwen3.6"));
    EXPECT_TRUE(densecore::models::IsKnownTokenizerModel("qwen3_5_moe"));
}

TEST(ModelDescriptorTest, BertTokenizerMetadataUsesWordPieceFamily) {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.tokenizer_type = "bert";

    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::BERT_WORDPIECE);
    EXPECT_TRUE(densecore::models::IsKnownTokenizerModel("bert"));
    EXPECT_STREQ(densecore::models::TokenizerFamilyName(densecore::models::TokenizerFamily::BERT_WORDPIECE),
                 "bert_wordpiece");
}

TEST(ModelDescriptorTest, JinaV2TokenizerMetadataUsesWordPieceFamily) {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.tokenizer_type = "jina-v2-en";

    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::BERT_WORDPIECE);
    EXPECT_TRUE(densecore::models::IsKnownTokenizerModel("jina-v2-en"));
}

TEST(ModelDescriptorTest, NomicBertArchAliasResolvesToBertDescriptor) {
    auto resolved = densecore::models::ResolveModelDescriptor("nomic-bert");

    EXPECT_TRUE(resolved.known);
    ASSERT_NE(resolved.descriptor, nullptr);
    EXPECT_EQ(resolved.descriptor->arch, ModelArch::BERT);
    EXPECT_EQ(resolved.descriptor->tokenizer_family, densecore::models::TokenizerFamily::BERT_WORDPIECE);
}

TEST(ModelDescriptorTest, T5TokenizerMetadataUsesSentencePieceFamily) {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.tokenizer_type = "t5";

    EXPECT_EQ(densecore::models::ResolveTokenizerFamily(&model),
              densecore::models::TokenizerFamily::LLAMA_SENTENCEPIECE);
    EXPECT_TRUE(densecore::models::IsKnownTokenizerModel("t5"));
}

TEST(ModelDescriptorTest, ChatTemplateMetadataOverridesPromptFamily) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.tokenizer_type = "llama";
    model.chat_template = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n";

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::TURN_TAGS);
}

TEST(ModelDescriptorTest, Gemma4PromptTemplatePrefersTurnTagsForGenericGemmaTokenizerMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.tokenizer_type = "gemma";

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::TURN_TAGS);
}

TEST(ModelDescriptorTest, Gemma4PromptTemplateDefaultsToTurnTagsWithoutChatMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::TURN_TAGS);
}

TEST(ModelDescriptorTest, PlainGemmaPromptTemplateRemainsPlainForGenericMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.tokenizer_type = "gemma";

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::PLAIN);
}

TEST(ModelDescriptorTest, ExplicitChatTemplateStillOverridesGemma4DefaultPromptFamily) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.chat_template = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

    EXPECT_EQ(densecore::models::ResolvePromptTemplateFamily(&model),
              densecore::models::PromptTemplateFamily::CHATML);
}

TEST(ModelDescriptorTest, Gemma4TextPreservesAttentionLogitSoftcapMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 50.0f), 50.0f);
}

TEST(ModelDescriptorTest, Gemma4TextNormalizesInvalidAttentionLogitSoftcapMetadataToDefault) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 0.0f), 50.0f);
    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, -3.0f), 50.0f);
}

TEST(ModelDescriptorTest, NonGemmaSoftcapMetadataIsUnchanged) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    EXPECT_FLOAT_EQ(densecore::models::SanitizeAttentionLogitSoftcapForLoad(&model, 50.0f), 50.0f);
}

TEST(ModelDescriptorTest, DenseDecoderGraphFamilySelectsGenericDecoder) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    const auto capabilities = densecore::models::ResolveModelGraphCapabilities(&model);
    const auto resolution = densecore::models::ResolveGraphFamily(&model);

    EXPECT_EQ(capabilities.topology, densecore::models::GraphTopology::DECODER_ONLY);
    EXPECT_TRUE(capabilities.has_dense_attention);
    EXPECT_EQ(resolution.preferred_family, densecore::models::GraphFamily::DecoderDenseAttention);
    EXPECT_TRUE(resolution.fallback_chain.empty());
}

TEST(ModelDescriptorTest, Qwen35GraphFamilySelectsHybridSsm) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.arch_flags.requires_q_norm = true;
    model.arch_flags.requires_k_norm = true;

    const auto resolution = densecore::models::ResolveGraphFamily(&model);

    EXPECT_EQ(resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    ASSERT_EQ(resolution.fallback_chain.size(), 1u);
    EXPECT_EQ(resolution.fallback_chain[0], densecore::models::GraphFamily::DecoderDenseAttention);
}

TEST(ModelDescriptorTest, Gemma4GraphFamilySelectsSlidingSharedKv) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_is_sliding = {1, 0};
    model.gemma4_layer_kv_source = {0, 0};
    model.gemma4_layer_n_head_kv = {4, 2};
    model.hparams.n_head_kv = 4;

    const auto resolution = densecore::models::ResolveGraphFamily(&model);

    EXPECT_EQ(resolution.preferred_family, densecore::models::GraphFamily::DecoderSlidingWindowSharedKV);
    ASSERT_EQ(resolution.fallback_chain.size(), 1u);
    EXPECT_EQ(resolution.fallback_chain[0], densecore::models::GraphFamily::DecoderDenseAttention);
}

TEST(ModelDescriptorTest, DenseDecoderAdmissionFailsClosedForUnsupportedMoe) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.hparams.n_experts = 8;

    const auto admission = densecore::models::AdmitGraphBuilder(
        densecore::models::ResolveGraphFamily(&model),
        densecore::models::MakeDenseDecoderGenericSupport("unit-test-generic"));

    EXPECT_FALSE(admission.admitted);
    EXPECT_NE(admission.Summary().find("MoE routing semantics"), std::string::npos);
}
