#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "model_types.h"

std::string DenseCoreTestOnlyApplyAutoChatTemplate(const TransformerModel* model, const std::string& prompt);
bool DenseCoreTestOnlyPromptStartsInThinkBlock(const std::string& prompt);
std::vector<int> DenseCoreTestOnlyQwenReasoningBlocklist(const TransformerModel* model);
std::vector<int> DenseCoreTestOnlyGemma4TextBlocklist(const TransformerModel* model);

TEST(ChatTemplateTest, GemmaAutoTemplateUsesOfficialTurnFormatWithImplicitInstructions) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.token_to_id["<|turn>"] = 1;
    model.token_to_id["<turn|>"] = 2;

    const std::string prompt = "hello";
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, prompt);

    EXPECT_EQ(wrapped,
              "<bos>"
              "<|turn>user\n"
              "hello"
              "<turn|>\n"
              "<|turn>model\n");
}

TEST(ChatTemplateTest, AutoTemplateLeavesGemmaPromptUntouchedWhenAlreadyTemplated) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.token_to_id["<|turn>"] = 1;
    model.token_to_id["<turn|>"] = 2;

    const std::string prompt = "<bos><|turn>user\nhello\n<turn|>\n<|turn>model\n";
    EXPECT_EQ(DenseCoreTestOnlyApplyAutoChatTemplate(&model, prompt), prompt);
}

TEST(ChatTemplateTest, QwenNoThinkingAutoTemplateKeepsChatMLScaffoldForUnicodePrompt) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.token_to_id["<|im_start|>"] = 1;
    model.token_to_id["<|im_end|>"] = 2;

    setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false", 1);
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "안녕?");
    unsetenv("DENSECORE_QWEN35_ENABLE_THINKING");

    EXPECT_EQ(wrapped,
              "<|im_start|>user\n"
              "안녕? /no_think<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(ChatTemplateTest, Qwen35DefaultsToNoThinkingWhenEnvIsUnset) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.token_to_id["<|im_start|>"] = 1;
    model.token_to_id["<|im_end|>"] = 2;

    unsetenv("DENSECORE_QWEN35_ENABLE_THINKING");
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");

    EXPECT_EQ(wrapped,
              "<|im_start|>user\n"
              "hello /no_think<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(ChatTemplateTest, Qwen3DefaultsToThinkingWhenEnvIsUnset) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN3;
    model.token_to_id["<|im_start|>"] = 1;
    model.token_to_id["<|im_end|>"] = 2;

    unsetenv("DENSECORE_QWEN3_ENABLE_THINKING");
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");

    EXPECT_TRUE(DenseCoreTestOnlyPromptStartsInThinkBlock(wrapped));
}

TEST(ChatTemplateTest, GemmaThinkingAutoTemplateInjectsThinkSystemTurn) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.token_to_id["<|turn>"] = 1;
    model.token_to_id["<turn|>"] = 2;

    setenv("DENSECORE_GEMMA4_ENABLE_THINKING", "true", 1);
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");
    unsetenv("DENSECORE_GEMMA4_ENABLE_THINKING");

    EXPECT_EQ(wrapped,
              "<bos>"
              "<|turn>system\n"
              "<|think|><turn|>\n"
              "<|turn>user\n"
              "hello"
              "<turn|>\n"
              "<|turn>model\n");
}

TEST(ChatTemplateTest, RoleTagAutoTemplateUsesDedicatedSystemAndAssistantTags) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.token_to_id["<|system|>"] = 1;
    model.token_to_id["<|user|>"] = 2;
    model.token_to_id["<|assistant|>"] = 3;

    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");

    EXPECT_EQ(wrapped,
              "<|system|>\n"
              "You are a helpful assistant.\n"
              "<|user|>\n"
              "hello"
              "\n<|assistant|>\n");
}

TEST(ChatTemplateTest, QwenThinkingPromptStartsInsideThinkBlock) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.token_to_id["<|im_start|>"] = 1;
    model.token_to_id["<|im_end|>"] = 2;

    setenv("DENSECORE_QWEN35_ENABLE_THINKING", "true", 1);
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");
    unsetenv("DENSECORE_QWEN35_ENABLE_THINKING");

    EXPECT_TRUE(DenseCoreTestOnlyPromptStartsInThinkBlock(wrapped));
}

TEST(ChatTemplateTest, QwenNoThinkingPromptDoesNotStartInsideThinkBlock) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.token_to_id["<|im_start|>"] = 1;
    model.token_to_id["<|im_end|>"] = 2;

    setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false", 1);
    const std::string wrapped = DenseCoreTestOnlyApplyAutoChatTemplate(&model, "hello");
    unsetenv("DENSECORE_QWEN35_ENABLE_THINKING");

    EXPECT_FALSE(DenseCoreTestOnlyPromptStartsInThinkBlock(wrapped));
}

TEST(QwenReasoningBlocklistTest, NoThinkingBlocksReasoningTagsButKeepsChatTerminator) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.eos_token_id = 99;
    model.token_to_id["<think>"] = 1;
    model.token_to_id["</think>"] = 2;
    model.token_to_id["<|im_start|>"] = 3;
    model.token_to_id["<|im_end|>"] = 4;
    model.token_to_id["<tool_call>"] = 5;
    model.token_to_id["</tool_call>"] = 6;
    model.token_to_id["<custom_control>"] = 7;

    setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false", 1);
    const std::vector<int> blocked = DenseCoreTestOnlyQwenReasoningBlocklist(&model);
    unsetenv("DENSECORE_QWEN35_ENABLE_THINKING");

    const std::vector<int> expected = {1, 2, 3, 5, 6};
    EXPECT_EQ(blocked, expected);
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 4), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 7), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 99), blocked.end());
}

TEST(Gemma4TextBlocklistTest, BlocksControlAndUnusedTokensButKeepsStopIds) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.eos_token_id = 99;
    model.stop_token_ids = {99};
    model.vocab_tokens = {"normal", "<unk>", "<|think|>", "<|turn>", "<unused7>", "<0x41>", "<eos>", "plain"};
    model.token_to_id["normal"] = 0;
    model.token_to_id["<unk>"] = 1;
    model.token_to_id["<|think|>"] = 2;
    model.token_to_id["<|turn>"] = 3;
    model.token_to_id["<unused7>"] = 4;
    model.token_to_id["<0x41>"] = 5;
    model.token_to_id["<eos>"] = 6;
    model.token_to_id["plain"] = 7;
    model.token_types = {1, 2, 3, 4, 5, 6, 1, 1};

    const std::vector<int> blocked = DenseCoreTestOnlyGemma4TextBlocklist(&model);

    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 0), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 1), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 2), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 3), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 4), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 5), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 6), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 7), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 99), blocked.end());
}

TEST(Gemma4TextBlocklistTest, HandlesPartialTokenTypeMetadata) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.eos_token_id = 99;
    model.stop_token_ids = {99};
    model.vocab_tokens = {"normal", "<unk>", "<|think|>", "<|turn>", "<unused7>", "<0x41>", "<eos>", "plain"};
    model.token_to_id["normal"] = 0;
    model.token_to_id["<unk>"] = 1;
    model.token_to_id["<|think|>"] = 2;
    model.token_to_id["<|turn>"] = 3;
    model.token_to_id["<unused7>"] = 4;
    model.token_to_id["<0x41>"] = 5;
    model.token_to_id["<eos>"] = 6;
    model.token_to_id["plain"] = 7;
    model.token_types = {1, 3, 3, 3};

    const std::vector<int> blocked = DenseCoreTestOnlyGemma4TextBlocklist(&model);

    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 1), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 2), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 3), blocked.end());
    EXPECT_NE(std::find(blocked.begin(), blocked.end(), 4), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 5), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 6), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 7), blocked.end());
    EXPECT_EQ(std::find(blocked.begin(), blocked.end(), 99), blocked.end());
}
