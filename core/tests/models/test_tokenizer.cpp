#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "densecore/models/model_types.h"
#include "densecore/models/tokenizer.h"

namespace {

TransformerModel MakeGemma4TokenizerModel() {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.tokenizer_type = "gemma";
    model.bos_token_id = 2;
    model.vocab_tokens.resize(4096);

    auto add_token = [&](int id, const char* token, int32_t token_type = 1) {
        if (static_cast<size_t>(id) >= model.vocab_tokens.size()) {
            model.vocab_tokens.resize(static_cast<size_t>(id) + 1);
        }
        model.vocab_tokens[static_cast<size_t>(id)] = token;
        model.token_to_id[token] = id;
        if (static_cast<size_t>(id) >= model.token_types.size()) {
            model.token_types.resize(static_cast<size_t>(id) + 1, 1);
        }
        model.token_types[static_cast<size_t>(id)] = token_type;
    };

    add_token(2, "<bos>", 3);
    add_token(105, "<|turn>", 3);
    add_token(106, "<turn|>", 3);
    add_token(107, "\n");

    add_token(200, "u");
    add_token(201, "us");
    add_token(202, "use");
    add_token(203, "user");

    add_token(300, "h");
    add_token(301, "he");
    add_token(302, "hel");
    add_token(303, "hell");
    add_token(304, "hello");

    add_token(400, "m");
    add_token(401, "mo");
    add_token(402, "mod");
    add_token(403, "mode");
    add_token(404, "model");
    return model;
}

TransformerModel MakeGemma4SentencePieceModel() {
    TransformerModel model = MakeGemma4TokenizerModel();

    auto add_token = [&](int id, const char* token, int32_t token_type = 1) {
        if (static_cast<size_t>(id) >= model.vocab_tokens.size()) {
            model.vocab_tokens.resize(static_cast<size_t>(id) + 1);
        }
        model.vocab_tokens[static_cast<size_t>(id)] = token;
        model.token_to_id[token] = id;
        if (static_cast<size_t>(id) >= model.token_types.size()) {
            model.token_types.resize(static_cast<size_t>(id) + 1, 1);
        }
        model.token_types[static_cast<size_t>(id)] = token_type;
    };

    add_token(500, "▁hello");
    add_token(501, "▁world");
    add_token(502, "<unk>", 2);
    add_token(503, "h");
    add_token(504, "e");
    add_token(505, "l");
    add_token(506, "o");
    add_token(507, "w");
    add_token(508, "r");
    add_token(509, "d");

    model.token_scores.resize(model.vocab_tokens.size(), -1000.0f);
    model.token_scores[500] = 12.0f;
    model.token_scores[501] = 11.0f;
    model.token_scores[502] = -50.0f;
    model.token_scores[503] = 0.1f;
    model.token_scores[504] = 0.1f;
    model.token_scores[505] = 0.1f;
    model.token_scores[506] = 0.1f;
    model.token_scores[507] = 0.1f;
    model.token_scores[508] = 0.1f;
    model.token_scores[509] = 0.1f;

    return model;
}

}  // namespace

TEST(TokenizerTest, Gemma4ControlTokensRemainAtomicWithoutMergeMetadata) {
    TransformerModel model = MakeGemma4TokenizerModel();

    const std::vector<int> tokens = Tokenizer::Tokenize(
        &model, "<bos><|turn>user\nhello<turn|>\n<|turn>model\n", /*add_bos=*/false, /*add_eos=*/false);

    const std::vector<int> expected = {2, 105, 203, 107, 300, 106, 107, 105, 404, 107};
    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, DoesNotPrependBosWhenPromptAlreadyStartsWithBosLiteral) {
    TransformerModel model = MakeGemma4TokenizerModel();

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "<bos><|turn>user\n", /*add_bos=*/true, /*add_eos=*/false);

    ASSERT_FALSE(tokens.empty());
    EXPECT_EQ(tokens.front(), model.bos_token_id);
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), model.bos_token_id), 1);
}

TEST(TokenizerTest, Gemma4RawPromptDoesNotReceiveAutoBosWhenDisabledByLoaderPolicy) {
    TransformerModel model = MakeGemma4SentencePieceModel();
    model.tokenizer_add_bos = false;

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "hello world", /*add_bos=*/model.tokenizer_add_bos, /*add_eos=*/false);

    ASSERT_FALSE(tokens.empty());
    EXPECT_EQ(tokens.front(), 500);
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), model.bos_token_id), 0);
}

TEST(TokenizerTest, Gemma4ChatTemplateWithExplicitBosKeepsExactlyOneBos) {
    TransformerModel model = MakeGemma4TokenizerModel();
    model.tokenizer_add_bos = false;

    const std::vector<int> tokens = Tokenizer::Tokenize(
        &model, "<bos><|turn>user\nhello<turn|>\n<|turn>model\n", /*add_bos=*/model.tokenizer_add_bos,
        /*add_eos=*/false);

    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), model.bos_token_id), 1);
    ASSERT_FALSE(tokens.empty());
    EXPECT_EQ(tokens.front(), model.bos_token_id);
}

TEST(TokenizerTest, Gemma4SentencePieceUsesUnigramScoresForWholePieces) {
    TransformerModel model = MakeGemma4SentencePieceModel();

    const std::vector<int> tokens = Tokenizer::Tokenize(&model, "hello world", /*add_bos=*/false, /*add_eos=*/false);

    const std::vector<int> expected = {500, 501};
    EXPECT_EQ(tokens, expected);
}
