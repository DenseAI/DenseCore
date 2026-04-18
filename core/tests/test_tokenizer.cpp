#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "model_types.h"
#include "tokenizer.h"

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
