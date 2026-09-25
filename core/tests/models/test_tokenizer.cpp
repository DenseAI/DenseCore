#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "densecore/models/model_types.h"
#include "densecore/models/tokenizer.h"

namespace {

TEST(TokenizerTest, StructuredDetokenizePreservesOnlyApiBoundaryTokens) {
    TransformerModel model{};
    model.vocab_tokens = {"plain", "</think>", "<tool_call>", "<|im_end|>"};
    model.token_types = {1, 3, 3, 3};
    model.token_to_id["</think>"] = 1;
    model.token_to_id["<tool_call>"] = 2;
    model.token_to_id["<|im_end|>"] = 3;
    Tokenizer::BuildStreamTokenPieceCache(&model);

    EXPECT_EQ(Tokenizer::Detokenize(&model, 1), "");
    EXPECT_EQ(Tokenizer::DetokenizeStructured(&model, 1), "</think>");
    EXPECT_EQ(Tokenizer::DetokenizeStructured(&model, 2), "<tool_call>");
    EXPECT_EQ(Tokenizer::DetokenizeStructured(&model, 3), "");
    EXPECT_EQ(Tokenizer::DetokenizeStructured(&model, 0), "plain");
}

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

TransformerModel MakeBertWordPieceModel() {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.variant = ModelVariant::BERT;
    model.tokenizer_type = "bert";
    model.bos_token_id = 101;
    model.sep_token_id = 102;
    model.unk_token_id = 100;
    model.pad_token_id = 0;
    model.mask_token_id = 103;
    model.vocab_tokens.resize(256);

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

    add_token(0, "[PAD]", 3);
    add_token(100, "[UNK]", 2);
    add_token(101, "[CLS]", 3);
    add_token(102, "[SEP]", 3);
    add_token(103, "[MASK]", 3);
    add_token(104, "the");
    add_token(105, "quick");
    add_token(106, "play");
    add_token(107, "##ing");
    add_token(108, ",");
    add_token(109, "dense");
    add_token(110, "##core");
    add_token(111, "works");

    return model;
}

TransformerModel MakeBertSentencePieceSurfaceModel() {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.variant = ModelVariant::BERT;
    model.tokenizer_type = "bert";
    model.bos_token_id = 101;
    model.sep_token_id = 102;
    model.eos_token_id = 102;
    model.unk_token_id = 100;
    model.pad_token_id = 0;
    model.mask_token_id = 103;
    model.vocab_tokens.resize(30000);

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

    add_token(0, "[PAD]", 3);
    add_token(100, "[UNK]", 2);
    add_token(101, "[CLS]", 3);
    add_token(102, "[SEP]", 3);
    add_token(103, "[MASK]", 3);
    add_token(9742, "▁dense");
    add_token(17345, "core");
    add_token(3216, "▁runs");
    add_token(17368, "▁cpu");
    add_token(28937, "▁inference");
    add_token(1999, "▁in");
    add_token(13970, "▁ku");
    add_token(5677, "ber");
    add_token(7159, "net");
    add_token(2229, "es");
    add_token(1012, "▁.");

    return model;
}

TransformerModel MakeBertBpeModel() {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.variant = ModelVariant::BERT;
    model.tokenizer_type = "bert-bpe";
    model.bos_token_id = 0;
    model.eos_token_id = 2;
    model.unk_token_id = 3;
    model.vocab_tokens.resize(130000);

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

    add_token(0, "<s>", 3);
    add_token(2, "</s>", 3);
    add_token(3, "<unk>", 2);
    add_token(5, "▁.");
    add_token(12, "▁:");
    add_token(13, "▁e");
    add_token(23, "▁▁in");
    add_token(41, "▁▁que");
    add_token(73, "▁in");
    add_token(184, "▁se");
    add_token(1294, "▁ry");
    add_token(1575, "▁▁Den");
    add_token(1636, "▁tes");
    add_token(3196, "▁erne");
    add_token(6620, "▁ence");
    add_token(50886, "▁Cor");
    add_token(53498, "▁▁infer");
    add_token(86039, "▁▁CPU");
    add_token(96305, "▁▁Kub");
    add_token(127877, "▁▁runs");

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

TEST(TokenizerTest, BertWordPieceGreedySplitsContinuationPieces) {
    TransformerModel model = MakeBertWordPieceModel();

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "The quick playing, DenseCore", /*add_bos=*/true, /*add_eos=*/true);

    const std::vector<int> expected = {101, 104, 105, 106, 107, 108, 109, 110, 102};
    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, BertWordPieceUsesUnknownForUnsegmentablePieces) {
    TransformerModel model = MakeBertWordPieceModel();

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "The unavailable", /*add_bos=*/true, /*add_eos=*/true);

    const std::vector<int> expected = {101, 104, 100, 102};
    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, BertWordPieceKeepsBracketSpecialTokensAtomic) {
    TransformerModel model = MakeBertWordPieceModel();

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "[CLS] the [MASK]", /*add_bos=*/false, /*add_eos=*/true);

    const std::vector<int> expected = {101, 104, 103, 102};
    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, BertSentencePieceSurfaceGreedyMatchesBgeSmallIds) {
    TransformerModel model = MakeBertSentencePieceSurfaceModel();

    const std::vector<int> tokens =
        Tokenizer::Tokenize(&model, "DenseCore runs CPU inference in Kubernetes.", /*add_bos=*/true,
                            /*add_eos=*/true);

    const std::vector<int> expected = {101, 9742, 17345, 3216, 17368, 28937, 1999, 13970, 5677, 7159, 2229, 1012, 102};
    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, BertBpeHandlesXlmRobertaStyleMetaspaceVocab) {
    TransformerModel model = MakeBertBpeModel();

    const std::vector<int> tokens = Tokenizer::Tokenize(
        &model, "query: DenseCore runs CPU inference in Kubernetes.", /*add_bos=*/true, /*add_eos=*/true);

    const std::vector<int> expected = {0, 41, 1294, 12, 1575, 184, 50886, 13, 127877,
                                       86039, 53498, 6620, 23, 96305, 3196, 1636, 5, 2};
    EXPECT_EQ(tokens, expected);
}
