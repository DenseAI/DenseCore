#include <gtest/gtest.h>

#include "worker_internal.h"

TEST(QwenLoopGuardTest, StopsEightIdenticalSuffixTokens) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {1, 2, 3, 9, 9, 9, 9, 9, 9, 9, 9};

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, StopsAlternatingTwoTokenLoop) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5};

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, IgnoresNormalShortSuffixes) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {10, 11, 12, 13, 13, 13, 13, 13, 13, 13};

    EXPECT_FALSE(ShouldTerminateRepetitiveLoop(&model, &req));
}
