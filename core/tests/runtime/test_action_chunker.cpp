/**
 * @file test_action_chunker.cpp
 * @brief Tests for robotics action chunking utility
 */

#include <gtest/gtest.h>

#include <vector>

#include "densecore/action_chunker.h"

namespace densecore {
namespace robotics {
namespace {

TEST(ActionChunkerTest, InterpolatesToControlRate) {
    ActionChunkerConfig cfg;
    cfg.action_dim = 1;
    cfg.control_hz = 50;
    cfg.inference_hz = 5;  // expected 10 steps/chunk
    cfg.linear_interpolation = true;
    cfg.max_pending_chunks = 2;

    ActionChunker chunker(cfg);
    ASSERT_TRUE(chunker.IsConfigured());
    EXPECT_EQ(chunker.ExpectedControlStepsPerChunk(), 10);

    const std::vector<float> chunk = {0.0f, 1.0f};  // [2 steps, 1 dim]
    ASSERT_TRUE(chunker.SubmitChunk(chunk, 2));
    EXPECT_EQ(chunker.PendingSteps(), 10);

    std::vector<float> action;
    ASSERT_TRUE(chunker.NextAction(&action));
    EXPECT_NEAR(action[0], 0.0f, 1e-6f);

    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(chunker.NextAction(&action));
    }
    ASSERT_TRUE(chunker.NextAction(&action));
    EXPECT_NEAR(action[0], 1.0f, 1e-6f);
    EXPECT_EQ(chunker.PendingSteps(), 0);
}

TEST(ActionChunkerTest, KeepsLatestWhenQueueIsFull) {
    ActionChunkerConfig cfg;
    cfg.action_dim = 2;
    cfg.expected_chunk_steps = 4;
    cfg.max_pending_chunks = 1;
    cfg.linear_interpolation = false;

    ActionChunker chunker(cfg);
    ASSERT_TRUE(chunker.IsConfigured());

    const std::vector<float> first = {
        1, 1, 2, 2, 3, 3, 4, 4,
    };
    const std::vector<float> second = {
        10, 10, 20, 20, 30, 30, 40, 40,
    };

    ASSERT_TRUE(chunker.SubmitChunk(first, 4));
    ASSERT_TRUE(chunker.SubmitChunk(second, 4));
    EXPECT_EQ(chunker.PendingSteps(), 4);

    std::vector<float> action;
    ASSERT_TRUE(chunker.NextAction(&action));
    EXPECT_FLOAT_EQ(action[0], 10.0f);
    EXPECT_FLOAT_EQ(action[1], 10.0f);
}

TEST(ActionChunkerTest, RejectsInvalidInput) {
    ActionChunkerConfig cfg;
    cfg.action_dim = 3;
    cfg.expected_chunk_steps = 5;
    ActionChunker chunker(cfg);

    const std::vector<float> bad = {1.0f, 2.0f, 3.0f};  // not chunk_steps * action_dim
    EXPECT_FALSE(chunker.SubmitChunk(bad, 2));
}

}  // namespace
}  // namespace robotics
}  // namespace densecore
