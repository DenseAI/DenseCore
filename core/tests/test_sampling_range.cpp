/**
 * @file test_sampling_range.cpp
 * @brief Tests for action-token range sampling optimization
 */

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "ggml.h"
#include "inference.h"

namespace {

struct GgmlContextGuard {
    ggml_context* ctx = nullptr;
    ~GgmlContextGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

ggml_tensor* MakeLogitsTensor(ggml_context* ctx, const std::vector<float>& logits) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int64_t>(logits.size()), 1);
    std::memcpy(t->data, logits.data(), logits.size() * sizeof(float));
    return t;
}

}  // namespace

namespace densecore {

TEST(SamplingRangeTest, GreedyUsesActionTokenSubrange) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    // Global argmax would be token 6, but action range [2, 5) should choose token 3.
    const std::vector<float> logits_data = {0.1f, 0.2f, 0.7f, 1.1f, 0.9f, 0.3f, 4.0f, 0.5f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.0f;
    params.action_token_start = 2;
    params.action_token_count = 3;  // valid token ids: 2,3,4

    const int token = SampleToken(logits, 0, params);
    EXPECT_EQ(token, 3);
}

TEST(SamplingRangeTest, InvalidRangeFallsBackToFullVocab) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {0.1f, 0.2f, 0.7f, 1.1f, 0.9f, 0.3f, 4.0f, 0.5f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.0f;
    params.action_token_start = 999;
    params.action_token_count = 32;

    const int token = SampleToken(logits, 0, params);
    EXPECT_EQ(token, 6);
}

}  // namespace densecore
