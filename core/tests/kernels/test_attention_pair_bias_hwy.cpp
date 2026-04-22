/**
 * @file test_attention_pair_bias_hwy.cpp
 * @brief Safety tests for Highway AttentionWithPairBias public API
 */

#include <vector>

#include <gtest/gtest.h>

#include "densecore/simd/hwy_ops.h"

namespace {

TEST(AttentionWithPairBiasHwy, ZeroPairTokensDoesNotDivideByZero) {
    constexpr int tokens = 4;
    constexpr int c_m = 4;
    constexpr int c_z = 2;
    constexpr int n_head = 1;
    constexpr int head_dim = 4;

    std::vector<float> msa_input(tokens * c_m, 0.1f);
    std::vector<float> pair_data(c_z, 0.0f);  // Not read in this test (invalid shape path).
    std::vector<float> wq(c_m * c_m, 0.01f);
    std::vector<float> wk(c_m * c_m, 0.02f);
    std::vector<float> wv(c_m * c_m, 0.03f);
    std::vector<float> wo(c_m * c_m, 0.04f);
    std::vector<float> pair_bias_w(n_head * c_z, 0.05f);
    std::vector<float> output(tokens * c_m, 123.0f);

    densecore::hwy_kernels::AttentionWithPairBias_Hwy(msa_input.data(), pair_data.data(), wq.data(), wk.data(),
                                                      wv.data(), wo.data(), pair_bias_w.data(), output.data(), tokens,
                                                      c_m, c_z, n_head, head_dim, /*pair_tokens=*/0, /*scale=*/1.0f);

    for (float v : output) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

}  // namespace
