#include "ggml-quants.h"
#include "llm/matmul/q8_small_batch.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

TEST(Q8SmallBatchTest, MatchesNativeDotWithOddBlockCountsStridesAndGuards) {
    ggml_cpu_init();
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    std::mt19937 rng(20260913);
    for (int n : {32, 96, 256, 2048, 4096, 8192}) {
        const size_t row = ggml_row_size(GGML_TYPE_Q8_0, n);
        for (size_t padding : {size_t(0), size_t(14), size_t(64)}) {
            const size_t stride = row + padding;
            std::vector<uint8_t> weights(row), inputs(4 * stride);
            auto fill = [&](void* data) {
                auto* blocks = static_cast<block_q8_0*>(data);
                for (int b = 0; b < n / QK8_0; ++b) {
                    blocks[b].d = ggml_fp32_to_fp16((int(rng() % 2001) - 1000) / 2048.0f);
                    for (auto& q : blocks[b].qs) q = int(rng() % 256) - 128;
                }
            };
            for (int repeat = 0; repeat < 8; ++repeat) {
                fill(weights.data());
                for (int m = 0; m < 4; ++m) fill(inputs.data() + m * stride);
                float actual[17], expected[17];
                std::fill_n(actual, 17, -123456.0f);
                std::fill_n(expected, 17, -123456.0f);
                Q8SmallBatchDot4(n, weights.data(), inputs.data(), stride, actual + 1, 4);
                for (int m = 0; m < 4; ++m) {
                    dot(n, expected + 1 + m * 4, 0, weights.data(), 0, inputs.data() + m * stride, 0, 1);
                    ASSERT_TRUE(std::isfinite(actual[1 + m * 4]));
                }
                ASSERT_EQ(std::memcmp(actual, expected, sizeof(actual)), 0)
                    << "n=" << n << " padding=" << padding << " repeat=" << repeat;
            }
        }
    }
}
