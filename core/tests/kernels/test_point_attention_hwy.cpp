/**
 * @file test_point_attention_hwy.cpp
 * @brief Safety tests for Highway PointAttention public API
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "densecore/simd/hwy_ops.h"

namespace {

TEST(PointAttentionHwy, InvalidNeighborIndicesDoNotReadOutOfBounds) {
    constexpr int64_t B = 1;
    constexpr int64_t N = 4;
    constexpr int64_t D = 4;
    constexpr int k = 2;
    constexpr int num_heads = 1;

    std::vector<float> query(B * N * D);
    std::vector<float> key(B * N * D);
    std::vector<float> value(B * N * D);
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = 0.01f * static_cast<float>(i + 1);
        key[i] = 0.02f * static_cast<float>(i + 1);
        value[i] = 1.0f + 0.1f * static_cast<float>(i);
    }

    // Per-point neighbors:
    // n=0 -> valid
    // n=1 -> invalid (idx == N)
    // n=2 -> valid
    // n=3 -> invalid (idx < 0)
    const std::vector<int32_t> knn_indices = {
        0, 1, 2, 4, 1, 3, -1, 0,
    };
    const std::vector<float> knn_dists(N * k, 0.0f);

    std::vector<float> output(B * N * D, -123.0f);
    std::vector<uint8_t> workspace(static_cast<size_t>(k) * sizeof(float));

    densecore::hwy_kernels::PointAttention_Hwy(query.data(), key.data(), value.data(), knn_indices.data(),
                                               knn_dists.data(), output.data(), workspace.data(), workspace.size(), B,
                                               N, D, k, num_heads, 1.0f, false);

    for (float x : output) {
        EXPECT_TRUE(std::isfinite(x));
    }

    // Valid points should still produce non-zero outputs.
    for (int64_t n : {0, 2}) {
        bool has_non_zero = false;
        for (int64_t d = 0; d < D; ++d) {
            if (std::fabs(output[n * D + d]) > 1e-7f) {
                has_non_zero = true;
                break;
            }
        }
        EXPECT_TRUE(has_non_zero) << "Expected non-zero output for valid point n=" << n;
    }

    // Invalid points should be zeroed rather than reading invalid memory.
    for (int64_t n : {1, 3}) {
        for (int64_t d = 0; d < D; ++d) {
            EXPECT_FLOAT_EQ(output[n * D + d], 0.0f) << "Expected zero output for invalid point n=" << n;
        }
    }
}

}  // namespace
