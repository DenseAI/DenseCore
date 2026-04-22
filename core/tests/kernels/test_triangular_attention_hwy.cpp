/**
 * @file test_triangular_attention_hwy.cpp
 * @brief Tests for Highway TriangularAttention tiled execution paths
 */

#include <gtest/gtest.h>

#include "densecore/simd/hwy_ops.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using densecore::hwy_kernels::TriangularAttentionTiled_Hwy;
using densecore::hwy_kernels::TriangularAttention_Hwy;

size_t FullIndex(int64_t b, int64_t i, int64_t j, int64_t d, int64_t L, int64_t D) {
    return (((static_cast<size_t>(b) * static_cast<size_t>(L) + static_cast<size_t>(i)) * static_cast<size_t>(L) +
             static_cast<size_t>(j)) *
                static_cast<size_t>(D) +
            static_cast<size_t>(d));
}

size_t TileIndex(int64_t b, int64_t i, int64_t j, int64_t d, int64_t tile_rows, int64_t tile_cols, int64_t D) {
    return (((static_cast<size_t>(b) * static_cast<size_t>(tile_rows) + static_cast<size_t>(i)) *
                 static_cast<size_t>(tile_cols) +
             static_cast<size_t>(j)) *
                static_cast<size_t>(D) +
            static_cast<size_t>(d));
}

size_t WorkspaceBytes(int64_t L, int64_t D, int num_heads) {
    const int64_t d_head = D / num_heads;
    return static_cast<size_t>(4 * d_head + L) * sizeof(float);
}

void InitInputs(int64_t B, int64_t L, int64_t D, int num_heads, std::vector<float>* pair, std::vector<float>* qw,
                std::vector<float>* kw, std::vector<float>* vw) {
    const int64_t d_head = D / num_heads;
    pair->assign(static_cast<size_t>(B * L * L * D), 0.0f);
    qw->assign(static_cast<size_t>(num_heads * d_head * D), 0.0f);
    kw->assign(static_cast<size_t>(num_heads * d_head * D), 0.0f);
    vw->assign(static_cast<size_t>(num_heads * d_head * D), 0.0f);

    for (size_t i = 0; i < pair->size(); ++i) {
        (*pair)[i] = 0.01f * static_cast<float>(static_cast<int>(i % 29) - 14);
    }
    for (size_t i = 0; i < qw->size(); ++i) {
        (*qw)[i] = 0.02f * static_cast<float>(static_cast<int>(i % 17) - 8);
        (*kw)[i] = 0.015f * static_cast<float>(static_cast<int>(i % 13) - 6);
        (*vw)[i] = 0.018f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
}

std::vector<float> RunFullReference(const std::vector<float>& pair, const std::vector<float>& qw, const std::vector<float>& kw,
                                    const std::vector<float>& vw, int64_t B, int64_t L, int64_t D, int num_heads,
                                    float scale, bool starting) {
    std::vector<float> out(static_cast<size_t>(B * L * L * D), 0.0f);
    std::vector<uint8_t> workspace(WorkspaceBytes(L, D, num_heads));
    TriangularAttention_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out.data(), workspace.data(), workspace.size(), B, L,
                            D, num_heads, scale, starting);
    return out;
}

TEST(TriangularAttentionHwy, PartialTileOutputIsTiledMatchesReference) {
    constexpr int64_t B = 2;
    constexpr int64_t L = 5;
    constexpr int64_t D = 8;
    constexpr int num_heads = 2;
    constexpr int64_t row_start = 1;
    constexpr int64_t row_end = 4;
    constexpr int64_t col_start = 2;
    constexpr int64_t col_end = 5;
    constexpr int64_t tile_rows = row_end - row_start;
    constexpr int64_t tile_cols = col_end - col_start;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D / num_heads));

    std::vector<float> pair, qw, kw, vw;
    InitInputs(B, L, D, num_heads, &pair, &qw, &kw, &vw);

    for (bool starting : {true, false}) {
        const std::vector<float> full_ref = RunFullReference(pair, qw, kw, vw, B, L, D, num_heads, scale, starting);

        std::vector<float> out_tile(static_cast<size_t>(B * tile_rows * tile_cols * D), 0.0f);
        std::vector<uint8_t> workspace(WorkspaceBytes(L, D, num_heads));
        TriangularAttentionTiled_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out_tile.data(), workspace.data(),
                                     workspace.size(), B, L, D, num_heads, scale, starting, row_start, row_end, col_start,
                                     col_end, true);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < tile_rows; ++i) {
                for (int64_t j = 0; j < tile_cols; ++j) {
                    for (int64_t d = 0; d < D; ++d) {
                        const float expected =
                            full_ref[FullIndex(b, row_start + i, col_start + j, d, L, D)];
                        const float actual = out_tile[TileIndex(b, i, j, d, tile_rows, tile_cols, D)];
                        EXPECT_NEAR(actual, expected, 1e-4f)
                            << "starting=" << starting << " b=" << b << " i=" << i << " j=" << j << " d=" << d;
                    }
                }
            }
        }
    }
}

TEST(TriangularAttentionHwy, PartialTileOutputIsFullWritesOnlyTile) {
    constexpr int64_t B = 2;
    constexpr int64_t L = 5;
    constexpr int64_t D = 8;
    constexpr int num_heads = 2;
    constexpr int64_t row_start = 1;
    constexpr int64_t row_end = 4;
    constexpr int64_t col_start = 0;
    constexpr int64_t col_end = 3;
    constexpr float kSentinel = -777.0f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D / num_heads));

    std::vector<float> pair, qw, kw, vw;
    InitInputs(B, L, D, num_heads, &pair, &qw, &kw, &vw);

    for (bool starting : {true, false}) {
        const std::vector<float> full_ref = RunFullReference(pair, qw, kw, vw, B, L, D, num_heads, scale, starting);

        std::vector<float> out_full(static_cast<size_t>(B * L * L * D), kSentinel);
        std::vector<uint8_t> workspace(WorkspaceBytes(L, D, num_heads));
        TriangularAttentionTiled_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out_full.data(), workspace.data(),
                                     workspace.size(), B, L, D, num_heads, scale, starting, row_start, row_end, col_start,
                                     col_end, false);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < L; ++i) {
                for (int64_t j = 0; j < L; ++j) {
                    const bool in_tile = (i >= row_start && i < row_end && j >= col_start && j < col_end);
                    for (int64_t d = 0; d < D; ++d) {
                        const size_t idx = FullIndex(b, i, j, d, L, D);
                        if (in_tile) {
                            EXPECT_NEAR(out_full[idx], full_ref[idx], 1e-4f)
                                << "starting=" << starting << " b=" << b << " i=" << i << " j=" << j << " d=" << d;
                        } else {
                            EXPECT_FLOAT_EQ(out_full[idx], kSentinel)
                                << "Unexpected write starting=" << starting << " b=" << b << " i=" << i << " j=" << j
                                << " d=" << d;
                        }
                    }
                }
            }
        }
    }
}

TEST(TriangularAttentionHwy, FullTileOutputIsTiledMatchesReference) {
    constexpr int64_t B = 2;
    constexpr int64_t L = 5;
    constexpr int64_t D = 8;
    constexpr int num_heads = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D / num_heads));

    std::vector<float> pair, qw, kw, vw;
    InitInputs(B, L, D, num_heads, &pair, &qw, &kw, &vw);

    for (bool starting : {true, false}) {
        const std::vector<float> full_ref = RunFullReference(pair, qw, kw, vw, B, L, D, num_heads, scale, starting);

        std::vector<float> out_full_tiled(static_cast<size_t>(B * L * L * D), 0.0f);
        std::vector<uint8_t> workspace(WorkspaceBytes(L, D, num_heads));
        TriangularAttentionTiled_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out_full_tiled.data(), workspace.data(),
                                     workspace.size(), B, L, D, num_heads, scale, starting, 0, L, 0, L, true);

        for (size_t i = 0; i < full_ref.size(); ++i) {
            EXPECT_NEAR(out_full_tiled[i], full_ref[i], 1e-4f) << "starting=" << starting << " idx=" << i;
        }
    }
}

TEST(TriangularAttentionHwy, FullTileOutputIsFullMatchesReference) {
    constexpr int64_t B = 2;
    constexpr int64_t L = 5;
    constexpr int64_t D = 8;
    constexpr int num_heads = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D / num_heads));

    std::vector<float> pair, qw, kw, vw;
    InitInputs(B, L, D, num_heads, &pair, &qw, &kw, &vw);

    for (bool starting : {true, false}) {
        const std::vector<float> full_ref = RunFullReference(pair, qw, kw, vw, B, L, D, num_heads, scale, starting);

        std::vector<float> out_full(static_cast<size_t>(B * L * L * D), -1.0f);
        std::vector<uint8_t> workspace(WorkspaceBytes(L, D, num_heads));
        TriangularAttentionTiled_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out_full.data(), workspace.data(),
                                     workspace.size(), B, L, D, num_heads, scale, starting, 0, L, 0, L, false);

        for (size_t i = 0; i < full_ref.size(); ++i) {
            EXPECT_NEAR(out_full[i], full_ref[i], 1e-4f) << "starting=" << starting << " idx=" << i;
        }
    }
}

TEST(TriangularAttentionHwy, OversizedStridesReturnWithoutWrite) {
    const int64_t B = 1;
    const int64_t D = 8;
    const int num_heads = 2;
    const int64_t huge_L =
        static_cast<int64_t>(std::sqrt(static_cast<long double>(std::numeric_limits<size_t>::max()))) + 1024;

    std::vector<float> pair(1, 0.0f);
    std::vector<float> qw(1, 1.0f);
    std::vector<float> kw(1, 1.0f);
    std::vector<float> vw(1, 1.0f);
    std::vector<float> out(1, 123.0f);
    std::vector<uint8_t> workspace(16, 0);

    TriangularAttentionTiled_Hwy(pair.data(), qw.data(), kw.data(), vw.data(), out.data(), workspace.data(), workspace.size(),
                                 B, huge_L, D, num_heads, 1.0f, true, 0, huge_L, 0, huge_L, false);

    EXPECT_FLOAT_EQ(out[0], 123.0f);
}

}  // namespace
