/**
 * @file test_universal_ops.cpp
 * @brief Unit tests for Universal Transformer Operations
 *
 * Tests for: WindowAttention, PosInterpolation, PatchEmbed3D, TemporalAttention,
 * Patchify, Unpatchify, BEVQuery, ScatterReduce, PointAttention, etc.
 */

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class UniversalOpsTest : public ::testing::Test {
protected:
    void SetUp() override { OpRegistry::Init(); }
};

// ============================================================================
// Patchify / Unpatchify Tests (Inverse Operations)
// ============================================================================

TEST_F(UniversalOpsTest, PatchifyBasic) {
    // Test Patchify: [B, C, H, W] -> [B, N, P*P*C]
    const int64_t B = 1, C = 3, H = 4, W = 4;
    const int64_t P = 2;                  // patch_size
    const int64_t N = (H / P) * (W / P);  // 4 patches
    const int64_t patch_dim = P * P * C;  // 12

    std::vector<float> image(B * C * H * W);
    std::iota(image.begin(), image.end(), 0.0f);

    std::vector<float> patches(B * N * patch_dim, 0.0f);

    Tensor in_tensor = Tensor::Wrap(image.data(), {B, C, H, W}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(patches.data(), {B, N, patch_dim}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::Patchify, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "Patchify op not registered";
    }

    std::vector<Tensor*> inputs = {&in_tensor};
    std::vector<Tensor*> outputs = {&out_tensor};

    // Use custom params struct matching the kernel
    struct {
        int patch_size = 2;
    } params;
    op->Execute(inputs, outputs, &params);

    // Verify: first patch should contain top-left 2x2 of each channel
    // For C=0: pixels [0,1,4,5] (0,0->0, 0,1->1, 1,0->4, 1,1->5)
    EXPECT_FLOAT_EQ(patches[0], 0.0f);  // C0, (0,0)
    EXPECT_FLOAT_EQ(patches[1], 1.0f);  // C0, (0,1)
    EXPECT_FLOAT_EQ(patches[2], 4.0f);  // C0, (1,0)
    EXPECT_FLOAT_EQ(patches[3], 5.0f);  // C0, (1,1)
}

TEST_F(UniversalOpsTest, UnpatchifyBasic) {
    // Test Unpatchify: [B, N, P*P*C] -> [B, C, H, W]
    const int64_t B = 1;
    const int64_t P = 2;
    const int64_t H = 4, W = 4;
    const int64_t C = 1;
    const int64_t N = 4;  // 2x2 patches
    const int64_t patch_dim = P * P * C;

    std::vector<float> patches(B * N * patch_dim);
    std::iota(patches.begin(), patches.end(), 1.0f);  // 1,2,3,...,16

    std::vector<float> image(B * C * H * W, 0.0f);

    Tensor in_tensor = Tensor::Wrap(patches.data(), {B, N, patch_dim}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(image.data(), {B, C, H, W}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::Unpatchify, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "Unpatchify op not registered";
    }

    std::vector<Tensor*> inputs = {&in_tensor};
    std::vector<Tensor*> outputs = {&out_tensor};

    struct {
        int patch_size = 2;
        int height = 4;
        int width = 4;
        int channels = 1;
    } params;
    op->Execute(inputs, outputs, &params);

    // First patch fills top-left 2x2: indices [0,1,4,5] in NCHW
    EXPECT_FLOAT_EQ(image[0], 1.0f);
    EXPECT_FLOAT_EQ(image[1], 2.0f);
    EXPECT_FLOAT_EQ(image[4], 3.0f);
    EXPECT_FLOAT_EQ(image[5], 4.0f);
}

TEST_F(UniversalOpsTest, PatchifyUnpatchifyInverse) {
    // Patchify then Unpatchify should give back original
    const int64_t B = 1, C = 3, H = 8, W = 8;
    const int64_t P = 4;
    const int64_t N = (H / P) * (W / P);
    const int64_t patch_dim = P * P * C;

    std::vector<float> original(B * C * H * W);
    std::iota(original.begin(), original.end(), 0.0f);

    std::vector<float> patches(B * N * patch_dim, 0.0f);
    std::vector<float> reconstructed(B * C * H * W, 0.0f);

    Tensor orig_tensor = Tensor::Wrap(original.data(), {B, C, H, W}, DType::F32);
    Tensor patch_tensor = Tensor::Wrap(patches.data(), {B, N, patch_dim}, DType::F32);
    Tensor recon_tensor = Tensor::Wrap(reconstructed.data(), {B, C, H, W}, DType::F32);

    auto* patchify_op = OpRegistry::Instance().Get(OpType::Patchify, DeviceType::CPU);
    auto* unpatchify_op = OpRegistry::Instance().Get(OpType::Unpatchify, DeviceType::CPU);

    if (!patchify_op || !unpatchify_op) {
        GTEST_SKIP() << "Patchify/Unpatchify ops not registered";
    }

    struct {
        int patch_size = 4;
    } p_params;
    patchify_op->Execute({&orig_tensor}, {&patch_tensor}, &p_params);

    struct {
        int patch_size = 4;
        int height = 8;
        int width = 8;
        int channels = 3;
    } up_params;
    unpatchify_op->Execute({&patch_tensor}, {&recon_tensor}, &up_params);

    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_FLOAT_EQ(original[i], reconstructed[i]) << "Mismatch at index " << i;
    }
}

// ============================================================================
// WindowAttention Tests
// ============================================================================

TEST_F(UniversalOpsTest, WindowAttentionBasic) {
    // [B, H, W, C] with window_size=7, num_heads=2
    const int64_t B = 1, H = 14, W = 14, C = 64;
    const int64_t window_size = 7;
    const int64_t num_heads = 2;

    std::vector<float> query(B * H * W * C, 1.0f);
    std::vector<float> key(B * H * W * C, 1.0f);
    std::vector<float> value(B * H * W * C, 1.0f);
    std::vector<float> output(B * H * W * C, 0.0f);

    Tensor q = Tensor::Wrap(query.data(), {B, H, W, C}, DType::F32);
    Tensor k = Tensor::Wrap(key.data(), {B, H, W, C}, DType::F32);
    Tensor v = Tensor::Wrap(value.data(), {B, H, W, C}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {B, H, W, C}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::WindowAttention, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "WindowAttention op not registered";
    }

    struct {
        int window_size = 7;
        int shift_size = 0;
        int num_heads = 2;
        float scale = 0.0f;
        bool use_relative_bias = false;
    } params;
    op->Execute({&q, &k, &v}, {&out}, &params);

    // With uniform Q/K/V of 1.0, output should be close to 1.0
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(output[i], 1.0f, 0.1f);
    }
}

// ============================================================================
// TemporalAttention Tests
// ============================================================================

TEST_F(UniversalOpsTest, TemporalAttentionBasic) {
    // [B, T, N, H, D] video attention
    const int64_t B = 1, T = 4, N = 16, H = 2, D = 32;

    std::vector<float> query(B * T * N * H * D, 1.0f);
    std::vector<float> key(B * T * N * H * D, 1.0f);
    std::vector<float> value(B * T * N * H * D, 1.0f);
    std::vector<float> output(B * T * N * H * D, 0.0f);

    Tensor q = Tensor::Wrap(query.data(), {B * T, N, H * D}, DType::F32);
    Tensor k = Tensor::Wrap(key.data(), {B * T, N, H * D}, DType::F32);
    Tensor v = Tensor::Wrap(value.data(), {B * T, N, H * D}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {B * T, N, H * D}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::TemporalAttention, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "TemporalAttention op not registered";
    }

    struct {
        int num_heads = 2;
        float scale = 0.0f;
        bool causal = false;
        int window_size = 0;
    } params;
    op->Execute({&q, &k, &v}, {&out}, &params);

    // Output should be defined (not NaN)
    EXPECT_FALSE(std::isnan(output[0]));
    EXPECT_NEAR(output[0], 1.0f, 0.1f);
}

// ============================================================================
// PatchEmbed3D Tests
// ============================================================================

TEST_F(UniversalOpsTest, PatchEmbed3DBasic) {
    // [B, C, T, H, W] -> [B, N, D]
    const int64_t B = 1, C = 3, T = 2, H = 4, W = 4;
    const int64_t patch_t = 2, patch_h = 2, patch_w = 2;
    const int64_t embed_dim = 64;

    const int64_t num_patches = (T / patch_t) * (H / patch_h) * (W / patch_w);  // 1*2*2 = 4

    std::vector<float> video(B * C * T * H * W);
    std::iota(video.begin(), video.end(), 0.0f);

    std::vector<float> output(B * num_patches * embed_dim, 0.0f);
    std::vector<float> weights(embed_dim * (C * patch_t * patch_h * patch_w), 0.01f);

    Tensor in_tensor = Tensor::Wrap(video.data(), {B, C * T, H, W}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(output.data(), {B, num_patches, embed_dim}, DType::F32);
    Tensor w_tensor = Tensor::Wrap(weights.data(), {embed_dim, C * patch_t * patch_h * patch_w}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::PatchEmbed3D, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "PatchEmbed3D op not registered";
    }

    struct {
        int in_channels = 3;
        int embed_dim = 64;
        int temporal_patch = 2;
        int patch_h = 2;
        int patch_w = 2;
    } params;
    params.embed_dim = embed_dim;
    params.temporal_patch = patch_t;

    op->Execute({&in_tensor, &w_tensor}, {&out_tensor}, &params);

    // Output should have values
    bool has_nonzero = false;
    for (float v : output) {
        if (v != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// ScatterReduce Tests
// ============================================================================

TEST_F(UniversalOpsTest, ScatterReduceSum) {
    // Scatter points to voxels with SUM reduction
    const int64_t N = 4;  // 4 points
    const int64_t D = 2;  // 2D features
    const int X = 2, Y = 2, Z = 1;

    // Coordinates as floats (casting from int)
    std::vector<float> coords_f = {
        0.0f, 0.0f, 0.0f,  // point 0 -> voxel (0,0,0)
        0.0f, 1.0f, 0.0f,  // point 1 -> voxel (0,1,0)
        1.0f, 0.0f, 0.0f,  // point 2 -> voxel (1,0,0)
        1.0f, 1.0f, 0.0f   // point 3 -> voxel (1,1,0)
    };

    std::vector<float> features = {
        1.0f, 2.0f,  // point 0
        3.0f, 4.0f,  // point 1
        5.0f, 6.0f,  // point 2
        7.0f, 8.0f   // point 3
    };

    std::vector<float> output(X * Y * Z * D, 0.0f);

    Tensor coord_tensor = Tensor::Wrap(coords_f.data(), {N, 3}, DType::F32);
    Tensor feat_tensor = Tensor::Wrap(features.data(), {N, D}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(output.data(), {X * Y * Z, D}, DType::F32);

    // Get op via ScatterReduce OpType
    auto* op = OpRegistry::Instance().Get(OpType::ScatterReduce, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "ScatterReduce op not registered";
    }

    struct {
        int mode = 0;
        int voxel_x = 2;
        int voxel_y = 2;
        int voxel_z = 1;
    } params;
    op->Execute({&coord_tensor, &feat_tensor}, {&out_tensor}, &params);

    // Each voxel should have its corresponding point's features
    EXPECT_NEAR(output[0], 1.0f, 1e-5);  // voxel 0 feature 0
    EXPECT_NEAR(output[1], 2.0f, 1e-5);  // voxel 0 feature 1
}

// ============================================================================
// NoiseSchedule Tests
// ============================================================================

TEST_F(UniversalOpsTest, NoiseScheduleLinear) {
    const int64_t num_steps = 1000;
    std::vector<float> timesteps = {0.0f, 250.0f, 500.0f, 750.0f, 999.0f};
    std::vector<float> alphas(timesteps.size(), 0.0f);
    std::vector<float> sigmas(timesteps.size(), 0.0f);

    Tensor t_tensor = Tensor::Wrap(timesteps.data(), {static_cast<int64_t>(timesteps.size())}, DType::F32);
    Tensor alpha_tensor = Tensor::Wrap(alphas.data(), {static_cast<int64_t>(alphas.size())}, DType::F32);
    Tensor sigma_tensor = Tensor::Wrap(sigmas.data(), {static_cast<int64_t>(sigmas.size())}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::NoiseSchedule, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "NoiseSchedule op not registered";
    }

    struct {
        int schedule_type = 0;
        int num_steps = 1000;
        float beta_start = 0.0001f;
        float beta_end = 0.02f;
        float s = 0.008f;
    } params;
    op->Execute({&t_tensor}, {&alpha_tensor, &sigma_tensor}, &params);

    // Alpha should decrease, sigma should increase with timestep
    // t=0: alpha high, sigma low
    // t=999: alpha low, sigma high
    EXPECT_GT(alphas[0], alphas[4]);
    EXPECT_LT(sigmas[0], sigmas[4]);

    // Verify alpha^2 + sigma^2 ≈ 1
    for (size_t i = 0; i < timesteps.size(); ++i) {
        float sum = alphas[i] * alphas[i] + sigmas[i] * sigmas[i];
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "At timestep " << timesteps[i];
    }
}

// ============================================================================
// BEVQuery Tests
// ============================================================================

TEST_F(UniversalOpsTest, BEVQueryBasic) {
    // Multi-camera to BEV projection
    const int64_t N_cam = 2;
    const int64_t H_feat = 4, W_feat = 4, D = 8;
    const int64_t H_bev = 4, W_bev = 4;

    std::vector<float> multi_cam_features(N_cam * H_feat * W_feat * D, 1.0f);
    std::vector<float> bev_queries(H_bev * W_bev * D, 0.0f);
    std::vector<float> cam_intrinsics(N_cam * 9, 0.0f);   // 3x3 per cam
    std::vector<float> cam_extrinsics(N_cam * 16, 0.0f);  // 4x4 per cam
    std::vector<float> output(H_bev * W_bev * D, 0.0f);

    // Setup simple camera matrices (identity-like)
    for (int64_t c = 0; c < N_cam; ++c) {
        // Intrinsic: focal=1, center at 0.5
        cam_intrinsics[c * 9 + 0] = 1.0f;  // fx
        cam_intrinsics[c * 9 + 4] = 1.0f;  // fy
        cam_intrinsics[c * 9 + 2] = 2.0f;  // cx
        cam_intrinsics[c * 9 + 5] = 2.0f;  // cy

        // Extrinsic: identity
        cam_extrinsics[c * 16 + 0] = 1.0f;
        cam_extrinsics[c * 16 + 5] = 1.0f;
        cam_extrinsics[c * 16 + 10] = 1.0f;
        cam_extrinsics[c * 16 + 15] = 1.0f;
    }

    Tensor feat = Tensor::Wrap(multi_cam_features.data(), {N_cam, H_feat, W_feat, D}, DType::F32);
    Tensor queries = Tensor::Wrap(bev_queries.data(), {H_bev, W_bev, D}, DType::F32);
    Tensor K = Tensor::Wrap(cam_intrinsics.data(), {N_cam, 3, 3}, DType::F32);
    Tensor T = Tensor::Wrap(cam_extrinsics.data(), {N_cam, 4, 4}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {H_bev, W_bev, D}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::BEVQuery, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "BEVQuery op not registered";
    }

    struct {
        int bev_h = 4;
        int bev_w = 4;
        float pc_range_x_min = -2.0f;
        float pc_range_x_max = 2.0f;
        float pc_range_y_min = -2.0f;
        float pc_range_y_max = 2.0f;
        float pc_range_z_min = -1.0f;
        float pc_range_z_max = 1.0f;
        int num_points = 4;
    } params;

    op->Execute({&feat, &queries, &K, &T}, {&out}, &params);

    // Output should have some values
    bool has_nonzero = false;
    for (float v : output) {
        if (v != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// AudioConv1D Tests
// ============================================================================

TEST_F(UniversalOpsTest, AudioConv1DBasic) {
    // [B, C_in, L] -> [B, C_out, L_out]
    const int64_t B = 1, C_in = 1, L_in = 8;
    const int64_t C_out = 2, K = 3, stride = 2, padding = 1;
    const int64_t L_out = (L_in + 2 * padding - K) / stride + 1;  // (8+2-3)/2+1 = 4

    std::vector<float> input(B * C_in * L_in);
    std::iota(input.begin(), input.end(), 1.0f);  // 1,2,3,...,8

    std::vector<float> weight(C_out * C_in * K, 1.0f);  // All 1s for sum
    std::vector<float> bias(C_out, 0.0f);
    std::vector<float> output(B * C_out * L_out, 0.0f);

    Tensor in_tensor = Tensor::Wrap(input.data(), {B, C_in, L_in}, DType::F32);
    Tensor w_tensor = Tensor::Wrap(weight.data(), {C_out, C_in, K}, DType::F32);
    Tensor b_tensor = Tensor::Wrap(bias.data(), {C_out}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(output.data(), {B, C_out, L_out}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::AudioConv1D, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "AudioConv1D op not registered";
    }

    AudioConv1DParams params;
    params.stride = stride;
    params.padding = padding;

    op->Execute({&in_tensor, &w_tensor, &b_tensor}, {&out_tensor}, &params);

    // With padding=1, stride=2, kernel=3, input=[1,2,3,4,5,6,7,8]
    // Position 0: [0,1,2] -> sum=3
    // Position 1: [2,3,4] -> sum=9
    // etc.
    EXPECT_NEAR(output[0], 3.0f, 1e-4);  // 0+1+2
    EXPECT_NEAR(output[1], 9.0f, 1e-4);  // 2+3+4
}

// ============================================================================
// PointAttention Tests
// ============================================================================

TEST_F(UniversalOpsTest, PointAttentionBasic) {
    // [B, N, C] features + [B, N, 3] positions -> [B, N, C] output
    const int64_t B = 1;
    const int64_t N = 100;  // Enough points to trigger KD-tree logic meaningfully
    const int64_t C = 32;

    std::vector<float> features(B * N * C, 1.0f);
    std::vector<float> positions(B * N * 3);

    // Create random positions in [0,1]
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::vector<float> output(B * N * C, 0.0f);

    Tensor feat_tensor = Tensor::Wrap(features.data(), {B, N, C}, DType::F32);
    Tensor pos_tensor = Tensor::Wrap(positions.data(), {B, N, 3}, DType::F32);
    Tensor out_tensor = Tensor::Wrap(output.data(), {B, N, C}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::PointAttention, DeviceType::CPU);
    if (!op) {
        GTEST_SKIP() << "PointAttention op not registered";
    }

    struct {
        int k = 16;
        int num_heads = 4;
        float scale = 0.0f;
        bool use_relative_pos = true;
    } params;

    op->Execute({&feat_tensor, &feat_tensor, &feat_tensor, &pos_tensor}, {&out_tensor}, &params);

    // Verify output is not all zeros/NaNs
    bool has_nonzero = false;
    for (float v : output) {
        if (std::isnan(v)) {
            FAIL() << "Output contains NaN";
        }
        if (v != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "PointAttention output should clearly not be all zeros for input 1.0";
}

TEST_F(UniversalOpsTest, PointAttentionSmallN) {
    // Edge case: N <= k
    const int64_t B = 1;
    const int64_t N = 5;  // Less than default k=16
    const int64_t C = 8;

    std::vector<float> features(B * N * C, 1.0f);
    std::vector<float> positions(B * N * 3);
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = (float)i / N;  // Line

    std::vector<float> output(B * N * C, 0.0f);

    Tensor feat = Tensor::Wrap(features.data(), {B, N, C}, DType::F32);
    Tensor pos = Tensor::Wrap(positions.data(), {B, N, 3}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {B, N, C}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::PointAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    struct {
        int k = 10;  // k > N
        int num_heads = 2;
        float scale = 0.0f;
        bool use_relative_pos = false;
    } params;

    // This should NOT crash or access invalid memory
    op->Execute({&feat, &feat, &feat, &pos}, {&out}, &params);

    // Verify output is computed (not zero)
    EXPECT_GT(output[0], 0.0f);
}

TEST_F(UniversalOpsTest, InvariantPointAttentionBasic) {
    // [B, L, D]
    const int64_t B = 1;
    const int64_t L = 10;
    const int64_t D = 16;
    const int64_t D_pair = 8;
    const int64_t H = 2;
    const int64_t Q = 4;
    const int64_t V = 4;

    std::vector<float> single_repr(B * L * D, 1.0f);
    std::vector<float> pair_repr(B * L * L * D_pair, 0.5f);
    std::vector<float> rotations(B * L * 3 * 3);
    std::vector<float> translations(B * L * 3, 0.0f);
    std::vector<float> query_points(H * Q * 3, 0.1f);
    std::vector<float> value_points(H * V * 3, 0.1f);
    std::vector<float> output(B * L * D, 0.0f);

    // Initialize rotations to identity
    for (int64_t i = 0; i < B * L; ++i) {
        rotations[i * 9 + 0] = 1.0f;
        rotations[i * 9 + 4] = 1.0f;
        rotations[i * 9 + 8] = 1.0f;
    }
    // Set some translations
    for (int64_t i = 0; i < L; ++i) {
        translations[i * 3 + 0] = (float)i;
    }

    Tensor s = Tensor::Wrap(single_repr.data(), {B, L, D}, DType::F32);
    Tensor p = Tensor::Wrap(pair_repr.data(), {B, L, L, D_pair}, DType::F32);
    Tensor r = Tensor::Wrap(rotations.data(), {B, L, 3, 3}, DType::F32);
    Tensor t = Tensor::Wrap(translations.data(), {B, L, 3}, DType::F32);
    Tensor qp = Tensor::Wrap(query_points.data(), {H, Q, 3}, DType::F32);
    Tensor vp = Tensor::Wrap(value_points.data(), {H, V, 3}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {B, L, D}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::InvariantPointAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    struct {
        int num_heads = 2;
        int num_query_points = 4;
        int num_value_points = 4;
        float scale = 0.0f;
    } params;

    op->Execute({&s, &p, &r, &t, &qp, &vp}, {&out}, &params);

    // Output should be non-zero (1.0 + attention updates)
    bool has_change = false;
    for (float v : output) {
        if (std::abs(v - 1.0f) > 1e-5f) has_change = true;
    }
    EXPECT_TRUE(has_change) << "Output should differ from input (initialized to 1.0)";
    EXPECT_FALSE(std::isnan(output[0]));
}

TEST_F(UniversalOpsTest, TriangularAttentionBasic) {
    // [B, L, L, D]
    const int64_t B = 1;
    const int64_t L = 4;
    const int64_t D = 8;
    const int64_t H = 2;
    const int64_t D_head = D / H;

    std::vector<float> pair_repr(B * L * L * D, 0.5f);
    // Weights: [H, D, D_head] or effectively [D, D_head] per head used in ProjectVector
    // Let's just random init them or fixed
    std::vector<float> qw(H * D * D_head, 0.1f);
    std::vector<float> kw(H * D * D_head, 0.1f);
    std::vector<float> vw(H * D * D_head, 0.1f);
    std::vector<float> output(B * L * L * D, 0.0f);

    Tensor pair = Tensor::Wrap(pair_repr.data(), {B, L, L, D}, DType::F32);
    Tensor q = Tensor::Wrap(qw.data(), {H, D, D_head}, DType::F32);
    Tensor k = Tensor::Wrap(kw.data(), {H, D, D_head}, DType::F32);
    Tensor v = Tensor::Wrap(vw.data(), {H, D, D_head}, DType::F32);
    Tensor out = Tensor::Wrap(output.data(), {B, L, L, D}, DType::F32);

    auto* op = OpRegistry::Instance().Get(OpType::TriangularAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    struct {
        int num_heads = 2;
        bool starting = true;
        float scale = 0.0f;
    } params;

    op->Execute({&pair, &q, &k, &v}, {&out}, &params);

    // Should differ from initial (0.5)
    bool has_change = false;
    for (float val : output) {
        if (std::abs(val - 0.5f) > 1e-5f) has_change = true;
    }
    EXPECT_TRUE(has_change);
}

}  // namespace
}  // namespace densecore
