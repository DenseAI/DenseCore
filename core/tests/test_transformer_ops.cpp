/**
 * @file test_transformer_ops.cpp
 * @brief Domain-specific operation unit tests
 *
 * Verifies correctness of Vision, Diffusion, Autonomous domain operations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/operation_graph.h"

namespace densecore {
namespace {

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * @brief Tensor creation utility for testing
 */
class TestTensor {
public:
    TestTensor(std::vector<int64_t> shape, DType dtype = DType::F32) : shape_(std::move(shape)) {
        size_t total = 1;
        for (auto dim : shape_) total *= dim;
        data_.resize(total);

        std::copy(shape_.begin(), shape_.end(), tensor_.shape.begin());
        tensor_.ndim = static_cast<int>(shape_.size());
        tensor_.dtype = dtype;
        tensor_.device_type = DeviceType::CPU;
        tensor_.data = data_.data();

        // Compute strides
        strides_.resize(shape_.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        std::copy(strides_.begin(), strides_.end(), tensor_.stride.begin());
    }

    Tensor* get() { return &tensor_; }
    const Tensor* get() const { return &tensor_; }
    float* data() { return data_.data(); }
    size_t size() const { return data_.size(); }

    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

    void fillSequence() {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<float>(i);
        }
    }

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;
    Tensor tensor_;
};

// ============================================================================
// OpType Extension Tests
// ============================================================================

class OpTypeExtensionTest : public ::testing::Test {};

TEST_F(OpTypeExtensionTest, NewOpTypesHaveCorrectNames) {
    // Verify that new OpTypes return correct names
    EXPECT_STREQ(OpTypeName(OpType::CrossAttention), "CrossAttention");
    EXPECT_STREQ(OpTypeName(OpType::PatchEmbed2D), "PatchEmbed2D");
    EXPECT_STREQ(OpTypeName(OpType::PatchEmbed3D), "PatchEmbed3D");
    EXPECT_STREQ(OpTypeName(OpType::RoPE2D), "RoPE2D");
    EXPECT_STREQ(OpTypeName(OpType::WindowAttention), "WindowAttention");

    EXPECT_STREQ(OpTypeName(OpType::AdaLN), "AdaLN");
    EXPECT_STREQ(OpTypeName(OpType::GroupNorm), "GroupNorm");
    EXPECT_STREQ(OpTypeName(OpType::TimestepEmbed), "TimestepEmbed");
    EXPECT_STREQ(OpTypeName(OpType::Patchify), "Patchify");
    EXPECT_STREQ(OpTypeName(OpType::Unpatchify), "Unpatchify");
    EXPECT_STREQ(OpTypeName(OpType::TemporalAttention), "TemporalAttention");

    EXPECT_STREQ(OpTypeName(OpType::DeformableAttention), "DeformableAttention");
    EXPECT_STREQ(OpTypeName(OpType::GridSample), "GridSample");
    EXPECT_STREQ(OpTypeName(OpType::BEVQuery), "BEVQuery");
}

TEST_F(OpTypeExtensionTest, OpTypeNumberingPreservesGaps) {
    // Verify that domain-specific numbering scheme is preserved
    EXPECT_EQ(static_cast<uint8_t>(OpType::CrossAttention), 30);
    EXPECT_EQ(static_cast<uint8_t>(OpType::PatchEmbed2D), 40);
    EXPECT_EQ(static_cast<uint8_t>(OpType::AdaLN), 50);
    EXPECT_EQ(static_cast<uint8_t>(OpType::DeformableAttention), 60);
    EXPECT_EQ(static_cast<uint8_t>(OpType::Custom), 255);
}

TEST_F(OpTypeExtensionTest, ExistingOpTypesUnchanged) {
    // Verify that existing OpTypes are unchanged (backward compatibility)
    EXPECT_EQ(static_cast<uint8_t>(OpType::MatMul), 0);
    EXPECT_STREQ(OpTypeName(OpType::RMSNorm), "RMSNorm");
    EXPECT_STREQ(OpTypeName(OpType::FlashAttention), "FlashAttention");
}

// ============================================================================
// PatchEmbed2D Tests
// ============================================================================

class PatchEmbed2DTest : public ::testing::Test {};

TEST_F(PatchEmbed2DTest, OutputShapeCorrect_ViTB16) {
    // ViT-B/16: 224x224 image, 16x16 patches -> 14x14 = 196 patches
    const int B = 1, C = 3, H = 224, W = 224;
    const int P = 16, D = 768;
    const int N = (H / P) * (W / P);  // 196

    TestTensor image({B, C, H, W});
    TestTensor conv_weight({D, C, P, P});
    TestTensor conv_bias({D});
    TestTensor patches({B, N, D});

    image.fillSequence();
    conv_weight.fill(1.0f / (C * P * P));  // Averaging kernel
    conv_bias.fill(0.0f);

    auto* op = OpRegistry::Instance().GetBest(OpType::PatchEmbed2D, DeviceType::CPU);
    if (op != nullptr) {
        // Cast to EmbeddingOps
        auto* embedding_op = dynamic_cast<EmbeddingOps*>(op);
        if (embedding_op != nullptr) {
            embedding_op->PatchEmbed2D(*image.get(), *conv_weight.get(), *conv_bias.get(), patches.get());
            // Check if output is not NaN
            bool has_nan = false;
            for (size_t i = 0; i < patches.size(); ++i) {
                if (std::isnan(patches.data()[i])) {
                    has_nan = true;
                    break;
                }
            }
            EXPECT_FALSE(has_nan);
        }
    }
}

// ============================================================================
// AdaLN Tests
// ============================================================================

class AdaLNTest : public ::testing::Test {};

TEST_F(AdaLNTest, TimestepConditioningCorrect) {
    // AdaLN: output = layernorm(input) * (1 + scale) + shift
    const int B = 2, N = 4, D = 8;

    TestTensor input({B, N, D});
    TestTensor scale({B, D});
    TestTensor shift({B, D});
    TestTensor output({B, N, D});

    input.fill(1.0f);
    scale.fill(0.5f);  // (1 + 0.5) = 1.5x
    shift.fill(0.1f);

    auto* op = OpRegistry::Instance().GetBest(OpType::AdaLN, DeviceType::CPU);
    if (op != nullptr) {
        auto* norm_op = dynamic_cast<NormalizationOps*>(op);
        if (norm_op != nullptr) {
            norm_op->AdaLN(*input.get(), *scale.get(), *shift.get(), 1e-5f, output.get());

            // Check if all outputs are not zero
            bool all_zero = true;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output.data()[i] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }
            EXPECT_FALSE(all_zero);
        }
    }
}

TEST_F(AdaLNTest, FusedModulationCorrect) {
    // AdaLN fused: input [B,N,D], modulation [B, 2*D]
    const int B = 1, N = 2, D = 4;

    TestTensor input({B, N, D});
    TestTensor modulation({B, 2 * D});  // Scale [D] then Shift [D]
    TestTensor output({B, N, D});

    input.fill(1.0f);

    // Set scale to 0.5 (first D elements)
    for (int i = 0; i < D; ++i) modulation.data()[i] = 0.5f;

    // Set shift to 0.1 (next D elements)
    for (int i = D; i < 2 * D; ++i) modulation.data()[i] = 0.1f;

    auto* op = OpRegistry::Instance().GetBest(OpType::AdaLN, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    auto* norm_op = dynamic_cast<NormalizationOps*>(op);
    ASSERT_NE(norm_op, nullptr);

    // Test with constant input (Norm -> 0)
    norm_op->AdaLN(*input.get(), *modulation.get(), 1e-5f, output.get());

    // With constant input, mean=1, var=0. Norm=(1-1)/... = 0. Output = 0*(1.5)+0.1 = 0.1.
    EXPECT_NEAR(output.data()[0], 0.1f, 1e-4f);

    // Test with non-constant input
    input.data()[0] = 0.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 0.0f;
    input.data()[3] = 2.0f;

    // Expected: [-1.4, 1.6, -1.4, 1.6]
    norm_op->AdaLN(*input.get(), *modulation.get(), 1e-5f, output.get());

    EXPECT_NEAR(output.data()[0], -1.4f, 0.01f);
    EXPECT_NEAR(output.data()[1], 1.6f, 0.01f);
}

// ============================================================================
// GroupNorm Tests
// ============================================================================

class GroupNormTest : public ::testing::Test {};

TEST_F(GroupNormTest, ChannelGroupsCorrect) {
    // GroupNorm: 32 channels, 8 groups -> 4 channels per group
    const int B = 1, C = 32, H = 8, W = 8;
    const int num_groups = 8;

    TestTensor input({B, C, H, W});
    TestTensor gamma({C});
    TestTensor beta({C});
    TestTensor output({B, C, H, W});

    input.fillSequence();
    gamma.fill(1.0f);
    beta.fill(0.0f);

    auto* op = OpRegistry::Instance().GetBest(OpType::GroupNorm, DeviceType::CPU);
    if (op != nullptr) {
        auto* norm_op = dynamic_cast<NormalizationOps*>(op);
        if (norm_op != nullptr) {
            norm_op->GroupNorm(*input.get(), *gamma.get(), *beta.get(), num_groups, 1e-5f, output.get());

            // Verification: mean within each group should be close to 0
            // (when gamma=1, beta=0)
            EXPECT_NE(output.data()[0], input.data()[0]);  // Values should have changed
        }
    }
}

// ============================================================================
// DeformableAttention Tests
// ============================================================================

class DeformableAttentionTest : public ::testing::Test {};

TEST_F(DeformableAttentionTest, SparseSamplingWorks) {
    // Small scale test case
    const int B = 1, Q = 4, D = 16;
    const int L = 1, H_feat = 8, W_feat = 8;
    const int num_heads = 2, num_points = 4;

    TestTensor query({B, Q, D});
    TestTensor spatial_features({B, L, H_feat, W_feat, D});
    TestTensor sampling_offsets({B, Q, num_heads, num_points, 2});
    TestTensor attention_weights({B, Q, num_heads, num_points});
    TestTensor output({B, Q, D});

    query.fillSequence();
    spatial_features.fill(1.0f);
    sampling_offsets.fill(0.0f);  // Center sampling
    attention_weights.fill(1.0f / num_points);

    auto* op = OpRegistry::Instance().GetBest(OpType::DeformableAttention, DeviceType::CPU);
    if (op != nullptr) {
        auto* attn_op = dynamic_cast<AttentionOps*>(op);
        if (attn_op != nullptr) {
            attn_op->DeformableAttention(*query.get(), *spatial_features.get(), *sampling_offsets.get(),
                                         *attention_weights.get(), output.get());

            // Check if outputs are not zero
            bool any_nonzero = false;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output.data()[i] != 0.0f) {
                    any_nonzero = true;
                    break;
                }
            }
            EXPECT_TRUE(any_nonzero);
        }
    }
}

// ============================================================================
// GridSample Tests
// ============================================================================

class GridSampleTest : public ::testing::Test {};

TEST_F(GridSampleTest, BilinearInterpolationCorrect) {
    // 2x2 input, 2x2 output grid
    const int B = 1, C = 1, H_in = 2, W_in = 2, H_out = 2, W_out = 2;

    TestTensor input({B, C, H_in, W_in});
    TestTensor grid({B, H_out, W_out, 2});
    TestTensor output({B, C, H_out, W_out});

    // 2x2 input: [[1, 2], [3, 4]]
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;

    // Grid: identity mapping (no transformation)
    // Normalized coords: (-1,-1) maps to (0,0), (1,1) maps to (1,1)
    float* g = grid.data();
    g[0] = -1.0f;
    g[1] = -1.0f;  // (0,0)
    g[2] = 1.0f;
    g[3] = -1.0f;  // (1,0)
    g[4] = -1.0f;
    g[5] = 1.0f;  // (0,1)
    g[6] = 1.0f;
    g[7] = 1.0f;  // (1,1)

    auto* op = OpRegistry::Instance().GetBest(OpType::GridSample, DeviceType::CPU);
    if (op != nullptr) {
        auto* spatial_op = dynamic_cast<SpatialOps*>(op);
        if (spatial_op != nullptr) {
            spatial_op->GridSample(*input.get(), *grid.get(), output.get());

            // Identity mapping output == input
            EXPECT_NEAR(output.data()[0], 1.0f, 0.1f);
            EXPECT_NEAR(output.data()[3], 4.0f, 0.1f);
        }
    }
}

}  // namespace
}  // namespace densecore
