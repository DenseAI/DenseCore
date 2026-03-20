#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class GroupNormOpTest : public ::testing::Test {
protected:
    void SetUp() override { OpRegistry::Init(); }

    // Helper to calculate expected GroupNorm manually
    void CalculateExpected(const std::vector<float>& input, int B, int C, int H, int W, int groups,
                           const std::vector<float>& gamma, const std::vector<float>& beta,
                           std::vector<float>& expected, bool is_nhwc) {

        int channels_per_group = C / groups;
        // Group size per batch
        int group_size = channels_per_group * H * W;

        for (int b = 0; b < B; ++b) {
            for (int g = 0; g < groups; ++g) {
                // 1. Mean
                float sum = 0;
                for (int c_local = 0; c_local < channels_per_group; ++c_local) {
                    int c = g * channels_per_group + c_local;
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx;
                            if (is_nhwc)
                                idx = b * H * W * C + h * W * C + w * C + c;
                            else
                                idx = b * C * H * W + c * H * W + h * W + w;
                            sum += input[idx];
                        }
                    }
                }
                float mean = sum / group_size;

                // 2. Var
                float sum_sq = 0;
                for (int c_local = 0; c_local < channels_per_group; ++c_local) {
                    int c = g * channels_per_group + c_local;
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx;
                            if (is_nhwc)
                                idx = b * H * W * C + h * W * C + w * C + c;
                            else
                                idx = b * C * H * W + c * H * W + h * W + w;
                            float d = input[idx] - mean;
                            sum_sq += d * d;
                        }
                    }
                }
                float var = sum_sq / group_size;
                float inv_std = 1.0f / std::sqrt(var + 1e-5f);

                // 3. Apply
                for (int c_local = 0; c_local < channels_per_group; ++c_local) {
                    int c = g * channels_per_group + c_local;
                    float scale = gamma[c] * inv_std;
                    float shift = beta[c] - mean * scale;

                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx;
                            if (is_nhwc)
                                idx = b * H * W * C + h * W * C + w * C + c;
                            else
                                idx = b * C * H * W + c * H * W + h * W + w;
                            expected[idx] = input[idx] * scale + shift;
                        }
                    }
                }
            }
        }
    }
};

TEST_F(GroupNormOpTest, NCHW_Basic) {
    const int B = 1, C = 4, H = 2, W = 2;
    const int G = 2;  // 2 channels per group

    std::vector<float> input(B * C * H * W);
    // Fill with pattern
    std::iota(input.begin(), input.end(), 0.0f);  // 0, 1, ...

    std::vector<float> gamma(C, 1.0f);
    std::vector<float> beta(C, 0.0f);
    std::vector<float> output(B * C * H * W);
    std::vector<float> expected(B * C * H * W);

    CalculateExpected(input, B, C, H, W, G, gamma, beta, expected, false);

    Tensor t_in = Tensor::Wrap(input.data(), {B, C, H, W}, DType::F32);
    // Default layout is unknown (assumed NCHW)
    Tensor t_out = Tensor::Wrap(output.data(), {B, C, H, W}, DType::F32);
    Tensor t_gamma = Tensor::Wrap(gamma.data(), {C}, DType::F32);
    Tensor t_beta = Tensor::Wrap(beta.data(), {C}, DType::F32);

    auto* op = OpRegistry::Instance().GetBest(OpType::GroupNorm, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* norm = dynamic_cast<NormalizationOps*>(op);
    ASSERT_NE(norm, nullptr);

    norm->GroupNorm(t_in, t_gamma, t_beta, G, 1e-5f, &t_out);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-4);
    }
}

TEST_F(GroupNormOpTest, NHWC_Basic) {
    const int B = 1, C = 4, H = 2, W = 2;
    const int G = 2;

    std::vector<float> input(B * C * H * W);
    std::iota(input.begin(), input.end(), 0.0f);

    std::vector<float> gamma(C, 1.0f);
    std::vector<float> beta(C, 0.0f);
    std::vector<float> output(B * C * H * W);
    std::vector<float> expected(B * C * H * W);

    CalculateExpected(input, B, C, H, W, G, gamma, beta, expected, true);

    // Wrap as NHWC
    std::vector<int64_t> shape = {B, H, W, C};
    Tensor t_in = Tensor::Wrap(input.data(), shape, DType::F32);
    t_in.layout = TensorLayout::NHWC;
    // IMPORTANT: Tensor::Wrap assigns layout::UNKNOWN. We manually set NHWC.

    Tensor t_out = Tensor::Wrap(output.data(), shape, DType::F32);
    t_out.layout = TensorLayout::NHWC;

    Tensor t_gamma = Tensor::Wrap(gamma.data(), {C}, DType::F32);
    Tensor t_beta = Tensor::Wrap(beta.data(), {C}, DType::F32);

    auto* op = OpRegistry::Instance().GetBest(OpType::GroupNorm, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* norm = dynamic_cast<NormalizationOps*>(op);

    norm->GroupNorm(t_in, t_gamma, t_beta, G, 1e-5f, &t_out);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-4);
    }
}

TEST_F(GroupNormOpTest, InvalidGroups) {
    const int B = 1, C = 4, H = 2, W = 2;
    std::vector<float> input(B * C * H * W, 1.0f);
    std::vector<float> gamma(C, 1.0f);
    std::vector<float> beta(C, 0.0f);
    std::vector<float> output(B * C * H * W, -1.0f);

    Tensor t_in = Tensor::Wrap(input.data(), {B, C, H, W}, DType::F32);
    Tensor t_out = Tensor::Wrap(output.data(), {B, C, H, W}, DType::F32);
    Tensor t_gamma = Tensor::Wrap(gamma.data(), {C}, DType::F32);
    Tensor t_beta = Tensor::Wrap(beta.data(), {C}, DType::F32);

    auto* op = OpRegistry::Instance().GetBest(OpType::GroupNorm, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* norm = dynamic_cast<NormalizationOps*>(op);

    // 1. num_groups = 0
    norm->GroupNorm(t_in, t_gamma, t_beta, 0, 1e-5f, &t_out);
    for (float v : output) EXPECT_EQ(v, -1.0f);

    // 2. C % num_groups != 0
    norm->GroupNorm(t_in, t_gamma, t_beta, 3, 1e-5f, &t_out);
    for (float v : output) EXPECT_EQ(v, -1.0f);

    // 3. num_groups < 0
    norm->GroupNorm(t_in, t_gamma, t_beta, -1, 1e-5f, &t_out);
    for (float v : output) EXPECT_EQ(v, -1.0f);
}

}  // namespace
}  // namespace densecore
