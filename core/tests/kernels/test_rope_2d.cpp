#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class RoPE2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CPU ops are registered
        // OpRegistry::Init() is called automatically
    }
};

TEST_F(RoPE2DTest, BasicRotation) {
    // Setup: 1 Batch, 1 Head, 1 Token, HeadDim=4 (Half=2)
    // Token at (y=1, x=1)
    // Pos H=1, Pos W=1
    // Input: [1.0, 0.0, 1.0, 0.0]
    //   Half 1 (H): [1.0, 0.0] -> Rotate by angle(1)
    //   Half 2 (W): [1.0, 0.0] -> Rotate by angle(1)

    // CosSin table: [MaxPos, HeadDim]
    // Row 1 (Pos 1):
    //   Cols [0, 1]: Cos(theta), Sin(theta) for H
    //   Cols [2, 3]: Cos(theta), Sin(theta) for W (assuming symmetric freqs)

    // Let theta = pi/2 (90 deg). Cos=0, Sin=1.
    // Rotated vector (x, y) by 90 deg -> (-y, x).
    // Input (1, 0) -> (0, 1).

    // So expected output: [0.0, 1.0, 0.0, 1.0]

    // 1. Create Tensors
    std::vector<int64_t> input_shape = {1, 1, 1, 4};  // B, H, N, D
    std::vector<float> input_data = {1.0f, 0.0f, 1.0f, 0.0f};

    std::vector<int64_t> pos_shape = {1};
    std::vector<int> pos_h_data = {1};
    std::vector<int> pos_w_data = {1};

    // CosSin: Size [2, 4] (Rows for pos 0, 1)
    // Row 1: 0.0, 1.0, 0.0, 1.0 (Cos, Sin, Cos, Sin)
    std::vector<int64_t> cs_shape = {2, 4};
    std::vector<float> cs_data = {
        1.0f, 0.0f, 1.0f, 0.0f,  // Pos 0 (Identity)
        0.0f, 1.0f, 0.0f, 1.0f   // Pos 1 (90 deg)
    };

    Tensor input = Tensor::Wrap(input_data.data(), input_shape, DType::F32);
    Tensor pos_h = Tensor::Wrap(pos_h_data.data(), pos_shape, DType::INT32);
    Tensor pos_w = Tensor::Wrap(pos_w_data.data(), pos_shape, DType::INT32);
    Tensor cos_sin = Tensor::Wrap(cs_data.data(), cs_shape, DType::F32);

    // Output tensor
    std::vector<float> output_data(4);
    Tensor output = Tensor::Wrap(output_data.data(), input_shape, DType::F32);

    // 2. Get Op
    // Need to manually register the op if we are stripping symbols?
    // densecore library should have registered it.
    auto* op = OpRegistry::Instance().GetBest(OpType::RoPE2D, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "RoPE2D op not found for CPU";

    // 3. Execute
    auto* embedding_op = dynamic_cast<EmbeddingOps*>(op);
    ASSERT_NE(embedding_op, nullptr);

    embedding_op->RoPE2D(input, pos_h, pos_w, cos_sin, &output);

    // 4. Verify
    // Expected: [0.0, 1.0, 0.0, 1.0]
    EXPECT_NEAR(output_data[0], 0.0f, 1e-5);
    EXPECT_NEAR(output_data[1], 1.0f, 1e-5);
    EXPECT_NEAR(output_data[2], 0.0f, 1e-5);
    EXPECT_NEAR(output_data[3], 1.0f, 1e-5);
}

TEST_F(RoPE2DTest, OutOfBoundsPosition) {
    // Setup: 1 Batch, 1 Head, 1 Token, HeadDim=4
    // Invalid position: pos_h = 10 (max_seq_len will be 2)
    std::vector<int64_t> input_shape = {1, 1, 1, 4};
    std::vector<float> input_data = {1.23f, 4.56f, 7.89f, 0.12f};
    std::vector<int64_t> pos_shape = {1};
    std::vector<int> pos_h_data = {10};  // OOB
    std::vector<int> pos_w_data = {0};

    // CosSin: Size [2, 4] -> max_seq_len = 2
    std::vector<int64_t> cs_shape = {2, 4};
    std::vector<float> cs_data(8, 0.0f);

    Tensor input = Tensor::Wrap(input_data.data(), input_shape, DType::F32);
    Tensor pos_h = Tensor::Wrap(pos_h_data.data(), pos_shape, DType::INT32);
    Tensor pos_w = Tensor::Wrap(pos_w_data.data(), pos_shape, DType::INT32);
    Tensor cos_sin = Tensor::Wrap(cs_data.data(), cs_shape, DType::F32);

    std::vector<float> output_data(4, -1.0f);
    Tensor output = Tensor::Wrap(output_data.data(), input_shape, DType::F32);

    auto* op = OpRegistry::Instance().GetBest(OpType::RoPE2D, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* embedding_op = dynamic_cast<EmbeddingOps*>(op);

    embedding_op->RoPE2D(input, pos_h, pos_w, cos_sin, &output);

    // Since pos_h is OOB, it should have copied input to output (Safe Fullback)
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
    }
}

}  // namespace
}  // namespace densecore
