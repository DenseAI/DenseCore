/**
 * @file test_activation_ops.cpp
 * @brief Unit tests for ActivationOps OpRegistry integration
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class ActivationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test instance to isolate from global registrations
        test_registry_ = std::make_unique<OpRegistry>();
        OpRegistry::SetTestInstance(std::move(test_registry_));
    }

    void TearDown() override { OpRegistry::ResetToGlobalInstance(); }

    std::unique_ptr<OpRegistry> test_registry_;
};

// Test that SiLU op is registered and executable
TEST_F(ActivationOpsTest, SiLURegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::SiLU, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "SiLU op should be registered for CPU";

    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr) << "Op should implement ActivationOps interface";
}

// Test that GELU op is registered and executable
TEST_F(ActivationOpsTest, GELURegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::GELU, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "GELU op should be registered for CPU";

    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr) << "Op should implement ActivationOps interface";
}

// Test that Softmax op is registered and executable
TEST_F(ActivationOpsTest, SoftmaxRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::Softmax, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "Softmax op should be registered for CPU";

    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr) << "Op should implement ActivationOps interface";
}

// Test SiLU correctness
TEST_F(ActivationOpsTest, SiLUCorrectness) {
    auto* op = OpRegistry::Instance().GetBest(OpType::SiLU, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr);

    // Create input tensor
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output_data(5, 0.0f);

    Tensor input;
    input.data = input_data.data();
    input.dtype = DType::F32;
    input.ndim = 1;
    input.shape[0] = 5;
    input.stride[0] = sizeof(float);

    Tensor output;
    output.data = output_data.data();
    output.dtype = DType::F32;
    output.ndim = 1;
    output.shape[0] = 5;
    output.stride[0] = sizeof(float);

    activation_op->SiLU(input, &output);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (int i = 0; i < 5; ++i) {
        float x = input_data[i];
        float expected = x / (1.0f + std::exp(-x));
        EXPECT_NEAR(output_data[i], expected, 1e-4f) << "SiLU mismatch at index " << i;
    }
}

// Test GELU correctness
TEST_F(ActivationOpsTest, GELUCorrectness) {
    auto* op = OpRegistry::Instance().GetBest(OpType::GELU, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr);

    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output_data(5, 0.0f);

    Tensor input;
    input.data = input_data.data();
    input.dtype = DType::F32;
    input.ndim = 1;
    input.shape[0] = 5;
    input.stride[0] = sizeof(float);

    Tensor output;
    output.data = output_data.data();
    output.dtype = DType::F32;
    output.ndim = 1;
    output.shape[0] = 5;
    output.stride[0] = sizeof(float);

    activation_op->GELU(input, &output);

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    for (int i = 0; i < 5; ++i) {
        float x = input_data[i];
        float x3 = x * x * x;
        float tanh_arg = kSqrt2OverPi * (x + kCoeff * x3);
        float expected = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        EXPECT_NEAR(output_data[i], expected, 1e-4f) << "GELU mismatch at index " << i;
    }
}

// Test Softmax correctness
TEST_F(ActivationOpsTest, SoftmaxCorrectness) {
    auto* op = OpRegistry::Instance().GetBest(OpType::Softmax, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> output_data(5, 0.0f);

    Tensor input;
    input.data = input_data.data();
    input.dtype = DType::F32;
    input.ndim = 1;
    input.shape[0] = 5;
    input.stride[0] = sizeof(float);

    Tensor output;
    output.data = output_data.data();
    output.dtype = DType::F32;
    output.ndim = 1;
    output.shape[0] = 5;
    output.stride[0] = sizeof(float);

    activation_op->Softmax(input, &output);

    // Verify sum = 1
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        sum += output_data[i];
        EXPECT_GE(output_data[i], 0.0f) << "Softmax output should be non-negative";
        EXPECT_LE(output_data[i], 1.0f) << "Softmax output should be <= 1";
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax outputs should sum to 1";

    // Verify monotonicity (larger input -> larger output)
    for (int i = 1; i < 5; ++i) {
        EXPECT_GT(output_data[i], output_data[i - 1])
            << "Softmax should preserve input ordering";
    }
}

// Test large tensor performance
TEST_F(ActivationOpsTest, SiLULargeTensor) {
    auto* op = OpRegistry::Instance().GetBest(OpType::SiLU, DeviceType::CPU);
    ASSERT_NE(op, nullptr);
    auto* activation_op = dynamic_cast<ActivationOps*>(op);
    ASSERT_NE(activation_op, nullptr);

    // 4096 * 11008 = typical LLaMA intermediate size
    constexpr size_t n = 4096 * 11008;
    std::vector<float> input_data(n, 1.5f);
    std::vector<float> output_data(n, 0.0f);

    Tensor input;
    input.data = input_data.data();
    input.dtype = DType::F32;
    input.ndim = 1;
    input.shape[0] = static_cast<int64_t>(n);
    input.stride[0] = sizeof(float);

    Tensor output;
    output.data = output_data.data();
    output.dtype = DType::F32;
    output.ndim = 1;
    output.shape[0] = static_cast<int64_t>(n);
    output.stride[0] = sizeof(float);

    activation_op->SiLU(input, &output);

    // Verify result
    float expected = 1.5f / (1.0f + std::exp(-1.5f));
    EXPECT_NEAR(output_data[0], expected, 1e-4f);
    EXPECT_NEAR(output_data[n - 1], expected, 1e-4f);
}

}  // namespace
}  // namespace densecore
