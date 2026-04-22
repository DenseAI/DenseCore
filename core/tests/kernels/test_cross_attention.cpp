/**
 * @file test_cross_attention.cpp
 * @brief Unit tests for CrossAttention kernel
 *
 * Tests for the Encoder-Decoder cross-attention implementation.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

namespace densecore {
namespace {

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * @brief Create a test tensor with specified shape
 */
class CrossAttentionTestTensor {
public:
    CrossAttentionTestTensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
        size_t total = 1;
        for (auto dim : shape_) total *= dim;
        data_.resize(total);

        std::copy(shape_.begin(), shape_.end(), tensor_.shape.begin());
        tensor_.ndim = static_cast<int>(shape_.size());
        tensor_.dtype = DType::F32;
        tensor_.device_type = DeviceType::CPU;
        tensor_.data = data_.data();
        tensor_.layout = TensorLayout::SEQ;

        // Compute strides (BHSD format)
        strides_.resize(shape_.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        std::copy(strides_.begin(), strides_.end(), tensor_.stride.begin());
    }

    Tensor* get() { return &tensor_; }
    float* data() { return data_.data(); }
    size_t size() const { return data_.size(); }

    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

    void fillSequence(float start = 0.0f, float step = 0.01f) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = start + step * static_cast<float>(i);
        }
    }

    void fillRandom(float min = -1.0f, float max = 1.0f) {
        for (size_t i = 0; i < data_.size(); ++i) {
            float t = static_cast<float>(rand()) / RAND_MAX;
            data_[i] = min + t * (max - min);
        }
    }

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;
    Tensor tensor_;
};

// ============================================================================
// CrossAttention Tests
// ============================================================================

class CrossAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        srand(42);
    }
};

TEST_F(CrossAttentionTest, KernelRegistered) {
    // Verify CrossAttention kernel is registered in OpRegistry
    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    EXPECT_NE(op, nullptr) << "CrossAttention kernel should be registered";
}

TEST_F(CrossAttentionTest, BasicEncoderDecoderShape) {
    // Test with typical Encoder-Decoder dimensions
    // Q: [batch=1, seq_q=4, n_head=4, head_dim=8]
    // K/V: [batch=1, seq_k=8, n_head=4, head_dim=8]
    const int B = 1, Lq = 4, Lk = 8, H = 4, D = 8;

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H, D});
    CrossAttentionTestTensor V({B, Lk, H, D});
    CrossAttentionTestTensor output({B, Lq, H, D});

    Q.fillSequence(0.1f, 0.01f);
    K.fillSequence(0.2f, 0.01f);
    V.fill(1.0f);  // Uniform values for easy verification
    output.fill(0.0f);

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    CrossAttentionParams params;
    params.scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<Tensor*> inputs = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs = {output.get()};

    op->Execute(inputs, outputs, &params);

    // Verify output is not all zeros (attention computed something)
    bool has_nonzero = false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output.data()[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";

    // Since V is all 1.0, and softmax sums to 1, output should be close to 1.0
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output.data()[i], 1.0f, 0.01f)
            << "Output at index " << i << " should be close to 1.0 when V is uniform";
    }
}

TEST_F(CrossAttentionTest, AttentionScaleEffect) {
    // Test that scale parameter affects attention scores
    const int B = 1, Lq = 2, Lk = 4, H = 2, D = 4;

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H, D});
    CrossAttentionTestTensor V({B, Lk, H, D});
    CrossAttentionTestTensor output1({B, Lq, H, D});
    CrossAttentionTestTensor output2({B, Lq, H, D});

    Q.fillSequence(0.5f, 0.1f);
    K.fillSequence(0.3f, 0.1f);
    V.fillSequence(1.0f, 0.5f);

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    // Execute with scale = 1.0
    CrossAttentionParams params1;
    params1.scale = 1.0f;
    std::vector<Tensor*> inputs1 = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs1 = {output1.get()};
    op->Execute(inputs1, outputs1, &params1);

    // Execute with scale = 0.5
    CrossAttentionParams params2;
    params2.scale = 0.5f;
    std::vector<Tensor*> inputs2 = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs2 = {output2.get()};
    op->Execute(inputs2, outputs2, &params2);

    // Outputs should be different due to different scales
    bool outputs_differ = false;
    for (size_t i = 0; i < output1.size(); ++i) {
        if (std::abs(output1.data()[i] - output2.data()[i]) > 1e-5f) {
            outputs_differ = true;
            break;
        }
    }
    EXPECT_TRUE(outputs_differ) << "Different scales should produce different outputs";
}

TEST_F(CrossAttentionTest, NoCausalMasking) {
    // Cross-attention should have no causal masking
    // Each query position should attend to ALL key positions
    const int B = 1, Lq = 1, Lk = 4, H = 1, D = 4;

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H, D});
    CrossAttentionTestTensor V({B, Lk, H, D});
    CrossAttentionTestTensor output({B, Lq, H, D});

    // Set Q to match only the last K position
    Q.fill(0.0f);
    Q.data()[3] = 10.0f;  // Last dimension high

    K.fill(0.0f);
    // K[0] has high value at dim 0
    K.data()[0] = 1.0f;
    // K[1] has high value at dim 1
    K.data()[4 + 1] = 1.0f;
    // K[2] has high value at dim 2
    K.data()[8 + 2] = 1.0f;
    // K[3] has high value at dim 3 (matches Q)
    K.data()[12 + 3] = 10.0f;

    // Different values for each V
    V.fill(0.0f);
    V.data()[0] = 1.0f;   // V[0]
    V.data()[4] = 2.0f;   // V[1]
    V.data()[8] = 3.0f;   // V[2]
    V.data()[12] = 4.0f;  // V[3] - should dominate

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    CrossAttentionParams params;
    params.scale = 1.0f;

    std::vector<Tensor*> inputs = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs = {output.get()};
    op->Execute(inputs, outputs, &params);

    // Output should be weighted toward V[3] since Q·K is highest for K[3]
    // The output at dim 0 should be closer to V[3] value (4.0)
    EXPECT_GT(output.data()[0], 3.0f) << "Output should be weighted toward V[3]";
}

TEST_F(CrossAttentionTest, MultiHeadParallel) {
    // Test that multiple heads produce independent results
    const int B = 1, Lq = 2, Lk = 3, H = 4, D = 4;

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H, D});
    CrossAttentionTestTensor V({B, Lk, H, D});
    CrossAttentionTestTensor output({B, Lq, H, D});

    Q.fillRandom();
    K.fillRandom();
    V.fillRandom();
    output.fill(0.0f);

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    CrossAttentionParams params;
    params.scale = 1.0f;

    std::vector<Tensor*> inputs = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs = {output.get()};
    op->Execute(inputs, outputs, &params);

    // Verify no NaN or Inf in output
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output.data()[i])) << "Output should not contain NaN";
        EXPECT_FALSE(std::isinf(output.data()[i])) << "Output should not contain Inf";
    }
}

TEST_F(CrossAttentionTest, BatchProcessing) {
    // Test batch processing
    const int B = 4, Lq = 2, Lk = 3, H = 2, D = 4;

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H, D});
    CrossAttentionTestTensor V({B, Lk, H, D});
    CrossAttentionTestTensor output({B, Lq, H, D});

    Q.fillRandom();
    K.fillRandom();
    V.fill(1.0f);  // Uniform for predictable output
    output.fill(0.0f);

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    CrossAttentionParams params;
    params.scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<Tensor*> inputs = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs = {output.get()};
    op->Execute(inputs, outputs, &params);

    // All outputs should be ~1.0 since V is uniform
    for (int b = 0; b < B; ++b) {
        for (size_t i = 0; i < static_cast<size_t>(Lq * H * D); ++i) {
            size_t idx = b * Lq * H * D + i;
            EXPECT_NEAR(output.data()[idx], 1.0f, 0.01f) << "Batch " << b << " output should be ~1.0";
        }
    }
}

TEST_F(CrossAttentionTest, GQASupport) {
    // Test Grouped Query Attention (GQA)
    // 4 Query Heads, 2 KV Heads -> Group Size = 2
    // Heads 0,1 attend to KV Head 0
    // Heads 2,3 attend to KV Head 1
    const int B = 1, Lq = 1, Lk = 1, H = 4, D = 4;
    const int H_kv = 2;  // Explicitly smaller

    CrossAttentionTestTensor Q({B, Lq, H, D});
    CrossAttentionTestTensor K({B, Lk, H_kv, D});
    CrossAttentionTestTensor V({B, Lk, H_kv, D});
    CrossAttentionTestTensor output({B, Lq, H, D});

    Q.fill(1.0f);
    K.fill(1.0f);

    // Set V distinct for each KV head to verify routing
    V.fill(0.0f);
    // KV Head 0: Set to 1.0 -> Output Heads 0,1 should get ~1.0
    for (int i = 0; i < D; ++i) V.data()[0 * D + i] = 1.0f;
    // KV Head 1: Set to 2.0 -> Output Heads 2,3 should get ~2.0
    for (int i = 0; i < D; ++i) V.data()[1 * D + i] = 2.0f;

    auto* op = OpRegistry::Instance().GetBest(OpType::CrossAttention, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    CrossAttentionParams params;
    params.scale = 1.0f;

    std::vector<Tensor*> inputs = {Q.get(), K.get(), V.get()};
    std::vector<Tensor*> outputs = {output.get()};
    op->Execute(inputs, outputs, &params);

    // Verify Head 0 (Mapped to KV 0 -> 1.0)
    EXPECT_NEAR(output.data()[0], 1.0f, 0.01f) << "Head 0 should map to KV Head 0";
    // Verify Head 1 (Mapped to KV 0 -> 1.0)
    EXPECT_NEAR(output.data()[1 * D], 1.0f, 0.01f) << "Head 1 should map to KV Head 0";

    // Verify Head 2 (Mapped to KV 1 -> 2.0)
    EXPECT_NEAR(output.data()[2 * D], 2.0f, 0.01f) << "Head 2 should map to KV Head 1";
    // Verify Head 3 (Mapped to KV 1 -> 2.0)
    EXPECT_NEAR(output.data()[3 * D], 2.0f, 0.01f) << "Head 3 should map to KV Head 1";
}

}  // namespace

}  // namespace densecore
