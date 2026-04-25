/**
 * @file test_moe_ops.cpp
 * @brief Tests for MoE HAL operations
 */

#include <gtest/gtest.h>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include "densecore/moe/moe_routing.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace densecore {
namespace {

class MoEOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        OpRegistry::Init();
    }
};

TEST_F(MoEOpsTest, OpTypeEnumValues) {
    // Verify MoE OpType enum values are in correct range (70-79)
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEGating), 70);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEScatter), 71);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEGather), 72);
    EXPECT_EQ(static_cast<uint8_t>(OpType::MoEForward), 73);
}

TEST_F(MoEOpsTest, OpTypeNames) {
    EXPECT_STREQ(OpTypeName(OpType::MoEGating), "MoEGating");
    EXPECT_STREQ(OpTypeName(OpType::MoEScatter), "MoEScatter");
    EXPECT_STREQ(OpTypeName(OpType::MoEGather), "MoEGather");
    EXPECT_STREQ(OpTypeName(OpType::MoEForward), "MoEForward");
}

TEST_F(MoEOpsTest, AudioOpTypeEnumValues) {
    // Verify Audio OpType enum values are in correct range (80-89)
    EXPECT_EQ(static_cast<uint8_t>(OpType::MelSpectrogram), 80);
    EXPECT_EQ(static_cast<uint8_t>(OpType::AudioConv1D), 81);
}

TEST_F(MoEOpsTest, AudioOpTypeNames) {
    EXPECT_STREQ(OpTypeName(OpType::MelSpectrogram), "MelSpectrogram");
    EXPECT_STREQ(OpTypeName(OpType::AudioConv1D), "AudioConv1D");
}

TEST_F(MoEOpsTest, MoEGatingRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGating, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEGating kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEScatterRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEScatter, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEScatter kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEGatherRegistered) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGather, DeviceType::CPU);
    ASSERT_NE(op, nullptr) << "MoEGather kernel not registered";
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(MoEOpsTest, MoEGatingBasic) {
    auto* op = OpRegistry::Instance().GetBest(OpType::MoEGating, DeviceType::CPU);
    ASSERT_NE(op, nullptr);

    const int batch_size = 4;
    const int hidden_dim = 8;
    const int num_experts = 4;
    const int top_k = 2;

    // Input: hidden states [batch, hidden_dim]
    std::vector<float> hidden_data(batch_size * hidden_dim);
    for (int i = 0; i < batch_size * hidden_dim; ++i) {
        hidden_data[i] = static_cast<float>(i % 10) * 0.1f;
    }

    // Gate weights [hidden_dim, num_experts]
    std::vector<float> gate_data(hidden_dim * num_experts, 0.1f);
    // Make expert 0 and 1 have higher scores for first tokens
    for (int i = 0; i < hidden_dim; ++i) {
        gate_data[i * num_experts + 0] = 0.5f;
        gate_data[i * num_experts + 1] = 0.3f;
    }

    // Output buffers
    std::vector<int> indices_data(batch_size * top_k);
    std::vector<float> weights_data(batch_size * top_k);

    Tensor hidden = Tensor::Make2D(hidden_data.data(), batch_size, hidden_dim);
    Tensor gate = Tensor::Make2D(gate_data.data(), hidden_dim, num_experts);
    // Note: indices stored as int but tensor uses default dtype, cast in kernel
    Tensor indices;
    indices.data = indices_data.data();
    indices.dtype = DType::F32;  // Placeholder - actual data is int
    indices.ndim = 2;
    indices.shape[0] = batch_size;
    indices.shape[1] = top_k;
    Tensor weights = Tensor::Make2D(weights_data.data(), batch_size, top_k);

    std::vector<Tensor*> inputs = {&hidden, &gate};
    std::vector<Tensor*> outputs = {&indices, &weights};

    MoEParams params;
    params.num_experts = num_experts;
    params.top_k = top_k;
    params.normalize_weights = true;

    op->Execute(inputs, outputs, &params);

    // Verify output shape is correct
    EXPECT_EQ(indices.shape[0], batch_size);
    EXPECT_EQ(indices.shape[1], top_k);
    EXPECT_EQ(weights.shape[0], batch_size);
    EXPECT_EQ(weights.shape[1], top_k);

    // Verify weights sum to ~1.0 for each token (normalized)
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += weights_data[b * top_k + k];
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Weights not normalized for token " << b;
    }

    // Verify expert indices are valid
    for (int i = 0; i < batch_size * top_k; ++i) {
        EXPECT_GE(indices_data[i], 0);
        EXPECT_LT(indices_data[i], num_experts);
    }
}

TEST_F(MoEOpsTest, MoEScatterGatherRoundtrip) {
    auto* scatter_op = OpRegistry::Instance().GetBest(OpType::MoEScatter, DeviceType::CPU);
    auto* gather_op = OpRegistry::Instance().GetBest(OpType::MoEGather, DeviceType::CPU);
    ASSERT_NE(scatter_op, nullptr);
    ASSERT_NE(gather_op, nullptr);

    const int batch_size = 2;
    const int hidden_dim = 4;
    const int top_k = 2;

    // Input data
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,    // Token 0
                                     5.0f, 6.0f, 7.0f, 8.0f};   // Token 1

    // Expert indices (each token goes to 2 experts)
    std::vector<int> indices_data = {0, 1,   // Token 0 → Expert 0, 1
                                     1, 2};  // Token 1 → Expert 1, 2

    // Equal weights (0.5 each)
    std::vector<float> weights_data = {0.5f, 0.5f, 0.5f, 0.5f};

    // Packed output (total_assignments = batch * top_k = 4)
    std::vector<float> packed_data(batch_size * top_k * hidden_dim, 0.0f);

    // Final output
    std::vector<float> output_data(batch_size * hidden_dim, 0.0f);

    Tensor input = Tensor::Make2D(input_data.data(), batch_size, hidden_dim);
    Tensor indices;
    indices.data = indices_data.data();
    indices.dtype = DType::F32;  // Placeholder - actual data is int
    indices.ndim = 2;
    indices.shape[0] = batch_size;
    indices.shape[1] = top_k;
    Tensor weights = Tensor::Make2D(weights_data.data(), batch_size, top_k);
    Tensor packed = Tensor::Make2D(packed_data.data(), batch_size * top_k, hidden_dim);
    Tensor output = Tensor::Make2D(output_data.data(), batch_size, hidden_dim);

    // Scatter
    std::vector<Tensor*> scatter_inputs = {&input, &indices};
    std::vector<Tensor*> scatter_outputs = {&packed};
    scatter_op->Execute(scatter_inputs, scatter_outputs, nullptr);

    // Simulate expert processing (identity for test)
    // packed_data already contains scattered input

    // Gather
    std::vector<Tensor*> gather_inputs = {&packed, &indices, &weights};
    std::vector<Tensor*> gather_outputs = {&output};
    gather_op->Execute(gather_inputs, gather_outputs, nullptr);

    // With identity expert and equal weights, output should equal input
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < hidden_dim; ++d) {
            float expected = input_data[b * hidden_dim + d];
            float actual = output_data[b * hidden_dim + d];
            EXPECT_NEAR(actual, expected, 0.001f)
                << "Mismatch at token " << b << " dim " << d;
        }
    }
}

TEST_F(MoEOpsTest, MoETopKRouteGateLogitsTensorMatchesRawPointer) {
    const int batch_size = 3;
    const int num_experts = 4;
    const int top_k = 2;

    std::vector<float> logits = {
        0.1f, 2.0f, 1.0f, -1.0f,
        3.0f, 0.5f, 0.4f, 0.2f,
        -1.0f, 0.0f, 4.0f, 1.0f,
    };

    Tensor gate_logits = Tensor::Make2D(logits.data(), batch_size, num_experts);

    moe::MoERouteResult expected;
    expected.expert_ids.resize(batch_size * top_k);
    expected.weights.resize(batch_size * top_k);
    expected.token_indices.resize(batch_size * top_k);

    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    std::vector<uint8_t> workspace_storage(workspace_bytes + 64);
    void* workspace_ptr = workspace_storage.data();
    const uintptr_t workspace_addr = reinterpret_cast<uintptr_t>(workspace_ptr);
    const uintptr_t aligned_addr = (workspace_addr + 63u) & ~static_cast<uintptr_t>(63u);
    moe::MoERoutingWorkspace ws;
    ASSERT_TRUE(moe::InitMoERoutingWorkspace(&ws, reinterpret_cast<void*>(aligned_addr), workspace_bytes, batch_size,
                                             num_experts, top_k));
    ASSERT_TRUE(moe::MoETopKRoute(logits.data(), batch_size, num_experts, top_k, true, &expected, &ws));

    const moe::MoERouteResult alloc_result = moe::MoETopKRoute(gate_logits, top_k);
    EXPECT_EQ(alloc_result.batch_size, batch_size);
    EXPECT_EQ(alloc_result.top_k, top_k);
    EXPECT_EQ(alloc_result.expert_ids, expected.expert_ids);
    EXPECT_EQ(alloc_result.token_indices, expected.token_indices);
    ASSERT_EQ(alloc_result.weights.size(), expected.weights.size());
    for (size_t i = 0; i < alloc_result.weights.size(); ++i) {
        EXPECT_NEAR(alloc_result.weights[i], expected.weights[i], 1e-6f) << "weight mismatch at " << i;
    }

    moe::MoERouteResult noalloc_result;
    noalloc_result.expert_ids.resize(batch_size * top_k);
    noalloc_result.weights.resize(batch_size * top_k);
    noalloc_result.token_indices.resize(batch_size * top_k);
    ASSERT_TRUE(moe::MoETopKRoute(gate_logits, top_k, &noalloc_result, &ws));
    EXPECT_EQ(noalloc_result.batch_size, batch_size);
    EXPECT_EQ(noalloc_result.top_k, top_k);
    EXPECT_EQ(noalloc_result.expert_ids, expected.expert_ids);
    EXPECT_EQ(noalloc_result.token_indices, expected.token_indices);
    ASSERT_EQ(noalloc_result.weights.size(), expected.weights.size());
    for (size_t i = 0; i < noalloc_result.weights.size(); ++i) {
        EXPECT_NEAR(noalloc_result.weights[i], expected.weights[i], 1e-6f) << "noalloc weight mismatch at " << i;
    }
}

TEST_F(MoEOpsTest, MoEParamsStruct) {
    MoEParams params;
    params.num_experts = 8;
    params.top_k = 2;
    params.normalize_weights = true;
    params.hidden_dim = 4096;
    params.intermediate_dim = 14336;

    EXPECT_EQ(params.num_experts, 8);
    EXPECT_EQ(params.top_k, 2);
    EXPECT_TRUE(params.normalize_weights);
    EXPECT_EQ(params.hidden_dim, 4096);
    EXPECT_EQ(params.intermediate_dim, 14336);
}

TEST_F(MoEOpsTest, MelSpectrogramParamsStruct) {
    MelSpectrogramParams params;
    params.n_fft = 400;
    params.hop_length = 160;
    params.n_mels = 80;
    params.sample_rate = 16000;

    EXPECT_EQ(params.n_fft, 400);
    EXPECT_EQ(params.hop_length, 160);
    EXPECT_EQ(params.n_mels, 80);
    EXPECT_EQ(params.sample_rate, 16000);
}

}  // namespace
}  // namespace densecore
