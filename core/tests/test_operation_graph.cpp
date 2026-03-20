/**
 * @file test_operation_graph.cpp
 * @brief Unit tests for operation graph optimization passes
 */

#include <gtest/gtest.h>

#include "densecore/hal/operation_graph.h"

using densecore::OperationGraph;
using densecore::GraphNode;
using densecore::OpType;
using densecore::RMSNormParams;
using densecore::Tensor;

TEST(OperationGraph, FuseAddRMSNorm_RewritesGraph) {
    float input_data[8] = {};
    float residual_data[8] = {};
    float add_out_data[8] = {};
    float rms_out_data[8] = {};
    float weight_data[4] = {};

    OperationGraph graph;
    size_t input_idx = graph.RegisterTensor(Tensor::Make2D(input_data, 2, 4));
    size_t residual_idx = graph.RegisterTensor(Tensor::Make2D(residual_data, 2, 4));
    size_t add_out_idx = graph.RegisterTensor(Tensor::Make2D(add_out_data, 2, 4));
    size_t weight_idx = graph.RegisterTensor(Tensor::Make1D(weight_data, 4));
    size_t rms_out_idx = graph.RegisterTensor(Tensor::Make2D(rms_out_data, 2, 4));

    GraphNode add_node;
    add_node.op = OpType::Add;
    add_node.inputs = {input_idx, residual_idx};
    add_node.outputs = {add_out_idx};
    graph.AddNode(std::move(add_node));

    GraphNode rms_node;
    rms_node.op = OpType::RMSNorm;
    rms_node.inputs = {add_out_idx, weight_idx};
    rms_node.outputs = {rms_out_idx};
    rms_node.params = RMSNormParams{1e-5f, false};
    graph.AddNode(std::move(rms_node));

    bool modified = densecore::FuseAddRMSNorm(graph);
    EXPECT_TRUE(modified);
    ASSERT_EQ(graph.NodeCount(), 1u);

    const GraphNode& fused = graph.GetNode(0);
    EXPECT_EQ(fused.op, OpType::AddRMSNorm);
    ASSERT_EQ(fused.inputs.size(), 3u);
    EXPECT_EQ(fused.inputs[0], input_idx);
    EXPECT_EQ(fused.inputs[1], residual_idx);
    EXPECT_EQ(fused.inputs[2], weight_idx);
    ASSERT_EQ(fused.outputs.size(), 1u);
    EXPECT_EQ(fused.outputs[0], rms_out_idx);

    const auto* params = std::get_if<RMSNormParams>(&fused.params);
    ASSERT_NE(params, nullptr);
    EXPECT_TRUE(params->fused_add);
}

TEST(OperationGraph, FuseAddRMSNorm_SkipsSharedAddOutput) {
    float input_data[8] = {};
    float residual_data[8] = {};
    float add_out_data[8] = {};
    float copy_out_data[8] = {};
    float rms_out_data[8] = {};
    float weight_data[4] = {};

    OperationGraph graph;
    size_t input_idx = graph.RegisterTensor(Tensor::Make2D(input_data, 2, 4));
    size_t residual_idx = graph.RegisterTensor(Tensor::Make2D(residual_data, 2, 4));
    size_t add_out_idx = graph.RegisterTensor(Tensor::Make2D(add_out_data, 2, 4));
    size_t copy_out_idx = graph.RegisterTensor(Tensor::Make2D(copy_out_data, 2, 4));
    size_t weight_idx = graph.RegisterTensor(Tensor::Make1D(weight_data, 4));
    size_t rms_out_idx = graph.RegisterTensor(Tensor::Make2D(rms_out_data, 2, 4));

    GraphNode add_node;
    add_node.op = OpType::Add;
    add_node.inputs = {input_idx, residual_idx};
    add_node.outputs = {add_out_idx};
    graph.AddNode(std::move(add_node));

    GraphNode copy_node;
    copy_node.op = OpType::Copy;
    copy_node.inputs = {add_out_idx};
    copy_node.outputs = {copy_out_idx};
    graph.AddNode(std::move(copy_node));

    GraphNode rms_node;
    rms_node.op = OpType::RMSNorm;
    rms_node.inputs = {add_out_idx, weight_idx};
    rms_node.outputs = {rms_out_idx};
    rms_node.params = RMSNormParams{1e-5f, false};
    graph.AddNode(std::move(rms_node));

    bool modified = densecore::FuseAddRMSNorm(graph);
    EXPECT_FALSE(modified);
    EXPECT_EQ(graph.NodeCount(), 3u);
}
