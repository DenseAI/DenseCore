#include <memory>

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/operation_graph.h"

TEST(GraphDomainOps, PatchEmbed2D_ExecuteGraph) {
    densecore::CpuBackend& backend = densecore::GetCpuBackend();

    float image_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // [1,1,2,2]
    float weight_data[1] = {1.0f};                   // [1,1,1,1]
    float bias_data[1] = {0.0f};                     // [1]
    float output_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // [1,4,1]

    densecore::Tensor image;
    image.data = image_data;
    image.ndim = 4;
    image.shape = {1, 1, 2, 2};
    image.stride = {1 * 2 * 2, 2 * 2, 2, 1};

    densecore::Tensor weight;
    weight.data = weight_data;
    weight.ndim = 4;
    weight.shape = {1, 1, 1, 1};
    weight.stride = {1, 1, 1, 1};

    densecore::Tensor bias = densecore::Tensor::Make1D(bias_data, 1);

    densecore::Tensor output;
    output.data = output_data;
    output.ndim = 3;
    output.shape = {1, 4, 1};
    output.stride = {4, 1, 1, 0};

    densecore::OperationGraph graph;
    size_t image_idx = graph.RegisterTensor(image);
    size_t weight_idx = graph.RegisterTensor(weight);
    size_t bias_idx = graph.RegisterTensor(bias);
    size_t out_idx = graph.RegisterTensor(output);

    densecore::GraphNode node;
    node.op = densecore::OpType::PatchEmbed2D;
    node.inputs = {image_idx, weight_idx, bias_idx};
    node.outputs = {out_idx};
    graph.AddNode(std::move(node));

    backend.ExecuteGraph(graph);

    EXPECT_NEAR(output_data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(output_data[1], 2.0f, 1e-4f);
    EXPECT_NEAR(output_data[2], 3.0f, 1e-4f);
    EXPECT_NEAR(output_data[3], 4.0f, 1e-4f);
}
