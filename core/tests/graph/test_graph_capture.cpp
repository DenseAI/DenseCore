#include <memory>

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/operation_graph.h"

TEST(CpuBackendGraphCapture, MatMulCaptureReplay) {
    densecore::CpuBackend& backend = densecore::GetCpuBackend();

    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c_data[4] = {-1.0f, -1.0f, -1.0f, -1.0f};

    densecore::Tensor A = densecore::Tensor::Make2D(a_data, 2, 2);
    densecore::Tensor B = densecore::Tensor::Make2D(b_data, 2, 2);
    densecore::Tensor C = densecore::Tensor::Make2D(c_data, 2, 2);

    backend.BeginCapture();
    backend.MatMul(A, B, &C);
    std::unique_ptr<densecore::OperationGraph> graph = backend.EndCapture();

    for (float v : c_data) {
        EXPECT_FLOAT_EQ(v, -1.0f);
    }

    backend.ExecuteGraph(*graph);

    EXPECT_NEAR(c_data[0], 19.0f, 1e-4f);
    EXPECT_NEAR(c_data[1], 22.0f, 1e-4f);
    EXPECT_NEAR(c_data[2], 43.0f, 1e-4f);
    EXPECT_NEAR(c_data[3], 50.0f, 1e-4f);
}
