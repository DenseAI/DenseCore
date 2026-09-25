#include <memory>

#include <gtest/gtest.h>

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/operation_graph.h"
#include "runtime/engine_internal.h"

TEST(CpuBackendGraphCapture, EngineTeardownPreservesGlobalRegistryAndSurvivingEngineCapture) {
    auto& global_cpu = densecore::GetCpuBackend();
    auto* original_registry = global_cpu.GetOpRegistryForTest();
    std::unique_ptr<void, decltype(&FreeEngine)> first(InitEngine("mock", nullptr, 2), FreeEngine);
    std::unique_ptr<void, decltype(&FreeEngine)> second(InitEngine("mock", nullptr, 2), FreeEngine);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    auto* surviving_state = static_cast<EngineState*>(second.get());
    auto* surviving_cpu = static_cast<densecore::CpuBackend*>(
        surviving_state->backend_registry->Get(densecore::DeviceType::CPU));
    ASSERT_NE(surviving_cpu, nullptr);
    first.reset();
    ASSERT_EQ(global_cpu.GetOpRegistryForTest(), original_registry);

    const auto check_capture = [](densecore::CpuBackend& backend) {
        float a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8}, c[4] = {};
        auto input = densecore::Tensor::Make2D(a, 2, 2);
        auto weight = densecore::Tensor::Make2D(b, 2, 2);
        auto output = densecore::Tensor::Make2D(c, 2, 2);
        backend.BeginCapture();
        backend.MatMul(input, weight, &output);
        auto graph = backend.EndCapture();
        ASSERT_NE(graph, nullptr);
        backend.ExecuteGraph(*graph);
        EXPECT_FLOAT_EQ(c[0], 19);
        EXPECT_FLOAT_EQ(c[1], 22);
        EXPECT_FLOAT_EQ(c[2], 43);
        EXPECT_FLOAT_EQ(c[3], 50);
    };
    check_capture(*surviving_cpu);
    second.reset();
    ASSERT_EQ(global_cpu.GetOpRegistryForTest(), original_registry);
    check_capture(global_cpu);
}

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
