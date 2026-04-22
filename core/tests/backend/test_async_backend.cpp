/**
 * @file test_async_backend.cpp
 * @brief Unit tests for AsyncBackend
 */

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "async_cpu_backend.h"

using namespace densecore;

// Simple MatMul test
TEST(AsyncBackend, MatMulBasic) {
    // 2x2 * 2x2 = 2x2
    // [1 2] * [1 0] = [1 2]
    // [3 4]   [0 1]   [3 4]

    // Data setup
    std::vector<float> data_A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_B = {1.0f, 0.0f, 0.0f, 1.0f};  // Identity
    std::vector<float> data_C(4, 0.0f);

    // View creation
    auto A = TensorView::Make2D(data_A.data(), 2, 2, 2 * sizeof(float), DType::F32);
    auto B = TensorView::Make2D(data_B.data(), 2, 2, 2 * sizeof(float), DType::F32);
    auto C = TensorView::Make2D(data_C.data(), 2, 2, 2 * sizeof(float), DType::F32);

    // Context and Backend
    KernelContext ctx;
    AsyncCpuBackend backend;

    // Async execution
    auto future = backend.MatMulAsync(ctx, A, B, C);

    // Wait for result
    future.wait();

    // Verify results
    EXPECT_FLOAT_EQ(data_C[0], 1.0f);
    EXPECT_FLOAT_EQ(data_C[1], 2.0f);
    EXPECT_FLOAT_EQ(data_C[2], 3.0f);
    EXPECT_FLOAT_EQ(data_C[3], 4.0f);
}

// Correctness test with larger non-identity matrix
TEST(AsyncBackend, MatMulCompute) {
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]

    std::vector<float> data_A = {1, 2, 3, 4};
    std::vector<float> data_B = {5, 6, 7, 8};
    std::vector<float> data_C(4, 0.0f);

    auto A = TensorView::Make2D(data_A.data(), 2, 2, 8, DType::F32);
    auto B = TensorView::Make2D(data_B.data(), 2, 2, 8, DType::F32);
    auto C = TensorView::Make2D(data_C.data(), 2, 2, 8, DType::F32);

    KernelContext ctx;
    AsyncCpuBackend backend;

    backend.MatMul(ctx, A, B, C);  // Use sync wrapper

    EXPECT_FLOAT_EQ(data_C[0], 19.0f);
    EXPECT_FLOAT_EQ(data_C[1], 22.0f);
    EXPECT_FLOAT_EQ(data_C[2], 43.0f);
    EXPECT_FLOAT_EQ(data_C[3], 50.0f);
}
