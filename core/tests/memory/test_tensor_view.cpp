/**
 * @file test_tensor_view.cpp
 * @brief Unit tests for TensorView struct
 */

#include <gtest/gtest.h>

#include "tensor_view.h"

using namespace densecore;

TEST(TensorView, Basic2D) {
    float data[16];  // 4x4
    auto view = TensorView::Make2D(data, 4, 4, 4 * sizeof(float), DType::F32);

    EXPECT_EQ(view.ndim, 2);
    EXPECT_EQ(view.shape[0], 4);
    EXPECT_EQ(view.shape[1], 4);
    EXPECT_EQ(view.strides[0], 16);  // 4 floats * 4 bytes
    EXPECT_EQ(view.strides[1], 4);   // 1 float * 4 bytes
    EXPECT_EQ(view.RowByteWidth(), 16);
}

TEST(TensorView, Strided2D) {
    // Simulating a sub-matrix view or padded matrix
    // 4x4 logical view inside a 4x8 physical buffer
    float data[32];
    size_t row_stride_bytes = 8 * sizeof(float);  // 32 bytes

    auto view = TensorView::Make2D(data, 4, 4, row_stride_bytes, DType::F32);

    EXPECT_EQ(view.RowByteWidth(), 32);
    EXPECT_EQ(view.strides[0], 32);
    EXPECT_EQ(view.strides[1], 4);
}

TEST(TensorView, Int4Handling) {
    // 32 columns = 16 bytes
    uint8_t data[16];
    auto view = TensorView::FromContiguous(data, 1, 32, DType::INT4);

    EXPECT_EQ(view.RowByteWidth(), 16);
    // INT4 stride[1] (element stride) is tricky. DTypeSizeBytes returns 0.
    EXPECT_EQ(view.strides[1], 0);
}
