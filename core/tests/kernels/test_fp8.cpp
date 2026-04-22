#include "densecore/hal/fp8_types.h"
#include "kernels/hwy/hwy_kernels.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace densecore;
using namespace densecore::hwy_kernels;

class FP8Test : public ::testing::Test {
protected:
    const float* e5m2_lut_ = nullptr;
    const float* e4m3fn_lut_ = nullptr;

    void SetUp() override {
        // Ensure LUTs are initialized
        InitFP8LUTs_Hwy();
        e5m2_lut_ = GetFP8E5M2LUT_Hwy();
        e4m3fn_lut_ = GetFP8E4M3FNLUT_Hwy();
        ASSERT_NE(e5m2_lut_, nullptr);
        ASSERT_NE(e4m3fn_lut_, nullptr);
    }
};

// Test E5M2 Helper Logic (Indirectly via LUT)
TEST_F(FP8Test, E5M2_ImportantValues) {
    // 0: S=0 E=0 M=0 => 0.0
    // 128: S=1 E=0 M=0 => -0.0
    EXPECT_FLOAT_EQ(e5m2_lut_[0], 0.0f);
    EXPECT_FLOAT_EQ(e5m2_lut_[128], -0.0f);

    // One: S=0 E=15(01111) M=0 => 1.0 (Bias 15, so 2^0 * 1.0)
    // 0 01111 00 = 0011 1100 = 0x3C = 60
    EXPECT_FLOAT_EQ(e5m2_lut_[60], 1.0f);

    // Max Normal: S=0 E=30(11110) M=3(11) => 57344
    // 0 11110 11 = 0111 1011 = 0x7B = 123
    EXPECT_FLOAT_EQ(e5m2_lut_[123], 57344.0f);

    // Min Normal: S=0 E=1 M=0 => 2^-14 = 6.1035e-5
    // 0 00001 00 = 0000 0100 = 4
    EXPECT_FLOAT_EQ(e5m2_lut_[4], std::pow(2.0f, -14));

    // Subnormal: S=0 E=0 M=1 => 2^-16 * 1 = 1.52587e-5
    EXPECT_FLOAT_EQ(e5m2_lut_[1], std::pow(2.0f, -16));

    // Inf: S=0 E=31 M=0 => 0x7C = 124
    EXPECT_TRUE(std::isinf(e5m2_lut_[124]));

    // NaN: S=0 E=31 M=1 => 0x7D = 125
    EXPECT_TRUE(std::isnan(e5m2_lut_[125]));
}

// Test E4M3FN Helper Logic
TEST_F(FP8Test, E4M3FN_ImportantValues) {
    // 0: 0.0
    EXPECT_FLOAT_EQ(e4m3fn_lut_[0], 0.0f);

    // NaN: 0x7F (127) and 0xFF (255)
    EXPECT_TRUE(std::isnan(e4m3fn_lut_[127]));
    EXPECT_TRUE(std::isnan(e4m3fn_lut_[255]));

    // Max: S=0 E=15 M=6 (110) => 448
    // 0 1111 110 = 0111 1110 = 0x7E = 126
    EXPECT_FLOAT_EQ(e4m3fn_lut_[126], 448.0f);

    // One: S=0 E=7 M=0 => 1.0
    // 0 0111 000 = 0x38 = 56
    EXPECT_FLOAT_EQ(e4m3fn_lut_[56], 1.0f);
}

// Test Kernels
TEST_F(FP8Test, ConvertE5M2_AVX2) {
    int n = 100;  // Large enough to test loop unrolling + tail
    std::vector<uint8_t> input(n);
    std::vector<float> output(n);

    // Pattern: 0, 1, ..., 99
    for (int i = 0; i < n; ++i) input[i] = static_cast<uint8_t>(i);

    ConvertFP8E5M2ToFP32_Hwy(input.data(), output.data(), n);

    for (int i = 0; i < n; ++i) {
        float expected = e5m2_lut_[input[i]];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(output[i])) << " expected NaN at index " << i;
        } else {
            EXPECT_EQ(output[i], expected) << " mismatch at index " << i;
        }
    }
}

TEST_F(FP8Test, ConvertE4M3FN_AVX2) {
    int n = 1024;
    std::vector<uint8_t> input(n);
    std::vector<float> output(n);

    for (int i = 0; i < n; ++i) input[i] = static_cast<uint8_t>(i % 256);

    ConvertFP8E4M3FNToFP32_Hwy(input.data(), output.data(), n);

    for (int i = 0; i < n; ++i) {
        float expected = e4m3fn_lut_[input[i]];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(output[i])) << " expected NaN at index " << i;
        } else {
            EXPECT_EQ(output[i], expected) << " mismatch at index " << i;
        }
    }
}

// Helper to compute reference dot product
float RefDot(int N, const uint8_t* row_fp8, const float* x, const float* lut) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += lut[row_fp8[i]] * x[i];
    }
    return sum;
}

TEST_F(FP8Test, Gemv_E5M2_Correctness) {
    int M = 4;
    int N = 65;  // Non-multiple of 32 to test tail

    std::vector<uint8_t> A(M * N);
    std::vector<float> x(N);
    std::vector<float> y(M, 0.0f);

    // Init data
    for (int i = 0; i < M * N; ++i) A[i] = static_cast<uint8_t>(i % 120);  // avoid nan/inf
    for (int i = 0; i < N; ++i) x[i] = 1.0f;

    float alpha = 0.5f;
    float beta = 0.0f;

    Gemv_FP8_E5M2_Hwy(M, N, alpha, A.data(), x.data(), beta, y.data());

    for (int m = 0; m < M; ++m) {
        float ref = RefDot(N, A.data() + m * N, x.data(), e5m2_lut_);
        EXPECT_NEAR(y[m], alpha * ref, 1e-3);
    }
}

TEST_F(FP8Test, Gemv_E5M2_WithBeta) {
    int M = 2;
    int N = 32;

    std::vector<uint8_t> A(M * N);
    std::vector<float> x(N);
    std::vector<float> y(M);
    y[0] = 10.0f;
    y[1] = 20.0f;  // Initial values

    // Init data (using small indices for precise float values)
    // Index 60 is 1.0 in E5M2
    for (int i = 0; i < M * N; ++i) A[i] = 60;
    for (int i = 0; i < N; ++i) x[i] = 2.0f;

    // Row sum should be 32 * (1.0 * 2.0) = 64.0

    float alpha = 1.0f;
    float beta = 0.5f;

    Gemv_FP8_E5M2_Hwy(M, N, alpha, A.data(), x.data(), beta, y.data());

    // Row 0: 1.0 * 64.0 + 0.5 * 10.0 = 64 + 5 = 69
    EXPECT_NEAR(y[0], 69.0f, 1e-3);

    // Row 1: 1.0 * 64.0 + 0.5 * 20.0 = 64 + 10 = 74
    EXPECT_NEAR(y[1], 74.0f, 1e-3);
}

TEST_F(FP8Test, Gemv_E4M3FN_Correctness) {
    int M = 8;
    int N = 256;

    std::vector<uint8_t> A(M * N);
    std::vector<float> x(N);
    std::vector<float> y(M, 0.0f);

    // Init data
    // Index 56 is 1.0 in E4M3FN
    for (int i = 0; i < M * N; ++i) A[i] = (i % 2 == 0) ? 56 : 0;
    for (int i = 0; i < N; ++i) x[i] = 1.0f;

    float alpha = 1.0f;
    float beta = 0.0f;

    Gemv_FP8_E4M3FN_Hwy(M, N, alpha, A.data(), x.data(), beta, y.data());

    for (int m = 0; m < M; ++m) {
        float ref = RefDot(N, A.data() + m * N, x.data(), e4m3fn_lut_);
        EXPECT_NEAR(y[m], ref, 1e-3);
    }
}
