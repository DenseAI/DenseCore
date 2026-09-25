#include "densecore/hal/fp8_types.h"
#include "densecore/models/model_types.h"
#include <cstring>
#include <memory>
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


namespace densecore::testing {
ggml_tensor* BuildFp8MatmulForExtractionTest(ggml_context*, ggml_tensor*, ggml_tensor*,
                                          const TransformerModel::FP8WeightBinding&);
}

TEST_F(FP8Test, GraphCallbackPreservesDecodeAndTiledPrefillForBothFormats) {
    // Read the GGML custom callback prefix without coupling the test to its
    // private packed-weight userdata representation.
    struct CustomPrefix {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    for (auto format : {TransformerModel::FP8Format::E5M2, TransformerModel::FP8Format::E4M3FN}) {
        for (int tokens : {1, 3}) {
            for (bool padded : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "format=" << static_cast<int>(format)
                                                 << " tokens=" << tokens << " padded=" << padded);
                const std::unique_ptr<ggml_context, decltype(&ggml_free)> arena(
                    ggml_init({1024 * 1024, nullptr, false}), ggml_free);
                ASSERT_NE(arena, nullptr);
                constexpr int k = 32, n = 13;
                auto* weights = ggml_new_tensor_2d(arena.get(), GGML_TYPE_I8, k, n);
                auto* input_storage = ggml_new_tensor_2d(arena.get(), GGML_TYPE_F32, k + (padded ? 4 : 0), tokens);
                auto* input = padded ? ggml_view_2d(arena.get(), input_storage, k, tokens,
                                                  input_storage->nb[1], 0) : input_storage;
                for (int row = 0; row < n; ++row) {
                    for (int col = 0; col < k; ++col) {
                        // Finite positive and negative encodings in both formats.
                        static_cast<uint8_t*>(weights->data)[row * k + col] =
                            static_cast<uint8_t>(0x28 + ((row + col) % 16) + ((col % 3 == 0) ? 128 : 0));
                    }
                }
                for (int row = 0; row < tokens; ++row) {
                    auto* values = reinterpret_cast<float*>(static_cast<char*>(input->data) + row * input->nb[1]);
                    for (int col = 0; col < k; ++col) values[col] = static_cast<float>((row + col) % 9 - 4) / 8.0f;
                }
                TransformerModel::FP8WeightBinding binding{weights, format, k, n};
                auto* output = densecore::testing::BuildFp8MatmulForExtractionTest(arena.get(), weights, input, binding);
                ASSERT_NE(output, nullptr);
                CustomPrefix callback{};
                std::memcpy(&callback, output->op_params, sizeof(callback));
                ASSERT_NE(callback.fun, nullptr);
                ASSERT_GE(callback.n_tasks, 1);
                for (int task = 0; task < callback.n_tasks; ++task) {
                    callback.fun(output, task, callback.n_tasks, callback.userdata);
                }
                const float* lut = format == TransformerModel::FP8Format::E5M2 ? e5m2_lut_ : e4m3fn_lut_;
                for (int row = 0; row < tokens; ++row) {
                    const auto* values = reinterpret_cast<const float*>(static_cast<const char*>(input->data) + row * input->nb[1]);
                    for (int out = 0; out < n; ++out) {
                        float expected = 0.0f;
                        for (int col = 0; col < k; ++col) {
                            expected += values[col] * lut[static_cast<const uint8_t*>(weights->data)[out * k + col]];
                        }
                        EXPECT_NEAR(static_cast<const float*>(output->data)[row * n + out], expected, 1e-5f);
                    }
                }
            }
        }
    }
}
