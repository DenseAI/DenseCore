/**
 * @file onednn_matmul_test.cpp
 * @brief Correctness tests for oneDNN matmul backend (BF16/INT8)
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "ggml.h"

#include "../core/include/matmul_backend.h"
#include "../core/include/simd_ops.h"

namespace {

float RandUniform(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

int8_t RandInt8(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-20, 20);
    return static_cast<int8_t>(dist(rng));
}

bool NearlyEqual(float a, float b, float atol, float rtol) {
    const float diff = std::fabs(a - b);
    const float tol = atol + rtol * std::fabs(b);
    return diff <= tol;
}

void RefMatMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const float* a_row = A.data() + m * K;
            const float* b_row = B.data() + n * K;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[m * N + n] = sum;
        }
    }
}

int TestBf16() {
    const int M = 32;
    const int K = 128;
    const int N = 64;

    std::mt19937 rng(42);
    std::vector<float> A_f32(M * K);
    std::vector<float> B_f32(N * K);

    for (auto& v : A_f32) v = RandUniform(rng, -1.0f, 1.0f);
    for (auto& v : B_f32) v = RandUniform(rng, -1.0f, 1.0f);

    std::vector<ggml_bf16_t> A_bf16(M * K);
    std::vector<ggml_bf16_t> B_bf16(N * K);
    ggml_fp32_to_bf16_row(A_f32.data(), A_bf16.data(), A_f32.size());
    ggml_fp32_to_bf16_row(B_f32.data(), B_bf16.data(), B_f32.size());

    std::vector<float> A_ref(M * K);
    std::vector<float> B_ref(N * K);
    ggml_bf16_to_fp32_row(A_bf16.data(), A_ref.data(), A_ref.size());
    ggml_bf16_to_fp32_row(B_bf16.data(), B_ref.data(), B_ref.size());

    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_out(M * N, 0.0f);

    RefMatMul(A_ref, B_ref, C_ref, M, K, N);

    densecore::MatmulParams params;
    params.a = A_bf16.data();
    params.b = B_bf16.data();
    params.c = C_out.data();
    params.M = M;
    params.K = K;
    params.N = N;
    params.lda = K;
    params.ldb = K;
    params.ldc = N;
    params.trans_b = true;
    params.a_type = densecore::DType::BF16;
    params.b_type = densecore::DType::BF16;
    params.c_type = densecore::DType::F32;

    auto& backend = densecore::GetOneDnnMatmulBackend();
    if (!backend.IsAvailable() || !backend.Supports(params)) {
        std::cout << "BF16 test skipped (oneDNN backend not available)\n";
        return 0;
    }

    backend.PrepareWeights(params, "test_bf16");
    backend.Execute(params);

    const float atol = 5e-2f;
    const float rtol = 5e-2f;
    for (int i = 0; i < M * N; ++i) {
        if (!NearlyEqual(C_out[i], C_ref[i], atol, rtol)) {
            std::cerr << "BF16 mismatch at " << i << ": got " << C_out[i] << " expected " << C_ref[i] << "\n";
            return 1;
        }
    }

    std::cout << "BF16 test passed\n";
    return 0;
}

int TestInt8() {
    const int M = 16;
    const int K = 64;
    const int N = 32;

    std::mt19937 rng(7);
    std::vector<int8_t> A_s8(M * K);
    std::vector<int8_t> B_s8(N * K);

    for (auto& v : A_s8) v = RandInt8(rng);
    for (auto& v : B_s8) v = RandInt8(rng);

    const float a_scale = 0.02f;
    const float b_scale = 0.03f;
    const float c_scale = 1.0f;

    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_out(M * N, 0.0f);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const int8_t* a_row = A_s8.data() + m * K;
            const int8_t* b_row = B_s8.data() + n * K;
            for (int k = 0; k < K; ++k) {
                const float a = static_cast<float>(a_row[k]) * a_scale;
                const float b = static_cast<float>(b_row[k]) * b_scale;
                sum += a * b;
            }
            C_ref[m * N + n] = sum * c_scale;
        }
    }

    densecore::MatmulParams params;
    params.a = A_s8.data();
    params.b = B_s8.data();
    params.c = C_out.data();
    params.M = M;
    params.K = K;
    params.N = N;
    params.lda = K;
    params.ldb = K;
    params.ldc = N;
    params.trans_b = true;
    params.a_type = densecore::DType::INT8;
    params.b_type = densecore::DType::INT8;
    params.c_type = densecore::DType::F32;
    params.quant.a_scales = &a_scale;
    params.quant.b_scales = &b_scale;
    params.quant.c_scales = &c_scale;

    auto& backend = densecore::GetOneDnnMatmulBackend();
    if (!backend.IsAvailable() || !backend.Supports(params)) {
        std::cout << "INT8 test skipped (oneDNN backend not available)\n";
        return 0;
    }

    backend.PrepareWeights(params, "test_int8");
    backend.Execute(params);

    const float atol = 1e-2f;
    const float rtol = 1e-2f;
    for (int i = 0; i < M * N; ++i) {
        if (!NearlyEqual(C_out[i], C_ref[i], atol, rtol)) {
            std::cerr << "INT8 mismatch at " << i << ": got " << C_out[i] << " expected " << C_ref[i] << "\n";
            return 1;
        }
    }

    std::cout << "INT8 test passed\n";
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= TestBf16();
    rc |= TestInt8();
    return rc;
}
