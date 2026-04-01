#include <gtest/gtest.h>

#include <vector>

#include "cpu_backend.h"
#include "matmul_backend.h"

namespace densecore {
namespace {

std::vector<float> ReferenceMatMulTransB(const float* a, const float* b, int m, int n, int k, int lda, int ldb) {
    std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += a[static_cast<size_t>(row) * lda + kk] * b[static_cast<size_t>(col) * ldb + kk];
            }
            out[static_cast<size_t>(row) * n + col] = sum;
        }
    }
    return out;
}

TEST(MatmulBackend, DenseCoreF32TransBHandlesStrides) {
    constexpr int M = 2;
    constexpr int N = 3;
    constexpr int K = 4;
    constexpr int LDA = 6;
    constexpr int LDB = 5;
    constexpr int LDC = 4;

    std::vector<float> a(static_cast<size_t>(M * LDA), -99.0f);
    std::vector<float> b(static_cast<size_t>(N * LDB), -77.0f);
    std::vector<float> c(static_cast<size_t>(M * LDC), -1.0f);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    a[LDA + 0] = 5.0f;
    a[LDA + 1] = 6.0f;
    a[LDA + 2] = 7.0f;
    a[LDA + 3] = 8.0f;

    b[0] = 1.0f;
    b[1] = 0.0f;
    b[2] = 1.0f;
    b[3] = 0.0f;
    b[LDB + 0] = 0.0f;
    b[LDB + 1] = 1.0f;
    b[LDB + 2] = 0.0f;
    b[LDB + 3] = 1.0f;
    b[2 * LDB + 0] = 1.0f;
    b[2 * LDB + 1] = 1.0f;
    b[2 * LDB + 2] = 1.0f;
    b[2 * LDB + 3] = 1.0f;

    MatmulParams params;
    params.a = a.data();
    params.b = b.data();
    params.c = c.data();
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = LDA;
    params.ldb = LDB;
    params.ldc = LDC;
    params.trans_b = true;
    params.a_type = DType::F32;
    params.b_type = DType::F32;
    params.c_type = DType::F32;

    GetDenseCoreMatmulBackend().Execute(params);

    const std::vector<float> expected = ReferenceMatMulTransB(a.data(), b.data(), M, N, K, LDA, LDB);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            EXPECT_NEAR(c[static_cast<size_t>(row) * LDC + col], expected[static_cast<size_t>(row) * N + col], 1e-5f);
        }
        EXPECT_FLOAT_EQ(c[static_cast<size_t>(row) * LDC + N], -1.0f);
    }
}

TEST(MatmulBackend, CpuBackendMatMulTransBMatchesReference) {
    constexpr int M = 2;
    constexpr int K = 4;
    constexpr int N = 3;

    std::vector<float> a = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    std::vector<float> b = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    std::vector<float> c(static_cast<size_t>(M * N), 0.0f);

    Tensor A = Tensor::Make2D(a.data(), M, K);
    Tensor B = Tensor::Make2D(b.data(), N, K);
    Tensor C = Tensor::Make2D(c.data(), M, N);

    GetCpuBackend().MatMulTransB(A, B, &C);

    const std::vector<float> expected = ReferenceMatMulTransB(a.data(), b.data(), M, N, K, K, K);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(c[i], expected[i], 1e-5f);
    }
}

}  // namespace
}  // namespace densecore
