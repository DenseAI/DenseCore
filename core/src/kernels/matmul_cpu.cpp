/**
 * @file matmul_cpu.cpp
 * @brief CPU implementation of Matrix Multiplication operations
 *
 * Implements MatMul and MatMulTransB using available SIMD backends (AVX, NEON, Accelerate).
 * Handles thread parallelism for large matrices and NUMA awareness.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/typed_tensor.h"

#include "../include/cpu_backend.h"
#include "../include/gemm_config.h"
#include "../include/matmul_backend.h"
#include "../include/simd_ops.h"  // For simd::DotF32, simd::MatMulTransB
#include "../include/simd_platform.h"
#include "../thread_pool_impl.h"  // Corrected path (src/thread_pool_impl.h)

#include <cstring>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace densecore {
namespace {
class CpuMatMulOp : public MatMulOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        // MatMul: C = A * B
        // MatMulTransB: C = A * B^T
        // Handled by specific methods below
        (void)inputs;
        (void)outputs;
        (void)params;
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    // =========================================================================
    // MatMul: C = A * B
    // =========================================================================
    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override {
        if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) return;

        // A: [M, K], B: [K, N], C: [M, N]
        const int M = static_cast<int>(A.shape[0]);
        const int K = static_cast<int>(A.shape[1]);
        const int N = static_cast<int>(B.shape[1]);

        const float* a_data = A.DataAs<float>();
        const float* b_data = B.DataAs<float>();
        float* c_data = C->DataAs<float>();
        auto& pool = GetCpuBackend().GetThreadPool(0);

        // DECODE Phase (M=1): Vector-Matrix Multiplication (GEMV)
        if (M == 1) {
            // Use backend thread pool for parallel GEMV
            pool.ParallelGemv(c_data, a_data, b_data, K, N);
            return;
        }

        // PREFILL Phase (M>1): Matrix-Matrix Multiplication (GEMM)

#if defined(DENSECORE_USE_ONEDNN)
        // oneDNN Path
        const auto& gemm_cfg = GetGemmConfig();
        if (gemm_cfg.ShouldUseOneDNN(M, K, N)) {
            MatmulParams params;
            params.a = a_data;
            params.b = b_data;
            params.c = c_data;
            params.M = M;
            params.N = N;
            params.K = K;
            params.lda = K;
            params.ldb = N;
            params.ldc = N;
            params.trans_a = false;
            params.trans_b = false;
            params.a_type = DType::F32;
            params.b_type = DType::F32;
            params.c_type = DType::F32;

            auto& backend = GetOneDnnMatmulBackend();
            if (backend.IsAvailable()) {
                backend.Execute(params);
                return;
            }
        }
#endif

#ifdef __APPLE__
        // Apple Accelerate Path
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a_data, K, b_data, N, 0.0f, c_data, N);
        return;
#endif

        // NOTE: Split-N Highway kernel expects weights in [N, K] row-major
        // (MatMulTransB layout). MatMul uses B as [K, N], so keep tiled path.

        // Standard Tiled Implementation
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int BLOCK_K = 64;

        std::memset(c_data, 0, M * N * sizeof(float));
        const int num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;

        pool.ParallelFor(num_m_blocks, [&](int block_start, int block_end, int /*tid*/) {
            for (int block_idx = block_start; block_idx < block_end; ++block_idx) {
                const int m0 = block_idx * BLOCK_M;
                const int m_end = std::min(m0 + BLOCK_M, M);

                for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
                    const int n_end = std::min(n0 + BLOCK_N, N);
                    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                        const int k_end = std::min(k0 + BLOCK_K, K);

                        for (int m = m0; m < m_end; ++m) {
                            const float* a_row = a_data + m * K;
                            float* c_row = c_data + m * N;
                            for (int k = k0; k < k_end; ++k) {
                                const float a_mk = a_row[k];
                                const float* b_row = b_data + k * N;
#pragma GCC ivdep
                                for (int n = n0; n < n_end; ++n) {
                                    c_row[n] += a_mk * b_row[n];
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // =========================================================================
    // MatMulTransB: C = A * B^T
    // =========================================================================
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override {
        if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) return;

        // A: [M, K], B: [N, K] (stored row-major), C: [M, N]
        const int M = static_cast<int>(A.shape[0]);
        const int K = static_cast<int>(A.shape[1]);
        const int N = static_cast<int>(B.shape[0]);  // N is dim 0 of B

        const float* a_data = A.DataAs<float>();
        const float* b_data = B.DataAs<float>();
        float* c_data = C->DataAs<float>();

#ifdef __APPLE__
        // B is [N, K] row-major. B^T is [K, N] row-major.
        // cblas expects B to be [K, N].
        // Since B is [N, K], we use CblasTrans.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K, b_data, K, 0.0f, c_data, N);
        return;
#endif

        auto& pool = GetCpuBackend().GetThreadPool(0);
        // GEMM (Prefill): Parallelize across M (rows).
        // NOTE: forcing split-N Highway for low-M was slower on batch=4/threads=8
        // in end-to-end decode benchmarks, so keep this path as default.
        pool.ParallelFor(M, [=](int m_start, int m_end, int) {
            for (int m = m_start; m < m_end; ++m) {
                const float* a_row = a_data + m * K;
                float* c_row = c_data + m * N;
                for (int n = 0; n < N; ++n) {
                    const float* b_row = b_data + n * K;
                    c_row[n] = simd::DotF32(a_row, b_row, K);
                }
            }
        });
    }
};

DENSECORE_REGISTER_OP(CpuMatMulOp, OpType::MatMul, DeviceType::CPU);
// Also register for MatMulTransB (same class handles both)
DENSECORE_REGISTER_OP(CpuMatMulOp, OpType::MatMulTransB, DeviceType::CPU);

}  // namespace
}  // namespace densecore
