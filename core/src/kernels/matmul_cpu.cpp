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

#include "../backend/thread_pool_impl.h"  // Corrected path (src/backend/thread_pool_impl.h)
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/matmul_backend.h"
#include "densecore/simd/simd_platform.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include "ggml.h"

namespace densecore {
namespace {
bool IsF32OrBF16(DType dtype) {
    return dtype == DType::F32 || dtype == DType::BF16;
}

void ExecuteMixedMatMulF32Accum(const Tensor& A, const Tensor& B, Tensor* C) {
    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[1]);

    const float* a_f32 = (A.dtype == DType::F32) ? A.DataAs<float>() : nullptr;
    const ggml_bf16_t* a_bf16 = (A.dtype == DType::BF16) ? A.DataAs<ggml_bf16_t>() : nullptr;
    const float* b_f32 = (B.dtype == DType::F32) ? B.DataAs<float>() : nullptr;
    const ggml_bf16_t* b_bf16 = (B.dtype == DType::BF16) ? B.DataAs<ggml_bf16_t>() : nullptr;
    float* c_data = C->DataAs<float>();

    auto& pool = GetCpuBackend().GetThreadPool(0);
    pool.ParallelFor(M, [&](int m_start, int m_end, int /*tid*/) {
        std::vector<float> a_row_storage(static_cast<size_t>(K));
        for (int m = m_start; m < m_end; ++m) {
            const float* a_row = nullptr;
            if (a_f32) {
                a_row = a_f32 + static_cast<size_t>(m) * K;
            } else {
                ggml_bf16_to_fp32_row(a_bf16 + static_cast<size_t>(m) * K, a_row_storage.data(), K);
                a_row = a_row_storage.data();
            }

            float* c_row = c_data + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float b = b_f32 ? b_f32[static_cast<size_t>(k) * N + n]
                                          : ggml_bf16_to_fp32(b_bf16[static_cast<size_t>(k) * N + n]);
                    sum += a_row[k] * b;
                }
                c_row[n] = sum;
            }
        }
    });
}

void ExecuteMixedMatMulTransBF32Accum(const Tensor& A, const Tensor& B, Tensor* C) {
    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[0]);

    const float* a_f32 = (A.dtype == DType::F32) ? A.DataAs<float>() : nullptr;
    const ggml_bf16_t* a_bf16 = (A.dtype == DType::BF16) ? A.DataAs<ggml_bf16_t>() : nullptr;
    const float* b_f32 = (B.dtype == DType::F32) ? B.DataAs<float>() : nullptr;
    const ggml_bf16_t* b_bf16 = (B.dtype == DType::BF16) ? B.DataAs<ggml_bf16_t>() : nullptr;
    float* c_data = C->DataAs<float>();

    auto& pool = GetCpuBackend().GetThreadPool(0);
    pool.ParallelFor(M, [&](int m_start, int m_end, int /*tid*/) {
        std::vector<float> a_row_storage(static_cast<size_t>(K));
        std::vector<float> b_row_storage(static_cast<size_t>(K));
        for (int m = m_start; m < m_end; ++m) {
            const float* a_row = nullptr;
            if (a_f32) {
                a_row = a_f32 + static_cast<size_t>(m) * K;
            } else {
                ggml_bf16_to_fp32_row(a_bf16 + static_cast<size_t>(m) * K, a_row_storage.data(), K);
                a_row = a_row_storage.data();
            }

            float* c_row = c_data + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                const float* b_row = nullptr;
                if (b_f32) {
                    b_row = b_f32 + static_cast<size_t>(n) * K;
                } else {
                    ggml_bf16_to_fp32_row(b_bf16 + static_cast<size_t>(n) * K, b_row_storage.data(), K);
                    b_row = b_row_storage.data();
                }
                c_row[n] = simd::DotF32(a_row, b_row, K);
            }
        }
    });
}

bool ExecuteTransBViaMatmulBackend(const Tensor& A, const Tensor& B, Tensor* C) {
    if (!IsF32OrBF16(A.dtype) || !IsF32OrBF16(B.dtype) || !C || C->dtype != DType::F32) {
        return false;
    }

    MatmulParams params;
    params.a = A.data;
    params.b = B.data;
    params.c = C->data;
    params.M = A.shape[0];
    params.N = B.shape[0];
    params.K = A.shape[1];
    params.lda = A.shape[1];
    params.ldb = B.shape[1];
    params.ldc = C->shape[1];
    params.trans_a = false;
    params.trans_b = true;
    params.a_type = A.dtype;
    params.b_type = B.dtype;
    params.c_type = C->dtype;

    thread_local std::vector<ggml_bf16_t> a_bf16_buffer;
    if (A.dtype == DType::F32 && B.dtype == DType::BF16 && params.M > 1) {
        MatmulParams bf16_candidate = params;
        bf16_candidate.a_type = DType::BF16;
        if (SelectMatmulBackend(bf16_candidate, true) == MatmulBackendKind::OneDNN) {
            const size_t total = static_cast<size_t>(params.M * params.K);
            a_bf16_buffer.resize(total);
            const float* a_f32 = A.DataAs<float>();
            for (int64_t m = 0; m < params.M; ++m) {
                ggml_fp32_to_bf16_row(a_f32 + static_cast<size_t>(m) * params.lda,
                                      a_bf16_buffer.data() + static_cast<size_t>(m) * params.K,
                                      static_cast<int>(params.K));
            }
            params.a = a_bf16_buffer.data();
            params.lda = params.K;
            params.a_type = DType::BF16;
        }
    }

    const bool is_prefill = params.M > 1;
    MatmulBackendKind kind = SelectMatmulBackend(params, is_prefill);
    MatmulBackend* backend =
        (kind == MatmulBackendKind::OneDNN) ? &GetOneDnnMatmulBackend() : &GetDenseCoreMatmulBackend();
    if (!backend->Supports(params)) {
        backend = &GetDenseCoreMatmulBackend();
        if (!backend->Supports(params)) {
            ExecuteMixedMatMulTransBF32Accum(A, B, C);
            return true;
        }
    }

    backend->Execute(params);
    return true;
}

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

    bool SupportsDType(DType dtype) const override { return dtype == DType::F32 || dtype == DType::BF16; }

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
        if (!IsF32OrBF16(A.dtype) || !IsF32OrBF16(B.dtype) || C->dtype != DType::F32) return;

        // A: [M, K], B: [K, N], C: [M, N]
        const int M = static_cast<int>(A.shape[0]);
        const int K = static_cast<int>(A.shape[1]);
        const int N = static_cast<int>(B.shape[1]);

        if (A.dtype == DType::BF16 || B.dtype == DType::BF16) {
            ExecuteMixedMatMulF32Accum(A, B, C);
            return;
        }

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
        if (!IsF32OrBF16(A.dtype) || !IsF32OrBF16(B.dtype) || C->dtype != DType::F32) return;

        // A: [M, K], B: [N, K] (stored row-major), C: [M, N]
        const int M = static_cast<int>(A.shape[0]);
        const int K = static_cast<int>(A.shape[1]);
        const int N = static_cast<int>(B.shape[0]);  // N is dim 0 of B

#ifdef __APPLE__
        if (A.dtype != DType::F32 || B.dtype != DType::F32) {
            ExecuteTransBViaMatmulBackend(A, B, C);
            return;
        }

        const float* a_data = A.DataAs<float>();
        const float* b_data = B.DataAs<float>();
        float* c_data = C->DataAs<float>();
        // B is [N, K] row-major. B^T is [K, N] row-major.
        // cblas expects B to be [K, N].
        // Since B is [N, K], we use CblasTrans.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a_data, K, b_data, K, 0.0f, c_data, N);
        return;
#endif

        ExecuteTransBViaMatmulBackend(A, B, C);
    }
};

DENSECORE_REGISTER_OP(CpuMatMulOp, OpType::MatMul, DeviceType::CPU);
// Also register for MatMulTransB (same class handles both)
DENSECORE_REGISTER_OP(CpuMatMulOp, OpType::MatMulTransB, DeviceType::CPU);

}  // namespace
}  // namespace densecore
