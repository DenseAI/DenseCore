/**
 * @file matmul_backend_densecore.cpp
 * @brief DenseCore matmul backend (baseline kernels)
 */

#include "densecore/backend/matmul_backend.h"

#include <algorithm>
#include <cstdint>

#include "densecore/backend/cpu_backend.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "thread_pool_impl.h"

namespace densecore {

namespace {

inline bool SupportsF32TransB(const MatmulParams& params) {
    return params.a_type == DType::F32 && params.b_type == DType::F32 && params.c_type == DType::F32;
}

inline bool SupportsBF16TransB(const MatmulParams& params) {
    return (params.a_type == DType::F32 || params.a_type == DType::BF16) && params.b_type == DType::BF16 &&
           params.c_type == DType::F32;
}

inline bool SupportsINT8TransB(const MatmulParams& params) {
    return params.a_type == DType::INT8 && params.b_type == DType::INT8 && params.c_type == DType::F32;
}

inline bool IsContiguousF32TransB(const MatmulParams& params) {
    return params.lda == params.K && params.ldb == params.K && params.ldc == params.N;
}

inline bool ShouldParallelizeF32TransB(int M, int N, int K, int64_t work) {
    const MatmulHeuristics heuristics = GetMatmulConfig().heuristics;
    return M >= heuristics.min_m && N >= heuristics.min_n && K >= heuristics.min_k && work >= heuristics.min_mnk;
}

inline int32_t DotInt8Int8(const int8_t* a, const int8_t* b, int n) {
#if defined(DENSECORE_ARM) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t sum = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const int8x16_t va = vld1q_s8(a + k);
        const int8x16_t vb = vld1q_s8(b + k);
        sum = vdotq_s32(sum, va, vb);
    }
    int32_t result = vaddvq_s32(sum);
    for (; k < n; ++k) {
        result += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);
    }
    return result;
#else
    int32_t result = 0;
    for (int k = 0; k < n; ++k) {
        result += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);
    }
    return result;
#endif
}

inline void ExecuteBF16TransB(const MatmulParams& params) {
    const int M = static_cast<int>(params.M);
    const int N = static_cast<int>(params.N);
    const int K = static_cast<int>(params.K);
    const int lda = static_cast<int>(params.lda > 0 ? params.lda : params.K);
    const int ldb = static_cast<int>(params.ldb > 0 ? params.ldb : params.K);
    const int ldc = static_cast<int>(params.ldc > 0 ? params.ldc : params.N);

    const float* a_f32 = (params.a_type == DType::F32) ? static_cast<const float*>(params.a) : nullptr;
    const ggml_bf16_t* a_bf16 = (params.a_type == DType::BF16) ? static_cast<const ggml_bf16_t*>(params.a) : nullptr;
    const ggml_bf16_t* b_bf16 = static_cast<const ggml_bf16_t*>(params.b);
    float* c = static_cast<float*>(params.c);

    constexpr int TILE_N = 8;
    thread_local std::vector<float> a_row;
    thread_local std::vector<float> b_tile;
    a_row.resize(static_cast<size_t>(K));
    b_tile.resize(static_cast<size_t>(TILE_N * K));

    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        const int n_len = std::min(TILE_N, N - n0);
        for (int nn = 0; nn < n_len; ++nn) {
            ggml_bf16_to_fp32_row(b_bf16 + static_cast<size_t>(n0 + nn) * ldb,
                                  b_tile.data() + static_cast<size_t>(nn) * K, K);
        }

        for (int m = 0; m < M; ++m) {
            const float* a_row_f32 = nullptr;
            if (a_f32) {
                a_row_f32 = a_f32 + static_cast<size_t>(m) * lda;
            } else {
                ggml_bf16_to_fp32_row(a_bf16 + static_cast<size_t>(m) * lda, a_row.data(), K);
                a_row_f32 = a_row.data();
            }

            float* c_row = c + static_cast<size_t>(m) * ldc;
            for (int nn = 0; nn < n_len; ++nn) {
                c_row[n0 + nn] = simd::DotF32(a_row_f32, b_tile.data() + static_cast<size_t>(nn) * K, K);
            }
        }
    }
}

inline void ExecuteINT8TransB(const MatmulParams& params) {
    const int M = static_cast<int>(params.M);
    const int N = static_cast<int>(params.N);
    const int K = static_cast<int>(params.K);
    const int lda = static_cast<int>(params.lda > 0 ? params.lda : params.K);
    const int ldb = static_cast<int>(params.ldb > 0 ? params.ldb : params.K);
    const int ldc = static_cast<int>(params.ldc > 0 ? params.ldc : params.N);

    const int8_t* a = static_cast<const int8_t*>(params.a);
    const int8_t* b = static_cast<const int8_t*>(params.b);
    float* c = static_cast<float*>(params.c);

    const float a_scale = params.quant.a_scales ? params.quant.a_scales[0] : 1.0f;
    const float b_scale = params.quant.b_scales ? params.quant.b_scales[0] : 1.0f;
    const float scale = a_scale * b_scale;
    const int32_t a_zp = params.quant.a_zero_points ? params.quant.a_zero_points[0] : 0;
    const int32_t b_zp = params.quant.b_zero_points ? params.quant.b_zero_points[0] : 0;

    // Precompute per-row sum of B (used for a_zp correction across all M rows)
    thread_local std::vector<int32_t> sum_b_rows;
    if (a_zp != 0) {
        sum_b_rows.resize(static_cast<size_t>(N));
        for (int n = 0; n < N; ++n) {
            const int8_t* b_row = b + static_cast<size_t>(n) * ldb;
            int32_t s = 0;
            for (int k = 0; k < K; ++k) s += static_cast<int32_t>(b_row[k]);
            sum_b_rows[n] = s;
        }
    }

    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = a + static_cast<size_t>(m) * lda;
        float* c_row = c + static_cast<size_t>(m) * ldc;

        int32_t sum_a = 0;
        if (b_zp != 0) {
            for (int k = 0; k < K; ++k) sum_a += static_cast<int32_t>(a_row[k]);
        }

        for (int n = 0; n < N; ++n) {
            const int8_t* b_row = b + static_cast<size_t>(n) * ldb;
            // dot(a - a_zp, b - b_zp) = dot(a,b) - a_zp*sum(b) - b_zp*sum(a) + K*a_zp*b_zp
            int32_t dot = DotInt8Int8(a_row, b_row, K);
            if (a_zp != 0) dot -= a_zp * sum_b_rows[n];
            if (b_zp != 0) dot -= b_zp * sum_a;
            if (a_zp != 0 && b_zp != 0) dot += K * a_zp * b_zp;
            c_row[n] = static_cast<float>(dot) * scale;
        }
    }
}

class DenseCoreMatmulBackend final : public MatmulBackend {
public:
    const char* Name() const override { return "DenseCore"; }
    bool IsAvailable() const override { return true; }

    bool Supports(const MatmulParams& params) const override {
        if (params.trans_a) return false;
        if (!params.trans_b) return false;
        return SupportsF32TransB(params) || SupportsBF16TransB(params) || SupportsINT8TransB(params);
    }

    void PrepareWeights(const MatmulParams& /*params*/, const char* /*name*/) override {}

    void Execute(const MatmulParams& params) override {
        if (!Supports(params) || !params.a || !params.b || !params.c) return;
        if (SupportsF32TransB(params)) {
            ExecuteDenseCoreMatmulTransBF32(params);
            return;
        }

        if (SupportsBF16TransB(params)) {
            ExecuteBF16TransB(params);
            return;
        }

        if (SupportsINT8TransB(params)) {
            ExecuteINT8TransB(params);
        }
    }
};

}  // namespace

void ExecuteDenseCoreMatmulTransBF32(const MatmulParams& params) {
    if (!SupportsF32TransB(params) || !params.a || !params.b || !params.c) return;

    const int M = static_cast<int>(params.M);
    const int N = static_cast<int>(params.N);
    const int K = static_cast<int>(params.K);
    const int lda = static_cast<int>(params.lda > 0 ? params.lda : params.K);
    const int ldb = static_cast<int>(params.ldb > 0 ? params.ldb : params.K);
    const int ldc = static_cast<int>(params.ldc > 0 ? params.ldc : params.N);
    if (M <= 0 || N <= 0 || K <= 0) return;

    const float* a = static_cast<const float*>(params.a);
    const float* b = static_cast<const float*>(params.b);
    float* c = static_cast<float*>(params.c);

    if (IsContiguousF32TransB(params)) {
        constexpr int64_t kMinWorkPerTask = 512 * 1024;
        const int64_t work = static_cast<int64_t>(M) * N * K;
        auto& pool = GetCpuBackend().GetThreadPool(0);
        const int tasks_by_work =
            static_cast<int>(std::min<int64_t>(pool.GetNumThreads(), std::max<int64_t>(1, work / kMinWorkPerTask)));
        const int tasks = std::min({pool.GetNumThreads(), M, tasks_by_work});

        if (tasks > 1 && ShouldParallelizeF32TransB(M, N, K, work)) {
            pool.ParallelFor(tasks, [=](int task_start, int task_end, int /*thread_id*/) {
                const int m_start = static_cast<int>(static_cast<int64_t>(task_start) * M / tasks);
                const int m_end = static_cast<int>(static_cast<int64_t>(task_end) * M / tasks);
                hwy_kernels::GemmFP32_Hwy(c + static_cast<size_t>(m_start) * N, a + static_cast<size_t>(m_start) * K, b,
                                          m_end - m_start, N, K, 0, N);
            });
            return;
        }

        hwy_kernels::GemmFP32_Hwy(c, a, b, M, N, K, 0, N);
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + static_cast<size_t>(m) * lda;
        float* c_row = c + static_cast<size_t>(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const float* b_row = b + static_cast<size_t>(n) * ldb;
            c_row[n] = simd::DotF32(a_row, b_row, K);
        }
    }
}

MatmulBackend& GetDenseCoreMatmulBackend() {
    static DenseCoreMatmulBackend backend;
    return backend;
}

}  // namespace densecore
