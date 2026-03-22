/**
 * @file matmul_backend_densecore.cpp
 * @brief DenseCore matmul backend (baseline kernels)
 */

#include "../include/matmul_backend.h"

#include "../include/simd_ops.h"

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
            ggml_bf16_to_fp32_row(b_bf16 + static_cast<size_t>(n0 + nn) * ldb, b_tile.data() + static_cast<size_t>(nn) * K,
                                  K);
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

    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = a + static_cast<size_t>(m) * lda;
        float* c_row = c + static_cast<size_t>(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const int8_t* b_row = b + static_cast<size_t>(n) * ldb;
            c_row[n] = static_cast<float>(DotInt8Int8(a_row, b_row, K));
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
            const float* a = static_cast<const float*>(params.a);
            const float* b = static_cast<const float*>(params.b);
            float* c = static_cast<float*>(params.c);
            simd::MatMulTransB(c, a, b, static_cast<int>(params.M), static_cast<int>(params.N),
                               static_cast<int>(params.K));
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

MatmulBackend& GetDenseCoreMatmulBackend() {
    static DenseCoreMatmulBackend backend;
    return backend;
}

}  // namespace densecore
