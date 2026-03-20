/**
 * @file hwy_matmul.cc
 * @brief FP32 GEMM kernel for small-batch Split-N parallelism
 *
 * Computes C[:, n_start:n_end) = A[M,K] x B[N,K]^T with register-blocked
 * accumulation across up to 4 rows of A while reusing one B row at a time.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_matmul.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include <hwy/cache_control.h>

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE int PrefetchElemsAhead(int lanes) {
    if (lanes >= 16) return lanes * 8;
    if (lanes >= 8) return lanes * 6;
    return lanes * 4;
}

template <int MR>
HWY_INLINE void GemmFP32BlockN(float* HWY_RESTRICT C, const float* HWY_RESTRICT A, const float* HWY_RESTRICT B, int N,
                               int K, int m_base, int n_start, int n_end) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const int k_step = static_cast<int>(lanes);
    if (k_step <= 0) return;

    const int prefetch_elems = PrefetchElemsAhead(k_step);

    const float* a0 = nullptr;
    const float* a1 = nullptr;
    const float* a2 = nullptr;
    const float* a3 = nullptr;

    if constexpr (MR >= 1) a0 = A + static_cast<int64_t>(m_base + 0) * K;
    if constexpr (MR >= 2) a1 = A + static_cast<int64_t>(m_base + 1) * K;
    if constexpr (MR >= 3) a2 = A + static_cast<int64_t>(m_base + 2) * K;
    if constexpr (MR >= 4) a3 = A + static_cast<int64_t>(m_base + 3) * K;

    for (int n = n_start; n < n_end; ++n) {
        const float* b_row = B + static_cast<int64_t>(n) * K;

        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        auto acc3 = hn::Zero(d);

        float tail0 = 0.0f;
        float tail1 = 0.0f;
        float tail2 = 0.0f;
        float tail3 = 0.0f;

        int k = 0;

#if defined(__clang__)
#pragma unroll(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
        for (; k + 2 * k_step <= K; k += 2 * k_step) {
            const int pf_k = k + prefetch_elems;
            if (pf_k < K) {
                ::hwy::Prefetch(b_row + pf_k);
                if constexpr (MR >= 1) ::hwy::Prefetch(a0 + pf_k);
                if constexpr (MR >= 2) ::hwy::Prefetch(a1 + pf_k);
                if constexpr (MR >= 3) ::hwy::Prefetch(a2 + pf_k);
                if constexpr (MR >= 4) ::hwy::Prefetch(a3 + pf_k);
            }

            const auto vb0 = hn::LoadU(d, b_row + k);
            const auto vb1 = hn::LoadU(d, b_row + k + k_step);

            if constexpr (MR >= 1) {
                acc0 = hn::MulAdd(hn::LoadU(d, a0 + k), vb0, acc0);
                acc0 = hn::MulAdd(hn::LoadU(d, a0 + k + k_step), vb1, acc0);
            }
            if constexpr (MR >= 2) {
                acc1 = hn::MulAdd(hn::LoadU(d, a1 + k), vb0, acc1);
                acc1 = hn::MulAdd(hn::LoadU(d, a1 + k + k_step), vb1, acc1);
            }
            if constexpr (MR >= 3) {
                acc2 = hn::MulAdd(hn::LoadU(d, a2 + k), vb0, acc2);
                acc2 = hn::MulAdd(hn::LoadU(d, a2 + k + k_step), vb1, acc2);
            }
            if constexpr (MR >= 4) {
                acc3 = hn::MulAdd(hn::LoadU(d, a3 + k), vb0, acc3);
                acc3 = hn::MulAdd(hn::LoadU(d, a3 + k + k_step), vb1, acc3);
            }
        }

#if defined(__clang__)
#pragma unroll(4)
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (; k + k_step <= K; k += k_step) {
            const auto vb = hn::LoadU(d, b_row + k);

            if constexpr (MR >= 1) acc0 = hn::MulAdd(hn::LoadU(d, a0 + k), vb, acc0);
            if constexpr (MR >= 2) acc1 = hn::MulAdd(hn::LoadU(d, a1 + k), vb, acc1);
            if constexpr (MR >= 3) acc2 = hn::MulAdd(hn::LoadU(d, a2 + k), vb, acc2);
            if constexpr (MR >= 4) acc3 = hn::MulAdd(hn::LoadU(d, a3 + k), vb, acc3);
        }

        for (; k < K; ++k) {
            const float vb = b_row[k];
            if constexpr (MR >= 1) tail0 += a0[k] * vb;
            if constexpr (MR >= 2) tail1 += a1[k] * vb;
            if constexpr (MR >= 3) tail2 += a2[k] * vb;
            if constexpr (MR >= 4) tail3 += a3[k] * vb;
        }

        if constexpr (MR >= 1) {
            C[static_cast<int64_t>(m_base + 0) * N + n] = hn::ReduceSum(d, acc0) + tail0;
        }
        if constexpr (MR >= 2) {
            C[static_cast<int64_t>(m_base + 1) * N + n] = hn::ReduceSum(d, acc1) + tail1;
        }
        if constexpr (MR >= 3) {
            C[static_cast<int64_t>(m_base + 2) * N + n] = hn::ReduceSum(d, acc2) + tail2;
        }
        if constexpr (MR >= 4) {
            C[static_cast<int64_t>(m_base + 3) * N + n] = hn::ReduceSum(d, acc3) + tail3;
        }
    }
}

void GemmFP32Impl(float* HWY_RESTRICT C, const float* HWY_RESTRICT A, const float* HWY_RESTRICT B, int M, int N, int K,
                  int n_start, int n_end) {
    if (!C || !A || !B || M <= 0 || N <= 0 || K <= 0) return;

    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (n_start >= n_end) return;

    int m = 0;
    for (; m + 4 <= M; m += 4) {
        GemmFP32BlockN<4>(C, A, B, N, K, m, n_start, n_end);
    }

    switch (M - m) {
    case 3: GemmFP32BlockN<3>(C, A, B, N, K, m, n_start, n_end); break;
    case 2: GemmFP32BlockN<2>(C, A, B, N, K, m, n_start, n_end); break;
    case 1: GemmFP32BlockN<1>(C, A, B, N, K, m, n_start, n_end); break;
    default: break;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(GemmFP32Impl);

void GemmFP32_Hwy(float* C, const float* A, const float* B, int M, int N, int K, int n_start, int n_end) {
    HWY_DYNAMIC_DISPATCH(GemmFP32Impl)(C, A, B, M, N, K, n_start, n_end);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
