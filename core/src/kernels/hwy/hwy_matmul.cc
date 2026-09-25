/**
 * @file hwy_matmul.cc
 * @brief FP32 GEMM kernel with MR×NR register blocking via Google Highway
 *
 * Computes C[:, n_start:n_end) = A[M,K] x B[N,K]^T using a two-level
 * register-tiled scheme:
 *
 *   MR (row tile)  : up to 4 rows of A accumulated simultaneously.
 *   NR (col tile)  : up to 2 rows of B per MR-tile, halving A-row reload cost.
 *
 * Memory-traffic analysis for MR=4, NR=2 (AVX2, lanes=8):
 *
 *   Old (NR=1):  per n:  load 4*K A-floats + K B-floats; 4 ReduceSum.
 *   New (NR=2):  per 2n: load 4*K A-floats (reused!) + 2*K B-floats; 8 ReduceSum.
 *
 *   A bandwidth halved for prefill (large M, large N).
 *   Arithmetic intensity: ~0.40 → ~0.67 FLOP/B  (+67 %).
 *
 * [P1 fix] Added NR-blocked GemmFP32BlockNR<MR,NR> replacing the scalar-output
 * loop. The inner N-loop now steps by NR=2, with NR=1 handling odd remainders.
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

// ============================================================================
// GemmFP32BlockNR<MR, NR>
//
// Computes a MR-row × NR-column output tile:
//   C[m_base : m_base+MR,  n_base : n_base+NR]
//     += A[m_base : m_base+MR, 0:K] × B[n_base : n_base+NR, 0:K]^T
//
// MR in {1,2,3,4}, NR in {1,2}.
// When NR=2, each A row is loaded ONCE and accumulated against both B rows,
// cutting A-bandwidth by half compared to the NR=1 loop.
// ============================================================================
template <int MR, int NR>
HWY_INLINE void GemmFP32BlockNR(float* HWY_RESTRICT C, const float* HWY_RESTRICT A, const float* HWY_RESTRICT B,
                                 int N, int K, int m_base, int n_base) {
    static_assert(MR >= 1 && MR <= 4, "MR must be 1..4");
    static_assert(NR >= 1 && NR <= 2, "NR must be 1..2");

    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes <= 0) return;
    const int prefetch_elems = PrefetchElemsAhead(lanes);

    // ---- A row pointers ----
    const float* a0 = nullptr;
    const float* a1 = nullptr;
    const float* a2 = nullptr;
    const float* a3 = nullptr;
    if constexpr (MR >= 1) a0 = A + static_cast<int64_t>(m_base + 0) * K;
    if constexpr (MR >= 2) a1 = A + static_cast<int64_t>(m_base + 1) * K;
    if constexpr (MR >= 3) a2 = A + static_cast<int64_t>(m_base + 2) * K;
    if constexpr (MR >= 4) a3 = A + static_cast<int64_t>(m_base + 3) * K;

    // ---- B row pointers ----
    const float* b0 = B + static_cast<int64_t>(n_base + 0) * K;
    const float* b1 = nullptr;
    if constexpr (NR >= 2) b1 = B + static_cast<int64_t>(n_base + 1) * K;

    // ---- MR×NR vector accumulators ----
    auto acc00 = hn::Zero(d);
    auto acc01 = hn::Zero(d);  // used only when NR >= 2
    auto acc10 = hn::Zero(d);
    auto acc11 = hn::Zero(d);
    auto acc20 = hn::Zero(d);
    auto acc21 = hn::Zero(d);
    auto acc30 = hn::Zero(d);
    auto acc31 = hn::Zero(d);

    // ---- Scalar tail accumulators ----
    float tail00 = 0.f, tail01 = 0.f;
    float tail10 = 0.f, tail11 = 0.f;
    float tail20 = 0.f, tail21 = 0.f;
    float tail30 = 0.f, tail31 = 0.f;

    int k = 0;

    // ---- Unrolled 2× SIMD loop ----
#if defined(__clang__)
#pragma unroll(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (; k + 2 * lanes <= K; k += 2 * lanes) {
        const int pf_k = k + prefetch_elems;
        if (pf_k < K) {
            ::hwy::Prefetch(b0 + pf_k);
            if constexpr (NR >= 2) ::hwy::Prefetch(b1 + pf_k);
            if constexpr (MR >= 1) ::hwy::Prefetch(a0 + pf_k);
            if constexpr (MR >= 2) ::hwy::Prefetch(a1 + pf_k);
            if constexpr (MR >= 3) ::hwy::Prefetch(a2 + pf_k);
            if constexpr (MR >= 4) ::hwy::Prefetch(a3 + pf_k);
        }

        // Load B vectors (shared across all MR rows)
        const auto vb0_lo = hn::LoadU(d, b0 + k);
        const auto vb0_hi = hn::LoadU(d, b0 + k + lanes);
        auto vb1_lo = hn::Zero(d);
        auto vb1_hi = hn::Zero(d);
        if constexpr (NR >= 2) {
            vb1_lo = hn::LoadU(d, b1 + k);
            vb1_hi = hn::LoadU(d, b1 + k + lanes);
        }

        // A row 0: load once, accumulate into all NR output columns
        if constexpr (MR >= 1) {
            const auto va_lo = hn::LoadU(d, a0 + k);
            const auto va_hi = hn::LoadU(d, a0 + k + lanes);
            acc00 = hn::MulAdd(va_lo, vb0_lo, acc00);
            acc00 = hn::MulAdd(va_hi, vb0_hi, acc00);
            if constexpr (NR >= 2) {
                acc01 = hn::MulAdd(va_lo, vb1_lo, acc01);
                acc01 = hn::MulAdd(va_hi, vb1_hi, acc01);
            }
        }
        if constexpr (MR >= 2) {
            const auto va_lo = hn::LoadU(d, a1 + k);
            const auto va_hi = hn::LoadU(d, a1 + k + lanes);
            acc10 = hn::MulAdd(va_lo, vb0_lo, acc10);
            acc10 = hn::MulAdd(va_hi, vb0_hi, acc10);
            if constexpr (NR >= 2) {
                acc11 = hn::MulAdd(va_lo, vb1_lo, acc11);
                acc11 = hn::MulAdd(va_hi, vb1_hi, acc11);
            }
        }
        if constexpr (MR >= 3) {
            const auto va_lo = hn::LoadU(d, a2 + k);
            const auto va_hi = hn::LoadU(d, a2 + k + lanes);
            acc20 = hn::MulAdd(va_lo, vb0_lo, acc20);
            acc20 = hn::MulAdd(va_hi, vb0_hi, acc20);
            if constexpr (NR >= 2) {
                acc21 = hn::MulAdd(va_lo, vb1_lo, acc21);
                acc21 = hn::MulAdd(va_hi, vb1_hi, acc21);
            }
        }
        if constexpr (MR >= 4) {
            const auto va_lo = hn::LoadU(d, a3 + k);
            const auto va_hi = hn::LoadU(d, a3 + k + lanes);
            acc30 = hn::MulAdd(va_lo, vb0_lo, acc30);
            acc30 = hn::MulAdd(va_hi, vb0_hi, acc30);
            if constexpr (NR >= 2) {
                acc31 = hn::MulAdd(va_lo, vb1_lo, acc31);
                acc31 = hn::MulAdd(va_hi, vb1_hi, acc31);
            }
        }
    }

    // ---- Single SIMD loop for remaining full vectors ----
#if defined(__clang__)
#pragma unroll(4)
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (; k + lanes <= K; k += lanes) {
        const auto vb0_v = hn::LoadU(d, b0 + k);
        auto vb1_v = hn::Zero(d);
        if constexpr (NR >= 2) vb1_v = hn::LoadU(d, b1 + k);

        if constexpr (MR >= 1) {
            const auto va = hn::LoadU(d, a0 + k);
            acc00 = hn::MulAdd(va, vb0_v, acc00);
            if constexpr (NR >= 2) acc01 = hn::MulAdd(va, vb1_v, acc01);
        }
        if constexpr (MR >= 2) {
            const auto va = hn::LoadU(d, a1 + k);
            acc10 = hn::MulAdd(va, vb0_v, acc10);
            if constexpr (NR >= 2) acc11 = hn::MulAdd(va, vb1_v, acc11);
        }
        if constexpr (MR >= 3) {
            const auto va = hn::LoadU(d, a2 + k);
            acc20 = hn::MulAdd(va, vb0_v, acc20);
            if constexpr (NR >= 2) acc21 = hn::MulAdd(va, vb1_v, acc21);
        }
        if constexpr (MR >= 4) {
            const auto va = hn::LoadU(d, a3 + k);
            acc30 = hn::MulAdd(va, vb0_v, acc30);
            if constexpr (NR >= 2) acc31 = hn::MulAdd(va, vb1_v, acc31);
        }
    }

    // ---- Scalar tail loop ----
    for (; k < K; ++k) {
        const float vb0_s = b0[k];
        if constexpr (MR >= 1) tail00 += a0[k] * vb0_s;
        if constexpr (MR >= 2) tail10 += a1[k] * vb0_s;
        if constexpr (MR >= 3) tail20 += a2[k] * vb0_s;
        if constexpr (MR >= 4) tail30 += a3[k] * vb0_s;
        if constexpr (NR >= 2) {
            const float vb1_s = b1[k];
            if constexpr (MR >= 1) tail01 += a0[k] * vb1_s;
            if constexpr (MR >= 2) tail11 += a1[k] * vb1_s;
            if constexpr (MR >= 3) tail21 += a2[k] * vb1_s;
            if constexpr (MR >= 4) tail31 += a3[k] * vb1_s;
        }
    }

    // ---- Store MR×NR results ----
    if constexpr (MR >= 1) {
        C[static_cast<int64_t>(m_base + 0) * N + n_base + 0] = hn::ReduceSum(d, acc00) + tail00;
        if constexpr (NR >= 2)
            C[static_cast<int64_t>(m_base + 0) * N + n_base + 1] = hn::ReduceSum(d, acc01) + tail01;
    }
    if constexpr (MR >= 2) {
        C[static_cast<int64_t>(m_base + 1) * N + n_base + 0] = hn::ReduceSum(d, acc10) + tail10;
        if constexpr (NR >= 2)
            C[static_cast<int64_t>(m_base + 1) * N + n_base + 1] = hn::ReduceSum(d, acc11) + tail11;
    }
    if constexpr (MR >= 3) {
        C[static_cast<int64_t>(m_base + 2) * N + n_base + 0] = hn::ReduceSum(d, acc20) + tail20;
        if constexpr (NR >= 2)
            C[static_cast<int64_t>(m_base + 2) * N + n_base + 1] = hn::ReduceSum(d, acc21) + tail21;
    }
    if constexpr (MR >= 4) {
        C[static_cast<int64_t>(m_base + 3) * N + n_base + 0] = hn::ReduceSum(d, acc30) + tail30;
        if constexpr (NR >= 2)
            C[static_cast<int64_t>(m_base + 3) * N + n_base + 1] = hn::ReduceSum(d, acc31) + tail31;
    }
}

// ============================================================================
// GemmFP32Impl
//
// Outer M-loop: MR=4 tiles, then remainder.
// Inner N-loop: NR=2 tiles, then NR=1 remainder.
// ============================================================================
void GemmFP32Impl(float* HWY_RESTRICT C, const float* HWY_RESTRICT A, const float* HWY_RESTRICT B, int M, int N, int K,
                  int n_start, int n_end) {
    if (!C || !A || !B || M <= 0 || N <= 0 || K <= 0) return;

    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (n_start >= n_end) return;

    // Helper: run the NR=2 then NR=1 N-loop for a given MR tile
    auto run_n_loop = [&]<int MR>() {
        int m = 0;
        for (; m + MR <= M; m += MR) {
            int n = n_start;
            for (; n + 2 <= n_end; n += 2) {
                GemmFP32BlockNR<MR, 2>(C, A, B, N, K, m, n);
            }
            if (n < n_end) {
                GemmFP32BlockNR<MR, 1>(C, A, B, N, K, m, n);
            }
        }
    };

    // MR=4 main loop
    {
        int m = 0;
        for (; m + 4 <= M; m += 4) {
            int n = n_start;
            for (; n + 2 <= n_end; n += 2) {
                GemmFP32BlockNR<4, 2>(C, A, B, N, K, m, n);
            }
            if (n < n_end) {
                GemmFP32BlockNR<4, 1>(C, A, B, N, K, m, n);
            }
        }

        // MR remainder
        switch (M - m) {
        case 3: {
            int n = n_start;
            for (; n + 2 <= n_end; n += 2) GemmFP32BlockNR<3, 2>(C, A, B, N, K, m, n);
            if (n < n_end) GemmFP32BlockNR<3, 1>(C, A, B, N, K, m, n);
            break;
        }
        case 2: {
            int n = n_start;
            for (; n + 2 <= n_end; n += 2) GemmFP32BlockNR<2, 2>(C, A, B, N, K, m, n);
            if (n < n_end) GemmFP32BlockNR<2, 1>(C, A, B, N, K, m, n);
            break;
        }
        case 1: {
            int n = n_start;
            for (; n + 2 <= n_end; n += 2) GemmFP32BlockNR<1, 2>(C, A, B, N, K, m, n);
            if (n < n_end) GemmFP32BlockNR<1, 1>(C, A, B, N, K, m, n);
            break;
        }
        default: break;
        }
    }
    (void)run_n_loop;  // suppress unused warning (lambda kept for readability)
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
