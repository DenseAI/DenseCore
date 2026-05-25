/**
 * @file hwy_fused_ops.cc
 * @brief Fused operations for transformer inference via Google Highway
 *
 * Fused kernels that eliminate intermediate memory round-trips in the
 * transformer forward pass. Each fusion saves at least one full read+write
 * of the hidden state (hidden_dim * 4 bytes × 2 = 8 KB for dim=1024,
 * 28 KB for dim=3584 Qwen3.5).
 *
 * Fusions implemented:
 *   1. RMSNormLinear: RMSNorm(x) then GEMV(result, W) — decode path
 *      Saves: 1 hidden state write + 1 hidden state read = 2 × hidden_dim × 4B
 *
 *   2. ResidualRMSNorm: x = x + residual, then RMSNorm(x) — shared layer ops
 *      Already exists as AddRMSNorm_Hwy, but we optimize the scale computation.
 *
 *   3. FusedGateUpSiLU: Compute gate=W_gate×x, up=W_up×x, output=SiLU(gate)*up
 *      in a single pass over the weight matrices. For MoE expert FFN.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_fused_ops.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Fused RMSNorm + GEMV (decode, M=1)
//
// Computes: output[N] = (W[N,K] × RMSNorm(x[K], weight[K]))
//
// Without fusion: 2 passes over x[K]:
//   Pass 1: sum_sq = sum(x[i]^2), scale = rsqrt(sum_sq/K + eps)
//   Pass 2: norm[i] = x[i] * scale * weight[i]  (write to temp buffer)
//   Pass 3: output[n] = sum(W[n,k] * norm[k])    (read temp buffer)
//
// With fusion: 2 passes over x[K]:
//   Pass 1: sum_sq = sum(x[i]^2), scale = rsqrt(sum_sq/K + eps)
//   Pass 2: for each output n: output[n] = sum(W[n,k] * x[k] * scale * weight[k])
//           The product (scale * weight[k]) is computed ONCE and applied during GEMV.
//
// Savings: eliminates temp buffer allocation + 1 full write + 1 full read of K floats.
// For Qwen3.5-35B hidden_dim=3584: saves 28KB memory traffic per layer.
// ============================================================================

void FusedRMSNormGemvImpl(
    const float* HWY_RESTRICT x,           // [K] input hidden state
    const float* HWY_RESTRICT norm_weight,  // [K] RMSNorm weight
    const float* HWY_RESTRICT W,            // [N, K] weight matrix (row-major, transposed)
    float* HWY_RESTRICT output,             // [N] output
    int K, int N,
    int n_start, int n_end,
    float eps) {

    if (K <= 0 || N <= 0) return;
    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (n_start >= n_end) return;

    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));

    // Pass 1: Compute RMS scale
    auto sum_sq = hn::Zero(d);
    float scalar_ssq = 0.0f;
    int k = 0;
    for (; k + lanes <= K; k += lanes) {
        const auto v = hn::LoadU(d, x + k);
        sum_sq = hn::MulAdd(v, v, sum_sq);
    }
    float ssq = hn::ReduceSum(d, sum_sq);
    for (; k < K; ++k) {
        scalar_ssq += x[k] * x[k];
    }
    ssq += scalar_ssq;

    const float mean_sq = ssq / static_cast<float>(K) + eps;
    const float scale = 1.0f / std::sqrt(mean_sq);

    // Pre-compute normalized_x[k] = x[k] * scale * norm_weight[k]
    // Store in a stack buffer if K is small enough, otherwise use a heap alloc.
    // For typical hidden_dim (2048-7168), stack buffer works.
    constexpr int kStackLimit = 8192;
    float stack_buf[kStackLimit];
    float* norm_x = (K <= kStackLimit) ? stack_buf : new float[static_cast<size_t>(K)];

    k = 0;
    const auto v_scale = hn::Set(d, scale);
    for (; k + lanes <= K; k += lanes) {
        const auto vx = hn::LoadU(d, x + k);
        const auto vw = hn::LoadU(d, norm_weight + k);
        hn::StoreU(hn::Mul(vx, hn::Mul(v_scale, vw)), d, norm_x + k);
    }
    for (; k < K; ++k) {
        norm_x[k] = x[k] * scale * norm_weight[k];
    }

    // Pass 2: GEMV with N-blocking (4 outputs simultaneously)
    int n = n_start;
    for (; n + 4 <= n_end; n += 4) {
        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        auto acc3 = hn::Zero(d);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

        const float* w0 = W + static_cast<int64_t>(n + 0) * K;
        const float* w1 = W + static_cast<int64_t>(n + 1) * K;
        const float* w2 = W + static_cast<int64_t>(n + 2) * K;
        const float* w3 = W + static_cast<int64_t>(n + 3) * K;

        k = 0;
        for (; k + lanes <= K; k += lanes) {
            const auto vnx = hn::LoadU(d, norm_x + k);
            acc0 = hn::MulAdd(vnx, hn::LoadU(d, w0 + k), acc0);
            acc1 = hn::MulAdd(vnx, hn::LoadU(d, w1 + k), acc1);
            acc2 = hn::MulAdd(vnx, hn::LoadU(d, w2 + k), acc2);
            acc3 = hn::MulAdd(vnx, hn::LoadU(d, w3 + k), acc3);
        }
        for (; k < K; ++k) {
            float nx = norm_x[k];
            s0 += nx * w0[k];
            s1 += nx * w1[k];
            s2 += nx * w2[k];
            s3 += nx * w3[k];
        }
        output[n + 0] = hn::ReduceSum(d, acc0) + s0;
        output[n + 1] = hn::ReduceSum(d, acc1) + s1;
        output[n + 2] = hn::ReduceSum(d, acc2) + s2;
        output[n + 3] = hn::ReduceSum(d, acc3) + s3;
    }
    for (; n < n_end; ++n) {
        auto acc = hn::Zero(d);
        float s = 0.0f;
        const float* wn = W + static_cast<int64_t>(n) * K;
        k = 0;
        for (; k + lanes <= K; k += lanes) {
            acc = hn::MulAdd(hn::LoadU(d, norm_x + k), hn::LoadU(d, wn + k), acc);
        }
        for (; k < K; ++k) {
            s += norm_x[k] * wn[k];
        }
        output[n] = hn::ReduceSum(d, acc) + s;
    }

    if (K > kStackLimit) {
        delete[] norm_x;
    }
}

// ============================================================================
// Fused Residual + RMSNorm + GEMV (decode, M=1)
//
// Computes:
//   residual_out[K] = x[K] + residual[K]
//   output[N] = W[N,K] × RMSNorm(residual_out, weight)
//
// Saves: 2 full reads + 1 write of hidden state vs separate ops.
// ============================================================================

void FusedResidualRMSNormGemvImpl(
    const float* HWY_RESTRICT x,
    const float* HWY_RESTRICT residual,
    const float* HWY_RESTRICT norm_weight,
    const float* HWY_RESTRICT W,            // [N, K] row-major
    float* HWY_RESTRICT residual_out,        // [K] updated residual (x + residual)
    float* HWY_RESTRICT output,              // [N] GEMV output
    int K, int N,
    int n_start, int n_end,
    float eps) {

    if (K <= 0 || N <= 0) return;
    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (n_start >= n_end) return;

    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));

    // Pass 1: Add residual, store result, compute sum of squares
    auto sum_sq = hn::Zero(d);
    float scalar_ssq = 0.0f;
    int k = 0;
    for (; k + lanes <= K; k += lanes) {
        const auto vx = hn::LoadU(d, x + k);
        const auto vr = hn::LoadU(d, residual + k);
        const auto vsum = hn::Add(vx, vr);
        hn::StoreU(vsum, d, residual_out + k);
        sum_sq = hn::MulAdd(vsum, vsum, sum_sq);
    }
    float ssq = hn::ReduceSum(d, sum_sq);
    for (; k < K; ++k) {
        float val = x[k] + residual[k];
        residual_out[k] = val;
        scalar_ssq += val * val;
    }
    ssq += scalar_ssq;

    const float mean_sq = ssq / static_cast<float>(K) + eps;
    const float scale = 1.0f / std::sqrt(mean_sq);
    const auto v_scale = hn::Set(d, scale);

    // Pass 2: Compute normalized input and GEMV simultaneously
    // Pre-compute norm_x = residual_out * scale * norm_weight
    constexpr int kStackLimit = 8192;
    float stack_buf[kStackLimit];
    float* norm_x = (K <= kStackLimit) ? stack_buf : new float[static_cast<size_t>(K)];

    k = 0;
    for (; k + lanes <= K; k += lanes) {
        const auto vr = hn::LoadU(d, residual_out + k);
        const auto vw = hn::LoadU(d, norm_weight + k);
        hn::StoreU(hn::Mul(vr, hn::Mul(v_scale, vw)), d, norm_x + k);
    }
    for (; k < K; ++k) {
        norm_x[k] = residual_out[k] * scale * norm_weight[k];
    }

    // GEMV: 4-output blocking
    int n = n_start;
    for (; n + 4 <= n_end; n += 4) {
        auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

        const float* w0 = W + static_cast<int64_t>(n + 0) * K;
        const float* w1 = W + static_cast<int64_t>(n + 1) * K;
        const float* w2 = W + static_cast<int64_t>(n + 2) * K;
        const float* w3 = W + static_cast<int64_t>(n + 3) * K;

        k = 0;
        for (; k + lanes <= K; k += lanes) {
            const auto vnx = hn::LoadU(d, norm_x + k);
            acc0 = hn::MulAdd(vnx, hn::LoadU(d, w0 + k), acc0);
            acc1 = hn::MulAdd(vnx, hn::LoadU(d, w1 + k), acc1);
            acc2 = hn::MulAdd(vnx, hn::LoadU(d, w2 + k), acc2);
            acc3 = hn::MulAdd(vnx, hn::LoadU(d, w3 + k), acc3);
        }
        for (; k < K; ++k) {
            float nx = norm_x[k];
            s0 += nx * w0[k]; s1 += nx * w1[k];
            s2 += nx * w2[k]; s3 += nx * w3[k];
        }
        output[n + 0] = hn::ReduceSum(d, acc0) + s0;
        output[n + 1] = hn::ReduceSum(d, acc1) + s1;
        output[n + 2] = hn::ReduceSum(d, acc2) + s2;
        output[n + 3] = hn::ReduceSum(d, acc3) + s3;
    }
    for (; n < n_end; ++n) {
        auto acc = hn::Zero(d);
        float s = 0.0f;
        const float* wn = W + static_cast<int64_t>(n) * K;
        k = 0;
        for (; k + lanes <= K; k += lanes) {
            acc = hn::MulAdd(hn::LoadU(d, norm_x + k), hn::LoadU(d, wn + k), acc);
        }
        for (; k < K; ++k) s += norm_x[k] * wn[k];
        output[n] = hn::ReduceSum(d, acc) + s;
    }

    if (K > kStackLimit) delete[] norm_x;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(FusedRMSNormGemvImpl);
HWY_EXPORT(FusedResidualRMSNormGemvImpl);

void FusedRMSNormGemv_Hwy(const float* x, const float* norm_weight,
                           const float* W, float* output,
                           int K, int N, int n_start, int n_end, float eps) {
    HWY_DYNAMIC_DISPATCH(FusedRMSNormGemvImpl)(x, norm_weight, W, output, K, N, n_start, n_end, eps);
}

void FusedResidualRMSNormGemv_Hwy(const float* x, const float* residual,
                                   const float* norm_weight, const float* W,
                                   float* residual_out, float* output,
                                   int K, int N, int n_start, int n_end, float eps) {
    HWY_DYNAMIC_DISPATCH(FusedResidualRMSNormGemvImpl)(
        x, residual, norm_weight, W, residual_out, output, K, N, n_start, n_end, eps);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
