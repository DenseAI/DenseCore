/**
 * @file hwy_normalization.cc
 * @brief Normalization (LayerNorm, RMSNorm, AdaLN) via Google Highway
 *
 * Implements standard normalization layers using portable SIMD.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_normalization.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// Add Implementation
// ============================================================================

void AddImpl(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, float* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<float> d;
    size_t i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

// ============================================================================
// RMSNorm Implementation
// ============================================================================

void RMSNormImpl(const float* HWY_RESTRICT x, const float* HWY_RESTRICT weight, float* HWY_RESTRICT out, size_t n,
                 float eps) {
    if (n == 0) return;

    const hn::ScalableTag<float> d;

    // 1. Calculate sum of squares
    auto sum_sq = hn::Zero(d);
    float scalar_sum_sq = 0.0f;
    size_t i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v = hn::LoadU(d, x + i);
        sum_sq = hn::MulAdd(v, v, sum_sq);
    }
    float ssq = hn::ReduceSum(d, sum_sq);
    for (; i < n; ++i) {
        float val = x[i];
        scalar_sum_sq += val * val;
    }
    ssq += scalar_sum_sq;

    // 2. Calculate RMS and scaling factor
    // Using fast rsqrt estimate + 1 Newton-Raphson iteration
    const float mean_sq = ssq / static_cast<float>(n) + eps;
    const auto v_mean_sq = hn::Set(d, mean_sq);
    auto v_rsqrt = hn::ApproximateReciprocalSqrt(v_mean_sq);

    // Newton-Raphson: y = y * (1.5 - 0.5 * x * y * y)
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_1_5 = hn::Set(d, 1.5f);

    // term = 1.5 - 0.5 * (x * y * y)
    auto v_term = hn::NegMulAdd(v_half, hn::Mul(v_mean_sq, hn::Mul(v_rsqrt, v_rsqrt)), v_1_5);
    v_rsqrt = hn::Mul(v_rsqrt, v_term);

    const float scale = hn::GetLane(v_rsqrt);
    const auto v_scale = hn::Set(d, scale);

    // 3. Normalize and scale
    i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v_x = hn::LoadU(d, x + i);
        const auto v_w = hn::LoadU(d, weight + i);
        const auto v_out = hn::Mul(hn::Mul(v_x, v_scale), v_w);
        hn::StoreU(v_out, d, out + i);
    }
    for (; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

// ============================================================================
// Add + RMSNorm Fused Implementation
// ============================================================================

void AddRMSNormImpl(float* HWY_RESTRICT x_out, const float* HWY_RESTRICT x, const float* HWY_RESTRICT residual,
                    const float* HWY_RESTRICT weight, size_t n, float eps) {
    if (n == 0) return;

    const hn::ScalableTag<float> d;

    // Pass 1: Add residual and compute sum of squares
    auto sum_sq = hn::Zero(d);
    size_t i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v_x = hn::LoadU(d, x + i);
        const auto v_res = hn::LoadU(d, residual + i);
        const auto v_sum = hn::Add(v_x, v_res);

        // Store intermediate result (x + residual)
        hn::StoreU(v_sum, d, x_out + i);

        sum_sq = hn::MulAdd(v_sum, v_sum, sum_sq);
    }
    float ssq = hn::ReduceSum(d, sum_sq);
    for (; i < n; ++i) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        ssq += val * val;
    }

    // Calculate scaling
    // Using fast rsqrt estimate + 1 Newton-Raphson iteration
    const float mean_sq = ssq / static_cast<float>(n) + eps;
    const auto v_mean_sq = hn::Set(d, mean_sq);
    auto v_rsqrt = hn::ApproximateReciprocalSqrt(v_mean_sq);

    // Newton-Raphson: y = y * (1.5 - 0.5 * x * y * y)
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_1_5 = hn::Set(d, 1.5f);
    auto v_term = hn::NegMulAdd(v_half, hn::Mul(v_mean_sq, hn::Mul(v_rsqrt, v_rsqrt)), v_1_5);
    v_rsqrt = hn::Mul(v_rsqrt, v_term);

    const float scale = hn::GetLane(v_rsqrt);
    const auto v_scale = hn::Set(d, scale);

    // Pass 2: Normalize and scale
    i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v_val = hn::LoadU(d, x_out + i);
        const auto v_w = hn::LoadU(d, weight + i);
        const auto v_res = hn::Mul(hn::Mul(v_val, v_scale), v_w);
        hn::StoreU(v_res, d, x_out + i);
    }
    for (; i < n; ++i) {
        x_out[i] = x_out[i] * scale * weight[i];
    }
}

// ============================================================================
// LayerNorm Implementation
// ============================================================================

void LayerNormImpl(const float* HWY_RESTRICT x, const float* HWY_RESTRICT gamma, const float* HWY_RESTRICT beta,
                   float* HWY_RESTRICT out, size_t n, float eps) {
    if (n == 0) return;

    const hn::ScalableTag<float> d;

    // 1. Calculate Mean
    auto v_sum = hn::Zero(d);
    auto v_sum_sq = hn::Zero(d);
    float scalar_sum = 0.0f;
    float scalar_sum_sq = 0.0f;
    size_t i = 0;

    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v = hn::LoadU(d, x + i);
        v_sum = hn::Add(v_sum, v);
        v_sum_sq = hn::MulAdd(v, v, v_sum_sq);
    }
    float sum = hn::ReduceSum(d, v_sum);
    float sum_sq = hn::ReduceSum(d, v_sum_sq);

    for (; i < n; ++i) {
        float val = x[i];
        scalar_sum += val;
        scalar_sum_sq += val * val;
    }
    sum += scalar_sum;
    sum_sq += scalar_sum_sq;

    float mean = sum / static_cast<float>(n);
    float var = (sum_sq / static_cast<float>(n)) - (mean * mean);
    if (var < 0.0f) var = 0.0f;

    // Fast Rsqrt
    const auto v_var = hn::Set(d, var + eps);
    auto v_rsqrt = hn::ApproximateReciprocalSqrt(v_var);

    // Newton-Raphson
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_1_5 = hn::Set(d, 1.5f);
    auto v_term = hn::NegMulAdd(v_half, hn::Mul(v_var, hn::Mul(v_rsqrt, v_rsqrt)), v_1_5);
    v_rsqrt = hn::Mul(v_rsqrt, v_term);

    float inv_std = hn::GetLane(v_rsqrt);

    const auto v_mean = hn::Set(d, mean);
    const auto v_inv_std = hn::Set(d, inv_std);

    // 2. Normalize and Apply Gamma/Beta
    i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v_x = hn::LoadU(d, x + i);
        const auto v_g = hn::LoadU(d, gamma + i);
        const auto v_b = hn::LoadU(d, beta + i);

        // out = (x - mean) * inv_std * gamma + beta
        auto v_norm = hn::Mul(hn::Sub(v_x, v_mean), v_inv_std);
        auto v_res = hn::MulAdd(v_norm, v_g, v_b);
        hn::StoreU(v_res, d, out + i);
    }
    for (; i < n; ++i) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================================
// AdaLN Implementation
// ============================================================================

void AdaLNImpl(const float* input, const float* scale, const float* shift, float* output, size_t n, float eps) {
    if (n == 0) return;

    const hn::ScalableTag<float> d;

    // Pass 1: Mean and Variance (Same as LayerNorm)
    auto v_sum = hn::Zero(d);
    auto v_sum_sq = hn::Zero(d);
    float scalar_sum = 0.0f;
    float scalar_sum_sq = 0.0f;
    size_t i = 0;

    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v = hn::LoadU(d, input + i);
        v_sum = hn::Add(v_sum, v);
        v_sum_sq = hn::MulAdd(v, v, v_sum_sq);
    }
    float sum = hn::ReduceSum(d, v_sum);
    float sum_sq = hn::ReduceSum(d, v_sum_sq);

    for (; i < n; ++i) {
        float val = input[i];
        scalar_sum += val;
        scalar_sum_sq += val * val;
    }
    sum += scalar_sum;
    sum_sq += scalar_sum_sq;

    float mean = sum / static_cast<float>(n);
    float var = (sum_sq / static_cast<float>(n)) - (mean * mean);
    if (var < 0.0f) var = 0.0f;

    // Fast Rsqrt
    const auto v_var = hn::Set(d, var + eps);
    auto v_rsqrt = hn::ApproximateReciprocalSqrt(v_var);

    // Newton-Raphson
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_1_5 = hn::Set(d, 1.5f);
    auto v_term = hn::NegMulAdd(v_half, hn::Mul(v_var, hn::Mul(v_rsqrt, v_rsqrt)), v_1_5);
    v_rsqrt = hn::Mul(v_rsqrt, v_term);

    float inv_std = hn::GetLane(v_rsqrt);

    const auto v_mean = hn::Set(d, mean);
    const auto v_inv_std = hn::Set(d, inv_std);
    const auto v_one = hn::Set(d, 1.0f);

    // Pass 2: Normalize and Apply Scale/Shift from tensors
    i = 0;
    for (; i + hn::Lanes(d) <= n; i += hn::Lanes(d)) {
        const auto v_in = hn::LoadU(d, input + i);
        const auto v_s = hn::LoadU(d, scale + i);
        const auto v_sh = hn::LoadU(d, shift + i);

        // normalized = (input - mean) * inv_std
        const auto v_norm = hn::Mul(hn::Sub(v_in, v_mean), v_inv_std);

        // output = normalized * (1 + s) + sh
        const auto v_factor = hn::Add(v_one, v_s);
        const auto v_out = hn::MulAdd(v_norm, v_factor, v_sh);

        hn::StoreU(v_out, d, output + i);
    }
    for (; i < n; ++i) {
        float normalized = (input[i] - mean) * inv_std;
        output[i] = normalized * (1.0f + scale[i]) + shift[i];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(AddImpl);
HWY_EXPORT(RMSNormImpl);
HWY_EXPORT(AddRMSNormImpl);
HWY_EXPORT(LayerNormImpl);
HWY_EXPORT(AdaLNImpl);

void Add_Hwy(float* out, const float* a, const float* b, size_t n) {
    return HWY_DYNAMIC_DISPATCH(AddImpl)(a, b, out, n);
}

void RMSNorm_Hwy(const float* x, const float* weight, float* out, size_t n, float eps) {
    return HWY_DYNAMIC_DISPATCH(RMSNormImpl)(x, weight, out, n, eps);
}

void AddRMSNorm_Hwy(float* x_out, const float* x, const float* residual, const float* weight, size_t n, float eps) {
    return HWY_DYNAMIC_DISPATCH(AddRMSNormImpl)(x_out, x, residual, weight, n, eps);
}

void LayerNorm_Hwy(const float* x, const float* gamma, const float* beta, float* out, size_t n, float eps) {
    return HWY_DYNAMIC_DISPATCH(LayerNormImpl)(x, gamma, beta, out, n, eps);
}

void AdaLN_Hwy(const float* input, const float* scale, const float* shift, float* output, size_t n, float eps) {
    return HWY_DYNAMIC_DISPATCH(AdaLNImpl)(input, scale, shift, output, n, eps);
}

}  // namespace hwy_kernels

// C++ linkage trampoline called from simd_ops.h or adaln.cpp inline
namespace simd {
void AddRMSNorm_HwyDispatch(float* x_out, const float* x, const float* residual, const float* weight, size_t n,
                            float eps) {
    hwy_kernels::AddRMSNorm_Hwy(x_out, x, residual, weight, n, eps);
}

void RMSNorm_HwyDispatch(const float* input, const float* weight, float* output, size_t n, float eps) {
    hwy_kernels::RMSNorm_Hwy(input, weight, output, n, eps);
}

void LayerNorm_HwyDispatch(const float* input, const float* gamma, const float* beta, float* output, size_t n,
                           float eps) {
    hwy_kernels::LayerNorm_Hwy(input, gamma, beta, output, n, eps);
}

void AdaLN_HwyDispatch(const float* input, const float* scale, const float* shift, float* output, size_t n, float eps) {
    hwy_kernels::AdaLN_Hwy(input, scale, shift, output, n, eps);
}

}  // namespace simd

}  // namespace densecore
#endif  // HWY_ONCE
