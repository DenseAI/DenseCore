/**
 * @file hwy_adaln.cc
 * @brief Adaptive Layer Normalization via Google Highway
 *
 * Implements AdaLN(x) = (x - mean) / sqrt(var + eps) * (1 + scale) + shift
 * Replaces manual AVX2 implementation in adaln.cpp.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_adaln.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"
#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void AdaLNImpl(const float* HWY_RESTRICT input, const float* HWY_RESTRICT modulation, float eps,
               float* HWY_RESTRICT output, int64_t N, int64_t D) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    // Modulation contains [scale | shift], stride is D.
    // scale is at modulation + 0, shift is at modulation + D.

    for (int64_t n = 0; n < N; ++n) {
        const float* x_ptr = input + n * D;
        float* out_ptr = output + n * D;

        // modulation is usually [Scale | Shift]
        const float* scale_ptr = modulation;
        const float* shift_ptr = modulation + D;

        // 1. Mean & Variance
        auto v_sum = hn::Zero(d);
        auto v_sq_sum = hn::Zero(d);

        int64_t i = 0;
        for (; i + static_cast<int64_t>(lanes) <= D; i += static_cast<int64_t>(lanes)) {
            const auto val = hn::Load(d, x_ptr + i);
            v_sum = hn::Add(v_sum, val);
            v_sq_sum = hn::MulAdd(val, val, v_sq_sum);
        }

        float sum = hn::ReduceSum(d, v_sum);
        float sq_sum = hn::ReduceSum(d, v_sq_sum);

        for (; i < D; ++i) {
            float val = x_ptr[i];
            sum += val;
            sq_sum += val * val;
        }

        float mean = sum / static_cast<float>(D);
        float var = (sq_sum / static_cast<float>(D)) - (mean * mean);
        float inv_std = 1.0f / std::sqrt(var + eps);

        const auto v_mean = hn::Set(d, mean);
        const auto v_inv_std = hn::Set(d, inv_std);

        // 2. Normalize & Modulate
        // out = (x - mean) * inv_std * (1 + scale) + shift
        const auto v_one = hn::Set(d, 1.0f);

        i = 0;
        for (; i + static_cast<int64_t>(lanes) <= D; i += static_cast<int64_t>(lanes)) {
            const auto val = hn::Load(d, x_ptr + i);
            const auto s = hn::Load(d, scale_ptr + i);
            const auto sh = hn::Load(d, shift_ptr + i);

            auto norm = hn::Mul(hn::Sub(val, v_mean), v_inv_std);
            auto scaled = hn::Mul(norm, hn::Add(v_one, s));
            auto res = hn::Add(scaled, sh);

            hn::Store(res, d, out_ptr + i);
        }

        for (; i < D; ++i) {
            float val = x_ptr[i];
            float s = scale_ptr[i];
            float sh = shift_ptr[i];
            float norm = (val - mean) * inv_std;
            out_ptr[i] = norm * (1.0f + s) + sh;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(AdaLNImpl);

void AdaLN_Hwy(const float* input, const float* modulation, float eps, float* output, int64_t N, int64_t D) {
    HWY_DYNAMIC_DISPATCH(AdaLNImpl)(input, modulation, eps, output, N, D);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
