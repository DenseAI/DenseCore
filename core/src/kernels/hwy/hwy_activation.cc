/**
 * @file hwy_activation.cc
 * @brief Activation functions (SiLU, GELU, Softmax, SiLUMul) via Google Highway
 *
 * Replaces hand-written AVX-512/AVX2/Scalar activation kernels with a single
 * portable source. This also provides ARM NEON/SVE coverage that was previously
 * missing (ARM fell back to scalar).
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_activation.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// FastExp inline template (must be inside open HWY_NAMESPACE)
#include "kernels/hwy/hwy_fastexp.h"

// Scalar FastExp for tails
// FastExpScalar is defined in hwy_fastexp.h

// ============================================================================
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================

void SiLUImpl(const float* HWY_RESTRICT input, float* HWY_RESTRICT output, int64_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0f);
    const auto neg_one = hn::Set(d, -1.0f);

    int64_t i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        const auto x = hn::LoadU(d, input + i);
        const auto neg_x = hn::Mul(x, neg_one);
        const auto exp_neg_x = FastExpHwy(d, neg_x);
        const auto sigmoid = hn::Div(one, hn::Add(one, exp_neg_x));
        hn::StoreU(hn::Mul(x, sigmoid), d, output + i);
    }

    // Scalar tail
    for (; i < n; ++i) {
        float val = input[i];
        output[i] = val / (1.0f + FastExpScalar(-val));
    }
}

// ============================================================================
// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

void GELUImpl(const float* HWY_RESTRICT input, float* HWY_RESTRICT output, int64_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto kSqrt2OverPi = hn::Set(d, 0.7978845608028654f);
    const auto kCoeff = hn::Set(d, 0.044715f);
    const auto half = hn::Set(d, 0.5f);
    const auto one = hn::Set(d, 1.0f);

    int64_t i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        const auto x = hn::LoadU(d, input + i);
        const auto x3 = hn::Mul(x, hn::Mul(x, x));
        const auto inner = hn::MulAdd(kCoeff, x3, x);
        const auto tanh_arg = hn::Mul(kSqrt2OverPi, inner);

        // tanh via (exp(2x) - 1) / (exp(2x) + 1)
        const auto two_arg = hn::Add(tanh_arg, tanh_arg);
        const auto exp_2x = FastExpHwy(d, two_arg);
        const auto tanh_val = hn::Div(hn::Sub(exp_2x, one), hn::Add(exp_2x, one));

        hn::StoreU(hn::Mul(half, hn::Mul(x, hn::Add(one, tanh_val))), d, output + i);
    }

    // Scalar tail
    constexpr float kSqrt2OverPi_s = 0.7978845608028654f;
    constexpr float kCoeff_s = 0.044715f;
    for (; i < n; ++i) {
        float val = input[i];
        float x3 = val * val * val;
        float tanh_arg = kSqrt2OverPi_s * (val + kCoeff_s * x3);
        output[i] = 0.5f * val * (1.0f + std::tanh(tanh_arg));
    }
}

// ============================================================================
// Softmax: exp(x - max) / sum(exp(x - max))
// ============================================================================

void SoftmaxImpl(const float* HWY_RESTRICT input, float* HWY_RESTRICT output, int64_t n) {
    if (n <= 0) return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Pass 1: Find max
    auto v_max = hn::Set(d, input[0]);
    int64_t i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        v_max = hn::Max(v_max, hn::LoadU(d, input + i));
    }
    float max_val = hn::ReduceMax(d, v_max);
    for (; i < n; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Pass 2: exp(x - max) and sum
    const auto v_max_bc = hn::Set(d, max_val);
    auto v_sum = hn::Zero(d);
    i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        const auto v = hn::LoadU(d, input + i);
        const auto exp_v = FastExpHwy(d, hn::Sub(v, v_max_bc));
        hn::StoreU(exp_v, d, output + i);
        v_sum = hn::Add(v_sum, exp_v);
    }
    float sum = hn::ReduceSum(d, v_sum);
    for (; i < n; ++i) {
        float exp_val = FastExpScalar(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    // Pass 3: Normalize
    float inv_sum = 1.0f / sum;
    const auto v_inv = hn::Set(d, inv_sum);
    i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        hn::StoreU(hn::Mul(hn::LoadU(d, output + i), v_inv), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

// ============================================================================
// SiLUMul: SiLU(gate) * up
// ============================================================================

void SiLUMulImpl(const float* HWY_RESTRICT gate, const float* HWY_RESTRICT up, float* HWY_RESTRICT output, int64_t n) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0f);
    const auto neg_one = hn::Set(d, -1.0f);

    int64_t i = 0;
    for (; i + static_cast<int64_t>(N) <= n; i += static_cast<int64_t>(N)) {
        const auto g = hn::LoadU(d, gate + i);
        const auto u = hn::LoadU(d, up + i);
        const auto neg_g = hn::Mul(g, neg_one);
        const auto exp_neg_g = FastExpHwy(d, neg_g);
        const auto sigmoid = hn::Div(one, hn::Add(one, exp_neg_g));
        hn::StoreU(hn::Mul(hn::Mul(g, sigmoid), u), d, output + i);
    }

    for (; i < n; ++i) {
        float g = gate[i];
        float sigmoid = 1.0f / (1.0f + FastExpScalar(-g));
        output[i] = g * sigmoid * up[i];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(SiLUImpl);
HWY_EXPORT(GELUImpl);
HWY_EXPORT(SoftmaxImpl);
HWY_EXPORT(SiLUMulImpl);

void SiLU_Hwy(const float* input, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(SiLUImpl)(input, output, n);
}

void GELU_Hwy(const float* input, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(GELUImpl)(input, output, n);
}

void Softmax_Hwy(const float* input, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(SoftmaxImpl)(input, output, n);
}

void SiLUMul_Hwy(const float* gate, const float* up, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(SiLUMulImpl)(gate, up, output, n);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
