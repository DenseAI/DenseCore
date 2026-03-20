/**
 * @file hwy_fp8.cc
 * @brief FP8 conversion and GEMV kernels via Google Highway
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_fp8.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

// #include "kernels/cpu_fp8.h"  // Removed

#include "kernels/hwy/hwy_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Helper: safe load of N bytes into a vector, promoting to uint32
template <class D, class DIdx> HWY_INLINE hn::Vec<DIdx> LoadAndExpand(D d, DIdx di, const uint8_t* ptr, size_t count) {
    (void)d;

    using Du8 = hn::Rebind<uint8_t, D>;
    using Du16 = hn::Rebind<uint16_t, D>;

    const Du8 du8;
    const Du16 du16;
    const size_t n = std::min(count, hn::Lanes(du8));

    // Load uint8 tail safely, zero-filling the rest of the vector.
    auto v_u8 = hn::LoadN(du8, ptr, n);

    // Promote uint8 -> uint16 -> uint32
    auto v_u16 = hn::PromoteTo(du16, v_u8);
    auto v_u32 = hn::PromoteTo(di, v_u16);

    return v_u32;
}

// Bitwise conversion for E5M2
// S EEEEE MM (Bias 15) -> S EEEEEEEE MMMMM... (Bias 127)
template <class D, class DIdx> HWY_INLINE hn::Vec<D> ConvertE5M2(D d, DIdx du32, hn::Vec<DIdx> v_val) {
    auto v_sign = hn::And(v_val, hn::Set(du32, 0x80));
    auto v_exp = hn::And(hn::ShiftRight<2>(v_val), hn::Set(du32, 0x1F));
    auto v_mant = hn::And(v_val, hn::Set(du32, 0x03));

    // Shift sign to bit 31
    auto v_sign_32 = hn::ShiftLeft<24>(v_sign);

    // Normal path (1 <= exp <= 30)
    // New Exp = Old Exp - 15 + 127 = Old Exp + 112
    auto v_exp_norm = hn::Add(v_exp, hn::Set(du32, 112));
    auto v_mant_norm = hn::ShiftLeft<21>(v_mant);
    auto v_bits_norm = hn::Or(v_sign_32, hn::Or(hn::ShiftLeft<23>(v_exp_norm), v_mant_norm));
    auto v_f_norm = hn::BitCast(d, v_bits_norm);

    // Subnormal path (exp == 0)
    // val = mant * 2^-16 * (-1)^S.
    // mant is small integer (0..3). Convert to float.
    // 2^-16 = 1.5258789e-5
    auto v_mant_f = hn::ConvertTo(d, hn::BitCast(hn::Rebind<int32_t, DIdx>(), v_mant));
    auto v_f_sub = hn::Mul(v_mant_f, hn::Set(d, 1.5258789e-5f));
    // Apply sign to subnormal result
    auto v_f_sub_signed = hn::BitCast(d, hn::Or(v_sign_32, hn::BitCast(du32, v_f_sub)));

    // Inf/NaN path (exp == 31)
    // Exp = 255 (0xFF), Mant = Mant << 21
    auto v_bits_nan = hn::Or(v_sign_32, hn::Or(hn::Set(du32, 0x7F800000), v_mant_norm));
    auto v_f_nan = hn::BitCast(d, v_bits_nan);

    // Select
    auto mask_sub = hn::RebindMask(d, hn::Eq(v_exp, hn::Zero(du32)));
    auto mask_inf = hn::RebindMask(d, hn::Eq(v_exp, hn::Set(du32, 31)));

    auto v_res = hn::IfThenElse(mask_sub, v_f_sub_signed, v_f_norm);
    v_res = hn::IfThenElse(mask_inf, v_f_nan, v_res);

    return v_res;
}

// Bitwise conversion for E4M3FN
// S EEEE MMM (Bias 7) -> S EEEEEEEE MMMMM... (Bias 127)
template <class D, class DIdx> HWY_INLINE hn::Vec<D> ConvertE4M3FN(D d, DIdx du32, hn::Vec<DIdx> v_val) {
    auto v_sign = hn::And(v_val, hn::Set(du32, 0x80));
    auto v_exp = hn::And(hn::ShiftRight<3>(v_val), hn::Set(du32, 0x0F));
    auto v_mant = hn::And(v_val, hn::Set(du32, 0x07));

    auto v_sign_32 = hn::ShiftLeft<24>(v_sign);

    // Normal path (Others)
    // New Exp = Old Exp - 7 + 127 = Old Exp + 120
    auto v_exp_norm = hn::Add(v_exp, hn::Set(du32, 120));
    auto v_mant_norm = hn::ShiftLeft<20>(v_mant);
    auto v_bits_norm = hn::Or(v_sign_32, hn::Or(hn::ShiftLeft<23>(v_exp_norm), v_mant_norm));
    auto v_f_norm = hn::BitCast(d, v_bits_norm);

    // Subnormal path (exp == 0)
    // val = mant * 2^-9 * (-1)^S
    // 2^-9 = 0.001953125
    auto v_mant_f = hn::ConvertTo(d, hn::BitCast(hn::Rebind<int32_t, DIdx>(), v_mant));
    auto v_f_sub = hn::Mul(v_mant_f, hn::Set(d, 0.001953125f));
    auto v_f_sub_signed = hn::BitCast(d, hn::Or(v_sign_32, hn::BitCast(du32, v_f_sub)));

    // NaN path (exp == 15, mant == 7) -> 0x7F or 0xFF
    // Standard map to NaN e.g. 0x7F800001
    auto v_bits_nan = hn::Or(v_sign_32, hn::Set(du32, 0x7F800001));
    auto v_f_nan = hn::BitCast(d, v_bits_nan);

    // Select
    auto mask_sub = hn::RebindMask(d, hn::Eq(v_exp, hn::Zero(du32)));
    // NaN if exp=15 and mant=7
    auto mask_nan_u32 = hn::And(hn::Eq(v_exp, hn::Set(du32, 15)), hn::Eq(v_mant, hn::Set(du32, 7)));
    auto mask_nan = hn::RebindMask(d, mask_nan_u32);

    auto v_res = hn::IfThenElse(mask_sub, v_f_sub_signed, v_f_norm);
    v_res = hn::IfThenElse(mask_nan, v_f_nan, v_res);

    return v_res;
}

template <bool IsE5M2> void ConvertFP8ImplT(const uint8_t* HWY_RESTRICT input, float* HWY_RESTRICT output, int64_t n) {
    const hn::ScalableTag<float> d;
    const hn::ScalableTag<uint32_t> du32;
    const size_t lanes = hn::Lanes(d);
    const float* lut = nullptr;
    if constexpr (IsE5M2) {
        lut = GetFP8E5M2LUT_Hwy();
    } else {
        lut = GetFP8E4M3FNLUT_Hwy();
    }

    int64_t i = 0;
    for (; i + static_cast<int64_t>(lanes) <= n; i += static_cast<int64_t>(lanes)) {
        auto v_val = LoadAndExpand(d, du32, input + i, lanes);

        hn::Vec<hn::ScalableTag<float>> v_res;
        if constexpr (IsE5M2) {
            v_res = ConvertE5M2(d, du32, v_val);
        } else {
            v_res = ConvertE4M3FN(d, du32, v_val);
        }

        hn::StoreU(v_res, d, output + i);
    }

    // Tail
    for (; i < n; ++i) {
        output[i] = lut[input[i]];
    }
}

void ConvertFP8E5M2ToFP32Impl(const uint8_t* input, float* output, int64_t n) {
    ConvertFP8ImplT<true>(input, output, n);
}

void ConvertFP8E4M3FNToFP32Impl(const uint8_t* input, float* output, int64_t n) {
    ConvertFP8ImplT<false>(input, output, n);
}

// Gemv Implementation
template <bool IsE5M2>
void GemvFP8ImplT(const int M, const int N, const float alpha, const void* A_fp8, const float* x_fp32, const float beta,
                  float* y_fp32) {
    const uint8_t* A = static_cast<const uint8_t*>(A_fp8);
    const float* lut = nullptr;
    if constexpr (IsE5M2) {
        lut = GetFP8E5M2LUT_Hwy();
    } else {
        lut = GetFP8E4M3FNLUT_Hwy();
    }

    const hn::ScalableTag<float> d;
    const hn::ScalableTag<uint32_t> du32;
    const size_t lanes = hn::Lanes(d);

    for (int m = 0; m < M; ++m) {
        auto v_row_sum = hn::Zero(d);
        const uint8_t* row_ptr = A + static_cast<int64_t>(m) * N;

        int n = 0;
        for (; n <= N - static_cast<int>(lanes); n += static_cast<int>(lanes)) {
            // Load x
            auto v_x = hn::LoadU(d, x_fp32 + n);

            // Load and convert FP8 weights
            auto v_val = LoadAndExpand(d, du32, row_ptr + n, lanes);

            hn::Vec<hn::ScalableTag<float>> v_w;
            if constexpr (IsE5M2) {
                v_w = ConvertE5M2(d, du32, v_val);
            } else {
                v_w = ConvertE4M3FN(d, du32, v_val);
            }

            v_row_sum = hn::MulAdd(v_w, v_x, v_row_sum);
        }

        float row_sum = hn::ReduceSum(d, v_row_sum);

        // Tail
        for (; n < N; ++n) {
            const float w_val = lut[row_ptr[n]];
            row_sum += w_val * x_fp32[n];
        }

        if (beta == 0.0f) {
            y_fp32[m] = alpha * row_sum;
        } else {
            y_fp32[m] = alpha * row_sum + beta * y_fp32[m];
        }
    }
}

void GemvFP8E5M2Impl(const int M, const int N, const float alpha, const void* A_fp8, const float* x_fp32,
                     const float beta, float* y_fp32) {
    GemvFP8ImplT<true>(M, N, alpha, A_fp8, x_fp32, beta, y_fp32);
}

void GemvFP8E4M3FNImpl(const int M, const int N, const float alpha, const void* A_fp8, const float* x_fp32,
                       const float beta, float* y_fp32) {
    GemvFP8ImplT<false>(M, N, alpha, A_fp8, x_fp32, beta, y_fp32);
}


}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

namespace {
// 64-byte alignment to match cache line size
alignas(64) float g_fp8_e5m2_lut[256];
alignas(64) float g_fp8_e4m3fn_lut[256];
}  // namespace

DENSECORE_API const float* GetFP8E5M2LUT_Hwy() {
    return g_fp8_e5m2_lut;
}

DENSECORE_API const float* GetFP8E4M3FNLUT_Hwy() {
    return g_fp8_e4m3fn_lut;
}

// Helper to convert single uint8 e5m2 representation to float
static float e5m2_to_float(uint8_t v) {
    // S EEEEE MM
    // Bias 15
    uint8_t sign = (v >> 7) & 0x1;
    uint8_t exp = (v >> 2) & 0x1F;
    uint8_t mant = v & 0x3;

    uint32_t val_bits = 0;
    if (exp == 0) {
        // Subnormal or Zero
        if (mant == 0) {
            val_bits = (sign << 31);
        } else {
            float f = (mant)*std::pow(2.0f, -16.0f);
            if (sign) f = -f;
            return f;
        }
    } else if (exp == 31) {
        // Inf or NaN
        val_bits = (sign << 31) | 0x7F800000;
        if (mant != 0) {
            val_bits |= (mant << 21) | 1;
        }
    } else {
        uint32_t new_exp = exp + 112;
        uint32_t new_mant = mant << 21;
        val_bits = (sign << 31) | (new_exp << 23) | new_mant;
    }
    float f;
    std::memcpy(&f, &val_bits, sizeof(float));
    return f;
}

// Helper to convert single uint8 e4m3fn representation to float
static float e4m3fn_to_float(uint8_t v) {
    uint8_t sign = (v >> 7) & 0x1;
    uint8_t exp = (v >> 3) & 0x0F;
    uint8_t mant = v & 0x07;

    if (exp == 15 && mant == 7) {
        uint32_t val_bits = (sign << 31) | 0x7F800001;
        float f;
        std::memcpy(&f, &val_bits, sizeof(float));
        return f;
    }

    if (exp == 0) {
        if (mant == 0) {
            uint32_t val_bits = (sign << 31);
            float f;
            std::memcpy(&f, &val_bits, sizeof(float));
            return f;
        } else {
            float f = (mant)*std::pow(2.0f, -9.0f);
            if (sign) f = -f;
            return f;
        }
    } else {
        uint32_t new_exp = exp + 120;
        uint32_t new_mant = mant << 20;
        uint32_t val_bits = (sign << 31) | (new_exp << 23) | new_mant;
        float f;
        std::memcpy(&f, &val_bits, sizeof(float));
        return f;
    }
}

DENSECORE_API void InitFP8LUTs_Hwy() {
    for (int i = 0; i < 256; ++i) {
        g_fp8_e5m2_lut[i] = e5m2_to_float(static_cast<uint8_t>(i));
        g_fp8_e4m3fn_lut[i] = e4m3fn_to_float(static_cast<uint8_t>(i));
    }
}

HWY_EXPORT(ConvertFP8E5M2ToFP32Impl);
HWY_EXPORT(ConvertFP8E4M3FNToFP32Impl);
HWY_EXPORT(GemvFP8E5M2Impl);
HWY_EXPORT(GemvFP8E4M3FNImpl);


DENSECORE_API void ConvertFP8E5M2ToFP32_Hwy(const uint8_t* input, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(ConvertFP8E5M2ToFP32Impl)(input, output, n);
}

DENSECORE_API void ConvertFP8E4M3FNToFP32_Hwy(const uint8_t* input, float* output, int64_t n) {
    HWY_DYNAMIC_DISPATCH(ConvertFP8E4M3FNToFP32Impl)(input, output, n);
}

DENSECORE_API void Gemv_FP8_E5M2_Hwy(const int M, const int N, const float alpha, const void* A_fp8,
                                     const float* x_fp32, const float beta, float* y_fp32) {
    HWY_DYNAMIC_DISPATCH(GemvFP8E5M2Impl)(M, N, alpha, A_fp8, x_fp32, beta, y_fp32);
}

DENSECORE_API void Gemv_FP8_E4M3FN_Hwy(const int M, const int N, const float alpha, const void* A_fp8,
                                       const float* x_fp32, const float beta, float* y_fp32) {
    HWY_DYNAMIC_DISPATCH(GemvFP8E4M3FNImpl)(M, N, alpha, A_fp8, x_fp32, beta, y_fp32);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
