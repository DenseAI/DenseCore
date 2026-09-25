#pragma once
// Experimental C4A M4 kernels adapted from pinned ggml Arm nrc=2 kernels.
// Private ABI layouts avoid importing global quant block types into runtime.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "ggml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_sve.h>
#endif
namespace densecore_arm_m4 {
struct BlockQ8_0 {
    ggml_fp16_t d;
    int8_t qs[32];
};
struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    ggml_fp16_t d;
};
struct BlockQ8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(BlockQ8_0) == 34 && offsetof(BlockQ8_0, qs) == 2);
static_assert(sizeof(BlockQ6_K) == 210 && offsetof(BlockQ6_K, d) == 208);
static_assert(sizeof(BlockQ6_K) == 210 && offsetof(BlockQ6_K, scales) == 192);
static_assert(sizeof(BlockQ8_K) == 292 && offsetof(BlockQ8_K, bsums) == 260);
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
static inline float ArmM4Scale(ggml_fp16_t bits) {
    __fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

// bx/by are byte strides; bs is the output stride in floats. Writes exactly
// s[m*bs+r], m=0..3, r=0..1. bx=0 supports a repeated final weight row.
static inline void ArmQ8M4Candidate(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by) {
    assert(n >= 0 && n % 32 == 0);
    const auto* x0 = static_cast<const BlockQ8_0*>(vx);
    const auto* x1 = reinterpret_cast<const BlockQ8_0*>(static_cast<const uint8_t*>(vx) + bx);
    const BlockQ8_0* y[4];
    for (int m = 0; m < 4; ++m) y[m] = reinterpret_cast<const BlockQ8_0*>(static_cast<const uint8_t*>(vy) + m * by);
    float32x4_t sums[2] = {vdupq_n_f32(0), vdupq_n_f32(0)};
    for (int b = 0; b < n / 32; ++b) {
        const int8x16_t xl0 = vld1q_s8(x0[b].qs), xh0 = vld1q_s8(x0[b].qs + 16);
        const int8x16_t xl1 = vld1q_s8(x1[b].qs), xh1 = vld1q_s8(x1[b].qs + 16);
        const int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(xl0), vreinterpretq_s64_s8(xl1)));
        const int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(xl0), vreinterpretq_s64_s8(xl1)));
        const int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(xh0), vreinterpretq_s64_s8(xh1)));
        const int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(xh0), vreinterpretq_s64_s8(xh1)));
        const float dx0 = ArmM4Scale(x0[b].d), dx1 = ArmM4Scale(x1[b].d);
        for (int p = 0; p < 2; ++p) {
            const auto& a = y[2 * p][b];
            const auto& c = y[2 * p + 1][b];
            const int8x16_t yl0 = vld1q_s8(a.qs), yh0 = vld1q_s8(a.qs + 16);
            const int8x16_t yl1 = vld1q_s8(c.qs), yh1 = vld1q_s8(c.qs + 16);
            const int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(yl0), vreinterpretq_s64_s8(yl1)));
            const int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(yl0), vreinterpretq_s64_s8(yl1)));
            const int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(yh0), vreinterpretq_s64_s8(yh1)));
            const int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(yh0), vreinterpretq_s64_s8(yh1)));
            int32x4_t dot = vmmlaq_s32(vdupq_n_s32(0), l0, r0);
            dot = vmmlaq_s32(dot, l1, r1);
            dot = vmmlaq_s32(dot, l2, r2);
            dot = vmmlaq_s32(dot, l3, r3);
            const float dy0 = ArmM4Scale(a.d), dy1 = ArmM4Scale(c.d);
            const float scale[4] = {dx0 * dy0, dx0 * dy1, dx1 * dy0, dx1 * dy1};
            sums[p] = vmlaq_f32(sums[p], vcvtq_f32_s32(dot), vld1q_f32(scale));
        }
    }
    for (int p = 0; p < 2; ++p) {
        const float32x4_t out = vzip1q_f32(sums[p], vextq_f32(sums[p], sums[p], 2));
        vst1_f32(s + (2 * p) * bs, vget_low_f32(out));
        vst1_f32(s + (2 * p + 1) * bs, vget_high_f32(out));
    }
}

#endif
inline bool ArmQ6M4CandidateAvailable() {
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    return svcntb() == 16;
#else
    return false;
#endif
}

// Output is input-major: out[input * output_stride + weight_row]. Stride in floats.
// Weight/input strides are bytes. Exactly two weight rows and four input rows.
// A zero weight stride repeats one row for the production odd-row tail.
inline bool ArmQ6M4Candidate(int n, const void* weights, size_t weight_stride, const void* inputs, size_t input_stride,
                             float* out, size_t output_stride) {
    if (!ArmQ6M4CandidateAvailable() || n <= 0 || n % 256 || !weights || !inputs || !out ||
        (weight_stride != 0 && weight_stride < size_t(n / 256) * sizeof(BlockQ6_K)) ||
        weight_stride % alignof(BlockQ6_K) || input_stride < size_t(n / 256) * sizeof(BlockQ8_K) ||
        input_stride % alignof(BlockQ8_K) || output_stride < 2)
        return false;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    const auto* vx0 = static_cast<const BlockQ6_K*>(weights);
    const auto* vx1 = reinterpret_cast<const BlockQ6_K*>(static_cast<const uint8_t*>(weights) + weight_stride);
    const auto* vy0 = static_cast<const BlockQ8_K*>(inputs);
    const auto* vy1 = reinterpret_cast<const BlockQ8_K*>(static_cast<const uint8_t*>(inputs) + input_stride);
    const auto* vy2 = reinterpret_cast<const BlockQ8_K*>(static_cast<const uint8_t*>(inputs) + 2 * input_stride);
    const auto* vy3 = reinterpret_cast<const BlockQ8_K*>(static_cast<const uint8_t*>(inputs) + 3 * input_stride);
    const int nb = n / 256;
    const svbool_t pg128_all = svptrue_b8();
    const svbool_t pg32_2 = svptrue_pat_b32(SV_VL2);
    svfloat32_t sum = svdup_n_f32(0), sum_b = svdup_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* ql0 = vx0[i].ql;
        const uint8_t* qh0 = vx0[i].qh;
        const uint8_t* ql1 = vx1[i].ql;
        const uint8_t* qh1 = vx1[i].qh;
        const int8_t* q80 = vy0[i].qs;
        const int8_t* q81 = vy1[i].qs;
        const int8_t* q82 = vy2[i].qs;
        const int8_t* q83 = vy3[i].qs;

        const int8_t* scale0 = vx0[i].scales;
        const int8_t* scale1 = vx1[i].scales;

        svfloat32_t vy_d = svuzp1_f32(svdup_n_f32(vy0[i].d), svdup_n_f32(vy1[i].d));
        svfloat32_t vx_d = svzip1_f32(svdup_n_f32(ArmM4Scale(vx0[i].d)), svdup_n_f32(ArmM4Scale(vx1[i].d)));
        svfloat32_t svsuper_block_scales = svmul_f32_x(pg128_all, vy_d, vx_d);
        // process q8sum summation 128 bit route
        const svint16_t q8sums_01 = svld1_s16(pg128_all, vy0[i].bsums);
        const svint16_t q8sums_02 = svld1_s16(pg128_all, vy0[i].bsums + 8);
        const svint16_t q8sums_11 = svld1_s16(pg128_all, vy1[i].bsums);
        const svint16_t q8sums_12 = svld1_s16(pg128_all, vy1[i].bsums + 8);
        const svint64x2_t q6scales_0_tmp = svld2_s64(pg128_all, (const int64_t*)scale0);
        const svint16_t q6scales_01 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(q6scales_0_tmp, 0)));
        const svint16_t q6scales_02 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(q6scales_0_tmp, 1)));
        const svint64x2_t q6scales_1_tmp = svld2_s64(pg128_all, (const int64_t*)scale1);
        const svint16_t q6scales_11 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(q6scales_1_tmp, 0)));
        const svint16_t q6scales_12 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(q6scales_1_tmp, 1)));
        const svint64_t prod = svdup_n_s64(0);

        svint32_t isum_tmp1 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_01, q6scales_01), q8sums_02, q6scales_02));
        svint32_t isum_tmp2 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_01, q6scales_11), q8sums_02, q6scales_12));
        svint32_t isum_tmp3 = svtrn1_s32(isum_tmp1, isum_tmp2);
        svint32_t isum_tmp4 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_11, q6scales_01), q8sums_12, q6scales_02));
        svint32_t isum_tmp5 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_11, q6scales_11), q8sums_12, q6scales_12));
        svint32_t isum_tmp6 = svtrn1_s32(isum_tmp4, isum_tmp5);
        svint32_t isum_tmp7 =
            svreinterpret_s32_s64(svtrn2_s64(svreinterpret_s64_s32(isum_tmp3), svreinterpret_s64_s32(isum_tmp6)));
        svint32_t isum_tmp8 =
            svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(isum_tmp3), svreinterpret_s64_s32(isum_tmp6)));
        svint32_t svisum_mins = svadd_s32_x(pg128_all, isum_tmp7, isum_tmp8);

        svfloat32_t vy_d_b = svuzp1_f32(svdup_n_f32(vy2[i].d), svdup_n_f32(vy3[i].d));
        svfloat32_t svsuper_block_scales_b = svmul_f32_x(pg128_all, vy_d_b, vx_d);
        // process q8sum summation 128 bit route
        const svint16_t q8sums_01_b = svld1_s16(pg128_all, vy2[i].bsums);
        const svint16_t q8sums_02_b = svld1_s16(pg128_all, vy2[i].bsums + 8);
        const svint16_t q8sums_11_b = svld1_s16(pg128_all, vy3[i].bsums);
        const svint16_t q8sums_12_b = svld1_s16(pg128_all, vy3[i].bsums + 8);

        svint32_t isum_tmp1_b =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_01_b, q6scales_01), q8sums_02_b, q6scales_02));
        svint32_t isum_tmp2_b =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_01_b, q6scales_11), q8sums_02_b, q6scales_12));
        svint32_t isum_tmp3_b = svtrn1_s32(isum_tmp1_b, isum_tmp2_b);
        svint32_t isum_tmp4_b =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_11_b, q6scales_01), q8sums_12_b, q6scales_02));
        svint32_t isum_tmp5_b =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(prod, q8sums_11_b, q6scales_11), q8sums_12_b, q6scales_12));
        svint32_t isum_tmp6_b = svtrn1_s32(isum_tmp4_b, isum_tmp5_b);
        svint32_t isum_tmp7_b =
            svreinterpret_s32_s64(svtrn2_s64(svreinterpret_s64_s32(isum_tmp3_b), svreinterpret_s64_s32(isum_tmp6_b)));
        svint32_t isum_tmp8_b =
            svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(isum_tmp3_b), svreinterpret_s64_s32(isum_tmp6_b)));
        svint32_t svisum_mins_b = svadd_s32_x(pg128_all, isum_tmp7_b, isum_tmp8_b);

        // process mmla
        svint8_t l0, l1, r0, r1;
        svint32_t isum_tmp = svdup_n_s32(0);
        svint32_t isum_b = svdup_n_s32(0);
        for (int j = 0; j < 256 / 128; ++j) {
            for (int k = 0; k < 8; ++k) {
                svuint8_t qhbits_0 = svld1_u8(pg128_all, qh0 + 16 * (k % 2));
                svuint8_t qhbits_1 = svld1_u8(pg128_all, qh1 + 16 * (k % 2));
                svuint8_t q6bits_0 = svld1_u8(pg128_all, ql0 + 16 * (k % 4));
                svuint8_t q6bits_1 = svld1_u8(pg128_all, ql1 + 16 * (k % 4));
                const int ql_pos = (k / 4) * 4;
                svuint8_t q6bytes_0_lo =
                    (ql_pos < 4) ? svand_n_u8_x(pg128_all, q6bits_0, 0xf) : svlsr_n_u8_x(pg128_all, q6bits_0, 4);
                svuint8_t q6bytes_1_lo =
                    (ql_pos < 4) ? svand_n_u8_x(pg128_all, q6bits_1, 0xf) : svlsr_n_u8_x(pg128_all, q6bits_1, 4);
                const int qh_pos = (k / 2) * 2;
                svuint8_t q6bytes_0_hi = svand_n_u8_x(pg128_all, qhbits_0, 0x3 << qh_pos);
                svuint8_t q6bytes_1_hi = svand_n_u8_x(pg128_all, qhbits_1, 0x3 << qh_pos);
                svint8_t q6bytes_0, q6bytes_1;
                if (qh_pos <= 4) {
                    q6bytes_0 =
                        svreinterpret_s8_u8(svmla_n_u8_x(pg128_all, q6bytes_0_lo, q6bytes_0_hi, 1 << (4 - qh_pos)));
                    q6bytes_1 =
                        svreinterpret_s8_u8(svmla_n_u8_x(pg128_all, q6bytes_1_lo, q6bytes_1_hi, 1 << (4 - qh_pos)));
                } else {
                    q6bytes_0 = svreinterpret_s8_u8(
                        svorr_u8_x(pg128_all, q6bytes_0_lo, svlsr_n_u8_x(pg128_all, q6bytes_0_hi, (qh_pos - 4))));
                    q6bytes_1 = svreinterpret_s8_u8(
                        svorr_u8_x(pg128_all, q6bytes_1_lo, svlsr_n_u8_x(pg128_all, q6bytes_1_hi, (qh_pos - 4))));
                }
                svint8_t q8bytes_0 = svld1_s8(pg128_all, q80 + 16 * (k % 8));
                svint8_t q8bytes_1 = svld1_s8(pg128_all, q81 + 16 * (k % 8));
                l0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(q6bytes_0), svreinterpret_s64_s8(q6bytes_1)));
                l1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(q6bytes_0), svreinterpret_s64_s8(q6bytes_1)));
                r0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(q8bytes_0), svreinterpret_s64_s8(q8bytes_1)));
                r1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(q8bytes_0), svreinterpret_s64_s8(q8bytes_1)));
                svint32_t svscale = svzip1_s32(svdup_n_s32(scale0[k]), svdup_n_s32(scale1[k]));
                isum_tmp =
                    svmla_s32_x(pg128_all, isum_tmp, svmmla_s32(svmmla_s32(svdup_n_s32(0), r0, l0), r1, l1), svscale);
                const svint8_t q8bytes_2 = svld1_s8(pg128_all, q82 + 16 * k);
                const svint8_t q8bytes_3 = svld1_s8(pg128_all, q83 + 16 * k);
                r0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(q8bytes_2), svreinterpret_s64_s8(q8bytes_3)));
                r1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(q8bytes_2), svreinterpret_s64_s8(q8bytes_3)));
                isum_b =
                    svmla_s32_x(pg128_all, isum_b, svmmla_s32(svmmla_s32(svdup_n_s32(0), r0, l0), r1, l1), svscale);
            }
            qh0 += 32;
            qh1 += 32;
            ql0 += 64;
            ql1 += 64;
            q80 += 128;
            q81 += 128;
            q82 += 128;
            q83 += 128;
            scale0 += 8;
            scale1 += 8;
        }
        sum = svmla_f32_x(pg128_all, sum,
                          svcvt_f32_x(pg128_all, svmla_s32_x(pg128_all, isum_tmp, svisum_mins, svdup_n_s32(-32))),
                          svsuper_block_scales);
        sum_b = svmla_f32_x(pg128_all, sum_b,
                            svcvt_f32_x(pg128_all, svmla_s32_x(pg128_all, isum_b, svisum_mins_b, svdup_n_s32(-32))),
                            svsuper_block_scales_b);
    }
    svst1_f32(pg32_2, out, sum);
    svst1_f32(pg32_2, out + output_stride, svreinterpret_f32_u8(svext_u8(svreinterpret_u8_f32(sum), svdup_n_u8(0), 8)));
    svst1_f32(pg32_2, out + 2 * output_stride, sum_b);
    svst1_f32(pg32_2, out + 3 * output_stride,
              svreinterpret_f32_u8(svext_u8(svreinterpret_u8_f32(sum_b), svdup_n_u8(0), 8)));
    return true;
#else
    return false;
#endif
}

// Reject unsupported calls without touching the output.
inline bool Compute(ggml_type type, int n, const void* weights, size_t bx, const void* inputs, size_t by, float* out,
                    size_t bs) {
    if (!weights || !inputs || !out || bs < 2 || n <= 0) return false;
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (type == GGML_TYPE_Q8_0) {
        if (n % 32 || (bx && bx < size_t(n / 32) * sizeof(BlockQ8_0)) || bx % alignof(BlockQ8_0) ||
            by < size_t(n / 32) * sizeof(BlockQ8_0) || by % alignof(BlockQ8_0))
            return false;
        ArmQ8M4Candidate(n, out, bs, weights, bx, inputs, by);
        return true;
    }
#endif
    if (type == GGML_TYPE_Q6_K) return ArmQ6M4Candidate(n, weights, bx, inputs, by, out, bs);
    return false;
}
}  // namespace densecore_arm_m4
