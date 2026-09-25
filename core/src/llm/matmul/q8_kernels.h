#pragma once
#include "densecore/memory/kv_cache.h"
#include <algorithm>
#include <cstring>
#include <ggml-cpu.h>
// Quantized GEMV cache, probes, and admission helpers used by matmul callbacks.
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__SSE4_1__)
#include <immintrin.h>
#endif
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define DENSECORE_Q8_4X8_NEON_DOTPROD 1
#endif
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>
#define DENSECORE_Q8_4X8_NEON_I8MM 1
#endif

static inline int DenseCoreQ8_0Dot8I8I8(const int8_t* weight, const int8_t* input) {
    if (!weight || !input) {
        return 0;
    }
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__SSE4_1__)
    const __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weight));
    const __m128i x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    const __m128i w16 = _mm_cvtepi8_epi16(w8);
    const __m128i x16 = _mm_cvtepi8_epi16(x8);
    const __m128i prod32 = _mm_madd_epi16(w16, x16);
    const __m128i sum64 = _mm_add_epi32(prod32, _mm_shuffle_epi32(prod32, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128i sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum32);
#else
    int acc = 0;
    for (int i = 0; i < 8; ++i) {
        acc += static_cast<int>(weight[i]) * static_cast<int>(input[i]);
    }
    return acc;
#endif
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
static inline __m256i DenseCoreQ8_0LoadPacked4x8RowAVX2(const int8_t* qs, int row) {
    const int8_t* row_qs = qs + row * 8;
    const __m128i q0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_qs));
    const __m128i q1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_qs + 32));
    const __m128i q2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_qs + 64));
    const __m128i q3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_qs + 96));
    const __m128i lo = _mm_unpacklo_epi64(q0, q1);
    const __m128i hi = _mm_unpacklo_epi64(q2, q3);
    return _mm256_set_m128i(hi, lo);
}

static inline __m256i DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(__m256i acc, __m256i weight, __m256i input) {
    const __m256i abs_weight = _mm256_sign_epi8(weight, weight);
    const __m256i signed_input = _mm256_sign_epi8(input, weight);
    const __m256i pair_sum_i16 = _mm256_maddubs_epi16(abs_weight, signed_input);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pair_sum_i16, ones));
}

static inline int DenseCoreQ8_0HsumI32x8AVX2(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

static inline int DenseCoreQ8_0DotPacked4x8RowAVX2(const int8_t* qs, int row, const int8_t* input) {
    const __m256i weight = DenseCoreQ8_0LoadPacked4x8RowAVX2(qs, row);
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const __m256i acc = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(_mm256_setzero_si256(), weight, x);
    return DenseCoreQ8_0HsumI32x8AVX2(acc);
}

static inline __m128i DenseCoreQ8_0DotPacked4x8RowsAVX2(const int8_t* qs, const int8_t* input) {
    __m256i acc = _mm256_setzero_si256();
    for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
        const __m256i weights =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + static_cast<size_t>(chunk) * 4 * 8));
        const __m128i input8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + static_cast<size_t>(chunk) * 8));
        const __m256i inputs = _mm256_broadcastq_epi64(input8);
        acc = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(acc, weights, inputs);
    }

    // Each 64-bit lane contains the two i32 partials for one packed row.
    const __m256i row_pairs = _mm256_hadd_epi32(acc, _mm256_setzero_si256());
    const __m128i lower = _mm256_castsi256_si128(row_pairs);
    const __m128i upper = _mm256_extracti128_si256(row_pairs, 1);
    return _mm_setr_epi32(_mm_extract_epi32(lower, 0), _mm_extract_epi32(lower, 1), _mm_extract_epi32(upper, 0),
                          _mm_extract_epi32(upper, 1));
}

static inline void DenseCoreQ8_0DotPacked4x8TileAVX2(const int8_t* weights, const int8_t* inputs,
                                                     int32_t (&dots)[4][4]) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    // A packed chunk contains eight values from each of four rows. Broadcast
    // each input row across the four weight rows without gathering either row.
    // Q8_0 from_float bounds activations to [-127, 127], as required by the
    // shared sign/maddubs helper; weight bytes may also contain -128.
    for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + chunk * 32));
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + chunk * 32));
        acc0 = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(acc0, w, _mm256_permute4x64_epi64(x, 0x00));
        acc1 = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(acc1, w, _mm256_permute4x64_epi64(x, 0x55));
        acc2 = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(acc2, w, _mm256_permute4x64_epi64(x, 0xaa));
        acc3 = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(acc3, w, _mm256_permute4x64_epi64(x, 0xff));
    }
    const auto store_rows = [](int32_t* out, __m256i acc) {
        const __m256i pairs = _mm256_hadd_epi32(acc, acc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                         _mm_unpacklo_epi64(_mm256_castsi256_si128(pairs), _mm256_extracti128_si256(pairs, 1)));
    };
    store_rows(dots[0], acc0);
    store_rows(dots[1], acc1);
    store_rows(dots[2], acc2);
    store_rows(dots[3], acc3);
}
#endif

#if defined(DENSECORE_Q8_4X8_NEON_DOTPROD)
// ARM SVE2/Neoverse (C4A) had no SIMD path here: the q8_0 4x8 GEMM/GEMV fell to the
// scalar #else below, making the Qwen hybrid-SSM ssm_qkv_gate projection (the #1
// prefill cost) ~2x slower than llama.cpp's i8mm kernels. These dotprod helpers
// vectorize the int8 dot. Integer dot products are EXACT, so the result is
// bit-identical to the scalar path (and to AVX2) — the parity-oracle tests
// (RunDenseCoreQ8RepackedGemvForTest / RunGemma4NativeQ8PrefillTrueGemmForTest)
// must confirm output_matches_vecdot_oracle on ARM before trusting this.
//
// 4x8 packed layout per block: a row's 32 int8 live in four 8-wide chunks strided
// by 32 bytes (qs + chunk*32 + row*8). chunk c of the weight row dots input chunk c.
static inline int DenseCoreQ8_0DotPacked4x8RowDotprodNeon(const int8_t* qs, int row, const int8_t* input) {
    const int8_t* w = qs + row * 8;
    const int8x16_t wA = vcombine_s8(vld1_s8(w), vld1_s8(w + 32));       // weight chunks 0,1
    const int8x16_t wB = vcombine_s8(vld1_s8(w + 64), vld1_s8(w + 96));  // weight chunks 2,3
    const int8x16_t xA = vld1q_s8(input);                                // input chunks 0,1
    const int8x16_t xB = vld1q_s8(input + 16);                           // input chunks 2,3
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, wA, xA);
    acc = vdotq_s32(acc, wB, xB);
    return vaddvq_s32(acc);
}

// Both operands in 4x8 packed (strided) layout — used by the M>1 true GEMM tile.
static inline int DenseCoreQ8_0DotPacked4x8RowsDotprodNeon(const int8_t* lhs_qs, int lhs_row, const int8_t* rhs_qs,
                                                           int rhs_row) {
    const int8_t* l = lhs_qs + lhs_row * 8;
    const int8_t* r = rhs_qs + rhs_row * 8;
    const int8x16_t lA = vcombine_s8(vld1_s8(l), vld1_s8(l + 32));
    const int8x16_t lB = vcombine_s8(vld1_s8(l + 64), vld1_s8(l + 96));
    const int8x16_t rA = vcombine_s8(vld1_s8(r), vld1_s8(r + 32));
    const int8x16_t rB = vcombine_s8(vld1_s8(r + 64), vld1_s8(r + 96));
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, lA, rA);
    acc = vdotq_s32(acc, lB, rB);
    return vaddvq_s32(acc);
}
#endif

static inline float DenseCoreFp16ToFp32Fast(ggml_fp16_t value) {
    float out_value = 0.0f;
    ggml_cpu_fp16_to_fp32(&value, &out_value, 1);
    return out_value;
}

template <typename Visitor>
static void DenseCoreForEachQ8_0_4x8Q8_0DotGeneric(int n, const void* packed_weight, const void* q8_input, int nc,
                                                   int row_offset, Visitor&& visit) {
    if (!packed_weight || !q8_input || n <= 0 || (n % QK8_0) != 0 || (nc % 4) != 0) {
        return;
    }
    const int nb = n / QK8_0;
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const auto* packed_base = static_cast<const uint8_t*>(packed_weight);
    const auto* input_blocks = static_cast<const block_q8_0*>(q8_input);
    for (int group = 0; group < nc / 4; ++group) {
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
#else
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
        const auto* group_base =
            packed_base + static_cast<size_t>(group) * static_cast<size_t>(nb) * packed_block_bytes;
        for (int b = 0; b < nb; ++b) {
            const auto* block_base = group_base + static_cast<size_t>(b) * packed_block_bytes;
            const auto* scales = reinterpret_cast<const ggml_fp16_t*>(block_base);
            const auto* qs = reinterpret_cast<const int8_t*>(block_base + 4 * sizeof(ggml_fp16_t));
            const block_q8_0& x = input_blocks[b];
            const float input_scale = DenseCoreFp16ToFp32Fast(x.d);
#if !defined(DENSECORE_Q8_4X8_NEON_I8MM)
            float row_scale[4];
            ggml_cpu_fp16_to_fp32(scales, row_scale, 4);
#endif
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
            int32x4_t acc01 = vdupq_n_s32(0);
            int32x4_t acc23 = vdupq_n_s32(0);
            for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                const int8x8_t input_chunk = vld1_s8(x.qs + chunk * 8);
                const int8x16_t input_pair = vcombine_s8(input_chunk, input_chunk);
                acc01 = vmmlaq_s32(acc01, input_pair, vld1q_s8(qs + chunk * 32));
                acc23 = vmmlaq_s32(acc23, input_pair, vld1q_s8(qs + chunk * 32 + 16));
            }
            const int32x4_t dots = vcombine_s32(vget_low_s32(acc01), vget_low_s32(acc23));
            const float32x4_t row_scale = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(scales)));
            sum_vec = vfmaq_f32(sum_vec, vcvtq_f32_s32(dots), vmulq_n_f32(row_scale, input_scale));
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
            alignas(16) int32_t row_dots[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(row_dots), DenseCoreQ8_0DotPacked4x8RowsAVX2(qs, x.qs));
            for (int row = 0; row < 4; ++row) {
                sum[row] += static_cast<float>(row_dots[row]) * row_scale[row] * input_scale;
            }
#elif defined(DENSECORE_Q8_4X8_NEON_DOTPROD)
            sum[0] +=
                static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 0, x.qs)) * row_scale[0] * input_scale;
            sum[1] +=
                static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 1, x.qs)) * row_scale[1] * input_scale;
            sum[2] +=
                static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 2, x.qs)) * row_scale[2] * input_scale;
            sum[3] +=
                static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 3, x.qs)) * row_scale[3] * input_scale;
#else
            for (int row = 0; row < 4; ++row) {
                int acc = 0;
                for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                    acc += DenseCoreQ8_0Dot8I8I8(qs + chunk * 4 * 8 + row * 8, x.qs + chunk * 8);
                }
                sum[row] += static_cast<float>(acc) * row_scale[row] * input_scale;
            }
#endif
        }
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
        float sum[4];
        vst1q_f32(sum, sum_vec);
#endif
        for (int row = 0; row < 4; ++row) {
            visit(row_offset + group * 4 + row, sum[row]);
        }
    }
}

static void DenseCoreGemvQ8_0_4x8Q8_0Generic(int n, float* out, const void* packed_weight, const void* q8_input,
                                             int nc) {
    if (!out) {
        return;
    }
    DenseCoreForEachQ8_0_4x8Q8_0DotGeneric(n, packed_weight, q8_input, nc, 0,
                                           [&](int row, float value) { out[static_cast<size_t>(row)] = value; });
}

static inline int DenseCoreQ8_0DotPacked4x8Rows(const int8_t* lhs_qs, int lhs_row, const int8_t* rhs_qs, int rhs_row) {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
    const __m256i lhs = DenseCoreQ8_0LoadPacked4x8RowAVX2(lhs_qs, lhs_row);
    const __m256i rhs = DenseCoreQ8_0LoadPacked4x8RowAVX2(rhs_qs, rhs_row);
    const __m256i acc = DenseCoreQ8_0MulSumI8PairsAccI32x8AVX2(_mm256_setzero_si256(), lhs, rhs);
    return DenseCoreQ8_0HsumI32x8AVX2(acc);
#elif defined(DENSECORE_Q8_4X8_NEON_DOTPROD)
    return DenseCoreQ8_0DotPacked4x8RowsDotprodNeon(lhs_qs, lhs_row, rhs_qs, rhs_row);
#else
    int sum = 0;
    for (int chunk = 0; chunk < 4; ++chunk) {
        sum += DenseCoreQ8_0Dot8I8I8(lhs_qs + chunk * 4 * 8 + lhs_row * 8, rhs_qs + chunk * 4 * 8 + rhs_row * 8);
    }
    return sum;
#endif
}

static void DenseCoreGemmQ8_0_4x8x4Q8_0Generic(int n, float* out, int64_t out_row_stride, const void* packed_weight,
                                               const void* packed_input, int nc) {
    if (!out || !packed_weight || !packed_input || n <= 0 || (n % QK8_0) != 0 || nc <= 0 || (nc % 4) != 0 ||
        out_row_stride <= 0) {
        return;
    }
    const int nb = n / QK8_0;
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const auto* weight_base = static_cast<const uint8_t*>(packed_weight);
    const auto* input_base = static_cast<const uint8_t*>(packed_input);

    for (int group = 0; group < nc / 4; ++group) {
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
        float32x4_t acc_f32[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
#else
        float sum[4][4] = {};
#endif
        const auto* weight_group =
            weight_base + static_cast<size_t>(group) * static_cast<size_t>(nb) * packed_block_bytes;
        for (int b = 0; b < nb; ++b) {
            const auto* weight_block = weight_group + static_cast<size_t>(b) * packed_block_bytes;
            const auto* input_block = input_base + static_cast<size_t>(b) * packed_block_bytes;
            const auto* weight_scales = reinterpret_cast<const ggml_fp16_t*>(weight_block);
            const auto* input_scales = reinterpret_cast<const ggml_fp16_t*>(input_block);
            const auto* weight_qs = reinterpret_cast<const int8_t*>(weight_block + 4 * sizeof(ggml_fp16_t));
            const auto* input_qs = reinterpret_cast<const int8_t*>(input_block + 4 * sizeof(ggml_fp16_t));
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
            int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
            for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                const int8x16_t x01 = vld1q_s8(input_qs + chunk * 32);
                const int8x16_t x23 = vld1q_s8(input_qs + chunk * 32 + 16);
                const int8x16_t w01 = vld1q_s8(weight_qs + chunk * 32);
                const int8x16_t w23 = vld1q_s8(weight_qs + chunk * 32 + 16);
                acc[0] = vmmlaq_s32(acc[0], x01, w01);
                acc[1] = vmmlaq_s32(acc[1], x01, w23);
                acc[2] = vmmlaq_s32(acc[2], x23, w01);
                acc[3] = vmmlaq_s32(acc[3], x23, w23);
            }
            const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
            const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
            const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
            const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
            const float32x4_t x_scale = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(input_scales)));
            const float32x4_t w_scale = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(weight_scales)));
            acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(w_scale, x_scale, 0));
            acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(w_scale, x_scale, 1));
            acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(w_scale, x_scale, 2));
            acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(w_scale, x_scale, 3));
#else
            float w_scale[4];
            float x_scale[4];
            ggml_cpu_fp16_to_fp32(weight_scales, w_scale, 4);
            ggml_cpu_fp16_to_fp32(input_scales, x_scale, 4);
#if defined(__AVX2__)
            int32_t dots[4][4];
            DenseCoreQ8_0DotPacked4x8TileAVX2(weight_qs, input_qs, dots);
#endif
            for (int m = 0; m < 4; ++m) {
                for (int row = 0; row < 4; ++row) {
#if defined(__AVX2__)
                    const int dot = dots[m][row];
#else
                    const int dot = DenseCoreQ8_0DotPacked4x8Rows(weight_qs, row, input_qs, m);
#endif
                    sum[m][row] += static_cast<float>(dot) * w_scale[row] * x_scale[m];
                }
            }
#endif
        }
        for (int m = 0; m < 4; ++m) {
            float* out_row =
                out + static_cast<size_t>(m) * static_cast<size_t>(out_row_stride) + static_cast<size_t>(group) * 4;
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
            vst1q_f32(out_row, acc_f32[m]);
#else
            out_row[0] = sum[m][0];
            out_row[1] = sum[m][1];
            out_row[2] = sum[m][2];
            out_row[3] = sum[m][3];
#endif
        }
    }
}

static bool DenseCorePackQ8_0RowsTo4x8(const uint8_t* q8_input_base, size_t q8_row_stride, int rows, int n,
                                       std::vector<uint8_t>& packed) {
    if (!q8_input_base || rows <= 0 || (rows % 4) != 0 || n <= 0 || (n % QK8_0) != 0) {
        return false;
    }
    const int blocks_per_row = n / QK8_0;
    const size_t q8_row_bytes = static_cast<size_t>(blocks_per_row) * sizeof(block_q8_0);
    if (q8_row_stride < q8_row_bytes) {
        return false;
    }
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    packed.resize(static_cast<size_t>(rows / 4) * static_cast<size_t>(blocks_per_row) * packed_block_bytes);

    for (int m = 0; m < rows; m += 4) {
        const auto* row0 =
            reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 0) * q8_row_stride);
        const auto* row1 =
            reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 1) * q8_row_stride);
        const auto* row2 =
            reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 2) * q8_row_stride);
        const auto* row3 =
            reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 3) * q8_row_stride);
        const block_q8_0* rows_in[4] = {row0, row1, row2, row3};
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* dst = packed.data() +
                           (static_cast<size_t>(m / 4) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(b)) *
                               packed_block_bytes;
            auto* dst_d = reinterpret_cast<ggml_fp16_t*>(dst);
            auto* dst_qs = reinterpret_cast<int8_t*>(dst + 4 * sizeof(ggml_fp16_t));
            for (int r = 0; r < 4; ++r) {
                dst_d[r] = rows_in[r][b].d;
            }
            for (int chunk = 0; chunk < 4; ++chunk) {
                for (int r = 0; r < 4; ++r) {
                    std::memcpy(dst_qs + chunk * 4 * 8 + r * 8, rows_in[r][b].qs + chunk * 8, 8);
                }
            }
        }
    }
    return true;
}

static size_t DenseCoreQ8_0RowsTo4x8PackedBytes(int rows, int n) {
    if (rows <= 0 || (rows % 4) != 0 || n <= 0 || (n % QK8_0) != 0) {
        return 0;
    }
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    return static_cast<size_t>(rows / 4) * static_cast<size_t>(n / QK8_0) * packed_block_bytes;
}

static constexpr bool DenseCoreQ8_0Gemm4x8FastBackendCompiled() {
    return true;
}

static inline float DenseCoreQ8_0BlockDot(const block_q8_0* weight_blocks, const block_q8_0* input_blocks, int n) {
    if (!weight_blocks || !input_blocks || n <= 0 || (n % QK8_0) != 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    const int nb = n / QK8_0;
    for (int b = 0; b < nb; ++b) {
        const block_q8_0& w = weight_blocks[b];
        const block_q8_0& x = input_blocks[b];
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(w.qs + 0), vld1q_s8(x.qs + 0));
        acc = vdotq_s32(acc, vld1q_s8(w.qs + 16), vld1q_s8(x.qs + 16));
        const int dot = vaddvq_s32(acc);
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__SSE4_1__)
        const int dot = DenseCoreQ8_0Dot8I8I8(w.qs + 0, x.qs + 0) + DenseCoreQ8_0Dot8I8I8(w.qs + 8, x.qs + 8) +
                        DenseCoreQ8_0Dot8I8I8(w.qs + 16, x.qs + 16) + DenseCoreQ8_0Dot8I8I8(w.qs + 24, x.qs + 24);
#else
        int dot = 0;
        for (int i = 0; i < QK8_0; ++i) {
            dot += static_cast<int>(w.qs[i]) * static_cast<int>(x.qs[i]);
        }
#endif
        sum += static_cast<float>(dot) * ggml_fp16_to_fp32(w.d) * ggml_fp16_to_fp32(x.d);
    }
    return sum;
}
constexpr int64_t kQwen36C4AmxSmartMatmulMaxTokens = 128;
