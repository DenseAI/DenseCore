#pragma once

#include "ggml-cpu.h"
#include "ggml.h"
#include <cstddef>
#include <cstdint>

namespace densecore_q8_small_batch {
struct Block {
    ggml_fp16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Block) == 34, "Q8_0 block layout changed");
static_assert(offsetof(Block, qs) == 2, "Q8_0 scale layout changed");
}  // namespace densecore_q8_small_batch

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>

static constexpr bool Q8SmallBatchDot4Supported() {
    return true;
}

// Native-layout four-column tile. The caller owns shape/phase admission and
// quantization; this function performs no allocation or weight repacking.
// Preconditions: n > 0 and divisible by QK8_0, four valid Q8_0 input rows,
// input_stride >= ggml_row_size(Q8_0, n) and aligned for densecore_q8_small_batch::Block, and four
// writable output locations at output_stride (in floats). Inputs do not alias output.
// Keep each output's block order and reduction identical to ggml's AVX2 dot.
static inline void Q8SmallBatchDot4(int n, const void* weight, const void* inputs, size_t input_stride, float* output,
                                    size_t output_stride) {
    const auto* x = static_cast<const densecore_q8_small_batch::Block*>(weight);
    const auto* base = static_cast<const uint8_t*>(inputs);
    const auto* y0 = reinterpret_cast<const densecore_q8_small_batch::Block*>(base);
    const auto* y1 = reinterpret_cast<const densecore_q8_small_batch::Block*>(base + input_stride);
    const auto* y2 = reinterpret_cast<const densecore_q8_small_batch::Block*>(base + 2 * input_stride);
    const auto* y3 = reinterpret_cast<const densecore_q8_small_batch::Block*>(base + 3 * input_stride);
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < n / 32; ++b) {
        const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].qs));
#if !defined(__AVXVNNIINT8__)
        const __m256i ax = _mm256_sign_epi8(qx, qx);
#endif
        const float dx = _cvtsh_ss(x[b].d);
        const auto accumulate = [&](const densecore_q8_small_batch::Block& y, __m256 acc) {
            const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y.qs));
#if defined(__AVXVNNIINT8__)
            const __m256i sums = _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy);
#else
            const __m256i sy = _mm256_sign_epi8(qy, qx);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const __m256i sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
#elif defined(__AVXVNNI__)
            const __m256i sums = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy);
#else
            const __m256i pairs = _mm256_maddubs_epi16(ax, sy);
            const __m256i sums = _mm256_madd_epi16(_mm256_set1_epi16(1), pairs);

#endif
#endif
            return _mm256_fmadd_ps(_mm256_set1_ps(dx * _cvtsh_ss(y.d)), _mm256_cvtepi32_ps(sums), acc);
        };
        a0 = accumulate(y0[b], a0);
        a1 = accumulate(y1[b], a1);
        a2 = accumulate(y2[b], a2);
        a3 = accumulate(y3[b], a3);
    }
    const auto reduce = [](__m256 value) {
        __m128 r = _mm_add_ps(_mm256_extractf128_ps(value, 1), _mm256_castps256_ps128(value));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        return _mm_cvtss_f32(r);
    };
    output[0] = reduce(a0);
    output[output_stride] = reduce(a1);
    output[2 * output_stride] = reduce(a2);
    output[3 * output_stride] = reduce(a3);
}
#else
static constexpr bool Q8SmallBatchDot4Supported() {
    return false;
}

// Callers should reject this route when unsupported. Retain a correct fallback
// for standalone users, rather than silently leaving outputs unwritten.
static inline void Q8SmallBatchDot4(int n, const void* weight, const void* inputs, size_t input_stride, float* output,
                                    size_t output_stride) {
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    for (size_t m = 0; m < 4; ++m) {
        dot(n, output + m * output_stride, 0, weight, 0, static_cast<const uint8_t*>(inputs) + m * input_stride, 0, 1);
    }
}
#endif
