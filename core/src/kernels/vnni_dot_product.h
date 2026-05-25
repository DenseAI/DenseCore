/**
 * @file vnni_dot_product.h
 * @brief AVX-512 VNNI-optimized dot product kernels for Q8_0 KV cache
 *
 * Replaces the Highway i8→i16→i32→f32 promote chain (6 instructions/16 elements)
 * with direct AVX-512 intrinsics and optional VNNI dpbusd (1 instruction/32 elements).
 *
 * Two optimization tiers:
 *   Tier 1 (AVX-512F): Direct _mm512_cvtepi8_epi32 bypasses Highway's 3-step promote.
 *   Tier 2 (AVX-512 VNNI): Pre-quantize query to uint8, then use _mm512_dpbusd_epi32
 *          for 4x throughput. Query quantization cost is amortized over all context tokens.
 *
 * IMPORTANT: This header must only be included in compilation units targeting
 * AVX-512F or higher. Never include in generic/portable code.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if (defined(__AVX512F__) && (defined(__x86_64__) || defined(_M_X64)))

#include <immintrin.h>

namespace densecore {
namespace vnni {

// ---------------------------------------------------------------------------
// fp16→fp32 helper (matches densecore::fp16_to_fp32 from hwy_paged_attention.cc)
// ---------------------------------------------------------------------------
inline float fp16_bits_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

// ---------------------------------------------------------------------------
// Tier 1: AVX-512F Q8_0 Dot Product (no VNNI required)
//
// Processes Q8_0 blocks (32 int8 + fp16 scale) using direct i8→i32 conversion
// instead of the Highway i8→i16→i32→f32 promote chain.
// ~30% fewer instructions per block vs Highway path.
// ---------------------------------------------------------------------------
inline float DotProductQ8_0_AVX512(const float* q, const void* k_data, int head_dim) {
    struct block_q8_0_view {
        uint16_t d;
        int8_t qs[32];
    };
    const auto* blocks = reinterpret_cast<const block_q8_0_view*>(k_data);
    const int nb = head_dim / 32;

    __m512 total = _mm512_setzero_ps();
    int q_offset = 0;

    for (int b = 0; b < nb; ++b) {
        const float scale = fp16_bits_to_fp32(blocks[b].d);
        const __m512 v_scale = _mm512_set1_ps(scale);
        const int8_t* qs = blocks[b].qs;

        // First 16 elements: i8→i32→f32 (direct, no intermediate i16)
        __m128i k8_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
        __m512i k32_lo = _mm512_cvtepi8_epi32(k8_lo);
        __m512 kf_lo = _mm512_cvtepi32_ps(k32_lo);
        __m512 qf_lo = _mm512_loadu_ps(q + q_offset);
        total = _mm512_fmadd_ps(qf_lo, _mm512_mul_ps(kf_lo, v_scale), total);

        // Next 16 elements
        __m128i k8_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));
        __m512i k32_hi = _mm512_cvtepi8_epi32(k8_hi);
        __m512 kf_hi = _mm512_cvtepi32_ps(k32_hi);
        __m512 qf_hi = _mm512_loadu_ps(q + q_offset + 16);
        total = _mm512_fmadd_ps(qf_hi, _mm512_mul_ps(kf_hi, v_scale), total);

        q_offset += 32;
    }

    // Tail elements (head_dim not divisible by 32)
    float result = _mm512_reduce_add_ps(total);
    int tail_start = nb * 32;
    if (tail_start < head_dim) {
        float block_scale = fp16_bits_to_fp32(blocks[nb].d);
        const int8_t* qs = blocks[nb].qs;
        for (int i = 0; i < head_dim - tail_start; ++i) {
            result += q[tail_start + i] * (static_cast<float>(qs[i]) * block_scale);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tier 1: AVX-512F Q8_0 Accumulate (weighted V addition)
// ---------------------------------------------------------------------------
inline void AccumulateQ8_0_AVX512(float* accum, const void* v_data, float weight, int head_dim) {
    struct block_q8_0_view {
        uint16_t d;
        int8_t qs[32];
    };
    const auto* blocks = reinterpret_cast<const block_q8_0_view*>(v_data);
    const int nb = head_dim / 32;
    int accum_offset = 0;

    for (int b = 0; b < nb; ++b) {
        const float combined_scale = fp16_bits_to_fp32(blocks[b].d) * weight;
        const __m512 v_scale = _mm512_set1_ps(combined_scale);
        const int8_t* qs = blocks[b].qs;

        // First 16 elements
        __m128i vi8_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
        __m512i vi32_lo = _mm512_cvtepi8_epi32(vi8_lo);
        __m512 vf_lo = _mm512_cvtepi32_ps(vi32_lo);
        __m512 acc_lo = _mm512_loadu_ps(accum + accum_offset);
        _mm512_storeu_ps(accum + accum_offset, _mm512_fmadd_ps(vf_lo, v_scale, acc_lo));

        // Next 16 elements
        __m128i vi8_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));
        __m512i vi32_hi = _mm512_cvtepi8_epi32(vi8_hi);
        __m512 vf_hi = _mm512_cvtepi32_ps(vi32_hi);
        __m512 acc_hi = _mm512_loadu_ps(accum + accum_offset + 16);
        _mm512_storeu_ps(accum + accum_offset + 16, _mm512_fmadd_ps(vf_hi, v_scale, acc_hi));

        accum_offset += 32;
    }

    // Tail
    int tail_start = nb * 32;
    if (tail_start < head_dim) {
        float block_scale = fp16_bits_to_fp32(blocks[nb].d) * weight;
        const int8_t* qs = blocks[nb].qs;
        for (int i = 0; i < head_dim - tail_start; ++i) {
            accum[tail_start + i] += static_cast<float>(qs[i]) * block_scale;
        }
    }
}

#if defined(__AVX512VNNI__)

// ---------------------------------------------------------------------------
// Tier 2: VNNI Query Pre-Quantization
//
// Quantizes FP32 query vector to UINT8 (zero_point=128) for use with dpbusd.
// Cost amortized over all context tokens in a single attention head.
// Returns inverse scale for dequantization.
// ---------------------------------------------------------------------------
struct QuantizedQuery {
    alignas(64) uint8_t q_u8[1024];  // Max head_dim=1024
    float inv_scale;
    int head_dim;
    // Pre-computed per-block zero-point correction sums are NOT stored here;
    // they depend on the key data and must be computed per-token.
};

inline void QuantizeQueryToU8(const float* query, QuantizedQuery* out, int head_dim) {
    out->head_dim = head_dim;

    // Find max absolute value across query
    __m512 vmax = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= head_dim; i += 16) {
        __m512 v = _mm512_loadu_ps(query + i);
        // abs via clearing sign bit
        vmax = _mm512_max_ps(vmax, _mm512_castsi512_ps(
            _mm512_andnot_si512(_mm512_set1_epi32(0x80000000),
                                _mm512_castps_si512(v))));
    }
    float max_abs = _mm512_reduce_max_ps(vmax);
    for (; i < head_dim; ++i) {
        float a = std::abs(query[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-10f) {
        std::memset(out->q_u8, 128, static_cast<size_t>(head_dim));
        out->inv_scale = 0.0f;
        return;
    }

    const float scale = 127.0f / max_abs;
    out->inv_scale = max_abs / 127.0f;

    const __m512 v_scale = _mm512_set1_ps(scale);
    const __m512 v_128 = _mm512_set1_ps(128.0f);
    const __m512 v_0 = _mm512_setzero_ps();
    const __m512 v_255 = _mm512_set1_ps(255.0f);

    i = 0;
    for (; i + 16 <= head_dim; i += 16) {
        __m512 v = _mm512_loadu_ps(query + i);
        v = _mm512_fmadd_ps(v, v_scale, v_128);
        v = _mm512_max_ps(v_0, _mm512_min_ps(v_255, v));
        __m512i vi = _mm512_cvtps_epi32(v);
        __m128i packed = _mm512_cvtsepi32_epi8(vi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out->q_u8 + i), packed);
    }
    for (; i < head_dim; ++i) {
        float v = query[i] * scale + 128.0f;
        v = std::max(0.0f, std::min(255.0f, v));
        out->q_u8[i] = static_cast<uint8_t>(v + 0.5f);
    }
}

// ---------------------------------------------------------------------------
// Tier 2: VNNI dpbusd Q8_0 Dot Product
//
// Uses _mm256_dpbusd_epi32 to compute dot product of pre-quantized uint8 query
// against int8 key in a single instruction per 32 elements (1 Q8_0 block).
//
// Per Q8_0 block (32 elements):
//   raw_dot = dpbusd(zero, q_u8[32], k_i8[32])  → 8 int32 partial sums
//   sum_k   = horizontal_sum(k_i8[32])           → int32
//   corrected = raw_dot - 128 * sum_k            → removes zero-point bias
//   result += corrected * query_inv_scale * key_scale
//
// ~3x fewer instructions than Tier 1 per block.
// ---------------------------------------------------------------------------
inline float DotProductQ8_0_VNNI(const QuantizedQuery* qq, const void* k_data, int head_dim) {
    struct block_q8_0_view {
        uint16_t d;
        int8_t qs[32];
    };
    const auto* blocks = reinterpret_cast<const block_q8_0_view*>(k_data);
    const int nb = head_dim / 32;

    float total_f = 0.0f;
    int q_offset = 0;

    for (int b = 0; b < nb; ++b) {
        const float k_scale = fp16_bits_to_fp32(blocks[b].d);
        const int8_t* ks = blocks[b].qs;

        // Load 32 uint8 query and 32 int8 key values
        __m256i q_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(qq->q_u8 + q_offset));
        __m256i k_vec = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(ks));

        // VNNI: 8 groups of 4-element unsigned×signed dot products
        __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_vec, k_vec);

        // Horizontal sum of 8 int32 → single int32
        // dp = [d0, d1, d2, d3, d4, d5, d6, d7]
        __m128i dp_hi = _mm256_extracti128_si256(dp, 1);
        __m128i dp_lo = _mm256_castsi256_si128(dp);
        __m128i sum4 = _mm_add_epi32(dp_lo, dp_hi);          // [d0+d4, d1+d5, d2+d6, d3+d7]
        __m128i sum2 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x4E)); // swap high/low pairs
        __m128i sum1 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 0xB1)); // swap adjacent
        int32_t raw_dot = _mm_cvtsi128_si32(sum1);

        // Compute sum of key int8 values for zero-point correction
        // Use _mm256_sad_epu8 trick: reinterpret signed as unsigned, adjust later
        // Actually simpler: promote to i16 and hadd
        __m256i k_16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(k_vec));
        __m256i k_16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(k_vec, 1));
        // Sum 16 int16 values from each half
        __m128i k_sum_lo = _mm_add_epi16(_mm256_castsi256_si128(k_16_lo),
                                          _mm256_extracti128_si256(k_16_lo, 1));
        __m128i k_sum_hi = _mm_add_epi16(_mm256_castsi256_si128(k_16_hi),
                                          _mm256_extracti128_si256(k_16_hi, 1));
        __m128i k_sum = _mm_add_epi16(k_sum_lo, k_sum_hi);
        // Pairwise horizontal add to collapse 8 int16 → 4 int32 → single
        __m128i k_sum32 = _mm_madd_epi16(k_sum, _mm_set1_epi16(1));
        k_sum32 = _mm_add_epi32(k_sum32, _mm_shuffle_epi32(k_sum32, 0x4E));
        k_sum32 = _mm_add_epi32(k_sum32, _mm_shuffle_epi32(k_sum32, 0xB1));
        int32_t sum_k = _mm_cvtsi128_si32(k_sum32);

        // Correct for zero-point: dot = raw_dot - 128 * sum_k
        int32_t corrected = raw_dot - 128 * sum_k;
        total_f += static_cast<float>(corrected) * k_scale;

        q_offset += 32;
    }

    // Tail
    int tail_start = nb * 32;
    if (tail_start < head_dim) {
        float block_scale = fp16_bits_to_fp32(blocks[nb].d);
        const int8_t* qs = blocks[nb].qs;
        for (int i = 0; i < head_dim - tail_start; ++i) {
            const int32_t corrected = static_cast<int32_t>(qq->q_u8[tail_start + i]) - 128;
            total_f += static_cast<float>(corrected) * (static_cast<float>(qs[i]) * block_scale);
        }
    }

    return total_f * qq->inv_scale;
}

#endif  // __AVX512VNNI__

}  // namespace vnni
}  // namespace densecore

#endif  // __AVX512F__
