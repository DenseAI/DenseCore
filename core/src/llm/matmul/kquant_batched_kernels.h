#pragma once
#include "densecore/memory/kv_cache.h"
#include "ggml-cpu.h"
#include "llm/matmul/userdata.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

struct DensecoreBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(DensecoreBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "DensecoreBlockQ8K layout mismatch");

struct DensecoreBlockQ5K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(DensecoreBlockQ5K) == 2 * sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
              "DensecoreBlockQ5K layout mismatch");

// True-batched Q4_K x Q8_K row dot:
// - Reuses Q4_K decode/scales once per weight row.
// - Computes all M column dots in one pass.
inline bool ComputeQ4KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base,
                                          size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    float lane_acc[kMaxSmallBatchColsHard][8];
    float min_acc[kMaxSmallBatchColsHard];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q4[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];
        const uint8_t* q4 = xb.qs;
        int8_t* uq4 = unpacked_q4;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] & 0xF);
            uq4 += 32;
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] >> 4);
            uq4 += 32;
            q4 += 32;
        }

        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            int32_t sumi = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                sumi += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q4;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * (static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]));
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = x_dmin * yd;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(sumi);
        }
    }

    for (int m = 0; m < M; ++m) {
        float sum = min_acc[m];
        for (int l = 0; l < 8; ++l) {
            sum += lane_acc[m][l];
        }
        out_sums[m] = sum;
    }

    return true;
}

inline bool ComputeQ5KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base,
                                          size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x_blocks = reinterpret_cast<const DensecoreBlockQ5K*>(weight_row);
    const int nb = N / QK_K;
    float lane_acc[kMaxSmallBatchColsHard][8];
    float min_acc[kMaxSmallBatchColsHard];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q5[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];
        const uint8_t* q4 = xb.qs;
        const uint8_t* high = xb.qh;
        int8_t* uq5 = unpacked_q5;
        uint8_t high_mask = 1;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) {
                uq5[l] = static_cast<int8_t>((q4[l] & 0xF) + ((high[l] & high_mask) ? 16 : 0));
            }
            uq5 += 32;
            high_mask <<= 1;
            for (int l = 0; l < 32; ++l) {
                uq5[l] = static_cast<int8_t>((q4[l] >> 4) + ((high[l] & high_mask) ? 16 : 0));
            }
            uq5 += 32;
            high_mask <<= 1;
            q4 += 32;
        }

        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            int32_t sumi = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                sumi += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q5;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * (static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]));
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = x_dmin * yd;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(sumi);
        }
    }

    for (int m = 0; m < M; ++m) {
        float sum = min_acc[m];
        for (int l = 0; l < 8; ++l) {
            sum += lane_acc[m][l];
        }
        out_sums[m] = sum;
    }

    return true;
}

#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
inline bool ComputeQ4KQ8KBatchedRowDotprod(const void* weight_row, const uint8_t* quant_input_base,
                                           size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    std::fill(out_sums, out_sums + M, 0.0f);
    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q4_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float q4_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q4_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        int8x16_t q4_vec[QK_K / 32][2];
        const uint8_t* q4 = xb.qs;
        for (int chunk = 0; chunk < QK_K / 64; ++chunk) {
            const uint8x16_t packed0 = vld1q_u8(q4);
            const uint8x16_t packed1 = vld1q_u8(q4 + 16);
            q4 += 32;
            q4_vec[2 * chunk][0] = vreinterpretq_s8_u8(vandq_u8(packed0, low_mask));
            q4_vec[2 * chunk][1] = vreinterpretq_s8_u8(vandq_u8(packed1, low_mask));
            q4_vec[2 * chunk + 1][0] = vreinterpretq_s8_u8(vshrq_n_u8(packed0, 4));
            q4_vec[2 * chunk + 1][1] = vreinterpretq_s8_u8(vshrq_n_u8(packed1, 4));
        }

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            int32_t dot_scaled = 0;
            const int8_t* q8 = yb.qs;
            for (int group = 0; group < QK_K / 32; ++group) {
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32), q4_vec[group][0]);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32 + 16), q4_vec[group][1]);
                dot_scaled += static_cast<int32_t>(scales[group]) * vaddvq_s32(acc);
            }

            const float d = q4_d * yb.d;
            const float dmin = q4_dmin * yb.d;
            out_sums[m] += d * static_cast<float>(dot_scaled) - dmin * static_cast<float>(min_dot);
        }
    }
    return true;
}

inline bool ComputeQ5KQ8KBatchedRowDotprod(const void* weight_row, const uint8_t* quant_input_base,
                                           size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    std::fill(out_sums, out_sums + M, 0.0f);
    const auto* q5_blocks = reinterpret_cast<const DensecoreBlockQ5K*>(weight_row);
    const int nb = N / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t high_value = vdupq_n_u8(16);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q5_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float q5_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q5_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        int8x16_t q5_vec[QK_K / 32][2];
        const uint8_t* q5 = xb.qs;
        const uint8_t* qh = xb.qh;
        uint8_t high_mask = 1;
        for (int chunk = 0; chunk < QK_K / 64; ++chunk) {
            const uint8x16_t packed0 = vld1q_u8(q5);
            const uint8x16_t packed1 = vld1q_u8(q5 + 16);
            const uint8x16_t high0 = vld1q_u8(qh);
            const uint8x16_t high1 = vld1q_u8(qh + 16);
            q5 += 32;

            uint8x16_t mask = vdupq_n_u8(high_mask);
            uint8x16_t add0 = vandq_u8(vcgtq_u8(vandq_u8(high0, mask), zero), high_value);
            uint8x16_t add1 = vandq_u8(vcgtq_u8(vandq_u8(high1, mask), zero), high_value);
            q5_vec[2 * chunk][0] = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(packed0, low_mask), add0));
            q5_vec[2 * chunk][1] = vreinterpretq_s8_u8(vaddq_u8(vandq_u8(packed1, low_mask), add1));

            high_mask <<= 1;
            mask = vdupq_n_u8(high_mask);
            add0 = vandq_u8(vcgtq_u8(vandq_u8(high0, mask), zero), high_value);
            add1 = vandq_u8(vcgtq_u8(vandq_u8(high1, mask), zero), high_value);
            q5_vec[2 * chunk + 1][0] = vreinterpretq_s8_u8(vaddq_u8(vshrq_n_u8(packed0, 4), add0));
            q5_vec[2 * chunk + 1][1] = vreinterpretq_s8_u8(vaddq_u8(vshrq_n_u8(packed1, 4), add1));
            high_mask <<= 1;
        }

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            int32_t dot_scaled = 0;
            const int8_t* q8 = yb.qs;
            for (int group = 0; group < QK_K / 32; ++group) {
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32), q5_vec[group][0]);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32 + 16), q5_vec[group][1]);
                dot_scaled += static_cast<int32_t>(scales[group]) * vaddvq_s32(acc);
            }

            const float d = q5_d * yb.d;
            const float dmin = q5_dmin * yb.d;
            out_sums[m] += d * static_cast<float>(dot_scaled) - dmin * static_cast<float>(min_dot);
        }
    }
    return true;
}
#endif

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
inline float HSumFloat8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline bool ComputeQ4KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base,
                                        size_t quant_row_stride, int M, int N, float* out_sums) {
    const bool debug_q4k_path = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    static std::atomic<bool> logged_invalid_args{false};
    static std::atomic<bool> logged_invalid_m{false};
    static std::atomic<bool> logged_invalid_n{false};
    static std::atomic<bool> logged_stride{false};
    static std::atomic<bool> logged_qkk{false};

    if (!weight_row || !quant_input_base || !out_sums) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_args.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_args" << std::endl;
            }
        }
        return false;
    }
    if (M <= 0 || M > kMaxSmallBatchColsHard) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_m.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_m M=" << M << std::endl;
            }
        }
        return false;
    }
    if (N <= 0 || (N % QK_K) != 0) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_n.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_n N=" << N << std::endl;
            }
        }
        return false;
    }
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_stride.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable stride stride=" << quant_row_stride << " need="
                          << (static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K))
                          << std::endl;
            }
        }
        return false;
    }
    if (QK_K != 256) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_qkk.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable QK_K=" << QK_K << std::endl;
            }
        }
        return false;
    }
    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums{};

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);

        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        // Reuse decoded q4 nibble vectors/scales for all M columns.
        __m256i q4l[QK_K / 64];
        __m256i q4h[QK_K / 64];
        __m256i scale_l[QK_K / 64];
        __m256i scale_h[QK_K / 64];
        const uint8_t* q4 = xb.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j]));
            scale_h[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j + 1]));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = -x_dmin * yd;

            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            __m128i sum32 = _mm_hadd_epi32(prod, prod);
            sum32 = _mm_hadd_epi32(sum32, sum32);
            const int32_t min_dot = _mm_cvtsi128_si32(sum32);

            const int8_t* q8 = yb.qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);

                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            sums[static_cast<size_t>(m)] +=
                d * HSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}

inline bool ComputeQ4KQ8KBatchedRow2Avx2(const void* weight_row0, const void* weight_row1,
                                         const uint8_t* quant_input_base, size_t quant_row_stride, int M, int N,
                                         float* out0_sums, float* out1_sums) {
    if (!weight_row0 || !weight_row1 || !quant_input_base || !out0_sums || !out1_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0 || QK_K != 256) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x0_blocks = reinterpret_cast<const block_q4_K*>(weight_row0);
    const auto* x1_blocks = reinterpret_cast<const block_q4_K*>(weight_row1);
    const int nb = N / QK_K;
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums0{};
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums1{};

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);

    for (int bi = 0; bi < nb; ++bi) {
        auto prepare_block = [&](const block_q4_K& xb, __m256i* q4l, __m256i* q4h, __m256i* scale_l, __m256i* scale_h,
                                 __m128i* mins, float* x_d, float* x_dmin) {
            uint32_t utmp[4];
            std::memcpy(utmp, xb.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            *mins = _mm256_extracti128_si256(mins_and_scales, 1);
            *x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
            *x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

            const uint8_t* q4 = xb.qs;
            for (int j = 0; j < QK_K / 64; ++j) {
                scale_l[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j]));
                scale_h[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j + 1]));
                const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
                q4 += 32;
                q4l[j] = _mm256_and_si256(q4bits, m4);
                q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            }
        };

        __m256i q4l0[QK_K / 64];
        __m256i q4h0[QK_K / 64];
        __m256i scale_l0[QK_K / 64];
        __m256i scale_h0[QK_K / 64];
        __m128i mins0;
        float x0_d = 0.0f;
        float x0_dmin = 0.0f;
        prepare_block(x0_blocks[bi], q4l0, q4h0, scale_l0, scale_h0, &mins0, &x0_d, &x0_dmin);

        __m256i q4l1[QK_K / 64];
        __m256i q4h1[QK_K / 64];
        __m256i scale_l1[QK_K / 64];
        __m256i scale_h1[QK_K / 64];
        __m128i mins1;
        float x1_d = 0.0f;
        float x1_dmin = 0.0f;
        prepare_block(x1_blocks[bi], q4l1, q4h1, scale_l1, scale_h1, &mins1, &x1_d, &x1_dmin);

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            const __m128i prod0 = _mm_madd_epi16(mins0, q8s);
            __m128i sum320 = _mm_hadd_epi32(prod0, prod0);
            sum320 = _mm_hadd_epi32(sum320, sum320);
            const int32_t min_dot0 = _mm_cvtsi128_si32(sum320);

            const __m128i prod1 = _mm_madd_epi16(mins1, q8s);
            __m128i sum321 = _mm_hadd_epi32(prod1, prod1);
            sum321 = _mm_hadd_epi32(sum321, sum321);
            const int32_t min_dot1 = _mm_cvtsi128_si32(sum321);

            const int8_t* q8 = yb.qs;
            __m256i sumi0 = _mm256_setzero_si256();
            __m256i sumi1 = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;

                __m256i p0l = _mm256_maddubs_epi16(q4l0[j], q8l);
                p0l = _mm256_madd_epi16(scale_l0[j], p0l);
                __m256i p0h = _mm256_maddubs_epi16(q4h0[j], q8h);
                p0h = _mm256_madd_epi16(scale_h0[j], p0h);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p0l, p0h));

                __m256i p1l = _mm256_maddubs_epi16(q4l1[j], q8l);
                p1l = _mm256_madd_epi16(scale_l1[j], p1l);
                __m256i p1h = _mm256_maddubs_epi16(q4h1[j], q8h);
                p1h = _mm256_madd_epi16(scale_h1[j], p1h);
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1l, p1h));
            }

            const float yd = yb.d;
            const float d0 = x0_d * yd;
            const float dmin0 = -x0_dmin * yd;
            sums0[static_cast<size_t>(m)] +=
                d0 * HSumFloat8(_mm256_cvtepi32_ps(sumi0)) + dmin0 * static_cast<float>(min_dot0);

            const float d1 = x1_d * yd;
            const float dmin1 = -x1_dmin * yd;
            sums1[static_cast<size_t>(m)] +=
                d1 * HSumFloat8(_mm256_cvtepi32_ps(sumi1)) + dmin1 * static_cast<float>(min_dot1);
        }
    }

    for (int m = 0; m < M; ++m) {
        out0_sums[m] = sums0[static_cast<size_t>(m)];
        out1_sums[m] = sums1[static_cast<size_t>(m)];
    }
    return true;
}

inline bool ComputeQ5KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base,
                                        size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0 || QK_K != 256) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x_blocks = reinterpret_cast<const DensecoreBlockQ5K*>(weight_row);
    const int nb = N / QK_K;
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums{};

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i high_value = _mm256_set1_epi8(16);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        __m256i q5l[QK_K / 64];
        __m256i q5h[QK_K / 64];
        __m256i scale_l[QK_K / 64];
        __m256i scale_h[QK_K / 64];
        const uint8_t* q5 = xb.qs;
        const uint8_t* qh = xb.qh;
        uint8_t high_mask = 1;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j]));
            scale_h[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j + 1]));
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5));
            const __m256i high_bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            q5 += 32;

            __m256i mask = _mm256_set1_epi8(static_cast<char>(high_mask));
            __m256i add = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high_bits, mask), zero), high_value);
            q5l[j] = _mm256_add_epi8(_mm256_and_si256(packed, low_mask), add);
            high_mask <<= 1;

            mask = _mm256_set1_epi8(static_cast<char>(high_mask));
            add = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high_bits, mask), zero), high_value);
            q5h[j] = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask), add);
            high_mask <<= 1;
        }

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            __m128i sum32 = _mm_hadd_epi32(prod, prod);
            sum32 = _mm_hadd_epi32(sum32, sum32);
            const int32_t min_dot = _mm_cvtsi128_si32(sum32);

            const int8_t* q8 = yb.qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q5l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);

                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q5h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = -x_dmin * yd;
            sums[static_cast<size_t>(m)] +=
                d * HSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}
#endif

inline bool ComputeQ4KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                    int M, int N, float* out_sums) {
    static const bool debug_q4k_path = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
    static std::atomic<bool> logged_dotprod{false};
    if (ComputeQ4KQ8KBatchedRowDotprod(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_dotprod.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] arm_dotprod" << std::endl;
            }
        }
        return true;
    }
#endif
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    static std::atomic<bool> logged_avx2{false};
    static std::atomic<bool> logged_scalar{false};
    // Keep the explicit AVX2 batched row kernel as the maintained x86 path.
    // C4 validation for the same Q4_K/Q8_K batched-row shape measured the
    // Highway VNNI experiment slower than this kernel; the Highway helper stays
    // available for direct tests and future requalification, but is not the
    // default matmul admission path.
    if (ggml_cpu_has_avx2() &&
        ComputeQ4KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_avx2.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2" << std::endl;
            }
        }
        return true;
    }
    if (debug_q4k_path) {
        bool expected = false;
        if (logged_scalar.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
            std::cerr << "[Q4K_BATCHED_PATH] scalar" << std::endl;
        }
    }
#endif
    return ComputeQ4KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, N, out_sums);
}

inline bool ComputeQ5KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                    int M, int N, float* out_sums) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
    if (ComputeQ5KQ8KBatchedRowDotprod(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        return true;
    }
#endif
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (ggml_cpu_has_avx2() &&
        ComputeQ5KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        return true;
    }
#endif
    return ComputeQ5KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, N, out_sums);
}
