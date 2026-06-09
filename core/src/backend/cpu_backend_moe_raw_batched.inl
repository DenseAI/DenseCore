struct MoEBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(MoEBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "MoE Q8_K block layout must match ggml block_q8_K");

struct MoEBlockQ5K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(MoEBlockQ5K) == 2 * sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
              "MoE Q5_K block layout must match ggml block_q5_K");

struct MoEBlockQ8Kx4 {
    float d[4];
    int8_t qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
};
static_assert(sizeof(MoEBlockQ8Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t),
              "MoE Q8_Kx4 block layout must match ggml block_q8_Kx4");

bool ComputeMoEQ4KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                      int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQ4KRawBatchedTileM || K <= 0 ||
        (K % QK_K) != 0) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    float lane_acc[kMoEQ4KRawBatchedTileM][8];
    float min_acc[kMoEQ4KRawBatchedTileM];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q4[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q4_blocks[bi];
        const uint8_t* q4 = xb.qs;
        int8_t* uq4 = unpacked_q4;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) {
                uq4[l] = static_cast<int8_t>(q4[l] & 0xF);
            }
            uq4 += 32;
            for (int l = 0; l < 32; ++l) {
                uq4[l] = static_cast<int8_t>(q4[l] >> 4);
            }
            uq4 += 32;
            q4 += 32;
        }

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

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q4;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]);
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float d = q4_d * yb.d;
            const float dmin = q4_dmin * yb.d;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(min_dot);
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
bool ComputeMoEQ4KQ8KBatchedRowDotprod(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                       int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQ4KRawBatchedTileM || K <= 0 ||
        (K % QK_K) != 0) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    std::fill(out_sums, out_sums + M, 0.0f);
    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
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
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
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
#endif

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline float MoEHSumFloat8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

bool ComputeMoEQ4KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                    int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQ4KRawBatchedTileM || K <= 0 ||
        (K % QK_K) != 0 || QK_K != 256) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    alignas(64) std::array<float, kMoEQ4KRawBatchedTileM> sums{};
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

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
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const float q4_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q4_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        __m256i q4l[QK_K / 64];
        __m256i q4h[QK_K / 64];
        __m256i scale_l[QK_K / 64];
        __m256i scale_h[QK_K / 64];
        const uint8_t* q4 = xb.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j]));
            scale_h[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j + 1]));
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
            q4 += 32;
            q4l[j] = _mm256_and_si256(packed, low_mask);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
        }

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];
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
            const float d = q4_d * yb.d;
            const float dmin = -q4_dmin * yb.d;
            sums[static_cast<size_t>(m)] +=
                d * MoEHSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }
    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}

struct MoEQ4KAvx2Block {
    __m256i q4l[QK_K / 64];
    __m256i q4h[QK_K / 64];
    __m256i scale_l[QK_K / 64];
    __m256i scale_h[QK_K / 64];
    __m128i mins;
    float d;
    float dmin;
};

static inline void PrepareMoEQ4KAvx2Block(const block_q4_K& xb, const __m256i low_mask, MoEQ4KAvx2Block* out) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    std::memcpy(utmp, xb.scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
    out->mins = _mm256_extracti128_si256(mins_and_scales, 1);
    out->d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
    out->dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

    const uint8_t* q4 = xb.qs;
    for (int j = 0; j < QK_K / 64; ++j) {
        out->scale_l[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j]));
        out->scale_h[j] = _mm256_set1_epi16(static_cast<int16_t>(scales[2 * j + 1]));
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
        q4 += 32;
        out->q4l[j] = _mm256_and_si256(packed, low_mask);
        out->q4h[j] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
    }
}

bool ComputeMoEQ4KQ8KBatchedRowPairAvx2(const void* gate_weight_row, const void* up_weight_row,
                                        const uint8_t* quant_input_base, size_t quant_row_stride, int M, int K,
                                        float* gate_out_sums, float* up_out_sums) {
    if (!gate_weight_row || !up_weight_row || !quant_input_base || !gate_out_sums || !up_out_sums || M <= 0 ||
        M > kMoEQ4KRawBatchedTileM || K <= 0 || (K % QK_K) != 0 || QK_K != 256) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* gate_blocks = reinterpret_cast<const block_q4_K*>(gate_weight_row);
    const auto* up_blocks = reinterpret_cast<const block_q4_K*>(up_weight_row);
    alignas(64) std::array<float, kMoEQ4KRawBatchedTileM> gate_sums{};
    alignas(64) std::array<float, kMoEQ4KRawBatchedTileM> up_sums{};
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (int bi = 0; bi < nb; ++bi) {
        MoEQ4KAvx2Block gate_block;
        MoEQ4KAvx2Block up_block;
        PrepareMoEQ4KAvx2Block(gate_blocks[bi], low_mask, &gate_block);
        PrepareMoEQ4KAvx2Block(up_blocks[bi], low_mask, &up_block);

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            const __m128i gate_prod = _mm_madd_epi16(gate_block.mins, q8s);
            __m128i gate_sum32 = _mm_hadd_epi32(gate_prod, gate_prod);
            gate_sum32 = _mm_hadd_epi32(gate_sum32, gate_sum32);
            const int32_t gate_min_dot = _mm_cvtsi128_si32(gate_sum32);

            const __m128i up_prod = _mm_madd_epi16(up_block.mins, q8s);
            __m128i up_sum32 = _mm_hadd_epi32(up_prod, up_prod);
            up_sum32 = _mm_hadd_epi32(up_sum32, up_sum32);
            const int32_t up_min_dot = _mm_cvtsi128_si32(up_sum32);

            const int8_t* q8 = yb.qs;
            __m256i gate_sumi = _mm256_setzero_si256();
            __m256i up_sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;

                __m256i gate_p16l = _mm256_maddubs_epi16(gate_block.q4l[j], q8l);
                gate_p16l = _mm256_madd_epi16(gate_block.scale_l[j], gate_p16l);
                __m256i gate_p16h = _mm256_maddubs_epi16(gate_block.q4h[j], q8h);
                gate_p16h = _mm256_madd_epi16(gate_block.scale_h[j], gate_p16h);
                gate_sumi = _mm256_add_epi32(gate_sumi, _mm256_add_epi32(gate_p16l, gate_p16h));

                __m256i up_p16l = _mm256_maddubs_epi16(up_block.q4l[j], q8l);
                up_p16l = _mm256_madd_epi16(up_block.scale_l[j], up_p16l);
                __m256i up_p16h = _mm256_maddubs_epi16(up_block.q4h[j], q8h);
                up_p16h = _mm256_madd_epi16(up_block.scale_h[j], up_p16h);
                up_sumi = _mm256_add_epi32(up_sumi, _mm256_add_epi32(up_p16l, up_p16h));
            }

            const float gate_d = gate_block.d * yb.d;
            const float gate_dmin = -gate_block.dmin * yb.d;
            gate_sums[static_cast<size_t>(m)] +=
                gate_d * MoEHSumFloat8(_mm256_cvtepi32_ps(gate_sumi)) + gate_dmin * static_cast<float>(gate_min_dot);

            const float up_d = up_block.d * yb.d;
            const float up_dmin = -up_block.dmin * yb.d;
            up_sums[static_cast<size_t>(m)] +=
                up_d * MoEHSumFloat8(_mm256_cvtepi32_ps(up_sumi)) + up_dmin * static_cast<float>(up_min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        gate_out_sums[m] = gate_sums[static_cast<size_t>(m)];
        up_out_sums[m] = up_sums[static_cast<size_t>(m)];
    }
    return true;
}
#endif

bool ComputeMoEQ4KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride, int M,
                                int K, float* out_sums) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
    if (ComputeMoEQ4KQ8KBatchedRowDotprod(weight_row, quant_input_base, quant_row_stride, M, K, out_sums)) {
        return true;
    }
#endif
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (ggml_cpu_has_avx2() &&
        ComputeMoEQ4KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, K, out_sums)) {
        return true;
    }
#endif
    return ComputeMoEQ4KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, K, out_sums);
}

bool ComputeMoEQ4KQ8KBatchedRowPair(const void* gate_weight_row, const void* up_weight_row,
                                    const uint8_t* quant_input_base, size_t quant_row_stride, int M, int K,
                                    float* gate_out_sums, float* up_out_sums) {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (ggml_cpu_has_avx2() &&
        ComputeMoEQ4KQ8KBatchedRowPairAvx2(gate_weight_row, up_weight_row, quant_input_base, quant_row_stride, M, K,
                                           gate_out_sums, up_out_sums)) {
        return true;
    }
#endif
    return ComputeMoEQ4KQ8KBatchedRow(gate_weight_row, quant_input_base, quant_row_stride, M, K, gate_out_sums) &&
           ComputeMoEQ4KQ8KBatchedRow(up_weight_row, quant_input_base, quant_row_stride, M, K, up_out_sums);
}

bool ComputeMoEQ5KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                      int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K <= 0 ||
        (K % QK_K) != 0) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* q5_blocks = reinterpret_cast<const MoEBlockQ5K*>(weight_row);
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    alignas(64) std::array<float, kMoEQuantizedProjectionMaxBatch> sums{};
    int8_t unpacked_q5[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q5_blocks[bi];
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

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float q5_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q5_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q5;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]);
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            float dot_sum = 0.0f;
            for (int l = 0; l < 8; ++l) {
                dot_sum += static_cast<float>(dot_chunks[l]);
            }
            const float d = q5_d * yb.d;
            const float dmin = -q5_dmin * yb.d;
            sums[static_cast<size_t>(m)] += d * dot_sum + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
bool ComputeMoEQ5KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                    int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K <= 0 ||
        (K % QK_K) != 0 || QK_K != 256) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* q5_blocks = reinterpret_cast<const MoEBlockQ5K*>(weight_row);
    alignas(64) std::array<float, kMoEQuantizedProjectionMaxBatch> sums{};
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i high_value = _mm256_set1_epi8(16);

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
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const float q5_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q5_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

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
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];
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

            const float d = q5_d * yb.d;
            const float dmin = -q5_dmin * yb.d;
            sums[static_cast<size_t>(m)] +=
                d * MoEHSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}
#endif

bool ComputeMoEQ5KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride, int M,
                                int K, float* out_sums) {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (ggml_cpu_has_avx2() &&
        ComputeMoEQ5KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, K, out_sums)) {
        return true;
    }
#endif
    return ComputeMoEQ5KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, K, out_sums);
}

bool RunMoEQ4KRawBatchedProjectionImpl(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                       size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                       int numa_node, bool allow_parallel) {
    if (!backend || !weight_ptr || !qinput_data || !out_data || M <= 0 || M > kMoEQuantizedProjectionMaxBatch ||
        N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float sums[kMoEQ4KRawBatchedTileM];
        for (int n = n_start; n < n_end; ++n) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const void* weight_row = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
                const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
                const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRow(weight_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sums[m];
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    const bool success = ok.load(std::memory_order_relaxed);
    if (success) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        RecordMoEKQuantRawBatchedUse(
            GetCurrentWorkContext(), GGML_TYPE_Q4_K,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            /*qwen_native_w2_q5k=*/false);
    }
    return success;
}

bool RunMoEKQuantRawBatchedProjectionImpl(CpuBackend* backend, ggml_type weight_type, const void* weight_ptr,
                                          const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data,
                                          int64_t M, int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    if (!backend || !weight_ptr || !qinput_data || !out_data || M <= 0 || M > kMoEQuantizedProjectionMaxBatch ||
        N <= 0 || K <= 0 || !ggml_is_quantized(weight_type)) {
        return false;
    }
    const auto* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K || K % ggml_blck_size(weight_type) != 0) {
        return false;
    }
    if (qinput_row_bytes < ggml_row_size(GGML_TYPE_Q8_K, K)) {
        return false;
    }

    const auto begin = std::chrono::steady_clock::now();
    const size_t weight_row_bytes = ggml_row_size(weight_type, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    std::atomic<bool> ok{true};

    if (weight_type == GGML_TYPE_Q5_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q5_K) == 0 &&
        qinput_row_bytes >= ggml_row_size(GGML_TYPE_Q8_K, K)) {
        const auto compute_rows = [&](int n_start, int n_end) {
            alignas(64) std::array<float, kMoEQuantizedProjectionMaxBatch> sums{};
            for (int n = n_start; n < n_end; ++n) {
                if (!ok.load(std::memory_order_relaxed)) {
                    return;
                }
                const void* row_ptr = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                if (!ComputeMoEQ5KQ8KBatchedRow(row_ptr, qinput_data, qinput_row_bytes, static_cast<int>(M),
                                                static_cast<int>(K), sums.data())) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int64_t m = 0; m < M; ++m) {
                    out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        sums[static_cast<size_t>(m)];
                }
            }
        };
        if (n_threads <= 1 || N < 64) {
            compute_rows(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
        }
        const bool success = ok.load(std::memory_order_relaxed);
        if (success) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            RecordMoEKQuantRawBatchedUse(
                GetCurrentWorkContext(), weight_type,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                /*qwen_native_w2_q5k=*/false);
        }
        return success;
    }

    const bool is_k_quant_weight =
        weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q5_K || weight_type == GGML_TYPE_Q6_K;
    const bool use_pair_vecdot = std::max<int>(1, static_cast<int>(traits->nrows)) >= 2 &&
                                 (!is_k_quant_weight || densecore::kernels::KQuantVecDotRowPairSupported());
    const int64_t pair_count = N / 2;
    const auto compute_pair_range = [&](int pair_start, int pair_end) {
        for (int pair = pair_start; pair < pair_end; ++pair) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const int64_t n = static_cast<int64_t>(pair) * 2;
            const void* row_ptr = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m = 0; m < M; ++m) {
                const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
                float sums[4] = {};
                traits->vec_dot(static_cast<int>(K), sums, 2, row_ptr, weight_row_bytes, qrow, 0, 2);
                out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sums[0];
                out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n + 1)] = sums[1];
            }
        }
    };

    if (use_pair_vecdot && pair_count > 0) {
        if (n_threads <= 1 || pair_count < 32) {
            compute_pair_range(0, static_cast<int>(pair_count));
        } else {
            pool.ParallelFor(static_cast<int>(pair_count),
                             [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
        }
    } else {
        const auto compute_row_range = [&](int n_start, int n_end) {
            for (int n = n_start; n < n_end; ++n) {
                const void* row_ptr = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                for (int64_t m = 0; m < M; ++m) {
                    const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
                    traits->vec_dot(static_cast<int>(K),
                                    out_data + static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n),
                                    0, row_ptr, 0, qrow, 0, 1);
                }
            }
        };
        if (n_threads <= 1 || N < 64) {
            compute_row_range(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N),
                             [&](int n_start, int n_end, int) { compute_row_range(n_start, n_end); });
        }
        const bool success = ok.load(std::memory_order_relaxed);
        if (success) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            RecordMoEKQuantRawBatchedUse(
                GetCurrentWorkContext(), weight_type,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                /*qwen_native_w2_q5k=*/false);
        }
        return success;
    }
    if ((N & 1) != 0) {
        const int64_t n = N - 1;
        const void* row_ptr = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
        for (int64_t m = 0; m < M; ++m) {
            const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
            float sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &sum, 0, row_ptr, 0, qrow, 0, 1);
            out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
    const bool success = ok.load(std::memory_order_relaxed);
    if (success) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        RecordMoEKQuantRawBatchedUse(
            GetCurrentWorkContext(), weight_type,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            /*qwen_native_w2_q5k=*/false);
    }
    return success;
}

bool RunMoEQ4KRawBatchedFusedSwiGLUImpl(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                        const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                        int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !out_data || M <= 0 ||
        M > kMoEQuantizedProjectionMaxBatch || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float gate_sums[kMoEQ4KRawBatchedTileM];
        alignas(64) float up_sums[kMoEQ4KRawBatchedTileM];
        for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
            const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
            const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
            for (int n = n_start; n < n_end; ++n) {
                if (!ok.load(std::memory_order_relaxed)) {
                    return;
                }
                const void* gate_row =
                    static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                const void* up_row =
                    static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRowPair(gate_row, up_row, q_tile, qinput_row_bytes, tile_m,
                                                    static_cast<int>(K), gate_sums, up_sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    const float gate = gate_sums[m];
                    const float up = up_sums[m];
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        (gate / (1.0f + internal::FastExp(-gate))) * up;
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    const bool success = ok.load(std::memory_order_relaxed);
    if (success) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        RecordMoEKQuantRawBatchedUse(
            GetCurrentWorkContext(), GGML_TYPE_Q4_K,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            /*qwen_native_w2_q5k=*/false);
    }
    return success;
}

bool RunMoEKQuantRawBatchedFusedSwiGLUImpl(CpuBackend* backend, ggml_type weight_type, const void* gate_weight_ptr,
                                           const void* up_weight_ptr, const uint8_t* qinput_data,
                                           size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                           int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !out_data || M <= 0 ||
        M > kMoEQuantizedProjectionMaxBatch || N <= 0 || K <= 0 || !ggml_is_quantized(weight_type)) {
        return false;
    }
    const auto* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K || K % ggml_blck_size(weight_type) != 0) {
        return false;
    }

    const auto begin = std::chrono::steady_clock::now();
    const size_t weight_row_bytes = ggml_row_size(weight_type, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    std::atomic<bool> ok{true};
    const auto log_path = [&]() {
        const char* path = weight_type == GGML_TYPE_Q5_K   ? "ggml_q5k_raw_batched_fused_swiglu"
                           : weight_type == GGML_TYPE_Q5_1 ? "ggml_q5_1_raw_batched_fused_swiglu"
                                                           : "ggml_kquant_raw_batched_fused_swiglu";
        LogMoEMatmulPath(path, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
    };

    const bool use_pair_vecdot = std::max<int>(1, static_cast<int>(traits->nrows)) >= 2;
    const int64_t pair_count = N / 2;
    if (weight_type == GGML_TYPE_Q5_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q5_K) == 0 &&
        qinput_row_bytes >= ggml_row_size(GGML_TYPE_Q8_K, K)) {
        const auto compute_rows = [&](int n_start, int n_end) {
            alignas(64) std::array<float, kMoEQuantizedProjectionMaxBatch> gate_sums{};
            alignas(64) std::array<float, kMoEQuantizedProjectionMaxBatch> up_sums{};
            for (int n = n_start; n < n_end; ++n) {
                if (!ok.load(std::memory_order_relaxed)) {
                    return;
                }
                const void* gate_row =
                    static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                const void* up_row =
                    static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                if (!ComputeMoEQ5KQ8KBatchedRow(gate_row, qinput_data, qinput_row_bytes, static_cast<int>(M),
                                                static_cast<int>(K), gate_sums.data()) ||
                    !ComputeMoEQ5KQ8KBatchedRow(up_row, qinput_data, qinput_row_bytes, static_cast<int>(M),
                                                static_cast<int>(K), up_sums.data())) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int64_t m = 0; m < M; ++m) {
                    const float gate = gate_sums[static_cast<size_t>(m)];
                    const float up = up_sums[static_cast<size_t>(m)];
                    out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        (gate / (1.0f + internal::FastExp(-gate))) * up;
                }
            }
        };
        if (n_threads <= 1 || N < 64) {
            compute_rows(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
        }
        if (ok.load(std::memory_order_relaxed)) {
            log_path();
        }
        const bool success = ok.load(std::memory_order_relaxed);
        if (success) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            RecordMoEKQuantRawBatchedUse(
                GetCurrentWorkContext(), weight_type,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                /*qwen_native_w2_q5k=*/false);
        }
        return success;
    }
    const auto compute_pair_range = [&](int pair_start, int pair_end) {
        for (int pair = pair_start; pair < pair_end; ++pair) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const int64_t n = static_cast<int64_t>(pair) * 2;
            const void* gate_row =
                static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            const void* up_row = static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m = 0; m < M; ++m) {
                const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
                float gate_sums[32] = {};
                float up_sums[32] = {};
                traits->vec_dot(static_cast<int>(K), gate_sums, 2, gate_row, weight_row_bytes, qrow, 0, 2);
                traits->vec_dot(static_cast<int>(K), up_sums, 2, up_row, weight_row_bytes, qrow, 0, 2);
                out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                    (gate_sums[0] / (1.0f + internal::FastExp(-gate_sums[0]))) * up_sums[0];
                out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n + 1)] =
                    (gate_sums[1] / (1.0f + internal::FastExp(-gate_sums[1]))) * up_sums[1];
            }
        }
    };

    if (use_pair_vecdot && pair_count > 0) {
        if (n_threads <= 1 || pair_count < 32) {
            compute_pair_range(0, static_cast<int>(pair_count));
        } else {
            pool.ParallelFor(static_cast<int>(pair_count),
                             [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
        }
    } else {
        const auto compute_row_range = [&](int n_start, int n_end) {
            for (int n = n_start; n < n_end; ++n) {
                const void* gate_row =
                    static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                const void* up_row =
                    static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                for (int64_t m = 0; m < M; ++m) {
                    const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
                    float gate_sum = 0.0f;
                    float up_sum = 0.0f;
                    traits->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qrow, 0, 1);
                    traits->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qrow, 0, 1);
                    out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
                }
            }
        };
        if (n_threads <= 1 || N < 64) {
            compute_row_range(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N),
                             [&](int n_start, int n_end, int) { compute_row_range(n_start, n_end); });
        }
        if (ok.load(std::memory_order_relaxed)) {
            log_path();
        }
        const bool success = ok.load(std::memory_order_relaxed);
        if (success) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            RecordMoEKQuantRawBatchedUse(
                GetCurrentWorkContext(), weight_type,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                /*qwen_native_w2_q5k=*/false);
        }
        return success;
    }
    if ((N & 1) != 0) {
        const int64_t n = N - 1;
        const void* gate_row = static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
        const void* up_row = static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
        for (int64_t m = 0; m < M; ++m) {
            const uint8_t* qrow = qinput_data + static_cast<size_t>(m) * qinput_row_bytes;
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qrow, 0, 1);
            traits->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qrow, 0, 1);
            out_data[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
        }
    }
    if (ok.load(std::memory_order_relaxed)) {
        log_path();
    }
    const bool success = ok.load(std::memory_order_relaxed);
    if (success) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        RecordMoEKQuantRawBatchedUse(
            GetCurrentWorkContext(), weight_type,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            /*qwen_native_w2_q5k=*/false);
    }
    return success;
}

bool RunMoEQ4KRawBatchedFusedGEGLU(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                   const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                   int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !out_data || M <= 0 ||
        M > kMoEQuantizedProjectionMaxBatch || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float gate_sums[kMoEQ4KRawBatchedTileM];
        alignas(64) float up_sums[kMoEQ4KRawBatchedTileM];
        for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
            const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
            const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
            for (int n = n_start; n < n_end; ++n) {
                if (!ok.load(std::memory_order_relaxed)) {
                    return;
                }
                const void* gate_row =
                    static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                const void* up_row =
                    static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRowPair(gate_row, up_row, q_tile, qinput_row_bytes, tile_m,
                                                    static_cast<int>(K), gate_sums, up_sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        GeluTanhApprox(gate_sums[m]) * up_sums[m];
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    return ok.load(std::memory_order_relaxed);
}
