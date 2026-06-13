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

struct Q8RepackedGemvWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    std::vector<uint8_t> data;
};

struct Q8RepackedGemvKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    bool operator==(const Q8RepackedGemvKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols;
    }
};

struct Q8RepackedGemvKeyHash {
    size_t operator()(const Q8RepackedGemvKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

constexpr int kQ8RepackedGemvMinOutputRows = 4096;

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
static inline int DenseCoreQ8_0DotPacked4x8RowsDotprodNeon(const int8_t* lhs_qs, int lhs_row,
                                                           const int8_t* rhs_qs, int rhs_row) {
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
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const auto* group_base = packed_base + static_cast<size_t>(group) * static_cast<size_t>(nb) * packed_block_bytes;
        for (int b = 0; b < nb; ++b) {
            const auto* block_base = group_base + static_cast<size_t>(b) * packed_block_bytes;
            const auto* scales = reinterpret_cast<const ggml_fp16_t*>(block_base);
            const auto* qs = reinterpret_cast<const int8_t*>(block_base + 4 * sizeof(ggml_fp16_t));
            const block_q8_0& x = input_blocks[b];
            const float input_scale = DenseCoreFp16ToFp32Fast(x.d);
            const float row_scale[4] = {
                DenseCoreFp16ToFp32Fast(scales[0]),
                DenseCoreFp16ToFp32Fast(scales[1]),
                DenseCoreFp16ToFp32Fast(scales[2]),
                DenseCoreFp16ToFp32Fast(scales[3]),
            };
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
            sum[0] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowAVX2(qs, 0, x.qs)) * row_scale[0] *
                      input_scale;
            sum[1] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowAVX2(qs, 1, x.qs)) * row_scale[1] *
                      input_scale;
            sum[2] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowAVX2(qs, 2, x.qs)) * row_scale[2] *
                      input_scale;
            sum[3] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowAVX2(qs, 3, x.qs)) * row_scale[3] *
                      input_scale;
#elif defined(DENSECORE_Q8_4X8_NEON_DOTPROD)
            sum[0] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 0, x.qs)) * row_scale[0] *
                      input_scale;
            sum[1] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 1, x.qs)) * row_scale[1] *
                      input_scale;
            sum[2] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 2, x.qs)) * row_scale[2] *
                      input_scale;
            sum[3] += static_cast<float>(DenseCoreQ8_0DotPacked4x8RowDotprodNeon(qs, 3, x.qs)) * row_scale[3] *
                      input_scale;
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
    DenseCoreForEachQ8_0_4x8Q8_0DotGeneric(
        n, packed_weight, q8_input, nc, 0,
        [&](int row, float value) { out[static_cast<size_t>(row)] = value; });
}

static inline int DenseCoreQ8_0DotPacked4x8Rows(const int8_t* lhs_qs, int lhs_row, const int8_t* rhs_qs,
                                                int rhs_row) {
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
        sum += DenseCoreQ8_0Dot8I8I8(lhs_qs + chunk * 4 * 8 + lhs_row * 8,
                                     rhs_qs + chunk * 4 * 8 + rhs_row * 8);
    }
    return sum;
#endif
}

static void DenseCoreGemmQ8_0_4x8x4Q8_0Generic(int n, float* out, int64_t out_row_stride,
                                               const void* packed_weight, const void* packed_input, int nc) {
    if (!out || !packed_weight || !packed_input || n <= 0 || (n % QK8_0) != 0 || nc <= 0 || (nc % 4) != 0 ||
        out_row_stride <= 0) {
        return;
    }
    const int nb = n / QK8_0;
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const auto* weight_base = static_cast<const uint8_t*>(packed_weight);
    const auto* input_base = static_cast<const uint8_t*>(packed_input);

    for (int group = 0; group < nc / 4; ++group) {
        float sum[4][4] = {};
        const auto* weight_group =
            weight_base + static_cast<size_t>(group) * static_cast<size_t>(nb) * packed_block_bytes;
        for (int b = 0; b < nb; ++b) {
            const auto* weight_block = weight_group + static_cast<size_t>(b) * packed_block_bytes;
            const auto* input_block = input_base + static_cast<size_t>(b) * packed_block_bytes;
            const auto* weight_scales = reinterpret_cast<const ggml_fp16_t*>(weight_block);
            const auto* input_scales = reinterpret_cast<const ggml_fp16_t*>(input_block);
            const auto* weight_qs = reinterpret_cast<const int8_t*>(weight_block + 4 * sizeof(ggml_fp16_t));
            const auto* input_qs = reinterpret_cast<const int8_t*>(input_block + 4 * sizeof(ggml_fp16_t));
            float w_scale[4];
            float x_scale[4];
            for (int row = 0; row < 4; ++row) {
                w_scale[row] = DenseCoreFp16ToFp32Fast(weight_scales[row]);
                x_scale[row] = DenseCoreFp16ToFp32Fast(input_scales[row]);
            }
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
            // i8mm (smmla) 4x4 micro-kernel — what llama.cpp uses for this shape.
            // The 4x8 pack is i8mm-native: out-rows {0,1} of a chunk are 16
            // contiguous int8 at qs+chunk*32, {2,3} at +16 (same for the 4 input
            // rows). vmmlaq_s32(a,b) accumulates the 2x2 tile [a0·b0, a0·b1, a1·b0,
            // a1·b1]. Integer-exact vs the scalar/dotprod path; accumulate the
            // block's 4 chunks, then apply per-block scales (sum[m][row] +=
            // dot * w_scale[row] * x_scale[m]). w0..3 = weight rows, m0..3 = input rows.
            int32x4_t acc_w01x01 = vdupq_n_s32(0);  // (w0·m0, w0·m1, w1·m0, w1·m1)
            int32x4_t acc_w01x23 = vdupq_n_s32(0);  // (w0·m2, w0·m3, w1·m2, w1·m3)
            int32x4_t acc_w23x01 = vdupq_n_s32(0);  // (w2·m0, w2·m1, w3·m0, w3·m1)
            int32x4_t acc_w23x23 = vdupq_n_s32(0);  // (w2·m2, w2·m3, w3·m2, w3·m3)
            for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                const int8x16_t w01 = vld1q_s8(weight_qs + chunk * 32);
                const int8x16_t w23 = vld1q_s8(weight_qs + chunk * 32 + 16);
                const int8x16_t x01 = vld1q_s8(input_qs + chunk * 32);
                const int8x16_t x23 = vld1q_s8(input_qs + chunk * 32 + 16);
                acc_w01x01 = vmmlaq_s32(acc_w01x01, w01, x01);
                acc_w01x23 = vmmlaq_s32(acc_w01x23, w01, x23);
                acc_w23x01 = vmmlaq_s32(acc_w23x01, w23, x01);
                acc_w23x23 = vmmlaq_s32(acc_w23x23, w23, x23);
            }
            sum[0][0] += static_cast<float>(vgetq_lane_s32(acc_w01x01, 0)) * w_scale[0] * x_scale[0];
            sum[1][0] += static_cast<float>(vgetq_lane_s32(acc_w01x01, 1)) * w_scale[0] * x_scale[1];
            sum[0][1] += static_cast<float>(vgetq_lane_s32(acc_w01x01, 2)) * w_scale[1] * x_scale[0];
            sum[1][1] += static_cast<float>(vgetq_lane_s32(acc_w01x01, 3)) * w_scale[1] * x_scale[1];
            sum[2][0] += static_cast<float>(vgetq_lane_s32(acc_w01x23, 0)) * w_scale[0] * x_scale[2];
            sum[3][0] += static_cast<float>(vgetq_lane_s32(acc_w01x23, 1)) * w_scale[0] * x_scale[3];
            sum[2][1] += static_cast<float>(vgetq_lane_s32(acc_w01x23, 2)) * w_scale[1] * x_scale[2];
            sum[3][1] += static_cast<float>(vgetq_lane_s32(acc_w01x23, 3)) * w_scale[1] * x_scale[3];
            sum[0][2] += static_cast<float>(vgetq_lane_s32(acc_w23x01, 0)) * w_scale[2] * x_scale[0];
            sum[1][2] += static_cast<float>(vgetq_lane_s32(acc_w23x01, 1)) * w_scale[2] * x_scale[1];
            sum[0][3] += static_cast<float>(vgetq_lane_s32(acc_w23x01, 2)) * w_scale[3] * x_scale[0];
            sum[1][3] += static_cast<float>(vgetq_lane_s32(acc_w23x01, 3)) * w_scale[3] * x_scale[1];
            sum[2][2] += static_cast<float>(vgetq_lane_s32(acc_w23x23, 0)) * w_scale[2] * x_scale[2];
            sum[3][2] += static_cast<float>(vgetq_lane_s32(acc_w23x23, 1)) * w_scale[2] * x_scale[3];
            sum[2][3] += static_cast<float>(vgetq_lane_s32(acc_w23x23, 2)) * w_scale[3] * x_scale[2];
            sum[3][3] += static_cast<float>(vgetq_lane_s32(acc_w23x23, 3)) * w_scale[3] * x_scale[3];
#else
            for (int m = 0; m < 4; ++m) {
                for (int row = 0; row < 4; ++row) {
                    const int dot = DenseCoreQ8_0DotPacked4x8Rows(weight_qs, row, input_qs, m);
                    sum[m][row] += static_cast<float>(dot) * w_scale[row] * x_scale[m];
                }
            }
#endif
        }
        for (int m = 0; m < 4; ++m) {
            float* out_row = out + static_cast<size_t>(m) * static_cast<size_t>(out_row_stride) +
                             static_cast<size_t>(group) * 4;
            out_row[0] = sum[m][0];
            out_row[1] = sum[m][1];
            out_row[2] = sum[m][2];
            out_row[3] = sum[m][3];
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
        const auto* row0 = reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 0) * q8_row_stride);
        const auto* row1 = reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 1) * q8_row_stride);
        const auto* row2 = reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 2) * q8_row_stride);
        const auto* row3 = reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m + 3) * q8_row_stride);
        const block_q8_0* rows_in[4] = {row0, row1, row2, row3};
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* dst = packed.data() +
                           (static_cast<size_t>(m / 4) * static_cast<size_t>(blocks_per_row) +
                            static_cast<size_t>(b)) *
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

static void DenseCoreClearQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->q8_gemm_packed_generation = 0;
    ctx->q8_gemm_packed_tensor = nullptr;
    ctx->q8_gemm_packed_source = nullptr;
    ctx->q8_gemm_packed_rows = 0;
    ctx->q8_gemm_packed_cols = 0;
    ctx->q8_gemm_packed_bytes = 0;
    ctx->q8_gemm_packed_token_pos = std::numeric_limits<int64_t>::min();
    ctx->q8_gemm_packed_buffer.clear();
}

static constexpr bool DenseCoreQ8_0Gemm4x8FastBackendCompiled() {
    return true;
}

static bool DenseCoreValidateQ8_0RowsTo4x8ActivationCache(const InferenceWorkContext* ctx,
                                                          const ggml_tensor* src_tensor, const void* source, int rows,
                                                          int n, size_t packed_bytes, int64_t token_pos) {
    return ctx && src_tensor && source && rows > 0 && n > 0 && packed_bytes > 0 &&
           ctx->q8_gemm_packed_generation == ctx->execution_generation &&
           ctx->q8_gemm_packed_tensor == src_tensor && ctx->q8_gemm_packed_source == source &&
           ctx->q8_gemm_packed_rows == rows && ctx->q8_gemm_packed_cols == n &&
           ctx->q8_gemm_packed_bytes == packed_bytes && ctx->q8_gemm_packed_token_pos == token_pos &&
           ctx->q8_gemm_packed_buffer.size() == packed_bytes;
}

static const uint8_t* GetOrFillQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx,
                                                            const ggml_tensor* src_tensor, const void* source,
                                                            const uint8_t* q8_input_base, size_t q8_row_stride,
                                                            int rows, int n, int64_t token_pos) {
    const size_t packed_bytes = DenseCoreQ8_0RowsTo4x8PackedBytes(rows, n);
    if (!ctx || !src_tensor || !source || !q8_input_base || q8_row_stride == 0 || packed_bytes == 0) {
        return nullptr;
    }
    if (DenseCoreValidateQ8_0RowsTo4x8ActivationCache(ctx, src_tensor, source, rows, n, packed_bytes, token_pos)) {
        return ctx->q8_gemm_packed_buffer.data();
    }
    if (!DenseCorePackQ8_0RowsTo4x8(q8_input_base, q8_row_stride, rows, n, ctx->q8_gemm_packed_buffer)) {
        DenseCoreClearQ8_0RowsTo4x8ActivationCache(ctx);
        return nullptr;
    }
    ctx->q8_gemm_packed_generation = ctx->execution_generation;
    ctx->q8_gemm_packed_tensor = src_tensor;
    ctx->q8_gemm_packed_source = source;
    ctx->q8_gemm_packed_rows = rows;
    ctx->q8_gemm_packed_cols = n;
    ctx->q8_gemm_packed_bytes = packed_bytes;
    ctx->q8_gemm_packed_token_pos = token_pos;
    return ctx->q8_gemm_packed_buffer.data();
}

static inline float DenseCoreQ8_0BlockDot(const block_q8_0* weight_blocks, const block_q8_0* input_blocks,
                                          int n) {
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
        const int dot = DenseCoreQ8_0Dot8I8I8(w.qs + 0, x.qs + 0) +
                        DenseCoreQ8_0Dot8I8I8(w.qs + 8, x.qs + 8) +
                        DenseCoreQ8_0Dot8I8I8(w.qs + 16, x.qs + 16) +
                        DenseCoreQ8_0Dot8I8I8(w.qs + 24, x.qs + 24);
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

enum class Q4KRepackedGemvRejectReason : int {
    None = 0,
    EnvOff = 1,
    UnsupportedIsa = 2,
    NotDecode = 3,
    NotQ4K = 4,
    DynamicLora = 5,
    Shape = 6,
    MissingVecDot = 7,
    Cache = 8,
    RealKernelUnavailable = 9,
    CopiedExperimentDisabled = 10,
    ProbeFailed = 11,
    ReferenceForced = 12,
    CacheThrashing = 13,
    CacheLimitTooSmall = 14,
    WorkingSetExceedsCache = 15,
    RepeatedRepack = 16,
    EvictionRatioHigh = 17,
    RepackBytesHigh = 18,
};

const char* Q4KRepackedGemvRejectReasonName(int reason) {
    switch (static_cast<Q4KRepackedGemvRejectReason>(reason)) {
        case Q4KRepackedGemvRejectReason::None:
            return "none";
        case Q4KRepackedGemvRejectReason::EnvOff:
            return "env_off";
        case Q4KRepackedGemvRejectReason::UnsupportedIsa:
            return "unsupported_isa";
        case Q4KRepackedGemvRejectReason::NotDecode:
            return "not_decode";
        case Q4KRepackedGemvRejectReason::NotQ4K:
            return "not_q4k";
        case Q4KRepackedGemvRejectReason::DynamicLora:
            return "dynamic_lora";
        case Q4KRepackedGemvRejectReason::Shape:
            return "shape";
        case Q4KRepackedGemvRejectReason::MissingVecDot:
            return "missing_vec_dot";
        case Q4KRepackedGemvRejectReason::Cache:
            return "cache";
        case Q4KRepackedGemvRejectReason::RealKernelUnavailable:
            return "real_kernel_unavailable";
        case Q4KRepackedGemvRejectReason::CopiedExperimentDisabled:
            return "copied_experiment_disabled";
        case Q4KRepackedGemvRejectReason::ProbeFailed:
            return "probe_failed";
        case Q4KRepackedGemvRejectReason::ReferenceForced:
            return "reference_forced";
        case Q4KRepackedGemvRejectReason::CacheThrashing:
            return "cache_thrashing";
        case Q4KRepackedGemvRejectReason::CacheLimitTooSmall:
            return "cache_limit_too_small";
        case Q4KRepackedGemvRejectReason::WorkingSetExceedsCache:
            return "working_set_exceeds_cache";
        case Q4KRepackedGemvRejectReason::RepeatedRepack:
            return "repeated_repack";
        case Q4KRepackedGemvRejectReason::EvictionRatioHigh:
            return "eviction_ratio_high";
        case Q4KRepackedGemvRejectReason::RepackBytesHigh:
            return "repack_bytes_high";
    }
    return "unknown";
}

bool Q4KRepackedGemvRejectReasonIsCacheThrash(int reason) {
    switch (static_cast<Q4KRepackedGemvRejectReason>(reason)) {
        case Q4KRepackedGemvRejectReason::CacheThrashing:
        case Q4KRepackedGemvRejectReason::CacheLimitTooSmall:
        case Q4KRepackedGemvRejectReason::WorkingSetExceedsCache:
        case Q4KRepackedGemvRejectReason::RepeatedRepack:
        case Q4KRepackedGemvRejectReason::EvictionRatioHigh:
        case Q4KRepackedGemvRejectReason::RepackBytesHigh:
            return true;
        default:
            return false;
    }
}

enum class GemvCustomTaskCapReason : int {
    Unknown = 0,
    PhysicalCore = 1,
    PerformanceProfileConfiguredThreads = 2,
    SmallK64 = 3,
    SmallK512 = 4,
    SmallK1536 = 5,
    SmallK3072 = 6,
    DecodeProjection = 7,
};

const char* GemvCustomTaskCapReasonName(int reason) {
    switch (static_cast<GemvCustomTaskCapReason>(reason)) {
        case GemvCustomTaskCapReason::Unknown:
            return "unknown";
        case GemvCustomTaskCapReason::PhysicalCore:
            return "physical_cores";
        case GemvCustomTaskCapReason::PerformanceProfileConfiguredThreads:
            return "perf_profile_configured_threads";
        case GemvCustomTaskCapReason::SmallK64:
            return "small_k_lt_64";
        case GemvCustomTaskCapReason::SmallK512:
            return "small_k_lt_512";
        case GemvCustomTaskCapReason::SmallK1536:
            return "small_k_lt_1536";
        case GemvCustomTaskCapReason::SmallK3072:
            return "small_k_lt_3072";
        case GemvCustomTaskCapReason::DecodeProjection:
            return "decode_projection";
    }
    return "unknown";
}

enum class Qwen36SSMQ8PrefillAMXRejectReason : int {
    None = 0,
    EnvOff = 1,
    NotQwen36HybridSsm = 2,
    NotPrefill = 3,
    NotSSMProjection = 4,
    NotQ8_0 = 5,
    DynamicLora = 6,
    BackendUnavailable = 7,
    AliasUnavailable = 8,
    ProbeUnavailable = 9,
    Admitted = 10,
    DecodeOriginalQ8 = 11,
    ResidentDecodeRegressionRisk = 12,
    PhaseUnknown = 13,
};

const char* Qwen36SSMQ8PrefillAMXRejectReasonName(int reason) {
    switch (static_cast<Qwen36SSMQ8PrefillAMXRejectReason>(reason)) {
        case Qwen36SSMQ8PrefillAMXRejectReason::None:
            return "none";
        case Qwen36SSMQ8PrefillAMXRejectReason::EnvOff:
            return "env_off";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotQwen36HybridSsm:
            return "not_qwen36_hybrid_ssm";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotPrefill:
            return "not_prefill";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotSSMProjection:
            return "not_ssm_projection";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotQ8_0:
            return "not_q8_0";
        case Qwen36SSMQ8PrefillAMXRejectReason::DynamicLora:
            return "dynamic_lora";
        case Qwen36SSMQ8PrefillAMXRejectReason::BackendUnavailable:
            return "backend_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::AliasUnavailable:
            return "alias_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::ProbeUnavailable:
            return "probe_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::Admitted:
            return "admitted";
        case Qwen36SSMQ8PrefillAMXRejectReason::DecodeOriginalQ8:
            return "decode_original_q8";
        case Qwen36SSMQ8PrefillAMXRejectReason::ResidentDecodeRegressionRisk:
            return "resident_decode_regression_risk";
        case Qwen36SSMQ8PrefillAMXRejectReason::PhaseUnknown:
            return "phase_unknown";
    }
    return "unknown";
}

static Qwen36SSMQ8PrefillAMXRejectReason ResolveQwen36SSMQ8PrefillAMXReason(
    densecore::llm::config::Qwen36SSMQ8PrefillAMXMode mode, InferenceExecutionPhase phase, bool lora_active) {
    if (mode == densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off) {
        return Qwen36SSMQ8PrefillAMXRejectReason::EnvOff;
    }
    if (phase == InferenceExecutionPhase::Unknown) {
        return Qwen36SSMQ8PrefillAMXRejectReason::PhaseUnknown;
    }
    if (phase != InferenceExecutionPhase::Prefill) {
        return Qwen36SSMQ8PrefillAMXRejectReason::NotPrefill;
    }
    if (lora_active) {
        return Qwen36SSMQ8PrefillAMXRejectReason::DynamicLora;
    }
    return Qwen36SSMQ8PrefillAMXRejectReason::None;
}

static void RecordQwen36SSMQ8PrefillAMXReject(InferenceWorkContext* ctx,
                                               Qwen36SSMQ8PrefillAMXRejectReason reason) {
    if (!ctx || reason == Qwen36SSMQ8PrefillAMXRejectReason::None) {
        return;
    }
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_last_reject_reason.store(static_cast<int>(reason),
                                                                           std::memory_order_relaxed);
}

static void RecordQwen36SSMQ8PrefillAMXUsed(InferenceWorkContext* ctx, int projection_kind) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_used.store(1, std::memory_order_relaxed);
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_used_ops.fetch_add(1, std::memory_order_relaxed);
    switch (projection_kind) {
        case 1:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_qkv_count.fetch_add(1, std::memory_order_relaxed);
            break;
        case 2:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_gate_count.fetch_add(1, std::memory_order_relaxed);
            break;
        case 3:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_out_count.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

static bool Q4KRepackedGemvEnabled(const densecore::llm::config::FastPathRuntimeConfig& config,
                                   Q4KRepackedGemvRejectReason* reject_reason) {
    if (config.q4k_repacked_gemv == densecore::env::RuntimeToggleMode::Off) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::EnvOff;
        return false;
    }
    const bool supported = densecore::kernels::Q4KRepackedGemvIsaSupported();
    if (!supported) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::UnsupportedIsa;
        return false;
    }
    if (!densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::RealKernelUnavailable;
        return false;
    }
    return true;
}

static inline void RecordQ4KRepackedGemvCacheLookup(InferenceWorkContext* work_ctx,
                                                    const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup) {
    if (!work_ctx) {
        return;
    }
    auto& profile = work_ctx->qwen36_profile;
    if (lookup.waited) {
        profile.q4k_repacked_gemv_cache_waited_hits.fetch_add(1, std::memory_order_relaxed);
    } else if (lookup.cache_hit) {
        profile.q4k_repacked_gemv_cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        profile.q4k_repacked_gemv_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    if (lookup.cache_evictions != 0) {
        profile.q4k_repacked_gemv_cache_evictions.fetch_add(lookup.cache_evictions, std::memory_order_relaxed);
    }
    if (lookup.cache_evicted_bytes != 0) {
        profile.q4k_repacked_gemv_cache_evicted_bytes.fetch_add(lookup.cache_evicted_bytes,
                                                                std::memory_order_relaxed);
    }
    if (lookup.repack_bytes != 0) {
        profile.q4k_repacked_gemv_repack_bytes.fetch_add(lookup.repack_bytes, std::memory_order_relaxed);
    }
    if (lookup.resident_bytes != 0) {
        profile.q4k_repacked_gemv_resident_bytes.store(lookup.resident_bytes, std::memory_order_relaxed);
    }
    if (lookup.weight_key != 0) {
        std::lock_guard<std::mutex> lock(work_ctx->q4k_repacked_gemv_request_mutex);
        auto [it, inserted] = work_ctx->q4k_repacked_gemv_repack_counts.emplace(lookup.weight_key, 0);
        if (inserted) {
            profile.q4k_repacked_gemv_distinct_weights_seen.store(
                static_cast<uint64_t>(work_ctx->q4k_repacked_gemv_repack_counts.size()), std::memory_order_relaxed);
        }
        if (lookup.repacked) {
            ++it->second;
            if (it->second > 1) {
                profile.q4k_repacked_gemv_repeated_repack_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

static inline void SetQ4KRepackedGemvAutoDisable(InferenceWorkContext* work_ctx, Q4KRepackedGemvRejectReason local_reason,
                                                 Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || local_reason == Q4KRepackedGemvRejectReason::None) {
        return;
    }
    const int encoded = static_cast<int>(local_reason);
    int expected = 0;
    work_ctx->qwen36_profile.q4k_repacked_gemv_primary_disable_reason.compare_exchange_strong(
        expected, encoded, std::memory_order_relaxed);
    work_ctx->qwen36_profile.q4k_repacked_gemv_last_reject_reason.store(encoded, std::memory_order_relaxed);
    if (reason) {
        *reason = local_reason;
    }
}

static inline bool Q4KRepackedGemvShouldAutoDisableForLookup(
    InferenceWorkContext* work_ctx, const densecore::llm::config::FastPathRuntimeConfig& config,
    const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup, Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || !config.q4k_repacked_gemv_disable_on_thrash ||
        config.q4k_repacked_gemv != densecore::env::RuntimeToggleMode::Auto) {
        return false;
    }
    Q4KRepackedGemvRejectReason local_reason = Q4KRepackedGemvRejectReason::None;
    const uint64_t repeated_repack_count =
        work_ctx->qwen36_profile.q4k_repacked_gemv_repeated_repack_count.load(std::memory_order_relaxed);
    const uint64_t cache_evictions =
        work_ctx->qwen36_profile.q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed);
    const uint64_t used_ops = work_ctx->qwen36_profile.q4k_repacked_gemv_used_ops.load(std::memory_order_relaxed);
    const uint64_t repack_bytes =
        work_ctx->qwen36_profile.q4k_repacked_gemv_repack_bytes.load(std::memory_order_relaxed);
    const uint64_t repack_threshold_bytes =
        static_cast<uint64_t>(std::max(1, config.q4k_repacked_gemv_thrash_repack_mb)) * 1024ULL * 1024ULL;
    const uint64_t cache_fraction_threshold =
        lookup.cache_limit_bytes == 0
            ? 0
            : static_cast<uint64_t>(static_cast<double>(lookup.cache_limit_bytes) *
                                    config.q4k_repacked_gemv_thrash_repack_cache_fraction);
    const uint64_t eviction_ratio_denominator = std::max<uint64_t>(1, used_ops);

    if (repeated_repack_count > 0) {
        local_reason = Q4KRepackedGemvRejectReason::RepeatedRepack;
    } else if (lookup.working_set_exceeds_cache) {
        local_reason = Q4KRepackedGemvRejectReason::WorkingSetExceedsCache;
    } else if (lookup.cache_limit_too_small) {
        local_reason = Q4KRepackedGemvRejectReason::CacheLimitTooSmall;
    } else if (cache_evictions > 0 && repack_bytes >= repack_threshold_bytes &&
               static_cast<double>(cache_evictions) / static_cast<double>(eviction_ratio_denominator) >
                   config.q4k_repacked_gemv_thrash_eviction_ratio) {
        local_reason = Q4KRepackedGemvRejectReason::EvictionRatioHigh;
    } else if (repack_bytes >= repack_threshold_bytes &&
               cache_fraction_threshold > 0 && repack_bytes > cache_fraction_threshold) {
        local_reason = Q4KRepackedGemvRejectReason::RepackBytesHigh;
    } else if (lookup.repacked && lookup.cache_evictions != 0 &&
               work_ctx->qwen36_profile.q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed) >
                   lookup.cache_evictions) {
        local_reason = Q4KRepackedGemvRejectReason::CacheThrashing;
    }
    if (local_reason == Q4KRepackedGemvRejectReason::None) {
        return false;
    }
    SetQ4KRepackedGemvAutoDisable(work_ctx, local_reason, reason);
    return true;
}

struct Q4KRepackedGemvProbeKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const Q4KRepackedGemvProbeKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols && fingerprint == other.fingerprint;
    }
};

struct Q4KRepackedGemvProbeKeyHash {
    size_t operator()(const Q4KRepackedGemvProbeKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct Q4KRepackedGemvProbeEntry {
    bool running = false;
    bool done = false;
    bool passed = false;
    std::condition_variable cv;
};

static bool RunQ4KRepackedGemvProbe(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                                    const void* weight_data, const void* quant_input, int64_t rows, int64_t cols) {
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!packed || !type_traits_cpu || !type_traits_cpu->vec_dot || !weight_data || !quant_input || rows <= 0 ||
        cols <= 0) {
        return false;
    }
    const int tile_count = static_cast<int>(rows / 8);
    if (tile_count <= 0) {
        return false;
    }
    std::array<int, 3> sample_tiles{0, tile_count / 2, tile_count - 1};
    std::vector<float> probe(static_cast<size_t>(rows), 0.0f);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const auto* weight_base = static_cast<const uint8_t*>(weight_data);
    for (const int tile : sample_tiles) {
        const int clamped_tile = std::clamp(tile, 0, tile_count - 1);
        if (!densecore::kernels::RunQ4KRepackedGemv(packed, static_cast<const uint8_t*>(quant_input), q8_row_bytes,
                                                   probe.data(), rows, clamped_tile, clamped_tile + 1)) {
            return false;
        }
        const int row_begin = clamped_tile * 8;
        const int row_end = std::min(row_begin + 8, static_cast<int>(rows));
        for (int row = row_begin; row < row_end; ++row) {
            const void* row_ptr = weight_base + static_cast<size_t>(row) * q4_row_bytes;
            float reference = 0.0f;
            type_traits_cpu->vec_dot(cols, &reference, 0, row_ptr, 0, quant_input, 0, 1);
            const float diff = std::fabs(reference - probe[static_cast<size_t>(row)]);
            const float tol = std::max(1.0e-3f, 1.0e-3f * std::fabs(reference));
            if (!std::isfinite(reference) || !std::isfinite(probe[static_cast<size_t>(row)]) || diff > tol) {
                return false;
            }
        }
    }
    return true;
}

static bool Q4KRepackedGemvProbePassed(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                                       const void* weight_data, const void* quant_input, int64_t rows, int64_t cols) {
    if (!packed) {
        return false;
    }
    static std::mutex mutex;
    static std::unordered_map<Q4KRepackedGemvProbeKey, std::shared_ptr<Q4KRepackedGemvProbeEntry>,
                              Q4KRepackedGemvProbeKeyHash>
        decisions;
    static std::vector<Q4KRepackedGemvProbeKey> insertion_order;
    constexpr size_t max_entries = 4096;
    const Q4KRepackedGemvProbeKey key{
        weight_data, rows, cols,
        densecore::kernels::Q4KRepackedGemvWeightFingerprint(weight_data, rows, cols),
    };

    std::shared_ptr<Q4KRepackedGemvProbeEntry> entry;
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto found = decisions.find(key);
        if (found != decisions.end()) {
            entry = found->second;
            if (entry->done) {
                return entry->passed;
            }
            entry->cv.wait(lock, [&]() { return entry->done; });
            return entry->passed;
        }
        entry = std::make_shared<Q4KRepackedGemvProbeEntry>();
        entry->running = true;
        decisions.emplace(key, entry);
        insertion_order.push_back(key);
        while (decisions.size() > max_entries && !insertion_order.empty()) {
            auto victim = decisions.find(insertion_order.front());
            if (victim != decisions.end() && victim->second && victim->second->running) {
                break;
            }
            if (victim != decisions.end()) {
                decisions.erase(victim);
            }
            insertion_order.erase(insertion_order.begin());
        }
    }

    const bool passed = RunQ4KRepackedGemvProbe(packed, weight_data, quant_input, rows, cols);
    {
        std::lock_guard<std::mutex> lock(mutex);
        entry->passed = passed;
        entry->done = true;
        entry->running = false;
        entry->cv.notify_all();
    }
    return passed;
}

static void RecordQActCacheHit(InferenceWorkContext* ctx, size_t bytes) {
    if (!ctx) return;
    ctx->qwen36_profile.qact_cache_hits.fetch_add(1, std::memory_order_relaxed);
    ctx->qwen36_profile.qact_cache_reused_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static void RecordQActCacheMiss(InferenceWorkContext* ctx) {
    if (!ctx) return;
    ctx->qwen36_profile.qact_cache_misses.fetch_add(1, std::memory_order_relaxed);
}

static bool QuantizedActivationCacheEnabled(const densecore::llm::config::FastPathRuntimeConfig& config) {
    if (config.qact_cache == densecore::env::RuntimeToggleMode::Off) {
        return false;
    }
    return config.qact_cache == densecore::env::RuntimeToggleMode::On ||
           config.qact_cache == densecore::env::RuntimeToggleMode::Auto;
}

static bool QuantizedActivationKeyMatches(uint64_t generation, const ggml_tensor* tensor, const void* source,
                                          int64_t len, ggml_type type, size_t bytes, int slot_id,
                                          int64_t token_pos, size_t buffer_size,
                                          const InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                          const void* expected_source, int64_t expected_len,
                                          ggml_type expected_type, size_t expected_bytes,
                                          int expected_slot_id, int64_t expected_token_pos) {
    return ctx && generation == ctx->execution_generation && tensor == src_tensor && source == expected_source &&
           len == expected_len && type == expected_type && bytes == expected_bytes &&
           slot_id == expected_slot_id && token_pos == expected_token_pos && buffer_size == expected_bytes;
}

static const uint8_t* FindQuantizedActivationCacheEntry(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                        const void* source, int64_t len, ggml_type quant_type,
                                                        size_t quant_bytes, int slot_id, int64_t token_pos) {
    if (!ctx) {
        return nullptr;
    }
    if (QuantizedActivationKeyMatches(ctx->qact_generation, ctx->qact_tensor, ctx->qact_source, ctx->qact_len,
                                      ctx->qact_type, ctx->qact_bytes, ctx->qact_slot_id, ctx->qact_token_pos,
                                      ctx->qact_buffer.size(), ctx, src_tensor, source, len, quant_type,
                                      quant_bytes, slot_id, token_pos)) {
        RecordQActCacheHit(ctx, quant_bytes);
        return ctx->qact_buffer.data();
    }
    for (auto& slot : ctx->qact_extra_slots) {
        if (QuantizedActivationKeyMatches(slot.generation, slot.tensor, slot.source, slot.len, slot.type,
                                          slot.bytes, slot.slot_id, slot.token_pos, slot.buffer.size(), ctx,
                                          src_tensor, source, len, quant_type, quant_bytes, slot_id, token_pos)) {
            RecordQActCacheHit(ctx, quant_bytes);
            return slot.buffer.data();
        }
    }
    return nullptr;
}

static QuantizedActivationCacheEntry* SelectExtraQuantizedActivationSlot(InferenceWorkContext* ctx) {
    if (!ctx) {
        return nullptr;
    }
    for (auto& slot : ctx->qact_extra_slots) {
        if (slot.generation != ctx->execution_generation || slot.buffer.empty()) {
            return &slot;
        }
    }
    int idx = ctx->qact_extra_next_slot;
    if (idx < 0 || idx >= kExtraQuantizedActivationCacheSlots) {
        idx = 0;
    }
    ctx->qact_extra_next_slot = (idx + 1) % kExtraQuantizedActivationCacheSlots;
    return &ctx->qact_extra_slots[static_cast<size_t>(idx)];
}

static bool PrimaryQuantizedActivationSlotAvailable(const InferenceWorkContext* ctx) {
    return !ctx || ctx->qact_generation != ctx->execution_generation || ctx->qact_buffer.empty();
}

static void StoreExtraQuantizedActivationMetadata(QuantizedActivationCacheEntry* slot, InferenceWorkContext* ctx,
                                                  const ggml_tensor* src_tensor, const void* source, int64_t len,
                                                  ggml_type quant_type, size_t quant_bytes, int slot_id,
                                                  int64_t token_pos) {
    if (!slot || !ctx) {
        return;
    }
    slot->generation = ctx->execution_generation;
    slot->tensor = src_tensor;
    slot->source = source;
    slot->len = len;
    slot->type = quant_type;
    slot->bytes = quant_bytes;
    slot->slot_id = slot_id;
    slot->token_pos = token_pos;
}

static const uint8_t* GetOrFillQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                        const void* source, const float* x_f32, int64_t len,
                                                        ggml_type quant_type, size_t quant_bytes, int slot_id,
                                                        int64_t token_pos,
                                                        const ggml_type_traits_cpu* input_type_traits) {
    if (!ctx || !src_tensor || !source || !x_f32 || len <= 0 || quant_type == GGML_TYPE_F32 || quant_bytes == 0 ||
        !input_type_traits || !input_type_traits->from_float) {
        return nullptr;
    }
#ifndef NDEBUG
    const bool data_pointer_matches_different_tensor =
        ctx->qact_source == source && ctx->qact_tensor && ctx->qact_tensor != src_tensor;
    if (data_pointer_matches_different_tensor) {
        static const bool strict_qact_cache_assert = []() {
            const char* env = std::getenv("DENSECORE_DEBUG_QACT_CACHE_ASSERT");
            return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        }();
        if (strict_qact_cache_assert) {
            assert(ctx->qact_tensor == src_tensor && "qact cache data pointer reused by a different tensor");
        }
    }
#endif
    if (const uint8_t* cached =
            FindQuantizedActivationCacheEntry(ctx, src_tensor, source, len, quant_type, quant_bytes, slot_id,
                                              token_pos)) {
        return cached;
    }

    if (PrimaryQuantizedActivationSlotAvailable(ctx)) {
        ctx->qact_buffer.resize(quant_bytes);
        input_type_traits->from_float(x_f32, ctx->qact_buffer.data(), len);
        ctx->qact_tensor = src_tensor;
        ctx->qact_generation = ctx->execution_generation;
        ctx->qact_source = source;
        ctx->qact_len = len;
        ctx->qact_type = quant_type;
        ctx->qact_bytes = quant_bytes;
        ctx->qact_slot_id = slot_id;
        ctx->qact_token_pos = token_pos;
        RecordQActCacheMiss(ctx);
        return ctx->qact_buffer.data();
    }

    QuantizedActivationCacheEntry* slot = SelectExtraQuantizedActivationSlot(ctx);
    if (!slot) {
        return nullptr;
    }
    slot->buffer.resize(quant_bytes);
    input_type_traits->from_float(x_f32, slot->buffer.data(), len);
    StoreExtraQuantizedActivationMetadata(slot, ctx, src_tensor, source, len, quant_type, quant_bytes, slot_id,
                                          token_pos);
    RecordQActCacheMiss(ctx);
    return slot->buffer.data();
}

static const uint8_t* GetOrFillBatchedQuantizedActivationCache(
    InferenceWorkContext* ctx, const ggml_tensor* src_tensor, const void* source,
    const std::vector<const float*>& x_rows, int M, int N, ggml_type quant_type, size_t quant_row_stride,
    size_t quant_bytes, int64_t token_pos, const ggml_type_traits_cpu* input_type_traits) {
    if (!ctx || !src_tensor || !source || M <= 0 || N <= 0 || quant_type == GGML_TYPE_F32 || quant_row_stride == 0 ||
        quant_bytes == 0 || !input_type_traits || !input_type_traits->from_float ||
        x_rows.size() < static_cast<size_t>(M)) {
        return nullptr;
    }
    for (int m = 0; m < M; ++m) {
        if (!x_rows[static_cast<size_t>(m)]) {
            return nullptr;
        }
    }
    const int64_t len = static_cast<int64_t>(M) * static_cast<int64_t>(N);
    if (const uint8_t* cached =
            FindQuantizedActivationCacheEntry(ctx, src_tensor, source, len, quant_type, quant_bytes, -1,
                                              token_pos)) {
        return cached;
    }

    if (PrimaryQuantizedActivationSlotAvailable(ctx)) {
        ctx->qact_buffer.resize(quant_bytes);
        for (int m = 0; m < M; ++m) {
            uint8_t* q_ptr = ctx->qact_buffer.data() + static_cast<size_t>(m) * quant_row_stride;
            input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
        }
        ctx->qact_tensor = src_tensor;
        ctx->qact_generation = ctx->execution_generation;
        ctx->qact_source = source;
        ctx->qact_len = len;
        ctx->qact_type = quant_type;
        ctx->qact_bytes = quant_bytes;
        ctx->qact_slot_id = -1;
        ctx->qact_token_pos = token_pos;
        RecordQActCacheMiss(ctx);
        return ctx->qact_buffer.data();
    }

    QuantizedActivationCacheEntry* slot = SelectExtraQuantizedActivationSlot(ctx);
    if (!slot) {
        return nullptr;
    }
    slot->buffer.resize(quant_bytes);
    for (int m = 0; m < M; ++m) {
        uint8_t* q_ptr = slot->buffer.data() + static_cast<size_t>(m) * quant_row_stride;
        input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
    }
    StoreExtraQuantizedActivationMetadata(slot, ctx, src_tensor, source, len, quant_type, quant_bytes, -1,
                                          token_pos);
    RecordQActCacheMiss(ctx);
    return slot->buffer.data();
}

enum class Qwen36Q4KBatchedAdmissionState : int { Unknown = 0, Pass = 1, Reject = 2 };

enum class Qwen36PrefillQ4KBatchedRejectReason : int {
    None = 0,
    EnvOff = 1,
    LoraActive = 2,
    UnsupportedShape = 3,
    MissingVecDot = 4,
    KernelUnavailable = 5,
    ProbeMismatch = 6,
    ProbeInternalError = 7,
    Admitted = 8,
    RejectedCached = 9,
    NotQ4K = 10,
};

const char* Qwen36PrefillQ4KBatchedRejectReasonName(int reason) {
    switch (static_cast<Qwen36PrefillQ4KBatchedRejectReason>(reason)) {
        case Qwen36PrefillQ4KBatchedRejectReason::None:
            return "none";
        case Qwen36PrefillQ4KBatchedRejectReason::EnvOff:
            return "env_off";
        case Qwen36PrefillQ4KBatchedRejectReason::LoraActive:
            return "lora_active";
        case Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape:
            return "unsupported_shape";
        case Qwen36PrefillQ4KBatchedRejectReason::MissingVecDot:
            return "missing_vec_dot";
        case Qwen36PrefillQ4KBatchedRejectReason::KernelUnavailable:
            return "kernel_unavailable";
        case Qwen36PrefillQ4KBatchedRejectReason::ProbeMismatch:
            return "probe_mismatch";
        case Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError:
            return "probe_internal_error";
        case Qwen36PrefillQ4KBatchedRejectReason::Admitted:
            return "admitted";
        case Qwen36PrefillQ4KBatchedRejectReason::RejectedCached:
            return "rejected_cached";
        case Qwen36PrefillQ4KBatchedRejectReason::NotQ4K:
            return "not_q4k";
    }
    return "unknown";
}

static Qwen36PrefillQ4KBatchedRejectReason ResolveQwen36PrefillQ4KBatchedReason(
    bool relevant, bool mode_off, bool lora_active, bool weight_is_q4k, bool shape_supported, bool kernel_available,
    bool has_vec_dot, bool candidate_ready, bool mode_on, bool mode_probe,
    Qwen36Q4KBatchedAdmissionState admission_state) {
    if (!relevant) {
        return Qwen36PrefillQ4KBatchedRejectReason::None;
    }
    if (mode_off) {
        return Qwen36PrefillQ4KBatchedRejectReason::EnvOff;
    }
    if (lora_active) {
        return Qwen36PrefillQ4KBatchedRejectReason::LoraActive;
    }
    if (!weight_is_q4k) {
        return Qwen36PrefillQ4KBatchedRejectReason::NotQ4K;
    }
    if (!shape_supported) {
        return Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape;
    }
    if (!kernel_available) {
        return Qwen36PrefillQ4KBatchedRejectReason::KernelUnavailable;
    }
    if (!has_vec_dot) {
        return Qwen36PrefillQ4KBatchedRejectReason::MissingVecDot;
    }
    if (!candidate_ready) {
        return Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape;
    }
    if (mode_probe && admission_state == Qwen36Q4KBatchedAdmissionState::Reject) {
        return Qwen36PrefillQ4KBatchedRejectReason::RejectedCached;
    }
    if (mode_on || (mode_probe && admission_state == Qwen36Q4KBatchedAdmissionState::Pass)) {
        return Qwen36PrefillQ4KBatchedRejectReason::Admitted;
    }
    return Qwen36PrefillQ4KBatchedRejectReason::None;
}

struct Qwen36Q4KBatchedAdmissionValue {
    Qwen36Q4KBatchedAdmissionState state = Qwen36Q4KBatchedAdmissionState::Unknown;
    float max_abs_error = 0.0f;
    Qwen36PrefillQ4KBatchedRejectReason reject_reason = Qwen36PrefillQ4KBatchedRejectReason::None;
};

static std::mutex& Qwen36Q4KBatchedAdmissionMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue>& Qwen36Q4KBatchedAdmissionMap() {
    static std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue> map;
    return map;
}

static std::atomic<int>& Qwen36Q4KBatchedProbeForceFailThreadForTest() {
    static std::atomic<int> value{-1};
    return value;
}

static uint64_t HashQwen36Q4KBatchedAdmissionKey(const TransformerModel* model, const ggml_tensor* weight,
                                                 const ggml_tensor* input, int M, int N, int K) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(reinterpret_cast<uintptr_t>(model));
    mix(model ? static_cast<uint64_t>(model->variant) : 0);
    mix(model ? static_cast<uint64_t>(model->arch) : 0);
    mix(reinterpret_cast<uintptr_t>(weight));
    mix(reinterpret_cast<uintptr_t>(weight ? weight->data : nullptr));
    mix(weight ? static_cast<uint64_t>(weight->type) : 0);
    mix(input ? static_cast<uint64_t>(input->type) : 0);
    mix(static_cast<uint64_t>(M));
    mix(static_cast<uint64_t>(N));
    mix(static_cast<uint64_t>(K));
    if (weight && weight->name[0]) {
        for (const char* p = weight->name; *p; ++p) {
            mix(static_cast<unsigned char>(*p));
        }
    }
    return h;
}

static Qwen36Q4KBatchedAdmissionValue LookupQwen36Q4KBatchedAdmission(uint64_t key) {
    std::lock_guard<std::mutex> lock(Qwen36Q4KBatchedAdmissionMutex());
    auto& map = Qwen36Q4KBatchedAdmissionMap();
    auto it = map.find(key);
    return it == map.end() ? Qwen36Q4KBatchedAdmissionValue{} : it->second;
}

static void StoreQwen36Q4KBatchedAdmission(uint64_t key, Qwen36Q4KBatchedAdmissionState state, float max_abs_error,
                                           Qwen36PrefillQ4KBatchedRejectReason reason) {
    std::lock_guard<std::mutex> lock(Qwen36Q4KBatchedAdmissionMutex());
    auto& value = Qwen36Q4KBatchedAdmissionMap()[key];
    if (value.state == Qwen36Q4KBatchedAdmissionState::Reject && state != Qwen36Q4KBatchedAdmissionState::Reject) {
        return;
    }
    value.state = state;
    value.max_abs_error = max_abs_error;
    value.reject_reason = reason;
}

static void RecordQwen36Q4KBatchedProbeResult(InferenceWorkContext* ctx, bool pass, float max_abs_error,
                                              Qwen36PrefillQ4KBatchedRejectReason reason) {
    if (!ctx) return;
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_probe_pass.store(pass ? 1 : 0, std::memory_order_relaxed);
    uint32_t bits = 0;
    std::memcpy(&bits, &max_abs_error, sizeof(float));
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_max_abs_error_bits.store(bits, std::memory_order_relaxed);
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason),
                                                                            std::memory_order_relaxed);
}

static void DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(
    uint64_t key, float max_abs_error, Qwen36PrefillQ4KBatchedRejectReason reason, InferenceWorkContext* ctx) {
    if (!key) {
        return;
    }
    StoreQwen36Q4KBatchedAdmission(key, Qwen36Q4KBatchedAdmissionState::Reject, max_abs_error, reason);
    if (ctx) {
        ctx->qwen36_profile.qwen36_prefill_q4k_admission_downgraded.fetch_add(1, std::memory_order_relaxed);
        ctx->qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason),
                                                                                std::memory_order_relaxed);
    }
}

static void AtomicMaxFloatBits(std::atomic<uint32_t>& target, float value) {
    if (!std::isfinite(value) || value < 0.0f) {
        value = std::numeric_limits<float>::infinity();
    }
    uint32_t desired = 0;
    std::memcpy(&desired, &value, sizeof(float));
    uint32_t current = target.load(std::memory_order_relaxed);
    float current_value = 0.0f;
    std::memcpy(&current_value, &current, sizeof(float));
    while (value > current_value &&
           !target.compare_exchange_weak(current, desired, std::memory_order_relaxed, std::memory_order_relaxed)) {
        std::memcpy(&current_value, &current, sizeof(float));
    }
}

static bool IsGemma4SharedDenseFfnWeightName(const char* weight_name) {
    if (!weight_name) {
        return false;
    }
    return std::strstr(weight_name, ".ffn_gate.weight") || std::strstr(weight_name, ".ffn_up.weight") ||
           std::strstr(weight_name, ".ffn_down.weight");
}

static bool IsQ8RepackedGemvEnabled() {
    static const bool enabled = []() -> bool {
#if defined(__aarch64__) || defined(_M_ARM64)
        return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
        return false;
#endif
    }();
    return enabled;
}

static std::shared_ptr<Q8RepackedGemvWeight> GetOrCreateQ8RepackedGemvWeight(const void* weight_data, int64_t rows,
                                                                              int64_t cols,
                                                                              bool force_enable = false) {
    if (!weight_data || rows <= 0 || cols <= 0 || (rows % 4) != 0 || (cols % QK8_0) != 0 ||
        (!force_enable && !IsQ8RepackedGemvEnabled())) {
        return nullptr;
    }
    const Q8RepackedGemvKey key{weight_data, rows, cols};
    thread_local Q8RepackedGemvKey tls_key{};
    thread_local std::shared_ptr<Q8RepackedGemvWeight> tls_packed;
    if (tls_packed && tls_key == key) {
        return tls_packed;
    }
    static std::mutex mutex;
    static std::unordered_map<Q8RepackedGemvKey, std::shared_ptr<Q8RepackedGemvWeight>, Q8RepackedGemvKeyHash> cache;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        tls_key = key;
        tls_packed = it->second;
        return tls_packed;
    }

    const size_t src_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q8_0, cols);
    const size_t block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const size_t dst_bytes = static_cast<size_t>(rows / 4) * static_cast<size_t>(cols / QK8_0) * block_bytes;
    auto packed = std::make_shared<Q8RepackedGemvWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = cols / QK8_0;
    packed->bytes = dst_bytes;
    packed->data.resize(dst_bytes);
    if (ggml_repack_q8_0_4x8(weight_data, src_bytes, rows, cols, packed->data.data(), packed->data.size()) != 0) {
        return nullptr;
    }

    auto [insert_it, inserted] = cache.emplace(key, packed);
    tls_key = key;
    tls_packed = inserted ? packed : insert_it->second;
    return tls_packed;
}
