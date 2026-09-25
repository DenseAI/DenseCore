// Native-layout Q6_K output row x four Q8_K activation rows. Keep the
// pinned ggml AVX2 dot's block FMA and lane-reduction order for each output.
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace densecore_q6_small_batch {
// Pinned ggml Q6_K and Q8_K native block ABI. Keep these local to avoid
// importing ggml-quants.h into worker.cpp's existing block declarations.
struct Q6Block {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    ggml_fp16_t d;
};
struct Q8Block {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(Q6Block) == 210 && offsetof(Q6Block, d) == 208, "Q6_K native ABI");
static_assert(sizeof(Q8Block) == 292 && offsetof(Q8Block, bsums) == 260, "Q8_K native ABI");
}  // namespace densecore_q6_small_batch

static inline constexpr bool Q6KQ8KM4NativeAvailable() {
#if defined(__AVX2__) && defined(__FMA__)
    return true;
#else
    return false;
#endif
}

static inline bool ComputeQ6KQ8KM4Native(const void* weight_row, const uint8_t* input_base, size_t input_stride,
                                         int tokens, int reduction, float* sums) {
#if defined(__AVX2__) && defined(__FMA__)
    if (!weight_row || !input_base || !sums || tokens != 4 || reduction <= 0 || reduction % 256 != 0 ||
        input_stride < size_t(reduction / 256) * sizeof(densecore_q6_small_batch::Q8Block) ||
        input_stride % alignof(densecore_q6_small_batch::Q8Block) != 0 ||
        reinterpret_cast<uintptr_t>(input_base) % alignof(densecore_q6_small_batch::Q8Block) != 0 ||
        reinterpret_cast<uintptr_t>(weight_row) % alignof(densecore_q6_small_batch::Q6Block) != 0)
        return false;
    const auto* x = static_cast<const densecore_q6_small_batch::Q6Block*>(weight_row);
    const densecore_q6_small_batch::Q8Block* y[4];
    for (int m = 0; m < 4; ++m)
        y[m] = reinterpret_cast<const densecore_q6_small_batch::Q8Block*>(input_base + m * input_stride);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32 = _mm256_set1_epi8(32);
    for (int i = 0; i < reduction / 256; ++i) {
        __m256i sum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                          _mm256_setzero_si256()};
        for (int j = 0; j < 2; ++j) {
            const auto lo0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].ql + j * 64));
            const auto lo1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].ql + j * 64 + 32));
            const auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qh + j * 32));
            const __m256i q[4] = {
                _mm256_or_si256(_mm256_and_si256(lo0, m4), _mm256_slli_epi16(_mm256_and_si256(hi, m2), 4)),
                _mm256_or_si256(_mm256_and_si256(lo1, m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hi, 2), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo0, 4), m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hi, 4), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo1, 4), m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hi, 6), m2), 4))};
            __m256i scales[4];
            for (int z = 0; z < 4; ++z) {
                scales[z] = _mm256_set_m128i(_mm_set1_epi16(x[i].scales[j * 8 + z * 2 + 1]),
                                             _mm_set1_epi16(x[i].scales[j * 8 + z * 2]));
            }
            for (int m = 0; m < 4; ++m) {
                __m256i p[4];
                for (int z = 0; z < 4; ++z) {
                    const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[m][i].qs + j * 128 + z * 32));
                    p[z] = _mm256_madd_epi16(
                        scales[z], _mm256_sub_epi16(_mm256_maddubs_epi16(q[z], v), _mm256_maddubs_epi16(m32, v)));
                }
                sum[m] = _mm256_add_epi32(sum[m], _mm256_add_epi32(p[0], p[1]));
                sum[m] = _mm256_add_epi32(sum[m], _mm256_add_epi32(p[2], p[3]));
            }
        }
        const float xd = ggml_fp16_to_fp32(x[i].d);
        for (int m = 0; m < 4; ++m)
            acc[m] = _mm256_fmadd_ps(_mm256_set1_ps(y[m][i].d * xd), _mm256_cvtepi32_ps(sum[m]), acc[m]);
    }
    for (int m = 0; m < 4; ++m) {
        __m128 r = _mm_add_ps(_mm256_extractf128_ps(acc[m], 1), _mm256_castps256_ps128(acc[m]));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        sums[m] = _mm_cvtss_f32(_mm_add_ss(r, _mm_movehdup_ps(r)));
    }
    return true;
#else
    (void)weight_row;
    (void)input_base;
    (void)input_stride;
    (void)tokens;
    (void)reduction;
    (void)sums;
    return false;
#endif
}
