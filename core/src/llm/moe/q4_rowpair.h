#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#include <arm_sve.h>
#endif

namespace densecore_q4_down_pair {
struct Q4Block {
    ggml_fp16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q8Block {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(Q4Block) == 144 && offsetof(Q4Block, qs) == 16, "Q4_K ABI");
static_assert(sizeof(Q8Block) == 292 && offsetof(Q8Block, bsums) == 260, "Q8_K ABI");
}  // namespace densecore_q4_down_pair

// The oracle is pinned ggml's SVE128 nrc=2 path, whose blockwise FP order
// differs from the NEON-only nrc=2 path. Reject other builds/vector lengths.
static inline bool Q4DownRowPairNativeAvailable() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)
    return svcntb() == 16;
#else
    return false;
#endif
}

static inline bool ComputeQ4DownRowPairNative(const void* weights, size_t weight_stride, const void* input,
                                              int reduction, float* output) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)
    using namespace densecore_q4_down_pair;
    if (!Q4DownRowPairNativeAvailable() || !weights || !input || !output || reduction <= 0 || reduction % 256 != 0 ||
        weight_stride < size_t(reduction / 256) * sizeof(Q4Block) || weight_stride % alignof(Q4Block) != 0 ||
        reinterpret_cast<uintptr_t>(weights) % alignof(Q4Block) != 0 ||
        reinterpret_cast<uintptr_t>(input) % alignof(Q8Block) != 0)
        return false;
    const auto* x0 = static_cast<const Q4Block*>(weights);
    const auto* x1 = reinterpret_cast<const Q4Block*>(static_cast<const uint8_t*>(weights) + weight_stride);
    const auto* y = static_cast<const Q8Block*>(input);
    float32x2_t sum = vdup_n_f32(0);
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int b = 0; b < reduction / 256; ++b) {
        int32_t dots[2] = {}, biases[2] = {};
        const Q4Block* x[2] = {x0 + b, x1 + b};
        for (int g = 0; g < 8; ++g) {
            const auto ya = vld1q_s8(y[b].qs + g * 32);
            const auto yb = vld1q_s8(y[b].qs + g * 32 + 16);
            const int ys = int(y[b].bsums[g * 2]) + int(y[b].bsums[g * 2 + 1]);
            for (int r = 0; r < 2; ++r) {
                const uint8_t* sc = x[r]->scales;
                const int scale = g < 4 ? sc[g] & 63 : (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4);
                const int minimum = g < 4 ? sc[g + 4] & 63 : (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
                const uint8_t* q = x[r]->qs + (g / 2) * 32;
                const auto a = vld1q_u8(q), c = vld1q_u8(q + 16);
                const auto qa = vreinterpretq_s8_u8(g & 1 ? vshrq_n_u8(a, 4) : vandq_u8(a, mask));
                const auto qb = vreinterpretq_s8_u8(g & 1 ? vshrq_n_u8(c, 4) : vandq_u8(c, mask));
                const auto dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), qa, ya), qb, yb);
                dots[r] += vaddvq_s32(dot) * scale;
                biases[r] += ys * minimum;
            }
        }
        const float scales[2] = {ggml_fp16_to_fp32(x0[b].d) * y[b].d, ggml_fp16_to_fp32(x1[b].d) * y[b].d};
        const float mins[2] = {-(ggml_fp16_to_fp32(x0[b].dmin) * y[b].d), -(ggml_fp16_to_fp32(x1[b].dmin) * y[b].d)};
        sum = vfma_f32(sum, vcvt_f32_s32(vld1_s32(dots)), vld1_f32(scales));
        sum = vfma_f32(sum, vld1_f32(mins), vcvt_f32_s32(vld1_s32(biases)));
    }
    vst1_f32(output, sum);
    return true;
#else
    (void)weights;
    (void)weight_stride;
    (void)input;
    (void)reduction;
    (void)output;
    return false;
#endif
}
