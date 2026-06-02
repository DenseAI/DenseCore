/**
 * @file hwy_moe_quant.cc
 * @brief GGML K-quant compatible MoE primitives owned by DenseCore.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_moe_quant.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

constexpr int kMoEQK_K = 256;
constexpr int kMoEKScaleSize = 12;

struct MoEQ4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[kMoEKScaleSize];
    uint8_t qs[kMoEQK_K / 2];
};
static_assert(sizeof(MoEQ4KBlock) == 2 * sizeof(uint16_t) + kMoEKScaleSize + kMoEQK_K / 2,
              "DenseCore MoE Q4_K block layout must match ggml block_q4_K");

struct MoEQ5KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[kMoEKScaleSize];
    uint8_t qh[kMoEQK_K / 8];
    uint8_t qs[kMoEQK_K / 2];
};
static_assert(sizeof(MoEQ5KBlock) == 2 * sizeof(uint16_t) + kMoEKScaleSize + kMoEQK_K / 2 + kMoEQK_K / 8,
              "DenseCore MoE Q5_K block layout must match ggml block_q5_K");

struct MoEQ8KBlock {
    float d;
    int8_t qs[kMoEQK_K];
    int16_t bsums[kMoEQK_K / 16];
};
static_assert(sizeof(MoEQ8KBlock) == sizeof(float) + kMoEQK_K + (kMoEQK_K / 16) * sizeof(int16_t),
              "DenseCore MoE Q8_K block layout must match ggml block_q8_K");

HWY_INLINE float Fp16ToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 113u;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

HWY_INLINE int NearestIntForQ8K(float value) {
    float biased = value + 12582912.0f;
    int bits = 0;
    std::memcpy(&bits, &biased, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

HWY_INLINE void DecodeQ4KScales(const uint8_t scales_packed[kMoEKScaleSize], uint32_t utmp[4]) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    std::memcpy(utmp, scales_packed, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

bool QuantizeRowQ8KImpl(const float* input, void* q8_output, int64_t cols) {
    if (!input || !q8_output || cols <= 0 || (cols % kMoEQK_K) != 0) {
        return false;
    }

    auto* blocks = static_cast<MoEQ8KBlock*>(q8_output);
    const int64_t nb = cols / kMoEQK_K;
    for (int64_t bi = 0; bi < nb; ++bi) {
        const float* x = input + static_cast<size_t>(bi) * kMoEQK_K;
        MoEQ8KBlock& y = blocks[bi];
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < kMoEQK_K; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y.d = 0.0f;
            std::memset(y.qs, 0, sizeof(y.qs));
            std::memset(y.bsums, 0, sizeof(y.bsums));
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < kMoEQK_K; ++j) {
            const int v = NearestIntForQ8K(iscale * x[j]);
            y.qs[j] = static_cast<int8_t>(std::min(127, v));
        }
        for (int j = 0; j < kMoEQK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y.qs[j * 16 + ii];
            }
            y.bsums[j] = static_cast<int16_t>(sum);
        }
        y.d = 1.0f / iscale;
    }
    return true;
}

template <typename BlockT, bool Q5>
bool DotKQ8KImpl(const void* weight_row, const void* q8_input_row, int64_t cols, float* output) {
    if (!weight_row || !q8_input_row || !output || cols <= 0 || (cols % kMoEQK_K) != 0) {
        return false;
    }

    const int nb = static_cast<int>(cols / kMoEQK_K);
    const auto* x = static_cast<const BlockT*>(weight_row);
    const auto* y = static_cast<const MoEQ8KBlock*>(q8_input_row);

    uint32_t utmp[4];
    int8_t unpacked[kMoEQK_K];
    int32_t dot_chunks[8];
    float sums[8] = {};
    float sumf = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* q = x[bi].qs;
        int8_t* dst = unpacked;
        if constexpr (Q5) {
            const uint8_t* high = x[bi].qh;
            uint8_t mask = 1;
            for (int j = 0; j < kMoEQK_K / 64; ++j) {
                for (int l = 0; l < 32; ++l) dst[l] = static_cast<int8_t>(q[l] & 0xF);
                for (int l = 0; l < 32; ++l) dst[l] += (high[l] & mask) ? 16 : 0;
                dst += 32;
                mask <<= 1;
                for (int l = 0; l < 32; ++l) dst[l] = static_cast<int8_t>(q[l] >> 4);
                for (int l = 0; l < 32; ++l) dst[l] += (high[l] & mask) ? 16 : 0;
                dst += 32;
                mask <<= 1;
                q += 32;
            }
        } else {
            for (int j = 0; j < kMoEQK_K / 64; ++j) {
                for (int l = 0; l < 32; ++l) dst[l] = static_cast<int8_t>(q[l] & 0xF);
                dst += 32;
                for (int l = 0; l < 32; ++l) dst[l] = static_cast<int8_t>(q[l] >> 4);
                dst += 32;
                q += 32;
            }
        }

        DecodeQ4KScales(x[bi].scales, utmp);

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const MoEQ8KBlock& yb = y[bi];

        int min_dot = 0;
        for (int j = 0; j < kMoEQK_K / 16; ++j) {
            min_dot += static_cast<int>(yb.bsums[j]) * static_cast<int>(mins[j / 2]);
        }

        std::memset(dot_chunks, 0, sizeof(dot_chunks));
        const int8_t* q8 = yb.qs;
        const int8_t* uq = unpacked;
        int is = 0;
        for (int j = 0; j < kMoEQK_K / 32; ++j) {
            const int32_t scale = static_cast<int32_t>(scales[is++]);
            for (int rep = 0; rep < 4; ++rep) {
                for (int l = 0; l < 8; ++l) {
                    dot_chunks[l] += scale * static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]);
                }
                q8 += 8;
                uq += 8;
            }
        }

        const float d = Fp16ToFloat(static_cast<uint16_t>(x[bi].d)) * yb.d;
        const float dmin = Fp16ToFloat(static_cast<uint16_t>(x[bi].dmin)) * yb.d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(dot_chunks[l]);
        }
        sumf -= dmin * static_cast<float>(min_dot);
    }

    for (float v : sums) {
        sumf += v;
    }
    *output = sumf;
    return true;
}

bool DotQ4KQ8KImpl(const void* q4_weight_row, const void* q8_input_row, int64_t cols, float* output) {
    return DotKQ8KImpl<MoEQ4KBlock, false>(q4_weight_row, q8_input_row, cols, output);
}

bool FusedSwiGLUQ4KQ8KRowsImpl(const void* q4_gate_rows, const void* q4_up_rows, const void* q8_input_row,
                               int64_t cols, int64_t row_count, size_t row_bytes, float* output) {
    if (!q4_gate_rows || !q4_up_rows || !q8_input_row || !output || cols <= 0 || row_count <= 0 ||
        (cols % kMoEQK_K) != 0 || row_bytes == 0) {
        return false;
    }

    const auto* gate_base = static_cast<const uint8_t*>(q4_gate_rows);
    const auto* up_base = static_cast<const uint8_t*>(q4_up_rows);
    for (int64_t row = 0; row < row_count; ++row) {
        float gate = 0.0f;
        float up = 0.0f;
        const void* gate_row = gate_base + static_cast<size_t>(row) * row_bytes;
        const void* up_row = up_base + static_cast<size_t>(row) * row_bytes;
        if (!DotKQ8KImpl<MoEQ4KBlock, false>(gate_row, q8_input_row, cols, &gate) ||
            !DotKQ8KImpl<MoEQ4KBlock, false>(up_row, q8_input_row, cols, &up)) {
            return false;
        }
        output[row] = (gate / (1.0f + std::exp(-gate))) * up;
    }
    return true;
}

bool DotQ5KQ8KImpl(const void* q5_weight_row, const void* q8_input_row, int64_t cols, float* output) {
    return DotKQ8KImpl<MoEQ5KBlock, true>(q5_weight_row, q8_input_row, cols, output);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(QuantizeRowQ8KImpl);
HWY_EXPORT(DotQ4KQ8KImpl);
HWY_EXPORT(FusedSwiGLUQ4KQ8KRowsImpl);
HWY_EXPORT(DotQ5KQ8KImpl);

bool QuantizeRowQ8K_Hwy(const float* input, void* q8_output, int64_t cols) {
    return HWY_DYNAMIC_DISPATCH(QuantizeRowQ8KImpl)(input, q8_output, cols);
}

bool DotQ4KQ8K_Hwy(const void* q4_weight_row, const void* q8_input_row, int64_t cols, float* output) {
    return HWY_DYNAMIC_DISPATCH(DotQ4KQ8KImpl)(q4_weight_row, q8_input_row, cols, output);
}

bool FusedSwiGLUQ4KQ8KRows_Hwy(const void* q4_gate_rows, const void* q4_up_rows, const void* q8_input_row,
                               int64_t cols, int64_t row_count, size_t row_bytes, float* output) {
    return HWY_DYNAMIC_DISPATCH(FusedSwiGLUQ4KQ8KRowsImpl)(q4_gate_rows, q4_up_rows, q8_input_row, cols, row_count,
                                                          row_bytes, output);
}

bool DotQ5KQ8K_Hwy(const void* q5_weight_row, const void* q8_input_row, int64_t cols, float* output) {
    return HWY_DYNAMIC_DISPATCH(DotQ5KQ8KImpl)(q5_weight_row, q8_input_row, cols, output);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif
