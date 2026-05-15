/**
 * @file hwy_paged_attention.cc
 * @brief Highway-accelerated PagedAttention implementation
 */

// 1. Set up Highway target
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_paged_attention.cc"
#include "hwy/foreach_target.h"

#include "densecore/memory/kv_cache.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hwy/cache_control.h>
#include <hwy/highway.h>
#include <vector>


// Use structs from densecore/memory/kv_cache.h
// block_q8_0, QK8_0 are available via densecore namespace or global if defined there.
// densecore/memory/kv_cache.h defines them in global scope based on the file content I wrote.

#ifndef DENSECORE_HWY_FP16_TO_FP32_DEFINED_
#define DENSECORE_HWY_FP16_TO_FP32_DEFINED_

// On ARM64, use hardware FP16→FP32 conversion (single FCVT instruction)
// instead of software bit manipulation. This is ~5x faster for the scalar
// tail path when head_dim is not a multiple of the SIMD vector width.
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__F16C__)
#include <immintrin.h>
#endif

namespace densecore {
// Guarded against foreach_target re-inclusion within the same TU.
inline float fp16_to_fp32(uint16_t h) {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 hardware path: reinterpret uint16_t as __fp16 via NEON.
    // vget_lane_f32(vcvt_f32_f16()) compiles to a single FCVT instruction.
    float16x4_t v16 = vreinterpret_f16_u16(vdup_n_u16(h));
    float32x4_t v32 = vcvt_f32_f16(v16);
    return vgetq_lane_f32(v32, 0);
#elif defined(__F16C__)
    // x86 with F16C: use _cvtsh_ss intrinsic
    return _cvtsh_ss(h);
#else
    // Portable software fallback
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;
    if (exp == 0) {
        result = sign;
    } else if (exp == 31) {
        result = sign | 0x7F800000 | (mant << 13);
    } else {
        result = sign | ((exp + 112) << 23) | (mant << 13);  // Bias 15 -> 127
    }
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
#endif
}
}  // namespace densecore
#endif  // DENSECORE_HWY_FP16_TO_FP32_DEFINED_

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Dot Product Kernels
// ============================================================================

// F32 Dot Product
template <class D> HWY_INLINE float DotProductF32(D d, const float* q, const float* k, int head_dim) {
    auto sum = hn::Zero(d);
    int i = 0;
    for (; i <= head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
        const auto vq = hn::LoadU(d, q + i);
        const auto vk = hn::LoadU(d, k + i);
        sum = hn::MulAdd(vq, vk, sum);
    }
    float scalar_sum = hn::ReduceSum(d, sum);
    for (; i < head_dim; ++i) {
        scalar_sum += q[i] * k[i];
    }
    return scalar_sum;
}

// F16 Dot Product (Keys are FP16 payload stored as uint16_t, Query is FP32).
//
// [P7 fix] The previous implementation used an AVX2+F16C explicit path and
// fell back to a scalar loop for all other targets (NEON, SVE, etc.).
// Highway's hn::PromoteTo(float, float16_t) is portable across AVX2+F16C,
// NEON (vcvt_f32_f16), SVE (fcvt), and any future Highway target, so we now
// use a single Highway SIMD path for all architectures.
//
// The #if guard is kept only to suppress the AVX2-specific BitCast pattern
// warning on compilers that don't support __F16C__ intrinsics natively; the
// logic is identical on both branches.
template <class D> HWY_INLINE float DotProductF16(D d, const float* q, const uint16_t* k, int head_dim) {
    auto sum = hn::Zero(d);
    const hn::Rebind<uint16_t, D> du16;
    const hn::Rebind<hwy::float16_t, D> df16;
    int i = 0;
    for (; i <= head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
        const auto vq = hn::LoadU(d, q + i);
        // KV cache stores FP16 bits as uint16_t. BitCast reinterprets the raw
        // bits without conversion, avoiding strict-aliasing UB.
        const auto vk_bits = hn::LoadU(du16, k + i);
        const auto vk16 = hn::BitCast(df16, vk_bits);
        const auto vk = hn::PromoteTo(d, vk16);  // portable: F16C / vcvt / fcvt
        sum = hn::MulAdd(vq, vk, sum);
    }
    float scalar_sum = hn::ReduceSum(d, sum);
    for (; i < head_dim; ++i) {
        scalar_sum += q[i] * densecore::fp16_to_fp32(k[i]);
    }
    return scalar_sum;
}

// Q8_0 Dot Product (Keys are Q8_0 blocks)
// Uses Highway int8→int16→int32→float promotion chain instead of scalar stack buffer.
template <class D> HWY_INLINE float DotProductQ8_0(D d, const float* q, const void* k_data, int head_dim) {
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(k_data);
    const int nb = head_dim / QK8_0;

    auto total_sum = hn::Zero(d);
    const int lanes = static_cast<int>(hn::Lanes(d));
    // Rebind preserves vector width. For float lanes N, int8 lanes are 4N.
    // We LoadN() only N bytes so promote-chain lower lanes exactly match scalar
    // dequant for q[q_offset + i ... + i+N).
    const hn::Rebind<int8_t, D> di8;
    const hn::Rebind<int16_t, D> di16;
    const hn::Rebind<int32_t, D> di32;
    float scalar_tail_sum = 0.0f;
    int q_offset = 0;

    for (int b = 0; b < nb; ++b) {
        float block_scale = densecore::fp16_to_fp32(blocks[b].d);
        const auto v_scale = hn::Set(d, block_scale);
        const int8_t* qs = blocks[b].qs;

        int i = 0;
        // Vectorized path: load quarter-width int8, promote to float32
        for (; i <= QK8_0 - lanes; i += lanes) {
            const auto vq = hn::LoadU(d, q + q_offset + i);

            // Load exactly `lanes` int8 values (no over-read), then promote.
            const auto vi8 = hn::LoadN(di8, qs + i, static_cast<size_t>(lanes));
            const auto vi16 = hn::PromoteTo(di16, vi8);
            const auto vi32 = hn::PromoteTo(di32, vi16);
            auto vk = hn::ConvertTo(d, vi32);
            vk = hn::Mul(vk, v_scale);

            total_sum = hn::MulAdd(vq, vk, total_sum);
        }
        // Scalar tail within block
        for (; i < QK8_0; ++i) {
            scalar_tail_sum += q[q_offset + i] * (static_cast<float>(qs[i]) * block_scale);
        }
        q_offset += QK8_0;
    }

    float scalar_sum = hn::ReduceSum(d, total_sum) + scalar_tail_sum;

    int tail_start = nb * QK8_0;
    if (tail_start < head_dim) {
        float block_scale = densecore::fp16_to_fp32(blocks[nb].d);
        const int8_t* qs = blocks[nb].qs;
        for (int i = 0; i < head_dim - tail_start; ++i) {
            scalar_sum += q[tail_start + i] * (static_cast<float>(qs[i]) * block_scale);
        }
    }
    return scalar_sum;
}

// Q4_0 Dot Product — Highway-vectorized.
//
// Unpacks each 32-element Q4_0 block into a 128-byte float scratch buffer
// (always L1-resident) and performs a SIMD FMA dot product against the query.
// The unpack loop (16 scalar iterations per block) is data-independent bit ops
// and is not the bottleneck; the SIMD dot product loop is the hot path.
// On NEON (4 float lanes) this is 8 FMA iterations vs 32 scalar; on SVE-512
// (16 float lanes) only 2 FMA iterations.
template <class D> HWY_INLINE float DotProductQ4_0(D d, const float* q, const void* k_data, int head_dim) {
    const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(k_data);
    const int nb = head_dim / QK4_0;
    const int lanes = static_cast<int>(hn::Lanes(d));
    float total = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const float scale = densecore::fp16_to_fp32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        const int q_base = b * QK4_0;

        // Unpack 16 bytes → 32 float values. Stack buffer is always L1-resident.
        float scratch[QK4_0];
        for (int j = 0; j < QK4_0 / 2; ++j) {
            scratch[2 * j + 0] = static_cast<float>((qs[j] & 0x0F) - 8);
            scratch[2 * j + 1] = static_cast<float>((qs[j] >> 4)   - 8);
        }

        // Vectorized dot product of q[q_base..q_base+32] × scratch[0..32]
        auto vsum = hn::Zero(d);
        int i = 0;
        for (; i <= QK4_0 - lanes; i += lanes) {
            vsum = hn::MulAdd(hn::LoadU(d, q + q_base + i), hn::LoadU(d, scratch + i), vsum);
        }
        float block_sum = hn::ReduceSum(d, vsum);
        for (; i < QK4_0; ++i) {
            block_sum += q[q_base + i] * scratch[i];
        }
        total += block_sum * scale;
    }

    // Tail: head_dim not a multiple of QK4_0 (uncommon but safe)
    const int tail_start = nb * QK4_0;
    if (tail_start < head_dim) {
        const float scale = densecore::fp16_to_fp32(blocks[nb].d);
        for (int i = 0; tail_start + i < head_dim; ++i) {
            const uint8_t packed = blocks[nb].qs[i / 2];
            const int qv = (i & 1) ? (((packed >> 4) & 0x0F) - 8) : ((packed & 0x0F) - 8);
            total += q[tail_start + i] * (static_cast<float>(qv) * scale);
        }
    }

    return total;
}

// ============================================================================
// Accumulate Kernels
// ============================================================================

template <class D> HWY_INLINE void AccumulateF32(D d, float* accum, const float* v, float weight, int head_dim) {
    const auto w = hn::Set(d, weight);
    int i = 0;
    for (; i <= head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
        auto va = hn::LoadU(d, accum + i);
        const auto vv = hn::LoadU(d, v + i);
        va = hn::MulAdd(vv, w, va);
        hn::StoreU(va, d, accum + i);
    }
    for (; i < head_dim; ++i) {
        accum[i] += v[i] * weight;
    }
}

// [P7 fix] Same portable SIMD treatment as DotProductF16 above.
// hn::PromoteTo handles F16→F32 on AVX2+F16C, NEON, SVE, and other targets.
template <class D> HWY_INLINE void AccumulateF16(D d, float* accum, const uint16_t* v, float weight, int head_dim) {
    const auto w = hn::Set(d, weight);
    const hn::Rebind<uint16_t, D> du16;
    const hn::Rebind<hwy::float16_t, D> df16;
    int i = 0;
    for (; i <= head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
        auto va = hn::LoadU(d, accum + i);
        const auto vv_bits = hn::LoadU(du16, v + i);
        // FP16 payload stored as uint16_t; BitCast avoids strict-aliasing UB.
        const auto vv16 = hn::BitCast(df16, vv_bits);
        const auto vv = hn::PromoteTo(d, vv16);  // portable: F16C / vcvt / fcvt
        va = hn::MulAdd(vv, w, va);
        hn::StoreU(va, d, accum + i);
    }
    for (; i < head_dim; ++i) {
        accum[i] += densecore::fp16_to_fp32(v[i]) * weight;
    }
}

template <class D> HWY_INLINE void AccumulateQ8_0(D d, float* accum, const void* v_data, float weight, int head_dim) {
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(v_data);
    const int nb = head_dim / QK8_0;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const hn::Rebind<int8_t, D> di8;
    const hn::Rebind<int16_t, D> di16;
    const hn::Rebind<int32_t, D> di32;
    int accum_offset = 0;

    for (int b = 0; b < nb; ++b) {
        float block_scale = densecore::fp16_to_fp32(blocks[b].d);
        auto v_scale = hn::Set(d, block_scale * weight);
        const int8_t* qs = blocks[b].qs;

        int i = 0;
        for (; i <= QK8_0 - lanes; i += lanes) {
            auto va = hn::LoadU(d, accum + accum_offset + i);

            // Load exactly `lanes` int8 values (no over-read), then promote.
            const auto vi8 = hn::LoadN(di8, qs + i, static_cast<size_t>(lanes));
            const auto vi16 = hn::PromoteTo(di16, vi8);
            const auto vi32 = hn::PromoteTo(di32, vi16);
            const auto vv = hn::ConvertTo(d, vi32);

            va = hn::MulAdd(vv, v_scale, va);
            hn::StoreU(va, d, accum + accum_offset + i);
        }
        for (; i < QK8_0; ++i) {
            accum[accum_offset + i] += static_cast<float>(qs[i]) * block_scale * weight;
        }
        accum_offset += QK8_0;
    }

    int tail_start = nb * QK8_0;
    if (tail_start < head_dim) {
        float block_scale = densecore::fp16_to_fp32(blocks[nb].d);
        const int8_t* qs = blocks[nb].qs;
        for (int i = 0; i < head_dim - tail_start; ++i) {
            accum[tail_start + i] += static_cast<float>(qs[i]) * block_scale * weight;
        }
    }
}

// Q4_0 Accumulate — Highway-vectorized (same unpack strategy as DotProductQ4_0).
template <class D> HWY_INLINE void AccumulateQ4_0(D d, float* accum, const void* v_data, float weight, int head_dim) {
    const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(v_data);
    const int nb = head_dim / QK4_0;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int accum_offset = 0;

    for (int b = 0; b < nb; ++b) {
        const float scale = densecore::fp16_to_fp32(blocks[b].d) * weight;
        const auto v_scale = hn::Set(d, scale);
        const uint8_t* qs = blocks[b].qs;

        // Unpack 16 bytes → 32 float values (L1-resident scratch buffer).
        float scratch[QK4_0];
        for (int j = 0; j < QK4_0 / 2; ++j) {
            scratch[2 * j + 0] = static_cast<float>((qs[j] & 0x0F) - 8);
            scratch[2 * j + 1] = static_cast<float>((qs[j] >> 4)   - 8);
        }

        // Vectorized accumulate: accum[i] += scratch[i] * scale
        int i = 0;
        for (; i <= QK4_0 - lanes; i += lanes) {
            auto va = hn::LoadU(d, accum + accum_offset + i);
            const auto vs = hn::LoadU(d, scratch + i);
            va = hn::MulAdd(vs, v_scale, va);
            hn::StoreU(va, d, accum + accum_offset + i);
        }
        for (; i < QK4_0; ++i) {
            accum[accum_offset + i] += scratch[i] * scale;
        }
        accum_offset += QK4_0;
    }

    // Tail
    if (accum_offset < head_dim) {
        const float scale = densecore::fp16_to_fp32(blocks[nb].d) * weight;
        for (int i = 0; accum_offset + i < head_dim; ++i) {
            const uint8_t packed = blocks[nb].qs[i / 2];
            const int qv = (i & 1) ? (((packed >> 4) & 0x0F) - 8) : ((packed & 0x0F) - 8);
            accum[accum_offset + i] += static_cast<float>(qv) * scale;
        }
    }
}

// ============================================================================
// Main Kernel Logic
// ============================================================================
// Main Paged Attention Implementation (Generic over Highway Target)
void PagedAttentionImpl(const float* query, const void* const* k_block_ptrs, const void* const* v_block_ptrs,
                        int32_t cache_type, int32_t num_heads, int32_t qk_head_dim, int32_t v_head_dim,
                        int32_t n_head_kv, int32_t block_table_size, int32_t context_len, int32_t context_start_pos,
                        int32_t query_pos, int32_t sliding_window,
                        int32_t mask_history_kept, int32_t mask_sink_kept, int32_t mask_tail_start,
                        int64_t k_head_stride_bytes_in, int64_t k_slot_stride_bytes_in,
                        int64_t v_head_stride_bytes_in, int64_t v_slot_stride_bytes_in, float scale,
                        float logit_softcap, float* output, int32_t head_start, int32_t head_end,
                        int32_t num_heads_total) {

    const hn::ScalableTag<float> d;

    if (!k_block_ptrs || !v_block_ptrs) return;
    if (n_head_kv <= 0) return;
    const int total_heads = (num_heads_total > 0) ? num_heads_total : num_heads;
    if (total_heads <= 0 || total_heads % n_head_kv != 0) return;
    const int kv_group_size = total_heads / n_head_kv;
    if (kv_group_size <= 0) return;
    if (qk_head_dim <= 0 || v_head_dim <= 0) return;
    if (k_head_stride_bytes_in <= 0 || k_slot_stride_bytes_in <= 0 || v_head_stride_bytes_in <= 0 ||
        v_slot_stride_bytes_in <= 0) {
        return;
    }
    const size_t k_head_stride_bytes = static_cast<size_t>(k_head_stride_bytes_in);
    const size_t k_slot_stride_bytes = static_cast<size_t>(k_slot_stride_bytes_in);
    const size_t v_head_stride_bytes = static_cast<size_t>(v_head_stride_bytes_in);
    const size_t v_slot_stride_bytes = static_cast<size_t>(v_slot_stride_bytes_in);

    float block_scores[BLOCK_SIZE];

    const int h_begin = std::max(0, head_start);
    const int h_limit = (head_end < 0) ? num_heads : std::min(num_heads, head_end);
    int prefetch_tokens = 1;
    if (qk_head_dim >= 256) {
        prefetch_tokens = 3;
    } else if (qk_head_dim >= 128) {
        prefetch_tokens = 2;
    }
    if (cache_type == 8 || cache_type == 4) {  // Quantized KV has higher decode-side unpack pressure.
        prefetch_tokens = std::min(prefetch_tokens + 1, 4);
    }
    static const bool prefetch_blocks = []() {
        const char* env = std::getenv("DENSECORE_PAGED_ATTN_PREFETCH_BLOCKS");
        if (!env || env[0] == '\0') return true;
        return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
    }();

    // Byte footprint of one KV head vector — cache_type:
    //   0=F32 (4B/elem), 1=F16 (2B/elem), 8=Q8_0 (~1B/elem), 4=Q4_0 (~0.5B/elem)
    // Hoisted out of the per-head/per-block loops; constant for the whole call.
    const size_t k_head_bytes =
        (cache_type == 1) ? static_cast<size_t>(qk_head_dim) * 2 :
        (cache_type == 8) ? static_cast<size_t>(qk_head_dim) :
        (cache_type == 4) ? static_cast<size_t>(qk_head_dim + 1) / 2 :
        static_cast<size_t>(qk_head_dim) * 4;  // F32
    const size_t v_head_bytes =
        (cache_type == 1) ? static_cast<size_t>(v_head_dim) * 2 :
        (cache_type == 8) ? static_cast<size_t>(v_head_dim) :
        (cache_type == 4) ? static_cast<size_t>(v_head_dim + 1) / 2 :
        static_cast<size_t>(v_head_dim) * 4;  // F32

    for (int h = h_begin; h < h_limit; ++h) {
        const float* q_head = query + h * qk_head_dim;
        float* out_head = output + h * v_head_dim;

        int kv_head = h / kv_group_size;
        if (kv_head < 0) kv_head = 0;
        if (kv_head >= n_head_kv) kv_head = n_head_kv - 1;
        // Offset in bytes for the specific KV head
        const size_t k_head_offset_bytes = static_cast<size_t>(kv_head) * k_head_stride_bytes;
        const size_t v_head_offset_bytes = static_cast<size_t>(kv_head) * v_head_stride_bytes;

        float m_prev = -1e30f;
        float d_prev = 0.0f;

        // Initialize output buffer to 0
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        const int context_end_pos = context_start_pos + context_len;
        const int start_block = std::max(0, context_start_pos / BLOCK_SIZE);
        const int start_slot = std::max(0, context_start_pos % BLOCK_SIZE);

        for (int logical_block = start_block; logical_block < block_table_size; ++logical_block) {
            const int block_abs_start = logical_block * BLOCK_SIZE;
            const int slot_begin = (logical_block == start_block) ? start_slot : 0;
            const int slot_end = std::min(BLOCK_SIZE, context_end_pos - block_abs_start);
            const int num_tokens = slot_end - slot_begin;
            if (num_tokens <= 0) {
                if (block_abs_start >= context_end_pos) break;
                continue;
            }

            const uint8_t* k_block_base = reinterpret_cast<const uint8_t*>(k_block_ptrs[logical_block]);
            const uint8_t* v_block_base = reinterpret_cast<const uint8_t*>(v_block_ptrs[logical_block]);
            if (!k_block_base || !v_block_base) {
                continue;
            }

            // Inter-block prefetch: cover the full KV head of the next block, not
            // just the first 64 B. A single Prefetch only warms one cache line;
            // for FP16 head_dim=128 the head is 256 B (4 lines), for F32 it is
            // 512 B (8 lines). Issuing all lines here gives the hardware enough
            // lead time before we begin processing that block.
            if (prefetch_blocks && logical_block + 1 < block_table_size) {
                const uint8_t* next_k_block = reinterpret_cast<const uint8_t*>(k_block_ptrs[logical_block + 1]);
                const uint8_t* next_v_block = reinterpret_cast<const uint8_t*>(v_block_ptrs[logical_block + 1]);
                if (next_k_block) {
                    for (size_t pf_off = 0; pf_off < k_head_bytes; pf_off += 64) {
                        ::hwy::Prefetch(next_k_block + k_head_offset_bytes + pf_off);
                    }
                }
                if (next_v_block) {
                    for (size_t pf_off = 0; pf_off < v_head_bytes; pf_off += 64) {
                        ::hwy::Prefetch(next_v_block + v_head_offset_bytes + pf_off);
                    }
                }
            }

            // 1. Compute Scores
            float m_block = -1e30f;

            for (int t = 0; t < num_tokens; ++t) {
                // Prefetch next token's K head: all cache lines (64 B each).
                // A single Prefetch only covers 64 B; for head_dim >= 64 (FP16) or
                // >= 32 (FP32) the remaining cache lines would be cold misses.
                const int pf_t = t + prefetch_tokens;
                if (pf_t < num_tokens) {
                    const int next_slot = slot_begin + pf_t;
                    const uint8_t* next_k = k_block_base + next_slot * k_slot_stride_bytes + k_head_offset_bytes;
                    for (size_t pf_off = 0; pf_off < k_head_bytes; pf_off += 64) {
                        ::hwy::Prefetch(next_k + pf_off);
                    }
                }

                float score = 0.0f;
                // Calculate byte pointer to the specific K token vector
                const int slot_idx = slot_begin + t;
                const int physical_key_pos = block_abs_start + slot_idx;
                const int context_index = physical_key_pos - context_start_pos;
                int key_pos = physical_key_pos;
                if (sliding_window >= 0 && mask_history_kept >= 0) {
                    if (context_index < mask_history_kept) {
                        key_pos = context_index < mask_sink_kept
                                      ? context_index
                                      : (mask_tail_start + (context_index - mask_sink_kept));
                    } else {
                        key_pos = query_pos + (context_index - mask_history_kept);
                    }
                }
                if (query_pos >= 0 && (key_pos > query_pos ||
                                       (sliding_window >= 0 && key_pos < (query_pos - sliding_window)))) {
                    block_scores[t] = -INFINITY;
                    continue;
                }
                const uint8_t* k_ptr_bytes = k_block_base + slot_idx * k_slot_stride_bytes + k_head_offset_bytes;

                if (cache_type == 8) {  // Q8_0
                    score = DotProductQ8_0(d, q_head, k_ptr_bytes, qk_head_dim);
                } else if (cache_type == 4) {  // Q4_0
                    score = DotProductQ4_0(d, q_head, k_ptr_bytes, qk_head_dim);
                } else if (cache_type == 1) {  // F16
                    score = DotProductF16(d, q_head, reinterpret_cast<const uint16_t*>(k_ptr_bytes), qk_head_dim);
                } else {  // F32
                    score = DotProductF32(d, q_head, reinterpret_cast<const float*>(k_ptr_bytes), qk_head_dim);
                }

                score *= scale;
                if (logit_softcap > 0.0f && std::isfinite(score)) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
                block_scores[t] = score;
                if (score > m_block) m_block = score;
            }

            // 2. Softmax / Rescale
            float m_new = std::max(m_prev, m_block);
            float alpha = FastExpScalar(m_prev - m_new);
            float d_block = 0.0f;

            // Rescale existing output
            if (alpha != 1.0f) {
                auto v_alpha = hn::Set(d, alpha);
                int i = 0;
                for (; i <= v_head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
                    auto val = hn::LoadU(d, out_head + i);
                    val = hn::Mul(val, v_alpha);
                    hn::StoreU(val, d, out_head + i);
                }
                for (; i < v_head_dim; ++i) {
                    out_head[i] *= alpha;
                }
            }

            // 3. Vectorized exp for block_scores, then accumulate
            {
                const auto v_m_new = hn::Set(d, m_new);
                // Process BLOCK_SIZE scores in SIMD lanes
                int t = 0;
                const int simd_lanes = static_cast<int>(hn::Lanes(d));
                for (; t <= num_tokens - simd_lanes; t += simd_lanes) {
                    auto vs = hn::LoadU(d, block_scores + t);
                    vs = hn::Sub(vs, v_m_new);
                    auto vexp = FastExpHwy(d, vs);
                    hn::StoreU(vexp, d, block_scores + t);
                }
                for (; t < num_tokens; ++t) {
                    block_scores[t] = FastExpScalar(block_scores[t] - m_new);
                }
                // Sum exp'd scores
                for (t = 0; t < num_tokens; ++t) {
                    d_block += block_scores[t];
                }
            }

            for (int t = 0; t < num_tokens; ++t) {
                float p = block_scores[t];

                const int slot_idx = slot_begin + t;
                const uint8_t* v_ptr = v_block_base + slot_idx * v_slot_stride_bytes + v_head_offset_bytes;

                // Prefetch next token's V head: all cache lines.
                const int pf_tv = t + prefetch_tokens;
                if (pf_tv < num_tokens) {
                    const int next_slot = slot_begin + pf_tv;
                    const uint8_t* next_v = v_block_base + next_slot * v_slot_stride_bytes + v_head_offset_bytes;
                    for (size_t pf_off = 0; pf_off < v_head_bytes; pf_off += 64) {
                        ::hwy::Prefetch(next_v + pf_off);
                    }
                }

                // Accumulate based on type
                if (cache_type == 0) {  // F32
                    AccumulateF32(d, out_head, reinterpret_cast<const float*>(v_ptr), p, v_head_dim);
                } else if (cache_type == 1) {  // F16
                    AccumulateF16(d, out_head, reinterpret_cast<const uint16_t*>(v_ptr), p, v_head_dim);
                } else if (cache_type == 8) {  // Q8_0
                    AccumulateQ8_0(d, out_head, v_ptr, p, v_head_dim);
                } else if (cache_type == 4) {  // Q4_0
                    AccumulateQ4_0(d, out_head, v_ptr, p, v_head_dim);
                }
            }
            // Update global stats
            d_prev = d_prev * alpha + d_block;
            m_prev = m_new;
        }

        // Final Normalization
        if (!(d_prev > 0.0f) || !std::isfinite(d_prev)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }
        float inv_sum = 1.0f / (d_prev + 1e-6f);
        auto v_inv_sum = hn::Set(d, inv_sum);
        int i = 0;
        for (; i <= v_head_dim - static_cast<int>(hn::Lanes(d)); i += hn::Lanes(d)) {
            auto val = hn::LoadU(d, out_head + i);
            val = hn::Mul(val, v_inv_sum);
            hn::StoreU(val, d, out_head + i);
        }
        for (; i < v_head_dim; ++i) {
            out_head[i] *= inv_sum;
        }
    }
}


}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();


#if HWY_ONCE
#include "densecore/hal/tensor.h"
#include "densecore/kernels/paged_attention.h"
#include <iostream>

namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(PagedAttentionImpl);

void PagedAttention_Hwy(const float* query, const void* const* k_block_ptrs, const void* const* v_block_ptrs,
                        int32_t cache_type, int32_t num_heads, int32_t qk_head_dim, int32_t v_head_dim,
                        int32_t n_head_kv, int32_t block_table_size, int32_t context_len, int32_t context_start_pos,
                        int32_t query_pos, int32_t sliding_window,
                        int32_t mask_history_kept, int32_t mask_sink_kept, int32_t mask_tail_start,
                        int64_t k_head_stride_bytes, int64_t k_slot_stride_bytes, int64_t v_head_stride_bytes,
                        int64_t v_slot_stride_bytes, float scale, float logit_softcap, float* output, int32_t head_start,
                        int32_t head_end, int32_t num_heads_total) {

    HWY_DYNAMIC_DISPATCH(PagedAttentionImpl)
    (query, k_block_ptrs, v_block_ptrs, cache_type, num_heads, qk_head_dim, v_head_dim, n_head_kv, block_table_size,
     context_len, context_start_pos, query_pos, sliding_window, mask_history_kept, mask_sink_kept, mask_tail_start,
     k_head_stride_bytes, k_slot_stride_bytes, v_head_stride_bytes, v_slot_stride_bytes, scale, logit_softcap, output,
     head_start, head_end, num_heads_total);
}

}  // namespace hwy_kernels

namespace kernels {

void PagedAttention(const Tensor& query, const PagedKVCache& cache, int layer, const std::vector<int>& block_table,
                    const PagedAttentionConfig& config, Tensor* output, int head_start, int head_end,
                    int num_heads_total) {
    // Input validation
    if (!query.data || !output->data || block_table.empty()) return;

    int num_heads = (int)query.shape[0];
    int head_dim = (int)query.shape[1];
    const int v_head_dim = cache.v_head_dim > 0 ? cache.v_head_dim : head_dim;
    if (num_heads_total <= 0) {
        num_heads_total = num_heads;
    }
    if (head_end < 0 || head_end > num_heads) {
        head_end = num_heads;
    }
    if (head_start < 0) {
        head_start = 0;
    }
    if (head_start >= head_end) {
        return;
    }

    // Check supported types
    // Note: Assuming query and output are always F32 for high precision accumulation
    if (query.dtype != DType::F32 || output->dtype != DType::F32) {
        throw std::runtime_error("[PagedAttention] Error: Query and Output must be F32");
    }
    if (output->ndim != 2 || output->shape[0] != num_heads || output->shape[1] != v_head_dim) {
        throw std::runtime_error("[PagedAttention] Error: Output shape must match [num_heads, v_head_dim]");
    }

    // Determine cache type ID for Highway kernel
    int32_t cache_type_id = -1;
    if (cache.cache_type == GGML_TYPE_F32) {
        cache_type_id = 0;
    } else if (cache.cache_type == GGML_TYPE_F16) {
        cache_type_id = 1;
    } else if (cache.cache_type == GGML_TYPE_Q4_0) {
        cache_type_id = 4;
    } else if (cache.cache_type == GGML_TYPE_Q8_0) {
        cache_type_id = 8;
    } else {
        throw std::runtime_error("[PagedAttention] Error: Unsupported cache type " +
                                 std::to_string((int)cache.cache_type) + ". Supported: F32, F16, Q4_0, Q8_0");
    }

    const auto k_layout = cache.GetBlockLayout();
    const auto v_layout = cache.GetVBlockLayout();
    std::vector<const void*> k_block_ptrs(static_cast<size_t>(block_table.size()), nullptr);
    std::vector<const void*> v_block_ptrs(static_cast<size_t>(block_table.size()), nullptr);
    for (size_t i = 0; i < block_table.size(); ++i) {
        const int block_id = block_table[i];
        if (block_id < 0 || block_id >= cache.max_blocks) {
            continue;
        }
        k_block_ptrs[i] = cache.GetKBlockPtr(block_id, layer);
        v_block_ptrs[i] = cache.GetVBlockPtr(block_id, layer);
    }

    densecore::hwy_kernels::PagedAttention_Hwy(
        (const float*)query.data, k_block_ptrs.data(), v_block_ptrs.data(), cache_type_id, num_heads, head_dim,
        v_head_dim, cache.n_head_kv, (int32_t)block_table.size(), config.context_len, config.context_start_pos,
        config.query_pos, config.sliding_window, -1, 0, 0, static_cast<int64_t>(k_layout.head_stride_bytes),
        static_cast<int64_t>(k_layout.slot_stride_bytes), static_cast<int64_t>(v_layout.head_stride_bytes),
        static_cast<int64_t>(v_layout.slot_stride_bytes), config.scale, config.logit_softcap, (float*)output->data,
        head_start, head_end, num_heads_total);
}

}  // namespace kernels
}  // namespace densecore
#endif
