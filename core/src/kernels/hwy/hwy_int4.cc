/**
 * @file hwy_int4.cc
 * @brief High-performance INT4 GEMV/GEMM kernels via Google Highway
 *
 * Key optimizations over the naive per-row implementation:
 *
 * 1. N-Blocked GEMV (M=1): Process 4 output channels simultaneously.
 *    The input vector is loaded ONCE and reused across 4 weight rows.
 *    Arithmetic Intensity: 2.67 FLOP/B vs 0.89 FLOP/B (3× improvement).
 *
 * 2. M-Blocked GEMM (M>1): Load and unpack each weight tile ONCE,
 *    apply to 4 batch input rows. Eliminates 3/4 of the expensive
 *    INT4→FP32 unpack chain (TableLookupBytes + shifts + promotes).
 *
 * 3. Fused Dequantization: scale*(w-zero) → MulAdd(scale, w, -scale*zero)
 *    Replaces Sub+Mul with a single FMA instruction.
 *
 * 4. Register Blocking: 4 independent accumulators keep the FMA pipeline
 *    full (hide 4-cycle FMA latency on modern CPUs).
 *
 * Unified for AVX-512, AVX2, NEON, SVE, and Scalar via Highway dispatch.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_int4.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include <hwy/cache_control.h>

#include "kernels/hwy/hwy_kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE int PackedBytesForInt4(int k) { return (k + 1) / 2; }
HWY_INLINE int PrefetchVecItersAhead(int lanes) {
    if (lanes >= 16) return 4;  // AVX-512/SVE-wide: pull further ahead.
    if (lanes >= 8) return 2;   // AVX2/NEON: balanced default.
    return 1;                   // Scalar/narrow vectors.
}

// ============================================================================
// Scalar INT4 nibble extraction (for tail loops)
// ============================================================================
HWY_INLINE float UnpackNibbleScalar(const uint8_t* packed, int k, float scale, float zero) {
    const uint8_t pb = packed[k / 2];
    int8_t q = (k & 1) ? static_cast<int8_t>((pb >> 4) & 0xF) : static_cast<int8_t>(pb & 0xF);
    if (q & 0x8) q |= static_cast<int8_t>(0xF0);  // sign-extend 4-bit → 8-bit
    return scale * (static_cast<float>(q) - zero);
}

// ============================================================================
// Prepack: Linear copy (layout already optimal for sequential access)
// ============================================================================
void PrepackInt4WeightsInterleavedImpl(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst, int K, int N,
                                       int group_size, int block_size) {
    if (!src || !dst || K <= 0 || N <= 0 || group_size <= 0) return;
    if (block_size <= 0 || (block_size & 1) != 0) block_size = 32;
    (void)block_size;

    const int packed_K = PackedBytesForInt4(K);

    // Keep prepack single-threaded here; inference-time parallelism is managed
    // by DenseCore/GGML thread pools and should not nest with OpenMP workers.
    for (int row = 0; row < N; ++row) {
        const uint8_t* src_row = src + static_cast<int64_t>(row) * packed_K;
        uint8_t* dst_row = dst + static_cast<int64_t>(row) * packed_K;
        std::memcpy(dst_row, src_row, packed_K);
    }
}

// ============================================================================
// N-Blocked GEMV: M=1, process 4 output rows simultaneously
// ============================================================================
//
// Memory traffic analysis per vectorized iteration (AVX-512, 16 float lanes):
//
//   BEFORE (1 row/iter):
//     Load: 64B input + 8B weights = 72B
//     Compute: 16 × (MulAdd_dequant + MulAdd_accum) = 16 × 4 = 64 FLOPs
//     AI = 64 / 72 ≈ 0.89 FLOP/B
//
//   AFTER (4 rows/iter, N_BLOCK=4):
//     Load: 64B input (REUSED 4×) + 4 × 8B weights = 96B
//     Compute: 4 × 64 = 256 FLOPs
//     AI = 256 / 96 ≈ 2.67 FLOP/B  →  3× bandwidth efficiency
//
// The 4 independent accumulator chains (v_acc0..3) provide enough ILP
// to saturate the FMA pipeline (4-cycle latency × 2 ports = 8 in-flight FMAs).
//
// ============================================================================
void GemvInt4Impl(float* HWY_RESTRICT output, const float* HWY_RESTRICT input, const uint8_t* HWY_RESTRICT weights,
                  const float* HWY_RESTRICT scales, const float* HWY_RESTRICT zeros, int K, int N, int group_size,
                  int n_start, int n_end) {
    if (!output || !input || !weights || K <= 0 || N <= 0 || group_size <= 0) return;
    if ((group_size & 1) != 0) return;

    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (n_start >= n_end) return;

    const hn::ScalableTag<float> df;
    using Di8 = hn::Rebind<int8_t, hn::ScalableTag<float>>;
    using Du8 = hn::Rebind<uint8_t, hn::ScalableTag<float>>;
    using Di16 = hn::Rebind<int16_t, hn::ScalableTag<float>>;
    using Di32 = hn::Rebind<int32_t, hn::ScalableTag<float>>;

    const Di8 di8;
    const Du8 du8;
    const Di16 di16;
    const Di32 di32;

    const size_t lanes_f = hn::Lanes(df);
    const int lanes = static_cast<int>(lanes_f);
    const int vec_step = (lanes >= 2 && (lanes % 2 == 0)) ? lanes : 0;
    const int prefetch_vec_iters_ahead = PrefetchVecItersAhead(lanes);

    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int packed_K = PackedBytesForInt4(K);
    if (num_full_groups > 0 && (!scales || !zeros)) return;

    // Precomputed nibble extraction masks (invariant across all iterations)
    //   tbl_indices: byte duplication [0,0,1,1,2,2,...] for TableLookupBytes
    //   mask_odd: selects high nibble (odd lanes) vs low nibble (even lanes)
    const auto iota_u8 = hn::Iota(du8, 0);
    const auto tbl_indices = hn::ShiftRight<1>(iota_u8);
    const auto mask_bit = hn::And(iota_u8, hn::Set(du8, 1));
    const auto mask_odd_u8 = hn::Eq(mask_bit, hn::Set(du8, 1));
    const auto mask_odd = hn::RebindMask(di8, mask_odd_u8);

    // ========================================================================
    // N-BLOCKED MAIN LOOP: compile-time blocking by SIMD width
    // ========================================================================
    // Register budget (hot loop body, worst case on N_BLOCK=8 path):
    //   4 accumulators (v_acc0..3)           = 4 regs
    //   1 input vector (v_in)                = 1 reg
    //   2 masks (tbl_indices, mask_odd)      = 2 regs
    //   2 scale+bias per active row          = 2 regs (others spill to L1)
    //   ~5 temporaries for unpack chain      = 5 regs
    //   Total: ~14 registers → fits AVX2(16) and AVX-512(32)
    // ========================================================================

    constexpr int N_BLOCK = (HWY_MAX_BYTES >= 64) ? 8 : 4;
    int n = n_start;

    if constexpr (N_BLOCK == 8) {
        // AVX-512/SVE-wide targets: increase register blocking to 8 accumulators.
        for (; n + 8 <= n_end; n += 8) {
            auto v_acc0 = hn::Zero(df);
            auto v_acc1 = hn::Zero(df);
            auto v_acc2 = hn::Zero(df);
            auto v_acc3 = hn::Zero(df);
            auto v_acc4 = hn::Zero(df);
            auto v_acc5 = hn::Zero(df);
            auto v_acc6 = hn::Zero(df);
            auto v_acc7 = hn::Zero(df);
            float scalar_acc[8] = {};

            for (int g = 0; g < num_full_groups; ++g) {
                const int k_base = g * group_size;
                const float* a_ptr = input + k_base;

                const float s0 = scales[static_cast<int64_t>(n + 0) * num_full_groups + g];
                const float s1 = scales[static_cast<int64_t>(n + 1) * num_full_groups + g];
                const float s2 = scales[static_cast<int64_t>(n + 2) * num_full_groups + g];
                const float s3 = scales[static_cast<int64_t>(n + 3) * num_full_groups + g];
                const float s4 = scales[static_cast<int64_t>(n + 4) * num_full_groups + g];
                const float s5 = scales[static_cast<int64_t>(n + 5) * num_full_groups + g];
                const float s6 = scales[static_cast<int64_t>(n + 6) * num_full_groups + g];
                const float s7 = scales[static_cast<int64_t>(n + 7) * num_full_groups + g];

                const float z0 = zeros[static_cast<int64_t>(n + 0) * num_full_groups + g];
                const float z1 = zeros[static_cast<int64_t>(n + 1) * num_full_groups + g];
                const float z2 = zeros[static_cast<int64_t>(n + 2) * num_full_groups + g];
                const float z3 = zeros[static_cast<int64_t>(n + 3) * num_full_groups + g];
                const float z4 = zeros[static_cast<int64_t>(n + 4) * num_full_groups + g];
                const float z5 = zeros[static_cast<int64_t>(n + 5) * num_full_groups + g];
                const float z6 = zeros[static_cast<int64_t>(n + 6) * num_full_groups + g];
                const float z7 = zeros[static_cast<int64_t>(n + 7) * num_full_groups + g];

                const auto v_s0 = hn::Set(df, s0);
                const auto v_s1 = hn::Set(df, s1);
                const auto v_s2 = hn::Set(df, s2);
                const auto v_s3 = hn::Set(df, s3);
                const auto v_s4 = hn::Set(df, s4);
                const auto v_s5 = hn::Set(df, s5);
                const auto v_s6 = hn::Set(df, s6);
                const auto v_s7 = hn::Set(df, s7);

                const auto v_nsz0 = hn::Set(df, -s0 * z0);
                const auto v_nsz1 = hn::Set(df, -s1 * z1);
                const auto v_nsz2 = hn::Set(df, -s2 * z2);
                const auto v_nsz3 = hn::Set(df, -s3 * z3);
                const auto v_nsz4 = hn::Set(df, -s4 * z4);
                const auto v_nsz5 = hn::Set(df, -s5 * z5);
                const auto v_nsz6 = hn::Set(df, -s6 * z6);
                const auto v_nsz7 = hn::Set(df, -s7 * z7);

                const uint8_t* w0 = weights + static_cast<int64_t>(n + 0) * packed_K + g * (group_size / 2);
                const uint8_t* w1 = weights + static_cast<int64_t>(n + 1) * packed_K + g * (group_size / 2);
                const uint8_t* w2 = weights + static_cast<int64_t>(n + 2) * packed_K + g * (group_size / 2);
                const uint8_t* w3 = weights + static_cast<int64_t>(n + 3) * packed_K + g * (group_size / 2);
                const uint8_t* w4 = weights + static_cast<int64_t>(n + 4) * packed_K + g * (group_size / 2);
                const uint8_t* w5 = weights + static_cast<int64_t>(n + 5) * packed_K + g * (group_size / 2);
                const uint8_t* w6 = weights + static_cast<int64_t>(n + 6) * packed_K + g * (group_size / 2);
                const uint8_t* w7 = weights + static_cast<int64_t>(n + 7) * packed_K + g * (group_size / 2);

                int k = 0;
                for (; vec_step > 0 && k + vec_step <= group_size; k += vec_step) {
                    if (k + vec_step * prefetch_vec_iters_ahead < group_size) {
                        const int pf_k = k + vec_step * prefetch_vec_iters_ahead;
                        const int pf_byte_off = pf_k / 2;
                        ::hwy::Prefetch(a_ptr + pf_k);
                        ::hwy::Prefetch(w0 + pf_byte_off);
                        ::hwy::Prefetch(w1 + pf_byte_off);
                        ::hwy::Prefetch(w2 + pf_byte_off);
                        ::hwy::Prefetch(w3 + pf_byte_off);
                        ::hwy::Prefetch(w4 + pf_byte_off);
                        ::hwy::Prefetch(w5 + pf_byte_off);
                        ::hwy::Prefetch(w6 + pf_byte_off);
                        ::hwy::Prefetch(w7 + pf_byte_off);
                    }

                    const auto v_in = hn::LoadU(df, a_ptr + k);
                    const int byte_off = k / 2;

                    auto accumulate_row = [&](const uint8_t* w_ptr, const auto& v_s, const auto& v_nsz, auto& v_acc) {
                        auto v_bytes = hn::LoadN(du8, w_ptr + byte_off, lanes_f / 2);
                        auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                        auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                        auto v_hi = hn::ShiftRight<4>(v_exp);
                        auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                        auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                        v_acc = hn::MulAdd(v_in, hn::MulAdd(v_s, v_w_f, v_nsz), v_acc);
                    };

                    accumulate_row(w0, v_s0, v_nsz0, v_acc0);
                    accumulate_row(w1, v_s1, v_nsz1, v_acc1);
                    accumulate_row(w2, v_s2, v_nsz2, v_acc2);
                    accumulate_row(w3, v_s3, v_nsz3, v_acc3);
                    accumulate_row(w4, v_s4, v_nsz4, v_acc4);
                    accumulate_row(w5, v_s5, v_nsz5, v_acc5);
                    accumulate_row(w6, v_s6, v_nsz6, v_acc6);
                    accumulate_row(w7, v_s7, v_nsz7, v_acc7);
                }

                if (k < group_size) {
                    const uint8_t* w_ptrs[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
                    const float s_arr[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
                    const float z_arr[8] = {z0, z1, z2, z3, z4, z5, z6, z7};
                    for (; k < group_size; ++k) {
                        const float a = a_ptr[k];
                        for (int j = 0; j < 8; ++j) {
                            scalar_acc[j] += a * UnpackNibbleScalar(w_ptrs[j], k, s_arr[j], z_arr[j]);
                        }
                    }
                }
            }

            output[n + 0] = hn::ReduceSum(df, v_acc0) + scalar_acc[0];
            output[n + 1] = hn::ReduceSum(df, v_acc1) + scalar_acc[1];
            output[n + 2] = hn::ReduceSum(df, v_acc2) + scalar_acc[2];
            output[n + 3] = hn::ReduceSum(df, v_acc3) + scalar_acc[3];
            output[n + 4] = hn::ReduceSum(df, v_acc4) + scalar_acc[4];
            output[n + 5] = hn::ReduceSum(df, v_acc5) + scalar_acc[5];
            output[n + 6] = hn::ReduceSum(df, v_acc6) + scalar_acc[6];
            output[n + 7] = hn::ReduceSum(df, v_acc7) + scalar_acc[7];

            if (remainder > 0) {
                for (int j = 0; j < 8; ++j) {
                    const int nj = n + j;
                    const float s_r = (num_full_groups > 0)
                                          ? scales[static_cast<int64_t>(nj) * num_full_groups + num_full_groups - 1]
                                          : 1.0f;
                    const float z_r = (num_full_groups > 0)
                                          ? zeros[static_cast<int64_t>(nj) * num_full_groups + num_full_groups - 1]
                                          : 0.0f;
                    float rem = 0.0f;
                    for (int kk = num_full_groups * group_size; kk < K; ++kk) {
                        rem += input[kk] *
                               UnpackNibbleScalar(weights + static_cast<int64_t>(nj) * packed_K, kk, s_r, z_r);
                    }
                    output[nj] += rem;
                }
            }
        }
    } else {
        for (; n + N_BLOCK <= n_end; n += N_BLOCK) {
            // 4 independent accumulators → saturate FMA ports
            auto v_acc0 = hn::Zero(df);
            auto v_acc1 = hn::Zero(df);
            auto v_acc2 = hn::Zero(df);
            auto v_acc3 = hn::Zero(df);
            float scalar_acc[N_BLOCK] = {};

            for (int g = 0; g < num_full_groups; ++g) {
                const int k_base = g * group_size;
                const float* a_ptr = input + k_base;

                // ----------------------------------------------------------------
                // Per-group quantization parameters for all 4 output rows.
                // Precompute neg_scale_zero = -(scale * zero) to fuse dequant:
                //   scale * (w - zero) = FMA(scale, w, -scale*zero)
                // This replaces Sub + Mul with a single FMA instruction.
                // ----------------------------------------------------------------
                const float s0 = scales[static_cast<int64_t>(n + 0) * num_full_groups + g];
                const float s1 = scales[static_cast<int64_t>(n + 1) * num_full_groups + g];
                const float s2 = scales[static_cast<int64_t>(n + 2) * num_full_groups + g];
                const float s3 = scales[static_cast<int64_t>(n + 3) * num_full_groups + g];
                const float z0 = zeros[static_cast<int64_t>(n + 0) * num_full_groups + g];
                const float z1 = zeros[static_cast<int64_t>(n + 1) * num_full_groups + g];
                const float z2 = zeros[static_cast<int64_t>(n + 2) * num_full_groups + g];
                const float z3 = zeros[static_cast<int64_t>(n + 3) * num_full_groups + g];

                const auto v_s0 = hn::Set(df, s0);
                const auto v_s1 = hn::Set(df, s1);
                const auto v_s2 = hn::Set(df, s2);
                const auto v_s3 = hn::Set(df, s3);
                // Fused dequant bias: -(scale * zero), precomputed per group
                const auto v_nsz0 = hn::Set(df, -s0 * z0);
                const auto v_nsz1 = hn::Set(df, -s1 * z1);
                const auto v_nsz2 = hn::Set(df, -s2 * z2);
                const auto v_nsz3 = hn::Set(df, -s3 * z3);

                // Weight base pointers for the 4 output rows
                const uint8_t* w0 = weights + static_cast<int64_t>(n + 0) * packed_K + g * (group_size / 2);
                const uint8_t* w1 = weights + static_cast<int64_t>(n + 1) * packed_K + g * (group_size / 2);
                const uint8_t* w2 = weights + static_cast<int64_t>(n + 2) * packed_K + g * (group_size / 2);
                const uint8_t* w3 = weights + static_cast<int64_t>(n + 3) * packed_K + g * (group_size / 2);

                int k = 0;
                for (; vec_step > 0 && k + vec_step <= group_size; k += vec_step) {
                    if (k + vec_step * prefetch_vec_iters_ahead < group_size) {
                        const int pf_k = k + vec_step * prefetch_vec_iters_ahead;
                        const int pf_byte_off = pf_k / 2;
                        ::hwy::Prefetch(a_ptr + pf_k);
                        ::hwy::Prefetch(w0 + pf_byte_off);
                        ::hwy::Prefetch(w1 + pf_byte_off);
                        ::hwy::Prefetch(w2 + pf_byte_off);
                        ::hwy::Prefetch(w3 + pf_byte_off);
                    }

                    // Load input vector ONCE (the key bandwidth optimization)
                    const auto v_in = hn::LoadU(df, a_ptr + k);
                    const int byte_off = k / 2;

                    // ---- Unpack + dequant + accumulate: Row 0 ----
                    {
                        auto v_bytes = hn::LoadN(du8, w0 + byte_off, lanes_f / 2);
                        auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                        auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                        auto v_hi = hn::ShiftRight<4>(v_exp);
                        auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                        auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                        auto v_dq = hn::MulAdd(v_s0, v_w_f, v_nsz0);
                        v_acc0 = hn::MulAdd(v_in, v_dq, v_acc0);
                    }

                    // ---- Row 1: Same input, different weight row ----
                    {
                        auto v_bytes = hn::LoadN(du8, w1 + byte_off, lanes_f / 2);
                        auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                        auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                        auto v_hi = hn::ShiftRight<4>(v_exp);
                        auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                        auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                        v_acc1 = hn::MulAdd(v_in, hn::MulAdd(v_s1, v_w_f, v_nsz1), v_acc1);
                    }

                    // ---- Row 2 ----
                    {
                        auto v_bytes = hn::LoadN(du8, w2 + byte_off, lanes_f / 2);
                        auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                        auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                        auto v_hi = hn::ShiftRight<4>(v_exp);
                        auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                        auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                        v_acc2 = hn::MulAdd(v_in, hn::MulAdd(v_s2, v_w_f, v_nsz2), v_acc2);
                    }

                    // ---- Row 3 ----
                    {
                        auto v_bytes = hn::LoadN(du8, w3 + byte_off, lanes_f / 2);
                        auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                        auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                        auto v_hi = hn::ShiftRight<4>(v_exp);
                        auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                        auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                        v_acc3 = hn::MulAdd(v_in, hn::MulAdd(v_s3, v_w_f, v_nsz3), v_acc3);
                    }
                }

                // Scalar tail: handles group_size % vec_step remainder
                if (k < group_size) {
                    const uint8_t* w_ptrs[N_BLOCK] = {w0, w1, w2, w3};
                    const float s_arr[N_BLOCK] = {s0, s1, s2, s3};
                    const float z_arr[N_BLOCK] = {z0, z1, z2, z3};

                    for (; k < group_size; ++k) {
                        const float a = a_ptr[k];
                        for (int j = 0; j < N_BLOCK; ++j) {
                            scalar_acc[j] += a * UnpackNibbleScalar(w_ptrs[j], k, s_arr[j], z_arr[j]);
                        }
                    }
                }
            }

            // Horizontal reduction: vector accumulators + scalar tail
            output[n + 0] = hn::ReduceSum(df, v_acc0) + scalar_acc[0];
            output[n + 1] = hn::ReduceSum(df, v_acc1) + scalar_acc[1];
            output[n + 2] = hn::ReduceSum(df, v_acc2) + scalar_acc[2];
            output[n + 3] = hn::ReduceSum(df, v_acc3) + scalar_acc[3];

            // K remainder (elements beyond last full group)
            if (remainder > 0) {
                for (int j = 0; j < N_BLOCK; ++j) {
                    const int nj = n + j;
                    const float s_r = (num_full_groups > 0)
                                          ? scales[static_cast<int64_t>(nj) * num_full_groups + num_full_groups - 1]
                                          : 1.0f;
                    const float z_r = (num_full_groups > 0)
                                          ? zeros[static_cast<int64_t>(nj) * num_full_groups + num_full_groups - 1]
                                          : 0.0f;
                    float rem = 0.0f;
                    for (int kk = num_full_groups * group_size; kk < K; ++kk) {
                        rem += input[kk] *
                               UnpackNibbleScalar(weights + static_cast<int64_t>(nj) * packed_K, kk, s_r, z_r);
                    }
                    output[nj] += rem;
                }
            }
        }
    }

    // ========================================================================
    // N-TAIL: Process remaining rows (< N_BLOCK) one at a time
    // ========================================================================
    // Uses the fused MulAdd dequant optimization but without N-blocking.
    // ========================================================================
    for (; n < n_end; ++n) {
        float sum = 0.0f;
        auto v_sum = hn::Zero(df);

        for (int g = 0; g < num_full_groups; ++g) {
            const int k_base = g * group_size;
            const float* a_ptr = input + k_base;
            const float s = scales[static_cast<int64_t>(n) * num_full_groups + g];
            const float z = zeros[static_cast<int64_t>(n) * num_full_groups + g];

            const auto v_s = hn::Set(df, s);
            const auto v_nsz = hn::Set(df, -s * z);

            const uint8_t* w_ptr = weights + static_cast<int64_t>(n) * packed_K + g * (group_size / 2);

            int k = 0;
            for (; vec_step > 0 && k + vec_step <= group_size; k += vec_step) {
                if (k + vec_step * prefetch_vec_iters_ahead < group_size) {
                    const int pf_k = k + vec_step * prefetch_vec_iters_ahead;
                    ::hwy::Prefetch(a_ptr + pf_k);
                    ::hwy::Prefetch(w_ptr + pf_k / 2);
                }

                auto v_in = hn::LoadU(df, a_ptr + k);
                auto v_bytes = hn::LoadN(du8, w_ptr + k / 2, lanes_f / 2);
                auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                auto v_hi = hn::ShiftRight<4>(v_exp);
                auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                v_sum = hn::MulAdd(v_in, hn::MulAdd(v_s, v_w_f, v_nsz), v_sum);
            }

            for (; k < group_size; ++k) {
                sum += a_ptr[k] * UnpackNibbleScalar(w_ptr, k, s, z);
            }
        }

        sum += hn::ReduceSum(df, v_sum);

        if (remainder > 0) {
            const float s_r = (num_full_groups > 0)
                                  ? scales[static_cast<int64_t>(n) * num_full_groups + num_full_groups - 1]
                                  : 1.0f;
            const float z_r =
                (num_full_groups > 0) ? zeros[static_cast<int64_t>(n) * num_full_groups + num_full_groups - 1] : 0.0f;
            for (int k = num_full_groups * group_size; k < K; ++k) {
                sum += input[k] * UnpackNibbleScalar(weights + static_cast<int64_t>(n) * packed_K, k, s_r, z_r);
            }
        }

        output[n] = sum;
    }
}

// ============================================================================
// M-Blocked GEMM: M>1, weight reuse across batch dimension
// ============================================================================
//
// For batch inference (M=4 typical during speculative decoding or small-batch
// prefill), the dominant cost is weight memory traffic. Each weight element
// (INT4 packed byte) requires an expensive unpack chain:
//
//   LoadN → TableLookupBytes → ShiftLeft → ShiftRight → ShiftRight →
//   IfThenElse → PromoteTo × 2 → ConvertTo  ≈ 10 instructions
//
// By loading and unpacking weights ONCE per (n, k_tile) and applying them
// to M_BLOCK=4 input rows, we eliminate 3/4 of the unpack compute:
//
//   Without M-blocking: 4 × 10 = 40 unpack instructions per vec_step
//   With M_BLOCK=4:     1 × 10 = 10 unpack instructions per vec_step
//   Savings: 30 instructions/iter → ~25% fewer total instructions
//
// Additional benefit: the 4 MulAdd instructions after unpack are independent
// (different accumulators), providing ILP for superscalar execution.
//
// Memory traffic (AVX-512, vec_step=16, per k-iteration):
//   Weight:  8B  (loaded once, REUSED 4×)
//   Inputs:  4 × 64B = 256B
//   Total:   264B  for  256 FLOPs → AI = 0.97 FLOP/B
//   vs naive: 4 × 72B = 288B for 256 FLOPs → AI = 0.89 FLOP/B
//
// The real win is the compute savings (25%) not bandwidth (9%).
// For memory-bound workloads, combine with N-tiling at the thread level.
//
// Layout:
//   input:  [M, K] row-major → input[m * K + k]
//   weights: [N, packed_K]   → weights[n * packed_K + byte]
//   scales:  [N, num_groups] → scales[n * num_groups + g]
//   zeros:   [N, num_groups] → zeros[n * num_groups + g]
//   output:  [M, N] row-major → output[m * N + n]
//
// ============================================================================
void GemmInt4BatchedImpl(float* HWY_RESTRICT output, const float* HWY_RESTRICT input,
                         const uint8_t* HWY_RESTRICT weights, const float* HWY_RESTRICT scales,
                         const float* HWY_RESTRICT zeros, int M, int K, int N, int group_size, int m_start,
                         int m_end, int n_start, int n_end, size_t input_stride_bytes) {
    if (!output || !input || !weights || M <= 0 || K <= 0 || N <= 0 || group_size <= 0) return;
    if ((group_size & 1) != 0) return;
    if (input_stride_bytes < static_cast<size_t>(K) * sizeof(float)) return;

    m_start = std::max(0, m_start);
    m_end = std::min(M, m_end);
    n_start = std::max(0, n_start);
    n_end = std::min(N, n_end);
    if (m_start >= m_end || n_start >= n_end) return;

    // Convert byte stride to float element stride for pointer arithmetic.
    // Callers guarantee input_stride_bytes is a multiple of sizeof(float).
    const size_t input_stride = input_stride_bytes / sizeof(float);

    // M=1 fast path: delegate to N-blocked GEMV (better arithmetic intensity)
    if (m_start == 0 && m_end == 1 && M == 1) {
        GemvInt4Impl(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
        return;
    }

    const hn::ScalableTag<float> df;
    using Di8 = hn::Rebind<int8_t, hn::ScalableTag<float>>;
    using Du8 = hn::Rebind<uint8_t, hn::ScalableTag<float>>;
    using Di16 = hn::Rebind<int16_t, hn::ScalableTag<float>>;
    using Di32 = hn::Rebind<int32_t, hn::ScalableTag<float>>;

    const Di8 di8;
    const Du8 du8;
    const Di16 di16;
    const Di32 di32;

    const size_t lanes_f = hn::Lanes(df);
    const int lanes = static_cast<int>(lanes_f);
    const int vec_step = (lanes >= 2 && (lanes % 2 == 0)) ? lanes : 0;
    const int prefetch_vec_iters_ahead = PrefetchVecItersAhead(lanes);

    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int packed_K = PackedBytesForInt4(K);
    if (num_full_groups > 0 && (!scales || !zeros)) return;

    const auto iota_u8 = hn::Iota(du8, 0);
    const auto tbl_indices = hn::ShiftRight<1>(iota_u8);
    const auto mask_bit = hn::And(iota_u8, hn::Set(du8, 1));
    const auto mask_odd_u8 = hn::Eq(mask_bit, hn::Set(du8, 1));
    const auto mask_odd = hn::RebindMask(di8, mask_odd_u8);

    // ========================================================================
    // COLUMN-FIRST M-BLOCKED GEMM (M_BLOCK=4):
    //   outer: n (output column range assigned by thread dispatcher)
    //   middle: k/group
    //   inner op: load ONE weight vector, apply to 4 batch rows
    // ========================================================================
    constexpr int M_BLOCK = 4;

    for (int n = n_start; n < n_end; ++n) {
        const uint8_t* w_row = weights + static_cast<int64_t>(n) * packed_K;
        const float s_tail = (num_full_groups > 0)
                                 ? scales[static_cast<int64_t>(n) * num_full_groups + num_full_groups - 1]
                                 : 1.0f;
        const float z_tail = (num_full_groups > 0)
                                 ? zeros[static_cast<int64_t>(n) * num_full_groups + num_full_groups - 1]
                                 : 0.0f;

        int m = m_start;
        for (; m + M_BLOCK <= m_end; m += M_BLOCK) {
            auto v_acc0 = hn::Zero(df);
            auto v_acc1 = hn::Zero(df);
            auto v_acc2 = hn::Zero(df);
            auto v_acc3 = hn::Zero(df);
            float scalar_acc0 = 0.0f;
            float scalar_acc1 = 0.0f;
            float scalar_acc2 = 0.0f;
            float scalar_acc3 = 0.0f;

            const float* in0 = input + static_cast<int64_t>(m + 0) * input_stride;
            const float* in1 = input + static_cast<int64_t>(m + 1) * input_stride;
            const float* in2 = input + static_cast<int64_t>(m + 2) * input_stride;
            const float* in3 = input + static_cast<int64_t>(m + 3) * input_stride;

            for (int g = 0; g < num_full_groups; ++g) {
                const int k_base = g * group_size;
                const float s = scales[static_cast<int64_t>(n) * num_full_groups + g];
                const float z = zeros[static_cast<int64_t>(n) * num_full_groups + g];
                const auto v_s = hn::Set(df, s);
                const auto v_nsz = hn::Set(df, -s * z);

                const uint8_t* w_ptr = w_row + g * (group_size / 2);

                int k = 0;
                for (; vec_step > 0 && k + vec_step <= group_size; k += vec_step) {
                    if (k + vec_step * prefetch_vec_iters_ahead < group_size) {
                        const int pf_k = k + vec_step * prefetch_vec_iters_ahead;
                        const int pf_byte_off = pf_k / 2;
                        const int pf_kk = k_base + pf_k;
                        ::hwy::Prefetch(w_ptr + pf_byte_off);
                        ::hwy::Prefetch(in0 + pf_kk);
                        ::hwy::Prefetch(in1 + pf_kk);
                        ::hwy::Prefetch(in2 + pf_kk);
                        ::hwy::Prefetch(in3 + pf_kk);
                    }

                    const int byte_off = k / 2;
                    auto v_bytes = hn::LoadN(du8, w_ptr + byte_off, lanes_f / 2);
                    auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                    auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                    auto v_hi = hn::ShiftRight<4>(v_exp);
                    auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                    auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                    auto v_dq = hn::MulAdd(v_s, v_w_f, v_nsz);

                    const int kk = k_base + k;
                    const auto v_in0 = hn::LoadU(df, in0 + kk);
                    const auto v_in1 = hn::LoadU(df, in1 + kk);
                    const auto v_in2 = hn::LoadU(df, in2 + kk);
                    const auto v_in3 = hn::LoadU(df, in3 + kk);
                    v_acc0 = hn::MulAdd(v_in0, v_dq, v_acc0);
                    v_acc1 = hn::MulAdd(v_in1, v_dq, v_acc1);
                    v_acc2 = hn::MulAdd(v_in2, v_dq, v_acc2);
                    v_acc3 = hn::MulAdd(v_in3, v_dq, v_acc3);
                }

                for (; k < group_size; ++k) {
                    const float dq = UnpackNibbleScalar(w_ptr, k, s, z);
                    const int kk = k_base + k;
                    scalar_acc0 += in0[kk] * dq;
                    scalar_acc1 += in1[kk] * dq;
                    scalar_acc2 += in2[kk] * dq;
                    scalar_acc3 += in3[kk] * dq;
                }
            }

            float out0 = hn::ReduceSum(df, v_acc0) + scalar_acc0;
            float out1 = hn::ReduceSum(df, v_acc1) + scalar_acc1;
            float out2 = hn::ReduceSum(df, v_acc2) + scalar_acc2;
            float out3 = hn::ReduceSum(df, v_acc3) + scalar_acc3;

            if (remainder > 0) {
                for (int kk = num_full_groups * group_size; kk < K; ++kk) {
                    const float dq = UnpackNibbleScalar(w_row, kk, s_tail, z_tail);
                    out0 += in0[kk] * dq;
                    out1 += in1[kk] * dq;
                    out2 += in2[kk] * dq;
                    out3 += in3[kk] * dq;
                }
            }

            output[static_cast<int64_t>(m + 0) * N + n] = out0;
            output[static_cast<int64_t>(m + 1) * N + n] = out1;
            output[static_cast<int64_t>(m + 2) * N + n] = out2;
            output[static_cast<int64_t>(m + 3) * N + n] = out3;
        }

        for (; m < m_end; ++m) {
            auto v_sum = hn::Zero(df);
            float sum = 0.0f;
            const float* in_m = input + static_cast<int64_t>(m) * input_stride;

            for (int g = 0; g < num_full_groups; ++g) {
                const int k_base = g * group_size;
                const float s = scales[static_cast<int64_t>(n) * num_full_groups + g];
                const float z = zeros[static_cast<int64_t>(n) * num_full_groups + g];
                const auto v_s = hn::Set(df, s);
                const auto v_nsz = hn::Set(df, -s * z);
                const uint8_t* w_ptr = w_row + g * (group_size / 2);

                int k = 0;
                for (; vec_step > 0 && k + vec_step <= group_size; k += vec_step) {
                    if (k + vec_step * prefetch_vec_iters_ahead < group_size) {
                        const int pf_k = k + vec_step * prefetch_vec_iters_ahead;
                        ::hwy::Prefetch(w_ptr + pf_k / 2);
                        ::hwy::Prefetch(in_m + k_base + pf_k);
                    }

                    const int byte_off = k / 2;
                    auto v_bytes = hn::LoadN(du8, w_ptr + byte_off, lanes_f / 2);
                    auto v_exp = hn::BitCast(di8, hn::TableLookupBytes(v_bytes, tbl_indices));
                    auto v_lo = hn::ShiftRight<4>(hn::ShiftLeft<4>(v_exp));
                    auto v_hi = hn::ShiftRight<4>(v_exp);
                    auto v_w_i8 = hn::IfThenElse(mask_odd, v_hi, v_lo);
                    auto v_w_f = hn::ConvertTo(df, hn::PromoteTo(di32, hn::PromoteTo(di16, v_w_i8)));
                    auto v_dq = hn::MulAdd(v_s, v_w_f, v_nsz);
                    v_sum = hn::MulAdd(hn::LoadU(df, in_m + k_base + k), v_dq, v_sum);
                }

                for (; k < group_size; ++k) {
                    sum += in_m[k_base + k] * UnpackNibbleScalar(w_ptr, k, s, z);
                }
            }

            sum += hn::ReduceSum(df, v_sum);
            if (remainder > 0) {
                for (int kk = num_full_groups * group_size; kk < K; ++kk) {
                    sum += in_m[kk] * UnpackNibbleScalar(w_row, kk, s_tail, z_tail);
                }
            }

            output[static_cast<int64_t>(m) * N + n] = sum;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic dispatch wrappers (resolved at first call to best ISA)
// ============================================================================
#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(GemvInt4Impl);
HWY_EXPORT(GemmInt4BatchedImpl);
HWY_EXPORT(PrepackInt4WeightsInterleavedImpl);

void GemvInt4_Hwy(float* output, const float* input, const uint8_t* weights, const float* scales, const float* zeros,
                  int K, int N, int group_size, int n_start, int n_end) {
    HWY_DYNAMIC_DISPATCH(GemvInt4Impl)(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

void GemmInt4Batched_Hwy(float* output, const float* input, const uint8_t* weights, const float* scales,
                         const float* zeros, int M, int K, int N, int group_size, int m_start, int m_end,
                         int n_start, int n_end, size_t input_stride_bytes) {
    HWY_DYNAMIC_DISPATCH(GemmInt4BatchedImpl)(output, input, weights, scales, zeros, M, K, N, group_size, m_start,
                                              m_end, n_start, n_end, input_stride_bytes);
}

void PrepackInt4WeightsInterleaved_Hwy(const uint8_t* src, uint8_t* dst, int K, int N, int group_size, int block_size) {
    HWY_DYNAMIC_DISPATCH(PrepackInt4WeightsInterleavedImpl)(src, dst, K, N, group_size, block_size);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
