/**
 * @file hwy_rope.cc
 * @brief RoPE (Rotary Positional Embedding) via Google Highway
 *
 * Replaces hand-written AVX-512/AVX2/NEON/Scalar variants with a single
 * portable source. Highway auto-selects the best ISA at runtime.
 *
 * The old AVX2 kernel caused segfaults on some CPUs (e.g. i7-10870H) due to
 * _mm256_permutevar8x32_ps + prefetch patterns. Highway's DupEven/DupOdd
 * approach avoids those issues entirely.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_rope.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void ApplyRoPEImpl(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, const float* HWY_RESTRICT cos_sin,
                   const int* HWY_RESTRICT positions, int n_tokens, int head_dim, int rope_dim, int max_seq_len,
                   int ith, int nth) {
    if (nth <= 0) return;
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens) return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (int t = t_start; t < t_end; ++t) {
        const int pos = positions[t];
        if (pos < 0 || pos >= max_seq_len) {
            // Security: Prevent OOB read if positions contains malicious values.
            // Copy input to output as a safe fallback or could std::abort().
            const float* in_ptr = in + t * head_dim;
            float* out_ptr = out + t * head_dim;
            std::copy(in_ptr, in_ptr + head_dim, out_ptr);
            continue;
        }
        const float* cs_ptr = cos_sin + pos * head_dim;
        const float* in_ptr = in + t * head_dim;
        float* out_ptr = out + t * head_dim;

        int dd = 0;

        // Process pairs in SIMD-width chunks.
        // cos_sin is interleaved: [c0, s0, c1, s1, ...].
        // We need even elements (cos) duplicated and odd elements (sin) duplicated
        // to broadcast across each pair.
        if (N >= 2) {
            // Process 2*N elements per iteration (N pairs)
            for (; dd + static_cast<int>(2 * N) <= rope_dim; dd += static_cast<int>(2 * N)) {
                // Load 2*N input values and 2*N cos/sin values
                const auto x0 = hn::LoadU(d, in_ptr + dd);
                const auto x1 = hn::LoadU(d, in_ptr + dd + N);
                const auto cs0 = hn::LoadU(d, cs_ptr + dd);
                const auto cs1 = hn::LoadU(d, cs_ptr + dd + N);

                // DupEven/DupOdd to get cos and sin broadcast per pair
                const auto cos0 = hn::DupEven(cs0);
                const auto sin0 = hn::DupOdd(cs0);
                const auto cos1 = hn::DupEven(cs1);
                const auto sin1 = hn::DupOdd(cs1);

                // Swap adjacent pairs: [x1, x0, x3, x2, ...]
                const auto x0_swap = hn::Reverse2(d, x0);
                const auto x1_swap = hn::Reverse2(d, x1);

                // Sign pattern: [-1, +1, -1, +1, ...]
                const auto neg = hn::Set(d, -1.0f);
                const auto pos_one = hn::Set(d, 1.0f);
                const auto sign = hn::OddEven(pos_one, neg);

                // result = x * cos + swap(x) * sign * sin
                const auto r0 = hn::MulAdd(hn::Mul(x0_swap, sign), sin0, hn::Mul(x0, cos0));
                const auto r1 = hn::MulAdd(hn::Mul(x1_swap, sign), sin1, hn::Mul(x1, cos1));

                hn::StoreU(r0, d, out_ptr + dd);
                hn::StoreU(r1, d, out_ptr + dd + N);
            }
        }

        // Scalar tail for remaining pairs
        for (; dd + 1 < rope_dim; dd += 2) {
            float x0 = in_ptr[dd];
            float x1 = in_ptr[dd + 1];
            float cos_val = cs_ptr[dd];
            float sin_val = cs_ptr[dd + 1];

            out_ptr[dd] = x0 * cos_val - x1 * sin_val;
            out_ptr[dd + 1] = x0 * sin_val + x1 * cos_val;
        }

        // Copy dimensions beyond rope_dim unchanged
        for (int j = rope_dim; j < head_dim; ++j) {
            out_ptr[j] = in_ptr[j];
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(ApplyRoPEImpl);

void ApplyRoPE_Hwy(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens, int head_dim,
                   int rope_dim, int max_seq_len, int ith, int nth) {
    HWY_DYNAMIC_DISPATCH(ApplyRoPEImpl)(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, max_seq_len, ith,
                                        nth);
}

}  // namespace hwy_kernels

// C++ linkage trampoline called from densecore/simd/simd_ops.h inline ApplyRoPE()
namespace simd {
void ApplyRoPE_HwyDispatch(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens,
                           int head_dim, int rope_dim, int max_seq_len, int ith, int nth) {
    hwy_kernels::ApplyRoPE_Hwy(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, max_seq_len, ith, nth);
}
}  // namespace simd

}  // namespace densecore
#endif  // HWY_ONCE
