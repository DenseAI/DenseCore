/**
 * @file hwy_ssm.cc
 * @brief Highway SIMD-optimized SSM (Mamba2/SSD) kernels for hybrid models.
 *
 * Provides three kernels:
 *   1. SSMConv1D  — causal depthwise 1D convolution (decode-only: single step)
 *   2. SSMScan    — selective scan state update + output (single token decode)
 *   3. SSMScanPrefill — sequential scan over a sequence (prefill path)
 *
 * All functions auto-dispatch to the best ISA (AVX-512, AVX2, NEON, SVE, …)
 * via Highway's foreach_target / HWY_DYNAMIC_DISPATCH mechanism.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_ssm.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <cmath>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Include fast exponential for softplus / exp(dt*A)
#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Helper: Softplus  f(x) = log(1 + exp(x))
// ============================================================================
static HWY_INLINE float SoftplusScalar(float x) {
    if (x > 20.0f) return x;          // Avoid overflow
    if (x < -20.0f) return 0.0f;
    return std::log1p(std::exp(x));
}

// ============================================================================
// SSMConv1DDecodeCleanImpl — single-step causal conv1d for decode
// ============================================================================

void SSMConv1DDecodeCleanImpl(
        float* HWY_RESTRICT conv_state,  // [channels * (kernel_size - 1)]
        const float* HWY_RESTRICT input, // [channels]
        const float* HWY_RESTRICT weight,// [channels * kernel_size] (ggml: ne[0]=ks, ne[1]=ch)
        float* HWY_RESTRICT output,      // [channels]
        int channels,
        int kernel_size) {

    const int hist = kernel_size - 1;

    // Qwen3.5: kernel_size=4 (conv_kernel_dim), channels ≈ 4096
    // Optimized path for kernel_size == 4 (hist == 3): fully unrolled, fused output + state update.
    // NOTE: The computation is strided-gather per channel, so SIMD writes via hn::Load/Store
    // are NOT used here — hn::Load requires alignment (e.g. 64-byte for AVX-512) but stack
    // arrays are only 16-byte aligned on x86_64, causing SIGBUS/SIGSEGV. Write directly to
    // output[c] instead and let the compiler auto-vectorize.
    if (kernel_size == 4) {
        for (int ci = 0; ci < channels; ++ci) {
            const int cs_base = ci * 3;
            const int w_base = ci * 4;
            output[ci] = conv_state[cs_base] * weight[w_base] +
                         conv_state[cs_base + 1] * weight[w_base + 1] +
                         conv_state[cs_base + 2] * weight[w_base + 2] +
                         input[ci] * weight[w_base + 3];
            conv_state[cs_base] = conv_state[cs_base + 1];
            conv_state[cs_base + 1] = conv_state[cs_base + 2];
            conv_state[cs_base + 2] = input[ci];
        }
        return;
    }

    // Generic path for other kernel sizes
    for (int ci = 0; ci < channels; ++ci) {
        float s = 0.0f;
        for (int k = 0; k < hist; ++k) {
            s += conv_state[ci * hist + k] * weight[ci * kernel_size + k];
        }
        s += input[ci] * weight[ci * kernel_size + hist];
        output[ci] = s;
    }
    for (int ci = 0; ci < channels; ++ci) {
        for (int j = 0; j < hist - 1; ++j) {
            conv_state[ci * hist + j] = conv_state[ci * hist + j + 1];
        }
        conv_state[ci * hist + hist - 1] = input[ci];
    }
}

// ============================================================================
// SSMScanDecodeImpl — single-token selective scan (Mamba2 SSD decode)
//
// For ONE token, updates the recurrent state and produces output.
//
// x:       [d_inner]             (activated SSM input for this token)
// dt:      [n_heads]             (discretized time steps, AFTER softplus)
// A_log:   [n_heads]             (log-space diagonal A, typically negative)
// B:       [n_groups * d_state]  (input-to-state projection)
// C:       [n_groups * d_state]  (state-to-output projection)
// state:   [n_heads * head_dim * d_state]  (recurrent state, updated in-place)
// output:  [d_inner]             (output for this token)
//
// n_heads, head_dim, d_state, n_groups: SSM parameters
// ============================================================================
void SSMScanDecodeImpl(
        const float* HWY_RESTRICT x,
        const float* HWY_RESTRICT dt_A,   // A discretization per head: A_bar = exp(dt_A * A_log)
        const float* HWY_RESTRICT dt_B,   // B input scaling per head:  B_eff = dt_B * B
        const float* HWY_RESTRICT A_log,
        const float* HWY_RESTRICT B,
        const float* HWY_RESTRICT C,
        float* HWY_RESTRICT state,
        float* HWY_RESTRICT output,
        int n_heads,
        int head_dim,
        int d_state,
        int n_groups) {

    const hn::ScalableTag<float> d;
    const size_t NL = hn::Lanes(d);
    const int heads_per_group = n_heads / n_groups;

    for (int h = 0; h < n_heads; ++h) {
        const int g = h / heads_per_group;

        // A discretization: A_bar = exp(dt_A[h] * A_log[h]) ∈ (0, 1)
        const float A_bar  = std::exp(dt_A[h] * A_log[h]);
        // B input scaling from dt_B
        const float b_scale = dt_B[h];

        const float* x_h = &x[h * head_dim];
        const float* B_g = &B[g * d_state];
        const float* C_g = &C[g * d_state];
        float* state_h = &state[h * head_dim * d_state];
        float* out_h = &output[h * head_dim];

        const auto v_abar   = hn::Set(d, A_bar);
        const auto v_bscale = hn::Set(d, b_scale);

        // For each feature i in head_dim:
        //   state[h][i][:] = A_bar * state[h][i][:] + dt_B[h] * x[h*hd+i] * B[g][:]
        //   out[h*hd+i]   = dot(state[h][i][:], C[g][:])
        for (int i = 0; i < head_dim; ++i) {
            float* row = &state_h[i * d_state];
            const float xi = x_h[i];
            const auto v_xi_b = hn::Set(d, xi * b_scale);  // pre-multiply xi * dt_B

            float dot_sum = 0.0f;

            size_t j = 0;
            for (; j + NL <= static_cast<size_t>(d_state); j += NL) {
                auto v_s = hn::Load(d, &row[j]);
                auto v_b = hn::Load(d, &B_g[j]);
                auto v_c = hn::Load(d, &C_g[j]);

                // state update: s = A_bar * s + dt_B * xi * B[j]
                v_s = hn::MulAdd(v_abar, v_s, hn::Mul(v_xi_b, v_b));
                hn::Store(v_s, d, &row[j]);

                dot_sum += hn::ReduceSum(d, hn::Mul(v_s, v_c));
            }
            for (; j < static_cast<size_t>(d_state); ++j) {
                row[j] = A_bar * row[j] + xi * b_scale * B_g[j];
                dot_sum += row[j] * C_g[j];
            }

            out_h[i] = dot_sum;
        }
    }
}

// ============================================================================
// SSMScanPrefillImpl — multi-token sequential scan (prefill path)
//
// Processes seq_len tokens sequentially. For each token t:
//   1. Compute dt, discretize
//   2. Update state
//   3. Produce output
//
// x:       [seq_len, d_inner]
// dt_raw:  [seq_len, n_heads]     (BEFORE softplus)
// A_log:   [n_heads]
// B:       [seq_len, n_groups * d_state]
// C:       [seq_len, n_groups * d_state]
// state:   [n_heads * head_dim * d_state]  (updated in-place at end)
// output:  [seq_len, d_inner]
// ============================================================================
void SSMScanPrefillImpl(
        const float* HWY_RESTRICT x,
        const float* HWY_RESTRICT dt_raw,
        const float* HWY_RESTRICT A_log,
        const float* HWY_RESTRICT B,
        const float* HWY_RESTRICT C,
        float* HWY_RESTRICT state,
        float* HWY_RESTRICT output,
        int seq_len,
        int n_heads,
        int head_dim,
        int d_state,
        int n_groups) {

    const int d_inner = n_heads * head_dim;
    const int bc_stride = n_groups * d_state;

    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = &x[t * d_inner];
        const float* dt_t = &dt_raw[t * n_heads];
        const float* B_t = &B[t * bc_stride];
        const float* C_t = &C[t * bc_stride];
        float* out_t = &output[t * d_inner];

        // thread_local reuse buffers — eliminates seq_len × 2 heap allocs per SSM layer
        static thread_local std::vector<float> dt_sp;
        static thread_local std::vector<float> dt_B_ones;
        if (static_cast<int>(dt_sp.size()) < n_heads) {
            dt_sp.resize(static_cast<size_t>(n_heads));
            dt_B_ones.assign(static_cast<size_t>(n_heads), 1.0f);
        }
        for (int h = 0; h < n_heads; ++h) {
            dt_sp[h] = SoftplusScalar(dt_t[h]);
            dt_B_ones[h] = 1.0f;
        }

        SSMScanDecodeImpl(x_t, dt_sp.data(), dt_B_ones.data(), A_log, B_t, C_t, state, out_t,
                          n_heads, head_dim, d_state, n_groups);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

// ============================================================================
// Public dispatch (compiled once)
// ============================================================================
#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(SSMConv1DDecodeCleanImpl);
HWY_EXPORT(SSMScanDecodeImpl);
HWY_EXPORT(SSMScanPrefillImpl);

void SSMConv1DDecode_Hwy(float* conv_state, const float* input,
                         const float* weight, float* output,
                         int channels, int kernel_size) {
    HWY_DYNAMIC_DISPATCH(SSMConv1DDecodeCleanImpl)(
        conv_state, input, weight, output, channels, kernel_size);
}

void SSMScanDecode_Hwy(const float* x, const float* dt_A, const float* dt_B,
                       const float* A_log, const float* B, const float* C,
                       float* state, float* output,
                       int n_heads, int head_dim, int d_state, int n_groups) {
    HWY_DYNAMIC_DISPATCH(SSMScanDecodeImpl)(
        x, dt_A, dt_B, A_log, B, C, state, output,
        n_heads, head_dim, d_state, n_groups);
}

void SSMScanPrefill_Hwy(const float* x, const float* dt_raw,
                        const float* A_log, const float* B, const float* C,
                        float* state, float* output,
                        int seq_len, int n_heads, int head_dim,
                        int d_state, int n_groups) {
    HWY_DYNAMIC_DISPATCH(SSMScanPrefillImpl)(
        x, dt_raw, A_log, B, C, state, output,
        seq_len, n_heads, head_dim, d_state, n_groups);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
