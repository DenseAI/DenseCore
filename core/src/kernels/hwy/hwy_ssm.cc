/**
 * @file hwy_ssm.cc
 * @brief Highway SIMD-optimized SSM (Mamba2/SSD) kernels for hybrid models.
 *
 * Provides the causal depthwise 1D convolution decode helper used by hybrid SSM
 * model graphs. Qwen-specific selective scan math lives in
 * core/src/models/qwen35_ssm_math.cpp, where its layout and state contract are
 * tested directly.
 *
 * All functions auto-dispatch to the best ISA (AVX-512, AVX2, NEON, SVE, …)
 * via Highway's foreach_target / HWY_DYNAMIC_DISPATCH mechanism.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_ssm.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

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

void SSMConv1DDecode_Hwy(float* conv_state, const float* input,
                         const float* weight, float* output,
                         int channels, int kernel_size) {
    HWY_DYNAMIC_DISPATCH(SSMConv1DDecodeCleanImpl)(
        conv_state, input, weight, output, channels, kernel_size);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
