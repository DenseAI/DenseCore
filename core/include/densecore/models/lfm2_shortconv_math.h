#ifndef DENSECORE_LFM2_SHORTCONV_MATH_H
#define DENSECORE_LFM2_SHORTCONV_MATH_H

#include <cstddef>

// ============================================================================
// LFM2 / LFM2.5 double-gated short convolution mixer math.
//
// The LFM2 conv block (a.k.a. "LIV" operator) replaces attention in most layers
// with a causal, depthwise short convolution wrapped in two multiplicative
// gates:
//
//     BCx = in_proj(x_norm)            // [3*C, T]
//     B, C, x = split(BCx)             // each [C, T]   (row order: B, C, x)
//     Bx  = B * x                      // elementwise gate before the conv
//     u   = depthwise_causal_conv(Bx)  // per-channel FIR, kernel length L
//     y   = C * u                      // elementwise gate after the conv
//     out = out_proj(y)
//
// Only the conv carries state across time. For streaming decode, the previous
// (L-1) values of `Bx` per channel are kept in a conv-state ring buffer that
// this module reads and updates in place. The in_proj / out_proj matmuls and
// the residual add are handled by the graph builder; this module owns the
// gate -> conv -> gate inner kernel so it can be validated in isolation against
// a reference implementation.
// ============================================================================

namespace densecore::models {

struct LFM2ShortConvConfig {
    int channels = 0;  // conv channel count (== hidden_size)
    int kernel = 0;    // conv_L_cache (causal FIR length, e.g. 3)
};

// Number of float elements in a per-sequence conv-state ring buffer.
inline size_t LFM2ShortConvStateElements(int channels, int kernel) {
    if (channels <= 0 || kernel <= 1) return 0;
    return static_cast<size_t>(channels) * static_cast<size_t>(kernel - 1);
}

// Advance the short-conv mixer by one timestep for a single sequence.
//
// Inputs (all length `cfg.channels`, contiguous):
//   b, c, x       : the B, C and x gate vectors for the current token.
//   conv_weight   : depthwise FIR weights, layout [channel * kernel + tap].
//                   tap index (kernel-1) multiplies the current Bx sample;
//                   tap index 0 multiplies the oldest cached sample.
//   conv_state    : [channels * (kernel-1)], layout [channel*(kernel-1) + j],
//                   j = 0 (oldest) .. kernel-2 (most recent past). Updated in
//                   place: the oldest sample is dropped and Bx is appended.
//   y             : output [channels] = C * conv(Bx).
//
// This is the exact arithmetic used by the serving custom op; keeping it here
// lets golden/parity tests exercise it without the ggml graph.
void LFM2ShortConvStep(const LFM2ShortConvConfig& cfg, const float* b, const float* c, const float* x,
                       const float* conv_weight, float* conv_state, float* y);

// Convenience wrapper: run `n_tokens` consecutive timesteps for one sequence.
//
// The B/C/x vectors for token t live at b[t*col_stride], c[t*col_stride],
// x[t*col_stride]; outputs are written to y[t*y_col_stride]. Strides are in
// float elements. `conv_state` is threaded through all tokens (prefill fills it
// so the first decode step continues correctly).
void LFM2ShortConvSequence(const LFM2ShortConvConfig& cfg, const float* b, const float* c, const float* x,
                           size_t col_stride, const float* conv_weight, float* conv_state, float* y,
                           size_t y_col_stride, int n_tokens);

}  // namespace densecore::models

#endif  // DENSECORE_LFM2_SHORTCONV_MATH_H
