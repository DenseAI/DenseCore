#include "densecore/models/lfm2_shortconv_math.h"

namespace densecore::models {

void LFM2ShortConvStep(const LFM2ShortConvConfig& cfg, const float* b, const float* c, const float* x,
                       const float* conv_weight, float* conv_state, float* y) {
    const int channels = cfg.channels;
    const int kernel = cfg.kernel;
    if (channels <= 0 || kernel <= 1 || !b || !c || !x || !conv_weight || !conv_state || !y) {
        return;
    }
    const int hist = kernel - 1;  // cached past samples per channel
    for (int ch = 0; ch < channels; ++ch) {
        const float bx = b[ch] * x[ch];  // gate before the conv
        const float* w = conv_weight + static_cast<size_t>(ch) * kernel;
        float* state = conv_state + static_cast<size_t>(ch) * hist;

        // Causal depthwise FIR over the window [oldest .. newest, current].
        // Taps 0..hist-1 hit the cached history; tap `hist` hits the current Bx.
        float acc = w[hist] * bx;
        for (int j = 0; j < hist; ++j) {
            acc += w[j] * state[j];
        }

        // Slide the ring buffer: drop the oldest sample, append the current Bx.
        for (int j = 0; j < hist - 1; ++j) {
            state[j] = state[j + 1];
        }
        state[hist - 1] = bx;

        y[ch] = c[ch] * acc;  // gate after the conv
    }
}

void LFM2ShortConvSequence(const LFM2ShortConvConfig& cfg, const float* b, const float* c, const float* x,
                           size_t col_stride, const float* conv_weight, float* conv_state, float* y,
                           size_t y_col_stride, int n_tokens) {
    for (int t = 0; t < n_tokens; ++t) {
        LFM2ShortConvStep(cfg, b + static_cast<size_t>(t) * col_stride, c + static_cast<size_t>(t) * col_stride,
                          x + static_cast<size_t>(t) * col_stride, conv_weight, conv_state,
                          y + static_cast<size_t>(t) * y_col_stride);
    }
}

}  // namespace densecore::models
