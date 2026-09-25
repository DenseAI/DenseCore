#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "densecore/models/lfm2_shortconv_math.h"

using densecore::models::LFM2ShortConvConfig;
using densecore::models::LFM2ShortConvSequence;
using densecore::models::LFM2ShortConvStateElements;
using densecore::models::LFM2ShortConvStep;

namespace {

// Independent reference for the double-gated short conv, written from the math
// definition (not the production code) so the two can disagree.
//
// inputs are full sequences laid out [channel][token]; conv_weight is
// [channel][tap] with tap (kernel-1) == current sample.
std::vector<float> ReferenceShortConv(const std::vector<float>& b, const std::vector<float>& c,
                                      const std::vector<float>& x, const std::vector<float>& conv_weight, int channels,
                                      int kernel, int n_tokens) {
    std::vector<float> y(static_cast<size_t>(channels) * n_tokens, 0.0f);
    for (int ch = 0; ch < channels; ++ch) {
        for (int t = 0; t < n_tokens; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < kernel; ++k) {
                const int src_t = t - (kernel - 1) + k;  // causal alignment
                if (src_t < 0) continue;
                const size_t idx = static_cast<size_t>(ch) * n_tokens + src_t;
                const float bx = b[idx] * x[idx];
                acc += conv_weight[static_cast<size_t>(ch) * kernel + k] * bx;
            }
            const size_t out_idx = static_cast<size_t>(ch) * n_tokens + t;
            y[out_idx] = c[out_idx] * acc;
        }
    }
    return y;
}

// Pack per-channel sequences into the column-major [3*C, T] BCx layout the
// production wrapper expects (B rows, then C rows, then x rows).
struct PackedBCx {
    std::vector<float> data;
    size_t col_stride = 0;
};

PackedBCx PackBCx(const std::vector<float>& b, const std::vector<float>& c, const std::vector<float>& x, int channels,
                  int n_tokens) {
    PackedBCx out;
    out.col_stride = static_cast<size_t>(3) * channels;
    out.data.assign(out.col_stride * n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int ch = 0; ch < channels; ++ch) {
            const size_t src = static_cast<size_t>(ch) * n_tokens + t;
            float* col = out.data.data() + static_cast<size_t>(t) * out.col_stride;
            col[ch] = b[src];
            col[channels + ch] = c[src];
            col[2 * channels + ch] = x[src];
        }
    }
    return out;
}

}  // namespace

TEST(LFM2ShortConvMath, MatchesReferencePrefill) {
    const int channels = 12;
    const int kernel = 3;
    const int n_tokens = 7;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> b(channels * n_tokens), c(channels * n_tokens), x(channels * n_tokens);
    std::vector<float> w(channels * kernel);
    for (auto& v : b) v = dist(rng);
    for (auto& v : c) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);

    const std::vector<float> ref = ReferenceShortConv(b, c, x, w, channels, kernel, n_tokens);

    PackedBCx packed = PackBCx(b, c, x, channels, n_tokens);
    const float* bptr = packed.data.data();
    const float* cptr = packed.data.data() + channels;
    const float* xptr = packed.data.data() + 2 * channels;

    std::vector<float> conv_state(LFM2ShortConvStateElements(channels, kernel), 0.0f);
    std::vector<float> y(static_cast<size_t>(channels) * n_tokens, 0.0f);

    LFM2ShortConvConfig cfg{channels, kernel};
    LFM2ShortConvSequence(cfg, bptr, cptr, xptr, packed.col_stride, w.data(), conv_state.data(), y.data(),
                          /*y_col_stride=*/channels, n_tokens);

    for (int ch = 0; ch < channels; ++ch) {
        for (int t = 0; t < n_tokens; ++t) {
            const float got = y[static_cast<size_t>(t) * channels + ch];
            const float expect = ref[static_cast<size_t>(ch) * n_tokens + t];
            EXPECT_NEAR(got, expect, 1e-5f) << "ch=" << ch << " t=" << t;
        }
    }
}

// Prefilling the whole sequence and then decoding token-by-token (carrying the
// conv state across the boundary) must equal one-shot prefill of the full
// sequence. This guards the prefill->decode conv-state handoff.
TEST(LFM2ShortConvMath, PrefillDecodeBoundaryContinuity) {
    const int channels = 8;
    const int kernel = 3;
    const int prefill = 5;
    const int decode = 4;
    const int total = prefill + decode;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> b(channels * total), c(channels * total), x(channels * total);
    std::vector<float> w(channels * kernel);
    for (auto& v : b) v = dist(rng);
    for (auto& v : c) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);

    const std::vector<float> ref = ReferenceShortConv(b, c, x, w, channels, kernel, total);

    PackedBCx packed = PackBCx(b, c, x, channels, total);
    const float* bptr = packed.data.data();
    const float* cptr = packed.data.data() + channels;
    const float* xptr = packed.data.data() + 2 * channels;

    std::vector<float> conv_state(LFM2ShortConvStateElements(channels, kernel), 0.0f);
    std::vector<float> y(static_cast<size_t>(channels) * total, 0.0f);
    LFM2ShortConvConfig cfg{channels, kernel};

    // Prefill the first `prefill` tokens.
    LFM2ShortConvSequence(cfg, bptr, cptr, xptr, packed.col_stride, w.data(), conv_state.data(), y.data(), channels,
                          prefill);
    // Decode the rest one token at a time, reusing the same conv state.
    for (int t = prefill; t < total; ++t) {
        LFM2ShortConvSequence(cfg, bptr + static_cast<size_t>(t) * packed.col_stride,
                              cptr + static_cast<size_t>(t) * packed.col_stride,
                              xptr + static_cast<size_t>(t) * packed.col_stride, packed.col_stride, w.data(),
                              conv_state.data(), y.data() + static_cast<size_t>(t) * channels, channels, 1);
    }

    for (int ch = 0; ch < channels; ++ch) {
        for (int t = 0; t < total; ++t) {
            const float got = y[static_cast<size_t>(t) * channels + ch];
            const float expect = ref[static_cast<size_t>(ch) * total + t];
            EXPECT_NEAR(got, expect, 1e-5f) << "ch=" << ch << " t=" << t;
        }
    }
}

// Single-channel hand-computed case to pin down tap ordering and gating.
TEST(LFM2ShortConvMath, HandComputedSingleChannel) {
    const int channels = 1;
    const int kernel = 3;
    // weights: tap0 (oldest)=0.5, tap1=-1.0, tap2 (current)=2.0
    std::vector<float> w{0.5f, -1.0f, 2.0f};
    // sequence of 3 tokens
    std::vector<float> b{1.0f, 1.0f, 1.0f};
    std::vector<float> x{1.0f, 2.0f, 3.0f};  // Bx = 1,2,3
    std::vector<float> c{1.0f, 1.0f, 1.0f};

    std::vector<float> conv_state(LFM2ShortConvStateElements(channels, kernel), 0.0f);
    std::vector<float> y(3, 0.0f);
    LFM2ShortConvConfig cfg{channels, kernel};
    PackedBCx packed = PackBCx(b, c, x, channels, 3);
    LFM2ShortConvSequence(cfg, packed.data.data(), packed.data.data() + channels, packed.data.data() + 2 * channels,
                          packed.col_stride, w.data(), conv_state.data(), y.data(), channels, 3);

    // t=0: window [0,0,1]   -> 0.5*0 + -1*0 + 2*1 = 2
    // t=1: window [0,1,2]   -> 0.5*0 + -1*1 + 2*2 = 3
    // t=2: window [1,2,3]   -> 0.5*1 + -1*2 + 2*3 = 4.5
    EXPECT_NEAR(y[0], 2.0f, 1e-6f);
    EXPECT_NEAR(y[1], 3.0f, 1e-6f);
    EXPECT_NEAR(y[2], 4.5f, 1e-6f);
}
