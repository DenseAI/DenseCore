#ifndef DENSECORE_LLM_SSM_STATE_TYPES_H
#define DENSECORE_LLM_SSM_STATE_TYPES_H

#include <algorithm>
#include <cstddef>
#include <vector>

#include "densecore/models/qwen35_ssm_math.h"

namespace densecore::llm::ssm {

// Model-bound, dequantized layer parameters. No per-sequence recurrence lives here.
struct SSMLayerRuntimeState {
    std::vector<float> conv1d_f32;   // canonical [conv_channels][kernel]
    std::vector<float> alpha_f32;    // canonical [n_heads][n_embd]
    std::vector<float> beta_f32;     // canonical [n_heads][n_embd]
    std::vector<float> dt_bias_f32;  // canonical [n_heads]
    std::vector<float> a_log_f32;    // canonical GGUF A_log values [n_heads], not materialized A
    std::vector<float> norm_f32;     // canonical shared [head_dim_v] or flattened [d_inner]
    Qwen35SSMNormLayout norm_layout = Qwen35SSMNormLayout::INVALID;
    void Init(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) {
        (void)conv_channels;
        (void)kernel_size;
        (void)n_heads;
        (void)head_dim;
        (void)d_state;
    }
};

// Request-owned recurrent state; snapshots copy this existing value type.
struct SSMSequenceRuntimeState {
    std::vector<float> conv_state;  // [conv_channels * (kernel_size - 1)]
    std::vector<float> ssm_state;   // canonical [n_heads][head_dim_k][head_dim_v], flattened as [K,V]
    static size_t ExpectedConvStateElements(int conv_channels, int kernel_size) {
        return static_cast<size_t>(conv_channels) * static_cast<size_t>(std::max(0, kernel_size - 1));
    }
    static size_t ExpectedStateElements(int n_heads, int head_dim_k, int head_dim_v) {
        return static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim_k) * static_cast<size_t>(head_dim_v);
    }
    void Init(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) {
        conv_state.assign(ExpectedConvStateElements(conv_channels, kernel_size), 0.0f);
        ssm_state.assign(ExpectedStateElements(n_heads, head_dim, d_state), 0.0f);
    }
    bool MatchesShape(int conv_channels, int kernel_size, int n_heads, int head_dim, int d_state) const {
        const size_t expected_conv = ExpectedConvStateElements(conv_channels, kernel_size);
        const size_t expected_ssm = ExpectedStateElements(n_heads, head_dim, d_state);
        return conv_state.size() == expected_conv && ssm_state.size() == expected_ssm;
    }
    void Reset() {
        std::fill(conv_state.begin(), conv_state.end(), 0.0f);
        std::fill(ssm_state.begin(), ssm_state.end(), 0.0f);
    }
};

}  // namespace densecore::llm::ssm

#endif
