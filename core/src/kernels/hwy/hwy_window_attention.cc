/**
 * @file hwy_window_attention.cc
 * @brief Window Attention via Google Highway
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_window_attention.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// FastExp inline
#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Helper Functions
// ============================================================================

// Partition image features into windows
// Input: [B, H, W, C] -> Output: [B*num_windows, window_size, window_size, C]
void PartitionWindowsHwy(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size,
                         float* output) {
    if (window_size <= 0) return;
    const int64_t num_windows_h = H / window_size;
    const int64_t num_windows_w = W / window_size;
    const int64_t num_windows = num_windows_h * num_windows_w;
    const int64_t window_area = window_size * window_size;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t wh = 0; wh < num_windows_h; ++wh) {
            for (int64_t ww = 0; ww < num_windows_w; ++ww) {
                const int64_t window_idx = b * num_windows + wh * num_windows_w + ww;
                float* window_out = output + window_idx * window_area * C;

                for (int64_t h = 0; h < window_size; ++h) {
                    const int64_t src_h = wh * window_size + h;
                    for (int64_t w = 0; w < window_size; ++w) {
                        const int64_t src_w = ww * window_size + w;
                        const float* src = input + b * H * W * C + src_h * W * C + src_w * C;
                        float* dst = window_out + (h * window_size + w) * C;
                        std::memcpy(dst, src, C * sizeof(float));
                    }
                }
            }
        }
    }
}

// Reverse window partition
// Input: [B*num_windows, window_size, window_size, C] -> Output: [B, H, W, C]
void ReverseWindowsHwy(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size,
                       float* output) {
    if (window_size <= 0) return;
    const int64_t num_windows_h = H / window_size;
    const int64_t num_windows_w = W / window_size;
    const int64_t num_windows = num_windows_h * num_windows_w;
    const int64_t window_area = window_size * window_size;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t wh = 0; wh < num_windows_h; ++wh) {
            for (int64_t ww = 0; ww < num_windows_w; ++ww) {
                const int64_t window_idx = b * num_windows + wh * num_windows_w + ww;
                const float* window_in = input + window_idx * window_area * C;

                for (int64_t h = 0; h < window_size; ++h) {
                    const int64_t dst_h = wh * window_size + h;
                    for (int64_t w = 0; w < window_size; ++w) {
                        const int64_t dst_w = ww * window_size + w;
                        const float* src = window_in + (h * window_size + w) * C;
                        float* dst = output + b * H * W * C + dst_h * W * C + dst_w * C;
                        std::memcpy(dst, src, C * sizeof(float));
                    }
                }
            }
        }
    }
}

// Cyclic shift
void CyclicShiftHwy(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t shift_h, int64_t shift_w,
                    float* output) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t src_h = (h + (shift_h % H) + H) % H;
                int64_t src_w = (w + (shift_w % W) + W) % W;

                const float* src = input + b * H * W * C + src_h * W * C + src_w * C;
                float* dst = output + b * H * W * C + h * W * C + w * C;
                std::memcpy(dst, src, C * sizeof(float));
            }
        }
    }
}

// ============================================================================
// Window Attention Implementation
// ============================================================================

void WindowAttentionImpl(const float* HWY_RESTRICT query, const float* HWY_RESTRICT key,
                         const float* HWY_RESTRICT value, const float* HWY_RESTRICT bias, float* HWY_RESTRICT output,
                         void* workspace, size_t workspace_size, int64_t B, int64_t H, int64_t W, int64_t C,
                         int window_size, int shift_size, int num_heads, float scale) {
    if (num_heads <= 0 || window_size <= 0) return;
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(C / num_heads));
    }

    const int64_t head_dim = C / num_heads;
    const int64_t num_windows_h = H / window_size;
    const int64_t num_windows_w = W / window_size;
    const int64_t num_windows = num_windows_h * num_windows_w;
    const int64_t window_area = window_size * window_size;
    const int64_t total_windows = B * num_windows;
    const int64_t img_size = B * H * W * C;
    const int64_t win_tensor_size = total_windows * window_area * C;
    const int64_t score_size = window_area * window_area;

    // Calculate required workspace size
    // 1. Shifted Q/K/V (Optional, size = 3 * img_size)
    // 2. Window Q/K/V/Out (Required, size = 4 * win_tensor_size)
    // 3. Attn Scores (Required, size = score_size)

    // Using floats, so multiply counts by sizeof(float)
    size_t required_bytes = 0;
    if (shift_size > 0) {
        required_bytes += 3 * img_size * sizeof(float);
        // +1 for unshifted intermediate in reverse
        required_bytes += img_size * sizeof(float);
    }
    required_bytes += 4 * win_tensor_size * sizeof(float);
    required_bytes += score_size * sizeof(float);

    if (workspace_size < required_bytes) {
        fprintf(stderr, "[WindowAttention] workspace too small: need %zu B, got %zu B\n",
                required_bytes, workspace_size);
        return;
    }

    float* ws_ptr = reinterpret_cast<float*>(workspace);

    // Assign pointers
    float* shifted_q_ptr = nullptr;
    float* shifted_k_ptr = nullptr;
    float* shifted_v_ptr = nullptr;
    float* unshifted_ptr = nullptr;

    const float* q_data = query;
    const float* k_data = key;
    const float* v_data = value;
    float* out_data = output;

    if (shift_size > 0) {
        shifted_q_ptr = ws_ptr;
        ws_ptr += img_size;
        shifted_k_ptr = ws_ptr;
        ws_ptr += img_size;
        shifted_v_ptr = ws_ptr;
        ws_ptr += img_size;
        unshifted_ptr = ws_ptr;
        ws_ptr += img_size;

        CyclicShiftHwy(query, B, H, W, C, -shift_size, -shift_size, shifted_q_ptr);
        CyclicShiftHwy(key, B, H, W, C, -shift_size, -shift_size, shifted_k_ptr);
        CyclicShiftHwy(value, B, H, W, C, -shift_size, -shift_size, shifted_v_ptr);
        q_data = shifted_q_ptr;
        k_data = shifted_k_ptr;
        v_data = shifted_v_ptr;
    }

    float* window_q_ptr = ws_ptr;
    ws_ptr += win_tensor_size;
    float* window_k_ptr = ws_ptr;
    ws_ptr += win_tensor_size;
    float* window_v_ptr = ws_ptr;
    ws_ptr += win_tensor_size;
    float* window_out_ptr = ws_ptr;
    ws_ptr += win_tensor_size;
    float* attn_scores_ptr = ws_ptr;
    ws_ptr += score_size;

    // 2. Partition
    PartitionWindowsHwy(q_data, B, H, W, C, window_size, window_q_ptr);
    PartitionWindowsHwy(k_data, B, H, W, C, window_size, window_k_ptr);
    PartitionWindowsHwy(v_data, B, H, W, C, window_size, window_v_ptr);

    // 3. Attention
    for (int64_t win = 0; win < total_windows; ++win) {
        const float* wq = window_q_ptr + win * window_area * C;
        const float* wk = window_k_ptr + win * window_area * C;
        const float* wv = window_v_ptr + win * window_area * C;
        float* wout = window_out_ptr + win * window_area * C;

        std::memset(wout, 0, window_area * C * sizeof(float));

        for (int64_t h = 0; h < num_heads; ++h) {
            const int64_t head_offset = h * head_dim;

            // Q @ K^T
            for (int64_t i = 0; i < window_area; ++i) {
                for (int64_t j = 0; j < window_area; ++j) {
                    float dot = 0.0f;
                    const float* qi = wq + i * C + head_offset;
                    const float* kj = wk + j * C + head_offset;

                    auto v_sum = hn::Zero(d);
                    size_t k = 0;
                    for (; k + N <= static_cast<size_t>(head_dim); k += N) {
                        v_sum = hn::MulAdd(hn::LoadU(d, qi + k), hn::LoadU(d, kj + k), v_sum);
                    }
                    dot = hn::ReduceSum(d, v_sum);
                    for (; k < static_cast<size_t>(head_dim); ++k) {
                        dot += qi[k] * kj[k];
                    }
                    attn_scores_ptr[i * window_area + j] = dot * scale;
                }
            }

            // Bias
            if (bias) {
                const float* head_bias = bias + h * window_area * window_area;
                for (int64_t i = 0; i < window_area * window_area; ++i) {
                    attn_scores_ptr[i] += head_bias[i];
                }
            }

            // Softmax
            for (int64_t i = 0; i < window_area; ++i) {
                float* row = attn_scores_ptr + i * window_area;

                // Max
                auto v_max = hn::Set(d, -1e30f);
                float scalar_max = -1e30f;
                size_t j = 0;
                for (; j + N <= static_cast<size_t>(window_area); j += N) {
                    v_max = hn::Max(v_max, hn::LoadU(d, row + j));
                }
                float max_val = hn::ReduceMax(d, v_max);
                for (; j < static_cast<size_t>(window_area); ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
                if (max_val < scalar_max) max_val = scalar_max;

                // Exp + Sum
                auto v_sum_exp = hn::Zero(d);
                const auto v_max_b = hn::Set(d, max_val);

                j = 0;
                for (; j + N <= static_cast<size_t>(window_area); j += N) {
                    auto v = hn::LoadU(d, row + j);
                    auto v_exp = FastExpHwy(d, hn::Sub(v, v_max_b));
                    hn::StoreU(v_exp, d, row + j);
                    v_sum_exp = hn::Add(v_sum_exp, v_exp);
                }
                float sum_exp = hn::ReduceSum(d, v_sum_exp);
                for (; j < static_cast<size_t>(window_area); ++j) {
                    float val = FastExpScalar(row[j] - max_val);  // Use FastExpScalar
                    row[j] = val;
                    sum_exp += val;
                }

                // Normalize
                float inv_sum = 1.0f / (sum_exp + 1e-9f);
                const auto v_inv = hn::Set(d, inv_sum);
                j = 0;
                for (; j + N <= static_cast<size_t>(window_area); j += N) {
                    hn::StoreU(hn::Mul(hn::LoadU(d, row + j), v_inv), d, row + j);
                }
                for (; j < static_cast<size_t>(window_area); ++j) {
                    row[j] *= inv_sum;
                }
            }

            // Attn @ V
            for (int64_t i = 0; i < window_area; ++i) {
                float* out_i = wout + i * C + head_offset;
                for (int64_t j = 0; j < window_area; ++j) {
                    float func_attn = attn_scores_ptr[i * window_area + j];
                    const float* vj = wv + j * C + head_offset;
                    const auto v_attn = hn::Set(d, func_attn);

                    size_t k = 0;
                    for (; k + N <= static_cast<size_t>(head_dim); k += N) {
                        auto v_out = hn::LoadU(d, out_i + k);
                        auto v_v = hn::LoadU(d, vj + k);
                        hn::StoreU(hn::MulAdd(v_attn, v_v, v_out), d, out_i + k);
                    }
                    for (; k < static_cast<size_t>(head_dim); ++k) {
                        out_i[k] += func_attn * vj[k];
                    }
                }
            }
        }
    }

    // 4. Reverse
    if (shift_size > 0) {
        ReverseWindowsHwy(window_out_ptr, B, H, W, C, window_size, unshifted_ptr);
        CyclicShiftHwy(unshifted_ptr, B, H, W, C, shift_size, shift_size, out_data);
    } else {
        ReverseWindowsHwy(window_out_ptr, B, H, W, C, window_size, out_data);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(WindowAttentionImpl);

void WindowAttention_Hwy(const float* query, const float* key, const float* value, const float* bias, float* output,
                         void* workspace, size_t workspace_size, int64_t B, int64_t H, int64_t W, int64_t C,
                         int window_size, int shift_size, int num_heads, float scale) {
    HWY_DYNAMIC_DISPATCH(WindowAttentionImpl)(query, key, value, bias, output, workspace, workspace_size, B, H, W, C,
                                              window_size, shift_size, num_heads, scale);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
