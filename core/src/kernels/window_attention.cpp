/**
 * @file window_attention.cpp
 * @brief CPU optimized Window Attention for Vision Transformers (Swin, Qwen-VL)
 *
 * Window Attention divides the input into non-overlapping windows and
 * performs attention within each window, reducing complexity from O(N²)
 * to O(W² × N/W²) where W is window size.
 *
 * ## Optimizations
 * - L2 cache-friendly window partitioning
 * - AVX2/AVX-512 vectorized softmax and attention
 * - Relative position bias support
 * - Cyclic shift for cross-window connections (Swin-style)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Window Attention Parameters
// ============================================================================

struct WindowAttentionParams {
    int window_size = 7;
    int shift_size = 0;  // 0 = no shift, >0 = cyclic shift
    int num_heads = 8;
    float scale = 0.0f;  // 0 = auto (1/sqrt(head_dim))
    bool use_relative_bias = true;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Partition image features into windows
// Input: [B, H, W, C] -> Output: [B*num_windows, window_size, window_size, C]
static void PartitionWindows(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size,
                             float* output) {
    const int64_t num_windows_h = H / window_size;
    const int64_t num_windows_w = W / window_size;
    const int64_t num_windows = num_windows_h * num_windows_w;
    const int64_t window_area = window_size * window_size;

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t wh = 0; wh < num_windows_h; ++wh) {
            for (int64_t ww = 0; ww < num_windows_w; ++ww) {
                const int64_t window_idx = b * num_windows + wh * num_windows_w + ww;
                float* window_out = output + window_idx * window_area * C;

                // Copy each row of the window
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
static void ReverseWindows(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size,
                           float* output) {
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

// Cyclic shift for cross-window attention (Swin-style)
static void CyclicShift(const float* input, int64_t B, int64_t H, int64_t W, int64_t C, int64_t shift_h,
                        int64_t shift_w, float* output) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                // Source position with cyclic wrap
                int64_t src_h = (h + shift_h + H) % H;
                int64_t src_w = (w + shift_w + W) % W;

                const float* src = input + b * H * W * C + src_h * W * C + src_w * C;
                float* dst = output + b * H * W * C + h * W * C + w * C;
                std::memcpy(dst, src, C * sizeof(float));
            }
        }
    }
}

// Vectorized softmax along last dimension
static void SoftmaxRow(float* row, int64_t len) {
    // Find max for numerical stability
    float max_val = row[0];
    for (int64_t i = 1; i < len; ++i) {
        if (row[i] > max_val) max_val = row[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 v_max = _mm256_set1_ps(max_val);
    __m256 v_sum = _mm256_setzero_ps();
    int64_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 v_x = _mm256_loadu_ps(row + i);
        v_x = _mm256_sub_ps(v_x, v_max);
        // Fast exp approximation using polynomial (accurate for [-87, 87])
        // exp(x) ≈ (1 + x/256)^256, but we use standard exp for accuracy
        __m256 v_exp =
            _mm256_set_ps(std::exp(((float*)&v_x)[7]), std::exp(((float*)&v_x)[6]), std::exp(((float*)&v_x)[5]),
                          std::exp(((float*)&v_x)[4]), std::exp(((float*)&v_x)[3]), std::exp(((float*)&v_x)[2]),
                          std::exp(((float*)&v_x)[1]), std::exp(((float*)&v_x)[0]));
        _mm256_storeu_ps(row + i, v_exp);
        v_sum = _mm256_add_ps(v_sum, v_exp);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(v_sum, 1);
    __m128 lo = _mm256_castps256_ps128(v_sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);

    for (; i < len; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
#else
    for (int64_t i = 0; i < len; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
#endif

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-9f);
#if defined(__AVX2__)
    __m256 v_inv = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v_x = _mm256_loadu_ps(row + i);
        v_x = _mm256_mul_ps(v_x, v_inv);
        _mm256_storeu_ps(row + i, v_x);
    }
    for (; i < len; ++i) {
        row[i] *= inv_sum;
    }
#else
    for (int64_t i = 0; i < len; ++i) {
        row[i] *= inv_sum;
    }
#endif
}

// ============================================================================
// Window Attention Implementation
// ============================================================================

class CpuWindowAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;

        const WindowAttentionParams* p = static_cast<const WindowAttentionParams*>(params);
        WindowAttentionParams default_params;
        if (!p) p = &default_params;

        WindowAttention(*inputs[0], *inputs[1], *inputs[2],
                        inputs.size() > 3 ? inputs[3] : nullptr,  // relative_bias (optional)
                        p->window_size, p->shift_size, p->num_heads, p->scale, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    /**
     * @brief Core Window Attention computation
     *
     * @param query  [B, H, W, C] - Query tensor
     * @param key    [B, H, W, C] - Key tensor 
     * @param value  [B, H, W, C] - Value tensor
     * @param relative_bias [num_heads, W², W²] - Optional relative position bias
     * @param window_size Window size (typically 7)
     * @param shift_size Cyclic shift size (0 for no shift)
     * @param num_heads Number of attention heads
     * @param scale Attention scale (0 = auto)
     * @param output [B, H, W, C] - Output tensor
     */
    void WindowAttention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* relative_bias,
                         int window_size, int shift_size, int num_heads, float scale, Tensor* output) {
        if (!query.IsValid() || !key.IsValid() || !value.IsValid() || !output) return;

        const int64_t B = query.shape[0];
        const int64_t H = query.shape[1];
        const int64_t W = query.shape[2];
        const int64_t C = query.shape[3];

        const float* q_data = query.DataAs<float>();
        const float* k_data = key.DataAs<float>();
        const float* v_data = value.DataAs<float>();
        const float* bias_data = (relative_bias && relative_bias->IsValid()) ? relative_bias->DataAs<float>() : nullptr;
        float* out_data = output->DataAs<float>();

        const int64_t window_area = window_size * window_size;
        // Workspace required: 7 buffers of size [B, H, W, C] + 1 buffer of size [window_area^2]
        // shifted_q, shifted_k, shifted_v, window_q, window_k, window_v, window_out -> 7 * total_elements
        const size_t total_elements = B * H * W * C;
        const size_t workspace_size = (7 * total_elements + window_area * window_area) * sizeof(float);
        std::vector<uint8_t> workspace(workspace_size);

        hwy_kernels::WindowAttention_Hwy(q_data, k_data, v_data, bias_data, out_data, workspace.data(), workspace_size,
                                         B, H, W, C, window_size, shift_size, num_heads, scale);
    }
};

DENSECORE_REGISTER_OP(CpuWindowAttentionOp, OpType::WindowAttention, DeviceType::CPU);

}  // namespace
}  // namespace densecore
