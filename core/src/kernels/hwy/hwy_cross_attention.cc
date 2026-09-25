/**
 * @file hwy_cross_attention.cc
 * @brief Cross-Attention via Google Highway
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_cross_attention.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include <hwy/cache_control.h>

#include "kernels/hwy/hwy_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Vectorized Dot Product: sum(a * b)
template <class D> float DotProductHwy(D d, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, int64_t size) {
    auto sum_v = hn::Zero(d);
    size_t i = 0;
    size_t lanes = hn::Lanes(d);
    for (; i + lanes <= static_cast<size_t>(size); i += lanes) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        sum_v = hn::MulAdd(va, vb, sum_v);
    }
    float sum = hn::ReduceSum(d, sum_v);
    for (; i < static_cast<size_t>(size); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vectorized Scaled Add: dst += scale * src
template <class D>
void ScaledAddHwy(D d, float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, float scale, int64_t size) {
    auto v_scale = hn::Set(d, scale);
    size_t i = 0;
    size_t lanes = hn::Lanes(d);
    for (; i + lanes <= static_cast<size_t>(size); i += lanes) {
        auto v_dst = hn::LoadU(d, dst + i);
        auto v_src = hn::LoadU(d, src + i);
        auto res = hn::MulAdd(v_scale, v_src, v_dst);
        hn::StoreU(res, d, dst + i);
    }
    for (; i < static_cast<size_t>(size); ++i) {
        dst[i] += scale * src[i];
    }
}

void CrossAttentionImpl(const float* HWY_RESTRICT q_data, const float* HWY_RESTRICT k_data,
                        const float* HWY_RESTRICT v_data, float* HWY_RESTRICT o_data, void* workspace,
                        size_t workspace_size, int64_t batch, int64_t seq_q, int64_t n_head, int64_t head_dim,
                        int64_t seq_k, int64_t n_kv_head, float scale) {
    const hn::ScalableTag<float> d;

    // Strides
    const int64_t q_stride_b = seq_q * n_head * head_dim;
    const int64_t q_stride_s = n_head * head_dim;
    const int64_t q_stride_h = head_dim;

    const int64_t k_stride_b = seq_k * n_kv_head * head_dim;
    const int64_t k_stride_s = n_kv_head * head_dim;
    const int64_t k_stride_h = head_dim;

    const int64_t o_stride_b = seq_q * n_head * head_dim;
    const int64_t o_stride_s = n_head * head_dim;
    const int64_t o_stride_h = head_dim;

    // Workspace Requirements:
    // attn_scores: seq_k
    size_t required_bytes = seq_k * sizeof(float);
    if (workspace_size < required_bytes) {
        fprintf(stderr, "[CrossAttention] workspace too small: need %zu B, got %zu B\n",
                required_bytes, workspace_size);
        return;
    }

    float* ws_ptr = reinterpret_cast<float*>(workspace);
    float* attn_scores = ws_ptr;
    // ws_ptr += seq_k; // Only one buffer needed, so increment optimization optional but good practice

    // GQA Group Size
    const int64_t group_size = n_head / n_kv_head;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            const int64_t h_kv = h / group_size;

            for (int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
                const float* q_ptr = q_data + b * q_stride_b + q_pos * q_stride_s + h * q_stride_h;

                // 1. Scores Q @ K^T
                float max_score = -std::numeric_limits<float>::infinity();  // -inf

                for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                    // Prefetch next K
                    if (k_pos + 2 < seq_k) {
                        const float* next_k = k_data + b * k_stride_b + (k_pos + 2) * k_stride_s + h_kv * k_stride_h;
                        hwy::Prefetch(next_k);
                    }

                    const float* k_ptr = k_data + b * k_stride_b + k_pos * k_stride_s + h_kv * k_stride_h;
                    float dot = DotProductHwy(d, q_ptr, k_ptr, head_dim);
                    dot *= scale;
                    attn_scores[k_pos] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // 2. Softmax
                float sum_exp = 0.0f;
                // Scalar loop for exp (sufficient for this granularity)
                for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                    float val = std::exp(attn_scores[k_pos] - max_score);
                    attn_scores[k_pos] = val;
                    sum_exp += val;
                }

                float inv_sum = 1.0f / (sum_exp + 1e-9f);

                // 3. Values
                float* out_ptr = o_data + b * o_stride_b + q_pos * o_stride_s + h * o_stride_h;
                std::memset(out_ptr, 0, head_dim * sizeof(float));

                for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                    // Prefetch next V
                    if (k_pos + 2 < seq_k) {
                        const float* next_v = v_data + b * k_stride_b + (k_pos + 2) * k_stride_s + h_kv * k_stride_h;
                        ::hwy::Prefetch(next_v);
                    }

                    float w = attn_scores[k_pos] * inv_sum;
                    const float* v_ptr = v_data + b * k_stride_b + k_pos * k_stride_s + h_kv * k_stride_h;
                    ScaledAddHwy(d, out_ptr, v_ptr, w, head_dim);
                }
            }
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

HWY_EXPORT(CrossAttentionImpl);

void CrossAttention_Hwy(const float* q_data, const float* k_data, const float* v_data, float* o_data, void* workspace,
                        size_t workspace_size, int64_t batch, int64_t seq_q, int64_t n_head, int64_t head_dim,
                        int64_t seq_k, int64_t n_kv_head, float scale) {
    HWY_DYNAMIC_DISPATCH(CrossAttentionImpl)(q_data, k_data, v_data, o_data, workspace, workspace_size, batch, seq_q,
                                             n_head, head_dim, seq_k, n_kv_head, scale);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
