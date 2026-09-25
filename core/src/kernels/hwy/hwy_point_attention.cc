/**
 * @file hwy_point_attention.cc
 * @brief Point Attention (Point Transformer V2) via Google Highway
 *
 * Implements efficient gather-based attention for point clouds.
 * Replaces manual AVX2 intrinsics with portable SIMD (AVX-512, NEON, SVE).
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_point_attention.cc"
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

// FastExpHwy must be included inside HWY_NAMESPACE
#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Point Attention Implementation
// ============================================================================

// ============================================================================
// Point Attention Implementation
// ============================================================================

void PointAttentionImpl(const float* HWY_RESTRICT query, const float* HWY_RESTRICT key, const float* HWY_RESTRICT value,
                        const int32_t* HWY_RESTRICT knn_indices, const float* HWY_RESTRICT knn_dists,
                        float* HWY_RESTRICT output, void* workspace, size_t workspace_size, int64_t B, int64_t N,
                        int64_t D, int k, int num_heads, float scale, bool use_rel_pos) {

    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    const int64_t head_dim = D / num_heads;

    // Workspace requirement: k floats for scores
    size_t required_bytes = k * sizeof(float);
    if (workspace_size < required_bytes) {
        fprintf(stderr, "[PointAttention] workspace too small: need %zu B, got %zu B\n",
                required_bytes, workspace_size);
        return;
    }

    float* scores = reinterpret_cast<float*>(workspace);

    // Iterate over Batch and Points
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {

            const float* q_ptr = query + b * N * D + n * D;
            float* out_ptr = output + b * N * D + n * D;

            // Base offset for this point's kNN
            const int64_t knn_base = b * N * k + n * k;

            // Public API safety: reject malformed neighbor indices to avoid OOB reads.
            bool invalid_neighbor = false;
            for (int j = 0; j < k; ++j) {
                const int32_t neighbor_idx = knn_indices[knn_base + j];
                if (neighbor_idx < 0 || static_cast<int64_t>(neighbor_idx) >= N) {
                    invalid_neighbor = true;
                    break;
                }
            }
            if (invalid_neighbor) {
                std::fill(out_ptr, out_ptr + D, 0.0f);
                continue;
            }

            for (int64_t h = 0; h < num_heads; ++h) {
                const int64_t head_offset = h * head_dim;
                const float* q_head = q_ptr + head_offset;
                float* out_head = out_ptr + head_offset;

                // 1. Compute Attention Scores: q * k^T
                float max_score = -1e30f;
                // Reuse workspace for scores
                // std::vector<float> scores(k); -> Replaced by workspace

                for (int j = 0; j < k; ++j) {
                    int32_t neighbor_idx = knn_indices[knn_base + j];
                    const float* k_knn = key + b * N * D + neighbor_idx * D + head_offset;

                    auto v_sum = hn::Zero(d);
                    size_t i = 0;
                    for (; i + lanes <= static_cast<size_t>(head_dim); i += lanes) {
                        const auto v_q = hn::LoadU(d, q_head + i);
                        const auto v_k = hn::LoadU(d, k_knn + i);
                        v_sum = hn::MulAdd(v_q, v_k, v_sum);
                    }
                    float dot = hn::ReduceSum(d, v_sum);
                    for (; i < static_cast<size_t>(head_dim); ++i) {
                        dot += q_head[i] * k_knn[i];
                    }

                    if (use_rel_pos) {
                        float dist = knn_dists[knn_base + j];
                        dot -= 0.1f * dist;
                    }

                    float score = dot * scale;
                    scores[j] = score;
                    if (score > max_score) max_score = score;
                }

                // 2. Softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < k; ++j) {
                    // Use FastExpScalar for consistency if available, but task didn't explicitly ask for PointAttention math consistency.
                    // However, it's good practice. I'll stick to std::exp as per strict instructions for this file unless I want to be extra.
                    // Task 1 said: "In hwy_window_attention.cc and hwy_triangular_attention.cc...". It didn't mention point_attention.
                    // So I will stick to std::exp here to minimize scope creep unless necessary.
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                float inv_sum = 1.0f / (sum_exp + 1e-9f);
                for (int j = 0; j < k; ++j) {
                    scores[j] *= inv_sum;
                }

                // 3. Weighted Sum: sum(score * value)
                size_t d_idx = 0;
                for (; d_idx + lanes <= static_cast<size_t>(head_dim); d_idx += lanes) {
                    auto v_out = hn::Zero(d);

                    for (int j = 0; j < k; ++j) {
                        int32_t neighbor_idx = knn_indices[knn_base + j];
                        const float* v_knn = value + b * N * D + neighbor_idx * D + head_offset;

                        const auto v_val = hn::LoadU(d, v_knn + d_idx);
                        const auto v_weight = hn::Set(d, scores[j]);
                        v_out = hn::MulAdd(v_val, v_weight, v_out);
                    }
                    hn::StoreU(v_out, d, out_head + d_idx);
                }

                // Scalar remainder for head_dim
                for (; d_idx < static_cast<size_t>(head_dim); ++d_idx) {
                    float val_acc = 0.0f;
                    for (int j = 0; j < k; ++j) {
                        int32_t neighbor_idx = knn_indices[knn_base + j];
                        const float* v_knn = value + b * N * D + neighbor_idx * D + head_offset;
                        val_acc += v_knn[d_idx] * scores[j];
                    }
                    out_head[d_idx] = val_acc;
                }
            }  // end num_heads
        }  // end N
    }  // end B
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(PointAttentionImpl);

void PointAttention_Hwy(const float* query, const float* key, const float* value, const int32_t* knn_indices,
                        const float* knn_dists, float* output, void* workspace, size_t workspace_size, int64_t B,
                        int64_t N, int64_t D, int k, int num_heads, float scale, bool use_rel_pos) {
    HWY_DYNAMIC_DISPATCH(PointAttentionImpl)
    (query, key, value, knn_indices, knn_dists, output, workspace, workspace_size, B, N, D, k, num_heads, scale,
     use_rel_pos);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
