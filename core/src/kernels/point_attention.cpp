/**
 * @file point_attention.cpp
 * @brief Point Attention for 3D Vision (Point Transformer, PointNet++)
 *
 * Sparse attention on point clouds using kNN neighborhoods.
 * Avoids O(N²) complexity by attending only to k nearest neighbors.
 *
 * ## Key Features
 * - kNN-based local attention
 * - Relative position encoding
 * - Vector attention (Point Transformer V2 style)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/kernels/kdtree.h"
#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Point Attention Parameters
// ============================================================================

struct PointAttentionParams {
    int k = 16;  // Number of neighbors
    int num_heads = 8;
    float scale = 0.0f;  // 0 = auto
    bool use_relative_pos = true;
};

// ============================================================================
// kNN Helper (KD-tree accelerated, O(N log N) build + O(Nk log N) query)
// ============================================================================

static void ComputeKNN(const float* positions, int64_t N, int k, std::vector<int32_t>& knn_indices,
                       std::vector<float>& knn_dists) {
    knn_indices.resize(N * k);
    knn_dists.resize(N * k);

    // Build KD-tree: O(N log N)
    densecore::kernels::KDTree3D tree(positions, N);

    // Query each point: O(Nk log N)
    // We query k+1 neighbors because the point itself (distance 0) will be returned.
    // We then skip the first result (self) to get the k nearest neighbors.
    std::vector<int32_t> temp_indices(k + 1);
    std::vector<float> temp_dists(k + 1);

    for (int64_t i = 0; i < N; ++i) {
        const float* query = positions + i * 3;

        // Clamp query size to available points
        // If N <= k, we can at most find N neighbors (including self)
        int query_k = std::min(static_cast<int64_t>(k + 1), N);
        tree.KNN(query, query_k, temp_indices.data(), temp_dists.data());

        // KD-tree returns sorted results (closest first).
        // Index 0 is self (dist ~ 0). We skip it to get k neighbors.
        bool self_found = false;
        int out_idx = 0;
        int32_t last_idx = (query_k > 0) ? temp_indices[0] : 0;  // Default to self/first if fails
        float last_dist = 0.0f;

        for (int j = 0; j < query_k; ++j) {
            if (out_idx >= k) break;

            // Skip self (approx match for float 0.0)
            if (!self_found && std::abs(temp_dists[j]) < 1e-6f) {
                self_found = true;
                if (j == 0) last_idx = temp_indices[j];
                continue;
            }

            knn_indices[i * k + out_idx] = temp_indices[j];
            knn_dists[i * k + out_idx] = temp_dists[j];

            last_idx = temp_indices[j];
            last_dist = temp_dists[j];
            out_idx++;
        }

        // Padding if neighbors < k (e.g. small point cloud)
        while (out_idx < k) {
            knn_indices[i * k + out_idx] = last_idx;
            knn_dists[i * k + out_idx] = last_dist;
            out_idx++;
        }
    }
}

// ============================================================================
// CpuPointAttentionOp
// ============================================================================

class CpuPointAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 4 || outputs.empty()) return;

        const PointAttentionParams* p = static_cast<const PointAttentionParams*>(params);
        PointAttentionParams default_params;
        if (!p) p = &default_params;

        PointAttention(*inputs[0], *inputs[1], *inputs[2], *inputs[3], p->k, p->num_heads, p->scale,
                       p->use_relative_pos, outputs[0]);
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
     * @brief Point-wise attention with kNN neighborhoods
     *
     * @param query      [B, N, D] - Query features
     * @param key        [B, N, D] - Key features  
     * @param value      [B, N, D] - Value features
     * @param positions  [B, N, 3] - 3D point positions
     * @param k          Number of neighbors
     * @param num_heads  Number of attention heads
     * @param scale      Attention scale (0 = auto)
     * @param use_rel_pos Use relative position encoding
     * @param output     [B, N, D] - Output features
     */
    void PointAttention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& positions, int k,
                        int num_heads, float scale, bool use_rel_pos, Tensor* output) {
        if (!query.IsValid() || !key.IsValid() || !value.IsValid() || !positions.IsValid() || !output) {
            return;
        }

        const int64_t B = query.shape[0];
        const int64_t N = query.shape[1];
        const int64_t D = query.shape[2];
        const int64_t head_dim = D / num_heads;

        if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        const float* q_data = query.DataAs<float>();
        const float* k_data = key.DataAs<float>();
        const float* v_data = value.DataAs<float>();
        const float* pos_data = positions.DataAs<float>();
        float* out_data = output->DataAs<float>();

        std::memset(out_data, 0, B * N * D * sizeof(float));

        std::vector<int32_t> knn_indices;
        std::vector<float> knn_dists;

        for (int64_t b = 0; b < B; ++b) {
            const float* pos_b = pos_data + b * N * 3;

            // Compute kNN for this batch
            ComputeKNN(pos_b, N, k, knn_indices, knn_dists);

            // Point attention (Highway)
            const float* knn_dists_ptr = knn_dists.data();
            const int32_t* knn_indices_ptr = knn_indices.data();

            // Workspace required: k floats for scores
            size_t workspace_size = k * sizeof(float);
            std::vector<uint8_t> workspace(workspace_size);

            hwy_kernels::PointAttention_Hwy(q_data, k_data, v_data, knn_indices_ptr, knn_dists_ptr, out_data,
                                            workspace.data(), workspace_size, B, N, D, k, num_heads, scale,
                                            use_rel_pos);
        }
    }
};

DENSECORE_REGISTER_OP(CpuPointAttentionOp, OpType::PointAttention, DeviceType::CPU);

}  // namespace
}  // namespace densecore
