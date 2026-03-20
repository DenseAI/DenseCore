/**
 * @file point_cloud_sampling.cpp
 * @brief Point Cloud Sampling Kernels (FPS, KNN, BallQuery)
 *
 * Core sampling and query operations for 3D point cloud processing.
 * Used by PointNet++, 3D object detection, and point cloud diffusion models.
 *
 * ## Algorithms
 * - FarthestPointSampling: O(N*S) greedy downsampling
 * - KNNQuery: O(N*log(k)) per query with partial sort
 * - BallQuery: O(N) radius search per query
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace densecore {
namespace {

// ============================================================================
// CpuFarthestPointSamplingOp
// ============================================================================

/**
 * @brief Farthest Point Sampling (FPS) for point cloud downsampling
 *
 * Iteratively selects points that are farthest from the already selected set.
 * Produces a representative subset with good coverage.
 *
 * ## Complexity
 * - Time: O(N * S) where N = num_points, S = num_samples
 * - Space: O(N) for distance buffer
 */
class CpuFarthestPointSamplingOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const auto* p = static_cast<const FarthestPointSamplingParams*>(params);
        FarthestPointSamplingParams default_params;
        if (!p) p = &default_params;

        const Tensor& points = *inputs[0];  // [B, N, 3]
        Tensor& indices = *outputs[0];      // [B, S]

        if (!points.IsValid() || points.ndim < 2) return;

        const int64_t batch = (points.ndim == 3) ? points.shape[0] : 1;
        const int64_t num_points = (points.ndim == 3) ? points.shape[1] : points.shape[0];
        const int num_samples = p->num_samples;

        if (num_samples <= 0 || num_samples > num_points) return;

        const float* pts_data = points.DataAs<float>();
        int32_t* idx_data = indices.DataAs<int32_t>();

        FarthestPointSampling(pts_data, idx_data, batch, num_points, num_samples);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 2 * 1024 * 1024,
                .memory_bandwidth_gbps = 50,
                .priority = 10};
    }

private:
    void FarthestPointSampling(const float* points, int32_t* indices, int64_t batch, int64_t num_points,
                               int num_samples) {
        std::vector<float> distances(num_points);

        for (int64_t b = 0; b < batch; ++b) {
            const float* pts = points + b * num_points * 3;
            int32_t* out_idx = indices + b * num_samples;

            // Initialize distances to infinity
            std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::max());

            // Start with first point
            out_idx[0] = 0;

            for (int s = 1; s < num_samples; ++s) {
                int32_t prev = out_idx[s - 1];
                const float* prev_pt = pts + prev * 3;

                // Update distances and find farthest
                float max_dist = -1.0f;
                int32_t farthest = 0;

                for (int64_t i = 0; i < num_points; ++i) {
                    const float* pt = pts + i * 3;
                    float dx = pt[0] - prev_pt[0];
                    float dy = pt[1] - prev_pt[1];
                    float dz = pt[2] - prev_pt[2];
                    float d = dx * dx + dy * dy + dz * dz;

                    distances[i] = std::min(distances[i], d);

                    if (distances[i] > max_dist) {
                        max_dist = distances[i];
                        farthest = static_cast<int32_t>(i);
                    }
                }

                out_idx[s] = farthest;
            }
        }
    }
};

// ============================================================================
// CpuKNNQueryOp
// ============================================================================

/**
 * @brief K-Nearest Neighbors Query for point clouds
 *
 * For each query point, finds the k closest reference points.
 *
 * ## Complexity
 * - Time: O(Q * N * log(k)) using partial_sort
 * - Space: O(N) for distance buffer
 */
class CpuKNNQueryOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.size() < 2) return;

        const auto* p = static_cast<const KNNQueryParams*>(params);
        KNNQueryParams default_params;
        if (!p) p = &default_params;

        const Tensor& query = *inputs[0];      // [B, Q, 3]
        const Tensor& reference = *inputs[1];  // [B, N, 3]
        Tensor& out_indices = *outputs[0];     // [B, Q, k]
        Tensor& out_distances = *outputs[1];   // [B, Q, k]

        if (!query.IsValid() || !reference.IsValid()) return;

        const int64_t batch = (query.ndim == 3) ? query.shape[0] : 1;
        const int64_t num_queries = (query.ndim == 3) ? query.shape[1] : query.shape[0];
        const int64_t num_refs = (reference.ndim == 3) ? reference.shape[1] : reference.shape[0];
        const int k = p->k;

        if (k <= 0 || k > num_refs) return;

        KNNQuery(query.DataAs<float>(), reference.DataAs<float>(), out_indices.DataAs<int32_t>(),
                 out_distances.DataAs<float>(), batch, num_queries, num_refs, k);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 50,
                .priority = 10};
    }

private:
    void KNNQuery(const float* query_points, const float* reference_points, int32_t* indices, float* distances_out,
                  int64_t batch, int64_t num_queries, int64_t num_refs, int k) {
        std::vector<std::pair<float, int32_t>> dists(num_refs);

        for (int64_t b = 0; b < batch; ++b) {
            const float* queries = query_points + b * num_queries * 3;
            const float* refs = reference_points + b * num_refs * 3;
            int32_t* out_idx = indices + b * num_queries * k;
            float* out_dist = distances_out + b * num_queries * k;

            for (int64_t q = 0; q < num_queries; ++q) {
                const float* qpt = queries + q * 3;

                // Compute all distances
                for (int64_t r = 0; r < num_refs; ++r) {
                    const float* rpt = refs + r * 3;
                    float dx = qpt[0] - rpt[0];
                    float dy = qpt[1] - rpt[1];
                    float dz = qpt[2] - rpt[2];
                    dists[r] = {dx * dx + dy * dy + dz * dz, static_cast<int32_t>(r)};
                }

                // Partial sort to get k smallest
                std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

                for (int i = 0; i < k; ++i) {
                    out_idx[q * k + i] = dists[i].second;
                    out_dist[q * k + i] = std::sqrt(dists[i].first);
                }
            }
        }
    }
};

// ============================================================================
// CpuBallQueryOp
// ============================================================================

/**
 * @brief Ball Query (Radius Search) for point clouds
 *
 * For each query point, finds all reference points within a given radius.
 *
 * ## Complexity
 * - Time: O(Q * N)
 * - Space: O(1) extra
 */
class CpuBallQueryOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const auto* p = static_cast<const BallQueryParams*>(params);
        BallQueryParams default_params;
        if (!p) p = &default_params;

        const Tensor& query = *inputs[0];      // [B, Q, 3]
        const Tensor& reference = *inputs[1];  // [B, N, 3]
        Tensor& out_indices = *outputs[0];     // [B, Q, max_samples]

        if (!query.IsValid() || !reference.IsValid()) return;

        const int64_t batch = (query.ndim == 3) ? query.shape[0] : 1;
        const int64_t num_queries = (query.ndim == 3) ? query.shape[1] : query.shape[0];
        const int64_t num_refs = (reference.ndim == 3) ? reference.shape[1] : reference.shape[0];
        const int max_samples = p->max_samples;

        // Validation: max_samples must be positive to prevent out-of-bounds pointer arithmetic
        // If max_samples < 0, "out_idx + q * max_samples" could point to invalid memory
        if (p->radius <= 0.0f || max_samples <= 0) {
            // TODO(TechnicalDebt): Use explicit error logging/throwing mechanism when available
            return;
        }

        BallQuery(query.DataAs<float>(), reference.DataAs<float>(), out_indices.DataAs<int32_t>(), batch, num_queries,
                  num_refs, p->radius, max_samples);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 2 * 1024 * 1024,
                .memory_bandwidth_gbps = 50,
                .priority = 10};
    }

private:
    void BallQuery(const float* query_points, const float* reference_points, int32_t* indices, int64_t batch,
                   int64_t num_queries, int64_t num_refs, float radius, int max_samples) {
        float radius_sq = radius * radius;

        for (int64_t b = 0; b < batch; ++b) {
            const float* queries = query_points + b * num_queries * 3;
            const float* refs = reference_points + b * num_refs * 3;
            int32_t* out_idx = indices + b * num_queries * max_samples;

            for (int64_t q = 0; q < num_queries; ++q) {
                const float* qpt = queries + q * 3;
                int32_t* out = out_idx + q * max_samples;
                int count = 0;

                for (int64_t r = 0; r < num_refs && count < max_samples; ++r) {
                    const float* rpt = refs + r * 3;
                    float dx = qpt[0] - rpt[0];
                    float dy = qpt[1] - rpt[1];
                    float dz = qpt[2] - rpt[2];
                    float dist_sq = dx * dx + dy * dy + dz * dz;

                    if (dist_sq <= radius_sq) {
                        out[count++] = static_cast<int32_t>(r);
                    }
                }

                // Fill remaining with -1 (invalid)
                for (; count < max_samples; ++count) {
                    out[count] = -1;
                }
            }
        }
    }
};

// Register operations
DENSECORE_REGISTER_OP(CpuFarthestPointSamplingOp, OpType::FarthestPointSampling, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuKNNQueryOp, OpType::KNNQuery, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuBallQueryOp, OpType::BallQuery, DeviceType::CPU);

}  // namespace
}  // namespace densecore
