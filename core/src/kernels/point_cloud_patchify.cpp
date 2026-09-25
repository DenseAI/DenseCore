/**
 * @file point_cloud_patchify.cpp
 * @brief Point Cloud Patchification Kernels
 *
 * Converts between point clouds and patch representations for transformer models.
 * Uses Morton code sorting for spatial locality preservation.
 *
 * ## Use Cases
 * - Point cloud diffusion (Point-E, ShapeE)
 * - 3D Transformers (Point Transformer)
 * - PointNet++ grouping alternative
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace densecore {
namespace {

// ============================================================================
// Morton Code Utilities
// ============================================================================

/**
 * @brief Expand 10-bit integer to 30 bits for Morton encoding
 */
inline uint32_t ExpandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

/**
 * @brief Compute 30-bit Morton code for 3D point
 */
inline uint32_t MortonCode3D(float x, float y, float z, float voxel_size) {
    // Normalize to [0, 1023] range for 10-bit encoding
    uint32_t ix = static_cast<uint32_t>(std::min(std::max(x / voxel_size, 0.0f), 1023.0f));
    uint32_t iy = static_cast<uint32_t>(std::min(std::max(y / voxel_size, 0.0f), 1023.0f));
    uint32_t iz = static_cast<uint32_t>(std::min(std::max(z / voxel_size, 0.0f), 1023.0f));
    return (ExpandBits(iz) << 2) | (ExpandBits(iy) << 1) | ExpandBits(ix);
}

// ============================================================================
// CpuPointCloudPatchifyOp
// ============================================================================

/**
 * @brief Patchify point cloud into tokens for transformers
 *
 * Groups points into patches using Morton code ordering, then aggregates
 * features via mean pooling within each patch.
 *
 * ## Algorithm
 * 1. Compute Morton codes for all points
 * 2. Sort points by Morton code (preserves spatial locality)
 * 3. Divide sorted points into equal-sized patches
 * 4. Mean-pool features within each patch
 *
 * ## Complexity
 * - Time: O(N log N) (sorting)
 * - Space: O(N) (Morton codes and indices)
 */
class CpuPointCloudPatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const auto* p = static_cast<const PointCloudPatchifyParams*>(params);
        PointCloudPatchifyParams default_params;
        if (!p) p = &default_params;

        const Tensor& points = *inputs[0];    // [B, N, 3]
        const Tensor& features = *inputs[1];  // [B, N, D]
        Tensor& patches = *outputs[0];        // [B, P, D']

        if (!points.IsValid() || !features.IsValid()) return;

        const int64_t batch = (points.ndim == 3) ? points.shape[0] : 1;
        const int64_t num_points = (points.ndim == 3) ? points.shape[1] : points.shape[0];
        const int64_t feat_dim = features.shape[features.ndim - 1];
        const int num_patches = p->num_patches;
        const int patch_dim = p->patch_dim;
        const float voxel_size = p->voxel_size;

        // Validation: num_patches must be positive and <= num_points
        // Validation: patch_dim must be positive to prevent bad alloc/segfault
        if (num_patches <= 0 || num_patches > num_points || patch_dim <= 0) {
            // TODO(TechnicalDebt): Use explicit error logging/throwing mechanism when available
            return;
        }

        Patchify(points.DataAs<float>(), features.DataAs<float>(), patches.DataAs<float>(), batch, num_points, feat_dim,
                 num_patches, patch_dim, voxel_size);
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
    /**
     * @brief Core implementation of point cloud patchification
     *
     * Groups sorted points into patches based on Morton codes and averages features within each patch.
     *
     * @param points Input point coordinates [B, N, 3]
     * @param features Input point features [B, N, D]
     * @param patches Output patch features [B, P, D']
     * @param batch Batch size
     * @param num_points Number of points
     * @param feat_dim Feature dimension (D)
     * @param num_patches Number of patches (P)
     * @param patch_dim Patch dimension (D')
     * @param voxel_size Voxel size for Morton code calculation
     */
    void Patchify(const float* points, const float* features, float* patches, int64_t batch, int64_t num_points,
                  int64_t feat_dim, int num_patches, int patch_dim, float voxel_size) {
        int64_t points_per_patch = num_points / num_patches;
        if (points_per_patch < 1) points_per_patch = 1;

        // Temporary storage for Morton-sorted indices
        std::vector<std::pair<uint32_t, int32_t>> morton_indices(num_points);

        for (int64_t b = 0; b < batch; ++b) {
            const float* pts = points + b * num_points * 3;
            // Use size_t for offset calculation to prevent overflow
            const float* feats = features + static_cast<size_t>(b) * num_points * feat_dim;
            float* out = patches + static_cast<size_t>(b) * num_patches * patch_dim;

            // Compute Morton codes
            for (int64_t i = 0; i < num_points; ++i) {
                uint32_t code = MortonCode3D(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], voxel_size);
                morton_indices[i] = {code, static_cast<int32_t>(i)};
            }

            // Sort by Morton code
            std::sort(morton_indices.begin(), morton_indices.end());

            // Aggregate features into patches
            // Safe multiplication: cast to size_t to prevent integer overflow
            size_t patch_size_bytes = static_cast<size_t>(num_patches) * patch_dim * sizeof(float);
            std::memset(out, 0, patch_size_bytes);

            for (int p = 0; p < num_patches; ++p) {
                // Safe offset calculation
                float* patch_out = out + static_cast<size_t>(p) * patch_dim;
                int64_t start = p * points_per_patch;
                int64_t end = std::min(start + points_per_patch, num_points);
                int64_t count = end - start;

                if (count == 0) continue;

                // Sum features
                for (int64_t i = start; i < end; ++i) {
                    int32_t pt_idx = morton_indices[i].second;
                    const float* pt_feat = feats + static_cast<size_t>(pt_idx) * feat_dim;
                    int64_t copy_dim = std::min(feat_dim, static_cast<int64_t>(patch_dim));
                    for (int64_t d = 0; d < copy_dim; ++d) {
                        patch_out[d] += pt_feat[d];
                    }
                }

                // Mean
                float inv_count = 1.0f / count;
                for (int d = 0; d < patch_dim && d < feat_dim; ++d) {
                    patch_out[d] *= inv_count;
                }
            }
        }
    }
};

// ============================================================================
// CpuPointCloudUnpatchifyOp
// ============================================================================

/**
 * @brief Unpatchify patches back to per-point features
 *
 * Broadcasts patch features to all points in each patch.
 * Inverse of patchify (approximately, due to mean pooling).
 */
class CpuPointCloudUnpatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const auto* p = static_cast<const PointCloudPatchifyParams*>(params);
        PointCloudPatchifyParams default_params;
        if (!p) p = &default_params;

        const Tensor& patches = *inputs[0];  // [B, P, D]
        const Tensor& points = *inputs[1];   // [B, N, 3] (for patch assignment)
        Tensor& features = *outputs[0];      // [B, N, D]

        if (!patches.IsValid() || !points.IsValid()) return;

        const int64_t batch = (points.ndim == 3) ? points.shape[0] : 1;
        const int64_t num_points = (points.ndim == 3) ? points.shape[1] : points.shape[0];
        const int64_t num_patches = patches.shape[patches.ndim - 2];
        const int64_t patch_dim = patches.shape[patches.ndim - 1];
        const float voxel_size = p->voxel_size;

        Unpatchify(patches.DataAs<float>(), points.DataAs<float>(), features.DataAs<float>(), batch, num_points,
                   num_patches, patch_dim, voxel_size);
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
    void Unpatchify(const float* patches, const float* points, float* features, int64_t batch, int64_t num_points,
                    int64_t num_patches, int64_t patch_dim, float voxel_size) {
        int64_t points_per_patch = num_points / num_patches;
        if (points_per_patch < 1) points_per_patch = 1;

        std::vector<std::pair<uint32_t, int32_t>> morton_indices(num_points);

        for (int64_t b = 0; b < batch; ++b) {
            const float* pts = points + b * num_points * 3;
            const float* patch_data = patches + b * num_patches * patch_dim;
            float* out = features + b * num_points * patch_dim;

            // Compute Morton codes for patch assignment
            for (int64_t i = 0; i < num_points; ++i) {
                uint32_t code = MortonCode3D(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], voxel_size);
                morton_indices[i] = {code, static_cast<int32_t>(i)};
            }

            std::sort(morton_indices.begin(), morton_indices.end());

            // Broadcast patch features to points
            for (int64_t p = 0; p < num_patches; ++p) {
                const float* patch_feat = patch_data + p * patch_dim;
                int64_t start = p * points_per_patch;
                int64_t end = std::min(start + points_per_patch, num_points);

                for (int64_t i = start; i < end; ++i) {
                    int32_t pt_idx = morton_indices[i].second;
                    float* pt_out = out + pt_idx * patch_dim;
                    std::memcpy(pt_out, patch_feat, patch_dim * sizeof(float));
                }
            }
        }
    }
};

// Register operations
DENSECORE_REGISTER_OP(CpuPointCloudPatchifyOp, OpType::PointCloudPatchify, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuPointCloudUnpatchifyOp, OpType::PointCloudUnpatchify, DeviceType::CPU);

}  // namespace
}  // namespace densecore
