/**
 * @file scatter_reduce.cpp
 * @brief Scatter-Reduce for 3D Vision (Point Cloud, PointNet++, 3DGS)
 *
 * Aggregates point cloud features into voxel grid representation.
 * Supports SUM, MEAN, MAX reduction modes.
 *
 * ## Use Cases
 * - Point cloud → voxel for 3D object detection
 * - Gaussian splatting accumulation
 * - LiDAR processing for autonomous driving
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Reduction Mode
// ============================================================================

enum class ReduceMode : int { SUM = 0, MEAN = 1, MAX = 2 };

struct ScatterReduceParams {
    int mode = 0;  // ReduceMode
    int voxel_x = 100;
    int voxel_y = 100;
    int voxel_z = 10;
};

// ============================================================================
// CpuScatterReduceOp
// ============================================================================

class CpuScatterReduceOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const ScatterReduceParams* p = static_cast<const ScatterReduceParams*>(params);
        ScatterReduceParams default_params;
        if (!p) p = &default_params;

        ScatterReduce(*inputs[0], *inputs[1], static_cast<ReduceMode>(p->mode), p->voxel_x, p->voxel_y, p->voxel_z,
                      outputs[0]);
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
     * @brief Scatter-Reduce point features to voxel grid
     *
     * @param voxel_coords [N, 3] - Integer voxel indices for each point
     * @param features     [N, D] - Point features
     * @param mode         Reduction mode (SUM, MEAN, MAX)
     * @param voxel_x/y/z  Voxel grid dimensions
     * @param output       [X, Y, Z, D] - Voxel features
     */
    void ScatterReduce(const Tensor& voxel_coords, const Tensor& features, ReduceMode mode, int voxel_x, int voxel_y,
                       int voxel_z, Tensor* output) {
        if (!voxel_coords.IsValid() || !features.IsValid() || !output) return;

        const int64_t N = voxel_coords.shape[0];
        const int64_t D = features.shape[1];
        const int64_t X = voxel_x;
        const int64_t Y = voxel_y;
        const int64_t Z = voxel_z;

        const int32_t* coord_data = voxel_coords.DataAs<int32_t>();
        const float* feat_data = features.DataAs<float>();
        float* out_data = output->DataAs<float>();

        const int64_t total_voxels = X * Y * Z;

        // Initialize output and count buffer
        std::memset(out_data, 0, total_voxels * D * sizeof(float));
        std::vector<int> counts(total_voxels, 0);

        // For MAX mode, initialize to very negative value
        if (mode == ReduceMode::MAX) {
            for (int64_t i = 0; i < total_voxels * D; ++i) {
                out_data[i] = -1e30f;
            }
        }

        // Scatter points to voxels
        for (int64_t n = 0; n < N; ++n) {
            int x = coord_data[n * 3 + 0];
            int y = coord_data[n * 3 + 1];
            int z = coord_data[n * 3 + 2];

            // Bounds check
            if (x < 0 || x >= X || y < 0 || y >= Y || z < 0 || z >= Z) {
                continue;
            }

            int64_t voxel_idx = (x * Y * Z + y * Z + z);
            float* voxel_feat = out_data + voxel_idx * D;
            const float* point_feat = feat_data + n * D;

            if (mode == ReduceMode::SUM || mode == ReduceMode::MEAN) {
#if defined(__AVX2__)
                int64_t d = 0;
                for (; d + 8 <= D; d += 8) {
                    __m256 v_vox = _mm256_loadu_ps(voxel_feat + d);
                    __m256 v_pt = _mm256_loadu_ps(point_feat + d);
                    v_vox = _mm256_add_ps(v_vox, v_pt);
                    _mm256_storeu_ps(voxel_feat + d, v_vox);
                }
                for (; d < D; ++d) {
                    voxel_feat[d] += point_feat[d];
                }
#else
                for (int64_t d = 0; d < D; ++d) {
                    voxel_feat[d] += point_feat[d];
                }
#endif
                counts[voxel_idx]++;

            } else if (mode == ReduceMode::MAX) {
                for (int64_t d = 0; d < D; ++d) {
                    voxel_feat[d] = std::max(voxel_feat[d], point_feat[d]);
                }
                counts[voxel_idx] = 1;  // Mark as having data
            }
        }

        // Post-processing for MEAN
        if (mode == ReduceMode::MEAN) {
            for (int64_t v = 0; v < total_voxels; ++v) {
                if (counts[v] > 0) {
                    float inv_count = 1.0f / counts[v];
                    float* voxel_feat = out_data + v * D;
#if defined(__AVX2__)
                    __m256 v_inv = _mm256_set1_ps(inv_count);
                    int64_t d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 v_f = _mm256_loadu_ps(voxel_feat + d);
                        v_f = _mm256_mul_ps(v_f, v_inv);
                        _mm256_storeu_ps(voxel_feat + d, v_f);
                    }
                    for (; d < D; ++d) {
                        voxel_feat[d] *= inv_count;
                    }
#else
                    for (int64_t d = 0; d < D; ++d) {
                        voxel_feat[d] *= inv_count;
                    }
#endif
                }
            }
        }

        // Reset MAX voxels with no data to zero
        if (mode == ReduceMode::MAX) {
            for (int64_t v = 0; v < total_voxels; ++v) {
                if (counts[v] == 0) {
                    std::memset(out_data + v * D, 0, D * sizeof(float));
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuScatterReduceOp, OpType::ScatterReduce, DeviceType::CPU);
// Note: Using Custom until we add ScatterReduce to OpType enum

}  // namespace
}  // namespace densecore
