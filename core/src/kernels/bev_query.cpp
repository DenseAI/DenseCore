/**
 * @file bev_query.cpp
 * @brief BEV (Bird's Eye View) Query for Autonomous Driving (BEVFormer, UniAD)
 *
 * Transforms multi-camera features to Bird's Eye View representation
 * using deformable cross-attention and camera projection matrices.
 *
 * ## Key Operations
 * - Camera intrinsic/extrinsic projection
 * - 3D to 2D coordinate mapping
 * - Multi-scale feature sampling
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// BEVQuery Parameters
// ============================================================================

struct BEVQueryParams {
    int bev_h = 200;  // BEV grid height
    int bev_w = 200;  // BEV grid width
    float pc_range_x_min = -50.0f;
    float pc_range_x_max = 50.0f;
    float pc_range_y_min = -50.0f;
    float pc_range_y_max = 50.0f;
    float pc_range_z_min = -5.0f;
    float pc_range_z_max = 3.0f;
    int num_points = 4;  // Sampling points per query
};

// ============================================================================
// Matrix multiplication helper: 4x4 @ 4x1
// ============================================================================

inline void MatMul4x4_4x1(const float* M, const float* v, float* out) {
    for (int i = 0; i < 4; ++i) {
        out[i] = M[i * 4 + 0] * v[0] + M[i * 4 + 1] * v[1] + M[i * 4 + 2] * v[2] + M[i * 4 + 3] * v[3];
    }
}

// ============================================================================
// CpuBEVQueryOp
// ============================================================================

class CpuBEVQueryOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 4 || outputs.empty()) return;

        const BEVQueryParams* p = static_cast<const BEVQueryParams*>(params);
        BEVQueryParams default_params;
        if (!p) p = &default_params;

        BEVQuery(*inputs[0],  // multi_cam_features
                 *inputs[1],  // bev_queries
                 *inputs[2],  // camera_intrinsics
                 *inputs[3],  // camera_extrinsics
                 *p, outputs[0]);
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
     * @brief BEV Query operation
     *
     * @param multi_cam_features [N_cam, H, W, D] - Multi-camera features
     * @param bev_queries        [H_bev, W_bev, D] - BEV query embeddings
     * @param cam_intrinsics     [N_cam, 3, 3] - Camera intrinsic matrices
     * @param cam_extrinsics     [N_cam, 4, 4] - Camera extrinsic matrices
     * @param params             BEV configuration
     * @param output             [H_bev, W_bev, D] - BEV output features
     */
    void BEVQuery(const Tensor& multi_cam_features, const Tensor& bev_queries, const Tensor& cam_intrinsics,
                  const Tensor& cam_extrinsics, const BEVQueryParams& params, Tensor* output) {
        if (!multi_cam_features.IsValid() || !bev_queries.IsValid() || !cam_intrinsics.IsValid() ||
            !cam_extrinsics.IsValid() || !output) {
            return;
        }

        const int64_t N_cam = multi_cam_features.shape[0];
        const int64_t H_feat = multi_cam_features.shape[1];
        const int64_t W_feat = multi_cam_features.shape[2];
        const int64_t D = multi_cam_features.shape[3];

        const int64_t H_bev = params.bev_h;
        const int64_t W_bev = params.bev_w;

        const float* feat_data = multi_cam_features.DataAs<float>();
        const float* query_data = bev_queries.DataAs<float>();
        const float* K_data = cam_intrinsics.DataAs<float>();
        const float* T_data = cam_extrinsics.DataAs<float>();
        float* out_data = output->DataAs<float>();

        // Initialize output with query embeddings
        std::memcpy(out_data, query_data, H_bev * W_bev * D * sizeof(float));

        // Resolution of BEV grid
        const float x_step = (params.pc_range_x_max - params.pc_range_x_min) / W_bev;
        const float y_step = (params.pc_range_y_max - params.pc_range_y_min) / H_bev;
        const float z_center = (params.pc_range_z_min + params.pc_range_z_max) / 2.0f;

        // For each BEV grid cell
        for (int64_t bh = 0; bh < H_bev; ++bh) {
            for (int64_t bw = 0; bw < W_bev; ++bw) {
                // 3D world coordinates (center of BEV cell)
                float world_x = params.pc_range_x_min + (bw + 0.5f) * x_step;
                float world_y = params.pc_range_y_min + (bh + 0.5f) * y_step;
                float world_z = z_center;
                float world_pt[4] = {world_x, world_y, world_z, 1.0f};

                float* bev_out = out_data + (bh * W_bev + bw) * D;
                float total_weight = 0.0f;

                // Project to each camera and sample features
                for (int64_t cam = 0; cam < N_cam; ++cam) {
                    const float* T = T_data + cam * 16;  // 4x4 extrinsic
                    const float* K = K_data + cam * 9;   // 3x3 intrinsic

                    // Transform to camera coordinates
                    float cam_pt[4];
                    MatMul4x4_4x1(T, world_pt, cam_pt);

                    if (cam_pt[2] < -1e-3f) continue;  // Behind camera
                    const float proj_depth = (cam_pt[2] > 1e-3f) ? cam_pt[2] : 1.0f;

                    // Project to image coordinates
                    float img_x = (K[0] * cam_pt[0] + K[1] * cam_pt[1] + K[2] * proj_depth) / proj_depth;
                    float img_y = (K[3] * cam_pt[0] + K[4] * cam_pt[1] + K[5] * proj_depth) / proj_depth;

                    // Normalize to [0, 1] range
                    float norm_x = img_x / W_feat;
                    float norm_y = img_y / H_feat;

                    // Check if in bounds
                    if (norm_x < 0.0f || norm_x >= 1.0f || norm_y < 0.0f || norm_y >= 1.0f) {
                        continue;
                    }

                    // Bilinear sampling
                    float x = norm_x * (W_feat - 1);
                    float y = norm_y * (H_feat - 1);
                    int x0 = static_cast<int>(std::floor(x));
                    int y0 = static_cast<int>(std::floor(y));
                    int x1 = std::min(x0 + 1, static_cast<int>(W_feat - 1));
                    int y1 = std::min(y0 + 1, static_cast<int>(H_feat - 1));

                    float wx1 = x - x0;
                    float wx0 = 1.0f - wx1;
                    float wy1 = y - y0;
                    float wy0 = 1.0f - wy1;

                    const float* f00 = feat_data + cam * H_feat * W_feat * D + y0 * W_feat * D + x0 * D;
                    const float* f01 = feat_data + cam * H_feat * W_feat * D + y0 * W_feat * D + x1 * D;
                    const float* f10 = feat_data + cam * H_feat * W_feat * D + y1 * W_feat * D + x0 * D;
                    const float* f11 = feat_data + cam * H_feat * W_feat * D + y1 * W_feat * D + x1 * D;

                    float weight = 1.0f;  // Could use depth-based weighting
                    total_weight += weight;

#if defined(__AVX2__)
                    __m256 vw00 = _mm256_set1_ps(wy0 * wx0 * weight);
                    __m256 vw01 = _mm256_set1_ps(wy0 * wx1 * weight);
                    __m256 vw10 = _mm256_set1_ps(wy1 * wx0 * weight);
                    __m256 vw11 = _mm256_set1_ps(wy1 * wx1 * weight);

                    int64_t d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 v00 = _mm256_loadu_ps(f00 + d);
                        __m256 v01 = _mm256_loadu_ps(f01 + d);
                        __m256 v10 = _mm256_loadu_ps(f10 + d);
                        __m256 v11 = _mm256_loadu_ps(f11 + d);
                        __m256 vout = _mm256_loadu_ps(bev_out + d);

                        vout = _mm256_fmadd_ps(v00, vw00, vout);
                        vout = _mm256_fmadd_ps(v01, vw01, vout);
                        vout = _mm256_fmadd_ps(v10, vw10, vout);
                        vout = _mm256_fmadd_ps(v11, vw11, vout);

                        _mm256_storeu_ps(bev_out + d, vout);
                    }
                    for (; d < D; ++d) {
                        float interp = wy0 * (wx0 * f00[d] + wx1 * f01[d]) + wy1 * (wx0 * f10[d] + wx1 * f11[d]);
                        bev_out[d] += weight * interp;
                    }
#else
                    for (int64_t d = 0; d < D; ++d) {
                        float interp = wy0 * (wx0 * f00[d] + wx1 * f01[d]) + wy1 * (wx0 * f10[d] + wx1 * f11[d]);
                        bev_out[d] += weight * interp;
                    }
#endif
                }

                // Normalize by total weight
                if (total_weight > 0.0f) {
                    float inv_weight = 1.0f / (1.0f + total_weight);
                    for (int64_t d = 0; d < D; ++d) {
                        bev_out[d] *= inv_weight;
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuBEVQueryOp, OpType::BEVQuery, DeviceType::CPU);

}  // namespace
}  // namespace densecore
