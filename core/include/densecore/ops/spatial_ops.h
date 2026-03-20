/**
 * @file spatial_ops.h
 * @brief Unified Spatial Transformation Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * Spatial transformation operations:
 * - GridSample (BEVFormer, Deformable Attention support)
 * - ScatterReduce (Point Cloud to Grid)
 * - BEVQuery (3D to BEV projection)
 */

#ifndef DENSECORE_OPS_SPATIAL_OPS_H
#define DENSECORE_OPS_SPATIAL_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Spatial Transformation Operations Interface
 *
 * Spatial sampling and transformation for Vision/3D models.
 */
class SpatialOps {
public:
    virtual ~SpatialOps() = default;

    // =========================================================================
    // Grid Sampling
    // =========================================================================

    /**
     * @brief 2D Grid Sampling (Bilinear interpolation)
     *
     * Sample from input using normalized grid coordinates.
     * Core operation for Spatial Transformer Networks and BEV projection.
     *
     * Time: O(B * C * H_out * W_out)
     *
     * @param input Input tensor [batch, channels, height_in, width_in]
     * @param grid Sampling grid [batch, height_out, width_out, 2] (normalized -1 to 1)
     * @param output Output tensor [batch, channels, height_out, width_out]
     * @param align_corners If true, grid corners align with pixel centers
     */
    virtual void GridSample2D(const Tensor& input, const Tensor& grid, Tensor* output, bool align_corners = false) = 0;

    /**
     * @brief 3D Grid Sampling (Trilinear interpolation)
     *
     * Sample from 3D volume using normalized grid coordinates.
     * Used in 3D vision and video models.
     *
     * @param input Input tensor [batch, channels, depth, height, width]
     * @param grid Sampling grid [batch, D_out, H_out, W_out, 3]
     * @param output Output tensor [batch, channels, D_out, H_out, W_out]
     * @param align_corners If true, grid corners align with voxel centers
     */
    virtual void GridSample3D(const Tensor& input, const Tensor& grid, Tensor* output, bool align_corners = false) = 0;

    // =========================================================================
    // Scatter/Reduce Operations
    // =========================================================================

    /**
     * @brief Scatter points to regular grid with reduction
     *
     * Convert irregular point cloud to regular grid via scatter-reduce.
     * Used in LiDAR perception and 3D Gaussian Splatting.
     *
     * @param points Point features [num_points, dim]
     * @param coords Point coordinates [num_points, 3] (x, y, z)
     * @param output Output grid [depth, height, width, dim]
     * @param reduce Reduction mode: "sum", "mean", "max"
     */
    virtual void ScatterReduce(const Tensor& points, const Tensor& coords, Tensor* output,
                               const char* reduce = "mean") = 0;

    // =========================================================================
    // BEV Query
    // =========================================================================

    /**
     * @brief Bird's Eye View query from 3D features
     *
     * Project 3D spatial features to BEV plane.
     * Core operation for BEVFormer and autonomous driving perception.
     *
     * @param bev_queries BEV query features [batch, H_bev, W_bev, dim]
     * @param spatial_features Multi-camera spatial features [batch, num_cams, H, W, dim]
     * @param lidar2img Camera projection matrices [batch, num_cams, 4, 4]
     * @param output Output BEV features [batch, H_bev, W_bev, dim]
     */
    virtual void BEVQuery(const Tensor& bev_queries, const Tensor& spatial_features, const Tensor& lidar2img,
                          Tensor* output) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_SPATIAL_OPS_H
