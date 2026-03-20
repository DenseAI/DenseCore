/**
 * @file image_ops.h
 * @brief Image Preprocessing Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * Image format conversion + normalization:
 * - RGBToFloat: RGB uint8 NHWC → float CHW + ImageNet normalize
 * - YUV422ToFloat: YUYV → float CHW + ImageNet normalize
 */

#ifndef DENSECORE_OPS_IMAGE_OPS_H
#define DENSECORE_OPS_IMAGE_OPS_H

#include "../hal/tensor.h"
#include "../hal/transformer_ops.h"

namespace densecore {
namespace ops {

/**
 * @brief Image Preprocessing Operations Interface
 *
 * Converts raw camera/image data to normalized float tensors
 * suitable for vision model input (ViT, LLaVA, VLA, etc.).
 */
class ImageOps : public DenseCoreOp {
public:
    virtual ~ImageOps() = default;

    /**
     * @brief RGB uint8 [H,W,3] NHWC → float [3,H,W] CHW + normalize
     *
     * Applies: output[c][y][x] = (input[y][x][c] / 255.0 - mean[c]) / std[c]
     *
     * @param input  INT8 tensor [H, W, 3] in NHWC layout
     * @param output F32 tensor [3, H, W] in NCHW layout
     * @param mean   Per-channel mean (e.g., ImageNet {0.485, 0.456, 0.406})
     * @param std    Per-channel std  (e.g., ImageNet {0.229, 0.224, 0.225})
     */
    virtual void RGBToFloat(const Tensor& input, Tensor* output, const float mean[3], const float std[3]) = 0;

    /**
     * @brief Fused RGB uint8 [H,W,3] NHWC -> normalized patch tokens [N, patch_dim]
     *
     * Single-pass kernel for robotics camera pipelines. This avoids intermediate
     * CHW buffers and writes patch tokens directly in channel-major patch order:
     *   token[k] layout = [R(PxP), G(PxP), B(PxP)]
     *
     * @param input      INT8 tensor [H, W, 3] in NHWC layout
     * @param output     F32 tensor [N, patch_dim] where
     *                   N=(H/patch_size)*(W/patch_size), patch_dim=3*patch_size*patch_size
     * @param patch_size Patch edge length (e.g. 14 or 16)
     * @param mean       Per-channel mean
     * @param std        Per-channel std
     * @return true on success, false if shapes/params are invalid
     */
    virtual bool RGBToPatchTokensFused(const Tensor& input, Tensor* output, int patch_size, const float mean[3],
                                       const float std[3]) {
        (void)input;
        (void)output;
        (void)patch_size;
        (void)mean;
        (void)std;
        return false;
    }

    /**
     * @brief YUV422 (YUYV) → float [3,H,W] CHW + normalize
     *
     * Converts YUYV packed format to RGB, then normalizes.
     * BT.601 color space conversion.
     *
     * @param input  INT8 tensor [H, W*2] (YUYV packed, 2 bytes/pixel)
     * @param output F32 tensor [3, H, W] in NCHW layout
     * @param mean   Per-channel mean
     * @param std    Per-channel std
     */
    virtual void YUV422ToFloat(const Tensor& input, Tensor* output, const float mean[3], const float std[3]) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_IMAGE_OPS_H
