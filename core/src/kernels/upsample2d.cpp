/**
 * @file upsample2d.cpp
 * @brief CPU 2D Upsampling Kernel (Nearest Neighbor / Bilinear)
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 */

#include "densecore/hal/op_registry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace densecore {
namespace {

/**
 * @brief Nearest-neighbor 2x upsample
 * 
 * Input: [B, C, H, W]
 * Output: [B, C, H*scale, W*scale]
 */
void UpsampleNearest(const float* input, float* output, int batch, int channels, int h, int w, int scale) {
    const int out_h = h * scale;
    const int out_w = w * scale;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_plane = input + (b * channels + c) * h * w;
            float* out_plane = output + (b * channels + c) * out_h * out_w;

            for (int iy = 0; iy < h; ++iy) {
                for (int ix = 0; ix < w; ++ix) {
                    const float v = in_plane[iy * w + ix];
                    // Fill scale x scale block with same value
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            const int oy = iy * scale + sy;
                            const int ox = ix * scale + sx;
                            out_plane[oy * out_w + ox] = v;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Bilinear 2x upsample
 * 
 * Input: [B, C, H, W]
 * Output: [B, C, H*scale, W*scale]
 */
void UpsampleBilinear(const float* input, float* output, int batch, int channels, int h, int w, int scale) {
    const int out_h = h * scale;
    const int out_w = w * scale;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_plane = input + (b * channels + c) * h * w;
            float* out_plane = output + (b * channels + c) * out_h * out_w;

            for (int oy = 0; oy < out_h; ++oy) {
                for (int ox = 0; ox < out_w; ++ox) {
                    // Map output coord to input coord
                    const float iy_f = (static_cast<float>(oy) + 0.5f) / static_cast<float>(scale) - 0.5f;
                    const float ix_f = (static_cast<float>(ox) + 0.5f) / static_cast<float>(scale) - 0.5f;

                    const int iy0 = std::max(0, static_cast<int>(std::floor(iy_f)));
                    const int ix0 = std::max(0, static_cast<int>(std::floor(ix_f)));
                    const int iy1 = std::min(h - 1, iy0 + 1);
                    const int ix1 = std::min(w - 1, ix0 + 1);

                    const float dy = iy_f - static_cast<float>(iy0);
                    const float dx = ix_f - static_cast<float>(ix0);

                    // Bilinear interpolation
                    const float v00 = in_plane[iy0 * w + ix0];
                    const float v01 = in_plane[iy0 * w + ix1];
                    const float v10 = in_plane[iy1 * w + ix0];
                    const float v11 = in_plane[iy1 * w + ix1];

                    const float v0 = v00 * (1.0f - dx) + v01 * dx;
                    const float v1 = v10 * (1.0f - dx) + v11 * dx;

                    out_plane[oy * out_w + ox] = v0 * (1.0f - dy) + v1 * dy;
                }
            }
        }
    }
}

/**
 * @brief CPU Upsample2D Operation
 */
class CpuUpsample2DOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const Tensor& input = *inputs[0];
        Tensor& output = *outputs[0];

        Upsample2DParams default_params;
        const Upsample2DParams* p = static_cast<const Upsample2DParams*>(params);
        if (!p) p = &default_params;

        const int batch = static_cast<int>(input.shape[0]);
        const int channels = static_cast<int>(input.shape[1]);
        const int h = static_cast<int>(input.shape[2]);
        const int w = static_cast<int>(input.shape[3]);

        // Validation: scale_factor must be positive
        if (p->scale_factor <= 0) {
            // TODO(TechnicalDebt): Error handling
            return;
        }

        // Validation: Check for integer overflow in output dimensions
        // out_h = h * scale. If h > INT_MAX / scale, it overflows.
        if (h > std::numeric_limits<int>::max() / p->scale_factor ||
            w > std::numeric_limits<int>::max() / p->scale_factor) {
            // TODO(TechnicalDebt): Error handling
            return;
        }

        const float* in_data = input.DataAs<float>();
        float* out_data = output.DataAs<float>();

        if (p->bilinear) {
            UpsampleBilinear(in_data, out_data, batch, channels, h, w, p->scale_factor);
        } else {
            UpsampleNearest(in_data, out_data, batch, channels, h, w, p->scale_factor);
        }
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
};

// Register Upsample2D kernel
DENSECORE_REGISTER_OP(CpuUpsample2DOp, OpType::Upsample2D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
