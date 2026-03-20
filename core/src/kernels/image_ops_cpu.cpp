/**
 * @file image_ops_cpu.cpp
 * @brief CPU ImageOps registration and lightweight dispatch
 *
 * RGB paths call Highway kernels directly (no extra wrapper layer).
 * YUV422 remains scalar due packed YUYV conversion complexity.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/ops/image_ops.h"
#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace densecore {
namespace kernels {
namespace {

static constexpr float kScale = 1.0f / 255.0f;

static void YUV422ToFloat_Scalar(const uint8_t* yuv, float* rgb_chw, int width, int height, const float mean[3],
                                 const float std[3]) {
    if (!yuv || !rgb_chw || !mean || !std || width <= 0 || height <= 0) {
        return;
    }

    const int n_pixels = width * height;
    float* r_plane = rgb_chw;
    float* g_plane = rgb_chw + n_pixels;
    float* b_plane = rgb_chw + 2 * n_pixels;

    const float inv_std_r = 1.0f / std[0];
    const float inv_std_g = 1.0f / std[1];
    const float inv_std_b = 1.0f / std[2];

    for (int i = 0; i < n_pixels; i += 2) {
        const int byte_idx = i * 2;
        const float y0 = static_cast<float>(yuv[byte_idx]);
        const float u = static_cast<float>(yuv[byte_idx + 1]) - 128.0f;
        const float y1 = static_cast<float>(yuv[byte_idx + 2]);
        const float v = static_cast<float>(yuv[byte_idx + 3]) - 128.0f;

        auto to_rgb = [&](float y, int idx) {
            float r = y + 1.402f * v;
            float g = y - 0.344136f * u - 0.714136f * v;
            float b = y + 1.772f * u;

            r = std::fmin(255.0f, std::fmax(0.0f, r)) * kScale;
            g = std::fmin(255.0f, std::fmax(0.0f, g)) * kScale;
            b = std::fmin(255.0f, std::fmax(0.0f, b)) * kScale;

            r_plane[idx] = (r - mean[0]) * inv_std_r;
            g_plane[idx] = (g - mean[1]) * inv_std_g;
            b_plane[idx] = (b - mean[2]) * inv_std_b;
        };

        to_rgb(y0, i);
        if (i + 1 < n_pixels) {
            to_rgb(y1, i + 1);
        }
    }
}

class CpuImageOps : public ops::ImageOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        (void)inputs;
        (void)outputs;
        (void)params;
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    void RGBToFloat(const Tensor& input, Tensor* output, const float mean[3], const float std[3]) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;
        if (input.dtype != DType::INT8 || output->dtype != DType::F32) return;
        if (input.ndim != 3 || input.shape[2] != 3) return;
        if (output->ndim != 3 || output->shape[0] != 3 || output->shape[1] != input.shape[0] ||
            output->shape[2] != input.shape[1]) {
            return;
        }

        const auto* rgb = input.DataAs<uint8_t>();
        auto* rgb_chw = output->DataAs<float>();
        const int height = static_cast<int>(input.shape[0]);
        const int width = static_cast<int>(input.shape[1]);

        hwy_kernels::RGBToFloat_Hwy(rgb, rgb_chw, width, height, mean, std);
    }

    bool RGBToPatchTokensFused(const Tensor& input, Tensor* output, int patch_size, const float mean[3],
                               const float std[3]) override {
        if (!input.IsValid() || !output || !output->IsValid()) return false;
        if (input.dtype != DType::INT8 || output->dtype != DType::F32) return false;
        if (input.ndim != 3 || input.shape[2] != 3) return false;
        if (output->ndim < 2) return false;

        const int height = static_cast<int>(input.shape[0]);
        const int width = static_cast<int>(input.shape[1]);
        if (patch_size <= 0 || width % patch_size != 0 || height % patch_size != 0) return false;

        const int patch_area = patch_size * patch_size;
        const int patch_dim = patch_area * 3;
        const int expected_patches = (height / patch_size) * (width / patch_size);
        if (output->shape[0] != expected_patches || output->shape[1] != patch_dim) return false;

        const auto* rgb = input.DataAs<uint8_t>();
        auto* out = output->DataAs<float>();
        return hwy_kernels::RGBToPatchTokensFused_Hwy(rgb, out, width, height, patch_size, mean, std);
    }

    void YUV422ToFloat(const Tensor& input, Tensor* output, const float mean[3], const float std[3]) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;
        if (input.dtype != DType::INT8 || output->dtype != DType::F32) return;
        if (input.ndim != 2) return;
        if (output->ndim != 3 || output->shape[0] != 3 || output->shape[1] != input.shape[0] ||
            output->shape[2] != input.shape[1] / 2) {
            return;
        }

        const auto* yuv = input.DataAs<uint8_t>();
        auto* rgb_chw = output->DataAs<float>();
        const int height = static_cast<int>(input.shape[0]);
        const int width = static_cast<int>(input.shape[1]) / 2;

        YUV422ToFloat_Scalar(yuv, rgb_chw, width, height, mean, std);
    }
};

DENSECORE_REGISTER_OP(CpuImageOps, OpType::RGBToFloat, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuImageOps, OpType::YUV422ToFloat, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
