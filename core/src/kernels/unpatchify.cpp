/**
 * @file unpatchify.cpp
 * @brief Unpatchify for Diffusion Transformers (DiT, Flux, SORA)
 *
 * Converts patch sequence back to image:
 * [B, (H/P)*(W/P), P*P*C] -> [B, C, H, W]
 *
 * This is the inverse of Patchify/PatchEmbed2D used in DiT decoders.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Unpatchify Parameters
// ============================================================================

struct UnpatchifyParams {
    int patch_size = 16;
    int height = 256;
    int width = 256;
    int channels = 3;
};

// ============================================================================
// CpuUnpatchifyOp
// ============================================================================

class CpuUnpatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const UnpatchifyParams* p = static_cast<const UnpatchifyParams*>(params);
        UnpatchifyParams default_params;
        if (!p) p = &default_params;

        Unpatchify(*inputs[0], p->height, p->width, p->patch_size, p->channels, outputs[0]);
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
     * @brief Convert patches back to image
     *
     * @param patches    [B, N, patch_dim] where N = (H/P)*(W/P), patch_dim = P*P*C
     * @param height     Original image height
     * @param width      Original image width
     * @param patch_size Patch size P
     * @param channels   Number of channels C
     * @param image      [B, C, H, W] output image
     */
    void Unpatchify(const Tensor& patches, int height, int width, int patch_size, int channels, Tensor* image) {
        if (!patches.IsValid() || !image) return;

        const int64_t B = patches.shape[0];
        const int64_t N = patches.shape[1];
        const int64_t patch_dim = patches.shape[2];

        const int64_t P = patch_size;
        const int64_t C = channels;
        const int64_t H = height;
        const int64_t W = width;

        if (P == 0) return;

        const int64_t num_patches_h = H / P;
        const int64_t num_patches_w = W / P;

        const float* patch_data = patches.DataAs<float>();
        float* img_data = image->DataAs<float>();

        // Reconstruct image from patches
        // Patch layout: [B, ph*pw, C*P*P]
        // Image layout: [B, C, H, W]

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t ph = 0; ph < num_patches_h; ++ph) {
                for (int64_t pw = 0; pw < num_patches_w; ++pw) {
                    const int64_t patch_idx = ph * num_patches_w + pw;
                    const float* patch = patch_data + b * N * patch_dim + patch_idx * patch_dim;

                    const int64_t h_start = ph * P;
                    const int64_t w_start = pw * P;

                    // Unflatten patch: C -> H -> W within patch
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < P; ++kh) {
                            for (int64_t kw = 0; kw < P; ++kw) {
                                const int64_t patch_offset = c * P * P + kh * P + kw;
                                const int64_t img_offset =
                                    b * C * H * W + c * H * W + (h_start + kh) * W + (w_start + kw);
                                img_data[img_offset] = patch[patch_offset];
                            }
                        }
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuUnpatchifyOp, OpType::Unpatchify, DeviceType::CPU);

}  // namespace
}  // namespace densecore
