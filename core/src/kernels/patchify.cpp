/**
 * @file patchify.cpp
 * @brief Patchify for Diffusion Transformers (DiT, Flux)
 *
 * Converts image to patch sequence (without projection):
 * [B, C, H, W] -> [B, (H/P)*(W/P), P*P*C]
 *
 * Unlike PatchEmbed2D which includes projection to embed_dim,
 * Patchify only reshapes the image into patches.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Patchify Parameters
// ============================================================================

struct PatchifyParams {
    int patch_size = 16;
};

// ============================================================================
// CpuPatchifyOp
// ============================================================================

class CpuPatchifyOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const PatchifyParams* p = static_cast<const PatchifyParams*>(params);
        PatchifyParams default_params;
        if (!p) p = &default_params;

        Patchify(*inputs[0], p->patch_size, outputs[0]);
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
     * @brief Convert image to patches
     *
     * @param image      [B, C, H, W] input image
     * @param patch_size Patch size P
     * @param patches    [B, N, patch_dim] where N = (H/P)*(W/P), patch_dim = P*P*C
     */
    void Patchify(const Tensor& image, int patch_size, Tensor* patches) {
        if (!image.IsValid() || !patches) return;

        const int64_t B = image.shape[0];
        const int64_t C = image.shape[1];
        const int64_t H = image.shape[2];
        const int64_t W = image.shape[3];
        const int64_t P = patch_size;

        if (P == 0) return;

        const int64_t num_patches_h = H / P;
        const int64_t num_patches_w = W / P;
        const int64_t N = num_patches_h * num_patches_w;
        const int64_t patch_dim = C * P * P;

        const float* img_data = image.DataAs<float>();
        float* patch_data = patches->DataAs<float>();

        // Convert image to patches
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t ph = 0; ph < num_patches_h; ++ph) {
                for (int64_t pw = 0; pw < num_patches_w; ++pw) {
                    const int64_t patch_idx = ph * num_patches_w + pw;
                    float* patch = patch_data + b * N * patch_dim + patch_idx * patch_dim;

                    const int64_t h_start = ph * P;
                    const int64_t w_start = pw * P;

                    // Flatten: C -> H -> W within patch
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < P; ++kh) {
                            for (int64_t kw = 0; kw < P; ++kw) {
                                const int64_t img_offset =
                                    b * C * H * W + c * H * W + (h_start + kh) * W + (w_start + kw);
                                const int64_t patch_offset = c * P * P + kh * P + kw;
                                patch[patch_offset] = img_data[img_offset];
                            }
                        }
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuPatchifyOp, OpType::Patchify, DeviceType::CPU);

}  // namespace
}  // namespace densecore
