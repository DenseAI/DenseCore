/**
 * @file pos_interpolation.cpp
 * @brief Positional Embedding Interpolation for Dynamic Resolution (LLaVA, Qwen-VL)
 *
 * Allows Vision Transformers trained at one resolution to process
 * images at different resolutions by interpolating position embeddings.
 *
 * Supports: Bilinear, Bicubic interpolation
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Interpolation Mode
// ============================================================================

enum class InterpolationMode : int { BILINEAR = 0, BICUBIC = 1, NEAREST = 2 };

struct PosInterpolationParams {
    int target_h = 0;
    int target_w = 0;
    int mode = 0;  // InterpolationMode
};

// ============================================================================
// Cubic Helper
// ============================================================================

inline float CubicWeight(float x) {
    const float a = -0.5f;  // Catmull-Rom
    float abs_x = std::abs(x);
    if (abs_x <= 1.0f) {
        return ((a + 2.0f) * abs_x - (a + 3.0f)) * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return ((a * abs_x - 5.0f * a) * abs_x + 8.0f * a) * abs_x - 4.0f * a;
    }
    return 0.0f;
}

// ============================================================================
// CpuPosInterpolationOp
// ============================================================================

class CpuPosInterpolationOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;

        const PosInterpolationParams* p = static_cast<const PosInterpolationParams*>(params);
        PosInterpolationParams default_params{14, 14, 0};
        if (!p) p = &default_params;

        InterpolatePositions(*inputs[0], p->target_h, p->target_w, static_cast<InterpolationMode>(p->mode), outputs[0]);
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
     * @brief Interpolate 2D position embeddings
     *
     * @param input      [H_orig, W_orig, D] or [N_orig, D] - Original position embeddings
     * @param target_h   Target height
     * @param target_w   Target width
     * @param mode       Interpolation mode
     * @param output     [N_target, D] where N_target = target_h * target_w
     */
    void InterpolatePositions(const Tensor& input, int target_h, int target_w, InterpolationMode mode, Tensor* output) {
        if (!input.IsValid() || !output) return;

        // Determine original grid size
        int64_t H_orig, W_orig, D;
        if (input.shape.size() == 3) {
            H_orig = input.shape[0];
            W_orig = input.shape[1];
            D = input.shape[2];
        } else {
            // Assume square grid
            int64_t N = input.shape[0];
            D = input.shape[1];
            H_orig = W_orig = static_cast<int64_t>(std::sqrt(static_cast<double>(N)));
        }

        const float* in_data = input.DataAs<float>();
        float* out_data = output->DataAs<float>();

        const float scale_h = static_cast<float>(H_orig) / static_cast<float>(target_h);
        const float scale_w = static_cast<float>(W_orig) / static_cast<float>(target_w);

        for (int th = 0; th < target_h; ++th) {
            for (int tw = 0; tw < target_w; ++tw) {
                float src_h = (th + 0.5f) * scale_h - 0.5f;
                float src_w = (tw + 0.5f) * scale_w - 0.5f;

                int64_t out_idx = (th * target_w + tw) * D;

                if (mode == InterpolationMode::NEAREST) {
                    int sh = static_cast<int>(std::round(src_h));
                    int sw = static_cast<int>(std::round(src_w));
                    sh = std::max(0, std::min(sh, static_cast<int>(H_orig - 1)));
                    sw = std::max(0, std::min(sw, static_cast<int>(W_orig - 1)));

                    const float* src = in_data + (sh * W_orig + sw) * D;
                    std::memcpy(out_data + out_idx, src, D * sizeof(float));

                } else if (mode == InterpolationMode::BILINEAR) {
                    int h0 = static_cast<int>(std::floor(src_h));
                    int w0 = static_cast<int>(std::floor(src_w));
                    int h1 = h0 + 1;
                    int w1 = w0 + 1;

                    float th_frac = src_h - h0;
                    float tw_frac = src_w - w0;

                    // Clamp
                    h0 = std::max(0, std::min(h0, static_cast<int>(H_orig - 1)));
                    h1 = std::max(0, std::min(h1, static_cast<int>(H_orig - 1)));
                    w0 = std::max(0, std::min(w0, static_cast<int>(W_orig - 1)));
                    w1 = std::max(0, std::min(w1, static_cast<int>(W_orig - 1)));

                    const float* p00 = in_data + (h0 * W_orig + w0) * D;
                    const float* p01 = in_data + (h0 * W_orig + w1) * D;
                    const float* p10 = in_data + (h1 * W_orig + w0) * D;
                    const float* p11 = in_data + (h1 * W_orig + w1) * D;

                    float w00 = (1 - th_frac) * (1 - tw_frac);
                    float w01 = (1 - th_frac) * tw_frac;
                    float w10 = th_frac * (1 - tw_frac);
                    float w11 = th_frac * tw_frac;

#if defined(__AVX2__)
                    __m256 vw00 = _mm256_set1_ps(w00);
                    __m256 vw01 = _mm256_set1_ps(w01);
                    __m256 vw10 = _mm256_set1_ps(w10);
                    __m256 vw11 = _mm256_set1_ps(w11);

                    int64_t d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 v00 = _mm256_loadu_ps(p00 + d);
                        __m256 v01 = _mm256_loadu_ps(p01 + d);
                        __m256 v10 = _mm256_loadu_ps(p10 + d);
                        __m256 v11 = _mm256_loadu_ps(p11 + d);

                        __m256 result = _mm256_mul_ps(v00, vw00);
                        result = _mm256_fmadd_ps(v01, vw01, result);
                        result = _mm256_fmadd_ps(v10, vw10, result);
                        result = _mm256_fmadd_ps(v11, vw11, result);

                        _mm256_storeu_ps(out_data + out_idx + d, result);
                    }
                    for (; d < D; ++d) {
                        out_data[out_idx + d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
                    }
#else
                    for (int64_t d = 0; d < D; ++d) {
                        out_data[out_idx + d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
                    }
#endif

                } else if (mode == InterpolationMode::BICUBIC) {
                    int h_floor = static_cast<int>(std::floor(src_h));
                    int w_floor = static_cast<int>(std::floor(src_w));
                    float frac_h = src_h - h_floor;
                    float frac_w = src_w - w_floor;

                    std::memset(out_data + out_idx, 0, D * sizeof(float));

                    for (int i = -1; i <= 2; ++i) {
                        float wh = CubicWeight(frac_h - i);
                        for (int j = -1; j <= 2; ++j) {
                            float ww = CubicWeight(frac_w - j);
                            float weight = wh * ww;

                            int sh = std::max(0, std::min(h_floor + i, static_cast<int>(H_orig - 1)));
                            int sw = std::max(0, std::min(w_floor + j, static_cast<int>(W_orig - 1)));

                            const float* src = in_data + (sh * W_orig + sw) * D;
                            for (int64_t d = 0; d < D; ++d) {
                                out_data[out_idx + d] += weight * src[d];
                            }
                        }
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuPosInterpolationOp, OpType::PosInterpolation, DeviceType::CPU);
// Note: Using Custom OpType until we add a dedicated PositionalInterpolation OpType

}  // namespace
}  // namespace densecore
