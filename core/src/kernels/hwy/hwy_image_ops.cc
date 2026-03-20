/**
 * @file hwy_image_ops.cc
 * @brief Image preprocessing kernels via Google Highway
 *
 * Provides portable SIMD for:
 * - RGB uint8 NHWC -> float CHW + normalize
 * - RGB uint8 NHWC -> fused normalized patch tokens
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_image_ops.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cstdint>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
static constexpr float kScale = 1.0f / 255.0f;

void RGBToFloatImpl(const uint8_t* HWY_RESTRICT rgb, float* HWY_RESTRICT rgb_chw, int width, int height,
                    const float mean[3], const float std[3]) {
    if (!rgb || !rgb_chw || !mean || !std || width <= 0 || height <= 0) {
        return;
    }

    const int64_t n_pixels = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (n_pixels <= 0) {
        return;
    }

    float* r_plane = rgb_chw;
    float* g_plane = rgb_chw + n_pixels;
    float* b_plane = rgb_chw + 2 * n_pixels;

    const float inv_std_r = 1.0f / std[0];
    const float inv_std_g = 1.0f / std[1];
    const float inv_std_b = 1.0f / std[2];

    const hn::CappedTag<float, 8> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes <= 0) {
        return;
    }

    const auto v_scale = hn::Set(d, kScale);
    const auto v_mean_r = hn::Set(d, mean[0]);
    const auto v_mean_g = hn::Set(d, mean[1]);
    const auto v_mean_b = hn::Set(d, mean[2]);
    const auto v_inv_std_r = hn::Set(d, inv_std_r);
    const auto v_inv_std_g = hn::Set(d, inv_std_g);
    const auto v_inv_std_b = hn::Set(d, inv_std_b);

    thread_local std::vector<float> r_lane;
    thread_local std::vector<float> g_lane;
    thread_local std::vector<float> b_lane;
    r_lane.resize(static_cast<size_t>(lanes));
    g_lane.resize(static_cast<size_t>(lanes));
    b_lane.resize(static_cast<size_t>(lanes));

    int64_t i = 0;
    for (; i + lanes <= n_pixels; i += lanes) {
        for (int j = 0; j < lanes; ++j) {
            const size_t pix = static_cast<size_t>((i + j) * 3);
            r_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 0]);
            g_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 1]);
            b_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 2]);
        }

        auto vr = hn::LoadU(d, r_lane.data());
        auto vg = hn::LoadU(d, g_lane.data());
        auto vb = hn::LoadU(d, b_lane.data());

        vr = hn::Mul(hn::Sub(hn::Mul(vr, v_scale), v_mean_r), v_inv_std_r);
        vg = hn::Mul(hn::Sub(hn::Mul(vg, v_scale), v_mean_g), v_inv_std_g);
        vb = hn::Mul(hn::Sub(hn::Mul(vb, v_scale), v_mean_b), v_inv_std_b);

        hn::StoreU(vr, d, r_plane + i);
        hn::StoreU(vg, d, g_plane + i);
        hn::StoreU(vb, d, b_plane + i);
    }

    for (; i < n_pixels; ++i) {
        const size_t pix = static_cast<size_t>(i * 3);
        const float r = static_cast<float>(rgb[pix + 0]) * kScale;
        const float g = static_cast<float>(rgb[pix + 1]) * kScale;
        const float b = static_cast<float>(rgb[pix + 2]) * kScale;

        r_plane[i] = (r - mean[0]) * inv_std_r;
        g_plane[i] = (g - mean[1]) * inv_std_g;
        b_plane[i] = (b - mean[2]) * inv_std_b;
    }
}

bool RGBToPatchTokensFusedImpl(const uint8_t* HWY_RESTRICT rgb, float* HWY_RESTRICT patch_tokens, int width, int height,
                               int patch_size, const float mean[3], const float std[3]) {
    if (!rgb || !patch_tokens || !mean || !std || width <= 0 || height <= 0 || patch_size <= 0) {
        return false;
    }
    if (width % patch_size != 0 || height % patch_size != 0) {
        return false;
    }
    if (std[0] == 0.0f || std[1] == 0.0f || std[2] == 0.0f) {
        return false;
    }

    const int patches_h = height / patch_size;
    const int patches_w = width / patch_size;
    const int patch_area = patch_size * patch_size;
    const int patch_dim = patch_area * 3;

    const float inv_std_r = 1.0f / std[0];
    const float inv_std_g = 1.0f / std[1];
    const float inv_std_b = 1.0f / std[2];

    const hn::CappedTag<float, 8> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes <= 0) {
        return false;
    }

    const auto v_scale = hn::Set(d, kScale);
    const auto v_mean_r = hn::Set(d, mean[0]);
    const auto v_mean_g = hn::Set(d, mean[1]);
    const auto v_mean_b = hn::Set(d, mean[2]);
    const auto v_inv_std_r = hn::Set(d, inv_std_r);
    const auto v_inv_std_g = hn::Set(d, inv_std_g);
    const auto v_inv_std_b = hn::Set(d, inv_std_b);

    thread_local std::vector<float> r_lane;
    thread_local std::vector<float> g_lane;
    thread_local std::vector<float> b_lane;
    r_lane.resize(static_cast<size_t>(lanes));
    g_lane.resize(static_cast<size_t>(lanes));
    b_lane.resize(static_cast<size_t>(lanes));

    for (int ph = 0; ph < patches_h; ++ph) {
        for (int pw = 0; pw < patches_w; ++pw) {
            float* patch_out = patch_tokens + (static_cast<int64_t>(ph) * patches_w + pw) * patch_dim;
            float* patch_r = patch_out;
            float* patch_g = patch_out + patch_area;
            float* patch_b = patch_out + 2 * patch_area;

            const int h0 = ph * patch_size;
            const int w0 = pw * patch_size;

            for (int kh = 0; kh < patch_size; ++kh) {
                const int row = h0 + kh;
                const int patch_row_off = kh * patch_size;
                const int64_t pix_base = (static_cast<int64_t>(row) * width + w0) * 3;

                int kw = 0;
                for (; kw + lanes <= patch_size; kw += lanes) {
                    for (int j = 0; j < lanes; ++j) {
                        const size_t pix = static_cast<size_t>(pix_base + static_cast<int64_t>(kw + j) * 3);
                        r_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 0]);
                        g_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 1]);
                        b_lane[static_cast<size_t>(j)] = static_cast<float>(rgb[pix + 2]);
                    }

                    auto vr = hn::LoadU(d, r_lane.data());
                    auto vg = hn::LoadU(d, g_lane.data());
                    auto vb = hn::LoadU(d, b_lane.data());

                    vr = hn::Mul(hn::Sub(hn::Mul(vr, v_scale), v_mean_r), v_inv_std_r);
                    vg = hn::Mul(hn::Sub(hn::Mul(vg, v_scale), v_mean_g), v_inv_std_g);
                    vb = hn::Mul(hn::Sub(hn::Mul(vb, v_scale), v_mean_b), v_inv_std_b);

                    const int off = patch_row_off + kw;
                    hn::StoreU(vr, d, patch_r + off);
                    hn::StoreU(vg, d, patch_g + off);
                    hn::StoreU(vb, d, patch_b + off);
                }

                for (; kw < patch_size; ++kw) {
                    const size_t pix = static_cast<size_t>(pix_base + static_cast<int64_t>(kw) * 3);
                    const int patch_idx = patch_row_off + kw;

                    const float r = static_cast<float>(rgb[pix + 0]) * kScale;
                    const float g = static_cast<float>(rgb[pix + 1]) * kScale;
                    const float b = static_cast<float>(rgb[pix + 2]) * kScale;

                    patch_r[patch_idx] = (r - mean[0]) * inv_std_r;
                    patch_g[patch_idx] = (g - mean[1]) * inv_std_g;
                    patch_b[patch_idx] = (b - mean[2]) * inv_std_b;
                }
            }
        }
    }

    return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(RGBToFloatImpl);
HWY_EXPORT(RGBToPatchTokensFusedImpl);

void RGBToFloat_Hwy(const uint8_t* rgb, float* rgb_chw, int width, int height, const float mean[3],
                    const float std[3]) {
    HWY_DYNAMIC_DISPATCH(RGBToFloatImpl)(rgb, rgb_chw, width, height, mean, std);
}

bool RGBToPatchTokensFused_Hwy(const uint8_t* rgb, float* patch_tokens, int width, int height, int patch_size,
                               const float mean[3], const float std[3]) {
    return HWY_DYNAMIC_DISPATCH(RGBToPatchTokensFusedImpl)(rgb, patch_tokens, width, height, patch_size, mean, std);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
