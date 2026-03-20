/**
 * @file hwy_groupnorm.cc
 * @brief GroupNorm (NCHW + NHWC) via Google Highway
 *
 * Replaces the hand-written AVX2 loops in groupnorm.cpp with portable SIMD.
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_groupnorm.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"

#include <cmath>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// NCHW GroupNorm
// ============================================================================

void GroupNormNCHWImpl(const float* HWY_RESTRICT input, const float* HWY_RESTRICT gamma, const float* HWY_RESTRICT beta,
                       float* HWY_RESTRICT output, int64_t B, int64_t C, int64_t H, int64_t W, int num_groups,
                       float eps) {
    const hn::ScalableTag<float> d;
    if (num_groups <= 0 || C % num_groups != 0) {
        return;
    }
    const size_t N = hn::Lanes(d);

    const int64_t channels_per_group = C / num_groups;
    const int64_t spatial_size = H * W;
    const int64_t group_size = channels_per_group * spatial_size;

    for (int64_t b = 0; b < B; ++b) {
        for (int g = 0; g < num_groups; ++g) {
            const int64_t group_start_c = g * channels_per_group;

            // Pass 1: Compute mean and variance
            auto v_sum = hn::Zero(d);
            auto v_sum_sq = hn::Zero(d);
            float scalar_sum = 0.0f;
            float scalar_sum_sq = 0.0f;

            for (int64_t c = group_start_c; c < group_start_c + channels_per_group; ++c) {
                const float* ptr = input + b * C * spatial_size + c * spatial_size;
                size_t i = 0;
                for (; i + N <= static_cast<size_t>(spatial_size); i += N) {
                    const auto v = hn::LoadU(d, ptr + i);
                    v_sum = hn::Add(v_sum, v);
                    v_sum_sq = hn::MulAdd(v, v, v_sum_sq);
                }
                for (; i < static_cast<size_t>(spatial_size); ++i) {
                    float val = ptr[i];
                    scalar_sum += val;
                    scalar_sum_sq += val * val;
                }
            }
            float sum = hn::ReduceSum(d, v_sum) + scalar_sum;
            float sum_sq = hn::ReduceSum(d, v_sum_sq) + scalar_sum_sq;
            float mean = sum / static_cast<float>(group_size);
            float var = (sum_sq / static_cast<float>(group_size)) - (mean * mean);
            if (var < 0.0f) var = 0.0f;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // Pass 2: Normalize, scale, shift
            for (int64_t c = group_start_c; c < group_start_c + channels_per_group; ++c) {
                float g_val = gamma ? gamma[c] : 1.0f;
                float b_val = beta ? beta[c] : 0.0f;
                float scale = inv_std * g_val;
                float shift = b_val - mean * scale;

                const auto v_scale = hn::Set(d, scale);
                const auto v_shift = hn::Set(d, shift);

                const float* ptr_in = input + b * C * spatial_size + c * spatial_size;
                float* ptr_out = output + b * C * spatial_size + c * spatial_size;

                size_t i = 0;
                for (; i + N <= static_cast<size_t>(spatial_size); i += N) {
                    const auto v = hn::LoadU(d, ptr_in + i);
                    hn::StoreU(hn::MulAdd(v, v_scale, v_shift), d, ptr_out + i);
                }
                for (; i < static_cast<size_t>(spatial_size); ++i) {
                    ptr_out[i] = ptr_in[i] * scale + shift;
                }
            }
        }
    }
}

// ============================================================================
// NHWC GroupNorm
// ============================================================================

void GroupNormNHWCImpl(const float* HWY_RESTRICT input, const float* HWY_RESTRICT gamma, const float* HWY_RESTRICT beta,
                       float* HWY_RESTRICT output, int64_t B, int64_t C, int64_t H, int64_t W, int num_groups,
                       float eps) {
    const hn::ScalableTag<float> d;
    if (num_groups <= 0 || C % num_groups != 0) {
        return;
    }
    const size_t N_lanes = hn::Lanes(d);

    const int64_t channels_per_group = C / num_groups;
    const int64_t spatial_size = H * W;
    const int64_t group_size = channels_per_group * spatial_size;

    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> group_sums(num_groups, 0.0f);
        std::vector<float> group_sq_sums(num_groups, 0.0f);

        // Pass 1: Sum and sum-of-squares
        for (int64_t pixel = 0; pixel < spatial_size; ++pixel) {
            const float* p_in = input + b * spatial_size * C + pixel * C;
            for (int g = 0; g < num_groups; ++g) {
                const float* group_ptr = p_in + g * channels_per_group;
                auto v_s = hn::Zero(d);
                auto v_ss = hn::Zero(d);
                size_t c = 0;
                for (; c + N_lanes <= static_cast<size_t>(channels_per_group); c += N_lanes) {
                    const auto val = hn::LoadU(d, group_ptr + c);
                    v_s = hn::Add(v_s, val);
                    v_ss = hn::MulAdd(val, val, v_ss);
                }
                float s = hn::ReduceSum(d, v_s);
                float ss = hn::ReduceSum(d, v_ss);
                for (; c < static_cast<size_t>(channels_per_group); ++c) {
                    float val = group_ptr[c];
                    s += val;
                    ss += val * val;
                }
                group_sums[g] += s;
                group_sq_sums[g] += ss;
            }
        }

        // Compute mean and inv_std per group
        std::vector<float> means(num_groups);
        std::vector<float> inv_stds(num_groups);
        for (int g = 0; g < num_groups; ++g) {
            float mean = group_sums[g] / static_cast<float>(group_size);
            float var = (group_sq_sums[g] / static_cast<float>(group_size)) - (mean * mean);
            if (var < 0.0f) var = 0.0f;
            means[g] = mean;
            inv_stds[g] = 1.0f / std::sqrt(var + eps);
        }

        // Pass 2: Normalize
        for (int64_t pixel = 0; pixel < spatial_size; ++pixel) {
            const float* p_in = input + b * spatial_size * C + pixel * C;
            float* p_out = output + b * spatial_size * C + pixel * C;

            for (int g = 0; g < num_groups; ++g) {
                float inv_std = inv_stds[g];
                float m = means[g];
                const int64_t c_start = g * channels_per_group;

                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t global_c = c_start + c;
                    float val = p_in[global_c];
                    float g_val = gamma ? gamma[global_c] : 1.0f;
                    float b_val = beta ? beta[global_c] : 0.0f;
                    p_out[global_c] = (val - m) * inv_std * g_val + b_val;
                }
            }
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(GroupNormNCHWImpl);
HWY_EXPORT(GroupNormNHWCImpl);

void GroupNormNCHW_Hwy(const float* input, const float* gamma, const float* beta, float* output, int64_t B, int64_t C,
                       int64_t H, int64_t W, int num_groups, float eps) {
    HWY_DYNAMIC_DISPATCH(GroupNormNCHWImpl)(input, gamma, beta, output, B, C, H, W, num_groups, eps);
}

void GroupNormNHWC_Hwy(const float* input, const float* gamma, const float* beta, float* output, int64_t B, int64_t C,
                       int64_t H, int64_t W, int num_groups, float eps) {
    HWY_DYNAMIC_DISPATCH(GroupNormNHWCImpl)(input, gamma, beta, output, B, C, H, W, num_groups, eps);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
