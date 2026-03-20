/**
 * @file noise_schedule.cpp
 * @brief Noise Schedule for Diffusion Models (DiT, Flux, SORA)
 *
 * Computes noise levels (σ, α) for diffusion timesteps.
 * Supports various schedules: linear, cosine, exponential.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace densecore {
namespace {

// ============================================================================
// Schedule Types
// ============================================================================

enum class ScheduleType : int { LINEAR = 0, COSINE = 1, EXPONENTIAL = 2, SQRT = 3 };

struct NoiseScheduleParams {
    int schedule_type = 0;  // ScheduleType
    int num_steps = 1000;
    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    float s = 0.008f;  // Offset for cosine schedule
};

// ============================================================================
// CpuNoiseScheduleOp
// ============================================================================

class CpuNoiseScheduleOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.size() < 2) return;

        const NoiseScheduleParams* p = static_cast<const NoiseScheduleParams*>(params);
        NoiseScheduleParams default_params;
        if (!p) p = &default_params;

        NoiseSchedule(*inputs[0], *p, outputs[0], outputs[1]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 1 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    /**
     * @brief Compute noise schedule values
     *
     * @param timesteps  [B] or [B, N] - Timestep indices (0 to num_steps-1)
     * @param params     Schedule configuration
     * @param alphas     [B] or [B, N] - Alpha values (signal scaling)
     * @param sigmas     [B] or [B, N] - Sigma values (noise level)
     */
    void NoiseSchedule(const Tensor& timesteps, const NoiseScheduleParams& params, Tensor* alphas, Tensor* sigmas) {
        if (!timesteps.IsValid() || !alphas || !sigmas) return;

        const int64_t total = timesteps.NumElements();
        const int32_t* t_data = timesteps.DataAs<int32_t>();
        float* alpha_data = alphas->DataAs<float>();
        float* sigma_data = sigmas->DataAs<float>();

        const float T = static_cast<float>(params.num_steps);
        const ScheduleType schedule = static_cast<ScheduleType>(params.schedule_type);

        // Precompute cumulative alphas for linear/exponential schedules
        std::vector<float> betas(params.num_steps);
        std::vector<float> alpha_cumprod(params.num_steps);

        if (schedule == ScheduleType::LINEAR) {
            for (int i = 0; i < params.num_steps; ++i) {
                float t_frac = static_cast<float>(i) / (T - 1);
                betas[i] = params.beta_start + t_frac * (params.beta_end - params.beta_start);
            }
        } else if (schedule == ScheduleType::EXPONENTIAL) {
            float log_start = std::log(params.beta_start);
            float log_end = std::log(params.beta_end);
            for (int i = 0; i < params.num_steps; ++i) {
                float t_frac = static_cast<float>(i) / (T - 1);
                betas[i] = std::exp(log_start + t_frac * (log_end - log_start));
            }
        } else if (schedule == ScheduleType::SQRT) {
            for (int i = 0; i < params.num_steps; ++i) {
                float t_frac = static_cast<float>(i) / (T - 1);
                betas[i] = params.beta_start + t_frac * t_frac * (params.beta_end - params.beta_start);
            }
        }

        // Compute cumulative products for non-cosine schedules
        if (schedule != ScheduleType::COSINE) {
            alpha_cumprod[0] = 1.0f - betas[0];
            for (int i = 1; i < params.num_steps; ++i) {
                alpha_cumprod[i] = alpha_cumprod[i - 1] * (1.0f - betas[i]);
            }
        }

        // Compute alpha and sigma for each timestep
        for (int64_t i = 0; i < total; ++i) {
            int t = t_data[i];
            t = std::max(0, std::min(t, params.num_steps - 1));

            float alpha_t, sigma_t;

            if (schedule == ScheduleType::COSINE) {
                // Cosine schedule: alpha_bar(t) = f(t)/f(0)
                // where f(t) = cos((t/T + s)/(1 + s) * π/2)²
                auto f = [&](float t_val) {
                    float x = (t_val / T + params.s) / (1.0f + params.s);
                    float cs = std::cos(x * 3.14159265f / 2.0f);
                    return cs * cs;
                };
                float alpha_bar_t = f(static_cast<float>(t)) / f(0.0f);
                alpha_bar_t = std::max(0.0001f, std::min(0.9999f, alpha_bar_t));

                alpha_t = std::sqrt(alpha_bar_t);
                sigma_t = std::sqrt(1.0f - alpha_bar_t);
            } else {
                float alpha_bar_t = alpha_cumprod[t];
                alpha_bar_t = std::max(0.0001f, std::min(0.9999f, alpha_bar_t));

                alpha_t = std::sqrt(alpha_bar_t);
                sigma_t = std::sqrt(1.0f - alpha_bar_t);
            }

            alpha_data[i] = alpha_t;
            sigma_data[i] = sigma_t;
        }
    }
};

DENSECORE_REGISTER_OP(CpuNoiseScheduleOp, OpType::NoiseSchedule, DeviceType::CPU);

}  // namespace
}  // namespace densecore
