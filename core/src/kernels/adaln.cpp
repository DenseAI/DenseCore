/**
 * @file adaln.cpp
 * @brief CPU Optimized AdaLN Execution
 */

#include "adaln.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/simd/simd_ops.h"

#include "kernels/hwy/hwy_kernels.h"
#include <cmath>

namespace densecore {
namespace kernels {

// ============================================================================
// AdaLN Kernel Implementation
// ============================================================================

void AdaLN(const Tensor& input, const Tensor& modulation, float eps, Tensor* output) {
    if (!input.IsValid() || !modulation.IsValid() || !output) return;

    const int64_t B = input.shape[0];
    const int64_t N = input.shape[1];
    const int64_t D = input.shape[2];

    const float* in_data = input.DataAs<float>();
    const float* mod_data = modulation.DataAs<float>();
    float* out_data = output->DataAs<float>();

    // Mod stride for batch; scale @0, shift @D
    const int64_t mod_stride_b = modulation.stride[0];

    for (int64_t b = 0; b < B; ++b) {
        const float* scale_ptr = mod_data + b * mod_stride_b;
        const float* shift_ptr = scale_ptr + D;  // Offset D (Fused: [scale, shift])

        const float* batch_in = in_data + b * N * D;
        float* batch_out = out_data + b * N * D;

        for (int64_t n = 0; n < N; ++n) {
            const float* token_in = batch_in + n * D;
            float* token_out = batch_out + n * D;
            simd::AdaLN(token_in, scale_ptr, shift_ptr, token_out, D, eps);
        }
    }
}

namespace {

class CpuAdaLNOp : public NormalizationOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        // We support two modes:
        // 1. Inputs: [Input, Scale, Shift] (Legacy/Split)
        // 2. Inputs: [Input, Modulation] (Fused)

        float eps = 1e-5f;
        if (params) {
            eps = static_cast<const AdaLNParams*>(params)->eps;
        }

        if (inputs.size() >= 3 && outputs.size() >= 1) {
            // Split mode
            AdaLN(*inputs[0], *inputs[1], *inputs[2], eps, outputs[0]);
        } else if (inputs.size() == 2 && outputs.size() >= 1) {
            // Fused mode
            kernels::AdaLN(*inputs[0], *inputs[1], eps, outputs[0]);
        }
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};  // Increased priority
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::SEQ || layout == TensorLayout::PATCH || layout == TensorLayout::UNKNOWN;
    }

    // Legacy/Split Overload
    void AdaLN(const Tensor& input, const Tensor& scale, const Tensor& shift, float eps, Tensor* output) override {
        if (!input.IsValid() || !scale.IsValid() || !shift.IsValid() || !output) return;

        const int64_t B = input.shape[0];
        const int64_t N = input.shape[1];
        const int64_t D = input.shape[2];

        const float* in_data = input.DataAs<float>();
        const float* scale_data = scale.DataAs<float>();
        const float* shift_data = shift.DataAs<float>();
        float* out_data = output->DataAs<float>();

        // Assume Scale/Shift are [B, D]
        for (int64_t b = 0; b < B; ++b) {
            const float* scale_ptr = scale_data + b * D;
            const float* shift_ptr = shift_data + b * D;

            const float* batch_in = in_data + b * N * D;
            float* batch_out = out_data + b * N * D;

            for (int64_t n = 0; n < N; ++n) {
                const float* token_in = batch_in + n * D;
                float* token_out = batch_out + n * D;
                simd::AdaLN(token_in, scale_ptr, shift_ptr, token_out, D, eps);
            }
        }
    }

    // New Fused Overload
    void AdaLN(const Tensor& input, const Tensor& modulation, float eps, Tensor* output) override {
        kernels::AdaLN(input, modulation, eps, output);
    }

    void GroupNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, int num_groups, float eps,
                   Tensor* output) override {
        // Delegate to pure CPU op logic
        bool is_nhwc = false;
        if (input.layout == TensorLayout::NHWC) {
            is_nhwc = true;
        } else if (input.layout == TensorLayout::UNKNOWN) {
            if (input.ndim == 4 && input.stride[1] == 1 && input.stride[3] > 1) {
                is_nhwc = true;
            }
        }

        int64_t B, C, H, W;
        if (is_nhwc) {
            B = input.shape[0];
            H = input.shape[1];
            W = input.shape[2];
            C = input.shape[3];
        } else {
            B = input.shape[0];
            C = input.shape[1];
            H = input.shape[2];
            W = input.shape[3];
        }

        const float* g_data = gamma.IsValid() ? gamma.DataAs<float>() : nullptr;
        const float* b_data = beta.IsValid() ? beta.DataAs<float>() : nullptr;

        if (is_nhwc) {
            hwy_kernels::GroupNormNHWC_Hwy(input.DataAs<float>(), g_data, b_data, output->DataAs<float>(), B, C, H, W,
                                           num_groups, eps);
        } else {
            hwy_kernels::GroupNormNCHW_Hwy(input.DataAs<float>(), g_data, b_data, output->DataAs<float>(), B, C, H, W,
                                           num_groups, eps);
        }
    }
};

DENSECORE_REGISTER_OP(CpuAdaLNOp, OpType::AdaLN, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
