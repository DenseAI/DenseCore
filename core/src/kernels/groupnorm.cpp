/**
 * @file groupnorm.cpp
 * @brief CPU GroupNorm — Highway SIMD delegation
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

#include <cmath>
#include <cstring>


namespace densecore {
namespace {

class CpuGroupNormOp : public NormalizationOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) {
            return;
        }
        int num_groups = 32;
        float eps = 1e-5f;
        if (params) {
            const auto* p = static_cast<const GroupNormParams*>(params);
            num_groups = p->num_groups;
            eps = p->eps;
        }
        Tensor gamma;
        Tensor beta;
        if (inputs.size() >= 2 && inputs[1]) {
            gamma = *inputs[1];
        }
        if (inputs.size() >= 3 && inputs[2]) {
            beta = *inputs[2];
        }

        GroupNorm(*inputs[0], gamma, beta, num_groups, eps, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 0};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::NCHW || layout == TensorLayout::NHWC || layout == TensorLayout::UNKNOWN;
    }

    void AdaLN(const Tensor& input, const Tensor& scale, const Tensor& shift, float eps, Tensor* output) override {
        if (!input.IsValid() || !scale.IsValid() || !shift.IsValid() || !output) return;
        simd::AdaLN(input.DataAs<float>(), scale.DataAs<float>(), shift.DataAs<float>(), output->DataAs<float>(),
                    input.NumElements(), eps);
    }

    void GroupNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, int num_groups, float eps,
                   Tensor* output) override {
        // Determine Layout
        bool is_nhwc = false;
        if (input.layout == TensorLayout::NHWC) {
            is_nhwc = true;
        } else if (input.layout == TensorLayout::UNKNOWN) {
            // Heuristic based on strides:
            // Check if the Channel dimension (index 1 in NCHW) has a stride of 1,
            // indicating an NHWC memory layout (N, H, W, C packed).
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

        if (is_nhwc) {
            GroupNormNHWC(input, gamma, beta, num_groups, eps, output, B, C, H, W);
        } else {
            GroupNormNCHW(input, gamma, beta, num_groups, eps, output, B, C, H, W);
        }
    }

private:
    void GroupNormNCHW(const Tensor& input, const Tensor& gamma, const Tensor& beta, int num_groups, float eps,
                       Tensor* output, int64_t B, int64_t C, int64_t H, int64_t W) {
        const float* gamma_data = gamma.IsValid() ? gamma.DataAs<float>() : nullptr;
        const float* beta_data = beta.IsValid() ? beta.DataAs<float>() : nullptr;
        hwy_kernels::GroupNormNCHW_Hwy(input.DataAs<float>(), gamma_data, beta_data, output->DataAs<float>(), B, C, H,
                                       W, num_groups, eps);
    }

    void GroupNormNHWC(const Tensor& input, const Tensor& gamma, const Tensor& beta, int num_groups, float eps,
                       Tensor* output, int64_t B, int64_t C, int64_t H, int64_t W) {
        const float* gamma_data = gamma.IsValid() ? gamma.DataAs<float>() : nullptr;
        const float* beta_data = beta.IsValid() ? beta.DataAs<float>() : nullptr;
        hwy_kernels::GroupNormNHWC_Hwy(input.DataAs<float>(), gamma_data, beta_data, output->DataAs<float>(), B, C, H,
                                       W, num_groups, eps);
    }
};

DENSECORE_REGISTER_OP(CpuGroupNormOp, OpType::GroupNorm, DeviceType::CPU);

}  // namespace
}  // namespace densecore
