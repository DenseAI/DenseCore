/**
 * @file timestep_embed.cpp
 * @brief CPU fallback for sinusoidal timestep embedding
 */

#include "densecore/kernels/timestep_embed.h"

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace densecore {
namespace kernels {

void TimestepEmbedding(const Tensor& timesteps, int dim, int max_period, Tensor* output) {
    if (!timesteps.IsValid() || !output || output->data == nullptr) {
        return;
    }
    if (dim <= 0 || max_period <= 0) {
        return;
    }
    if (timesteps.dtype != DType::F32 || output->dtype != DType::F32) {
        return;
    }

    if (timesteps.ndim != 1 && !(timesteps.ndim == 2 && timesteps.shape[1] == 1)) {
        return;
    }
    if (output->ndim < 2 || output->shape[1] < dim) {
        return;
    }

    const int half_dim = dim / 2;
    if (half_dim <= 0) {
        return;
    }

    const float log_max_period = std::log(static_cast<float>(max_period));
    const float inv_half_dim = 1.0f / static_cast<float>(half_dim);

    std::vector<float> freqs(static_cast<size_t>(half_dim));
    for (int i = 0; i < half_dim; ++i) {
        freqs[static_cast<size_t>(i)] = std::exp(-log_max_period * (static_cast<float>(i) * inv_half_dim));
    }

    const int64_t batch = std::min<int64_t>(timesteps.shape[0], output->shape[0]);

    const float* t_data = timesteps.DataAs<float>();
    const int64_t t_stride0 = timesteps.stride[0] > 0 ? timesteps.stride[0] : 1;

    float* out_data = output->DataAs<float>();
    const int64_t out_stride0 = output->stride[0] > 0 ? output->stride[0] : dim;
    const int64_t out_stride1 = output->stride[1] > 0 ? output->stride[1] : 1;

    for (int64_t b = 0; b < batch; ++b) {
        const float t = t_data[b * t_stride0];
        float* out_row = out_data + b * out_stride0;

        for (int i = 0; i < half_dim; ++i) {
            const float angle = t * freqs[static_cast<size_t>(i)];
            out_row[(2 * i) * out_stride1] = std::sin(angle);
            out_row[(2 * i + 1) * out_stride1] = std::cos(angle);
        }

        if (dim % 2 != 0) {
            out_row[(dim - 1) * out_stride1] = 0.0f;
        }
    }
}

}  // namespace kernels

namespace {

class CpuTimestepEmbedOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) {
            return;
        }

        int dim = 0;
        int max_period = 10000;

        if (outputs[0] && outputs[0]->ndim >= 2) {
            dim = static_cast<int>(outputs[0]->shape[1]);
        }

        if (params) {
            const auto* p = static_cast<const TimestepEmbedParams*>(params);
            if (p->dim > 0) {
                dim = p->dim;
            }
            if (p->max_period > 0) {
                max_period = p->max_period;
            }
        }

        if (dim <= 0) {
            return;
        }

        kernels::TimestepEmbedding(*inputs[0], dim, max_period, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 0};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::SEQ || layout == TensorLayout::UNKNOWN;
    }
};

DENSECORE_REGISTER_OP(CpuTimestepEmbedOp, OpType::TimestepEmbed, DeviceType::CPU);

}  // namespace
}  // namespace densecore
