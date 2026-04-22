/**
 * @file normalization_ops.cpp
 * @brief CPU implementations for core normalization ops (Add, RMSNorm, AddRMSNorm, LayerNorm)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/typed_tensor.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

#include <cmath>

namespace densecore {
namespace kernels {
namespace {

class CpuAddOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* /*params*/) override {
        if (inputs.size() < 2 || outputs.empty()) return;
        const Tensor* a = inputs[0];
        const Tensor* b = inputs[1];
        Tensor* out = outputs[0];
        if (!a || !b || !out || !a->IsValid() || !b->IsValid() || !out->IsValid()) return;
        if (a->dtype != DType::F32 || b->dtype != DType::F32 || out->dtype != DType::F32) return;

        TypedTensor<float> a_t(*a);
        TypedTensor<float> b_t(*b);
        TypedTensor<float> out_t(*out);
        const size_t n = static_cast<size_t>(out_t.NumElements());
        hwy_kernels::Add_Hwy(out_t.Data(), a_t.Data(), b_t.Data(), n);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

class CpuRMSNormOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;
        const Tensor* input = inputs[0];
        const Tensor* weight = inputs[1];
        Tensor* output = outputs[0];
        if (!input || !weight || !output || !input->IsValid() || !weight->IsValid() || !output->IsValid()) return;
        if (input->dtype != DType::F32 || weight->dtype != DType::F32 || output->dtype != DType::F32) return;

        float eps = 1e-5f;
        if (auto* p = static_cast<const RMSNormParams*>(params)) {
            eps = p->eps;
        }

        const int64_t hidden_dim = weight->shape[0];
        if (hidden_dim <= 0) return;

        const int64_t n_elements = input->NumElements();
        const int64_t n_tokens = n_elements / hidden_dim;
        TypedTensor<float> x_t(*input);
        TypedTensor<float> w_t(*weight);
        TypedTensor<float> out_t(*output);
        const float* x = x_t.Data();
        const float* w = w_t.Data();
        float* out = out_t.Data();

        for (int64_t t = 0; t < n_tokens; ++t) {
            const float* x_ptr = x + t * hidden_dim;
            float* out_ptr = out + t * hidden_dim;
            hwy_kernels::RMSNorm_Hwy(x_ptr, w, out_ptr, hidden_dim, eps);
        }
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

class CpuAddRMSNormOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;
        const Tensor* input = inputs[0];
        const Tensor* residual = inputs[1];
        const Tensor* weight = inputs[2];
        Tensor* output = outputs[0];
        if (!input || !residual || !weight || !output) return;
        if (!input->IsValid() || !residual->IsValid() || !weight->IsValid() || !output->IsValid()) return;
        if (input->dtype != DType::F32 || residual->dtype != DType::F32 || weight->dtype != DType::F32 ||
            output->dtype != DType::F32) {
            return;
        }

        float eps = 1e-5f;
        if (auto* p = static_cast<const RMSNormParams*>(params)) {
            eps = p->eps;
        }

        const int64_t hidden_dim = weight->shape[0];
        if (hidden_dim <= 0) return;

        const int64_t n_elements = input->NumElements();
        const int64_t n_tokens = n_elements / hidden_dim;
        TypedTensor<float> x_t(*input);
        TypedTensor<float> res_t(*residual);
        TypedTensor<float> w_t(*weight);
        TypedTensor<float> out_t(*output);
        const float* x = x_t.Data();
        const float* res = res_t.Data();
        const float* w = w_t.Data();
        float* out = out_t.Data();

        for (int64_t t = 0; t < n_tokens; ++t) {
            const float* x_ptr = x + t * hidden_dim;
            const float* res_ptr = res + t * hidden_dim;
            float* out_ptr = out + t * hidden_dim;
            hwy_kernels::AddRMSNorm_Hwy(out_ptr, x_ptr, res_ptr, w, static_cast<size_t>(hidden_dim), eps);
        }
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

class CpuLayerNormOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;
        const Tensor* input = inputs[0];
        const Tensor* gamma = inputs[1];
        const Tensor* beta = inputs[2];
        Tensor* output = outputs[0];
        if (!input || !gamma || !beta || !output) return;
        if (!input->IsValid() || !gamma->IsValid() || !beta->IsValid() || !output->IsValid()) return;
        if (input->dtype != DType::F32 || gamma->dtype != DType::F32 || beta->dtype != DType::F32 ||
            output->dtype != DType::F32) {
            return;
        }

        float eps = 1e-5f;
        if (auto* p = static_cast<const LayerNormParams*>(params)) {
            eps = p->eps;
        }

        const int64_t hidden_dim = gamma->shape[0];
        if (hidden_dim <= 0) return;

        const int64_t n_elements = input->NumElements();
        const int64_t n_tokens = n_elements / hidden_dim;
        TypedTensor<float> x_t(*input);
        TypedTensor<float> g_t(*gamma);
        TypedTensor<float> b_t(*beta);
        TypedTensor<float> out_t(*output);
        const float* x = x_t.Data();
        const float* g = g_t.Data();
        const float* b = b_t.Data();
        float* out = out_t.Data();

        for (int64_t t = 0; t < n_tokens; ++t) {
            const float* x_ptr = x + t * hidden_dim;
            float* out_ptr = out + t * hidden_dim;
            hwy_kernels::LayerNorm_Hwy(x_ptr, g, b, out_ptr, hidden_dim, eps);
        }
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

DENSECORE_REGISTER_OP(CpuAddOp, OpType::Add, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuRMSNormOp, OpType::RMSNorm, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuAddRMSNormOp, OpType::AddRMSNorm, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuLayerNormOp, OpType::LayerNorm, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
