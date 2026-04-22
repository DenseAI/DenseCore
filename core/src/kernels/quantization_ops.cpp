/**
 * @file quantization_ops.cpp
 * @brief CPU implementations for quantize/dequantize ops
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/typed_tensor.h"
#include "densecore/quantization/quantized_tensor.h"

namespace densecore {
namespace kernels {
namespace {

class CpuQuantizeOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty()) return;
        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        if (!input || !output || !input->IsValid() || !output->IsValid()) return;
        if (input->dtype != DType::F32) return;

        QuantType target_type = QuantType::INT8;
        if (auto* p = static_cast<const QuantizeParams*>(params)) {
            target_type = p->target_type;
        }

        TypedTensor<float> input_t(*input);
        QuantizedTensorView qview = QuantizedTensorView::FromTensor(*output, target_type, QuantizationParams::None());
        GetCpuBackend().Quantize(input_t.AsUntyped(), &qview, target_type);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = true,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

class CpuDequantizeOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* /*params*/) override {
        if (inputs.empty() || outputs.empty()) return;
        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        if (!input || !output || !input->IsValid() || !output->IsValid()) return;
        if (output->dtype != DType::F32) return;

        QuantType src_type = QuantType::INT8;
        switch (input->dtype) {
        case DType::F16: src_type = QuantType::FP16; break;
        case DType::INT8: src_type = QuantType::INT8; break;
        case DType::INT4: src_type = QuantType::Q4_K; break;
        default: src_type = QuantType::FP32; break;
        }

        QuantizedTensorView qview = QuantizedTensorView::FromTensor(*input, src_type, QuantizationParams::None());
        GetCpuBackend().Dequantize(qview, output);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = true,
                .supports_int4 = true,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

DENSECORE_REGISTER_OP(CpuQuantizeOp, OpType::Quantize, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuDequantizeOp, OpType::Dequantize, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
