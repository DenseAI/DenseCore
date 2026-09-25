/**
 * @file memory_ops.cpp
 * @brief CPU memory ops (Copy)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cstring>

namespace densecore {
namespace kernels {
namespace {

class CpuCopyOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* /*params*/) override {
        if (inputs.empty() || outputs.empty()) return;
        const Tensor* src = inputs[0];
        Tensor* dst = outputs[0];
        if (!src || !dst || !src->IsValid() || !dst->IsValid()) return;
        if (src->SizeBytes() == 0 || dst->SizeBytes() == 0) return;
        std::memcpy(dst->data, src->data, src->SizeBytes());
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

DENSECORE_REGISTER_OP(CpuCopyOp, OpType::Copy, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
