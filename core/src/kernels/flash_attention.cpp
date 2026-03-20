/**
 * @file flash_attention.cpp
 * @brief CPU FlashAttention op wrapper for OpRegistry dispatch
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "cpu_backend.h"

namespace densecore {
namespace kernels {
namespace {

class CpuFlashAttentionOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;
        const Tensor* q = inputs[0];
        const Tensor* k = inputs[1];
        const Tensor* v = inputs[2];
        Tensor* out = outputs[0];
        if (!q || !k || !v || !out) return;
        if (!q->IsValid() || !k->IsValid() || !v->IsValid() || !out->IsValid()) return;

        float scale = 1.0f;
        bool causal = true;
        int n_head_kv = -1;
        if (auto* p = static_cast<const FlashAttentionParams*>(params)) {
            scale = p->scale;
            causal = p->causal;
            n_head_kv = p->n_head_kv;
        }

        GetCpuBackend().FlashAttention(*q, *k, *v, out, scale, causal, n_head_kv);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 8 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 5};
    }
};

DENSECORE_REGISTER_OP(CpuFlashAttentionOp, OpType::FlashAttention, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
