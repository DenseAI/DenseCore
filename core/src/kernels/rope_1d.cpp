/**
 * @file rope_1d.cpp
 * @brief CPU RoPE (1D) implementation for OpRegistry dispatch
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

#include "simd_ops.h"

namespace densecore {
namespace {

class CpuRoPEOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) {
            return;
        }

        const Tensor& input = *inputs[0];
        const Tensor& cos_sin = *inputs[1];
        const Tensor& positions_tensor = *inputs[2];
        Tensor* output = outputs[0];

        if (!input.IsValid() || !cos_sin.IsValid() || !positions_tensor.IsValid() || !output || !output->IsValid()) {
            return;
        }

        if (input.dtype != DType::F32 || cos_sin.dtype != DType::F32 || output->dtype != DType::F32) {
            return;
        }

        int rope_dim = -1;
        if (auto* p = reinterpret_cast<const RoPEParams*>(params)) {
            rope_dim = p->rope_dim;
        }

        const int* positions = positions_tensor.DataAs<int>();

        int n_tokens = 0;
        int head_dim = 0;
        int n_heads = 1;

        if (input.ndim == 2) {
            n_tokens = static_cast<int>(input.shape[0]);
            head_dim = static_cast<int>(input.shape[1]);
            n_heads = 1;
        } else if (input.ndim == 3) {
            n_heads = static_cast<int>(input.shape[0]);
            n_tokens = static_cast<int>(input.shape[1]);
            head_dim = static_cast<int>(input.shape[2]);
        } else {
            return;
        }

        if (rope_dim < 0 || rope_dim > head_dim) {
            rope_dim = head_dim;
        }

        const float* in = input.DataAs<float>();
        float* out = output->DataAs<float>();
        const float* cs = cos_sin.DataAs<float>();

        const int max_seq_len = static_cast<int>(cos_sin.shape[0]);
        const int64_t head_stride = static_cast<int64_t>(n_tokens) * head_dim;
        for (int h = 0; h < n_heads; ++h) {
            const float* in_h = in + h * head_stride;
            float* out_h = out + h * head_stride;
            simd::ApplyRoPE(out_h, in_h, cs, positions, n_tokens, head_dim, rope_dim, max_seq_len);
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
                .priority = 10};
    }
};

DENSECORE_REGISTER_OP(CpuRoPEOp, OpType::RoPE, DeviceType::CPU);

}  // namespace
}  // namespace densecore
