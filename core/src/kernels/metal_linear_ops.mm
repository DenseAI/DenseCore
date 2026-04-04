/**
 * @file metal_linear_ops.mm
 * @brief Metal OpRegistry wrappers for core linear/attention ops
 */

#ifdef __APPLE__

#include <memory>
#include <mutex>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "metal_backend.h"

namespace densecore {
namespace kernels {
namespace {

MetalBackend* GetMetalBackend() {
    static std::once_flag init_flag;
    static std::unique_ptr<MetalBackend> backend;
    std::call_once(init_flag, []() {
        if (MetalBackend::IsAvailable()) {
            backend = std::make_unique<MetalBackend>();
        }
    });
    return backend.get();
}

class MetalMatMulOp : public MatMulOps {
public:
    void Execute(const std::vector<Tensor*>&, const std::vector<Tensor*>&, const void*) override {
        // MatMul handled via MatMul/MatMulTransB below
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::METAL; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 200,
                .priority = 20};
    }

    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override {
        auto* backend = GetMetalBackend();
        if (!backend || !C)
            return;
        backend->MatMul(A, B, C);
    }

    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override {
        auto* backend = GetMetalBackend();
        if (!backend || !C)
            return;
        backend->MatMulTransB(A, B, C);
    }
};

class MetalFlashAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 3 || outputs.empty())
            return;
        auto* backend = GetMetalBackend();
        if (!backend)
            return;

        const Tensor* q = inputs[0];
        const Tensor* k = inputs[1];
        const Tensor* v = inputs[2];
        Tensor* out = outputs[0];
        if (!q || !k || !v || !out)
            return;

        float scale = 1.0f;
        bool causal = false;
        int n_head_kv = -1;
        int q_start_offset = 0;
        int kv_start_offset = 0;
        int sliding_window = -1;
        float logit_softcap = 0.0f;
        uint32_t semantic_flags = 0;
        if (auto* p = static_cast<const FlashAttentionParams*>(params)) {
            scale = p->scale;
            causal = p->causal;
            n_head_kv = p->n_head_kv;
            q_start_offset = p->q_start_offset;
            kv_start_offset = p->kv_start_offset;
            sliding_window = p->sliding_window;
            logit_softcap = p->logit_softcap;
            semantic_flags = p->semantic_flags;
        }

        backend->FlashAttention(*q, *k, *v, out, scale, causal, n_head_kv, sliding_window,
                                logit_softcap, semantic_flags, q_start_offset, kv_start_offset);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::METAL; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 200,
                .priority = 20};
    }
};

class MetalRMSNormOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 2 || outputs.empty())
            return;
        auto* backend = GetMetalBackend();
        if (!backend)
            return;

        const Tensor* input = inputs[0];
        const Tensor* weight = inputs[1];
        Tensor* output = outputs[0];
        if (!input || !weight || !output)
            return;

        float eps = 1e-5f;
        if (auto* p = static_cast<const RMSNormParams*>(params)) {
            eps = p->eps;
        }
        backend->RMSNorm(*input, *weight, output, eps);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::METAL; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 200,
                .priority = 20};
    }
};

class MetalAddRMSNormOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        if (inputs.size() < 3 || outputs.empty())
            return;
        auto* backend = GetMetalBackend();
        if (!backend)
            return;

        const Tensor* input = inputs[0];
        const Tensor* residual = inputs[1];
        const Tensor* weight = inputs[2];
        Tensor* output = outputs[0];
        if (!input || !residual || !weight || !output)
            return;

        float eps = 1e-5f;
        if (auto* p = static_cast<const RMSNormParams*>(params)) {
            eps = p->eps;
        }
        backend->AddRMSNorm(*input, *residual, *weight, output, eps);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::METAL; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 200,
                .priority = 20};
    }
};

DENSECORE_REGISTER_OP(MetalMatMulOp, OpType::MatMul, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalMatMulOp, OpType::MatMulTransB, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalFlashAttentionOp, OpType::FlashAttention, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalRMSNormOp, OpType::RMSNorm, DeviceType::METAL);
DENSECORE_REGISTER_OP(MetalAddRMSNormOp, OpType::AddRMSNorm, DeviceType::METAL);

}  // namespace
}  // namespace kernels
}  // namespace densecore

#endif  // __APPLE__
