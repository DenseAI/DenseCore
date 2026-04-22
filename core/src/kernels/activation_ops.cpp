/**
 * @file activation_ops.cpp
 * @brief CPU Activation Functions (SiLU, GELU, Softmax) — Highway SIMD
 *
 * All ISA-specific kernels (AVX-512, AVX2, Scalar) have been replaced by
 * Google Highway portable SIMD implementations. Highway auto-selects the
 * best ISA at runtime from a single source, also providing ARM NEON/SVE
 * coverage that was previously missing.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace densecore {
namespace kernels {

// ============================================================================
// Unified Dispatch Functions — delegate to Highway
// ============================================================================

void SiLU(const float* input, float* output, int64_t n) {
    hwy_kernels::SiLU_Hwy(input, output, n);
}

void GELU(const float* input, float* output, int64_t n) {
    hwy_kernels::GELU_Hwy(input, output, n);
}

void Softmax(const float* input, float* output, int64_t n) {
    hwy_kernels::Softmax_Hwy(input, output, n);
}

void SiLUMul(const float* gate, const float* up, float* output, int64_t n) {
    hwy_kernels::SiLUMul_Hwy(gate, up, output, n);
}

// ============================================================================
// OpRegistry Integration
// ============================================================================

namespace {

struct F16Scratch {
    std::vector<float> in;
    std::vector<float> out;
};

inline bool PrepareF16Input(const Tensor& input, std::vector<float>& scratch, const float** in_ptr) {
    if (input.dtype != DType::F16) return false;
    const int64_t n = input.NumElements();
    scratch.resize(n);
    simd::ConvertF16ToF32(scratch.data(), input.DataAs<ggml_fp16_t>(), n);
    *in_ptr = scratch.data();
    return true;
}

inline bool PrepareF16Output(const Tensor& output, std::vector<float>& scratch, float** out_ptr) {
    if (output.dtype != DType::F16) return false;
    const int64_t n = output.NumElements();
    scratch.resize(n);
    *out_ptr = scratch.data();
    return true;
}

class CpuActivationOps : public ActivationOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        // Generic Execute dispatches based on params type
        // For most cases, use the typed methods directly
        (void)inputs;
        (void)outputs;
        (void)params;
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = true,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,  // Unlimited
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    void SiLU(const Tensor& input, Tensor* output) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;

        const int64_t n = input.NumElements();
        if (input.dtype == DType::F32 && output->dtype == DType::F32) {
            kernels::SiLU(input.DataAs<float>(), output->DataAs<float>(), n);
            return;
        }

        const float* in_ptr = nullptr;
        float* out_ptr = nullptr;
        thread_local F16Scratch scratch;

        if (input.dtype == DType::F16) {
            PrepareF16Input(input, scratch.in, &in_ptr);
        } else if (input.dtype == DType::F32) {
            in_ptr = input.DataAs<float>();
        } else {
            return;
        }

        if (output->dtype == DType::F32) {
            out_ptr = output->DataAs<float>();
        } else if (output->dtype == DType::F16) {
            PrepareF16Output(*output, scratch.out, &out_ptr);
        } else {
            return;
        }

        kernels::SiLU(in_ptr, out_ptr, n);

        if (output->dtype == DType::F16) {
            simd::ConvertF32ToF16(output->DataAs<ggml_fp16_t>(), scratch.out.data(), n);
        }
    }

    void GELU(const Tensor& input, Tensor* output) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;

        const int64_t n = input.NumElements();
        if (input.dtype == DType::F32 && output->dtype == DType::F32) {
            kernels::GELU(input.DataAs<float>(), output->DataAs<float>(), n);
            return;
        }

        const float* in_ptr = nullptr;
        float* out_ptr = nullptr;
        thread_local F16Scratch scratch;

        if (input.dtype == DType::F16) {
            PrepareF16Input(input, scratch.in, &in_ptr);
        } else if (input.dtype == DType::F32) {
            in_ptr = input.DataAs<float>();
        } else {
            return;
        }

        if (output->dtype == DType::F32) {
            out_ptr = output->DataAs<float>();
        } else if (output->dtype == DType::F16) {
            PrepareF16Output(*output, scratch.out, &out_ptr);
        } else {
            return;
        }

        kernels::GELU(in_ptr, out_ptr, n);

        if (output->dtype == DType::F16) {
            simd::ConvertF32ToF16(output->DataAs<ggml_fp16_t>(), scratch.out.data(), n);
        }
    }

    void Softmax(const Tensor& input, Tensor* output) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;

        // Softmax along last dimension
        const int64_t last_dim = input.shape[input.ndim - 1];
        int64_t batch_size = 1;
        for (int i = 0; i < input.ndim - 1; ++i) {
            batch_size *= input.shape[i];
        }

        const float* in_data = nullptr;
        float* out_data = nullptr;
        thread_local F16Scratch scratch;

        if (input.dtype == DType::F16) {
            PrepareF16Input(input, scratch.in, &in_data);
        } else if (input.dtype == DType::F32) {
            in_data = input.DataAs<float>();
        } else {
            return;
        }

        if (output->dtype == DType::F32) {
            out_data = output->DataAs<float>();
        } else if (output->dtype == DType::F16) {
            PrepareF16Output(*output, scratch.out, &out_data);
        } else {
            return;
        }

        for (int64_t b = 0; b < batch_size; ++b) {
            kernels::Softmax(in_data + b * last_dim, out_data + b * last_dim, last_dim);
        }

        if (output->dtype == DType::F16) {
            simd::ConvertF32ToF16(output->DataAs<ggml_fp16_t>(), scratch.out.data(), input.NumElements());
        }
    }

    void ReLU(const Tensor& input, Tensor* output) override {
        if (!input.IsValid() || !output || !output->IsValid()) return;

        const int64_t n = input.NumElements();
        const float* in = nullptr;
        float* out = nullptr;
        thread_local F16Scratch scratch;

        if (input.dtype == DType::F16) {
            PrepareF16Input(input, scratch.in, &in);
        } else if (input.dtype == DType::F32) {
            in = input.DataAs<float>();
        } else {
            return;
        }

        if (output->dtype == DType::F32) {
            out = output->DataAs<float>();
        } else if (output->dtype == DType::F16) {
            PrepareF16Output(*output, scratch.out, &out);
        } else {
            return;
        }

        // Simple ReLU - could be vectorized but rarely a bottleneck
        for (int64_t i = 0; i < n; ++i) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }

        if (output->dtype == DType::F16) {
            simd::ConvertF32ToF16(output->DataAs<ggml_fp16_t>(), scratch.out.data(), n);
        }
    }

    void SiLUMul(const Tensor& gate, const Tensor& up, Tensor* output) override {
        if (!gate.IsValid() || !up.IsValid() || !output || !output->IsValid()) return;

        const int64_t n = gate.NumElements();
        kernels::SiLUMul(gate.DataAs<float>(), up.DataAs<float>(), output->DataAs<float>(), n);
    }
};

// Register for SiLU, GELU, Softmax OpTypes
DENSECORE_REGISTER_OP(CpuActivationOps, OpType::SiLU, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuActivationOps, OpType::GELU, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuActivationOps, OpType::Softmax, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore
