/**
 * @file moe_ops.cpp
 * @brief MoE Operations for HAL OpRegistry
 *
 * Wraps existing moe_routing.cpp implementations as DenseCoreOp for HAL integration.
 * This enables graph-based MoE execution and vendor plugin support.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include "densecore/hal/transformer_ops.h"
#include "moe/moe_routing.h"
#include "moe/moe_types.h"
#include "simd_ops.h"

#include <cstring>

namespace densecore {

// ============================================================================
// MoE Gating Op (TopK Softmax Routing)
// ============================================================================

class CpuMoEGatingOp : public MoEOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.size() < 2) {
            return;
        }

        const Tensor& hidden_states = *inputs[0];
        const Tensor& gate_weights = *inputs[1];
        Tensor* expert_indices = outputs[0];
        Tensor* expert_weights = outputs[1];

        int top_k = 2;  // default
        if (params) {
            const auto* moe_params = static_cast<const MoEParams*>(params);
            top_k = moe_params->top_k;
        }

        Gating(hidden_states, gate_weights, top_k, expert_indices, expert_weights);
    }

    void Gating(const Tensor& hidden_states, const Tensor& gate_weights, int top_k, Tensor* expert_indices,
                Tensor* expert_weights) override {
        if (hidden_states.dtype != DType::F32 || gate_weights.dtype != DType::F32) {
            return;
        }

        // Use the Tensor-based overload which handles GEMM + TopK internally
        moe::MoERouteResult result = moe::MoETopKRoute(hidden_states, gate_weights, top_k, true);

        // Copy results to output tensors
        int* indices_out = expert_indices->DataAs<int>();
        float* weights_out = expert_weights->DataAs<float>();

        std::memcpy(indices_out, result.expert_ids.data(), result.expert_ids.size() * sizeof(int));
        std::memcpy(weights_out, result.weights.data(), result.weights.size() * sizeof(float));
    }

    void Scatter(const Tensor& /*input*/, const Tensor& /*expert_indices*/, Tensor* /*packed_output*/) override {
        // Implemented in CpuMoEScatterOp
    }

    void Gather(const Tensor& /*packed_output*/, const Tensor& /*expert_indices*/, const Tensor& /*expert_weights*/,
                Tensor* /*output*/) override {
        // Implemented in CpuMoEGatherOp
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 0,
                .priority = 10};
    }
};

// ============================================================================
// MoE Scatter Op (Reorder inputs by expert)
// ============================================================================

class CpuMoEScatterOp : public MoEOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* /*params*/) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return;
        }

        const Tensor& input = *inputs[0];
        const Tensor& expert_indices = *inputs[1];
        Tensor* packed_output = outputs[0];

        Scatter(input, expert_indices, packed_output);
    }

    void Gating(const Tensor& /*hidden_states*/, const Tensor& /*gate_weights*/, int /*top_k*/,
                Tensor* /*expert_indices*/, Tensor* /*expert_weights*/) override {
        // Implemented in CpuMoEGatingOp
    }

    void Scatter(const Tensor& input, const Tensor& expert_indices, Tensor* packed_output) override {
        const int batch_size = static_cast<int>(input.shape[0]);
        const int hidden_dim = static_cast<int>(input.shape[1]);
        const int top_k = static_cast<int>(expert_indices.shape[1]);
        const int total_assignments = batch_size * top_k;

        const float* input_data = input.DataAs<float>();
        const int* indices = expert_indices.DataAs<int>();
        float* output_data = packed_output->DataAs<float>();

        // Simple scatter: copy each token to its assignment positions
        for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < top_k; ++k) {
                const int assignment_idx = b * top_k + k;
                const float* src = input_data + static_cast<size_t>(b) * hidden_dim;
                float* dst = output_data + static_cast<size_t>(assignment_idx) * hidden_dim;
                std::memcpy(dst, src, static_cast<size_t>(hidden_dim) * sizeof(float));
            }
        }
        (void)indices;  // Used for expert-grouped reordering in full impl
    }

    void Gather(const Tensor& /*packed_output*/, const Tensor& /*expert_indices*/, const Tensor& /*expert_weights*/,
                Tensor* /*output*/) override {
        // Implemented in CpuMoEGatherOp
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 0,
                .priority = 10};
    }
};

// ============================================================================
// MoE Gather Op (Weighted sum of expert outputs)
// ============================================================================

class CpuMoEGatherOp : public MoEOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* /*params*/) override {
        if (inputs.size() < 3 || outputs.empty()) {
            return;
        }

        const Tensor& packed_output = *inputs[0];
        const Tensor& expert_indices = *inputs[1];
        const Tensor& expert_weights = *inputs[2];
        Tensor* output = outputs[0];

        Gather(packed_output, expert_indices, expert_weights, output);
    }

    void Gating(const Tensor& /*hidden_states*/, const Tensor& /*gate_weights*/, int /*top_k*/,
                Tensor* /*expert_indices*/, Tensor* /*expert_weights*/) override {
        // Implemented in CpuMoEGatingOp
    }

    void Scatter(const Tensor& /*input*/, const Tensor& /*expert_indices*/, Tensor* /*packed_output*/) override {
        // Implemented in CpuMoEScatterOp
    }

    void Gather(const Tensor& packed_output, const Tensor& expert_indices, const Tensor& expert_weights,
                Tensor* output) override {
        const int batch_size = static_cast<int>(output->shape[0]);
        const int hidden_dim = static_cast<int>(output->shape[1]);
        const int top_k = static_cast<int>(expert_indices.shape[1]);

        const float* packed_data = packed_output.DataAs<float>();
        const int* indices = expert_indices.DataAs<int>();
        const float* weights = expert_weights.DataAs<float>();
        float* output_data = output->DataAs<float>();

        // Initialize output to zero
        std::memset(output_data, 0, static_cast<size_t>(batch_size * hidden_dim) * sizeof(float));

        // Weighted sum of expert outputs
        for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < top_k; ++k) {
                const int assignment_idx = b * top_k + k;
                const float weight = weights[assignment_idx];
                const float* src = packed_data + static_cast<size_t>(assignment_idx) * hidden_dim;
                float* dst = output_data + static_cast<size_t>(b) * hidden_dim;

                for (int d = 0; d < hidden_dim; ++d) {
                    dst[d] += weight * src[d];
                }
            }
        }
        (void)indices;  // Expert IDs not needed for simple gather
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 0,
                .memory_bandwidth_gbps = 0,
                .priority = 10};
    }
};

// ============================================================================
// Static Registration
// ============================================================================

DENSECORE_REGISTER_OP(CpuMoEGatingOp, OpType::MoEGating, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuMoEScatterOp, OpType::MoEScatter, DeviceType::CPU);
DENSECORE_REGISTER_OP(CpuMoEGatherOp, OpType::MoEGather, DeviceType::CPU);

}  // namespace densecore
