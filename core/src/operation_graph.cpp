/**
 * @file operation_graph.cpp
 * @brief Implementation of operation graph execution for CPU backend
 *
 * This file provides:
 * 1. ExecuteGraph() - CPU execution of operation graphs
 * 2. Fusion passes - RMSNorm+Linear, QKV projection fusion
 * 3. Memory optimization - Buffer reuse analysis
 */

#include "densecore/hal/operation_graph.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "cpu_backend.h"
#include "densecore/utils/logging.h"
#include "simd_ops.h"

namespace densecore {

// ============================================================================
// Graph Executor for CPU Backend
// ============================================================================

/**
 * @brief Execute a single graph node on CPU
 *
 * Dispatches to the appropriate SIMD kernel based on operation type.
 * Uses the optimized kernels from simd_ops.h.
 */
void ExecuteNode(const GraphNode& node, std::vector<Tensor>& tensors) {
    CpuBackend& backend = GetCpuBackend();

    switch (node.op) {
    case OpType::Add: {
        if (node.inputs.size() >= 2 && node.outputs.size() >= 1) {
            const Tensor& A = tensors[node.inputs[0]];
            const Tensor& B = tensors[node.inputs[1]];
            Tensor& C = tensors[node.outputs[0]];

            const size_t n = C.NumElements();
            simd::AddF32(static_cast<float*>(C.data), static_cast<const float*>(A.data),
                         static_cast<const float*>(B.data), n);
        }
        break;
    }

    case OpType::MatMul: {
        // Standard matrix multiplication: C = A @ B
        const Tensor& A = tensors[node.inputs[0]];
        const Tensor& B = tensors[node.inputs[1]];
        Tensor& C = tensors[node.outputs[0]];
        backend.MatMul(A, B, &C);
        break;
    }

    case OpType::RMSNorm: {
        const auto& params = std::get<RMSNormParams>(node.params);
        const Tensor& input = tensors[node.inputs[0]];
        const Tensor& weight = tensors[node.inputs[1]];
        Tensor& output = tensors[node.outputs[0]];
        backend.RMSNorm(input, weight, &output, params.eps);
        break;
    }

    case OpType::AddRMSNorm: {
        // Fused: output = RMSNorm(input + residual)
        const auto& params = std::get<RMSNormParams>(node.params);
        const Tensor& input = tensors[node.inputs[0]];
        const Tensor& residual = tensors[node.inputs[1]];
        const Tensor& weight = tensors[node.inputs[2]];
        Tensor& output = tensors[node.outputs[0]];
        backend.AddRMSNorm(input, residual, weight, &output, params.eps);
        break;
    }

    case OpType::SiLU: {
        const Tensor& input = tensors[node.inputs[0]];
        Tensor& output = tensors[node.outputs[0]];
        const size_t n = input.NumElements();
        const float* src = static_cast<const float*>(input.data);
        float* dst = static_cast<float*>(output.data);

        for (size_t i = 0; i < n; ++i) {
            float x = src[i];
            dst[i] = x / (1.0f + std::exp(-x));
        }
        break;
    }

    case OpType::GELU: {
        const Tensor& input = tensors[node.inputs[0]];
        Tensor& output = tensors[node.outputs[0]];
        const size_t n = input.NumElements();
        const float* src = static_cast<const float*>(input.data);
        float* dst = static_cast<float*>(output.data);

        for (size_t i = 0; i < n; ++i) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
        break;
    }

    case OpType::Softmax: {
        const Tensor& input = tensors[node.inputs[0]];
        Tensor& output = tensors[node.outputs[0]];
        const size_t n = static_cast<size_t>(input.NumElements());
        std::memcpy(output.data, input.data, n * sizeof(float));
        simd::SoftmaxF32(static_cast<float*>(output.data), n);
        break;
    }

    case OpType::Copy: {
        const Tensor& src = tensors[node.inputs[0]];
        Tensor& dst = tensors[node.outputs[0]];
        std::memcpy(dst.data, src.data, src.SizeBytes());
        break;
    }

    default: LOG_WARN("Unhandled op type: ", OpTypeName(node.op)); break;
    }
}

/**
 * @brief Execute an entire operation graph on CPU
 *
 * Iterates through all nodes in topological order and executes them.
 * Assumes the graph has been validated (no cycles, all tensors exist).
 */
void ExecuteGraph(OperationGraph& graph) {
    auto& tensors = graph.MutableTensors();

    for (size_t i = 0; i < graph.NodeCount(); i++) {
        ExecuteNode(graph.GetNode(i), tensors);
    }
}

// ============================================================================
// Graph Optimization Passes
// ============================================================================

/**
 * @brief Fuse Add + RMSNorm into a single AddRMSNorm node
 *
 * Detects and fuses the pattern:
 *   x = input + residual
 *   y = RMSNorm(x)
 *
 * Into:
 *   y = AddRMSNorm(input, residual)  // Single fused kernel
 *
 * Benefits:
 * - Reduces memory bandwidth by not storing intermediate 'x'
 * - Better cache utilization
 *
 * Implementation:
 *   1. Detects Add -> RMSNorm pattern where Add output is exclusively used by RMSNorm
 *   2. Creates AddRMSNorm node with merged inputs (input, residual, rms_weight)
 *   3. Redirects consumers of RMSNorm output to fused node
 *   4. Replaces original nodes via graph.ReplaceNodes()
 *
 * @param graph The operation graph to optimize (mutated in place if fusion applies)
 * @return true if any fusion was applied, false otherwise
 */

bool FuseAddRMSNorm(OperationGraph& graph) {
    if (graph.NodeCount() < 2) {
        return false;
    }

    std::vector<int> tensor_use_count(graph.TensorCount(), 0);
    for (const auto& node : graph.Nodes()) {
        for (size_t input_idx : node.inputs) {
            if (input_idx < tensor_use_count.size()) {
                tensor_use_count[input_idx]++;
            }
        }
    }

    std::vector<GraphNode> new_nodes;
    new_nodes.reserve(graph.NodeCount());

    bool modified = false;
    for (size_t i = 0; i < graph.NodeCount(); ++i) {
        const GraphNode& add_node = graph.GetNode(i);
        if (add_node.op == OpType::Add && i + 1 < graph.NodeCount()) {
            const GraphNode& rms_node = graph.GetNode(i + 1);
            if (rms_node.op == OpType::RMSNorm && add_node.outputs.size() == 1 && rms_node.inputs.size() >= 2 &&
                rms_node.outputs.size() == 1 && rms_node.inputs[0] == add_node.outputs[0]) {
                size_t add_out = add_node.outputs[0];
                bool exclusive_use = add_out < tensor_use_count.size() && tensor_use_count[add_out] == 1;
                if (exclusive_use && add_node.inputs.size() >= 2) {
                    GraphNode fused;
                    fused.op = OpType::AddRMSNorm;
                    fused.inputs = {add_node.inputs[0], add_node.inputs[1], rms_node.inputs[1]};
                    fused.outputs = rms_node.outputs;
                    fused.params = rms_node.params;
                    if (auto* params = std::get_if<RMSNormParams>(&fused.params)) {
                        params->fused_add = true;
                    }
                    fused.name = rms_node.name.empty() ? "AddRMSNorm_fused" : rms_node.name;

                    new_nodes.push_back(std::move(fused));
                    modified = true;
                    ++i;  // Skip RMSNorm node
                    continue;
                }
            }
        }

        new_nodes.push_back(add_node);
    }

    if (modified) {
        graph.ReplaceNodes(std::move(new_nodes));
    }

    return modified;
}

/**
 * @brief Analyze buffer lifetimes for memory reuse
 *
 * Computes liveness intervals for all tensors and identifies
 * opportunities to reuse buffers.
 *
 * @param graph The operation graph to analyze
 * @return Map of tensor index -> reusable tensor index (-1 if no reuse)
 */
std::vector<int> AnalyzeBufferReuse(const OperationGraph& graph) {
    const size_t n_tensors = graph.TensorCount();
    const size_t n_nodes = graph.NodeCount();

    // last_use[tensor_idx] = node index where tensor is last used
    std::vector<size_t> last_use(n_tensors, 0);

    // First pass: find last use of each tensor
    for (size_t node_idx = 0; node_idx < n_nodes; node_idx++) {
        const auto& node = graph.GetNode(node_idx);
        for (size_t input_idx : node.inputs) {
            last_use[input_idx] = node_idx;
        }
    }

    // Second pass: identify reuse opportunities
    std::vector<int> reuse_map(n_tensors, -1);

    for (size_t node_idx = 0; node_idx < n_nodes; node_idx++) {
        const auto& node = graph.GetNode(node_idx);

        // For each output, check if any input's buffer is now free
        for (size_t output_idx : node.outputs) {
            for (size_t input_idx : node.inputs) {
                if (last_use[input_idx] == node_idx) {
                    // Input buffer can be reused for output
                    const auto& input_tensor = graph.GetTensor(input_idx);
                    const auto& output_tensor = graph.GetTensor(output_idx);

                    // Check size compatibility (must compare bytes, not elements, for mixed dtypes)
                    if (input_tensor.SizeBytes() >= output_tensor.SizeBytes()) {
                        reuse_map[output_idx] = static_cast<int>(input_idx);
                        break;
                    }
                }
            }
        }
    }

    return reuse_map;
}

/**
 * @brief Print graph statistics for debugging
 */
void PrintGraphStats(const OperationGraph& graph) {
    LOG_INFO("Nodes: ", graph.NodeCount(), ", Tensors: ", graph.TensorCount(),
             ", Compiled: ", (graph.IsCompiled() ? "yes" : "no"));

    // Count operations by type
    std::vector<int> op_counts(static_cast<int>(OpType::Custom) + 1, 0);
    for (size_t i = 0; i < graph.NodeCount(); i++) {
        op_counts[static_cast<int>(graph.GetNode(i).op)]++;
    }

    std::ostringstream oss;
    oss << "Operation breakdown:";
    for (int i = 0; i <= static_cast<int>(OpType::Custom); i++) {
        if (op_counts[i] > 0) {
            oss << " " << OpTypeName(static_cast<OpType>(i)) << ":" << op_counts[i];
        }
    }
    LOG_INFO(oss.str());
}

}  // namespace densecore
