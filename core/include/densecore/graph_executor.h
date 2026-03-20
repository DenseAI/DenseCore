/**
 * @file graph_executor.h
 * @brief GGML-Independent Graph Execution Engine
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Design Philosophy
 *
 * Independent engine that executes OperationGraph without ggml dependency.
 * Selects and dispatches best kernel for each operation via OpRegistry.
 *
 * **Gradual Transition Strategy:**
 * 1. Maintain existing ggml-based execution path
 * 2. Add GraphExecutor as a parallel path
 * 3. Gradually remove ggml dependency after stabilization
 *
 * **Usage:**
 * @code
 * GraphExecutor executor;
 * 
 * // Build graph
 * OperationGraph graph;
 * // ... add nodes ...
 * 
 * // Execute
 * executor.Execute(graph, DeviceType::CPU);
 *
 * // Or with phase awareness
 * executor.ExecutePrefill(graph, prompt_tokens);
 * executor.ExecuteDecode(graph, new_token);
 * @endcode
 */

#ifndef DENSECORE_GRAPH_EXECUTOR_H
#define DENSECORE_GRAPH_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "densecore/hal/macros.h"
#include "densecore/hal/operation_graph.h"
#include "hal/op_registry.h"
#include "hal/tensor.h"

namespace densecore {

/**
 * @brief Execution phase for prefill/decode optimization
 */
enum class ExecutionPhase {
    PREFILL,  ///< Full sequence processing (GEMM-heavy)
    DECODE    ///< Single token generation (GEMV-heavy)
};

/**
 * @brief GGML-Independent Graph Execution Engine
 *
 * Dispatches optimal kernel for all OpTypes based on OpRegistry.
 * Can execute graph independently without ggml dependency.
 */
class DENSECORE_API GraphExecutor {
public:
    GraphExecutor() = default;
    ~GraphExecutor() = default;

    // Non-copyable
    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    // =========================================================================
    // Graph Execution
    // =========================================================================

    /**
     * @brief Execute entire graph on specified device
     *
     * Execute all nodes sequentially.
     * Selects optimal kernel from OpRegistry for each node.
     *
     * @param graph Operation graph to execute
     * @param device Target device type
     * @param numa_node NUMA node hint (-1 = auto)
     */
    void Execute(OperationGraph& graph, DeviceType device = DeviceType::CPU, int numa_node = -1);

    /**
     * @brief Execute graph in prefill phase
     *
     * Process entire prompt. Uses GEMM optimized path.
     *
     * @param graph Operation graph
     * @param input Input tensor (prompt tokens)
     */
    void ExecutePrefill(OperationGraph& graph, const Tensor& input);

    /**
     * @brief Execute graph in decode phase
     *
     * Generate single token. Uses GEMV optimized path.
     *
     * @param graph Operation graph
     * @param token Current token
     */
    void ExecuteDecode(OperationGraph& graph, const Tensor& token);

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set execution phase hint
     *
     * @param phase Prefill or Decode
     */
    void SetPhase(ExecutionPhase phase) { current_phase_ = phase; }

    /**
     * @brief Get current execution phase
     */
    ExecutionPhase GetPhase() const { return current_phase_; }

    /**
     * @brief Set admission policy used before execution starts.
     */
    void SetAdmissionPolicy(const OpRegistry::AdmissionPolicy& policy) { admission_policy_ = policy; }

    /**
     * @brief Get current admission policy.
     */
    const OpRegistry::AdmissionPolicy& GetAdmissionPolicy() const { return admission_policy_; }

private:
    /**
     * @brief Dispatch single node via OpRegistry
     */
    void DispatchNode(const GraphNode& node, std::vector<Tensor>& tensors, DeviceType device);

    /**
     * @brief Generic OpRegistry-based dispatch
     */
    void DispatchViaRegistry(OpType op, DeviceType device, const std::vector<Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs, const OpParams& params);

    /**
     * @brief Fallback execution for ops without OpRegistry implementation
     */
    void ExecuteFallback(const GraphNode& node, std::vector<Tensor>& tensors);

    /**
     * @brief Ensure output tensors have concrete shape/type/data before dispatch.
     */
    void PrepareNodeOutputs(const GraphNode& node, std::vector<Tensor>& tensors);

    /**
     * @brief Infer output metadata from op + input tensors when builder omitted it.
     */
    void InferOutputTensorMeta(OpType op, const std::vector<Tensor*>& inputs, const OpParams& params, Tensor* output);

    /**
     * @brief Allocate backing storage for tensor data and keep ownership for executor lifetime.
     */
    void EnsureTensorData(Tensor* tensor);

    ExecutionPhase current_phase_ = ExecutionPhase::PREFILL;
    OpRegistry::AdmissionPolicy admission_policy_{};
    std::vector<std::unique_ptr<uint8_t[]>> owned_buffers_;
    std::vector<size_t> owned_buffer_sizes_;
};

}  // namespace densecore

#endif  // DENSECORE_GRAPH_EXECUTOR_H
