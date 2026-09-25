#ifndef DENSECORE_MOE_ROUTING_H
#define DENSECORE_MOE_ROUTING_H

#include "densecore/hal/macros.h"
#include "densecore/hal/tensor.h"
#include "densecore/moe/moe_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace densecore {

class ThreadPool;

namespace moe {

/**
 * @brief Workspace for MoE routing (pre-allocated)
 *
 * All pointers are non-owning and point into a caller-provided buffer.
 * Use InitMoERoutingWorkspace() to bind them.
 */
struct DENSECORE_ALIGN(64) MoERoutingWorkspace {
    void* buffer = nullptr;
    size_t size_bytes = 0;

    int max_tokens = 0;
    int max_experts = 0;
    int max_top_k = 0;

    float* probs = nullptr;               ///< [max_experts]
    float* topk_scores = nullptr;         ///< [max_top_k]
    int* topk_indices = nullptr;          ///< [max_top_k]
    float* logits = nullptr;              ///< [max_tokens * max_experts]
    int* expert_counts = nullptr;         ///< [max_experts]
    int* expert_offsets = nullptr;        ///< [max_experts + 1]
    int* expert_cursor = nullptr;         ///< [max_experts]
    int* assignment_tokens = nullptr;     ///< [max_tokens * max_top_k]
    float* assignment_weights = nullptr;  ///< [max_tokens * max_top_k]
};

/**
 * @brief Expert-grouped permutation map (non-owning view)
 */
struct MoEReorderMapView {
    int* expert_offsets = nullptr;  ///< [num_experts + 1] prefix sums
    int* token_indices = nullptr;   ///< [total_assignments] packed token indices
    float* weights = nullptr;       ///< [total_assignments] packed routing weights
    int num_experts = 0;
    int total_assignments = 0;
    int max_expert_batch = 0;
};

/**
 * @brief Expert-grouped permutation map (allocating convenience type)
 */
struct MoEReorderMap {
    std::vector<int> expert_offsets;  ///< [num_experts + 1] prefix sums
    std::vector<int> token_indices;   ///< [total_assignments] packed token indices
    std::vector<float> weights;       ///< [total_assignments] packed routing weights
    int num_experts = 0;
    int total_assignments = 0;
    int max_expert_batch = 0;
};

/**
 * @brief Compute required workspace size (bytes)
 */
size_t GetMoERoutingWorkspaceSize(int max_tokens, int max_experts, int max_top_k);

/**
 * @brief Initialize workspace from pre-allocated buffer
 */
bool InitMoERoutingWorkspace(MoERoutingWorkspace* ws, void* buffer, size_t bytes, int max_tokens, int max_experts,
                             int max_top_k);

/**
 * @brief Select top-k experts based on gate logits (allocating convenience)
 *
 * Applies Softmax to logits, selects top-k, and normalizes weights.
 * This overload allocates internal buffers and should not be used in hot paths.
 */
MoERouteResult MoETopKRoute(const Tensor& gate_logits, int k);

/**
 * @brief Compute router logits from hidden states and select top-k experts (allocating convenience)
 *
 * Computes: logits = hidden_states @ gate_weights^T, then softmax + top-k.
 * This overload allocates internal buffers and should not be used in hot paths.
 */
MoERouteResult MoETopKRoute(const Tensor& hidden_states, const Tensor& gate_weights, int k,
                            bool normalize_weights = true);

/**
 * @brief Select top-k experts based on router logits (no allocations)
 *
 * @param router_logits [batch, n_experts] logits from router
 * @param batch_size Number of tokens
 * @param n_experts Number of experts
 * @param top_k Experts to select
 * @param normalize_weights Whether to renormalize selected weights
 * @param result Output routing result (pre-sized arrays)
 * @param ws Pre-allocated workspace
 * @return true on success
 */
bool MoETopKRoute(const float* router_logits, int batch_size, int n_experts, int top_k, bool normalize_weights,
                  MoERouteResult* result, MoERoutingWorkspace* ws);

/**
 * @brief Select top-k experts using independent sigmoid routing
 *
 * Applies sigmoid independently to each logit (not softmax), selects top-k
 * by sigmoid score, and uses those scores directly as weights without renormalization.
 * Gemma4 does not use this path; it uses softmax top-k renormalization.
 *
 * @param router_logits [batch_size * n_experts] flat row-major logits
 * @param batch_size Number of tokens
 * @param n_experts Number of experts
 * @param top_k Experts to select per token
 * @param result Output routing result (pre-sized arrays)
 * @param ws Pre-allocated workspace
 * @return true on success
 */
bool MoETopKRouteSigmoid(const float* router_logits, int batch_size, int n_experts, int top_k, MoERouteResult* result,
                         MoERoutingWorkspace* ws);

/**
 * @brief Select top-k experts from gate logits (no allocations)
 */
bool MoETopKRoute(const Tensor& gate_logits, int k, MoERouteResult* result, MoERoutingWorkspace* ws);

/**
 * @brief Compute router logits from hidden states and select top-k experts (no allocations)
 */
bool MoETopKRoute(const Tensor& hidden_states, const Tensor& gate_weights, int k, bool normalize_weights,
                  MoERouteResult* result, MoERoutingWorkspace* ws);

/**
 * @brief Build expert-grouped permutation map from routing result (allocating convenience)
 */
MoEReorderMap BuildMoEReorderMap(const MoERouteResult& routing, int num_experts);

/**
 * @brief Build expert-grouped permutation map from routing result (no allocations)
 */
bool BuildMoEReorderMap(const MoERouteResult& routing, int num_experts, MoEReorderMapView* map,
                        MoERoutingWorkspace* ws);

/**
 * @brief Reorder input tokens into expert-grouped contiguous layout
 */
void ReorderInputs(const float* input, int batch_size, int hidden_dim, const MoEReorderMapView& map,
                   float* packed_output, ThreadPool* pool);

/**
 * @brief Reorder expert outputs back to original token order (weighted sum)
 */
void ReorderOutputs(const float* packed_output, int hidden_dim, const MoEReorderMapView& map, float* output,
                    ThreadPool* pool);

}  // namespace moe
}  // namespace densecore

#endif  // DENSECORE_MOE_ROUTING_H
