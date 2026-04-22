/**
 * @file moe_types.h
 * @brief MoE (Mixture of Experts) data structures
 *
 * Defines structures for MoE routing results and layer configuration.
 * Used by ForwardMoE and MoETopKRoute functions.
 */

#ifndef DENSECORE_MOE_TYPES_H
#define DENSECORE_MOE_TYPES_H

#include <cstdint>
#include <vector>

namespace densecore {
namespace moe {

/**
 * @brief Result of MoE routing (expert selection)
 *
 * Contains the selected expert IDs and their routing weights for each token.
 * Uses a flat layout for efficient batch processing.
 */
struct MoERouteResult {
    std::vector<int> expert_ids;     ///< [batch * top_k] Selected expert indices
    std::vector<float> weights;      ///< [batch * top_k] Routing weights (normalized)
    std::vector<int> token_indices;  ///< [batch * top_k] Token index for each selection
    int batch_size;                  ///< Number of tokens in batch
    int top_k;                       ///< Number of experts per token

    /**
     * @brief Get expert and weight for a specific token and k
     * @param token Token index (0 to batch_size-1)
     * @param k Expert rank (0 to top_k-1)
     */
    int GetExpertId(int token, int k) const { return expert_ids[token * top_k + k]; }

    float GetWeight(int token, int k) const { return weights[token * top_k + k]; }
};

/**
 * @brief Configuration for an MoE layer
 */
struct MoELayerConfig {
    int n_experts;         ///< Total number of experts
    int top_k;             ///< Experts selected per token (typically 2)
    int hidden_dim;        ///< Model hidden dimension
    int intermediate_dim;  ///< FFN intermediate dimension
    bool norm_topk_prob;   ///< Whether to normalize top-k probabilities
};

/**
 * @brief Expert weights for a single expert
 *
 * Standard MoE FFN structure: gate_proj (w1), down_proj (w2), up_proj (w3)
 * Layout: [intermediate_dim, hidden_dim] for TransB operations
 */
struct ExpertFFNWeights {
    const float* w1;  ///< Gate projection [intermediate, hidden]
    const float* w2;  ///< Down projection [hidden, intermediate]
    const float* w3;  ///< Up projection [intermediate, hidden] (for SwiGLU)
    int64_t hidden_dim;
    int64_t intermediate_dim;
};

}  // namespace moe
}  // namespace densecore

#endif  // DENSECORE_MOE_TYPES_H
