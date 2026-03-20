#ifndef DENSECORE_ATTENTION_PRUNER_H
#define DENSECORE_ATTENTION_PRUNER_H

#include "pruner.h"

namespace densecore {

/**
 * @brief Attention head pruner that reduces num_heads while maintaining
 * alignment.
 *
 * Inspired by NVIDIA Model-Optimizer's attention pruning, this pruner:
 * 1. Computes per-head importance scores using L2 norm of Wq/Wk/Wv
 * 2. Selects heads to keep based on importance
 * 3. Slices weight tensors to remove pruned heads
 *
 * The target number of heads should be divisible by n_head_kv for GQA
 * compatibility.
 */
class AttentionPruner : public Pruner {
public:
    explicit AttentionPruner(const PruneConfig& config) : Pruner(config) {}

    std::vector<float> ComputeImportanceScores(const TransformerModel& model) override;
    void PruneModel(TransformerModel* model) override;

private:
    // Compute importance for each attention head across all layers
    std::vector<float> ComputeHeadImportance(const TransformerModel& model);

    // Select which heads to keep based on importance scores
    std::vector<int> SelectHeadsToKeep(const std::vector<float>& scores, int target_n_heads);

    // Prune attention weights to remove selected heads
    void PruneAttentionHeads(TransformerModel* model, const std::vector<int>& heads_to_keep);
};

}  // namespace densecore

#endif  // DENSECORE_ATTENTION_PRUNER_H
