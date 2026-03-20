#ifndef DENSECORE_WIDTH_PRUNER_H
#define DENSECORE_WIDTH_PRUNER_H

#include "pruner.h"

namespace densecore {

// Width pruner: reduces model dimensions (hidden_size, ffn_hidden_size)
// Based on magnitude/L2 norm importance scoring
class WidthPruner : public Pruner {
public:
    explicit WidthPruner(const PruneConfig& config) : Pruner(config) {}

    std::vector<float> ComputeImportanceScores(const TransformerModel& model) override;
    void PruneModel(TransformerModel* model) override;

private:
    // Compute importance scores for each dimension
    std::vector<float> ComputeDimensionImportance(const TransformerModel& model);

    // Select dimensions to keep based on importance
    std::vector<int> SelectDimensionsToKeep(const std::vector<float>& scores, int target_dim);

    // Prune embedding dimension (hidden_size)
    void PruneEmbeddingDimension(TransformerModel* model, const std::vector<int>& dims_to_keep);
};

}  // namespace densecore

#endif  // DENSECORE_WIDTH_PRUNER_H
