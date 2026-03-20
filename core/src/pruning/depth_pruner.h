#ifndef DENSECORE_DEPTH_PRUNER_H
#define DENSECORE_DEPTH_PRUNER_H

#include "pruner.h"

namespace densecore {

// Depth pruner: removes entire transformer layers
// Uses magnitude-based importance scoring
class DepthPruner : public Pruner {
public:
    explicit DepthPruner(const PruneConfig& config) : Pruner(config) {}

    std::vector<float> ComputeImportanceScores(const TransformerModel& model) override;
    void PruneModel(TransformerModel* model) override;

private:
    // Compute importance score for a single layer
    float ComputeLayerImportance(const TransformerLayer& layer);

    // Select layers to remove based on importance scores
    std::vector<int> SelectLayersToRemove(const std::vector<float>& scores, int target_count);

    // Remove specified layers from model
    void RemoveLayers(TransformerModel* model, const std::vector<int>& layers_to_remove);
};

}  // namespace densecore

#endif  // DENSECORE_DEPTH_PRUNER_H
