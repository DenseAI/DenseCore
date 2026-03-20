#ifndef DENSECORE_COMBINED_PRUNER_H
#define DENSECORE_COMBINED_PRUNER_H

#include "pruner.h"

namespace densecore {

/**
 * @brief Combined pruner that applies both depth and width pruning.
 *
 * Inspired by NVIDIA Minitron's combined pruning approach, this pruner:
 * 1. First applies depth pruning (removes entire layers)
 * 2. Then applies width pruning (reduces hidden dimensions)
 *
 * This approach is more efficient than single-type pruning as it
 * allows trading off computation vs accuracy at multiple levels.
 */
class CombinedPruner : public Pruner {
public:
    explicit CombinedPruner(const PruneConfig& config) : Pruner(config) {}

    std::vector<float> ComputeImportanceScores(const TransformerModel& model) override;
    void PruneModel(TransformerModel* model) override;

private:
    // Store intermediate stats from each phase
    struct PhaseStats {
        int layers_before = 0;
        int layers_after = 0;
        int hidden_before = 0;
        int hidden_after = 0;
    };

    PhaseStats depth_stats_;
    PhaseStats width_stats_;
};

}  // namespace densecore

#endif  // DENSECORE_COMBINED_PRUNER_H
