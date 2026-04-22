#ifndef DENSECORE_PRUNER_H
#define DENSECORE_PRUNER_H

#include <memory>
#include <vector>

#include "densecore/models/model_types.h"
#include "densecore/pruning/pruning_config.h"

namespace densecore {

// Abstract base pruner class
class Pruner {
public:
    explicit Pruner(const PruneConfig& config) : config_(config) {}
    virtual ~Pruner() = default;

    // Compute importance scores for each layer/dimension
    // Returns vector of scores (one per layer for depth pruning)
    virtual std::vector<float> ComputeImportanceScores(const TransformerModel& model) = 0;

    // Prune model in-place (modify GGUF tensors and metadata)
    virtual void PruneModel(TransformerModel* model) = 0;

    // Get pruning statistics
    struct PruneStats {
        int original_n_layer = 0;
        int pruned_n_layer = 0;
        int original_hidden_size = 0;
        int pruned_hidden_size = 0;
        int original_ffn_size = 0;
        int pruned_ffn_size = 0;

        float GetLayerReduction() const {
            return original_n_layer > 0 ? 1.0f - (float)pruned_n_layer / original_n_layer : 0.0f;
        }

        float GetWidthReduction() const {
            return original_hidden_size > 0 ? 1.0f - (float)pruned_hidden_size / original_hidden_size : 0.0f;
        }
    };

    PruneStats GetStats() const { return stats_; }

protected:
    PruneConfig config_;
    PruneStats stats_;
};

// Factory function to create pruner based on config
std::unique_ptr<Pruner> CreatePruner(const PruneConfig& config);

}  // namespace densecore

#endif  // DENSECORE_PRUNER_H
