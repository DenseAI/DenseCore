#include "pruner.h"

#include <stdexcept>

#include "pruning/attention_pruner.h"
#include "pruning/combined_pruner.h"
#include "pruning/depth_pruner.h"
#include "pruning/width_pruner.h"

namespace densecore {

std::unique_ptr<Pruner> CreatePruner(const PruneConfig& config) {
    if (!config.IsValid()) {
        throw std::runtime_error("Invalid pruning configuration: no targets specified");
    }

    switch (config.strategy) {
    case PruneStrategy::DEPTH:
        if (config.target_n_layer <= 0) {
            throw std::runtime_error("Depth pruning requires target_n_layer > 0");
        }
        return std::make_unique<DepthPruner>(config);

    case PruneStrategy::WIDTH:
        if (config.target_hidden_size <= 0) {
            throw std::runtime_error("Width pruning requires target_hidden_size > 0");
        }
        return std::make_unique<WidthPruner>(config);

    case PruneStrategy::ATTENTION:
        if (config.target_n_heads <= 0) {
            throw std::runtime_error("Attention pruning requires target_n_heads > 0");
        }
        return std::make_unique<AttentionPruner>(config);

    case PruneStrategy::COMBINED:
        // Combined pruning can use any combination of targets
        return std::make_unique<CombinedPruner>(config);

    default: throw std::runtime_error("Unknown pruning strategy");
    }
}

}  // namespace densecore
