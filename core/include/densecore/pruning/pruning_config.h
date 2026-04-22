#ifndef DENSECORE_PRUNING_CONFIG_H
#define DENSECORE_PRUNING_CONFIG_H

#include <cstdint>
#include <string>

namespace densecore {

// Pruning strategies (inspired by Minitron)
enum class PruneStrategy {
    DEPTH,      // Remove transformer layers
    WIDTH,      // Reduce hidden_size, ffn_hidden_size
    ATTENTION,  // Reduce num_heads
    COMBINED,   // Combination of depth + width pruning
};

// Importance scoring methods
enum class ImportanceMethod {
    MAGNITUDE,   // Sum of absolute weight values (simple, fast)
    ACTIVATION,  // Requires calibration data (like Minitron)
    L2_NORM,     // L2 norm of weights per layer/channel
};

// Pruning configuration
struct PruneConfig {
    // Target architecture after pruning (-1 = no change)
    int target_n_layer = -1;          // Depth pruning: target layer count
    int target_hidden_size = -1;      // Width pruning: target embedding dimension
    int target_ffn_hidden_size = -1;  // Width pruning: target FFN dimension
    int target_n_heads = -1;          // Attention pruning: target head count

    // Pruning strategy
    PruneStrategy strategy = PruneStrategy::DEPTH;

    // Importance scoring
    ImportanceMethod importance_method = ImportanceMethod::MAGNITUDE;

    // Calibration parameters (for activation-based scoring)
    int calib_size = 512;
    std::string calib_dataset = "cnn_dailymail";

    // Helper methods
    std::string GetStrategyName() const {
        switch (strategy) {
        case PruneStrategy::DEPTH: return "depth";
        case PruneStrategy::WIDTH: return "width";
        case PruneStrategy::ATTENTION: return "attention";
        case PruneStrategy::COMBINED: return "combined";
        default: return "unknown";
        }
    }

    std::string GetImportanceMethodName() const {
        switch (importance_method) {
        case ImportanceMethod::MAGNITUDE: return "magnitude";
        case ImportanceMethod::ACTIVATION: return "activation";
        case ImportanceMethod::L2_NORM: return "l2_norm";
        default: return "unknown";
        }
    }

    bool IsValid() const {
        // At least one target should be specified
        return (target_n_layer > 0 || target_hidden_size > 0 || target_ffn_hidden_size > 0 || target_n_heads > 0);
    }
};

// Predefined configs (inspired by Minitron examples)
inline PruneConfig DEPTH_PRUNE_50_CFG() {
    PruneConfig cfg;
    cfg.strategy = PruneStrategy::DEPTH;
    cfg.importance_method = ImportanceMethod::MAGNITUDE;
    // target_n_layer will be computed as 50% of original
    return cfg;
}

inline PruneConfig WIDTH_PRUNE_LLAMA_8B_TO_4B_CFG() {
    PruneConfig cfg;
    cfg.strategy = PruneStrategy::WIDTH;
    cfg.importance_method = ImportanceMethod::L2_NORM;
    cfg.target_hidden_size = 3072;      // 4096 → 3072 (75%)
    cfg.target_ffn_hidden_size = 9216;  // 14336 → 9216 (64%)
    return cfg;
}

inline PruneConfig COMBINED_PRUNE_CFG() {
    PruneConfig cfg;
    cfg.strategy = PruneStrategy::COMBINED;
    cfg.importance_method = ImportanceMethod::MAGNITUDE;
    // Will prune both depth and width based on targets
    return cfg;
}

}  // namespace densecore

#endif  // DENSECORE_PRUNING_CONFIG_H
