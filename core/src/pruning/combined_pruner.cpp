#include "combined_pruner.h"

#include <iostream>

#include "depth_pruner.h"
#include "width_pruner.h"

namespace densecore {

std::vector<float> CombinedPruner::ComputeImportanceScores(const TransformerModel& model) {
    // For combined pruning, we return layer importance scores
    // Width dimension scores are computed separately during width pruning
    PruneConfig depth_config = config_;
    depth_config.strategy = PruneStrategy::DEPTH;

    DepthPruner depth_pruner(depth_config);
    return depth_pruner.ComputeImportanceScores(model);
}

void CombinedPruner::PruneModel(TransformerModel* model) {
    if (!model) {
        std::cerr << "[CombinedPruner] Error: null model" << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "[CombinedPruner] Starting combined pruning" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Store original stats
    stats_.original_n_layer = model->layers.size();
    stats_.original_hidden_size = model->hparams.n_embd;
    stats_.original_ffn_size = 0;

    // ===== Phase 1: Depth Pruning =====
    if (config_.target_n_layer > 0 && config_.target_n_layer < (int)model->layers.size()) {
        std::cout << "[CombinedPruner] Phase 1: Depth Pruning" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        depth_stats_.layers_before = model->layers.size();

        PruneConfig depth_config;
        depth_config.strategy = PruneStrategy::DEPTH;
        depth_config.target_n_layer = config_.target_n_layer;
        depth_config.importance_method = config_.importance_method;

        DepthPruner depth_pruner(depth_config);
        depth_pruner.PruneModel(model);

        depth_stats_.layers_after = model->layers.size();

        std::cout << "[CombinedPruner] Depth phase complete: " << depth_stats_.layers_before << " -> "
                  << depth_stats_.layers_after << " layers" << std::endl
                  << std::endl;
    } else {
        std::cout << "[CombinedPruner] Skipping depth pruning (no target or target "
                     ">= original)"
                  << std::endl;
        depth_stats_.layers_before = model->layers.size();
        depth_stats_.layers_after = model->layers.size();
    }

    // ===== Phase 2: Width Pruning =====
    if (config_.target_hidden_size > 0 && config_.target_hidden_size < (int)model->hparams.n_embd) {
        std::cout << "[CombinedPruner] Phase 2: Width Pruning" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        width_stats_.hidden_before = model->hparams.n_embd;

        PruneConfig width_config;
        width_config.strategy = PruneStrategy::WIDTH;
        width_config.target_hidden_size = config_.target_hidden_size;
        width_config.target_ffn_hidden_size = config_.target_ffn_hidden_size;
        width_config.importance_method = config_.importance_method;

        WidthPruner width_pruner(width_config);
        width_pruner.PruneModel(model);

        width_stats_.hidden_after = model->hparams.n_embd;

        std::cout << "[CombinedPruner] Width phase complete: " << width_stats_.hidden_before << " -> "
                  << width_stats_.hidden_after << " hidden size" << std::endl
                  << std::endl;
    } else {
        std::cout << "[CombinedPruner] Skipping width pruning (no target or target "
                     ">= original)"
                  << std::endl;
        width_stats_.hidden_before = model->hparams.n_embd;
        width_stats_.hidden_after = model->hparams.n_embd;
    }

    // Update final stats
    stats_.pruned_n_layer = model->layers.size();
    stats_.pruned_hidden_size = model->hparams.n_embd;
    stats_.pruned_ffn_size = 0;

    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "[CombinedPruner] Pruning Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Layers: " << stats_.original_n_layer << " -> " << stats_.pruned_n_layer << " ("
              << (100.0f * stats_.GetLayerReduction()) << "% reduction)" << std::endl;
    std::cout << "  Hidden: " << stats_.original_hidden_size << " -> " << stats_.pruned_hidden_size << " ("
              << (100.0f * stats_.GetWidthReduction()) << "% reduction)" << std::endl;

    // Estimate total compression
    float layer_factor = (float)stats_.pruned_n_layer / stats_.original_n_layer;
    float width_factor = (float)stats_.pruned_hidden_size / stats_.original_hidden_size;
    // For transformer, params roughly scale with layers * hidden^2
    float estimated_compression = layer_factor * width_factor * width_factor;
    std::cout << "  Estimated param reduction: " << (1.0f - estimated_compression) * 100 << "%" << std::endl;
}

}  // namespace densecore
