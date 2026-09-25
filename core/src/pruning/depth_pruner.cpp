#include "depth_pruner.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace densecore {

std::vector<float> DepthPruner::ComputeImportanceScores(const TransformerModel& model) {
    std::vector<float> scores;
    scores.reserve(model.layers.size());

    std::cout << "[Pruner] Computing importance scores for " << model.layers.size() << " layers..." << std::endl;

    for (size_t i = 0; i < model.layers.size(); ++i) {
        float score = ComputeLayerImportance(model.layers[i]);
        scores.push_back(score);
        std::cout << "  Layer " << i << ": importance = " << score << std::endl;
    }

    return scores;
}

float DepthPruner::ComputeLayerImportance(const TransformerLayer& layer) {
    // Compute importance based on magnitude of layer weights
    // Higher magnitude = more important layer

    float total_magnitude = 0.0f;
    int tensor_count = 0;

    // Helper lambda to compute tensor magnitude
    auto compute_tensor_magnitude = [](const ggml_tensor* tensor) -> float {
        if (!tensor || !tensor->data) return 0.0f;

        const int64_t nelements = ggml_nelements(tensor);
        float sum = 0.0f;

        // Compute sum of absolute values (L1 norm)
        // Note: This is simplified - in practice, we'd need to handle different
        // tensor types
        if (tensor->type == GGML_TYPE_F32) {
            const float* data = (const float*)tensor->data;
            for (int64_t i = 0; i < nelements; ++i) {
                sum += std::abs(data[i]);
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            // For FP16, we'd convert to FP32 first
            // Simplified: just count as important
            sum = nelements * 0.01f;  // Heuristic
        }
        // For quantized types (Q4, Q8), we assume they're important
        else {
            sum = nelements * 0.01f;  // Heuristic
        }

        return sum / nelements;  // Average magnitude
    };

    // Attention weights
    if (auto* wq = layer.Get(model_keys::kAttnQWeight)) {
        total_magnitude += compute_tensor_magnitude(wq);
        tensor_count++;
    }
    if (auto* wk = layer.Get(model_keys::kAttnKWeight)) {
        total_magnitude += compute_tensor_magnitude(wk);
        tensor_count++;
    }
    if (auto* wv = layer.Get(model_keys::kAttnVWeight)) {
        total_magnitude += compute_tensor_magnitude(wv);
        tensor_count++;
    }
    if (auto* wo = layer.Get(model_keys::kAttnOWeight)) {
        total_magnitude += compute_tensor_magnitude(wo);
        tensor_count++;
    }

    // FFN weights
    if (auto* w1 = layer.Get(model_keys::kFfnGate)) {
        total_magnitude += compute_tensor_magnitude(w1);
        tensor_count++;
    }
    if (auto* w2 = layer.Get(model_keys::kFfnDown)) {
        total_magnitude += compute_tensor_magnitude(w2);
        tensor_count++;
    }
    if (auto* w3 = layer.Get(model_keys::kFfnUp)) {
        total_magnitude += compute_tensor_magnitude(w3);
        tensor_count++;
    }

    // Return average magnitude across all tensors in this layer
    return tensor_count > 0 ? total_magnitude / tensor_count : 0.0f;
}

std::vector<int> DepthPruner::SelectLayersToRemove(const std::vector<float>& scores, int target_count) {
    const int original_count = scores.size();
    const int layers_to_remove = original_count - target_count;

    if (layers_to_remove <= 0) {
        std::cout << "[Pruner] No layers to remove (target >= original)" << std::endl;
        return {};
    }

    // Create indices sorted by importance (ascending)
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] < scores[b];  // Ascending: least important first
    });

    // Select least important layers to remove
    std::vector<int> to_remove(indices.begin(), indices.begin() + layers_to_remove);

    // Sort in descending order for safe removal (remove from end first)
    std::sort(to_remove.rbegin(), to_remove.rend());

    std::cout << "[Pruner] Selected " << to_remove.size() << " layers to remove (least important)" << std::endl;
    for (int idx : to_remove) {
        std::cout << "  Removing layer " << idx << " (importance: " << scores[idx] << ")" << std::endl;
    }

    return to_remove;
}

void DepthPruner::RemoveLayers(TransformerModel* model, const std::vector<int>& layers_to_remove) {
    // Remove layers in descending order to avoid index shifting
    for (int idx : layers_to_remove) {
        if (idx >= 0 && idx < (int)model->layers.size()) {
            model->layers.erase(model->layers.begin() + idx);
        }
    }

    // Update model metadata
    model->hparams.n_layer = model->layers.size();

    std::cout << "[Pruner] Updated n_layer: " << model->hparams.n_layer << std::endl;
}

void DepthPruner::PruneModel(TransformerModel* model) {
    if (!model) {
        std::cerr << "[Pruner] Error: null model" << std::endl;
        return;
    }

    // Store original stats
    stats_.original_n_layer = model->layers.size();
    stats_.original_hidden_size = model->hparams.n_embd;
    stats_.original_ffn_size = 0;  // FFN size not tracked in current model

    std::cout << "\n[DepthPruner] Starting depth pruning..." << std::endl;
    std::cout << "  Original layers: " << model->layers.size() << std::endl;
    std::cout << "  Target layers: " << config_.target_n_layer << std::endl;

    // Compute importance scores
    auto scores = ComputeImportanceScores(*model);

    // Select layers to remove
    auto layers_to_remove = SelectLayersToRemove(scores, config_.target_n_layer);

    // Remove selected layers
    RemoveLayers(model, layers_to_remove);

    // Update pruned stats
    stats_.pruned_n_layer = model->layers.size();
    stats_.pruned_hidden_size = model->hparams.n_embd;  // Unchanged for depth pruning
    stats_.pruned_ffn_size = 0;                         // FFN size not tracked

    std::cout << "\n[DepthPruner] Pruning complete!" << std::endl;
    std::cout << "  Final layers: " << stats_.pruned_n_layer << std::endl;
    std::cout << "  Reduction: " << (stats_.GetLayerReduction() * 100) << "%" << std::endl;
}

}  // namespace densecore
