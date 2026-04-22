#include "width_pruner.h"

#include <densecore/runtime/tensor_utils.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace densecore {

std::vector<float> WidthPruner::ComputeImportanceScores(const TransformerModel& model) {
    // For width pruning, we compute importance per dimension
    return ComputeDimensionImportance(model);
}

// NVIDIA Alignment: Ensure dimensions are multiple of 16 for SIMD/Tensor Cores
int AlignToMultiple(int val, int multiple) {
    if (multiple <= 0) return val;
    return (val / multiple) * multiple;
}

std::vector<float> WidthPruner::ComputeDimensionImportance(const TransformerModel& model) {
    const int hidden_dim = model.hparams.n_embd;
    std::vector<float> importance(hidden_dim, 0.0f);

    std::cout << "[WidthPruner] Computing dimension importance for " << hidden_dim << " dimensions..." << std::endl;

    // Aggregate importance across all layers
    // Use L2 norm of weights for each dimension
    int layer_count = 0;

    for (const auto& layer : model.layers) {
        // Process attention weights (wq, wk, wv output dimension)
        auto process_weight = [&](const ggml_tensor* tensor, [[maybe_unused]] const char* name) {
            if (!tensor || !tensor->data) return;

            // For simplicity, compute L2 norm per output dimension
            // Tensor shape typically: [hidden_dim, ...]
            if (tensor->type == GGML_TYPE_F32) {
                const float* data = (const float*)tensor->data;
                const int64_t dim0 = tensor->ne[0];  // Usually hidden_dim
                const int64_t dim1 = tensor->ne[1] > 0 ? tensor->ne[1] : 1;

                for (int i = 0; i < std::min((int64_t)hidden_dim, dim1); ++i) {
                    float sum_sq = 0.0f;
                    for (int64_t j = 0; j < dim0; ++j) {
                        const float val = data[i * dim0 + j];
                        sum_sq += val * val;
                    }
                    importance[i] += std::sqrt(sum_sq);
                }
            }
            // For quantized tensors, use heuristic
            else {
                for (int i = 0; i < hidden_dim; ++i) {
                    importance[i] += 1.0f;  // Assume all dimensions are important
                }
            }
        };

        process_weight(layer.Get(model_keys::kAttnQWeight), "wq");
        process_weight(layer.Get(model_keys::kAttnKWeight), "wk");
        process_weight(layer.Get(model_keys::kAttnVWeight), "wv");
        process_weight(layer.Get(model_keys::kAttnOWeight), "wo");

        layer_count++;
    }

    // Normalize by layer count
    for (auto& score : importance) {
        score /= std::max(layer_count, 1);
    }

    std::cout << "[WidthPruner] Computed importance scores (showing top 10):" << std::endl;
    auto sorted_indices = std::vector<int>(hidden_dim);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return importance[a] > importance[b]; });

    for (int i = 0; i < std::min(10, hidden_dim); ++i) {
        int idx = sorted_indices[i];
        std::cout << "  Dim " << idx << ": " << importance[idx] << std::endl;
    }

    return importance;
}

std::vector<int> WidthPruner::SelectDimensionsToKeep(const std::vector<float>& scores, int target_dim) {
    const int original_dim = scores.size();

    if (target_dim >= original_dim) {
        std::cout << "[WidthPruner] Target >= original, keeping all dimensions" << std::endl;
        std::vector<int> all_dims(original_dim);
        std::iota(all_dims.begin(), all_dims.end(), 0);
        return all_dims;
    }

    // Sort dimensions by importance (descending)
    std::vector<int> indices(original_dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Keep top target_dim dimensions
    std::vector<int> to_keep(indices.begin(), indices.begin() + target_dim);

    // Sort by original index for easier processing
    std::sort(to_keep.begin(), to_keep.end());

    std::cout << "[WidthPruner] Keeping " << target_dim << "/" << original_dim << " dimensions ("
              << (100.0f * target_dim / original_dim) << "%)" << std::endl;

    return to_keep;
}

void WidthPruner::PruneEmbeddingDimension(TransformerModel* model, const std::vector<int>& dims_to_keep) {
    std::cout << "[WidthPruner] Pruning embedding dimension..." << std::endl;

    struct ggml_context* ctx = model->ctx_w;
    if (!ctx) {
        std::cerr << "[WidthPruner] Error: Model context is null" << std::endl;
        return;
    }

    const int new_hidden = dims_to_keep.size();

    // Helper to safely slice a 2D tensor and replace the pointer
    auto slice_2d = [&](ggml_tensor*& tensor, int axis, const char* name) {
        if (!tensor) return;

        // Check tensor type - only F32/F16 supported
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            std::cout << "  Skipping " << name << " (quantized type " << tensor->type << ")" << std::endl;
            return;
        }

        ggml_tensor* sliced = TensorUtils::SliceTensor(ctx, tensor, dims_to_keep, axis);
        if (sliced) {
            std::cout << "  Sliced " << name << ": [" << tensor->ne[0] << ", " << tensor->ne[1] << "] -> ["
                      << sliced->ne[0] << ", " << sliced->ne[1] << "]" << std::endl;
            tensor = sliced;
        } else {
            std::cerr << "  Failed to slice " << name << std::endl;
        }
    };

    // Helper to slice a 1D tensor (norms, biases)
    auto slice_1d = [&](ggml_tensor*& tensor, const char* name) {
        if (!tensor) return;

        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            std::cout << "  Skipping " << name << " (quantized)" << std::endl;
            return;
        }

        ggml_tensor* sliced = TensorUtils::Slice1DTensor(ctx, tensor, dims_to_keep);
        if (sliced) {
            std::cout << "  Sliced " << name << ": [" << tensor->ne[0] << "] -> [" << sliced->ne[0] << "]" << std::endl;
            tensor = sliced;
        } else {
            std::cerr << "  Failed to slice " << name << std::endl;
        }
    };

    // 1. Slice token embeddings: [n_embd, n_vocab] -> prune rows (n_embd)
    slice_2d(model->tok_embeddings, TensorUtils::AXIS_ROWS, "tok_embeddings");

    // 2. Slice output (tied or separate): same shape as tok_embeddings
    if (!model->tied_embeddings && model->output) {
        slice_2d(model->output, TensorUtils::AXIS_ROWS, "output");
    }

    // 3. Slice output_norm: [n_embd] -> 1D slice
    slice_1d(model->output_norm, "output_norm");

    // 4. Slice each layer's tensors
    for (size_t layer_idx = 0; layer_idx < model->layers.size(); ++layer_idx) {
        TransformerLayer& layer = model->layers[layer_idx];
        std::cout << "  Layer " << layer_idx << ":" << std::endl;

        // Attention weights:
        // wq, wk, wv: Output dimension is rows (Y = XW, W shape is [in, out])
        // For Y = X @ W where X is [batch, hidden] and W is [hidden, hidden]:
        //   Pruning output hidden means slicing W rows
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        auto* wv = layer.Get(model_keys::kAttnVWeight);
        slice_2d(wq, TensorUtils::AXIS_ROWS, "wq");
        slice_2d(wk, TensorUtils::AXIS_ROWS, "wk");
        slice_2d(wv, TensorUtils::AXIS_ROWS, "wv");
        layer.Set(model_keys::kAttnQWeight, wq);
        layer.Set(model_keys::kAttnKWeight, wk);
        layer.Set(model_keys::kAttnVWeight, wv);

        // wo: Input dimension is columns (takes concatenated head outputs)
        // wo shape: [head_dim * n_heads (or n_embd), hidden_out]
        // Pruning hidden means slicing columns (input dim)
        auto* wo = layer.Get(model_keys::kAttnOWeight);
        slice_2d(wo, TensorUtils::AXIS_COLS, "wo");
        layer.Set(model_keys::kAttnOWeight, wo);

        // Attention biases (if present)
        auto* bq = layer.Get(model_keys::kAttnQBias);
        auto* bk = layer.Get(model_keys::kAttnKBias);
        auto* bv = layer.Get(model_keys::kAttnVBias);
        auto* bo = layer.Get(model_keys::kAttnOBias);
        slice_1d(bq, "bq");
        slice_1d(bk, "bk");
        slice_1d(bv, "bv");
        slice_1d(bo, "bo");
        layer.Set(model_keys::kAttnQBias, bq);
        layer.Set(model_keys::kAttnKBias, bk);
        layer.Set(model_keys::kAttnVBias, bv);
        layer.Set(model_keys::kAttnOBias, bo);

        // QK norms (if present, e.g., Qwen3)
        auto* attn_q_norm = layer.Get(model_keys::kAttnQNorm);
        auto* attn_k_norm = layer.Get(model_keys::kAttnKNorm);
        slice_1d(attn_q_norm, "attn_q_norm");
        slice_1d(attn_k_norm, "attn_k_norm");
        layer.Set(model_keys::kAttnQNorm, attn_q_norm);
        layer.Set(model_keys::kAttnKNorm, attn_k_norm);

        // Layer norms
        auto* attn_norm = layer.Get(model_keys::kAttnNorm);
        auto* ffn_norm = layer.Get(model_keys::kFfnNorm);
        slice_1d(attn_norm, "attention_norm");
        slice_1d(ffn_norm, "ffn_norm");
        layer.Set(model_keys::kAttnNorm, attn_norm);
        layer.Set(model_keys::kFfnNorm, ffn_norm);

        // FFN weights:
        // w1 (gate), w3 (up): Output dimension is rows
        // Note: FFN intermediate size is separate from hidden, but input uses
        // hidden For now, only prune the hidden dimension interface w1, w3:
        // [hidden_in, ffn_hidden] -> prune columns (input is hidden)
        auto* w1 = layer.Get(model_keys::kFfnGate);
        auto* w3 = layer.Get(model_keys::kFfnUp);
        slice_2d(w1, TensorUtils::AXIS_COLS, "w1");
        slice_2d(w3, TensorUtils::AXIS_COLS, "w3");
        layer.Set(model_keys::kFfnGate, w1);
        layer.Set(model_keys::kFfnUp, w3);

        // w2 (down): [ffn_hidden, hidden_out] -> prune rows (output is hidden)
        auto* w2 = layer.Get(model_keys::kFfnDown);
        slice_2d(w2, TensorUtils::AXIS_ROWS, "w2");
        layer.Set(model_keys::kFfnDown, w2);
    }

    // Update model metadata
    model->hparams.n_embd = new_hidden;

    std::cout << "  Updated n_embd: " << model->hparams.n_embd << std::endl;
}

void WidthPruner::PruneModel(TransformerModel* model) {
    if (!model) {
        std::cerr << "[WidthPruner] Error: null model" << std::endl;
        return;
    }

    // Store original stats
    stats_.original_n_layer = model->layers.size();
    stats_.original_hidden_size = model->hparams.n_embd;
    stats_.original_ffn_size = 0;

    std::cout << "\n[WidthPruner] Starting width pruning..." << std::endl;
    std::cout << "  Original hidden_size: " << model->hparams.n_embd << std::endl;
    std::cout << "  Target hidden_size: " << config_.target_hidden_size << std::endl;

    // Compute dimension importance
    auto scores = ComputeDimensionImportance(*model);

    // Select dimensions to keep

    // NVIDIA Alignment: Enforce 8-byte alignment (or 16) for vectorization
    int aligned_target = AlignToMultiple(config_.target_hidden_size, 8);
    if (aligned_target != config_.target_hidden_size) {
        std::cout << "[WidthPruner] Aligning target hidden size from " << config_.target_hidden_size << " to "
                  << aligned_target << " (multiple of 8)" << std::endl;
    }

    auto dims_to_keep = SelectDimensionsToKeep(scores, aligned_target);

    // Prune dimensions (metadata only for now)
    PruneEmbeddingDimension(model, dims_to_keep);

    // Update pruned stats
    stats_.pruned_n_layer = model->layers.size();  // Unchanged
    stats_.pruned_hidden_size = model->hparams.n_embd;
    stats_.pruned_ffn_size = 0;

    std::cout << "\n[WidthPruner] Pruning complete!" << std::endl;
    std::cout << "  Final hidden_size: " << stats_.pruned_hidden_size << std::endl;
    std::cout << "  Reduction: " << (stats_.GetWidthReduction() * 100) << "%" << std::endl;
}

}  // namespace densecore
