#include "attention_pruner.h"

#include <densecore/runtime/tensor_utils.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace densecore {

std::vector<float> AttentionPruner::ComputeImportanceScores(const TransformerModel& model) {
    return ComputeHeadImportance(model);
}

std::vector<float> AttentionPruner::ComputeHeadImportance(const TransformerModel& model) {
    const int n_heads = model.hparams.n_head;
    const int n_heads_kv = model.hparams.n_head_kv;
    const int hidden_dim = model.hparams.n_embd;
    const int head_dim = hidden_dim / n_heads;

    std::vector<float> importance(n_heads, 0.0f);

    std::cout << "[AttentionPruner] Computing head importance for " << n_heads << " heads (head_dim=" << head_dim
              << ")..." << std::endl;

    // Aggregate importance across all layers
    int layer_count = 0;

    for (const auto& layer : model.layers) {
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        // For each head, compute L2 norm of its weights in Wq
        // Wq shape: [hidden_dim, hidden_dim] = [n_heads * head_dim, hidden_dim]
        if (wq && wq->data && wq->type == GGML_TYPE_F32) {
            const float* wq_data = (const float*)wq->data;
            const int64_t out_features = wq->ne[1];  // hidden_dim (output)
            const int64_t in_features = wq->ne[0];   // hidden_dim (input)

            for (int h = 0; h < n_heads; ++h) {
                // Each head's weights are in rows [h*head_dim, (h+1)*head_dim)
                float sum_sq = 0.0f;
                for (int row = h * head_dim; row < (h + 1) * head_dim && row < out_features; ++row) {
                    for (int64_t col = 0; col < in_features; ++col) {
                        const float val = wq_data[row * in_features + col];
                        sum_sq += val * val;
                    }
                }
                importance[h] += std::sqrt(sum_sq);
            }
        }
        // For FP16 or quantized, use heuristic
        else if (wq) {
            for (int h = 0; h < n_heads; ++h) {
                importance[h] += 1.0f;
            }
        }

        // Also consider Wk and Wv (but they may have different shapes for GQA)
        // For GQA: Wk/Wv have n_heads_kv heads
        if (wk && wk->data && wk->type == GGML_TYPE_F32) {
            const float* wk_data = (const float*)wk->data;
            const int groups_per_head = n_heads / n_heads_kv;
            const int64_t in_features = wk->ne[0];
            const int kv_head_dim = hidden_dim / n_heads;  // Same head_dim

            for (int kv_h = 0; kv_h < n_heads_kv; ++kv_h) {
                float sum_sq = 0.0f;
                for (int row = kv_h * kv_head_dim; row < (kv_h + 1) * kv_head_dim && row < (int64_t)wk->ne[1]; ++row) {
                    for (int64_t col = 0; col < in_features; ++col) {
                        const float val = wk_data[row * in_features + col];
                        sum_sq += val * val;
                    }
                }
                // Distribute to corresponding Q heads
                for (int g = 0; g < groups_per_head; ++g) {
                    importance[kv_h * groups_per_head + g] += std::sqrt(sum_sq) / groups_per_head;
                }
            }
        }

        layer_count++;
    }

    // Normalize by layer count
    for (auto& score : importance) {
        score /= std::max(layer_count, 1);
    }

    // Print top heads
    std::cout << "[AttentionPruner] Head importance scores (showing top 5):" << std::endl;
    std::vector<int> sorted_indices(n_heads);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return importance[a] > importance[b]; });

    for (int i = 0; i < std::min(5, n_heads); ++i) {
        int idx = sorted_indices[i];
        std::cout << "  Head " << idx << ": " << importance[idx] << std::endl;
    }

    return importance;
}

std::vector<int> AttentionPruner::SelectHeadsToKeep(const std::vector<float>& scores, int target_n_heads) {
    const int original_n_heads = scores.size();

    if (target_n_heads >= original_n_heads) {
        std::cout << "[AttentionPruner] Target >= original, keeping all heads" << std::endl;
        std::vector<int> all_heads(original_n_heads);
        std::iota(all_heads.begin(), all_heads.end(), 0);
        return all_heads;
    }

    // Sort heads by importance (descending)
    std::vector<int> indices(original_n_heads);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Keep top target_n_heads
    std::vector<int> to_keep(indices.begin(), indices.begin() + target_n_heads);

    // Sort by original index for consistent ordering
    std::sort(to_keep.begin(), to_keep.end());

    std::cout << "[AttentionPruner] Keeping " << target_n_heads << "/" << original_n_heads << " heads ("
              << (100.0f * target_n_heads / original_n_heads) << "%)" << std::endl;

    return to_keep;
}

// Helper: Convert head indices to dimension indices
// Each head covers [head_idx * head_dim, (head_idx + 1) * head_dim) dimensions
static std::vector<int> HeadsToDimensions(const std::vector<int>& heads, int head_dim) {
    std::vector<int> dims;
    dims.reserve(heads.size() * head_dim);
    for (int head : heads) {
        for (int d = 0; d < head_dim; ++d) {
            dims.push_back(head * head_dim + d);
        }
    }
    return dims;
}

void AttentionPruner::PruneAttentionHeads(TransformerModel* model, const std::vector<int>& heads_to_keep) {
    const int n_heads = model->hparams.n_head;
    const int n_heads_kv = model->hparams.n_head_kv;
    const int hidden_dim = model->hparams.n_embd;
    const int head_dim = hidden_dim / n_heads;
    const int new_n_heads = heads_to_keep.size();

    std::cout << "[AttentionPruner] Pruning attention heads..." << std::endl;

    struct ggml_context* ctx = model->ctx_w;
    if (!ctx) {
        std::cerr << "[AttentionPruner] Error: Model context is null" << std::endl;
        return;
    }

    // Convert head indices to dimension indices for Q projections
    std::vector<int> q_dims = HeadsToDimensions(heads_to_keep, head_dim);

    // For GQA: determine which KV heads to keep
    // A KV head is kept if ANY of its corresponding Q heads is kept
    std::vector<int> kv_heads_to_keep;
    if (n_heads_kv > 0 && n_heads_kv < n_heads) {
        const int groups_per_head = n_heads / n_heads_kv;
        std::vector<bool> kv_keep(n_heads_kv, false);
        for (int q_head : heads_to_keep) {
            int kv_head = q_head / groups_per_head;
            kv_keep[kv_head] = true;
        }
        for (int kv_h = 0; kv_h < n_heads_kv; ++kv_h) {
            if (kv_keep[kv_h]) {
                kv_heads_to_keep.push_back(kv_h);
            }
        }
    } else {
        // MHA: KV heads = Q heads
        kv_heads_to_keep = heads_to_keep;
    }

    std::vector<int> kv_dims = HeadsToDimensions(kv_heads_to_keep, head_dim);

    // Helper to slice 2D tensor
    auto slice_2d = [&](ggml_tensor*& tensor, const std::vector<int>& dims, int axis, const char* name) {
        if (!tensor) return;

        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            std::cout << "    Skipping " << name << " (quantized type " << tensor->type << ")" << std::endl;
            return;
        }

        ggml_tensor* sliced = TensorUtils::SliceTensor(ctx, tensor, dims, axis);
        if (sliced) {
            std::cout << "    Sliced " << name << ": [" << tensor->ne[0] << ", " << tensor->ne[1] << "] -> ["
                      << sliced->ne[0] << ", " << sliced->ne[1] << "]" << std::endl;
            tensor = sliced;
        } else {
            std::cerr << "    Failed to slice " << name << std::endl;
        }
    };

    // Helper to slice 1D tensor (biases, norms)
    auto slice_1d = [&](ggml_tensor*& tensor, const std::vector<int>& dims, const char* name) {
        if (!tensor) return;

        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
            std::cout << "    Skipping " << name << " (quantized)" << std::endl;
            return;
        }

        ggml_tensor* sliced = TensorUtils::Slice1DTensor(ctx, tensor, dims);
        if (sliced) {
            std::cout << "    Sliced " << name << ": [" << tensor->ne[0] << "] -> [" << sliced->ne[0] << "]"
                      << std::endl;
            tensor = sliced;
        } else {
            std::cerr << "    Failed to slice " << name << std::endl;
        }
    };

    // Slice each layer's attention tensors
    for (size_t layer_idx = 0; layer_idx < model->layers.size(); ++layer_idx) {
        TransformerLayer& layer = model->layers[layer_idx];
        std::cout << "  Layer " << layer_idx << ":" << std::endl;

        // wq: [hidden_in, n_heads * head_dim] -> prune output (row) dimension
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        slice_2d(wq, q_dims, TensorUtils::AXIS_ROWS, "wq");
        layer.Set(model_keys::kAttnQWeight, wq);

        // wk, wv: [hidden_in, n_heads_kv * head_dim] -> may have different size for
        // GQA
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        auto* wv = layer.Get(model_keys::kAttnVWeight);
        slice_2d(wk, kv_dims, TensorUtils::AXIS_ROWS, "wk");
        slice_2d(wv, kv_dims, TensorUtils::AXIS_ROWS, "wv");
        layer.Set(model_keys::kAttnKWeight, wk);
        layer.Set(model_keys::kAttnVWeight, wv);

        // wo: [n_heads * head_dim, hidden_out] -> prune input (col) dimension
        auto* wo = layer.Get(model_keys::kAttnOWeight);
        slice_2d(wo, q_dims, TensorUtils::AXIS_COLS, "wo");
        layer.Set(model_keys::kAttnOWeight, wo);

        // Biases (if present)
        auto* bq = layer.Get(model_keys::kAttnQBias);
        auto* bk = layer.Get(model_keys::kAttnKBias);
        auto* bv = layer.Get(model_keys::kAttnVBias);
        auto* bo = layer.Get(model_keys::kAttnOBias);
        slice_1d(bq, q_dims, "bq");
        slice_1d(bk, kv_dims, "bk");
        slice_1d(bv, kv_dims, "bv");
        slice_1d(bo, q_dims, "bo");
        layer.Set(model_keys::kAttnQBias, bq);
        layer.Set(model_keys::kAttnKBias, bk);
        layer.Set(model_keys::kAttnVBias, bv);
        layer.Set(model_keys::kAttnOBias, bo);

        // QK norms (per-head, if present)
        auto* attn_q_norm = layer.Get(model_keys::kAttnQNorm);
        auto* attn_k_norm = layer.Get(model_keys::kAttnKNorm);
        slice_1d(attn_q_norm, q_dims, "attn_q_norm");
        slice_1d(attn_k_norm, kv_dims, "attn_k_norm");
        layer.Set(model_keys::kAttnQNorm, attn_q_norm);
        layer.Set(model_keys::kAttnKNorm, attn_k_norm);
    }

    // Update model metadata
    model->hparams.n_head = new_n_heads;

    // GQA: also update n_head_kv
    if (n_heads_kv > 0 && n_heads_kv < n_heads) {
        model->hparams.n_head_kv = kv_heads_to_keep.size();
        std::cout << "  Updated n_head_kv: " << n_heads_kv << " -> " << model->hparams.n_head_kv << std::endl;
    }

    std::cout << "  Updated n_head: " << n_heads << " -> " << new_n_heads << std::endl;
}

void AttentionPruner::PruneModel(TransformerModel* model) {
    if (!model) {
        std::cerr << "[AttentionPruner] Error: null model" << std::endl;
        return;
    }

    // Store original stats
    stats_.original_n_layer = model->layers.size();
    stats_.original_hidden_size = model->hparams.n_embd;
    stats_.original_ffn_size = 0;

    std::cout << "\n[AttentionPruner] Starting attention head pruning..." << std::endl;
    std::cout << "  Original heads: " << model->hparams.n_head << std::endl;
    std::cout << "  Target heads: " << config_.target_n_heads << std::endl;

    // Compute head importance
    auto scores = ComputeHeadImportance(*model);

    // Select heads to keep
    auto heads_to_keep = SelectHeadsToKeep(scores, config_.target_n_heads);

    // Prune selected heads
    PruneAttentionHeads(model, heads_to_keep);

    // Update pruned stats
    stats_.pruned_n_layer = model->layers.size();       // Unchanged
    stats_.pruned_hidden_size = model->hparams.n_embd;  // Unchanged for attention pruning

    std::cout << "\n[AttentionPruner] Pruning complete!" << std::endl;
    std::cout << "  Final heads: " << model->hparams.n_head << std::endl;
}

}  // namespace densecore
