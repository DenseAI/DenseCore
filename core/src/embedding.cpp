/**
 * @file embedding.cpp
 * @brief Embedding extraction implementation with SIMD optimization
 */

#include "embedding.h"

#include <algorithm>
#include <cstring>

#include "simd_ops.h"

namespace densecore {

/**
 * Apply pooling strategy to hidden states
 *
 * @param hidden_states Input tensor [seq_len, hidden_dim]
 * @param output Output vector [hidden_dim]
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param strategy Pooling strategy
 * @param mask Optional attention mask [seq_len] (1=valid, 0=pad)
 */
void ApplyPooling(const float* hidden_states, float* output, int seq_len, int hidden_dim, PoolingStrategy strategy,
                  const int* mask = nullptr) {
    switch (strategy) {
    case PoolingStrategy::MEAN:
        if (mask) {
            simd::MeanPoolMasked(hidden_states, mask, output, seq_len, hidden_dim);
        } else {
            simd::MeanPool(hidden_states, output, seq_len, hidden_dim);
        }
        break;

    case PoolingStrategy::CLS: simd::ClsPool(hidden_states, output, hidden_dim); break;

    case PoolingStrategy::LAST: simd::LastPool(hidden_states, output, seq_len, hidden_dim); break;

    case PoolingStrategy::MAX: simd::MaxPool(hidden_states, output, seq_len, hidden_dim); break;

    default: simd::MeanPool(hidden_states, output, seq_len, hidden_dim); break;
    }
}

/**
 * Process embedding with pooling and optional normalization
 *
 * @param hidden_states Input tensor [seq_len, hidden_dim]
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param config Embedding configuration
 * @param mask Optional attention mask
 * @return EmbeddingResult with processed embedding
 */
EmbeddingResult ProcessEmbedding(const float* hidden_states, int seq_len, int hidden_dim, const EmbeddingConfig& config,
                                 const int* mask = nullptr) {
    EmbeddingResult result;
    result.dimension = hidden_dim;
    result.embedding.resize(hidden_dim);

    // Handle truncation
    int actual_len = seq_len;
    if (config.truncate && seq_len > config.max_length) {
        actual_len = config.max_length;
        result.truncated = true;
    }
    result.tokens_used = actual_len;

    // Apply pooling
    ApplyPooling(hidden_states, result.embedding.data(), actual_len, hidden_dim, config.pooling, mask);

    // Apply L2 normalization if requested
    if (config.normalize) {
        simd::NormalizeL2(result.embedding.data(), hidden_dim);
    }

    return result;
}

/**
 * Process batch of embeddings
 *
 * @param hidden_states Input tensor [batch_size, seq_len, hidden_dim]
 * @param batch_size Number of sequences
 * @param seq_len Sequence length (same for all)
 * @param hidden_dim Hidden dimension
 * @param config Embedding configuration
 * @return BatchEmbeddingResult with all embeddings
 */
BatchEmbeddingResult ProcessBatchEmbedding(const float* hidden_states, int batch_size, int seq_len, int hidden_dim,
                                           const EmbeddingConfig& config) {
    BatchEmbeddingResult result;
    result.batch_size = batch_size;
    result.dimension = hidden_dim;
    result.embeddings.resize(batch_size * hidden_dim);
    result.tokens_used.resize(batch_size);

    int stride = seq_len * hidden_dim;

    for (int b = 0; b < batch_size; b++) {
        const float* batch_input = hidden_states + b * stride;
        float* batch_output = result.embeddings.data() + b * hidden_dim;

        // Handle truncation
        int actual_len = std::min(seq_len, config.max_length);
        result.tokens_used[b] = actual_len;

        // Apply pooling
        ApplyPooling(batch_input, batch_output, actual_len, hidden_dim, config.pooling, nullptr);
    }

    // Apply batch L2 normalization if requested
    if (config.normalize) {
        simd::BatchNormalizeL2(result.embeddings.data(), batch_size, hidden_dim);
    }

    return result;
}

/**
 * Compute pairwise cosine similarities
 *
 * @param embeddings Embeddings [n, dim]
 * @param n Number of embeddings
 * @param dim Dimension
 * @param similarities Output [n, n] similarity matrix
 */
void ComputeSimilarityMatrix(const float* embeddings, int n, int dim, float* similarities) {
    for (int i = 0; i < n; i++) {
        const float* emb_i = embeddings + i * dim;
        for (int j = i; j < n; j++) {
            const float* emb_j = embeddings + j * dim;
            float sim = simd::CosineSimilarity(emb_i, emb_j, dim);
            similarities[i * n + j] = sim;
            similarities[j * n + i] = sim;  // Symmetric
        }
    }
}

/**
 * Find top-k most similar embeddings to a query
 *
 * @param query Query embedding [dim]
 * @param corpus Corpus embeddings [n, dim]
 * @param n Number of corpus embeddings
 * @param dim Dimension
 * @param k Number of results
 * @param indices Output indices of top-k
 * @param scores Output similarity scores
 */
void TopKSimilar(const float* query, const float* corpus, int n, int dim, int k, int* indices, float* scores) {
    // Simple O(n*k) algorithm - sufficient for moderate n
    // For large n, use approximate methods (HNSW, IVF)

    std::vector<std::pair<float, int>> all_scores(n);

    for (int i = 0; i < n; i++) {
        const float* emb = corpus + i * dim;
        all_scores[i] = {simd::DotF32(query, emb, dim), i};
    }

    // Partial sort for top-k
    std::partial_sort(all_scores.begin(), all_scores.begin() + k, all_scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;  // Descending
    });

    for (int i = 0; i < k; i++) {
        indices[i] = all_scores[i].second;
        scores[i] = all_scores[i].first;
    }
}

}  // namespace densecore
