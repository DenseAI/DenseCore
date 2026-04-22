/**
 * @file densecore/models/embedding.h
 * @brief Embedding model support for RAG/Semantic Search
 *
 * Provides CPU-optimized embedding extraction with:
 * - Multiple pooling strategies (MEAN, CLS, LAST, MAX)
 * - L2 normalization for cosine similarity
 * - Batch embedding support
 */

#ifndef DENSECORE_EMBEDDING_H
#define DENSECORE_EMBEDDING_H

#include <cstdint>
#include <vector>

namespace densecore {

/**
 * Pooling strategy for sequence-to-vector conversion
 */
enum class PoolingStrategy {
    MEAN = 0,  // Mean of all tokens (sentence-transformers default)
    CLS = 1,   // First token [CLS] (BERT-style)
    LAST = 2,  // Last token
    MAX = 3    // Element-wise max
};

/**
 * Configuration for embedding extraction
 */
struct EmbeddingConfig {
    PoolingStrategy pooling = PoolingStrategy::MEAN;
    bool normalize = true;  // L2 normalization
    int max_length = 512;   // Max sequence length (truncate if longer)
    bool truncate = true;   // Truncate if exceeds max_length
};

/**
 * Result of embedding extraction
 */
struct EmbeddingResult {
    std::vector<float> embedding;  // [dim] normalized embedding
    int dimension = 0;             // Embedding dimension
    int tokens_used = 0;           // Actual tokens processed
    bool truncated = false;        // Was input truncated
};

/**
 * Batch embedding result
 */
struct BatchEmbeddingResult {
    std::vector<float> embeddings;  // [batch_size * dim] flattened
    int batch_size = 0;
    int dimension = 0;
    std::vector<int> tokens_used;  // Per-sequence token counts
};

/**
 * Get pooling strategy from string (for API compatibility)
 */
inline PoolingStrategy PoolingFromString(const char* str) {
    if (!str) return PoolingStrategy::MEAN;

    if (str[0] == 'm' || str[0] == 'M') {
        if (str[1] == 'e' || str[1] == 'E') return PoolingStrategy::MEAN;
        if (str[1] == 'a' || str[1] == 'A') return PoolingStrategy::MAX;
    }
    if (str[0] == 'c' || str[0] == 'C') return PoolingStrategy::CLS;
    if (str[0] == 'l' || str[0] == 'L') return PoolingStrategy::LAST;

    return PoolingStrategy::MEAN;  // Default
}

/**
 * Get pooling strategy name
 */
inline const char* PoolingName(PoolingStrategy strategy) {
    switch (strategy) {
    case PoolingStrategy::MEAN: return "mean";
    case PoolingStrategy::CLS: return "cls";
    case PoolingStrategy::LAST: return "last";
    case PoolingStrategy::MAX: return "max";
    default: return "mean";
    }
}

}  // namespace densecore

#endif  // DENSECORE_EMBEDDING_H
