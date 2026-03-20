/**
 * @file paged_attention.h
 * @brief AVX2 Paged Attention Kernel (vLLM-style)
 *
 * Provides optimized attention computation over non-contiguous KV blocks.
 * Designed for CPU inference with AVX2 support.
 */

#ifndef DENSECORE_KERNELS_PAGED_ATTENTION_H
#define DENSECORE_KERNELS_PAGED_ATTENTION_H

#include "densecore/hal/tensor.h"
#include "kv_cache.h"
#include <vector>

namespace densecore {
namespace kernels {

/**
 * @brief Compute Attention with Paged KV Cache (Highway/SIMD)
 *
 * Computes: Output = Softmax(Q * K^T / sqrt(d)) * V
 * Accesses K/V data from PagedKVCache block table.
 * 
 * Uses Google Highway to dispatch to the best available SIMD instruction set 
 * (AVX2, AVX-512, NEON, etc.) at runtime.
 *
 * @param query         [NumHeads, HeadDim] (Current token's query)
 * @param cache         PagedKVCache instance managing the blocks
 * @param layer         Layer index to access the correct cache blocks
 * @param block_table   [NumBlocks] List of physical block IDs for this sequence
 * @param context_len   Total number of tokens in context (valid tokens in blocks)
 * @param scale         Softmax scale factor (1/sqrt(dim))
 * @param output        [NumHeads, HeadDim] Output buffer
 * @param head_start    Optional start head index (inclusive) for parallel slicing
 * @param head_end      Optional end head index (exclusive) for parallel slicing
 * @param num_heads_total Optional total query heads (for GQA mapping when slicing)
 *
 * @note Assumes BLOCK_SIZE = 16.
 * @note Supports Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
 *       by broadcasting KV heads.
 */
void PagedAttention(const Tensor& query, const PagedKVCache& cache, int layer, const std::vector<int>& block_table,
                    int context_len, float scale, Tensor* output, int head_start = 0, int head_end = -1,
                    int num_heads_total = -1);

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_PAGED_ATTENTION_H
