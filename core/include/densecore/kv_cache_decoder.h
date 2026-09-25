/**
 * @file kv_cache_decoder.h
 * @brief Incremental KV Cache for Decoder (Ring Buffer / Contiguous)
 *
 * Designed for Encoder-Decoder architectures where decoding happens step-by-step.
 * Uses contiguous memory for maximum SIMD efficiency.
 */

#ifndef DENSECORE_KV_CACHE_DECODER_H
#define DENSECORE_KV_CACHE_DECODER_H

#include "densecore/hal/tensor.h"
#include <vector>

namespace densecore {

class DecoderKVCache {
public:
    /**
     * @brief Initialize Decoder KV Cache
     * @param n_layers Number of layers
     * @param batch_size Maximum batch size
     * @param max_seq_len Maximum sequence length (capacity)
     * @param n_head Number of heads
     * @param head_dim Dimension per head
     * @param dtype Data type (F16/F32)
     */
    DecoderKVCache(int n_layers, int batch_size, int max_seq_len, int n_head, int head_dim, DType dtype);

    ~DecoderKVCache();

    /**
     * @brief Add a new token's KV to the cache
     * @param layer Layer index
     * @param batch_idx Batch index
     * @param k_new Key tensor for the new token [1, 1, H, D]
     * @param v_new Value tensor for the new token [1, 1, H, D]
     */
    void AddToken(int layer, int batch_idx, const Tensor& k_new, const Tensor& v_new);

    /**
     * @brief Get the full Key tensor view for attention (up to current length)
     * @param layer Layer index
     * @return Tensor view [Batch, CurrentSeq, Head, Dim]
     */
    Tensor GetCurrentKey(int layer);

    /**
     * @brief Get the full Value tensor view
     */
    Tensor GetCurrentValue(int layer);

    /**
     * @brief Reset cache state (current_length = 0)
     */
    void Reset();

    /**
     * @brief Increment sequence length counter (Call after processing all layers)
     */
    void Step();

    /**
     * @brief Get current sequence length
     */
    int GetCurrentLength() const { return current_seq_len_; }

    /**
     * @brief Get total memory usage in bytes
     */
    size_t GetMemoryUsage() const;

private:
    int n_layers_;
    int batch_size_;
    int max_seq_len_;
    int n_head_;
    int head_dim_;
    DType dtype_;

    int current_seq_len_ = 0;

    // We hold full capacity tensors
    // layout: [Batch, MaxSeq, Head, Dim]
    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;

    void AllocateTensor(Tensor& tensor);
    void FreeTensor(Tensor& tensor);
};

}  // namespace densecore

#endif  // DENSECORE_KV_CACHE_DECODER_H
