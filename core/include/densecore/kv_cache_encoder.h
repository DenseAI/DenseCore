/**
 * @file kv_cache_encoder.h
 * @brief Fixed KV Cache for Encoder-Decoder Architectures (Whisper, T5)
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 */

#ifndef DENSECORE_KV_CACHE_ENCODER_H
#define DENSECORE_KV_CACHE_ENCODER_H

#include "densecore/hal/tensor.h"
#include <memory>
#include <vector>

namespace densecore {

/**
 * @brief Encoder Key-Value Cache
 * 
 * Stores pre-computed Key and Value tensors for Cross-Attention layers.
 * Unlike Decoder KV cache, this is:
 * 1. Fixed size (determined by encoder output length)
 * 2. Written once (after encoder pass)
 * 3. Read-only during decoding
 */
class EncoderKVCache {
public:
    /**
     * @brief Initialize Encoder KV Cache
     * 
     * @param n_layers Number of decoder layers using cross-attention
     * @param batch_size Batch size
     * @param seq_len Encoder sequence length
     * @param n_head Number of KV heads
     * @param head_dim Dimension per head
     * @param dtype Data type (FP16 or FP32)
     */
    EncoderKVCache(int n_layers, int batch_size, int seq_len, int n_head, int head_dim, DType dtype);

    ~EncoderKVCache();

    // Disable copy
    EncoderKVCache(const EncoderKVCache&) = delete;
    EncoderKVCache& operator=(const EncoderKVCache&) = delete;

    // Move support
    EncoderKVCache(EncoderKVCache&&) = default;
    EncoderKVCache& operator=(EncoderKVCache&&) = default;

    /**
     * @brief Get Key tensor for specific layer
     * @return Pointer to Tensor (valid as long as cache exists)
     */
    Tensor* GetKey(int layer);

    /**
     * @brief Get Value tensor for specific layer
     * @return Pointer to Tensor (valid as long as cache exists)
     */
    Tensor* GetValue(int layer);

    /**
     * @brief Compute memory usage in bytes
     */
    size_t GetMemoryUsage() const;

private:
    int n_layers_;
    int batch_size_;
    int seq_len_;
    int n_head_;
    int head_dim_;
    DType dtype_;

    // Tensors for each layer
    // We store actual Tensors, managing their data buffers
    std::vector<Tensor> keys_;
    std::vector<Tensor> values_;

    // Helper to allocate memory for a tensor
    void AllocateTensor(Tensor& tensor);
    void FreeTensor(Tensor& tensor);
};

}  // namespace densecore

#endif  // DENSECORE_KV_CACHE_ENCODER_H
