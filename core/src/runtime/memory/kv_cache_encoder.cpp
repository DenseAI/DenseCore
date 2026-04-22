/**
 * @file kv_cache_encoder.cpp
 * @brief Implementation of Encoder Key-Value Cache
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 */

#include "densecore/kv_cache_encoder.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

namespace densecore {

EncoderKVCache::EncoderKVCache(int n_layers, int batch_size, int seq_len, int n_head, int head_dim, DType dtype)
    : n_layers_(n_layers),
      batch_size_(batch_size),
      seq_len_(seq_len),
      n_head_(n_head),
      head_dim_(head_dim),
      dtype_(dtype) {

    keys_.resize(n_layers);
    values_.resize(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        // Initialize K Tensor
        keys_[i].dtype = dtype;
        keys_[i].layout = TensorLayout::SEQ;  // [Batch, Seq, Head, Dim] per CrossAttention kernel
        keys_[i].ndim = 4;
        keys_[i].shape[0] = batch_size;
        keys_[i].shape[1] = seq_len;
        keys_[i].shape[2] = n_head;
        keys_[i].shape[3] = head_dim;
        AllocateTensor(keys_[i]);

        // Initialize V Tensor
        values_[i].dtype = dtype;
        values_[i].layout = TensorLayout::SEQ;
        values_[i].ndim = 4;
        values_[i].shape[0] = batch_size;
        values_[i].shape[1] = seq_len;
        values_[i].shape[2] = n_head;
        values_[i].shape[3] = head_dim;
        AllocateTensor(values_[i]);
    }
}

EncoderKVCache::~EncoderKVCache() {
    for (auto& tensor : keys_) {
        FreeTensor(tensor);
    }
    for (auto& tensor : values_) {
        FreeTensor(tensor);
    }
}

Tensor* EncoderKVCache::GetKey(int layer) {
    if (layer < 0 || layer >= n_layers_) {
        return nullptr;
    }
    return &keys_[layer];
}

Tensor* EncoderKVCache::GetValue(int layer) {
    if (layer < 0 || layer >= n_layers_) {
        return nullptr;
    }
    return &values_[layer];
}

size_t EncoderKVCache::GetMemoryUsage() const {
    size_t element_size = DTypeSizeBytes(dtype_);
    size_t elements_per_tensor = static_cast<size_t>(batch_size_) * seq_len_ * n_head_ * head_dim_;
    return n_layers_ * 2 * elements_per_tensor * element_size;
}

void EncoderKVCache::AllocateTensor(Tensor& tensor) {
    size_t element_size = DTypeSizeBytes(tensor.dtype);
    size_t num_elements = 1;
    for (int i = 0; i < tensor.ndim; ++i) {
        num_elements *= tensor.shape[i];
    }
    size_t total_bytes = num_elements * element_size;

    // Round up size to 64-byte multiple.
    // Standard aligned_alloc requires size to be a multiple of alignment.
    // This also ensures safety for SIMD loads/stores that might read past end of exact size.
    size_t aligned_size = (total_bytes + 63) & ~((size_t)63);

    // 64-byte alignment for AVX-512
#ifdef _WIN32
    tensor.data = aligned_alloc(64, aligned_size);
#else
    tensor.data = aligned_alloc(64, aligned_size);
#endif

    if (!tensor.data) {
        throw std::runtime_error("EncoderKVCache: Failed to allocate aligned memory");
    }

    // Zero initialize including padding
    std::memset(tensor.data, 0, aligned_size);
}

void EncoderKVCache::FreeTensor(Tensor& tensor) {
    if (tensor.data) {
        aligned_free(tensor.data);
        tensor.data = nullptr;
    }
}

}  // namespace densecore
