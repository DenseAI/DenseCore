#include "densecore/kv_cache_decoder.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace densecore {

DecoderKVCache::DecoderKVCache(int n_layers, int batch_size, int max_seq_len, int n_head, int head_dim, DType dtype)
    : n_layers_(n_layers),
      batch_size_(batch_size),
      max_seq_len_(max_seq_len),
      n_head_(n_head),
      head_dim_(head_dim),
      dtype_(dtype),
      current_seq_len_(0) {

    keys_.resize(n_layers);
    values_.resize(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        // [Batch, MaxSeq, Head, HeadDim]
        keys_[i].dtype = dtype;
        keys_[i].layout = TensorLayout::SEQ;
        keys_[i].ndim = 4;
        keys_[i].shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len),
                          static_cast<int64_t>(n_head), static_cast<int64_t>(head_dim)};
        AllocateTensor(keys_[i]);

        values_[i].dtype = dtype;
        values_[i].layout = TensorLayout::SEQ;
        values_[i].ndim = 4;
        values_[i].shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len),
                            static_cast<int64_t>(n_head), static_cast<int64_t>(head_dim)};
        AllocateTensor(values_[i]);
    }
}

DecoderKVCache::~DecoderKVCache() {
    for (auto& t : keys_) FreeTensor(t);
    for (auto& t : values_) FreeTensor(t);
}

void DecoderKVCache::AllocateTensor(Tensor& tensor) {
    size_t element_size = DTypeSizeBytes(tensor.dtype);
    size_t num_elements = 1;
    for (int i = 0; i < tensor.ndim; ++i) {
        num_elements *= tensor.shape[i];
    }
    size_t total_bytes = num_elements * element_size;

    // Round up size to 64-byte multiple to satisfy aligned_alloc on POSIX
    // and ensure safety for SIMD access.
    size_t aligned_size = (total_bytes + 63) & ~((size_t)63);

    // 64-byte alignment
#ifdef _WIN32
    tensor.data = _aligned_malloc(aligned_size, 64);
#else
    tensor.data = aligned_alloc(64, aligned_size);
#endif

    if (!tensor.data) {
        throw std::runtime_error("DecoderKVCache: Failed to allocate memory");
    }
    std::memset(tensor.data, 0, aligned_size);

    // Compute strides (Row-major)
    tensor.stride[3] = 1;
    tensor.stride[2] = tensor.shape[3];
    tensor.stride[1] = tensor.shape[2] * tensor.shape[3];
    tensor.stride[0] = tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
}

void DecoderKVCache::FreeTensor(Tensor& tensor) {
    if (tensor.data) {
#ifdef _WIN32
        _aligned_free(tensor.data);
#else
        free(tensor.data);
#endif
        tensor.data = nullptr;
    }
}

void DecoderKVCache::AddToken(int layer, int batch_idx, const Tensor& k_new, const Tensor& v_new) {
    if (layer < 0 || layer >= n_layers_) return;
    if (current_seq_len_ >= max_seq_len_) {
        // Ring buffer logic: overwrite from beginning?
        // For now, simple error or clamp. Decoder usually handles finite len.
        return;
    }

    // Assumptions: k_new is [1, 1, H, D]
    // We update current length OUTSIDE this per-layer call usually,
    // but here let's assume AddToken is called for all layers then we increment seq_len?
    // Actually simpler: caller manages seq_len increment? No, encapsulate here.
    // BUT: separate AddToken calls for each layer means we can't increment seq_len easily PER layer.
    // Wait, typical usage: model loop iterates layers.
    // So we should verify if this is the first layer update for this step or ...
    // Better design: Just copy to [batch, current_pos, ...]
    // And allow caller to explicitly Increment() or rely on implicit logic.
    // Let's assume AddToken writes to 'current_seq_len_' slot.
    // NOTE: This implies existing 'current_seq_len_' is the write index.

    size_t head_dim_bytes = head_dim_ * DTypeSizeBytes(dtype_);
    size_t entry_size = n_head_ * head_dim_bytes;  // Size of one token's KV (all heads)

    // Destination handling
    // Tensor Layout: [B, S, H, D]
    // Stride[0] = S*H*D, Stride[1] = H*D (which is entry_size)

    // Calculate offset for [batch_idx, current_seq_len_, 0, 0]
    size_t offset = batch_idx * keys_[layer].stride[0] + current_seq_len_ * keys_[layer].stride[1];  // in elements

    // Copy K
    uint8_t* dst_k = static_cast<uint8_t*>(keys_[layer].data) + offset * DTypeSizeBytes(dtype_);
    const uint8_t* src_k = static_cast<const uint8_t*>(k_new.data);
    std::memcpy(dst_k, src_k, entry_size);

    // Copy V
    uint8_t* dst_v = static_cast<uint8_t*>(values_[layer].data) + offset * DTypeSizeBytes(dtype_);
    const uint8_t* src_v = static_cast<const uint8_t*>(v_new.data);
    std::memcpy(dst_v, src_v, entry_size);
}

// NOTE: This must be called ONCE per step, after all layers updated
// OR we check if layer == n_layers-1. But explicit is safer.
// Actually, to keep API simple, let's fix the logic:
// AddToken DOES NOT increment. We need an Explicit 'Step()' or manage it differently.
// For now, let's assume the user manually calls Step() or we assume single-token stepping.
// Let's modify: AddToken just writes.
// BUT we typically need to read what we just wrote for SelfAttention?
// No, SelfAttention processes Past + Current.
// So we write Current, then GetFullView sends (0..current+1). Then we increment.

void DecoderKVCache::Step() {
    if (current_seq_len_ < max_seq_len_) {
        current_seq_len_++;
    }
}

Tensor DecoderKVCache::GetCurrentKey(int layer) {
    Tensor t = keys_[layer];

    // View includes only committed tokens (after Step()).
    int64_t valid_count = current_seq_len_;
    if (valid_count > max_seq_len_) {
        valid_count = max_seq_len_;
    }

    t.shape[1] = valid_count;
    return t;
}

Tensor DecoderKVCache::GetCurrentValue(int layer) {
    Tensor t = values_[layer];

    // View includes only committed tokens (after Step()).
    int64_t valid_count = current_seq_len_;
    if (valid_count > max_seq_len_) {
        valid_count = max_seq_len_;
    }

    t.shape[1] = valid_count;
    return t;
}

void DecoderKVCache::Reset() {
    current_seq_len_ = 0;
}

size_t DecoderKVCache::GetMemoryUsage() const {
    size_t element_size = DTypeSizeBytes(dtype_);
    // 2 tensors per layer (K, V)
    size_t per_tensor = batch_size_ * max_seq_len_ * n_head_ * head_dim_ * element_size;
    return n_layers_ * 2 * per_tensor;
}

}  // namespace densecore
