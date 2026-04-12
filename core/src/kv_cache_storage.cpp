#include "kv_cache_internal.h"

#include "simd_ops.h"

size_t PagedKVCache::GetBytesPerSlot() const {
    const int64_t elements = static_cast<int64_t>(head_dim) * static_cast<int64_t>(n_head_kv);
    if (elements <= 0) {
        return 0;
    }

    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        return ggml_row_size(cache_type, static_cast<int64_t>(head_dim)) * static_cast<size_t>(n_head_kv);
    }

    return ggml_row_size(cache_type, elements);
}

size_t PagedKVCache::GetVBytesPerSlot() const {
    const int64_t elements = static_cast<int64_t>(v_head_dim) * static_cast<int64_t>(n_head_kv);
    if (elements <= 0) {
        return 0;
    }

    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        return ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim)) * static_cast<size_t>(n_head_kv);
    }

    return ggml_row_size(cache_type, elements);
}

size_t PagedKVCache::GetIndexBytesPerSlot() const {
    if (!has_index_cache || index_head_dim <= 0) {
        return 0;
    }
    return sizeof(ggml_fp16_t) * static_cast<size_t>(index_head_dim);
}

size_t PagedKVCache::GetBytesPerBlock() const {
    return GetBytesPerSlot() * BLOCK_SIZE;
}
size_t PagedKVCache::GetVBytesPerBlock() const {
    return GetVBytesPerSlot() * BLOCK_SIZE;
}
size_t PagedKVCache::GetIndexBytesPerBlock() const {
    return GetIndexBytesPerSlot() * BLOCK_SIZE;
}

PagedKVCache::BlockLayout PagedKVCache::GetBlockLayout() const {
    BlockLayout layout;
    layout.cache_type = cache_type;
    layout.slot_stride_bytes = GetBytesPerSlot();
    layout.block_stride_bytes = GetBytesPerBlock();
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        layout.head_stride_bytes = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        const int qk = ggml_blck_size(cache_type);
        layout.packed_values_per_block = qk;
        layout.packed_blocks_per_head = (head_dim + qk - 1) / qk;
    } else {
        if (n_head_kv > 0) {
            layout.head_stride_bytes = layout.slot_stride_bytes / static_cast<size_t>(n_head_kv);
        }
        layout.packed_values_per_block = 1;
        layout.packed_blocks_per_head = head_dim;
    }
    return layout;
}

PagedKVCache::BlockLayout PagedKVCache::GetVBlockLayout() const {
    BlockLayout layout;
    layout.cache_type = cache_type;
    layout.slot_stride_bytes = GetVBytesPerSlot();
    layout.block_stride_bytes = GetVBytesPerBlock();
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        layout.head_stride_bytes = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        const int qk = ggml_blck_size(cache_type);
        layout.packed_values_per_block = qk;
        layout.packed_blocks_per_head = (v_head_dim + qk - 1) / qk;
    } else {
        if (n_head_kv > 0) {
            layout.head_stride_bytes = layout.slot_stride_bytes / static_cast<size_t>(n_head_kv);
        }
        layout.packed_values_per_block = 1;
        layout.packed_blocks_per_head = v_head_dim;
    }
    return layout;
}

void* PagedKVCache::GetKBlockPtr(int block_id, int layer) {
    if (block_id < 0 || block_id >= max_blocks || layer < 0 || layer >= n_layer) return nullptr;

    int64_t block_index = static_cast<int64_t>(layer) * max_blocks + block_id;
    size_t byte_offset = block_index * GetBytesPerBlock();

    if (use_block_allocator && k_allocator) {
        void* ptr = static_cast<char*>(k_allocator->ArenaBase()) + byte_offset;
        DENSECORE_ASSERT_ALIGNED_64(ptr);
        return ptr;
    }
    return nullptr;
}

const void* PagedKVCache::GetKBlockPtr(int block_id, int layer) const {
    return const_cast<PagedKVCache*>(this)->GetKBlockPtr(block_id, layer);
}

void* PagedKVCache::GetVBlockPtr(int block_id, int layer) {
    if (block_id < 0 || block_id >= max_blocks || layer < 0 || layer >= n_layer) return nullptr;

    int64_t block_index = static_cast<int64_t>(layer) * max_blocks + block_id;
    size_t byte_offset = block_index * GetVBytesPerBlock();

    if (use_block_allocator && v_allocator) {
        void* ptr = static_cast<char*>(v_allocator->ArenaBase()) + byte_offset;
        DENSECORE_ASSERT_ALIGNED_64(ptr);
        return ptr;
    }
    return nullptr;
}

const void* PagedKVCache::GetVBlockPtr(int block_id, int layer) const {
    return const_cast<PagedKVCache*>(this)->GetVBlockPtr(block_id, layer);
}

void* PagedKVCache::GetIndexBlockPtr(int block_id, int layer) {
    if (!has_index_cache || !index_allocator) return nullptr;
    if (block_id < 0 || block_id >= max_blocks || layer < 0 || layer >= n_layer) return nullptr;

    int64_t block_index = static_cast<int64_t>(layer) * max_blocks + block_id;
    size_t byte_offset = block_index * GetIndexBytesPerBlock();
    void* ptr = static_cast<char*>(index_allocator->ArenaBase()) + byte_offset;
    DENSECORE_ASSERT_ALIGNED_64(ptr);
    return ptr;
}

const void* PagedKVCache::GetIndexBlockPtr(int block_id, int layer) const {
    return const_cast<PagedKVCache*>(this)->GetIndexBlockPtr(block_id, layer);
}

void* PagedKVCache::GetKSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetKBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE) return nullptr;
    return static_cast<char*>(block_ptr) + static_cast<size_t>(slot) * GetBytesPerSlot();
}

void* PagedKVCache::GetVSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetVBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE) return nullptr;
    return static_cast<char*>(block_ptr) + static_cast<size_t>(slot) * GetVBytesPerSlot();
}

void* PagedKVCache::GetIndexSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetIndexBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE) return nullptr;
    return static_cast<char*>(block_ptr) + static_cast<size_t>(slot) * GetIndexBytesPerSlot();
}

void PagedKVCache::CopyBlockData(int src_block_id, int dst_block_id) {
    if (src_block_id == dst_block_id || src_block_id < 0 || dst_block_id < 0) return;

    size_t bytes_per_k_block = GetBytesPerBlock();
    size_t bytes_per_v_block = GetVBytesPerBlock();
    size_t bytes_per_index_block = GetIndexBytesPerBlock();

    for (int layer = 0; layer < n_layer; layer++) {
        void* k_src = GetKBlockPtr(src_block_id, layer);
        void* k_dst = GetKBlockPtr(dst_block_id, layer);
        void* v_src = GetVBlockPtr(src_block_id, layer);
        void* v_dst = GetVBlockPtr(dst_block_id, layer);
        void* i_src = GetIndexBlockPtr(src_block_id, layer);
        void* i_dst = GetIndexBlockPtr(dst_block_id, layer);

        if (k_src && k_dst) memcpy(k_dst, k_src, bytes_per_k_block);
        if (v_src && v_dst) memcpy(v_dst, v_src, bytes_per_v_block);
        if (i_src && i_dst && bytes_per_index_block > 0) memcpy(i_dst, i_src, bytes_per_index_block);
    }
}

void PagedKVCache::CopyBlockDataLayer(int src_block_id, int dst_block_id, int layer) {
    if (src_block_id == dst_block_id || src_block_id < 0 || dst_block_id < 0 || layer < 0 || layer >= n_layer) return;

    size_t bytes_per_k_block = GetBytesPerBlock();
    size_t bytes_per_v_block = GetVBytesPerBlock();
    size_t bytes_per_index_block = GetIndexBytesPerBlock();

    void* k_src = GetKBlockPtr(src_block_id, layer);
    void* k_dst = GetKBlockPtr(dst_block_id, layer);
    void* v_src = GetVBlockPtr(src_block_id, layer);
    void* v_dst = GetVBlockPtr(dst_block_id, layer);
    void* i_src = GetIndexBlockPtr(src_block_id, layer);
    void* i_dst = GetIndexBlockPtr(dst_block_id, layer);

    if (k_src && k_dst) memcpy(k_dst, k_src, bytes_per_k_block);
    if (v_src && v_dst) memcpy(v_dst, v_src, bytes_per_v_block);
    if (i_src && i_dst && bytes_per_index_block > 0) memcpy(i_dst, i_src, bytes_per_index_block);
}

void PagedKVCache::CopyBlocksToHost(const std::vector<int>& block_ids, std::vector<uint8_t>* k_out,
                                    std::vector<uint8_t>* v_out) const {
    if (!k_out || !v_out) return;

    size_t bytes_per_k_block = GetBytesPerBlock();
    size_t bytes_per_v_block = GetVBytesPerBlock();
    size_t bytes_per_index_block = GetIndexBytesPerBlock();
    size_t blocks = block_ids.size();
    size_t total_k_bytes = blocks * static_cast<size_t>(n_layer) * bytes_per_k_block;
    size_t total_v_bytes = blocks * static_cast<size_t>(n_layer) * bytes_per_v_block;
    size_t total_index_bytes = has_index_cache ? blocks * static_cast<size_t>(n_layer) * bytes_per_index_block : 0;

    k_out->assign(total_k_bytes + total_index_bytes, 0);
    v_out->assign(total_v_bytes, 0);

    for (size_t i = 0; i < blocks; ++i) {
        int block_id = block_ids[i];
        if (block_id < 0) continue;

        for (int layer = 0; layer < n_layer; ++layer) {
            const void* k_src = GetKBlockPtr(block_id, layer);
            const void* v_src = GetVBlockPtr(block_id, layer);
            const void* index_src = GetIndexBlockPtr(block_id, layer);
            size_t offset_k = (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_k_block;
            size_t offset_v = (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_v_block;
            if (k_src) memcpy(k_out->data() + offset_k, k_src, bytes_per_k_block);
            if (v_src) memcpy(v_out->data() + offset_v, v_src, bytes_per_v_block);
            if (index_src && total_index_bytes > 0) {
                size_t index_base = total_k_bytes;
                size_t offset_i = index_base + (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) *
                                                   bytes_per_index_block;
                memcpy(k_out->data() + offset_i, index_src, bytes_per_index_block);
            }
        }
    }
}

void PagedKVCache::RestoreBlocksFromHost(const std::vector<int>& block_ids, const std::vector<uint8_t>& k_in,
                                         const std::vector<uint8_t>& v_in) {
    size_t bytes_per_k_block = GetBytesPerBlock();
    size_t bytes_per_v_block = GetVBytesPerBlock();
    size_t bytes_per_index_block = GetIndexBytesPerBlock();
    size_t blocks = block_ids.size();
    size_t total_k_bytes = blocks * static_cast<size_t>(n_layer) * bytes_per_k_block;
    size_t total_v_bytes = blocks * static_cast<size_t>(n_layer) * bytes_per_v_block;
    size_t total_index_bytes = has_index_cache ? blocks * static_cast<size_t>(n_layer) * bytes_per_index_block : 0;

    if (k_in.size() < total_k_bytes || v_in.size() < total_v_bytes) {
        return;
    }

    for (size_t i = 0; i < blocks; ++i) {
        int block_id = block_ids[i];
        if (block_id < 0) continue;

        for (int layer = 0; layer < n_layer; ++layer) {
            void* k_dst = GetKBlockPtr(block_id, layer);
            void* v_dst = GetVBlockPtr(block_id, layer);
            void* index_dst = GetIndexBlockPtr(block_id, layer);
            size_t offset_k = (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_k_block;
            size_t offset_v = (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_v_block;
            if (k_dst) memcpy(k_dst, k_in.data() + offset_k, bytes_per_k_block);
            if (v_dst) memcpy(v_dst, v_in.data() + offset_v, bytes_per_v_block);
            if (index_dst && total_index_bytes > 0 && k_in.size() >= total_k_bytes + total_index_bytes) {
                size_t index_base = total_k_bytes;
                size_t offset_i = index_base + (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) *
                                                   bytes_per_index_block;
                memcpy(index_dst, k_in.data() + offset_i, bytes_per_index_block);
            }
        }
    }
}

bool PagedKVCache::IsQuantized() const {
    return ggml_is_quantized(cache_type);
}
int PagedKVCache::GetHeadCountForLayer(int layer) const {
    if (layer >= 0 && layer < static_cast<int>(layer_n_head_kv.size()) &&
        layer_n_head_kv[static_cast<size_t>(layer)] > 0) {
        return layer_n_head_kv[static_cast<size_t>(layer)];
    }
    return n_head_kv;
}
int PagedKVCache::GetHeadDimForLayer(int layer) const {
    if (layer >= 0 && layer < static_cast<int>(layer_head_dims.size()) &&
        layer_head_dims[static_cast<size_t>(layer)] > 0) {
        return layer_head_dims[static_cast<size_t>(layer)];
    }
    return head_dim;
}
int PagedKVCache::GetVHeadDimForLayer(int layer) const {
    if (layer >= 0 && layer < static_cast<int>(layer_v_head_dims.size()) &&
        layer_v_head_dims[static_cast<size_t>(layer)] > 0) {
        return layer_v_head_dims[static_cast<size_t>(layer)];
    }
    return v_head_dim;
}
int PagedKVCache::GetElementsPerSlot() const {
    return head_dim * n_head_kv;
}
int PagedKVCache::GetVElementsPerSlot() const {
    return v_head_dim * n_head_kv;
}
int PagedKVCache::GetElementsPerSlot(int layer) const {
    return GetHeadDimForLayer(layer) * GetHeadCountForLayer(layer);
}
int PagedKVCache::GetVElementsPerSlot(int layer) const {
    return GetVHeadDimForLayer(layer) * GetHeadCountForLayer(layer);
}

void PagedKVCache::WriteKSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetKSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    const int n = GetElementsPerSlot(layer);
    const int storage_n = GetElementsPerSlot();
    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        densecore::simd::ConvertF32ToF16(dst, data, n);
        if (storage_n > n) {
            std::fill(dst + n, dst + storage_n, ggml_fp16_t(0));
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        std::vector<float> padded_head(static_cast<size_t>(head_dim), 0.0f);
        for (int h = 0; h < n_head_kv; ++h) {
            const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            void* dst_head = dst + static_cast<size_t>(h) * head_stride;
            std::fill(padded_head.begin(), padded_head.end(), 0.0f);
            if (h < layer_n_head_kv) {
                std::memcpy(padded_head.data(), src_head, static_cast<size_t>(layer_head_dim) * sizeof(float));
            }
            ggml_quantize_chunk(cache_type, padded_head.data(), dst_head, 0, 1, head_dim, nullptr);
        }
    } else {
        auto* dst = static_cast<float*>(ptr);
        densecore::simd::CopyF32(dst, data, n);
        if (storage_n > n) {
            std::fill(dst + n, dst + storage_n, 0.0f);
        }
    }
}

void PagedKVCache::WriteVSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    const int n = GetVElementsPerSlot(layer);
    const int storage_n = GetVElementsPerSlot();
    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        densecore::simd::ConvertF32ToF16(dst, data, n);
        if (storage_n > n) {
            std::fill(dst + n, dst + storage_n, ggml_fp16_t(0));
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        std::vector<float> padded_head(static_cast<size_t>(v_head_dim), 0.0f);
        for (int h = 0; h < n_head_kv; ++h) {
            const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            void* dst_head = dst + static_cast<size_t>(h) * head_stride;
            std::fill(padded_head.begin(), padded_head.end(), 0.0f);
            if (h < layer_n_head_kv) {
                std::memcpy(padded_head.data(), src_head, static_cast<size_t>(layer_head_dim) * sizeof(float));
            }
            ggml_quantize_chunk(cache_type, padded_head.data(), dst_head, 0, 1, v_head_dim, nullptr);
        }
    } else {
        auto* dst = static_cast<float*>(ptr);
        densecore::simd::CopyF32(dst, data, n);
        if (storage_n > n) {
            std::fill(dst + n, dst + storage_n, 0.0f);
        }
    }
}

void PagedKVCache::WriteIndexSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetIndexSlotPtr(block_id, layer, slot);
    if (!ptr || !has_index_cache || index_head_dim <= 0) return;
    densecore::simd::ConvertF32ToF16(static_cast<ggml_fp16_t*>(ptr), data, index_head_dim);
}

void PagedKVCache::ReadKSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetKSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    const int n = GetElementsPerSlot(layer);
    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, static_cast<const ggml_fp16_t*>(ptr), n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        std::vector<float> tmp_head(static_cast<size_t>(head_dim), 0.0f);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            type_traits->to_float(src_head, tmp_head.data(), head_dim);
            std::memcpy(out_head, tmp_head.data(), static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    } else {
        densecore::simd::CopyF32(out, static_cast<const float*>(ptr), n);
    }
}

void PagedKVCache::ReadVSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    const int n = GetVElementsPerSlot(layer);
    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, static_cast<const ggml_fp16_t*>(ptr), n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        std::vector<float> tmp_head(static_cast<size_t>(v_head_dim), 0.0f);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            type_traits->to_float(src_head, tmp_head.data(), v_head_dim);
            std::memcpy(out_head, tmp_head.data(), static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    } else {
        densecore::simd::CopyF32(out, static_cast<const float*>(ptr), n);
    }
}

void PagedKVCache::ReadIndexSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetIndexSlotPtr(block_id, layer, slot);
    if (!ptr || !has_index_cache || index_head_dim <= 0) return;
    densecore::simd::ConvertF16ToF32(out, static_cast<const ggml_fp16_t*>(ptr), index_head_dim);
}

void PagedKVCache::WriteKSlots(int block_id, int layer, int start_slot, int num_slots, const float* data) {
    if (num_slots <= 0 || !data || start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    if (layer_n_head_kv != n_head_kv || layer_head_dim != head_dim) {
        const int elems_per_slot = GetElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            WriteKSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    if (!UseKVBulkSlotPath()) {
        const int elems_per_slot = GetElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            WriteKSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    void* ptr = GetKSlotPtr(block_id, layer, start_slot);
    if (!ptr) return;

    const int elems_per_slot = GetElementsPerSlot(layer);
    const int total_elements = elems_per_slot * num_slots;
    RecordKVWriteBulkUsage(num_slots);

    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        densecore::simd::ConvertF32ToF16(dst, data, total_elements);
        const int storage_total = GetElementsPerSlot() * num_slots;
        if (storage_total > total_elements) {
            std::fill(dst + total_elements, dst + storage_total, ggml_fp16_t(0));
        }
        return;
    }
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const int64_t total_rows = static_cast<int64_t>(num_slots) * static_cast<int64_t>(n_head_kv);
        if (layer_head_dim == head_dim) {
            ggml_quantize_chunk(cache_type, data, ptr, 0, total_rows, head_dim, nullptr);
        } else {
            std::vector<float> padded(static_cast<size_t>(GetElementsPerSlot()) * static_cast<size_t>(num_slots), 0.0f);
            for (int row = 0; row < total_rows; ++row) {
                const float* src_row = data + static_cast<size_t>(row) * static_cast<size_t>(layer_head_dim);
                float* dst_row = padded.data() + static_cast<size_t>(row) * static_cast<size_t>(head_dim);
                std::memcpy(dst_row, src_row, static_cast<size_t>(layer_head_dim) * sizeof(float));
            }
            ggml_quantize_chunk(cache_type, padded.data(), ptr, 0, total_rows, head_dim, nullptr);
        }
        return;
    }
    auto* dst = static_cast<float*>(ptr);
    densecore::simd::CopyF32(dst, data, total_elements);
    const int storage_total = GetElementsPerSlot() * num_slots;
    if (storage_total > total_elements) {
        std::fill(dst + total_elements, dst + storage_total, 0.0f);
    }
}

void PagedKVCache::WriteVSlots(int block_id, int layer, int start_slot, int num_slots, const float* data) {
    if (num_slots <= 0 || !data || start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    if (layer_n_head_kv != n_head_kv || layer_head_dim != v_head_dim) {
        const int elems_per_slot = GetVElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            WriteVSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    if (!UseKVBulkSlotPath()) {
        const int elems_per_slot = GetVElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            WriteVSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    void* ptr = GetVSlotPtr(block_id, layer, start_slot);
    if (!ptr) return;

    const int elems_per_slot = GetVElementsPerSlot(layer);
    const int total_elements = elems_per_slot * num_slots;
    RecordKVWriteBulkUsage(num_slots);

    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        densecore::simd::ConvertF32ToF16(dst, data, total_elements);
        const int storage_total = GetVElementsPerSlot() * num_slots;
        if (storage_total > total_elements) {
            std::fill(dst + total_elements, dst + storage_total, ggml_fp16_t(0));
        }
        return;
    }
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const int64_t total_rows = static_cast<int64_t>(num_slots) * static_cast<int64_t>(n_head_kv);
        if (layer_head_dim == v_head_dim) {
            ggml_quantize_chunk(cache_type, data, ptr, 0, total_rows, v_head_dim, nullptr);
        } else {
            std::vector<float> padded(static_cast<size_t>(GetVElementsPerSlot()) * static_cast<size_t>(num_slots),
                                      0.0f);
            for (int row = 0; row < total_rows; ++row) {
                const float* src_row = data + static_cast<size_t>(row) * static_cast<size_t>(layer_head_dim);
                float* dst_row = padded.data() + static_cast<size_t>(row) * static_cast<size_t>(v_head_dim);
                std::memcpy(dst_row, src_row, static_cast<size_t>(layer_head_dim) * sizeof(float));
            }
            ggml_quantize_chunk(cache_type, padded.data(), ptr, 0, total_rows, v_head_dim, nullptr);
        }
        return;
    }
    auto* dst = static_cast<float*>(ptr);
    densecore::simd::CopyF32(dst, data, total_elements);
    const int storage_total = GetVElementsPerSlot() * num_slots;
    if (storage_total > total_elements) {
        std::fill(dst + total_elements, dst + storage_total, 0.0f);
    }
}

void PagedKVCache::ReadKSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const {
    if (num_slots <= 0 || !out || start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    if (layer_n_head_kv != n_head_kv || layer_head_dim != head_dim) {
        const int elems_per_slot = GetElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            ReadKSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    if (!UseKVBulkSlotPath()) {
        const int elems_per_slot = GetElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            ReadKSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    const void* ptr = const_cast<PagedKVCache*>(this)->GetKSlotPtr(block_id, layer, start_slot);
    if (!ptr) return;

    const int elems_per_slot = GetElementsPerSlot(layer);
    const int total_elements = elems_per_slot * num_slots;
    RecordKVReadBulkUsage(num_slots);

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, static_cast<const ggml_fp16_t*>(ptr), total_elements);
        return;
    }
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        if (!type_traits || !type_traits->to_float) return;
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        const int total_rows = num_slots * n_head_kv;
        std::vector<float> tmp_head(static_cast<size_t>(head_dim), 0.0f);
        for (int row = 0; row < total_rows; ++row) {
            type_traits->to_float(src + static_cast<size_t>(row) * head_stride, tmp_head.data(), head_dim);
            std::memcpy(out + static_cast<size_t>(row) * static_cast<size_t>(layer_head_dim), tmp_head.data(),
                        static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
        return;
    }
    densecore::simd::CopyF32(out, static_cast<const float*>(ptr), total_elements);
}

void PagedKVCache::ReadVSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const {
    if (num_slots <= 0 || !out || start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    if (layer_n_head_kv != n_head_kv || layer_head_dim != v_head_dim) {
        const int elems_per_slot = GetVElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            ReadVSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    if (!UseKVBulkSlotPath()) {
        const int elems_per_slot = GetVElementsPerSlot(layer);
        for (int i = 0; i < num_slots; ++i) {
            ReadVSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
        }
        return;
    }
    const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, start_slot);
    if (!ptr) return;

    const int elems_per_slot = GetVElementsPerSlot(layer);
    const int total_elements = elems_per_slot * num_slots;
    RecordKVReadBulkUsage(num_slots);

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, static_cast<const ggml_fp16_t*>(ptr), total_elements);
        return;
    }
    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        if (!type_traits || !type_traits->to_float) return;
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        const int total_rows = num_slots * n_head_kv;
        std::vector<float> tmp_head(static_cast<size_t>(v_head_dim), 0.0f);
        for (int row = 0; row < total_rows; ++row) {
            type_traits->to_float(src + static_cast<size_t>(row) * head_stride, tmp_head.data(), v_head_dim);
            std::memcpy(out + static_cast<size_t>(row) * static_cast<size_t>(layer_head_dim), tmp_head.data(),
                        static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
        return;
    }
    densecore::simd::CopyF32(out, static_cast<const float*>(ptr), total_elements);
}

void PagedKVCache::PrefetchBlock(int block_id, int layer) const {
    const void* k_ptr = const_cast<PagedKVCache*>(this)->GetKBlockPtr(block_id, layer);
    const void* v_ptr = const_cast<PagedKVCache*>(this)->GetVBlockPtr(block_id, layer);

    if (k_ptr) densecore::simd::PrefetchRange(k_ptr, GetBytesPerBlock());
    if (v_ptr) densecore::simd::PrefetchRange(v_ptr, GetBytesPerBlock());
}

void PagedKVCache::PrefetchNextLayer(const std::vector<int>& block_ids, int next_layer) const {
    if (next_layer >= n_layer) return;
    for (int block_id : block_ids) {
        PrefetchBlock(block_id, next_layer);
    }
}
