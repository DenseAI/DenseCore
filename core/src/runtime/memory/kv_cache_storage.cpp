#include "runtime/memory/kv_cache_internal.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "densecore/simd/simd_ops.h"

namespace {

std::vector<float>& GetKVFloatScratch(size_t elements) {
    static thread_local std::vector<float> scratch;
    if (scratch.size() < elements) {
        scratch.resize(elements);
        RecordKVScratchGrow();
    }
    return scratch;
}

void PackLayerSlotsToStorageLayout(float* dst, const float* src, int num_slots, int storage_head_dim,
                                   int storage_heads, int layer_head_dim, int layer_heads) {
    const int storage_elems = storage_head_dim * storage_heads;
    const int layer_elems = layer_head_dim * layer_heads;
    std::fill(dst, dst + static_cast<size_t>(storage_elems) * static_cast<size_t>(num_slots), 0.0f);
    for (int s = 0; s < num_slots; ++s) {
        float* slot_dst = dst + static_cast<size_t>(s) * static_cast<size_t>(storage_elems);
        const float* slot_src = src + static_cast<size_t>(s) * static_cast<size_t>(layer_elems);
        for (int h = 0; h < layer_heads; ++h) {
            std::memcpy(slot_dst + static_cast<size_t>(h) * static_cast<size_t>(storage_head_dim),
                        slot_src + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim),
                        static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    }
}

void UnpackStorageSlotsToLayerLayout(float* dst, const float* src, int num_slots, int storage_head_dim,
                                     int storage_heads, int layer_head_dim, int layer_heads) {
    (void)storage_heads;
    const int storage_elems = storage_head_dim * storage_heads;
    const int layer_elems = layer_head_dim * layer_heads;
    for (int s = 0; s < num_slots; ++s) {
        const float* slot_src = src + static_cast<size_t>(s) * static_cast<size_t>(storage_elems);
        float* slot_dst = dst + static_cast<size_t>(s) * static_cast<size_t>(layer_elems);
        for (int h = 0; h < layer_heads; ++h) {
            std::memcpy(slot_dst + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim),
                        slot_src + static_cast<size_t>(h) * static_cast<size_t>(storage_head_dim),
                        static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    }
}

void WriteLayerSlotsToF16Storage(ggml_fp16_t* dst, const float* src, int num_slots, int storage_head_dim,
                                 int storage_heads, int layer_head_dim, int layer_heads) {
    const int storage_elems = storage_head_dim * storage_heads;
    const int layer_elems = layer_head_dim * layer_heads;
    std::fill(dst, dst + static_cast<size_t>(storage_elems) * static_cast<size_t>(num_slots), ggml_fp16_t(0));
    for (int s = 0; s < num_slots; ++s) {
        ggml_fp16_t* slot_dst = dst + static_cast<size_t>(s) * static_cast<size_t>(storage_elems);
        const float* slot_src = src + static_cast<size_t>(s) * static_cast<size_t>(layer_elems);
        for (int h = 0; h < layer_heads; ++h) {
            densecore::simd::ConvertF32ToF16(slot_dst + static_cast<size_t>(h) * static_cast<size_t>(storage_head_dim),
                                             slot_src + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim),
                                             layer_head_dim);
        }
    }
}

void ReadLayerSlotsFromF16Storage(float* dst, const ggml_fp16_t* src, int num_slots, int storage_head_dim,
                                  int storage_heads, int layer_head_dim, int layer_heads) {
    const int storage_elems = storage_head_dim * storage_heads;
    const int layer_elems = layer_head_dim * layer_heads;
    for (int s = 0; s < num_slots; ++s) {
        const ggml_fp16_t* slot_src = src + static_cast<size_t>(s) * static_cast<size_t>(storage_elems);
        float* slot_dst = dst + static_cast<size_t>(s) * static_cast<size_t>(layer_elems);
        for (int h = 0; h < layer_heads; ++h) {
            densecore::simd::ConvertF16ToF32(
                slot_dst + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim),
                slot_src + static_cast<size_t>(h) * static_cast<size_t>(storage_head_dim), layer_head_dim);
        }
    }
}

}  // namespace

size_t PagedKVCache::GetBytesPerSlot() const {
    if (k_bytes_per_slot != 0) {
        return k_bytes_per_slot;
    }
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
    if (v_bytes_per_slot != 0) {
        return v_bytes_per_slot;
    }
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
    if (index_bytes_per_slot != 0) {
        return index_bytes_per_slot;
    }
    if (!has_index_cache || index_head_dim <= 0) {
        return 0;
    }
    return sizeof(ggml_fp16_t) * static_cast<size_t>(index_head_dim);
}

size_t PagedKVCache::GetBytesPerBlock() const {
    if (k_bytes_per_block != 0) {
        return k_bytes_per_block;
    }
    return GetBytesPerSlot() * BLOCK_SIZE;
}
size_t PagedKVCache::GetVBytesPerBlock() const {
    if (v_bytes_per_block != 0) {
        return v_bytes_per_block;
    }
    return GetVBytesPerSlot() * BLOCK_SIZE;
}
size_t PagedKVCache::GetIndexBytesPerBlock() const {
    if (index_bytes_per_block != 0) {
        return index_bytes_per_block;
    }
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

    if (use_block_allocator && k_allocator) {
        const size_t layer_offset =
            k_layer_stride_bytes != 0 ? k_layer_stride_bytes * static_cast<size_t>(layer)
                                      : GetBytesPerBlock() * static_cast<size_t>(max_blocks) *
                                            static_cast<size_t>(layer);
        const size_t block_offset = GetBytesPerBlock() * static_cast<size_t>(block_id);
        void* ptr = static_cast<char*>(k_allocator->ArenaBase()) + layer_offset + block_offset;
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

    if (use_block_allocator && v_allocator) {
        const size_t layer_offset =
            v_layer_stride_bytes != 0 ? v_layer_stride_bytes * static_cast<size_t>(layer)
                                      : GetVBytesPerBlock() * static_cast<size_t>(max_blocks) *
                                            static_cast<size_t>(layer);
        const size_t block_offset = GetVBytesPerBlock() * static_cast<size_t>(block_id);
        void* ptr = static_cast<char*>(v_allocator->ArenaBase()) + layer_offset + block_offset;
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

    const size_t layer_offset =
        index_layer_stride_bytes != 0
            ? index_layer_stride_bytes * static_cast<size_t>(layer)
            : GetIndexBytesPerBlock() * static_cast<size_t>(max_blocks) * static_cast<size_t>(layer);
    const size_t block_offset = GetIndexBytesPerBlock() * static_cast<size_t>(block_id);
    void* ptr = static_cast<char*>(index_allocator->ArenaBase()) + layer_offset + block_offset;
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

void PagedKVCache::FillBlockPtrsForLayer(const std::vector<int>& block_ids, int layer,
                                         std::vector<const void*>* k_blocks, std::vector<const void*>* v_blocks) const {
    if (!k_blocks || !v_blocks) return;
    k_blocks->assign(block_ids.size(), nullptr);
    v_blocks->assign(block_ids.size(), nullptr);
    if (layer < 0 || layer >= n_layer || !use_block_allocator || !k_allocator || !v_allocator) return;

    const auto* k_base = static_cast<const char*>(k_allocator->ArenaBase());
    const auto* v_base = static_cast<const char*>(v_allocator->ArenaBase());
    const size_t k_layer_offset =
        k_layer_stride_bytes != 0
            ? k_layer_stride_bytes * static_cast<size_t>(layer)
            : GetBytesPerBlock() * static_cast<size_t>(max_blocks) * static_cast<size_t>(layer);
    const size_t v_layer_offset =
        v_layer_stride_bytes != 0
            ? v_layer_stride_bytes * static_cast<size_t>(layer)
            : GetVBytesPerBlock() * static_cast<size_t>(max_blocks) * static_cast<size_t>(layer);
    const size_t k_block_stride = GetBytesPerBlock();
    const size_t v_block_stride = GetVBytesPerBlock();

    for (size_t i = 0; i < block_ids.size(); ++i) {
        const int block_id = block_ids[i];
        if (block_id < 0 || block_id >= max_blocks) continue;
        (*k_blocks)[i] = k_base + k_layer_offset + k_block_stride * static_cast<size_t>(block_id);
        (*v_blocks)[i] = v_base + v_layer_offset + v_block_stride * static_cast<size_t>(block_id);
    }
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
    RecordKVWriteSingleSlotUsage();

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    const int n = GetElementsPerSlot(layer);
    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            ggml_fp16_t* dst_head = dst + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            std::fill(dst_head, dst_head + head_dim, ggml_fp16_t(0));
            if (h < layer_n_head_kv) {
                const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
                densecore::simd::ConvertF32ToF16(dst_head, src_head, layer_head_dim);
            }
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        RecordKVWriteSlotFallbackUsage();
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        auto& padded_head = GetKVFloatScratch(static_cast<size_t>(head_dim));
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
        for (int h = 0; h < n_head_kv; ++h) {
            float* dst_head = dst + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            std::fill(dst_head, dst_head + head_dim, 0.0f);
            if (h < layer_n_head_kv) {
                const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
                densecore::simd::CopyF32(dst_head, src_head, layer_head_dim);
            }
        }
    }
}

void PagedKVCache::WriteVSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;
    RecordKVWriteSingleSlotUsage();

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    const int n = GetVElementsPerSlot(layer);
    if (cache_type == GGML_TYPE_F16) {
        auto* dst = static_cast<ggml_fp16_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            ggml_fp16_t* dst_head = dst + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            std::fill(dst_head, dst_head + v_head_dim, ggml_fp16_t(0));
            if (h < layer_n_head_kv) {
                const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
                densecore::simd::ConvertF32ToF16(dst_head, src_head, layer_head_dim);
            }
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        RecordKVWriteSlotFallbackUsage();
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        auto& padded_head = GetKVFloatScratch(static_cast<size_t>(v_head_dim));
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
        for (int h = 0; h < n_head_kv; ++h) {
            float* dst_head = dst + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            std::fill(dst_head, dst_head + v_head_dim, 0.0f);
            if (h < layer_n_head_kv) {
                const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
                densecore::simd::CopyF32(dst_head, src_head, layer_head_dim);
            }
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
    RecordKVReadSingleSlotUsage();

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetHeadDimForLayer(layer);
    if (cache_type == GGML_TYPE_F16) {
        const auto* src = static_cast<const ggml_fp16_t*>(ptr);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const ggml_fp16_t* src_head = src + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            densecore::simd::ConvertF16ToF32(out_head, src_head, layer_head_dim);
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        RecordKVReadSlotFallbackUsage();
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        auto& tmp_head = GetKVFloatScratch(static_cast<size_t>(head_dim));
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            type_traits->to_float(src_head, tmp_head.data(), head_dim);
            std::memcpy(out_head, tmp_head.data(), static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    } else {
        const auto* src = static_cast<const float*>(ptr);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const float* src_head = src + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            densecore::simd::CopyF32(out_head, src_head, layer_head_dim);
        }
    }
}

void PagedKVCache::ReadVSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;
    RecordKVReadSingleSlotUsage();

    const int layer_n_head_kv = GetHeadCountForLayer(layer);
    const int layer_head_dim = GetVHeadDimForLayer(layer);
    if (cache_type == GGML_TYPE_F16) {
        const auto* src = static_cast<const ggml_fp16_t*>(ptr);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const ggml_fp16_t* src_head = src + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            densecore::simd::ConvertF16ToF32(out_head, src_head, layer_head_dim);
        }
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        RecordKVReadSlotFallbackUsage();
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        auto& tmp_head = GetKVFloatScratch(static_cast<size_t>(v_head_dim));
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            type_traits->to_float(src_head, tmp_head.data(), v_head_dim);
            std::memcpy(out_head, tmp_head.data(), static_cast<size_t>(layer_head_dim) * sizeof(float));
        }
    } else {
        const auto* src = static_cast<const float*>(ptr);
        for (int h = 0; h < layer_n_head_kv; ++h) {
            const float* src_head = src + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(layer_head_dim);
            densecore::simd::CopyF32(out_head, src_head, layer_head_dim);
        }
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
        if (!UseKVBulkSlotPath()) {
            const int elems_per_slot = GetElementsPerSlot(layer);
            for (int i = 0; i < num_slots; ++i) {
                WriteKSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
            }
            return;
        }
        void* ptr = GetKSlotPtr(block_id, layer, start_slot);
        if (!ptr) return;
        RecordKVWriteBulkUsage(num_slots);
        if (cache_type == GGML_TYPE_F16) {
            WriteLayerSlotsToF16Storage(static_cast<ggml_fp16_t*>(ptr), data, num_slots, head_dim, n_head_kv,
                                        layer_head_dim, layer_n_head_kv);
            return;
        }
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            auto& padded = GetKVFloatScratch(static_cast<size_t>(GetElementsPerSlot()) * static_cast<size_t>(num_slots));
            PackLayerSlotsToStorageLayout(padded.data(), data, num_slots, head_dim, n_head_kv, layer_head_dim,
                                          layer_n_head_kv);
            ggml_quantize_chunk(cache_type, padded.data(), ptr, 0,
                                static_cast<int64_t>(num_slots) * static_cast<int64_t>(n_head_kv), head_dim, nullptr);
            return;
        }
        PackLayerSlotsToStorageLayout(static_cast<float*>(ptr), data, num_slots, head_dim, n_head_kv, layer_head_dim,
                                      layer_n_head_kv);
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
            auto& padded =
                GetKVFloatScratch(static_cast<size_t>(GetElementsPerSlot()) * static_cast<size_t>(num_slots));
            std::fill(padded.begin(), padded.begin() + static_cast<size_t>(GetElementsPerSlot() * num_slots), 0.0f);
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
        if (!UseKVBulkSlotPath()) {
            const int elems_per_slot = GetVElementsPerSlot(layer);
            for (int i = 0; i < num_slots; ++i) {
                WriteVSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
            }
            return;
        }
        void* ptr = GetVSlotPtr(block_id, layer, start_slot);
        if (!ptr) return;
        RecordKVWriteBulkUsage(num_slots);
        if (cache_type == GGML_TYPE_F16) {
            WriteLayerSlotsToF16Storage(static_cast<ggml_fp16_t*>(ptr), data, num_slots, v_head_dim, n_head_kv,
                                        layer_head_dim, layer_n_head_kv);
            return;
        }
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            auto& padded =
                GetKVFloatScratch(static_cast<size_t>(GetVElementsPerSlot()) * static_cast<size_t>(num_slots));
            PackLayerSlotsToStorageLayout(padded.data(), data, num_slots, v_head_dim, n_head_kv, layer_head_dim,
                                          layer_n_head_kv);
            ggml_quantize_chunk(cache_type, padded.data(), ptr, 0,
                                static_cast<int64_t>(num_slots) * static_cast<int64_t>(n_head_kv), v_head_dim, nullptr);
            return;
        }
        PackLayerSlotsToStorageLayout(static_cast<float*>(ptr), data, num_slots, v_head_dim, n_head_kv, layer_head_dim,
                                      layer_n_head_kv);
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
            auto& padded =
                GetKVFloatScratch(static_cast<size_t>(GetVElementsPerSlot()) * static_cast<size_t>(num_slots));
            std::fill(padded.begin(), padded.begin() + static_cast<size_t>(GetVElementsPerSlot() * num_slots), 0.0f);
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
        if (!UseKVBulkSlotPath()) {
            const int elems_per_slot = GetElementsPerSlot(layer);
            for (int i = 0; i < num_slots; ++i) {
                ReadKSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
            }
            return;
        }
        const void* ptr = const_cast<PagedKVCache*>(this)->GetKSlotPtr(block_id, layer, start_slot);
        if (!ptr) return;
        RecordKVReadBulkUsage(num_slots);
        if (cache_type == GGML_TYPE_F16) {
            ReadLayerSlotsFromF16Storage(out, static_cast<const ggml_fp16_t*>(ptr), num_slots, head_dim, n_head_kv,
                                         layer_head_dim, layer_n_head_kv);
            return;
        }
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            const auto* type_traits = ggml_get_type_traits(cache_type);
            if (!type_traits || !type_traits->to_float) return;
            const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
            const auto* src = static_cast<const uint8_t*>(ptr);
            auto& padded = GetKVFloatScratch(static_cast<size_t>(GetElementsPerSlot()) * static_cast<size_t>(num_slots));
            for (int row = 0; row < num_slots * n_head_kv; ++row) {
                type_traits->to_float(src + static_cast<size_t>(row) * head_stride,
                                      padded.data() + static_cast<size_t>(row) * static_cast<size_t>(head_dim),
                                      head_dim);
            }
            UnpackStorageSlotsToLayerLayout(out, padded.data(), num_slots, head_dim, n_head_kv, layer_head_dim,
                                            layer_n_head_kv);
            return;
        }
        UnpackStorageSlotsToLayerLayout(out, static_cast<const float*>(ptr), num_slots, head_dim, n_head_kv,
                                        layer_head_dim, layer_n_head_kv);
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
        auto& tmp_head = GetKVFloatScratch(static_cast<size_t>(head_dim));
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
        if (!UseKVBulkSlotPath()) {
            const int elems_per_slot = GetVElementsPerSlot(layer);
            for (int i = 0; i < num_slots; ++i) {
                ReadVSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
            }
            return;
        }
        const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, start_slot);
        if (!ptr) return;
        RecordKVReadBulkUsage(num_slots);
        if (cache_type == GGML_TYPE_F16) {
            ReadLayerSlotsFromF16Storage(out, static_cast<const ggml_fp16_t*>(ptr), num_slots, v_head_dim, n_head_kv,
                                         layer_head_dim, layer_n_head_kv);
            return;
        }
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            const auto* type_traits = ggml_get_type_traits(cache_type);
            if (!type_traits || !type_traits->to_float) return;
            const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
            const auto* src = static_cast<const uint8_t*>(ptr);
            auto& padded =
                GetKVFloatScratch(static_cast<size_t>(GetVElementsPerSlot()) * static_cast<size_t>(num_slots));
            for (int row = 0; row < num_slots * n_head_kv; ++row) {
                type_traits->to_float(src + static_cast<size_t>(row) * head_stride,
                                      padded.data() + static_cast<size_t>(row) * static_cast<size_t>(v_head_dim),
                                      v_head_dim);
            }
            UnpackStorageSlotsToLayerLayout(out, padded.data(), num_slots, v_head_dim, n_head_kv, layer_head_dim,
                                            layer_n_head_kv);
            return;
        }
        UnpackStorageSlotsToLayerLayout(out, static_cast<const float*>(ptr), num_slots, v_head_dim, n_head_kv,
                                        layer_head_dim, layer_n_head_kv);
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
        auto& tmp_head = GetKVFloatScratch(static_cast<size_t>(v_head_dim));
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
    if (v_ptr) densecore::simd::PrefetchRange(v_ptr, GetVBytesPerBlock());
}

void PagedKVCache::PrefetchNextLayer(const std::vector<int>& block_ids, int next_layer) const {
    if (next_layer >= n_layer) return;
    for (int block_id : block_ids) {
        PrefetchBlock(block_id, next_layer);
    }
}
