#include "kv_cache.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "block_allocator.h"
#include "densecore/license_guard.h"
#include "numa_allocator.h"
#include "simd_ops.h"

// ============================================================================
// BlockManager Implementation
// ============================================================================
namespace {
uint64_t GetTokenHashSalt() {
    static const uint64_t salt = []() {
        std::random_device rd;
        uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        if (seed == 0) {
            seed = 0x9e3779b97f4a7c15ULL;
        }
        return seed;
    }();
    return salt;
}
}  // namespace

BlockManager::BlockManager(int num_blocks, int block_size) : num_blocks(num_blocks), block_size(block_size) {
    // Initialize shards
    shards.reserve(NUM_SHARDS);
    for (int i = 0; i < NUM_SHARDS; ++i) {
        shards.push_back(std::make_unique<Shard>());
    }

    // Initialize physical blocks
    blocks.resize(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        blocks[i].id = i;
        blocks[i].ref_count = 0;
        blocks[i].num_filled_slots = 0;
        blocks[i].content_hash = 0;

        // Distribute blocks to shards
        shards[i % NUM_SHARDS]->free_blocks.push_back(i);
    }
}

std::vector<int> BlockManager::Allocate(int n) {
    if (n <= 0) return {};
    if (n == 1) {
        int id = AllocateSingle();
        if (id >= 0) return {id};
        return {};
    }

    // OSS-safe: if enterprise plugin is not loaded, guard returns allow.
    if (!DenseCoreEntLicenseGuardAllow()) {
        return {};
    }

    std::vector<int> allocated;
    allocated.reserve(n);

    // Try to satisfy full allocation from a single shard first (better locality)
    int start_shard = rand() % NUM_SHARDS;
    for (int i = 0; i < NUM_SHARDS; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        std::unique_lock<std::mutex> lock(shard.mu, std::try_to_lock);
        if (!lock.owns_lock()) continue;  // Skip contested shards for speed

        if ((int)shard.free_blocks.size() >= n) {
            for (int k = 0; k < n; ++k) {
                int block_id = shard.free_blocks.back();
                shard.free_blocks.pop_back();

                blocks[block_id].ref_count = 1;
                blocks[block_id].num_filled_slots = 0;
                blocks[block_id].content_hash = 0;
                allocated.push_back(block_id);
            }
            return allocated;
        }
    }

    // Fallback: Aggregate from multiple shards
    for (int i = 0; i < NUM_SHARDS && (int)allocated.size() < n; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        std::lock_guard<std::mutex> lock(shard.mu);
        while (!shard.free_blocks.empty() && (int)allocated.size() < n) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            allocated.push_back(block_id);
        }
    }

    // If failed to allocate enough, rollback
    if ((int)allocated.size() < n) {
        Free(allocated);
        return {};
    }

    return allocated;
}

int BlockManager::AllocateSingle() {
    // OSS-safe: if enterprise plugin is not loaded, guard returns allow.
    if (!DenseCoreEntLicenseGuardAllow()) {
        return -1;
    }

    // [P3 fix] Round-robin shard selection via atomic counter.
    // rand() is not thread-safe (global state, data race under TSAN).
    // An atomic counter gives uniform distribution without the race.
    static std::atomic<uint32_t> rr_counter{0};
    int start_shard = static_cast<int>(rr_counter.fetch_add(1, std::memory_order_relaxed) % NUM_SHARDS);

    for (int i = 0; i < NUM_SHARDS; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        // Try lock to avoid contention
        std::unique_lock<std::mutex> lock(shard.mu, std::try_to_lock);
        if (!lock.owns_lock()) continue;

        if (!shard.free_blocks.empty()) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            return block_id;
        }
    }

    // Retry with blocking lock if first pass failed
    for (int i = 0; i < NUM_SHARDS; ++i) {
        auto& shard = *shards[i];
        std::lock_guard<std::mutex> lock(shard.mu);
        if (!shard.free_blocks.empty()) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            return block_id;
        }
    }

    return -1;  // OOM
}

void BlockManager::Free(const std::vector<int>& block_ids) {
    for (int id : block_ids) {
        FreeSingle(id);
    }
}

void BlockManager::FreeSingle(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];

    // Step 1: Lock Shard and decrement ref_count
    {
        std::lock_guard<std::mutex> lock(shard.mu);
        blocks[block_id].ref_count--;

        if (blocks[block_id].ref_count > 0) return;
    }

    // Step 2: Complex case - remove from prefix cache (Requires Prefix Lock)
    // We released Shard Lock to avoid deadlock (Prefix -> Shard order)

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    std::lock_guard<std::mutex> s_lock(shard.mu);

    // Re-check ref_count in case it was resurrected by FindCachedBlock
    if (blocks[block_id].ref_count > 0) {
        return;  // Resurrected
    }

    // Remove from prefix cache
    if (blocks[block_id].content_hash != 0) {
        prefix_cache.erase(blocks[block_id].content_hash);
        blocks[block_id].content_hash = 0;
    }
    block_tokens.erase(block_id);

    // Add to free list
    shard.free_blocks.push_back(block_id);
}

int BlockManager::GetFreeBlockCount() {
    int total = 0;
    for (auto& shard : shards) {
        std::lock_guard<std::mutex> lock(shard->mu);
        total += shard->free_blocks.size();
    }
    return total;
}

int BlockManager::GetUsedBlockCount() {
    return num_blocks - GetFreeBlockCount();
}

// ============================================================================
// Copy-on-Write Implementation
// ============================================================================

int BlockManager::Fork(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return -1;

    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);

    if (blocks[block_id].ref_count <= 0) {
        return -1;  // Block not in use
    }

    blocks[block_id].ref_count++;
    return block_id;
}

int BlockManager::CopyOnWrite(int block_id, const std::function<void(int src, int dst)>& copy_callback) {
    if (block_id < 0 || block_id >= num_blocks) return -1;

    int shard_idx = block_id % NUM_SHARDS;

    // Check ref count (optimistic check)
    {
        std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);
        if (blocks[block_id].ref_count <= 1) {
            return block_id;
        }
    }

    // Allocate new block (can take from any shard)
    int new_block_id = AllocateSingle();
    if (new_block_id < 0) return -1;  // OOM

    // Re-check ref count and read metadata safely
    int old_num_filled_slots = 0;
    {
        std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);
        if (blocks[block_id].ref_count <= 1) {
            FreeSingle(new_block_id);
            return block_id;
        }
        old_num_filled_slots = blocks[block_id].num_filled_slots;
    }

    // Set metadata for new block
    {
        int new_shard_idx = new_block_id % NUM_SHARDS;
        std::lock_guard<std::mutex> lock(shards[new_shard_idx]->mu);
        blocks[new_block_id].num_filled_slots = old_num_filled_slots;
        blocks[new_block_id].content_hash = 0;
    }

    // Perform data copy using callback
    try {
        if (copy_callback) {
            copy_callback(block_id, new_block_id);
        }
    } catch (...) {
        FreeSingle(new_block_id);
        return -1;
    }

    // Decrement old block ref count only after successful copy
    {
        std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);
        blocks[block_id].ref_count--;
    }

    return new_block_id;
}

bool BlockManager::IsShared(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return false;
    int shard_idx = block_id % NUM_SHARDS;
    // const_cast needed because vector<unique_ptr> is const in const method?
    // No, but accessing mutex is mutable.
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    return blocks[block_id].ref_count > 1;
}

int BlockManager::GetRefCount(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return 0;
    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    return blocks[block_id].ref_count;
}

// ============================================================================
// Slot Management
// ============================================================================

void BlockManager::SetFilledSlots(int block_id, int num_slots) {
    if (block_id < 0 || block_id >= num_blocks) return;
    // Metadata update should be protected
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);
    blocks[block_id].num_filled_slots = std::min(num_slots, block_size);
}

int BlockManager::GetFilledSlots(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return 0;
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> lock(shards[shard_idx]->mu);
    return blocks[block_id].num_filled_slots;
}

// ============================================================================
// Prefix Caching
// ============================================================================

int BlockManager::FindCachedBlock(uint64_t hash) {
    if (hash == 0) return -1;

    std::lock_guard<std::mutex> p_lock(prefix_mu);

    auto it = prefix_cache.find(hash);
    if (it == prefix_cache.end()) {
        return -1;
    }

    int block_id = it->second;
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    // Verify block is still valid
    if (blocks[block_id].ref_count <= 0) {
        // Should have been removed by FreeSingle, but just in case
        prefix_cache.erase(it);
        block_tokens.erase(block_id);
        return -1;
    }

    // Fork the block
    blocks[block_id].ref_count++;
    return block_id;
}

int BlockManager::FindCachedBlockWithVerification(uint64_t hash, const int* tokens, int n_tokens) {
    if (hash == 0 || !tokens || n_tokens <= 0) return -1;

    std::lock_guard<std::mutex> p_lock(prefix_mu);

    auto it = prefix_cache.find(hash);
    if (it == prefix_cache.end()) {
        return -1;
    }

    int block_id = it->second;
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    if (blocks[block_id].ref_count <= 0) {
        prefix_cache.erase(it);
        block_tokens.erase(block_id);
        return -1;
    }

    // Verify tokens
    auto tokens_it = block_tokens.find(block_id);
    if (tokens_it == block_tokens.end()) {
        blocks[block_id].ref_count++;
        return block_id;
    }

    const std::vector<int>& stored_tokens = tokens_it->second;
    if (static_cast<int>(stored_tokens.size()) != n_tokens) return -1;

    for (int i = 0; i < n_tokens; ++i) {
        if (stored_tokens[i] != tokens[i]) return -1;
    }

    blocks[block_id].ref_count++;
    return block_id;
}

void BlockManager::RegisterPrefixBlock(int block_id, uint64_t hash) {
    if (block_id < 0 || block_id >= num_blocks || hash == 0) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    blocks[block_id].content_hash = hash;
    prefix_cache[hash] = block_id;
}

void BlockManager::RegisterPrefixBlockWithTokens(int block_id, uint64_t hash, const int* tokens, int n_tokens) {
    if (block_id < 0 || block_id >= num_blocks || hash == 0) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    blocks[block_id].content_hash = hash;
    prefix_cache[hash] = block_id;

    if (tokens && n_tokens > 0) {
        block_tokens[block_id] = std::vector<int>(tokens, tokens + n_tokens);
    }
}

void BlockManager::UnregisterPrefixBlock(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    if (blocks[block_id].content_hash != 0) {
        prefix_cache.erase(blocks[block_id].content_hash);
        blocks[block_id].content_hash = 0;
    }
    block_tokens.erase(block_id);
}

uint64_t BlockManager::ComputeTokenHash(const int* tokens, int n_tokens) {
    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL ^ GetTokenHashSalt();
    for (int i = 0; i < n_tokens; i++) {
        hash ^= (uint64_t)tokens[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ============================================================================
// PagedKVCache Implementation
// ============================================================================

PagedKVCache::~PagedKVCache() {
    // Note: ggml_free does not call free() on mem_buffer if it was provided
    // externally, so we must free our NUMA buffer ourselves AFTER ggml_free
    if (ctx) ggml_free(ctx);

    // Block allocators are unique_ptr, will be auto-deleted
    // k_allocator.reset() and v_allocator.reset() called automatically

    if (block_manager) delete block_manager;
}

size_t PagedKVCache::GetBytesPerSlot() const {
    const int64_t elements = static_cast<int64_t>(head_dim) * static_cast<int64_t>(n_head_kv);
    if (elements <= 0) {
        return 0;
    }

    if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        // Keep each KV head in an independently quantized row so per-head stride is
        // stable even when head_dim is quantized per-KV-head.
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
    if (block_id < 0 || block_id >= max_blocks) return nullptr;
    if (layer < 0 || layer >= n_layer) return nullptr;

    // Layout: [head_dim, n_head_kv, BLOCK_SIZE, num_blocks * n_layer]
    // Block index in 4th dimension = block_id * n_layer + layer
    int64_t block_index = (int64_t)block_id * n_layer + layer;
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
    if (block_id < 0 || block_id >= max_blocks) return nullptr;
    if (layer < 0 || layer >= n_layer) return nullptr;

    int64_t block_index = (int64_t)block_id * n_layer + layer;
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
    if (block_id < 0 || block_id >= max_blocks) return nullptr;
    if (layer < 0 || layer >= n_layer) return nullptr;

    int64_t block_index = (int64_t)block_id * n_layer + layer;
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
    return (char*)block_ptr + (size_t)slot * GetBytesPerSlot();
}

void* PagedKVCache::GetVSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetVBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE) return nullptr;
    return (char*)block_ptr + (size_t)slot * GetVBytesPerSlot();
}

void* PagedKVCache::GetIndexSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetIndexBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE) return nullptr;
    return (char*)block_ptr + (size_t)slot * GetIndexBytesPerSlot();
}

void PagedKVCache::CopyBlockData(int src_block_id, int dst_block_id) {
    if (src_block_id == dst_block_id) return;
    if (src_block_id < 0 || dst_block_id < 0) return;

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
    if (src_block_id == dst_block_id) return;
    if (src_block_id < 0 || dst_block_id < 0) return;
    if (layer < 0 || layer >= n_layer) return;

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
            if (k_src) {
                memcpy(k_out->data() + offset_k, k_src, bytes_per_k_block);
            }
            if (v_src) {
                memcpy(v_out->data() + offset_v, v_src, bytes_per_v_block);
            }
            if (index_src && total_index_bytes > 0) {
                size_t index_base = total_k_bytes;
                size_t offset_i =
                    index_base + (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_index_block;
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
            if (k_dst) {
                memcpy(k_dst, k_in.data() + offset_k, bytes_per_k_block);
            }
            if (v_dst) {
                memcpy(v_dst, v_in.data() + offset_v, bytes_per_v_block);
            }
            if (index_dst && total_index_bytes > 0 && k_in.size() >= total_k_bytes + total_index_bytes) {
                size_t index_base = total_k_bytes;
                size_t offset_i =
                    index_base + (i * static_cast<size_t>(n_layer) + static_cast<size_t>(layer)) * bytes_per_index_block;
                memcpy(index_dst, k_in.data() + offset_i, bytes_per_index_block);
            }
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

namespace {
bool IsEnvEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0;
}

bool TryEnableHugePages(void* ptr, size_t size, const char* label) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (!ptr || size == 0) return false;
    if (madvise(ptr, size, MADV_HUGEPAGE) == 0) {
        std::cout << "[KVCache] Hugepages advised for " << label << " (" << (size / 1024 / 1024) << " MB)" << std::endl;
        return true;
    }
    std::cerr << "[KVCache] Hugepages advise failed for " << label << " (" << (size / 1024 / 1024)
              << " MB): " << std::strerror(errno) << std::endl;
    return false;
#elif defined(__linux__)
    (void)ptr;
    (void)size;
    std::cout << "[KVCache] Hugepages advise unsupported (MADV_HUGEPAGE unavailable) for " << label << std::endl;
    return false;
#else
    (void)ptr;
    (void)size;
    std::cout << "[KVCache] Hugepages advise unsupported on this platform for " << label << std::endl;
    return false;
#endif
}
}  // namespace

PagedKVCache* InitPagedKVCache(TransformerModel* model, int max_num_seqs, int max_seq_len, ggml_type type,
                               int numa_node_id) {
    PagedKVCache* cache = new PagedKVCache();
    if (max_num_seqs <= 0 || max_seq_len <= 0) {
        std::cerr << "[KVCache] Error: Invalid KV cache dimensions (max_num_seqs=" << max_num_seqs
                  << ", max_seq_len=" << max_seq_len << ")" << std::endl;
        delete cache;
        return nullptr;
    }

    // llama.cpp style: Use pre-computed head dimension from hparams
    cache->head_dim = model->hparams.n_embd_head_k;
    cache->v_head_dim = model->hparams.n_embd_head_v > 0 ? model->hparams.n_embd_head_v : model->hparams.n_embd_head_k;
    cache->index_head_dim = model->arch_flags.is_glm_dsa ? model->glm_index_head_dim : 0;
    cache->n_head_kv = model->hparams.n_head_kv;
    cache->n_layer = model->hparams.n_layer;
    cache->cache_type = type;
    cache->numa_node_id = numa_node_id;
    cache->has_index_cache = model->arch_flags.is_glm_dsa && cache->index_head_dim > 0;

    // Validate supported types for KV cache.
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q4_0) {
        std::cerr << "[KVCache] Warning: Unsupported cache type, falling back to F16" << std::endl;
        type = GGML_TYPE_F16;
        cache->cache_type = type;
    }

    if (ggml_is_quantized(type) &&
        ((cache->head_dim % ggml_blck_size(type)) != 0 || (cache->v_head_dim % ggml_blck_size(type)) != 0)) {
        std::cerr << "[KVCache] Warning: K/V head dims (" << cache->head_dim << ", " << cache->v_head_dim
                  << ") are not divisible by "
                  << ggml_blck_size(type) << " for quantized cache type " << ggml_type_name(type)
                  << ", falling back to F16" << std::endl;
        type = GGML_TYPE_F16;
        cache->cache_type = type;
    }

    // Calculate total blocks needed
    int64_t total_tokens = static_cast<int64_t>(max_num_seqs) * static_cast<int64_t>(max_seq_len);
    int64_t max_blocks = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (max_blocks > std::numeric_limits<int>::max()) {
        std::cerr << "[KVCache] Error: Requested KV cache too large (" << max_blocks << " blocks)." << std::endl;
        delete cache;
        return nullptr;
    }
    cache->max_blocks = static_cast<int>(max_blocks);

    // Initialize BlockManager
    cache->block_manager = new BlockManager(cache->max_blocks, BLOCK_SIZE);

    // Calculate memory size per tensor (K or V) from runtime block stride to keep
    // quantized layouts (e.g., Q8_0 with padded rows) exact.
    int64_t n_layer_blocks = (int64_t)cache->max_blocks * cache->n_layer;
    size_t k_block_stride = cache->GetBytesPerBlock();
    size_t v_block_stride = cache->GetVBytesPerBlock();
    size_t index_block_stride = cache->GetIndexBytesPerBlock();
    size_t total_logical_blocks = static_cast<size_t>(n_layer_blocks);
    size_t k_tensor_size = k_block_stride * total_logical_blocks;
    size_t v_tensor_size = v_block_stride * total_logical_blocks;
    size_t index_tensor_size = cache->has_index_cache ? index_block_stride * total_logical_blocks : 0;
    size_t total_size = k_tensor_size + v_tensor_size + index_tensor_size;

    // ==========================================================================
    // KVBlockAllocator-based memory pool (PRIMARY PATH)
    // Pre-allocates a single contiguous arena per K/V to eliminate fragmentation
    // ==========================================================================

    const bool use_hugepages = IsEnvEnabled("DENSECORE_USE_HUGEPAGES");
    const bool strict_numa = (numa_node_id >= 0);

    // Block stride: bytes per "block" in our linear arena
    // Each block_id maps to: block_id * n_layer + layer_idx
    // So total "logical blocks" = max_blocks * n_layer

    // Create K allocator with NUMA awareness
    cache->k_allocator = std::make_unique<densecore::KVBlockAllocator>(total_logical_blocks, k_block_stride, 64,
                                                                       numa_node_id, use_hugepages, strict_numa);

    // Create V allocator with NUMA awareness
    cache->v_allocator = std::make_unique<densecore::KVBlockAllocator>(total_logical_blocks, v_block_stride, 64,
                                                                       numa_node_id, use_hugepages, strict_numa);

    if (cache->has_index_cache) {
        cache->index_allocator = std::make_unique<densecore::KVBlockAllocator>(total_logical_blocks, index_block_stride,
                                                                               64, numa_node_id, use_hugepages,
                                                                               strict_numa);
    }

    if (!cache->k_allocator->IsValid() || !cache->v_allocator->IsValid() ||
        (cache->has_index_cache && (!cache->index_allocator || !cache->index_allocator->IsValid()))) {
        std::cerr << "[KVCache] Error: Failed to allocate " << (total_size / 1024 / 1024)
                  << " MB for KV cache via block allocators. Try reducing max_seq_len." << std::endl;
        delete cache;
        return nullptr;
    }

    // Enable block allocator mode
    cache->use_block_allocator = true;

    // ==========================================================================
    // ALIGNMENT CHECK: Verify 64-byte alignment for AVX-512 safety
    // ==========================================================================
    DENSECORE_ASSERT_ALIGNED_64(cache->k_allocator->ArenaBase());
    DENSECORE_ASSERT_ALIGNED_64(cache->v_allocator->ArenaBase());

    bool k_hugepages_enabled = false;
    bool v_hugepages_enabled = false;
    bool index_hugepages_enabled = false;
    if (use_hugepages) {
        k_hugepages_enabled =
            TryEnableHugePages(cache->k_allocator->ArenaBase(), cache->k_allocator->ArenaSize(), "KV cache K arena");
        v_hugepages_enabled =
            TryEnableHugePages(cache->v_allocator->ArenaBase(), cache->v_allocator->ArenaSize(), "KV cache V arena");
        if (cache->has_index_cache && cache->index_allocator) {
            index_hugepages_enabled = TryEnableHugePages(cache->index_allocator->ArenaBase(),
                                                         cache->index_allocator->ArenaSize(), "KV cache index arena");
        }
    }

    // Type name for logging
    const char* type_name = "F32";
    if (type == GGML_TYPE_F16)
        type_name = "F16";
    else if (type == GGML_TYPE_Q8_0)
        type_name = "Q8_0 (INT8)";
    else if (type == GGML_TYPE_Q4_0)
        type_name = "Q4_0 (INT4)";

    std::cout << "[KVCache] Initialized PagedKVCache (Block Allocator):" << std::endl;
    std::cout << "  - max_blocks: " << cache->max_blocks << std::endl;
    std::cout << "  - block_size: " << BLOCK_SIZE << " tokens" << std::endl;
    std::cout << "  - n_layer: " << cache->n_layer << std::endl;
    std::cout << "  - k_head_dim: " << cache->head_dim << std::endl;
    std::cout << "  - v_head_dim: " << cache->v_head_dim << std::endl;
    std::cout << "  - n_head_kv: " << cache->n_head_kv << std::endl;
    std::cout << "  - cache_type: " << type_name << std::endl;
    std::cout << "  - k_arena: " << (cache->k_allocator->ArenaSize() / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - v_arena: " << (cache->v_allocator->ArenaSize() / 1024 / 1024) << " MB" << std::endl;
    if (cache->has_index_cache && cache->index_allocator) {
        std::cout << "  - index_head_dim: " << cache->index_head_dim << std::endl;
        std::cout << "  - index_arena: " << (cache->index_allocator->ArenaSize() / 1024 / 1024) << " MB"
                  << std::endl;
    }
    std::cout << "  - total_memory: " << (total_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - allocation_mode: BLOCK_ALLOCATOR (zero-fragmentation)" << std::endl;
    std::cout << "  - hugepages: " << (use_hugepages ? "requested" : "disabled") << std::endl;
    if (use_hugepages) {
        std::cout << "  - hugepages_k: " << (k_hugepages_enabled ? "enabled" : "not enabled") << std::endl;
        std::cout << "  - hugepages_v: " << (v_hugepages_enabled ? "enabled" : "not enabled") << std::endl;
        if (cache->has_index_cache) {
            std::cout << "  - hugepages_index: " << (index_hugepages_enabled ? "enabled" : "not enabled")
                      << std::endl;
        }
    }
    if (numa_node_id >= 0) {
        std::cout << "  - numa_node: " << numa_node_id << " (strict)" << std::endl;
    }

    return cache;
}

// ============================================================================
// KV Cache Read/Write Operations (vLLM-style)
// FP16 is the primary format (similar to FP8 on GPU)
// ============================================================================

bool PagedKVCache::IsQuantized() const {
    return ggml_is_quantized(cache_type);
}

int PagedKVCache::GetElementsPerSlot() const {
    return head_dim * n_head_kv;
}

int PagedKVCache::GetVElementsPerSlot() const {
    return v_head_dim * n_head_kv;
}

void PagedKVCache::WriteKSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetKSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF32ToF16((ggml_fp16_t*)ptr, data, n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            void* dst_head = dst + static_cast<size_t>(h) * head_stride;
            ggml_quantize_chunk(cache_type, src_head, dst_head, 0, 1, head_dim, nullptr);
        }
    } else {
        densecore::simd::CopyF32((float*)ptr, data, n);
    }
}

void PagedKVCache::WriteVSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int n = GetVElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF32ToF16((ggml_fp16_t*)ptr, data, n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        auto* dst = static_cast<uint8_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            const float* src_head = data + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            void* dst_head = dst + static_cast<size_t>(h) * head_stride;
            ggml_quantize_chunk(cache_type, src_head, dst_head, 0, 1, v_head_dim, nullptr);
        }
    } else {
        densecore::simd::CopyF32((float*)ptr, data, n);
    }
}

void PagedKVCache::WriteIndexSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetIndexSlotPtr(block_id, layer, slot);
    if (!ptr || !has_index_cache || index_head_dim <= 0) return;
    densecore::simd::ConvertF32ToF16((ggml_fp16_t*)ptr, data, index_head_dim);
}

void PagedKVCache::ReadKSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetKSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, (const ggml_fp16_t*)ptr, n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
            type_traits->to_float(src_head, out_head, head_dim);
        }
    } else {
        densecore::simd::CopyF32(out, (const float*)ptr, n);
    }
}

void PagedKVCache::ReadVSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, slot);
    if (!ptr) return;

    const int n = GetVElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, (const ggml_fp16_t*)ptr, n);
    } else if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
        const auto* type_traits = ggml_get_type_traits(cache_type);
        const size_t head_stride = ggml_row_size(cache_type, static_cast<int64_t>(v_head_dim));
        const auto* src = static_cast<const uint8_t*>(ptr);
        for (int h = 0; h < n_head_kv; ++h) {
            const void* src_head = src + static_cast<size_t>(h) * head_stride;
            float* out_head = out + static_cast<size_t>(h) * static_cast<size_t>(v_head_dim);
            type_traits->to_float(src_head, out_head, v_head_dim);
        }
    } else {
        densecore::simd::CopyF32(out, (const float*)ptr, n);
    }
}

void PagedKVCache::ReadIndexSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetIndexSlotPtr(block_id, layer, slot);
    if (!ptr || !has_index_cache || index_head_dim <= 0) return;
    densecore::simd::ConvertF16ToF32(out, (const ggml_fp16_t*)ptr, index_head_dim);
}

void PagedKVCache::WriteKSlots(int block_id, int layer, int start_slot, int num_slots, const float* data) {
    if (num_slots <= 0 || !data) return;
    if (start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int elems_per_slot = GetElementsPerSlot();
    for (int i = 0; i < num_slots; ++i) {
        WriteKSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
    }
}

void PagedKVCache::WriteVSlots(int block_id, int layer, int start_slot, int num_slots, const float* data) {
    if (num_slots <= 0 || !data) return;
    if (start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int elems_per_slot = GetElementsPerSlot();
    for (int i = 0; i < num_slots; ++i) {
        WriteVSlot(block_id, layer, start_slot + i, data + static_cast<size_t>(i) * elems_per_slot);
    }
}

void PagedKVCache::ReadKSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const {
    if (num_slots <= 0 || !out) return;
    if (start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int elems_per_slot = GetElementsPerSlot();
    for (int i = 0; i < num_slots; ++i) {
        ReadKSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
    }
}

void PagedKVCache::ReadVSlots(int block_id, int layer, int start_slot, int num_slots, float* out) const {
    if (num_slots <= 0 || !out) return;
    if (start_slot < 0 || start_slot + num_slots > BLOCK_SIZE) return;
    const int elems_per_slot = GetElementsPerSlot();
    for (int i = 0; i < num_slots; ++i) {
        ReadVSlot(block_id, layer, start_slot + i, out + static_cast<size_t>(i) * elems_per_slot);
    }
}

// ============================================================================
// SIMD Prefetch Operations
// ============================================================================

void PagedKVCache::PrefetchBlock(int block_id, int layer) const {
    const void* k_ptr = const_cast<PagedKVCache*>(this)->GetKBlockPtr(block_id, layer);
    const void* v_ptr = const_cast<PagedKVCache*>(this)->GetVBlockPtr(block_id, layer);

    if (k_ptr) {
        densecore::simd::PrefetchRange(k_ptr, GetBytesPerBlock());
    }
    if (v_ptr) {
        densecore::simd::PrefetchRange(v_ptr, GetBytesPerBlock());
    }
}

void PagedKVCache::PrefetchNextLayer(const std::vector<int>& block_ids, int next_layer) const {
    if (next_layer >= n_layer) return;

    for (int block_id : block_ids) {
        PrefetchBlock(block_id, next_layer);
    }
}
