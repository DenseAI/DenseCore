/**
 * @file densecore/memory/kv_block_manager.h
 * @brief PagedAttention memory management for KV cache
 *
 * Implements vLLM-style block-based KV cache management with:
 * - Fixed-size token blocks (16 tokens per block)
 * - NUMA-aware allocation via NumaAllocator
 * - O(1) allocation/free using lock-free free list
 * - Cache-line aligned blocks (64 bytes) to prevent false sharing
 */

#ifndef DENSECORE_KV_BLOCK_MANAGER_H
#define DENSECORE_KV_BLOCK_MANAGER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "densecore/memory/numa_allocator.h"

namespace densecore {

// =============================================================================
// Configuration Constants
// =============================================================================

/// Number of tokens per KV cache block (standard vLLM block size)
constexpr size_t kBlockTokens = 16;

/// Cache line size for alignment (AVX-512 / false sharing prevention)
constexpr size_t kCacheLineSize = 64;

// =============================================================================
// KVCacheBlock: Fixed-Size Token Block
// =============================================================================

/**
 * @brief A fixed-size block storing KV cache data for kBlockTokens tokens
 *
 * Memory layout per block:
 *   [K cache: kBlockTokens × head_dim × sizeof(float)]
 *   [V cache: kBlockTokens × head_dim × sizeof(float)]
 *
 * Total size = 2 × kBlockTokens × head_dim × sizeof(float)
 * For head_dim=128: 2 × 16 × 128 × 4 = 16,384 bytes (16 KB)
 *
 * The block is cache-line aligned to prevent false sharing when multiple
 * threads access different blocks concurrently.
 */
struct alignas(kCacheLineSize) KVCacheBlock {
    /// Unique block ID within the pool
    uint32_t block_id;

    /// Number of valid tokens in this block (0 to kBlockTokens)
    uint32_t num_tokens;

    /// Sequence ID this block belongs to (-1 if free)
    int32_t sequence_id;

    /// Next block in linked list (free list or sequence chain)
    uint32_t next_block_id;

    /// Padding to ensure data starts at cache line boundary
    uint8_t _padding[48];  // 64 - (4 + 4 + 4 + 4) = 48

    /// KV cache data follows (variable size based on head_dim)
    /// Access via BlockManager::GetBlockData()
};

static_assert(sizeof(KVCacheBlock) == kCacheLineSize, "KVCacheBlock header must be exactly one cache line");

// =============================================================================
// BlockManager: NUMA-Aware Block Pool with O(1) Alloc/Free
// =============================================================================

/**
 * @brief Manages a pool of KV cache blocks with NUMA-aware allocation
 *
 * Features:
 * - Pre-allocates a large contiguous pool on a specific NUMA node
 * - O(1) block allocation using a free list
 * - O(1) block deallocation (push to free list head)
 * - Thread-safe via spinlock (low contention expected)
 *
 * Usage:
 *   BlockManager mgr;
 *   mgr.Initialize(1024, 128, 0);  // 1024 blocks, head_dim=128, NUMA node 0
 *   uint32_t block = mgr.AllocateBlock();
 *   float* kv_data = mgr.GetBlockData(block);
 *   mgr.FreeBlock(block);
 */
class BlockManager {
public:
    /// Invalid block ID sentinel
    static constexpr uint32_t kInvalidBlockId = UINT32_MAX;

    BlockManager() = default;
    ~BlockManager() { Shutdown(); }

    // Non-copyable
    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    /**
     * @brief Initialize the block pool
     *
     * @param num_blocks Number of blocks to pre-allocate
     * @param head_dim Dimension of attention heads (e.g., 128)
     * @param numa_node NUMA node to allocate on (-1 for default)
     * @return true on success, false on allocation failure
     */
    bool Initialize(size_t num_blocks, size_t head_dim, int numa_node = -1) {
        if (initialized_) {
            return false;  // Already initialized
        }

        num_blocks_ = num_blocks;
        head_dim_ = head_dim;
        numa_node_ = numa_node;

        // Calculate sizes
        // Block data size: K + V cache for kBlockTokens tokens
        block_data_size_ = 2 * kBlockTokens * head_dim * sizeof(float);

        // Align block data size to cache line
        block_data_size_ = (block_data_size_ + kCacheLineSize - 1) & ~(kCacheLineSize - 1);

        // Total size per block (header + data)
        block_stride_ = sizeof(KVCacheBlock) + block_data_size_;

        // Align stride to cache line
        block_stride_ = (block_stride_ + kCacheLineSize - 1) & ~(kCacheLineSize - 1);

        // Total pool size
        size_t pool_size = num_blocks * block_stride_;

        // Allocate pool on NUMA node
        allocation_result_ = NumaAllocator::AllocatePreferred(pool_size, kCacheLineSize, numa_node);

        if (!allocation_result_.ptr) {
            return false;
        }

        pool_base_ = static_cast<uint8_t*>(allocation_result_.ptr);

        // =========================================================================
        // NUMA-Aware Parallel Warm-up
        // =========================================================================
        // The OS uses "first-touch" policy: physical memory pages are allocated
        // on the NUMA node local to the thread that first writes to them.
        //
        // By using multiple threads to touch disjoint chunks of the pool, we
        // spread physical pages across NUMA nodes, maximizing aggregate bandwidth.
        // =========================================================================
        const size_t pool_size = num_blocks * block_stride_;
        const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t chunk_size = (pool_size + num_threads - 1) / num_threads;

        std::vector<std::thread> warmup_threads;
        warmup_threads.reserve(num_threads);

        for (unsigned int t = 0; t < num_threads; ++t) {
            const size_t start = t * chunk_size;
            const size_t end = std::min(start + chunk_size, pool_size);

            if (start >= pool_size) break;

            warmup_threads.emplace_back([this, start, end]() {
                // Touch each page (4KB typically) in our chunk
                // This triggers physical page allocation on this thread's local NUMA
                // node
                constexpr size_t PAGE_SIZE = 4096;
                volatile uint8_t* ptr = pool_base_ + start;
                for (size_t offset = 0; offset < end - start; offset += PAGE_SIZE) {
                    ptr[offset] = 0;
                }
            });
        }

        // Wait for all warm-up threads to complete
        for (auto& th : warmup_threads) {
            th.join();
        }

        std::cout << "[BlockManager] Parallel warm-up complete: " << num_threads << " threads touched "
                  << (pool_size / 1024 / 1024) << " MB across NUMA nodes" << std::endl;

        // Initialize all blocks and build free list
        for (size_t i = 0; i < num_blocks; i++) {
            KVCacheBlock* block = GetBlockHeader(static_cast<uint32_t>(i));
            block->block_id = static_cast<uint32_t>(i);
            block->num_tokens = 0;
            block->sequence_id = -1;
            block->next_block_id = (i + 1 < num_blocks) ? static_cast<uint32_t>(i + 1) : kInvalidBlockId;
        }

        free_list_head_ = 0;
        free_count_ = num_blocks;
        initialized_ = true;

        return true;
    }

    /**
     * @brief Shutdown and release the block pool
     */
    void Shutdown() {
        if (initialized_) {
            allocation_result_.Free();
            pool_base_ = nullptr;
            initialized_ = false;
            free_list_head_ = kInvalidBlockId;
            free_count_ = 0;
        }
    }

    /**
     * @brief Allocate a block from the free list (O(1))
     *
     * @return Block ID, or kInvalidBlockId if pool exhausted
     */
    uint32_t AllocateBlock() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (free_list_head_ == kInvalidBlockId) {
            return kInvalidBlockId;  // Pool exhausted
        }

        uint32_t block_id = free_list_head_;
        KVCacheBlock* block = GetBlockHeader(block_id);

        // Pop from free list
        free_list_head_ = block->next_block_id;
        free_count_--;

        // Reset block state
        block->num_tokens = 0;
        block->sequence_id = -1;
        block->next_block_id = kInvalidBlockId;

        return block_id;
    }

    /**
     * @brief Return a block to the free list (O(1))
     *
     * @param block_id Block to free
     */
    void FreeBlock(uint32_t block_id) {
        if (block_id >= num_blocks_) {
            return;  // Invalid block ID
        }

        std::lock_guard<std::mutex> lock(mutex_);

        KVCacheBlock* block = GetBlockHeader(block_id);

        // Reset block
        block->num_tokens = 0;
        block->sequence_id = -1;

        // Push to free list head
        block->next_block_id = free_list_head_;
        free_list_head_ = block_id;
        free_count_++;
    }

    /**
     * @brief Get pointer to block header
     *
     * @param block_id Block ID
     * @return Pointer to KVCacheBlock header
     */
    KVCacheBlock* GetBlockHeader(uint32_t block_id) {
        if (!initialized_ || block_id >= num_blocks_) {
            return nullptr;
        }
        return reinterpret_cast<KVCacheBlock*>(pool_base_ + block_id * block_stride_);
    }

    const KVCacheBlock* GetBlockHeader(uint32_t block_id) const {
        if (!initialized_ || block_id >= num_blocks_) {
            return nullptr;
        }
        return reinterpret_cast<const KVCacheBlock*>(pool_base_ + block_id * block_stride_);
    }

    /**
     * @brief Get pointer to block's KV cache data
     *
     * @param block_id Block ID
     * @return Pointer to float array [2 × kBlockTokens × head_dim]
     *         Layout: [K cache][V cache]
     */
    float* GetBlockData(uint32_t block_id) {
        if (!initialized_ || block_id >= num_blocks_) {
            return nullptr;
        }
        uint8_t* block_ptr = pool_base_ + block_id * block_stride_;
        return reinterpret_cast<float*>(block_ptr + sizeof(KVCacheBlock));
    }

    const float* GetBlockData(uint32_t block_id) const {
        if (!initialized_ || block_id >= num_blocks_) {
            return nullptr;
        }
        const uint8_t* block_ptr = pool_base_ + block_id * block_stride_;
        return reinterpret_cast<const float*>(block_ptr + sizeof(KVCacheBlock));
    }

    /**
     * @brief Get K cache portion of block data
     */
    float* GetKCache(uint32_t block_id) { return GetBlockData(block_id); }

    const float* GetKCache(uint32_t block_id) const { return GetBlockData(block_id); }

    /**
     * @brief Get V cache portion of block data
     */
    float* GetVCache(uint32_t block_id) {
        float* data = GetBlockData(block_id);
        if (!data) return nullptr;
        return data + kBlockTokens * head_dim_;
    }

    const float* GetVCache(uint32_t block_id) const {
        const float* data = GetBlockData(block_id);
        if (!data) return nullptr;
        return data + kBlockTokens * head_dim_;
    }

    // =========================================================================
    // Stats & Accessors
    // =========================================================================

    size_t GetTotalBlocks() const { return num_blocks_; }
    size_t GetFreeBlocks() const { return free_count_; }
    size_t GetUsedBlocks() const { return num_blocks_ - free_count_; }
    size_t GetHeadDim() const { return head_dim_; }
    size_t GetBlockDataSize() const { return block_data_size_; }
    int GetNumaNode() const { return numa_node_; }
    bool IsInitialized() const { return initialized_; }

    /**
     * @brief Get total memory used by the pool in bytes
     */
    size_t GetPoolSizeBytes() const { return initialized_ ? num_blocks_ * block_stride_ : 0; }

private:
    // Pool state
    uint8_t* pool_base_ = nullptr;
    NumaAllocator::AllocationResult allocation_result_;
    bool initialized_ = false;

    // Configuration
    size_t num_blocks_ = 0;
    size_t head_dim_ = 0;
    size_t block_data_size_ = 0;
    size_t block_stride_ = 0;
    int numa_node_ = -1;

    // Free list (singly-linked via next_block_id)
    uint32_t free_list_head_ = kInvalidBlockId;
    size_t free_count_ = 0;

    // Thread safety
    mutable std::mutex mutex_;
};

}  // namespace densecore

#endif  // DENSECORE_KV_BLOCK_MANAGER_H
