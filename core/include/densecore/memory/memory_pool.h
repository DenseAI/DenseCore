/**
 * @file densecore/memory/memory_pool.h
 * @brief Memory pool for efficient scratch buffer management
 *
 * Reduces allocation overhead during inference by:
 * - Pre-allocating large scratch buffers
 * - Arena allocation within forward passes
 * - Thread-local pools for parallel execution
 */

#ifndef DENSECORE_MEMORY_POOL_H
#define DENSECORE_MEMORY_POOL_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace densecore {

// Alignment for SIMD operations
constexpr size_t MEMORY_ALIGNMENT = 64;

/**
 * Align size up to specified alignment
 */
inline size_t AlignUp(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * Simple arena allocator for scratch memory
 * Fast O(1) allocation, O(1) reset, no individual free
 */
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t initial_size = 64 * 1024 * 1024)
        : buffer_(nullptr), size_(0), used_(0), capacity_(0) {
        Reserve(initial_size);
    }

    ~ArenaAllocator() {
        if (buffer_) {
#ifdef _WIN32
            _aligned_free(buffer_);
#else
            free(buffer_);
#endif
        }
    }

    // Non-copyable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Movable
    ArenaAllocator(ArenaAllocator&& other) noexcept
        : buffer_(other.buffer_), size_(other.size_), used_(other.used_), capacity_(other.capacity_) {
        other.buffer_ = nullptr;
        other.size_ = other.used_ = other.capacity_ = 0;
    }

    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept {
        if (this != &other) {
            // Free existing buffer
            if (buffer_) {
#ifdef _WIN32
                _aligned_free(buffer_);
#else
                free(buffer_);
#endif
            }
            // Take ownership
            buffer_ = other.buffer_;
            size_ = other.size_;
            used_ = other.used_;
            capacity_ = other.capacity_;
            // Clear other
            other.buffer_ = nullptr;
            other.size_ = other.used_ = other.capacity_ = 0;
        }
        return *this;
    }

    /**
     * Allocate aligned memory from the arena
     */
    void* Allocate(size_t size, size_t alignment = MEMORY_ALIGNMENT) {
        size_t aligned_used = AlignUp(used_, alignment);
        size_t new_used = aligned_used + size;

        if (new_used > capacity_) {
            // Need to grow
            size_t new_capacity = std::max(capacity_ * 2, new_used);
            Reserve(new_capacity);
        }

        void* ptr = static_cast<char*>(buffer_) + aligned_used;
        used_ = new_used;
        size_ = std::max(size_, used_);
        return ptr;
    }

    /**
     * Allocate typed array
     */
    template <typename T> T* Allocate(size_t count) { return static_cast<T*>(Allocate(count * sizeof(T), alignof(T))); }

    /**
     * Reset arena for reuse (O(1) - just reset pointer)
     */
    void Reset() { used_ = 0; }

    /**
     * Get current usage
     */
    size_t Used() const { return used_; }

    /**
     * Get high water mark
     */
    size_t Size() const { return size_; }

    /**
     * Get capacity
     */
    size_t Capacity() const { return capacity_; }

private:
    void Reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        void* new_buffer;
#ifdef _WIN32
        new_buffer = _aligned_malloc(new_capacity, MEMORY_ALIGNMENT);
#else
        if (posix_memalign(&new_buffer, MEMORY_ALIGNMENT, new_capacity) != 0) {
            new_buffer = nullptr;
        }
#endif

        if (!new_buffer) {
            throw std::bad_alloc();
        }

        if (buffer_ && used_ > 0) {
            memcpy(new_buffer, buffer_, used_);
        }

        if (buffer_) {
#ifdef _WIN32
            _aligned_free(buffer_);
#else
            free(buffer_);
#endif
        }

        buffer_ = new_buffer;
        capacity_ = new_capacity;
    }

    void* buffer_;
    size_t size_;      // High water mark
    size_t used_;      // Current allocation point
    size_t capacity_;  // Total buffer size
};

/**
 * Fixed-size memory block pool
 * For frequently allocated same-size buffers
 */
class BlockPool {
public:
    BlockPool(size_t block_size, size_t initial_blocks = 16) : block_size_(AlignUp(block_size, MEMORY_ALIGNMENT)) {
        for (size_t i = 0; i < initial_blocks; i++) {
            void* block;
#ifdef _WIN32
            block = _aligned_malloc(block_size_, MEMORY_ALIGNMENT);
#else
            if (posix_memalign(&block, MEMORY_ALIGNMENT, block_size_) != 0) {
                block = nullptr;
            }
#endif
            if (block) {
                free_blocks_.push_back(block);
                all_blocks_.push_back(block);
            }
        }
    }

    ~BlockPool() {
        for (void* block : all_blocks_) {
#ifdef _WIN32
            _aligned_free(block);
#else
            free(block);
#endif
        }
    }

    void* Allocate() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (free_blocks_.empty()) {
            // Grow pool
            void* block;
#ifdef _WIN32
            block = _aligned_malloc(block_size_, MEMORY_ALIGNMENT);
#else
            if (posix_memalign(&block, MEMORY_ALIGNMENT, block_size_) != 0) {
                block = nullptr;
            }
#endif
            if (!block) return nullptr;
            all_blocks_.push_back(block);
            return block;
        }

        void* block = free_blocks_.back();
        free_blocks_.pop_back();
        return block;
    }

    void Free(void* block) {
        if (!block) return;
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_.push_back(block);
    }

    size_t BlockSize() const { return block_size_; }
    size_t TotalBlocks() const { return all_blocks_.size(); }
    size_t FreeBlocks() const { return free_blocks_.size(); }

private:
    size_t block_size_;
    std::vector<void*> free_blocks_;
    std::vector<void*> all_blocks_;
    std::mutex mutex_;
};

/**
 * Inference scratch buffer pool
 * Manages temporary buffers needed during forward pass
 */
class InferenceScratch {
public:
    InferenceScratch() = default;

    /**
     * Initialize for specific model dimensions
     */
    void Initialize(int n_embd, int n_head, int n_head_kv, int n_layer, int max_batch_tokens, int max_seq_len) {
        n_embd_ = n_embd;
        n_head_ = n_head;
        n_head_kv_ = n_head_kv;
        n_layer_ = n_layer;
        max_batch_tokens_ = max_batch_tokens;
        max_seq_len_ = max_seq_len;

        // Calculate buffer sizes
        size_t qkv_size = (size_t)max_batch_tokens * n_embd * sizeof(float);
        size_t attn_size = (size_t)max_batch_tokens * max_seq_len * sizeof(float);
        size_t ffn_size = (size_t)max_batch_tokens * n_embd * 4 * sizeof(float);

        // Pre-allocate main arena
        size_t total_size = qkv_size * 3         // Q, K, V
                            + attn_size          // Attention scores
                            + qkv_size           // Attention output
                            + ffn_size           // FFN intermediate
                            + 64 * 1024 * 1024;  // Extra buffer

        arena_.Reset();
        if (arena_.Capacity() < total_size) {
            arena_ = ArenaAllocator(total_size);
        }
    }

    /**
     * Allocate temporary buffer for current forward pass
     */
    template <typename T> T* Allocate(size_t count) { return arena_.Allocate<T>(count); }

    /**
     * Reset for next forward pass
     */
    void Reset() { arena_.Reset(); }

    /**
     * Get memory usage stats
     */
    size_t MemoryUsed() const { return arena_.Used(); }
    size_t MemoryCapacity() const { return arena_.Capacity(); }

private:
    ArenaAllocator arena_;
    int n_embd_ = 0;
    int n_head_ = 0;
    int n_head_kv_ = 0;
    int n_layer_ = 0;
    int max_batch_tokens_ = 0;
    int max_seq_len_ = 0;
};

/**
 * Global memory pool singleton
 */
class MemoryPoolManager {
public:
    static MemoryPoolManager& Instance() {
        static MemoryPoolManager instance;
        return instance;
    }

    /**
     * Get thread-local scratch buffer
     */
    InferenceScratch& GetScratch() {
        // Thread-local scratch
        static thread_local InferenceScratch scratch;
        return scratch;
    }

    /**
     * Get shared block pool for specific size
     */
    BlockPool& GetBlockPool(size_t block_size) {
        std::lock_guard<std::mutex> lock(pools_mutex_);

        // Find existing pool
        for (auto& pool : block_pools_) {
            if (pool->BlockSize() == block_size) {
                return *pool;
            }
        }

        // Create new pool
        block_pools_.push_back(std::make_unique<BlockPool>(block_size));
        return *block_pools_.back();
    }

    /**
     * Print memory stats
     */
    void PrintStats() const {
        printf("[MemoryPool] Block pools:\n");
        for (const auto& pool : block_pools_) {
            printf("  - Size %zu: %zu/%zu blocks free\n", pool->BlockSize(), pool->FreeBlocks(), pool->TotalBlocks());
        }
    }

private:
    MemoryPoolManager() = default;

    std::vector<std::unique_ptr<BlockPool>> block_pools_;
    std::mutex pools_mutex_;
};

/**
 * RAII wrapper for scratch allocation
 */
class ScopedScratch {
public:
    ScopedScratch() : scratch_(MemoryPoolManager::Instance().GetScratch()) {}

    ~ScopedScratch() { scratch_.Reset(); }

    template <typename T> T* Allocate(size_t count) { return scratch_.Allocate<T>(count); }

private:
    InferenceScratch& scratch_;
};

}  // namespace densecore

#endif  // DENSECORE_MEMORY_POOL_H
