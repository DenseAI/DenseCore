/**
 * @file densecore/memory/block_allocator.h
 * @brief Fixed-Size Block Allocator for KV Cache Memory Pool
 *
 * Implements a slab/arena allocator to eliminate heap fragmentation in
 * long-running inference servers. Pre-allocates a contiguous memory chunk
 * and manages blocks via a free list for O(1) allocation/deallocation.
 *
 * Features:
 * - 64-byte aligned allocation for AVX-512 compatibility
 * - NUMA-aware allocation for multi-socket systems
 * - Thread-safe with std::mutex protection
 * - Double-free prevention via allocation bitmap
 * - O(1) allocate/free operations
 *
 * @author DenseCore Team
 */

#ifndef DENSECORE_BLOCK_ALLOCATOR_H
#define DENSECORE_BLOCK_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

#include "densecore/memory/numa_allocator.h"

#if defined(_WIN32)
#include <malloc.h>  // For _aligned_malloc/_aligned_free
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

// libnuma, not hwloc: gate on DENSECORE_HAS_NUMA. DENSECORE_USE_HWLOC is
// defined even in the hwloc-only build, which does not link libnuma.
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
#include <numa.h>
#endif

namespace densecore {

/**
 * @brief Fixed-size block allocator for KV cache memory management
 *
 * Pre-allocates a large contiguous arena at initialization and manages
 * fixed-size blocks via a LIFO free list. Designed for PagedKVCache to
 * eliminate heap fragmentation during long-running inference.
 *
 * NUMA Support:
 * - Pass numa_node >= 0 to allocate arena on specific NUMA node
 * - Falls back to standard allocation if NUMA unavailable or fails
 *
 * Thread Safety:
 * - All public methods are protected by std::mutex
 * - Safe for concurrent Allocate()/Free() from multiple threads
 *
 * Memory Layout:
 *   Arena: [Block0][Block1][Block2]...[BlockN-1]
 *   Each block is block_stride bytes, 64-byte aligned
 */
class KVBlockAllocator {
public:
    /**
     * @brief Construct a block allocator with pre-allocated arena
     *
     * @param num_blocks Total number of blocks in the pool
     * @param block_stride Size of each block in bytes (will be aligned to 64)
     * @param alignment Memory alignment (default: 64 for AVX-512)
     * @param numa_node NUMA node for allocation (-1 for default/any node)
     * @param use_hugepages Enable HugePage allocation when available
     * @param strict_numa Enforce allocation strictly on the specified NUMA node
     */
    KVBlockAllocator(size_t num_blocks, size_t block_stride, size_t alignment = 64, int numa_node = -1,
                     bool use_hugepages = false, bool strict_numa = true)
        : arena_(nullptr),
          arena_size_(0),
          num_blocks_(num_blocks),
          block_stride_(AlignUp(block_stride, alignment)),
          alignment_(alignment),
          numa_node_(numa_node),
          alloc_type_(AllocationType::Aligned),
          use_hugepages_(use_hugepages),
          strict_numa_(strict_numa) {
        if (num_blocks == 0 || block_stride == 0) {
            std::cerr << "[KVBlockAllocator] Error: Invalid parameters (num_blocks=" << num_blocks
                      << ", block_stride=" << block_stride << ")" << std::endl;
            return;
        }

        // Calculate total arena size with alignment
        arena_size_ = num_blocks_ * block_stride_;

        // Try NUMA allocation first if requested
        if (numa_node >= 0) {
            if (use_hugepages_) {
                size_t hugepage_size = 0;
                arena_ = NumaAllocator::AllocateHugePagesOnNode(arena_size_, numa_node, strict_numa_, &hugepage_size);
                if (arena_) {
                    alloc_type_ = AllocationType::Mmap;
                    if (hugepage_size > 0) {
                        arena_size_ = hugepage_size;
                    }
                }
            }

            if (!arena_) {
                arena_ = AllocateNuma(arena_size_, numa_node, strict_numa_);
                if (arena_) {
                    alloc_type_ = AllocationType::Numa;
                }
            }

            if (!arena_ && strict_numa_) {
                std::cerr << "[KVBlockAllocator] ERROR: Strict NUMA allocation failed on node " << numa_node << " for "
                          << (arena_size_ / 1024 / 1024) << " MB" << std::endl;
                return;
            }
        }

        // Reserve the full KV capacity without faulting every page into RAM.
        // KV slots are written before they are read, so anonymous zero pages
        // can be committed on first use while the logical block pool retains
        // its fixed-capacity and zero-fragmentation properties.
        if (!arena_) {
#if defined(__linux__)
            void* mapped = mmap(nullptr, arena_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mapped != MAP_FAILED) {
                arena_ = mapped;
                alloc_type_ = AllocationType::Mmap;
            }
#endif
        }

        // Portable fallback. Unlike an anonymous mapping, this allocation is
        // not guaranteed to start as zero-filled demand pages.
        if (!arena_) {
            arena_ = AllocateAligned(arena_size_, alignment_);
            alloc_type_ = AllocationType::Aligned;
        }

        if (!arena_) {
            std::cerr << "[KVBlockAllocator] Error: Failed to allocate " << (arena_size_ / 1024 / 1024) << " MB arena"
                      << std::endl;
            return;
        }

        if (alloc_type_ == AllocationType::Aligned) {
            std::memset(arena_, 0, arena_size_);
        }

        // Initialize free list (LIFO - push in reverse for cache locality)
        free_list_.reserve(num_blocks_);
        for (int i = static_cast<int>(num_blocks_) - 1; i >= 0; --i) {
            free_list_.push_back(i);
        }

        // Initialize allocation bitmap (all false = not allocated)
        allocated_.resize(num_blocks_, false);

        const char* alloc_type_str = "Aligned";
        if (alloc_type_ == AllocationType::Numa) {
            alloc_type_str = "NUMA";
        } else if (alloc_type_ == AllocationType::Mmap) {
            alloc_type_str = use_hugepages_ ? "HugePages" : "DemandPaged";
        }
        std::cout << "[KVBlockAllocator] Initialized: " << num_blocks_ << " blocks × " << block_stride_
                  << " bytes = " << (arena_size_ / 1024 / 1024) << " MB (" << alloc_type_str << ")" << std::endl;
    }

    /**
     * @brief Destructor - frees the arena using appropriate method
     */
    ~KVBlockAllocator() {
        if (arena_) {
            NumaAllocator::Free(arena_, arena_size_, alloc_type_);
            arena_ = nullptr;
        }
    }

    // Non-copyable, movable
    KVBlockAllocator(const KVBlockAllocator&) = delete;
    KVBlockAllocator& operator=(const KVBlockAllocator&) = delete;

    KVBlockAllocator(KVBlockAllocator&& other) noexcept
        : arena_(other.arena_),
          arena_size_(other.arena_size_),
          num_blocks_(other.num_blocks_),
          block_stride_(other.block_stride_),
          alignment_(other.alignment_),
          numa_node_(other.numa_node_),
          alloc_type_(other.alloc_type_),
          use_hugepages_(other.use_hugepages_),
          strict_numa_(other.strict_numa_),
          free_list_(std::move(other.free_list_)),
          allocated_(std::move(other.allocated_)) {
        other.arena_ = nullptr;
        other.arena_size_ = 0;
    }

    KVBlockAllocator& operator=(KVBlockAllocator&& other) noexcept {
        if (this != &other) {
            // Free existing arena
            if (arena_) {
                NumaAllocator::Free(arena_, arena_size_, alloc_type_);
            }

            // Move from other
            arena_ = other.arena_;
            arena_size_ = other.arena_size_;
            num_blocks_ = other.num_blocks_;
            block_stride_ = other.block_stride_;
            alignment_ = other.alignment_;
            numa_node_ = other.numa_node_;
            alloc_type_ = other.alloc_type_;
            use_hugepages_ = other.use_hugepages_;
            strict_numa_ = other.strict_numa_;
            free_list_ = std::move(other.free_list_);
            allocated_ = std::move(other.allocated_);

            other.arena_ = nullptr;
            other.arena_size_ = 0;
        }
        return *this;
    }

    // ===========================================================================
    // Allocation Interface
    // ===========================================================================

    /**
     * @brief Allocate a single block from the pool
     *
     * Thread-safe. O(1) operation via LIFO free list.
     *
     * @return Block ID (0 to num_blocks-1), or -1 if pool is exhausted (OOM)
     */
    [[nodiscard]] int Allocate() {
        std::lock_guard<std::mutex> lock(mu_);

        if (free_list_.empty()) {
            return -1;  // OOM
        }

        int block_id = free_list_.back();
        free_list_.pop_back();
        allocated_[block_id] = true;

        return block_id;
    }

    /**
     * @brief Allocate multiple blocks from the pool
     *
     * Thread-safe. Atomic - either all blocks are allocated or none.
     *
     * @param n Number of blocks to allocate
     * @return Vector of block IDs, empty if insufficient blocks available
     */
    [[nodiscard]] std::vector<int> AllocateN(size_t n) {
        std::lock_guard<std::mutex> lock(mu_);

        if (free_list_.size() < n) {
            return {};  // OOM
        }

        std::vector<int> result;
        result.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            int block_id = free_list_.back();
            free_list_.pop_back();
            allocated_[block_id] = true;
            result.push_back(block_id);
        }

        return result;
    }

    /**
     * @brief Free a single block back to the pool
     *
     * Thread-safe. O(1) operation. Includes double-free protection.
     *
     * @param block_id Block ID to free
     * @return true if successfully freed, false if invalid or double-free
     */
    bool Free(int block_id) {
        if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_) {
            std::cerr << "[KVBlockAllocator] Warning: Invalid block_id " << block_id << " (valid range: 0-"
                      << (num_blocks_ - 1) << ")" << std::endl;
            return false;
        }

        std::lock_guard<std::mutex> lock(mu_);

        // Double-free protection
        if (!allocated_[block_id]) {
            std::cerr << "[KVBlockAllocator] Warning: Double-free detected for block " << block_id << std::endl;
            return false;
        }

        allocated_[block_id] = false;
        free_list_.push_back(block_id);
        return true;
    }

    /**
     * @brief Free multiple blocks back to the pool
     *
     * Thread-safe. Includes double-free protection for each block.
     *
     * @param block_ids Vector of block IDs to free
     * @return Number of blocks successfully freed
     */
    size_t FreeN(const std::vector<int>& block_ids) {
        std::lock_guard<std::mutex> lock(mu_);

        size_t freed_count = 0;
        for (int block_id : block_ids) {
            if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_) {
                continue;
            }
            if (!allocated_[block_id]) {
                continue;  // Already free
            }
            allocated_[block_id] = false;
            free_list_.push_back(block_id);
            ++freed_count;
        }
        return freed_count;
    }

    // ===========================================================================
    // Pointer Access
    // ===========================================================================

    /**
     * @brief Get raw pointer to a block's data
     *
     * Computes: arena_base + (block_id * block_stride)
     *
     * @param block_id Block ID (0 to num_blocks-1)
     * @return Pointer to block data, nullptr if invalid block_id
     */
    void* GetBlockPtr(int block_id) {
        if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_) {
            return nullptr;
        }
        return static_cast<char*>(arena_) + (static_cast<size_t>(block_id) * block_stride_);
    }

    /**
     * @brief Get const raw pointer to a block's data
     */
    const void* GetBlockPtr(int block_id) const {
        if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_) {
            return nullptr;
        }
        return static_cast<const char*>(arena_) + (static_cast<size_t>(block_id) * block_stride_);
    }

    /**
     * @brief Get typed pointer to a block's data
     *
     * @tparam T Type to cast pointer to
     * @param block_id Block ID
     * @return Typed pointer, nullptr if invalid
     */
    template <typename T> T* GetBlockPtrAs(int block_id) { return static_cast<T*>(GetBlockPtr(block_id)); }

    template <typename T> const T* GetBlockPtrAs(int block_id) const {
        return static_cast<const T*>(GetBlockPtr(block_id));
    }

    // ===========================================================================
    // Statistics
    // ===========================================================================

    /**
     * @brief Get number of free blocks available
     */
    size_t FreeCount() const {
        std::lock_guard<std::mutex> lock(mu_);
        return free_list_.size();
    }

    /**
     * @brief Get number of allocated blocks
     */
    size_t AllocatedCount() const {
        std::lock_guard<std::mutex> lock(mu_);
        return num_blocks_ - free_list_.size();
    }

    /**
     * @brief Get total number of blocks in pool
     */
    size_t TotalBlocks() const { return num_blocks_; }

    /**
     * @brief Get block stride (size per block in bytes)
     */
    size_t BlockStride() const { return block_stride_; }

    /**
     * @brief Get total arena size in bytes
     */
    size_t ArenaSize() const { return arena_size_; }

    /**
     * @brief Get base arena pointer (for debugging)
     */
    void* ArenaBase() { return arena_; }
    const void* ArenaBase() const { return arena_; }

    /**
     * @brief Get NUMA node used for allocation (-1 if not NUMA)
     */
    int NumaNode() const { return numa_node_; }

    /**
     * @brief Get allocation type (NUMA or Aligned)
     */
    AllocationType AllocType() const { return alloc_type_; }

    /**
     * @brief Check if a block is currently allocated
     */
    bool IsAllocated(int block_id) const {
        if (block_id < 0 || static_cast<size_t>(block_id) >= num_blocks_) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        return allocated_[block_id];
    }

    /**
     * @brief Check if allocator is valid (successfully initialized)
     */
    bool IsValid() const { return arena_ != nullptr && num_blocks_ > 0; }

    /**
     * @brief Check if NUMA allocation was used
     */
    bool IsNumaAllocated() const { return alloc_type_ == AllocationType::Numa; }

private:
    // ===========================================================================
    // Helper Functions
    // ===========================================================================

    /**
     * @brief Align size up to specified alignment
     */
    static size_t AlignUp(size_t size, size_t alignment) { return (size + alignment - 1) & ~(alignment - 1); }

    /**
     * @brief Allocate aligned memory (platform-independent)
     */
    static void* AllocateAligned(size_t size, size_t alignment) {
#if defined(_WIN32)
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
#endif
    }

    /**
     * @brief Free aligned memory (platform-independent)
     */
    static void FreeAligned(void* ptr) {
        if (!ptr) return;
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    /**
     * @brief Allocate memory on a specific NUMA node
     *
     * @param size Size in bytes
     * @param numa_node Target NUMA node
     * @return Pointer to allocated memory, nullptr on failure
     */
    static void* AllocateNuma(size_t size, int numa_node, bool strict_numa) {
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
        if (numa_available() < 0) {
            std::cerr << "[KVBlockAllocator] NUMA not available on this system" << std::endl;
            return nullptr;
        }

        // Allocate on specific NUMA node
        void* ptr = numa_alloc_onnode(size, numa_node);
        if (ptr) {
            // numa_alloc_onnode retains the node policy for later page faults.
            // Do not pre-touch the full capacity: KV pages should become
            // resident only as requests consume logical cache blocks.
            return ptr;
        }

        if (!strict_numa) {
            // Try local allocation as fallback
            std::cerr << "[KVBlockAllocator] numa_alloc_onnode failed, trying local" << std::endl;
            ptr = numa_alloc_local(size);
            if (ptr) {
                return ptr;
            }
        }

        return nullptr;
#else
        (void)size;
        (void)numa_node;
        (void)strict_numa;
        return nullptr;  // NUMA not available
#endif
    }

    /**
     * @brief Free NUMA-allocated memory
     */
    static void FreeNuma(void* ptr, size_t size) {
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
        if (ptr && size > 0) {
            numa_free(ptr, size);
        }
#else
        (void)ptr;
        (void)size;
#endif
    }

    // ===========================================================================
    // Member Variables
    // ===========================================================================

    void* arena_;                ///< Contiguous pre-allocated memory
    size_t arena_size_;          ///< Total arena size in bytes
    size_t num_blocks_;          ///< Number of blocks in pool
    size_t block_stride_;        ///< Size of each block (aligned)
    size_t alignment_;           ///< Memory alignment
    int numa_node_;              ///< NUMA node for allocation (-1 = any)
    AllocationType alloc_type_;  ///< Allocation type for proper cleanup
    bool use_hugepages_;         ///< True if huge pages were requested
    bool strict_numa_;           ///< Enforce strict NUMA allocation

    std::vector<int> free_list_;   ///< LIFO stack of free block IDs
    std::vector<bool> allocated_;  ///< Bitmap for double-free detection
    mutable std::mutex mu_;        ///< Mutex for thread safety
};

}  // namespace densecore

#endif  // DENSECORE_BLOCK_ALLOCATOR_H
