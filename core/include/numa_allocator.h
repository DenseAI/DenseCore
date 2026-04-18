/**
 * @file numa_allocator.h
 * @brief NUMA-aware memory allocation for KV Cache and large buffers
 *
 * Uses libnuma's mbind() or mmap with explicit NUMA policy to ensure
 * memory is allocated on the correct physical NUMA node, eliminating
 * remote memory access penalties on multi-socket servers.
 */

#ifndef DENSECORE_NUMA_ALLOCATOR_H
#define DENSECORE_NUMA_ALLOCATOR_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#if defined(__linux__)
#include <unistd.h>

#include <sys/mman.h>

#ifdef DENSECORE_USE_HWLOC
#include <numa.h>
#include <numaif.h>
#endif
#endif

namespace densecore {

/**
 * Allocation type for correct deallocation dispatch.
 * Each allocation method requires its own deallocator.
 */
enum class AllocationType {
    Aligned,  ///< posix_memalign/_aligned_malloc: use free()/_aligned_free()
    Mmap,     ///< mmap: use munmap()
    Numa      ///< numa_alloc_*: use numa_free()
};

/**
 * NUMA-aware memory allocator
 *
 * Provides explicit NUMA node binding for large memory allocations.
 * Falls back to standard aligned allocation when libnuma is not available.
 */
class NumaAllocator {
public:
    /**
     * Allocate memory on a specific NUMA node
     *
     * @param bytes Size of allocation in bytes
     * @param numa_node Target NUMA node (-1 for default allocation)
     * @return Pointer to allocated memory, nullptr on failure
     */
    static void* AllocateOnNode(size_t bytes, int numa_node) {
        if (bytes == 0) return nullptr;

#if !defined(__linux__) || !defined(DENSECORE_USE_HWLOC)
        (void)numa_node;
#endif

#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (numa_node >= 0 && numa_available() >= 0) {
            return AllocateWithNumaBind(bytes, numa_node);
        }
#endif
        // Fallback to standard allocation
        return AllocateAligned(bytes, 64);
    }

    /**
     * Allocate aligned memory on a specific NUMA node
     *
     * @param bytes Size of allocation
     * @param alignment Required alignment (must be power of 2)
     * @param numa_node Target NUMA node (-1 for default)
     * @return Pointer to aligned memory on specified node
     */
    static void* AllocateAlignedOnNode(size_t bytes, size_t alignment, int numa_node) {
        if (bytes == 0) return nullptr;

        // Round up bytes to alignment boundary
        size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);

#if !defined(__linux__) || !defined(DENSECORE_USE_HWLOC)
        (void)numa_node;
#endif

#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (numa_node >= 0 && numa_available() >= 0) {
            // Use mmap for aligned NUMA allocation
            void* ptr = AllocateWithMmap(aligned_size, numa_node);
            if (ptr && ((uintptr_t)ptr & (alignment - 1)) == 0) {
                return ptr;
            }
            // Fallback if mmap didn't produce aligned result
            if (ptr) Free(ptr, aligned_size, AllocationType::Mmap);
        }
#endif
        return AllocateAligned(aligned_size, alignment);
    }

    /**
     * Result of preferred allocation with fallback status
     */
    struct AllocationResult {
        void* ptr = nullptr;                                       ///< Allocated memory pointer
        size_t size = 0;                                           ///< Allocation size (for proper deallocation)
        AllocationType allocation_type = AllocationType::Aligned;  ///< How memory was allocated
        bool on_preferred_node = false;                            ///< True if allocated on requested NUMA node
        int actual_node = -1;                                      ///< Actual NUMA node (-1 if unknown/fallback)

        /**
         * Free the allocated memory using the correct deallocator.
         * Sets ptr to nullptr after freeing.
         */
        void Free() {
            if (!ptr) return;
            NumaAllocator::Free(ptr, size, allocation_type);
            ptr = nullptr;
            size = 0;
        }
    };

    /**
     * Allocate memory with PREFERRED NUMA policy (graceful fallback)
     *
     * Tries to allocate on the specified NUMA node first. If that fails
     * (e.g., OOM on that node), falls back to local/interleaved allocation
     * instead of returning nullptr.
     *
     * @param bytes Size of allocation in bytes
     * @param alignment Required alignment (must be power of 2)
     * @param preferred_node Preferred NUMA node (-1 for any)
     * @return AllocationResult with pointer and allocation status
     */
    static AllocationResult AllocatePreferred(size_t bytes, size_t alignment, int preferred_node) {
        AllocationResult result;
        result.ptr = nullptr;
        result.on_preferred_node = false;
        result.actual_node = -1;

        if (bytes == 0) return result;

        size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);

#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (preferred_node >= 0 && numa_available() >= 0) {
            // Step 1: Try strict allocation on preferred node
            void* ptr = numa_alloc_onnode(aligned_size, preferred_node);
            if (ptr) {
                // Touch pages to ensure physical allocation
                TouchPages(ptr, aligned_size);
                result.ptr = ptr;
                result.size = aligned_size;
                result.allocation_type = AllocationType::Numa;
                result.on_preferred_node = true;
                result.actual_node = preferred_node;
                return result;
            }

            // Step 2: Preferred node OOM - log warning and fallback
            std::cerr << "[NumaAllocator] WARNING: NUMA node " << preferred_node << " OOM for "
                      << (aligned_size / 1024 / 1024) << " MB, falling back to local allocation" << std::endl;

            // Try numa_alloc_local (allocates on calling thread's node)
            ptr = numa_alloc_local(aligned_size);
            if (ptr) {
                TouchPages(ptr, aligned_size);
                result.ptr = ptr;
                result.size = aligned_size;
                result.allocation_type = AllocationType::Numa;
                result.on_preferred_node = false;
                result.actual_node = GetMemoryNode(ptr);
                std::cerr << "[NumaAllocator] Fallback allocation on node " << result.actual_node << " succeeded"
                          << std::endl;
                return result;
            }

            // Step 3: Even local failed - try interleaved (last resort before malloc)
            ptr = numa_alloc_interleaved(aligned_size);
            if (ptr) {
                TouchPages(ptr, aligned_size);
                result.ptr = ptr;
                result.size = aligned_size;
                result.allocation_type = AllocationType::Numa;
                result.on_preferred_node = false;
                result.actual_node = -1;  // Interleaved across nodes
                std::cerr << "[NumaAllocator] Using interleaved NUMA allocation" << std::endl;
                return result;
            }
        }
#else
        (void)preferred_node;
#endif

        // Final fallback: standard aligned allocation
        result.ptr = AllocateAligned(aligned_size, alignment);
        result.size = aligned_size;
        result.allocation_type = AllocationType::Aligned;
        result.on_preferred_node = false;
        result.actual_node = -1;
        if (result.ptr) {
            std::cerr << "[NumaAllocator] Using standard malloc (NUMA unavailable)" << std::endl;
        }
        return result;
    }

    /**
     * Free NUMA-allocated memory (from numa_alloc_*).
     * MUST be used for memory from AllocatePreferred when it used libnuma.
     *
     * @param ptr Pointer to memory
     * @param bytes Size of allocation
     */
    static void FreeNuma(void* ptr, size_t bytes) {
        if (!ptr) return;
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        numa_free(ptr, bytes);
#else
        // If libnuma is not available, this path should never be taken.
        // Fallback to aligned free as a safety measure.
        (void)bytes;
        FreeAligned(ptr);
#endif
    }

    /**
     * Free mmap-allocated memory.
     *
     * @param ptr Pointer to memory
     * @param bytes Size of allocation (required for munmap)
     */
    static void FreeMmap(void* ptr, size_t bytes) {
        if (!ptr) return;
#if defined(__linux__)
        munmap(ptr, bytes);
#else
        // mmap not available on this platform
        (void)bytes;
        FreeAligned(ptr);
#endif
    }

    /**
     * Free memory allocated by this allocator using explicit allocation type.
     * This is the preferred method as it dispatches to the correct deallocator.
     *
     * @param ptr Pointer to memory
     * @param bytes Size of allocation (required for mmap/numa deallocation)
     * @param type How the memory was allocated
     */
    static void Free(void* ptr, size_t bytes, AllocationType type) {
        if (!ptr) return;
        switch (type) {
        case AllocationType::Numa: FreeNuma(ptr, bytes); break;
        case AllocationType::Mmap: FreeMmap(ptr, bytes); break;
        case AllocationType::Aligned:
        default: FreeAligned(ptr); break;
        }
    }

    /**
     * Get the NUMA node for an existing memory address
     *
     * @param ptr Pointer to query
     * @return NUMA node ID, or -1 if unknown
     */
    static int GetMemoryNode(const void* ptr) {
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (numa_available() >= 0) {
            int node = -1;
            if (get_mempolicy(&node, nullptr, 0, const_cast<void*>(ptr), MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                return node;
            }
        }
#else
        (void)ptr;
#endif
        return -1;
    }

    /**
     * Check if NUMA allocation is available
     */
    static bool IsNumaAvailable() {
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        return numa_available() >= 0;
#else
        return false;
#endif
    }

    /**
     * Allocate memory with huge pages on a specific NUMA node
     *
     * Uses mmap with MAP_HUGETLB for explicit huge pages, with fallback to
     * madvise(MADV_HUGEPAGE) for transparent huge pages.
     *
     * @param bytes Size of allocation in bytes
     * @param numa_node Target NUMA node (-1 for default node)
     * @return Pointer to allocated memory, nullptr on failure
     */
    static void* AllocateHugePagesOnNode(size_t bytes, int numa_node) {
        return AllocateHugePagesOnNode(bytes, numa_node, false, nullptr);
    }

    /**
     * Allocate memory with huge pages on a specific NUMA node (extended API)
     *
     * @param bytes Size of allocation in bytes
     * @param numa_node Target NUMA node (-1 for default node)
     * @param strict_numa If true, fail when NUMA binding cannot be enforced
     * @param out_aligned_bytes Optional output for the actual aligned size
     * @return Pointer to allocated memory, nullptr on failure
     */
    static void* AllocateHugePagesOnNode(size_t bytes, int numa_node, bool strict_numa, size_t* out_aligned_bytes) {
        if (out_aligned_bytes) {
            *out_aligned_bytes = 0;
        }
        if (bytes == 0) return nullptr;

#if defined(__linux__)
        // Align to 2MB for huge page compatibility
        constexpr size_t kHugePageSize = 2 * 1024 * 1024;  // 2MB
        size_t aligned_size = (bytes + kHugePageSize - 1) & ~(kHugePageSize - 1);
        if (out_aligned_bytes) {
            *out_aligned_bytes = aligned_size;
        }
        void* ptr = nullptr;

#ifdef MAP_HUGETLB
        // Try explicit huge pages first (requires hugetlb configured)
        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

        if (ptr != MAP_FAILED) {
            // Bind to NUMA node if specified
            if (numa_node >= 0) {
                if (!BindToNumaNode(ptr, aligned_size, numa_node) && strict_numa) {
                    munmap(ptr, aligned_size);
                    return nullptr;
                }
            }
            // Touch pages - first-touch policy ensures allocation on local node
            TouchPages(ptr, aligned_size);
            std::cout << "[NumaAllocator] Allocated " << (aligned_size / 1024 / 1024) << " MB via huge pages";
            if (numa_node >= 0) std::cout << " on NUMA node " << numa_node;
            std::cout << std::endl;
            return ptr;
        }
#endif

        // Fallback: normal mmap with transparent huge pages hint
        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr == MAP_FAILED) {
            std::cerr << "[NumaAllocator] mmap failed for huge page allocation" << std::endl;
            return nullptr;
        }

#ifdef MADV_HUGEPAGE
        // Request transparent huge pages
        madvise(ptr, aligned_size, MADV_HUGEPAGE);
#endif

        // Bind to NUMA node
        if (numa_node >= 0) {
            if (!BindToNumaNode(ptr, aligned_size, numa_node) && strict_numa) {
                munmap(ptr, aligned_size);
                return nullptr;
            }
        }

        // Touch all pages to enforce first-touch allocation on correct node
        TouchPages(ptr, aligned_size);

        std::cout << "[NumaAllocator] Allocated " << (aligned_size / 1024 / 1024) << " MB via mmap (THP requested)";
        if (numa_node >= 0) std::cout << " on NUMA node " << numa_node;
        std::cout << std::endl;

        return ptr;
#else
        (void)bytes;
        (void)numa_node;
        (void)strict_numa;
        if (out_aligned_bytes) {
            *out_aligned_bytes = 0;
        }
        return nullptr;
#endif
    }

    /**
     * Touch all pages in a memory region to enforce first-touch policy
     * Should be called from the thread that will primarily access the memory
     */
    static void TouchPages(void* ptr, size_t bytes) {
        if (!ptr || bytes == 0) return;

        // Touch each page to ensure allocation on local NUMA node
        volatile char* p = static_cast<volatile char*>(ptr);
        size_t page_size = GetPageSize();
        for (size_t i = 0; i < bytes; i += page_size) {
            p[i] = 0;
        }
        // Also touch last byte if not aligned
        if (bytes > 0) {
            p[bytes - 1] = 0;
        }
    }

private:
    /**
     * Bind a memory region to a specific NUMA node using mbind
     */
    static bool BindToNumaNode(void* ptr, size_t bytes, int numa_node) {
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (numa_node < 0 || numa_available() < 0) return false;

        unsigned long nodemask = 1UL << numa_node;
        if (mbind(ptr, bytes, MPOL_BIND, &nodemask, numa_max_node() + 2, MPOL_MF_STRICT | MPOL_MF_MOVE) != 0) {
            std::cerr << "[NumaAllocator] mbind to node " << numa_node << " failed" << std::endl;
            return false;
        }
        return true;
#else
        (void)ptr;
        (void)bytes;
        (void)numa_node;
        return false;
#endif
    }

#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
    /**
     * Allocate using mmap with explicit NUMA binding via mbind
     */
    static void* AllocateWithMmap(size_t bytes, int numa_node) {
        size_t page_size = GetPageSize();
        size_t aligned_size = (bytes + page_size - 1) & ~(page_size - 1);

        // Allocate anonymous memory
        void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr == MAP_FAILED) {
            std::cerr << "[NumaAllocator] mmap failed for " << bytes << " bytes" << std::endl;
            return nullptr;
        }

        // Bind to NUMA node
        unsigned long nodemask = 1UL << numa_node;
        if (mbind(ptr, aligned_size, MPOL_BIND, &nodemask, numa_max_node() + 2, MPOL_MF_STRICT | MPOL_MF_MOVE) != 0) {
            std::cerr << "[NumaAllocator] mbind to node " << numa_node << " failed" << std::endl;
            // Continue anyway - memory will work, just not NUMA-optimized
        }

        // Touch pages to ensure they're allocated on correct node
        memset(ptr, 0, aligned_size);

        return ptr;
    }

    /**
     * Allocate using libnuma's numa_alloc_onnode
     */
    static void* AllocateWithNumaBind(size_t bytes, int numa_node) {
        void* ptr = numa_alloc_onnode(bytes, numa_node);
        if (!ptr) {
            std::cerr << "[NumaAllocator] numa_alloc_onnode failed for " << bytes << " bytes on node " << numa_node
                      << std::endl;
            return nullptr;
        }
        // Touch to ensure allocation
        memset(ptr, 0, bytes);
        return ptr;
    }
#endif

    /**
     * Standard aligned allocation fallback
     */
    static void* AllocateAligned(size_t bytes, size_t alignment) {
#if defined(_WIN32)
        return _aligned_malloc(bytes, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, bytes) != 0) {
            return nullptr;
        }
        return ptr;
#endif
    }

    /**
     * Free aligned allocation
     */
    static void FreeAligned(void* ptr) {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    static size_t GetPageSize() {
#if defined(__linux__)
        static size_t page_size = sysconf(_SC_PAGESIZE);
        return page_size;
#elif defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return si.dwPageSize;
#else
        return 4096;
#endif
    }
};

/**
 * Memory diagnostics for verifying NUMA and HugePage allocation
 *
 * Provides production-ready diagnostic functions to validate that
 * memory is actually allocated where expected, not silently fallen back.
 */
class MemoryDiagnostics {
public:
    /**
     * Diagnostic result for memory allocation verification
     */
    struct DiagResult {
        int requested_node;      ///< NUMA node requested during allocation
        int actual_node;         ///< Actual NUMA node of pages
        bool nodes_match;        ///< True if requested == actual
        bool huge_pages_active;  ///< True if huge pages are in use
        size_t huge_page_count;  ///< Number of huge pages (if active)
        size_t total_size_mb;    ///< Total allocation size in MB
    };

    /**
     * Get the actual NUMA node of a memory region using get_mempolicy/move_pages
     *
     * @param ptr Pointer to memory region
     * @param size Size of region in bytes
     * @return NUMA node ID, or -1 if cannot be determined
     */
    static int GetActualNumaNode(void* ptr, size_t size) {
#if defined(__linux__) && defined(DENSECORE_USE_HWLOC)
        if (!ptr || size == 0) return -1;

        // Use move_pages to probe actual page location
        // This is more reliable than get_mempolicy for already-allocated memory
        void* pages[1] = {ptr};
        int status[1] = {-1};

        // move_pages with nullptr nodes just reports current location
        if (move_pages(0, 1, pages, nullptr, status, 0) == 0) {
            return status[0];
        }

        // Fallback: try get_mempolicy
        int mode = 0;
        unsigned long nodemask = 0;
        if (get_mempolicy(&mode, &nodemask, sizeof(nodemask) * 8, ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            // Find first set bit in nodemask
            for (int i = 0; i < 64; ++i) {
                if (nodemask & (1UL << i)) return i;
            }
        }

        return -1;
#else
        (void)ptr;
        (void)size;
        return -1;
#endif
    }

    /**
     * Check if huge pages are active for a memory region by parsing
     * /proc/self/smaps
     *
     * @param ptr Pointer to memory region start
     * @param size Size of region in bytes
     * @param out_huge_count Output: number of huge pages found
     * @return True if huge pages are detected
     */
    static bool CheckHugePages(void* ptr, size_t size, size_t& out_huge_count) {
#if defined(__linux__)
        out_huge_count = 0;
        if (!ptr || size == 0) return false;

        FILE* smaps = fopen("/proc/self/smaps", "r");
        if (!smaps) return false;

        uintptr_t target_start = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t target_end = target_start + size;

        char line[512];
        bool in_target_region = false;
        size_t huge_kb = 0;

        while (fgets(line, sizeof(line), smaps)) {
            // Check for address range line (e.g., "7f1234560000-7f1234570000 rw-p
            // ...")
            uintptr_t start = 0, end = 0;
            if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
                // Check if this mapping overlaps with our target region
                in_target_region = (start < target_end && end > target_start);
                continue;
            }

            // Parse HugePages or AnonHugePages line within target region
            if (in_target_region) {
                size_t value_kb = 0;
                if (sscanf(line, "AnonHugePages: %zu kB", &value_kb) == 1 ||
                    sscanf(line, "HugePages: %zu kB", &value_kb) == 1) {
                    huge_kb += value_kb;
                }
            }
        }

        fclose(smaps);

        // Convert KB to page count (2MB pages)
        out_huge_count = huge_kb / 2048;
        return huge_kb > 0;
#else
        (void)ptr;
        (void)size;
        out_huge_count = 0;
        return false;
#endif
    }

    /**
     * Print comprehensive system topology report for KV cache allocation
     *
     * Verifies NUMA placement and huge page usage, printing structured logs.
     *
     * @param ptr Pointer to allocated memory
     * @param size Size in bytes
     * @param requested_node NUMA node that was requested
     * @param label Human-readable label (e.g., "KV Cache")
     * @return DiagResult with all diagnostic information
     */
    static DiagResult PrintSystemTopologyReport(void* ptr, size_t size, int requested_node, const char* label) {
        DiagResult result = {};
        result.requested_node = requested_node;
        result.total_size_mb = size / (1024 * 1024);

        // Get actual NUMA node
        result.actual_node = GetActualNumaNode(ptr, size);
        result.nodes_match = (result.actual_node == requested_node) || (requested_node < 0 && result.actual_node >= 0);

        // Check huge pages
        result.huge_pages_active = CheckHugePages(ptr, size, result.huge_page_count);

        // Print structured report
        std::cout << "[System Check] " << label << ": " << result.total_size_mb << " MB allocated";

        if (result.actual_node >= 0) {
            std::cout << " on Node " << result.actual_node;
            if (requested_node >= 0) {
                if (result.nodes_match) {
                    std::cout << " (Match)";
                } else {
                    std::cout << " (MISMATCH - requested Node " << requested_node << ")";
                }
            }
        } else {
            std::cout << " (NUMA node unknown)";
        }

        std::cout << ", HugePages: ";
        if (result.huge_pages_active) {
            std::cout << "Active (" << result.huge_page_count << " pages)";
        } else {
            std::cout << "Inactive";
        }
        std::cout << std::endl;

        // Print warning if mismatch detected
        if (requested_node >= 0 && result.actual_node >= 0 && !result.nodes_match) {
            std::cerr << "[System Check] WARNING: " << label << " requested on Node " << requested_node
                      << ", but resident on Node " << result.actual_node << " (Mismatch). Performance may be degraded."
                      << std::endl;
        }

        if (requested_node >= 0 && !result.huge_pages_active) {
            std::cerr << "[System Check] WARNING: " << label << " HugePages inactive. Consider configuring hugepages: "
                      << "echo 1024 | sudo tee /proc/sys/vm/nr_hugepages" << std::endl;
        }

        return result;
    }
};

}  // namespace densecore

#endif  // DENSECORE_NUMA_ALLOCATOR_H
