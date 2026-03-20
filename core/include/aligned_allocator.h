/**
 * @file aligned_allocator.h
 * @brief Cross-platform aligned memory allocation for SIMD/AMX operations
 *
 * Provides:
 * - AlignedAllocator<T, N>: STL-compatible allocator for std::vector
 * - AlignedDeleter<T>: Custom deleter for std::unique_ptr
 * - Utility functions: aligned_alloc, aligned_free, make_aligned
 *
 * All allocations are 64-byte aligned by default (required for AVX-512, AMX).
 */

#ifndef DENSECORE_ALIGNED_ALLOCATOR_H
#define DENSECORE_ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace densecore {

/// Default alignment for SIMD/AMX operations (64 bytes = AVX-512 cache line)
constexpr size_t DEFAULT_ALIGNMENT = 64;

// =============================================================================
// Low-level Allocation Functions
// =============================================================================

/**
 * @brief Allocate aligned memory
 *
 * @param size Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2, >= sizeof(void*))
 * @return Pointer to aligned memory, nullptr on failure
 */
inline void* aligned_alloc_raw(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
    if (size == 0) return nullptr;

    // Ensure alignment is at least sizeof(void*) and power of 2
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

#ifdef _WIN32
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
 * @brief Free aligned memory
 * @param ptr Pointer previously returned by aligned_alloc_raw
 */
inline void aligned_free(void* ptr) noexcept {
    if (!ptr) return;

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// =============================================================================
// Typed Allocation Functions
// =============================================================================

/**
 * @brief Allocate aligned array of type T
 *
 * @tparam T Element type
 * @param count Number of elements
 * @param alignment Required alignment in bytes
 * @return Pointer to aligned array, nullptr on failure
 */
template <typename T> T* aligned_alloc(size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
    return static_cast<T*>(aligned_alloc_raw(count * sizeof(T), alignment));
}

// =============================================================================
// Custom Deleter for std::unique_ptr
// =============================================================================

/**
 * @brief Custom deleter for aligned memory (use with std::unique_ptr<T[],
 * AlignedDeleter<T>>)
 */
template <typename T> struct AlignedDeleter {
    void operator()(T* ptr) const noexcept { aligned_free(ptr); }
};

/**
 * @brief Alias for unique_ptr with aligned memory
 *
 * Usage:
 *   AlignedPtr<float> buffer = make_aligned<float>(1024);
 *   buffer[0] = 1.0f;
 */
template <typename T> using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

/**
 * @brief Create an aligned unique_ptr array
 *
 * @tparam T Element type
 * @param count Number of elements
 * @param alignment Required alignment in bytes
 * @return AlignedPtr<T> owning the allocated memory
 * @throws std::bad_alloc if allocation fails
 */
template <typename T> AlignedPtr<T> make_aligned(size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
    T* ptr = aligned_alloc<T>(count, alignment);
    if (!ptr && count > 0) {
        throw std::bad_alloc();
    }
    return AlignedPtr<T>(ptr);
}

class DeviceAllocator {
    virtual void* Allocate(size_t size) = 0;
    virtual void CopyToDevice(void* dst, const void* src, size_t size) = 0;
};

// =============================================================================
// STL-Compatible Allocator
// =============================================================================

/**
 * @brief STL-compatible allocator for aligned memory
 *
 * Usage:
 *   std::vector<float, AlignedAllocator<float>> vec(1024);
 *   // vec.data() is guaranteed to be 64-byte aligned
 *
 * @tparam T Element type
 * @tparam Alignment Required alignment in bytes (default: 64)
 */
template <typename T, size_t Alignment = DEFAULT_ALIGNMENT> class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    constexpr AlignedAllocator() noexcept = default;

    template <typename U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0) return nullptr;

        // Check for overflow
        if (n > static_cast<size_type>(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }

        T* ptr = aligned_alloc<T>(n, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* ptr, size_type /*n*/) noexcept { aligned_free(ptr); }

    // Rebind for containers that allocate auxiliary nodes (e.g., std::list)
    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

// Allocator comparison operators
template <typename T, typename U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template <typename T, typename U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// =============================================================================
// Convenience Type Aliases
// =============================================================================

/// Aligned vector using 64-byte alignment
template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T, DEFAULT_ALIGNMENT>>;

}  // namespace densecore

#endif  // DENSECORE_ALIGNED_ALLOCATOR_H
