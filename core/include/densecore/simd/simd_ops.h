/**
 * @file densecore/simd/simd_ops.h
 * @brief SIMD-optimized operations for CPU inference
 *
 * THREADING CONTRACT:
 * -------------------
 * All functions in this file are SINGLE-THREADED kernels designed to be
 * called from GGML worker threads. DO NOT add #pragma omp parallel to any
 * function as this would cause thread oversubscription.
 *
 * For functions that support parallel execution across tokens/heads, they
 * accept (ith, nth) parameters for manual work partitioning, where:
 *   - ith: Current thread index (0 to nth-1)
 *   - nth: Total number of threads
 *
 * The caller (GGML callback or inference engine) is responsible for invoking
 * these functions from multiple threads with appropriate (ith, nth) values.
 *
 * Supports:
 * - AVX-512 (Intel Skylake-X+)
 * - AVX2 (Intel Haswell+, AMD Zen+)
 * - AVX (Intel Sandy Bridge+)
 * - SSE4.1 (Intel Penryn+)
 * - ARM SVE (AWS Graviton 3/4, scalable 256-bit+ vectors)
 * - ARM NEON (Apple Silicon, ARM64, fixed 128-bit vectors)
 *
 * Auto-detects best available SIMD and provides unified API.
 */

#ifndef DENSECORE_SIMD_OPS_H
#define DENSECORE_SIMD_OPS_H

#include <algorithm>  // For std::partial_sort
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // For std::abort, posix_memalign, free
#include <cstring>
#include <functional>   // For std::function
#include <new>          // For std::bad_alloc
#include <thread>       // For ParallelFor
#include <type_traits>  // For std::true_type
#include <vector>

#if defined(__linux__)
#include <pthread.h>  // For pthread_setaffinity_np
#include <sched.h>    // For cpu_set_t
#include <unistd.h>   // For sysconf, _SC_NPROCESSORS_ONLN
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // For SYSTEM_INFO, GetSystemInfo
#endif

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DENSECORE_X86
#include "densecore/simd/simd_platform.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DENSECORE_ARM
#include <arm_neon.h>
// SVE/SVE2 headers (may not be available on all systems)
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
// Linux HWCAP for runtime SVE/SVE2 detection
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif
#endif

// =============================================================================
// FMA Detection Macro
// =============================================================================
// FMA instructions (_mm256_fmadd_ps, etc.) require the -mfma flag on GCC/Clang
// which sets the __FMA__ macro. AVX2 alone is NOT sufficient for FMA.
// This macro provides a clear guard for FMA-dependent code paths.
// =============================================================================
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define DENSECORE_HAS_FMA 1
#else
#define DENSECORE_HAS_FMA 0
#endif

// FP16 support from ggml
extern "C" {
#include "ggml.h"
}

// =============================================================================
// Aligned Memory Allocation for SIMD (64-byte for AVX-512)
// =============================================================================
// AVX-512 instructions like vmovaps require 64-byte aligned memory.
// std::vector does NOT guarantee this alignment. Use AlignedVector<T> instead.
// =============================================================================

namespace densecore {
namespace simd {

/// AVX-512 cache line size (64 bytes)
constexpr size_t SIMD_ALIGNMENT = 64;

// =============================================================================
// Tunable Prefetch Distance for Runtime Optimization
// =============================================================================
// These can be adjusted at runtime for optimal performance on different
// hardware. Default values are tuned for Intel Skylake-X / Ice Lake.
// =============================================================================

/// Prefetch distance for AVX-512 kernels (bytes ahead)
inline int g_prefetch_dist_avx512 = 128;

/// Prefetch distance for AVX2 kernels (bytes ahead)
inline int g_prefetch_dist_avx2 = 64;

/**
 * @brief Set prefetch distances for SIMD kernels
 *
 * Tune these based on hardware characteristics:
 * - Larger values for high-latency memory (server DIMMs)
 * - Smaller values for low-latency (L3 resident data)
 *
 * @param dist_avx512 Prefetch distance for AVX-512 in bytes (default: 128)
 * @param dist_avx2 Prefetch distance for AVX2 in bytes (default: 64)
 */
inline void SetPrefetchDistance(int dist_avx512, int dist_avx2) {
    g_prefetch_dist_avx512 = dist_avx512;
    g_prefetch_dist_avx2 = dist_avx2;
}

/**
 * @brief Get current AVX-512 prefetch distance
 */
inline int GetPrefetchDistanceAVX512() {
    return g_prefetch_dist_avx512;
}

/**
 * @brief Get current AVX2 prefetch distance
 */
inline int GetPrefetchDistanceAVX2() {
    return g_prefetch_dist_avx2;
}

/**
 * @brief Check if pointer is aligned to given boundary
 *
 * @param ptr Pointer to check
 * @param alignment Required alignment (must be power of 2)
 * @return true if aligned, false otherwise
 */
inline bool IsAligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/**
 * @brief Check if pointer is 64-byte aligned (AVX-512 compatible)
 */
inline bool IsAligned64(const void* ptr) {
    return IsAligned(ptr, 64);
}

}  // namespace simd
}  // namespace densecore

/**
 * Alignment assertion macro for DEBUG builds.
 * In DEBUG mode: aborts if pointer is not properly aligned.
 * In RELEASE mode: no-op (zero overhead).
 */
#ifdef NDEBUG
#define DENSECORE_ASSERT_ALIGNED(ptr, alignment) ((void)0)
#else
#define DENSECORE_ASSERT_ALIGNED(ptr, alignment)                                  \
    do {                                                                          \
        if (!densecore::simd::IsAligned((ptr), (alignment))) {                    \
            std::fprintf(stderr,                                                  \
                         "[DenseCore] FATAL: Pointer %p not %zu-byte aligned at " \
                         "%s:%d\n",                                               \
                         (void*)(ptr), (size_t)(alignment), __FILE__, __LINE__);  \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

/// Convenience macro for 64-byte alignment assertion (AVX-512)
#define DENSECORE_ASSERT_ALIGNED_64(ptr) DENSECORE_ASSERT_ALIGNED(ptr, 64)

namespace densecore {
namespace simd {

/**
 * @brief STL-compatible allocator providing aligned memory allocation
 *
 * Required for AVX-512 instructions that benefit from aligned loads/stores.
 * Default alignment is 64 bytes (AVX-512 cache line size).
 *
 * Usage:
 *   std::vector<float, AlignedAllocator<float>> vec;
 *   // or use the convenience alias:
 *   AlignedVector<float> vec;
 *
 * @tparam T Element type
 * @tparam Alignment Alignment in bytes (default: 64 for AVX-512)
 */
template <typename T, size_t Alignment = SIMD_ALIGNMENT> struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    // Required for std::vector rebind compatibility (C++17 allocator
    // requirements)
    template <class U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <class U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>& /*other*/) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0) return nullptr;

        size_t bytes = n * sizeof(T);
        T* ptr = nullptr;

#if defined(_WIN32)
        ptr = static_cast<T*>(_aligned_malloc(bytes, Alignment));
#else
        void* raw_ptr = nullptr;
        if (posix_memalign(&raw_ptr, Alignment, bytes) == 0) {
            ptr = static_cast<T*>(raw_ptr);
        }
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void deallocate(T* ptr, size_type /*n*/) noexcept {
        if (!ptr) return;

#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    template <class U> bool operator==(const AlignedAllocator<U, Alignment>& /*other*/) const noexcept { return true; }

    template <class U> bool operator!=(const AlignedAllocator<U, Alignment>& /*other*/) const noexcept { return false; }
};

/**
 * @brief Convenience type alias for 64-byte aligned vectors
 *
 * Use this instead of std::vector<T> when the data will be passed to
 * AVX-512 kernels that require or benefit from aligned memory.
 *
 * @tparam T Element type
 */
template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

// =============================================================================
// Runtime SIMD Detection
// =============================================================================

enum class SimdLevel {
    NONE = 0,
    SSE41 = 1,
    AVX = 2,
    AVX2 = 3,
    AVX512 = 4,
    AMX = 5,    // Intel Advanced Matrix Extensions (Sapphire Rapids+)
    NEON = 10,  // ARM NEON (fixed 128-bit vectors)
    SVE = 11,   // ARM SVE (scalable 256-bit+ vectors, Graviton 3)
    SVE2 = 12   // ARM SVE2 (Graviton 4, Neoverse V2)
};

inline SimdLevel DetectSimdLevel() {
#ifdef DENSECORE_X86
    // -------------------------------------------------------------------------
    // Runtime CPUID-based detection (portable binary support)
    // -------------------------------------------------------------------------
#if defined(_MSC_VER)
    // MSVC: Use __cpuid intrinsic
    int cpuid_info[4] = {0};

    // Check for AVX-512F: CPUID(7, 0).EBX bit 16
    __cpuidex(cpuid_info, 7, 0);
    bool has_avx512f = (cpuid_info[1] & (1 << 16)) != 0;

    // Check for AVX2: CPUID(7, 0).EBX bit 5
    bool has_avx2 = (cpuid_info[1] & (1 << 5)) != 0;

    // Check for FMA: CPUID(1, 0).ECX bit 12
    __cpuid(cpuid_info, 1);
    bool has_fma = (cpuid_info[2] & (1 << 12)) != 0;

    // Check for AVX: CPUID(1, 0).ECX bit 28
    bool has_avx = (cpuid_info[2] & (1 << 28)) != 0;

    // Check for SSE4.1: CPUID(1, 0).ECX bit 19
    bool has_sse41 = (cpuid_info[2] & (1 << 19)) != 0;

    // Check for AMX-TILE: CPUID(7, 0).EDX bit 24
    __cpuidex(cpuid_info, 7, 0);
    bool has_amx_tile = (cpuid_info[3] & (1 << 24)) != 0;

    if (has_amx_tile && has_avx512f) {
        return SimdLevel::AMX;
    } else if (has_avx512f) {
        return SimdLevel::AVX512;
    } else if (has_avx2 && has_fma) {
        return SimdLevel::AVX2;
    } else if (has_avx) {
        return SimdLevel::AVX;
    } else if (has_sse41) {
        return SimdLevel::SSE41;
    } else {
        return SimdLevel::NONE;
    }

#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: Use __builtin_cpu_supports (simpler and reliable)
    // NOTE: __builtin_cpu_init() is called automatically on modern compilers

    // Check from highest to lowest capability
    // NOTE: "amx-tile" support in __builtin_cpu_supports requires GCC 11+ or Clang 12+
#if (defined(__GNUC__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
    if (__builtin_cpu_supports("amx-tile") && __builtin_cpu_supports("avx512f")) {
        return SimdLevel::AMX;
    } else
#endif
        if (__builtin_cpu_supports("avx512f")) {
        return SimdLevel::AVX512;
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return SimdLevel::AVX2;
    } else if (__builtin_cpu_supports("avx")) {
        return SimdLevel::AVX;
    } else if (__builtin_cpu_supports("sse4.1")) {
        return SimdLevel::SSE41;
    } else {
        return SimdLevel::NONE;
    }

#else
    // Unknown compiler: Fallback to compile-time detection
#if defined(__AVX512F__)
    return SimdLevel::AVX512;
#elif defined(__AVX2__)
    return SimdLevel::AVX2;
#elif defined(__AVX__)
    return SimdLevel::AVX;
#elif defined(__SSE4_1__)
    return SimdLevel::SSE41;
#else
    return SimdLevel::NONE;
#endif
#endif

#elif defined(DENSECORE_ARM)
    // =========================================================================
    // ARM Runtime Detection via HWCAP (Linux) for portable binaries
    // =========================================================================
    // This enables a single binary to run on:
    // - Graviton2 / Raspberry Pi 4: NEON only
    // - Graviton3 / Neoverse V1: SVE (256-bit)
    // - Graviton4 / Neoverse V2: SVE2 (128-bit scalable, better instructions)
    // =========================================================================
#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // Check SVE2 first (Graviton 4, Neoverse V2)
    if (hwcap2 & HWCAP2_SVE2) {
        return SimdLevel::SVE2;
    }
    // Check SVE (Graviton 3, Neoverse V1)
    if (hwcap & HWCAP_SVE) {
        return SimdLevel::SVE;
    }
    // Fallback to NEON (always available on AArch64)
    return SimdLevel::NEON;
#elif defined(__APPLE__)
    // Apple Silicon: Always NEON (SVE not supported by Apple)
    return SimdLevel::NEON;
#else
    // Fallback for other ARM platforms: compile-time detection
#if defined(__ARM_FEATURE_SVE2)
    return SimdLevel::SVE2;
#elif defined(__ARM_FEATURE_SVE)
    return SimdLevel::SVE;
#else
    return SimdLevel::NEON;
#endif
#endif
#else
    return SimdLevel::NONE;
#endif
}

inline const char* SimdLevelName(SimdLevel level) {
    switch (level) {
    case SimdLevel::AMX: return "Intel AMX";
    case SimdLevel::AVX512: return "AVX-512";
    case SimdLevel::AVX2: return "AVX2";
    case SimdLevel::AVX: return "AVX";
    case SimdLevel::SSE41: return "SSE4.1";
    case SimdLevel::NEON: return "ARM NEON";
    case SimdLevel::SVE: return "ARM SVE";
    case SimdLevel::SVE2: return "ARM SVE2";
    default: return "Scalar";
    }
}

// -----------------------------------------------------------------------------
// ISA family helpers
// -----------------------------------------------------------------------------
// SimdLevel encodes both x86 and ARM families in one enum. Numeric ordering is
// not comparable across families (e.g. NEON has a larger enum value than
// AVX512), so use these predicates instead of direct >= comparisons.
inline bool IsX86Family(SimdLevel level) {
    switch (level) {
    case SimdLevel::SSE41:
    case SimdLevel::AVX:
    case SimdLevel::AVX2:
    case SimdLevel::AVX512:
    case SimdLevel::AMX: return true;
    default: return false;
    }
}

inline bool IsArmFamily(SimdLevel level) {
    switch (level) {
    case SimdLevel::NEON:
    case SimdLevel::SVE:
    case SimdLevel::SVE2: return true;
    default: return false;
    }
}

inline bool HasX86Avx2OrBetter(SimdLevel level) {
    return level == SimdLevel::AVX2 || level == SimdLevel::AVX512 || level == SimdLevel::AMX;
}

inline bool HasX86Avx512OrBetter(SimdLevel level) {
    return level == SimdLevel::AVX512 || level == SimdLevel::AMX;
}

inline bool HasIntelAmx(SimdLevel level) {
    return level == SimdLevel::AMX;
}

inline bool HasArmSveOrBetter(SimdLevel level) {
    return level == SimdLevel::SVE || level == SimdLevel::SVE2;
}

inline bool HasArmSve2(SimdLevel level) {
    return level == SimdLevel::SVE2;
}

inline SimdLevel GetCachedSimdLevel() {
    static const SimdLevel level = DetectSimdLevel();
    return level;
}

inline bool RuntimeHasArmSveOrBetter() {
#if defined(__ARM_FEATURE_SVE)
    return HasArmSveOrBetter(GetCachedSimdLevel());
#else
    return false;
#endif
}

// =============================================================================
// Thread Affinity & NUMA Topology
// =============================================================================

/**
 * Pin the calling thread to a specific CPU core.
 * For multi-socket servers, this prevents costly remote memory access.
 *
 * @param core_id The CPU core ID to pin to (0-indexed)
 * @return true on success, false on failure
 */
inline bool PinThreadToCore(int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    return result == 0;
#elif defined(_WIN32)
    // Windows: Use SetThreadAffinityMask
    if (core_id < 0 || core_id >= 64) return false;  // Windows mask limit
    DWORD_PTR mask = 1ULL << core_id;
    DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
    return prev != 0;
#else
    (void)core_id;
    return false;
#endif
}

/**
 * Get the number of NUMA nodes in the system.
 * Returns 1 for single-socket or non-NUMA systems.
 */
inline int GetNumaNodeCount() {
#if defined(__linux__)
    // Parse /sys/devices/system/node/nodeN directories
    int count = 0;
    for (int i = 0; i < 256; ++i) {
        char path[64];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
        if (access(path, F_OK) == 0) {
            ++count;
        } else {
            break;  // Nodes are numbered contiguously
        }
    }
    return count > 0 ? count : 1;
#elif defined(_WIN32)
    ULONG highest_node = 0;
    if (GetNumaHighestNodeNumber(&highest_node)) {
        return static_cast<int>(highest_node) + 1;
    }
    return 1;
#else
    return 1;
#endif
}

/**
 * Get list of CPU core IDs belonging to a NUMA node.
 * @param node_id NUMA node ID (0-indexed)
 * @return Vector of core IDs, empty on failure or unsupported platform
 */
inline std::vector<int> GetCoresInNumaNode(int node_id) {
    std::vector<int> cores;

#if defined(__linux__)
    // Parse /sys/devices/system/node/nodeN/cpulist
    // Format examples: "0-7", "0-7,16-23", "0,1,2,3"
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);

    FILE* f = fopen(path, "r");
    if (!f) return cores;

    char buf[256];
    if (fgets(buf, sizeof(buf), f)) {
        // Remove trailing newline
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

        // Parse comma-separated entries
        char* saveptr = nullptr;
        char* token = strtok_r(buf, ",", &saveptr);
        while (token) {
            // Check if it's a range (contains '-')
            char* dash = strchr(token, '-');
            if (dash) {
                int start = atoi(token);
                int end = atoi(dash + 1);
                for (int i = start; i <= end; ++i) {
                    cores.push_back(i);
                }
            } else {
                cores.push_back(atoi(token));
            }
            token = strtok_r(nullptr, ",", &saveptr);
        }
    }
    fclose(f);

#elif defined(_WIN32)
    ULONGLONG node_mask = 0;
    if (GetNumaNodeProcessorMask(static_cast<UCHAR>(node_id), &node_mask)) {
        for (int i = 0; i < 64; ++i) {
            if (node_mask & (1ULL << i)) {
                cores.push_back(i);
            }
        }
    }
#else
    (void)node_id;
#endif

    return cores;
}

/**
 * Get the number of available CPU cores.
 */
inline int GetNumCores() {
#if defined(__linux__)
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return static_cast<int>(sysinfo.dwNumberOfProcessors);
#else
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    return hw_threads > 0 ? static_cast<int>(hw_threads) : 1;
#endif
}

// =============================================================================
// Memory Prefetch
// =============================================================================

/**
 * Prefetch memory for read (T0 = all cache levels)
 */
inline void Prefetch(const void* ptr) {
#ifdef DENSECORE_X86
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
    __builtin_prefetch(ptr, 0, 3);  // read, high locality
#else
    (void)ptr;
#endif
}

/**
 * Prefetch a range of memory (cache-line aligned)
 */
inline void PrefetchRange(const void* ptr, size_t bytes) {
    constexpr size_t CACHE_LINE = 64;
    const char* p = reinterpret_cast<const char*>(ptr);
    for (size_t i = 0; i < bytes; i += CACHE_LINE) {
        Prefetch(p + i);
    }
}

/**
 * Prefetch for write (non-temporal if supported)
 */
inline void PrefetchWrite(void* ptr) {
#ifdef DENSECORE_X86
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
    __builtin_prefetch(ptr, 1, 3);  // write, high locality
#else
    (void)ptr;
#endif
}

// =============================================================================
// SIMD Copy Operations
// =============================================================================

#if defined(__ARM_FEATURE_SVE)
inline void SimdCopy_SVE(void* dst, const void* src, size_t bytes) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
    size_t i = 0;
    const uint64_t vl8 = svcntb();
    for (; i + vl8 <= bytes; i += vl8) {
        svuint8_t v = svld1_u8(svptrue_b8(), s + i);
        svst1_u8(svptrue_b8(), d + i, v);
    }
    if (i < bytes) {
        svbool_t pg = svwhilelt_b8_u64(0UL, static_cast<uint64_t>(bytes - i));
        svuint8_t v = svld1_u8(pg, s + i);
        svst1_u8(pg, d + i, v);
    }
}

inline void ScaleF32_SVE(float* dst, const float* src, float scale, size_t n) {
    const svfloat32_t vscale = svdup_f32(scale);
    size_t i = 0;
    const uint64_t vl = svcntw();
    for (; i + vl <= n; i += vl) {
        svfloat32_t v = svld1_f32(svptrue_b32(), src + i);
        v = svmul_f32_x(svptrue_b32(), v, vscale);
        svst1_f32(svptrue_b32(), dst + i, v);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(0UL, static_cast<uint64_t>(n - i));
        svfloat32_t v = svld1_f32(pg, src + i);
        v = svmul_f32_x(pg, v, vscale);
        svst1_f32(pg, dst + i, v);
    }
}

inline void AddF32_SVE(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    const uint64_t vl = svcntw();
    for (; i + vl <= n; i += vl) {
        svfloat32_t va = svld1_f32(svptrue_b32(), a + i);
        svfloat32_t vb = svld1_f32(svptrue_b32(), b + i);
        svst1_f32(svptrue_b32(), dst + i, svadd_f32_x(svptrue_b32(), va, vb));
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(0UL, static_cast<uint64_t>(n - i));
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svst1_f32(pg, dst + i, svadd_f32_x(pg, va, vb));
    }
}

inline float DotF32_SVE(const float* a, const float* b, size_t n) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;
    const uint64_t vl = svcntw();
    for (; i + vl <= n; i += vl) {
        svfloat32_t va = svld1_f32(svptrue_b32(), a + i);
        svfloat32_t vb = svld1_f32(svptrue_b32(), b + i);
        sum = svmla_f32_x(svptrue_b32(), sum, va, vb);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(0UL, static_cast<uint64_t>(n - i));
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        sum = svmla_f32_m(pg, sum, va, vb);
    }
    return svaddv_f32(svptrue_b32(), sum);
}

inline float MaxF32_SVE(const float* a, size_t n) {
    svfloat32_t vmax = svdup_f32(-1e30f);
    size_t i = 0;
    const uint64_t vl = svcntw();
    for (; i + vl <= n; i += vl) {
        svfloat32_t v = svld1_f32(svptrue_b32(), a + i);
        vmax = svmax_f32_x(svptrue_b32(), vmax, v);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(0UL, static_cast<uint64_t>(n - i));
        svfloat32_t v = svld1_f32(pg, a + i);
        vmax = svmax_f32_m(pg, vmax, v);
    }
    return svmaxv_f32(svptrue_b32(), vmax);
}

inline float SumF32_SVE(const float* a, size_t n) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;
    const uint64_t vl = svcntw();
    for (; i + vl <= n; i += vl) {
        svfloat32_t v = svld1_f32(svptrue_b32(), a + i);
        sum = svadd_f32_x(svptrue_b32(), sum, v);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b32_u64(0UL, static_cast<uint64_t>(n - i));
        svfloat32_t v = svld1_f32(pg, a + i);
        sum = svadd_f32_m(pg, sum, v);
    }
    return svaddv_f32(svptrue_b32(), sum);
}

inline void MeanPool_SVE(const float* input, float* output, int seq_len, int hidden_dim) {
    const uint64_t vl = svcntw();
    const svfloat32_t vdiv = svdup_f32(1.0f / seq_len);
    for (int d = 0; d < hidden_dim; d += static_cast<int>(vl)) {
        svbool_t pg = svwhilelt_b32_s32(d, hidden_dim);
        svfloat32_t sum = svdup_f32(0.0f);
        for (int s = 0; s < seq_len; s++) {
            svfloat32_t v = svld1_f32(pg, input + s * hidden_dim + d);
            sum = svadd_f32_m(pg, sum, v);
        }
        sum = svmul_f32_x(pg, sum, vdiv);
        svst1_f32(pg, output + d, sum);
    }
}

inline void MaxPool_SVE(const float* input, float* output, int seq_len, int hidden_dim) {
    const uint64_t vl = svcntw();
    for (int d = 0; d < hidden_dim; d += static_cast<int>(vl)) {
        svbool_t pg = svwhilelt_b32_s32(d, hidden_dim);
        svfloat32_t vmax = svld1_f32(pg, input + d);
        for (int s = 1; s < seq_len; s++) {
            svfloat32_t v = svld1_f32(pg, input + s * hidden_dim + d);
            vmax = svmax_f32_m(pg, vmax, v);
        }
        svst1_f32(pg, output + d, vmax);
    }
}
#endif

/**
 * Fast memory copy using SIMD (aligned or unaligned)
 */
inline void SimdCopy(void* dst, const void* src, size_t bytes) {
#if defined(__AVX512F__)
    // AVX-512: 64 bytes per iteration
    const size_t vec_size = 64;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(reinterpret_cast<const char*>(src) + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    // Handle remainder
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(__AVX2__)
    // AVX2: 32 bytes per iteration
    const size_t vec_size = 32;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(src) + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(__SSE2__)
    // SSE2: 16 bytes per iteration
    const size_t vec_size = 16;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const char*>(src) + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(DENSECORE_ARM)
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        SimdCopy_SVE(dst, src, bytes);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        // NEON: 16 bytes per iteration
        const size_t vec_size = 16;
        size_t i = 0;
        const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
        uint8_t* d = reinterpret_cast<uint8_t*>(dst);
        for (; i + vec_size <= bytes; i += vec_size) {
            uint8x16_t v = vld1q_u8(s + i);
            vst1q_u8(d + i, v);
        }
        if (i < bytes) {
            memcpy(d + i, s + i, bytes - i);
        }
    }
#else
    memcpy(dst, src, bytes);
#endif
}

// =============================================================================
// Float32 Operations
// =============================================================================

/**
 * Copy float32 array using SIMD
 */
inline void CopyF32(float* dst, const float* src, size_t n) {
    SimdCopy(dst, src, n * sizeof(float));
}

/**
 * Scale float32 array: dst[i] = src[i] * scale
 */
inline void ScaleF32(float* dst, const float* src, float scale, size_t n) {
#if defined(__AVX2__)
    __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        v = _mm256_mul_ps(v, vscale);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; i++) {
        dst[i] = src[i] * scale;
    }
#elif defined(DENSECORE_ARM)
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        ScaleF32_SVE(dst, src, scale, n);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        float32x4_t vscale = vdupq_n_f32(scale);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(src + i);
            v = vmulq_f32(v, vscale);
            vst1q_f32(dst + i, v);
        }
        for (; i < n; i++) {
            dst[i] = src[i] * scale;
        }
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * scale;
    }
#endif
}

/**
 * Add float32 arrays: dst[i] = a[i] + b[i]
 */
inline void AddF32(float* dst, const float* a, const float* b, size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
#elif defined(DENSECORE_ARM)
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        AddF32_SVE(dst, a, b, n);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(dst + i, vc);
        }
        for (; i < n; i++) {
            dst[i] = a[i] + b[i];
        }
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
#endif
}

/**
 * Dot product of float32 arrays
 */
inline float DotF32(const float* a, const float* b, size_t n) {
#if defined(__AVX2__) && DENSECORE_HAS_FMA
    // AVX2 + FMA path (Intel Haswell+, AMD Zen+)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    // Remainder
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#elif defined(__AVX2__)
    // AVX2 without FMA (rare: some AMD Piledriver CPUs)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    // Remainder
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#elif defined(DENSECORE_ARM)
    float result = 0.0f;
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        result = DotF32_SVE(a, b, n);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            sum = vmlaq_f32(sum, va, vb);
        }
        result = vaddvq_f32(sum);
        for (; i < n; i++) {
            result += a[i] * b[i];
        }
    }
    return result;
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

/**
 * Find max value in float32 array
 */
inline float MaxF32(const float* a, size_t n) {
    if (n == 0) return 0.0f;

#if defined(__AVX2__)
    __m256 vmax = _mm256_set1_ps(-1e30f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    // Reduce
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
    float result = _mm_cvtss_f32(max128);
    for (; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#elif defined(DENSECORE_ARM)
    float result = -1e30f;
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        result = MaxF32_SVE(a, n);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        float32x4_t vmax = vdupq_n_f32(-1e30f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(a + i);
            vmax = vmaxq_f32(vmax, v);
        }
        result = vmaxvq_f32(vmax);
        for (; i < n; i++) {
            if (a[i] > result) result = a[i];
        }
    }
    return result;
#else
    float result = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#endif
}

/**
 * Sum of float32 array
 */
inline float SumF32(const float* a, size_t n) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        sum = _mm256_add_ps(sum, v);
    }
    // Reduce
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    for (; i < n; i++) {
        result += a[i];
    }
    return result;
#elif defined(DENSECORE_ARM)
    float result = 0.0f;
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        result = SumF32_SVE(a, n);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(a + i);
            sum = vaddq_f32(sum, v);
        }
        result = vaddvq_f32(sum);
        for (; i < n; i++) {
            result += a[i];
        }
    }
    return result;
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result += a[i];
    }
    return result;
#endif
}

/**
 * Softmax in-place: a[i] = exp(a[i] - max) / sum(exp)
 */
inline void SoftmaxF32(float* a, size_t n) {
    float max_val = MaxF32(a, n);

#if defined(__AVX2__)
    // Note: Using fast exp approximation would be better here
    // For now, use scalar exp
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        a[i] = expf(a[i] - max_val);
        sum += a[i];
    }
    float inv_sum = 1.0f / sum;
    ScaleF32(a, a, inv_sum, n);
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        a[i] = expf(a[i] - max_val);
        sum += a[i];
    }
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        a[i] *= inv_sum;
    }
#endif
}

// =============================================================================
// FP16 Conversion (using ggml)
// =============================================================================

/**
 * Convert FP32 to FP16 using SIMD
 */
inline void ConvertF32ToF16(ggml_fp16_t* dst, const float* src, size_t n) {
    ggml_fp32_to_fp16_row(src, dst, n);
}

/**
 * Convert FP16 to FP32 using SIMD
 */
inline void ConvertF16ToF32(float* dst, const ggml_fp16_t* src, size_t n) {
    ggml_fp16_to_fp32_row(src, dst, n);
}

// =============================================================================
// Matrix Operations (used by Flash Attention)
// =============================================================================

/**
 * Matrix multiply: C = A @ B^T (for attention scores)
 * A: [M, K], B: [N, K] (row-major), C: [M, N]
 */
inline void MatMulTransB(float* C, const float* A, const float* B, int M, int N, int K) {
    // Simple implementation - can be optimized further
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = DotF32(A + m * K, B + n * K, K);
        }
    }
}

// =============================================================================
// Embedding Operations (for RAG/Semantic Search)
// =============================================================================

/**
 * L2 Normalize a vector in-place: v[i] = v[i] / ||v||
 * Essential for cosine similarity in embedding models
 */
inline void NormalizeL2(float* data, size_t n) {
    // Compute L2 norm using SIMD dot product
    float norm_sq = DotF32(data, data, n);
    if (norm_sq < 1e-12f) return;  // Avoid division by zero

    float inv_norm = 1.0f / sqrtf(norm_sq);
    ScaleF32(data, data, inv_norm, n);
}

/**
 * Mean pooling over sequence dimension
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 * Standard pooling for sentence-transformers models
 */
inline void MeanPool(const float* input, float* output, int seq_len, int hidden_dim) {
    if (seq_len <= 0) return;

    // Initialize output to zero
    std::memset(output, 0, hidden_dim * sizeof(float));

#if defined(__AVX2__)
    // Process 8 floats at a time
    for (int d = 0; d < hidden_dim; d += 8) {
        int remaining = (d + 8 <= hidden_dim) ? 8 : hidden_dim - d;
        if (remaining == 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int s = 0; s < seq_len; s++) {
                __m256 v = _mm256_loadu_ps(input + s * hidden_dim + d);
                sum = _mm256_add_ps(sum, v);
            }
            // Divide by seq_len
            __m256 div = _mm256_set1_ps(1.0f / seq_len);
            sum = _mm256_mul_ps(sum, div);
            _mm256_storeu_ps(output + d, sum);
        } else {
            // Handle remainder
            for (int dd = d; dd < hidden_dim; dd++) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; s++) {
                    sum += input[s * hidden_dim + dd];
                }
                output[dd] = sum / seq_len;
            }
            break;
        }
    }
#elif defined(DENSECORE_ARM)
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        MeanPool_SVE(input, output, seq_len, hidden_dim);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        for (int d = 0; d < hidden_dim; d += 4) {
            int remaining = (d + 4 <= hidden_dim) ? 4 : hidden_dim - d;
            if (remaining == 4) {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (int s = 0; s < seq_len; s++) {
                    float32x4_t v = vld1q_f32(input + s * hidden_dim + d);
                    sum = vaddq_f32(sum, v);
                }
                float32x4_t div = vdupq_n_f32(1.0f / seq_len);
                sum = vmulq_f32(sum, div);
                vst1q_f32(output + d, sum);
            } else {
                for (int dd = d; dd < hidden_dim; dd++) {
                    float sum = 0.0f;
                    for (int s = 0; s < seq_len; s++) {
                        sum += input[s * hidden_dim + dd];
                    }
                    output[dd] = sum / seq_len;
                }
                break;
            }
        }
    }
#else
    // Scalar fallback
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < hidden_dim; d++) {
            output[d] += input[s * hidden_dim + d];
        }
    }
    float inv_len = 1.0f / seq_len;
    for (int d = 0; d < hidden_dim; d++) {
        output[d] *= inv_len;
    }
#endif
}

/**
 * Mean pooling with attention mask
 * Only pools over non-masked positions (mask[i] = 1 means valid)
 */
inline void MeanPoolMasked(const float* input, const int* mask, float* output, int seq_len, int hidden_dim) {
    std::memset(output, 0, hidden_dim * sizeof(float));

    int valid_count = 0;
    for (int s = 0; s < seq_len; s++) {
        if (mask[s]) {
            valid_count++;
            for (int d = 0; d < hidden_dim; d++) {
                output[d] += input[s * hidden_dim + d];
            }
        }
    }

    if (valid_count > 0) {
        float inv_count = 1.0f / valid_count;
        ScaleF32(output, output, inv_count, hidden_dim);
    }
}

/**
 * Max pooling over sequence dimension
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void MaxPool(const float* input, float* output, int seq_len, int hidden_dim) {
    if (seq_len <= 0) return;

    // Initialize with first row
    std::memcpy(output, input, hidden_dim * sizeof(float));

#if defined(__AVX2__)
    for (int d = 0; d < hidden_dim; d += 8) {
        int remaining = (d + 8 <= hidden_dim) ? 8 : hidden_dim - d;
        if (remaining == 8) {
            __m256 vmax = _mm256_loadu_ps(input + d);
            for (int s = 1; s < seq_len; s++) {
                __m256 v = _mm256_loadu_ps(input + s * hidden_dim + d);
                vmax = _mm256_max_ps(vmax, v);
            }
            _mm256_storeu_ps(output + d, vmax);
        } else {
            for (int dd = d; dd < hidden_dim; dd++) {
                float maxv = input[dd];
                for (int s = 1; s < seq_len; s++) {
                    float v = input[s * hidden_dim + dd];
                    if (v > maxv) maxv = v;
                }
                output[dd] = maxv;
            }
            break;
        }
    }
#elif defined(DENSECORE_ARM)
    // Try SVE path if runtime supports it and code is compiled with SVE
    bool sve_executed = false;
    if (RuntimeHasArmSveOrBetter()) {
#if defined(__ARM_FEATURE_SVE)
        MaxPool_SVE(input, output, seq_len, hidden_dim);
        sve_executed = true;
#endif
    }

    // Fallback to NEON if SVE was not available or failed
    if (!sve_executed) {
        for (int d = 0; d < hidden_dim; d += 4) {
            int remaining = (d + 4 <= hidden_dim) ? 4 : hidden_dim - d;
            if (remaining == 4) {
                float32x4_t vmax = vld1q_f32(input + d);
                for (int s = 1; s < seq_len; s++) {
                    float32x4_t v = vld1q_f32(input + s * hidden_dim + d);
                    vmax = vmaxq_f32(vmax, v);
                }
                vst1q_f32(output + d, vmax);
            } else {
                // Handle remaining elements with scalar code
                for (int dd = d; dd < hidden_dim; dd++) {
                    float maxv = input[dd];
                    for (int s = 1; s < seq_len; s++) {
                        float v = input[s * hidden_dim + dd];
                        if (v > maxv) maxv = v;
                    }
                    output[dd] = maxv;
                }
                break;
            }
        }
    }
#else
    // Scalar fallback
    for (int s = 1; s < seq_len; s++) {
        for (int d = 0; d < hidden_dim; d++) {
            float v = input[s * hidden_dim + d];
            if (v > output[d]) output[d] = v;
        }
    }
#endif
}

/**
 * CLS pooling - extract first token
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void ClsPool(const float* input, float* output, int hidden_dim) {
    CopyF32(output, input, hidden_dim);
}

/**
 * Last token pooling
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void LastPool(const float* input, float* output, int seq_len, int hidden_dim) {
    if (seq_len <= 0) return;
    CopyF32(output, input + (seq_len - 1) * hidden_dim, hidden_dim);
}

/**
 * Batch L2 normalization
 * Normalize each row of a [batch_size, dim] matrix
 */
inline void BatchNormalizeL2(float* data, int batch_size, int dim) {
    for (int b = 0; b < batch_size; b++) {
        NormalizeL2(data + b * dim, dim);
    }
}

/**
 * Cosine similarity between two vectors
 */
inline float CosineSimilarity(const float* a, const float* b, size_t n) {
    float dot = DotF32(a, b, n);
    float norm_a = sqrtf(DotF32(a, a, n));
    float norm_b = sqrtf(DotF32(b, b, n));
    if (norm_a < 1e-12f || norm_b < 1e-12f) return 0.0f;
    return dot / (norm_a * norm_b);
}

// =============================================================================
// INT4 GEMM Kernels (declarations)
// =============================================================================

void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W, const float* scales, const float* zeros, int M,
                         int N, int K, int group_size);

void GemmInt4Fp32_AVX2(float* C, const float* A, const uint8_t* W, const float* scales, const float* zeros, int M,
                       int N, int K, int group_size);

void GemmInt4Fp32_SVE(float* C, const float* A, const uint8_t* W, const float* scales, const float* zeros, int M, int N,
                      int K, int group_size);

void GemmInt4Fp32_NEON(float* C, const float* A, const uint8_t* W, const float* scales, const float* zeros, int M,
                       int N, int K, int group_size);

// =============================================================================
// Rotary Positional Embedding (RoPE) - HIGHLY OPTIMIZED
// =============================================================================

/**
 * @brief Pre-computed RoPE frequency table for a given context length
 *
 * Pre-computes cos(m * theta) and sin(m * theta) for all positions m and
 * all frequency dimensions. This avoids expensive transcendental function
 * calls in the inner loop.
 *
 * Layout: [max_seq_len, head_dim / 2] where each entry is (cos, sin) pair
 * Access: cos_sin_table[pos * head_dim + 2*d] = cos(pos * theta_d)
 *         cos_sin_table[pos * head_dim + 2*d + 1] = sin(pos * theta_d)
 */
struct RoPETable {
    std::vector<float> cos_sin;  ///< Interleaved [cos, sin, cos, sin, ...]
    int max_seq_len;
    int head_dim;
    float freq_base;

    /**
     * @brief Initialize RoPE table for given parameters
     *
     * @param max_len Maximum sequence length to pre-compute
     * @param dim Head dimension (must be even)
     * @param base RoPE frequency base (default 10000.0 for Llama)
     */
    void Init(int max_len, int dim, float base = 10000.0f) {
        max_seq_len = max_len;
        head_dim = dim;
        freq_base = base;

        // Allocate interleaved cos/sin: [max_len, head_dim]
        cos_sin.resize(static_cast<size_t>(max_len) * dim);

        // Pre-compute frequencies: theta[d] = 1 / (base ** (2d / dim))
        std::vector<float> freqs(dim / 2);
        for (int d = 0; d < dim / 2; d++) {
            float exp = (2.0f * d) / static_cast<float>(dim);
            freqs[d] = 1.0f / std::pow(base, exp);
        }

        // Compute cos/sin for all positions
        for (int pos = 0; pos < max_len; pos++) {
            for (int d = 0; d < dim / 2; d++) {
                float angle = static_cast<float>(pos) * freqs[d];
                cos_sin[pos * dim + 2 * d] = std::cos(angle);
                cos_sin[pos * dim + 2 * d + 1] = std::sin(angle);
            }
        }
    }

    /**
     * @brief Get cos value for position and dimension pair index
     */
    inline float Cos(int pos, int pair_d) const { return cos_sin[pos * head_dim + 2 * pair_d]; }

    /**
     * @brief Get sin value for position and dimension pair index
     */
    inline float Sin(int pos, int pair_d) const { return cos_sin[pos * head_dim + 2 * pair_d + 1]; }

    /**
     * @brief Get pointer to cos/sin data for a specific position
     */
    inline const float* GetCosSinPtr(int pos) const { return cos_sin.data() + pos * head_dim; }
};

// Legacy ISA-specific RoPE variants removed — replaced by Highway (hwy_rope.cc)

/**
 * @brief Apply Rotary Positional Embedding (unified entry point)
 *
 * Delegates to Google Highway for portable SIMD across all ISAs.
 */
inline void ApplyRoPE(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens,
                      int head_dim, int rope_dim, int max_seq_len, int ith = 0, int nth = 1) {
    // Defined in hwy_rope.cc — Highway auto-selects best ISA at runtime.
    extern void ApplyRoPE_HwyDispatch(float*, const float*, const float*, const int*, int, int, int, int, int, int);
    ApplyRoPE_HwyDispatch(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, max_seq_len, ith, nth);
}

// NOTE: ApplyRoPE_MultiHead was removed as it was unused.
// If multi-head RoPE is needed, use cb_rope_avx512 callback in inference.cpp
// which handles the [head_dim, n_heads, n_tokens] GGML tensor layout.

// =============================================================================
// INT4 Quantized GEMM (AVX512) - HIGHLY OPTIMIZED
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Unpack 64 × 4-bit signed integers using AVX512 (OPTIMIZED)
 *
 * Uses shift trick for sign extension:
 * 1. Shift left 12 bits (positions 4-bit value at top of 16-bit)
 * 2. Arithmetic right shift 12 bits (propagates sign bit)
 *
 * Processes 32 bytes (64 packed 4-bit values) into 64 × int16_t
 *
 * @param packed Input: 32 bytes containing 64 packed 4-bit values
 * @param unpacked_lo Output: first 32 × int16_t values (low 16 bytes input)
 * @param unpacked_hi Output: second 32 × int16_t values (high 16 bytes input)
 */
inline void UnpackInt4x64_AVX512(const uint8_t* packed, __m512i& unpacked_lo, __m512i& unpacked_hi) {
    // Load 32 bytes = 64 packed 4-bit values into AVX512 register
    __m256i packed_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));

    // Expand 8-bit elements to 16-bit
    __m512i packed_16 = _mm512_cvtepu8_epi16(packed_256);

    // Extract lower nibbles (bits 0-3)
    __m512i low_nibbles = _mm512_and_si512(packed_16, _mm512_set1_epi16(0x0F));

    // Extract upper nibbles (bits 4-7), shift down
    __m512i high_nibbles = _mm512_srli_epi16(packed_16, 4);
    high_nibbles = _mm512_and_si512(high_nibbles, _mm512_set1_epi16(0x0F));

    // Sign extension using shift trick:
    // slli by 12 moves 4-bit value to bits 12-15 (sign bit at position 15)
    // srai by 12 propagates sign bit through bits 4-15
    low_nibbles = _mm512_slli_epi16(low_nibbles, 12);
    low_nibbles = _mm512_srai_epi16(low_nibbles, 12);

    high_nibbles = _mm512_slli_epi16(high_nibbles, 12);
    high_nibbles = _mm512_srai_epi16(high_nibbles, 12);

    // Interleave: [a0, b0, a1, b1, ...] where a=low, b=high
    // unpacklo takes low halves of each lane, unpackhi takes high halves
    __m512i interleaved_lo = _mm512_unpacklo_epi16(low_nibbles, high_nibbles);
    __m512i interleaved_hi = _mm512_unpackhi_epi16(low_nibbles, high_nibbles);

    // The 512-bit unpack works on 256-bit lanes, so we need to permute
    // to get correct final order
    unpacked_lo =
        _mm512_permutex2var_epi64(interleaved_lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), interleaved_hi);
    unpacked_hi =
        _mm512_permutex2var_epi64(interleaved_lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), interleaved_hi);
}

/**
 * @brief High-performance GEMM kernel: C = A * W^T (HIGHLY OPTIMIZED)
 *
 * Optimizations:
 * - 16x register blocking along N dimension (maximum ILP)
 * - Shift-based sign extension (no branches, no masks)
 * - Aggressive prefetching (128 bytes ahead)
 * - Minimized loop overhead with unrolling
 *
 * Performance target: >90% of theoretical AVX512 peak
 */
inline void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;  // Bytes per weight row

    // Constants for prefetch distance
    constexpr int PREFETCH_DIST = 128;  // Bytes ahead

    // Outer loop: rows of output
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

        // Middle loop: columns of output in blocks of 16 (REGISTER BLOCKING)
        int n = 0;
        for (; n + 16 <= N; n += 16) {
            // 16 accumulators for 16 output elements
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            __m512 acc8 = _mm512_setzero_ps();
            __m512 acc9 = _mm512_setzero_ps();
            __m512 acc10 = _mm512_setzero_ps();
            __m512 acc11 = _mm512_setzero_ps();
            __m512 acc12 = _mm512_setzero_ps();
            __m512 acc13 = _mm512_setzero_ps();
            __m512 acc14 = _mm512_setzero_ps();
            __m512 acc15 = _mm512_setzero_ps();

            // Loop over quantization groups
            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;

                // Prefetch activations for next group
                if (g + 1 < num_groups) {
                    _mm_prefetch(reinterpret_cast<const char*>(a_row + (g + 1) * group_size), _MM_HINT_T0);
                }

                // Process 32 weights at a time (64 bits packed = 16 bytes)
                for (int k = 0; k < group_size; k += 32) {
                    // Load 2 × 16 floats from activations
                    __m512 a0 = _mm512_loadu_ps(a_ptr + k);
                    __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

// Process all 16 weight rows with 2-way unrolling
#define PROCESS_ROW(idx)                                                                 \
    do {                                                                                 \
        const int row = n + (idx);                                                       \
        const float scale = scales[row * num_groups + g];                                \
        const float zero = zero_points[row * num_groups + g];                            \
        const __m512 vscale = _mm512_set1_ps(scale);                                     \
        const __m512 vzero = _mm512_set1_ps(zero);                                       \
                                                                                         \
        const uint8_t* w_ptr = W_int4 + row * packed_K + packed_offset + k / 2;          \
        _mm_prefetch(reinterpret_cast<const char*>(w_ptr + PREFETCH_DIST), _MM_HINT_T0); \
                                                                                         \
        /* Load 16 bytes = 32 packed weights */                                          \
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr));       \
        __m256i packed_256 = _mm256_cvtepu8_epi16(packed);                               \
                                                                                         \
        /* Extract low nibbles */                                                        \
        __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));             \
        /* Extract high nibbles */                                                       \
        __m256i high = _mm256_srli_epi16(packed_256, 4);                                 \
        high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));                          \
                                                                                         \
        /* Sign extension via shift trick */                                             \
        low = _mm256_slli_epi16(low, 12);                                                \
        low = _mm256_srai_epi16(low, 12);                                                \
        high = _mm256_slli_epi16(high, 12);                                              \
        high = _mm256_srai_epi16(high, 12);                                              \
                                                                                         \
        /* Interleave to restore order */                                                \
        __m256i interleaved_lo = _mm256_unpacklo_epi16(low, high);                       \
        __m256i interleaved_hi = _mm256_unpackhi_epi16(low, high);                       \
                                                                                         \
        /* Permute for correct lane order after interleave */                            \
        __m256i w16_0 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20); \
        __m256i w16_1 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31); \
                                                                                         \
        /* Convert to FP32 */                                                            \
        __m512i w32_0 = _mm512_cvtepi16_epi32(w16_0);                                    \
        __m512i w32_1 = _mm512_cvtepi16_epi32(w16_1);                                    \
        __m512 wf0 = _mm512_cvtepi32_ps(w32_0);                                          \
        __m512 wf1 = _mm512_cvtepi32_ps(w32_1);                                          \
                                                                                         \
        /* Dequantize: w_dequant = scale * (q - zero) */                                 \
        wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));                          \
        wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));                          \
                                                                                         \
        /* FMA */                                                                        \
        acc##idx = _mm512_fmadd_ps(a0, wf0, acc##idx);                                   \
        acc##idx = _mm512_fmadd_ps(a1, wf1, acc##idx);                                   \
    } while (0)

                    PROCESS_ROW(0);
                    PROCESS_ROW(1);
                    PROCESS_ROW(2);
                    PROCESS_ROW(3);
                    PROCESS_ROW(4);
                    PROCESS_ROW(5);
                    PROCESS_ROW(6);
                    PROCESS_ROW(7);
                    PROCESS_ROW(8);
                    PROCESS_ROW(9);
                    PROCESS_ROW(10);
                    PROCESS_ROW(11);
                    PROCESS_ROW(12);
                    PROCESS_ROW(13);
                    PROCESS_ROW(14);
                    PROCESS_ROW(15);

#undef PROCESS_ROW
                }
            }

            // Horizontal reduction and store results
            C[m * N + n + 0] = _mm512_reduce_add_ps(acc0);
            C[m * N + n + 1] = _mm512_reduce_add_ps(acc1);
            C[m * N + n + 2] = _mm512_reduce_add_ps(acc2);
            C[m * N + n + 3] = _mm512_reduce_add_ps(acc3);
            C[m * N + n + 4] = _mm512_reduce_add_ps(acc4);
            C[m * N + n + 5] = _mm512_reduce_add_ps(acc5);
            C[m * N + n + 6] = _mm512_reduce_add_ps(acc6);
            C[m * N + n + 7] = _mm512_reduce_add_ps(acc7);
            C[m * N + n + 8] = _mm512_reduce_add_ps(acc8);
            C[m * N + n + 9] = _mm512_reduce_add_ps(acc9);
            C[m * N + n + 10] = _mm512_reduce_add_ps(acc10);
            C[m * N + n + 11] = _mm512_reduce_add_ps(acc11);
            C[m * N + n + 12] = _mm512_reduce_add_ps(acc12);
            C[m * N + n + 13] = _mm512_reduce_add_ps(acc13);
            C[m * N + n + 14] = _mm512_reduce_add_ps(acc14);
            C[m * N + n + 15] = _mm512_reduce_add_ps(acc15);
        }

        // Handle remaining columns (N % 16)
        for (; n < N; n++) {
            __m512 acc = _mm512_setzero_ps();

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];
                const __m512 vscale = _mm512_set1_ps(scale);
                const __m512 vzero = _mm512_set1_ps(zero);
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 32) {
                    // Load 16 bytes = 32 packed weights
                    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr + k / 2));
                    __m256i packed_256 = _mm256_cvtepu8_epi16(packed);

                    // Extract and sign-extend nibbles
                    __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));
                    __m256i high = _mm256_srli_epi16(packed_256, 4);
                    high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));

                    low = _mm256_slli_epi16(low, 12);
                    low = _mm256_srai_epi16(low, 12);
                    high = _mm256_slli_epi16(high, 12);
                    high = _mm256_srai_epi16(high, 12);

                    // Interleave and permute
                    __m256i lo = _mm256_unpacklo_epi16(low, high);
                    __m256i hi = _mm256_unpackhi_epi16(low, high);
                    __m256i w16_0 = _mm256_permute2x128_si256(lo, hi, 0x20);
                    __m256i w16_1 = _mm256_permute2x128_si256(lo, hi, 0x31);

                    // Convert and dequantize
                    __m512 wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_0));
                    __m512 wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_1));

                    wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));
                    wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));

                    // Load activations and FMA
                    __m512 a0 = _mm512_loadu_ps(a_ptr + k);
                    __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

                    acc = _mm512_fmadd_ps(a0, wf0, acc);
                    acc = _mm512_fmadd_ps(a1, wf1, acc);
                }
            }

            C[m * N + n] = _mm512_reduce_add_ps(acc);
        }
    }
}

#else  // No AVX512 support

/**
 * Fallback GEMM for INT4 weights (scalar implementation)
 */
inline void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int g = 0; g < num_groups; g++) {
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];

                const int k_start = g * group_size;
                const uint8_t* w_packed = W_int4 + n * (K / 2) + g * (group_size / 2);

                for (int k = 0; k < group_size; k++) {
                    // Unpack 4-bit weight
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = w_packed[byte_idx];

                    int8_t q;
                    if (nibble_idx == 0) {
                        q = static_cast<int8_t>(packed_byte & 0x0F);
                    } else {
                        q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    }

                    // Sign extend
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    // Dequantize and accumulate
                    float w_dequant = scale * (static_cast<float>(q) - zero);
                    sum += A[m * K + k_start + k] * w_dequant;
                }
            }

            C[m * N + n] = sum;
        }
    }
}

#endif  // __AVX512F__

#if defined(__AVX2__) && DENSECORE_HAS_FMA

/**
 * @brief High-performance AVX2 GEMM kernel: C = A * W^T (INT4 weights)
 *
 * Optimizations:
 * - 8x register blocking along N dimension (vs 16x for AVX-512)
 * - Shift-based sign extension (no branches, no masks)
 * - Prefetching for weights and activations
 * - FMA3 instructions for dot products
 * - Processes 16 weights per iteration (vs 32 for AVX-512)
 *
 * @param C Output [M, N]
 * @param A Input activations [M, K]
 * @param W_int4 Packed INT4 weights [N, K/2]
 * @param scales Per-group scales [N, num_groups]
 * @param zero_points Per-group zero points [N, num_groups]
 * @param M Batch dimension
 * @param N Output features
 * @param K Input features
 * @param group_size Quantization group size (K must be divisible)
 */
inline void GemmInt4Fp32_AVX2(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                              const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;  // Bytes per weight row

    // Constants for prefetch distance
    constexpr int PREFETCH_DIST = 64;  // Bytes ahead

    // Outer loop: rows of output
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

        // Middle loop: columns of output in blocks of 8 (REGISTER BLOCKING)
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            // 8 accumulators for 8 output elements
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            // Loop over quantization groups
            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;

                // Prefetch activations for next group
                if (g + 1 < num_groups) {
                    _mm_prefetch(reinterpret_cast<const char*>(a_row + (g + 1) * group_size), _MM_HINT_T0);
                }

                // Process 16 weights at a time (8 bytes packed)
                for (int k = 0; k < group_size; k += 16) {
                    // Load 1 × 8 floats from activations
                    __m256 a0 = _mm256_loadu_ps(a_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

// Process all 8 weight rows
#define PROCESS_ROW_AVX2(idx)                                                            \
    do {                                                                                 \
        const int row = n + (idx);                                                       \
        const float scale = scales[row * num_groups + g];                                \
        const float zero = zero_points[row * num_groups + g];                            \
        const __m256 vscale = _mm256_set1_ps(scale);                                     \
        const __m256 vzero = _mm256_set1_ps(zero);                                       \
                                                                                         \
        const uint8_t* w_ptr = W_int4 + row * packed_K + packed_offset + k / 2;          \
        _mm_prefetch(reinterpret_cast<const char*>(w_ptr + PREFETCH_DIST), _MM_HINT_T0); \
                                                                                         \
        /* Load 8 bytes = 16 packed weights */                                           \
        __m128i packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr));    \
        __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);                                \
                                                                                         \
        /* Extract low nibbles (bits 0-3) */                                             \
        __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));                    \
        /* Extract high nibbles (bits 4-7) */                                            \
        __m128i high = _mm_srli_epi16(packed_16, 4);                                     \
        high = _mm_and_si128(high, _mm_set1_epi16(0x0F));                                \
                                                                                         \
        /* Sign extension via shift trick */                                             \
        low = _mm_slli_epi16(low, 12);                                                   \
        low = _mm_srai_epi16(low, 12);                                                   \
        high = _mm_slli_epi16(high, 12);                                                 \
        high = _mm_srai_epi16(high, 12);                                                 \
                                                                                         \
        /* Interleave to restore order: [l0, h0, l1, h1, l2, h2, l3, h3] */              \
        __m128i interleaved_lo = _mm_unpacklo_epi16(low, high);                          \
        __m128i interleaved_hi = _mm_unpackhi_epi16(low, high);                          \
                                                                                         \
        /* Convert to FP32 via int32 */                                                  \
        __m256i w32_0 = _mm256_cvtepi16_epi32(interleaved_lo);                           \
        __m256i w32_1 = _mm256_cvtepi16_epi32(interleaved_hi);                           \
        __m256 wf0 = _mm256_cvtepi32_ps(w32_0);                                          \
        __m256 wf1 = _mm256_cvtepi32_ps(w32_1);                                          \
                                                                                         \
        /* Dequantize: w_dequant = scale * (q - zero) */                                 \
        wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));                          \
        wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));                          \
                                                                                         \
        /* FMA */                                                                        \
        acc##idx = _mm256_fmadd_ps(a0, wf0, acc##idx);                                   \
        acc##idx = _mm256_fmadd_ps(a1, wf1, acc##idx);                                   \
    } while (0)

                    PROCESS_ROW_AVX2(0);
                    PROCESS_ROW_AVX2(1);
                    PROCESS_ROW_AVX2(2);
                    PROCESS_ROW_AVX2(3);
                    PROCESS_ROW_AVX2(4);
                    PROCESS_ROW_AVX2(5);
                    PROCESS_ROW_AVX2(6);
                    PROCESS_ROW_AVX2(7);

#undef PROCESS_ROW_AVX2
                }
            }

            // Horizontal reduction and store results
            // AVX2 doesn't have _mm256_reduce_add_ps, so we do it manually
            auto hsum_avx2 = [](__m256 v) -> float {
                __m128 hi = _mm256_extractf128_ps(v, 1);
                __m128 lo = _mm256_castps256_ps128(v);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                return _mm_cvtss_f32(sum128);
            };

            C[m * N + n + 0] = hsum_avx2(acc0);
            C[m * N + n + 1] = hsum_avx2(acc1);
            C[m * N + n + 2] = hsum_avx2(acc2);
            C[m * N + n + 3] = hsum_avx2(acc3);
            C[m * N + n + 4] = hsum_avx2(acc4);
            C[m * N + n + 5] = hsum_avx2(acc5);
            C[m * N + n + 6] = hsum_avx2(acc6);
            C[m * N + n + 7] = hsum_avx2(acc7);
        }

        // Handle remaining columns (N % 8)
        for (; n < N; n++) {
            __m256 acc = _mm256_setzero_ps();

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];
                const __m256 vscale = _mm256_set1_ps(scale);
                const __m256 vzero = _mm256_set1_ps(zero);
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 16) {
                    // Load 8 bytes = 16 packed weights
                    __m128i packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr + k / 2));
                    __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);

                    // Extract and sign-extend nibbles
                    __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));
                    __m128i high = _mm_srli_epi16(packed_16, 4);
                    high = _mm_and_si128(high, _mm_set1_epi16(0x0F));

                    low = _mm_slli_epi16(low, 12);
                    low = _mm_srai_epi16(low, 12);
                    high = _mm_slli_epi16(high, 12);
                    high = _mm_srai_epi16(high, 12);

                    // Interleave
                    __m128i lo128 = _mm_unpacklo_epi16(low, high);
                    __m128i hi128 = _mm_unpackhi_epi16(low, high);

                    // Convert and dequantize
                    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo128));
                    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi128));

                    wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));
                    wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));

                    // Load activations and FMA
                    __m256 a0 = _mm256_loadu_ps(a_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

                    acc = _mm256_fmadd_ps(a0, wf0, acc);
                    acc = _mm256_fmadd_ps(a1, wf1, acc);
                }
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            C[m * N + n] = _mm_cvtss_f32(sum128);
        }
    }
}

/**
 * @brief Batched INT4 GEMM with weight reuse across M tokens (AVX2)
 *
 * Loop order: N-outer → M-tile → K-inner
 * Maximizes weight reuse: each weight tile is dequantized once and applied
 * to TILE_M activation rows simultaneously.
 *
 * Arithmetic intensity improvement over per-row GEMV:
 * - Per-row GEMV: Weight loaded M times (once per token)
 * - Batched GEMM: Weight loaded once, reused across min(M, TILE_M) tokens
 * - For M=4, TILE_M=4: 4× reduction in weight memory traffic
 *
 * Register allocation (TILE_M=4, TILE_N=2):
 * - 8 accumulators [4M × 2N]
 * - 2 weight regs (dequantized, per column)
 * - 2 activation regs (per row)
 * - ~2 temp regs for dequant intermediates
 * Total: 14 of 16 YMM registers
 *
 * @param n_start First output column (inclusive) for thread partitioning
 * @param n_end   Last output column (exclusive), -1 means N
 */
inline void GemmInt4Fp32Batched_AVX2(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                     const float* zero_points, int M, int N, int K, int group_size, int n_start = 0,
                                     int n_end_param = -1) {
    const int n_end = (n_end_param < 0) ? N : n_end_param;
    if (K % group_size != 0 || n_start >= n_end || M <= 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;

    constexpr int TILE_M = 4;
    constexpr int TILE_N = 2;

    // Horizontal sum helper
    auto hsum_avx2 = [](__m256 v) -> float {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    };

    // INT4 dequantize helper: 8 packed bytes → 2 × __m256 (16 FP32 values)
    auto dequant_int4 = [](const uint8_t* w_ptr, __m256 vscale, __m256 vzero, __m256& out0, __m256& out1) {
        __m128i packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr));
        __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);

        __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));
        __m128i high = _mm_srli_epi16(packed_16, 4);
        high = _mm_and_si128(high, _mm_set1_epi16(0x0F));

        low = _mm_slli_epi16(low, 12);
        low = _mm_srai_epi16(low, 12);
        high = _mm_slli_epi16(high, 12);
        high = _mm_srai_epi16(high, 12);

        __m128i lo128 = _mm_unpacklo_epi16(low, high);
        __m128i hi128 = _mm_unpackhi_epi16(low, high);

        __m256 raw0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo128));
        __m256 raw1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi128));

        out0 = _mm256_mul_ps(vscale, _mm256_sub_ps(raw0, vzero));
        out1 = _mm256_mul_ps(vscale, _mm256_sub_ps(raw1, vzero));
    };

    // =========================================================================
    // N-OUTER LOOP: Weight columns in tiles of TILE_N
    // =========================================================================
    int n = n_start;
    for (; n + TILE_N <= n_end; n += TILE_N) {
        // M-tile loop
        for (int m = 0; m < M; m += TILE_M) {
            const int actual_m = std::min(TILE_M, M - m);

            // Accumulators: [TILE_M][TILE_N] - compiler keeps in YMM regs
            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

            // K loop (groups × group_size)
            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_g_offset = g * (group_size / 2);

                // Preload scale/zero for both weight columns
                const __m256 vs0 = _mm256_set1_ps(scales[(n + 0) * num_groups + g]);
                const __m256 vz0 = _mm256_set1_ps(zero_points[(n + 0) * num_groups + g]);
                const __m256 vs1 = _mm256_set1_ps(scales[(n + 1) * num_groups + g]);
                const __m256 vz1 = _mm256_set1_ps(zero_points[(n + 1) * num_groups + g]);

                for (int k = 0; k < group_size; k += 16) {
                    // === Dequantize 2 weight columns: done ONCE per k-step ===
                    __m256 wf0_0, wf1_0, wf0_1, wf1_1;
                    const uint8_t* w0_ptr = W_int4 + (n + 0) * packed_K + packed_g_offset + k / 2;
                    const uint8_t* w1_ptr = W_int4 + (n + 1) * packed_K + packed_g_offset + k / 2;

                    // Prefetch next k-step weights
                    _mm_prefetch(reinterpret_cast<const char*>(w0_ptr + group_size / 2), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w1_ptr + group_size / 2), _MM_HINT_T0);

                    dequant_int4(w0_ptr, vs0, vz0, wf0_0, wf1_0);
                    dequant_int4(w1_ptr, vs1, vz1, wf0_1, wf1_1);

                    // === Apply to TILE_M activation rows (WEIGHT REUSE!) ===
                    // Activation rows cycle through L1 cache; weights stay in regs
                    if (actual_m >= 1) {
                        const float* a0 = A + (m + 0) * K + k_offset + k;
                        __m256 av0 = _mm256_loadu_ps(a0);
                        __m256 av1 = _mm256_loadu_ps(a0 + 8);
                        acc00 = _mm256_fmadd_ps(av0, wf0_0, acc00);
                        acc00 = _mm256_fmadd_ps(av1, wf1_0, acc00);
                        acc01 = _mm256_fmadd_ps(av0, wf0_1, acc01);
                        acc01 = _mm256_fmadd_ps(av1, wf1_1, acc01);
                    }
                    if (actual_m >= 2) {
                        const float* a1 = A + (m + 1) * K + k_offset + k;
                        __m256 av0 = _mm256_loadu_ps(a1);
                        __m256 av1 = _mm256_loadu_ps(a1 + 8);
                        acc10 = _mm256_fmadd_ps(av0, wf0_0, acc10);
                        acc10 = _mm256_fmadd_ps(av1, wf1_0, acc10);
                        acc11 = _mm256_fmadd_ps(av0, wf0_1, acc11);
                        acc11 = _mm256_fmadd_ps(av1, wf1_1, acc11);
                    }
                    if (actual_m >= 3) {
                        const float* a2 = A + (m + 2) * K + k_offset + k;
                        __m256 av0 = _mm256_loadu_ps(a2);
                        __m256 av1 = _mm256_loadu_ps(a2 + 8);
                        acc20 = _mm256_fmadd_ps(av0, wf0_0, acc20);
                        acc20 = _mm256_fmadd_ps(av1, wf1_0, acc20);
                        acc21 = _mm256_fmadd_ps(av0, wf0_1, acc21);
                        acc21 = _mm256_fmadd_ps(av1, wf1_1, acc21);
                    }
                    if (actual_m >= 4) {
                        const float* a3 = A + (m + 3) * K + k_offset + k;
                        __m256 av0 = _mm256_loadu_ps(a3);
                        __m256 av1 = _mm256_loadu_ps(a3 + 8);
                        acc30 = _mm256_fmadd_ps(av0, wf0_0, acc30);
                        acc30 = _mm256_fmadd_ps(av1, wf1_0, acc30);
                        acc31 = _mm256_fmadd_ps(av0, wf0_1, acc31);
                        acc31 = _mm256_fmadd_ps(av1, wf1_1, acc31);
                    }
                }
            }

            // Store results
            if (actual_m >= 1) {
                C[(m + 0) * N + n] = hsum_avx2(acc00);
                C[(m + 0) * N + n + 1] = hsum_avx2(acc01);
            }
            if (actual_m >= 2) {
                C[(m + 1) * N + n] = hsum_avx2(acc10);
                C[(m + 1) * N + n + 1] = hsum_avx2(acc11);
            }
            if (actual_m >= 3) {
                C[(m + 2) * N + n] = hsum_avx2(acc20);
                C[(m + 2) * N + n + 1] = hsum_avx2(acc21);
            }
            if (actual_m >= 4) {
                C[(m + 3) * N + n] = hsum_avx2(acc30);
                C[(m + 3) * N + n + 1] = hsum_avx2(acc31);
            }
        }
    }

    // Handle N tail (single column)
    for (; n < n_end; n++) {
        for (int m = 0; m < M; m += TILE_M) {
            const int actual_m = std::min(TILE_M, M - m);

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const __m256 vs = _mm256_set1_ps(scales[n * num_groups + g]);
                const __m256 vz = _mm256_set1_ps(zero_points[n * num_groups + g]);
                const uint8_t* w_base = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 16) {
                    __m256 wf0, wf1;
                    dequant_int4(w_base + k / 2, vs, vz, wf0, wf1);

                    for (int mi = 0; mi < actual_m; mi++) {
                        const float* a_ptr = A + (m + mi) * K + k_offset + k;
                        __m256 av0 = _mm256_loadu_ps(a_ptr);
                        __m256 av1 = _mm256_loadu_ps(a_ptr + 8);
                        __m256& acc_ref = (mi == 0) ? acc0 : (mi == 1) ? acc1 : (mi == 2) ? acc2 : acc3;
                        acc_ref = _mm256_fmadd_ps(av0, wf0, acc_ref);
                        acc_ref = _mm256_fmadd_ps(av1, wf1, acc_ref);
                    }
                }
            }

            if (actual_m >= 1) C[(m + 0) * N + n] = hsum_avx2(acc0);
            if (actual_m >= 2) C[(m + 1) * N + n] = hsum_avx2(acc1);
            if (actual_m >= 3) C[(m + 2) * N + n] = hsum_avx2(acc2);
            if (actual_m >= 4) C[(m + 3) * N + n] = hsum_avx2(acc3);
        }
    }
}

#endif  // __AVX2__ && DENSECORE_HAS_FMA

// =============================================================================
// INT4 Quantized GEMM (ARM SVE) - Scalable Vector Extension
// =============================================================================
// Targets: Google Cloud C4A (Axion/Neoverse V2, SVE2 128-bit scalable),
//          AWS Graviton 3 (Neoverse V1, SVE 256-bit),
//          AWS Graviton 4 (Neoverse V2, SVE2)
//
// SVE vectors are scalable — svcntw() returns the number of 32-bit lanes
// at runtime (4 for 128-bit, 8 for 256-bit, 16 for 512-bit).
// All loops use svwhilelt predicates for clean tail handling without
// separate remainder code.
// =============================================================================

#if defined(__ARM_FEATURE_SVE)

/**
 * @brief High-performance SVE GEMM kernel: C = A * W^T (INT4 weights)
 *
 * Optimizations:
 * - Scalable vector width (adapts to 128/256/512-bit SVE automatically)
 * - 4x register blocking along N dimension
 * - Predicated operations for clean tail handling (no remainder loops)
 * - Shift-based INT4 sign extension (branchless)
 * - Software prefetch for weights and activations
 * - FMA via svmla_f32_x for peak throughput
 *
 * On Neoverse V2 (C4A, 128-bit SVE2): processes 4 floats per vector
 * On Neoverse V1 (Graviton 3, 256-bit SVE): processes 8 floats per vector
 *
 * @param C Output [M, N]
 * @param A Input activations [M, K]
 * @param W_int4 Packed INT4 weights [N, K/2]
 * @param scales Per-group scales [N, num_groups]
 * @param zero_points Per-group zero points [N, num_groups]
 * @param M Batch dimension
 * @param N Output features
 * @param K Input features
 * @param group_size Quantization group size (K must be divisible)
 */
inline void GemmInt4Fp32_SVE(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                             const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;
    const uint64_t vl = svcntw();  // Number of 32-bit lanes per SVE vector

    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

        // N-blocking: process 4 output columns at a time
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            svfloat32_t acc0 = svdup_f32(0.0f);
            svfloat32_t acc1 = svdup_f32(0.0f);
            svfloat32_t acc2 = svdup_f32(0.0f);
            svfloat32_t acc3 = svdup_f32(0.0f);

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_g_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;

                // Broadcast scale/zero for each of the 4 weight rows
                const svfloat32_t vs0 = svdup_f32(scales[(n + 0) * num_groups + g]);
                const svfloat32_t vz0 = svdup_f32(zero_points[(n + 0) * num_groups + g]);
                const svfloat32_t vs1 = svdup_f32(scales[(n + 1) * num_groups + g]);
                const svfloat32_t vz1 = svdup_f32(zero_points[(n + 1) * num_groups + g]);
                const svfloat32_t vs2 = svdup_f32(scales[(n + 2) * num_groups + g]);
                const svfloat32_t vz2 = svdup_f32(zero_points[(n + 2) * num_groups + g]);
                const svfloat32_t vs3 = svdup_f32(scales[(n + 3) * num_groups + g]);
                const svfloat32_t vz3 = svdup_f32(zero_points[(n + 3) * num_groups + g]);

                // Prefetch next group's activations
                if (g + 1 < num_groups) {
                    __builtin_prefetch(a_row + (g + 1) * group_size, 0, 3);
                }

                // Process K dimension in steps of 2*vl elements
                // Each packed byte holds 2 INT4 values → process 2*vl weights
                // from vl packed bytes per iteration
                for (int k = 0; k < group_size; k += static_cast<int>(2 * vl)) {
                    const int remaining = group_size - k;
                    const int step = (remaining < static_cast<int>(2 * vl)) ? remaining : static_cast<int>(2 * vl);
                    const uint64_t packed_step = static_cast<uint64_t>((step + 1) / 2);

                    // Load activations with predication for tail
                    svbool_t pg0 = svwhilelt_b32_u64(
                        0UL, static_cast<uint64_t>(
                                 step > 0 ? (step < static_cast<int>(vl) ? step : static_cast<int>(vl)) : 0));
                    svbool_t pg1 = (step > static_cast<int>(vl))
                                       ? svwhilelt_b32_u64(0UL, static_cast<uint64_t>(step - static_cast<int>(vl)))
                                       : svpfalse_b();

                    svfloat32_t a0 = svld1_f32(pg0, a_ptr + k);
                    svfloat32_t a1 = svld1_f32(pg1, a_ptr + k + vl);

// Macro to process one weight row for SVE
#define SVE_PROCESS_ROW(idx, acc_var)                                                                          \
    do {                                                                                                       \
        const uint8_t* w_ptr = W_int4 + (n + (idx)) * packed_K + packed_g_offset + k / 2;                      \
        __builtin_prefetch(w_ptr + 64, 0, 3);                                                                  \
                                                                                                               \
        /* Load only the packed bytes needed for this step */                                                  \
        svuint8_t packed_bytes = svld1_u8(svwhilelt_b8_u64(0UL, packed_step), w_ptr);                          \
                                                                                                               \
        /* Extract low nibbles (even indices) */                                                               \
        svuint8_t lo_u8 = svand_u8_x(svptrue_b8(), packed_bytes, svdup_u8(0x0F));                              \
        /* Extract high nibbles (odd indices) */                                                               \
        svuint8_t hi_u8 = svlsr_n_u8_x(svptrue_b8(), packed_bytes, 4);                                         \
                                                                                                               \
        /* Zero-extend u8→u32 */                                                                             \
        svuint32_t lo_u32 = svunpklo_u32(svunpklo_u16(lo_u8));                                                 \
        svuint32_t hi_u32 = svunpklo_u32(svunpklo_u16(hi_u8));                                                 \
                                                                                                               \
        /* Sign extend 4-bit: shift left 28, arithmetic shift right 28 */                                      \
        svint32_t lo_s32 =                                                                                     \
            svasr_n_s32_x(svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(lo_u32), 28), 28); \
        svint32_t hi_s32 =                                                                                     \
            svasr_n_s32_x(svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(hi_u32), 28), 28); \
                                                                                                               \
        /* Interleave low and high nibbles to restore original order */                                        \
        /* Element order: [lo0, hi0, lo1, hi1, ...] */                                                         \
        svint32_t w_even = svzip1_s32(lo_s32, hi_s32);                                                         \
        svint32_t w_odd = svzip2_s32(lo_s32, hi_s32);                                                          \
                                                                                                               \
        /* Convert to FP32 and dequantize */                                                                   \
        svfloat32_t wf0 = svmul_f32_x(pg0, vs##idx, svsub_f32_x(pg0, svcvt_f32_s32_x(pg0, w_even), vz##idx));  \
        svfloat32_t wf1 = svmul_f32_x(pg1, vs##idx, svsub_f32_x(pg1, svcvt_f32_s32_x(pg1, w_odd), vz##idx));   \
                                                                                                               \
        /* FMA: acc += a * w */                                                                                \
        acc_var = svmla_f32_m(pg0, acc_var, a0, wf0);                                                          \
        acc_var = svmla_f32_m(pg1, acc_var, a1, wf1);                                                          \
    } while (0)

                    SVE_PROCESS_ROW(0, acc0);
                    SVE_PROCESS_ROW(1, acc1);
                    SVE_PROCESS_ROW(2, acc2);
                    SVE_PROCESS_ROW(3, acc3);

#undef SVE_PROCESS_ROW
                }
            }

            // Horizontal reduction using SVE addv
            C[m * N + n + 0] = svaddv_f32(svptrue_b32(), acc0);
            C[m * N + n + 1] = svaddv_f32(svptrue_b32(), acc1);
            C[m * N + n + 2] = svaddv_f32(svptrue_b32(), acc2);
            C[m * N + n + 3] = svaddv_f32(svptrue_b32(), acc3);
        }

        // Handle remaining columns (N % 4)
        for (; n < N; n++) {
            svfloat32_t acc = svdup_f32(0.0f);

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_g_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;
                const svfloat32_t vs = svdup_f32(scales[n * num_groups + g]);
                const svfloat32_t vz = svdup_f32(zero_points[n * num_groups + g]);

                for (int k = 0; k < group_size; k += static_cast<int>(2 * vl)) {
                    const int remaining = group_size - k;
                    const int step = (remaining < static_cast<int>(2 * vl)) ? remaining : static_cast<int>(2 * vl);
                    const uint64_t packed_step = static_cast<uint64_t>((step + 1) / 2);

                    svbool_t pg0 = svwhilelt_b32_u64(
                        0UL, static_cast<uint64_t>(
                                 step > 0 ? (step < static_cast<int>(vl) ? step : static_cast<int>(vl)) : 0));
                    svbool_t pg1 = (step > static_cast<int>(vl))
                                       ? svwhilelt_b32_u64(0UL, static_cast<uint64_t>(step - static_cast<int>(vl)))
                                       : svpfalse_b();

                    svfloat32_t a0 = svld1_f32(pg0, a_ptr + k);
                    svfloat32_t a1 = svld1_f32(pg1, a_ptr + k + vl);

                    const uint8_t* w_ptr = W_int4 + n * packed_K + packed_g_offset + k / 2;

                    svuint8_t packed_bytes = svld1_u8(svwhilelt_b8_u64(0UL, packed_step), w_ptr);
                    svuint8_t lo_u8 = svand_u8_x(svptrue_b8(), packed_bytes, svdup_u8(0x0F));
                    svuint8_t hi_u8 = svlsr_n_u8_x(svptrue_b8(), packed_bytes, 4);

                    svuint32_t lo_u32 = svunpklo_u32(svunpklo_u16(lo_u8));
                    svuint32_t hi_u32 = svunpklo_u32(svunpklo_u16(hi_u8));

                    svint32_t lo_s32 = svasr_n_s32_x(
                        svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(lo_u32), 28), 28);
                    svint32_t hi_s32 = svasr_n_s32_x(
                        svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(hi_u32), 28), 28);

                    svint32_t w_even = svzip1_s32(lo_s32, hi_s32);
                    svint32_t w_odd = svzip2_s32(lo_s32, hi_s32);

                    svfloat32_t wf0 = svmul_f32_x(pg0, vs, svsub_f32_x(pg0, svcvt_f32_s32_x(pg0, w_even), vz));
                    svfloat32_t wf1 = svmul_f32_x(pg1, vs, svsub_f32_x(pg1, svcvt_f32_s32_x(pg1, w_odd), vz));

                    acc = svmla_f32_m(pg0, acc, a0, wf0);
                    acc = svmla_f32_m(pg1, acc, a1, wf1);
                }
            }

            C[m * N + n] = svaddv_f32(svptrue_b32(), acc);
        }
    }
}

/**
 * @brief Batched INT4 GEMM with weight reuse across M tokens (SVE)
 *
 * Loop order: N-outer → M-tile → K-inner
 * Same weight-reuse strategy as AVX2 batched variant, adapted for SVE.
 * Dequantizes each weight tile ONCE and applies to TILE_M=4 rows.
 *
 * @param n_start First output column (inclusive)
 * @param n_end_param Last output column (exclusive), -1 means N
 */
inline void GemmInt4Fp32Batched_SVE(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                    const float* zero_points, int M, int N, int K, int group_size, int n_start = 0,
                                    int n_end_param = -1) {
    const int n_end = (n_end_param < 0) ? N : n_end_param;
    if (K % group_size != 0 || n_start >= n_end || M <= 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;
    const uint64_t vl = svcntw();
    constexpr int TILE_M = 4;

    for (int n = n_start; n < n_end; n++) {
        for (int m = 0; m < M; m += TILE_M) {
            const int actual_m = std::min(TILE_M, M - m);

            svfloat32_t acc0 = svdup_f32(0.0f);
            svfloat32_t acc1 = svdup_f32(0.0f);
            svfloat32_t acc2 = svdup_f32(0.0f);
            svfloat32_t acc3 = svdup_f32(0.0f);

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_g_offset = g * (group_size / 2);
                const svfloat32_t vs = svdup_f32(scales[n * num_groups + g]);
                const svfloat32_t vz = svdup_f32(zero_points[n * num_groups + g]);

                for (int k = 0; k < group_size; k += static_cast<int>(2 * vl)) {
                    const int remaining = group_size - k;
                    const int step = (remaining < static_cast<int>(2 * vl)) ? remaining : static_cast<int>(2 * vl);
                    const uint64_t packed_step = static_cast<uint64_t>((step + 1) / 2);

                    svbool_t pg0 = svwhilelt_b32_u64(
                        0UL, static_cast<uint64_t>(
                                 step > 0 ? (step < static_cast<int>(vl) ? step : static_cast<int>(vl)) : 0));
                    svbool_t pg1 = (step > static_cast<int>(vl))
                                       ? svwhilelt_b32_u64(0UL, static_cast<uint64_t>(step - static_cast<int>(vl)))
                                       : svpfalse_b();

                    // Dequantize weight ONCE
                    const uint8_t* w_ptr = W_int4 + n * packed_K + packed_g_offset + k / 2;
                    __builtin_prefetch(w_ptr + 64, 0, 3);

                    svuint8_t packed_bytes = svld1_u8(svwhilelt_b8_u64(0UL, packed_step), w_ptr);
                    svuint8_t lo_u8 = svand_u8_x(svptrue_b8(), packed_bytes, svdup_u8(0x0F));
                    svuint8_t hi_u8 = svlsr_n_u8_x(svptrue_b8(), packed_bytes, 4);

                    svuint32_t lo_u32 = svunpklo_u32(svunpklo_u16(lo_u8));
                    svuint32_t hi_u32 = svunpklo_u32(svunpklo_u16(hi_u8));

                    svint32_t lo_s32 = svasr_n_s32_x(
                        svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(lo_u32), 28), 28);
                    svint32_t hi_s32 = svasr_n_s32_x(
                        svptrue_b32(), svlsl_n_s32_x(svptrue_b32(), svreinterpret_s32_u32(hi_u32), 28), 28);

                    svint32_t w_even = svzip1_s32(lo_s32, hi_s32);
                    svint32_t w_odd = svzip2_s32(lo_s32, hi_s32);

                    svfloat32_t wf0 = svmul_f32_x(pg0, vs, svsub_f32_x(pg0, svcvt_f32_s32_x(pg0, w_even), vz));
                    svfloat32_t wf1 = svmul_f32_x(pg1, vs, svsub_f32_x(pg1, svcvt_f32_s32_x(pg1, w_odd), vz));

                    // Apply to TILE_M rows (weight reuse!)
                    if (actual_m >= 1) {
                        svfloat32_t av0 = svld1_f32(pg0, A + (m + 0) * K + k_offset + k);
                        svfloat32_t av1 = svld1_f32(pg1, A + (m + 0) * K + k_offset + k + vl);
                        acc0 = svmla_f32_m(pg0, acc0, av0, wf0);
                        acc0 = svmla_f32_m(pg1, acc0, av1, wf1);
                    }
                    if (actual_m >= 2) {
                        svfloat32_t av0 = svld1_f32(pg0, A + (m + 1) * K + k_offset + k);
                        svfloat32_t av1 = svld1_f32(pg1, A + (m + 1) * K + k_offset + k + vl);
                        acc1 = svmla_f32_m(pg0, acc1, av0, wf0);
                        acc1 = svmla_f32_m(pg1, acc1, av1, wf1);
                    }
                    if (actual_m >= 3) {
                        svfloat32_t av0 = svld1_f32(pg0, A + (m + 2) * K + k_offset + k);
                        svfloat32_t av1 = svld1_f32(pg1, A + (m + 2) * K + k_offset + k + vl);
                        acc2 = svmla_f32_m(pg0, acc2, av0, wf0);
                        acc2 = svmla_f32_m(pg1, acc2, av1, wf1);
                    }
                    if (actual_m >= 4) {
                        svfloat32_t av0 = svld1_f32(pg0, A + (m + 3) * K + k_offset + k);
                        svfloat32_t av1 = svld1_f32(pg1, A + (m + 3) * K + k_offset + k + vl);
                        acc3 = svmla_f32_m(pg0, acc3, av0, wf0);
                        acc3 = svmla_f32_m(pg1, acc3, av1, wf1);
                    }
                }
            }

            if (actual_m >= 1) C[(m + 0) * N + n] = svaddv_f32(svptrue_b32(), acc0);
            if (actual_m >= 2) C[(m + 1) * N + n] = svaddv_f32(svptrue_b32(), acc1);
            if (actual_m >= 3) C[(m + 2) * N + n] = svaddv_f32(svptrue_b32(), acc2);
            if (actual_m >= 4) C[(m + 3) * N + n] = svaddv_f32(svptrue_b32(), acc3);
        }
    }
}

#endif  // __ARM_FEATURE_SVE

// =============================================================================
// INT4 Quantized GEMM (ARM NEON) - Fixed 128-bit Vectors
// =============================================================================
// Targets: Apple Silicon, Graviton2, Raspberry Pi 4, any AArch64 without SVE
// NEON: 128-bit fixed width, processes 4 floats per vector
// =============================================================================

#if defined(DENSECORE_ARM)

#if defined(__ARM_FEATURE_DOTPROD)
inline bool UseNeonInt4DotProdApprox() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_ARM_INT4_DOTPROD");
        return env && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
    }();
    return enabled;
}

inline int8x16_t UnpackInt4x16ToInt8_NEON_DotProd(const uint8_t* w_ptr) {
    const uint8x8_t packed = vld1_u8(w_ptr);
    const uint8x8_t lo = vand_u8(packed, vdup_n_u8(0x0F));
    const uint8x8_t hi = vshr_n_u8(packed, 4);
    const uint8x8_t interleaved_lo = vzip1_u8(lo, hi);
    const uint8x8_t interleaved_hi = vzip2_u8(lo, hi);
    const uint8x16_t qu = vcombine_u8(interleaved_lo, interleaved_hi);
    return vreinterpretq_s8_u8(vsubq_u8(qu, vdupq_n_u8(8)));
}

inline int8x16_t QuantizeF32x16ToI8_NEON_DotProd(const float* a, float* out_scale, float* out_sum) {
    const float32x4_t a0 = vld1q_f32(a + 0);
    const float32x4_t a1 = vld1q_f32(a + 4);
    const float32x4_t a2 = vld1q_f32(a + 8);
    const float32x4_t a3 = vld1q_f32(a + 12);

    const float sum_a = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3);
    if (out_sum) {
        *out_sum = sum_a;
    }

    const float32x4_t abs0 = vabsq_f32(a0);
    const float32x4_t abs1 = vabsq_f32(a1);
    const float32x4_t abs2 = vabsq_f32(a2);
    const float32x4_t abs3 = vabsq_f32(a3);
    const float32x4_t max01 = vmaxq_f32(abs0, abs1);
    const float32x4_t max23 = vmaxq_f32(abs2, abs3);
    const float32x4_t maxv = vmaxq_f32(max01, max23);
    const float max_abs = std::max(std::max(vgetq_lane_f32(maxv, 0), vgetq_lane_f32(maxv, 1)),
                                   std::max(vgetq_lane_f32(maxv, 2), vgetq_lane_f32(maxv, 3)));

    if (max_abs < 1e-12f) {
        if (out_scale) {
            *out_scale = 0.0f;
        }
        return vdupq_n_s8(0);
    }

    const float a_scale = max_abs / 127.0f;
    if (out_scale) {
        *out_scale = a_scale;
    }
    const float32x4_t inv_scale = vdupq_n_f32(127.0f / max_abs);

    const int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(a0, inv_scale));
    const int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(a1, inv_scale));
    const int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(a2, inv_scale));
    const int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(a3, inv_scale));

    const int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    const int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    return vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
}

inline float DotProdApproxF32Q4_16_NEON(const float* a, const uint8_t* w_ptr, float scale, float zero) {
    float a_scale = 0.0f;
    float sum_a = 0.0f;
    const int8x16_t qa = QuantizeF32x16ToI8_NEON_DotProd(a, &a_scale, &sum_a);
    if (a_scale == 0.0f) {
        return 0.0f;
    }

    const int8x16_t qw = UnpackInt4x16ToInt8_NEON_DotProd(w_ptr);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, qa, qw);
    const float dot_q = static_cast<float>(vaddvq_s32(acc));
    return scale * (a_scale * dot_q - zero * sum_a);
}

inline void GemmInt4Fp32_NEON_DotProdApprox(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                            const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;

    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;

        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            for (int g = 0; g < num_groups; ++g) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float z0 = zero_points ? zero_points[(n + 0) * num_groups + g] : 0.0f;
                const float z1 = zero_points ? zero_points[(n + 1) * num_groups + g] : 0.0f;
                const float z2 = zero_points ? zero_points[(n + 2) * num_groups + g] : 0.0f;
                const float z3 = zero_points ? zero_points[(n + 3) * num_groups + g] : 0.0f;
                const float s0 = scales[(n + 0) * num_groups + g];
                const float s1 = scales[(n + 1) * num_groups + g];
                const float s2 = scales[(n + 2) * num_groups + g];
                const float s3 = scales[(n + 3) * num_groups + g];
                const uint8_t* w0 = W_int4 + (n + 0) * packed_K + g * (group_size / 2);
                const uint8_t* w1 = W_int4 + (n + 1) * packed_K + g * (group_size / 2);
                const uint8_t* w2 = W_int4 + (n + 2) * packed_K + g * (group_size / 2);
                const uint8_t* w3 = W_int4 + (n + 3) * packed_K + g * (group_size / 2);

                int k = 0;
                for (; k + 16 <= group_size; k += 16) {
                    acc0 += DotProdApproxF32Q4_16_NEON(a_ptr + k, w0 + k / 2, s0, z0);
                    acc1 += DotProdApproxF32Q4_16_NEON(a_ptr + k, w1 + k / 2, s1, z1);
                    acc2 += DotProdApproxF32Q4_16_NEON(a_ptr + k, w2 + k / 2, s2, z2);
                    acc3 += DotProdApproxF32Q4_16_NEON(a_ptr + k, w3 + k / 2, s3, z3);
                }
                for (; k < group_size; ++k) {
                    const float a_val = a_ptr[k];
                    const int packed_idx = k / 2;
                    const int nibble_shift = (k & 1) ? 4 : 0;
                    int q0 = (w0[packed_idx] >> nibble_shift) & 0x0F;
                    int q1 = (w1[packed_idx] >> nibble_shift) & 0x0F;
                    int q2 = (w2[packed_idx] >> nibble_shift) & 0x0F;
                    int q3 = (w3[packed_idx] >> nibble_shift) & 0x0F;
                    if (q0 > 7) q0 -= 16;
                    if (q1 > 7) q1 -= 16;
                    if (q2 > 7) q2 -= 16;
                    if (q3 > 7) q3 -= 16;
                    acc0 += s0 * (static_cast<float>(q0) - z0) * a_val;
                    acc1 += s1 * (static_cast<float>(q1) - z1) * a_val;
                    acc2 += s2 * (static_cast<float>(q2) - z2) * a_val;
                    acc3 += s3 * (static_cast<float>(q3) - z3) * a_val;
                }
            }

            C[m * N + n + 0] = acc0;
            C[m * N + n + 1] = acc1;
            C[m * N + n + 2] = acc2;
            C[m * N + n + 3] = acc3;
        }

        for (; n < N; ++n) {
            float acc = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points ? zero_points[n * num_groups + g] : 0.0f;
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                int k = 0;
                for (; k + 16 <= group_size; k += 16) {
                    acc += DotProdApproxF32Q4_16_NEON(a_ptr + k, w_ptr + k / 2, scale, zero);
                }
                for (; k < group_size; ++k) {
                    const float a_val = a_ptr[k];
                    int q = (w_ptr[k / 2] >> ((k & 1) ? 4 : 0)) & 0x0F;
                    if (q > 7) q -= 16;
                    acc += scale * (static_cast<float>(q) - zero) * a_val;
                }
            }
            C[m * N + n] = acc;
        }
    }
}
#endif

/**
 * @brief High-performance NEON GEMM kernel: C = A * W^T (INT4 weights)
 *
 * Optimizations:
 * - 4x register blocking along N dimension
 * - Shift-based INT4 sign extension (branchless)
 * - FMA via vfmaq_f32 (available on all AArch64)
 * - Software prefetch for weights
 * - Processes 8 weights per iteration (4 packed bytes → 8 INT4 → 8 FP32)
 *
 * Performance: ~4x faster than scalar fallback on NEON
 */
inline void GemmInt4Fp32_NEON(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                              const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0) return;

#if defined(__ARM_FEATURE_DOTPROD)
    if (UseNeonInt4DotProdApprox()) {
        GemmInt4Fp32_NEON_DotProdApprox(C, A, W_int4, scales, zero_points, M, N, K, group_size);
        return;
    }
#endif

    const int num_groups = K / group_size;
    const int packed_K = K / 2;

    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

        // N-blocking: process 4 output columns at a time
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            // Second set of accumulators for 2-way unrolling
            float32x4_t acc0b = vdupq_n_f32(0.0f);
            float32x4_t acc1b = vdupq_n_f32(0.0f);
            float32x4_t acc2b = vdupq_n_f32(0.0f);
            float32x4_t acc3b = vdupq_n_f32(0.0f);

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_g_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;

                // Prefetch next group
                if (g + 1 < num_groups) {
                    __builtin_prefetch(a_row + (g + 1) * group_size, 0, 3);
                }

                // Process 8 weights per iteration (4 packed bytes)
                for (int k = 0; k < group_size; k += 8) {
                    // Load 2 × 4 floats from activations
                    float32x4_t a0 = vld1q_f32(a_ptr + k);
                    float32x4_t a1 = vld1q_f32(a_ptr + k + 4);

// Macro to process one weight row for NEON
#define NEON_PROCESS_ROW(idx, acc_a, acc_b)                                                  \
    do {                                                                                     \
        const int row = n + (idx);                                                           \
        const float scale = scales[row * num_groups + g];                                    \
        const float zero = zero_points[row * num_groups + g];                                \
        const float32x4_t vscale = vdupq_n_f32(scale);                                       \
        const float32x4_t vzero = vdupq_n_f32(zero);                                         \
                                                                                             \
        const uint8_t* w_ptr = W_int4 + row * packed_K + packed_g_offset + k / 2;            \
        __builtin_prefetch(w_ptr + 32, 0, 3);                                                \
                                                                                             \
        /* Load exactly 4 bytes = 8 packed INT4 weights.                                   \
         * vld1_u8() would read 8 bytes and can overrun the final group tail on ARM. */ \
        uint8_t packed_tmp[8] = {};                                                          \
        std::memcpy(packed_tmp, w_ptr, sizeof(uint32_t));                                    \
        uint8x8_t packed_u8 = vld1_u8(packed_tmp);                                           \
        /* Widen u8 → u16 */                                                               \
        uint16x8_t packed_u16 = vmovl_u8(packed_u8);                                         \
        /* Extract low nibbles (even positions) */                                           \
        uint16x8_t lo_u16 = vandq_u16(packed_u16, vdupq_n_u16(0x0F));                        \
        /* Extract high nibbles (odd positions) */                                           \
        uint16x8_t hi_u16 = vshrq_n_u16(packed_u16, 4);                                      \
        hi_u16 = vandq_u16(hi_u16, vdupq_n_u16(0x0F));                                       \
                                                                                             \
        /* Sign extension via shift trick (16-bit): shl 12, asr 12 */                        \
        int16x8_t lo_s16 = vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_u16(lo_u16), 12), 12);  \
        int16x8_t hi_s16 = vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_u16(hi_u16), 12), 12);  \
                                                                                             \
        /* Interleave low and high: [lo0,hi0,lo1,hi1,...] for the 4 loaded bytes. */         \
        int16x8_t interleaved_lo = vzip1q_s16(lo_s16, hi_s16);                               \
                                                                                             \
        /* Convert first 4 elements to FP32 */                                               \
        int32x4_t w32_0 = vmovl_s16(vget_low_s16(interleaved_lo));                           \
        float32x4_t wf0 = vcvtq_f32_s32(w32_0);                                              \
        /* Convert next 4 elements to FP32 */                                                \
        int32x4_t w32_1 = vmovl_s16(vget_high_s16(interleaved_lo));                          \
        float32x4_t wf1 = vcvtq_f32_s32(w32_1);                                              \
                                                                                             \
        /* Dequantize: w_dequant = scale * (q - zero) */                                     \
        wf0 = vmulq_f32(vscale, vsubq_f32(wf0, vzero));                                      \
        wf1 = vmulq_f32(vscale, vsubq_f32(wf1, vzero));                                      \
                                                                                             \
        /* FMA: acc += a * w */                                                              \
        acc_a = vfmaq_f32(acc_a, a0, wf0);                                                   \
        acc_b = vfmaq_f32(acc_b, a1, wf1);                                                   \
    } while (0)

                    NEON_PROCESS_ROW(0, acc0, acc0b);
                    NEON_PROCESS_ROW(1, acc1, acc1b);
                    NEON_PROCESS_ROW(2, acc2, acc2b);
                    NEON_PROCESS_ROW(3, acc3, acc3b);

#undef NEON_PROCESS_ROW
                }
            }

            // Merge paired accumulators and horizontal sum
            acc0 = vaddq_f32(acc0, acc0b);
            acc1 = vaddq_f32(acc1, acc1b);
            acc2 = vaddq_f32(acc2, acc2b);
            acc3 = vaddq_f32(acc3, acc3b);

            C[m * N + n + 0] = vaddvq_f32(acc0);
            C[m * N + n + 1] = vaddvq_f32(acc1);
            C[m * N + n + 2] = vaddvq_f32(acc2);
            C[m * N + n + 3] = vaddvq_f32(acc3);
        }

        // Handle remaining columns (N % 4)
        for (; n < N; n++) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            float32x4_t acc_b = vdupq_n_f32(0.0f);

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float32x4_t vscale = vdupq_n_f32(scales[n * num_groups + g]);
                const float32x4_t vzero = vdupq_n_f32(zero_points[n * num_groups + g]);
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 8) {
                    float32x4_t a0 = vld1q_f32(a_ptr + k);
                    float32x4_t a1 = vld1q_f32(a_ptr + k + 4);

                    uint8_t packed_tmp[8] = {};
                    std::memcpy(packed_tmp, w_ptr + k / 2, sizeof(uint32_t));
                    uint8x8_t packed_u8 = vld1_u8(packed_tmp);
                    uint16x8_t packed_u16 = vmovl_u8(packed_u8);
                    uint16x8_t lo_u16 = vandq_u16(packed_u16, vdupq_n_u16(0x0F));
                    uint16x8_t hi_u16 = vshrq_n_u16(packed_u16, 4);
                    hi_u16 = vandq_u16(hi_u16, vdupq_n_u16(0x0F));

                    int16x8_t lo_s16 = vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_u16(lo_u16), 12), 12);
                    int16x8_t hi_s16 = vshrq_n_s16(vshlq_n_s16(vreinterpretq_s16_u16(hi_u16), 12), 12);

                    int16x8_t interleaved_lo = vzip1q_s16(lo_s16, hi_s16);

                    int32x4_t w32_0 = vmovl_s16(vget_low_s16(interleaved_lo));
                    int32x4_t w32_1 = vmovl_s16(vget_high_s16(interleaved_lo));

                    float32x4_t wf0 = vmulq_f32(vscale, vsubq_f32(vcvtq_f32_s32(w32_0), vzero));
                    float32x4_t wf1 = vmulq_f32(vscale, vsubq_f32(vcvtq_f32_s32(w32_1), vzero));

                    acc = vfmaq_f32(acc, a0, wf0);
                    acc_b = vfmaq_f32(acc_b, a1, wf1);
                }
            }

            acc = vaddq_f32(acc, acc_b);
            C[m * N + n] = vaddvq_f32(acc);
        }
    }
}

#endif  // DENSECORE_ARM

// =============================================================================
// FlashAttention AVX-512 Micro-Kernels (Cache-Optimized)
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Fast exp approximation using polynomial (AVX-512)
 *
 * Approximates exp(x) for x in [-87, 87] using a minimax polynomial.
 * Faster than standard exp but with ~1e-6 relative error.
 */
inline __m512 _mm512_fast_exp_ps(__m512 x) {
    // Clamp to valid range
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(87.0f));

    // exp(x) = 2^(x/ln(2)) = 2^k * 2^f where k=floor(x/ln(2)), f=x/ln(2)-k
    const __m512 c_invlog2 = _mm512_set1_ps(1.44269504f);  // 1/ln(2)
    const __m512 c_log2 = _mm512_set1_ps(0.69314718f);     // ln(2)

    __m512 z = _mm512_mul_ps(x, c_invlog2);
    __m512i k = _mm512_cvttps_epi32(z);
    __m512 f = _mm512_sub_ps(x, _mm512_mul_ps(_mm512_cvtepi32_ps(k), c_log2));

    // Polynomial approximation for 2^f (f in [0,1]) via Taylor series:
    // 2^f ≈ c0 + c1*f + c2*f² + c3*f³
    // where c0=1, c1=ln(2), c2=ln(2)²/2!, c3=ln(2)³/3!
    const __m512 c0 = _mm512_set1_ps(1.0f);         // 1.0
    const __m512 c1 = _mm512_set1_ps(0.69314718f);  // ln(2)
    const __m512 c2 = _mm512_set1_ps(0.24022651f);  // ln(2)^2 / 2
    const __m512 c3 = _mm512_set1_ps(0.05550410f);  // ln(2)^3 / 6

    // Horner's method: ((c3*f + c2)*f + c1)*f + c0
    __m512 poly = c3;
    poly = _mm512_fmadd_ps(poly, f, c2);
    poly = _mm512_fmadd_ps(poly, f, c1);
    poly = _mm512_fmadd_ps(poly, f, c0);

    // 2^k using bit manipulation
    k = _mm512_slli_epi32(_mm512_add_epi32(k, _mm512_set1_epi32(127)), 23);
    __m512 pow2k = _mm512_castsi512_ps(k);

    return _mm512_mul_ps(poly, pow2k);
}

/**
 * @brief Compute Q @ K^T with scaling (AVX-512 optimized)
 *
 * Computes S[i,j] = scale * sum_d(Q[i,d] * K[j,d]) for block of queries and
 * keys.
 *
 * @param Q Query block [q_len, head_dim]
 * @param K Key block [kv_len, head_dim]
 * @param S Output scores [q_len, kv_len]
 * @param q_len Number of query vectors in block
 * @param kv_len Number of key vectors in block
 * @param head_dim Dimension per head
 * @param scale Scaling factor (typically 1/sqrt(head_dim))
 */
inline void ComputeQK_AVX512(const float* Q, const float* K, float* S, int q_len, int kv_len, int head_dim,
                             float scale) {
    [[maybe_unused]] const __m512 scale_vec = _mm512_set1_ps(scale);

    for (int qi = 0; qi < q_len; qi++) {
        const float* q_row = Q + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const float* k_row = K + ki * head_dim;

            // Dot product with AVX-512
            __m512 sum = _mm512_setzero_ps();
            int d = 0;

            // Process 16 elements at a time
            for (; d + 16 <= head_dim; d += 16) {
                __m512 q_vec = _mm512_loadu_ps(q_row + d);
                __m512 k_vec = _mm512_loadu_ps(k_row + d);
                sum = _mm512_fmadd_ps(q_vec, k_vec, sum);
            }

            // Horizontal sum of 16-element vector
            float dot = _mm512_reduce_add_ps(sum);

            // Handle remainder with scalar
            for (; d < head_dim; d++) {
                dot += q_row[d] * k_row[d];
            }

            S[qi * kv_len + ki] = dot * scale;
        }
    }
}

/**
 * @brief Apply causal mask to attention scores (AVX-512 optimized)
 *
 * Sets S[qi, ki] = -inf if (qi + q_start) < (ki + kv_start) for causal
 * masking.
 *
 * @param S Scores [q_len, kv_len] (modified in-place)
 * @param q_start Global query offset
 * @param kv_start Global kv offset
 * @param q_len Number of query vectors
 * @param kv_len Number of kv vectors
 */
inline void ApplyMask_AVX512(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    const __m512 neg_inf = _mm512_set1_ps(-1e10f);

    for (int qi = 0; qi < q_len; qi++) {
        const int global_qi = q_start + qi;
        float* s_row = S + qi * kv_len;

        int ki = 0;
        // Vectorized masking
        for (; ki + 16 <= kv_len; ki += 16) {
            // Create mask: true where (global_qi < kv_start + ki + lane)
            __m512i ki_vec = _mm512_add_epi32(_mm512_set1_epi32(kv_start + ki),
                                              _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
            __mmask16 mask = _mm512_cmplt_epi32_mask(_mm512_set1_epi32(global_qi), ki_vec);

            // Load current scores
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            // Blend: if mask=1, use -inf, else keep s_vec
            s_vec = _mm512_mask_blend_ps(mask, s_vec, neg_inf);
            _mm512_storeu_ps(s_row + ki, s_vec);
        }

        // Scalar remainder
        for (; ki < kv_len; ki++) {
            if (global_qi < (kv_start + ki)) {
                s_row[ki] = -1e10f;
            }
        }
    }
}

/**
 * @brief Online softmax with AVX-512
 *
 * Computes softmax over block and updates running statistics.
 * Uses online algorithm to handle arbitrary sequence lengths.
 *
 * @param S Input/output scores [q_len, kv_len]
 * @param row_max Max value per row [q_len] (updated)
 * @param row_sum Exponential sum per row [q_len] (updated)
 * @param q_len Number of query rows
 * @param kv_len Number of attention scores per row
 * @param first_block If true, initialize max/sum; else update
 */
inline void SoftmaxBlock_AVX512(float* S, float* row_max, float* row_sum, int q_len, int kv_len, bool first_block) {
    for (int qi = 0; qi < q_len; qi++) {
        float* s_row = S + qi * kv_len;

        // Find max in this block
        __m512 max_vec = _mm512_set1_ps(-1e10f);
        int ki = 0;
        for (; ki + 16 <= kv_len; ki += 16) {
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            max_vec = _mm512_max_ps(max_vec, s_vec);
        }
        float local_max = _mm512_reduce_max_ps(max_vec);

        // Scalar remainder for max
        for (; ki < kv_len; ki++) {
            local_max = std::max(local_max, s_row[ki]);
        }

        // Update global max
        float m_old = first_block ? -1e10f : row_max[qi];
        float m_new = std::max(m_old, local_max);
        const __m512 m_new_vec = _mm512_set1_ps(m_new);

        // Compute exp(s - m_new) and sum
        __m512 sum_vec = _mm512_setzero_ps();
        ki = 0;
        for (; ki + 16 <= kv_len; ki += 16) {
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            __m512 exp_vec = _mm512_sub_ps(s_vec, m_new_vec);
            exp_vec = _mm512_fast_exp_ps(exp_vec);  // Fast exp
            _mm512_storeu_ps(s_row + ki, exp_vec);
            sum_vec = _mm512_add_ps(sum_vec, exp_vec);
        }
        float local_sum = _mm512_reduce_add_ps(sum_vec);

        // Scalar remainder
        for (; ki < kv_len; ki++) {
            float exp_val = expf(s_row[ki] - m_new);
            s_row[ki] = exp_val;
            local_sum += exp_val;
        }

        // Update running statistics
        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

/**
 * @brief Compute P @ V (attention-weighted values) with AVX-512
 *
 * Computes O[qi] += sum_ki(P[qi,ki] * V[ki]) for a block.
 *
 * @param P Attention probabilities [q_len, kv_len]
 * @param V Value vectors [kv_len, head_dim]
 * @param O Output accumulator [q_len, head_dim] (incremented)
 * @param q_len Number of query vectors
 * @param kv_len Number of key/value vectors
 * @param head_dim Head dimension
 */
inline void ComputePV_AVX512(const float* P, const float* V, float* O, int q_len, int kv_len, int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const float* p_row = P + qi * kv_len;
        float* o_row = O + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const float p_val = p_row[ki];
            const __m512 p_vec = _mm512_set1_ps(p_val);
            const float* v_row = V + ki * head_dim;

            int d = 0;
            // Vectorized accumulation
            for (; d + 16 <= head_dim; d += 16) {
                __m512 o_vec = _mm512_loadu_ps(o_row + d);
                __m512 v_vec = _mm512_loadu_ps(v_row + d);
                o_vec = _mm512_fmadd_ps(p_vec, v_vec, o_vec);
                _mm512_storeu_ps(o_row + d, o_vec);
            }

            // Scalar remainder
            for (; d < head_dim; d++) {
                o_row[d] += p_val * v_row[d];
            }
        }
    }
}

/**
 * @brief Update output with rescaling (AVX-512)
 *
 * Rescales existing output and adds new contribution:
 * O_new[qi] = alpha[qi] * O_old[qi] + beta[qi] * PV[qi]
 *
 * @param O Output [q_len, head_dim] (updated in-place)
 * @param PV New contribution [q_len, head_dim]
 * @param alpha Rescale factors [q_len]
 * @param beta New contribution factors [q_len]
 * @param q_len Number of query vectors
 * @param head_dim Head dimension
 */
inline void UpdateOutput_AVX512(float* O, const float* PV, const float* alpha, const float* beta, int q_len,
                                int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const __m512 alpha_vec = _mm512_set1_ps(alpha[qi]);
        const __m512 beta_vec = _mm512_set1_ps(beta[qi]);
        float* o_row = O + qi * head_dim;
        const float* pv_row = PV + qi * head_dim;

        int d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            __m512 o_vec = _mm512_loadu_ps(o_row + d);
            __m512 pv_vec = _mm512_loadu_ps(pv_row + d);

            // O = alpha * O + beta * PV
            o_vec = _mm512_mul_ps(o_vec, alpha_vec);
            o_vec = _mm512_fmadd_ps(beta_vec, pv_vec, o_vec);

            _mm512_storeu_ps(o_row + d, o_vec);
        }

        // Scalar remainder
        for (; d < head_dim; d++) {
            o_row[d] = alpha[qi] * o_row[d] + beta[qi] * pv_row[d];
        }
    }
}

#else  // Non-AVX512 fallback (scalar implementations)

// Scalar fallback versions — named _Scalar to avoid confusion with the
// actual AVX-512 intrinsic implementations in the #if branch above.
inline void ComputeQK_Scalar(const float* Q, const float* K, float* S, int q_len, int kv_len, int head_dim,
                             float scale) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            float dot = DotF32(Q + qi * head_dim, K + ki * head_dim, head_dim);
            S[qi * kv_len + ki] = dot * scale;
        }
    }
}

inline void ApplyMask_Scalar(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            if ((q_start + qi) < (kv_start + ki)) {
                S[qi * kv_len + ki] = -1e10f;
            }
        }
    }
}

inline void SoftmaxBlock_Scalar(float* S, float* row_max, float* row_sum, int q_len, int kv_len, bool first_block) {
    for (int qi = 0; qi < q_len; qi++) {
        float* s_row = S + qi * kv_len;
        float local_max = MaxF32(s_row, kv_len);
        float m_old = first_block ? -1e10f : row_max[qi];
        float m_new = std::max(m_old, local_max);

        float local_sum = 0.0f;
        for (int ki = 0; ki < kv_len; ki++) {
            s_row[ki] = expf(s_row[ki] - m_new);
            local_sum += s_row[ki];
        }

        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

inline void ComputePV_Scalar(const float* P, const float* V, float* O, int q_len, int kv_len, int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            float p_val = P[qi * kv_len + ki];
            const float* v_row = V + ki * head_dim;
            float* o_row = O + qi * head_dim;
            for (int d = 0; d < head_dim; d++) {
                o_row[d] += p_val * v_row[d];
            }
        }
    }
}

inline void UpdateOutput_Scalar(float* O, const float* PV, const float* alpha, const float* beta, int q_len,
                                int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = alpha[qi] * O[qi * head_dim + d] + beta[qi] * PV[qi * head_dim + d];
        }
    }
}

#endif  // __AVX512F__

// =============================================================================
// FlashAttention ARM Kernels
// =============================================================================

#if defined(__ARM_FEATURE_SVE)

inline void ComputeQK_SVE(const float* Q, const float* K, float* S, int q_len, int kv_len, int head_dim, float scale) {
    for (int qi = 0; qi < q_len; qi++) {
        const float* q_row = Q + qi * head_dim;
        for (int ki = 0; ki < kv_len; ki++) {
            const float* k_row = K + ki * head_dim;
            S[qi * kv_len + ki] = DotF32(q_row, k_row, static_cast<size_t>(head_dim)) * scale;
        }
    }
}

inline void ApplyMask_SVE(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    const float neg_inf = -1e10f;
    const uint64_t vl = svcntw();
    const svfloat32_t neg_inf_vec = svdup_f32(neg_inf);
    for (int qi = 0; qi < q_len; ++qi) {
        const int global_qi = q_start + qi;
        float* s_row = S + qi * kv_len;
        const int mask_start = std::max(0, global_qi - kv_start + 1);

        int ki = mask_start;
        for (; ki + static_cast<int>(vl) <= kv_len; ki += static_cast<int>(vl)) {
            svst1_f32(svptrue_b32(), s_row + ki, neg_inf_vec);
        }
        if (ki < kv_len) {
            const svbool_t pg = svwhilelt_b32_u64(static_cast<uint64_t>(ki), static_cast<uint64_t>(kv_len));
            svst1_f32(pg, s_row + ki, neg_inf_vec);
        }
    }
}

inline void SoftmaxBlock_SVE(float* S, float* row_max, float* row_sum, int q_len, int kv_len, bool first_block) {
    const uint64_t vl = svcntw();
    for (int qi = 0; qi < q_len; ++qi) {
        float* s_row = S + qi * kv_len;

        svfloat32_t max_vec = svdup_f32(-1e10f);
        int ki = 0;
        for (; ki + static_cast<int>(vl) <= kv_len; ki += static_cast<int>(vl)) {
            const svbool_t pg = svptrue_b32();
            const svfloat32_t s_vec = svld1_f32(pg, s_row + ki);
            max_vec = svmax_f32_x(pg, max_vec, s_vec);
        }
        float local_max = svmaxv_f32(svptrue_b32(), max_vec);
        for (; ki < kv_len; ++ki) {
            local_max = std::max(local_max, s_row[ki]);
        }

        const float m_old = first_block ? -1e10f : row_max[qi];
        const float m_new = std::max(m_old, local_max);
        const svfloat32_t m_new_vec = svdup_f32(m_new);
        svfloat32_t sum_vec = svdup_f32(0.0f);

        ki = 0;
        for (; ki + static_cast<int>(vl) <= kv_len; ki += static_cast<int>(vl)) {
            const svbool_t pg = svptrue_b32();
            const svfloat32_t s_vec = svld1_f32(pg, s_row + ki);
            const svfloat32_t shifted = svsub_f32_x(pg, s_vec, m_new_vec);
            alignas(64) float tmp[64];
            svst1_f32(pg, tmp, shifted);
            for (uint64_t lane = 0; lane < vl; ++lane) {
                tmp[lane] = expf(tmp[lane]);
            }
            const svfloat32_t exp_vec = svld1_f32(pg, tmp);
            svst1_f32(pg, s_row + ki, exp_vec);
            sum_vec = svadd_f32_x(pg, sum_vec, exp_vec);
        }

        float local_sum = svaddv_f32(svptrue_b32(), sum_vec);
        for (; ki < kv_len; ++ki) {
            const float exp_val = expf(s_row[ki] - m_new);
            s_row[ki] = exp_val;
            local_sum += exp_val;
        }

        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            const float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

inline void ComputePV_SVE(const float* P, const float* V, float* O, int q_len, int kv_len, int head_dim) {
    const uint64_t vl = svcntw();
    for (int qi = 0; qi < q_len; qi++) {
        const float* p_row = P + qi * kv_len;
        float* o_row = O + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const svfloat32_t p_vec = svdup_f32(p_row[ki]);
            const float* v_row = V + ki * head_dim;

            int d = 0;
            for (; d + static_cast<int>(vl) <= head_dim; d += static_cast<int>(vl)) {
                svbool_t pg = svptrue_b32();
                svfloat32_t o_vec = svld1_f32(pg, o_row + d);
                svfloat32_t v_vec = svld1_f32(pg, v_row + d);
                o_vec = svmla_f32_x(pg, o_vec, p_vec, v_vec);
                svst1_f32(pg, o_row + d, o_vec);
            }
            if (d < head_dim) {
                svbool_t pg = svwhilelt_b32_u64(static_cast<uint64_t>(d), static_cast<uint64_t>(head_dim));
                svfloat32_t o_vec = svld1_f32(pg, o_row + d);
                svfloat32_t v_vec = svld1_f32(pg, v_row + d);
                o_vec = svmla_f32_m(pg, o_vec, p_vec, v_vec);
                svst1_f32(pg, o_row + d, o_vec);
            }
        }
    }
}

inline void UpdateOutput_SVE(float* O, const float* PV, const float* alpha, const float* beta, int q_len,
                             int head_dim) {
    const uint64_t vl = svcntw();
    for (int qi = 0; qi < q_len; qi++) {
        const svfloat32_t alpha_vec = svdup_f32(alpha[qi]);
        const svfloat32_t beta_vec = svdup_f32(beta[qi]);
        float* o_row = O + qi * head_dim;
        const float* pv_row = PV + qi * head_dim;

        int d = 0;
        for (; d + static_cast<int>(vl) <= head_dim; d += static_cast<int>(vl)) {
            svbool_t pg = svptrue_b32();
            svfloat32_t o_vec = svld1_f32(pg, o_row + d);
            svfloat32_t pv_vec = svld1_f32(pg, pv_row + d);
            o_vec = svmul_f32_x(pg, o_vec, alpha_vec);
            o_vec = svmla_f32_x(pg, o_vec, beta_vec, pv_vec);
            svst1_f32(pg, o_row + d, o_vec);
        }
        if (d < head_dim) {
            svbool_t pg = svwhilelt_b32_u64(static_cast<uint64_t>(d), static_cast<uint64_t>(head_dim));
            svfloat32_t o_vec = svld1_f32(pg, o_row + d);
            svfloat32_t pv_vec = svld1_f32(pg, pv_row + d);
            o_vec = svmul_f32_m(pg, o_vec, alpha_vec);
            o_vec = svmla_f32_m(pg, o_vec, beta_vec, pv_vec);
            svst1_f32(pg, o_row + d, o_vec);
        }
    }
}

#endif  // __ARM_FEATURE_SVE

#if defined(DENSECORE_ARM)

inline void ComputeQK_NEON(const float* Q, const float* K, float* S, int q_len, int kv_len, int head_dim, float scale) {
    for (int qi = 0; qi < q_len; qi++) {
        const float* q_row = Q + qi * head_dim;
        for (int ki = 0; ki < kv_len; ki++) {
            const float* k_row = K + ki * head_dim;
            S[qi * kv_len + ki] = DotF32(q_row, k_row, static_cast<size_t>(head_dim)) * scale;
        }
    }
}

inline void ApplyMask_NEON(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    const float neg_inf = -1e10f;
    const float32x4_t neg_inf_vec = vdupq_n_f32(neg_inf);
    for (int qi = 0; qi < q_len; qi++) {
        const int global_qi = q_start + qi;
        float* s_row = S + qi * kv_len;

        // Causal mask: set s_row[ki] = -inf where global_qi < kv_start + ki.
        // Equivalently, all ki > (global_qi - kv_start) get masked.
        // The mask boundary is a single index per row, so the real win is
        // vectorized stores for the contiguous -inf tail region.
        const int mask_start = std::max(0, global_qi - kv_start + 1);

        // Everything before mask_start is kept as-is (no work needed).
        // Fill the tail [mask_start, kv_len) with -inf using NEON stores.
        int ki = mask_start;
        for (; ki + 4 <= kv_len; ki += 4) {
            vst1q_f32(s_row + ki, neg_inf_vec);
        }
        for (; ki < kv_len; ki++) {
            s_row[ki] = neg_inf;
        }
    }
}

inline void SoftmaxBlock_NEON(float* S, float* row_max, float* row_sum, int q_len, int kv_len, bool first_block) {
    for (int qi = 0; qi < q_len; qi++) {
        float* s_row = S + qi * kv_len;

        // Find max using NEON
        float32x4_t max_vec = vdupq_n_f32(-1e10f);
        int ki = 0;
        for (; ki + 4 <= kv_len; ki += 4) {
            float32x4_t s_vec = vld1q_f32(s_row + ki);
            max_vec = vmaxq_f32(max_vec, s_vec);
        }
        // Horizontal max reduction
        float local_max = vmaxvq_f32(max_vec);
        for (; ki < kv_len; ki++) {
            local_max = std::max(local_max, s_row[ki]);
        }

        float m_old = first_block ? -1e10f : row_max[qi];
        float m_new = std::max(m_old, local_max);

        // Compute exp(s - m_new) and sum
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        float32x4_t m_new_vec = vdupq_n_f32(m_new);
        ki = 0;
        for (; ki + 4 <= kv_len; ki += 4) {
            float32x4_t s_vec = vld1q_f32(s_row + ki);
            // exp via scalar fallback (no vexpq on NEON)
            float tmp[4];
            vst1q_f32(tmp, vsubq_f32(s_vec, m_new_vec));
            tmp[0] = expf(tmp[0]);
            tmp[1] = expf(tmp[1]);
            tmp[2] = expf(tmp[2]);
            tmp[3] = expf(tmp[3]);
            float32x4_t exp_vec = vld1q_f32(tmp);
            vst1q_f32(s_row + ki, exp_vec);
            sum_vec = vaddq_f32(sum_vec, exp_vec);
        }
        float local_sum = vaddvq_f32(sum_vec);
        for (; ki < kv_len; ki++) {
            float exp_val = expf(s_row[ki] - m_new);
            s_row[ki] = exp_val;
            local_sum += exp_val;
        }

        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

inline void ComputePV_NEON(const float* P, const float* V, float* O, int q_len, int kv_len, int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const float* p_row = P + qi * kv_len;
        float* o_row = O + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const float32x4_t p_vec = vdupq_n_f32(p_row[ki]);
            const float* v_row = V + ki * head_dim;

            int d = 0;
            for (; d + 4 <= head_dim; d += 4) {
                float32x4_t o_vec = vld1q_f32(o_row + d);
                float32x4_t v_vec = vld1q_f32(v_row + d);
                o_vec = vfmaq_f32(o_vec, p_vec, v_vec);
                vst1q_f32(o_row + d, o_vec);
            }
            for (; d < head_dim; d++) {
                o_row[d] += p_row[ki] * v_row[d];
            }
        }
    }
}

inline void UpdateOutput_NEON(float* O, const float* PV, const float* alpha, const float* beta, int q_len,
                              int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const float32x4_t alpha_vec = vdupq_n_f32(alpha[qi]);
        const float32x4_t beta_vec = vdupq_n_f32(beta[qi]);
        float* o_row = O + qi * head_dim;
        const float* pv_row = PV + qi * head_dim;

        int d = 0;
        for (; d + 4 <= head_dim; d += 4) {
            float32x4_t o_vec = vld1q_f32(o_row + d);
            float32x4_t pv_vec = vld1q_f32(pv_row + d);
            o_vec = vmulq_f32(o_vec, alpha_vec);
            o_vec = vfmaq_f32(o_vec, beta_vec, pv_vec);
            vst1q_f32(o_row + d, o_vec);
        }
        for (; d < head_dim; d++) {
            o_row[d] = alpha[qi] * o_row[d] + beta[qi] * pv_row[d];
        }
    }
}

#endif  // DENSECORE_ARM

inline void ComputeQK(const float* Q, const float* K, float* S, int q_len, int kv_len, int head_dim, float scale) {
    static const SimdLevel level = DetectSimdLevel();
    (void)level;
#if defined(__ARM_FEATURE_SVE)
    if (HasArmSveOrBetter(level)) {
        ComputeQK_SVE(Q, K, S, q_len, kv_len, head_dim, scale);
        return;
    }
#endif
#if defined(DENSECORE_ARM)
    if (IsArmFamily(level)) {
        ComputeQK_NEON(Q, K, S, q_len, kv_len, head_dim, scale);
        return;
    }
#endif
    // x86: real AVX-512 intrinsics when compiled with __AVX512F__,
    // otherwise ComputeQK_Scalar (renamed from the confusing _AVX512 suffix).
#if defined(__AVX512F__)
    ComputeQK_AVX512(Q, K, S, q_len, kv_len, head_dim, scale);
#else
    ComputeQK_Scalar(Q, K, S, q_len, kv_len, head_dim, scale);
#endif
}

inline void ApplyMask(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    static const SimdLevel level = DetectSimdLevel();
    (void)level;
#if defined(__ARM_FEATURE_SVE)
    if (HasArmSveOrBetter(level)) {
        ApplyMask_SVE(S, q_start, kv_start, q_len, kv_len);
        return;
    }
#endif
#if defined(DENSECORE_ARM)
    if (IsArmFamily(level)) {
        ApplyMask_NEON(S, q_start, kv_start, q_len, kv_len);
        return;
    }
#endif
#if defined(__AVX512F__)
    ApplyMask_AVX512(S, q_start, kv_start, q_len, kv_len);
#else
    ApplyMask_Scalar(S, q_start, kv_start, q_len, kv_len);
#endif
}

inline void SoftmaxBlock(float* S, float* row_max, float* row_sum, int q_len, int kv_len, bool first_block) {
    static const SimdLevel level = DetectSimdLevel();
    (void)level;
#if defined(__ARM_FEATURE_SVE)
    if (HasArmSveOrBetter(level)) {
        SoftmaxBlock_SVE(S, row_max, row_sum, q_len, kv_len, first_block);
        return;
    }
#endif
#if defined(DENSECORE_ARM)
    if (IsArmFamily(level)) {
        SoftmaxBlock_NEON(S, row_max, row_sum, q_len, kv_len, first_block);
        return;
    }
#endif
#if defined(__AVX512F__)
    SoftmaxBlock_AVX512(S, row_max, row_sum, q_len, kv_len, first_block);
#else
    SoftmaxBlock_Scalar(S, row_max, row_sum, q_len, kv_len, first_block);
#endif
}

inline void ComputePV(const float* P, const float* V, float* O, int q_len, int kv_len, int head_dim) {
    static const SimdLevel level = DetectSimdLevel();
    (void)level;
#if defined(__ARM_FEATURE_SVE)
    if (HasArmSveOrBetter(level)) {
        ComputePV_SVE(P, V, O, q_len, kv_len, head_dim);
        return;
    }
#endif
#if defined(DENSECORE_ARM)
    if (IsArmFamily(level)) {
        ComputePV_NEON(P, V, O, q_len, kv_len, head_dim);
        return;
    }
#endif
#if defined(__AVX512F__)
    ComputePV_AVX512(P, V, O, q_len, kv_len, head_dim);
#else
    ComputePV_Scalar(P, V, O, q_len, kv_len, head_dim);
#endif
}

inline void UpdateOutput(float* O, const float* PV, const float* alpha, const float* beta, int q_len, int head_dim) {
    static const SimdLevel level = DetectSimdLevel();
    (void)level;
#if defined(__ARM_FEATURE_SVE)
    if (HasArmSveOrBetter(level)) {
        UpdateOutput_SVE(O, PV, alpha, beta, q_len, head_dim);
        return;
    }
#endif
#if defined(DENSECORE_ARM)
    if (IsArmFamily(level)) {
        UpdateOutput_NEON(O, PV, alpha, beta, q_len, head_dim);
        return;
    }
#endif
#if defined(__AVX512F__)
    UpdateOutput_AVX512(O, PV, alpha, beta, q_len, head_dim);
#else
    UpdateOutput_Scalar(O, PV, alpha, beta, q_len, head_dim);
#endif
}

// =============================================================================
// Fused Kernels for Memory Bandwidth Optimization
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Fused Add + RMSNorm in one pass (AVX-512 optimized)
 *
 * Combines residual addition and RMSNorm into a single kernel to reduce
 * memory bandwidth by loading/storing data once instead of twice.
 *
 * Operation:
 *   1. x_out[i] = x[i] + residual[i]  (residual add)
 *   2. Compute sum_of_squares for RMSNorm
 *   3. Apply normalization: x_out[i] = (x_out[i] * rms_w[i]) / sqrt(sos/n +
 * eps)
 *
 * @param x_out Output tensor [n] (can be same as x for in-place)
 * @param x Input tensor [n]
 * @param residual Residual tensor [n]
 * @param rms_w RMSNorm weight tensor [n]
 * @param n Number of elements
 * @param eps Epsilon for numerical stability (default: 1e-5)
 */
inline void AddRMSNorm_AVX512(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                              float eps = 1e-5f) {
    // Pass 1: Add residual and compute sum of squares simultaneously
    __m512 sos_vec = _mm512_setzero_ps();
    size_t i = 0;

    // Vectorized: add and accumulate sum-of-squares
    for (; i + 16 <= n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 res_vec = _mm512_loadu_ps(residual + i);

        // Fused add: x_out = x + residual
        __m512 sum = _mm512_add_ps(x_vec, res_vec);
        _mm512_storeu_ps(x_out + i, sum);

        // Accumulate sum of squares for RMSNorm
        sos_vec = _mm512_fmadd_ps(sum, sum, sos_vec);
    }

    // Scalar remainder for pass 1
    float sos_scalar = _mm512_reduce_add_ps(sos_vec);
    for (; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos_scalar += val * val;
    }

    // Compute normalization factor: 1 / sqrt(mean(x^2) + eps)
    float rms = sqrtf(sos_scalar / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;
    const __m512 scale_vec = _mm512_set1_ps(scale);

    // Pass 2: Apply normalized weight multiplication
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 val = _mm512_loadu_ps(x_out + i);
        __m512 w = _mm512_loadu_ps(rms_w + i);

        // x_out = (x_out * scale) * weight = x_out * w * scale
        val = _mm512_mul_ps(val, scale_vec);
        val = _mm512_mul_ps(val, w);
        _mm512_storeu_ps(x_out + i, val);
    }

    // Scalar remainder for pass 2
    for (; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

/**
 * @brief Fused Q/K/V projection in one pass (AVX-512 optimized)
 *
 * Computes Q = x @ W_q, K = x @ W_k, V = x @ W_v simultaneously.
 * Reduces memory bandwidth by loading input x into registers once
 * and computing all three projections with interleaved instructions.
 *
 * Note: This is a simplified version for decode (batch=1) scenarios.
 * For larger batches, use standard GEMM calls.
 *
 * @param q Output Q projection [n_embd]
 * @param k Output K projection [dim_k]
 * @param v Output V projection [dim_v]
 * @param x Input tensor [n_embd]
 * @param w_q Q weight [n_embd, dim_q] (row-major)
 * @param w_k K weight [n_embd, dim_k] (row-major)
 * @param w_v V weight [n_embd, dim_v] (row-major)
 * @param n_embd Input dimension
 * @param dim_q Q output dimension (= n_head * head_dim)
 * @param dim_k K output dimension (= n_head_kv * head_dim)
 * @param dim_v V output dimension (= n_head_kv * head_dim)
 */
inline void ComputeQKV_AVX512(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                              const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    // Virtual column space: [0, dim_q) = Q, [dim_q, dim_q+dim_k) = K,
    //                       [dim_q+dim_k, total) = V
    // Each thread computes a slice [start_col, end_col) of this virtual space.
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;  // Ceiling division
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // Early exit if this thread has no work
    if (start_col >= total_cols) return;

    // For each output position, compute dot product with corresponding weight row
    // Process multiple output positions for better ILP

    constexpr int UNROLL = 4;  // Process 4 outputs at a time for better pipelining

    // ==========================================================================
    // Q Projection: virtual columns [0, dim_q)
    // ==========================================================================
    // Compute intersection with this thread's range
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);

    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_q + oq * n_embd;
            const float* w1 = w_q + (oq + 1) * n_embd;
            const float* w2 = w_q + (oq + 2) * n_embd;
            const float* w3 = w_q + (oq + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);

                // Load weights and FMA
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            // Reduce and store
            q[oq + 0] = _mm512_reduce_add_ps(acc0);
            q[oq + 1] = _mm512_reduce_add_ps(acc1);
            q[oq + 2] = _mm512_reduce_add_ps(acc2);
            q[oq + 3] = _mm512_reduce_add_ps(acc3);

            // Handle remainder for this output group
            for (; d < n_embd; d++) {
                q[oq + 0] += x[d] * w0[d];
                q[oq + 1] += x[d] * w1[d];
                q[oq + 2] += x[d] * w2[d];
                q[oq + 3] += x[d] * w3[d];
            }
        }

        // Handle remaining Q outputs in this thread's range
        for (; oq < q_end; oq++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_q + oq * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            q[oq] = sum;
        }
    }

    // ==========================================================================
    // K Projection: virtual columns [dim_q, dim_q + dim_k)
    // ==========================================================================
    // Map virtual start/end to K-local indices
    const int k_virt_start = dim_q;
    [[maybe_unused]] const int k_virt_end = dim_q + dim_k;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);

    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_k + ok * n_embd;
            const float* w1 = w_k + (ok + 1) * n_embd;
            const float* w2 = w_k + (ok + 2) * n_embd;
            const float* w3 = w_k + (ok + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            k[ok + 0] = _mm512_reduce_add_ps(acc0);
            k[ok + 1] = _mm512_reduce_add_ps(acc1);
            k[ok + 2] = _mm512_reduce_add_ps(acc2);
            k[ok + 3] = _mm512_reduce_add_ps(acc3);

            for (; d < n_embd; d++) {
                k[ok + 0] += x[d] * w0[d];
                k[ok + 1] += x[d] * w1[d];
                k[ok + 2] += x[d] * w2[d];
                k[ok + 3] += x[d] * w3[d];
            }
        }

        for (; ok < k_end; ok++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_k + ok * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            k[ok] = sum;
        }
    }

    // ==========================================================================
    // V Projection: virtual columns [dim_q + dim_k, total_cols)
    // ==========================================================================
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);

    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_v + ov * n_embd;
            const float* w1 = w_v + (ov + 1) * n_embd;
            const float* w2 = w_v + (ov + 2) * n_embd;
            const float* w3 = w_v + (ov + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            v[ov + 0] = _mm512_reduce_add_ps(acc0);
            v[ov + 1] = _mm512_reduce_add_ps(acc1);
            v[ov + 2] = _mm512_reduce_add_ps(acc2);
            v[ov + 3] = _mm512_reduce_add_ps(acc3);

            for (; d < n_embd; d++) {
                v[ov + 0] += x[d] * w0[d];
                v[ov + 1] += x[d] * w1[d];
                v[ov + 2] += x[d] * w2[d];
                v[ov + 3] += x[d] * w3[d];
            }
        }

        for (; ov < v_end; ov++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_v + ov * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            v[ov] = sum;
        }
    }
}

#else  // Scalar fallback for non-AVX512

// Forward declare scalar implementations
inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                              float eps);

inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                              const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith, int nth);

/**
 * @brief Scalar fallback for fused Add + RMSNorm
 */
inline void AddRMSNorm_AVX512(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                              float eps = 1e-5f) {
    AddRMSNorm_Scalar(x_out, x, residual, rms_w, n, eps);
}

/**
 * @brief Scalar fallback for fused Q/K/V projection
 */
inline void ComputeQKV_AVX512(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                              const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
}

#endif  // __AVX512F__

// =============================================================================
// AVX2 Fused Kernels
// =============================================================================

#if defined(__AVX2__) && DENSECORE_HAS_FMA

/**
 * @brief Fused Add + RMSNorm in one pass (AVX2 optimized)
 *
 * SAFETY: This function handles arbitrary dimension sizes (dim % 8 != 0) via:
 *   - Main loop: processes 8 elements at a time (i + 8 <= n)
 *   - Scalar fallback: handles remaining 0-7 elements
 *   - Uses _mm256_loadu_ps (unaligned loads) for safety with arbitrary pointers
 */
inline void AddRMSNorm_AVX2(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                            float eps = 1e-5f) {
    // Early exit for empty input
    if (n == 0) return;

    __m256 sos_vec = _mm256_setzero_ps();
    size_t i = 0;

    // Pass 1: Fused Add + Sum of Squares
    for (; i + 8 <= n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 res_vec = _mm256_loadu_ps(residual + i);

        // x_out = x + residual
        __m256 sum = _mm256_add_ps(x_vec, res_vec);
        _mm256_storeu_ps(x_out + i, sum);

        // Accumulate sum of squares
        sos_vec = _mm256_fmadd_ps(sum, sum, sos_vec);
    }

    // Horizontal reduction for AVX2
    // 1. Extract high 128 bits and add to low 128 bits
    __m128 hi = _mm256_extractf128_ps(sos_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sos_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);

    // 2. Reduce 128 vectors using movehl and shuffle (faster than hadd)
    // [a, b, c, d] -> [a+c, b+d, c, d]
    __m128 movehl = _mm_movehl_ps(sum128, sum128);
    sum128 = _mm_add_ps(sum128, movehl);
    // [a+c, b+d, ...] -> [a+c+b+d, ...]
    __m128 shuffle = _mm_shuffle_ps(sum128, sum128, 1);
    sum128 = _mm_add_ss(sum128, shuffle);

    float sos_scalar = _mm_cvtss_f32(sum128);

    // Handle remainder for Pass 1
    for (; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos_scalar += val * val;
    }

    // Compute scale
    float rms = sqrtf(sos_scalar / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;
    __m256 scale_vec = _mm256_set1_ps(scale);

    // Pass 2: Apply normalization and weight
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 val = _mm256_loadu_ps(x_out + i);
        __m256 w = _mm256_loadu_ps(rms_w + i);

        val = _mm256_mul_ps(val, scale_vec);  // normalize
        val = _mm256_mul_ps(val, w);          // weight
        _mm256_storeu_ps(x_out + i, val);
    }

    // Handle remainder for Pass 2
    for (; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

/**
 * @brief Fused Q/K/V projection (AVX2 optimized)
 *
 * SAFETY: This function handles arbitrary dimension sizes (dim % 8 != 0) via:
 *   - Main loop: processes 8 elements at a time for dot product (d + 8 <=
 * n_embd)
 *   - Scalar fallback: handles remaining 0-7 elements in inner dimension
 *   - Output loop handles odd output dimensions via single-element remainder
 * loop
 *   - Uses _mm256_loadu_ps (unaligned loads) for safety with arbitrary pointers
 *
 * Structure:
 *   - Q projection with 2x unrolling + scalar remainder
 *   - K projection with 2x unrolling + scalar remainder
 *   - V projection with 2x unrolling + scalar remainder
 */
inline void ComputeQKV_AVX2(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                            const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // BARRIER SAFETY: Explicit early exit for zero-work threads.
    // This is critical for multi-threaded correctness - threads with no work
    // must still return cleanly to reach barrier sync without executing any
    // loops.
    if (start_col >= end_col) return;

    constexpr int UNROLL = 2;

    // Helper lambda for horizontal sum
    auto hsum256 = [](__m256 v) -> float {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    };

    // ==========================================================================
    // Q Projection: virtual columns [0, dim_q)
    // ==========================================================================
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);

    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_q + oq * n_embd;
            const float* w1 = w_q + (oq + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            q[oq + 0] = hsum256(acc0);
            q[oq + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                q[oq + 0] += x[d] * w0[d];
                q[oq + 1] += x[d] * w1[d];
            }
        }

        for (; oq < q_end; oq++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_q + oq * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            q[oq] = sum;
        }
    }

    // ==========================================================================
    // K Projection: virtual columns [dim_q, dim_q + dim_k)
    // ==========================================================================
    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);

    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_k + ok * n_embd;
            const float* w1 = w_k + (ok + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            k[ok + 0] = hsum256(acc0);
            k[ok + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                k[ok + 0] += x[d] * w0[d];
                k[ok + 1] += x[d] * w1[d];
            }
        }

        for (; ok < k_end; ok++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_k + ok * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            k[ok] = sum;
        }
    }

    // ==========================================================================
    // V Projection: virtual columns [dim_q + dim_k, total_cols)
    // ==========================================================================
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);

    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_v + ov * n_embd;
            const float* w1 = w_v + (ov + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            v[ov + 0] = hsum256(acc0);
            v[ov + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                v[ov + 0] += x[d] * w0[d];
                v[ov + 1] += x[d] * w1[d];
            }
        }

        for (; ov < v_end; ov++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_v + ov * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            v[ov] = sum;
        }
    }
}

#else

// Forward declare to allow wrapper usage
inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                              float eps);
inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                              const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith, int nth);

inline void AddRMSNorm_AVX2(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                            float eps = 1e-5f) {
    AddRMSNorm_Scalar(x_out, x, residual, rms_w, n, eps);
}

inline void ComputeQKV_AVX2(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                            const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
}

#endif  // __AVX2__ && DENSECORE_HAS_FMA

// =============================================================================
// Scalar Implementation (Always Available)
// =============================================================================

inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                              float eps = 1e-5f) {
    // Pass 1: Add and compute sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos += val * val;
    }

    // Compute scale
    float rms = sqrtf(sos / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;

    // Pass 2: Apply normalization
    for (size_t i = 0; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                              const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // BARRIER SAFETY: Explicit early exit for zero-work threads.
    // This is critical for multi-threaded correctness - threads with no work
    // must still return cleanly to reach barrier sync without executing any
    // loops.
    if (start_col >= end_col) return;

    // Q projection: virtual columns [0, dim_q)
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);
    for (int o = q_start; o < q_end; o++) {
        float sum = 0.0f;
        const float* w = w_q + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        q[o] = sum;
    }

    // K projection: virtual columns [dim_q, dim_q + dim_k)
    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);
    for (int o = k_start; o < k_end; o++) {
        float sum = 0.0f;
        const float* w = w_k + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        k[o] = sum;
    }

    // V projection: virtual columns [dim_q + dim_k, total_cols)
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);
    for (int o = v_start; o < v_end; o++) {
        float sum = 0.0f;
        const float* w = w_v + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        v[o] = sum;
    }
}

#if defined(__ARM_FEATURE_SVE)

inline float DotRow_SVE(const float* x, const float* w, int n_embd) {
    svfloat32_t acc = svdup_f32(0.0f);
    const uint64_t vl = svcntw();
    int d = 0;
    for (; d + static_cast<int>(vl) <= n_embd; d += static_cast<int>(vl)) {
        svbool_t pg = svptrue_b32();
        svfloat32_t x_vec = svld1_f32(pg, x + d);
        svfloat32_t w_vec = svld1_f32(pg, w + d);
        acc = svmla_f32_x(pg, acc, x_vec, w_vec);
    }
    if (d < n_embd) {
        svbool_t pg = svwhilelt_b32_u64(static_cast<uint64_t>(d), static_cast<uint64_t>(n_embd));
        svfloat32_t x_vec = svld1_f32(pg, x + d);
        svfloat32_t w_vec = svld1_f32(pg, w + d);
        acc = svmla_f32_m(pg, acc, x_vec, w_vec);
    }
    return svaddv_f32(svptrue_b32(), acc);
}

inline void ComputeQKV_SVE(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                           const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    if (start_col >= end_col) return;

    constexpr int UNROLL = 2;

    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);
    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            q[oq + 0] = DotRow_SVE(x, w_q + (oq + 0) * n_embd, n_embd);
            q[oq + 1] = DotRow_SVE(x, w_q + (oq + 1) * n_embd, n_embd);
        }
        for (; oq < q_end; oq++) {
            q[oq] = DotRow_SVE(x, w_q + oq * n_embd, n_embd);
        }
    }

    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);
    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            k[ok + 0] = DotRow_SVE(x, w_k + (ok + 0) * n_embd, n_embd);
            k[ok + 1] = DotRow_SVE(x, w_k + (ok + 1) * n_embd, n_embd);
        }
        for (; ok < k_end; ok++) {
            k[ok] = DotRow_SVE(x, w_k + ok * n_embd, n_embd);
        }
    }

    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);
    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            v[ov + 0] = DotRow_SVE(x, w_v + (ov + 0) * n_embd, n_embd);
            v[ov + 1] = DotRow_SVE(x, w_v + (ov + 1) * n_embd, n_embd);
        }
        for (; ov < v_end; ov++) {
            v[ov] = DotRow_SVE(x, w_v + ov * n_embd, n_embd);
        }
    }
}

#endif  // __ARM_FEATURE_SVE

#if defined(DENSECORE_ARM)

inline float DotRow_NEON(const float* x, const float* w, int n_embd) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 4 <= n_embd; d += 4) {
        float32x4_t x_vec = vld1q_f32(x + d);
        float32x4_t w_vec = vld1q_f32(w + d);
        acc = vfmaq_f32(acc, x_vec, w_vec);
    }
    float sum = vaddvq_f32(acc);
    for (; d < n_embd; d++) {
        sum += x[d] * w[d];
    }
    return sum;
}

inline void ComputeQKV_NEON(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                            const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    if (start_col >= end_col) return;

    constexpr int UNROLL = 2;

    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);
    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            q[oq + 0] = DotRow_NEON(x, w_q + (oq + 0) * n_embd, n_embd);
            q[oq + 1] = DotRow_NEON(x, w_q + (oq + 1) * n_embd, n_embd);
        }
        for (; oq < q_end; oq++) {
            q[oq] = DotRow_NEON(x, w_q + oq * n_embd, n_embd);
        }
    }

    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);
    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            k[ok + 0] = DotRow_NEON(x, w_k + (ok + 0) * n_embd, n_embd);
            k[ok + 1] = DotRow_NEON(x, w_k + (ok + 1) * n_embd, n_embd);
        }
        for (; ok < k_end; ok++) {
            k[ok] = DotRow_NEON(x, w_k + ok * n_embd, n_embd);
        }
    }

    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);
    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            v[ov + 0] = DotRow_NEON(x, w_v + (ov + 0) * n_embd, n_embd);
            v[ov + 1] = DotRow_NEON(x, w_v + (ov + 1) * n_embd, n_embd);
        }
        for (; ov < v_end; ov++) {
            v[ov] = DotRow_NEON(x, w_v + ov * n_embd, n_embd);
        }
    }
}

#endif  // DENSECORE_ARM

// =============================================================================
// Unified Dispatch Wrappers
// =============================================================================

/**
 * @brief Unified AddRMSNorm dispatcher — delegates to Highway
 */
inline void AddRMSNorm(float* x_out, const float* x, const float* residual, const float* rms_w, size_t n,
                       float eps = 1e-5f) {
    // Defined in hwy_normalization.cc — Highway auto-selects best ISA.
    extern void AddRMSNorm_HwyDispatch(float*, const float*, const float*, const float*, size_t, float);
    AddRMSNorm_HwyDispatch(x_out, x, residual, rms_w, n, eps);
}

/**
 * @brief RMSNorm dispatcher
 */
inline void RMSNorm(const float* input, const float* weight, float* output, size_t n, float eps = 1e-5f) {
    extern void RMSNorm_HwyDispatch(const float*, const float*, float*, size_t, float);
    RMSNorm_HwyDispatch(input, weight, output, n, eps);
}

/**
 * @brief LayerNorm dispatcher
 */
inline void LayerNorm(const float* input, const float* gamma, const float* beta, float* output, size_t n,
                      float eps = 1e-5f) {
    extern void LayerNorm_HwyDispatch(const float*, const float*, const float*, float*, size_t, float);
    LayerNorm_HwyDispatch(input, gamma, beta, output, n, eps);
}

/**
 * @brief AdaLN dispatcher
 */
inline void AdaLN(const float* input, const float* scale, const float* shift, float* output, size_t n,
                  float eps = 1e-5f) {
    extern void AdaLN_HwyDispatch(const float*, const float*, const float*, float*, size_t, float);
    AdaLN_HwyDispatch(input, scale, shift, output, n, eps);
}

/**
 * @brief Unified ComputeQKV dispatcher (runtime SIMD selection)
 *
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 */
inline void ComputeQKV(float* q, float* k, float* v, const float* x, const float* w_q, const float* w_k,
                       const float* w_v, int n_embd, int dim_q, int dim_k, int dim_v, int ith = 0, int nth = 1) {
    static const SimdLevel level = DetectSimdLevel();
    if (HasX86Avx512OrBetter(level)) {
        ComputeQKV_AVX512(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
    } else if (HasX86Avx2OrBetter(level)) {
        ComputeQKV_AVX2(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
#if defined(__ARM_FEATURE_SVE)
    } else if (HasArmSveOrBetter(level)) {
        ComputeQKV_SVE(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
#endif
#if defined(DENSECORE_ARM)
    } else if (IsArmFamily(level)) {
        ComputeQKV_NEON(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
#endif
    } else {
        ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
    }
}

// =============================================================================
// Parallel GEMV for Decode-Phase (N=1) Inference
// =============================================================================
// GGML's ggml_mul_mat parallelizes along M (batch) dimension.
// During decode (batch_size=1), there's NO parallelism opportunity.
// This kernel parallelizes along K (output features) dimension instead.
//
// Threading Contract:
//   - Each thread computes output[start:end] = x @ W[start:end, :]^T
//   - Thread-safe: no shared state between threads
//   - ith: thread index (0-based), nth: total threads
// =============================================================================

/**
 * @brief Parallel GEMV for decode-phase linear operations
 *
 * Computes: output[start:end] = x @ W[start:end, :]^T
 * where start/end are computed from ith/nth thread indices.
 *
 * @param output Output vector [K] (only portion [start:end] is written)
 * @param x Input vector [N] (shared by all threads, read-only)
 * @param weight Weight matrix [K, N] row-major (W[k, :] is row k)
 * @param N Input dimension (number of features)
 * @param K Output dimension (number of output features)
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 */
inline void GemvParallel(float* output, const float* x, const float* weight, int N, int K, int ith, int nth) {
    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;  // Ceiling division
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    // Early exit if this thread has no work
    if (k_start >= K) return;

    [[maybe_unused]] static const SimdLevel level = DetectSimdLevel();

#if defined(__AVX512F__) && DENSECORE_HAS_FMA
    // AVX-512 path with 4-row blocking. This is used by decode-time router
    // projections such as Qwen A3B ffn_gate_inp, where K is small enough that
    // reusing each loaded activation vector across several output rows matters.
    constexpr int UNROLL = 4;

    auto hsum512 = [](__m512 v) -> float {
        return _mm512_reduce_add_ps(v);
    };

    constexpr int PREFETCH_DISTANCE = 8;

    int k = k_start;
    for (; k + UNROLL <= k_end; k += UNROLL) {
        if (k + PREFETCH_DISTANCE < k_end) {
            const float* pf_w0 = weight + (k + PREFETCH_DISTANCE) * N;
            const float* pf_w1 = weight + (k + PREFETCH_DISTANCE + 1) * N;
            const float* pf_w2 = weight + (k + PREFETCH_DISTANCE + 2) * N;
            const float* pf_w3 = weight + (k + PREFETCH_DISTANCE + 3) * N;
            _mm_prefetch(reinterpret_cast<const char*>(pf_w0), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w1), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w2), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w3), _MM_HINT_T0);
        }

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        const float* w0 = weight + (k + 0) * N;
        const float* w1 = weight + (k + 1) * N;
        const float* w2 = weight + (k + 2) * N;
        const float* w3 = weight + (k + 3) * N;

        int n = 0;
        for (; n + 16 <= N; n += 16) {
            if (n + 128 < N) {
                _mm_prefetch(reinterpret_cast<const char*>(w0 + n + 128), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(w1 + n + 128), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(w2 + n + 128), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(w3 + n + 128), _MM_HINT_T1);
            }
            const __m512 x_vec = _mm512_loadu_ps(x + n);
            acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + n), acc0);
            acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + n), acc1);
            acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + n), acc2);
            acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + n), acc3);
        }

        float sum0 = hsum512(acc0);
        float sum1 = hsum512(acc1);
        float sum2 = hsum512(acc2);
        float sum3 = hsum512(acc3);
        for (; n < N; ++n) {
            const float xv = x[n];
            sum0 += xv * w0[n];
            sum1 += xv * w1[n];
            sum2 += xv * w2[n];
            sum3 += xv * w3[n];
        }
        output[k + 0] = sum0;
        output[k + 1] = sum1;
        output[k + 2] = sum2;
        output[k + 3] = sum3;
    }

    for (; k < k_end; ++k) {
        if (k + 1 < k_end) {
            _mm_prefetch(reinterpret_cast<const char*>(weight + (k + 1) * N), _MM_HINT_T0);
        }
        __m512 acc = _mm512_setzero_ps();
        const float* w = weight + k * N;
        int n = 0;
        for (; n + 16 <= N; n += 16) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + n), _mm512_loadu_ps(w + n), acc);
        }
        float sum = hsum512(acc);
        for (; n < N; ++n) {
            sum += x[n] * w[n];
        }
        output[k] = sum;
    }
#elif defined(__AVX2__) && DENSECORE_HAS_FMA
    // AVX2 path with 2x unrolling
    constexpr int UNROLL = 2;

    auto hsum256 = [](__m256 v) -> float {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    };

    // =========================================================================
    // SOFTWARE PREFETCH CONSTANTS
    // =========================================================================
    // Prefetch distance: 8 output rows ahead (~8 cache lines per row)
    // This hides L3→L1 latency (~40-60 cycles) while computing current rows.
    // _MM_HINT_T0 = L1 cache (for data we'll use very soon)
    // _MM_HINT_T1 = L2 cache (for data we'll use in a few iterations)
    // =========================================================================
    constexpr int PREFETCH_DISTANCE = 8;

    int k = k_start;
    for (; k + UNROLL <= k_end; k += UNROLL) {
        // =====================================================================
        // PREFETCH: Fetch weight rows 8 iterations ahead into L1/L2 cache
        // =====================================================================
        // During this iteration (computing rows k, k+1), prefetch rows k+8, k+9.
        // This allows memory subsystem to start fetching while we compute.
        // =====================================================================
        if (k + PREFETCH_DISTANCE < k_end) {
            const float* pf_w0 = weight + (k + PREFETCH_DISTANCE) * N;
            const float* pf_w1 = weight + (k + PREFETCH_DISTANCE + 1) * N;
            // Prefetch first 4 cache lines (256 bytes = 64 floats) of each row
            _mm_prefetch(reinterpret_cast<const char*>(pf_w0), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w0 + 16), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w1), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w1 + 16), _MM_HINT_T0);
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        const float* w0 = weight + k * N;
        const float* w1 = weight + (k + 1) * N;

        int n = 0;
        for (; n + 8 <= N; n += 8) {
            // Prefetch ahead within current row (for very wide matrices)
            if (n + 64 < N) {
                _mm_prefetch(reinterpret_cast<const char*>(w0 + n + 64), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(w1 + n + 64), _MM_HINT_T0);
            }

            __m256 x_vec = _mm256_loadu_ps(x + n);
            acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + n), acc0);
            acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + n), acc1);
        }

        float sum0 = hsum256(acc0);
        float sum1 = hsum256(acc1);

        // Scalar remainder
        for (; n < N; n++) {
            sum0 += x[n] * w0[n];
            sum1 += x[n] * w1[n];
        }

        output[k + 0] = sum0;
        output[k + 1] = sum1;
    }

    // Handle remaining output elements (with prefetch for next iteration)
    for (; k < k_end; k++) {
        // Prefetch next row if available
        if (k + 1 < k_end) {
            const float* pf_w = weight + (k + 1) * N;
            // Prefetch first 4 cache lines (256 bytes = 64 floats) to match main loop intent
            _mm_prefetch(reinterpret_cast<const char*>(pf_w), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w + 16), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf_w + 48), _MM_HINT_T0);
        }

        __m256 acc = _mm256_setzero_ps();
        const float* w = weight + k * N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + n), _mm256_loadu_ps(w + n), acc);
        }
        float sum = hsum256(acc);
        for (; n < N; n++) {
            sum += x[n] * w[n];
        }
        output[k] = sum;
    }
#else
    // Scalar fallback
    for (int k = k_start; k < k_end; k++) {
        float sum = 0.0f;
        const float* w = weight + k * N;
        for (int n = 0; n < N; n++) {
            sum += x[n] * w[n];
        }
        output[k] = sum;
    }
#endif
}

// =============================================================================
// Quantized Dot Product Dispatcher (defined in simd_ops.cpp)
// =============================================================================
// These functions use GGML's native vec_dot kernels for zero-allocation
// dot products. The input must be PRE-QUANTIZED by the caller.
// =============================================================================

/**
 * @brief Compute dot product between quantized weight row and pre-quantized
 * input
 *
 * Uses GGML's native vec_dot kernels for maximum performance.
 * ZERO ALLOCATION: No memory is allocated inside this function.
 *
 * @param weight_type GGML type of the weight row (e.g., GGML_TYPE_Q4_K)
 * @param w_row Pointer to quantized weight row
 * @param input Pointer to PRE-QUANTIZED input (Q8_K for K-quants, Q8_0 for
 * Q8_0, F32 for F32)
 * @param n Number of elements
 * @param output Pointer to output scalar (single float result)
 *
 * NOTE: The caller must pre-quantize the input to the correct format:
 *   - Q4_K, Q5_K, Q6_K weights → input must be Q8_K
 *   - Q8_0 weights → input must be Q8_0
 *   - F32/F16 weights → input must be F32
 */
void ComputeDotProduct(int weight_type, const void* w_row, const void* input, int n, float* output);

/**
 * @brief Compute multiple dot products for a range of output rows
 *
 * Uses native vec_dot kernels with pre-quantized input for zero-allocation
 * performance. Useful for parallel GEMV where each thread handles a subset
 * of output rows.
 *
 * @param weight_type GGML type of the weight tensor
 * @param weight Base pointer to weight tensor data
 * @param row_stride Stride in bytes between rows
 * @param input Pre-quantized input vector (same format requirements as above)
 * @param n Number of elements per row
 * @param output Float output vector [k_end - k_start]
 * @param k_start First output row index (inclusive)
 * @param k_end Last output row index (exclusive)
 */
void ComputeDotProductBatch(int weight_type, const void* weight, size_t row_stride, const void* input, int n,
                            float* output, int k_start, int k_end);

/**
 * @brief Query the maximum supported buffer size for dequantization
 * @return Maximum number of float elements that can be dequantized
 * @note With pre-quantization, this is less relevant but kept for API compat
 */
size_t GetDequantizationBufferSize();

/**
 * @brief Check if a GGML type is supported by ComputeDotProduct
 * @param type GGML type to check
 * @return true if type is supported, false otherwise
 */
bool IsTypeSupported(int type);

// =============================================================================
// SiLU×Mul Fused Kernel (SwiGLU FFN Optimization)
// =============================================================================
// Computes: out[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// This fuses two operations (silu + elementwise_mul) into one pass,
// reducing memory I/O from 4 reads + 2 writes to 2 reads + 1 write.
// =============================================================================

/**
 * @brief SiLU activation function approximation using SIMD
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Uses fast exp approximation for high throughput.
 */
/**
 * @brief SiLU activation: x * sigmoid(x)
 * Delegated to Highway kernel in simd_ops.cpp
 */
void SiLUMul(float* out, const float* gate, const float* up, size_t size);

/**
 * @brief Parallel SiLU×Mul
 */
void SiLUMulParallel(float* out, const float* gate, const float* up, size_t size, int ith, int nth);

}  // namespace simd

namespace native {

/**
 * @brief Token Embedding Lookup
 *
 * Replaces ggml_get_rows() for embedding table lookup.
 * Uses SIMD-optimized memory copy.
 *
 * @param output Output buffer [n_tokens, n_embd]
 * @param embeddings Embedding table [n_vocab, n_embd]
 * @param token_ids Token IDs array [n_tokens]
 * @param n_tokens Number of tokens
 * @param n_embd Embedding dimension
 */
inline void EmbeddingLookup(float* output, const float* embeddings, const int* token_ids, int n_tokens, int n_embd) {
    for (int t = 0; t < n_tokens; t++) {
        const int token_id = token_ids[t];
        const float* src = embeddings + token_id * n_embd;
        float* dst = output + t * n_embd;
        simd::CopyF32(dst, src, n_embd);
    }
}

/**
 * @brief Parallel Token Embedding Lookup
 *
 * Multi-threaded version for large batch sizes.
 */
inline void EmbeddingLookupParallel(float* output, const float* embeddings, const int* token_ids, int n_tokens,
                                    int n_embd, int ith, int nth) {
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    for (int t = t_start; t < t_end; t++) {
        const int token_id = token_ids[t];
        const float* src = embeddings + token_id * n_embd;
        float* dst = output + t * n_embd;
        simd::CopyF32(dst, src, n_embd);
    }
}

/**
 * @brief General Matrix Multiplication: C = A @ B^T
 *
 * SIMD-optimized GEMM for prefill phase (batch > 1).
 * Replaces ggml_mul_mat() for F32 tensors.
 *
 * Memory layout:
 *   A: [M, K] row-major (activations)
 *   B: [N, K] row-major (weights, transposed during computation)
 *   C: [M, N] row-major (output)
 *
 * @param C Output matrix [M, N]
 * @param A Input matrix [M, K]
 * @param B Weight matrix [N, K] (will be transposed)
 * @param M Batch size (number of tokens)
 * @param N Output dimension
 * @param K Input dimension
 */
inline void GemmF32(float* C, const float* A, const float* B, int M, int N, int K) {
#if defined(__AVX2__)
    // Tiled GEMM for better cache utilization
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 8;

    for (int m = 0; m < M; m += TILE_M) {
        const int m_end = std::min(m + TILE_M, M);

        for (int n = 0; n < N; n += TILE_N) {
            const int n_end = std::min(n + TILE_N, N);

            // Compute tile (DotF32 directly assigns, no init needed)
            for (int mm = m; mm < m_end; mm++) {
                const float* a_row = A + mm * K;
                for (int nn = n; nn < n_end; nn++) {
                    const float* b_row = B + nn * K;
                    C[mm * N + nn] = simd::DotF32(a_row, b_row, K);
                }
            }
        }
    }
#else
    // Scalar fallback
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] = simd::DotF32(A + m * K, B + n * K, K);
        }
    }
#endif
}

/**
 * @brief Parallel GEMM with K+N cache tiling
 *
 * Partitions along M dimension for parallel execution. Within each thread,
 * tiles the K and N dimensions for L1/L2 cache optimization:
 *
 * - TILE_K = 512 floats (2KB per row tile) — ensures activation tiles fit in L1
 * - TILE_N = 8 weight rows per tile — weight tile = 8 × 2KB = 16KB, fits in L1
 *
 * Loop order: K-tile → N-tile → M-rows
 * Weight tile B[nt:nt+TILE_N, kt:kt+TILE_K] stays in L1 while all M rows
 * in this thread's partition access it, providing N×M weight reuse.
 *
 * Original code called DotF32(K) per (m,n) pair, causing weight rows to be
 * reloaded from L2/L3 for each m. The tiled version keeps weight data in L1.
 */
inline void GemmF32Parallel(float* C, const float* A, const float* B, int M, int N, int K, int ith, int nth) {
    const int rows_per_thread = (M + nth - 1) / nth;
    const int m_start = ith * rows_per_thread;
    const int m_end = std::min(m_start + rows_per_thread, M);

    if (m_start >= M) return;

    // L1 cache tiling parameters (i7-10870H: 32KB L1d per core)
    // TILE_K × 4 bytes × (TILE_N + M_per_thread) should fit in L1
    constexpr int TILE_K = 512;  // 2KB per row slice
    constexpr int TILE_N = 8;    // Weight tile: 8 × 2KB = 16KB

    // Zero output for accumulation across K tiles
    const int out_rows = m_end - m_start;
    std::memset(C + m_start * N, 0, static_cast<size_t>(out_rows) * N * sizeof(float));

    // K-outer: ensures weight tile B[n, kt:kt+TILE_K] stays hot in L1
    for (int kt = 0; kt < K; kt += TILE_K) {
        const int k_len = std::min(TILE_K, K - kt);

        // N-tile: process TILE_N weight rows at a time
        for (int nt = 0; nt < N; nt += TILE_N) {
            const int n_len = std::min(TILE_N, N - nt);

            // M-inner: all rows in this thread reuse the same weight tile
            for (int m = m_start; m < m_end; m++) {
                const float* a_tile = A + m * K + kt;
                for (int nn = nt; nn < nt + n_len; nn++) {
                    const float* b_tile = B + nn * K + kt;
                    C[m * N + nn] += simd::DotF32(a_tile, b_tile, static_cast<size_t>(k_len));
                }
            }
        }
    }
}

/**
 * @brief GQA KV Expansion
 *
 * Expands KV heads to match Q heads for Grouped Query Attention.
 * Input: [head_dim, n_head_kv, seq_len]
 * Output: [head_dim, n_head, seq_len]
 *
 * Each KV head is repeated (n_head / n_head_kv) times.
 *
 * @param K_expanded Output [head_dim, n_head, seq_len]
 * @param K Input [head_dim, n_head_kv, seq_len]
 * @param n_head Number of Q heads
 * @param n_head_kv Number of KV heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 */
inline void GQAExpandKV(float* K_expanded, const float* K, int n_head, int n_head_kv, int seq_len, int head_dim) {
    if (n_head == n_head_kv) {
        // No expansion needed (MHA case)
        std::memcpy(K_expanded, K, n_head * seq_len * head_dim * sizeof(float));
        return;
    }

    const int n_rep = n_head / n_head_kv;

    for (int s = 0; s < seq_len; s++) {
        for (int h_kv = 0; h_kv < n_head_kv; h_kv++) {
            const float* src = K + (s * n_head_kv + h_kv) * head_dim;

            // Repeat this KV head n_rep times
            for (int r = 0; r < n_rep; r++) {
                const int h_q = h_kv * n_rep + r;
                float* dst = K_expanded + (s * n_head + h_q) * head_dim;
                simd::CopyF32(dst, src, head_dim);
            }
        }
    }
}

// =============================================================================
// MoE (Mixture of Experts) Routing
// =============================================================================

/**
 * @brief MoE routing result structure (separate from GGML tensors)
 */
struct MoERouteResult {
    std::vector<int> expert_ids;        // [n_tokens * n_experts_used]
    std::vector<float> expert_weights;  // [n_tokens * n_experts_used]
    int n_tokens = 0;
    int n_experts_used = 0;

    void Resize(int tokens, int experts_used) {
        n_tokens = tokens;
        n_experts_used = experts_used;
        expert_ids.resize(tokens * experts_used);
        expert_weights.resize(tokens * experts_used);
    }

    // Get expert ID for token t, expert slot s
    int GetExpertId(int t, int s) const { return expert_ids[t * n_experts_used + s]; }

    // Get expert weight for token t, expert slot s
    float GetWeight(int t, int s) const { return expert_weights[t * n_experts_used + s]; }
};

/**
 * @brief MoE TopK Routing with Softmax
 *
 * Computes expert routing from logits:
 * 1. Apply softmax to get probabilities
 * 2. Select top-k experts per token
 * 3. Renormalize weights for selected experts
 *
 * @param result Output routing result
 * @param logits Router logits [n_tokens, n_experts]
 * @param n_tokens Number of tokens
 * @param n_experts Total number of experts
 * @param n_experts_used Number of experts to select per token (top-k)
 * @param profiler_hit_fn Optional callback to record expert hits (e.g., for profiling)
 *
 * @note For NUMA optimization, integrate with CpuBackend::RecordExpertAccess():
 * @code
 *   auto& backend = densecore::GetCpuBackend();
 *   backend.InitMoEProfiler(n_experts);  // Once at startup
 *   // Option 1: Use profiler_hit_fn callback
 *   MoETopKRoute(&result, logits, n_tokens, n_experts, k,
 *                [&](int id) { backend.GetProfiler()->RecordHit(id); });
 *   // Option 2: Batch record after routing
 *   backend.RecordExpertAccess(result.expert_ids.data(), result.expert_ids.size());
 * @endcode
 */
inline void MoETopKRoute(MoERouteResult* result, const float* logits, int n_tokens, int n_experts, int n_experts_used,
                         std::function<void(int)> profiler_hit_fn = nullptr) {
    result->Resize(n_tokens, n_experts_used);

    // Pre-allocate buffers outside the loop to avoid repeated allocations
    std::vector<float> probs(n_experts);
    std::vector<std::pair<float, int>> scored_experts(n_experts);

    for (int t = 0; t < n_tokens; t++) {
        const float* token_logits = logits + t * n_experts;

        // Step 1: Softmax
        std::memcpy(probs.data(), token_logits, n_experts * sizeof(float));
        simd::SoftmaxF32(probs.data(), n_experts);

        // Step 2: TopK selection (reuse scored_experts buffer)
        for (int e = 0; e < n_experts; e++) {
            scored_experts[e] = {probs[e], e};
        }

        // Partial sort to get top-k
        std::partial_sort(scored_experts.begin(), scored_experts.begin() + n_experts_used, scored_experts.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Step 3: Store results and renormalize weights
        float weight_sum = 0.0f;
        for (int k = 0; k < n_experts_used; k++) {
            int expert_id = scored_experts[k].second;
            result->expert_ids[t * n_experts_used + k] = expert_id;
            result->expert_weights[t * n_experts_used + k] = scored_experts[k].first;
            weight_sum += scored_experts[k].first;

            // Record hit for profiling (if callback provided)
            if (profiler_hit_fn) {
                profiler_hit_fn(expert_id);
            }
        }

        // Renormalize weights to sum to 1
        if (weight_sum > 1e-8f) {
            float inv_sum = 1.0f / weight_sum;
            for (int k = 0; k < n_experts_used; k++) {
                result->expert_weights[t * n_experts_used + k] *= inv_sum;
            }
        }
    }
}

/**
 * @brief MoE Expert FFN Configuration (separate struct, not GGML)
 */
struct MoEExpertFFN {
    const float* w1 = nullptr;  // Gate projection [intermediate_dim, n_embd]
    const float* w2 = nullptr;  // Down projection [n_embd, intermediate_dim]
    const float* w3 = nullptr;  // Up projection [intermediate_dim, n_embd]
    int n_embd = 0;
    int intermediate_dim = 0;
};

/**
 * @brief MoE Layer Configuration
 */
struct MoELayerConfig {
    std::vector<MoEExpertFFN> experts;
    const float* gate_weights = nullptr;  // Router [n_embd, n_experts]
    int n_experts = 0;
    int n_experts_used = 0;
    int n_embd = 0;
    int intermediate_dim = 0;
};

/**
 * @brief Execute MoE FFN for a single token
 *
 * Computes: output = sum_k(weight_k * FFN_expert_k(input))
 *
 * @param output Output buffer [n_embd]
 * @param input Input buffer [n_embd]
 * @param config MoE layer configuration
 * @param route Routing result for this token
 * @param token_idx Token index in route
 * @param scratch Scratch buffer [2 * intermediate_dim]
 */
inline void MoEFFNToken(float* output, const float* input, const MoELayerConfig& config, const MoERouteResult& route,
                        int token_idx, float* scratch) {
    const int n_embd = config.n_embd;
    const int intermediate = config.intermediate_dim;

    // Initialize output to zero
    std::memset(output, 0, n_embd * sizeof(float));

    float* gate_out = scratch;
    float* up_out = scratch + intermediate;
    float* down_out = scratch + 2 * intermediate;  // Temp buffer for down projection

    for (int k = 0; k < route.n_experts_used; k++) {
        const int expert_id = route.GetExpertId(token_idx, k);
        const float weight = route.GetWeight(token_idx, k);
        const MoEExpertFFN& expert = config.experts[expert_id];

        // Gate projection: gate_out = input @ w1^T (using GemvParallel with single thread)
        simd::GemvParallel(gate_out, input, expert.w1, n_embd, intermediate, 0, 1);

        // Up projection: up_out = input @ w3^T
        simd::GemvParallel(up_out, input, expert.w3, n_embd, intermediate, 0, 1);

        // SiLU × Mul (in-place on gate_out)
        simd::SiLUMul(gate_out, gate_out, up_out, intermediate);

        // Down projection: down_out = gate_out @ w2^T
        simd::GemvParallel(down_out, gate_out, expert.w2, intermediate, n_embd, 0, 1);

        // Accumulate: output += weight * down_out
        for (int i = 0; i < n_embd; i++) {
            output[i] += weight * down_out[i];
        }
    }
}

// =============================================================================
// Simple Parallel Task Scheduler
// =============================================================================

/**
 * @brief Execute function in parallel across n items
 *
 * Simple parallel for loop using std::thread.
 * Each call receives (item_idx, thread_idx, n_threads).
 *
 * @param n Total number of items to process
 * @param fn Function to execute: fn(int item_idx, int thread_idx, int n_threads)
 * @param n_threads Number of threads (0 = auto-detect)
 */
inline void ParallelFor(int n, std::function<void(int, int, int)> fn, int n_threads = 0) {
    if (n_threads <= 0) {
        n_threads = simd::GetNumCores();
        if (n_threads <= 0) n_threads = 4;
    }

    // For small workloads, run single-threaded
    if (n < n_threads * 2) {
        for (int i = 0; i < n; i++) {
            fn(i, 0, 1);
        }
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    const int items_per_thread = (n + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; t++) {
        const int start = t * items_per_thread;
        const int end = std::min(start + items_per_thread, n);

        if (start >= n) break;

        threads.emplace_back([=, &fn]() {
            for (int i = start; i < end; i++) {
                fn(i, t, n_threads);
            }
        });
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }
}

/**
 * @brief Parallel For with range-based partitioning
 *
 * Calls fn(start, end, thread_idx, n_threads) for each thread's range.
 */
inline void ParallelForRange(int n, std::function<void(int, int, int, int)> fn, int n_threads = 0) {
    if (n_threads <= 0) {
        n_threads = simd::GetNumCores();
        if (n_threads <= 0) n_threads = 4;
    }

    if (n < n_threads * 2) {
        fn(0, n, 0, 1);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    const int items_per_thread = (n + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; t++) {
        const int start = t * items_per_thread;
        const int end = std::min(start + items_per_thread, n);

        if (start >= n) break;

        threads.emplace_back([=, &fn]() { fn(start, end, t, n_threads); });
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }
}

}  // namespace native
}  // namespace densecore

#endif  // DENSECORE_SIMD_OPS_H
