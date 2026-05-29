/**
 * @file cpu_backend_internal.h
 * @brief Internal declarations shared across cpu_backend split files
 *
 * This header is NOT part of the public API. It provides shared
 * declarations, utilities, and includes for the CPU backend implementation
 * files under core/src/backend/.
 */

#ifndef DENSECORE_CPU_BACKEND_INTERNAL_H
#define DENSECORE_CPU_BACKEND_INTERNAL_H

// =============================================================================
// Standard Library Includes
// =============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// =============================================================================
// DenseCore Includes
// =============================================================================
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/flash_attention.h"
#include "densecore/backend/gemm_config.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/backend/matmul_backend.h"
#include "densecore/models/lora_storage.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/optimization_bridge.h"
#include "densecore/simd/simd_platform.h"

#include "backend/thread_pool_impl.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/typed_tensor.h"
#include "densecore/moe/moe_routing.h"
#include "ggml.h"

// =============================================================================
// Platform-Specific Includes
// =============================================================================

// For _mm_pause() / yield intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <malloc.h>  // For _aligned_malloc/_aligned_free
#else
#include <cstdlib>  // For aligned_alloc/free
#endif

// Linux-specific headers for Transparent Huge Pages (THP) support
#if defined(__linux__)
#include <sys/mman.h>  // For madvise() with MADV_HUGEPAGE
#endif

// Apple Accelerate Framework (AMX-backed BLAS on Apple Silicon)
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Linux-specific NUMA headers
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
#include <cerrno>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace densecore {
namespace internal {

// =============================================================================
// Internal Utility Functions
// =============================================================================

/**
 * @brief Fast exp approximation for activation hot paths
 */
inline float FastExp(float x) {
    if (x < -50.0f) x = -50.0f;
    if (x > 50.0f) x = 50.0f;

    constexpr float kLog2E = 1.4426950408889634f;
    const float y = x * kLog2E;
    const int32_t i = std::floor(y);
    const float f = y - static_cast<float>(i);

    const float p = 1.0f + f * (0.6960656421638072f + f * (0.224494337302845f + f * 0.07944023841053369f));

    const int32_t exp_bits = (i + 127) << 23;
    float two_i = 0.0f;
    std::memcpy(&two_i, &exp_bits, sizeof(two_i));
    return two_i * p;
}

/**
 * @brief Get LoRA weight data as F32, converting from F16 if needed
 * @param tensor GGML tensor
 * @param scratch Scratch buffer for F16->F32 conversion
 * @return Pointer to F32 data, or nullptr if invalid
 */
inline const float* GetLoRAWeightF32(const ggml_tensor* tensor, std::vector<float>& scratch) {
    if (!tensor || !tensor->data) {
        return nullptr;
    }

    const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
    if (elements == 0) {
        return nullptr;
    }

    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float*>(tensor->data);
    }

    if (tensor->type == GGML_TYPE_F16) {
        scratch.resize(elements);
        simd::ConvertF16ToF32(scratch.data(), static_cast<const ggml_fp16_t*>(tensor->data), elements);
        return scratch.data();
    }

    return nullptr;
}

inline bool ParseCpuBackendEnvBool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return default_value;
    }
    return std::strcmp(value, "0") != 0;
}

inline size_t GetMoEPrefetchBytes() {
    static const size_t bytes = []() -> size_t {
        const char* env = std::getenv("DENSECORE_MOE_PREFETCH_BYTES");
        if (!env || *env == '\0') {
            return 8192;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            return 8192;
        }
        return static_cast<size_t>(std::max<unsigned long long>(64ULL, parsed));
    }();
    return bytes;
}

inline int ParseCpuBackendEnvInt(const char* name, int default_value, int min_value) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }
    return static_cast<int>(std::max<long>(min_value, parsed));
}

inline bool IsMoELocalityOrderingEnabled() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_LOCALITY_ORDERING", true);
    return enabled;
}

inline bool IsMoENextExpertPrefetchEnabled() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_PREFETCH_NEXT_EXPERT", true);
    return enabled;
}

inline int GetMoELocalityOrderingMinActiveExperts() {
    static const int value = ParseCpuBackendEnvInt("DENSECORE_MOE_LOCALITY_ORDERING_MIN_ACTIVE_EXPERTS", 3, 2);
    return value;
}

inline int GetMoELocalityOrderingMinReuseIntersection() {
    static const int value = ParseCpuBackendEnvInt("DENSECORE_MOE_LOCALITY_ORDERING_MIN_REUSE_INTERSECTION", 1, 0);
    return value;
}

inline int GetMoEPrefetchMinCurrentExpertTokens() {
    static const int value = ParseCpuBackendEnvInt("DENSECORE_MOE_PREFETCH_MIN_CURRENT_EXPERT_TOKENS", 2, 1);
    return value;
}

inline int GetMoEPrefetchMaxActiveExperts() {
    static const int value = ParseCpuBackendEnvInt("DENSECORE_MOE_PREFETCH_MAX_ACTIVE_EXPERTS", 8, 1);
    return value;
}

inline int GetMoEPrefetchMaxThreadCount() {
    static const int value = ParseCpuBackendEnvInt("DENSECORE_MOE_PREFETCH_MAX_THREAD_COUNT", 32, 1);
    return value;
}

inline size_t GetMoEDequantCacheBytes() {
    static const size_t bytes = []() -> size_t {
        const char* env = std::getenv("DENSECORE_MOE_DEQUANT_CACHE_MB");
        if (!env || *env == '\0') {
            return 512ULL * 1024ULL * 1024ULL;
        }
        char* end = nullptr;
        const unsigned long long parsed_mb = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            return 512ULL * 1024ULL * 1024ULL;
        }
        return static_cast<size_t>(parsed_mb) * 1024ULL * 1024ULL;
    }();
    return bytes;
}

inline bool IsMoEDequantCacheEnabled() {
    return GetMoEDequantCacheBytes() > 0;
}

inline bool ShouldCacheAllActiveExperts() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_DEQUANT_CACHE_ALL_ACTIVE", false);
    return enabled;
}

}  // namespace internal
}  // namespace densecore

#endif  // DENSECORE_CPU_BACKEND_INTERNAL_H
