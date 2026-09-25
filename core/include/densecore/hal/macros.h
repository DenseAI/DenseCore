/**
 * @file macros.h
 * @brief DenseCore API export macros for shared library builds
 */
#ifndef DENSECORE_HAL_MACROS_H
#define DENSECORE_HAL_MACROS_H

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define DENSECORE_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
#define DENSECORE_PLATFORM_APPLE 1
#elif defined(__linux__)
#define DENSECORE_PLATFORM_LINUX 1
#elif defined(__ANDROID__)
#define DENSECORE_PLATFORM_ANDROID 1
#else
#define DENSECORE_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// API Export/Import Macros
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef DENSECORE_BUILD_SHARED
// Building DenseCore as a shared library: export symbols
#define DENSECORE_API __declspec(dllexport)
#elif defined(DENSECORE_SHARED)
// Consuming DenseCore as a shared library: import symbols
#define DENSECORE_API __declspec(dllimport)
#else
// Static library build: no import/export decoration needed
#define DENSECORE_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define DENSECORE_API __attribute__((visibility("default")))
#else
#define DENSECORE_API
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#define DENSECORE_LOCAL
#else
#define DENSECORE_LOCAL __attribute__((visibility("hidden")))
#endif

// =============================================================================
// C++ Feature Detection
// =============================================================================

#if __cplusplus >= 201703L
#define DENSECORE_CPP17 1
#endif

#if __cplusplus >= 202002L
#define DENSECORE_CPP20 1
#endif

#ifdef DENSECORE_CPP17
#define DENSECORE_NODISCARD [[nodiscard]]
#else
#define DENSECORE_NODISCARD
#endif

#ifdef DENSECORE_CPP17
#define DENSECORE_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define DENSECORE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#endif

// Compiler hints
#if defined(_MSC_VER)
#define DENSECORE_ALWAYS_INLINE __forceinline
#define DENSECORE_NOINLINE __declspec(noinline)
#define DENSECORE_LIKELY(x) (x)
#define DENSECORE_UNLIKELY(x) (x)
#define DENSECORE_ALIGN(x) __declspec(align(x))
#elif defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ALWAYS_INLINE inline __attribute__((always_inline))
#define DENSECORE_NOINLINE __attribute__((noinline))
#define DENSECORE_LIKELY(x) __builtin_expect(!!(x), 1)
#define DENSECORE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define DENSECORE_ALIGN(x) __attribute__((aligned(x)))
#else
#define DENSECORE_ALWAYS_INLINE inline
#define DENSECORE_NOINLINE
#define DENSECORE_LIKELY(x) (x)
#define DENSECORE_UNLIKELY(x) (x)
#define DENSECORE_ALIGN(x)
#endif

#endif  // DENSECORE_HAL_MACROS_H
