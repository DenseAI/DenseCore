/**
 * @file simd_platform.h
 * @brief Platform abstraction for SIMD intrinsics
 *
 * Provides a portable way to use x86 AVX/AVX2/AVX-512 intrinsics on other
 * architectures via SIMDe (SIMD Everywhere).
 */

#ifndef DENSECORE_SIMD_PLATFORM_H
#define DENSECORE_SIMD_PLATFORM_H

// If we are on x86_64, use native intrinsics directly
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#else
// On non-x86 (e.g., ARM64), use SIMDe to emulate x86 intrinsics
// Define ENABLE_NATIVE_ALIASES so we can use _mm512_add_ps instead of
// simde_mm512_add_ps
#define SIMDE_ENABLE_NATIVE_ALIASES

// Include specific SIMDe headers based on what we need
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <simde/x86/fma.h>
#endif

#endif  // DENSECORE_SIMD_PLATFORM_H
