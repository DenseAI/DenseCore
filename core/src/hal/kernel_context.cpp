/**
 * @file kernel_context.cpp
 * @brief Implementation of KernelContext
 *
 * Provides hardware context management for optimized kernel execution across
 * all supported platforms:
 * - x86: Intel AMX tile configuration (Sapphire Rapids+), AVX-512 scratchpad
 * - ARM64: SVE vector length configuration, NEON register management
 * - Apple Silicon: Accelerate AMX context, Metal compute pipeline state
 *
 * KernelContext abstracts platform-specific hardware state to enable
 * portable high-performance SIMD/matrix kernels.
 */


#include "densecore/hal/kernel_context.h"

#include <cstring>
#include <iostream>
#include <utility>

#if defined(__linux__) && defined(__x86_64__)
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace densecore {

// ============================================================================
// Platform-Specific: Intel AMX (x86_64 only)
// ============================================================================
// This section implements Intel AMX (Advanced Matrix Extensions) support
// for Sapphire Rapids and later CPUs. AMX provides hardware acceleration
// for matrix operations using 8 tile registers (tmm0-tmm7).
//
// Other platforms have their own matrix acceleration:
// - ARM64: SVE/SVE2 scalable vectors (Graviton 3/4, Neoverse V1/V2)
// - Apple Silicon: Accelerate AMX via BLAS (M1/M2/M3)
// ============================================================================


#if defined(__x86_64__)

// XFEATURE flags for xsave/xrstor
constexpr uint64_t XFEATURE_XTILECFG = 1ULL << 17;   // Tile configuration
constexpr uint64_t XFEATURE_XTILEDATA = 1ULL << 18;  // Tile data

// arch_prctl request codes for AMX (Linux 5.16+)
constexpr int ARCH_GET_XCOMP_PERM = 0x1022;
constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;

/**
 * @brief AMX Tile Configuration Structure (64 bytes)
 *
 * This structure must be 64-byte aligned for LDTILECFG instruction.
 * Format defined by Intel AMX specification:
 * - Byte 0: palette (must be 1 for valid config)
 * - Bytes 1-15: reserved (zero)
 * - Bytes 16-17: colsb for each tile (tmm0-tmm7)
 * - Bytes 48-55: rows for each tile (tmm0-tmm7)
 */
struct alignas(64) TileConfig {
    uint8_t palette;        // Byte 0: palette ID (must be 1)
    uint8_t start_row;      // Byte 1: start row (must be 0)
    uint8_t reserved0[14];  // Bytes 2-15: reserved (zero)
    uint16_t colsb[8];      // Bytes 16-31: columns in bytes for tmm0-tmm7
    uint16_t reserved1[8];  // Bytes 32-47: reserved (zero)
    uint8_t rows[8];        // Bytes 48-55: rows for tmm0-tmm7
    uint8_t reserved2[8];   // Bytes 56-63: reserved (zero)

    TileConfig() { std::memset(this, 0, sizeof(*this)); }
};

/**
 * @brief Request AMX permission via arch_prctl (Linux 5.16+)
 *
 * Intel AMX requires explicit permission request via arch_prctl on Linux.
 * This is a one-time operation per thread.
 *
 * @return true if AMX is available and permission granted, false otherwise
 */
static bool RequestAMXPermission() {
#if defined(__linux__)
    // Check if AMX permission is already granted
    unsigned long features = 0;
    long ret = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &features);
    if (ret == 0 && (features & XFEATURE_XTILEDATA)) {
        return true;  // Already have permission
    }

    // Request XTILEDATA permission
    ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (ret != 0) {
        // Permission denied - likely kernel < 5.16 or AMX disabled
        return false;
    }

    return true;
#else
    // Non-Linux: AMX not supported via this mechanism
    return false;
#endif
}

/**
 * @brief Configure AMX tiles using LDTILECFG instruction
 *
 * Sets up tile dimensions for BF16 GEMM operations:
 * - tmm0-tmm3: Accumulator tiles (16 rows × 64 bytes = 16×16 floats)
 * - tmm4-tmm5: A matrix tiles (rows × 64 bytes for BF16)
 * - tmm6-tmm7: B matrix tiles (16 rows × cols×2 bytes for BF16)
 *
 * @param rows Number of rows for the operation (max 16)
 * @param cols Number of columns for the operation (in elements, max 16)
 */
static void LoadTileConfig(int rows, int cols) {
    TileConfig cfg;
    cfg.palette = 1;  // Valid configuration

    // Clamp dimensions to AMX limits
    const int tile_rows = std::min(rows, 16);
    const int tile_cols = std::min(cols, 16);
    const int colsb = tile_cols * 4;       // 4 bytes per float (accumulator)
    const int colsb_bf16 = tile_cols * 2;  // 2 bytes per BF16

    // Configure accumulator tiles (tmm0-tmm3): 16×16 floats
    for (int i = 0; i < 4; ++i) {
        cfg.rows[i] = tile_rows;
        cfg.colsb[i] = colsb;
    }

    // Configure A input tiles (tmm4-tmm5): rows × K (BF16)
    cfg.rows[4] = tile_rows;
    cfg.colsb[4] = 64;  // K dimension in bytes (32 BF16 elements)
    cfg.rows[5] = tile_rows;
    cfg.colsb[5] = 64;

    // Configure B input tiles (tmm6-tmm7): K × cols (BF16)
    cfg.rows[6] = 16;  // K tiles
    cfg.colsb[6] = colsb_bf16;
    cfg.rows[7] = 16;
    cfg.colsb[7] = colsb_bf16;

    // Load tile configuration
#if defined(__AMX_TILE__)
    _tile_loadconfig(&cfg);
#else
    // Inline assembly fallback for compilers without AMX intrinsics
    asm volatile("ldtilecfg %0" ::"m"(cfg) : "memory");
#endif
}

#endif  // __x86_64__

// ============================================================================
// KernelContext Implementation
// ============================================================================

KernelContext::KernelContext(size_t scratchpad_bytes) : scratchpad_size_(scratchpad_bytes) {
    if (scratchpad_bytes > 0) {
        scratchpad_ = make_aligned<uint8_t>(scratchpad_bytes);
    }
}

KernelContext::~KernelContext() {
    // Release AMX state if configured
#if defined(__x86_64__) && defined(__AMX_TILE__)
    if (amx_configured_) {
        _tile_release();
    }
#endif
}

KernelContext::KernelContext(KernelContext&& other) noexcept
    : scratchpad_(std::move(other.scratchpad_)),
      scratchpad_size_(other.scratchpad_size_),
      amx_configured_(other.amx_configured_),
      cached_rows_(other.cached_rows_),
      cached_cols_(other.cached_cols_) {
    other.scratchpad_size_ = 0;
    other.amx_configured_ = false;
}

KernelContext& KernelContext::operator=(KernelContext&& other) noexcept {
    if (this != &other) {
        scratchpad_ = std::move(other.scratchpad_);
        scratchpad_size_ = other.scratchpad_size_;
        amx_configured_ = other.amx_configured_;
        cached_rows_ = other.cached_rows_;
        cached_cols_ = other.cached_cols_;

        other.scratchpad_size_ = 0;
        other.amx_configured_ = false;
    }
    return *this;
}

void KernelContext::ConfigureAMX(int rows, int cols) {
    // Skip if already configured with same dimensions
    if (amx_configured_ && cached_rows_ == rows && cached_cols_ == cols) {
        return;
    }

#if defined(__x86_64__)
    // Request AMX permission on first use (thread-local, one-time)
    static thread_local bool amx_permission_requested = false;
    static thread_local bool amx_available = false;

    if (!amx_permission_requested) {
        amx_permission_requested = true;
        amx_available = RequestAMXPermission();
        if (amx_available) {
            std::cerr << "[KernelContext] Intel AMX enabled" << std::endl;
        }
    }

    if (!amx_available) {
        // AMX not available - silently fall back to SIMD kernels
        return;
    }

    // Configure tiles with requested dimensions
    LoadTileConfig(rows, cols);

    cached_rows_ = rows;
    cached_cols_ = cols;
    amx_configured_ = true;
#else
    // Non-x86_64: Mark as configured but no-op
    (void)rows;
    (void)cols;
    cached_rows_ = rows;
    cached_cols_ = cols;
    amx_configured_ = true;
#endif
}

void KernelContext::ResizeScratchpad(size_t new_size) {
    if (new_size > scratchpad_size_) {
        scratchpad_ = make_aligned<uint8_t>(new_size);
        scratchpad_size_ = new_size;
    }
}

}  // namespace densecore
