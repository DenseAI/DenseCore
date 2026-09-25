/**
 * @file simd_ops.cpp
 * @brief Out-of-line SIMD operation definitions
 */

#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cstdint>

namespace densecore {
namespace simd {

// =============================================================================
// Activation Functions (Delegated to Highway)
// =============================================================================

void SiLUMul(float* out, const float* gate, const float* up, size_t size) {
    hwy_kernels::SiLUMul_Hwy(gate, up, out, static_cast<int64_t>(size));
}

void SiLUMulParallel(float* out, const float* gate, const float* up, size_t size, int ith, int nth) {
    // Partition work across threads
    size_t chunk = (size + nth - 1) / nth;
    size_t start = ith * chunk;
    size_t end = std::min(start + chunk, size);

    if (start >= size) return;

    SiLUMul(out + start, gate + start, up + start, end - start);
}

}  // namespace simd
}  // namespace densecore
