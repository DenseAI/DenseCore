#pragma once

#include "densecore/simd/simd_ops.h"

namespace densecore::llm::matmul {

// Keep the established SIMD primitive for Gemma float projections. Without
// it, sequential F32 accumulation loses small terms across long reductions or
// cancellation; widen both operands before multiplying and round only once.
inline float Gemma4FloatProjectionDot(const float* weight, const float* input, size_t count) {
#if defined(__AVX2__) || defined(DENSECORE_ARM)
    return densecore::simd::DotF32(weight, input, count);
#else
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(weight[i]) * static_cast<double>(input[i]);
    }
    return static_cast<float>(sum);
#endif
}

}  // namespace densecore::llm::matmul
