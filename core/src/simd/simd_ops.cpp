/**
 * @file simd_ops.cpp
 * @brief Implementation of SIMD-optimized operations for quantized types
 *
 * This file provides the ComputeDotProduct dispatcher for GGUF quantized types.
 * It uses GGML's native vec_dot kernels for zero-allocation dot products.
 *
 * OPTIMIZATION: Input is pre-quantized by the caller (inference.cpp).
 * This file performs NO allocations and NO quantization.
 */

#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

// GGML headers
extern "C" {
#include "ggml-cpu.h"  // For ggml_get_type_traits_cpu and vec_dot
#include "ggml.h"
}

#include <cmath>
#include <cstdint>

namespace densecore {
namespace simd {

// =============================================================================
// Helper: SIMD-optimized dot product for float arrays
// =============================================================================
static inline float DotProductF32(const float* a, const float* b, int n) {
    return DotF32(a, b, static_cast<size_t>(n));
}

// =============================================================================
// ComputeDotProduct - Zero-Allocation Dispatcher for GGUF Quantized Types
// =============================================================================
/**
 * @brief Compute dot product between quantized weight row and pre-quantized
 * input
 *
 * This function dispatches to GGML's native vec_dot kernels for maximum
 * performance. The input is ASSUMED to already be in the correct quantized
 * format (e.g., Q8_K for Q4_K weights, Q8_0 for Q8_0 weights).
 *
 * ZERO ALLOCATION: This function performs no memory allocation.
 * The caller (inference.cpp) is responsible for pre-quantizing the input.
 *
 * @param weight_type GGML type of the weight row (e.g., GGML_TYPE_Q4_K)
 * @param w_row Pointer to quantized weight row
 * @param input Pointer to pre-quantized input (Q8_K, Q8_0, or F32)
 * @param n Number of elements
 * @param output Pointer to output scalar (single float result)
 */
void ComputeDotProduct(int weight_type, const void* w_row, const void* input, int n, float* output) {
    if (n <= 0) {
        *output = 0.0f;
        return;
    }

    enum ggml_type gtype = static_cast<enum ggml_type>(weight_type);

    // ==========================================================================
    // CASE 1: F32 weights - Direct dot product
    // ==========================================================================
    if (gtype == GGML_TYPE_F32) {
        const float* w = static_cast<const float*>(w_row);
        const float* x = static_cast<const float*>(input);
        *output = DotProductF32(w, x, n);
        return;
    }

    // ==========================================================================
    // CASE 2: Use GGML's native vec_dot kernel
    // ==========================================================================
    // The input has been pre-quantized by the caller to the correct format
    // (obtained via ggml_get_type_traits_cpu(weight_type)->vec_dot_type).
    // We can directly call vec_dot without any allocation or conversion.
    // ==========================================================================
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(gtype);

    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        // vec_dot signature: void(int n, float *s, size_t bs,
        //                         const void *x, size_t bx,
        //                         const void *y, size_t by, int nrc)
        // - n: number of elements
        // - s: pointer to result (single float)
        // - bs: block stride for s (unused, set to 0)
        // - x: first operand (weight row)
        // - bx: block stride for x (unused, set to 0)
        // - y: second operand (pre-quantized input)
        // - by: block stride for y (unused, set to 0)
        // - nrc: number of result columns (1 for single dot product)
        type_traits_cpu->vec_dot(n, output, 0, w_row, 0, input, 0, 1);
        return;
    }

    // ==========================================================================
    // CASE 3: F16 weights - fallback (rare, F16 usually doesn't need vec_dot)
    // ==========================================================================
    if (gtype == GGML_TYPE_F16) {
        // For F16, input should be F32 (no pre-quantization needed)
        // We need to dequantize F16 weights to F32 using thread-local buffer
        // NOTE: This is a fallback path; the hot path is vec_dot above
        static thread_local std::vector<float> dequant_buffer;
        if (n > 16384) {
            *output = 0.0f;
            return;
        }
        dequant_buffer.resize(static_cast<size_t>(n));
        const ggml_fp16_t* w = static_cast<const ggml_fp16_t*>(w_row);
        const float* x = static_cast<const float*>(input);
        ggml_fp16_to_fp32_row(w, dequant_buffer.data(), static_cast<int64_t>(n));
        *output = DotProductF32(dequant_buffer.data(), x, n);
        return;
    }

    // ==========================================================================
    // Unsupported type
    // ==========================================================================
    *output = 0.0f;
}

// =============================================================================
// ComputeDotProductBatch - Batch dispatcher for multiple output rows
// =============================================================================
/**
 * @brief Compute multiple dot products for a range of output rows
 *
 * Uses native vec_dot kernels with pre-quantized input for zero-allocation
 * performance.
 *
 * @param weight_type GGML type of the weight tensor
 * @param weight Base pointer to weight tensor data
 * @param row_stride Stride in bytes between rows
 * @param input Pre-quantized input vector
 * @param n Number of elements per row
 * @param output Float output vector [k_end - k_start]
 * @param k_start First output row index (inclusive)
 * @param k_end Last output row index (exclusive)
 */
void ComputeDotProductBatch(int weight_type, const void* weight, size_t row_stride, const void* input, int n,
                            float* output, int k_start, int k_end) {
    if (k_start >= k_end || n <= 0) {
        return;
    }

    enum ggml_type gtype = static_cast<enum ggml_type>(weight_type);

    // ==========================================================================
    // FAST PATH: FP32 - Direct dot product
    // ==========================================================================
    if (gtype == GGML_TYPE_F32) {
        const char* base = static_cast<const char*>(weight);
        const float* x = static_cast<const float*>(input);
        for (int k = k_start; k < k_end; k++) {
            const float* w_row = reinterpret_cast<const float*>(base + k * row_stride);
            output[k - k_start] = DotProductF32(w_row, x, n);
        }
        return;
    }

    // ==========================================================================
    // QUANTIZED PATH: Use native vec_dot kernel
    // ==========================================================================
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(gtype);

    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const char* base = static_cast<const char*>(weight);
        for (int k = k_start; k < k_end; k++) {
            const void* w_row = base + k * row_stride;
            type_traits_cpu->vec_dot(n, &output[k - k_start], 0, w_row, 0, input, 0, 1);
        }
        return;
    }

    // ==========================================================================
    // FALLBACK: F16 (rare path)
    // ==========================================================================
    if (gtype == GGML_TYPE_F16) {
        static thread_local std::vector<float> dequant_buffer;
        if (n > 16384) {
            return;
        }
        dequant_buffer.resize(static_cast<size_t>(n));
        const char* base = static_cast<const char*>(weight);
        const float* x = static_cast<const float*>(input);
        for (int k = k_start; k < k_end; k++) {
            const ggml_fp16_t* w_row = reinterpret_cast<const ggml_fp16_t*>(base + k * row_stride);
            ggml_fp16_to_fp32_row(w_row, dequant_buffer.data(), static_cast<int64_t>(n));
            output[k - k_start] = DotProductF32(dequant_buffer.data(), x, n);
        }
        return;
    }

    // Zero output for unsupported types
    for (int k = k_start; k < k_end; k++) {
        output[k - k_start] = 0.0f;
    }
}

// =============================================================================
// GetDequantizationBufferSize - Query the maximum supported buffer size
// =============================================================================
// NOTE: With pre-quantization, this is less relevant but kept for API compat
size_t GetDequantizationBufferSize() {
    return 16384;
}

// =============================================================================
// IsTypeSupported - Check if a type is supported by ComputeDotProduct
// =============================================================================
bool IsTypeSupported(int type) {
    enum ggml_type gtype = static_cast<enum ggml_type>(type);

    // F32 and F16 are always supported
    if (gtype == GGML_TYPE_F32 || gtype == GGML_TYPE_F16) {
        return true;
    }

    // Check if GGML has vec_dot for this type
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(gtype);
    return type_traits_cpu && type_traits_cpu->vec_dot;
}


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
