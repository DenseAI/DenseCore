#ifndef DENSECORE_KERNELS_CPU_FP8_H
#define DENSECORE_KERNELS_CPU_FP8_H

#include "../../include/densecore/hal/fp8_types.h"
#include "densecore/simd/hwy_ops.h"
#include <cstdint>

namespace densecore {
namespace kernels {

// Immutable lookup tables for fast FP8 dequantization.
inline const float* GetFP8E5M2LUT() {
    return hwy_kernels::GetFP8E5M2LUT_Hwy();
}

inline const float* GetFP8E4M3FNLUT() {
    return hwy_kernels::GetFP8E4M3FNLUT_Hwy();
}

/**
 * @brief Initialize the global FP8 lookup tables.
 * Should be called once at startup (e.g., in backend definition).
 */
void InitFP8LUTs();

/**
 * @brief Convert a batch of FP8 E5M2 values to FP32 using AVX2.
 * Uses a LUT-gather approach for maximum throughput.
 * 
 * @param input Pointer to array of uint8_t (casted from float8_e5m2)
 * @param output Pointer to array of float (FP32)
 * @param n Number of elements to convert
 */
void ConvertFP8E5M2ToFP32_AVX2(const uint8_t* input, float* output, int n);

/**
 * @brief Convert a batch of FP8 E4M3FN values to FP32 using AVX2.
 * Uses a LUT-gather approach for maximum throughput.
 * 
 * @param input Pointer to array of uint8_t (casted from float8_e4m3fn)
 * @param output Pointer to array of float (FP32)
 * @param n Number of elements to convert
 */
void ConvertFP8E4M3FNToFP32_AVX2(const uint8_t* input, float* output, int n);

/**
 * @brief Initialize FP8 environment (LUTs).
 * Wrapper around InitFP8LUTs for easier discovery.
 */
inline void InitFP8() {
    InitFP8LUTs();
}

/**
 * @brief Computes y = alpha * A * x + beta * y
 * A is stored in FP8 E5M2 (Row-Major)
 * 
 * @param M Rows (Batch size / Output dim)
 * @param N Columns (Hidden size / Inner dim)
 * @param alpha Scalar multiplier for A*x
 * @param A_fp8 Pointer to matrix A [M, N] in float8_e5m2
 * @param x_fp32 Pointer to vector x [N] in float32
 * @param beta Scalar multiplier for y (accumulate)
 * @param y_fp32 Pointer to output vector y [M] in float32
 */
void Gemv_FP8_E5M2_AVX2(const int M, const int N, const float alpha, const void* A_fp8, const float* x_fp32,
                        const float beta, float* y_fp32);

/**
 * @brief Computes y = alpha * A * x + beta * y
 * A is stored in FP8 E4M3FN (Row-Major)
 * 
 * @param M Rows (Batch size / Output dim)
 * @param N Columns (Hidden size / Inner dim)
 * @param alpha Scalar multiplier for A*x
 * @param A_fp8 Pointer to matrix A [M, N] in float8_e4m3fn
 * @param x_fp32 Pointer to vector x [N] in float32
 * @param beta Scalar multiplier for y (accumulate)
 * @param y_fp32 Pointer to output vector y [M] in float32
 */
void Gemv_FP8_E4M3FN_AVX2(const int M, const int N, const float alpha, const void* A_fp8, const float* x_fp32,
                          const float beta, float* y_fp32);

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_CPU_FP8_H
