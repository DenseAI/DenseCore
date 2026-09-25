/**
 * @file linear_ops.h
 * @brief Unified Linear/Matrix Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * Unified linear operations:
 * - MatMul (Standard Matrix Multiplication)
 * - GemmInt4 (INT4 Quantized GEMV/GEMM)
 * - Fused QKV Projection
 */

#ifndef DENSECORE_OPS_LINEAR_OPS_H
#define DENSECORE_OPS_LINEAR_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Linear Operations Interface
 *
 * Matrix operations and quantized linear layers.
 */
class LinearOps {
public:
    virtual ~LinearOps() = default;

    // =========================================================================
    // Standard Matrix Multiplication
    // =========================================================================

    /**
     * @brief Standard Matrix Multiplication
     *
     * C = A @ B (fp32 or bf16/fp16)
     *
     * Time: O(M * N * K)
     *
     * @param A Input matrix [M, K]
     * @param B Weight matrix [K, N]
     * @param C Output matrix [M, N]
     */
    virtual void MatMul(const Tensor& A, const Tensor& B, Tensor* C) = 0;

    /**
     * @brief Matrix Multiplication with B transposed
     *
     * C = A @ B^T
     * Used for weight layout optimization.
     *
     * @param A Input matrix [M, K]
     * @param B Weight matrix [N, K] (stored transposed)
     * @param C Output matrix [M, N]
     */
    virtual void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) = 0;

    // =========================================================================
    // Quantized Operations
    // =========================================================================

    /**
     * @brief INT4 Quantized GEMV/GEMM
     *
     * Core operation for LLM Decode. Quantization is essential as it is memory-bound.
     *
     * output = input @ dequantize(weights, scales, zeros)
     *
     * @param input Input tensor [batch, K]
     * @param weights Packed INT4 weights [K/2, N]
     * @param scales Quantization scales [num_groups, N]
     * @param zeros Quantization zero points [num_groups, N]
     * @param output Output tensor [batch, N]
     * @param group_size Quantization group size (32, 64, 128)
     */
    virtual void GemmInt4(const Tensor& input, const Tensor& weights, const Tensor& scales, const Tensor& zeros,
                          Tensor* output, int group_size = 128) = 0;

    // =========================================================================
    // Fused Projections
    // =========================================================================

    /**
     * @brief Fused Q, K, V Projection
     *
     * Compute Q, K, V in one pass from a single weight matrix.
     * Minimizes memory access.
     *
     * [Q, K, V] = input @ qkv_weight
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param qkv_weight Combined QKV weight [dim, 3 * dim]
     * @param Q Query output [batch, seq_len, n_head, head_dim]
     * @param K Key output [batch, seq_len, n_head_kv, head_dim]
     * @param V Value output [batch, seq_len, n_head_kv, head_dim]
     */
    virtual void FusedQKVProj(const Tensor& input, const Tensor& qkv_weight, Tensor* Q, Tensor* K, Tensor* V) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_LINEAR_OPS_H
