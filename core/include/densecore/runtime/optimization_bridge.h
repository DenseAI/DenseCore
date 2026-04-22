/**
 * @file densecore/runtime/optimization_bridge.h
 * @brief Runtime SIMD kernel dispatch for CPU-optimal execution
 *
 * Provides a function pointer registry that gets populated at startup based
 * on detected CPU capabilities. This enables the same binary to run optimally
 * on different hardware (AVX512 vs AVX2 vs Scalar) without recompilation.
 *
 * Usage:
 *   // At engine startup
 *   densecore::OpsRegistry::Init();
 *
 *   // Use dispatched kernels
 *   densecore::Ops::RoPE(out, in, cos_sin, positions, n_tokens, head_dim,
 * rope_dim);
 */

#ifndef DENSECORE_OPTIMIZATION_BRIDGE_H
#define DENSECORE_OPTIMIZATION_BRIDGE_H

#include <cstdint>
#include <iostream>

namespace densecore {

// =============================================================================
// Function Pointer Type Definitions
// =============================================================================

/**
 * RoPE kernel signature
 * @param out Output tensor [n_tokens, head_dim]
 * @param in Input tensor [n_tokens, head_dim]
 * @param cos_sin Pre-computed [cos, sin] pairs [max_pos, head_dim]
 * @param positions Token positions array [n_tokens]
 * @param n_tokens Number of tokens
 * @param head_dim Head dimension
 * @param rope_dim Dimensions to apply RoPE (typically == head_dim)
 * @param max_seq_len Maximum sequence length (for bounds check)
 * @param ith Thread index for work partitioning
 * @param nth Total threads for work partitioning
 */
using RoPE_fn = void (*)(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens,
                         int head_dim, int rope_dim, int max_seq_len, int ith, int nth);

/**
 * INT4 GEMM kernel signature: C = A @ W^T (dequantized)
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
using GemmInt4_fn = void (*)(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                             const float* zero_points, int M, int N, int K, int group_size);

/**
 * FP32 GEMM kernel signature (Split-N tile)
 * @param C Output [M, N]
 * @param A Input activations [M, K]
 * @param B Weights [N, K] (row-major, each output channel contiguous)
 * @param M Batch dimension
 * @param N Output features
 * @param K Input features
 * @param n_start Start column (inclusive)
 * @param n_end End column (exclusive)
 */
using GemmFP32_fn = void (*)(float* C, const float* A, const float* B, int M, int N, int K, int n_start, int n_end);

/**
 * Softmax kernel signature (in-place)
 * @param data Input/output array
 * @param n Length
 */
using Softmax_fn = void (*)(float* data, size_t n);

/**
 * Dot product kernel signature
 * @param a First vector
 * @param b Second vector
 * @param n Length
 * @return Dot product result
 */
using DotF32_fn = float (*)(const float* a, const float* b, size_t n);

/**
 * Batched INT4 GEMM kernel signature: C[M,N] = A[M,K] @ W[N,K/2]^T (dequantized)
 *
 * Loads each weight tile ONCE and applies to M_BLOCK input rows simultaneously,
 * eliminating redundant INT4→FP32 unpack chains. For M=1, internally delegates
 * to the N-blocked GEMV path.
 *
 * @param C Output [M, N] row-major (must be contiguous: stride = N * sizeof(float))
 * @param A Input activations [M, K]. Elements within each row must be contiguous
 *          (stride between elements = sizeof(float)), but rows may have padding.
 * @param W_int4 Packed INT4 weights [N, K/2]
 * @param scales Per-group scales [N, num_groups]
 * @param zero_points Per-group zero points [N, num_groups]
 * @param M Batch dimension (number of tokens)
 * @param K Input features (reduction dimension)
 * @param N Output features
 * @param group_size Quantization group size
 * @param m_start Start row (inclusive)
 * @param m_end End row (exclusive)
 * @param n_start Start column (inclusive)
 * @param n_end End column (exclusive)
 * @param input_stride_bytes Byte stride between consecutive rows of A.
 *        Use K * sizeof(float) for contiguous input. Allows handling
 *        padded GGML tensors (nb[1]) without gather/scatter overhead.
 */
using GemmInt4Batched_fn = void (*)(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                    const float* zero_points, int M, int K, int N, int group_size, int m_start,
                                    int m_end, int n_start, int n_end, size_t input_stride_bytes);

// =============================================================================
// OpsRegistry: Singleton holding dispatched function pointers
// =============================================================================

/**
 * Central registry for runtime-dispatched SIMD kernels.
 *
 * Call Init() once at startup. After that, use the static Ops struct
 * to access the optimized kernels.
 */
class OpsRegistry {
public:
    // Kernel function pointers (populated by Init())
    RoPE_fn RoPE = nullptr;
    GemmInt4_fn GemmInt4 = nullptr;
    GemmInt4Batched_fn GemmInt4Batched = nullptr;
    GemmFP32_fn GemmFP32 = nullptr;
    Softmax_fn Softmax = nullptr;
    DotF32_fn DotF32 = nullptr;

    // Which ISA was selected
    const char* selected_isa = "Unknown";

    /**
     * Initialize the registry based on detected CPU capabilities.
     * Must be called once before using any kernel.
     */
    static void Init();

    /**
     * Get the singleton instance.
     */
    static OpsRegistry& Instance() {
        static OpsRegistry instance;
        return instance;
    }

    /**
     * Check if initialization has been done.
     */
    static bool IsInitialized() { return Instance().RoPE != nullptr; }

private:
    OpsRegistry() = default;
    OpsRegistry(const OpsRegistry&) = delete;
    OpsRegistry& operator=(const OpsRegistry&) = delete;
};

// =============================================================================
// Ops: Convenience namespace for direct kernel access
// =============================================================================

/**
 * Convenience interface for accessing dispatched kernels.
 *
 * Usage:
 *   densecore::Ops::RoPE(out, in, cos_sin, positions, n, dim, rope_dim, 0, 1);
 */
struct Ops {
    static void RoPE(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens,
                     int head_dim, int rope_dim, int max_seq_len, int ith = 0, int nth = 1) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.RoPE) {
            OpsRegistry::Init();
        }
        if (!reg.RoPE) return;
        reg.RoPE(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, max_seq_len, ith, nth);
    }

    static void GemmInt4(float* C, const float* A, const uint8_t* W_int4, const float* scales, const float* zero_points,
                         int M, int N, int K, int group_size) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.GemmInt4) {
            OpsRegistry::Init();
        }
        if (!reg.GemmInt4) return;
        reg.GemmInt4(C, A, W_int4, scales, zero_points, M, N, K, group_size);
    }

    static void GemmInt4Batched(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                const float* zero_points, int M, int K, int N, int group_size, int m_start, int m_end,
                                int n_start, int n_end, size_t input_stride_bytes) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.GemmInt4Batched) {
            OpsRegistry::Init();
        }
        if (!reg.GemmInt4Batched) return;
        reg.GemmInt4Batched(C, A, W_int4, scales, zero_points, M, K, N, group_size, m_start, m_end, n_start, n_end,
                            input_stride_bytes);
    }

    static void GemmFP32(float* C, const float* A, const float* B, int M, int N, int K, int n_start, int n_end) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.GemmFP32) {
            OpsRegistry::Init();
        }
        if (reg.GemmFP32) {
            reg.GemmFP32(C, A, B, M, N, K, n_start, n_end);
            return;
        }

        // Scalar fallback to preserve correctness if dispatch is unavailable.
        for (int m = 0; m < M; ++m) {
            const float* a_row = A + static_cast<size_t>(m) * static_cast<size_t>(K);
            float* c_row = C + static_cast<size_t>(m) * static_cast<size_t>(N);
            for (int n = n_start; n < n_end; ++n) {
                const float* b_row = B + static_cast<size_t>(n) * static_cast<size_t>(K);
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a_row[k] * b_row[k];
                }
                c_row[n] = sum;
            }
        }
    }

    static void Softmax(float* data, size_t n) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.Softmax) {
            OpsRegistry::Init();
        }
        if (reg.Softmax) {
            reg.Softmax(data, n);
        }
    }

    static float DotF32(const float* a, const float* b, size_t n) {
        auto& reg = OpsRegistry::Instance();
        if (!reg.DotF32) {
            OpsRegistry::Init();
        }
        if (reg.DotF32) {
            return reg.DotF32(a, b, n);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
};

}  // namespace densecore

#endif  // DENSECORE_OPTIMIZATION_BRIDGE_H
