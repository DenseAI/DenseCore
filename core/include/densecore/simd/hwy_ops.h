/**
 * @file hwy_ops.h
 * @brief DenseCore SIMD Operations — Public API
 *
 * Portable, high-performance SIMD kernels powered by Google Highway.
 * All functions use Highway's dynamic dispatch to auto-select the best ISA
 * (AVX-512, AVX2, NEON, SVE, SSE4, scalar) at runtime from a single source.
 *
 * This header is the **stable public API** for downstream repositories
 * (DenseBio, DenseDiffusion, DenseVLA) to consume DenseCore SIMD primitives.
 *
 * Usage:
 *   #include "densecore/simd/hwy_ops.h"
 *   densecore::hwy_kernels::SiLU_Hwy(input, output, n);
 *
 * Link against libdensecore (.so/.dylib/.dll) — no additional dependencies
 * required; Highway is statically linked into the DenseCore shared library.
 */

#ifndef DENSECORE_SIMD_HWY_OPS_H
#define DENSECORE_SIMD_HWY_OPS_H

#include "densecore/hal/macros.h"

#include <cstddef>
#include <cstdint>

namespace densecore {
namespace hwy_kernels {

// ============================================================================
// RoPE (Rotary Positional Embedding)
// ============================================================================

/**
 * @brief Apply RoPE using Highway SIMD
 *
 * Replaces ApplyRoPE_AVX512 / ApplyRoPE_AVX2 / ApplyRoPE_NEON / ApplyRoPE_Scalar.
 * cos_sin table is interleaved: [cos0, sin0, cos1, sin1, ...] per position.
 */
DENSECORE_API void ApplyRoPE_Hwy(float* out, const float* in, const float* cos_sin, const int* positions, int n_tokens,
                                 int head_dim, int rope_dim, int max_seq_len, int ith = 0, int nth = 1);

// ============================================================================
// Activation Functions
// ============================================================================

DENSECORE_API void SiLU_Hwy(const float* input, float* output, int64_t n);
DENSECORE_API void GELU_Hwy(const float* input, float* output, int64_t n);
DENSECORE_API void Softmax_Hwy(const float* input, float* output, int64_t n);
DENSECORE_API void SiLUMul_Hwy(const float* gate, const float* up, float* output, int64_t n);

// ============================================================================
// Image Preprocessing
// ============================================================================

// RGB uint8 NHWC -> float CHW + normalize
DENSECORE_API void RGBToFloat_Hwy(const uint8_t* rgb, float* rgb_chw, int width, int height, const float mean[3],
                                  const float std[3]);

// RGB uint8 NHWC -> normalized patch tokens [N, 3*P*P]
DENSECORE_API bool RGBToPatchTokensFused_Hwy(const uint8_t* rgb, float* patch_tokens, int width, int height,
                                             int patch_size, const float mean[3], const float std[3]);

// ============================================================================
// Normalization
// ============================================================================

DENSECORE_API void Add_Hwy(float* out, const float* a, const float* b, size_t n);
DENSECORE_API void RMSNorm_Hwy(const float* x, const float* weight, float* out, size_t n, float eps = 1e-5f);
DENSECORE_API void LayerNorm_Hwy(const float* x, const float* gamma, const float* beta, float* out, size_t n,
                                 float eps = 1e-5f);

/**
 * @brief Fused Add + RMSNorm using Highway SIMD
 *
 * Operation:
 *   1. x_out[i] = x[i] + residual[i]
 *   2. rms = sqrt(mean(x_out^2) + eps)
 *   3. x_out[i] = x_out[i] / rms * weight[i]
 */
DENSECORE_API void AddRMSNorm_Hwy(float* x_out, const float* x, const float* residual, const float* weight, size_t n,
                                  float eps = 1e-5f);

// ============================================================================
// GroupNorm
// ============================================================================

DENSECORE_API void GroupNormNCHW_Hwy(const float* input, const float* gamma, const float* beta, float* output,
                                     int64_t B, int64_t C, int64_t H, int64_t W, int num_groups, float eps);

DENSECORE_API void GroupNormNHWC_Hwy(const float* input, const float* gamma, const float* beta, float* output,
                                     int64_t B, int64_t C, int64_t H, int64_t W, int num_groups, float eps);

// AdaLN (Adaptive Layer Normalization)
DENSECORE_API void AdaLN_Hwy(const float* input, const float* modulation, float eps, float* output, int64_t N,
                             int64_t D);

// ============================================================================
// Attention Variants
// ============================================================================

// AttentionWithPairBias (AlphaFold-style)
// Safety: `pair_tokens` must be > 0 and represent a square grid (`n_res * n_res`).
// Invalid inputs result in zeroed `output`.
DENSECORE_API void AttentionWithPairBias_Hwy(const float* msa_input, const float* pair_data, const float* wq,
                                             const float* wk, const float* wv, const float* wo,
                                             const float* pair_bias_w, float* output, int tokens, int c_m, int c_z,
                                             int n_head, int head_dim, int pair_tokens, float scale);

// Invariant Point Attention (IPA) — AlphaFold2 structure module
DENSECORE_API void InvariantPointAttention_Hwy(const float* s, const float* pair, const float* R, const float* t,
                                               const float* qp, const float* vp, float* out, void* workspace,
                                               size_t workspace_size, int64_t B, int64_t L, int64_t D, int64_t D_pair,
                                               int num_heads, int num_query_points, int num_value_points, float scale);

// Triangular Attention (AlphaFold-style)
DENSECORE_API void TriangularAttention_Hwy(const float* pair_data, const float* qw, const float* kw, const float* vw,
                                           float* out_data, void* workspace, size_t workspace_size, int64_t B,
                                           int64_t L, int64_t D, int num_heads, float scale, bool starting);

// Triangular Attention (AlphaFold-style) with optional tile output.
// If output_is_tiled=true, out_data layout must be [B, tile_rows, tile_cols, D].
// If output_is_tiled=false, out_data layout must be [B, L, L, D] and only the tile
// region is updated when tile bounds are not the full range.
DENSECORE_API void TriangularAttentionTiled_Hwy(const float* pair_data, const float* qw, const float* kw,
                                                const float* vw, float* out_data, void* workspace,
                                                size_t workspace_size, int64_t B, int64_t L, int64_t D, int num_heads,
                                                float scale, bool starting, int64_t tile_row_start,
                                                int64_t tile_row_end, int64_t tile_col_start, int64_t tile_col_end,
                                                bool output_is_tiled);

// CrossAttention (text-to-image, multimodal)
DENSECORE_API void CrossAttention_Hwy(const float* q_data, const float* k_data, const float* v_data, float* o_data,
                                      void* workspace, size_t workspace_size, int64_t batch, int64_t seq_q,
                                      int64_t n_head, int64_t head_dim, int64_t seq_k, int64_t n_kv_head, float scale);

// PagedAttention (KV cache efficient serving)
DENSECORE_API void PagedAttention_Hwy(const float* query, const void* const* k_block_ptrs,
                                      const void* const* v_block_ptrs, int32_t cache_type, int32_t num_heads,
                                      int32_t head_dim, int32_t n_head_kv, int32_t block_table_size,
                                      int32_t context_len, int64_t head_stride_bytes, int64_t slot_stride_bytes,
                                      float scale, float* output, int32_t head_start = 0, int32_t head_end = -1,
                                      int32_t num_heads_total = -1);

// PointAttention (3D point cloud)
// Safety: if any index in `knn_indices` for a query point is outside [0, N-1],
// that query point's output is zeroed.
DENSECORE_API void PointAttention_Hwy(const float* query, const float* key, const float* value,
                                      const int32_t* knn_indices, const float* knn_dists, float* output,
                                      void* workspace, size_t workspace_size, int64_t B, int64_t N, int64_t D, int k,
                                      int num_heads, float scale, bool use_rel_pos);

// WindowAttention (Swin Transformer-style)
DENSECORE_API void WindowAttention_Hwy(const float* query, const float* key, const float* value, const float* bias,
                                       float* output, void* workspace, size_t workspace_size, int64_t B, int64_t H,
                                       int64_t W, int64_t C, int window_size, int shift_size, int num_heads,
                                       float scale);

// ============================================================================
// FP32 GEMM
// ============================================================================

/**
 * @brief FP32 GEMM tile for Split-N parallelism
 *
 * Computes C[:, n_start:n_end) = A[M,K] x B[N,K]^T for all M rows.
 * Intended for small-batch decode/prefill where work is split over N.
 */
DENSECORE_API void GemmFP32_Hwy(float* C, const float* A, const float* B, int M, int N, int K, int n_start, int n_end);

// ============================================================================
// INT4 GEMV
// ============================================================================

DENSECORE_API void GemvInt4_Hwy(float* output, const float* input, const uint8_t* weights, const float* scales,
                                const float* zeros, int K, int N, int group_size, int n_start, int n_end);

/**
 * @brief Batched INT4 GEMM with M-blocking for weight reuse
 *
 * For M>1, loads each weight tile once and applies to M_BLOCK=4 input rows.
 * For M=1, delegates to N-blocked GemvInt4_Hwy internally.
 *
 * Layout: input[M,K] × weights[N,packed_K] → output[M,N] (row-major)
 */
DENSECORE_API void GemmInt4Batched_Hwy(float* output, const float* input, const uint8_t* weights, const float* scales,
                                       const float* zeros, int M, int K, int N, int group_size, int m_start, int m_end,
                                       int n_start, int n_end, size_t input_stride_bytes);

DENSECORE_API void PrepackInt4WeightsInterleaved_Hwy(const uint8_t* src, uint8_t* dst, int K, int N, int group_size,
                                                     int block_size);

// ============================================================================
// FP8
// ============================================================================

DENSECORE_API void InitFP8LUTs_Hwy();

// Returns immutable LUTs owned by DenseCore.
DENSECORE_API const float* GetFP8E5M2LUT_Hwy();
DENSECORE_API const float* GetFP8E4M3FNLUT_Hwy();

DENSECORE_API void ConvertFP8E5M2ToFP32_Hwy(const uint8_t* input, float* output, int64_t n);
DENSECORE_API void ConvertFP8E4M3FNToFP32_Hwy(const uint8_t* input, float* output, int64_t n);

DENSECORE_API void Gemv_FP8_E5M2_Hwy(const int M, const int N, const float alpha, const void* A_fp8,
                                     const float* x_fp32, const float beta, float* y_fp32);

DENSECORE_API void Gemv_FP8_E4M3FN_Hwy(const int M, const int N, const float alpha, const void* A_fp8,
                                       const float* x_fp32, const float beta, float* y_fp32);

// ============================================================================
// SSM / Mamba2 Kernels (Hybrid Transformer-SSM models: Qwen3.5, Jamba, …)
// ============================================================================

/**
 * @brief Causal depthwise Conv1D for SSM decode (single token).
 *
 * Updates the conv ring buffer and computes one output step.
 *
 * @param conv_state  [channels * (kernel_size-1)] ring buffer (read/write)
 * @param input       [channels] new timestep input
 * @param weight      [channels * kernel_size] conv weights (ggml layout: ne[0]=ks)
 * @param output      [channels] conv output
 * @param channels    total conv channels (d_inner + 2*n_groups*d_state)
 * @param kernel_size convolution kernel size (typically 4)
 */
DENSECORE_API void SSMConv1DDecode_Hwy(float* conv_state, const float* input, const float* weight, float* output,
                                       int channels, int kernel_size);

/**
 * @brief Mamba2 selective scan — single token decode.
 *
 * Updates recurrent state in-place and produces output for one token.
 * `dt_A` and `dt_B` must be AFTER softplus.
 *
 * @param x       [d_inner]                  activated SSM input
 * @param dt_A    [n_heads]                  per-head timestep used for A discretization
 * @param dt_B    [n_heads]                  per-head timestep scaling applied to B/input update
 * @param A_log   [n_heads]                  log-space diagonal A
 * @param B       [n_groups*d_state]         input→state projection
 * @param C       [n_groups*d_state]         state→output projection
 * @param state   [n_heads*head_dim*d_state] recurrent state (read/write)
 * @param output  [d_inner]                  scan output
 */
DENSECORE_API void SSMScanDecode_Hwy(const float* x, const float* dt_A, const float* dt_B, const float* A_log,
                                     const float* B, const float* C, float* state, float* output, int n_heads,
                                     int head_dim, int d_state, int n_groups);

/**
 * @brief Mamba2 selective scan — multi-token prefill (sequential).
 *
 * Processes seq_len tokens sequentially, updating state.
 * dt_raw is BEFORE softplus (softplus applied internally).
 */
DENSECORE_API void SSMScanPrefill_Hwy(const float* x, const float* dt_raw, const float* A_log, const float* B,
                                      const float* C, float* state, float* output, int seq_len, int n_heads,
                                      int head_dim, int d_state, int n_groups);

}  // namespace hwy_kernels
}  // namespace densecore

#endif  // DENSECORE_SIMD_HWY_OPS_H
