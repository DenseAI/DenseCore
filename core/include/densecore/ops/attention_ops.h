/**
 * @file attention_ops.h
 * @brief Unified Attention Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design Philosophy
 *
 * Unified interface for all Attention variants:
 * - Self-Attention (LLM decode/prefill)
 * - Cross-Attention (Encoder-Decoder: Whisper, T5)
 * - Deformable Attention (BEVFormer, UniAD)
 * - Window Attention (Qwen-VL high-res)
 * - Temporal Attention (SORA video)
 *
 * Classified by **nature of operation**, not by domain.
 */

#ifndef DENSECORE_OPS_ATTENTION_OPS_H
#define DENSECORE_OPS_ATTENTION_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Attention Operations Interface
 *
 * Base interface for kernels implementing any Attention variant.
 * Optimal kernel selected per DeviceType via OpRegistry.
 *
 * **Thread Safety:** Execute method must be thread-safe.
 */
class AttentionOps {
public:
    virtual ~AttentionOps() = default;

    // =========================================================================
    // Self-Attention (FlashAttention CPU Optimized)
    // =========================================================================

    /**
     * @brief Self-Attention with optional GQA support
     *
     * FlashAttention-style memory efficient implementation.
     * Supports Causal masking and GQA (Grouped Query Attention).
     *
     * Time: O(N² * D) for causal, with tiling optimization
     * Memory: O(N * D) output only, no full attention matrix
     *
     * @param Q Query tensor [batch, seq_len, n_head, head_dim]
     * @param K Key tensor [batch, seq_len, n_head_kv, head_dim]
     * @param V Value tensor [batch, seq_len, n_head_kv, head_dim]
     * @param output Output tensor [batch, seq_len, n_head, head_dim]
     * @param scale Attention scale (typically 1/sqrt(head_dim))
     * @param causal Apply causal mask (autoregressive decoding)
     * @param n_head_kv KV heads for GQA (-1 = MHA, same as Q heads)
     */
    virtual void SelfAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale = 1.0f,
                               bool causal = true, int n_head_kv = -1) = 0;

    // =========================================================================
    // Cross-Attention (Encoder-Decoder)
    // =========================================================================

    /**
     * @brief Cross-Attention for Encoder-Decoder models
     *
     * Decoder Query attends to Encoder Key/Value.
     * Used in Encoder-Decoder architectures (Whisper, T5, BART).
     *
     * **Important**: encoder_k, encoder_v are fixed after encoding and can be cached.
     *
     * Time: O(Lq * Le * D)
     *
     * @param decoder_q Decoder query [batch, Lq, n_head, head_dim]
     * @param encoder_k Encoder key (cached) [batch, Le, n_head_kv, head_dim]
     * @param encoder_v Encoder value (cached) [batch, Le, n_head_kv, head_dim]
     * @param output Output [batch, Lq, n_head, head_dim]
     * @param scale Attention scale
     */
    virtual void CrossAttention(const Tensor& decoder_q, const Tensor& encoder_k, const Tensor& encoder_v,
                                Tensor* output, float scale = 1.0f) = 0;

    // =========================================================================
    // Deformable Attention (Autonomous Driving)
    // =========================================================================

    /**
     * @brief Deformable Attention with learnable sampling locations
     *
     * Core operation for Autonomous Driving models (BEVFormer, UniAD).
     * Irregular memory access patterns - Optimized via CPU ILP/Prefetch.
     *
     * Time: O(B * Q * H * P * D)
     *
     * @param query Query tensor [batch, num_queries, dim]
     * @param spatial_features Multi-level spatial features [batch, levels, H, W, dim]
     * @param sampling_offsets Normalized sampling coords [batch, Q, heads, points, 2]
     * @param attention_weights Per-point attention weights [batch, Q, heads, points]
     * @param output Output tensor [batch, num_queries, dim]
     */
    virtual void DeformableAttention(const Tensor& query, const Tensor& spatial_features,
                                     const Tensor& sampling_offsets, const Tensor& attention_weights,
                                     Tensor* output) = 0;

    // =========================================================================
    // Window Attention (High-Resolution Vision)
    // =========================================================================

    /**
     * @brief Window-based local attention for high-resolution inputs
     *
     * Local attention for high-resolution inputs (Qwen-VL, Swin Transformer).
     * O(N * W²) complexity instead of O(N²) for high-res images.
     *
     * @param input Input tensor [batch, seq_len, n_head, head_dim]
     * @param window_size Local window size
     * @param output Output tensor [batch, seq_len, n_head, head_dim]
     * @param shift_size Shifted window size (0 = no shift)
     */
    virtual void WindowAttention(const Tensor& input, int window_size, Tensor* output, int shift_size = 0) = 0;

    // =========================================================================
    // Temporal Attention (Video)
    // =========================================================================

    /**
     * @brief Frame-to-frame temporal attention for video models
     *
     * Temporal attention for video models (SORA, Video DiT).
     * Requires separate temporal KV cache management.
     *
     * @param frames Frame features [batch, n_frames, seq_len, dim]
     * @param output Output [batch, n_frames, seq_len, dim]
     * @param causal Causal mask over time axis
     */
    virtual void TemporalAttention(const Tensor& frames, Tensor* output, bool causal = true) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_ATTENTION_OPS_H
