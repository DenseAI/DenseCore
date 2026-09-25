/**
 * @file embedding_ops.h
 * @brief Unified Embedding & Positional Encoding Operations
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * Unified Embedding/Positional Encoding Operations:
 * - Token Embedding (LLM)
 * - Patch Embedding (ViT, Vision)
 * - RoPE (Llama, Mistral - 1D)
 * - RoPE2D (Qwen-VL, Vision - 2D)
 */

#ifndef DENSECORE_OPS_EMBEDDING_OPS_H
#define DENSECORE_OPS_EMBEDDING_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Embedding Operations Interface
 *
 * Token/Patch embedding and positional encoding for LLM & Vision models.
 */
class EmbeddingOps {
public:
    virtual ~EmbeddingOps() = default;

    // =========================================================================
    // Token Embedding (LLM)
    // =========================================================================

    /**
     * @brief Lookup token embeddings from vocabulary
     *
     * @param token_ids Token IDs [batch, seq_len]
     * @param weight Embedding weight [vocab_size, dim]
     * @param output Embeddings [batch, seq_len, dim]
     */
    virtual void TokenEmbed(const Tensor& token_ids, const Tensor& weight, Tensor* output) = 0;

    // =========================================================================
    // Patch Embedding (Vision)
    // =========================================================================

    /**
     * @brief 2D Patch Embedding for ViT/CLIP/LLaVA
     *
     * Convert image to patch sequence. Implemented as Conv2D(stride=patch_size).
     *
     * Image [B, C, H, W] → Patches [B, N, D]
     * N = (H/P) * (W/P)
     *
     * @param image Input image [batch, channels, height, width]
     * @param conv_weight Convolution weights [dim, channels, patch, patch]
     * @param conv_bias Convolution bias [dim]
     * @param patches Output patches [batch, num_patches, dim]
     */
    virtual void PatchEmbed2D(const Tensor& image, const Tensor& conv_weight, const Tensor& conv_bias,
                              Tensor* patches) = 0;

    /**
     * @brief 3D Patch Embedding for Video (SORA)
     *
     * Convert video to spatiotemporal patch sequence.
     *
     * Video [B, C, T, H, W] → Patches [B, N, D]
     * N = (T/Pt) * (H/Ph) * (W/Pw)
     *
     * @param video Input video [batch, channels, frames, height, width]
     * @param conv_weight 3D convolution weights
     * @param conv_bias Convolution bias
     * @param patches Output patches [batch, num_patches, dim]
     */
    virtual void PatchEmbed3D(const Tensor& video, const Tensor& conv_weight, const Tensor& conv_bias,
                              Tensor* patches) = 0;

    // =========================================================================
    // Rotary Position Embedding (RoPE)
    // =========================================================================

    /**
     * @brief 1D Rotary Position Embedding
     *
     * Position encoding for LLMs (Llama, Mistral).
     * Encodes relative position via rotary transformation.
     *
     * @param input Input tensor [batch, seq_len, n_head, head_dim]
     * @param freqs_cos Cosine frequencies [seq_len, head_dim/2]
     * @param freqs_sin Sine frequencies [seq_len, head_dim/2]
     * @param output Output tensor [batch, seq_len, n_head, head_dim]
     * @param rope_dim Dimensions to rotate (-1 = all)
     */
    virtual void RoPE(const Tensor& input, const Tensor& freqs_cos, const Tensor& freqs_sin, Tensor* output,
                      int rope_dim = -1) = 0;

    /**
     * @brief 2D Rotary Position Embedding
     *
     * 2D position encoding for Vision-Language models (Qwen-VL).
     * Encodes Height/Width positions separately.
     *
     * @param input Input tensor [batch, num_patches, n_head, head_dim]
     * @param pos_h Height positions [num_patches]
     * @param pos_w Width positions [num_patches]
     * @param cos_sin Precomputed cos/sin [max_pos, head_dim]
     * @param output Output tensor [batch, num_patches, n_head, head_dim]
     */
    virtual void RoPE2D(const Tensor& input, const Tensor& pos_h, const Tensor& pos_w, const Tensor& cos_sin,
                        Tensor* output) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_EMBEDDING_OPS_H
