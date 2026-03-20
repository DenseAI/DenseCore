/**
 * @file transformer_ops.h
 * @brief Universal Transformer Operation Interfaces
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Universal Transformer Middleware (Ops-First)
 *
 * DenseCore HAL is organized by operation type, not by domain.
 * Domains (LLM/Vision/Diffusion/Auto/etc.) are expressed as a composition of Ops.
 *
 * Current interfaces in this header:
 * - **EmbeddingOps**: PatchEmbed2D, RoPE2D
 * - **NormalizationOps**: AdaLN, GroupNorm
 * - **AttentionOps**: DeformableAttention
 * - **SpatialOps**: GridSample
 *
 * Additional Ops (Flash/Cross/Temporal attention, Patchify/Unpatchify, etc.)
 * are tracked in the ops roadmap and will be added as new interfaces or method
 * groups without domain-specific types.
 *
 * ## Vendor Implementation Guide
 *
 * Vendors implement optimized kernels by inheriting DenseCoreOp or
 * a specific operation interface:
 *
 * @code
 * class MyNPUVisionOps : public EmbeddingOps {
 * public:
 *     void PatchEmbed2D(...) override {
 *         // NPU-optimized implementation
 *     }
 * };
 * @endcode
 */

#ifndef DENSECORE_HAL_TRANSFORMER_OPS_H
#define DENSECORE_HAL_TRANSFORMER_OPS_H

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

namespace densecore {

// Note: OpParams is defined as a std::variant in operation_graph.h

// ============================================================================
// Op capability query (kernel metadata for selection)
// ============================================================================

struct OpCapabilities {
    bool supports_fp16 = false;        ///< FP16 support
    bool supports_int8 = false;        ///< INT8 quantization support
    bool supports_int4 = false;        ///< INT4 quantization support
    size_t max_batch_size = 0;         ///< Max batch size (0 = unlimited)
    size_t l2_cache_bytes = 0;         ///< L2 cache size (tiling hint)
    size_t memory_bandwidth_gbps = 0;  ///< Memory bandwidth
    int priority = 0;                  ///< Higher is preferred
};

// ============================================================================
// Base DenseCoreOp Interface (vendor-implemented)
// ============================================================================

class DenseCoreOp {
public:
    virtual ~DenseCoreOp() = default;

    virtual void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         const void* params) = 0;

    virtual bool Supports(DeviceType device) const = 0;

    virtual OpCapabilities GetCapabilities() const = 0;

    virtual bool SupportsDType(DType dtype) const {
        const OpCapabilities caps = GetCapabilities();
        switch (dtype) {
        case DType::F16:
        case DType::BF16: return caps.supports_fp16;
        case DType::INT8: return caps.supports_int8;
        case DType::INT4: return caps.supports_int4;
        default: return true;
        }
    }

    virtual bool SupportsLayout(TensorLayout /*layout*/) const { return true; }

    virtual std::tuple<int, int, int> GetOptimalTileConfig(const std::vector<int64_t>& /*input_shape*/,
                                                           DeviceType /*device*/) const {
        return {32, 32, 64};
    }
};

// ============================================================================
// Matrix Multiplication Ops Interface
// ============================================================================

/**
 * @brief Matrix multiplication operations (Linear, MLP, Projections)
 */
class MatMulOps : public DenseCoreOp {
public:
    /**
     * @brief Standard Matrix Multiplication: C = A * B
     *
     * @param A Input matrix [M, K]
     * @param B Input matrix [K, N]
     * @param C Output matrix [M, N]
     */
    virtual void MatMul(const Tensor& A, const Tensor& B, Tensor* C) = 0;

    /**
     * @brief Matrix Multiplication with Transposed B: C = A * B^T
     *
     * Optimized for weight matrices stored in row-major [N, K] format.
     *
     * @param A Input matrix [M, K]
     * @param B Input matrix [N, K] (interpreted as B^T)
     * @param C Output matrix [M, N]
     */
    virtual void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) = 0;
};

// ============================================================================
// Embedding Ops Interface
// ============================================================================

/**
 * @brief Embedding operations (token/patch/time/position)
 */
class EmbeddingOps : public DenseCoreOp {
public:
    /**
     * @brief 2D patch embedding
     */
    virtual void PatchEmbed2D(const Tensor& image,        // [B, C, H, W]
                              const Tensor& conv_weight,  // [D, C, P, P]
                              const Tensor& conv_bias,    // [D]
                              Tensor* patches             // [B, N, D]
                              ) = 0;

    /**
     * @brief 2D rotary position embedding
     */
    virtual void RoPE2D(const Tensor& input,    // [B, N, H, D]
                        const Tensor& pos_h,    // [N]
                        const Tensor& pos_w,    // [N]
                        const Tensor& cos_sin,  // [max_pos, D]
                        Tensor* output) = 0;
};

// ============================================================================
// Normalization Ops Interface
// ============================================================================

/**
 * @brief Advanced normalization operations
 */
class NormalizationOps : public DenseCoreOp {
public:
    /**
     * @brief Adaptive LayerNorm
     */
    virtual void AdaLN(const Tensor& input,  // [B, N, D]
                       const Tensor& scale,  // [B, D]
                       const Tensor& shift,  // [B, D]
                       float eps, Tensor* output) = 0;

    /**
     * @brief Adaptive LayerNorm with fused modulation
     */
    virtual void AdaLN(const Tensor& input,       // [B, N, D]
                       const Tensor& modulation,  // [B, D*2] or [B, D*6]
                       float eps, Tensor* output) {
        // Default implementation splits modulation (naive fallback)
        // Vendors should override this for performance!
        (void)input;
        (void)modulation;
        (void)eps;
        (void)output;
    }

    /**
     * @brief Group Normalization
     */
    virtual void GroupNorm(const Tensor& input,  // [B, C, H, W]
                           const Tensor& gamma,  // [C]
                           const Tensor& beta,   // [C]
                           int num_groups, float eps, Tensor* output) = 0;
};

// ============================================================================
// Attention Ops Interface
// ============================================================================

/**
 * @brief Specialized attention mechanisms
 */
class AttentionOps : public DenseCoreOp {
public:
    /**
     * @brief Deformable attention (sparse sampling)
     */
    virtual void DeformableAttention(const Tensor& query,              // [B, Q, D]
                                     const Tensor& spatial_features,   // [B, L, H, W, D]
                                     const Tensor& sampling_offsets,   // [B, Q, H, P, 2]
                                     const Tensor& attention_weights,  // [B, Q, H, P]
                                     Tensor* output                    // [B, Q, D]
                                     ) = 0;

    // Note: CrossAttention and FlashAttention might be merged here in future phases
};

// ============================================================================
// Spatial Ops Interface
// ============================================================================

/**
 * @brief Spatial transformation operations
 */
class SpatialOps : public DenseCoreOp {
public:
    /**
     * @brief Grid sampling
     */
    virtual void GridSample(const Tensor& input,  // [B, C, H_in, W_in]
                            const Tensor& grid,   // [B, H_out, W_out, 2]
                            Tensor* output        // [B, C, H_out, W_out]
                            ) = 0;
};

// ============================================================================
// MoE Ops Interface
// ============================================================================

/**
 * @brief Mixture of Experts operations (Mixtral, DeepSeek, Grok)
 *
 * MoE layers route each token to a subset of expert networks.
 * This enables sparse computation with large model capacity.
 */
class MoEOps : public DenseCoreOp {
public:
    /**
     * @brief TopK gating - select top-k experts per token
     *
     * @param hidden_states Input token representations [B, D]
     * @param gate_weights Router weight matrix [D, num_experts]
     * @param top_k Number of experts to select per token
     * @param expert_indices Output expert indices [B, top_k]
     * @param expert_weights Output routing weights [B, top_k]
     */
    virtual void Gating(const Tensor& hidden_states,  // [B, D]
                        const Tensor& gate_weights,   // [D, num_experts]
                        int top_k,
                        Tensor* expert_indices,  // [B, top_k] int32
                        Tensor* expert_weights   // [B, top_k] float
                        ) = 0;

    /**
     * @brief Scatter tokens to expert-grouped layout
     *
     * @param input Token representations [B, D]
     * @param expert_indices Expert assignments [B, top_k]
     * @param packed_output Reordered tokens grouped by expert [total_assignments, D]
     */
    virtual void Scatter(const Tensor& input,           // [B, D]
                         const Tensor& expert_indices,  // [B, top_k]
                         Tensor* packed_output          // [total_assignments, D]
                         ) = 0;

    /**
     * @brief Gather expert outputs back to original token order
     *
     * @param packed_output Expert outputs [total_assignments, D]
     * @param expert_indices Expert assignments [B, top_k]
     * @param expert_weights Routing weights [B, top_k]
     * @param output Combined output [B, D] (weighted sum)
     */
    virtual void Gather(const Tensor& packed_output,   // [total_assignments, D]
                        const Tensor& expert_indices,  // [B, top_k]
                        const Tensor& expert_weights,  // [B, top_k]
                        Tensor* output                 // [B, D]
                        ) = 0;
};

// ============================================================================
// Activation Ops Interface
// ============================================================================

/**
 * @brief Activation function operations (SiLU, GELU, Softmax, ReLU)
 *
 * These ops are used across all transformer domains:
 * - LLM: SiLU (LLaMA FFN), GELU (GPT)
 * - Vision: GELU (ViT), ReLU (ResNet backbone)
 * - Diffusion: SiLU (DiT), GELU (SD3)
 *
 * By separating from CpuBackend, vendors can provide optimized implementations
 * for their hardware (NPU, DSP, AMX, etc.)
 */
class ActivationOps : public DenseCoreOp {
public:
    /**
     * @brief SiLU (Swish) activation: x * sigmoid(x)
     *
     * Used in LLaMA, Mistral, Qwen FFN layers.
     * Can be fused with MLP projections for memory bandwidth savings.
     */
    virtual void SiLU(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief GELU activation (approximated tanh version)
     *
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     * Used in GPT, BERT, ViT models.
     */
    virtual void GELU(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief Softmax along last dimension
     *
     * Used in attention score normalization.
     * Numerically stable implementation with max subtraction.
     */
    virtual void Softmax(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief ReLU activation: max(0, x)
     *
     * Used in classic CNNs and some MLP layers.
     */
    virtual void ReLU(const Tensor& input, Tensor* output) {
        // Default implementation (vendors can override)
        (void)input;
        (void)output;
    }

    /**
     * @brief Fused SiLU × Mul for SwiGLU FFN
     *
     * Computes: output = SiLU(gate) * up
     * Saves memory by avoiding intermediate allocation.
     */
    virtual void SiLUMul(const Tensor& gate, const Tensor& up, Tensor* output) {
        // Default fallback - vendors should override for performance
        (void)gate;
        (void)up;
        (void)output;
    }
};

// ============================================================================
// Audio Ops Interface
// ============================================================================

/**
 * @brief Audio processing operations (Whisper)
 *
 * Audio frontend operations for speech recognition models.
 */
class AudioOps : public DenseCoreOp {
public:
    /**
     * @brief Compute Mel spectrogram from raw audio
     *
     * @param waveform Raw audio samples [num_samples]
     * @param n_fft FFT window size
     * @param hop_length Hop between frames
     * @param n_mels Number of mel filterbanks
     * @param output Mel spectrogram [n_mels, num_frames]
     */
    virtual void MelSpectrogram(const Tensor& waveform,  // [num_samples]
                                int n_fft, int hop_length, int n_mels, int sample_rate,
                                Tensor* output  // [n_mels, num_frames]
                                ) = 0;
};

}  // namespace densecore

#endif  // DENSECORE_HAL_TRANSFORMER_OPS_H
