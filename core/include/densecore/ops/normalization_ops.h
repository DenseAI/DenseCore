/**
 * @file normalization_ops.h
 * @brief Unified Normalization Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * 모든 Normalization 변형을 하나의 인터페이스로 통합:
 * - RMSNorm (LLM: Llama, Mistral)
 * - LayerNorm (GPT, BERT)
 * - GroupNorm (Diffusion: DiT, Flux)
 * - AdaLN (Diffusion timestep conditioning)
 */

#ifndef DENSECORE_OPS_NORMALIZATION_OPS_H
#define DENSECORE_OPS_NORMALIZATION_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Normalization Operations Interface
 *
 * LLM, Vision, Diffusion 모델에서 사용하는 모든 정규화 연산.
 */
class NormalizationOps {
public:
    virtual ~NormalizationOps() = default;

    // =========================================================================
    // RMSNorm (LLM Standard)
    // =========================================================================

    /**
     * @brief Root Mean Square Layer Normalization
     *
     * Llama, Mistral 등 현대 LLM의 표준 정규화.
     * LayerNorm보다 계산량이 적음 (mean 계산 불필요).
     *
     * output = (input / RMS(input)) * weight
     * RMS(x) = sqrt(mean(x²) + eps)
     *
     * Time: O(N * D)
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param weight Scale weights [dim]
     * @param output Output tensor [batch, seq_len, dim]
     * @param eps Epsilon for numerical stability
     */
    virtual void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps = 1e-5f) = 0;

    /**
     * @brief Fused Add + RMSNorm
     *
     * Residual connection과 RMSNorm을 융합하여 메모리 대역폭 절감.
     * output = RMSNorm(input + residual, weight)
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param residual Residual tensor [batch, seq_len, dim]
     * @param weight Scale weights [dim]
     * @param output Output tensor [batch, seq_len, dim]
     * @param eps Epsilon
     */
    virtual void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                            float eps = 1e-5f) = 0;

    // =========================================================================
    // LayerNorm (Classic)
    // =========================================================================

    /**
     * @brief Standard Layer Normalization
     *
     * GPT, BERT 등 전통적 Transformer의 정규화.
     * output = (input - mean) / sqrt(var + eps) * gamma + beta
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param gamma Scale [dim]
     * @param beta Shift [dim]
     * @param output Output tensor [batch, seq_len, dim]
     * @param eps Epsilon
     */
    virtual void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output,
                           float eps = 1e-5f) = 0;

    // =========================================================================
    // GroupNorm (Diffusion)
    // =========================================================================

    /**
     * @brief Group Normalization
     *
     * 채널을 그룹으로 나누어 정규화. 배치 크기에 독립적.
     * Diffusion 모델(DiT, Flux, U-Net)에서 표준.
     *
     * @param input Input tensor [batch, channels, height, width]
     * @param gamma Scale [channels]
     * @param beta Shift [channels]
     * @param num_groups Number of channel groups
     * @param output Output tensor [batch, channels, height, width]
     * @param eps Epsilon
     */
    virtual void GroupNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, int num_groups, Tensor* output,
                           float eps = 1e-5f) = 0;

    // =========================================================================
    // AdaLN (Adaptive LayerNorm for Diffusion)
    // =========================================================================

    /**
     * @brief Adaptive Layer Normalization
     *
     * Timestep/class conditioning을 통한 동적 정규화.
     * DiT, Flux의 핵심 연산.
     *
     * output = LayerNorm(input) * (1 + scale) + shift
     * scale, shift는 timestep MLP에서 생성.
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param scale Conditioning scale [batch, dim]
     * @param shift Conditioning shift [batch, dim]
     * @param output Output tensor [batch, seq_len, dim]
     * @param eps Epsilon
     */
    virtual void AdaLN(const Tensor& input, const Tensor& scale, const Tensor& shift, Tensor* output,
                       float eps = 1e-5f) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_NORMALIZATION_OPS_H
