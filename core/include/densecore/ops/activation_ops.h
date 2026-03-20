/**
 * @file activation_ops.h
 * @brief Unified Activation Operations Interface
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Operation-Centric Design
 *
 * Unified activation functions:
 * - SiLU (Llama, Mistral FFN)
 * - GELU (GPT, BERT)
 * - Softmax (Attention)
 */

#ifndef DENSECORE_OPS_ACTIVATION_OPS_H
#define DENSECORE_OPS_ACTIVATION_OPS_H

#include "../hal/tensor.h"

namespace densecore {
namespace ops {

/**
 * @brief Unified Activation Operations Interface
 *
 * Activation functions for Transformer models.
 */
class ActivationOps {
public:
    virtual ~ActivationOps() = default;

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /**
     * @brief SiLU (Swish) Activation
     *
     * Standard activation for modern LLMs like Llama, Mistral.
     * SiLU(x) = x * sigmoid(x)
     *
     * @param input Input tensor
     * @param output Output tensor (can be in-place)
     */
    virtual void SiLU(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief GELU Activation
     *
     * Standard activation for traditional Transformers like GPT, BERT.
     * GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     *
     * @param input Input tensor
     * @param output Output tensor
     */
    virtual void GELU(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief Softmax along specified dimension
     *
     * Attention score normalization.
     * softmax(x)_i = exp(x_i) / sum(exp(x))
     *
     * @param input Input tensor
     * @param output Output tensor
     * @param dim Dimension to apply softmax (-1 = last)
     */
    virtual void Softmax(const Tensor& input, Tensor* output, int dim = -1) = 0;

    // =========================================================================
    // Fused Operations
    // =========================================================================

    /**
     * @brief Fused SiLU Gate (SwiGLU)
     *
     * SwiGLU in Llama FFN: output = SiLU(gate) * up
     * Fused branches to optimize memory access.
     *
     * @param gate Gate branch [batch, seq, dim]
     * @param up Up-projection branch [batch, seq, dim]
     * @param output Output [batch, seq, dim]
     */
    virtual void SiLUGate(const Tensor& gate, const Tensor& up, Tensor* output) = 0;
};

}  // namespace ops
}  // namespace densecore

#endif  // DENSECORE_OPS_ACTIVATION_OPS_H
