/**
 * @file adaln.h
 * @brief AdaLN (Adaptive Layer Normalization) Kernel Definitions
 *
 * This file is part of DenseCore.
 */

#ifndef DENSECORE_KERNELS_ADALN_H
#define DENSECORE_KERNELS_ADALN_H

#include "densecore/hal/tensor.h"

namespace densecore {
namespace kernels {

/**
 * @brief Optimized AdaLN Kernel
 *
 * Implements AdaLN(x, c) = (1 + scale(c)) * Norm(x) + shift(c)
 *
 * @param input Input tensor [Batch, Seq, Hidden]
 * @param modulation Modulation tensor [Batch, Hidden * 2] (Scale | Shift)
 * @param eps Epsilon for numerical stability
 * @param output Output tensor [Batch, Seq, Hidden]
 */
void AdaLN(const Tensor& input, const Tensor& modulation, float eps, Tensor* output);

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_ADALN_H
