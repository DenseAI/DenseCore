/**
 * @file timestep_embed.h
 * @brief Sinusoidal Timestep Embedding kernel for diffusion models
 */

#ifndef DENSECORE_KERNELS_TIMESTEP_EMBED_H
#define DENSECORE_KERNELS_TIMESTEP_EMBED_H

#include "densecore/hal/tensor.h"

namespace densecore {
namespace kernels {

/**
 * @brief Compute Sinusoidal Timestep Embeddings
 *
 * @param timesteps  [Batch] Input timesteps (usually float or int)
 * @param dim        Output embedding dimension (must be even)
 * @param max_period Max period for frequency calculation (default 10000)
 * @param output     [Batch, Dim] Output tensor
 */
void TimestepEmbedding(const Tensor& timesteps, int dim, int max_period, Tensor* output);

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_TIMESTEP_EMBED_H
