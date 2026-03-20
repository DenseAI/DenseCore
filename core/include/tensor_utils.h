#ifndef DENSECORE_TENSOR_UTILS_H
#define DENSECORE_TENSOR_UTILS_H

#include <ggml.h>

#include <vector>

namespace densecore {
namespace TensorUtils {

// Axis constants for clarity in slicing operations
constexpr int AXIS_ROWS = 0;  // Slice along first dimension (rows)
constexpr int AXIS_COLS = 1;  // Slice along second dimension (columns)

/**
 * @brief Align a value down to a multiple (for SIMD compatibility).
 * @param val Value to align
 * @param multiple Alignment boundary (e.g., 32 for AVX2)
 * @return Aligned value (<= val)
 */
int AlignToMultiple(int val, int multiple);

/**
 * @brief Create a new tensor containing only selected indices from the source.
 *
 * This function physically copies data from `src` to a new tensor, selecting
 * only the specified indices along the given dimension.
 *
 * @param ctx     GGML context for new tensor allocation
 * @param src     Source tensor to slice (must be F32 or F16)
 * @param indices Sorted indices to keep (0-indexed)
 * @param dim     Dimension to slice along:
 *                - AXIS_ROWS (0): Slice rows (e.g., output dimension of weight)
 *                - AXIS_COLS (1): Slice columns (e.g., input dimension of weight)
 *
 * @return Pointer to new sliced tensor, or nullptr on error.
 *
 * @note Only supports GGML_TYPE_F32 and GGML_TYPE_F16. Returns nullptr for
 *       quantized types (pruning should happen before quantization).
 * @note Indices should be sorted in ascending order for optimal performance.
 * @note The original tensor remains in memory; caller is responsible for
 *       context lifecycle management.
 *
 * Example:
 *   // Slice rows 0, 2, 4 from a [8, 1024] tensor -> [3, 1024] tensor
 *   auto sliced = SliceTensor(ctx, weight, {0, 2, 4}, AXIS_ROWS);
 */
struct ggml_tensor* SliceTensor(struct ggml_context* ctx, struct ggml_tensor* src, const std::vector<int>& indices,
                                int dim);

/**
 * @brief Slice a 1D tensor (e.g., layer norms, biases).
 *
 * Convenience wrapper for slicing 1D tensors along their only dimension.
 *
 * @param ctx     GGML context for new tensor allocation
 * @param src     Source 1D tensor
 * @param indices Indices to keep
 * @return Pointer to new sliced 1D tensor, or nullptr on error.
 */
struct ggml_tensor* Slice1DTensor(struct ggml_context* ctx, struct ggml_tensor* src, const std::vector<int>& indices);

}  // namespace TensorUtils
}  // namespace densecore

#endif  // DENSECORE_TENSOR_UTILS_H
