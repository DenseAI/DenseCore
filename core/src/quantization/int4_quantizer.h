/**
 * @file int4_quantizer.h
 * @brief Custom INT4 Block-wise Quantizer
 *
 * Implements weight-only INT4 quantization with simple block structure
 * optimized for AVX512 SIMD operations, as described in
 * "Efficient LLM Inference on CPUs".
 *
 * @author DenseCore Team
 */

#ifndef DENSECORE_INT4_QUANTIZER_H
#define DENSECORE_INT4_QUANTIZER_H

#include <memory>
#include <vector>

#include "quantization/int4_types.h"
#include "quantizer.h"

namespace densecore {

/**
 * @brief Custom INT4 block-wise quantizer
 *
 * Quantizes weight tensors to INT4 format using block-wise quantization.
 * Each block uses asymmetric quantization with per-block scale and zero-point.
 *
 * Features:
 * - Simple linear block structure (no hierarchical super-blocks)
 * - AVX512-optimized memory layout (64-byte aligned)
 * - Configurable block size (32 or 128 recommended)
 * - Independent from GGML quantization types
 */
class INT4Quantizer : public Quantizer {
public:
    /**
     * @brief Construct INT4 quantizer with configuration
     *
     * @param config Quantization configuration
     *               - block_size: group size (32 or 128)
     *               - format should be INT4_BLOCKWISE
     */
    explicit INT4Quantizer(const QuantConfig& config);

    /**
     * @brief Destructor
     */
    ~INT4Quantizer() override = default;

    /**
     * @brief Quantize a weight tensor to INT4 format
     *
     * Pipeline:
     * 1. Extract FP32 weights from GGML tensor
     * 2. Divide into blocks of group_size
     * 3. Compute per-block scale and zero-point
     * 4. Quantize to 4-bit signed integers [-8, 7]
     * 5. Pack into AVX512-friendly memory layout
     * 6. Attach TensorInt4 metadata to GGML tensor
     *
     * @param tensor GGML tensor to quantize (must be F32 or F16)
     *
     * @note The original GGML tensor is modified to store a pointer
     *       to the INT4 quantized data in its user data field.
     */
    void QuantizeWeight(struct ggml_tensor* tensor) override;

private:
    int group_size_;  ///< Block size (number of weights per block)

    /**
     * @brief Quantize a single block of weights
     *
     * Computes asymmetric quantization:
     * 1. Find min/max values in block
     * 2. scale = (max - min) / 15.0
     * 3. zero_point = round(-min / scale)
     * 4. q[i] = clamp(round(w[i] / scale + zero_point), 0, 15) - 8
     *
     * @param weights Input FP32 weights (block_size elements)
     * @param block_size Number of weights in this block
     * @param scale Output: computed scale factor
     * @param zero_point Output: computed zero point
     * @param output Output: packed INT4 data (block_size / 2 bytes)
     */
    void QuantizeBlock(const float* weights, int block_size, float& scale, float& zero_point, uint8_t* output);

    /**
     * @brief Convert GGML tensor to FP32 array
     *
     * Handles both F32 and F16 GGML tensors.
     *
     * @param tensor Input GGML tensor
     * @param output Output FP32 buffer (caller must allocate)
     * @return Number of elements extracted
     */
    int64_t ExtractFP32(struct ggml_tensor* tensor, float* output);

    /**
     * @brief Allocate 64-byte aligned memory
     *
     * @param size Size in bytes
     * @return Pointer to aligned memory (must be freed with AlignedFree)
     */
    static void* AlignedAlloc(size_t size);

    /**
     * @brief Free 64-byte aligned memory
     *
     * @param ptr Pointer returned by AlignedAlloc
     */
    static void AlignedFree(void* ptr);

public:
    /**
     * @brief Free INT4 quantized data attached to a tensor
     *
     * This function MUST be called when unloading a model that contains
     * INT4 quantized tensors to prevent memory leaks. The function safely
     * handles null pointers and non-quantized tensors.
     *
     * @param tensor GGML tensor with INT4 data attached via tensor->extra
     */
    DENSECORE_API static void FreeINT4Data(struct ggml_tensor* tensor);

    /**
     * @brief Check if a tensor has INT4 quantized data attached
     *
     * @param tensor GGML tensor to check
     * @return true if tensor->extra contains valid TensorInt4 data
     */
    static bool IsINT4Quantized(const struct ggml_tensor* tensor);
};

}  // namespace densecore

#endif  // DENSECORE_INT4_QUANTIZER_H
