/**
 * @file int4_types.h
 * @brief INT4 Block-wise Quantization Data Structures
 *
 * Custom INT4 quantization format optimized for AVX512 SIMD operations.
 * Implements simple block-based quantization as described in
 * "Efficient LLM Inference on CPUs".
 *
 * Memory Layout (group_size=128):
 *   Block: [4B scale][4B zero][64B packed_weights] = 72 bytes
 *   Alignment: 64 bytes (cache line + AVX512 optimized)
 *
 * @author DenseCore Team
 */

#ifndef DENSECORE_INT4_TYPES_H
#define DENSECORE_INT4_TYPES_H

#include <cstddef>
#include <cstdint>

namespace densecore {

/**
 * @brief INT4 quantization block structure
 *
 * Each block quantizes a group of weights using:
 * - Asymmetric quantization: q = round((w - zero) / scale)
 * - 4-bit signed range: [-8, 7]
 * - Two weights packed per byte
 */
struct INT4Block {
    float scale;       ///< Quantization scale factor
    float zero_point;  ///< Zero point for asymmetric quantization

    /**
     * Packed 4-bit weights (flexible array member)
     * Size: group_size / 2 bytes
     * Layout: Each byte contains two 4-bit values [low | high]
     */
    uint8_t data[];
};

/**
 * @brief Quantized INT4 tensor metadata
 *
 * Holds all information needed to work with a quantized INT4 tensor.
 * The quantized data is stored in a flat array of blocks, with
 * separate arrays for fast access to scales and zero points.
 */
struct TensorInt4 {
    void* q_data;        ///< Pointer to quantized blocks (64-byte aligned)
    float* scales;       ///< Per-block scales (num_blocks elements)
    float* zero_points;  ///< Per-block zero points (num_blocks elements)

    int group_size;        ///< Number of weights per block (32 or 128)
    int num_blocks;        ///< Total number of blocks
    int64_t num_elements;  ///< Total number of weights

    // Original tensor shape (for reconstruction)
    int64_t ne[4];  ///< Dimensions [cols, rows, batch, ?]

    size_t data_alignment;  ///< Alignment in bytes (should be 64 for AVX512)
};

// ============================================================================
// Packing / Unpacking Utilities
// ============================================================================

/**
 * @brief Pack two 4-bit signed values into one byte
 *
 * Layout: [low_nibble | high_nibble]
 * - Bits 0-3: low value
 * - Bits 4-7: high value
 *
 * @param low Lower 4-bit value (will be masked to 4 bits)
 * @param high Upper 4-bit value (will be masked to 4 bits)
 * @return Packed byte
 *
 * @note Input values are assumed to be in range [-8, 7].
 *       Only lower 4 bits are preserved.
 */
inline uint8_t PackInt4(int8_t low, int8_t high) {
    return static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
}

/**
 * @brief Unpack one byte into two 4-bit signed values
 *
 * Performs sign extension from 4-bit to 8-bit signed integers.
 *
 * @param packed Input byte containing two 4-bit values
 * @param low Output: lower 4-bit value (sign-extended)
 * @param high Output: upper 4-bit value (sign-extended)
 *
 * @note Sign extension: if bit 3 is set, fill upper 4 bits with 1s
 */
inline void UnpackInt4(uint8_t packed, int8_t& low, int8_t& high) {
    // Extract lower and upper nibbles
    low = static_cast<int8_t>(packed & 0x0F);
    high = static_cast<int8_t>((packed >> 4) & 0x0F);

    // Sign extend from 4-bit to 8-bit
    // If bit 3 (0x08) is set, the number is negative in 2's complement
    if (low & 0x08) {
        low |= static_cast<int8_t>(0xF0);  // Fill upper 4 bits with 1s
    }
    if (high & 0x08) {
        high |= static_cast<int8_t>(0xF0);
    }
}

/**
 * @brief Dequantize a single 4-bit value to FP32
 *
 * Formula: w = scale * (q - zero_point)
 *
 * @param q Quantized 4-bit value
 * @param scale Scale factor
 * @param zero_point Zero point
 * @return Dequantized FP32 value
 */
inline float DequantizeInt4(int8_t q, float scale, float zero_point) {
    return scale * (static_cast<float>(q) - zero_point);
}

// ============================================================================
// Block Size Helpers
// ============================================================================

/**
 * @brief Calculate the size of a packed block in bytes
 *
 * @param group_size Number of weights in the block
 * @return Size in bytes: 8 (scale + zero) + group_size/2 (packed data)
 */
inline size_t GetBlockSize(int group_size) {
    return sizeof(float) * 2 + (group_size / 2);
}

/**
 * @brief Calculate total memory needed for quantized tensor
 *
 * @param num_elements Total number of weights
 * @param group_size Block size
 * @return Total bytes needed (including alignment padding)
 */
inline size_t GetQuantizedSize(int64_t num_elements, int group_size) {
    int num_blocks = (num_elements + group_size - 1) / group_size;
    size_t block_size = GetBlockSize(group_size);

    // Add padding to ensure 64-byte alignment for each block
    size_t aligned_block_size = (block_size + 63) & ~63ULL;
    return num_blocks * aligned_block_size;
}

}  // namespace densecore

#endif  // DENSECORE_INT4_TYPES_H
