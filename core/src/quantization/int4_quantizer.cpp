/**
 * @file int4_quantizer.cpp
 * @brief Implementation of custom INT4 block-wise quantizer
 */

#include "int4_quantizer.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "densecore/simd/simd_ops.h"

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc
#else
#include <cstdlib>  // posix_memalign
#endif

namespace densecore {

// ============================================================================
// Constructor
// ============================================================================

INT4Quantizer::INT4Quantizer(const QuantConfig& config) : Quantizer(config) {
    group_size_ = config.block_size;

    // Validate group size
    if (group_size_ != 32 && group_size_ != 64 && group_size_ != 128) {
        std::cerr << "[INT4Quantizer] Warning: Unusual group_size=" << group_size_ << ". Recommended: 32, 64, or 128"
                  << std::endl;
    }

    // Ensure group_size is even (required for packing)
    if (group_size_ % 2 != 0) {
        throw std::runtime_error("[INT4Quantizer] group_size must be even for INT4 packing");
    }

    std::cout << "[INT4Quantizer] Initialized with group_size=" << group_size_ << std::endl;
}

// ============================================================================
// Memory Alignment Utilities
// ============================================================================

void* INT4Quantizer::AlignedAlloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void INT4Quantizer::AlignedFree(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// Memory Cleanup Functions
// ============================================================================

void INT4Quantizer::FreeINT4Data(struct ggml_tensor* tensor) {
    if (!tensor || !tensor->extra) {
        return;
    }

    TensorInt4* int4_data = static_cast<TensorInt4*>(tensor->extra);

    // Free aligned weight data buffer
    if (int4_data->q_data) {
        AlignedFree(int4_data->q_data);
        int4_data->q_data = nullptr;
    }

    // Free scales array
    if (int4_data->scales) {
        delete[] int4_data->scales;
        int4_data->scales = nullptr;
    }

    // Free zero_points array
    if (int4_data->zero_points) {
        delete[] int4_data->zero_points;
        int4_data->zero_points = nullptr;
    }

    // Free the TensorInt4 struct itself
    delete int4_data;
    tensor->extra = nullptr;
}

bool INT4Quantizer::IsINT4Quantized(const struct ggml_tensor* tensor) {
    // Note: This is a best-effort check. The caller is responsible for
    // ensuring tensor->extra actually points to TensorInt4 data.
    // A more robust solution would use a magic number or type tag.
    return tensor && tensor->extra != nullptr;
}

// ============================================================================
// FP32 Extraction from GGML Tensor
// ============================================================================

int64_t INT4Quantizer::ExtractFP32(struct ggml_tensor* tensor, float* output) {
    const int64_t nelements = ggml_nelements(tensor);

    if (tensor->type == GGML_TYPE_F32) {
        // Direct copy
        std::memcpy(output, tensor->data, nelements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        // Convert FP16 to FP32
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(tensor->data), output, nelements);
    } else {
        std::cerr << "[INT4Quantizer] Unsupported tensor type: " << tensor->type << std::endl;
        return 0;
    }

    return nelements;
}

// ============================================================================
// Block-wise Quantization
// ============================================================================

void INT4Quantizer::QuantizeBlock(const float* weights, int block_size, float& scale, float& zero_point,
                                  uint8_t* output) {
    // Find min and max values in the block
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int i = 0; i < block_size; ++i) {
        min_val = std::min(min_val, weights[i]);
        max_val = std::max(max_val, weights[i]);
    }

    // Compute scale and zero-point for asymmetric quantization
    // Map [min_val, max_val] to [0, 15] (4-bit unsigned range)
    // Then shift to [-8, 7] (4-bit signed range)
    const float range = max_val - min_val;
    const float epsilon = 1e-8f;

    if (range < epsilon) {
        // All values are the same, use zero scale
        scale = 1.0f;
        zero_point = 0.0f;
        std::memset(output, 0, block_size / 2);
        return;
    }

    // Scale: map range to [0, 15]
    scale = range / 15.0f;

    // Zero-point: where does 0 map to in quantized space?
    zero_point = std::round(-min_val / scale);

    // Clamp zero_point to valid range [0, 15]
    zero_point = std::max(0.0f, std::min(15.0f, zero_point));

    // Quantize and pack weights
    for (int i = 0; i < block_size; i += 2) {
        // Quantize two consecutive weights
        float w0 = weights[i];
        float w1 = (i + 1 < block_size) ? weights[i + 1] : 0.0f;

        // Quantize to [0, 15] range
        int q0 = static_cast<int>(std::round(w0 / scale + zero_point));
        int q1 = static_cast<int>(std::round(w1 / scale + zero_point));

        // Clamp to [0, 15]
        q0 = std::max(0, std::min(15, q0));
        q1 = std::max(0, std::min(15, q1));

        // Shift to [-8, 7] for signed 4-bit representation
        int8_t qs0 = static_cast<int8_t>(q0 - 8);
        int8_t qs1 = static_cast<int8_t>(q1 - 8);

        // Pack into single byte
        output[i / 2] = PackInt4(qs0, qs1);
    }

    // Adjust zero_point for signed representation
    zero_point -= 8.0f;
}

// ============================================================================
// Main Quantization Pipeline
// ============================================================================

void INT4Quantizer::QuantizeWeight(struct ggml_tensor* tensor) {
    if (!ShouldQuantize(tensor->name)) {
        std::cout << "[INT4Quantizer] Skipping: " << tensor->name << std::endl;
        return;
    }

    // Only quantize FP32 or FP16 tensors
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        std::cout << "[INT4Quantizer] Skipping non-FP tensor: " << tensor->name << " (type=" << tensor->type << ")"
                  << std::endl;
        return;
    }

    const int64_t nelements = ggml_nelements(tensor);

    // Get tensor dimensions
    // For 2D weight tensors: [K, N] where K is inner dim, N is output dim
    const int64_t K = tensor->ne[0];  // cols (inner dimension)
    const int64_t N = tensor->ne[1];  // rows (output dimension)
    const bool is_2d = (ggml_n_dims(tensor) == 2);

    // Number of groups per row (for 2D) or total (for 1D)
    if (K <= 0 || N <= 0) {
        std::cerr << "[INT4Quantizer] Invalid tensor shape for " << tensor->name << " (K=" << K << ", N=" << N << ")"
                  << std::endl;
        return;
    }
    if ((K % group_size_) != 0) {
        std::cerr << "[INT4Quantizer] Skipping INT4 quantization for " << tensor->name
                  << " (K not divisible by group_size=" << group_size_ << ")" << std::endl;
        return;
    }
    if ((group_size_ & 1) != 0) {
        std::cerr << "[INT4Quantizer] Skipping INT4 quantization for " << tensor->name
                  << " (group_size must be even for packed INT4)" << std::endl;
        return;
    }

    const int groups_per_row = static_cast<int>(K / group_size_);
    const int total_blocks = is_2d ? (N * groups_per_row) : groups_per_row;

    std::cout << "[INT4Quantizer] Quantizing " << tensor->name << ": " << K << "x" << N << " (" << nelements
              << " elements), " << total_blocks << " blocks (" << groups_per_row << " per row)" << std::endl;

    // Step 1: Extract FP32 weights
    std::vector<float> fp32_weights(nelements);
    int64_t extracted = ExtractFP32(tensor, fp32_weights.data());
    if (extracted != nelements) {
        std::cerr << "[INT4Quantizer] Failed to extract FP32 weights" << std::endl;
        return;
    }

    // Step 2: Allocate quantized data structures
    // Packed weights: [N × K/2] row-major
    const int packed_k = static_cast<int>((K + 1) / 2);
    const size_t packed_size = static_cast<size_t>(N) * static_cast<size_t>(packed_k);
    std::vector<uint8_t> packed_data(packed_size);

    // Scales and zeros: [N × groups_per_row] - CRITICAL for cache efficiency
    // Access pattern in kernel: scales[n * groups_per_row + g]
    // This ensures linear access when iterating n, then g
    std::vector<float> scales(total_blocks);
    std::vector<float> zero_points(total_blocks);

    // Step 3: Quantize each row, then each group within the row
    // This produces [N, NumGroups] layout for scales/zeros
    for (int64_t row = 0; row < N; ++row) {
        const float* row_weights = fp32_weights.data() + row * K;
        uint8_t* row_packed = packed_data.data() + row * packed_k;

        for (int g = 0; g < groups_per_row; ++g) {
            const int64_t k_start = g * group_size_;
            const int64_t k_end = std::min(k_start + group_size_, K);
            const int actual_size = static_cast<int>(k_end - k_start);

            // Handle partial last group by zero-padding
            std::vector<float> block_weights(group_size_, 0.0f);
            std::memcpy(block_weights.data(), row_weights + k_start, actual_size * sizeof(float));

            // Index in [N, groups_per_row] layout
            const int meta_idx = row * groups_per_row + g;

            // Output location in packed data
            uint8_t* block_output = row_packed + g * (group_size_ / 2);

            QuantizeBlock(block_weights.data(), group_size_, scales[meta_idx], zero_points[meta_idx], block_output);
        }
    }

    // Step 4: Create TensorInt4 metadata
    TensorInt4* persistent_int4 = new TensorInt4();
    persistent_int4->group_size = group_size_;
    persistent_int4->num_blocks = total_blocks;
    persistent_int4->num_elements = nelements;
    persistent_int4->data_alignment = 64;

    // Copy tensor shape
    for (int i = 0; i < 4; ++i) {
        persistent_int4->ne[i] = tensor->ne[i];
    }

    // Step 5: Allocate persistent aligned memory for packed weights
    void* aligned_buffer = AlignedAlloc(packed_size);
    if (!aligned_buffer) {
        std::cerr << "[INT4Quantizer] Failed to allocate aligned memory" << std::endl;
        delete persistent_int4;
        return;
    }
    std::memcpy(aligned_buffer, packed_data.data(), packed_size);
    persistent_int4->q_data = aligned_buffer;

    // Allocate persistent copies of scales and zero_points (already in [N,
    // NumGroups] layout)
    persistent_int4->scales = new float[total_blocks];
    persistent_int4->zero_points = new float[total_blocks];
    std::memcpy(persistent_int4->scales, scales.data(), total_blocks * sizeof(float));
    std::memcpy(persistent_int4->zero_points, zero_points.data(), total_blocks * sizeof(float));

    // Attach to GGML tensor (store pointer in extra field)
    tensor->extra = persistent_int4;

    // Log success
    const float compression_ratio = (float)(nelements * sizeof(float)) / (float)(packed_size + total_blocks * 8);
    std::cout << "[INT4Quantizer] Quantized " << tensor->name << " successfully" << std::endl;
    std::cout << "  Original: " << (nelements * sizeof(float))
              << " bytes, Quantized: " << (packed_size + total_blocks * 8) << " bytes" << std::endl;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "  Scales/zeros layout: [N=" << N << ", NumGroups=" << groups_per_row << "] (cache-optimized)"
              << std::endl;
}

}  // namespace densecore
