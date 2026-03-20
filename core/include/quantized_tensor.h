/**
 * @file quantized_tensor.h
 * @brief First-class quantization support for edge hardware inference
 *
 * This header provides `QuantizedTensorView`, a tensor descriptor that
 * carries quantization metadata alongside the data pointer. Essential for:
 *
 * **Apple ANE:** Requires INT8 tensors with per-channel or per-tensor scales
 * **Qualcomm Hexagon:** Supports INT8 and INT4 with per-block quantization
 * **CPU SIMD:** AVX-512 VNNI provides native INT8 dot products
 *
 * **Design Notes:**
 * - Extends the existing `TensorView` pattern with quantization params
 * - Zero-allocation: all pointers are non-owning
 * - Compatible with GGML quantization formats (Q4_0, Q4_K, Q8_0)
 *
 * @see accelerator_traits.h for QuantType enum definition
 */

#ifndef DENSECORE_QUANTIZED_TENSOR_H
#define DENSECORE_QUANTIZED_TENSOR_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "densecore/hal/tensor.h"

#include "accelerator_traits.h"

namespace densecore {

// ============================================================================
// Quantization Parameter Structures
// ============================================================================

/**
 * @brief Per-block quantization parameters
 *
 * Used by Q4_0, Q4_K, and other block-wise quantization schemes where
 * each block of elements shares a scale and zero-point.
 */
struct BlockQuantParams {
    const void* scales = nullptr;       ///< Scale per block [num_blocks]
    const void* zero_points = nullptr;  ///< Zero point per block [num_blocks]
    int32_t block_size = 32;            ///< Elements per quantization block
    int32_t num_blocks = 0;             ///< Total number of blocks

    /**
     * @brief Calculate number of blocks for a tensor dimension
     */
    static int32_t CalcNumBlocks(int64_t num_elements, int32_t block_size) {
        return static_cast<int32_t>((num_elements + block_size - 1) / block_size);
    }
};

/**
 * @brief Per-channel quantization parameters
 *
 * Used for activations and some weight formats where each output channel
 * has independent quantization parameters.
 */
struct ChannelQuantParams {
    const float* scales = nullptr;         ///< Scale per channel [num_channels]
    const int32_t* zero_points = nullptr;  ///< Zero point per channel [num_channels]
    int32_t num_channels = 0;
    int32_t channel_axis = -1;  ///< Axis along which to apply (-1 = last)
};

/**
 * @brief Per-tensor (scalar) quantization parameters
 *
 * Simplest quantization scheme with a single scale/zero for the entire tensor.
 * Used for activations in symmetric quantization schemes.
 */
struct TensorQuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

/**
 * @brief Union of all quantization parameter types
 */
struct QuantizationParams {
    enum class Granularity : uint8_t {
        None = 0,    ///< No quantization (FP32/FP16)
        PerTensor,   ///< Single scale/zero for entire tensor
        PerChannel,  ///< Per-output-channel scales
        PerBlock     ///< Per-block scales (GGML-style)
    };

    Granularity granularity = Granularity::None;

    union {
        TensorQuantParams tensor;
        ChannelQuantParams channel;
        BlockQuantParams block;
    };

    QuantizationParams() : tensor{} {}

    // Factory methods
    static QuantizationParams None() {
        QuantizationParams p;
        p.granularity = Granularity::None;
        return p;
    }

    static QuantizationParams PerTensor(float scale, int32_t zero_point = 0) {
        QuantizationParams p;
        p.granularity = Granularity::PerTensor;
        p.tensor.scale = scale;
        p.tensor.zero_point = zero_point;
        return p;
    }

    static QuantizationParams PerChannel(const float* scales, const int32_t* zero_points, int32_t num_channels,
                                         int32_t axis = -1) {
        QuantizationParams p;
        p.granularity = Granularity::PerChannel;
        p.channel.scales = scales;
        p.channel.zero_points = zero_points;
        p.channel.num_channels = num_channels;
        p.channel.channel_axis = axis;
        return p;
    }

    static QuantizationParams PerBlock(const void* scales, const void* zero_points, int32_t block_size,
                                       int32_t num_blocks) {
        QuantizationParams p;
        p.granularity = Granularity::PerBlock;
        p.block.scales = scales;
        p.block.zero_points = zero_points;
        p.block.block_size = block_size;
        p.block.num_blocks = num_blocks;
        return p;
    }
};

// ============================================================================
// Quantized Tensor View
// ============================================================================

/**
 * @brief Tensor descriptor with quantization metadata
 *
 * This struct extends the `Tensor` concept with quantization parameters,
 * enabling backends to perform quantized operations directly without
 * separate dequantization passes.
 *
 * **Memory Layout:**
 * For block-quantized tensors (Q4_0, Q4_K), the data layout is typically:
 * ```
 * [block_0_data][block_0_scale][block_1_data][block_1_scale]...
 * ```
 * The `quant_params.block` fields point to the scale/zero arrays.
 *
 * **Thread Safety:** Non-owning view, immutable after construction.
 */
struct QuantizedTensorView {
    void* data = nullptr;                      ///< Quantized data pointer
    std::array<int64_t, 4> shape;              ///< Logical dimensions (pre-quantization)
    std::array<int64_t, 4> stride;             ///< Strides in elements (not bytes)
    int ndim = 0;                              ///< Number of dimensions
    QuantType type = QuantType::FP32;          ///< Quantization type
    DeviceType device_type = DeviceType::CPU;  ///< Device placement (enum)
    class Device* device = nullptr;            ///< HAL Device handle
    class Allocator* allocator = nullptr;      ///< HAL Allocator handle
    QuantizationParams quant_params;           ///< Quantization parameters

    QuantizedTensorView() : shape{0, 0, 0, 0}, stride{0, 0, 0, 0} {}

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create from an existing Tensor (assumes already quantized)
     */
    static QuantizedTensorView FromTensor(const Tensor& t, QuantType type, QuantizationParams params) {
        QuantizedTensorView v;
        v.data = t.data;
        v.shape = t.shape;
        v.stride = t.stride;
        v.ndim = t.ndim;
        v.type = type;
        v.device_type = t.device_type;
        v.device = t.device;
        v.allocator = t.allocator;
        v.quant_params = params;
        return v;
    }

    /**
     * @brief Create FP32 view (no quantization)
     */
    static QuantizedTensorView MakeFP32(void* data, int64_t rows, int64_t cols,
                                        DeviceType device_type = DeviceType::CPU) {
        QuantizedTensorView v;
        v.data = data;
        v.shape = {rows, cols, 0, 0};
        v.stride = {cols, 1, 0, 0};
        v.ndim = 2;
        v.type = QuantType::FP32;
        v.device_type = device_type;
        v.quant_params = QuantizationParams::None();
        return v;
    }

    /**
     * @brief Create INT8 per-tensor quantized view
     */
    static QuantizedTensorView MakeINT8(void* data, int64_t rows, int64_t cols, float scale, int32_t zero_point = 0,
                                        DeviceType device_type = DeviceType::CPU) {
        QuantizedTensorView v;
        v.data = data;
        v.shape = {rows, cols, 0, 0};
        v.stride = {cols, 1, 0, 0};
        v.ndim = 2;
        v.type = QuantType::INT8;
        v.device_type = device_type;
        v.quant_params = QuantizationParams::PerTensor(scale, zero_point);
        return v;
    }

    /**
     * @brief Create Q4_K block-quantized view (GGML style)
     */
    static QuantizedTensorView MakeQ4K(void* data, int64_t rows, int64_t cols, const void* scales, const void* zeros,
                                       int32_t block_size = 32, DeviceType device_type = DeviceType::CPU) {
        QuantizedTensorView v;
        v.data = data;
        v.shape = {rows, cols, 0, 0};
        v.stride = {cols, 1, 0, 0};  // Logical stride
        v.ndim = 2;
        v.type = QuantType::Q4_K;
        v.device_type = device_type;

        int32_t num_blocks = BlockQuantParams::CalcNumBlocks(rows * cols, block_size);
        v.quant_params = QuantizationParams::PerBlock(scales, zeros, block_size, num_blocks);
        return v;
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Total logical elements (not accounting for packing)
     */
    int64_t NumElements() const {
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            n *= shape[i];
        }
        return n;
    }

    /**
     * @brief Size in bytes of the quantized data
     */
    size_t DataSizeBytes() const {
        int64_t n = NumElements();
        switch (type) {
        case QuantType::FP32: return static_cast<size_t>(n) * 4;
        case QuantType::FP16:
        case QuantType::BF16: return static_cast<size_t>(n) * 2;
        case QuantType::INT8: return static_cast<size_t>(n);
        case QuantType::Q4_0:
        case QuantType::Q4_K: return static_cast<size_t>((n + 1) / 2);  // 4 bits per element
        default: return 0;
        }
    }

    /**
     * @brief Check if this view requires dequantization before use
     */
    bool RequiresDequantization() const { return type != QuantType::FP32 && type != QuantType::FP16; }

    /**
     * @brief Check if view is valid
     */
    bool IsValid() const { return data != nullptr && ndim > 0 && shape[0] > 0; }
};

// =============================================================================
// UnifiedTensorRef Method Implementations
// =============================================================================
// These implementations require complete QuantizedTensorView type, so they
// are defined here after QuantizedTensorView is fully defined.

inline const void* UnifiedTensorRef::Data() const {
    if (kind == Kind::Dense) {
        return dense ? dense->data : nullptr;
    }
    if (kind == Kind::Quantized && quantized) {
        return quantized->data;
    }
    return nullptr;
}

inline DType UnifiedTensorRef::GetDType() const {
    if (kind == Kind::Dense && dense) {
        return dense->dtype;
    }
    if (kind == Kind::Quantized && quantized) {
        // Map QuantType to DType
        switch (quantized->type) {
        case QuantType::FP32: return DType::F32;
        case QuantType::FP16: return DType::F16;
        case QuantType::BF16: return DType::BF16;
        case QuantType::INT8: return DType::INT8;
        case QuantType::Q4_0:
        case QuantType::Q4_K: return DType::INT4;
        default: return DType::UNKNOWN;
        }
    }
    return DType::UNKNOWN;
}

}  // namespace densecore

#endif  // DENSECORE_QUANTIZED_TENSOR_H
