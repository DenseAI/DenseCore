/**
 * @file tensor_view.h
 * @brief Tensor view with explicit byte strides for hardware kernels
 *
 * Designed for AMX and other accelerators that require explicit memory
 * layout information (e.g., row byte width) rather than just element
 * dimensions.
 */

#ifndef DENSECORE_TENSOR_VIEW_H
#define DENSECORE_TENSOR_VIEW_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "densecore/hal/tensor.h"

namespace densecore {

/**
 * @brief Tensor descriptor with explicit byte strides
 *
 * Unlike the standard Tensor struct (which uses element strides), TensorView
 * stores strides in bytes. This is critical for tiling hardware like
 * Intel AMX where the "stride" register expects bytes between rows.
 */
struct TensorView {
    void* data = nullptr;                      ///< Pointer to data
    std::array<size_t, 4> shape;               ///< Dimensions [d0, d1, d2, d3]
    std::array<size_t, 4> strides;             ///< Strides in BYTES for each dimension
    int ndim = 0;                              ///< Number of dimensions
    DType dtype = DType::F32;                  ///< Data type
    DeviceType device_type = DeviceType::CPU;  ///< Device location (enum)
    class Device* device = nullptr;            ///< HAL Device handle
    class Allocator* allocator = nullptr;      ///< HAL Allocator handle

    TensorView() : shape{0}, strides{0} {}

    // ===========================================================================
    // AMX / Hardware Helpers
    // ===========================================================================

    /**
     * @brief Get row byte width (stride between rows)
     * Essential for AMX tileloadd instruction.
     * Assumes dimension 0 is the row dimension (e.g. M in MxN matrix).
     */
    size_t RowByteWidth() const {
        // For 2D [Rows, Cols], strides[0] is the bytes to skip to reach next row
        return strides[0];
    }

    // ===========================================================================
    // Factory Methods
    // ===========================================================================

    /**
     * @brief Create a 2D view with explicit row byte stride
     *
     * @param data Pointer to data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param row_byte_stride Bytes between start of row i and row i+1
     * @param dtype Data type
     */
    static TensorView Make2D(void* data, size_t rows, size_t cols, size_t row_byte_stride, DType dtype = DType::F32) {
        TensorView t;
        t.data = data;
        t.shape = {rows, cols, 0, 0};
        // stride[0] = bytes to next row
        // stride[1] = bytes to next col (element size)
        t.strides = {row_byte_stride, DTypeSizeBytes(dtype), 0, 0};
        t.ndim = 2;
        t.dtype = dtype;
        return t;
    }

    /**
     * @brief Create a view from contiguous memory
     */
    static TensorView FromContiguous(void* data, size_t rows, size_t cols, DType dtype = DType::F32) {
        size_t elem_size = DTypeSizeBytes(dtype);
        // For packed types like INT4 (0.5 bytes), we need special handling if not
        // byte-aligned? But strides are usually bytes. Current DTypeSizeBytes
        // returns 0 for INT4 in tensor.h? Let me check. Yes, in tensor.h
        // DTypeSizeBytes(INT4) returns 0.

        if (dtype == DType::INT4) {
            // Special handling for INT4: 2 elements per byte
            // Byte stride between rows = (cols + 1) / 2
            return Make2D(data, rows, cols, (cols + 1) / 2, dtype);
        }

        return Make2D(data, rows, cols, cols * elem_size, dtype);
    }
};

}  // namespace densecore

#endif  // DENSECORE_TENSOR_VIEW_H
