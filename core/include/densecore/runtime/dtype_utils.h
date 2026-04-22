/**
 * @file densecore/runtime/dtype_utils.h
 * @brief DType conversion utilities
 *
 * Provides conversion between GGML types and DenseCore DType enum.
 * Extracted to avoid code duplication across model_loader.cpp and inference.cpp.
 */
#ifndef DENSECORE_DTYPE_UTILS_H
#define DENSECORE_DTYPE_UTILS_H

#include "densecore/hal/tensor.h"

#include <ggml.h>

namespace densecore {

/**
 * @brief Convert GGML tensor type to DenseCore DType
 * @param type GGML type enum value
 * @return Corresponding DType, or DType::UNKNOWN if not supported
 */
inline DType GgmlTypeToDType(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32: return DType::F32;
    case GGML_TYPE_F16: return DType::F16;
    case GGML_TYPE_BF16: return DType::BF16;
    case GGML_TYPE_I8: return DType::INT8;
    default: return DType::UNKNOWN;
    }
}

}  // namespace densecore

#endif  // DENSECORE_DTYPE_UTILS_H
