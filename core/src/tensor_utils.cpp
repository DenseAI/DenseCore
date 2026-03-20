#include <tensor_utils.h>

#include <cstring>
#include <iostream>

namespace densecore {
namespace TensorUtils {

int AlignToMultiple(int val, int multiple) {
    if (multiple <= 0) return val;
    return (val / multiple) * multiple;
}

struct ggml_tensor* SliceTensor(struct ggml_context* ctx, struct ggml_tensor* src, const std::vector<int>& indices,
                                int dim) {
    if (!ctx || !src || indices.empty()) {
        std::cerr << "[TensorUtils] Error: Invalid arguments to SliceTensor" << std::endl;
        return nullptr;
    }

    // Only allow F32 and F16 - pruning should happen before quantization
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        std::cerr << "[TensorUtils] Error: SliceTensor only supports F32/F16. Got type " << src->type
                  << ". Prune before quantization." << std::endl;
        return nullptr;
    }

    const int64_t src_rows = src->ne[0];  // First dimension (rows)
    const int64_t src_cols = src->ne[1];  // Second dimension (columns)
    const int new_size = static_cast<int>(indices.size());

    // Validate indices
    for (int idx : indices) {
        if (dim == AXIS_ROWS && (idx < 0 || idx >= src_rows)) {
            std::cerr << "[TensorUtils] Error: Row index " << idx << " out of bounds [0, " << src_rows << ")"
                      << std::endl;
            return nullptr;
        }
        if (dim == AXIS_COLS && (idx < 0 || idx >= src_cols)) {
            std::cerr << "[TensorUtils] Error: Col index " << idx << " out of bounds [0, " << src_cols << ")"
                      << std::endl;
            return nullptr;
        }
    }

    // Calculate new tensor shape
    int64_t new_rows, new_cols;
    if (dim == AXIS_ROWS) {
        new_rows = new_size;
        new_cols = src_cols;
    } else if (dim == AXIS_COLS) {
        new_rows = src_rows;
        new_cols = new_size;
    } else {
        std::cerr << "[TensorUtils] Error: Invalid dimension " << dim << ". Use AXIS_ROWS (0) or AXIS_COLS (1)."
                  << std::endl;
        return nullptr;
    }

    // Allocate new tensor with same type
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, src->type, new_rows, new_cols);
    if (!dst) {
        std::cerr << "[TensorUtils] Error: Failed to allocate tensor [" << new_rows << ", " << new_cols << "]"
                  << std::endl;
        return nullptr;
    }

    // Copy name with "_sliced" suffix
    if (src->name[0] != '\0') {
        char new_name[GGML_MAX_NAME];
        snprintf(new_name, sizeof(new_name), "%s_sliced", src->name);
        ggml_set_name(dst, new_name);
    }

    // Perform the slice copy based on data type
    if (src->type == GGML_TYPE_F32) {
        const float* src_data = static_cast<const float*>(src->data);
        float* dst_data = static_cast<float*>(dst->data);

        if (dim == AXIS_ROWS) {
            // Slice rows: copy selected rows entirely
            // Memory layout: row-major, so row i is at src_data[i * src_cols]
            for (int i = 0; i < new_size; ++i) {
                const int src_row = indices[i];
                std::memcpy(dst_data + i * src_cols, src_data + src_row * src_cols, src_cols * sizeof(float));
            }
        } else {  // AXIS_COLS
            // Slice columns: for each row, copy only selected column values
            for (int64_t row = 0; row < src_rows; ++row) {
                for (int i = 0; i < new_size; ++i) {
                    const int src_col = indices[i];
                    dst_data[row * new_cols + i] = src_data[row * src_cols + src_col];
                }
            }
        }
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src_data = static_cast<const ggml_fp16_t*>(src->data);
        ggml_fp16_t* dst_data = static_cast<ggml_fp16_t*>(dst->data);

        if (dim == AXIS_ROWS) {
            // Slice rows: copy selected rows
            for (int i = 0; i < new_size; ++i) {
                const int src_row = indices[i];
                std::memcpy(dst_data + i * src_cols, src_data + src_row * src_cols, src_cols * sizeof(ggml_fp16_t));
            }
        } else {  // AXIS_COLS
            // Slice columns
            for (int64_t row = 0; row < src_rows; ++row) {
                for (int i = 0; i < new_size; ++i) {
                    const int src_col = indices[i];
                    dst_data[row * new_cols + i] = src_data[row * src_cols + src_col];
                }
            }
        }
    }

    return dst;
}

struct ggml_tensor* Slice1DTensor(struct ggml_context* ctx, struct ggml_tensor* src, const std::vector<int>& indices) {
    if (!ctx || !src || indices.empty()) {
        std::cerr << "[TensorUtils] Error: Invalid arguments to Slice1DTensor" << std::endl;
        return nullptr;
    }

    // Only allow F32 and F16
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        std::cerr << "[TensorUtils] Error: Slice1DTensor only supports F32/F16. Got type " << src->type << std::endl;
        return nullptr;
    }

    const int64_t src_size = src->ne[0];
    const int new_size = static_cast<int>(indices.size());

    // Validate indices
    for (int idx : indices) {
        if (idx < 0 || idx >= src_size) {
            std::cerr << "[TensorUtils] Error: Index " << idx << " out of bounds [0, " << src_size << ")" << std::endl;
            return nullptr;
        }
    }

    // Allocate new 1D tensor
    struct ggml_tensor* dst = ggml_new_tensor_1d(ctx, src->type, new_size);
    if (!dst) {
        std::cerr << "[TensorUtils] Error: Failed to allocate 1D tensor of size " << new_size << std::endl;
        return nullptr;
    }

    // Copy name with "_sliced" suffix
    if (src->name[0] != '\0') {
        char new_name[GGML_MAX_NAME];
        snprintf(new_name, sizeof(new_name), "%s_sliced", src->name);
        ggml_set_name(dst, new_name);
    }

    // Copy selected elements
    if (src->type == GGML_TYPE_F32) {
        const float* src_data = static_cast<const float*>(src->data);
        float* dst_data = static_cast<float*>(dst->data);
        for (int i = 0; i < new_size; ++i) {
            dst_data[i] = src_data[indices[i]];
        }
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src_data = static_cast<const ggml_fp16_t*>(src->data);
        ggml_fp16_t* dst_data = static_cast<ggml_fp16_t*>(dst->data);
        for (int i = 0; i < new_size; ++i) {
            dst_data[i] = src_data[indices[i]];
        }
    }

    return dst;
}

}  // namespace TensorUtils
}  // namespace densecore
