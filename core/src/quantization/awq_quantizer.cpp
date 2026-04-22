#include "awq_quantizer.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "densecore/simd/simd_ops.h"

namespace densecore {

// =============================================================================
// GGML Type Mapping
// =============================================================================

static enum ggml_type GetGGMLType(QuantFormat format) {
    switch (format) {
    case QuantFormat::Q4_0: return GGML_TYPE_Q4_0;
    case QuantFormat::Q4_K_M: return GGML_TYPE_Q4_K;
    case QuantFormat::Q5_K_M: return GGML_TYPE_Q5_K;
    case QuantFormat::Q8_0: return GGML_TYPE_Q8_0;
    case QuantFormat::FP16:
    default: return GGML_TYPE_F16;
    }
}

// =============================================================================
// Calibration Implementation
// =============================================================================

void AWQQuantizer::Calibrate(const std::vector<float*>& samples, size_t sample_size) {
    if (samples.empty() || sample_size == 0) {
        std::cerr << "[AWQQuantizer] Calibrate: No samples provided" << std::endl;
        return;
    }

    // Resize and zero-initialize channel scales
    channel_scales_.resize(sample_size, 0.0f);

    // Find max absolute activation per channel across all samples
    for (const float* sample : samples) {
        if (!sample) continue;
        for (size_t c = 0; c < sample_size; ++c) {
            float abs_val = std::abs(sample[c]);
            if (abs_val > channel_scales_[c]) {
                channel_scales_[c] = abs_val;
            }
        }
    }

    // Find global max using SIMD
    float max_scale = simd::MaxF32(channel_scales_.data(), sample_size);

    // Normalize to [1.0, AWQ_PROTECT_RATIO] range
    // Channels with max activation get AWQ_PROTECT_RATIO boost
    // Channels with min activation get 1.0 (no boost)
    if (max_scale > 1e-6f) {
        float inv_max = 1.0f / max_scale;
        for (size_t c = 0; c < sample_size; ++c) {
            float normalized = channel_scales_[c] * inv_max;  // [0, 1]
            channel_scales_[c] = 1.0f + (AWQ_PROTECT_RATIO - 1.0f) * normalized;
        }
    } else {
        // All activations near zero, use uniform scaling
        std::fill(channel_scales_.begin(), channel_scales_.end(), 1.0f);
    }

    calibrated_ = true;
    std::cout << "[AWQQuantizer] Calibrated with " << samples.size() << " samples, " << sample_size << " channels"
              << std::endl;
}

// =============================================================================
// Per-Column Scaling Helper
// =============================================================================

void AWQQuantizer::ApplyChannelScaling(float* weights, size_t rows, size_t cols) {
    // Weight matrix is [rows, cols] in row-major order
    // Apply per-column scaling: W_scaled[r,c] = W[r,c] * scale[c]
    for (size_t r = 0; r < rows; ++r) {
        float* row = weights + r * cols;
        size_t scale_len = std::min(cols, channel_scales_.size());

        // Use SIMD for bulk of the row if possible
        for (size_t c = 0; c < scale_len; ++c) {
            row[c] *= channel_scales_[c];
        }
    }
}

// =============================================================================
// Quantization with AWQ Scaling
// =============================================================================

void AWQQuantizer::QuantizeWeight(struct ggml_tensor* tensor) {
    if (!ShouldQuantize(tensor->name)) {
        std::cout << "[AWQQuantizer] Skipping: " << tensor->name << std::endl;
        return;
    }

    // Determine target GGML type from config
    enum ggml_type target_type = GetGGMLType(config_.format);

    // Don't re-quantize if already at target type
    if (tensor->type == target_type) {
        std::cout << "[AWQQuantizer] Already quantized: " << tensor->name << std::endl;
        return;
    }

    // Skip if already quantized to another format
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        std::cout << "[AWQQuantizer] Skipping non-FP tensor: " << tensor->name << " (type=" << tensor->type << ")"
                  << std::endl;
        return;
    }

    const int64_t nelements = ggml_nelements(tensor);
    const size_t cols = tensor->ne[0];  // Number of columns (channels)
    const size_t rows = nelements / cols;

    // Get source data as FP32
    std::vector<float> fp32_buffer;
    float* work_data = nullptr;

    if (tensor->type == GGML_TYPE_F32) {
        // Make a copy so we can apply scaling without modifying original
        fp32_buffer.resize(nelements);
        std::memcpy(fp32_buffer.data(), tensor->data, nelements * sizeof(float));
        work_data = fp32_buffer.data();
    } else if (tensor->type == GGML_TYPE_F16) {
        fp32_buffer.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tensor->data, fp32_buffer.data(), nelements);
        work_data = fp32_buffer.data();
    } else {
        std::cerr << "[AWQQuantizer] Unsupported source type" << std::endl;
        return;
    }

    // Apply AWQ-style per-channel scaling if calibrated
    if (calibrated_ && !channel_scales_.empty()) {
        ApplyChannelScaling(work_data, rows, cols);
        std::cout << "[AWQQuantizer] Applied AWQ scaling to " << tensor->name << std::endl;
    }

    // Calculate quantized buffer size
    const size_t quant_size = ggml_row_size(target_type, nelements);
    std::vector<uint8_t> quant_buffer(quant_size);

    // Quantize using GGML's optimized blockwise quantizer
    ggml_quantize_chunk(target_type, work_data, quant_buffer.data(), 0, nelements, static_cast<int64_t>(cols), nullptr);

    // Update tensor metadata and copy quantized data
    tensor->type = target_type;
    std::memcpy(tensor->data, quant_buffer.data(), quant_size);

    // Log success
    const char* type_name = ggml_type_name(target_type);
    std::cout << "[AWQQuantizer] Quantized " << tensor->name << " to " << type_name << " (" << nelements
              << " elements, " << quant_size << " bytes)" << (calibrated_ ? " [AWQ scaled]" : "") << std::endl;
}

}  // namespace densecore
