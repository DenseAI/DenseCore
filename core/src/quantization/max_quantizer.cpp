#include "max_quantizer.h"

#include <ggml.h>

#include <cmath>
#include <cstring>
#include <iostream>

namespace densecore {

void MaxQuantizer::QuantizeWeight(struct ggml_tensor* tensor) {
    if (!ShouldQuantize(tensor->name)) {
        std::cout << "Skipping quantization for: " << tensor->name << std::endl;
        return;
    }

    switch (config_.format) {
    case QuantFormat::Q4_0:
    case QuantFormat::Q4_K_M: QuantizeToINT4(tensor); break;

    case QuantFormat::Q5_K_M:
        // Q5_K requires different handling, fall back to INT4 for now
        QuantizeToINT4(tensor);
        break;

    case QuantFormat::Q8_0: QuantizeToINT8(tensor); break;

    case QuantFormat::FP16:
        // No quantization needed
        break;

    case QuantFormat::FP8_E4M3FN:
    case QuantFormat::FP8_E5M2:
        std::cerr << "Warning: MaxQuantizer does not support FP8 formats." << " Use quantize CLI fp8_* custom path."
                  << std::endl;
        break;

    case QuantFormat::INT4_BLOCKWISE:
        // Custom format - use INT4Quantizer instead
        std::cerr << "Warning: MaxQuantizer does not support INT4_BLOCKWISE format." << " Use INT4Quantizer instead."
                  << std::endl;
        break;
    }
}

void MaxQuantizer::QuantizeToINT4(struct ggml_tensor* tensor) {
    // Use GGML's built-in INT4 quantization (Q4_0 or Q4_1)
    const int64_t nelements = ggml_nelements(tensor);

    // Check if tensor is already quantized
    if (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_1) {
        std::cout << "Tensor " << tensor->name << " already quantized (INT4)" << std::endl;
        return;
    }

    // Get source data (assume FP32 or FP16)
    const float* src_data = nullptr;
    std::vector<float> fp32_buffer;

    if (tensor->type == GGML_TYPE_F32) {
        src_data = (const float*)tensor->data;
    } else if (tensor->type == GGML_TYPE_F16) {
        // Convert FP16 to FP32 first
        fp32_buffer.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tensor->data, fp32_buffer.data(), nelements);
        src_data = fp32_buffer.data();
    } else {
        std::cerr << "Unsupported source type for quantization: " << tensor->type << std::endl;
        return;
    }

    // Buffer for quantized data
    // Q4_0 uses 4 bits per weight + scale
    const size_t q4_size = ggml_row_size(GGML_TYPE_Q4_0, nelements);
    std::vector<uint8_t> quant_buffer(q4_size);

    // Quantize
    ggml_quantize_chunk(GGML_TYPE_Q4_0, src_data, quant_buffer.data(), 0, nelements, tensor->ne[0], nullptr);

    // Update tensor
    tensor->type = GGML_TYPE_Q4_0;
    std::memcpy(tensor->data, quant_buffer.data(), q4_size);

    std::cout << "Quantized " << tensor->name << " to INT4 (Q4_0)" << std::endl;
}

void MaxQuantizer::QuantizeToINT8(struct ggml_tensor* tensor) {
    // Use GGML's INT8 quantization (Q8_0)
    const int64_t nelements = ggml_nelements(tensor);

    if (tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q8_1) {
        std::cout << "Tensor " << tensor->name << " already quantized (INT8)" << std::endl;
        return;
    }

    // Get source data
    const float* src_data = nullptr;
    std::vector<float> fp32_buffer;

    if (tensor->type == GGML_TYPE_F32) {
        src_data = (const float*)tensor->data;
    } else if (tensor->type == GGML_TYPE_F16) {
        fp32_buffer.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tensor->data, fp32_buffer.data(), nelements);
        src_data = fp32_buffer.data();
    } else {
        std::cerr << "Unsupported source type for quantization: " << tensor->type << std::endl;
        return;
    }

    // Calculate required size for Q8_0 format
    const size_t q8_size = ggml_row_size(GGML_TYPE_Q8_0, nelements);

    // Allocate buffer for quantized data
    std::vector<uint8_t> quant_buffer(q8_size);

    // Quantize using GGML's block quantizer
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src_data, quant_buffer.data(), 0, nelements, tensor->ne[0], nullptr);

    // Update tensor type and data
    tensor->type = GGML_TYPE_Q8_0;
    memcpy(tensor->data, quant_buffer.data(), q8_size);

    std::cout << "Quantized " << tensor->name << " to INT8 (Q8_0)" << std::endl;
}

}  // namespace densecore
