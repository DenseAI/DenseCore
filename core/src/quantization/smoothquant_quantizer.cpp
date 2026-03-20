#include "smoothquant_quantizer.h"

#include <ggml.h>

#include <cstring>
#include <iostream>

namespace densecore {

void SymmetricInt8Quantizer::Calibrate(const std::vector<float*>& samples, size_t sample_size) {
    // Q8_0 uses per-block scaling computed from weights, no calibration needed
    (void)samples;
    (void)sample_size;

    std::cout << "[SymmetricInt8Quantizer] Calibration skipped (Q8_0 uses "
                 "per-block weight scaling)"
              << std::endl;
}

void SymmetricInt8Quantizer::QuantizeWeight(struct ggml_tensor* tensor) {
    if (!ShouldQuantize(tensor->name)) {
        std::cout << "[SymmetricInt8Quantizer] Skipping: " << tensor->name << std::endl;
        return;
    }

    const enum ggml_type target_type = GGML_TYPE_Q8_0;

    // Don't re-quantize if already Q8_0
    if (tensor->type == target_type) {
        std::cout << "[SymmetricInt8Quantizer] Already quantized: " << tensor->name << std::endl;
        return;
    }

    // Skip if already quantized to another format
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        std::cout << "[SymmetricInt8Quantizer] Skipping non-FP tensor: " << tensor->name << " (type=" << tensor->type
                  << ")" << std::endl;
        return;
    }

    const int64_t nelements = ggml_nelements(tensor);

    // Get source data as FP32
    const float* src_data = nullptr;
    std::vector<float> fp32_buffer;

    if (tensor->type == GGML_TYPE_F32) {
        src_data = (const float*)tensor->data;
    } else if (tensor->type == GGML_TYPE_F16) {
        fp32_buffer.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tensor->data, fp32_buffer.data(), nelements);
        src_data = fp32_buffer.data();
    } else {
        std::cerr << "[SymmetricInt8Quantizer] Unsupported source type" << std::endl;
        return;
    }

    // Calculate Q8_0 buffer size
    const size_t q8_size = ggml_row_size(target_type, nelements);
    std::vector<uint8_t> quant_buffer(q8_size);

    // Quantize using GGML's symmetric INT8 quantizer
    // Q8_0 uses blockwise symmetric quantization with per-block scaling
    ggml_quantize_chunk(target_type, src_data, quant_buffer.data(), 0, nelements, tensor->ne[0], nullptr);

    // Update tensor metadata and copy quantized data
    tensor->type = target_type;
    memcpy(tensor->data, quant_buffer.data(), q8_size);

    // Log success
    std::cout << "[SymmetricInt8Quantizer] Quantized " << tensor->name << " to Q8_0 (" << nelements << " elements, "
              << q8_size << " bytes)" << std::endl;
}

}  // namespace densecore
