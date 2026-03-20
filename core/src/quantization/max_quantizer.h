#ifndef DENSECORE_MAX_QUANTIZER_H
#define DENSECORE_MAX_QUANTIZER_H

#include "quantizer.h"

namespace densecore {

// Simple max calibration quantizer (fast, no calibration data needed)
class MaxQuantizer : public Quantizer {
public:
    explicit MaxQuantizer(const QuantConfig& config) : Quantizer(config) {}

    void QuantizeWeight(struct ggml_tensor* tensor) override;

private:
    // Convert FP16/FP32 tensor to quantized format
    void QuantizeToINT4(struct ggml_tensor* tensor);
    void QuantizeToINT8(struct ggml_tensor* tensor);
};

}  // namespace densecore

#endif  // DENSECORE_MAX_QUANTIZER_H
