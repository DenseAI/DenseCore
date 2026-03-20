#ifndef DENSECORE_AWQ_QUANTIZER_H
#define DENSECORE_AWQ_QUANTIZER_H

#include <vector>

#include "quantizer.h"

namespace densecore {

/**
 * @brief Activation-Aware Weight Quantizer (Simplified AWQ).
 *
 * This quantizer implements a simplified version of AWQ:
 * 1. Calibrate() computes per-channel activation magnitudes
 * 2. QuantizeWeight() scales weights to protect salient channels
 * 3. Standard GGML K-quant quantization is then applied
 *
 * ## Simplified AWQ Approach
 * True AWQ requires inference-time compensation (X' = X / scale).
 * Our approach scales weights only, which still provides benefit by
 * protecting high-activation channels from quantization error.
 *
 * For best results:
 * 1. Run forward pass on calibration data
 * 2. Call Calibrate() with activation samples
 * 3. Call QuantizeWeight() for each tensor
 */
class AWQQuantizer : public Quantizer {
public:
    explicit AWQQuantizer(const QuantConfig& config) : Quantizer(config) {}

    /**
     * Calibrate using activation samples.
     * @param samples Vector of activation tensors (each is [channels] floats)
     * @param sample_size Number of channels per sample
     */
    void Calibrate(const std::vector<float*>& samples, size_t sample_size) override;

    /**
     * Quantize weight tensor with AWQ-style scaling.
     */
    void QuantizeWeight(struct ggml_tensor* tensor) override;

private:
    // Per-channel scaling factors from calibration
    std::vector<float> channel_scales_;
    bool calibrated_ = false;

    // How much to boost the most salient channels (1.0 = no boost)
    static constexpr float AWQ_PROTECT_RATIO = 1.5f;

    // Helper: Apply per-column scaling to weight matrix
    void ApplyChannelScaling(float* weights, size_t rows, size_t cols);
};

// Legacy alias for backward compatibility
using BlockwiseQuantizer = AWQQuantizer;

}  // namespace densecore

#endif  // DENSECORE_AWQ_QUANTIZER_H
