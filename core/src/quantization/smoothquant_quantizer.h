#ifndef DENSECORE_SYMMETRIC_INT8_QUANTIZER_H
#define DENSECORE_SYMMETRIC_INT8_QUANTIZER_H

#include <vector>

#include "quantizer.h"

namespace densecore {

/**
 * @brief Symmetric INT8 quantizer using GGML's Q8_0 format.
 *
 * This quantizer implements straightforward symmetric INT8 quantization
 * using GGML's Q8_0 format. It provides high accuracy with moderate
 * compression (roughly 4x vs FP32).
 *
 * ## Why not SmoothQuant?
 * True SmoothQuant requires:
 * 1. Calibration pass to collect per-channel activation statistics
 * 2. Computing smooth factors: s = max(|X|)^α / max(|W|)^(1-α)
 * 3. Scaling weights: W' = W * s
 * 4. Scaling activations at runtime: X' = X / s (REQUIRES MODIFIED INFERENCE)
 *
 * Without step 4, applying smooth scaling to weights RUINS accuracy because
 * the activations are not compensated. The inference path would need custom
 * kernels to apply the inverse scaling, which is not implemented in GGML.
 *
 * GGML's Q8_0 format provides:
 * - Block-level symmetric quantization
 * - Per-block scaling factors
 * - Fast dequantization during inference
 * - No modifications to inference path required
 *
 * For production use, Q8_0 provides excellent accuracy for INT8 models.
 */
class SymmetricInt8Quantizer : public Quantizer {
public:
    explicit SymmetricInt8Quantizer(const QuantConfig& config) : Quantizer(config) {}

    /**
     * @brief Calibrate is a no-op for symmetric quantization.
     *
     * Q8_0 uses per-block scaling computed directly from weights,
     * so no calibration data is needed.
     */
    void Calibrate(const std::vector<float*>& samples, size_t sample_size) override;

    void QuantizeWeight(struct ggml_tensor* tensor) override;
};

// Legacy alias for API compatibility
using SmoothQuantQuantizer = SymmetricInt8Quantizer;

}  // namespace densecore

#endif  // DENSECORE_SYMMETRIC_INT8_QUANTIZER_H
