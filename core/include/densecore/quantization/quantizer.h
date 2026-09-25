#ifndef DENSECORE_QUANTIZER_H
#define DENSECORE_QUANTIZER_H

#include <ggml.h>

#include <memory>
#include <vector>

#include "densecore.h"  // For DENSECORE_API
#include "densecore/quantization/quantization_config.h"

namespace densecore {

// Abstract base quantizer class
class Quantizer {
public:
    explicit Quantizer(const QuantConfig& config) : config_(config) {}
    virtual ~Quantizer() = default;

    // Calibrate using sample activations (if needed)
    // samples: vector of activation tensors from forward passes
    virtual void Calibrate(const std::vector<float*>& samples, size_t sample_size) {
        // Default: no-op (not all quantizers need calibration)
    }

    // Quantize a weight tensor in-place
    // tensor: GGML tensor to quantize (will be converted to quantized type)
    virtual void QuantizeWeight(struct ggml_tensor* tensor) = 0;

    // Helper: Check if a tensor should be quantized based on its name
    virtual bool ShouldQuantize(const std::string& tensor_name) const {
        // Skip output layer (lm_head)
        if (config_.skip_output_layer &&
            (tensor_name.find("output") != std::string::npos || tensor_name.find("lm_head") != std::string::npos)) {
            return false;
        }

        // Skip embeddings if configured
        if (config_.skip_embeddings && tensor_name.find("embeddings") != std::string::npos) {
            return false;
        }

        return true;
    }

protected:
    QuantConfig config_;
};

// Factory function to create quantizer based on config
DENSECORE_API std::unique_ptr<Quantizer> CreateQuantizer(const QuantConfig& config);

}  // namespace densecore

#endif  // DENSECORE_QUANTIZER_H
