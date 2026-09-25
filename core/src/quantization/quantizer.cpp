#include "densecore/quantization/quantizer.h"

#include <stdexcept>

#include "quantization/awq_quantizer.h"
#include "quantization/int4_quantizer.h"
#include "quantization/max_quantizer.h"
#include "quantization/smoothquant_quantizer.h"

namespace densecore {

std::unique_ptr<Quantizer> CreateQuantizer(const QuantConfig& config) {
    // Check if custom INT4 block-wise quantization is requested
    if (config.format == QuantFormat::INT4_BLOCKWISE) {
        return std::make_unique<INT4Quantizer>(config);
    }

    // Legacy algorithm-based dispatch
    switch (config.algorithm) {
    case QuantAlgorithm::MAX: return std::make_unique<MaxQuantizer>(config);

    case QuantAlgorithm::AWQ_LITE:
    case QuantAlgorithm::AWQ_CLIP: return std::make_unique<AWQQuantizer>(config);

    case QuantAlgorithm::SMOOTHQUANT: return std::make_unique<SmoothQuantQuantizer>(config);

    case QuantAlgorithm::FP8_CUSTOM:
        throw std::runtime_error(
            "FP8 custom quantization is handled by quantize CLI custom path (fp8_e4m3fn/fp8_e5m2)");

    default: throw std::runtime_error("Unknown quantization algorithm");
    }
}

}  // namespace densecore
