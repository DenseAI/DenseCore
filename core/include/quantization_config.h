#ifndef DENSECORE_QUANTIZATION_CONFIG_H
#define DENSECORE_QUANTIZATION_CONFIG_H

#include <cstdint>
#include <string>

namespace densecore {

/**
 * Quantization algorithms.
 *
 * Note: Advanced algorithms like AWQ and SmoothQuant require special kernel
 * support to un-scale activations at runtime. Without this infrastructure,
 * they damage model accuracy. For MVP, we use reliable GGML-native methods.
 */
enum class QuantAlgorithm {
    // Reliable GGML-native quantization (recommended)
    GGML_Q4_K_M,  // 4-bit K-quants with medium accuracy (best quality/size ratio)
    GGML_Q5_K_M,  // 5-bit K-quants with medium accuracy (higher quality)
    GGML_Q8_0,    // 8-bit symmetric quantization (highest quality)
    GGML_Q4_0,    // 4-bit basic quantization (fastest, lowest quality)

    // Custom DenseCore algorithms
    INT4_PAPER,  // "Efficient LLM Inference on CPUs" paper implementation
                 // Custom block-wise INT4 with AVX512-optimized kernels
    FP8_CUSTOM,  // DenseCore FP8 (E4M3FN/E5M2) metadata-driven format

    // Legacy (kept for API compatibility, maps to safe defaults)
    MAX,          // Maps to GGML_Q4_0
    SMOOTHQUANT,  // Maps to GGML_Q8_0 (deprecated: true SmoothQuant not
                  // implemented)
    AWQ_LITE,     // Maps to GGML_Q4_K_M (deprecated: true AWQ not implemented)
    AWQ_CLIP,     // Maps to GGML_Q4_K_M (deprecated: true AWQ not implemented)
};

/**
 * Quantization formats (GGML types + custom formats).
 */
enum class QuantFormat {
    // GGML-native formats
    FP16,    // No quantization (baseline, GGML_TYPE_F16)
    Q4_0,    // 4-bit basic quantization (GGML_TYPE_Q4_0)
    Q4_K_M,  // 4-bit K-quants medium (GGML_TYPE_Q4_K, recommended for INT4)
    Q5_K_M,  // 5-bit K-quants medium (GGML_TYPE_Q5_K)
    Q8_0,    // 8-bit symmetric (GGML_TYPE_Q8_0, recommended for INT8)

    // Custom DenseCore formats
    INT4_BLOCKWISE,  // Custom INT4 block-wise quantization
                     // - Block size: 32, 64, or 128
                     // - AVX512-optimized kernels
                     // - Asymmetric quantization with per-block scale/zero
                     // - Stored in TensorInt4 format (tensor->extra)

    // Custom DenseCore FP8 formats (serialized as GGML_TYPE_I8 + metadata)
    FP8_E4M3FN,
    FP8_E5M2,

    // Legacy aliases (kept for API compatibility)
    FP8_E4M3 = FP8_E4M3FN,
    INT8 = Q8_0,
};

/**
 * Quantization configuration.
 */
struct QuantConfig {
    QuantFormat format = QuantFormat::Q4_K_M;
    QuantAlgorithm algorithm = QuantAlgorithm::GGML_Q4_K_M;

    // Blockwise quantization parameters
    int block_size = 128;  // Block size for custom INT4 (32, 64, or 128)

    // Quantization targets
    bool quantize_weights = true;
    bool quantize_activations = false;  // Weight-only (activation quant not supported)

    // Layer selection
    bool skip_output_layer = true;  // Skip lm_head for better accuracy
    bool skip_embeddings = true;    // Keep embeddings in FP16

    // Calibration parameters (unused in current implementation)
    int calib_size = 0;
    std::string calib_dataset = "";

    // Helper methods
    std::string GetFormatName() const {
        switch (format) {
        case QuantFormat::FP16: return "fp16";
        case QuantFormat::Q4_0: return "q4_0";
        case QuantFormat::Q4_K_M: return "q4_k_m";
        case QuantFormat::Q5_K_M: return "q5_k_m";
        case QuantFormat::Q8_0: return "q8_0";
        case QuantFormat::INT4_BLOCKWISE: return "int4_blockwise";
        case QuantFormat::FP8_E4M3FN: return "fp8_e4m3fn";
        case QuantFormat::FP8_E5M2: return "fp8_e5m2";
        default: return "unknown";
        }
    }

    std::string GetAlgorithmName() const {
        switch (algorithm) {
        case QuantAlgorithm::GGML_Q4_K_M: return "q4_k_m";
        case QuantAlgorithm::GGML_Q5_K_M: return "q5_k_m";
        case QuantAlgorithm::GGML_Q8_0: return "q8_0";
        case QuantAlgorithm::GGML_Q4_0: return "q4_0";
        case QuantAlgorithm::INT4_PAPER: return "int4_paper";
        case QuantAlgorithm::FP8_CUSTOM: return "fp8_custom";
        case QuantAlgorithm::MAX: return "max";
        case QuantAlgorithm::SMOOTHQUANT: return "smoothquant_compat";
        case QuantAlgorithm::AWQ_LITE:
        case QuantAlgorithm::AWQ_CLIP: return "awq_compat";
        default: return "unknown";
        }
    }

    /**
     * Check if this config uses custom DenseCore format (not GGML-native)
     */
    bool IsCustomFormat() const {
        return format == QuantFormat::INT4_BLOCKWISE || format == QuantFormat::FP8_E4M3FN ||
               format == QuantFormat::FP8_E5M2;
    }
};

// ============================================================================
// Predefined Configurations (Recommended)
// ============================================================================

/**
 * High-quality INT4 quantization (recommended for most use cases).
 * Uses Q4_K_M which provides excellent quality/size tradeoff.
 */
inline QuantConfig Q4_K_M_CFG() {
    QuantConfig cfg;
    cfg.format = QuantFormat::Q4_K_M;
    cfg.algorithm = QuantAlgorithm::GGML_Q4_K_M;
    cfg.quantize_weights = true;
    cfg.skip_output_layer = true;
    cfg.skip_embeddings = true;
    return cfg;
}

/**
 * Higher-quality INT5 quantization (when accuracy is critical).
 */
inline QuantConfig Q5_K_M_CFG() {
    QuantConfig cfg;
    cfg.format = QuantFormat::Q5_K_M;
    cfg.algorithm = QuantAlgorithm::GGML_Q5_K_M;
    cfg.quantize_weights = true;
    cfg.skip_output_layer = true;
    cfg.skip_embeddings = true;
    return cfg;
}

/**
 * INT8 quantization (highest quality, larger model size).
 */
inline QuantConfig Q8_0_CFG() {
    QuantConfig cfg;
    cfg.format = QuantFormat::Q8_0;
    cfg.algorithm = QuantAlgorithm::GGML_Q8_0;
    cfg.quantize_weights = true;
    cfg.skip_output_layer = false;  // Q8 is accurate enough for output layer
    cfg.skip_embeddings = false;
    return cfg;
}

/**
 * Custom INT4 block-wise quantization using DenseCore's optimized kernels.
 * Based on "Efficient LLM Inference on CPUs" paper.
 *
 * Features:
 * - AVX512-optimized on-the-fly dequantization
 * - Block-wise asymmetric quantization (scale + zero per block)
 * - 5-7× inference speedup on memory-bound workloads
 *
 * @param block_size Block size (32, 64, or 128). Default: 128 for best
 *                   cache efficiency on modern CPUs.
 */
inline QuantConfig INT4_PAPER_CFG(int block_size = 128) {
    QuantConfig cfg;
    cfg.format = QuantFormat::INT4_BLOCKWISE;
    cfg.algorithm = QuantAlgorithm::INT4_PAPER;
    cfg.block_size = block_size;
    cfg.quantize_weights = true;
    cfg.skip_output_layer = true;  // Keep lm_head in FP16 for accuracy
    cfg.skip_embeddings = true;    // Keep tok_embeddings in FP16
    return cfg;
}

// ============================================================================
// Legacy Configurations (Deprecated, kept for API compatibility)
// ============================================================================

/**
 * @deprecated Use Q4_K_M_CFG() instead.
 * AWQ is not properly implemented; this uses standard Q4_K_M quantization.
 */
inline QuantConfig INT4_AWQ_CFG() {
    return Q4_K_M_CFG();
}

/**
 * @deprecated Use Q8_0_CFG() instead.
 * SmoothQuant requires runtime activation scaling which is not implemented.
 */
inline QuantConfig INT8_SMOOTHQUANT_CFG() {
    return Q8_0_CFG();
}

/**
 * FP8 default configuration for DenseCore custom FP8 path.
 */
inline QuantConfig FP8_DEFAULT_CFG() {
    QuantConfig cfg;
    cfg.format = QuantFormat::FP8_E4M3FN;
    cfg.algorithm = QuantAlgorithm::FP8_CUSTOM;
    cfg.quantize_weights = true;
    cfg.skip_output_layer = true;
    cfg.skip_embeddings = true;
    return cfg;
}

}  // namespace densecore

#endif  // DENSECORE_QUANTIZATION_CONFIG_H
