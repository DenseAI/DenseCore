/**
 * @file densecore/quantization/mixed_precision.h
 * @brief Mixed Precision (Hybrid Quantization) Configuration
 *
 * Memory/accuracy trade-off optimization via per-layer quantization settings.
 *
 * Common strategies:
 *   - Attention (Q/K/V/O): FP16 or INT8 (requires high precision)
 *   - FFN (gate/up/down): INT4 (memory efficient)
 *   - Embedding/Output: FP16 (preserves embedding quality)
 *
 * Usage example:
 * @code
 *   MixedPrecisionConfig config = MixedPrecisionConfig::AttentionFP16_FFN_INT4(32);
 *   model->SetMixedPrecisionConfig(config);
 * @endcode
 */

#ifndef DENSECORE_MIXED_PRECISION_H
#define DENSECORE_MIXED_PRECISION_H

#include <string>
#include <vector>

namespace densecore {

/**
 * @brief Supported precision types
 */
enum class Precision {
    FP32,  ///< 32-bit floating point (baseline)
    FP16,  ///< 16-bit floating point (IEEE half)
    BF16,  ///< Brain floating point 16
    INT8,  ///< 8-bit integer (symmetric)
    INT4,  ///< 4-bit integer (K-quant style)
    AUTO   ///< Keep original model precision
};

/**
 * @brief Convert Precision type to string
 */
inline const char* PrecisionToString(Precision p) {
    switch (p) {
    case Precision::FP32: return "FP32";
    case Precision::FP16: return "FP16";
    case Precision::BF16: return "BF16";
    case Precision::INT8: return "INT8";
    case Precision::INT4: return "INT4";
    case Precision::AUTO: return "AUTO";
    default: return "UNKNOWN";
    }
}

/**
 * @brief Per-layer quantization configuration
 */
struct LayerQuantConfig {
    // Attention block
    Precision qkv_proj = Precision::AUTO;  ///< Q, K, V projection
    Precision o_proj = Precision::AUTO;    ///< Output projection

    // FFN block
    Precision ffn_gate = Precision::AUTO;  ///< Gate projection (w1)
    Precision ffn_up = Precision::AUTO;    ///< Up projection (w3)
    Precision ffn_down = Precision::AUTO;  ///< Down projection (w2)

    // Normalization
    Precision norm = Precision::FP32;  ///< LayerNorm/RMSNorm weights (FP32 always recommended)

    /**
     * @brief Set all weights to the same precision
     */
    static LayerQuantConfig Uniform(Precision p) { return {p, p, p, p, p, Precision::FP32}; }

    /**
     * @brief High precision for Attention, low precision for FFN
     */
    static LayerQuantConfig AttentionHigh_FFN_Low(Precision attn, Precision ffn) {
        return {attn, attn, ffn, ffn, ffn, Precision::FP32};
    }
};

/**
 * @brief Model-wide Mixed Precision configuration
 */
struct MixedPrecisionConfig {
    std::vector<LayerQuantConfig> layers;  ///< Per-layer configuration

    // Global settings
    Precision embedding = Precision::AUTO;    ///< Token embedding
    Precision output_head = Precision::AUTO;  ///< LM head (output projection)
    Precision kv_cache = Precision::FP16;     ///< KV Cache storage precision

    // Runtime settings
    bool enable_dynamic_casting = true;  ///< Allow dynamic type casting during computation
    bool log_precision_stats = false;    ///< Enable precision statistics logging

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Set entire model to uniform precision
     *
     * @param n_layers Number of layers
     * @param precision Precision to apply
     */
    static MixedPrecisionConfig Uniform(int n_layers, Precision precision) {
        MixedPrecisionConfig config;
        config.layers.resize(n_layers, LayerQuantConfig::Uniform(precision));
        config.embedding = precision;
        config.output_head = precision;
        return config;
    }

    /**
     * @brief Attention FP16, FFN INT4 (recommended setting)
     *
     * Optimal quality/performance trade-off for most LLMs.
     * Attention is precision-sensitive, so keep at FP16.
     * FFN contains most parameters, so compress to INT4.
     *
     * @param n_layers Number of layers
     */
    static MixedPrecisionConfig AttentionFP16_FFN_INT4(int n_layers) {
        MixedPrecisionConfig config;
        config.layers.resize(n_layers, LayerQuantConfig::AttentionHigh_FFN_Low(Precision::FP16, Precision::INT4));
        config.embedding = Precision::FP16;
        config.output_head = Precision::FP16;
        config.kv_cache = Precision::FP16;
        return config;
    }

    /**
     * @brief Full INT8 (high throughput for server deployment)
     *
     * @param n_layers Number of layers
     */
    static MixedPrecisionConfig AllINT8(int n_layers) {
        MixedPrecisionConfig config;
        config.layers.resize(n_layers, LayerQuantConfig::Uniform(Precision::INT8));
        config.embedding = Precision::INT8;
        config.output_head = Precision::INT8;
        config.kv_cache = Precision::FP16;  // FP16 recommended for KV Cache
        return config;
    }

    /**
     * @brief Custom per-layer configuration
     *
     * Apply different precision to early/middle/late layers.
     * Research suggests early and late layers are more sensitive.
     *
     * @param n_layers Number of layers
     * @param first_n First N layers (FP16)
     * @param last_n Last N layers (FP16)
     */
    static MixedPrecisionConfig SensitiveEdges(int n_layers, int first_n, int last_n) {
        MixedPrecisionConfig config;
        config.layers.reserve(n_layers);

        for (int i = 0; i < n_layers; i++) {
            if (i < first_n || i >= n_layers - last_n) {
                // Edge layers: higher precision
                config.layers.push_back(LayerQuantConfig::AttentionHigh_FFN_Low(Precision::FP16, Precision::INT8));
            } else {
                // Middle layers: lower precision
                config.layers.push_back(LayerQuantConfig::AttentionHigh_FFN_Low(Precision::INT8, Precision::INT4));
            }
        }

        config.embedding = Precision::FP16;
        config.output_head = Precision::FP16;
        return config;
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /**
     * @brief Print configuration summary
     */
    std::string Summary() const {
        std::string s = "MixedPrecisionConfig:\n";
        s += "  Layers: " + std::to_string(layers.size()) + "\n";
        s += "  Embedding: " + std::string(PrecisionToString(embedding)) + "\n";
        s += "  Output Head: " + std::string(PrecisionToString(output_head)) + "\n";
        s += "  KV Cache: " + std::string(PrecisionToString(kv_cache)) + "\n";

        if (!layers.empty()) {
            // Sample first layer
            const auto& l0 = layers[0];
            s += "  Layer[0] QKV: " + std::string(PrecisionToString(l0.qkv_proj)) +
                 ", FFN: " + std::string(PrecisionToString(l0.ffn_gate)) + "\n";
        }
        return s;
    }

    /**
     * @brief Estimate total memory saving ratio
     *
     * @param base_precision Baseline precision (usually FP16)
     * @return Saving ratio (e.g., 0.5 = 50% reduction)
     */
    float EstimateMemorySaving(Precision base_precision = Precision::FP16) const {
        // Rough estimation based on bit widths
        auto bits = [](Precision p) -> int {
            switch (p) {
            case Precision::FP32: return 32;
            case Precision::FP16:
            case Precision::BF16: return 16;
            case Precision::INT8: return 8;
            case Precision::INT4: return 4;
            default: return 16;
            }
        };

        int base_bits = bits(base_precision);
        float total_ratio = 0.0f;
        int count = 0;

        for (const auto& layer : layers) {
            // Weight distribution: ~60% FFN, ~40% Attention
            float attn_ratio = 0.4f * (float)bits(layer.qkv_proj) / base_bits;
            float ffn_ratio = 0.6f * (float)bits(layer.ffn_gate) / base_bits;
            total_ratio += attn_ratio + ffn_ratio;
            count++;
        }

        if (count == 0) return 0.0f;
        return 1.0f - (total_ratio / count);
    }
};

}  // namespace densecore

#endif  // DENSECORE_MIXED_PRECISION_H
