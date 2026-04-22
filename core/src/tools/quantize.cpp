/**
 * @file quantize.cpp
 * @brief DenseCore Model Quantization CLI Tool
 *
 * Supports both GGML-native quantization and custom INT4_BLOCKWISE format.
 * For INT4_BLOCKWISE, uses the DenseCore Quantizer abstraction instead
 * of raw ggml_quantize_chunk calls.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "densecore/quantization/int4_types.h"
#include "densecore/quantization/quantization_config.h"
#include "densecore/quantization/quantizer.h"
#include "ggml.h"
#include "gguf.h"
#include "quantization/int4_quantizer.h"

using namespace densecore;

// ============================================================================
// Tensor Selection Logic
// ============================================================================

/**
 * Create a GGML context dedicated to tensor metadata definitions.
 *
 * Quantization writes tensor bytes manually, so output tensor data buffers are
 * unnecessary; we only need shape/type/name descriptors.
 */
static struct ggml_context* CreateTensorMetaContext(size_t tensor_slots) {
    // Reserve generous headroom to avoid allocator edge cases with alignment.
    const size_t bytes = (tensor_slots + 16) * ggml_tensor_overhead() + (1u << 20);
    struct ggml_init_params params = {
        .mem_size = bytes,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    return ggml_init(params);
}

/**
 * Create a tensor descriptor that mirrors an existing tensor's shape.
 */
static struct ggml_tensor* CreateTensorMetaLike(struct ggml_context* ctx, const struct ggml_tensor* src, ggml_type type,
                                                const char* name) {
    struct ggml_tensor* meta = ggml_new_tensor(ctx, type, ggml_n_dims(src), src->ne);
    ggml_set_name(meta, name);
    return meta;
}

/**
 * Check if a tensor should be quantized based on its name.
 */
bool ShouldQuantize(const std::string& name, const QuantConfig& config) {
    // Skip embeddings if configured
    if (config.skip_embeddings) {
        if (name.find("tok_embed") != std::string::npos || name.find("token_embed") != std::string::npos ||
            name.find("wte") != std::string::npos) {
            return false;
        }
    }

    // Skip output layer (LM head) if configured
    if (config.skip_output_layer) {
        if (name.find("output") != std::string::npos && name.find("layer") == std::string::npos) {
            return false;  // Skip lm_head but not layer outputs
        }
        if (name.find("lm_head") != std::string::npos) {
            return false;
        }
    }

    // Quantize weights (2D), but skip norms and biases
    if (name.find("weight") != std::string::npos) {
        if (name.find("norm") != std::string::npos) return false;  // Skip normalization weights (keep F32)
        if (name.find("bias") != std::string::npos) return false;  // Skip biases (keep F32)
        return true;                                               // Quantize attention/ffn weights
    }

    return false;
}

/**
 * Convert QuantFormat to GGML type for GGML-native quantization.
 */
ggml_type FormatToGGMLType(QuantFormat format) {
    switch (format) {
    case QuantFormat::Q4_0: return GGML_TYPE_Q4_0;
    case QuantFormat::Q4_K_M: return GGML_TYPE_Q4_K;
    case QuantFormat::Q5_K_M: return GGML_TYPE_Q5_K;
    case QuantFormat::Q8_0: return GGML_TYPE_Q8_0;
    case QuantFormat::FP16: return GGML_TYPE_F16;
    default: return GGML_TYPE_Q4_0;
    }
}

/**
 * Check whether a tensor can be represented by the target GGML quant type.
 *
 * Many K-quant formats require ne[0] to be divisible by their block size.
 */
static bool CanUseGGMLQuantType(const struct ggml_tensor* tensor, ggml_type qtype) {
    if (!tensor) return false;
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) return false;
    if (ggml_n_dims(tensor) != 2) return false;

    const int64_t k = tensor->ne[0];
    const int block = ggml_blck_size(qtype);
    if (block <= 0) return false;
    return (k % block) == 0;
}

/**
 * Parse command line to create QuantConfig.
 */
QuantConfig ParseConfig(int argc, char** argv) {
    std::string type_str = (argc > 3) ? argv[3] : "q4_0";
    int block_size = (argc > 4) ? std::atoi(argv[4]) : 128;

    // Parse format
    if (type_str == "int4" || type_str == "int4_blockwise" || type_str == "int4_paper") {
        return INT4_PAPER_CFG(block_size);
    } else if (type_str == "q4_k_m" || type_str == "q4_k") {
        return Q4_K_M_CFG();
    } else if (type_str == "q5_k_m" || type_str == "q5_k") {
        return Q5_K_M_CFG();
    } else if (type_str == "q8_0" || type_str == "int8") {
        return Q8_0_CFG();
    } else if (type_str == "q4_0") {
        QuantConfig cfg;
        cfg.format = QuantFormat::Q4_0;
        cfg.algorithm = QuantAlgorithm::GGML_Q4_0;
        return cfg;
    } else if (type_str == "q4_1") {
        // Q4_1 uses Q4_0 config but different GGML type
        QuantConfig cfg;
        cfg.format = QuantFormat::Q4_0;
        cfg.algorithm = QuantAlgorithm::GGML_Q4_0;
        return cfg;
    } else if (type_str == "f16" || type_str == "fp16") {
        QuantConfig cfg;
        cfg.format = QuantFormat::FP16;
        cfg.quantize_weights = false;
        return cfg;
    } else if (type_str == "fp8" || type_str == "fp8_e4m3" || type_str == "fp8_e4m3fn") {
        QuantConfig cfg = FP8_DEFAULT_CFG();
        cfg.format = QuantFormat::FP8_E4M3FN;
        return cfg;
    } else if (type_str == "fp8_e5m2") {
        QuantConfig cfg = FP8_DEFAULT_CFG();
        cfg.format = QuantFormat::FP8_E5M2;
        return cfg;
    }

    // Default
    QuantConfig cfg;
    cfg.format = QuantFormat::Q4_0;
    return cfg;
}

// ============================================================================
// GGML-Native Quantization Path
// ============================================================================

/**
 * Quantize model using GGML-native quantization (q4_0, q8_0, etc.)
 */
int QuantizeGGML(const char* input_path, const char* output_path, const QuantConfig& config) {
    ggml_type qtype = FormatToGGMLType(config.format);

    std::cout << "[Quantize] Loading model '" << input_path << "'..." << std::endl;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = nullptr,
    };

    struct ggml_context* ctx_in = nullptr;
    params.ctx = &ctx_in;

    struct gguf_context* ctx_gguf = gguf_init_from_file(input_path, params);
    if (!ctx_gguf) {
        std::cerr << "Error: Failed to load GGUF file: " << input_path << std::endl;
        return 1;
    }

    std::cout << "[Quantize] Model loaded. Quantizing to " << config.GetFormatName() << "..." << std::endl;

    // Create output GGUF context
    struct gguf_context* ctx_out = gguf_init_empty();

    // Copy KV pairs (Metadata)
    int n_kv = gguf_get_n_kv(ctx_gguf);
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx_gguf, i);
        gguf_type type = gguf_get_kv_type(ctx_gguf, i);

        if (type == GGUF_TYPE_ARRAY) {
            gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
            int n = gguf_get_arr_n(ctx_gguf, i);
            const void* data = gguf_get_arr_data(ctx_gguf, i);
            gguf_set_arr_data(ctx_out, key, arr_type, data, n);
        } else if (type == GGUF_TYPE_STRING) {
            const char* val = gguf_get_val_str(ctx_gguf, i);
            gguf_set_val_str(ctx_out, key, val);
        } else {
            switch (type) {
            case GGUF_TYPE_UINT8: gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_gguf, i)); break;
            case GGUF_TYPE_INT8: gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT16: gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_gguf, i)); break;
            case GGUF_TYPE_INT16: gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT32: gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_gguf, i)); break;
            case GGUF_TYPE_INT32: gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT64: gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_gguf, i)); break;
            case GGUF_TYPE_INT64: gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_gguf, i)); break;
            case GGUF_TYPE_BOOL: gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_gguf, i)); break;
            default: std::cerr << "Warning: Skipping unknown KV type for key: " << key << std::endl;
            }
        }
    }

    // Initialize quantization tables
    ggml_quantize_init(qtype);

    // Process tensors (Pass 1: Add tensor info)
    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    struct ggml_context* ctx_meta = CreateTensorMetaContext(static_cast<size_t>(n_tensors));
    if (!ctx_meta) {
        std::cerr << "Error: Failed to initialize output tensor metadata context." << std::endl;
        gguf_free(ctx_out);
        gguf_free(ctx_gguf);
        ggml_free(ctx_in);
        return 1;
    }

    std::vector<struct ggml_tensor*> out_tensors;
    std::vector<bool> will_quantize;
    out_tensors.reserve(n_tensors);
    will_quantize.reserve(n_tensors);

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);

        const bool quantize_requested = ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2;
        const bool quantize = quantize_requested && CanUseGGMLQuantType(tensor, qtype);
        if (quantize_requested && !quantize) {
            std::cout << "  Skipping incompatible tensor for " << config.GetFormatName() << ": " << name << " ["
                      << tensor->ne[0] << "x" << tensor->ne[1] << "]" << std::endl;
        }
        ggml_type target_type = quantize ? qtype : tensor->type;

        struct ggml_tensor* out_t = CreateTensorMetaLike(ctx_meta, tensor, target_type, name);
        gguf_add_tensor(ctx_out, out_t);
        out_tensors.push_back(out_t);
        will_quantize.push_back(quantize);
    }

    // Write header
    std::cout << "[Quantize] Writing header to " << output_path << "..." << std::endl;
    gguf_write_to_file(ctx_out, output_path, true);

    // Re-open for appending
    FILE* f = fopen(output_path, "ab");
    if (!f) {
        std::cerr << "Error: Failed to open output file for appending." << std::endl;
        return 1;
    }

    const size_t alignment = gguf_get_alignment(ctx_out);

    // Padding helper
    auto pad_file = [&](size_t alignment) {
        long pos = ftell(f);
        size_t padding = (alignment - (pos % alignment)) % alignment;
        if (padding > 0) {
            char buf[32] = {0};
            fwrite(buf, 1, padding, f);
        }
    };

    // Process tensors (Pass 2: Write data)
    std::cout << "[Quantize] Writing tensors..." << std::endl;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);
        struct ggml_tensor* out_t = out_tensors[i];

        const bool quantize = will_quantize[i];

        pad_file(alignment);

        if (quantize) {
            int64_t nelements = ggml_nelements(tensor);
            size_t row_size = ggml_row_size(out_t->type, tensor->ne[0]);
            size_t data_size = row_size * tensor->ne[1];

            std::vector<uint8_t> qdata(data_size);
            std::vector<float> f32_data;
            const float* src_data = nullptr;

            if (tensor->type == GGML_TYPE_F32) {
                src_data = (const float*)tensor->data;
            } else if (tensor->type == GGML_TYPE_F16) {
                f32_data.resize(nelements);
                ggml_fp16_to_fp32_row((const ggml_fp16_t*)tensor->data, f32_data.data(), nelements);
                src_data = f32_data.data();
            }

            ggml_quantize_chunk(out_t->type, src_data, qdata.data(), 0, tensor->ne[1], tensor->ne[0], nullptr);

            fwrite(qdata.data(), 1, data_size, f);

            std::cout << "  Converted " << name << " [" << tensor->ne[0] << "x" << tensor->ne[1] << "]" << std::endl;
        } else {
            size_t size = ggml_nbytes(tensor);
            fwrite(tensor->data, 1, size, f);
        }
    }

    // GGUF data section expects each tensor contribution to be padded, including
    // the final tensor.
    pad_file(alignment);

    fclose(f);

    gguf_free(ctx_out);
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    ggml_free(ctx_in);

    std::cout << "[Quantize] Done. Output saved to " << output_path << std::endl;
    return 0;
}

// ============================================================================
// Custom FP8 Quantization Path (DenseCore metadata-driven format)
// ============================================================================

inline uint8_t QuantizeScalarToFP8E5M2(float x) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    const float ax = std::fabs(x);

    if (std::isnan(ax)) return static_cast<uint8_t>(sign | 0x7F);
    if (!std::isfinite(ax)) return static_cast<uint8_t>(sign | 0x7B);  // Saturate to max finite.
    if (ax == 0.0f) return sign;

    constexpr float kMinNormal = 6.103515625e-5f;  // 2^-14
    constexpr float kSubScale = 65536.0f;          // 2^16

    auto quantize_subnormal = [&](float v) -> uint8_t {
        int q = static_cast<int>(std::lrintf(v * kSubScale));
        if (q <= 0) return sign;
        if (q >= 4) return static_cast<uint8_t>(sign | 0x04);  // Promote to minimum normal.
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(q & 0x03));
    };

    if (ax < kMinNormal) {
        return quantize_subnormal(ax);
    }

    int exp2 = 0;
    float m = std::frexp(ax, &exp2);  // ax = m * 2^exp2, m in [0.5, 1)
    int exp_field = (exp2 - 1) + 15;
    float norm = std::ldexp(m, 1);  // [1, 2)
    int mant = static_cast<int>(std::lrintf((norm - 1.0f) * 4.0f));

    if (mant == 4) {
        mant = 0;
        exp_field += 1;
    }
    if (exp_field <= 0) {
        return quantize_subnormal(ax);
    }
    if (exp_field >= 31) {
        return static_cast<uint8_t>(sign | 0x7B);  // Max finite: exp=30, mant=3
    }

    return static_cast<uint8_t>(sign | ((exp_field & 0x1F) << 2) | (mant & 0x03));
}

inline uint8_t QuantizeScalarToFP8E4M3FN(float x) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    const float ax = std::fabs(x);

    if (std::isnan(ax)) return static_cast<uint8_t>(sign | 0x7F);
    if (!std::isfinite(ax)) return static_cast<uint8_t>(sign | 0x7E);  // Saturate to max finite.
    if (ax == 0.0f) return sign;

    constexpr float kMinNormal = 0.015625f;  // 2^-6
    constexpr float kSubScale = 512.0f;      // 2^9

    auto quantize_subnormal = [&](float v) -> uint8_t {
        int q = static_cast<int>(std::lrintf(v * kSubScale));
        if (q <= 0) return sign;
        if (q >= 8) return static_cast<uint8_t>(sign | 0x08);  // Promote to minimum normal.
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(q & 0x07));
    };

    if (ax < kMinNormal) {
        return quantize_subnormal(ax);
    }

    int exp2 = 0;
    float m = std::frexp(ax, &exp2);
    int exp_field = (exp2 - 1) + 7;
    float norm = std::ldexp(m, 1);
    int mant = static_cast<int>(std::lrintf((norm - 1.0f) * 8.0f));

    if (mant == 8) {
        mant = 0;
        exp_field += 1;
    }
    if (exp_field <= 0) {
        return quantize_subnormal(ax);
    }
    if (exp_field >= 16) {
        return static_cast<uint8_t>(sign | 0x7E);  // Max finite: exp=15, mant=6
    }
    if (exp_field == 15 && mant == 7) {
        mant = 6;  // 0x7F/0xFF are NaN encodings.
    }

    return static_cast<uint8_t>(sign | ((exp_field & 0x0F) << 3) | (mant & 0x07));
}

inline void QuantizeRowToFP8(const float* src, uint8_t* dst, int64_t K, QuantFormat format) {
    if (format == QuantFormat::FP8_E5M2) {
        for (int64_t k = 0; k < K; ++k) dst[k] = QuantizeScalarToFP8E5M2(src[k]);
    } else {
        for (int64_t k = 0; k < K; ++k) dst[k] = QuantizeScalarToFP8E4M3FN(src[k]);
    }
}

int QuantizeFP8Custom(const char* input_path, const char* output_path, const QuantConfig& config) {
    const bool is_e5m2 = (config.format == QuantFormat::FP8_E5M2);
    const char* format_tag = is_e5m2 ? "e5m2" : "e4m3fn";
    const char* quant_format_name = is_e5m2 ? "fp8_e5m2" : "fp8_e4m3fn";

    std::cout << "[Quantize] Loading model '" << input_path << "'..." << std::endl;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = nullptr,
    };

    struct ggml_context* ctx_in = nullptr;
    params.ctx = &ctx_in;

    struct gguf_context* ctx_gguf = gguf_init_from_file(input_path, params);
    if (!ctx_gguf) {
        std::cerr << "Error: Failed to load GGUF file: " << input_path << std::endl;
        return 1;
    }

    std::cout << "[Quantize] Model loaded. Using custom FP8 quantization (" << quant_format_name << ")..." << std::endl;

    struct gguf_context* ctx_out = gguf_init_empty();

    // Copy KV pairs (Metadata) from input
    int n_kv = gguf_get_n_kv(ctx_gguf);
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx_gguf, i);
        gguf_type type = gguf_get_kv_type(ctx_gguf, i);

        if (type == GGUF_TYPE_ARRAY) {
            gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
            int n = gguf_get_arr_n(ctx_gguf, i);
            const void* data = gguf_get_arr_data(ctx_gguf, i);
            gguf_set_arr_data(ctx_out, key, arr_type, data, n);
        } else if (type == GGUF_TYPE_STRING) {
            const char* val = gguf_get_val_str(ctx_gguf, i);
            gguf_set_val_str(ctx_out, key, val);
        } else {
            switch (type) {
            case GGUF_TYPE_UINT8: gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_gguf, i)); break;
            case GGUF_TYPE_INT8: gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT16: gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_gguf, i)); break;
            case GGUF_TYPE_INT16: gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT32: gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_gguf, i)); break;
            case GGUF_TYPE_INT32: gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT64: gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_gguf, i)); break;
            case GGUF_TYPE_INT64: gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_gguf, i)); break;
            case GGUF_TYPE_BOOL: gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_gguf, i)); break;
            default: break;
            }
        }
    }

    gguf_set_val_str(ctx_out, "densecore.quantization_format", quant_format_name);

    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    struct ggml_context* ctx_meta = CreateTensorMetaContext(static_cast<size_t>(n_tensors));
    if (!ctx_meta) {
        std::cerr << "Error: Failed to initialize output tensor metadata context." << std::endl;
        gguf_free(ctx_out);
        gguf_free(ctx_gguf);
        ggml_free(ctx_in);
        return 1;
    }
    std::vector<std::string> tensor_names;
    std::vector<bool> is_quantized;
    int quantized_count = 0;

    std::cout << "[Quantize] Processing " << n_tensors << " tensors..." << std::endl;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);

        if (!tensor) {
            std::cerr << "Warning: Tensor not found: " << name << std::endl;
            tensor_names.push_back(name);
            is_quantized.push_back(false);
            continue;
        }

        bool should_quantize = ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2 &&
                               (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
        if (should_quantize) {
            ++quantized_count;
            std::cout << "  Quantizing " << name << " [" << tensor->ne[0] << "x" << tensor->ne[1] << "]..."
                      << std::endl;
        }

        tensor_names.push_back(name);
        is_quantized.push_back(should_quantize);
    }

    // Pass 2: Add tensor definitions to GGUF context.
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = tensor_names[i].c_str();
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);
        if (!tensor) continue;

        if (is_quantized[i]) {
            const int64_t K = tensor->ne[0];
            const int64_t N = tensor->ne[1];
            struct ggml_tensor* q = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_I8, K, N);
            ggml_set_name(q, name);
            gguf_add_tensor(ctx_out, q);

            std::string meta_key = std::string("densecore.fp8.") + name;
            gguf_set_val_str(ctx_out, (meta_key + ".format").c_str(), format_tag);
            gguf_set_val_i64(ctx_out, (meta_key + ".K").c_str(), K);
            gguf_set_val_i64(ctx_out, (meta_key + ".N").c_str(), N);
        } else {
            struct ggml_tensor* t = CreateTensorMetaLike(ctx_meta, tensor, tensor->type, name);
            gguf_add_tensor(ctx_out, t);
        }
    }

    std::cout << "[Quantize] Writing header to " << output_path << "..." << std::endl;
    gguf_write_to_file(ctx_out, output_path, true);

    FILE* f = fopen(output_path, "ab");
    if (!f) {
        std::cerr << "Error: Failed to open output file for appending." << std::endl;
        gguf_free(ctx_out);
        gguf_free(ctx_gguf);
        ggml_free(ctx_in);
        return 1;
    }

    const size_t alignment = gguf_get_alignment(ctx_out);

    auto pad_file = [&](size_t alignment) {
        long pos = ftell(f);
        size_t padding = (alignment - (pos % alignment)) % alignment;
        if (padding > 0) {
            char buf[32] = {0};
            fwrite(buf, 1, padding, f);
        }
    };

    std::cout << "[Quantize] Writing tensors..." << std::endl;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = tensor_names[i].c_str();
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);
        if (!tensor) continue;

        if (is_quantized[i]) {
            const int64_t K = tensor->ne[0];
            const int64_t N = tensor->ne[1];
            std::vector<float> row_f32(static_cast<size_t>(K));
            std::vector<uint8_t> row_fp8(static_cast<size_t>(K));

            pad_file(alignment);
            for (int64_t n = 0; n < N; ++n) {
                if (tensor->type == GGML_TYPE_F32) {
                    const auto* src_row =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(tensor->data) + n * tensor->nb[1]);
                    QuantizeRowToFP8(src_row, row_fp8.data(), K, config.format);
                } else {
                    const auto* src_row = reinterpret_cast<const ggml_fp16_t*>(
                        reinterpret_cast<const char*>(tensor->data) + n * tensor->nb[1]);
                    ggml_fp16_to_fp32_row(src_row, row_f32.data(), K);
                    QuantizeRowToFP8(row_f32.data(), row_fp8.data(), K, config.format);
                }
                fwrite(row_fp8.data(), 1, static_cast<size_t>(K), f);
            }

            std::cout << "  Wrote FP8 tensor " << name << " (" << (K * N) << " bytes)" << std::endl;
        } else {
            pad_file(alignment);
            size_t size = ggml_nbytes(tensor);
            fwrite(tensor->data, 1, size, f);
        }
    }

    pad_file(alignment);

    fclose(f);
    gguf_free(ctx_out);
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    ggml_free(ctx_in);

    std::cout << std::endl;
    std::cout << "[Quantize] Done! Quantized " << quantized_count << "/" << n_tensors << " tensors to "
              << quant_format_name << "." << std::endl;
    std::cout << "[Quantize] Output saved to " << output_path << std::endl;
    std::cout << std::endl;
    std::cout << "FP8 tensors saved in-place with metadata:" << std::endl;
    std::cout << "  densecore.fp8.{name}.format = " << format_tag << std::endl;
    std::cout << "  densecore.fp8.{name}.K / .N" << std::endl;

    return 0;
}

// ============================================================================
// Custom INT4 Quantization Path (DenseCore Quantizer)
// ============================================================================

/**
 * Check if a tensor has been quantized to INT4 format.
 */
inline bool IsINT4Quantized(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->extra) return false;
    const TensorInt4* int4 = static_cast<const TensorInt4*>(tensor->extra);
    return (int4 && int4->q_data && int4->scales && int4->zero_points);
}

/**
 * Quantize model using DenseCore's custom INT4_BLOCKWISE format.
 *
 * Uses the "Split Tensor" approach for GGUF serialization:
 * - {tensor_name}        - Packed INT4 weights (raw bytes)
 * - {tensor_name}_scales - Per-block scale factors (FP32)
 * - {tensor_name}_zeros  - Per-block zero points (FP32)
 * - {tensor_name}_meta   - Metadata (group_size, num_blocks) as KV
 */
int QuantizeINT4Custom(const char* input_path, const char* output_path, const QuantConfig& config) {
    std::cout << "[Quantize] Loading model '" << input_path << "'..." << std::endl;

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = nullptr,
    };

    struct ggml_context* ctx_in = nullptr;
    params.ctx = &ctx_in;

    struct gguf_context* ctx_gguf = gguf_init_from_file(input_path, params);
    if (!ctx_gguf) {
        std::cerr << "Error: Failed to load GGUF file: " << input_path << std::endl;
        return 1;
    }

    std::cout << "[Quantize] Model loaded. Using custom INT4 quantization " << "(block_size=" << config.block_size
              << ")..." << std::endl;

    // Create the quantizer
    std::unique_ptr<Quantizer> quantizer;
    try {
        quantizer = CreateQuantizer(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to create quantizer: " << e.what() << std::endl;
        gguf_free(ctx_gguf);
        ggml_free(ctx_in);
        return 1;
    }

    // Create output GGUF context
    struct gguf_context* ctx_out = gguf_init_empty();

    // Copy KV pairs (Metadata) from input
    int n_kv = gguf_get_n_kv(ctx_gguf);
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx_gguf, i);
        gguf_type type = gguf_get_kv_type(ctx_gguf, i);

        if (type == GGUF_TYPE_ARRAY) {
            gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
            int n = gguf_get_arr_n(ctx_gguf, i);
            const void* data = gguf_get_arr_data(ctx_gguf, i);
            gguf_set_arr_data(ctx_out, key, arr_type, data, n);
        } else if (type == GGUF_TYPE_STRING) {
            const char* val = gguf_get_val_str(ctx_gguf, i);
            gguf_set_val_str(ctx_out, key, val);
        } else {
            switch (type) {
            case GGUF_TYPE_UINT8: gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_gguf, i)); break;
            case GGUF_TYPE_INT8: gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT16: gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_gguf, i)); break;
            case GGUF_TYPE_INT16: gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT32: gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_gguf, i)); break;
            case GGUF_TYPE_INT32: gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_gguf, i)); break;
            case GGUF_TYPE_UINT64: gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_gguf, i)); break;
            case GGUF_TYPE_INT64: gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_gguf, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_gguf, i)); break;
            case GGUF_TYPE_BOOL: gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_gguf, i)); break;
            default: break;
            }
        }
    }

    // Add INT4 format metadata
    gguf_set_val_str(ctx_out, "densecore.quantization_format", "int4_blockwise");
    gguf_set_val_u32(ctx_out, "densecore.block_size", config.block_size);

    // Pass 1: Quantize tensors and collect info
    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    std::vector<std::string> tensor_names;
    std::vector<bool> is_quantized;
    int quantized_count = 0;

    std::cout << "[Quantize] Processing " << n_tensors << " tensors..." << std::endl;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);

        if (!tensor) {
            std::cerr << "Warning: Tensor not found: " << name << std::endl;
            tensor_names.push_back(name);
            is_quantized.push_back(false);
            continue;
        }

        bool should_quantize = ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2;

        if (should_quantize) {
            std::cout << "  Quantizing " << name << " [" << tensor->ne[0] << "x" << tensor->ne[1] << "]..."
                      << std::endl;

            quantizer->QuantizeWeight(tensor);
            quantized_count++;
        }

        tensor_names.push_back(name);
        is_quantized.push_back(should_quantize && IsINT4Quantized(tensor));
    }

    // Allocate metadata-only context for output tensor definitions.
    const size_t out_tensor_slots = static_cast<size_t>(n_tensors) + static_cast<size_t>(quantized_count) * 2;
    struct ggml_context* ctx_meta = CreateTensorMetaContext(out_tensor_slots);
    if (!ctx_meta) {
        std::cerr << "Error: Failed to initialize output tensor metadata context." << std::endl;
        gguf_free(ctx_out);
        gguf_free(ctx_gguf);
        for (int i = 0; i < n_tensors; ++i) {
            struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, tensor_names[i].c_str());
            if (tensor && is_quantized[i]) {
                INT4Quantizer::FreeINT4Data(tensor);
            }
        }
        ggml_free(ctx_in);
        return 1;
    }

    // Pass 2: Add tensor definitions to GGUF context
    // For INT4 tensors, we add 3 tensors: qweight, scales, zeros
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = tensor_names[i].c_str();
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);

        if (!tensor) continue;

        if (is_quantized[i] && IsINT4Quantized(tensor)) {
            const TensorInt4* int4 = static_cast<const TensorInt4*>(tensor->extra);

            // Calculate sizes
            int64_t K = int4->ne[0];  // cols (inner dim)
            int64_t N = int4->ne[1];  // rows (output dim)
            int num_groups = K / int4->group_size;
            int64_t packed_size = N * ((K + 1) / 2);  // Packed INT4 bytes

            // 1. Packed INT4 weights: [N, K/2] as UINT8
            // We store as 1D for simplicity (can't use UINT4 in GGML)
            struct ggml_tensor* qw = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_I8, packed_size);
            std::string qw_name = std::string(name);
            ggml_set_name(qw, qw_name.c_str());
            gguf_add_tensor(ctx_out, qw);

            // 2. Scales: [N * num_groups] as FP32
            int64_t meta_size = N * num_groups;
            struct ggml_tensor* scales_t = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_F32, meta_size);
            std::string scales_name = std::string(name) + "_scales";
            ggml_set_name(scales_t, scales_name.c_str());
            gguf_add_tensor(ctx_out, scales_t);

            // 3. Zeros: [N * num_groups] as FP32
            struct ggml_tensor* zeros_t = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_F32, meta_size);
            std::string zeros_name = std::string(name) + "_zeros";
            ggml_set_name(zeros_t, zeros_name.c_str());
            gguf_add_tensor(ctx_out, zeros_t);

            // Add per-tensor metadata as KV
            std::string meta_key = std::string("densecore.int4.") + name;
            gguf_set_val_u32(ctx_out, (meta_key + ".group_size").c_str(), int4->group_size);
            gguf_set_val_u32(ctx_out, (meta_key + ".num_blocks").c_str(), int4->num_blocks);
            gguf_set_val_i64(ctx_out, (meta_key + ".K").c_str(), K);
            gguf_set_val_i64(ctx_out, (meta_key + ".N").c_str(), N);
        } else {
            // Standard tensor - keep same descriptor (data written manually).
            struct ggml_tensor* t = CreateTensorMetaLike(ctx_meta, tensor, tensor->type, name);
            gguf_add_tensor(ctx_out, t);
        }
    }

    // Write header
    std::cout << "[Quantize] Writing header to " << output_path << "..." << std::endl;
    gguf_write_to_file(ctx_out, output_path, true);

    // Re-open for appending
    FILE* f = fopen(output_path, "ab");
    if (!f) {
        std::cerr << "Error: Failed to open output file for appending." << std::endl;
        gguf_free(ctx_out);
        gguf_free(ctx_gguf);
        ggml_free(ctx_in);
        return 1;
    }

    // Padding helper
    const size_t alignment = gguf_get_alignment(ctx_out);

    auto pad_file = [&](size_t alignment) {
        long pos = ftell(f);
        size_t padding = (alignment - (pos % alignment)) % alignment;
        if (padding > 0) {
            char buf[32] = {0};
            fwrite(buf, 1, padding, f);
        }
    };

    // Pass 3: Write tensor data
    std::cout << "[Quantize] Writing tensors..." << std::endl;

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = tensor_names[i].c_str();
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, name);

        if (!tensor) continue;

        if (is_quantized[i] && IsINT4Quantized(tensor)) {
            const TensorInt4* int4 = static_cast<const TensorInt4*>(tensor->extra);

            int64_t K = int4->ne[0];
            int64_t N = int4->ne[1];
            int num_groups = K / int4->group_size;
            int64_t packed_size = N * ((K + 1) / 2);
            int64_t meta_size = N * num_groups;

            // Write packed INT4 weights
            pad_file(alignment);
            fwrite(int4->q_data, 1, packed_size, f);

            // Write scales
            pad_file(alignment);
            fwrite(int4->scales, sizeof(float), meta_size, f);

            // Write zeros
            pad_file(alignment);
            fwrite(int4->zero_points, sizeof(float), meta_size, f);

            std::cout << "  Wrote INT4 tensor " << name << " (" << packed_size << " + " << meta_size * 4 * 2
                      << " bytes)" << std::endl;
        } else {
            // Standard tensor - write raw data
            pad_file(alignment);
            size_t size = ggml_nbytes(tensor);
            fwrite(tensor->data, 1, size, f);
        }
    }

    pad_file(alignment);

    fclose(f);

    gguf_free(ctx_out);
    gguf_free(ctx_gguf);

    // Free INT4 quantized data before freeing ggml context
    // (ggml_free does not manage tensor->extra allocations)
    for (int i = 0; i < n_tensors; ++i) {
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_in, tensor_names[i].c_str());
        if (tensor && is_quantized[i]) {
            INT4Quantizer::FreeINT4Data(tensor);
        }
    }
    ggml_free(ctx_in);
    ggml_free(ctx_meta);

    std::cout << std::endl;
    std::cout << "[Quantize] Done! Quantized " << quantized_count << "/" << n_tensors << " tensors to INT4_BLOCKWISE."
              << std::endl;
    std::cout << "[Quantize] Output saved to " << output_path << std::endl;
    std::cout << std::endl;
    std::cout << "INT4 tensors saved with split format:" << std::endl;
    std::cout << "  {name}        - Packed 4-bit weights" << std::endl;
    std::cout << "  {name}_scales - Per-block scale factors" << std::endl;
    std::cout << "  {name}_zeros  - Per-block zero points" << std::endl;

    return 0;
}

// ============================================================================
// Main Entry Point
// ============================================================================

void PrintUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <input.gguf> <output.gguf> [type] [block_size]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Quantization types:" << std::endl;
    std::cerr << "  q4_0        - 4-bit basic GGML quantization (default)" << std::endl;
    std::cerr << "  q4_k_m      - 4-bit K-quants medium (recommended for quality)" << std::endl;
    std::cerr << "  q5_k_m      - 5-bit K-quants medium (higher quality)" << std::endl;
    std::cerr << "  q8_0        - 8-bit symmetric (highest quality)" << std::endl;
    std::cerr << "  f16         - Keep FP16 (no quantization)" << std::endl;
    std::cerr << "  fp8_e4m3fn  - Custom FP8 E4M3FN (DenseCore metadata format)" << std::endl;
    std::cerr << "  fp8_e5m2    - Custom FP8 E5M2 (DenseCore metadata format)" << std::endl;
    std::cerr << "  int4_paper  - Custom INT4 (DenseCore optimized kernels)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Block size (for int4_paper only): 32, 64, or 128 (default: 128)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        PrintUsage(argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // Parse configuration
    QuantConfig config = ParseConfig(argc, argv);

    std::cout << "[Quantize] Format: " << config.GetFormatName() << std::endl;
    std::cout << "[Quantize] Algorithm: " << config.GetAlgorithmName() << std::endl;

    // Dispatch to appropriate quantization path
    if (config.IsCustomFormat()) {
        if (config.format == QuantFormat::INT4_BLOCKWISE) {
            return QuantizeINT4Custom(input_path, output_path, config);
        }
        if (config.format == QuantFormat::FP8_E4M3FN || config.format == QuantFormat::FP8_E5M2) {
            return QuantizeFP8Custom(input_path, output_path, config);
        }
        std::cerr << "Error: Unsupported custom format: " << config.GetFormatName() << std::endl;
        return 1;
    } else {
        return QuantizeGGML(input_path, output_path, config);
    }
}
