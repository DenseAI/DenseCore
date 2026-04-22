#include "densecore.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace {

enum class ApiQuantFormat {
    Q4_0,
    Q4_1,
    Q4_K_M,
    Q5_K_M,
    Q8_0,
    PASSTHROUGH,
    UNSUPPORTED,
};

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string Trim(std::string s) {
    const size_t first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

std::string ParseJsonStringValue(const std::string& json_lower, const char* key) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json_lower.find(token);
    if (key_pos == std::string::npos) {
        return {};
    }

    const size_t colon_pos = json_lower.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos) {
        return {};
    }

    const size_t quote_start = json_lower.find('"', colon_pos + 1);
    if (quote_start == std::string::npos) {
        return {};
    }

    const size_t quote_end = json_lower.find('"', quote_start + 1);
    if (quote_end == std::string::npos || quote_end <= quote_start + 1) {
        return {};
    }

    return json_lower.substr(quote_start + 1, quote_end - quote_start - 1);
}

bool ParseJsonBoolValue(const std::string& json_lower, const char* key, bool default_value) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json_lower.find(token);
    if (key_pos == std::string::npos) {
        return default_value;
    }

    size_t colon_pos = json_lower.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos) {
        return default_value;
    }

    size_t value_pos = colon_pos + 1;
    while (value_pos < json_lower.size() && (json_lower[value_pos] == ' ' || json_lower[value_pos] == '\t' ||
                                             json_lower[value_pos] == '\n' || json_lower[value_pos] == '\r')) {
        ++value_pos;
    }
    if (value_pos >= json_lower.size()) {
        return default_value;
    }

    if (json_lower.compare(value_pos, 4, "true") == 0 || json_lower.compare(value_pos, 1, "1") == 0) {
        return true;
    }
    if (json_lower.compare(value_pos, 5, "false") == 0 || json_lower.compare(value_pos, 1, "0") == 0) {
        return false;
    }
    return default_value;
}

ApiQuantFormat ParseQuantFormat(std::string format_or_json) {
    format_or_json = Trim(ToLower(std::move(format_or_json)));

    if (format_or_json.empty()) {
        return ApiQuantFormat::Q4_K_M;
    }

    if (format_or_json.find("q4_k_m") != std::string::npos || format_or_json == "q4_k") {
        return ApiQuantFormat::Q4_K_M;
    }
    if (format_or_json.find("q5_k_m") != std::string::npos || format_or_json == "q5_k") {
        return ApiQuantFormat::Q5_K_M;
    }
    if (format_or_json.find("q8_0") != std::string::npos || format_or_json.find("int8") != std::string::npos) {
        return ApiQuantFormat::Q8_0;
    }
    if (format_or_json.find("q4_1") != std::string::npos) {
        return ApiQuantFormat::Q4_1;
    }
    if (format_or_json.find("q4_0") != std::string::npos) {
        return ApiQuantFormat::Q4_0;
    }
    if (format_or_json.find("f16") != std::string::npos || format_or_json.find("fp16") != std::string::npos ||
        format_or_json.find("f32") != std::string::npos || format_or_json.find("fp32") != std::string::npos) {
        return ApiQuantFormat::PASSTHROUGH;
    }

    if (format_or_json.find("int4_blockwise") != std::string::npos) {
        // Fallback: map custom int4 request to widely-supported GGUF Q4_K_M.
        return ApiQuantFormat::Q4_K_M;
    }

    return ApiQuantFormat::UNSUPPORTED;
}

ggml_type ToGGMLType(ApiQuantFormat format) {
    switch (format) {
    case ApiQuantFormat::Q4_0: return GGML_TYPE_Q4_0;
    case ApiQuantFormat::Q4_1: return GGML_TYPE_Q4_1;
    case ApiQuantFormat::Q4_K_M: return GGML_TYPE_Q4_K;
    case ApiQuantFormat::Q5_K_M: return GGML_TYPE_Q5_K;
    case ApiQuantFormat::Q8_0: return GGML_TYPE_Q8_0;
    default: return GGML_TYPE_Q4_K;
    }
}

bool ShouldQuantizeTensor(const std::string& tensor_name, bool skip_output_layer, bool skip_embeddings) {
    const std::string name = ToLower(tensor_name);

    if (skip_embeddings) {
        if (name.find("tok_embed") != std::string::npos || name.find("token_embed") != std::string::npos ||
            name.find("wte") != std::string::npos || name.find("embeddings") != std::string::npos) {
            return false;
        }
    }

    if (skip_output_layer) {
        if (name.find("lm_head") != std::string::npos) {
            return false;
        }
        if (name.find("output") != std::string::npos && name.find("layer") == std::string::npos) {
            return false;
        }
    }

    if (name.find("weight") == std::string::npos) {
        return false;
    }
    if (name.find("norm") != std::string::npos || name.find("bias") != std::string::npos) {
        return false;
    }
    return true;
}

bool CopyFileBinary(const std::string& src_path, const std::string& dst_path) {
    if (src_path == dst_path) {
        return true;
    }

    std::ifstream src(src_path, std::ios::binary);
    if (!src.good()) {
        return false;
    }
    std::ofstream dst(dst_path, std::ios::binary | std::ios::trunc);
    if (!dst.good()) {
        return false;
    }

    dst << src.rdbuf();
    return src.good() && dst.good();
}

int QuantizeGGUF(const char* input_path, const char* output_path, ggml_type qtype, bool skip_output_layer,
                 bool skip_embeddings) {
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = nullptr;

    struct ggml_context* ctx_in = nullptr;
    params.ctx = &ctx_in;

    struct gguf_context* ctx_in_gguf = nullptr;
    struct gguf_context* ctx_out_gguf = nullptr;
    struct ggml_context* ctx_out_meta = nullptr;
    FILE* out_file = nullptr;

    auto cleanup = [&]() {
        if (out_file) {
            fclose(out_file);
            out_file = nullptr;
        }
        if (ctx_out_gguf) {
            gguf_free(ctx_out_gguf);
            ctx_out_gguf = nullptr;
        }
        if (ctx_in_gguf) {
            gguf_free(ctx_in_gguf);
            ctx_in_gguf = nullptr;
        }
        if (ctx_out_meta) {
            ggml_free(ctx_out_meta);
            ctx_out_meta = nullptr;
        }
        if (ctx_in) {
            ggml_free(ctx_in);
            ctx_in = nullptr;
        }
    };

    ctx_in_gguf = gguf_init_from_file(input_path, params);
    if (!ctx_in_gguf || !ctx_in) {
        cleanup();
        return -2;
    }

    ctx_out_gguf = gguf_init_empty();
    if (!ctx_out_gguf) {
        cleanup();
        return -4;
    }

    gguf_set_kv(ctx_out_gguf, ctx_in_gguf);

    const int n_tensors = gguf_get_n_tensors(ctx_in_gguf);
    std::vector<struct ggml_tensor*> out_tensors;
    std::vector<bool> quantized;
    out_tensors.reserve(n_tensors);
    quantized.reserve(n_tensors);

    struct ggml_init_params out_meta_params;
    out_meta_params.mem_size = (static_cast<size_t>(n_tensors) + 128) * ggml_tensor_overhead();
    out_meta_params.mem_buffer = nullptr;
    out_meta_params.no_alloc = true;
    ctx_out_meta = ggml_init(out_meta_params);
    if (!ctx_out_meta) {
        cleanup();
        return -4;
    }

    ggml_quantize_init(qtype);

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_in_gguf, i);
        struct ggml_tensor* in_tensor = ggml_get_tensor(ctx_in, name);
        if (!in_tensor) {
            cleanup();
            return -2;
        }

        const bool do_quantize = ShouldQuantizeTensor(name, skip_output_layer, skip_embeddings) &&
                                 ggml_n_dims(in_tensor) == 2 &&
                                 (in_tensor->type == GGML_TYPE_F32 || in_tensor->type == GGML_TYPE_F16);

        const ggml_type target_type = do_quantize ? qtype : in_tensor->type;
        struct ggml_tensor* out_tensor =
            ggml_new_tensor(ctx_out_meta, target_type, ggml_n_dims(in_tensor), in_tensor->ne);
        if (!out_tensor) {
            cleanup();
            return -4;
        }
        ggml_set_name(out_tensor, name);
        gguf_add_tensor(ctx_out_gguf, out_tensor);
        out_tensors.push_back(out_tensor);
        quantized.push_back(do_quantize);
    }

    if (!gguf_write_to_file(ctx_out_gguf, output_path, true)) {
        cleanup();
        return -4;
    }

    out_file = fopen(output_path, "ab");
    if (!out_file) {
        cleanup();
        return -4;
    }

    const size_t alignment = gguf_get_alignment(ctx_out_gguf);
    auto pad_file = [&](size_t align) {
        const long pos = ftell(out_file);
        const size_t padding = (align - (static_cast<size_t>(pos) % align)) % align;
        if (padding == 0) {
            return true;
        }
        char zeros[64] = {0};
        size_t remaining = padding;
        while (remaining > 0) {
            const size_t chunk = std::min(remaining, sizeof(zeros));
            if (fwrite(zeros, 1, chunk, out_file) != chunk) {
                return false;
            }
            remaining -= chunk;
        }
        return true;
    };

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx_in_gguf, i);
        struct ggml_tensor* in_tensor = ggml_get_tensor(ctx_in, name);
        if (!in_tensor) {
            cleanup();
            return -2;
        }

        if (!pad_file(alignment > 0 ? alignment : GGUF_DEFAULT_ALIGNMENT)) {
            cleanup();
            return -4;
        }

        if (quantized[i]) {
            const int64_t rows = in_tensor->ne[1];
            const int64_t cols = in_tensor->ne[0];
            const size_t row_size = ggml_row_size(out_tensors[i]->type, cols);
            const size_t data_size = row_size * static_cast<size_t>(rows);

            std::vector<uint8_t> qdata(data_size);
            std::vector<float> f32_data;
            const float* src_data = nullptr;

            if (in_tensor->type == GGML_TYPE_F32) {
                src_data = static_cast<const float*>(in_tensor->data);
            } else {
                const int64_t nelements = ggml_nelements(in_tensor);
                f32_data.resize(static_cast<size_t>(nelements));
                ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(in_tensor->data), f32_data.data(), nelements);
                src_data = f32_data.data();
            }

            ggml_quantize_chunk(out_tensors[i]->type, src_data, qdata.data(), 0, rows, cols, nullptr);
            if (fwrite(qdata.data(), 1, data_size, out_file) != data_size) {
                cleanup();
                return -4;
            }
        } else {
            const size_t size = ggml_nbytes(in_tensor);
            if (fwrite(in_tensor->data, 1, size, out_file) != size) {
                cleanup();
                return -4;
            }
        }
    }

    cleanup();
    return 0;
}

}  // namespace

int QuantizeModel(const char* model_path, const char* output_path, const char* config_json) {
    try {
        if (!model_path || !output_path || model_path[0] == '\0' || output_path[0] == '\0') {
            return -1;
        }

        const std::string input = model_path;
        const std::string output = output_path;

        std::ifstream input_check(input, std::ios::binary);
        if (!input_check.good()) {
            return -2;
        }

        const std::string config_lower = config_json ? ToLower(std::string(config_json)) : std::string();
        std::string format = ParseJsonStringValue(config_lower, "format");
        if (format.empty()) {
            // Backward-compatible fallback for raw shorthand (e.g., "q4_k_m")
            format = config_lower;
        }

        const ApiQuantFormat parsed_format = ParseQuantFormat(format);
        if (parsed_format == ApiQuantFormat::UNSUPPORTED) {
            return -3;
        }

        if (parsed_format == ApiQuantFormat::PASSTHROUGH) {
            return CopyFileBinary(input, output) ? 0 : -4;
        }

        const bool skip_output_layer = ParseJsonBoolValue(config_lower, "skip_output_layer", true);
        const bool skip_embeddings = ParseJsonBoolValue(config_lower, "skip_embeddings", true);

        return QuantizeGGUF(model_path, output_path, ToGGMLType(parsed_format), skip_output_layer, skip_embeddings);
    } catch (...) {
        return -5;
    }
}
