#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>

#include "densecore/utils/logging.h"

#include "densecore/hal/backend_registry.h"  // Backend abstraction layer

#include "apple_silicon.h"
#include "cpu_backend.h"
#include "densecore.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/hal/device_interface.h"  // For Device/Allocator
#include "densecore/hal/tensor.h"            // For Tensor struct
#include "embedding.h"
#include "engine_internal.h"
#include "hardware_topology.h"
#include "inference.h"
#include "kv_cache.h"
#include "model_loader.h"
#include "model_types.h"
#include "optimization_bridge.h"  // Runtime SIMD dispatch
#include "simd_ops.h"
#include "tokenizer.h"

#ifdef __APPLE__
static void ConfigureHybridScheduler(EngineState* state, TransformerModel* model) {
    if (!densecore::apple::IsAppleHybridEnabled()) {
        return;
    }

#ifdef DENSECORE_TEST_BUILD
    if (model && model->is_mock) {
        return;
    }
#endif

    state->hybrid_enabled = true;
    state->hybrid_scheduler = std::make_unique<densecore::HybridScheduler>();
    state->hybrid_scheduler->SetCpuBackend(&densecore::GetCpuBackend());

    if (densecore::MetalBackend::IsAvailable()) {
        state->hybrid_gpu_backend = std::make_unique<densecore::MetalBackend>();
        state->hybrid_scheduler->SetGpuBackend(state->hybrid_gpu_backend.get());
    }

    if (densecore::ANEBackend::IsAvailable()) {
        state->hybrid_ane_backend = std::make_unique<densecore::ANEBackend>();
        state->hybrid_scheduler->SetAneBackend(state->hybrid_ane_backend.get());
    }

    densecore::HybridScheduler::ModelConfig config = {};
    config.n_layers = model->hparams.n_layer;
    config.hidden_dim = model->hparams.n_embd;
    config.n_heads = model->hparams.n_head;
    config.n_kv_heads = model->hparams.n_head_kv;
    config.head_dim = model->hparams.n_embd_head_k > 0 ? model->hparams.n_embd_head_k
                                                       : (model->hparams.n_embd / model->hparams.n_head);
    config.vocab_size = model->hparams.n_vocab;
    config.max_seq_len = model->hparams.n_ctx;
    if (!model->layers.empty()) {
        config.use_qk_norm = model->layers[0].Get(model_keys::kAttnQNorm) != nullptr;
    } else {
        config.use_qk_norm = false;
    }
    config.intermediate_dim = config.hidden_dim * 4;
    if (!model->layers.empty()) {
        auto* w1 = model->layers[0].Get(model_keys::kFfnGate);
        if (w1) {
            config.intermediate_dim = static_cast<int>(w1->ne[1]);
        }
    }

    state->hybrid_scheduler->ProfileModel(config);

    if (state->hybrid_ane_backend) {
        const char* bootstrap_env = std::getenv("DENSECORE_ANE_BUCKET_BOOTSTRAP");
        const bool bootstrap_buckets =
            !bootstrap_env || bootstrap_env[0] == '\0' || std::strcmp(bootstrap_env, "0") != 0;
        if (bootstrap_buckets) {
            densecore::ANEBackend::TransformerLayerConfig ane_cfg = {};
            ane_cfg.hidden_dim = std::max(1, config.hidden_dim);
            ane_cfg.intermediate_dim = std::max(1, config.intermediate_dim);
            ane_cfg.n_heads = std::max(1, config.n_heads);
            ane_cfg.n_kv_heads = std::max(1, config.n_kv_heads);
            ane_cfg.head_dim = std::max(1, config.head_dim);
            ane_cfg.max_seq_len = std::max(1, config.max_seq_len);
            ane_cfg.rms_norm_eps = model->hparams.f_norm_rms_eps > 0.0f ? model->hparams.f_norm_rms_eps : 1e-5f;
            ane_cfg.use_qk_norm = config.use_qk_norm;

            const char* prefix_env = std::getenv("DENSECORE_ANE_LAYER_PREFIX");
            const std::string layer_prefix =
                (prefix_env && prefix_env[0] != '\0') ? std::string(prefix_env) : std::string("layer_0");

            state->hybrid_ane_backend->PrecompileBucketedModels(layer_prefix, ane_cfg, nullptr);

            // Prime the dynamic bucket path once so missing buckets enter
            // background compile queue and runtime fallback policy is active.
            std::vector<float> warm_input(static_cast<size_t>(ane_cfg.hidden_dim), 0.0f);
            std::vector<float> warm_output(static_cast<size_t>(ane_cfg.hidden_dim), 0.0f);
            std::vector<int> warm_pos(1, 0);
            state->hybrid_ane_backend->ExecuteTransformerLayerDynamic(layer_prefix, warm_input.data(),
                                                                      warm_output.data(), warm_pos.data(), 1, ane_cfg);
        }
    }
}
#endif

namespace {
namespace fs = std::filesystem;

int ReadEnvInt(const char* name, int default_value, bool* was_set = nullptr) {
    const char* env = std::getenv(name);
    if (!env || *env == '\0') {
        if (was_set) {
            *was_set = false;
        }
        return default_value;
    }

    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || value <= 0) {
        if (was_set) {
            *was_set = false;
        }
        return default_value;
    }

    if (was_set) {
        *was_set = true;
    }
    return static_cast<int>(value);
}

bool IsVerboseTokenTraceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_VERBOSE_TOKEN_TRACE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

int ResolveKVHeadDim(const TransformerModel* model) {
    if (!model) {
        return 0;
    }
    if (model->hparams.n_embd_head_k > 0) {
        return model->hparams.n_embd_head_k;
    }
    if (model->hparams.n_head <= 0) {
        return 0;
    }
    return model->hparams.n_embd / model->hparams.n_head;
}

int ResolveKVValueHeadDim(const TransformerModel* model) {
    if (!model) {
        return 0;
    }
    if (model->hparams.n_embd_head_v > 0) {
        return model->hparams.n_embd_head_v;
    }
    if (model->hparams.n_embd_head_k > 0) {
        return model->hparams.n_embd_head_k;
    }
    if (model->hparams.n_head <= 0) {
        return 0;
    }
    return model->hparams.n_embd / model->hparams.n_head;
}

int ResolveKVIndexHeadDim(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_glm_dsa || model->glm_index_head_dim <= 0) {
        return 0;
    }
    return model->glm_index_head_dim;
}

bool IsSupportedKVCacheType(ggml_type cache_type) {
    return cache_type == GGML_TYPE_F32 || cache_type == GGML_TYPE_F16 || cache_type == GGML_TYPE_Q8_0 ||
           cache_type == GGML_TYPE_Q4_0;
}

const char* KVCacheTypeDisplayName(ggml_type cache_type) {
    switch (cache_type) {
    case GGML_TYPE_F32: return "FP32";
    case GGML_TYPE_Q8_0: return "INT8 (Q8_0)";
    case GGML_TYPE_Q4_0: return "INT4 (Q4_0)";
    case GGML_TYPE_F16:
    default: return "FP16";
    }
}

}  // namespace

ggml_type ResolveEffectiveKVCacheType(const TransformerModel* model, ggml_type requested_cache_type) {
    if (!IsSupportedKVCacheType(requested_cache_type)) {
        return GGML_TYPE_F16;
    }

    if (!ggml_is_quantized(requested_cache_type)) {
        return requested_cache_type;
    }

    const int k_head_dim = ResolveKVHeadDim(model);
    const int v_head_dim = ResolveKVValueHeadDim(model);
    const int quant_block_size = ggml_blck_size(requested_cache_type);
    if (k_head_dim <= 0 || v_head_dim <= 0 || quant_block_size <= 0 || (k_head_dim % quant_block_size) != 0 ||
        (v_head_dim % quant_block_size) != 0) {
        return GGML_TYPE_F16;
    }

    return requested_cache_type;
}

size_t ComputeKVCacheBytesPerToken(ggml_type cache_type, int k_head_dim, int v_head_dim, int n_head_kv, int n_layer,
                                   int index_head_dim) {
    if (k_head_dim <= 0 || v_head_dim <= 0 || n_head_kv <= 0 || n_layer <= 0) {
        return 1;
    }

    if (ggml_is_quantized(cache_type) &&
        ((k_head_dim % ggml_blck_size(cache_type)) != 0 || (v_head_dim % ggml_blck_size(cache_type)) != 0)) {
        cache_type = GGML_TYPE_F16;
    }

    auto bytes_per_slot_for_dim = [&](int head_dim) -> size_t {
        if (cache_type == GGML_TYPE_Q8_0 || cache_type == GGML_TYPE_Q4_0) {
            return ggml_row_size(cache_type, static_cast<int64_t>(head_dim)) * static_cast<size_t>(n_head_kv);
        }
        return ggml_row_size(cache_type, static_cast<int64_t>(head_dim) * static_cast<int64_t>(n_head_kv));
    };

    const size_t k_bytes_per_slot = bytes_per_slot_for_dim(k_head_dim);
    const size_t v_bytes_per_slot = bytes_per_slot_for_dim(v_head_dim);
    const size_t index_bytes_per_slot =
        index_head_dim > 0 ? ggml_row_size(GGML_TYPE_F16, static_cast<int64_t>(index_head_dim)) : 0;

    return (k_bytes_per_slot + v_bytes_per_slot + index_bytes_per_slot) * static_cast<size_t>(n_layer);
}

KVCacheConfig ComputeKVCacheConfig(const TransformerModel* model, ggml_type requested_cache_type) {
    KVCacheConfig config;
    bool max_seq_len_env_set = false;

    config.requested_cache_type = requested_cache_type;
    config.effective_cache_type = ResolveEffectiveKVCacheType(model, requested_cache_type);
    config.max_num_seqs = ReadEnvInt("DENSECORE_MAX_NUM_SEQS", 4);
    config.max_seq_len = ReadEnvInt("DENSECORE_MAX_SEQ_LEN", 4096, &max_seq_len_env_set);

    int target_mb = ReadEnvInt("DENSECORE_KV_TARGET_MB", 512);
    config.target_kv_memory = static_cast<size_t>(target_mb) * 1024ULL * 1024ULL;

    const int k_head_dim = ResolveKVHeadDim(model);
    const int v_head_dim = ResolveKVValueHeadDim(model);
    const int index_head_dim = ResolveKVIndexHeadDim(model);
    const int n_head_kv = model ? model->hparams.n_head_kv : 0;
    const int n_layer = model ? model->hparams.n_layer : 0;

    config.bytes_per_token = ComputeKVCacheBytesPerToken(config.effective_cache_type, k_head_dim, v_head_dim, n_head_kv,
                                                         n_layer, index_head_dim);
    if (config.bytes_per_token == 0) {
        config.bytes_per_token = 1;
    }

    int optimal_seq_len = static_cast<int>(config.target_kv_memory / config.bytes_per_token);
    optimal_seq_len = std::max(256, std::min(optimal_seq_len, config.max_seq_len));

    if (!max_seq_len_env_set && config.bytes_per_token < 1024) {
        optimal_seq_len = std::min(optimal_seq_len * 2, 8192);
    }

    config.max_seq_len = optimal_seq_len;
    return config;
}

namespace {

void SetError(DenseCoreStatus status, const std::string& message) {
    SetLastError(status, message);
}

void ClearError() {
    ClearLastError();
}

std::string ReadEnvString(const char* name) {
    const char* env = std::getenv(name);
    if (!env || *env == '\0') {
        return {};
    }
    return std::string(env);
}

TransformerModel* LoadModelWithNuma(const char* model_path, int numa_node_id);

std::string TrimWhitespace(std::string value) {
    const size_t first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

bool IsPathWithinBase(const fs::path& candidate, const fs::path& base_dir) {
    const fs::path relative = candidate.lexically_relative(base_dir);
    if (relative.empty()) {
        return false;
    }
    auto it = relative.begin();
    if (it == relative.end()) {
        return true;
    }
    return *it != "..";
}

bool ContainsParentTraversalSegment(const fs::path& input_path) {
    for (const auto& component : input_path) {
        if (component == "..") {
            return true;
        }
    }
    return false;
}

bool ValidateModelPathArg(const std::string& path, const char* api_name, const char* arg_name,
                          std::string* canonical_out) {
    if (path.find('\0') != std::string::npos) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(api_name) + ": " + arg_name + " contains null byte");
        return false;
    }

    const std::string trimmed = TrimWhitespace(path);
    if (trimmed.empty()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " is empty or whitespace");
        return false;
    }

    const fs::path input_path(trimmed);
    if (ContainsParentTraversalSegment(input_path)) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " contains path traversal");
        return false;
    }

#ifdef DENSECORE_TEST_BUILD
    if (trimmed == "mock") {
        if (canonical_out) {
            *canonical_out = trimmed;
        }
        return true;
    }
#endif

    std::error_code ec;
    const fs::path canonical_path = fs::canonical(input_path, ec);
    if (ec) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " cannot be resolved to a canonical path");
        return false;
    }

    if (!fs::is_regular_file(canonical_path, ec) || ec) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                 std::string(api_name) + ": " + arg_name + " must point to an existing file");
        return false;
    }

    const std::string model_base_dir = TrimWhitespace(ReadEnvString("DENSECORE_MODEL_BASE_DIR"));
    if (!model_base_dir.empty()) {
        ec.clear();
        const fs::path canonical_base = fs::canonical(fs::path(model_base_dir), ec);
        if (ec || !fs::is_directory(canonical_base)) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                     std::string(api_name) + ": DENSECORE_MODEL_BASE_DIR is invalid");
            return false;
        }
        if (!IsPathWithinBase(canonical_path, canonical_base)) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                     std::string(api_name) + ": " + arg_name + " must be under DENSECORE_MODEL_BASE_DIR");
            return false;
        }
    }

    if (canonical_out) {
        *canonical_out = canonical_path.string();
    }
    return true;
}

std::string ResolveDraftModelPath(const char* reserved) {
    if (reserved && *reserved != '\0') {
        return TrimWhitespace(std::string(reserved));
    }

    std::string env_draft = TrimWhitespace(ReadEnvString("DRAFT_MODEL_PATH"));
    if (!env_draft.empty()) {
        return env_draft;
    }

    return TrimWhitespace(ReadEnvString("DENSECORE_DRAFT_MODEL_PATH"));
}

bool LoadOptionalDraftModel(EngineState* state, const std::string& main_model_path, const std::string& draft_model_path,
                            ggml_type cache_type, const char* api_name) {
    if (!state || draft_model_path.empty()) {
        return true;
    }

    if (draft_model_path == main_model_path) {
        state->draft_model_id = "default";
        state->draft_model_path = draft_model_path;
        LOG_WARN("Draft model path equals main model path; reusing default model");
        return true;
    }

    TransformerModel* draft_model = LoadModelWithNuma(draft_model_path.c_str(), state->numa_node_id);
    if (!draft_model) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, std::string(api_name) + ": failed to load draft model");
        return false;
    }

    KVCacheConfig draft_kv_config = ComputeKVCacheConfig(draft_model, cache_type);
    if (draft_kv_config.effective_cache_type != draft_kv_config.requested_cache_type) {
        LOG_WARN("Draft model KV cache request {} is incompatible with head_dim={}, using {} instead",
                 KVCacheTypeDisplayName(draft_kv_config.requested_cache_type), ResolveKVHeadDim(draft_model),
                 KVCacheTypeDisplayName(draft_kv_config.effective_cache_type));
    }
    PagedKVCache* draft_cache = InitPagedKVCache(draft_model, draft_kv_config.max_num_seqs, draft_kv_config.max_seq_len,
                                                 draft_kv_config.effective_cache_type, state->numa_node_id);
    if (!draft_cache) {
        delete draft_model;
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, std::string(api_name) + ": failed to initialize draft KV cache");
        return false;
    }

    auto draft_entry = std::make_unique<ModelEntry>();
    draft_entry->model_id = "draft";
    draft_entry->model_path = draft_model_path;
    draft_entry->model = std::unique_ptr<TransformerModel>(draft_model);
    draft_entry->kv_cache = std::unique_ptr<PagedKVCache>(draft_cache);
    draft_entry->last_used = std::chrono::steady_clock::now();
    draft_entry->is_loaded = true;

    state->models["draft"] = std::move(draft_entry);
    state->draft_model_id = "draft";
    state->draft_model_path = draft_model_path;
    LOG_INFO("Draft model loaded: {}", draft_model_path);

    return true;
}

std::string ToLower(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool ParseBoolEnv(const char* name, bool default_value) {
    std::string value = ToLower(ReadEnvString(name));
    if (value.empty()) {
        return default_value;
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

void ApplyActionTokenRangeFromEnv(SamplingParams* params) {
    if (!params) return;

    const int action_token_count = ReadEnvInt("DENSECORE_ACTION_TOKEN_COUNT", 0);
    if (action_token_count <= 0) {
        params->action_token_count = 0;
        params->action_token_start = 0;
        return;
    }

    const int action_token_start = std::max(0, ReadEnvInt("DENSECORE_ACTION_TOKEN_START", 0));
    params->action_token_count = action_token_count;
    params->action_token_start = action_token_start;
}

bool HasTokenLiteral(const TransformerModel* model, const char* token) {
    if (!model || !token) return false;
    return model->token_to_id.find(token) != model->token_to_id.end();
}

bool ShouldPrimeQwenNoThinking(const TransformerModel* model) {
    if (!model) return false;
    if (model->arch == ModelArch::QWEN3) {
        return !ParseBoolEnv("DENSECORE_QWEN3_ENABLE_THINKING", true);
    }
    if (model->arch == ModelArch::QWEN35) {
        return !ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true);
    }
    return false;
}

void MaybePrimeQwenNoThinking(const TransformerModel* model, std::vector<int>* tokens) {
    if (!model || !tokens || !ShouldPrimeQwenNoThinking(model)) {
        return;
    }
    const std::vector<int> suffix = Tokenizer::Tokenize(model, "Answer: ", /*add_bos=*/false, /*add_eos=*/false);
    if (suffix.empty()) {
        return;
    }
    if (tokens->size() >= suffix.size()) {
        bool already_primed = true;
        const size_t start = tokens->size() - suffix.size();
        for (size_t i = 0; i < suffix.size(); ++i) {
            if ((*tokens)[start + i] != suffix[i]) {
                already_primed = false;
                break;
            }
        }
        if (already_primed) {
            return;
        }
    }
    tokens->insert(tokens->end(), suffix.begin(), suffix.end());
}

void ConfigureQwenReasoningTokenBlocklist(const TransformerModel* model, Request* req) {
    if (!model || !req) return;
    req->disallowed_token_ids.clear();
    if (!ShouldPrimeQwenNoThinking(model)) {
        return;
    }

    if (model->token_types.size() == model->vocab_tokens.size()) {
        for (size_t i = 0; i < model->token_types.size(); ++i) {
            if (model->token_types[i] == 3 && static_cast<int>(i) != model->eos_token_id) {
                req->disallowed_token_ids.push_back(static_cast<int>(i));
            }
        }
    } else {
        static const char* kBlockedLiterals[] = {"<think>", "</think>", "<|im_start|>", "<|im_end|>", nullptr};
        for (int i = 0; kBlockedLiterals[i] != nullptr; ++i) {
            auto it = model->token_to_id.find(kBlockedLiterals[i]);
            if (it != model->token_to_id.end()) {
                req->disallowed_token_ids.push_back(it->second);
            }
        }
    }
    std::sort(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end());
    req->disallowed_token_ids.erase(std::unique(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end()),
                                    req->disallowed_token_ids.end());
}

bool PromptAlreadyTemplated(const std::string& prompt) {
    return prompt.find("<|im_start|>") != std::string::npos || prompt.find("<|im_end|>") != std::string::npos ||
           prompt.find("<|assistant|>") != std::string::npos || prompt.find("<|user|>") != std::string::npos ||
           prompt.find("<|turn>") != std::string::npos || prompt.find("<turn|>") != std::string::npos;
}

void DebugPrintPromptTokens(const TransformerModel* model, const std::vector<int>& tokens, const char* tag) {
    if (std::getenv("DENSECORE_DEBUG_PROMPT_TOKENS") == nullptr) {
        return;
    }
    fprintf(stderr, "[PROMPT_TOKENS] %s count=%zu\n", tag ? tag : "prompt", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int tok = tokens[i];
        std::string piece = Tokenizer::Detokenize(model, tok);
        for (char& ch : piece) {
            if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        }
        fprintf(stderr, "  [%zu] %d '%s'\n", i, tok, piece.c_str());
    }
}

std::string MaybeApplyAutoChatTemplate(const TransformerModel* model, const std::string& prompt) {
    if (!model || prompt.empty()) {
        return prompt;
    }
    if (!ParseBoolEnv("DENSECORE_AUTO_CHAT_TEMPLATE", true)) {
        return prompt;
    }
    if (PromptAlreadyTemplated(prompt)) {
        return prompt;
    }

    const bool has_chatml = HasTokenLiteral(model, "<|im_start|>") && HasTokenLiteral(model, "<|im_end|>");
    const bool has_role_tokens = HasTokenLiteral(model, "<|user|>") && HasTokenLiteral(model, "<|assistant|>");
    const bool has_gemma_turn_tokens = HasTokenLiteral(model, "<|turn>") && HasTokenLiteral(model, "<turn|>");
    const bool template_looks_chatml = model->chat_template.find("<|im_start|>") != std::string::npos;
    const bool template_looks_gemma_turn = model->chat_template.find("<|turn>") != std::string::npos &&
                                           model->chat_template.find("<turn|>") != std::string::npos;
    const bool prefer_chat_template =
        model->arch == ModelArch::QWEN2 || model->arch == ModelArch::QWEN3 || model->arch == ModelArch::QWEN35;

    if (has_chatml || template_looks_chatml || prefer_chat_template) {
        const bool qwen_enable_thinking =
            (model->arch == ModelArch::QWEN3)    ? ParseBoolEnv("DENSECORE_QWEN3_ENABLE_THINKING", true)
            : (model->arch == ModelArch::QWEN35) ? ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true)
                                                 : true;
        std::string wrapped;
        wrapped.reserve(prompt.size() + 192);
        wrapped += "<|im_start|>system\nYou are a helpful assistant.";
        if ((model->arch == ModelArch::QWEN3 || model->arch == ModelArch::QWEN35) && !qwen_enable_thinking) {
            wrapped += "\nProvide only the answer. Do not output any thinking process, analysis, reasoning steps, "
                       "or preamble. Never start with 'Thinking Process'.";
        }
        wrapped += "<|im_end|>\n";
        wrapped += "<|im_start|>user\n";
        wrapped += prompt;
        wrapped += "<|im_end|>\n";
        wrapped += "<|im_start|>assistant\n";
        if ((model->arch == ModelArch::QWEN3 || model->arch == ModelArch::QWEN35) && !qwen_enable_thinking) {
            wrapped += "Answer: ";
        }
        return wrapped;
    }

    if (has_role_tokens) {
        std::string wrapped;
        wrapped.reserve(prompt.size() + 96);
        wrapped += "<|system|>\nYou are a helpful assistant.\n";
        wrapped += "<|user|>\n";
        wrapped += prompt;
        wrapped += "\n<|assistant|>\n";
        return wrapped;
    }

    if (has_gemma_turn_tokens || template_looks_gemma_turn) {
        std::string wrapped;
        wrapped.reserve(prompt.size() + 48);
        wrapped += "<|turn>user\n";
        wrapped += prompt;
        wrapped += "<turn|>\n<|turn>model\n";
        return wrapped;
    }

    return prompt;
}

TransformerModel* LoadModelWithNuma(const char* model_path, int numa_node_id) {
    const std::string mode_raw = ToLower(ReadEnvString("DENSECORE_NUMA_WEIGHTS"));
    const bool use_huge_pages = ParseBoolEnv("DENSECORE_NUMA_HUGEPAGES", false);

    if (mode_raw.empty()) {
        if (numa_node_id >= 0) {
            std::cout << "[DenseCore] NUMA weights: pinned to node " << numa_node_id << std::endl;
            return LoadGGUFModelNuma(model_path, numa_node_id, use_huge_pages);
        }
        return LoadGGUFModel(model_path);
    }

    if (mode_raw == "off" || mode_raw == "0" || mode_raw == "false") {
        return LoadGGUFModel(model_path);
    }

    if (mode_raw == "auto" || mode_raw == "interleaved") {
        std::cout << "[DenseCore] NUMA weights: interleaved" << std::endl;
        return LoadGGUFModelNuma(model_path, -1, use_huge_pages);
    }

    if (mode_raw == "round_robin" || mode_raw == "round-robin" || mode_raw == "rr") {
        std::cout << "[DenseCore] NUMA weights: round-robin" << std::endl;
        return LoadGGUFModelNuma(model_path, -2, use_huge_pages);
    }

    if (mode_raw == "pinned" || mode_raw == "pin") {
        const int target_node = (numa_node_id >= 0) ? numa_node_id : 0;
        std::cout << "[DenseCore] NUMA weights: pinned to node " << target_node << std::endl;
        return LoadGGUFModelNuma(model_path, target_node, use_huge_pages);
    }

    char* end = nullptr;
    long node = std::strtol(mode_raw.c_str(), &end, 10);
    if (end != mode_raw.c_str() && *end == '\0' && node >= 0) {
        std::cout << "[DenseCore] NUMA weights: pinned to node " << node << std::endl;
        return LoadGGUFModelNuma(model_path, static_cast<int>(node), use_huge_pages);
    }

    std::cerr << "[DenseCore] Unknown DENSECORE_NUMA_WEIGHTS mode '" << mode_raw << "', using default loader"
              << std::endl;
    return LoadGGUFModel(model_path);
}
}  // namespace

// Global request ID counter (shared across all Submit* functions)
static std::atomic<int> global_req_id{1};

namespace {

Request* AcquireAndInitRequest(EngineState* state) {
    Request* req = state->request_pool.Acquire();
    req->Reset();
    req->id = global_req_id.fetch_add(1);
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}

void SetupJsonGrammar(Request* req, EngineState* state) {
    if (!req->json_mode) return;

    ModelEntry* model_entry = state->GetDefaultModel();
    if (model_entry && model_entry->model && !model_entry->model->vocab_tokens.empty()) {
        req->grammar.enabled = true;
        req->grammar.is_json_mode = true;
        req->grammar.state = JSONState::EXPECT_OBJECT_START;
        InitGrammarConstraint(&req->grammar, model_entry->model->vocab_tokens);
    } else {
        std::cerr << "[DenseCore] Warning: JSON mode requested but model vocab not available" << std::endl;
    }
}

void AssignGenerationTier(Request* req) {
    size_t n_tokens = req->tokens.size();
    if (n_tokens < 100) {
        req->tier = "premium";
    } else if (n_tokens < 500) {
        req->tier = "standard";
    } else {
        req->tier = "batch";
    }
}

void EnqueueRequest(EngineState* state, Request* req) {
    state->pending_requests.Push(req, req->tier);
    state->RecordPendingRequest(req->id);
    {
        std::lock_guard<std::mutex> lock(state->cv_mu);
        state->queue_cv.notify_one();
    }
}

void ApplyDefaultLora(EngineState* state, Request* req) {
    if (!req || !req->lora_name.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(state->lora_mu);
    if (!state->default_lora_adapter.empty()) {
        req->lora_name = state->default_lora_adapter;
    }
}

void InitCommonRequest(EngineState* state, Request* req, int max_tokens, float temperature, float top_p, int top_k,
                       float repetition_penalty, const char** stop_sequences, int json_mode, TokenCallback callback,
                       void* user_data) {
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->user_data = user_data;
    req->json_mode = (json_mode != 0);

    // Apply default LoRA adapter (backward compatibility)
    ApplyDefaultLora(state, req);

    // Set sampling parameters
    req->sampling_params.temperature = temperature;
    req->sampling_params.top_p = top_p;
    req->sampling_params.top_k = top_k;
    req->sampling_params.repetition_penalty = repetition_penalty;
    ApplyActionTokenRangeFromEnv(&req->sampling_params);

    if (stop_sequences) {
        for (int i = 0; stop_sequences[i] != nullptr; ++i) {
            req->stop_sequences.push_back(std::string(stop_sequences[i]));
        }
    }

    if (!req->stop_sequences.empty()) {
        for (const auto& seq : req->stop_sequences) {
            req->stop_buffer_max = std::max(req->stop_buffer_max, seq.size());
        }
    }

    // Initialize grammar constraint if JSON mode is enabled
    SetupJsonGrammar(req, state);
}
}  // namespace

// API implementations

int SubmitEmbeddingRequest(DenseCoreHandle handle, const char* prompt, EmbeddingCallback callback, void* user_data) {
    // Default: MEAN pooling with normalization
    return SubmitEmbeddingRequestEx(handle, prompt, 0, 1, callback, user_data);
}

int SubmitEmbeddingRequestEx(DenseCoreHandle handle, const char* prompt, int pooling_type, int normalize,
                             EmbeddingCallback callback, void* user_data) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitEmbeddingRequestEx: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    if (!prompt) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitEmbeddingRequestEx: prompt is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    // Get model for tokenization
    ModelEntry* model_entry = state->GetDefaultModel();
    if (!model_entry || !model_entry->model) {
        std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitEmbeddingRequestEx: no model loaded for tokenization");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    Request* req = AcquireAndInitRequest(state);

    req->prompt = prompt;
    req->max_tokens = 0;  // No generation
    req->is_embedding = true;
    req->embedding_callback = callback;
    req->user_data = user_data;

    // Store pooling config
    req->pooling_type = static_cast<densecore::PoolingStrategy>(pooling_type);
    req->normalize_embedding = (normalize != 0);

    // Tokenize immediately (outside hot path)
    req->tokens = Tokenizer::Tokenize(model_entry->model.get(), prompt, true);

    // Embeddings get premium tier explicitly
    req->priority = 50;
    req->tier = "premium";

    EnqueueRequest(state, req);

    ClearError();
    return req->id;
}

int SubmitRequestWithSampling(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature,
                              float top_p, int top_k, float repetition_penalty, const char** stop_sequences,
                              int json_mode, TokenCallback callback, void* user_data) {
    return SubmitRequestWithSamplingEx(handle, prompt, max_tokens, nullptr, temperature, top_p, top_k,
                                       repetition_penalty, stop_sequences, json_mode, callback, user_data);
}

int SubmitRequestWithSamplingEx(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                                float temperature, float top_p, int top_k, float repetition_penalty,
                                const char** stop_sequences, int json_mode, TokenCallback callback, void* user_data) {
    if (!handle || !prompt) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithSamplingEx: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    ModelEntry* model_entry = state->GetDefaultModel();
    if (!model_entry || !model_entry->is_loaded) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequestWithSamplingEx: default model not loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    Request* req = AcquireAndInitRequest(state);
    req->lora_name = lora_name ? lora_name : "";

    InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences, json_mode,
                      callback, user_data);

    // Tokenize prompt
    req->prompt = MaybeApplyAutoChatTemplate(model_entry->model.get(), prompt);
    req->tokens = Tokenizer::Tokenize(model_entry->model.get(), req->prompt, true);
    MaybePrimeQwenNoThinking(model_entry->model.get(), &req->tokens);
    ConfigureQwenReasoningTokenBlocklist(model_entry->model.get(), req);
    DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "sampling");
    req->token_history = req->tokens;

    AssignGenerationTier(req);
    EnqueueRequest(state, req);
    ClearError();
    return req->id;
}

int SubmitRequestWithTokenResults(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature,
                                  float top_p, int top_k, float repetition_penalty, TokenResultCallback callback,
                                  void* user_data) {
    if (!handle || !prompt) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithTokenResults: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    ModelEntry* model_entry = state->GetDefaultModel();
    if (!model_entry || !model_entry->is_loaded) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequestWithTokenResults: default model not loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    Request* req = AcquireAndInitRequest(state);
    req->token_result_callback = callback;

    InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty,
                      /*stop_sequences=*/nullptr, /*json_mode=*/0,
                      /*callback=*/nullptr, user_data);

    req->prompt = prompt;
    req->tokens = Tokenizer::Tokenize(model_entry->model.get(), req->prompt, true);
    MaybePrimeQwenNoThinking(model_entry->model.get(), &req->tokens);
    ConfigureQwenReasoningTokenBlocklist(model_entry->model.get(), req);
    DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "token_results");
    req->token_history = req->tokens;

    AssignGenerationTier(req);
    EnqueueRequest(state, req);
    ClearError();
    return req->id;
}

int SubmitRequestIdsWithSampling(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                 float temperature, float top_p, int top_k, float repetition_penalty,
                                 const char** stop_sequences, int json_mode, TokenCallback callback, void* user_data) {
    return SubmitRequestIdsWithSamplingEx(handle, tokens, n_tokens, max_tokens, nullptr, temperature, top_p, top_k,
                                          repetition_penalty, stop_sequences, json_mode, callback, user_data);
}

int SubmitRequestIdsWithSamplingEx(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                   const char* lora_name, float temperature, float top_p, int top_k,
                                   float repetition_penalty, const char** stop_sequences, int json_mode,
                                   TokenCallback callback, void* user_data) {
    if (!handle || !tokens || n_tokens <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestIdsWithSamplingEx: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    Request* req = AcquireAndInitRequest(state);
    req->lora_name = lora_name ? lora_name : "";

    InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences, json_mode,
                      callback, user_data);

    // Assign pre-tokenized input
    req->tokens.assign(tokens, tokens + n_tokens);
    ModelEntry* model_entry = state->GetDefaultModel();
    if (model_entry && model_entry->model) {
        MaybePrimeQwenNoThinking(model_entry->model.get(), &req->tokens);
        ConfigureQwenReasoningTokenBlocklist(model_entry->model.get(), req);
        DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "ids_sampling");
    }
    req->token_history = req->tokens;

    AssignGenerationTier(req);
    EnqueueRequest(state, req);
    ClearError();
    return req->id;
}

int SubmitBatchEmbeddingRequest(DenseCoreHandle handle, const char** prompts, int num_prompts, int pooling_type,
                                int normalize, EmbeddingCallback callback, void* user_data) {
    if (!handle || !prompts || num_prompts <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitBatchEmbeddingRequest: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    // Submit each prompt as a separate request (batching happens in worker)
    int first_id = -1;
    for (int i = 0; i < num_prompts; i++) {
        int id = SubmitEmbeddingRequestEx(handle, prompts[i], pooling_type, normalize, callback, user_data);
        if (i == 0) first_id = id;
    }

    if (first_id > 0) {
        ClearError();
    }
    return first_id;
}

int GetEmbeddingDimension(DenseCoreHandle handle) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "GetEmbeddingDimension: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        ClearError();
        return entry->model->hparams.n_embd;
    }
    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "GetEmbeddingDimension: no model loaded");
    return DENSECORE_STATUS_MODEL_LOAD_FAILED;
}

int GetMaxContextTokens(DenseCoreHandle handle) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "GetMaxContextTokens: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        ClearError();
        return static_cast<int>(entry->model->hparams.n_ctx);
    }
    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "GetMaxContextTokens: no model loaded");
    return DENSECORE_STATUS_MODEL_LOAD_FAILED;
}

int CountTokens(DenseCoreHandle handle, const char* text, int add_bos, int add_eos) {
    if (!handle || !text) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CountTokens: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        const std::vector<int> tokens = Tokenizer::Tokenize(entry->model.get(), text, add_bos != 0, add_eos != 0);
        ClearError();
        return static_cast<int>(tokens.size());
    }

    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "CountTokens: no model loaded");
    return DENSECORE_STATUS_MODEL_LOAD_FAILED;
}

/**
 * @brief Initialize engine with default NUMA settings (simplified API)
 *
 * This is a convenience wrapper that calls InitEngineEx with:
 * - numa_node_id = -1 (auto-detect)
 * - pinning_policy = 0 (SCATTER)
 */
DENSECORE_API DenseCoreHandle InitEngine(const char* model_path, const char* reserved, int threads) {
    return InitEngineEx(model_path, reserved, threads, -1, 0);
}

/**
 * @brief Initialize engine with NUMA and thread pinning control (extended API)
 */
DENSECORE_API DenseCoreHandle InitEngineEx(const char* model_path, const char* reserved, int threads, int numa_node_id,
                                           int pinning_policy) {
    try {
        ClearError();
        // Initialize logging subsystem (once)
        static std::once_flag log_init_flag;
        std::call_once(log_init_flag, []() { densecore::utils::InitLogging(); });

        if (!model_path) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "InitEngineEx: model_path is null");
            return nullptr;
        }

        std::string canonical_model_path;
        if (!ValidateModelPathArg(model_path, "InitEngineEx", "model_path", &canonical_model_path)) {
            return nullptr;
        }
        const std::string draft_model_path = ResolveDraftModelPath(reserved);
        std::string canonical_draft_model_path;
        if (!draft_model_path.empty() &&
            !ValidateModelPathArg(draft_model_path, "InitEngineEx", "draft_model_path", &canonical_draft_model_path)) {
            return nullptr;
        }

        int init_delay_ms = 0;
        if (const char* env_val = std::getenv("DENSECORE_INIT_DELAY_MS")) {
            init_delay_ms = std::atoi(env_val);
            if (init_delay_ms > 0) {
                LOG_INFO("Init delay: {}ms (large model tolerance)", init_delay_ms);
                std::this_thread::sleep_for(std::chrono::milliseconds(init_delay_ms));
            }
        }

        // =========================================================================
        // THREAD CONFIGURATION (Pure std::thread + GGML thread pool)
        // =========================================================================
        // Auto-detect optimal thread count if not specified. Prefer physical
        // cores for SIMD-heavy inference and fall back to logical cores only
        // when topology information is unavailable.
        if (threads <= 0) {
            auto& topo = densecore::HardwareTopology::GetInstance();
            threads = topo.GetPhysicalCoreCount();
            if (threads <= 0) {
                threads = topo.GetLogicalCoreCount();
            }
            if (threads <= 0) {
                threads = densecore::simd::GetNumCores();
            }
        }
        LOG_INFO("Threading configured: {} threads (GGML thread pool + std::thread, no OpenMP)", threads);

        // =========================================================================
        // SIMD DISPATCH TABLE INITIALIZATION (Must happen before any inference!)
        // =========================================================================
        // OpsRegistry selects optimal kernel implementations based on CPU caps.
        // Initializing here ensures the dispatch table is ready before generate()
        // is called, even if the worker thread hasn't started yet.
        // =========================================================================
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }

        // =========================================================================
        // BACKEND REGISTRY INITIALIZATION
        // =========================================================================
        // Register the CPU backend (AVX2/AVX-512 kernels wrapped in ComputeBackend
        // interface). Future ASIC backends can be registered similarly.
        // =========================================================================
        densecore::BackendRegistry::Instance().RegisterCpuBackend();

        // Update DenseCore thread config and CPU backend pools together.
        densecore::UpdateBackendThreads(threads);

        TransformerModel* model = LoadModelWithNuma(canonical_model_path.c_str(), numa_node_id);
        if (!model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "InitEngineEx: failed to load model");
            return nullptr;
        }

        // RAII: Use unique_ptr to guarantee cleanup on exception
        auto state = std::make_unique<EngineState>();
        // Use provided NUMA/pinning configuration
        state->numa_node_id = numa_node_id;
        state->pinning_policy = pinning_policy;
        state->n_threads = threads;  // Store for worker thread

        // Initialize OpRegistry (Dependency Injection)
        state->op_registry = std::make_unique<densecore::OpRegistry>();

        KVCacheConfig kv_config = ComputeKVCacheConfig(model, GGML_TYPE_F16);

        std::cout << "[DenseCore] Auto-configured max_seq_len: " << kv_config.max_seq_len << " (KV cache: ~"
                  << (kv_config.bytes_per_token * kv_config.max_seq_len / 1024 / 1024) << " MB, "
                  << kv_config.bytes_per_token << " bytes/token, max_num_seqs=" << kv_config.max_num_seqs << ")"
                  << std::endl;

        // Log Flash Attention status based on CPU capabilities
        densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
        if (densecore::simd::HasX86Avx512OrBetter(simd_level)) {
            LOG_INFO("Native x86 Flash Attention Enabled ({} detected)", densecore::simd::SimdLevelName(simd_level));
        } else if (densecore::simd::IsArmFamily(simd_level)) {
            LOG_INFO("Native x86 Flash Attention Unavailable on {}; portable CPU flash attention remains eligible",
                     densecore::simd::SimdLevelName(simd_level));
        } else {
            LOG_WARN("Native x86 Flash Attention Disabled (requires AVX-512, detected: {})",
                     densecore::simd::SimdLevelName(simd_level));
        }

        // Log NUMA configuration
        if (state->numa_node_id >= 0) {
            LOG_INFO("NUMA binding: node {}", state->numa_node_id);
        }

        // Initialize Paged KV Cache with NUMA-aware allocation
        PagedKVCache* cache =
            InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len, GGML_TYPE_F16, state->numa_node_id);
        if (!cache) {
            delete model;
            // state is automatically cleaned up by unique_ptr
            SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "InitEngineEx: failed to initialize KV cache");
            return nullptr;
        }

        // Wrap model and cache into ModelEntry and add to pool
        auto entry = std::make_unique<ModelEntry>();
        entry->model_id = "default";
        entry->model_path = canonical_model_path;
        entry->model = std::unique_ptr<TransformerModel>(model);
        entry->kv_cache = std::unique_ptr<PagedKVCache>(cache);
        entry->last_used = std::chrono::steady_clock::now();
        entry->is_loaded = true;

        state->models["default"] = std::move(entry);
        state->default_model_id = "default";
        if (!LoadOptionalDraftModel(state.get(), canonical_model_path, canonical_draft_model_path, GGML_TYPE_F16,
                                    "InitEngineEx")) {
            return nullptr;
        }

#ifdef __APPLE__
        ConfigureHybridScheduler(state.get(), state->models["default"]->model.get());
        // Create platform-specific BackendSelector (uses HybridScheduler on Apple)
        state->backend_selector = densecore::CreateBackendSelector(state->hybrid_scheduler.get());
#else
        // Create default CPU-only BackendSelector on non-Apple platforms
        state->backend_selector = densecore::CreateBackendSelector(nullptr);
#endif

        // Initialize Scheduler with the default model's block manager
        // Note: The scheduler takes a raw pointer to BlockManager, ownership
        // remains with PagedKVCache
        if (state->models["default"]->kv_cache && state->models["default"]->kv_cache->block_manager) {
            state->scheduler =
                std::make_unique<densecore::Scheduler>(state->models["default"]->kv_cache->block_manager);
            LOG_INFO("Scheduler initialized.");

            // Initialize compute buffer (Persistent, Aligned)
            state->InitComputeBuffer();

            // Start background threads
            state->status = EngineStatus::RUNNING;
            state->worker_thread = std::thread(EngineLoop, state.get());
            state->callback_thread = std::thread(CallbackLoop, state.get());

            LOG_INFO("Started worker thread and callback thread");

            ClearError();
            return (DenseCoreHandle)state.release();  // Transfer ownership to caller
        } else {
            LOG_CRITICAL("FATAL: Failed to initialize Scheduler. BlockManager is missing.");
            SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineEx: failed to initialize scheduler");
            return nullptr;
        }
    } catch (const densecore::DenseCoreException& e) {
        LOG_ERROR("Exception in InitEngine: {}", e.what());
        SetError(MapErrorCodeToStatus(e.Code()), e.what());
        return nullptr;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in InitEngine: {}", e.what());
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineEx: exception");
        return nullptr;
    }
}

/**
 * @brief Initialize engine with KV cache type specification
 *
 * Extended initialization with control over KV cache precision.
 * INT8 KV cache reduces memory usage by 50% with minimal quality impact.
 */
DENSECORE_API DenseCoreHandle InitEngineWithKVType(const char* model_path, const char* reserved, int threads,
                                                   int numa_node_id, int pinning_policy,
                                                   DenseCoreKVType kv_cache_type) {
    try {
        ClearError();
        if (!model_path) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "InitEngineWithKVType: model_path is null");
            return nullptr;
        }

        std::string canonical_model_path;
        if (!ValidateModelPathArg(model_path, "InitEngineWithKVType", "model_path", &canonical_model_path)) {
            return nullptr;
        }
        const std::string draft_model_path = ResolveDraftModelPath(reserved);
        std::string canonical_draft_model_path;
        if (!draft_model_path.empty() && !ValidateModelPathArg(draft_model_path, "InitEngineWithKVType",
                                                               "draft_model_path", &canonical_draft_model_path)) {
            return nullptr;
        }
        // Convert enum to ggml_type
        ggml_type cache_type = GGML_TYPE_F16;  // Default
        switch (kv_cache_type) {
        case DENSECORE_KV_FP32: cache_type = GGML_TYPE_F32; break;
        case DENSECORE_KV_INT8:
            cache_type = GGML_TYPE_Q8_0;
            std::cout << "[DenseCore] Using INT8 KV Cache (50% memory reduction)" << std::endl;
            break;
        case DENSECORE_KV_INT4:
            cache_type = GGML_TYPE_Q4_0;
            std::cout << "[DenseCore] Using INT4 KV Cache (75% memory reduction)" << std::endl;
            break;
        case DENSECORE_KV_FP16:
        default: cache_type = GGML_TYPE_F16; break;
        }

        // Reuse InitEngineEx logic but with custom KV cache type
        // Thread configuration
        if (threads <= 0) {
            threads = densecore::simd::GetNumCores();
        }
        std::cout << "[DenseCore] Threading configured: " << threads << " threads" << std::endl;

        // SIMD dispatch initialization
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }

        // Backend registration
        densecore::BackendRegistry::Instance().RegisterCpuBackend();
        densecore::UpdateBackendThreads(threads);

        TransformerModel* model = LoadModelWithNuma(canonical_model_path.c_str(), numa_node_id);
        if (!model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "InitEngineWithKVType: failed to load model");
            return nullptr;
        }

        // RAII: Use unique_ptr to guarantee cleanup on exception
        auto state = std::make_unique<EngineState>();
        state->numa_node_id = numa_node_id;
        state->pinning_policy = pinning_policy;
        state->n_threads = threads;

        // Initialize OpRegistry (Dependency Injection)
        state->op_registry = std::make_unique<densecore::OpRegistry>();

        KVCacheConfig kv_config = ComputeKVCacheConfig(model, cache_type);
        if (kv_config.effective_cache_type != kv_config.requested_cache_type) {
            std::cout << "[DenseCore] Requested " << KVCacheTypeDisplayName(kv_config.requested_cache_type)
                      << " KV cache, but head_dim=" << ResolveKVHeadDim(model) << " is incompatible; using "
                      << KVCacheTypeDisplayName(kv_config.effective_cache_type) << " instead" << std::endl;
        }

        std::cout << "[DenseCore] Auto-configured max_seq_len: " << kv_config.max_seq_len << " (KV cache: ~"
                  << (kv_config.bytes_per_token * kv_config.max_seq_len / 1024 / 1024)
                  << " MB, max_num_seqs=" << kv_config.max_num_seqs << ")" << std::endl;

        // Initialize PagedKVCache with specified type
        PagedKVCache* cache = InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len,
                                               kv_config.effective_cache_type, state->numa_node_id);
        if (!cache) {
            delete model;
            // state is automatically cleaned up by unique_ptr
            SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "InitEngineWithKVType: failed to initialize KV cache");
            return nullptr;
        }

        // Log cache type
        const char* type_name = KVCacheTypeDisplayName(kv_config.effective_cache_type);
        std::cout << "[DenseCore] KV Cache initialized with type: " << type_name << std::endl;

        // Create model entry
        auto entry = std::make_unique<ModelEntry>();
        entry->model_id = "default";
        entry->model_path = canonical_model_path;
        entry->model = std::unique_ptr<TransformerModel>(model);
        entry->kv_cache = std::unique_ptr<PagedKVCache>(cache);
        entry->last_used = std::chrono::steady_clock::now();
        entry->is_loaded = true;

        state->models["default"] = std::move(entry);
        state->default_model_id = "default";
        if (!LoadOptionalDraftModel(state.get(), canonical_model_path, canonical_draft_model_path, cache_type,
                                    "InitEngineWithKVType")) {
            return nullptr;
        }

#ifdef __APPLE__
        ConfigureHybridScheduler(state.get(), state->models["default"]->model.get());
        state->backend_selector = densecore::CreateBackendSelector(state->hybrid_scheduler.get());
#else
        state->backend_selector = densecore::CreateBackendSelector(nullptr);
#endif

        // Initialize Scheduler
        if (state->models["default"]->kv_cache && state->models["default"]->kv_cache->block_manager) {
            state->scheduler =
                std::make_unique<densecore::Scheduler>(state->models["default"]->kv_cache->block_manager);
            std::cout << "[DenseCore] Scheduler initialized." << std::endl;
        } else {
            std::cerr << "[DenseCore] FATAL: Failed to initialize Scheduler." << std::endl;
            SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineWithKVType: failed to initialize scheduler");
            // state is automatically cleaned up by unique_ptr
            return nullptr;
        }

        // Initialize compute buffer
        state->InitComputeBuffer();

        // Start background threads
        state->status = EngineStatus::RUNNING;
        state->worker_thread = std::thread(EngineLoop, state.get());
        state->callback_thread = std::thread(CallbackLoop, state.get());

        std::cout << "[DenseCore] Started worker thread and callback thread" << std::endl;
        ClearError();
        return (DenseCoreHandle)state.release();  // Transfer ownership to caller

    } catch (const densecore::DenseCoreException& e) {
        std::cerr << "[DenseCore] Exception in InitEngineWithKVType: " << e.what() << std::endl;
        SetError(MapErrorCodeToStatus(e.Code()), e.what());
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] Exception in InitEngineWithKVType: " << e.what() << std::endl;
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, "InitEngineWithKVType: exception");
        return nullptr;
    }
}

// =============================================================================
// HAL & Plugin API Implementation
// =============================================================================

DENSECORE_API int DenseCoreLoadPlugin(const char* path) {
    if (!path) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreLoadPlugin: path is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    try {
        densecore::BackendRegistry::Instance().LoadPlugin(path);
        ClearError();
        return DENSECORE_STATUS_OK;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("DenseCoreLoadPlugin: ") + e.what());
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }
}

DENSECORE_API DenseCoreDeviceHandle DenseCoreCreateDevice(const char* backend_name, int device_id) {
    (void)device_id;
    if (!backend_name) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreCreateDevice: backend_name is null");
        return nullptr;
    }
    try {
        std::string name = backend_name;
        densecore::DeviceType type = densecore::DeviceType::CPU;

        // Simple string parsing (case-insensitive preferred but strict here for simplicity)
        if (name == "CPU")
            type = densecore::DeviceType::CPU;
        else if (name == "METAL")
            type = densecore::DeviceType::METAL;
        else if (name == "NPU")
            type = densecore::DeviceType::NPU;
        else if (name == "ASIC")
            type = densecore::DeviceType::ASIC;
        else {
            // Try to find custom device by ID if needed, or fail
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreCreateDevice: unknown backend name");
            return nullptr;
        }

        densecore::ComputeBackend* backend = densecore::BackendRegistry::Instance().Get(type);
        if (!backend) {
            SetError(DENSECORE_STATUS_INTERNAL_ERROR, "DenseCoreCreateDevice: backend not registered");
            return nullptr;
        }

        // Return the backend pointer as the handle (it effectively acts as the device/context provider)
        // In a real multi-device scenario, we might return a specific Device instance.
        // For this API level, ComputeBackend suffices as the entry point.
        ClearError();
        return (DenseCoreDeviceHandle)backend;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("DenseCoreCreateDevice: ") + e.what());
        return nullptr;
    }
}

namespace {
// Helper to parse "layers.N.name"
struct TensorPath {
    int layer_idx = -1;
    std::string subname;
    bool is_layer = false;
};

TensorPath ParseTensorName(const std::string& name) {
    TensorPath res;
    if (name.rfind("layers.", 0) == 0) {
        // Starts with "layers."
        size_t first_dot = 6;  // len("layers.")
        size_t second_dot = name.find('.', first_dot);
        if (second_dot != std::string::npos) {
            std::string num_str = name.substr(first_dot, second_dot - first_dot);
            try {
                res.layer_idx = std::stoi(num_str);
                res.subname = name.substr(second_dot + 1);
                res.is_layer = true;
            } catch (...) {
                // Parse error
            }
        }
    } else {
        res.subname = name;  // Top-level
    }
    return res;
}

ggml_tensor** FindTensorPointer(TransformerModel* model, const std::string& name) {
    // 1. Check top-level mappings
    if (name == "model.embed_tokens.weight") return &model->tok_embeddings;
    if (name == "model.norm.weight") return &model->output_norm;
    if (name == "lm_head.weight") return &model->output;

    // 2. Check layers
    TensorPath path = ParseTensorName(name);
    if (!path.is_layer || path.layer_idx < 0 || path.layer_idx >= (int)model->layers.size()) {
        return nullptr;
    }

    TransformerLayer& layer = model->layers[path.layer_idx];
    const std::string& sub = path.subname;

    static const std::unordered_map<std::string, const char*> layer_map = {
        {"self_attn.q_proj.weight", model_keys::kAttnQWeight},
        {"self_attn.k_proj.weight", model_keys::kAttnKWeight},
        {"self_attn.v_proj.weight", model_keys::kAttnVWeight},
        {"self_attn.o_proj.weight", model_keys::kAttnOWeight},
        {"mlp.gate_proj.weight", model_keys::kFfnGate},
        {"mlp.up_proj.weight", model_keys::kFfnUp},
        {"mlp.down_proj.weight", model_keys::kFfnDown},
        {"input_layernorm.weight", model_keys::kAttnNorm},
        {"post_attention_layernorm.weight", model_keys::kFfnNorm},

        // RoPE Norms (Qwen)
        {"self_attn.q_norm.weight", model_keys::kAttnQNorm},
        {"self_attn.k_norm.weight", model_keys::kAttnKNorm},

        // Biases
        {"self_attn.q_proj.bias", model_keys::kAttnQBias},
        {"self_attn.k_proj.bias", model_keys::kAttnKBias},
        {"self_attn.v_proj.bias", model_keys::kAttnVBias},
        {"self_attn.o_proj.bias", model_keys::kAttnOBias},
    };

    auto it = layer_map.find(sub);
    if (it != layer_map.end()) {
        return layer.GetMutable(it->second);
    }

    return nullptr;
}
}  // namespace

DENSECORE_API int DenseCoreSetWeight(DenseCoreHandle engine, const char* name, const DenseCoreTensor* tensor) {
    if (!engine || !name || !tensor) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreSetWeight: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)engine;
    ModelEntry* entry = state->GetDefaultModel();
    if (!entry || !entry->model) {
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCoreSetWeight: no model loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    TransformerModel* model = entry->model.get();
    ggml_tensor** target_ptr = FindTensorPointer(model, name);

    if (!target_ptr) {
        // Tensor not found in model structure
        // This is expected for some keys that might be in the safetensors file but unused by our runtime
        return DENSECORE_STATUS_OK;
    }

    ggml_tensor* target = *target_ptr;
    if (!target) {
        // Start initialization if null? Usually model structure is pre-allocated.
        // Assuming target IS instantiated but data might be empty.
        // If pointers are null, we might need to verify if we should allocate them.
        SetError(DENSECORE_STATUS_INTERNAL_ERROR,
                 std::string("DenseCoreSetWeight: target tensor structure is null for ") + name);
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }

    // Basic shape validation
    // GGML is column-major in dimension definitions? No, ne[0] is fastest changing.
    // safetensors/numpy is row-major.
    // DenseCoreTensor shape is [d0, d1, d2, d3] row-major.
    // We should copy data bytes directly.

    size_t size_bytes = tensor->shape[0] * std::max((int64_t)1, tensor->shape[1]) *
                        std::max((int64_t)1, tensor->shape[2]) * std::max((int64_t)1, tensor->shape[3]);

    // Calculate bytes based on dtype
    switch (tensor->dtype) {
    case 0: size_bytes *= 4; break;  // F32
    case 1: size_bytes *= 2; break;  // F16
    case 2: size_bytes *= 2; break;  // BF16
    case 3: size_bytes *= 1; break;  // INT8
    default: break;
    }

    if (ggml_nbytes(target) != size_bytes) {
        std::cerr << "[DenseCore] Warning: DenseCoreSetWeight size mismatch for tensor '" << name
                  << "'. Target bytes: " << ggml_nbytes(target) << ", Source bytes: " << size_bytes << std::endl;
        // Mismatch size - logging warning or error?
        // This might happen due to transposition.
        // Let's assume the user knows what they are doing and simply copy for now,
        // but strictly checking size is safer.
    }

    // Perform copy
    // If target->data is null, we need to allocate it?
    // GGUF loader usually mmap, but here we are manual loading.
    // We assume the model structure was created with "empty" tensors or we need to allocate them.
    if (!target->data) {
        // Allocate host memory (or backend memory)
        // Here we just use malloc/posix_memalign or backend->AllocateUnified
        // Simplified: use malloc
        target->data = malloc(size_bytes);
    }

    std::memcpy(target->data, tensor->data, size_bytes);

    ClearError();
    return DENSECORE_STATUS_OK;
}

// =============================================================================
// Internal Workers
// =============================================================================
/**
 * CallbackLoop - Dedicated thread for callback execution.
 *
 * This function runs on a dedicated thread to execute callbacks without
 * blocking the worker thread. It waits on result_cv for new ResultEvents,
 * pops events from the result_queue, and executes the callbacks.
 *
 * This decoupling prevents the Python GIL from blocking the inference
 * hot path, resolving the streaming deadlock.
 */
void CallbackLoop(EngineState* state) {
    try {
        while (true) {
            ResultEvent event;
            {
                std::unique_lock<std::mutex> lock(state->result_mu);
                state->result_cv.wait(
                    lock, [state]() { return !state->result_queue.empty() || state->status == EngineStatus::STOPPED; });

                // Check exit condition: STOPPED and queue empty
                if (state->result_queue.empty() && state->status == EngineStatus::STOPPED) {
                    break;  // Exit: shutdown complete
                }

                if (state->result_queue.empty()) continue;

                event = std::move(state->result_queue.front());
                state->result_queue.pop_front();

                // ===========================================================
                // BACKPRESSURE RELEASE: Notify producers if queue drains
                // ===========================================================
                if (state->result_queue.size() <= EngineState::RESULT_QUEUE_LOW_WATERMARK) {
                    state->result_drain_cv.notify_all();
                }
            }

            if (IsVerboseTokenTraceEnabled()) {
                std::cerr << "[TRACE] CallbackLoop popped event for request " << event.request_id
                          << " (finished=" << event.finished << ")" << std::endl;
            }

            // Execute callback OUTSIDE the lock (GIL acquisition happens here)
            // This is the ONLY place callbacks should be invoked!
            try {
                if (event.callback) {
                    event.callback(event.token_str.c_str(), event.finished ? 1 : (event.error ? 1 : 0),
                                   event.user_data);
                } else if (event.token_result_callback) {
                    TokenResult result;
                    result.token_id = event.token_id;
                    result.text = event.token_str.c_str();
                    result.is_finished = event.finished ? 1 : (event.error ? 1 : 0);
                    event.token_result_callback(&result, event.user_data);
                } else if (event.emb_callback && !event.embedding_data.empty()) {
                    event.emb_callback(event.embedding_data.data(), static_cast<int>(event.embedding_data.size()),
                                       event.user_data);
                    // Note: embedding_data is RAII-managed by std::vector, no manual free
                }
            } catch (const std::exception& e) {
                std::cerr << "[DenseCore] Exception in callback (request " << event.request_id << "): " << e.what()
                          << std::endl;
            } catch (...) {
                std::cerr << "[DenseCore] Unknown exception in callback (request " << event.request_id << ")"
                          << std::endl;
            }
        }

        std::cerr << "[DenseCore] CallbackLoop exiting" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] CallbackLoop thread exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[DenseCore] CallbackLoop thread unknown exception" << std::endl;
    }
}

int SubmitRequest(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                  TokenCallback callback, void* user_data) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequest: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    if (!prompt) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequest: prompt is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    // Get model for tokenization
    ModelEntry* model_entry = state->GetDefaultModel();
    if (!model_entry || !model_entry->model) {
        std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequest: no model loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    Request* req = AcquireAndInitRequest(state);

    req->prompt = prompt;
    req->lora_name = lora_name ? lora_name : "";
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->user_data = user_data;

    // Tokenize immediately (outside hot path)
    req->prompt = MaybeApplyAutoChatTemplate(model_entry->model.get(), prompt);
    req->tokens = Tokenizer::Tokenize(model_entry->model.get(), req->prompt, true);
    req->token_history = req->tokens;

    ApplyDefaultLora(state, req);
    AssignGenerationTier(req);
    EnqueueRequest(state, req);

    ClearError();
    return req->id;
}

int SubmitRequestIds(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, TokenCallback callback,
                     void* user_data) {
    if (!handle || !tokens || n_tokens <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestIds: invalid handle or tokens");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    Request* req = AcquireAndInitRequest(state);

    // Copy tokens
    req->tokens.assign(tokens, tokens + n_tokens);
    req->token_history = req->tokens;
    req->prompt = "";  // No text prompt
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->user_data = user_data;

    ApplyDefaultLora(state, req);
    AssignGenerationTier(req);
    EnqueueRequest(state, req);

    ClearError();
    return req->id;
}

int SubmitRequestWithFormat(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode,
                            TokenCallback callback, void* user_data) {
    return SubmitRequestWithFormatEx(handle, prompt, max_tokens, nullptr, json_mode, callback, user_data);
}

int SubmitRequestWithFormatEx(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                              int json_mode, TokenCallback callback, void* user_data) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithFormatEx: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    if (!prompt) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithFormatEx: prompt is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    // Get model for tokenization
    ModelEntry* model_entry = state->GetDefaultModel();
    if (!model_entry || !model_entry->model) {
        std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequestWithFormatEx: no model loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    Request* req = AcquireAndInitRequest(state);

    req->prompt = prompt;
    req->lora_name = lora_name ? lora_name : "";
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->user_data = user_data;
    req->json_mode = (json_mode != 0);

    ApplyDefaultLora(state, req);

    // Initialize grammar constraint if JSON mode is enabled
    SetupJsonGrammar(req, state);

    // Tokenize immediately (outside hot path)
    req->prompt = MaybeApplyAutoChatTemplate(model_entry->model.get(), prompt);
    req->tokens = Tokenizer::Tokenize(model_entry->model.get(), req->prompt, true);
    req->token_history = req->tokens;

    AssignGenerationTier(req);
    EnqueueRequest(state, req);

    ClearError();
    return req->id;
}

int SubmitRequestIdsWithFormat(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, int json_mode,
                               TokenCallback callback, void* user_data) {
    return SubmitRequestIdsWithFormatEx(handle, tokens, n_tokens, max_tokens, nullptr, json_mode, callback, user_data);
}

int SubmitRequestIdsWithFormatEx(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                 const char* lora_name, int json_mode, TokenCallback callback, void* user_data) {
    if (!handle || !tokens || n_tokens <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestIdsWithFormatEx: invalid handle or tokens");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    Request* req = AcquireAndInitRequest(state);

    // Copy tokens
    req->tokens.assign(tokens, tokens + n_tokens);
    req->token_history = req->tokens;
    req->prompt = "";  // No text prompt
    req->lora_name = lora_name ? lora_name : "";
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->user_data = user_data;
    req->json_mode = (json_mode != 0);

    ApplyDefaultLora(state, req);

    // Initialize grammar constraint if JSON mode is enabled
    SetupJsonGrammar(req, state);

    AssignGenerationTier(req);
    EnqueueRequest(state, req);

    ClearError();
    return req->id;
}

int CancelRequest(DenseCoreHandle handle, int request_id) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CancelRequest: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;

    // ==========================================================================
    // LOCK-FREE QUEUE LIMITATION: Cannot iterate/remove from pending queue
    // ==========================================================================
    // With lock-free queues, we can only cancel active requests.
    // Pending requests will be checked for cancellation when dequeued.
    // ==========================================================================
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        for (Request* req : state->active_requests) {
            if (req->id == request_id) {
                req->cancelled.store(true, std::memory_order_relaxed);
                LOG_INFO("Marked active request ", request_id, " for cancellation.");
                ClearError();
                return 0;  // Success
            }
        }
    }

    // Request not found in active list - it might be pending
    // Mark cancellation will happen lazily when request is dequeued
    if (state->IsPendingRequest(request_id)) {
        state->RecordPendingCancellation(request_id);
        LOG_INFO("Request ", request_id, " not active - cancellation queued.");
        ClearError();
        return 0;
    }
    LOG_INFO("Request ", request_id, " not active.");
    SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CancelRequest: request not active");
    return DENSECORE_STATUS_INVALID_ARGUMENT;
}

DenseCoreMetrics GetMetrics(DenseCoreHandle handle) {
    DenseCoreMetrics m = {};
    if (handle) {
        EngineState* state = (EngineState*)handle;
        m.active_requests = state->metrics.active_requests;
        m.total_tokens_generated = state->metrics.total_tokens_generated;
        auto now = std::chrono::steady_clock::now();
        double elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - state->metrics.start_time).count();
        if (elapsed_s > 0.0) {
            m.tokens_per_second = static_cast<float>(state->metrics.total_tokens_generated.load() / elapsed_s);
            m.requests_per_second = static_cast<float>(state->metrics.total_requests.load() / elapsed_s);
        }
    }
    return m;
}

DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle) {
    DetailedMetrics m = {};
    if (!handle) return m;

    EngineState* state = (EngineState*)handle;

    // Request metrics
    m.active_requests = state->metrics.active_requests;
    m.total_requests = state->metrics.total_requests;
    m.completed_requests = state->metrics.completed_requests;
    m.failed_requests = state->metrics.failed_requests;

    // Count pending requests (lock-free approximate count)
    m.pending_requests = state->pending_requests.Size();
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        m.current_batch_size = state->active_requests.size();
    }

    // Token metrics
    m.total_tokens_generated = state->metrics.total_tokens_generated;
    m.total_prompt_tokens = state->metrics.total_prompt_tokens;

    // Calculate TPS (tokens per second) over engine lifetime
    {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - state->metrics.start_time).count();
        if (elapsed_s > 0.0) {
            m.tokens_per_second = static_cast<float>(m.total_tokens_generated / elapsed_s);
        }
    }

    // Latency metrics from samples
    {
        std::lock_guard<std::mutex> lock(state->metrics.metrics_mu);

        // TTFT metrics
        m.avg_time_to_first_token = state->metrics.CalculateAverage(state->metrics.ttft_samples);
        m.p50_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.5f);
        m.p90_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.9f);
        m.p99_time_to_first_token = state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.99f);

        // ITL metrics
        m.avg_inter_token_latency = state->metrics.CalculateAverage(state->metrics.itl_samples);
        m.p50_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.5f);
        m.p90_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.9f);
        m.p99_inter_token_latency = state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.99f);

        // Queue wait time
        m.avg_queue_wait_time = state->metrics.CalculateAverage(state->metrics.queue_wait_samples);
        m.p99_queue_wait_time = state->metrics.CalculatePercentile(state->metrics.queue_wait_samples, 0.99f);
    }

    // KV Cache metrics
    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->kv_cache && entry->kv_cache->block_manager) {
        m.kv_cache_total_blocks = entry->kv_cache->block_manager->num_blocks;
        int free_blocks = entry->kv_cache->block_manager->GetFreeBlockCount();
        m.kv_cache_usage_blocks = m.kv_cache_total_blocks - free_blocks;
        m.kv_cache_usage_percent = (float)m.kv_cache_usage_blocks / m.kv_cache_total_blocks * 100.0f;
    }

    // Batch metrics
    if (m.completed_requests > 0) {
        m.avg_batch_size = (float)m.total_requests / m.completed_requests;
    }

    // Error metrics
    m.oom_errors = state->metrics.oom_errors;
    m.timeout_errors = state->metrics.timeout_errors;

    return m;
}

void FreeEngine(DenseCoreHandle handle) {
    if (handle) {
        EngineState* state = (EngineState*)handle;
        state->Shutdown();  // Graceful shutdown with draining
        delete state;
    }
}

// Multi-Model API implementations

int LoadModel(DenseCoreHandle handle, const char* model_id, const char* model_path, int /*threads*/) {
    if (!handle || !model_id || !model_path) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "LoadModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::string canonical_model_path;
    if (!ValidateModelPathArg(model_path, "LoadModel", "model_path", &canonical_model_path)) {
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    // Check if model already exists
    {
        std::lock_guard<std::mutex> lock(state->models_mu);
        if (state->models.find(model_id) != state->models.end()) {
            std::cerr << "[DenseCore] Model " << model_id << " already loaded" << std::endl;
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "LoadModel: model already loaded");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
    }

// Load main model
// Note: The user's provided snippet had a syntax error and type mismatch for
// 'main_model' and 'return nullptr'. I've corrected it to use 'model' as in the
// original code and return -3. Assuming LOG_INFO and LOG_ERROR are defined
// elsewhere. If not, std::cerr will be used as a fallback for the error
// message.
#ifdef LOG_INFO
    LOG_INFO("Loading model from: ", model_path);
#endif
    TransformerModel* model = LoadModelWithNuma(canonical_model_path.c_str(), state->numa_node_id);
    if (!model) {
#ifdef LOG_ERROR
        LOG_ERROR("Failed to load main model");
#else
        std::cerr << "[DenseCore] Failed to load model from " << model_path << std::endl;
#endif
        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "LoadModel: failed to load model");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    }

    // Initialize KV cache for this model
    KVCacheConfig kv_config = ComputeKVCacheConfig(model, GGML_TYPE_F16);
    PagedKVCache* cache =
        InitPagedKVCache(model, kv_config.max_num_seqs, kv_config.max_seq_len, GGML_TYPE_F16, state->numa_node_id);
    if (!cache) {
        delete model;
        std::cerr << "[DenseCore] Failed to initialize KV cache for " << model_id << std::endl;
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "LoadModel: failed to initialize KV cache");
        return DENSECORE_STATUS_OUT_OF_MEMORY;
    }

    // Create model entry with unique_ptr ownership
    auto entry = std::make_unique<ModelEntry>();
    entry->model_id = model_id;
    entry->model_path = canonical_model_path;
    entry->model = std::unique_ptr<TransformerModel>(model);
    entry->kv_cache = std::unique_ptr<PagedKVCache>(cache);
    entry->last_used = std::chrono::steady_clock::now();
    entry->is_loaded = true;

    // Add to pool
    {
        std::lock_guard<std::mutex> lock(state->models_mu);
        state->models[model_id] = std::move(entry);

        // Set as default if no default exists
        if (state->default_model_id.empty()) {
            state->default_model_id = model_id;
        }
    }

    ClearError();
    return 0;
}

int UnloadModel(DenseCoreHandle handle, const char* model_id) {
    if (!handle || !model_id) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;

    std::lock_guard<std::mutex> lock(state->models_mu);
    auto it = state->models.find(model_id);
    if (it == state->models.end()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: model not found");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Not found
    }

    // Cannot unload default model if it's the only one
    if (state->default_model_id == model_id && state->models.size() == 1) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "UnloadModel: cannot unload last model");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Cannot unload last model
    }

    // Smart pointers automatically cleanup when erased
    state->models.erase(it);

    // Update default if needed
    if (state->default_model_id == model_id) {
        if (!state->models.empty()) {
            state->default_model_id = state->models.begin()->first;
        } else {
            state->default_model_id = "";
        }
    }

    ClearError();
    return 0;
}

int ListModels(DenseCoreHandle handle, char* out_models, int buffer_size) {
    if (!handle || !out_models || buffer_size <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ListModels: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::lock_guard<std::mutex> lock(state->models_mu);

    std::stringstream ss;
    for (auto it = state->models.begin(); it != state->models.end(); ++it) {
        if (it != state->models.begin()) ss << ",";
        ss << it->first;
    }

    std::string s = ss.str();
    if (s.length() >= (size_t)buffer_size) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ListModels: output buffer too small");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Buffer too small
    }

    strcpy(out_models, s.c_str());
    ClearError();
    return state->models.size();
}

int SetDefaultModel(DenseCoreHandle handle, const char* model_id) {
    if (!handle || !model_id) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SetDefaultModel: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    std::lock_guard<std::mutex> lock(state->models_mu);

    if (state->models.find(model_id) == state->models.end()) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SetDefaultModel: model not found");
        return DENSECORE_STATUS_INVALID_ARGUMENT;  // Not found
    }

    state->default_model_id = model_id;
    ClearError();
    return 0;
}

// =============================================================================
// Universal Graph Execution API
// =============================================================================

/**
 * @brief Submit a graph execution request with tensor inputs (Non-blocking)
 */
DENSECORE_API int SubmitGraphRequest(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                     const char* graph_name, GraphResultCallback callback, void* user_data) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    if (!inputs || num_inputs <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: invalid inputs");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    if (!graph_name) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitGraphRequest: graph_name is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    EngineState* state = (EngineState*)handle;
    Request* req = AcquireAndInitRequest(state);

    req->is_graph_execution = true;
    req->graph_name = graph_name;
    req->graph_callback = callback;
    req->user_data = user_data;

    // Copy input descriptors (ownership of data pointer remains with caller)
    req->graph_inputs.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        req->graph_inputs.push_back(inputs[i]);
    }

    // Graph requests get high priority by default
    req->tier = "premium";
    req->priority = 50;

    EnqueueRequest(state, req);
    ClearError();
    return req->id;
}

namespace {
// Context for synchronous execution
struct GraphSyncContext {
    DenseCoreTensorOutput* user_outputs;
    int num_user_outputs;
    std::promise<void> promise;
};

// Callback for synchronous execution
void GraphSyncCallback(const DenseCoreTensorOutput* outputs, int num_outputs, void* user_data) {
    GraphSyncContext* ctx = (GraphSyncContext*)user_data;
    if (!ctx) return;

    try {
        int count = std::min(ctx->num_user_outputs, num_outputs);
        for (int i = 0; i < count; ++i) {
            const auto& src = outputs[i];
            auto& dst = ctx->user_outputs[i];

            // Copy metadata
            dst.ndim = src.ndim;
            std::memcpy(dst.shape, src.shape, sizeof(dst.shape));
            dst.dtype = src.dtype;

            // Copy data if destination buffer is provided
            if (dst.data && src.data) {
                // Calculate size
                size_t size = 1;
                for (int d = 0; d < src.ndim; ++d) size *= src.shape[d];

                size_t bytes_per_element = 4;  // Default F32
                switch (src.dtype) {
                case DENSECORE_DTYPE_F16:
                case DENSECORE_DTYPE_BF16: bytes_per_element = 2; break;
                case DENSECORE_DTYPE_INT8: bytes_per_element = 1; break;
                case DENSECORE_DTYPE_INT32: bytes_per_element = 4; break;
                case DENSECORE_DTYPE_INT4:
                    bytes_per_element = 0;
                    break;  // Packed, handled differently? Assuming byte-aligned for now
                default: break;
                }
                // INT4 packed: size / 2? Let's assume standard types for now.
                size_t total_bytes = size * bytes_per_element;
                if (src.dtype == DENSECORE_DTYPE_INT4) total_bytes = (size + 1) / 2;

                std::memcpy(dst.data, src.data, total_bytes);
            }
        }
        ctx->promise.set_value();
    } catch (...) {
        ctx->promise.set_exception(std::current_exception());
    }
}
}  // namespace

/**
 * @brief Execute graph synchronously and get output tensors directly
 */
DENSECORE_API int ExecuteGraphSync(DenseCoreHandle handle, const DenseCoreTensorInput* inputs, int num_inputs,
                                   const char* graph_name, DenseCoreTensorOutput* outputs, int num_outputs) {
    if (!handle || !inputs || num_inputs <= 0 || !graph_name || !outputs || num_outputs <= 0) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "ExecuteGraphSync: invalid arguments");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }

    GraphSyncContext ctx;
    ctx.user_outputs = outputs;
    ctx.num_user_outputs = num_outputs;
    auto future = ctx.promise.get_future();

    int req_id = SubmitGraphRequest(handle, inputs, num_inputs, graph_name, GraphSyncCallback, &ctx);
    if (req_id < 0) {
        return req_id;  // Error from SubmitGraphRequest
    }

    // Wait for completion
    try {
        future.wait();
        ClearError();
        return DENSECORE_STATUS_OK;
    } catch (const std::exception& e) {
        SetError(DENSECORE_STATUS_INTERNAL_ERROR, std::string("ExecuteGraphSync error: ") + e.what());
        return DENSECORE_STATUS_INTERNAL_ERROR;
    }
}
