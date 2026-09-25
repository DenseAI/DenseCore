// C API request preparation and submission. Engine lifecycle lives in engine.cpp.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include "densecore/utils/logging.h"

#include "densecore/hal/backend_registry.h"  // Backend abstraction layer

#include "densecore.h"
#include "densecore/backend/apple/apple_silicon.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/hal/device_interface.h"  // For Device/Allocator
#include "densecore/hal/tensor.h"            // For Tensor struct
#include "densecore/memory/kv_cache.h"
#include "densecore/models/embedding.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/model_loader.h"
#include "densecore/models/model_types.h"
#include "densecore/models/tokenizer.h"
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/optimization_bridge.h"  // Runtime SIMD dispatch
#include "densecore/simd/simd_ops.h"
#include "llm/config/runtime_config.h"
#include "models/model_prompt_templates.h"
#include "runtime/engine_internal.h"
#include "runtime/request_builder.h"
#include "runtime/runtime_env.h"
#include "runtime/worker_internal.h"

// Exceptions from preparation must not cross the C ABI. Pool and snapshot
// guards unwind before this boundary translates the failure to a status.
int RequestPreparationFailure(DenseCoreStatus status, const char* api, const char* detail) noexcept {
    try {
        SetLastError(status, std::string(api) + ": " + detail);
    } catch (...) {
        // Reporting an allocation failure must not require another allocation.
        ClearLastError();
    }
    return status;
}


namespace {
void SetError(DenseCoreStatus status, const std::string& message) {
    SetLastError(status, message);
}
void ClearError() {
    ClearLastError();
}
bool ParseBoolEnv(const char* name, bool default_value) {
    return densecore::llm::config::ReadBoolEnv(name, default_value);
}
const densecore::llm::config::FastPathRuntimeConfig& ResolveFastPathRuntimeConfig(const EngineState* state) {
    if (state) {
        return state->fast_path_config;
    }
    static const densecore::llm::config::FastPathRuntimeConfig config =
        densecore::llm::config::LoadFastPathRuntimeConfig();
    return config;
}

const densecore::llm::config::EngineRuntimeDebugConfig& ResolveEngineRuntimeDebugConfig(const EngineState* state) {
    return ResolveFastPathRuntimeConfig(state).engine_debug;
}

bool IsVerboseTokenTraceEnabled(const EngineState* state) {
    return ResolveEngineRuntimeDebugConfig(state).verbose_token_trace;
}

bool IsRuntimePathLoggingEnabled(const EngineState* state) {
    return ResolveEngineRuntimeDebugConfig(state).runtime_path_logging;
}

bool IsRuntimePathTokenLoggingEnabled(const EngineState* state) {
    return ResolveEngineRuntimeDebugConfig(state).runtime_path_token_logging;
}

bool IsParityDebugEnabled(const EngineState* state) {
    return ResolveEngineRuntimeDebugConfig(state).parity_debug;
}

void LogRequestRuntimePath(const EngineState* state, const TransformerModel* model, const std::string& original_prompt,
                           const Request* req) {
    if ((!IsRuntimePathLoggingEnabled(state) && !IsParityDebugEnabled(state)) || !model || !req) {
        return;
    }

    const auto descriptor = densecore::models::DescribeModel(model);
    const auto tokenizer_family = densecore::models::ResolveTokenizerFamily(model);
    const auto prompt_family = densecore::models::ResolvePromptTemplateFamily(model);
    const auto graph_resolution = densecore::models::ResolveGraphFamily(model);
    std::cerr << "[RuntimePath] variant=" << densecore::models::ModelVariantName(descriptor.variant)
              << " prompt_family=" << densecore::models::PromptTemplateFamilyName(prompt_family)
              << " tokenizer_family=" << densecore::models::TokenizerFamilyName(tokenizer_family)
              << " tokenizer_type=" << (model->tokenizer_type.empty() ? "<unset>" : model->tokenizer_type.c_str())
              << " graph_family=" << densecore::models::GraphFamilyName(graph_resolution.preferred_family)
              << " template_applied=" << (req->parity_debug_template_applied ? "1" : "0")
              << " text_primed=" << (req->parity_debug_text_primed ? "1" : "0")
              << " token_primed=" << (req->parity_debug_token_primed ? "1" : "0") << " submit_api="
              << (req->parity_debug_submit_api.empty() ? "<unset>" : req->parity_debug_submit_api.c_str())
              << " temperature=" << req->sampling_params.temperature << " top_p=" << req->sampling_params.top_p
              << " top_k=" << req->sampling_params.top_k
              << " repetition_penalty=" << req->sampling_params.repetition_penalty
              << " json_mode=" << (req->json_mode ? "1" : "0") << " stop_sequences=" << req->stop_sequences.size()
              << " disallowed_tokens=" << req->disallowed_token_ids.size() << " prompt_tokens=" << req->tokens.size()
              << std::endl;
    if (IsParityDebugEnabled(state)) {
        std::cerr << "[ParityRequest] original_prompt=" << original_prompt << std::endl;
        std::cerr << "[ParityRequest] rendered_prompt=" << req->prompt << std::endl;
    }
    if (IsRuntimePathTokenLoggingEnabled(state)) {
        std::cerr << "[RuntimePathTokens] ids=";
        for (size_t i = 0; i < req->tokens.size(); ++i) {
            if (i != 0) {
                std::cerr << ",";
            }
            std::cerr << req->tokens[i];
        }
        std::cerr << std::endl;
    }
}

void ResetTokenIdConstraints(Request* req) {
    if (!req) return;
    req->allowed_token_ids.clear();
    req->sampling_params.allowed_token_ids = nullptr;
    req->sampling_params.disallowed_token_ids =
        req->disallowed_token_ids.empty() ? nullptr : &req->disallowed_token_ids;
}

void ApplyTokenIdConstraints(Request* req, const TransformerModel* model, const int* allowed_token_ids,
                             int num_allowed_token_ids, bool allowed_token_ids_strict, const int* disallowed_token_ids,
                             int num_disallowed_token_ids) {
    if (!req) return;

    req->allowed_token_ids.clear();

    if (allowed_token_ids && num_allowed_token_ids > 0) {
        req->allowed_token_ids.reserve(static_cast<size_t>(num_allowed_token_ids));
        for (int i = 0; i < num_allowed_token_ids; ++i) {
            const int token_id = allowed_token_ids[i];
            if (token_id >= 0) {
                req->allowed_token_ids.push_back(token_id);
            }
        }

        if (model && !allowed_token_ids_strict) {
            for (int stop_id : model->stop_token_ids) {
                if (stop_id >= 0) req->allowed_token_ids.push_back(stop_id);
            }
            if (model->eos_token_id >= 0) {
                req->allowed_token_ids.push_back(model->eos_token_id);
            }
        }

        std::sort(req->allowed_token_ids.begin(), req->allowed_token_ids.end());
        req->allowed_token_ids.erase(std::unique(req->allowed_token_ids.begin(), req->allowed_token_ids.end()),
                                     req->allowed_token_ids.end());
    }

    if (disallowed_token_ids && num_disallowed_token_ids > 0) {
        req->disallowed_token_ids.reserve(req->disallowed_token_ids.size() +
                                          static_cast<size_t>(num_disallowed_token_ids));
        for (int i = 0; i < num_disallowed_token_ids; ++i) {
            const int token_id = disallowed_token_ids[i];
            if (token_id >= 0) {
                req->disallowed_token_ids.push_back(token_id);
            }
        }
    }
    std::sort(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end());
    req->disallowed_token_ids.erase(std::unique(req->disallowed_token_ids.begin(), req->disallowed_token_ids.end()),
                                    req->disallowed_token_ids.end());

    req->sampling_params.allowed_token_ids = req->allowed_token_ids.empty() ? nullptr : &req->allowed_token_ids;
    req->sampling_params.disallowed_token_ids =
        req->disallowed_token_ids.empty() ? nullptr : &req->disallowed_token_ids;
}

bool ShouldPrimeQwenNoThinkingPrompt(const TransformerModel* model) {
    if (model && (model->variant == ModelVariant::QWEN36 || model->variant == ModelVariant::QWEN38)) {
        return false;
    }
    const auto& descriptor = densecore::models::DescribeModel(model);
    if (!descriptor.uses_qwen_thinking_env) {
        return false;
    }
    if (descriptor.variant == ModelVariant::QWEN3) {
        return !ParseBoolEnv("DENSECORE_QWEN3_ENABLE_THINKING", true);
    }
    if (descriptor.variant == ModelVariant::QWEN35) {
        return false;
    }
    return false;
}

bool ShouldPrimeQwenNoThinkingPromptText(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    const auto& descriptor = densecore::models::DescribeModel(model);
    if (!descriptor.uses_qwen_thinking_env) {
        return false;
    }
    if (descriptor.variant == ModelVariant::QWEN36 || descriptor.variant == ModelVariant::QWEN38 ||
        model->variant == ModelVariant::QWEN36 || model->variant == ModelVariant::QWEN38) {
        const char* env = std::getenv("DENSECORE_QWEN36_ENABLE_THINKING");
        bool thinking_disabled = false;
        if (env && env[0] != '\0') {
            thinking_disabled =
                std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "False") == 0;
        } else {
            thinking_disabled = !ParseBoolEnv("DENSECORE_QWEN35_ENABLE_THINKING", true);
        }
        return thinking_disabled && ParseBoolEnv("DENSECORE_QWEN36_PRIME_NO_THINKING", false);
    }
    return ShouldPrimeQwenNoThinkingPrompt(model);
}

std::string PrimeQwenNoThinkingText(std::string prompt_text) {
    if (prompt_text.find("You are a helpful assistant.") != std::string::npos &&
        prompt_text.find("final answer only") == std::string::npos) {
        const std::string replacement =
            "You are a helpful assistant.\n"
            "Provide only the answer. Do not output any thinking process, analysis, reasoning steps, or preamble. "
            "Never start with 'Thinking Process'.";
        prompt_text.replace(prompt_text.find("You are a helpful assistant."),
                            std::strlen("You are a helpful assistant."), replacement);
    }
    if (prompt_text.find("<|im_start|>assistant\n") != std::string::npos &&
        prompt_text.find("Answer:") == std::string::npos) {
        prompt_text.replace(prompt_text.find("<|im_start|>assistant\n"), std::strlen("<|im_start|>assistant\n"),
                            "<|im_start|>assistant\nAnswer: ");
    }
    return prompt_text;
}

void MaybePrimeQwenNoThinkingPromptText(const TransformerModel* model, std::string* prompt_text) {
    if (!prompt_text || !ShouldPrimeQwenNoThinkingPromptText(model)) {
        return;
    }
    *prompt_text = PrimeQwenNoThinkingText(std::move(*prompt_text));
}

void MaybePrimeQwenNoThinking(const TransformerModel* model, std::vector<int>* tokens) {
    if (!model || !tokens || tokens->empty()) {
        return;
    }
    if (!ShouldPrimeQwenNoThinkingPrompt(model)) {
        return;
    }

    const std::string prompt_text = Tokenizer::DetokenizeMultiple(model, *tokens);
    const std::string primed_text = PrimeQwenNoThinkingText(prompt_text);
    if (primed_text == prompt_text) {
        return;
    }

    *tokens = Tokenizer::Tokenize(model, primed_text, /*add_bos=*/false, /*add_eos=*/false);
}

bool ShouldDisableBosForRenderedQwenPrompt(const TransformerModel* model, const std::string& prompt) {
    if (!model) {
        return false;
    }
    const auto variant = densecore::models::DescribeModel(model).variant;
    if (variant != ModelVariant::QWEN35 && variant != ModelVariant::QWEN36 && variant != ModelVariant::QWEN38) {
        return false;
    }
    return prompt.find("<|im_start|>") != std::string::npos || prompt.find("<|im_end|>") != std::string::npos;
}

bool ResolveAddBosForPrompt(const TransformerModel* model, const std::string& prompt) {
    if (!model) {
        return false;
    }
    if (ShouldDisableBosForRenderedQwenPrompt(model, prompt)) {
        return false;
    }
    return model->tokenizer_add_bos;
}

void ConfigureQwenReasoningTokenBlocklist(const TransformerModel* model, Request* req) {
    densecore::models::ConfigureQwenReasoningTokenBlocklistForModel(model, req);
}

void ConfigureGemma4TextTokenBlocklist(const TransformerModel* model, Request* req) {
    densecore::models::ConfigureGemma4TextTokenBlocklistForModel(model, req);
}

void ConfigureLFM2TextTokenBlocklist(const TransformerModel* model, Request* req) {
    densecore::models::ConfigureLFM2TextTokenBlocklistForModel(model, req);
}

bool PromptEndsInsideTaggedBlock(const std::string& prompt, const char* open_tag, const char* close_tag) {
    if (!open_tag || !close_tag) {
        return false;
    }
    const size_t last_open = prompt.rfind(open_tag);
    if (last_open == std::string::npos) {
        return false;
    }
    const size_t last_close = prompt.rfind(close_tag);
    return last_close == std::string::npos || last_close < last_open;
}

void InitializePromptSuppressionState(Request* req) {
    if (!req) {
        return;
    }
    req->think_tag_pending.clear();
    req->tool_call_tag_pending.clear();
    req->tool_response_tag_pending.clear();
    req->in_think_block = PromptEndsInsideTaggedBlock(req->prompt, "<think>", "</think>");
    req->in_tool_call_block = PromptEndsInsideTaggedBlock(req->prompt, "<tool_call>", "</tool_call>");
    req->in_tool_response_block = PromptEndsInsideTaggedBlock(req->prompt, "<tool_response>", "</tool_response>");
    req->suppress_reasoning_tags = !req->in_think_block;
}

void DebugPrintPromptTokens(const TransformerModel* model, const std::vector<int>& tokens, const char* tag) {
    if (densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_PROMPT_TOKENS") == nullptr) {
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
    return densecore::models::ApplyModelAutoChatTemplate(model, prompt);
}


}  // namespace

#ifdef DENSECORE_TEST_BUILD
std::string DenseCoreTestOnlyApplyAutoChatTemplate(const TransformerModel* model, const std::string& prompt) {
    return MaybeApplyAutoChatTemplate(model, prompt);
}

bool DenseCoreTestOnlyPromptStartsInThinkBlock(const std::string& prompt) {
    Request req{};
    req.prompt = prompt;
    InitializePromptSuppressionState(&req);
    return req.in_think_block;
}

bool DenseCoreTestOnlySuppressesReasoningTagsForPrompt(const std::string& prompt) {
    Request req{};
    req.prompt = prompt;
    InitializePromptSuppressionState(&req);
    return req.suppress_reasoning_tags;
}

bool DenseCoreTestOnlySuppressesReasoningTagsForModelPrompt(const TransformerModel* model, const std::string& prompt) {
    (void)model;
    Request req{};
    req.prompt = prompt;
    InitializePromptSuppressionState(&req);
    return req.suppress_reasoning_tags;
}

std::vector<int> DenseCoreTestOnlyQwenReasoningBlocklist(const TransformerModel* model) {
    Request req{};
    ConfigureQwenReasoningTokenBlocklist(model, &req);
    return req.disallowed_token_ids;
}

std::vector<int> DenseCoreTestOnlyQwenReasoningBlocklistForPrompt(const TransformerModel* model,
                                                                  const std::string& prompt) {
    Request req{};
    req.prompt = prompt;
    InitializePromptSuppressionState(&req);
    ConfigureQwenReasoningTokenBlocklist(model, &req);
    return req.disallowed_token_ids;
}

std::vector<int> DenseCoreTestOnlyGemma4TextBlocklist(const TransformerModel* model) {
    Request req{};
    ConfigureGemma4TextTokenBlocklist(model, &req);
    return req.disallowed_token_ids;
}

std::vector<int> DenseCoreTestOnlyLFM2TextBlocklist(const TransformerModel* model) {
    Request req{};
    ConfigureLFM2TextTokenBlocklist(model, &req);
    return req.disallowed_token_ids;
}

bool DenseCoreTestOnlyGemma4TextBlocklistReachesSamplingParams(const TransformerModel* model) {
    Request req{};
    ConfigureGemma4TextTokenBlocklist(model, &req);
    ResetTokenIdConstraints(&req);
    return req.sampling_params.disallowed_token_ids == &req.disallowed_token_ids &&
           std::binary_search(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), 0);
}

bool DenseCoreTestOnlyGemma4CallerDisallowMergesWithModelBlocklist(const TransformerModel* model, int caller_token_id) {
    Request req{};
    ConfigureGemma4TextTokenBlocklist(model, &req);
    ApplyTokenIdConstraints(&req, model, /*allowed_token_ids=*/nullptr, /*num_allowed_token_ids=*/0,
                            /*allowed_token_ids_strict=*/false, &caller_token_id, caller_token_id >= 0 ? 1 : 0);
    return req.sampling_params.disallowed_token_ids == &req.disallowed_token_ids &&
           std::binary_search(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), 0) &&
           std::binary_search(req.disallowed_token_ids.begin(), req.disallowed_token_ids.end(), caller_token_id);
}

std::string DenseCoreTestOnlyPrimeQwenNoThinkingPromptText(const TransformerModel* model, const std::string& prompt) {
    if (!ShouldPrimeQwenNoThinkingPromptText(model)) {
        return prompt;
    }
    return PrimeQwenNoThinkingText(prompt);
}

std::string DenseCoreTestOnlyPrimeQwenNoThinking(const TransformerModel* model, const std::string& prompt) {
    if (!ShouldPrimeQwenNoThinkingPrompt(model)) {
        return prompt;
    }
    return PrimeQwenNoThinkingText(prompt);
}

bool DenseCoreTestOnlyResolveAddBosForPrompt(const TransformerModel* model, const std::string& prompt) {
    return ResolveAddBosForPrompt(model, prompt);
}
#endif

// Global request ID counter (shared across all Submit* functions)
static std::atomic<int> global_req_id{1};

// Queue publication is the sole ownership transfer for prepared C API requests.
// Cache the ID before publication: the worker may complete and recycle req before
// the submitting thread returns. RequestGuard owns every pre-publication failure.
int PublishPreparedRequest(EngineState* state, Request* req, RequestGuard& guard, const std::string& draining_error) {
    const int request_id = req->id;
    try {
        if (!state->TryAdmitPendingRequest(req)) {
            SetError(DENSECORE_STATUS_ENGINE_STOPPED, draining_error);
            return DENSECORE_STATUS_ENGINE_STOPPED;
        }
    } catch (const std::bad_alloc&) {
        SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "Request queue allocation failed");
        return DENSECORE_STATUS_OUT_OF_MEMORY;
    }
    guard.release();
    ClearError();
    return request_id;
}

namespace {

enum class TextPreparationPolicy { Legacy, Sampling, TokenResults };

void PrepareTextRequest(TransformerModel* model, Request* req, const char* prompt, TextPreparationPolicy policy) {
    req->prompt = MaybeApplyAutoChatTemplate(model, prompt);
    const bool trace = policy != TextPreparationPolicy::Legacy;
    if (trace) req->parity_debug_template_applied = (req->prompt != prompt);
    const std::string before_priming = trace ? req->prompt : std::string{};
    MaybePrimeQwenNoThinkingPromptText(model, &req->prompt);
    if (trace) req->parity_debug_text_primed = (req->prompt != before_priming);
    InitializePromptSuppressionState(req);
    req->tokens = Tokenizer::Tokenize(model, req->prompt, ResolveAddBosForPrompt(model, req->prompt));
    if (trace) {
        const std::vector<int> before = req->tokens;
        MaybePrimeQwenNoThinking(model, &req->tokens);
        req->parity_debug_token_primed = (req->tokens != before);
    }
    ConfigureQwenReasoningTokenBlocklist(model, req);
    // The structured-result entry point historically omits this blocklist.
    // Preserve that public behavior rather than normalizing it during refactoring.
    if (policy != TextPreparationPolicy::TokenResults) {
        densecore::models::ConfigureQwen36TextTokenBlocklistForModel(model, req);
    }
    ConfigureGemma4TextTokenBlocklist(model, req);
    ConfigureLFM2TextTokenBlocklist(model, req);
}

}  // namespace

Request* AcquireAndInitRequest(EngineState* state) {
    Request* req = state->request_pool.Acquire();
    req->Reset();
    req->id = global_req_id.fetch_add(1);
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}

namespace {

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
                       void* user_data, TokenCallbackEx callback_ex = nullptr) {
    ResetMoEGraphWiringDebugCounter();
    req->max_tokens = max_tokens;
    req->callback = callback;
    req->callback_ex = callback_ex;
    req->user_data = user_data;
    req->json_mode = (json_mode != 0);

    // Apply default LoRA adapter (backward compatibility)
    ApplyDefaultLora(state, req);

    // Set sampling parameters
    req->sampling_params.temperature = temperature;
    req->sampling_params.top_p = top_p;
    req->sampling_params.top_k = top_k;
    req->sampling_params.repetition_penalty = repetition_penalty;
    req->sampling_params.allowed_token_ids = nullptr;
    req->sampling_params.disallowed_token_ids = nullptr;

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

int DenseCoreSetGenerationCompletionCallback(DenseCoreHandle handle, GenerationCompletionCallback callback) {
    if (!handle) return DENSECORE_STATUS_INVALID_ARGUMENT;
    static_cast<EngineState*>(handle)->generation_completion_callback.store(callback, std::memory_order_release);
    return DENSECORE_STATUS_OK;
}


int SubmitEmbeddingRequest(DenseCoreHandle handle, const char* prompt, EmbeddingCallback callback, void* user_data) {
    // Default: MEAN pooling with normalization
    return SubmitEmbeddingRequestEx(handle, prompt, 0, 1, callback, user_data);
}

int SubmitEmbeddingRequestEx(DenseCoreHandle handle, const char* prompt, int pooling_type, int normalize,
                             EmbeddingCallback callback, void* user_data) {
    return RequestApiBoundary("SubmitEmbeddingRequestEx", [&]() -> int {
        if (!handle) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitEmbeddingRequestEx: handle is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!prompt) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitEmbeddingRequestEx: prompt is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!callback) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitEmbeddingRequestEx: callback is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        // Get model for tokenization
        ModelEntry* model_entry = state->GetDefaultModel();
        if (!model_entry || !model_entry->model) {
            std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitEmbeddingRequestEx: no model loaded for tokenization");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);

        req->prompt = prompt;
        req->max_tokens = 0;  // No generation
        req->is_embedding = true;
        req->embedding_callback = callback;
        req->user_data = user_data;

        // Store pooling config
        req->pooling_type = static_cast<densecore::PoolingStrategy>(pooling_type);
        req->normalize_embedding = (normalize != 0);

        // Tokenize immediately (outside hot path). BERT-family encoders such as
        // bge-m3 are trained with both <s>/</s> sentinel tokens.
        const bool add_eos = model_entry->model->arch == ModelArch::BERT &&
                             (model_entry->model->sep_token_id >= 0 || model_entry->model->eos_token_id >= 0);
        req->tokens =
            Tokenizer::Tokenize(model_entry->model.get(), prompt, model_entry->model->tokenizer_add_bos, add_eos);

        // Embeddings get premium tier explicitly
        req->priority = 50;
        req->tier = "premium";

        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, "SubmitEmbeddingRequestEx: engine is draining");
    });
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
    return SubmitRequestWithSamplingConstraintsEx(handle, prompt, max_tokens, lora_name, temperature, top_p, top_k,
                                                  repetition_penalty, stop_sequences, json_mode,
                                                  /*allowed_token_ids=*/nullptr, /*num_allowed_token_ids=*/0,
                                                  /*allowed_token_ids_strict=*/0,
                                                  /*disallowed_token_ids=*/nullptr,
                                                  /*num_disallowed_token_ids=*/0, callback, user_data);
}

static int SubmitRequestWithSamplingConstraintsImpl(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                                    const char* lora_name, float temperature, float top_p, int top_k,
                                                    float repetition_penalty, const char** stop_sequences,
                                                    int json_mode, const int* allowed_token_ids,
                                                    int num_allowed_token_ids, int allowed_token_ids_strict,
                                                    const int* disallowed_token_ids, int num_disallowed_token_ids,
                                                    TokenCallback callback, TokenCallbackEx callback_ex,
                                                    void* user_data, const char* error_context) {
    return RequestApiBoundary(error_context, [&]() -> int {
        if (!handle || !prompt) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(error_context) + ": invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        ModelEntry* model_entry = state->GetDefaultModel();
        if (!model_entry || !model_entry->is_loaded) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, std::string(error_context) + ": default model not loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);
        req->lora_name = lora_name ? lora_name : "";
        req->parity_debug_submit_api = "SubmitRequestWithSamplingConstraintsEx";

        InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences,
                          json_mode, callback, user_data, callback_ex);

        // Tokenize prompt
        PrepareTextRequest(model_entry->model.get(), req, prompt, TextPreparationPolicy::Sampling);
        if ((allowed_token_ids && num_allowed_token_ids > 0) ||
            (disallowed_token_ids && num_disallowed_token_ids > 0)) {
            ApplyTokenIdConstraints(req, model_entry->model.get(), allowed_token_ids, num_allowed_token_ids,
                                    allowed_token_ids_strict != 0, disallowed_token_ids, num_disallowed_token_ids);
        } else {
            ResetTokenIdConstraints(req);
        }
        DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "sampling");
        req->token_history = req->tokens;
        LogRequestRuntimePath(state, model_entry->model.get(), prompt, req);

        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, std::string(error_context) + ": engine is draining");
    });
}

int SubmitRequestWithSamplingConstraintsEx(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                           const char* lora_name, float temperature, float top_p, int top_k,
                                           float repetition_penalty, const char** stop_sequences, int json_mode,
                                           const int* allowed_token_ids, int num_allowed_token_ids,
                                           int allowed_token_ids_strict, const int* disallowed_token_ids,
                                           int num_disallowed_token_ids, TokenCallback callback, void* user_data) {
    return SubmitRequestWithSamplingConstraintsImpl(
        handle, prompt, max_tokens, lora_name, temperature, top_p, top_k, repetition_penalty, stop_sequences, json_mode,
        allowed_token_ids, num_allowed_token_ids, allowed_token_ids_strict, disallowed_token_ids,
        num_disallowed_token_ids, callback, nullptr, user_data, "SubmitRequestWithSamplingConstraintsEx");
}

int SubmitRequestWithSamplingConstraintsCallbackEx(DenseCoreHandle handle, const char* prompt, int max_tokens,
                                                   const char* lora_name, float temperature, float top_p, int top_k,
                                                   float repetition_penalty, const char** stop_sequences, int json_mode,
                                                   const int* allowed_token_ids, int num_allowed_token_ids,
                                                   int allowed_token_ids_strict, const int* disallowed_token_ids,
                                                   int num_disallowed_token_ids, TokenCallbackEx callback,
                                                   void* user_data) {
    return SubmitRequestWithSamplingConstraintsImpl(
        handle, prompt, max_tokens, lora_name, temperature, top_p, top_k, repetition_penalty, stop_sequences, json_mode,
        allowed_token_ids, num_allowed_token_ids, allowed_token_ids_strict, disallowed_token_ids,
        num_disallowed_token_ids, nullptr, callback, user_data, "SubmitRequestWithSamplingConstraintsCallbackEx");
}

int SubmitRequestWithTokenResults(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature,
                                  float top_p, int top_k, float repetition_penalty, TokenResultCallback callback,
                                  void* user_data) {
    return RequestApiBoundary("SubmitRequestWithTokenResults", [&]() -> int {
        if (!handle || !prompt) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithTokenResults: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        ModelEntry* model_entry = state->GetDefaultModel();
        if (!model_entry || !model_entry->is_loaded) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequestWithTokenResults: default model not loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);
        req->token_result_callback = callback;
        req->parity_debug_submit_api = "SubmitRequestWithTokenResults";

        InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty,
                          /*stop_sequences=*/nullptr, /*json_mode=*/0,
                          /*callback=*/nullptr, user_data);

        PrepareTextRequest(model_entry->model.get(), req, prompt, TextPreparationPolicy::TokenResults);
        ResetTokenIdConstraints(req);
        DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "token_results");
        req->token_history = req->tokens;
        LogRequestRuntimePath(state, model_entry->model.get(), prompt, req);

        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, "SubmitRequestWithTokenResults: engine is draining");
    });
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
    return SubmitRequestIdsWithSamplingConstraintsEx(handle, tokens, n_tokens, max_tokens, lora_name, temperature,
                                                     top_p, top_k, repetition_penalty, stop_sequences, json_mode,
                                                     /*allowed_token_ids=*/nullptr, /*num_allowed_token_ids=*/0,
                                                     /*allowed_token_ids_strict=*/0,
                                                     /*disallowed_token_ids=*/nullptr,
                                                     /*num_disallowed_token_ids=*/0, callback, user_data);
}

static int SubmitRequestIdsWithSamplingConstraintsImpl(
    DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, const char* lora_name, float temperature,
    float top_p, int top_k, float repetition_penalty, const char** stop_sequences, int json_mode,
    const int* allowed_token_ids, int num_allowed_token_ids, int allowed_token_ids_strict,
    const int* disallowed_token_ids, int num_disallowed_token_ids, TokenCallback callback, TokenCallbackEx callback_ex,
    void* user_data, const char* rendered_prompt, bool tokens_already_snapshot_primed, const char* submit_api,
    const char* error_context) {
    return RequestApiBoundary(error_context, [&]() -> int {
        if (!handle || !tokens || n_tokens <= 0) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(error_context) + ": invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);
        req->lora_name = lora_name ? lora_name : "";
        req->parity_debug_submit_api = submit_api ? submit_api : "SubmitRequestIdsWithSamplingConstraintsEx";
        req->token_id_submit_used = true;

        InitCommonRequest(state, req, max_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences,
                          json_mode, callback, user_data, callback_ex);

        // Assign pre-tokenized input
        req->tokens.assign(tokens, tokens + n_tokens);
        ModelEntry* model_entry = state->GetDefaultModel();
        if (model_entry && model_entry->model) {
            if (rendered_prompt && rendered_prompt[0] != '\0') {
                // The Go chat path has already rendered and tokenized this prompt
                // via DenseCoreBuildRenderedRequestSnapshot. Keep that text on the
                // request for prompt-state decisions, but never re-template or
                // re-tokenize it here.
                req->prompt = rendered_prompt;
                InitializePromptSuppressionState(req);
                req->parity_debug_template_applied = false;
                req->parity_debug_text_primed = false;
            }
            if (!tokens_already_snapshot_primed) {
                const std::vector<int> ids_tokens_before_priming = req->tokens;
                MaybePrimeQwenNoThinking(model_entry->model.get(), &req->tokens);
                req->parity_debug_token_primed = (req->tokens != ids_tokens_before_priming);
            }
            ConfigureQwenReasoningTokenBlocklist(model_entry->model.get(), req);
            densecore::models::ConfigureQwen36TextTokenBlocklistForModel(model_entry->model.get(), req);
            ConfigureGemma4TextTokenBlocklist(model_entry->model.get(), req);
            ConfigureLFM2TextTokenBlocklist(model_entry->model.get(), req);
            if ((allowed_token_ids && num_allowed_token_ids > 0) ||
                (disallowed_token_ids && num_disallowed_token_ids > 0)) {
                ApplyTokenIdConstraints(req, model_entry->model.get(), allowed_token_ids, num_allowed_token_ids,
                                        allowed_token_ids_strict != 0, disallowed_token_ids, num_disallowed_token_ids);
            } else {
                ResetTokenIdConstraints(req);
            }
            DebugPrintPromptTokens(model_entry->model.get(), req->tokens, "ids_sampling");
            LogRequestRuntimePath(state, model_entry->model.get(), req->prompt.c_str(), req);
        }
        req->token_history = req->tokens;

        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, std::string(error_context) + ": engine is draining");
    });
}

int SubmitRequestIdsWithSamplingConstraintsEx(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                              const char* lora_name, float temperature, float top_p, int top_k,
                                              float repetition_penalty, const char** stop_sequences, int json_mode,
                                              const int* allowed_token_ids, int num_allowed_token_ids,
                                              int allowed_token_ids_strict, const int* disallowed_token_ids,
                                              int num_disallowed_token_ids, TokenCallback callback, void* user_data) {
    return SubmitRequestIdsWithSamplingConstraintsImpl(
        handle, tokens, n_tokens, max_tokens, lora_name, temperature, top_p, top_k, repetition_penalty, stop_sequences,
        json_mode, allowed_token_ids, num_allowed_token_ids, allowed_token_ids_strict, disallowed_token_ids,
        num_disallowed_token_ids, callback, nullptr, user_data, /*rendered_prompt=*/nullptr,
        /*tokens_already_snapshot_primed=*/false, "SubmitRequestIdsWithSamplingConstraintsEx",
        "SubmitRequestIdsWithSamplingConstraintsEx");
}

int SubmitRequestIdsWithSamplingConstraintsCallbackEx(
    DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, const char* lora_name, float temperature,
    float top_p, int top_k, float repetition_penalty, const char** stop_sequences, int json_mode,
    const int* allowed_token_ids, int num_allowed_token_ids, int allowed_token_ids_strict,
    const int* disallowed_token_ids, int num_disallowed_token_ids, TokenCallbackEx callback, void* user_data) {
    return SubmitRequestIdsWithSamplingConstraintsImpl(
        handle, tokens, n_tokens, max_tokens, lora_name, temperature, top_p, top_k, repetition_penalty, stop_sequences,
        json_mode, allowed_token_ids, num_allowed_token_ids, allowed_token_ids_strict, disallowed_token_ids,
        num_disallowed_token_ids, nullptr, callback, user_data, /*rendered_prompt=*/nullptr,
        /*tokens_already_snapshot_primed=*/false, "SubmitRequestIdsWithSamplingConstraintsCallbackEx",
        "SubmitRequestIdsWithSamplingConstraintsCallbackEx");
}

int SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx(
    DenseCoreHandle handle, const char* rendered_prompt, const int* tokens, int n_tokens, int max_tokens,
    const char* lora_name, float temperature, float top_p, int top_k, float repetition_penalty,
    const char** stop_sequences, int json_mode, const int* allowed_token_ids, int num_allowed_token_ids,
    int allowed_token_ids_strict, const int* disallowed_token_ids, int num_disallowed_token_ids,
    TokenCallbackEx callback, void* user_data) {
    return RequestApiBoundary("SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx", [&]() -> int {
        if (!rendered_prompt || rendered_prompt[0] == '\0') {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT,
                     "SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx: rendered_prompt is required");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        return SubmitRequestIdsWithSamplingConstraintsImpl(
            handle, tokens, n_tokens, max_tokens, lora_name, temperature, top_p, top_k, repetition_penalty,
            stop_sequences, json_mode, allowed_token_ids, num_allowed_token_ids, allowed_token_ids_strict,
            disallowed_token_ids, num_disallowed_token_ids, nullptr, callback, user_data, rendered_prompt,
            /*tokens_already_snapshot_primed=*/true, "SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx",
            "SubmitRenderedRequestIdsWithSamplingConstraintsCallbackEx");
    });
}

int SubmitBatchEmbeddingRequest(DenseCoreHandle handle, const char** prompts, int num_prompts, int pooling_type,
                                int normalize, EmbeddingCallback callback, void* user_data) {
    return RequestApiBoundary("SubmitBatchEmbeddingRequest", [&]() -> int {
        if (!handle || !prompts || num_prompts <= 0 || !callback) {
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
    });
}

int GetEmbeddingDimension(DenseCoreHandle handle) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "GetEmbeddingDimension: handle is null");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    EngineState* state = (EngineState*)handle;
    std::unique_lock<std::mutex> construction_lock(state->construction_mu);

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
    std::unique_lock<std::mutex> construction_lock(state->construction_mu);

    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        ClearError();
        return static_cast<int>(entry->model->hparams.n_ctx);
    }
    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "GetMaxContextTokens: no model loaded");
    return DENSECORE_STATUS_MODEL_LOAD_FAILED;
}

int CountTokens(DenseCoreHandle handle, const char* text, int add_bos, int add_eos) {
    return RequestApiBoundary("CountTokens", [&]() -> int {
        if (!handle || !text) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "CountTokens: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        ModelEntry* entry = state->GetDefaultModel();
        if (entry && entry->model) {
            const std::vector<int> tokens = Tokenizer::Tokenize(entry->model.get(), text, add_bos != 0, add_eos != 0);
            ClearError();
            return static_cast<int>(tokens.size());
        }

        SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "CountTokens: no model loaded");
        return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    });
}

const char* GetTokenizerType(DenseCoreHandle handle) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "GetTokenizerType: handle is null");
        return nullptr;
    }
    EngineState* state = (EngineState*)handle;
    std::unique_lock<std::mutex> construction_lock(state->construction_mu);
    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        ClearError();
        return entry->model->tokenizer_type.empty() ? nullptr : entry->model->tokenizer_type.c_str();
    }
    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "GetTokenizerType: no model loaded");
    return nullptr;
}

const char* GetChatTemplate(DenseCoreHandle handle) {
    if (!handle) {
        SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "GetChatTemplate: handle is null");
        return nullptr;
    }
    EngineState* state = (EngineState*)handle;
    std::unique_lock<std::mutex> construction_lock(state->construction_mu);
    ModelEntry* entry = state->GetDefaultModel();
    if (entry && entry->model) {
        ClearError();
        return entry->model->chat_template.empty() ? nullptr : entry->model->chat_template.c_str();
    }
    SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "GetChatTemplate: no model loaded");
    return nullptr;
}

namespace {
thread_local std::string g_rendered_chat_prompt;
thread_local std::string g_rendered_model_variant;
thread_local std::string g_rendered_prompt_family;
thread_local std::vector<int> g_preview_token_ids;

DenseCoreSubmitPath ResolveSubmitPathForPreview(bool token_ids, bool json_mode, bool sampling) {
    if (token_ids) {
        if (json_mode) {
            return sampling ? DENSECORE_SUBMIT_PATH_IDS_WITH_SAMPLING : DENSECORE_SUBMIT_PATH_IDS_WITH_FORMAT;
        }
        return sampling ? DENSECORE_SUBMIT_PATH_IDS_WITH_SAMPLING : DENSECORE_SUBMIT_PATH_IDS;
    }
    if (json_mode) {
        return sampling ? DENSECORE_SUBMIT_PATH_TEXT_WITH_SAMPLING : DENSECORE_SUBMIT_PATH_TEXT_WITH_FORMAT;
    }
    return sampling ? DENSECORE_SUBMIT_PATH_TEXT_WITH_SAMPLING : DENSECORE_SUBMIT_PATH_TEXT;
}

char* CopyCStringOwned(const std::string& value) {
    if (value.empty()) {
        return nullptr;
    }
    void* raw = std::malloc(value.size() + 1);
    if (!raw) {
        return nullptr;
    }
    char* out = static_cast<char*>(raw);
    std::memcpy(out, value.data(), value.size());
    out[value.size()] = '\0';
    return out;
}

int* CopyTokenIdsOwned(const std::vector<int>& ids) {
    if (ids.empty()) {
        return nullptr;
    }
    void* raw = std::malloc(sizeof(int) * ids.size());
    if (!raw) {
        return nullptr;
    }
    int* out = static_cast<int*>(raw);
    std::memcpy(out, ids.data(), sizeof(int) * ids.size());
    return out;
}

void FillSnapshotMetadata(DenseCoreRequestSnapshot* out, const TransformerModel* model, DenseCoreSubmitPath path,
                          float temperature, float top_p, int top_k, float repetition_penalty, int json_mode,
                          bool template_applied, bool text_primed, bool token_primed) {
    out->submit_path = path;
    out->temperature = temperature;
    out->top_p = top_p;
    out->top_k = top_k;
    out->repetition_penalty = repetition_penalty;
    out->json_mode = json_mode != 0 ? 1 : 0;
    out->template_applied = template_applied ? 1 : 0;
    out->text_primed = text_primed ? 1 : 0;
    out->token_primed = token_primed ? 1 : 0;
    out->tokenizer_type = (model && !model->tokenizer_type.empty()) ? model->tokenizer_type.c_str() : nullptr;
    out->model_variant = g_rendered_model_variant.empty() ? nullptr : g_rendered_model_variant.c_str();
    out->prompt_family = g_rendered_prompt_family.empty() ? nullptr : g_rendered_prompt_family.c_str();
    out->caller_owns_buffers = 0;
}
}  // namespace

int DenseCoreRenderChatPrompt(DenseCoreHandle handle, const DenseCoreChatMessage* messages, int num_messages,
                              const DenseCoreChatTemplateOptions* options, DenseCoreRenderedChatPrompt* out) {
    return RequestApiBoundary("DenseCoreRenderChatPrompt", [&]() -> int {
        if (!handle || !messages || num_messages <= 0 || !out) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreRenderChatPrompt: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }

        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        ModelEntry* entry = state->GetDefaultModel();
        if (!entry || !entry->model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCoreRenderChatPrompt: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        std::vector<densecore::models::CanonicalChatMessage> canonical_messages;
        canonical_messages.reserve(static_cast<size_t>(num_messages));
        for (int i = 0; i < num_messages; ++i) {
            densecore::models::CanonicalChatMessage message;
            message.role = messages[i].role ? messages[i].role : "";
            message.content = messages[i].content ? messages[i].content : "";
            message.reasoning_content = messages[i].reasoning_content ? messages[i].reasoning_content : "";
            message.name = messages[i].name ? messages[i].name : "";
            canonical_messages.push_back(std::move(message));
        }

        densecore::models::CanonicalChatRenderOptions render_options;
        render_options.enable_thinking = options ? options->enable_thinking : -1;
        render_options.preserve_thinking = options ? options->preserve_thinking : -1;
        bool thinking_enabled = false;
        g_rendered_chat_prompt = densecore::models::RenderModelChatMessages(entry->model.get(), canonical_messages,
                                                                            render_options, &thinking_enabled);
        const auto descriptor = densecore::models::DescribeModel(entry->model.get());
        g_rendered_model_variant = densecore::models::ModelVariantName(descriptor.variant);
        g_rendered_prompt_family = densecore::models::PromptTemplateFamilyName(
            densecore::models::ResolvePromptTemplateFamily(entry->model.get()));

        out->rendered_prompt = g_rendered_chat_prompt.c_str();
        out->tokenizer_type = entry->model->tokenizer_type.empty() ? nullptr : entry->model->tokenizer_type.c_str();
        out->chat_template = entry->model->chat_template.empty() ? nullptr : entry->model->chat_template.c_str();
        out->model_variant = g_rendered_model_variant.c_str();
        out->prompt_family = g_rendered_prompt_family.c_str();
        out->thinking_enabled = thinking_enabled ? 1 : 0;
        ClearError();
        return DENSECORE_STATUS_OK;
    });
}

int DenseCorePreviewTextRequest(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature,
                                float top_p, int top_k, float repetition_penalty, int json_mode,
                                DenseCoreRequestSnapshot* out) {
    return RequestApiBoundary("DenseCorePreviewTextRequest", [&]() -> int {
        (void)max_tokens;
        if (!handle || !prompt || !out) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCorePreviewTextRequest: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }

        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        ModelEntry* entry = state->GetDefaultModel();
        if (!entry || !entry->model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCorePreviewTextRequest: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        std::string rendered_prompt = MaybeApplyAutoChatTemplate(entry->model.get(), prompt);
        const bool template_applied = rendered_prompt != prompt;
        const std::string before_text_priming = rendered_prompt;
        MaybePrimeQwenNoThinkingPromptText(entry->model.get(), &rendered_prompt);
        const bool text_primed = rendered_prompt != before_text_priming;
        g_preview_token_ids = Tokenizer::Tokenize(entry->model.get(), rendered_prompt,
                                                  ResolveAddBosForPrompt(entry->model.get(), rendered_prompt));
        const std::vector<int> before_token_priming = g_preview_token_ids;
        MaybePrimeQwenNoThinking(entry->model.get(), &g_preview_token_ids);
        const bool token_primed = g_preview_token_ids != before_token_priming;
        g_rendered_chat_prompt = rendered_prompt;

        out->rendered_prompt = g_rendered_chat_prompt.c_str();
        out->token_ids = g_preview_token_ids.empty() ? nullptr : g_preview_token_ids.data();
        out->num_token_ids = static_cast<int>(g_preview_token_ids.size());
        out->submit_path = ResolveSubmitPathForPreview(false, json_mode != 0, true);
        const auto descriptor = densecore::models::DescribeModel(entry->model.get());
        g_rendered_model_variant = densecore::models::ModelVariantName(descriptor.variant);
        g_rendered_prompt_family = densecore::models::PromptTemplateFamilyName(
            densecore::models::ResolvePromptTemplateFamily(entry->model.get()));
        FillSnapshotMetadata(out, entry->model.get(), ResolveSubmitPathForPreview(false, json_mode != 0, true),
                             temperature, top_p, top_k, repetition_penalty, json_mode, template_applied, text_primed,
                             token_primed);
        ClearError();
        return DENSECORE_STATUS_OK;
    });
}

int DenseCorePreviewTokenRequest(DenseCoreHandle handle, const int* token_ids, int num_token_ids, int max_tokens,
                                 float temperature, float top_p, int top_k, float repetition_penalty, int json_mode,
                                 DenseCoreRequestSnapshot* out) {
    return RequestApiBoundary("DenseCorePreviewTokenRequest", [&]() -> int {
        (void)max_tokens;
        if (!handle || !token_ids || num_token_ids <= 0 || !out) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCorePreviewTokenRequest: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }

        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        ModelEntry* entry = state->GetDefaultModel();
        if (!entry || !entry->model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCorePreviewTokenRequest: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        g_preview_token_ids.assign(token_ids, token_ids + num_token_ids);
        const std::vector<int> before_token_priming = g_preview_token_ids;
        MaybePrimeQwenNoThinking(entry->model.get(), &g_preview_token_ids);
        const bool token_primed = g_preview_token_ids != before_token_priming;
        g_rendered_chat_prompt.clear();

        out->rendered_prompt = nullptr;
        out->token_ids = g_preview_token_ids.data();
        out->num_token_ids = static_cast<int>(g_preview_token_ids.size());
        const auto descriptor = densecore::models::DescribeModel(entry->model.get());
        g_rendered_model_variant = densecore::models::ModelVariantName(descriptor.variant);
        g_rendered_prompt_family = densecore::models::PromptTemplateFamilyName(
            densecore::models::ResolvePromptTemplateFamily(entry->model.get()));
        FillSnapshotMetadata(out, entry->model.get(), ResolveSubmitPathForPreview(true, json_mode != 0, true),
                             temperature, top_p, top_k, repetition_penalty, json_mode, false, false, token_primed);
        ClearError();
        return DENSECORE_STATUS_OK;
    });
}

namespace {

int BuildRequestSnapshotImpl(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature, float top_p,
                             int top_k, float repetition_penalty, int json_mode, bool input_already_rendered,
                             DenseCoreRequestSnapshot* out) {
    return RequestApiBoundary("DenseCoreBuildRequestSnapshot", [&]() -> int {
        (void)max_tokens;
        if (!handle || !prompt || !out) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreBuildRequestSnapshot: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        *out = DenseCoreRequestSnapshot{};

        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        ModelEntry* entry = state->GetDefaultModel();
        if (!entry || !entry->model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCoreBuildRequestSnapshot: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        std::string rendered_prompt =
            input_already_rendered ? std::string(prompt) : MaybeApplyAutoChatTemplate(entry->model.get(), prompt);
        const bool template_applied = !input_already_rendered && rendered_prompt != prompt;
        const std::string before_text_priming = rendered_prompt;
        if (!input_already_rendered) {
            MaybePrimeQwenNoThinkingPromptText(entry->model.get(), &rendered_prompt);
        }
        const bool text_primed = rendered_prompt != before_text_priming;
        std::vector<int> token_ids = Tokenizer::Tokenize(entry->model.get(), rendered_prompt,
                                                         ResolveAddBosForPrompt(entry->model.get(), rendered_prompt));
        const std::vector<int> before_token_priming = token_ids;
        if (!input_already_rendered) {
            MaybePrimeQwenNoThinking(entry->model.get(), &token_ids);
        }
        const bool token_primed = token_ids != before_token_priming;

        const auto descriptor = densecore::models::DescribeModel(entry->model.get());
        const std::string variant = densecore::models::ModelVariantName(descriptor.variant);
        const std::string prompt_family = densecore::models::PromptTemplateFamilyName(
            densecore::models::ResolvePromptTemplateFamily(entry->model.get()));

        char* owned_prompt = CopyCStringOwned(rendered_prompt);
        int* owned_tokens = CopyTokenIdsOwned(token_ids);
        char* owned_tokenizer = CopyCStringOwned(entry->model->tokenizer_type);
        char* owned_variant = CopyCStringOwned(variant);
        char* owned_family = CopyCStringOwned(prompt_family);
        if (!owned_prompt || (!token_ids.empty() && !owned_tokens)) {
            std::free(owned_prompt);
            std::free(owned_tokens);
            std::free(owned_tokenizer);
            std::free(owned_variant);
            std::free(owned_family);
            SetError(DENSECORE_STATUS_OUT_OF_MEMORY, "DenseCoreBuildRequestSnapshot: allocation failed");
            return DENSECORE_STATUS_OUT_OF_MEMORY;
        }

        out->rendered_prompt = owned_prompt;
        out->token_ids = owned_tokens;
        out->num_token_ids = static_cast<int>(token_ids.size());
        out->submit_path = ResolveSubmitPathForPreview(false, json_mode != 0, true);
        out->temperature = temperature;
        out->top_p = top_p;
        out->top_k = top_k;
        out->repetition_penalty = repetition_penalty;
        out->json_mode = json_mode != 0 ? 1 : 0;
        out->template_applied = template_applied ? 1 : 0;
        out->text_primed = text_primed ? 1 : 0;
        out->token_primed = token_primed ? 1 : 0;
        out->tokenizer_type = owned_tokenizer;
        out->model_variant = owned_variant;
        out->prompt_family = owned_family;
        out->caller_owns_buffers = 1;
        ClearError();
        return DENSECORE_STATUS_OK;
    });
}
}  // namespace

int DenseCoreBuildRequestSnapshot(DenseCoreHandle handle, const char* prompt, int max_tokens, float temperature,
                                  float top_p, int top_k, float repetition_penalty, int json_mode,
                                  DenseCoreRequestSnapshot* out) {
    return BuildRequestSnapshotImpl(handle, prompt, max_tokens, temperature, top_p, top_k, repetition_penalty,
                                    json_mode, /*input_already_rendered=*/false, out);
}

int DenseCoreBuildRenderedRequestSnapshot(DenseCoreHandle handle, const char* rendered_prompt, int max_tokens,
                                          float temperature, float top_p, int top_k, float repetition_penalty,
                                          int json_mode, DenseCoreRequestSnapshot* out) {
    return BuildRequestSnapshotImpl(handle, rendered_prompt, max_tokens, temperature, top_p, top_k, repetition_penalty,
                                    json_mode, /*input_already_rendered=*/true, out);
}

int DenseCoreSubmitRenderedChatWithSamplingConstraintsCallbackEx(
    DenseCoreHandle handle, const char* rendered_prompt, int max_tokens, const char* lora_name, float temperature,
    float top_p, int top_k, float repetition_penalty, const char** stop_sequences, int json_mode,
    const int* allowed_token_ids, int num_allowed_token_ids, int allowed_token_ids_strict,
    const int* disallowed_token_ids, int num_disallowed_token_ids, TokenCallbackEx callback, void* user_data) {
    DenseCoreRequestSnapshot snapshot{};
    const auto release_snapshot = [](DenseCoreRequestSnapshot* value) { DenseCoreFreeRequestSnapshot(value); };
    const std::unique_ptr<DenseCoreRequestSnapshot, decltype(release_snapshot)> snapshot_guard(&snapshot,
                                                                                               release_snapshot);
    const int status =
        BuildRequestSnapshotImpl(handle, rendered_prompt, max_tokens, temperature, top_p, top_k, repetition_penalty,
                                 json_mode, /*input_already_rendered=*/true, &snapshot);
    if (status < 0) {
        return status;
    }
    const int request_id = SubmitRequestIdsWithSamplingConstraintsImpl(
        handle, snapshot.token_ids, snapshot.num_token_ids, max_tokens, lora_name, temperature, top_p, top_k,
        repetition_penalty, stop_sequences, json_mode, allowed_token_ids, num_allowed_token_ids,
        allowed_token_ids_strict, disallowed_token_ids, num_disallowed_token_ids, nullptr, callback, user_data,
        snapshot.rendered_prompt, /*tokens_already_snapshot_primed=*/true,
        "DenseCoreSubmitRenderedChatWithSamplingConstraintsCallbackEx",
        "DenseCoreSubmitRenderedChatWithSamplingConstraintsCallbackEx");
    return request_id;
}

void DenseCoreFreeRequestSnapshot(DenseCoreRequestSnapshot* snapshot) {
    if (!snapshot) {
        return;
    }
    if (snapshot->caller_owns_buffers != 0) {
        std::free(const_cast<char*>(snapshot->rendered_prompt));
        std::free(const_cast<int*>(snapshot->token_ids));
        std::free(const_cast<char*>(snapshot->tokenizer_type));
        std::free(const_cast<char*>(snapshot->model_variant));
        std::free(const_cast<char*>(snapshot->prompt_family));
    }
    *snapshot = DenseCoreRequestSnapshot{};
}

int DenseCoreTokenizeText(DenseCoreHandle handle, const char* text, int add_bos, int add_eos, int* out_ids,
                          int max_ids) {
    return RequestApiBoundary("DenseCoreTokenizeText", [&]() -> int {
        if (!handle) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreTokenizeText: handle is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!text || !out_ids || max_ids <= 0) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreTokenizeText: invalid arguments");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        ModelEntry* entry = state->GetDefaultModel();
        if (!entry || !entry->model) {
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "DenseCoreTokenizeText: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }
        const std::vector<int> tokens = Tokenizer::Tokenize(entry->model.get(), text, add_bos != 0, add_eos != 0);
        const int count = std::min<int>(max_ids, static_cast<int>(tokens.size()));
        for (int i = 0; i < count; ++i) {
            out_ids[i] = tokens[static_cast<size_t>(i)];
        }
        ClearError();
        return count;
    });
}

int SubmitRequest(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                  TokenCallback callback, void* user_data) {
    return RequestApiBoundary("SubmitRequest", [&]() -> int {
        if (!handle) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequest: handle is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!prompt) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequest: prompt is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        // Get model for tokenization
        ModelEntry* model_entry = state->GetDefaultModel();
        if (!model_entry || !model_entry->model) {
            std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequest: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);

        req->prompt = prompt;
        req->lora_name = lora_name ? lora_name : "";
        req->max_tokens = max_tokens;
        req->callback = callback;
        req->user_data = user_data;

        // Tokenize immediately (outside hot path)
        PrepareTextRequest(model_entry->model.get(), req, prompt, TextPreparationPolicy::Legacy);
        ResetTokenIdConstraints(req);
        req->token_history = req->tokens;
        LogRequestRuntimePath(state, model_entry->model.get(), prompt, req);

        ApplyDefaultLora(state, req);
        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, "SubmitRequest: engine is draining");
    });
}

namespace {
int SubmitLegacyTokenRequest(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                             const char* lora_name, int json_mode, TokenCallback callback, void* user_data,
                             const char* api_name) {
    return RequestApiBoundary(api_name, [&]() -> int {
        if (!handle || !tokens || n_tokens <= 0) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, std::string(api_name) + ": invalid handle or tokens");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        auto* state = static_cast<EngineState*>(handle);
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);
        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);
        // Basic and format-ID APIs intentionally do not apply sampling-ID priming
        // or model blocklists. They preserve the caller's exact token sequence.
        req->tokens.assign(tokens, tokens + n_tokens);
        req->token_history = req->tokens;
        req->lora_name = lora_name ? lora_name : "";
        req->max_tokens = max_tokens;
        req->callback = callback;
        req->user_data = user_data;
        req->json_mode = (json_mode != 0);
        ApplyDefaultLora(state, req);
        SetupJsonGrammar(req, state);
        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, std::string(api_name) + ": engine is draining");
    });
}
}  // namespace

int SubmitRequestIds(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, TokenCallback callback,
                     void* user_data) {
    return SubmitLegacyTokenRequest(handle, tokens, n_tokens, max_tokens, nullptr, 0, callback, user_data,
                                    "SubmitRequestIds");
}

int SubmitRequestWithFormat(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode,
                            TokenCallback callback, void* user_data) {
    return SubmitRequestWithFormatEx(handle, prompt, max_tokens, nullptr, json_mode, callback, user_data);
}

int SubmitRequestWithFormatEx(DenseCoreHandle handle, const char* prompt, int max_tokens, const char* lora_name,
                              int json_mode, TokenCallback callback, void* user_data) {
    return RequestApiBoundary("SubmitRequestWithFormatEx", [&]() -> int {
        if (!handle) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithFormatEx: handle is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        if (!prompt) {
            SetError(DENSECORE_STATUS_INVALID_ARGUMENT, "SubmitRequestWithFormatEx: prompt is null");
            return DENSECORE_STATUS_INVALID_ARGUMENT;
        }
        EngineState* state = (EngineState*)handle;
        std::unique_lock<std::mutex> construction_lock(state->construction_mu);

        // Get model for tokenization
        ModelEntry* model_entry = state->GetDefaultModel();
        if (!model_entry || !model_entry->model) {
            std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
            SetError(DENSECORE_STATUS_MODEL_LOAD_FAILED, "SubmitRequestWithFormatEx: no model loaded");
            return DENSECORE_STATUS_MODEL_LOAD_FAILED;
        }

        Request* req = AcquireAndInitRequest(state);
        RequestGuard request_guard(state, req);

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
        PrepareTextRequest(model_entry->model.get(), req, prompt, TextPreparationPolicy::Legacy);
        ResetTokenIdConstraints(req);
        req->token_history = req->tokens;
        LogRequestRuntimePath(state, model_entry->model.get(), prompt, req);

        AssignGenerationTier(req);
        construction_lock.unlock();
        return PublishPreparedRequest(state, req, request_guard, "SubmitRequestWithFormatEx: engine is draining");
    });
}

int SubmitRequestIdsWithFormat(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens, int json_mode,
                               TokenCallback callback, void* user_data) {
    return SubmitRequestIdsWithFormatEx(handle, tokens, n_tokens, max_tokens, nullptr, json_mode, callback, user_data);
}

int SubmitRequestIdsWithFormatEx(DenseCoreHandle handle, const int* tokens, int n_tokens, int max_tokens,
                                 const char* lora_name, int json_mode, TokenCallback callback, void* user_data) {
    return SubmitLegacyTokenRequest(handle, tokens, n_tokens, max_tokens, lora_name, json_mode, callback, user_data,
                                    "SubmitRequestIdsWithFormatEx");
}
