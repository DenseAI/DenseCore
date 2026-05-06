#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "densecore.h"
#include "densecore/backend/cpu_backend.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "densecore/runtime/inference.h"
#include "densecore/models/model_loader.h"
#include "densecore/models/tokenizer.h"

namespace {

constexpr int kMaxNewTokens = 8;
constexpr int kTopK = 20;
constexpr float kRepetitionPenalty = 1.05f;
constexpr const char* kDefaultUserPrompt = "What is the capital of France? Answer in one short sentence.";

int ParsePositiveEnvOrDefault(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

std::chrono::seconds ProbeGenerationTimeout() {
    return std::chrono::seconds(ParsePositiveEnvOrDefault("DENSECORE_PARITY_PROBE_TIMEOUT_SEC", 300));
}

int ProbeThreadCount() {
    return ParsePositiveEnvOrDefault("DENSECORE_PARITY_PROBE_THREADS", 2);
}

struct ScopedEnvOverride {
    explicit ScopedEnvOverride(const char* name, const char* value) : name_(name) {
        const char* current = std::getenv(name_);
        if (current) {
            had_original_ = true;
            original_ = current;
        }
        if (value) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

    ~ScopedEnvOverride() {
        if (had_original_) {
            setenv(name_, original_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_original_ = false;
    std::string original_;
};

struct TokenResultCollector {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int> token_ids;
    bool finished = false;
    bool error = false;
    std::string error_text;
};

struct StreamCollector {
    std::mutex mu;
    std::condition_variable cv;
    bool finished = false;
    bool error = false;
    std::string error_text;
};

struct PromptArtifacts {
    std::string user_prompt;
    std::string prompt;
    std::vector<int> input_token_ids;
    uint64_t input_token_hash = 0;
    bool parity_trace_enabled = false;
    std::vector<int> token_results_ids;
    std::vector<int> sampling_constraints_ids;
    int first_sampled_token_id = -1;
    std::vector<std::string> debug_lines;
    std::vector<densecore::CpuBackend::MoEPathTraceEntry> moe_path_trace;
    uint64_t moe_forward_invocation_count = 0;
    uint64_t moe_graph_wiring_count = 0;
    uint64_t moe_callback_entry_count = 0;
    uint64_t moe_callback_missing_userdata_count = 0;
    uint64_t moe_callback_missing_backend_count = 0;
    uint64_t moe_callback_missing_experts_count = 0;
    uint64_t moe_callback_routing_failure_count = 0;
    uint64_t moe_callback_empty_routing_count = 0;
    uint64_t moe_callback_fail_closed_count = 0;
    std::vector<SamplingDebugTraceEntry> sampling_trace;
};

std::string JsonEscape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
                    << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

template <typename T>
std::string JsonArray(const std::vector<T>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string JsonStringArray(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << "\"" << JsonEscape(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

uint64_t HashTokenIds(const std::vector<int>& token_ids) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(token_ids.size()));
    for (int token_id : token_ids) {
        mix(static_cast<uint64_t>(static_cast<uint32_t>(token_id)));
    }
    return hash;
}

void OnTokenResult(const TokenResult* result, void* user_data) {
    auto* collector = static_cast<TokenResultCollector*>(user_data);
    std::lock_guard<std::mutex> lock(collector->mu);
    if (!result) {
        collector->error = true;
        collector->finished = true;
        collector->error_text = "null TokenResult";
        collector->cv.notify_all();
        return;
    }
    if (!result->is_finished && result->token_id >= 0) {
        collector->token_ids.push_back(result->token_id);
    }
    if (result->is_finished) {
        collector->finished = true;
        collector->cv.notify_all();
    }
}

void OnStreamToken(const char* token, int is_finished, void* user_data) {
    (void)token;
    auto* collector = static_cast<StreamCollector*>(user_data);
    std::lock_guard<std::mutex> lock(collector->mu);
    if (is_finished) {
        collector->finished = true;
        collector->cv.notify_all();
    }
}

std::vector<int> TokenizeText(DenseCoreHandle handle, const std::string& text) {
    DenseCoreRequestSnapshot snapshot{};
    const int rc = DenseCorePreviewTextRequest(handle, text.c_str(), kMaxNewTokens,
                                               /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/1,
                                               /*repetition_penalty=*/kRepetitionPenalty, /*json_mode=*/0, &snapshot);
    if (rc != 0) {
        throw std::runtime_error(DenseCoreGetLastError());
    }
    if (!snapshot.token_ids || snapshot.num_token_ids <= 0) {
        return {};
    }
    return std::vector<int>(snapshot.token_ids, snapshot.token_ids + snapshot.num_token_ids);
}

DenseCoreRequestSnapshot PreviewRuntimeRequest(DenseCoreHandle handle, const std::string& text) {
    DenseCoreRequestSnapshot snapshot{};
    const int rc = DenseCorePreviewTextRequest(handle, text.c_str(), kMaxNewTokens,
                                               /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/1,
                                               /*repetition_penalty=*/kRepetitionPenalty, /*json_mode=*/0, &snapshot);
    if (rc != 0) {
        throw std::runtime_error(DenseCoreGetLastError());
    }
    return snapshot;
}

std::string RenderChatPrompt(DenseCoreHandle handle, const std::string& prompt) {
    DenseCoreChatMessage message{};
    message.role = "user";
    message.content = prompt.c_str();
    DenseCoreChatTemplateOptions options{};
    options.enable_thinking = -1;
    DenseCoreRenderedChatPrompt rendered{};
    const int rc = DenseCoreRenderChatPrompt(handle, &message, 1, &options, &rendered);
    if (rc != 0) {
        throw std::runtime_error(DenseCoreGetLastError());
    }
    return rendered.rendered_prompt ? rendered.rendered_prompt : "";
}

std::vector<int> GenerateTokenIds(DenseCoreHandle handle, const std::string& prompt) {
    TokenResultCollector collector;
    const int request_id = SubmitRequestWithTokenResults(handle, prompt.c_str(), kMaxNewTokens,
                                                         /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/1,
                                                         /*repetition_penalty=*/kRepetitionPenalty, OnTokenResult,
                                                         &collector);
    if (request_id < 0) {
        throw std::runtime_error(DenseCoreGetLastError());
    }

    std::unique_lock<std::mutex> lock(collector.mu);
    const bool completed = collector.cv.wait_for(lock, ProbeGenerationTimeout(), [&]() { return collector.finished; });
    if (!completed) {
        throw std::runtime_error("generation timed out");
    }
    if (collector.error) {
        throw std::runtime_error(collector.error_text.empty() ? "generation failed" : collector.error_text);
    }
    return collector.token_ids;
}

std::vector<int> GenerateTokenIdsWithSamplingConstraints(DenseCoreHandle handle, const std::string& prompt) {
    StreamCollector collector;
    const int request_id = SubmitRequestWithSamplingConstraintsEx(
        handle, prompt.c_str(), kMaxNewTokens, /*lora_name=*/nullptr, /*temperature=*/0.0f, /*top_p=*/1.0f, /*top_k=*/1,
        /*repetition_penalty=*/kRepetitionPenalty, /*stop_sequences=*/nullptr, /*json_mode=*/0,
        /*allowed_token_ids=*/nullptr, /*num_allowed_token_ids=*/0, /*allowed_token_ids_strict=*/0,
        /*disallowed_token_ids=*/nullptr, /*num_disallowed_token_ids=*/0, OnStreamToken, &collector);
    if (request_id < 0) {
        throw std::runtime_error(DenseCoreGetLastError());
    }

    std::unique_lock<std::mutex> lock(collector.mu);
    const bool completed = collector.cv.wait_for(lock, ProbeGenerationTimeout(), [&]() { return collector.finished; });
    if (!completed) {
        throw std::runtime_error("generation timed out");
    }
    if (collector.error) {
        throw std::runtime_error(collector.error_text.empty() ? "generation failed" : collector.error_text);
    }
    std::vector<int> ids;
    for (const auto& trace : GetSamplingDebugTraceSnapshot()) {
        if (trace.request_id == request_id) {
            ids.push_back(trace.sampled_token_id);
        }
    }
    return ids;
}

std::string CaptureStderr(const std::function<void()>& fn) {
    char tmp_path[] = "/tmp/gemma4-parity-probe-XXXXXX";
    const int tmp_fd = mkstemp(tmp_path);
    if (tmp_fd < 0) {
        throw std::runtime_error("mkstemp failed for stderr capture");
    }

    const int saved_stderr = dup(STDERR_FILENO);
    if (saved_stderr < 0) {
        close(tmp_fd);
        unlink(tmp_path);
        throw std::runtime_error("dup(STDERR_FILENO) failed");
    }

    fflush(stderr);
    if (dup2(tmp_fd, STDERR_FILENO) < 0) {
        close(saved_stderr);
        close(tmp_fd);
        unlink(tmp_path);
        throw std::runtime_error("dup2(stderr) failed");
    }

    try {
        fn();
        fflush(stderr);
    } catch (...) {
        fflush(stderr);
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
        lseek(tmp_fd, 0, SEEK_SET);
        std::ifstream file(tmp_path);
        std::string ignored((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        close(tmp_fd);
        unlink(tmp_path);
        throw;
    }

    if (dup2(saved_stderr, STDERR_FILENO) < 0) {
        close(saved_stderr);
        close(tmp_fd);
        unlink(tmp_path);
        throw std::runtime_error("dup2(restore stderr) failed");
    }
    close(saved_stderr);

    lseek(tmp_fd, 0, SEEK_SET);
    std::ifstream file(tmp_path);
    const std::string captured((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    close(tmp_fd);
    unlink(tmp_path);
    return captured;
}

std::vector<std::string> ExtractDebugLines(const std::string& captured_stderr) {
    std::vector<std::string> lines;
    std::istringstream input(captured_stderr);
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("[LM_HEAD_TOP]", 0) == 0 || line.rfind("  [TOP]", 0) == 0 ||
            line.rfind("[SAMPLE_TOP]", 0) == 0 || line.rfind("[SAMPLE_DBG", 0) == 0 ||
            line.rfind("[HIDDEN_SNAPSHOT]", 0) == 0 || line.rfind("[GEMMA4_SHARED_KV]", 0) == 0 ||
            line.rfind("[KV_ROUNDTRIP]", 0) == 0 ||
            line.rfind("[GEMMA4_KV_WRITE]", 0) == 0 ||
            line.rfind("[RuntimePath]", 0) == 0 || line.rfind("[REQUEST_STATE]", 0) == 0 ||
            line.rfind("[ATTN_CORE_REF]", 0) == 0 || line.rfind("[ATTN_POST_REF]", 0) == 0 ||
            line.rfind("[ADD_RMS_REF]", 0) == 0 || line.rfind("[SHARED_GATE_REF]", 0) == 0 ||
            line.rfind("[MOE_LOADER_LAYER]", 0) == 0 ||
            line.rfind("[MOE_WIRING_MODEL]", 0) == 0 || line.rfind("[MOE_WIRING_LAYER]", 0) == 0 ||
            line.rfind("[MOE_WIRING_SUMMARY]", 0) == 0 ||
            line.rfind("[MOE_GRAPH_SUMMARY]", 0) == 0 || line.rfind("[MOE_GRAPH_LAYER]", 0) == 0 ||
            line.rfind("[DBG] cb_moe_forward", 0) == 0 || line.rfind("[MOE_TRACE_ROUTE]", 0) == 0 ||
            line.rfind("[MOE_TRACE_OUT]", 0) == 0 || line.rfind("[MOE_STAGE]", 0) == 0 ||
            line.rfind("[GEMMA4_PACKED]", 0) == 0) {
            lines.push_back(line);
        }
    }
    return lines;
}

PromptArtifacts BuildArtifacts(DenseCoreHandle handle, const std::string& user_prompt, const std::string& rendered_prompt) {
    PromptArtifacts artifacts;
    artifacts.user_prompt = user_prompt;
    const bool parity_trace = []() {
        const char* env = std::getenv("DENSECORE_GEMMA4_PARITY_TRACE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    ScopedEnvOverride disable_auto_template("DENSECORE_AUTO_CHAT_TEMPLATE", "0");
    ScopedEnvOverride enable_lm_head_top("DENSECORE_DEBUG_LM_HEAD_TOP", "20");
    ScopedEnvOverride enable_request_state("DENSECORE_DEBUG_REQUEST_STATE", "1");
    ScopedEnvOverride enable_sampler_trace("DENSECORE_DEBUG_SAMPLER_TRACE", "1");
    ScopedEnvOverride enable_gemma4_packed_checksum("DENSECORE_DEBUG_GEMMA4_PACKED_CHECKSUM", "1");
    ScopedEnvOverride enable_moe_trace("DENSECORE_DEBUG_MOE_TRACE", parity_trace ? "1" : nullptr);
    ScopedEnvOverride select_moe_token("DENSECORE_DEBUG_MOE_TOKEN", parity_trace ? "0" : nullptr);
    ScopedEnvOverride enable_sample_top("DENSECORE_DEBUG_SAMPLE", parity_trace ? "1" : nullptr);
    ScopedEnvOverride enable_shared_kv("DENSECORE_DEBUG_GEMMA4_SHARED_KV", parity_trace ? "1" : nullptr);
    ScopedEnvOverride enable_attention_path("DENSECORE_LOG_DECODE_ATTENTION_PATH", parity_trace ? "1" : nullptr);
    ScopedEnvOverride enable_attention_core("DENSECORE_DEBUG_ATTN_CORE_REFERENCE", parity_trace ? "1" : nullptr);
    ScopedEnvOverride attention_core_max_calls("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_MAX_CALLS",
                                               parity_trace ? "8" : nullptr);
    ScopedEnvOverride enable_attention_post("DENSECORE_DEBUG_ATTN_POST_REFERENCE", parity_trace ? "1" : nullptr);
    ScopedEnvOverride attention_post_max_calls("DENSECORE_DEBUG_ATTN_POST_REFERENCE_MAX_CALLS",
                                               parity_trace ? "8" : nullptr);
    const DenseCoreRequestSnapshot snapshot = PreviewRuntimeRequest(handle, rendered_prompt);
    artifacts.prompt = snapshot.rendered_prompt ? snapshot.rendered_prompt : rendered_prompt;
    artifacts.input_token_ids = TokenizeText(handle, rendered_prompt);
    artifacts.input_token_hash = HashTokenIds(artifacts.input_token_ids);
    artifacts.parity_trace_enabled = parity_trace;
    densecore::GetTelemetryCpuBackend().ResetMoEPathTrace();
    ResetMoECallbackEntryCounter();
    ResetSamplingDebugTrace();
    const std::string captured = CaptureStderr([&]() {
        artifacts.token_results_ids = GenerateTokenIds(handle, rendered_prompt);
        artifacts.sampling_constraints_ids = GenerateTokenIdsWithSamplingConstraints(handle, rendered_prompt);
    });
    artifacts.debug_lines = ExtractDebugLines(captured);
    artifacts.moe_path_trace = densecore::GetTelemetryCpuBackend().GetMoEPathTraceSnapshot();
    artifacts.moe_forward_invocation_count = densecore::GetTelemetryCpuBackend().GetMoEForwardInvocationCount();
    artifacts.moe_graph_wiring_count = GetMoEGraphWiringDebugCounter();
    artifacts.moe_callback_entry_count = GetMoECallbackEntryCounter();
    artifacts.moe_callback_missing_userdata_count = GetMoECallbackMissingUserdataCounter();
    artifacts.moe_callback_missing_backend_count = GetMoECallbackMissingBackendCounter();
    artifacts.moe_callback_missing_experts_count = GetMoECallbackMissingExpertsCounter();
    artifacts.moe_callback_routing_failure_count = GetMoECallbackRoutingFailureCounter();
    artifacts.moe_callback_empty_routing_count = GetMoECallbackEmptyRoutingCounter();
    artifacts.moe_callback_fail_closed_count = GetMoECallbackFailClosedCounter();
    artifacts.sampling_trace = GetSamplingDebugTraceSnapshot();
    if (!artifacts.sampling_trace.empty()) {
        artifacts.first_sampled_token_id = artifacts.sampling_trace.front().sampled_token_id;
    } else if (!artifacts.token_results_ids.empty()) {
        artifacts.first_sampled_token_id = artifacts.token_results_ids.front();
    }
    return artifacts;
}

std::string JsonMoEPathTrace(const std::vector<densecore::CpuBackend::MoEPathTraceEntry>& entries) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i) out << ", ";
        out << "{";
        out << "\"layer_idx\": " << entries[i].layer_idx << ", ";
        out << "\"seq_id\": " << entries[i].seq_id << ", ";
        out << "\"token_idx\": " << entries[i].token_idx << ", ";
        out << "\"decode_step\": " << entries[i].decode_step << ", ";
        out << "\"n_past\": " << entries[i].n_past << ", ";
        out << "\"expert_id\": " << entries[i].expert_id << ", ";
        out << "\"force_safe_reference\": " << (entries[i].force_safe_reference ? "true" : "false") << ", ";
        out << "\"safe_reference_mode\": " << (entries[i].safe_reference_mode ? "true" : "false") << ", ";
        out << "\"projection\": \"" << entries[i].projection << "\", ";
        out << "\"selected_path\": " << static_cast<int>(entries[i].selected_path);
        out << "}";
    }
    out << "]";
    return out.str();
}

std::string JsonSamplingTrace(const std::vector<SamplingDebugTraceEntry>& entries) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i) out << ", ";
        out << "{";
        out << "\"request_id\": " << entries[i].request_id << ", ";
        out << "\"output_token_index\": " << entries[i].output_token_index << ", ";
        out << "\"sampled_token_id\": " << entries[i].sampled_token_id << ", ";
        out << "\"top_pre_penalty\": [";
        for (size_t j = 0; j < entries[i].top_pre_penalty.size(); ++j) {
            if (j) out << ", ";
            const auto& c = entries[i].top_pre_penalty[j];
            out << "{\"token_id\": " << c.token_id << ", \"pre_penalty_logit\": " << c.pre_penalty_logit
                << ", \"post_penalty_logit\": " << c.post_penalty_logit << "}";
        }
        out << "], ";
        out << "\"top_post_penalty\": [";
        for (size_t j = 0; j < entries[i].top_post_penalty.size(); ++j) {
            if (j) out << ", ";
            const auto& c = entries[i].top_post_penalty[j];
            out << "{\"token_id\": " << c.token_id << ", \"pre_penalty_logit\": " << c.pre_penalty_logit
                << ", \"post_penalty_logit\": " << c.post_penalty_logit << "}";
        }
        out << "]";
        out << "}";
    }
    out << "]";
    return out.str();
}

void WriteArtifactsJson(const PromptArtifacts& artifacts, const std::string& out_path) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"user_prompt\": \"" << JsonEscape(artifacts.user_prompt) << "\",\n";
    out << "  \"prompt\": \"" << JsonEscape(artifacts.prompt) << "\",\n";
    out << "  \"input_token_ids\": " << JsonArray(artifacts.input_token_ids) << ",\n";
    out << "  \"input_token_hash\": \"0x" << std::hex << artifacts.input_token_hash << std::dec << "\",\n";
    out << "  \"parity_trace_enabled\": " << (artifacts.parity_trace_enabled ? "true" : "false") << ",\n";
    out << "  \"token_results_ids\": " << JsonArray(artifacts.token_results_ids) << ",\n";
    out << "  \"sampling_constraints_ids\": " << JsonArray(artifacts.sampling_constraints_ids) << ",\n";
    out << "  \"first_sampled_token_id\": " << artifacts.first_sampled_token_id << ",\n";
    out << "  \"debug_lines\": " << JsonStringArray(artifacts.debug_lines) << ",\n";
    out << "  \"moe_forward_invocation_count\": " << artifacts.moe_forward_invocation_count << ",\n";
    out << "  \"moe_graph_wiring_count\": " << artifacts.moe_graph_wiring_count << ",\n";
    out << "  \"moe_callback_entry_count\": " << artifacts.moe_callback_entry_count << ",\n";
    out << "  \"moe_callback_missing_userdata_count\": " << artifacts.moe_callback_missing_userdata_count << ",\n";
    out << "  \"moe_callback_missing_backend_count\": " << artifacts.moe_callback_missing_backend_count << ",\n";
    out << "  \"moe_callback_missing_experts_count\": " << artifacts.moe_callback_missing_experts_count << ",\n";
    out << "  \"moe_callback_routing_failure_count\": " << artifacts.moe_callback_routing_failure_count << ",\n";
    out << "  \"moe_callback_empty_routing_count\": " << artifacts.moe_callback_empty_routing_count << ",\n";
    out << "  \"moe_callback_fail_closed_count\": " << artifacts.moe_callback_fail_closed_count << ",\n";
    out << "  \"moe_path_trace\": " << JsonMoEPathTrace(artifacts.moe_path_trace) << ",\n";
    out << "  \"sampling_trace\": " << JsonSamplingTrace(artifacts.sampling_trace) << "\n";
    out << "}\n";

    std::ofstream file(out_path);
    file << out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: gemma4_parity_probe <gguf_path> <output_json> [user_prompt]\n";
            return 2;
        }

        const std::string gguf_path = argv[1];
        const std::string output_json = argv[2];
        const std::string user_prompt = argc == 4 ? argv[3] : std::string(kDefaultUserPrompt);

        DenseCoreHandle handle = InitEngine(gguf_path.c_str(), nullptr, ProbeThreadCount());
        if (!handle) {
            throw std::runtime_error(DenseCoreGetLastError());
        }
        const std::string chat_prompt = RenderChatPrompt(handle, user_prompt);
        const PromptArtifacts artifacts = BuildArtifacts(handle, user_prompt, chat_prompt);
        WriteArtifactsJson(artifacts, output_json);
        FreeEngine(handle);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[gemma4_parity_probe] " << ex.what() << "\n";
        return 1;
    }
}
