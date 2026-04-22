#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
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
#include "ggml.h"
#include "ggml-cpu.h"
#include "inference.h"
#include "model_loader.h"
#include "tokenizer.h"

namespace {

constexpr int kMaxNewTokens = 8;
constexpr int kTopK = 20;
constexpr float kRepetitionPenalty = 1.05f;
constexpr const char* kDefaultUserPrompt = "What is the capital of France? Answer in one short sentence.";

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

struct PromptArtifacts {
    std::string user_prompt;
    std::string prompt;
    std::vector<int> input_token_ids;
    std::vector<int> generated_token_ids;
    std::vector<std::string> debug_lines;
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
    const bool completed = collector.cv.wait_for(lock, std::chrono::minutes(5), [&]() { return collector.finished; });
    if (!completed) {
        throw std::runtime_error("generation timed out");
    }
    if (collector.error) {
        throw std::runtime_error(collector.error_text.empty() ? "generation failed" : collector.error_text);
    }
    return collector.token_ids;
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
            line.rfind("[HIDDEN_SNAPSHOT]", 0) == 0 || line.rfind("[GEMMA4_SHARED_KV]", 0) == 0 ||
            line.rfind("[KV_ROUNDTRIP]", 0) == 0 ||
            line.rfind("[GEMMA4_KV_WRITE]", 0) == 0 ||
            line.rfind("[RuntimePath]", 0) == 0) {
            lines.push_back(line);
        }
    }
    return lines;
}

PromptArtifacts BuildArtifacts(DenseCoreHandle handle, const std::string& user_prompt, const std::string& rendered_prompt) {
    PromptArtifacts artifacts;
    artifacts.user_prompt = user_prompt;
    ScopedEnvOverride disable_auto_template("DENSECORE_AUTO_CHAT_TEMPLATE", "0");
    ScopedEnvOverride enable_lm_head_top("DENSECORE_DEBUG_LM_HEAD_TOP", "20");
    const DenseCoreRequestSnapshot snapshot = PreviewRuntimeRequest(handle, rendered_prompt);
    artifacts.prompt = snapshot.rendered_prompt ? snapshot.rendered_prompt : rendered_prompt;
    artifacts.input_token_ids = TokenizeText(handle, rendered_prompt);
    const std::string captured = CaptureStderr([&]() { artifacts.generated_token_ids = GenerateTokenIds(handle, rendered_prompt); });
    artifacts.debug_lines = ExtractDebugLines(captured);
    return artifacts;
}

void WriteArtifactsJson(const PromptArtifacts& artifacts, const std::string& out_path) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"user_prompt\": \"" << JsonEscape(artifacts.user_prompt) << "\",\n";
    out << "  \"prompt\": \"" << JsonEscape(artifacts.prompt) << "\",\n";
    out << "  \"input_token_ids\": " << JsonArray(artifacts.input_token_ids) << ",\n";
    out << "  \"generated_token_ids\": " << JsonArray(artifacts.generated_token_ids) << ",\n";
    out << "  \"debug_lines\": " << JsonStringArray(artifacts.debug_lines) << "\n";
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

        DenseCoreHandle handle = InitEngine(gguf_path.c_str(), nullptr, /*threads=*/2);
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
