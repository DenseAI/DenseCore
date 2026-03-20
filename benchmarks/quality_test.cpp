/**
 * @file quality_test.cpp
 * @brief Output quality verification - prints actual generated text
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "densecore.h"

struct QualityState {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> finished{false};
    std::vector<std::string> tokens;
};

namespace {
struct SamplingPreset {
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
};

// Realistic chat sampling for human quality checks.
constexpr SamplingPreset kChatPreset = {
    0.7f,
    0.9f,
    40,
    1.1f,
};

// Deterministic mode for parity checks.
constexpr SamplingPreset kParityPreset = {
    0.0f,
    1.0f,
    1,
    1.0f,
};

struct PromptCase {
    const char* prompt;
    int max_tokens;
};

const PromptCase kPromptSuite[] = {
    {"The capital of France is", 32},
    {"Explain what machine learning is in one paragraph.", 80},
    {"1 + 1 =", 16},
    {"Can you answer briefly in Korean: what is overfitting?", 64},
};

std::string BuildChatPrompt(const std::string& user_prompt) {
    // Keep user prompt raw; DenseCore can auto-apply model chat template.
    return user_prompt;
}

bool SetEnvVar(const char* key, const char* value) {
#if defined(_WIN32)
    return _putenv_s(key, value) == 0;
#else
    return setenv(key, value, 1) == 0;
#endif
}

std::string JoinTokens(const std::vector<std::string>& tokens) {
    std::string out;
    size_t reserve = 0;
    for (const auto& t : tokens) reserve += t.size();
    out.reserve(reserve);
    for (const auto& t : tokens) out += t;
    return out;
}

struct PromptRun {
    bool ok = false;
    std::vector<std::string> tokens;
    std::string output;
};
}  // namespace

void QualityCallback(const char* token, int is_final, void* user_data) {
    auto* s = static_cast<QualityState*>(user_data);
    if (!s) return;
    {
        std::lock_guard<std::mutex> lock(s->mu);
        if (token) s->tokens.emplace_back(token);
        if (is_final) s->finished.store(true);
    }
    s->cv.notify_one();
}

PromptRun RunPrompt(DenseCoreHandle engine, const char* user_prompt, int max_tokens, const SamplingPreset& sampling,
                    bool verbose) {
    if (verbose) {
        printf("\n--- User Prompt: \"%s\" (max_tokens=%d) ---\n", user_prompt, max_tokens);
    }

    QualityState state;
    const std::string prompt = BuildChatPrompt(user_prompt);
    const char* stop_sequences[] = {"\nUser:", "\nSystem:", nullptr};
    int req_id = SubmitRequestWithSamplingEx(engine, prompt.c_str(), max_tokens, nullptr, sampling.temperature,
                                             sampling.top_p, sampling.top_k, sampling.repetition_penalty, stop_sequences,
                                             0, QualityCallback, &state);
    if (req_id < 0) {
        printf("ERROR: SubmitRequestWithSamplingEx failed: %s\n", DenseCoreGetLastError());
        return {};
    }

    {
        std::unique_lock<std::mutex> lock(state.mu);
        if (!state.cv.wait_for(lock, std::chrono::seconds(60), [&] { return state.finished.load(); })) {
            printf("ERROR: request timed out\n");
            return {};
        }
    }

    PromptRun run;
    run.ok = true;
    run.tokens = std::move(state.tokens);
    run.output = JoinTokens(run.tokens);

    if (verbose) {
        printf("Output (%zu tokens): \"", run.tokens.size());
        printf("%s", run.output.c_str());
        printf("\"\n");
    }

    // Analyze quality
    bool all_same = true;
    for (size_t i = 1; i < run.tokens.size(); ++i) {
        if (run.tokens[i] != run.tokens[0]) {
            all_same = false;
            break;
        }
    }

    if (verbose && all_same && run.tokens.size() > 3) {
        printf("WARNING: All %zu tokens are identical (\"%s\") - GARBAGE OUTPUT!\n",
               run.tokens.size(), run.tokens[0].c_str());
    }

    // Check for non-ASCII garbage
    int garbage_count = 0;
    for (const auto& t : run.tokens) {
        for (char c : t) {
            const unsigned char uc = static_cast<unsigned char>(c);
            if (uc < 0x20 && uc != '\n' && uc != '\t' && uc != '\r') garbage_count++;
        }
    }
    if (verbose && garbage_count > 0) {
        printf("WARNING: Found %d non-printable characters - possible corruption\n", garbage_count);
    }

    return run;
}

bool RunQualitySuite(DenseCoreHandle engine, const SamplingPreset& sampling, bool verbose,
                     std::vector<std::string>* outputs = nullptr) {
    if (outputs) outputs->clear();
    for (const auto& tc : kPromptSuite) {
        PromptRun run = RunPrompt(engine, tc.prompt, tc.max_tokens, sampling, verbose);
        if (!run.ok) return false;
        if (outputs) outputs->push_back(std::move(run.output));
    }
    return true;
}

bool RunParityMode(const char* model_path, int threads, const char* paged_mode_rhs) {
    if (!SetEnvVar("DENSECORE_AUTO_CHAT_TEMPLATE", "1")) {
        printf("WARNING: failed to set DENSECORE_AUTO_CHAT_TEMPLATE\n");
    }

    if (!SetEnvVar("DENSECORE_PAGED_ATTN_DECODE_MODE", "off")) {
        printf("ERROR: failed to set DENSECORE_PAGED_ATTN_DECODE_MODE=off\n");
        return false;
    }
    DenseCoreHandle engine_off = InitEngine(model_path, nullptr, threads);
    if (!engine_off) {
        printf("ERROR: InitEngine(off) failed: %s\n", DenseCoreGetLastError());
        return false;
    }
    std::vector<std::string> outputs_off;
    bool ok_off = RunQualitySuite(engine_off, kParityPreset, false, &outputs_off);
    FreeEngine(engine_off);
    if (!ok_off) return false;

    if (!SetEnvVar("DENSECORE_PAGED_ATTN_DECODE_MODE", paged_mode_rhs)) {
        printf("ERROR: failed to set DENSECORE_PAGED_ATTN_DECODE_MODE=%s\n", paged_mode_rhs);
        return false;
    }
    DenseCoreHandle engine_rhs = InitEngine(model_path, nullptr, threads);
    if (!engine_rhs) {
        printf("ERROR: InitEngine(%s) failed: %s\n", paged_mode_rhs, DenseCoreGetLastError());
        return false;
    }
    std::vector<std::string> outputs_rhs;
    bool ok_rhs = RunQualitySuite(engine_rhs, kParityPreset, false, &outputs_rhs);
    FreeEngine(engine_rhs);
    if (!ok_rhs) return false;

    if (outputs_off.size() != outputs_rhs.size()) {
        printf("[PARITY] FAIL: output vector size mismatch (%zu vs %zu)\n", outputs_off.size(), outputs_rhs.size());
        return false;
    }

    bool parity_ok = true;
    for (size_t i = 0; i < outputs_off.size(); ++i) {
        if (outputs_off[i] != outputs_rhs[i]) {
            parity_ok = false;
            printf("[PARITY] MISMATCH prompt #%zu\n", i + 1);
            printf("  off : \"%s\"\n", outputs_off[i].c_str());
            printf("  %s: \"%s\"\n", paged_mode_rhs, outputs_rhs[i].c_str());
        }
    }

    if (!parity_ok) {
        printf("[PARITY] FAIL\n");
        return false;
    }

    printf("[PARITY] PASS (off vs %s)\n", paged_mode_rhs);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [threads] [--paged-mode off|auto|on] [--parity]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int threads = 0;
    const char* paged_mode = nullptr;
    bool parity_mode = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--paged-mode") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERROR: --paged-mode requires value: off|auto|on\n");
                return 1;
            }
            paged_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--parity") == 0) {
            parity_mode = true;
        } else {
            threads = atoi(argv[i]);
        }
    }

    if (!SetEnvVar("DENSECORE_AUTO_CHAT_TEMPLATE", "1")) {
        printf("WARNING: failed to set DENSECORE_AUTO_CHAT_TEMPLATE\n");
    }

    if (parity_mode) {
        const char* rhs_mode = paged_mode ? paged_mode : "on";
        const bool ok = RunParityMode(model_path, threads, rhs_mode);
        return ok ? 0 : 2;
    }

    if (paged_mode) {
        if (!SetEnvVar("DENSECORE_PAGED_ATTN_DECODE_MODE", paged_mode)) {
            fprintf(stderr, "ERROR: failed to set DENSECORE_PAGED_ATTN_DECODE_MODE=%s\n", paged_mode);
            return 1;
        }
    }

    printf("=== DenseCore Output Quality Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Paged decode mode: %s\n", paged_mode ? paged_mode : "(env default)");

    DenseCoreHandle engine = InitEngine(model_path, nullptr, threads);
    if (!engine) {
        fprintf(stderr, "InitEngine failed: %s\n", DenseCoreGetLastError());
        return 1;
    }

    if (!RunQualitySuite(engine, kChatPreset, true, nullptr)) {
        FreeEngine(engine);
        return 1;
    }

    FreeEngine(engine);
    printf("\n=== Quality test complete ===\n");
    return 0;
}
