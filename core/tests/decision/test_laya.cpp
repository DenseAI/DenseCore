#include "densecore/decision.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}  // namespace

// No downloaded checkpoint is required for these malformed-input checks.
// Passing a checkpoint as argv[1] additionally verifies pre-inference guards.
int main(int argc, char** argv) {
    char error[512] = {};
    try {
        check(DenseCoreDecisionLoad(nullptr, 1, error, sizeof(error)) == nullptr, "null path accepted");
        check(std::strstr(error, "path") != nullptr, "missing path diagnostic");
        check(DenseCoreDecisionLoad("unused", 0, error, sizeof(error)) == nullptr, "zero threads accepted");
        check(DenseCoreDecisionMetadata(nullptr) == nullptr, "null metadata handle accepted");
        check(DenseCoreDecisionTokenizerJSON(nullptr) == nullptr, "null tokenizer handle accepted");
        DenseCoreDecisionFree(nullptr);
        float logits[2] = {123, 456}, acts[2] = {789, 101};
        int32_t tokens[3] = {50281, 50284, 50282}, markers[2] = {1, 1};
        check(!DenseCoreDecisionPredict(nullptr, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
              "null predict accepted");

        auto path = std::filesystem::temp_directory_path() /
                    ("densecore-invalid-laya-" + std::to_string(reinterpret_cast<uintptr_t>(error)) + ".gguf");
        auto* metadata = gguf_init_empty();
        gguf_set_val_str(metadata, "general.architecture", "llama");
        check(gguf_write_to_file(metadata, path.string().c_str(), true), "cannot write test fixture");
        gguf_free(metadata);
        auto* invalid = DenseCoreDecisionLoad(path.string().c_str(), 1, error, sizeof(error));
        std::filesystem::remove(path);
        check(invalid == nullptr, "non-Laya architecture accepted");
        check(std::strstr(error, "architecture") != nullptr, "missing architecture diagnostic");

        if (argc > 1) {
            auto* model = DenseCoreDecisionLoad(argv[1], 1, error, sizeof(error));
            check(model != nullptr, error);
            check(std::strstr(DenseCoreDecisionMetadata(model), "laya.temperature.0"), "missing temperature metadata");
            check(std::strstr(DenseCoreDecisionTokenizerJSON(model), "added_tokens"), "missing tokenizer data");
            check(!DenseCoreDecisionPredict(model, tokens, 3, 3, markers, 1, logits, acts, error, sizeof(error)),
                  "bad question type accepted");
            check(!DenseCoreDecisionPredict(model, tokens, 513, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "overlong input accepted");
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 2, logits, acts, error, sizeof(error)),
                  "duplicate markers accepted");
            markers[0] = 3;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "out of bounds marker accepted");
            markers[0] = 0;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "non-MASK marker accepted");
            tokens[0] = 50368;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "out of range token accepted");
            check(logits[0] == 123 && acts[0] == 789, "failed inference changed output");
            DenseCoreDecisionFree(model);
        }
        std::puts("Laya ABI validation passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
