#include "densecore/decision.h"
#include "gguf.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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
        check(DenseCoreDecisionLoadEx("unused", 1, 99, error, sizeof(error)) == nullptr, "invalid precision accepted");
        check(std::strstr(error, "precision") != nullptr, "missing precision diagnostic");
        check(DenseCoreDecisionMetadata(nullptr) == nullptr, "null metadata handle accepted");
        check(DenseCoreDecisionTokenizerJSON(nullptr) == nullptr, "null tokenizer handle accepted");
        DenseCoreDecisionFree(nullptr);
        float logits[2] = {123, 456}, acts[2] = {789, 101};
        int32_t tokens[3] = {50281, 50284, 50282}, markers[2] = {1, 1};
        check(!DenseCoreDecisionPredict(nullptr, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
              "null predict accepted");
        check(!DenseCoreDecisionPredictBatch(nullptr, nullptr, 0, nullptr, error, sizeof(error)),
              "null batch accepted");

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

        // A recognized architecture does not permit arbitrary dimensions or a
        // relabeled Laya checkpoint. Reject before touching any tensor payload.
        metadata = gguf_init_empty();
        gguf_set_val_str(metadata, "general.architecture", "julia");
        gguf_set_val_str(metadata, "julia.weight_layout", "ggml-canonical");
        gguf_set_val_u32(metadata, "julia.hidden_size", 384);
        gguf_set_val_u32(metadata, "julia.num_layers", 22);
        gguf_set_val_u32(metadata, "julia.num_heads", 16);
        check(gguf_write_to_file(metadata, path.string().c_str(), true), "cannot write Julia fixture");
        gguf_free(metadata);
        invalid = DenseCoreDecisionLoad(path.string().c_str(), 1, error, sizeof(error));
        std::filesystem::remove(path);
        check(invalid == nullptr, "Julia accepted Laya attention dimensions");
        check(std::strstr(error, "julia.num_heads") != nullptr, "missing Julia dimension diagnostic");

        if (argc > 1) {
            const int threads = argc > 2 ? std::stoi(argv[2]) : 2;
            const int precision = argc > 3 ? std::stoi(argv[3]) : DENSECORE_DECISION_FP32;
            auto* model = DenseCoreDecisionLoadEx(argv[1], threads, precision, error, sizeof(error));
            check(model != nullptr, error);
            const bool julia = std::strstr(DenseCoreDecisionMetadata(model), "\"general.architecture\":\"julia\"");
            check(std::strstr(DenseCoreDecisionMetadata(model),
                              julia ? "\"densecore.native_max_len\":4829" : "\"densecore.native_max_len\":512"),
                  "missing native context resource bound");
            check(std::strstr(DenseCoreDecisionMetadata(model), "\"densecore.weight_buffer\":\"raw_cpu\""),
                  "incorrect weight storage diagnostic");
            check(std::strstr(DenseCoreDecisionMetadata(model), "\"densecore.cpu_extra_buffer\":\"none\""),
                  "unverified extra CPU buffer claimed");
            check(std::strstr(DenseCoreDecisionMetadata(model), precision == DENSECORE_DECISION_Q8_0
                                                                    ? "\"densecore.execution_precision\":\"q8_0\""
                                                                    : "\"densecore.execution_precision\":\"fp32\""),
                  "incorrect execution precision diagnostic");
            const int32_t cls = julia ? 2 : 50281, mask = julia ? 4 : 50284, sep = julia ? 1 : 50282;
            const int32_t vocab = julia ? 256000 : 50368;
            tokens[0] = cls;
            tokens[1] = mask;
            tokens[2] = sep;
            check(std::strstr(DenseCoreDecisionMetadata(model), julia ? "julia.temperature.0" : "laya.temperature.0"),
                  "missing temperature metadata");
            check(std::strstr(DenseCoreDecisionTokenizerJSON(model), "added_tokens"), "missing tokenizer data");
            check(!DenseCoreDecisionPredict(model, tokens, 3, 3, markers, 1, logits, acts, error, sizeof(error)),
                  "bad question type accepted");
            check(!DenseCoreDecisionPredict(model, tokens, julia ? 8193 : 513, 0, markers, 1, logits, acts, error,
                                            sizeof(error)),
                  "overlong input accepted");
            if (julia) {
                check(!DenseCoreDecisionPredict(model, tokens, 4830, 0, markers, 1, logits, acts, error, sizeof(error)),
                      "oversized Julia graph accepted");
                check(std::strstr(error, "workspace") != nullptr, "missing Julia workspace diagnostic");
            }
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 2, logits, acts, error, sizeof(error)),
                  "duplicate markers accepted");
            markers[0] = 3;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "out of bounds marker accepted");
            markers[0] = 0;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "non-MASK marker accepted");
            tokens[0] = vocab;
            check(!DenseCoreDecisionPredict(model, tokens, 3, 0, markers, 1, logits, acts, error, sizeof(error)),
                  "out of range token accepted");
            check(logits[0] == 123 && acts[0] == 789, "failed inference changed output");

            // Equal lengths execute together; the third request exercises exact
            // length grouping. Mixed types, token IDs and marker counts must
            // retain their own attention, RoPE positions and action features.
            int32_t first[] = {cls, mask, sep};
            int32_t second[] = {mask, mask, sep};
            std::vector<int32_t> longest(512, 10);
            longest[0] = cls;
            longest[1] = mask;
            longest.back() = sep;
            int32_t one[] = {1}, two[] = {0, 1};
            DenseCoreDecisionInput inputs[] = {
                {first, 3, 0, one, 1}, {second, 3, 2, two, 2}, {longest.data(), longest.size(), 1, one, 1}};
            std::array<std::array<float, 2>, 3> single_logits{}, single_acts{}, batch_logits{}, batch_acts{};
            DenseCoreDecisionOutput outputs[3];
            for (size_t i = 0; i < 3; ++i) {
                const auto& in = inputs[i];
                check(DenseCoreDecisionPredict(model, in.tokens, in.token_count, in.question_type, in.markers,
                                               in.marker_count, single_logits[i].data(), single_acts[i].data(), error,
                                               sizeof(error)),
                      error);
                outputs[i] = {batch_logits[i].data(), batch_acts[i].data()};
            }
            // Validation of a later invalid request must not publish earlier
            // successful outputs or read from an invalid token buffer.
            const size_t saved_length = inputs[2].token_count;
            inputs[2].token_count = julia ? 8193 : 513;
            batch_logits[0][0] = 321;
            check(!DenseCoreDecisionPredictBatch(model, inputs, 3, outputs, error, sizeof(error)),
                  "invalid later batch request accepted");
            check(batch_logits[0][0] == 321, "failed batch changed output");
            inputs[2].token_count = saved_length;
            for (int order = 0; order < 2; ++order) {
                check(DenseCoreDecisionPredictBatch(model, inputs, 3, outputs, error, sizeof(error)), error);
                for (size_t i = 0; i < 3; ++i) {
                    const size_t reference = order ? 2 - i : i;
                    for (size_t j = 0; j < inputs[i].marker_count; ++j)
                        check(std::abs(batch_logits[i][j] - single_logits[reference][j]) <= 1e-3f,
                              "batch option logit mismatch");
                    for (size_t j = 0; j < 2; ++j)
                        check(std::abs(batch_acts[i][j] - single_acts[reference][j]) <=
                                  1e-3f + 1e-6f * std::abs(single_acts[reference][j]),
                              "batch action logit mismatch");
                }
                std::swap(inputs[0], inputs[2]);
            }
            check(DenseCoreDecisionPredict(model, first, 3, 0, one, 1, logits, acts, error, sizeof(error)), error);
            check(std::abs(logits[0] - single_logits[0][0]) <= 1e-3f,
                  "short request after maximum input changed output");
            DenseCoreDecisionFree(model);
        }
        std::puts("Laya ABI validation passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
