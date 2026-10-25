#pragma once
#include <string>
#include <cstdint>
#include <functional>
#include "ggml.h"
#include "llm/speculative/qualification.h"
struct EngineState;
// Called before returning a newly initialized engine to callers. Selection is
// published atomically and remains fixed throughout that engine's lifetime.
bool ConfigureSpeculation(EngineState* state, int mode, int draft_n_max, std::string* error);

struct TransformerModel;
struct SpeculationMemoryCost {
    uint64_t head_kv = 0, prompt_hidden = 0, prompt_tokens = 0, head_metadata = 0, graph = 0, distributions = 0, checkpoints = 0;
    uint64_t Total() const;
};
// Shared by pre-KV concurrency selection and final MTP initialization.
SpeculationMemoryCost CalculateSpeculationMemory(const TransformerModel& model, uint64_t context,
                                                 uint64_t concurrency, int draft_n_max);

// Initialization-only, private dependency injection. Production uses the
// curated registry and native capture; tests can supply explicit proof data.
struct AutomaticSpeculationContext {
    const std::vector<densecore::llm::speculative::qualification::QualificationProfile>* profiles = nullptr;
    densecore::llm::speculative::qualification::LazyModelIdentity* identity = nullptr;
    std::function<densecore::llm::speculative::qualification::RuntimeConditions(int,int,ggml_type)> runtime;
};

inline densecore::llm::speculative::qualification::RuntimeConditions BindAutomaticPrefillChunk(
    densecore::llm::speculative::qualification::RuntimeConditions runtime, int chunk) {
    using namespace densecore::llm::speculative::qualification;
    if (!IsDigest(runtime.cpu.runtime_policy_sha256) || chunk < 1) {
        runtime.cpu.runtime_policy_sha256.clear();
    } else {
        runtime.cpu.runtime_policy_sha256 = Sha256Hex(runtime.cpu.runtime_policy_sha256 +
            "\neffective_prefill_chunk=" + std::to_string(chunk));
    }
    return runtime;
}
