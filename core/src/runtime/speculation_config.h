#pragma once
#include <string>
#include <cstdint>
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
