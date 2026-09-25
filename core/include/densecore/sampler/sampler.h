#pragma once

#include <cstdint>
#include <vector>

#include "densecore/hal/tensor.h"

namespace densecore {
namespace sampler {

struct SamplingParams {
    int top_k = 1;       // 1 = Greedy
    float top_p = 1.0f;  // 1.0 = Disable Top-P
    float temperature = 1.0f;
    float repetition_penalty = 1.0f;
    const std::vector<int>* token_history = nullptr;
    uint64_t seed = 42;
};

class Sampler {
public:
    // Expects logits shaped [batch, vocab] or [vocab]. Samples from batch 0.
    int Sample(const Tensor& logits, const SamplingParams& params) const;

private:
    mutable std::vector<float> workspace_probs_;
    mutable std::vector<int> workspace_indices_;
};

}  // namespace sampler
}  // namespace densecore
