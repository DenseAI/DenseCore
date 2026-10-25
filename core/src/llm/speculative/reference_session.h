#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include "llm/speculative/nextn_graph.h"
#include "llm/speculative/verification.h"

namespace densecore::llm::speculative {

struct NextNBootstrapBatch {
    std::vector<int> tokens;
    std::vector<float> hidden;
};
// Prompt token t is paired with normalized target h[t-1]; t=0 uses zeros.
// Absolute positions remain unchanged, including across bootstrap chunks.
NextNBootstrapBatch BuildNextNBootstrapBatch(const std::vector<int>& prompt,
                                             const std::vector<float>& normalized_hidden,
                                             size_t embedding, size_t begin, size_t count);

// Correctness reference: target verification remains one ordinary decode step
// per worker iteration. K head proposals are generated ahead, but no unverified
// token enters the output path or the target KV/recurrent state. This is not a
// performance path and cannot qualify an automatic MTP profile.
class ReferenceSession {
public:
    ReferenceSession(TransformerModel& model, int max_context, int draft_n_max, uint64_t seed, int threads);
    void ObserveTarget(const int* tokens, const float* hidden, size_t stride, int count, bool prefill);
    int Sample(ggml_tensor* target_logits, int column, const SamplingParams& params, std::mt19937& target_rng,
               const InferenceDependencies* deps, const std::atomic<bool>& cancelled);
    uint64_t Accepted() const { return accepted_; }
    uint64_t Proposed() const { return proposed_; }
private:
    struct HeadOutput { std::vector<float> hidden; Distribution distribution; int token = -1; };
    struct Proposal { int token; Distribution distribution; };
    HeadOutput ForwardHead(const std::vector<int>& tokens, const std::vector<float>& hidden,
                           const SamplingParams* params, const InferenceDependencies* deps);
    void Bootstrap(const InferenceDependencies* deps, const std::atomic<bool>& cancelled);
    void MakeProposals(int selected_token, const SamplingParams& params, const InferenceDependencies* deps,
                       const std::atomic<bool>& cancelled);
    TransformerModel& source_;
    NextNGraph head_;
    std::unique_ptr<PagedKVCache> cache_;
    std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work_;
    std::vector<int> blocks_, prompt_tokens_;
    std::vector<float> prompt_hidden_, last_hidden_;
    std::deque<Proposal> pending_;
    std::mt19937 draft_rng_, accept_rng_;
    int max_context_, draft_n_max_, threads_, head_past_ = 0, target_seen_ = 0;
    bool bootstrapped_ = false;
    uint64_t accepted_ = 0, proposed_ = 0;
};
}  // namespace densecore::llm::speculative
