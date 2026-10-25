#ifndef DENSECORE_LLM_SPECULATIVE_VERIFICATION_H
#define DENSECORE_LLM_SPECULATIVE_VERIFICATION_H

#include <cstddef>
#include <random>
#include <vector>

#include "densecore/llm/ssm/state_types.h"

namespace densecore { class Scheduler; }

namespace densecore::llm::speculative {

// Indexed by vocabulary token ID, after every sampler transform (including
// grammar). Draft and target must use the same transform order and vocabulary.
using Distribution = std::vector<double>;
struct TokenVerificationResult { int token; bool accepted; };
TokenVerificationResult VerifyToken(int draft_token, const Distribution& draft, const Distribution& target,
                                    bool greedy, std::mt19937& accept_rng, std::mt19937& target_rng);

struct VerificationResult {
    size_t accepted_draft_tokens = 0;
    // Accepted draft prefix followed by one residual/target bonus token. Only
    // these tokens may enter the ordinary output commit path.
    std::vector<int> tokens;
};

// Target has K+1 positions; draft has K. Acceptance and target draws use
// independent request-owned RNG streams. Throws invalid_argument on malformed
// distributions instead of silently biasing the sampler.
VerificationResult Verify(const std::vector<int>& draft_tokens,
                          const std::vector<Distribution>& draft,
                          const std::vector<Distribution>& target,
                          bool greedy, std::mt19937& accept_rng, std::mt19937& target_rng);

// The caller captures state after each verified token, including boundary zero.
// This owns deep copies; restoring cannot alias or mutate a prefix snapshot.
// KV/scheduler rollback is a separate required part of the same transaction.
class RecurrentCheckpoints {
public:
    using States = std::vector<ssm::SSMSequenceRuntimeState>;
    explicit RecurrentCheckpoints(size_t draft_n_max);
    static size_t RequiredBytes(const States& states, size_t draft_n_max);
    void Capture(size_t boundary, const States& states);
    void Restore(size_t boundary, States& states) const;
    void Clear();
private:
    std::vector<States> states_;
    std::vector<bool> valid_;
};

// Restores a prepared recurrent boundary and KV/scheduler/request state as one
// worker operation. Copy/allocation happens before destructive KV rollback.
bool RestoreAcceptanceBoundary(Scheduler& scheduler, int seq_id, int keep_tokens,
                               std::vector<int>& block_ids, int& n_past,
                               const RecurrentCheckpoints& checkpoints, size_t boundary,
                               RecurrentCheckpoints::States& recurrent_states);

}  // namespace densecore::llm::speculative
#endif
