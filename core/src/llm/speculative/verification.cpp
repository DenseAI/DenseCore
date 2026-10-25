#include "llm/speculative/verification.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace densecore::llm::speculative {
namespace {
void Validate(const Distribution& p, size_t vocab) {
    if (p.empty() || p.size() != vocab) throw std::invalid_argument("speculative vocabulary mismatch");
    double sum = 0;
    for (const double value : p) {
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid speculative probability");
        sum += value;
    }
    if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1e-5) {
        throw std::invalid_argument("speculative distribution must be normalized");
    }
}
int Draw(const Distribution& p, std::mt19937& rng) {
    // generate_canonical is in [0,1); avoid a zero-probability token at either
    // endpoint, including when a rounded cumulative sum falls short of one.
    const double u = std::generate_canonical<double, 53>(rng);
    double cumulative = 0;
    int last_positive = -1;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] <= 0) continue;
        last_positive = static_cast<int>(i);
        cumulative += p[i];
        if (u < cumulative) return static_cast<int>(i);
    }
    return last_positive;
}
int Argmax(const Distribution& p) {
    return static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
}
}  // namespace

VerificationResult Verify(const std::vector<int>& draft_tokens, const std::vector<Distribution>& draft,
                          const std::vector<Distribution>& target, bool greedy,
                          std::mt19937& accept_rng, std::mt19937& target_rng) {
    const size_t k = draft_tokens.size();
    if (k == 0 || k > 3 || draft.size() != k || target.size() != k + 1) {
        throw std::invalid_argument("speculative verification requires K=1..3 and K+1 target positions");
    }
    const size_t vocab = target.front().size();
    for (const auto& p : target) Validate(p, vocab);
    for (size_t i = 0; i < k; ++i) {
        Validate(draft[i], vocab);
        if (draft_tokens[i] < 0 || static_cast<size_t>(draft_tokens[i]) >= vocab ||
            draft[i][draft_tokens[i]] <= 0) {
            throw std::invalid_argument("draft token has no probability mass");
        }
    }
    VerificationResult result;
    result.tokens.reserve(k + 1);
    for (size_t i = 0; i < k; ++i) {
        const int token = draft_tokens[i];
        const auto& p = target[i];
        const auto& q = draft[i];
        const bool accepted = greedy ? token == Argmax(p) :
            std::generate_canonical<double, 53>(accept_rng) < std::min(1.0, p[token] / q[token]);
        if (accepted) {
            result.tokens.push_back(token);
            ++result.accepted_draft_tokens;
            continue;
        }
        if (greedy) {
            result.tokens.push_back(Argmax(p));
        } else {
            Distribution residual(vocab);
            double sum = 0;
            for (size_t t = 0; t < vocab; ++t) {
                residual[t] = std::max(0.0, p[t] - q[t]);
                sum += residual[t];
            }
            // A genuine rejection implies nonzero residual mass. Never replace
            // it with p, which would change the target sampling distribution.
            if (!(sum > 0) || !std::isfinite(sum)) throw std::runtime_error("empty speculative residual");
            for (double& mass : residual) mass /= sum;
            result.tokens.push_back(Draw(residual, target_rng));
        }
        return result;
    }
    result.tokens.push_back(greedy ? Argmax(target[k]) : Draw(target[k], target_rng));
    return result;
}

RecurrentCheckpoints::RecurrentCheckpoints(size_t draft_n_max) {
    if (draft_n_max < 1 || draft_n_max > 3) throw std::invalid_argument("draft_n_max must be 1..3");
    states_.resize(draft_n_max + 1);
    valid_.resize(draft_n_max + 1, false);
}
size_t RecurrentCheckpoints::RequiredBytes(const States& states, size_t draft_n_max) {
    if (draft_n_max < 1 || draft_n_max > 3) throw std::invalid_argument("draft_n_max must be 1..3");
    const size_t max = std::numeric_limits<size_t>::max();
    size_t elements = 0;
    for (const auto& state : states) {
        for (size_t count : {state.conv_state.size(), state.ssm_state.size()}) {
            if (count > max - elements) throw std::overflow_error("MTP checkpoint size overflow");
            elements += count;
        }
    }
    const size_t multiplier = sizeof(float) * (draft_n_max + 1);
    if (elements > max / multiplier) throw std::overflow_error("MTP checkpoint size overflow");
    return elements * multiplier;
}
void RecurrentCheckpoints::Capture(size_t boundary, const States& states) {
    if (boundary >= states_.size()) throw std::out_of_range("MTP checkpoint boundary");
    states_[boundary] = states;
    valid_[boundary] = true;
}
void RecurrentCheckpoints::Restore(size_t boundary, States& states) const {
    if (boundary >= states_.size() || !valid_[boundary]) throw std::out_of_range("MTP checkpoint unavailable");
    states = states_[boundary];
}
void RecurrentCheckpoints::Clear() {
    for (auto& state : states_) state.clear();
    std::fill(valid_.begin(), valid_.end(), false);
}
}  // namespace densecore::llm::speculative
