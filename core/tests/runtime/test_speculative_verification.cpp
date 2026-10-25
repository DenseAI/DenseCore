#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <stdexcept>

#include "llm/speculative/verification.h"

namespace {
using namespace densecore::llm::speculative;

TEST(SpeculativeVerification, GreedyAcceptsPrefixAndReturnsCorrectionWithoutRejectedSuffix) {
    std::mt19937 accept(7), target(8);
    const auto result = Verify({1, 0, 1}, {{0, 1}, {1, 0}, {0, 1}},
                               {{0, 1}, {0, 1}, {0, 1}, {1, 0}}, true, accept, target);
    EXPECT_EQ(result.accepted_draft_tokens, 1u);
    EXPECT_EQ(result.tokens, (std::vector<int>{1, 1}));
}

TEST(SpeculativeVerification, GreedyAllAcceptedGetsOneTargetBonusAndDoesNotConsumeRandomness) {
    std::mt19937 accept(7), target(8);
    const auto original_accept = accept, original_target = target;
    const auto result = Verify({0, 1}, {{1, 0}, {0, 1}}, {{1, 0}, {0, 1}, {1, 0}},
                               true, accept, target);
    EXPECT_EQ(result.accepted_draft_tokens, 2u);
    EXPECT_EQ(result.tokens, (std::vector<int>{0, 1, 0}));
    EXPECT_EQ(accept, original_accept);
    EXPECT_EQ(target, original_target);
}

TEST(SpeculativeVerification, RejectionUsesResidualRatherThanTargetDistribution) {
    std::mt19937 accept(7), target(8);
    // The proposed token is impossible under p; residual has only token 1.
    const auto result = Verify({0}, {{1, 0}}, {{0, 1}, {0, 1}}, false, accept, target);
    EXPECT_EQ(result.accepted_draft_tokens, 0u);
    EXPECT_EQ(result.tokens, (std::vector<int>{1}));
}

TEST(SpeculativeVerification, SamplingRecoversTargetMarginalAcrossAcceptAndReject) {
    std::mt19937 proposal(19), accept(29), target(39);
    std::discrete_distribution<int> propose({0.8, 0.2});
    int ones = 0;
    constexpr int trials = 100000;
    for (int i = 0; i < trials; ++i) {
        const auto result = Verify({propose(proposal)}, {{0.8, 0.2}}, {{0.3, 0.7}, {0.5, 0.5}},
                                   false, accept, target);
        ones += result.tokens.front() == 1;
    }
    EXPECT_NEAR(static_cast<double>(ones) / trials, 0.7, 0.005);
}

TEST(SpeculativeVerification, RejectsMalformedInputBeforeAdvancingRandomness) {
    std::mt19937 accept(7), target(8);
    const auto original_accept = accept, original_target = target;
    EXPECT_THROW(Verify({0}, {{1, 0}}, {{1, 0}}, false, accept, target), std::invalid_argument);
    EXPECT_THROW(Verify({0}, {{0, 1}}, {{1, 0}, {1, 0}}, false, accept, target), std::invalid_argument);
    EXPECT_THROW(Verify({0}, {{1, 0}}, {{0.4, 0.4}, {1, 0}}, false, accept, target), std::invalid_argument);
    EXPECT_THROW(Verify({0}, {{1, 0}}, {{std::numeric_limits<double>::quiet_NaN(), 0}, {1, 0}},
                        false, accept, target), std::invalid_argument);
    EXPECT_EQ(accept, original_accept);
    EXPECT_EQ(target, original_target);
}

TEST(SpeculativeCheckpoints, RestoresAllRecurrentAndConvStateAtAcceptanceBoundary) {
    RecurrentCheckpoints::States states(2);
    states[0].conv_state = {1, 2}; states[0].ssm_state = {3, 4};
    states[1].conv_state = {5}; states[1].ssm_state = {6};
    EXPECT_EQ(RecurrentCheckpoints::RequiredBytes(states, 2), 6u * sizeof(float) * 3);
    RecurrentCheckpoints bank(2);
    bank.Capture(0, states);
    states[0].conv_state[0] = 10; states[1].ssm_state[0] = 60;
    bank.Capture(1, states);
    states[0].conv_state[0] = 100; states[1].ssm_state[0] = 600;
    bank.Capture(2, states);
    bank.Restore(1, states);
    EXPECT_EQ(states[0].conv_state[0], 10);
    EXPECT_EQ(states[1].ssm_state[0], 60);
    states[0].conv_state[0] = 999;
    bank.Restore(1, states);
    EXPECT_EQ(states[0].conv_state[0], 10); // restoration does not alias checkpoint
    bank.Restore(0, states);
    EXPECT_EQ(states[0].conv_state[0], 1);
    EXPECT_EQ(states[1].ssm_state[0], 6);
    bank.Clear();
    EXPECT_THROW(bank.Restore(0, states), std::out_of_range);
    EXPECT_THROW(bank.Capture(3, states), std::out_of_range);
}

TEST(SpeculativeCheckpoints, RequiresBoundedDraftAndCapturedBoundary) {
    EXPECT_THROW(RecurrentCheckpoints(0), std::invalid_argument);
    EXPECT_THROW(RecurrentCheckpoints(4), std::invalid_argument);
    RecurrentCheckpoints bank(3);
    RecurrentCheckpoints::States states;
    EXPECT_THROW(bank.Restore(1, states), std::out_of_range);
}
}  // namespace
