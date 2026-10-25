#include <gtest/gtest.h>

#include <limits>
#include <cmath>
#include <cstring>
#include <numeric>
#include "llm/sampling/sampling_rng.h"
#include "densecore/runtime/scheduler.h"
#include "llm/speculative/nextn_graph.h"
#include "ggml-cpu.h"
#include <random>
#include <stdexcept>

#include "llm/speculative/verification.h"

namespace {
using namespace densecore::llm::speculative;

TEST(NextNGraph, ExecutesHiddenTokenFusionAndSharedOutputHead) {
    TransformerModel source{};
    source.arch = ModelArch::QWEN35;
    source.variant = ModelVariant::QWEN35;
    source.arch_flags.is_hybrid_ssm = true;
    source.arch_flags.requires_q_norm = true;
    source.arch_flags.requires_k_norm = true;
    source.hparams.n_embd = 2;
    source.hparams.n_vocab = 2;
    source.hparams.n_head = 1;
    source.hparams.n_head_kv = 1;
    source.hparams.n_embd_head_k = 2;
    source.hparams.n_embd_head_v = 2;
    source.hparams.n_rot = 2;
    source.hparams.n_ctx = 16;
    source.hparams.n_layer = 4;
    source.hparams.n_ff = 2;
    source.ctx_w = ggml_init({1024 * 1024, nullptr, false});
    ASSERT_NE(source.ctx_w, nullptr);
    const auto weight = [&](int columns, int rows, float value) {
        auto* t = ggml_new_tensor_2d(source.ctx_w, GGML_TYPE_F32, columns, rows);
        std::fill_n(static_cast<float*>(t->data), columns * rows, value);
        return t;
    };
    source.tok_embeddings = weight(2, 2, 1);
    source.output_norm = weight(2, 1, 1);
    source.output = weight(2, 2, 0);
    const float identity[] = {1, 0, 0, 1};
    std::memcpy(source.output->data, identity, sizeof(identity));
    source.mtp_heads.resize(1);
    auto& head = source.mtp_heads[0];
    head.gguf_layer_index = 4;
    head.tensors["nextn.enorm.weight"] = weight(2, 1, 1);
    head.tensors["nextn.hnorm.weight"] = weight(2, 1, 1);
    auto* eh = weight(4, 2, 0);
    const float hidden_selector[] = {0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(eh->data, hidden_selector, sizeof(hidden_selector));
    head.tensors["nextn.eh_proj.weight"] = eh;
    for (const char* name : {"attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight",
                             "attn_k_norm.weight"}) head.tensors[name] = weight(2, 1, 1);
    for (const char* name : {"attn_k.weight", "attn_v.weight", "attn_output.weight",
                             "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
        head.tensors[name] = weight(2, 2, 0);
    }
    head.tensors["attn_q.weight"] = weight(2, 4, 0); // [q | gate] per head
    NextNGraph nextn(source);
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&nextn.Model(), 1, 16, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);
    const auto blocks = cache->block_manager->Allocate(1);
    ASSERT_EQ(blocks.size(), 1u);
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = {1}; batch.pos = {0}; batch.seq_id = {0}; batch.n_past = {0};
    batch.block_tables = {blocks};
    std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    SetCurrentWorkContext(work.get());
    ResetInferenceWorkContext(work.get());
    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    SetCurrentBatch(&batch);
    auto* ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    ASSERT_NE(ctx, nullptr);
    auto* hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    const float h[] = {3, 4};
    std::memcpy(hidden->data, h, sizeof(h));
    auto* graph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor* next_hidden = nullptr;
    auto* output = nextn.Build(ctx, cache.get(), batch, hidden, graph, &next_hidden);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(next_hidden, nullptr);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx, graph, 1), GGML_STATUS_SUCCESS);
    const auto* logits = static_cast<float*>(output->data);
    EXPECT_NEAR(logits[0], 3.0f / std::sqrt(12.5f), 2e-5f);
    EXPECT_NEAR(logits[1], 4.0f / std::sqrt(12.5f), 2e-5f);
    EXPECT_NEAR(static_cast<float*>(next_hidden->data)[0], logits[0], 1e-6f);
    EXPECT_NEAR(static_cast<float*>(next_hidden->data)[1], logits[1], 1e-6f);
    ggml_free(ctx);
    SetCurrentBatch(nullptr);
    SetCurrentWorkContext(nullptr);
    cache->block_manager->Free(blocks);
}

TEST(SpeculativeDistribution, MatchesRegularSamplerWithPenaltiesAndFilters) {
    ggml_init_params init{8192, nullptr, false};
    auto* ctx = ggml_init(init);
    ASSERT_NE(ctx, nullptr);
    auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 1);
    const float values[] = {-2, 1, 2, 1.5, -1, 0.5};
    std::memcpy(logits->data, values, sizeof(values));
    const std::vector<int> history = {2, 2, 1};
    const std::vector<int> banned = {0};
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode);
        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = mode == 0 ? 3 : 6;
        params.top_p = mode == 1 ? 0.75f : 1.0f;
        params.min_p = mode == 2 ? 0.3f : 0;
        params.repetition_penalty = 1.2f;
        params.frequency_penalty = 0.1f;
        params.presence_penalty = 0.1f;
        params.token_history = &history;
        params.disallowed_token_ids = &banned;
        const auto distribution = BuildSamplingDistribution(logits, 0, params);
        ASSERT_EQ(distribution.size(), 6u);
        EXPECT_NEAR(std::accumulate(distribution.begin(), distribution.end(), 0.0), 1, 1e-12);
        EXPECT_EQ(distribution[0], 0);
        std::mt19937 rng(91);
        std::vector<int> counts(6);
        constexpr int draws = 30000;
        for (int i = 0; i < draws; ++i) ++counts[SampleTokenWithRng(logits, 0, params, rng)];
        for (size_t token = 0; token < counts.size(); ++token) {
            EXPECT_NEAR(static_cast<double>(counts[token]) / draws, distribution[token], 0.012);
        }
    }
    ggml_free(ctx);
}

TEST(SpeculativeDistribution, GreedyAndGrammarUseRegularSamplerDistribution) {
    ggml_init_params init{8192, nullptr, false};
    auto* ctx = ggml_init(init);
    ASSERT_NE(ctx, nullptr);
    auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    const float values[] = {1, 3, 2};
    std::memcpy(logits->data, values, sizeof(values));
    SamplingParams params;
    params.temperature = 0;
    EXPECT_EQ(BuildSamplingDistribution(logits, 0, params), (Distribution{0, 1, 0}));
    const std::vector<std::string> vocab = {"{", "invalid", " "};
    GrammarConstraint grammar{};
    grammar.enabled = true;
    grammar.is_json_mode = true;
    grammar.state = JSONState::EXPECT_OBJECT_START;
    InitGrammarConstraint(&grammar, vocab);
    params.grammar = &grammar;
    params.vocab = &vocab;
    const auto distribution = BuildSamplingDistribution(logits, 0, params);
    std::mt19937 rng(1);
    const int token = SampleTokenWithRng(logits, 0, params, rng);
    EXPECT_EQ(distribution[token], 1);
    EXPECT_EQ(distribution[1], 0);
    EXPECT_THROW(BuildSamplingDistribution(nullptr, 0, params), std::invalid_argument);
    ggml_free(ctx);
}

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

TEST(SpeculativeRollback, ReleasesOnlyExclusiveSuffixAndResetsTailSlots) {
    BlockManager manager(32, BLOCK_SIZE);
    auto blocks = manager.Allocate(3);
    ASSERT_EQ(blocks.size(), 3u);
    const auto original = blocks;
    for (int id : blocks) manager.SetFilledSlots(id, BLOCK_SIZE);
    const int free_before = manager.GetFreeBlockCount();
    ASSERT_TRUE(manager.RollbackExclusiveSuffix(&blocks, BLOCK_SIZE + 2));
    EXPECT_EQ(blocks, (std::vector<int>{original[0], original[1]}));
    EXPECT_EQ(manager.GetFilledSlots(original[0]), BLOCK_SIZE);
    EXPECT_EQ(manager.GetFilledSlots(original[1]), 2);
    EXPECT_EQ(manager.GetFilledSlots(original[2]), 0);
    EXPECT_EQ(manager.GetRefCount(original[2]), 0);
    EXPECT_EQ(manager.GetFreeBlockCount(), free_before + 1);
    manager.Free(blocks);
    EXPECT_EQ(manager.GetFreeBlockCount(), 32);
}

TEST(SpeculativeRollback, RejectsBoundaryBeyondFilledSlotsWithoutMutation) {
    BlockManager manager(32, BLOCK_SIZE);
    auto blocks = manager.Allocate(2);
    const auto original = blocks;
    manager.SetFilledSlots(blocks.front(), 2);
    const int free_before = manager.GetFreeBlockCount();
    EXPECT_FALSE(manager.RollbackExclusiveSuffix(&blocks, 3));
    EXPECT_EQ(blocks, original);
    EXPECT_EQ(manager.GetFilledSlots(blocks.front()), 2);
    EXPECT_EQ(manager.GetFreeBlockCount(), free_before);
    manager.Free(blocks);
}

TEST(SpeculativeRollback, RejectsSharedTailWithoutChangingAnyOwnershipOrSlots) {
    BlockManager manager(32, BLOCK_SIZE);
    auto blocks = manager.Allocate(3);
    ASSERT_EQ(blocks.size(), 3u);
    const auto original = blocks;
    for (int id : blocks) manager.SetFilledSlots(id, BLOCK_SIZE);
    manager.Fork(blocks[1]);
    const int free_before = manager.GetFreeBlockCount();
    EXPECT_FALSE(manager.RollbackExclusiveSuffix(&blocks, BLOCK_SIZE + 2));
    EXPECT_EQ(blocks, original);
    EXPECT_EQ(manager.GetFilledSlots(blocks[1]), BLOCK_SIZE);
    EXPECT_EQ(manager.GetRefCount(blocks[1]), 2);
    EXPECT_EQ(manager.GetFreeBlockCount(), free_before);
    manager.FreeSingle(blocks[1]);
    // Cached residency is also immutable even with one active request reference.
    const int tokens[] = {1};
    manager.RegisterPrefixBlockWithTokens(blocks[1], 1234, tokens, 1);
    EXPECT_FALSE(manager.RollbackExclusiveSuffix(&blocks, BLOCK_SIZE + 2));
    EXPECT_EQ(blocks, original);
    manager.UnregisterPrefixBlock(blocks[1]);
    manager.Free(blocks);
}

TEST(SpeculativeRollback, SchedulerAndRecurrentStateRestoreTogetherAtBoundary) {
    BlockManager manager(32, BLOCK_SIZE);
    densecore::SchedulerConfig config;
    config.max_num_seqs = 1;
    config.max_prefill_seqs = 1;
    config.max_num_batched_tokens = 64;
    densecore::Scheduler scheduler(&manager, config);
    const int seq = scheduler.AddRequest(123, 15, 64);
    const auto prompt = scheduler.Schedule();
    ASSERT_EQ(prompt.new_block_allocations.size(), 1u);
    auto blocks = prompt.new_block_allocations[0].second;
    scheduler.OnPrefillChunkComplete(seq, 15, true);
    scheduler.OnDecodeStepComplete(seq, 1);
    const auto decode = scheduler.Schedule();
    for (const auto& allocation : decode.new_block_allocations) {
        blocks.insert(blocks.end(), allocation.second.begin(), allocation.second.end());
    }
    ASSERT_EQ(blocks.size(), 2u);
    scheduler.OnDecodeStepComplete(seq, 3);
    int n_past = 19;
    manager.SetFilledSlots(blocks.front(), BLOCK_SIZE);
    manager.SetFilledSlots(blocks.back(), 3);
    RecurrentCheckpoints::States states(1);
    states[0].conv_state = {1}; states[0].ssm_state = {2};
    RecurrentCheckpoints bank(3);
    bank.Capture(0, states);
    states[0].conv_state[0] = 10; states[0].ssm_state[0] = 20;
    bank.Capture(1, states);
    states[0].conv_state[0] = 100;
    ASSERT_TRUE(RestoreAcceptanceBoundary(scheduler, seq, 17, blocks, n_past, bank, 1, states));
    EXPECT_EQ(n_past, 17);
    EXPECT_EQ(states[0].conv_state[0], 10);
    EXPECT_EQ(manager.GetFilledSlots(blocks.back()), 1);
    EXPECT_FALSE(RestoreAcceptanceBoundary(scheduler, seq, 14, blocks, n_past, bank, 0, states));
    EXPECT_EQ(n_past, 17);
    EXPECT_EQ(states[0].conv_state[0], 10);
    const auto after = scheduler.Schedule();
    EXPECT_EQ(after.batch_context_len, 17);
    scheduler.RemoveRequest(seq);
    EXPECT_EQ(manager.GetFreeBlockCount(), 32);
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
