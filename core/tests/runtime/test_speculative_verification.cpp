#include <gtest/gtest.h>

#include <limits>
#include <cmath>
#include <cstring>
#include <numeric>
#include "llm/sampling/sampling_rng.h"
#include "densecore/runtime/scheduler.h"
#include "llm/speculative/nextn_graph.h"
#include "llm/graph/build_internal.h"
#include "llm/speculative/reference_session.h"
#include "runtime/speculation_config.h"
#include "runtime/engine_internal.h"
#include "ggml-cpu.h"
#include <random>
#include <stdexcept>

#include "llm/speculative/verification.h"

namespace densecore::llm::speculative {
struct ReferenceSessionTestAccess {
    static void TrimToTargetBoundary(ReferenceSession& session) {
        if (!session.cache_->block_manager->RollbackExclusiveSuffix(&session.blocks_, session.target_seen_))
            throw std::runtime_error("test head rollback failed");
        session.head_past_ = session.target_seen_;
    }
    static void AppendExpected(ReferenceSession& session, int token, const std::vector<float>& input_hidden,
                               const std::vector<float>& resulting_target_hidden) {
        session.ForwardHead({token}, input_hidden, nullptr, nullptr);
        ++session.target_seen_;
        session.last_hidden_ = resulting_target_hidden;
    }
    static std::vector<float> Probe(ReferenceSession& session, int token) {
        TrimToTargetBoundary(session);
        std::vector<float> logits;
        session.ForwardHead({token}, session.last_hidden_, nullptr, nullptr, &logits);
        return logits;
    }
};
}

namespace {
using namespace densecore::llm::speculative;

std::unique_ptr<TransformerModel> MakeTinyNextNSource() {
    auto owner = std::make_unique<TransformerModel>();
    auto& source = *owner;
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
    if (!source.ctx_w) throw std::bad_alloc();
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
    return owner;
}

TEST(MtpReferenceSession, BootstrapRetainsFirstTokenAndCarriesPreviousTargetHiddenAcrossChunks) {
    const std::vector<int> tokens = {10, 20, 30};
    const std::vector<float> hidden = {1, 2, 3, 4, 5, 6};
    const auto full = BuildNextNBootstrapBatch(tokens, hidden, 2, 0, 3);
    EXPECT_EQ(full.tokens, tokens);
    EXPECT_EQ(full.hidden, (std::vector<float>{0, 0, 1, 2, 3, 4}));
    const auto next_chunk = BuildNextNBootstrapBatch(tokens, hidden, 2, 1, 2);
    EXPECT_EQ(next_chunk.tokens, (std::vector<int>{20, 30}));
    EXPECT_EQ(next_chunk.hidden, (std::vector<float>{1, 2, 3, 4}));
    EXPECT_THROW(BuildNextNBootstrapBatch(tokens, hidden, 2, 2, 2), std::invalid_argument);
}

TEST(MtpReferenceSession, VerifiesGreedyProposalsAndRecoversFromRejection) {
    auto source = MakeTinyNextNSource();
    auto* ctx = ggml_init({1024 * 1024, nullptr, false});
    ASSERT_NE(ctx, nullptr);
    auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ReferenceSession session(*source, 16, 2, 1234, 1);
    SamplingParams params;
    params.temperature = 0;
    std::vector<int> history;
    params.token_history = &history;
    std::atomic<bool> cancelled{false};
    std::mt19937 rng(7);
    const int prompt = 1;
    const float first_hidden[] = {3, 4};
    const float second_hidden[] = {4, 3};
    const float first_logits[] = {1, 3};
    const float second_logits[] = {3, 1};
    session.ObserveTarget(&prompt, first_hidden, 2, 1, true);
    std::memcpy(logits->data, first_logits, sizeof(first_logits));
    EXPECT_EQ(session.Sample(logits, 0, params, rng, nullptr, cancelled), 1);
    EXPECT_GT(session.Proposed(), 0u);
    history.push_back(1);
    session.ObserveTarget(&prompt, second_hidden, 2, 1, false);
    std::memcpy(logits->data, second_logits, sizeof(second_logits));
    EXPECT_EQ(session.Sample(logits, 0, params, rng, nullptr, cancelled), 0);
    EXPECT_EQ(session.Accepted(), 0u);
    history.push_back(0);
    const int zero = 0;
    session.ObserveTarget(&zero, second_hidden, 2, 1, false);
    EXPECT_EQ(session.Sample(logits, 0, params, rng, nullptr, cancelled), 0);
    EXPECT_EQ(session.Accepted(), 1u);
    cancelled.store(true);
    EXPECT_THROW(session.Sample(logits, 0, params, rng, nullptr, cancelled), std::runtime_error);
    ggml_free(ctx);
}

TEST(MtpBatchedSession, EveryAcceptanceBoundaryAndFullAcceptCatchupSupportsAllDraftLimits) {
    for (int maximum = 1; maximum <= 3; ++maximum) for (int accept = 0; accept <= maximum; ++accept) {
        SCOPED_TRACE(maximum);
        SCOPED_TRACE(accept);
        auto source = MakeTinyNextNSource();
        auto* ctx = ggml_init({1024*1024, nullptr, false});
        ASSERT_NE(ctx, nullptr);
        auto* first = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
        static_cast<float*>(first->data)[0] = 1; static_cast<float*>(first->data)[1] = 3;
        ReferenceSession session(*source, 16, maximum, 1234, 1);
        SamplingParams params;
        params.temperature = 0;
        std::vector<int> history;
        params.token_history = &history;
        std::atomic<bool> cancelled{false};
        std::mt19937 rng(7);
        const int prompt = 1;
        const float hidden[] = {3, 4};
        session.ObserveTarget(&prompt, hidden, 2, 1, true);
        ASSERT_EQ(session.Sample(first, 0, params, rng, nullptr, cancelled), 1);
        history.push_back(1);
        const auto proposals = session.VerificationDraft(maximum);
        ASSERT_EQ(proposals, std::vector<int>(maximum, 1));
        auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, maximum+1);
        for (int row = 0; row <= maximum; ++row) {
            auto* values = reinterpret_cast<float*>(static_cast<char*>(logits->data) + row*logits->nb[1]);
            const int selected = row < accept ? 1 : 0;
            values[selected] = 3; values[1-selected] = 1;
        }
        const auto result = session.VerifyBatched(logits, 0, maximum, params, rng);
        ASSERT_EQ(result.accepted_draft_tokens, static_cast<size_t>(accept));
        std::vector<int> expected(accept, 1); expected.push_back(0);
        EXPECT_EQ(result.tokens, expected);
        std::vector<int> inputs(maximum+1, 1);
        std::vector<float> target_hidden;
        for (int row = 0; row <= maximum; ++row) target_hidden.insert(target_hidden.end(), {3,4});
        EXPECT_NO_THROW(session.CompleteBatched(inputs.data(), target_hidden.data(), 2,
            result, params, nullptr, cancelled));
        EXPECT_FALSE(session.VerificationDraft(maximum).empty());
        EXPECT_EQ(session.Verified(), static_cast<uint64_t>(std::min(accept+1, maximum)));
        EXPECT_EQ(session.Rejected(), static_cast<uint64_t>(accept != maximum));
        ggml_free(ctx);
    }
}

TEST(MtpBatchedSession, AcceptedEosDoesNotReadOrSampleThePostEosBonusDistribution) {
    auto source = MakeTinyNextNSource();
    source->eos_token_id = 1;
    auto* ctx = ggml_init({1024*1024, nullptr, false});
    ASSERT_NE(ctx, nullptr);
    auto* initial = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    static_cast<float*>(initial->data)[0] = 3; static_cast<float*>(initial->data)[1] = 1;
    ReferenceSession session(*source, 16, 3, 1234, 1);
    SamplingParams params;
    params.temperature = 0;
    std::vector<int> history;
    params.token_history = &history;
    std::atomic<bool> cancelled{false};
    std::mt19937 rng(7);
    const int prompt = 0;
    const float hidden[] = {3, 4};
    session.ObserveTarget(&prompt, hidden, 2, 1, true);
    ASSERT_EQ(session.Sample(initial, 0, params, rng, nullptr, cancelled), 0);
    history.push_back(0);
    ASSERT_EQ(session.VerificationDraft(3), (std::vector<int>{1}));
    auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    auto* values = static_cast<float*>(logits->data);
    values[0] = 1; values[1] = 3;
    values[2] = values[3] = std::numeric_limits<float>::quiet_NaN();
    const auto result = session.VerifyBatched(logits, 0, 1, params, rng);
    EXPECT_EQ(result.tokens, (std::vector<int>{1}));
    EXPECT_EQ(result.accepted_draft_tokens, 0u);
    const int input[] = {0, 1};
    const float states[] = {3,4,3,4};
    EXPECT_NO_THROW(session.CompleteBatched(input, states, 2, result, params, nullptr, cancelled));
    EXPECT_TRUE(session.VerificationDraft(3).empty());
    ggml_free(ctx);
}

TEST(MtpBatchedSession, AcceptedHeadPrefixMatchesSequentialActualTargetHiddenLogits) {
    for (int maximum = 1; maximum <= 3; ++maximum) for (int accept = 0; accept <= maximum; ++accept) {
        SCOPED_TRACE(maximum);
        SCOPED_TRACE(accept);
        auto source = MakeTinyNextNSource();
        auto& tensors = source->mtp_heads[0].tensors;
        const float identity[] = {1,0,0,1};
        for (const char* name : {"attn_k.weight", "attn_v.weight", "attn_output.weight"})
            std::memcpy(tensors[name]->data, identity, sizeof(identity));
        const float query[] = {1,0,0,1,0,0,0,0};
        std::memcpy(tensors["attn_q.weight"]->data, query, sizeof(query));
        auto* ctx = ggml_init({1024*1024, nullptr, false});
        ASSERT_NE(ctx, nullptr);
        auto* initial = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
        static_cast<float*>(initial->data)[0] = 1; static_cast<float*>(initial->data)[1] = 3;
        ReferenceSession batched(*source, 16, maximum, 1234, 1), sequential(*source, 16, maximum, 1234, 1);
        SamplingParams params; params.temperature = 0;
        std::vector<int> history;
        params.token_history = &history;
        std::atomic<bool> cancelled{false};
        std::mt19937 rng(7), other_rng(7);
        const int prompt = 1;
        const std::vector<float> anchor{3,4};
        batched.ObserveTarget(&prompt, anchor.data(), 2, 1, true);
        sequential.ObserveTarget(&prompt, anchor.data(), 2, 1, true);
        ASSERT_EQ(batched.Sample(initial, 0, params, rng, nullptr, cancelled), 1);
        ASSERT_EQ(sequential.Sample(initial, 0, params, other_rng, nullptr, cancelled), 1);
        history.push_back(1);
        const auto proposals = batched.VerificationDraft(maximum);
        ASSERT_EQ(proposals.size(), static_cast<size_t>(maximum));
        auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, maximum+1);
        for (int row = 0; row <= maximum; ++row) {
            const int selected = row < accept ? proposals[row] : (row < maximum ? 1-proposals[row] : 0);
            auto* values = reinterpret_cast<float*>(static_cast<char*>(logits->data) + row*logits->nb[1]);
            values[selected] = 3; values[1-selected] = 1;
        }
        const auto result = batched.VerifyBatched(logits, 0, maximum, params, rng);
        ASSERT_EQ(result.accepted_draft_tokens, static_cast<size_t>(accept));
        std::vector<int> inputs{1}; inputs.insert(inputs.end(), proposals.begin(), proposals.end());
        const std::vector<float> target_hidden{-4,1, 1,-5, -2,-1, 3,2};
        batched.CompleteBatched(inputs.data(), target_hidden.data(), 2, result, params, nullptr, cancelled);
        ReferenceSessionTestAccess::TrimToTargetBoundary(sequential);
        for (int row = 0; row <= accept; ++row) {
            const std::vector<float> input = row == 0 ? anchor :
                std::vector<float>(target_hidden.begin()+(row-1)*2, target_hidden.begin()+row*2);
            const std::vector<float> output(target_hidden.begin()+row*2, target_hidden.begin()+(row+1)*2);
            ReferenceSessionTestAccess::AppendExpected(sequential, inputs[row], input, output);
        }
        const auto actual = ReferenceSessionTestAccess::Probe(batched, 0);
        const auto expected = ReferenceSessionTestAccess::Probe(sequential, 0);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t token = 0; token < actual.size(); ++token) EXPECT_NEAR(actual[token], expected[token], 2e-5f);
        ggml_free(ctx);
    }
}

TEST(MtpConfiguration, KeepsAutomaticModeOffAndRejectsUnsupportedOrUnbudgetedExplicitMode) {
    EngineState state;
    std::string error;
    ASSERT_TRUE(ConfigureSpeculation(&state, 0, 2, &error));
    EXPECT_EQ(state.speculation_selected.load(), 1);
    EXPECT_FALSE(ConfigureSpeculation(&state, 2, 2, &error));
    EXPECT_FALSE(ConfigureSpeculation(&state, 0, 4, &error));
    auto entry = std::make_unique<ModelEntry>();
    entry->model = MakeTinyNextNSource();
    entry->kv_cache = std::make_unique<PagedKVCache>();
    entry->kv_config.max_seq_len = 16;
    entry->kv_config.max_num_seqs = 1;
    entry->kv_config.memory_plan.available_bytes = 1;
    state.default_model_id = "default";
    state.models.emplace("default", std::move(entry));
    EXPECT_FALSE(ConfigureSpeculation(&state, 2, 2, &error));
    EXPECT_NE(error.find("required_bytes="), std::string::npos);
    EXPECT_EQ(state.speculation_selected.load(), 1);
    state.models["default"]->kv_config.memory_plan.available_bytes = 1024ULL * 1024 * 1024;
    auto& plan = state.models["default"]->kv_config.memory_plan;
    plan.resource_capacity_bytes = plan.available_bytes;
    plan.kv_bytes = 1024; // ensure the synthetic base plan is registered
    state.models["default"]->resource_lease = std::make_unique<densecore::runtime::ResourcePlanLease>(plan);
    ASSERT_TRUE(ConfigureSpeculation(&state, 2, 2, &error)) << error;
    EXPECT_EQ(state.speculation_selected.load(), 2);
    EXPECT_EQ(state.speculation_max_context, 16);
    EXPECT_GT(state.speculation_reserved_bytes, 64ULL * 1024 * 1024);
    EXPECT_FALSE(ConfigureSpeculation(&state, 1, 2, &error));
}

TEST(NextNGraph, RejectsMalformedHeadBeforeWorkerExecution) {
    auto source = MakeTinyNextNSource();
    auto* original = source->mtp_heads[0].tensors["attn_q.weight"];
    source->mtp_heads[0].tensors["attn_q.weight"] = source->mtp_heads[0].tensors["ffn_up.weight"];
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
    source->mtp_heads[0].tensors["attn_q.weight"] = original;
    source->mtp_heads[0].tensors["attn_k_norm.weight"] = ggml_new_tensor_1d(source->ctx_w, GGML_TYPE_F16, 2);
    EXPECT_THROW((void)NextNGraph(*source), std::invalid_argument);
}

TEST(NextNGraph, TargetCaptureIsOutputNormalizedAndKeepsEveryPromptRow) {
    auto owner = MakeTinyNextNSource();
    const float embeddings[] = {1, 2, 3, 4};
    const float norm_weight[] = {2, 3};
    std::memcpy(owner->tok_embeddings->data, embeddings, sizeof(embeddings));
    std::memcpy(owner->output_norm->data, norm_weight, sizeof(norm_weight));
    NextNGraph view(*owner);
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&view.Model(), 1, 16, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);
    const auto blocks = cache->block_manager->Allocate(1);
    ASSERT_EQ(blocks.size(), 1u);
    BatchSpec batch{};
    batch.num_seqs = 1;
    batch.tokens = {0, 1}; batch.pos = {0, 1}; batch.seq_id = {0, 0}; batch.n_past = {0};
    batch.block_tables = {blocks};
    std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    SetCurrentWorkContext(work.get()); ResetInferenceWorkContext(work.get());
    SetCurrentExecutionPhase(InferenceExecutionPhase::Prefill); SetCurrentBatch(&batch);
    auto* ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    ASSERT_NE(ctx, nullptr);
    auto* graph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor* hidden = nullptr;
    GraphAssemblyOverrides overrides;
    overrides.full_normalized_hidden = &hidden;
    overrides.retain_all_logits = true;
    auto* output = BuildTransformerGraphInlineImpl(&view.Model(), cache.get(), ctx, batch,
                                                   false, graph, nullptr, nullptr, &overrides);
    ASSERT_NE(hidden, nullptr);
    ASSERT_EQ(hidden->ne[1], 2); // retain both rows despite last-token LM-head policy
    EXPECT_EQ(output->ne[1], 2); // verifier needs every target position
    ASSERT_EQ(ggml_graph_compute_with_ctx(ctx, graph, 1), GGML_STATUS_SUCCESS);
    const auto* data = static_cast<float*>(hidden->data);
    for (size_t row = 0; row < 2; ++row) {
        const float a = embeddings[row * 2], b = embeddings[row * 2 + 1];
        const float inverse = 1 / std::sqrt((a * a + b * b) / 2 + owner->hparams.f_norm_rms_eps);
        EXPECT_NEAR(data[row * 2], a * inverse * 2, 1e-5f);
        EXPECT_NEAR(data[row * 2 + 1], b * inverse * 3, 1e-5f);
    }
    const auto* logits = static_cast<float*>(output->data);
    EXPECT_NEAR(logits[0], data[2], 1e-6f);
    EXPECT_NEAR(logits[1], data[3], 1e-6f);
    ggml_free(ctx); ClearCurrentBatch(); SetCurrentWorkContext(nullptr);
    cache->block_manager->Free(blocks);
}

TEST(NextNGraph, ExecutesHiddenTokenFusionAndSharedOutputHead) {
    auto owner = MakeTinyNextNSource();
    auto& source = *owner;
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
    ClearCurrentBatch();
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
    ASSERT_TRUE(decode.new_block_allocations.empty()); // ordinary decode grows blocks in the worker
    ASSERT_TRUE(scheduler.ReserveSpeculativeSlots(seq, 19, &blocks));
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
    // Ordinary blocks remain worker-owned; scheduler removal only drops metadata.
    manager.Free(blocks);
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
