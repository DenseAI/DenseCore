/**
 * @file test_sampling_range.cpp
 * @brief Tests for action-token range sampling optimization
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "densecore/runtime/inference.h"
#include "runtime/worker_internal.h"

namespace {

struct GgmlContextGuard {
    ggml_context* ctx = nullptr;
    ~GgmlContextGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

ggml_tensor* MakeLogitsTensor(ggml_context* ctx, const std::vector<float>& logits) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int64_t>(logits.size()), 1);
    std::memcpy(t->data, logits.data(), logits.size() * sizeof(float));
    return t;
}

TEST(JsonGrammarConstraint, RejectsCombinedStructuralTokensAndConsumesWholeTokens) {
    const std::vector<std::string> vocab = {"{", "\"", "key", "\":", ":", "value", "}", " ", "\"value\""};
    GrammarConstraint grammar{};
    grammar.enabled = true;
    grammar.is_json_mode = true;
    grammar.state = JSONState::EXPECT_OBJECT_START;
    InitGrammarConstraint(&grammar, vocab);

    std::vector<float> logits(vocab.size(), 1.0f);
    ApplyGrammarMask(logits.data(), static_cast<int>(logits.size()), &grammar, vocab);
    EXPECT_TRUE(std::isfinite(logits[0]));
    EXPECT_FALSE(std::isfinite(logits[3]));

    grammar.UpdateState("{");
    grammar.UpdateState("\"");
    grammar.UpdateState("key");
    grammar.UpdateState("\"");
    EXPECT_EQ(grammar.state, JSONState::EXPECT_COLON);

    logits.assign(vocab.size(), 1.0f);
    ApplyGrammarMask(logits.data(), static_cast<int>(logits.size()), &grammar, vocab);
    EXPECT_TRUE(std::isfinite(logits[4]));
    EXPECT_FALSE(std::isfinite(logits[3]));

    grammar.UpdateState(":");
    grammar.UpdateState("\"");
    grammar.UpdateState("value");
    grammar.UpdateState("\"");
    grammar.UpdateState("}");
    EXPECT_EQ(grammar.state, JSONState::COMPLETED);
}

int SampleTokenLegacyReference(ggml_tensor* logits, int idx, const SamplingParams& params) {
    if (!logits || !logits->data) return 0;

    float* logits_data = static_cast<float*>(logits->data);
    const int n_vocab = static_cast<int>(logits->ne[0]);
    if (n_vocab <= 0) return 0;

    const int n_cols = std::max<int>(1, static_cast<int>(logits->ne[1]));
    idx = std::clamp(idx, 0, n_cols - 1);
    const ptrdiff_t row_stride = static_cast<ptrdiff_t>(logits->nb[1] / sizeof(float));
    if (row_stride < n_vocab) return 0;

    float* last_logits = logits_data + static_cast<ptrdiff_t>(idx) * row_stride;

    int range_start = 0;
    int range_end = n_vocab;
    if (params.action_token_count > 0 && params.grammar == nullptr) {
        const int requested_start = std::max(0, params.action_token_start);
        if (requested_start < n_vocab) {
            const int64_t requested_end = static_cast<int64_t>(requested_start) + params.action_token_count;
            if (requested_end > requested_start) {
                range_start = requested_start;
                range_end = static_cast<int>(std::min<int64_t>(n_vocab, requested_end));
            }
        }
    }
    if (range_end <= range_start) {
        range_start = 0;
        range_end = n_vocab;
    }
    const int active_vocab = range_end - range_start;

    auto is_disallowed = [&](int token_id) -> bool {
        if (!params.disallowed_token_ids || params.disallowed_token_ids->empty()) return false;
        return std::binary_search(params.disallowed_token_ids->begin(), params.disallowed_token_ids->end(), token_id);
    };
    auto finite_argmax_raw = [&]() -> int {
        int best_idx = range_start;
        float best_val = -INFINITY;
        bool found = false;
        for (int i = range_start; i < range_end; ++i) {
            const float v = last_logits[i];
            if (!std::isfinite(v) || is_disallowed(i)) continue;
            if (!found || v > best_val) {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        return found ? best_idx : range_start;
    };

    const bool has_history = params.token_history && !params.token_history->empty();
    const bool has_penalty = has_history && (params.repetition_penalty != 1.0f || params.frequency_penalty != 0.0f ||
                                             params.presence_penalty != 0.0f);
    if (params.temperature <= 0.0f && params.grammar == nullptr && !has_penalty) {
        return finite_argmax_raw();
    }
    if (params.temperature == 1.0f && params.top_k <= 1 && params.top_p >= 1.0f && params.min_p <= 0.0f &&
        params.grammar == nullptr && !has_penalty) {
        return finite_argmax_raw();
    }

    std::vector<float> working_logits(last_logits + range_start, last_logits + range_end);
    if (params.disallowed_token_ids && !params.disallowed_token_ids->empty()) {
        for (int token_id : *params.disallowed_token_ids) {
            if (token_id >= range_start && token_id < range_end) {
                working_logits[static_cast<size_t>(token_id - range_start)] = -INFINITY;
            }
        }
    }
    if (params.grammar && params.vocab) {
        ApplyGrammarMask(working_logits.data(), active_vocab, params.grammar, *params.vocab);
    }

    if (params.repetition_penalty != 1.0f && params.token_history && !params.token_history->empty()) {
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                const int local_token = token - range_start;
                if (working_logits[static_cast<size_t>(local_token)] < 0.0f) {
                    working_logits[static_cast<size_t>(local_token)] *= params.repetition_penalty;
                } else {
                    working_logits[static_cast<size_t>(local_token)] /= params.repetition_penalty;
                }
            }
        }
    }

    if ((params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) && params.token_history &&
        !params.token_history->empty()) {
        std::unordered_map<int, int> token_counts;
        token_counts.reserve(params.token_history->size());
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                token_counts[token - range_start]++;
            }
        }
        for (const auto& kv : token_counts) {
            const int token = kv.first;
            const int count = kv.second;
            working_logits[static_cast<size_t>(token)] -=
                params.frequency_penalty * count + params.presence_penalty * (count > 0 ? 1.0f : 0.0f);
        }
    }

    auto finite_argmax = [&]() -> int {
        int best_idx = 0;
        float best_val = -INFINITY;
        bool found = false;
        for (int i = 0; i < static_cast<int>(working_logits.size()); ++i) {
            const float v = working_logits[static_cast<size_t>(i)];
            if (!std::isfinite(v)) continue;
            if (!found || v > best_val) {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        return found ? (range_start + best_idx) : range_start;
    };

    if (params.temperature <= 0.0f) {
        return finite_argmax();
    }
    if (params.temperature != 1.0f && params.temperature > 0.0f) {
        for (float& v : working_logits) {
            v /= params.temperature;
        }
    }
    int finite_count = 0;
    for (float& v : working_logits) {
        if (std::isfinite(v)) {
            finite_count++;
        } else {
            v = -INFINITY;
        }
    }
    if (finite_count == 0) {
        return finite_argmax_raw();
    }

    float max_logit = -INFINITY;
    for (float v : working_logits) {
        if (std::isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!std::isfinite(max_logit)) {
        return finite_argmax();
    }

    int k = params.top_k;
    if (k <= 0 || k > active_vocab) {
        k = active_vocab;
    }

    std::vector<std::pair<float, int>> prob_idx;
    prob_idx.reserve(static_cast<size_t>(active_vocab));
    float sum_exp = 0.0f;
    for (int i = 0; i < active_vocab; ++i) {
        float prob = std::isfinite(working_logits[static_cast<size_t>(i)])
                         ? std::exp(working_logits[static_cast<size_t>(i)] - max_logit)
                         : 0.0f;
        if (!std::isfinite(prob)) prob = 0.0f;
        sum_exp += prob;
        prob_idx.push_back({prob, range_start + i});
    }
    if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) {
        return finite_argmax();
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (auto& p : prob_idx) {
        p.first *= inv_sum_exp;
    }

    auto prob_desc = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first > b.first;
    };
    if (k < active_vocab) {
        std::nth_element(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(), prob_desc);
        prob_idx.resize(static_cast<size_t>(k));
    }
    std::sort(prob_idx.begin(), prob_idx.end(), prob_desc);

    if (params.min_p > 0.0f && !prob_idx.empty()) {
        const float threshold = params.min_p * prob_idx.front().first;
        auto it = std::remove_if(prob_idx.begin(), prob_idx.end(),
                                 [threshold](const auto& p) { return p.first < threshold; });
        prob_idx.erase(it, prob_idx.end());
    }
    if (params.top_p < 1.0f && !prob_idx.empty()) {
        float cumulative = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < prob_idx.size(); ++i) {
            cumulative += prob_idx[i].first;
            cutoff = i + 1;
            if (cumulative >= params.top_p) break;
        }
        prob_idx.resize(cutoff);
    }
    if (prob_idx.empty()) {
        return finite_argmax();
    }

    float total = 0.0f;
    for (const auto& p : prob_idx) total += p.first;
    if (total <= 0.0f || !std::isfinite(total)) {
        return prob_idx.front().second;
    }
    const float inv_total = 1.0f / total;
    for (auto& p : prob_idx) {
        p.first *= inv_total;
    }

    std::mt19937 rng;
    const uint64_t seed = params.seed != 0 ? params.seed : 123456789ULL;
    std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    rng.seed(seq);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float random_val = dist(rng);
    float cumulative = 0.0f;
    for (const auto& p : prob_idx) {
        cumulative += p.first;
        if (random_val <= cumulative) {
            return p.second;
        }
    }
    return prob_idx.front().second;
}

}  // namespace

namespace densecore {

TEST(SamplingRangeTest, GreedyUsesActionTokenSubrange) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    // Global argmax would be token 6, but action range [2, 5) should choose token 3.
    const std::vector<float> logits_data = {0.1f, 0.2f, 0.7f, 1.1f, 0.9f, 0.3f, 4.0f, 0.5f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.0f;
    params.action_token_start = 2;
    params.action_token_count = 3;  // valid token ids: 2,3,4

    const int token = SampleToken(logits, 0, params);
    EXPECT_EQ(token, 3);
}

TEST(SamplingRangeTest, InvalidRangeFallsBackToFullVocab) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {0.1f, 0.2f, 0.7f, 1.1f, 0.9f, 0.3f, 4.0f, 0.5f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.0f;
    params.action_token_start = 999;
    params.action_token_count = 32;

    const int token = SampleToken(logits, 0, params);
    EXPECT_EQ(token, 6);
}

TEST(SamplingRangeTest, FusedTopKTopPMatchesLegacyReference) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {2.0f, 1.3f, -0.1f, 0.8f, 1.9f, -1.0f, 1.1f, 0.2f, 0.4f, -0.7f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.85f;
    params.top_k = 5;
    params.top_p = 0.72f;
    params.seed = 42;

    EXPECT_EQ(SampleToken(logits, 0, params), SampleTokenLegacyReference(logits, 0, params));
}

TEST(SamplingRangeTest, FusedMinPMatchesLegacyReference) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {3.2f, 1.0f, 0.7f, 0.1f, -0.4f, 2.3f, 1.9f, 0.8f, 0.2f, -1.4f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 1.1f;
    params.top_k = 6;
    params.top_p = 1.0f;
    params.min_p = 0.18f;
    params.seed = 7;

    EXPECT_EQ(SampleToken(logits, 0, params), SampleTokenLegacyReference(logits, 0, params));
}

TEST(SamplingRangeTest, PenaltiesAndDisallowedMatchLegacyReference) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {0.9f, 1.8f, 1.1f, -0.4f, 0.5f, 1.4f, 0.3f, 1.2f, -0.8f, 0.6f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    const std::vector<int> token_history = {2, 2, 5, 7, 7};
    const std::vector<int> disallowed = {4};

    SamplingParams params;
    params.temperature = 0.9f;
    params.top_k = 7;
    params.top_p = 0.9f;
    params.min_p = 0.05f;
    params.repetition_penalty = 1.1f;
    params.frequency_penalty = 0.25f;
    params.presence_penalty = 0.15f;
    params.token_history = &token_history;
    params.disallowed_token_ids = &disallowed;
    params.seed = 19;

    EXPECT_EQ(SampleToken(logits, 0, params), SampleTokenLegacyReference(logits, 0, params));
}

TEST(SamplingRangeTest, ActionRangeWithPenaltiesMatchesLegacyReference) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {4.5f, 0.2f, 1.9f, 1.7f, 0.4f, 2.5f, 2.1f, 0.8f, -0.2f, 3.0f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    const std::vector<int> token_history = {5, 5, 6};
    SamplingParams params;
    params.temperature = 0.95f;
    params.top_k = 3;
    params.top_p = 0.8f;
    params.repetition_penalty = 1.05f;
    params.frequency_penalty = 0.2f;
    params.presence_penalty = 0.1f;
    params.action_token_start = 2;
    params.action_token_count = 6;
    params.token_history = &token_history;
    params.seed = 123;

    EXPECT_EQ(SampleToken(logits, 0, params), SampleTokenLegacyReference(logits, 0, params));
}

TEST(SamplingRangeTest, TemperatureZeroUsesGreedyEvenWithTopKAndTopP) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    const std::vector<float> logits_data = {0.1f, 0.2f, 0.3f, 0.9f, 0.8f};
    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, logits_data);
    ASSERT_NE(logits, nullptr);

    SamplingParams params;
    params.temperature = 0.0f;
    params.top_k = 3;
    params.top_p = 0.5f;
    params.seed = 123;

    EXPECT_EQ(SampleToken(logits, 0, params), 3);
}

TEST(SamplingRangeTest, AllNanLogitsDoNotFallbackToDisallowedRangeStart) {
    ggml_init_params p = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    GgmlContextGuard guard;
    guard.ctx = ggml_init(p);
    ASSERT_NE(guard.ctx, nullptr);

    ggml_tensor* logits = MakeLogitsTensor(guard.ctx, {NAN, NAN, NAN, NAN});
    ASSERT_NE(logits, nullptr);

    std::vector<int> disallowed{0};
    SamplingParams params;
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.disallowed_token_ids = &disallowed;

    EXPECT_EQ(SampleToken(logits, 0, params), 1);
}

TEST(SamplingRangeTest, PrefillSamplingGuardUsesLastPromptColumnForUnchunkedPrefill) {
    int logits_idx = -1;
    std::string error;
    EXPECT_TRUE(ResolveSamplingLogitsColumnForRequest(
        /*token_offset=*/0, /*processed_count=*/6, /*output_columns=*/6, /*sampled_from_prefill=*/true,
        /*remaining_prompt_tokens=*/6, /*n_past_before=*/0, /*n_past_after=*/6, &logits_idx, &error));
    EXPECT_EQ(logits_idx, 5);
    EXPECT_TRUE(error.empty());
}

TEST(SamplingRangeTest, PrefillSamplingGuardAcceptsSingleColumnLastLogitPrefill) {
    int logits_idx = -1;
    std::string error;
    EXPECT_TRUE(ResolveSamplingLogitsColumnForRequest(
        /*token_offset=*/0, /*processed_count=*/6, /*output_columns=*/1, /*sampled_from_prefill=*/true,
        /*remaining_prompt_tokens=*/6, /*n_past_before=*/0, /*n_past_after=*/6, &logits_idx, &error));
    EXPECT_EQ(logits_idx, 0);
    EXPECT_TRUE(error.empty());
}

TEST(SamplingRangeTest, PrefillSamplingGuardUsesFinalChunkColumnForChunkedPrefill) {
    int logits_idx = -1;
    std::string error;
    EXPECT_TRUE(ResolveSamplingLogitsColumnForRequest(
        /*token_offset=*/4, /*processed_count=*/3, /*output_columns=*/9, /*sampled_from_prefill=*/true,
        /*remaining_prompt_tokens=*/3, /*n_past_before=*/8, /*n_past_after=*/11, &logits_idx, &error));
    EXPECT_EQ(logits_idx, 6);
    EXPECT_TRUE(error.empty());
}

TEST(SamplingRangeTest, PrefillSamplingGuardRejectsNonFinalPromptChunkSampling) {
    int logits_idx = -1;
    std::string error;
    EXPECT_FALSE(ResolveSamplingLogitsColumnForRequest(
        /*token_offset=*/0, /*processed_count=*/4, /*output_columns=*/4, /*sampled_from_prefill=*/true,
        /*remaining_prompt_tokens=*/7, /*n_past_before=*/0, /*n_past_after=*/4, &logits_idx, &error));
    EXPECT_NE(error.find("final prompt chunk"), std::string::npos);
}

TEST(SamplingRangeTest, MixedCompactLogitsSelectDecodeColumnAfterPrefill) {
    int column = -1;
    std::string error;
    ASSERT_TRUE(ResolveSamplingLogitsColumnForRequest(
        390, 1, 2, false, 0, 12, 13, &column, &error, 1, 2)) << error;
    EXPECT_EQ(column, 1);
}

TEST(SamplingRangeTest, MixedCompactLogitsSelectBothSequenceOrders) {
    for (int prefill_sequence : {0, 1}) {
        int column = -1;
        std::string error;
        ASSERT_TRUE(ResolveSamplingLogitsColumnForRequest(
            prefill_sequence, 390, 2, true, 390, 0, 390,
            &column, &error, prefill_sequence, 2)) << error;
        EXPECT_EQ(column, prefill_sequence);
        const int decode_sequence = 1 - prefill_sequence;
        ASSERT_TRUE(ResolveSamplingLogitsColumnForRequest(
            prefill_sequence == 0 ? 390 : 0, 1, 2, false, 0, 12, 13,
            &column, &error, decode_sequence, 2)) << error;
        EXPECT_EQ(column, decode_sequence);
    }
}

TEST(SamplingRangeTest, CompactLogitsRetainPromptProgressAndSequenceBoundsGuards) {
    int column = -1;
    std::string error;
    EXPECT_FALSE(ResolveSamplingLogitsColumnForRequest(
        0, 4, 2, true, 7, 0, 4, &column, &error, 0, 2));
    EXPECT_NE(error.find("final prompt chunk"), std::string::npos);
    EXPECT_FALSE(ResolveSamplingLogitsColumnForRequest(
        0, 1, 2, false, 0, 12, 13, &column, &error, 2, 2));
}

TEST(SamplingRangeTest, BatchMetadataPreservesFullLogitsAndPureDecodeColumns) {
    int column = -1;
    std::string error;
    ASSERT_TRUE(ResolveSamplingLogitsColumnForRequest(
        390, 1, 391, false, 0, 12, 13, &column, &error, 1, 2)) << error;
    EXPECT_EQ(column, 390);
    for (int sequence = 0; sequence < 4; ++sequence) {
        ASSERT_TRUE(ResolveSamplingLogitsColumnForRequest(
            sequence, 1, 4, false, 0, 12, 13, &column, &error, sequence, 4)) << error;
        EXPECT_EQ(column, sequence);
    }
}

}  // namespace densecore
