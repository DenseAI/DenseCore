#include "densecore/runtime/inference.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
// Grammar-Based Sampling Implementation
// ============================================================================

namespace {

std::mutex& SamplingDebugTraceMutex() {
    static std::mutex mu;
    return mu;
}

std::vector<SamplingDebugTraceEntry>& SamplingDebugTraceStorage() {
    static std::vector<SamplingDebugTraceEntry> entries;
    return entries;
}

bool ShouldCaptureSamplingDebugTrace(const SamplingParams& params) {
    if (params.request_id < 0 || params.output_token_index < 0) {
        return false;
    }
    const char* env = std::getenv("DENSECORE_DEBUG_SAMPLER_TRACE");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

float ApplyFinalLogitSoftcap(float v, float softcap) {
    if (softcap <= 0.0f || !std::isfinite(v)) {
        return v;
    }
    return std::tanh(v / softcap) * softcap;
}

}  // namespace

void ResetSamplingDebugTrace() {
    std::lock_guard<std::mutex> lock(SamplingDebugTraceMutex());
    SamplingDebugTraceStorage().clear();
}

std::vector<SamplingDebugTraceEntry> GetSamplingDebugTraceSnapshot() {
    std::lock_guard<std::mutex> lock(SamplingDebugTraceMutex());
    return SamplingDebugTraceStorage();
}

void InitGrammarConstraint(GrammarConstraint* grammar, const std::vector<std::string>& vocab) {
    if (!grammar) return;

    // Find token IDs for JSON special characters
    for (size_t i = 0; i < vocab.size(); i++) {
        const std::string& token = vocab[i];
        if (token == "{" || token == " {")
            grammar->token_lbrace = i;
        else if (token == "}" || token == " }")
            grammar->token_rbrace = i;
        else if (token == "[" || token == " [")
            grammar->token_lbracket = i;
        else if (token == "]" || token == " ]")
            grammar->token_rbracket = i;
        else if (token == "\"" || token == " \"")
            grammar->token_quote = i;
        else if (token == ":" || token == " :")
            grammar->token_colon = i;
        else if (token == "," || token == " ,")
            grammar->token_comma = i;
    }
}

void GrammarConstraint::UpdateState(const std::string& token_text) {
    if (!enabled || !is_json_mode) return;

    accumulated += token_text;

    // Trim leading whitespace for state transitions
    std::string trimmed = token_text;
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
        trimmed = trimmed.substr(start);
    }

    if (trimmed.empty()) return;

    char first_char = trimmed[0];

    switch (state) {
    case JSONState::EXPECT_OBJECT_START:
        if (first_char == '{') {
            state = JSONState::EXPECT_KEY_OR_END;
            brace_depth = 1;
        }
        break;

    case JSONState::EXPECT_KEY_OR_END:
        if (first_char == '"') {
            state = JSONState::IN_KEY;
        } else if (first_char == '}') {
            brace_depth--;
            if (brace_depth == 0) {
                state = JSONState::COMPLETED;
            }
        }
        break;

    case JSONState::IN_KEY:
        if (first_char == '"' && !in_escape) {
            state = JSONState::EXPECT_COLON;
        } else if (first_char == '\\') {
            in_escape = !in_escape;
        } else {
            in_escape = false;
        }
        break;

    case JSONState::EXPECT_COLON:
        if (first_char == ':') {
            state = JSONState::EXPECT_VALUE;
        }
        break;

    case JSONState::EXPECT_VALUE:
        if (first_char == '"') {
            state = JSONState::IN_STRING_VALUE;
        } else if (first_char == '{') {
            brace_depth++;
            state = JSONState::EXPECT_KEY_OR_END;
        } else if (first_char == '[') {
            bracket_depth++;
            state = JSONState::IN_ARRAY;
        } else if (isdigit(first_char) || first_char == '-') {
            state = JSONState::IN_NUMBER;
        } else if (trimmed.find("true") == 0 || trimmed.find("false") == 0 || trimmed.find("null") == 0) {
            state = JSONState::EXPECT_COMMA_OR_END;
        }
        break;

    case JSONState::IN_STRING_VALUE:
        if (first_char == '"' && !in_escape) {
            state = JSONState::EXPECT_COMMA_OR_END;
        } else if (first_char == '\\') {
            in_escape = !in_escape;
        } else {
            in_escape = false;
        }
        break;

    case JSONState::IN_NUMBER:
        if (first_char == ',' || first_char == '}' || first_char == ']') {
            state = JSONState::EXPECT_COMMA_OR_END;
            if (first_char == ',') {
                state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
            } else if (first_char == '}') {
                brace_depth--;
                if (brace_depth == 0) state = JSONState::COMPLETED;
            } else if (first_char == ']') {
                bracket_depth--;
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        }
        break;

    case JSONState::EXPECT_COMMA_OR_END:
        if (first_char == ',') {
            state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
        } else if (first_char == '}') {
            brace_depth--;
            if (brace_depth == 0) {
                state = JSONState::COMPLETED;
            }
        } else if (first_char == ']') {
            bracket_depth--;
            if (bracket_depth == 0) {
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        }
        break;

    case JSONState::IN_ARRAY:
        if (first_char == ']') {
            bracket_depth--;
            if (bracket_depth == 0) {
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        } else if (first_char == ',') {
            // Stay in array
        } else if (first_char == '"') {
            state = JSONState::IN_STRING_VALUE;
        } else if (first_char == '{') {
            brace_depth++;
            state = JSONState::EXPECT_KEY_OR_END;
        }
        break;

    case JSONState::COMPLETED: break;
    }
}

bool IsDigitToken(const std::string& token) {
    if (token.empty()) return false;
    for (char c : token) {
        if (!isdigit(c) && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+' && c != ' ') return false;
    }
    return true;
}

bool IsWhitespaceToken(const std::string& token) {
    if (token.empty()) return false;
    for (char c : token) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return false;
    }
    return true;
}

bool ContainsChar(const std::string& token, char ch) {
    return token.find(ch) != std::string::npos;
}

void ApplyGrammarMask(float* logits, int n_vocab, const GrammarConstraint* grammar,
                      const std::vector<std::string>& vocab) {
    if (!grammar || !grammar->enabled || !grammar->is_json_mode) {
        return;
    }

    const float NEG_INF = -INFINITY;
    std::vector<bool> allowed(n_vocab, false);

    // Always allow whitespace
    for (int i = 0; i < n_vocab; i++) {
        if (IsWhitespaceToken(vocab[i])) {
            allowed[i] = true;
        }
    }

    switch (grammar->state) {
    case JSONState::EXPECT_OBJECT_START:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], '{')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_KEY_OR_END:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], '"') || ContainsChar(vocab[i], '}')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_KEY:
    case JSONState::IN_STRING_VALUE:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            bool has_control = false;
            for (char c : token) {
                if (c < 32 && c != '\t' && c != '\n') {
                    has_control = true;
                    break;
                }
            }
            if (!has_control) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_COLON:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], ':')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_VALUE:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                IsDigitToken(token) || token.find("true") != std::string::npos ||
                token.find("false") != std::string::npos || token.find("null") != std::string::npos) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_NUMBER:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (IsDigitToken(token) || ContainsChar(token, ',') || ContainsChar(token, '}') ||
                ContainsChar(token, ']')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_COMMA_OR_END:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], ',') || ContainsChar(vocab[i], '}') || ContainsChar(vocab[i], ']')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_ARRAY:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                ContainsChar(token, ']') || ContainsChar(token, ',') || IsDigitToken(token)) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::COMPLETED: break;
    }

    for (int i = 0; i < n_vocab; i++) {
        if (!allowed[i]) {
            logits[i] = NEG_INF;
        }
    }
}

// ============================================================================
// Token Sampling
// ============================================================================

int SampleToken(struct ggml_tensor* logits, int idx, const SamplingParams& params) {
    if (!logits || !logits->data) return 0;

    float* logits_data = (float*)logits->data;
    int n_vocab = logits->ne[0];
    if (n_vocab <= 0) return 0;

    const int n_cols = std::max<int>(1, (int)logits->ne[1]);
    if (idx < 0) idx = 0;
    if (idx >= n_cols) idx = n_cols - 1;
    const ptrdiff_t row_stride = static_cast<ptrdiff_t>(logits->nb[1] / sizeof(float));
    if (row_stride < n_vocab) {
        return 0;
    }
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
        if (!params.disallowed_token_ids || params.disallowed_token_ids->empty()) {
            return false;
        }
        return std::binary_search(params.disallowed_token_ids->begin(), params.disallowed_token_ids->end(), token_id);
    };
    auto is_allowed = [&](int token_id) -> bool {
        if (!params.allowed_token_ids || params.allowed_token_ids->empty()) {
            return true;
        }
        return std::binary_search(params.allowed_token_ids->begin(), params.allowed_token_ids->end(), token_id);
    };
    auto resolve_debug_top_n = [&]() -> int {
        const char* env = std::getenv("DENSECORE_DEBUG_SAMPLE_TOP");
        if (!env || env[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0') || parsed <= 0) {
            return 5;
        }
        return static_cast<int>(std::min<long>(parsed, 20));
    };
    const int debug_top_n = resolve_debug_top_n();

    auto first_allowed_token = [&]() -> int {
        if (!params.allowed_token_ids || params.allowed_token_ids->empty()) {
            return range_start;
        }
        for (int token_id : *params.allowed_token_ids) {
            if (token_id >= range_start && token_id < range_end) {
                return token_id;
            }
        }
        return range_start;
    };
    auto finite_argmax_raw = [last_logits, range_start, range_end, &is_disallowed, &is_allowed, &first_allowed_token,
                              &params]() -> int {
        int best_idx = range_start;
        float best_val = -INFINITY;
        bool found = false;
        const bool unrestricted_allow = !params.allowed_token_ids || params.allowed_token_ids->empty();
        const auto* disallowed = params.disallowed_token_ids;
        const bool unrestricted = unrestricted_allow && (!disallowed || disallowed->empty());
        if (unrestricted) {
            for (int i = range_start; i < range_end; ++i) {
                const float v = last_logits[i];
                if (!std::isfinite(v)) continue;
                if (!found || v > best_val) {
                    best_val = v;
                    best_idx = i;
                    found = true;
                }
            }
            return found ? best_idx : first_allowed_token();
        }
        size_t disallowed_pos = 0;
        if (unrestricted_allow && disallowed && !disallowed->empty()) {
            disallowed_pos = static_cast<size_t>(std::lower_bound(disallowed->begin(), disallowed->end(), range_start) -
                                                 disallowed->begin());
        }
        for (int i = range_start; i < range_end; ++i) {
            const float v = last_logits[i];
            if (!std::isfinite(v)) continue;
            bool blocked = false;
            if (unrestricted_allow && disallowed && !disallowed->empty()) {
                while (disallowed_pos < disallowed->size() && (*disallowed)[disallowed_pos] < i) {
                    ++disallowed_pos;
                }
                blocked = disallowed_pos < disallowed->size() && (*disallowed)[disallowed_pos] == i;
            } else {
                blocked = is_disallowed(i) || !is_allowed(i);
            }
            if (blocked) continue;
            if (!found || v > best_val) {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        return found ? best_idx : first_allowed_token();
    };
    const bool debug_sample = (std::getenv("DENSECORE_DEBUG_SAMPLE") != nullptr);
    auto debug_log_sample = [&](int token_id) {
        if (!debug_sample) return;
        static int debug_sample_count = 0;
        if (debug_sample_count >= 8) return;
        const float logit = (token_id >= range_start && token_id < range_end) ? last_logits[token_id] : NAN;
        fprintf(stderr, "[SAMPLE_DBG #%d] idx=%d token=%d logit=%.6f range=[%d,%d)\n", debug_sample_count, idx,
                token_id, logit, range_start, range_end);
        if (params.vocab && debug_sample_count < 2) {
            std::vector<std::pair<float, int>> top;
            top.reserve(8);
            for (int i = range_start; i < range_end; ++i) {
                const float v = last_logits[i];
                if (!std::isfinite(v)) continue;
                if (top.size() < 8) {
                    top.emplace_back(v, i);
                    std::push_heap(top.begin(), top.end(),
                                   [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                    continue;
                }
                if (v > top.front().first) {
                    std::pop_heap(top.begin(), top.end(),
                                  [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                    top.back() = {v, i};
                    std::push_heap(top.begin(), top.end(),
                                   [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                }
            }
            std::sort(top.begin(), top.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
            for (const auto& [score, top_id] : top) {
                std::string tok =
                    (top_id >= 0 && top_id < static_cast<int>(params.vocab->size())) ? (*params.vocab)[top_id] : "";
                for (char& ch : tok) {
                    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                }
                fprintf(stderr, "  [TOP] token=%d logit=%.6f text='%s'\n", top_id, score, tok.c_str());
            }
        }
        debug_sample_count++;
    };
    auto capture_raw_top_candidates = [&]() {
        std::vector<SamplingDebugCandidate> top;
        top.reserve(8);
        auto worse_first = [](const SamplingDebugCandidate& a, const SamplingDebugCandidate& b) {
            if (a.post_penalty_logit == b.post_penalty_logit) return a.token_id < b.token_id;
            return a.post_penalty_logit > b.post_penalty_logit;
        };
        for (int token_id = range_start; token_id < range_end; ++token_id) {
            if (is_disallowed(token_id) || !is_allowed(token_id)) {
                continue;
            }
            const float v = last_logits[token_id];
            if (!std::isfinite(v)) {
                continue;
            }
            SamplingDebugCandidate cand;
            cand.token_id = token_id;
            cand.pre_penalty_logit = v;
            cand.post_penalty_logit = v;
            if (top.size() < 8) {
                top.push_back(cand);
                std::push_heap(top.begin(), top.end(), worse_first);
            } else if (cand.post_penalty_logit > top.front().post_penalty_logit ||
                       (cand.post_penalty_logit == top.front().post_penalty_logit &&
                        cand.token_id < top.front().token_id)) {
                std::pop_heap(top.begin(), top.end(), worse_first);
                top.back() = cand;
                std::push_heap(top.begin(), top.end(), worse_first);
            }
        }
        std::sort(top.begin(), top.end(), [](const SamplingDebugCandidate& a, const SamplingDebugCandidate& b) {
            if (a.post_penalty_logit == b.post_penalty_logit) return a.token_id < b.token_id;
            return a.post_penalty_logit > b.post_penalty_logit;
        });
        return top;
    };
    auto maybe_record_raw_sampling_trace = [&](int sampled_token) {
        if (!ShouldCaptureSamplingDebugTrace(params)) {
            return;
        }
        SamplingDebugTraceEntry entry;
        entry.request_id = params.request_id;
        entry.output_token_index = params.output_token_index;
        entry.sampled_token_id = sampled_token;
        entry.temperature = params.temperature;
        entry.top_p = params.top_p;
        entry.top_k = params.top_k;
        entry.repetition_penalty = params.repetition_penalty;
        entry.top_pre_penalty = capture_raw_top_candidates();
        entry.top_post_penalty = entry.top_pre_penalty;
        std::lock_guard<std::mutex> lock(SamplingDebugTraceMutex());
        auto& storage = SamplingDebugTraceStorage();
        storage.push_back(std::move(entry));
        if (storage.size() > 64) {
            storage.erase(storage.begin(), storage.begin() + static_cast<std::ptrdiff_t>(storage.size() - 64));
        }
    };

    const bool has_history = params.token_history && !params.token_history->empty();
    const bool has_penalty = has_history && (params.repetition_penalty != 1.0f || params.frequency_penalty != 0.0f ||
                                             params.presence_penalty != 0.0f);
    if (params.temperature <= 0.0f && params.grammar == nullptr && !has_penalty) {
        const int token = finite_argmax_raw();
        maybe_record_raw_sampling_trace(token);
        debug_log_sample(token);
        return token;
    }

    const bool trace_enabled = ShouldCaptureSamplingDebugTrace(params);
    const bool greedy_repetition_only = params.temperature <= 0.0f && params.grammar == nullptr && has_history &&
                                        params.repetition_penalty != 1.0f && params.frequency_penalty == 0.0f &&
                                        params.presence_penalty == 0.0f && params.final_logit_softcap <= 0.0f &&
                                        debug_top_n <= 0 && !trace_enabled;
    if (greedy_repetition_only) {
        thread_local std::vector<int> repeated_tokens;
        repeated_tokens.clear();
        repeated_tokens.reserve(params.token_history->size());
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                repeated_tokens.push_back(token);
            }
        }
        std::sort(repeated_tokens.begin(), repeated_tokens.end());

        int best_idx = range_start;
        float best_val = -INFINITY;
        bool found = false;
        size_t repeat_pos = 0;
        const bool unrestricted_allow = !params.allowed_token_ids || params.allowed_token_ids->empty();
        const auto* disallowed = params.disallowed_token_ids;
        const bool unrestricted = unrestricted_allow && (!disallowed || disallowed->empty());
        size_t disallowed_pos = 0;
        if (unrestricted_allow && disallowed && !disallowed->empty()) {
            disallowed_pos = static_cast<size_t>(std::lower_bound(disallowed->begin(), disallowed->end(), range_start) -
                                                 disallowed->begin());
        }
        for (int token_id = range_start; token_id < range_end; ++token_id) {
            bool blocked = false;
            if (!unrestricted) {
                if (unrestricted_allow && disallowed && !disallowed->empty()) {
                    while (disallowed_pos < disallowed->size() && (*disallowed)[disallowed_pos] < token_id) {
                        ++disallowed_pos;
                    }
                    blocked = disallowed_pos < disallowed->size() && (*disallowed)[disallowed_pos] == token_id;
                } else {
                    blocked = is_disallowed(token_id) || !is_allowed(token_id);
                }
            }
            if (blocked) {
                continue;
            }

            float v = last_logits[token_id];
            if (!std::isfinite(v)) {
                continue;
            }
            while (repeat_pos < repeated_tokens.size() && repeated_tokens[repeat_pos] < token_id) {
                ++repeat_pos;
            }
            if (repeat_pos < repeated_tokens.size() && repeated_tokens[repeat_pos] == token_id) {
                size_t repeat_end = repeat_pos + 1;
                while (repeat_end < repeated_tokens.size() && repeated_tokens[repeat_end] == token_id) {
                    ++repeat_end;
                }
                const size_t count = repeat_end - repeat_pos;
                for (size_t rep = 0; rep < count; ++rep) {
                    if (v < 0.0f) {
                        v *= params.repetition_penalty;
                    } else {
                        v /= params.repetition_penalty;
                    }
                }
                repeat_pos = repeat_end;
            }
            if (!std::isfinite(v)) {
                continue;
            }
            if (!found || v > best_val || (v == best_val && token_id < best_idx)) {
                best_val = v;
                best_idx = token_id;
                found = true;
            }
        }
        const int token = found ? best_idx : first_allowed_token();
        debug_log_sample(token);
        return token;
    }

    thread_local std::vector<float> working_logits;
    const bool requires_working_logits = params.grammar && params.vocab;
    if (requires_working_logits) {
        working_logits.assign(last_logits + range_start, last_logits + range_end);
        if (params.disallowed_token_ids && !params.disallowed_token_ids->empty()) {
            for (int token_id : *params.disallowed_token_ids) {
                if (token_id >= range_start && token_id < range_end) {
                    working_logits[token_id - range_start] = -INFINITY;
                }
            }
        }
        ApplyGrammarMask(working_logits.data(), active_vocab, params.grammar, *params.vocab);
    }

    thread_local std::vector<int> token_counts;
    if (has_history) {
        token_counts.assign(static_cast<size_t>(active_vocab), 0);
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                token_counts[static_cast<size_t>(token - range_start)]++;
            }
        }
    } else {
        token_counts.clear();
    }

    auto base_logit_at = [&](int local_token) -> float {
        if (requires_working_logits) {
            return ApplyFinalLogitSoftcap(working_logits[static_cast<size_t>(local_token)], params.final_logit_softcap);
        }
        return ApplyFinalLogitSoftcap(last_logits[range_start + local_token], params.final_logit_softcap);
    };

    auto adjusted_logit_at = [&](int local_token) -> float {
        const int token_id = range_start + local_token;
        if (!requires_working_logits && (is_disallowed(token_id) || !is_allowed(token_id))) {
            return -INFINITY;
        }

        float v = base_logit_at(local_token);
        if (!std::isfinite(v)) {
            return -INFINITY;
        }

        if (has_penalty) {
            const int count = token_counts[static_cast<size_t>(local_token)];
            if (count > 0) {
                if (params.repetition_penalty != 1.0f) {
                    for (int rep = 0; rep < count; ++rep) {
                        if (v < 0.0f) {
                            v *= params.repetition_penalty;
                        } else {
                            v /= params.repetition_penalty;
                        }
                    }
                }
                if (params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) {
                    v -= params.frequency_penalty * count + params.presence_penalty * 1.0f;
                }
            }
        }

        if (params.temperature > 0.0f && params.temperature != 1.0f) {
            v /= params.temperature;
        }

        return std::isfinite(v) ? v : -INFINITY;
    };
    auto capture_top_candidates = [&](bool post_penalty) {
        std::vector<SamplingDebugCandidate> top;
        top.reserve(8);
        auto worse_first = [](const SamplingDebugCandidate& a, const SamplingDebugCandidate& b) {
            if (a.post_penalty_logit == b.post_penalty_logit) return a.token_id < b.token_id;
            return a.post_penalty_logit > b.post_penalty_logit;
        };
        for (int i = 0; i < active_vocab; ++i) {
            const int token_id = range_start + i;
            float pre = base_logit_at(i);
            if (!std::isfinite(pre) ||
                (!requires_working_logits && (is_disallowed(token_id) || !is_allowed(token_id)))) {
                pre = -INFINITY;
            }
            const float post = adjusted_logit_at(i);
            if (!std::isfinite(post)) {
                continue;
            }
            SamplingDebugCandidate cand;
            cand.token_id = token_id;
            cand.pre_penalty_logit = pre;
            cand.post_penalty_logit = post;
            if (!post_penalty) {
                cand.post_penalty_logit = pre;
            }
            if (top.size() < 8) {
                top.push_back(cand);
                std::push_heap(top.begin(), top.end(), worse_first);
            } else if (cand.post_penalty_logit > top.front().post_penalty_logit ||
                       (cand.post_penalty_logit == top.front().post_penalty_logit &&
                        cand.token_id < top.front().token_id)) {
                std::pop_heap(top.begin(), top.end(), worse_first);
                top.back() = cand;
                std::push_heap(top.begin(), top.end(), worse_first);
            }
        }
        std::sort(top.begin(), top.end(), [](const SamplingDebugCandidate& a, const SamplingDebugCandidate& b) {
            if (a.post_penalty_logit == b.post_penalty_logit) return a.token_id < b.token_id;
            return a.post_penalty_logit > b.post_penalty_logit;
        });
        return top;
    };
    auto maybe_record_sampling_trace = [&](int sampled_token) {
        if (!ShouldCaptureSamplingDebugTrace(params)) {
            return;
        }
        SamplingDebugTraceEntry entry;
        entry.request_id = params.request_id;
        entry.output_token_index = params.output_token_index;
        entry.sampled_token_id = sampled_token;
        entry.temperature = params.temperature;
        entry.top_p = params.top_p;
        entry.top_k = params.top_k;
        entry.repetition_penalty = params.repetition_penalty;
        entry.top_pre_penalty = capture_top_candidates(false);
        entry.top_post_penalty = capture_top_candidates(true);
        std::lock_guard<std::mutex> lock(SamplingDebugTraceMutex());
        auto& storage = SamplingDebugTraceStorage();
        storage.push_back(std::move(entry));
        if (storage.size() > 64) {
            storage.erase(storage.begin(), storage.begin() + static_cast<std::ptrdiff_t>(storage.size() - 64));
        }
    };
    auto debug_dump_top_candidates = [&](const char* stage, int sampled_token) {
        if (debug_top_n <= 0 || !params.vocab) {
            return;
        }
        struct DebugCandidate {
            float logit = -INFINITY;
            int token_id = -1;
        };
        std::vector<DebugCandidate> top;
        top.reserve(static_cast<size_t>(debug_top_n));
        auto worse_first = [](const DebugCandidate& a, const DebugCandidate& b) {
            if (a.logit == b.logit) return a.token_id < b.token_id;
            return a.logit > b.logit;
        };
        for (int i = 0; i < active_vocab; ++i) {
            const float v = adjusted_logit_at(i);
            if (!std::isfinite(v)) continue;
            const int token_id = range_start + i;
            DebugCandidate cand{v, token_id};
            if (static_cast<int>(top.size()) < debug_top_n) {
                top.push_back(cand);
                std::push_heap(top.begin(), top.end(), worse_first);
            } else if (v > top.front().logit || (v == top.front().logit && token_id < top.front().token_id)) {
                std::pop_heap(top.begin(), top.end(), worse_first);
                top.back() = cand;
                std::push_heap(top.begin(), top.end(), worse_first);
            }
        }
        std::sort(top.begin(), top.end(), [](const DebugCandidate& a, const DebugCandidate& b) {
            if (a.logit == b.logit) return a.token_id < b.token_id;
            return a.logit > b.logit;
        });
        fprintf(stderr,
                "[SAMPLE_TOP] stage=%s idx=%d sampled=%d temp=%.4f top_p=%.4f top_k=%d rep=%.4f active_vocab=%d\n",
                stage ? stage : "unknown", idx, sampled_token, params.temperature, params.top_p, params.top_k,
                params.repetition_penalty, active_vocab);
        for (const DebugCandidate& cand : top) {
            std::string tok = (cand.token_id >= 0 && cand.token_id < static_cast<int>(params.vocab->size()))
                                  ? (*params.vocab)[cand.token_id]
                                  : "";
            for (char& ch : tok) {
                if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            }
            fprintf(stderr, "  [TOP] token=%d logit=%.6f raw='%s'%s\n", cand.token_id, cand.logit, tok.c_str(),
                    cand.token_id == sampled_token ? " <sampled>" : "");
        }
        const char* watch_env = std::getenv("DENSECORE_DEBUG_SAMPLE_WATCH_IDS");
        if (watch_env && watch_env[0] != '\0') {
            std::string ids_spec(watch_env);
            size_t start = 0;
            while (start < ids_spec.size()) {
                size_t end = ids_spec.find(',', start);
                if (end == std::string::npos) {
                    end = ids_spec.size();
                }
                const std::string token_id_text = ids_spec.substr(start, end - start);
                char* parse_end = nullptr;
                const long parsed = std::strtol(token_id_text.c_str(), &parse_end, 10);
                if (parse_end != token_id_text.c_str() && (!parse_end || *parse_end == '\0') && parsed >= range_start &&
                    parsed < range_end) {
                    const int token_id = static_cast<int>(parsed);
                    const float logit = adjusted_logit_at(token_id - range_start);
                    std::string tok = (token_id >= 0 && token_id < static_cast<int>(params.vocab->size()))
                                          ? (*params.vocab)[token_id]
                                          : "";
                    for (char& ch : tok) {
                        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                    }
                    fprintf(stderr, "  [WATCH] token=%d logit=%.6f raw='%s'%s\n", token_id, logit, tok.c_str(),
                            token_id == sampled_token ? " <sampled>" : "");
                }
                start = end + 1;
            }
        }
    };

    auto finite_argmax_adjusted = [&]() -> int {
        int best_idx = range_start;
        float best_val = -INFINITY;
        bool found = false;
        for (int i = 0; i < active_vocab; ++i) {
            const float v = adjusted_logit_at(i);
            if (!std::isfinite(v)) continue;
            const int token_id = range_start + i;
            if (!found || v > best_val || (v == best_val && token_id < best_idx)) {
                best_val = v;
                best_idx = token_id;
                found = true;
            }
        }
        return found ? best_idx : first_allowed_token();
    };

    if (params.temperature <= 0.0f || (params.top_k <= 1 && params.top_p >= 1.0f && params.min_p <= 0.0f)) {
        const int token = finite_argmax_adjusted();
        maybe_record_sampling_trace(token);
        debug_dump_top_candidates("argmax_adjusted", token);
        debug_log_sample(token);
        return token;
    }

    struct SamplingCandidate {
        float logit = -INFINITY;
        float mass = 0.0f;
        int token_id = -1;
    };

    auto candidate_desc = [](const SamplingCandidate& a, const SamplingCandidate& b) {
        if (a.logit == b.logit) return a.token_id < b.token_id;
        return a.logit > b.logit;
    };
    auto candidate_worse_first = [](const SamplingCandidate& a, const SamplingCandidate& b) {
        if (a.logit == b.logit) return a.token_id < b.token_id;
        return a.logit > b.logit;
    };

    int k = params.top_k;
    if (k <= 0 || k > active_vocab) {
        k = active_vocab;
    }

    thread_local std::vector<SamplingCandidate> candidates;
    candidates.clear();
    candidates.reserve(static_cast<size_t>(k));

    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    int best_token = range_start;
    bool found_finite = false;

    for (int i = 0; i < active_vocab; ++i) {
        const float v = adjusted_logit_at(i);
        if (!std::isfinite(v)) continue;
        const int token_id = range_start + i;
        if (!found_finite) {
            max_logit = v;
            best_token = token_id;
            found_finite = true;
            sum_exp = 1.0f;
        } else {
            if (v > max_logit) {
                const float rescaled_sum = sum_exp * std::exp(max_logit - v);
                sum_exp = (std::isfinite(rescaled_sum) ? rescaled_sum : 0.0f) + 1.0f;
                max_logit = v;
                best_token = token_id;
            } else {
                const float mass = std::exp(v - max_logit);
                if (std::isfinite(mass)) {
                    sum_exp += mass;
                }
                if (v == max_logit && token_id < best_token) {
                    best_token = token_id;
                }
            }
        }

        if (k < active_vocab) {
            SamplingCandidate cand;
            cand.logit = v;
            cand.token_id = token_id;
            if (static_cast<int>(candidates.size()) < k) {
                candidates.push_back(cand);
                std::push_heap(candidates.begin(), candidates.end(), candidate_worse_first);
            } else if (v > candidates.front().logit ||
                       (v == candidates.front().logit && token_id < candidates.front().token_id)) {
                std::pop_heap(candidates.begin(), candidates.end(), candidate_worse_first);
                candidates.back() = cand;
                std::push_heap(candidates.begin(), candidates.end(), candidate_worse_first);
            }
        }
    }

    if (!found_finite || !std::isfinite(max_logit)) {
        const int token = finite_argmax_raw();
        maybe_record_sampling_trace(token);
        debug_dump_top_candidates("argmax_raw_fallback", token);
        debug_log_sample(token);
        return token;
    }

    if (k < active_vocab && params.top_p >= 1.0f && params.min_p <= 0.0f) {
        std::sort(candidates.begin(), candidates.end(), candidate_desc);
        if (candidates.empty()) {
            maybe_record_sampling_trace(best_token);
            debug_dump_top_candidates("empty_candidates_fallback", best_token);
            debug_log_sample(best_token);
            return best_token;
        }

        const float candidate_max_logit = candidates.front().logit;
        float total_mass = 0.0f;
        for (auto& candidate : candidates) {
            float mass = std::exp(candidate.logit - candidate_max_logit);
            if (!std::isfinite(mass)) {
                mass = 0.0f;
            }
            candidate.mass = mass;
            total_mass += mass;
        }
        if (!(total_mass > 0.0f) || !std::isfinite(total_mass)) {
            maybe_record_sampling_trace(candidates.front().token_id);
            debug_dump_top_candidates("front_candidate_fallback", candidates.front().token_id);
            debug_log_sample(candidates.front().token_id);
            return candidates.front().token_id;
        }

        thread_local std::unique_ptr<std::mt19937> rng;
        thread_local uint64_t last_seed = 0;
        thread_local bool seeded_from_device = false;
        if (!rng) {
            rng = std::make_unique<std::mt19937>(std::random_device{}());
            seeded_from_device = true;
            last_seed = 0;
        }
        if (params.seed != 0) {
            if (last_seed != params.seed || seeded_from_device) {
                std::seed_seq seq{static_cast<uint32_t>(params.seed), static_cast<uint32_t>(params.seed >> 32)};
                rng->seed(seq);
                last_seed = params.seed;
                seeded_from_device = false;
            }
        } else if (!seeded_from_device) {
            rng->seed(std::random_device{}());
            last_seed = 0;
            seeded_from_device = true;
        }

        std::uniform_real_distribution<float> dist(0.0f, total_mass);
        const float random_val = dist(*rng);
        float cumulative_mass = 0.0f;
        for (const auto& candidate : candidates) {
            cumulative_mass += candidate.mass;
            if (random_val <= cumulative_mass) {
                maybe_record_sampling_trace(candidate.token_id);
                debug_dump_top_candidates("sampled", candidate.token_id);
                debug_log_sample(candidate.token_id);
                return candidate.token_id;
            }
        }

        maybe_record_sampling_trace(candidates.front().token_id);
        debug_dump_top_candidates("sorted_front_fallback", candidates.front().token_id);
        debug_log_sample(candidates.front().token_id);
        return candidates.front().token_id;
    }

    if (k < active_vocab) {
        for (auto& candidate : candidates) {
            float mass = std::exp(candidate.logit - max_logit);
            if (!std::isfinite(mass)) {
                mass = 0.0f;
            }
            candidate.mass = mass;
        }
    } else {
        candidates.reserve(static_cast<size_t>(active_vocab));
        sum_exp = 0.0f;
        for (int i = 0; i < active_vocab; ++i) {
            const float v = adjusted_logit_at(i);
            if (!std::isfinite(v)) continue;
            float mass = std::exp(v - max_logit);
            if (!std::isfinite(mass)) {
                mass = 0.0f;
            }
            sum_exp += mass;
            const int token_id = range_start + i;
            SamplingCandidate cand;
            cand.logit = v;
            cand.mass = mass;
            cand.token_id = token_id;
            candidates.push_back(cand);
        }
    }

    if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
        maybe_record_sampling_trace(best_token);
        debug_dump_top_candidates("best_token_fallback", best_token);
        debug_log_sample(best_token);
        return best_token;
    }

    std::sort(candidates.begin(), candidates.end(), candidate_desc);

    if (params.min_p > 0.0f && !candidates.empty()) {
        const float threshold = params.min_p * candidates.front().mass;
        auto it = std::remove_if(candidates.begin(), candidates.end(), [threshold](const SamplingCandidate& candidate) {
            return candidate.mass < threshold;
        });
        candidates.erase(it, candidates.end());
    }

    if (params.top_p < 1.0f && !candidates.empty()) {
        const float cutoff_mass = params.top_p * sum_exp;
        float cumulative_mass = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumulative_mass += candidates[i].mass;
            cutoff = i + 1;
            if (cumulative_mass >= cutoff_mass) {
                break;
            }
        }
        candidates.resize(cutoff);
    }

    if (candidates.empty()) {
        maybe_record_sampling_trace(best_token);
        debug_dump_top_candidates("empty_candidates_fallback", best_token);
        debug_log_sample(best_token);
        return best_token;
    }

    float total_mass = 0.0f;
    for (const auto& candidate : candidates) {
        total_mass += candidate.mass;
    }
    if (!(total_mass > 0.0f) || !std::isfinite(total_mass)) {
        maybe_record_sampling_trace(candidates.front().token_id);
        debug_dump_top_candidates("front_candidate_fallback", candidates.front().token_id);
        debug_log_sample(candidates.front().token_id);
        return candidates.front().token_id;
    }

    thread_local std::unique_ptr<std::mt19937> rng;
    thread_local uint64_t last_seed = 0;
    thread_local bool seeded_from_device = false;
    if (!rng) {
        rng = std::make_unique<std::mt19937>(std::random_device{}());
        seeded_from_device = true;
        last_seed = 0;
    }
    if (params.seed != 0) {
        if (last_seed != params.seed || seeded_from_device) {
            std::seed_seq seq{static_cast<uint32_t>(params.seed), static_cast<uint32_t>(params.seed >> 32)};
            rng->seed(seq);
            last_seed = params.seed;
            seeded_from_device = false;
        }
    } else if (!seeded_from_device) {
        rng->seed(std::random_device{}());
        last_seed = 0;
        seeded_from_device = true;
    }

    std::uniform_real_distribution<float> dist(0.0f, total_mass);
    const float random_val = dist(*rng);
    float cumulative_mass = 0.0f;
    for (const auto& candidate : candidates) {
        cumulative_mass += candidate.mass;
        if (random_val <= cumulative_mass) {
            maybe_record_sampling_trace(candidate.token_id);
            debug_dump_top_candidates("sampled", candidate.token_id);
            debug_log_sample(candidate.token_id);
            return candidate.token_id;
        }
    }

    maybe_record_sampling_trace(candidates.front().token_id);
    debug_dump_top_candidates("sorted_front_fallback", candidates.front().token_id);
    debug_log_sample(candidates.front().token_id);
    return candidates.front().token_id;
}
