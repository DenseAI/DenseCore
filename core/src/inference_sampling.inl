// ============================================================================
// Grammar-Based Sampling Implementation
// ============================================================================

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

    auto finite_argmax_raw = [last_logits, range_start, range_end, &is_disallowed]() -> int {
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

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check logits stats for first 5 calls
        static int dbg_cnt = 0;
        if (dbg_cnt < 5) {
            int nan_ct = 0;
            int zero_ct = 0;
            float mn = last_logits[range_start];
            float mx = last_logits[range_start];
            for (int i = range_start; i < range_end; i++) {
                if (std::isnan(last_logits[i]))
                    nan_ct++;
                else {
                    if (last_logits[i] == 0.0f) zero_ct++;
                    if (last_logits[i] < mn) mn = last_logits[i];
                    if (last_logits[i] > mx) mx = last_logits[i];
                }
            }
            fprintf(stderr, "[LOGITS #%d] idx=%d vocab=[%d,%d) min=%.4f max=%.4f nan=%d zero=%d\n", dbg_cnt, idx,
                    range_start, range_end, mn, mx, nan_ct, zero_ct);
            dbg_cnt++;
        }
    }

    const bool has_history = params.token_history && !params.token_history->empty();
    const bool has_penalty = has_history && (params.repetition_penalty != 1.0f || params.frequency_penalty != 0.0f ||
                                             params.presence_penalty != 0.0f);
    if (params.temperature <= 0.0f && params.grammar == nullptr && !has_penalty) {
        const int token = finite_argmax_raw();
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

    thread_local std::unordered_map<int, int> token_counts;
    token_counts.clear();
    if (has_history) {
        token_counts.reserve(params.token_history->size());
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                token_counts[token]++;
            }
        }
    }

    auto base_logit_at = [&](int local_token) -> float {
        if (requires_working_logits) {
            return working_logits[static_cast<size_t>(local_token)];
        }
        return last_logits[range_start + local_token];
    };

    auto adjusted_logit_at = [&](int local_token) -> float {
        const int token_id = range_start + local_token;
        if (!requires_working_logits && is_disallowed(token_id)) {
            return -INFINITY;
        }

        float v = base_logit_at(local_token);
        if (!std::isfinite(v)) {
            return -INFINITY;
        }

        if (has_penalty) {
            auto it = token_counts.find(token_id);
            if (it != token_counts.end()) {
                const int count = it->second;
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
        return found ? best_idx : finite_argmax_raw();
    };

    if (params.temperature <= 0.0f || (params.top_k <= 1 && params.top_p >= 1.0f && params.min_p <= 0.0f)) {
        const int token = finite_argmax_adjusted();
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
    thread_local std::unordered_map<int, int> candidate_lookup;
    candidates.clear();
    candidate_lookup.clear();
    candidates.reserve(static_cast<size_t>(k));
    candidate_lookup.reserve(static_cast<size_t>(k) * 2U + 1U);

    float max_logit = -INFINITY;
    int best_token = range_start;
    bool found_finite = false;

    for (int i = 0; i < active_vocab; ++i) {
        const float v = adjusted_logit_at(i);
        if (!std::isfinite(v)) continue;
        const int token_id = range_start + i;
        if (!found_finite || v > max_logit || (v == max_logit && token_id < best_token)) {
            max_logit = v;
            best_token = token_id;
            found_finite = true;
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
        debug_log_sample(token);
        return token;
    }

    if (k < active_vocab) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            candidate_lookup.emplace(candidates[i].token_id, static_cast<int>(i));
        }
    } else {
        candidates.reserve(static_cast<size_t>(active_vocab));
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < active_vocab; ++i) {
        const float v = adjusted_logit_at(i);
        if (!std::isfinite(v)) continue;
        float mass = std::exp(v - max_logit);
        if (!std::isfinite(mass)) {
            mass = 0.0f;
        }
        sum_exp += mass;
        const int token_id = range_start + i;
        if (k < active_vocab) {
            auto it = candidate_lookup.find(token_id);
            if (it != candidate_lookup.end()) {
                candidates[static_cast<size_t>(it->second)].mass = mass;
            }
        } else {
            SamplingCandidate cand;
            cand.logit = v;
            cand.mass = mass;
            cand.token_id = token_id;
            candidates.push_back(cand);
        }
    }

    if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
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
        debug_log_sample(best_token);
        return best_token;
    }

    float total_mass = 0.0f;
    for (const auto& candidate : candidates) {
        total_mass += candidate.mass;
    }
    if (!(total_mass > 0.0f) || !std::isfinite(total_mass)) {
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
            debug_log_sample(candidate.token_id);
            return candidate.token_id;
        }
    }

    debug_log_sample(candidates.front().token_id);
    return candidates.front().token_id;
}
