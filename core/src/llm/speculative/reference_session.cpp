#include "llm/speculative/reference_session.h"
#include "llm/speculative/head_memory.h"
#include "llm/sampling/sampling_rng.h"
#include "llm/sampling/grammar_completion.h"
#include "densecore/models/tokenizer.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace densecore::llm::speculative {
namespace {
class RestoreInferenceContext {
public:
    RestoreInferenceContext() : work_(GetCurrentWorkContext()), batch_(GetCurrentBatch()), phase_(GetCurrentExecutionPhase()) {}
    ~RestoreInferenceContext() {
        SetCurrentWorkContext(work_);
        if (batch_) SetCurrentBatch(batch_);
        else ClearCurrentBatch();
        SetCurrentExecutionPhase(phase_);
    }
private:
    InferenceWorkContext* work_;
    const BatchSpec* batch_;
    InferenceExecutionPhase phase_;
};
void CheckCancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("MTP request canceled");
}
}
NextNBootstrapBatch BuildNextNBootstrapBatch(const std::vector<int>& prompt,
                                             const std::vector<float>& normalized_hidden,
                                             size_t embedding, size_t begin, size_t count) {
    if (embedding == 0 || normalized_hidden.size() / embedding != prompt.size() ||
        normalized_hidden.size() % embedding || begin > prompt.size() || count > prompt.size() - begin) {
        throw std::invalid_argument("MTP prompt/hidden alignment mismatch");
    }
    NextNBootstrapBatch result;
    result.tokens.assign(prompt.begin() + begin, prompt.begin() + begin + count);
    result.hidden.assign(count * embedding, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const size_t position = begin + i;
        if (position > 0) {
            std::copy_n(normalized_hidden.data() + (position - 1) * embedding, embedding,
                        result.hidden.data() + i * embedding);
        }
    }
    return result;
}

ReferenceSession::ReferenceSession(TransformerModel& model, int max_context, int draft_n_max,
                                   uint64_t seed, int threads)
    : source_(model), head_(model), work_(CreateInferenceWorkContext(), DestroyInferenceWorkContext),
      max_context_(max_context), draft_n_max_(draft_n_max), threads_(std::max(1, threads)) {
    if (max_context < 1 || draft_n_max < 1 || draft_n_max > 3 || !work_) {
        throw std::invalid_argument("invalid MTP reference session configuration");
    }
    const size_t embedding = source_.hparams.n_embd;
    if (!embedding || static_cast<size_t>(max_context_) > std::numeric_limits<size_t>::max() / embedding)
        throw std::overflow_error("MTP prompt storage size overflow");
    // Reserve the charged maximum once: row-by-row insert must not grow a
    // geometric capacity or retain old+new context-sized buffers at its peak.
    prompt_tokens_.reserve(static_cast<size_t>(max_context_));
    prompt_hidden_.reserve(static_cast<size_t>(max_context_) * embedding);
    observed_target_tokens_.reserve(static_cast<size_t>(draft_n_max_) + 1);
    observed_target_inputs_.reserve((static_cast<size_t>(draft_n_max_) + 1) * embedding);
    // Keep streams separate from ordinary target sampling and other requests.
    if (seed == 0) seed = (static_cast<uint64_t>(std::random_device{}()) << 32) | std::random_device{}();
    std::seed_seq draft_seed{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), 0x4d545044u};
    std::seed_seq accept_seed{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), 0x41434345u};
    draft_rng_.seed(draft_seed); accept_rng_.seed(accept_seed);
    cache_.reset(InitPagedKVCache(&head_.Model(), 1, max_context, GGML_TYPE_F16, -1));
    if (!cache_) throw std::runtime_error("MTP head KV allocation failed");
}
void ReferenceSession::ObserveTarget(const int* tokens, const float* hidden, size_t stride, int count, bool prefill) {
    const size_t embd = source_.hparams.n_embd;
    if (!tokens || !hidden || count < 1 || stride < embd) throw std::invalid_argument("invalid target hidden batch");
    if (target_seen_ > max_context_ - count) throw std::runtime_error("MTP target context exhausted");
    if (prefill) {
        if (bootstrapped_) throw std::invalid_argument("MTP prefix admission cannot skip bootstrap state");
        prompt_tokens_.insert(prompt_tokens_.end(), tokens, tokens + count);
        for (int i = 0; i < count; ++i) {
            prompt_hidden_.insert(prompt_hidden_.end(), hidden + static_cast<size_t>(i) * stride,
                                  hidden + static_cast<size_t>(i) * stride + embd);
        }
    }
    if (!prefill) {
        if (last_hidden_.size() != embd || !observed_target_tokens_.empty())
            throw std::runtime_error("MTP target observation must follow the previous head replay");
        observed_target_tokens_.assign(tokens, tokens + count);
        observed_target_inputs_.resize(static_cast<size_t>(count) * embd);
        std::copy(last_hidden_.begin(), last_hidden_.end(), observed_target_inputs_.begin());
        for (int row = 1; row < count; ++row)
            std::copy_n(hidden + static_cast<size_t>(row-1)*stride, embd,
                        observed_target_inputs_.data() + static_cast<size_t>(row)*embd);
    }
    last_hidden_.assign(hidden + static_cast<size_t>(count - 1) * stride,
                        hidden + static_cast<size_t>(count - 1) * stride + embd);
    target_seen_ += count;
}
ReferenceSession::HeadOutput ReferenceSession::ForwardHead(const std::vector<int>& tokens,
                                                           const std::vector<float>& hidden,
                                                           const SamplingParams* params,
                                                           const InferenceDependencies* deps,
                                                           std::vector<float>* raw_logits) {
    const size_t embd = source_.hparams.n_embd;
    if (!embd || tokens.empty() || tokens.size()>kHeadBootstrapTokens ||
        hidden.size()/embd != tokens.size() || hidden.size()%embd ||
        head_past_<0 || head_past_>max_context_ ||
        tokens.size() > static_cast<size_t>(max_context_ - head_past_)) {
        throw std::invalid_argument("MTP head context/hidden shape mismatch");
    }
    const int needed = SequenceBlockTable::BlocksNeeded(head_past_ + static_cast<int>(tokens.size()));
    if (static_cast<int>(blocks_.size()) < needed) {
        auto allocated = cache_->block_manager->Allocate(needed - static_cast<int>(blocks_.size()));
        if (static_cast<int>(allocated.size()) != needed - static_cast<int>(blocks_.size())) {
            throw std::runtime_error("MTP head KV blocks exhausted");
        }
        blocks_.insert(blocks_.end(), allocated.begin(), allocated.end());
    }
    BatchSpec batch{};
    batch.tokens = tokens; batch.num_seqs = 1;
    batch.seq_id.assign(tokens.size(), 0);
    for (size_t i = 0; i < tokens.size(); ++i) batch.pos.push_back(head_past_ + static_cast<int>(i));
    batch.n_past = {head_past_}; batch.block_tables = {blocks_}; batch.deps = deps;
    RestoreInferenceContext restore;
    SetCurrentWorkContext(work_.get()); ResetInferenceWorkContext(work_.get());
    SetInferenceWorkContextModelVariant(work_.get(), source_.variant);
    SetCurrentExecutionPhase(tokens.size() > 1 ? InferenceExecutionPhase::Prefill : InferenceExecutionPhase::Decode);
    SetCurrentBatch(&batch);
    // Same checked bound used before target KV allocation/concurrency choice.
    const auto scratch_bytes=CalculateHeadGraphMemory(source_,tokens.size(),
        static_cast<uint64_t>(head_past_)+tokens.size()).Total();
    if (scratch_bytes>SIZE_MAX) throw std::overflow_error("MTP head arena exceeds address space");
    const size_t scratch=static_cast<size_t>(scratch_bytes);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({scratch, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("MTP head graph allocation failed");
    auto* input_hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, embd, tokens.size());
    std::memcpy(input_hidden->data, hidden.data(), hidden.size() * sizeof(float));
    auto* graph = ggml_new_graph_custom(ctx.get(), 32768, false);
    ggml_tensor* next_hidden = nullptr;
    auto* logits = head_.Build(ctx.get(), cache_.get(), batch, input_hidden, graph, &next_hidden);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, threads_) != GGML_STATUS_SUCCESS ||
        !logits || !logits->data || !next_hidden || !next_hidden->data) {
        throw std::runtime_error("MTP head graph execution failed");
    }
    HeadOutput result;
    const auto* h = reinterpret_cast<const float*>(static_cast<const char*>(next_hidden->data) +
                          (next_hidden->ne[1] - 1) * next_hidden->nb[1]);
    result.hidden.assign(h, h + embd);
    if (raw_logits) {
        const auto* row = reinterpret_cast<const float*>(static_cast<const char*>(logits->data) +
            (logits->ne[1]-1)*logits->nb[1]);
        raw_logits->assign(row, row + logits->ne[0]);
    }
    if (params) {
        result.distribution = BuildSamplingDistribution(logits, static_cast<int>(logits->ne[1] - 1), *params);
        result.token = SampleTokenWithRng(logits, static_cast<int>(logits->ne[1] - 1), *params, draft_rng_);
    }
    head_past_ += static_cast<int>(tokens.size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
        cache_->block_manager->SetFilledSlots(blocks_[i], std::clamp(head_past_ - static_cast<int>(i) * BLOCK_SIZE, 0, BLOCK_SIZE));
    }
    return result;
}
void ReferenceSession::Bootstrap(const InferenceDependencies* deps, const std::atomic<bool>& cancelled) {
    if (bootstrapped_) return;
    if (prompt_tokens_.empty() || static_cast<int>(prompt_tokens_.size()) != target_seen_) {
        throw std::invalid_argument("MTP requires every prompt token hidden state");
    }
    const size_t embd = source_.hparams.n_embd;
    for (size_t begin = 0; begin < prompt_tokens_.size();) {
        CheckCancelled(cancelled);
        const size_t count = std::min(static_cast<size_t>(kHeadBootstrapTokens), prompt_tokens_.size() - begin);
        auto input = BuildNextNBootstrapBatch(prompt_tokens_, prompt_hidden_, embd, begin, count);
        ForwardHead(input.tokens, input.hidden, nullptr, deps);
        begin += count;
    }
    prompt_tokens_.clear(); prompt_tokens_.shrink_to_fit();
    prompt_hidden_.clear(); prompt_hidden_.shrink_to_fit();
    bootstrapped_ = true;
}
void ReferenceSession::ReplayObservedTarget(const InferenceDependencies* deps, const std::atomic<bool>& cancelled) {
    if (observed_target_tokens_.empty()) return;
    CheckCancelled(cancelled);
    const int keep = target_seen_ - static_cast<int>(observed_target_tokens_.size());
    if (keep < 0 || keep > head_past_ || !cache_->block_manager->RollbackExclusiveSuffix(&blocks_, keep))
        throw std::runtime_error("MTP accepted head-prefix rollback failed");
    head_past_ = keep;
    // Draft rollout used approximate hidden feedback. Accepted prefix KV must
    // instead use previous actual target h followed by current target h[0..j-1],
    // matching pinned llama speculative process(), not just the newest anchor.
    ForwardHead(observed_target_tokens_, observed_target_inputs_, nullptr, deps);
    observed_target_tokens_.clear();
    observed_target_inputs_.clear();
}
void ReferenceSession::MakeProposals(int selected_token, const SamplingParams& params,
                                     const InferenceDependencies* deps, const std::atomic<bool>& cancelled) {
    CheckCancelled(cancelled);
    if (densecore::llm::sampling::IsJsonGrammarComplete(params.grammar)) {
        pending_.clear();
        return;
    }
    const int keep = target_seen_;
    if (keep < 0 || keep > head_past_ || !cache_->block_manager->RollbackExclusiveSuffix(&blocks_, keep)) {
        throw std::runtime_error("MTP head rollback failed");
    }
    head_past_ = keep;
    std::vector<int> history = params.token_history ? *params.token_history : std::vector<int>{};
    GrammarConstraint grammar = params.grammar ? *params.grammar : GrammarConstraint{};
    SamplingParams draft_params = params;
    draft_params.token_history = &history;
    if (params.grammar) draft_params.grammar = &grammar;
    std::vector<float> hidden = last_hidden_;
    const int available = std::min(draft_n_max_, max_context_ - head_past_);
    for (int i = 0; i < available; ++i) {
        CheckCancelled(cancelled);
        history.push_back(selected_token);
        if (history.size() > 2048) history.erase(history.begin(), history.begin() + (history.size() - 2048));
        if (params.grammar && densecore::llm::sampling::AdvanceJsonGrammarAndCheckCompletion(
                                  &grammar, Tokenizer::Detokenize(&source_, selected_token))) break;
        auto step = ForwardHead({selected_token}, hidden, &draft_params, deps);
        pending_.push_back({step.token, std::move(step.distribution)});
        ++proposed_;
        selected_token = step.token;
        hidden = std::move(step.hidden);
        if (selected_token == source_.eos_token_id ||
            std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), selected_token) != source_.stop_token_ids.end()) break;
    }
}
std::vector<int> ReferenceSession::VerificationDraft(int maximum) const {
    if (maximum < 0) throw std::invalid_argument("negative verification draft limit");
    const size_t count = std::min(pending_.size(), static_cast<size_t>(maximum));
    std::vector<int> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(pending_[i].token);
    return result;
}
VerificationResult ReferenceSession::VerifyBatched(ggml_tensor* logits, int first_column, size_t draft_count,
                                                   const SamplingParams& params, std::mt19937& target_rng) {
    if (!logits || first_column < 0 || draft_count < 1 || draft_count > pending_.size() ||
        static_cast<size_t>(first_column) + draft_count >= static_cast<size_t>(logits->ne[1]))
        throw std::invalid_argument("invalid batched target logits/proposals");
    VerificationResult result;
    result.tokens.reserve(draft_count + 1);
    auto history = params.token_history ? *params.token_history : std::vector<int>{};
    GrammarConstraint grammar = params.grammar ? *params.grammar : GrammarConstraint{};
    SamplingParams transformed = params;
    transformed.token_history = &history;
    if (params.grammar) transformed.grammar = &grammar;
    for (size_t i = 0; i < draft_count; ++i) {
        transformed.output_token_index = params.output_token_index + static_cast<int>(i);
        const auto target = BuildSamplingDistribution(logits, first_column + static_cast<int>(i), transformed);
        const auto step = VerifyToken(pending_[i].token, pending_[i].distribution, target,
                                      params.temperature <= 0, accept_rng_, target_rng);
        result.tokens.push_back(step.token);
        ++verified_;
        if (!step.accepted) { ++rejected_; break; }
        ++accepted_;
        if (step.token == source_.eos_token_id ||
            std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), step.token) != source_.stop_token_ids.end()) {
            // Treat terminal output as the final committed token, retaining
            // target state before consuming EOS instead of sampling beyond it.
            break;
        }
        if (params.grammar && densecore::llm::sampling::AdvanceJsonGrammarAndCheckCompletion(
                                  &grammar, Tokenizer::Detokenize(&source_, step.token))) {
            // Like EOS, this is a committed output, not an input to retain.
            // Do not inspect a following target row or sample a bonus token.
            break;
        }
        ++result.accepted_draft_tokens;
        history.push_back(step.token);
        if (history.size() > 2048) history.erase(history.begin());
    }
    if (result.accepted_draft_tokens == draft_count) {
        transformed.output_token_index = params.output_token_index + static_cast<int>(draft_count);
        result.tokens.push_back(SampleTokenWithRng(logits, first_column + static_cast<int>(draft_count), transformed, target_rng));
    }
    pending_.clear();
    return result;
}
void ReferenceSession::CompleteBatched(const int* inputs, const float* hidden, size_t stride,
                                        const VerificationResult& result, const SamplingParams& params,
                                        const InferenceDependencies* deps, const std::atomic<bool>& cancelled) {
    if (result.tokens.size() != result.accepted_draft_tokens + 1 || result.tokens.empty())
        throw std::invalid_argument("invalid verification output boundary");
    ObserveTarget(inputs, hidden, stride, static_cast<int>(result.accepted_draft_tokens + 1), false);
    ReplayObservedTarget(deps, cancelled);
    auto history = params.token_history ? *params.token_history : std::vector<int>{};
    GrammarConstraint grammar = params.grammar ? *params.grammar : GrammarConstraint{};
    for (size_t i = 0; i + 1 < result.tokens.size(); ++i) {
        history.push_back(result.tokens[i]);
        if (history.size() > 2048) history.erase(history.begin());
        if (params.grammar) grammar.UpdateState(Tokenizer::Detokenize(&source_, result.tokens[i]));
    }
    SamplingParams future = params;
    future.token_history = &history;
    if (params.grammar) future.grammar = &grammar;
    const int selected = result.tokens.back();
    const bool contains_stop = std::any_of(result.tokens.begin(), result.tokens.end(), [&](int token) {
        return token == source_.eos_token_id ||
            std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), token) != source_.stop_token_ids.end();
    });
    if (!contains_stop && target_seen_ < max_context_ && selected != source_.eos_token_id &&
        std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), selected) == source_.stop_token_ids.end())
        MakeProposals(selected, future, deps, cancelled);
}
int ReferenceSession::Sample(ggml_tensor* target_logits, int column, const SamplingParams& params,
                             std::mt19937& target_rng, const InferenceDependencies* deps,
                             const std::atomic<bool>& cancelled) {
    CheckCancelled(cancelled);
    int selected;
    if (pending_.empty()) {
        selected = SampleTokenWithRng(target_logits, column, params, target_rng);
    } else {
        const auto p = BuildSamplingDistribution(target_logits, column, params);
        const auto proposal = std::move(pending_.front()); pending_.pop_front();
        const auto result = VerifyToken(proposal.token, proposal.distribution, p,
                                        params.temperature <= 0, accept_rng_, target_rng);
        selected = result.token;
        ++verified_;
        if (result.accepted) ++accepted_;
        else { ++rejected_; pending_.clear(); }
    }
    Bootstrap(deps, cancelled);
    ReplayObservedTarget(deps, cancelled);
    if (pending_.empty() && target_seen_ < max_context_ && selected != source_.eos_token_id &&
        std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), selected) == source_.stop_token_ids.end()) {
        MakeProposals(selected, params, deps, cancelled);
    }
    return selected;
}
}  // namespace densecore::llm::speculative
