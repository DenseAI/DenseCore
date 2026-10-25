#include "llm/speculative/reference_session.h"
#include "llm/sampling/sampling_rng.h"
#include "densecore/models/tokenizer.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace densecore::llm::speculative {
namespace {
class RestoreInferenceContext {
public:
    RestoreInferenceContext() : work_(GetCurrentWorkContext()), batch_(GetCurrentBatch()), phase_(GetCurrentExecutionPhase()) {}
    ~RestoreInferenceContext() {
        SetCurrentWorkContext(work_); SetCurrentBatch(batch_); SetCurrentExecutionPhase(phase_);
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
ReferenceSession::ReferenceSession(TransformerModel& model, int max_context, int draft_n_max,
                                   uint64_t seed, int threads)
    : source_(model), head_(model), work_(CreateInferenceWorkContext(), DestroyInferenceWorkContext),
      max_context_(max_context), draft_n_max_(draft_n_max), threads_(std::max(1, threads)) {
    if (max_context < 1 || draft_n_max < 1 || draft_n_max > 3 || !work_) {
        throw std::invalid_argument("invalid MTP reference session configuration");
    }
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
    last_hidden_.assign(hidden + static_cast<size_t>(count - 1) * stride,
                        hidden + static_cast<size_t>(count - 1) * stride + embd);
    target_seen_ += count;
}
ReferenceSession::HeadOutput ReferenceSession::ForwardHead(const std::vector<int>& tokens,
                                                           const std::vector<float>& hidden,
                                                           const SamplingParams* params,
                                                           const InferenceDependencies* deps) {
    const size_t embd = source_.hparams.n_embd;
    if (tokens.empty() || hidden.size() != tokens.size() * embd ||
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
    // Bound bootstrap chunks to 16. Include the attention score matrix in the
    // reference arena; this allocation must be reserved by the engine planner.
    const size_t scratch = 64ULL * 1024 * 1024 +
        sizeof(float) * tokens.size() * (embd * 32 + source_.hparams.n_head * static_cast<size_t>(head_past_ + tokens.size()));
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
    for (size_t begin = 0; begin + 1 < prompt_tokens_.size();) {
        CheckCancelled(cancelled);
        const size_t count = std::min(size_t{16}, prompt_tokens_.size() - 1 - begin);
        std::vector<int> tokens(prompt_tokens_.begin() + begin + 1, prompt_tokens_.begin() + begin + 1 + count);
        std::vector<float> hidden(prompt_hidden_.begin() + begin * embd, prompt_hidden_.begin() + (begin + count) * embd);
        ForwardHead(tokens, hidden, nullptr, deps);
        begin += count;
    }
    prompt_tokens_.clear(); prompt_tokens_.shrink_to_fit();
    prompt_hidden_.clear(); prompt_hidden_.shrink_to_fit();
    bootstrapped_ = true;
}
void ReferenceSession::MakeProposals(int selected_token, const SamplingParams& params,
                                     const InferenceDependencies* deps, const std::atomic<bool>& cancelled) {
    const int keep = target_seen_ - 1;
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
        if (params.grammar) grammar.UpdateState(Tokenizer::Detokenize(&source_, selected_token));
        auto step = ForwardHead({selected_token}, hidden, &draft_params, deps);
        pending_.push_back({step.token, std::move(step.distribution)});
        ++proposed_;
        selected_token = step.token;
        hidden = std::move(step.hidden);
        if (selected_token == source_.eos_token_id ||
            std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), selected_token) != source_.stop_token_ids.end()) break;
    }
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
        if (result.accepted) ++accepted_;
        else pending_.clear();
    }
    Bootstrap(deps, cancelled);
    if (pending_.empty() && target_seen_ < max_context_ && selected != source_.eos_token_id &&
        std::find(source_.stop_token_ids.begin(), source_.stop_token_ids.end(), selected) == source_.stop_token_ids.end()) {
        MakeProposals(selected, params, deps, cancelled);
    }
    return selected;
}
}  // namespace densecore::llm::speculative
