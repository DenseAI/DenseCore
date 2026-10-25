#include "runtime/speculation_config.h"
#include "runtime/engine_internal.h"
#include "llm/speculative/nextn_graph.h"
#include <limits>

namespace {
uint64_t Add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("MTP resource size overflow");
    return a + b;
}
uint64_t Mul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("MTP resource size overflow");
    return a * b;
}
}
uint64_t SpeculationMemoryCost::Total() const {
    return Add(Add(Add(head_kv, prompt_hidden), prompt_tokens), Add(Add(graph, distributions), checkpoints));
}
SpeculationMemoryCost CalculateSpeculationMemory(const TransformerModel& model, uint64_t context,
                                                 uint64_t concurrency, int draft_n_max) {
    const auto dimension_fits = [](uint64_t value) {
        return value > 0 && value <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    };
    if (!context || !concurrency || draft_n_max < 1 || draft_n_max > 3 ||
        !dimension_fits(model.hparams.n_embd) || !dimension_fits(model.hparams.n_head) ||
        !dimension_fits(model.hparams.n_vocab) || !dimension_fits(model.hparams.n_head_kv) ||
        model.hparams.n_embd_head_k > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        model.hparams.n_embd_head_v > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid MTP resource dimensions");
    const uint64_t rounded = Mul(Add(context, BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    const int key_dim = model.hparams.n_embd_head_k > 0 ? model.hparams.n_embd_head_k
        : model.hparams.n_embd / model.hparams.n_head;
    const int value_dim = model.hparams.n_embd_head_v > 0 ? model.hparams.n_embd_head_v : key_dim;
    if (key_dim <= 0 || value_dim <= 0 || model.hparams.n_head_kv <= 0)
        throw std::invalid_argument("invalid MTP head KV shape");
    const uint64_t head_bpt = ComputeKVCacheBytesPerToken(GGML_TYPE_F16, key_dim, value_dim, model.hparams.n_head_kv, 1);
    if (!head_bpt) throw std::invalid_argument("invalid MTP head KV shape");
    SpeculationMemoryCost cost;
    cost.head_kv = Mul(Mul(head_bpt, rounded), concurrency);
    cost.prompt_hidden = Mul(Mul(Mul(context, model.hparams.n_embd), sizeof(float)), concurrency);
    cost.prompt_tokens = Mul(Mul(context, sizeof(int)), concurrency);
    const uint64_t scores = Mul(Mul(16, model.hparams.n_head), context);
    cost.graph = Add(64ULL * 1024 * 1024,
        Mul(sizeof(float), Add(Mul(Mul(16, model.hparams.n_embd), 32), scores)));
    cost.distributions = Mul(Mul(Mul(draft_n_max + 3, model.hparams.n_vocab), sizeof(double)), concurrency);
    cost.checkpoints = densecore::llm::speculative::TargetCheckpointBank::RequiredBytes(model, draft_n_max);
    (void)cost.Total();
    return cost;
}

bool ConfigureSpeculation(EngineState* state, int mode, int draft_n_max, std::string* error) {
    const auto fail = [&](const std::string& message) { if (error) *error = message; return false; };
    if (!state || mode < 0 || mode > 2 || draft_n_max < 1 || draft_n_max > 3) {
        return fail("spec_type must be auto/none/mtp and spec_draft_n_max must be 1..3");
    }
    if (state->speculation_selected.load(std::memory_order_acquire) == 2) {
        return fail("MTP mode is fixed at engine initialization");
    }
    state->speculation_requested.store(mode, std::memory_order_relaxed);
    state->spec_draft_n_max.store(draft_n_max, std::memory_order_relaxed);
    if (mode != 2) {
        state->speculation_reason = mode == 1 ? "explicitly disabled" : "no qualified automatic MTP profile";
        state->speculation_selected.store(1, std::memory_order_release);
        return true;
    }
    if (state->models.size() != 1) return fail("explicit MTP requires one target model; separate draft/additional models are unsupported");
    auto* entry = state->GetDefaultModel();
    if (!entry || !entry->model || !entry->kv_cache) return fail("MTP requires a loaded target model");
    auto& model = *entry->model;
    if (model.variant != ModelVariant::QWEN35 && model.variant != ModelVariant::QWEN36) {
        return fail("explicit MTP supports Qwen3.5/3.6 NextN graphs only");
    }
    try {
        densecore::llm::speculative::NextNGraph validate(model);
        auto plan = entry->kv_config.memory_plan;
        const uint64_t context = static_cast<uint64_t>(entry->kv_config.max_seq_len);
        const uint64_t concurrency = static_cast<uint64_t>(entry->kv_config.max_num_seqs);
        if (context == 0 || concurrency == 0 || plan.available_bytes == 0) {
            return fail("MTP requires an effective context/concurrency memory plan");
        }
        const auto cost = CalculateSpeculationMemory(model, context, concurrency, draft_n_max);
        const uint64_t additional = cost.Total();
        const uint64_t reserved_mtp = plan.mtp_checkpoint_bytes;
        // One fenced verification graph at a time shares a model-owned bank.
        // Sessions keep independent head KV and hidden/distribution state.
        plan.mtp_checkpoint_bytes = additional;
        if (!plan.Fits()) {
            return fail("MTP resource plan required_bytes=" + std::to_string(plan.RequiredBytes()) +
                        " available_bytes=" + std::to_string(plan.available_bytes));
        }
        if (!entry->resource_lease) return fail("MTP requires an enforced model resource reservation");
        std::string reservation_error;
        if (reserved_mtp == 0) {
            if (!entry->resource_lease->ExtendMTP(additional, &reservation_error)) return fail(reservation_error);
        } else if (reserved_mtp != additional) {
            return fail("MTP preallocation resource contract does not match initialization cost");
        }
        entry->mtp_target_checkpoints = std::make_unique<densecore::llm::speculative::TargetCheckpointBank>(model, draft_n_max);
        entry->kv_config.memory_plan = plan;
        state->speculation_max_context = static_cast<int>(context);
        state->speculation_reserved_bytes = additional;
        state->speculation_reason = "explicit batched MTP with serialized verification; automatic performance qualification pending";
        state->speculation_selected.store(2, std::memory_order_release);
        LOG_INFO("MTP reference selected: context={} concurrency={} draft_n_max={} head_kv={} prompt_hidden={} graph={} distributions={} reservation={}",
                 context,concurrency,draft_n_max,cost.head_kv,cost.prompt_hidden,cost.graph,cost.distributions,additional);
        return true;
    } catch (const std::exception& e) {
        return fail(std::string("MTP initialization failed: ") + e.what());
    }
}
