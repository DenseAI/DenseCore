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
        const uint64_t rounded = Add(context, BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const uint64_t head_bpt = ComputeKVCacheBytesPerToken(GGML_TYPE_F16, model.hparams.n_embd_head_k,
            model.hparams.n_embd_head_v, model.hparams.n_head_kv, 1);
        const uint64_t head_kv = Mul(Mul(head_bpt, rounded), concurrency);
        const uint64_t prompt_hidden = Mul(Mul(Mul(context, model.hparams.n_embd), sizeof(float)), concurrency);
        const uint64_t scores = Mul(Mul(16, model.hparams.n_head), context);
        const uint64_t graph = Add(64ULL * 1024 * 1024,
                                  Mul(sizeof(float), Add(Mul(Mul(16, model.hparams.n_embd),32), scores)));
        const uint64_t distributions = Mul(Mul(Mul(draft_n_max + 3, model.hparams.n_vocab), sizeof(double)), concurrency);
        const uint64_t additional = Add(Add(head_kv,prompt_hidden),Add(graph,distributions));
        // The reference path does not speculate the target, so it allocates no
        // recurrent target checkpoints. This single MTP reservation includes
        // independent head KV, prompt hidden copies, graph and probability data.
        plan.mtp_checkpoint_bytes = additional;
        if (!plan.Fits()) {
            return fail("MTP resource plan required_bytes=" + std::to_string(plan.RequiredBytes()) +
                        " available_bytes=" + std::to_string(plan.available_bytes));
        }
        entry->kv_config.memory_plan = plan;
        state->speculation_max_context = static_cast<int>(context);
        state->speculation_reserved_bytes = additional;
        state->speculation_reason = "explicit sequential MTP reference; automatic performance qualification pending";
        state->speculation_selected.store(2, std::memory_order_release);
        LOG_INFO("MTP reference selected: context={} concurrency={} draft_n_max={} head_kv={} prompt_hidden={} graph={} distributions={} reservation={}",
                 context,concurrency,draft_n_max,head_kv,prompt_hidden,graph,distributions,additional);
        return true;
    } catch (const std::exception& e) {
        return fail(std::string("MTP initialization failed: ") + e.what());
    }
}
