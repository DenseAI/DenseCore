#include "densecore.h"

#include "runtime/engine_internal.h"
#include <cstdio>

namespace {
// A profile must have same-file parity, rollback/streaming/isolation checks and
// matched performance evidence before becoming eligible here. No profile has
// passed those gates yet. Explicit MTP uses the sequential correctness reference.
constexpr const char* kAutoReason = "none: no qualified model/CPU/concurrency MTP profile registered";
}

extern "C" {

DENSECORE_API const char* DenseCoreSpeculationAutoSelectionReason(void) {
    return kAutoReason;
}

DENSECORE_API DenseCoreHandle InitEngineWithOptions(const char* model_path, const char* reserved, int threads,
    int numa_node_id, int pinning_policy, int kv_cache_type, const DenseCoreInitOptions* options) {
    ClearLastError();
    const auto invalid = [](const char* reason) -> DenseCoreHandle {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, reason);
        return nullptr;
    };
    int spec = DENSECORE_SPEC_AUTO;
    int draft_limit = 2;
    if (options) {
        // Check the header before reading the rest of a potentially older struct.
        if (options->struct_size < sizeof(DenseCoreInitOptions)) {
            return invalid("InitEngineWithOptions: truncated options structure");
        }
        if (options->version != DENSECORE_INIT_OPTIONS_VERSION) {
            return invalid("InitEngineWithOptions: unsupported options version");
        }
        spec = options->spec_type;
        draft_limit = options->spec_draft_n_max;
    }
    if (spec < DENSECORE_SPEC_AUTO || spec > DENSECORE_SPEC_MTP) {
        return invalid("InitEngineWithOptions: spec_type must be auto, none or mtp");
    }
    if (draft_limit < 1 || draft_limit > 3) {
        return invalid("InitEngineWithOptions: spec_draft_n_max must be 1, 2 or 3");
    }
    if (kv_cache_type < -1 || kv_cache_type > DENSECORE_KV_INT4) {
        return invalid("InitEngineWithOptions: unsupported KV cache type");
    }
    return InitializeEngineWithOptionsInternal(model_path, reserved, threads, numa_node_id, pinning_policy,
                                               kv_cache_type, spec, draft_limit);
}

DENSECORE_API int DenseCoreGetRuntimePlan(DenseCoreHandle handle, DenseCoreRuntimePlan* out) {
    if (!handle || !out || out->struct_size < sizeof(DenseCoreRuntimePlan) || out->version != 1) {
        SetLastError(DENSECORE_STATUS_INVALID_ARGUMENT, "DenseCoreGetRuntimePlan: invalid handle or structure header");
        return DENSECORE_STATUS_INVALID_ARGUMENT;
    }
    auto* state = static_cast<EngineState*>(handle);
    std::lock_guard<std::mutex> lock(state->models_mu);
    const auto found = state->models.find(state->default_model_id);
    if (found == state->models.end() || !found->second) return DENSECORE_STATUS_INVALID_ARGUMENT;
    const auto& config = found->second->kv_config;
    const auto& memory = config.memory_plan;
    DenseCoreRuntimePlan result{};
    result.struct_size = sizeof(result);
    result.version = 1;
    result.effective_threads = state->n_threads;
    result.effective_context = config.max_seq_len;
    result.effective_concurrency = config.max_num_seqs;
    result.spec_requested = state->speculation_requested.load(std::memory_order_acquire);
    result.spec_selected = state->speculation_selected.load(std::memory_order_acquire);
    result.spec_draft_n_max = state->spec_draft_n_max.load(std::memory_order_acquire);
    result.available_bytes = memory.available_bytes;
    result.required_bytes = memory.RequiredBytes();
    result.kv_bytes = memory.kv_bytes;
    result.graph_bytes = memory.graph_bytes;
    result.repack_bytes = memory.repack_bytes;
    result.recurrent_bytes = memory.recurrent_bytes;
    result.prefix_snapshot_bytes = memory.prefix_snapshot_bytes;
    result.mtp_bytes = memory.mtp_checkpoint_bytes;
    result.allocator_bytes = memory.allocator_bytes;
    std::snprintf(result.spec_reason, sizeof(result.spec_reason), "%s", state->speculation_reason.c_str());
    *out = result;
    ClearLastError();
    return DENSECORE_STATUS_OK;
}

}  // extern "C"
