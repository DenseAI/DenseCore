#include "densecore.h"

#include "runtime/engine_internal.h"

namespace {
// A profile must have same-file parity, rollback/streaming/isolation checks and
// matched performance evidence before becoming eligible here. No profile has
// passed those gates yet; accepting explicit MTP would silently run plain decode.
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
    if (spec == DENSECORE_SPEC_MTP) {
        SetLastError(DENSECORE_STATUS_UNSUPPORTED_OPERATION,
            "InitEngineWithOptions: explicit mtp is unavailable; no validated execution profile is registered");
        return nullptr;
    }
    if (kv_cache_type == -1) {
        return InitEngineEx(model_path, reserved, threads, numa_node_id, pinning_policy);
    }
    return InitEngineWithKVType(model_path, reserved, threads, numa_node_id, pinning_policy,
        static_cast<DenseCoreKVType>(kv_cache_type));
}

}  // extern "C"
