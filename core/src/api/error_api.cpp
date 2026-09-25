#include "densecore.h"

#include <string>

#include "runtime/engine_internal.h"

namespace {
thread_local std::string g_last_error;
}  // namespace

extern "C" {

DENSECORE_API const char* DenseCoreGetLastError(void) {
    return g_last_error.c_str();
}

}  // extern "C"

void SetLastError(DenseCoreStatus status, const std::string& message) {
    (void)status;
    g_last_error = message;
}

void ClearLastError() {
    g_last_error.clear();
}

DenseCoreStatus MapErrorCodeToStatus(densecore::ErrorCode code) {
    switch (code) {
    case densecore::ErrorCode::SUCCESS: return DENSECORE_STATUS_OK;
    case densecore::ErrorCode::INVALID_PARAMETERS: return DENSECORE_STATUS_INVALID_ARGUMENT;
    case densecore::ErrorCode::MODEL_LOAD_FAILED: return DENSECORE_STATUS_MODEL_LOAD_FAILED;
    case densecore::ErrorCode::OUT_OF_MEMORY: return DENSECORE_STATUS_OUT_OF_MEMORY;
    case densecore::ErrorCode::KV_CACHE_FULL: return DENSECORE_STATUS_OUT_OF_MEMORY;
    case densecore::ErrorCode::BACKEND_ERROR: return DENSECORE_STATUS_BACKEND_ERROR;
    case densecore::ErrorCode::UNSUPPORTED_OPERATION: return DENSECORE_STATUS_UNSUPPORTED_OPERATION;
    case densecore::ErrorCode::INFERENCE_FAILED: return DENSECORE_STATUS_INTERNAL_ERROR;
    default: return DENSECORE_STATUS_INTERNAL_ERROR;
    }
}
