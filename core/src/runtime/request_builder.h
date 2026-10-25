#pragma once

#include <exception>
#include <cstdint>
#include <new>
#include <string>

#include "densecore.h"
#include "densecore/exceptions.h"

struct EngineState;
struct Request;
class RequestGuard;

// Shared request-pool initialization for text and generic-graph C API adapters.
// The caller owns the result until pending publication succeeds.
Request* AcquireAndInitRequest(EngineState* state, uint64_t payload_bytes = 0);
Request* AcquireBoundedTextRequest(EngineState* state, const char* prompt, int input_tokens,
    int max_tokens, const char* lora = nullptr, const char* const* stops = nullptr,
    int allowed = 0, int disallowed = 0, bool reserve_terminal = true);
int PublishPreparedRequest(EngineState* state, Request* request, RequestGuard& guard,
                           const std::string& draining_error);

int RequestPreparationFailure(DenseCoreStatus status, const char* api, const char* detail) noexcept;

template <typename Prepare> int RequestApiBoundary(const char* api, Prepare&& prepare) {
    try {
        return prepare();
    } catch (const densecore::OutOfMemoryException& error) {
        return RequestPreparationFailure(DENSECORE_STATUS_OUT_OF_MEMORY, api, error.what());
    } catch (const std::bad_alloc&) {
        return RequestPreparationFailure(DENSECORE_STATUS_OUT_OF_MEMORY, api, "allocation failed");
    } catch (const std::exception& error) {
        return RequestPreparationFailure(DENSECORE_STATUS_INTERNAL_ERROR, api, error.what());
    } catch (...) {
        return RequestPreparationFailure(DENSECORE_STATUS_INTERNAL_ERROR, api, "request preparation failed");
    }
}
