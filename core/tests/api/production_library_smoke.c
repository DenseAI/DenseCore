/* Public-header-only consumer. Link against the production shared library,
 * never the separately compiled unit-test runtime. Optional argv[1] is a GGUF.
 */
#define _POSIX_C_SOURCE 200809L
#include "densecore.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

struct generation_observation {
    atomic_int terminals;
    atomic_int errors;
    atomic_int bytes;
};

static void observe_token(const char* data, int length, int token_id, int finished, void* opaque) {
    struct generation_observation* observation = opaque;
    (void)token_id;
    if (length > 0) {
        atomic_fetch_add(&observation->bytes, length);
    }
    if (finished) {
        if (length >= 6 && data && memcmp(data, "Error:", 6) == 0) {
            atomic_fetch_add(&observation->errors, 1);
        }
        atomic_fetch_add(&observation->terminals, 1);
    }
}

static int require_invalid(int status, const char* operation) {
    const char* error = DenseCoreGetLastError();
    if (status != DENSECORE_STATUS_INVALID_ARGUMENT || !error || !error[0]) {
        fprintf(stderr, "%s: expected invalid argument and thread-local error, got %d\n", operation, status);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    const DenseCoreVersionInfo* version = GetLibraryVersion();
    const char* version_string = GetLibraryVersionString();
    if (!version || !version->version || !version_string || !version_string[0] ||
        strcmp(version->version, version_string) != 0) {
        fprintf(stderr, "production library did not expose consistent version information\n");
        return 1;
    }
    if (!require_invalid(CancelRequest(NULL, 1), "CancelRequest") ||
        !require_invalid(SubmitRequest(NULL, "hello", 8, NULL, NULL, NULL), "SubmitRequest") ||
        !require_invalid(SubmitRequestIds(NULL, NULL, 0, 8, NULL, NULL), "SubmitRequestIds")) {
        return 1;
    }
    printf("production library version: %s\n", version_string);
    if (argc < 2) {
        puts("PASS: public version and invalid-argument ABI");
        return 0;
    }

    DenseCoreHandle engine = InitEngine(argv[1], NULL, 2);
    if (!engine) {
        fprintf(stderr, "InitEngine: %s\n", DenseCoreGetLastError());
        return 1;
    }
    struct generation_observation observation = {0};
    const int request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine, "What is the capital of France? Answer briefly.", 8, NULL, 0.0f, 1.0f, 1, 1.0f, NULL, 0, NULL, 0, 0,
        NULL, 0, observe_token, &observation);
    if (request_id <= 0) {
        fprintf(stderr, "generation submission: %d: %s\n", request_id, DenseCoreGetLastError());
        FreeEngine(engine);
        return 1;
    }

#ifndef _WIN32
    const struct timespec pause = {0, 10000000};
#endif
    for (int attempt = 0; attempt < 6000 && atomic_load(&observation.terminals) == 0; ++attempt) {
#ifdef _WIN32
        Sleep(10);
#else
        nanosleep(&pause, NULL);
#endif
    }
    if (atomic_load(&observation.terminals) == 0) {
        fprintf(stderr, "generation did not terminate within 60 seconds\n");
        CancelRequest(engine, request_id);
        FreeEngine(engine);
        return 1;
    }
    /* Callback completion is not retirement. FreeEngine joins producers. */
    FreeEngine(engine);
    if (atomic_load(&observation.terminals) != 1 || atomic_load(&observation.errors) != 0 ||
        atomic_load(&observation.bytes) <= 0) {
        fprintf(stderr, "generation terminal contract: terminals=%d errors=%d bytes=%d\n",
                atomic_load(&observation.terminals), atomic_load(&observation.errors), atomic_load(&observation.bytes));
        return 1;
    }
    puts("PASS: production-library generation callback and teardown");
    return 0;
}
