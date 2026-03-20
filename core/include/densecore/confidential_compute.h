/**
 * @file confidential_compute.h
 * @brief Enterprise confidential-computing state bridge
 *
 * Go-side attestation verification can publish a verified confidential
 * execution state into the enterprise plugin. Runtime middleware then reads
 * the same state via a single vtable, preserving fail-close semantics.
 */

#ifndef DENSECORE_CONFIDENTIAL_COMPUTE_H
#define DENSECORE_CONFIDENTIAL_COMPUTE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ENT_API __attribute__((visibility("default")))
#else
#define DENSECORE_ENT_API
#endif

#define DENSECORE_ENT_CONF_PROVIDER_LEN 16
#define DENSECORE_ENT_CONF_MEASUREMENT_LEN 96
#define DENSECORE_ENT_CONF_HOST_DATA_LEN 96

typedef struct {
    int32_t is_attested;
    char provider[DENSECORE_ENT_CONF_PROVIDER_LEN];
    char measurement[DENSECORE_ENT_CONF_MEASUREMENT_LEN];
    char host_data[DENSECORE_ENT_CONF_HOST_DATA_LEN];
    uint64_t verified_at_unix;
} DenseCoreEntConfidentialState;

typedef int (*DenseCoreEntSetConfidentialStateFn)(const DenseCoreEntConfidentialState* state);
typedef int (*DenseCoreEntGetConfidentialStateFn)(DenseCoreEntConfidentialState* out_state);
typedef int (*DenseCoreEntIsConfidentialReadyFn)(void);

typedef struct {
    DenseCoreEntSetConfidentialStateFn set_state;
    DenseCoreEntGetConfidentialStateFn get_state;
    DenseCoreEntIsConfidentialReadyFn is_ready;
} DenseCoreEntConfidentialVTable;

DENSECORE_ENT_API void DenseCoreEntRegisterConfidential(const DenseCoreEntConfidentialVTable* vtable);
DENSECORE_ENT_API const DenseCoreEntConfidentialVTable* DenseCoreEntGetConfidentialVTable(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_CONFIDENTIAL_COMPUTE_H
