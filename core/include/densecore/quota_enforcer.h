/**
 * @file quota_enforcer.h
 * @brief Optional enterprise request-quota ABI.
 *
 * DenseCore owns only the neutral registration point. The commercial plugin
 * owns policy and state and may be absent in OSS deployments.
 */
#ifndef DENSECORE_QUOTA_ENFORCER_H
#define DENSECORE_QUOTA_ENFORCER_H

#include "densecore/license_validator.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ENT_API __attribute__((visibility("default")))
#else
#define DENSECORE_ENT_API
#endif

typedef DenseCoreEntLicenseStatus (*DenseCoreEntQuotaTryBeginRequestFn)(int64_t reserved_tokens);
typedef void (*DenseCoreEntQuotaEndRequestFn)(void);
typedef DenseCoreEntLicenseStatus (*DenseCoreEntQuotaRecordTokensFn)(int64_t token_count);
typedef int32_t (*DenseCoreEntQuotaGetActiveRequestsFn)(void);

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    DenseCoreEntQuotaTryBeginRequestFn try_begin_request;
    DenseCoreEntQuotaEndRequestFn end_request;
    DenseCoreEntQuotaRecordTokensFn record_tokens;
    DenseCoreEntQuotaGetActiveRequestsFn get_active_requests;
} DenseCoreEntQuotaVTable;

DENSECORE_ENT_API void DenseCoreEntRegisterQuota(const DenseCoreEntQuotaVTable* vtable);
DENSECORE_ENT_API const DenseCoreEntQuotaVTable* DenseCoreEntGetQuotaVTable(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_QUOTA_ENFORCER_H
