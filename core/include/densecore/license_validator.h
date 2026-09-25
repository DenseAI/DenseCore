/**
 * @file license_validator.h
 * @brief Neutral license-validation ABI for an enterprise plugin
 *
 * DenseCore OSS only registers and returns this function table. Signature
 * verification, state management, expiry checks, and fail-close enforcement
 * are responsibilities of the separately loaded enterprise plugin and its
 * consumers.
 */

#ifndef DENSECORE_LICENSE_VALIDATOR_H
#define DENSECORE_LICENSE_VALIDATOR_H

#include <stddef.h>
#include <stdint.h>
#include "densecore/enterprise_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ENT_API __attribute__((visibility("default")))
#else
#define DENSECORE_ENT_API
#endif

// =============================================================================
// License Status Codes
// =============================================================================

typedef enum {
    DENSECORE_LIC_OK = 0,              ///< License is valid and active
    DENSECORE_LIC_EXPIRED = -1,        ///< License has expired
    DENSECORE_LIC_INVALID_SIG = -2,    ///< Signature verification failed
    DENSECORE_LIC_MALFORMED = -3,      ///< Payload cannot be parsed
    DENSECORE_LIC_NOT_SET = -4,        ///< No license has been set
    DENSECORE_LIC_CORE_EXCEEDED = -5,  ///< Core count exceeds license limit
    DENSECORE_LIC_RATE_EXCEEDED = -6,  ///< Token rate exceeds license limit
} DenseCoreEntLicenseStatus;

// =============================================================================
// License Feature Flags (subset of capability bitmask for license tiers)
// =============================================================================

#define DENSECORE_LIC_FEAT_TELEMETRY (1ULL << 0)
#define DENSECORE_LIC_FEAT_NUMA_ROUTING (1ULL << 1)
#define DENSECORE_LIC_FEAT_CONFIDENTIAL (1ULL << 2)
#define DENSECORE_LIC_FEAT_AUDIT (1ULL << 3)
#define DENSECORE_LIC_FEAT_OIDC (1ULL << 4)
#define DENSECORE_LIC_FEAT_QUOTA (1ULL << 5)
#define DENSECORE_LIC_FEAT_SIEM (1ULL << 6)

// =============================================================================
// License Payload (validated claims)
// =============================================================================

/**
 * @brief License claims exchanged with the enterprise plugin
 *
 * @note All string fields are null-terminated, max 64 chars.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    /** License expiry time (Unix epoch seconds, UTC) */
    int64_t expiry_unix;

    /** Maximum allowed CPU cores (0 = unlimited) */
    int32_t max_cores;

    /** Enabled feature bitmask (DENSECORE_LIC_FEAT_*) */
    uint64_t features_bitmask;

    /** Licensed customer identifier (not an inference tenant) */
    char customer_id[64];

    /** Deployment identifier bound to the tenant bundle */
    char deployment_id[64];

    /** License issue time (Unix epoch seconds) */
    int64_t issued_at;

    /** License serial number */
    char serial[64];

    /** License tier: "starter", "scale", "custom" */
    char tier[16];

    /** Licensed product identifier. v2 permits one product per token. */
    char product[16];

    /** Custom tier requires a fresh verified attestation when non-zero. */
    int32_t attestation_required;
} DenseCoreEntLicensePayload;

// =============================================================================
// License Validation API (implemented by enterprise plugin)
// =============================================================================

/**
 * @brief Verify Ed25519 signature
 *
 * Verifies the signature over the payload using the given public key.
 * Returns DENSECORE_LIC_OK on valid signature, negative error code on failure.
 *
 * @param pubkey       Ed25519 public key (32 bytes)
 * @param pubkey_len   Must be 32
 * @param signature    Ed25519 signature (64 bytes)
 * @param sig_len      Must be 64
 * @param payload      Raw payload bytes (the signed message)
 * @param payload_len  Length of payload
 * @return DENSECORE_LIC_OK on valid signature, negative on failure
 *
 * @note Thread-safety: thread-safe (pure function, no state).
 */
typedef DenseCoreEntLicenseStatus (*DenseCoreEntValidateLicenseFn)(const uint8_t* pubkey, size_t pubkey_len,
                                                                   const uint8_t* signature, size_t sig_len,
                                                                   const uint8_t* payload, size_t payload_len);

/**
 * @brief Activate license state (fail-close enforcement)
 *
 * Updates the state owned by the enterprise plugin. Passing NULL requests
 * invalidation; the plugin defines the resulting enforcement behavior.
 *
 * @param payload  Validated license payload (NULL to invalidate)
 * @return DENSECORE_LIC_OK on success
 *
 * @note Thread-safety: NOT thread-safe. Single writer (enforcement goroutine).
 */
typedef DenseCoreEntLicenseStatus (*DenseCoreEntSetLicenseStateFn)(const DenseCoreEntLicensePayload* payload);

/**
 * @brief Query current license state
 *
 * @param out_payload  Output buffer for current license payload (caller-owned)
 * @return Current license status (DENSECORE_LIC_OK if valid)
 *
 * @note Thread-safety: thread-safe (atomic read).
 */
typedef DenseCoreEntLicenseStatus (*DenseCoreEntGetLicenseStateFn)(DenseCoreEntLicensePayload* out_payload);

/**
 * @brief Fast license validity check
 *
 * Consumers may use this callback in an enforcement path. The enterprise
 * plugin owns its implementation and latency characteristics.
 *
 * @return 1 if license is valid, 0 if invalid/expired/not-set
 *
 * @note Thread-safety: thread-safe (atomic load, acquire).
 */
typedef int (*DenseCoreEntIsLicenseValidFn)(void);

// =============================================================================
// License Function Table (registered by enterprise plugin)
// =============================================================================

/**
 * @brief License validation function table
 *
 * Function pointer table registered by the enterprise plugin during init.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    DenseCoreEntValidateLicenseFn validate;
    DenseCoreEntSetLicenseStateFn set_state;
    DenseCoreEntGetLicenseStateFn get_state;
    DenseCoreEntIsLicenseValidFn is_valid;
} DenseCoreEntLicenseVTable;

/**
 * @brief Register the license VTable (called by plugin during init)
 *
 * @param vtable  Function table (plugin-owned, must remain valid until shutdown)
 *
 * @note Thread-safety: NOT thread-safe. Call once during plugin init.
 */
DENSECORE_ENT_API void DenseCoreEntRegisterLicense(const DenseCoreEntLicenseVTable* vtable);

/**
 * @brief Get the registered license VTable
 *
 * @return Registered vtable, or NULL if no license plugin
 *
 * @note Thread-safety: thread-safe after registration.
 */
DENSECORE_ENT_API const DenseCoreEntLicenseVTable* DenseCoreEntGetLicenseVTable(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_LICENSE_VALIDATOR_H
