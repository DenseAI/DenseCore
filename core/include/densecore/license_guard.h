/**
 * @file license_guard.h
 * @brief Inline license + quota check for DenseCore hot paths
 *
 * Include this header in DenseCore's memory allocator or inference scheduler
 * to enforce fail-close licensing and resource quotas.
 *
 * Usage in DenseCore allocator:
 *   #include "densecore/license_guard.h"
 *
 *   void* AllocateKVBlock() {
 *       if (!DenseCoreEntLicenseGuardAllow()) {
 *           return NULL; // License invalid or quota exceeded
 *       }
 *       // ... actual allocation
 *   }
 *
 * Design:
 *   - Zero overhead when enterprise plugin is not loaded (NULL vtable → allow)
 *   - Single branch (predicted as "allow") on the hot path
 *   - No dynamic memory allocation in check path
 */

#ifndef DENSECORE_LICENSE_GUARD_H
#define DENSECORE_LICENSE_GUARD_H

#include "densecore/license_validator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fast license validity check for allocator hot path.
 *
 * Returns 1 (allow) if:
 *   - No enterprise plugin loaded (OSS mode)
 *   - Enterprise plugin loaded and license is valid
 *
 * Returns 0 (deny) if:
 *   - Enterprise plugin loaded but license is invalid/expired/not-set
 *
 * @return 1 to allow allocation, 0 to deny
 *
 * @note Thread-safety: fully thread-safe (atomic reads only)
 * @note Complexity: O(1) — single atomic load + branch
 */
static inline int DenseCoreEntLicenseGuardAllow(void) {
    const DenseCoreEntLicenseVTable* vt = DenseCoreEntGetLicenseVTable();
    /* No enterprise plugin loaded → OSS mode → allow */
    if (!vt || !vt->is_valid) {
        return 1;
    }
    return vt->is_valid();
}

/**
 * @brief Get the current license status for detailed error reporting.
 *
 * Unlike DenseCoreEntLicenseGuardAllow() which is a fast boolean check,
 * this function returns the full license payload for diagnostics.
 *
 * @param out_payload  Optional output: current license payload
 * @return License status code, or DENSECORE_LIC_NOT_SET if no plugin
 *
 * @note Use this in error paths, not on the hot path.
 */
static inline DenseCoreEntLicenseStatus DenseCoreEntLicenseGuardStatus(DenseCoreEntLicensePayload* out_payload) {
    const DenseCoreEntLicenseVTable* vt = DenseCoreEntGetLicenseVTable();
    if (!vt || !vt->get_state) {
        return DENSECORE_LIC_NOT_SET;
    }
    return vt->get_state(out_payload);
}

#ifdef __cplusplus
}
#endif

#endif /* DENSECORE_LICENSE_GUARD_H */
