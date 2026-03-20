/**
 * @file enterprise_plugin.h
 * @brief DenseCore Enterprise Plugin Discovery & Loading API
 *
 * C API interface for dynamically loading enterprise plugins via dlopen/dlsym.
 * libdensecore_ent.so is loaded at runtime, injecting commercial features
 * without polluting the open-source core.
 *
 * Architecture:
 *   DenseCore (OSS) ──dlopen──▶ libdensecore_ent.so (Proprietary)
 *                    ◀──C API──
 *
 * The enterprise plugin is OPTIONAL. If the .so is not found,
 * DenseCore operates normally without any enterprise features.
 *
 * @note Thread-safety: Init/Shutdown are NOT thread-safe (call once at startup).
 *       All other query functions are thread-safe after init.
 */

#ifndef DENSECORE_ENTERPRISE_PLUGIN_H
#define DENSECORE_ENTERPRISE_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ENT_API __attribute__((visibility("default")))
#else
#define DENSECORE_ENT_API
#endif

// =============================================================================
// API Version & Plugin Symbol
// =============================================================================

/**
 * API version for binary compatibility checks.
 * Increment major on breaking changes, minor on additions.
 */
#define DENSECORE_ENT_API_VERSION_MAJOR 1
#define DENSECORE_ENT_API_VERSION_MINOR 0
#define DENSECORE_ENT_API_VERSION ((DENSECORE_ENT_API_VERSION_MAJOR << 16) | DENSECORE_ENT_API_VERSION_MINOR)

/**
 * Symbol name resolved via dlsym() after dlopen().
 * The enterprise .so MUST export this exact symbol.
 */
#define DENSECORE_ENT_PLUGIN_SYMBOL "DenseCoreEntPluginInit"

// =============================================================================
// Capability Flags (Bitmask)
// =============================================================================

/** Deep telemetry: KV fragmentation, expert hit ratio, tenant counters */
#define DENSECORE_ENT_CAP_TELEMETRY (1ULL << 0)

/** Cryptographic license validation (Ed25519) */
#define DENSECORE_ENT_CAP_LICENSE (1ULL << 1)

/** Enterprise NUMA routing (Zero-Cross-NUMA-Traffic) */
#define DENSECORE_ENT_CAP_NUMA_ROUTING (1ULL << 2)

/** Confidential computing (SEV-SNP / TDX attestation) */
#define DENSECORE_ENT_CAP_CONFIDENTIAL (1ULL << 3)

/** OIDC/SAML authentication integration */
#define DENSECORE_ENT_CAP_AUTH (1ULL << 4)

/** SIEM audit logging with PII redaction */
#define DENSECORE_ENT_CAP_AUDIT (1ULL << 5)

// =============================================================================
// Plugin Info & Lifecycle
// =============================================================================

/**
 * @brief Enterprise plugin metadata
 *
 * Self-description struct returned by the plugin after dlopen.
 * The api_version field is used for binary compatibility verification.
 */
typedef struct {
    uint32_t api_version;   ///< Must match DENSECORE_ENT_API_VERSION
    const char* name;       ///< Plugin name (e.g., "DenseEnterprise")
    const char* version;    ///< Semantic version string (e.g., "1.0.0")
    uint64_t capabilities;  ///< Bitmask of DENSECORE_ENT_CAP_* flags
} DenseCoreEntPluginInfo;

/**
 * @brief Plugin init function signature
 *
 * Function pointer type resolved via dlsym(DENSECORE_ENT_PLUGIN_SYMBOL).
 * Receives engine handle to initialize internal state, and returns
 * plugin info via out_info.
 *
 * @param engine    DenseCore engine handle (opaque, from InitEngine)
 * @param out_info  Output: pointer to static plugin info (owned by plugin)
 * @return 0 on success, negative on failure
 *
 * @note Thread-safety: NOT thread-safe. Call once during startup.
 * @note Ownership: out_info points to plugin-internal static storage.
 *       Valid until DenseCoreEntShutdown is called.
 */
typedef int (*DenseCoreEntInitFn)(void* engine, const DenseCoreEntPluginInfo** out_info);

/**
 * @brief Plugin shutdown function signature
 *
 * Called during graceful shutdown. Releases internal resources.
 *
 * @note Thread-safety: NOT thread-safe. Call once during shutdown.
 */
typedef void (*DenseCoreEntShutdownFn)(void);

// =============================================================================
// Plugin Loader API (called by DenseCore host process)
// =============================================================================

/**
 * @brief Attempt to load the enterprise plugin
 *
 * Opens the .so file via dlopen and resolves the plugin init symbol.
 * Gracefully fails if .so is absent or symbol is missing (runs without
 * enterprise features).
 *
 * Search policy:
 *   1. If plugin_path is provided, it is resolved against the DenseCore module directory.
 *      The resolved path must stay within that trusted directory and use the default
 *      enterprise plugin filename for the current platform.
 *   2. If plugin_path is NULL/empty, load the default enterprise plugin filename from
 *      the DenseCore module directory.
 *
 * @param plugin_path  Optional plugin path override (NULL/empty for default location)
 * @param engine       DenseCore engine handle
 * @return 0 on success (plugin loaded and initialized),
 *         1 if plugin not found (graceful, not an error),
 *         negative on validation/load/init failure
 *
 * @note Thread-safety: NOT thread-safe. Call once at startup.
 */
DENSECORE_ENT_API int DenseCoreEntLoadPlugin(const char* plugin_path, void* engine);

/**
 * @brief Query loaded plugin info
 *
 * @return Plugin info, or NULL if no plugin is loaded
 *
 * @note Thread-safety: thread-safe after DenseCoreEntLoadPlugin returns.
 */
DENSECORE_ENT_API const DenseCoreEntPluginInfo* DenseCoreEntGetPluginInfo(void);

/**
 * @brief Check if a specific capability is supported
 *
 * @param capability  DENSECORE_ENT_CAP_* flag to check
 * @return 1 if supported, 0 if not (or no plugin loaded)
 *
 * @note Thread-safety: thread-safe after DenseCoreEntLoadPlugin returns.
 */
DENSECORE_ENT_API int DenseCoreEntHasCapability(uint64_t capability);

/**
 * @brief Unload plugin and release resources
 *
 * @note Thread-safety: NOT thread-safe. Call once at shutdown.
 */
DENSECORE_ENT_API void DenseCoreEntUnloadPlugin(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_ENTERPRISE_PLUGIN_H
