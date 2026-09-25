/**
 * @file telemetry_sink.h
 * @brief Lock-Free Enterprise Telemetry Counter Interface
 *
 * Deep metrics collection interface for SRE teams.
 * Beyond basic open-source metrics (queue length), this provides
 * KV cache fragmentation, MoE expert hit ratio, per-tenant token counters,
 * and other detailed indicators via lock-free atomic counters.
 *
 * Design:
 *   - All counters use std::atomic with relaxed writes / acquire reads
 *   - Zero overhead when enterprise plugin is not loaded (null pointer check)
 *   - Snapshot is a point-in-time consistent read (acquire fence)
 *
 * @note Thread-safety: All functions are thread-safe after plugin init.
 */

#ifndef DENSECORE_TELEMETRY_SINK_H
#define DENSECORE_TELEMETRY_SINK_H

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
// Constants
// =============================================================================

/** Maximum tenant counter slots (lock-free hash map size) */
#define DENSECORE_ENT_MAX_TENANTS 64

/** Maximum tenant UUID length (UUID v4 = 36 chars + null) */
#define DENSECORE_ENT_TENANT_ID_LEN 37

// =============================================================================
// Telemetry Snapshot (read-only point-in-time view)
// =============================================================================

/**
 * @brief Per-tenant token counters
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    char tenant_id[DENSECORE_ENT_TENANT_ID_LEN];  ///< Tenant UUID (null-terminated)
    uint64_t tokens_generated;                    ///< Total tokens generated for this tenant
    uint64_t prompt_tokens;                       ///< Total prompt tokens processed
} DenseCoreEntTenantMetrics;

/**
 * @brief Enterprise telemetry snapshot
 *
 * Point-in-time consistent data read via lock-free acquire fence.
 * Consumed by the Go exporter on each Prometheus scrape.
 *
 * Complexity: O(1) snapshot read (fixed-size counters)
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    // ---- KV Cache Deep Metrics ----
    /** KV Cache fragmentation ratio: 1.0 - (used_blocks / total_blocks) */
    float kv_fragmentation_ratio;

    /** Currently active KV cache blocks */
    uint64_t kv_active_blocks;

    /** Total available KV cache blocks */
    uint64_t kv_total_blocks;

    /** Copy-on-Write trigger count (prefix sharing efficiency indicator) */
    uint64_t kv_cow_count;

    // ---- MoE Expert Routing Metrics ----
    /** Hot expert cache hit ratio (0.0 ~ 1.0) */
    float hot_expert_hit_ratio;

    /** Cold expert cache hit ratio (0.0 ~ 1.0) */
    float cold_expert_hit_ratio;

    /** Total expert routing dispatches */
    uint64_t total_expert_dispatches;

    // ---- Per-Tenant Metrics ----
    /** Number of active tenants */
    uint32_t active_tenant_count;

    /** Per-tenant metrics array (only active_tenant_count entries are valid) */
    DenseCoreEntTenantMetrics tenants[DENSECORE_ENT_MAX_TENANTS];

    // ---- Resource Utilization ----
    /** Cross-NUMA memory transfer bytes (cross-socket traffic indicator) */
    uint64_t cross_numa_bytes;

    /** License utilization: current active cores / max allowed cores */
    float license_core_utilization;

    // ---- Timestamp ----
    /** Snapshot creation time (Unix epoch, nanoseconds) */
    uint64_t snapshot_timestamp_ns;
} DenseCoreEntTelemetrySnapshot;

// =============================================================================
// Telemetry Sink API (implemented by enterprise plugin)
// =============================================================================

/**
 * @brief Get current telemetry snapshot
 *
 * Reads atomically via lock-free acquire fence.
 * Copies data into the caller-provided out_snapshot buffer.
 *
 * @param out_snapshot  Output buffer (caller-owned)
 * @return 0 on success, -1 if telemetry not available (no plugin)
 *
 * @note Thread-safety: thread-safe. O(1) complexity.
 * @note Ownership: out_snapshot is owned by the caller.
 */
typedef int (*DenseCoreEntGetTelemetrySnapshotFn)(DenseCoreEntTelemetrySnapshot* out_snapshot);

/**
 * @brief Record per-tenant token counters
 *
 * Called on inference completion. Lock-free atomic increment.
 *
 * @param tenant_id     Tenant UUID (null-terminated, max 36 chars)
 * @param prompt_tokens Number of prompt tokens in this request
 * @param gen_tokens    Number of generated tokens in this request
 *
 * @note Thread-safety: thread-safe (lock-free atomic).
 */
typedef void (*DenseCoreEntRecordTenantTokensFn)(const char* tenant_id, uint64_t prompt_tokens, uint64_t gen_tokens);

/**
 * @brief Record KV Cache events
 *
 * Called on block allocation/deallocation. Lock-free atomic increment/decrement.
 *
 * @param allocated  Number of blocks allocated (positive) or freed (negative)
 * @param total      Current total block count
 * @param cow_delta  Copy-on-Write triggers in this event (0 if none)
 *
 * @note Thread-safety: thread-safe (lock-free atomic).
 */
typedef void (*DenseCoreEntRecordKVEventFn)(int64_t allocated, uint64_t total, uint64_t cow_delta);

/**
 * @brief Record MoE expert routing events
 *
 * Called on expert selection. is_hot indicates whether the expert resides
 * in the hot cache.
 *
 * @param expert_id  Expert index
 * @param is_hot     1 if expert is in hot cache, 0 if cold
 *
 * @note Thread-safety: thread-safe (lock-free atomic).
 */
typedef void (*DenseCoreEntRecordExpertRouteFn)(int expert_id, int is_hot);

/**
 * @brief Record cross-NUMA traffic events
 *
 * Called on cross-NUMA memory access.
 *
 * @param bytes  Number of bytes transferred across NUMA boundary
 *
 * @note Thread-safety: thread-safe (lock-free atomic).
 */
typedef void (*DenseCoreEntRecordCrossNumaFn)(uint64_t bytes);

// =============================================================================
// Telemetry Function Table (registered by enterprise plugin)
// =============================================================================

/**
 * @brief Telemetry function table
 *
 * Function pointer table registered by the enterprise plugin during init.
 * The DenseCore engine records telemetry events through this table.
 * NULL pointer checks ensure zero-overhead when no plugin is loaded.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    DenseCoreEntGetTelemetrySnapshotFn get_snapshot;
    DenseCoreEntRecordTenantTokensFn record_tenant_tokens;
    DenseCoreEntRecordKVEventFn record_kv_event;
    DenseCoreEntRecordExpertRouteFn record_expert_route;
    DenseCoreEntRecordCrossNumaFn record_cross_numa;
} DenseCoreEntTelemetryVTable;

/**
 * @brief Register the telemetry VTable (called by plugin during init)
 *
 * @param vtable  Function table (plugin-owned, must remain valid until shutdown)
 *
 * @note Thread-safety: NOT thread-safe. Call once during plugin init.
 */
DENSECORE_ENT_API void DenseCoreEntRegisterTelemetry(const DenseCoreEntTelemetryVTable* vtable);

/**
 * @brief Get the registered telemetry VTable (used by core to record events)
 *
 * @return Registered vtable, or NULL if no telemetry plugin
 *
 * @note Thread-safety: thread-safe after registration.
 */
DENSECORE_ENT_API const DenseCoreEntTelemetryVTable* DenseCoreEntGetTelemetryVTable(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_TELEMETRY_SINK_H
