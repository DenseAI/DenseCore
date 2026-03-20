/**
 * @file numa_routing.h
 * @brief Enterprise NUMA routing policy bridge
 *
 * Allows an optional enterprise plugin to steer DenseCore thread-pool
 * selection without hard-linking proprietary policy code into the OSS core.
 *
 * The core always retains a graceful fallback path:
 *   - no plugin / no vtable / invalid decision => existing round-robin logic
 *   - invalid or non-custom license => policy engine returns "no override"
 */

#ifndef DENSECORE_NUMA_ROUTING_H
#define DENSECORE_NUMA_ROUTING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DENSECORE_ENT_API __attribute__((visibility("default")))
#else
#define DENSECORE_ENT_API
#endif

typedef enum {
    DENSECORE_ENT_NUMA_POLICY_DISABLED = 0,
    DENSECORE_ENT_NUMA_POLICY_LOCAL = 1,
    DENSECORE_ENT_NUMA_POLICY_PREFER_EXPLICIT = 2,
    DENSECORE_ENT_NUMA_POLICY_STRICT_EXPLICIT = 3,
} DenseCoreEntNumaPolicyMode;

typedef struct {
    int32_t requested_node;  /**< Core-selected NUMA node hint (-1 if none) */
    int32_t current_node;    /**< Current thread NUMA node hint (-1 if unknown) */
    int32_t available_nodes; /**< Total thread pools / NUMA nodes available */
    int32_t reserved;
} DenseCoreEntNumaRoutingRequest;

typedef struct {
    int32_t target_node; /**< Selected NUMA node, or -1 to keep core fallback */
    int32_t policy_mode; /**< DenseCoreEntNumaPolicyMode */
    int32_t used_fallback;
    int32_t reserved;
} DenseCoreEntNumaRoutingDecision;

typedef int (*DenseCoreEntResolveNumaNodeFn)(const DenseCoreEntNumaRoutingRequest* request,
                                             DenseCoreEntNumaRoutingDecision* out_decision);
typedef int (*DenseCoreEntGetNumaPolicyFn)(void);

typedef struct {
    DenseCoreEntResolveNumaNodeFn resolve_node;
    DenseCoreEntGetNumaPolicyFn get_policy;
} DenseCoreEntNumaRoutingVTable;

DENSECORE_ENT_API void DenseCoreEntRegisterNumaRouting(const DenseCoreEntNumaRoutingVTable* vtable);
DENSECORE_ENT_API const DenseCoreEntNumaRoutingVTable* DenseCoreEntGetNumaRoutingVTable(void);

#ifdef __cplusplus
}
#endif

#undef DENSECORE_ENT_API

#endif  // DENSECORE_NUMA_ROUTING_H
