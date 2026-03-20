# NUMA and Multi-Socket Optimization Guide

DenseCore supports advanced NUMA-aware memory allocation and thread affinity to maximize performance on multi-socket servers. This guide covers all low-level optimizations for production environments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [NUMA Node Binding](#numa-node-binding)
- [Thread Pinning Policies](#thread-pinning-policies)
- [Memory Allocation](#memory-allocation)
- [System Diagnostics](#system-diagnostics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Modern multi-socket servers have multiple NUMA (Non-Uniform Memory Access) nodes. Memory access latency varies significantly depending on whether the memory is "local" (attached to the same socket as the CPU) or "remote" (attached to another socket).

**DenseCore's NUMA optimizations:**
- **Memory Binding**: Allocate KV cache on a specific NUMA node
- **Thread Pinning**: Pin compute threads to physical cores on the same node
- **Huge Pages**: Use 2MB huge pages to reduce TLB pressure
- **Diagnostics**: Runtime verification of NUMA placement

### Performance Impact

| Configuration | Latency | Throughput |
|---------------|---------|------------|
| No NUMA awareness | Baseline | Baseline |
| Memory + Thread binding | -30% | +40% |
| + Huge pages | -35% | +50% |

---

## Quick Start

### C API

```c
#include <densecore.h>

// Initialize with NUMA node 0 and SCATTER pinning policy
DenseCoreHandle engine = InitEngine(
    "model.gguf",  // model path
    NULL,          // reserved
    8,             // threads
    0,             // numa_node_id: bind to node 0
    0              // pinning_policy: 0=SCATTER (default)
);
```

### Python SDK

```python
import densecore

# NUMA-aware initialization
engine = densecore.Engine(
    "model.gguf",
    numa_node_id=0,      # Bind to NUMA node 0
    pinning_policy=0     # SCATTER for latency
)
```

---

## NUMA Node Binding

### `numa_node_id` Parameter

Controls where KV cache memory is allocated and where threads are pinned.

| Value | Behavior |
|-------|----------|
| `-1` | Auto-detect (uses calling thread's local node) |
| `0+` | Explicit node binding |

### Graceful Fallback

If the requested NUMA node runs out of memory, DenseCore gracefully falls back:

1. **Strict allocation** on requested node (`numa_alloc_onnode`)
2. **Local allocation** on calling thread's node (`numa_alloc_local`)
3. **Interleaved** across all nodes (`numa_alloc_interleaved`)
4. **Standard malloc** as last resort

You'll see log messages indicating allocation status:

```
[KVCache] Allocated 256 MB on NUMA node 0 (OPTIMAL)
```

or

```
[NumaAllocator] WARNING: NUMA node 0 OOM for 256 MB, falling back to local allocation
[NumaAllocator] Fallback allocation on node 1 succeeded
```

---

## Thread Pinning Policies

### `pinning_policy` Parameter

Controls how compute threads are distributed across CPU cores.

| Value | Policy | Use Case |
|-------|--------|----------|
| `0` | **SCATTER** | Distribute across physical cores for max L3 cache and memory bandwidth. Best for **latency-sensitive single-user** workloads. |
| `1` | **COMPACT** | Pack threads on adjacent cores sharing L2 cache. Leaves cores for other processes. Best for **throughput/batch** workloads. |

### SCATTER (Default)

```
Core 0  Core 2  Core 4  Core 6
  ↓       ↓       ↓       ↓
Thread0 Thread1 Thread2 Thread3
```

- Maximizes memory bandwidth
- Each thread has dedicated L3 slice
- Ideal for single inference stream

### COMPACT

```
Core 0  Core 1  Core 2  Core 3
  ↓       ↓       ↓       ↓
Thread0 Thread1 Thread2 Thread3
```

- Shares L2 cache between threads
- Leaves remaining cores for other processes
- Ideal for multi-tenant batch processing

---

## Memory Allocation

### Huge Pages

DenseCore automatically uses 2MB huge pages when available:

```bash
# Configure system for huge pages (requires root)
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Verify
cat /proc/meminfo | grep Huge
```

### Allocation Functions

| Function | Description |
|----------|-------------|
| `NumaAllocator::AllocatePreferred()` | Allocates with graceful NUMA fallback |
| `NumaAllocator::AllocateHugePagesOnNode()` | Allocates 2MB huge pages on node |
| `NumaAllocator::TouchPages()` | First-touch policy enforcement |

---

## System Diagnostics

DenseCore includes production-ready diagnostics to verify NUMA placement.

### Automatic Report

When NUMA allocation is enabled, the engine prints a diagnostic report:

```
[System Check] KV Cache: 256 MB allocated on Node 0 (Match), HugePages: Active (128 pages)
```

### Warning Messages

If configuration doesn't match reality:

```
[System Check] WARNING: KV Cache requested on Node 0, but resident on Node 1 (Mismatch). Performance may be degraded.
[System Check] WARNING: KV Cache HugePages inactive. Consider: echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

### API Reference

```cpp
#include "numa_allocator.h"

// Verify allocation placement
densecore::MemoryDiagnostics::DiagResult result =
    densecore::MemoryDiagnostics::PrintSystemTopologyReport(
        ptr,           // Memory pointer
        size,          // Size in bytes
        requested_node, // Expected NUMA node
        "KV Cache"     // Label for logs
    );

// Access diagnostic results
if (!result.nodes_match) {
    // Handle mismatch
}
if (!result.huge_pages_active) {
    // HugePages not configured
}
```

---

## Best Practices

### 1. For Latency-Sensitive Workloads

```c
// Single inference stream: maximize cache locality
InitEngine("model.gguf", NULL, num_cores, 0, 0);  // SCATTER
```

### 2. For Throughput Workloads

```c
// Multiple concurrent requests: leave room for I/O
InitEngine("model.gguf", NULL, num_cores/2, 0, 1);  // COMPACT
```

### 3. Multi-Socket Servers

```c
// Pin to specific socket for predictable latency
InitEngine("model.gguf", NULL, cores_per_node, socket_id, 0);
```

### 4. Verifying Configuration

```bash
# Check NUMA topology
numactl --hardware

# Check thread affinity at runtime
ps -eo pid,comm,psr | grep densecore

# Check memory placement
numastat -p <pid>
```

---

## Troubleshooting

### "NUMA node OOM" Warning

**Cause**: Insufficient memory on specified node.

**Solutions**:
1. Reduce `max_seq_len` to decrease KV cache size
2. Use a different NUMA node with more free memory
3. Accept fallback (performance may vary)

### HugePages Inactive

**Cause**: System huge pages not configured.

**Solution**:
```bash
# Allocate 1024 huge pages (2GB total)
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Make persistent (add to /etc/sysctl.conf)
vm.nr_hugepages = 1024
```

### Node Mismatch Warning

**Cause**: Memory migrated to different node (e.g., by `numabalanced`).

**Solutions**:
1. Disable automatic NUMA balancing:
   ```bash
   echo 0 | sudo tee /proc/sys/kernel/numa_balancing
   ```
2. Use `numcmd --membind=N` to enforce binding

### Thread Pinning Not Working

**Cause**: Insufficient permissions for `sched_setaffinity`.

**Solution**:
```bash
# Run with elevated capabilities
setcap cap_sys_nice+ep ./densecore_server

# Or run as root (not recommended for production)
```

---

## Related Documentation

- [Architecture Guide](ARCHITECTURE.md) - Internal design details
- [API Reference](API_REFERENCE.md) - Full C/Python API docs
- [Deployment Guide](DEPLOYMENT.md) - Docker/Kubernetes setup
- [Benchmarks](BENCHMARKS.md) - Performance methodology

---

*Last updated: December 2024*
