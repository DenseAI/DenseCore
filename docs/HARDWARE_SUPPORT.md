# Hardware Support

DenseCore is CPU-first, not CPU-only. The full-model LLM baseline is maintained on
x86-64 and Arm64; optional accelerator and generic HAL paths have narrower
qualification.

## Maturity Matrix

| Path | Status | Qualification boundary |
| --- | --- | --- |
| x86-64 CPU | maintained | main GGUF LLM path; model/quantization still requires QA |
| Linux Arm64 CPU | maintained | correctness-first generic build plus qualified target builds |
| Google C4A / Axion | maintained target profile | Armv9, SVE2, I8MM, BF16, dot-product; current Qwen decode target still misses llama-server |
| Apple CPU | maintained baseline | Arm CPU path; validate wheel and model behavior on each macOS target |
| Apple Metal | beta | selected kernels and shaders; not a blanket full-model speed claim |
| Apple ANE/CoreML | experimental | selected conditional paths, not a general LLM backend |
| Qualcomm QNN | experimental | backend surface and tests exist; product-specific qualification required |
| Generic OperationGraph/HAL | partial | operation-level admission and fallback, separate from main GGML LLM graph |
| External plugin | interface only in OSS | plugin loading boundary exists; plugin capability is external |

## x86-64

DenseCore and its GGML dependency use runtime/build admission for AVX2, AVX-512,
VNNI, AMX, and model-specific repacked layouts. Google Cloud C4 is the current
flagship benchmark platform for Qwen3.5 and Qwen3.6.

Build the maintained C4 performance preset on the target host:

```bash
cmake --preset c4-perf -S core
cmake --build build-c4-perf -j
```

Native binaries are not portable across older CPUs. Use `DENSECORE_PORTABLE=ON`
for a heterogeneous fleet and benchmark the resulting performance separately.

## Linux Arm64

Generic Linux Arm64 defaults to a correctness-first profile. It disables optional
third-party fast-kernel features unless explicitly selected. This prevents an
unqualified kernel from silently becoming the production path.

### Google C4A

Use the maintained performance preset rather than a generic Arm build:

```bash
cmake --preset c4a-perf -S core
cmake --build build-c4a-perf -j
```

The preset selects `neoverse-v2`, Armv9 SVE2/I8MM/BF16/dotprod, GGML repack, and
the current maintained fast-kernel policy while leaving KleidiAI and llamafile off.
It disables the generic correctness-first profile, so the produced binary must pass
the full C4A QA sweep before deployment.

Other Arm profiles are defined in `core/cmake/arm_profiles.cmake`. Cross-compiles
must set an explicit `DENSECORE_ARM_MCPU` or `DENSECORE_ARM_MARCH`; do not assume
host-native detection in a cross toolchain.

## Apple Silicon

The repository contains Objective-C++ backends, Metal shaders, an ANE/CoreML
surface, and a hybrid selector. Those components are real but unevenly qualified.
The documented public Python constructor does not expose a universal `device` or
`backend_name` selector, so deployment must use the supported build/runtime path
rather than copied examples from older documents.

Build on macOS:

```bash
xcode-select --install
cmake -S core -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the model QA suite on the produced library before treating Metal or ANE as a
production backend.

## NUMA

DenseCore includes NUMA allocation, expert profiling, forward planning, and
per-NUMA thread-pool support. A scoped two-socket Qwen3.6 decode checkpoint
measured a 26.04% isolated-decode advantage over `llama-server --numa distribute`
with two scored repetitions. It is not a general NUMA, end-to-end latency, or
all-model claim; see [NUMA_STICKY_ROUTING.md](NUMA_STICKY_ROUTING.md) for the
exact contract and limitations.

### Build requirement

NUMA code paths are compiled only when both `libnuma` and `hwloc` are present at
configure time. Install `libnuma-dev` and `libhwloc-dev`, then confirm the
configure log contains:

```
-- NUMA support enabled: hwloc <version>, numa <version>
```

If it instead says `NUMA libs not found` or `Partial NUMA support`, every setting
below is inert regardless of the environment variables you set.

### Enabling MoE sticky expert routing

Expert **partitioning** is off by default; sticky *dispatch* is not. `InitEngine`
passes `numa_node_id = -1`, which loads the model through the plain loader, so no
expert partitioning happens and first-touch puts every expert weight on the
loader thread's node. Placement detection then reports that truthfully — every
expert on one node — which is a valid placement, so sticky routing arms and pins
**all** MoE compute to that single node. On a multi-node host this is reported as
`native_moe_numa_sticky_state=enabled_degenerate`: routing works, but the
remaining nodes contribute nothing to MoE execution.

Setting `DENSECORE_NUMA_WEIGHTS` before starting the server is what spreads the
weights, and therefore the compute, across nodes:

| Value | Effect |
|---|---|
| `round_robin` | Partitions MoE expert weights across nodes round-robin. This is the mode sticky routing is designed around. |
| `auto` / `interleaved` | Interleaves weights across all nodes. |
| `pinned` or a node number | Pins all weights to one node. |

Related settings (defaults in parentheses):

| Variable | Default |
|---|---|
| `DENSECORE_NUMA_EXPERT_PARTITION` | on |
| `DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY` | `0` (diagnostic disable only) |
| `DENSECORE_MOE_ENABLE_PAGE_MIGRATION` | off |
| `DENSECORE_MOE_REBALANCE_INTERVAL_MS` | 5000 |
| `DENSECORE_MOE_REBALANCE_TOP_K` | 4 |
| `DENSECORE_NUMA_HUGEPAGES` | off |

### Confirming it is actually active

Enabling the environment variable is not proof the router ran. Two independent
consumers of the expert→node map exist, and they report differently:

- The small-decode dispatcher logs `[NUMA] MoE sticky decode dispatch active`.
- The native MoE path — which the large MoE models actually use — logs
  `[NUMA] Native MoE sticky routing active` and reports per-dispatch counters in
  the decode summary line:

```
native_moe_numa_sticky_state=enabled
native_moe_numa_sticky_dispatch_ops=<n>   # expert runs pinned to their node
native_moe_numa_legacy_dispatch_ops=<n>   # expert runs left unpinned
native_moe_numa_node_dispatch_ops=0:<n>,1:<n>
```

`native_moe_numa_sticky_state` reports which of these applies:

| State | Meaning |
|---|---|
| `enabled` | Experts span more than one node; MoE compute spreads across them. |
| `enabled_degenerate` | Routing is armed, but every expert sits on **one** node of several, so MoE compute stays on that node. This is the default-load layout — set `DENSECORE_NUMA_WEIGHTS=round_robin` to partition. |
| `single_node` | Host has one NUMA node; nothing to route across. |
| `placement_unavailable` | Expert→node map could not be read. |
| `placement_invalid` | At least one expert's node is unknown or out of range; those dispatches take the legacy path. |

The first two both arm dispatch, so `enabled_degenerate` still shows
`legacy_dispatch_ops=0`. Read `node_dispatch_ops` to see where the work went: a
single `0:<n>` entry means everything ran on one node.

## Qualification Rule

A backend is usable only when all of the following hold for the target model:

1. build and runtime admission select the intended path
2. deterministic short and long output-quality gates pass
3. no slower fallback masks a failed target path
4. memory, concurrency, and shutdown behavior pass on the deployment host
5. any performance statement is measured through the real serving path
