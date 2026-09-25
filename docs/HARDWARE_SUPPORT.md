# Hardware Support

DenseCore is CPU-first, not CPU-only. The full-model LLM baseline is maintained on
x86-64 and Arm64; optional accelerator and generic HAL paths have narrower
qualification.

Platform support does not imply release-artifact qualification. The v0.1
release candidate has local qualification for `linux/amd64` binaries and
containers only. Linux Arm64 and the C4A profile remain source-build and
engineering-validation surfaces until a native Arm runner completes the same
real-model HTTP and shutdown gate.

## Maturity Matrix

| Path | Status | Qualification boundary |
| --- | --- | --- |
| x86-64 CPU | maintained, v0.1.0 image qualified on linux/amd64 | main GGUF LLM path; model/quantization still requires QA |
| Linux Arm64 CPU | maintained source support | correctness-first generic and target builds; no v0.1.0 artifact qualification |
| Google C4A / Axion | engineering-validated target profile | Armv9, SVE2, I8MM, BF16, dot-product; no v0.1.0 artifact qualification |
| Apple CPU | experimental | source surface only; no release artifacts or CI qualification |
| Apple Metal | experimental | selected kernels and shaders; no blanket full-model speed claim or CI qualification |
| Apple ANE/CoreML | experimental | selected conditional paths, not a general LLM backend or release target |
| Qualcomm QNN | experimental | backend surface and tests exist; product-specific qualification required |
| Generic OperationGraph/HAL | partial | operation-level admission and fallback, separate from main GGML LLM graph |
| External plugin | interface only in OSS | plugin loading boundary exists; plugin capability is external |

The QNN scaffold is not registered by default. Set
`DENSECORE_QNN_EXPERIMENTAL=1` only for explicit development or qualification
runs; this opt-in does not promote the backend beyond experimental status.

## x86-64

DenseCore and its GGML dependency use runtime/build admission for AVX2, AVX-512,
VNNI, AMX, and model-specific repacked layouts. Google Cloud C4 is a supported
performance profile for Qwen3.5 and Qwen3.6.

Build the maintained C4 performance preset on the target host:

```bash
cmake --preset c4-perf -S core
cmake --build build-c4-perf -j
```

Native binaries are not portable across older CPUs. Use `DENSECORE_PORTABLE=ON`
or `DENSECORE_CPU_TARGET=portable` for a heterogeneous fleet and benchmark the
resulting performance separately. Use `DENSECORE_CPU_TARGET=amd64-v3` for the
deterministic optimized release profile instead of distributing a C4-native
binary.

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
surface, and a hybrid selector. These are exploratory source surfaces, not current
release targets: DenseCore does not publish macOS artifacts or run macOS CI.
The public Python constructor does not expose a universal `device` or
`backend_name` selector. Do not treat a local macOS build as a qualified Metal or
ANE deployment without an owned platform test matrix and model-quality evidence.

## NUMA

DenseCore includes NUMA allocation, expert profiling, forward planning, and
per-NUMA thread-pool support. Its effectiveness depends on the model, memory
placement, and host topology, so validate it on the intended workload.

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

Sticky MoE dispatch is experimental and **off by default**. It requires
`DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY=1`, NUMA support in the build, and a
qualified model path. The default model loader does not request NUMA weight
placement. Enabling sticky dispatch without checking placement can pin all
expert work to one node on a multi-node host; the runtime reports this as
`native_moe_numa_sticky_state=enabled_degenerate`.

Set `DENSECORE_NUMA_WEIGHTS` before starting the server to select weight
placement. Check actual placement and dispatch counters before inferring that
compute is spread across nodes:

| Value | Effect |
|---|---|
| `round_robin` | Partitions MoE expert weights across nodes round-robin. This is the mode sticky routing is designed around. |
| `auto` / `interleaved` | Interleaves weights across all nodes. |
| `pinned` or a node number | Pins all weights to one node. |

Related settings (defaults in parentheses):

| Variable | Default |
|---|---|
| `DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY` | off |
| `DENSECORE_NUMA_EXPERT_PARTITION` | on when the NUMA loader is selected |
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

When sticky dispatch is enabled, `native_moe_numa_sticky_state` reports which
of these applies. Otherwise it reports a disabled state:

| State | Meaning |
|---|---|
| `enabled` | Experts span more than one node; MoE compute spreads across them. |
| `enabled_degenerate` | Routing is armed, but every expert sits on **one** node of several, so MoE compute stays on that node. Select and verify a partitioned placement before using sticky routing. |
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
