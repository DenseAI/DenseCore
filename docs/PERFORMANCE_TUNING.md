# Performance Tuning

DenseCore performance depends on model topology, quantization, prompt length,
decode length, concurrency, CPU ISA, memory capacity, and graph admission. There
is no universal environment-variable preset.

## Start With a Qualified Build

Use a release build and a target profile that matches the deployment CPU. See
[HARDWARE_SUPPORT.md](HARDWARE_SUPPORT.md). Record the compiler, CMake cache,
binary hash, model hash, and CPU topology before comparing results.

Do not mix binaries built for C4 x86 and C4A Arm. A shared model disk can contain
both checkouts, but executables are architecture-specific.

## Threading

`THREADS` controls the model-load/server input, while
`DENSECORE_ENGINE_THREADS` is the explicit engine compute override. Zero lets the
runtime choose.

Choose compute threads from the target host's physical cores and validate the
setting with the intended model and workload. There is no universal thread count.

Measure:

- physical cores versus SMT threads
- one NUMA node versus cross-node execution
- prefill and decode separately
- concurrency 1/2/4/8 rather than only a single request

## Scheduler and Concurrency

The default active-sequence bound is four and concurrent prefill defaults to one.
Those values protect graph memory and long-prompt stability; they are not a claim
that four requests always improve throughput.

Primary overrides:

| Variable | Use |
| --- | --- |
| `DENSECORE_MAX_NUM_SEQS` | bound active sequences |
| `DENSECORE_MAX_SEQ_LEN` | cap context length when required |
| `DENSECORE_SERVER_INFLIGHT` | bound server admission |
| `DENSECORE_GO_WORKERS` | Go worker count |

Long prefill can serialize behind `max_prefill_seqs=1`. Increasing parallel
prefill raises graph-memory pressure and has regressed measured Qwen workloads, so
promote it only after long-QA and memory validation.

## KV and Graph Memory

Prefer the runtime-derived KV budget. `DENSECORE_KV_TARGET_MB` is a bounded
deployment override, not a default sizing formula. Account for:

- model weights and repacked aliases
- KV dtype and context length
- active sequences
- graph/context pools
- temporary quantization and callback buffers
- host safety margin

On Linux, DenseCore reserves the derived KV capacity as demand-paged anonymous
memory. The configured context/concurrency ceiling therefore does not become
resident RAM at model load; pages are committed as requests write KV blocks.
The runtime still sizes the ceiling from loaded-model metadata, KV dtype,
context, concurrency, and current host/container availability.

Gemma4 demonstrates why residency matters: removing an unused expert alias restored
enough memory for a long prompt to avoid forced chunking. Memory savings are a
performance change only after the same request passes QA and is remeasured.

## NUMA

On multi-socket hosts, inspect topology first:

```bash
lscpu
numactl --hardware
numastat -p "$(pidof densecore-server)"
```

DenseCore contains expert-to-node profiling, NUMA-aware forward planning, and
per-node thread pools. Treat each host, model, concurrency level, and end-to-end
workload as requiring its own validation.

Do not publish estimated migration percentages or synthetic log examples as
measured gains.

## Huge Pages

On Linux the KV cache always requests transparent huge pages; this is a qualified
default and there is no DenseCore switch for it. If a host must not use them,
change the kernel policy:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
grep -i huge /proc/meminfo
```

Do not require system-wide `always` mode without measuring the target service and
co-located workloads.

## Prompt Cache

Repeated prefixes can reduce prefill work only when requests return to the same
process and the cache identity is safe. In Kubernetes, configure an explicit
affinity key; replica round-robin destroys locality. See
[prompt_cache.md](prompt_cache.md).

Hybrid SSM and sliding/shared-KV models have stricter restore requirements. A
cache decision must fail closed when state cannot be restored exactly.

## Benchmark Loop

For each tuning change:

1. run focused unit tests for the touched policy or kernel
2. run strict short and long inference QA
3. verify target fast-path counters and zero forbidden fallback
4. run one matched server pair to reject obvious regressions
5. only after it passes, run five-cycle ABBA and report p50/spread
6. retain losing experiments as notes or Git history, not as public defaults

Keep the environment, model artifact, prompts, and quality checks fixed for a
meaningful comparison.

## Environment Variable Reference

Everything DenseCore's C++ core reads at runtime is listed here. Two rules apply:

- **This list is the whole supported surface.** A `DENSECORE_*` name that does not
  appear here is not configuration. Most such names were removed in favour of a
  single qualified path; setting them does nothing.
- **Booleans have one spelling.** `1/true/yes/on` enable, `0/false/no/off` disable,
  in any case. An unrecognised value keeps the default rather than flipping it.

Names retired in favour of a replacement are rejected at startup with an error
naming the successor, so a stale deployment fails loudly instead of silently
running an unintended configuration.

### Model and capacity

| Variable | Default | Purpose |
| --- | --- | --- |
| `DENSECORE_MAX_SEQ_LEN` | model `n_ctx` | Context length ceiling. |
| `DENSECORE_MAX_NUM_SEQS` | scheduler config | Concurrent sequence ceiling. |
| `DENSECORE_KV_TARGET_MB` | derived | KV cache budget. |
| `DENSECORE_KV_AVAILABLE_MB_HINT` | probed | Override host-memory probe for KV sizing. |
| `DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT` | probed | Same, for graph pool sizing. |
| `DENSECORE_SLIDING_WINDOW_SIZE` | `-1` (off) | KV retention window. |
| `DENSECORE_SINK_TOKENS` | `0` | Attention-sink tokens kept outside the window. |
| `DENSECORE_MOE_DEQUANT_CACHE_MB` | `512` | Dequantised-expert cache cap. |
| `DENSECORE_PREFILL_GRAPH_CACHE_MAX_MB` | derived | Prefill graph cache cap. |
| `DENSECORE_PREFILL_GRAPH_CACHE_LRU` | derived | Prefill graph cache entries. |
| `DENSECORE_PREPOPULATE_WEIGHTS` | auto | Force/disable up-front weight fault-in. |

### Execution policy

| Variable | Default | Purpose |
| --- | --- | --- |
| `DENSECORE_PREFERRED_DEVICE` | `cpu` | Backend selection. |
| `DENSECORE_CALLBACK_MODE` | `async` | `async` or `direct` token callbacks. |
| `DENSECORE_ALLOW_DECODE_THREADS_OVER_BASE` | `false` | Let decode exceed the configured thread budget. |
| `DENSECORE_DECODE_GRAPH_CACHE` | `true` | Emergency disable for decode graph reuse. |
| `DENSECORE_KV_USE_BULK_PATH` | `true` | Emergency disable for the bulk KV slot path. |
| `DENSECORE_PAGED_DECODE_MIN_CONTEXT` | ISA-derived | Paged decode admission threshold. |
| `DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT` | ISA-derived | Same, batched decode. |
| `DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED` | `true` | Allow quantised KV on the paged decode kernel. |
| `DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8` | `true` | Older name; used as the default for the row above when that is unset. |
| `DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE` | auto | Paged decode head tiling. |
| `DENSECORE_HYBRID_SSM_QKV_FORCE_GGML` | `false` | Fall back to generic GGML for hybrid-SSM QKV. |
| `DENSECORE_MOE_STRICT` | `false` | Fail closed instead of falling back on MoE. Use in CI. |

### Prompt and decoding behaviour

| Variable | Default | Purpose |
| --- | --- | --- |
| `DENSECORE_AUTO_CHAT_TEMPLATE` | `true` | Apply the model's chat template automatically. |
| `DENSECORE_QWEN3_ENABLE_THINKING` | model default | Thinking-mode toggle, Qwen3. |
| `DENSECORE_QWEN35_ENABLE_THINKING` | model default | Thinking-mode toggle, Qwen3.5. |
| `DENSECORE_QWEN36_ENABLE_THINKING` | model default | Thinking-mode toggle, Qwen3.6 and beta Qwen3.8 text-only. |
| `DENSECORE_GEMMA4_ENABLE_THINKING` | model default | Thinking-mode toggle for Gemma4. |
| `DENSECORE_QWEN36_PRIME_NO_THINKING` | `false` | Prime Qwen3.6 into non-thinking mode. |
| `DENSECORE_SUPPRESS_REASONING_TAGS` | `true` | Strip reasoning tags from streamed output. |

### NUMA (multi-socket hosts)

| Variable | Default | Purpose |
| --- | --- | --- |
| `DENSECORE_NUMA_WEIGHTS` | default loader | Select NUMA weight placement; `round_robin` partitions expert weights. |
| `DENSECORE_NUMA_EXPERT_PARTITION` | on with NUMA loader | Partition MoE experts across nodes when supported. |
| `DENSECORE_NUMA_HUGEPAGES` | off | Huge pages for NUMA-local allocations. |
| `DENSECORE_MOE_REBALANCE_INTERVAL_MS` | `5000` | Expert rebalance cadence in milliseconds. |
| `DENSECORE_MOE_REBALANCE_TOP_K` | `4` | Experts considered per rebalance. |

### Experimental — not qualified

Off by default and not covered by the release gate. Do not enable in production
without your own validation on your topology.

| Variable | Purpose |
| --- | --- |
| `DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY` | Sticky MoE expert-to-node routing. Pending 2-socket validation. |
| `DENSECORE_MOE_ENABLE_PAGE_MIGRATION` | Migrate expert pages toward the consuming node. |
| `DENSECORE_APPLE_HYBRID` | Apple CPU/ANE hybrid scheduling. |
| `DENSECORE_ANE_LAYER_PREFIX` | Layer prefix routed to the Apple Neural Engine. |

### Apple Silicon (macOS builds only)

These are compiled only on macOS and cover the AMX, Metal, and ANE paths, which
`docs/HARDWARE_SUPPORT.md` does not list as gate-enforced. Treat them as
experimental and measure on your own hardware.

`DENSECORE_APPLE_AMX_INT4` (`true`), `DENSECORE_APPLE_AMX_INT4_TILE_N` (`128`),
`DENSECORE_APPLE_AMX_INT4_MIN_K` (`128`), `DENSECORE_APPLE_AMX_INT4_MIN_SLICE_N`
(`64`), `DENSECORE_APPLE_HYBRID`, `DENSECORE_APPLE_HYBRID_GPU_MIN_BATCH`,
`DENSECORE_APPLE_HYBRID_GPU_MIN_SEQ`, `DENSECORE_METAL_INT4_SIMDGROUP_GEMM`,
`DENSECORE_METAL_SMALL_BATCH_GEMV_MAX_COLS`,
`DENSECORE_METAL_WRAPPED_BUFFER_CACHE`, `DENSECORE_ANE_LAYER_PREFIX`,
`DENSECORE_ANE_INT4_CACHE_FP32`, `DENSECORE_ANE_INT4_REPACK_FP16`,
`DENSECORE_ANE_PREFER_REPACKED_MODELS`, `DENSECORE_ANE_SPEC_PRELOAD_RATIO`,
`DENSECORE_ANE_SYNC_EACH_FALLBACK`, `DENSECORE_ANE_THERMAL_EVICT_HYSTERESIS_MS`.

### Benchmark harness

`DENSECORE_BENCH_MODE` and `DENSECORE_BENCH_RESPECT_THREADS` change thread policy
for measurement runs. A benchmark result produced with either of these set is not
comparable to a default-configuration serving run; say so when reporting.

### Diagnostics

`DENSECORE_DEBUG_*`, `DENSECORE_LOG_*`, `DENSECORE_VERBOSE_*`, `DENSECORE_CHECK_*`,
`DENSECORE_PROFILE_*`, and `DENSECORE_*_TRACE_*` are developer diagnostics. They are
**compiled out of release builds** and cost nothing on hot paths. To use them
against a release build, configure with:

```bash
cmake -S core -B build -DCMAKE_BUILD_TYPE=Release -DDENSECORE_ENABLE_DEBUG_ENV=ON
```

Debug builds and the test binaries enable them automatically. These names are not
a stable interface and may change between releases.
