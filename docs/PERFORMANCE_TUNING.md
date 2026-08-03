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

For the current 16-vCPU C4/C4A benchmark lane, 16 compute threads were the best
validated control. Reducing C4A decode threads regressed Qwen, Gemma, and LFM2 in
the measured sweeps. This is a workload result, not a rule for every CPU.

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
per-node thread pools. Keep the feature classified as unqualified until a dedicated
idle multi-socket host shows lower remote access and higher end-to-end throughput
than both OS interleave and the comparison runtime.

Do not publish estimated migration percentages or synthetic log examples as
measured gains.

## Huge Pages

Transparent or explicit huge pages can help some large memory-bound workloads but
can also change startup latency and host behavior. Treat them as an A/B:

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

See [BENCHMARKS.md](BENCHMARKS.md) for the comparison contract and current evidence.
