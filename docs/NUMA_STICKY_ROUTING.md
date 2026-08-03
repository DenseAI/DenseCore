# NUMA Sticky Routing Evidence

This document is the public summary for DenseCore's MoE NUMA sticky-routing
experiment. It separates the measured decode result from the claims it does not
support. The maintained rerun harness lives in
[`benchmarks/numa`](../benchmarks/numa/).

## Scope

The qualified case is deliberately narrow:

- model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- host: Google Cloud `n2-standard-96`, two sockets and two NUMA nodes
- runtime: 16 compute threads through the Go `/v1/chat/completions` server
- cache: disabled for both engines
- comparison: DenseCore sticky routing enabled versus `llama-server --numa distribute`
- metric: external non-stream decode throughput after an identical one-token
  prefill probe

The experiment measured a real remote-memory penalty before comparing engines:
remote bandwidth was up to 53.2% lower and pointer-chase latency up to 50.3%
higher than local access. A virtual NUMA distance table alone is not accepted as
qualification.

## Measured Result

| Engine | External isolated decode median | Two-run range | Output gate |
| --- | ---: | ---: | --- |
| DenseCore, sticky routing | **20.272 tok/s** | 20.271-20.273 | pass |
| llama.cpp, `--numa distribute` | 16.084 tok/s | 16.018-16.149 | pass |

For this exact workload, DenseCore measured **26.04% faster isolated decode**
than the selected llama.cpp NUMA mode. The same DenseCore binary and model also
measured a **11.88%** steady-decode improvement with sticky routing enabled
versus its diagnostic sticky-off control (20.3432 versus 18.1827 tok/s).

The result is a final engineering checkpoint, not a blanket product claim. It
has two scored repetitions per engine, not an ABBA median series or a workload
distribution. It does not establish concurrent throughput, tail latency, all
models, all two-socket hosts, or a general DenseCore advantage over llama.cpp.

## Fairness and Quality Gates

Both engines used the same GGUF, prompt, model option, temperature, thread
count, completion target, and OpenAI-compatible non-stream client. The client
computes the decode window as:

```text
decode tok/s = (completion_tokens - 1) /
               (time(max_tokens=128) - time(max_tokens=1))
```

The run rejects missing token usage, empty output, prompt echo, repetition,
thinking leakage, short output, cache reuse, an unavailable NUMA build, or a
failed target fast path. The sticky control uses
`DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY`; that variable is diagnostic only and
is not a maintained performance selector.

## What Did Not Win

Page migration executed successfully, preserved the sticky-on output, and was
kept disabled by default because it reduced steady decode by 2.05% in this
workload. A short whole-request API metric also remained worse for DenseCore
(8.196 versus 13.441 tok/s), where prefill and time-to-first-token dominated the
short response. This does not contradict the isolated decode result because the
time windows differ.

## Implementation Boundary

The retained implementation is stage-selective. Model loading registers
expert-to-node placement, the native MoE path selects the corresponding
node-local work, and unknown placement falls back to the legacy dispatch path.
It does not guess a NUMA node. The ownership chain is:

```text
model placement
  -> core/src/moe/profiler.cpp
  -> core/src/backend/cpu_backend.cpp
  -> core/src/backend/cpu_backend_moe_registry.cpp
  -> core/src/backend/cpu_backend_moe_ops.cpp
  -> core/src/runtime/inference_graph_support.inl
```

`DENSECORE_NUMA_WEIGHTS=round_robin` is required to partition MoE weights on a
multi-node host. Without it, ordinary first-touch loading can place all experts
on one node; routing remains mechanically enabled but is reported as
`enabled_degenerate` and does not distribute MoE work.

## Reproducing the Checkpoint

Install `libnuma-dev`, `libhwloc-dev`, `numactl`, `jq`, Python, and curl. Build
with NUMA enabled, confirm the configure log says `NUMA support enabled`, provide
the model hash, and run:

```bash
EXPECTED_GCE_MACHINE_TYPE=n2-standard-96 \
MODEL_PATH=/data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
MODEL_SHA256=<verified-sha256> THREADS=16 \
bash benchmarks/numa/fair_decode_claim_e2e.sh
```

The script performs the physical-NUMA preflight before starting either server.
For a promotion-grade public comparison, freeze a release candidate, rerun
strict QA on that binary, and collect five-cycle ABBA medians with spread,
concurrency, TTFT, memory, and cost reported separately.
