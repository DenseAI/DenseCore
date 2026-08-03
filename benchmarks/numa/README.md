# Tier-0 NUMA validation (sticky routing vs llama-server)

Cheap, falsifiable check of whether DenseCore's NUMA-aware MoE (sticky routing)
is a *real* differentiator — on a 2-socket box, before spending on big rigs.

## What it answers
| Gate | Comparison | Pass |
|---|---|---|
| A mechanism | `numastat` remote% B1/best-llama vs N1 | N1 remote% lower |
| B self-gain | N1 (sticky) vs B1 (off + `numactl --interleave`) decode tok/s | > ~1.05x |
| C **differentiator** | N1 vs **best of** L0..L3 (all realistic llama NUMA modes) | > 1.0x |
| D quality | greedy output N1 == B0 (same engine), **non-empty** | identical |

If C ≤ 1.0x → "OS interleave / llama is enough" → differentiator is weak → stop. Honest by design.

## Latest physical 2-socket result (2026-08-03)

`n2-standard-96` exposed 48 physical cores across 2 sockets/NUMA nodes. The final
cross-engine boot measured up to 53.2% remote-bandwidth loss and 50.3%
remote-latency increase (the preceding self-gain boot measured 54.7% / 55.5%).
The raw two-turn QA runs are archived outside the source tree; the public scope
and reproduction contract are in
[`docs/NUMA_STICKY_ROUTING.md`](../../docs/NUMA_STICKY_ROUTING.md).

The final cross-engine decode claim uses the same external non-stream client and
the same `max_tokens=1` prefill-probe subtraction for both engines. It runs exactly
two scored repetitions per engine with the same prompt, GGUF, 16 threads,
temperature 0, no-thinking template option, and caches disabled:

| Engine | External isolated decode median | 2-run range | Output gate |
|---|---:|---:|---|
| DenseCore sticky ON | **20.272 tok/s** | 20.271-20.273 | pass; hashes identical |
| llama `--numa distribute` | 16.084 tok/s | 16.018-16.149 | pass; hashes identical |

DenseCore is **26.04% faster**, so the scoped `2-socket NUMA-ON decode >= llama x
1.10` gate passes. Raw results are intentionally not tracked with the source
tree; rerun this harness to reproduce them.

| Condition | Engine-native decode | Identical API E2E | QA |
|---|---:|---:|---|
| DenseCore sticky OFF | 18.1827 tok/s | 9.002 tok/s | pass: 2/2 |
| DenseCore sticky ON | 20.3432 tok/s | 8.196 tok/s | pass: 2/2 |
| DenseCore + migration | 19.9261 tok/s | 8.173 tok/s | pass: 2/2, ON parity |
| llama `--numa distribute` | 16.6261 tok/s | 13.441 tok/s | pass: 2/2 |

Stage-selective sticky routing improved same-engine steady decode by 11.88%; both
individual turns improved by more than 10%. Cross-engine native timing indicators
put DenseCore 22.36% ahead, but those timing windows are not identical and are no
longer the cross-engine evidence. On the separate full-request E2E metric DenseCore
ON is 39.02% slower than llama because TTFT/prefill dominates these short answers.
Migration executed 21 events / 27,437 pages, preserved ON output byte-for-byte, and
cost 2.05% steady decode. The decode claim passes; full-request latency and migration
speedup claims do not.

Cost-bounded fair decode rerun:

```bash
EXPECTED_GCE_MACHINE_TYPE=n2-standard-96 \
MODEL_PATH=/data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
MODEL_SHA256=<verified-sha256> THREADS=16 \
bash benchmarks/numa/fair_decode_claim_e2e.sh
```

The script runs the physical-NUMA penalty preflight first, then DenseCore and llama,
and fails unless the quality checks pass and DenseCore is at least 10% faster.

For a cost-bounded regression rerun, `focused_dense_ab_e2e.sh` defaults to only
`dense_off,dense_on`, always uses two turns, and accepts a previously verified model
hash so a large GGUF is not re-read before every short VM run:

```bash
MODEL_PATH=/data/models/model.gguf \
CASE_FILTER=dense_migration \
PRECOMPUTED_MODEL_SHA256=<verified-sha256> \
bash benchmarks/numa/focused_dense_ab_e2e.sh
```

## Conditions
- **B0** naive (NUMA-off `.so`, first-touch) · **B1** NUMA-off + `numactl --interleave=all` (honest baseline)
- **N1** sticky (NUMA-on `.so`, migration off) · **N2** + adaptive page migration
- **L0** llama plain · **L1** `--numa distribute` · **L2** `--numa isolate` · **L3** `numactl --interleave=all` + `--numa numactl`

## Rig
**GCP c3-standard-176** is the preferred Tier-0 stress shape for this script:
176 vCPU / 704GB RAM, large enough for `Qwen3.5-122B-A10B` **Q4_K_M (~76GB, 3
shards)** with plenty of headroom. It is still expensive enough that the first run
should be a smoke, not the full 8-condition matrix.

Run only if `numactl -H` shows `available: 2 nodes` **and the host actually charges
a remote penalty**; preflight **aborts otherwise**. The script records host info,
model shard sizes, NUMA topology, THP/NUMA balancing knobs, and GCE machine type
when metadata is available.

> **A declared distance matrix is not evidence.** On a virtualized host the SLIT
> comes from the hypervisor and can advertise a penalty the hardware never charges.
> GCP `n2-standard-32` reports `2 nodes, distance 10/20` but measures **−1%
> bandwidth / +4% latency** remote, with remote bandwidth above what any UPI link
> could carry. It passes `check_numa_distances` and would produce a confident,
> meaningless throughput number. Preflight therefore also runs
> `check_numa_penalty`, which measures triad bandwidth and pointer-chase latency
> local-vs-remote under `numactl` bindings (verifying page placement via
> `get_mempolicy` first) and aborts when the measured penalty is below
> `NUMA_PENALTY_BW_MIN_PCT` / `NUMA_PENALTY_LAT_MIN_PCT`. Those thresholds are
> intentionally **not** env-overridable — an override is a bypass. See
> `NUMA_TIER1_FINDINGS.md` §2.

## How it talks to the servers (matches `remote_server_eval.sh`)
- DenseCore: **one** binary `bin/densecore-server serve --grpc=false`; the NUMA-on/off
  choice is the **`libdensecore.so` selected by `LD_LIBRARY_PATH`** (`build-numa` vs
  `build-nonuma`), `HOST/PORT/THREADS` are **env vars**, model via `POST /v1/models/load`.
- Measurement is **token-accurate**: non-stream responses, real `usage.completion_tokens`,
  prefill/decode separated by a max_tokens=1 probe. **No SSE chunk counting** (chunking
  differs per engine and would fake the result).

## Setup
```bash
sudo apt-get update && sudo apt-get install -y numactl jq python3 curl libhwloc-dev libnuma-dev
# build the Go server once:   go build -o bin/densecore-server ./server/cmd/densecore
# build llama.cpp (CPU) for llama-server
# download Qwen3.5-122B-A10B Q4_K_M shards; point MODEL_PATH at shard 1
```

## Run
```bash
CASE_FILTER=smoke \
EXPECTED_GCE_MACHINE_TYPE=c3-standard-176 \
MODEL_PATH=/data/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server \
ROOT_DIR=~/DenseCore \
./tier0_numa_validate.sh
```
`CASE_FILTER=smoke` runs B0/B1/N1/L3 only, with `REPEATS=1`,
`MAX_TOKENS=128`, `MIN_TOKENS=32` unless overridden. If that passes, run the full
matrix:

```bash
EXPECTED_GCE_MACHINE_TYPE=c3-standard-176 \
MODEL_PATH=/data/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server \
ROOT_DIR=~/DenseCore \
REPEATS=2 MAX_TOKENS=256 MIN_TOKENS=64 \
./tier0_numa_validate.sh
```

Useful controls:
- `CASE_FILTER=all|smoke|densecore|llama|B1_interleave,N1_sticky,...`
- `STRICT_PREFLIGHT=1` aborts on kernel `numa_balancing=1` instead of warning.
- `MIN_RAM_MODEL_MULTIPLIER=2` requires host RAM to be at least 2x the total GGUF
  shard bytes.
- `PARITY_STRICT=1` makes Gate D failure exit nonzero.

Edit `SERVER_BIN`/paths in the CONFIG block if your layout differs. Summary table +
per-condition JSON + numastat/numa_maps snapshots land in `numa_tier0_<timestamp>/`.

## Honest limits
- The cost-bounded final rows use exactly two turns per condition. They show a
  repeatable direction across both turns, not a confidence interval or a broad
  workload distribution.
- The 2/2 gate is mechanical coherence QA, not a factual benchmark. It catches
  empty/short/echo/repetition/thinking-leak failures; it does not establish model
  accuracy or replace logits/token parity.
- **Page migration (N2)** needs `CAP_SYS_NICE`; often denied in cloud guests. On the
  qualified 96-vCPU host the two-cycle stable admission and already-resident skip policy
  executed 21 migration events / 27,437 pages and passed 2/2 serving QA. It remained
  2.05% slower in steady decode, so execution/stability are validated but performance
  utility is not.
- Cross-engine output is **not** bit-exact (different kernels). The *hard* parity gate is
  **N1 vs B0** (same engine, routing-only change).
- **Parity is not quality.** Gate D diffs two runs of the *same* engine, so an engine that
  replays the prompt replays it identically in both and the diff passes. `measure.py`
  therefore rejects each response on its own (`output_defects`): a response that appears
  verbatim inside its prompt, or that collapses to a repetition loop, fails the run in both
  `capture` and `measure` mode. This is the gate that was missing when a fully broken decode
  still reported `model_execution_contract_valid=1` and `graph_kernel_coverage=100`
  (`NUMA_TIER1_FINDINGS.md` §3.5). Containment is checked in the whole-output direction only,
  so a correct answer that quotes the prompt (the quicksort stub above) is not flagged.
- **remote%** is **system-wide** `numastat` over the timed window (warmup/parity excluded),
  valid only on a **dedicated, otherwise-idle box**. `numa_maps` snapshots show per-PID page
  *placement*; true per-**access** locality needs `perf` (PEBS / UPI/xGMI counters) — add it
  if you have perf privileges. (`numastat -p` only shows placement, not hit/miss, so it
  cannot give access deltas.)
- Build gate: NUMA-on build must log `NUMA support enabled`; off build must log
  `NUMA libs not found` (else the OFF baseline is contaminated). The script aborts if not.
- **Caching is forced OFF** during measurement (`DENSECORE_AGENT_PROMPT_CACHE=off` +
  `DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE=1`; llama `cache_prompt=false`) so the
  prefill/decode wall-clock split is real, not a cache-hit artifact. The script checks
  `/metrics` and warns if `prompt_cache_tokens_reused_total` is nonzero; measure.py
  hard-fails if the decode window collapses to ≤0.
- DenseCore model load uses `DENSECORE_MODEL_LOAD_STRATEGY=force` so each condition
  does a straightforward one-engine load in its own fresh server process.
