# Benchmark Evidence

This document separates measured evidence from product claims. DenseCore does not
publish a speedup solely because a process completed or a single kernel counter
improved. The real server path and inference-quality gates are part of the result.

## Current Evidence Boundary

The newest complete matched checkpoint available on 2026-07-13 covers Google Cloud
C4 and C4A with Qwen3.5, Qwen3.6, and Gemma4. It is useful engineering evidence,
but it is **not yet a promotion-grade result for commit `22b672d`**:

- each row is one sequential DenseCore/llama-server pair
- the six rows were not all produced by one identical final binary
- five-cycle ABBA medians have not been collected
- strict QA was run separately for the final candidates

Consequently, the table supports prioritization and a dated checkpoint, not a
blanket "DenseCore is faster than llama.cpp" claim.

The separately documented 2026-08-03 two-socket Qwen3.6 decode checkpoint is
not folded into this C4/C4A table: it uses a different host shape and measures
isolated decode. See [NUMA sticky routing](NUMA_STICKY_ROUTING.md) for its
two-repeat evidence, strict scope, and rerun contract.

## Matched Single-Run Checkpoint

Units are prompt prefill and steady visible decode tokens per second. Higher is
better. Delta is `(DenseCore / llama-server - 1) * 100`.

| Platform | Model | llama PF | Dense PF | PF delta | llama DEC | Dense DEC | DEC delta | +10% both phases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GCP C4 x86 | Qwen3.5-35B-A3B | 71.1831 | 102.3342 | +43.8% | 17.0785 | 21.6691 | +26.9% | checkpoint pass |
| GCP C4 x86 | Qwen3.6-35B-A3B | 76.6175 | 102.1235 | +33.3% | 18.2489 | 21.9770 | +20.4% | checkpoint pass |
| GCP C4 x86 | Gemma4-26B-A4B | 65.5584 | 68.2530 | +4.1% | 15.1203 | 14.2204 | -6.0% | fail |
| GCP C4A Arm | Qwen3.5-35B-A3B | 118.6814 | 121.5738 | +2.4% | 40.0621 | 37.1703 | -7.2% | fail |
| GCP C4A Arm | Qwen3.6-35B-A3B | 118.2867 | 121.1326 | +2.4% | 39.9335 | 36.9819 | -7.4% | fail |
| GCP C4A Arm | Gemma4-26B-A4B | 61.0829 | 99.5260 | +62.9% | 36.4330 | 32.2212 | -11.6% | fail |

The strongest checkpoint is Qwen on C4 x86. Gemma4 has a strong C4A prefill row,
but decode misses the comparison baseline. No C4A model in this checkpoint clears
the 10% gate in both phases.

## Quality Gate

The strict candidate checks exercise the real Go `POST /v1/chat/completions` path.
They require:

- short factual QA, including Paris and Seoul checks
- the long exact-answer key `cobalt-river-913`
- `QA_ALL_OK=1`, `LONG_QA_OK=1`, and `OUTPUT_QUALITY_OK=1`
- the intended target fast path
- no exact-answer promotion, canned response, reference override, or slower path
  masking a failed target path
- no target model fast-path rejection or unapproved fallback

Strict QA passed separately for the Qwen3.5 candidate, Qwen3.6 candidate at
`aaa9962`, and the Gemma4 template-fixed candidate. Those QA artifacts do not by
themselves establish throughput because they did not contain a matched baseline.

## Comparison Contract

A publishable comparison must hold these inputs constant:

1. Same cloud machine type and CPU count.
2. Same GGUF file and quantization.
3. Same prompt bytes, rendered chat template, sampling parameters, and generated
   token target.
4. Same thread count and CPU-affinity policy.
5. DenseCore through its Go server and llama.cpp through `llama-server` using the
   same API-level workload.
6. Cold/warm state declared; no silent prompt-cache reuse.
7. Five-cycle ABBA ordering, reporting p50 and spread rather than the best run.
8. Strict output-quality and fast-path gates on the measured DenseCore binary.

The maintained remote harnesses are:

- `benchmarks/remote_server_eval.sh`
- `benchmarks/remote_llama_server_eval.sh`
- `benchmarks/fair_llm_report.py`

## Allowed Public Claim

Before a final-HEAD ABBA rerun, use wording no stronger than:

> A QA-gated engineering checkpoint showed 20-44% higher single-stream Qwen
> prefill/decode throughput than llama-server on Google Cloud C4 x86.

After the rerun, replace "checkpoint" only if the same final binary passes strict
QA and both phase medians remain at least 1.10x.

Do not claim:

- all models are faster than llama.cpp
- C4A or Arm has reached the same Qwen advantage
- the table proves concurrent throughput, p95 latency, or cost per token
- a later code change inherits an earlier benchmark without rerunning it

## Next Qualification

The smallest useful publication run is:

1. freeze one release candidate and record compiler, CMake cache, model hashes,
   server binaries, and commit
2. rerun strict QA on all six DenseCore rows
3. run five-cycle ABBA for C4 Qwen3.5 and Qwen3.6 first
4. report concurrency 1/2/4/8, TTFT p50/p95, steady decode, memory peak, and
   tokens per dollar as separate dimensions
