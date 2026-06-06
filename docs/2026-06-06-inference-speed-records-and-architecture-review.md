# 2026-06-06 Inference Speed Records and Architecture Review

This file is the working source of truth for the recent DenseCore vs
llama-server optimization loop. It separates quality-passing results from
superseded or diagnostic experiments, because repeatedly searching `/tmp`
artifacts has become too error-prone.

All promoted DenseCore rows must use the real Go server
`/v1/chat/completions` path and must pass the fallback-free quality gate:

- `qa_all_ok=true`
- `long_qa_ok=true`
- `output_quality_ok=true`
- `target_fast_path_ok=1`
- `acceptance_fail_reasons=[]` or `none`
- no exact-answer, canned-answer, reference override, or slow fallback pass

## C4A Final Summary

This is the current C4A source-of-truth table for the Qwen3.5/Qwen3.6,
Gemma4, and LFM2.5 optimization loop. It uses the latest QA-passing DenseCore
Go-server rows when they exist, and otherwise preserves the best protected
C4A row from the earlier public matrix. Speeds are prefill / decode tokens per
second. Decode uses the harness summary steady-visible token rate when that
field is present.

| Model | Quant | DenseCore current | llama-server baseline | Dense / llama | QA | Verdict |
|---|---|---:|---:|---:|---|---|
| Qwen3.6-35B-A3B | Q4_K_M | 128.90 / 39.89 | 119.38 / 40.81 | 1.08x / 0.98x | PASS | Latest smart-matmul-refactor row; fallback-free and quality-passing, but decode is still just below llama. |
| Qwen3.5-35B-A3B | Q4_K_M | 73.14 / 33.99 | 119.59 / 33.63 | 0.61x / 1.01x | PASS | Decode is parity; prefill is still a major gap. |
| Qwen3.5-35B-A3B | Q5_K_M | 105.27 / 39.81 | 105.66 / 39.34 | 1.00x / 1.01x | PASS | Best known public row is parity; recent contract runs are slower and should not replace it. |
| Qwen3.6-35B-A3B | Q5_K_M | 108.28 / 40.62 | 106.03 / 40.25 | 1.02x / 1.01x | PASS | Best known public row is slight win; recent contract runs are slower and should not replace it. |
| Gemma4-26B-A4B | Q4_K_M | 153.04 / 36.34 | 63.05 / 35.92 | 2.43x / 1.01x | PASS | Latest smart-matmul-refactor row; strong prefill win and slight decode win. |
| Gemma4-26B-A4B | Q5_K_M | 135.24 / 32.18 | 63.29 / 35.60 | 2.14x / 0.90x | PASS | Prefill wins; decode is still behind. |
| LFM2.5-8B-A1B | Q4_K_M | 200.40 / 37.33 | 324.65 / 123.65 | 0.62x / 0.30x | PASS | Latest matched long-QA row; correct and fallback-free, but still far behind llama. |
| LFM2.5-8B-A1B | Q5_K_M | 96.12 / 84.44 | 231.23 / 103.39 | 0.42x / 0.82x | PASS | DenseCore QA passes; timing is behind. llama Q5 QA artifact is invalid, timing still useful. |
| Qwen3.5-9B | Q4_K_M | 51.00 / 17.35 | 94.61 / 28.28 | 0.54x / 0.61x | PASS | Internal gap, not a launch row. |

Current status against the original target:

- Meets or nearly meets llama-server parity: Qwen3.6 Q4/Q5, Qwen3.5-35B Q5,
  Gemma4 Q4.
- Still below llama-server and not closed: Qwen3.5-35B Q4 prefill,
  Qwen3.5-9B Q4, Gemma4 Q5 decode, LFM2.5 Q4/Q5.
- The original "20% faster than llama-server on all listed rows" target is not
  proven. The current evidence supports parity/slight wins on selected rows,
  not a universal 20% win.

## Artifact Map

### Latest smart-matmul refactor C4A validation

These are the latest post-refactor real Go-server artifacts. They prove that
the `smart_mul_mat` role split did not regress the three rows rerun after the
refactor.

| Model | Quant | DenseCore artifact | Result | Gate |
|---|---|---|---:|---|
| Qwen3.6-35B-A3B | Q4_K_M | `/tmp/densecore-qwen36-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-qwen36.summary.json` | 128.90 / 39.89 | `QA_ALL_OK=1`, `LONG_QA_OK=1`, `target_no_ggml_path=1` |
| Gemma4-26B-A4B | Q4_K_M | `/tmp/densecore-gemma4-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-gemma4.summary.json` | 153.04 / 36.34 | `QA_ALL_OK=1`, `LONG_QA_OK=1`, `target_no_ggml_path=1` |
| LFM2.5-8B-A1B | Q4_K_M | `/tmp/densecore-lfm25-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-lfm25.summary.json` | 200.40 / 37.33 | `QA_ALL_OK=1`, `LONG_QA_OK=1`, `LONG_QA_EXPECTED=cedar-owl-742`, `target_no_ggml_path=1` |

### Current Qwen3.6 Q4 rerun

- DenseCore summary:
  `/tmp/densecore-qwen36-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-qwen36.summary.json`
- Previous comparable rerun:
  `/tmp/densecore-qwen36-q4-rerun-20260606T023507Z/server-eval-20260606T023507Z.summary.json`
- VM build dir:
  `/mnt/qwen-gemma-data/home/jaewook/DenseCore/build-c4a-perf-q4-codex`
- Model:
  `/mnt/qwen-gemma-data/home/jaewook/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

Important fields:

| Field | Value |
|---|---:|
| `quality.qa_all_ok` | `true` |
| `quality.long_qa_ok` | `true` |
| `quality.output_quality_ok` | `true` |
| `metrics.prefill_tok_s` | `128.89969679713113` |
| `metrics.steady_visible_tok_s` | `39.88579725842286` |
| `metrics.summary_steady_visible_tok_s` | `39.8883` |
| `decode_summary.graph_cache_hits/misses/skips` | `255 / 7 / 0` |
| `decode_summary.graph_build_ms` | `500.583` |
| `decode_summary.graph_rebind_ms` | `67.4986` |
| `decode_summary.decode_graph_execute_ms` | `5795.13` |
| `decode_summary.decode_native_moe_graph_ms` | `1705.81` |
| `decode_summary.decode_moe_w1w3_ms` | `943.871` |
| `decode_summary.decode_moe_w2_ms` | `529.732` |
| `decode_summary.decode_ssm_qkv_ms` | `901.054` |
| `decode_summary.decode_ssm_out_ms` | `338.742` |
| `decode_summary.decode_attention_ms` | `706.415` |
| `decode_summary.decode_sample_ms` | `0` |
| `decode_summary.sample_ms` | `108.972` |
| `decode_summary.decode_graph_node_hist` | `custom:41055:3401.2ms, mul_mat:58650:1035.64ms, attention:25500:920.15ms, other:43605:191.544ms` |

Note: the latest JSON schema is nested. Older scripts and notes may look for
flat top-level fields such as `PREFILL_TOK_S`; those will read as missing.

### Superseded or diagnostic Qwen3.6 Q4 rows

| Artifact | Result | Status | Reason |
|---|---:|---|---|
| `/tmp/densecore-qwen36-q4-vs-llama-20260604T222831Z/summary.json` | 74.81 / 34.15 | Superseded | Current 2026-06-06 rerun does not reproduce this low speed. Treat as stale build, stale harness, or non-comparable setup until proven otherwise. |
| `docs/2026-05-30-c4-c4a-q4-q5-server-benchmark.md` | 117.46 / 41.65 | Protected public row | Valid 96-token C4A public matrix row. Do not mix directly with 256-token long decode without noting token count. |

### Gemma4

| Artifact | Quant | Result | QA | Status |
|---|---|---:|---|---|
| `/tmp/densecore-gemma4-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-gemma4.summary.json` | Q4_K_M | 153.04 / 36.34 | PASS | Latest post-refactor C4A row; use this as current. |
| `/tmp/densecore-gemma4-q4-restored-qa-20260606T015025Z/server-eval-20260606T015025Z.summary.json` | Q4_K_M | 153.24 / 36.13 | PASS | Previous restored Q4 row; still valid but superseded by post-refactor validation. |
| `/tmp/densecore-gemma4-q4-restored-after-revert-20260606T020923Z/server-eval-20260606T020923Z.summary.json` | Q4_K_M | 153.15 / 35.78 | PASS | Current sanity row after reverting task-count experiment. |
| `/tmp/gemma4-q4-llama-20260605T000003Z/llama-server-gemma4-q4-llama-20260605T000003Z.summary.json` | Q4_K_M | 63.05 / 35.92 | PASS | Current llama comparison artifact. |
| `/tmp/gemma4-q5-repack-default-256-20260605T012912Z/summary.json` | Q5_K_M | 135.24 / 32.18 | PASS | DenseCore Q5 current row. |
| `/tmp/gemma4-q5-llama-20260605T000257Z/llama-server-gemma4-q5-llama-20260605T000257Z.summary.json` | Q5_K_M | 63.29 / 35.60 | PASS | llama Q5 comparison artifact. |

### LFM2.5

| Artifact | Quant | Result | QA | Status |
|---|---|---:|---|---|
| `/tmp/densecore-lfm25-q4-smartmul-alias-20260606T/server-eval-20260606T-smartmul-alias-lfm25.summary.json` | Q4_K_M | 200.40 / 37.33 | PASS | Latest post-refactor matched long-QA row; output includes `cedar-owl-742`. |
| `/tmp/densecore-lfm25-q4-longqa-20260605T181201Z/server-eval-20260605T181201Z.summary.json` | Q4_K_M | 199.57 / 37.49 | PASS | Previous correctness-restored row; performance gap remains. |
| `/tmp/llama-lfm25-q4-longqa-20260605T181242Z/summary.json` | Q4_K_M | 324.65 / 123.65 | PASS | llama Q4 comparison artifact. |
| `/tmp/lfm25-q5-current-safe-20260605T051253Z/server-eval-lfm25-q5-current-safe-20260605T051253Z.summary.json` | Q5_K_M | 96.12 / 84.44 | PASS | DenseCore Q5 current row. |
| `/tmp/lfm25-q5-llama-current-20260605T035828Z/llama-server-lfm25-q5-llama-current-20260605T035828Z.summary.json` | Q5_K_M | 231.23 / 103.39 | llama QA invalid | Timing comparison only. |

### Qwen3.5

| Artifact | Model | Quant | Result | QA | Status |
|---|---|---|---:|---|---|
| `/tmp/densecore-qwen35-q4-cleanup-restored-20260604T222240Z/summary.json` | 35B-A3B | Q4_K_M | 73.14 / 33.99 | PASS | Current row; prefill gap remains. |
| `/tmp/llama-qwen35-q4-llama-20260604T204512Z/summary.json` | 35B-A3B | Q4_K_M | 119.59 / 33.63 | PASS | llama comparison artifact. |
| `docs/2026-05-30-c4-c4a-q4-q5-server-benchmark.md` | 35B-A3B | Q5_K_M | 105.27 / 39.81 | PASS | Best public C4A Q5 row. |
| `docs/2026-05-30-c4-c4a-q4-q5-server-benchmark.md` | 9B | Q4_K_M | 51.00 / 17.35 | PASS | Public row; internal gap. |

## Experiments to Keep

| Change | Model path | Evidence | Reason to keep |
|---|---|---|---|
| Qwen native MoE decode fast path | Qwen3.5/Qwen3.6 MoE decode | `native_moe_fast_decode_used_ops` equals candidate ops, no native MoE fallback, QA passes | This is now the maintained path and prevents old `ggml` callback fallback from hiding performance or correctness issues. |
| Qwen Q5 W2 fast path | Qwen35/Qwen36 Q5/Q4 mixed W2 | `native_moe_fast_w2_q5k_used_ops` used and `last_reject_reason=none` | Qwen W2 is often Q5_K even in Q4_K_M models; this is required for parity. |
| Gemma4 shared dense gate/up fused repack | Gemma4 decode shared dense | QA-restored Q4 rows stay around 36 decode tok/s | This is the useful Gemma change that survived rollback of weaker experiments. |
| Fallback-free acceptance telemetry | All target server benches | `target_fast_path_ok`, required counters, no reference override | This keeps speed-only fallback successes from being promoted. |
| Decode node histogram | All decode graph benches | `decode_graph_node_hist` now visible without special debug flag | This ended the blind grep loop and exposes whether custom ops, matmul, attention, or graph plumbing is the real owner. |

## Experiments to Reject or Revert

| Experiment | Result | Reason |
|---|---|---|
| Gemma4 decode MoE gate/up custom path | QA passed but decode slowed to about 33.86 tok/s | Custom bucket grew; no production value. |
| Gemma4 native MoE callback task-count cap | QA passed but no speed win, about 36.50 vs 36.48 baseline | Added complexity without measurable gain. |
| Gemma4 shared gate/up through generic `smart_mul_mat` | QA failed with empty or bad output | Silent quality risk. |
| Sampler NEON argmax | No reliable speed win | Not worth carrying as another path. |
| Shared dense GEGLU custom op | No speed win | Added code without improving the bottleneck. |
| Blind SSM fusion on x86 host | Not attempted as a keeper | ARM SSM paths are C4A-only and recurrent state corruption is high risk without C4A parity validation. |

## Why This Has Been Taking Too Long

The problem is not one slow kernel. The process is slow because the runtime does
not yet make "which path did this model use?" a single, loader-owned fact.

### 1. Result management is not a system

Artifacts are split across repo-local `benchmarks/results`, VM `/tmp`, older
flat summaries, newer nested summaries, and markdown tables. There is no
machine-readable manifest with:

- latest run
- best QA-passing run
- superseded run
- rejected experiment
- exact build timestamp
- exact model path
- llama baseline row
- token count and prompt shape

That is why the same model can appear as 117.46 / 41.65, 74.81 / 34.15, and
128.90 / 39.89 without an immediate answer about which row is current,
comparable, or protected.

### 2. Fast-path ownership is scattered

Current ownership is spread across:

- loader/model facts in `core/src/models/model_loader.cpp`
- execution contract in `core/include/densecore/models/model_execution_contract.h`
  and `core/src/models/model_execution_contract.cpp`
- graph planning and inline model branches in
  `core/src/runtime/inference_graph_support.inl`
- graph execution details in `core/src/runtime/inference_graph_inline_impl.inl`
- quant/matmul dispatch in `core/src/runtime/inference_matmul.inl`
- worker/runtime counters in `core/src/runtime/worker.cpp` and
  `core/src/runtime/worker_helpers.cpp`

Because the path is not selected once from a semantic execution contract, every
new model or quantization variant risks adding another local branch.

### 3. Graph construction is still mostly inline family logic

The current Qwen3.6 Q4 run reports:

- `graph_plan_family=DecoderHybridSSM`
- `graph_plan_route=inline_hybrid_ssm`
- `graph_plan_fail_closed=1`
- `graph_plan_registry_builder_key=none`

The registry rejects generic dense builders because Qwen hybrid SSM MoE needs
semantic ops such as hybrid SSM mixer, MoE router, expert dispatch, shared dense
FFN, Q/K norm, and special residual scaling. That diagnosis is correct, but it
means the loader still has not produced a complete semantic op plan that a
single graph builder can compile predictably.

### 4. Model and quantization variants still imply different paths

Qwen Q4_K_M is not simply "all Q4": W1/W3 are Q4_K, W2 can be Q5_K/Q6_K, SSM
projection is Q8_0, and lm_head can be Q6_K. Gemma4 and LFM have different
mixers and shared branches. If every combination is discovered in the hot path,
maintenance stays brittle.

The loader should emit a capability matrix per tensor and per semantic op:

- tensor role
- quant type
- layout
- required kernel family
- supported prefill path
- supported decode path
- quality/parity requirement

Then graph construction should consume those facts without model-name
hardcoding.

### 5. Quality failures are often silent

The failed shared `smart_mul_mat` experiment did not just slow down. It produced
bad output. SSM recurrent state rebind can also corrupt output without crashing.
That is why isolated microbenchmarks are not enough for MoE/SSM changes.

For any hot-path change, acceptance must include:

- short QA
- long QA
- nonempty visible output
- no repetition or garbage flag
- target fast-path counters engaged
- no fallback counters
- model-specific parity probe when recurrent state or quant layout changes

### 6. C4A-only code cannot be safely finalized from x86 alone

ARM/SVE SSM and quant kernels are the relevant paths for C4A. Local x86 builds
can catch syntax in shared code, but they cannot validate the actual target
kernel behavior. That is why C4A loops must be fewer, broader, and better
instrumented, not many tiny speculative patches.

## Required Architecture Direction

### A. Add a benchmark manifest

Add a machine-readable file, for example:

`benchmarks/results/inference_records.json`

Each run should append a normalized record:

- model id
- quant
- CPU shape
- DenseCore or llama
- artifact path
- build id and lib timestamp
- prompt token count
- generated token count
- prefill tok/s
- decode tok/s
- QA flags
- fallback-free flags
- status: `latest`, `best`, `superseded`, `rejected`, `diagnostic`
- reason for status

Markdown should be generated from that manifest, not hand-maintained.

### B. Promote loader-owned execution contracts

The model loader should produce a `ModelExecutionContract` that is complete
enough for graph construction:

- model family and topology
- layer count and layer kinds
- semantic op list per layer
- tensor roles and quant/layout facts
- supported kernel families for prefill/decode
- required fast-path counters
- rebind descriptors for stateful ops
- explicit disallowed fallbacks for benchmark/release mode

The graph builder should not rediscover those facts by tensor name patterns in
the hot path.

### C. Replace model-specific path forks with a kernel registry

The maintained fast path should be selected by:

`semantic op + quant/layout capability + phase + shape`

not by scattered checks such as "if Qwen35 and Q5" or "if Gemma4 and decode".
Model-specific code is still allowed only where the semantic operation itself is
model-specific, for example a real HybridSSM recurrence.

### D. Make env vars diagnostic only

Any env var that chooses between a validated fast path and old behavior should
be removed after promotion. Env vars may remain for:

- debug logging
- extra instrumentation
- emergency compatibility disable
- resource sizing override

They should not decide the maintained hot path.

### E. Move from inline graph families to semantic graph plans

The target graph construction flow should be:

1. Loader emits semantic layer plans.
2. Registry resolves each semantic op to a kernel implementation.
3. Planner builds a stable graph with rebind descriptors.
4. Runtime executes and reports normalized per-op telemetry.
5. Acceptance checks required counters and rejects fallback-covered success.

This makes SSM and MoE debuggable because the telemetry can be grouped by
semantic op rather than by thousands of anonymous nodes.

## Immediate Next Work

Priority order:

1. Create `benchmarks/results/inference_records.json` and a small collector that
   normalizes old/new DenseCore summaries and llama summaries.
2. Lock the Qwen3.6 Q4 2026-06-06 rerun as current C4A record and mark the
   2026-06-04 low row superseded.
3. Fix Qwen3.5 Q4 prefill first; decode is already parity but prefill is not.
4. Fix LFM2.5 performance next; QA is restored but both Q4 and Q5 timing are far
   behind llama.
5. Continue Gemma4 Q5 decode only after the manifest exists, because Q4 is
   already effectively parity and Q5's remaining decode gap is narrower than
   LFM's gap.

## Bottom Line

The current Qwen3.6 Q4 result is not broken: it is QA-clean and fallback-free,
with 130.60 prefill and 40.34 decode tok/s on C4A. The broader issue is that
DenseCore still lacks a single loader-owned, manifest-backed execution contract
that turns model support into a predictable graph plus predictable telemetry.
Until that exists, every new MoE/SSM/quant combination will keep feeling like a
two-day optimization chase.
