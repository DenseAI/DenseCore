# v0.2 implementation and release gates

D0 is 2026-10-25; D21 is 2026-11-15. Scope is A–H: personal PCs,
resource-limited Kubernetes pods and two-socket NUMA. DeepSeek V4/V4.1 is
excluded. This is an execution ledger, not a released-support announcement.

Code, real-model correctness, matched performance and automatic activation are
separate gates. Fixture or compilation success does not qualify a model. A
missing or truncated final answer fails quality even if reasoning contains the
expected text. The current automatic MTP profile registry is empty.

The current evidence snapshot is frozen source
`7ecaf86072246752ac15a1bf7caab1b2ff5f9217`. Its global stored-KV-precision
change affects cold non-F32 attention, so earlier `8c5433a` model successes do
not automatically carry forward. Later fixes require their own integrated
binary and model/package checks.

| Track | Verified on canonical `7ecaf86` | Remaining release gate |
|---|---|---|
| A | Full native: 1263 passed, 14 existing skips, zero failures (1277 total). Selected ASAN/LSan suite: 600 passed, 9 existing skips, zero failures/reports (609 total). Qwen3 cold/warm native ID parity and HTTP reasoning/final separation passed | Gemma E2B c=4 same-prompt isolation fails; actual stochastic HTTP stop strings leak in both NONE and MTP. Fix and requalify affected paths |
| B/H | Resource/planner fixtures, pinned 0.8B serving within 4 GiB, explicit initialization rejection within 1 GiB without OOM kill; embedding AUTO/LAST C/Python/HTTP 12 comparisons bitwise equal, finite and normalized | Final integrated image/pod verification after fixes; broader c=8 resource/isolation and platform qualification |
| B Gemma | Pinned E2B c=1 long-answer/repeat HTTP quality passed 6/6 | E2B c=4 same-prompt outputs differ by slot despite 24 semantic/EOS successes; native-ID investigation and E4B canonical long/repeat/c=1/4 qualification remain |
| C | README-pinned 0.8B and plain 2B combined strict checks passed: c=1/2/4, five-turn final answers, actual restore/skipped prefill, cancellation/recovery and real snapshot/KV eviction | Qwen3.6 model-specific qualification; matched performance and default-policy promotion remain separate |
| D | 0.8B/2B MTP files, K=2, c=1/c=4: same-file NONE/MTP 128 native IDs and HTTP/stream/cancel/recovery passed | Actual stochastic 0.8B c=1 fails HTTP stop handling (NONE 3 leaks, MTP 2); remaining stochastic cells, current K=1/3, real 35B MoE head and performance qualification remain |
| E | Local portable amd64 image built; 4 GiB serving/1 GiB rejection and kind chart/downloader/limits/checksum components passed | Strict Count and full active drain fail. Go-only readiness fix plus stricter active-observation harness is a separate pending image validation; final integrated package and native Arm remain unqualified |
| F | GPT-OSS packed MXFP4/bias/router/activation, attention sink/window and tokenizer/Harmony fixtures | Real-model HTTP/long output/c=1/4/stop/stream and peak RSS/context unavailable on this host; safe capacity refusal is not inference support |
| G | Public physical-NUMA preflight and matched ABBA harness | Actual two-socket 35B/122B measurements, 16/48 physical-core placement and remote-memory penalty; no default promotion |

Qwen3.5 prefix qualification used `DENSECORE_QWEN35_PREFIX_CACHE_REUSE=1`
with global prefix/snapshot controls enabled. The opt-in remains separate from
quality and performance promotion. Qwen3.6 retains its model-specific gate.
The combined 0.8B/2B runs each observed 14 real restores. Separate near-full
pressure runs reclaimed 105 KV blocks and respectively 10/11 snapshots
(202,014,720 / 222,216,192 logical payload bytes). Pressure-control output is
not complete-answer quality evidence, and turn-five single samples are not a
performance comparison.

Explicit MTP executes batched K+1 verification with suffix rollback and SSM
recovery. Its 128-token parity runs may finish by length; they do not prove a
complete long answer or stochastic distribution equivalence. The first real
stochastic run used temperature 0.7, top-p 0.95, top-k 40 and repetition penalty
1.05 and failed the HTTP stop gate. Native acceptance/rejection/rollback was
observed, but that does not excuse literal stop text reaching clients. Native
request seed wiring and real-model distribution qualification remain absent.
The production AUTO registry contains zero profiles and selects NONE without
model hashing or runtime-profile capture. No MTP acceleration is claimed.

## Reproduction and evidence

Use `dc-build` with one shared build lock in the local multi-worktree setup.
Build/test source must remain stable during compilation; rebuild after changing
headers or registration before collecting any real-model evidence.

- Native: complete `densecore_tests`, A/D ASAN, and focused model-logit/rollback tests.
- Adapters: changed Go and Python tests; Go race checks for callback/admission paths.
- Real quality: the [final-answer evaluator](../benchmarks/economics/evaluate_openai_quality.py)
  requires a final answer, successful finish and complete SSE stream. Save raw
  output and all failures. Textual HTTP parity is weaker than full native-ID parity.
- State parity: [prefix/eviction](../benchmarks/personal/check_prefix_parity.py),
  [greedy MTP](../benchmarks/personal/check_mtp_parity.py) and
  [stochastic MTP](../benchmarks/personal/check_mtp_sampling.py) entry points.
  Match each ledger's actual harness/case SHA and parameters: separately recorded
  harness revisions are not interchangeable merely because filenames match.
- Final package: [container smoke](../scripts/container_model_smoke.sh) and
  [kind smoke](../scripts/kind_model_smoke.sh); [Kubernetes instructions](KUBERNETES.md).
- Artifact identities: [model catalog](MODEL_CATALOG.md).
- CPU comparisons: [personal PC](../benchmarks/personal/README.md) and
  [physical NUMA](../benchmarks/numa/README.md) harnesses.

Performance requires fixed model bytes, binary/library hashes, CPU/build flags,
sampling, cache state and input/output lengths. Keep all five ABBA cycles and
predeclared A/A noise. Judge C4/C4A × c=1/c=4 independently with no compensating
regressions. Promotion requires at least max(3%, A/A noise), with additional
1.15× decode for MTP or +15% 48-core decode for NUMA and no completion-time,
TTFT, throughput or memory regressions. K=2 remains preferred if K=3 has no
significant advantage.

Cloud spending is capped at $50, with $10 held for billing delay/disk/transfer
and price changes; see [session budget](../benchmarks/economics/SESSION_BUDGET.md).
On D0 the configured GCP project had billing disabled and Compute API disabled.
No paid validation was performed or charged. Cloud/platform performance and
native Arm gates therefore remain unavailable until those prerequisites change.

If mandatory correctness or final-package checks remain incomplete on D21,
record release readiness as incomplete and the release delayed. Do not silently
remove a track or turn fixture passes into real-model support.

## Retained failures and current integration boundary

- Qwen3 HTTP default/off thinking modes passed 16 semantic/protocol cases,
  including reasoning/final separation and complete streams. The separate
  explicit-off “only a number” arithmetic gate failed: output included
  `6 × 7 = 42`. The pinned llama.cpp oracle (`74bca3950054eed1795f525cb9cf06bfe796492d`)
  also failed that format requirement. Keep the format failure; an oracle
  failure does not relax the DenseCore gate. Native prefix OFF/ON parity is a
  separate passed check on canonical F16 KV.
- E2B c=4 long-answer outputs differ for the same input across slots, with
  reported completion counts 129/128/128/132. HTTP 200, correct semantic content
  and EOS do not establish request isolation. Raw native-ID investigation is
  pending. Serializing c=4 or silently reducing concurrency does not pass it.
  Canonical E4B validation remains pending; earlier Paris or fixture successes
  are not complete model support.
- The canonical image is a **local** portable amd64 candidate with image ID
  `sha256:affcd8eda8028aa5124a769b62d22533024821c17cdbc2a8b7fe3ebc3bd55cf9`.
  Strict Count still fails. An accepted stream completed more than five seconds
  after shutdown began, but readiness remained 200 while work was active;
  full active drain fails. Late readiness/admission observations do not pass
  the lifecycle gate.
- Separate Go-only candidate `f50d958b283d76441c1ff774febfef0d6108ab05`
  includes early readiness withdrawal, drain-time read-only diagnostics and a
  harness requiring same-round readiness 503/new-admission refusal while an
  accepted request is active. Focused repeated and full relevant Go race checks
  passed. Its core tree is identical to `7ecaf86`; actual image/container/kind
  validation is pending. This is not the final image for later Gemma/stop fixes.

Local raw ledger filenames (not published artifact download links):

- `v02-canonical-7ec-artifact.json`: exact source/build/library/server identities,
  native/ASAN/Go checks and dependency audits. Python harness 35 passed on this
  candidate; Python binding 49 passed on unchanged `8c5433a` sources.
- `v02-correctness-canonical-7ec-results.json`,
  `v02-canonical7ec-qwen3-default-qa.json`,
  `v02-canonical7ec-qwen3-off-qa.json`,
  `v02-canonical7ec-qwen3-off-strict-number.json` and
  `v02-qwen3-canonical7ec-llama-oracle.json`: prefix, embedding and separated
  Qwen3 native/HTTP/oracle outcomes.
- `v02-mtp-canonical7ec-matrix-report.json` and
  `v02-mtp-sampling7ec-08b-c1/summary.json`: greedy matrix and retained
  stochastic stop/punctuation failures.
- `v02-canonical7ec-e2b-c1-qa.json` and
  `v02-canonical7ec-e2b-c4-qa.json`: c=1 quality and c=4 isolation failure.
- `v02-packaging-canonical7ec-reviewed-components.json` and
  `v02-readiness-evidence.json`: reviewed package failures and the separate
  Go-only fix boundary. An older component summary's drain “pass” is superseded
  by the reviewed full-lifecycle failure.

Archive raw results with their identities before publishing claims. Public
reproduction entry points are the relative script/document links above, not
local evidence paths. AUTO collection initially accepts only proven single-node,
full-online-affinity Linux execution; partial cpusets, unknown runtime conditions
and unverified dependency/allocator overrides choose NONE. Ordinary resource-aware
serving is independent of that conservative automatic-qualification restriction.
