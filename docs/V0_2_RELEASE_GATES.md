# v0.2 implementation and release gates

D0 is 2026-10-25; D21 is 2026-11-15. Scope is A–H: personal PCs,
resource-limited Kubernetes pods and two-socket NUMA. DeepSeek V4/V4.1 is
excluded. Release readiness remains incomplete.

Implementation, real-model correctness, matched performance and automatic
activation are separate gates. A missing or truncated final answer fails even
when reasoning contains the expected text. No automatic MTP profiles or new
performance/default promotions are qualified.

## Evidence boundaries

The integrated source snapshot is
`4ad6a35896230d4d40a99e998916460a4954023f`. Full native: 1347 run, 1333 passed,
exactly the same 14 existing skips, zero failures/disabled. Same-source Go
all-package/race checks, SDK 164 unit checks with 27 existing optional skips,
and 88 standard-library benchmark tests passed. These adapter/tool checks are
separate from actual model execution and are not real Python JSON proof.

Critical ASAN/LSan on this exact source: 702 run, 693 passed, the same 9 skips
as archived `db9883a`, zero failures/disabled/reports with leak detection enabled.
The selected suite includes all 12 added JSON lifecycle/session/helper and Gemma
Q4/F32/PLE-capture cases; two existing JSON tests are newly selected by the filter.
It is not the complete 1347-test normal suite. The tested ASAN binary is
`f26937b6a2306f4e7f6f0634ecfc0915cc51ab5d82c6951ab8a7c1659b5e4442`.
Final actual wrapper audits are no-work. Source, vendor, generated headers,
dependencies and tested binaries are stable; initial audit relinks remain
recorded. Serving library SHA is
`220bf8cbac2858e66ef5703c2ffe16fcf7db710da203d89975e35947608f662d` and server SHA is
`b180e2e6004180e9291de556061be8edb42e1cce526a91b3315dc452661324c5`.
Historical `db9883a` native1335/ASAN688 and `3ed5428` ASAN archives remain separate.

The shared sampler treats top-k=0 as disabling that filter without forcing
greedy selection, conditions nucleus filtering on prior-filter survivors and
intersects active grammar with global masks/valid action ranges. Mock-native
unsatisfiable-grammar error/peer/recovery and ordinary/MTP JSON closing-token
boundaries pass. The old `4545e05` legacy-reference failure and test-only
`db9883a` oracle correction remain preserved; neither is relabeled as a model run.

Real Python generation on the exact integrated `143a2ad` library passed 12 cases
and three native-token controls, covering completion counts/finish reasons,
concurrent metadata, cancellation of the actual native request ID, iterator/task
cleanup, incomplete stop-prefix flush and recovery. Completion accounting is
native-backed. Prompt-token accounting remains a client estimate (zero when no
Python tokenizer is available), not native prompt accounting.

| Track | Verified evidence and source | Remaining gate |
|---|---|---|
| A | Integrated `4ad6a35` full native and 693-pass/9-skip critical ASAN; prior `143a2ad` Go/Python lifecycle | Prior actual c=8/Gemma parity failures remain unresolved; future functional fixes require new sanitizer/model qualification |
| B/H | Resource/planner fixtures; earlier 4 GiB serving and explicit 1 GiB rejection without OOM kill; historical `7ecaf86` embedding AUTO/LAST 12 bitwise C/Python/HTTP comparisons | Actual `143a2ad` c=8 resource/cleanup evidence passes, but request isolation does not; final image/platform checks remain |
| B Gemma | Historical E2B c1 and private `ad1aa84` checks retained; private `d4088c7` cache-ON c1 repeats pass but c4 parity is 6/8 | Rank-0 requests differ from c1. Cache-OFF/mixed/E4B were not run on d408. Q4/F32 fixes are in `4ad6a35` native/ASAN; model requalification remains separate |
| C | Historical `7ecaf86` README-pinned 0.8B/plain 2B combined prefix checks passed c=1/2/4, five-turn answers, restore/skipped prefill, cancellation and eviction | Do not inherit those results across later numerical changes; Qwen3.6, matched performance and default-policy promotion remain separate |
| D | Historical `7ecaf86` greedy parity retained; integrated `4ad6a35` 0.8B/2B JSON NONE/MTP c1/c4 native+HTTP pass and actual probabilistic rejection/rollback/stop suppression | Current probabilistic matrix overall FAIL: one 0.8B c4 French punctuation failure plus separate stop-instruction negative. Population distribution, speed, 35B MoE and auto qualification absent |
| E | Go-only `386b800` portable amd64 image passed actual container and kind strict active drain, resource limits, checksum rejection and safe memory refusal | Strict Count quality still fails. Current root core has no final image validation; original init-log capture has syntax checks only; native Arm remains unverified |
| F | GPT-OSS packed MXFP4/bias/router/activation, attention sink/window and tokenizer/Harmony fixtures | Actual HTTP/long/c=1/4/stop/stream inference and supported peak RSS/context remain unverified; safe capacity refusal is not model support |
| G | Public physical-NUMA preflight and matched ABBA harness | Actual 35B/122B two-socket measurements, 16/48 physical cores and remote-memory penalty; no promotion |

## Retained correctness failures

Actual H resource testing on `143a2ad` observed a pressure snapshot with 8 active
and 8 pending requests (maximum observed pending 12). Oversized prompt admission
returned an explicit memory error with no callbacks; all 36 accepted requests
published one terminal and one completion. These pass bounded resource/lifecycle
checks, not isolation or throughput qualification. Greedy parity fails: the last
admitted c=8 request replaces baseline EOS at zero-based token position 50 with a
newline and continues to the 128-token cap. Separate cache-OFF and cache-OFF with
all thread policies fixed to 8 reproduce the failure. This does not establish a
physical-slot cause or qualify either control as a workaround.

Gemma E2B private `d4088c7` contains the canonical Q4 decode/stride guard changes.
Its cache-ON c1 repeats have 127 native IDs, but two concurrent groups fail:
admission-rank 0 requests (IDs 1/5) produce 135 IDs, first differing at zero-based
index 52 (the 53rd token); the other six requests match c1. Same-source cache-OFF,
mixed-model and E4B runs are **not run**, not passes or reproduced controls.
Older `ad1aa84` rank-2/cache-OFF failures remain preserved separately.

Private DEBUG observer `658e626` reproduces the previous visible per-layer
embedding (PLE) delta with all 17 attention/FFN core stages bit-identical.
Global projection/lookup stages are exact; the local F32 gate delta disappears
through GELU, but identical gated inputs yield different F32 PLE projection
outputs (about `4.77e-7` / `2.38e-7`). The later Q4/F32 fixes and capture fixtures
are included in `4ad6a35` native/ASAN; source-matched model requalification remains
separate. Diagnostic reproduction does not establish
full-model isolation, absence of every observer effect, or model support.

Private stop candidate `5ad54a5` completed 72 native requests, eight native
cancel/recovery cases and eight HTTP disconnect/recovery cases. Its 32 configured
literal-stop probes leaked no stop markers; native logs observed 27 actual
`stop_sequence` finishes. Native-ID-to-HTTP case mapping at c=4 is unproven, so
these counts are aggregate evidence. The 0.8B/2B × c=1/c=4 stochastic profiles
all retain strict FAIL: six HTTP quality failures among 160 main requests.
French formatting and stop-instruction failures are preserved without assigning
them to model or engine by assumption. These samples prove neither population
distribution equivalence nor acceleration. They are private-source evidence,
not the current `4ad6a35` matrix described below.

Integrated `4ad6a35` closes the JSON completion gap: ordinary and MTP output
commit the root-closing token before STOP; fixtures cover an exact max-token
boundary. Actual model probes below use a 512-token output budget.
Actual pinned 0.8B and 2B each pass 16 main native and 32 HTTP checks with NONE/MTP,
c=1/4, greedy K2, context2048 and threads8. Strict brace-whitelist `{}` and
ordinary `{"answer":42}` cases, native cancel/recovery, HTTP stream/nonstream/disconnect/
recovery and UTF-8/usage/terminal checks pass. Each model records 54 actual
`grammar_complete` finishes and batched target width3 with acceptance/rollback.
JSON rejection was not observed; formal rejection fixtures and probabilistic
coverage remain separate. Delivered JSON CallbackEx IDs are a subset of native
samples; real Python JSON and public global action range are not qualified.
Two tiny objects do not establish arbitrary JSON/model support. c=4 denotes
logical engine capacity; width3 is one request's K+1 verification graph, not
four-sequence physical batching or a throughput qualification.

The unchanged non-JSON probabilistic matrix on the same `4ad6a35` artifact is
**overall FAIL**. 0.8B c1 and 2B c1/c4 pass; 0.8B c4 fails one streamed NONE French
answer missing final punctuation (normal STOP/DONE and usage32, budget1024).
A separate NONE stop probe also fails the exact Paris instruction; the existing
format side metric does not include that probe. No engine/model cause is proven.
All 72 native finals including recovery, eight native cancellations and eight
HTTP disconnect/recoveries pass. Of 160 main HTTP requests, one fails the final
quality gate. All 32 stop probes suppress the literal marker and 31 actual
`stop_sequence` finishes are logged. All MTP profiles observe width3,
acceptance/rejection/rollback (885 verifications, 329 rejections, 568 rollbacks in
aggregate). These counts are execution evidence, not latency/throughput results.
Current failures and the earlier private six/160 remain independently recorded;
unseeded samples establish neither population-distribution equivalence nor causal
quality improvement. JSON success does not waive this failure.

Reproduction uses the unchanged [public sampling cases](../benchmarks/personal/mtp_sampling_cases.json)
and validators: context2048, threads8, K2, c=1/4, repeats2, temperature0.7,
top-k40, top-p0.95, repetition-penalty1.05 and native/HTTP output budget1024.
Min-p remains the native default0 (not exposed by the public API); there is no
public seed parameter. Each cell holds the shared lock independently, uses a
new output directory and checks the exact artifact/model/dependency hashes.
No prompt/threshold edits or retries were used. JSON commands and the required
source-bound unit-report preflight are documented in the JSON probe guide below.

Historical Qwen3 `7ecaf86` default/off HTTP modes passed 16 semantic/protocol
cases, reasoning/final separation and cold/warm native parity. Explicit-off
number-only formatting still failed (also on pinned llama.cpp oracle
`74bca3950054eed1795f525cb9cf06bfe796492d`). Oracle failure does not relax a gate.
Historical prefix checks used the model-specific opt-in; actual restore and
snapshot/KV eviction are distinct from performance promotion. Near-full pressure
outputs are control workloads, not complete-answer quality proof.

## Package boundary

The separate Go-only source `386b800d8e27a775a45d8a6798430200d5ceb292` produced
local portable amd64 image
`sha256:17e914ce8e30a1a9a651855c3e2b6ffd43907aaea143d3b51ef58c85b05835fd`.
Its native/runtime file hashes match `7ecaf86`; only the Go server changed.
Actual container and kind tests both observed readiness 503, new-admission 503
and native active=1 in the same probe round while the accepted stream remained
incomplete. Each bounded 32-token stream finished with one finish and one DONE,
7.596/7.903 seconds after the signal command, within the original 30-second
budget. Resource limits, corrupt-cache rejection and cleanup passed. Strict
Count quality remains FAIL; the image is not release-qualified.

Earlier `7ecaf86` readiness-200 and `f50d958` 149×HTTP-500 admission failures remain
recorded. The 386 run did not preserve the original successful init stdout
before pod deletion. The later capture change binds original pod UID/container
ID, image IDs, exit status and log SHA before deletion; it has only shell/Python
syntax qualification so far. A final kind run must prove that new artifact.
Neither the Go-only image proof nor that syntax check qualifies the changed root
core, later Gemma/stop work or native Arm.

## Reproduction and promotion

Use `dc-build` with the shared build/model lock in the local multi-worktree
setup. Keep build source stable and preserve exact source, binary, library,
model and harness identities with every raw failure.

- Quality: the [final-answer evaluator](../benchmarks/economics/evaluate_openai_quality.py)
  requires final content and a successful complete response.
- State: [prefix](../benchmarks/personal/check_prefix_parity.py),
  [MTP parity](../benchmarks/personal/check_mtp_parity.py),
  [stochastic MTP](../benchmarks/personal/check_mtp_sampling.py),
  [JSON probe](../benchmarks/personal/README_mtp_json.md),
  [concurrency](../benchmarks/personal/check_concurrency_budget.py) and
  [Python SDK](../python/README.md).
- Package: [container smoke](../scripts/container_model_smoke.sh),
  [kind smoke](../scripts/kind_model_smoke.sh), [Kubernetes](KUBERNETES.md)
  and the [model catalog](MODEL_CATALOG.md).
- Performance: [personal PC](../benchmarks/personal/README.md),
  [physical NUMA](../benchmarks/numa/README.md) and
  [session budget](../benchmarks/economics/SESSION_BUDGET.md).

Performance requires matched bytes, CPU/build, sampling, cache and token lengths,
all five ABBA cycles and predeclared A/A noise. Judge C4/C4A × c=1/c=4 separately.
Promotion requires max(3%, A/A noise), with additional 1.15× MTP decode or +15%
48-core NUMA decode, and no TTFT/completion/throughput/memory regression. K=2
remains preferred if K=3 has no significant advantage. No such result is claimed.
AUTO profiles remain zero; unsupported/unknown profiles select NONE.

The recorded read-only GCP check (2026-10-25 13:36 UTC) found billing disabled
on both configured projects and Compute API disabled on the primary project.
Paid validation spending is $0; the cap is $50 with $10 reserved for billing
lag/disk/transfer/price changes. Actual GPT-OSS, physical NUMA/122B, C4/C4A and
native Arm gates remain unverified. If mandatory checks remain incomplete on
D21, record release readiness as incomplete and the release delayed.

## Local evidence ledger names

These are local ledger identifiers, not public download promises. Archive raw
identities before publishing any claim; use the relative reproduction links above.

- `v02-integrated-4ad6-artifact.json`, `v02-json-4ad6-asan-report.json`;
  `v02-mtp-json-4ad6-08b-report.json`, `v02-mtp-json-4ad6-2b-report.json`;
  `v02-4ad6-sampling-commands.json`, `v02-mtp-sampling-4ad6-matrix-report.json`
  and each `v02-mtp-sampling-4ad6-{08b,2b}-{c1,c4}-report.json` retain raw hashes,
  exact pins, counter/stop log references and per-cell failures. The JSON probe
  requires a source-bound unit-qualified artifact/report; see its reproduction link.
- `v02-sampler-db988-report.json`, `v02-sampler-db988-native-results.json`,
  `v02-sampler-db988-focused-results.json`, `v02-sampler-db988-audit.log`;
  `v02-sampler-db988-audit-receipt.json` confirms the later no-work audit on
  matching source/test binary. `v02-sampler-db988-asan-report.json` records
  the selected critical sanitizer run and final actual wrapper audit.
- `v02-integrated-143-artifact.json`, `v02-integrated-143-native-results.json`,
  `v02-followup-3ed-asan-report.json`, `v02-followup-804-python-all.log`,
  `v02-followup-personal-stdlib-verified.log`.
- `v02-python-generation-integrated143/result.json` and `manifest.json`.
- `v02-h-integrated143-c8-ctx512/analysis.json`, and the sibling
  `v02-h-integrated143-c8-ctx512-cache0-control` /
  `v02-h-integrated143-c8-ctx512-cache0-fixed8-control` native ledgers.
- `v02-mtp-sampling-stop5ad-matrix-report.json`,
  `v02-stop5ad-native-stop-correlation.json`.
- `v02-gemma-e2b-ad1-native-comparison.json`,
  `v02-gemma-e2b-e966-projection-inputs-comparison.json`,
  `v02-gemma-d408-native-isolation-summary.json`,
  `v02-gemma-e2b-658-ple-inputs-verified.json`.
- `v02-packaging-admission386-reviewed-components.json`,
  `v02-packaging-kind-admission386/summary.json`;
  prior `v02-packaging-canonical7ec-reviewed-components.json` and
  `v02-packaging-readiness-f50-reviewed-components.json` failures remain retained.
- `v02-correctness-canonical-7ec-results.json`,
  `v02-mtp-canonical7ec-matrix-report.json`,
  `v02-canonical7ec-qwen3-off-strict-number.json`,
  `v02-cloud-current-readonly-summary.json`.
