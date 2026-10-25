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
`db9883aabeb22ed450681ea2ddcb4d0771d99526`. Its full native run executed 1335
checks: 1321 passed, exactly the same 14 existing skips, zero failures and zero
disabled tests; 108 focused checks passed. The initial dependency audit relinked
the test executable, so its successful exit alone is not a no-work audit claim.
The recorded executable SHA remained unchanged (`935d4e946718ea3e0a5f0c7449f5385122b4eb3313ec0d41a68ac2e0f2d896e0`).
A subsequent no-work audit was confirmed against the same source and executable
SHA in the supplemental audit receipt. Both observations are retained; this
qualifies only the native test artifact, not a serving library/server or model.

The shared sampler now treats top-k=0 as disabling that filter without forcing
greedy selection, conditions nucleus filtering
on candidates retained by prior filters, and intersects grammar with global
whitelists/disallowed tokens and action ranges. A full-worker **mock-native**
regression observed the unsatisfiable-grammar failure before the fix, then
NONE-decode request-local error cleanup with successful peer and recovery on the
corrected `4545e05`/`db9883a` lineage. This is not real-model HTTP or Python proof.
The earlier `4545e05` focused run still failed a legacy sampling-reference test;
`db9883a` corrects the conditional reference and passes all 108 focused checks.

Go all-package checks and three repeated race runs, Python SDK 164 checks with
27 existing optional skips, and 61 standard-library benchmark tests belong to
prior integrated `143a2ad` (the SDK suite used unchanged SDK modules). They are
not new `db9883a` adapter or serving qualifications.

Real Python generation on the exact integrated `143a2ad` library passed 12 cases
and three native-token controls, covering completion counts/finish reasons,
concurrent metadata, cancellation of the actual native request ID, iterator/task
cleanup, incomplete stop-prefix flush and recovery. Completion accounting is
native-backed. Prompt-token accounting remains a client estimate (zero when no
Python tokenizer is available), not native prompt accounting.

The expanded critical ASAN/LSan suite now also qualifies exact source
**`db9883a`**: 688 checks, 679 passed, exactly the same 9 prior skips, zero
failures/disabled tests and zero ASAN/LSan reports with leak detection enabled.
It includes composed filters, disabled top-k, grammar intersections, typed
request isolation and raw diagnostics. Testing used the final
`0cd90110583e91c8638d3b8ab6d6a3cdc2eb3509b992e607fe5f95397e065339` executable
after a wrapper relink; the final actual wrapper audit was no-work with unchanged
source, header and tested binary. This is a selected critical suite, not all
1335 normal tests or a real-model test. The earlier `3ed5428` 653-pass/9-skip
sanitizer archive remains separate. Unmerged Gemma PLE functional changes,
private observers and pending JSON completion work are outside this result.

| Track | Verified evidence and source | Remaining gate |
|---|---|---|
| A | Integrated `db9883a` full native/focused checks and 679-pass/9-skip critical ASAN; prior `143a2ad` Go/Python lifecycle | Prior actual c=8/Gemma parity failures remain unresolved; future functional fixes require new sanitizer/model qualification |
| B/H | Resource/planner fixtures; earlier 4 GiB serving and explicit 1 GiB rejection without OOM kill; historical `7ecaf86` embedding AUTO/LAST 12 bitwise C/Python/HTTP comparisons | Actual `143a2ad` c=8 resource/cleanup evidence passes, but request isolation does not; final image/platform checks remain |
| B Gemma | Historical E2B c1 and private `ad1aa84` checks retained; private `d4088c7` cache-ON c1 repeats pass but c4 parity is 6/8 | Rank-0 requests differ from c1. Cache-OFF/mixed/E4B are not run on d408; F32 per-layer projection fix is pending |
| C | Historical `7ecaf86` README-pinned 0.8B/plain 2B combined prefix checks passed c=1/2/4, five-turn answers, restore/skipped prefill, cancellation and eviction | Do not inherit those results across later numerical changes; Qwen3.6, matched performance and default-policy promotion remain separate |
| D | Historical `7ecaf86` 0.8B/2B K2 c1/c4 native 128-ID parity; private `5ad54a5` native completion/cancel/recovery and literal-stop checks | All four private stochastic profiles fail strict HTTP quality (6 failures/160). Final-root matrix, population distribution, speed, 35B MoE and automatic qualification remain absent |
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
outputs (about `4.77e-7` / `2.38e-7`). The observer is not merged into this root,
and a separate F32 fix is pending. Diagnostic reproduction does not establish
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
not a completed matrix for integrated `db9883a` or its new shared sampler.

A separate MTP JSON completion gap is source-confirmed: completed grammar
permits whitespace, while common token commit does not yet finish the request
at the closing token. A separate implementation is pending. Earlier stochastic
cases without JSON fields do not qualify actual JSON generation; new combined
native, HTTP and Python JSON/finish checks remain required.

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
