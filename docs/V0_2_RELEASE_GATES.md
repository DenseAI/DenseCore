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
`143a2ad5d87cd96d654e51b0e411d1da28eefaec`. Its full native run executed 1312
checks: 1298 passed, the same 14 existing skips, zero failures and zero disabled
tests. Go all-package checks and three repeated race runs passed. The Python SDK
suite passed 164 checks with 27 existing optional skips on unchanged SDK modules;
61 standard-library benchmark tests passed.

Real Python generation on the exact integrated `143a2ad` library passed 12 cases
and three native-token controls, covering completion counts/finish reasons,
concurrent metadata, cancellation of the actual native request ID, iterator/task
cleanup, incomplete stop-prefix flush and recovery. Completion accounting is
native-backed. Prompt-token accounting remains a client estimate (zero when no
Python tokenizer is available), not native prompt accounting.

The expanded ASAN/LSan result belongs to **`3ed5428`**, not full source `143a2ad`:
662 checks, 653 passed, the same 9 existing skips, zero failures/disabled tests
and zero sanitizer reports with leak detection enabled. It covers ordered
terminals, stop output, Gemma per-request attention and Q4 KV changes; subsequent
`83e0307` diagnostic user-data ownership changes are outside that run.

| Track | Verified evidence and source | Remaining gate |
|---|---|---|
| A | Integrated `143a2ad` native/Go checks and real Python lifecycle checks above; `3ed5428` scoped sanitizers | Actual c=8 greedy parity and Gemma concurrent parity still fail; final combined sanitizer/model qualification remains required |
| B/H | Resource/planner fixtures; earlier 4 GiB serving and explicit 1 GiB rejection without OOM kill; historical `7ecaf86` embedding AUTO/LAST 12 bitwise C/Python/HTTP comparisons | Actual `143a2ad` c=8 resource/cleanup evidence passes, but request isolation does not; final image/platform checks remain |
| B Gemma | Historical `7ecaf86` E2B c=1 long/repeat HTTP quality 6/6; private `ad1aa84` concurrent parity 6/8 | Two requests differ from c=1 even with cache off. Projection diagnosis is incomplete; E4B canonical/mixed qualification is held |
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

Gemma E2B private `ad1aa84` matches c=1's 127 native IDs in six of eight concurrent
requests; admission-rank 2 requests differ (121 IDs, first difference 23). Cache
OFF reproduces the same outputs. Paired debug `e9662e5` observes bit-identical
projection inputs and the first difference in Q4_K query projection output
(maximum absolute difference `6.103515625e-05`). This narrows investigation;
it is not a completed fix or model-support result. E4B canonical and mixed-model
runs remain held behind the isolation defect.

Private stop candidate `5ad54a5` completed 72 native requests, eight native
cancel/recovery cases and eight HTTP disconnect/recovery cases. Its 32 configured
literal-stop probes leaked no stop markers; native logs observed 27 actual
`stop_sequence` finishes. Native-ID-to-HTTP case mapping at c=4 is unproven, so
these counts are aggregate evidence. The 0.8B/2B × c=1/c=4 stochastic profiles
all retain strict FAIL: six HTTP quality failures among 160 main requests.
French formatting and stop-instruction failures are preserved without assigning
them to model or engine by assumption. These samples prove neither population
distribution equivalence nor acceleration. They are private-source evidence,
not a completed matrix for integrated `143a2ad`.

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
  `v02-gemma-e2b-e966-projection-inputs-comparison.json`.
- `v02-packaging-admission386-reviewed-components.json`,
  `v02-packaging-kind-admission386/summary.json`;
  prior `v02-packaging-canonical7ec-reviewed-components.json` and
  `v02-packaging-readiness-f50-reviewed-components.json` failures remain retained.
- `v02-correctness-canonical-7ec-results.json`,
  `v02-mtp-canonical7ec-matrix-report.json`,
  `v02-canonical7ec-qwen3-off-strict-number.json`,
  `v02-cloud-current-readonly-summary.json`.
