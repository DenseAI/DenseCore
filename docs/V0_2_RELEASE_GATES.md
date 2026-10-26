# v0.2 implementation and release gates

D0 is 2026-10-25; D21 is 2026-11-15. Scope is A–H: personal PCs,
resource-limited Kubernetes pods and two-socket NUMA. DeepSeek V4/V4.1 is
excluded. Release readiness remains incomplete.

Implementation, real-model correctness, matched performance and automatic
activation are separate gates. A missing or truncated final answer fails even
when reasoning contains the expected text. No automatic MTP profiles or new
performance/default promotions are qualified.

## Evidence boundaries

## Latest source and portable package (`07fe153`)

Latest reviewed runtime source is `07fe15377f6c98cd21e4072b6b98b1a2bb6c743a`;
`ef9fc61` model evidence below remains a separate source/profile archive.
On `07fe153`, native tests are 1378/1364 passed/14 existing skips; portable
1378/1354 passed/24 skips, exactly the prior set plus ten unavailable compiled
packed-Q4/Q5 cases. No Float case is skipped. Portable focused strict ASAN passes
49/49 with actual mapped dependencies; independent float-fix review passes.
The local portable amd64 image `sha256:95f8e7e70e157222c652611db2deea10f4d081b48305794456f14b183b4bd2db`
passes initialization/Paris, active drain and explicit 1 GiB rejection, but strict
stream Count 1-to-2 returns `2` and **fails**. Kind chart/downloader/checksum/
limits/drain/cleanup components pass; final-answer quality is outside that gate
and `release_qualified` remains false. No image was published.
Portable `07fe153` E4B sustained c1 passes its own 246-ID EOS answer. Its c4 run
fails the unchanged 300-second request deadline with cancellation/incomplete
waves; no retry or context reduction was made. Native `ef9fc61` 254-ID c4 success
is not inherited. A separate amd64-v3 build is pending, with no qualification.
On `ef9fc61`, Qwen3 six-phase regression checks pass but strict HTTP number-only
formatting remains FAIL (4/4 off-mode arithmetic answers). Pinned plain 0.8B/2B
prefix main/pressure/stream HTTP gates pass, including 12 actual hits/restores per streaming model and
five positive client TTFT samples each; this is not a performance comparison.
ARM, physical NUMA, C4/C4A, actual GPT-OSS, MTP performance/auto, and final
publication remain incomplete; automatic profiles, paid spend and publication
remain zero. Final portable/source-matched model and prefix checks stay separate.

Local evidence: `artifacts/portable-float-fix/final-summary.json` SHA
`bbcea9dd0463df7d67c6f592973a0b9331805ca11c50d0af25febd35759e7ccf`;
independent review SHA `f11621f20d5eebd6079a0dc13959abbed3f8fe5cb47bef4b3cb8de93fbac8ec2`.
The `package-07fe153-20261026T083750Z/final-package-seal.json` SHA is
`dd4f505412fd1c2345391d84a21a58f35d95afc8976216170df677ae51c6ae77`;
its kind component summary SHA is
`14ee13052c2753746fc28842c57df96021e450ecc71395dc8de19baaaa545258`.
Portable E4B c1/c4 receipts retain their native hashes and own baseline.

## Frozen native source qualification (`ef9fc61`)

The frozen runtime source is `ef9fc61d6dde68bed4f5d6fb72cd52e9afa58b6c`.
The immutable local manifest `v02-integrated-resource-isolation-artifact.json`
has SHA-256 `2fc9bc7640a04aad7796c4a8022828170fbbc31351bdb9071cdc9c10c9e0a5a0`.
Native: 1377 run / 1363 passed / the same 14 skips; selected strict ASAN/LSan:
761 run / 752 passed / the same 9 skips, zero failures/disabled/sanitizer reports.
All prior 727 sanitizer cases and the added required cases execute. Policy 38,
Go all and targeted race count=3 pass. SDK 164 passed / 27 optional skips is a
separate cached-UV run, outside the sealed manifest; no personal-test result is
inherited. SDK unit checks do not qualify all Linux loader contexts.

The initial library and ASAN audits successfully relinked before their respective
tests; they are historical build steps, not failed test runs. Final library/tests
and ASAN audits are no-work. ASAN report V2 binds the executed binary and actual
mapped dependencies to final audit snapshots/current bytes; the preliminary
before-relink binary hash is explicitly unqualified.

Actual H gates pass in order: original200 c1 and eight c8 requests each retain
51-token EOS/raw-ID parity; separate80 c1 reaches the 128-token LENGTH limit;
full80 has 37 recorded requests and a pressure snapshot active=8/pending=8.
These are bounded native correctness/admission checks. Original200/80c1 use
cache-OFF/fixed8; full80 retains cache-ON/adaptive settings. Their differences do
not support a causal or performance comparison, or a complete 80-number answer.
Earlier `143a2ad` and private `613a` failures remain historical evidence.

Pinned E4B now initializes at unchanged context 2048, capacity 4, threads 8,
Q8 KV and NONE speculation: required 3,468,717,568 <= available 3,505,483,776 plan
bytes, peak whole-child RSS 6,240,676 KiB. No capacity/context reduction or retry
was used. A separate c1 generation gate passes one 59-token EOS answer with all
five planets and DONE; two subsequent repeats each match its 59-token EOS IDs
and complete answer. This short-answer gate is separate from sustained generation,
HTTP, package, broad support, minimum-RAM or performance qualification.
Same-source pinned 0.8B and 2B greedy/native/HTTP MTP pass. The `48dd55a` hard-mask and
`4ad6a35` JSON/probabilistic results below keep their original sources.
Automatic MTP profiles remain zero; no performance/NUMA/GPT-OSS promotion follows.


Current `ef9fc61` E4B c4 now passes two waves of four requests: all eight
retain the own-c1 59-token EOS IDs and complete five-planet/DONE answer.
This is native request isolation, not mixed physical graph or sustained
c4 generation. The pinned 0.8B MTP artifact
`ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
passes c1/repeat1 and c4/repeat2 greedy NONE/MTP exact-128 native parity,
actual accept/reject/rollback, cancel/recovery and HTTP
stream/nonstream/disconnect checks. It is separate from README's `f5b14d…` file.
Portable-package and fresh-source prefix results remain pending; no performance, automatic
activation, package or broad support qualification follows.

The same pinned 2B MTP file also passes c1/repeat1 and c4/repeat2 native
exact-128 greedy NONE/MTP parity and HTTP/lifecycle checks. These are explicit
MTP checks; automatic profiles remain zero.
A separate README-file native default-policy smoke passes: derived context
23,338, threads 16, concurrency 1, AUTO selects NONE, and the short Paris answer
finishes at 8-token EOS with Free/source/mapping pins. This does not qualify
HTTP, an SDK wheel or a full-23k-context workload.
Offline V3 qualification of the existing sustained E4B c1 capture passes:
254 native IDs, EOS, 216 body words and five ordered three-sentence paragraphs.
The independently reviewed correction permits interior cross-planet references;
original raw FAIL remains immutable, and reevaluation sends zero new requests.
Sustained c4 V3 now passes all eight 254-ID EOS requests against its qualified
own-c1 baseline, retaining five ordered paragraphs, three sentences each and
216 body words. This is native isolation; HTTP/package/support remain separate.
The new four-cell stochastic main strict gate passes 160 HTTP requests with zero
quality failures, 72 native final requests, eight cancellations and eight HTTP
disconnect/recoveries. All 32 stop probes suppress literal markers; 31 finish by
actual string stop. Two separate exact-stop-instruction negatives remain:
0.8B c1 NONE is verbose and 0.8B c4 MTP returns `Paris ` with a trailing space.
Historical `4ad6a35` French-punctuation/stop failures are preserved. Unseeded
cross-run results establish no causal, distribution, speed or automatic claim.

Prior `4ad6a35` native 1347/1333/14 and ASAN 702/693/9 remain source-labeled,
including its JSON model passes and probabilistic overall FAIL below. They do
not automatically transfer to `ef9fc61`. Earlier `db9883a` native 1335/ASAN 688 and
`3ed5428` sanitizer archives remain preserved as well.

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
| A | Integrated `ef9fc61` native 1363-pass/14-skip, selected ASAN 752-pass/9-skip, Go/race and policy 38 pass | SDK TLS/package contexts remain limited; fresh-source prefix and final package qualification pending |
| B/H | `ef9fc61` original200 c1/c8 51-EOS parity, separate80 c1 128-LENGTH, full80 37-request gate PASS with active8/pending8 | Native bounded correctness/admission only; final image/platform and matched performance remain |
| B Gemma | `ef9fc61` E4B unchanged ctx2048/cap4/t8/Q8/NONE init PASS and complete 59-EOS c1/repeats/c4 and sustained 254-EOS c1/c4 own-ID gates; historical `48dd55a` E2B grid retained | HTTP/package/support/performance remain pending |
| C | Historical `7ecaf86` README-pinned 0.8B/plain 2B combined prefix checks passed c=1/2/4, five-turn answers, restore/skipped prefill, cancellation and eviction | Do not inherit those results across later numerical changes; Qwen3.6, matched performance and default-policy promotion remain separate |
| D | Current `ef9fc61` pinned 0.8B/2B greedy native/HTTP MTP c1/c4 and four-cell stochastic main strict PASS; two stop-instruction negatives retained. Historical root `48dd55a` narrow hard-mask NONE/MTP lifecycle PASS. Source `4ad6a35` JSON both-model PASS and probabilistic rejection/rollback/stop evidence retained; historical `7ecaf86` greedy parity separate | Source `4ad6a35` probabilistic matrix overall FAIL: one 0.8B c4 French punctuation failure plus separate stop-instruction negative. No inheritance to `48dd55a`; distribution/speed/35B/auto absent |
| E | Go-only `386b800` portable amd64 image passed actual container and kind strict active drain, resource limits, checksum rejection and safe memory refusal | Strict Count quality still fails. Current root core has no final image validation; original init-log capture has syntax checks only; native Arm remains unverified |
| F | GPT-OSS packed MXFP4/bias/router/activation, attention sink/window and tokenizer/Harmony fixtures | Actual HTTP/long/c=1/4/stop/stream inference and supported peak RSS/context remain unverified; safe capacity refusal is not model support |
| G | Public physical-NUMA preflight and matched ABBA harness | Actual 35B/122B two-socket measurements, 16/48 physical cores and remote-memory penalty; no promotion |

## Historical correctness failures

Actual H resource testing on `143a2ad` observed a pressure snapshot with 8 active
and 8 pending requests (maximum observed pending 12). Oversized prompt admission
returned an explicit memory error with no callbacks; all 36 accepted requests
published one terminal and one completion. These pass bounded resource/lifecycle
checks, not isolation or throughput qualification. Greedy parity fails: the last
admitted c=8 request replaces baseline EOS at zero-based token position 50 with a
newline and continues to the 128-token cap. Separate cache-OFF and cache-OFF with
all thread policies fixed to 8 reproduce the failure. This does not establish a
physical-slot cause or qualify either control as a workaround.

Private H candidate `613a` also remains FAIL: three of eight c8 requests
replace the original 51-token EOS boundary and continue to the 128-token cap in
the fixed 8-thread/cache-OFF control. It is not merged into `48dd55a`; no H fix, workaround,
quality or performance promotion follows.

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

Private `a9efe78fec43c25f29510098a17f61cbe8dd9f55` subsequently passes the
E2B formal cache-ON/OFF c=1/c=4 grid: 20 complete native requests all match the
original 127-ID c1 baseline and EOS. With context2048, capacity4, Q8 KV, NONE,
prefix OFF and requested threads8, a separate four-request staggered test also
matches each case's own c1 raw IDs, complete answer and EOS. Submission timing
proves overlapping request lifetimes, not a mixed physical graph. These actual
private-source passes coexist with the earlier ad1/d408 failures; neither is
silently substituted for the other or inherited by root `4ad6a35`.

On the same private a9 source, E4B context 2048/capacity 4 initialization fails explicitly
before generation: required plan 4,532,711,424 bytes versus available 3,428,679,680.
No request or c4 wave ran, and that profile was not reduced or retried. A separately
requested PC context 2048/capacity 1 profile passes one native long-answer case with
59 sampled tokens and one STOP/terminal/completion. It is not a cap4, repeat,
cache-OFF, HTTP, supported-RAM-range, broad-model or performance qualification.
The capacity failure remains preserved; successful cap1 is a distinct profile.

Actual root `48dd55a` requalifies the E2B formal 20 cache-ON/OFF c=1/c=4 grid:
all match the old 127-ID baseline and EOS. Four additional owned-case c1 baselines
and four staggered requests also pass; each staggered case matches its own c1
IDs, complete answer and EOS. The formal 20 plus staggered 4 are the requested
isolation cases, with four extra baseline captures (28 captured requests total).
Overlapping lifetimes do not prove a mixed physical graph. No HTTP, arbitrary
Gemma support, default or performance qualification follows from this grid.

The SAME pinned E4B file on root `48dd55a`, context 2048/capacity 4, fails initialization:
plan required 5,921,446,144 bytes > available 4,681,576,448 bytes. Generation and
native input/EOS preparation are not reached. Failed-init whole-process peak
RSS is 6,236,108 KiB. Neither that highwater nor its difference from prior private
runs establishes RAM savings or minimum/sufficient RAM for successful serving.
Private a9 capacity 1 success remains a separate source/profile, not a root `48dd55a` pass.

Actual root `48dd55a` hard-mask probe V2 passes NONE/MTP: strict allow 16/deny 16,
JSON-off, temperatures 0/0.7 yield four zero-commit ERROR requests with one
metadata and terminal each while healthy peers are active. Six healthy peer/
recovery requests finish normally. Healthy MTP controls have actual width 3
proposal/verification/accept/reject/rollback evidence. Failing MTP requests
have no successful proposal-verification telemetry; do not infer head work from
absent logs. V1 failed at ctypes static-TLS loading before any engine/model
initialization and remains UNEXECUTED. Reviewed V2 changes only verified-library
startup preload, preserves the same inputs/settings/validators, and proves
actual engine load/free, process exit and complete EOF. This is lifecycle,
not final-answer quality, population distribution, speed or default promotion.

Private stop candidate `5ad54a5` completed 72 native requests, eight native
cancel/recovery cases and eight HTTP disconnect/recovery cases. Its 32 configured
literal-stop probes leaked no stop markers; native logs observed 27 actual
`stop_sequence` finishes. Native-ID-to-HTTP case mapping at c=4 is unproven, so
these counts are aggregate evidence. The 0.8B/2B × c=1/c=4 stochastic profiles
all retain strict FAIL: six HTTP quality failures among 160 main requests.
French formatting and stop-instruction failures are preserved without assigning
them to model or engine by assumption. These samples prove neither population
distribution equivalence nor acceleration. They are private-source evidence,
not the later `4ad6a35` matrix described below.

Integrated `4ad6a35` closes the JSON completion gap: ordinary and MTP output
commit the root-closing token before STOP; fixtures cover an exact max-token
boundary. Actual model probes below use a 512-token output budget.
Actual pinned 0.8B and 2B each pass 16 main native and 32 HTTP checks with NONE/MTP,
c=1/4, greedy K2, context2048 and threads8. Strict brace-whitelist `{}` and
ordinary `{"answer":42}` cases, native cancel/recovery, HTTP stream/nonstream/disconnect/
recovery and UTF-8/usage/terminal checks pass. Each model records 54 actual
`grammar_complete` finishes and batched target width 3 with acceptance/rollback.
JSON rejection was not observed; formal rejection fixtures and probabilistic
coverage remain separate. Delivered JSON CallbackEx IDs are a subset of native
samples; real Python JSON and public global action range are not qualified.
Two tiny objects do not establish arbitrary JSON/model support. c=4 denotes
logical engine capacity; width 3 is one request's K+1 verification graph, not
four-sequence physical batching or a throughput qualification.

The unchanged non-JSON probabilistic matrix on the same `4ad6a35` artifact is
**overall FAIL**. 0.8B c1 and 2B c1/c4 pass; 0.8B c4 fails one streamed NONE French
answer missing final punctuation (normal STOP/DONE and usage32, budget1024).
A separate NONE stop probe also fails the exact Paris instruction; the existing
format side metric does not include that probe. No engine/model cause is proven.
All 72 native finals including recovery, eight native cancellations and eight
HTTP disconnect/recoveries pass. Of 160 main HTTP requests, one fails the final
quality gate. All 32 stop probes suppress the literal marker and 31 actual
`stop_sequence` finishes are logged. All MTP profiles observe width 3,
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

## Linux SDK loading boundary

Matched `4ad6a35` and `48dd55a` SDK runs in the recorded interpreter/NumPy environment
show file execution succeeds via exactly one automatic execve restart. Both
`-c` and stdin fail with the same static-TLS OSError; this is not established
as a new `48dd55a` regression. These probes call get_library/symbol-owner checks only,
not an engine or model, and do not generalize to all user environments.

Private source `a70cd4e23892947d7391e9674f369ea6527f35ee` Linux dynamic-TLS
candidate passes three SDK loader modes without execve. All three raw ctypes
modes load but SIGSEGV during process teardown, so the candidate is unqualified
and unmerged. Exact allocator provenance records a 31-byte Registry allocation
through libc malloc/global libstdc++ new, followed by DenseCore sized delete
reaching mimalloc during destruction. SDK loader passes do not waive that
raw-process failure or establish model support. No engine/model was called.

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

- `artifacts/portable-float-fix/{final-summary,final-skip-identity,independent-source-review-07fe153}.json`.
- `artifacts/package-07fe153-20261026T083750Z/{final-package-seal,container-qa-receipt,local-image-receipt}.json`
  and `kind-qa/summary.json`; container strict Count FAIL is retained.
- `v02-gemma-portable-07fe-{c1,c4}/receipt.json`: portable own246 c1 PASS,
  c4 deadline/cancel FAIL, separate from native EF9 own254.
- `v02-qwen3-resource-isolation-matrix.json`,
  `v02-prefix-resource-isolation-{08b,2b}/all-qualification.json`.
- `v02-integrated-resource-isolation-artifact.json` (SHA above),
  `v02-resource-isolation-asan-report-v2.json`,
  `v02-integrated-resource-isolation-execution-receipt.json`,
  `v02-integrated-resource-isolation-sdk.log` (separate cached-UV run).
- `v02-h-resource-isolation-original200-c8-ctx512-cache0-fixed8-control/phase-gate.json`,
  `v02-h-resource-isolation-80-c1-pressure/phase-gate.json`,
  `v02-h-resource-isolation-80-full-ctx512/phase-gate.json`.
- `v02-gemma-resource-isolation-e4b-cap4-init-only/result.json`,
  `v02-gemma-resource-isolation-e4b-quality-c1/quality-gate.json`,
  `v02-gemma-resource-isolation-e4b-quality-repeat/quality-gate.json`,
  `v02-gemma-resource-isolation-e4b-quality-c4/quality-gate.json`.
- `v02-mtp-resource-isolation-{08b,2b}/receipt.json` and pinned c1/c4 summaries.
- `v02-mtp-sampling-resource-isolation-matrix-{receipt,report}.json`;
  `v02-gemma-resource-isolation-e4b-quality-sustained-c4-v3/quality-gate.json`.
- `v02-resource-isolation-default-auto-native-readme/result.json`;
  `v02-gemma-resource-isolation-e4b-quality-sustained-c1-v3/qualification.json`
  binds independent review SHA `c1c7c7c092a310fda03eca22fb1b7e468e0085eb94a04a38a5f4056f3c3f6a28`
  and preserves original raw FAIL/native hash `886bbe0ed79ae0d9760292b4c68a691d671502d99bcf7341bda495343c5e6e46`.
- Historical `v02-integrated-48dd-artifact.json`, `v02-staging-48dd-asan-report.json`;
  immutable manifest SHA is recorded above. `v02-hard-mask-48dd-v2-report.json`
  binds reviewed local V2 tool/script hashes, callbacks, owners and process receipts;
  `v02-hard-mask-48dd-infrastructure-failure.json` preserves V1 UNEXECUTED.
- `v02-gemma-48dd55-native-isolation-summary.json`,
  `v02-gemma-48dd55-e4b-capacity-result.json` retain the root `48dd55a` model boundaries.
- `v02-sdk-tls-loader-matched/report.json`, `v02-linux-dlopen-a70-report.json`,
  `v02-linux-dlopen-a70-raw-allocation-provenance.log`, and
  `v02-h-613a-original200-c8-ctx512-cache0-fixed8-control/analysis.json` retain
  matched loader failures and unmerged candidate negatives.
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
- `v02-gemma-a9-native-isolation-summary.json`,
  `v02-gemma-a9-staggered-isolation-summary.json`,
  `v02-gemma-a9-e4b-capacity-result.json`, `v02-gemma-a9-e4b-pc-cap1-result.json`
  bind the private `a9efe78` actual cases, capacity failure and separate cap1 pass.
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
