# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
The latest reviewed runtime source is `1ccca905`; historical private-model and earlier
package evidence below retain separate identities.

Intended deployments are personal PCs, resource-limited CPU Kubernetes pods and
two-socket NUMA. Candidates add resource budgeting, checked model artifacts,
metadata-driven pooling, request/prefix ownership, speculative verification and
native-backed Python completion metadata. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md).

The latest local review candidate includes source `1ccca9053b98473c84fe19f6d768f8486ef8ae7a`
(core tree `c827c2bcdca6fe11a846ae0098f5337c2641712a`). It releases the idle
fallback graph scratch at the completed-worker batch boundary before a separately
owned Gemma transient prefill arena, preserving the admitted graph budget,
independent cached graph owners, KV and scheduler state. Scaled exact-capacity
and ownership tests pass; current native is 1381/1367 passed/14 unchanged skips,
and selected strict ASAN is 148/148 passed, zero skips/errors.
A new native MI_OFF installed wheel passes 72 SDK/packaging tests and 12 actual
completion/lifecycle cases. New-source Gemma installed-DSO raw C API init, C1,
two sequential repeats and C4 two waves of four pass: all quality requests have
254 native IDs ending EOS, 216 body words, five ordered paragraphs of three
sentences, and exact fresh own-C1 IDs/render metadata with post-Free callbacks.
The fixed profile remains ctx2048/cap4/t8/Q8/NONE/max512/request300/lifetime7200;
no budget increase, retries, context reduction or old-source ID inheritance.
This is installed-DSO raw C API evidence, not Python SDK Gemma API, HTTP, new
server/image, broad support, minimum-RAM, RAM-savings or performance qualification.
Resource availability/profile differences prevent a cross-run memory-saving claim.

The prior installed `08c2848`/core `07fe153` C4 failure remains preserved: a
512 MiB metadata lease exceeded the 2368 MiB arena cap with 1920 MiB in use,
causing a worker exception and failed requests. The current same-budget handoff
algorithm and actual formerly failing workload pass; the old result is not erased.
Earlier portable C4 timeout, amd64-v3 Gemma init refusal, strict image Count and
Qwen3 formatting failures also remain source/profile-labeled. Current `1ccca`
performance and other-model regressions are UNKNOWN; the K2/K3 performance FAIL
measurements below belong to `07fe153` and do not transfer to the new source.
Automatic MTP profiles remain zero. Release readiness remains incomplete with
billing disabled and GPT-OSS/ARM/physical NUMA/C4/C4A/publication gaps unchanged.

Current `07fe153` embedding qualification passes on the original four texts.
Installed native MI_OFF SDK Python AUTO/LAST and same-handle C default/CEx-LAST
pass eight comparisons; separate amd64-v3 MI_ON HTTP omitted/LAST passes four.
Each pair has 1024 finite normalized elements and identical literal F32 bytes,
with the original norm/cosine/max-error thresholds retained. The old `48dd55a`
`array_equal` evidence was numeric equality and C-default/Python-LAST comparison;
it is preserved, not relabeled as this stronger same-interface byte proof.
Accepted C callbacks occur once, NULL submission has zero callbacks, SDK empty
batch returns shape `(0,)`, and accepted work drains before Free. HTTP rejects
empty inputs and numeric `pooling_type:17` with 400, then recovers; unknown strings
retain the source's AUTO normalization contract. Profiles are never cross-compared.

Current 2B-c1 K2 and K3 each complete all 20 fresh samples / five ABBA blocks,
with all blocks retained, but both are **FAIL_MATERIAL_REGRESSION**. K2 median
ratios (decode/completion-speed/aggregate) are
`0.8443473740423546 / 0.8399898177112242 / 0.8400002680001134`; K3 ratios are
`0.9604052183066641 / 0.9408523169856156 / 0.9408498530449593`.
The gate is max(A/A noise,3%) = 3%, plus the decode 1.15/nonregression criteria.
These are end-to-end production policies, NONE decode-cache ON versus MTP OFF,
on this host/profile, not isolated algorithm or cross-host causal results.
Direct K2-versus-K3 selection is NOT_RUN because both fail eligibility; auto stays 0.

Separate current 2B-c1 controlled NONE/K2 and fresh NONE/K3 stop proofs pass:
each mode has four baselines and eight matching/EOF-partial UTF8 stop responses,
exact text/API usage parity, observable terminal metadata once and four actual
native stop-sequence summaries. No native callback-once or unseen duplicate-DONE
claim follows. Current K3 stochastic integration also passes: 18 native final
responses, two cancellations, 40 main HTTP responses, two disconnect/recoveries,
28/28 separate strict-format checks and all four engine-mapping roles. This is
unseeded integration, not distribution equality, other-model/c4 or speed proof.
The 2026-10-26 read-only cloud preflight stops at disabled billing (exit 3);
inventory, prices and session estimates are NOT_RUN. Paid spend remains zero.
Release readiness remains incomplete; all earlier image/model failures persist.

The native MI_OFF installed wheel now passes 69 SDK contract tests (one
`asyncio_mode` configuration warning) and 12 actual Python/native completion,
stream/async/cancel/recovery cases. It is `densecore-0.1.0-cp313-cp313-linux_x86_64.whl`,
packaging source `08c2848` with core `07fe153`, not a published v0.2 wheel.
Initial/final actual process maps each bind 24 records to installed artifacts;
ordinary loading uses no preload/reexec or source fallback. This native-host ISA/
GLIBC profile is not manylinux, generic-CPU, portable-image or broad model proof.
Container strict Count and portable E4B c4 failures remain unchanged. MTP
performance V1 failed a precondition before measurement; final K2/K3 results
below are separate fresh runs and neither qualifies automatic activation.

Prior reviewed runtime source is `07fe15377f6c98cd21e4072b6b98b1a2bb6c743a`;
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
is not inherited. The separate amd64-v3 container also fails strict Count, while
initialization/drain/explicit memory rejection pass. Its E4B c1 run refuses
initialization at ctx2048/cap4: required 3,445,450,752 > available
3,319,349,248 plan bytes. No generation capture exists; cascading missing-capture
diagnostics are not semantic-quality failures. amd64-v3 E4B c4 is NOT_RUN.
Portable own246 c1 PASS/c4 deadline FAIL remain unchanged and separate.
On `ef9fc61`, Qwen3 six-phase regression checks pass but strict HTTP number-only
formatting remains FAIL (4/4 off-mode arithmetic answers). Pinned plain 0.8B/2B
prefix main/pressure/stream HTTP gates pass, including 12 actual hits/restores per streaming model and
five positive client TTFT samples each; this is not a performance comparison.
ARM, physical NUMA, C4/C4A, actual GPT-OSS, untested MTP profiles/automatic activation, and final
publication remain incomplete; automatic profiles, paid spend and publication
remain zero. Final portable/source-matched model and prefix checks stay separate.

- Integrated `ef9fc61`: native 1377/1363 passed/14 existing skips; selected
  ASAN/LSan 761/752 passed/9 existing skips, zero failures/reports. Go all,
  targeted race count=3 and policy 38 pass. Final actual audits are no-work;
  initial pretest relinks are historical build steps. ASAN V2 binds executed
  bytes. Separate cached-UV SDK 164/27 optional skips is outside the manifest.
- Actual `ef9fc61` H passes original200 c1/c8 51-token EOS parity, separate80
  c1 128-token LENGTH and full80 37-request gate with active8/pending8.
  Cache-OFF/fixed8 and cache-ON/adaptive profiles stay distinct; no complete
  80-number-answer or performance claim follows.
- Actual `ef9fc61` E4B unchanged ctx2048/cap4/t8/Q8/NONE init passes:
  required 3,468,717,568 <= available 3,505,483,776 plan bytes; peak child RSS
  6,240,676 KiB. Separate c1 passes one complete five-planet/DONE answer at
  59-token EOS; two repeats match its IDs and complete answer. HTTP/package/support
  and performance remain pending. Same-source pinned 0.8B and 2B MTP greedy/native/HTTP requalification passes.
  Manifest SHA: `2fc9bc7640a04aad7796c4a8022828170fbbc31351bdb9071cdc9c10c9e0a5a0`.

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

- Historical `48dd55a` hard-mask NONE/MTP lifecycle passes: four contradictory-mask
  errors with zero output/generated tokens, six healthy peers/recoveries and
  exact terminal/metadata counts. Only healthy MTP requests show proposal
  verification. Original V1 static-TLS failure before engine init is UNEXECUTED;
  reviewed V2 corrects only qualified-library startup preload.
- Historical root `48dd55a` E2B formal 20 cache-ON/OFF c=1/c=4 plus four staggered requests
  match old/own c1 IDs and complete answers; no mixed physical graph is claimed.
  E4B context 2048/capacity 4 init FAIL (plan 5,921,446,144 > 4,681,576,448 available bytes),
  zero generation, failed-init RSS 6,236,108 KiB. This is not RAM savings or a
  successful supported-RAM estimate. Earlier private a9 capacity 1 remains separate.
- Private H `613a` still fails three of eight c8 comparisons and is unmerged.
  Private Linux TLS `a70cd4e` passes three SDK loader modes without execve but
  all three raw modes SIGSEGV during teardown (allocator provenance); unqualified
  and unmerged. Matched `4ad6a35`/`48dd55a` SDK file mode restarts once, `-c`/stdin fail
  static TLS. These loader probes invoke no engine/model.
- Shared-sampler fixes cover disabled top-k, conditional nucleus filtering and
  grammar intersections. Mock-native full-worker NONE error/peer/recovery checks
  pass. Historical `48dd55a` Go all/race checks, SDK 164 unit checks/27 optional skips and
  88 standard-library benchmark tests passed; they are not actual Python JSON proof.
- Source-labeled `4ad6a35` pinned 0.8B and 2B JSON NONE/MTP c1/c4 native and HTTP probes
  pass: each model has 16 main native/32 HTTP requests, native cancel/recovery and
  HTTP stream/nonstream/disconnect/recovery, plus 54 actual grammar-complete
  finishes. Two tiny object cases and a strict brace-ID whitelist are the scope;
  public global action range and real Python JSON are unqualified. These model
  passes do not automatically transfer to source `48dd55a`.
- The unchanged probabilistic matrix on `4ad6a35` is **overall FAIL**: 0.8B c1
  and 2B c1/c4 pass; 0.8B c4 fails one NONE French final-punctuation check among
  160 HTTP requests. One separate NONE stop-instruction negative is retained.
  All 32 stop probes suppress the marker; 31 actual string-stop finishes and
  actual MTP rejection/rollback are recorded. Native completion/cancel/recovery
  and HTTP disconnect/recovery pass. No random-ID/distribution, causal quality
  improvement, speed or automatic/default claim follows, and this failed matrix
  is not silently inherited or replaced by source `48dd55a` lifecycle probes.
- Real Python on integrated `143a2ad` passed 12 cases plus three raw native-token
  controls: completion counts/finish, cancellation, stream/task cleanup and
  recovery. Prompt counts remain client estimates, not native prompt accounting.
- Prior `143a2ad` actual c=8 resource testing observed active=8/pending=8,
  rejected oversized prompt admission without callbacks and finalized all 36
  accepted requests once. Greedy parity still fails at EOS token position 50, continuing to the
  128-token limit. Cache OFF and all-thread-policy=8 controls also fail.
- Historical `7ecaf86` pinned 0.8B/plain 2B prefix and embedding checks passed.
  Those results are not automatically inherited by later numerical changes.
  Qwen3 number-only formatting remains a recorded failure.
- Historical private `5ad54a5` stochastic profiles all remain FAIL: six HTTP
  failures/160 requests, with zero marker leaks/32 probes and 27 string stops.
  Those raw failures are preserved rather than replaced by current samples.
- Private Gemma `d4088c7` cache-ON c4 failed: rank-0 requests 1/5 produced 135 IDs
  against c1's 127, first difference index 52. Same-source cache-OFF, mixed and
  E4B matrices were not run. DEBUG `658e626` narrowed a surviving PLE delta;
  Private `a9efe78` subsequently passes 20 E2B cache-ON/OFF c=1/c=4 raw-ID checks
  and four staggered requests, each matching its own c1 complete answer/EOS.
  E4B context 2048/capacity 4 fails capacity before generation; a separately requested
  capacity 1 profile passes one native long case, without a capacity 4 retry or silent reduction.
  Q4/F32 fixes/fixtures are in `4ad6a35` native/ASAN. Private model outcomes do not
  transfer automatically to root4ad6, broad support, HTTP or performance/default gates.
- Separate Go-only `386b800` portable amd64 image passed actual container and
  kind strict drain: same-round readiness 503/new-admission 503/native active=1,
  accepted stream finish/DONE once, 7.596/7.903 seconds after signal commands
  within the original 30-second budget. Strict Count still fails. The changed
  root core has no final image qualification; original init-log capture has
  syntax checks only and awaits a real final kind run.

No new speed headline is supported. Historical TTFT gains do not establish
aggregate throughput or completion-time superiority. Automatic MTP profiles
remain zero; MTP/NUMA defaults are unpromoted. Actual GPT-OSS, physical NUMA,
122B, C4/C4A and native Arm remain unverified; no GPT-OSS PC RAM range is claimed.
The recorded GCP read-only check found both configured projects' billing disabled
and primary Compute API disabled. Paid validation spending remains $0/$50.

Before publication, identify exact source/image/model bytes, CPU/RAM/context/
concurrency, complete-answer and isolation outcomes, matched performance samples
and unsupported combinations. Local raw ledger names in the release gates are
not publicly available artifact promises. Mandatory failures at D21 mean release
readiness remains incomplete and the release is delayed.
