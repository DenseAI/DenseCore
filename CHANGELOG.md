# Changelog

All notable changes to DenseCore are documented here. This file records release
material and platform maturity. Comparative benchmark research is outside the
public source release.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### v0.2 candidates (qualification in progress)

- The latest local review candidate includes source `1ccca9053b98473c84fe19f6d768f8486ef8ae7a`
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
- Current `07fe153` embedding qualification passes on the original four texts.
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
- The native MI_OFF installed wheel now passes 69 SDK contract tests (one
  `asyncio_mode` configuration warning) and 12 actual Python/native completion,
  stream/async/cancel/recovery cases. It is `densecore-0.1.0-cp313-cp313-linux_x86_64.whl`,
  packaging source `08c2848` with core `07fe153`, not a published v0.2 wheel.
  Initial/final actual process maps each bind 24 records to installed artifacts;
  ordinary loading uses no preload/reexec or source fallback. This native-host ISA/
  GLIBC profile is not manylinux, generic-CPU, portable-image or broad model proof.
  Container strict Count and portable E4B c4 failures remain unchanged. MTP
  performance V1 failed a precondition before measurement; final K2/K3 results
  below are separate fresh runs and neither qualifies automatic activation.
- Prior reviewed runtime source is `07fe15377f6c98cd21e4072b6b98b1a2bb6c743a`;
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
- Current `ef9fc61` E4B c4 now passes two waves of four requests: all eight
  retain the own-c1 59-token EOS IDs and complete five-planet/DONE answer.
  This is native request isolation, not mixed physical graph or sustained
  >=128-token generation. The pinned 0.8B MTP artifact
  `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
  passes c1/repeat1 and c4/repeat2 greedy NONE/MTP exact-128 native parity,
  actual accept/reject/rollback, cancel/recovery and HTTP
  stream/nonstream/disconnect checks. It is separate from README's `f5b14d…` file.
  Portable-package and fresh-source prefix results remain pending; no performance, automatic
  activation, package or broad support qualification follows.
- The same pinned 2B MTP file also passes c1/repeat1 and c4/repeat2 native
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
- Requalify frozen `ef9fc61`: native 1377/1363 passed/14 existing skips;
  selected ASAN/LSan 761/752 passed/9 existing skips; Go all/race count=3 and
  policy 38 pass. Final audits are no-work; successful initial pretest relinks
  remain historical build steps. ASAN V2 pins executed bytes. Separate cached-UV
  SDK has 164 passed/27 optional skips, outside the sealed manifest.
- Actual `ef9fc61` H passes original200 c1/c8 51-EOS parity, separate80 c1
  128-LENGTH and full80 37-request gate with active8/pending8. Profiles retain
  their original cache/thread settings; no speed or complete80-answer claim.
- Actual `ef9fc61` E4B unchanged ctx2048/cap4/t8/Q8/NONE initializes and passes
  one complete five-planet/DONE c1 answer at 59-token EOS, plus two exact-ID
  complete-answer repeats. Init plan required
  3,468,717,568 <= available 3,505,483,776 bytes; peak child RSS 6,240,676 KiB.
  HTTP/package/support remain pending.
  Same-source pinned 0.8B and 2B MTP greedy/native/HTTP requalification passes; historical results below keep their own sources.
  Manifest SHA: `2fc9bc7640a04aad7796c4a8022828170fbbc31351bdb9071cdc9c10c9e0a5a0`.
- Add common Linux resource budgeting, context-preserving automatic concurrency,
  versioned diagnostics and embedding pooling AUTO. Actual integrated `143a2ad`
  c8 resource/admission/cancellation checks pass, but greedy EOS parity fails,
  including cache-OFF and fixed-thread-policy controls. No concurrency promotion.
- Isolate request RNG, retain prefix snapshot/KV ownership through restoration,
  terminate failed workers and order terminal publication. Native-backed Python
  completion counts/finish metadata and native-request-ID cancellation passed
  12 real generation/lifecycle cases plus three native-token controls on
  integrated `143a2ad`. Prompt counts remain client estimates.
- Add NextN loading, batched MTP verification and suffix/SSM recovery. Historical
  `7ecaf86` 0.8B/2B K2 c1/c4 greedy checks passed. Private `5ad54a5` literal-stop
  probes show zero leaks/32 probes and 27 actual string-stop finishes, while all
  four stochastic profile quality gates still fail (6/160 HTTP requests).
  These historical failures remain preserved. On integrated `4ad6a35`, pinned
  0.8B/2B JSON NONE/MTP c1/c4 native and HTTP checks pass. The unchanged
  probabilistic matrix passes 0.8B c1 and 2B c1/c4, but 0.8B c4 still fails one
  French final-punctuation check; a separate stop-instruction negative is retained.
  Across 32 configured stop probes there are zero marker leaks and 31 actual
  string-stop finishes; actual MTP rejection/rollback is observed. The overall
  probabilistic gate remains FAIL. No population-distribution, speed or automatic
  qualification follows; automatic profiles stay empty.
- Make non-F32 prefill attention consume stored KV precision. Historical Qwen3
  cold/warm parity and pinned 0.8B/2B prefix restoration/eviction checks passed;
  those results do not automatically carry over later numerical changes.
- Add Gemma per-request attention/Q4 KV and GPT-OSS packed-expert/Harmony
  candidates. Private `d4088c7` E2B c4 parity still fails (rank-0 requests 1/5:
  135 versus 127 IDs, first difference 52). Same-source cache-OFF/mixed/E4B were
  not run. Private `658e626` DEBUG narrowed a surviving delta to F32 PLE projection
  from identical input. Private `a9efe78` then passes 20 E2B cache-ON/OFF
  c1/c4 checks and four staggered requests against their own c1 IDs/final answers.
  E4B context 2048/capacity 4 fails capacity before generation; a separately requested
  capacity 1 profile passes one native long case. Q4/F32 fixes and capture fixtures are
  now in `4ad6a35` native/ASAN; private model results do not transfer automatically
  to that root, broad Gemma support or performance/default qualification.
- Correct shared-sampler top-k=0, conditional nucleus probabilities and grammar
  intersections with whitelist/range constraints. Mock-native full-worker
  NONE request error/peer/recovery checks pass. Common NONE/MTP JSON completion
  now commits the root-closing token and finishes STOP, including the exact token
  budget boundary. Actual `4ad6a35` pinned-model native/HTTP JSON checks pass; public global
  action-range composition remains fixture-only and real Python JSON is untested.
- Verify model downloads/cached bytes and add native image-profile, limited-pod
  and reproducible CPU gates. Separate Go-only `386b800` portable amd64 actual
  container/kind strict drain observed readiness 503, admission 503 and native active=1
  during accepted work, bounded stream completion and cleanup. Strict Count
  still fails. Current integrated core image is untested; original successful
  init-log capture has syntax checks only, awaiting final kind validation.
- Historical integrated `48dd55a` native: 1358 run / 1344 passed / the same 14 skips;
  strict critical ASAN/LSan: 727 run / 718 passed / the same 9 skips, zero
  failures/disabled/reports. The sanitizer selection contains all prior 702,
  11 added cases and 14 existing loader cases newly selected. Go all/race 3,
  SDK 164/27 optional skips and personal-tool 88 checks pass. Final actual
  wrapper audits are no-work; the immutable artifact pins 33 files. Earlier
  `4ad6a35`, `db9883a` and `3ed5428` evidence remains source-labeled.
- Historical `48dd55a` hard-mask lifecycle: NONE/MTP contradictory allow 16/deny 16
  JSON-off requests at temperatures 0/0.7 produce four zero-output ERROR finishes;
  six active-peer/recovery requests pass. Healthy MTP controls show actual
  verification/rejection/rollback; no successful verification is claimed for
  failing requests. V1 static-TLS failure before engine initialization remains
  UNEXECUTED; reviewed V2 changes only verified-library startup preload.
- Historical `48dd55a` E2B formal 20 cache-ON/OFF c=1/c=4 and four staggered requests
  match their old/own c1 IDs and complete answers. E4B context 2048/capacity 4 still
  fails initialization capacity (required 5,921,446,144 > available 4,681,576,448
  plan bytes), with zero generation. Failed-init peak RSS 6,236,108 KiB is not
  successful RAM sizing or a RAM-savings result. Earlier a9 cap1 is separate.
- Private H candidate `613a` still diverges in three of eight c8 requests and
  is not merged. Linux TLS candidate `a70cd4e` passes three SDK loader modes
  without execve, but all three raw ctypes modes SIGSEGV during teardown with
  allocator provenance; it is unqualified/unmerged. Matched `4ad6a35`/`48dd55a` SDK file
  runs restart automatically while `-c`/stdin fail static TLS, without model calls.
  The `4ad6a35` JSON passes and overall probabilistic FAIL are not inherited by `48dd55a`.
  No final package, broad support, speed, automatic/default or release promotion.

See [v0.2 release gates](docs/V0_2_RELEASE_GATES.md) for incomplete checks;
these entries are not a released model-support or performance announcement.

### Changed

- **Environment-variable surface reduced to the supported set.** The C++ core read
  280 distinct `DENSECORE_*` names, of which 257 were undocumented. The release
  build now reads 40, all listed in
  [docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md). Removed names were
  experiment leftovers whose current default is now the only behaviour; setting
  them has no effect. Names with a direct successor are rejected at startup with
  an error naming the replacement.
- **One boolean spelling for every toggle.** `DENSECORE_*` booleans accept
  `1/true/yes/on` and `0/false/no/off` in any case, and keep the default on an
  unrecognised value. Previously two parsers disagreed: under the "non-zero is
  true" parser, `VAR=false` **enabled** the feature. Affected default-on switches
  included `DENSECORE_KV_USE_BULK_PATH` and `DENSECORE_DECODE_GRAPH_CACHE`.
- **Diagnostic variables are compiled out of release builds.** `DENSECORE_DEBUG_*`,
  `DENSECORE_LOG_*`, `DENSECORE_VERBOSE_*`, `DENSECORE_CHECK_*` and
  `DENSECORE_PROFILE_*` now fold to their defaults under `NDEBUG`, removing
  `getenv` calls from MoE and attention hot paths. Configure with
  `-DDENSECORE_ENABLE_DEBUG_ENV=ON` to restore them in a release build; debug and
  test builds enable them automatically.
- **Legacy decode forks removed.** The `legacy` decode thread policy and decode
  graph-cache policy, the env override of MoE router selection, the batched-decode
  thread override, and the ARM compute-cluster override are gone; the qualified
  model-aware paths are now the only paths.
- **Scheduler configuration comes from `SchedulerConfig` only.** The scheduler no
  longer silently overrides caller-supplied config from the environment;
  `decode_homogeneous_batch_n_past` is now a config field.
- **Transparent huge pages are unconditional on Linux** for the KV cache. Hosts
  that must not use them should set the kernel THP policy.

### Added

- **Qwen3.6-35B-A3B text-only execution path**: DenseCore aliases Qwen3.6 GGUF/HF
  metadata onto the `qwen35moe` hybrid-SSM runtime, reconstructs the
  3-linear/1-full layer schedule from `full_attention_interval=4` when needed,
  and keeps multimodal execution fail-closed while skipping auxiliary vision
  tensors on the text path. v0.2 real-model/platform/performance qualification
  remains pending.

## [0.1.0] - 2026-09-25

### Release Status

The `denseai/densecore:0.1.0` linux/amd64 image is published on Docker Hub.
The source tag and any downloadable GitHub assets are listed under
[GitHub Releases](https://github.com/DenseAI/DenseCore/releases/tag/v0.1.0).
Other distribution channels have separate publication status.
Release scope and qualification gates are defined in
[`docs/RELEASE.md`](docs/RELEASE.md).

### Highlights

DenseCore is a memory-centric inference runtime for large language models on
heterogeneous CPU architectures. Comparative benchmark research is not part of
the v0.1.0 public release claim.

### Added

- **Model-loader coverage** for Qwen3/Qwen3.5 sparse MoE variants, Gemma, Phi,
  Mistral, and Mixtral families. This is not blanket production-support status;
  each model needs its own output-quality qualification.
- **NUMA-aware scheduling and experimental MoE sticky routing** with node-local
  allocation, expert placement reporting, and explicit diagnostic state. Sticky
  routing is off by default.
- **Inference runtime surfaces** including paged KV cache management, decode graph
  reuse, batched Q4_K kernels, and token-level inter-token-latency recording.
- **Linux CPU source targets** for Google C4A/Axion, C4, C4D, and generic Arm64.
  The v0.1.0 artifact-qualified platform is linux/amd64;
  Arm64 remains an engineering-validation surface. Apple Metal and ANE source
  surfaces remain experimental and are not release targets.
- **Python SDK** features including Hugging Face model loading, LoRA helpers,
  async generation, and optional LangChain/LangGraph wrappers.
- **Go server** endpoints for OpenAI-compatible chat and embeddings, optional
  gRPC, Prometheus metrics, API-key authentication, and lifecycle probes.
- **Packaging and deployment** for the published linux/amd64 container image,
  Linux x86_64 Python wheels (beta/source-preview), and the
  `charts/densecore` Helm chart (source-preview). Arm64 wheels and containers are
  not v0.1.0 artifact-qualified.

### Fixed

- GEMV dimension mismatch handling for multi-head projection paths.
- KV cache block release on early termination.
- Type-safe NUMA allocator deallocation.
- Long-context RoPE scaling and tokenizer byte-fallback handling.

### Changed

- Benchmark CSVs include `peak_rss_kb` and `tps_per_dollar` fields.
- `DENSECORE_BENCH_RECORD_TOKEN_TIMES` defaults to `1` so ITL is collected.
