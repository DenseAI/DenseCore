# Changelog

All notable changes to DenseCore are documented here. This file records release
material and platform maturity. Comparative benchmark research is outside the
public source release.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### v0.2 candidates (qualification in progress)

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
  Population-distribution, speed and automatic qualification remain absent;
  automatic profiles stay empty.
- Make non-F32 prefill attention consume stored KV precision. Historical Qwen3
  cold/warm parity and pinned 0.8B/2B prefix restoration/eviction checks passed;
  those results do not automatically carry over later numerical changes.
- Add Gemma per-request attention/Q4 KV and GPT-OSS packed-expert/Harmony
  candidates. Private `d4088c7` E2B c4 parity still fails (rank-0 requests 1/5:
  135 versus 127 IDs, first difference 52). Same-source cache-OFF/mixed/E4B are
  not run. Private `658e626` DEBUG narrows a surviving delta to F32 PLE projection
  from identical input; observer/root integration and functional fix remain pending.
- Correct shared-sampler top-k=0, conditional nucleus probabilities and grammar
  intersections with whitelist/range constraints. Mock-native full-worker
  NONE request error/peer/recovery checks pass; actual new-source HTTP/Python sampling
  is unqualified. MTP JSON completion has a separate pending implementation.
- Verify model downloads/cached bytes and add native image-profile, limited-pod
  and reproducible CPU gates. Separate Go-only `386b800` portable amd64 actual
  container/kind strict drain observed readiness 503, admission 503 and native active=1
  during accepted work, bounded stream completion and cleanup. Strict Count
  still fails. Current integrated core image is untested; original successful
  init-log capture has syntax checks only, awaiting final kind validation.
- Integrated `db9883a` native: 1335 run, 1321 passed, exactly the same 14 skips,
  zero failures/disabled; focused 108 passed. Initial audit relinked with unchanged
  executable SHA, so it is not a no-work result. A repeat no-work audit matches
  the same source/test binary. Selected critical ASAN/LSan on exact `db9883a`:
  688 run, 679 passed, the same 9 prior skips, zero failures/disabled/reports;
  final actual wrapper audit is no-work on the tested artifact. Unmerged PLE
  and pending JSON changes remain outside that result. Prior `143a2ad` Go/Python
  and archived `3ed5428` ASAN remain source-labeled, not new-source adapter/model proof.
  No final package, model-support or performance/default promotion follows.

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
