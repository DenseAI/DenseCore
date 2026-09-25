# Changelog

All notable changes to DenseCore are documented here. This file records release
material and platform maturity. Comparative benchmark research is outside the
public source release.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **Qwen3.6-35B-A3B text-only support**: DenseCore aliases Qwen3.6 GGUF/HF
  metadata onto the `qwen35moe` hybrid-SSM runtime, reconstructs the
  3-linear/1-full layer schedule from `full_attention_interval=4` when needed,
  and keeps multimodal execution fail-closed while skipping auxiliary vision
  tensors on the text path.

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
