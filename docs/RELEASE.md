# DenseCore v0.1.0 Release Documentation

This document defines the public release boundary for the Apache-2.0 v0.1
developer preview.

## Release scope

- **Version:** `v0.1.0`
- **Current maturity:** `v0.1 developer preview`
- Status: the `denseai/densecore:0.1.0` linux/amd64 image is published on
  Docker Hub. Dated local candidate qualification is recorded in the
  [Qwen3.6 qualification report](reports/2026-09-12-qwen36-v01-qualification.md).
  The [GitHub release](https://github.com/DenseAI/DenseCore/releases/tag/v0.1.0)
  lists the source tag and any downloadable tarballs. Other registry artifacts
  have separate publication status.

## Source of truth

All public-facing release claims in this repository must be consistent with:

- [`CHANGELOG.md`](../CHANGELOG.md)
- [`docs/HARDWARE_SUPPORT.md`](HARDWARE_SUPPORT.md)
- [`docs/API_REFERENCE.md`](API_REFERENCE.md)
- [`server/openapi.yaml`](../server/openapi.yaml)

## Public surfaces for v0.1.0

### Primary surfaces

1. **Go HTTP server (`server`)**
2. **Docker image (`denseai/densecore`) as the containerized server path**

The release workflow qualifies the published Docker image and produces these
GitHub release artifacts:

- `densecore-0.1.0-source.tar.gz`
- `densecore-0.1.0-linux-amd64.tar.gz`

The `denseai/densecore:0.1.0` image is published separately, then the release
workflow checks its platform, source revision label, vulnerability scan, real
model API response, and graceful shutdown before uploading GitHub assets.

Release containers and the Linux tarball must use an explicit release CPU
profile and must not contain `-march=native` or `-mcpu=native`. Native builds
remain the source-build default for development and benchmark qualification and
are not general-purpose release artifacts.

The Linux AMD64 tarball is qualified on Ubuntu 24.04. It requires glibc 2.38
or newer, libstdc++ providing `GLIBCXX_3.4.32`, and the OpenMP runtime
`libgomp.so.1`. “Portable” describes the CPU build profile, not compatibility
with every Linux distribution. Use the Docker image on hosts without these
runtime libraries.

Extract the tarball into its own directory and run `./densecore-server serve
--model /absolute/path/model.gguf` (on one line). Keep the launcher,
`densecore-server.bin`, and bundled shared libraries together. The launcher
resolves adjacent libraries automatically and uses `exec` so signals reach the
server. No `LD_LIBRARY_PATH` configuration is required. Verify a downloaded
archive from its directory with `sha256sum -c <archive-name>.sha256`.

Repository builds require a recursive clone because ggml is a pinned submodule.
Automatically generated GitHub source archives do not include submodule content;
they are source snapshots, not standalone build bundles.

For v0.1.0, the qualified binary and container platform is
`linux/amd64` only. Linux Arm64 remains a supported source-build surface with
C4A engineering validation, but no Arm64 binary, wheel, or container is
published as an RC-qualified artifact until native Arm execution passes the
same real-model serving and graceful-shutdown gate.

### Source-preview / beta surfaces

1. **Python SDK (`python/`)**
   - Local embedding/inference surface for SDK users.
   - Not the primary server deployment surface.
2. **Helm chart (`charts/densecore`)**
   - Kubernetes preview surface for experimentation and deployment templates.
   - Requires explicit platform-level validation before production claims.

## DenseCloud public dependency contract

DenseCore `v0.1.0` consumes DenseCloud through these public artifacts:

- Go module: `github.com/DenseAI/DenseCloud@v1.1.0`
- Helm chart dependency: `dense-base 1.1.0`
- OCI registry: `oci://ghcr.io/denseai/charts`

Qualification status:

- Public module resolution: `PASS`
- Anonymous OCI chart resolution: `PASS`
- DenseCore consumer build and Go test path: `PASS`
- Helm dependency rebuild, lint, and render: `PASS`
- Kubernetes/API smoke with a real model: `PENDING`

Do not promote the DenseCore/DenseCloud compatibility row beyond `pending`
until the Kubernetes/API smoke runs against those public artifacts without a
sibling checkout, `go.work`, committed `replace`, vendored chart archive, or
`file://` chart dependency. Use
[`scripts/qualify_densecloud_public.sh`](../scripts/qualify_densecloud_public.sh)
for the clean-room check and retain its metadata artifact with the release
evidence.

## Qualification gates

Before any `v0.1.0` public claim, ensure each checkpoint is complete:

- OpenAI-compatible API behavior is validated via
  [`/v1/chat/completions`](API_REFERENCE.md) + representative QA set.
- Comparative performance claims are outside the v0.1.0 public release scope.
- Non-primary surfaces are explicitly labeled as beta/source-preview and are not
  used to define MVP release grade.
- A startup model supplied through `--model` or `MAIN_MODEL_PATH` is mandatory.
- The startup model is loaded before listeners open; an invalid or unreadable
  model fails process startup instead of leaving a permanently unready server.
- Dynamic model load/unload and speculative draft models are not exposed by the
  v0.1.0 server.
- Direct CLI serving binds to loopback by default and CORS is disabled by default.
  The Docker quick start publishes its host port on loopback. Network exposure
  requires an explicit bind/publish decision, API-key authentication, an origin
  allowlist when browser access is needed, and TLS at the ingress or proxy.
- Queue saturation is returned as a stable retryable HTTP capacity response, not
  as an internal server failure.
- Every C++ generation or embedding request accepted while the engine is running
  produces exactly one terminal success, cancellation, or error result before
  recycling. Rerank inherits this guarantee from its embedding requests.
- Server shutdown stops API admission and queue dequeue, terminally rejects
  queued requests, joins Go submitters and completion trackers, drains the C++
  runtime, and only then closes the native engine.
- Source and binary archives contain the Apache license, NOTICE, collected
  third-party notices, release documentation, OpenAPI contract, and the release
  smoke tooling. Missing mandatory artifacts fail packaging.
- The clean-room qualification builds the portable source archive, runs C++ and
  Go tests, starts the built Docker image with a pinned SHA-256-verified real
  GGUF mounted read-only, exercises health/models plus streaming and non-stream
  chat, and verifies a zero-exit graceful container stop.
- Run the actual packaged launcher after relocation, without a manually supplied
  `LD_LIBRARY_PATH`, and verify native usage, stream termination, and clean exit.
  A source-rebuild smoke does not substitute for executing the shipped binary.
- Critical and high vulnerabilities in the shipped tarball or container block
  publication unless an explicit release waiver documents the affected artifact,
  exposure, and mitigation.

## Local qualification

Run the source-archive qualification with a locally available test model:

```bash
DENSECORE_SMOKE_MODEL=/absolute/path/to/qualified-model.gguf \
  DENSECORE_BUILD_JOBS=2 \
  DENSECORE_SMOKE_KEEP_ARTIFACTS=1 \
  bash scripts/release_smoke.sh
```

Record the model SHA-256 alongside the result. `DENSECORE_BUILD_JOBS` limits
native build parallelism; its default is the detected processor count. Failed
release and container smoke runs retain their temporary artifacts automatically.
`DENSECORE_SMOKE_KEEP_ARTIFACTS=1` also retains successful runs and prints their
artifact directories. Remove retained directories when they are no longer needed.

`DENSECORE_SOURCE_TARBALL` selects an already packaged source archive.
`DENSECORE_SKIP_DOCKER=1` validates the source-built server only; it does not
qualify a container. Test the actual candidate image separately with
`scripts/container_model_smoke.sh IMAGE MODEL` when using that mode.

## Runtime environment contract

| Canonical variable | Removed pre-release alias |
| --- | --- |
| `DENSECORE_ALLOW_DECODE_THREADS_OVER_BASE` | `DENSECORE_ALLOW_EXCEED_BASE_THREADS` |
| `DENSECORE_PAGED_DECODE_MIN_CONTEXT` | `DENSECORE_PAGED_ATTN_DECODE_MIN_CONTEXT` |
| `DENSECORE_SLIDING_WINDOW_SIZE` | `DENSECORE_KV_SLIDING_WINDOW` |
| `DENSECORE_SINK_TOKENS` | `DENSECORE_KV_SINK_TOKENS` |
| `DENSECORE_CALLBACK_MODE` | `DENSECORE_DIRECT_CALLBACK` |

These aliases were removed before the first public tag and are not part of the
compatibility contract. If any removed alias is present, engine initialization
fails with the canonical replacement name instead of silently changing runtime
behavior.
