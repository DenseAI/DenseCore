# DenseCore Documentation

This directory contains the public usage and operation guides for DenseCore.
The dated large-model qualification summary describes a frozen source revision
and workload; it is not a benchmark of the current release image.

## Start Here

| Goal | Document |
| --- | --- |
| Understand the runtime | [Architecture](ARCHITECTURE.md) |
| Use Python or the server APIs | [API Reference](API_REFERENCE.md) |
| Run the source-built CLI | [CLI](CLI.md) |
| Deploy with Docker or Kubernetes | [Deployment](DEPLOYMENT.md) |
| Select a hardware/build profile | [Hardware Support](HARDWARE_SUPPORT.md) |
| Tune threads, memory, and concurrency | [Performance Tuning](PERFORMANCE_TUNING.md) |
| Convert or quantize a model | [Hugging Face to GGUF](HF_TO_GGUF.md) |
| Read release scope and gates | [Release (v0.1.0)](RELEASE.md) |

## Operational Contracts

- [Prompt Cache](prompt_cache.md): cache-safety and pod-affinity contract
- [SLO](SLO.md): example recording rules and default objectives
- [Operations Runbook](OPERATIONS_RUNBOOK.md): alert response procedures
- [Helm Chart](../charts/densecore/README.md): chart-specific values and install flow

## Product Surfaces

DenseCore's primary v0.1 developer-preview entry point is the Go CLI and
OpenAI-compatible server under `server/`:

- The published `denseai/densecore:0.1.0` container is a linux/amd64 developer
  preview; source builds remain available for development.
- The C++ runtime and C API under `core/` power the server. The Python package
  under `python/` and the Helm chart are source-preview surfaces.

The main LLM serving path is CPU-first and GGUF-based. Optional HAL, Apple, QNN,
and generic operation-graph components exist, but their maturity is not uniform.
See [Hardware Support](HARDWARE_SUPPORT.md) before treating an optional backend as
a qualified path.

## Documentation Policy

- Performance claims must link to a reproducible method and quality gate.
- A single diagnostic run is not a release claim.
- Dated qualification summaries are historical evidence. Do not combine results
  from different revisions or workloads into a current comparison.
- Feature maturity is stated as maintained, beta, experimental, or scaffold.
- Environment variables used only for diagnostics are not advertised as tuning
  defaults.
- Public docs describe committed behavior, not planned work.
