# DenseCore Documentation

This directory contains the public documentation for DenseCore. It is intentionally
small: implementation notes, benchmark experiments, cost snapshots, and refactor
plans belong in Git history or benchmark artifacts rather than the public docs tree.

The implementation status in these documents was reviewed on 2026-07-13 against
the performance branch at commit `22b672d`. Benchmark numbers have a separate
evidence boundary described in [BENCHMARKS.md](BENCHMARKS.md).

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
| Review performance evidence | [Benchmarks](BENCHMARKS.md) |
| Review the scoped two-socket MoE result | [NUMA sticky routing](NUMA_STICKY_ROUTING.md) |

## Operational Contracts

- [Prompt Cache](prompt_cache.md): cache-safety and pod-affinity contract
- [SLO](SLO.md): example recording rules and default objectives
- [Operations Runbook](OPERATIONS_RUNBOOK.md): alert response procedures
- [Helm Chart](../charts/densecore/README.md): chart-specific values and install flow

## Product Surfaces

DenseCore currently exposes three maintained entry points:

- the C++ runtime and C API under `core/`
- the Python package under `python/`
- the Go CLI and OpenAI-compatible server under `server/`

The main LLM serving path is CPU-first and GGUF-based. Optional HAL, Apple, QNN,
and generic operation-graph components exist, but their maturity is not uniform.
See [Hardware Support](HARDWARE_SUPPORT.md) before treating an optional backend as
a production path.

## Documentation Policy

- Performance claims must link to a reproducible method and quality gate.
- A single diagnostic run is not a release claim.
- Feature maturity is stated as maintained, beta, experimental, or scaffold.
- Environment variables used only for diagnostics are not advertised as tuning
  defaults.
- Public docs describe committed behavior, not planned work.
