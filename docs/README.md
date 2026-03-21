# DenseCore Documentation

DenseCore is the **memory-centric execution runtime for heterogeneous AI inference** — the runtime substrate of the Dense Series stack. It maximizes locality, utilization, and determinism across x86, ARM64, and Apple Silicon hardware.

This documentation set is organized around the three public distribution surfaces in this repository:

- PyPI package: `densecore`
- Docker Hub image: `densecore/densecore`
- Source-built Go CLI and API server

## Start Here

| Goal | Document |
| --- | --- |
| Use the Python SDK | [Python SDK Guide](../python/README.md) |
| Run the Go CLI locally | [CLI Guide](CLI.md) |
| Deploy the API server with Docker or Kubernetes | [Deployment Guide](DEPLOYMENT.md) |
| Call the HTTP or gRPC APIs | [API Reference](API_REFERENCE.md) |
| Tune production behavior | [Performance Tuning](PERFORMANCE_TUNING.md) |

## Product Surfaces

### Python SDK

Use `pip install densecore` when you want:

- local GGUF inference from Python
- Hugging Face Hub downloads
- embedding and rerank helpers
- LangChain and LangGraph integrations

Primary doc: [../python/README.md](../python/README.md)

### Go CLI

Use the CLI when you want:

- `densecore run` interactive local chat
- `densecore serve` for a standalone API process

Primary doc: [CLI.md](CLI.md)

### API Server

Use the API server when you want:

- OpenAI-compatible chat completions
- embeddings and rerank HTTP endpoints
- Prometheus metrics and Kubernetes probes
- optional gRPC service on port `50051`

Primary docs:

- [API_REFERENCE.md](API_REFERENCE.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)

## Production Docs

- [SLO](SLO.md)
- [Operations Runbook](OPERATIONS_RUNBOOK.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Benchmarks](BENCHMARKS.md)
- [Architecture](ARCHITECTURE.md)

## Notes

- `pip install densecore` installs the Python SDK. It does not install the Go CLI.
- The Docker Hub image runs the API server. It does not launch the interactive TUI.
- The server expects `MAIN_MODEL_PATH` for a preloaded model in Docker and Kubernetes deployments.
