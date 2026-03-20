# DenseCore

[![CI](https://github.com/DenseCore/DenseCore/actions/workflows/ci.yml/badge.svg)](https://github.com/DenseCore/DenseCore/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/densecore)](https://pypi.org/project/densecore/)
[![Docker Hub](https://img.shields.io/docker/pulls/densecore/densecore)](https://hub.docker.com/r/densecore/densecore)

DenseCore is an open-source inference runtime built around a native C++ engine, a Python SDK, and a production API server.

The repository currently exposes three primary surfaces:

- Python package on PyPI: `densecore`
- Production container image on Docker Hub: `densecore/densecore`
- Go CLI / API server from source in [`server/`](server)

## What Works Today

- Python SDK for local GGUF inference
- Hugging Face Hub downloads for GGUF models
- OpenAI-compatible chat completions over HTTP
- Embeddings and rerank HTTP endpoints
- Optional gRPC server on port `50051`
- LoRA adapter loading from Python
- Docker deployment for the API server
- Helm chart for Kubernetes deployment

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`python/README.md`](python/README.md) | Python SDK install and usage |
| [`docs/CLI.md`](docs/CLI.md) | Go CLI and server commands |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Python, HTTP, and gRPC API overview |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker, Compose, and Kubernetes deployment |
| [`charts/densecore/README.md`](charts/densecore/README.md) | Helm chart values and install flow |

## Quick Start

### Python SDK

```bash
pip install densecore
```

```python
from densecore import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-GGUF")

for token in model.stream("The capital of France is", max_tokens=64):
    print(token, end="", flush=True)
```

### Docker API Server

Run the published API image from Docker Hub with a local GGUF model:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  densecore/densecore:latest
```

Check the server:

```bash
curl http://localhost:8080/health/live
curl http://localhost:8080/v1/models
```

Example chat request:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Go CLI

The interactive `densecore run` command is provided by the Go CLI in [`server/`](server), not by the PyPI package.

Build it from source:

```bash
make lib
make server
./bin/densecore run
```

## Installation Matrix

| Need | Install path |
| --- | --- |
| Python embedding / local inference | `pip install densecore` |
| Containerized HTTP/gRPC server | `docker pull densecore/densecore:latest` |
| Interactive TUI or source-built server binary | build from this repository |

## HTTP API Summary

The Go server exposes:

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/rerank`
- `GET /v1/models`
- `POST /v1/models/load`
- `POST /v1/models/unload`
- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /health/startup`
- `GET /metrics`

See [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) for request examples.

## Deployment

- Docker and Docker Compose: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- Helm chart: [`charts/densecore/README.md`](charts/densecore/README.md)
- Documentation index: [`docs/README.md`](docs/README.md)

## Build From Source

```bash
make lib
make server
```

Run tests:

```bash
cd core/build && ctest --output-on-failure
go test ./server/...
```

## Contributing

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [Issues](https://github.com/DenseCore/DenseCore/issues)
- [Discussions](https://github.com/DenseCore/DenseCore/discussions)

## License

[Apache License 2.0](LICENSE)
