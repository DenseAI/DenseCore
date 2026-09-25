# DenseCore

**Run GGUF models as APIs on CPUs. No GPU required.**

DenseCore is an open-source inference server with an OpenAI-compatible API,
Docker deployment, health and metrics endpoints, and graceful request shutdown.
Its native C++ runtime and Go server are built for Linux CPU serving, including
large GGUF and MoE model paths. Qwen3.6-35B-A3B is a focus of the project's
CPU serving and optimization work. Kubernetes and Arm64 are available as source
surfaces with narrower qualification.

[![CI](https://github.com/DenseAI/DenseCore/actions/workflows/ci.yml/badge.svg)](https://github.com/DenseAI/DenseCore/actions/workflows/ci.yml)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Developer preview](https://img.shields.io/badge/status-developer%20preview-orange)](docs/RELEASE.md)

[Quick start](#quick-start) · [Run Qwen3.6-35B-A3B](#run-qwen36-35b-a3b) ·
[Try the CLI](#try-the-cli) ·
[Capabilities](#what-you-get) · [Support](#support-and-maturity) ·
[Documentation](#documentation)

## Quick start

Download the small GGUF used by the release smoke test, verify it, and start
the published `linux/amd64` API image:

```bash
mkdir -p models
curl -fL --retry 3 -o models/model.gguf \
  https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/resolve/7925ccdc665d4efdb1034791e6b553e11128e6f8/Qwen3.5-0.8B-Q4_K_M.gguf
printf '%s\n' 'f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec  models/model.gguf' | sha256sum -c -
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  denseai/densecore:0.1.0
```

In another terminal, check readiness and make a request:

```bash
curl --fail http://127.0.0.1:8080/health/ready

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

The image contains no model downloader; the GGUF file must exist before
startup. The port is bound to loopback. See [Deployment](docs/DEPLOYMENT.md)
for authentication, TLS, and Kubernetes setup.

### Run Qwen3.6-35B-A3B

For the larger model we have actively tested and optimized, download the
regular Q4_K_M GGUF and point the same API image at it:

```bash
mkdir -p models
curl -fL -C - --retry 3 -o models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf
printf '%s  %s\n' \
  '671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7' \
  'models/Qwen3.6-35B-A3B-Q4_K_M.gguf' | sha256sum -c -
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  denseai/densecore:0.1.0
```

Use the readiness and chat requests above in another terminal. The GGUF alone
is about 20 GB, so allow additional memory for inference. The published Docker
image is `linux/amd64`; C4A/Arm64 uses a source build. Our
[Qwen3.6 qualification report](docs/reports/2026-09-12-qwen36-v01-qualification.md)
describes historical C4/C4A runs, not a speed guarantee for this image.

## Try the CLI

To download a small default GGUF and start interactive chat instead, build the
native runtime and Go CLI from a recursive source checkout:

```bash
git clone --recurse-submodules https://github.com/DenseAI/DenseCore.git
cd DenseCore
make server
./bin/densecore-server run
```

The first run caches its model under `~/.densecore/models`. Source builds need a
C++ toolchain, CMake, and Go 1.25.13 or newer. The [CLI guide](docs/CLI.md)
covers model selection and direct API serving without Docker.

## What you get

| Surface | Included |
| --- | --- |
| **Native runtime** | C++ GGUF execution, CPU kernels, model-specific paths, and memory management |
| **Go server** | OpenAI-style chat and prompt completions, embeddings, rerank, and optional gRPC |
| **Operations** | Startup/readiness/liveness probes, Prometheus metrics, bounded admission, and graceful request shutdown |
| **Local tools** | Interactive CLI plus a Python SDK in source preview |

```text
CLI / HTTP / gRPC  →  Go server  →  C++ runtime  →  GGUF model on CPU
```

The HTTP server keeps one startup model for its process lifetime. Dynamic model
load/unload and speculative draft models are outside the v0.1 server contract.
See [Architecture](docs/ARCHITECTURE.md) for execution and ownership details.

## Support and maturity

| Path | Current status |
| --- | --- |
| Linux x86-64 CPU | Maintained GGUF serving path; `denseai/densecore:0.1.0` published for linux/amd64 |
| Linux Arm64 CPU | Maintained source-build path; no qualified v0.1 Arm64 release artifact |
| Python SDK and Helm chart | Beta/source-preview surfaces |
| Apple and optional accelerator paths | Experimental or narrower than full-model LLM serving |

These labels do not qualify every model or quantization. Check
[Hardware Support](docs/HARDWARE_SUPPORT.md) and the
[release scope](docs/RELEASE.md) before deploying. Comparative performance is
outside the v0.1 release claim; dated experiment reports describe their own
revisions and workloads.

## Documentation

| I want to… | Read |
| --- | --- |
| Run the CLI or configure a model | [CLI guide](docs/CLI.md) |
| Integrate with HTTP or gRPC | [API reference](docs/API_REFERENCE.md) · [OpenAPI](server/openapi.yaml) |
| Build or deploy a container | [Deployment guide](docs/DEPLOYMENT.md) |
| Configure Kubernetes | [Helm chart](charts/densecore/README.md) |
| Select hardware and tune execution | [Hardware support](docs/HARDWARE_SUPPORT.md) · [Performance tuning](docs/PERFORMANCE_TUNING.md) |
| Understand internals | [Architecture](docs/ARCHITECTURE.md) · [Contributor map](docs/CONTRIBUTOR_MAP.md) |

The [documentation index](docs/README.md) links the remaining operation and
conversion guides. The `0.1.0` Docker image is available for linux/amd64;
check the [release scope](docs/RELEASE.md) for the status of other artifacts.

## Contributing

Start with the [contributing guide](CONTRIBUTING.md). Bug reports and feature
discussions are welcome in [Issues](https://github.com/DenseAI/DenseCore/issues)
and [Discussions](https://github.com/DenseAI/DenseCore/discussions).

## License

DenseCore is licensed under [Apache License 2.0](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for bundled dependencies.
