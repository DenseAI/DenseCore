# DenseCore

**Serve 35B MoE models on plain CPUs. First token up to 39% sooner than llama.cpp.**

A CPU-first LLM inference server: native C++ runtime, Go API server,
OpenAI-compatible endpoints. No GPU.

[![CI](https://github.com/DenseAI/DenseCore/actions/workflows/ci.yml/badge.svg)](https://github.com/DenseAI/DenseCore/actions/workflows/ci.yml)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Developer preview](https://img.shields.io/badge/status-developer%20preview-orange)](docs/RELEASE.md)

## Performance

Qwen3.6-35B-A3B Q4_K_M · 16 vCPU · real `/v1/chat/completions` requests ·
DenseCore vs llama.cpp on the same host

| Host | Concurrent requests | Time to first token | Output tokens/s |
|---|---:|---:|---:|
| C4A · Google Axion (Arm) | 1 | **2.07 s** vs 2.36 s (−12%) | **25.9** vs 23.8 (+9%) |
| C4A · Google Axion (Arm) | 4 | **5.12 s** vs 7.58 s (−32%) | 36.0 vs **39.9** (−10%) |
| C4 · Intel Xeon (x86) | 1 | **2.90 s** vs 3.62 s (−20%) | **13.4** vs 13.0 (+3%) |
| C4 · Intel Xeon (x86) | 4 | **8.06 s** vs 13.21 s (−39%) | 20.2 vs **20.7** (−2%) |

DenseCore answers first in every cell and wins single-stream throughput;
llama.cpp still leads batch throughput at four concurrent requests. Both engines
passed the same output-quality suite. Native builds, measured 2026-09-13.
[Method and limits →](docs/BENCHMARKS.md)

## Quick start

```bash
mkdir -p models
curl -fL -o models/model.gguf \
  https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/resolve/7925ccdc665d4efdb1034791e6b553e11128e6f8/Qwen3.5-0.8B-Q4_K_M.gguf
echo 'f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec  models/model.gguf' | sha256sum -c -

docker run --rm -p 127.0.0.1:8080:8080 -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf denseai/densecore:0.1.0
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

## Run Qwen3.6-35B-A3B

The benchmark numbers come from a native source build. The Docker image uses a
portable CPU profile and is slower on this model.

```bash
git clone --recurse-submodules https://github.com/DenseAI/DenseCore.git && cd DenseCore
make server && mkdir -p models

curl -fL -C - -o models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf
echo '671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7  models/Qwen3.6-35B-A3B-Q4_K_M.gguf' | sha256sum -c -

./bin/densecore-server serve --model models/Qwen3.6-35B-A3B-Q4_K_M.gguf --threads 16
```

The GGUF is about 20 GB; plan for more RAM than that. Source builds need a C++
toolchain, CMake and Go 1.25.13+. For interactive chat with a small default
model, run `./bin/densecore-server run`.

## What you get

- **OpenAI-compatible API**: chat and prompt completions, embeddings, rerank, optional gRPC
- **Native CPU runtime**: C++ GGUF execution with MoE paths tuned for x86 and Arm
- **Production basics**: readiness/liveness probes, Prometheus metrics, bounded admission, graceful shutdown
- **Packaging**: Docker image, CLI, Helm chart and Python SDK (preview)
- **Typed decisions**: [Laya GGUF with a native System One API](docs/LAYA.md) (preview)

## Status

| Path | Status |
| --- | --- |
| Linux x86-64 | Maintained; `denseai/densecore:0.1.0` for linux/amd64 |
| Linux Arm64 | Maintained source build; no release artifact yet |
| Python SDK, Helm chart | Preview |
| Apple and accelerator paths | Experimental |

Developer preview: not every model or quantization is qualified. See
[Hardware support](docs/HARDWARE_SUPPORT.md) and [Release scope](docs/RELEASE.md).

## Documentation

[CLI](docs/CLI.md) · [API reference](docs/API_REFERENCE.md) ·
[Deployment](docs/DEPLOYMENT.md) · [Helm](charts/densecore/README.md) ·
[Performance tuning](docs/PERFORMANCE_TUNING.md) · [Benchmarks](docs/BENCHMARKS.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) ·
[All docs](docs/README.md)

## License

[Apache 2.0](LICENSE). Bundled dependencies: [third-party notices](THIRD_PARTY_NOTICES.md).
