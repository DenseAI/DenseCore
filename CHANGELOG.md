# Changelog

All notable changes to DenseCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-12-21

### Added

- **Full Qwen3 Support**: Fixed GEMV dimension mismatch for Qwen3 projection layers
- **Llama 3.2 Support**: Verified Llama-3.2-1B-Instruct compatibility
- **CI/CD Improvements**: Simplified GitHub Actions workflows with relaxed lint rules

### Fixed

- **GEMV Memory Corruption**: Refactored from `GGML_OP_MAP_CUSTOM1` to `GGML_OP_CUSTOM` to support independent output tensor dimensions
- **Userdata Pool Corruption**: Weight tensor now stored in `dst->src[1]` instead of potentially stale userdata pool
- **CPU Backend Build**: Fixed `__attribute__` placement in `cpu_backend_opt.cpp` for GCC/Clang compatibility

### Performance

| Model | Quantization | TTFT (ms) | Speed (tok/s) |
|-------|--------------|-----------|---------------|
| **Qwen3-0.6B** | Q8_0 | 56.58 | 22.81 |
| **Qwen3-4B** | Q4_K_M | 186.41 | 8.38 |
| **Qwen3-8B** | Q4_K_M | 346.75 | 5.11 |
| **Llama-3.2-1B** | Q8_0 | 71.46 | 17.05 |

*Tested on Intel i7-10870H (8 cores, AVX2)*

---

## [0.2.0] - 2025-12-16

### Added

#### Core Features
- **Universal Model Support:** Added support for Qwen2.5, Qwen3, TinyLlama, SmolLM, and all Llama-based architectures
- **Speculative Decoding:** Implemented draft-target model pairing for 1.5-2.5x speedup on long generations
- **Paged KV Cache:** Implemented vLLM-inspired paged memory management for 95% memory savings on short sequences
- **Async Streaming:** Added `stream_async()` method with native async/await support
- **Batch Embeddings:** Added `embed_batch()` for efficient multi-text embedding generation
- **HuggingFace Tokenization:** Integrated `transformers` library for better tokenization quality
- **System Diagnostics:** Added topology reporting and configurable thread pinning (`SCATTER`/`COMPACT`)
- **Symbol Visibility:** Optimized binary size by hiding internal symbols
- **Version Info:** Embedded build metadata for runtime diagnostics

#### Model Optimization
- **Quantization Module:** INT4/INT8/FP8 quantization with AWQ and MAX algorithms
- **Pruning Module:** Depth pruning with magnitude-based importance scoring
- **Predefined Configs:** Ready-to-use configs (`INT4_AWQ_CFG`, `DEPTH_PRUNE_50_CFG`, etc.)

#### Python SDK
- **from_pretrained():** Factory method to auto-download models from HuggingFace Hub
- **GenerationConfig:** Advanced configuration object for fine-grained control
- **Context Manager:** Support for `with DenseCore(...) as model:` syntax
- **Metrics API:** `get_metrics()` for monitoring throughput and latency
- **Custom Tokenization:** `hf_repo_id` parameter for using official HF tokenizers
- **LangChain Integration:** Added `DenseCoreEmbeddings` for vector store compatibility
- **Batch Optimization:** Non-blocking C++ submission for higher throughput
- **Type Safety:** Hardened ctypes bindings and error handling

#### Server (Go)
- **OpenAI-Compatible API:** `/v1/chat/completions` endpoint with SSE streaming
- **Authentication:** API key-based auth with tier-based rate limiting
- **Prometheus Metrics:** Comprehensive metrics at `/metrics` endpoint
- **Health Probes:** Kubernetes-ready health checks (`/health/live`, `/health/ready`, `/health/startup`)
- **Graceful Shutdown:** Request draining with configurable timeout
- **Request Tracing:** Added UUID-based request IDs for observability
- **Resilience:** Enhanced middleware chain and fixed channel leaks

#### DevOps
- **Docker Multi-Stage Build:** Optimized production image (<200MB)
- **Kubernetes Manifests:** Complete K8s deployment with HPA, ingress, and service monitor
- **Docker Compose:** Development stack with optional Prometheus/Grafana
- **GitHub Actions:** CI/CD workflows for C++, Python, and Go (planned)

#### Documentation
- **Complete Rewrite:** Production-ready documentation matching top OSS standards
- **Architecture Guide:** Deep dive into system design and optimizations
- **Model Optimization Guide:** Comprehensive quantization and pruning documentation
- **API Reference:** Complete Python, C, and REST API documentation
- **Deployment Guide:** Docker, K8s, and cloud platform deployment instructions

### Changed

- **Benchmark Performance:** Achieved 3-13x speedup over HuggingFace Transformers on CPU
- **Memory Efficiency:** Reduced runtime memory by 95% for short sequences via paged cache
- **Python Package:** Simplified installation to `pip install densecore` (pre-built wheels)
- **Error Handling:** Improved exception hierarchy with specific error types

### Fixed

- **QK-Norm Support:** Fixed Qwen3 model compatibility
- **Garbage Output:** Resolved tokenization issues via HF tokenizer integration
- **Memory Leaks:** Fixed KV cache memory leaks in long-running sessions
- **NaN Logits:** Fixed numerical stability issues in attention computation
- **Vocab Size Mismatch:** Corrected vocabulary size handling across architectures
- **NUMA Allocator:** Fixed memory corruption via type-safe deallocation
- **Linker Errors:** Resolved symbol visibility issues in tests
- **Docker:** Fixed build failures and optimized layer caching

### Performance

- **Qwen2.5-0.5B:** 32.0 TPS (2.6x faster than Transformers)
- **TinyLlama-1.1B:** 26.2 TPS (13.3x faster than Transformers)
- **Average Speedup:** 3.13x across all tested workloads
- **Memory Usage:** 3.5GB for 7B INT4 model (vs. 14GB FP16)

## [0.1.0] - 2025-12-15

### Added

- **Core:** Initial C++ inference engine with GGML backend
- **Python:** Basic ctypes bindings, GGUF loading, and text generation
- **Server:** MVP Go REST API
- **Memory:** Basic contiguous KV cache allocation

### Known Issues

- Limited model architecture support
- High memory usage (no paging)
- No quantization support
- CLI-only interface

---

## Version History

- **0.2.0** - Production release (2025-12-16)
- **0.1.0** - Production release (2025-12-15)
---

## Migration Guides

### Migrating to 0.2.0

**Python API Changes:**

```python
from densecore import DenseCore
model = DenseCore("model.gguf")
output = model.generate("Hello", max_tokens=100)
```

**Breaking Changes:**
- Renamed `Engine` → `DenseCore`
- Renamed `infer()` → `generate()`
- Renamed `max_len` → `max_tokens`
- Removed `setup()` (automatic initialization)

---

## Links

- [GitHub Repository](https://github.com/Jake-Network/DenseCore)
- [Documentation](https://github.com/Jake-Network/DenseCore/tree/main/docs)
- [PyPI Package](https://pypi.org/project/densecore/)
- [Docker Images](https://ghcr.io/jake-network/densecore)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.
