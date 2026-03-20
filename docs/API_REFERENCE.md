# API Reference

This document covers the public interfaces that are visible in the current repository state:

- Python SDK
- HTTP API
- gRPC server

## Python SDK

Install:

```bash
pip install densecore
```

Primary objects exported by `densecore`:

- `DenseCore`
- `from_pretrained(...)`
- `AutoModel`
- `AutoModelForCausalLM`
- `AutoTokenizer`
- `GenerationConfig`
- `EmbeddingModel`
- `LoRAConfig`
- `LoRAManager`

### `DenseCore`

Basic construction:

```python
from densecore import DenseCore

model = DenseCore(
    model_path="./model.gguf",
    threads=0,
    hf_repo_id="Qwen/Qwen3-0.6B",
)
```

Common methods:

```python
text = model.generate("Hello", max_tokens=64)

for token in model.stream("Hello", max_tokens=64):
    print(token, end="")

response = model.chat(
    [{"role": "user", "content": "What is DenseCore?"}],
    max_tokens=64,
)

vector = model.embed("DenseCore")
vectors = model.embed_batch(["hello", "world"])

metrics = model.get_metrics()
```

Async streaming:

```python
async for token in model.stream_async("Hello", max_tokens=64):
    print(token, end="")
```

LoRA helpers:

```python
model.load_lora("./adapter.gguf", scale=0.8, name="adapter")
model.enable_lora("adapter")
model.disable_lora()
```

### `from_pretrained(...)`

Download from Hugging Face Hub and create a `DenseCore` instance:

```python
from densecore import from_pretrained

model = from_pretrained("Qwen/Qwen3-0.6B-GGUF")
```

### `AutoModelForCausalLM`

Transformers-style convenience wrapper:

```python
from densecore import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-GGUF")
```

## HTTP API

The Go server exposes the following routes.

### `POST /v1/chat/completions`

OpenAI-compatible chat completion route.

Request:

```json
{
  "messages": [
    {"role": "user", "content": "Say hello."}
  ],
  "max_tokens": 64,
  "stream": false
}
```

Streaming is supported with Server-Sent Events when `stream=true`.

### `POST /v1/embeddings`

Request:

```json
{
  "input": ["hello", "world"]
}
```

### `POST /v1/rerank`

Cohere-style rerank route.

Request:

```json
{
  "query": "cpu inference",
  "documents": [
    {"text": "DenseCore is a CPU inference runtime."},
    {"text": "This document is unrelated."}
  ],
  "top_n": 2,
  "return_documents": true
}
```

### `GET /v1/models`

Returns the currently loaded model identity.

### `POST /v1/models/load`

Loads a model into a running server:

```json
{
  "model_path": "/models/model.gguf",
  "draft_model_path": "",
  "threads": 4
}
```

### `POST /v1/models/unload`

Unloads the active model.

### Health and metrics

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /health/startup`
- `GET /metrics`
- `GET /v1/runtime/profile`

## gRPC

The server can also start a gRPC service on port `50051`.

Current implementation details visible in the codebase:

- enabled by default in `serve` mode
- optional TLS support
- optional auth interceptor support
- reflection enabled by default for `grpcurl`
- gRPC health service registered

If you do not want gRPC, start the server with:

```bash
densecore serve --grpc=false --model ./model.gguf
```

## Operational Semantics

- The server may start before a configured model has finished loading.
- `GET /health/ready` reports model readiness and KV cache pressure.
- `GET /health/startup` returns a non-OK status while the initial model is loading or when no model is configured.
