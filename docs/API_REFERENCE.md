# API Reference

DenseCore exposes a Python SDK, an OpenAI-compatible HTTP server, and an optional
gRPC service. The Python package and Go CLI are separate distribution surfaces.

## Python SDK

Install the local inference package:

```bash
pip install densecore
```

### Load and Generate

```python
from densecore import DenseCore

model = DenseCore(
    model_path="./model.gguf",
    threads=0,
    hf_repo_id="Qwen/Qwen3-0.6B",
)

text = model.generate("Hello", max_tokens=64)

for token in model.stream("Hello", max_tokens=64):
    print(token, end="", flush=True)
```

Chat, embeddings, and async streaming are available on the same object:

```python
response = model.chat(
    [{"role": "user", "content": "What is DenseCore?"}],
    max_tokens=64,
)

vector = model.embed("DenseCore")
vectors = model.embed_batch(["hello", "world"])

async for token in model.stream_async("Hello", max_tokens=64):
    print(token, end="")
```

### Hugging Face Loading

```python
from densecore import AutoModelForCausalLM, from_pretrained

model = from_pretrained("Qwen/Qwen3-0.6B-GGUF")
auto_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-GGUF")
```

### LoRA

```python
model.load_lora("./adapter.gguf", scale=0.8, name="support")
model.enable_lora("support")
result = model.generate("Answer politely.", max_tokens=64)
model.disable_lora()
model.unload_lora("support")
```

LoRA support is request-local through the public model methods. Do not depend on
the private Python LoRA manager.

## HTTP Server

Start the source-built server:

```bash
./bin/densecore-server serve --model ./model.gguf
```

### Chat Completions

`POST /v1/chat/completions` accepts OpenAI-style messages and supports SSE when
`stream` is true.

```json
{
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": false
}
```

### Tool Calling

The chat route accepts `tools` and `tool_choice`. Supported parser families are
Qwen XML-style blocks, Hermes/OpenAI JSON, and a conservative generic JSON
fallback.

```json
{
  "messages": [{"role": "user", "content": "What is the weather in Seoul?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

`tool_choice` supports `none`, `auto`, `required`, or a forced function object.
The server validates that emitted calls reference a declared function. Reasoning
inside `<think>...</think>` is separated from visible assistant content.

### Prompt Completions

`POST /v1/completions` accepts a prompt string. `POST /completion` is a
compatibility alias.

```json
{
  "prompt": "The capital of France is",
  "max_tokens": 8,
  "stream": false
}
```

### Embeddings and Rerank

`POST /v1/embeddings`:

```json
{"input": ["hello", "world"]}
```

`POST /v1/rerank`:

```json
{
  "query": "CPU inference",
  "documents": [
    {"text": "DenseCore serves GGUF models."},
    {"text": "This document is unrelated."}
  ],
  "top_n": 2,
  "return_documents": true
}
```

### Model Lifecycle

- `GET /v1/models`
- `POST /v1/models/load`
- `POST /v1/models/unload`

```json
{
  "model_path": "/models/model.gguf",
  "draft_model_path": "",
  "threads": 16
}
```

`draft_model_path` is accepted by the loader, but speculative decoding is not a
qualified serving feature. The current decode loop does not consume the draft
model.

### Runtime and Health

- `GET /v1/runtime/profile`
- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /health/startup`
- `GET /metrics`

Liveness means the process is alive. Startup and readiness include model lifecycle
state and can return a non-OK status while no model is ready.

## Prompt Cache

Chat requests may include `cache_control` fields such as `conversation_id`,
`cache_id`, and `affinity_key`. Cache reuse remains fail-closed when model,
template, tokens, LoRA, KV, graph, sliding-window, or SSM identity is incompatible.
See [prompt_cache.md](prompt_cache.md).

## gRPC

gRPC is **disabled by default**. Enable it explicitly:

```bash
./bin/densecore-server serve --model ./model.gguf --grpc --grpc-port 50051
```

The implementation registers DenseCore RPCs, the standard gRPC health service,
and optional reflection, auth, TLS, and mTLS controls. The protobuf contract is in
[`proto/densecore.proto`](../proto/densecore.proto).
