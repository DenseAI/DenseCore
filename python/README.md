# DenseCore Python SDK

[![PyPI](https://img.shields.io/pypi/v/densecore)](https://pypi.org/project/densecore/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](../LICENSE)

The `densecore` PyPI package provides the native Python bindings for local GGUF inference.

## Installation

```bash
pip install densecore
```

Optional extras:

```bash
pip install "densecore[hf]"
pip install "densecore[langchain]"
pip install "densecore[full]"
pip install "densecore[torch]"
```

## Scope

The Python package includes:

- `DenseCore`
- `from_pretrained(...)`
- `AutoModel` and `AutoModelForCausalLM`
- `AutoTokenizer` wrapper when `transformers` is installed
- embedding helpers
- LoRA helpers
- LangChain integrations as optional extras

The Python package does not install the Go CLI. For `densecore run` or `densecore serve`, use the source-built CLI documented in [`../docs/CLI.md`](../docs/CLI.md).

## Quick Start

```python
from densecore import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-GGUF")

for token in model.stream("Explain CPU inference in one sentence.", max_tokens=64):
    print(token, end="", flush=True)
```

Hugging Face style kwargs are supported on `generate()`:

```python
text = model.generate(
    "Write one sentence about DenseCore.",
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(text)
```

## Core APIs

### Load a local GGUF

```python
from densecore import DenseCore

model = DenseCore("./models/model.gguf", threads=0)
print(model.generate("Hello", max_tokens=32))
```

### Download from Hugging Face Hub

```python
from densecore import from_pretrained

model = from_pretrained("Qwen/Qwen3-0.6B-GGUF")
print(model.generate("Hello", max_tokens=32))
```

### Streaming

```python
for token in model.stream("Count to five.", max_tokens=32):
    print(token, end="", flush=True)
```

```python
import asyncio

async def main():
    async for token in model.stream_async("Count to five.", max_tokens=32):
        print(token, end="", flush=True)

asyncio.run(main())
```

### Chat

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is DenseCore?"},
]

print(model.chat(messages, max_tokens=64))
```

### Embeddings

```python
embedding = model.embed("DenseCore runs GGUF models.")
batch = model.embed_batch(["hello", "world"])
```

### LoRA

```python
model.load_lora("./adapters/support.gguf", scale=0.8, name="support")
model.enable_lora("support")
print(model.generate("Answer politely.", max_tokens=64))
model.disable_lora()
```

## LangChain

Install the extra first:

```bash
pip install "densecore[langchain]"
```

Then see [docs/LANGCHAIN_GUIDE.md](docs/LANGCHAIN_GUIDE.md).

## Packaging Notes

- The package builds a native extension with CMake.
- If a wheel is unavailable for your platform, installation falls back to a local native build.
- Python package metadata currently targets Python `>=3.10`.

## Troubleshooting

### `Illegal instruction`

Your machine may not support the SIMD path enabled in the build. Rebuild locally with the relevant backend flags disabled.

### `Failed to initialize DenseCore engine`

Check that:

- the GGUF path exists
- the file is readable
- the model is compatible with the current native library build

### Wrong or low-quality output

If you are using a local GGUF, pass the matching tokenizer repository:

```python
from densecore import DenseCore

model = DenseCore("./model.gguf", hf_repo_id="Qwen/Qwen3-0.6B")
```

## Links

- [Main Repository](https://github.com/DenseCore/DenseCore)
- [API Reference](../docs/API_REFERENCE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
- [CLI Guide](../docs/CLI.md)

Apache 2.0 License. Issues: <https://github.com/DenseCore/DenseCore/issues>
