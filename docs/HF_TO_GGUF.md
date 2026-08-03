# Hugging Face to GGUF

DenseCore loads GGUF models. Prefer an existing model-author or trusted publisher
GGUF when available. Convert a model yourself when you need a specific revision,
quantization, or reproducible model hash.

## Supported Python Workflow

Install the conversion dependencies and a DenseCore build that exports
`QuantizeModel`:

```bash
pip install "densecore[hf]" torch transformers huggingface-hub
git clone https://github.com/ggml-org/llama.cpp
```

The current Python converter locates `llama.cpp/convert_hf_to_gguf.py`, converts
to an intermediate GGUF, then calls the DenseCore quantization C API.

```python
from densecore import convert_from_hf

path = convert_from_hf(
    "Qwen/Qwen3-4B",
    output_path="qwen3-4b-q4-k-m.gguf",
    quantization="Q4_K_M",
)
print(path)
```

Supported public quantization labels are:

- `Q4_0`
- `Q4_1`
- `Q4_K_M`
- `Q5_K_M`
- `Q8_0`
- `F16`
- `F32`

`quick_convert(model_id)` selects `Q4_K_M`.

The advertised native-Python conversion fallback is not implemented. If the
llama.cpp converter cannot be found, `convert_from_hf` raises
`NotImplementedError` rather than producing a partial model.

## Direct llama.cpp Workflow

Use the converter and quantizer from the same tested llama.cpp revision:

```bash
python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
  /path/to/hf-model \
  --outfile model-f16.gguf

/path/to/llama.cpp/build/bin/llama-quantize \
  model-f16.gguf model-q4-k-m.gguf Q4_K_M
```

CLI paths vary by llama.cpp revision. Check its current build output rather than
assuming `quantize` or a Makefile-era binary name.

## Load the Result

```python
from densecore import DenseCore

model = DenseCore(
    "./model-q4-k-m.gguf",
    hf_repo_id="OWNER/ORIGINAL-MODEL",
)
print(model.generate("Hello", max_tokens=32))
```

Passing `hf_repo_id` gives DenseCore a tokenizer/template identity when the GGUF
metadata alone is insufficient.

## Validation

Conversion success is not inference qualification. For every converted model:

1. record the source revision, converter revision, output SHA-256, and quantization
2. compare prompt rendering and token IDs with the source model
3. run deterministic short factual prompts
4. run at least one long-context exact or semantic QA case
5. inspect visible output for repetition, empty responses, leaked thinking, and
   malformed tool calls
6. only then run performance benchmarks

Quantization changes output probabilities. Use task-level quality checks when
choosing between Q4, Q5, and Q8; do not infer a universal quality loss or speedup
from the format name alone.
