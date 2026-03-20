# Converting HuggingFace Models to GGUF

## Why Convert Models Yourself?

Pre-converted GGUF files from community sources may have issues:
- ❌ Incomplete tensor conversions (missing QK-Norm for Qwen3)
- ❌ Incorrect quantization settings
- ❌ Outdated conversion tools

Converting from the **official HuggingFace model** ensures:
- ✅ All model weights properly converted
- ✅ Latest architecture features supported
- ✅ Control over quantization quality

---

## Quick Start

**TL;DR** - Convert Qwen3-4B in 3 commands:

```bash
# 1. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp

# 2. Convert HF model to GGUF (F16)
python3 convert-hf-to-gguf.py Qwen/Qwen3-4B --outfile qwen3-4b-f16.gguf

# 3. Quantize to Q4_K_M (recommended)
./quantize qwen3-4b-f16.gguf qwen3-4b-q4.gguf Q4_K_M
```

**That's it!** Now use `qwen3-4b-q4.gguf` with DenseCore.

### Or Use Python API (Even Easier!)

```python
from densecore import convert_from_hf

# One-liner conversion!
gguf_path = convert_from_hf("Qwen/Qwen3-4B", quantization="Q4_K_M")
print(f"Model saved to: {gguf_path}")
```

**That's all you need!** DenseCore automatically:
- ✅ Finds llama.cpp converter (or guides you to install)
- ✅ Downloads the HF model
- ✅ Converts to GGUF
- ✅ Quantizes to your chosen format

---

## Prerequisites

### 1. Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

### 2. Install Python Dependencies

```bash
pip install torch transformers sentencepiece protobuf
```

---

## Step-by-Step Guide

### Step 1: Download HuggingFace Model

The conversion script will auto-download, but you can pre-download:

```bash
huggingface-cli download Qwen/Qwen3-4B
```

Or let the converter download automatically (recommended).

### Step 2: Convert to GGUF (F16)

```bash
cd llama.cpp
python3 convert-hf-to-gguf.py MODEL_NAME --outfile OUTPUT.gguf
```

**Examples:**

```bash
# Qwen3-4B
python3 convert-hf-to-gguf.py Qwen/Qwen3-4B --outfile qwen3-4b-f16.gguf

# Llama 3.1 8B
python3 convert-hf-to-gguf.py meta-llama/Meta-Llama-3.1-8B-Instruct --outfile llama3.1-8b-f16.gguf

# Mistral 7B
python3 convert-hf-to-gguf.py mistralai/Mistral-7B-Instruct-v0.2 --outfile mistral-7b-f16.gguf
```

**Output:** F16 GGUF file (~2x model size, e.g., 8B → 16GB)

### Step 3: Quantize (Compress)

Quantization reduces file size with minimal quality loss:

```bash
./quantize INPUT.gguf OUTPUT.gguf QUANT_TYPE
```

**Quantization Types:**

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| `Q4_K_M` | ~2.5GB | ⭐⭐⭐⭐ | **Recommended** - Best balance |
| `Q5_K_M` | ~3GB | ⭐⭐⭐⭐⭐ | Higher quality, slightly larger |
| `Q8_0` | ~4.5GB | ⭐⭐⭐⭐⭐ | Near-original quality |
| `Q4_0` | ~2.2GB | ⭐⭐⭐ | Maximum compression |

**Example:**

```bash
./quantize qwen3-4b-f16.gguf qwen3-4b-q4.gguf Q4_K_M
```

---

## Python API (Recommended)

DenseCore provides a native Python API for model conversion - no manual CLI commands needed!

### Basic Usage

```python
from densecore import convert_from_hf

# Convert with default settings (Q4_K_M)
gguf_path = convert_from_hf("Qwen/Qwen3-4B")

# Or specify quantization level
gguf_path = convert_from_hf(
    "Qwen/Qwen3-4B",
   output_path="./models/qwen3.gguf",
    quantization="Q5_K_M"  # Higher quality
)
```

### Quick Convert

For fastest conversion with recommended settings:

```python
from densecore import quick_convert

# One-liner with best defaults
path = quick_convert("Qwen/Qwen3-4B")
```

### Full Example

```python
from densecore import convert_from_hf, DenseCore

# 1. Convert model
print("Converting Qwen3-4B...")
gguf_path = convert_from_hf(
    model_id="Qwen/Qwen3-4B",
    output_path="qwen3-q4.gguf",
    quantization="Q4_K_M"
)

# 2. Use immediately
print("Loading model...")
engine = DenseCore(gguf_path, hf_repo_id="Qwen/Qwen3-4B")

# 3. Generate
result = engine.generate("The capital of France is")
print(result)
```

### Available Quantization Types

```python
# Fastest, smallest (2.2GB for 7B model)
convert_from_hf("model-id", quantization="Q4_0")

# Recommended balance (2.5GB)
convert_from_hf("model-id", quantization="Q4_K_M")

# Higher quality (3GB)
convert_from_hf("model-id", quantization="Q5_K_M")

# Near-original quality (4.5GB)
convert_from_hf("model-id", quantization="Q8_0")

# No quantization (full precision, 16GB)
convert_from_hf("model-id", quantization="F16")
```

---

## CLI Method (Alternative)

If you prefer command-line tools or don't have Python available:

## Using with DenseCore

### CLI Method (Alternative)

```python
from densecore import DenseCore

# Use your converted model
engine = DenseCore(
    main_model_path="path/to/qwen3-4b-q4.gguf",
    hf_repo_id="Qwen/Qwen3-4B",  # For tokenizer
    threads=4
)

result = engine.generate("The capital of France is", max_tokens=50)
print(result)
```

### CLI Example Script

```bash
cd DenseCore/python
python example.py --model-path qwen3-4b-q4.gguf --hf-repo Qwen/Qwen3-4B
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'gguf'"

Install llama.cpp Python requirements:
```bash
pip install -r llama.cpp/requirements.txt
```

### "RuntimeError: Unsupported model architecture"

The model architecture may not be supported by llama.cpp yet. Check:
- [llama.cpp supported models](https://github.com/ggerganov/llama.cpp#supported-models)
- Use a compatible model or wait for llama.cpp update

### Conversion is Very Slow

**Normal.** Converting 7B+ models takes 5-15 minutes. Quantization is faster (1-3 min).

### Out of Memory During Conversion

**F16 conversion** requires ~2x model RAM (e.g., 7B model needs ~16GB RAM)

**Solutions:**
- Close other applications
- Use a machine with more RAM
- Use a smaller model variant

---

## Recommended Models

### For Qwen3 (Latest, Best Multilingual)

```bash
# Qwen3-4B (Good for most tasks)
python3 convert-hf-to-gguf.py Qwen/Qwen3-4B --outfile qwen3-4b-f16.gguf
./quantize qwen3-4b-f16.gguf qwen3-4b-q4.gguf Q4_K_M

# Qwen3-8B (Better quality, more RAM)
python3 convert-hf-to-gguf.py Qwen/Qwen3-8B --outfile qwen3-8b-f16.gguf
./quantize qwen3-8b-f16.gguf qwen3-8b-q4.gguf Q4_K_M
```

### For Llama (Best English, Most Compatible)

```bash
# Llama 3.1 8B
python3 convert-hf-to-gguf.py meta-llama/Meta-Llama-3.1-8B-Instruct --outfile llama3.1-8b-f16.gguf
./quantize llama3.1-8b-f16.gguf llama3.1-8b-q4.gguf Q4_K_M
```

### For Mistral (Compact, Fast)

```bash
# Mistral 7B
python3 convert-hf-to-gguf.py mistralai/Mistral-7B-Instruct-v0.2 --outfile mistral-7b-f16.gguf
./quantize mistral-7b-f16.gguf mistral-7b-q4.gguf Q4_K_M
```

---

## Advanced: Batch Conversion Script

Save as `convert_model.sh`:

```bash
#!/bin/bash
MODEL_ID=$1
OUTPUT_NAME=$2
QUANT=${3:-Q4_K_M}

echo "Converting $MODEL_ID..."

# Convert to F16
python3 llama.cpp/convert-hf-to-gguf.py $MODEL_ID --outfile ${OUTPUT_NAME}-f16.gguf

# Quantize
./llama.cpp/quantize ${OUTPUT_NAME}-f16.gguf ${OUTPUT_NAME}-${QUANT}.gguf $QUANT

# Cleanup F16 (optional)
# rm ${OUTPUT_NAME}-f16.gguf

echo "Done! Output: ${OUTPUT_NAME}-${QUANT}.gguf"
```

**Usage:**
```bash
chmod +x convert_model.sh
./convert_model.sh Qwen/Qwen3-4B qwen3-4b Q4_K_M
```

---

## Next Steps

- ✅ Model converted? → [Use with DenseCore](../README.md)
- ❓ Issues? → [Check compatibility guide](GGUF_COMPATIBILITY.md)
- 🚀 Production deployment? → [Server setup guide](../server/README.md)
