# Multi-LoRA Preloading Guide

DenseCore supports loading multiple LoRA adapters simultaneously with fast switching for efficient multi-tenant inference.

## Overview

**Key Features:**
- **Adapter Preloading Pool**: Keep frequently-used adapters in memory for instant switching
- **LRU Eviction**: Automatic memory management when pool capacity is exceeded
- **Request-Level Routing**: Specify which adapter to use per inference request
- **CPU-First Design**: Optimized for CPU inference (no CUDA required)

## Quick Start

```python
from densecore import DenseCore

# Load base model
model = DenseCore(model_path="base_model.gguf")

# Preload multiple adapters
model._lora_manager.preload([
    ("customer_support", "./adapters/customer_support.gguf", 1.0),
    ("code_assistant", "./adapters/code_assistant.gguf", 1.0),
    ("korean_chat", "./adapters/korean_chat.gguf", 0.8),
])

# Set pool capacity (optional, default=8)
model._lora_manager.set_pool_capacity(4)

# Generate with specific adapter
response1 = model.generate("Help me fix this bug", lora_adapter="code_assistant")
response2 = model.generate("고객 문의 도와주세요", lora_adapter="korean_chat")
```

## API Reference

### LoRAManager Methods

#### `preload(adapters)`

Load multiple adapters into memory without activating them.

```python
# adapters: List of (name, path, scale) tuples
loaded = model._lora_manager.preload([
    ("adapter1", "/path/to/adapter1.gguf", 1.0),
    ("adapter2", "/path/to/adapter2.gguf"),  # scale defaults to 1.0
])
print(f"Loaded: {loaded}")
```

#### `set_pool_capacity(capacity)`

Set maximum number of adapters to keep in memory.

```python
model._lora_manager.set_pool_capacity(8)  # Keep up to 8 adapters
```

### Generate with Adapter

Use `lora_adapter` parameter to specify which adapter to use:

```python
# Switch between adapters per-request
response = model.generate(
    prompt="Translate to French:",
    max_tokens=100,
    lora_adapter="french_translator"
)
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Python SDK                      │
│  LoRAManager                                 │
│  ├── preload()                              │
│  ├── set_pool_capacity()                    │
│  └── adapters: Dict[str, LoRAConfig]        │
└─────────────────┬───────────────────────────┘
                  │ ctypes
┌─────────────────▼───────────────────────────┐
│              C++ Core                        │
│  LoRAStorage                                 │
│  ├── Load() / Unload()                      │
│  ├── Activate() / DeactivateAll()           │
│  ├── LRU eviction                           │
│  └── adapters_: Map<name, LoRAAdapter>      │
└─────────────────────────────────────────────┘
```

## Performance Tips

1. **Preload frequently-used adapters** to eliminate load latency
2. **Set pool capacity** based on available memory
3. **Use request-level routing** for multi-tenant scenarios

## C API

```c
// Load adapter
int LoadLoraAdapter(DenseCoreHandle h, const char* path, float scale, const char* name);

// Activate/Deactivate
int ActivateLoraAdapter(DenseCoreHandle h, const char* name);
int DeactivateLoraAdapters(DenseCoreHandle h);

// Unload
int UnloadLoraAdapter(DenseCoreHandle h, const char* name);
```

## See Also

- [API Reference](API_REFERENCE.md)
- [Model Optimization](MODEL_OPTIMIZATION.md)
