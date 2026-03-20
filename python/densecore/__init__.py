"""
DenseCore - High-Performance CPU Inference Engine for AI Models

DenseCore is a production-ready inference engine optimized for running
AI models on CPU with HuggingFace integration.

Supported Model Types:
    - 🦙 LLMs: Llama, Qwen, Mistral, Gemma, Phi, DeepSeek
    - 👁️ Vision: ViT, CLIP, SigLIP
    - 🎤 Audio: Whisper (speech-to-text)
    - 🎨 Multimodal: LLaVA, Qwen-VL

Quick Start:
    >>> import densecore

    # Load from HuggingFace Hub (recommended)
    >>> model = densecore.from_pretrained("TheBloke/Llama-2-7B-GGUF")
    >>> response = model.generate("Hello, how are you?")

    # Or load from local file
    >>> model = densecore.DenseCore("./model.gguf")
    >>> response = model.generate("Hello!", max_tokens=100)

    # Streaming generation
    >>> for token in model.stream("Tell me a story"):
    ...     print(token, end="", flush=True)

    # Async support
    >>> async for token in model.stream_async("Hello"):
    ...     print(token, end="", flush=True)

Features:
    - 🚀 High-performance CPU inference with SIMD optimization
    - 🤗 Native HuggingFace Hub integration
    - 📦 GGUF format support (llama.cpp compatible)
    - 🔗 LangChain & LangGraph integration with tool calling
    - 🔄 Streaming and async support
    - 💾 Automatic model caching
"""

from typing import TYPE_CHECKING

# Core exports
from .auto import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .config import GenerationConfig, ModelConfig, SamplingParams
from .convert import convert_from_hf, quick_convert, quantize_gguf
from .embedding import EmbeddingConfig, EmbeddingModel, embed
from .engine import DenseCore
from .generate_output import (
    EosTokenCriteria,
    GenerateBeamOutput,
    GenerateOutput,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .hub import detect_model_type, download_model, from_pretrained, get_model_info, list_gguf_files
from .lora import LoRAAdapterInfo, LoRAConfig, LoRAManager
from .smart_loader import get_system_resources, recommend_quantization, smart_load

# Version info
__version__ = "1.0.0"
__author__ = "DenseCore Team"

# Type annotations for IDE support
if TYPE_CHECKING:
    from .engine import DenseCore as DenseCoreType

__all__ = [
    # Main class
    "DenseCore",
    # Auto classes (HuggingFace-style)
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    # Generation output (HuggingFace-compatible)
    "GenerateOutput",
    "GenerateBeamOutput",
    # Stopping criteria (HuggingFace-compatible)
    "StoppingCriteria",
    "StoppingCriteriaList",
    "MaxLengthCriteria",
    "MaxNewTokensCriteria",
    "EosTokenCriteria",
    # Embedding
    "EmbeddingModel",
    "EmbeddingConfig",
    "embed",
    # Config classes
    "GenerationConfig",
    "ModelConfig",
    "SamplingParams",
    # HuggingFace Hub functions
    "from_pretrained",
    "list_gguf_files",
    "download_model",
    "get_model_info",
    "detect_model_type",
    # Smart loading
    "smart_load",
    "recommend_quantization",
    "get_system_resources",
    # LoRA adapters
    "LoRAConfig",
    "LoRAAdapterInfo",
    "LoRAManager",
    # Conversion functions
    "convert_from_hf",
    "quick_convert",
    "quantize_gguf",
    # Version
    "__version__",
    # Integrations (lazy-loaded)
    "integrations",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_device_info() -> dict:
    """Get information about available compute devices."""
    import os
    import platform

    return {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }
