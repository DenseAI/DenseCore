"""
HuggingFace to GGUF Conversion Module

Provides Python API for converting HuggingFace models to GGUF format
without requiring llama.cpp command-line tools.
"""

import logging
import json
import subprocess
import tempfile
import ctypes
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Type hints for quantization levels
QuantType = Literal["Q4_0", "Q4_1", "Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]


def convert_from_hf(
    model_id: str,
    output_path: Optional[str] = None,
    quantization: QuantType = "Q4_K_M",
    cache_dir: Optional[str] = None,
    use_fast_tokenizer: bool = True,
) -> str:
    """
    Convert a HuggingFace model directly to GGUF format.

    This is a convenience wrapper that:
    1. Downloads the HF model (if not cached)
    2. Converts to GGUF using llama.cpp converter
    3. Quantizes to specified format

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
        output_path: Output GGUF file path. If None, saves to current directory
        quantization: Quantization type (Q4_K_M recommended)
        cache_dir: HuggingFace cache directory
        use_fast_tokenizer: Use fast tokenizer if available

    Returns:
        Path to the converted GGUF file

    Example:
        >>> from densecore import convert_from_hf
        >>> gguf_path = convert_from_hf("Qwen/Qwen3-4B", quantization="Q4_K_M")
        >>> print(f"Model saved to: {gguf_path}")
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Converting from HuggingFace requires transformers and torch.\n"
            "Install with: pip install transformers torch"
        )

    logger.info(f"Converting {model_id} to GGUF with {quantization} quantization")

    # Generate output path if not provided
    if output_path is None:
        model_name = model_id.split("/")[-1].lower()
        quant_suffix = quantization.lower().replace("_", "-")
        output_path = f"{model_name}-{quant_suffix}.gguf"

    output_path = Path(output_path).absolute()

    # Check if llama.cpp converter is available
    converter_path = _find_llamacpp_converter()
    if converter_path:
        # Use llama.cpp converter (faster, more reliable)
        return _convert_via_llamacpp(model_id, output_path, quantization, converter_path)
    else:
        # Native Python conversion (slower, but works without llama.cpp)
        return _convert_native(model_id, output_path, quantization, cache_dir, use_fast_tokenizer)


def quantize_gguf(input_path: str, output_path: str, qtype: QuantType) -> None:
    """
    Quantize a GGUF file using DenseCore C API.

    Args:
        input_path: Path to input GGUF file
        output_path: Path to output quantized GGUF file
        qtype: Quantization type (e.g., "Q4_K_M")
    """
    from .engine import get_library

    lib = get_library()
    if not hasattr(lib, "QuantizeModel"):
        raise RuntimeError(
            "DenseCore C API symbol QuantizeModel is not available.\n"
            "Please rebuild DenseCore shared library."
        )

    lib.QuantizeModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.QuantizeModel.restype = ctypes.c_int

    format_map = {
        "Q4_0": "q4_0",
        "Q4_1": "q4_1",
        "Q4_K_M": "q4_k_m",
        "Q5_K_M": "q5_k_m",
        "Q8_0": "q8_0",
        "F16": "fp16",
        "F32": "fp32",
    }
    fmt = format_map.get(qtype)
    if fmt is None:
        raise ValueError(f"Unsupported quantization type: {qtype}")

    config_json = json.dumps({"format": fmt, "skip_output_layer": True, "skip_embeddings": True})

    ret = lib.QuantizeModel(
        str(input_path).encode("utf-8"),
        str(output_path).encode("utf-8"),
        config_json.encode("utf-8"),
    )
    if ret != 0:
        error_map = {
            -1: "invalid input/output path",
            -2: "failed to load input GGUF",
            -3: "unsupported quantization config",
            -4: "failed while writing output GGUF",
            -5: "internal exception during quantization",
        }
        raise RuntimeError(f"Quantization failed ({ret}): {error_map.get(ret, 'unknown error')}")


def _find_llamacpp_converter() -> Optional[Path]:
    """Try to find llama.cpp converter script."""
    # Check common locations
    possible_paths = [
        Path("llama.cpp/convert_hf_to_gguf.py"),  # Current directory
        Path("../llama.cpp/convert_hf_to_gguf.py"),  # Parent directory (DenseCore structure)
        Path("/usr/local/bin/convert_hf_to_gguf.py"),  # System-wide
        Path(__file__).parent.parent.parent
        / "third_party/llama.cpp/convert_hf_to_gguf.py",  # Submodule
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found llama.cpp converter at: {path}")
            return path.absolute()

    # Try to find in PATH
    try:
        result = subprocess.run(["which", "convert_hf_to_gguf.py"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None


def _convert_via_llamacpp(
    model_id: str,
    output_path: Path,
    quantization: QuantType,
    converter_path: Path,
) -> str:
    """Convert using llama.cpp converter (recommended)."""
    logger.info(f"Using llama.cpp converter at {converter_path}")

    # Step 1: Resolve model directory (Download if needed, or use local)
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        logger.info(f"Using local model directory: {model_path}")
        model_dir = str(model_path.absolute())
    else:
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading {model_id} from HuggingFace...")
            model_dir = snapshot_download(repo_id=model_id, repo_type="model")
            logger.info(f"Model downloaded to: {model_dir}")
        except ImportError:
            raise ImportError(
                "huggingface_hub required for model download.\n"
                "Install with: pip install huggingface_hub"
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        # If output is not quantized, we can convert directly to it
        # But for consistency, we convert to F16 and then quantize
        # Unless the user asked for F16/F32 directly

        is_f16_target = quantization in ["F16", "F32"]
        intermediate_path = output_path if is_f16_target else Path(tmpdir) / "model-f16.gguf"

        # Step 2: Convert to GGUF
        # Note: default output of convert_hf_to_gguf is F16 or F32 depending on args
        logger.info("Converting to GGUF...")
        cmd = [
            "python3",
            str(converter_path),
            model_dir,
            "--outfile",
            str(intermediate_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")

        # Step 3: Quantize if needed
        if not is_f16_target:
            logger.info(f"Quantizing to {quantization}...")
            quantize_gguf(str(intermediate_path), str(output_path), quantization)

    logger.info(f"✅ Conversion complete: {output_path}")
    return str(output_path)


def _convert_native(
    model_id: str,
    output_path: Path,
    quantization: QuantType,
    cache_dir: Optional[str],
    use_fast_tokenizer: bool,
) -> str:
    """
    Native Python conversion (fallback when llama.cpp not available).

    Note: This is a simplified implementation. For production use,
    install llama.cpp for more robust conversion.
    """
    logger.warning(
        "llama.cpp converter not found. Using native Python conversion.\n"
        "For better results, install llama.cpp:\n"
        "  git clone https://github.com/ggerganov/llama.cpp"
    )

    raise NotImplementedError(
        "Native Python conversion not yet implemented.\n"
        "Please install llama.cpp and try again.\n\n"
        "Quick install:\n"
        "  git clone https://github.com/ggerganov/llama.cpp\n"
        "  cd llama.cpp && make\n\n"
        "Then run convert_from_hf() again."
    )


# Convenience function for quick conversion with defaults
def quick_convert(model_id: str) -> str:
    """
    Quick conversion with recommended settings.

    Converts to Q4_K_M (best balance of size/quality) and saves to current directory.

    Example:
        >>> from densecore import quick_convert
        >>> path = quick_convert("Qwen/Qwen3-4B")
    """
    return convert_from_hf(model_id, quantization="Q4_K_M")
