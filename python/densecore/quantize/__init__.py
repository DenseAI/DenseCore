"""
DenseCore Quantization Module

Provides quantization utilities inspired by NVIDIA Model-Optimizer.
Supports INT4, INT8, and FP8 quantization with AWQ and SmoothQuant algorithms.
"""

import ctypes
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional

import densecore.engine as engine_mod


@dataclass
class QuantConfig:
    """
    Quantization configuration (mirrors core/include/quantization_config.h)

    Args:
        format: Quantization format ('fp16', 'fp8_e4m3', 'int8', 'int4_blockwise')
        algorithm: Quantization algorithm ('max', 'smoothquant', 'awq_lite', 'awq_clip')
        block_size: Block size for blockwise quantization (default 128)
        quantize_weights: Whether to quantize weights (default True)
        quantize_activations: Whether to quantize activations (default False)
        skip_output_layer: Skip lm_head/output quantization (default True)
        skip_embeddings: Skip embedding layer quantization (default False)

    Example:
        >>> config = QuantConfig(format="int4_blockwise", algorithm="awq_lite")
        >>> quantize_model("model.gguf", "model-q4.gguf", config)
    """

    format: Literal["fp16", "fp8_e4m3", "int8", "int4_blockwise"] = "int4_blockwise"
    algorithm: Literal["max", "smoothquant", "awq_lite", "awq_clip"] = "awq_lite"
    block_size: int = 128
    quantize_weights: bool = True
    quantize_activations: bool = False
    skip_output_layer: bool = True
    skip_embeddings: bool = False
    calib_size: int = 512
    calib_dataset: str = "cnn_dailymail"


# Load library
try:
    _lib = engine_mod.get_library()
except Exception:
    _lib = None

# Define C function signatures
if _lib is not None and hasattr(_lib, "QuantizeModel"):
    _lib.QuantizeModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    _lib.QuantizeModel.restype = ctypes.c_int


# Error code mapping
_ERROR_CODES = {
    0: "Success",
    -1: "Invalid input/output path",
    -2: "Failed to load model",
    -3: "Failed to create quantizer",
    -4: "Failed to save quantized model",
    -5: "Exception during quantization",
}


def quantize_model(
    input_path: str,
    output_path: str,
    config: Optional[QuantConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Quantize a GGUF model.

    Args:
        input_path: Path to input GGUF model
        output_path: Path to save quantized GGUF model
        config: Quantization configuration (defaults to INT4_AWQ_CFG)
        progress_callback: Optional callback(current, total, message) for progress

    Raises:
        FileNotFoundError: If input model doesn't exist
        RuntimeError: If quantization fails

    Example:
        >>> from densecore.quantize import quantize_model, INT4_AWQ_CFG
        >>> quantize_model("model.gguf", "model-q4.gguf", INT4_AWQ_CFG)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Model file not found: {input_path}")

    if config is None:
        config = INT4_AWQ_CFG

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert config to JSON
    config_dict = asdict(config)
    config_json = json.dumps(config_dict)

    print(f"Quantizing '{input_path}' -> '{output_path}'")
    print(f"  Format: {config.format}")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Block size: {config.block_size}")

    if _lib is None or not hasattr(_lib, "QuantizeModel"):
        raise RuntimeError(
            "DenseCore C++ library not available. "
            "Please rebuild the library with: cd core && mkdir -p build && cd build && cmake .. && make"
        )

    ret = _lib.QuantizeModel(
        input_path.encode("utf-8"), output_path.encode("utf-8"), config_json.encode("utf-8")
    )

    if ret != 0:
        error_msg = _ERROR_CODES.get(ret, f"Unknown error code: {ret}")
        raise RuntimeError(f"Quantization failed: {error_msg}")

    print(f"Quantization successful! Output saved to: {output_path}")


# Predefined configs (inspired by Model-Optimizer)
INT4_AWQ_CFG = QuantConfig(
    format="int4_blockwise",
    algorithm="awq_lite",
    block_size=128,
    quantize_weights=True,
    quantize_activations=False,
)

INT8_SMOOTHQUANT_CFG = QuantConfig(
    format="int8",
    algorithm="smoothquant",
    quantize_weights=True,
    quantize_activations=True,
)

FP8_DEFAULT_CFG = QuantConfig(
    format="fp8_e4m3",
    algorithm="max",
    quantize_weights=True,
    quantize_activations=True,
)

INT4_MAX_CFG = QuantConfig(
    format="int4_blockwise",
    algorithm="max",
    block_size=128,
)

INT8_MAX_CFG = QuantConfig(
    format="int8",
    algorithm="max",
)


__all__ = [
    "QuantConfig",
    "quantize_model",
    "INT4_AWQ_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "FP8_DEFAULT_CFG",
    "INT4_MAX_CFG",
    "INT8_MAX_CFG",
]
