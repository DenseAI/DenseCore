"""
DenseCore Pruning Module

Provides pruning utilities for model compression inspired by NVIDIA Minitron.
Supports depth, width, attention, and combined pruning strategies.
"""

import ctypes
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional

import densecore.engine as engine_mod

# Load library
try:
    _lib = engine_mod.get_library()
except Exception:
    _lib = None

# Define C signatures
if _lib is not None and hasattr(_lib, "PruneModel"):
    _lib.PruneModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    _lib.PruneModel.restype = ctypes.c_int


# Error code mapping
_ERROR_CODES = {
    0: "Success",
    -1: "Invalid input/output path",
    -2: "Failed to load model",
    -3: "Failed to create pruner (invalid config)",
    -4: "Failed to save pruned model",
    -5: "Exception during pruning",
}


@dataclass
class PruneConfig:
    """
    Pruning configuration.

    Args:
        strategy: Pruning strategy
            - 'depth': Remove entire transformer layers
            - 'width': Reduce hidden dimensions
            - 'attention': Reduce number of attention heads
            - 'combined': Apply depth + width pruning in sequence
        importance_method: Method for scoring importance ('magnitude', 'l2_norm', 'activation')
        target_n_layer: Target number of layers (for depth pruning)
        target_hidden_size: Target hidden dimension (for width pruning)
        target_n_heads: Target number of attention heads (for attention pruning)
        target_ffn_hidden_size: Target FFN hidden size (for width pruning)

    Example:
        >>> config = PruneConfig(strategy="depth", target_n_layer=16)
        >>> prune_model("model.gguf", "model-pruned.gguf", config)
    """

    strategy: Literal["depth", "width", "attention", "combined"] = "depth"
    importance_method: Literal["magnitude", "l2_norm", "activation"] = "magnitude"
    target_n_layer: int = 0
    target_hidden_size: int = 0
    target_n_heads: int = 0
    target_ffn_hidden_size: int = 0


def prune_model(
    input_path: str,
    output_path: str,
    config: PruneConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """
    Prune a GGUF model.

    Args:
        input_path: Input GGUF model path
        output_path: Output GGUF model path
        config: Pruning configuration
        progress_callback: Optional callback(current, total, message) for progress

    Raises:
        FileNotFoundError: If input model doesn't exist
        RuntimeError: If pruning fails

    Example:
        >>> from densecore.prune import prune_model, DEPTH_PRUNE_50_CFG
        >>> prune_model("llama-7b.gguf", "llama-7b-pruned.gguf", DEPTH_PRUNE_50_CFG)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Model file not found: {input_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    config_dict = asdict(config)
    config_json = json.dumps(config_dict)

    print(f"Pruning '{input_path}' -> '{output_path}'")
    print(f"  Strategy: {config.strategy}")
    print(f"  Importance: {config.importance_method}")
    if config.target_n_layer > 0:
        print(f"  Target layers: {config.target_n_layer}")
    if config.target_hidden_size > 0:
        print(f"  Target hidden: {config.target_hidden_size}")
    if config.target_n_heads > 0:
        print(f"  Target heads: {config.target_n_heads}")

    if _lib is None or not hasattr(_lib, "PruneModel"):
        raise RuntimeError(
            "DenseCore C++ library not available. "
            "Please rebuild with: cd core && mkdir -p build && cd build && cmake .. && make"
        )

    ret = _lib.PruneModel(
        input_path.encode("utf-8"), output_path.encode("utf-8"), config_json.encode("utf-8")
    )

    if ret != 0:
        error_msg = _ERROR_CODES.get(ret, f"Unknown error code: {ret}")
        raise RuntimeError(f"Pruning failed: {error_msg}")

    print(f"Pruning successful! Output saved to: {output_path}")


# Predefined configs (inspired by NVIDIA Minitron)
DEPTH_PRUNE_50_CFG = PruneConfig(
    strategy="depth",
    importance_method="magnitude",
    target_n_layer=16,  # Typical: 32 -> 16 layers (50%)
)

DEPTH_PRUNE_33_CFG = PruneConfig(
    strategy="depth",
    importance_method="magnitude",
    target_n_layer=10,  # Typical: 32 -> ~10 layers (33%)
)

WIDTH_PRUNE_LLAMA_8B_TO_4B_CFG = PruneConfig(
    strategy="width",
    importance_method="l2_norm",
    target_hidden_size=3072,  # 4096 -> 3072 (75%)
    target_ffn_hidden_size=9216,  # 14336 -> 9216 (64%)
)

ATTENTION_PRUNE_50_CFG = PruneConfig(
    strategy="attention",
    importance_method="l2_norm",
    target_n_heads=16,  # 32 -> 16 heads (50%)
)

COMBINED_PRUNE_CFG = PruneConfig(
    strategy="combined",
    importance_method="magnitude",
    target_n_layer=24,  # 32 -> 24 layers
    target_hidden_size=3584,  # 4096 -> 3584
)


__all__ = [
    "PruneConfig",
    "prune_model",
    "DEPTH_PRUNE_50_CFG",
    "DEPTH_PRUNE_33_CFG",
    "WIDTH_PRUNE_LLAMA_8B_TO_4B_CFG",
    "ATTENTION_PRUNE_50_CFG",
    "COMBINED_PRUNE_CFG",
]
