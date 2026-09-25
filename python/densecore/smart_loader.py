"""
RAM-Aware Smart Model Loader for DenseCore.

This module provides intelligent model loading that automatically selects
the optimal quantization level based on available system RAM. It helps
developers who are unfamiliar with quantization to easily load models
that fit their hardware constraints.

Example:
    >>> import densecore

    # Auto-select best quantization for your system
    >>> model = densecore.smart_load("TheBloke/Llama-2-7B-GGUF")
    [Smart Loader] Available RAM: 16.0 GB
    [Smart Loader] Selected: Q4_K_M (4.1 GB) - fits comfortably

    # Or use with from_pretrained
    >>> model = densecore.from_pretrained(
    ...     "TheBloke/Llama-2-7B-GGUF",
    ...     auto_select_quant=True
    ... )
"""

import os
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .engine import DenseCore


@dataclass
class SystemResources:
    """
    System hardware information for memory-aware loading.

    Attributes:
        total_ram_gb: Total system RAM in gigabytes
        available_ram_gb: Currently available RAM in gigabytes
        cpu_cores: Number of CPU cores (physical)
        cpu_threads: Number of CPU threads (logical)
    """

    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_threads: int

    def __str__(self) -> str:
        return (
            f"SystemResources(RAM: {self.available_ram_gb:.1f}/{self.total_ram_gb:.1f} GB, "
            f"CPU: {self.cpu_cores} cores / {self.cpu_threads} threads)"
        )


def get_system_resources() -> SystemResources:
    """
    Detect available system resources.

    Returns:
        SystemResources: Container with RAM and CPU information

    Example:
        >>> resources = get_system_resources()
        >>> print(f"Available RAM: {resources.available_ram_gb:.1f} GB")
        Available RAM: 12.5 GB
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)

        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1

    except ImportError:
        # Fallback without psutil
        total_ram_gb = _get_ram_fallback()
        available_ram_gb = total_ram_gb * 0.7  # Assume 70% available

        cpu_threads = os.cpu_count() or 1
        cpu_cores = max(1, cpu_threads // 2)  # Estimate physical cores

    return SystemResources(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
    )


def _get_ram_fallback() -> float:
    """Get total RAM without psutil (platform-specific)."""
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal: 16384000 kB
                        kb = int(line.split()[1])
                        return kb / (1024**2)  # Convert KB to GB
        except (OSError, ValueError):
            pass

    elif system == "Darwin":  # macOS
        try:
            import subprocess

            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            if result.returncode == 0:
                bytes_ram = int(result.stdout.strip())
                return bytes_ram / (1024**3)
        except (subprocess.SubprocessError, ValueError):
            pass

    elif system == "Windows":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong

            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", c_ulong),
                    ("dwMemoryLoad", c_ulong),
                    ("dwTotalPhys", c_ulong),
                    ("dwAvailPhys", c_ulong),
                    ("dwTotalPageFile", c_ulong),
                    ("dwAvailPageFile", c_ulong),
                    ("dwTotalVirtual", c_ulong),
                    ("dwAvailVirtual", c_ulong),
                ]

            mem_status = MEMORYSTATUS()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(mem_status))
            return mem_status.dwTotalPhys / (1024**3)
        except Exception:
            pass

    # Default fallback: assume 8 GB
    return 8.0


# Quantization tiers ordered by quality (highest first)
# Maps quant type to approximate compression ratio vs FP16
QUANT_COMPRESSION_RATIOS = {
    "F16": 1.0,  # No compression
    "BF16": 1.0,
    "Q8_0": 0.5,  # 2x compression
    "Q6_K": 0.4,  # 2.5x compression
    "Q5_K_M": 0.35,  # ~3x compression
    "Q5_K_S": 0.33,
    "Q5_0": 0.33,
    "Q4_K_M": 0.28,  # ~3.5x compression (RECOMMENDED)
    "Q4_K_S": 0.26,
    "Q4_0": 0.25,  # 4x compression
    "Q4_1": 0.27,
    "IQ4_XS": 0.24,
    "Q3_K_M": 0.22,  # ~4.5x compression
    "Q3_K_S": 0.20,
    "IQ3_M": 0.20,
    "Q2_K": 0.15,  # ~6.5x compression (lowest quality)
    "IQ2_M": 0.14,
}

# Ordered by quality (best first, for selecting highest quality that fits)
QUANT_QUALITY_ORDER = [
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "IQ4_XS",
    "Q3_K_M",
    "Q3_K_S",
    "IQ3_M",
    "Q2_K",
    "IQ2_M",
]


def estimate_model_memory(
    file_size_bytes: int,
    safety_multiplier: float = 1.2,
) -> float:
    """
    Estimate runtime memory usage for a GGUF model.

    GGUF models require additional memory at runtime for:
    - KV cache
    - Temporary computation buffers
    - Model metadata

    Args:
        file_size_bytes: Size of the GGUF file in bytes
        safety_multiplier: Multiplier for safety margin (default 1.2 = 20% extra)

    Returns:
        Estimated memory usage in GB

    Example:
        >>> estimate_model_memory(4 * 1024**3)  # 4 GB file
        5.25  # ~5.25 GB estimated runtime usage
    """
    file_size_gb = file_size_bytes / (1024**3)

    # Base memory: file size + overhead
    # KV cache and buffers typically add 10-30% depending on context length
    estimated_gb = file_size_gb * safety_multiplier

    # Minimum overhead for small models
    min_overhead_gb = 0.5

    return max(estimated_gb, file_size_gb + min_overhead_gb)


def recommend_quantization(
    repo_id: str,
    available_ram_gb: Optional[float] = None,
    safety_margin: float = 0.8,
    token: Optional[str] = None,
    verbose: bool = True,
) -> tuple[Optional[str], str]:
    """
    Recommend the best quantization level for available system RAM.

    Analyzes available GGUF files in a HuggingFace repository and selects
    the highest quality quantization that fits within available RAM.

    Args:
        repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        available_ram_gb: Override auto-detected available RAM (GB)
        safety_margin: Fraction of available RAM to use (0.8 = use 80%)
        token: HuggingFace API token for private repos
        verbose: Print informational messages

    Returns:
        Tuple of (recommended_filename, reason_message)
        If no suitable file found, returns (None, error_message)

    Example:
        >>> filename, msg = recommend_quantization("TheBloke/TinyLlama-1.1B-GGUF")
        >>> print(msg)
        Selected Q4_K_M (0.7 GB) - highest quality that fits in 12.5 GB available RAM
    """
    from .hub import list_gguf_files

    # Get system resources
    if available_ram_gb is None:
        resources = get_system_resources()
        available_ram_gb = resources.available_ram_gb

    usable_ram_gb = available_ram_gb * safety_margin

    if verbose:
        print(
            f"[Smart Loader] Available RAM: {available_ram_gb:.1f} GB (using {usable_ram_gb:.1f} GB with {safety_margin:.0%} safety margin)"
        )

    # Get available GGUF files
    try:
        files = list_gguf_files(repo_id, token=token)
    except Exception as e:
        return None, f"Failed to list files in {repo_id}: {e}"

    if not files:
        return None, f"No GGUF files found in {repo_id}"

    # Analyze each file and find best fit
    candidates = []
    for f in files:
        file_size = f.get("size", 0)
        quant = f.get("quant")
        filename = f["filename"]

        if file_size == 0:
            # Skip files with unknown size
            continue

        estimated_memory = estimate_model_memory(file_size)
        fits = estimated_memory <= usable_ram_gb

        # Get quality rank (lower is better)
        quality_rank = QUANT_QUALITY_ORDER.index(quant) if quant in QUANT_QUALITY_ORDER else 999

        candidates.append(
            {
                "filename": filename,
                "quant": quant,
                "size_gb": file_size / (1024**3),
                "estimated_memory_gb": estimated_memory,
                "fits": fits,
                "quality_rank": quality_rank,
            }
        )

    if not candidates:
        return None, f"No GGUF files with size information in {repo_id}"

    # Sort by quality (best first), then by size (smaller first as tiebreaker)
    candidates.sort(key=lambda x: (x["quality_rank"], x["size_gb"]))

    # Find best fitting candidate
    fitting = [c for c in candidates if c["fits"]]

    if fitting:
        best = fitting[0]
        msg = (
            f"Selected {best['quant'] or 'unknown'} ({best['size_gb']:.1f} GB) - "
            f"highest quality that fits in {available_ram_gb:.1f} GB RAM"
        )
        return best["filename"], msg

    # No file fits - recommend smallest with warning
    smallest = min(candidates, key=lambda x: x["size_gb"])
    msg = (
        f"⚠️ WARNING: No quantization fits in {available_ram_gb:.1f} GB RAM. "
        f"Smallest available: {smallest['quant'] or 'unknown'} ({smallest['size_gb']:.1f} GB, "
        f"needs ~{smallest['estimated_memory_gb']:.1f} GB). "
        f"Consider using a smaller model or adding more RAM."
    )

    # Return smallest anyway (user can decide to proceed)
    return smallest["filename"], msg


def smart_load(
    repo_id: str,
    allow_fallback: bool = True,
    min_quality: str = "Q2_K",
    verbose: bool = True,
    **kwargs,
) -> "DenseCore":
    """
    Smart model loader with automatic RAM-aware quantization selection.

    This function automatically detects available system RAM and selects
    the highest quality quantization that will fit comfortably. It's designed
    for developers who are unfamiliar with quantization options.

    Args:
        repo_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        allow_fallback: If True, auto-select smaller quant when preferred doesn't fit
        min_quality: Minimum acceptable quantization (refuse to load below this)
        verbose: Print informational messages
        **kwargs: Additional arguments passed to DenseCore (threads, etc.)

    Returns:
        DenseCore: Initialized model instance

    Raises:
        MemoryError: If no quantization fits and allow_fallback is False
        ValueError: If no suitable GGUF files found

    Example:
        >>> import densecore

        # Automatic selection
        >>> model = densecore.smart_load("TheBloke/Llama-2-7B-GGUF")
        [Smart Loader] Available RAM: 16.0 GB (using 12.8 GB with 80% safety margin)
        [Smart Loader] Selected Q4_K_M (4.1 GB) - highest quality that fits in 16.0 GB RAM

        # With constraints
        >>> model = densecore.smart_load(
        ...     "TheBloke/Llama-2-7B-GGUF",
        ...     min_quality="Q4_K_M",  # Don't go below Q4_K_M quality
        ... )
    """
    from .hub import from_pretrained

    # Get recommendation
    filename, message = recommend_quantization(
        repo_id=repo_id,
        token=kwargs.pop("token", None),
        verbose=verbose,
    )

    if filename is None:
        raise ValueError(message)

    if verbose:
        print(f"[Smart Loader] {message}")

    # Check if we're falling back to a low-quality option
    if "WARNING" in message and not allow_fallback:
        raise MemoryError(
            f"Insufficient RAM for any quantization of {repo_id}. "
            f"Set allow_fallback=True to proceed anyway."
        )

    # Check minimum quality constraint
    detected_quant = None
    for q in QUANT_QUALITY_ORDER:
        if q.lower() in filename.lower():
            detected_quant = q
            break

    if detected_quant and min_quality:
        min_rank = (
            QUANT_QUALITY_ORDER.index(min_quality) if min_quality in QUANT_QUALITY_ORDER else 999
        )
        detected_rank = QUANT_QUALITY_ORDER.index(detected_quant)

        if detected_rank > min_rank:
            raise ValueError(
                f"Selected quantization {detected_quant} is below minimum quality {min_quality}. "
                f"Consider using a smaller model or increasing available RAM."
            )

    # Load the model
    return from_pretrained(
        repo_id,
        filename=filename,
        verbose=verbose,
        **kwargs,
    )


def get_recommended_threads(resources: Optional[SystemResources] = None) -> int:
    """
    Get recommended number of threads for inference.

    Uses physical core count for optimal performance, as hyperthreading
    typically doesn't help and may hurt inference performance.

    Args:
        resources: Pre-computed SystemResources, or None to auto-detect

    Returns:
        Recommended thread count
    """
    if resources is None:
        resources = get_system_resources()

    # Use physical cores, leave 1-2 for system
    recommended = max(1, resources.cpu_cores - 1)

    return recommended
