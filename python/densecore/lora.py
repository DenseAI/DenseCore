"""
LoRA (Low-Rank Adaptation) Adapter Support for DenseCore.

This module provides functionality to load and apply LoRA adapters for inference,
similar to how HuggingFace transformers handles PEFT adapters. LoRA adapters allow
fine-tuning large models with minimal additional parameters.

Key Features:
- Auto-detect if a HuggingFace repo contains LoRA adapter vs base model
- Load LoRA adapters in GGUF format
- Apply/remove adapters at runtime
- Download adapters from HuggingFace Hub

Example:
    >>> import densecore

    # Load a LoRA fine-tuned model (auto-detects adapter)
    >>> model = densecore.from_pretrained("user/my-lora-adapter")
    [LoRA] Detected LoRA adapter, loading with base model...

    # Or explicitly load adapter on top of base
    >>> model = densecore.DenseCore(
    ...     model_path="base_model.gguf",
    ...     lora_adapter_path="adapter.gguf"
    ... )

    # Load adapter at runtime
    >>> model.load_lora("path/to/adapter.gguf", scale=0.8)

    # Disable adapter temporarily
    >>> model.disable_lora()
    >>> model.enable_lora()
"""

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .engine import DenseCore


@dataclass
class LoRAConfig:
    """
    Configuration for a LoRA adapter.

    Attributes:
        adapter_path: Path to the GGUF LoRA adapter file
        scale: LoRA scaling factor (alpha). Higher values = stronger adapter effect
        enabled: Whether the adapter is currently active

    Example:
        >>> config = LoRAConfig(
        ...     adapter_path="./my-adapter.gguf",
        ...     scale=1.0
        ... )
    """

    adapter_path: str
    scale: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ValueError(f"LoRA scale must be >= 0, got {self.scale}")


@dataclass
class LoRAAdapterInfo:
    """
    Information about a LoRA adapter from HuggingFace Hub.

    Attributes:
        repo_id: HuggingFace repository ID
        adapter_filename: GGUF adapter file name
        base_model_id: Base model this adapter was trained on (if known)
        description: Adapter description from model card
        size_bytes: File size in bytes
    """

    repo_id: str
    adapter_filename: str
    base_model_id: Optional[str] = None
    description: Optional[str] = None
    size_bytes: int = 0

    def size_human(self) -> str:
        """Human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


# Patterns to detect LoRA adapters
LORA_PATTERNS = [
    r"lora",
    r"adapter",
    r"peft",
    r"ft\b",  # fine-tune
    r"finetune",
]

# Files that indicate a LoRA adapter repository
LORA_INDICATOR_FILES = [
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
]


def is_lora_adapter_repo(repo_id: str, token: Optional[str] = None) -> bool:
    """
    Check if a HuggingFace repository contains a LoRA adapter.

    Detection methods:
    1. Check for PEFT indicator files (adapter_config.json, etc.)
    2. Check repo name patterns (contains "lora", "adapter", etc.)
    3. Check model card for LoRA/PEFT mentions

    Args:
        repo_id: HuggingFace repository ID
        token: Optional HuggingFace API token

    Returns:
        True if the repo appears to be a LoRA adapter

    Example:
        >>> is_lora_adapter_repo("user/llama-2-lora-finetuned")
        True
        >>> is_lora_adapter_repo("meta-llama/Llama-2-7B")
        False
    """
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError:
        # Can't check without huggingface_hub, assume not LoRA
        return False

    # Check repo name patterns
    repo_lower = repo_id.lower()
    for pattern in LORA_PATTERNS:
        if re.search(pattern, repo_lower):
            return True

    # Check for PEFT indicator files
    try:
        files = list_repo_files(repo_id, token=token)
        for indicator in LORA_INDICATOR_FILES:
            if indicator in files:
                return True
    except Exception:
        pass

    return False


def list_lora_files(
    repo_id: str,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List LoRA adapter files in a repository.

    Looks for GGUF files that appear to be LoRA adapters,
    as well as PEFT format adapters.

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token

    Returns:
        List of adapter file information dictionaries
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

    api = HfApi(token=token)

    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        raise RuntimeError(f"Failed to list files in {repo_id}: {e}")

    adapter_files = []

    for filename in files:
        # Check for LoRA GGUF files
        if filename.endswith(".gguf"):
            name_lower = filename.lower()
            is_lora = any(p in name_lower for p in ["lora", "adapter"])

            if is_lora:
                try:
                    info = api.get_paths_info(repo_id, paths=[filename])
                    size = info[0].size if info else 0
                except Exception:
                    size = 0

                adapter_files.append(
                    {
                        "filename": filename,
                        "format": "gguf",
                        "size": size,
                    }
                )

        # Check for PEFT format
        elif filename in LORA_INDICATOR_FILES:
            adapter_files.append(
                {
                    "filename": filename,
                    "format": "peft",
                    "size": 0,
                }
            )

    return adapter_files


def download_lora_adapter(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = "main",
) -> str:
    """
    Download a LoRA adapter from HuggingFace Hub.

    If the repository contains PEFT format adapters, this function
    will guide the user to convert them to GGUF format.

    Args:
        repo_id: HuggingFace repository ID
        filename: Specific adapter file to download
        cache_dir: Directory to cache downloads
        token: HuggingFace API token
        revision: Git revision

    Returns:
        Path to the downloaded adapter file

    Raises:
        ValueError: If no suitable adapter file found
        RuntimeError: If download fails

    Example:
        >>> path = download_lora_adapter("user/my-lora-adapter")
        >>> print(path)
        /home/user/.cache/huggingface/.../lora-adapter.gguf
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

    # Find adapter files
    adapter_files = list_lora_files(repo_id, token=token)

    if not adapter_files:
        raise ValueError(
            f"No LoRA adapter files found in {repo_id}. Expected GGUF format adapter files."
        )

    # Filter to GGUF format
    gguf_adapters = [f for f in adapter_files if f["format"] == "gguf"]
    peft_adapters = [f for f in adapter_files if f["format"] == "peft"]

    if not gguf_adapters and peft_adapters:
        raise ValueError(
            f"Repository {repo_id} contains PEFT format adapters but not GGUF format. "
            f"Please convert using: llama.cpp/convert_lora_to_gguf.py or "
            f"use HuggingFace's 'GGUF-my-LoRA' tool."
        )

    # Select file
    if filename:
        matching = [f for f in gguf_adapters if f["filename"] == filename]
        if not matching:
            raise ValueError(f"File '{filename}' not found in {repo_id}")
        selected = matching[0]
    else:
        selected = gguf_adapters[0]

    print(f"[LoRA] Downloading adapter: {selected['filename']}")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=selected["filename"],
        cache_dir=cache_dir,
        token=token,
        revision=revision,
    )

    return local_path


def get_base_model_from_adapter(
    adapter_path: str,
) -> Optional[str]:
    """
    Try to determine the base model from a LoRA adapter.

    Reads adapter metadata to find the base model it was trained on.

    Args:
        adapter_path: Path to GGUF adapter file

    Returns:
        Base model identifier if found, None otherwise
    """
    # TODO: Parse GGUF metadata for base model info
    # For now, return None - user must specify base model
    return None


class LoRAManager:
    """
    Manages LoRA adapters for a DenseCore model instance.

    This class handles loading, switching, and managing multiple LoRA adapters
    on a single base model. When integrated with a DenseCore engine, it syncs
    Python state with C++ engine state.

    Attributes:
        active_adapter: Currently active LoRA configuration
        adapters: Dictionary of loaded adapters by name
        engine: Optional reference to DenseCore engine for C++ sync
    """

    def __init__(self, engine: Optional["DenseCore"] = None) -> None:
        self.adapters: Dict[str, LoRAConfig] = {}
        self.active_adapter: Optional[str] = None
        self._engine = engine  # Optional engine reference for C++ sync

    def load(
        self,
        name: str,
        adapter_path: str,
        scale: float = 1.0,
        activate: bool = True,
    ) -> LoRAConfig:
        """
        Load a LoRA adapter.

        Args:
            name: Identifier for this adapter
            adapter_path: Path to the GGUF adapter file
            scale: LoRA scaling factor
            activate: Whether to activate this adapter immediately

        Returns:
            LoRAConfig for the loaded adapter
        """
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

        config = LoRAConfig(
            adapter_path=adapter_path,
            scale=scale,
            enabled=activate,
        )

        self.adapters[name] = config

        if activate:
            self.active_adapter = name

        return config

    def unload(self, name: str) -> None:
        """Unload and remove an adapter."""
        if name in self.adapters:
            del self.adapters[name]
            if self.active_adapter == name:
                self.active_adapter = None

    def activate(self, name: str) -> None:
        """Activate a loaded adapter."""
        if name not in self.adapters:
            raise KeyError(f"Adapter '{name}' not loaded")
        self.adapters[name].enabled = True
        self.active_adapter = name

    def deactivate(self) -> None:
        """Deactivate all adapters (use base model only)."""
        if self.active_adapter:
            self.adapters[self.active_adapter].enabled = False
        self.active_adapter = None

    def get_active(self) -> Optional[LoRAConfig]:
        """Get the currently active adapter config."""
        if self.active_adapter and self.active_adapter in self.adapters:
            return self.adapters[self.active_adapter]
        return None

    def list_adapters(self) -> List[str]:
        """List all loaded adapter names."""
        return list(self.adapters.keys())

    @property
    def is_active(self) -> bool:
        """Check if any adapter is currently active."""
        config = self.get_active()
        return config is not None and config.enabled

    def preload(
        self,
        adapters: List[tuple],
    ) -> List[str]:
        """
        Preload multiple LoRA adapters into memory.

        Preloading eliminates loading latency during adapter switching,
        enabling fast runtime switching between adapters.

        Args:
            adapters: List of tuples (name, path, scale).
                      Each tuple contains (adapter name, GGUF file path, scale factor)

        Returns:
            List of successfully loaded adapter names

        Example:
            >>> manager.preload([
            ...     ("adapter1", "./adapter1.gguf", 1.0),
            ...     ("adapter2", "./adapter2.gguf", 0.8),
            ... ])
            ['adapter1', 'adapter2']
        """
        loaded_names: List[str] = []

        for item in adapters:
            if len(item) == 2:
                name, path = item
                scale = 1.0
            elif len(item) == 3:
                name, path, scale = item
            else:
                raise ValueError(
                    f"Invalid adapter tuple: {item}. Expected (name, path) or (name, path, scale)"
                )

            try:
                self.load(name, path, scale=scale, activate=False)
                loaded_names.append(name)
            except FileNotFoundError as e:
                import warnings

                warnings.warn(f"Skipping adapter '{name}': {e}", stacklevel=2)

        return loaded_names

    def set_pool_capacity(self, capacity: int) -> None:
        """
        Set the maximum number of adapters in the preloading pool.

        When more adapters are loaded than the pool capacity allows,
        the least recently used (LRU) adapters are automatically evicted
        to free memory.

        Args:
            capacity: Maximum number of adapters to keep in memory (default: 8)

        Example:
            >>> manager.set_pool_capacity(4)  # Keep up to 4 adapters
        """
        if capacity < 1:
            raise ValueError("Pool capacity must be at least 1")

        # Call C++ API if engine is available
        if self._engine is not None:
            if hasattr(self._engine, "_lib") and hasattr(self._engine._lib, "SetLoRAPoolCapacity"):
                self._engine._lib.SetLoRAPoolCapacity(self._engine._handle, capacity)
