"""
HuggingFace Hub integration for DenseCore.

This module provides seamless integration with the HuggingFace Hub,
allowing users to download and use GGUF models with a simple API.

Example:
    >>> import densecore

    # One-liner to load a model
    >>> model = densecore.from_pretrained("TheBloke/Llama-2-7B-GGUF")

    # List available GGUF files
    >>> files = densecore.list_gguf_files("TheBloke/Llama-2-7B-GGUF")
    >>> print(files)
    ['llama-2-7b.Q4_K_M.gguf', 'llama-2-7b.Q5_K_M.gguf', ...]

    # Download specific quantization
    >>> model = densecore.from_pretrained(
    ...     "TheBloke/Llama-2-7B-GGUF",
    ...     filename="llama-2-7b.Q4_K_M.gguf"
    ... )
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .engine import DenseCore

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]


# Model type detection keywords
MODEL_TYPE_KEYWORDS = {
    "llm": ["llama", "qwen", "mistral", "phi", "gemma", "deepseek", "gpt", "opt", "falcon"],
    "vision": ["clip", "vit", "siglip", "vision", "image", "dino", "eva"],
    "whisper": ["whisper", "audio", "speech", "asr"],
    "multimodal": ["llava", "qwen-vl", "qwen2-vl", "cogvlm", "internvl"],
}

# Common GGUF quantization types ordered by quality (highest first)
QUANT_PRIORITY = [
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


def list_gguf_files(
    repo_id: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> list[dict[str, Any]]:
    """
    List all GGUF files in a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        token: HuggingFace API token for private repos
        revision: Git revision (branch, tag, or commit)

    Returns:
        List of dictionaries with file information:
        - filename: Name of the GGUF file
        - size: File size in bytes
        - size_human: Human-readable size
        - quant: Detected quantization type

    Example:
        >>> files = list_gguf_files("TheBloke/Llama-2-7B-GGUF")
        >>> for f in files:
        ...     print(f"{f['filename']}: {f['size_human']}")
    """
    if HfApi is None:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

    api = HfApi(token=token)

    try:
        files = api.list_repo_files(repo_id, revision=revision)
    except Exception as e:
        raise RuntimeError(f"Failed to list files in {repo_id}: {e}")

    gguf_files = []
    for filename in files:
        if filename.endswith(".gguf"):
            # Try to get file info
            try:
                info = api.get_paths_info(repo_id, paths=[filename], revision=revision)
                if info:
                    size = info[0].size or 0
                else:
                    size = 0
            except Exception:
                size = 0

            # Detect quantization type
            quant = None
            for q in QUANT_PRIORITY:
                if q.lower() in filename.lower() or q.replace("_", ".") in filename:
                    quant = q
                    break

            gguf_files.append(
                {
                    "filename": filename,
                    "size": size,
                    "size_human": _format_size(size),
                    "quant": quant,
                }
            )

    # Sort by quantization priority
    def sort_key(f):
        quant = f.get("quant")
        if quant and quant in QUANT_PRIORITY:
            return QUANT_PRIORITY.index(quant)
        return 999

    gguf_files.sort(key=sort_key)

    return gguf_files


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_model(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = "main",
    force_download: bool = False,
    quant: Optional[str] = None,
) -> str:
    """
    Download a GGUF model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        filename: Specific file to download (auto-selected if not provided)
        cache_dir: Directory to cache the model
        token: HuggingFace API token
        revision: Git revision
        force_download: Force re-download even if cached
        quant: Preferred quantization (e.g., "Q4_K_M")

    Returns:
        Path to the downloaded model file

    Example:
        >>> path = download_model("TheBloke/Llama-2-7B-GGUF", quant="Q4_K_M")
        >>> print(path)
        /home/user/.cache/huggingface/hub/.../llama-2-7b.Q4_K_M.gguf
    """
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

    # If no filename specified, find the best GGUF file
    if filename is None:
        available = list_gguf_files(repo_id, token=token, revision=revision)

        if not available:
            raise ValueError(f"No GGUF files found in {repo_id}")

        # Filter by quantization if specified
        if quant:
            matching = [f for f in available if f.get("quant") == quant]
            if matching:
                filename = matching[0]["filename"]
            else:
                print(f"Warning: Quantization '{quant}' not found, using best available")

        if filename is None:
            # Prefer balanced defaults for broad usability.
            preferred_defaults = ["Q4_K_M", "Q5_K_M", "Q8_0"]
            selected = None
            for candidate_quant in preferred_defaults:
                candidate = next((f for f in available if f.get("quant") == candidate_quant), None)
                if candidate:
                    selected = candidate["filename"]
                    break

            filename = selected or available[0]["filename"]

    print(f"Downloading {filename} from {repo_id}...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        force_download=force_download,
    )

    print(f"Model downloaded to: {local_path}")
    return local_path


def from_pretrained(
    repo_id_or_path: str,
    filename: Optional[str] = None,
    auto_select_quant: bool = False,
    max_ram_usage_gb: Optional[float] = None,
    threads: int = 0,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: str = "main",
    quant: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> "DenseCore":
    """
    Load a DenseCore model from HuggingFace Hub or local path.

    This is the recommended way to load models, providing a familiar
    interface similar to HuggingFace Transformers. Automatically detects
    LoRA adapters and selects optimal quantization based on system RAM.

    Args:
        repo_id_or_path: HuggingFace repo ID or local path to GGUF file
        filename: Specific GGUF file to download (auto-selected if not provided)
        auto_select_quant: Enable RAM-aware quantization selection (recommended)
        max_ram_usage_gb: Override RAM detection (for auto_select_quant)
        threads: Number of threads (0 = auto)
        cache_dir: Cache directory for downloaded models
        token: HuggingFace API token for private repos
        revision: Git revision (branch, tag, commit)
        quant: Preferred quantization type (e.g., "Q4_K_M")
        trust_remote_code: Ignored (for HF compatibility)
        **kwargs: Additional arguments passed to DenseCore

    Returns:
        DenseCore: Initialized model instance

    Examples:
        # Load from HuggingFace Hub
        >>> model = densecore.from_pretrained("TheBloke/Llama-2-7B-GGUF")

        # With RAM-aware auto-selection
        >>> model = densecore.from_pretrained(
        ...     "TheBloke/Llama-2-7B-GGUF",
        ...     auto_select_quant=True  # Automatically picks best quantization
        ... )

        # Specify quantization
        >>> model = densecore.from_pretrained(
        ...     "TheBloke/Llama-2-7B-GGUF",
        ...     quant="Q4_K_M"
        ... )

        # LoRA adapter (auto-detected)
        >>> model = densecore.from_pretrained("user/my-lora-adapter")
        [LoRA] Detected LoRA adapter repository
        [LoRA] Loading with base model...

        # From local file
        >>> model = densecore.from_pretrained("./models/llama.gguf")
    """
    from .engine import DenseCore
    from .lora import download_lora_adapter, is_lora_adapter_repo

    # Detect model type for Vision/Whisper support
    model_type = None
    if not os.path.exists(repo_id_or_path):
        model_type = detect_model_type(repo_id_or_path, token=token)

        if model_type == "vision":
            print(f"[DenseCore] Detected Vision model: {repo_id_or_path}")
            print("[DenseCore] Vision inference uses DenseCore.encode_image() API")

        elif model_type == "whisper":
            print(f"[DenseCore] Detected Whisper model: {repo_id_or_path}")
            print("[DenseCore] Whisper inference uses DenseCore.transcribe() API")

        elif model_type == "multimodal":
            print(f"[DenseCore] Detected Multimodal model: {repo_id_or_path}")
            print("[DenseCore] Use DenseCore.generate() with image_path parameter")

    # Check if this is a LoRA adapter repository
    lora_adapter_path = None
    base_model_path = None

    if not os.path.exists(repo_id_or_path):
        # Repository mode - check if it's a LoRA adapter
        is_lora = is_lora_adapter_repo(repo_id_or_path, token=token)

        if is_lora:
            print(f"[LoRA] Detected LoRA adapter repository: {repo_id_or_path}")

            # Download the LoRA adapter
            lora_adapter_path = download_lora_adapter(
                repo_id=repo_id_or_path,
                filename=filename,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
            )

            # For now, user must provide base model separately
            # TODO: Auto-detect base model from adapter metadata
            print("[LoRA] ⚠️  Please provide base model via 'base_model_path' in kwargs")
            print("[LoRA] Example: model = densecore.from_pretrained(")
            print("[LoRA]              'adapter-repo',")
            print("[LoRA]              base_model_path='./base.gguf')")

            base_model_path = kwargs.pop("base_model_path", None)
            if not base_model_path:
                raise ValueError(
                    f"Repository {repo_id_or_path} contains a LoRA adapter. "
                    "Please provide base_model_path parameter."
                )

    # Handle both local paths and HuggingFace repos
    if os.path.exists(repo_id_or_path):
        target_model_path = repo_id_or_path
    elif base_model_path:
        # LoRA case: use provided base model
        target_model_path = base_model_path
    else:
        # Standard model download from HF Hub

        # Auto-select quantization based on RAM if requested
        if auto_select_quant and filename is None:
            from .smart_loader import recommend_quantization

            filename, message = recommend_quantization(
                repo_id=repo_id_or_path,
                available_ram_gb=max_ram_usage_gb,
                token=token,
                verbose=kwargs.get("verbose", True),
            )

            if filename is None:
                raise ValueError(message)

            if kwargs.get("verbose", True):
                print(f"[Smart Loader] {message}")

        # Download from HuggingFace Hub
        target_model_path = download_model(
            repo_id=repo_id_or_path,
            filename=filename,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            quant=quant,
        )

    # Initialize engine
    verbose = kwargs.pop("verbose", True)
    return DenseCore(
        model_path=target_model_path,
        lora_adapter_path=lora_adapter_path,
        threads=threads,
        hf_repo_id=repo_id_or_path if not os.path.exists(repo_id_or_path) else None,
        verbose=verbose,
        **kwargs,
    )


def detect_model_type(repo_id: str, token: Optional[str] = None) -> str:
    """
    Detect the model type from a HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token

    Returns:
        Model type: "llm", "vision", "whisper", "multimodal", or "unknown"

    Example:
        >>> detect_model_type("openai/whisper-large-v3")
        'whisper'
        >>> detect_model_type("TheBloke/Llama-2-7B-GGUF")
        'llm'
        >>> detect_model_type("openai/clip-vit-base-patch32")
        'vision'
    """
    repo_lower = repo_id.lower()

    # Check for each model type
    for model_type, keywords in MODEL_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in repo_lower:
                return model_type

    # Try to get more info from repo metadata
    try:
        if HfApi is None:
            return "unknown"
        api = HfApi(token=token)
        info = api.model_info(repo_id)

        # Check tags
        if hasattr(info, "tags") and info.tags:
            tags_lower = [t.lower() for t in info.tags]
            if any(
                t in tags_lower
                for t in ["whisper", "speech-recognition", "automatic-speech-recognition"]
            ):
                return "whisper"
            if any(t in tags_lower for t in ["vision", "image-classification", "image-to-text"]):
                return "vision"
            if any(t in tags_lower for t in ["text-generation", "conversational"]):
                return "llm"
            if any(t in tags_lower for t in ["multimodal", "visual-question-answering"]):
                return "multimodal"
    except Exception:
        pass

    return "unknown"


def get_model_info(repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a model repository.

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token

    Returns:
        Dictionary with model information
    """
    if HfApi is None:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

    api = HfApi(token=token)

    try:
        info = api.model_info(repo_id)
    except Exception as e:
        raise RuntimeError(f"Failed to get info for {repo_id}: {e}")

    gguf_files = list_gguf_files(repo_id, token=token)

    model_type = detect_model_type(repo_id, token=token)

    return {
        "id": info.id,
        "author": info.author,
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags,
        "model_type": model_type,
        "gguf_files": gguf_files,
        "total_size": sum(f["size"] for f in gguf_files),
    }
