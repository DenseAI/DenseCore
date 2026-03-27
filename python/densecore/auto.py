"""
HuggingFace Transformers-style Auto classes for DenseCore.

This module provides a familiar API for users coming from HuggingFace Transformers,
allowing them to load DenseCore models with the same syntax they're used to.

Example:
    >>> from densecore import AutoModel, AutoTokenizer

    # Load model from HuggingFace Hub
    >>> model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Generate text
    >>> output = model.generate("Hello, how are you?")
    >>> print(output)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .engine import DenseCore

from . import hub

try:
    import transformers as _transformers

    HFAutoTokenizer = _transformers.AutoTokenizer
except ImportError:
    HFAutoTokenizer = None


def from_pretrained(*args: Any, **kwargs: Any):
    """
    Backward-compatible forwarding helper.

    Kept at module level so tests and external code can patch
    `densecore.auto.from_pretrained`.
    """
    return hub.from_pretrained(*args, **kwargs)


class AutoModel:
    """
    HuggingFace-style AutoModel class for DenseCore.

    Provides a familiar interface for loading GGUF models from HuggingFace Hub
    or local paths. This is a convenience wrapper around `densecore.from_pretrained()`.

    Example:
        >>> from densecore import AutoModel

        # From HuggingFace Hub (auto-downloads GGUF)
        >>> model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

        # With specific quantization
        >>> model = AutoModel.from_pretrained(
        ...     "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        ...     filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ... )

        # From local path
        >>> model = AutoModel.from_pretrained("./models/model.gguf")

        # With RAM-aware auto-selection
        >>> model = AutoModel.from_pretrained(
        ...     "TheBloke/Llama-2-7B-GGUF",
        ...     auto_select_quant=True
        ... )
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoModel is designed to be instantiated using the "
            "`AutoModel.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        filename: Optional[str] = None,
        auto_select_quant: bool = False,
        threads: int = 0,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        revision: str = "main",
        quant: Optional[str] = None,
        trust_remote_code: bool = False,
        # HuggingFace Transformers compatibility kwargs (ignored or mapped)
        device_map: Optional[Any] = None,
        torch_dtype: Optional[Any] = None,
        attn_implementation: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention_2: bool = False,
        **kwargs,
    ) -> DenseCore:
        """
        Load a DenseCore model from HuggingFace Hub or local path.

        This method provides API compatibility with HuggingFace Transformers,
        making it easy for developers to switch to DenseCore.

        Args:
            model_name_or_path: HuggingFace repo ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
                               or path to local GGUF file
            filename: Specific GGUF file to download (auto-selected if not provided)
            auto_select_quant: Enable RAM-aware quantization selection
            threads: Number of CPU threads (0 = auto)
            cache_dir: Cache directory for downloaded models
            token: HuggingFace API token for private repos
            revision: Git revision (branch, tag, commit)
            quant: Preferred quantization type (e.g., "Q4_K_M")
            trust_remote_code: Ignored (for HuggingFace compatibility)

            # HuggingFace Transformers compatibility (mapped or ignored):
            device_map: Ignored (DenseCore is CPU-only)
            torch_dtype: Ignored (uses GGUF quantization)
            attn_implementation: Ignored (uses optimized CPU attention)
            low_cpu_mem_usage: Ignored (DenseCore is memory-optimized by default)
            load_in_8bit: Maps to Q8_0 quantization if True
            load_in_4bit: Maps to Q4_K_M quantization if True
            use_flash_attention_2: Ignored (uses CPU-optimized attention)
            **kwargs: Additional arguments passed to DenseCore

        Returns:
            DenseCore: Initialized model instance ready for inference

        Example:
            >>> # Drop-in replacement for HuggingFace Transformers
            >>> model = AutoModel.from_pretrained(
            ...     "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            ...     device_map="auto",  # Ignored (CPU-only)
            ...     torch_dtype="auto",  # Ignored (uses GGUF quantization)
            ... )
            >>> response = model.generate("Hello!")
            >>> print(response)
        """
        # Handle HuggingFace-style quantization flags
        if load_in_4bit and quant is None:
            quant = "Q4_K_M"
        elif load_in_8bit and quant is None:
            quant = "Q8_0"

        # Warn about ignored HuggingFace kwargs
        _ignored_kwargs = []
        if device_map is not None:
            _ignored_kwargs.append(f"device_map={device_map!r}")
        if torch_dtype is not None:
            _ignored_kwargs.append(f"torch_dtype={torch_dtype!r}")
        if attn_implementation is not None:
            _ignored_kwargs.append(f"attn_implementation={attn_implementation!r}")
        if use_flash_attention_2:
            _ignored_kwargs.append("use_flash_attention_2=True")
        if low_cpu_mem_usage:
            _ignored_kwargs.append("low_cpu_mem_usage=True")

        if _ignored_kwargs:
            warnings.warn(
                f"DenseCore is CPU-only and uses GGUF quantization. "
                f"The following kwargs are ignored: {', '.join(_ignored_kwargs)}",
                UserWarning,
                stacklevel=2,
            )

        return from_pretrained(
            repo_id_or_path=model_name_or_path,
            filename=filename,
            auto_select_quant=auto_select_quant,
            threads=threads,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            quant=quant,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


class AutoModelForCausalLM(AutoModel):
    """
    HuggingFace-style AutoModelForCausalLM for DenseCore.

    This is an alias for AutoModel, as all DenseCore models are causal language models.
    Provided for API compatibility with HuggingFace Transformers.

    Example:
        >>> from densecore import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    """

    pass


class AutoTokenizer:
    """
    HuggingFace-style AutoTokenizer wrapper for DenseCore.

    This class wraps `transformers.AutoTokenizer` to provide a consistent
    experience within the DenseCore ecosystem. It's useful when you need
    to pre-process inputs or post-process outputs manually.

    Note:
        DenseCore models typically handle tokenization internally via the
        C++ engine. This class is provided for advanced use cases where
        you need direct access to the tokenizer.

    Example:
        >>> from densecore import AutoTokenizer

        # Load tokenizer (requires transformers library)
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        # Tokenize text
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(tokens)

        # Decode tokens
        >>> text = tokenizer.decode(tokens)
        >>> print(text)
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoTokenizer is designed to be instantiated using the "
            "`AutoTokenizer.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        revision: str = "main",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Load a tokenizer from HuggingFace Hub.

        This method delegates to `transformers.AutoTokenizer.from_pretrained()`,
        providing a seamless experience for users familiar with HuggingFace.

        Args:
            model_name_or_path: HuggingFace model ID or path to tokenizer files
            cache_dir: Cache directory for downloaded files
            token: HuggingFace API token for private repos
            revision: Git revision (branch, tag, commit)
            trust_remote_code: Whether to trust remote code (passed to transformers)
            **kwargs: Additional arguments passed to AutoTokenizer

        Returns:
            PreTrainedTokenizer: HuggingFace tokenizer instance

        Raises:
            ImportError: If transformers library is not installed

        Example:
            >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            >>> tokens = tokenizer("Hello!", return_tensors="pt")
        """
        auto_tokenizer_cls = HFAutoTokenizer
        if auto_tokenizer_cls is None:
            raise ImportError(
                "Transformers is not installed. "
                "Please install it via `pip install densecore[hf]` (or `densecore[full]`) "
                "to use AutoTokenizer, "
                "or use the native `densecore.LlamaTokenizer` if available."
            ) from None

        return auto_tokenizer_cls.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


__all__ = [
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoTokenizer",
]
