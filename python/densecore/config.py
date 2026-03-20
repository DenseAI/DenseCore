"""
Configuration classes for DenseCore.

This module provides configuration dataclasses for model loading and text generation,
following HuggingFace's configuration patterns for familiarity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ModelConfig:
    """
    Configuration for model loading.

    Args:
        model_path: Path to the GGUF model file
        threads: Number of threads to use (0 = auto-detect)
        context_length: Maximum context length
        batch_size: Batch size for processing
        use_mmap: Use memory-mapped file for model loading
        use_mlock: Lock model in memory (prevent swapping)

    Example:
        >>> config = ModelConfig(
        ...     model_path="./model.gguf",
        ...     threads=8,
        ...     context_length=4096
        ... )
        >>> model = DenseCore(config=config)
    """

    model_path: str = ""
    threads: int = 0  # 0 = auto-detect
    context_length: int = 4096
    batch_size: int = 512
    use_mmap: bool = True
    use_mlock: bool = False

    # GPU offload (future)
    n_gpu_layers: int = 0

    # Quantization
    kv_cache_type: str = "f16"  # "f32", "f16", "q8_0"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.threads < 0:
            raise ValueError(f"threads must be >= 0, got {self.threads}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {self.context_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.kv_cache_type not in ("f32", "f16", "q8_0"):
            raise ValueError(f"Invalid kv_cache_type: {self.kv_cache_type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_path": self.model_path,
            "threads": self.threads,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "n_gpu_layers": self.n_gpu_layers,
            "kv_cache_type": self.kv_cache_type,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ModelConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> ModelConfig:
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Fully compatible with HuggingFace Transformers' GenerationConfig, enabling
    drop-in replacement for existing HF-based inference code.

    Args:
        max_tokens: Maximum number of tokens to generate (DenseCore native)
        max_new_tokens: Alias for max_tokens (HuggingFace compatibility)
        max_length: Maximum total length including prompt (HuggingFace compatibility)
        min_length: Minimum total length (HuggingFace compatibility)
        min_new_tokens: Minimum new tokens to generate (HuggingFace compatibility)
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        do_sample: Enable sampling (False = greedy decoding)
        num_beams: Number of beams for beam search (only 1 supported)
        repetition_penalty: Penalty for repeating tokens
        stop_sequences: List of sequences that stop generation

    Example:
        >>> # DenseCore style
        >>> config = GenerationConfig(max_tokens=256, temperature=0.7)

        >>> # HuggingFace style (fully compatible)
        >>> config = GenerationConfig(
        ...     max_new_tokens=256,
        ...     do_sample=True,
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1,
        ...     eos_token_id=2,
        ... )
    """

    # ==========================================================================
    # Length Parameters
    # ==========================================================================
    max_tokens: int = 256  # DenseCore native
    max_new_tokens: Optional[int] = None  # HF alias for max_tokens
    max_length: Optional[int] = None  # HF: max total length (prompt + generation)
    min_length: int = 0  # HF: minimum total length
    min_new_tokens: Optional[int] = None  # HF: minimum new tokens
    min_tokens: int = 0  # DenseCore native (alias for min_new_tokens)

    # ==========================================================================
    # Sampling Parameters
    # ==========================================================================
    do_sample: bool = False  # HF default: greedy decoding
    temperature: float = 0.0
    top_p: float = 1.0  # Nucleus sampling
    top_k: int = 0  # 0 = disabled
    typical_p: float = 1.0  # HF: typical decoding
    epsilon_cutoff: float = 0.0  # HF: eta sampling
    eta_cutoff: float = 0.0  # HF: eta sampling

    # ==========================================================================
    # Beam Search (Limited Support)
    # ==========================================================================
    num_beams: int = 1  # Only 1 supported (greedy/sampling)
    num_beam_groups: int = 1  # HF: diverse beam search
    diversity_penalty: float = 0.0  # HF: diverse beam search
    length_penalty: float = 1.0  # HF: beam search length penalty
    early_stopping: Union[bool, str] = False  # HF: beam search early stopping
    num_return_sequences: int = 1  # HF: number of sequences to return

    # ==========================================================================
    # Penalties
    # ==========================================================================
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0  # OpenAI-style
    presence_penalty: float = 0.0  # OpenAI-style
    no_repeat_ngram_size: int = 0  # HF: prevent n-gram repetition
    encoder_no_repeat_ngram_size: int = 0  # HF: encoder-decoder models

    # ==========================================================================
    # Token IDs
    # ==========================================================================
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: int | list[int] | None = None
    decoder_start_token_id: Optional[int] = None  # HF: encoder-decoder

    # ==========================================================================
    # Stopping Criteria
    # ==========================================================================
    stop_sequences: list[str] = field(default_factory=list)  # DenseCore native
    stop_token_ids: list[int] = field(default_factory=list)  # DenseCore native
    # Note: stopping_criteria is handled at runtime, not stored in config

    # ==========================================================================
    # Output Control
    # ==========================================================================
    output_scores: bool = False  # HF: return generation scores
    output_attentions: bool = False  # HF: return attention weights
    output_hidden_states: bool = False  # HF: return hidden states
    return_dict_in_generate: bool = False  # HF: return GenerateOutput

    # ==========================================================================
    # Logits Processing (Advanced)
    # ==========================================================================
    bad_words_ids: list[list[int]] | None = None  # HF: banned tokens
    force_words_ids: list[list[int]] | None = None  # HF: forced tokens
    forced_bos_token_id: int | None = None  # HF: forced BOS
    forced_eos_token_id: int | None = None  # HF: forced EOS
    suppress_tokens: list[int] | None = None  # HF: suppressed tokens

    # ==========================================================================
    # DenseCore Extensions
    # ==========================================================================
    stream: bool = False
    json_mode: bool = False  # Enable JSON output mode
    grammar: Optional[str] = None  # GBNF grammar constraint
    seed: Optional[int] = None  # Random seed for reproducibility

    def __post_init__(self) -> None:
        """Handle HuggingFace compatibility aliases and validate parameters."""
        # HF compatibility: max_new_tokens takes precedence
        if self.max_new_tokens is not None:
            self.max_tokens = self.max_new_tokens
        elif self.max_length is not None:
            # max_length includes prompt, but we don't know prompt length here
            # Use as max_tokens with a warning potential
            self.max_tokens = self.max_length

        # HF compatibility: min_new_tokens
        if self.min_new_tokens is not None:
            self.min_tokens = self.min_new_tokens
        elif self.min_length > 0:
            self.min_tokens = self.min_length

        # EOS token handling: convert single int to list for uniform handling
        if isinstance(self.eos_token_id, int):
            self.stop_token_ids = [self.eos_token_id] + self.stop_token_ids

        # Greedy decoding: disable sampling when do_sample=False
        if not self.do_sample:
            self.temperature = 0.0
            self.top_p = 1.0
            self.top_k = 0

        # Validation
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0 <= self.top_p <= 1):
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.num_beams != 1:
            import warnings

            warnings.warn(
                f"DenseCore only supports num_beams=1 (greedy/sampling). "
                f"Requested num_beams={self.num_beams} will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            self.num_beams = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "json_mode": self.json_mode,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> GenerationConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> GenerationConfig:
        """
        Load generation config from a pretrained model.

        This is a compatibility method for HuggingFace patterns.
        Returns default config for now.
        """
        # Try to load from local file
        config_path = os.path.join(model_name_or_path, "generation_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return cls.from_dict(json.load(f))

        # Return default config
        return cls()


@dataclass
class SamplingParams:
    """
    Sampling parameters for batch inference.

    This is an alias for GenerationConfig that follows vLLM naming conventions.
    """

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)

    def to_generation_config(self) -> GenerationConfig:
        """Convert to GenerationConfig."""
        return GenerationConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else 0,
            repetition_penalty=self.repetition_penalty,
            stop_sequences=self.stop,
        )
