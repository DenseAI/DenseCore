"""
HuggingFace-compatible generation output classes.

This module provides output dataclasses that mirror HuggingFace Transformers'
`ModelOutput` classes, enabling seamless integration with existing HF-based code.

Example:
    >>> output = model.generate("Hello", return_dict_in_generate=True)
    >>> print(output.sequences)  # Token IDs
    >>> print(output.text)       # Decoded text (DenseCore extension)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Try to import torch for tensor compatibility
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


@dataclass
class GenerateOutput:
    """
    HuggingFace-compatible output for text generation.

    This class mirrors `transformers.generation.utils.GenerateOutput` to provide
    a familiar interface for users migrating from HuggingFace Transformers.

    Attributes:
        sequences: Generated token IDs as a 2D list [batch_size, sequence_length]
        scores: Log probabilities for each generated token (optional)
        attentions: Attention weights (not supported, always None)
        hidden_states: Hidden states (not supported, always None)
        past_key_values: KV cache (not supported, always None)

    DenseCore Extensions:
        text: Decoded text string(s) - not in HF but useful
        finish_reason: Why generation stopped ("length", "stop", "eos")
        usage: Token usage statistics

    Example:
        >>> output = model.generate("Hello", return_dict_in_generate=True)
        >>> print(output.sequences)
        [[1, 2, 3, 4, 5]]
        >>> print(output.text)
        "Hello, how are you?"
    """

    # HuggingFace-compatible fields
    sequences: list[list[int]] = field(default_factory=list)
    scores: tuple[Any, ...] | None = None
    attentions: tuple[Any, ...] | None = None
    hidden_states: tuple[Any, ...] | None = None
    past_key_values: tuple[Any, ...] | None = None

    # DenseCore extensions
    text: str | list[str] = ""
    finish_reason: str | list[str] = "stop"
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize output format."""
        # Ensure sequences is 2D
        if self.sequences and isinstance(self.sequences[0], int):
            self.sequences = [self.sequences]  # type: ignore

    def __getitem__(self, key: str | int) -> Any:
        """
        Enable dict-like and tuple-like access for HuggingFace compatibility.

        HuggingFace ModelOutput supports both attribute and index access.
        """
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, int):
            # Tuple-like access for unpacking: sequences, scores, ...
            fields = ["sequences", "scores", "attentions", "hidden_states"]
            if 0 <= key < len(fields):
                return getattr(self, fields[key])
            raise IndexError(f"Index {key} out of range")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self):
        """Enable unpacking: sequences, scores = output."""
        yield self.sequences
        yield self.scores

    def to_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for unpacking."""
        return (self.sequences, self.scores, self.attentions, self.hidden_states)

    def keys(self) -> list[str]:
        """Return dict-like keys for HuggingFace compatibility."""
        return [
            "sequences",
            "scores",
            "attentions",
            "hidden_states",
            "text",
            "finish_reason",
            "usage",
        ]

    def values(self) -> list[Any]:
        """Return dict-like values."""
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        """Return dict-like items."""
        return [(k, getattr(self, k)) for k in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        return getattr(self, key, default)

    @property
    def last_hidden_state(self) -> None:
        """Alias for hidden_states[-1] (not supported)."""
        return None

    def to_tensor(self) -> Any:
        """
        Convert sequences to torch.Tensor if available.

        Returns:
            torch.Tensor if torch is available, else numpy array, else list

        Example:
            >>> tensor = output.to_tensor()
            >>> print(tensor.shape)
            torch.Size([1, 50])
        """
        if TORCH_AVAILABLE and torch is not None:
            return torch.tensor(self.sequences)
        elif NUMPY_AVAILABLE and np is not None:
            return np.array(self.sequences)
        return self.sequences


@dataclass
class GenerateEncoderDecoderOutput(GenerateOutput):
    """
    Output for encoder-decoder models (not used by DenseCore, for API compatibility).

    Provided for completeness with HuggingFace API.
    """

    encoder_attentions: tuple[Any, ...] | None = None
    encoder_hidden_states: tuple[Any, ...] | None = None
    decoder_attentions: tuple[Any, ...] | None = None
    decoder_hidden_states: tuple[Any, ...] | None = None
    cross_attentions: tuple[Any, ...] | None = None


@dataclass
class GenerateBeamOutput(GenerateOutput):
    """
    Output for beam search (not fully supported by DenseCore).

    Beam search is not implemented in DenseCore's CPU-optimized engine.
    This class is provided for API compatibility when num_beams > 1 is requested.
    """

    sequences_scores: list[float] | None = None
    beam_indices: list[list[int]] | None = None


@dataclass
class StoppingCriteriaOutput:
    """
    Result from stopping criteria evaluation.

    Compatible with HuggingFace's StoppingCriteria interface.
    """

    should_stop: bool = False
    reason: str = ""


class StoppingCriteria:
    """
    Base class for stopping criteria (HuggingFace-compatible interface).

    Subclass this to implement custom stopping logic.

    Example:
        >>> class MaxLengthCriteria(StoppingCriteria):
        ...     def __init__(self, max_length: int):
        ...         self.max_length = max_length
        ...
        ...     def __call__(self, input_ids, scores=None) -> bool:
        ...         return len(input_ids[0]) >= self.max_length
    """

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Evaluate stopping criteria.

        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Log probabilities (optional)
            **kwargs: Additional arguments

        Returns:
            True if generation should stop, False otherwise
        """
        raise NotImplementedError("Subclasses must implement __call__")


class StoppingCriteriaList(list):
    """
    List of stopping criteria (HuggingFace-compatible).

    Calling the list evaluates all criteria and returns True if any match.

    Example:
        >>> criteria = StoppingCriteriaList([
        ...     MaxLengthCriteria(100),
        ...     EosTokenCriteria(eos_token_id=2),
        ... ])
        >>> should_stop = criteria(input_ids, scores)
    """

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Evaluate all stopping criteria."""
        return any(criteria(input_ids, scores, **kwargs) for criteria in self)


class MaxLengthCriteria(StoppingCriteria):
    """
    Stop generation when maximum length is reached.

    HuggingFace-compatible implementation.
    """

    def __init__(self, max_length: int, max_position_embeddings: int | None = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        cur_len = len(input_ids[0]) if input_ids else 0
        return cur_len >= self.max_length


class MaxNewTokensCriteria(StoppingCriteria):
    """
    Stop generation when maximum new tokens are generated.

    HuggingFace-compatible implementation.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        cur_len = len(input_ids[0]) if input_ids else 0
        return cur_len >= self.max_length


class EosTokenCriteria(StoppingCriteria):
    """
    Stop generation when EOS token is generated.

    HuggingFace-compatible implementation.
    """

    def __init__(self, eos_token_id: int | list[int]):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = set(eos_token_id)

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        if not input_ids or not input_ids[0]:
            return False
        # Check if last token is EOS
        return input_ids[0][-1] in self.eos_token_id


class StopStringCriteria(StoppingCriteria):
    """
    Stop generation when a specific string is generated.

    DenseCore-specific extension for string-based stopping.
    """

    def __init__(self, stop_strings: list[str], tokenizer: Any = None):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(
        self,
        input_ids: list[list[int]],
        scores: list[float] | None = None,
        **kwargs: Any,
    ) -> bool:
        if not self.tokenizer or not input_ids:
            return False

        # Decode current sequence
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Check for stop strings
        return any(stop in text for stop in self.stop_strings)


__all__ = [
    "GenerateOutput",
    "GenerateEncoderDecoderOutput",
    "GenerateBeamOutput",
    "StoppingCriteriaOutput",
    "StoppingCriteria",
    "StoppingCriteriaList",
    "MaxLengthCriteria",
    "MaxNewTokensCriteria",
    "EosTokenCriteria",
    "StopStringCriteria",
]
