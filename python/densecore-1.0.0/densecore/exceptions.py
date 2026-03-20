"""
DenseCore exception hierarchy for structured error handling.

This module provides Python exceptions that map to C++ error codes,
enabling more specific error handling and better debugging.
"""


class DenseCoreError(Exception):
    """
    Base exception for all DenseCore errors.

    Attributes:
        message: Human-readable error message
        error_code: C error code (negative integer)
    """

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code})"


class ModelLoadError(DenseCoreError):
    """
    Failed to load or initialize model.

    Raised when:
    - Model file not found
    - Invalid GGUF format
    - Unsupported model architecture
    - Model initialization failed

    Example:
        >>> try:
        ...     model = DenseCore("nonexistent.gguf")
        ... except ModelLoadError as e:
        ...     print(f"Failed to load model: {e}")
    """

    pass


class InferenceError(DenseCoreError):
    """
    Error during inference/generation.

    Raised when:
    - Graph execution failed
    - Invalid tensor operations
    - Backend computation error

    Example:
        >>> try:
        ...     response = model.generate("Hello")
        ... except InferenceError as e:
        ...     print(f"Inference failed: {e}")
    """

    pass


class OutOfMemoryError(DenseCoreError):
    """
    KV cache or memory allocation failed.

    Raised when:
    - KV cache blocks exhausted
    - Cannot allocate new blocks
    - Context buffer too small

    Example:
        >>> try:
        ...     response = model.generate("Very long prompt...", max_tokens=4096)
        ... except OutOfMemoryError:
        ...     # Reduce max_tokens or use smaller model
        ...     response = model.generate("Shorter prompt", max_tokens=256)
    """

    pass


class TokenizationError(DenseCoreError):
    """
    Failed to tokenize input text.

    Raised when:
    - Invalid characters in prompt
    - Tokenizer not loaded
    - Encoding error

    Example:
        >>> try:
        ...     response = model.generate(invalid_text)
        ... except TokenizationError as e:
        ...     print(f"Cannot tokenize input: {e}")
    """

    pass


class ConfigurationError(DenseCoreError):
    """
    Invalid configuration parameters.

    Raised when:
    - Invalid parameter values (e.g., negative max_tokens)
    - Incompatible configuration options
    - Missing required parameters

    Example:
        >>> try:
        ...     response = model.generate("Hello", max_tokens=100000)
        ... except ConfigurationError as e:
        ...     print(f"Invalid config: {e}")
    """

    pass


class RequestCancelledError(DenseCoreError):
    """
    Request was cancelled before completion.

    Raised when:
    - User cancelled generation
    - Timeout exceeded
    - Engine shutdown during generation

    Example:
        >>> try:
        ...     response = model.generate("Hello")
        ... except RequestCancelledError:
        ...     print("Request was cancelled")
    """

    pass


class BackendError(DenseCoreError):
    """
    Error in underlying GGML backend.

    Raised when:
    - Backend initialization failed
    - Computation backend error
    - Thread pool error

    Example:
        >>> try:
        ...     model = DenseCore("model.gguf")
        ... except BackendError as e:
        ...     print(f"Backend error: {e}")
    """

    pass


# Error code mapping from C to Python exceptions
# Based on ErrorCode enum in utils/error.h
ERROR_CODE_MAP = {
    # Model errors (100-199)
    -100: ModelLoadError,  # MODEL_NOT_LOADED
    -101: ModelLoadError,  # MODEL_LOAD_FAILED
    -102: ModelLoadError,  # MODEL_INVALID_FORMAT
    -103: ModelLoadError,  # MODEL_UNSUPPORTED_ARCH
    -104: ConfigurationError,  # MODEL_INVALID_HYPERPARAMS
    # Memory errors (200-299)
    -200: OutOfMemoryError,  # OUT_OF_MEMORY
    -201: OutOfMemoryError,  # KV_CACHE_FULL
    -202: OutOfMemoryError,  # ALLOCATION_FAILED
    -203: OutOfMemoryError,  # CONTEXT_INIT_FAILED
    # Request errors (300-399)
    -300: ConfigurationError,  # INVALID_REQUEST
    -301: TokenizationError,  # TOKENIZATION_FAILED
    -302: InferenceError,  # INFERENCE_FAILED
    -303: RequestCancelledError,  # REQUEST_CANCELLED
    -304: ConfigurationError,  # INVALID_PARAMETERS
    # System errors (400-999)
    -400: BackendError,  # THREAD_ERROR
    -401: DenseCoreError,  # IO_ERROR
    -402: BackendError,  # BACKEND_ERROR
    -999: DenseCoreError,  # UNKNOWN_ERROR
}


def map_error_code(code: int, message: str) -> DenseCoreError:
    """
    Map C error code to appropriate Python exception.

    Args:
        code: Negative error code from C API
        message: Error message

    Returns:
        Instance of appropriate exception class

    Example:
        >>> exc = map_error_code(-200, "KV cache full")
        >>> isinstance(exc, OutOfMemoryError)
        True
    """
    exc_class = ERROR_CODE_MAP.get(code, DenseCoreError)
    return exc_class(message, error_code=code)


def raise_for_error_code(code: int, message: str) -> None:
    """
    Raise exception for C error code if non-zero.

    Args:
        code: Return code from C API (negative = error)
        message: Error message

    Raises:
        Appropriate DenseCoreError subclass if code < 0

    Example:
        >>> raise_for_error_code(-200, "OOM")  # Raises OutOfMemoryError
        >>> raise_for_error_code(0, "OK")  # Does nothing
    """
    if code < 0:
        raise map_error_code(code, message)
