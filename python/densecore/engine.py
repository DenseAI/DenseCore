"""
DenseCore Engine - Main inference interface.

This module provides the core DenseCore class that wraps the C++ inference engine
with a Pythonic API supporting both synchronous and asynchronous generation.
"""

import asyncio
import ctypes
import json
import os
import platform
import queue
import re
import sys
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None
    list_repo_files = None
    HUGGINGFACE_HUB_AVAILABLE = False

try:
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

import numpy as np

from .chat_template import format_chat_prompt
from .config import GenerationConfig, ModelConfig
from .lora import LoRAManager

# ==============================================================================
# GIL Release Architecture
# ==============================================================================
#
# DenseCore achieves true non-blocking AsyncIO through the following design:
#
# 1. THREAD POOL OFFLOADING:
#    - SubmitRequest is offloaded to a ThreadPoolExecutor via run_in_executor
#    - This frees the Python event loop to handle other async tasks
#
# 2. GIL RELEASE DURING C++ EXECUTION:
#    - ctypes.CDLL (not ctypes.pythonapi) is used for bindings
#    - The GIL is released during C function calls by default
#    - C++ inference runs on native threads without holding the GIL
#
# 3. THREAD-SAFE CALLBACK BRIDGING:
#    - Token callbacks use loop.call_soon_threadsafe() to safely
#      communicate from C++ worker threads back to the event loop
#
# This enables 100+ concurrent async requests without event loop starvation.
# ==============================================================================

# Dedicated thread pool for C++ inference submissions
# Lazily initialized to avoid overhead if only sync methods are used
_INFERENCE_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_inference_executor() -> ThreadPoolExecutor:
    """
    Get or create the inference thread pool executor (Singleton).

    The executor is sized to handle high concurrency while avoiding
    thread explosion. Workers are named for easy debugging.
    """
    global _INFERENCE_EXECUTOR
    if _INFERENCE_EXECUTOR is None:
        max_workers = min(32, (os.cpu_count() or 4) * 4)
        _INFERENCE_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="densecore-submit",
        )
    return _INFERENCE_EXECUTOR


def shutdown_executor() -> None:
    """
    Shut down the inference executor gracefully.

    Call this during application shutdown to ensure clean termination.
    """
    global _INFERENCE_EXECUTOR
    if _INFERENCE_EXECUTOR is not None:
        _INFERENCE_EXECUTOR.shutdown(wait=True)
        _INFERENCE_EXECUTOR = None


# ==============================================================================
# Custom Exception Hierarchy for ctypes Error Handling
# ==============================================================================


class DenseCoreError(Exception):
    """Base exception for all DenseCore errors."""

    pass


class DenseCoreRuntimeError(DenseCoreError):
    """Runtime error during inference (e.g., model loading failure)."""

    pass


class ContextLimitExceededError(DenseCoreError):
    """Prompt exceeds the model's context length."""

    pass


class OOMError(DenseCoreError):
    """Out of memory error."""

    pass


class InvalidRequestError(DenseCoreError):
    """Invalid request parameters."""

    pass


class EngineNotInitializedError(DenseCoreError):
    """Engine handle is null or not initialized."""

    pass


# Error code to exception mapping
_ERROR_CODE_MAP: Dict[int, Type[DenseCoreError]] = {
    -1: DenseCoreRuntimeError,  # Generic error
    -2: InvalidRequestError,  # Invalid parameters
    -3: ContextLimitExceededError,  # Context overflow
    -4: OOMError,  # Out of memory
    -5: EngineNotInitializedError,  # Null handle
}


# DenseCoreKVType mapping from core/include/densecore.h
kv_type_map: Dict[str, int] = {
    "fp16": 0,
    "f16": 0,
    "fp32": 1,
    "f32": 1,
    "int8": 2,
    "q8_0": 2,
}


def _error_code_to_exception(code: int, context: str = "") -> DenseCoreError:
    """Convert a C++ error code to a Python exception."""
    exc_class = _ERROR_CODE_MAP.get(code, DenseCoreRuntimeError)
    msg = f"DenseCore error (code={code})"
    if context:
        msg = f"{msg}: {context}"
    return exc_class(msg)


def _errcheck(result: int, func: Any, args: tuple) -> int:
    """
    ctypes errcheck callback for validating C function return values.

    Raises appropriate Python exceptions for negative error codes.
    """
    if result < 0:
        func_name = getattr(func, "__name__", "unknown")
        raise _error_code_to_exception(result, context=f"in {func_name}()")
    return result


# Type variable for generic typing
T = TypeVar("T")

# C callback function type (legacy string-based)
# typedef void (*TokenCallback)(const char *token, int is_finished, void *user_data);
CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)


# TokenResult struct for structured token callbacks
class TokenResult(ctypes.Structure):
    """
    Structured token result from C++ engine.

    Attributes:
        token_id: Token ID for HuggingFace tokenizer decoding
        text: Pre-decoded text (may be empty if using HF tokenizer)
        is_finished: 1 if generation is complete, 0 otherwise
    """

    _fields_ = [
        ("token_id", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("is_finished", ctypes.c_int),
    ]


# Callback type for TokenResult struct
TOKEN_RESULT_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(TokenResult), ctypes.c_void_p)


class DenseCoreMetrics(ctypes.Structure):
    _fields_ = [
        ("requests_per_second", ctypes.c_float),
        ("tokens_per_second", ctypes.c_float),
        ("active_requests", ctypes.c_int),
        ("total_tokens_generated", ctypes.c_long),
    ]


class DenseCoreTensor(ctypes.Structure):
    """
    C-compatible tensor descriptor for manual weight loading.

    Layout matches DenseCoreTensor in densecore.h:
    typedef struct {
        void* data;
        int64_t shape[4];
        int ndim;
        int dtype;
    } DenseCoreTensor;
    """

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_int64 * 4),
        ("ndim", ctypes.c_int),
        ("dtype", ctypes.c_int),
    ]

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "DenseCoreTensor":
        """Create a DenseCoreTensor from a numpy array."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(arr)}")

        # Ensure contiguous
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        shape = (ctypes.c_int64 * 4)(*[0] * 4)
        for i, s in enumerate(arr.shape[:4]):
            shape[i] = s

        ndim = len(arr.shape)

        # dtype mapping: 0=F32, 1=F16, 2=BF16, 3=INT8
        dtype_map = {
            np.dtype("float32"): 0,
            np.dtype("float16"): 1,
            np.dtype("int8"): 3,
            # BF16 not standard in numpy, usually handled as generic bytes or via torch
        }

        dtype_val = dtype_map.get(arr.dtype)
        if dtype_val is None:
            # Fallback for BF16 if user passes special type, or error
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        tensor = cls()
        tensor.data = arr.ctypes.data
        tensor.shape = shape
        tensor.ndim = ndim
        tensor.dtype = dtype_val

        # Keep reference to array to prevent GC
        tensor._arr = arr
        return tensor


DENSECORE_MAX_DIMS = 4


class DenseCoreTensorInput(ctypes.Structure):
    """
    C-compatible tensor input descriptor for graph execution.

    Matches DenseCoreTensorInput in densecore.h
    """

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * DENSECORE_MAX_DIMS),
        ("dtype", ctypes.c_int),
    ]

    @classmethod
    def from_numpy(cls, arr: np.ndarray, name: str = "input") -> "DenseCoreTensorInput":
        """Create a DenseCoreTensorInput from a numpy array."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(arr)}")

        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        shape = (ctypes.c_int64 * DENSECORE_MAX_DIMS)(*[0] * DENSECORE_MAX_DIMS)
        for i, s in enumerate(arr.shape[:DENSECORE_MAX_DIMS]):
            shape[i] = s

        ndim = len(arr.shape)

        dtype_map = {
            np.dtype("float32"): 0,
            np.dtype("float16"): 1,
            np.dtype("int8"): 3,
        }

        dtype_val = dtype_map.get(arr.dtype)
        if dtype_val is None:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        tensor_input = cls()
        tensor_input.name = name.encode("utf-8")
        tensor_input.data = arr.ctypes.data
        tensor_input.ndim = ndim
        tensor_input.shape = shape
        tensor_input.dtype = dtype_val

        tensor_input._arr = arr
        tensor_input._name_bytes = name.encode("utf-8")
        return tensor_input


class DenseCoreTensorOutput(ctypes.Structure):
    """
    C-compatible tensor output descriptor for graph execution results.

    Matches DenseCoreTensorOutput in densecore.h
    """

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * DENSECORE_MAX_DIMS),
        ("dtype", ctypes.c_int),
    ]

    def to_numpy(self) -> np.ndarray:
        """Convert output tensor to numpy array."""
        dtype_map = {
            0: np.float32,
            1: np.float16,
            3: np.int8,
        }

        np_dtype = dtype_map.get(self.dtype, np.float32)
        shape = tuple(self.shape[i] for i in range(self.ndim))

        size = 1
        for s in shape:
            size *= s

        arr = np.ctypeslib.as_array(
            ctypes.cast(self.data, ctypes.POINTER(ctypes.c_float)), shape=(size,)
        )
        return arr.reshape(shape).astype(np_dtype)


@dataclass
class GenerationOutput:
    """
    Output from text generation.

    Attributes:
        text: Generated text
        tokens: Number of tokens generated
        finish_reason: Why generation stopped ("length", "stop", "error")
        prompt_tokens: Number of prompt tokens
        generation_time: Time taken for generation in seconds
    """

    text: str
    tokens: int = 0
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    generation_time: float = 0.0

    def __str__(self) -> str:
        return self.text


def _find_library() -> str:
    """
    Find the DenseCore shared library.

    Searches in:
    1. Package directory (including pip-installed cpython-suffixed names)
    2. Parent directories (for development)
    3. System library paths
    4. LD_LIBRARY_PATH / DYLD_LIBRARY_PATH

    Returns:
        Path to the library file

    Raises:
        RuntimeError: If library cannot be found
    """
    if platform.system() == "Darwin":
        lib_names = ["libdensecore.dylib"]
        glob_pattern = "libdensecore*.dylib"
    else:
        lib_names = ["libdensecore.so", "libdensecore.so.1"]
        glob_pattern = "libdensecore*.so"

    repo_root = Path(__file__).parent.parent.parent
    search_paths = [
        repo_root / "build",
        repo_root / "core" / "build",
        Path(__file__).parent,
        Path(__file__).parent / "lib",
        Path(__file__).parent.parent,
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]

    # Add environment library paths
    env_var = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
    if env_var in os.environ:
        for path_str in os.environ[env_var].split(os.pathsep):
            search_paths.append(Path(path_str))

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

    # Fallback: glob for pip-installed wheel names (e.g. libdensecore.cpython-311-x86_64-linux-gnu.so)
    candidates = sorted(Path(__file__).parent.glob(glob_pattern))
    if candidates:
        return str(candidates[0])

    raise RuntimeError(
        f"Could not find DenseCore library. Searched in: {[str(p) for p in search_paths]}"
    )


_LIB_INSTANCE = None


def _preload_linux_runtime_dependencies(lib_path: str) -> None:
    """
    Preload runtime deps that are known to fail when libdensecore is dlopened late.

    `libggml-cpu.so` links against `libgomp.so.1`, which uses static TLS on some
    Linux distributions. Loading `libgomp` globally first avoids:
    "cannot allocate memory in static TLS block".
    """
    if platform.system() != "Linux":
        return

    lib_dir = Path(lib_path).resolve().parent
    candidates = [
        str(lib_dir / "libgomp.so.1"),
        "libgomp.so.1",
        str(lib_dir / "libggml-base.so.0"),
        str(lib_dir / "libggml-cpu.so.0"),
        str(Path(lib_path).resolve()),
    ]

    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def _reexec_with_linux_preload(lib_path: str) -> None:
    """
    Re-exec the current Python process with LD_PRELOAD when static TLS prevents
    late dlopen() of libdensecore.
    """
    if platform.system() != "Linux":
        return
    if os.environ.get("DENSECORE_TLS_REEXEC_DONE") == "1":
        return
    if not sys.argv or sys.argv[0] in {"", "-", "-c"}:
        return

    lib_dir = Path(lib_path).resolve().parent
    preload_libs = [
        str(lib_dir / "libgomp.so.1"),
        str(lib_dir / "libggml-base.so.0"),
        str(lib_dir / "libggml-cpu.so.0"),
        str(Path(lib_path).resolve()),
    ]
    existing = [entry for entry in os.environ.get("LD_PRELOAD", "").split(os.pathsep) if entry]
    merged: list[str] = []
    for entry in preload_libs + existing:
        if entry and entry not in merged:
            merged.append(entry)

    env = os.environ.copy()
    env["LD_PRELOAD"] = os.pathsep.join(merged)
    env["DENSECORE_TLS_REEXEC_DONE"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def get_library() -> ctypes.CDLL:
    """
    Get the shared library instance (Singleton).
    """
    global _LIB_INSTANCE
    if _LIB_INSTANCE is None:
        lib_path = _find_library()
        _preload_linux_runtime_dependencies(lib_path)
        try:
            _LIB_INSTANCE = ctypes.CDLL(lib_path)
        except OSError as exc:
            if "static TLS block" in str(exc):
                _reexec_with_linux_preload(lib_path)
            raise
    return _LIB_INSTANCE


class DenseCore:
    """
    High-performance CPU inference engine for large language models.

    DenseCore provides a simple interface for running inference on GGUF models
    with support for streaming and async operations.

    Args:
        model_path: Path to the GGUF model file (required)
        threads: Number of CPU threads (0 = auto-detect)
        hf_repo_id: HuggingFace repo ID for loading tokenizer
        config: Optional ModelConfig for advanced settings

    Examples:
        Basic usage:

        >>> from densecore import DenseCore
        >>> model = DenseCore("./model.gguf")
        >>> response = model.generate("Hello, how are you?")
        >>> print(response)

        With context manager:

        >>> with DenseCore("./model.gguf") as model:
        ...     print(model.generate("Hello!"))

        Streaming:

        >>> for token in model.stream("Tell me a story"):
        ...     print(token, end="", flush=True)

        Async streaming:

        >>> async for token in model.stream_async("Hello"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        lora_scale: float = 1.0,
        threads: int = 0,
        hf_repo_id: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        verbose: bool = True,
        kv_cache_dtype: str = "fp16",
        backend: str = "cpu",
        custom_backend_path: Optional[str] = None,
    ):
        self._handle = None
        self._closed = True
        self._requests: Dict[int, Any] = {}
        self._active_ctypes_refs: Dict[int, List[Any]] = {}  # Prevent GC of ctypes objects
        self._req_id_counter = 0
        self._lock = threading.Lock()
        self._verbose = verbose
        self._model_path = ""
        self._hf_repo_id = hf_repo_id or ""
        self._kv_cache_dtype = kv_cache_dtype
        self.tokenizer = None

        # LoRA adapter management
        self._lora_manager = LoRAManager()

        # Handle config
        if config is not None:
            model_path = config.model_path or model_path
            threads = config.threads if config.threads > 0 else threads

        if not model_path:
            raise ValueError("model_path is required")

        self._model_path = model_path

        # Load shared library
        self._lib = get_library()

        # Setup C function signatures
        self._setup_cfunctions()

        # Keep callback reference to prevent GC
        self._c_callback = CALLBACK_TYPE(self._global_callback)

        # Initialize engine
        model_path_bytes = model_path.encode("utf-8")

        if hf_repo_id:
            self._init_tokenizer(hf_repo_id)

        # Map kv_cache_dtype to C API enum value
        if kv_cache_dtype.lower() not in kv_type_map:
            raise ValueError(
                f"kv_cache_dtype must be one of {list(kv_type_map.keys())}, got: {kv_cache_dtype}"
            )
        kv_type_value = kv_type_map[kv_cache_dtype.lower()]

        # HAL: Load custom backend plugin if provided
        if custom_backend_path:
            if getattr(self, "_has_hal_api", False):
                if self._verbose:
                    print(f"[DenseCore] Loading backend plugin from: {custom_backend_path}")
                ret = self._lib.DenseCoreLoadPlugin(custom_backend_path.encode("utf-8"))
                if ret < 0:
                    raise RuntimeError(f"Failed to load plugin {custom_backend_path}, error {ret}")
            else:
                warnings.warn(
                    "Custom backend path provided but DenseCoreLoadPlugin not available in C lib"
                )

        # HAL: Create/select device (backend)
        # Note: InitEngine currently initializes the default backend (CPU/Metal) implicitly.
        # We call CreateDevice to ensure the requested backend is available and ready.
        if getattr(self, "_has_hal_api", False):
            if self._verbose:
                print(f"[DenseCore] Selecting backend: {backend}")
            device_handle = self._lib.DenseCoreCreateDevice(backend.upper().encode("utf-8"), 0)
            if not device_handle:
                warnings.warn(
                    f"Failed to create device for backend '{backend}'. Types: CPU, METAL, NPU, ASIC"
                )
            else:
                if self._verbose:
                    print(f"[DenseCore] Backend '{backend}' initialized successfully")

        # Initialize native engine with KV cache type
        if self._verbose:
            print(
                f"[DenseCore] Initializing with model: {model_path}, kv_cache_dtype: {kv_cache_dtype}"
            )

        # Use extended API if kv_cache_dtype is specified (non-default)
        if kv_cache_dtype.lower() != "fp16" and hasattr(self._lib, "InitEngineWithKVType"):
            self._handle = self._lib.InitEngineWithKVType(
                model_path_bytes, None, threads, -1, 0, kv_type_value
            )
        else:
            self._handle = self._lib.InitEngine(model_path_bytes, None, threads)

        if self._verbose:
            print(f"[DenseCore] Engine initialized, handle: {self._handle}")
        if not self._handle:
            raise RuntimeError(f"Failed to initialize DenseCore engine for model: {model_path}")

        # Load LoRA adapter if provided
        if lora_adapter_path:
            self.load_lora(lora_adapter_path, scale=lora_scale)

        self._closed = False  # Mark as open so close() works correctly

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DenseCore":
        """
        Load a model directly from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: GGUF filename to download (e.g., "llama-2-7b.Q4_K_M.gguf").
                      If None, attempts to find a suitable GGUF file automatically.
            revision: Specific model revision (branch/tag/commit)
            cache_dir: Directory to cache downloaded files
            **kwargs: Additional arguments passed to DenseCore constructor

        Returns:
            Initialized DenseCore instance

        Raises:
            ImportError: If huggingface_hub is not installed
            ValueError: If finding a GGUF file fails
        """
        if not HUGGINGFACE_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Please install it with: pip install huggingface-hub"
            )

        if filename is None:
            # Try to auto-detect GGUF file
            print(f"[DenseCore] No filename provided, searching in {repo_id}...")
            files = list_repo_files(repo_id=repo_id, revision=revision)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                raise ValueError(f"No .gguf files found in {repo_id}")

            # Prefer Q4_K_M if available, otherwise pick the first one
            preferred = [f for f in gguf_files if "Q4_K_M" in f]
            if preferred:
                filename = preferred[0]
            else:
                filename = gguf_files[0]

            print(f"[DenseCore] Auto-selected model file: {filename}")

        print(f"[DenseCore] Downloading {filename} from {repo_id}...")
        model_path = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
        )

        # Pass hf_repo_id to constructor to auto-load tokenizer
        if "hf_repo_id" not in kwargs:
            kwargs["hf_repo_id"] = repo_id

        return cls(model_path=model_path, **kwargs)

    def _setup_cfunctions(self) -> None:
        """Setup C function argument and return types with error checking."""
        # InitEngine
        self._lib.InitEngine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.InitEngine.restype = ctypes.c_void_p

        # InitEngineWithKVType (extended API with KV cache type)
        try:
            self._lib.InitEngineWithKVType.argtypes = [
                ctypes.c_char_p,  # model_path
                ctypes.c_char_p,  # reserved
                ctypes.c_int,  # threads
                ctypes.c_int,  # numa_node_id
                ctypes.c_int,  # pinning_policy
                ctypes.c_int,  # kv_cache_type (DenseCoreKVType enum)
            ]
            self._lib.InitEngineWithKVType.restype = ctypes.c_void_p
        except AttributeError:
            pass  # Function may not exist in older library versions

        # SubmitRequest - returns request ID or negative error code
        self._lib.SubmitRequest.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.SubmitRequest.restype = ctypes.c_int
        self._lib.SubmitRequest.errcheck = _errcheck

        # SubmitRequestIds (for tokenized input) - returns request ID or negative error
        self._lib.SubmitRequestIds.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.SubmitRequestIds.restype = ctypes.c_int
        self._lib.SubmitRequestIds.errcheck = _errcheck

        # SubmitRequestIdsWithFormat (new JSON support)
        try:
            self._lib.SubmitRequestIdsWithFormat.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestIdsWithFormat.restype = ctypes.c_int
            self._lib.SubmitRequestIdsWithFormat.errcheck = _errcheck
            self._has_json_app = True
        except AttributeError:
            self._has_json_app = False

        # SubmitRequestWithSampling (Full Sampling Support)
        try:
            self._lib.SubmitRequestWithSampling.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_float,  # temperature
                ctypes.c_float,  # top_p
                ctypes.c_int,  # top_k
                ctypes.c_float,  # repetition_penalty
                ctypes.POINTER(ctypes.c_char_p),  # stop_sequences
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestWithSampling.restype = ctypes.c_int
            self._lib.SubmitRequestWithSampling.errcheck = _errcheck

            self._lib.SubmitRequestIdsWithSampling.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,  # temperature
                ctypes.c_float,  # top_p
                ctypes.c_int,  # top_k
                ctypes.c_float,  # repetition_penalty
                ctypes.POINTER(ctypes.c_char_p),  # stop_sequences
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestIdsWithSampling.restype = ctypes.c_int
            self._lib.SubmitRequestIdsWithSampling.errcheck = _errcheck
            self._has_sampling_api = True
        except AttributeError:
            self._has_sampling_api = False

        # SubmitRequestWithSamplingEx (per-request LoRA)
        try:
            self._lib.SubmitRequestWithSamplingEx.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,  # lora_name
                ctypes.c_float,  # temperature
                ctypes.c_float,  # top_p
                ctypes.c_int,  # top_k
                ctypes.c_float,  # repetition_penalty
                ctypes.POINTER(ctypes.c_char_p),  # stop_sequences
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestWithSamplingEx.restype = ctypes.c_int
            self._lib.SubmitRequestWithSamplingEx.errcheck = _errcheck

            self._lib.SubmitRequestIdsWithSamplingEx.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,  # lora_name
                ctypes.c_float,  # temperature
                ctypes.c_float,  # top_p
                ctypes.c_int,  # top_k
                ctypes.c_float,  # repetition_penalty
                ctypes.POINTER(ctypes.c_char_p),  # stop_sequences
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestIdsWithSamplingEx.restype = ctypes.c_int
            self._lib.SubmitRequestIdsWithSamplingEx.errcheck = _errcheck
            self._has_lora_sampling_api = True
        except AttributeError:
            self._has_lora_sampling_api = False

        # SubmitRequestWithFormatEx (per-request LoRA)
        try:
            self._lib.SubmitRequestIdsWithFormatEx.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,  # lora_name
                ctypes.c_int,  # json_mode
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.SubmitRequestIdsWithFormatEx.restype = ctypes.c_int
            self._lib.SubmitRequestIdsWithFormatEx.errcheck = _errcheck
            self._has_lora_format_api = True
        except AttributeError:
            self._has_lora_format_api = False

        # FreeEngine - void return, no errcheck needed
        self._lib.FreeEngine.argtypes = [ctypes.c_void_p]
        self._lib.FreeEngine.restype = None

        # GetMetrics (optional)
        try:
            self._lib.GetMetrics.argtypes = [ctypes.c_void_p]
            self._lib.GetMetrics.restype = DenseCoreMetrics
            self._has_metrics = True
        except AttributeError:
            self._has_metrics = False

        # LoRA Runtime API
        try:
            self._lib.LoadLoraAdapter.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_float,
                ctypes.c_char_p,
            ]
            self._lib.LoadLoraAdapter.restype = ctypes.c_int
            self._lib.LoadLoraAdapter.errcheck = _errcheck

            self._lib.ActivateLoraAdapter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._lib.ActivateLoraAdapter.restype = ctypes.c_int
            self._lib.ActivateLoraAdapter.errcheck = _errcheck

            self._lib.DeactivateLoraAdapters.argtypes = [ctypes.c_void_p]
            self._lib.DeactivateLoraAdapters.restype = ctypes.c_int
            self._lib.DeactivateLoraAdapters.errcheck = _errcheck

            self._lib.UnloadLoraAdapter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._lib.UnloadLoraAdapter.restype = ctypes.c_int
            self._lib.UnloadLoraAdapter.errcheck = _errcheck

            self._has_lora_api = True
        except AttributeError:
            self._has_lora_api = False

        # CancelRequest API for graceful request cancellation
        try:
            self._lib.CancelRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._lib.CancelRequest.restype = ctypes.c_int
            self._has_cancel_api = True
            self._has_cancel_api = True
        except AttributeError:
            self._has_cancel_api = False

        # HAL & Weight Loading API
        try:
            self._lib.DenseCoreLoadPlugin.argtypes = [ctypes.c_char_p]
            self._lib.DenseCoreLoadPlugin.restype = ctypes.c_int

            self._lib.DenseCoreCreateDevice.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self._lib.DenseCoreCreateDevice.restype = ctypes.c_void_p

            self._lib.DenseCoreSetWeight.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(DenseCoreTensor),
            ]
            self._lib.DenseCoreSetWeight.restype = ctypes.c_int
            self._lib.DenseCoreSetWeight.errcheck = _errcheck

            self._has_hal_api = True
        except AttributeError:
            self._has_hal_api = False

        # Universal Graph Execution API
        try:
            self._lib.ExecuteGraphSync.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.POINTER(DenseCoreTensorInput),  # inputs
                ctypes.c_int,  # num_inputs
                ctypes.c_char_p,  # graph_name
                ctypes.POINTER(DenseCoreTensorOutput),  # outputs
                ctypes.c_int,  # num_outputs
            ]
            self._lib.ExecuteGraphSync.restype = ctypes.c_int
            self._lib.ExecuteGraphSync.errcheck = _errcheck
            self._has_graph_api = True
        except AttributeError:
            self._has_graph_api = False

    def load_weights(self, model_path: str) -> None:
        """
        Load weights from a safetensors file into the engine.

        Args:
            model_path: Path to .safetensors file
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not installed. Please install with: pip install safetensors"
            )

        if not getattr(self, "_has_hal_api", False):
            raise NotImplementedError(
                "C library does not support manual weight loading (DenseCoreSetWeight missing)"
            )

        if self._verbose:
            print(f"[DenseCore] Loading weights from {model_path}...")

        count = 0
        with safe_open(model_path, framework="numpy", device="cpu") as f:
            for key in f.keys():
                # Get tensor as numpy array
                arr = f.get_tensor(key)

                # Create C struct
                # Note: We must hold a reference to `dc_tensor` until SetWeight returns
                try:
                    dc_tensor = DenseCoreTensor.from_numpy(arr)

                    # Call C API
                    self._lib.DenseCoreSetWeight(
                        self._handle, key.encode("utf-8"), ctypes.byref(dc_tensor)
                    )
                    count += 1
                except ValueError as e:
                    if self._verbose:
                        print(f"[DenseCore] Warning: Skipping tensor {key}: {e}")
                except Exception as e:
                    print(f"[DenseCore] Error loading tensor {key}: {e}")
                    raise

        if self._verbose:
            print(f"[DenseCore] Successfully loaded {count} tensors from {model_path}")

    def _init_tokenizer(self, hf_repo_id: str) -> None:
        """Initialize tokenizer from HuggingFace repo."""
        try:
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
                if self._verbose:
                    print(f"[DenseCore] Loaded tokenizer from {hf_repo_id}")
            except Exception as e:
                # Retry with trust_remote_code for models that require custom tokenizers
                should_trust = os.getenv("DENSECORE_HF_TRUST_REMOTE_CODE", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                hf_lower = hf_repo_id.lower()
                if not should_trust and any(key in hf_lower for key in ("glm", "minimax")):
                    should_trust = True
                if should_trust:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            hf_repo_id, trust_remote_code=True
                        )
                        if self._verbose:
                            print(
                                f"[DenseCore] Loaded tokenizer with trust_remote_code from {hf_repo_id}"
                            )
                        return
                    except Exception as e2:
                        if getattr(self, "_verbose", False):
                            print(
                                f"[DenseCore] Warning: Failed to load tokenizer with trust_remote_code: {e2}"
                            )
                if getattr(self, "_verbose", False):
                    print(f"[DenseCore] Warning: Failed to load tokenizer from {hf_repo_id}: {e}")
                self.tokenizer = None
        except ImportError as e:
            if getattr(self, "_verbose", False):
                print(f"[DenseCore] transformers load failed: {e}")
                print("[DenseCore] tokenizer disabled")
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer from {hf_repo_id}: {e}", stacklevel=2)

    def _clean_bpe_token(self, token: str) -> str:
        """Clean BPE artifacts from token string."""
        # Replace Ġ with space
        # Replace Ċ with newline
        # This is a simple heuristic for BPE/SentencePiece tokens
        if not token:
            return token

        # Common BPE replacements
        s = token.replace("Ġ", " ").replace("Ċ", "\n")
        # Handle other common replacement characters if needed
        return s

    def _global_callback(self, token: bytes, is_finished: int, user_data: int) -> None:
        """Global callback called from C.

        Handles both legacy string format and new TokenResult struct format.
        For TokenResult, the token parameter contains the struct data.

        Thread Safety:
            - Uses try/finally to ensure _active_ctypes_refs cleanup even on exception
            - Prevents memory leaks from callback errors
        """
        req_id = user_data
        finished = bool(is_finished)

        with self._lock:
            handler = self._requests.get(req_id)

        if not handler:
            # Still need to clean up refs if finished, even without handler
            if finished:
                with self._lock:
                    self._requests.pop(req_id, None)
                    self._active_ctypes_refs.pop(req_id, None)
            return

        try:
            # Decode token string
            token_str = token.decode("utf-8", errors="replace") if token else ""

            # Since C++ now sends decoded tokens, just clean BPE artifacts
            token_str = self._clean_bpe_token(token_str)

            handler(token_str, finished)
        except Exception as e:
            import traceback

            error_msg = f"Error in callback handler for req {req_id}: {e}\n{traceback.format_exc()}"
            print(f"[DenseCore] {error_msg}")
            warnings.warn(error_msg, stacklevel=2)
        finally:
            if finished:
                with self._lock:
                    self._requests.pop(req_id, None)
                    self._active_ctypes_refs.pop(req_id, None)

    def _register_request(self, handler: Callable[[str, bool], None]) -> int:
        """Register a request handler and return request ID."""
        with self._lock:
            self._req_id_counter += 1
            req_id = self._req_id_counter
            self._requests[req_id] = handler
            return req_id

    def _resolve_generation_params(
        self,
        config: Optional[GenerationConfig],
        kwargs: Dict[str, Any],
        default_max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Resolve generation parameters from config and kwargs.
        Returns a dictionary of effective parameters.
        """
        # 1. Initialize with Defaults
        params = {
            "max_tokens": default_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "json_mode": False,
            "stop_sequences": [],
            "return_dict": False,
        }

        # 2. Apply Config (if provided)
        if config:
            params["max_tokens"] = config.max_tokens
            params["temperature"] = config.temperature
            params["top_p"] = config.top_p
            params["top_k"] = config.top_k
            params["repetition_penalty"] = config.repetition_penalty
            params["json_mode"] = config.json_mode
            params["stop_sequences"] = list(config.stop_sequences)
            if config.return_dict_in_generate:
                params["return_dict"] = True

        # 3. Apply Kwargs (Overrides)
        # Max tokens logic
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_new_tokens"]
        elif "max_length" in kwargs:
            # Logic handled by caller or heuristic here? Caller has prompt length context.
            # For simplicity, we assume caller handles max_length-to-max_new_tokens conversion
            # or we leave it as is if not resolvable.
            pass

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            params["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
            params["repetition_penalty"] = kwargs["repetition_penalty"]

        # JSON mode
        if "json_mode" in kwargs:
            params["json_mode"] = kwargs["json_mode"]

        # return_dict
        if "return_dict_in_generate" in kwargs:
            params["return_dict"] = kwargs["return_dict_in_generate"]

        # Handle do_sample=False
        if kwargs.get("do_sample") is False:
            params["temperature"] = 0.0
            params["top_p"] = 1.0
            params["top_k"] = 0

        # Stop sequences merging
        stop_from_kwargs = kwargs.get("stop")
        if isinstance(stop_from_kwargs, str):
            params["stop_sequences"].append(stop_from_kwargs)
        elif isinstance(stop_from_kwargs, (list, tuple)):
            params["stop_sequences"].extend([s for s in stop_from_kwargs if isinstance(s, str)])

        stopping_criteria = kwargs.get("stopping_criteria")
        if stopping_criteria:
            try:
                if isinstance(stopping_criteria, (list, tuple)):
                    for item in stopping_criteria:
                        if isinstance(item, str):
                            params["stop_sequences"].append(item)
                        elif hasattr(item, "stop_sequences"):
                            params["stop_sequences"].extend(getattr(item, "stop_sequences"))
                        elif hasattr(item, "stops"):
                            params["stop_sequences"].extend(getattr(item, "stops"))
                elif hasattr(stopping_criteria, "stop_sequences"):
                    params["stop_sequences"].extend(getattr(stopping_criteria, "stop_sequences"))
                elif hasattr(stopping_criteria, "stops"):
                    params["stop_sequences"].extend(getattr(stopping_criteria, "stops"))
            except Exception:
                pass

        return params

    def _submit_request(
        self,
        req_id: int,
        prompt: Union[str, List[int]],
        params: Dict[str, Any],
    ) -> int:
        """Submit a request (text or tokens) to the engine."""
        input_mode = "ids"
        prompt_bytes = None
        tokens_array = None
        n_tokens = 0
        prompt = self._maybe_prime_qwen_no_thinking(prompt)

        # 1. Prepare input
        if isinstance(prompt, str):
            # Prefer tokenizer path for parity with token-id submit APIs.
            # Fall back to raw prompt submit APIs when tokenizer is unavailable.
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                tokens_array = (ctypes.c_int * len(tokens))(*tokens)
                n_tokens = len(tokens)
            else:
                input_mode = "text"
                prompt_bytes = prompt.encode("utf-8")
        else:
            tokens_array = (ctypes.c_int * len(prompt))(*prompt)
            n_tokens = len(prompt)

        # CRITICAL: keep ctypes references alive until callback finishes
        with self._lock:
            refs: List[Any] = []
            if tokens_array is not None:
                refs.append(tokens_array)
            if prompt_bytes is not None:
                refs.append(prompt_bytes)
            self._active_ctypes_refs[req_id] = refs

        # 2. Prepare Stop Sequences
        c_stop_sequences = None
        if params["stop_sequences"]:
            c_strs = [s.encode("utf-8") for s in params["stop_sequences"]]
            c_stop_sequences = (ctypes.c_char_p * (len(c_strs) + 1))(*c_strs, None)
            with self._lock:
                # Append to existing list
                self._active_ctypes_refs[req_id].append(c_stop_sequences)

        # 3. Call Appropriate C API
        lora_name = params.get("lora_name")
        lora_name_bytes = (
            lora_name.encode("utf-8") if isinstance(lora_name, str) and lora_name else None
        )

        if getattr(self, "_has_sampling_api", False):
            if input_mode == "text":
                if getattr(self, "_has_lora_sampling_api", False) and lora_name_bytes is not None:
                    func = self._lib.SubmitRequestWithSamplingEx
                    args = [
                        self._handle,
                        prompt_bytes,
                        params["max_tokens"],
                        lora_name_bytes,
                        ctypes.c_float(params["temperature"]),
                        ctypes.c_float(params["top_p"]),
                        ctypes.c_int(params["top_k"]),
                        ctypes.c_float(params["repetition_penalty"]),
                        c_stop_sequences,
                        ctypes.c_int(1 if params["json_mode"] else 0),
                        ctypes.cast(self._c_callback, ctypes.c_void_p),
                        ctypes.c_void_p(req_id),
                    ]
                    return func(*args)

                func = self._lib.SubmitRequestWithSampling
                args = [
                    self._handle,
                    prompt_bytes,
                    params["max_tokens"],
                    ctypes.c_float(params["temperature"]),
                    ctypes.c_float(params["top_p"]),
                    ctypes.c_int(params["top_k"]),
                    ctypes.c_float(params["repetition_penalty"]),
                    c_stop_sequences,
                    ctypes.c_int(1 if params["json_mode"] else 0),
                    ctypes.cast(self._c_callback, ctypes.c_void_p),
                    ctypes.c_void_p(req_id),
                ]
                return func(*args)

            if getattr(self, "_has_lora_sampling_api", False) and lora_name_bytes is not None:
                func = self._lib.SubmitRequestIdsWithSamplingEx
                args = [
                    self._handle,
                    tokens_array,
                    n_tokens,
                    params["max_tokens"],
                    lora_name_bytes,
                    ctypes.c_float(params["temperature"]),
                    ctypes.c_float(params["top_p"]),
                    ctypes.c_int(params["top_k"]),
                    ctypes.c_float(params["repetition_penalty"]),
                    c_stop_sequences,
                    ctypes.c_int(1 if params["json_mode"] else 0),
                    ctypes.cast(self._c_callback, ctypes.c_void_p),
                    ctypes.c_void_p(req_id),
                ]
                return func(*args)

            func = self._lib.SubmitRequestIdsWithSampling
            args = [
                self._handle,
                tokens_array,
                n_tokens,
                params["max_tokens"],
                ctypes.c_float(params["temperature"]),
                ctypes.c_float(params["top_p"]),
                ctypes.c_int(params["top_k"]),
                ctypes.c_float(params["repetition_penalty"]),
                c_stop_sequences,
                ctypes.c_int(1 if params["json_mode"] else 0),
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            ]
            return func(*args)

        # Fallback APIs (without explicit sampling support)
        if input_mode == "text":
            if params["json_mode"]:
                warnings.warn(
                    "JSON mode requested for raw prompt input, but this engine path does not support it. Ignoring.",
                    RuntimeWarning,
                )
            args = [
                self._handle,
                prompt_bytes,
                params["max_tokens"],
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            ]
            return self._lib.SubmitRequest(*args)

        args = [
            self._handle,
            tokens_array,
            n_tokens,
            params["max_tokens"],
        ]

        if params["json_mode"] and hasattr(self._lib, "SubmitRequestIdsWithFormat"):
            if getattr(self, "_has_lora_format_api", False) and lora_name_bytes is not None:
                func = self._lib.SubmitRequestIdsWithFormatEx
                args.append(lora_name_bytes)
                args.append(1)  # json_mode
            else:
                func = self._lib.SubmitRequestIdsWithFormat
                args.append(1)  # json_mode
        else:
            if params["json_mode"]:
                warnings.warn(
                    "JSON mode requested but underlying C++ engine does not support it (version mismatch). Ignoring.",
                    RuntimeWarning,
                )
            func = self._lib.SubmitRequestIds

        args.extend(
            [
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            ]
        )

        return func(*args)

    # =========================================================================
    # Public API - Generation Methods
    # =========================================================================

    def generate(
        self,
        prompt: Optional[Union[str, List[int], Any]] = None,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        # LoRA adapter selection
        lora_adapter: Optional[str] = None,
        # HuggingFace Transformers compatible kwargs
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: int = 0,
        min_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        repetition_penalty: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        stopping_criteria: Optional[Any] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # DenseCore specific
        json_mode: bool = False,
        **kwargs,
    ) -> Union[str, "GenerateOutput"]:  # noqa: F821 - forward reference
        """
        Generate text from a prompt.

        This method is fully compatible with HuggingFace Transformers' generate() API,
        enabling drop-in replacement for existing inference code.

        Args:
            prompt: Input text string or token IDs (list/tensor)
            max_tokens: Maximum tokens to generate (DenseCore style)
            config: Optional GenerationConfig for advanced settings
            lora_adapter: Name of preloaded LoRA adapter to use for this request.
                          If specified and different from current, switches adapter.

            # HuggingFace Transformers compatible args:
            input_ids: Token IDs as list, numpy array, or torch.Tensor
            attention_mask: Attention mask (ignored, for API compatibility)
            max_new_tokens: Maximum new tokens to generate (HF style)
            max_length: Maximum total length including prompt (HF style)
            min_length: Minimum total length (HF style)
            min_new_tokens: Minimum new tokens (HF style)
            do_sample: Enable sampling (False = greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            num_beams: Beam search (only 1 supported)
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID(s)
            stopping_criteria: StoppingCriteriaList for custom stopping
            return_dict_in_generate: Return GenerateOutput instead of string
            output_scores: Return generation scores (not supported)
            output_attentions: Return attention weights (not supported)
            output_hidden_states: Return hidden states (not supported)
            **kwargs: Additional parameters

        Returns:
            str: Generated text (default)
            GenerateOutput: If return_dict_in_generate=True

        Example:
            >>> # DenseCore style
            >>> response = model.generate("Hello!", max_tokens=100)

            >>> # HuggingFace style
            >>> output = model.generate(
            ...     "Hello!",
            ...     max_new_tokens=100,
            ...     do_sample=True,
            ...     temperature=0.7,
            ...     top_p=0.9,
            ...     return_dict_in_generate=True,
            ... )
            >>> print(output.text)

            >>> # Token ID input (HuggingFace style)
            >>> input_ids = tokenizer.encode("Hello!", return_tensors="pt")
            >>> output = model.generate(input_ids=input_ids, max_new_tokens=100)
        """
        # Import here to avoid circular import
        from .generate_output import GenerateOutput

        # =======================================================================
        # LoRA Adapter Selection: Switch adapter if specified
        # =======================================================================
        if lora_adapter is not None:
            current_adapter = self._lora_manager.active_adapter
            if lora_adapter != current_adapter:
                if lora_adapter not in self._lora_manager.adapters:
                    raise ValueError(
                        f"LoRA adapter '{lora_adapter}' not loaded. "
                        f"Available: {self._lora_manager.list_adapters()}"
                    )

        # =======================================================================
        # Input Processing: Support string, list[int], numpy array, torch.Tensor
        # =======================================================================
        prompt_arg = self._normalize_prompt(prompt, input_ids)

        # =======================================================================
        # Parameter Resolution: config > explicit kwargs > defaults
        # =======================================================================
        resolve_kwargs = dict(kwargs)
        if max_new_tokens is not None:
            resolve_kwargs["max_new_tokens"] = max_new_tokens
        if max_length is not None:
            resolve_kwargs["max_length"] = max_length
        if min_length > 0:
            resolve_kwargs["min_length"] = min_length
        if min_new_tokens is not None:
            resolve_kwargs["min_new_tokens"] = min_new_tokens
        resolve_kwargs["do_sample"] = do_sample
        if temperature is not None:
            resolve_kwargs["temperature"] = temperature
        if top_p is not None:
            resolve_kwargs["top_p"] = top_p
        if top_k is not None:
            resolve_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            resolve_kwargs["repetition_penalty"] = repetition_penalty
        if stopping_criteria is not None:
            resolve_kwargs["stopping_criteria"] = stopping_criteria
        resolve_kwargs["return_dict_in_generate"] = return_dict_in_generate
        resolve_kwargs["json_mode"] = json_mode

        params = self._resolve_generation_params(
            config, resolve_kwargs, default_max_tokens=max_tokens
        )
        if lora_adapter is not None:
            params["lora_name"] = lora_adapter

        # Helper logic for max_length (needs prompt len)
        if "max_length" in resolve_kwargs and resolve_kwargs["max_length"] is not None:
            prompt_len = (
                len(prompt_arg) if isinstance(prompt_arg, list) else len(prompt_arg or "") // 4
            )
            params["max_tokens"] = max(1, resolve_kwargs["max_length"] - prompt_len)

        if eos_token_id is not None:
            warnings.warn(
                "eos_token_id is currently accepted for API compatibility but is not applied "
                "at the C++ boundary yet. Use stop sequences for deterministic stopping.",
                UserWarning,
            )

        # Warn about unsupported features
        if num_beams > 1:
            warnings.warn(
                f"DenseCore only supports num_beams=1. Requested num_beams={num_beams} will be ignored.",
                UserWarning,
            )
        if output_scores or output_attentions or output_hidden_states:
            warnings.warn(
                "output_scores, output_attentions, and output_hidden_states are not supported.",
                UserWarning,
            )

        effective_return_dict = params["return_dict"]

        # =======================================================================
        # Generation Execution
        # =======================================================================
        result_queue: queue.Queue = queue.Queue()
        generated_token_ids: List[int] = []

        def handler(token: str, finished: bool) -> None:
            result_queue.put((token, finished))

        req_id = self._register_request(handler)

        try:
            # Submit request (handled by helper)
            ret = self._submit_request(req_id, prompt_arg, params)

            if ret < 0:
                raise _error_code_to_exception(ret, context="in generate()")
        except Exception:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise

        # Collect generated tokens
        tokens: List[str] = []
        while True:
            token, finished = result_queue.get()
            tokens.append(token)
            if finished:
                break

        generated_text = "".join(tokens)

        # =======================================================================
        # Return Format
        # =======================================================================
        if effective_return_dict:
            # Build GenerateOutput compatible with HuggingFace
            prompt_token_ids = prompt_arg if isinstance(prompt_arg, list) else []
            if self.tokenizer and isinstance(prompt_arg, str):
                prompt_token_ids = self.tokenizer.encode(prompt_arg, add_special_tokens=False)

            # Decode generated tokens to IDs if tokenizer available
            if self.tokenizer:
                generated_token_ids = self.tokenizer.encode(
                    generated_text, add_special_tokens=False
                )
            else:
                generated_token_ids = []

            # Full sequence = prompt + generated
            full_sequence = list(prompt_token_ids) + generated_token_ids

            return GenerateOutput(
                sequences=[full_sequence],
                text=generated_text,
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(generated_token_ids),
                    "total_tokens": len(full_sequence),
                },
            )

        return generated_text

    def _coerce_to_token_ids(self, input_data: Any) -> List[int]:
        """
        Convert various input types to a flat list of token IDs.

        Supports:
        - list[int]: Direct token IDs
        - list[list[int]]: Batched token IDs (takes first sequence)
        - numpy.ndarray: NumPy array of token IDs
        - torch.Tensor: PyTorch tensor of token IDs

        Args:
            input_data: Input in any supported format

        Returns:
            list[int]: Flat list of token IDs

        Raises:
            ValueError: If input format is not supported
        """
        # Already a list
        if isinstance(input_data, list):
            if not input_data:
                return []
            # Check if it's a list of lists (batched)
            if isinstance(input_data[0], list):
                return list(input_data[0])  # Take first sequence
            # Flat list of ints
            return [int(x) for x in input_data]

        # NumPy array
        try:
            import numpy as np

            if isinstance(input_data, np.ndarray):
                flat = input_data.flatten() if input_data.ndim > 1 else input_data
                return flat.tolist()
        except ImportError:
            pass

        # PyTorch tensor
        try:
            import torch

            if isinstance(input_data, torch.Tensor):
                flat = input_data.flatten() if input_data.dim() > 1 else input_data
                return flat.tolist()
        except ImportError:
            pass

        # Try generic iteration
        try:
            return [int(x) for x in input_data]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot convert input to token IDs. Expected list[int], numpy array, or torch.Tensor. "
                f"Got {type(input_data).__name__}"
            ) from e

    def _normalize_prompt(
        self,
        prompt: Optional[Union[str, List[int], Any]] = None,
        input_ids: Optional[Any] = None,
    ) -> Union[str, List[int]]:
        """
        Normalize input to either a string prompt or a list of token IDs.

        Prioritizes input_ids > prompt.
        Handles list, numpy, torch, and nested lists for token IDs.
        """
        prompt_tokens: Optional[List[int]] = None
        prompt_text: Optional[str] = None

        if input_ids is not None:
            prompt_tokens = self._coerce_to_token_ids(input_ids)
        elif prompt is not None:
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_tokens = self._coerce_to_token_ids(prompt)

        if prompt_text is None and prompt_tokens is None:
            raise ValueError("Either 'prompt' or 'input_ids' must be provided")

        return prompt_tokens if prompt_tokens is not None else prompt_text

    def _should_prime_qwen_no_thinking(self) -> bool:
        model_name = str(getattr(self, "model_path", "")).lower()
        if "qwen3.5" in model_name or "qwen35" in model_name:
            return os.getenv("DENSECORE_QWEN35_ENABLE_THINKING", "0") not in ("1", "true", "True")
        if "qwen3" in model_name:
            return os.getenv("DENSECORE_QWEN3_ENABLE_THINKING", "1") not in ("1", "true", "True")
        return False

    def _maybe_prime_qwen_no_thinking(self, prompt: Union[str, List[int]]) -> Union[str, List[int]]:
        if not self._should_prime_qwen_no_thinking():
            return prompt

        def apply_no_thinking_text(text: str) -> str:
            if "You are a helpful assistant." in text and "final answer only" not in text:
                text = text.replace(
                    "You are a helpful assistant.",
                    "You are a helpful assistant.\nProvide only the answer. Do not output any thinking process, analysis, reasoning steps, or preamble. Never start with 'Thinking Process'.",
                    1,
                )
            if "<|im_start|>assistant\n" in text and "Answer:" not in text:
                text = text.replace("<|im_start|>assistant\n", "<|im_start|>assistant\nAnswer: ", 1)
            return text

        if isinstance(prompt, str):
            return apply_no_thinking_text(prompt)

        if self.tokenizer is None:
            return prompt

        prompt_text = self.tokenizer.decode(prompt, skip_special_tokens=False)
        primed_text = apply_no_thinking_text(prompt_text)
        if primed_text == prompt_text:
            return prompt
        return self.tokenizer.encode(primed_text, add_special_tokens=False)

    def stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text prompt (str) or pre-tokenized token IDs (list[int])
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig
            json_mode: If True, attempts to generate JSON output.
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens one at a time

        Example:
            >>> for token in model.stream("Tell me a story"):
            ...     print(token, end="", flush=True)

            >>> # With pre-tokenized input
            >>> token_ids = [1, 2, 3, 4, 5]
            >>> for token in model.stream(token_ids, max_tokens=50):
            ...     print(token, end="", flush=True)
        """
        # Consolidate params
        resolve_kwargs = dict(kwargs)
        resolve_kwargs["json_mode"] = json_mode
        params = self._resolve_generation_params(
            config, resolve_kwargs, default_max_tokens=max_tokens
        )

        result_queue: queue.Queue = queue.Queue()

        def handler(token: str, finished: bool) -> None:
            result_queue.put((token, finished))

        req_id = self._register_request(handler)

        try:
            # Submit request
            prompt_arg = self._normalize_prompt(prompt, resolve_kwargs.get("input_ids"))
            ret = self._submit_request(req_id, prompt_arg, params)
            if ret < 0:
                raise RuntimeError(f"Failed to submit request: error code {ret}")
        except Exception:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise

        while True:
            token, finished = result_queue.get()

            if token:
                yield token
            if finished:
                break

    # Alias for HuggingFace-style API
    generate_stream = stream

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Asynchronously generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig

        Returns:
            Generated text string

        Example:
            >>> response = await model.generate_async("Hello!")
            >>> print(response)
        """
        # Resolve params
        params = self._resolve_generation_params(config, kwargs, default_max_tokens=max_tokens)

        # Async logic
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        accumulated: List[str] = []

        def handler(token: str, finished: bool) -> None:
            if token:
                accumulated.append(token)
            if finished:
                result = "".join(accumulated)
                loop.call_soon_threadsafe(future.set_result, result)

        req_id = self._register_request(handler)

        def blocking_submit() -> int:
            """Execute SubmitRequest in executor."""
            try:
                prompt_arg = self._normalize_prompt(prompt, kwargs.get("input_ids"))
                return self._submit_request(req_id, prompt_arg, params)
            except Exception as e:
                # Need to handle exception propagation to future if submission fails?
                # For now just return err code simulation if exceptions happen
                print(f"[DenseCore] SubmitAsync failed: {e}")
                return -1

        ret = await loop.run_in_executor(_get_inference_executor(), blocking_submit)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)  # Clean up if we added failure
            raise _error_code_to_exception(ret, context="in generate_async()")

        return await future

    async def stream_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream generated tokens.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig

        Yields:
            Generated tokens one at a time

        Example:
            >>> async for token in model.stream_async("Hello"):
            ...     print(token, end="", flush=True)
        """
        # Consolidate params
        params = self._resolve_generation_params(config, kwargs, default_max_tokens=max_tokens)

        # Async input processing
        prompt_arg = self._normalize_prompt(prompt, kwargs.get("input_ids"))

        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue = asyncio.Queue()

        def handler(token: str, finished: bool) -> None:
            loop.call_soon_threadsafe(async_queue.put_nowait, (token, finished))

        req_id = self._register_request(handler)

        def blocking_submit() -> int:
            """Execute SubmitRequest in executor."""
            try:
                return self._submit_request(req_id, prompt_arg, params)
            except Exception as e:
                print(f"[DenseCore] stream_async submit failed: {e}")
                return -1

        ret = await loop.run_in_executor(_get_inference_executor(), blocking_submit)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise _error_code_to_exception(ret, context="in stream_async()")

        while True:
            token, finished = await async_queue.get()
            if token:
                yield token
            if finished:
                break

    # Alias for compatibility
    generate_stream_async = stream_async

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
    ) -> List[str]:
        """
        Generate responses for multiple prompts using C++ continuous batching.

        This method submits all requests to the C++ engine's scheduler before
        waiting, allowing the engine to form efficient batches internally.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response

        Returns:
            List of generated responses in the same order as prompts

        Example:
            >>> responses = model.generate_batch([
            ...     "Hello!",
            ...     "How are you?",
            ...     "Tell me a joke"
            ... ])
        """
        if not prompts:
            return []

        # Create a queue for each request to collect results
        result_queues: List[queue.Queue] = [queue.Queue() for _ in prompts]
        req_ids: List[int] = []

        # Submit all requests (non-blocking) - this pushes them to the C++
        # pending_queue, enabling the Scheduler to form efficient batches
        for idx, prompt in enumerate(prompts):
            result_queue = result_queues[idx]

            def make_handler(q: queue.Queue) -> Callable[[str, bool], None]:
                """Factory to capture the queue in closure."""
                tokens: List[str] = []

                def handler(token: str, finished: bool) -> None:
                    tokens.append(token)
                    if finished:
                        q.put("".join(tokens))

                return handler  # noqa: B023 - handler is defined in factory function

            handler = make_handler(result_queue)
            req_id = self._register_request(handler)
            req_ids.append(req_id)

            # Prepare default parameters
            params = {
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "json_mode": False,
                "stop_sequences": [],
                "return_dict": False,
            }

            ret = self._submit_request(req_id, prompt, params)

            if ret < 0:
                # Clean up the registered request on failure
                with self._lock:
                    self._requests.pop(req_id, None)
                raise _error_code_to_exception(
                    ret, context=f"submitting prompt {idx} in generate_batch"
                )

        # Wait for all results (order preserved by queue indices)
        results: List[str] = []
        for result_queue in result_queues:
            result = result_queue.get()  # Blocks until result is ready
            results.append(result)

        return results

    # =========================================================================
    # Chat Interface (OpenAI-compatible)
    # =========================================================================

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Chat-style interface compatible with OpenAI API format.

        Supports function calling when tools are provided.

        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to prepend
            tools: Optional list of tool definitions for function calling
            tool_choice: Tool selection mode ('none', 'auto', 'required')
            return_dict: If True, return dict with role/content/tool_calls
            **kwargs: Additional generation parameters

        Returns:
            str: Assistant's response (default)
            dict: If return_dict=True or tools provided, returns structured response

        Example:
            >>> # Simple usage
            >>> response = model.chat([{"role": "user", "content": "Hello!"}])
            >>> print(response)

            >>> # With function calling
            >>> result = model.chat(
            ...     messages=[{"role": "user", "content": "날씨 알려줘"}],
            ...     tools=[{"type": "function", "function": {...}}],
            ... )
            >>> if result.get("tool_calls"):
            ...     print("Tool called:", result["tool_calls"])
        """
        extra_system_messages = []

        # Add tool definitions if provided
        if tools and tool_choice != "none":
            tool_desc = self._format_tools_prompt(tools)
            extra_system_messages.append(tool_desc)

        if system_prompt:
            extra_system_messages.append(system_prompt)

        model_hint = self._hf_repo_id or self._model_path
        prompt = format_chat_prompt(
            model_hint,
            messages,
            extra_system_messages=extra_system_messages,
        )

        response_text = self.generate(prompt, max_tokens, **kwargs)

        # If tools provided, try to parse tool calls
        tool_calls = None
        if tools and tool_choice != "none":
            tool_calls, clean_text = self._parse_tool_calls(response_text)
            if tool_calls:
                response_text = clean_text

        # Return format based on configuration
        if return_dict or tools:
            result = {
                "role": "assistant",
                "content": response_text if not tool_calls else None,
            }
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        return response_text

    # =========================================================================
    # LoRA Adapter Management
    # =========================================================================

    def load_lora(
        self,
        adapter_path: str,
        scale: float = 1.0,
        name: str = "default",
    ) -> None:
        """
        Load a LoRA adapter for fine-tuned inference.

        Args:
            adapter_path: Path to the GGUF LoRA adapter file
            scale: LoRA scaling factor (alpha). Higher = stronger adapter effect
            name: Identifier for this adapter

        Example:
            >>> model = DenseCore("base_model.gguf")
            >>> model.load_lora("./my-adapter.gguf", scale=0.8)
        """
        # Track in Python LoRA manager
        self._lora_manager.load(name, adapter_path, scale=scale, activate=False)

        # Call C++ API to actually load the adapter
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.LoadLoraAdapter(
                self._handle,
                adapter_path.encode("utf-8"),
                ctypes.c_float(scale),
                name.encode("utf-8"),
            )
            if ret < 0:
                raise RuntimeError(f"Failed to load LoRA adapter: error code {ret}")

            # Activate immediately
            self._lib.ActivateLoraAdapter(self._handle, name.encode("utf-8"))
            self._lora_manager.activate(name)
        else:
            # Fallback: Python-level tracking only (legacy warning removed)
            self._lora_manager.activate(name)

        if self._verbose:
            print(f"[LoRA] Loaded adapter '{name}' with scale={scale}")

    def unload_lora(self, name: str = "default") -> None:
        """
        Unload a LoRA adapter.

        Args:
            name: Adapter identifier to unload
        """
        # Call C++ API first
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.UnloadLoraAdapter(self._handle, name.encode("utf-8"))
            if ret < 0:
                warnings.warn(
                    f"Failed to unload LoRA adapter in C++: error code {ret}", stacklevel=2
                )

        self._lora_manager.unload(name)

    def enable_lora(self, name: Optional[str] = None) -> None:
        """
        Enable a loaded LoRA adapter.

        Args:
            name: Adapter to enable (None = activate last loaded)
        """
        if name is None and self._lora_manager.list_adapters():
            name = self._lora_manager.list_adapters()[0]

        if not name:
            return

        # Call C++ API
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.ActivateLoraAdapter(self._handle, name.encode("utf-8"))
            if ret < 0:
                raise RuntimeError(f"Failed to activate LoRA adapter: error code {ret}")

        self._lora_manager.activate(name)

    def disable_lora(self) -> None:
        """Disable all LoRA adapters (use base model only)."""
        # Call C++ API
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.DeactivateLoraAdapters(self._handle)
            if ret < 0:
                warnings.warn(
                    f"Failed to deactivate LoRA adapters in C++: error code {ret}", stacklevel=2
                )

        self._lora_manager.deactivate()

    @property
    def has_lora(self) -> bool:
        """Check if any LoRA adapter is active."""
        return self._lora_manager.is_active

    def list_lora_adapters(self) -> List[str]:
        """List all loaded LoRA adapter names."""
        return self._lora_manager.list_adapters()

    # =========================================================================
    # Tool Calling Helper Methods
    # =========================================================================

    def _format_tools_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool definitions into a prompt section."""
        lines = ["# Available Tools", ""]
        lines.append("You can call the following tools by responding with JSON:")
        lines.append("```json")
        lines.append(
            '{"tool_calls": [{"id": "call_<id>", "type": "function", "function": {"name": "<name>", "arguments": "<json>"}}]}'
        )
        lines.append("```")
        lines.append("")

        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            lines.append(f"## {name}")
            if desc:
                lines.append(desc)
            lines.append("Parameters:")
            lines.append(f"```json\n{json.dumps(params, indent=2)}\n```")
            lines.append("")

        return "\n".join(lines)

    def _parse_tool_calls(self, text: str) -> tuple:
        """
        Parse tool calls from model output.

        Handles multiple output formats:
        1. JSON in markdown code blocks (```json ... ```)
        2. Raw JSON with tool_calls key
        """
        json_candidates = []

        # Strategy 1: Extract JSON from markdown code blocks first
        # This handles: ```json\n{"tool_calls": [...]}\n```
        code_block_pattern = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)
        for match in code_block_pattern.finditer(text):
            content = match.group(1).strip()
            if '"tool_calls"' in content:
                json_candidates.append((content, match.group(0)))

        # Strategy 2: Direct JSON pattern (fallback)
        # Handles raw JSON without code blocks
        direct_pattern = re.compile(r'\{[^{}]*"tool_calls"\s*:\s*\[[\s\S]*?\]\s*\}')
        for match in direct_pattern.finditer(text):
            json_candidates.append((match.group(0), match.group(0)))

        # Try parsing each candidate
        for json_str, full_match in json_candidates:
            try:
                parsed = json.loads(json_str)
                tool_calls = parsed.get("tool_calls", [])
                if tool_calls:
                    # Clean output by removing the matched content
                    clean_text = text.replace(full_match, "").strip()
                    # Ensure all tool calls have required fields
                    for i, tc in enumerate(tool_calls):
                        if "id" not in tc:
                            tc["id"] = f"call_{i}"
                        if "type" not in tc:
                            tc["type"] = "function"
                    return tool_calls, clean_text
            except json.JSONDecodeError:
                continue

        return None, text

    # =========================================================================
    # Rerank API
    # =========================================================================

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to a query.

        Note:
            **Current Implementation**: Uses Jaccard word similarity as a
            lightweight placeholder. This will be replaced with embedding-based
            cosine similarity once embedding model support is implemented.

        Args:
            query: Query text to rank documents against
            documents: List of document texts to rerank
            top_n: Number of top results to return (default: all)
            return_documents: Include document text in results

        Returns:
            List of dicts with 'index', 'relevance_score', and optionally 'document'

        Example:
            >>> results = model.rerank(
            ...     query="What is DenseCore?",
            ...     documents=[
            ...         "DenseCore is a CPU inference engine",
            ...         "Python is a programming language",
            ...         "LLM optimization techniques"
            ...     ],
            ...     top_n=2
            ... )
            >>> for r in results:
            ...     print(f"Index {r['index']}: {r['relevance_score']:.3f}")
        """
        if not documents:
            return []

        # Placeholder implementation using Jaccard word similarity.
        # TODO(embeddings): Replace with embedding-based cosine similarity
        # once the embedding API is implemented in the C++ engine.
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scores = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())

            # Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            score = intersection / union if union > 0 else 0.0

            scores.append({"index": i, "relevance_score": score, "text": doc})

        # Sort by score descending
        scores.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_n
        if top_n is not None and top_n < len(scores):
            scores = scores[:top_n]

        # Format results
        results = []
        for s in scores:
            result = {"index": s["index"], "relevance_score": s["relevance_score"]}
            if return_documents:
                result["document"] = {"text": s["text"]}
            results.append(result)

        return results

    # =========================================================================
    # Request Cancellation
    # =========================================================================

    def stop_generation(self, req_id: int) -> bool:
        """
        Request graceful cancellation of an in-progress generation.

        Signals the C++ backend to stop generating tokens for the specified
        request. The actual stopping may not be immediate as it depends on
        when the inference loop checks the cancellation flag.

        Args:
            req_id: Request ID returned from internal registration

        Returns:
            True if cancellation was successfully requested, False otherwise

        Raises:
            EngineNotInitializedError: If the engine is not initialized

        Note:
            This method only performs actual cancellation when the C++ backend's
            CancelRequest API is available. Otherwise, only Python-side cleanup
            is performed.

        Example:
            >>> req_id = model._register_request(handler)
            >>> # ... later, if user presses Ctrl+C ...
            >>> model.stop_generation(req_id)
        """
        if not self._handle:
            raise EngineNotInitializedError("Engine handle is null")

        # Attempt to call C++ CancelRequest if available
        # The C++ side should set a thread-safe cancellation flag
        if hasattr(self, "_has_cancel_api") and self._has_cancel_api:
            try:
                ret = self._lib.CancelRequest(self._handle, ctypes.c_int(req_id))
                if ret < 0:
                    if self._verbose:
                        print(f"[DenseCore] CancelRequest returned error: {ret}")
                    return False
            except Exception as e:
                if self._verbose:
                    print(f"[DenseCore] CancelRequest failed: {e}")
                return False

        # Clean up Python-side tracking regardless of C++ result
        with self._lock:
            self._requests.pop(req_id, None)
            self._active_ctypes_refs.pop(req_id, None)

        if self._verbose:
            print(f"[DenseCore] Cancellation requested for req_id={req_id}")

        return True

    def cancel_all_requests(self) -> int:
        """
        Request cancellation of all active requests.

        Calls stop_generation for all currently in-progress generation requests.
        Useful for resource cleanup or during shutdown.

        Returns:
            Number of requests for which cancellation was requested

        Example:
            >>> # During shutdown or interrupt handling
            >>> cancelled_count = model.cancel_all_requests()
            >>> print(f"Cancelled {cancelled_count} requests")
        """
        with self._lock:
            req_ids = list(self._requests.keys())

        cancelled = 0
        for req_id in req_ids:
            if self.stop_generation(req_id):
                cancelled += 1

        return cancelled

    # =========================================================================
    # Vision Inference (ViT, CLIP, SigLIP)
    # =========================================================================

    def encode_image(
        self,
        image: Union[str, np.ndarray, "Image.Image"],  # noqa: F821
        normalize: bool = True,
        image_size: int = 224,
        graph_name: str = "vit_base",
    ) -> np.ndarray:
        """
        Encode an image into embeddings using a Vision model.

        This method is available when the loaded model is a Vision Transformer
        (ViT, CLIP Vision, SigLIP). It processes the image and returns the
        embedding vector.

        Args:
            image: Image input - can be:
                   - str: Path to image file
                   - np.ndarray: Image array [H, W, C] or [C, H, W]
                   - PIL.Image: PIL Image object
            normalize: Whether to L2-normalize the output embeddings
            image_size: Target image size for resizing (default: 224)
            graph_name: Name of the registered graph builder

        Returns:
            np.ndarray: Image embedding vector [embed_dim] or [N, embed_dim] for batch

        Example:
            >>> model = densecore.from_pretrained("openai/clip-vit-base-patch32")
            >>> embedding = model.encode_image("photo.jpg")
            >>> print(embedding.shape)
            (768,)

            # For similarity comparison
            >>> emb1 = model.encode_image("cat.jpg")
            >>> emb2 = model.encode_image("dog.jpg")
            >>> similarity = np.dot(emb1, emb2)
        """
        if not getattr(self, "_has_graph_api", False):
            raise NotImplementedError(
                "Graph execution API not available. Please update your DenseCore library."
            )

        # Preprocess image to tensor
        image_tensor = self._preprocess_image(image, image_size)

        # Execute graph
        embedding = self._execute_graph_sync(
            inputs=[("image", image_tensor)],
            graph_name=graph_name,
            num_outputs=1,
        )

        # L2 normalize if requested
        if normalize and embedding is not None:
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
            embedding = embedding / (norm + 1e-8)

        return embedding

    def _preprocess_image(
        self,
        image: Union[str, np.ndarray, "Image.Image"],  # noqa: F821
        target_size: int = 224,
    ) -> np.ndarray:
        """Preprocess image to model input format [B, C, H, W]."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "PIL is required for image processing. Install with: pip install Pillow"
            )

        # Load image if path
        if isinstance(image, str):
            pil_image = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:  # [H, W, C]
                pil_image = PILImage.fromarray(image.astype(np.uint8))
            elif image.ndim == 3 and image.shape[0] == 3:  # [C, H, W]
                pil_image = PILImage.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            pil_image = image.convert("RGB")

        # Resize
        pil_image = pil_image.resize((target_size, target_size), PILImage.BILINEAR)

        # Convert to numpy [H, W, C] -> [C, H, W]
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        # Add batch dimension [1, C, H, W]
        return arr[np.newaxis, :, :, :].astype(np.float32)

    def _execute_graph_sync(
        self,
        inputs: List[tuple],
        graph_name: str,
        num_outputs: int = 1,
    ) -> np.ndarray:
        """Execute a graph synchronously and return output tensor."""
        # Create input tensor array
        input_tensors = []
        input_refs = []  # Keep references to prevent GC

        for name, arr in inputs:
            tensor_input = DenseCoreTensorInput.from_numpy(arr, name)
            input_tensors.append(tensor_input)
            input_refs.append(tensor_input)

        inputs_array = (DenseCoreTensorInput * len(input_tensors))(*input_tensors)

        # Create output buffer (estimate size from model config)
        # For ViT-Base: embed_dim = 768, seq_len = 197 (1 cls + 196 patches)
        max_output_size = 768 * 197 * 4  # Assume float32
        output_buffer = np.zeros(max_output_size, dtype=np.float32)

        # Create output tensor
        output_tensor = DenseCoreTensorOutput()
        output_tensor.name = b"output"
        output_tensor.data = output_buffer.ctypes.data
        output_tensor.ndim = 0  # Will be filled by C++
        output_tensor.dtype = 0  # F32

        outputs_array = (DenseCoreTensorOutput * num_outputs)(output_tensor)

        # Execute graph
        result = self._lib.ExecuteGraphSync(
            self._handle,
            inputs_array,
            len(input_tensors),
            graph_name.encode("utf-8"),
            outputs_array,
            num_outputs,
        )

        if result < 0:
            raise DenseCoreRuntimeError(f"Graph execution failed with code {result}")

        # Extract output tensor
        out = outputs_array[0]
        shape = tuple(out.shape[i] for i in range(out.ndim))

        # Calculate size and reshape
        size = 1
        for s in shape:
            size *= s

        return output_buffer[:size].reshape(shape)

    def process_image(
        self,
        image: Union[str, np.ndarray, "Image.Image"],  # noqa: F821
        normalize: bool = True,
        image_size: int = 224,
        graph_name: str = "vit_base",
    ) -> np.ndarray:
        """
        Process an image through a vision model and return embeddings.

        This is an alias for encode_image() for API consistency with the
        Universal Graph Execution design.

        Args:
            image: Image input (path, numpy array, or PIL Image)
            normalize: Whether to L2-normalize output embeddings
            image_size: Target image size for resizing
            graph_name: Name of the graph builder (e.g., "vit_base", "siglip")

        Returns:
            np.ndarray: Image embedding vector
        """
        return self.encode_image(
            image=image,
            normalize=normalize,
            image_size=image_size,
            graph_name=graph_name,
        )

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text into embeddings using CLIP/SigLIP text encoder.

        This method is available for CLIP and SigLIP models that include
        a text encoder alongside the vision encoder.

        Args:
            text: Text input - single string or list of strings
            normalize: Whether to L2-normalize the output embeddings

        Returns:
            np.ndarray: Text embedding vector [embed_dim] or [N, embed_dim] for batch

        Example:
            >>> model = densecore.from_pretrained("openai/clip-vit-base-patch32")
            >>> text_emb = model.encode_text("a photo of a cat")
            >>> image_emb = model.encode_image("cat.jpg")
            >>> similarity = np.dot(text_emb, image_emb)
        """
        raise NotImplementedError(
            "Text encoding for CLIP/SigLIP is not yet implemented. Use the C API directly for now."
        )

    # =========================================================================
    # Whisper Audio Inference
    # =========================================================================

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        return_timestamps: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio to text using Whisper.

        This method is available when the loaded model is a Whisper model.
        It processes audio and returns the transcription.

        Args:
            audio: Audio input - can be:
                   - str: Path to audio file (wav, mp3, flac, etc.)
                   - np.ndarray: Audio waveform [samples] at 16kHz
            language: Language code (e.g., "en", "ko", "ja"). None for auto-detect.
            task: "transcribe" or "translate" (to English)
            initial_prompt: Optional prompt to condition the model
            temperature: Sampling temperature (0 for greedy)
            beam_size: Beam search size (1 for greedy)
            return_timestamps: Whether to return word-level timestamps

        Returns:
            str: Transcription text (if return_timestamps=False)
            dict: {"text": str, "segments": [...]} (if return_timestamps=True)

        Example:
            >>> model = densecore.from_pretrained("openai/whisper-large-v3")
            >>> text = model.transcribe("audio.wav")
            >>> print(text)
            "Hello, this is a test recording."

            # With timestamps
            >>> result = model.transcribe("audio.wav", return_timestamps=True)
            >>> for seg in result["segments"]:
            ...     print(f"[{seg['start']:.2f}s] {seg['text']}")
        """
        # TODO: Implement C++ binding for Whisper inference
        raise NotImplementedError(
            "Whisper inference is not yet fully implemented in the Python SDK. "
            "The C++ core supports Whisper models - please use the C API directly "
            "or wait for the next release."
        )

    def detect_language(
        self,
        audio: Union[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Detect the language of audio using Whisper.

        Args:
            audio: Audio input (path or waveform)

        Returns:
            dict: Language probabilities {language_code: probability}

        Example:
            >>> model = densecore.from_pretrained("openai/whisper-large-v3")
            >>> probs = model.detect_language("audio.wav")
            >>> print(probs)
            {"en": 0.95, "fr": 0.03, "de": 0.02}
        """
        raise NotImplementedError(
            "Language detection is not yet implemented. "
            "Use transcribe() which auto-detects language."
        )

    # =========================================================================
    # Multimodal Inference (LLaVA, Qwen-VL)
    # =========================================================================

    def generate_with_image(
        self,
        prompt: str,
        image: Union[str, np.ndarray, "Image.Image"],  # noqa: F821
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text response given a prompt and image (multimodal).

        This method is available for multimodal models like LLaVA and Qwen-VL.
        It processes both the image and text prompt to generate a response.

        Args:
            prompt: Text prompt/question about the image
            image: Image input (path, array, or PIL Image)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text response

        Example:
            >>> model = densecore.from_pretrained("liuhaotian/llava-v1.6-34b")
            >>> response = model.generate_with_image(
            ...     prompt="What is in this image?",
            ...     image="photo.jpg"
            ... )
            >>> print(response)
            "This image shows a cat sitting on a windowsill..."
        """
        raise NotImplementedError(
            "Multimodal inference is not yet fully implemented. "
            "The C++ core supports multimodal models - please use the C API directly "
            "or wait for the next release."
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def model_path(self) -> str:
        """Return the path to the loaded model."""
        return self._model_path

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self._model_path,
            "has_tokenizer": self.tokenizer is not None,
            "has_lora": self.has_lora,
            "lora_adapters": self.list_lora_adapters(),
            "closed": self._closed,
        }

    def close(self) -> None:
        """Close the engine and release resources."""
        if not hasattr(self, "_closed"):
            return
        if not self._closed and getattr(self, "_handle", None):
            self._lib.FreeEngine(self._handle)
            self._handle = None
            self._closed = True
            if self._verbose:
                print("[DenseCore] Engine closed")

    def get_metrics(self) -> Optional[dict[str, float | int]]:
        if (
            self._closed
            or not getattr(self, "_handle", None)
            or not getattr(self, "_has_metrics", False)
        ):
            return None
        metrics = self._lib.GetMetrics(self._handle)
        return {
            "requests_per_second": float(metrics.requests_per_second),
            "tokens_per_second": float(metrics.tokens_per_second),
            "active_requests": int(metrics.active_requests),
            "total_tokens_generated": int(metrics.total_tokens_generated),
        }

    def __enter__(self) -> "DenseCore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return f"DenseCore(model='{self._model_path}', status={status})"
