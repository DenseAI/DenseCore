"""
DenseCore Embedding Module - High-performance text embeddings for RAG.

This module provides CPU-optimized embedding extraction with:
- Multiple pooling strategies (MEAN, CLS, LAST, MAX)
- L2 normalization for cosine similarity
- Batch embedding support
- HuggingFace Hub integration

Example:
    >>> from densecore.embedding import EmbeddingModel

    # Load from HuggingFace
    >>> model = EmbeddingModel.from_pretrained("BAAI/bge-small-en-v1.5")

    # Embed single text
    >>> embedding = model.embed("Hello, world!")
    >>> print(embedding.shape)  # (384,)

    # Embed batch
    >>> embeddings = model.embed(["Hello", "World", "Test"])
    >>> print(embeddings.shape)  # (3, 384)

    # Similarity search
    >>> scores = model.similarity("Query", ["Doc 1", "Doc 2", "Doc 3"])
"""

import ctypes
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding extraction.

    Args:
        pooling: Pooling strategy ("mean", "cls", "last", "max")
        normalize: Whether to L2 normalize embeddings
        max_length: Maximum sequence length
    """

    pooling: str = "mean"  # mean, cls, last, max
    normalize: bool = True
    max_length: int = 512


# Pooling strategy mapping
POOLING_MAP = {
    "mean": 0,
    "cls": 1,
    "last": 2,
    "max": 3,
}


class EmbeddingModel:
    """
    High-performance embedding model for RAG and semantic search.

    Optimized for CPU inference with SIMD-accelerated pooling and normalization.

    Args:
        model_path: Path to the GGUF model file
        config: Optional EmbeddingConfig for customization
        threads: Number of CPU threads (0 = auto)

    Example:
        >>> model = EmbeddingModel("./bge-small.gguf")
        >>> embedding = model.embed("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[EmbeddingConfig] = None,
        threads: int = 0,
        verbose: bool = True,
    ):
        self._model_path = model_path
        self._config = config or EmbeddingConfig()
        self._threads = threads
        self._verbose = verbose
        self._handle = None
        self._lib = None
        self._dimension = 0

        # Load library
        from .engine import _find_library, _preload_linux_runtime_dependencies

        lib_path = _find_library()
        _preload_linux_runtime_dependencies(lib_path)
        self._lib = ctypes.CDLL(lib_path)

        # Setup C functions
        self._setup_cfunctions()

        # Initialize engine
        main_path_bytes = model_path.encode("utf-8")
        self._handle = self._lib.InitEngine(main_path_bytes, None, threads)

        if not self._handle:
            raise RuntimeError(f"Failed to load embedding model: {model_path}")

        # Get embedding dimension
        self._dimension = self._lib.GetEmbeddingDimension(self._handle)
        if self._dimension <= 0:
            self._dimension = 768  # Default fallback

        if self._verbose:
            print(f"[EmbeddingModel] Loaded: {model_path}")
            print(f"[EmbeddingModel] Dimension: {self._dimension}")

    def _setup_cfunctions(self) -> None:
        """Setup C function signatures."""
        # InitEngine
        self._lib.InitEngine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.InitEngine.restype = ctypes.c_void_p

        # SubmitEmbeddingRequestEx (single embedding)
        self._lib.SubmitEmbeddingRequestEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._lib.SubmitEmbeddingRequestEx.restype = ctypes.c_int

        # SubmitBatchEmbeddingRequest (batch embedding - avoids Python loop)
        self._BATCH_EMBEDDING_CALLBACK = ctypes.CFUNCTYPE(
            None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
        )
        try:
            self._lib.SubmitBatchEmbeddingRequest.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.POINTER(ctypes.c_char_p),  # prompts array
                ctypes.c_int,  # num_prompts
                ctypes.c_int,  # pooling_type
                ctypes.c_int,  # normalize
                self._BATCH_EMBEDDING_CALLBACK,  # callback
                ctypes.c_void_p,  # user_data
            ]
            self._lib.SubmitBatchEmbeddingRequest.restype = ctypes.c_int
            self._has_batch_embedding = True
        except AttributeError:
            self._has_batch_embedding = False

        # GetEmbeddingDimension
        self._lib.GetEmbeddingDimension.argtypes = [ctypes.c_void_p]
        self._lib.GetEmbeddingDimension.restype = ctypes.c_int

        # FreeEngine
        self._lib.FreeEngine.argtypes = [ctypes.c_void_p]
        self._lib.FreeEngine.restype = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def pooling(self) -> str:
        """Get the current pooling strategy."""
        return self._config.pooling

    @pooling.setter
    def pooling(self, value: str) -> None:
        """Set the pooling strategy."""
        if value not in POOLING_MAP:
            raise ValueError(f"Invalid pooling: {value}. Must be one of {list(POOLING_MAP.keys())}")
        self._config.pooling = value

    def embed(
        self,
        texts: Union[str, list[str]],
        pooling: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Embed text(s) into dense vectors.

        Args:
            texts: Single text or list of texts to embed
            pooling: Override pooling strategy (mean, cls, last, max)
            normalize: Override L2 normalization setting

        Returns:
            numpy array of shape (dim,) for single text or (n, dim) for list

        Example:
            >>> embedding = model.embed("Hello, world!")
            >>> embeddings = model.embed(["Hello", "World"])
        """
        # Handle single text
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # Get config
        pool_type = POOLING_MAP.get(pooling or self._config.pooling, 0)
        norm = normalize if normalize is not None else self._config.normalize

        # Embed each text
        embeddings = []
        for text in texts:
            emb = self._embed_single(text, pool_type, norm)
            embeddings.append(emb)

        result = np.array(embeddings, dtype=np.float32)
        return result[0] if single else result

    def _embed_single(self, text: str, pooling_type: int, normalize: bool) -> np.ndarray:
        """Embed a single text."""
        import queue

        result_queue = queue.Queue()

        # Callback to receive embedding
        def callback(emb_ptr, size, user_data):
            if emb_ptr and size > 0:
                # Copy embedding data
                emb = np.ctypeslib.as_array(emb_ptr, shape=(size,)).copy()
                result_queue.put(emb)
            else:
                result_queue.put(np.zeros(self._dimension, dtype=np.float32))

        # Create callback
        callback_type = ctypes.CFUNCTYPE(
            None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
        )
        c_callback = callback_type(callback)

        # Submit request
        ret = self._lib.SubmitEmbeddingRequestEx(
            self._handle,
            text.encode("utf-8"),
            pooling_type,
            1 if normalize else 0,
            c_callback,
            None,
        )

        if ret < 0:
            raise RuntimeError(f"Embedding request failed: {ret}")

        # Wait for result (with timeout)
        try:
            embedding = result_queue.get(timeout=30.0)
            return embedding
        except queue.Empty:
            raise RuntimeError("Embedding request timed out")

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a large batch of texts efficiently using C++ batch API.

        This method avoids Python loop overhead by passing the entire batch
        to the C++ engine for processing.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (for memory management)
            show_progress: Show progress bar (requires tqdm)

        Returns:
            numpy array of shape (n, dim)
        """

        # Get config
        pool_type = POOLING_MAP.get(self._config.pooling, 0)
        norm = 1 if self._config.normalize else 0

        # Check if C++ batch API is available
        if hasattr(self, "_has_batch_embedding") and self._has_batch_embedding:
            return self._embed_batch_native(texts, pool_type, norm, show_progress)
        else:
            # Fallback to Python loop (legacy behavior)
            return self._embed_batch_loop(texts, batch_size, show_progress)

    def _embed_batch_native(
        self,
        texts: list[str],
        pooling_type: int,
        normalize: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Embed batch using native C++ SubmitBatchEmbeddingRequest."""
        import threading

        n_texts = len(texts)
        results: list[Optional[np.ndarray]] = [None] * n_texts
        result_idx = [0]  # Mutable counter for callback
        result_lock = threading.Lock()
        done_event = threading.Event()

        # Callback to collect results in order
        def batch_callback(emb_ptr, size, user_data):
            with result_lock:
                idx = result_idx[0]
                if idx < n_texts and emb_ptr and size > 0:
                    results[idx] = np.ctypeslib.as_array(emb_ptr, shape=(size,)).copy()
                else:
                    results[idx] = np.zeros(self._dimension, dtype=np.float32)
                result_idx[0] += 1
                if result_idx[0] >= n_texts:
                    done_event.set()

        # Create ctypes callback
        c_callback = self._BATCH_EMBEDDING_CALLBACK(batch_callback)

        # Prepare ctypes array of strings
        c_texts = (ctypes.c_char_p * n_texts)(*[t.encode("utf-8") for t in texts])

        # Submit batch to C++
        ret = self._lib.SubmitBatchEmbeddingRequest(
            self._handle, c_texts, n_texts, pooling_type, normalize, c_callback, None
        )

        if ret < 0:
            raise RuntimeError(f"Batch embedding request failed: {ret}")

        # Wait for all results (with timeout)
        if not done_event.wait(timeout=60.0 * (n_texts / 10 + 1)):
            raise RuntimeError("Batch embedding request timed out")

        return np.array(
            [r if r is not None else np.zeros(self._dimension) for r in results], dtype=np.float32
        )

    def _embed_batch_loop(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Fallback: embed batch using Python loop (legacy)."""
        embeddings = []

        # Optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(0, len(texts), batch_size), desc="Embedding")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embed(batch)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def similarity(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> Union[np.ndarray, list[tuple]]:
        """
        Compute similarity between query and documents.

        Args:
            query: Query text
            documents: List of document texts
            top_k: If provided, return top-k most similar (idx, score) pairs

        Returns:
            Similarity scores array, or list of (index, score) if top_k specified
        """
        # Embed query and documents
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)

        # Compute cosine similarity (embeddings are already normalized)
        scores = np.dot(doc_embs, query_emb)

        if top_k:
            # Return top-k
            indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in indices]

        return scores

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
        threads: int = 0,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> "EmbeddingModel":
        """
        Load an embedding model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
            filename: Specific file to download (auto-selected if None)
            config: Embedding configuration
            threads: CPU threads (0 = auto)
            cache_dir: Cache directory
            token: HuggingFace token

        Returns:
            EmbeddingModel instance

        Example:
            >>> model = EmbeddingModel.from_pretrained("BAAI/bge-small-en-v1.5")
        """
        from .hub import download_model

        model_path = download_model(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            token=token,
        )

        return cls(model_path, config=config, threads=threads)

    def close(self) -> None:
        """Close the model and release resources."""
        if self._handle and self._lib:
            self._lib.FreeEngine(self._handle)
            self._handle = None

    def __enter__(self) -> "EmbeddingModel":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"EmbeddingModel(dim={self._dimension}, pooling='{self._config.pooling}')"


# Convenience function
def embed(
    texts: Union[str, list[str]],
    model: Optional[EmbeddingModel] = None,
    model_path: Optional[str] = None,
) -> np.ndarray:
    """
    Quick embedding function.

    Args:
        texts: Text(s) to embed
        model: Pre-loaded EmbeddingModel
        model_path: Path to model (creates temporary model)

    Returns:
        Embedding array
    """
    if model:
        return model.embed(texts)

    if model_path:
        with EmbeddingModel(model_path, verbose=False) as m:
            return m.embed(texts)

    raise ValueError("Either model or model_path must be provided")
