import numpy as np
import pytest

from densecore.embedding import EmbeddingConfig, EmbeddingModel


def fake_model():
    model = EmbeddingModel.__new__(EmbeddingModel)
    model._config = EmbeddingConfig()
    model._dimension = 2
    model._handle = None
    return model


def test_omitted_pooling_reaches_native_as_auto_and_explicit_last_is_preserved():
    model = fake_model()
    seen = []
    model._embed_single = lambda text, pooling, normalize: seen.append(pooling) or np.zeros(2)
    model.embed("a")
    model.embed("b", pooling="last")
    model.embed("c", pooling="mean")
    assert seen == [-1, 2, 0]


def test_batch_default_also_reaches_native_as_auto():
    model = fake_model()
    model._has_batch_embedding = True
    seen = []
    model._embed_batch_native = lambda texts, pooling, normalize, progress: seen.append(pooling) or np.zeros((len(texts), 2))
    model.embed_batch(["a", "b"])
    assert seen == [-1]


def test_invalid_explicit_pooling_fails_before_submission():
    model = fake_model()
    with pytest.raises(ValueError, match="Invalid pooling"):
        model.embed("a", pooling="typo")


def test_native_worker_error_is_reported_instead_of_zero_embedding():
    model = fake_model()

    class FailedNative:
        def SubmitEmbeddingRequestEx(self, handle, text, pooling, normalize, callback, user_data):
            callback(None, 0, None)
            return 1

    model._lib = FailedNative()
    with pytest.raises(RuntimeError, match="worker failed"):
        model._embed_single("text", -1, True)


def test_admission_rejection_raises_without_waiting_for_a_callback():
    model = fake_model()

    class RejectedNative:
        def SubmitEmbeddingRequestEx(self, *args):
            return -3

    model._lib = RejectedNative()
    with pytest.raises(RuntimeError, match="Embedding request failed: -3"):
        model._embed_single("text", -1, True)


def test_rejected_batch_raises_without_waiting_for_callbacks():
    model = fake_model()
    model._BATCH_EMBEDDING_CALLBACK = lambda callback: callback

    class RejectedNative:
        def SubmitBatchEmbeddingRequest(self, *args):
            return -3

    model._lib = RejectedNative()
    with pytest.raises(RuntimeError, match="Batch embedding request failed: -3"):
        model._embed_batch_native(["a", "b"], -1, True, False)


def test_partial_batch_admission_failure_finishes_before_raising():
    import ctypes

    model = fake_model()
    model._BATCH_EMBEDDING_CALLBACK = lambda callback: callback

    class PartialNative:
        def SubmitBatchEmbeddingRequest(self, handle, texts, count, pooling, norm, callback, user):
            vector = np.array([1.0, 0.0], dtype=np.float32)
            callback(vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 2, None)
            callback(None, -3, None)
            return 1

    model._lib = PartialNative()
    with pytest.raises(RuntimeError, match="Batch embedding worker failed"):
        model._embed_batch_native(["a", "b"], -1, True, False)
