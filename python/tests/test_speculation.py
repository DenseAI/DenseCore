import ctypes

import pytest

from densecore.speculation import NativeInitOptions, init_options, initialize_native


@pytest.mark.parametrize("mode,value", [("auto", 0), ("none", 1), ("mtp", 2), (" AUTO ", 0)])
def test_versioned_options(mode, value):
    result = init_options(mode, 3)
    assert result.struct_size == ctypes.sizeof(NativeInitOptions)
    assert result.version == 1
    assert result.spec_type == value
    assert result.spec_draft_n_max == 3


@pytest.mark.parametrize("mode,limit", [("draft", 2), (None, 2), ("auto", 0), ("mtp", 4), ("none", True), ("auto", 2.0)])
def test_reject_invalid_options(mode, limit):
    with pytest.raises(ValueError):
        init_options(mode, limit)


def test_forward_versioned_options_and_kv_type():
    class Library:
        def InitEngineWithOptions(self, path, reserved, threads, numa, pinning, kv_type, options):
            received = ctypes.cast(options, ctypes.POINTER(NativeInitOptions)).contents
            assert (path, reserved, threads, numa, pinning, kv_type) == (b"model.gguf", None, 4, -1, 0, 2)
            assert (received.spec_type, received.spec_draft_n_max) == (1, 3)
            return 42

    assert initialize_native(Library(), b"model.gguf", 4, 2, init_options("none", 3)) == 42


def test_older_library_never_silently_accepts_explicit_mtp():
    class Library:
        def InitEngine(self, *args):
            raise AssertionError("must reject before allocating")

    with pytest.raises(RuntimeError, match="explicit mtp"):
        initialize_native(Library(), b"model", 2, 0, init_options("mtp"))


def test_older_library_rejects_unavailable_explicit_kv_type():
    with pytest.raises(RuntimeError, match="KV cache type"):
        initialize_native(object(), b"model", 2, 2, init_options())


def test_model_config_roundtrip_preserves_speculation():
    from densecore.config import ModelConfig

    config = ModelConfig(spec_type="none", spec_draft_n_max=3)
    restored = ModelConfig.from_dict(config.to_dict())
    assert (restored.spec_type, restored.spec_draft_n_max) == ("none", 3)
    with pytest.raises(ValueError, match="spec_type"):
        ModelConfig(spec_type="draft")
