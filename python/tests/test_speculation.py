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


def test_runtime_plan_preserves_native_selection_reason_and_reservations():
    from densecore.speculation import NativeRuntimePlan, runtime_plan

    class Library:
        def DenseCoreGetRuntimePlan(self, handle, out):
            assert handle == 42
            value = ctypes.cast(out, ctypes.POINTER(NativeRuntimePlan)).contents
            assert value.struct_size == ctypes.sizeof(NativeRuntimePlan)
            assert value.version == 1
            value.effective_context = 4096
            value.effective_concurrency = 2
            value.mtp_bytes = 1024
            value.spec_reason = b"explicit reference"
            return 0

    result = runtime_plan(Library(), 42)
    assert result["effective_context"] == 4096
    assert result["effective_concurrency"] == 2
    assert result["mtp_bytes"] == 1024
    assert result["spec_reason"] == "explicit reference"


def test_versioned_initialization_preserves_legacy_default_kv_policy():
    class Library:
        def InitEngineWithOptions(self, path, reserved, threads, numa, pinning, kv_type, options):
            assert kv_type == -1
            return 42

    assert initialize_native(Library(), b"model.gguf", 2, 0, init_options()) == 42
