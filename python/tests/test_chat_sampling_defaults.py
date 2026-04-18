from densecore.engine import DenseCore


def test_chat_sampling_defaults_match_go_server_for_qwen35_no_thinking(monkeypatch):
    monkeypatch.setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")

    params = DenseCore._apply_chat_sampling_defaults(
        model_hint="/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        profile_family="qwen",
        kwargs={},
    )

    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.8
    assert params["top_k"] == 20
    assert params["repetition_penalty"] == 1.05


def test_chat_sampling_defaults_match_go_server_for_qwen35_thinking(monkeypatch):
    monkeypatch.setenv("DENSECORE_QWEN35_ENABLE_THINKING", "true")

    params = DenseCore._apply_chat_sampling_defaults(
        model_hint="/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        profile_family="qwen",
        kwargs={},
    )

    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["top_k"] == 20
    assert params["repetition_penalty"] == 1.05


def test_chat_sampling_defaults_preserve_explicit_kwargs():
    params = DenseCore._apply_chat_sampling_defaults(
        model_hint="/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        profile_family="qwen",
        kwargs={
            "temperature": 0.2,
            "top_p": 0.77,
            "top_k": 9,
            "repetition_penalty": 1.2,
        },
    )

    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.77
    assert params["top_k"] == 9
    assert params["repetition_penalty"] == 1.2


def test_chat_sampling_defaults_force_greedy_overrides_when_temperature_zero():
    params = DenseCore._apply_chat_sampling_defaults(
        model_hint="/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        profile_family="qwen",
        kwargs={"temperature": 0.0},
    )

    assert params["temperature"] == 0.0
    assert params["top_p"] == 1.0
    assert params["top_k"] == 1
    assert params["repetition_penalty"] == 1.0


def test_chat_sampling_defaults_match_go_server_for_gemma():
    params = DenseCore._apply_chat_sampling_defaults(
        model_hint="/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
        profile_family="gemma",
        kwargs={},
    )

    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["top_k"] == 64
    assert params["repetition_penalty"] == 1.0
