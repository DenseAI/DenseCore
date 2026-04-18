from densecore.chat_template import (
    format_chat_prompt,
    _profile_specs,
    qwen_thinking_enabled,
    resolve_prompt_profile,
)
from densecore.engine import DenseCore


def test_resolve_prompt_profile_uses_shared_families():
    qwen = resolve_prompt_profile("/tmp/Qwen3.5-2B-Q4_K_M.gguf")
    assert qwen.family == "qwen"
    assert qwen.kind == "chatml"

    gemma = resolve_prompt_profile("/tmp/gemma-4-E2B-it-Q4_K_M.gguf")
    assert gemma.family == "gemma"
    assert gemma.kind == "turn_tags"

    generic = resolve_prompt_profile("/tmp/llama-3.2-base.gguf")
    assert generic.family == "generic"


def test_resolve_prompt_profile_keeps_known_families_over_generic_override(tmp_path, monkeypatch):
    profile_path = tmp_path / "prompt_profiles.json"
    profile_path.write_text(
        """
        {"profiles":[{"family":"generic","kind":"generic_transcript","match_substrings":["qwen3-0.6b"],"roles":{"system":"System","user":"User","assistant":"Assistant"},"tags":{"open":"","close":""},"multimodal":{"image":"","video":"","audio":""},"default_system_prompt":"You are a helpful assistant."}]}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSECORE_PROMPT_PROFILE_PATH", str(profile_path))
    _profile_specs.cache_clear()

    profile = resolve_prompt_profile("/tmp/Qwen3-0.6B-Q4_K_M.unsloth.gguf")
    assert profile.family == "qwen"
    assert profile.kind == "chatml"


def test_format_chat_prompt_qwen_matches_chatml_suffix(monkeypatch):
    monkeypatch.setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
    prompt = format_chat_prompt(
        "/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        [{"role": "user", "content": "안녕?"}],
    )

    assert "<|im_start|>user\n안녕? /no_think<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_qwen_thinking_enabled_defaults_are_model_specific():
    assert qwen_thinking_enabled("/tmp/Qwen3.5-2B-Q4_K_M.gguf", None) is False
    assert qwen_thinking_enabled("/tmp/Qwen3-0.6B-Q4_K_M.gguf", None) is True


def test_format_chat_prompt_gemma_uses_turn_tags():
    prompt = format_chat_prompt(
        "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
        [{"role": "user", "content": "안녕?"}],
    )

    assert prompt == "<bos><|turn>user\n안녕?<turn|>\n<|turn>model\n"


def test_gemma_chat_passthrough_helper_prefers_raw_single_turn():
    assert DenseCore._should_passthrough_raw_chat_prompt(
        "gemma",
        [{"role": "user", "content": "What is the capital of France?"}],
        [],
    )


def test_gemma_chat_passthrough_helper_rejects_history():
    assert not DenseCore._should_passthrough_raw_chat_prompt(
        "gemma",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        [],
    )


def test_format_chat_prompt_generic_transcript_fallback():
    prompt = format_chat_prompt(
        "mystery-model",
        [{"role": "user", "content": "Hello"}],
    )

    assert "System: You are a helpful assistant." in prompt
    assert "User: Hello" in prompt
    assert prompt.endswith("Assistant: ")
