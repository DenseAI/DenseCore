from densecore.chat_template import format_chat_prompt, resolve_prompt_profile


def test_resolve_prompt_profile_uses_shared_families():
    qwen = resolve_prompt_profile("/tmp/Qwen3.5-2B-Q4_K_M.gguf")
    assert qwen.family == "qwen"
    assert qwen.kind == "chatml"

    gemma = resolve_prompt_profile("/tmp/gemma-4-E2B-it-Q4_K_M.gguf")
    assert gemma.family == "gemma"
    assert gemma.kind == "turn_tags"

    generic = resolve_prompt_profile("/tmp/llama-3.2-base.gguf")
    assert generic.family == "generic"


def test_format_chat_prompt_qwen_matches_chatml_suffix(monkeypatch):
    monkeypatch.setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")
    prompt = format_chat_prompt(
        "/tmp/Qwen3.5-2B-Q4_K_M.gguf",
        [{"role": "user", "content": "안녕?"}],
    )

    assert "<|im_start|>user\n안녕?<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_format_chat_prompt_gemma_uses_turn_tags():
    prompt = format_chat_prompt(
        "/tmp/gemma-4-E2B-it-Q4_K_M.gguf",
        [{"role": "user", "content": "안녕?"}],
    )

    assert prompt == "<|turn>user\n안녕?<turn|>\n<|turn>model\n"


def test_format_chat_prompt_generic_transcript_fallback():
    prompt = format_chat_prompt(
        "mystery-model",
        [{"role": "user", "content": "Hello"}],
    )

    assert "System: You are a helpful assistant." in prompt
    assert "User: Hello" in prompt
    assert prompt.endswith("Assistant: ")
