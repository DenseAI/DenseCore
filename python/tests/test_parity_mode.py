import threading

from densecore.chat_template import resolve_prompt_profile
from densecore.engine import CALLBACK_TYPE, DenseCore


class _DummyTokenizer:
    def encode(self, *_args, **_kwargs):
        raise AssertionError("HF tokenizer should not be used for string prompts in parity path")


class _FakeLib:
    def __init__(self):
        self.last_call = None

    @staticmethod
    def _int_value(value):
        return int(value.value) if hasattr(value, "value") else int(value)

    def SubmitRequestWithSampling(
        self,
        handle,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        stop_sequences,
        json_mode,
        callback,
        req_id,
    ):
        self.last_call = {
            "api": "SubmitRequestWithSampling",
            "handle": handle,
            "prompt": prompt.decode("utf-8"),
            "max_tokens": max_tokens,
            "json_mode": self._int_value(json_mode),
        }
        return 1

    def SubmitRequestIdsWithSampling(
        self,
        handle,
        tokens,
        n_tokens,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        stop_sequences,
        json_mode,
        callback,
        req_id,
    ):
        self.last_call = {
            "api": "SubmitRequestIdsWithSampling",
            "handle": handle,
            "tokens": [int(tokens[index]) for index in range(n_tokens)],
            "max_tokens": max_tokens,
            "json_mode": self._int_value(json_mode),
        }
        return 1


def _make_engine(model_path="/tmp/model.gguf"):
    engine = DenseCore.__new__(DenseCore)
    engine._handle = object()
    engine._lock = threading.Lock()
    engine._active_ctypes_refs = {}
    engine._requests = {}
    engine._request_parity_mode = {}
    engine._c_callback = CALLBACK_TYPE(lambda _token, _finished, _req_id: None)
    engine._lib = _FakeLib()
    engine._has_sampling_api = True
    engine._has_lora_sampling_api = False
    engine._has_lora_format_api = False
    engine._has_tokenizer_metadata_api = False
    engine._has_chat_template_api = False
    engine._has_count_tokens_api = False
    engine._has_native_tokenizer_api = False
    engine._has_native_chat_render_api = False
    engine._model_path = model_path
    engine._hf_repo_id = ""
    engine.tokenizer = _DummyTokenizer()
    return engine


def test_resolve_prompt_profile_prefers_metadata_over_model_name():
    profile = resolve_prompt_profile(
        "/tmp/local-model.gguf",
        tokenizer_type="qwen35",
        chat_template="<|im_start|>user\n",
    )

    assert profile.family == "qwen"
    assert profile.kind == "chatml"


def test_generate_string_prompt_uses_raw_text_submit_path():
    engine = _make_engine()

    params = {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "json_mode": False,
        "stop_sequences": [],
        "parity_mode": False,
    }

    engine._submit_request(1, "Hello parity", params)

    assert engine._lib.last_call["api"] == "SubmitRequestWithSampling"
    assert engine._lib.last_call["prompt"] == "Hello parity"


def test_explicit_input_ids_preserve_ids_submit_path():
    engine = _make_engine()

    params = {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "json_mode": False,
        "stop_sequences": [],
        "parity_mode": True,
    }

    engine._submit_request(1, [11, 22, 33], params)

    assert engine._lib.last_call["api"] == "SubmitRequestIdsWithSampling"
    assert engine._lib.last_call["tokens"] == [11, 22, 33]


def test_submit_request_never_applies_python_qwen_priming():
    engine = _make_engine("/tmp/Qwen3-0.6B-Q4_K_M.gguf")

    params = {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "json_mode": False,
        "stop_sequences": [],
        "parity_mode": False,
    }

    prompt = "<|im_start|>assistant\n"
    engine._submit_request(1, prompt, params)

    assert engine._lib.last_call["prompt"] == prompt
    assert "Answer:" not in engine._lib.last_call["prompt"]


def test_parity_mode_callback_skips_cleanup():
    engine = _make_engine()
    captured = []
    engine._requests[7] = lambda token, finished: captured.append((token, finished))
    engine._request_parity_mode[7] = True

    engine._global_callback("Ġhello".encode("utf-8"), 1, 7)

    assert captured == [("Ġhello", True)]


def test_chat_parity_mode_uses_engine_metadata_and_skips_defaults(monkeypatch):
    engine = _make_engine("/tmp/local-model.gguf")
    engine._has_native_chat_render_api = True
    captured = {}

    def fake_generate(prompt, max_tokens, parity_mode=False, **kwargs):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        captured["parity_mode"] = parity_mode
        captured["kwargs"] = kwargs
        return "ok"

    engine.generate = fake_generate
    engine.render_chat_prompt = (
        lambda messages, enable_thinking=None, preserve_thinking=None, extra_system_messages=None: {
            "prompt": "<|im_start|>user\nHello /no_think<|im_end|>\n<|im_start|>assistant\n",
            "tokenizer_type": "qwen35",
            "chat_template": "<|im_start|>user\n",
        }
    )
    monkeypatch.setenv("DENSECORE_QWEN35_ENABLE_THINKING", "false")

    result = engine.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=4,
        parity_mode=True,
    )

    assert result == "ok"
    assert captured["parity_mode"] is True
    assert "temperature" not in captured["kwargs"]
    assert "<|im_start|>user\nHello /no_think<|im_end|>\n" in captured["prompt"]


def test_chat_parity_mode_disables_gemma_raw_passthrough():
    engine = _make_engine("/tmp/local-model.gguf")
    engine._has_native_chat_render_api = True
    captured = {}

    def fake_generate(prompt, max_tokens, parity_mode=False, **kwargs):
        captured["prompt"] = prompt
        return "ok"

    engine.generate = fake_generate
    engine.render_chat_prompt = (
        lambda messages, enable_thinking=None, preserve_thinking=None, extra_system_messages=None: {
            "prompt": "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
            "tokenizer_type": "gemma4",
            "chat_template": "<|turn>user\n",
        }
    )

    engine.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=4,
        parity_mode=True,
    )

    assert (
        captured["prompt"]
        == "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
    )
