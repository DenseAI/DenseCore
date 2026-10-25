"""Synthetic text flush compatibility; no library/model is constructed."""
import asyncio
import threading

import pytest

from densecore.engine import DenseCore


def _engine(monkeypatch, suffix="<ST"):
    engine = DenseCore.__new__(DenseCore)
    engine._handle = None
    engine._lock = threading.Lock()
    engine._req_id_counter = 0
    engine._requests = {}
    engine._active_ctypes_refs = {}
    engine._request_parity_mode = {}

    class Tokenizer:
        def encode(self, text, **_kwargs):
            return [42] if text == "Hi" + suffix else [1]

    engine.tokenizer = Tokenizer()
    monkeypatch.setattr(engine, "_normalize_prompt", lambda prompt, _ids=None: prompt)
    monkeypatch.setattr(
        engine,
        "_resolve_generation_params",
        lambda _config, kwargs, **_rest: {"return_dict": kwargs.get("return_dict_in_generate", False)},
    )

    def submit(req_id, _prompt, _params):
        engine._global_callback(b"Hi", 0, req_id)
        # The active Python ABI is a string callback: token_id=-1 is not exposed.
        engine._global_callback(suffix.encode("utf-8"), 0, req_id)
        assert req_id in engine._requests  # Flush is content, not terminal/EOS.
        engine._global_callback(b"", 1, req_id)
        return 1

    monkeypatch.setattr(engine, "_submit_request", submit)
    return engine


@pytest.mark.parametrize("suffix", ["<ST", "Error:", "끝"])
def test_legacy_python_generate_and_stream_keep_nonterminal_stop_prefix(monkeypatch, suffix):
    engine = _engine(monkeypatch, suffix)
    assert engine.generate("prompt") == "Hi" + suffix
    assert list(engine.stream("prompt")) == ["Hi", suffix]
    assert engine._requests == {}
    assert engine._active_ctypes_refs == {}


def test_python_return_dict_counts_retokenized_text_not_callback_chunks(monkeypatch):
    engine = _engine(monkeypatch)
    output = engine.generate("prompt", return_dict_in_generate=True)
    assert output.text == "Hi<ST"
    assert output.usage["completion_tokens"] == 1
    # This is wrapper retokenization, not native generation-completion metadata.
    assert output.sequences == [[1, 42]]


def test_async_python_generate_and_stream_keep_nonterminal_stop_prefix(monkeypatch):
    engine = _engine(monkeypatch)

    async def check():
        assert await engine.generate_async("prompt") == "Hi<ST"
        assert [token async for token in engine.stream_async("prompt")] == ["Hi", "<ST"]

    asyncio.run(check())
    assert engine._requests == {}
