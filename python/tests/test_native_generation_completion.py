"""Native generation metadata/cancellation contracts, using no models or C++ build."""
import concurrent.futures
import gc
import threading
import time
import weakref

import pytest

import densecore.engine as engine_module
from densecore.engine import DenseCore, DenseCoreError, DenseCoreRuntimeError


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class NativeLibrary:
    def __init__(self, metadata=True):
        self.callback = None
        self.engine = None
        self.pending = {}
        self.cancelled = []
        self.free_calls = 0
        self.cancel_result = 0
        self.cancel_finishes = True
        self.free_entered = threading.Event()
        self.release_free = None
        if metadata:
            self.DenseCoreSetGenerationCompletionCallback = Function(self.install)

    def install(self, _handle, callback):
        # C keeps only a function address; engine must own the Python CFUNCTYPE.
        self.callback = weakref.ref(callback)
        return 0

    def metadata(self, user_data, count, reason):
        if self.callback:
            callback = self.callback()
            assert callback is not None, "CFUNCTYPE released before native producer joined"
            callback(count, reason, user_data)

    def finish(self, native_id, count=1, reason=1, text=b""):
        user_data = self.pending.pop(native_id)
        self.metadata(user_data, count, reason)
        self.engine._global_callback(text, 1, user_data)

    def CancelRequest(self, _handle, native_id):
        native_id = native_id.value
        self.cancelled.append(native_id)
        if self.cancel_result >= 0 and self.cancel_finishes:
            self.finish(native_id, reason=3, text=b"Error: request canceled")
        return self.cancel_result

    def FreeEngine(self, _handle):
        self.free_calls += 1
        self.free_entered.set()
        if self.release_free:
            assert self.release_free.wait(2)
        for native_id in list(self.pending):
            self.finish(native_id, reason=3, text=b"Error: engine shutdown")
        if self.callback:
            assert self.callback() is not None


def make_engine(monkeypatch, metadata=True):
    engine = DenseCore.__new__(DenseCore)
    engine._handle = 123
    engine._closed = False
    engine._closing = False
    engine._lock = threading.Lock()
    engine._native_call_lock = threading.RLock()
    engine._requests = {}
    engine._active_ctypes_refs = {}
    engine._request_parity_mode = {}
    engine._request_completions = {}
    engine._req_id_counter = 0
    engine._has_cancel_api = True
    engine._verbose = False
    library = NativeLibrary(metadata)
    library.engine = engine
    engine._lib = library
    engine._install_generation_completion_callback()

    class Tokenizer:
        def encode(self, text, **_kwargs):
            return [10] if text == "prompt" else [20, 21, 22]

    engine.tokenizer = Tokenizer()
    monkeypatch.setattr(engine, "_normalize_prompt", lambda prompt, _ids=None: prompt)
    monkeypatch.setattr(engine, "_resolve_generation_params", lambda _config, kwargs, **_rest: {
        "return_dict": kwargs.get("return_dict_in_generate", False)
    })
    return engine, library


def tracked_submit(engine, library, action=None):
    def submit(user_data, prompt, _params):
        native_id = 700 + user_data
        library.pending[native_id] = user_data
        with engine._lock:
            engine._active_ctypes_refs[user_data] = [object()]
        if action:
            action(user_data, native_id, prompt)
        return native_id
    return submit


def empty_tracking(engine):
    assert engine._requests == {}
    assert engine._active_ctypes_refs == {}
    assert engine._request_parity_mode == {}
    assert engine._request_completions == {}


@pytest.mark.parametrize("reason,expected", [(1, "stop"), (2, "length")])
def test_native_usage_and_finish_survive_terminal_before_submit_returns(monkeypatch, reason, expected):
    engine, library = make_engine(monkeypatch)
    def output(user_data, native_id, _prompt):
        engine._global_callback(b"Hi", 0, user_data)
        engine._global_callback(b"<ST", 0, user_data)  # Synthetic text, no sampled token.
        library.finish(native_id, count=1, reason=reason)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, output))
    result = engine.generate("prompt", return_dict_in_generate=True)
    assert result.text == "Hi<ST"
    assert result.usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    assert result.finish_reason == expected
    assert result.sequences == [[10, 20, 21, 22]]  # Reconstruction is explicitly separate.
    empty_tracking(engine)
    engine.close()


@pytest.mark.parametrize("count,reason", [(0, 3), (1, 99), (-1, 1)])
def test_native_error_or_invalid_metadata_cannot_be_successful_empty_terminal(monkeypatch, count, reason):
    engine, library = make_engine(monkeypatch)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda _req, native, _prompt: library.finish(native, count, reason)
    ))
    with pytest.raises(DenseCoreRuntimeError, match="native generation"):
        engine.generate("prompt", return_dict_in_generate=True)
    empty_tracking(engine)
    engine.close()


def test_current_library_missing_metadata_fails_and_old_library_fallback_is_preserved(monkeypatch):
    for metadata in (True, False):
        engine, library = make_engine(monkeypatch, metadata)
        def output(user_data, native, _prompt):
            engine._global_callback(b"legacy", 0, user_data)
            library.pending.pop(native)
            engine._global_callback(b"", 1, user_data)
        monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, output))
        if metadata:
            with pytest.raises(DenseCoreRuntimeError, match="metadata is missing"):
                engine.generate("prompt", return_dict_in_generate=True)
        else:
            result = engine.generate("prompt", return_dict_in_generate=True)
            assert result.usage["completion_tokens"] == 3
            assert result.finish_reason == "stop"  # Legacy fallback, not a native reason.
        empty_tracking(engine)
        engine.close()


def test_interleaved_requests_never_share_completion_metadata(monkeypatch):
    engine, library = make_engine(monkeypatch)
    submitted = __import__("queue").Queue()
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda req, native, prompt: submitted.put((req, native, prompt))
    ))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        first = pool.submit(engine.generate, "first", return_dict_in_generate=True)
        second = pool.submit(engine.generate, "second", return_dict_in_generate=True)
        requests = [submitted.get(timeout=2), submitted.get(timeout=2)]
        for req, native, prompt in reversed(requests):
            engine._global_callback(prompt.encode(), 0, req)
            library.finish(native, 7 if prompt == "second" else 1, 2 if prompt == "second" else 1)
        a, b = first.result(2), second.result(2)
    assert (a.text, a.usage["completion_tokens"], a.finish_reason) == ("first", 1, "stop")
    assert (b.text, b.usage["completion_tokens"], b.finish_reason) == ("second", 7, "length")
    empty_tracking(engine)
    engine.close()


def test_unknown_late_userdata_and_handler_exception_cannot_leak_metadata(monkeypatch):
    engine, library = make_engine(monkeypatch)
    library.metadata(999, 17, 2)
    def broken(_text, _finished):
        raise RuntimeError("handler failed")
    req = engine._register_request(broken)
    with engine._lock:
        engine._active_ctypes_refs[req] = [object()]
    library.metadata(req, 1, 1)
    with pytest.warns(UserWarning, match="handler failed"):
        engine._global_callback(b"", 1, req)
    library.metadata(req, 99, 2)
    engine._global_callback(b"", 1, req)
    empty_tracking(engine)
    engine.close()


@pytest.mark.parametrize("stream", [False, True])
def test_cancel_uses_actual_native_id_and_keeps_waiter_until_real_terminal(monkeypatch, stream):
    engine, library = make_engine(monkeypatch)
    library.cancel_finishes = False
    submitted = threading.Event()
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda _req, _native, _prompt: submitted.set()
    ))
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        work = pool.submit(lambda: list(engine.stream("prompt")) if stream else engine.generate("prompt"))
        assert submitted.wait(2)
        assert engine.stop_generation(1)
        # Cancellation may have been deferred until submission returns.
        deadline = time.monotonic() + 2
        while not library.cancelled and time.monotonic() < deadline:
            time.sleep(.001)
        assert library.cancelled == [701]
        assert 1 in engine._requests
        assert 1 in engine._active_ctypes_refs
        assert not work.done()
        library.finish(701, reason=3, text=b"Error: request canceled")
        with pytest.raises(DenseCoreRuntimeError, match="canceled"):
            work.result(2)
    empty_tracking(engine)
    engine.close()


def test_cancel_before_submit_returns_is_deferred_to_returned_native_id(monkeypatch):
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def in_native(_req, _native, _prompt):
        entered.set()
        assert release.wait(2)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, in_native))
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        work = pool.submit(engine.generate, "prompt")
        assert entered.wait(2)
        assert engine.stop_generation(1)
        assert library.cancelled == []
        release.set()
        with pytest.raises(DenseCoreRuntimeError, match="canceled"):
            work.result(2)
    assert library.cancelled == [701]
    empty_tracking(engine)
    engine.close()


def test_failed_cancel_keeps_tracking_and_never_cancels_another_request(monkeypatch):
    engine, library = make_engine(monkeypatch)
    events = {1: [], 2: []}
    a = engine._register_request(lambda text, finished: events[1].append((text, finished)))
    b = engine._register_request(lambda text, finished: events[2].append((text, finished)))
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library))
    assert engine._submit_request(a, "a", {}) == 701
    assert engine._submit_request(b, "b", {}) == 702
    library.cancel_result = -9
    assert not engine.stop_generation(b)
    assert a in engine._requests and b in engine._requests
    assert library.cancelled == [702]
    library.cancel_result = 0
    assert engine.stop_generation(b)
    assert a in engine._requests and b not in engine._requests
    assert events[1] == [] and "canceled" in events[2][0][0]
    library.finish(701)
    empty_tracking(engine)
    engine.close()


def test_close_keeps_callback_and_request_refs_until_native_join(monkeypatch):
    engine, library = make_engine(monkeypatch)
    library.release_free = threading.Event()
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library))
    req = engine._register_request(lambda _text, _finished: None)
    engine._submit_request(req, "prompt", {})
    callback = weakref.ref(engine._c_generation_completion_callback)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        closed = pool.submit(engine.close)
        assert library.free_entered.wait(2)
        again = pool.submit(engine.close)
        gc.collect()
        assert callback() is not None
        assert req in engine._requests and req in engine._active_ctypes_refs
        assert not closed.done() and not again.done()
        library.release_free.set()
        closed.result(2)
        again.result(2)
    assert library.free_calls == 1 and engine._closed
    empty_tracking(engine)


@pytest.mark.parametrize("raises", [False, True])
def test_submit_failure_retires_all_completion_tracking(monkeypatch, raises):
    engine, library = make_engine(monkeypatch)
    def failed(_req, _prompt, _params):
        if raises:
            raise RuntimeError("submit failed")
        return -3
    monkeypatch.setattr(engine, "_submit_request_impl", failed)
    with pytest.raises((DenseCoreError, RuntimeError)):
        engine.generate("prompt", return_dict_in_generate=True)
    empty_tracking(engine)
    engine.close()


def test_present_metadata_setter_failure_frees_initialized_handle(monkeypatch):
    library = NativeLibrary()
    library.DenseCoreSetGenerationCompletionCallback = Function(lambda _handle, _callback: -3)
    monkeypatch.setattr(engine_module, "get_library", lambda: library)
    monkeypatch.setattr(engine_module, "initialize_native", lambda *_args: 123)
    monkeypatch.setattr(DenseCore, "_setup_cfunctions", lambda _self: None)
    with pytest.raises(DenseCoreError, match="registering generation completion"):
        DenseCore("fake.gguf")
    assert library.free_calls == 1

@pytest.mark.parametrize("stream", [False, True])
def test_async_native_empty_error_terminates_and_cleans_request(monkeypatch, stream):
    import asyncio
    engine, library = make_engine(monkeypatch)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda _req, native, _prompt: library.finish(native, reason=3)
    ))
    async def run():
        with pytest.raises(DenseCoreRuntimeError, match="native generation failed"):
            if stream:
                _ = [piece async for piece in engine.stream_async("prompt")]
            else:
                await engine.generate_async("prompt")
    asyncio.run(run())
    empty_tracking(engine)
    engine.close()


def test_cancel_without_native_api_preserves_normal_completion_tracking(monkeypatch):
    engine, library = make_engine(monkeypatch, metadata=False)
    engine._has_cancel_api = False
    events = []
    req = engine._register_request(lambda text, finished: events.append((text, finished)))
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library))
    engine._submit_request(req, "prompt", {})
    assert not engine.stop_generation(req)
    assert req in engine._requests and req in engine._active_ctypes_refs
    library.finish(701)
    assert events == [("", True)]
    empty_tracking(engine)
    engine.close()


def test_cancel_then_close_waits_for_submit_and_keeps_terminal_owner(monkeypatch):
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def in_native(_req, _native, _prompt):
        entered.set()
        assert release.wait(2)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, in_native))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        request = pool.submit(engine.generate, "prompt")
        assert entered.wait(2)
        assert engine.stop_generation(1)  # No native ID has returned yet.
        closed = pool.submit(engine.close)
        deadline = time.monotonic() + 2
        while not engine._closing and time.monotonic() < deadline:
            time.sleep(.001)
        assert engine._closing
        assert not library.free_entered.is_set()  # Submit still borrows native handle.
        release.set()
        closed.result(2)
        with pytest.raises(DenseCoreRuntimeError, match="shutdown"):
            request.result(2)
    assert library.free_calls == 1
    assert library.cancelled == []  # Native close owns cancellation now.
    empty_tracking(engine)


@pytest.mark.parametrize("raises", [False, True])
def test_cancel_before_submit_failure_does_not_cancel_an_unrelated_native_id(monkeypatch, raises):
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def failed(_req, _prompt, _params):
        entered.set()
        assert release.wait(2)
        if raises:
            raise RuntimeError("submit failed")
        return -3
    monkeypatch.setattr(engine, "_submit_request_impl", failed)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        request = pool.submit(engine.generate, "prompt")
        assert entered.wait(2)
        assert engine.stop_generation(1)
        release.set()
        with pytest.raises((DenseCoreError, RuntimeError)):
            request.result(2)
    assert library.cancelled == []
    empty_tracking(engine)
    engine.close()


def test_terminal_owner_rejects_duplicate_and_late_callbacks_while_handler_runs(monkeypatch):
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    events = []
    def handler(text, finished):
        events.append((text, finished))
        entered.set()
        assert release.wait(2)
    req = engine._register_request(handler)
    state = engine._request_completions[req]
    library.metadata(req, 1, 1)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        terminal = pool.submit(engine._global_callback, b"", 1, req)
        assert entered.wait(2)
        engine._global_callback(b"duplicate", 1, req)
        engine._global_callback(b"late content", 0, req)
        library.metadata(req, 99, 2)
        assert state.generated_tokens == 1 and state.finish_reason == "stop"
        assert events == [("", True)]
        release.set()
        terminal.result(2)
    empty_tracking(engine)
    engine.close()


@pytest.mark.parametrize("during_submit", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_async_task_cancel_maps_native_id_and_never_sets_cancelled_future(monkeypatch, during_submit, streaming):
    import asyncio
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def in_native(_req, _native, _prompt):
        entered.set()
        if during_submit:
            assert release.wait(2)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, in_native))
    errors = []
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        iterator = engine.stream_async("prompt") if streaming else None
        async def execute():
            return await anext(iterator) if streaming else await engine.generate_async("prompt")
        task = asyncio.create_task(execute())
        for _ in range(2000):
            if entered.is_set() and (during_submit or engine._request_completions[1].native_request_id is not None):
                break
            await asyncio.sleep(.001)
        else:
            pytest.fail("async submission did not enter")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if during_submit:
            assert library.cancelled == []
        release.set()
        for _ in range(2000):
            if not engine._requests:
                break
            await asyncio.sleep(.001)
        assert library.cancelled == [701]
        await asyncio.sleep(0)  # Execute the scheduled done-guarded completion.
        gc.collect()
        await asyncio.sleep(0)
    asyncio.run(run())
    assert errors == []
    empty_tracking(engine)
    engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_abandoned_stream_cancels_only_its_actual_native_request(monkeypatch, asynchronous):
    import asyncio
    engine, library = make_engine(monkeypatch)
    def emit(req, _native, _prompt):
        engine._global_callback(b"first", 0, req)
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, emit))
    # Advance the Python/native IDs independently and leave a peer owned.
    peer = engine._register_request(lambda _text, _finished: None)
    assert engine._submit_request(peer, "peer", {}) == 701
    if asynchronous:
        async def run():
            stream = engine.stream_async("prompt")
            assert await anext(stream) == "first"
            await stream.aclose()
            await asyncio.sleep(0)
        asyncio.run(run())
    else:
        stream = engine.stream("prompt")
        assert next(stream) == "first"
        stream.close()
    assert library.cancelled == [702]
    assert peer in engine._requests
    library.finish(701)
    empty_tracking(engine)
    engine.close()


def test_failed_native_close_quarantines_refs_without_retry_or_stranding_waiter(monkeypatch):
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    failure = RuntimeError("FreeEngine failed before confirming join")
    attempts = []
    def failed_free(_handle):
        attempts.append(_handle)
        entered.set()
        assert release.wait(2)
        raise failure
    monkeypatch.setattr(library, "FreeEngine", failed_free)
    submitted = threading.Event()
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda _req, _native, _prompt: submitted.set()
    ))
    try:
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            request = pool.submit(engine.generate, "prompt")
            assert submitted.wait(2)
            closing = pool.submit(engine.close)
            assert entered.wait(2)
            again = pool.submit(engine.close)
            release.set()
            for closed in (closing, again):
                with pytest.raises(RuntimeError) as error:
                    closed.result(2)
                assert error.value is failure
            assert not engine._closed and engine._handle is None
            assert engine._quarantined_handle == 123
            assert engine_module._failed_close_owners[id(engine)] is engine
            assert 1 in engine._requests and 1 in engine._active_ctypes_refs
            assert not request.done()
            with pytest.raises(RuntimeError) as error:
                engine.close()
            assert error.value is failure
            with pytest.raises(engine_module.EngineNotInitializedError):
                engine.stop_generation(1)
            with pytest.raises(RuntimeError):
                engine._lib.FreeEngine(123)  # Proxy rejects all native entry points.
            assert attempts == [123] and library.cancelled == []
            # Existing native producers retain their callback owner and can
            # still resolve the accepted waiter after failed cleanup.
            library.finish(701, reason=3, text=b"Error: engine shutdown")
            with pytest.raises(DenseCoreRuntimeError, match="shutdown"):
                request.result(2)
        empty_tracking(engine)
    finally:
        # Fake-only teardown; production intentionally retains an unknown owner.
        engine_module._failed_close_owners.pop(id(engine), None)


def test_free_then_raise_never_retries_already_destroyed_native_handle(monkeypatch):
    engine, library = make_engine(monkeypatch)
    free = library.FreeEngine
    def free_then_raise(handle):
        free(handle)
        raise RuntimeError("exception after native destruction")
    monkeypatch.setattr(library, "FreeEngine", free_then_raise)
    try:
        with pytest.raises(RuntimeError):
            engine.close()
        with pytest.raises(RuntimeError):
            engine.close()
        with pytest.raises(engine_module.EngineNotInitializedError):
            engine.stop_generation(1)
        engine.__del__()
        assert library.free_calls == 1
        assert library.cancelled == []
        assert not engine._closed  # No successful join was confirmed to Python.
    finally:
        engine_module._failed_close_owners.pop(id(engine), None)


def test_async_cancel_keeps_peer_running_and_late_terminal_after_loop_close_is_safe(monkeypatch):
    import asyncio
    engine, library = make_engine(monkeypatch)
    library.cancel_finishes = False
    submitted = {}
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda req, native, prompt: submitted.update({prompt: (req, native)})
    ))
    errors = []
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        target = asyncio.create_task(engine.generate_async("target"))
        peer = asyncio.create_task(engine.generate_async("peer"))
        for _ in range(2000):
            if len(submitted) == 2 and all(state.native_request_id is not None for state in engine._request_completions.values()):
                break
            await asyncio.sleep(.001)
        target_req, target_native = submitted["target"]
        peer_req, peer_native = submitted["peer"]
        target.cancel()
        with pytest.raises(asyncio.CancelledError):
            await target
        assert library.cancelled == [target_native]
        assert target_req in engine._requests and peer_req in engine._requests
        engine._global_callback(b"peer succeeded", 0, peer_req)
        library.finish(peer_native, count=3, reason=2)
        assert await peer == "peer succeeded"
    asyncio.run(run())
    target_req, target_native = submitted["target"]
    # Closing the caller's loop cannot turn a late native terminal into an
    # unhandled InvalidStateError or discard its request cleanup owner.
    library.finish(target_native, reason=3, text=b"Error: request canceled")
    assert errors == []
    empty_tracking(engine)
    engine.close()


def test_task_cancel_consumes_native_error_already_delivered_before_submit_returns(monkeypatch):
    import asyncio
    engine, library = make_engine(monkeypatch)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def in_native(_req, native, _prompt):
        library.finish(native, reason=3)
        entered.set()
        assert release.wait(2)
        returned.set()
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(engine, library, in_native))
    errors = []
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        task = asyncio.create_task(engine.generate_async("prompt"))
        for _ in range(2000):
            if entered.is_set():
                break
            await asyncio.sleep(.001)
        await asyncio.sleep(0)  # Native ERROR has settled the not-yet-awaited future.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(2000):
            if returned.is_set():
                break
            await asyncio.sleep(.001)
        gc.collect()
        await asyncio.sleep(0)
    asyncio.run(run())
    assert errors == [] and library.cancelled == []
    empty_tracking(engine)
    engine.close()


@pytest.mark.parametrize("operation", [
    "metrics", "count", "tokenizer", "template", "tokenize", "runtime_plan",
    "render", "lora_load", "lora_unload", "lora_enable", "lora_disable", "graph", "weights",
])
def test_every_opaque_handle_operation_finishes_before_native_free(monkeypatch, operation):
    import ctypes
    import types
    import numpy as np
    engine, library = make_engine(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def blocked(result, *args):
        entered.set()
        assert release.wait(2)
        return result
    class Lora:
        def load(self, *_args, **_kwargs): pass
        def activate(self, *_args): pass
        def unload(self, *_args): pass
        def deactivate(self): pass
        def list_adapters(self): return ["adapter"]
    engine._lora_manager = Lora()
    if operation == "metrics":
        engine._has_metrics = True
        library.GetMetrics = lambda *_args: blocked(types.SimpleNamespace(
            requests_per_second=1., tokens_per_second=2., active_requests=1, total_tokens_generated=3
        ))
        call = engine.get_metrics
    elif operation == "count":
        engine._has_count_tokens_api = True
        library.CountTokens = lambda *_args: blocked(1)
        def call():
            return engine.count_tokens("x")
    elif operation in ("tokenizer", "template"):
        engine._has_tokenizer_metadata_api = engine._has_chat_template_api = True
        library.GetTokenizerType = library.GetChatTemplate = lambda *_args: blocked(b"value")
        call = engine.get_tokenizer_type if operation == "tokenizer" else engine.get_chat_template
    elif operation == "tokenize":
        engine._has_count_tokens_api = engine._has_native_tokenizer_api = True
        library.CountTokens = lambda *_args: 1
        def tokenize(*args):
            blocked(0)
            args[4][0] = 42
            return 1
        library.DenseCoreTokenizeText = tokenize
        def call():
            return engine.tokenize_text("x")
    elif operation == "runtime_plan":
        monkeypatch.setattr(engine_module, "runtime_plan", lambda *_args: blocked({"context": 16}))
        call = engine.get_runtime_plan
    elif operation == "render":
        engine._has_native_chat_render_api = True
        def render(*args):
            blocked(0)
            output = ctypes.cast(args[-1], ctypes.POINTER(engine_module.DenseCoreRenderedChatPrompt)).contents
            output.rendered_prompt = b"rendered"
            output.tokenizer_type = b"unit"
            return 0
        library.DenseCoreRenderChatPrompt = render
        def call():
            return engine.render_chat_prompt([{"role": "user", "content": "x"}])
    elif operation.startswith("lora_"):
        engine._has_lora_api = True
        library.LoadLoraAdapter = library.UnloadLoraAdapter = library.DeactivateLoraAdapters = lambda *_args: 0
        library.ActivateLoraAdapter = lambda *_args: 0
        symbol = {"lora_load": "LoadLoraAdapter", "lora_unload": "UnloadLoraAdapter",
                  "lora_enable": "ActivateLoraAdapter", "lora_disable": "DeactivateLoraAdapters"}[operation]
        setattr(library, symbol, lambda *_args: blocked(0))
        call = {"lora_load": lambda: engine.load_lora("file", name="adapter"),
                "lora_unload": lambda: engine.unload_lora("adapter"),
                "lora_enable": lambda: engine.enable_lora("adapter"),
                "lora_disable": engine.disable_lora}[operation]
    elif operation == "graph":
        def graph(*args):
            blocked(0)
            args[4][0].ndim = 1
            args[4][0].shape[0] = 1
            return 0
        library.ExecuteGraphSync = graph
        def call():
            return engine._execute_graph_sync([("input", np.zeros(1, dtype=np.float32))], "unit")
    else:
        class Weights:
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def keys(self): return ["weight"]
            def get_tensor(self, _key): return np.zeros(1, dtype=np.float32)
        monkeypatch.setattr(engine_module, "SAFETENSORS_AVAILABLE", True)
        monkeypatch.setattr(engine_module, "safe_open", lambda *_args, **_kwargs: Weights(), raising=False)
        engine._has_hal_api = True
        library.DenseCoreSetWeight = lambda *_args: blocked(0)
        def call():
            return engine.load_weights("file")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        operation_done = pool.submit(call)
        assert entered.wait(2)
        closed = pool.submit(engine.close)
        deadline = time.monotonic() + 2
        while not engine._closing and time.monotonic() < deadline:
            time.sleep(.001)
        assert engine._closing and library.free_calls == 0
        assert not closed.done()
        release.set()
        operation_done.result(2)
        closed.result(2)
    assert library.free_calls == 1


def test_bound_handle_operation_captured_before_quarantine_cannot_enter_native(monkeypatch):
    engine, library = make_engine(monkeypatch)
    engine._has_count_tokens_api = True
    calls = []
    library.CountTokens = lambda *_args: calls.append("count") or 1
    captured_operation = engine.count_tokens
    failure = RuntimeError("native free outcome unknown")
    monkeypatch.setattr(library, "FreeEngine", lambda _handle: (_ for _ in ()).throw(failure))
    try:
        with pytest.raises(RuntimeError):
            engine.close()
        with pytest.raises(RuntimeError) as error:
            captured_operation("x")
        assert error.value is failure and calls == []
    finally:
        engine_module._failed_close_owners.pop(id(engine), None)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("close_engine", [False, True])
def test_queued_async_submit_cancelled_before_executor_starts_never_enters_native(monkeypatch, streaming, close_engine):
    import asyncio
    engine, library = make_engine(monkeypatch)
    executor = concurrent.futures.ThreadPoolExecutor(1)
    occupied, release = threading.Event(), threading.Event()
    blocker = executor.submit(lambda: (occupied.set(), release.wait(3)))
    assert occupied.wait(2)
    monkeypatch.setattr(engine_module, "_get_inference_executor", lambda: executor)
    native_calls = []
    monkeypatch.setattr(engine, "_submit_request_impl", tracked_submit(
        engine, library, lambda *_args: native_calls.append("submit")
    ))
    errors = []
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        async def execute():
            if streaming:
                return [piece async for piece in engine.stream_async("prompt")]
            return await engine.generate_async("prompt")
        task = asyncio.create_task(execute())
        for _ in range(2000):
            if engine._request_completions and engine._request_completions[1].submit_job is not None:
                break
            await asyncio.sleep(.001)
        else:
            pytest.fail("submit job was not queued")
        if close_engine:
            engine.close()
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        empty_tracking(engine)
        assert native_calls == [] and library.cancelled == []
        await asyncio.sleep(0)
    try:
        asyncio.run(run())
    finally:
        release.set()
        blocker.result(2)
        executor.shutdown(wait=True, cancel_futures=True)
    assert errors == [] and native_calls == []
    engine.close()


def test_executor_cancelling_never_started_submit_retires_request_without_loop_dependency(monkeypatch):
    import asyncio
    engine, library = make_engine(monkeypatch)
    executor = concurrent.futures.ThreadPoolExecutor(1)
    occupied, release = threading.Event(), threading.Event()
    blocker = executor.submit(lambda: (occupied.set(), release.wait(3)))
    assert occupied.wait(2)
    monkeypatch.setattr(engine_module, "_get_inference_executor", lambda: executor)
    native_calls = []
    monkeypatch.setattr(engine, "_submit_request_impl", lambda *_args: native_calls.append("submit") or 701)
    async def run():
        task = asyncio.create_task(engine.generate_async("prompt"))
        for _ in range(2000):
            if engine._request_completions and engine._request_completions[1].submit_job is not None:
                break
            await asyncio.sleep(.001)
        executor.shutdown(wait=False, cancel_futures=True)
        with pytest.raises(asyncio.CancelledError):
            await task
        empty_tracking(engine)
    try:
        asyncio.run(run())
    finally:
        release.set()
        blocker.result(2)
        executor.shutdown(wait=True)
    assert native_calls == []
    engine.close()
