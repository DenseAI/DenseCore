import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('drain', Path(__file__).with_name('active_drain_smoke.py'))
drain = importlib.util.module_from_spec(spec); spec.loader.exec_module(drain)


class ActiveDrainProtocolTest(unittest.TestCase):
    def exercise(self, max_tokens=None, duplicate_done=False):
        signaled = threading.Event()
        started = threading.Event()
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(503 if signaled.is_set() else 200); self.end_headers()
            def do_POST(self):
                if signaled.is_set():
                    self.send_response(503); self.end_headers(); return
                bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"1"}}]}\n\n'); self.wfile.flush()
                started.set()
                signaled.wait(3)
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                if duplicate_done: self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def signal(*args, **kwargs):
            self.assertTrue(started.is_set()); signaled.set()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(drain.subprocess, 'run', side_effect=signal):
                out = Path(tmp) / 'result.json'
                call = lambda: drain.run(f'http://127.0.0.1:{server.server_port}', ['test-signal'], out, 3,
                    **({} if max_tokens is None else {'max_tokens': max_tokens}))
                if duplicate_done:
                    with self.assertRaises(RuntimeError): call()
                else:
                    call()
                return json.loads(out.read_text()), bodies
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_bounded_workload_signal_and_exact_terminal(self):
        result, bodies = self.exercise(max_tokens=32)
        self.assertEqual(result['request_max_tokens'], 32)
        self.assertEqual(bodies[0]['max_tokens'], 32)
        self.assertTrue(result['stream_incomplete_before_signal'])
        self.assertTrue(result['admission_blocked'])
        self.assertTrue(result['ready_removed'])
        self.assertEqual(result['done_count'], 1)
        self.assertEqual(result['finish_count'], 1)
        self.assertGreaterEqual(result['signal_seconds'], result['first_token_seconds'])

    def test_standalone_default_remains_256(self):
        result, bodies = self.exercise()
        self.assertEqual(result['request_max_tokens'], 256)
        self.assertEqual(bodies[0]['max_tokens'], 256)

    def test_duplicate_done_is_rejected(self):
        result, _ = self.exercise(duplicate_done=True)
        self.assertEqual(result['done_count'], 2)
        self.assertIn('duplicate', result['error'])

    def test_invalid_bound_does_not_start_request(self):
        with self.assertRaises(ValueError): drain.run('http://unused', [], 'unused.json', 3, max_tokens=0)


if __name__ == '__main__': unittest.main()
