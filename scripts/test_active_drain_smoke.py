import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import errno
import http.client
import json
import socket
import time
import urllib.error
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('drain', Path(__file__).with_name('active_drain_smoke.py'))
drain = importlib.util.module_from_spec(spec); spec.loader.exec_module(drain)


class ActiveDrainProtocolTest(unittest.TestCase):
    def exercise(self, max_tokens=None, duplicate_done=False, inactive=False, signal_delay=0, timeout=3):
        signaled = threading.Event()
        started = threading.Event()
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200); self.end_headers()
                    self.wfile.write(b'densecore_active_requests 0\n' if inactive else b'densecore_active_requests 1\n')
                    return
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
            self.assertLessEqual(kwargs["timeout"], timeout)
            time.sleep(signal_delay)
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(drain.subprocess, 'run', side_effect=signal):
                out = Path(tmp) / 'result.json'
                call = lambda: drain.run(f'http://127.0.0.1:{server.server_port}', ['test-signal'], out, timeout,
                    **({} if max_tokens is None else {'max_tokens': max_tokens}))
                if duplicate_done or inactive or signal_delay > timeout:
                    with self.assertRaises(RuntimeError): call()
                else:
                    call()
                return json.loads(out.read_text()), bodies
        finally:
            signaled.set()
            server.shutdown(); server.server_close(); thread.join()

    def test_bounded_workload_signal_and_exact_terminal(self):
        result, bodies = self.exercise(max_tokens=32)
        self.assertEqual(result['request_max_tokens'], 32)
        self.assertEqual(bodies[0]['max_tokens'], 32)
        self.assertTrue(result['stream_incomplete_before_signal'])
        self.assertEqual(result['active_requests_before_signal'], 1)
        self.assertTrue(result['admission_blocked'])
        self.assertTrue(result['ready_removed'])
        self.assertEqual(result['legacy_5s_regression'], 'not_exercised')
        self.assertEqual([event['status'] for event in result['probe_events']], [503, 503])
        self.assertTrue(all(event['unavailable'] for event in result['probe_events']))
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

    def test_buffered_stream_without_active_native_work_does_not_qualify(self):
        result, _ = self.exercise(inactive=True)
        self.assertEqual(result['active_requests_before_signal'], 0)
        self.assertNotIn('signal_seconds', result)

    def test_signal_and_probes_share_one_absolute_deadline(self):
        result, _ = self.exercise(signal_delay=0.4, timeout=0.3)
        self.assertIn('deadline', result['failure'])
        self.assertEqual(result['legacy_5s_regression'], 'not_exercised')

    def test_timeout_and_dns_failure_do_not_prove_admission_closed(self):
        for error in (TimeoutError('timed out'), socket.timeout('timed out'),
                socket.gaierror(-2, 'name resolution failed')):
            self.assertFalse(drain.connection_closed(error))
            self.assertFalse(drain.connection_closed(urllib.error.URLError(error)))
        for error in (ConnectionRefusedError(errno.ECONNREFUSED, 'refused'),
                ConnectionResetError(errno.ECONNRESET, 'reset'), http.client.RemoteDisconnected()):
            self.assertTrue(drain.connection_closed(error))
            self.assertTrue(drain.connection_closed(urllib.error.URLError(error)))

    def test_invalid_bound_does_not_start_request(self):
        with self.assertRaises(ValueError): drain.run('http://unused', [], 'unused.json', 3, max_tokens=0)


if __name__ == '__main__': unittest.main()
