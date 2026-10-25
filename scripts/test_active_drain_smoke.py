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
    def exercise(self, max_tokens=None, duplicate_done=False, inactive=False, signal_delay=0, timeout=3, premature_finish=False, late_ready=False, late_admission=False, native_active_after_signal=1):
        signaled = threading.Event()
        started = threading.Event()
        admission_probed = threading.Event()
        readiness_probed = threading.Event()
        metrics_probed = threading.Event()
        stream_ended = threading.Event()
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200); self.end_headers()
                    count = 0 if inactive else (native_active_after_signal if signaled.is_set() else 1)
                    self.wfile.write(f'densecore_active_requests {count}\n'.encode())
                    if signaled.is_set(): metrics_probed.set()
                    return
                unavailable = signaled.is_set() and (not late_ready or stream_ended.is_set())
                self.send_response(503 if unavailable else 200); self.end_headers()
                if signaled.is_set(): readiness_probed.set()
            def do_POST(self):
                if signaled.is_set():
                    self.send_response(200 if late_admission and not stream_ended.is_set() else 503)
                    self.end_headers(); admission_probed.set(); return
                bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"1"}}]}\n\n'); self.wfile.flush()
                started.set()
                signaled.wait(3)
                if not premature_finish:
                    for event in (admission_probed, readiness_probed, metrics_probed): event.wait(3)
                    time.sleep(0.03)
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                if duplicate_done: self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush(); stream_ended.set()
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
                if duplicate_done or inactive or signal_delay > timeout or premature_finish or late_ready or late_admission or native_active_after_signal == 0:
                    with self.assertRaises(RuntimeError): call()
                else:
                    call()
                return json.loads(out.read_text()), bodies
        finally:
            signaled.set(); admission_probed.set(); readiness_probed.set(); metrics_probed.set()
            server.shutdown(); server.server_close(); thread.join()

    def test_bounded_workload_signal_and_exact_terminal(self):
        result, bodies = self.exercise(max_tokens=32)
        self.assertEqual(result['request_max_tokens'], 32)
        self.assertEqual(bodies[0]['max_tokens'], 32)
        self.assertTrue(result['stream_incomplete_before_signal'])
        self.assertEqual(result['active_requests_before_signal'], 1)
        self.assertTrue(result['admission_blocked'])
        self.assertTrue(result['ready_removed'])
        self.assertTrue(result['ready_removed_while_stream_active'])
        self.assertTrue(result['admission_blocked_while_stream_active'])
        self.assertEqual(result['legacy_5s_regression'], 'not_exercised')
        self.assertEqual(sorted(event['status'] for event in result['probe_events']), [200, 503, 503])
        self.assertTrue(result['active_probe_rounds'][0]['qualified'])
        self.assertEqual(result['active_probe_rounds'][0]['native_active_requests'], 1)
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

    def test_closed_server_only_after_stream_finished_does_not_qualify(self):
        result, _ = self.exercise(premature_finish=True, signal_delay=0.05)
        self.assertFalse(result['ready_removed_while_stream_active'])
        self.assertFalse(result['admission_blocked_while_stream_active'])
        self.assertIn('while accepted stream remained active', result['failure'])

    def test_readiness_only_removed_after_stream_completion_fails(self):
        result, _ = self.exercise(late_ready=True)
        self.assertFalse(result['ready_removed_while_stream_active'])
        self.assertNotIn('qualified_active_probe_round', result)

    def test_admission_only_blocked_after_stream_completion_fails(self):
        result, _ = self.exercise(late_admission=True)
        self.assertFalse(result['admission_blocked_while_stream_active'])
        self.assertNotIn('qualified_active_probe_round', result)

    def test_incomplete_buffered_stream_without_native_work_during_probes_fails(self):
        result, _ = self.exercise(native_active_after_signal=0)
        self.assertFalse(result['native_active_during_shutdown_probes'])
        self.assertNotIn('qualified_active_probe_round', result)

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
