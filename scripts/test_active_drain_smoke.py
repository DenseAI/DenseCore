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
    def test_signal_follows_real_token_and_stops_new_admission(self):
        signaled = threading.Event()
        started = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(503 if signaled.is_set() else 200); self.end_headers()
            def do_POST(self):
                if signaled.is_set():
                    self.send_response(503); self.end_headers(); return
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"1"}}]}\n\n'); self.wfile.flush()
                started.set()
                signaled.wait(3)
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def signal(*args, **kwargs):
            self.assertTrue(started.is_set()); signaled.set()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(drain.subprocess, 'run', side_effect=signal):
                out = Path(tmp) / 'result.json'
                drain.run(f'http://127.0.0.1:{server.server_port}', ['test-signal'], out, 3)
                result = json.loads(out.read_text())
                self.assertTrue(result['done'])
                self.assertTrue(result['admission_blocked'])
                self.assertTrue(result['ready_removed'])
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
