import http.server
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_openai_quality.py"
SPEC = importlib.util.spec_from_file_location("economics_quality", MODULE_PATH)
quality = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(quality)


class QualityHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"reasoning_content": "hidden"}}]},
            {"choices": [{"delta": {"content": "Policy: routine; renewal in 30 days."}}]},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        pass


class QualityEvaluationTest(unittest.TestCase):
    def test_visible_content_ignores_reasoning_and_checks_terms(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QualityHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = {
                "id": "case",
                "messages": [{"role": "user", "content": "classify"}],
                "expected_all": ["routine", "30 days"],
                "forbidden_any": ["hidden"],
            }
            result = quality.evaluate_case(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", case, 5)
            self.assertTrue(result["passed"])
            self.assertEqual(result["visible_content"], "Policy: routine; renewal in 30 days.")
            self.assertEqual(result["reasoning_content"], "hidden")
        finally:
            server.shutdown()
            server.server_close()

    def test_quality_request_disables_thinking_and_sets_greedy_top_k(self):
        received: dict[str, object] = {}

        class CaptureHandler(QualityHandler):
            def do_POST(self):
                received.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                super().do_POST()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = {"id": "case", "messages": [{"role": "user", "content": "classify"}], "expected_all": ["routine"]}
            quality.evaluate_case(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions", case, 5, enable_thinking=False
            )
            self.assertEqual(received["top_k"], 1)
            self.assertEqual(received["chat_template_kwargs"], {"enable_thinking": False})
        finally:
            server.shutdown()
            server.server_close()

    def test_quality_request_can_enable_thinking(self):
        received: dict[str, object] = {}

        class CaptureHandler(QualityHandler):
            def do_POST(self):
                received.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                super().do_POST()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = {"id": "case", "messages": [{"role": "user", "content": "classify"}], "expected_all": ["routine"]}
            quality.evaluate_case(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions", case, 5, enable_thinking=True
            )
            self.assertEqual(received["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(received["max_tokens"], 1024)
            result = quality.evaluate_case(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions", case, 5, enable_thinking=True
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["evaluation_channel"], "reasoning_content+content")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
