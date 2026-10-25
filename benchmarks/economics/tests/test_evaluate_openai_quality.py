import http.server
import importlib.util
import json
import tempfile
from unittest.mock import patch
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
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
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
            self.assertEqual(result["evaluation_channel"], "final_content")
        finally:
            server.shutdown()
            server.server_close()

    def test_reasoning_only_correct_word_cannot_pass_with_thinking_enabled(self):
        case = {"id": "reasoning-only", "expected_all": ["Paris"]}
        for visible, reasoning in [("", "Paris"), ("<think>Paris</think>London", ""), ("<think>Paris", ""), ("Paris<think>unfinished", "")]:
            with self.subTest(visible=visible):
                with patch.object(quality, "request_visible_content", return_value=(200, visible, reasoning, None)):
                    result = quality.evaluate_case("unused", case, 1, enable_thinking=True)
                self.assertFalse(result["passed"])

    def test_gemma_channel_leaks_never_qualify_or_hide_raw_response(self):
        for visible in ["<|channel>thought\nParis", "<|channel>analysis\nParis<channel|>London", "Paris<channel|>"]:
            with self.subTest(visible=visible):
                with patch.object(quality, "request_visible_content", return_value=(200, visible, "", None)):
                    result = quality.evaluate_case("unused", {"expected_all": ["Paris"]}, 1)
                self.assertFalse(result["passed"])
                self.assertEqual(result["visible_content"], visible)
                self.assertEqual(result["final_content"], "")
                self.assertIn("control token leaked", result["error"])

    def test_final_answer_can_pass_after_embedded_thinking(self):
        with patch.object(quality, "request_visible_content", return_value=(200, "<think>London?</think>Paris", "", None)):
            result = quality.evaluate_case("unused", {"expected_all": ["Paris"], "forbidden_any": ["London"]}, 1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["final_content"], "Paris")

    def test_truncated_or_unterminated_stream_never_qualifies(self):
        import io
        for reason, done in [("length", True), ("stop", False), (None, True)]:
            class Response(io.BytesIO):
                status = 200
            events = [{"choices": [{"delta": {"content": "Paris"}, "finish_reason": reason}]}]
            raw = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            if done:
                raw += "data: [DONE]\n\n"
            with self.subTest(reason=reason, done=done):
                with patch.object(quality.urllib.request, "urlopen", return_value=Response(raw.encode())):
                    result = quality.evaluate_case("http://localhost/unused", {"messages": [], "expected_all": ["Paris"]}, 1)
                self.assertFalse(result["passed"])
                self.assertIn("incomplete final answer", result["error"])


if __name__ == "__main__":
    unittest.main()
