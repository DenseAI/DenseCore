import http.server
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_openai_load.py"


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"content": "ready"}}]},
            {"choices": [], "usage": {"completion_tokens": 7}},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        pass


class FailingHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b"unavailable")

    def log_message(self, format, *args):
        pass


class OpenAILoadTest(unittest.TestCase):
    def test_fixed_end_mode_repeats_workload_and_records_monotonic_activity(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                workload = root / "workload.jsonl"
                output = root / "raw.jsonl"
                workload.write_text(json.dumps({"id": "case", "messages": [{"role": "user", "content": "hello"}]}) + "\n")
                registered_end = time.monotonic() + 0.25
                subprocess.run(
                    [
                        "python3", str(SCRIPT), "--url", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        "--workload", str(workload), "--output", str(output), "--end-monotonic-s", str(registered_end),
                        "--concurrency", "1", "--label", "duration", "--stream", "--require-success",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                metadata = json.loads((root / "raw.metadata.json").read_text())
                self.assertEqual(metadata["execution_mode"], "fixed_end_monotonic")
                self.assertGreater(metadata["requests"], 1)
                self.assertEqual(metadata["successful_requests"], metadata["requests"])
                self.assertIsNotNone(metadata["request_activity_started_at_monotonic_s"])
                self.assertGreaterEqual(metadata["request_activity_ended_at_monotonic_s"], registered_end)
        finally:
            server.shutdown()
            server.server_close()

    def test_streaming_artifact_contains_usage_and_ttft(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                workload = root / "workload.jsonl"
                output = root / "raw.jsonl"
                workload.write_text(json.dumps({"id": "case", "messages": [{"role": "user", "content": "hello"}]}) + "\n")
                subprocess.run(
                    [
                        "python3", str(SCRIPT), "--url", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        "--workload", str(workload), "--output", str(output), "--requests", "2", "--concurrency", "2",
                        "--label", "cpu", "--stream", "--synchronized-start",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["http_status"] == 200 for row in rows))
                self.assertTrue(all(row["completion_tokens"] == 7 for row in rows))
                self.assertTrue(all(row["ttft_s"] is not None for row in rows))
                self.assertTrue(all(row["output_text"] == "ready" for row in rows))
                metadata = json.loads((root / "raw.metadata.json").read_text())
                self.assertEqual(metadata["actual_completion_tokens"], 14)
                self.assertEqual(metadata["failed_requests"], 0)
                self.assertGreater(metadata["aggregate_completed_output_tps"], 0)
                self.assertLess(metadata["synchronized_start_span_s"], 0.25)
        finally:
            server.shutdown()
            server.server_close()

    def test_require_success_fails_closed_after_preserving_artifacts(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FailingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                workload = root / "workload.jsonl"
                output = root / "raw.jsonl"
                workload.write_text(
                    json.dumps({"id": "case", "messages": [{"role": "user", "content": "hello"}]}) + "\n"
                )
                result = subprocess.run(
                    [
                        "python3", str(SCRIPT), "--url",
                        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        "--workload", str(workload), "--output", str(output), "--requests", "1",
                        "--concurrency", "1", "--label", "cpu", "--require-success",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(output.exists())
                metadata = json.loads((root / "raw.metadata.json").read_text())
                self.assertEqual(metadata["failed_requests"], 1)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
