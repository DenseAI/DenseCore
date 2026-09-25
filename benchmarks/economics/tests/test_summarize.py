import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize.py"
SPEC = importlib.util.spec_from_file_location("economics_summarize", MODULE_PATH)
summarize = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(summarize)


class SummarizeRunTest(unittest.TestCase):
    def write_rows(self, root: Path) -> Path:
        raw = root / "raw.jsonl"
        rows = [
            {"started_at_unix_s": 100.0, "elapsed_s": 2.0, "http_status": 200, "completion_tokens": 20, "ttft_s": 0.5},
            {"started_at_unix_s": 101.0, "elapsed_s": 3.0, "http_status": 500, "completion_tokens": None},
            {"started_at_unix_s": 102.0, "elapsed_s": 2.0, "http_status": 200, "completion_tokens": 10, "ttft_s": 1.0},
        ]
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return raw

    def test_excludes_failed_rows_but_charges_full_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.write_rows(Path(temp))
            summary = summarize.summarize_run(
                "cpu", raw, {"deployments": {"cpu": {"hourly_usd": 3.6, "node_count": 1}}}, {"passed": True}, [0.5], None, None
            )
        self.assertEqual(summary["requests_total"], 3)
        self.assertEqual(summary["requests_successful"], 2)
        self.assertEqual(summary["completion_tokens_successful"], 30)
        self.assertAlmostEqual(summary["active_runtime_cost_usd"], 0.004)
        self.assertAlmostEqual(summary["provisioned_duty_cycle_cost_usd"]["0.5"], 0.008)
        self.assertAlmostEqual(summary["usd_per_million_completed_output_tokens_active"], 400 / 3)

    def test_failed_quality_gate_suppresses_cost_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.write_rows(Path(temp))
            summary = summarize.summarize_run(
                "cpu", raw, {"deployments": {"cpu": {"hourly_usd": 1, "node_count": 1}}}, {"passed": False}, [1.0], None, None
            )
        self.assertFalse(summary["eligible_for_cost_claim"])
        self.assertIsNone(summary["usd_per_million_completed_output_tokens_active"])

    def test_cli_requires_a_quality_artifact_for_each_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.write_rows(root)
            pricing = root / "pricing.json"
            quality = root / "quality.json"
            output = root / "summary.json"
            pricing.write_text(
                json.dumps({"deployments": {"cpu": {"hourly_usd": 1}, "gpu": {"hourly_usd": 1}}}),
                encoding="utf-8",
            )
            quality.write_text(json.dumps({"passed": True}), encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "summarize.py", "--pricing", str(pricing), "--quality", f"cpu={quality}",
                    "--run", f"cpu={raw}", "--run", f"gpu={raw}", "--duty-cycle", "1", "--output", str(output),
                ],
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        summarize.main()
            self.assertEqual(raised.exception.code, 2)

    def test_slo_failure_suppresses_cost_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.write_rows(Path(temp))
            summary = summarize.summarize_run(
                "cpu", raw, {"deployments": {"cpu": {"hourly_usd": 1}}}, {"passed": True}, [0.2, 1.0], 0.75, 1.5
            )
        self.assertFalse(summary["slo"]["passed"])
        self.assertFalse(summary["eligible_for_cost_claim"])
        self.assertEqual(summary["provisioned_duty_cycle_cost_usd"].keys(), {"0.2", "1.0"})


if __name__ == "__main__":
    unittest.main()
