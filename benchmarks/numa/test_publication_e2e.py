#!/usr/bin/env python3
"""Regression tests for the fail-closed NUMA publication summarizer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_publication_e2e.py")
SPEC = importlib.util.spec_from_file_location("summarize_publication_e2e", MODULE_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
CONVO_PATH = Path(__file__).with_name("convo_e2e.py")
CONVO_SPEC = importlib.util.spec_from_file_location("convo_e2e", CONVO_PATH)
CONVO = importlib.util.module_from_spec(CONVO_SPEC)
CONVO_SPEC.loader.exec_module(CONVO)


def transcript(path, assistant="stable answer"):
    turns = []
    for turn in (1, 2):
        turns.append(
            {
                "turn": turn,
                "assistant": assistant,
                "completion_tokens": 100,
                "prompt_tokens": 20 + turn,
                "finish_reason": "length",
                "seconds": 5.0,
                "problems": [],
            }
        )
    path.write_text(json.dumps({"turns": turns, "ok_turns": 2}), encoding="utf-8")


def summary_line(state, sticky, direct, node_dispatch, fallback=0):
    return (
        "[Qwen36DecodeSummary] steady_visible_tokens=99 decode_visible_ms=5000 "
        "target_fast_path_ok=1 target_no_native_moe_fallback=1 target_no_native_moe_reject=1 "
        f"native_moe_fallback_w1w3_ops={fallback} native_moe_fallback_w2_ops=0 "
        "native_moe_fast_decode_rejected_ops=0 native_moe_fast_w2_q5k_rejected_ops=0 "
        "qwen36_prefill_native_moe_fast_rejected_ops=0 "
        f"native_moe_numa_sticky_state={state} native_moe_numa_sticky_dispatch_ops={sticky} "
        f"native_moe_numa_direct_decode_used_ops={direct} native_moe_numa_direct_decode_rejected_ops=0 "
        "native_moe_numa_grouped_decode_used_ops=0 "
        f"native_moe_numa_node_dispatch_ops={node_dispatch}\n"
    )


class PublicationSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        transcript(self.root / "off.json")
        transcript(self.root / "on.json")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_accepts_true_same_binary_off_on_pair(self):
        off_log = self.root / "off.log"
        on_log = self.root / "on.log"
        off_log.write_text(summary_line("disabled_by_debug", 0, 0, "0:0,1:0") * 2, encoding="utf-8")
        on_log.write_text(summary_line("enabled", 100, 80, "0:40,1:40") * 2, encoding="utf-8")

        SUMMARY.assert_same_engine_parity(self.root / "off.json", self.root / "on.json", "off/on")
        off = SUMMARY.dense_result("off", self.root / "off.json", off_log, "disabled_by_debug", False)
        on = SUMMARY.dense_result("on", self.root / "on.json", on_log, "enabled", True)
        self.assertEqual(off["direct_decode_used_ops"], 0)
        self.assertEqual(on["node_dispatch_ops"], {"0": 40, "1": 40})

    def test_rejects_hidden_native_moe_fallback(self):
        log = self.root / "bad.log"
        log.write_text(summary_line("enabled", 100, 80, "0:40,1:40", fallback=1) * 2, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "native_moe_fallback_w1w3_ops=0"):
            SUMMARY.dense_result("bad", self.root / "on.json", log, "enabled", True)

    def test_parity_includes_usage_and_finish_reason(self):
        candidate = json.loads((self.root / "on.json").read_text(encoding="utf-8"))
        candidate["turns"][1]["finish_reason"] = "stop"
        (self.root / "on.json").write_text(json.dumps(candidate), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "output parity failed"):
            SUMMARY.assert_same_engine_parity(self.root / "off.json", self.root / "on.json", "off/on")

    def test_no_thinking_rejects_explicit_reasoning_leak(self):
        problems = CONVO.judge(
            ["Answer in two sentences"],
            "Here's a thinking process: 1. Analyze User Input. The direct answer follows.",
            32,
            no_thinking=True,
        )
        self.assertIn("THINKING_LEAK", problems)

    def test_cross_engine_comparison_uses_only_identical_api_e2e_for_speedup(self):
        comparison = SUMMARY.cross_engine_comparison(
            {"steady_decode_tok_s": 20.0, "e2e_completion_tok_s": 8.0},
            {"decode_tok_s": 10.0, "e2e_completion_tok_s": 16.0},
        )
        self.assertEqual(comparison["identical_api_e2e_speedup_pct"], -50.0)
        self.assertFalse(comparison["engine_native_indicators_comparable"])
        self.assertNotIn("engine_native_speedup_pct", comparison)


if __name__ == "__main__":
    unittest.main()
