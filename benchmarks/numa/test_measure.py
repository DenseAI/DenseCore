#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("measure.py")
SPEC = importlib.util.spec_from_file_location("numa_measure", MODULE_PATH)
MEASURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEASURE)


class MeasurePayloadTest(unittest.TestCase):
    def test_no_thinking_uses_request_template_kwargs(self):
        body = MEASURE.build_chat_body("local", "hello", 128, 0.0, no_thinking=True)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])

    def test_default_payload_does_not_override_template(self):
        body = MEASURE.build_chat_body("local", "hello", 128, 0.0)
        self.assertNotIn("chat_template_kwargs", body)

    def test_no_thinking_rejects_reasoning_trace(self):
        defects = MEASURE.output_defects(
            "Answer directly", "Here's a thinking process: analyze user input before answering.", True
        )
        self.assertIn("THINKING_LEAK", defects)


if __name__ == "__main__":
    unittest.main()
