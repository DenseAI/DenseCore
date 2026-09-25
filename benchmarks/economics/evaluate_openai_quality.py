#!/usr/bin/env python3
"""Run deterministic visible-content checks against an OpenAI chat endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


EVALUATOR_VERSION = "qwen36-channel-aware-v3"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: expected a non-empty JSON array")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("messages"), list):
            raise ValueError(f"{path}:{index}: each case needs messages")
        if not isinstance(case.get("expected_all"), list) or not case["expected_all"]:
            raise ValueError(f"{path}:{index}: each case needs non-empty expected_all")
    return cases


def assistant_stream(response: Any) -> tuple[str, str]:
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in event.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                visible_parts.append(content)
            reasoning = (choice.get("delta") or {}).get("reasoning_content")
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
    return "".join(visible_parts), "".join(reasoning_parts)


def request_visible_content(
    url: str, case: dict[str, Any], timeout_s: float, enable_thinking: bool
) -> tuple[int | None, str, str, str | None]:
    max_tokens = int(case.get("max_tokens", 256))
    if enable_thinking:
        max_tokens = int(case.get("thinking_max_tokens", max(max_tokens, 1024)))
    payload = {
        "messages": case["messages"],
        "max_tokens": max_tokens,
        "temperature": case.get("temperature", 0),
        "top_p": case.get("top_p", 1),
        "top_k": case.get("top_k", 1),
        "seed": case.get("seed", 123),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            visible, reasoning = assistant_stream(response)
            return response.status, visible, reasoning, None
    except urllib.error.HTTPError as exc:
        return exc.code, "", "", exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:
        return None, "", "", f"{type(exc).__name__}: {exc}"


def evaluate_case(url: str, case: dict[str, Any], timeout_s: float, enable_thinking: bool = False) -> dict[str, Any]:
    status, visible_content, reasoning_content, error = request_visible_content(url, case, timeout_s, enable_thinking)
    evaluation_content = reasoning_content + "\n" + visible_content if enable_thinking else visible_content
    normalized = evaluation_content.casefold()
    expected = [str(value) for value in case["expected_all"]]
    missing = [value for value in expected if value.casefold() not in normalized]
    forbidden = [value for value in case.get("forbidden_any", []) if str(value).casefold() in normalized]
    min_visible_chars = int(case.get("min_visible_chars", 1))
    passed = (
        status == 200
        and not error
        and bool(evaluation_content.strip())
        and len(evaluation_content.strip()) >= min_visible_chars
        and not missing
        and not forbidden
    )
    return {
        "id": str(case.get("id", "unnamed")),
        "enable_thinking": enable_thinking,
        "requested_max_tokens": int(
            case.get("thinking_max_tokens", max(int(case.get("max_tokens", 256)), 1024))
            if enable_thinking
            else case.get("max_tokens", 256)
        ),
        "passed": passed,
        "http_status": status,
        "visible_content": visible_content,
        "reasoning_content": reasoning_content,
        "evaluation_channel": "reasoning_content+content" if enable_thinking else "content",
        "missing_expected": missing,
        "forbidden_found": forbidden,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    cases = load_cases(args.cases)
    started = time.time()
    results = [evaluate_case(args.url, case, args.timeout_s, args.enable_thinking) for case in cases]
    failures = [result["id"] for result in results if not result["passed"]]
    payload = {
        "label": args.label,
        "model_revision": args.model_revision,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "evaluator_version": EVALUATOR_VERSION,
        "enable_thinking": args.enable_thinking,
        "cases_path": str(args.cases),
        "started_at_unix_s": started,
        "passed": not failures,
        "failures": failures,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pass and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
