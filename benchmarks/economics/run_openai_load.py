#!/usr/bin/env python3
"""Record request-level OpenAI-compatible serving evidence without dependencies."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_workload(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: request must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no request rows")
    return rows


def stream_response(response: Any, started: float) -> tuple[float | None, str, int | None, list[float]]:
    first_token_s: float | None = None
    text_parts: list[str] = []
    completion_tokens: int | None = None
    token_event_times: list[float] = []
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
        usage = event.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            completion_tokens = usage["completion_tokens"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            chunk = delta.get("content")
            if isinstance(chunk, str) and chunk:
                token_event_times.append(time.monotonic() - started)
                if first_token_s is None:
                    first_token_s = time.monotonic() - started
                text_parts.append(chunk)
    itl_s = [right - left for left, right in zip(token_event_times, token_event_times[1:])]
    return first_token_s, "".join(text_parts), completion_tokens, itl_s


def execute(url: str, payload: dict[str, Any], timeout_s: float, stream: bool, label: str, request_id: str,
            start_barrier: threading.Barrier | None) -> dict[str, Any]:
    if start_barrier is not None:
        start_barrier.wait(timeout=timeout_s)
    started_wall = time.time()
    started = time.monotonic()
    body = dict(payload)
    body["stream"] = stream
    if stream:
        body.setdefault("stream_options", {"include_usage": True})
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        method="POST",
    )
    result: dict[str, Any] = {
        "label": label,
        "request_id": request_id,
        "started_at_unix_s": started_wall,
        "started_at_monotonic_s": started,
        "stream": stream,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result["http_status"] = response.status
            if stream:
                ttft_s, text, completion_tokens, itl_s = stream_response(response, started)
                result["ttft_s"] = ttft_s
                result["output_text"] = text
                result["completion_tokens"] = completion_tokens
                result["itl_p50_s"] = percentile(itl_s, 0.50)
                result["itl_p95_s"] = percentile(itl_s, 0.95)
            else:
                value = json.loads(response.read().decode("utf-8"))
                result["output_text"] = "".join(
                    choice.get("message", {}).get("content", "") for choice in value.get("choices") or []
                )
                usage = value.get("usage") or {}
                result["completion_tokens"] = usage.get("completion_tokens")
                result["ttft_s"] = None
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:  # Network failures are benchmark outcomes.
        result["http_status"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = time.monotonic() - started
    result["finished_at_monotonic_s"] = started + result["elapsed_s"]
    result["finished_at_unix_s"] = started_wall + result["elapsed_s"]
    completion_tokens = result.get("completion_tokens")
    result["e2e_output_tps"] = completion_tokens / result["elapsed_s"] if completion_tokens else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--requests", type=int)
    execution.add_argument("--end-monotonic-s", type=float)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synchronized-start", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    if args.requests is not None and args.requests <= 0:
        parser.error("--requests must be positive")
    if args.end_monotonic_s is not None and args.end_monotonic_s <= time.monotonic():
        parser.error("--end-monotonic-s must be in the future")
    if args.concurrency <= 0 or args.timeout_s <= 0:
        parser.error("--concurrency and --timeout-s must be positive")
    if args.requests is not None and args.synchronized_start and args.requests > args.concurrency:
        parser.error("--synchronized-start requires --concurrency >= --requests")

    workload = load_workload(args.workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    run_started_monotonic = time.monotonic()
    results: list[dict[str, Any]] = []
    if args.requests is not None:
        submitted = [(index, workload[index % len(workload)]) for index in range(args.requests)]
        start_barrier = threading.Barrier(args.requests) if args.synchronized_start else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    execute,
                    args.url,
                    {key: value for key, value in item.items() if key != "id"},
                    args.timeout_s,
                    args.stream,
                    args.label,
                    str(item.get("id", index)),
                    start_barrier,
                )
                for index, item in submitted
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        counter = 0
        counter_lock = threading.Lock()
        worker_barrier = threading.Barrier(args.concurrency) if args.synchronized_start else None

        def duration_worker() -> list[dict[str, Any]]:
            nonlocal counter
            rows: list[dict[str, Any]] = []
            if worker_barrier is not None:
                worker_barrier.wait(timeout=args.timeout_s)
            while time.monotonic() < args.end_monotonic_s and not (
                args.stop_file is not None and args.stop_file.exists()
            ):
                with counter_lock:
                    index = counter
                    counter += 1
                item = workload[index % len(workload)]
                rows.append(
                    execute(
                        args.url,
                        {key: value for key, value in item.items() if key != "id"},
                        args.timeout_s,
                        args.stream,
                        args.label,
                        f"{item.get('id', 'request')}-{index}",
                        None,
                    )
                )
            return rows

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for worker_rows in pool.map(lambda _: duration_worker(), range(args.concurrency)):
                results.extend(worker_rows)
    run_ended = time.time()
    run_ended_monotonic = time.monotonic()
    results.sort(key=lambda row: row["started_at_monotonic_s"])
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    elapsed = [row["elapsed_s"] for row in results]
    successful = [row for row in results if row.get("http_status") == 200 and row.get("completion_tokens")]
    batch_started = min((row["started_at_monotonic_s"] for row in results), default=run_started_monotonic)
    batch_finished = max((row["finished_at_monotonic_s"] for row in results), default=run_ended_monotonic)
    batch_elapsed = batch_finished - batch_started
    completion_tokens = sum(row["completion_tokens"] for row in successful)
    ttft = [row["ttft_s"] for row in successful if row.get("ttft_s") is not None]
    e2e_tps = [row["e2e_output_tps"] for row in successful if row.get("e2e_output_tps") is not None]
    metadata = {
        "label": args.label,
        "url": args.url,
        "workload": str(args.workload),
        "requests": len(results),
        "execution_mode": "fixed_end_monotonic" if args.end_monotonic_s is not None else "request_count",
        "registered_end_monotonic_s": args.end_monotonic_s,
        "stop_file": str(args.stop_file) if args.stop_file is not None else None,
        "stopped_by_orchestrator": bool(args.stop_file is not None and args.stop_file.exists()),
        "concurrency": args.concurrency,
        "stream": args.stream,
        "synchronized_start": args.synchronized_start,
        "run_started_at_unix_s": run_started,
        "run_ended_at_unix_s": run_ended,
        "run_started_at_monotonic_s": run_started_monotonic,
        "run_ended_at_monotonic_s": run_ended_monotonic,
        "request_activity_started_at_monotonic_s": min(
            (row["started_at_monotonic_s"] for row in results), default=None
        ),
        "request_activity_ended_at_monotonic_s": max(
            (row["finished_at_monotonic_s"] for row in results), default=None
        ),
        "elapsed_s": run_ended_monotonic - run_started_monotonic,
        "synchronized_start_span_s": max(
            (row["started_at_monotonic_s"] for row in results), default=run_started_monotonic
        ) - batch_started,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "actual_completion_tokens": completion_tokens,
        "completed_request_window_s": batch_elapsed,
        "aggregate_completed_output_tps": completion_tokens / batch_elapsed if batch_elapsed > 0 else None,
        "p50_ttft_s": percentile(ttft, 0.50),
        "p95_ttft_s": percentile(ttft, 0.95),
        "p50_e2e_output_tps": percentile(e2e_tps, 0.50),
        "p95_e2e_output_tps": percentile(e2e_tps, 0.95),
        "mean_request_elapsed_s": statistics.fmean(elapsed) if elapsed else None,
        "p50_request_elapsed_s": percentile(elapsed, 0.50),
        "p95_request_elapsed_s": percentile(elapsed, 0.95),
    }
    # Each concurrency leg shares a directory, so metadata must follow its raw
    # artifact rather than overwrite a sibling leg's provenance.
    args.output.with_name(f"{args.output.stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if args.require_success and (metadata["failed_requests"] or not results):
        print(
            f"benchmark failed: {metadata['failed_requests']} of {metadata['requests']} requests failed",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
