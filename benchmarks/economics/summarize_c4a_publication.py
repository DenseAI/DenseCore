#!/usr/bin/env python3
"""Summarize balanced C4A serving cycles into publication statistics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "pstdev": None}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def decode_summaries(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[Qwen36DecodeSummary]" not in line:
            continue
        rows.append(dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line)))
    return rows


def rss_kib(path: Path) -> int | None:
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return None
    fields = lines[1].split()
    return int(fields[1]) if len(fields) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    order = [int(value) for value in (root / "run-order.txt").read_text().split()]
    warmup_order = [int(value) for value in (root / "warmup-order.txt").read_text().split()]
    summaries = decode_summaries(root / "server-performance.log")
    expected = sum(warmup_order) + sum(order)
    if len(summaries) != expected:
        raise ValueError(f"decode summaries={len(summaries)}, expected={expected}")
    summaries = summaries[sum(warmup_order) :]

    cycles: list[dict[str, Any]] = []
    by_concurrency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    offset = 0
    for cycle_index, concurrency in enumerate(order, start=1):
        stem = f"cycle-{cycle_index:02d}-c{concurrency}"
        metadata = json.loads((root / "cycles" / f"{stem}.metadata.json").read_text())
        requests = summaries[offset : offset + concurrency]
        offset += concurrency
        for request in requests:
            required = {
                "target_fast_path_ok": "1",
                "qwen_fast_path_ok": "1",
                "target_no_native_moe_fallback": "1",
                "target_no_native_moe_reject": "1",
            }
            failed = {key: request.get(key) for key, expected in required.items() if request.get(key) != expected}
            if failed:
                raise ValueError(f"cycle={cycle_index} c={concurrency}: fast-path qualification failed: {failed}")
        steady = [float(row["steady_visible_tok_s"]) for row in requests]
        rss_path = root / "cycles" / f"{stem}-rss-after.txt"
        if not rss_path.exists():
            rss_path = root / "cycles" / f"cycle-{cycle_index:02d}-rss-after.txt"
        cycle = {
            "cycle": cycle_index,
            "concurrency": concurrency,
            "aggregate_completed_output_tps": metadata["aggregate_completed_output_tps"],
            "completed_request_window_s": metadata["completed_request_window_s"],
            "actual_completion_tokens": metadata["actual_completion_tokens"],
            "failed_requests": metadata["failed_requests"],
            "synchronized_start_span_s": metadata["synchronized_start_span_s"],
            "steady_decode_tps": steady,
            "mean_steady_decode_tps": statistics.fmean(steady),
            "prompt_tokens": [int(row["prompt_tokens"]) for row in requests],
            "rss_after_kib": rss_kib(rss_path),
        }
        cycles.append(cycle)
        by_concurrency[concurrency].append(cycle)

    results: dict[str, Any] = {}
    for concurrency, rows in sorted(by_concurrency.items()):
        aggregate = [float(row["aggregate_completed_output_tps"]) for row in rows]
        steady = [value for row in rows for value in row["steady_decode_tps"]]
        raw_rows: list[dict[str, Any]] = []
        for row in rows:
            raw_path = root / "cycles" / f"cycle-{row['cycle']:02d}-c{concurrency}.jsonl"
            raw_rows.extend(json.loads(line) for line in raw_path.read_text().splitlines() if line.strip())
        ttft = [float(row["ttft_s"]) for row in raw_rows if row.get("ttft_s") is not None]
        e2e = [float(row["elapsed_s"]) for row in raw_rows]
        e2e_output_tps = [float(row["e2e_output_tps"]) for row in raw_rows if row.get("e2e_output_tps") is not None]
        completion_tokens = [int(row["completion_tokens"]) for row in raw_rows if row.get("completion_tokens") is not None]
        itl_p50 = [float(row["itl_p50_s"]) for row in raw_rows if row.get("itl_p50_s") is not None]
        itl_p95 = [float(row["itl_p95_s"]) for row in raw_rows if row.get("itl_p95_s") is not None]
        results[str(concurrency)] = {
            "scored_cycles": len(rows),
            "requests": len(raw_rows),
            "failures": sum(int(row["failed_requests"]) for row in rows),
            "fast_path_qualified_requests": len(raw_rows),
            "peak_rss_kib": max(
                (int(row["rss_after_kib"]) for row in rows if row["rss_after_kib"] is not None), default=None
            ),
            "aggregate_completed_output_tps": stats(aggregate),
            "per_request_steady_decode_tps": stats(steady),
            "per_request_e2e_output_tps": stats(e2e_output_tps),
            "completion_tokens": stats(completion_tokens),
            "synchronized_start_span_s": stats([float(row["synchronized_start_span_s"]) for row in rows]),
            "ttft_p50_s": percentile(ttft, 0.50),
            "ttft_p95_s": percentile(ttft, 0.95),
            "e2e_latency_p50_s": percentile(e2e, 0.50),
            "e2e_latency_p95_s": percentile(e2e, 0.95),
            "itl_p50_s": percentile(itl_p50, 0.50),
            "itl_p95_s": percentile(itl_p95, 0.95),
        }

    output = {"cycles": cycles, "by_concurrency": results}
    (root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
