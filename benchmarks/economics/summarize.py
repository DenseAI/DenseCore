#!/usr/bin/env python3
"""Summarize raw economics runs without hiding failed or unpriced responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--run must be LABEL=RAW_JSONL, got {value!r}")
        if label in runs:
            raise ValueError(f"duplicate run label: {label}")
        runs[label] = Path(raw_path)
    return runs


def summarize_run(
    label: str,
    raw_path: Path,
    pricing: dict[str, Any],
    quality: dict[str, Any],
    duty_cycles: list[float],
    slo_p95_ttft_s: float | None,
    slo_p95_e2e_s: float | None,
) -> dict[str, Any]:
    deployment = (pricing.get("deployments") or {}).get(label)
    if not deployment:
        raise ValueError(f"pricing deployments has no {label!r} entry")
    rows = load_jsonl(raw_path)
    if not rows:
        raise ValueError(f"{raw_path}: no rows")
    started = min(float(row["started_at_unix_s"]) for row in rows)
    ended = max(float(row["started_at_unix_s"]) + float(row["elapsed_s"]) for row in rows)
    active_hours = (ended - started) / 3600.0
    hourly_usd = float(deployment["hourly_usd"])
    node_count = int(deployment.get("node_count", 1))
    successful = [
        row for row in rows
        if row.get("http_status") == 200 and isinstance(row.get("completion_tokens"), int) and row["completion_tokens"] > 0
    ]
    output_tokens = sum(row["completion_tokens"] for row in successful)
    request_elapsed = [float(row["elapsed_s"]) for row in rows]
    ttft = [float(row["ttft_s"]) for row in successful if isinstance(row.get("ttft_s"), (int, float))]
    active_cost = hourly_usd * node_count * active_hours
    request_p95 = percentile(request_elapsed, 0.95)
    ttft_p95 = percentile(ttft, 0.95)
    quality_passed = quality.get("passed") is True
    slo_passed = (
        (slo_p95_ttft_s is None or (ttft_p95 is not None and ttft_p95 <= slo_p95_ttft_s))
        and (slo_p95_e2e_s is None or (request_p95 is not None and request_p95 <= slo_p95_e2e_s))
    )
    eligible = quality_passed and slo_passed and output_tokens > 0
    return {
        "label": label,
        "raw_path": str(raw_path),
        "quality_gate_passed": quality_passed,
        "requests_total": len(rows),
        "requests_successful": len(successful),
        "success_rate": len(successful) / len(rows),
        "completion_tokens_successful": output_tokens,
        "active_hours": active_hours,
        "active_runtime_cost_usd": active_cost,
        "provisioned_duty_cycle_cost_usd": {
            str(duty_cycle): active_cost / duty_cycle for duty_cycle in duty_cycles
        },
        "eligible_for_cost_claim": eligible,
        "usd_per_million_completed_output_tokens_active": (active_cost * 1_000_000 / output_tokens) if eligible else None,
        "usd_per_million_completed_output_tokens_provisioned": {
            str(duty_cycle): (active_cost / duty_cycle * 1_000_000 / output_tokens) if eligible else None
            for duty_cycle in duty_cycles
        },
        "request_elapsed_s": {"p50": percentile(request_elapsed, 0.50), "p95": request_p95},
        "ttft_s": {"p50": percentile(ttft, 0.50), "p95": ttft_p95},
        "aggregate_output_tok_s": output_tokens / (ended - started) if ended > started else None,
        "slo": {
            "p95_ttft_limit_s": slo_p95_ttft_s,
            "p95_e2e_limit_s": slo_p95_e2e_s,
            "passed": slo_passed,
        },
        "excluded_rows": len(rows) - len(successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--quality", action="append", required=True, help="LABEL=QUALITY_JSON; each run needs its own gate")
    parser.add_argument("--run", action="append", required=True, help="LABEL=RAW_JSONL; labels must match pricing.deployments")
    parser.add_argument("--duty-cycle", type=float, action="append", required=True)
    parser.add_argument("--slo-p95-ttft-s", type=float)
    parser.add_argument("--slo-p95-e2e-s", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 < duty_cycle <= 1 for duty_cycle in args.duty_cycle):
        parser.error("--duty-cycle must be in (0, 1]")
    if any(limit is not None and limit <= 0 for limit in (args.slo_p95_ttft_s, args.slo_p95_e2e_s)):
        parser.error("SLO limits must be positive")
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    runs = parse_runs(args.run)
    quality_paths = parse_runs(args.quality)
    if runs.keys() != quality_paths.keys():
        parser.error("--run and --quality labels must match exactly")
    quality_by_label = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in quality_paths.items()
    }
    summaries = [
        summarize_run(
            label, path, pricing, quality_by_label[label], args.duty_cycle,
            args.slo_p95_ttft_s, args.slo_p95_e2e_s,
        )
        for label, path in runs.items()
    ]
    payload = {
        "pricing_snapshot_id": pricing.get("snapshot_id"),
        "provider": pricing.get("provider"),
        "region": pricing.get("region"),
        "duty_cycles": args.duty_cycle,
        "quality": quality_by_label,
        "runs": summaries,
        "limitations": [
            "Provider pricing is a supplied snapshot; verify it before publication.",
            "Duty-cycle cost models idle compute but does not measure cold-start latency or retained storage.",
            "Cost eligibility requires a passing quality gate and observed completion-token usage.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
