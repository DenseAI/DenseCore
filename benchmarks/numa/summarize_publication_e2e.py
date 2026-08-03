#!/usr/bin/env python3
"""Validate and summarize the publication-grade two-turn NUMA MoE A/B."""

import argparse
import json
import re
from pathlib import Path


SUMMARY_RE = re.compile(r"\[(?:Qwen35|Qwen36|Gemma4|LFM2)?DecodeSummary\]\s+(.*)")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_fields(payload):
    fields = {}
    for item in payload.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def dense_result(label, transcript_path, log_path, expected_state, expect_sticky):
    transcript = load_json(transcript_path)
    turns = transcript.get("turns") or []
    if len(turns) != 2 or transcript.get("ok_turns") != 2:
        raise RuntimeError(f"{label}: expected exactly 2/2 coherent turns")

    summaries = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUMMARY_RE.search(line)
        if match:
            fields = parse_fields(match.group(1))
            if int(fields.get("steady_visible_tokens", "0")) > 0:
                summaries.append(fields)
    if len(summaries) < 2:
        raise RuntimeError(f"{label}: fewer than two steady decode summaries")
    summaries = summaries[-2:]

    required_one = (
        "target_fast_path_ok",
        "target_no_native_moe_fallback",
        "target_no_native_moe_reject",
    )
    required_zero = (
        "native_moe_fallback_w1w3_ops",
        "native_moe_fallback_w2_ops",
        "native_moe_fast_decode_rejected_ops",
        "native_moe_fast_w2_q5k_rejected_ops",
        "qwen36_prefill_native_moe_fast_rejected_ops",
    )
    for index, row in enumerate(summaries, 1):
        for field in required_one:
            if row.get(field) != "1":
                raise RuntimeError(f"{label}: summary {index} requires {field}=1, got {row.get(field)!r}")
        for field in required_zero:
            if int(row.get(field, "0")) != 0:
                raise RuntimeError(f"{label}: summary {index} requires {field}=0, got {row.get(field)!r}")

    steady_tokens = sum(int(row["steady_visible_tokens"]) for row in summaries)
    decode_ms = sum(float(row["decode_visible_ms"]) for row in summaries)
    if steady_tokens <= 0 or decode_ms <= 0:
        raise RuntimeError(f"{label}: invalid steady decode window")

    final = summaries[-1]
    state = final.get("native_moe_numa_sticky_state")
    if state != expected_state:
        raise RuntimeError(f"{label}: sticky state {state!r}, expected {expected_state!r}")

    direct_used = int(final.get("native_moe_numa_direct_decode_used_ops", "0"))
    direct_rejected = int(final.get("native_moe_numa_direct_decode_rejected_ops", "0"))
    grouped_used = int(final.get("native_moe_numa_grouped_decode_used_ops", "0"))
    sticky_dispatch = int(final.get("native_moe_numa_sticky_dispatch_ops", "0"))
    dispatch = final.get("native_moe_numa_node_dispatch_ops", "")
    node_dispatch = {}
    for entry in dispatch.split(","):
        if ":" in entry:
            node, count = entry.split(":", 1)
            node_dispatch[node] = int(count)

    if expect_sticky:
        if direct_used <= 0 or direct_rejected != 0 or grouped_used != 0 or sticky_dispatch <= 0:
            raise RuntimeError(
                f"{label}: direct={direct_used}, rejected={direct_rejected}, grouped={grouped_used}, "
                f"sticky_dispatch={sticky_dispatch}"
            )
        active_nodes = [node for node, count in node_dispatch.items() if count > 0]
        if len(active_nodes) < 2:
            raise RuntimeError(f"{label}: expected dispatch on at least two nodes, got {node_dispatch}")
    elif direct_used != 0 or sticky_dispatch != 0 or any(count != 0 for count in node_dispatch.values()):
        raise RuntimeError(f"{label}: diagnostic OFF still executed sticky work")

    e2e_tokens = sum(int(turn.get("completion_tokens") or 0) for turn in turns)
    e2e_seconds = sum(float(turn.get("seconds") or 0.0) for turn in turns)
    if e2e_tokens <= 0 or e2e_seconds <= 0:
        raise RuntimeError(f"{label}: invalid end-to-end timing window")

    return {
        "label": label,
        "qa": "2/2 coherent",
        "steady_tokens": steady_tokens,
        "decode_visible_ms": round(decode_ms, 3),
        "steady_decode_tok_s": round(steady_tokens / (decode_ms / 1000.0), 4),
        "e2e_completion_tokens": e2e_tokens,
        "e2e_seconds": round(e2e_seconds, 3),
        "e2e_completion_tok_s": round(e2e_tokens / e2e_seconds, 4),
        "sticky_state": state,
        "direct_decode_used_ops": direct_used,
        "direct_decode_rejected_ops": direct_rejected,
        "grouped_decode_used_ops": grouped_used,
        "node_dispatch_ops": node_dispatch,
        "fast_path_gate": "passed_without_fallback_or_reject",
    }


def llama_result(transcript_path):
    transcript = load_json(transcript_path)
    turns = transcript.get("turns") or []
    if len(turns) != 2 or transcript.get("ok_turns") != 2:
        raise RuntimeError("llama: expected exactly 2/2 coherent turns")

    predicted_tokens = 0
    predicted_ms = 0.0
    for turn in turns:
        timings = turn.get("timings") or {}
        token_count = timings.get("predicted_n")
        elapsed_ms = timings.get("predicted_ms")
        if token_count is None or elapsed_ms is None:
            raise RuntimeError("llama: response did not expose timings.predicted_n/predicted_ms")
        predicted_tokens += int(token_count)
        predicted_ms += float(elapsed_ms)
    if predicted_tokens <= 0 or predicted_ms <= 0:
        raise RuntimeError("llama: invalid decode timing window")
    e2e_tokens = sum(int(turn.get("completion_tokens") or 0) for turn in turns)
    e2e_seconds = sum(float(turn.get("seconds") or 0.0) for turn in turns)
    if e2e_tokens <= 0 or e2e_seconds <= 0:
        raise RuntimeError("llama: invalid end-to-end timing window")
    return {
        "label": "llama_numa_distribute",
        "qa": "2/2 coherent",
        "predicted_tokens": predicted_tokens,
        "predicted_ms": round(predicted_ms, 3),
        "decode_tok_s": round(predicted_tokens / (predicted_ms / 1000.0), 4),
        "e2e_completion_tokens": e2e_tokens,
        "e2e_seconds": round(e2e_seconds, 3),
        "e2e_completion_tok_s": round(e2e_tokens / e2e_seconds, 4),
    }


def assert_same_engine_parity(reference, candidate, label):
    reference_turns = load_json(reference).get("turns") or []
    candidate_turns = load_json(candidate).get("turns") or []
    parity_fields = ("assistant", "completion_tokens", "prompt_tokens", "finish_reason")
    reference_rows = [{field: turn.get(field) for field in parity_fields} for turn in reference_turns]
    candidate_rows = [{field: turn.get(field) for field in parity_fields} for turn in candidate_turns]
    if reference_rows != candidate_rows:
        raise RuntimeError(f"{label}: deterministic same-engine output parity failed")


def cross_engine_comparison(dense, llama):
    e2e_speedup = (
        dense["e2e_completion_tok_s"] / llama["e2e_completion_tok_s"] - 1.0
    ) * 100.0
    return {
        "identical_api_e2e_speedup_pct": round(e2e_speedup, 2),
        "identical_api_metric": "completion_tokens / client wall-clock seconds",
        "engine_native_indicators": {
            "densecore_steady_visible_tok_s": dense["steady_decode_tok_s"],
            "llama_predicted_tok_s": llama["decode_tok_s"],
        },
        "engine_native_indicators_comparable": False,
        "engine_native_note": (
            "DenseCore steady-visible and llama predicted timing use different internal windows; "
            "no cross-engine speedup percentage is computed from them"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)

    off_json = outdir / "dense_off.json"
    on_json = outdir / "dense_on.json"
    migration_json = outdir / "dense_migration.json"
    assert_same_engine_parity(off_json, on_json, "sticky off/on")

    dense_off = dense_result(
        "dense_off", off_json, outdir / "dense_off.server.log", "disabled_by_debug", False
    )
    dense_on = dense_result("dense_on", on_json, outdir / "dense_on.server.log", "enabled", True)
    llama = llama_result(outdir / "llama.json")

    cross_engine = cross_engine_comparison(dense_on, llama)
    result = {
        "schema_version": 2,
        "turns_per_condition": 2,
        "dense_off": dense_off,
        "dense_on": dense_on,
        "llama": llama,
        "sticky_speedup_vs_same_binary_off_pct": round(
            (dense_on["steady_decode_tok_s"] / dense_off["steady_decode_tok_s"] - 1.0) * 100.0, 2
        ),
        "sticky_e2e_speedup_vs_same_binary_off_pct": round(
            (dense_on["e2e_completion_tok_s"] / dense_off["e2e_completion_tok_s"] - 1.0) * 100.0, 2
        ),
        "cross_engine": cross_engine,
        "same_engine_off_on_output_identical": True,
    }

    if migration_json.exists():
        assert_same_engine_parity(on_json, migration_json, "sticky/page-migration")
        migration_log = outdir / "dense_migration.server.log"
        log_text = migration_log.read_text(encoding="utf-8", errors="replace")
        migrated_pages = [int(value) for value in re.findall(r"NUMA rebalance: ([1-9][0-9]*) pages", log_text)]
        if not migrated_pages:
            raise RuntimeError("migration: no verified non-zero page migration log")
        if "permission denied - disabling NUMA rebalance" in log_text:
            raise RuntimeError("migration: permission fail-disable fired instead of successful migration")
        migration = dense_result(
            "dense_migration", migration_json, migration_log, "enabled", True
        )
        migration["verified_migrated_pages"] = sum(migrated_pages)
        result["dense_migration"] = migration
        result["migration_output_identical_to_sticky"] = True

    manifest = outdir / "manifest.json"
    if manifest.exists():
        result["manifest"] = load_json(manifest)

    migration_requested = bool(result.get("manifest", {}).get("run_migration"))
    gates = {
        "measured_physical_numa_penalty": result.get("manifest", {}).get("numa_preflight")
        == "measured_penalty_passed",
        "sticky_causal_speedup_positive": result["sticky_speedup_vs_same_binary_off_pct"] > 0.0,
        "dense_on_at_least_10pct_faster_than_llama_e2e": (
            result["cross_engine"]["identical_api_e2e_speedup_pct"] >= 10.0
        ),
        "same_engine_quality_parity": result["same_engine_off_on_output_identical"],
        "fallback_free_fast_path": dense_off["fast_path_gate"] == dense_on["fast_path_gate"]
        == "passed_without_fallback_or_reject",
        "page_migration_e2e": (not migration_requested) or result.get("migration_output_identical_to_sticky") is True,
    }
    gates["publication_ready_for_10pct_claim"] = all(gates.values())
    result["publication_gates"] = gates

    destination = outdir / "publication_result.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
