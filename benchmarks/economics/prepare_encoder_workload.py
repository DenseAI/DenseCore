#!/usr/bin/env python3
"""Freeze naturally sized, already-tokenized encoder requests without padding.

Token IDs must come from the same input builder as the measured implementation.
This tool deliberately does not approximate a model's tokenizer using word counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

DATASETS = {"fancyzhx/ag_news", "LocalLLaMA/typed-decisions"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def prepare(records: list[dict], provenance: dict, lower_fraction: float = 0.75, quality_all_lengths: bool = False, excluded_source_ids: set[str] | None = None) -> dict:
    """Select natural lengths in (previous bucket, bucket], never alter inputs."""
    if not 0 < lower_fraction <= 1:
        raise ValueError("lower_fraction must be in (0, 1]")
    for key in ("model_sha256", "tokenizer_sha256", "source_sha256"):
        if not sha(provenance.get(key)):
            raise ValueError(f"missing pinned {key}")
    if provenance.get("dataset") not in DATASETS:
        raise ValueError("unsupported dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance.get("revision", "")):
        raise ValueError("dataset revision must be an immutable 40-character commit")
    if provenance.get("split") not in ("train", "test", "validation"):
        raise ValueError("explicit source split required")
    selected, rejected, seen = [], [], set()
    excluded_source_ids = excluded_source_ids or set()
    for row in records:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("duplicate or invalid case id")
        seen.add(rid)
        if row.get("source_id", rid) in excluded_source_ids:
            rejected.append({"id": rid, "reason": "excluded_tuning_or_diagnostic_source"})
            continue
        ids = row.get("token_ids")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"{rid}: actual token_ids required")
        # Require affirmative evidence rather than accepting an omitted counter.
        if any(key not in row for key in ("padding_tokens", "truncated", "dropped_tokens")):
            raise ValueError(f"{rid}: explicit padding/truncation accounting required")
        if row["padding_tokens"] != 0 or row["truncated"] is not False or row["dropped_tokens"] != 0:
            rejected.append({"id": rid, "reason": "padding_or_truncation"})
            continue
        attention_mask = row.get("attention_mask")
        if (not isinstance(attention_mask, list) or len(attention_mask) != len(ids)
                or any(type(value) is not int or value != 1 for value in attention_mask)):
            raise ValueError(f"{rid}: explicit all-one attention_mask required for unpadded native input")
        keys = row.get("token_keys")
        if not isinstance(keys, list) or len(keys) != len(ids) or any(k is not None and (not isinstance(k, str) or not k) for k in keys):
            raise ValueError(f"{rid}: actual input-builder token_keys required")
        useful = len(set(k for k in keys if k is not None))
        if not useful or useful != sum(k is not None for k in keys):
            raise ValueError(f"{rid}: useful token keys must be nonempty and unique within one question")
        request = row.get("request", {})
        questions = request.get("questions", {})
        if len(questions) != 1:
            rejected.append({"id": rid, "reason": "not_one_question"})
            continue
        question = next(iter(questions.values()))
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, (dict, list)) or len(criteria) != 4:
            rejected.append({"id": rid, "reason": "not_four_option_choice"})
            continue
        options = list(criteria) if isinstance(criteria, dict) else criteria
        if len(set(options)) != 4 or row.get("gold_label") not in options:
            raise ValueError(f"{rid}: invalid label/options")
        bucket = next((n for n in (128, 256, 512) if n * lower_fraction < len(ids) <= n), None)
        if bucket is None and not (quality_all_lengths and len(ids) <= 512):
            rejected.append({"id": rid, "reason": "outside_natural_length_buckets", "input_tokens": len(ids)})
            continue
        selected.append({**row, "bucket_tokens": bucket, "useful_input_tokens": useful, "compute_input_tokens": len(ids),
                         "source_id": row.get("source_id", rid), "dataset": provenance["dataset"], "source_split": provenance["split"]})
    for case in selected:
        case["sequences"] = [{"token_ids": case["token_ids"], "token_keys": case["token_keys"], "attention_mask": case["attention_mask"],
                              "markers": case.get("markers", []), "question_type": 0}]
    return {"schema": "systemone-workload-v1", "requests": selected, "provenance": provenance,
            "purpose": "final_quality_holdout" if quality_all_lengths else "natural_length_performance",
            "excluded_source_ids": sorted(excluded_source_ids),
            "length_policy": {"buckets": [128, 256, 512], "lower_fraction_exclusive": lower_fraction,
                              "padding": False, "truncation": False, "result_cache": False,
                              "quality_all_lengths": quality_all_lengths},
            "rejected": rejected,
            "bucket_status": {str(n): "ready" if any(c["bucket_tokens"] == n for c in selected) else "unavailable" for n in (128, 256, 512)}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenized-cases", type=Path, required=True, help="JSONL from the measured model's actual input builder")
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="original immutable dataset file")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quality-all-lengths", action="store_true")
    p.add_argument("--exclude-source-ids", type=Path, help="explicit JSON source IDs used by tuning/diagnostics")
    a = p.parse_args()
    provenance = json.loads(a.provenance.read_text())
    if provenance.get("source_sha256") != digest(a.source):
        p.error("source checksum does not match provenance")
    provenance["tokenized_cases_sha256"] = digest(a.tokenized_cases)
    excluded = json.loads(a.exclude_source_ids.read_text()) if a.exclude_source_ids else []
    if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
        p.error("exclude-source-ids must contain a JSON array of source ids")
    if a.quality_all_lengths and a.exclude_source_ids is None:
        p.error("quality-all-lengths requires an explicit exclude-source-ids artifact")
    result = prepare([json.loads(line) for line in a.tokenized_cases.read_text().splitlines() if line.strip()], provenance,
                     quality_all_lengths=a.quality_all_lengths, excluded_source_ids=set(excluded))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
