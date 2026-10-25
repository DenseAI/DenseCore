#!/usr/bin/env python3
"""Evaluate paired encoder predictions; uncertainty never implies a quality pass."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from systemone_economics import validate_runtime_dependencies


def _probabilities(values: object, size: int) -> list[float]:
    if not isinstance(values, list) or len(values) != size:
        raise ValueError("probability shape mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("invalid probability")
    if abs(sum(values) - 1) > 1e-5:
        raise ValueError("probabilities must sum to one")
    return values


def load_predictions(path: Path) -> dict:
    """Bind prediction evidence to exact file bytes, ignoring self-declared IDs."""
    raw = path.read_bytes()
    run = json.loads(raw)
    run["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    return run


def quality_target(case: dict, labels: list[str]) -> list[float]:
    """Use authoritative annotated distribution, or one-hot observed class."""
    gold = labels.index(case["gold_label"])
    soft = case.get("source_gold_probabilities")
    if soft is None:
        return [float(i == gold) for i in range(len(labels))]
    if not isinstance(soft, dict) or set(soft) != set(labels):
        raise ValueError("gold probability labels mismatch")
    return _probabilities([soft[label] for label in labels], len(labels))


def _interval(values: list[float], width: float) -> list[float]:
    # Two-sided empirical Bernstein bounds; union bound across both metrics
    # gives at least 95% simultaneous coverage for independent dataset examples.
    # Group all questions for a source example before applying this function.
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean - width, mean + width]
    log_term = math.log(8 / 0.05)
    radius = math.sqrt(2 * statistics.variance(values) * log_term / len(values)) + 7 * width * log_term / (3 * (len(values) - 1))
    return [max(-width / 2, mean - radius), min(width / 2, mean + radius)]


def evaluate(workload: dict, reference: dict, candidate: dict, tuning_ids: set[str], mode: str = "int8") -> dict:
    provenance = workload["provenance"]
    if provenance.get("split") not in ("test", "validation"):
        raise ValueError("final evaluation requires a held-out split")
    for run in (reference, candidate):
        validate_runtime_dependencies(run.get("runtime_dependencies_sha256"))
        if run.get("runtime_dependencies_stable") is not True:
            raise ValueError("prediction runtime dependency stability unverified")
        for key in ("model_sha256", "tokenizer_sha256"):
            if run.get(key) != provenance.get(key):
                raise ValueError(f"{key} mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", run.get("artifact_sha256", "")):
            raise ValueError("run artifact SHA256 required")
        if run.get("workload_sha256") != workload["artifact_sha256"]:
            raise ValueError("workload hash mismatch")
    if reference.get("precision") != "fp32" or candidate.get("precision") != mode:
        raise ValueError("precision mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate.get("implementation_sha256", "")):
        raise ValueError("candidate implementation SHA256 required")
    cases = workload["requests"]
    if not cases:
        raise ValueError("empty evaluation")
    expected = {c["id"] for c in cases}
    if len(expected) != len(cases):
        raise ValueError("duplicate workload ids")
    for c in cases:
        # source_id remains shared by derived questions and variants.
        if c["source_id"] in tuning_ids:
            raise ValueError("tuning/final evaluation leakage")
    by_run = []
    for run in (reference, candidate):
        rows = run.get("predictions", [])
        mapped = {r["id"]: r for r in rows}
        if len(rows) != len(mapped) or set(mapped) != expected:
            raise ValueError("missing, extra, or duplicate predictions")
        by_run.append(mapped)
    ref_acc, cand_acc, ref_brier, cand_brier = [], [], [], []
    grouped: dict[str, list[tuple[float, float]]] = {}
    parity = True
    for case in cases:
        questions = case["request"]["questions"]
        criteria = next(iter(questions.values()))["criteria"]
        labels = list(criteria) if isinstance(criteria, dict) else criteria
        gold = labels.index(case["gold_label"])
        predictions = [run[case["id"]] for run in by_run]
        if any(row.get("labels") != labels for row in predictions):
            raise ValueError("prediction option order mismatch")
        ps = [_probabilities(row.get("probabilities"), len(labels)) for row in predictions]
        accuracy = [float(max(range(len(p)), key=p.__getitem__) == gold) for p in ps]
        target = quality_target(case, labels)
        brier = [sum((value - target[i]) ** 2 for i, value in enumerate(p)) for p in ps]
        ref_acc.append(accuracy[0]); cand_acc.append(accuracy[1])
        ref_brier.append(brier[0]); cand_brier.append(brier[1])
        grouped.setdefault(case["source_id"], []).append((accuracy[0] - accuracy[1], brier[1] - brier[0]))
        if mode == "fp32":
            logits = [row.get("logits") for row in predictions]
            if any(not isinstance(l, list) or len(l) != len(labels) or any(not math.isfinite(v) for v in l) for l in logits):
                raise ValueError("FP32 parity needs finite option logits")
            # Action parity is independent and must already be supplied by its
            # canonical evaluator; this routine never weakens that threshold.
            parity &= max(abs(a-b) for a,b in zip(*logits)) <= 1e-3
            parity &= max(abs(a-b) for a,b in zip(*ps)) <= 1e-4
    deltas = [[statistics.mean(v[i] for v in group) for group in grouped.values()] for i in (0, 1)]
    intervals = [_interval(deltas[0], 2), _interval(deltas[1], 4)]
    if mode in ("int8", "q8_0"):
        status = "pass" if all(ci[1] <= .005 for ci in intervals) else "fail" if any(ci[0] > .005 for ci in intervals) else "unverified"
    else:
        action_status = candidate.get("independent_action_parity")
        status = "fail" if not parity or action_status == "fail" else "pass" if action_status == "pass" else "unverified"
    return {"schema": "densecore.encoder-quality.v1", "status": status, "precision": mode,
            "workload_sha256": workload["artifact_sha256"], "model_sha256": provenance["model_sha256"],
            "tokenizer_sha256": provenance["tokenizer_sha256"], "implementation_sha256": candidate["implementation_sha256"],
            "runtime_dependencies_sha256": candidate["runtime_dependencies_sha256"],
            "reference_runtime_dependencies_sha256": reference["runtime_dependencies_sha256"],
            "runtime_dependencies_stable": True,
            "reference_artifact_sha256": reference["artifact_sha256"], "candidate_artifact_sha256": candidate["artifact_sha256"],
            "examples": len(cases), "independent_sources": len(grouped), "brier_definition": "sum_over_options_squared_error_against_annotated_distribution_else_onehot",
            "reference_accuracy": statistics.mean(ref_acc), "candidate_accuracy": statistics.mean(cand_acc),
            "reference_brier": statistics.mean(ref_brier), "candidate_brier": statistics.mean(cand_brier),
            "accuracy_drop": statistics.mean(deltas[0]), "brier_worsening": statistics.mean(deltas[1]),
            "accuracy_drop_95_interval": intervals[0], "brier_worsening_95_interval": intervals[1],
            "uncertainty_method": "paired source-cluster empirical Bernstein; simultaneous 95%; independent sources assumed",
            "limits": {"accuracy_drop": .005, "brier_worsening": .005},
            "independent_fp32_parity": {"status": status if mode == "fp32" else "not_evaluated", "logit_atol": .001, "probability_atol": .0001}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("workload", "reference", "candidate", "tuning-ids", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=("fp32", "int8", "q8_0"), default="int8")
    a = p.parse_args()
    workload = json.loads(a.workload.read_text())
    workload["artifact_sha256"] = hashlib.sha256(a.workload.read_bytes()).hexdigest()
    tuning = json.loads(a.tuning_ids.read_text())
    if not isinstance(tuning, list) or any(not isinstance(x, str) for x in tuning):
        p.error("tuning-ids must be an explicit JSON array of source ids (empty only if no tuning)")
    result = evaluate(workload, load_predictions(a.reference), load_predictions(a.candidate), set(tuning), a.mode)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"])
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
