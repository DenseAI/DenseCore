#!/usr/bin/env python3
"""Verify pinned reference bytes and reconstruct shootout requests without executing upstream code."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request


MANIFEST = Path(__file__).with_name("laya-cuda-bench.manifest.json")


def verify(root, manifest, download=False):
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source path")
        target = root / relative
        if not target.exists() and download:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"download hash mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        data = target.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"source mismatch: {relative}")


def requests(root, bucket, batch):
    """Reproduce workload.load_workload(seed=0) + shootout.make_requests exactly."""
    fixture = root / "fixtures/workload"
    states = [json.loads(line) for line in (fixture / "states.jsonl").read_text().splitlines() if line.strip()]
    questions = json.loads((fixture / "questions.json").read_text())
    lo, hi = {"short": (0, 80), "long": (400, 472)}[bucket]
    selected = [state for state in states if lo <= state["tokens"]["laya"] <= hi]
    random.Random(0).shuffle(selected)
    if len(selected) < 32:
        raise ValueError("reference bucket has fewer than 32 states")
    qids = list(questions)
    for i in range(batch):
        state = selected[i % len(selected)]
        qid = qids[i % len(qids)]
        yield {"id": f"external-laya-{bucket}-b{batch}-r{i}", "state": state["text"],
               "questions": {qid: questions[qid]}, "source_state_id": state["id"],
               "workload_bucket": bucket, "external_batch": batch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "download", "requests"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bucket", choices=("short", "long"))
    parser.add_argument("--batch", type=int, choices=(1, 16, 64, 256))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    verify(args.root, manifest, args.command == "download")
    if args.command == "requests":
        if args.bucket is None or args.batch is None or args.output is None:
            parser.error("requests requires --bucket, --batch and --output")
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in requests(args.root, args.bucket, args.batch)))
    else:
        print(json.dumps({"status": "verified", "files": len(manifest["files"]),
                          "revision": manifest["source_revision"]}))


if __name__ == "__main__":
    main()
