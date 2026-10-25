#!/usr/bin/env python3
"""Build/run a pinned standalone llama NextN oracle under DenseCore's shared lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION = "74bca3950054eed1795f525cb9cf06bfe796492d"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", required=True, type=Path)
    parser.add_argument("--build-lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--hidden", type=Path, help="exact token-count × model-embedding little-endian float32 matrix")
    parser.add_argument("--tokens", help="comma-separated token IDs, positioned at 0..N-1")
    parser.add_argument("--chunk-size", type=int, default=1, choices=range(1, 17))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.build_only and not all((args.model, args.model_sha256, args.hidden, args.tokens)):
        parser.error("run requires --model --model-sha256 --hidden --tokens")
    root = args.llama_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"oracle revision mismatch: {revision}")
    # A pinned commit does not prove a dirty source tree is pinned.
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("oracle source has tracked changes")
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("llama_nextn_logits.cpp").resolve()
    binary = args.out.resolve() / "llama-nextn-logits"
    libdir = root / "build/bin"
    command = ["c++", "-std=c++17", "-O2", str(source), "-I" + str(root / "include"),
               "-I" + str(root / "src"), "-I" + str(root / "ggml/include"),
               "-L" + str(libdir), "-Wl,-rpath," + str(libdir), "-lllama", "-o", str(binary)]
    with args.build_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = {"oracle_revision": revision, "source_sha256": sha256(source), "build_command": command,
                    "native_byte_order": sys.byteorder, "qualification": "independent oracle; no DenseCore head comparison yet"}
        if sys.byteorder != "little":
            raise RuntimeError("probe artifact protocol currently requires little-endian host")
        if not args.build_only:
            model_hash = sha256(args.model)
            if model_hash != args.model_sha256.lower():
                raise RuntimeError("model SHA256 mismatch")
            manifest.update(model=str(args.model.resolve()), model_sha256=model_hash,
                            hidden_sha256=sha256(args.hidden), tokens=args.tokens)
        with (args.out / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["binary_sha256"] = sha256(binary)
        dependencies = sorted(libdir.glob("libllama.so*")) + sorted(libdir.glob("libggml*.so*"))
        manifest["dependency_sha256"] = {str(path.resolve()): sha256(path) for path in dependencies if path.is_file()}
        if not args.build_only:
            run = [str(binary), str(args.model.resolve()), str(args.hidden.resolve()), args.tokens,
                   str(args.out.resolve() / "oracle"), str(args.chunk_size), str(args.threads)]
            env = dict(os.environ)
            # Never inject DenseCore ggml into the separate llama oracle process.
            env.pop("LD_PRELOAD", None)
            env["LD_LIBRARY_PATH"] = str(libdir)
            manifest["run_command"] = run
            with (args.out / "run.log").open("w") as log:
                result = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT, env=env)
            manifest["exit_code"] = result.returncode
            if result.returncode == 0:
                manifest["outputs"] = {name: sha256(args.out / name) for name in
                                       ("oracle.logits.f32", "oracle.hidden.f32", "oracle.json")}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if manifest.get("exit_code", 0):
            raise RuntimeError("oracle execution failed; see run.log and manifest.json")

if __name__ == "__main__":
    main()
