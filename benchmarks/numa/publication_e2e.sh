#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-numa}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/bin/densecore-server}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the exact GGUF used by both engines}"
THREADS="${THREADS:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18082}"
RUN_MIGRATION="${RUN_MIGRATION:-1}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
OUTDIR="${OUTDIR:-$ROOT_DIR/benchmarks/results/numa_publication_$(date -u +%Y%m%dT%H%M%SZ)}"
CONVO="$ROOT_DIR/benchmarks/numa/convo_e2e.py"
SUMMARIZER="$ROOT_DIR/benchmarks/numa/summarize_publication_e2e.py"
PREFLIGHT="$ROOT_DIR/benchmarks/numa/tier0_numa_validate.sh"

mkdir -p "$OUTDIR"
for path in "$SERVER_BIN" "$LLAMA_SERVER_BIN" "$MODEL_PATH" "$CONVO" "$SUMMARIZER" "$PREFLIGHT"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done
[[ -f "$BUILD_DIR/libdensecore.so" ]] || { echo "missing $BUILD_DIR/libdensecore.so" >&2; exit 1; }

NODE_COUNT="$(find /sys/devices/system/node -maxdepth 1 -type d -name 'node[0-9]*' | wc -l)"
[[ "$NODE_COUNT" -ge 2 ]] || { echo "requires at least two NUMA nodes; found $NODE_COUNT" >&2; exit 1; }

# A declared vNUMA topology is not enough. This compiles and runs the measured
# local-vs-remote bandwidth/latency probe, verifies page placement, and exits
# before any build or model load. Publication runs fail closed on OS balancing.
mkdir -p "$OUTDIR/preflight"
PREFLIGHT_ONLY=1 STRICT_PREFLIGHT=1 DO_BUILD=0 \
  ROOT_DIR="$ROOT_DIR" SERVER_BIN="$SERVER_BIN" MODEL_PATH="$MODEL_PATH" \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" THREADS="$THREADS" \
  OUTDIR="$OUTDIR/preflight" \
  bash "$PREFLIGHT"

SOURCE_COMMIT="${SOURCE_COMMIT:-}"
SOURCE_PATCH_PATH="${SOURCE_PATCH_PATH:-}"
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT_DIR" rev-parse HEAD > "$OUTDIR/source_commit.txt"
  git -C "$ROOT_DIR" diff --binary -- core server benchmarks/numa > "$OUTDIR/source.patch"
  SOURCE_DIRTY="$(git -C "$ROOT_DIR" status --porcelain | wc -l)"
else
  [[ -n "$SOURCE_COMMIT" && -f "$SOURCE_PATCH_PATH" ]] || {
    echo "non-git source tree requires SOURCE_COMMIT and SOURCE_PATCH_PATH" >&2
    exit 1
  }
  printf '%s\n' "$SOURCE_COMMIT" > "$OUTDIR/source_commit.txt"
  cp "$SOURCE_PATCH_PATH" "$OUTDIR/source.patch"
  SOURCE_DIRTY=-1
fi
sha256sum "$OUTDIR/source.patch" > "$OUTDIR/source.patch.sha256"

# Preserve the executable validation logic itself. `git diff` cannot contain
# untracked harness files, so the result would otherwise be impossible to audit.
mkdir -p "$OUTDIR/harness"
for path in "$CONVO" "$SUMMARIZER" "$PREFLIGHT" \
            "$ROOT_DIR/benchmarks/numa/measure.py" "${BASH_SOURCE[0]}"; do
  cp "$path" "$OUTDIR/harness/$(basename "$path")"
done
sha256sum "$OUTDIR"/harness/* > "$OUTDIR/harness.sha256"
if [[ -n "${PRECOMPUTED_ARTIFACT_SHA256:-}" ]]; then
  [[ -s "$PRECOMPUTED_ARTIFACT_SHA256" ]] || {
    echo "PRECOMPUTED_ARTIFACT_SHA256 is missing or empty: $PRECOMPUTED_ARTIFACT_SHA256" >&2
    exit 1
  }
  cp "$PRECOMPUTED_ARTIFACT_SHA256" "$OUTDIR/artifact.sha256"
else
  sha256sum "$SERVER_BIN" "$BUILD_DIR/libdensecore.so" "$LLAMA_SERVER_BIN" "$MODEL_PATH" \
    > "$OUTDIR/artifact.sha256"
fi
lscpu > "$OUTDIR/lscpu.txt"
numactl --hardware > "$OUTDIR/numactl-hardware.txt"
uname -a > "$OUTDIR/uname.txt"
cp /etc/os-release "$OUTDIR/os-release.txt"
ldd "$SERVER_BIN" > "$OUTDIR/densecore-server.ldd.txt" 2>&1 || true
ldd "$BUILD_DIR/libdensecore.so" > "$OUTDIR/libdensecore.ldd.txt" 2>&1 || true
ldd "$LLAMA_SERVER_BIN" > "$OUTDIR/llama-server.ldd.txt" 2>&1 || true
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  grep -E '^(CMAKE_(BUILD_TYPE|CXX_COMPILER|CXX_FLAGS)|GGML_|DENSECORE_|HWLOC_|NUMA_)' \
    "$BUILD_DIR/CMakeCache.txt" > "$OUTDIR/cmake-cache-relevant.txt" || true
fi
for distance in /sys/devices/system/node/node*/distance; do
  printf '%s: ' "$distance"
  cat "$distance"
done > "$OUTDIR/numa-distances.txt"

python3 - "$OUTDIR" "$ROOT_DIR" "$BUILD_DIR" "$SERVER_BIN" "$LLAMA_SERVER_BIN" "$MODEL_PATH" \
  "$THREADS" "$LLAMA_CTX" "$RUN_MIGRATION" "$SOURCE_DIRTY" <<'PY'
import json, pathlib, sys

outdir, root, build, server, llama, model, threads, llama_ctx, run_migration, source_dirty = sys.argv[1:]
def read(path):
    return pathlib.Path(path).read_text(encoding="utf-8").strip()

source_dirty_count = int(source_dirty)
source_control_state = "git_worktree" if source_dirty_count >= 0 else "non_git_source_archive"
manifest = {
    "source_commit": read(pathlib.Path(outdir) / "source_commit.txt"),
    "source_control_state": source_control_state,
    "source_dirty": source_dirty_count != 0 if source_dirty_count >= 0 else None,
    "source_dirty_entry_count": source_dirty_count if source_dirty_count >= 0 else None,
    "source_patch_sha256": read(pathlib.Path(outdir) / "source.patch.sha256").split()[0],
    "root_dir": root,
    "build_dir": build,
    "densecore_server": server,
    "llama_server": llama,
    "model_path": model,
    "threads": int(threads),
    "llama_ctx": int(llama_ctx),
    "turns_per_condition": 2,
    "run_migration": run_migration == "1",
    "numa_preflight": "measured_penalty_passed",
    "artifact_sha256": read(pathlib.Path(outdir) / "artifact.sha256").splitlines(),
    "harness_sha256": read(pathlib.Path(outdir) / "harness.sha256").splitlines(),
    "conditions": {
        "dense_off": {
            "binary": server,
            "library": str(pathlib.Path(build) / "libdensecore.so"),
            "placement": "round_robin",
            "sticky": "disabled_by_debug",
            "page_migration": False,
        },
        "dense_on": {
            "binary": server,
            "library": str(pathlib.Path(build) / "libdensecore.so"),
            "placement": "round_robin",
            "sticky": "enabled",
            "page_migration": False,
        },
        "dense_migration": {
            "binary": server,
            "library": str(pathlib.Path(build) / "libdensecore.so"),
            "placement": "round_robin",
            "sticky": "enabled",
            "page_migration": True,
        },
        "llama": {
            "binary": llama,
            "argv": ["-m", model, "-t", threads, "-tb", threads, "-c", llama_ctx,
                     "--numa", "distribute", "--jinja"],
        },
    },
}
pathlib.Path(outdir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  sleep 2
}
trap stop_server EXIT

wait_url() {
  local url="$1"
  for _ in $(seq 1 600); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

run_dense() {
  local label="$1" debug_disable="$2" migration="$3"
  local log="$OUTDIR/$label.server.log"
  env LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      HOST="$HOST" PORT="$PORT" THREADS="$THREADS" MAIN_MODEL_PATH="$MODEL_PATH" \
      DENSECORE_AGENT_PROMPT_CACHE=off DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE=1 \
      DENSECORE_MODEL_LOAD_STRATEGY=force \
      DENSECORE_MAX_SEQ_LEN="$LLAMA_CTX" DENSECORE_MAX_NUM_SEQS=1 \
      DENSECORE_NUMA_WEIGHTS=round_robin DENSECORE_NUMA_EXPERT_PARTITION=1 \
      DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY="$debug_disable" \
      DENSECORE_MOE_ENABLE_PAGE_MIGRATION="$migration" \
      DENSECORE_MOE_REBALANCE_INTERVAL_MS=250 DENSECORE_MOE_REBALANCE_TOP_K=8 \
      DENSECORE_QWEN36_PROFILE=1 \
      "$SERVER_BIN" serve --grpc=false > "$log" 2>&1 &
  SERVER_PID=$!
  wait_url "http://$HOST:$PORT/health/startup" || {
    echo "$label failed startup; see $log" >&2
    return 1
  }
  cat "/proc/$SERVER_PID/maps" > "$OUTDIR/$label.proc-maps.txt" 2>/dev/null || true
  python3 "$CONVO" --url "http://$HOST:$PORT/v1/chat/completions" --label "$label" \
    --turns 2 --max-tokens 300 --no-thinking --out "$OUTDIR/$label.json" || true
  stop_server
}

run_llama() {
  local log="$OUTDIR/llama.server.log"
  "$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --host "$HOST" --port "$PORT" \
    -t "$THREADS" -tb "$THREADS" -c "$LLAMA_CTX" --numa distribute --jinja > "$log" 2>&1 &
  SERVER_PID=$!
  wait_url "http://$HOST:$PORT/v1/models" || {
    echo "llama failed startup; see $log" >&2
    return 1
  }
  cat "/proc/$SERVER_PID/maps" > "$OUTDIR/llama.proc-maps.txt" 2>/dev/null || true
  python3 "$CONVO" --url "http://$HOST:$PORT/v1/chat/completions" --label llama \
    --turns 2 --max-tokens 300 --no-thinking --out "$OUTDIR/llama.json" || true
  stop_server
}

run_dense dense_off 1 0
run_dense dense_on 0 0
if [[ "$RUN_MIGRATION" == "1" ]]; then
run_dense dense_migration 0 1
fi
run_llama
summary_status=0
python3 "$SUMMARIZER" --outdir "$OUTDIR" | tee "$OUTDIR/publication_result.stdout.json" || summary_status=$?
find "$OUTDIR" -type f ! -name 'result-files.sha256' -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTDIR/result-files.sha256"
echo "publication artifacts: $OUTDIR"
exit "$summary_status"
