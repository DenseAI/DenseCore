#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-numa}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/bin/densecore-server}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF under test}"
THREADS="${THREADS:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18082}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
OUTDIR="${OUTDIR:-$ROOT_DIR/benchmarks/results/numa_dense_ab_$(date -u +%Y%m%dT%H%M%SZ)}"
CONVO="$ROOT_DIR/benchmarks/numa/convo_e2e.py"
CASE_FILTER="${CASE_FILTER:-dense_off,dense_on}"
PRECOMPUTED_MODEL_SHA256="${PRECOMPUTED_MODEL_SHA256:-}"

mkdir -p "$OUTDIR"
for path in "$SERVER_BIN" "$BUILD_DIR/libdensecore.so" "$MODEL_PATH" "$CONVO"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done

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

lscpu > "$OUTDIR/lscpu.txt"
numactl --hardware > "$OUTDIR/numactl-hardware.txt"
sha256sum "$SERVER_BIN" "$BUILD_DIR/libdensecore.so" "$CONVO" \
  "${BASH_SOURCE[0]}" \
  "$ROOT_DIR/core/src/models/model_prompt_templates.cpp" \
  "$ROOT_DIR/server/internal/service/chat_prompt.go" > "$OUTDIR/artifact.sha256"
if [[ -n "$PRECOMPUTED_MODEL_SHA256" ]]; then
  printf '%s  %s\n' "$PRECOMPUTED_MODEL_SHA256" "$MODEL_PATH" >> "$OUTDIR/artifact.sha256"
else
  sha256sum "$MODEL_PATH" >> "$OUTDIR/artifact.sha256"
fi

case_selected() {
  [[ ",$CASE_FILTER," == *",$1,"* ]]
}

case_selected dense_off && run_dense dense_off 1 0
case_selected dense_on && run_dense dense_on 0 0
case_selected dense_migration && run_dense dense_migration 0 1

python3 - "$OUTDIR" "$CASE_FILTER" <<'PY'
import json
import pathlib
import sys

outdir = pathlib.Path(sys.argv[1])
rows = {}
for label in sys.argv[2].split(","):
    label = label.strip()
    if not label:
        continue
    data = json.loads((outdir / f"{label}.json").read_text(encoding="utf-8"))
    rows[label] = {
        "turns": len(data["turns"]),
        "ok_turns": data["ok_turns"],
        "completion_tokens": [turn["completion_tokens"] for turn in data["turns"]],
        "aggregate_e2e_tok_per_s": data["aggregate_tok_per_s"],
        "problems": [turn["problems"] for turn in data["turns"]],
    }
(outdir / "focused_result.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY

find "$OUTDIR" -type f ! -name result-files.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTDIR/result-files.sha256"
echo "focused DenseCore AB artifacts: $OUTDIR"
