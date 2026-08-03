#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-numa}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/bin/densecore-server}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the shared GGUF}"
MODEL_SHA256="${MODEL_SHA256:?MODEL_SHA256 must be a previously verified digest}"
PROMPTS="${PROMPTS:-$ROOT_DIR/benchmarks/numa/decode_claim_prompt.txt}"
OUTDIR="${OUTDIR:-$ROOT_DIR/benchmarks/results/numa_fair_decode_$(date -u +%Y%m%dT%H%M%SZ)}"
THREADS="${THREADS:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18082}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
MEASURE="$ROOT_DIR/benchmarks/numa/measure.py"
PREFLIGHT="$ROOT_DIR/benchmarks/numa/tier0_numa_validate.sh"

mkdir -p "$OUTDIR"
for path in "$SERVER_BIN" "$BUILD_DIR/libdensecore.so" "$LLAMA_SERVER_BIN" "$MODEL_PATH" "$PROMPTS" "$MEASURE" "$PREFLIGHT"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done
[[ $(find /sys/devices/system/node -maxdepth 1 -type d -name 'node[0-9]*' | wc -l) -ge 2 ]] || {
  echo "requires at least two NUMA nodes" >&2
  exit 1
}

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

PREFLIGHT_ONLY=1 STRICT_PREFLIGHT=1 DO_BUILD=0 \
  ROOT_DIR="$ROOT_DIR" BUILD_DIR="$BUILD_DIR" SERVER_BIN="$SERVER_BIN" \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" MODEL_PATH="$MODEL_PATH" \
  THREADS="$THREADS" OUTDIR="$OUTDIR/preflight" \
  bash "$PREFLIGHT"

wait_url() {
  local url="$1"
  for _ in $(seq 1 600); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

run_measure() {
  local label="$1"
  python3 "$MEASURE" --url "http://$HOST:$PORT" --model local --prompts "$PROMPTS" \
    --label "$label" --max-tokens 128 --min-tokens 96 --repeats 2 \
    --prefill-repeats 1 --warmup 1 --temperature 0 --no-thinking \
    --out "$OUTDIR/$label.json"
}

env LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    HOST="$HOST" PORT="$PORT" THREADS="$THREADS" MAIN_MODEL_PATH="$MODEL_PATH" \
    DENSECORE_AGENT_PROMPT_CACHE=off DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE=1 \
    DENSECORE_MODEL_LOAD_STRATEGY=force DENSECORE_MAX_SEQ_LEN="$LLAMA_CTX" DENSECORE_MAX_NUM_SEQS=1 \
    DENSECORE_NUMA_WEIGHTS=round_robin DENSECORE_NUMA_EXPERT_PARTITION=1 \
    DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY=0 DENSECORE_MOE_ENABLE_PAGE_MIGRATION=0 \
    DENSECORE_QWEN36_PROFILE=1 \
    "$SERVER_BIN" serve --grpc=false > "$OUTDIR/dense_on.server.log" 2>&1 &
SERVER_PID=$!
wait_url "http://$HOST:$PORT/health/startup"
cat "/proc/$SERVER_PID/maps" > "$OUTDIR/dense_on.proc-maps.txt" 2>/dev/null || true
run_measure dense_on
stop_server

"$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --host "$HOST" --port "$PORT" \
  -t "$THREADS" -tb "$THREADS" -c "$LLAMA_CTX" --numa distribute --jinja \
  > "$OUTDIR/llama.server.log" 2>&1 &
SERVER_PID=$!
wait_url "http://$HOST:$PORT/v1/models"
cat "/proc/$SERVER_PID/maps" > "$OUTDIR/llama.proc-maps.txt" 2>/dev/null || true
run_measure llama_numa_distribute
stop_server

claim_status=0
python3 - "$OUTDIR" <<'PY' || claim_status=$?
import json
import pathlib
import sys

outdir = pathlib.Path(sys.argv[1])
dense = json.loads((outdir / "dense_on.json").read_text(encoding="utf-8"))
llama = json.loads((outdir / "llama_numa_distribute.json").read_text(encoding="utf-8"))
speedup = (dense["decode_tps_median"] / llama["decode_tps_median"] - 1.0) * 100.0
result = {
    "schema_version": 1,
    "metric": "external non-stream usage-token decode wall-clock after identical max_tokens=1 prefill probe",
    "scored_repeats_per_engine": 2,
    "dense_on_decode_tps_median": dense["decode_tps_median"],
    "llama_numa_distribute_decode_tps_median": llama["decode_tps_median"],
    "dense_on_speedup_pct": round(speedup, 2),
    "claim_gate_at_least_10pct": speedup >= 10.0,
    "quality_gate": dense["quality_gate"] == llama["quality_gate"],
}
(outdir / "claim_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["claim_gate_at_least_10pct"] or not result["quality_gate"]:
    raise SystemExit(2)
PY

lscpu > "$OUTDIR/lscpu.txt"
numactl --hardware > "$OUTDIR/numactl-hardware.txt"
sha256sum "$SERVER_BIN" "$BUILD_DIR/libdensecore.so" "$LLAMA_SERVER_BIN" "$MEASURE" "$PREFLIGHT" "$PROMPTS" \
  "${BASH_SOURCE[0]}" > "$OUTDIR/artifact.sha256"
printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_PATH" >> "$OUTDIR/artifact.sha256"
find "$OUTDIR" -type f ! -name result-files.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTDIR/result-files.sha256"
echo "fair decode claim artifacts: $OUTDIR"
exit "$claim_status"
