#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model.gguf> <decode|prefill>"
  exit 1
fi

MODEL="$1"
PHASE="$2"

if [[ "$PHASE" != "decode" && "$PHASE" != "prefill" ]]; then
  echo "Phase must be decode or prefill" >&2
  exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found. Install linux-tools and rerun." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/results}"
mkdir -p "$OUT_DIR"

THREADS="${THREADS:-0}"
RUNS="${RUNS:-1}"
BENCH_BIN="${BENCH_BIN:-$ROOT_DIR/benchmarks/bench_native}"
PERF_FREQ="${PERF_FREQ:-999}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,context-switches,cpu-migrations,minor-faults,major-faults}"

DECODE_PROMPT_TOKENS="${DECODE_PROMPT_TOKENS:-2048}"
DECODE_GEN_TOKENS="${DECODE_GEN_TOKENS:-256}"
PREFILL_PROMPT_TOKENS="${PREFILL_PROMPT_TOKENS:-1024}"
PREFILL_BATCH="${PREFILL_BATCH:-8}"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Benchmark binary not found at $BENCH_BIN" >&2
  echo "Run: benchmarks/run_llm_perf_matrix.sh <model.gguf>" >&2
  exit 1
fi

if [[ "$PHASE" == "decode" ]]; then
  PROMPT_TOKENS="$DECODE_PROMPT_TOKENS"
  GEN_TOKENS="$DECODE_GEN_TOKENS"
  BATCH_SIZE=1
else
  PROMPT_TOKENS="$PREFILL_PROMPT_TOKENS"
  GEN_TOKENS=1
  BATCH_SIZE="$PREFILL_BATCH"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
STAT_OUT="$OUT_DIR/perf_${PHASE}_${STAMP}.stat.csv"
DATA_OUT="$OUT_DIR/perf_${PHASE}_${STAMP}.data"
REPORT_OUT="$OUT_DIR/perf_${PHASE}_${STAMP}.report.txt"

echo "[config] phase=$PHASE threads=$THREADS runs=$RUNS prompt=$PROMPT_TOKENS gen=$GEN_TOKENS batch=$BATCH_SIZE"
echo "[config] INT4 threading mode: ${DENSECORE_INT4_THREADING_MODE:-<default>}"
echo "[config] prefill flash mask mode: ${DENSECORE_PREFILL_FLASH_MASK_MODE:-<default>}"
echo "[config] rope precomputed mode: ${DENSECORE_ROPE_PRECOMPUTED_MODE:-<default>}"
echo "[config] split thread policy: ${DENSECORE_SPLIT_THREAD_POLICY:-<default>}"

run_cmd=("$BENCH_BIN" "$MODEL" "$PROMPT_TOKENS" "$GEN_TOKENS" "$THREADS" "$RUNS" --batch-size "$BATCH_SIZE" --csv)

echo "[run] perf stat"
LD_LIBRARY_PATH="$ROOT_DIR/build:$ROOT_DIR/core/build:${LD_LIBRARY_PATH:-}" \
DENSECORE_BENCH_MODE=1 \
perf stat -x, -o "$STAT_OUT" -e "$PERF_EVENTS" -- "${run_cmd[@]}" >/dev/null

# perf record may require root or perf_event_paranoid tuning.
echo "[run] perf record"
LD_LIBRARY_PATH="$ROOT_DIR/build:$ROOT_DIR/core/build:${LD_LIBRARY_PATH:-}" \
DENSECORE_BENCH_MODE=1 \
perf record -F "$PERF_FREQ" -g --call-graph dwarf -o "$DATA_OUT" -- "${run_cmd[@]}" >/dev/null

perf report -i "$DATA_OUT" --stdio > "$REPORT_OUT"

echo "Saved stat:   $STAT_OUT"
echo "Saved record: $DATA_OUT"
echo "Saved report: $REPORT_OUT"
