#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model.gguf>"
  echo "Env overrides:"
  echo "  RUNS=2 GEN_TOKENS=128"
  echo "  BATCH_SIZES=\"1 4 8 16 32\""
  echo "  CONTEXT_LENS=\"128 1024 4096\""
  echo "  THREAD_GRID=\"4 8 16 32 64 128\""
  echo "  BENCH_BIN=benchmarks/bench_native OUT_DIR=benchmarks/results"
  exit 1
fi

MODEL="$1"
RUNS="${RUNS:-2}"
GEN_TOKENS="${GEN_TOKENS:-128}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32}"
CONTEXT_LENS="${CONTEXT_LENS:-128 1024 4096}"
THREAD_GRID="${THREAD_GRID:-4 8 16 32 64 128}"
BENCH_BIN="${BENCH_BIN:-benchmarks/bench_native}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/results}"
mkdir -p "$OUT_DIR"

stamp="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="$OUT_DIR/decode_batch_matrix_${stamp}.csv"

ensure_bench_native() {
  if [[ -x "$ROOT_DIR/$BENCH_BIN" ]]; then
    return
  fi
  if [[ ! -f "$ROOT_DIR/benchmarks/bench_native.cpp" ]]; then
    echo "bench_native.cpp not found" >&2
    exit 1
  fi
  echo "[build] Compiling bench_native"
  g++ -O2 -std=c++17 "$ROOT_DIR/benchmarks/bench_native.cpp" \
      -I"$ROOT_DIR/core/include" -L"$ROOT_DIR/build" -ldensecore -lpthread \
      -o "$ROOT_DIR/benchmarks/bench_native"
}

run_decode_case() {
  local context_len="$1"
  local batch_size="$2"
  local threads="$3"

  local cmd=(
    "$ROOT_DIR/$BENCH_BIN"
    "$MODEL"
    "$context_len"
    "$GEN_TOKENS"
    "$threads"
    "$RUNS"
    --batch-size "$batch_size"
    --csv
  )
  local line
  line="$(
    LD_LIBRARY_PATH="$ROOT_DIR/build:$ROOT_DIR/core/build:${LD_LIBRARY_PATH:-}" \
      DENSECORE_BENCH_MODE=1 \
      "${cmd[@]}" | tail -n1
  )"

  IFS=',' read -r ttft_ms decode_tps tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$line"
  if [[ -z "${ttft_ms:-}" || -z "${decode_tps:-}" ]]; then
    echo "Failed to parse benchmark output: $line" >&2
    exit 1
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$threads" "$batch_size" "$context_len" "$GEN_TOKENS" "$ttft_ms" "$decode_tps" "$tokens" \
    "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" >> "$OUT_CSV"

  echo "[decode] threads=$threads batch=$batch_size context=$context_len gen=$GEN_TOKENS tps=$decode_tps"
}

ensure_bench_native

echo "threads,batch_size,context_len,gen_tokens,ttft_ms,decode_tps,tokens,itl_avg_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms" > "$OUT_CSV"

for threads in $THREAD_GRID; do
  for context_len in $CONTEXT_LENS; do
    for batch_size in $BATCH_SIZES; do
      run_decode_case "$context_len" "$batch_size" "$threads"
    done
  done
done

echo "Saved: $OUT_CSV"
