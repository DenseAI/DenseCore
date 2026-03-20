#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model.gguf>"
  echo "Env overrides:"
  echo "  THREADS=0 RUNS=3"
  echo "  DECODE_PROMPT_TOKENS=2048 DECODE_GEN_TOKENS=256"
  echo "  PREFILL_SEQ_LENS=\"512 1024 2048\" PREFILL_BATCHES=\"4 8 16\""
  exit 1
fi

MODEL="$1"
THREADS="${THREADS:-0}"
RUNS="${RUNS:-3}"
DECODE_PROMPT_TOKENS="${DECODE_PROMPT_TOKENS:-2048}"
DECODE_GEN_TOKENS="${DECODE_GEN_TOKENS:-256}"
PREFILL_SEQ_LENS="${PREFILL_SEQ_LENS:-512 1024 2048}"
PREFILL_BATCHES="${PREFILL_BATCHES:-4 8 16}"
BENCH_BIN="${BENCH_BIN:-benchmarks/bench_native}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/results}"
mkdir -p "$OUT_DIR"

stamp="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="$OUT_DIR/perf_matrix_${stamp}.csv"

ensure_bench_native() {
  if [[ -x "$BENCH_BIN" ]]; then
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

run_bench_csv() {
  local prompt_tokens="$1"
  local gen_tokens="$2"
  local batch_size="$3"

  local cmd=("$ROOT_DIR/$BENCH_BIN" "$MODEL" "$prompt_tokens" "$gen_tokens" "$THREADS" "$RUNS" --batch-size "$batch_size" --csv)
  local line
  line="$(LD_LIBRARY_PATH="$ROOT_DIR/build:$ROOT_DIR/core/build:${LD_LIBRARY_PATH:-}" \
          DENSECORE_BENCH_MODE=1 \
          "${cmd[@]}" | tail -n1)"
  echo "$line"
}

append_row() {
  local phase="$1"
  local prompt_tokens="$2"
  local gen_tokens="$3"
  local batch_size="$4"

  local line
  line="$(run_bench_csv "$prompt_tokens" "$gen_tokens" "$batch_size")"
  IFS=',' read -r ttft_ms decode_tps tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$line"

  if [[ -z "${ttft_ms:-}" || -z "${decode_tps:-}" ]]; then
    echo "Failed to parse benchmark line: $line" >&2
    exit 1
  fi

  local prefill_tps=""
  if [[ "$phase" == "prefill" ]]; then
    prefill_tps="$(awk -v p="$prompt_tokens" -v b="$batch_size" -v t="$ttft_ms" 'BEGIN { if (t <= 0) { print 0.0 } else { printf "%.2f", (p*b)/(t/1000.0) } }')"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$phase" "$batch_size" "$prompt_tokens" "$gen_tokens" "$THREADS" \
    "$ttft_ms" "$decode_tps" "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" "$prefill_tps" >> "$OUT_CSV"

  if [[ "$phase" == "prefill" ]]; then
    echo "[prefill] batch=$batch_size seq=$prompt_tokens ttft_ms=$ttft_ms prefill_tps=$prefill_tps"
  else
    echo "[decode]  batch=$batch_size prompt=$prompt_tokens gen=$gen_tokens ttft_ms=$ttft_ms decode_tps=$decode_tps"
  fi
}

ensure_bench_native

echo "phase,batch_size,prompt_tokens,gen_tokens,threads,ttft_ms,decode_tps,itl_avg_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms,prefill_tps" > "$OUT_CSV"

echo "[run] decode baseline (batch=1, long context)"
append_row "decode" "$DECODE_PROMPT_TOKENS" "$DECODE_GEN_TOKENS" 1

echo "[run] prefill matrix (batch>=4, varying seq lengths)"
for b in $PREFILL_BATCHES; do
  for s in $PREFILL_SEQ_LENS; do
    append_row "prefill" "$s" 1 "$b"
  done
done

echo "Saved: $OUT_CSV"
