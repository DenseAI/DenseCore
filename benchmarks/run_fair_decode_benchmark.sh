#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

THREADS="${THREADS:-8}"
PROMPT_TOK="${PROMPT_TOK:-64}"
GEN_TOK="${GEN_TOK:-64}"
REPEATS="${REPEATS:-5}"
BATCHES="${BATCHES:-1 4}"
OUT_PREFIX="${OUT_PREFIX:-benchmarks/results/fair_decode}"
FAIR_MODE="${FAIR_MODE:-1}"
PARALLEL_SUBMIT="${PARALLEL_SUBMIT:-0}"
COMPARE_METRIC="${COMPARE_METRIC:-}"
COMPARE_STEADY_FOR_BATCH_GE2="${COMPARE_STEADY_FOR_BATCH_GE2:-}"
FILTER_INVALID="${FILTER_INVALID:-1}"
STRICT_TOKEN_CHECK="${STRICT_TOKEN_CHECK:-0}"
INTERLEAVE_MODE="${INTERLEAVE_MODE:-alternate}"
DENSE_RUN_TIMEOUT_S="${DENSE_RUN_TIMEOUT_S:-180}"
LLAMA_RUN_TIMEOUT_S="${LLAMA_RUN_TIMEOUT_S:-180}"
# DenseCore fair benchmark defaults (override by exporting explicit values).
# - Skip unnecessary cont materialization for N>1 attention path on AVX2.
# - Disable decode graph cache for deterministic A/B perf isolation.
DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE="${DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE:-on}"
DENSECORE_DECODE_GRAPH_CACHE="${DENSECORE_DECODE_GRAPH_CACHE:-0}"
# Fixed fast-profile defaults:
# - Keep experimental true-batched Q4_K kernel OFF (currently slower in fair decode).
# - Keep quant nrc-batch OFF on x86 AVX2 path (currently slower than baseline).
DENSECORE_ENABLE_Q4K_BATCHED_KERNEL="${DENSECORE_ENABLE_Q4K_BATCHED_KERNEL:-0}"
DENSECORE_ENABLE_QUANT_NRC_BATCH="${DENSECORE_ENABLE_QUANT_NRC_BATCH:-0}"
# Benchmark-overhead controls (favors kernel/scheduler cost visibility).
DENSECORE_BENCH_DIRECT_CALLBACK="${DENSECORE_BENCH_DIRECT_CALLBACK:-0}"
DENSECORE_BENCH_DECODE_BATCH_FAST_PATH="${DENSECORE_BENCH_DECODE_BATCH_FAST_PATH:-1}"
DENSECORE_BENCH_FAST_PATH_MAX_BATCH="${DENSECORE_BENCH_FAST_PATH_MAX_BATCH:-8}"
DENSECORE_BENCH_TPS_MODE="${DENSECORE_BENCH_TPS_MODE:-all}"
DENSECORE_BENCH_RECORD_TOKEN_TIMES="${DENSECORE_BENCH_RECORD_TOKEN_TIMES:-0}"

# Keep benchmark thread count fixed unless explicitly overridden by caller.
export DENSECORE_BENCH_RESPECT_THREADS="${DENSECORE_BENCH_RESPECT_THREADS:-1}"

# Backward compatibility for older workflow knobs.
if [[ -z "$COMPARE_METRIC" ]]; then
  if [[ "$COMPARE_STEADY_FOR_BATCH_GE2" == "1" ]]; then
    COMPARE_METRIC="tps_steady"
  else
    COMPARE_METRIC="tps"
  fi
fi
if [[ "$COMPARE_METRIC" != "tps" && "$COMPARE_METRIC" != "tps_steady" ]]; then
  echo "[ERROR] COMPARE_METRIC must be 'tps' or 'tps_steady' (got '$COMPARE_METRIC')" >&2
  exit 1
fi
if [[ "$COMPARE_METRIC" == "tps_steady" && "$DENSECORE_BENCH_RECORD_TOKEN_TIMES" == "0" ]]; then
  echo "[WARN] COMPARE_METRIC=tps_steady requires per-token timestamps; enabling DENSECORE_BENCH_RECORD_TOKEN_TIMES=1" >&2
  DENSECORE_BENCH_RECORD_TOKEN_TIMES=1
fi

# Set DENSECORE_DEBUG_MATMUL_DISPATCH=1 when verifying dispatch paths
# (e.g., confirming GGML_QUANT_NRC_M is hit for batch>=4 quant models).
# export DENSECORE_DEBUG_MATMUL_DISPATCH=1

MODEL_DIR="${MODEL_DIR:-/mnt/c/Users/jwsong/PycharmProjects/DenseCore/models}"
LLAMA_BB_BIN="${LLAMA_BB_BIN:-/home/jaewook/llama.cpp/build/bin/llama-batched-bench}"

MODELS=(
  "qwen3_0.6b:Qwen3-0.6B-Q4_K_M.gguf"
  "qwen3_4b:Qwen3-4B-Q4_K_M.gguf"
  "llama3.2_1b:Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)

read -r -a BATCH_ARRAY <<< "$BATCHES"
if [[ ${#BATCH_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] no batches provided in BATCHES='$BATCHES'" >&2
  exit 1
fi

RAW="${OUT_PREFIX}_raw.csv"
SUM="${OUT_PREFIX}_summary.csv"
PAIR="${OUT_PREFIX}_paired.csv"
JSON="${OUT_PREFIX}_summary.json"

mkdir -p "$(dirname "$RAW")"

echo "suite,model_tag,model_file,batch,framework,run,tps,tps_steady,tokens_generated,expected_tokens,valid,note" > "$RAW"

batch_order_for_run() {
  local run="$1"
  local -n out_ref="$2"
  out_ref=()

  if [[ "$INTERLEAVE_MODE" == "random" ]]; then
    if command -v shuf >/dev/null 2>&1; then
      while IFS= read -r b; do
        out_ref+=("$b")
      done < <(printf "%s\n" "${BATCH_ARRAY[@]}" | shuf)
      return
    fi
    echo "[WARN] INTERLEAVE_MODE=random requested but 'shuf' not found; using alternate mode" >&2
  fi

  if (( run % 2 == 1 )); then
    out_ref=("${BATCH_ARRAY[@]}")
  else
    for ((idx=${#BATCH_ARRAY[@]} - 1; idx >= 0; --idx)); do
      out_ref+=("${BATCH_ARRAY[idx]}")
    done
  fi
}

run_dense() {
  local model_path="$1"
  local model_file="$2"
  local tag="$3"
  local batch="$4"
  local run="$5"

  local -a dense_args=("$model_path" "$PROMPT_TOK" "$GEN_TOK" "$THREADS" "1" "--batch-size" "$batch" "--csv")
  if [[ "$FAIR_MODE" == "1" ]]; then
    dense_args+=("--fair-mode")
  fi
  if [[ "$PARALLEL_SUBMIT" == "1" ]]; then
    dense_args+=("--parallel-submit")
  fi

  local dense_out=""
  local dense_cmd_status=0
  if dense_out=$(DENSECORE_AUTO_CHAT_TEMPLATE=0 DENSECORE_BENCH_MODE=1 DENSECORE_GRAPH_CTX_MAX_MB=6144 \
    DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE="$DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE" \
    DENSECORE_DECODE_GRAPH_CACHE="$DENSECORE_DECODE_GRAPH_CACHE" \
    DENSECORE_ENABLE_Q4K_BATCHED_KERNEL="$DENSECORE_ENABLE_Q4K_BATCHED_KERNEL" \
    DENSECORE_ENABLE_QUANT_NRC_BATCH="$DENSECORE_ENABLE_QUANT_NRC_BATCH" \
    DENSECORE_BENCH_DIRECT_CALLBACK="$DENSECORE_BENCH_DIRECT_CALLBACK" \
    DENSECORE_BENCH_DECODE_BATCH_FAST_PATH="$DENSECORE_BENCH_DECODE_BATCH_FAST_PATH" \
    DENSECORE_BENCH_FAST_PATH_MAX_BATCH="$DENSECORE_BENCH_FAST_PATH_MAX_BATCH" \
    DENSECORE_BENCH_TPS_MODE="$DENSECORE_BENCH_TPS_MODE" \
    DENSECORE_BENCH_RECORD_TOKEN_TIMES="$DENSECORE_BENCH_RECORD_TOKEN_TIMES" \
    LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH:-} \
    timeout "${DENSE_RUN_TIMEOUT_S}s" ./benchmarks/bench_native "${dense_args[@]}" 2>&1); then
    dense_cmd_status=0
  else
    dense_cmd_status=$?
  fi

  # Fallback path for engines where concurrent submit is not safe.
  if [[ "$dense_cmd_status" -ne 0 && "$PARALLEL_SUBMIT" == "1" ]]; then
    echo "[WARN] DenseCore parallel-submit run failed or timed out (status=$dense_cmd_status): model=$tag batch=$batch run=$run. Retrying with sequential submit fallback." >&2
    dense_out=""
    if dense_out=$(DENSECORE_AUTO_CHAT_TEMPLATE=0 DENSECORE_BENCH_MODE=1 DENSECORE_GRAPH_CTX_MAX_MB=6144 \
      DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE="$DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE" \
      DENSECORE_DECODE_GRAPH_CACHE="$DENSECORE_DECODE_GRAPH_CACHE" \
      DENSECORE_ENABLE_Q4K_BATCHED_KERNEL="$DENSECORE_ENABLE_Q4K_BATCHED_KERNEL" \
      DENSECORE_ENABLE_QUANT_NRC_BATCH="$DENSECORE_ENABLE_QUANT_NRC_BATCH" \
      DENSECORE_BENCH_SUBMIT_THREAD_SAFE=0 \
      DENSECORE_BENCH_DIRECT_CALLBACK="$DENSECORE_BENCH_DIRECT_CALLBACK" \
      DENSECORE_BENCH_DECODE_BATCH_FAST_PATH="$DENSECORE_BENCH_DECODE_BATCH_FAST_PATH" \
      DENSECORE_BENCH_FAST_PATH_MAX_BATCH="$DENSECORE_BENCH_FAST_PATH_MAX_BATCH" \
      DENSECORE_BENCH_TPS_MODE="$DENSECORE_BENCH_TPS_MODE" \
      DENSECORE_BENCH_RECORD_TOKEN_TIMES="$DENSECORE_BENCH_RECORD_TOKEN_TIMES" \
      LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH:-} \
      timeout "${DENSE_RUN_TIMEOUT_S}s" ./benchmarks/bench_native "${dense_args[@]}" 2>&1); then
      :
    else
      echo "[ERROR] DenseCore fallback run also failed: model=$tag batch=$batch run=$run" >&2
      printf "%s\n" "$dense_out" | tail -n 40 >&2
      exit 1
    fi
  elif [[ "$dense_cmd_status" -ne 0 ]]; then
    echo "[ERROR] DenseCore run failed (status=$dense_cmd_status): model=$tag batch=$batch run=$run" >&2
    printf "%s\n" "$dense_out" | tail -n 40 >&2
    exit 1
  fi

  local dense_csv
  dense_csv=$(printf "%s\n" "$dense_out" | awk -F',' '
    function trim(s){ gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }
    function isnum(s){ s=trim(s); return (s ~ /^-?[0-9]+([.][0-9]+)?$/) }
    NF == 8 {
      ok = 1
      for (i = 1; i <= 8; ++i) if (!isnum($i)) ok = 0
      if (ok) line = $0
    }
    END { print line }
  ')

  if [[ -z "$dense_csv" ]]; then
    echo "[ERROR] failed to parse DenseCore CSV for $tag batch=$batch run=$run" >&2
    printf "%s\n" "$dense_out" | tail -n 40 >&2
    exit 1
  fi

  local dense_ttft dense_tps dense_tps_steady dense_tokens dense_itl_avg dense_itl_p50 dense_itl_p90 dense_itl_p99
  IFS=',' read -r dense_ttft dense_tps dense_tps_steady dense_tokens dense_itl_avg dense_itl_p50 dense_itl_p90 dense_itl_p99 <<< "$dense_csv"

  local expected_tokens=$((batch * GEN_TOK))
  local valid=1
  local note="ok"
  if [[ "$dense_tokens" != "$expected_tokens" ]]; then
    valid=0
    note="token_mismatch"
    echo "[WARN] DenseCore token mismatch: model=$tag batch=$batch run=$run got=$dense_tokens expected=$expected_tokens" >&2
    if [[ "$STRICT_TOKEN_CHECK" == "1" ]]; then
      echo "[ERROR] STRICT_TOKEN_CHECK=1 and DenseCore token mismatch detected" >&2
      exit 1
    fi
  fi

  echo "decode,$tag,$model_file,$batch,densecore,$run,$dense_tps,$dense_tps_steady,$dense_tokens,$expected_tokens,$valid,$note" >> "$RAW"
  echo "$dense_tps,$dense_tps_steady,$dense_tokens,$valid"
}

run_llama() {
  local model_path="$1"
  local model_file="$2"
  local tag="$3"
  local batch="$4"
  local run="$5"

  local npp_arg="$PROMPT_TOK"
  if [[ "$FAIR_MODE" == "1" ]]; then
    # One process, two identical rows: first is warmup, second is measured.
    npp_arg="${PROMPT_TOK},${PROMPT_TOK}"
  fi

  local lb_log
  lb_log=$(mktemp)
  local llama_cmd_status=0
  if timeout "${LLAMA_RUN_TIMEOUT_S}s" "$LLAMA_BB_BIN" \
    -m "$model_path" \
    -t "$THREADS" \
    -tb "$THREADS" \
    -npp "$npp_arg" \
    -ntg "$GEN_TOK" \
    -npl "$batch" \
    --output-format md > "$lb_log" 2>&1; then
    llama_cmd_status=0
  else
    llama_cmd_status=$?
  fi
  if [[ "$llama_cmd_status" -ne 0 ]]; then
    echo "[ERROR] llama-batched-bench failed or timed out (status=$llama_cmd_status): model=$tag batch=$batch run=$run" >&2
    tail -n 40 "$lb_log" >&2
    rm -f "$lb_log"
    exit 1
  fi

  local lb_row
  lb_row=$(awk '/^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|/{row=$0} END{print row}' "$lb_log")
  if [[ -z "$lb_row" ]]; then
    echo "[ERROR] failed to parse llama-batched-bench table row for $tag batch=$batch run=$run" >&2
    tail -n 40 "$lb_log" >&2
    rm -f "$lb_log"
    exit 1
  fi

  if [[ "$FAIR_MODE" == "1" ]]; then
    local lb_rows
    lb_rows=$(awk '/^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|/{n++} END{print n+0}' "$lb_log")
    if [[ "$lb_rows" -lt 2 ]]; then
      echo "[ERROR] fair mode expects >=2 llama rows (warmup+measure), got $lb_rows for $tag batch=$batch run=$run" >&2
      tail -n 40 "$lb_log" >&2
      rm -f "$lb_log"
      exit 1
    fi
  fi

  local lb_vals
  lb_vals=$(echo "$lb_row" | awk -F'|' '
    function trim(s){ gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    function isnum(s){ s=trim(s); return (s ~ /^-?[0-9]+([.][0-9]+)?$/) }
    {
      tg = trim($3)
      b  = trim($4)
      stg = trim($9)
      if (!isnum(tg) || !isnum(b) || !isnum(stg)) exit 1
      printf "%s,%s,%s\n", stg, tg, b
    }
  ')
  if [[ -z "$lb_vals" ]]; then
    echo "[ERROR] failed to parse llama row values for $tag batch=$batch run=$run" >&2
    echo "$lb_row" >&2
    rm -f "$lb_log"
    exit 1
  fi

  local llama_tps llama_tg llama_b
  IFS=',' read -r llama_tps llama_tg llama_b <<< "$lb_vals"
  local llama_tps_steady="$llama_tps"
  local llama_tokens
  llama_tokens=$(awk -v tg="$llama_tg" -v b="$llama_b" 'BEGIN{printf "%.0f", tg*b}')

  local expected_tokens=$((batch * GEN_TOK))
  local valid=1
  local note="ok"
  if [[ "$llama_tokens" != "$expected_tokens" ]]; then
    valid=0
    note="token_mismatch"
    echo "[WARN] llama token mismatch: model=$tag batch=$batch run=$run got=$llama_tokens expected=$expected_tokens" >&2
    if [[ "$STRICT_TOKEN_CHECK" == "1" ]]; then
      echo "[ERROR] STRICT_TOKEN_CHECK=1 and llama token mismatch detected" >&2
      rm -f "$lb_log"
      exit 1
    fi
  fi

  echo "decode,$tag,$model_file,$batch,llama_batched_bench,$run,$llama_tps,$llama_tps_steady,$llama_tokens,$expected_tokens,$valid,$note" >> "$RAW"
  rm -f "$lb_log"
  echo "$llama_tps,$llama_tps_steady,$llama_tokens,$valid"
}

for entry in "${MODELS[@]}"; do
  IFS=':' read -r tag model_file <<< "$entry"
  model_path="$MODEL_DIR/$model_file"

  for run in $(seq 1 "$REPEATS"); do
    batch_order=()
    batch_order_for_run "$run" batch_order

    for batch in "${batch_order[@]}"; do
      dense_vals=""
      llama_vals=""

      if [[ "$FAIR_MODE" == "1" && $((run % 2)) -eq 1 ]]; then
        echo "[RUN] $tag batch=$batch run=$run llama_batched_bench (first)"
        llama_vals=$(run_llama "$model_path" "$model_file" "$tag" "$batch" "$run")
        echo "[RUN] $tag batch=$batch run=$run densecore_decode (second)"
        dense_vals=$(run_dense "$model_path" "$model_file" "$tag" "$batch" "$run")
      else
        echo "[RUN] $tag batch=$batch run=$run densecore_decode (first)"
        dense_vals=$(run_dense "$model_path" "$model_file" "$tag" "$batch" "$run")
        echo "[RUN] $tag batch=$batch run=$run llama_batched_bench (second)"
        llama_vals=$(run_llama "$model_path" "$model_file" "$tag" "$batch" "$run")
      fi

      dense_tps=$(echo "$dense_vals" | cut -d',' -f1)
      dense_tps_steady=$(echo "$dense_vals" | cut -d',' -f2)
      dense_tokens=$(echo "$dense_vals" | cut -d',' -f3)

      llama_tps=$(echo "$llama_vals" | cut -d',' -f1)
      llama_tps_steady=$(echo "$llama_vals" | cut -d',' -f2)
      llama_tokens=$(echo "$llama_vals" | cut -d',' -f3)

      echo "[DONE] $tag batch=$batch run=$run dense_tps=$dense_tps dense_steady=$dense_tps_steady dense_tok=$dense_tokens llama_tps=$llama_tps llama_steady=$llama_tps_steady llama_tok=$llama_tokens"
    done
  done
done

awk -F',' -v filter_invalid="$FILTER_INVALID" '
NR > 1 {
  if (filter_invalid == "1" && $11 != "1") next
  k = $2","$3","$4","$5
  n[k]++
  sum_tps[k] += $7
  sumsq_tps[k] += $7 * $7
  sum_st[k] += $8
  sumsq_st[k] += $8 * $8
  sum_tok[k] += $9
}
END {
  for (k in n) {
    mean_tps = sum_tps[k] / n[k]
    var_tps = (n[k] > 1) ? (sumsq_tps[k] - sum_tps[k] * sum_tps[k] / n[k]) / (n[k] - 1) : 0
    if (var_tps < 0) var_tps = 0
    std_tps = sqrt(var_tps)
    ci_tps = 1.96 * std_tps / sqrt(n[k])

    mean_st = sum_st[k] / n[k]
    var_st = (n[k] > 1) ? (sumsq_st[k] - sum_st[k] * sum_st[k] / n[k]) / (n[k] - 1) : 0
    if (var_st < 0) var_st = 0
    std_st = sqrt(var_st)
    ci_st = 1.96 * std_st / sqrt(n[k])

    mean_tok = sum_tok[k] / n[k]

    split(k, a, ",")
    printf "%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n", a[1], a[2], a[3], a[4], n[k], mean_tps, std_tps, ci_tps, mean_st, std_st, ci_st, mean_tok
  }
}
' "$RAW" | sort -t, -k1,1 -k3,3n -k4,4 > /tmp/fair_decode_summary_body.csv

{
  echo "model_tag,model_file,batch,framework,n,mean_tps,std_tps,ci95_tps,mean_tps_steady,std_tps_steady,ci95_tps_steady,mean_tokens_generated"
  cat /tmp/fair_decode_summary_body.csv
} > "$SUM"

awk -F',' -v filter_invalid="$FILTER_INVALID" -v compare_metric="$COMPARE_METRIC" '
NR > 1 {
  if (filter_invalid == "1" && $11 != "1") next
  key = $2","$3","$4","$6
  if ($5 == "densecore") {
    d_tps[key] = $7
    d_st[key] = $8
    d_tok[key] = $9
  }
  if ($5 == "llama_batched_bench") {
    l_tps[key] = $7
    l_st[key] = $8
    l_tok[key] = $9
  }
}
END {
  for (k in d_tps) {
    if (!(k in l_tps)) continue

    split(k, a, ",")
    cmp = compare_metric
    d_cmp = (cmp == "tps_steady") ? d_st[k] : d_tps[k]
    l_cmp = (cmp == "tps_steady") ? l_st[k] : l_tps[k]
    delta = (l_cmp > 0.0) ? ((d_cmp / l_cmp) - 1.0) * 100.0 : 0.0

    printf "%s,%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%s,%.4f,%.4f,%.2f\n", a[1], a[2], a[3], a[4], d_tps[k], l_tps[k], d_st[k], l_st[k], d_tok[k], l_tok[k], cmp, d_cmp, l_cmp, delta
  }
}
' "$RAW" | sort -t, -k1,1 -k3,3n -k4,4n > /tmp/fair_decode_pair_body.csv

{
  echo "model_tag,model_file,batch,run,densecore_tps,llama_tps,densecore_tps_steady,llama_tps_steady,densecore_tokens,llama_tokens,compare_metric,densecore_compare,llama_compare,delta_pct"
  cat /tmp/fair_decode_pair_body.csv
} > "$PAIR"

{
  echo "["
  awk -F',' '
    NR > 1 {
      c++
      printf "%s  {\"model_tag\":\"%s\",\"model_file\":\"%s\",\"batch\":%s,\"framework\":\"%s\",\"n\":%s,\"mean_tps\":%s,\"std_tps\":%s,\"ci95_tps\":%s,\"mean_tps_steady\":%s,\"std_tps_steady\":%s,\"ci95_tps_steady\":%s,\"mean_tokens_generated\":%s}", (c == 1 ? "" : ",\n"), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
    }
    END {
      if (c > 0) printf "\n"
    }
  ' "$SUM"
  echo "]"
} > "$JSON"

echo "WROTE $RAW"
echo "WROTE $SUM"
echo "WROTE $PAIR"
echo "WROTE $JSON"
