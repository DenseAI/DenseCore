#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

THREADS="${THREADS:-8}"
PROMPT_TOK="${PROMPT_TOK:-64}"
GEN_TOK="${GEN_TOK:-64}"
REPEATS="${REPEATS:-3}"
BATCHES="${BATCHES:-1 4}"
OUT_PREFIX="${OUT_PREFIX:-benchmarks/results/neurips_serving}"

MODELS=(
  "qwen3_0.6b:Qwen3-0.6B-Q4_K_M.gguf"
  "qwen3_4b:Qwen3-4B-Q4_K_M.gguf"
  "llama3.2_1b:Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)

MODEL_DIR="/mnt/c/Users/jwsong/PycharmProjects/DenseCore/models"
LLAMA_PARALLEL_BIN="/home/jaewook/llama.cpp/build/bin/llama-parallel"

RAW="${OUT_PREFIX}_raw.csv"
SUM="${OUT_PREFIX}_summary.csv"
PAIR="${OUT_PREFIX}_paired.csv"
JSON="${OUT_PREFIX}_summary.json"

mkdir -p "$(dirname "$RAW")"

PROMPT="Write a comprehensive guide to "
for _ in $(seq 1 $((PROMPT_TOK / 6))); do
  PROMPT+="Write a comprehensive guide to "
done

echo "suite,model_tag,model_file,batch,framework,run,tps" > "$RAW"

for entry in "${MODELS[@]}"; do
  IFS=':' read -r tag model_file <<< "$entry"
  model_path="$MODEL_DIR/$model_file"

  for batch in $BATCHES; do
    for run in $(seq 1 "$REPEATS"); do
      echo "[RUN] $tag batch=$batch run=$run densecore"
      dense_csv=$(DENSECORE_AUTO_CHAT_TEMPLATE=0 DENSECORE_BENCH_MODE=1 DENSECORE_GRAPH_CTX_MAX_MB=6144 \
          LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH:-} \
          ./benchmarks/bench_native "$model_path" "$PROMPT_TOK" "$GEN_TOK" "$THREADS" 1 \
          --batch-size "$batch" --csv 2>&1 | \
          grep -E '^[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?$' | \
          tail -1)
      dense_tps=$(echo "$dense_csv" | cut -d',' -f2)
      echo "serving,$tag,$model_file,$batch,densecore,$run,$dense_tps" >> "$RAW"

      echo "[RUN] $tag batch=$batch run=$run llama_parallel"
      lp_log=$(mktemp)
      "$LLAMA_PARALLEL_BIN" \
        -m "$model_path" \
        -t "$THREADS" \
        -np "$batch" \
        -ns "$batch" \
        -pps \
        -p "$PROMPT" \
        -n "$GEN_TOK" \
        --ignore-eos \
        --temp 0 \
        --seed $((123 + run)) > "$lp_log" 2>&1

      lp_line=$(grep -E 'Total gen tokens:' "$lp_log" | tail -1 || true)
      if [[ -z "$lp_line" ]]; then
        echo "[ERROR] failed to parse llama-parallel for $tag batch=$batch run=$run" >&2
        tail -n 40 "$lp_log" >&2
        rm -f "$lp_log"
        exit 1
      fi
      lp_tps=$(echo "$lp_line" | sed -E 's/.*speed: *([0-9]+\.[0-9]+) t\/s.*/\1/')
      echo "serving,$tag,$model_file,$batch,llama_parallel,$run,$lp_tps" >> "$RAW"

      rm -f "$lp_log"
      echo "[DONE] $tag batch=$batch run=$run dense=$dense_tps llama=$lp_tps"
    done
  done
done

awk -F',' 'NR>1 {
  k=$2","$3","$4","$5;
  n[k]++;
  sum[k]+=$7;
  sumsq[k]+=$7*$7;
}
END {
  for (k in n) {
    mean=sum[k]/n[k];
    var=(n[k]>1)?(sumsq[k]-sum[k]*sum[k]/n[k])/(n[k]-1):0;
    if (var<0) var=0;
    std=sqrt(var);
    ci=1.96*std/sqrt(n[k]);
    split(k,a,",");
    printf "%s,%s,%s,%s,%d,%.4f,%.4f,%.4f\n", a[1],a[2],a[3],a[4],n[k],mean,std,ci;
  }
}' "$RAW" | sort -t, -k1,1 -k3,3n -k4,4 > /tmp/serving_summary_body.csv

{
  echo "model_tag,model_file,batch,framework,n,mean_tps,std_tps,ci95_tps"
  cat /tmp/serving_summary_body.csv
} > "$SUM"

awk -F',' 'NR>1 {
  key=$2","$3","$4","$6;
  if ($5=="densecore") d[key]=$7;
  if ($5=="llama_parallel") l[key]=$7;
}
END {
  for (k in d) {
    if (k in l) {
      split(k,a,",");
      delta=((d[k]/l[k])-1.0)*100.0;
      printf "%s,%s,%s,%s,%.4f,%.4f,%.2f\n", a[1],a[2],a[3],a[4],d[k],l[k],delta;
    }
  }
}' "$RAW" | sort -t, -k1,1 -k3,3n -k4,4n > /tmp/serving_pair_body.csv

{
  echo "model_tag,model_file,batch,run,densecore_tps,llama_parallel_tps,delta_pct"
  cat /tmp/serving_pair_body.csv
} > "$PAIR"

{
  echo "["
  awk -F',' 'NR>1 {printf "%s  {\"model_tag\":\"%s\",\"model_file\":\"%s\",\"batch\":%s,\"framework\":\"%s\",\"n\":%s,\"mean_tps\":%s,\"std_tps\":%s,\"ci95_tps\":%s}", (NR==2?"":",\n"), $1,$2,$3,$4,$5,$6,$7,$8} END{printf "\n"}' "$SUM"
  echo "]"
} > "$JSON"

echo "WROTE $RAW"
echo "WROTE $SUM"
echo "WROTE $PAIR"
echo "WROTE $JSON"
