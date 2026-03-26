#!/usr/bin/env bash
# =============================================================================
# gcp_autorun.sh — DenseCore one-shot GCP benchmark runner
#
# Runs: host validation → build → mock tests → per-model benchmark → shutdown
# Aligns with DenseSeries CPO KPIs:
#   TTFT, p99 ITL (jitter), tok/s per dollar, peak memory footprint,
#   NUMA-local vs cross-node comparison, hardware portability matrix
#
# Usage:
#   TIER=practical MODEL_QWEN35_35B=/data/models/... PRICE_PER_HOUR=3.45 \
#   benchmarks/gcp_autorun.sh
#
# Model path env vars (set the ones you have downloaded):
#   MODEL_QWEN35_35B       — Qwen3.5-35B-A3B   (Tier A practical)
#   MODEL_QWEN35_27B       — Qwen3.5-27B        (Tier A practical)
#   MODEL_QWEN3_CODER_30B  — Qwen3-Coder-30B-A3B-Instruct (Tier A practical)
#   MODEL_GEMMA3_27B       — Gemma 3 27B IT     (Tier A practical)
#   MODEL_QWEN35_397B      — Qwen3.5-397B-A17B  (Tier S viral)
#   MODEL_GLM45_AIR        — GLM-4.5-Air        (Tier A bringup)
#   MODEL_GLM5             — GLM-5              (Tier S bringup)
#
# Key env overrides:
#   TIER=practical|viral|bringup|all     (default: practical)
#   PRICE_PER_HOUR=X.XX                  (default: 0 — skip cost calc)
#   AUTO_SHUTDOWN=1|0                    (default: 1)
#   SHUTDOWN_DELAY_MIN=N                 (default: 2)
#   RUNS=N                               (default: 3)
#   SKIP_BUILD=1|0                       (default: 0)
#   SKIP_APT=1|0                         (default: 0)
#   JOBS=N                               (default: nproc)
#   DRY_RUN=1|0                          (default: 0 — print plan without running)
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIER="${TIER:-practical}"
PRICE_PER_HOUR="${PRICE_PER_HOUR:-0}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
SHUTDOWN_DELAY_MIN="${SHUTDOWN_DELAY_MIN:-2}"
RUNS="${RUNS:-3}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_APT="${SKIP_APT:-0}"
JOBS="${JOBS:-$(nproc)}"
DRY_RUN="${DRY_RUN:-0}"
FAIR_MODE="${FAIR_MODE:-1}"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BENCH_BIN="$ROOT_DIR/benchmarks/bench_native"

timestamp="$(date +%Y%m%d_%H%M%S)"
HOSTNAME_SHORT="$(hostname -s)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/results/autorun_${HOSTNAME_SHORT}_${timestamp}}"

# ---------------------------------------------------------------------------
# GCP metadata
# ---------------------------------------------------------------------------
gcp_meta() {
  curl -fsS --max-time 1 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${1}" 2>/dev/null || true
}
MACHINE_TYPE_RAW="$(gcp_meta instance/machine-type)"
MACHINE_TYPE="${MACHINE_TYPE_RAW##*/}"
INSTANCE_NAME="$(gcp_meta instance/name)"
ZONE_RAW="$(gcp_meta instance/zone)"
ZONE="${ZONE_RAW##*/}"
PROJECT_ID="$(gcp_meta project/project-id)"
ARCH="$(uname -m)"
CPU_MODEL="$(lscpu | awk -F: '/Model name:/{print $2; exit}' | xargs)"
LOGICAL_CORES="$(nproc)"
PHYSICAL_CORES="$(lscpu -p=core | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
NUMA_NODES="$(lscpu | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2; exit}')"
MEM_GB="$(free -g | awk '/^Mem:/{print $2}')"
[[ -z "$PHYSICAL_CORES" || "$PHYSICAL_CORES" -le 0 ]] && PHYSICAL_CORES="$LOGICAL_CORES"
[[ -z "$NUMA_NODES" || "$NUMA_NODES" -le 0 ]] && NUMA_NODES=1

# Thread grid: auto-build from physical cores if not overridden
build_thread_grid() {
  local -a pts=(1 $((PHYSICAL_CORES/4)) $((PHYSICAL_CORES/2)) $(((PHYSICAL_CORES*3)/4)) "$PHYSICAL_CORES")
  (( LOGICAL_CORES > PHYSICAL_CORES )) && pts+=("$LOGICAL_CORES")
  local -A seen=()
  local out=()
  for v in "${pts[@]}"; do
    [[ "$v" -le 0 ]] && continue
    [[ -z "${seen[$v]:-}" ]] && { seen[$v]=1; out+=("$v"); }
  done
  printf '%s' "${out[*]}"
}

# Per-tier sweep parameters
case "$TIER" in
  practical|all)
    PRACTICAL_THREAD_GRID="${PRACTICAL_THREAD_GRID:-$(build_thread_grid)}"
    PRACTICAL_DECODE_BATCHES="${PRACTICAL_DECODE_BATCHES:-1 4 8}"
    PRACTICAL_DECODE_CONTEXT="${PRACTICAL_DECODE_CONTEXT:-1024 8192 32768}"
    PRACTICAL_PREFILL_SEQ="${PRACTICAL_PREFILL_SEQ:-2048 8192 32768}"
    PRACTICAL_GEN_TOKENS="${PRACTICAL_GEN_TOKENS:-128}"
    ;;
esac
case "$TIER" in
  viral|all)
    VIRAL_THREAD_GRID="${VIRAL_THREAD_GRID:-1 16 32 48 64}"
    VIRAL_DECODE_BATCHES="${VIRAL_DECODE_BATCHES:-1 2 4}"
    VIRAL_DECODE_CONTEXT="${VIRAL_DECODE_CONTEXT:-1024 8192 32768 131072}"
    VIRAL_PREFILL_SEQ="${VIRAL_PREFILL_SEQ:-2048 8192 32768 131072}"
    VIRAL_GEN_TOKENS="${VIRAL_GEN_TOKENS:-64}"
    ;;
esac

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log_section() {
  echo ""
  printf '=%.0s' {1..60}; echo ""
  printf '  %s\n' "$*"
  printf '=%.0s' {1..60}; echo ""
}

# ---------------------------------------------------------------------------
# Shutdown trap — fires on EXIT (success or error)
# ---------------------------------------------------------------------------
_SHUTDOWN_TRIGGERED=0
schedule_shutdown() {
  if [[ "$AUTO_SHUTDOWN" == "1" && "$_SHUTDOWN_TRIGGERED" == "0" && "$DRY_RUN" == "0" ]]; then
    _SHUTDOWN_TRIGGERED=1
    log "Scheduling VM shutdown in ${SHUTDOWN_DELAY_MIN} minutes (AUTO_SHUTDOWN=0 to disable)"
    sudo shutdown -h +"${SHUTDOWN_DELAY_MIN}" "DenseCore benchmark complete — auto shutdown" || true
  fi
}
trap 'schedule_shutdown' EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
require_time_v() {
  if ! /usr/bin/time -v true 2>&1 | grep -q "Maximum resident"; then
    log "WARNING: /usr/bin/time -v not available — peak RSS tracking disabled"
    echo "unavailable"
  else
    echo "available"
  fi
}

# Run bench_native once, capture CSV output + peak RSS via /usr/bin/time -v
# Returns: CSV line written to $csv_out, RSS written to $rss_out (in kB)
run_bench_with_rss() {
  local model="$1"
  local prompt_tokens="$2"
  local gen_tokens="$3"
  local threads="$4"
  local batch_size="$5"
  local csv_out="$6"   # file to write CSV line into
  local rss_out="$7"   # file to write peak RSS kB into
  local extra_flags="${8:-}"

  local time_stderr
  time_stderr="$(mktemp)"

  local args=(
    "$BENCH_BIN"
    "$model"
    "$prompt_tokens"
    "$gen_tokens"
    "$threads"
    "$RUNS"
    --batch-size "$batch_size"
    --csv
  )
  [[ "$FAIR_MODE" == "1" ]] && args+=(--fair-mode)
  # shellcheck disable=SC2206
  [[ -n "$extra_flags" ]] && args+=($extra_flags)

  local csv_line=""
  if [[ "$TIME_V_AVAILABLE" == "available" ]]; then
    csv_line="$(
      env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
        DENSECORE_BENCH_RECORD_TOKEN_TIMES=1 \
        /usr/bin/time -v \
        "${args[@]}" 2>"$time_stderr" | tail -n1
    )" || true
    local rss_kb
    rss_kb="$(grep "Maximum resident" "$time_stderr" | awk '{print $NF}' || echo 0)"
    echo "$rss_kb" > "$rss_out"
  else
    csv_line="$(
      env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
        DENSECORE_BENCH_RECORD_TOKEN_TIMES=1 \
        "${args[@]}" | tail -n1
    )" || true
    echo "0" > "$rss_out"
  fi
  rm -f "$time_stderr"
  echo "$csv_line" > "$csv_out"
}

# Parse CSV line from bench_native
parse_bench_csv() {
  local line="$1"
  IFS=',' read -r ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$line"
  printf '%s %s %s %s %s %s %s %s' \
    "${ttft_ms:-0}" "${decode_tps:-0}" "${tps_steady:-0}" "${tokens:-0}" \
    "${itl_avg:-0}" "${itl_p50:-0}" "${itl_p90:-0}" "${itl_p99:-0}"
}

# Calculate tok/s per dollar
calc_tok_per_dollar() {
  local tps_steady="$1"
  if [[ "${PRICE_PER_HOUR:-0}" == "0" || -z "${PRICE_PER_HOUR:-}" ]]; then
    echo "N/A (set PRICE_PER_HOUR)"
    return
  fi
  awk -v tps="$tps_steady" -v ph="$PRICE_PER_HOUR" \
    'BEGIN { if (ph > 0 && tps > 0) printf "%.0f", tps / (ph / 3600); else print "0" }'
}

# Compute jitter ratio (p99/p50)
calc_jitter_ratio() {
  local p50="$1" p99="$2"
  awk -v p50="$p50" -v p99="$p99" \
    'BEGIN { if (p50 > 0) printf "%.2f", p99/p50; else print "N/A" }'
}

# Run a single bench case and append to RESULT_CSV
bench_case() {
  local phase="$1" model="$2" model_label="$3"
  local prompt_tokens="$4" gen_tokens="$5" batch_size="$6" threads="$7"
  local notes="${8:-}"

  local csv_file rss_file
  csv_file="$(mktemp)"
  rss_file="$(mktemp)"

  run_bench_with_rss "$model" "$prompt_tokens" "$gen_tokens" "$threads" "$batch_size" \
    "$csv_file" "$rss_file"

  local csv_line rss_kb
  csv_line="$(cat "$csv_file")"
  rss_kb="$(cat "$rss_file")"
  rm -f "$csv_file" "$rss_file"

  if [[ -z "$csv_line" ]]; then
    log "  [SKIP] empty result for $model_label phase=$phase threads=$threads batch=$batch_size"
    return
  fi

  read -r ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$(parse_bench_csv "$csv_line")"
  local rss_gb jitter tok_per_dollar
  rss_gb="$(awk -v k="$rss_kb" 'BEGIN { if (k>0) printf "%.2f", k/1048576; else print "0" }')"
  jitter="$(calc_jitter_ratio "$itl_p50" "$itl_p99")"
  tok_per_dollar="$(calc_tok_per_dollar "$tps_steady")"

  local prefill_tps=""
  if [[ "$phase" == "prefill" ]]; then
    prefill_tps="$(awk -v p="$prompt_tokens" -v b="$batch_size" -v t="$ttft_ms" \
      'BEGIN { if (t>0) printf "%.2f", (p*b)/(t/1000.0); else print 0 }')"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$phase" "$model_label" "$threads" "$batch_size" "$prompt_tokens" "$gen_tokens" \
    "$ttft_ms" "$decode_tps" "$tps_steady" "$tokens" \
    "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" \
    "$jitter" "$rss_gb" "$tok_per_dollar" \
    "$prefill_tps" "${MACHINE_TYPE:-unknown}" "$ARCH" "$(echo "$notes" | xargs)" \
    >> "$RESULT_CSV"

  if [[ "$phase" == "prefill" ]]; then
    log "  prefill t=$threads b=$batch_size seq=$prompt_tokens ttft=${ttft_ms}ms prefill_tps=${prefill_tps}"
  else
    log "  decode  t=$threads b=$batch_size ctx=$prompt_tokens gen=$gen_tokens tps=${tps_steady} p99_itl=${itl_p99}ms jitter=${jitter} rss=${rss_gb}GB tok/\$=${tok_per_dollar}"
  fi
}

# Run smoke test for a single model — returns 0 on pass
run_smoke() {
  local model="$1" model_label="$2"
  log "Smoke test: $model_label"
  local csv_file rss_file
  csv_file="$(mktemp)"
  rss_file="$(mktemp)"
  run_bench_with_rss "$model" 8 16 0 1 "$csv_file" "$rss_file"
  local csv_line
  csv_line="$(cat "$csv_file")"
  rm -f "$csv_file" "$rss_file"
  if [[ -z "$csv_line" ]]; then
    log "  FAIL: no output from bench for $model_label"
    return 1
  fi
  read -r ttft_ms _ tps_steady _ <<< "$(parse_bench_csv "$csv_line")"
  log "  PASS: $model_label ttft=${ttft_ms}ms tps_steady=${tps_steady}"
  return 0
}

# Run NUMA comparison for a model
run_numa_comparison() {
  local model="$1" model_label="$2" threads_per_node="$3"
  [[ "$NUMA_NODES" -le 1 ]] && return
  log_section "NUMA comparison: $model_label"

  for node in $(seq 0 $((NUMA_NODES - 1))); do
    local csv_file rss_file
    csv_file="$(mktemp)"
    rss_file="$(mktemp)"
    local time_stderr
    time_stderr="$(mktemp)"

    local args=(
      "$BENCH_BIN" "$model"
      1024 32 "$threads_per_node" "$RUNS"
      --batch-size 1 --csv --fair-mode
    )
    local csv_line=""
    if [[ "$TIME_V_AVAILABLE" == "available" ]]; then
      csv_line="$(
        env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
          DENSECORE_BENCH_RECORD_TOKEN_TIMES=1 \
          numactl --cpunodebind="$node" --membind="$node" \
          /usr/bin/time -v "${args[@]}" 2>"$time_stderr" | tail -n1
      )" || true
    else
      csv_line="$(
        env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
          DENSECORE_BENCH_RECORD_TOKEN_TIMES=1 \
          numactl --cpunodebind="$node" --membind="$node" \
          "${args[@]}" | tail -n1
      )" || true
    fi
    local rss_kb
    rss_kb="$(grep "Maximum resident" "$time_stderr" 2>/dev/null | awk '{print $NF}' || echo 0)"
    rm -f "$time_stderr" "$csv_file" "$rss_file"

    if [[ -z "$csv_line" ]]; then
      log "  SKIP: numa node=$node no output"
      continue
    fi
    read -r ttft_ms _ tps_steady _ itl_avg itl_p50 itl_p90 itl_p99 <<< "$(parse_bench_csv "$csv_line")"
    local rss_gb jitter tok_per_dollar
    rss_gb="$(awk -v k="$rss_kb" 'BEGIN { if(k>0) printf "%.2f", k/1048576; else print "0" }')"
    jitter="$(calc_jitter_ratio "$itl_p50" "$itl_p99")"
    tok_per_dollar="$(calc_tok_per_dollar "$tps_steady")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "numa_node${node}" "$model_label" "$threads_per_node" "1" "1024" "32" \
      "$ttft_ms" "0" "$tps_steady" "0" \
      "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" \
      "$jitter" "$rss_gb" "$tok_per_dollar" \
      "" "${MACHINE_TYPE:-unknown}" "$ARCH" "numa_node=${node}" \
      >> "$RESULT_CSV"

    log "  numa node=$node t=$threads_per_node tps=${tps_steady} p99_itl=${itl_p99}ms jitter=${jitter}"
  done
}

# Full sweep for a practical-tier model
run_practical_sweep() {
  local model="$1" model_label="$2"
  log_section "Practical sweep: $model_label"
  local threads batch ctx seq
  for threads in $PRACTICAL_THREAD_GRID; do
    for batch in $PRACTICAL_DECODE_BATCHES; do
      for ctx in $PRACTICAL_DECODE_CONTEXT; do
        bench_case "decode" "$model" "$model_label" \
          "$ctx" "$PRACTICAL_GEN_TOKENS" "$batch" "$threads"
      done
    done
  done
  for threads in $PRACTICAL_THREAD_GRID; do
    for batch in $PRACTICAL_DECODE_BATCHES; do
      for seq in $PRACTICAL_PREFILL_SEQ; do
        bench_case "prefill" "$model" "$model_label" \
          "$seq" "1" "$batch" "$threads"
      done
    done
  done
  local threads_per_node=$((PHYSICAL_CORES / NUMA_NODES))
  [[ "$threads_per_node" -le 0 ]] && threads_per_node=1
  run_numa_comparison "$model" "$model_label" "$threads_per_node"
}

# Selective sweep for viral-tier (giant sparse) models
run_viral_sweep() {
  local model="$1" model_label="$2"
  log_section "Viral selective sweep: $model_label"
  local threads batch ctx seq
  for threads in $VIRAL_THREAD_GRID; do
    for batch in $VIRAL_DECODE_BATCHES; do
      for ctx in $VIRAL_DECODE_CONTEXT; do
        bench_case "decode" "$model" "$model_label" \
          "$ctx" "$VIRAL_GEN_TOKENS" "$batch" "$threads"
      done
    done
  done
  for threads in $VIRAL_THREAD_GRID; do
    for batch in $VIRAL_DECODE_BATCHES; do
      for seq in $VIRAL_PREFILL_SEQ; do
        bench_case "prefill" "$model" "$model_label" \
          "$seq" "1" "$batch" "$threads"
      done
    done
  done
  local threads_per_node=$((PHYSICAL_CORES / NUMA_NODES))
  [[ "$threads_per_node" -le 0 ]] && threads_per_node=1
  run_numa_comparison "$model" "$model_label" "$threads_per_node"
}

# Smoke-only for bring-up tier models
run_bringup_smoke() {
  local model="$1" model_label="$2"
  log_section "Bring-up smoke: $model_label"
  if run_smoke "$model" "$model_label"; then
    # Single data point for bring-up confirmation
    bench_case "decode" "$model" "$model_label" \
      1024 32 1 0 "bringup_probe"
  fi
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
install_deps() {
  [[ "$SKIP_APT" == "1" ]] && { log "Skipping apt install"; return; }
  log "Installing build dependencies"
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential cmake curl g++ git jq ninja-build \
    numactl python3 python3-pip hwloc libhwloc-dev libnuma-dev time
  [[ "$ARCH" == "x86_64" ]] && sudo apt-get install -y libdnnl-dev || true
}

build_densecore() {
  [[ "$SKIP_BUILD" == "1" ]] && { log "Skipping build (SKIP_BUILD=1)"; return; }
  local onednn="OFF"
  [[ "$ARCH" == "x86_64" ]] && onednn="ON"
  log "Configuring DenseCore build"
  cmake -S core -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DDENSECORE_LTO=ON \
    -DDENSECORE_USE_SPDLOG=OFF \
    -DDENSECORE_BUILD_TESTS=ON \
    -DDENSECORE_BUILD_BENCHMARKS=OFF \
    -DDENSECORE_USE_ONEDNN="$onednn"
  log "Building DenseCore"
  cmake --build "$BUILD_DIR" -j "$JOBS"
  log "Building bench_native"
  g++ -O3 -DNDEBUG -std=c++17 \
    "$ROOT_DIR/benchmarks/bench_native.cpp" \
    -I"$ROOT_DIR/core/include" \
    -L"$BUILD_DIR" \
    -ldensecore -lpthread \
    -o "$BENCH_BIN"
}

run_mock_tests() {
  log "Running mock unit tests"
  LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
    "$BUILD_DIR/densecore_tests" --gtest_filter=NumaStickyRouting.* | tee -a "$TEST_LOG"
  LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
    "$BUILD_DIR/densecore_tests" --gtest_filter=EngineE2ETest.* | tee -a "$TEST_LOG"
}

# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
generate_summary() {
  log_section "Generating summary"

  local total_rows
  total_rows="$(awk -F',' 'NR>1{c++} END{print c+0}' "$RESULT_CSV")"

  {
    echo "# DenseCore GCP Benchmark Summary"
    echo ""
    echo "**Date**: ${timestamp}"
    echo "**Host**: \`${HOSTNAME_SHORT}\`"
    echo "**Machine type**: \`${MACHINE_TYPE:-unknown}\`"
    echo "**Zone**: \`${ZONE:-unknown}\`"
    echo "**Arch**: \`${ARCH}\`"
    echo "**CPU**: \`${CPU_MODEL}\`"
    echo "**Physical cores**: ${PHYSICAL_CORES} | **Logical cores**: ${LOGICAL_CORES} | **NUMA nodes**: ${NUMA_NODES}"
    echo "**Total memory**: ${MEM_GB} GB"
    echo "**Tier**: ${TIER} | **RUNS per case**: ${RUNS}"
    echo ""
    echo "---"
    echo ""
    echo "## CPO KPI Summary (Best decode case per model)"
    echo ""
    echo "| Model | TTFT (ms) | ITL p99 (ms) | Jitter (p99/p50) | Peak RSS (GB) | tps_steady | tok/s per \$ |"
    echo "|---|---|---|---|---|---|---|"

    # Per-model best decode row (highest tps_steady, columns: phase,model_label,threads,batch,ctx,gen,ttft,dtps,tps_steady,tokens,itl_avg,itl_p50,itl_p90,itl_p99,jitter,rss_gb,tok_per_dollar,prefill_tps,machine,arch,notes)
    awk -F',' '
      NR>1 && $1=="decode" {
        key=$2
        if (!(key in best) || $9+0 > best_tps[key]+0) {
          best[key]=$0
          best_tps[key]=$9+0
        }
      }
      END {
        for (k in best) {
          split(best[k], f, ",")
          printf "| %s | %s | %s | %s | %s | %s | %s |\n", \
            f[2], f[7], f[14], f[15], f[16], f[9], f[17]
        }
      }
    ' "$RESULT_CSV"

    echo ""
    echo "---"
    echo ""
    echo "## NUMA Comparison"
    echo ""
    echo "| Model | NUMA node | tps_steady | ITL p99 (ms) | Jitter |"
    echo "|---|---|---|---|---|"
    awk -F',' '
      NR>1 && $1 ~ /^numa_/ {
        printf "| %s | %s | %s | %s | %s |\n", $2, $1, $9, $14, $15
      }
    ' "$RESULT_CSV"

    echo ""
    echo "---"
    echo ""
    echo "## Thread Scaling (decode, batch=1)"
    echo ""
    echo "| Model | Threads | tps_steady | TTFT (ms) | ITL p99 (ms) |"
    echo "|---|---|---|---|---|"
    awk -F',' '
      NR>1 && $1=="decode" && $4=="1" {
        printf "| %s | %s | %s | %s | %s |\n", $2, $3, $9, $7, $14
      }
    ' "$RESULT_CSV" | sort -t'|' -k2,2 -k3,3n

    echo ""
    echo "---"
    echo ""
    echo "## Batch Scaling (decode, max threads, ctx=8192)"
    echo ""
    echo "| Model | Batch | tps_steady | TTFT (ms) | ITL p99 (ms) | Jitter |"
    echo "|---|---|---|---|---|---|"
    awk -F',' -v maxth="$PHYSICAL_CORES" '
      NR>1 && $1=="decode" && $3==maxth && $5=="8192" {
        printf "| %s | %s | %s | %s | %s | %s |\n", $2, $4, $9, $7, $14, $15
      }
    ' "$RESULT_CSV" | sort -t'|' -k2,2 -k3,3n

    echo ""
    echo "---"
    echo ""
    echo "## Artifacts"
    echo ""
    echo "- Results CSV: \`$RESULT_CSV\`"
    echo "- Test log: \`$TEST_LOG\`"
    echo "- Inventory: \`$INVENTORY_TXT\`"
    echo ""
    echo "Total data points collected: ${total_rows}"

  } > "$SUMMARY_MD"

  log "Summary written to: $SUMMARY_MD"
  echo ""
  cat "$SUMMARY_MD"
}

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
write_inventory() {
  {
    echo "timestamp=$timestamp"
    echo "hostname=$HOSTNAME_SHORT"
    echo "machine_type=${MACHINE_TYPE:-unknown}"
    echo "instance_name=${INSTANCE_NAME:-unknown}"
    echo "zone=${ZONE:-unknown}"
    echo "project_id=${PROJECT_ID:-unknown}"
    echo "arch=$ARCH"
    echo "cpu_model=$CPU_MODEL"
    echo "logical_cores=$LOGICAL_CORES"
    echo "physical_cores=$PHYSICAL_CORES"
    echo "numa_nodes=$NUMA_NODES"
    echo "mem_gb=${MEM_GB:-unknown}"
    echo "tier=$TIER"
    echo "price_per_hour=${PRICE_PER_HOUR}"
    echo "runs=$RUNS"
    echo ""
    echo "===== lscpu ====="
    lscpu
    echo ""
    echo "===== numactl --hardware ====="
    numactl --hardware 2>/dev/null || true
    echo ""
    echo "===== free -h ====="
    free -h
  } > "$INVENTORY_TXT"
  env | sort > "$ENV_TXT"
}

# ---------------------------------------------------------------------------
# Model list builder per tier
# ---------------------------------------------------------------------------
# Returns newline-separated "LABEL:PATH" pairs for the selected tier
get_model_list() {
  local -a models=()

  case "$TIER" in
    practical|all)
      [[ -n "${MODEL_QWEN35_35B:-}" ]] && models+=("Qwen3.5-35B-A3B:${MODEL_QWEN35_35B}:practical")
      [[ -n "${MODEL_QWEN35_27B:-}" ]] && models+=("Qwen3.5-27B:${MODEL_QWEN35_27B}:practical")
      [[ -n "${MODEL_QWEN3_CODER_30B:-}" ]] && models+=("Qwen3-Coder-30B-A3B:${MODEL_QWEN3_CODER_30B}:practical")
      [[ -n "${MODEL_GEMMA3_27B:-}" ]] && models+=("Gemma3-27B-IT:${MODEL_GEMMA3_27B}:practical")
      ;;
  esac
  case "$TIER" in
    viral|all)
      [[ -n "${MODEL_QWEN35_397B:-}" ]] && models+=("Qwen3.5-397B-A17B:${MODEL_QWEN35_397B}:viral")
      ;;
  esac
  case "$TIER" in
    bringup|all)
      [[ -n "${MODEL_GLM45_AIR:-}" ]] && models+=("GLM-4.5-Air:${MODEL_GLM45_AIR}:bringup")
      [[ -n "${MODEL_GLM5:-}" ]] && models+=("GLM-5:${MODEL_GLM5}:bringup")
      ;;
  esac

  if [[ "${#models[@]}" -eq 0 ]]; then
    log "ERROR: No model paths provided for tier='$TIER'."
    log "Set at least one of: MODEL_QWEN35_35B, MODEL_QWEN35_27B, MODEL_QWEN3_CODER_30B,"
    log "  MODEL_GEMMA3_27B, MODEL_QWEN35_397B, MODEL_GLM45_AIR, MODEL_GLM5"
    exit 1
  fi

  printf '%s\n' "${models[@]}"
}

# ---------------------------------------------------------------------------
# Dry run: print plan only
# ---------------------------------------------------------------------------
print_plan() {
  log_section "DRY RUN — benchmark plan"
  log "Tier: $TIER"
  log "Machine: ${MACHINE_TYPE:-unknown} (${ARCH}) physical_cores=${PHYSICAL_CORES} numa_nodes=${NUMA_NODES}"
  log "AUTO_SHUTDOWN: $AUTO_SHUTDOWN (delay: ${SHUTDOWN_DELAY_MIN}m)"
  log "PRICE_PER_HOUR: ${PRICE_PER_HOUR}"
  log "RUNS per case: $RUNS"
  echo ""
  log "Models queued:"
  while IFS=: read -r label path tier; do
    if [[ -f "$path" ]]; then
      local size_gb
      size_gb="$(du -sh "$path" 2>/dev/null | cut -f1 || echo '?')"
      log "  [$tier] $label  →  $path  ($size_gb)"
    else
      log "  [$tier] $label  →  $path  [FILE NOT FOUND — will be skipped]"
    fi
  done < <(get_model_list)
  echo ""
  log "Thread grids:"
  case "$TIER" in practical|all)
    log "  practical: $PRACTICAL_THREAD_GRID"
    log "  practical decode batches: $PRACTICAL_DECODE_BATCHES"
    log "  practical decode context: $PRACTICAL_DECODE_CONTEXT"
    ;;
  esac
  case "$TIER" in viral|all)
    log "  viral: $VIRAL_THREAD_GRID"
    log "  viral decode batches: $VIRAL_DECODE_BATCHES"
    log "  viral decode context: $VIRAL_DECODE_CONTEXT"
    ;;
  esac
  log ""
  log "To run for real, remove DRY_RUN=1"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  mkdir -p "$OUT_DIR"
  RESULT_CSV="$OUT_DIR/results.csv"
  TEST_LOG="$OUT_DIR/test.log"
  SUMMARY_MD="$OUT_DIR/summary.md"
  INVENTORY_TXT="$OUT_DIR/inventory.txt"
  ENV_TXT="$OUT_DIR/env.txt"

  export DENSECORE_BENCH_MODE=1
  export DENSECORE_BENCH_TPS_MODE=all
  export DENSECORE_BENCH_DIRECT_CALLBACK=1
  export DENSECORE_BENCH_DECODE_BATCH_FAST_PATH=1
  export DENSECORE_BENCH_FAST_PATH_MAX_BATCH=16
  export DENSECORE_BENCH_RESPECT_THREADS=1
  export DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE=on

  if [[ "$DRY_RUN" == "1" ]]; then
    print_plan
    return 0
  fi

  log_section "DenseCore GCP Autorun — $(date)"
  log "Output dir: $OUT_DIR"
  log "Tier: $TIER | RUNS: $RUNS | AUTO_SHUTDOWN: $AUTO_SHUTDOWN"

  write_inventory

  # Check /usr/bin/time -v availability
  TIME_V_AVAILABLE="$(require_time_v)"
  export TIME_V_AVAILABLE

  # CSV header
  printf '%s\n' \
    "phase,model_label,threads,batch_size,prompt_tokens,gen_tokens,ttft_ms,decode_tps,tps_steady,tokens_generated,itl_avg_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms,jitter_p99p50,peak_rss_gb,tok_per_dollar,prefill_tps,machine_type,arch,notes" \
    > "$RESULT_CSV"

  # Stage 1: Install deps
  log_section "Stage 1: Install dependencies"
  install_deps

  # Stage 2: Build
  log_section "Stage 2: Build DenseCore"
  build_densecore

  # Stage 3: Mock tests
  log_section "Stage 3: Mock e2e tests"
  run_mock_tests

  # Stage 4: Per-model smoke + bench
  log_section "Stage 4: Per-model benchmark"

  local failed_models=()
  local passed_models=()

  while IFS=: read -r label path tier; do
    echo ""
    log "--- Model: $label ---"
    if [[ ! -f "$path" ]]; then
      log "  SKIP: file not found: $path"
      failed_models+=("$label (file not found)")
      continue
    fi

    # Smoke test first
    if ! run_smoke "$path" "$label"; then
      log "  FAIL: smoke failed for $label — skipping bench"
      failed_models+=("$label (smoke failed)")
      continue
    fi
    passed_models+=("$label")

    # Tier-appropriate sweep
    case "$tier" in
      practical)
        run_practical_sweep "$path" "$label"
        ;;
      viral)
        run_viral_sweep "$path" "$label"
        ;;
      bringup)
        run_bringup_smoke "$path" "$label"
        ;;
      *)
        log "  Unknown tier '$tier' for $label — running practical sweep"
        run_practical_sweep "$path" "$label"
        ;;
    esac

  done < <(get_model_list)

  # Stage 5: Summary
  log_section "Stage 5: Summary"
  generate_summary

  # Status report
  echo ""
  log "Passed models: ${passed_models[*]:-none}"
  if [[ "${#failed_models[@]}" -gt 0 ]]; then
    log "Failed/skipped models: ${failed_models[*]}"
  fi

  log "All artifacts in: $OUT_DIR"
  log "Autorun complete."
}

main "$@"
