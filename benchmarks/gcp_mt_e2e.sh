#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  benchmarks/gcp_mt_e2e.sh --model /path/to/model.gguf [--mode all|prepare|build|test|bench]

Purpose:
  Prepare a Google Cloud CPU VM, build DenseCore, run mock/real e2e checks,
  and sweep multi-thread prefill/decode cases on Arm (Axion, Ampere) / Intel / AMD hosts.

Required:
  --model PATH     GGUF model path on the Google Cloud VM

Optional:
  --mode MODE      all | prepare | build | test | bench (default: all)
  --out-dir DIR    Output directory (default: benchmarks/results/gcp_mt_e2e_<host>_<ts>)

Useful env overrides:
  JOBS=0
  THREAD_GRID="1 16 32 48 64 96 128"
  DECODE_BATCHES="1 4 8 16"
  DECODE_CONTEXT_LENS="128 1024 4096 8192"
  PREFILL_BATCHES="1 4 8 16"
  PREFILL_SEQ_LENS="512 1024 2048 4096"
  DECODE_GEN_TOKENS=128
  RUNS=2
  FAIR_MODE=1
  PARALLEL_SUBMIT=0
  RUN_NUMA_SMOKE=1
  SKIP_APT=0
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="all"
MODEL=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "[ERROR] --model is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "[ERROR] model not found: $MODEL" >&2
  exit 1
fi

case "$MODE" in
  all|prepare|build|test|bench) ;;
  *)
    echo "[ERROR] invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
HOSTNAME_SHORT="$(hostname -s)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/results/gcp_mt_e2e_${HOSTNAME_SHORT}_${timestamp}}"
mkdir -p "$OUT_DIR"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BENCH_BIN="${BENCH_BIN:-$ROOT_DIR/benchmarks/bench_native}"
RUNS="${RUNS:-2}"
FAIR_MODE="${FAIR_MODE:-1}"
PARALLEL_SUBMIT="${PARALLEL_SUBMIT:-0}"
SKIP_APT="${SKIP_APT:-0}"
RUN_NUMA_SMOKE="${RUN_NUMA_SMOKE:-0}"
DECODE_GEN_TOKENS="${DECODE_GEN_TOKENS:-128}"
DECODE_BATCHES="${DECODE_BATCHES:-1 4 8 16}"
DECODE_CONTEXT_LENS="${DECODE_CONTEXT_LENS:-128 1024 4096 8192}"
PREFILL_BATCHES="${PREFILL_BATCHES:-1 4 8 16}"
PREFILL_SEQ_LENS="${PREFILL_SEQ_LENS:-512 1024 2048 4096}"
REAL_SMOKE_PROMPT_TOKENS="${REAL_SMOKE_PROMPT_TOKENS:-8}"
REAL_SMOKE_GEN_TOKENS="${REAL_SMOKE_GEN_TOKENS:-16}"
NUMA_SMOKE_PROMPT_TOKENS="${NUMA_SMOKE_PROMPT_TOKENS:-1024}"
NUMA_SMOKE_GEN_TOKENS="${NUMA_SMOKE_GEN_TOKENS:-32}"

RESULT_CSV="$OUT_DIR/multithread_results.csv"
TEST_LOG="$OUT_DIR/test.log"
REAL_SMOKE_LOG="$OUT_DIR/real_model_smoke.log"
SUMMARY_MD="$OUT_DIR/summary.md"
INVENTORY_TXT="$OUT_DIR/inventory.txt"
ENV_TXT="$OUT_DIR/environment.txt"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}

gcp_metadata_get() {
  local path="$1"
  curl -fsS --max-time 1 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${path}" 2>/dev/null || true
}

basename_or_unknown() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf 'unknown'
    return
  fi
  printf '%s' "${value##*/}"
}

ARCH="$(uname -m)"
MACHINE_TYPE_RAW="$(gcp_metadata_get instance/machine-type)"
MACHINE_TYPE="$(basename_or_unknown "$MACHINE_TYPE_RAW")"
INSTANCE_ID="$(gcp_metadata_get instance/id)"
INSTANCE_NAME="$(gcp_metadata_get instance/name)"
PROJECT_ID="$(gcp_metadata_get project/project-id)"
ZONE_RAW="$(gcp_metadata_get instance/zone)"
ZONE="$(basename_or_unknown "$ZONE_RAW")"
CPU_MODEL="$(lscpu | awk -F: '/Model name:/ {print $2; exit}' | xargs)"
LOGICAL_CORES="$(nproc)"
PHYSICAL_CORES="$(lscpu -p=core | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
NUMA_NODES="$(lscpu | awk -F: '/NUMA node\(s\)/ {gsub(/ /, "", $2); print $2; exit}')"
MEM_GB="$(free -g | awk '/^Mem:/ {print $2}')"

if [[ -z "$PHYSICAL_CORES" || "$PHYSICAL_CORES" -le 0 ]]; then
  PHYSICAL_CORES="$LOGICAL_CORES"
fi
if [[ -z "$NUMA_NODES" || "$NUMA_NODES" -le 0 ]]; then
  NUMA_NODES=1
fi

build_thread_grid_default() {
  local -A seen=()
  local candidates=()
  local value

  candidates+=(1)
  candidates+=($((PHYSICAL_CORES / 4)))
  candidates+=($((PHYSICAL_CORES / 2)))
  candidates+=($(((PHYSICAL_CORES * 3) / 4)))
  candidates+=("$PHYSICAL_CORES")
  if (( LOGICAL_CORES > PHYSICAL_CORES )); then
    candidates+=("$LOGICAL_CORES")
  fi

  local out=()
  for value in "${candidates[@]}"; do
    if [[ -z "$value" || "$value" -le 0 ]]; then
      continue
    fi
    if [[ -z "${seen[$value]:-}" ]]; then
      seen[$value]=1
      out+=("$value")
    fi
  done

  printf '%s\n' "${out[*]}"
}

THREAD_GRID="${THREAD_GRID:-$(build_thread_grid_default)}"
JOBS="${JOBS:-$LOGICAL_CORES}"
if [[ "$JOBS" -le 0 ]]; then
  JOBS=1
fi

export DENSECORE_BENCH_MODE=1
export DENSECORE_BENCH_TPS_MODE="${DENSECORE_BENCH_TPS_MODE:-all}"
export DENSECORE_BENCH_DIRECT_CALLBACK="${DENSECORE_BENCH_DIRECT_CALLBACK:-1}"
export DENSECORE_BENCH_DECODE_BATCH_FAST_PATH="${DENSECORE_BENCH_DECODE_BATCH_FAST_PATH:-1}"
export DENSECORE_BENCH_FAST_PATH_MAX_BATCH="${DENSECORE_BENCH_FAST_PATH_MAX_BATCH:-16}"
export DENSECORE_BENCH_RECORD_TOKEN_TIMES="${DENSECORE_BENCH_RECORD_TOKEN_TIMES:-0}"
export DENSECORE_BENCH_RESPECT_THREADS="${DENSECORE_BENCH_RESPECT_THREADS:-1}"
export DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE="${DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE:-on}"
export DENSECORE_DECODE_GRAPH_CACHE="${DENSECORE_DECODE_GRAPH_CACHE:-0}"
export DENSECORE_ENABLE_Q4K_BATCHED_KERNEL="${DENSECORE_ENABLE_Q4K_BATCHED_KERNEL:-0}"
export DENSECORE_ENABLE_QUANT_NRC_BATCH="${DENSECORE_ENABLE_QUANT_NRC_BATCH:-0}"

write_inventory() {
  {
    echo "timestamp=$timestamp"
    echo "hostname=$HOSTNAME_SHORT"
    echo "machine_type=${MACHINE_TYPE:-unknown}"
    echo "instance_name=${INSTANCE_NAME:-unknown}"
    echo "instance_id=${INSTANCE_ID:-unknown}"
    echo "project_id=${PROJECT_ID:-unknown}"
    echo "zone=${ZONE:-unknown}"
    echo "arch=$ARCH"
    echo "cpu_model=$CPU_MODEL"
    echo "logical_cores=$LOGICAL_CORES"
    echo "physical_cores=$PHYSICAL_CORES"
    echo "numa_nodes=$NUMA_NODES"
    echo "mem_gb=${MEM_GB:-unknown}"
    echo "thread_grid=$THREAD_GRID"
    echo
    echo "===== uname -a ====="
    uname -a
    echo
    echo "===== lscpu ====="
    lscpu
    echo
    echo "===== numactl --hardware ====="
    numactl --hardware 2>/dev/null || true
    echo
    echo "===== free -h ====="
    free -h
  } > "$INVENTORY_TXT"

  env | sort > "$ENV_TXT"
}

prepare_host() {
  if [[ "$SKIP_APT" == "1" ]]; then
    log "Skipping apt dependency installation"
    return
  fi

  log "Installing build/runtime dependencies"
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    git \
    jq \
    ninja-build \
    numactl \
    python3 \
    python3-pip \
    hwloc \
    libhwloc-dev \
    libnuma-dev

  if [[ "$ARCH" == "x86_64" ]]; then
    sudo apt-get install -y libdnnl-dev || true
  fi
}

build_densecore() {
  local onednn="OFF"
  if [[ "$ARCH" == "x86_64" ]]; then
    onednn="ON"
  fi

  log "Configuring DenseCore build in $BUILD_DIR"
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
    -ldensecore \
    -lpthread \
    -o "$BENCH_BIN"
}

run_mock_tests() {
  log "Running thread/NUMA unit tests"
  LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
    "$BUILD_DIR/densecore_tests" --gtest_filter=NumaStickyRouting.* | tee -a "$TEST_LOG"

  log "Running mock engine e2e tests"
  LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
    "$BUILD_DIR/densecore_tests" --gtest_filter=EngineE2ETest.* | tee -a "$TEST_LOG"
}

run_real_model_smoke() {
  log "Running real-model smoke benchmark"
  local args=(
    "$BENCH_BIN"
    "$MODEL"
    "$REAL_SMOKE_PROMPT_TOKENS"
    "$REAL_SMOKE_GEN_TOKENS"
    "0"
    "1"
    --csv
  )
  if [[ "$FAIR_MODE" == "1" ]]; then
    args+=(--fair-mode)
  fi
  if [[ "$PARALLEL_SUBMIT" == "1" ]]; then
    args+=(--parallel-submit)
  fi

  env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
    "${args[@]}" | tee "$REAL_SMOKE_LOG" >/dev/null
}

parse_bench_csv_line() {
  local line="$1"
  IFS=',' read -r ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$line"
  if [[ -z "${ttft_ms:-}" || -z "${decode_tps:-}" || -z "${tps_steady:-}" || -z "${tokens:-}" ]]; then
    echo "[ERROR] failed to parse benchmark CSV: $line" >&2
    exit 1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$ttft_ms" "$decode_tps" "$tps_steady" "$tokens" "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99"
}

run_case() {
  local phase="$1"
  local prompt_tokens="$2"
  local gen_tokens="$3"
  local batch_size="$4"
  local threads="$5"
  local notes="${6:-}"

  local args=(
    "$BENCH_BIN"
    "$MODEL"
    "$prompt_tokens"
    "$gen_tokens"
    "$threads"
    "$RUNS"
    --batch-size "$batch_size"
    --csv
  )
  if [[ "$FAIR_MODE" == "1" ]]; then
    args+=(--fair-mode)
  fi
  if [[ "$PARALLEL_SUBMIT" == "1" ]]; then
    args+=(--parallel-submit)
  fi

  local line
  line="$(env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" "${args[@]}" | tail -n1)"
  local parsed
  parsed="$(parse_bench_csv_line "$line")"

  local ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99
  IFS=',' read -r ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$parsed"

  local prefill_tps=""
  if [[ "$phase" == "prefill" ]]; then
    prefill_tps="$(awk -v p="$prompt_tokens" -v b="$batch_size" -v t="$ttft_ms" \
      'BEGIN { if (t <= 0) { print 0.0 } else { printf "%.2f", (p * b) / (t / 1000.0) } }')"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$phase" "$threads" "$batch_size" "$prompt_tokens" "$gen_tokens" \
    "$ttft_ms" "$decode_tps" "$tps_steady" "$tokens" "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" \
    "$prefill_tps" "${MACHINE_TYPE:-unknown}" "$ARCH" "$(trim "$notes")" >> "$RESULT_CSV"

  if [[ "$phase" == "prefill" ]]; then
    log "prefill threads=$threads batch=$batch_size seq=$prompt_tokens ttft_ms=$ttft_ms prefill_tps=$prefill_tps"
  else
    log "decode  threads=$threads batch=$batch_size ctx=$prompt_tokens gen=$gen_tokens tps=$decode_tps steady=$tps_steady"
  fi
}

run_decode_sweep() {
  log "Running decode thread sweep"
  local threads batch_size context_len
  for threads in $THREAD_GRID; do
    for batch_size in $DECODE_BATCHES; do
      for context_len in $DECODE_CONTEXT_LENS; do
        run_case "decode" "$context_len" "$DECODE_GEN_TOKENS" "$batch_size" "$threads"
      done
    done
  done
}

run_prefill_sweep() {
  log "Running prefill thread sweep"
  local threads batch_size seq_len
  for threads in $THREAD_GRID; do
    for batch_size in $PREFILL_BATCHES; do
      for seq_len in $PREFILL_SEQ_LENS; do
        run_case "prefill" "$seq_len" "1" "$batch_size" "$threads"
      done
    done
  done
}

run_numa_smoke() {
  if [[ "$RUN_NUMA_SMOKE" != "1" || "$NUMA_NODES" -le 1 ]]; then
    return
  fi

  log "Running NUMA-local smoke checks"
  local threads_per_node=$((PHYSICAL_CORES / NUMA_NODES))
  if [[ "$threads_per_node" -le 0 ]]; then
    threads_per_node=1
  fi

  local node
  for ((node = 0; node < NUMA_NODES; ++node)); do
    local line
    line="$(
      numactl --cpunodebind="$node" --membind="$node" \
        env LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
        "$BENCH_BIN" "$MODEL" "$NUMA_SMOKE_PROMPT_TOKENS" "$NUMA_SMOKE_GEN_TOKENS" "$threads_per_node" 1 \
        --fair-mode --csv | tail -n1
    )"

    local parsed
    parsed="$(parse_bench_csv_line "$line")"
    local ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99
    IFS=',' read -r ttft_ms decode_tps tps_steady tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$parsed"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "numa_smoke" "$threads_per_node" "1" "$NUMA_SMOKE_PROMPT_TOKENS" "$NUMA_SMOKE_GEN_TOKENS" \
      "$ttft_ms" "$decode_tps" "$tps_steady" "$tokens" "$itl_avg" "$itl_p50" "$itl_p90" "$itl_p99" \
      "" "${MACHINE_TYPE:-unknown}" "$ARCH" "node=${node}" >> "$RESULT_CSV"
    log "numa node=$node threads=$threads_per_node tps=$decode_tps"
  done
}

generate_summary() {
  local best_decode best_prefill
  best_decode="$(awk -F',' 'NR > 1 && $1 == "decode" { if ($7 + 0 > best) { best = $7 + 0; line = $0 } } END { print line }' "$RESULT_CSV")"
  best_prefill="$(awk -F',' 'NR > 1 && $1 == "prefill" { if ($14 + 0 > best) { best = $14 + 0; line = $0 } } END { print line }' "$RESULT_CSV")"

  {
    echo "# Google Cloud Multi-thread E2E Summary"
    echo
    echo "- Host: \`$HOSTNAME_SHORT\`"
    echo "- Machine type: \`${MACHINE_TYPE:-unknown}\`"
    echo "- Project: \`${PROJECT_ID:-unknown}\`"
    echo "- Zone: \`${ZONE:-unknown}\`"
    echo "- Arch: \`$ARCH\`"
    echo "- CPU: \`$CPU_MODEL\`"
    echo "- Physical cores: \`$PHYSICAL_CORES\`"
    echo "- Logical cores: \`$LOGICAL_CORES\`"
    echo "- NUMA nodes: \`$NUMA_NODES\`"
    echo "- Thread grid: \`$THREAD_GRID\`"
    echo "- Model: \`$MODEL\`"
    echo
    echo "## Best Decode Case"
    echo
    echo '```csv'
    if [[ -n "$best_decode" ]]; then
      echo "$best_decode"
    else
      echo "none"
    fi
    echo '```'
    echo
    echo "## Best Prefill Case"
    echo
    echo '```csv'
    if [[ -n "$best_prefill" ]]; then
      echo "$best_prefill"
    else
      echo "none"
    fi
    echo '```'
    echo
    echo "## Artifacts"
    echo
    echo "- \`$RESULT_CSV\`"
    echo "- \`$TEST_LOG\`"
    echo "- \`$REAL_SMOKE_LOG\`"
    echo "- \`$INVENTORY_TXT\`"
    echo "- \`$ENV_TXT\`"
  } > "$SUMMARY_MD"
}

write_inventory
echo "phase,threads,batch_size,prompt_tokens,gen_tokens,ttft_ms,decode_tps,tps_steady,tokens_generated,itl_avg_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms,prefill_tps,machine_type,arch,notes" > "$RESULT_CSV"

case "$MODE" in
  prepare)
    prepare_host
    ;;
  build)
    build_densecore
    ;;
  test)
    run_mock_tests
    run_real_model_smoke
    ;;
  bench)
    run_decode_sweep
    run_prefill_sweep
    run_numa_smoke
    generate_summary
    ;;
  all)
    prepare_host
    build_densecore
    run_mock_tests
    run_real_model_smoke
    run_decode_sweep
    run_prefill_sweep
    run_numa_smoke
    generate_summary
    ;;
esac

log "Completed mode=$MODE"
log "Artifacts: $OUT_DIR"
