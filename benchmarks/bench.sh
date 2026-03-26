/fg:#!/bin/bash
# =============================================================================
# DenseCore Benchmark Suite
# =============================================================================
# Non-interactive benchmark runner with llama.cpp comparison.
# Outputs structured JSON for CI/reporting.
#
# Usage:
#   ./bench.sh                         # Auto-detect model from cache
#   ./bench.sh --model /path/to.gguf   # Specific model
#   ./bench.sh --compare-llamacpp      # Include llama.cpp comparison
#   ./bench.sh --prompt-tokens 512     # Custom prompt length
#   ./bench.sh --gen-tokens 256        # Custom generation length
#   ./bench.sh --threads 8             # Explicit thread count
#   ./bench.sh --runs 3                # Average over N runs
#   ./bench.sh --json                  # JSON-only output (for CI)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
MODEL_PATH=""
PROMPT_TOKENS=128
GEN_TOKENS=128
THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
RUNS=1
BATCH_SIZE=1
COMPARE_LLAMACPP=false
JSON_ONLY=false
VERIFY_PAGED_PARITY=false
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"

# Colors (disabled in JSON mode)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --compare-llamacpp) COMPARE_LLAMACPP=true; shift ;;
        --verify-paged-parity) VERIFY_PAGED_PARITY=true; shift ;;
        --json) JSON_ONLY=true; shift ;;
        --llamacpp-dir) LLAMACPP_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH          Path to GGUF model file"
            echo "  --prompt-tokens N     Prompt length in tokens (default: 128)"
            echo "  --gen-tokens N        Tokens to generate (default: 128)"
            echo "  --threads N           CPU threads (default: auto)"
            echo "  --runs N              Number of runs to average (default: 1)"
            echo "  --batch-size N        Concurrent requests per run (default: 1)"
            echo "  --compare-llamacpp    Build and run llama.cpp for comparison"
            echo "  --verify-paged-parity Run deterministic parity check (dense decode vs paged decode)"
            echo "  --llamacpp-dir PATH   Path to llama.cpp source (default: ~/llama.cpp)"
            echo "  --json                JSON-only output (for CI pipelines)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $JSON_ONLY; then
    RED="" GREEN="" YELLOW="" CYAN="" BOLD="" NC=""
fi

log() {
    if ! $JSON_ONLY; then
        echo -e "$@"
    fi
}

# =============================================================================
# Hardware Detection
# =============================================================================
detect_hardware() {
    local cpu_model arch cores simd_level mem_gb

    if [[ "$(uname)" == "Darwin" ]]; then
        cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        arch=$(uname -m)
        cores=$(sysctl -n hw.ncpu)
        mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
        simd_level="NEON"
        if [[ "$arch" == "arm64" ]]; then
            simd_level="NEON + Apple AMX"
        fi
    else
        cpu_model=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || echo "Unknown")
        arch=$(uname -m)
        cores=$(nproc)
        mem_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 ))

        if [[ "$arch" == "aarch64" ]]; then
            if grep -q sve2 /proc/cpuinfo 2>/dev/null; then
                simd_level="SVE2"
            elif grep -q sve /proc/cpuinfo 2>/dev/null; then
                simd_level="SVE"
            elif grep -q asimddp /proc/cpuinfo 2>/dev/null; then
                simd_level="NEON DOTPROD"
            else
                simd_level="NEON"
            fi
        else
            if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
                if grep -q amx /proc/cpuinfo 2>/dev/null; then
                    simd_level="AVX-512 + AMX"
                elif grep -q avx512vnni /proc/cpuinfo 2>/dev/null; then
                    simd_level="AVX-512 VNNI"
                else
                    simd_level="AVX-512"
                fi
            elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
                simd_level="AVX2"
            elif grep -q sse4_1 /proc/cpuinfo 2>/dev/null; then
                simd_level="SSE4.1"
            else
                simd_level="Scalar"
            fi
        fi
    fi

    HW_CPU="$cpu_model"
    HW_ARCH="$arch"
    HW_CORES="$cores"
    HW_SIMD="$simd_level"
    HW_MEM_GB="$mem_gb"
}

# =============================================================================
# Model Discovery
# =============================================================================
find_model() {
    if [[ -n "$MODEL_PATH" ]]; then
        if [[ ! -f "$MODEL_PATH" ]]; then
            echo "Error: Model not found: $MODEL_PATH" >&2
            exit 1
        fi
        return
    fi

    # Search HuggingFace cache
    local cache_dir="$HOME/.cache/huggingface/hub"
    if [[ -d "$cache_dir" ]]; then
        local preferred_patterns=(
            "*qwen3*0.6b*.gguf"
            "*qwen3*.gguf"
            "*qwen2.5*.gguf"
            "*.gguf"
        )
        local pattern
        for pattern in "${preferred_patterns[@]}"; do
            MODEL_PATH=$(find "$cache_dir" -type f -iname "$pattern" 2>/dev/null | head -1 || true)
            if [[ -n "$MODEL_PATH" ]]; then
                break
            fi
        done
    fi

    if [[ -z "$MODEL_PATH" ]]; then
        echo "Error: No model found. Use --model or download one first:" >&2
        echo "  pip install huggingface-hub" >&2
        echo "  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF --include '*.gguf'" >&2
        exit 1
    fi

    log "${CYAN}Auto-detected model:${NC} $(basename "$MODEL_PATH")"
}

# =============================================================================
# DenseCore Benchmark
# =============================================================================
ensure_densecore_built() {
    local lib_path=""

    # Check common build locations
    for candidate in \
        "$ROOT_DIR/core/build/libdensecore.so" \
        "$ROOT_DIR/core/build/libdensecore.dylib" \
        "$ROOT_DIR/build/libdensecore.so" \
        "$ROOT_DIR/build/libdensecore.dylib"; do
        if [[ -f "$candidate" ]]; then
            lib_path="$candidate"
            break
        fi
    done

    if [[ -z "$lib_path" ]]; then
        log "${YELLOW}Building DenseCore...${NC}"
        (cd "$ROOT_DIR" && make lib 2>&1 | tail -5)
    fi
}

ensure_bench_native_built() {
    local bench_bin="$SCRIPT_DIR/bench_native"
    local src="$SCRIPT_DIR/bench_native.cpp"

    local lib_dir="$ROOT_DIR/build"
    if [[ ! -f "$lib_dir/libdensecore.so" ]] && [[ ! -f "$lib_dir/libdensecore.dylib" ]]; then
        lib_dir="$ROOT_DIR/core/build"
    fi

    log "${YELLOW}Building bench_native...${NC}"
    g++ -std=c++17 -O2 -o "$bench_bin" "$src" -I"$ROOT_DIR/core/include" -L"$lib_dir" -ldensecore -lpthread \
        -Wl,-rpath,"$lib_dir" >/dev/null 2>&1
}

ensure_quality_test_built() {
    local quality_bin="$SCRIPT_DIR/quality_test"
    local src="$SCRIPT_DIR/quality_test.cpp"

    local lib_dir="$ROOT_DIR/build"
    if [[ ! -f "$lib_dir/libdensecore.so" ]] && [[ ! -f "$lib_dir/libdensecore.dylib" ]]; then
        lib_dir="$ROOT_DIR/core/build"
    fi

    log "${YELLOW}Building quality_test...${NC}"
    g++ -std=c++17 -O2 -o "$quality_bin" "$src" -I"$ROOT_DIR/core/include" -L"$lib_dir" -ldensecore -lpthread \
        -Wl,-rpath,"$lib_dir" >/dev/null 2>&1
}

verify_paged_parity() {
    PARITY_STATUS="not_run"
    if ! $VERIFY_PAGED_PARITY; then
        return
    fi

    ensure_densecore_built
    ensure_quality_test_built

    local lib_dir="$ROOT_DIR/build"
    if [[ ! -f "$lib_dir/libdensecore.so" ]] && [[ ! -f "$lib_dir/libdensecore.dylib" ]]; then
        lib_dir="$ROOT_DIR/core/build"
    fi

    log "${BOLD}Running paged decode parity check...${NC}"
    if LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}" \
        "$SCRIPT_DIR/quality_test" "$MODEL_PATH" "$THREADS" --parity --paged-mode on >/tmp/densecore_parity.log 2>&1; then
        PARITY_STATUS="pass"
        log "  Parity: ${GREEN}PASS${NC}"
    else
        PARITY_STATUS="fail"
        log "  Parity: ${RED}FAIL${NC}"
        if ! $JSON_ONLY; then
            tail -n 20 /tmp/densecore_parity.log || true
        fi
        echo "Error: paged decode parity check failed" >&2
        return 1
    fi
}

bench_densecore() {
    local run_results=()

    ensure_densecore_built
    ensure_bench_native_built
    local lib_dir="$ROOT_DIR/build"
    if [[ ! -f "$lib_dir/libdensecore.so" ]] && [[ ! -f "$lib_dir/libdensecore.dylib" ]]; then
        lib_dir="$ROOT_DIR/core/build"
    fi

    log "${BOLD}Running DenseCore benchmark (${RUNS} run(s))...${NC}"

    for i in $(seq 1 "$RUNS"); do
        log "  Run $i/$RUNS..."

        local result
        result=$(DENSECORE_AUTO_CHAT_TEMPLATE=0 DENSECORE_BENCH_MODE=1 \
            LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}" \
            "$SCRIPT_DIR/bench_native" "$MODEL_PATH" "$PROMPT_TOKENS" "$GEN_TOKENS" "$THREADS" 1 --batch-size "$BATCH_SIZE" --csv 2>&1 | \
            grep -E '^[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?$' | tail -1 || true)

        if [[ "$result" == ERROR:* ]]; then
            log "${RED}  Error: ${result#ERROR:}${NC}"
            return 1
        fi
        if [[ -z "$result" ]]; then
            log "${RED}  Error: failed to parse bench_native CSV output${NC}"
            return 1
        fi

        run_results+=("$result")
    done

    # Average results
    local sum_ttft=0 sum_tps=0 sum_itl_avg=0 sum_itl_p50=0 sum_itl_p90=0 sum_itl_p99=0 total_tokens=0
    for r in "${run_results[@]}"; do
        IFS=',' read -r ttft tps tokens itl_avg itl_p50 itl_p90 itl_p99 <<< "$r"
        sum_ttft=$(echo "$sum_ttft + $ttft" | bc)
        sum_tps=$(echo "$sum_tps + $tps" | bc)
        sum_itl_avg=$(echo "$sum_itl_avg + $itl_avg" | bc)
        sum_itl_p50=$(echo "$sum_itl_p50 + $itl_p50" | bc)
        sum_itl_p90=$(echo "$sum_itl_p90 + $itl_p90" | bc)
        sum_itl_p99=$(echo "$sum_itl_p99 + $itl_p99" | bc)
        total_tokens=$tokens
    done

    DC_TTFT=$(echo "scale=2; $sum_ttft / $RUNS" | bc)
    DC_TPS=$(echo "scale=2; $sum_tps / $RUNS" | bc)
    DC_TOKENS=$total_tokens
    DC_ITL_AVG=$(echo "scale=2; $sum_itl_avg / $RUNS" | bc)
    DC_ITL_P50=$(echo "scale=2; $sum_itl_p50 / $RUNS" | bc)
    DC_ITL_P90=$(echo "scale=2; $sum_itl_p90 / $RUNS" | bc)
    DC_ITL_P99=$(echo "scale=2; $sum_itl_p99 / $RUNS" | bc)
}

# =============================================================================
# llama.cpp Benchmark
# =============================================================================
bench_llamacpp() {
    if ! $COMPARE_LLAMACPP; then
        return
    fi

    log "${BOLD}Running llama.cpp benchmark...${NC}"

    # Build llama.cpp if needed
    if [[ ! -f "$LLAMACPP_DIR/build/bin/llama-bench" ]] && [[ ! -f "$LLAMACPP_DIR/llama-bench" ]]; then
        if [[ ! -d "$LLAMACPP_DIR" ]]; then
            log "  Cloning llama.cpp..."
            git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR" 2>&1 | tail -1
        fi
        log "  Building llama.cpp..."
        (cd "$LLAMACPP_DIR" && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-bench -j"$(nproc)") 2>&1 | tail -3
    fi

    local bench_bin=""
    for candidate in "$LLAMACPP_DIR/build/bin/llama-bench" "$LLAMACPP_DIR/llama-bench"; do
        if [[ -f "$candidate" ]]; then
            bench_bin="$candidate"
            break
        fi
    done

    if [[ -z "$bench_bin" ]]; then
        log "${RED}  llama-bench not found${NC}"
        return
    fi

    LC_MODE="single_process"
    LC_THREADS_PER_WORKER="$THREADS"

    if [[ "$BATCH_SIZE" -gt 1 ]]; then
        LC_MODE="single_stream_baseline"
    fi

    local result
    result=$("$bench_bin" \
        -m "$MODEL_PATH" \
        --n-prompt "$PROMPT_TOKENS" \
        --n-gen "$GEN_TOKENS" \
        -t "$THREADS" \
        -r "$RUNS" \
        -o csv 2>/dev/null | tail -1)

    if [[ -n "$result" ]]; then
        LC_PP=$("$bench_bin" \
            -m "$MODEL_PATH" \
            --n-prompt "$PROMPT_TOKENS" \
            --n-gen 0 \
            -t "$THREADS" \
            -r "$RUNS" \
            -o csv 2>/dev/null | tail -1 | awk -F',' '{print $(NF-1)}' | tr -d '"')

        LC_TG=$("$bench_bin" \
            -m "$MODEL_PATH" \
            --n-prompt 1 \
            --n-gen "$GEN_TOKENS" \
            -t "$THREADS" \
            -r "$RUNS" \
            -o csv 2>/dev/null | tail -1 | awk -F',' '{print $(NF-1)}' | tr -d '"')
    fi

    LC_PP="${LC_PP:-0}"
    LC_TG="${LC_TG:-0}"
}

# =============================================================================
# Output
# =============================================================================
output_results() {
    local model_name
    model_name=$(basename "$MODEL_PATH")
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if $JSON_ONLY; then
        cat <<ENDJSON
{
  "timestamp": "$timestamp",
  "hardware": {
    "cpu": "$HW_CPU",
    "arch": "$HW_ARCH",
    "cores": $HW_CORES,
    "simd": "$HW_SIMD",
    "memory_gb": $HW_MEM_GB
  },
    "config": {
      "model": "$model_name",
      "model_path": "$MODEL_PATH",
      "prompt_tokens": $PROMPT_TOKENS,
      "gen_tokens": $GEN_TOKENS,
      "batch_size": $BATCH_SIZE,
      "threads": $THREADS,
      "runs": $RUNS,
      "verify_paged_parity": $VERIFY_PAGED_PARITY
  },
  "densecore": {
    "ttft_ms": $DC_TTFT,
    "tokens_per_second": $DC_TPS,
    "tokens_generated": $DC_TOKENS,
    "itl_avg_ms": $DC_ITL_AVG,
    "itl_p50_ms": $DC_ITL_P50,
    "itl_p90_ms": $DC_ITL_P90,
    "itl_p99_ms": $DC_ITL_P99,
    "paged_decode_parity": "$PARITY_STATUS"
  }$(if $COMPARE_LLAMACPP; then cat <<ENDLC
,
  "llamacpp": {
    "prompt_processing_tps": ${LC_PP:-0},
    "text_generation_tps": ${LC_TG:-0},
    "mode": "${LC_MODE:-single_process}",
    "threads_per_worker": ${LC_THREADS_PER_WORKER:-$THREADS}
  }
ENDLC
fi)
}
ENDJSON
        return
    fi

    echo ""
    echo -e "${BOLD}================================================================${NC}"
    echo -e "${BOLD}  DenseCore Benchmark Results${NC}"
    echo -e "${BOLD}================================================================${NC}"
    echo ""
    echo -e "${CYAN}Hardware${NC}"
    echo "  CPU:    $HW_CPU"
    echo "  Arch:   $HW_ARCH ($HW_SIMD)"
    echo "  Cores:  $HW_CORES"
    echo "  Memory: ${HW_MEM_GB}GB"
    echo ""
    echo -e "${CYAN}Config${NC}"
    echo "  Model:         $model_name"
    echo "  Prompt tokens: $PROMPT_TOKENS"
    echo "  Gen tokens:    $GEN_TOKENS"
    echo "  Batch size:    $BATCH_SIZE"
    echo "  Threads:       $THREADS"
    echo "  Runs:          $RUNS"
    echo "  Parity check:  $VERIFY_PAGED_PARITY"
    echo ""
    echo -e "${CYAN}DenseCore Results${NC}"
    echo "  +--------------------------+-----------+"
    echo "  | Metric                   | Value     |"
    echo "  +--------------------------+-----------+"
    printf "  | %-24s | %7s ms |\n" "Time To First Token" "$DC_TTFT"
    printf "  | %-24s | %6s t/s |\n" "Generation Speed" "$DC_TPS"
    printf "  | %-24s | %7s ms |\n" "Inter-Token Latency avg" "$DC_ITL_AVG"
    printf "  | %-24s | %7s ms |\n" "Inter-Token Latency p50" "$DC_ITL_P50"
    printf "  | %-24s | %7s ms |\n" "Inter-Token Latency p90" "$DC_ITL_P90"
    printf "  | %-24s | %7s ms |\n" "Inter-Token Latency p99" "$DC_ITL_P99"
    printf "  | %-24s | %7s    |\n" "Paged Decode Parity" "$PARITY_STATUS"
    echo "  +--------------------------+-----------+"

    if $COMPARE_LLAMACPP && [[ "${LC_TG:-0}" != "0" ]]; then
        echo ""
        echo -e "${CYAN}Comparison: DenseCore vs llama.cpp${NC}"
        echo "  +------------------+------------+------------+"
        echo "  | Metric           | DenseCore  | llama.cpp  |"
        echo "  +------------------+------------+------------+"
        printf "  | %-16s | %8s   | %8s   |\n" "Gen Speed (t/s)" "$DC_TPS" "$LC_TG"
        printf "  | %-16s | %8s   | %8s   |\n" "Prefill (t/s)" "-" "$LC_PP"
        echo "  +------------------+------------+------------+"

        if [[ "$DC_TPS" != "0" ]] && [[ "$LC_TG" != "0" ]]; then
            local ratio
            ratio=$(echo "scale=1; $DC_TPS * 100 / $LC_TG" | bc 2>/dev/null || echo "?")
            echo ""
            echo -e "  DenseCore is ${BOLD}${ratio}%${NC} of llama.cpp generation speed"
        fi
    fi

    echo ""
    echo -e "${GREEN}Tip:${NC} Run with --json to get machine-readable output for CI."
    echo -e "${GREEN}Tip:${NC} Run with --compare-llamacpp to include llama.cpp comparison."
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Initialize comparison variables
    LC_PP=0
    LC_TG=0
    PARITY_STATUS="not_run"

    detect_hardware
    find_model

    log ""
    log "${BOLD}DenseCore Benchmark Suite${NC}"
    log "  Model:   $(basename "$MODEL_PATH")"
    log "  CPU:     $HW_CPU ($HW_SIMD)"
    log "  Threads: $THREADS"
    log ""

    verify_paged_parity
    bench_densecore
    bench_llamacpp
    output_results
}

main
