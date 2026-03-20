#!/bin/bash
# =============================================================================
# AWS Graviton4 & Xeon Benchmark Script for DenseCore
# =============================================================================
# Usage:
#   ./aws_benchmark.sh graviton4  # Run on m7g.16xlarge
#   ./aws_benchmark.sh xeon       # Run on m7i.16xlarge
# =============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect hardware
detect_hardware() {
    echo -e "${YELLOW}[Detecting Hardware]${NC}"
    
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
    CPU_ARCH=$(uname -m)
    CPU_CORES=$(nproc)
    
    echo "  CPU: $CPU_MODEL"
    echo "  Architecture: $CPU_ARCH"
    echo "  Cores: $CPU_CORES"
    
    # Detect SIMD capabilities
    if [[ "$CPU_ARCH" == "aarch64" ]]; then
        if grep -q sve2 /proc/cpuinfo; then
            SIMD_LEVEL="SVE2 (Graviton4)"
        elif grep -q sve /proc/cpuinfo; then
            SIMD_LEVEL="SVE (Graviton3)"
        else
            SIMD_LEVEL="NEON"
        fi
    else
        if grep -q avx512 /proc/cpuinfo; then
            SIMD_LEVEL="AVX-512"
        elif grep -q avx2 /proc/cpuinfo; then
            SIMD_LEVEL="AVX2"
        else
            SIMD_LEVEL="SSE4.1"
        fi
    fi
    
    echo "  SIMD: $SIMD_LEVEL"
    echo ""
}

# Install dependencies
install_deps() {
    echo -e "${YELLOW}[Installing Dependencies]${NC}"
    
    # Update system
    sudo apt-get update -qq
    
    # Install build tools
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        python3-pip \
        python3-venv \
        libnuma-dev \
        wget \
        curl
    
    # Install oneDNN for Intel AMX/VNNI
    if [[ "$CPU_ARCH" != "aarch64" ]]; then
        echo "  Installing oneDNN for x86..."
        sudo apt-get install -y libdnnl-dev || true
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Build DenseCore
build_densecore() {
    echo -e "${YELLOW}[Building DenseCore]${NC}"
    
    cd "$(git rev-parse --show-toplevel)"
    
    # Clean previous build
    rm -rf core/build
    mkdir -p core/build
    cd core/build
    
    # Configure with optimal flags
    if [[ "$CPU_ARCH" == "aarch64" ]]; then
        # ARM: Enable SVE2 if available
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_FLAGS="-march=armv8.2-a+sve2+bf16 -O3" \
            -DCMAKE_CXX_FLAGS="-march=armv8.2-a+sve2+bf16 -O3" \
            -DGGML_NATIVE=ON
    else
        # x86: Enable AVX-512, AMX, VNNI
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_FLAGS="-march=native -O3" \
            -DCMAKE_CXX_FLAGS="-march=native -O3" \
            -DGGML_NATIVE=ON \
            -DDENSECORE_USE_ONEDNN=ON
    fi
    
    # Build with all cores
    make -j$(nproc)
    
    # Install Python bindings
    cd ../..
    pip3 install -e python/
    
    echo -e "${GREEN}✓ DenseCore built successfully${NC}"
}

# Download benchmark models
download_models() {
    echo -e "${YELLOW}[Downloading Models]${NC}"
    
    pip3 install -q huggingface-hub
    
    # Create cache directory
    CACHE_DIR="$HOME/.cache/huggingface/hub"
    mkdir -p "$CACHE_DIR"
    
    # Download models (GGUF format)
    python3 - <<EOF
from huggingface_hub import hf_hub_download

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Llama-3.1-8B-Instruct-GGUF", "Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
]

print("Downloading benchmark models...")
for repo_id, filename in models:
    try:
        print(f"  → {repo_id}/{filename}")
        hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"    Warning: {e}")
EOF
    
    echo -e "${GREEN}✓ Models downloaded${NC}"
}

# Run benchmarks
run_benchmarks() {
    echo -e "${YELLOW}[Running Benchmarks]${NC}"
    
    cd "$(git rev-parse --show-toplevel)"
    
    # Output file
    RESULT_FILE="benchmark_results_$(hostname)_$(date +%Y%m%d_%H%M%S).json"
    
    python3 benchmarks/benchmark_throughput.py \
        --scan \
        --threads $(nproc) \
        --n-tokens 128 \
        > "$RESULT_FILE"
    
    echo -e "${GREEN}✓ Benchmarks complete${NC}"
    echo "  Results: $RESULT_FILE"
    
    # Pretty print results
    echo ""
    echo -e "${YELLOW}[Results Summary]${NC}"
    cat "$RESULT_FILE" | grep -E "Model|TTFT|Speed|tok/s" || true
}

# Compare with llama.cpp
compare_llamacpp() {
    echo -e "${YELLOW}[Comparing with llama.cpp]${NC}"
    
    # Install llama.cpp
    if [ ! -d "$HOME/llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git "$HOME/llama.cpp"
        cd "$HOME/llama.cpp"
        make -j$(nproc)
    else
        cd "$HOME/llama.cpp"
        git pull
        make -j$(nproc)
    fi
    
    # Find a test model
    TEST_MODEL=$(find "$HOME/.cache/huggingface/hub" -name "*3B*.gguf" | head -n1)
    
    if [ -z "$TEST_MODEL" ]; then
        echo "  No model found for comparison"
        return
    fi
    
    echo "  Testing with: $(basename $TEST_MODEL)"
    
    # Run llama.cpp benchmark
    ./llama-bench \
        -m "$TEST_MODEL" \
        -p 128 \
        -n 128 \
        -t $(nproc) \
        > /tmp/llamacpp_bench.txt
    
    LLAMACPP_SPEED=$(grep "decode time" /tmp/llamacpp_bench.txt | awk '{print $(NF-1)}')
    
    echo -e "${GREEN}  llama.cpp: $LLAMACPP_SPEED tok/s${NC}"
    echo "  (DenseCore results in benchmark_results_*.json)"
}

# Generate markdown report
generate_report() {
    echo -e "${YELLOW}[Generating Benchmark Report]${NC}"
    
    cat > "BENCHMARK_REPORT_$(hostname).md" <<EOF
# DenseCore Benchmark Report

**Hardware**: $CPU_MODEL  
**Architecture**: $CPU_ARCH  
**Cores**: $CPU_CORES  
**SIMD**: $SIMD_LEVEL  
**Date**: $(date)

## Results

\`\`\`json
$(cat benchmark_results_*.json 2>/dev/null || echo "No results found")
\`\`\`

## Comparison

| Engine | Model | Speed (tok/s) |
|--------|-------|---------------|
| **DenseCore** | Llama-3.2-3B | TBD |
| llama.cpp | Llama-3.2-3B | TBD |

## System Info

\`\`\`bash
$(lscpu)
\`\`\`

---
*Generated by DenseCore Benchmark Suite*
EOF
    
    echo -e "${GREEN}✓ Report saved to BENCHMARK_REPORT_$(hostname).md${NC}"
}

# Main workflow
main() {
    echo "=============================================="
    echo "  DenseCore AWS Benchmark Suite"
    echo "=============================================="
    echo ""
    
    detect_hardware
    
    read -p "Install dependencies? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_deps
    fi
    
    read -p "Build DenseCore? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_densecore
    fi
    
    read -p "Download models? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    fi
    
    run_benchmarks
    
    read -p "Compare with llama.cpp? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        compare_llamacpp
    fi
    
    generate_report
    
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  Benchmark Complete!"
    echo "==============================================${NC}"
}

# Run with command-line args
if [[ $# -gt 0 ]]; then
    case "$1" in
        graviton4)
            echo "Optimizing for AWS Graviton4 (SVE2)..."
            ;;
        xeon)
            echo "Optimizing for Intel Xeon (AVX-512/AMX)..."
            ;;
        *)
            echo "Usage: $0 [graviton4|xeon]"
            exit 1
            ;;
    esac
fi

main
