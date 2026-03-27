# ============================================
# DenseCore Production Dockerfile
# Multi-stage build with optimized layer caching
# ============================================
#
# Cache Optimization Strategy:
# 1. Copy dependency files (ggml, CMake) first
# 2. Build ggml (slow, but rarely changes)
# 3. Copy application source
# 4. Build DenseCore (fast, changes frequently)
#
# This ensures changing a .cpp file doesn't rebuild ggml

# Debian version pinned for reproducibility (glibc-based for runtime compatibility)
ARG DEBIAN_VERSION=bookworm

# ============================================
# Stage 1: Dependency Builder (ggml cache layer)
# ============================================
FROM golang:1.24-${DEBIAN_VERSION} AS deps-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY dependency files first (ggml + CMake config)
# This layer will be cached unless these files change
COPY core/third_party/ core/third_party/
COPY core/CMakeLists.txt core/CMakeLists.txt
COPY core/cmake/ core/cmake/
COPY core/include/ core/include/

# Create stub sources to satisfy CMake (will be replaced later)
RUN mkdir -p core/src core/src/quantization core/src/pruning core/src/kernels core/tests && \
    for f in engine worker inference model_types model_loader tokenizer kv_cache \
    optimization_bridge save_model quantizer pruner quantize version tensor_utils \
    scheduler embedding simd_ops cpu_backend cpu_backend_opt backend_registry \
    kernel_context async_cpu_backend; do \
    echo "" > core/src/${f}.cpp; \
    done && \
    for f in cpu_int4 cpu_amx; do \
    echo "" > core/src/kernels/${f}.cpp; \
    done && \
    for f in max_quantizer awq_quantizer smoothquant_quantizer int4_quantizer; do \
    echo "" > core/src/quantization/${f}.cpp; \
    done && \
    for f in depth_pruner width_pruner attention_pruner combined_pruner; do \
    echo "" > core/src/pruning/${f}.cpp; \
    done && \
    for f in test_simd_ops test_memory_pool test_kv_cache test_aligned_allocator \
    test_async_backend test_cpu_int4 test_kernel_context test_tensor_view; do \
    echo "" > core/tests/${f}.cpp; \
    done

# Pre-build ggml (this is the slow part - now cached)
# For multi-arch ARM builds:
# - GGML_NATIVE=OFF: Disable native CPU feature detection
# - DENSEVLA_ARM_TARGET: Select ARM ISA profile (generic/jetson_orin/rpi5/qualcomm_rb5/custom)
ARG TARGETARCH
ARG DENSEVLA_ARM_TARGET=generic
RUN mkdir -p build && cd build && \
    if [ "$TARGETARCH" = "arm64" ]; then \
    EXTRA_FLAGS="-DCMAKE_TOOLCHAIN_FILE=../core/cmake/aarch64-toolchain.cmake -DDENSEVLA_ARM_TARGET=${DENSEVLA_ARM_TARGET}"; \
    else \
    EXTRA_FLAGS=""; \
    fi && \
    cmake ../core -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF $EXTRA_FLAGS && \
    cmake --build . --target ggml -j$(nproc)

# ============================================
# Stage 2: Application Builder
# ============================================
FROM deps-builder AS builder

# Install Go protobuf plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

# Install protobuf
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Go module files for caching
COPY server/go.mod server/go.sum* ./server/
WORKDIR /app/server
RUN go mod download

# Now copy the REAL source files (invalidates from here down on code changes)
WORKDIR /app
COPY core/src/ core/src/
COPY server/ server/
COPY proto/ proto/
COPY Makefile ./

# Generate Proto files (optional, may fail if not needed)
RUN protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/densecore.proto 2>/dev/null || true

# Rebuild only DenseCore (ggml is already built and cached)
RUN cd build && \
    cmake --build . --target densecore -j$(nproc)

# Build Go server CLI (supports "serve" subcommand and --model flag)
WORKDIR /app/server
ENV CGO_LDFLAGS="-L/app/build -ldensecore -lstdc++ -ldl"
ENV CGO_CFLAGS="-I/app/core/include"
RUN CGO_ENABLED=1 GOOS=linux go build -o /densecore-server ./cmd/densecore

# ============================================
# Stage 3: Runtime (Debian glibc for C++ performance)
# ============================================
FROM debian:bookworm-slim

# Install runtime dependencies
# - libgomp1: OpenMP runtime for parallel C++ kernels
# - libstdc++6: C++ standard library (glibc version)
# - tini: proper init for signal handling in containers
# - curl: for health checks (lighter than wget on Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Debian syntax)
RUN groupadd -g 1000 densecore && \
    useradd -u 1000 -g densecore -m -s /sbin/nologin densecore

# Create directories
RUN mkdir -p /app/models /app/lib && \
    chown -R densecore:densecore /app

WORKDIR /app

# Copy entrypoint script for OMP thread tuning
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy binaries and libraries
COPY --from=builder /densecore-server /app/densecore-server
COPY --from=builder /app/build/libdensecore.so* /app/lib/
COPY --from=builder /app/build/libggml*.so* /app/lib/

# Create symlink for versioned library name
RUN cd /app/lib && \
    [ -f libdensecore.so.1 ] || ln -s libdensecore.so libdensecore.so.1

# Set library path
ENV LD_LIBRARY_PATH=/app/lib

# Switch to non-root user
USER densecore

# Default configuration
ENV PORT=8080 \
    HOST=0.0.0.0 \
    READ_TIMEOUT=30s \
    WRITE_TIMEOUT=120s \
    SHUTDOWN_TIMEOUT=30s \
    RATE_LIMIT_ENABLED=true \
    RATE_LIMIT_RPS=100 \
    LOG_FORMAT=json \
    THREADS=0 \
    MODEL=""

# Expose port
EXPOSE 8080

# Health check (using curl instead of wget for Debian)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health/live || exit 1

# Use tini as init system with entrypoint for OMP configuration
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
