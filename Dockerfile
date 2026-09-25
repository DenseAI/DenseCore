# ============================================
# DenseCore v0.1 developer-preview Dockerfile
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

# Keep the established build toolchain while shipping a minimal runtime image.
ARG BUILD_DEBIAN_VERSION=bookworm

# ============================================
# Stage 1: Dependency Builder (ggml cache layer)
# ============================================
FROM golang:1.25-${BUILD_DEBIAN_VERSION} AS deps-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    gcc \
    g++ \
    cmake \
    libnuma-dev \
    libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY dependency files first (ggml + CMake config)
# This layer will be cached unless these files change
COPY core/third_party/ core/third_party/
COPY core/CMakeLists.txt core/CMakeLists.txt
COPY core/cmake/ core/cmake/
COPY core/include/ core/include/

# Create stub sources to satisfy CMake (will be replaced later)
# Keep this derived from CMakeLists.txt so new source files do not break the ggml cache layer.
RUN mkdir -p core/src core/tests && \
    { \
        awk '/^set\(SOURCES$/ { in_sources=1; next } \
             in_sources && /^\)/ { in_sources=0; next } \
             /^[[:space:]]*set\(TEST_SOURCES$/ { in_tests=1; next } \
             in_tests && /^[[:space:]]*\)/ { in_tests=0; next } \
             in_sources || in_tests { print $1 }' core/CMakeLists.txt; \
        printf '%s\n' src/tools/quantize.cpp; \
    } | \
    sed -e 's/#.*$//' \
        -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]]*$//' \
        -e 's#^\\${CMAKE_CURRENT_SOURCE_DIR}/##' | \
    grep -E '^(src/|tests/)' | \
    while read -r path; do \
        mkdir -p "core/$(dirname "$path")" && : > "core/$path"; \
    done

# Pre-build ggml (this is the slow part - now cached)
# Published images use deterministic CPU targets. Local source builds keep the
# CMake native default; Docker rejects native/unknown targets for release safety.
ARG DENSECORE_CPU_TARGET=portable
ARG DENSECORE_ARM_TARGET=generic
ARG BUILD_JOBS
ARG TARGETARCH
ARG DENSECORE_PORTABLE=ON
RUN mkdir -p build && cd build && \
    case "${DENSECORE_CPU_TARGET}" in \
        portable) \
            EXTRA_FLAGS="-DDENSECORE_CPU_TARGET=portable -DDENSECORE_PORTABLE=ON -DGGML_NATIVE=OFF" ;; \
        amd64-v3) \
            if [ -n "${TARGETARCH}" ] && [ "${TARGETARCH}" != "amd64" ]; then \
                echo "DENSECORE_CPU_TARGET=amd64-v3 requires TARGETARCH=amd64, got ${TARGETARCH}" >&2; exit 1; \
            fi; \
            EXTRA_FLAGS="-DDENSECORE_CPU_TARGET=amd64-v3 -DGGML_NATIVE=OFF" ;; \
        arm64) \
            if [ -n "${TARGETARCH}" ] && [ "${TARGETARCH}" != "arm64" ]; then \
                echo "DENSECORE_CPU_TARGET=arm64 requires TARGETARCH=arm64, got ${TARGETARCH}" >&2; exit 1; \
            fi; \
            EXTRA_FLAGS="-DDENSECORE_CPU_TARGET=arm64 -DDENSECORE_ARM_TARGET=${DENSECORE_ARM_TARGET} -DDENSECORE_ARM_CORRECTNESS_FIRST=ON -DGGML_NATIVE=OFF" ;; \
        *) \
            echo "DENSECORE_CPU_TARGET must be portable, amd64-v3, or arm64; got ${DENSECORE_CPU_TARGET}" >&2; exit 1 ;; \
    esac && \
    cmake ../core -DCMAKE_BUILD_TYPE=Release $EXTRA_FLAGS && \
    cmake --build . --target ggml -j"${BUILD_JOBS:-$(nproc)}"

# ============================================
# Stage 2: Native Runtime Builder
# ============================================
FROM deps-builder AS native-builder
ARG BUILD_JOBS

# Keep the native runtime independent of Go module and server changes. The
# dependency stage configured these source paths using stubs; replace them before
# building DenseCore, while retaining the already-built ggml dependency.
COPY core/src/ core/src/
RUN cd build && \
    cmake --build . --target densecore -j"${BUILD_JOBS:-$(nproc)}"

# ============================================
# Stage 3: Application Builder
# ============================================
FROM native-builder AS builder

# Install Go protobuf plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

# Install protobuf
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Go module metadata and release attribution assets
COPY server/go.mod server/go.sum ./server/
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY scripts/collect_release_licenses.sh /app/scripts/collect_release_licenses.sh

RUN chmod +x /app/scripts/collect_release_licenses.sh && \
    cd /app/server && go mod download

# Server and protocol edits do not invalidate the native runtime build.
WORKDIR /app
COPY server/ server/
COPY proto/ proto/
COPY Makefile ./

# Generate Proto files
RUN protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/densecore.proto

# Build Go server CLI (supports "serve" subcommand and --model flag)
WORKDIR /app/server
ENV CGO_LDFLAGS="-L/app/build -ldensecore -lstdc++ -ldl"
ENV CGO_CFLAGS="-I/app/core/include"
RUN CGO_ENABLED=1 GOOS=linux go build -mod=mod -o /densecore-server ./cmd/densecore

WORKDIR /app
RUN /app/scripts/collect_release_licenses.sh /tmp/densecore-licenses
RUN cd /app/build && \
    [ -e libdensecore.so.1 ] || ln -s libdensecore.so libdensecore.so.1

# ============================================
# Stage 4: Runtime dependency closure
# ============================================
# Collect current Debian 13 runtime libraries separately from the pinned build
# toolchain. The final image copies only this dependency closure.
FROM debian:trixie-slim@sha256:3a39a0592364683e6bab97937b72cad5a8fa6dcbbee90edb3bb48c7f8e94f258 AS runtime-libs
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        libnuma1 \
        libhwloc15 && \
    rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 5: Minimal runtime (Debian 13 glibc, no package manager or OpenSSL)
# ============================================
FROM gcr.io/distroless/base-nossl-debian13:nonroot@sha256:86554c46a420d507ff2d678fd261ab8691fba4875a20302f38a49e684b42a33f

ARG VERSION=0.1.0
ARG COMMIT_SHA=unknown
ARG DENSECORE_CPU_TARGET=portable

LABEL org.opencontainers.image.title="DenseCore" \
      org.opencontainers.image.description="CPU-first memory-centric inference runtime" \
      org.opencontainers.image.vendor="DenseAI" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.densecore.cpu-target="${DENSECORE_CPU_TARGET}"

WORKDIR /app

# Copy binaries and libraries
COPY --from=builder /densecore-server /app/densecore-server
COPY --from=builder /app/build/libdensecore.so* /app/lib/
COPY --from=builder /app/build/libggml*.so* /app/lib/
COPY --from=builder /tmp/densecore-licenses /app/licenses

# DenseCore is release-qualified on linux/amd64. Copy only the dynamic runtime
# libraries required by the portable C++ and NUMA-enabled build.
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /app/lib/
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libgomp.so.1* /app/lib/
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libhwloc.so.15* /app/lib/
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libnuma.so.1* /app/lib/
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libudev.so.1* /app/lib/
COPY --from=runtime-libs /usr/lib/x86_64-linux-gnu/libcap.so.2* /app/lib/
COPY --from=runtime-libs /lib/x86_64-linux-gnu/libgcc_s.so.1 /app/lib/

# Set library path
ENV LD_LIBRARY_PATH=/app/lib

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

# Health check uses the server binary so the runtime image needs no shell or HTTP client.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/densecore-server", "healthcheck", "--url", "http://127.0.0.1:8080/health/live", "--timeout", "5s"]

# The Go runtime installs signal handlers as PID 1 and the server owns graceful shutdown.
ENTRYPOINT ["/app/densecore-server", "serve"]
