# =============================================================================
# DenseCore Model Downloader Image
# =============================================================================
# Lightweight container for downloading models from Hugging Face Hub.
# Pre-bakes huggingface_hub to eliminate cold-start pip install latency.
#
# Build: docker build -f docker/downloader/Dockerfile -t densecore-downloader:local docker/downloader
# Size: ~150MB (python:3.11-slim base + huggingface_hub)
# =============================================================================

FROM python:3.11-slim AS builder

# Install huggingface_hub with minimal dependencies
# --no-cache-dir: Reduce image size by not caching pip packages
# --no-compile: Skip .pyc generation during install (will be done at runtime if needed)
RUN pip install --no-cache-dir --no-compile huggingface_hub

# =============================================================================
# Final Stage: Minimal runtime image
# =============================================================================
FROM python:3.11-slim

# Labels for image metadata
LABEL org.opencontainers.image.title="DenseCore Model Downloader"
LABEL org.opencontainers.image.description="Downloads GGUF models from Hugging Face Hub for DenseCore inference"
LABEL org.opencontainers.image.vendor="DenseCore"
LABEL org.opencontainers.image.version="0.1.0"

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash downloader

# Copy the download script
COPY download_model.py /app/download_model.py
RUN chmod +x /app/download_model.py

# Set working directory
WORKDIR /app

# Switch to non-root user
USER downloader

# Environment variable defaults (can be overridden at runtime)
ENV MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
ENV MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
ENV HF_TOKEN=""

# Health check - verify Python and huggingface_hub are available
HEALTHCHECK --interval=30s --timeout=5s --retries=1 \
    CMD python3 -c "import huggingface_hub; print('OK')" || exit 1

# Entrypoint
ENTRYPOINT ["python3", "/app/download_model.py"]
