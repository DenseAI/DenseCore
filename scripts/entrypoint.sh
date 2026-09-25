#!/bin/bash
# =============================================================================
# DenseCore Entrypoint Script
# =============================================================================
# Detects Cgroup CPU limit and configures OMP thread count for optimal
# performance, reserving 1 core for the Go runtime.
# =============================================================================
set -e

# -----------------------------------------------------------------------------
# detect_cpu_limit: Read CPU quota from Cgroup v2 or v1
# Returns the number of CPUs allocated (or nproc as fallback)
# -----------------------------------------------------------------------------
detect_cpu_limit() {
    local limit

    # Cgroup v2 (modern kernels, default in K8s 1.25+)
    if [[ -f /sys/fs/cgroup/cpu.max ]]; then
        local quota period
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [[ "$quota" != "max" && "$period" -gt 0 ]]; then
            limit=$((quota / period))
        fi
    # Cgroup v1 fallback
    elif [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
        local quota period
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [[ "$quota" -gt 0 && "$period" -gt 0 ]]; then
            limit=$((quota / period))
        fi
    fi

    # Clamp to minimum of 1, default to nproc if no limit detected
    if [[ -z "$limit" || "$limit" -lt 1 ]]; then
        limit=$(nproc)
    fi

    echo "$limit"
}

# -----------------------------------------------------------------------------
# Main Entrypoint Logic
# -----------------------------------------------------------------------------
CPU_LIMIT=$(detect_cpu_limit)

# Reserve 1 core for Go runtime scheduler, minimum 1 OMP thread
if [[ "$CPU_LIMIT" -gt 1 ]]; then
    OMP_THREADS=$((CPU_LIMIT - 1))
else
    OMP_THREADS=1
fi

# Allow environment override, otherwise use computed value
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_THREADS}"
export GOMAXPROCS="${GOMAXPROCS:-$CPU_LIMIT}"

# OpenMP thread binding for NUMA awareness (optional, can be overridden)
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "[entrypoint] CPU limit detected: ${CPU_LIMIT} cores"
echo "[entrypoint] OMP_NUM_THREADS=${OMP_NUM_THREADS}, GOMAXPROCS=${GOMAXPROCS}"
echo "[entrypoint] OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}"

# -----------------------------------------------------------------------------
# Model resolution: local path or Hugging Face repo
# -----------------------------------------------------------------------------
MODELS_DIR="/models"
if ! mkdir -p "${MODELS_DIR}" 2>/dev/null; then
    MODELS_DIR="/app/models"
    mkdir -p "${MODELS_DIR}"
fi

resolve_hf_model() {
    local repo="$1"
    local file="$2"
    local dest="${MODELS_DIR}/${file}"
    local tmp="${dest}.download"

    if [[ -f "${dest}" ]]; then
        echo "[entrypoint] Model already present: ${dest}" >&2
    else
        echo "[entrypoint] Downloading Hugging Face model: ${repo}/${file}" >&2
        if [[ -n "${HF_TOKEN:-}" ]]; then
            curl -fL -H "Authorization: Bearer ${HF_TOKEN}" \
                -o "${tmp}" "https://huggingface.co/${repo}/resolve/main/${file}"
        else
            curl -fL -o "${tmp}" "https://huggingface.co/${repo}/resolve/main/${file}"
        fi
        mv "${tmp}" "${dest}"
    fi

    ln -sf "${dest}" "${MODELS_DIR}/main_model.gguf"
    echo "${MODELS_DIR}/main_model.gguf"
}

# Build command arguments
CMD_ARGS=("serve")

# Add model path if specified via environment variable
if [[ -n "${MODEL:-}" ]]; then
    echo "[entrypoint] MODEL=${MODEL}"
    if [[ -f "${MODEL}" ]]; then
        CMD_ARGS+=("--model" "${MODEL}")
    else
        HF_REPO=""
        HF_FILE=""
        if [[ "${MODEL}" == *:* ]]; then
            HF_REPO="${MODEL%%:*}"
            HF_FILE="${MODEL#*:}"
        elif [[ "${MODEL}" == */* ]]; then
            HF_REPO="${MODEL%/*}"
            HF_FILE="${MODEL##*/}"
        fi

        if [[ -n "${HF_REPO}" && -n "${HF_FILE}" ]]; then
            MODEL_PATH="$(resolve_hf_model "${HF_REPO}" "${HF_FILE}")"
            CMD_ARGS+=("--model" "${MODEL_PATH}")
        else
            echo "[entrypoint] ERROR: MODEL is not a file path or repo spec (repo/file or repo:file)"
            exit 1
        fi
    fi
elif [[ -n "${MODEL_REPO:-}" && -n "${MODEL_FILE:-}" ]]; then
    MODEL_PATH="$(resolve_hf_model "${MODEL_REPO}" "${MODEL_FILE}")"
    CMD_ARGS+=("--model" "${MODEL_PATH}")
fi

# Add any additional arguments passed to the container
CMD_ARGS+=("$@")

# Execute the main binary
exec /app/densecore-server "${CMD_ARGS[@]}"
