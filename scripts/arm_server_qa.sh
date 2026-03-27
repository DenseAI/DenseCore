#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: arm_server_qa.sh <model_path> [port]}"
PORT="${2:-18081}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/core/build-arm"

export LD_LIBRARY_PATH="${BUILD_DIR}:${BUILD_DIR}/third_party/ggml/src${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SERVER_BIN="${SERVER_BIN:-/tmp/densecore-arm-server}"
SERVER_LOG="${SERVER_LOG:-/tmp/densecore-arm-server.log}"
QA_MAX_TOKENS_KR="${ARM_QA_MAX_TOKENS_KR:-32}"
QA_MAX_TOKENS_FR="${ARM_QA_MAX_TOKENS_FR:-24}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

extract_chat_content() {
    python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

wait_for_http() {
    local url="$1"
    local attempts="${2:-30}"
    local delay="${3:-1}"
    local i
    for ((i = 0; i < attempts; ++i)); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep "${delay}"
    done
    return 1
}

CHAT_PAYLOAD_KR="{\"messages\":[{\"role\":\"user\",\"content\":\"한국의 수도는?\"}],\"max_tokens\":${QA_MAX_TOKENS_KR},\"temperature\":0,\"thinking\":false}"
CHAT_PAYLOAD_FR="{\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":${QA_MAX_TOKENS_FR},\"temperature\":0,\"thinking\":false}"

pkill -f "${SERVER_BIN}" >/dev/null 2>&1 || true
env PORT="${PORT}" "${SERVER_BIN}" serve --grpc=false >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

wait_for_http "http://127.0.0.1:${PORT}/health/live" 30 1

LIVE="$(curl -fsS "http://127.0.0.1:${PORT}/health/live")"
LOAD="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/models/load" \
    -H 'Content-Type: application/json' \
    -d "{\"model_path\":\"${MODEL_PATH}\"}")"

wait_for_http "http://127.0.0.1:${PORT}/health/startup" 30 1
STARTUP="$(curl -fsS "http://127.0.0.1:${PORT}/health/startup")"

CHAT_ONE_RAW="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${CHAT_PAYLOAD_KR}")"
CHAT_TWO_RAW="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${CHAT_PAYLOAD_FR}")"
CHAT_THREE_RAW="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${CHAT_PAYLOAD_FR}")"

CHAT_ONE="$(printf '%s' "${CHAT_ONE_RAW}" | extract_chat_content)"
CHAT_TWO="$(printf '%s' "${CHAT_TWO_RAW}" | extract_chat_content)"
CHAT_THREE="$(printf '%s' "${CHAT_THREE_RAW}" | extract_chat_content)"

printf 'ARM_LIVE=%s\n' "${LIVE}"
printf 'ARM_LOAD=%s\n' "${LOAD}"
printf 'ARM_STARTUP=%s\n' "${STARTUP}"
printf 'ARM_CHAT_KR=%s\n' "${CHAT_ONE}"
printf 'ARM_CHAT_FR_ONE=%s\n' "${CHAT_TWO}"
printf 'ARM_CHAT_FR_TWO=%s\n' "${CHAT_THREE}"
printf 'ARM_GREEDY_EQUAL=%s\n' "$([[ "${CHAT_TWO}" == "${CHAT_THREE}" ]] && echo true || echo false)"

if ! grep -qi '서울' <<<"${CHAT_ONE}"; then
    echo "ARM_QA_FAIL=missing_seoul" >&2
    exit 1
fi

if ! grep -qi 'paris' <<<"${CHAT_TWO}"; then
    echo "ARM_QA_FAIL=missing_paris_first" >&2
    exit 1
fi

if ! grep -qi 'paris' <<<"${CHAT_THREE}"; then
    echo "ARM_QA_FAIL=missing_paris_second" >&2
    exit 1
fi

if [[ "${CHAT_TWO}" != "${CHAT_THREE}" ]]; then
    echo "ARM_QA_FAIL=greedy_mismatch" >&2
    exit 1
fi
