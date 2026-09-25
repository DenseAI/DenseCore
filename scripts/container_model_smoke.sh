#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: container_model_smoke.sh IMAGE MODEL [PORT]}
model=${2:?usage: container_model_smoke.sh IMAGE MODEL [PORT]}
port=${3:-${DENSECORE_SMOKE_PORT:-18080}}
container_name="densecore-model-smoke-$$"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/densecore-container-smoke.XXXXXX")

cleanup() {
  local status=$?
  if [ -n "$container_name" ]; then
    docker logs "$container_name" >"$tmp_dir/container.log" 2>&1 || true
    docker rm -f "$container_name" >/dev/null 2>&1 || true
  fi
  if [ "$status" -ne 0 ] || [ "${DENSECORE_SMOKE_KEEP_ARTIFACTS:-0}" = "1" ]; then
    echo "container smoke artifacts: $tmp_dir" >&2
  else
    rm -rf "$tmp_dir"
  fi
  return "$status"
}
trap cleanup EXIT

[ -f "$model" ] || { echo "smoke model does not exist: $model" >&2; exit 1; }

docker run --detach \
  --name "$container_name" \
  --publish "127.0.0.1:$port:8080" \
  --mount "type=bind,src=$(realpath "$model"),dst=/models/release-smoke.gguf,readonly" \
  --env MAIN_MODEL_PATH=/models/release-smoke.gguf \
  "$image" >/dev/null

wait_for_http() {
  local url=$1
  local i
  for i in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != "true" ]; then
      break
    fi
    sleep 1
  done
  docker logs "$container_name" >&2 || true
  echo "timed out waiting for $url" >&2
  return 1
}

base_url="http://127.0.0.1:$port"
wait_for_http "$base_url/health/live"
wait_for_http "$base_url/health/startup"
wait_for_http "$base_url/health/ready"
curl -fsS "$base_url/v1/models" >"$tmp_dir/models.json"

curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"densecore-v1","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":16,"stream":false}' \
  "$base_url/v1/chat/completions" >"$tmp_dir/nonstream.json"

curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"densecore-v1","messages":[{"role":"user","content":"Count to two."}],"max_tokens":16,"stream":true}' \
  "$base_url/v1/chat/completions" >"$tmp_dir/stream.sse"

python3 - "$tmp_dir/models.json" "$tmp_dir/nonstream.json" "$tmp_dir/stream.sse" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    models = json.load(handle)
if not models.get("data"):
    raise SystemExit("models response is empty")

with open(sys.argv[2], encoding="utf-8") as handle:
    response = json.load(handle)
choices = response.get("choices")
content = choices[0].get("message", {}).get("content") if choices else None
if not isinstance(content, str) or not content.strip():
    raise SystemExit("non-streaming response content is empty or invalid")

terminal = False
visible = []
with open(sys.argv[3], encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            terminal = True
            continue
        event = json.loads(payload)
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            visible.append(delta.get("content") or delta.get("reasoning_content") or "")
if not terminal:
    raise SystemExit("streaming response did not reach [DONE]")
if not "".join(visible).strip():
    raise SystemExit("streaming response content is empty")
PY

docker stop --time 30 "$container_name" >/dev/null
exit_code=$(docker wait "$container_name")
if [ "$exit_code" != "0" ]; then
  docker logs "$container_name" >&2 || true
  echo "container exited with status $exit_code after SIGTERM" >&2
  exit 1
fi

docker logs "$container_name" >"$tmp_dir/container.log" 2>&1
docker rm "$container_name" >/dev/null
container_name=""
echo "real-model container smoke passed: $image"
