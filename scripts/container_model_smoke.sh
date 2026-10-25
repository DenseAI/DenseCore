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
  if [ "$status" -ne 0 ] || [ "${DENSECORE_SMOKE_KEEP_ARTIFACTS:-1}" = "1" ]; then
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
  --cpus "${DENSECORE_SMOKE_CPUS:-2}" --memory "${DENSECORE_SMOKE_MEMORY:-4g}" \
  --publish "127.0.0.1:$port:8080" \
  --mount "type=bind,src=$(realpath "$model"),dst=/models/release-smoke.gguf,readonly" \
  --env MAIN_MODEL_PATH=/models/release-smoke.gguf \
  --env DENSECORE_MAX_SEQ_LEN="${DENSECORE_SMOKE_CONTEXT:-2048}" \
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
docker inspect "$container_name" > "$tmp_dir/positive-container-config.json"
curl -fsS "$base_url/v1/models" >"$tmp_dir/models.json"

curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"densecore-v1","messages":[{"role":"user","content":"What is the capital of France? Answer with the city name only."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"stream":false}' \
  "$base_url/v1/chat/completions" >"$tmp_dir/nonstream.json"

curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"densecore-v1","messages":[{"role":"user","content":"Count from 1 to 2. Reply only with the two digits."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"stream":true}' \
  "$base_url/v1/chat/completions" >"$tmp_dir/stream.sse"

answer_status=pass
if python3 - "$tmp_dir/models.json" "$tmp_dir/nonstream.json" "$tmp_dir/stream.sse" <<'PY'
import json
import re
import sys


def final_text(text):
    if "<think>" in text:
        if "</think>" not in text:
            return ""
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


with open(sys.argv[1], encoding="utf-8") as handle:
    models = json.load(handle)
if not models.get("data"):
    raise SystemExit("models response is empty")

with open(sys.argv[2], encoding="utf-8") as handle:
    response = json.load(handle)
choices = response.get("choices")
content = choices[0].get("message", {}).get("content") if choices else None
if not isinstance(content, str) or not re.fullmatch(r"Paris[.!]?", final_text(content), flags=re.IGNORECASE):
    raise SystemExit("non-streaming response has no correct final answer")
if choices[0].get("finish_reason") != "stop":
    raise SystemExit("non-streaming final answer was truncated or incomplete")

terminal = False
finish_reason = None
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
            visible.append(delta.get("content") or "")
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
if not terminal:
    raise SystemExit("streaming response did not reach [DONE]")
if finish_reason != "stop":
    raise SystemExit("streaming final answer was truncated or incomplete")
if not re.search(r"1\D*2", final_text("".join(visible))):
    raise SystemExit("streaming response has no correct final answer")
PY
then
  :
else
  answer_status=fail
fi

check_active_drain() {
  python3 "$(dirname "$0")/active_drain_smoke.py" --url "$base_url" \
    --output "$tmp_dir/drain.json" --timeout 30 --max-tokens "${DENSECORE_SMOKE_DRAIN_MAX_TOKENS:-32}" -- docker kill --signal TERM "$container_name" || return 1

  # Bound the process exit independently from the request drain timeout.
  for _ in $(seq 1 35); do
    [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" == false ]] && break
    sleep 1
  done
  [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" == false ]] || { echo "process exceeded shutdown deadline" >&2; return 1; }
  exit_code=$(docker wait "$container_name")
  if [ "$exit_code" != "0" ]; then
    docker logs "$container_name" >&2 || true
    echo "container exited with status $exit_code after SIGTERM" >&2
    return 1
  fi

}
drain_status=pass
if check_active_drain; then
  :
else
  drain_status=fail
  # Clean up a failed phase without converting its verdict to a pass.
  if [[ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == true ]]; then
    docker stop --time 30 "$container_name" >/dev/null || true
  fi
fi

docker logs "$container_name" >"$tmp_dir/positive-container.log" 2>&1
docker rm "$container_name" >/dev/null
container_name=""
check_insufficient_memory() {
  # Model weights may fit while the explicit context/concurrency plan cannot.
  # A kernel OOM kill is a failure, not proof of a safe initialization rejection.
  container_name="densecore-memory-reject-$$"
  docker run --detach --name "$container_name" --cpus 1 --memory "${DENSECORE_SMOKE_REJECT_MEMORY:-1g}" \
    --mount "type=bind,src=$(realpath "$model"),dst=/models/release-smoke.gguf,readonly" \
    --env MAIN_MODEL_PATH=/models/release-smoke.gguf \
    --env DENSECORE_MAX_SEQ_LEN=32768 --env DENSECORE_MAX_NUM_SEQS=8 "$image" > /dev/null || return 1
  for _ in $(seq 1 60); do
    [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" == false ]] && break
    sleep 1
  done
  docker logs "$container_name" > "$tmp_dir/memory-reject.log" 2>&1
  docker inspect "$container_name" > "$tmp_dir/negative-container-config.json"
  [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" == false ]] || { echo "insufficient-memory initialization did not reject configuration" >&2; return 1; }
  [[ "$(docker inspect --format '{{.State.OOMKilled}}' "$container_name")" == false ]] || { echo "initialization was OOM-killed instead of safely rejecting" >&2; return 1; }
  [[ "$(docker wait "$container_name")" != 0 ]] || { echo "unsafe memory configuration initialized successfully" >&2; return 1; }
  grep -Eq 'resource plan required_bytes=[0-9]+ available_bytes=[0-9]+' "$tmp_dir/memory-reject.log" || { cat "$tmp_dir/memory-reject.log" >&2; echo "missing explicit memory plan rejection" >&2; return 1; }
  docker rm "$container_name" > /dev/null
  container_name=""

}
negative_status=pass
if check_insufficient_memory; then
  :
else
  negative_status=fail
fi
python3 - "$tmp_dir" "$answer_status" "$drain_status" "$negative_status" <<'PY'
import json
import sys
from pathlib import Path
result = dict(zip(('final_answers', 'active_drain', 'insufficient_memory_rejection'), sys.argv[2:]))
result['initialization'] = 'pass'
result['overall_pass'] = all(value == 'pass' for value in result.values())
Path(sys.argv[1], 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(0 if result['overall_pass'] else 1)
PY
