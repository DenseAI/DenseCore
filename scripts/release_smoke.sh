#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
version=${DENSECORE_RELEASE_VERSION:-0.1.0}
port=${DENSECORE_SMOKE_PORT:-18080}
build_jobs=${DENSECORE_BUILD_JOBS:-$(nproc)}
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "DENSECORE_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/densecore-clean-smoke.XXXXXX")
server_pid=""

cleanup() {
  local status=$?
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ] || [ "${DENSECORE_SMOKE_KEEP_ARTIFACTS:-0}" = "1" ]; then
    echo "release smoke artifacts: $tmp_dir" >&2
  else
    rm -rf "$tmp_dir"
  fi
  return "$status"
}
trap cleanup EXIT

wait_for_http() {
  local url=$1
  local attempts=${2:-60}
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

post_json() {
  local url=$1
  local data=$2
  curl -fsS -H 'Content-Type: application/json' -d "$data" "$url"
}

source_tar=${DENSECORE_SOURCE_TARBALL:-}
if [ -z "$source_tar" ]; then
  bash "$root_dir/scripts/package_release.sh" "$version" --source-only --out "$tmp_dir"
  source_tar="$tmp_dir/densecore-${version}-source.tar.gz"
fi

mkdir -p "$tmp_dir/src"
tar -xzf "$source_tar" -C "$tmp_dir/src" --strip-components=1

cmake -S "$tmp_dir/src/core" -B "$tmp_dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDENSECORE_PORTABLE=ON \
  -DGGML_NATIVE=OFF \
  -DDENSECORE_BUILD_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
grep -q '^DENSECORE_PORTABLE:BOOL=ON$' "$tmp_dir/build/CMakeCache.txt"
grep -q '^GGML_NATIVE:BOOL=OFF$' "$tmp_dir/build/CMakeCache.txt"
cmake --build "$tmp_dir/build" --target densecore densecore_tests densecore_api_smoke -j"$build_jobs"
if grep -R -E --include='flags.make' -- '-march=native|-mcpu=native' "$tmp_dir/build"; then
  echo "portable release build contains a native CPU compiler flag" >&2
  exit 1
fi
ctest --test-dir "$tmp_dir/build" --output-on-failure

(
  cd "$tmp_dir/src/server"
  CGO_ENABLED=1 \
  CGO_CFLAGS="-I$tmp_dir/src/core/include" \
  CGO_LDFLAGS="-L$tmp_dir/build -ldensecore -lstdc++ -ldl" \
  LD_LIBRARY_PATH="$tmp_dir/build:${LD_LIBRARY_PATH:-}" \
  go test -buildvcs=false -mod=mod ./...
  CGO_ENABLED=1 \
  CGO_CFLAGS="-I$tmp_dir/src/core/include" \
  CGO_LDFLAGS="-L$tmp_dir/build -ldensecore -lstdc++ -ldl" \
  go build -buildvcs=false -mod=mod -o "$tmp_dir/densecore-server" ./cmd/densecore
)

image="densecore:release-smoke-$version"
if [ "${DENSECORE_SKIP_DOCKER:-0}" != "1" ]; then
  docker build "$tmp_dir/src" \
    --build-arg VERSION="$version" \
    --build-arg COMMIT_SHA="$(git -C "$root_dir" rev-parse HEAD 2>/dev/null || echo unknown)" \
    --build-arg DENSECORE_PORTABLE=ON \
    -t "$image"
fi

if [ -z "${DENSECORE_SMOKE_MODEL:-}" ]; then
  echo "DENSECORE_SMOKE_MODEL is required for release qualification" >&2
  exit 1
fi
if [ ! -f "$DENSECORE_SMOKE_MODEL" ]; then
  echo "DENSECORE_SMOKE_MODEL does not exist: $DENSECORE_SMOKE_MODEL" >&2
  exit 1
fi

if [ "${DENSECORE_SKIP_DOCKER:-0}" = "1" ]; then
  LD_LIBRARY_PATH="$tmp_dir/build:${LD_LIBRARY_PATH:-}" \
  MAIN_MODEL_PATH="$DENSECORE_SMOKE_MODEL" \
  "$tmp_dir/densecore-server" serve --host 127.0.0.1 --port "$port" --model "$DENSECORE_SMOKE_MODEL" >"$tmp_dir/server.log" 2>&1 &
  server_pid=$!
else
  bash "$root_dir/scripts/container_model_smoke.sh" "$image" "$DENSECORE_SMOKE_MODEL" "$port"
  exit 0
fi

if ! wait_for_http "http://127.0.0.1:$port/health/live" 120 ||
   ! wait_for_http "http://127.0.0.1:$port/health/startup" 120 ||
   ! wait_for_http "http://127.0.0.1:$port/health/ready" 120; then
  exit 1
fi
curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null

post_json "http://127.0.0.1:$port/v1/chat/completions" \
  '{"model":"densecore-v1","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8,"stream":false}' \
  > "$tmp_dir/nonstream.json"
python3 - "$tmp_dir/nonstream.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
choices = response.get("choices")
if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
    raise SystemExit("non-streaming response has no valid choice content")
if not choices[0]["message"]["content"].strip():
    raise SystemExit("non-streaming response content is empty")
PY

post_json "http://127.0.0.1:$port/v1/chat/completions" \
  '{"model":"densecore-v1","messages":[{"role":"user","content":"Count to two."}],"max_tokens":8,"stream":true}' \
  > "$tmp_dir/stream.sse"
python3 - "$tmp_dir/stream.sse" <<'PY'
import json
import sys

terminal = False
finish_reasons = []
visible = []
with open(sys.argv[1], encoding="utf-8") as handle:
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
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
            delta = choice.get("delta", {})
            visible.append(delta.get("content") or delta.get("reasoning_content") or "")
if len(finish_reasons) != 1 or finish_reasons[0] not in ("stop", "length"):
    raise SystemExit("streaming response must have one valid finish_reason")
if not terminal:
    raise SystemExit("streaming response did not reach [DONE]")
if not "".join(visible).strip():
    raise SystemExit("streaming response content is empty")
PY

kill -TERM "$server_pid"
wait "$server_pid"
server_pid=""
