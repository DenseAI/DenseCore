#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_dir=${ARTIFACT_DIR:?ARTIFACT_DIR is required}
model_path=${MODEL_PATH:?MODEL_PATH is required}
port=${PORT:-18080}
threads=${THREADS:-16}
source_provenance_dir=${SOURCE_PROVENANCE_DIR:-}
mkdir -p "$artifact_dir" "$artifact_dir/cycles"
mkdir -p "$artifact_dir/warmup"

cp "$root_dir/core/CMakePresets.json" "$artifact_dir/CMakePresets.json"
cp "$root_dir/build-c4a-perf/CMakeCache.txt" "$artifact_dir/CMakeCache.txt"
sha256sum "$model_path" > "$artifact_dir/model.sha256"
stat --printf='model_bytes=%s\n' "$model_path" > "$artifact_dir/model-size.txt"
if [[ -n "$source_provenance_dir" && -d "$source_provenance_dir" ]]; then
  cp "$source_provenance_dir"/* "$artifact_dir/"
else
  git rev-parse HEAD > "$artifact_dir/source-commit.txt"
  git branch --show-current > "$artifact_dir/source-branch.txt"
  git status --short > "$artifact_dir/source-status.txt"
  git diff --binary > "$artifact_dir/source.patch"
fi
lscpu > "$artifact_dir/lscpu.txt"
numactl --hardware > "$artifact_dir/numa.txt" 2>&1 || true
uname -a > "$artifact_dir/uname.txt"
free -h > "$artifact_dir/memory.txt"
cp /etc/os-release "$artifact_dir/os-release.txt"
gcc-13 --version > "$artifact_dir/compiler.txt"
cmake --version > "$artifact_dir/cmake-version.txt"
sha256sum "$root_dir/bin/densecore-server" "$root_dir/build-c4a-perf/libdensecore.so" > "$artifact_dir/binaries.sha256"
{
  for key in project/project-id instance/name instance/id instance/zone instance/machine-type instance/cpu-platform; do
    printf '%s=' "$key"
    curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$key" || true
    printf '\n'
  done
} > "$artifact_dir/gcp-metadata.txt"
printf '%s\n' \
  'DENSECORE_REQUIRE_C4A_SVE=1' \
  'CHAT_ENABLE_THINKING=0' \
  'DENSECORE_PREFIX_CACHE_REUSE=0' \
  'DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE=0' \
  'DENSECORE_QWEN36_PROFILE=1' \
  "threads=$threads" "port=$port" > "$artifact_dir/server-env.txt"

python3 - "$artifact_dir/workload.jsonl" <<'PY'
import json, sys
prompt = " ".join([
    "DenseCore runs a memory centric inference runtime on CPU first systems.",
    "Explain how long prompt prefill, expert routing, and paged attention interact in a production serving path.",
    "Keep the answer factual and structured, but do not use bullet points.",
] * 34)
payload = {"id":"long-1675","model":"densecore-v1","messages":[{"role":"user","content":prompt}],"max_tokens":256,"temperature":0,"top_p":1,"top_k":1,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
open(sys.argv[1], "w").write(json.dumps(payload, separators=(",", ":")) + "\n")
PY
sha256sum "$artifact_dir/workload.jsonl" > "$artifact_dir/workload.sha256"

server_pid=
start_server() {
  local log_path=$1
  DENSECORE_REQUIRE_C4A_SVE=1 CHAT_ENABLE_THINKING=0 DENSECORE_PREFIX_CACHE_REUSE=0 \
  DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE=0 DENSECORE_QWEN36_PROFILE=1 \
    "$root_dir/bin/densecore-server" serve --host 127.0.0.1 --port "$port" --model "$model_path" --threads "$threads" \
    > "$log_path" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/health/live" >/dev/null 2>&1; then return; fi
    sleep 1
  done
  echo "server did not become ready" >&2
  return 1
}
stop_server() {
  if [[ -n ${server_pid:-} ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap stop_server EXIT

start_server "$artifact_dir/server-qa.log"
curl -fsS "http://127.0.0.1:$port/health/live" > "$artifact_dir/health.json"

python3 "$root_dir/benchmarks/economics/evaluate_openai_quality.py" \
  --url "http://127.0.0.1:$port/v1/chat/completions" \
  --cases "$root_dir/private/benchmarks/economics/qwen36/quality_cases.json" \
  --workload "$artifact_dir/workload.jsonl" --label c4a-publication --model-revision "$(cat "$artifact_dir/source-commit.txt")" \
  --no-enable-thinking --require-pass --output "$artifact_dir/quality-think-off.json" > "$artifact_dir/quality-think-off.stdout.json"
python3 "$root_dir/benchmarks/economics/evaluate_openai_quality.py" \
  --url "http://127.0.0.1:$port/v1/chat/completions" \
  --cases "$root_dir/private/benchmarks/economics/qwen36/quality_cases.json" \
  --workload "$artifact_dir/workload.jsonl" --label c4a-publication-think-on --model-revision "$(cat "$artifact_dir/source-commit.txt")" \
  --enable-thinking --require-pass --output "$artifact_dir/quality-think-on.json" > "$artifact_dir/quality-think-on.stdout.json"

# QA and performance use the same binary/config but separate server processes
# so scored timings do not inherit prefix, graph-arena, or RSS history from QA.
stop_server
start_server "$artifact_dir/server-performance.log"

warmup_order=(1 2 4)
printf '%s\n' "${warmup_order[@]}" > "$artifact_dir/warmup-order.txt"
for c in "${warmup_order[@]}"; do
  python3 "$root_dir/benchmarks/economics/run_openai_load.py" --url "http://127.0.0.1:$port/v1/chat/completions" \
    --workload "$artifact_dir/workload.jsonl" --output "$artifact_dir/warmup/c$c.jsonl" \
    --requests "$c" --concurrency "$c" --label "c4a-c$c-warmup" --stream --synchronized-start --require-success \
    > "$artifact_dir/warmup/c$c.stdout.json"
done

order=(1 2 4 4 2 1 1 2 4 4 2 1 1 2 4)
printf '%s\n' "${order[@]}" > "$artifact_dir/run-order.txt"
for i in "${!order[@]}"; do
  c=${order[$i]}; n=$((i + 1)); out="$artifact_dir/cycles/cycle-$(printf '%02d' "$n")-c$c.jsonl"
  ps -o pid,rss,vsz,cmd -p "$server_pid" > "$artifact_dir/cycles/cycle-$(printf '%02d' "$n")-rss-before.txt"
  python3 "$root_dir/benchmarks/economics/run_openai_load.py" --url "http://127.0.0.1:$port/v1/chat/completions" \
    --workload "$artifact_dir/workload.jsonl" --output "$out" --requests "$c" --concurrency "$c" --label "c4a-c$c-cycle$n" --stream --synchronized-start --require-success \
    > "$artifact_dir/cycles/cycle-$(printf '%02d' "$n").stdout.json"
  ps -o pid,rss,vsz,cmd -p "$server_pid" > "$artifact_dir/cycles/cycle-$(printf '%02d' "$n")-rss-after.txt"
done
cp "$artifact_dir/server-performance.log" "$artifact_dir/server-final.log"
python3 "$root_dir/benchmarks/economics/summarize_c4a_publication.py" \
  --artifact-dir "$artifact_dir" > "$artifact_dir/summary.stdout.json"
sha256sum "$artifact_dir"/* "$artifact_dir"/cycles/* "$artifact_dir"/warmup/* \
  > "$artifact_dir/checksums.sha256" 2>/dev/null || true
