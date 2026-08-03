#!/usr/bin/env bash
# =============================================================================
# MoE NUMA sticky-routing mechanism probe
# =============================================================================
# Answers ONE question: on a multi-node host, does sticky expert routing
# actually dispatch on the native MoE path, and where does it send the work?
#
# This is deliberately NOT tier0_numa_validate.sh. That script measures
# THROUGHPUT and correctly refuses to run on synthetic NUMA (`numa=fake=`),
# where node distance is 10/10 and a speedup number would be a false green.
# This script measures MECHANISM ONLY -- which node each expert dispatch went
# to -- which IS observable under synthetic NUMA. It prints no timing numbers
# on purpose, so its output can never be mistaken for a performance result.
#
#   DO NOT add throughput reporting here. Use tier0_numa_validate.sh on real
#   two-socket hardware for that.
#
# It runs two conditions back to back against the same binary:
#   B0_naive  : no NUMA env             -> expert weights land wherever
#                                          first-touch put them
#   N1_sticky : DENSECORE_NUMA_WEIGHTS=round_robin
#                                       -> expert weights partitioned across nodes
#
# Both conditions send the identical prompts with temperature 0, so the total
# dispatch count is identical and ONLY the per-node distribution differs. That
# is what makes the two rows comparable.
#
# Requires: a build with NUMA enabled (configure log must say
# "NUMA support enabled"), and a host with >= 2 NUMA nodes.
#
# Usage:
#   ROOT=~/DenseCore MODEL=/data/models/foo.gguf ./sticky_probe.sh
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD="${BUILD:-$ROOT/build-numa}"
SERVER="${SERVER:-$ROOT/bin/densecore-server}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
THREADS="${THREADS:-6}"
HOST="${HOST:-127.0.0.1}"
OUT="${OUT:-$ROOT/numa_probe}"
BASE_PORT="${BASE_PORT:-18080}"

mkdir -p "$OUT"

# ---- preflight ------------------------------------------------------------
[ -x "$SERVER" ] || { echo "FATAL: server binary missing: $SERVER"; exit 1; }
[ -r "$MODEL" ]  || { echo "FATAL: model missing: $MODEL"; exit 1; }
[ -e "$BUILD/libdensecore.so" ] || { echo "FATAL: no libdensecore.so in $BUILD"; exit 1; }

nodes=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
if [ "$nodes" -lt 2 ]; then
  echo "FATAL: host has $nodes NUMA node(s). This probe needs >= 2;"
  echo "       with one node sticky routing correctly reports state=single_node."
  exit 1
fi

if ! ldd "$BUILD/libdensecore.so" 2>/dev/null | grep -q libnuma; then
  echo "FATAL: $BUILD/libdensecore.so is not linked against libnuma."
  echo "       Rebuild with libnuma-dev + libhwloc-dev installed; the CMake"
  echo "       configure log must print 'NUMA support enabled'."
  exit 1
fi
echo "[preflight] ok: $nodes NUMA nodes, libnuma linked"

# ---- helpers --------------------------------------------------------------
wait_endpoint() { # host port path timeout_s
  local h=$1 p=$2 path=$3 t=${4:-1200} i=0
  while [ "$i" -lt "$t" ]; do
    curl -fsS "http://$h:$p/$path" >/dev/null 2>&1 && return 0
    sleep 2; i=$((i+2))
  done
  return 1
}

run_condition() { # label port [env words...]
  local label=$1 port=$2; shift 2
  local logf="$OUT/$label.server.log"
  echo
  echo "=============================================================="
  echo "### $label   env: ${*:-<none>}"
  echo "=============================================================="
  rm -f "$logf"

  env LD_LIBRARY_PATH="$BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      HOST="$HOST" PORT="$port" THREADS="$THREADS" \
      "$@" "$SERVER" serve --grpc=false >"$logf" 2>&1 &
  local pid=$!

  wait_endpoint "$HOST" "$port" "health/live" 300 \
    || { echo "[$label] ABORT: never became live"; tail -30 "$logf"; kill $pid 2>/dev/null; return 1; }

  # The load POST can run for minutes. Fire it in the background and treat
  # health/startup as the authoritative readiness signal -- a long-running
  # request can lose its response while the server completes the load fine.
  echo "[$label] loading model (minutes)..."
  curl -sS --max-time 3600 -X POST "http://$HOST:$port/v1/models/load" \
       -H 'Content-Type: application/json' \
       -d "{\"model_path\":\"$MODEL\",\"threads\":$THREADS}" >"$OUT/$label.load.json" 2>&1 &
  local loadpid=$!
  local ready=0
  wait_endpoint "$HOST" "$port" "health/startup" 1800 && ready=1
  wait $loadpid 2>/dev/null
  [ "$ready" = 1 ] || { echo "[$label] ABORT: model never ready"; tail -30 "$logf"; kill $pid 2>/dev/null; return 1; }
  echo "[$label] loaded."

  # Identical deterministic workload in both conditions.
  curl -fsS --max-time 900 -X POST "http://$HOST:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"Write one short paragraph about why memory bandwidth matters for inference."}],"max_tokens":96,"temperature":0.0,"stream":false}' \
    >"$OUT/$label.raw.json" 2>&1
  for _ in 1 2; do
    curl -fsS --max-time 900 -X POST "http://$HOST:$port/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"local","messages":[{"role":"user","content":"List three CPU inference bottlenecks."}],"max_tokens":64,"temperature":0.0,"stream":false}' \
      >/dev/null 2>&1
  done

  echo "--- NUMA load-time log ---"
  grep -hE '^\[NUMA\]|NUMA expert partition|NUMA weights' "$logf" | sort -u | sed 's/^/  /' || echo "  (none)"
  echo "--- sticky counters (process-cumulative, last decode summary) ---"
  grep -o 'native_moe_numa_[a-z_]*=[^ ]*' "$logf" | tail -5 | sed 's/^/  /' || echo "  (none)"
  echo "--- which MoE consumer ran ---"
  grep -o 'native_moe_fast_decode_used_ops=[0-9]*\|moe_small_decode_parallel_candidate_ops=[0-9]*' \
    "$logf" | tail -2 | sed 's/^/  /' || echo "  (none)"

  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}

run_condition B0_naive  "$BASE_PORT"
run_condition N1_sticky $((BASE_PORT+1)) \
  DENSECORE_NUMA_WEIGHTS=round_robin \
  DENSECORE_NUMA_EXPERT_PARTITION=1 \
  DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY=0

echo
echo "=============================================================="
echo "### output channel + finish_reason (harness sanity)"
echo "=============================================================="
python3 - "$OUT" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + '/*.raw.json')):
    try:
        o = json.load(open(f))
    except Exception as e:
        print(f"{f.split('/')[-1]}: unparseable ({e})"); continue
    ch = (o.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    print(f"{f.split('/')[-1]}: finish_reason={ch.get('finish_reason')} "
          f"content_len={len(m.get('content') or '')} "
          f"reasoning_len={len(m.get('reasoning_content') or '')} "
          f"usage={o.get('usage')}")
PY

cat <<'EOF'

==============================================================
### How to read this
==============================================================
  native_moe_numa_sticky_state
      enabled               armed, experts span >1 node -> compute spreads
      enabled_degenerate    armed, but ALL experts on ONE node of several, so
                            compute cannot spread. Expected for B0_naive.
      single_node           host has one node
      placement_unavailable expert->node map unreadable
      placement_invalid     some expert's node unknown -> legacy path

  Both armed states dispatch identically; they differ only in what they report.
  B0_naive should read enabled_degenerate and N1_sticky enabled -- if BOTH say
  enabled_degenerate, the round-robin partition did not take effect, so check
  the "NUMA expert partition" line in the load-time log above.

  native_moe_numa_legacy_dispatch_ops
      > 0  means some expert runs were NOT pinned. If state=enabled and this
           is 0, every dispatch went through RunOnNumaNode.

  native_moe_numa_node_dispatch_ops   <- THE INTERESTING ONE
      "0:N,1:M"  work spread across nodes (expected for N1_sticky)
      "0:N"      ALL expert compute pinned to node 0. Expected for B0_naive:
                 with no partitioning, first-touch puts every expert on the
                 loader's node, detection reports that truthfully, and sticky
                 routing faithfully pins everything to that one node. This is
                 the layout state=enabled_degenerate names.

Total dispatch counts must MATCH between the two conditions (identical
workload). If they differ, the conditions were not comparable -- do not
compare the distributions.

These counters are process-cumulative, not per-request: the expert->node map
is built once at model load, so a per-request counter would not answer
"did sticky routing ever run".
EOF
