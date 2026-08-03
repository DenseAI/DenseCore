#!/usr/bin/env bash
# =============================================================================
# Tier-0 NUMA validation: DenseCore NUMA-aware MoE (sticky routing) vs
# NUMA-off baselines AND vs llama-server's own NUMA modes, on a 2-socket box.
#
# Gates (falsifiable):
#   A mechanism : sticky LOWERS remote memory access      -> system numastat + numa_maps
#   B self-gain : N1 (sticky) vs B1 (off + numactl interleave) decode tok/s
#   C DIFFER.   : N1 vs best llama NUMA mode (L0..L3)     <- the real bar
#   D quality   : N1 greedy output == B0 (same engine)    <- HARD, fails on empty
#
# DenseCore server contract (mirrors benchmarks/remote_server_eval.sh):
#   ONE binary  bin/densecore-server serve --grpc=false
#   build select via  LD_LIBRARY_PATH=<core build dir with libdensecore.so>   (numa vs nonuma)
#   HOST/PORT/THREADS are ENV VARS;  model loaded via POST /v1/models/load
# Token-accurate measurement (measure.py) uses non-stream usage, never SSE chunks.
# =============================================================================
set -euo pipefail

# ===== CONFIG (edit) =========================================================
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/bin/densecore-server}"   # prebuilt Go server (one binary)
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
# physical cores (unique core,socket pairs) — NOT logical-cpu rows
THREADS="${THREADS:-$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)}"
CASE_FILTER="${CASE_FILTER:-all}" # all | smoke | densecore | llama | comma-separated labels
if [ "$CASE_FILTER" = "smoke" ]; then
  MAX_TOKENS="${MAX_TOKENS:-128}"
  MIN_TOKENS="${MIN_TOKENS:-32}"
  REPEATS="${REPEATS:-1}"
else
  MAX_TOKENS="${MAX_TOKENS:-256}"
  MIN_TOKENS="${MIN_TOKENS:-64}"
  REPEATS="${REPEATS:-5}"
fi
WARMUP="${WARMUP:-1}"
HOST="${HOST:-127.0.0.1}"
PORT_DC="${PORT_DC:-18080}"
PORT_LL="${PORT_LL:-18090}"
LLAMA_CTX="${LLAMA_CTX:-4096}"
OUTDIR="${OUTDIR:-$PWD/numa_tier0_$(date +%Y%m%dT%H%M%SZ)}"
DO_BUILD="${DO_BUILD:-1}"        # build core .so variants (numa / nonuma)
NUMA_BUILD="${NUMA_BUILD:-$ROOT_DIR/build-numa}"
NONUMA_BUILD="${NONUMA_BUILD:-$ROOT_DIR/build-nonuma}"
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-0}" # 1 turns host confound warnings into aborts
MIN_RAM_MODEL_MULTIPLIER="${MIN_RAM_MODEL_MULTIPLIER:-2}" # abort if host RAM < split model bytes * this
EXPECTED_GCE_MACHINE_TYPE="${EXPECTED_GCE_MACHINE_TYPE:-}" # e.g. c3-standard-176
PARITY_STRICT="${PARITY_STRICT:-1}" # 1 makes Gate D a hard process failure
# Minimum MEASURED local-vs-remote penalty required to call a host 2-socket.
# Deliberately NOT env-overridable: an override is a bypass, and bypassing this
# gate is exactly how you publish a throughput number from a host that has no
# remote penalty at all. Real dual-socket boxes clear these by a wide margin
# (typically 30-60% bandwidth / 50-100% latency). A vNUMA host that only
# *declares* distance 20 lands near 1% / 4% and is rejected here.
NUMA_PENALTY_BW_MIN_PCT=15
NUMA_PENALTY_LAT_MIN_PCT=20

mkdir -p "$OUTDIR"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEASURE="$SELF_DIR/measure.py"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/run.log"; }

# ===== Preflight =============================================================
bytes_to_gib() {
  python3 - "$1" <<'PY'
import sys
print(f"{int(sys.argv[1]) / (1024 ** 3):.1f}")
PY
}

capture_host_info() {
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo
    echo "== lscpu =="
    lscpu || true
    echo
    echo "== memory =="
    free -h || true
    echo
    echo "== numactl -H =="
    numactl -H || true
    echo
    echo "== kernel knobs =="
    printf "numa_balancing="; cat /proc/sys/kernel/numa_balancing 2>/dev/null || true
    printf "transparent_hugepage_enabled="; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    printf "transparent_hugepage_defrag="; cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
  } > "$OUTDIR/host_info.txt"
  curl -fsS --max-time 0.25 \
    -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/machine-type' \
    > "$OUTDIR/gce_machine_type.txt" 2>/dev/null || true
}

check_port_free() {
  python3 - "$HOST" "$1" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port))
except OSError as e:
    print(f"port busy: {host}:{port} ({e})", file=sys.stderr)
    sys.exit(1)
finally:
    s.close()
PY
}

check_numa_distances() {
  python3 - "$OUTDIR/numactl_topology.txt" <<'PY'
import re, sys
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    if re.match(r"^\s*\d+:\s+", line):
        vals = [int(x) for x in line.split(":", 1)[1].split()]
        if vals:
            rows.append(vals)
if len(rows) < 2:
    print("ABORT: could not parse at least two NUMA distance rows", file=sys.stderr)
    sys.exit(1)
for i, row in enumerate(rows):
    if i >= len(row):
        print("ABORT: malformed NUMA distance matrix", file=sys.stderr)
        sys.exit(1)
    local = row[i]
    remote = [v for j, v in enumerate(row) if j != i]
    if not remote or max(remote) <= local:
        print(f"ABORT: NUMA distance row {i} has no remote penalty (local={local}, remote={remote})", file=sys.stderr)
        sys.exit(1)
print("ok")
PY
}

# The distance matrix above is only what the KERNEL WAS TOLD. On a virtualized
# host the SLIT is supplied by the hypervisor and can advertise a remote penalty
# that the hardware never charges: GCP n2-standard-32 reports "2 nodes, distance
# 10/20" but measures -1% bandwidth and +4% latency remote, and its remote
# bandwidth exceeds what any UPI link could carry. That host PASSES
# check_numa_distances, so the distance gate alone cannot keep a fabricated
# topology out. Measure the penalty instead of trusting the declaration.
node_phys_cores() {
  lscpu -p=CORE,NODE | grep -v '^#' | awk -F, -v n="$1" '$2==n {print $1}' | sort -u | wc -l
}

check_numa_penalty() {
  local src="$OUTDIR/numa_penalty.c" bin="$OUTDIR/numa_penalty" out="$OUTDIR/numa_penalty.txt"
  cat > "$src" <<'CEOF'
/* Local-vs-remote NUMA penalty probe: STREAM triad bandwidth + pointer-chase
 * latency, plus proof that the pages actually landed on the bound node.
 * argv[1] = node the caller bound memory to, so we can verify placement. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numaif.h>
#include <time.h>
#include <omp.h>
#define BW_N   (64UL*1024*1024)        /* 512 MiB per array, 3 arrays */
#define LAT_SZ (1UL*1024*1024*1024)    /* 1 GiB chase buffer */
#define STRIDE 4096
#define SAMPLES 500
int main(int argc, char **argv) {
    int expect = (argc > 1) ? atoi(argv[1]) : -1;
    double *a = aligned_alloc(64, BW_N*8), *b = aligned_alloc(64, BW_N*8),
           *c = aligned_alloc(64, BW_N*8);
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 2; }
    #pragma omp parallel for
    for (size_t i = 0; i < BW_N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }
    int pok = 0;
    for (int s = 0; s < SAMPLES; s++) {
        void *p = (char *)a + (size_t)s * ((BW_N*8) / SAMPLES);
        int node = -1;
        if (get_mempolicy(&node, NULL, 0, p, MPOL_F_NODE | MPOL_F_ADDR) == 0 && node == expect)
            pok++;
    }
    double best = 0;
    for (int r = 0; r < 5; r++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < BW_N; i++) a[i] = b[i] + 3.0 * c[i];
        double dt = omp_get_wtime() - t0, gbs = (3.0 * BW_N * 8.0) / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    free(a); free(b); free(c);
    char *buf = aligned_alloc(4096, LAT_SZ);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    for (size_t i = 0; i < LAT_SZ; i += 4096) buf[i] = 1;
    size_t nslot = LAT_SZ / STRIDE, *idx = malloc(nslot * sizeof(size_t));
    if (!idx) { fprintf(stderr, "alloc failed\n"); return 2; }
    for (size_t i = 0; i < nslot; i++) idx[i] = i;
    for (size_t i = nslot - 1; i > 0; i--) {         /* deterministic shuffle */
        size_t j = ((i * 2654435761UL) >> 3) % (i + 1), t = idx[i];
        idx[i] = idx[j]; idx[j] = t;
    }
    for (size_t i = 0; i < nslot; i++)
        *(size_t *)(buf + idx[i]*STRIDE) = (size_t)(buf + idx[(i+1) % nslot]*STRIDE);
    volatile size_t *p = (size_t *)(buf + idx[0]*STRIDE);
    struct timespec t0, t1;
    size_t iters = 10000000;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < iters; i++) p = (size_t *)*p;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec)) / iters;
    printf("pages_ok=%d/%d bw_gbs=%.3f lat_ns=%.2f\n", pok, SAMPLES, best, ns);
    return 0;
}
CEOF
  ${CC:-cc} -O3 -fopenmp -o "$bin" "$src" -lnuma > "$OUTDIR/numa_penalty_build.log" 2>&1 || {
    echo "ABORT: could not build the NUMA penalty probe (needs a C compiler with OpenMP + libnuma-dev)."
    echo "       see $OUTDIR/numa_penalty_build.log"; exit 1; }

  local node_ids cpun memn threads result
  node_ids="$(numactl -H | awk '/^node [0-9]+ cpus:/ {print $2}')"
  : > "$out"
  for cpun in $node_ids; do
    threads="$(node_phys_cores "$cpun")"
    [ "$threads" -ge 1 ] 2>/dev/null || threads=1
    for memn in $node_ids; do
      result="$(OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
                numactl --cpunodebind="$cpun" --membind="$memn" "$bin" "$memn" 2>&1)" || {
        echo "ABORT: NUMA penalty probe failed on cpu=$cpun mem=$memn: $result"; exit 1; }
      printf 'cpu=%s mem=%s threads=%s %s\n' "$cpun" "$memn" "$threads" "$result" >> "$out"
    done
  done
  cat "$out" | tee -a "$OUTDIR/run.log"

  python3 - "$out" "$NUMA_PENALTY_BW_MIN_PCT" "$NUMA_PENALTY_LAT_MIN_PCT" <<'PY'
import re, sys
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    m = re.search(r"cpu=(\d+) mem=(\d+) threads=\d+ pages_ok=(\d+)/(\d+) "
                  r"bw_gbs=([\d.]+) lat_ns=([\d.]+)", line)
    if m:
        rows.append(dict(cpu=int(m[1]), mem=int(m[2]), ok=int(m[3]), tot=int(m[4]),
                         bw=float(m[5]), lat=float(m[6])))
bw_min, lat_min = float(sys.argv[2]), float(sys.argv[3])
if not rows:
    sys.exit("ABORT: NUMA penalty probe produced no parseable rows")

# A run whose pages did not land on the bound node measures nothing. Refuse to
# draw any conclusion from it rather than silently reporting "no penalty".
bad = [r for r in rows if r["ok"] != r["tot"]]
if bad:
    d = bad[0]
    sys.exit(f"ABORT: mbind did not place pages (cpu={d['cpu']} mem={d['mem']}: "
             f"{d['ok']}/{d['tot']} on the bound node). Penalty is unmeasurable here.")

best_bw = best_lat = -1.0
for cpu in sorted({r["cpu"] for r in rows}):
    loc = next((r for r in rows if r["cpu"] == cpu and r["mem"] == cpu), None)
    rem = [r for r in rows if r["cpu"] == cpu and r["mem"] != cpu]
    if not loc or not rem:
        continue
    worst_bw = min(r["bw"] for r in rem)       # remote should be SLOWER
    worst_lat = max(r["lat"] for r in rem)     # remote should be HIGHER
    bw_pen = (loc["bw"] - worst_bw) / loc["bw"] * 100.0
    lat_pen = (worst_lat - loc["lat"]) / loc["lat"] * 100.0
    print(f"node {cpu}: local {loc['bw']:.1f} GB/s / {loc['lat']:.1f} ns  ->  "
          f"remote {worst_bw:.1f} GB/s / {worst_lat:.1f} ns  "
          f"(bw penalty {bw_pen:+.1f}%, lat penalty {lat_pen:+.1f}%)")
    best_bw = max(best_bw, bw_pen)
    best_lat = max(best_lat, lat_pen)

if best_bw < 0 or best_lat < 0:
    sys.exit("ABORT: could not pair a local and a remote measurement per node")
if best_bw < bw_min and best_lat < lat_min:
    sys.exit(
        f"ABORT: this host declares a NUMA remote penalty but does not charge one "
        f"(best measured: bandwidth {best_bw:.1f}% < {bw_min:.0f}%, "
        f"latency {best_lat:.1f}% < {lat_min:.0f}%).\n"
        f"       The distance matrix is advertised by the hypervisor, not the hardware. "
        f"Throughput or remote% numbers from this host would be meaningless.\n"
        f"       Use a shape that is physically dual-socket (e.g. n2-standard-96 = 48 "
        f"physical cores, which cannot fit one socket) or bare metal.")
print(f"ok: measured NUMA penalty bandwidth {best_bw:.1f}% / latency {best_lat:.1f}%")
PY
}

check_model_shards() {
  local dir file prefix first count i shard size total=0
  dir="$(dirname "$MODEL_PATH")"
  file="$(basename "$MODEL_PATH")"
  : > "$OUTDIR/model_shards.txt"
  if [[ "$file" =~ ^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
    prefix="${BASH_REMATCH[1]}"
    first=$((10#${BASH_REMATCH[2]}))
    count=$((10#${BASH_REMATCH[3]}))
    if [ "$first" -ne 1 ]; then
      echo "ABORT: MODEL_PATH must point at shard 00001 for split GGUF: $MODEL_PATH"; exit 1
    fi
    for i in $(seq 1 "$count"); do
      printf -v shard "%s/%s-%05d-of-%05d.gguf" "$dir" "$prefix" "$i" "$count"
      [ -s "$shard" ] || { echo "ABORT: missing/empty split GGUF shard: $shard"; exit 1; }
      size="$(stat -c '%s' "$shard")"
      total=$((total + size))
      printf "%s\t%s\n" "$size" "$shard" >> "$OUTDIR/model_shards.txt"
    done
  else
    [ -s "$MODEL_PATH" ] || { echo "ABORT: MODEL_PATH empty: $MODEL_PATH"; exit 1; }
    total="$(stat -c '%s' "$MODEL_PATH")"
    printf "%s\t%s\n" "$total" "$MODEL_PATH" >> "$OUTDIR/model_shards.txt"
  fi
  echo "$total" > "$OUTDIR/model_total_bytes.txt"
  log "model bytes: $(bytes_to_gib "$total") GiB across $(wc -l < "$OUTDIR/model_shards.txt") file(s)"
}

check_ram_headroom() {
  local model_bytes mem_kib mem_bytes min_bytes
  model_bytes="$(cat "$OUTDIR/model_total_bytes.txt")"
  mem_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  mem_bytes=$((mem_kib * 1024))
  min_bytes=$((model_bytes * MIN_RAM_MODEL_MULTIPLIER))
  log "host RAM: $(bytes_to_gib "$mem_bytes") GiB; required floor: $(bytes_to_gib "$min_bytes") GiB (${MIN_RAM_MODEL_MULTIPLIER}x model bytes)"
  if [ "$mem_bytes" -lt "$min_bytes" ]; then
    echo "ABORT: RAM headroom too small for this expensive run. Lower MIN_RAM_MODEL_MULTIPLIER only for an intentional smoke."; exit 1
  fi
}

preflight() {
  for t in numactl numastat python3 jq curl; do
    command -v "$t" >/dev/null || { echo "MISSING: $t  (apt-get install -y numactl jq python3 curl)"; exit 1; }
  done
  # check_numa_penalty compiles its probe; fail here with a clear cause rather than
  # deep inside the gate that is supposed to be protecting the run.
  command -v "${CC:-cc}" >/dev/null \
    || { echo "MISSING: ${CC:-cc}  (apt-get install -y build-essential libnuma-dev)  -- needed by check_numa_penalty"; exit 1; }
  [ -x "$SERVER_BIN" ] || { echo "SERVER_BIN missing/!x: $SERVER_BIN  (build: go build -o bin/densecore-server ./server/cmd/densecore)"; exit 1; }
  [ -x "$LLAMA_SERVER_BIN" ] || { echo "LLAMA_SERVER_BIN missing: $LLAMA_SERVER_BIN"; exit 1; }
  "$LLAMA_SERVER_BIN" --help > "$OUTDIR/llama_server_help.txt" 2>&1 || true
  grep -q -- "--numa" "$OUTDIR/llama_server_help.txt" \
    || { echo "ABORT: llama-server does not advertise --numa; wrong binary/version: $LLAMA_SERVER_BIN"; exit 1; }
  [ -e "$MODEL_PATH" ] || { echo "MODEL_PATH missing: $MODEL_PATH"; exit 1; }
  check_port_free "$PORT_DC" || exit 1
  check_port_free "$PORT_LL" || exit 1
  check_model_shards
  check_ram_headroom
  numactl -H | tee "$OUTDIR/numactl_topology.txt" >/dev/null
  capture_host_info
  local nodes; nodes="$(awk '/available:/{print $2}' "$OUTDIR/numactl_topology.txt")"
  log "NUMA nodes: ${nodes:-?}   THREADS=$THREADS   MODEL=$MODEL_PATH   CASE_FILTER=$CASE_FILTER"
  if [ "${nodes:-1}" -lt 2 ]; then
    echo "ABORT: <2 NUMA nodes (this shape landed on one socket). Pick a bigger instance/zone."; exit 1
  fi
  check_numa_distances >/dev/null
  log "measuring real local-vs-remote NUMA penalty (declared distances are not evidence)"
  check_numa_penalty
  if [ -s "$OUTDIR/gce_machine_type.txt" ]; then
    local mt; mt="$(cat "$OUTDIR/gce_machine_type.txt")"
    log "GCE machine: $mt"
    if [ -n "$EXPECTED_GCE_MACHINE_TYPE" ] && [ "${mt##*/}" != "$EXPECTED_GCE_MACHINE_TYPE" ]; then
      echo "ABORT: expected GCE machine $EXPECTED_GCE_MACHINE_TYPE, got ${mt##*/}"; exit 1
    fi
  elif [ -n "$EXPECTED_GCE_MACHINE_TYPE" ]; then
    log "[warn] EXPECTED_GCE_MACHINE_TYPE=$EXPECTED_GCE_MACHINE_TYPE set, but GCE metadata unavailable"
  fi
  local nb; nb="$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unknown)"
  if [ "$nb" = "1" ]; then
    if [ "$STRICT_PREFLIGHT" = "1" ]; then
      echo "ABORT: kernel numa_balancing=1. Disable it or set STRICT_PREFLIGHT=0 for an explicitly noisy run."; exit 1
    fi
    log "[warn] kernel numa_balancing=1; OS page migration may confound sticky-routing causality"
  fi
}

# ===== Build core .so variants ===============================================
build_variants() {
  [ "$DO_BUILD" = "1" ] || { log "skip core build"; return; }
  log "build NUMA-ON  (needs libhwloc-dev libnuma-dev) -> $NUMA_BUILD"
  cmake -S "$ROOT_DIR/core" -B "$NUMA_BUILD" -DCMAKE_BUILD_TYPE=Release > "$OUTDIR/cmake-numa.log" 2>&1
  grep -q "NUMA support enabled" "$OUTDIR/cmake-numa.log" \
    || { echo "ABORT: NUMA build did not enable HAS_NUMA (install libhwloc-dev libnuma-dev); see cmake-numa.log"; exit 1; }
  cmake --build "$NUMA_BUILD" -j"$THREADS" >> "$OUTDIR/cmake-numa.log" 2>&1

  log "build NUMA-OFF (hide hwloc/numa from pkg-config) -> $NONUMA_BUILD"
  env PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=/dev/null \
      cmake -S "$ROOT_DIR/core" -B "$NONUMA_BUILD" -DCMAKE_BUILD_TYPE=Release > "$OUTDIR/cmake-nonuma.log" 2>&1
  grep -q "NUMA libs not found" "$OUTDIR/cmake-nonuma.log" \
    || { echo "ABORT: NUMA-OFF build still found NUMA libs (OFF baseline contaminated); see cmake-nonuma.log"; exit 1; }
  cmake --build "$NONUMA_BUILD" -j"$THREADS" >> "$OUTDIR/cmake-nonuma.log" 2>&1
  for d in "$NUMA_BUILD" "$NONUMA_BUILD"; do
    ls "$d"/libdensecore.so* >/dev/null 2>&1 || { echo "ABORT: no libdensecore.so in $d"; exit 1; }
  done
  log "core builds done"
}

# ===== Server lifecycle ======================================================
SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; sleep 2; }
trap stop_server EXIT

wait_health() { # host port path
  local i; for i in $(seq 1 240); do
    curl -fsS "http://$1:$2/${3#/}" >/dev/null 2>&1 && return 0; sleep 0.5
  done; return 1
}

# Disable ALL caching for DenseCore so the prefill/decode wall-clock split is valid
# (prompt cache or prefix reuse would make t_N a cache-hit artifact).
DC_RUNTIME_ENV=(
  DENSECORE_AGENT_PROMPT_CACHE=off
  DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE=1
  DENSECORE_MODEL_LOAD_STRATEGY=force
)

start_densecore() { # build_dir port logfile  + trailing "VAR=val" env words
  local build_dir="$1" port="$2" logf="$3"; shift 3
  env LD_LIBRARY_PATH="$build_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      HOST="$HOST" PORT="$port" THREADS="$THREADS" "${DC_RUNTIME_ENV[@]}" "$@" \
      "$SERVER_BIN" serve --grpc=false >"$logf" 2>&1 &
  SERVER_PID=$!
  wait_health "$HOST" "$port" "health/live" || { echo "ABORT: densecore not healthy ($logf)"; exit 1; }
  curl -fsS -X POST "http://$HOST:$port/v1/models/load" -H 'Content-Type: application/json' \
       -d "{\"model_path\":\"$MODEL_PATH\",\"threads\":$THREADS}" >"$logf.load.json" \
       || { echo "ABORT: model load failed ($logf.load.json)"; exit 1; }
  jq . "$logf.load.json" > "$logf.load.pretty.json" 2>/dev/null || true
  wait_health "$HOST" "$port" "health/startup" || { echo "ABORT: densecore model not startup-ready ($logf)"; exit 1; }
}

start_densecore_sticky() { # build_dir port logfile + optional migration env
  local build_dir="$1" port="$2" logf="$3"; shift 3
  start_densecore "$build_dir" "$port" "$logf" \
    DENSECORE_NUMA_WEIGHTS=round_robin \
    DENSECORE_NUMA_EXPERT_PARTITION=1 \
    DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY=0 "$@"
  grep -q "NUMA expert partition" "$logf" \
    || { echo "ABORT: sticky condition did not engage expert partition ($logf)"; exit 1; }
}

start_densecore_wrapped() { # numactl_prefix... -- build_dir port logfile envwords...
  local pfx=(); while [ "$1" != "--" ]; do pfx+=("$1"); shift; done; shift
  local build_dir="$1" port="$2" logf="$3"; shift 3
  "${pfx[@]}" env LD_LIBRARY_PATH="$build_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      HOST="$HOST" PORT="$port" THREADS="$THREADS" "${DC_RUNTIME_ENV[@]}" "$@" \
      "$SERVER_BIN" serve --grpc=false >"$logf" 2>&1 &
  SERVER_PID=$!
  wait_health "$HOST" "$port" "health/live" || { echo "ABORT: densecore(wrapped) not healthy ($logf)"; exit 1; }
  curl -fsS -X POST "http://$HOST:$port/v1/models/load" -H 'Content-Type: application/json' \
       -d "{\"model_path\":\"$MODEL_PATH\",\"threads\":$THREADS}" >"$logf.load.json" \
       || { echo "ABORT: model load failed ($logf.load.json)"; exit 1; }
  jq . "$logf.load.json" > "$logf.load.pretty.json" 2>/dev/null || true
  wait_health "$HOST" "$port" "health/startup" || { echo "ABORT: densecore model not startup-ready ($logf)"; exit 1; }
}

# llama is launched inline in run_ll using arrays LL_PREFIX (e.g. numactl ...)
# and LL_FLAGS (e.g. --numa distribute), so a numactl prefix AND a trailing
# --numa flag can coexist (the L3 case).

# ===== Per-condition measurement (numastat window EXCLUDES warmup+parity) =====
measure_running() { # label port strict(1=DC parity-critical, 0=llama soft)
  local label="$1" port="$2" strict="${3:-1}"
  # 1) warmup + parity capture FIRST (kept out of the mechanism window).
  #    measure.py exits!=0 on empty/short output, and on output that is a prompt echo
  #    or a repetition loop -> a broken engine cannot pass Gate D. That last part is
  #    load-bearing: Gate D diffs N1 against B0, so an engine that echoes the prompt
  #    echoes it identically in both and the diff goes green (TIER1 section 3.5).
  if [ "$strict" = 1 ]; then
    python3 "$MEASURE" --mode capture --url "http://$HOST:$port" --prompts "$OUTDIR/parity_prompts.txt" \
            --max-tokens 128 --min-tokens "$MIN_TOKENS" --out "$OUTDIR/parity_$label"
  else
    python3 "$MEASURE" --mode capture --url "http://$HOST:$port" --prompts "$OUTDIR/parity_prompts.txt" \
            --max-tokens 128 --min-tokens "$MIN_TOKENS" --out "$OUTDIR/parity_$label" \
            || log "  [warn] $label parity capture failed (non-DC, ignored)"
  fi
  # 2) placement + access snapshot BEFORE timed window
  cat "/proc/$SERVER_PID/numa_maps" > "$OUTDIR/$label.numa_maps.before" 2>/dev/null || true
  numastat > "$OUTDIR/$label.numastat.before" 2>/dev/null || true
  # 3) timed measurement (warmup=0 here; already warmed via capture)
  python3 "$MEASURE" --mode measure --url "http://$HOST:$port" --prompts "$OUTDIR/bench_prompts.txt" \
          --label "$label" --max-tokens "$MAX_TOKENS" --min-tokens "$MIN_TOKENS" \
          --repeats "$REPEATS" --warmup 0 --out "$OUTDIR/$label.json"
  # 4) snapshot AFTER timed window
  numastat > "$OUTDIR/$label.numastat.after" 2>/dev/null || true
  cat "/proc/$SERVER_PID/numa_maps" > "$OUTDIR/$label.numa_maps.after" 2>/dev/null || true
  # 5) DenseCore: prove the prompt cache stayed OFF (else timing split is invalid)
  if [ "$strict" = 1 ]; then
    curl -fsS "http://$HOST:$port/metrics" 2>/dev/null \
      | grep -E "prompt_cache_(hit_total|tokens_reused_total|prefill_tokens_skipped_total)" \
      > "$OUTDIR/$label.cache_metrics.txt" || true
    local reused; reused="$(awk '/prompt_cache_tokens_reused_total/{print $2}' "$OUTDIR/$label.cache_metrics.txt" 2>/dev/null | tail -1)"
    if [ -n "${reused:-}" ] && [ "${reused%.*}" != "0" ]; then
      log "  [WARN] $label prompt_cache_tokens_reused_total=$reused (cache NOT off -> decode timing suspect)"
    fi
  fi
}

# ===== Prompts ===============================================================
write_prompts() {
  cat > "$OUTDIR/bench_prompts.txt" <<'EOF'
You are a meticulous systems engineer. Explain, in depth and step by step, how NUMA (non-uniform memory access) affects the decode throughput of a mixture-of-experts language model running on a dual-socket CPU server. Cover memory bandwidth aggregation, remote access penalty over the socket interconnect, expert weight placement, and thread affinity. Be concrete and quantitative where possible, and finish with a prioritized checklist of mitigations.
===
Write a detailed technical postmortem for a hypothetical incident where a two-socket LLM inference server lost 40% of its decode tokens-per-second after a kernel upgrade changed default memory placement. Walk through hypotheses, the diagnostics you would run, the numastat and perf counters you would inspect, the root cause, and the permanent fix. Keep it long and specific.
EOF
  # SKEW workload: domain-repeated -> concentrates hot experts (exercises profiler/sticky).
  cat > "$OUTDIR/parity_prompts.txt" <<'EOF'
Translate to French and then explain each grammatical choice in detail: "The quick brown fox jumps over the lazy dog near the riverbank at dawn while the farmer watches quietly."
===
Complete this Python function, add a full docstring, and explain the algorithm's complexity in detail:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
EOF
}

# ===== Main ==================================================================
preflight
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  log "PREFLIGHT_ONLY=1: host NUMA qualification passed; stopping before builds or model execution"
  exit 0
fi
build_variants
write_prompts

run_dc() { # label  start-callback...
  local label="$1"; shift
  should_run "$label" || { log "=== skip: $label (CASE_FILTER=$CASE_FILTER) ==="; return; }
  log "=== condition: $label ==="
  "$@"
  measure_running "$label" "$PORT_DC" 1
  stop_server
}
run_ll() { # label ; consumes arrays LL_PREFIX and LL_FLAGS
  local label="$1"; local logf="$OUTDIR/$label.server.log"
  should_run "$label" || { log "=== skip: $label (CASE_FILTER=$CASE_FILTER) ==="; return; }
  log "=== condition: $label ==="
  "${LL_PREFIX[@]}" "$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --host "$HOST" --port "$PORT_LL" \
      -t "$THREADS" -c "$LLAMA_CTX" "${LL_FLAGS[@]}" >"$logf" 2>&1 &
  SERVER_PID=$!
  wait_health "$HOST" "$PORT_LL" "v1/models" || { echo "ABORT: llama not healthy ($logf)"; exit 1; }
  measure_running "$label" "$PORT_LL" 0
  stop_server
}

should_run() {
  local label="$1"
  case "$CASE_FILTER" in
    all) return 0 ;;
    smoke)
      case "$label" in B0_naive|B1_interleave|N1_sticky|L3_llama_numactl) return 0 ;; *) return 1 ;; esac
      ;;
    densecore)
      case "$label" in B0_*|B1_*|N1_*|N2_*) return 0 ;; *) return 1 ;; esac
      ;;
    llama)
      case "$label" in L0_*|L1_*|L2_*|L3_*) return 0 ;; *) return 1 ;; esac
      ;;
    *)
      case ",$CASE_FILTER," in *,"$label",*) return 0 ;; *) return 1 ;; esac
      ;;
  esac
}

# --- DenseCore ---
run_dc "B0_naive"      start_densecore "$NONUMA_BUILD" "$PORT_DC" "$OUTDIR/B0_naive.server.log"
run_dc "B1_interleave" start_densecore_wrapped numactl --interleave=all -- "$NONUMA_BUILD" "$PORT_DC" "$OUTDIR/B1_interleave.server.log"
run_dc "N1_sticky"     start_densecore_sticky "$NUMA_BUILD" "$PORT_DC" "$OUTDIR/N1_sticky.server.log" DENSECORE_MOE_ENABLE_PAGE_MIGRATION=0
run_dc "N2_migration"  start_densecore_sticky "$NUMA_BUILD" "$PORT_DC" "$OUTDIR/N2_migration.server.log" \
        DENSECORE_MOE_ENABLE_PAGE_MIGRATION=1 DENSECORE_MOE_REBALANCE_INTERVAL_MS=2000 DENSECORE_MOE_REBALANCE_TOP_K=8

# --- llama-server: test ALL realistic NUMA configs, summary picks the best as the bar ---
LL_PREFIX=();                       LL_FLAGS=();                  run_ll "L0_llama_plain"
LL_PREFIX=();                       LL_FLAGS=(--numa distribute); run_ll "L1_llama_distribute"
LL_PREFIX=();                       LL_FLAGS=(--numa isolate);    run_ll "L2_llama_isolate"
LL_PREFIX=(numactl --interleave=all); LL_FLAGS=(--numa numactl);  run_ll "L3_llama_numactl"

# ===== Summary ===============================================================
log "=== SUMMARY ==="
python3 - "$OUTDIR" <<'PY' | tee -a "$OUTDIR/run.log"
import json, os, sys, glob
d = sys.argv[1]; rows = {}
for f in glob.glob(os.path.join(d, "*.json")):
    if f.endswith(".load.json"): continue
    try:
        o = json.load(open(f));  rows[o["label"]] = o
    except Exception: pass

def remote_pct(label):
    def parse(p):
        v = {}
        if os.path.exists(p):
            for ln in open(p):
                a = ln.split()
                if len(a) >= 2 and a[0] in ("local_node","other_node"):
                    try: v[a[0]] = sum(float(x) for x in a[1:])
                    except ValueError: pass
        return v
    a = parse(os.path.join(d,f"{label}.numastat.before")); b = parse(os.path.join(d,f"{label}.numastat.after"))
    ln_ = b.get("local_node",0)-a.get("local_node",0); on_ = b.get("other_node",0)-a.get("other_node",0)
    t = ln_+on_; return (on_/t*100.0) if t>0 else float("nan")

order = ["B0_naive","B1_interleave","N1_sticky","N2_migration",
         "L0_llama_plain","L1_llama_distribute","L2_llama_isolate","L3_llama_numactl"]
print(f"{'condition':<20}{'prefill_tps':>12}{'decode_tps':>12}{'remote%*':>10}")
for k in order:
    if k in rows:
        o = rows[k]; pf = o.get('prefill_tps_median'); pf = '-' if pf is None else f"{pf:.1f}"
        print(f"{k:<20}{pf:>12}{o['decode_tps_median']:>12.3f}{remote_pct(k):>10.1f}")
print("* remote% = system-wide numastat other/(local+other) over the timed window (dedicated box; "
      "true per-access locality needs perf PEBS/UPI).")

def g(k): return rows[k]['decode_tps_median'] if k in rows else None
best_llama = max([k for k in ("L0_llama_plain","L1_llama_distribute","L2_llama_isolate","L3_llama_numactl") if k in rows],
                 key=lambda k: rows[k]['decode_tps_median'], default=None)
print("\n--- VERDICTS (decode tok/s) ---")
if g('N1_sticky') and g('B1_interleave'): print(f"B self-gain : N1/B1            = {g('N1_sticky')/g('B1_interleave'):.3f}x  (>1.05 = sticky beats OS interleave)")
if best_llama and g('N1_sticky'):         print(f"C DIFFER.   : N1/{best_llama:<18} = {g('N1_sticky')/rows[best_llama]['decode_tps_median']:.3f}x  (>1.0 = real differentiator vs best llama)")
if g('N2_migration') and g('N1_sticky'):  print(f"migration   : N2/N1             = {g('N2_migration')/g('N1_sticky'):.3f}x")
print("Gate D: parity diff is checked by the shell below (must be identical & non-empty).")
PY

# Gate D — hard. Non-emptiness AND answer-shape are already proven by measure.py
# capture; this only has to establish that routing did not change the output.
if should_run "B0_naive" || should_run "N1_sticky"; then
  log "Gate D (quality): N1_sticky greedy output must equal B0_naive, and be non-empty"
  n1="$OUTDIR/parity_N1_sticky"; b0="$OUTDIR/parity_B0_naive"
  if ! should_run "B0_naive" || ! should_run "N1_sticky"; then
    log "  SKIP: filter omitted B0 or N1, so same-engine sticky parity cannot be checked"
  elif ! find "$n1" -name 'p*.txt' -size +0c 2>/dev/null | grep -q . || \
     ! find "$b0" -name 'p*.txt' -size +0c 2>/dev/null | grep -q .; then
    log "  FAIL: parity outputs missing/empty -> gate INVALID (do not trust speed numbers)"
    [ "$PARITY_STRICT" = "1" ] && exit 1
  elif diff -rq "$n1" "$b0" >/dev/null 2>&1; then
    log "  PASS: N1 == B0 (sticky changed routing only, not output)"
  else
    log "  FAIL: N1 != B0 -> sticky routing changed output; speed numbers are invalid"
    diff -r "$n1" "$b0" | head -40 | tee -a "$OUTDIR/run.log" || true
    [ "$PARITY_STRICT" = "1" ] && exit 1
  fi
else
  log "Gate D (quality): SKIP (CASE_FILTER=$CASE_FILTER omits DenseCore sticky parity)"
fi
log "Done. Results: $OUTDIR"
