#!/usr/bin/env bash
# Reviewed extraction from the historical private Tier-0 probe. No cloud mutations or DenseCore builds.
set -euo pipefail
OUTDIR=${OUTDIR:?set OUTDIR to preserve physical NUMA evidence}
mkdir -p "$OUTDIR"
NUMA_PENALTY_BW_MIN_PCT=15
NUMA_PENALTY_LAT_MIN_PCT=20
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

command -v numactl >/dev/null
lscpu > "$OUTDIR/lscpu.txt"
numactl -H > "$OUTDIR/numactl_topology.txt"
check_numa_distances
# Requires actual bound-page placement as well as measured bandwidth/latency.
check_numa_penalty
