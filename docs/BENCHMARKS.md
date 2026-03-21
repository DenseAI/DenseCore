# DenseCore Decode Benchmark (Best Combo, Reproducible)

> **Benchmark scope note:** DenseCore's primary differentiation is not peak tok/s alone. The full KPI set is: TTFT, p95/p99 latency, jitter, tokens/sec per dollar, joules/token, memory footprint, queue pressure, and hardware portability. The results below document decode throughput on a single x86 host; broader hardware and KPI coverage is tracked in ongoing benchmark work.

## 1) Goal

- Updated: `2026-02-17`
- Objective: `batch=4`, `threads=8` decode throughput를 최대화하고 `llama.cpp`(`llama-batched-bench`)와 비교
- Workload: `prompt=32`, `gen=32`, `fair-mode`, `repeats=2`
- Models: `Qwen3-0.6B-Q4_K_M.gguf`, `Qwen3-4B-Q4_K_M.gguf`, `Llama-3.2-1B-Instruct-Q4_K_M.gguf`

## 2) Revision / Host

- Repository: `DenseCore`
- Commit: `568d3814025aa49763c5cd7ef0c6e4c7844355a2`
- Branch: `feat/universal_graph_builder`
- OS: `Linux 5.15.153.1-microsoft-standard-WSL2 x86_64 GNU/Linux`
- CPU: `Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz`
- Core/Thread: `8 cores / 16 threads`
- ISA: AVX2 (`avx2`, `f16c`), no AVX-512

## 3) Best Build Combo

Build command:

```bash
cmake -S core -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDENSECORE_LTO=ON \
  -DDENSECORE_USE_SPDLOG=OFF \
  -DDENSECORE_BUILD_TESTS=OFF \
  -DDENSECORE_BUILD_BENCHMARKS=OFF
cmake --build build -j8
```

Verified from `build/CMakeCache.txt`:

- `CMAKE_BUILD_TYPE=Release`
- `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`
- `DENSECORE_LTO=ON`
- `DENSECORE_USE_SPDLOG=OFF`
- `DENSECORE_BUILD_TESTS=OFF`
- `DENSECORE_BUILD_BENCHMARKS=OFF`
- `DENSECORE_USE_MIMALLOC=ON`
- `DENSECORE_USE_ONEDNN=OFF`

## 4) Best Runtime Combo

- `DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE=on`
- `DENSECORE_DECODE_GRAPH_CACHE=0`
- `DENSECORE_ENABLE_Q4K_BATCHED_KERNEL=0`
- `DENSECORE_ENABLE_QUANT_NRC_BATCH=1`
- `DENSECORE_BENCH_DIRECT_CALLBACK=1`
- `DENSECORE_BENCH_DECODE_BATCH_FAST_PATH=1`
- `DENSECORE_BENCH_FAST_PATH_MAX_BATCH=8`
- `DENSECORE_BENCH_RESPECT_THREADS=0`
- `DENSECORE_BENCH_TPS_MODE=all`
- `DENSECORE_BENCH_RECORD_TOKEN_TIMES=0`

## 5) Reproduction Command

```bash
THREADS=8 PROMPT_TOK=32 GEN_TOK=32 REPEATS=2 BATCHES="4" \
OUT_PREFIX=benchmarks/results/fair_decode_bestcombo_final_20260217 \
FAIR_MODE=1 PARALLEL_SUBMIT=0 COMPARE_METRIC=tps FILTER_INVALID=1 \
DENSECORE_BENCH_TPS_MODE=all \
DENSECORE_BENCH_DIRECT_CALLBACK=1 \
DENSECORE_BENCH_DECODE_BATCH_FAST_PATH=1 \
DENSECORE_BENCH_RECORD_TOKEN_TIMES=0 \
DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE=on \
DENSECORE_DECODE_GRAPH_CACHE=0 \
DENSECORE_ENABLE_Q4K_BATCHED_KERNEL=0 \
DENSECORE_ENABLE_QUANT_NRC_BATCH=1 \
DENSECORE_BENCH_RESPECT_THREADS=0 \
./benchmarks/run_fair_decode_benchmark.sh
```

## 6) Final Result (Best Combo)

Source: `benchmarks/results/fair_decode_bestcombo_final_20260217_summary.csv`

| Model | DenseCore `mean_tps` | llama.cpp `mean_tps` | Delta (`mean_tps`) |
|---|---:|---:|---:|
| Qwen3-0.6B | 144.0250 | 160.4350 | -10.23% |
| Qwen3-4B | 23.3950 | 25.4450 | -8.06% |
| Llama-3.2-1B | 72.8650 | 90.4800 | -19.47% |

Key point:

- `qwen3_0.6b` DenseCore `mean_tps=144.0250` (이번 탐색 라운드 최고)

## 7) Why This Combo Was Selected

Build A/B (same runtime profile, 1-repeat head-to-head):

| Build config | Qwen3-0.6B Dense `mean_tps` | 3-model Dense 평균 TPS |
|---|---:|---:|
| `LTO=ON, SPDLOG=OFF` | 140.230 | 78.963 |
| `LTO=OFF, SPDLOG=ON` | 123.300 | 73.313 |

Runtime A/B (same build family, 2-repeat):

| Runtime config | Qwen3-0.6B Dense `mean_tps` | 3-model Dense 평균 TPS |
|---|---:|---:|
| `base` (cache off, nrc on) | 140.345 | 77.313 |
| `cache_on` | 123.010 | 73.207 |
| `cache_on + nrc0` | 117.340 | 69.492 |

Light-build 대비 개선:

- Reference: `benchmarks/results/fair_decode_light_build_20260217_summary.csv`
- Qwen3-0.6B Dense `121.2200 -> 144.0250` (`+18.81%`)
- 3-model Dense 평균 TPS `73.020 -> 80.095` (`+9.69%`)

## 8) Historical Peak Snapshot (Reference Only)

- Source: `benchmarks/results/fair_decode_cycle2_b4_p32_base_summary.csv`
- DenseCore Qwen3-0.6B `mean_tps=145.7200`
- DenseCore Qwen3-4B `mean_tps=33.2350`
- DenseCore Llama-3.2-1B `mean_tps=102.7150`
- Note: 위 값은 이전 실험 스냅샷이며, 최종 재현 기준은 섹션 5-6의 `fair_decode_bestcombo_final_20260217` 결과.

## 9) Artifacts

- `benchmarks/results/fair_decode_bestcombo_final_20260217_raw.csv`
- `benchmarks/results/fair_decode_bestcombo_final_20260217_paired.csv`
- `benchmarks/results/fair_decode_bestcombo_final_20260217_summary.csv`
- `benchmarks/results/fair_decode_bestcombo_final_20260217_summary.json`
- `benchmarks/results/fair_decode_lto1_log0_r1_20260217_summary.csv`
- `benchmarks/results/fair_decode_lto0_log1_r1_20260217_summary.csv`
- `benchmarks/results/fair_decode_bestsearch_base_20260217_summary.csv`
- `benchmarks/results/fair_decode_bestsearch_cache_on_20260217_summary.csv`
- `benchmarks/results/fair_decode_bestsearch_cache_on_nrc0_20260217_summary.csv`
- `benchmarks/results/fair_decode_light_build_20260217_summary.csv`
