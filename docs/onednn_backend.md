# oneDNN Backend (CPU MatMul)

## Build

Configure with oneDNN enabled and optional tests/benchmarks:

```
cmake -S core -B build \
  -DDENSECORE_USE_ONEDNN=ON \
  -DDENSECORE_BUILD_TESTS=ON \
  -DDENSECORE_BUILD_BENCHMARKS=ON
cmake --build build
```

## Runtime toggles

- `DENSECORE_ONEDNN=1` enables oneDNN in AUTO mode.
- `DENSECORE_ONEDNN=0` disables oneDNN in AUTO mode.
- `FORCE_ONEDNN=1` forces oneDNN (overrides thresholds).
- `FORCE_ONEDNN=0` forces DenseCore kernels.
- Thresholds (AUTO only):
  - `DENSECORE_ONEDNN_MIN_M` (default 4)
  - `DENSECORE_ONEDNN_MIN_N` (default 256)
  - `DENSECORE_ONEDNN_MIN_K` (default 256)
  - `DENSECORE_ONEDNN_MIN_MNK` (default 2000000)
  - `DENSECORE_ONEDNN_PACK_M` (default 128)

## Threading

The oneDNN backend runs inside a single GGML custom op (`n_tasks=1`) and uses
oneDNN’s own threading. Thread count is set to `InferenceConfig::num_threads`
(or physical core count if unset) to avoid oversubscription.

## Tests

Run unit tests:

```
./build/densecore_onednn_matmul_test
```

Run microbenchmark:

```
./build/densecore_onednn_matmul_bench
```

## Benchmark harness

Use the Python harness to measure prefill/decode latency and throughput:

```
python python/benchmark_onednn.py --model-path /path/to/model.gguf --threads 16 --prompt-tokens 512 --max-tokens 64
```

Modes (run separately or let the script run all three):

```
python python/benchmark_onednn.py --mode baseline --model-path /path/to/model.gguf
python python/benchmark_onednn.py --mode auto --model-path /path/to/model.gguf
python python/benchmark_onednn.py --mode onednn --model-path /path/to/model.gguf
```

## Troubleshooting checklist

- AMX not detected:
  - Verify CPU supports AMX and BIOS enables it.
  - Check logs: `simd_level` and `effective_cpu_isa` are printed at first oneDNN use.
- oneDNN unexpectedly off:
  - Ensure `DENSECORE_ONEDNN=1` and no `FORCE_ONEDNN=0`.
  - Check shape thresholds (see env vars below).
  - Confirm weights are BF16/INT8 and not INT4.
- Tail latency spikes:
  - Reduce thread count (`threads=`) and ensure no oversubscription.
  - Consider pinning policy and NUMA placement.

## Deployment guide (summary)

- Build flags:
  - `-DDENSECORE_USE_ONEDNN=ON`
  - Optional: `-DDENSECORE_BUILD_TESTS=ON`, `-DDENSECORE_BUILD_BENCHMARKS=ON`
- Runtime env:
  - `DENSECORE_ONEDNN=1|0`
  - `FORCE_ONEDNN=1|0`
  - Thresholds: `DENSECORE_ONEDNN_MIN_M/N/K/MNK`, `DENSECORE_ONEDNN_PACK_M`
- Known pitfalls:
  - INT4 decode path should remain DenseCore-only.
  - Mixed precision mismatch (F32 weights) will fallback to DenseCore.
## Notes

- Prefill-only gating: oneDNN is only used for `M > 1` (batched/prefill shapes).
- BF16 path is enabled for `BF16 x BF16 -> F32`.
- INT8 path is enabled for `INT8 x INT8 -> F32` with per-tensor scales and optional zero points.
- If AMX is not available or oneDNN is disabled, DenseCore kernels are used.
