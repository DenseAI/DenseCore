# oneDNN Benchmark Report

## Environment

- CPU:
- AMX enabled (Y/N):
- OS:
- oneDNN version:
- DenseCore commit:
- Threads:
- NUMA policy:
- Model:
- Quantization:

## Workloads

- Prefill: prompt_tokens=
- Decode: max_tokens=
- Concurrency:

## Results (Prefill)

| Mode | tokens/sec | p50 prefill (ms) | p95 prefill (ms) | notes |
| ---- | ---------- | ---------------- | ---------------- | ----- |
| baseline | | | | |
| auto | | | | |
| onednn | | | | |

## Results (Decode)

| Mode | tokens/sec | p50 decode (ms) | p95 decode (ms) | notes |
| ---- | ---------- | --------------- | --------------- | ----- |
| baseline | | | | |
| auto | | | | |
| onednn | | | | |

## Interpretation

- When oneDNN helps:
- When oneDNN does not:
- Decode impact:

## Recommended heuristics

- min_m:
- min_n:
- min_k:
- min_mnk:
- dtype gates:

## Deployment guide

- Build flags:
- Env vars:
- Troubleshooting:
