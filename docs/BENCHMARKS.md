# Benchmarks

DenseCore and llama.cpp serving Qwen3.6-35B-A3B on the same 16-vCPU Google
Cloud hosts through real `/v1/chat/completions` requests. Measured 2026-09-13.
These are dated engineering measurements, not a speed guarantee for the current
source tree or the published Docker image.

## Results

Medians of six scored runs per engine and concurrency (three ABBA cycles).
DenseCore is listed first; the difference is DenseCore relative to llama.cpp.

### Time to first token

| Host | Requests in flight | DenseCore | llama.cpp | Difference |
|---|---:|---:|---:|---:|
| C4A (Google Axion, Arm) | 1 | 2.07 s | 2.36 s | −12.4% |
| C4A (Google Axion, Arm) | 4 | 5.12 s | 7.58 s | −32.4% |
| C4 (Intel Xeon, x86) | 1 | 2.90 s | 3.62 s | −20.1% |
| C4 (Intel Xeon, x86) | 4 | 8.06 s | 13.21 s | −39.0% |

### Completed output throughput

Output tokens per second over the whole batch span, including prefill.

| Host | Requests in flight | DenseCore | llama.cpp | Difference |
|---|---:|---:|---:|---:|
| C4A (Google Axion, Arm) | 1 | 25.9 | 23.8 | +8.7% |
| C4A (Google Axion, Arm) | 4 | 36.0 | 39.9 | −9.8% |
| C4 (Intel Xeon, x86) | 1 | 13.4 | 13.0 | +3.1% |
| C4 (Intel Xeon, x86) | 4 | 20.2 | 20.7 | −2.5% |

DenseCore delivers the first token sooner in every cell. With four requests in
flight, llama.cpp still finishes the batch faster: DenseCore's median request
completion time was 14.2 s versus 12.8 s on C4A, and 24.6 s versus 24.8 s on C4.

## Output quality

Both engines passed the same output-quality suite on both hosts before scoring:
deterministic extraction, summarisation, classification and arithmetic cases
checked on the visible answer, plus a mixed-arrival concurrency check in which
overlapping requests must return their own distinct English and Korean markers.
DenseCore's small-batch Q8/Q6 kernels follow ggml's pinned arithmetic order and
are tested bitwise against that reference. Cross-engine output is not byte-for-byte
identical, because the two engines use different kernels.

## Method

| Item | Value |
|---|---|
| Model | `ggml-org/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-Q4_K_M.gguf`, revision `baec3ebee244827cda0f4557eafa8b28f7545fa6`, SHA-256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` |
| Hosts | `c4a-standard-16` and `c4-highmem-16`, one engine at a time on each host |
| DenseCore | Native source build of pre-v0.1.0 snapshot `9cd94b5`, 16 inference threads |
| llama.cpp | `llama-server` at `718f7b4175bf8b6af6f5eac09fee10754b3ecddd`, 16 threads. Default CPU configuration on C4A; `--no-repack` on C4, because the stock C4 configuration failed this run's quality suite |
| Workload | Identical 390-token prompts, 128 generated tokens, greedy decoding, warm-up, prompt caching disabled |
| Scoring | Three ABBA cycles, six scored runs per engine and concurrency, all slow samples kept |

## Limits

- DenseCore was a native build (`-march=native` class). The published
  `denseai/densecore:0.1.0` image uses the portable CPU profile and can be
  much slower on this model; use a native or `amd64-v3` source build to
  reproduce these numbers. See [Hardware Support](HARDWARE_SUPPORT.md).
- The C4 comparison is against llama.cpp with `--no-repack`, not its default
  configuration.
- One model, one prompt length, concurrency 1 and 4. Longer prompts, higher
  concurrency and other models were not part of this comparison.
- Three cycles meet this project's engineering bar but not its proposed
  five-cycle bar for a release-qualified comparative claim. v0.1.0 makes no
  release-qualified comparative claim; see [Release](RELEASE.md).
- Raw run archives are not published. The model, revisions and settings above
  are enough to rerun the comparison.
