# Qwen3.6 35B serving qualification — historical snapshot

This report records a frozen engineering run from 2026-09-12. It does not
measure the current source tree or the published `denseai/densecore:0.1.0`
image. It is retained to state the scope and limits of the large-model work,
not as a release speed claim.

## Scope

- Model: regular `ggml-org/Qwen3.6-35B-A3B-Q4_K_M.gguf`, SHA-256
  `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`.
- Runtime revision: `e7bf005be81adccf22714d63ff7b6469201402c8`.
- Hosts: 16-vCPU Google C4 and C4A, with host-specific native builds and 16
  inference threads.
- Workload: real `/v1/chat/completions` requests with 390 prompt tokens, 128
  output tokens, greedy decoding, warm-up, prompt caching disabled, and three
  paired measurement cycles. Concurrent throughput uses completed output
  tokens over the full request span, including prefill.

## Outcome and limits

Both host lanes passed the run's serving, output-quality smoke, and request
lifecycle checks. The target of at least 10% more concurrent throughput than
the qualified llama.cpp controls was **not met**; DenseCore was slower at
concurrency four. The C4 control required `--no-repack` after its stock
configuration failed this run's quality check, so the result is not a general
comparison with stock llama.cpp. Three cycles also fall short of the proposed
five-cycle standard for a public comparative claim.

Local portable packages passed compatibility checks but missed the run's
first-token latency target on this large model. A separate amd64-v3 candidate
passed that target on a compatible CPU. These candidate outcomes do not qualify
the published Docker image, an Arm64 artifact, or every model and quantization.

DenseCore makes no comparative performance claim for v0.1.0. Use the current
[release scope](../RELEASE.md) and [hardware support](../HARDWARE_SUPPORT.md)
for deployable surfaces and their limits.
