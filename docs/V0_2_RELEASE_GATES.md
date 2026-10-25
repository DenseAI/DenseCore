# v0.2 implementation and release gates

D0 is 2026-10-25; D21 is 2026-11-15. Scope is A–H: personal PCs,
resource-limited Kubernetes pods and two-socket NUMA. DeepSeek V4/V4.1 is
excluded. This is an execution ledger, not a released-support announcement.

Code, real-model correctness, matched performance and automatic activation are
separate gates. Fixture or compilation success does not qualify a model. A
missing or truncated final answer fails quality even if reasoning contains the
expected text. The current automatic MTP profile registry is empty.

| Track | Implemented candidate | Required remaining release evidence |
|---|---|---|
| A | Request-owned RNG; fatal worker shutdown and terminal-publication fallback; coherent native thread-pool generations | Post-fix ASAN/full-suite and real c=1/2/4 isolation; external same-seed stochastic validation is unavailable until request seed wiring exists |
| B/H | Common cgroup v1/v2, ancestor quota/cpuset/affinity budget; AUTO pooling; context-preserving concurrency fit; shared graph/repack reservations | Final integrated native tests, real embedding parity, limited-container model success and explicit memory rejection |
| B Gemma | Variant labels and stride-aware native float projections | E2B then pinned E4B complete long answers, repeat/c=1/4 quality and layer logits |
| C | Admission-owned snapshot/KV leases, restoration before hit commit, bounded captures | Exact restore/hit/skipped-prefill evidence, #1/#4 and five turns, eviction/cancel/concurrency for each target model |
| D | NextN trunk/head split, executable head, shared sampler distribution, rollback/checkpoint primitives, explicit sequential reference | Same-file live greedy parity and sampling/stop/stream/cancel/isolation; batched K+1 target verification and per-position worker SSM checkpoint integration remain unfinished |
| E | Pinned model cache/downloads; native image profile CI; chart/container/drain harnesses | Final image real serving, checksum failure, actual limited pod and active drain; native Arm validation |
| F | GPT-OSS packed MXFP4 expert math, biases/router/activation, attention sink/window, tokenizer/Harmony | Complete native fixture suite, real-model HTTP/long output/c=1/4/stop/stream and measured peak RSS/context |
| G | Public physical-NUMA preflight and matched ABBA harness | Actual two-socket 35B and 122B measurements; 16/48 physical-core placements and remote-memory penalty |

Qwen3.5 prefix qualification uses `DENSECORE_QWEN35_PREFIX_CACHE_REUSE=1` with
both global prefix/snapshot controls enabled. It remains off by default. Qwen3.6
keeps its model-specific qualification gate. Explicit MTP currently selects a
sequential correctness reference; it makes no acceleration claim. Its target
never advances an unverified suffix. This is separate from completing batched
speculation and its SSM recovery integration.

## Reproduction and evidence

Use `dc-build` with one shared build lock in the local multi-worktree setup.
Build/test source must remain stable during compilation; rebuild after changing
headers or registration before collecting any real-model evidence.

- Native: complete `densecore_tests`, A/D ASAN, and focused model-logit/rollback tests.
- Adapters: changed Go and Python tests; Go race checks for callback/admission paths.
- Real quality: `benchmarks/economics/evaluate_openai_quality.py` requires a final
  answer, successful finish and complete SSE stream. Save raw output and all
  failures. Textual HTTP parity is weaker than full generated-token-ID parity.
- Final package: [container smoke](../scripts/container_model_smoke.sh) and
  [kind smoke](../scripts/kind_model_smoke.sh); [Kubernetes instructions](KUBERNETES.md).
- Artifact identities: [model catalog](MODEL_CATALOG.md).
- CPU comparisons: [personal PC](../benchmarks/personal/README.md) and
  [physical NUMA](../benchmarks/numa/README.md) harnesses.

Performance requires fixed model bytes, binary/library hashes, CPU/build flags,
sampling, cache state and input/output lengths. Keep all five ABBA cycles and
predeclared A/A noise. Judge C4/C4A × c=1/c=4 independently with no compensating
regressions. Promotion requires at least max(3%, A/A noise), with additional
1.15× decode for MTP or +15% 48-core decode for NUMA and no completion-time,
TTFT, throughput or memory regressions. K=2 remains preferred if K=3 has no
significant advantage.

Cloud spending is capped at $50, with $10 held for billing delay/disk/transfer
and price changes; see [session budget](../benchmarks/economics/SESSION_BUDGET.md).
On D0 the configured GCP project had billing disabled and Compute API disabled.
No paid validation was performed or charged. Cloud/platform performance and
native Arm gates therefore remain unavailable until those prerequisites change.

If mandatory correctness or final-package checks remain incomplete on D21,
record release readiness as incomplete and the release delayed. Do not silently
remove a track or turn fixture passes into real-model support.
