# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
Evidence below belongs to frozen candidate `7ecaf86`, not to unverified follow-up
fixes or a future release.

Intended deployments are a personal PC, a resource-limited CPU Kubernetes pod
and a two-socket NUMA host. The candidate adds resource budgeting, checked model
artifacts, metadata-driven embedding pooling, request/prefix ownership and
stored-KV-precision consistency. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md) for exact scope.

- Native tests: 1263 passed with 14 existing skips; selected ASAN/LSan: 600 passed with
  9 existing skips, zero failures/reports. These passes do not override the
  real-model failures below.
- README-pinned 0.8B and plain 2B prefix checks passed exact native parity,
  five-turn answers, actual snapshot restoration, cancellation and eviction.
  Embedding AUTO/LAST passed 12 bitwise C/Python/HTTP comparisons.
- Qwen3 default/off HTTP modes passed 16 semantic/protocol cases. Explicit-off
  number-only formatting still failed, also in the pinned llama oracle.
- MTP 0.8B/2B K=2 c=1/c=4 greedy/native/HTTP checks passed. First real stochastic
  sampling failed HTTP stop handling in both NONE and MTP. Automatic profiles
  remain zero; complete long-answer/distribution and performance gates remain.
- Gemma E2B c=1 long/repeat checks passed 6/6; c=4 same-input slot isolation
  failed despite semantic/EOS successes. Canonical E4B qualification is pending.
- Local portable amd64 image components passed 4 GiB serving, explicit 1 GiB
  rejection and kind checksum checks. Strict Count and full active drain failed.
  Separate Go-only candidate `f50d958` observed active readiness 503 and normal
  accepted-stream completion after 7.962 seconds, but 149 new POST probes returned
  500 instead of required 503: full drain still fails. Its obsolete kind run was
  canceled; the admission fix and repeat image/kind validation remain pending.

No new speed headline is supported. Historical TTFT gains do not establish
aggregate throughput or completion-time superiority. MTP/NUMA defaults remain
unpromoted. C4/C4A, physical NUMA, 35B/122B, native Arm and real GPT-OSS inference
remain unverified for this candidate; no PC RAM range is claimed for GPT-OSS.
Paid validation spending is $0 of the $50 cap.

Before publication, name the exact source/image/model identities, CPU/RAM/
context/concurrency, complete-answer and isolation outcomes, matched performance
samples and unsupported combinations. Publish required PC RAM only with measured
peak RSS and context. Local raw ledger filenames are recorded in the release
gates; they are not publicly available artifact promises.
