# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
The current integrated source is `4ad6a35`; historical private-model and earlier
package evidence below retain separate identities.

Intended deployments are personal PCs, resource-limited CPU Kubernetes pods and
two-socket NUMA. Candidates add resource budgeting, checked model artifacts,
metadata-driven pooling, request/prefix ownership, speculative verification and
native-backed Python completion metadata. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md).

- Integrated `4ad6a35`: native 1347 run / 1333 passed / the exact same 14 skips /
  zero failures or disabled tests. Selected critical ASAN/LSan: 702 run / 693
  passed / the same 9 skips / zero failures, disabled tests or sanitizer reports.
  All 12 added JSON and Gemma Q4/F32/capture cases are included. Final wrapper
  audits are no-work with stable tested binaries, source, generated headers and
  dependencies; initial relinks remain recorded. This is fixture qualification,
  separate from the real-model probes below; prior `db9883a`/`3ed5428` archives remain.
- Shared-sampler fixes cover disabled top-k, conditional nucleus filtering and
  grammar intersections. Mock-native full-worker NONE error/peer/recovery checks
  pass. Same-source Go all/race checks, SDK 164 unit checks/27 optional skips and
  88 standard-library benchmark tests passed; they are not actual Python JSON proof.
- Actual `4ad6a35` pinned 0.8B and 2B JSON NONE/MTP c1/c4 native and HTTP probes
  pass: each model has 16 main native/32 HTTP requests, native cancel/recovery and
  HTTP stream/nonstream/disconnect/recovery, plus 54 actual grammar-complete
  finishes. Two tiny object cases and a strict brace-ID whitelist are the scope;
  public global action range and real Python JSON are unqualified.
- The unchanged probabilistic matrix on `4ad6a35` is **overall FAIL**: 0.8B c1
  and 2B c1/c4 pass; 0.8B c4 fails one NONE French final-punctuation check among
  160 HTTP requests. One separate NONE stop-instruction negative is retained.
  All 32 stop probes suppress the marker; 31 actual string-stop finishes and
  actual MTP rejection/rollback are recorded. Native completion/cancel/recovery
  and HTTP disconnect/recovery pass. No random-ID/distribution, causal quality
  improvement, speed or automatic/default claim follows.
- Real Python on integrated `143a2ad` passed 12 cases plus three raw native-token
  controls: completion counts/finish, cancellation, stream/task cleanup and
  recovery. Prompt counts remain client estimates, not native prompt accounting.
- Prior `143a2ad` actual c=8 resource testing observed active=8/pending=8,
  rejected oversized prompt admission without callbacks and finalized all 36
  accepted requests once. Greedy parity still fails at EOS token position 50, continuing to the
  128-token limit. Cache OFF and all-thread-policy=8 controls also fail.
- Historical `7ecaf86` pinned 0.8B/plain 2B prefix and embedding checks passed.
  Those results are not automatically inherited by later numerical changes.
  Qwen3 number-only formatting remains a recorded failure.
- Historical private `5ad54a5` stochastic profiles all remain FAIL: six HTTP
  failures/160 requests, with zero marker leaks/32 probes and 27 string stops.
  Those raw failures are preserved rather than replaced by current samples.
- Private Gemma `d4088c7` cache-ON c4 failed: rank-0 requests 1/5 produced 135 IDs
  against c1's 127, first difference index 52. Same-source cache-OFF, mixed and
  E4B matrices were not run. DEBUG `658e626` narrowed a surviving PLE delta;
  Q4/F32 fixes and capture fixtures are now in `4ad6a35` native/ASAN. Historical
  model failures remain recorded; current-source model requalification is separate.
- Separate Go-only `386b800` portable amd64 image passed actual container and
  kind strict drain: same-round readiness 503/new-admission 503/native active=1,
  accepted stream finish/DONE once, 7.596/7.903 seconds after signal commands
  within the original 30-second budget. Strict Count still fails. The changed
  root core has no final image qualification; original init-log capture has
  syntax checks only and awaits a real final kind run.

No new speed headline is supported. Historical TTFT gains do not establish
aggregate throughput or completion-time superiority. Automatic MTP profiles
remain zero; MTP/NUMA defaults are unpromoted. Actual GPT-OSS, physical NUMA,
122B, C4/C4A and native Arm remain unverified; no GPT-OSS PC RAM range is claimed.
The recorded GCP read-only check found both configured projects' billing disabled
and primary Compute API disabled. Paid validation spending remains $0/$50.

Before publication, identify exact source/image/model bytes, CPU/RAM/context/
concurrency, complete-answer and isolation outcomes, matched performance samples
and unsupported combinations. Local raw ledger names in the release gates are
not publicly available artifact promises. Mandatory failures at D21 mean release
readiness remains incomplete and the release is delayed.
