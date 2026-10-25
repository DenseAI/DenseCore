# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
The current integrated source is `db9883a`; sanitizer, private model and earlier
package evidence below have separate identities.

Intended deployments are personal PCs, resource-limited CPU Kubernetes pods and
two-socket NUMA. Candidates add resource budgeting, checked model artifacts,
metadata-driven pooling, request/prefix ownership, speculative verification and
native-backed Python completion metadata. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md).

- Integrated `db9883a`: native 1335 run / 1321 passed / the exact same 14 skips /
  zero failures or disabled tests; 108 focused checks passed. The first audit
  relinked the executable with unchanged SHA. The repeat audit is no-work on the
  same source/test binary; this is not serving or model qualification.
- Shared-sampler fixes cover disabled top-k, conditional nucleus filtering and
  grammar intersections. Mock-native full-worker NONE error/peer/recovery regression
  passes; new-source actual HTTP/Python sampling is not qualified.
- Prior `143a2ad` Go all/race3, SDK 164/27 optional skips and benchmark stdlib 61
  results retain that source boundary.
- Selected critical ASAN/LSan on exact `db9883a`: 688 run / 679 passed / the same
  9 prior skips / zero failures, disabled tests or reports. The final actual
  wrapper audit is no-work with unchanged tested binary/source/header. This
  excludes unmerged PLE/JSON work and is not real-model qualification; the prior
  `3ed5428` sanitizer archive remains separate.
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
- Private `5ad54a5` stochastic 0.8B/2B × c1/c4 profiles all fail strict quality:
  six HTTP failures/160 main requests. Native completion/cancel/recovery passes
  and zero literal-stop leaks across 32 probes (27 actual string stops) are
  separate checks. No population-distribution or speed claim follows.
- Private Gemma `d4088c7` cache-ON c4 still fails: rank-0 requests 1/5 produce
  135 IDs against c1's 127, first difference index 52. Same-source cache-OFF,
  mixed and E4B matrices are not run. Private DEBUG `658e626` reproduces the PLE
  delta with 17 exact core stages; identical gated inputs still diverge through
  F32 PLE projection. Its observer is not root-merged and the fix is pending.
- MTP JSON completion remains a source-confirmed gap with separate implementation
  pending; no new-source native/HTTP/Python completion proof is claimed.
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
