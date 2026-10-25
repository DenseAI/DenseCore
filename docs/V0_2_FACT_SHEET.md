# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
The current integrated source is `48dd55a`; historical private-model and earlier
package evidence below retain separate identities.

Intended deployments are personal PCs, resource-limited CPU Kubernetes pods and
two-socket NUMA. Candidates add resource budgeting, checked model artifacts,
metadata-driven pooling, request/prefix ownership, speculative verification and
native-backed Python completion metadata. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md).

- Integrated `48dd55a`: native 1358 / passed 1344 / the same 14 skips; strict
  critical ASAN/LSan 727 / passed 718 / the same 9 skips, zero failures/disabled/
  reports. All prior 702 plus 11 added and 14 existing newly selected loader cases
  execute. Actual final wrapper audits are no-work; an immutable 33-file artifact
  binds source and runtime bytes. Go all/race 3, SDK 164/27 optional skips and
  personal stdlib 88 pass. These fixtures do not establish all SDK loader modes.
- Actual `48dd55a` hard-mask NONE/MTP lifecycle passes: four contradictory-mask
  errors with zero output/generated tokens, six healthy peers/recoveries and
  exact terminal/metadata counts. Only healthy MTP requests show proposal
  verification. Original V1 static-TLS failure before engine init is UNEXECUTED;
  reviewed V2 corrects only qualified-library startup preload.
- Actual root `48dd55a` E2B formal 20 cache-ON/OFF c=1/c=4 plus four staggered requests
  match old/own c1 IDs and complete answers; no mixed physical graph is claimed.
  E4B context 2048/capacity 4 init FAIL (plan 5,921,446,144 > 4,681,576,448 available bytes),
  zero generation, failed-init RSS 6,236,108 KiB. This is not RAM savings or a
  successful supported-RAM estimate. Earlier private a9 capacity 1 remains separate.
- Private H `613a` still fails three of eight c8 comparisons and is unmerged.
  Private Linux TLS `a70cd4e` passes three SDK loader modes without execve but
  all three raw modes SIGSEGV during teardown (allocator provenance); unqualified
  and unmerged. Matched `4ad6a35`/`48dd55a` SDK file mode restarts once, `-c`/stdin fail
  static TLS. These loader probes invoke no engine/model.
- Shared-sampler fixes cover disabled top-k, conditional nucleus filtering and
  grammar intersections. Mock-native full-worker NONE error/peer/recovery checks
  pass. Same-source Go all/race checks, SDK 164 unit checks/27 optional skips and
  88 standard-library benchmark tests passed; they are not actual Python JSON proof.
- Source-labeled `4ad6a35` pinned 0.8B and 2B JSON NONE/MTP c1/c4 native and HTTP probes
  pass: each model has 16 main native/32 HTTP requests, native cancel/recovery and
  HTTP stream/nonstream/disconnect/recovery, plus 54 actual grammar-complete
  finishes. Two tiny object cases and a strict brace-ID whitelist are the scope;
  public global action range and real Python JSON are unqualified. These model
  passes do not automatically transfer to source `48dd55a`.
- The unchanged probabilistic matrix on `4ad6a35` is **overall FAIL**: 0.8B c1
  and 2B c1/c4 pass; 0.8B c4 fails one NONE French final-punctuation check among
  160 HTTP requests. One separate NONE stop-instruction negative is retained.
  All 32 stop probes suppress the marker; 31 actual string-stop finishes and
  actual MTP rejection/rollback are recorded. Native completion/cancel/recovery
  and HTTP disconnect/recovery pass. No random-ID/distribution, causal quality
  improvement, speed or automatic/default claim follows, and this failed matrix
  is not silently inherited or replaced by source `48dd55a` lifecycle probes.
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
  Private `a9efe78` subsequently passes 20 E2B cache-ON/OFF c=1/c=4 raw-ID checks
  and four staggered requests, each matching its own c1 complete answer/EOS.
  E4B context 2048/capacity 4 fails capacity before generation; a separately requested
  capacity 1 profile passes one native long case, without a capacity 4 retry or silent reduction.
  Q4/F32 fixes/fixtures are in `4ad6a35` native/ASAN. Private model outcomes do not
  transfer automatically to root4ad6, broad support, HTTP or performance/default gates.
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
