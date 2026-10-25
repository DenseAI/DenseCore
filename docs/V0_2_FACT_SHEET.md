# v0.2 launch fact sheet — draft

**Release readiness is incomplete.** No v0.2 image, tag or release has been
published by this work. Community post text remains the user's responsibility.
The current integrated source is `143a2ad`; sanitizer, private model and earlier
package evidence below have separate identities.

Intended deployments are personal PCs, resource-limited CPU Kubernetes pods and
two-socket NUMA. Candidates add resource budgeting, checked model artifacts,
metadata-driven pooling, request/prefix ownership, speculative verification and
native-backed Python completion metadata. See the [release gates](V0_2_RELEASE_GATES.md)
and [model catalog](MODEL_CATALOG.md).

- Integrated `143a2ad`: native 1312 run / 1298 passed / the same 14 skips /
  zero failures or disabled tests; Go all-package and three repeated race runs
  passed. Python SDK 164 passed / 27 existing optional skips on unchanged SDK
  modules, plus 61 standard-library benchmark tests.
- ASAN/LSan belongs to `3ed5428`: 662 run / 653 passed / 9 existing skips /
  zero failures/reports. It excludes subsequent diagnostic ownership changes;
  it is not a full-source `143a2ad` sanitizer result.
- Real Python on integrated `143a2ad` passed 12 cases plus three raw native-token
  controls: completion counts/finish, cancellation, stream/task cleanup and
  recovery. Prompt counts remain client estimates, not native prompt accounting.
- Actual c=8 resource testing observed active=8/pending=8, rejected oversized
  prompt admission without callbacks and finalized all 36 accepted requests
  once. Greedy parity still fails at EOS token position 50, continuing to the
  128-token limit. Cache OFF and all-thread-policy=8 controls also fail.
- Historical `7ecaf86` pinned 0.8B/plain 2B prefix and embedding checks passed.
  Those results are not automatically inherited by later numerical changes.
  Qwen3 number-only formatting remains a recorded failure.
- Private `5ad54a5` stochastic 0.8B/2B × c1/c4 profiles all fail strict quality:
  six HTTP failures/160 main requests. Native completion/cancel/recovery passes
  and zero literal-stop leaks across 32 probes (27 actual string stops) are
  separate checks. No population-distribution or speed claim follows.
- Private Gemma `ad1aa84` matches c1 IDs in six of eight concurrent requests;
  rank-2 failures persist with cache OFF. Debug `e9662e5` first observes a
  Q4_K projection difference from identical inputs. A fix and support promotion
  remain pending; E4B canonical/mixed validation is held.
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
