# Real c8 resource and request isolation gate

`check_concurrency_budget.py` is a native correctness harness. It requires a
qualified artifact manifest and the README model SHA
`f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec`.
It does not build, download, change defaults, or make speed or answer-quality
claims. Run only after the coordinator identifies a qualified combined binary.

```sh
python3 benchmarks/personal/check_concurrency_budget.py \
  --artifact /absolute/qualified-artifact.json \
  --library /absolute/qualified-build/libdensecore.so \
  --model /absolute/Qwen3.5-0.8B-Q4_K_M.gguf \
  --context 512 --out /absolute/new-evidence-directory
```

Each invocation holds `build.lock` throughout the model lifetime and fixes
context512 (or explicit1024), concurrency8, threads8, greedy sampling, specNONE,
prefixOFF. Effective runtime values must match; there is no automatic retry with
a shorter context or lower concurrency. Both sequential baseline and concurrent
waves use the same engine with capacity8. Exact128 sampled IDs, authoritative
native count128, length termination, and exactly one native completion and token
terminal must match. This length-limited task does not certify a complete answer.

An active cancellation target must terminate with native error metadata while
its peer retains exact baseline IDs. The pressure phase holds an asynchronous
callback behind a bounded gate, submits16 ordinary requests, tries queued
cancellation, then submits at most two oversized inputs: one8MiB stop string and
one8MiB prompt. The caller buffer is distinct from native-owned allocations;
Python string encoding can temporarily duplicate it. A stop may legitimately
fit. Accepted probes are canceled and drained. At least one probe must reject
synchronously with `required_bytes > available_bytes` and no callbacks, including
after recovery. If both fit, pressure is unestablished and qualification remains
incomplete; the harness never escalates caller buffers towards physical OOM.
A final eight-request wave proves recovery and token isolation after drain.

`DetailedMetrics` samples record actual native active count, approximate pending
queue count, active-list size, and KV used/total/free blocks. Active-list size is
not physical batch width. Raw runtime physical-batch histogram lines are retained
separately when emitted. The native active metric must reach8 and never be
observed above8. Public metrics omit worker-deferred queue entries and provide no
recurrent-state allocation count or admission remaining-byte query; these remain
explicit unknowns. An accepted Cancel on the last held request does not by itself
prove its exact physical queue location.

C submissions and barriers use bounded daemon workers. The callback gate is
released in `finally`, before drain/FreeEngine. A stalled C call cannot be freed
concurrently: the child exits with failed evidence and the parent process imposes
a model-lifetime deadline. Normal completion joins metrics polling before freeing
the engine and checks terminal counts again after native shutdown.

The manifest preserves model/artifact/harness/dependency hashes, source revision,
CPU profile/affinity, Python/host, configuration, environment flags, raw callback
IDs/errors, and metrics. Native failure logs and initial failed manifests survive
exceptions and lifetime timeouts. Source and artifact files are verified under
the lock and checked again after execution. No real execution was performed when
this tool was prepared; pure validation tests only establish the harness gates.
