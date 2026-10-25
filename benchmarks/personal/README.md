# Personal PC request comparison

Start each engine with the same model bytes, context, sampling and CPU limits,
then record its actual build flags, binary/library SHA-256 and host information.
The standard-library client stores raw SSE events, usage, TTFT, full completion
time and aggregate completed-output throughput. It requires a complete final
answer and an expected pattern in that answer; a correct word only in reasoning
does not pass. `max_tokens` truncation fails quality.

```bash
printf '["What is the capital of France? Reply with one word."]\n' > /tmp/prompts.json
python3 benchmarks/personal/compare_servers.py --url http://127.0.0.1:8080 \
  --prompts /tmp/prompts.json --expected-pattern '^Paris[.!]?$' --concurrency 1 \
  --out /tmp/dense-c1.json
```

Use a list of at least four prompts for `--concurrency 4`. A single request does
not become a concurrent workload merely because the worker limit is four.
This tool is a request recorder and answer gate; it does not automatically infer
a speedup. Public performance claims require matched five-block ABBA samples,
A/A noise, same-engine token parity for state changes, and every required
platform/concurrency cell passing independently.

`check_prefix_parity.py` runs fresh native prefix-off/on processes against the
same model/library bytes and records exact sampled token IDs, including stop
tokens, together with authoritative generated counts. It checks fresh request
#1/#4, submitted concurrency 1/2/4, fixed five-turn histories, cancellation and
recovery, and a cache-pressure sequence. It requires actual hit, skipped-prefill,
and hybrid snapshot restoration counters. Submitted concurrency may serialize
on models whose runtime requires exclusive execution.

```bash
python3 benchmarks/personal/check_prefix_parity.py \
  --library build/libdensecore.so --model /path/to/model.gguf \
  --server bin/densecore-server --build-lock /path/to/shared-build.lock \
  --context 2048 --concurrency 1 2 4 --churn 128 --require-eviction \
  --out /tmp/prefix-parity
```

If a large KV pool reclaims ordinary cached blocks while retaining every
snapshot-bearing block, a separate `--concurrency 1 --near-full-pressure`
run sizes one cold prompt using native owned request snapshots and reserves
one output token plus a 16-token margin. This targets snapshot reclamation
without assuming a particular hot probe must be evicted. It remains subject
to the same positive KV/snapshot removal counters and exact off/on token gate;
its c1 result does not replace the 1/2/4 request-isolation matrix.

The optional server probe validates complete final chat answers with
`cache_prompt=true`; text found only in reasoning and truncated answers fail.
Without `--server`, the report explicitly leaves HTTP final answers unverified.
`--require-eviction` requires positive allocator counter deltas for physically
reclaimed KV blocks and removed hybrid snapshot entries and logical payload
bytes. Snapshot removal does not claim immediate physical memory release while
an admission lease still owns a copy. Counters are process cumulative; each
off/on child is a fresh process with one model. A particular hot probe can remain
cached while other entries are reclaimed; its hit/miss is recorded separately.
Old binaries without these counters cannot pass this gate. Increase `--churn`
as needed for the configured KV capacity. Every request, raw native log and HTTP response is preserved in the
output directory. This tool does not enable model defaults or establish speed.

For candidate qualification, use `qualify_prefix.py` once per pinned model. It
holds the shared lock for that model and verifies every file hash in the
candidate artifact before and after both phases. Existing output directories
must be empty, so earlier failed evidence is preserved.

```bash
python3 benchmarks/personal/qualify_prefix.py \
  --artifact /path/to/candidate-artifact.json \
  --library build/libdensecore.so --server bin/densecore-server \
  --model /path/to/pinned-model.gguf --expected-model-sha256 MODEL_SHA256 \
  --build-lock /path/to/shared-build.lock --out /tmp/candidate-prefix
```

The combined gate requires both:

- Main: submitted c1/2/4, request #1/#4, fixed five-turn histories, cancellation
  and recovery, eight cold churn requests, and complete HTTP five-turn final
  answers with prefix OFF/ON.
- Pressure: a separate c1 run with eight churn requests and a native-sized
  near-full cold context, exact OFF/ON token parity, and positive actual KV
  reclamation, snapshot-entry removal, and logical snapshot-byte counters.

A passing main phase alone cannot pass `combined.json`. The targeted pressure
phase replaces blind churn expansion, and does not replace the c1/2/4 or HTTP
checks. Results remain tied to the candidate/model bytes and do not promote
support, defaults, or performance claims.
