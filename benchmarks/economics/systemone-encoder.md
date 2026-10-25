# System One encoder economics

`run_systemone_load.py` records fixed-workload HTTP or native full-model execution.
`summarize_systemone.py` validates the bound artifacts and counts useful input tokens
only for successful requests. Neither tool creates quality evidence or declares a
GPU comparison. Missing comparators remain `unavailable`.

The runner uses Python's standard library. Native execution loads the existing
`DenseCoreDecisionLoad/Predict/PredictBatch/Free` C ABI with ctypes. A handle
serializes calls; concurrency measures queueing on that handle. `--batch-size N`
uses the native independent-request batch ABI (up to 64 encoder sequences per
batch), with one raw record per request. Native code groups exact sequence lengths
without adding padding. Batch records share their submission/end times; per-request
latencies therefore include the entire native batch. `--batch-size 1` retains the
existing single-request ABI for old-library comparisons.
`--warmup` counts rounds; each round warms `batch-size` requests. Metadata
records both `warmup_rounds` and the multiplied `warmup_requests`. Model loading
and warmup precede the measured interval. Native timing includes
Python/C ABI overhead and output copying. It is not an encoder-kernel-only timing.
HTTP timing includes connection, request/response transport and response validation.
There are no result caches or automatic retries.

## Bound inputs

A workload JSON has `schema: "systemone-workload-v1"`, a `provenance` object
with `model_sha256` and `tokenizer_sha256` matching the manifest, and `requests`:

```json
{
  "id": "stable-id",
  "truncated": false,
  "request": {"state": "fixed input", "questions": {"q": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "A", "b": "B"}}}},
  "sequences": [{"token_ids": [1, 2, 3], "token_keys": [null, "q:0", "state:0"], "markers": [1], "question_type": 0}]
}
```

The shown IDs are illustrative. Produce real IDs and markers with the exact model's
tokenizer and request builder. Each token key names one logical useful input token;
`null` excludes padding and special computation. Repeated shared-state keys across
questions count once and must map to the same token ID. Duplicate keys within one
sequence are rejected. Optional `attention_mask` must contain exactly one integer
0/1 per token; positions masked to zero must have a null token key. When supplied,
`padding_tokens` must match the total mask zeros. The native ABI rejects explicit
padding because it has no padding-mask argument. Legacy fixtures may omit masks;
omission does not prove that an arbitrary sequence is unpadded. New tokenization
receipts emit explicit all-one masks. The producer must establish that sequences and HTTP request
represent the same input; the HTTP runner additionally checks actual input-token
counts, no truncation, all answers present, and no collapsed options. A schema is
not independent tokenizer proof. Preserve the producer's provenance/receipts.

A manifest contains `model_name`, `model_sha256`, `tokenizer_sha256`,
`implementation_sha256`, `precision`, `workload_sha256`, `quality_sha256`,
`pricing_sha256`, `boundary` (`http_e2e` or `full_model`), `result_cache: false`,
and `resource: {"provider": "...", "region": "...", "machine_type": "..."}`.
Hashes refer to exact file bytes. The native runner verifies model/library hashes;
the HTTP deployment identity must be independently established against the manifest.

The quality JSON must contain matching model, tokenizer, implementation, precision,
and workload hashes, `status: "pass"`, `evaluation_artifact` (relative to its own
location), and `evaluation_artifact_sha256`. The evidence file must exist and match. Its JSON must also contain `status: "pass"`
and the same five model/tokenizer/implementation/precision/workload bindings.
The independent quality evaluator, dataset, and thresholds determine whether the
pass is justified. The runner does not evaluate quality itself.

The pricing JSON requires `currency: "USD"`, positive `hourly_usd`,
`includes_all_instance_and_disk_costs: true`, and nonempty `snapshot_id`,
`captured_at`, `purchase_basis`, `provider`, `region`, `machine_type`, `source_url`.
The manifest resource must match these three pricing fields exactly. Run metadata
records the same declared resource and labels cloud identity `operator_attested`.
It separately observes CPU model, architecture, OS, logical CPU count, affinity and
host memory. Those observations do not independently prove a cloud provider,
region, or instance SKU; retain provider identity evidence alongside the run.
Aggregation rejects a run resource that differs from its manifest and price.
For local unpriced experiments, retain raw timings without inventing a zero price;
economics summarization requires a real positive all-instance price.

## Run and aggregate

```sh
python3 benchmarks/economics/run_systemone_load.py \
  --manifest manifest.json --workload workload.json --quality quality.json \
  --pricing pricing.json --output run-a --duration 120 --warmup 2 \
  --concurrency 1 --threads 8 --library build/libdensecore_decision.so --model model.gguf
python3 benchmarks/economics/summarize_systemone.py \
  --manifest manifest.json --workload workload.json --quality quality.json \
  --pricing pricing.json --records run-a.records.jsonl \
  --metadata run-a.metadata.json --output run-a.summary.json
```

For HTTP choose `boundary: "http_e2e"` and `--endpoint URL/v1/systemone` instead
of library/model arguments. Each raw response is retained for inspection. Metadata binds the exact records
file SHA256 and a run UUID carried by every request record. The
client stops submitting at the deadline and waits for all submitted calls before
recording elapsed time. Failures and 429s stay in records and consume interval cost.
HTTP client exceptions, transport failures and gateway timeout responses leave server compute
unresolved; the aggregator rejects these runs until drain can be proven by another
measurement path. Native mode samples whole-process VmRSS every 50 milliseconds;
this includes Python and model allocations, and short peaks may be missed. HTTP
server memory and internal stage times remain unavailable.

The interval cost covers the warm measured interval plus request drain. Provisioning,
load, warmup, teardown, retained disk and other out-of-interval spend still belong in
the experiment's total budget ledger. This report is sustained-run economics, not a
claim about total experiment spend. Select native or HTTP boundaries that match an
external reference before comparing. Perform separate exploration, then final
120-second-or-longer runs under the five-cycle ABBA protocol.

```sh
python3 -m unittest discover -s benchmarks/economics/tests -p test_systemone_economics.py -v
```

## Local diagnostic probes

`run_encoder_probe.py` accepts either the workload schema above or the existing
independent parity fixture. It needs no price or passed-quality wrapper and always
labels its output `diagnostic_only`, quality `unverified`, and economics
`unavailable`. A run cannot be promoted merely because its option logits match.

```sh
python3 benchmarks/economics/run_encoder_probe.py \
  --baseline-library /tmp/baseline/libdensecore.so \
  --candidate-library /tmp/candidate/libdensecore.so --model model.gguf \
  --workload benchmarks/fixtures/laya/parity-inputs.json \
  --output /tmp/encoder-probe --threads 1,2,4,8,16 --batches 1,2,4,8,16 \
  --cases choice,local_attention --duration 1 --cycles 1
```

Each configuration runs ABBA by default. `--legs B` enables candidate-only
exploration and does not produce an A/B estimate. Duration/cycles are configurable; short probes are
exploration and do not satisfy final verification. Full raw outputs, per-leg
configuration and library/model/workload hashes are retained. Both variants use the real native batch ABI when present. A library without the
batch ABI uses the explicit `serial_single` fallback, processing the same request
unit with independent single calls. The
reported execution label prevents mistaking that fallback for old-library native
batching. Probe results include sampled native process memory, p50/p95/p99 request
latencies, and summed useful input tokens when logical token keys exist. A batch
request latency includes completion of its entire batch. Fixture token throughput includes all original token IDs, including
special tokens, so it is diagnostic compute throughput. Useful input tokens are
`null` for these fixtures because they lack logical token-key accounting. It is never useful-token
cost efficiency. Action-logit deltas are recorded but remain under the separate
unverified action gate.

## Final repeatability checks

`check_encoder_abba.py` recomputes twenty bound economic runs (five ABBA
cycles) from their raw artifacts. Every submission window must be at least
120 seconds after warmup, with unique record hashes/run IDs, matching host and
controls, a complete drain interval and passed quality/price evidence. It
reports a paired Student-t lower 95% endpoint for improvement over the CPU
baseline. It does not attach an uncertainty interval to external GPU results.

Local unpriced measurements use a separate checker:

```sh
python3 benchmarks/economics/check_encoder_probe_abba.py \
  /tmp/encoder-final/probe-summary.json --workload /tmp/fixed-workload.json \
  --model models/laya/laya-f16.gguf \
  --baseline-library /tmp/baseline/libdensecore_decision.so \
  --candidate-library /tmp/candidate/libdensecore_decision.so \
  --output /tmp/encoder-final/local-abba.json
```

The local checker requires the same five full ABBA cycles. Quality and economics
remain separate from its performance result. A short diagnostic probe or
candidate-only `--legs B` sweep cannot pass this final check. Both checkers
assume independent cycle estimates; shared-host noise and thermal changes
still need to be considered when interpreting the interval.

[The frozen external reference](encoder_baselines/README.md) distinguishes
primary 128/256-token four-choice comparisons, which remain unavailable, from
the original study's reproducible workload and provisional price-derived
targets. Its static GPU boundary includes preprocessing and postprocessing;
neither native model-only nor HTTP measurements match it automatically.

## Encoder pipeline boundary without HTTP

`server/cmd/benchmark-encoder-pipeline` measures prompt construction, actual
model tokenization, independent-request collation, native forward execution,
calibration, and Go answer-map formatting. It preloads and parses fixed requests
before measurement. Model loading, warmup, JSON parsing/serialization, HTTP and
post-call receipt checks are outside the per-batch pipeline timer. This boundary
can match public studies that time the whole decision pipeline without HTTP;
matching the boundary alone does not establish comparable model, weights, input,
precision, workload, batching, or quality.

```sh
cd server
go build -o /tmp/benchmark-encoder-pipeline ./cmd/benchmark-encoder-pipeline
/tmp/benchmark-encoder-pipeline --model /path/model.gguf --threads 8 \
  --precision fp32 --batch-size 4 --duration 120 --warmup 2 \
  --workload /path/workload.json --output /tmp/pipeline-a
```

Use the build's actual shared native library through the existing rpath or an
explicit `LD_LIBRARY_PATH`. The helper hashes the library actually mapped into
its process, executable, model, workload and actual loaded tokenizer JSON. It
requires model/tokenizer provenance to match. Every measured encoder's token IDs,
markers and question type must exactly match the fixed receipt. Usage must show
no truncation, dropped state or collapsed choices. Unexpected internal encoder
repeats are rejected. Failed batches retain all attempted request records and
consume elapsed wall time while contributing zero useful tokens.

The helper emits the same `.records.jsonl` and `.metadata.json` shape with
`boundary: "encoder_pipeline"`. Its `.config.json` is explicitly a diagnostic
configuration, and its hash occupies `manifest_sha256`; this is not a
passed-quality economics manifest. No dollar efficiency or GPU success is
inferred. To promote evidence, use an independently validated complete manifest,
quality evidence and resource/price artifacts rather than relabeling diagnostics.

`pipeline_active_s` sums only the timed pipeline calls. `elapsed_s` covers the
entire submission loop and full synchronous drain, including evidence checking
and record construction between calls; throughput uses this conservative full
wall interval. JSON files are serialized after the interval. Per-record stage
times describe the entire native batch and are repeated for its requests. Use
`batch_stage_totals_s` for totals; summing repeated per-request stage values would
inflate times. Native sequence batches are chunked at 64 by the serving API while
retaining independent logical requests. Memory remains explicitly unavailable in
this Go helper.

```sh
cd server
env CGO_ENABLED=0 GOCACHE=/tmp/densecore-serving-gocache \
  go test -mod=readonly ./cmd/benchmark-encoder-pipeline
```

For `encoder_pipeline`, the implementation identity binds both the Go executable
and native library. Compute SHA256 over the ASCII canonical JSON bytes below,
with lowercase 64-character file digests substituted, keys sorted exactly as
shown, no whitespace, and no trailing newline:

```text
{"executable_sha256":"<executable digest>","native_library_sha256":"<native digest>"}
```

This is Go `json.Marshal(map[string]string{...})` or Python
`json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')`.
Manifest, quality wrapper, independent evaluation evidence and run metadata must
bind this composite `implementation_sha256` and both component hashes. The
aggregator checks the observed executable/native hashes against the manifest,
checks the independently discovered loaded-library hash, and recomputes the
composite; matching a native library alone cannot clear a changed Go binary.
Diagnostic configuration files still do not provide quality or resource evidence.

## Runtime GGML dependency identity

The main library SHA alone does not identify its dynamically loaded kernels.
Every new manifest, quality wrapper, parsed evaluation, prediction export and
native run must bind the same `runtime_dependencies_sha256` map: actual mapped
GGML library basenames to lowercase SHA256 file digests. Native/probe/pipeline
collect this map from `/proc/self/maps` and reject changes through the run. The
CPU/base GGML libraries are required; all other mapped GGML libraries are retained.
Actual mapped basenames may be versioned (for example `libggml-cpu.so.0.9.8`).
Unneeded `libggml.so` is not assumed to be mapped merely because it was a linker
input. This separate map preserves the existing main-library/composite digest.

Both ABBA checkers require the map to agree between listed/configured/measured
artifacts and to remain identical within each variant. Differently qualified
A/B variants may have different maps. Missing historical maps are not silently
filled in. This covers loaded GGML components; it does not claim to capture the
entire operating system, every system library or every hardware behavior.

For HTTP, the client cannot observe the server's loaded maps. Supply
`--server-identity server-identity.json` to both the runner and aggregator. Its
exact bytes must match manifest `server_identity_sha256`. It must contain:

```json
{
  "origin": "operator_attested_server_deployment",
  "endpoint": "http://server:8080/v1/systemone",
  "model_sha256": "<64 hex>",
  "tokenizer_sha256": "<64 hex>",
  "implementation_sha256": "<64 hex>",
  "precision": "fp32",
  "runtime_dependencies_sha256": {
    "libggml-cpu.so.0.9.8": "<64 hex>",
    "libggml-base.so.0.9.8": "<64 hex>"
  }
}
```

The endpoint must exactly match manifest `endpoint` and CLI `--endpoint`; all
model/runtime bindings must match the quality-qualified manifest. Metadata carries
the identity artifact SHA, endpoint and `operator_attested_server_deployment`
origin. Aggregation reopens and validates the same artifact. This is an operator
attestation of server deployment, not remote observation, a cryptographic
attestation, or a claim that client-loaded libraries describe the server. HTTP
still fails closed when the artifact is missing or mismatched.

## Local validation snapshot, 2026-10-17

These working-copy observations use an i7-10870H under WSL, FP32, eight threads
and batches of sixteen. The original runtime uses sixteen serial single calls;
the candidate uses independent native batching. Both load the same model and
GGML dependencies. Each workload completed five ABBA cycles with 120-second
submission windows, one warmup round and full synchronous drain.

| Actual encoder length | Reserved sources | Original decisions/s | Candidate decisions/s | Result |
|---|---:|---:|---:|---|
| 127 tokens | 4 | 1.638 | 2.077 | Diagnostic only: strict checker rejected two realtime clock corrections |
| 247 tokens | 1 | 0.910 | 1.018 | Local performance verified: mean ratio 1.118, lower 95% endpoint 1.111 |

The 247-token workload contains 239 useful input tokens. Mean useful throughput
was 217.50 versus 243.27 tokens/s; both variants had zero request errors.
Sampled whole-process peak RSS increased from 1.04 to 3.21 GiB. Batch-completion
p95 was 18.052 versus 16.068 seconds, including the serial wrapper for the
original runtime. These are batch latencies, not isolated request latencies.
The 127-token workload contains 119 useful tokens. Its observed point estimate
is retained without a final repeatability pass; no original records were changed
or repeated to clear the clock failure.

The later 247-token measurement uses boot-bound absolute monotonic timestamps;
its clock evidence passed without realtime discontinuity warnings. These results
apply to the reserved inputs and this host, not every input in a length range.
Artifacts, exact commands, raw records, source hashes and binary snapshots reside
in the ignored local directory `private/benchmarks/encoder-economics-20261016/`;
`local-results.json` indexes them. They are not a published benchmark artifact.

Independent numerical parity and HTTP queue/drain checks passed for Laya and
Julia. Separate held-out task diagnostics used only sixteen requests per
dataset/model and do not qualify full task accuracy or INT8 noninferiority.
Q8 remains experimental. Matched public GPU results for the primary workloads,
Julia GPU throughput, ONNX Runtime and the installed llama.cpp comparator remain
unavailable. No tokens/USD or twofold GPU-efficiency conclusion follows from
these local observations. No cloud or GPU execution was performed.

## Native probe clock evidence

New native probe runs write `systemone-run-v2` metadata with observed Linux
`boot_id`, the nonadjustable monotonic-clock implementation, and absolute
`monotonic_started_ns` / `monotonic_finished_ns`. Request offsets and `elapsed_s`
come from that same monotonic origin. Realtime `finished_at` is captured at the
finish boundary before RSS shutdown, dependency hashing or file serialization.

The native probe ABBA checker verifies the nanosecond interval against elapsed
time, orders all legs by absolute monotonic intervals, and requires the same host,
boot identity and clock throughout each ABBA group. Realtime duration, order or
gap discrepancies remain visible as explicit warnings. A realtime clock correction
can therefore be reported without replacing independently observed monotonic
measurement evidence. It does not clear any quality or economic gate.

A partial v2 clock record is rejected. Historical records with no monotonic fields
retain the original strict realtime duration and ordering checks. Historical clock
jumps remain failures; missing evidence is never reconstructed from later host
uptime or boot observations. Raw earlier runs and their archived runner/checker
sources must remain intact. This improved acceptance protocol applies to the native
probe checker; the economic checker retains its strict realtime rules.
