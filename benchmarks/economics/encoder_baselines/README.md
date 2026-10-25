# Frozen encoder references

`laya-cuda-bench.manifest.json` pins commit
`3520301f965434e1c4235941cd0121c0a3017bed` of
[bhushankinge/laya-cuda-bench](https://github.com/bhushankinge/laya-cuda-bench/tree/3520301f965434e1c4235941cd0121c0a3017bed),
captured 2026-10-16. It records 71 source byte hashes, the original model
manifest, and 149 static timing cells for the standard Laya checkpoint across
all four measured GPU configurations and all available backends. It retains
unavailable cells through source references rather than manufacturing results.
No external Python code is executed by the capture utility.

Download the pinned data (network required), or verify an existing checkout:

```sh
python3 benchmarks/economics/encoder_baselines/reference.py download --root /tmp/laya-gpu-reference
python3 benchmarks/economics/encoder_baselines/reference.py verify --root /tmp/laya-gpu-reference
python3 benchmarks/economics/encoder_baselines/reference.py requests --root /tmp/laya-gpu-reference --bucket short --batch 256 --output /tmp/laya-short-b256.jsonl
```

The request export reproduces the original seed-0 bucket shuffle and
round-robin state/question selection. It contains one named question per row,
rotating `route` (3 choices), `severity` (3 scores), and `needs_human` (2 options).
It exports request payloads; it does not tokenize them or establish runtime
parity. Use the pinned tokenizer, upstream prompt construction and limits
before comparing token totals. The raw rows record the complete nonpadding
token count and padded sequence length for each timed iteration.

The labels `short` (0–80) and `long` (400–472) measure **state-only** token
lengths. They do not mean 80- or 472-token model inputs. These workloads do
not establish a baseline for either 128- or 256-token complete inputs with
four choices. Those primary GPU comparison cells remain `unavailable`.

The original static timing boundary includes prompt construction, tokenization,
collation, transfers, forward, synchronization, calibration and formatting.
It excludes model loading and HTTP. Each cell uses 20 warm-up and 50 measured
iterations; repeat counts are recorded individually. Model-only CPU timing or
HTTP CPU timing cannot be compared directly with these cells.

The derived rates use the median of each repeat's median milliseconds, then
divide batch decisions or nonpadding input tokens by that time. They are
static reference estimates, not a sustained 120-second test or an external
confidence interval. The frozen raw data recomputes the best H100 NVL short
rate as 1490.079 decisions/s, while the report table says 1493. RTX PRO 6000
short recomputes as 1043.235 decisions/s. Preserve this distinction instead of
silently replacing raw results with report numbers.

The checkpoint SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
matches the checkpoint hash recorded by the repository's existing
`benchmarks/fixtures/laya/reference-parity.json`. This match establishes the
reported weight-file identity; current CPU artifacts still require their own
hash and tensor/parity verification. External FP16 parity used probability
tolerance 0.02 and 63/63 argmax agreement. That does not establish DenseCore's
1e-4 probability parity or task accuracy noninferiority.

`laya-tokenizer-identity-c5d78730.json` separately verifies the tokenizer omitted
from the original study's SHA list. The exact pinned checkpoint's
`tokenizer/tokenizer.json` was downloaded and compared byte-for-byte with the
locally extracted CPU tokenizer: both are 3,583,228 bytes with SHA-256
`6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`.
This establishes tokenizer-file identity; matching token counts alone would
not establish it. The frozen source model manifest and 149 timing cells remain
unchanged. Prompt construction, token/marker receipts and numerical quality
still require their separate evidence.

The study's card amortization prices cannot be combined with CPU cloud
instance prices. `pricing-candidates-20261016.json` captures current official
Runpod Pod rates for the measured GPU models, including CPU/RAM allocations,
and GCP C4/C4A instance and disk rates. Seven original HTML captures, their
SHA-256 hashes and compressed capture hashes are retained in the private
experiment artifacts. Published Pod packages are useful price candidates;
their regions, actual host/CPU allocations, storage configuration and deployment
availability have not been verified through an instance quote. Applying the
study's performance to a different cloud host is a derived planning estimate.

The pricing manifest includes 32 explicitly provisional target scenarios using
20 GB of Runpod container storage and 20 GiB of GCP Hyperdisk Balanced. The
Runpod monthly storage rate is divided by an assumed 730 hours; this divisor
is not a verified billing rule. GCP's published hourly disk rate needs no such
conversion. The targets concern the original external workloads and originally use raw
nonpadding input-token sums, including structural tokens. Logical useful GPU
rates can now be derived for each exact fixed batch by reconstructing its seed-0
requests, obtaining real tokenizer receipts, subtracting structural tokens, and
scaling that cell by `logical_useful_tokens / nonpadding_tokens`. The study builds
its request set once and repeats it for every timed iteration; this is a mapping
of the same fixed inputs, not a length interpolation. Local preparation verified
all 149 cells and 20,050 timed raw rows against reconstructed nonpadding sums.
These preparation artifacts are optional local evidence; this manifest and the
commands below remain self-contained and do not require ignored private copies.

Long batch 64 contains four truncated requests; long batch 256 contains 24.
Their post-truncation logical rates may be retained as diagnostics, but they have
no qualified valid-input numerator under this task's no-truncation requirement.
Do not remove those requests or use their rates to qualify the original workload.
All short batches and long batches 1/16 avoid this truncation gate. Matching
quality, actual price/host and the full pipeline boundary still remain required;
qualified CPU/GPU economics are unavailable.

To reproduce a short-batch derivation after the request export above, use the
pinned model's tokenizer JSON and its matching input-builder Config JSON:

```sh
python3 - <<'PYCODE'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path('/tmp/laya-short-b256.jsonl').read_text().splitlines()]
Path('/tmp/laya-short.requests.jsonl').write_text(''.join(json.dumps({
    'id': r['id'], 'request': {'state': r['state'], 'questions': r['questions']}
}) + '\n' for r in rows))
PYCODE
env CGO_ENABLED=0 go -C server run -mod=readonly ./cmd/prepare-encoder-workload \
  --tokenizer /path/to/pinned/tokenizer.json --config /path/to/model-config.json \
  --input /tmp/laya-short.requests.jsonl --output /tmp/laya-short.receipts.jsonl
python3 - <<'PYCODE'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path('/tmp/laya-short.receipts.jsonl').read_text().splitlines()]
assert not any(r['truncated'] for r in rows), 'No qualifying numerator for truncated request sets'
useful = sum(r['useful_input_tokens'] for r in rows)
compute = sum(r['compute_input_tokens'] for r in rows)
m = json.loads(Path('benchmarks/economics/encoder_baselines/laya-cuda-bench.manifest.json').read_text())
for c in m['cells']:
    if c['state_bucket'] == 'short' and c['batch'] == 256:
        assert c['valid_input_tokens_per_iteration'] == [compute]
        print(c['box'], c['backend'], 'derived useful tokens/s',
              c['valid_input_tokens_per_second_derived'] * useful / compute,
              'quality and price/host unqualified')
PYCODE
```

Neither primary 128/256 GPU cell becomes available through these estimates.
No scenario qualifies economics, establishes actual cloud throughput, or
selects the strongest comparable GPU price/performance baseline. GCP H100 SXM
must still not substitute for the study's H100 NVL.

`julia-1.manifest.json` pins the currently published
[SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85)
revision. Its published provenance says `posttrained-candidate` and
`quality_gate: false`. The 577,189,056-byte checkpoint was downloaded locally
and its SHA-256 independently matched the listed weight hash. The local
conversion receipt is retained under
`private/benchmarks/encoder-economics-20261016/julia-conversion.json`. This
integrity check does not establish task quality. Historical H200 BF16 accuracy
in its model card supplies no matched throughput baseline and cannot establish
the quality of this current candidate. Julia GPU economics remain `unavailable`.
