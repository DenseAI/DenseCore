# Encoder workload and quality artifacts

`normalize_encoder_dataset.py` converts official Parquet files into source-preserving
single-question requests. AG News uses its four class names. Typed Decisions retains
only existing four-option `choice` questions; it preserves the original serialized
state, source ID, and gold probability distribution. Derived questions share a source
ID so the uncertainty calculation groups them rather than counting them as independent.

Pinned evaluation files verified on 2026-10-16:

| Dataset | Commit | File | SHA256 |
|---|---|---|---|
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | `eb185aade064a813bc0b7f42de02595523103ca4` | `data/test-00000-of-00001.parquet` | `71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb` |
| [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `7189f46b02248bd2569df7325c9df8e3b24e90a7` | `all/test-00000-of-00001.parquet` | `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` |

Download using `/resolve/<commit>/<file>`, never a mutable `main` URL. Preserve the
original file. Parquet conversion alone uses optional `pyarrow`; it adds no product
dependency:

```sh
uv run --with pyarrow python benchmarks/economics/normalize_encoder_dataset.py \
  --source /tmp/ag-news-test.parquet --dataset fancyzhx/ag_news --split test \
  --output /tmp/ag-news-requests.jsonl
```

From `server/`, run the exact existing serving tokenizer and input builder:

```sh
CGO_ENABLED=0 go run -mod=readonly ./cmd/prepare-encoder-workload \
  --tokenizer /tmp/model-tokenizer.json --config /tmp/model-config.json \
  --input /tmp/ag-news-requests.jsonl --output /tmp/ag-news-tokenized.jsonl
```

`--config` is JSON of `decision.Config` extracted from the measured model metadata,
including `Model`, `MaxLen` and `HeadMaxLen`; tokenizer marker IDs are read from the tokenizer. `Model` defaults to `laya` for existing configs. `Model: "julia-1"` selects the real Julia tokenizer, strict lossless builder and `<bos>/<eos>/<mask>` markers; unknown model names fail.
The command calls the shared model-specific `PrepareQuestion` before both `Build` and full input accounting, preserving Julia choice descriptions, score rubrics and Boolean defaults. It preserves the input request and emits actual IDs, option markers, and
per-token useful keys and an explicit all-one attention mask. Structural CLS/SEP/MASK markers have null useful keys. Padding
is absent. It detects state, head, and option truncation by comparing the full original
input token count with the built sequence. Tokenization failures fail the command.
Do not use a different model's tokenizer or context configuration.

`prepare_encoder_workload.py --tokenized-cases ... --provenance ... --source ...
--output ...` verifies the pinned source hash and produces the economics runner's
`systemone-workload-v1` format. Provenance requires `dataset`, `revision` (40-character
immutable commit), `split`, and 64-character `source_sha256`, `model_sha256`, and
`tokenizer_sha256`. Model hash identifies the shared source model; precision artifacts
have separate hashes. Requests with any truncation/padding are rejected; unpadded receipts require an explicit all-one mask of exactly the token count. Natural
sequence lengths in `(96,128]`, `(192,256]`, `(384,512]` form the three buckets; exact
lengths remain in every row. Missing buckets are `unavailable`. These are length
ranges, not exact 128/256/512 measurements. Nothing pads, repeats, or truncates source
text to create coverage. Main economics comparisons must use the actual matched
request/token manifest, never infer GPU throughput for a missing length range.

`evaluate_encoder_quality.py --workload ... --reference ... --candidate ...
--tuning-ids ... --mode int8 --output ...` compares paired predictions. Run JSON
requires `model_sha256`, `tokenizer_sha256`,
`implementation_sha256`, `workload_sha256`, `precision`, and `predictions`.
Each prediction has `id`, `labels` in request option order, and `probabilities`.
An explicit JSON array of all source IDs used for calibration/tuning is mandatory;
an empty array means no tuning. Never select final examples using their measured
accuracy or performance. Train splits cannot be final evaluation inputs.

The report includes both models' accuracy against source gold labels and multiclass
Brier score (sum across options, no division by class count). AG News uses its
observed class as a one-hot target. Typed Decisions uses the full annotated gold
probability distribution, as required by its pinned dataset card; malformed or
mislabelled distributions fail validation. This vector-sum scale is explicitly
reported and must not be compared directly with leaderboard scores using a different
Brier normalization. Reference/candidate prediction artifact identities are SHA256
of their actual JSON file bytes, computed by the evaluator rather than trusted from
a self-declared field. Model identity is a separate hash. Thresholds are 0.005 absolute accuracy
drop and 0.005 Brier worsening. Paired empirical Bernstein intervals group source IDs
and use simultaneous 95% coverage assuming independent source examples. Bounds are
conservative; small or correlated samples cannot establish a pass. A point estimate
inside thresholds produces `unverified` when its upper bound crosses a threshold.

FP32 mode keeps `1e-3` option logit and `1e-4` probability tolerances. It additionally
requires `independent_action_parity` from the existing independent action evaluator;
it never derives an action tolerance from an INT8 accuracy allowance. Quality status
is `pass`, `fail`, or `unverified`. Only `pass` can qualify an economics run, whose
model/tokenizer/implementation/precision/workload bindings must match this report.
No real accuracy, parity, or throughput result follows from preparing a workload.

Export actual native predictions with the loaded model's calibration:

```sh
python3 benchmarks/economics/export_encoder_predictions.py \
  --library /tmp/build/libdensecore_decision.so --model /path/to/laya.gguf \
  --workload /tmp/workload.json --precision fp32 --threads 4 \
  --output /tmp/fp32-predictions.json
```

Repeat with `--precision q8_0` and a fresh output path for the candidate, then use
`evaluate_encoder_quality.py --mode q8_0`. The exporter checks the actual model
checksum and loaded tokenizer hash, retains native logits and action logits,
applies the loaded `choice:3-5` calibration and serving temperature clamp, and
writes flushed per-request records to `<output>.raw.jsonl`. An interrupted run
has no complete predictions artifact. Existing output/raw paths are refused.
Accuracy/Brier on a workload labelled `diagnostic` are diagnostic observations;
this exporter sets `quality_qualification: not_evaluated` and cannot supply an
independent parity pass by itself.

Final task-quality preparation can include all complete natural lengths rather than
only performance buckets. Use `prepare_encoder_workload.py --quality-all-lengths
--exclude-source-ids /tmp/excluded-source-ids.json ...`; the exclusion list is explicit
and contains all tuning/diagnostic source IDs. These examples are selected before
prediction and retain exact lengths. Short examples are useful quality evidence but
do not manufacture 128/256-token GPU comparisons. The conservative simultaneous
intervals may remain unverified even on the complete dataset; additional examples
cannot be invented or copied to narrow them.

The prediction exporter accepts `--batch-size 1..64` (default `1`). It groups
requests by exact actual token count, calls the native batch ABI for groups of two
or more, and never pads inputs. Raw records carry request IDs, original workload
indices, actual batch sizes, batch IDs, and batch elapsed time; they are flushed
only after an entire batch succeeds. A failed batch cannot produce a complete
final artifact. Final predictions return to the original workload order. The raw
header binds requested batch size and grouping policy; the final artifact records
all actual batch sizes, including partial final groups. These timings describe
batch completion and are not individual-request latency measurements.

Julia prediction export uses raw softmax with temperature exactly one and ignores
Laya calibration buckets. Non-unit Julia metadata temperatures fail validation.
Artifacts record actual loaded architecture and requested precision; metadata must
match the workload's architecture when pinned and the requested execution precision
when the runtime reports it. Actual model and loaded tokenizer hashes remain required
for either family. Go receipts record model name and numeric question type.

Native prediction evidence additionally requires `runtime_dependencies_sha256`, a
nonempty basename-to-SHA256 map of the actual mapped GGML libraries, including
CPU and base. An unused linker input is not assumed to be mapped.
The exporter obtains it from the loaded process, writes it into the raw header and
final bindings, and requires it to remain unchanged through prediction export.
If dependencies change, only the partial raw artifact remains. Completed prediction
files carry `runtime_dependencies_stable: true`. The evaluator requires this evidence
for both variants and reports candidate dependencies at the top level plus separate
`reference_runtime_dependencies_sha256`; the variants may legitimately differ.
Economics quality/manifest/evaluation joins must compare the candidate map exactly.
An unchanged main native library SHA does not substitute for its dynamic dependencies.
