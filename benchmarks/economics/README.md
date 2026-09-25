# DenseCore Economics Benchmark

This harness measures the cost of successful, quality-gated API completions. It
exists to test a narrow product claim: a CPU deployment can cost less than a
GPU deployment for a declared model, workload, and completion SLO. It does not
assert that CPU inference is cheaper for interactive traffic.

## Evidence contract

Compare CPU and GPU runs only when all of these are identical:

- model file, SHA-256, tokenizer, chat template, sampling parameters, and
  request bodies;
- generated-token target and completion SLO;
- cloud provider, region, operating system, and pricing basis; and
- quality gate.

Run the real `POST /v1/chat/completions` server path. The load client writes one
JSON object per request with queue-inclusive TTFT, end-to-end duration, emitted
text, and API usage. The summary tool counts only responses that are HTTP 2xx,
contain a positive completion-token count, and belong to a passing quality gate.
Failures still consume wall-clock cost and therefore reduce the result.

The first public claim needs five ABBA cycles, not the feasibility run described
below. Keep raw request JSONL, the frozen manifest, pricing snapshot, quality
gate result, server image digest, and binary/model hashes together.

## Billing-safe feasibility run

1. Run `./gcp_preflight.sh` before creating an instance. It is read-only and
   refuses to continue if it sees existing running instances unless explicitly
   overridden.
2. Use a single CPU node first. Run a 10-request local/server smoke and a
   30-request feasibility load. Do not create a GPU node until the CPU artifact,
   quality-gate command, and exact model manifest exist.
3. Stop the instance immediately after copying the artifact. Delete disposable
   instances and their disks after the user-approved comparison is complete.
4. Record retained disks, static IPs, and snapshots separately. Stopping an
   instance ends compute charges but not every retained-resource charge.

The preflight script deliberately does not create, stop, or delete resources.

```bash
./benchmarks/economics/gcp_preflight.sh
```

## Run a load sample

Create a workload file with one OpenAI request body per line. `id` is optional
and is copied into the raw output.

```json
{"id":"doc-001","messages":[{"role":"user","content":"Summarize this document."}],"max_tokens":256,"temperature":0,"seed":123}
```

Run the exact same workload against both servers. `--stream` records first
visible output timing; use `--no-stream` only when the target API cannot return
streaming usage. Each raw artifact writes a sibling `<raw-stem>.metadata.json`,
so `raw-c1.jsonl` and `raw-c4.jsonl` retain separate provenance.

```bash
python3 benchmarks/economics/run_openai_load.py \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --workload /path/to/frozen-async-document.jsonl \
  --output artifacts/economics/cpu/raw.jsonl \
  --requests 30 --concurrency 4 --label densecore-c4 --stream
```

## Summarize cost

The pricing file is an immutable manual snapshot from the provider calculator or
SKU export. Do not fetch a live price during summarization: the result must be
reproducible after public prices change.

```bash
python3 benchmarks/economics/summarize.py \
  --pricing benchmarks/economics/pricing.example.json \
  --quality cpu=artifacts/economics/cpu/quality.json \
  --quality gpu=artifacts/economics/gpu/quality.json \
  --run cpu=artifacts/economics/cpu/raw.jsonl \
  --run gpu=artifacts/economics/gpu/raw.jsonl \
  --duty-cycle 0.20 --duty-cycle 0.50 --duty-cycle 1.00 \
  --slo-p95-ttft-s 15 --slo-p95-e2e-s 45 \
  --output artifacts/economics/summary.json
```

The output reports two distinct metrics:

- `active_runtime_cost_usd`: price during the measured active interval;
- `provisioned_duty_cycle_cost_usd`: an illustrative cost at the declared duty
  cycle, including idle compute time. It is not a measured cold-start result.

Cold-start-to-ready, scale-up failures, and retained storage are separate
experiment dimensions. Do not describe a stopped node as zero cost.

Each compared deployment needs an independent quality result. A passing CPU
quality result cannot qualify an unverified GPU response, or the reverse.

## Run the Qwen3.6 quality gate

The private Qwen3.6 cases assert visible final content only. This catches a
misconfigured reasoning parser that emits hidden reasoning while leaving final
content empty. Run it independently against each measured engine before cost
summarization.

```bash
python3 benchmarks/economics/evaluate_openai_quality.py \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --cases private/benchmarks/economics/qwen36/quality_cases.json \
  --workload private/benchmarks/economics/qwen36/workload.jsonl \
  --label qwen36-densecore-cpu-c4 \
  --model-revision '<immutable-model-revision>' \
  --output private/benchmarks/economics/qwen36/<run-id>/cpu/quality.json
```
