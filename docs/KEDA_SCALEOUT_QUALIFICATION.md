# DenseCore KEDA Scale-Out Qualification

This repository now includes a reproducible harness at
[`scripts/qualify_keda_scaleout.sh`](../scripts/qualify_keda_scaleout.sh) for
real KEDA plus Prometheus scale-out qualification against the DenseCore server
path and real `/v1/chat/completions` requests.

## Scope

The harness is intentionally narrow:

- it installs pinned KEDA and `kube-prometheus-stack` releases only into
  harness-owned namespaces and release names, or reuses an already matching
  harness-owned install without upgrading it;
- it discovers the Prometheus service by stable release labels unless a service
  override is explicitly provided;
- it deploys DenseCore from the local chart plus the public `dense-base`
  dependency in a temporary repo copy so the working tree is not polluted;
- it exercises a fixed single-replica control run and a `minReplicaCount: 1`,
  `maxReplicaCount: 2` autoscaled run;
- it records 5-second telemetry to JSONL and CSV;
- it records per-request logs for streaming and non-streaming chat calls;
- it captures per-pod metric snapshots before and after load so the second pod
  must show real request handling, not just a replica-count change;
- it captures deployment, pod, HPA, ScaledObject, and namespace event artifacts
  for timing and readiness reconstruction;
- it requires explicit frozen metadata for real runs before deployment;
- it limits cleanup to a harness-owned DenseCore namespace and release.

The harness fails closed when:

- the PromQL query is omitted;
- the query uses `or vector(0)` to mask missing scrape data;
- the query does not return a real value from Prometheus;
- the query still returns no real value after the bounded scrape-settle wait;
- real control/autoscale metadata is missing;
- DenseCore health or metrics endpoints do not come up;
- DenseCore metrics do not include `densecore_pending_requests`;
- DenseCloud metrics do not include `densecloud_http_requests_total`;
- request results do not complete cleanly;
- autoscale runs do not prove `1 -> 2 -> 1`, backlog remaining while two pods
  are ready, and second-pod request handling;
- autoscale runs in `all` mode do not show an improvement versus the recorded
  control evidence when that control evidence exists.

## Pinned defaults

The script pins these defaults and allows explicit overrides:

- KEDA chart: `2.18.0`
- `kube-prometheus-stack` chart: `78.5.0`
- DenseCloud chart dependency: `dense-base 1.1.0`
- KEDA limits: `minReplicaCount: 1`, `maxReplicaCount: 2`
- owned KEDA namespace/release defaults: `<namespace>-keda` / `<release>-keda`
- owned monitoring namespace/release defaults: `<namespace>-monitoring` /
  `<release>-prom`
- Prometheus service resolution: auto-discover exactly one service with
  `app.kubernetes.io/instance=<prom-release>` and
  `app.kubernetes.io/name=prometheus`

These are script defaults, not a claim that the current environment has already
qualified those exact releases end to end.

## Real metric requirement

Use a real product-owned PromQL query. The chart already documents the current
candidate metrics:

- `densecore_pending_requests`
- `densecore_active_requests`

Example:

```bash
bash scripts/qualify_keda_scaleout.sh autoscale \
  --helm-values charts/densecore/examples/values-gcp-c4-qwen36-35b.yaml \
  --metric-query 'sum(densecore_pending_requests{namespace="densecore-keda"})' \
  --metric-threshold 3 \
  --image-digest sha256:REPLACE_WITH_REAL_IMAGE_DIGEST \
  --model-name Qwen3.6-35B-A3B \
  --model-quantization Q4_K_M \
  --model-sha256 REPLACE_WITH_REAL_MODEL_SHA256 \
  --context-length 32768 \
  --densecore-max-prefill-seqs 2 \
  --output-dir release/keda-scaleout/manual-20260816
```

The query must be verified directly against Prometheus before the script will
continue.

The generated DenseCore `ServiceMonitor` label is also wired to the chosen
Prometheus Helm release rather than a hardcoded `prometheus` label.

The harness waits for a bounded scrape-settle window before failing a missing
Prometheus query result. It still fails closed if the metric never appears.

## Required real-run metadata

Dry-run may record `NOT_AVAILABLE`, but real `control`, `autoscale`, and `all`
runs fail before deployment unless the frozen metadata is complete.

Required explicit inputs:

- `--image-digest`
- `--model-name`
- `--model-quantization`
- `--model-sha256`
- `--context-length`

Required explicit input or reliable harvest from the values file:

- `THREADS`
- `DENSECORE_MAX_NUM_SEQS`
- `DENSECORE_MAX_PREFILL_SEQS`

The values file must also resolve the model PVC/path details actually used by
the workload.

## Result artifacts

By default the harness writes a timestamped result directory under
`release/keda-scaleout/`. Expected artifacts include:

- `results/*-manifest.json`
- `results/*-telemetry.jsonl`
- `results/*-telemetry.csv`
- `results/*-requests.jsonl`
- `results/*-request-summary.json`
- `results/*-assertions.json`
- `results/*-per-pod-densecore_total_requests.json`
- `results/*-runtime-metadata.json`
- `logs/*.log`

The JSONL telemetry rows include:

- requested/current/ready/available replicas;
- ScaledObject and HPA state;
- selected Prometheus metric value;
- per-pod readiness, restarts, creation timestamps, scheduling timestamps, and
  start timestamps.

The request log includes:

- request start time;
- HTTP status;
- total latency;
- TTFT when observed from streaming events;
- prompt/output token counts when returned by the API;
- stream terminal status and error text.

## Cleanup

Safe default cleanup removes only the harness-owned DenseCore release namespace:

```bash
bash scripts/qualify_keda_scaleout.sh cleanup --namespace densecore-keda
```

Cleanup refuses to run unless the target namespace carries the harness ownership
labels/annotations for the expected DenseCore release. This harness does not
delete shared clusters, shared model storage, or unrelated namespaces.

Owned observability cleanup is opt-in only:

```bash
bash scripts/qualify_keda_scaleout.sh cleanup \
  --namespace densecore-keda \
  --cleanup-observability
```

When enabled, the script deletes only harness-owned KEDA/monitoring releases
and namespaces after exact ownership verification. It never deletes shared
infra.

## Current status on 2026-08-16

Implementation status in this workspace:

- Harness implementation: PASS
- Shell syntax validation: PASS
- Help text validation: PASS
- Dry-run argument/path generation validation: PASS

Qualified product evidence in this workspace:

- Real KEDA install on a 35B-capable cluster: NOT RUN
- Real Prometheus scrape verification against a DenseCore workload: NOT RUN
- One-replica control experiment: NOT RUN
- `1 -> 2 -> 1` autoscaled experiment: NOT RUN
- Second-pod request-handling proof: NOT RUN

Environment note:

- A local Kubernetes context may exist, but this document does not count a
  placeholder or underprovisioned local cluster as DenseCore 35B product
  evidence.
- No qualified cloud node pool, model path, image digest, or measured 35B
  memory envelope was provided in this task.

## Commands to clear the blocker

At minimum, a real qualification run still needs:

1. A Kubernetes cluster with enough memory for two DenseCore replicas of the
   target model.
2. A real DenseCore image digest and model PVC/path in the Helm values file.
3. A real model SHA-256 and explicit frozen metadata inputs.
4. A real product-owned PromQL query verified against Prometheus.

Planning estimate only, not measured evidence:

- current example values request `56Gi` and limit `60Gi` per pod, so a practical
  planning floor is one DenseCore pod per node with at least `64Gi` allocatable
  memory and spare kube/system headroom;
- for two 35B MoE GGUF replicas, plan for at least two high-memory nodes;
- a conservative GKE planning example is two `n2-highmem-16` nodes, one pod per
  node, with a PVC mounted at `/models` and the model file available at
  `/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

Example planning command only, do not treat this as measured qualification:

```bash
gcloud container node-pools create densecore-35b \
  --cluster YOUR_CLUSTER \
  --region YOUR_REGION \
  --machine-type n2-highmem-16 \
  --num-nodes 2
```

Then run:

```bash
bash scripts/qualify_keda_scaleout.sh setup \
  --output-dir release/keda-scaleout/real-run

bash scripts/qualify_keda_scaleout.sh control \
  --helm-values charts/densecore/examples/values-gcp-c4-qwen36-35b.yaml \
  --metric-query 'sum(densecore_pending_requests{namespace="densecore-keda"})' \
  --metric-threshold 3 \
  --image-digest sha256:REPLACE_WITH_REAL_IMAGE_DIGEST \
  --model-name Qwen3.6-35B-A3B \
  --model-quantization Q4_K_M \
  --model-sha256 REPLACE_WITH_REAL_MODEL_SHA256 \
  --context-length 32768 \
  --densecore-max-prefill-seqs 2 \
  --output-dir release/keda-scaleout/real-run

bash scripts/qualify_keda_scaleout.sh autoscale \
  --helm-values charts/densecore/examples/values-gcp-c4-qwen36-35b.yaml \
  --metric-query 'sum(densecore_pending_requests{namespace="densecore-keda"})' \
  --metric-threshold 3 \
  --image-digest sha256:REPLACE_WITH_REAL_IMAGE_DIGEST \
  --model-name Qwen3.6-35B-A3B \
  --model-quantization Q4_K_M \
  --model-sha256 REPLACE_WITH_REAL_MODEL_SHA256 \
  --context-length 32768 \
  --densecore-max-prefill-seqs 2 \
  --output-dir release/keda-scaleout/real-run
```

Until those runs complete with real artifacts, DenseCore KEDA scale-out proof
remains pending.
