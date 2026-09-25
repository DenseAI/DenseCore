# DenseCore Helm Chart

This chart wraps the DenseAI `dense-base` library chart and supplies DenseCore
server defaults. The chart is intentionally model, cloud, and node-type neutral:
SREs should choose the model source, resource shape, scheduling rules, network
policy, and scaling signal for their own cluster.

## Install

From source:

```bash
# First edit the example PVC, model filename, resources, and node selector for your cluster.
helm dependency update ./charts/densecore
helm upgrade --install densecore ./charts/densecore \
  -f ./charts/densecore/examples/values-cpu-inference.yaml
```

The chart depends on the public DenseCloud `dense-base` `1.1.0` OCI chart from
`oci://ghcr.io/denseai/charts`. `Chart.lock` pins the dependency digest, while
`charts/densecore/charts/*.tgz` stays ignored so qualification cannot be faked
by a committed archive.

## What The Chart Deploys

- DenseCore API server
- HTTP service on port `8080`
- optional gRPC service on port `50051`
- optional model init container
- startup, readiness, and liveness probes
- optional ServiceMonitor, KEDA ScaledObject, NetworkPolicy, PDB, and Grafana dashboards

Most platform settings live under `dense-base.*`.

## SRE Override Surface

Use these values instead of editing templates:

- `dense-base.image.*`: server image repository, tag, and pull policy.
- `dense-base.model.*`: model volume source, path, filename, and env var name.
- `dense-base.initContainers`: model downloader or any product-specific bootstrap.
- `dense-base.resources`: CPU and memory requests/limits.
- `dense-base.nodeSelector`, `affinity`, `tolerations`, `topologySpreadConstraints`: provider-specific scheduling.
- `dense-base.serviceMonitor.*`: Prometheus Operator scraping.
- `dense-base.keda.triggers.custom`: inference-aware autoscaling signals.
- `dense-base.networkPolicy.*`: explicit ingress and egress policy.
- `dense-base.extraEnv`, `extraEnvFrom`, `extraVolumes`, `extraVolumeMounts`: cluster-specific extension points.

Top-level `autoscaling` and `networkPolicy` are legacy helpers kept for
compatibility. Prefer `dense-base.keda` and `dense-base.networkPolicy` for new
production installs.

The checked-in defaults do not download or mount a model. A deployment becomes
ready only after the operator supplies `dense-base.model.*` and any required
volumes or init containers. This prevents an implicit demo model from becoming
the release server's startup model.

## Generic CPU Inference Override

This example does not assume a cloud provider or model family. Replace the PVC,
filename, resource shape, and scheduling labels with cluster-local values.

```yaml
dense-base:
  image:
    repository: denseai/densecore
    tag: "0.1.0"

  model:
    source: pvc
    existingClaim: densecore-models
    path: /models
    filename: model.gguf
    envVarName: MAIN_MODEL_PATH

  initContainers: []

  resources:
    requests:
      cpu: "16"
      memory: "64Gi"
    limits:
      cpu: "16"
      memory: "64Gi"

  extraEnv:
    - name: THREADS
      value: "16"
    - name: DENSECORE_MAX_NUM_SEQS
      value: "4"

  nodeSelector:
    inference.dense.ai/cpu: "true"

  tolerations: []
  affinity: {}
```

## C4/C4A LLM MVP Profiles

The chart also ships reference values for the current CPU-first LLM MVP target:
one model per pod, model weights mounted from a PVC, DenseCloud health/metrics
contracts enabled, and KEDA driven by DenseCore inference metrics.

| Profile | Use when | Values file |
| --- | --- | --- |
| GCP C4 Qwen3.6 35B | x86 C4 16-core Qwen3.6-35B-A3B serving | `examples/values-gcp-c4-qwen36-35b.yaml` |
| GCP C4A Qwen3.6 35B | ARM C4A 16-core Qwen3.6-35B-A3B serving | `examples/values-gcp-c4a-qwen36-35b.yaml` |
| GCP C4A Gemma4 26B | ARM C4A 16-core Gemma4-26B-A4B serving | `examples/values-gcp-c4a-gemma4-26b.yaml` |

These files are not universal production defaults. SREs should still replace
PVC names, image tags, node labels, ServiceMonitor labels, ingress policy, and
KEDA thresholds with cluster-local values. They are intended to make the
DenseCloud + DenseCore contract concrete enough to render and review before
cluster-specific edits.

The published v0.1.0 image targets `linux/amd64` only. The C4A values
files therefore require an operator-built Arm64 image that has passed native
model, HTTP, and graceful-shutdown qualification. Replace the image repository
and tag in every example with an artifact verified for the target platform.

Validate the source checkout against the public OCI dependency:

```bash
helm dependency update ./charts/densecore
helm lint ./charts/densecore
helm template densecore ./charts/densecore -f ./charts/densecore/examples/values-cpu-inference.yaml
```

For a full clean-room check from isolated caches, run
`bash scripts/qualify_densecloud_public.sh`.

## Autoscaling

For LLM inference, queue depth and active requests are usually better scaling
signals than raw CPU utilization. Use KEDA when Prometheus is available:

```yaml
autoscaling:
  enabled: false

dense-base:
  keda:
    enabled: true
    minReplicaCount: 1
    maxReplicaCount: 10
    triggers:
      custom:
        - type: prometheus
          metadata:
            serverAddress: http://prometheus-server.monitoring.svc.cluster.local:80
            metricName: densecore_pending_requests
            threshold: "5"
            query: sum(densecore_pending_requests)
        - type: prometheus
          metadata:
            serverAddress: http://prometheus-server.monitoring.svc.cluster.local:80
            metricName: densecore_active_requests
            threshold: "8"
            query: sum(densecore_active_requests)
```

Do not enable top-level HPA and `dense-base.keda.enabled` at the same time.
Avoid hiding missing scrape data with fallback PromQL such as `or vector(0)` in
production autoscaling signals.

## Prompt Cache Affinity

DenseCore prefix/KV cache is local to each pod. Multi-replica Kubernetes
serving should route repeated conversation or document prefixes back to the
same pod instead of using plain round-robin.

Clients can send a stable key with either:

- `X-DenseCore-Cache-Affinity`
- `cache_control.affinity_key`
- `cache_control.cache_id`
- `cache_control.conversation_id`

DenseCore returns the hashed key in `X-DenseCore-Cache-Affinity-Key` for client
replay. Configure the ingress or service mesh to hash on
`X-DenseCore-Cache-Affinity` before forwarding to the service. For NGINX
Ingress:

```yaml
dense-base:
  ingress:
    enabled: true
    annotations:
      nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_densecore_cache_affinity"
```

Envoy/Istio deployments should use an equivalent route hash policy on the same
header. This preserves pod-local cache hits without moving KV blocks through an
external cache.

## Monitoring

DenseCore exposes DenseCloud HTTP RED metrics and DenseCore inference metrics on
the same `/metrics` endpoint. Enable a ServiceMonitor when Prometheus Operator is
installed:

```yaml
dense-base:
  serviceMonitor:
    enabled: true
    labels:
      release: prometheus
    path: /metrics
```

Optional Grafana dashboard ConfigMaps can be enabled separately:

```yaml
grafana:
  dashboards:
    enabled: true
```

## Network Policy

Prefer the DenseCloud shared NetworkPolicy shape for new installs:

```yaml
networkPolicy:
  enabled: false

dense-base:
  networkPolicy:
    enabled: true
    ingress:
      enabled: true
      allowAll: false
      allowSameNamespace: false
      peers:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
    egress:
      enabled: true
      allowDNS: true
      allowAll: false
      peers: []
      ports: []
```

## Provider-Specific Scheduling

Cloud and on-prem labels differ. Keep provider-specific placement in values:

```yaml
dense-base:
  nodeSelector:
    cloud.google.com/machine-family: c4

  tolerations:
    - key: dedicated
      operator: Equal
      value: inference
      effect: NoSchedule
```

Use the equivalent labels for AWS, Azure, bare metal, or private Kubernetes.

## Validation

Chart rendering only proves Kubernetes resources are valid. API readiness needs
separate server checks:

```bash
curl --fail http://127.0.0.1:8080/health/live
curl --fail http://127.0.0.1:8080/v1/models
```

After loading a generation model, include chat and completion checks:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

If `/v1/embeddings` or `/v1/rerank` are part of the exposed product surface,
validate them with the matching model class and representative application inputs.
Generation-model chat checks do not prove embedding or rerank correctness.
