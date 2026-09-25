# Deployment Guide

This guide covers the Go API server. The Python SDK is intended for in-process local
inference and has a separate packaging lifecycle.

## Docker

The published `denseai/densecore:0.1.0` image targets `linux/amd64` only.
Arm64 remains available as a source build and C4A engineering target, without
a qualified v0.1.0 container image.

The release image uses a non-root distroless runtime without a shell, package
manager, or in-container model downloader. Mount an immutable GGUF read-only and
set `MAIN_MODEL_PATH`; use an external download step or Kubernetes init container
when the model is not already present. The image healthcheck calls the DenseCore
binary directly instead of bundling an HTTP client.

Build and run the image from source with a local GGUF:

```bash
docker build -t densecore:local .
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  densecore:local
```

With authentication:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  -e AUTH_ENABLED=true \
  -e API_KEYS="sk-prod:user:default" \
  densecore:local
```

Smoke-test the process and model lifecycle separately:

```bash
curl --fail http://localhost:8080/health/live
curl --fail http://localhost:8080/health/startup
curl --fail http://localhost:8080/health/ready
curl --fail http://localhost:8080/v1/models
```

The same liveness check is available inside the container without `curl`:

```bash
docker exec CONTAINER /app/densecore-server healthcheck \
  --url http://127.0.0.1:8080/health/live
```

Build the image from this repository with:

```bash
docker build -t denseai/densecore:local .
```

## Docker Compose

The checked-in `docker-compose.yml` is a loopback-only local development stack.
It starts DenseCore, builds the local image, adds Redis, mounts `./models`
read-only, and expects the configured model to exist there. It is not a
networked production deployment configuration.

```bash
docker compose up -d
```

Redis can back distributed API-key storage and rate limiting. Treat Redis failure
behavior as a deployment decision: verify whether the selected configuration is
allowed to fall back to in-memory state before using multiple replicas.

## Kubernetes and Helm

The chart lives in `charts/densecore` and consumes the shared DenseCloud chart
dependency declared in `charts/densecore/Chart.yaml`.

```bash
# First edit the example PVC, model filename, resources, and node selector for your cluster.
helm dependency update ./charts/densecore
helm upgrade --install densecore ./charts/densecore \
  -f ./charts/densecore/examples/values-cpu-inference.yaml
```

`Chart.lock` pins the public `dense-base` `1.1.0` OCI dependency from
`oci://ghcr.io/denseai/charts`. Generated chart archives under
`charts/densecore/charts/*.tgz` are intentionally ignored and should be
recreated with `helm dependency update` instead of committed.

Example model PVC override:

```yaml
dense-base:
  image:
    repository: denseai/densecore
    tag: "0.1.0"

  model:
    source: pvc
    existingClaim: densecore-models-pvc
    filename: model.gguf
```

Use [`charts/densecore/README.md`](../charts/densecore/README.md) as the chart
schema source of truth. Do not copy values from older docs without running
`helm lint` and rendering the chart.

### Probes and Shutdown

- `/health/live`: process liveness
- `/health/startup`: initial model lifecycle
- `/health/ready`: startup complete, not shutting down, model loaded, and optional dependency health
- `/metrics`: Prometheus scrape endpoint

Keep startup, readiness, and liveness distinct. The server uses the DenseCloud
runner for middleware ordering, RED metrics, signal handling, and graceful
shutdown. The live readiness route does not currently gate on queue pressure,
KV pressure, or autoscaling heuristics by itself.

### Optional Chart Features

The chart includes optional PDB, HPA/KEDA, ServiceMonitor, Grafana dashboard, and
NetworkPolicy surfaces. Their presence is not proof that a cluster integration is
qualified. Validate CRDs, Prometheus queries, ingress-controller namespaces, OTel
collector access, and model-storage behavior in the target cluster.

CPU inference is memory-heavy and often has long prefill latency. Autoscaling on
CPU utilization alone may react too late; queue metrics are useful, but scale-up
time and model-load time must be included in the policy.

## Model Loading

Mount one immutable GGUF and set `MAIN_MODEL_PATH`. The v0.1.0 server does not
expose dynamic load/unload; deploy a new process or workload revision to change
models so in-flight requests drain through the normal shutdown contract.

## Server Configuration

Important server variables include:

| Variable | Purpose |
| --- | --- |
| `MAIN_MODEL_PATH` | model loaded at startup |
| `DENSECORE_ENGINE_THREADS` | compute-thread override |
| `DENSECORE_MAX_NUM_SEQS` | active sequence bound |
| `DENSECORE_MAX_SEQ_LEN` | context override |
| `DENSECORE_KV_TARGET_MB` | KV memory override |
| `DENSECORE_SERVER_INFLIGHT` | server admission bound |
| `REQUEST_TIMEOUT` | request deadline |
| `SHUTDOWN_TIMEOUT` | graceful shutdown deadline used by the DenseCloud runner |
| `AUTH_ENABLED`, `API_KEYS` | local authentication |
| `REDIS_URL` | Redis endpoint for enabled integrations |
| `GRPC_ENABLED`, `GRPC_PORT` | optional gRPC service |
| `GRPC_TLS_ENABLED` | gRPC TLS |
| `METRICS_ENABLED`, `METRICS_PATH` | Prometheus endpoint |
| `CORS_ENABLED`, `CORS_ALLOWED_ORIGINS` | explicit browser-origin allowlist |

Do not hard-code a universal KV or context budget. Model size, quantization,
context length, active sequences, and available host memory all affect the safe
envelope.

## Operational Checks

Before treating the developer preview as qualified for a deployment:

1. Run a real `/v1/chat/completions` short and long QA set.
2. Confirm the intended model fast path and absence of rejected fallback.
3. Load-test concurrency and record TTFT, inter-token latency, queue wait, memory
   peak, and output quality.
4. Verify graceful termination while requests are active.
5. Verify alert and dashboard queries against the emitted metric names.
6. Test prompt-cache affinity if repeated prefixes are part of the workload.

See [SLO.md](SLO.md) and [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for the
shipped example objectives and response procedures.

## Public DenseCloud Qualification

Use the clean-room harness before treating a DenseCloud dependency change as
qualified:

```bash
bash scripts/qualify_densecloud_public.sh
```

The harness forces `GOWORK=off`, rejects `replace` directives and `file://`
chart dependencies, resolves `github.com/DenseAI/DenseCloud@v1.1.0`, pulls
`oci://ghcr.io/denseai/charts/dense-base:1.1.0` anonymously, rebuilds the
local chart dependency from OCI, and runs the DenseCore Go server tests against
the rebuilt native runtime from isolated caches.
