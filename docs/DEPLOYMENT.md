# Deployment Guide

This guide covers the Go API server. The Python SDK is intended for in-process local
inference and has a separate packaging lifecycle.

## Docker

Run the published image with a local GGUF:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  denseai/densecore:latest
```

With authentication:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  -e AUTH_ENABLED=true \
  -e API_KEYS="sk-prod:user:default" \
  denseai/densecore:latest
```

Smoke-test the process and model lifecycle separately:

```bash
curl --fail http://localhost:8080/health/live
curl --fail http://localhost:8080/health/startup
curl --fail http://localhost:8080/health/ready
curl --fail http://localhost:8080/v1/models
```

Build the image from this repository with:

```bash
docker build -t denseai/densecore:local .
```

## Docker Compose

The checked-in `docker-compose.yml` starts DenseCore and an optional Redis service.
It mounts `./models` read-only and expects the configured model to exist there.

```bash
docker compose up -d
```

Redis can back distributed API-key storage and rate limiting. Treat Redis failure
behavior as a deployment decision: verify whether the selected configuration is
allowed to fall back to in-memory state before using multiple replicas.

## Kubernetes and Helm

The chart lives in `charts/densecore` and consumes the shared DenseCloud chart
dependency.

```bash
helm dependency update ./charts/densecore
helm upgrade --install densecore ./charts/densecore -f values.production.yaml
```

Example model PVC override:

```yaml
dense-base:
  image:
    repository: denseai/densecore
    tag: "latest"

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
- `/health/ready`: request readiness and runtime pressure
- `/metrics`: Prometheus scrape endpoint

Keep startup, readiness, and liveness distinct. The server uses the DenseCloud
runner for middleware ordering, RED metrics, signal handling, and graceful
shutdown.

### Optional Chart Features

The chart includes optional PDB, HPA/KEDA, ServiceMonitor, Grafana dashboard, and
NetworkPolicy surfaces. Their presence is not proof that a cluster integration is
qualified. Validate CRDs, Prometheus queries, ingress-controller namespaces, OTel
collector access, and model-storage behavior in the target cluster.

CPU inference is memory-heavy and often has long prefill latency. Autoscaling on
CPU utilization alone may react too late; queue metrics are useful, but scale-up
time and model-load time must be included in the policy.

## Model Loading

For predictable startup, mount one immutable GGUF and set `MAIN_MODEL_PATH`. Hot
loading is available when the process starts without a model:

```bash
curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model_path":"/models/model.gguf","threads":16}'
```

Unload with:

```bash
curl -X POST http://localhost:8080/v1/models/unload
```

Avoid using hot load as a rollout mechanism unless memory release, readiness, and
request draining have been tested for the exact model.

## Production Configuration

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
| `SHUTDOWN_TIMEOUT` | graceful shutdown deadline |
| `AUTH_ENABLED`, `API_KEYS` | local authentication |
| `REDIS_URL` | Redis endpoint for enabled integrations |
| `GRPC_ENABLED`, `GRPC_PORT` | optional gRPC service |
| `GRPC_TLS_ENABLED` | gRPC TLS |
| `METRICS_ENABLED`, `METRICS_PATH` | Prometheus endpoint |

Do not hard-code a universal KV or context budget. Model size, quantization,
context length, active sequences, and available host memory all affect the safe
envelope.

## Operational Checks

Before production traffic:

1. Run a real `/v1/chat/completions` short and long QA set.
2. Confirm the intended model fast path and absence of rejected fallback.
3. Load-test concurrency and record TTFT, inter-token latency, queue wait, memory
   peak, and output quality.
4. Verify graceful termination while requests are active.
5. Verify alert and dashboard queries against the emitted metric names.
6. Test prompt-cache affinity if repeated prefixes are part of the workload.

See [SLO.md](SLO.md) and [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for the
shipped example objectives and response procedures.
