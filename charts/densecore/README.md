# DenseCore Helm Chart

This chart wraps `dense-base` and supplies DenseCore-specific defaults for the API server.

## Install

From source:

```bash
helm dependency update ./charts/densecore
helm install densecore ./charts/densecore
```

## What The Chart Deploys

- DenseCore API server
- HTTP service on port `8080`
- optional gRPC service on port `50051`
- model init container by default
- readiness, liveness, and startup probes

Most runtime settings live under `dense-base.*`.

## Important Defaults

The checked-in values currently default to:

- server image: `densecore/densecore`
- downloader image: `densecore/downloader:0.3.0`
- HTTP port: `8080`
- gRPC enabled: `true`
- init-container model repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- init-container model file: `qwen2.5-0.5b-instruct-q4_k_m.gguf`

## Common Overrides

```yaml
dense-base:
  image:
    repository: densecore/densecore
    tag: "latest"

  model:
    source: pvc
    existingClaim: densecore-models-pvc
    filename: main_model.gguf

  env:
    - name: MAIN_MODEL_PATH
      value: /models/main_model.gguf
    - name: GRPC_ENABLED
      value: "true"
```

Apply:

```bash
helm upgrade --install densecore ./charts/densecore -f my-values.yaml
```

## Autoscaling

HPA:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
```

KEDA through `dense-base`:

```yaml
autoscaling:
  enabled: false

dense-base:
  keda:
    enabled: true
```

Do not enable both HPA and KEDA at the same time.

## Notes

- The chart uses `dense-base` as the main platform layer.
- The model downloader creates `/models/main_model.gguf`.
- For the full value surface, inspect [`values.yaml`](values.yaml).
