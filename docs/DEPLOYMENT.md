# Deployment Guide

This guide covers the production API server, not the Python SDK.

## Docker Hub Image

The published server image is:

```bash
docker pull densecore/densecore:latest
```

The container expects `MAIN_MODEL_PATH` if you want a model loaded on startup.

### Run with a local model

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  densecore/densecore:latest
```

### Run with authentication

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/models:/models:ro" \
  -e MAIN_MODEL_PATH=/models/model.gguf \
  -e AUTH_ENABLED=true \
  -e API_KEYS="sk-prod:user:default" \
  densecore/densecore:latest
```

### Health and smoke checks

```bash
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
curl http://localhost:8080/v1/models
```

### Chat completion example

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64
  }'
```

## Build Your Own Image

```bash
docker build -t densecore/densecore:latest .
```

The root [`Dockerfile`](../Dockerfile) builds:

- the native C++ library
- the Go API server binary
- a Debian runtime image exposing port `8080`

The runtime image sets:

- `PORT=8080`
- `HOST=0.0.0.0`
- `THREADS=0`
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_RPS=100`

## Docker Compose

[`../docker-compose.yml`](../docker-compose.yml) starts:

- `densecore` API server
- `redis` for distributed rate limiting and API key storage

Start it with:

```bash
docker compose up -d
```

Important notes about the checked-in compose file:

- it mounts `./models:/models:ro`
- it expects `MAIN_MODEL_PATH=/models/model.gguf`
- Redis-backed rate limiting and keystore are enabled in the example configuration

## Kubernetes

The repository ships a Helm chart in [`../charts/densecore`](../charts/densecore).

Install from source:

```bash
helm dependency update ./charts/densecore
helm install densecore ./charts/densecore
```

Common override pattern:

```yaml
dense-base:
  image:
    repository: densecore/densecore
    tag: "latest"

  model:
    source: pvc
    existingClaim: densecore-models-pvc
    filename: main_model.gguf
```

Apply:

```bash
helm upgrade --install densecore ./charts/densecore -f my-values.yaml
```

### Ports

- HTTP service: `8080`
- gRPC service: `50051` when enabled

### Probes

The server exposes:

- `/health/live`
- `/health/ready`
- `/health/startup`

These are already wired into the Helm values.

## Hot Loading

If you start the server without `MAIN_MODEL_PATH`, you can load a model later:

```bash
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/model.gguf",
    "threads": 4
  }'
```

Unload:

```bash
curl -X POST http://localhost:8080/v1/models/unload
```

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `MAIN_MODEL_PATH` | preload model path |
| `DRAFT_MODEL_PATH` | optional draft model path |
| `PORT` | HTTP port |
| `HOST` | HTTP bind address |
| `THREADS` | inference threads |
| `AUTH_ENABLED` | enable API key auth |
| `API_KEYS` | in-memory API key list |
| `GRPC_ENABLED` | enable gRPC server |
| `GRPC_PORT` | gRPC port |
| `GRPC_TLS_ENABLED` | enable gRPC TLS |
| `RATE_LIMIT_ENABLED` | enable HTTP rate limiting |
| `RATE_LIMIT_RPS` | HTTP rate limit |
| `RATE_LIMIT_BURST` | HTTP rate limit burst |
| `REDIS_URL` | Redis endpoint |
| `REDIS_KEYSTORE_ENABLED` | Redis-backed key store |
| `REDIS_RATELIMIT_ENABLED` | Redis-backed rate limiting |
| `METRICS_ENABLED` | Prometheus endpoint toggle |
| `METRICS_PATH` | metrics path |

## Production Notes

- The API server starts before background model loading completes.
- `startup` and `ready` probes intentionally reflect model load state.
- If authentication is enabled without `API_KEYS` and without a Redis keystore, startup fails.
