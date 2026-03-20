# Cloud-Native Deployment Guide

DenseCore is designed for cloud-native deployments with full Kubernetes, Helm, and Docker Compose support.

## Features

| Feature | Description |
|---------|-------------|
| **Helm Chart** | Production-ready chart with ServiceAccount, PDB, HPA, KEDA |
| **Redis Integration** | Distributed rate limiting and API key storage |
| **Prometheus Metrics** | ServiceMonitor for Prometheus Operator |
| **Network Policy** | Pod network isolation |
| **Graceful Shutdown** | preStop hooks and terminationGracePeriodSeconds |

## Quick Start

### Docker Compose (with Redis)

```bash
docker-compose up -d
```

This starts:
- DenseCore server on port 8080
- Redis for distributed rate limiting

### Helm

```bash
helm install densecore ./charts/densecore \
  --set model.repoId=Qwen/Qwen3-0.6B-GGUF \
  --set model.filename=qwen3-0.6b-q4_k_m.gguf
```

## Redis Configuration

Redis enables distributed rate limiting and API key storage for multi-replica deployments.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | - | Redis connection URL (e.g., `redis://redis:6379`) |
| `REDIS_PASSWORD` | - | Redis password (optional) |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_RATELIMIT_ENABLED` | `false` | Enable Redis-based rate limiting |
| `REDIS_KEYSTORE_ENABLED` | `false` | Enable Redis-based API key storage |
| `REDIS_KEYSTORE_CACHE_TTL` | `5m` | Local LRU cache TTL |
| `REDIS_KEYSTORE_CACHE_SIZE` | `1000` | Local LRU cache size |

### Fallback Behavior

If Redis is unavailable or connection fails:
- Server logs a warning and falls back to in-memory stores
- Circuit breaker prevents repeated connection attempts
- Automatic recovery when Redis becomes available

## Helm Chart Values

### Cloud-Native Features

```yaml
# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1

# ServiceMonitor for Prometheus Operator
serviceMonitor:
  enabled: true
  interval: 15s

# Network Policy
networkPolicy:
  enabled: true
  allowIngressController: true

# KEDA Autoscaling (queue-based)
keda:
  enabled: true
  minReplicaCount: 1
  maxReplicaCount: 10
  prometheusServerAddress: "http://prometheus-server:80"

# Redis Configuration
redis:
  enabled: true
  url: "redis://redis:6379"
```

### Topology Spread

```yaml
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
```

## Security Features

### API Key Storage

- Keys stored as SHA-256 hashes (never plaintext)
- Constant-time comparison to prevent timing attacks
- LRU cache reduces Redis load

### Rate Limiting

- Token bucket algorithm via Lua script (atomic)
- Per-key rate limiting support
- Circuit breaker for Redis failures

## Monitoring

### Prometheus Metrics

Enable ServiceMonitor:
```yaml
serviceMonitor:
  enabled: true
  labels:
    release: prometheus
```

Available metrics:
- `densecore_requests_total` - Total requests
- `densecore_pending_requests` - Pending queue size
- `densecore_active_requests` - Active requests
- `densecore_tokens_generated` - Total tokens generated

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health/live` | Liveness probe |
| `/health/ready` | Readiness probe |
| `/health/startup` | Startup probe |
| `/metrics` | Prometheus metrics |

## KEDA Autoscaling

KEDA enables scaling based on pending requests instead of CPU (leading vs. lagging indicator).

Install KEDA:
```bash
helm install keda kedacore/keda -n keda --create-namespace
```

Enable in values:
```yaml
keda:
  enabled: true
  prometheusServerAddress: "http://prometheus-server.monitoring:80"
  pendingRequestsThreshold: "5"
```
