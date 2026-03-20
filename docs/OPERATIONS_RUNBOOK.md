# DenseCore Operations Runbook

This runbook is the first-response guide for DenseCore production incidents.

## 0. Quick Triage Checklist

1. Confirm impact: API 5xx, latency spike, or total outage
2. Check deployment health:
- `kubectl get pods -l app.kubernetes.io/name=densecore`
- `kubectl describe deploy <release>-densecore`
3. Check service endpoints:
- `kubectl get endpoints <release>-densecore`
4. Check core metrics:
- `densecore_pending_requests`
- `densecore_failed_requests`
- `densecore_time_to_first_token_seconds{quantile="0.99"}`

## 1. DenseCoreDown

Symptoms:
- Alert `DenseCoreDown`
- `up{job="densecore"} == 0`

Actions:
1. Verify pod status and recent events.
2. Inspect container logs and probe failures.
3. If rollout is stuck, rollback to previous chart release.
4. If nodes are unhealthy, reschedule by cordon/drain strategy.

## 2. DenseCoreHighErrorRate

Symptoms:
- Alert `DenseCoreHighErrorRate`

Actions:
1. Check recent config/rollout changes.
2. Split errors by endpoint (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`).
3. Verify model loading state (`/health/startup`, `/health/ready`).
4. Check Redis availability if distributed auth/rate-limit is enabled.
5. Scale up if queue is also increasing.

## 3. DenseCoreTTFTP99High

Symptoms:
- Alert `DenseCoreTTFTP99High`

Actions:
1. Check pending queue and active requests for saturation.
2. Validate CPU throttling and thread settings (`THREADS`, `GOMAXPROCS`, `OMP_NUM_THREADS`).
3. Check if noisy neighbors or resource limits changed.
4. Scale out via KEDA/HPA as temporary mitigation.

## 4. DenseCoreQueueBacklogHigh

Symptoms:
- Alert `DenseCoreQueueBacklogHigh`

Actions:
1. Confirm autoscaler events (`kubectl describe hpa` or `kubectl describe scaledobject`).
2. If autoscaler not reacting, verify Prometheus query and KEDA trigger status.
3. Temporarily scale deployment manually.
4. Investigate upstream traffic spikes and rate-limit policy.

## 5. DenseCoreQueueBacklogCritical

Symptoms:
- Alert `DenseCoreQueueBacklogCritical`

Actions:
1. Immediate manual scale-out.
2. Enforce stricter request shaping/rate limit.
3. Evaluate model size and per-request token limits.
4. If capacity exhausted, activate degraded mode policy.

## 6. DenseCoreKVCachePressure

Symptoms:
- Alert `DenseCoreKVCachePressure`
- Readiness may degrade

Actions:
1. Reduce max token limits for incoming workloads.
2. Increase replica count to reduce per-pod pressure.
3. If persistent, reload with smaller model or adjust workload mix.

## 7. DenseCoreTimeoutErrorsBurst

Symptoms:
- Alert `DenseCoreTimeoutErrorsBurst`

Actions:
1. Check `REQUEST_TIMEOUT`, ingress timeout, and downstream latency.
2. Verify queue backlog and TTFT concurrently.
3. Tune timeout budget and scaling policy together (not independently).

## 8. DenseCoreOOMDetected

Symptoms:
- Alert `DenseCoreOOMDetected`

Actions:
1. Confirm OOMKilled in pod status/events.
2. Increase memory limit/request or use smaller model.
3. Reduce concurrency and max tokens as immediate mitigation.
4. Validate rolling restart stability after mitigation.

## 9. Recovery Validation

After mitigation, validate:

1. `/health/live`, `/health/ready`, `/health/startup` all stable
2. Error rate normalized
3. Queue depth recovered below warning threshold
4. TTFT p99 back within SLO
5. No ongoing OOM/timeout counter spikes
