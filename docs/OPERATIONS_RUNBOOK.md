# DenseCore Operations Runbook

This runbook matches the alert names in `prometheus/alerts.yml`. Adapt commands,
release names, and escalation policy to the deployment.

## Quick Triage

1. Confirm whether impact is process availability, model readiness, errors, queue
   delay, or bad inference output.
2. Check pods, events, endpoints, and the active rollout.
3. Query `/health/live`, `/health/startup`, `/health/ready`, and `/v1/models`
   separately.
4. Inspect the model-load log, OOM events, request queue, and recent config changes.
5. Run one known-good short request before declaring recovery.

```bash
kubectl get pods -l app.kubernetes.io/name=densecore
kubectl get events --sort-by=.lastTimestamp
kubectl describe deploy <release>
kubectl logs deploy/<release> --tail=300
```

## 1. DenseCoreDown

Trigger: `up{job="densecore"} == 0` for one minute.

1. Distinguish scrape/network failure from process failure.
2. Inspect pod state, restart count, probes, Service endpoints, and NetworkPolicy.
3. If the current rollout introduced the failure, roll back to the last qualified
   image and model pair.
4. After recovery, verify metrics scraping and a real chat request.

## 2. DenseCoreHighErrorRate

Trigger: `densecore:slo_error_rate5m > 0.05` with request rate above `0.2/s` for
five minutes.

1. Split failures by endpoint and status code.
2. Check model readiness and fast-path rejection logs.
3. Check authentication/rate-limit stores if they are enabled.
4. Correlate with queue, OOM, timeout, and rollout changes.
5. Do not treat HTTP 200 as recovery until output quality passes a known request.

## 3. DenseCoreTTFTP99High

Trigger: `densecore:slo_ttft_p99_seconds > 2` for ten minutes.

1. Split short-chat and long-context traffic; the shared threshold may be invalid
   for a mixed workload.
2. Inspect queue wait separately from model prefill time.
3. Check CPU throttling, affinity, thread count, prompt-cache locality, and noisy
   neighbors.
4. Compare current model/binary/config hashes with the qualified deployment.
5. Scale only after confirming that model-load time and cache cold starts will not
   worsen the incident.

## 4. DenseCoreQueueBacklogHigh

Trigger: pending requests above 20 for five minutes.

1. Confirm arrival rate, service rate, active sequences, and prefill serialization.
2. Check HPA/KEDA events and the Prometheus query used by the scaler.
3. Apply temporary request shaping or a bounded replica increase.
4. Revalidate p95 TTFT and output quality after the queue drains.

## 5. DenseCoreQueueBacklogCritical

Trigger: pending requests above 100 for two minutes.

1. Protect the service with admission limits and rate limiting.
2. Stop unbounded retries from upstream clients.
3. Add qualified capacity or shed non-critical long-context traffic.
4. Preserve enough logs and metrics to identify whether the owner was arrival
   rate, model regression, memory pressure, or failed autoscaling.

## 6. DenseCoreKVCachePressure

Trigger: KV usage above 90% for five minutes.

1. Check active sequences, context distribution, KV dtype, and configured budget.
2. Reduce new long-context admission or generation limits as a temporary measure.
3. Prefer a model/host-derived capacity change over an arbitrary KV increase.
4. Verify readiness recovery and run long QA after any reload.

## 7. DenseCoreTimeoutErrorsBurst

Trigger: more than ten timeout errors in ten minutes.

1. Compare request, ingress, load-balancer, and client timeout budgets.
2. Inspect queue wait, TTFT, and decode latency together.
3. Check whether shutdown or model loading interrupted requests.
4. Tune capacity and deadlines as one system; increasing a timeout alone can hide
   saturation.

## 8. DenseCoreOOMDetected

Trigger: any OOM counter increase in fifteen minutes.

1. Confirm container OOMKill versus an application allocation failure.
2. Record model hash, resident memory, KV budget, graph pool, context length, and
   active sequences.
3. Reduce admission or use a smaller/safer model while restoring service.
4. Do not simply raise the limit without verifying node capacity and eviction risk.
5. After restart, run short QA, long QA, and a bounded concurrency smoke test.

## Recovery Gate

Recovery requires all of the following:

- stable liveness, startup, and readiness
- normalized error and queue metrics
- no continuing OOM or timeout increase
- one deterministic short response and one representative long response
- intended model fast path with no forbidden fallback
- stable behavior through at least one termination/replacement cycle when rollout
  or node health was involved
