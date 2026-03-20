# DenseCore SLO

This document defines service level objectives for production DenseCore clusters.

## 1. Scope

- Service: DenseCore inference API (`/v1/*`)
- Environments: production Kubernetes clusters
- Measurement source: Prometheus metrics scraped from `/metrics`

## 2. SLO Targets

1. Availability SLO
- Objective: `99.9%` monthly availability
- Indicator: `up{job="densecore"}`

2. Error Rate SLO
- Objective: `<= 1%` 5m rolling error rate under normal traffic
- Indicator:
  - Numerator: `rate(densecore_failed_requests[5m])`
  - Denominator: `rate(densecore_total_requests[5m])`

3. Latency SLO (TTFT)
- Objective: `p99 <= 2s` for time-to-first-token
- Indicator: `densecore_time_to_first_token_seconds{quantile="0.99"}`

4. Queue SLO
- Objective: pending requests should remain `< 20` for steady-state workloads
- Indicator: `densecore_pending_requests`

## 3. Error Budget Policy

- Monthly error budget for availability: `0.1%`
- When 50% of monthly budget is consumed:
  - freeze non-critical feature releases
  - prioritize reliability fixes and scaling checks
- When 100% is consumed:
  - halt production feature rollout until corrective actions complete

## 4. Alert Mapping

- `DenseCoreDown` -> availability objective
- `DenseCoreHighErrorRate` -> error-rate objective
- `DenseCoreTTFTP99High` -> latency objective
- `DenseCoreQueueBacklogHigh`/`Critical` -> queue objective

See `docs/OPERATIONS_RUNBOOK.md` for response procedures.
