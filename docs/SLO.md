# DenseCore SLO Example

This document describes the example objectives shipped in
`prometheus/slo_rules.yml` and `prometheus/alerts.yml`. They are starting points,
not a universal product guarantee. Set targets from the deployed model, prompt
distribution, concurrency, and capacity plan.

## Scope

- service: DenseCore inference API
- measurement: Prometheus metrics from `/metrics`
- availability source: successful scrape plus application-level request metrics
- latency dimensions: TTFT, inter-token latency, and queue wait

## Shipped Objectives

| Signal | Example objective | Recording rule / metric |
| --- | --- | --- |
| availability | 99.9% monthly target | `up{job="densecore"}` plus request success |
| error rate | at most 1% under normal traffic | `densecore:slo_error_rate5m` |
| TTFT | p99 at most 2 seconds for the qualified workload | `densecore:slo_ttft_p99_seconds` |
| queue | fewer than 20 pending requests in steady state | `densecore_pending_requests` |

The checked-in critical error alert fires at 5% for five minutes with a minimum
request-rate guard. That alert threshold is intentionally looser than the 1%
objective; it is not the SLO itself.

The 2-second TTFT example is not realistic for every 26B/35B CPU workload,
especially long prompts. Replace it with workload-class objectives if the service
mix includes both short chat and long RAG requests.

## Recording Rules

The repository defines:

- `densecore:slo_request_rate5m`
- `densecore:slo_error_rate5m`
- `densecore:slo_ttft_p99_seconds`
- `densecore:slo_itl_p99_seconds`
- `densecore:slo_queue_wait_p99_seconds`

Verify that the running server exports the source metrics before enabling the
rules. A rule file loading successfully does not prove that a metric is populated.

## Error Budget Policy

For a 99.9% availability target:

- at 50% budget consumption, pause non-critical performance experiments and focus
  on correctness/capacity regressions
- at 100%, stop rollout until the failure mode and recovery are verified

Output-quality failures belong in the release gate even when HTTP availability is
green. Track empty output, repetition, template errors, and model fast-path
rejection outside the basic HTTP SLO.

## Qualification

Before adopting these objectives:

1. define prompt-length and generation-length classes
2. measure concurrency and queue behavior on the production machine type
3. set separate TTFT targets for short chat and long-context work
4. validate alert routing and [runbook](OPERATIONS_RUNBOOK.md) anchors
5. exercise rollback, model-load failure, OOM, and graceful shutdown
