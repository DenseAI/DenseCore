# Read-only cloud session cost preflight

`gcp_preflight.sh` checks billing, existing running instances, disks and reserved
addresses. With no arguments it reports **inventory only**, not a price check.
It never enables APIs or billing or creates/deletes resources.

Before an approved session, obtain current USD rates for the exact region,
machine type, provisioning model, disk and network from official GCP pricing or
the Cloud Billing Catalog API. Save the quote/source and time, then supply a JSON
configuration to `--estimate-config`. The calculator refuses stale (>24h),
unofficial or invalid rates, missing phases, budget overruns and spending the
$10 reserve. Pending/unreported liabilities must be included explicitly.

```json
{
  "currency": "USD",
  "prices_checked_at": "<current ISO-8601 UTC time>",
  "price_source": "https://cloud.google.com/compute/all-pricing",
  "minutes": {"startup": 5, "download": 10, "build": 10, "benchmark": 20, "cleanup": 5},
  "compute_usd_per_hour": 0,
  "disk_gb": 100,
  "disk_usd_per_gb_hour": 0,
  "transfer_gb": 0,
  "transfer_usd_per_gb": 0,
  "spent_usd": 0,
  "pending_usd": 0,
  "session_cap_usd": 8
}
```

Zero placeholders above are not quoted prices. Replace them and retain the
original price evidence. The estimator cannot independently verify supplied
price content. Rate freshness and an official URL are necessary but insufficient
for an actual billing guarantee. Session caps in the approved plan are $8/$8/
$12/$12; total spend plus pending liabilities must leave $10 reserved.

```bash
bash benchmarks/economics/gcp_preflight.sh --estimate-config /tmp/session.json
```

Retain the estimate, enforce its printed UTC deadline, and keep a cost ledger.
After result retrieval independently list VM, disks and reserved addresses and
verify deletion, including resources created before an interrupted run. An
unavailable API or disabled billing is a blocked cloud prerequisite, not a reason
to auto-enable services or substitute local fixtures for cloud evidence.
