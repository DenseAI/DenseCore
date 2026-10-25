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

## Read-only cleanup receipt

Record every session-created VM, disk (including boot disks) and reserved address
with its project, scope, name and numeric ID before cleanup. Keep those records
outside the repository with the session's raw evidence. After independently
deleting them, verify their absence with a new receipt:

```json
{
  "schema_version": 1,
  "project": "densecore-project",
  "resources": [
    {"kind": "instance", "scope": "zones/us-central1-a", "name": "session-vm", "id": "123"},
    {"kind": "disk", "scope": "zones/us-central1-a", "name": "session-boot", "id": "124"},
    {"kind": "address", "scope": "regions/us-central1", "name": "session-ip", "id": "125"}
  ]
}
```

```bash
python3 benchmarks/economics/verify_session_cleanup.py /tmp/session-resources.json \
  --output /tmp/session-cleanup.json
```

Before any provider call, the verifier locally checks that the installed SDK
recognizes and disables `core/should_prompt_to_enable_api` and
`compute/allow_partial_error`, forced through environment properties. Unknown or
enabled guards fail before inventory. This matters because SDK prompts can
accept their default on EOF; closed input alone does not prevent API enablement.
The provider commands only describe the project and list instances, disks and
addresses. The verifier never enables an API or changes a cloud resource.
An inactive/wrong project, failed command, warning, invalid inventory,
or remaining target produces a failed receipt. A same-name replacement in the
same scope also fails even when its numeric ID changed. Stopped VMs, unattached
disks and unused addresses still count as present.

The new receipt retains the manifest/verifier hashes and the exact commands,
stdout, stderr and their SHA-256 hashes (commands use compact JSON arrays).
Existing evidence files are never overwritten.
Unlisted resources are reported separately; a pass covers only the explicitly
listed targets, not an assertion that the project is empty or the final bill is
settled. Review that inventory for resources omitted after an interrupted run,
and retain pending liabilities in the cost ledger until billing settles.

The underlying list commands are unfiltered and unlimited. In particular,
[address listing includes global and regional addresses by default](https://docs.cloud.google.com/sdk/gcloud/reference/compute/addresses/list).
This verifier has local fake-command regression coverage; an actual GCP receipt
must still be obtained for each paid session.
