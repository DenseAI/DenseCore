# v0.2 launch fact sheet — draft

**Status: release readiness incomplete.** No v0.2 image, tag or release has been
published by this implementation task. This sheet is for preparing a later
launch; community post text remains the user's responsibility.

Intended deployments are a personal PC, an existing CPU Kubernetes pod and a
two-socket NUMA host. The candidate adds explicit resource budgeting, checked
model artifacts, metadata-driven embedding pooling and safer request/prefix
state ownership. Model-specific execution and platform qualification are tracked
in the [release gates](V0_2_RELEASE_GATES.md) and [model catalog](MODEL_CATALOG.md).

Use no new performance headline yet. Historical TTFT gains do not establish an
aggregate-throughput or completion-time lead. The MTP implementation currently
includes an explicit sequential reference, while its automatic profile registry
is empty. NUMA defaults remain unpromoted. The gpt-oss packed native execution
candidate has no current speed or PC RAM-support claim.

A publishable fact sheet must name the exact source and image digests, model
revision/SHA, tested CPU/RAM/context/concurrency, complete final-answer results,
matched performance artifacts and all unsupported combinations. Only report
required PC RAM with measured peak RSS and the corresponding context. Do not
replace unavailable Arm/NUMA/model evidence with CI configuration or fixtures.
