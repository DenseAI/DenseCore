# DenseCore on an existing CPU fleet

Use a pinned model and give each pod CPU and memory limits. A chart render is
configuration evidence; only a ready pod serving the real model proves the
runtime fits those limits. Arm and NUMA performance require measurements on the
actual hardware; they cannot be inferred from an amd64 fixture or an image build.

The [limited-pod example](../charts/densecore/examples/values-local-pod.yaml)
uses Qwen3.5-0.8B at a full revision and SHA-256, a checksum-verifying init
container, a shared model volume, an explicit 2048-token context and limits of
2 CPU / 4 GiB. This context is a test/deployment setting, not an automatic
production context reduction. It is a local image
example, not a published v0.2 image claim. Replace image repositories/tags and
pull policies with your own validated images in a real cluster.

```bash
docker build --build-arg DENSECORE_CPU_TARGET=portable -t densecore:local .
docker build -f docker/downloader/Dockerfile -t densecore-downloader:local docker/downloader
helm dependency build charts/densecore
helm template smoke charts/densecore -f charts/densecore/examples/values-local-pod.yaml
bash scripts/kind_model_smoke.sh
```

The kind harness creates and removes its own cluster using an isolated
kubeconfig in the artifact directory; it preserves your existing cluster context. It installs the actual
chart, runs the downloader, waits for ready, checks installed resource limits,
opens a real generation stream, deletes the pod while generating, and requires
that the accepted stream finishes while readiness and new admission disappear.
It preserves rendered manifests, pod state and drain output under its printed
artifact directory. Image builds, model downloads and kind have their own local
disk/RAM requirements; run native builds serially on small development machines.

For a persistent deployment use a PVC instead of `emptyDir`. The downloader
rehashes a cached file and refuses mismatched bytes; it never treats an existing
`main_model.gguf` alias as proof that the requested model is correct. A bad
checksum must fail the init container and prevent the pod becoming ready.
Supply `MODEL_REVISION` and `MODEL_SHA256` for custom artifacts. Store `HF_TOKEN`
in a Kubernetes Secret if the model requires access.

Leave `THREADS=0` for automatic CPU budgeting, or choose an explicit thread count
that fits your CPU allocation. Keep the effective context length explicit when
comparing deployments, and inspect native resource diagnostics before increasing
concurrency. Model bytes alone do not include KV, scratch, recurrent snapshots,
repack or allocator headroom.

DenseCore handles SIGTERM and drains accepted requests. The runtime image is
distroless and has no shell: `preStopSleepSeconds` must remain `0` with this image.
Set `terminationGracePeriodSeconds` longer than `SHUTDOWN_TIMEOUT`; the example
uses 40 seconds and 30 seconds. A preStop sleep cannot prove active drain. Test
with a request that is still generating when termination begins.

Portable amd64/arm64 and optimized amd64-v3/arm64 profiles have separate native
CI jobs. Portable manifests are assembled from tested image digests; optimized
profiles use `-amd64-v3` and `-arm64` suffixes. This configuration does not itself
mean those artifacts have been published or passed on the current revision.
