# DenseCore model downloader

The Helm init container downloads one pinned Hugging Face file into `/models`.
It verifies the bytes before atomically publishing `/models/main_model.gguf`.
Artifacts use a SHA-addressed cache under `/models/.artifacts/` so different
revisions cannot overwrite each other. Every cache hit is rehashed, including
legacy files already present on a shared PVC. Legacy files are copied only after
verification and preserved for existing readers.
A wrong checksum or a mutable/missing revision fails initialization.

```bash
docker build -f docker/downloader/Dockerfile -t densecore-downloader:local docker/downloader
```

Inputs:

- `MODEL_REPO` and `MODEL_FILE`: repository and relative repository filename.
- `MODEL_REVISION`: full immutable 40-character repository commit.
- `MODEL_SHA256`: expected 64-character SHA-256, also checked on existing cache files.
- `HF_TOKEN`: optional secret for gated/private repositories.

With no overrides the downloader uses the same Qwen3.5-0.8B revision/checksum as
[the CLI and quick start](../../docs/MODEL_CATALOG.md). Custom repo/file pairs
must supply both revision and checksum. A mismatched existing cache fails rather
than overwriting shared data. Remove or repair that specific bad file and retry.
The downloader preserves the previous verified alias while a new download is
staged; an unsuccessful init must not be bypassed by manually starting the server.
