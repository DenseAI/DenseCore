# DenseCore Downloader Image

This image is used by the Helm init container to place a model into `/models`.

## Build

```bash
docker build -f docker/downloader/Dockerfile -t densecore/downloader:latest docker/downloader
```

## Runtime Inputs

- `MODEL_REPO`
- `MODEL_FILE`
- `HF_TOKEN` optional for gated or private repositories

The downloader writes the requested file under `/models` and creates the `/models/main_model.gguf` symlink expected by the chart defaults.
