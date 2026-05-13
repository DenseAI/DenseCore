# DenseCore CLI

The CLI lives in the Go server module under [`server/`](../server). It is separate from the PyPI package.

## Build

From the repository root:

```bash
make lib
make server
```

This produces `bin/densecore-server`, which exposes both `serve` and `run`.

## Commands

### `densecore run`

`densecore run` starts a local TUI chat experience.

Behavior implemented in the current CLI:

1. checks `~/.densecore/models`
2. downloads a GGUF from Hugging Face if the file is missing
3. starts the local HTTP server on `127.0.0.1`
4. opens the interactive terminal chat UI

Examples:

```bash
./bin/densecore-server run
./bin/densecore-server run Qwen/Qwen2.5-0.5B-Instruct-GGUF
./bin/densecore-server run TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf
./bin/densecore-server run --port 9090
```

Notes:

- default model repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- default cache directory: `~/.densecore/models`
- default bind address for the local background server: `127.0.0.1`
- gRPC is disabled in `run` mode
- gated model downloads are not handled by the current downloader path

### `densecore serve`

`densecore serve` starts the production API server.

Examples:

```bash
./bin/densecore-server serve --model ./models/model.gguf
./bin/densecore-server serve --model ./models/model.gguf --port 9090
./bin/densecore-server serve --model ./models/model.gguf --grpc=false
AUTH_ENABLED=true API_KEYS=sk-example:user:default ./bin/densecore-server serve --model ./models/model.gguf
```

Flags currently defined by the CLI:

| Flag | Description |
| --- | --- |
| `--model`, `-m` | Path to a GGUF model file |
| `--threads`, `-t` | Inference threads (`0` = auto) |
| `--auth` | Enable API key authentication |
| `--grpc` | Enable or disable the gRPC server |
| `--grpc-port` | gRPC port, default `50051` |
| `--host` | HTTP bind address, default `0.0.0.0` |
| `--port` | HTTP port, default `8080` |
| `--verbose`, `-v` | Verbose logging |

## HTTP Endpoints

When `serve` is running, the HTTP server exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /completion` (compatibility alias)
- `POST /v1/embeddings`
- `POST /v1/rerank`
- `GET /v1/models`
- `POST /v1/models/load`
- `POST /v1/models/unload`
- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /health/startup`
- `GET /metrics`
- `GET /v1/runtime/profile`

## Environment Variables

The server reads configuration from environment variables. The commonly used ones are:

| Variable | Meaning | Default |
| --- | --- | --- |
| `MAIN_MODEL_PATH` | model to preload at startup | unset |
| `PORT` | HTTP port | `8080` |
| `HOST` | HTTP bind address | `0.0.0.0` |
| `THREADS` | inference threads | `0` |
| `AUTH_ENABLED` | enable API key auth | `false` |
| `API_KEYS` | comma-separated API keys | unset |
| `GRPC_ENABLED` | enable gRPC | `false` |
| `GRPC_PORT` | gRPC port | `50051` |
| `RATE_LIMIT_ENABLED` | enable rate limiting | `true` |
| `RATE_LIMIT_RPS` | request rate limit | `100` |
| `RATE_LIMIT_BURST` | rate limit burst | `200` |
| `LOG_FORMAT` | `json` or `text` | `json` |

## Operational Notes

- If no model is configured at startup, the server still starts and can later load a model through `POST /v1/models/load`.
- `GET /health/startup` returns `503` while the model is still loading or when no model is configured.
- `GET /health/ready` reflects model readiness and KV cache pressure.
