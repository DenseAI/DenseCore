# DenseCore CLI

The Go CLI lives under `server/` and is not installed by the `densecore` Python
package.

## Build

From the repository root:

```bash
make lib
make server
```

The output binary is `bin/densecore-server`.

## Local Chat

`run` starts a loopback HTTP server and terminal chat client. It caches downloaded
models under `~/.densecore/models`.

```bash
./bin/densecore-server run
./bin/densecore-server run Qwen/Qwen2.5-0.5B-Instruct-GGUF
./bin/densecore-server run OWNER/REPO --filename model.Q4_K_M.gguf
./bin/densecore-server run --port 9090
```

The default repository is `Qwen/Qwen2.5-0.5B-Instruct-GGUF`. gRPC is disabled in
`run` mode. The built-in downloader does not implement every gated-repository
authentication flow.

## API Server

```bash
./bin/densecore-server serve --model ./models/model.gguf
./bin/densecore-server serve --model ./models/model.gguf --threads 16 --port 9090
./bin/densecore-server serve --model ./models/model.gguf --grpc
./bin/densecore-server serve --host 0.0.0.0 --auth --model ./models/model.gguf
AUTH_ENABLED=true API_KEYS=sk-example:user:default \
  ./bin/densecore-server serve --model ./models/model.gguf
```

## Health Check

`healthcheck` performs one HTTP GET and exits non-zero for connection errors,
timeouts, or non-2xx responses. The distroless container uses this command for
its built-in Docker healthcheck.

```bash
./bin/densecore-server healthcheck \
  --url http://127.0.0.1:8080/health/live \
  --timeout 5s
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | unset | GGUF model path |
| `--threads`, `-t` | `0` | shared model-load and engine thread hint; zero selects runtime auto policy |
| `--auth` | `false` | enable API-key authentication |
| `--grpc` | `false` | enable the gRPC service |
| `--grpc-port` | `50051` | gRPC port |
| `--host` | `127.0.0.1` | HTTP bind address for direct CLI runs |
| `--port` | `8080` | HTTP port |
| `--verbose`, `-v` | `false` | verbose logging |

## Common Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAIN_MODEL_PATH` | unset | startup model path |
| `THREADS` | `0` | model-load thread count |
| `DENSECORE_ENGINE_THREADS` | `0` | engine compute threads |
| `DENSECORE_GO_WORKERS` | `0` | Go request workers |
| `DENSECORE_SERVER_INFLIGHT` | `1024` | server inflight admission bound |
| `DENSECORE_MAX_NUM_SEQS` | `4` | active engine sequences |
| `DENSECORE_MAX_SEQ_LEN` | auto | sequence-length override |
| `DENSECORE_KV_TARGET_MB` | auto | KV budget override |
| `DENSECORE_KV_TYPE` | `fp16` | KV storage type |
| `AUTH_ENABLED` | `false` | enable API-key auth |
| `API_KEYS` | unset | comma-separated key records |
| `GRPC_ENABLED` | `false` | enable gRPC for the direct server/container; the CLI `--grpc` flag takes precedence |
| `GRPC_PORT` | `50051` | gRPC port |
| `RATE_LIMIT_ENABLED` | `true` | enable HTTP rate limiting |
| `RATE_LIMIT_RPS` | `100` | rate limit per second |
| `RATE_LIMIT_BURST` | `200` | burst capacity |
| `CORS_ENABLED` | `false` | enable CORS middleware |
| `CORS_ALLOWED_ORIGINS` | unset | comma-separated explicit CORS origins; setting this also enables CORS |
| `LOG_FORMAT` | `json` | `json` or `text` |

Direct CLI serve defaults are loopback-only with auth disabled. For remote exposure,
set `--host 0.0.0.0` or `HOST=0.0.0.0`, enable auth explicitly, and configure only
the origins you intend to allow over CORS.

Prefer model- and host-derived memory sizing. Use explicit sequence/KV overrides
only when the deployment has measured capacity limits.

## Startup Semantics

The v0.1.0 HTTP process requires `--model` or `MAIN_MODEL_PATH`. Startup fails
before the serving runtime is opened when the model path is missing or when a
draft model is configured. Readiness stays non-OK while the startup model loads,
and becomes OK once startup has completed, the model is loaded, and optional
dependencies such as auth/rate-limit backends are healthy.
