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
AUTH_ENABLED=true API_KEYS=sk-example:user:default \
  ./bin/densecore-server serve --model ./models/model.gguf
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | unset | GGUF model path |
| `--threads`, `-t` | `0` | inference threads; zero selects runtime auto policy |
| `--auth` | `false` | enable API-key authentication |
| `--grpc` | `false` | enable the gRPC service |
| `--grpc-port` | `50051` | gRPC port |
| `--host` | `0.0.0.0` | HTTP bind address |
| `--port` | `8080` | HTTP port |
| `--verbose`, `-v` | `false` | verbose logging |

## Common Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAIN_MODEL_PATH` | unset | startup model path |
| `DRAFT_MODEL_PATH` | unset | loader-only draft model path |
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
| `LOG_FORMAT` | `json` | `json` or `text` |

Prefer model- and host-derived memory sizing. Use explicit sequence/KV overrides
only when the deployment has measured capacity limits.

## Startup Semantics

The HTTP process can start without a loaded model. In that state liveness succeeds,
while startup/readiness report that inference is unavailable. A model can then be
loaded through `POST /v1/models/load`.
