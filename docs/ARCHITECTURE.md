# DenseCore Architecture

DenseCore is a CPU-first, memory-centric inference runtime. The maintained LLM
path combines a C++ engine, model-aware scheduler, paged KV cache, GGML-backed
decoder graphs, quantized CPU kernels, and a Go serving layer.

This description reflects implemented ownership in the current source tree. It
describes the maintained path, not the broader set of experimental operation
classes in the repository.

## Request Ownership

The maintained v0.1 chat path is:

```text
HTTP POST /v1/chat/completions
  -> server/internal/api/chat_handler.go
  -> server/internal/service/chat_service.go
  -> server/internal/domain/generation.go (typed input and lifecycle)
  -> server/internal/engine/generation.go (admission and event ownership)
  -> server/internal/engine/engine.go (CGO/C API)
  -> core/src/runtime/request_builder.cpp
  -> core/src/runtime/scheduler*.cpp
  -> core/src/runtime/worker.cpp
  -> core/src/llm/graph/build.cpp (model contract and graph-family dispatch)
  -> core/src/llm/graph/decoder.cpp
  -> model-specific graph support and quantized kernels
  -> token/result callback to the Go server
```

Prompt ownership is deliberately shared across the Go and C++ boundary. The Go
service owns OpenAI request validation and streaming; C++ owns model templates,
tokenization, execution, KV state, and sampling. Qwen template/no-thinking
behavior is kept aligned through `model_prompt_templates.cpp` and
`server/internal/service/chat_prompt.go`.

Accepted generation requests have one native terminal owner. Admission is
serialized with the transition from `RUNNING` to `DRAINING`; after that
transition, new submissions are rejected before entering the pending queue.
Pending cancellation, active cancellation, shutdown, execution error, and normal
success all converge on an exactly-once finalizer. Request storage is recycled
only after terminal ownership is resolved, and the callback loop remains alive
until worker result producers have stopped and their result queue is drained.

## Runtime Components

### Engine

`core/src/runtime/engine.cpp` owns model loading, runtime policy construction,
KV and scheduler sizing, thread startup, and engine lifecycle C API entry points.
`request_builder.cpp` owns prompt rendering, tokenization, request preparation and
submission. A short per-engine construction mutex protects preparation; it is
released before publishing a request. Native submission does not wait for completion. The scheduler
defaults to four active sequences and one concurrent prefill sequence. KV memory
is derived from the loaded model and host budget unless explicitly overridden.

The `EngineState` input compute buffer and the GGML graph pool are different
allocations. A fixed input buffer still exists, while graph contexts are sized
from request/model/host facts and can grow or shrink in `worker.cpp`.

### Scheduler and Worker

`scheduler.cpp`, `scheduler_policy.cpp`, and `scheduler_queue.cpp` normalize
admission and queue policy. `worker.cpp` owns the execution loop, graph-context
pool, batching, prefill chunking, result dispatch, and target fast-path summary.

Paged KV storage is implemented under `core/src/runtime/memory/`. Prefix reuse is
local to the engine process and must preserve model, template, token, RoPE, KV,
LoRA, graph-family, sliding-window, and SSM identity.

### Graph Planning

Model family and topology planning live in:

- `core/src/llm/graph/planning.cpp`
- `core/src/models/decoder_model_spec.cpp`
- `core/src/models/model_graph_capabilities.cpp`
- `core/src/models/model_execution_contract.cpp`

The main decoder graph remains GGML-backed. The separate `OperationGraph` and
`GraphExecutor` surface supports generic operations and may use admitted fallback
implementations. It must not be confused with fallback-free qualification of the
Qwen/Gemma target path.

Prefill topology reuse excludes MoE and hybrid SSM models. Decode graph reuse is
enabled only when the model execution contract admits it. Gemma4 currently reuses
arena/context storage, not a cached prefill topology.

## Execution and Resource Ownership

Each engine owns its inference configuration, operation registry and CPU backend.
The CPU backend receives execution options and telemetry explicitly; it does not
read worker state. Non-CPU plugin registration retains process-wide compatibility.
Legacy standalone CPU operations remain available outside engine execution.

Each cached graph owns a stable work context. Callback userdata binds that context,
including its model identity, backend and activation caches. Work-context allocation
and reset live in `llm/runtime/work_context.cpp`; profiling lives in
`llm/runtime/profile.cpp`. `llm/graph/cache.cpp` owns graph storage, arena reuse,
LRU accounting and eviction. Attention, matmul, generic MoE and SSM callbacks compile
as separate translation units under their corresponding `llm/` directories.
Native MoE routing and shared/Qwen execution live in `llm/moe/native.cpp`;
Gemma packed execution lives in `llm/moe/native_gemma.cpp`. General graph policy,
reference-probe policy and fused callbacks live in `llm/graph/policy.cpp`,
`reference_policy.cpp` and `fused_ops.cpp`.

`TransformerModel` owns canonical weights and a `PreparedWeights` component for
prepared aliases and backing storage. Graphs must release their references before
prepared storage is reset. Per-sequence recurrent state remains request-owned.

The Go generation adapter owns event delivery, cancellation and callback cleanup.
The service retains the same engine lease from preparation through generation
completion. `Done` means callback cleanup has completed; native worker retirement
can follow it. Engine teardown joins producers before destroying their resources.
When effective prefix reuse and recurrent snapshots require exclusive execution,
the Go adapter bounds admission and the native worker independently waits for the
previous request to retire before activating another.

`core/cmake/densecore_modules.cmake` assigns sources to internal object modules.
Production, test and probe objects use separate compilation variants while the
public C library and exported API remain the compatibility boundary.

## Model-Specific Paths

| Family | Implemented path | Current maturity |
| --- | --- | --- |
| Qwen3.5 / Qwen3.6 | hybrid SSM graph, model-specific prompt policy, native MoE, Q4/Q5/Q6/Q8 quant paths, custom GEMV and LM-head paths | maintained and fallback-free gated; performance remains model, quantization, build-profile, and host qualified |
| Qwen3.8-27B `UD-Q4_K_M` | Qwen3.5-family hybrid SSM graph with the trailing NextN/MTP block excluded from trunk execution; thinking-on chat template by default | beta text-only; vision/mmproj, MTP speculative decoding, and 262K context are not qualified |
| Gemma4 | sliding/shared KV, model transforms, native MoE, quantized matmul planning, residency-aware loader policy | maintained and QA-gated; comparison target still misses on decode |
| LFM2 / LFM2.5 | short-convolution and MoE graph callbacks, maintained ARM repack path | functional and measured in earlier server runs; not a current speed headline |
| Dense decoder families | descriptor-driven GGML graph path | family-dependent; validate each model/quantization |
| Multimodal and generic operation graphs | HAL/operation registry and selected kernels | partial or experimental unless a product-specific e2e test qualifies the path |

Qwen-specific owners include `llm/matmul/execution_policy.cpp`,
`llm/matmul/gemv_exec.cpp`, and `llm/moe/native.cpp`. Gemma4 graph topology is primarily in
`llm/graph/decoder.cpp`; loader residency and expert layout live in
`model_loader.cpp` and `gemma4_packed_expert_layout.h`.

## Fast-Path and Fallback Policy

DenseCore has two distinct fallback contracts:

- Generic compatibility paths may select a supported fallback operation.
- Qualified Qwen/Gemma inference paths fail the target gate if the intended fast
  path is rejected, bypassed, or covered by a slower implementation.

`model_execution_contract.cpp` defines target model requirements. Worker summaries
enforce those requirements and expose rejection/fallback counters. Output quality
is an independent gate; a fast path is not accepted because it merely avoids a
crash.

## Hardware Boundary

The maintained text path is CPU-first on x86-64 and Arm64. Highway, GGML CPU,
AMX/AVX-512/VNNI, NEON/SVE2/I8MM, and model-specific repacked layouts are selected
by build and runtime admission.

Apple Metal/ANE, QNN, plugin backends, and the generic HAL are real code surfaces,
but they are not all full-model production paths. See
[HARDWARE_SUPPORT.md](HARDWARE_SUPPORT.md) for the maturity boundary.

## Serving Boundary

The Go server owns HTTP/gRPC lifecycle, authentication, rate limiting, request
validation, SSE streaming, health, metrics, and graceful shutdown. DenseCloud
provides the common server chassis; it does not own DenseCore kernel or scheduler
performance.

The v0.1.0 server owns one startup model for the lifetime of the process. Model
replacement happens by deploying a new process and draining the old one through
the DenseCloud shutdown contract. Dynamic load/unload and speculative draft
models are not public serving features.

Shutdown ownership is ordered as follows:

```text
stop HTTP/API admission
  -> close queue admission and terminally reject queued requests
  -> join Go submitter workers
  -> join native-completion trackers and release engine leases
  -> drain/terminate accepted C++ requests
  -> close the native engine
```

An engine generation is detached before unload and closed only after its active
leases reach zero. This prevents a Go request path from retaining a usable bare
native handle across engine destruction without serializing unrelated requests.

## Known Limits

- Dated benchmark research in this repository does not qualify current source
  or release artifacts for a comparative performance claim.
- Multi-socket NUMA routing requires workload-specific validation before a
  deployment claim.
- Generic operation coverage is broader than end-to-end model coverage.
- Standalone compatibility backends, plugin registration and some diagnostics remain
  process-scoped; engine ownership does not imply isolation of every legacy API.
- Optional accelerator presence does not imply a production throughput advantage.
