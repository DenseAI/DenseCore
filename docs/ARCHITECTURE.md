# DenseCore Architecture

DenseCore is a CPU-first, memory-centric inference runtime. The maintained LLM
path combines a C++ engine, model-aware scheduler, paged KV cache, GGML-backed
decoder graphs, quantized CPU kernels, and a Go serving layer.

This description reflects commit `22b672d` reviewed on 2026-07-13. It describes
implemented ownership, not the broader set of experimental operation classes in
the repository.

## Request Ownership

The production chat path is:

```text
HTTP POST /v1/chat/completions
  -> server/internal/api/chat_handler.go
  -> server/internal/service/chat_service.go
  -> server/internal/engine/engine.go (CGO/C API)
  -> core/src/runtime/engine.cpp
  -> core/src/runtime/scheduler*.cpp
  -> core/src/runtime/worker.cpp
  -> core/src/runtime/inference_graph_inline_impl.inl
  -> model-specific graph support and quantized kernels
  -> token/result callback to the Go server
```

Prompt ownership is deliberately shared across the Go and C++ boundary. The Go
service owns OpenAI request validation and streaming; C++ owns model templates,
tokenization, execution, KV state, and sampling. Qwen template/no-thinking
behavior is kept aligned through `model_prompt_templates.cpp` and
`server/internal/service/chat_prompt.go`.

## Runtime Components

### Engine

`core/src/runtime/engine.cpp` owns model loading, runtime policy construction,
KV and scheduler sizing, thread startup, and the C API entry points. The scheduler
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

## Model-Specific Paths

| Family | Implemented path | Current maturity |
| --- | --- | --- |
| Qwen3.5 / Qwen3.6 | hybrid SSM graph, model-specific prompt policy, native MoE, Q4/Q5/Q6/Q8 quant paths, custom GEMV and LM-head paths | maintained and fallback-free gated; C4 x86 checkpoint clears the 10% comparison target, C4A decode does not |
| Gemma4 | sliding/shared KV, model transforms, native MoE, quantized matmul planning, residency-aware loader policy | maintained and QA-gated; comparison target still misses on decode |
| LFM2 / LFM2.5 | short-convolution and MoE graph callbacks, maintained ARM repack path | functional and measured in earlier server runs; not a current speed headline |
| Dense decoder families | descriptor-driven GGML graph path | family-dependent; validate each model/quantization |
| Multimodal and generic operation graphs | HAL/operation registry and selected kernels | partial or experimental unless a product-specific e2e test qualifies the path |

Qwen-specific owners include `inference_matmul_plan.inl`,
`inference_matmul_gemv_custom.inl`, and the native MoE blocks in
`inference_graph_support.inl`. Gemma4 graph topology is primarily in
`inference_graph_inline_impl.inl`; loader residency and expert layout live in
`model_loader.cpp` and `gemma4_packed_expert_layout.h`.

## Fast-Path and Fallback Policy

DenseCore has two distinct fallback contracts:

- Generic compatibility paths may select a supported fallback operation.
- Qualified Qwen/Gemma benchmark paths fail the target gate if the intended fast
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

The server supports model load/unload, but production deployments should normally
preload one known model and use readiness/startup probes. Loading a draft model is
currently scaffold only: the draft model identifier is stored, but the decode loop
does not consume it for speculative decoding.

## Known Limits

- There is no final-HEAD, six-row, five-cycle benchmark matrix yet.
- C4A Qwen decode and Gemma4 decode remain below the current llama-server target.
- Multi-socket NUMA routing has a scoped two-socket Qwen3.6 isolated-decode
  checkpoint; it is not a general latency or all-model claim. See
  [NUMA_STICKY_ROUTING.md](NUMA_STICKY_ROUTING.md).
- Generic operation coverage is broader than end-to-end model coverage.
- Optional accelerator presence does not imply a production throughput advantage.
