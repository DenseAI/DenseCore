# Contributor map

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for the Linux development setup.
This map describes code responsibilities, rather than assigning individual maintainers.
Source and tests are authoritative; [ARCHITECTURE.md](ARCHITECTURE.md) provides wider context.

## Find the change boundary

| Change | Start here | Contract to preserve |
| --- | --- | --- |
| Model loading and engine setup | `core/src/runtime/engine.cpp`, `core/src/runtime/engine_internal.h` | Model ownership, engine initialization and teardown, C API behavior |
| Request preparation and submission | `core/src/runtime/request_builder.cpp` | Prompt and token input semantics, short construction guard, publication rollback |
| Prepared weights | `core/src/llm/weights/prepared_weights.cpp` | Model-bound aliases and backing storage; release graph references before reset |
| Engine execution dependencies | `core/src/llm/runtime/deps.cpp`, `core/src/llm/runtime/cpu_execution.cpp` | Engine CPU ownership and explicit backend options |
| Generation lifecycle | `server/internal/domain/generation.go`, `server/internal/engine/generation.go` | Typed inputs, bounded admission, events, cancellation and callback cleanup |
| Scheduling, cancellation and completion | `core/src/runtime/worker.cpp`, `core/src/runtime/scheduler.cpp` | Request state and exactly-once terminal callbacks |
| Graph construction and dispatch | `core/src/llm/graph/build.cpp`, `core/src/llm/graph/decoder.cpp`, `core/src/llm/graph/policy.cpp` | Model execution contracts, tensor lifetime and graph reuse |
| Matmul selection and activation preparation | `core/src/llm/matmul/graph_ops.cpp`, `core/src/llm/matmul/execution_policy.cpp`, `core/src/llm/matmul/quant_cache.cpp` | Layout, quantization type, worker partitioning and execution-scoped activation validity |
| Small-batch native dots | `core/src/llm/matmul/q8_small_batch.h`, `core/src/llm/matmul/q6_small_batch.h`, `core/src/llm/matmul/arm_m4.h`, `core/src/llm/moe/q4_rowpair.h` | Pinned ggml arithmetic order, ISA/model/phase admission, padded strides and output partitions |
| Cached graph model identity | `core/src/llm/runtime/work_context.cpp`, `core/src/llm/runtime/profile.cpp`, `core/src/llm/graph/cache.cpp` | Model identity belongs to the graph context; resetting profiling counters must preserve it |
| Q4 weight packing and kernels | `core/src/kernels/q4k_repacked_gemv.cpp`, `core/src/backend/cpu_backend_moe_raw_batched.inl` | Weight mutation detection, cache budget/lifetime, numerical parity and tails |
| Native MoE | `core/src/llm/moe/native.cpp`, `core/src/llm/moe/native_gemma.cpp` | Routing, expert layouts, arithmetic order and NUMA ownership |
| Fused graph operations | `core/src/llm/graph/fused_ops.cpp`, `core/src/llm/graph/reference_policy.cpp` | Tensor strides, callback partitions and graph-bound userdata |
| KV memory | `core/src/runtime/memory/kv_cache.cpp` | Sequence isolation, allocation and release |
| Prefix eligibility and admission | `core/src/runtime/worker_prefix_cache_policy.inl`, `core/src/runtime/engine.cpp`, `server/internal/service/worker_pool.go` | Effective runtime status matches worker policy; completion-scoped admission protects active recurrent snapshot reuse |
| HTTP and streaming | `server/internal/api/`, `server/internal/service/`, `server/internal/engine/` | Public request/response format, terminal events and CGO ownership |
| Server startup | `server/cmd/densecore/cmd/serve.go`, `server/internal/server/` | Startup model required before serving requests |
| Python API | `python/densecore/`, `python/tests/` | Public Python behavior and native-library lifetime |

Remaining runtime `.inl` files share one translation unit through `inference.cpp`.
Graph construction, graph caches, matmul, attention, SSM and generic MoE now have
independently compiled owners under `core/src/llm/`.
Before extracting one into a `.cpp`, identify anonymous-namespace helpers,
thread-local state and callback linkage. Moving text into another file is not by
itself a safe ownership change. Separate behavior-preserving moves from performance
changes so regressions can be attributed.

## Focused C++ checks

Build the `densecore_tests` target with `DENSECORE_BUILD_TESTS=ON` as shown in
[Testing](../CONTRIBUTING.md#testing). Run these from the repository root:

```bash
# Worker, cancellation and request completion
./build-tests/densecore_tests --gtest_filter='RequestLifecycle*.*:WorkerResultDispatchTest.*:PendingRequestBookkeepingTest.*:EmbeddingFinalizerTest.*'

# Graph dispatch and cache policy
./build-tests/densecore_tests --gtest_filter='BuildTransformerGraphDispatchTest.*:DecodeGraphCachePolicyTest.*:RuntimeOptimizationStateTest.*'

# Matmul reference parity and specialized Q4 prefill tails
./build-tests/densecore_tests --gtest_filter='MatmulBackend.*:NumaStickyRouting.GgmlQ4KRepackedPrefill*:Q4KFingerprintTest.*:BatchedActivationPackCacheTest.*'

# Sampling contracts
./build-tests/densecore_tests --gtest_filter='SamplingRangeTest.*'

# Small-batch dispatch, cached identity and quantized output parity
./build-tests/densecore_tests --gtest_filter='SmallBatchDecodeTest.*:Q8SmallBatchTest.*'
DENSECORE_QWEN36_PROFILE=1 ./build-tests/densecore_tests --gtest_filter='SmallBatchDecodeTest.*:Q8SmallBatchTest.*'
```

These filters are iteration aids, not complete release qualification. Use
`./build-tests/densecore_tests --gtest_list_tests` to find adjacent cases and verify
that a filter actually selects tests. Some hardware-specific cases skip when the
required kernel is unavailable; retain skip output and validate on the target CPU.

For editor navigation, use a Makefiles or Ninja CMake build with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Point clangd at the generated database with
`--compile-commands-dir=build-tests` (or use an absolute path). The database belongs
to that configuration and must be regenerated after build-option changes.

## API changes and performance evidence

After `make lib`, run focused HTTP checks before the broader Go suite:

```bash
(cd server && go test -mod=mod ./internal/api -run '^TestSSEStreamWriter|^TestChatCompletionHandler' -count=1)
(cd server && go test -mod=mod ./...)
```

For runtime or kernel performance changes, also measure the real Go server's
`/v1/chat/completions` with a supported model. Record the source revision, model
checksum, CPU, thread settings, prompts, output lengths, quality gate, and repeated
baseline/candidate runs. Separate prefill/TTFT, decode rate, completed-request
latency and concurrent throughput. A kernel microbenchmark alone does not prove
that the serving path uses the kernel or that API requests become faster.

Dispatch census and other diagnostic environment variables require
`DENSECORE_ENABLE_DEBUG_ENV=ON` in a release build. Missing diagnostic labels in
the default release build do not prove that a kernel was bypassed. Keep diagnostic
runs separate from scored performance runs. Keep detailed run artifacts outside
the public documentation tree until a release claim requires a reviewed summary.

The borrowed-string CGO regression needs a real model. After building the native
library, set `DENSECORE_TEST_MODEL` to a local GGUF and run
`go test -mod=mod -race ./internal/engine -run '^TestRenderChatPromptConcurrentBorrowedBuffers$' -count=1`
from `server/`, with `LD_LIBRARY_PATH` pointing to the native build. This catches
C++ thread-local buffer reuse that the Go race detector alone cannot observe.

Keep release claims and required checks aligned with [RELEASE.md](RELEASE.md).
