# DenseCore Architecture

A deep dive into DenseCore's system design, execution ownership, and performance-sensitive subsystems.

---

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [System Components](#system-components)
- [Request Lifecycle](#request-lifecycle)
- [Memory Management](#memory-management)
- [Performance Optimizations](#performance-optimizations)
- [Extension Points](#extension-points)
- [Advanced Subsystems](#advanced-subsystems)
  - [Flash Attention for CPU](#flash-attention-for-cpu)
  - [Mixture of Experts (MoE)](#mixture-of-experts-moe)
  - [Hybrid Scheduler (Apple Silicon)](#hybrid-scheduler-apple-silicon)
  - [Concurrency Primitives](#concurrency-primitives)
  - [Memory Subsystem](#memory-subsystem)

---

## Overview

DenseCore is a **three-layer execution architecture** for memory-centric AI inference. It is CPU-first, but it is designed to expose heterogeneous backend paths without collapsing correctness, operability, or fallback behavior into benchmark-only tradeoffs.

```mermaid
graph TB
    subgraph "Layer 1: User Interface"
        PySDK[Python SDK]
        GoServer[Go REST Server]
        CLI[CLI Tools]
    end

    subgraph "Layer 2: Inference Engine (C++)"
        Scheduler[Request Scheduler]
        Worker[Inference Worker]
        KVCache[Paged KV Cache]
        ModelMgr[Model Manager]
    end

    subgraph "Layer 3: Compute Backend"
        GGML[GGML Tensor Ops]
        Quant[Quantization]
        SIMD[SIMD Kernels AVX2/AVX-512]
    end

    PySDK --> |ctypes FFI| Scheduler
    GoServer --> |CGO| Scheduler
    CLI --> |Direct Link| Scheduler

    Scheduler --> Worker
    Worker --> KVCache
    Worker --> ModelMgr
    Worker --> GGML

    GGML --> Quant
    GGML --> SIMD

    style PySDK fill:#e1f5ff
    style GoServer fill:#e1ffe1
    style Worker fill:#fff4e1
    style GGML fill:#ffe1f5
```

**Key Design Decisions:**

1. **C++ Core on the Critical Path:** Tensor execution, memory management, scheduling, and cache behavior stay in native code.
2. **Python for Local and Embedded UX:** The Python SDK provides a low-friction surface for local inference, embedding, and app integration.
3. **Go for Production Service Surfaces:** The server and CLI provide observable, operational entry points rather than example-only wrappers.
4. **CPU-First Portability with Heterogeneous Extensions:** DenseCore prioritizes portable CPU execution while allowing backend-specific acceleration where it can be introduced without weakening fallback semantics.

---

## Design Philosophy

### 1. **Python-First Developer Experience**

Unlike llama.cpp (CLI-focused) or raw GGML (C-only), DenseCore prioritizes Python developers:

```python
# This should "just work" - no compilation, no configuration
import densecore
model = densecore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
response = model.generate("Hello!")
```

**Implementation:**
- ctypes FFI (no C extension compilation required)
- Automatic model downloading from HuggingFace Hub
- Async/await support for modern Python
- Type hints and docstrings

### 2. **Production Operability by Default**

Many inference libraries are research-oriented. DenseCore treats the serving surface as a product surface:

- explicit health checks (`/health/live`, `/health/ready`, `/health/startup`)
- Prometheus metrics and runtime profiling surfaces
- graceful shutdown with request draining
- request cancellation and timeouts
- structured logging and deployable configuration

### 3. **CPU-First, Heterogeneous by Design**

DenseCore starts from CPU realities such as memory locality, NUMA behavior, cache pressure, and batching overhead. Optional backend acceleration should extend that execution model rather than replace it.

- quantization paths are used to improve footprint and deployability, not just headline throughput
- paged KV cache reduces allocation churn and keeps memory behavior explicit
- SIMD and backend-specific kernels are valuable only when they preserve correctness and fallback semantics
- graph reuse and scheduler behavior matter because memory movement and runtime overhead can dominate real serving latency

### 4. **Benchmark Fairness and Explicit Feature Maturity**

DenseCore documents performance-sensitive features with two guardrails:

- benchmark claims should be reproducible and separated from measurement artifacts
- backend maturity should be called out honestly as production-ready, beta, experimental, or conditional rather than implied by a single fast path

---

## System Components

### C++ Inference Engine

**Location:** `core/src/`

The heart of DenseCore. Written in C++17 for performance and control.

#### Key Files

| File | Purpose |
|------|---------|
| `inference.cpp` | Main inference loop, transformer graph building |
| `worker.cpp` | Request scheduler, batch processing |
| `kv_cache.cpp` | Paged KV cache implementation |
| `model_loader.cpp` | GGUF model parsing and loading |
| `densecore.h` | Public C API |

#### Inference Engine Responsibilities

1. **Model Loading:** Parse GGUF files, allocate tensors
2. **Request Scheduling:** Queue incoming requests, batch when possible
3. **Graph Execution:** Build computation graph, execute via GGML
4. **KV Cache Management:** Allocate/free cache blocks, eviction policy
5. **Token Sampling:** Top-K, Top-P, temperature, repetition penalty

#### Memory Layout

```
┌─────────────────────────────────────┐
│  Model Weights (Read-Only)          │  <-- mmap()'d from GGUF file
│  - Quantized (INT4/INT8)            │
│  - Embedded in GGUF file            │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  KV Cache (Paged)                   │  <-- Dynamically allocated
│  - 16KB blocks per page             │
│  - LRU eviction                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Compute Scratch (Arena Allocator)  │  <-- Temporary for graph execution
│  - Pre-allocated 8GB buffer         │
│  - Reset after each request         │
└─────────────────────────────────────┘
```

**Memory Efficiency:**
- Model weights: mmap (zero-copy read from disk)
- KV cache: Paged allocation (no upfront allocation)
- Scratch space: Arena allocator (eliminates malloc overhead)

---

### Python SDK

**Location:** `python/densecore/`

Pythonic wrapper around the C++ core.

#### Architecture

```python
# High-level API
from densecore import DenseCore, from_pretrained

# Low-level bindings (ctypes)
from densecore.bindings import libdensecore, DenseCoreHandle
```

**Key Components:**

| Module | Purpose |
|--------|---------|
| `densecore/__init__.py` | Main `DenseCore` class, public API |
| `densecore/bindings.py` | ctypes FFI to `libdensecore.so` |
| `densecore/async_sdk.py` | Async/await support for streaming |
| `densecore/config.py` | `GenerationConfig`, `ModelConfig` |
| `densecore/exceptions.py` | Custom exception hierarchy |
| `densecore/quantize/` | Quantization config and API |
| `densecore/prune/` | Pruning config and API |

#### Async Streaming Implementation

```python
# Internal implementation (simplified)
class DenseCore:
    async def stream_async(self, prompt: str, **kwargs):
        queue = asyncio.Queue()

        def callback(token: str, is_finished: int, user_data):
            asyncio.create_task(queue.put((token, is_finished)))

        # Submit to C++ engine (non-blocking)
        request_id = libdensecore.SubmitRequest(
            self._handle, prompt, callback, None
        )

        # Yield tokens as they arrive
        while True:
            token, is_finished = await queue.get()
            yield token
            if is_finished:
                break
```

---

### Go REST Server

**Location:** `server/`

Production HTTP/REST server with OpenAI-compatible API.

#### Features

- **OpenAI-Compatible API:** Drop-in replacement for OpenAI's `/v1/chat/completions`
- **Streaming SSE:** Server-Sent Events for streaming responses
- **Authentication:** API key-based auth with tier-based rate limiting
- **Observability:**
  - Prometheus metrics (`/metrics`)
  - Health checks (`/health/*`)
  - Structured JSON logging

#### Request Flow

```
Client Request (HTTP)
    ↓
Go HTTP Handler (main.go)
    ↓
Authentication Middleware (auth.go)
    ↓
Rate Limiter (rate_limiter.go)
    ↓
C++ Engine (via CGO)
    ↓
Streaming Callback
    ↓
SSE Response to Client
```

#### Key Files

| File | Purpose |
|------|---------|
| `main.go` | HTTP server setup, routing |
| `internal/handlers/chat.go` | `/v1/chat/completions` endpoint |
| `internal/handlers/embeddings.go` | `/v1/embeddings` endpoint |
| `internal/middleware/auth.go` | API key authentication |
| `internal/metrics/prometheus.go` | Metrics collection |

---

## Request Lifecycle

### 1. Request Submission

```python
# User code
response = model.generate("Hello, world!", max_tokens=100)
```

```
Python SDK
    ↓
ctypes → SubmitRequest(handle, prompt, max_tokens, callback, user_data)
    ↓
C++ RequestScheduler::AddRequest()
    ↓
Queue request in pending_requests_
```

### 2. Batching & Scheduling

The scheduler attempts to batch compatible requests:

```cpp
// Pseudo-code from worker.cpp
while (true) {
    std::vector<Request*> batch;

    // Collect up to max_batch_size requests
    while (batch.size() < max_batch_size && !pending_requests_.empty()) {
        auto req = pending_requests_.front();
        pending_requests_.pop();
        batch.push_back(req);
    }

    // Process batch
    ProcessBatch(batch);
}
```

**Batching Rules:**
- Same model only
- Compatible max_tokens
- KV cache space available

### 3. Prefill Phase (Prompt Processing)

For a batch of N requests with prompts of length L₁, L₂, ..., Lₙ:

```
For each request i:
    1. Tokenize prompt → token_ids[L_i]
    2. Build transformer graph (attention, FFN, etc.)
    3. Execute graph → logits[vocab_size]
    4. Sample next token → next_token_id
    5. Store KV cache for this sequence
```

**KV Cache Storage:**

```
seq_123 → [K_block_0, K_block_1, V_block_0, V_block_1]
          ↑ Each block = 16KB = 16 tokens × hidden_size × sizeof(fp16)
```

### 4. Decode Phase (Token Generation)

Iteratively generate tokens until `max_tokens` or `<eos>`:

```
Loop until stopping condition:
    1. For each active request:
        - Retrieve KV cache blocks
        - Build graph for single-token inference
        - Execute graph → logits
        - Sample next token
        - Append to KV cache

    2. Invoke callback with new token

    3. Check stopping:
        - max_tokens reached?
        - EOS token generated?
        - User cancellation?
```

### 5. Completion & Cleanup

```cpp
// Callback to Python
callback(final_token, /*is_finished=*/1, user_data);

// Release KV cache blocks
kv_cache_->ReleaseSequence(seq_id);

// Remove from active requests
active_requests_.erase(request_id);
```

---

## Memory Management

### Paged KV Cache (Inspired by vLLM)

Traditional KV cache pre-allocates contiguous memory for max sequence length:

```
❌ Traditional: [K][K][K]...[K][K][K]  (10,000 slots, 80% wasted)
                 ↑ Used    ↑ Wasted
```

**DenseCore's Paged KV Cache:**

```
✅ Paged: [Block 0] → [Block 1] → [Block 2] → ...
           16 tokens   16 tokens   16 tokens

Only allocate blocks as needed!
```

**Benefits:**
- **95% memory savings** for short sequences
- **Dynamic allocation:** No upfront commitment
- **LRU eviction:** Reuse blocks for new requests

**Implementation (`kv_cache.cpp`):**

```cpp
class PagedKVCache {
    std::vector<KVBlock> free_blocks_;  // Available blocks
    std::unordered_map<int, std::vector<int>> seq_to_blocks_;  // seq_id → block_ids

public:
    std::vector<int> AllocateBlocks(int seq_id, int num_tokens) {
        int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<int> allocated;

        for (int i = 0; i < num_blocks; i++) {
            if (free_blocks_.empty()) {
                EvictLRU();  // Evict least recently used sequence
            }
            allocated.push_back(free_blocks_.back().id);
            free_blocks_.pop_back();
        }

        seq_to_blocks_[seq_id] = allocated;
        return allocated;
    }
};
```

### Arena Allocator for Compute Graph

GGML uses an arena allocator for temporary tensors during graph execution:

```cpp
// Allocate 8GB scratch space upfront
ggml_context* ctx = ggml_init({
    .mem_size = 8ULL * 1024 * 1024 * 1024,  // 8GB
    .mem_buffer = nullptr,  // Auto-allocate
    .no_alloc = false
});

// All ggml_new_tensor_*() calls use this arena (no malloc)
struct ggml_tensor* tmp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, seq_len);

// After graph execution, reset arena (instant "free")
ggml_reset(ctx);
```

**Why this is fast:**
- **Zero malloc overhead:** Single allocation
- **Zero fragmentation:** Bump allocator
- **Instant cleanup:** Reset pointer, no iteration

---

## Performance Optimizations

### 1. **Quantization (INT4/INT8)**

Traditional LLMs store weights as FP32 (4 bytes per parameter):

```
7B model × 4 bytes = 28GB memory
```

DenseCore uses **block-wise quantization:**

```cpp
// INT4 quantization (32 weights → 1 scale + 16 bytes)
struct Block {
    float scale;       // 4 bytes
    uint8_t data[16];  // 16 bytes (32 × 4-bit packed)
};

// Dequantize on-the-fly during matmul
float dequantize_int4(uint8_t val, float scale) {
    int8_t signed_val = (val & 0xF) - 8;  // Convert to [-8, 7]
    return signed_val * scale;
}
```

**Result:**
- 7B model × 0.5 bytes (INT4) = **3.5GB**
- 8x memory reduction!

### 2. **SIMD Kernels (AVX2/AVX-512)**

GGML automatically selects optimal kernels for your CPU:

```cpp
// Matrix multiplication with AVX2 (8 floats at once)
void ggml_vec_dot_f32(int n, float* result, const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);  // Fused multiply-add
    }

    // Horizontal sum
    *result = _mm256_reduce_add_ps(sum);
}
```

**Speedup:** 4-8x vs. scalar code on modern CPUs

### 3. **Graph Caching**

Rebuilding the computation graph for every token generation step is expensive on CPU. DenseCore implements **Graph Caching**:

```cpp
// First run: Build graph, allocate nodes, compute
auto graph = BuildGraph(ctx, tokens);

// Subsequent runs: Reuse structure, just update pointers
graph->UpdateInput(new_token);
graph->Compute();
```

**Benefit:** Reduces CPU overhead by 30-40% for small batches.

### 4. **Smart Preemption**

When the system is overloaded, the scheduler must decide which request to pause to free up memory (KV cache blocks).

**Strategy:**
1.  **Priority**: Lower priority requests are preempted first.
2.  **Least Progress**: If priorities match, requests with the fewest generated tokens are preempted.

This minimizes wasted computation compared to random or LIFO preemption.

### 5. **Continuous Batching**

Unlike traditional batching (wait for batch to fill), DenseCore uses **continuous batching**:

```
Traditional Batching:
  Wait for 8 requests → Process batch → Wait for next 8

Continuous Batching:
  Process whatever requests are available
  Requests can join/leave mid-generation
```

**Benefit:** Higher throughput, lower latency

---

## Extension Points

### Adding Custom Sampling Strategies

Extend `token_sampler.cpp`:

```cpp
class CustomSampler : public TokenSampler {
public:
    int Sample(const float* logits, int vocab_size, SamplingConfig config) override {
        // Your custom logic (e.g., nucleus sampling, mirostat)
        return selected_token_id;
    }
};

// Register in factory
SamplerFactory::Register("custom", []() { return std::make_unique<CustomSampler>(); });
```

### Adding New Model Architectures

Implement `TransformerBuilder` interface:

```cpp
class Qwen3Builder : public TransformerBuilder {
public:
    ggml_tensor* BuildAttention(ggml_context* ctx, ggml_tensor* input) override {
        // Qwen3-specific: QK normalization
        ggml_tensor* q = BuildQuery(ctx, input);
        ggml_tensor* q_norm = ggml_norm(ctx, q);  // QK-Norm
        // ... rest of attention
    }
};
```

### Adding Custom Metrics

Extend Prometheus metrics:

```go
// In server/internal/metrics/custom.go
var customMetric = prometheus.NewHistogram(prometheus.HistogramOpts{
    Name: "densecore_custom_metric",
    Help: "Your custom metric",
})

func RecordCustom(value float64) {
    customMetric.Observe(value)
}
```

---

## Performance Characteristics

### Latency Profile

Typical request latency breakdown (Qwen2.5-0.5B, 100 token generation):

```
┌─────────────────┬──────────────┐
│ Phase           │ Time         │
├─────────────────┼──────────────┤
│ Queue Wait      │ 2ms          │
│ Prefill (prompt)│ 50ms (10 tok)│
│ Decode (gen)    │ 3000ms       │
│   First Token   │ 50ms (TTFT)  │
│   Subsequent    │ 30ms/token   │
└─────────────────┴──────────────┘
Total: ~3052ms for 100 tokens = 32.7 TPS
```

### Throughput Scaling

```
Batch Size = 1:  30 TPS
Batch Size = 4:  24 TPS per request (96 total)
Batch Size = 8:  20 TPS per request (160 total)
```

**Lesson:** Batching improves total throughput but increases per-request latency.

---

## Comparison Table

|Feature | DenseCore | llama.cpp | vLLM | Transformers |
|---------|-----------|-----------|------|--------------|
| **Quantization** | INT4/INT8/FP8 | INT4/INT8 | FP16/BF16 | FP32/FP16 |
| **KV Cache** | Paged (vLLM-style) | Contiguous | Paged | Implicit |
| **Batching** | Continuous | None | Continuous | Static |
| **Graph Caching** | ✅ Yes | ❌ No | ✅ CUDA Graph | ❌ No |
| **Python API** | ✅ Native | ⚠️ Bindings | ✅ Native | ✅ Native |
| **Async Support** | ✅ Yes | ❌ No | ✅ Yes | ⚠️ Limited |
| **Production** | ✅ K8s-ready | ❌ CLI-only | ✅ K8s-ready | ⚠️ Manual |

---

## Next Steps

- [Deploy to Production](DEPLOYMENT.md)
- [Optimize Your Models](MODEL_OPTIMIZATION.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

## Advanced Subsystems

This section provides implementation details for senior architects and contributors.

### Flash Attention for CPU

**Location:** `core/include/flash_attention.h`

DenseCore implements Flash Attention (Dao et al., 2022) optimized for CPU inference:

```mermaid
graph LR
    subgraph "Standard Attention O(n²)"
        A1[Q×K^T] --> A2[Full S matrix n×n]
        A2 --> A3[Softmax]
        A3 --> A4[S×V]
    end

    subgraph "Flash Attention O(n)"
        B1[Tile Q into blocks] --> B2[Stream K,V blocks]
        B2 --> B3[Online Softmax]
        B3 --> B4[Accumulate O]
    end

    style A2 fill:#ff6b6b
    style B3 fill:#51cf66
```

**Algorithm Overview:**

1. **Tiling**: Q, K, V are processed in 64×64 blocks (tuned for L2 cache)
2. **Online Softmax**: Maintains running max and sum, avoiding O(n²) intermediate storage
3. **Rescaling**: Output is incrementally rescaled as new blocks are processed

```cpp
// Tiled attention loop (simplified from flash_attention.h)
for (int j = 0; j < seq_len_kv; j += Bc) {       // Iterate KV blocks
    for (int i = 0; i < seq_len_q; i += Br) {    // Iterate Q blocks

        // Step 1: Compute Q @ K^T for this tile (SIMD)
        simd::ComputeQK_AVX512(Q + i*d, K + j*d, scratch.qk_block, ...);

        // Step 2: Apply causal mask (vectorized)
        if (config.causal) simd::ApplyMask_AVX512(scratch.qk_block, i, j, ...);

        // Step 3: Online softmax update
        simd::SoftmaxBlock_AVX512(scratch.qk_block, block_max, block_sum, ...);

        // Step 4: P @ V accumulation
        simd::ComputePV_AVX512(scratch.qk_block, V + j*d, scratch.pv_block, ...);

        // Step 5: Rescale and update output
        simd::UpdateOutput_AVX512(O + i*d, scratch.pv_block, alpha, beta, ...);
    }
}
```

**Memory Complexity:**

| Approach | Memory | Notes |
|----------|--------|-------|
| Standard | O(n²) | Stores full n×n attention matrix |
| Flash Attention | O(n) | Only stores Br×Bc tile + accumulators |

**GQA (Grouped Query Attention) Support:**

```cpp
// FlashAttentionGQA handles n_head_q != n_head_kv
void FlashAttentionGQA(Q, K, V, O, batch, n_head_q, n_head_kv, ...) {
    int n_rep = n_head_q / n_head_kv;  // KV head repetition
    for (int h = 0; h < n_head_q; h++) {
        int kv_head = h / n_rep;  // Which KV head to use
        FlashAttentionForward(Q[h], K[kv_head], V[kv_head], O[h], ...);
    }
}
```

---

### Mixture of Experts (MoE)

**Location:** `core/include/moe/`

DenseCore supports MoE architectures (Mixtral, DeepSeek-V2, Qwen2-MoE):

```mermaid
graph TB
    Input[Hidden State] --> Router[Gating Router]
    Router --> |Top-K| E1[Expert 1]
    Router --> |Top-K| E2[Expert 2]
    Router --> |weights| Combine[Weighted Sum]
    E1 --> Combine
    E2 --> Combine
    Combine --> Output[Output]

    style Router fill:#ffd43b
    style E1 fill:#748ffc
    style E2 fill:#748ffc
```

**Routing Mechanism:**

```cpp
struct MoERouteResult {
    std::vector<int> expert_ids;     // [batch * top_k] Selected experts
    std::vector<float> weights;      // [batch * top_k] Normalized weights
    std::vector<int> token_indices;  // For batch reassembly

    int GetExpertId(int token, int k) const {
        return expert_ids[token * top_k + k];
    }
};

// Top-K routing with softmax normalization
MoERouteResult MoETopKRoute(const float* router_logits, int batch, int n_experts, int top_k);
```

**Expert FFN Structure (SwiGLU):**

```cpp
struct ExpertFFNWeights {
    const float* w1;  // Gate projection [intermediate, hidden]
    const float* w2;  // Down projection [hidden, intermediate]
    const float* w3;  // Up projection [intermediate, hidden]
};

// SwiGLU: gate(x) * up(x) → down
output = matmul(silu(matmul(x, w1)) * matmul(x, w3), w2);
```

---

### Hybrid Scheduler (Apple Silicon)

**Location:** `core/include/hybrid_scheduler.h`

On Apple Silicon, DenseCore coordinates three compute units:

```
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ Operation   │ Prefill (B>1)│ Decode (B=1) │ Best Unit    │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ Embedding   │ CPU          │ CPU          │ Memory bound │
│ Q/K/V Proj  │ GPU          │ GPU/ANE      │ Compute bound│
│ Attention   │ GPU (Flash)  │ GPU          │ Memory bound │
│ FFN Up/Down │ GPU          │ GPU/ANE      │ Compute bound│
│ RMSNorm     │ GPU/CPU      │ CPU          │ Memory bound │
│ Sampling    │ CPU          │ CPU          │ Sequential   │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

**Profiling-Based Scheduling:**

```cpp
class HybridScheduler {
public:
    // Profile model at load time
    bool ProfileModel(const ModelConfig& config) {
        for (auto op : all_operations) {
            OpProfile profile;
            profile.cpu_latency_us = MeasureOnCPU(op);
            profile.gpu_latency_us = MeasureOnGPU(op);
            profile.ane_latency_us = MeasureOnANE(op);
            profile.recommended_unit = SelectOptimal(profile);
        }
    }

    // Generate execution plan
    ExecutionPlan GetExecutionPlan(int batch_size, int seq_len, bool is_prefill);
};
```

**Thermal Adaptation:**

When GPU throttles due to heat, the scheduler dynamically shifts FFN layers to ANE:

```cpp
void AdaptToThermalState() {
    if (GetThermalPressure() > kHighThreshold) {
        // Shift compute-bound ops from GPU to ANE
        ForceUnit(LayerOpType::FFNUp, ComputeUnit::ANE);
        ForceUnit(LayerOpType::FFNDown, ComputeUnit::ANE);
    }
}
```

---

### Concurrency Primitives

**Location:** `core/include/lockfree_queue.h`

#### Lock-Free MPSC Queue

Based on Michael-Scott algorithm with **Tagged Pointer ABA Protection**:

```cpp
// x86-64: Only 48 bits used for virtual addresses
// Upper 16 bits store version counter to prevent ABA problem
struct TaggedPtr {
    static constexpr uint64_t PTR_MASK = 0x0000FFFFFFFFFFFFULL;
    static constexpr int TAG_SHIFT = 48;

    uint64_t packed;

    void* Ptr() const {
        uint64_t addr = packed & PTR_MASK;
        if (addr & (1ULL << 47)) addr |= 0xFFFF000000000000ULL;  // Sign extend
        return reinterpret_cast<void*>(addr);
    }

    uint16_t Tag() const { return static_cast<uint16_t>(packed >> TAG_SHIFT); }
};
```

**Push/Pop with CAS:**

```cpp
void Push(T* item) {
    Node* new_node = new Node(item);
    while (true) {
        TaggedPtr tail = tail_.load(acquire);
        TaggedPtr next = tail->next.load(acquire);

        if (next.Ptr() == nullptr) {
            // Try to link new node
            TaggedPtr new_next(new_node, next.Tag() + 1);
            if (tail->next.compare_exchange_weak(next, new_next, release)) {
                // Advance tail
                tail_.compare_exchange_strong(tail, TaggedPtr(new_node, tail.Tag() + 1));
                return;
            }
        } else {
            // Help advance lagging tail
            tail_.compare_exchange_strong(tail, TaggedPtr(next.Ptr(), tail.Tag() + 1));
        }
    }
}
```

#### Sharded Priority Queue

Tiered scheduling without lock-free priority queue complexity:

```cpp
class ShardedPriorityQueue<T> {
    std::queue<T*> premium_;   // Tier 0: highest priority
    std::queue<T*> standard_;  // Tier 1: default
    std::queue<T*> batch_;     // Tier 2: lowest priority

    T* Pop() {
        if (!premium_.empty()) return premium_.pop();
        if (!standard_.empty()) return standard_.pop();
        if (!batch_.empty()) return batch_.pop();
        return nullptr;
    }
};
```

---

### Memory Subsystem

**Location:** `core/include/block_allocator.h`, `core/include/numa_allocator.h`

#### KV Block Allocator (Slab)

Eliminates heap fragmentation in long-running servers:

```cpp
class KVBlockAllocator {
    void* arena_;                 // Contiguous pre-allocated memory
    std::vector<int> free_list_;  // LIFO stack of free block IDs
    std::vector<bool> allocated_; // Bitmap for double-free detection

public:
    // O(1) allocation
    int Allocate() {
        if (free_list_.empty()) return -1;  // OOM
        int block_id = free_list_.back();
        free_list_.pop_back();
        allocated_[block_id] = true;
        return block_id;
    }

    // O(1) deallocation with double-free protection
    bool Free(int block_id) {
        if (!allocated_[block_id]) {
            std::cerr << "Double-free detected!" << std::endl;
            return false;
        }
        allocated_[block_id] = false;
        free_list_.push_back(block_id);
        return true;
    }
};
```

#### NUMA Allocator

Eliminates remote memory access penalties on multi-socket servers:

```cpp
// Preferred allocation with graceful fallback
AllocationResult AllocatePreferred(size_t bytes, size_t alignment, int preferred_node) {
    // Step 1: Try strict allocation on preferred node
    void* ptr = numa_alloc_onnode(bytes, preferred_node);
    if (ptr) return {ptr, AllocationType::Numa, true};

    // Step 2: Fallback to local node
    ptr = numa_alloc_local(bytes);
    if (ptr) return {ptr, AllocationType::Numa, false};

    // Step 3: Interleaved allocation (last resort)
    ptr = numa_alloc_interleaved(bytes);
    if (ptr) return {ptr, AllocationType::Numa, false};

    // Final fallback: posix_memalign
    return {AllocateAligned(bytes, alignment), AllocationType::Aligned, false};
}
```

#### HugePages Integration

For large models (>100B parameters), 2MB HugePages reduce TLB misses:

```cpp
void* AllocateHugePagesOnNode(size_t bytes, int numa_node) {
    // Try explicit huge pages (requires hugetlb configured)
    void* ptr = mmap(nullptr, aligned_size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

    if (ptr != MAP_FAILED) {
        BindToNumaNode(ptr, aligned_size, numa_node);
        return ptr;
    }

    // Fallback: normal mmap with THP hint
    ptr = mmap(..., MAP_PRIVATE | MAP_ANONYMOUS, ...);
    madvise(ptr, aligned_size, MADV_HUGEPAGE);
    return ptr;
}
```

#### Runtime Diagnostics

Verify NUMA placement at runtime:

```cpp
// Check if memory is actually on the requested node
DiagResult PrintSystemTopologyReport(void* ptr, size_t size, int requested_node) {
    int actual_node = GetActualNumaNode(ptr, size);  // via move_pages()
    bool nodes_match = (actual_node == requested_node);

    size_t huge_page_count;
    bool huge_pages_active = CheckHugePages(ptr, size, huge_page_count);  // via /proc/self/smaps

    if (!nodes_match) {
        std::cerr << "WARNING: Memory on wrong NUMA node! Performance degraded." << std::endl;
    }
}
```
