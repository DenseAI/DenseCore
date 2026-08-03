/**
 * @file densecore/backend/cpu_backend.h
 * @brief CPU backend implementation with UMA support for x86-64 and ARM64
 *
 * This backend provides an immediate-mode fallback for systems without
 * dedicated accelerators. All memory is inherently "unified" on CPU.
 *
 * **Features:**
 * - Runtime SIMD detection (AVX-512 → AVX2 → NEON → Scalar)
 * - 64-byte aligned memory allocation for cache efficiency
 * - Immediate-mode graph execution (records and replays ops)
 * - Hardware-accelerated quantization using SIMD intrinsics
 *
 * **UMA on CPU:**
 * Since all memory is accessible by the single processor, `AllocateUnified()`
 * is equivalent to `AllocateDevice()`. `SynchronizeMemory()` is a no-op due
 * to the strong memory model on x86 and appropriate barriers on ARM.
 *
 * @see compute_backend.h for the abstract interface
 * @see densecore/backend/accelerator_traits.h for GenericCPU profile
 */

#ifndef DENSECORE_CPU_BACKEND_H
#define DENSECORE_CPU_BACKEND_H

#include "densecore.h"
#include "densecore/hal/compute_backend.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "densecore/moe/moe_types.h"
#include "densecore/moe/profiler.h"
#include "densecore/runtime/tensor_view.h"
#include "densecore/simd/simd_ops.h"

// =============================================================================
// NUMA Debug Logging (disabled by default, enable for diagnostics)
// =============================================================================
#ifndef DENSECORE_NUMA_DEBUG
#define DENSECORE_NUMA_DEBUG 0
#endif

// Forward declaration shared across namespaces.
struct TransformerLayer;
struct BatchSpec;
struct TransformerModel;

namespace densecore {

// Forward declaration for NUMA-aware thread pool
class ThreadPool;
class OpRegistry;
class MatMulOps;
class ActivationOps;
struct LoRAAdapter;
using ::TransformerLayer;

/**
 * @brief CPU backend implementation with UMA semantics
 *
 * This backend wraps the existing AVX2/AVX-512/NEON SIMD kernels from
 * densecore/simd/simd_ops.h and provides them through the ComputeBackend interface.
 *
 * **Threading:**
 * - Uses internal thread pool for parallelization
 * - Thread count is determined by hardware concurrency or explicit config
 *
 * **Memory:**
 * - All allocations are 64-byte aligned for AVX-512 compatibility
 * - Uses platform-specific aligned allocation
 * - `AllocateUnified` == `AllocateDevice` (all CPU memory is unified)
 *
 * **Graph Capture:**
 * - Supports immediate-mode graph capture for API compatibility
 * - Records operation lambdas and replays them on `ExecuteGraph()`
 * - No compilation step (NPU-style optimization not available on CPU)
 */
class CpuBackend : public ComputeBackend {
public:
    struct GgmlQuantizedMatrixView {
        const void* data = nullptr;
        int32_t type_id = -1;
        int64_t rows = 0;
        int64_t cols = 0;
        size_t row_bytes = 0;

        bool IsValid() const { return data != nullptr && type_id >= 0 && rows > 0 && cols > 0 && row_bytes > 0; }
    };

    enum class GgmlQuantizedMatMulPath : uint8_t {
        Rejected = 0,
        Ggml,
        Q4KRawBatched,
    };

    CpuBackend();
    ~CpuBackend() override;

    // ===========================================================================
    // Backend Identification
    // ===========================================================================

    const char* Name() const override { return selected_isa_; }
    DeviceType Device() const override { return DeviceType::CPU; }

    /**
     * @brief Get CPU-specific hardware traits
     *
     * Returns GenericCPU profile:
     * - supports_unified_memory: true (all CPU memory is unified)
     * - supports_graph_execution: false (immediate mode)
     * - preferred_quantization: Q4_K (memory-bandwidth optimized)
     */
    AcceleratorTraits GetTraits() const override { return AcceleratorTraits::GenericCPU(); }

    // ===========================================================================
    // Unified Memory Management
    // ===========================================================================

    /**
     * @brief Allocate unified memory (same as device allocation on CPU)
     *
     * On CPU, all memory is inherently accessible by the single processor.
     * This method delegates to `AllocateDevice()`.
     *
     * @param size_bytes Number of bytes to allocate
     * @param alignment Alignment (default: 64 bytes for AVX-512)
     * @return Aligned memory pointer, nullptr on failure
     */
    void* AllocateUnified(size_t size_bytes, size_t alignment = 64) override {
        return AllocateDevice(size_bytes, alignment);
    }

    /**
     * @brief Synchronize memory (no-op on CPU)
     *
     * CPU has a strong memory model (x86) or uses appropriate barriers (ARM).
     * Memory writes are visible without explicit synchronization.
     */
    void SynchronizeMemory(void* ptr, size_t size_bytes, MemorySyncDirection direction) override {
        (void)ptr;
        (void)size_bytes;
        (void)direction;
        // No-op: CPU memory is always coherent
#if defined(__aarch64__) || defined(_M_ARM64)
        // ARM64: issue data memory barrier for completeness
        __asm__ __volatile__("dmb sy" ::: "memory");
#else
        // x86: compiler barrier is sufficient
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }

    void* AllocateDevice(size_t size_bytes, size_t alignment = 64) override;
    void FreeDevice(void* ptr) override;

    // ===========================================================================
    // NUMA-Aware Memory Management
    // ===========================================================================

    /**
     * @brief Bind existing memory to a specific NUMA node
     *
     * Uses mbind() on Linux to set memory policy for the given range.
     * Requires DENSECORE_HAS_NUMA. Memory should be touched after binding
     * for the policy to take effect on page faults.
     *
     * @param ptr Pointer to memory range (should be page-aligned for best results)
     * @param size_bytes Size of memory range
     * @param numa_node Target NUMA node
     * @return true on success, false if NUMA unavailable or binding failed
     */
    bool BindMemoryToNumaNode(void* ptr, size_t size_bytes, int numa_node);

    /**
     * @brief Query which NUMA node a memory address resides on
     *
     * Uses move_pages() with null destination to query current location.
     *
     * @param ptr Pointer to query
     * @return NUMA node ID, or -1 if unavailable or query failed
     */
    int QueryMemoryNumaNode(void* ptr);

    /**
     * @brief Verify that sampled, fully-owned pages in a memory range share one NUMA node
     *
     * Boundary pages are excluded because adjacent expert slices can share them.
     * Returns -1 when placement cannot be verified or samples span nodes.
     */
    int QueryMemoryNumaNodeRange(void* ptr, size_t size_bytes, int max_samples = 8);


    /**
     * @brief Copy to device (deprecated, just memcpy on CPU)
     */
    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override;

    /**
     * @brief Copy from device (deprecated, just memcpy on CPU)
     */
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override;

    // ===========================================================================
    // Graph Capture (Immediate Mode Fallback)
    // ===========================================================================

    /**
     * @brief Begin recording operations
     *
     * Creates an ImmediateModeGraph that stores operation callbacks.
     * Operations called after this will be recorded instead of executed.
     */
    void BeginCapture() override;

    /**
     * @brief End recording and return the graph
     *
     * @return ImmediateModeGraph containing recorded operation callbacks
     */
    std::unique_ptr<OperationGraph> EndCapture() override;

    /**
     * @brief Execute a recorded graph
     *
     * For CPU, this replays all recorded operations synchronously.
     *
     * @param graph The graph to execute (must be ImmediateModeGraph)
     */
    void ExecuteGraph(const OperationGraph& graph) override;

    /**
     * @brief Execute a recorded graph with NUMA-aware dispatching
     *
     * Routes all compute operations to the thread pool for the specified NUMA node.
     * Use this when the KV cache for the request is pinned to a specific node.
     *
     * @param graph The graph to execute
     * @param numa_node_id Target NUMA node (-1 for round-robin)
     */
    void ExecuteGraph(const OperationGraph& graph, int numa_node_id);

    // ===========================================================================
    // Quantization Operations
    // ===========================================================================

    /**
     * @brief Quantize tensor using SIMD-accelerated kernels
     *
     * Uses AVX2/AVX-512 (x86) or NEON (ARM) for fast quantization.
     * Computes optimal scale/zero-point if not provided.
     *
     * @param src Source FP32 tensor
     * @param dst Destination quantized tensor (pre-allocated)
     * @param type Target quantization type (INT8, Q4_K, etc.)
     */
    void Quantize(const Tensor& src, QuantizedTensorView* dst, QuantType type) override;

    /**
     * @brief Dequantize tensor back to FP32
     *
     * @param src Quantized source tensor
     * @param dst Destination FP32 tensor (pre-allocated)
     */
    void Dequantize(const QuantizedTensorView& src, Tensor* dst) override;

    // ===========================================================================
    // Core Operations (with NUMA-aware overloads)
    // ===========================================================================

    // Base class overrides (default to round-robin dispatch)
    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override;
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override;
    /**
     * Multiply F32 activations by raw GGML block-quantized weight rows.
     *
     * The weight view must describe an [N, K] row-major matrix whose rows use
     * the exact GGML type layout identified by type_id. This is intentionally
     * separate from GemmInt4, whose packed-nibble scale/zero contract is not
     * compatible with raw GGUF Q4 blocks.
     */
    DENSECORE_API bool MatMulGgmlQuantizedTransB(const Tensor& A, const GgmlQuantizedMatrixView& W, Tensor* C,
                                                 int numa_node_id = 0);
    DENSECORE_API GgmlQuantizedMatMulPath MatMulGgmlQuantizedTransBWithPath(const Tensor& A,
                                                                            const GgmlQuantizedMatrixView& W, Tensor* C,
                                                                            int numa_node_id = 0);
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                  int group_size) override;
    void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps = 1e-5f) override;
    void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                    float eps = 1e-5f) override;
    void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output,
                   float eps = 1e-5f) override;
    void Softmax(const Tensor& input, Tensor* output) override;
    void SoftmaxInplace(Tensor* data) override;
    void SiLU(const Tensor& input, Tensor* output) override;
    void GELU(const Tensor& input, Tensor* output) override;
    void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
              int rope_dim = -1) override;
    void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, Tensor* q_out,
                            Tensor* k_out, Tensor* v_out) override;
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                        bool causal = true, int n_head_kv = -1, int sliding_window = -1, float logit_softcap = 0.0f,
                        uint32_t semantic_flags = 0, int q_start_offset = 0, int kv_start_offset = 0) override;
    void Synchronize() override { /* No-op for CPU */ }

    // ===========================================================================
    // Multi-LoRA Operations
    // ===========================================================================

    void ApplyMultiLoRA(const Tensor& input, const std::string& layer_name,
                        const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map,
                        Tensor* output) override;

    // ===========================================================================
    // NUMA-Aware Compute Operations (Sticky Routing)
    // ===========================================================================
    // These overloads allow explicit NUMA node selection for data locality.
    // Use when KV cache or weights are pinned to a specific NUMA node.

    void MatMul(const Tensor& A, const Tensor& B, Tensor* C, int numa_node_id);
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C, int numa_node_id);
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                  int group_size, int numa_node_id);
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale, bool causal,
                        int n_head_kv, int sliding_window, float logit_softcap, uint32_t semantic_flags,
                        int numa_node_id, int q_start_offset, int kv_start_offset);

    // ===========================================================================
    // Multi-NUMA Thread Pool Management
    // ===========================================================================

    /**
     * @brief Get thread pool for a specific NUMA node
     *
     * @param numa_node Target NUMA node (-1 for round-robin across all pools)
     * @return Reference to the thread pool for the specified node
     */
    ThreadPool& GetThreadPool(int numa_node = -1);

    /**
     * @brief Execute a range on a persistent NUMA-local CPU thread pool.
     */
    DENSECORE_API void ParallelFor(int total_work, const std::function<void(int, int, int)>& task, int numa_node = 0);

    /**
     * @brief Get the number of NUMA nodes (thread pools)
     */
    int GetNumaNodeCount() const;

    /**
     * @brief Run task(node) for node in [0, n_tasks) concurrently
     *
     * Node 0 runs on the calling thread; node n > 0 runs on a persistent
     * helper thread pinned to that NUMA node, so nested
     * GetThreadPool(n).ParallelFor calls execute node-local. Blocks until all
     * tasks return. Degrades to sequential execution on the calling thread
     * when the dispatcher is busy (nested/concurrent use) or n_tasks exceeds
     * the pool count.
     */
    void RunConcurrentNodeTasks(int n_tasks, const std::function<void(int)>& task);
    void RunOnNumaNode(int numa_node, const std::function<void()>& task);

    // ===========================================================================
    // CPU-Specific Accessors
    // ===========================================================================

    /**
     * @brief Get detected SIMD level
     */
    simd::SimdLevel GetSimdLevel() const { return simd_level_; }

    // ===========================================================================
    // NUMA-Aware MoE Expert Rebalancing
    // ===========================================================================

    /**
     * @brief Expert weight buffer descriptor for NUMA migration
     */
    struct ExpertWeight {
        void* ptr;    ///< Pointer to weight data
        size_t size;  ///< Size in bytes
    };

    struct ExpertPackedInt4Weight {
        const uint8_t* packed_weights = nullptr;
        const float* scales = nullptr;
        const float* zeros = nullptr;
        int group_size = 0;
        int K = 0;
        int N = 0;

        bool IsValid() const {
            return packed_weights && scales && zeros && group_size > 0 && K > 0 && N > 0 && (K % group_size) == 0;
        }
    };

    /**
     * @brief Expert weights for a single MoE expert (w1, w2, w3)
     */
    struct ExpertWeights {
        ExpertWeight w1;                    ///< Gate projection [hidden, intermediate]
        ExpertWeight w2;                    ///< Down projection [intermediate, hidden]
        ExpertWeight w3;                    ///< Up projection [hidden, intermediate] (SwiGLU)
        int hidden_dim;                     ///< Model hidden dimension
        int intermediate_dim;               ///< FFN intermediate dimension
        bool use_gelu_activation = false;   ///< Gemma4-style gated GELU instead of SiLU
        bool force_safe_reference = false;  ///< Force F32 reference matmul path for parity-sensitive experts
        int w1_type = 0;                    ///< ggml_type of w1 (0 = GGML_TYPE_F32)
        int w2_type = 0;                    ///< ggml_type of w2
        int w3_type = 0;                    ///< ggml_type of w3
        const struct ggml_tensor* gate_up_tensor = nullptr;
        const struct ggml_tensor* w1_tensor = nullptr;
        const struct ggml_tensor* w2_tensor = nullptr;
        const struct ggml_tensor* w3_tensor = nullptr;
        const struct ggml_tensor* w2_scale_tensor = nullptr;
        bool uses_canonical_gemma4_packed_layout = false;
        ExpertPackedInt4Weight w1_int4;
        ExpertPackedInt4Weight w2_int4;
        ExpertPackedInt4Weight w3_int4;
    };

    struct MoERuntimeStatsSnapshot {
        uint64_t batches = 0;
        uint64_t total_active_experts = 0;
        uint64_t total_assignments = 0;
        uint64_t total_local_hot_experts = 0;
        uint64_t total_reuse_intersection = 0;
        uint64_t total_reuse_union = 0;
        uint64_t total_max_expert_batch = 0;
        uint64_t total_ordering_considered = 0;
        uint64_t total_ordering_applied = 0;
        uint64_t total_ordering_skipped_small_batch = 0;
        uint64_t total_ordering_skipped_low_reuse = 0;
        uint64_t total_ordering_numa_switches_before = 0;
        uint64_t total_ordering_numa_switches_after = 0;
        uint64_t total_prefetch_candidates = 0;
        uint64_t total_prefetch_calls = 0;
        uint64_t total_prefetch_bytes = 0;
        uint64_t total_prefetch_skipped_distance = 0;
        uint64_t total_prefetch_skipped_pressure = 0;
        uint64_t total_prefetch_skipped_signal = 0;
        uint64_t total_cached_experts = 0;
        uint64_t total_dequantized_experts = 0;
        uint64_t total_dequantized_bytes = 0;
    };

    enum class MoEProjectionPath : uint8_t {
        Unknown = 0,
        PackedInt4Fast,
        RuntimeGemmInt4,
        GgmlQuantizedVecDot,
        ReferenceF32,
        DenseF32,
    };

    struct MoEPathTraceEntry {
        int layer_idx = -1;
        int seq_id = -1;
        int token_idx = -1;
        int decode_step = -1;
        int n_past = -1;
        int expert_id = -1;
        bool force_safe_reference = false;
        bool safe_reference_mode = false;
        char projection[3] = {'?', '\0', '\0'};
        MoEProjectionPath selected_path = MoEProjectionPath::Unknown;
    };

    struct MoEForwardProfile {
        uint64_t route_ns = 0;
        uint64_t reorder_ns = 0;
        uint64_t expert_ns = 0;
        uint64_t reduce_ns = 0;
        uint64_t w1w3_ns = 0;
        uint64_t w2_ns = 0;
        uint64_t rowblock_ns = 0;
        uint64_t rowblock_w1w3_ns = 0;
        uint64_t rowblock_w2_ns = 0;
        uint64_t decode_scratch_reused = 0;
        uint64_t decode_allocations_avoided = 0;
        int rowblock_used = 0;
        int rowblock_tasks = 0;
    };

    /**
     * @brief Migrate hot expert weights to local NUMA node
     *
     * Uses move_pages() syscall to migrate memory pages of frequently-used
     * expert FFN weights to the NUMA node where inference threads are running.
     * Migrates entire weight buffers in batches to avoid syscall overhead.
     *
     * Thread safety: Acquires internal mutex during page migration.
     *
     * @param hot_expert_ids Vector of expert IDs to migrate (from ExpertProfiler::GetHotExperts())
     * @param expert_weights Vector of expert weight descriptors (ptr + size for w1/w2/w3)
     * @param n_experts Total number of experts
     * @param target_numa_node Target NUMA node (-1 for current thread's node)
     * @return Number of pages successfully migrated
     */
    int RebalanceExperts(const std::vector<int>& hot_expert_ids, const std::vector<ExpertWeights>& expert_weights,
                         int n_experts, int target_numa_node = -1);
    int RebalanceExperts(const TransformerLayer* layer_key, const std::vector<int>& hot_expert_ids,
                         const std::vector<ExpertWeights>& expert_weights, int n_experts, int target_numa_node = -1);

    /**
     * @brief Start background rebalancing thread
     *
     * Periodically profiles expert usage and optionally migrates hot expert
     * weights to the local NUMA node. When page migration is disabled
     * (default), the thread still runs to maintain profiler statistics for
     * Sticky Routing (DispatchExpertFFN dispatches compute to the NUMA node
     * where the expert's weight data already resides).
     *
     * @param interval_ms Rebalance interval in milliseconds (default: 5000)
     * @param top_k Number of hot experts to track/migrate (default: 4)
     * @param enable_page_migration If true, actively migrate hot expert weight
     *        pages via move_pages() syscall. If false (default), rely on
     *        Sticky Routing for NUMA locality. Page migration is useful when
     *        initial weight placement is suboptimal (e.g., random malloc).
     */
    void StartRebalanceThread(int interval_ms = 5000, int top_k = 4, bool enable_page_migration = false);

    /**
     * @brief Stop background rebalancing thread
     *
     * Blocks until the thread exits.
     */
    void StopRebalanceThread();

    /**
     * @brief Check if rebalancing thread is running
     */
    bool IsRebalanceThreadRunning() const { return rebalance_running_.load(); }

#ifdef DENSECORE_TEST_BUILD
    struct NumaRebalanceTestStats {
        uint64_t cycles = 0;
        uint64_t interval_decreases = 0;
        uint64_t interval_increases = 0;
        int current_interval_ms = 0;
    };

    using MovePagesTestHook =
        std::function<long(size_t count, void** pages, const int* nodes, int* status)>;
    using QueryNumaNodeRangeTestHook = std::function<int(const void* ptr, size_t size)>;

    void SetNumaRebalanceTestHooks(MovePagesTestHook move_pages_hook,
                                   QueryNumaNodeRangeTestHook query_range_hook);
    void ClearNumaRebalanceTestHooks();
    bool IsNumaRebalanceDisabledForTest() const { return rebalance_disabled_.load(); }
    std::vector<int> GetLocalExpertIdsForTest(const TransformerLayer* layer_key) const;
    NumaRebalanceTestStats GetNumaRebalanceTestStats() const;
#endif

    /**
     * @brief Initialize MoE expert profiler
     *
     * @param n_experts Number of experts in the MoE layer
     * @param ema_alpha EMA smoothing factor (default: 0.1)
     */
    void InitMoEProfiler(int n_experts, float ema_alpha = 0.1f);
    void InitMoEProfiler(const TransformerLayer* layer_key, int n_experts, float ema_alpha = 0.1f);

    /**
     * @brief Get profiler for recording hits during inference
     */
    moe::ExpertProfiler* GetProfiler() { return GetProfiler(nullptr); }
    moe::ExpertProfiler* GetProfiler(const TransformerLayer* layer_key);
    int GetExpertNumaNode(const TransformerLayer* layer_key, int expert_id) const;
    bool CopyExpertNumaNodes(const TransformerLayer* layer_key, int n_experts, std::vector<int>* nodes) const;

    /**
     * @brief Record expert accesses from MoE routing (hot path)
     *
     * Call this after MoE routing to update profiler statistics.
     * Thread-safe and lock-free for minimal inference overhead.
     *
     * @param expert_ids Array of selected expert IDs
     * @param count Number of expert IDs
     */
    void RecordExpertAccess(const int* expert_ids, int count);
    void RecordExpertAccess(const TransformerLayer* layer_key, const int* expert_ids, int count);

    /**
     * @brief Register MoE expert weights for NUMA rebalancing
     *
     * Call this after model loading to enable automatic NUMA migration
     * of hot expert weights.
     *
     * @param experts Vector of expert weights (one per expert)
     */
    void RegisterMoEExperts(const std::vector<ExpertWeights>& experts);
    void RegisterMoEExperts(const TransformerLayer* layer_key, const std::vector<ExpertWeights>& experts);

    /**
     * @brief Get registered expert weights (thread-safe)
     */
    std::vector<ExpertWeights> GetRegisteredExperts() const;
    std::vector<ExpertWeights> GetRegisteredExperts(const TransformerLayer* layer_key) const;
    bool GetRegisteredExpertsView(const TransformerLayer* layer_key, const ExpertWeights** experts, int* count) const;

    // =========================================================================
    // Sticky Routing: NUMA-Aware Expert FFN Dispatch
    // =========================================================================

    /**
     * @brief Dispatch expert FFN computation to the optimal NUMA node
     *
     * This is the core "Sticky Routing" API. It queries the profiler for the
     * NUMA node where the expert's weights reside, then routes the FFN
     * computation to that node's thread pool for optimal data locality.
     *
     * If the expert has no assigned NUMA node (hasn't been migrated), falls
     * back to round-robin dispatch.
     *
     * @param expert_id Expert index (0 to n_experts-1)
     * @param input Input tensor (activations from router gate)
     * @param w1 Expert FFN w1 weights (gate projection)
     * @param w2 Expert FFN w2 weights (down projection)
     * @param w3 Expert FFN w3 weights (up projection, for SwiGLU)
     * @param output Output tensor (pre-allocated)
     */
    void DispatchExpertFFN(int expert_id, const Tensor& input, const ExpertWeights& expert, const Tensor& w1,
                           const Tensor& w2, const Tensor& w3, Tensor* output);
    void DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                           const ExpertWeights& expert, const Tensor& w1, const Tensor& w2, const Tensor& w3,
                           Tensor* output);
    void DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2, const Tensor& w3,
                           Tensor* output);
    void DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input, const Tensor& w1,
                           const Tensor& w2, const Tensor& w3, Tensor* output);

    /**
     * @brief Full MoE layer forward pass
     *
     * Performs the complete MoE forward:
     * 1. Record expert accesses in profiler (batch-level)
     * 2. Group tokens by expert for efficient dispatch
     * 3. Dispatch each expert's tokens using DispatchExpertFFN (NUMA-aware)
     * 4. Combine expert outputs with routing weights
     *
     * @param input Input hidden states [batch, hidden_dim]
     * @param routing Routing result from MoETopKRoute
     * @param experts Vector of expert weights (one per expert)
     * @param output Output tensor [batch, hidden_dim] (pre-allocated)
     */
    void ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing, const std::vector<ExpertWeights>& experts,
                    Tensor* output);
    void ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                    const std::vector<ExpertWeights>& experts, Tensor* output);
    void ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch, const Tensor& input,
                    const moe::MoERouteResult& routing, const std::vector<ExpertWeights>& experts, Tensor* output);
    void ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                    const ExpertWeights* experts, int num_experts, Tensor* output);
    void ForwardMoE(const TransformerModel* model, const TransformerLayer* layer_key, int layer_idx,
                    const BatchSpec* batch, const Tensor& input, const moe::MoERouteResult& routing,
                    const ExpertWeights* experts, int num_experts, Tensor* output,
                    MoEForwardProfile* profile = nullptr);
    void ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch, const Tensor& input,
                    const moe::MoERouteResult& routing, const ExpertWeights* experts, int num_experts, Tensor* output);
    MoERuntimeStatsSnapshot GetMoERuntimeStatsSnapshot() const;
    void ResetMoEPathTrace();
    std::vector<MoEPathTraceEntry> GetMoEPathTraceSnapshot() const;
    void RecordMoEPathTrace(const MoEPathTraceEntry& entry);
    uint64_t GetMoEForwardInvocationCount() const;

    // ===========================================================================
    // Dependency Injection (optional)
    // ===========================================================================
    void SetOpRegistry(OpRegistry* registry) {
        op_registry_ = registry;
        // Invalidate cached dispatch pointers when registry changes
        cached_matmul_ops_ = nullptr;
        cached_activation_ops_ = nullptr;
        ops_resolved_ = false;
    }

private:
    class CpuMemoryManager;
    class CpuThreadManager;
    OpRegistry& GetOpRegistry();
    const OpRegistry& GetOpRegistry() const;
    MatMulOps* ResolveMatMulOps();

    class CaptureGuard {
    public:
        explicit CaptureGuard(CpuBackend* backend);
        ~CaptureGuard();

    private:
        CpuBackend* backend_;
        bool was_capturing_;
    };

    ImmediateModeGraph* GetCaptureGraph();

    struct MoELayerRegistry {
        struct DequantizedExpertCacheEntry {
            int expert_id = -1;
            size_t bytes = 0;
            uint64_t last_used = 0;
            simd::AlignedVector<float> w1;
            simd::AlignedVector<float> w2;
            simd::AlignedVector<float> w3;
            Tensor w1_tensor;
            Tensor w2_tensor;
            Tensor w3_tensor;
        };

        std::vector<ExpertWeights> experts;
        std::shared_ptr<moe::ExpertProfiler> profiler;
        std::vector<int> local_expert_ids;
        std::vector<int> last_batch_experts;
        std::unordered_map<int, std::shared_ptr<DequantizedExpertCacheEntry>> dequant_cache;
        size_t dequant_cache_bytes = 0;
        uint64_t dequant_cache_use_counter = 0;
        float ema_alpha = 0.1f;
        std::mutex mutex;
    };

    std::shared_ptr<MoELayerRegistry> GetMoELayerRegistry(const TransformerLayer* layer_key) const;
    std::shared_ptr<MoELayerRegistry> GetOrCreateMoELayerRegistry(const TransformerLayer* layer_key, int n_experts,
                                                                  float ema_alpha);
    int RebalanceExpertsInternal(const std::vector<int>& hot_expert_ids,
                                 const std::vector<ExpertWeights>& expert_weights, int n_experts, int target_numa_node,
                                 moe::ExpertProfiler* profiler, std::vector<int>* local_expert_ids);

    const char* selected_isa_;    ///< Human-readable ISA name
    simd::SimdLevel simd_level_;  ///< Detected SIMD level

    OpRegistry* op_registry_ = nullptr;
    std::unique_ptr<CpuMemoryManager> memory_manager_;
    std::unique_ptr<CpuThreadManager> thread_manager_;

    // NUMA rebalancing state
    std::thread rebalance_thread_;
    std::atomic<bool> rebalance_running_{false};
    std::atomic<bool> rebalance_stop_{false};
    std::atomic<bool> rebalance_disabled_{false};
    mutable std::mutex rebalance_mutex_;  ///< Protects page migration
#ifdef DENSECORE_TEST_BUILD
    MovePagesTestHook move_pages_test_hook_;
    QueryNumaNodeRangeTestHook query_numa_node_range_test_hook_;
    std::atomic<uint64_t> rebalance_test_cycles_{0};
    std::atomic<uint64_t> rebalance_test_interval_decreases_{0};
    std::atomic<uint64_t> rebalance_test_interval_increases_{0};
    std::atomic<int> rebalance_test_current_interval_ms_{0};
#endif

    // MoE Expert Registries (one per layer)
    std::unordered_map<const TransformerLayer*, std::shared_ptr<MoELayerRegistry>> moe_registries_;
    mutable std::mutex registry_mutex_;  ///< Protects MoE registry map
    std::atomic<uint64_t> moe_stats_batches_{0};
    std::atomic<uint64_t> moe_stats_total_active_experts_{0};
    std::atomic<uint64_t> moe_stats_total_assignments_{0};
    std::atomic<uint64_t> moe_stats_total_local_hot_experts_{0};
    std::atomic<uint64_t> moe_stats_total_reuse_intersection_{0};
    std::atomic<uint64_t> moe_stats_total_reuse_union_{0};
    std::atomic<uint64_t> moe_stats_total_max_expert_batch_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_considered_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_applied_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_skipped_small_batch_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_skipped_low_reuse_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_numa_switches_before_{0};
    std::atomic<uint64_t> moe_stats_total_ordering_numa_switches_after_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_candidates_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_calls_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_bytes_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_skipped_distance_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_skipped_pressure_{0};
    std::atomic<uint64_t> moe_stats_total_prefetch_skipped_signal_{0};
    std::atomic<uint64_t> moe_stats_total_cached_experts_{0};
    std::atomic<uint64_t> moe_stats_total_dequantized_experts_{0};
    std::atomic<uint64_t> moe_stats_total_dequantized_bytes_{0};
    mutable std::mutex moe_path_trace_mutex_;
    std::vector<MoEPathTraceEntry> moe_path_trace_;
    std::atomic<uint64_t> moe_forward_invocation_count_{0};
    std::unordered_map<int, int> moe_decode_step_ordinals_;
    std::unordered_map<int, int> moe_decode_last_n_past_;

    // Cached OpRegistry dispatch pointers (resolved on first use)
    mutable MatMulOps* cached_matmul_ops_ = nullptr;
    mutable ActivationOps* cached_activation_ops_ = nullptr;
    mutable bool ops_resolved_ = false;

    // Thread pools now managed by CpuThreadManager
};

/**
 * @brief Get global CPU backend instance (singleton)
 *
 * Thread-safe initialization using C++11 magic statics.
 * The singleton is created on first use and destroyed at program exit.
 */
DENSECORE_API CpuBackend& GetCpuBackend();
DENSECORE_API CpuBackend& GetTelemetryCpuBackend();
DENSECORE_API void UpdateBackendThreads(int n_threads);
DENSECORE_API int GetBackendThreadCount();

}  // namespace densecore

#endif  // DENSECORE_CPU_BACKEND_H
