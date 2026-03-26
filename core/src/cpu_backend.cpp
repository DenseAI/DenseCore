/**
 * @file cpu_backend.cpp
 * @brief Cross-platform CPU backend implementation
 *
 * This file implements the CpuBackend class, providing optimized kernels
 * for all supported CPU architectures:
 * - x86_64: AVX-512, AVX2, SSE4.1 (Intel/AMD)
 * - ARM64: SVE, NEON with DOTPROD/FP16 (AWS Graviton, Ampere)
 * - Apple Silicon: Accelerate framework with AMX (M1/M2/M3)
 *
 * Includes a resizable static thread pool for parallel MatMul execution during
 * both decode (M=1) and prefill (M>1) phases.
 */


#include "../include/cpu_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../include/flash_attention.h"
#include "../include/gemm_config.h"
#include "../include/hardware_topology.h"
#include "../include/inference.h"  // For InferenceConfig
#include "../include/lora_storage.h"
#include "../include/matmul_backend.h"
#include "../include/optimization_bridge.h"
#include "../include/simd_ops.h"
#include "../include/simd_platform.h"
#ifdef __APPLE__
#include "../include/apple_silicon.h"
#endif

#include "densecore/exceptions.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/typed_tensor.h"
#include "densecore/numa_routing.h"
#include "ggml.h"
#include "kernels/hwy/hwy_kernels.h"
#include "moe/moe_routing.h"
#include "quantized_tensor.h"
#include "thread_pool_impl.h"

// For _mm_pause() / yield intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <malloc.h>  // For _aligned_malloc/_aligned_free
#else
#include <cstdlib>  // For aligned_alloc/free
#endif

// Linux-specific headers for Transparent Huge Pages (THP) support
#if defined(__linux__)
#include <sys/mman.h>  // For madvise() with MADV_HUGEPAGE
#endif

// =============================================================================
// Apple Accelerate Framework (AMX-backed BLAS on Apple Silicon)
// =============================================================================
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Linux-specific headers
#if defined(__linux__)
#include <cerrno>
#endif

// Linux-specific NUMA headers (only include when available)
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace densecore {

#ifndef DENSECORE_THREADPOOL_DEBUG
#define DENSECORE_THREADPOOL_DEBUG 0
#endif

namespace {
int DetectCurrentNumaNodeHint();
int ResolveEnterpriseThreadPoolNode(int requested_node, int pool_count);
}  // namespace

// =============================================================================
// CpuBackend Composed Managers
// =============================================================================

class CpuBackend::CpuMemoryManager {
public:
    void* Allocate(size_t size_bytes, size_t alignment) {
        if (size_bytes == 0) {
            return nullptr;
        }

        void* ptr = nullptr;

        // =========================================================================
        // THP (Transparent Huge Pages) Optimization for Linux
        // =========================================================================
        // Mitigating TLB thrashing for large KV caches:
        //
        // When running large LLM inference workloads (e.g., 70B models with >40GB
        // KV cache), standard 4KB pages cause excessive TLB misses and page walks
        // during token generation. By using 2MB huge pages, we reduce TLB pressure
        // by 512x (4KB -> 2MB), significantly improving memory access latency.
        //
        // Strategy:
        // 1. For allocations >= 2MB, use 2MB alignment (huge page boundary)
        // 2. Call madvise(MADV_HUGEPAGE) to advise kernel to use THP
        // 3. Maintain minimum 64-byte alignment for AVX-512 compatibility
        // =========================================================================

        // Huge page threshold: 2MB (standard Linux huge page size)
        constexpr size_t kHugePageSize = 2 * 1024 * 1024;  // 2MB
        constexpr size_t kMinSimdAlignment = 64;           // AVX-512 requirement

#if defined(_WIN32)
        // Windows: Use _aligned_malloc (VirtualAlloc with large pages requires
        // special privileges, so we keep standard allocation for now)
        ptr = _aligned_malloc(size_bytes, alignment);

#elif defined(__linux__)
        // Linux: Enable THP for large allocations to reduce TLB thrashing
        size_t effective_alignment = alignment;

        // For large allocations, upgrade to 2MB alignment for huge page support
        if (size_bytes >= kHugePageSize) {
            effective_alignment = std::max(alignment, kHugePageSize);
        } else {
            // Ensure minimum SIMD alignment even for smaller allocations
            effective_alignment = std::max(alignment, kMinSimdAlignment);
        }

        // C11 aligned_alloc requires size to be a multiple of alignment
        size_t aligned_size = ((size_bytes + effective_alignment - 1) / effective_alignment) * effective_alignment;

        // Use posix_memalign for better compatibility with large alignments
        // (std::aligned_alloc may have issues with alignments > page size on some systems)
        int result = posix_memalign(&ptr, effective_alignment, aligned_size);
        if (result != 0) {
            ptr = nullptr;
        }

        // Advise kernel to back this region with Transparent Huge Pages
        if (ptr != nullptr && size_bytes >= kHugePageSize) {
            // MADV_HUGEPAGE: Advise that this memory region would benefit from
            // huge pages. The kernel will attempt to use THP when available.
            // This is a hint, not a guarantee - gracefully degrades if THP is disabled.
            int madvise_result = madvise(ptr, aligned_size, MADV_HUGEPAGE);
            (void)madvise_result;  // Ignore return value - THP is best-effort

#if DENSECORE_THP_DEBUG
            if (madvise_result == 0) {
                std::cerr << "[THP] Enabled huge pages for " << (aligned_size / kHugePageSize) << " MB allocation"
                          << std::endl;
            } else {
                std::cerr << "[THP] madvise(MADV_HUGEPAGE) failed (errno=" << errno << ")" << std::endl;
            }
#endif
        }

#else
        // macOS and other POSIX systems: Use standard aligned_alloc
        // Note: macOS uses a different huge page mechanism (VM_FLAGS_SUPERPAGE_SIZE_2MB)
        // which requires vm_allocate() - not implemented here for simplicity.
        size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;
        ptr = std::aligned_alloc(alignment, aligned_size);
#endif

        if (!ptr) {
            throw OutOfMemoryException("CpuBackend::AllocateDevice failed");
        }
        return ptr;
    }

    void Free(void* ptr) {
        if (ptr == nullptr) {
            return;
        }
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    void CopyToDevice(void* dst, const void* src, size_t size_bytes) {
        if (dst && src && size_bytes > 0) {
            std::memcpy(dst, src, size_bytes);
        }
    }

    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) {
        if (dst && src && size_bytes > 0) {
            std::memcpy(dst, src, size_bytes);
        }
    }

    bool BindMemoryToNumaNode(void* ptr, size_t size_bytes, int numa_node) {
        if (!ptr || size_bytes == 0 || numa_node < 0) {
            return false;
        }
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
        // Check if NUMA is available at runtime
        if (numa_available() < 0) {
            return false;
        }

        // Validate numa_node is within range
        int max_node = numa_max_node();
        if (numa_node > max_node) {
            return false;
        }

        // Create nodemask with only the target node set
        struct bitmask* nodemask = numa_allocate_nodemask();
        if (!nodemask) {
            return false;
        }
        numa_bitmask_setbit(nodemask, numa_node);

        // Align pointer to page boundary
        const long page_size = sysconf(_SC_PAGESIZE);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned_addr = addr & ~(page_size - 1);
        size_t aligned_size = size_bytes + (addr - aligned_addr);

        // Bind memory to the specified node
        // MPOL_BIND: strictly allocate on specified nodes
        // MPOL_MF_MOVE: migrate existing pages to the node
        int result = mbind(reinterpret_cast<void*>(aligned_addr), aligned_size, MPOL_BIND, nodemask->maskp,
                           nodemask->size, MPOL_MF_MOVE | MPOL_MF_STRICT);

        numa_free_nodemask(nodemask);

#if DENSECORE_NUMA_DEBUG
        if (result == 0) {
            std::cerr << "[NUMA] Bound " << (aligned_size / 1024) << " KB to node " << numa_node << std::endl;
        } else {
            std::cerr << "[NUMA] mbind failed (errno=" << errno << "): " << strerror(errno) << std::endl;
        }
#endif

        return result == 0;
#else
        (void)ptr;
        (void)size_bytes;
        (void)numa_node;
        return false;
#endif
    }

    int QueryMemoryNumaNode(void* ptr) {
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
        if (!ptr) {
            return -1;
        }

        if (numa_available() < 0) {
            return -1;
        }

        int status = -1;
        // Use move_pages with null nodes array to query current location
        long result = move_pages(0, 1, &ptr, nullptr, &status, 0);

        if (result == 0 && status >= 0) {
#if DENSECORE_NUMA_DEBUG
            std::cerr << "[NUMA] Memory at " << ptr << " is on node " << status << std::endl;
#endif
            return status;
        }

        return -1;
#else
        return -1;
#endif
    }
};

class CpuBackend::CpuThreadManager {
public:
    CpuThreadManager() {
        int numa_count = HardwareTopology::GetInstance().GetNumaNodeCount();
        if (numa_count <= 0) {
            numa_count = 1;
        }
        thread_pools_.reserve(numa_count);
        for (int node = 0; node < numa_count; ++node) {
            thread_pools_.push_back(std::make_unique<ThreadPool>(node));
        }
    }

    ~CpuThreadManager() { Shutdown(); }

    void Shutdown() {
        for (auto& pool : thread_pools_) {
            if (pool) {
                pool->Shutdown();
            }
        }
        thread_pools_.clear();
    }

    ThreadPool& GetThreadPool(int numa_node) {
        if (thread_pools_.empty()) {
            thread_pools_.push_back(std::make_unique<ThreadPool>(0));
        }
        const int enterprise_node = ResolveEnterpriseThreadPoolNode(numa_node, static_cast<int>(thread_pools_.size()));
        if (enterprise_node >= 0 && enterprise_node < static_cast<int>(thread_pools_.size())) {
            return *thread_pools_[enterprise_node];
        }
        if (numa_node < 0 || numa_node >= static_cast<int>(thread_pools_.size())) {
            int idx =
                round_robin_counter_.fetch_add(1, std::memory_order_relaxed) % static_cast<int>(thread_pools_.size());
            return *thread_pools_[idx];
        }
        return *thread_pools_[numa_node];
    }

    int GetNumaNodeCount() const { return static_cast<int>(thread_pools_.size()); }

private:
    std::vector<std::unique_ptr<ThreadPool>> thread_pools_;
    std::atomic<int> round_robin_counter_{0};
};

CpuBackend::CaptureGuard::CaptureGuard(CpuBackend* backend) : backend_(backend), was_capturing_(backend->capturing_) {
    backend_->capturing_ = false;
}

CpuBackend::CaptureGuard::~CaptureGuard() {
    backend_->capturing_ = was_capturing_;
}

ImmediateModeGraph* CpuBackend::GetCaptureGraph() {
    if (!capturing_) {
        return nullptr;
    }
    if (!captured_graph_) {
        captured_graph_ = std::make_unique<ImmediateModeGraph>();
    }
    return captured_graph_.get();
}

namespace {

int DetectCurrentNumaNodeHint() {
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
    if (numa_available() < 0) {
        return -1;
    }
    const int cpu = sched_getcpu();
    if (cpu < 0) {
        return -1;
    }
    const int node = numa_node_of_cpu(cpu);
    return node >= 0 ? node : -1;
#else
    return -1;
#endif
}

int ResolveEnterpriseThreadPoolNode(int requested_node, int pool_count) {
    if (pool_count <= 0) {
        return -1;
    }

    const auto* vt = DenseCoreEntGetNumaRoutingVTable();
    if (!vt || !vt->resolve_node) {
        return -1;
    }

    DenseCoreEntNumaRoutingRequest request{};
    request.requested_node = requested_node;
    request.current_node = DetectCurrentNumaNodeHint();
    request.available_nodes = pool_count;

    DenseCoreEntNumaRoutingDecision decision{};
    if (vt->resolve_node(&request, &decision) != 0) {
        return -1;
    }
    if (decision.target_node < 0 || decision.target_node >= pool_count) {
        return -1;
    }
    return decision.target_node;
}

// Fast exp approximation for activation hot paths.
static inline float FastExp(float x) {
    // Clamp to avoid overflow in the polynomial path.
    if (x < -50.0f) x = -50.0f;
    if (x > 50.0f) x = 50.0f;

    // Approximate exp(x) via exp2(x * log2e) with a small polynomial on the mantissa.
    constexpr float kLog2E = 1.4426950408889634f;
    const float y = x * kLog2E;
    const int32_t i = std::floor(y);
    const float f = y - static_cast<float>(i);

    const float p = 1.0f + f * (0.6960656421638072f + f * (0.224494337302845f + f * 0.07944023841053369f));

    const int32_t exp_bits = (i + 127) << 23;
    float two_i = 0.0f;
    std::memcpy(&two_i, &exp_bits, sizeof(two_i));
    return two_i * p;
}

static const float* GetLoRAWeightF32(const ggml_tensor* tensor, std::vector<float>& scratch) {
    if (!tensor || !tensor->data) {
        return nullptr;
    }

    const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
    if (elements == 0) {
        return nullptr;
    }

    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float*>(tensor->data);
    }

    if (tensor->type == GGML_TYPE_F16) {
        scratch.resize(elements);
        simd::ConvertF16ToF32(scratch.data(), static_cast<const ggml_fp16_t*>(tensor->data), elements);
        return scratch.data();
    }

    return nullptr;
}

}  // namespace

// =============================================================================
// RESIZABLE THREAD POOL IMPLEMENTATION (NUMA-AWARE)
// =============================================================================
// A lightweight thread pool that supports dynamic resizing and NUMA-aware
// thread pinning for performance tuning on multi-socket systems.
// =============================================================================

/**
 * @brief NUMA-aware thread pool for CpuBackend parallelism
 *
 * Features:
 * - Constructor takes numa_node for NUMA-aware thread pinning
 * - Lazy initialization (threads created on first use)
 * - Dynamic resizing via Configure() for hardware tuning
 * - Barrier-based synchronization for parallel_for
 * - Thread-safe reconfiguration
 */
// ThreadPool implementation moved to thread_pool_impl.h

// =============================================================================
// Public API for Thread Pool Configuration
// =============================================================================

/**
 * @brief Update the number of threads used by the CPU backend
 *
 * This allows runtime tuning of parallelism for different hardware:
 * - AVX2 systems may benefit from more threads to compensate for lower IPC
 * - AVX-512 systems may use fewer threads for better cache locality
 *
 * @param n_threads Number of threads to use (<=0 for auto-detect)
 */
void UpdateBackendThreads(int n_threads) {
    if (n_threads <= 0) {
        auto& topo = HardwareTopology::GetInstance();
        n_threads = topo.GetPhysicalCoreCount();
        if (n_threads <= 0) {
            n_threads = topo.GetLogicalCoreCount();
        }
        if (n_threads <= 0) {
            n_threads = simd::GetNumCores();
        }
    }

    InferenceConfig::Instance().num_threads = n_threads;

    // Configure all pools in the global backend using public interface
    auto& backend = GetCpuBackend();
    int numa_count = backend.GetNumaNodeCount();
    for (int i = 0; i < numa_count; ++i) {
        backend.GetThreadPool(i).Configure(n_threads);
    }
}

/**
 * @brief Get the current number of threads used by the CPU backend
 */
int GetBackendThreadCount() {
    auto& backend = GetCpuBackend();
    if (backend.GetNumaNodeCount() == 0) return 0;
    return backend.GetThreadPool(0).GetNumThreads();
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

CpuBackend::CpuBackend() {
    // Detect SIMD level at runtime
    simd_level_ = simd::DetectSimdLevel();
    selected_isa_ = simd::SimdLevelName(simd_level_);

    // Initialize FP8 lookup tables
    hwy_kernels::InitFP8LUTs_Hwy();

    op_registry_ = &OpRegistry::Instance();
    memory_manager_ = std::make_unique<CpuMemoryManager>();
    thread_manager_ = std::make_unique<CpuThreadManager>();

    std::cout << "[CpuBackend] Initialized " << thread_manager_->GetNumaNodeCount()
              << " thread pool(s) with SIMD: " << selected_isa_ << std::endl;
}

CpuBackend::~CpuBackend() {
#if DENSECORE_THREADPOOL_DEBUG
    std::cerr << "[DEBUG] Destroying CpuBackend start" << std::endl;
#endif
    // Stop the NUMA rebalance thread if running to prevent dangling this pointer
    StopRebalanceThread();
    if (thread_manager_) {
        thread_manager_->Shutdown();
    }
#if DENSECORE_THREADPOOL_DEBUG
    std::cerr << "[DEBUG] Destroying CpuBackend end" << std::endl;
#endif
}

// =============================================================================
// Multi-NUMA Thread Pool Management
// =============================================================================

ThreadPool& CpuBackend::GetThreadPool(int numa_node) {
    return thread_manager_->GetThreadPool(numa_node);
}

int CpuBackend::GetNumaNodeCount() const {
    return thread_manager_ ? thread_manager_->GetNumaNodeCount() : 0;
}

OpRegistry& CpuBackend::GetOpRegistry() {
    return op_registry_ ? *op_registry_ : OpRegistry::Instance();
}

const OpRegistry& CpuBackend::GetOpRegistry() const {
    return op_registry_ ? *op_registry_ : OpRegistry::Instance();
}

// =============================================================================
// Memory Management
// =============================================================================

void* CpuBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
    return memory_manager_->Allocate(size_bytes, alignment);
}

void CpuBackend::FreeDevice(void* ptr) {
    memory_manager_->Free(ptr);
}

void CpuBackend::CopyToDevice(void* dst, const void* src, size_t size_bytes) {
    memory_manager_->CopyToDevice(dst, src, size_bytes);
}

void CpuBackend::CopyFromDevice(void* dst, const void* src, size_t size_bytes) {
    memory_manager_->CopyFromDevice(dst, src, size_bytes);
}

// =============================================================================
// NUMA-Aware Memory Binding
// =============================================================================

bool CpuBackend::BindMemoryToNumaNode(void* ptr, size_t size_bytes, int numa_node) {
    return memory_manager_->BindMemoryToNumaNode(ptr, size_bytes, numa_node);
}

int CpuBackend::QueryMemoryNumaNode(void* ptr) {
    return memory_manager_->QueryMemoryNumaNode(ptr);
}

// =============================================================================
// Graph Capture (Immediate Mode Fallback)
// =============================================================================

void CpuBackend::BeginCapture() {
    if (capturing_) {
        throw InvalidArgumentException("CpuBackend::BeginCapture called while already capturing");
    }
    capturing_ = true;
    captured_graph_ = std::make_unique<ImmediateModeGraph>();
}

std::unique_ptr<OperationGraph> CpuBackend::EndCapture() {
    if (!capturing_) {
        throw InvalidArgumentException("CpuBackend::EndCapture called without BeginCapture");
    }
    capturing_ = false;
    return std::move(captured_graph_);
}

void CpuBackend::ExecuteGraph(const OperationGraph& graph) {
    // Delegate to NUMA-aware version with round-robin dispatch
    ExecuteGraph(graph, -1);
}

void CpuBackend::ExecuteGraph(const OperationGraph& graph, int numa_node_id) {
    //
    // This file is part of DenseCore Reference Implementation.
    // Licensed under Apache 2.0 (Open Source) or Commercial License.
    //
    // This graph replay demonstrates the Graph API without exposing
    // proprietary graph compilation logic (used in Metal/ANE backends).
    //
    (void)numa_node_id;

    // Fast path: ImmediateModeGraph stores lambdas and replays directly
    // Note: ImmediateModeGraph captures don't support NUMA routing (lambdas are pre-bound)
    // Use polymorphic Execute() - if graph has recorded ops, they'll be replayed
    graph.Execute();
    if (graph.NodeCount() == 0) {
        // No nodes means this is likely an ImmediateModeGraph with recorded callbacks
        return;
    }

    // Generic graph replay: iterate nodes and dispatch via OpRegistry
    for (size_t i = 0; i < graph.NodeCount(); ++i) {
        const GraphNode& node = graph.GetNode(i);

        std::vector<Tensor> input_tensors;
        std::vector<Tensor*> inputs;
        input_tensors.reserve(node.inputs.size());
        inputs.reserve(node.inputs.size());
        for (size_t idx : node.inputs) {
            input_tensors.push_back(graph.GetTensor(idx));
            inputs.push_back(&input_tensors.back());
        }

        std::vector<Tensor> output_tensors;
        std::vector<Tensor*> outputs;
        output_tensors.reserve(node.outputs.size());
        outputs.reserve(node.outputs.size());
        for (size_t idx : node.outputs) {
            output_tensors.push_back(graph.GetTensor(idx));
            outputs.push_back(&output_tensors.back());
        }

        OpRegistry::SelectionCriteria criteria;
        const Tensor* criteria_tensor = nullptr;
        if (!inputs.empty() && inputs[0]) {
            criteria_tensor = inputs[0];
        } else if (!outputs.empty() && outputs[0]) {
            criteria_tensor = outputs[0];
        }
        if (criteria_tensor) {
            criteria.dtype = criteria_tensor->dtype;
            criteria.layout = criteria_tensor->layout;
            if (criteria_tensor->ndim > 0) {
                criteria.batch = criteria_tensor->shape[0];
            }
            criteria.shape.assign(criteria_tensor->shape.begin(), criteria_tensor->shape.end());
        }

        DenseCoreOp* op = GetOpRegistry().GetBest(node.op, DeviceType::CPU, criteria);
        if (!op) {
            throw OperationNotSupportedException(std::string("ExecuteGraph missing kernel for op '") +
                                                 OpTypeName(node.op) + "' at node " + std::to_string(i));
        }

        const void* params = nullptr;
        std::visit(
            [&params](const auto& p) {
                using T = std::decay_t<decltype(p)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    params = &p;
                }
            },
            node.params);

        op->Execute(inputs, outputs, params);
    }
}
// =============================================================================
// Quantization Operations
// =============================================================================

void CpuBackend::Quantize(const Tensor& src, QuantizedTensorView* dst, QuantType type) {
    if (!src.IsValid() || !dst) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        graph->RecordOperation([this, src, dst, type]() {
            CaptureGuard guard(this);
            Quantize(src, dst, type);
        });
        return;
    }

    const float* src_data = src.DataAs<float>();
    const int64_t n_elements = src.NumElements();

    switch (type) {
    case QuantType::INT8: {
        // Symmetric INT8 quantization
        // Find max absolute value
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n_elements; ++i) {
            float abs_val = std::fabs(src_data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        // Compute scale
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) {
            scale = 1.0f;  // Avoid division by zero
        }

        // Quantize
        int8_t* dst_data = static_cast<int8_t*>(dst->data);
        float inv_scale = 1.0f / scale;

#if defined(__AVX2__)
        // AVX2 vectorized quantization
        const __m256 v_scale = _mm256_set1_ps(inv_scale);
        const __m256 v_min = _mm256_set1_ps(-127.0f);
        const __m256 v_max = _mm256_set1_ps(127.0f);

        int64_t i = 0;
        for (; i + 8 <= n_elements; i += 8) {
            __m256 v_src = _mm256_loadu_ps(src_data + i);
            __m256 v_scaled = _mm256_mul_ps(v_src, v_scale);
            v_scaled = _mm256_max_ps(_mm256_min_ps(v_scaled, v_max), v_min);
            __m256i v_int = _mm256_cvtps_epi32(v_scaled);

            // Pack to int8 (AVX2 doesn't have direct pack to i8, so we do it
            // manually)
            alignas(32) int32_t temp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v_int);
            for (int j = 0; j < 8; ++j) {
                dst_data[i + j] = static_cast<int8_t>(temp[j]);
            }
        }

        // Scalar tail
        for (; i < n_elements; ++i) {
            float val = src_data[i] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            dst_data[i] = static_cast<int8_t>(std::round(val));
        }
#else
        // Scalar fallback
        for (int64_t i = 0; i < n_elements; ++i) {
            float val = src_data[i] * inv_scale;
            val = std::max(-127.0f, std::min(127.0f, val));
            dst_data[i] = static_cast<int8_t>(std::round(val));
        }
#endif

        // Update quantization params
        dst->type = QuantType::INT8;
        dst->quant_params = QuantizationParams::PerTensor(scale, 0);
        break;
    }

    case QuantType::Q4_0:
    case QuantType::Q4_K: {
        // Block-wise 4-bit quantization (simplified)
        // For production, use GGML's optimized quantization routines
        throw NotImplementedException("Quantize Q4 is not implemented (use GGML model loader for pre-quantized)");
    }

    default: throw QuantizationException(std::string("Unsupported quantization type ") + QuantTypeName(type));
    }
}

void CpuBackend::Dequantize(const QuantizedTensorView& src, Tensor* dst) {
    if (!src.IsValid() || !dst || !dst->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        graph->RecordOperation([this, src, dst]() {
            CaptureGuard guard(this);
            Dequantize(src, dst);
        });
        return;
    }

    const int64_t n_elements = src.NumElements();
    float* dst_data = dst->DataAs<float>();

    switch (src.type) {
    case QuantType::INT8: {
        const int8_t* src_data = static_cast<const int8_t*>(src.data);
        float scale = 1.0f;

        if (src.quant_params.granularity == QuantizationParams::Granularity::PerTensor) {
            scale = src.quant_params.tensor.scale;
        }

#if defined(__AVX2__)
        const __m256 v_scale = _mm256_set1_ps(scale);
        int64_t i = 0;
        for (; i + 8 <= n_elements; i += 8) {
            // Load 8 int8 values and convert to float
            alignas(32) float temp[8];
            for (int j = 0; j < 8; ++j) {
                temp[j] = static_cast<float>(src_data[i + j]);
            }
            __m256 v_src = _mm256_load_ps(temp);
            __m256 v_out = _mm256_mul_ps(v_src, v_scale);
            _mm256_storeu_ps(dst_data + i, v_out);
        }

        // Scalar tail
        for (; i < n_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]) * scale;
        }
#else
        for (int64_t i = 0; i < n_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]) * scale;
        }
#endif
        break;
    }

    case QuantType::FP16: {
        // FP16 to FP32 conversion
        // Note: Requires FP16 support headers
        throw NotImplementedException("Dequantize FP16 is not implemented");
    }

    default: throw QuantizationException(std::string("Unsupported dequantization type ") + QuantTypeName(src.type));
    }
}

// =============================================================================
// Matrix Operations
// =============================================================================

void CpuBackend::MatMul(const Tensor& A, const Tensor& B, Tensor* C) {
    // Delegate to NUMA-aware version with round-robin dispatch
    MatMul(A, B, C, -1);
}

void CpuBackend::MatMul(const Tensor& A, const Tensor& B, Tensor* C, int numa_node_id) {
    // Basic validation
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor output = *C;
        graph->RecordOperation([this, A, B, output, numa_node_id]() mutable {
            CaptureGuard guard(this);
            MatMul(A, B, &output, numa_node_id);
        });
        return;
    }

    // Delegate to registered kernel
    if (auto* op = GetOpRegistry().GetBest(OpType::MatMul, DeviceType::CPU)) {
        auto* matmul_op = dynamic_cast<MatMulOps*>(op);
        if (matmul_op) {
            matmul_op->MatMul(A, B, C);
            return;
        }
    }

    // Fallback if no kernel registered
    throw BackendException("No MatMul kernel registered");
}

void CpuBackend::MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) {
    // Delegate to NUMA-aware version with round-robin dispatch
    MatMulTransB(A, B, C, -1);
}

void CpuBackend::MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C, int numa_node_id) {
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor output = *C;
        graph->RecordOperation([this, A, B, output, numa_node_id]() mutable {
            CaptureGuard guard(this);
            MatMulTransB(A, B, &output, numa_node_id);
        });
        return;
    }

    // Sticky routing NUMA path:
    // When a specific NUMA node is requested, execute on that node's pool so the
    // compute thread location follows expert-to-NUMA mapping.
    if (numa_node_id >= 0 && A.dtype == DType::F32 && B.dtype == DType::F32 && C->dtype == DType::F32) {
        const int M = static_cast<int>(A.shape[0]);
        const int K = static_cast<int>(A.shape[1]);
        const int N = static_cast<int>(B.shape[0]);
        if (M <= 0 || K <= 0 || N <= 0) {
            return;
        }

        const float* a_data = A.DataAs<float>();
        const float* b_data = B.DataAs<float>();
        float* c_data = C->DataAs<float>();

        auto& pool = GetThreadPool(numa_node_id);
        pool.ParallelFor(M, [=](int m_start, int m_end, int) {
            for (int m = m_start; m < m_end; ++m) {
                const float* a_row = a_data + static_cast<size_t>(m) * K;
                float* c_row = c_data + static_cast<size_t>(m) * N;
                for (int n = 0; n < N; ++n) {
                    const float* b_row = b_data + static_cast<size_t>(n) * K;
                    c_row[n] = simd::DotF32(a_row, b_row, K);
                }
            }
        });
        return;
    }

    // Delegate to registered kernel
    if (auto* op = GetOpRegistry().GetBest(OpType::MatMulTransB, DeviceType::CPU)) {
        auto* matmul_op = dynamic_cast<MatMulOps*>(op);
        if (matmul_op) {
            matmul_op->MatMulTransB(A, B, C);
            return;
        }
    }

    throw BackendException("No MatMulTransB kernel registered");
}

void CpuBackend::GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                          int group_size) {
    // Delegate to NUMA-aware version with round-robin dispatch
    GemmInt4(A, W, scales, zero_points, C, group_size, -1);
}

void CpuBackend::GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                          int group_size, int numa_node_id) {
    if (!A.IsValid() || !W.IsValid() || !C || !C->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor output = *C;
        graph->RecordOperation([this, A, W, scales, zero_points, output, group_size, numa_node_id]() mutable {
            CaptureGuard guard(this);
            GemmInt4(A, W, scales, zero_points, &output, group_size, numa_node_id);
        });
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(W.shape[0]);

    const float* a_data = A.DataAs<float>();
    const uint8_t* w_data = W.DataAs<uint8_t>();
    const float* scales_data = scales.DataAs<float>();
    const float* zeros_data = zero_points.DataAs<float>();
    float* c_data = C->DataAs<float>();

#if defined(DENSECORE_ARM_CORRECTNESS_FIRST) && (defined(__aarch64__) || defined(_M_ARM64))
    if (!OpsRegistry::IsInitialized()) {
        OpsRegistry::Init();
    }
    auto& reg = OpsRegistry::Instance();
    if (reg.GemmInt4) {
        reg.GemmInt4(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size);
        return;
    }
#endif

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(DENSECORE_ARM_CORRECTNESS_FIRST)
    // ARM correctness issue narrowed to the Highway INT4 path. Route through
    // the runtime-selected DenseCore kernel (NEON/SVE) until Highway INT4 on
    // ARM is verified against the same reference path.
    if (!OpsRegistry::IsInitialized()) {
        OpsRegistry::Init();
    }
    auto& reg = OpsRegistry::Instance();
    if (reg.GemmInt4) {
        reg.GemmInt4(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size);
        return;
    }
#endif

    // ===========================================================================
    // DECODE OPTIMIZATION: Use GEMV kernel for M=1 (token generation)
    // ===========================================================================
    // During decode, we generate one token at a time (M=1), making this a
    // matrix-vector multiplication rather than matrix-matrix. The GEMV kernel
    // parallelizes across the N dimension for better utilization.
    // ===========================================================================
    if (M == 1) {
        auto& pool = GetThreadPool(numa_node_id);
        const int n_threads = pool.GetNumThreads();
#ifdef __APPLE__
        if (apple::HasCustomInt4Kernels()) {
            if (n_threads <= 1) {
                apple::GemmInt4CustomRange(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size, 0, N);
            } else {
                pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                    apple::GemmInt4CustomRange(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size,
                                               n_start, n_end);
                });
            }
            return;
        }
#endif
        if (n_threads <= 1) {
            // Single-threaded: process all N at once
            hwy_kernels::GemvInt4_Hwy(c_data, a_data, w_data, scales_data, zeros_data, K, N, group_size, 0, N);
        } else {
            // Multi-threaded: partition N across threads using ParallelFor
            pool.ParallelFor(N, [&](int n_start, int n_end, int thread_id) {
                (void)thread_id;
                hwy_kernels::GemvInt4_Hwy(c_data, a_data, w_data, scales_data, zeros_data, K, N, group_size, n_start,
                                          n_end);
            });
        }
        return;
    }

    // ===========================================================================
    // PREFILL / IMAGE PATH: Use the batched kernel for M>1
    // ===========================================================================
    // The old path called the single-threaded GemmInt4 kernel directly, so
    // DenseDiffusion denoise prefill effectively behaved like a 1-thread path
    // even when the backend thread pool had been resized. We already have a
    // split-N batched INT4 kernel (`GemmInt4Batched`) that reuses weights
    // across multiple activation rows; use it here and partition the output
    // columns across the thread pool.
    const size_t input_stride_bytes = static_cast<size_t>(K) * sizeof(float);
    auto run_batched = [&](int n_start, int n_end) -> bool {
#ifdef __APPLE__
        if (apple::HasCustomInt4Kernels()) {
            apple::GemmInt4CustomRange(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size, n_start,
                                       n_end);
            return true;
        }
#endif
        if (!OpsRegistry::IsInitialized()) {
            OpsRegistry::Init();
        }
        auto& reg = OpsRegistry::Instance();
        if (reg.GemmInt4Batched) {
            reg.GemmInt4Batched(c_data, a_data, w_data, scales_data, zeros_data, M, K, N, group_size, 0, M, n_start,
                                n_end, input_stride_bytes);
            return true;
        }
#if defined(__AVX2__)
        simd::GemmInt4Fp32Batched_AVX2(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size, n_start,
                                       n_end);
        return true;
#else
        return false;
#endif
    };

    auto& pool = GetThreadPool(numa_node_id);
    const int n_threads = pool.GetNumThreads();
    const int64_t total_work = static_cast<int64_t>(M) * static_cast<int64_t>(N) * static_cast<int64_t>(K);
    const bool should_parallelize = n_threads > 1 && N >= 128 && total_work >= (1ll << 18);

    if (should_parallelize) {
        pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
            if (!run_batched(n_start, n_end)) {
                for (int m = 0; m < M; ++m) {
                    hwy_kernels::GemvInt4_Hwy(c_data + static_cast<size_t>(m) * N, a_data + static_cast<size_t>(m) * K,
                                              w_data, scales_data, zeros_data, K, N, group_size, n_start, n_end);
                }
            }
        });
        return;
    }

    if (run_batched(0, N)) {
        return;
    }

#if defined(__AVX512F__)
    simd::GemmInt4Fp32_AVX512(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size);
#elif defined(__AVX2__)
    simd::GemmInt4Fp32_AVX2(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size);
#else
    std::cerr << "[CpuBackend] GemmInt4: No SIMD support, operation skipped" << std::endl;
#endif
}

// =============================================================================
// Normalization Operations
// =============================================================================

void CpuBackend::RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps) {
    if (!input.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, weight, out, eps]() mutable {
            CaptureGuard guard(this);
            RMSNorm(input, weight, &out, eps);
        });
        return;
    }

    const int64_t n_elements = input.NumElements();
    const int64_t hidden_dim = weight.shape[0];

    if (hidden_dim == 0) {
        return;
    }

    const int64_t n_tokens = n_elements / hidden_dim;
    const float* x = input.DataAs<float>();
    const float* w = weight.DataAs<float>();
    float* out = output->DataAs<float>();

    // Process each token
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float* x_ptr = x + t * hidden_dim;
        float* out_ptr = out + t * hidden_dim;

        // Compute RMS
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < hidden_dim; ++i) {
            sum_sq += x_ptr[i] * x_ptr[i];
        }
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_dim) + eps);

        // Normalize and apply weight
        for (int64_t i = 0; i < hidden_dim; ++i) {
            out_ptr[i] = x_ptr[i] * rms * w[i];
        }
    }
}

void CpuBackend::AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                            float eps) {
    if (!input.IsValid() || !residual.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, residual, weight, out, eps]() mutable {
            CaptureGuard guard(this);
            AddRMSNorm(input, residual, weight, &out, eps);
        });
        return;
    }

    const int64_t hidden_dim = weight.shape[0];
    const int64_t n_elements = input.NumElements();

    if (hidden_dim == 0) {
        return;
    }

    const int64_t n_tokens = n_elements / hidden_dim;
    const float* x = input.DataAs<float>();
    const float* res = residual.DataAs<float>();
    const float* w = weight.DataAs<float>();
    float* out = output->DataAs<float>();

    // Use simd::AddRMSNorm for each token (it handles one vector at a time)
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float* x_ptr = x + t * hidden_dim;
        const float* res_ptr = res + t * hidden_dim;
        float* out_ptr = out + t * hidden_dim;

        simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, w, static_cast<size_t>(hidden_dim), eps);
    }
}

void CpuBackend::LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output, float eps) {
    if (!input.IsValid() || !gamma.IsValid() || !beta.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, gamma, beta, out, eps]() mutable {
            CaptureGuard guard(this);
            LayerNorm(input, gamma, beta, &out, eps);
        });
        return;
    }

    const int64_t hidden_dim = gamma.shape[0];
    const int64_t n_elements = input.NumElements();

    if (hidden_dim == 0) {
        return;
    }

    const int64_t n_tokens = n_elements / hidden_dim;
    const float* x = input.DataAs<float>();
    const float* g = gamma.DataAs<float>();
    const float* b = beta.DataAs<float>();
    float* out = output->DataAs<float>();

    // Process each token
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float* x_ptr = x + t * hidden_dim;
        float* out_ptr = out + t * hidden_dim;

        // Compute mean
        float sum = 0.0f;
        for (int64_t i = 0; i < hidden_dim; ++i) {
            sum += x_ptr[i];
        }
        float mean = sum / static_cast<float>(hidden_dim);

        // Compute variance
        float var_sum = 0.0f;
        for (int64_t i = 0; i < hidden_dim; ++i) {
            float diff = x_ptr[i] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(hidden_dim) + eps);

        // Normalize and apply affine transform
        for (int64_t i = 0; i < hidden_dim; ++i) {
            out_ptr[i] = (x_ptr[i] - mean) * inv_std * g[i] + b[i];
        }
    }
}

// =============================================================================
// Activation Operations
// =============================================================================

void CpuBackend::Softmax(const Tensor& input, Tensor* output) {
    if (!input.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, out]() mutable {
            CaptureGuard guard(this);
            Softmax(input, &out);
        });
        return;
    }

    // Delegate to OpRegistry for vendor-optimized implementation
    if (auto* op = GetOpRegistry().GetBest(OpType::Softmax, DeviceType::CPU)) {
        Tensor input_copy = input;
        std::vector<Tensor*> inputs = {&input_copy};
        std::vector<Tensor*> outputs = {output};
        op->Execute(inputs, outputs, nullptr);
        return;
    }

    // Fallback: Copy input to output first, then do in-place softmax
    const size_t size = input.SizeBytes();
    std::memcpy(output->data, input.data, size);
    SoftmaxInplace(output);
}

void CpuBackend::SoftmaxInplace(Tensor* data) {
    if (!data || !data->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor inout = *data;
        graph->RecordOperation([this, inout]() mutable {
            CaptureGuard guard(this);
            SoftmaxInplace(&inout);
        });
        return;
    }

    // Apply softmax along last dimension
    const int64_t n = data->shape[data->ndim - 1];
    int64_t batch_size = 1;
    for (int i = 0; i < data->ndim - 1; ++i) {
        batch_size *= data->shape[i];
    }

    float* ptr = data->DataAs<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        simd::SoftmaxF32(ptr + b * n, static_cast<size_t>(n));
    }
}

void CpuBackend::SiLU(const Tensor& input, Tensor* output) {
    if (!input.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, out]() mutable {
            CaptureGuard guard(this);
            SiLU(input, &out);
        });
        return;
    }

    // Delegate to OpRegistry for vendor-optimized implementation
    if (auto* op = GetOpRegistry().GetBest(OpType::SiLU, DeviceType::CPU)) {
        auto* activation_op = dynamic_cast<ActivationOps*>(op);
        if (activation_op) {
            activation_op->SiLU(input, output);
            return;
        }
    }

    // Fallback: scalar implementation
    const int64_t n_elements = input.NumElements();
    const float* x = input.DataAs<float>();
    float* out = output->DataAs<float>();

    for (int64_t i = 0; i < n_elements; ++i) {
        float val = x[i];
        out[i] = val / (1.0f + FastExp(-val));
    }
}

void CpuBackend::GELU(const Tensor& input, Tensor* output) {
    if (!input.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, out]() mutable {
            CaptureGuard guard(this);
            GELU(input, &out);
        });
        return;
    }

    // Delegate to OpRegistry for vendor-optimized implementation
    if (auto* op = GetOpRegistry().GetBest(OpType::GELU, DeviceType::CPU)) {
        auto* activation_op = dynamic_cast<ActivationOps*>(op);
        if (activation_op) {
            activation_op->GELU(input, output);
            return;
        }
    }

    // Fallback: scalar implementation
    const int64_t n_elements = input.NumElements();
    const float* x = input.DataAs<float>();
    float* out = output->DataAs<float>();

    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    for (int64_t i = 0; i < n_elements; ++i) {
        float val = x[i];
        float x3 = val * val * val;
        float tanh_arg = kSqrt2OverPi * (val + kCoeff * x3);
        out[i] = 0.5f * val * (1.0f + std::tanh(tanh_arg));
    }
}

// =============================================================================
// Position Encoding Operations
// =============================================================================

void CpuBackend::RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output, int rope_dim) {
    if (!input.IsValid() || !cos_sin.IsValid() || !positions || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, input, cos_sin, positions, out, rope_dim]() mutable {
            CaptureGuard guard(this);
            RoPE(input, cos_sin, positions, &out, rope_dim);
        });
        return;
    }

    // Determine dimensions from input shape
    // Supported layouts: [seq_len, head_dim] or [n_heads, seq_len, head_dim]
    int n_tokens, head_dim, n_heads;

    if (input.ndim == 2) {
        n_tokens = static_cast<int>(input.shape[0]);
        head_dim = static_cast<int>(input.shape[1]);
        n_heads = 1;
    } else if (input.ndim == 3) {
        n_heads = static_cast<int>(input.shape[0]);
        n_tokens = static_cast<int>(input.shape[1]);
        head_dim = static_cast<int>(input.shape[2]);
    } else {
        return;  // Unsupported layout
    }

    if (rope_dim < 0) {
        rope_dim = head_dim;
    }

    const float* in = input.DataAs<float>();
    const float* cs = cos_sin.DataAs<float>();
    float* out = output->DataAs<float>();

    // Use simd::ApplyRoPE which handles the actual rotation
    const int max_seq_len = static_cast<int>(cos_sin.shape[0]);
    simd::ApplyRoPE(out, in, cs, positions, n_tokens, head_dim, rope_dim, max_seq_len);
}

// =============================================================================
// Fused Operations
// =============================================================================

void CpuBackend::FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv,
                                    Tensor* q_out, Tensor* k_out, Tensor* v_out) {
    if (!input.IsValid() || !wq.IsValid() || !wk.IsValid() || !wv.IsValid() || !q_out || !k_out || !v_out) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor q = *q_out;
        Tensor k = *k_out;
        Tensor v = *v_out;
        graph->RecordOperation([this, input, wq, wk, wv, q, k, v]() mutable {
            CaptureGuard guard(this);
            FusedQKVProjection(input, wq, wk, wv, &q, &k, &v);
        });
        return;
    }

    const int n_tokens = static_cast<int>(input.shape[0]);
    const int n_embd = static_cast<int>(input.shape[1]);
    const int dim_q = static_cast<int>(wq.shape[0]);
    const int dim_k = static_cast<int>(wk.shape[0]);
    const int dim_v = static_cast<int>(wv.shape[0]);

    const float* x = input.DataAs<float>();
    const float* w_q = wq.DataAs<float>();
    const float* w_k = wk.DataAs<float>();
    const float* w_v = wv.DataAs<float>();
    float* q = q_out->DataAs<float>();
    float* k = k_out->DataAs<float>();
    float* v = v_out->DataAs<float>();

    // Process each token
    for (int t = 0; t < n_tokens; ++t) {
        const float* x_t = x + t * n_embd;
        float* q_t = q + t * dim_q;
        float* k_t = k + t * dim_k;
        float* v_t = v + t * dim_v;

        // Use simd::ComputeQKV for the actual computation
        // Single-threaded per token; parallelism handled at higher level
        simd::ComputeQKV(q_t, k_t, v_t, x_t, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, 0, 1);
    }
}

void CpuBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                                bool causal, int n_head_kv) {
    // Delegate to NUMA-aware version with round-robin dispatch
    FlashAttention(Q, K, V, output, scale, causal, n_head_kv, -1);
}

void CpuBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                                bool causal, int n_head_kv, int numa_node_id) {
    if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, Q, K, V, out, scale, causal, n_head_kv, numa_node_id]() mutable {
            CaptureGuard guard(this);
            FlashAttention(Q, K, V, &out, scale, causal, n_head_kv, numa_node_id);
        });
        return;
    }

    // Expected layout: [batch, n_head, seq, head_dim]
    const int batch = static_cast<int>(Q.shape[0]);
    const int n_head = static_cast<int>(Q.shape[1]);
    const int seq_q = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[2]);

    // GQA Support: If n_head_kv not provided, infer from K tensor or assume MHA
    if (n_head_kv <= 0) {
        // Try to infer from K tensor shape (K.shape[1] is n_head_kv)
        n_head_kv = static_cast<int>(K.shape[1]);
        if (n_head_kv <= 0) {
            n_head_kv = n_head;  // Fallback to MHA
        }
    }

    if (n_head_kv > n_head) {
        std::cerr << "[CpuBackend] Error: n_head_kv (" << n_head_kv << ") > n_head (" << n_head
                  << "). Layout mismatch or invalid GQA config?" << std::endl;
        return;
    }

    const float* q_data = Q.DataAs<float>();
    const float* k_data = K.DataAs<float>();
    const float* v_data = V.DataAs<float>();
    float* o_data = output->DataAs<float>();

    FlashAttentionConfig config;
    config.scale = scale;
    config.causal = causal;

    auto& pool = GetThreadPool(numa_node_id);
    config.num_threads = std::max(1, pool.GetNumThreads());

    // Route both explicit NUMA dispatch and the default backend path through
    // the backend thread pool. The previous default path delegated directly to
    // FlashAttentionGQA(..., nth=1), which left attention effectively single-threaded.
    const int total_work = batch * n_head;
    if (config.num_threads <= 1 || total_work <= 1) {
        FlashAttentionGQA(q_data, k_data, v_data, o_data, batch, n_head, n_head_kv, seq_q, seq_kv, head_dim, config);
        return;
    }

    pool.ParallelFor(total_work, [&](int start, int end, int /*tid*/) {
        static thread_local FlashAttentionScratch tl_scratch;
        tl_scratch.Resize(config.block_m, config.block_n, head_dim);

        const int n_rep = n_head / n_head_kv;
        for (int work_idx = start; work_idx < end; ++work_idx) {
            const int b = work_idx / n_head;
            const int h = work_idx % n_head;
            const int h_kv = h / n_rep;

            const float* q_ptr = q_data + (b * n_head + h) * seq_q * head_dim;
            const float* k_ptr = k_data + (b * n_head_kv + h_kv) * seq_kv * head_dim;
            const float* v_ptr = v_data + (b * n_head_kv + h_kv) * seq_kv * head_dim;
            float* o_ptr = o_data + (b * n_head + h) * seq_q * head_dim;

            FlashAttentionForward(q_ptr, k_ptr, v_ptr, o_ptr, seq_q, seq_kv, head_dim, config, tl_scratch);
        }
    });
}

// =============================================================================
// MoE Expert Access Recording (Hot Path Integration)
// =============================================================================

std::shared_ptr<CpuBackend::MoELayerRegistry> CpuBackend::GetMoELayerRegistry(const TransformerLayer* layer_key) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = moe_registries_.find(layer_key);
    if (it == moe_registries_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<CpuBackend::MoELayerRegistry> CpuBackend::GetOrCreateMoELayerRegistry(const TransformerLayer* layer_key,
                                                                                      int n_experts, float ema_alpha) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = moe_registries_.find(layer_key);
    if (it != moe_registries_.end()) {
        auto& registry = it->second;
        if (registry) {
            std::lock_guard<std::mutex> registry_lock(registry->mutex);
            if (n_experts > 0 && (!registry->profiler || registry->profiler->GetNumExperts() != n_experts)) {
                registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
            }
            registry->ema_alpha = ema_alpha;
        }
        return registry;
    }

    auto registry = std::make_shared<MoELayerRegistry>();
    registry->ema_alpha = ema_alpha;
    if (n_experts > 0) {
        registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
    }
    moe_registries_.emplace(layer_key, registry);
    return registry;
}

void CpuBackend::RecordExpertAccess(const int* expert_ids, int count) {
    // Lock-free path for hot inference loop
    RecordExpertAccess(nullptr, expert_ids, count);
}

void CpuBackend::RecordExpertAccess(const TransformerLayer* layer_key, const int* expert_ids, int count) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry || !expert_ids || count <= 0) {
        return;
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
    }
    if (profiler) {
        profiler->RecordHitBatch(expert_ids, count);
    }
}

/**
 * @brief Stub MoE routing function demonstrating profiler integration
 *
 * In production, the actual routing happens in:
 * - simd_ops.h::MoETopKRoute() - which already has profiler_hit_fn callback
 * - inference.cpp::ForwardMoE() - can call backend.RecordExpertAccess()
 *
 * Example integration in inference.cpp:
 * @code
 *   void Engine::ForwardMoE(const Tensor& input, Tensor* output) {
 *       MoERouteResult result = MoETopKRoute(...);
 *       // Record access for NUMA optimization
 *       backend_->RecordExpertAccess(result.expert_ids.data(), result.expert_ids.size());
 *       // ... process experts ...
 *   }
 * @endcode
 */
void RouteMoEStub(CpuBackend& backend, int n_tokens, int n_experts_used, int n_experts) {
    // Simulated expert selection (in reality, comes from router gate)
    std::vector<int> selected_experts(n_tokens * n_experts_used);
    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < n_experts_used; k++) {
            // Simplified: hash-based distribution for testing
            selected_experts[t * n_experts_used + k] = (t + k) % n_experts;
        }
    }

    // Record accesses for profiler
    backend.RecordExpertAccess(selected_experts.data(), static_cast<int>(selected_experts.size()));
}

// =============================================================================
// NUMA-Aware MoE Expert Rebalancing
// =============================================================================


// Batch size for move_pages syscall (4096 pages * 4KB = 16MB per syscall)
// Balanced between syscall overhead and memory pressure
static constexpr size_t MOVE_PAGES_BATCH_SIZE = 4096;
static constexpr float REBALANCE_HYSTERESIS_RATIO = 0.15f;

int CpuBackend::RebalanceExperts(const std::vector<int>& hot_expert_ids,
                                 const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                 int target_numa_node) {
    return RebalanceExperts(nullptr, hot_expert_ids, expert_weights, n_experts, target_numa_node);
}

int CpuBackend::RebalanceExperts(const TransformerLayer* layer_key, const std::vector<int>& hot_expert_ids,
                                 const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                 int target_numa_node) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return 0;
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    std::vector<int>* local_expert_ids = nullptr;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
        local_expert_ids = &registry->local_expert_ids;
    }

    return RebalanceExpertsInternal(hot_expert_ids, expert_weights, n_experts, target_numa_node,
                                    profiler ? profiler.get() : nullptr, local_expert_ids);
}

int CpuBackend::RebalanceExpertsInternal(const std::vector<int>& hot_expert_ids,
                                         const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                         int target_numa_node, moe::ExpertProfiler* profiler,
                                         std::vector<int>* local_expert_ids) {
    if (rebalance_disabled_.load()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(rebalance_mutex_);

    // Validate inputs
    if (hot_expert_ids.empty() || expert_weights.empty()) {
        return 0;
    }

    std::vector<int> experts_to_migrate = hot_expert_ids;
    if (profiler && local_expert_ids && !hot_expert_ids.empty()) {
        const size_t target_size = hot_expert_ids.size();
        std::vector<int> previous_local = *local_expert_ids;

        if (previous_local.empty()) {
            *local_expert_ids = hot_expert_ids;
        } else {
            std::vector<std::pair<float, int>> local_ranked;
            local_ranked.reserve(previous_local.size());
            std::unordered_set<int> local_set;

            for (int expert_id : previous_local) {
                if (expert_id < 0 || expert_id >= n_experts) continue;
                if (local_set.insert(expert_id).second) {
                    local_ranked.emplace_back(profiler->GetEmaLoad(expert_id), expert_id);
                }
            }

            if (local_ranked.size() > target_size) {
                std::partial_sort(local_ranked.begin(), local_ranked.begin() + target_size, local_ranked.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                local_ranked.resize(target_size);
            }

            local_set.clear();
            for (const auto& entry : local_ranked) {
                local_set.insert(entry.second);
            }

            std::vector<std::pair<float, int>> candidates;
            candidates.reserve(hot_expert_ids.size());
            for (int expert_id : hot_expert_ids) {
                if (expert_id < 0 || expert_id >= n_experts) continue;
                if (local_set.find(expert_id) == local_set.end()) {
                    candidates.emplace_back(profiler->GetEmaLoad(expert_id), expert_id);
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            std::sort(local_ranked.begin(), local_ranked.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (const auto& candidate : candidates) {
                if (local_ranked.size() < target_size) {
                    local_ranked.push_back(candidate);
                    local_set.insert(candidate.second);
                    std::sort(local_ranked.begin(), local_ranked.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    continue;
                }

                if (local_ranked.empty()) {
                    break;
                }

                const auto& lowest_local = local_ranked.front();
                if (candidate.first > lowest_local.first * (1.0f + REBALANCE_HYSTERESIS_RATIO)) {
                    local_set.erase(lowest_local.second);
                    local_ranked.front() = candidate;
                    local_set.insert(candidate.second);
                    std::sort(local_ranked.begin(), local_ranked.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            }

            local_expert_ids->clear();
            local_expert_ids->reserve(local_ranked.size());
            for (const auto& entry : local_ranked) {
                local_expert_ids->push_back(entry.second);
            }
        }

        if (!local_expert_ids->empty()) {
            std::unordered_set<int> previous_set(previous_local.begin(), previous_local.end());
            experts_to_migrate.clear();
            for (int expert_id : *local_expert_ids) {
                if (previous_set.find(expert_id) == previous_set.end()) {
                    experts_to_migrate.push_back(expert_id);
                }
            }
        }
    }

#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
    // Use current thread's NUMA node if not specified (-1)
    if (target_numa_node < 0) {
        int cpu = sched_getcpu();
        if (cpu >= 0) {
            target_numa_node = numa_node_of_cpu(cpu);
        }
        // Fallback to node 0 if sched_getcpu() or numa_node_of_cpu() fails
        if (target_numa_node < 0) {
            target_numa_node = 0;
        }
    }
    // Check if libnuma is available at runtime
    if (numa_available() < 0) {
        std::cerr << "[CpuBackend] NUMA not available on this system" << std::endl;
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        std::cerr << "[CpuBackend] Failed to get page size" << std::endl;
        return 0;
    }

    int total_pages_migrated = 0;
    int total_pages_failed = 0;
    static bool logged_permission_error = false;

    // Helper lambda to migrate a single weight buffer
    auto migrate_buffer = [&](const ExpertWeight& weight) -> int {
        if (!weight.ptr || weight.size == 0) return 0;

        // Calculate page range
        uintptr_t start_addr = reinterpret_cast<uintptr_t>(weight.ptr);
        uintptr_t aligned_start = start_addr & ~(page_size - 1);
        uintptr_t end_addr = start_addr + weight.size;
        size_t num_pages = (end_addr - aligned_start + page_size - 1) / page_size;

        if (num_pages == 0) return 0;

        int pages_migrated = 0;

        // Process in batches to avoid stack overflow and reduce syscall overhead
        std::vector<void*> pages(std::min(num_pages, MOVE_PAGES_BATCH_SIZE));
        std::vector<int> nodes(pages.size(), target_numa_node);
        std::vector<int> status(pages.size(), -1);

        for (size_t batch_start = 0; batch_start < num_pages; batch_start += MOVE_PAGES_BATCH_SIZE) {
            size_t batch_size = std::min(MOVE_PAGES_BATCH_SIZE, num_pages - batch_start);

            // Populate page addresses for this batch
            for (size_t i = 0; i < batch_size; i++) {
                uintptr_t page_addr = aligned_start + (batch_start + i) * page_size;
                pages[i] = reinterpret_cast<void*>(page_addr);
                nodes[i] = target_numa_node;
                status[i] = -1;
            }

            // Call move_pages for this batch
            long result = move_pages(0, batch_size, pages.data(), nodes.data(), status.data(), MPOL_MF_MOVE);

            if (result == 0) {
                // Count successful migrations and check per-page status
                for (size_t i = 0; i < batch_size; i++) {
                    if (status[i] == target_numa_node) {
                        pages_migrated++;
                    } else if (status[i] >= 0) {
                        // Page already on a different node (not an error)
                    } else {
                        // Negative status indicates per-page error
                        total_pages_failed++;
                    }
                }
            } else if (result < 0) {
                // Syscall-level error
                int err = errno;
                if ((err == EACCES || err == EPERM) && !logged_permission_error) {
                    std::cerr << "[CpuBackend] move_pages: permission denied - disabling NUMA rebalance "
                              << "(need CAP_SYS_NICE)" << std::endl;
                    logged_permission_error = true;
                    rebalance_disabled_.store(true);
                } else if (err == ESRCH) {
                    std::cerr << "[CpuBackend] move_pages: ESRCH - process not found" << std::endl;
                } else if (err != EINVAL && err != EACCES) {
                    std::cerr << "[CpuBackend] move_pages batch failed (errno=" << err << "): " << strerror(err)
                              << std::endl;
                }
                // Continue with next batch despite error
            }
        }

        return pages_migrated;
    };

    // Migrate each hot expert's weights
    for (int expert_id : experts_to_migrate) {
        if (expert_id < 0 || expert_id >= n_experts || static_cast<size_t>(expert_id) >= expert_weights.size()) {
            continue;
        }

        const ExpertWeights& weights = expert_weights[expert_id];
        size_t expert_pages = 0;
        expert_pages += migrate_buffer(weights.w1);
        expert_pages += migrate_buffer(weights.w2);
        expert_pages += migrate_buffer(weights.w3);
        total_pages_migrated += expert_pages;

        // =====================================================================
        // STICKY ROUTING: Record NUMA node for this expert after migration
        // =====================================================================
        // This enables DispatchExpertFFN to route compute to the correct node.
        bool bound = true;
        if (weights.w1.ptr && weights.w1.size) {
            bound &= BindMemoryToNumaNode(weights.w1.ptr, weights.w1.size, target_numa_node);
        }
        if (weights.w2.ptr && weights.w2.size) {
            bound &= BindMemoryToNumaNode(weights.w2.ptr, weights.w2.size, target_numa_node);
        }
        if (weights.w3.ptr && weights.w3.size) {
            bound &= BindMemoryToNumaNode(weights.w3.ptr, weights.w3.size, target_numa_node);
        }
        if (profiler && target_numa_node >= 0 && bound) {
            profiler->SetExpertNumaNode(expert_id, target_numa_node);
        }
#if DENSECORE_NUMA_DEBUG
        if (!bound) {
            std::cerr << "[NUMA] mbind failed for expert " << expert_id << " on node " << target_numa_node << std::endl;
        }
#endif
    }

    if (total_pages_migrated > 0 || total_pages_failed > 0) {
        size_t bytes_migrated = total_pages_migrated * page_size;
        std::cerr << "[CpuBackend] NUMA rebalance: " << total_pages_migrated << " pages ("
                  << (bytes_migrated / (1024 * 1024)) << " MB) migrated to node " << target_numa_node;
        if (total_pages_failed > 0) {
            std::cerr << " (" << total_pages_failed << " pages failed)";
        }
        std::cerr << std::endl;
    }

    return total_pages_migrated;

#else
    // Non-Linux or no libnuma: log intent only
    // Default to node 0 for logging purposes
    if (target_numa_node < 0) {
        target_numa_node = 0;
    }
    size_t total_bytes = 0;
    for (int expert_id : experts_to_migrate) {
        if (expert_id >= 0 && expert_id < n_experts && static_cast<size_t>(expert_id) < expert_weights.size()) {
            const ExpertWeights& w = expert_weights[expert_id];
            total_bytes += w.w1.size + w.w2.size + w.w3.size;
        }
    }

    if (total_bytes > 0) {
        std::cerr << "[CpuBackend] NUMA rebalance (no-op): " << (total_bytes / (1024 * 1024))
                  << " MB targeted for node " << target_numa_node << std::endl;
    }
    return 0;
#endif
}

void CpuBackend::StartRebalanceThread(int interval_ms, int top_k, bool enable_page_migration) {
    if (interval_ms <= 0) {
        interval_ms = 5000;
    }
    if (top_k <= 0) {
        top_k = 4;
    }

    if (rebalance_running_.load()) {
        return;  // Already running
    }
    if (rebalance_disabled_.load()) {
        std::cerr << "[CpuBackend] NUMA rebalance disabled; thread not started" << std::endl;
        return;
    }

    rebalance_stop_.store(false);
    rebalance_running_.store(true);

    rebalance_thread_ = std::thread([this, interval_ms, top_k, enable_page_migration]() {
        struct LayerRebalanceState {
            std::vector<int> previous_hot;
            uint64_t previous_total_hits = 0;
            std::chrono::steady_clock::time_point previous_time;
            double hit_rate_ema = 0.0;
            bool has_baseline = false;
        };

        const int min_interval_ms = std::max(250, interval_ms / 4);
        const int max_interval_ms = std::max(interval_ms, interval_ms * 4);
        int current_interval_ms = interval_ms;

        std::unordered_map<const TransformerLayer*, LayerRebalanceState> layer_states;

        while (!rebalance_stop_.load()) {
            if (rebalance_disabled_.load()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(current_interval_ms));
            if (rebalance_stop_.load()) break;

            std::vector<std::pair<const TransformerLayer*, std::shared_ptr<MoELayerRegistry>>> registries;
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                registries.reserve(moe_registries_.size());
                for (const auto& entry : moe_registries_) {
                    registries.emplace_back(entry.first, entry.second);
                }
            }

            bool any_high = false;
            bool all_low = !registries.empty();

            for (const auto& entry : registries) {
                const TransformerLayer* layer_key = entry.first;
                const auto& registry = entry.second;
                if (!registry) {
                    continue;
                }

                std::shared_ptr<moe::ExpertProfiler> profiler;
                std::vector<ExpertWeights> weights;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    profiler = registry->profiler;
                    weights = registry->experts;
                }

                if (!profiler) {
                    all_low = false;
                    continue;
                }

                profiler->ApplyDecay();
                auto hot_experts = profiler->GetHotExperts(top_k);
                auto now = std::chrono::steady_clock::now();
                uint64_t total_hits = profiler->GetTotalHits();

                LayerRebalanceState& state = layer_states[layer_key];
                if (!state.has_baseline) {
                    state.previous_total_hits = total_hits;
                    state.previous_time = now;
                }

                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - state.previous_time).count();
                if (elapsed_sec <= 0.0) {
                    elapsed_sec = static_cast<double>(current_interval_ms) / 1000.0;
                }

                uint64_t delta_hits =
                    (total_hits >= state.previous_total_hits) ? (total_hits - state.previous_total_hits) : 0;
                double hit_rate = (elapsed_sec > 0.0) ? (delta_hits / elapsed_sec) : 0.0;

                if (!state.has_baseline) {
                    state.hit_rate_ema = hit_rate;
                    state.has_baseline = true;
                } else {
                    state.hit_rate_ema = 0.8 * state.hit_rate_ema + 0.2 * hit_rate;
                }

                float churn = 0.0f;
                if (!state.previous_hot.empty() || !hot_experts.empty()) {
                    if (state.previous_hot.empty() || hot_experts.empty()) {
                        churn = 1.0f;
                    } else {
                        std::unordered_set<int> prev_set(state.previous_hot.begin(), state.previous_hot.end());
                        size_t overlap = 0;
                        for (int expert_id : hot_experts) {
                            if (prev_set.find(expert_id) != prev_set.end()) {
                                overlap++;
                            }
                        }
                        size_t union_size = prev_set.size() + hot_experts.size() - overlap;
                        churn = union_size > 0 ? 1.0f - (static_cast<float>(overlap) / union_size) : 0.0f;
                    }
                }

                const float churn_high = 0.35f;
                const float churn_low = 0.10f;
                const double high_rate_threshold = std::max(1000.0, 512.0 * static_cast<double>(top_k));
                bool high_churn = churn >= churn_high;
                bool low_churn = churn <= churn_low;
                bool high_rate =
                    (state.hit_rate_ema > 0.0 && hit_rate > state.hit_rate_ema * 1.5) || hit_rate > high_rate_threshold;

                any_high = any_high || high_churn || high_rate;
                all_low = all_low && low_churn;

                // NUMA Locality Strategy (two modes):
                //
                // Mode 1 (default): Sticky Routing
                //   DispatchExpertFFN routes compute to the NUMA node where the
                //   expert's weights already reside. Zero syscall overhead, no TLB
                //   shootdowns, works without CAP_SYS_NICE. Requires reasonable
                //   initial placement (interleaved/round-robin).
                //
                // Mode 2 (enable_page_migration=true): Page Migration
                //   Actively migrates hot expert weight pages to the local NUMA
                //   node via move_pages(). Higher overhead but corrects poor
                //   initial placement (e.g., malloc-allocated third-party weights).
                //   After migration, updates expert-to-NUMA mapping so Sticky
                //   Routing dispatches to the new location.
                if (enable_page_migration) {
                    if (!hot_experts.empty() && !weights.empty()) {
                        RebalanceExperts(layer_key, hot_experts, weights, profiler->GetNumExperts(), -1);
                    } else {
                        all_low = false;
                    }
                }

                state.previous_hot = std::move(hot_experts);
                state.previous_total_hits = total_hits;
                state.previous_time = now;
            }

            if (any_high) {
                current_interval_ms = std::max(min_interval_ms, static_cast<int>(current_interval_ms * 0.7));
            } else if (all_low) {
                current_interval_ms = std::min(max_interval_ms, static_cast<int>(current_interval_ms * 1.3));
            }
        }
        rebalance_running_.store(false);
    });
}

void CpuBackend::StopRebalanceThread() {
    if (!rebalance_running_.load()) {
        return;  // Not running
    }

    rebalance_stop_.store(true);

    if (rebalance_thread_.joinable()) {
        rebalance_thread_.join();
    }
}

void CpuBackend::InitMoEProfiler(int n_experts, float ema_alpha) {
    InitMoEProfiler(nullptr, n_experts, ema_alpha);
}

void CpuBackend::InitMoEProfiler(const TransformerLayer* layer_key, int n_experts, float ema_alpha) {
    if (n_experts <= 0) {
        return;
    }

    auto registry = GetOrCreateMoELayerRegistry(layer_key, n_experts, ema_alpha);
    if (!registry) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(rebalance_mutex_);
        std::lock_guard<std::mutex> registry_lock(registry->mutex);
        const bool recreate = !registry->profiler || registry->profiler->GetNumExperts() != n_experts;
        registry->ema_alpha = ema_alpha;
        if (recreate) {
            registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
            registry->local_expert_ids.clear();
        }
    }
}

moe::ExpertProfiler* CpuBackend::GetProfiler(const TransformerLayer* layer_key) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(registry->mutex);
    return registry->profiler.get();
}

void CpuBackend::RegisterMoEExperts(const std::vector<ExpertWeights>& experts) {
    RegisterMoEExperts(nullptr, experts);
}

void CpuBackend::RegisterMoEExperts(const TransformerLayer* layer_key, const std::vector<ExpertWeights>& experts) {
    if (experts.empty()) {
        return;
    }

    auto registry = GetOrCreateMoELayerRegistry(layer_key, static_cast<int>(experts.size()), 0.1f);
    if (!registry) {
        return;
    }

    auto experts_match = [](const std::vector<ExpertWeights>& a, const std::vector<ExpertWeights>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            const ExpertWeights& lhs = a[i];
            const ExpertWeights& rhs = b[i];
            if (lhs.w1.ptr != rhs.w1.ptr || lhs.w2.ptr != rhs.w2.ptr || lhs.w3.ptr != rhs.w3.ptr ||
                lhs.w1.size != rhs.w1.size || lhs.w2.size != rhs.w2.size || lhs.w3.size != rhs.w3.size ||
                lhs.hidden_dim != rhs.hidden_dim || lhs.intermediate_dim != rhs.intermediate_dim) {
                return false;
            }
        }
        return true;
    };

    std::lock_guard<std::mutex> rebalance_lock(rebalance_mutex_);
    std::lock_guard<std::mutex> registry_lock(registry->mutex);
    if (experts_match(registry->experts, experts)) {
        return;
    }

    registry->experts = experts;
    registry->local_expert_ids.clear();

    // ==========================================================================
    // Initial NUMA Placement Detection
    // ==========================================================================
    // Query where each expert's weights are currently placed and record it in
    // the profiler. This provides a baseline mapping before any rebalancing.
    // ==========================================================================
    if (registry->profiler) {
        int detected_count = 0;
        auto detect_expert_numa_node = [this](const ExpertWeights& expert) -> int {
            std::unordered_map<int, int> node_votes;

            auto vote = [this, &node_votes](void* ptr) {
                if (!ptr) return;
                int node = QueryMemoryNumaNode(ptr);
                if (node >= 0) {
                    node_votes[node]++;
                }
            };

            vote(expert.w1.ptr);
            vote(expert.w2.ptr);
            vote(expert.w3.ptr);

            int best_node = -1;
            int best_votes = 0;
            for (const auto& entry : node_votes) {
                if (entry.second > best_votes || (entry.second == best_votes && entry.first < best_node)) {
                    best_node = entry.first;
                    best_votes = entry.second;
                }
            }
            return best_node;
        };

        for (size_t i = 0; i < experts.size(); ++i) {
            int numa_node = detect_expert_numa_node(experts[i]);
            if (numa_node >= 0) {
                registry->profiler->SetExpertNumaNode(static_cast<int>(i), numa_node);
                detected_count++;
#if DENSECORE_NUMA_DEBUG
                std::cerr << "[NUMA] Expert " << i << " initially on node " << numa_node << std::endl;
#endif
            }
        }
#if DENSECORE_NUMA_DEBUG
        std::cerr << "[NUMA] Detected initial placement for " << detected_count << "/" << experts.size() << " experts"
                  << std::endl;
#endif
    }

    std::cerr << "[CpuBackend] Registered " << experts.size() << " MoE experts for layer " << layer_key
              << " NUMA rebalancing" << std::endl;
}

std::vector<CpuBackend::ExpertWeights> CpuBackend::GetRegisteredExperts() const {
    return GetRegisteredExperts(nullptr);
}

std::vector<CpuBackend::ExpertWeights> CpuBackend::GetRegisteredExperts(const TransformerLayer* layer_key) const {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return {};
    }
    std::lock_guard<std::mutex> lock(registry->mutex);
    return registry->experts;
}

// =============================================================================
// RAII wrapper for thread-local aligned scratch buffers
// =============================================================================
struct AlignedScratch {
    float* ptr = nullptr;
    size_t capacity = 0;

    ~AlignedScratch() {
        if (ptr) {
            free(ptr);  // Use free() directly since AllocateDevice uses aligned_alloc/posix_memalign
        }
    }

    void Resize(CpuBackend* b, size_t required) {
        if (required > capacity) {
            if (ptr) free(ptr);
            ptr = static_cast<float*>(b->AllocateDevice(required * sizeof(float)));
            capacity = required;
        }
    }
};

// =============================================================================
// Multi-LoRA Batching (Per-Adapter Token Grouping)
// =============================================================================

void CpuBackend::ApplyMultiLoRA(
    const Tensor& input, const std::string& layer_name,
    const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map, Tensor* output) {
    if (!output || !input.IsValid() || !output->IsValid()) {
        return;
    }
    if (adapter_token_map.empty()) {
        return;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32) {
        return;
    }
    if (input.ndim != 2 || output->ndim != 2) {
        return;
    }

    const int64_t total_tokens = input.shape[0];
    const int64_t input_dim = input.shape[1];
    const int64_t output_dim = output->shape[1];
    if (input_dim <= 0 || output_dim <= 0) {
        return;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();

    struct AdapterTask {
        const LoRAAdapter* adapter = nullptr;
        const std::vector<int>* indices = nullptr;
        int numa_node = -1;
    };

    std::unordered_map<int, std::vector<AdapterTask>> tasks_by_node;
    tasks_by_node.reserve(adapter_token_map.size());

    for (const auto& entry : adapter_token_map) {
        const LoRAAdapter* adapter = entry.first.get();
        const std::vector<int>& indices = entry.second;
        if (!adapter || indices.empty()) {
            continue;
        }

        int numa_node = -1;
        if (!adapter->weights.empty()) {
            // Use any weight to determine NUMA affinity (assuming all on same node or interleaved)
            const auto& weight = adapter->weights.begin()->second;
            if (weight.lora_a && weight.lora_a->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_a->data);
            }
            if (numa_node < 0 && weight.lora_b && weight.lora_b->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_b->data);
            }
        }

        tasks_by_node[numa_node].push_back(AdapterTask{adapter, &indices, numa_node});
    }

    for (auto& group : tasks_by_node) {
        auto& tasks = group.second;
        if (tasks.empty()) {
            continue;
        }

        auto& pool = GetThreadPool(group.first);
        const int task_count = static_cast<int>(tasks.size());

        pool.ParallelFor(task_count, [&](int start, int end, int) {
            static thread_local AlignedScratch input_scratch;
            static thread_local AlignedScratch down_scratch;
            static thread_local AlignedScratch out_scratch;
            static thread_local std::vector<float> lora_a_f32;
            static thread_local std::vector<float> lora_b_f32;

            for (int t = start; t < end; ++t) {
                const AdapterTask& task = tasks[t];
                const LoRAAdapter* adapter = task.adapter;
                const std::vector<int>& indices = *task.indices;
                if (!adapter || indices.empty()) {
                    continue;
                }

                auto it_w = adapter->weights.find(layer_name);
                if (it_w == adapter->weights.end()) {
                    continue;
                }
                const densecore::LoRALayerWeight* layer_weight = &it_w->second;
                if (!layer_weight || !layer_weight->lora_a || !layer_weight->lora_b) {
                    continue;
                }

                const ggml_tensor* lora_a = layer_weight->lora_a;
                const ggml_tensor* lora_b = layer_weight->lora_b;

                const int64_t lora_a_in = lora_a->ne[0];
                const int64_t lora_a_rank = lora_a->ne[1];
                const int64_t lora_b_rank = lora_b->ne[0];
                const int64_t lora_b_out = lora_b->ne[1];
                const int64_t rank = std::min<int64_t>({layer_weight->rank, lora_a_rank, lora_b_rank});

                if (lora_a_in != input_dim || lora_b_out != output_dim || rank <= 0) {
                    continue;
                }

                const float* lora_a_ptr = GetLoRAWeightF32(lora_a, lora_a_f32);
                const float* lora_b_ptr = GetLoRAWeightF32(lora_b, lora_b_f32);
                if (!lora_a_ptr || !lora_b_ptr) {
                    continue;
                }

                const int64_t token_count = static_cast<int64_t>(indices.size());
                if (token_count <= 0) {
                    continue;
                }

                input_scratch.Resize(this, static_cast<size_t>(token_count * input_dim));
                down_scratch.Resize(this, static_cast<size_t>(token_count * rank));
                out_scratch.Resize(this, static_cast<size_t>(token_count * output_dim));

                float* input_subset = input_scratch.ptr;
                float* down = down_scratch.ptr;
                float* out = out_scratch.ptr;

                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    std::memcpy(input_subset + i * input_dim, input_data + static_cast<int64_t>(idx) * input_dim,
                                static_cast<size_t>(input_dim) * sizeof(float));
                }

                native::GemmF32(down, input_subset, lora_a_ptr, static_cast<int>(token_count), static_cast<int>(rank),
                                static_cast<int>(input_dim));
                native::GemmF32(out, down, lora_b_ptr, static_cast<int>(token_count), static_cast<int>(output_dim),
                                static_cast<int>(rank));

                const float scale = adapter->scale;
                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    float* dst = output_data + static_cast<int64_t>(idx) * output_dim;
                    const float* src = out + i * output_dim;
                    if (scale == 1.0f) {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j];
                        }
                    } else {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j] * scale;
                        }
                    }
                }
            }
        });
    }
}

// =============================================================================
// Sticky Routing: NUMA-Aware Expert FFN Dispatch
// =============================================================================

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2,
                                   const Tensor& w3, Tensor* output) {
    DispatchExpertFFN(nullptr, expert_id, input, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const Tensor& w1, const Tensor& w2, const Tensor& w3, Tensor* output) {
    // Query the profiler for the NUMA node where this expert's weights reside
    int numa_node = -1;  // Default to round-robin
    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            numa_node = profiler->GetExpertNumaNode(expert_id);
        }
    }

    // =========================================================================
    // Thread-local scratch buffers to avoid allocation on hot path
    // =========================================================================
    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;

    const int64_t batch = input.shape[0];
    const int64_t intermediate_dim = w1.shape[0];
    const size_t hidden_size = static_cast<size_t>(batch * intermediate_dim);

    // Resize scratch if needed (amortized O(1) for stable sizes)
    hidden_scratch.Resize(this, hidden_size);
    if (w3.IsValid()) {
        gate_scratch.Resize(this, hidden_size);
    }

    // Wrap scratch in Tensor views using Make2D factory
    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);

    // =========================================================================
    // STICKY ROUTING: Route computation to the expert's NUMA node
    // =========================================================================

    // Step 1: Gate projection (w1 * input) with NUMA affinity
    // Note: MoE weights are typically [out, in] requiring TransB
    MatMulTransB(input, w1, &hidden, numa_node);

    // Step 2: SwiGLU activation with w3 (if present)
    if (w3.IsValid()) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        MatMulTransB(input, w3, &gate, numa_node);

        // Element-wise SiLU(hidden) * gate
        auto& pool = GetThreadPool(numa_node);
        const int total = static_cast<int>(hidden_size);

        float* h_ptr = hidden_scratch.ptr;
        const float* g_ptr = gate_scratch.ptr;

        pool.ParallelFor(total, [=](int start, int end, int) {
            for (int i = start; i < end; i++) {
                float x = h_ptr[i];
                float silu = x / (1.0f + FastExp(-x));  // SiLU activation (fast exp)
                h_ptr[i] = silu * g_ptr[i];
            }
        });
    }

    // Step 3: Down projection (w2 * hidden) -> output
    MatMulTransB(hidden, w2, output, numa_node);

    // Note: RecordExpertAccess removed - caller (ForwardMoE) handles batch recording
}

// =============================================================================
// Full MoE Layer Forward Pass
// =============================================================================

void CpuBackend::ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, input, routing, experts, output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    const int batch_size = routing.batch_size;
    const int top_k = routing.top_k;
    const size_t hidden_dim = input.shape[1];
    const int num_experts = static_cast<int>(experts.size());
    const size_t assignment_count = routing.expert_ids.size();

    if (batch_size <= 0 || top_k <= 0 || num_experts <= 0 || assignment_count == 0) {
        return;
    }
    if (routing.weights.size() != assignment_count) {
        return;
    }
    if (!routing.token_indices.empty() && routing.token_indices.size() != assignment_count) {
        return;
    }
    if (routing.token_indices.empty() && assignment_count != static_cast<size_t>(batch_size * top_k)) {
        return;
    }

    // =========================================================================
    // Step 1: Record expert accesses for profiler (batch-level, once)
    // =========================================================================
    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            profiler->RecordHitBatch(routing.expert_ids.data(), static_cast<int>(routing.expert_ids.size()));
        }
    }

    // =========================================================================
    // Step 2: Initialize output to zero (for weighted accumulation)
    // =========================================================================
    float* out_data = output->DataAs<float>();
    std::memset(out_data, 0, batch_size * hidden_dim * sizeof(float));

    // =========================================================================
    // Step 3: Thread-local buffers for routing + expert input/output
    // =========================================================================
    static thread_local AlignedScratch routing_scratch;
    static thread_local AlignedScratch expert_input_scratch;
    static thread_local AlignedScratch expert_output_scratch;

    // =========================================================================
    // Step 4: Group assignments by expert for batched dispatch (no allocations)
    // =========================================================================
    const int total_assignments = static_cast<int>(assignment_count);
    if (total_assignments == 0 || num_experts == 0 || top_k <= 0) {
        return;
    }

    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    routing_scratch.Resize(this, workspace_floats);

    moe::MoERoutingWorkspace ws;
    if (!moe::InitMoERoutingWorkspace(&ws, routing_scratch.ptr, routing_scratch.capacity * sizeof(float), batch_size,
                                      num_experts, top_k)) {
        return;
    }

    moe::MoEReorderMapView reorder_map;
    if (!moe::BuildMoEReorderMap(routing, num_experts, &reorder_map, &ws)) {
        return;
    }
    if (reorder_map.total_assignments == 0) {
        return;
    }

    const int hidden_dim_i = static_cast<int>(hidden_dim);
    const size_t packed_size = static_cast<size_t>(reorder_map.total_assignments) * hidden_dim;
    expert_input_scratch.Resize(this, packed_size);
    expert_output_scratch.Resize(this, packed_size);

    // =========================================================================
    // Step 5: Reorder inputs into expert-grouped layout
    // =========================================================================
    const float* input_data = input.DataAs<float>();
    float* packed_input = expert_input_scratch.ptr;
    float* packed_output = expert_output_scratch.ptr;
    auto& reorder_pool = GetThreadPool(-1);
    moe::ReorderInputs(input_data, batch_size, hidden_dim_i, reorder_map, packed_input, &reorder_pool);

    // =========================================================================
    // Step 6: Process each expert with a batched dispatch
    // =========================================================================
    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int start = reorder_map.expert_offsets[static_cast<size_t>(expert_id)];
        const int end = reorder_map.expert_offsets[static_cast<size_t>(expert_id + 1)];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        const ExpertWeights& exp = experts[expert_id];

        Tensor w1 = Tensor::Make2D(exp.w1.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                   static_cast<int64_t>(exp.hidden_dim));
        Tensor w2 = Tensor::Make2D(exp.w2.ptr, static_cast<int64_t>(exp.hidden_dim),
                                   static_cast<int64_t>(exp.intermediate_dim));

        Tensor w3;
        if (exp.w3.ptr != nullptr) {
            w3 = Tensor::Make2D(exp.w3.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                static_cast<int64_t>(exp.hidden_dim));
        }

        float* expert_input_buf = packed_input + static_cast<size_t>(start) * hidden_dim;
        int expert_numa_node = -1;
        if (registry) {
            std::shared_ptr<moe::ExpertProfiler> profiler;
            {
                std::lock_guard<std::mutex> lock(registry->mutex);
                profiler = registry->profiler;
            }
            if (profiler) {
                expert_numa_node = profiler->GetExpertNumaNode(expert_id);
            }
        }


        Tensor expert_input =
            Tensor::Make2D(expert_input_buf, static_cast<int64_t>(count), static_cast<int64_t>(hidden_dim));
        Tensor expert_out = Tensor::Make2D(packed_output + static_cast<size_t>(start) * hidden_dim,
                                           static_cast<int64_t>(count), static_cast<int64_t>(hidden_dim));

        DispatchExpertFFN(layer_key, expert_id, expert_input, w1, w2, w3, &expert_out);
    }

    // =========================================================================
    // Step 7: Reorder outputs back to original token order (parallelized)
    // =========================================================================
    moe::ReorderOutputs(packed_output, hidden_dim_i, reorder_map, out_data, &reorder_pool);
}

// =============================================================================
// Singleton Accessor
// =============================================================================

CpuBackend& GetCpuBackend() {
    static CpuBackend instance;
    return instance;
}

ComputeBackend& GetComputeBackend() {
    return GetCpuBackend();
}

}  // namespace densecore
