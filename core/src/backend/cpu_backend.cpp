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


#include "densecore/backend/cpu_backend.h"

#include "densecore/hal/backend_registry.h"
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

#include "densecore/backend/flash_attention.h"
#include "densecore/backend/gemm_config.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/backend/matmul_backend.h"
#include "densecore/models/lora_storage.h"
#include "densecore/runtime/inference.h"  // For InferenceConfig
#include "densecore/runtime/optimization_bridge.h"
#include "densecore/simd/simd_ops.h"
#include "densecore/simd/simd_platform.h"
#ifdef __APPLE__
#include "densecore/backend/apple/apple_silicon.h"
#endif

#include "backend/thread_pool_impl.h"
#include "densecore/exceptions.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/hal/typed_tensor.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/numa_routing.h"
#include "densecore/quantization/quantized_tensor.h"
#include "ggml.h"
#include "kernels/hwy/hwy_kernels.h"

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

bool ParseCpuBackendEnvBool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return default_value;
    }
    return std::strcmp(value, "0") != 0;
}

int ParseCpuBackendEnvPositiveInt(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return default_value;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
}

bool IsArmInt4SplitNEnabled() {
    return ParseCpuBackendEnvBool("DENSECORE_ARM_INT4_SPLIT_N", false);
}

int GetArmInt4SplitNMinN() {
    return ParseCpuBackendEnvPositiveInt("DENSECORE_ARM_INT4_SPLIT_N_MIN_N", 128);
}

bool IsInt4PathDebugEnabled() {
    return ParseCpuBackendEnvBool("DENSECORE_DEBUG_INT4_PATHS", false) ||
           ParseCpuBackendEnvBool("DENSECORE_DEBUG_MOE_MATMUL_PATHS", false);
}

void LogInt4PathSelection(const char* path, int M, int K, int N, int group_size, int threads) {
    if (!IsInt4PathDebugEnabled()) {
        return;
    }
    std::cerr << "[INT4_PATH] path=" << (path ? path : "unknown") << " M=" << M << " K=" << K << " N=" << N
              << " group_size=" << group_size << " threads=" << threads << '\n';
}

size_t GetMoEPrefetchBytes() {
    static const size_t bytes = []() -> size_t {
        const char* env = std::getenv("DENSECORE_MOE_PREFETCH_BYTES");
        if (!env || *env == '\0') {
            return 8192;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            return 8192;
        }
        return static_cast<size_t>(std::max<unsigned long long>(64ULL, parsed));
    }();
    return bytes;
}

bool IsMoELocalityOrderingEnabled() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_LOCALITY_ORDERING", true);
    return enabled;
}

bool IsMoENextExpertPrefetchEnabled() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_PREFETCH_NEXT_EXPERT", true);
    return enabled;
}

size_t GetMoEDequantCacheBytes() {
	    static const size_t bytes = []() -> size_t {
	        const char* env = std::getenv("DENSECORE_MOE_DEQUANT_CACHE_MB");
	        if (!env || *env == '\0') {
	            return 512ULL * 1024ULL * 1024ULL;
	        }
        char* end = nullptr;
        const unsigned long long parsed_mb = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
	            return 512ULL * 1024ULL * 1024ULL;
        }
        return static_cast<size_t>(parsed_mb) * 1024ULL * 1024ULL;
    }();
    return bytes;
}

bool IsMoEDequantCacheEnabled() {
    return GetMoEDequantCacheBytes() > 0;
}

bool ShouldCacheAllActiveExperts() {
    static const bool enabled = ParseCpuBackendEnvBool("DENSECORE_MOE_DEQUANT_CACHE_ALL_ACTIVE", false);
    return enabled;
}
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
// ThreadPool implementation moved to backend/thread_pool_impl.h

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
        const int n_threads = std::max(1, pool.GetNumThreads());
        const bool prefer_n_parallel = n_threads > 1 && ((M <= 2 && N >= 64) || (M <= 4 && N >= 128) || (M < N / 4));

        if (prefer_n_parallel) {
            pool.ParallelFor(N, [=](int n_start, int n_end, int) {
                for (int n = n_start; n < n_end; ++n) {
                    const float* b_row = b_data + static_cast<size_t>(n) * K;
                    for (int m = 0; m < M; ++m) {
                        const float* a_row = a_data + static_cast<size_t>(m) * K;
                        c_data[static_cast<size_t>(m) * N + n] = simd::DotF32(a_row, b_row, K);
                    }
                }
            });
        } else {
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
        }
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
    auto& pool = GetThreadPool(numa_node_id);
    const int n_threads = pool.GetNumThreads();

#if defined(__aarch64__) || defined(_M_ARM64)
    const bool arm_split_n_decode =
        (M == 1) && n_threads > 1 && IsArmInt4SplitNEnabled() && N >= GetArmInt4SplitNMinN();
    // ARM correctness issue narrowed to the Highway INT4 path. Route through
    // the runtime-selected DenseCore kernel (NEON/SVE) until Highway INT4 on
    // ARM is verified against the same reference path.
    if (!arm_split_n_decode) {
        if (!OpsRegistry::IsInitialized()) {
            OpsRegistry::Init();
        }
        auto& reg = OpsRegistry::Instance();
        if (reg.GemmInt4) {
            LogInt4PathSelection("arm_runtime_gemmint4", M, K, N, group_size, n_threads);
            reg.GemmInt4(c_data, a_data, w_data, scales_data, zeros_data, M, N, K, group_size);
            return;
        }
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
#if defined(__aarch64__) || defined(_M_ARM64)
        if (IsArmInt4SplitNEnabled() && N >= GetArmInt4SplitNMinN() && n_threads > 1) {
            LogInt4PathSelection("arm_split_n_hwy", M, K, N, group_size, n_threads);
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                hwy_kernels::GemvInt4_Hwy(c_data, a_data, w_data, scales_data, zeros_data, K, N, group_size, n_start,
                                          n_end);
            });
            return;
        }
#endif
        LogInt4PathSelection("gemv_hwy", M, K, N, group_size, n_threads);
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
        for (int m = 0; m < M; ++m) {
            hwy_kernels::GemvInt4_Hwy(c_data + static_cast<size_t>(m) * N, a_data + static_cast<size_t>(m) * K, w_data,
                                      scales_data, zeros_data, K, N, group_size, n_start, n_end);
        }
        return true;
#else
        return false;
#endif
    };

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
                                bool causal, int n_head_kv, int sliding_window, float logit_softcap,
                                uint32_t semantic_flags, int q_start_offset, int kv_start_offset) {
    // Delegate to NUMA-aware version with round-robin dispatch
    FlashAttention(Q, K, V, output, scale, causal, n_head_kv, sliding_window, logit_softcap, semantic_flags, -1,
                   q_start_offset, kv_start_offset);
}

void CpuBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                                bool causal, int n_head_kv, int sliding_window, float logit_softcap,
                                uint32_t semantic_flags, int numa_node_id, int q_start_offset, int kv_start_offset) {
    if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output || !output->IsValid()) {
        return;
    }

    if (ImmediateModeGraph* graph = GetCaptureGraph()) {
        Tensor out = *output;
        graph->RecordOperation([this, Q, K, V, out, scale, causal, n_head_kv, sliding_window, logit_softcap,
                                semantic_flags, numa_node_id, q_start_offset, kv_start_offset]() mutable {
            CaptureGuard guard(this);
            FlashAttention(Q, K, V, &out, scale, causal, n_head_kv, sliding_window, logit_softcap, semantic_flags,
                           numa_node_id, q_start_offset, kv_start_offset);
        });
        return;
    }

    // Expected layout: [batch, n_head, seq, head_dim]
    // Decode fast path also accepts native GGML-style strides where dim is
    // contiguous but the seq axis is strided by the number of heads.
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

    if (static_cast<int>(K.shape[3]) != head_dim || static_cast<int>(V.shape[3]) != head_dim ||
        static_cast<int>(output->shape[3]) != head_dim) {
        return;
    }

    FlashAttentionConfig config;
    config.scale = scale;
    config.logit_softcap = logit_softcap;
    config.causal = causal;
    config.sliding_window = sliding_window;
    config.semantic_flags = semantic_flags;
    config.q_start_offset = std::max(0, q_start_offset);
    config.kv_start_offset = std::max(0, kv_start_offset);

    auto& pool = GetThreadPool(numa_node_id);
    config.num_threads = std::max(1, pool.GetNumThreads());

    const bool has_unit_dim_stride = Q.stride[3] == 1 && K.stride[3] == 1 && V.stride[3] == 1 && output->stride[3] == 1;
    const bool has_nonpacked_seq_stride =
        Q.stride[2] != head_dim || K.stride[2] != head_dim || V.stride[2] != head_dim || output->stride[2] != head_dim;
    const bool has_nonpacked_head_stride = Q.stride[1] != seq_q * head_dim || K.stride[1] != seq_kv * head_dim ||
                                           V.stride[1] != seq_kv * head_dim || output->stride[1] != seq_q * head_dim;
    const bool decode_native_strided_layout = seq_q == 1 && has_unit_dim_stride && has_nonpacked_seq_stride;
    const bool prefill_native_strided_layout =
        seq_q > 1 && has_unit_dim_stride && (has_nonpacked_seq_stride || has_nonpacked_head_stride);

    auto run_decode_native_strided = [&](int start, int end) {
        static thread_local FlashAttentionScratch tl_scratch;
        const int n_rep = n_head / n_head_kv;
        for (int work_idx = start; work_idx < end; ++work_idx) {
            const int b = work_idx / n_head;
            const int h = work_idx % n_head;
            const int h_kv = h / n_rep;

            const float* q_ptr = q_data + b * Q.stride[0] + h * Q.stride[1];
            const float* k_ptr = k_data + b * K.stride[0] + h_kv * K.stride[1];
            const float* v_ptr = v_data + b * V.stride[0] + h_kv * V.stride[1];
            float* o_ptr = o_data + b * output->stride[0] + h * output->stride[1];

            FlashAttentionSingleQueryStridedKV(q_ptr, k_ptr, K.stride[2], v_ptr, V.stride[2], o_ptr, seq_kv, head_dim,
                                               config, tl_scratch);
        }
    };

    auto run_prefill_native_strided = [&](int start, int end) {
        static thread_local FlashAttentionScratch tl_scratch;
        const int n_rep = n_head / n_head_kv;
        for (int work_idx = start; work_idx < end; ++work_idx) {
            const int b = work_idx / n_head;
            const int h = work_idx % n_head;
            const int h_kv = h / n_rep;

            const float* q_ptr = q_data + b * Q.stride[0] + h * Q.stride[1];
            const float* k_ptr = k_data + b * K.stride[0] + h_kv * K.stride[1];
            const float* v_ptr = v_data + b * V.stride[0] + h_kv * V.stride[1];
            float* o_ptr = o_data + b * output->stride[0] + h * output->stride[1];

            FlashAttentionForwardStrided(q_ptr, Q.stride[2], k_ptr, K.stride[2], v_ptr, V.stride[2], o_ptr,
                                         output->stride[2], seq_q, seq_kv, head_dim, config, tl_scratch);
        }
    };

    // Route both explicit NUMA dispatch and the default backend path through
    // the backend thread pool. The previous default path delegated directly to
    // FlashAttentionGQA(..., nth=1), which left attention effectively single-threaded.
    const int total_work = batch * n_head;
    if (decode_native_strided_layout) {
        if (config.num_threads <= 1 || total_work <= 1) {
            run_decode_native_strided(0, total_work);
        } else {
            pool.ParallelFor(total_work,
                             [&](int start, int end, int /*tid*/) { run_decode_native_strided(start, end); });
        }
        return;
    }

    if (prefill_native_strided_layout) {
        if (config.num_threads <= 1 || total_work <= 1) {
            run_prefill_native_strided(0, total_work);
        } else {
            pool.ParallelFor(total_work,
                             [&](int start, int end, int /*tid*/) { run_prefill_native_strided(start, end); });
        }
        return;
    }

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
// Singleton Accessor
// =============================================================================

CpuBackend& GetCpuBackend() {
    static CpuBackend instance;
    return instance;
}

CpuBackend& GetTelemetryCpuBackend() {
    if (auto* backend = dynamic_cast<CpuBackend*>(BackendRegistry::Instance().Get(DeviceType::CPU))) {
        return *backend;
    }
    return GetCpuBackend();
}

ComputeBackend& GetComputeBackend() {
    return GetCpuBackend();
}

}  // namespace densecore
