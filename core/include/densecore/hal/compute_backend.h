/**
 * @file compute_backend.h
 * @brief Abstract interface for Unified Memory Architecture (UMA) compute
 * backends
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Provides hardware-agnostic operations for LLM inference on:
 * - **Apple Silicon (M1-M4):** Unified memory, Metal GPU, ANE (Neural Engine)
 * - **Qualcomm SoCs:** Unified memory, Adreno GPU, Hexagon DSP
 * - **ARM64 CPUs:** Generic ARM with NEON SIMD
 * - **x86-64 CPUs:** Intel/AMD with AVX2/AVX-512
 *
 * ## Key Design Principles
 *
 * **1. Zero-Copy Memory Model**
 * Instead of separate host/device allocations with explicit copies, UMA
 * backends use `AllocateUnified()` which returns a pointer accessible by both
 * CPU and accelerator. This eliminates PCIe/DMA transfer overhead.
 *
 * **2. Graph Capture for NPUs**
 * NPUs (Apple ANE, Qualcomm Hexagon) achieve peak efficiency with pre-compiled
 * operation graphs. Use `BeginCapture()`/`EndCapture()` to record operations,
 * then `ExecuteGraph()` to run them.
 *
 * **3. First-Class Quantization**
 * Edge hardware relies on INT4/INT8 quantization. Operations accept
 * `QuantizedTensorView` with embedded scale/zero-point metadata.
 *
 * ## Migration from Discrete GPU Model
 *
 * Old pattern (Discrete GPU-style):
 * @code
 * void* host = malloc(size);
 * void* device = backend.AllocateDevice(size);
 * backend.CopyToDevice(device, host, size);  // Slow DMA transfer
 * backend.MatMul(...);
 * backend.CopyFromDevice(host, device, size); // Another transfer
 * @endcode
 *
 * New pattern (UMA-style):
 * @code
 * void* unified = backend.AllocateUnified(size);  // Shared pointer
 * // CPU writes directly to unified memory
 * memcpy(unified, data, size);
 * backend.SynchronizeMemory(unified, size, MemorySyncDirection::HostToDevice);
 * backend.MatMul(...);  // Accelerator reads same memory
 * // No copy back needed - CPU reads same memory
 * @endcode
 *
 * @see accelerator_traits.h for hardware capability queries
 * @see operation_graph.h for graph capture/execution
 * @see quantized_tensor.h for quantization metadata
 */

#ifndef DENSECORE_COMPUTE_BACKEND_H
#define DENSECORE_COMPUTE_BACKEND_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "accelerator_traits.h"
#include "densecore/hal/inference_profile.h"
#include "densecore/hal/operation_graph.h"
#include "quantized_tensor.h"
#include "tensor.h"

#include <unordered_map>
#include <vector>

namespace densecore {

struct LoRAAdapter;  // Forward declaration

/**
 * @brief Thread context for parallel kernel execution
 *
 * Provides thread indexing for work partitioning within kernels.
 * Backends use this to distribute work across available cores.
 */
struct ThreadContext {
    int thread_id = 0;    ///< Current thread index (0-based)
    int num_threads = 1;  ///< Total number of threads

    /// Default single-threaded context
    static ThreadContext Single() { return {0, 1}; }

    /// Create context for thread i of n
    static ThreadContext Make(int i, int n) { return {i, n}; }
};

/**
 * @brief Declared backend op coverage for admission gating.
 *
 * `native_ops`: ops implemented natively on the backend.
 * `fallback_ops`: ops served through explicit fallback path (typically CPU/Metal).
 */
struct BackendCapabilityManifest {
    std::vector<OpType> native_ops;
    std::vector<OpType> fallback_ops;
    bool allow_cpu_fallback = true;
    bool declared_complete = false;
};

/**
 * @brief Abstract interface for UMA compute backends
 *
 * This is the core abstraction layer that decouples the inference graph
 * from hardware-specific implementations. Designed for Apple Silicon,
 * Qualcomm, and ARM-based unified memory architectures.
 *
 * **Performance Guidelines:**
 * - Use `AllocateUnified()` instead of separate host/device allocations
 * - Batch operations into graphs for NPU backends
 * - Prefer fused operations (e.g., AddRMSNorm) over separate calls
 * - Use quantized tensors to reduce memory bandwidth
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // ===========================================================================
    // Backend Identification
    // ===========================================================================

    /**
     * @brief Get backend name (e.g., "CPU-AVX512", "Metal-M3", "Hexagon-v73")
     */
    virtual const char* Name() const = 0;

    /**
     * @brief Get device type
     */
    virtual DeviceType Device() const = 0;

    /**
     * @brief Get hardware capability traits for this backend
     *
     * Use this to query UMA support, graph execution capabilities,
     * and preferred quantization formats.
     */
    virtual AcceleratorTraits GetTraits() const { return AcceleratorTraits::GenericCPU(); }

    /**
     * @brief Backend op capability manifest for profile admission checks.
     *
     * Backends can override to expose precise native/fallback coverage.
     * Default implementation provides conservative baseline by device class.
     */
    virtual BackendCapabilityManifest GetCapabilityManifest() const {
        BackendCapabilityManifest manifest;
        auto append_unique = [](std::vector<OpType>& dst, const std::vector<OpType>& src) {
            std::set<OpType> seen(dst.begin(), dst.end());
            dst.reserve(dst.size() + src.size());
            for (OpType op : src) {
                if (seen.insert(op).second) {
                    dst.push_back(op);
                }
            }
        };

        switch (Device()) {
        case DeviceType::CPU:
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::LLMCore));
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::LLMCoreMoE));
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::Embedding));
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::VisionTransformer));
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::DiffusionTransformer));
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::PointCloud3D));
            manifest.allow_cpu_fallback = false;
            manifest.declared_complete = true;
            break;
        case DeviceType::METAL:
            append_unique(manifest.native_ops, RequiredOpsForProfile(InferenceProfile::LLMCore));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::LLMCoreMoE));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::Embedding));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::VisionTransformer));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::DiffusionTransformer));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::PointCloud3D));
            manifest.allow_cpu_fallback = true;
            break;
        case DeviceType::NPU:
        case DeviceType::ASIC:
            append_unique(manifest.native_ops, {OpType::MatMul, OpType::MatMulTransB, OpType::LayerNorm,
                                                OpType::RMSNorm, OpType::Softmax, OpType::SiLU, OpType::GELU});
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::LLMCore));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::LLMCoreMoE));
            append_unique(manifest.fallback_ops, RequiredOpsForProfile(InferenceProfile::Embedding));
            manifest.allow_cpu_fallback = true;
            break;
        default: manifest.allow_cpu_fallback = true; break;
        }

        return manifest;
    }

    /**
     * @brief Query whether backend advertises support for one op.
     *
     * If `include_fallback` is true, fallback coverage is accepted.
     */
    virtual bool SupportsOp(OpType op, bool include_fallback = true) const {
        const BackendCapabilityManifest manifest = GetCapabilityManifest();
        const auto in_native = std::find(manifest.native_ops.begin(), manifest.native_ops.end(), op);
        if (in_native != manifest.native_ops.end()) {
            return true;
        }
        if (!include_fallback || !manifest.allow_cpu_fallback) {
            return false;
        }
        const auto in_fallback = std::find(manifest.fallback_ops.begin(), manifest.fallback_ops.end(), op);
        return in_fallback != manifest.fallback_ops.end();
    }

    /**
     * @brief Query whether backend can satisfy the whole profile.
     *
     * For `InferenceProfile::Custom`, returns true because required ops are
     * model-defined and cannot be known statically.
     */
    virtual bool SupportsProfile(InferenceProfile profile, bool include_fallback = true) const {
        if (profile == InferenceProfile::Custom) {
            return true;
        }
        const BackendCapabilityManifest manifest = GetCapabilityManifest();
        const auto contains = [](const std::vector<OpType>& ops, OpType op) {
            return std::find(ops.begin(), ops.end(), op) != ops.end();
        };
        for (OpType op : RequiredOpsForProfile(profile)) {
            if (contains(manifest.native_ops, op)) {
                continue;
            }
            if (include_fallback && manifest.allow_cpu_fallback && contains(manifest.fallback_ops, op)) {
                continue;
            }
            return false;
        }
        return true;
    }

    // ===========================================================================
    // Unified Memory Management (UMA)
    // ===========================================================================

    /**
     * @brief Allocate unified memory accessible by CPU and accelerator
     *
     * Returns a pointer that can be used by both the host CPU and the target
     * accelerator without explicit copy operations. This is the primary
     * allocation method for UMA architectures.
     *
     * **Apple Silicon:** Maps to Metal's `MTLStorageModeShared`
     * **Qualcomm:** Maps to ION shared memory
     * **CPU:** Falls back to `aligned_alloc`
     *
     * @param size_bytes Number of bytes to allocate
     * @param alignment Required alignment (default from traits)
     * @return Pointer usable by both CPU and accelerator, nullptr on failure
     *
     * @note Prefer this over `AllocateDevice()` for new code
     */
    virtual void* AllocateUnified(size_t size_bytes, size_t alignment = 0) {
        // Default implementation falls back to device allocation
        if (alignment == 0) {
            alignment = GetTraits().unified_memory_alignment;
        }
        return AllocateDevice(size_bytes, alignment);
    }

    /**
     * @brief Synchronize unified memory between CPU and accelerator
     *
     * On hardware-coherent systems (Apple Silicon), this is typically a no-op.
     * On systems requiring explicit cache management (some ARM SoCs), this
     * performs necessary cache flush/invalidate operations.
     *
     * @param ptr Pointer to unified memory region
     * @param size_bytes Size of region to synchronize
     * @param direction Direction of synchronization
     *
     * @note Only needed on backends where `GetTraits().requires_explicit_sync`
     *       returns true.
     */
    virtual void SynchronizeMemory(void* ptr, size_t size_bytes, MemorySyncDirection direction) {
        // Default: no-op for hardware-coherent systems
        (void)ptr;
        (void)size_bytes;
        (void)direction;
    }

    /**
     * @brief Allocate device memory
     *
     * For CPU backend, this returns aligned host memory.
     * For accelerator backends, this allocates in device address space.
     *
     * @param size_bytes Number of bytes to allocate
     * @param alignment Required alignment in bytes (default: 64 for cache line)
     * @return Pointer to allocated memory, nullptr on failure
     *
     * @note For UMA backends, prefer `AllocateUnified()` instead
     */
    virtual void* AllocateDevice(size_t size_bytes, size_t alignment = 64) = 0;

    /**
     * @brief Free device/unified memory
     * @param ptr Pointer previously returned by AllocateDevice or AllocateUnified
     */
    virtual void FreeDevice(void* ptr) = 0;

    /**
     * @brief Copy data to device memory
     *
     * @deprecated Use `AllocateUnified()` + `SynchronizeMemory()` instead.
     *             For UMA backends, this is a memcpy (no performance benefit).
     *
     * @param dst Destination pointer (device memory)
     * @param src Source pointer (host memory)
     * @param size_bytes Number of bytes to copy
     */
    [[deprecated("Use AllocateUnified() for zero-copy UMA access")]]
    virtual void CopyToDevice(void* dst, const void* src, size_t size_bytes) = 0;

    /**
     * @brief Copy data from device memory
     *
     * @deprecated Use `AllocateUnified()` + `SynchronizeMemory()` instead.
     *             For UMA backends, this is a memcpy (no performance benefit).
     *
     * @param dst Destination pointer (host memory)
     * @param src Source pointer (device memory)
     * @param size_bytes Number of bytes to copy
     */
    [[deprecated("Use AllocateUnified() for zero-copy UMA access")]]
    virtual void CopyFromDevice(void* dst, const void* src, size_t size_bytes) = 0;

    // ===========================================================================
    // Graph Capture API (NPU Support)
    // ===========================================================================

    /**
     * @brief Begin capturing operations into a graph
     *
     * When capturing, operations like `MatMul()` record nodes to an internal
     * graph instead of executing immediately. Call `EndCapture()` to retrieve
     * the recorded graph.
     *
     * **When to use:**
     * - Targeting NPUs (Apple ANE, Qualcomm Hexagon)
     * - Executing the same sequence of operations repeatedly
     * - Wanting to reduce per-operation dispatch overhead
     *
     * @note Only supported on backends where
     * `GetTraits().supports_graph_execution` returns true. CPU backend provides
     * immediate-mode fallback.
     */
    virtual void BeginCapture() {
        capturing_ = true;
        captured_graph_ = std::make_unique<ImmediateModeGraph>();
    }

    /**
     * @brief End capture and return the recorded operation graph
     *
     * @return OperationGraph containing all operations recorded since
     * BeginCapture()
     * @throws std::runtime_error if not currently capturing
     */
    virtual std::unique_ptr<OperationGraph> EndCapture() {
        capturing_ = false;
        return std::move(captured_graph_);
    }

    /**
     * @brief Execute a previously captured operation graph
     *
     * For NPU backends, this submits the compiled graph for execution.
     * For CPU backend, this replays recorded operations immediately.
     *
     * @param graph The graph to execute (must be compiled if required by backend)
     */
    virtual void ExecuteGraph(const OperationGraph& graph) {
        // Polymorphic dispatch - no dynamic_cast needed
        graph.Execute();
    }

    /**
     * @brief Check if currently capturing operations
     */
    bool IsCapturing() const { return capturing_; }

    // ===========================================================================
    // Core Linear Algebra Operations
    // ===========================================================================

    /**
     * @brief Apply Multi-LoRA adapters to a layer input
     */
    virtual void
    ApplyMultiLoRA(const Tensor& input, const std::string& layer_name,
                   const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map,
                   Tensor* output) {
        // Default: no-op
        (void)input;
        (void)layer_name;
        (void)adapter_token_map;
        (void)output;
    }

    /**
     * @brief Matrix multiplication: C = A @ B
     *
     * Standard matrix multiplication for dense matrices.
     * A: [M, K], B: [K, N], C: [M, N]
     *
     * @param A Input tensor [M, K]
     * @param B Input tensor [K, N]
     * @param C Output tensor [M, N]
     */
    virtual void MatMul(const Tensor& A, const Tensor& B, Tensor* C) = 0;

    /**
     * @brief Mixed-precision matrix multiplication: C = A @ B
     *
     * Accepts any combination of dense or quantized tensors via UnifiedTensorRef.
     * This is the preferred method for weight matrices that may be quantized
     * (INT4/INT8) while activations remain in FP32.
     *
     * **Supported Combinations:**
     * - Dense x Dense: Falls back to standard MatMul
     * - Dense x Quantized: Dequantizes B on-the-fly (or uses native INT4 GEMM)
     * - Quantized x Dense: Dequantizes A on-the-fly
     * - Quantized x Quantized: Backend-specific (may require explicit dequant)
     *
     * @param input Input tensor (typically FP32 activations)
     * @param weight Weight tensor (may be quantized INT4/INT8)
     * @param output Output tensor [M, N] (always dense FP32)
     *
     * @note Third-party chip vendors implementing ComputeBackend should
     *       implement this method for mixed-precision inference support.
     */
    virtual void MatMulMixed(const UnifiedTensorRef& input, const UnifiedTensorRef& weight, Tensor* output) {
        // Default implementation: dispatch to appropriate method
        if (input.IsDense() && weight.IsDense()) {
            // Both are dense - use standard MatMul
            MatMul(input.AsDense(), weight.AsDense(), output);
        } else if (input.IsDense() && weight.IsQuantized()) {
            // FP32 input x Quantized weights: most common LLM inference case
            // Extract quantization info and call GemmInt4 if applicable
            // For now, throw - subclasses should override for proper support
            throw std::runtime_error("MatMulMixed: Dense x Quantized requires backend override");
        } else {
            throw std::runtime_error("MatMulMixed: Unsupported tensor combination, override in subclass");
        }
    }

    /**
     * @brief Matrix multiplication with transposed B: C = A @ B^T
     *
     * Optimized path for weight matrices stored in row-major format.
     * A: [M, K], B: [N, K], C: [M, N]
     *
     * @param A Input tensor [M, K]
     * @param B Input tensor [N, K] (will be transposed internally)
     * @param C Output tensor [M, N]
     */
    virtual void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) = 0;

    /**
     * @brief INT4 quantized GEMM: C = A @ dequant(W_int4)^T
     *
     * Dequantizes INT4 weights on-the-fly and computes GEMM.
     * Scales/zeros are applied per group: w_f32 = scale * (w_int4 - zero)
     *
     * @param A Activations [M, K], F32
     * @param W Quantized weights [N, K/2], INT4 packed (2 weights per byte)
     * @param scales Per-group scales [N, num_groups]
     * @param zero_points Per-group zero points [N, num_groups]
     * @param C Output [M, N], F32
     * @param group_size Quantization group size (K must be divisible)
     */
    virtual void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                          int group_size) = 0;

    // ===========================================================================
    // Quantization Operations
    // ===========================================================================

    /**
     * @brief Quantize tensor to specified format
     *
     * Hardware-accelerated quantization using NEON/AVX intrinsics.
     * Useful for quantizing activations at runtime.
     *
     * @param src Source tensor (typically FP32)
     * @param dst Destination quantized tensor view (pre-allocated)
     * @param type Target quantization type
     *
     * @note The `dst.quant_params` should be pre-populated with scale/zero info,
     *       or the backend will compute appropriate values.
     */
    virtual void Quantize(const Tensor& src, QuantizedTensorView* dst, QuantType type) {
        // Default: no-op, subclasses implement
        (void)src;
        (void)dst;
        (void)type;
    }

    /**
     * @brief Dequantize tensor back to FP32
     *
     * @param src Quantized source tensor
     * @param dst Destination FP32 tensor (pre-allocated)
     */
    virtual void Dequantize(const QuantizedTensorView& src, Tensor* dst) {
        // Default: no-op, subclasses implement
        (void)src;
        (void)dst;
    }

    // ===========================================================================
    // Normalization Operations
    // ===========================================================================

    /**
     * @brief RMS Normalization: out = (x / rms(x)) * weight
     *
     * Root Mean Square Layer Normalization (used in LLaMA, Qwen, etc.)
     * rms(x) = sqrt(mean(x^2) + eps)
     *
     * @param input Input tensor [*, hidden_dim]
     * @param weight Normalization weights [hidden_dim]
     * @param output Output tensor [*, hidden_dim]
     * @param eps Epsilon for numerical stability (default: 1e-5)
     */
    virtual void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps = 1e-5f) = 0;

    /**
     * @brief Fused Add + RMSNorm: out = rmsnorm(x + residual) * weight
     *
     * Combines residual addition and RMSNorm into a single kernel to reduce
     * memory bandwidth by loading/storing data once instead of twice.
     *
     * @param input Current hidden state [*, hidden_dim]
     * @param residual Residual to add [*, hidden_dim]
     * @param weight Normalization weights [hidden_dim]
     * @param output Output tensor [*, hidden_dim]
     * @param eps Epsilon for numerical stability
     */
    virtual void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                            float eps = 1e-5f) = 0;

    /**
     * @brief Standard Layer Normalization (GPT, BERT style)
     *
     * output = (input - mean) / sqrt(var + eps) * gamma + beta
     *
     * @param input Input tensor [batch, seq_len, dim]
     * @param gamma Scale weights [dim]
     * @param beta Shift weights [dim]
     * @param output Output tensor (same shape as input)
     * @param eps Epsilon for numerical stability
     */
    virtual void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output,
                           float eps = 1e-5f) = 0;

    // ===========================================================================
    // Activation Operations
    // ===========================================================================

    /**
     * @brief Softmax: out[i] = exp(x[i]) / sum(exp(x))
     *
     * Applied along the last dimension of the input tensor.
     *
     * @param input Input tensor (last dimension is softmax dimension)
     * @param output Output tensor (same shape as input)
     */
    virtual void Softmax(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief In-place softmax
     * @param data Input/output tensor
     */
    virtual void SoftmaxInplace(Tensor* data) = 0;

    /**
     * @brief SiLU (Swish) activation: out = x * sigmoid(x)
     *
     * Standard activation for modern LLMs (Llama, Mistral).
     *
     * @param input Input tensor
     * @param output Output tensor (can be same as input for in-place)
     */
    virtual void SiLU(const Tensor& input, Tensor* output) = 0;

    /**
     * @brief GELU activation (Gaussian Error Linear Unit)
     *
     * Standard activation for GPT, BERT.
     * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     *
     * @param input Input tensor
     * @param output Output tensor (can be same as input for in-place)
     */
    virtual void GELU(const Tensor& input, Tensor* output) = 0;

    // ===========================================================================
    // Position Encoding Operations
    // ===========================================================================

    /**
     * @brief Rotary Positional Embedding (RoPE)
     *
     * Applies rotation to pairs of elements using pre-computed cos/sin values.
     * Standard for LLaMA, Mistral, Qwen, and other modern LLMs.
     *
     * RoPE formula for pair (x_{2d}, x_{2d+1}):
     *   x'_{2d}   = x_{2d} * cos(θ) - x_{2d+1} * sin(θ)
     *   x'_{2d+1} = x_{2d} * sin(θ) + x_{2d+1} * cos(θ)
     *
     * @param input Input tensor [seq_len, head_dim] or [n_heads, seq_len,
     * head_dim]
     * @param cos_sin Pre-computed [max_pos, head_dim] interleaved cos/sin
     * @param positions Position indices for each token [seq_len]
     * @param output Output tensor (same shape as input)
     * @param rope_dim Number of dimensions to rotate (default: head_dim)
     */
    virtual void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
                      int rope_dim = -1) = 0;

    // ===========================================================================
    // Fused Operations (Performance Critical)
    // ===========================================================================

    /**
     * @brief Fused Q/K/V projection
     *
     * Computes Q = x @ Wq, K = x @ Wk, V = x @ Wv in a single pass.
     * Reduces memory bandwidth by loading input x into registers once
     * and writing three outputs in sequence.
     *
     * @param input Input tensor [seq_len, hidden_dim]
     * @param wq Q weight [dim_q, hidden_dim]
     * @param wk K weight [dim_k, hidden_dim]
     * @param wv V weight [dim_v, hidden_dim]
     * @param q_out Output Q [seq_len, dim_q]
     * @param k_out Output K [seq_len, dim_k]
     * @param v_out Output V [seq_len, dim_v]
     */
    virtual void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv,
                                    Tensor* q_out, Tensor* k_out, Tensor* v_out) = 0;

    /**
     * @brief Flash Attention with GQA Support
     *
     * Memory-efficient attention implementation that tiles computation
     * to fit in L2 cache. Essential for long sequence support.
     *
     * Supports Grouped Query Attention (GQA) where n_head_kv < n_head.
     * For MHA models, n_head_kv == n_head (or pass -1 for auto-detect).
     *
     * @param Q Query [batch, n_head, seq_q, head_dim]
     * @param K Key [batch, n_head_kv, seq_kv, head_dim]
     * @param V Value [batch, n_head_kv, seq_kv, head_dim]
     * @param output Output [batch, n_head, seq_q, head_dim]
     * @param scale Attention scale factor (typically 1/sqrt(head_dim))
     * @param causal Whether to apply causal masking
     * @param n_head_kv Number of KV heads (-1 = same as n_head, MHA mode)
     */
    virtual void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                                bool causal = true, int n_head_kv = -1, int sliding_window = -1,
                                float logit_softcap = 0.0f, uint32_t semantic_flags = 0, int q_start_offset = 0,
                                int kv_start_offset = 0) = 0;

    // ===========================================================================
    // Synchronization
    // ===========================================================================

    /**
     * @brief Synchronize all pending operations
     *
     * For CPU backend, this is a no-op.
     * For async backends (Metal/GPU), blocks until all queued operations
     * complete.
     */
    virtual void Synchronize() = 0;

protected:
    bool capturing_ = false;
    std::unique_ptr<ImmediateModeGraph> captured_graph_;
};

/// Factory function type for backend creation
using BackendFactory = std::unique_ptr<ComputeBackend> (*)();

}  // namespace densecore

#endif  // DENSECORE_COMPUTE_BACKEND_H
