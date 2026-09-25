/**
 * @file densecore/backend/apple/metal_backend.h
 * @brief Apple Metal GPU backend for Apple Silicon (M1/M2/M3/M4)
 *
 * This backend provides GPU acceleration for LLM inference on Apple Silicon
 * devices using the Metal API. It implements the ComputeBackend interface
 * through DenseCore's ComputeBackend interface.
 *
 * Implementation:
 * - GGML Metal handles graph execution and tensor management.
 * - Custom Metal shaders handle performance-critical paths:
 *   - GEMV (decode phase): Parallel reduction across output dimension
 *   - FlashAttention: Tiled attention with shared memory
 *   - Quantized GEMM: INT4/INT8 dequantization kernels
 *
 * Apple Silicon Advantages:
 * - Unified Memory Architecture (UMA): Zero-copy CPU↔GPU transfers
 * - High memory bandwidth: 100-120+ GB/s
 * - Tile-based deferred rendering: Efficient for reduction ops
 *
 * @see ComputeBackend for interface documentation
 * @see densecore/backend/apple/apple_silicon.h for chip detection utilities
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_METAL_BACKEND_H
#define DENSECORE_METAL_BACKEND_H

#include <memory>

#include "densecore/hal/compute_backend.h"

// Only compile on Apple platforms
#ifdef __APPLE__

#include <TargetConditionals.h>

// Forward declarations for Objective-C Metal types
// These are opaque pointers when compiled as C++
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLLibrary;
@class MTLComputePipelineState;
@protocol MTLBuffer;
#else
typedef void* id;
#endif

namespace densecore {

/**
 * @brief Apple Silicon chip information
 *
 * Contains detailed information about the current Apple Silicon chip,
 * useful for optimization decisions and capability detection.
 */
struct AppleSiliconChipInfo {
    const char* chip_name;               ///< Human-readable name: "M1", "M2 Pro", etc.
    int chip_generation;                 ///< 1=M1, 2=M2, 3=M3, 4=M4
    int gpu_cores;                       ///< Number of GPU cores (7-40)
    int performance_cores;               ///< Number of P-cores (4-12)
    int efficiency_cores;                ///< Number of E-cores (4-6)
    uint64_t unified_memory_bytes;       ///< Total unified memory
    float memory_bandwidth_gbps;         ///< Memory bandwidth in GB/s
    int neural_engine_tops;              ///< Neural Engine TOPS (11-38)
    bool supports_simd_group_reduction;  ///< Metal 2.4+ feature
    bool supports_bfloat16;              ///< BFloat16 support (M3+)
    bool supports_ray_tracing;           ///< Hardware ray tracing (M3+)
};

/**
 * @brief Metal GPU compute backend for Apple Silicon
 *
 * Implements the ComputeBackend interface using Metal for GPU acceleration.
 * Leverages Apple's Unified Memory Architecture for efficient CPU↔GPU
 * data sharing without explicit copies.
 *
 * Thread Safety:
 * - Metal command buffers are thread-safe when properly synchronized
 * - This class uses per-thread command encoders for parallel graph building
 * - Synchronize() must be called before reading GPU outputs
 *
 * Memory Model:
 * - All allocations use MTLStorageModeShared (UMA - zero copy)
 * - 64-byte alignment for optimal cache line usage
 * - Metal buffers are reference-counted via ARC
 *
 * Example Usage:
 * @code
 *   auto& registry = BackendRegistry::Instance();
 *   if (MetalBackend::IsAvailable()) {
 *     registry.Register(DeviceType::METAL, std::make_unique<MetalBackend>());
 *     registry.SetDefault(DeviceType::METAL);
 *   }
 * @endcode
 */
class MetalBackend : public ComputeBackend {
public:
    // =========================================================================
    // Static Methods
    // =========================================================================

    /**
     * @brief Check if Metal is available on this system
     *
     * Checks for:
     * 1. macOS 10.14+ or iOS 12+ (Metal 2.0)
     * 2. A valid Metal device exists
     * 3. Required shader features are supported
     *
     * @return true if Metal backend can be initialized
     */
    static bool IsAvailable();

    /**
     * @brief Get the default Metal device name
     * @return Device name string or nullptr if not available
     */
    static const char* GetDeviceName();

    /**
     * @brief Get Apple Silicon chip information
     * @return Struct containing chip details
     */
    static AppleSiliconChipInfo GetChipInfo();

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    /**
     * @brief Initialize Metal backend
     *
     * Performs:
     * 1. Metal device discovery
     * 2. Command queue creation
     * 3. Shader library compilation (from .metallib or source)
     * 4. Pipeline state creation for custom kernels
     *
     * @throws std::runtime_error if Metal initialization fails
     */
    MetalBackend();

    /**
     * @brief Cleanup Metal resources
     *
     * Waits for pending GPU work to complete, then releases:
     * - Command queue
     * - Shader library
     * - Pipeline states
     * - Device reference
     */
    ~MetalBackend() override;

    // Non-copyable, non-movable (Metal resources are not transferable)
    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;
    MetalBackend(MetalBackend&&) = delete;
    MetalBackend& operator=(MetalBackend&&) = delete;

    // =========================================================================
    // ComputeBackend Interface - Identification
    // =========================================================================

    /**
     * @brief Get backend name
     * @return "Apple-Metal" or "Apple-Metal-M1/M2/M3/M4" with chip suffix
     */
    const char* Name() const override;

    /**
     * @brief Get device type
     * @return DeviceType::METAL
     */
    DeviceType Device() const override { return DeviceType::METAL; }

    /**
     * @brief Capability manifest reflecting actual Metal kernel coverage
     *
     * Native ops include all operations with custom Metal shaders or MPS
     * implementations: MatMul, GEMV, Softmax, RMSNorm, AddRMSNorm, RoPE,
     * FlashAttention, FusedQKV, and quantized INT4/INT8 GEMV/GEMM.
     */
    BackendCapabilityManifest GetCapabilityManifest() const override;

    /**
     * @brief Allocate unified memory via MTLStorageModeShared
     *
     * Explicitly allocates a Metal buffer with MTLStorageModeShared,
     * returning the contents pointer for true zero-copy CPU↔GPU access.
     * This is the preferred allocation path on Apple Silicon UMA.
     */
    void* AllocateUnified(size_t size_bytes, size_t alignment = 0) override;

    // =========================================================================
    // ComputeBackend Interface - Memory Management
    // =========================================================================

    /**
     * @brief Allocate device memory (unified memory on Apple Silicon)
     *
     * Uses MTLStorageModeShared for zero-copy CPU↔GPU access.
     * Memory is aligned to 64 bytes for optimal cache performance.
     *
     * @param size_bytes Number of bytes to allocate
     * @param alignment Alignment requirement (default 64)
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * @note The returned pointer can be used directly by both CPU and GPU
     *       due to Apple Silicon's Unified Memory Architecture.
     */
    void* AllocateDevice(size_t size_bytes, size_t alignment = 64) override;

    /**
     * @brief Free device memory
     * @param ptr Pointer returned by AllocateDevice
     */
    void FreeDevice(void* ptr) override;

    /**
     * @brief Copy data to device (no-op on UMA)
     *
     * On Apple Silicon with Unified Memory, this is effectively a memcpy
     * since CPU and GPU share the same physical memory. Included for
     * API compatibility with discrete GPU backends.
     *
     * @param dst Destination pointer (device memory)
     * @param src Source pointer (host memory)
     * @param size_bytes Number of bytes to copy
     */
    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override;

    /**
     * @brief Copy data from device (no-op on UMA)
     *
     * @see CopyToDevice for UMA behavior notes
     */
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override;

    // =========================================================================
    // ComputeBackend Interface - Matrix Operations
    // =========================================================================

    /**
     * @brief General matrix multiplication: C = A @ B
     *
     * Implementation strategy:
     * - Batch > 1 (prefill): Use GGML Metal backend or MPS
     * - Batch = 1 (decode): Use custom GEMV Metal kernel
     *
     * @param A Input matrix [M, K]
     * @param B Weight matrix [K, N]
     * @param C Output matrix [M, N]
     */
    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override;

    /**
     * @brief Matrix multiplication with transposed B: C = A @ B^T
     *
     * Optimized for attention score computation where K/V are stored
     * in [seq, head_dim] format.
     *
     * @param A Input matrix [M, K]
     * @param B Weight matrix [N, K] (stored row-major, will be transposed)
     * @param C Output matrix [M, N]
     */
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override;

    /**
     * @brief INT4 quantized GEMM with group-wise dequantization
     *
     * Uses custom Metal kernel for efficient INT4 unpacking and FMA.
     * Supports:
     * - Q4_0, Q4_1 (GGML legacy formats)
     * - Q4_K, Q4_K_M (K-quants with super-blocks)
     *
     * @param A Input activations [M, K] (FP32)
     * @param W Packed INT4 weights [N, K/2]
     * @param scales Per-group scales [N, K/group_size]
     * @param zero_points Per-group zero points [N, K/group_size]
     * @param C Output [M, N]
     * @param group_size Quantization group size (typically 32 or 128)
     */
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                  int group_size) override;

    // =========================================================================
    // ComputeBackend Interface - Normalization Operations
    // =========================================================================

    /**
     * @brief RMS Normalization
     *
     * Computes: output = (input / RMS(input)) * weight
     * where RMS(x) = sqrt(mean(x^2) + eps)
     *
     * @param input Input tensor [N, hidden_dim]
     * @param weight Scale weights [hidden_dim]
     * @param output Output tensor [N, hidden_dim]
     * @param eps Epsilon for numerical stability (default 1e-5)
     */
    void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps = 1e-5f) override;

    /**
     * @brief Fused Add + RMS Normalization
     *
     * Computes: output = RMSNorm(input + residual) * weight
     * Fuses residual addition with normalization for better memory bandwidth.
     *
     * @param input Input tensor [N, hidden_dim]
     * @param residual Residual tensor [N, hidden_dim]
     * @param weight Scale weights [hidden_dim]
     * @param output Output tensor [N, hidden_dim]
     * @param eps Epsilon for numerical stability
     */
    void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                    float eps = 1e-5f) override;

    /**
     * @brief Layer Normalization
     *
     * Computes: output = (input - mean) / sqrt(var + eps) * gamma + beta
     */
    void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output,
                   float eps = 1e-5f) override;

    // =========================================================================
    // ComputeBackend Interface - Activation Operations
    // =========================================================================

    /**
     * @brief SiLU activation
     */
    void SiLU(const Tensor& input, Tensor* output) override;

    /**
     * @brief GELU activation
     */
    void GELU(const Tensor& input, Tensor* output) override;

    /**
     * @brief Softmax activation
     *
     * Computes: output[i] = exp(input[i] - max) / sum(exp(input - max))
     * Uses numerically stable implementation with max subtraction.
     *
     * @param input Input tensor
     * @param output Output tensor (same shape as input)
     */
    void Softmax(const Tensor& input, Tensor* output) override;

    /**
     * @brief In-place softmax activation
     * @param data Tensor to transform in-place
     */
    void SoftmaxInplace(Tensor* data) override;

    // =========================================================================
    // ComputeBackend Interface - Position Encoding
    // =========================================================================

    /**
     * @brief Rotary Position Embedding (RoPE)
     *
     * Applies rotation to input vectors based on position:
     *   x'[2d]   = x[2d]   * cos(θ) - x[2d+1] * sin(θ)
     *   x'[2d+1] = x[2d]   * sin(θ) + x[2d+1] * cos(θ)
     *
     * @param input Input tensor [n_tokens, head_dim] or [n_heads, n_tokens, head_dim]
     * @param cos_sin Pre-computed cos/sin values [max_seq_len, head_dim]
     * @param positions Token positions array [n_tokens]
     * @param output Output tensor (same shape as input)
     * @param rope_dim Number of dimensions to apply RoPE (default: full head_dim)
     */
    void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
              int rope_dim = -1) override;

    // =========================================================================
    // ComputeBackend Interface - Fused Operations
    // =========================================================================

    /**
     * @brief Fused Q/K/V projection
     *
     * Computes all three attention projections in a single pass:
     *   Q = input @ Wq
     *   K = input @ Wk
     *   V = input @ Wv
     *
     * @param input Input tensor [n_tokens, n_embd]
     * @param wq Query weight [n_head * head_dim, n_embd]
     * @param wk Key weight [n_head_kv * head_dim, n_embd]
     * @param wv Value weight [n_head_kv * head_dim, n_embd]
     * @param q_out Query output
     * @param k_out Key output
     * @param v_out Value output
     */
    void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, Tensor* q_out,
                            Tensor* k_out, Tensor* v_out) override;

    /**
     * @brief Flash Attention with memory-efficient tiling
     *
     * Implements FlashAttention-2 algorithm for O(N) memory usage:
     * 1. Tile Q, K, V into blocks that fit in GPU shared memory
     * 2. Compute attention scores block-by-block
     * 3. Apply online softmax with running max/sum
     *
     * Supports both MHA and GQA (Grouped Query Attention).
     *
     * @param Q Query tensor [batch, n_head, seq_q, head_dim]
     * @param K Key tensor [batch, n_head_kv, seq_kv, head_dim]
     * @param V Value tensor [batch, n_head_kv, seq_kv, head_dim]
     * @param output Output tensor [batch, n_head, seq_q, head_dim]
     * @param scale Attention scale (typically 1/sqrt(head_dim))
     * @param causal Whether to apply causal masking
     * @param n_head_kv Number of KV heads (for GQA; -1 = infer from K)
     */
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                        bool causal = true, int n_head_kv = -1, int sliding_window = -1, float logit_softcap = 0.0f,
                        uint32_t semantic_flags = 0, int q_start_offset = 0, int kv_start_offset = 0) override;

    // =========================================================================
    // ComputeBackend Interface - Synchronization
    // =========================================================================

    /**
     * @brief Wait for all pending GPU operations to complete
     *
     * Commits the current command buffer and blocks until execution finishes.
     * Must be called before reading GPU output on the CPU.
     */
    void Synchronize() override;

    // =========================================================================
    // Metal-Specific APIs
    // =========================================================================

    /**
     * @brief Get detailed chip information
     * @return AppleSiliconChipInfo struct
     */
    AppleSiliconChipInfo GetDetailedChipInfo() const;

    /**
     * @brief Check if a specific Metal GPU family is supported
     *
     * Useful for checking feature support:
     * - Family 7 (apple7): M1 baseline
     * - Family 8 (apple8): M2, adds simd_shuffle
     * - Family 9 (apple9): M3, adds ray tracing, mesh shaders
     *
     * @param family Metal GPU family enum value
     * @return true if family is supported
     */
    bool SupportsGPUFamily(int family) const;

    /**
     * @brief Get current GPU memory usage
     * @return Bytes of GPU memory currently allocated
     */
    size_t GetCurrentMemoryUsage() const;

    /**
     * @brief Get peak GPU memory usage since initialization
     * @return Peak bytes of GPU memory allocated
     */
    size_t GetPeakMemoryUsage() const;

    /**
     * @brief Enable Metal GPU capture for debugging
     *
     * When enabled, creates GPU trace files that can be analyzed
     * with Xcode's GPU Debugger or Metal System Trace.
     *
     * @param capture_path Path to save capture file (nullptr for default)
     */
    void EnableGPUCapture(const char* capture_path = nullptr);

    /**
     * @brief Disable GPU capture
     */
    void DisableGPUCapture();

    /**
     * @brief GPU-accelerated INT4 dequantization with group-wise scales
     *
     * Dispatches Metal compute kernel to dequantize packed INT4 weights to FP32.
     * Uses group-wise scales and zero points for accurate dequantization.
     *
     * @param weights_packed Packed INT4 weights [N, K/2] (2 values per byte)
     * @param scales Per-group scale factors [N, K/group_size]
     * @param zeros Per-group zero points [N, K/group_size]
     * @param output Output FP32 weights [N, K]
     * @param N Output dimension (rows)
     * @param K Input dimension (cols)
     * @param group_size Quantization group size
     * @return true if GPU dequant succeeded, false to fall back to CPU
     */
    bool DequantizeInt4Grouped(const void* weights_packed, const float* scales, const float* zeros, float* output,
                               int64_t N, int64_t K, int group_size);

    /**
     * @brief Fused GPU INT4 GEMM with group-wise scales/zero points
     *
     * Multiplies FP32 activations by packed INT4 weights directly on the GPU
     * without materializing a dequantized FP32 weight matrix in unified memory.
     * This is primarily used by the ANE backend's INT4 fallback path.
     *
     * @param A Activations [M, K] FP32
     * @param W Packed INT4 weights [N, K/2]
     * @param scales Per-group scale factors [N, K/group_size]
     * @param zero_points Optional per-group zero points [N, K/group_size]
     * @param C Output [M, N] FP32
     * @param group_size Quantization group size
     * @return true if the fused GPU path was dispatched successfully
     */
    bool GemmInt4Grouped(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                         int group_size);

private:
    /**
     * @brief Private implementation (Pimpl idiom)
     *
     * Hides Objective-C++ implementation details from C++ headers.
     * Contains:
     * - MTLDevice
     * - MTLCommandQueue
     * - MTLLibrary (compiled shaders)
     * - MTLComputePipelineState objects
     * - Buffer pool for memory management
     */
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Backend name string (set at initialization)
    char name_[64];
};

}  // namespace densecore

#endif  // __APPLE__

#endif  // DENSECORE_METAL_BACKEND_H
