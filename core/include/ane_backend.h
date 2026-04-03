/**
 * @file ane_backend.h
 * @brief Apple Neural Engine (ANE) backend via CoreML
 *
 * The Neural Engine is a fixed-function accelerator on Apple Silicon that
 * provides exceptional performance for matrix operations. On M4, it delivers
 * up to 38 TOPS (Trillion Operations Per Second).
 *
 * ANE Characteristics:
 * - Fixed function: Not programmable like GPU, operations must be supported
 * - High throughput: 11-38 TOPS depending on chip generation
 * - Low power: More efficient than GPU for supported workloads
 * - Latency: ~1-2ms model loading overhead per inference
 *
 * Supported Operations (via CoreML):
 * - Matrix multiplication (with some restrictions)
 * - Convolutions
 * - Common activations (ReLU, GeLU, SiLU)
 * - Normalization layers
 *
 * Strategy:
 * 1. Pre-compile LLM layers to CoreML format at model load time
 * 2. Cache compiled models for instant execution
 * 3. Use ANE for attention/FFN layers that fit constraints
 * 4. Fallback to Metal GPU for unsupported operations
 *
 * Constraints:
 * - Model size < 1GB for optimal ANE placement
 * - No dynamic shapes (fixed sequence length required)
 * - Limited quantization support (INT8, FP16)
 *
 * Usage:
 * @code
 *   if (ANEBackend::IsAvailable()) {
 *     auto ane = std::make_unique<ANEBackend>();
 *     ane->CompileMatMul("layer0_qkv", hidden_dim, 3 * hidden_dim, weights);
 *     ane->ExecuteMatMul("layer0_qkv", input, output);
 *   }
 * @endcode
 *
 * @see metal_backend.h for GPU acceleration
 * @see HybridScheduler for CPU+GPU+ANE coordination
 *
 * Copyright (c) 2024 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_ANE_BACKEND_H
#define DENSECORE_ANE_BACKEND_H

#ifdef __APPLE__

#include <memory>
#include <string>
#include <unordered_map>

#include "densecore/hal/compute_backend.h"

// =============================================================================
// Forward Declarations (must be at global scope for Objective-C)
// =============================================================================
#ifdef __OBJC__
@class MLModel;
@class MLMultiArray;
@class MLModelConfiguration;
#else
typedef void* MLModel;
typedef void* MLMultiArray;
typedef void* MLModelConfiguration;
#endif

namespace densecore {

// =============================================================================
// ANE Operation Types
// =============================================================================

/**
 * @brief Types of operations that can be compiled to ANE
 */
enum class ANEOpType {
    MatMul,      ///< Matrix multiplication
    MatMulBias,  ///< MatMul with fused bias addition
    RMSNorm,     ///< RMS Normalization
    LayerNorm,   ///< Layer Normalization
    SiLU,        ///< SiLU activation (x * sigmoid(x))
    GeLU,        ///< Gaussian Error Linear Unit
    Softmax,     ///< Softmax activation
    RoPE,        ///< Rotary Positional Embedding (offline-compiled ANE path)
    Attention,   ///< Full attention block / FlashAttention (offline-compiled ANE path)
    FFN,         ///< Feed-forward network block (experimental)
};

/**
 * @brief Status of an ANE compiled operation
 */
enum class ANEOpStatus {
    NotCompiled,  ///< Operation not yet compiled
    Compiling,    ///< Compilation in progress
    Ready,        ///< Ready for execution
    Failed,       ///< Compilation failed
    Unsupported,  ///< Operation not supported on ANE
};

/**
 * @brief Statistics for an ANE operation
 */
struct ANEOpStats {
    uint64_t total_executions;  ///< Number of times executed
    double total_time_ms;       ///< Total execution time in ms
    double avg_time_ms;         ///< Average execution time
    double min_time_ms;         ///< Minimum execution time
    double max_time_ms;         ///< Maximum execution time
    bool uses_ane;              ///< True if actually running on ANE (vs CPU fallback)
};

// =============================================================================
// ANE Backend Class
// =============================================================================

/**
 * @brief Apple Neural Engine backend via CoreML
 *
 * This backend compiles operations to CoreML format and executes them on
 * the Neural Engine when possible. It provides a ComputeBackend interface
 * for integration with DenseCore's HAL.
 *
 * Thread Safety:
 * - Compilation is thread-safe (uses internal locking)
 * - Execution is thread-safe for different operations
 * - Same operation should not be executed concurrently
 *
 * Memory Model:
 * - Uses MLMultiArray for data transfer
 * - Shared memory with CPU (no explicit copies needed)
 * - CoreML handles ANE memory management internally
 */
class ANEBackend : public ComputeBackend {
public:
    // ===========================================================================
    // Static Methods
    // ===========================================================================

    /**
     * @brief Check if Neural Engine is available
     *
     * Checks for:
     * 1. Apple Silicon (M1 or later)
     * 2. CoreML availability
     * 3. Neural Engine compute unit access
     *
     * @return true if ANE can be used for inference
     */
    static bool IsAvailable();

    /**
     * @brief Get Neural Engine compute power
     * @return TOPS (Trillion Operations Per Second)
     */
    static int GetTOPS();

    /**
     * @brief Get CoreML version
     * @return Version string (e.g., "7.0")
     */
    static const char* GetCoreMLVersion();

    // ===========================================================================
    // Constructor / Destructor
    // ===========================================================================

    /**
     * @brief Initialize ANE backend
     *
     * Performs:
     * 1. CoreML framework initialization
     * 2. MLModelConfiguration setup for ANE-only execution
     * 3. Cache directory setup for compiled models
     *
     * @throws std::runtime_error if CoreML initialization fails
     */
    ANEBackend();
    ~ANEBackend() override;

    // Non-copyable
    ANEBackend(const ANEBackend&) = delete;
    ANEBackend& operator=(const ANEBackend&) = delete;

    // ===========================================================================
    // ComputeBackend Interface
    // ===========================================================================

    const char* Name() const override { return "Apple-ANE"; }
    DeviceType Device() const override { return DeviceType::ASIC; }
    BackendCapabilityManifest GetCapabilityManifest() const override;

    void* AllocateDevice(size_t size_bytes, size_t alignment = 64) override;
    void FreeDevice(void* ptr) override;
    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override;
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override;

    /**
     * @brief Allocate unified memory for zero-copy CPU↔GPU↔ANE access
     *
     * Delegates to the Metal fallback backend's AllocateDevice() which
     * creates an MTLStorageModeShared buffer. This ensures that memory
     * allocated through ANEBackend is truly zero-copy accessible by
     * Metal GPU kernels used in the fallback path, avoiding hidden
     * copies between posix_memalign'd CPU memory and GPU buffers.
     *
     * Falls back to AllocateDevice (posix_memalign) if Metal is unavailable.
     */
    void* AllocateUnified(size_t size_bytes, size_t alignment = 0) override;

    // Core operations (use pre-compiled CoreML models)
    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override;
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override;

    // Not supported on ANE - throws or falls back
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                  int group_size) override;
    void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps = 1e-5f) override;
    void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                    float eps = 1e-5f) override;
    void LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output,
                   float eps = 1e-5f) override;
    void SiLU(const Tensor& input, Tensor* output) override;
    void GELU(const Tensor& input, Tensor* output) override;
    void Softmax(const Tensor& input, Tensor* output) override;
    void SoftmaxInplace(Tensor* data) override;
    void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
              int rope_dim = -1) override;
    void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv, Tensor* q_out,
                            Tensor* k_out, Tensor* v_out) override;
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                        bool causal = true, int n_head_kv = -1, int sliding_window = -1, float logit_softcap = 0.0f,
                        uint32_t semantic_flags = 0) override;

    void Synchronize() override;

    // ===========================================================================
    // Operation Support Query
    // ===========================================================================

    /**
     * @brief Check if an operation runs natively on ANE hardware
     *
     * IMPORTANT: Individual op calls (MatMul, RMSNorm, etc.) delegate to
     * Metal GPU or CPU Accelerate — they do NOT run on the Neural Engine.
     *
     * True ANE execution requires pre-compiled CoreML models loaded via:
     * - CompileTransformerLayer() → ExecuteTransformerLayer()
     * - CompileMatMul() with a cached .mlmodelc → ExecuteMatMul()
     *
     * The fused transformer layer path (CompileTransformerLayer) compiles
     * QKV+RoPE+Attention+FFN as a single CoreML graph, eliminating the
     * per-op ANE↔CPU transition overhead that would otherwise make
     * individual ANE dispatches slower than GPU.
     *
     * @param op Operation type to check
     * @return true only if a pre-compiled .mlmodelc can run this on ANE
     */
    static bool SupportsOperation(ANEOpType op) {
        switch (op) {
        case ANEOpType::MatMul:
        case ANEOpType::MatMulBias:
            // Requires pre-compiled .mlmodelc; without it, falls back to Metal GPU.
            return true;
        case ANEOpType::Attention:
        case ANEOpType::FFN:
            // Supported via fused CompileTransformerLayer path (requires offline
            // CoreML compilation with coremltools + xcrun coremlcompiler).
            return true;
        case ANEOpType::RMSNorm:
        case ANEOpType::LayerNorm:
        case ANEOpType::SiLU:
        case ANEOpType::GeLU:
        case ANEOpType::Softmax:
            // These use Metal GPU fallback (preferred) or CPU Accelerate vDSP.
            // They are included in fused transformer layers when compiled offline.
            return false;
        default: return false;
        }
    }

    // ===========================================================================
    // ANE-Specific APIs
    // ===========================================================================

    /**
     * @brief Compile a MatMul operation for ANE execution
     *
     * Creates a CoreML model that represents: output = input @ weight^T + bias
     * The model is compiled with ANE-only compute units for maximum performance.
     *
     * @param name Unique identifier for this operation (e.g., "layer0_wq")
     * @param M Output dimension (rows of weight)
     * @param K Input dimension (columns of weight)
     * @param weight_data Pre-loaded weight tensor [M, K] row-major
     * @param bias_data Optional bias [M], nullptr for no bias
     * @return true if compilation succeeded
     */
    bool CompileMatMul(const std::string& name, int M, int K, const float* weight_data,
                       const float* bias_data = nullptr);

    /**
     * @brief Compile a FP16 MatMul for better ANE performance
     *
     * ANE performs best with FP16 operations. This method quantizes weights
     * to FP16 and creates an optimized CoreML model.
     *
     * @param name Unique identifier
     * @param M Output dimension
     * @param K Input dimension
     * @param weight_fp32 FP32 weights (will be converted to FP16)
     * @param bias_fp32 Optional FP32 bias
     * @return true if compilation succeeded
     */
    bool CompileMatMulFP16(const std::string& name, int M, int K, const float* weight_fp32,
                           const float* bias_fp32 = nullptr);

    /**
     * @brief Load a pre-compiled offline CoreML model for RoPE execution on ANE
     *
     * Looks for `<cacheDirectory>/<name>.mlmodelc` compiled with coremltools.
     * When found, subsequent RoPE() calls whose input dimensions match this model
     * will execute entirely on the ANE, avoiding ANE↔Metal context switches.
     *
     * Offline compilation (coremltools):
     * @code
     *   import coremltools as ct
     *   # Build RoPE MIL program for fixed seq_len/n_heads/head_dim
     *   # Expected inputs:  "input"     [seq_len * n_heads, head_dim] float16
     *   #                   "positions" [seq_len]                     int32
     *   # Expected output:  "output"    [seq_len * n_heads, head_dim] float16
     *   model.save("rope_seq128_h32_d128.mlpackage")
     *   # xcrun coremlcompiler compile rope_seq128_h32_d128.mlpackage <cacheDir>
     * @endcode
     *
     * @param name      Model identifier and filename stem (e.g. "rope_seq128_h32_d128")
     * @param seq_len   Sequence length the model was compiled for
     * @param n_heads   Number of attention heads
     * @param head_dim  Dimension per head
     * @return true if the cached model was loaded successfully
     */
    bool CompileRoPE(const std::string& name, int seq_len, int n_heads, int head_dim);

    /**
     * @brief Load a pre-compiled offline CoreML model for FlashAttention on ANE
     *
     * Looks for `<cacheDirectory>/<name>.mlmodelc` compiled with coremltools.
     * When found, FlashAttention() calls whose Q shape matches will run on ANE,
     * keeping the entire attention computation in a single ANE graph dispatch.
     *
     * Offline compilation (coremltools):
     * @code
     *   # Expected inputs:  "q"     [seq_len, n_heads,    head_dim] float16
     *   #                   "k"     [kv_len,  n_kv_heads, head_dim] float16
     *   #                   "v"     [kv_len,  n_kv_heads, head_dim] float16
     *   # Expected output:  "output" [seq_len, n_heads,   head_dim] float16
     *   # xcrun coremlcompiler compile flash_attn_seq128_h32_d128.mlpackage <cacheDir>
     * @endcode
     *
     * @param name      Model identifier and filename stem
     * @param seq_len   Query sequence length the model was compiled for
     * @param n_heads   Number of query attention heads
     * @param head_dim  Dimension per head
     * @return true if the cached model was loaded successfully
     */
    bool CompileFlashAttention(const std::string& name, int seq_len, int n_heads, int head_dim);

    /**
     * @brief Execute a pre-compiled MatMul
     *
     * @param name Operation identifier (must have been compiled)
     * @param input Input vector [K]
     * @param output Output vector [M]
     * @return true if execution succeeded
     */
    bool ExecuteMatMul(const std::string& name, const float* input, float* output);

    /**
     * @brief Compile entire transformer layer
     *
     * For maximum efficiency, compiles attention + FFN as a single CoreML model.
     * This reduces CoreML invocation overhead and allows ANE to optimize the
     * entire computation graph.
     *
     * @param name Layer identifier (e.g., "layer_0")
     * @param config Layer configuration
     * @return true if compilation succeeded
     */
    struct TransformerLayerConfig {
        int hidden_dim;
        int intermediate_dim;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int max_seq_len;
        float rms_norm_eps;
        bool use_qk_norm;
    };
    bool CompileTransformerLayer(const std::string& name, const TransformerLayerConfig& config,
                                 const void* layer_weights);

    /**
     * @brief Execute a pre-compiled transformer layer
     *
     * Performs fused attention + FFN in a single ANE dispatch.
     * Significantly reduces CPU-ANE context switching overhead.
     *
     * @param name Layer identifier (must have been compiled)
     * @param input Hidden states [seq_len, hidden_dim]
     * @param output Output hidden states [seq_len, hidden_dim]
     * @param positions Token positions for RoPE [seq_len]
     * @param seq_len Current sequence length
     * @return true if execution succeeded
     */
    bool ExecuteTransformerLayer(const std::string& name, const float* input, float* output, const int* positions,
                                 int seq_len);

    // ===========================================================================
    // Dynamic Sequence Length Support (Bucketed Models)
    // ===========================================================================

    /**
     * @brief Execute transformer layer with automatic bucket selection
     *
     * ANE requires fixed-shape models, but we can support variable sequence
     * lengths by pre-compiling models at common "bucket" sizes (e.g., 1, 32,
     * 128, 512, 1024, 2048, 4096). This method selects the appropriate bucket
     * and pads/slices the input accordingly.
     *
     * @param layer_prefix Layer name prefix (e.g., "layer_0")
     * @param input Hidden states [seq_len, hidden_dim]
     * @param output Output hidden states [seq_len, hidden_dim]
     * @param positions Token positions for RoPE [seq_len]
     * @param seq_len Actual sequence length (may be smaller than bucket)
     * @param hidden_dim Hidden dimension
     * @return true if execution succeeded
     *
     * @note Buckets must be pre-compiled using `PrecompileBucketedModels()`
     *       or manually with coremltools scripts.
     */
    bool ExecuteTransformerLayerDynamic(const std::string& layer_prefix, const float* input, float* output,
                                        const int* positions, int seq_len, const TransformerLayerConfig& config);

    /**
     * @brief Get available bucket sizes for dynamic sequence support
     * @return Vector of bucket sizes in ascending order
     */
    std::vector<int> GetBucketSizes() const;

    /**
     * @brief Set custom bucket sizes
     *
     * Override default bucket sizes for specific model requirements.
     * Must be called before PrecompileBucketedModels().
     *
     * @param sizes Bucket sizes in ascending order
     */
    void SetBucketSizes(const std::vector<int>& sizes);

    /**
     * @brief Pre-compile models at all bucket sizes
     *
     * Generates CoreML models for each bucket size. This is a slow operation
     * that should be done offline or during model initialization.
     *
     * @param layer_prefix Layer name prefix (e.g., "layer_0")
     * @param config Layer configuration
     * @param layer_weights Layer weight data
     * @return Number of successfully compiled buckets
     */
    int PrecompileBucketedModels(const std::string& layer_prefix, const TransformerLayerConfig& config,
                                 const void* layer_weights);

    /**
     * @brief Get operation status
     * @param name Operation identifier
     * @return Current status of the operation
     */
    ANEOpStatus GetOpStatus(const std::string& name) const;

    /**
     * @brief Get operation statistics
     * @param name Operation identifier
     * @return Statistics for the operation
     */
    ANEOpStats GetOpStats(const std::string& name) const;

    /**
     * @brief Check if operation is using ANE (vs CPU fallback)
     *
     * CoreML may fall back to CPU if ANE doesn't support the operation
     * or if ANE is busy with other work.
     *
     * @param name Operation identifier
     * @return true if operation runs on ANE
     */
    bool IsRunningOnANE(const std::string& name) const;

    /**
     * @brief Set compute units preference
     *
     * @param ane_only If true, only use ANE (fail if unsupported)
     *                 If false, allow CPU fallback
     */
    void SetANEOnlyMode(bool ane_only);

    /**
     * @brief Clear all compiled operations
     *
     * Useful for memory cleanup when switching models.
     */
    void ClearCompiledOps();

    /**
     * @brief Get cache directory for compiled models
     * @return Path to cache directory
     */
    const char* GetCacheDirectory() const;

    /**
     * @brief Set cache directory
     * @param path Path to cache directory (must exist)
     */
    void SetCacheDirectory(const char* path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace densecore

#endif  // __APPLE__

#endif  // DENSECORE_ANE_BACKEND_H
