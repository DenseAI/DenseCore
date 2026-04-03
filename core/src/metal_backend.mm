/**
 * @file metal_backend.mm
 * @brief Apple Metal GPU backend implementation
 *
 * Objective-C++ implementation of the Metal compute backend.
 * Integrates with GGML's Metal backend while providing custom
 * kernels for performance-critical paths (GEMV, FlashAttention).
 *
 * Architecture:
 * - Uses GGML Metal for graph execution (leverages proven codebase)
 * - Custom Metal shaders for decode-phase GEMV (parallel reduction)
 * - MTLStorageModeShared for zero-copy unified memory
 * - Per-thread command encoders for concurrent graph building
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/metal_backend.h"

#include "../include/apple_silicon.h"

#ifdef __APPLE__

#import <Accelerate/Accelerate.h>  // For cblas_sgemm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// GGML Metal backend integration
extern "C" {
#include "ggml-backend.h"
#include "ggml-metal.h"
}

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <mutex>
#include <unordered_map>

namespace densecore {

// ============================================================================
// Private Implementation (Pimpl)
// ============================================================================

struct MetalBackend::Impl {
    // Core Metal objects
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> shaderLibrary = nil;

    // Custom compute pipeline states
    id<MTLComputePipelineState> gemvPipeline = nil;
    id<MTLComputePipelineState> gemvBatchedPipeline = nil;
    id<MTLComputePipelineState> softmaxPipeline = nil;
    id<MTLComputePipelineState> rmsNormPipeline = nil;
    id<MTLComputePipelineState> addRmsNormPipeline = nil;
    id<MTLComputePipelineState> flashAttentionDecodePipeline = nil;
    id<MTLComputePipelineState> flashAttentionPrefillPipeline = nil;

    // Quantized GEMV pipeline states
    id<MTLComputePipelineState> gemvQ4_0Pipeline = nil;
    id<MTLComputePipelineState> gemvQ4_1Pipeline = nil;
    id<MTLComputePipelineState> gemvQ8_0Pipeline = nil;
    id<MTLComputePipelineState> gemmQ4_0FusedPipeline = nil;
    id<MTLComputePipelineState> gemmQ4_1FusedPipeline = nil;
    id<MTLComputePipelineState> gemmInt4GroupedFusedPipeline = nil;
    id<MTLComputePipelineState> gemmQ4_0SimdgroupPipeline = nil;
    id<MTLComputePipelineState> gemmQ4_1SimdgroupPipeline = nil;
    id<MTLComputePipelineState> gemmInt4GroupedSimdgroupPipeline = nil;

    // Fused QKV pipeline state
    id<MTLComputePipelineState> fusedQKVPipeline = nil;

    // RoPE pipeline state
    id<MTLComputePipelineState> ropePipeline = nil;

    // Dequantization pipeline for M>1 GEMM
    id<MTLComputePipelineState> dequantizeQ4_0Pipeline = nil;
    id<MTLComputePipelineState> dequantizeQ4_1Pipeline = nil;

    // Group-wise INT4 dequantization (for ANE GEMM fallback)
    id<MTLComputePipelineState> dequantizeInt4GroupedPipeline = nil;

    // GGML Metal backend (for graph execution)
    ggml_backend_t ggmlMetalBackend = nullptr;

    // Memory tracking
    std::atomic<size_t> currentMemoryUsage{0};
    std::atomic<size_t> peakMemoryUsage{0};

    // Buffer registry: maps contents pointer -> MTLBuffer for proper deallocation
    std::mutex bufferRegistryMutex;
    std::unordered_map<void*, id<MTLBuffer>> bufferRegistry;

    // Wrapped buffer cache for untracked pointers (zero-copy wrappers)
    std::mutex wrappedBufferMutex;
    struct WrappedBufferEntry {
        id<MTLBuffer> buffer = nil;
        std::list<void*>::iterator lru_it;
    };
    std::unordered_map<void*, WrappedBufferEntry> wrappedBufferRegistry;
    std::list<void*> wrappedBufferLru;
    size_t maxWrappedBuffers = 1024;

    // Buffer pool for small allocations
    std::mutex bufferPoolMutex;
    std::vector<id<MTLBuffer>> bufferPool;

    // Chip information (cached)
    AppleSiliconChipInfo chipInfo;

    // GPU capture state
    bool captureEnabled = false;
    bool supportsSimdgroupMatrix = false;
    bool enableSimdgroupInt4Gemm = true;

    // Helper: Get MTLBuffer for a tracked pointer (for MPS zero-copy)
    id<MTLBuffer> GetBufferForPointer(void* ptr) {
        std::lock_guard<std::mutex> lock(bufferRegistryMutex);
        auto it = bufferRegistry.find(ptr);
        return (it != bufferRegistry.end()) ? it->second : nil;
    }

    // Helper: Wrap an untracked pointer with zero-copy MTLBuffer
    // Uses newBufferWithBytesNoCopy for UMA zero-copy access
    id<MTLBuffer> WrapPointerNoCopy(void* ptr, size_t size) {
        if (!ptr || size == 0)
            return nil;
        // newBufferWithBytesNoCopy requires page-aligned memory on some systems
        // For non-aligned, fall back to newBufferWithBytes
        return [device newBufferWithBytesNoCopy:ptr
                                         length:size
                                        options:MTLResourceStorageModeShared
                                    deallocator:nil];
    }

    // Helper: Get or wrap buffer (prefers tracked, falls back to zero-copy wrap)
    id<MTLBuffer> GetOrWrapBuffer(void* ptr, size_t size) {
        id<MTLBuffer> buffer = GetBufferForPointer(ptr);
        if (buffer)
            return buffer;
        {
            std::lock_guard<std::mutex> lock(wrappedBufferMutex);
            auto it = wrappedBufferRegistry.find(ptr);
            if (it != wrappedBufferRegistry.end()) {
                wrappedBufferLru.splice(wrappedBufferLru.begin(), wrappedBufferLru,
                                        it->second.lru_it);
                return it->second.buffer;
            }
        }
        // Try zero-copy wrap first
        buffer = WrapPointerNoCopy(ptr, size);
        if (buffer) {
            std::lock_guard<std::mutex> lock(wrappedBufferMutex);
            if (maxWrappedBuffers > 0) {
                CFRetain((__bridge CFTypeRef)buffer);
                wrappedBufferLru.push_front(ptr);
                WrappedBufferEntry entry;
                entry.buffer = buffer;
                entry.lru_it = wrappedBufferLru.begin();
                wrappedBufferRegistry[ptr] = entry;
                if (wrappedBufferRegistry.size() > maxWrappedBuffers) {
                    void* evict_ptr = wrappedBufferLru.back();
                    wrappedBufferLru.pop_back();
                    auto evict_it = wrappedBufferRegistry.find(evict_ptr);
                    if (evict_it != wrappedBufferRegistry.end()) {
                        if (evict_it->second.buffer) {
                            CFRelease((__bridge CFTypeRef)evict_it->second.buffer);
                        }
                        wrappedBufferRegistry.erase(evict_it);
                    }
                }
            }
            return buffer;
        }
        // Last resort: copy data (shouldn't happen on Apple Silicon)
        buffer = [device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
        if (buffer) {
            std::lock_guard<std::mutex> lock(wrappedBufferMutex);
            if (maxWrappedBuffers > 0) {
                CFRetain((__bridge CFTypeRef)buffer);
                wrappedBufferLru.push_front(ptr);
                WrappedBufferEntry entry;
                entry.buffer = buffer;
                entry.lru_it = wrappedBufferLru.begin();
                wrappedBufferRegistry[ptr] = entry;
                if (wrappedBufferRegistry.size() > maxWrappedBuffers) {
                    void* evict_ptr = wrappedBufferLru.back();
                    wrappedBufferLru.pop_back();
                    auto evict_it = wrappedBufferRegistry.find(evict_ptr);
                    if (evict_it != wrappedBufferRegistry.end()) {
                        if (evict_it->second.buffer) {
                            CFRelease((__bridge CFTypeRef)evict_it->second.buffer);
                        }
                        wrappedBufferRegistry.erase(evict_it);
                    }
                }
            }
        }
        return buffer;
    }

    // Scratch buffer cache for temporary allocations (reduces alloc overhead)
    std::mutex scratchBufferMutex;
    id<MTLBuffer> scratchBuffer = nil;
    size_t scratchBufferSize = 0;

    // Helper: Get a scratch buffer of at least the requested size
    // Reuses existing buffer if large enough, otherwise reallocates
    id<MTLBuffer> GetScratchBuffer(size_t size) {
        std::lock_guard<std::mutex> lock(scratchBufferMutex);
        if (scratchBuffer && scratchBufferSize >= size) {
            return scratchBuffer;
        }
        // Allocate with some headroom to reduce reallocs
        size_t allocSize = size + (size / 4);  // 25% headroom
        scratchBuffer = [device newBufferWithLength:allocSize options:MTLResourceStorageModeShared];
        if (scratchBuffer) {
            scratchBufferSize = allocSize;
        }
        return scratchBuffer;
    }

    ~Impl() {
        // Release GGML Metal backend
        if (ggmlMetalBackend) {
            ggml_backend_free(ggmlMetalBackend);
            ggmlMetalBackend = nullptr;
        }

        // Release pipeline states
        gemvPipeline = nil;
        gemvBatchedPipeline = nil;
        softmaxPipeline = nil;
        rmsNormPipeline = nil;
        addRmsNormPipeline = nil;
        flashAttentionDecodePipeline = nil;
        flashAttentionPrefillPipeline = nil;
        gemvQ4_0Pipeline = nil;
        gemvQ4_1Pipeline = nil;
        gemvQ8_0Pipeline = nil;
        gemmQ4_0FusedPipeline = nil;
        gemmQ4_1FusedPipeline = nil;
        gemmInt4GroupedFusedPipeline = nil;
        gemmQ4_0SimdgroupPipeline = nil;
        gemmQ4_1SimdgroupPipeline = nil;
        gemmInt4GroupedSimdgroupPipeline = nil;
        fusedQKVPipeline = nil;
        ropePipeline = nil;
        dequantizeQ4_0Pipeline = nil;
        dequantizeQ4_1Pipeline = nil;
        dequantizeInt4GroupedPipeline = nil;

        // Release scratch buffer
        scratchBuffer = nil;
        scratchBufferSize = 0;

        // Release shader library
        shaderLibrary = nil;

        // Release all tracked buffers
        {
            std::lock_guard<std::mutex> lock(bufferRegistryMutex);
            for (auto& [ptr, buffer] : bufferRegistry) {
                if (buffer) {
                    CFRelease((__bridge CFTypeRef)buffer);
                }
            }
            bufferRegistry.clear();
        }

        // Clear buffer pool
        {
            std::lock_guard<std::mutex> lock(bufferPoolMutex);
            bufferPool.clear();
        }

        // Release wrapped buffers
        {
            std::lock_guard<std::mutex> lock(wrappedBufferMutex);
            for (auto& entry : wrappedBufferRegistry) {
                if (entry.second.buffer) {
                    CFRelease((__bridge CFTypeRef)entry.second.buffer);
                }
            }
            wrappedBufferRegistry.clear();
            wrappedBufferLru.clear();
        }

        // Release command queue and device
        commandQueue = nil;
        device = nil;
    }
};

// ============================================================================
// Custom Metal Shader Source
// ============================================================================

namespace {

int ParsePositiveEnvInt(const char* key, int fallback, int min_value, int max_value) {
    if (const char* env = std::getenv(key)) {
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end != env) {
            if (parsed < min_value)
                parsed = min_value;
            if (parsed > max_value)
                parsed = max_value;
            return static_cast<int>(parsed);
        }
    }
    return fallback;
}

int MetalSmallBatchGemvMax() {
    static const int value =
        ParsePositiveEnvInt("DENSECORE_METAL_SMALL_BATCH_GEMV_MAX_COLS", 8, 1, 32);
    return value;
}

/**
 * @brief Embedded Metal shader source for custom kernels
 *
 * These shaders are compiled at runtime if the .metallib is not found.
 * Production builds should use pre-compiled .metallib for faster startup.
 *
 * Optimizations:
 * - SIMD-group intrinsics for warp-level reductions (32-wide on Apple GPUs)
 * - Minimal threadgroup barriers
 * - FMA instructions for better throughput
 */
const char* kMetalShaderSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Apple GPU SIMD width constant
constant uint SIMD_WIDTH = 32;

// =============================================================================
// SIMD-Optimized GEMV Kernel: output = weight @ input ([M,K] @ [K] = [M])
// =============================================================================
// Key optimizations:
// 1. Uses simd_sum() for warp-level reduction (no shared memory needed for first step)
// 2. Only one threadgroup barrier after SIMD reduction
// 3. Each simdgroup handles reduction independently
// 4. Final reduction across simdgroups uses minimal shared memory
// =============================================================================

kernel void gemv_f32(
    device const float* input [[buffer(0)]],      // [K]
    device const float* weight [[buffer(1)]],     // [M, K]
    device float* output [[buffer(2)]],           // [M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    // Each threadgroup handles one output row
    uint row = tgid;
    if (row >= M) return;

    device const float* weight_row = weight + row * K;

    // Phase 1: Each thread accumulates its portion of the dot product
    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input[k], sum);
    }

    // Phase 2: SIMD-level reduction using simd_sum (warp-level, no barrier needed)
    sum = simd_sum(sum);

    // Phase 3: First lane of each simdgroup writes to shared memory
    // Only need as many slots as simdgroups (typically 8 for 256 threads)
    threadgroup float simd_results[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    if (simd_lane == 0 && simd_group < num_simdgroups) {
        simd_results[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: First thread reduces across simdgroups
    if (tid == 0) {
        float final_sum = 0.0f;
        for (uint i = 0; i < num_simdgroups; ++i) {
            final_sum += simd_results[i];
        }
        output[row] = final_sum;
    }
}

// =============================================================================
// Small-Batch GEMV Kernel: output = input @ weight, for 2 <= batch <= 8
// =============================================================================
// Launch layout:
// - threadgroup_position_in_grid.x -> output row index (N)
// - threadgroup_position_in_grid.y -> batch row index (B)
// =============================================================================
kernel void gemv_f32_batched(
    device const float* input [[buffer(0)]],      // [B, K]
    device const float* weight [[buffer(1)]],     // [N, K] (same layout as gemv_f32 path)
    device float* output [[buffer(2)]],           // [B, N]
    constant uint& B [[buffer(3)]],               // Batch size
    constant uint& N [[buffer(4)]],               // Output dimension
    constant uint& K [[buffer(5)]],               // Input dimension
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid.x;
    const uint batch = tgid.y;
    if (batch >= B || row >= N) return;

    device const float* input_row = input + batch * K;
    device const float* weight_row = weight + row * K;
    device float* output_row = output + batch * N;

    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input_row[k], sum);
    }

    sum = simd_sum(sum);

    threadgroup float simd_results[8];
    const uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;
    if (simd_lane == 0 && simd_group < num_simdgroups) {
        simd_results[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_sum = 0.0f;
        for (uint i = 0; i < num_simdgroups; ++i) {
            final_sum += simd_results[i];
        }
        output_row[row] = final_sum;
    }
}

// =============================================================================
// SIMD-Optimized Softmax Kernel
// =============================================================================

kernel void softmax_f32(
    device float* data [[buffer(0)]],     // [rows, stride]
    constant uint& N [[buffer(1)]],       // Softmax width
    constant uint& stride [[buffer(2)]],  // Row stride in elements
    constant uint& rows [[buffer(3)]],    // Number of rows
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (row_idx >= rows) return;

    device float* row = data + row_idx * stride;
    threadgroup float simd_max[8];
    threadgroup float simd_sum_vals[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Phase 1: Find local max
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, row[i]);
    }

    // SIMD reduction for max
    local_max = simd_max(local_max);
    if (simd_lane == 0) { simd_max[simd_group] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max across simdgroups
    float max_val = simd_max[0];
    for (uint i = 1; i < num_simdgroups; ++i) {
        max_val = max(max_val, simd_max[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exp and local sum
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float e = exp(row[i] - max_val);
        row[i] = e;
        local_sum += e;
    }

    // SIMD reduction for sum
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) { simd_sum_vals[simd_group] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global sum across simdgroups
    float sum_val = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        sum_val += simd_sum_vals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum_val;
    for (uint i = tid; i < N; i += tg_size) {
        row[i] *= inv_sum;
    }
}

// =============================================================================
// SIMD-Optimized RMS Normalization Kernel
// =============================================================================

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],      // [N, dim]
    device const float* weight [[buffer(1)]],     // [dim]
    device float* output [[buffer(2)]],           // [N, dim]
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint row = tgid;
    device const float* input_row = input + row * dim;
    device float* output_row = output + row * dim;

    // Phase 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = input_row[i];
        sum_sq = fma(val, val, sum_sq);
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) { simd_sums[simd_group] = sum_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global sum
    float total_sum_sq = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        total_sum_sq += simd_sums[i];
    }

    float rms = rsqrt(total_sum_sq / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Apply normalization and weight
    for (uint i = tid; i < dim; i += tg_size) {
        output_row[i] = input_row[i] * rms * weight[i];
    }
}

// =============================================================================
// SIMD-Optimized Fused Add + RMSNorm Kernel
// =============================================================================

kernel void add_rms_norm_f32(
    device const float* input [[buffer(0)]],      // [N, dim]
    device const float* residual [[buffer(1)]],   // [N, dim]
    device const float* weight [[buffer(2)]],     // [dim]
    device float* output [[buffer(3)]],           // [N, dim]
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint row = tgid;
    device const float* input_row = input + row * dim;
    device const float* residual_row = residual + row * dim;
    device float* output_row = output + row * dim;

    // Phase 1: Accumulate sum((x + residual)^2).
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = input_row[i] + residual_row[i];
        sum_sq = fma(val, val, sum_sq);
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) { simd_sums[simd_group] = sum_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        total_sum_sq += simd_sums[i];
    }
    float rms = rsqrt(total_sum_sq / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Apply fused add + RMSNorm + weight scale.
    for (uint i = tid; i < dim; i += tg_size) {
        float val = input_row[i] + residual_row[i];
        output_row[i] = val * rms * weight[i];
    }
}

// =============================================================================
// RoPE (Rotary Positional Embedding) Kernel
// =============================================================================
// Layout: [n_heads, n_tokens, head_dim] or [batch*n_heads, seq_len, head_dim]
// Each thread handles one pair of elements (2*d, 2*d+1)

kernel void rope_f32(
    device float* data [[buffer(0)]],              // In-place modification
    device const float* cos_sin [[buffer(1)]],    // [max_seq, head_dim]
    device const int* positions [[buffer(2)]],    // [n_tokens]
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_tokens [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rope_dim [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])        // (pair_idx, token, head)
{
    uint pair_idx = tid.x;  // Which pair (0 to rope_dim/2 - 1)
    uint token = tid.y;
    uint head = tid.z;

    if (pair_idx >= rope_dim / 2 || token >= n_tokens || head >= n_heads) return;

    int pos = positions[token];
    device const float* pos_cs = cos_sin + pos * head_dim;

    float cos_theta = pos_cs[2 * pair_idx];
    float sin_theta = pos_cs[2 * pair_idx + 1];

    // Index into data: [head, token, dim]
    uint base_idx = (head * n_tokens + token) * head_dim + 2 * pair_idx;

    float x0 = data[base_idx];
    float x1 = data[base_idx + 1];

    data[base_idx] = fma(x0, cos_theta, -x1 * sin_theta);
    data[base_idx + 1] = fma(x0, sin_theta, x1 * cos_theta);
}
)METAL";

/**
 * @brief Quantized GEMV shader source for INT4/INT8 weights
 *
 * Kernels:
 * - gemv_q4_0: Q4_0 format (scale only, 4-bit weights centered at 8)
 * - gemv_q4_1: Q4_1 format (scale + min, unsigned 4-bit weights)
 * - gemv_q8_0: Q8_0 format (scale, 8-bit weights)
 */
const char* kQuantizedGemvShaderSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint QK4_0 = 32;
constant uint QK4_1 = 32;
constant uint QK8_0 = 32;
constant uint THREADGROUP_SIZE = 256;
constant uint PREFILL_TILE_M = 8;
constant uint PREFILL_TILE_N = 8;
constant uint SIMD_TILE = 8;
constant uint SIMDGROUP_THREADS = 32;

struct block_q4_0 {
    half scale;
    uint8_t quants[QK4_0 / 2];
};

struct block_q4_1 {
    half scale;
    half min;
    uint8_t quants[QK4_1 / 2];
};

struct block_q8_0 {
    half scale;
    int8_t quants[QK8_0];
};

inline int8_t extract_q4(uint8_t packed, uint idx) {
    int8_t val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return val - 8;
}

inline uint8_t extract_q4_unsigned(uint8_t packed, uint idx) {
    return (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
}

inline float grouped_int4_value(device const uchar* weights, device const float* scales,
                                device const float* zeros, uint n, uint k, uint K,
                                uint group_size, bool has_zero_points) {
    const uint k_packed = K / 2;
    const uint byte_idx = n * k_packed + (k / 2);
    const uchar packed = weights[byte_idx];
    int q = (k & 1) ? int((packed >> 4) & 0x0F) : int(packed & 0x0F);
    if (q > 7) {
        q -= 16;
    }
    const uint groups_per_row = K / group_size;
    const uint group_idx = min(k / group_size, groups_per_row - 1);
    const uint qparam_idx = n * groups_per_row + group_idx;
    const float zero = has_zero_points ? zeros[qparam_idx] : 0.0f;
    return scales[qparam_idx] * (float(q) - zero);
}

kernel void gemv_q4_0(
    device const float* input [[buffer(0)]],
    device const block_q4_0* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK4_0;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q4_0* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        uint k_start = block_idx * QK4_0;

        for (uint i = 0; i < QK4_0; i += 2) {
            uint byte_idx = i / 2;
            uint8_t packed = block->quants[byte_idx];
            float w0 = float(extract_q4(packed, 0)) * scale;
            float w1 = float(extract_q4(packed, 1)) * scale;
            sum = fma(w0, input[k_start + i], sum);
            sum = fma(w1, input[k_start + i + 1], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

kernel void gemv_q4_1(
    device const float* input [[buffer(0)]],
    device const block_q4_1* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK4_1;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q4_1* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        float min_val = float(block->min);
        uint k_start = block_idx * QK4_1;

        for (uint i = 0; i < QK4_1; i += 2) {
            uint byte_idx = i / 2;
            uint8_t packed = block->quants[byte_idx];
            float w0 = float(extract_q4_unsigned(packed, 0)) * scale + min_val;
            float w1 = float(extract_q4_unsigned(packed, 1)) * scale + min_val;
            sum = fma(w0, input[k_start + i], sum);
            sum = fma(w1, input[k_start + i + 1], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

kernel void gemv_q8_0(
    device const float* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK8_0;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q8_0* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        uint k_start = block_idx * QK8_0;

        for (uint i = 0; i < QK8_0; ++i) {
            float w = float(block->quants[i]) * scale;
            sum = fma(w, input[k_start + i], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

// =============================================================================
// Fused Q4 GEMM Kernels (for M>1 prefill)
// =============================================================================
// These kernels keep packed INT4 weights in their compressed form until the
// dot-product stage. Each threadgroup cooperatively stages an activation tile
// and a dequantized weight tile in threadgroup memory, avoiding the previous
// global-memory FP32 expansion pass before GEMM.

kernel void gemm_q4_0_fused(
    device const float* input [[buffer(0)]],          // [M, K]
    device const block_q4_0* weight [[buffer(1)]],    // [N, ceil(K / 32)]
    device float* output [[buffer(2)]],               // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    const uint local_n = lid.x;
    const uint local_m = lid.y;
    const uint linear_tid = local_m * PREFILL_TILE_N + local_n;
    const uint m = group_id.y * PREFILL_TILE_M + local_m;
    const uint n = group_id.x * PREFILL_TILE_N + local_n;
    const uint blocks_per_row = (K + QK4_0 - 1) / QK4_0;

    threadgroup float a_tile[PREFILL_TILE_M][QK4_0];
    threadgroup float w_tile[PREFILL_TILE_N][QK4_0];

    float sum = 0.0f;

    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint k_start = block_idx * QK4_0;

        for (uint idx = linear_tid; idx < PREFILL_TILE_M * QK4_0; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_m = idx / QK4_0;
            const uint tile_k = idx % QK4_0;
            const uint global_m = group_id.y * PREFILL_TILE_M + tile_m;
            const uint global_k = k_start + tile_k;
            a_tile[tile_m][tile_k] =
                (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = linear_tid; idx < PREFILL_TILE_N * QK4_0; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_n = idx / QK4_0;
            const uint tile_k = idx % QK4_0;
            const uint global_n = group_id.x * PREFILL_TILE_N + tile_n;
            const uint global_k = k_start + tile_k;
            float value = 0.0f;
            if (global_n < N && global_k < K) {
                device const block_q4_0* block = &weight[global_n * blocks_per_row + block_idx];
                const uint8_t packed = block->quants[tile_k / 2];
                value = float(extract_q4(packed, tile_k & 1)) * float(block->scale);
            }
            w_tile[tile_n][tile_k] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            const uint k_limit = min(QK4_0, K - k_start);
            for (uint kk = 0; kk < k_limit; ++kk) {
                sum = fma(a_tile[local_m][kk], w_tile[local_n][kk], sum);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        output[m * N + n] = sum;
    }
}

kernel void gemm_q4_1_fused(
    device const float* input [[buffer(0)]],          // [M, K]
    device const block_q4_1* weight [[buffer(1)]],    // [N, ceil(K / 32)]
    device float* output [[buffer(2)]],               // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    const uint local_n = lid.x;
    const uint local_m = lid.y;
    const uint linear_tid = local_m * PREFILL_TILE_N + local_n;
    const uint m = group_id.y * PREFILL_TILE_M + local_m;
    const uint n = group_id.x * PREFILL_TILE_N + local_n;
    const uint blocks_per_row = (K + QK4_1 - 1) / QK4_1;

    threadgroup float a_tile[PREFILL_TILE_M][QK4_1];
    threadgroup float w_tile[PREFILL_TILE_N][QK4_1];

    float sum = 0.0f;

    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint k_start = block_idx * QK4_1;

        for (uint idx = linear_tid; idx < PREFILL_TILE_M * QK4_1; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_m = idx / QK4_1;
            const uint tile_k = idx % QK4_1;
            const uint global_m = group_id.y * PREFILL_TILE_M + tile_m;
            const uint global_k = k_start + tile_k;
            a_tile[tile_m][tile_k] =
                (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = linear_tid; idx < PREFILL_TILE_N * QK4_1; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_n = idx / QK4_1;
            const uint tile_k = idx % QK4_1;
            const uint global_n = group_id.x * PREFILL_TILE_N + tile_n;
            const uint global_k = k_start + tile_k;
            float value = 0.0f;
            if (global_n < N && global_k < K) {
                device const block_q4_1* block = &weight[global_n * blocks_per_row + block_idx];
                const uint8_t packed = block->quants[tile_k / 2];
                value = float(extract_q4_unsigned(packed, tile_k & 1)) * float(block->scale) +
                        float(block->min);
            }
            w_tile[tile_n][tile_k] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            const uint k_limit = min(QK4_1, K - k_start);
            for (uint kk = 0; kk < k_limit; ++kk) {
                sum = fma(a_tile[local_m][kk], w_tile[local_n][kk], sum);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        output[m * N + n] = sum;
    }
}

kernel void gemm_int4_grouped_fused(
    device const float* input [[buffer(0)]],     // [M, K]
    device const uchar* weights [[buffer(1)]],   // [N, K/2]
    device const float* scales [[buffer(2)]],    // [N, K/group_size]
    device const float* zeros [[buffer(3)]],     // [N, K/group_size] (optional)
    device float* output [[buffer(4)]],          // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& has_zero_points [[buffer(9)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    if (group_size == 0) {
        return;
    }

    const uint local_n = lid.x;
    const uint local_m = lid.y;
    const uint linear_tid = local_m * PREFILL_TILE_N + local_n;
    const uint m = group_id.y * PREFILL_TILE_M + local_m;
    const uint n = group_id.x * PREFILL_TILE_N + local_n;
    const uint groups_per_row = K / group_size;
    const uint k_packed = K / 2;

    if (groups_per_row == 0 || k_packed == 0) {
        return;
    }

    threadgroup float a_tile[PREFILL_TILE_M][QK4_0];
    threadgroup float w_tile[PREFILL_TILE_N][QK4_0];

    float sum = 0.0f;
    const uint blocks_per_row = (K + QK4_0 - 1) / QK4_0;

    for (uint block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint k_start = block_idx * QK4_0;

        for (uint idx = linear_tid; idx < PREFILL_TILE_M * QK4_0; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_m = idx / QK4_0;
            const uint tile_k = idx % QK4_0;
            const uint global_m = group_id.y * PREFILL_TILE_M + tile_m;
            const uint global_k = k_start + tile_k;
            a_tile[tile_m][tile_k] =
                (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = linear_tid; idx < PREFILL_TILE_N * QK4_0; idx += PREFILL_TILE_M * PREFILL_TILE_N) {
            const uint tile_n = idx / QK4_0;
            const uint tile_k = idx % QK4_0;
            const uint global_n = group_id.x * PREFILL_TILE_N + tile_n;
            const uint global_k = k_start + tile_k;
            float value = 0.0f;
            if (global_n < N && global_k < K) {
                const uint byte_idx = global_n * k_packed + (global_k / 2);
                const uchar packed = weights[byte_idx];
                int q = (global_k & 1) ? int((packed >> 4) & 0x0F) : int(packed & 0x0F);
                if (q > 7) {
                    q -= 16;
                }
                const uint group_idx = min(global_k / group_size, groups_per_row - 1);
                const uint qparam_idx = global_n * groups_per_row + group_idx;
                const float zero = has_zero_points ? zeros[qparam_idx] : 0.0f;
                value = scales[qparam_idx] * (float(q) - zero);
            }
            w_tile[tile_n][tile_k] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            const uint k_limit = min(QK4_0, K - k_start);
            for (uint kk = 0; kk < k_limit; ++kk) {
                sum = fma(a_tile[local_m][kk], w_tile[local_n][kk], sum);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        output[m * N + n] = sum;
    }
}

kernel void gemm_q4_0_simdgroup_fused(
    device const float* input [[buffer(0)]],
    device const block_q4_0* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint tile_m0 = group_id.y * SIMD_TILE;
    const uint tile_n0 = group_id.x * SIMD_TILE;
    const uint blocks_per_row = (K + QK4_0 - 1) / QK4_0;

    threadgroup float a_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float b_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float c_tile[SIMD_TILE * SIMD_TILE];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += SIMD_TILE) {
        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_m = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_m = tile_m0 + local_m;
            const uint global_k = k0 + local_k;
            a_tile[idx] = (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_n = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_n = tile_n0 + local_n;
            const uint global_k = k0 + local_k;
            float value = 0.0f;
            if (global_n < N && global_k < K) {
                device const block_q4_0* block = &weight[global_n * blocks_per_row + (global_k / QK4_0)];
                const uint within_block = global_k % QK4_0;
                const uint8_t packed = block->quants[within_block / 2];
                value = float(extract_q4(packed, within_block & 1)) * float(block->scale);
            }
            b_tile[idx] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma;
        simdgroup_float8x8 mb;
        simdgroup_load(ma, a_tile, SIMD_TILE, ulong2(0, 0), false);
        simdgroup_load(mb, b_tile, SIMD_TILE, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_tile, SIMD_TILE, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
        const uint local_m = idx / SIMD_TILE;
        const uint local_n = idx % SIMD_TILE;
        const uint global_m = tile_m0 + local_m;
        const uint global_n = tile_n0 + local_n;
        if (global_m < M && global_n < N) {
            output[global_m * N + global_n] = c_tile[idx];
        }
    }
}

kernel void gemm_q4_1_simdgroup_fused(
    device const float* input [[buffer(0)]],
    device const block_q4_1* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint tile_m0 = group_id.y * SIMD_TILE;
    const uint tile_n0 = group_id.x * SIMD_TILE;
    const uint blocks_per_row = (K + QK4_1 - 1) / QK4_1;

    threadgroup float a_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float b_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float c_tile[SIMD_TILE * SIMD_TILE];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += SIMD_TILE) {
        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_m = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_m = tile_m0 + local_m;
            const uint global_k = k0 + local_k;
            a_tile[idx] = (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_n = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_n = tile_n0 + local_n;
            const uint global_k = k0 + local_k;
            float value = 0.0f;
            if (global_n < N && global_k < K) {
                device const block_q4_1* block = &weight[global_n * blocks_per_row + (global_k / QK4_1)];
                const uint within_block = global_k % QK4_1;
                const uint8_t packed = block->quants[within_block / 2];
                value = float(extract_q4_unsigned(packed, within_block & 1)) * float(block->scale) +
                        float(block->min);
            }
            b_tile[idx] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma;
        simdgroup_float8x8 mb;
        simdgroup_load(ma, a_tile, SIMD_TILE, ulong2(0, 0), false);
        simdgroup_load(mb, b_tile, SIMD_TILE, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_tile, SIMD_TILE, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
        const uint local_m = idx / SIMD_TILE;
        const uint local_n = idx % SIMD_TILE;
        const uint global_m = tile_m0 + local_m;
        const uint global_n = tile_n0 + local_n;
        if (global_m < M && global_n < N) {
            output[global_m * N + global_n] = c_tile[idx];
        }
    }
}

kernel void gemm_int4_grouped_simdgroup_fused(
    device const float* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& has_zero_points [[buffer(9)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (group_size == 0) {
        return;
    }

    const uint tile_m0 = group_id.y * SIMD_TILE;
    const uint tile_n0 = group_id.x * SIMD_TILE;

    threadgroup float a_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float b_tile[SIMD_TILE * SIMD_TILE];
    threadgroup float c_tile[SIMD_TILE * SIMD_TILE];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += SIMD_TILE) {
        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_m = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_m = tile_m0 + local_m;
            const uint global_k = k0 + local_k;
            a_tile[idx] = (global_m < M && global_k < K) ? input[global_m * K + global_k] : 0.0f;
        }

        for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
            const uint local_n = idx / SIMD_TILE;
            const uint local_k = idx % SIMD_TILE;
            const uint global_n = tile_n0 + local_n;
            const uint global_k = k0 + local_k;
            b_tile[idx] = (global_n < N && global_k < K)
                              ? grouped_int4_value(weights, scales, zeros, global_n, global_k, K,
                                                   group_size, has_zero_points != 0)
                              : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma;
        simdgroup_float8x8 mb;
        simdgroup_load(ma, a_tile, SIMD_TILE, ulong2(0, 0), false);
        simdgroup_load(mb, b_tile, SIMD_TILE, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_tile, SIMD_TILE, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < SIMD_TILE * SIMD_TILE; idx += SIMDGROUP_THREADS) {
        const uint local_m = idx / SIMD_TILE;
        const uint local_n = idx % SIMD_TILE;
        const uint global_m = tile_m0 + local_m;
        const uint global_n = tile_n0 + local_n;
        if (global_m < M && global_n < N) {
            output[global_m * N + global_n] = c_tile[idx];
        }
    }
}

// =============================================================================
// Q4_0 Block Dequantization Kernel (for M>1 GEMM)
// =============================================================================
// Converts packed Q4_0 weights to FP32 for MPSMatrixMultiplication
// Grid: (blocks_per_row, N, 1), each thread handles one block

kernel void dequantize_q4_0(
    device const block_q4_0* input [[buffer(0)]],   // [N, blocks_per_row]
    device float* output [[buffer(1)]],              // [N, K]
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])           // (block_idx, row_n)
{
    uint block_idx = tid.x;
    uint row = tid.y;
    uint blocks_per_row = (K + QK4_0 - 1) / QK4_0;  // Ceiling division

    if (block_idx >= blocks_per_row || row >= N) return;

    device const block_q4_0* block = &input[row * blocks_per_row + block_idx];
    float scale = float(block->scale);
    uint k_start = block_idx * QK4_0;
    uint k_end = min(k_start + QK4_0, K);  // Handle partial last block

    device float* out_row = output + row * K + k_start;

    for (uint i = 0; i < k_end - k_start; i += 2) {
        uint8_t packed = block->quants[i / 2];
        out_row[i] = float(extract_q4(packed, 0)) * scale;
        if (i + 1 < k_end - k_start) {
            out_row[i + 1] = float(extract_q4(packed, 1)) * scale;
        }
    }
}

// =============================================================================
// Q4_1 Block Dequantization Kernel (for M>1 GEMM)
// =============================================================================
// Converts packed Q4_1 weights to FP32 for MPSMatrixMultiplication
// Grid: (blocks_per_row, N, 1), each thread handles one block
kernel void dequantize_q4_1(
    device const block_q4_1* input [[buffer(0)]],   // [N, blocks_per_row]
    device float* output [[buffer(1)]],              // [N, K]
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])           // (block_idx, row_n)
{
    uint block_idx = tid.x;
    uint row = tid.y;
    uint blocks_per_row = (K + QK4_1 - 1) / QK4_1;  // Ceiling division

    if (block_idx >= blocks_per_row || row >= N) return;

    device const block_q4_1* block = &input[row * blocks_per_row + block_idx];
    float scale = float(block->scale);
    float min_val = float(block->min);
    uint k_start = block_idx * QK4_1;
    uint k_end = min(k_start + QK4_1, K);

    device float* out_row = output + row * K + k_start;

    for (uint i = 0; i < k_end - k_start; i += 2) {
        uint8_t packed = block->quants[i / 2];
        out_row[i] = float(extract_q4_unsigned(packed, 0)) * scale + min_val;
        if (i + 1 < k_end - k_start) {
            out_row[i + 1] = float(extract_q4_unsigned(packed, 1)) * scale + min_val;
        }
    }
}

// =============================================================================
// Group-Wise INT4 Dequantization Kernel (for ANE fallback GEMM)
// =============================================================================
// Input layout:
// - weights: [N, K/2] packed INT4 (2 values per byte)
// - scales: [N, K/group_size]
// - zeros : [N, K/group_size]
// - output: [N, K] FP32
//
// Formula: output = scale * (q - zero), where q in [-8, 7]
// Grid: (K/2, N, 1), each thread handles one packed byte -> two FP32 outputs.
kernel void dequantize_int4_grouped(
    device const uchar* weights [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* zeros [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint k_packed = tid.x;
    const uint row = tid.y;
    const uint K_packed = K / 2;

    if (row >= N || k_packed >= K_packed || group_size == 0) return;

    const uint groups_per_row = K / group_size;
    if (groups_per_row == 0) return;
    const uint k0 = k_packed * 2;
    const uint byte_idx = row * K_packed + k_packed;
    const uchar packed = weights[byte_idx];

    int q0 = int(packed & 0x0F);
    int q1 = int((packed >> 4) & 0x0F);
    if (q0 > 7) q0 -= 16;
    if (q1 > 7) q1 -= 16;

    const uint g0 = min(k0 / group_size, groups_per_row - 1);
    const uint scale_idx0 = row * groups_per_row + g0;
    const float s0 = scales[scale_idx0];
    const float z0 = zeros[scale_idx0];
    output[row * K + k0] = s0 * (float(q0) - z0);

    const uint k1 = k0 + 1;
    if (k1 < K) {
        const uint g1 = min(k1 / group_size, groups_per_row - 1);
        const uint scale_idx1 = row * groups_per_row + g1;
        const float s1 = scales[scale_idx1];
        const float z1 = zeros[scale_idx1];
        output[row * K + k1] = s1 * (float(q1) - z1);
    }
}
)METAL";

/**
 * @brief FlashAttention decode kernel for single-query attention
 *
 * Optimized for decode phase (seq_q = 1). One threadgroup per (batch, head).
 */
const char* kFlashAttentionShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float NEG_INF = -1e9f;
constant uint MAX_HEAD_DIM = 128;
constant uint DECODE_THREADS = 64;

kernel void flash_attention_decode(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& seq_kv [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tg_size > DECODE_THREADS || head_dim == 0 || head_dim > MAX_HEAD_DIM) return;

    uint batch_idx = tgid.z;
    uint head_idx = tgid.x;
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);

    device const float* Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
    device const float* K_head = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_head = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_ptr = output + (batch_idx * n_heads + head_idx) * head_dim;

    float q[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) {
        q[d] = Q_ptr[d];
    }

    threadgroup float shared_max[DECODE_THREADS];
    threadgroup float shared_sum[DECODE_THREADS];
    threadgroup float shared_output[DECODE_THREADS][MAX_HEAD_DIM];

    float local_max = NEG_INF;
    float local_sum = 0.0f;
    float local_output[MAX_HEAD_DIM] = {0.0f};

    for (uint k = tid; k < seq_kv; k += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot = fma(q[d], K_head[k * head_dim + d], dot);
        }
        float score = dot * scale;

        float old_max = local_max;
        local_max = max(local_max, score);
        float scale_factor = exp(old_max - local_max);
        float exp_score = exp(score - local_max);

        local_sum = local_sum * scale_factor + exp_score;
        for (uint d = 0; d < head_dim; ++d) {
            local_output[d] = local_output[d] * scale_factor +
                              V_head[k * head_dim + d] * exp_score;
        }
    }

    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float m1 = shared_max[tid];
            float m2 = shared_max[tid + stride];
            float new_max = max(m1, m2);
            float s1 = shared_sum[tid] * exp(m1 - new_max);
            float s2 = shared_sum[tid + stride] * exp(m2 - new_max);
            shared_max[tid] = new_max;
            shared_sum[tid] = s1 + s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    float my_scale = exp(local_max - global_max) / global_sum;

    for (uint d = 0; d < head_dim; ++d) {
        shared_output[tid][d] = local_output[d] * my_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            for (uint d = 0; d < head_dim; ++d) {
                shared_output[tid][d] += shared_output[tid + stride][d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        for (uint d = 0; d < head_dim; ++d) {
            O_ptr[d] = shared_output[0][d];
        }
    }
}
)METAL";

/**
 * @brief FlashAttention prefill kernel for multi-query attention (seq_q > 1)
 *
 * Tile-based FlashAttention-2 implementation using Apple Silicon's simdgroup_matrix
 * for hardware-accelerated matrix operations (equivalent to Tensor Cores).
 *
 * Architecture:
 * - Grid: (n_heads, ceil(seq_q / BQ), batch)
 * - Each threadgroup handles BQ=16 query positions against the entire KV sequence
 * - Uses simdgroup_float8x8 for Q×K^T and S×V matrix multiplications
 * - Online softmax with per-tile max tracking for numerical stability
 *
 * Tile Sizes (optimized for Apple Silicon):
 * - BQ = 16: Query block size (2 simdgroups × 8 rows)
 * - BK = 64: Key/Value block size (8 × 8-row tiles)
 * - D = 64-128: Head dimension (must be multiple of 8)
 *
 * Performance Features:
 * - simdgroup_load/store for efficient shared memory access
 * - simdgroup_multiply_accumulate for fused matrix multiply-add
 * - Vectorized float4 loads for global memory bandwidth
 * - Minimal threadgroup barriers
 *
 * Compatibility: Requires Metal 2.3+ (M1/M2/M3 Apple Silicon)
 */
const char* kFlashAttentionPrefillSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Constants and Configuration
// ============================================================================

constant float NEG_INF = -1e9f;
constant uint SIMD_WIDTH = 32;

// Tile sizes optimized for Apple Silicon M1/M2/M3
// BQ: Query block size - 16 queries per threadgroup (2 simdgroups × 8 rows each)
// BK: Key block size - 64 keys per iteration (8 × 8-row matrix tiles)
// These values balance register pressure with compute efficiency
constant uint BQ = 16;                     // Query tile height
constant uint BK = 64;                     // K/V tile width (loop iteration size)
constant uint TILE_SIZE = 8;               // simdgroup_matrix dimension (8x8)
constant uint NUM_SIMDGROUPS = 4;          // 4 simdgroups = 128 threads

// Function constant for head dimension (set at pipeline creation)
constant uint MAX_HEAD_DIM [[function_constant(0)]];

// ============================================================================
// Tile-Based FlashAttention Prefill Kernel
// ============================================================================

kernel void flash_attention_prefill(
    device const float* Q [[buffer(0)]],         // [batch, n_head, seq_q, head_dim]
    device const float* K [[buffer(1)]],         // [batch, n_kv_head, seq_kv, head_dim]
    device const float* V [[buffer(2)]],         // [batch, n_kv_head, seq_kv, head_dim]
    device float* output [[buffer(3)]],          // [batch, n_head, seq_q, head_dim]
    constant float& scale [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& n_heads [[buffer(8)]],
    constant uint& n_kv_heads [[buffer(9)]],
    constant uint& causal [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],      // (head, q_block, batch)
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    // ========================================================================
    // Step 1: Compute indices and bounds
    // ========================================================================
    
    uint head_idx = tgid.x;
    uint q_block_idx = tgid.y;
    uint batch_idx = tgid.z;
    
    // Starting query position for this threadgroup
    uint q_start = q_block_idx * BQ;
    if (q_start >= seq_q) return;
    
    // Number of valid queries in this block (may be < BQ at boundary)
    uint q_count = min(BQ, seq_q - q_start);
    
    // Guard head dimension
    uint safe_head_dim = min(head_dim, MAX_HEAD_DIM);
    uint head_dim_tiles = (safe_head_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    // GQA: Map Q head to KV head
    uint n_rep = n_heads / n_kv_heads;
    uint kv_head_idx = head_idx / n_rep;
    
    // ========================================================================
    // Step 2: Setup pointers to Q, K, V, Output
    // ========================================================================
    
    device const float* Q_base = Q + ((batch_idx * n_heads + head_idx) * seq_q + q_start) * head_dim;
    device const float* K_base = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_base = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_base = output + ((batch_idx * n_heads + head_idx) * seq_q + q_start) * head_dim;
    
    // ========================================================================
    // Step 3: Allocate threadgroup (shared) memory
    // ========================================================================
    // Layout:
    //   sq[BQ][MAX_HEAD_DIM]   - Query tile
    //   sk[BK][MAX_HEAD_DIM]   - Key tile (reused each iteration)
    //   sv[BK][MAX_HEAD_DIM]   - Value tile (overlaps with sk in practice)
    //   ss[BQ][BK]             - Attention scores S = Q × K^T
    //   so[BQ][MAX_HEAD_DIM]   - Output accumulator
    //   sm[BQ], sl[BQ]         - Online softmax state (max, sum)
    // ========================================================================
    
    // Shared memory declarations (threadgroup storage)
    threadgroup float sq[BQ * 128];           // Q tile: [BQ, head_dim] max 128
    threadgroup float sk[BK * 128];           // K tile: [BK, head_dim]
    threadgroup float sv[BK * 128];           // V tile: [BK, head_dim]
    threadgroup float ss[BQ * BK];            // Scores: [BQ, BK]
    threadgroup float so[BQ * 128];           // Output: [BQ, head_dim]
    threadgroup float sm[BQ];                 // Per-query max
    threadgroup float sl[BQ];                 // Per-query sum
    
    // ========================================================================
    // Step 4: Load Q tile into shared memory (cooperative load)
    // ========================================================================
    // Each thread loads multiple elements using float4 for bandwidth
    
    for (uint i = tid; i < q_count * safe_head_dim; i += tg_size) {
        uint q_idx = i / safe_head_dim;
        uint d_idx = i % safe_head_dim;
        sq[q_idx * safe_head_dim + d_idx] = Q_base[q_idx * head_dim + d_idx];
    }
    
    // Initialize output accumulator and softmax state
    for (uint i = tid; i < q_count * safe_head_dim; i += tg_size) {
        so[i] = 0.0f;
    }
    for (uint i = tid; i < q_count; i += tg_size) {
        sm[i] = NEG_INF;
        sl[i] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================================================
    // Step 5: Main loop over K/V blocks (FlashAttention-2 algorithm)
    // ========================================================================
    
    for (uint kv_start = 0; kv_start < seq_kv; kv_start += BK) {
        uint kv_count = min(BK, seq_kv - kv_start);
        
        // --------------------------------------------------------------------
        // Step 5a: Load K tile into shared memory
        // --------------------------------------------------------------------
        for (uint i = tid; i < kv_count * safe_head_dim; i += tg_size) {
            uint k_idx = i / safe_head_dim;
            uint d_idx = i % safe_head_dim;
            sk[k_idx * safe_head_dim + d_idx] = K_base[(kv_start + k_idx) * head_dim + d_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // --------------------------------------------------------------------
        // Step 5b: Compute S = Q × K^T using simdgroup_matrix
        // --------------------------------------------------------------------
        // Each simdgroup computes a portion of the [BQ, BK] score matrix
        // Using 8x8 tiles: we need (BQ/8) × (BK/8) = 2 × 8 = 16 tile computations
        
        // Determine which tiles this simdgroup handles
        uint tiles_per_row = BK / TILE_SIZE;           // 8 tiles across K dimension
        uint tiles_per_col = BQ / TILE_SIZE;           // 2 tiles down Q dimension
        uint total_tiles = tiles_per_row * tiles_per_col;  // 16 tiles total
        
        // Each simdgroup handles ceil(16/4) = 4 tiles
        for (uint tile_idx = simd_group; tile_idx < total_tiles; tile_idx += NUM_SIMDGROUPS) {
            uint tile_row = tile_idx / tiles_per_row;  // Which Q tile row (0-1)
            uint tile_col = tile_idx % tiles_per_row;  // Which K tile col (0-7)
            
            uint q_tile_start = tile_row * TILE_SIZE;
            uint k_tile_start = tile_col * TILE_SIZE;
            
            // Skip if this tile is fully masked (causal attention)
            if (causal) {
                // For causal: check if any query in this tile can attend to any key in this tile
                uint q_pos_min = q_start + q_tile_start;
                uint q_pos_max = q_pos_min + TILE_SIZE - 1;
                uint k_pos_min = kv_start + k_tile_start;
                // The furthest a query at q_pos_max can attend to is q_pos_max + (seq_kv - seq_q)
                uint causal_limit_max = q_pos_max + (seq_kv - seq_q);
                
                if (k_pos_min > causal_limit_max) {
                    // This entire tile is masked, write -INF to scores
                    for (uint lane = simd_lane; lane < TILE_SIZE * TILE_SIZE; lane += SIMD_WIDTH) {
                        uint local_q = lane / TILE_SIZE;
                        uint local_k = lane % TILE_SIZE;
                        if (q_tile_start + local_q < q_count && k_tile_start + local_k < kv_count) {
                            ss[(q_tile_start + local_q) * BK + k_tile_start + local_k] = NEG_INF;
                        }
                    }
                    continue;
                }
            }
            
            // Compute 8x8 matrix multiply: S_tile = Q_tile × K_tile^T
            // Accumulate over head_dim in chunks of 8
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            
            for (uint d_tile = 0; d_tile < head_dim_tiles; ++d_tile) {
                uint d_start = d_tile * TILE_SIZE;
                uint d_end = min(d_start + TILE_SIZE, safe_head_dim);
                
                // Load Q tile [8,8] from sq (row-major, stride = head_dim)
                simdgroup_float8x8 mq;
                simdgroup_load(mq, sq + q_tile_start * safe_head_dim + d_start, 
                               safe_head_dim, ulong2(0, 0), false);
                
                // Load K tile [8,8] from sk, transposed for K^T
                simdgroup_float8x8 mk;
                simdgroup_load(mk, sk + k_tile_start * safe_head_dim + d_start, 
                               safe_head_dim, ulong2(0, 0), true);  // transpose = true
                
                // Accumulate: acc += mq × mk^T
                simdgroup_multiply_accumulate(acc, mq, mk, acc);
            }
            
            // Store result to shared memory scores [BQ, BK]
            // Apply scale factor here
            // Note: simdgroup matrices don't support scalar multiply directly,
            // so we store and scale in shared memory
            simdgroup_store(acc, ss + q_tile_start * BK + k_tile_start, BK, ulong2(0, 0), false);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // --------------------------------------------------------------------
        // Step 5c: Apply scale and causal mask, compute online softmax
        // --------------------------------------------------------------------
        
        // Each thread handles a subset of queries
        for (uint q_idx = tid / SIMD_WIDTH; q_idx < q_count; q_idx += NUM_SIMDGROUPS) {
            uint global_q_pos = q_start + q_idx;
            uint causal_limit = causal ? (global_q_pos + (seq_kv - seq_q) + 1) : seq_kv;
            
            float row_max = sm[q_idx];
            float row_sum = sl[q_idx];
            
            // Process this query's scores across all K positions in this block
            // Handle full BK width to ensure stale scores from partial blocks are masked
            for (uint k_idx = simd_lane; k_idx < BK; k_idx += SIMD_WIDTH) {
                uint global_k_pos = kv_start + k_idx;
                
                // Determine if this key position is valid (within sequence and causal mask)
                bool is_valid_k = (k_idx < kv_count);
                bool mask = !is_valid_k || (global_k_pos >= causal_limit);
                
                float score = ss[q_idx * BK + k_idx];
                
                if (mask) {
                    score = NEG_INF;
                } else {
                    score *= scale;
                }
                
                ss[q_idx * BK + k_idx] = score;
                
                // Track max for this row
                row_max = max(row_max, score);
            }
            
            // Reduce max across simdgroup
            row_max = simd_max(row_max);
            
            // Compute exp(score - max) and sum
            float prev_max = sm[q_idx];
            float rescale = exp(prev_max - row_max);
            float new_sum = row_sum * rescale;
            
            for (uint k_idx = simd_lane; k_idx < BK; k_idx += SIMD_WIDTH) {
                float score = ss[q_idx * BK + k_idx];
                float exp_score = exp(score - row_max);
                ss[q_idx * BK + k_idx] = exp_score;
                new_sum += exp_score;
            }
            
            new_sum = simd_sum(new_sum);
            
            // Rescale previous output accumulator
            for (uint d = simd_lane; d < safe_head_dim; d += SIMD_WIDTH) {
                so[q_idx * safe_head_dim + d] *= rescale;
            }
            
            // Update softmax state (first lane writes)
            if (simd_lane == 0) {
                sm[q_idx] = row_max;
                sl[q_idx] = new_sum;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // --------------------------------------------------------------------
        // Step 5d: Load V tile and compute O += S × V using simdgroup_matrix
        // --------------------------------------------------------------------
        
        // Load V tile into shared memory
        // Zero-pad partial V tiles to avoid accumulating stale data from previous blocks
        uint v_tiles = (kv_count + TILE_SIZE - 1) / TILE_SIZE;
        uint v_end_offset = v_tiles * TILE_SIZE * safe_head_dim;
        uint v_valid_offset = kv_count * safe_head_dim;
        
        for (uint i = tid; i < v_end_offset; i += tg_size) {
            if (i < v_valid_offset) {
                uint v_idx = i / safe_head_dim;
                uint d_idx = i % safe_head_dim;
                sv[i] = V_base[(kv_start + v_idx) * head_dim + d_idx];
            } else {
                sv[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute O_tile += S_tile × V_tile using simdgroup_matrix
        // S: [BQ, BK], V: [BK, D] -> O: [BQ, D]
        // Process in 8x8 tiles
        
        uint out_tiles_row = BQ / TILE_SIZE;           // 2 tiles down Q
        uint out_tiles_col = (safe_head_dim + TILE_SIZE - 1) / TILE_SIZE;  // tiles across D
        uint out_total_tiles = out_tiles_row * out_tiles_col;
        
        for (uint tile_idx = simd_group; tile_idx < out_total_tiles; tile_idx += NUM_SIMDGROUPS) {
            uint tile_row = tile_idx / out_tiles_col;  // Q tile row
            uint tile_col = tile_idx % out_tiles_col;  // D tile col
            
            uint q_tile_start = tile_row * TILE_SIZE;
            uint d_tile_start = tile_col * TILE_SIZE;
            
            // Load current O tile
            simdgroup_float8x8 mo;
            simdgroup_load(mo, so + q_tile_start * safe_head_dim + d_tile_start,
                          safe_head_dim, ulong2(0, 0), false);
            
            // Accumulate S × V for each K tile
            // CRITICAL: Only iterate over tiles containing valid keys for this block.
            // When seq_kv is not a multiple of BK (64), the final block has fewer
            // than 64 keys, so we must use ceil(kv_count / TILE_SIZE) to avoid
            // accumulating stale ss/sv data from previous iterations.
            uint k_tiles = (kv_count + TILE_SIZE - 1) / TILE_SIZE;  // ceil division
            for (uint k_tile = 0; k_tile < k_tiles; ++k_tile) {
                uint k_tile_start = k_tile * TILE_SIZE;
                
                // Load S tile [8, 8] from ss (attention weights)
                simdgroup_float8x8 ms;
                simdgroup_load(ms, ss + q_tile_start * BK + k_tile_start,
                              BK, ulong2(0, 0), false);
                
                // Load V tile [8, 8] from sv
                simdgroup_float8x8 mv;
                simdgroup_load(mv, sv + k_tile_start * safe_head_dim + d_tile_start,
                              safe_head_dim, ulong2(0, 0), false);
                
                // Accumulate: mo += ms × mv
                simdgroup_multiply_accumulate(mo, ms, mv, mo);
            }
            
            // Store back to shared memory
            simdgroup_store(mo, so + q_tile_start * safe_head_dim + d_tile_start,
                           safe_head_dim, ulong2(0, 0), false);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // ========================================================================
    // Step 6: Finalize output (divide by softmax sum and write to global)
    // ========================================================================
    
    for (uint i = tid; i < q_count * safe_head_dim; i += tg_size) {
        uint q_idx = i / safe_head_dim;
        uint d_idx = i % safe_head_dim;
        
        float sum = sl[q_idx];
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        
        O_base[q_idx * head_dim + d_idx] = so[q_idx * safe_head_dim + d_idx] * inv_sum;
    }
}
)METAL";

/**
 * @brief Fused Q/K/V projection kernel for decode phase
 *
 * Computes Q, K, V projections in a single dispatch:
 *   Q = input @ Wq^T, K = input @ Wk^T, V = input @ Wv^T
 *
 * For decode phase (batch=1), this reduces kernel launch overhead by 3x
 * compared to three separate MatMulTransB calls.
 *
 * Grid layout: Each threadgroup handles one output row from Q, K, or V.
 * - Rows 0..hidden_q-1: Q output
 * - Rows hidden_q..hidden_q+hidden_kv-1: K output
 * - Rows hidden_q+hidden_kv..total-1: V output
 */
const char* kFusedQKVShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;

kernel void fused_qkv_gemv(
    device const float* input [[buffer(0)]],     // [K]
    device const float* wq [[buffer(1)]],        // [hidden_q, K]
    device const float* wk [[buffer(2)]],        // [hidden_kv, K]
    device const float* wv [[buffer(3)]],        // [hidden_kv, K]
    device float* q_out [[buffer(4)]],           // [hidden_q]
    device float* k_out [[buffer(5)]],           // [hidden_kv]
    device float* v_out [[buffer(6)]],           // [hidden_kv]
    constant uint& K [[buffer(7)]],              // Input dimension
    constant uint& hidden_q [[buffer(8)]],       // Q output dimension
    constant uint& hidden_kv [[buffer(9)]],      // K/V output dimension
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Determine which output tensor and row this threadgroup handles
    // First hidden_q rows -> Q, next hidden_kv -> K, last hidden_kv -> V
    uint row = tgid;
    device const float* weight;
    device float* output;

    if (row < hidden_q) {
        // Q projection
        weight = wq + row * K;
        output = q_out + row;
    } else if (row < hidden_q + hidden_kv) {
        // K projection
        uint kv_row = row - hidden_q;
        weight = wk + kv_row * K;
        output = k_out + kv_row;
    } else {
        // V projection
        uint kv_row = row - hidden_q - hidden_kv;
        weight = wv + kv_row * K;
        output = v_out + kv_row;
    }

    threadgroup float shared_sum[THREADGROUP_SIZE];

    // Compute dot product: weight_row @ input
    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight[k], input[k], sum);
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *output = shared_sum[0];
    }
}
)METAL";

}  // anonymous namespace

// ============================================================================
// Static Methods
// ============================================================================

bool MetalBackend::IsAvailable() {
    @autoreleasepool {
        // Check for Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return false;
        }

        // Check for required features
        // All Apple Silicon supports Metal 2.0+ which has everything we need
        bool supported = [device supportsFamily:MTLGPUFamilyApple7] ||  // M1+
                         [device supportsFamily:MTLGPUFamilyMac2];      // Intel Mac

        return supported;
    }
}

const char* MetalBackend::GetDeviceName() {
    static char deviceName[128] = {0};

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return nullptr;
        }

        NSString* name = [device name];
        if (name) {
            strncpy(deviceName, [name UTF8String], sizeof(deviceName) - 1);
            return deviceName;
        }
    }

    return nullptr;
}

AppleSiliconChipInfo MetalBackend::GetChipInfo() {
    AppleSiliconChipInfo info = {};

    apple::ChipGeneration gen = apple::DetectChipGeneration();
    info.chip_name = apple::ChipGenerationName(gen);
    info.chip_generation = static_cast<int>(gen);
    info.performance_cores = apple::GetPerformanceCoreCount();
    info.efficiency_cores = apple::GetEfficiencyCoreCount();
    info.neural_engine_tops = apple::GetNeuralEngineTOPS();

    apple::MemoryInfo memInfo = apple::GetMemoryInfo();
    info.unified_memory_bytes = memInfo.total_bytes;
    info.memory_bandwidth_gbps = memInfo.bandwidth_gbps;

    // Check GPU features
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            // Count GPU cores (approximation based on registry name)
            info.gpu_cores = 8;  // Default for M1
            if (apple::IsM1Family(gen)) {
                if (gen == apple::ChipGeneration::M1_Pro)
                    info.gpu_cores = 16;
                else if (gen == apple::ChipGeneration::M1_Max)
                    info.gpu_cores = 32;
                else if (gen == apple::ChipGeneration::M1_Ultra)
                    info.gpu_cores = 64;
            } else if (apple::IsM2Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M2_Pro)
                    info.gpu_cores = 19;
                else if (gen == apple::ChipGeneration::M2_Max)
                    info.gpu_cores = 38;
                else if (gen == apple::ChipGeneration::M2_Ultra)
                    info.gpu_cores = 76;
            } else if (apple::IsM3Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M3_Pro)
                    info.gpu_cores = 18;
                else if (gen == apple::ChipGeneration::M3_Max)
                    info.gpu_cores = 40;
            } else if (apple::IsM4Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M4_Pro)
                    info.gpu_cores = 20;
                else if (gen == apple::ChipGeneration::M4_Max)
                    info.gpu_cores = 40;
            }

            // Check feature support
            info.supports_simd_group_reduction = [device supportsFamily:MTLGPUFamilyApple7];
            info.supports_bfloat16 = [device supportsFamily:MTLGPUFamilyApple9];     // M3+
            info.supports_ray_tracing = [device supportsFamily:MTLGPUFamilyApple9];  // M3+
        }
    }

    return info;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        // Get Metal device
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device == nil) {
            throw std::runtime_error("Failed to create Metal device");
        }

        // Configure wrapped buffer cache size
        if (const char* env = std::getenv("DENSECORE_METAL_WRAPPED_BUFFER_CACHE")) {
            char* end = nullptr;
            long value = std::strtol(env, &end, 10);
            if (end != env && value >= 0) {
                impl_->maxWrappedBuffers = static_cast<size_t>(value);
            }
        }

        // Create command queue
        impl_->commandQueue = [impl_->device newCommandQueue];
        if (impl_->commandQueue == nil) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        impl_->supportsSimdgroupMatrix = [impl_->device supportsFamily:MTLGPUFamilyApple7];
        if (const char* env = std::getenv("DENSECORE_METAL_INT4_SIMDGROUP_GEMM")) {
            impl_->enableSimdgroupInt4Gemm =
                std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
        }

        // Try to load pre-compiled shader library
        NSError* error = nil;
        NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"densecore"
                                                                ofType:@"metallib"];
        if (libraryPath) {
            NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
            impl_->shaderLibrary = [impl_->device newLibraryWithURL:libraryURL error:&error];
        }

        // Fall back to runtime compilation
        MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.fastMathEnabled = YES;

        if (impl_->shaderLibrary == nil) {
            NSString* source = [NSString stringWithUTF8String:kMetalShaderSource];

            impl_->shaderLibrary = [impl_->device newLibraryWithSource:source
                                                               options:compileOptions
                                                                 error:&error];
            if (impl_->shaderLibrary == nil) {
                throw std::runtime_error("Failed to compile Metal shaders: " +
                                         std::string([[error localizedDescription] UTF8String]));
            }
        }

        // Create pipeline states for custom kernels
        id<MTLFunction> gemvFunction = [impl_->shaderLibrary newFunctionWithName:@"gemv_f32"];
        if (gemvFunction) {
            impl_->gemvPipeline = [impl_->device newComputePipelineStateWithFunction:gemvFunction
                                                                               error:&error];
        }
        id<MTLFunction> gemvBatchedFunction =
            [impl_->shaderLibrary newFunctionWithName:@"gemv_f32_batched"];
        if (gemvBatchedFunction) {
            impl_->gemvBatchedPipeline =
                [impl_->device newComputePipelineStateWithFunction:gemvBatchedFunction
                                                             error:&error];
        }

        id<MTLFunction> softmaxFunction = [impl_->shaderLibrary newFunctionWithName:@"softmax_f32"];
        if (softmaxFunction) {
            impl_->softmaxPipeline =
                [impl_->device newComputePipelineStateWithFunction:softmaxFunction error:&error];
        }

        id<MTLFunction> rmsNormFunction =
            [impl_->shaderLibrary newFunctionWithName:@"rms_norm_f32"];
        if (rmsNormFunction) {
            impl_->rmsNormPipeline =
                [impl_->device newComputePipelineStateWithFunction:rmsNormFunction error:&error];
        }
        id<MTLFunction> addRmsNormFunction =
            [impl_->shaderLibrary newFunctionWithName:@"add_rms_norm_f32"];
        if (addRmsNormFunction) {
            impl_->addRmsNormPipeline =
                [impl_->device newComputePipelineStateWithFunction:addRmsNormFunction error:&error];
        }

        // RoPE kernel
        id<MTLFunction> ropeFunction = [impl_->shaderLibrary newFunctionWithName:@"rope_f32"];
        if (ropeFunction) {
            impl_->ropePipeline = [impl_->device newComputePipelineStateWithFunction:ropeFunction
                                                                               error:&error];
            if (impl_->ropePipeline) {
                std::cout << "[MetalBackend] RoPE kernel compiled" << std::endl;
            }
        }

        // FlashAttention decode kernel: try external metallib first, then runtime
        // compile
        id<MTLLibrary> externalLibrary = nil;
        NSString* metalLibPath = [[NSBundle mainBundle] pathForResource:@"densecore"
                                                                 ofType:@"metallib"];
        if (metalLibPath) {
            NSURL* metalLibURL = [NSURL fileURLWithPath:metalLibPath];
            externalLibrary = [impl_->device newLibraryWithURL:metalLibURL error:&error];
        }

        uint32_t maxHeadDim = 128;  // Supports head_dim up to 128

        if (externalLibrary) {
            id<MTLFunction> flashAttnDecodeFunction =
                [externalLibrary newFunctionWithName:@"flash_attention_decode"];
            if (flashAttnDecodeFunction) {
                impl_->flashAttentionDecodePipeline =
                    [impl_->device newComputePipelineStateWithFunction:flashAttnDecodeFunction
                                                                 error:&error];
                if (impl_->flashAttentionDecodePipeline) {
                    std::cout << "[MetalBackend] FlashAttention decode kernel loaded "
                                 "(MAX_HEAD_DIM="
                              << maxHeadDim << ")" << std::endl;
                }
            }
        }

        // Fallback: compile FlashAttention from embedded source if metallib failed
        if (!impl_->flashAttentionDecodePipeline) {
            std::cout << "[MetalBackend] FlashAttention metallib not found or failed. "
                         "Falling back to runtime compilation."
                      << std::endl;
            NSString* flashAttnSource = [NSString stringWithUTF8String:kFlashAttentionShaderSource];
            id<MTLLibrary> flashAttnLib = [impl_->device newLibraryWithSource:flashAttnSource
                                                                      options:compileOptions
                                                                        error:&error];
            if (flashAttnLib) {
                id<MTLFunction> flashAttnDecodeFunction =
                    [flashAttnLib newFunctionWithName:@"flash_attention_decode"];
                if (flashAttnDecodeFunction) {
                    impl_->flashAttentionDecodePipeline =
                        [impl_->device newComputePipelineStateWithFunction:flashAttnDecodeFunction
                                                                     error:&error];
                    if (impl_->flashAttentionDecodePipeline) {
                        std::cout << "[MetalBackend] FlashAttention decode kernel compiled "
                                     "(MAX_HEAD_DIM="
                                  << maxHeadDim << ")" << std::endl;
                    } else {
                        std::cerr << "[MetalBackend] Warning: Failed to create FlashAttention "
                                     "decode pipeline: "
                                  << [[error localizedDescription] UTF8String] << std::endl;
                    }
                } else {
                    std::cerr << "[MetalBackend] Warning: Failed to create FlashAttention "
                                 "decode function: "
                              << [[error localizedDescription] UTF8String] << std::endl;
                }
            } else {
                std::cerr << "[MetalBackend] Warning: Failed to compile FlashAttention "
                             "shader: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }

        // Compile FlashAttention prefill kernel from embedded source
        // Note: Uses function constants for compile-time configuration
        {
            MTLFunctionConstantValues* constantValues = [[MTLFunctionConstantValues alloc] init];
            [constantValues setConstantValue:&maxHeadDim
                                        type:MTLDataTypeUInt
                                     atIndex:0];  // function_constant(0)

            NSString* flashAttnPrefillSource =
                [NSString stringWithUTF8String:kFlashAttentionPrefillSource];
            id<MTLLibrary> flashAttnPrefillLib =
                [impl_->device newLibraryWithSource:flashAttnPrefillSource
                                            options:compileOptions
                                              error:&error];
            if (flashAttnPrefillLib) {
                // Use shared constantValues (MAX_HEAD_DIM=128)

                id<MTLFunction> flashAttnPrefillFunction =
                    [flashAttnPrefillLib newFunctionWithName:@"flash_attention_prefill"
                                              constantValues:constantValues
                                                       error:&error];
                if (flashAttnPrefillFunction) {
                    impl_->flashAttentionPrefillPipeline =
                        [impl_->device newComputePipelineStateWithFunction:flashAttnPrefillFunction
                                                                     error:&error];
                    if (impl_->flashAttentionPrefillPipeline) {
                        std::cout << "[MetalBackend] FlashAttention prefill kernel compiled "
                                     "(MAX_HEAD_DIM="
                                  << maxHeadDim << ")" << std::endl;
                    } else {
                        std::cerr << "[MetalBackend] Warning: Failed to create FlashAttention "
                                     "prefill pipeline: "
                                  << [[error localizedDescription] UTF8String] << std::endl;
                    }
                } else {
                    std::cerr << "[MetalBackend] Warning: Failed to create FlashAttention "
                                 "prefill function with constants: "
                              << [[error localizedDescription] UTF8String] << std::endl;
                }
            } else {
                std::cerr << "[MetalBackend] Warning: Failed to compile FlashAttention "
                             "prefill shader: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }

        // Compile quantized GEMV kernels (Q4_0, Q4_1, Q8_0)
        NSString* quantizedGemvSource = [NSString stringWithUTF8String:kQuantizedGemvShaderSource];
        id<MTLLibrary> quantizedGemvLib = [impl_->device newLibraryWithSource:quantizedGemvSource
                                                                      options:compileOptions
                                                                        error:&error];
        if (quantizedGemvLib) {
            id<MTLFunction> gemvQ4_0Function = [quantizedGemvLib newFunctionWithName:@"gemv_q4_0"];
            if (gemvQ4_0Function) {
                impl_->gemvQ4_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ4_0Function
                                                                 error:&error];
            }

            id<MTLFunction> gemvQ4_1Function = [quantizedGemvLib newFunctionWithName:@"gemv_q4_1"];
            if (gemvQ4_1Function) {
                impl_->gemvQ4_1Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ4_1Function
                                                                 error:&error];
            }

            id<MTLFunction> gemvQ8_0Function = [quantizedGemvLib newFunctionWithName:@"gemv_q8_0"];
            if (gemvQ8_0Function) {
                impl_->gemvQ8_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ8_0Function
                                                                 error:&error];
            }

            id<MTLFunction> gemmQ4_0Function =
                [quantizedGemvLib newFunctionWithName:@"gemm_q4_0_fused"];
            if (gemmQ4_0Function) {
                impl_->gemmQ4_0FusedPipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemmQ4_0Function
                                                                 error:&error];
            }

            id<MTLFunction> gemmQ4_1Function =
                [quantizedGemvLib newFunctionWithName:@"gemm_q4_1_fused"];
            if (gemmQ4_1Function) {
                impl_->gemmQ4_1FusedPipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemmQ4_1Function
                                                                 error:&error];
            }

            id<MTLFunction> gemmInt4GroupedFunction =
                [quantizedGemvLib newFunctionWithName:@"gemm_int4_grouped_fused"];
            if (gemmInt4GroupedFunction) {
                impl_->gemmInt4GroupedFusedPipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemmInt4GroupedFunction
                                                                 error:&error];
            }

            if (impl_->supportsSimdgroupMatrix && impl_->enableSimdgroupInt4Gemm) {
                id<MTLFunction> gemmQ4_0SimdgroupFunction =
                    [quantizedGemvLib newFunctionWithName:@"gemm_q4_0_simdgroup_fused"];
                if (gemmQ4_0SimdgroupFunction) {
                    impl_->gemmQ4_0SimdgroupPipeline =
                        [impl_->device newComputePipelineStateWithFunction:gemmQ4_0SimdgroupFunction
                                                                     error:&error];
                }

                id<MTLFunction> gemmQ4_1SimdgroupFunction =
                    [quantizedGemvLib newFunctionWithName:@"gemm_q4_1_simdgroup_fused"];
                if (gemmQ4_1SimdgroupFunction) {
                    impl_->gemmQ4_1SimdgroupPipeline =
                        [impl_->device newComputePipelineStateWithFunction:gemmQ4_1SimdgroupFunction
                                                                     error:&error];
                }

                id<MTLFunction> gemmInt4GroupedSimdgroupFunction =
                    [quantizedGemvLib newFunctionWithName:@"gemm_int4_grouped_simdgroup_fused"];
                if (gemmInt4GroupedSimdgroupFunction) {
                    impl_->gemmInt4GroupedSimdgroupPipeline = [impl_->device
                        newComputePipelineStateWithFunction:gemmInt4GroupedSimdgroupFunction
                                                      error:&error];
                }
            }

            if (impl_->gemvQ4_0Pipeline || impl_->gemvQ4_1Pipeline || impl_->gemvQ8_0Pipeline) {
                std::cout << "[MetalBackend] Quantized GEMV kernels compiled: "
                          << (impl_->gemvQ4_0Pipeline ? "Q4_0 " : "")
                          << (impl_->gemvQ4_1Pipeline ? "Q4_1 " : "")
                          << (impl_->gemvQ8_0Pipeline ? "Q8_0 " : "") << std::endl;
            }

            if (impl_->gemmQ4_0FusedPipeline || impl_->gemmQ4_1FusedPipeline ||
                impl_->gemmInt4GroupedFusedPipeline) {
                std::cout << "[MetalBackend] Fused INT4 GEMM kernels compiled: "
                          << (impl_->gemmQ4_0FusedPipeline ? "Q4_0 " : "")
                          << (impl_->gemmQ4_1FusedPipeline ? "Q4_1 " : "")
                          << (impl_->gemmInt4GroupedFusedPipeline ? "INT4_GROUPED " : "")
                          << std::endl;
            }

            if (impl_->gemmQ4_0SimdgroupPipeline || impl_->gemmQ4_1SimdgroupPipeline ||
                impl_->gemmInt4GroupedSimdgroupPipeline) {
                std::cout << "[MetalBackend] simdgroup_matrix INT4 GEMM kernels compiled: "
                          << (impl_->gemmQ4_0SimdgroupPipeline ? "Q4_0 " : "")
                          << (impl_->gemmQ4_1SimdgroupPipeline ? "Q4_1 " : "")
                          << (impl_->gemmInt4GroupedSimdgroupPipeline ? "INT4_GROUPED " : "")
                          << std::endl;
            }

            // Compile dequantization kernel for M>1 GEMM path
            id<MTLFunction> dequantQ4_0Function =
                [quantizedGemvLib newFunctionWithName:@"dequantize_q4_0"];
            if (dequantQ4_0Function) {
                impl_->dequantizeQ4_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:dequantQ4_0Function
                                                                 error:&error];
                if (impl_->dequantizeQ4_0Pipeline) {
                    std::cout << "[MetalBackend] Dequantize Q4_0 kernel compiled" << std::endl;
                }
            }

            id<MTLFunction> dequantQ4_1Function =
                [quantizedGemvLib newFunctionWithName:@"dequantize_q4_1"];
            if (dequantQ4_1Function) {
                impl_->dequantizeQ4_1Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:dequantQ4_1Function
                                                                 error:&error];
                if (impl_->dequantizeQ4_1Pipeline) {
                    std::cout << "[MetalBackend] Dequantize Q4_1 kernel compiled" << std::endl;
                }
            }

            // Compile group-wise INT4 dequantization kernel (for ANE GEMM fallback)
            id<MTLFunction> dequantInt4GroupedFunction =
                [quantizedGemvLib newFunctionWithName:@"dequantize_int4_grouped"];
            if (dequantInt4GroupedFunction) {
                impl_->dequantizeInt4GroupedPipeline =
                    [impl_->device newComputePipelineStateWithFunction:dequantInt4GroupedFunction
                                                                 error:&error];
                if (impl_->dequantizeInt4GroupedPipeline) {
                    std::cout << "[MetalBackend] Dequantize INT4 Grouped kernel compiled"
                              << std::endl;
                }
            }
        } else {
            std::cerr << "[MetalBackend] Warning: Failed to compile quantized GEMV "
                         "shaders: "
                      << [[error localizedDescription] UTF8String] << std::endl;
        }

        // Compile Fused QKV kernel
        NSString* fusedQKVSource = [NSString stringWithUTF8String:kFusedQKVShaderSource];
        id<MTLLibrary> fusedQKVLib = [impl_->device newLibraryWithSource:fusedQKVSource
                                                                 options:compileOptions
                                                                   error:&error];
        if (fusedQKVLib) {
            id<MTLFunction> fusedQKVFunction = [fusedQKVLib newFunctionWithName:@"fused_qkv_gemv"];
            if (fusedQKVFunction) {
                impl_->fusedQKVPipeline =
                    [impl_->device newComputePipelineStateWithFunction:fusedQKVFunction
                                                                 error:&error];
                if (impl_->fusedQKVPipeline) {
                    std::cout << "[MetalBackend] Fused QKV kernel compiled" << std::endl;
                }
            }
        } else {
            std::cerr << "[MetalBackend] Warning: Failed to compile Fused QKV "
                         "shader: "
                      << [[error localizedDescription] UTF8String] << std::endl;
        }

        // Initialize GGML Metal backend
        impl_->ggmlMetalBackend = ggml_backend_metal_init();
        if (impl_->ggmlMetalBackend == nullptr) {
            std::cerr << "[MetalBackend] Warning: Failed to initialize GGML Metal backend, "
                      << "falling back to custom kernels only" << std::endl;
        }

        // Cache chip info
        impl_->chipInfo = GetChipInfo();

        // Set backend name
        snprintf(name_, sizeof(name_), "Apple-Metal-%s", impl_->chipInfo.chip_name);

        std::cout << "[MetalBackend] Initialized: " << name_ << std::endl;
        std::cout << "  GPU Cores: " << impl_->chipInfo.gpu_cores << std::endl;
        std::cout << "  Unified Memory: " << (impl_->chipInfo.unified_memory_bytes >> 30) << " GB"
                  << std::endl;
        std::cout << "  Memory Bandwidth: " << impl_->chipInfo.memory_bandwidth_gbps << " GB/s"
                  << std::endl;
    }
}

MetalBackend::~MetalBackend() {
    // Wait for all GPU work to complete
    Synchronize();

    // impl_ destructor handles cleanup via RAII
}

// ============================================================================
// ComputeBackend Interface - Identification
// ============================================================================

const char* MetalBackend::Name() const {
    return name_;
}

BackendCapabilityManifest MetalBackend::GetCapabilityManifest() const {
    BackendCapabilityManifest manifest;

    // Native ops: operations with custom Metal shaders or MPS implementations.
    // All these have actual GPU kernel paths in this backend.
    manifest.native_ops = {
        OpType::MatMul,              // MPS GEMM + custom GEMV kernel
        OpType::MatMulTransB,        // MPS GEMM (transposeRight)
        OpType::GemmInt4,            // Custom quantized GEMV (Q4_0/Q4_1) + GPU dequant→MPS GEMM
        OpType::Softmax,             // Custom Metal softmax kernel
        OpType::RMSNorm,             // Custom Metal RMSNorm kernel
        OpType::AddRMSNorm,          // Custom Metal fused add+RMSNorm kernel
        OpType::RoPE,                // Custom Metal RoPE kernel
        OpType::FlashAttention,      // Custom Metal FlashAttention decode + prefill kernels
        OpType::FusedQKVProjection,  // Custom Metal fused QKV GEMV kernel
    };

    // Fallback ops: operations that this backend can serve via CPU (Accelerate.framework)
    // but does not have dedicated GPU kernels for.
    manifest.fallback_ops = {
        OpType::Embedding,  // Table lookup — memory bound, CPU is fine
        OpType::LayerNorm,  // Can use CPU Accelerate; Metal kernel TODO
        OpType::SiLU,       // Element-wise — CPU Accelerate is sufficient
        OpType::GELU,       // Element-wise — CPU Accelerate is sufficient
    };

    manifest.allow_cpu_fallback = true;
    manifest.declared_complete = false;
    return manifest;
}

// ============================================================================
// ComputeBackend Interface - Memory Management
// ============================================================================

void* MetalBackend::AllocateUnified(size_t size_bytes, size_t alignment) {
    // Explicitly route to AllocateDevice which creates MTLStorageModeShared buffers.
    // This documents that Metal's AllocateDevice already provides UMA zero-copy.
    if (alignment == 0) {
        alignment = 64;  // Metal optimal alignment
    }
    return AllocateDevice(size_bytes, alignment);
}

void* MetalBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
    if (size_bytes == 0) {
        return nullptr;
    }

    @autoreleasepool {
        // Ensure minimum alignment for Metal
        alignment = std::max(alignment, static_cast<size_t>(64));

        // Round up size to alignment
        size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;

        // Create Metal buffer with shared storage mode (UMA zero-copy)
        id<MTLBuffer> buffer = [impl_->device newBufferWithLength:aligned_size
                                                          options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            std::cerr << "[MetalBackend] Failed to allocate " << aligned_size << " bytes"
                      << std::endl;
            return nullptr;
        }

        // Track memory usage
        size_t current = impl_->currentMemoryUsage.fetch_add(aligned_size) + aligned_size;
        size_t peak = impl_->peakMemoryUsage.load();
        while (current > peak && !impl_->peakMemoryUsage.compare_exchange_weak(peak, current)) {}

        // Return the buffer's contents pointer
        // Note: The buffer itself is retained by ARC, but we need to track it
        // for deallocation. We use the contents pointer as the key.
        void* ptr = [buffer contents];

        // Retain buffer to prevent ARC from releasing
        CFRetain((__bridge CFTypeRef)buffer);

        // Register pointer -> buffer mapping for deallocation
        {
            std::lock_guard<std::mutex> lock(impl_->bufferRegistryMutex);
            impl_->bufferRegistry[ptr] = buffer;
        }

        return ptr;
    }
}

void MetalBackend::FreeDevice(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = nil;
        size_t bufferLength = 0;

        // Look up the buffer associated with this pointer
        {
            std::lock_guard<std::mutex> lock(impl_->bufferRegistryMutex);
            auto it = impl_->bufferRegistry.find(ptr);
            if (it != impl_->bufferRegistry.end()) {
                buffer = it->second;
                bufferLength = [buffer length];
                impl_->bufferRegistry.erase(it);
            }
        }

        if (buffer) {
            // Update memory tracking
            impl_->currentMemoryUsage.fetch_sub(bufferLength);

            // Release the CFRetain we did in AllocateDevice
            CFRelease((__bridge CFTypeRef)buffer);
        } else {
            std::cerr << "[MetalBackend] Warning: FreeDevice called with untracked "
                         "pointer: "
                      << ptr << std::endl;
        }
    }
}

void MetalBackend::CopyToDevice(void* dst, const void* src, size_t size_bytes) {
    // On Apple Silicon with UMA, this is just a memcpy
    if (dst && src && size_bytes > 0) {
        std::memcpy(dst, src, size_bytes);
    }
}

void MetalBackend::CopyFromDevice(void* dst, const void* src, size_t size_bytes) {
    // On Apple Silicon with UMA, this is just a memcpy
    if (dst && src && size_bytes > 0) {
        std::memcpy(dst, src, size_bytes);
    }
}

// ============================================================================
// ComputeBackend Interface - Matrix Operations
// ============================================================================

void MetalBackend::MatMul(const Tensor& A, const Tensor& B, Tensor* C) {
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[1]);
    const int small_batch_max = MetalSmallBatchGemvMax();

    @autoreleasepool {
        if (M == 1 && impl_->gemvPipeline) {
            // GEMV path: Use custom kernel for decode phase
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferB =
                impl_->GetOrWrapBuffer(const_cast<void*>(B.data), B.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());
            if (!bufferA || !bufferB || !bufferC) {
                std::cerr
                    << "[MetalBackend] MatMul GEMV: Buffer creation failed, falling back to CPU"
                    << std::endl;
                apple::GemmAccelerate(C->DataAs<float>(), A.DataAs<float>(), B.DataAs<float>(), M,
                                      N, K);
                return;
            }

            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:impl_->gemvPipeline];
            [encoder setBuffer:bufferA offset:0 atIndex:0];
            [encoder setBuffer:bufferB offset:0 atIndex:1];
            [encoder setBuffer:bufferC offset:0 atIndex:2];
            uint M_u = static_cast<uint>(N);  // GEMV output dimension
            uint K_u = static_cast<uint>(K);
            [encoder setBytes:&M_u length:sizeof(uint) atIndex:3];
            [encoder setBytes:&K_u length:sizeof(uint) atIndex:4];

            // Launch one threadgroup per output element
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
        } else if (M > 1 && M <= small_batch_max &&
                   (impl_->gemvBatchedPipeline || impl_->gemvPipeline)) {
            // Small-batch decode path: avoid generic MPS GEMM dispatch overhead.
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferB =
                impl_->GetOrWrapBuffer(const_cast<void*>(B.data), B.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());
            if (!bufferA || !bufferB || !bufferC) {
                std::cerr << "[MetalBackend] MatMul small-batch GEMV: Buffer creation failed, "
                             "falling back to CPU"
                          << std::endl;
                apple::GemmAccelerate(C->DataAs<float>(), A.DataAs<float>(), B.DataAs<float>(), M,
                                      N, K);
                return;
            }

            if (impl_->gemvBatchedPipeline) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl_->gemvBatchedPipeline];
                [encoder setBuffer:bufferA offset:0 atIndex:0];
                [encoder setBuffer:bufferB offset:0 atIndex:1];
                [encoder setBuffer:bufferC offset:0 atIndex:2];

                const uint B_u = static_cast<uint>(M);
                const uint N_u = static_cast<uint>(N);
                const uint K_u = static_cast<uint>(K);
                [encoder setBytes:&B_u length:sizeof(uint) atIndex:3];
                [encoder setBytes:&N_u length:sizeof(uint) atIndex:4];
                [encoder setBytes:&K_u length:sizeof(uint) atIndex:5];

                const MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                const MTLSize gridSize =
                    MTLSizeMake(static_cast<NSUInteger>(N), static_cast<NSUInteger>(M), 1);
                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            } else {
                // Fallback small-batch path: encode B independent GEMV launches in one command
                // buffer.
                for (int b = 0; b < M; ++b) {
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    [encoder setComputePipelineState:impl_->gemvPipeline];
                    [encoder setBuffer:bufferA
                                offset:static_cast<NSUInteger>(b) * static_cast<NSUInteger>(K) *
                                       sizeof(float)
                               atIndex:0];
                    [encoder setBuffer:bufferB offset:0 atIndex:1];
                    [encoder setBuffer:bufferC
                                offset:static_cast<NSUInteger>(b) * static_cast<NSUInteger>(N) *
                                       sizeof(float)
                               atIndex:2];

                    const uint N_u = static_cast<uint>(N);
                    const uint K_u = static_cast<uint>(K);
                    [encoder setBytes:&N_u length:sizeof(uint) atIndex:3];
                    [encoder setBytes:&K_u length:sizeof(uint) atIndex:4];

                    const MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                    const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);
                    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];
                }
            }

            [commandBuffer commit];
        } else {
            // GEMM path (M > 1): Use Metal Performance Shaders
            @autoreleasepool {
                id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

                // Get or wrap MTLBuffer handles for the tensor data (zero-copy)
                // Uses newBufferWithBytesNoCopy for untracked pointers (UMA zero-copy)
                id<MTLBuffer> bufferA =
                    impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
                id<MTLBuffer> bufferB =
                    impl_->GetOrWrapBuffer(const_cast<void*>(B.data), B.SizeBytes());
                id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

                if (!bufferA || !bufferB || !bufferC) {
                    // Should never happen on Apple Silicon, but safety fallback
                    std::cerr << "[MetalBackend] MatMul: Buffer creation failed, falling "
                                 "back to CPU"
                              << std::endl;
                    apple::GemmAccelerate(C->DataAs<float>(), A.DataAs<float>(), B.DataAs<float>(),
                                          M, N, K);
                    return;
                }

                // Calculate row bytes (must be 4-byte aligned for MPS)
                NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
                NSUInteger rowBytesB = static_cast<NSUInteger>(N) * sizeof(float);
                NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

                // Create MPS matrix descriptors
                MPSMatrixDescriptor* descA =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                          columns:static_cast<NSUInteger>(K)
                                                         rowBytes:rowBytesA
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descB =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(K)
                                                          columns:static_cast<NSUInteger>(N)
                                                         rowBytes:rowBytesB
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descC =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                          columns:static_cast<NSUInteger>(N)
                                                         rowBytes:rowBytesC
                                                         dataType:MPSDataTypeFloat32];

                // Wrap buffers as MPSMatrix (zero-copy)
                MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
                MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
                MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

                // Create and encode MPS GEMM: C = A @ B
                MPSMatrixMultiplication* gemm =
                    [[MPSMatrixMultiplication alloc] initWithDevice:impl_->device
                                                      transposeLeft:NO
                                                     transposeRight:NO
                                                         resultRows:static_cast<NSUInteger>(M)
                                                      resultColumns:static_cast<NSUInteger>(N)
                                                    interiorColumns:static_cast<NSUInteger>(K)
                                                              alpha:1.0
                                                               beta:0.0];

                [gemm encodeToCommandBuffer:commandBuffer
                                 leftMatrix:matrixA
                                rightMatrix:matrixB
                               resultMatrix:matrixC];

                [commandBuffer commit];
            }
        }
    }
}

void MetalBackend::MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) {
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[0]);

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        id<MTLBuffer> bufferA = impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
        id<MTLBuffer> bufferB = impl_->GetOrWrapBuffer(const_cast<void*>(B.data), B.SizeBytes());
        id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

        if (commandBuffer && bufferA && bufferB && bufferC) {
            const NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
            const NSUInteger rowBytesB = static_cast<NSUInteger>(K) * sizeof(float);
            const NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

            MPSMatrixDescriptor* descA =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                      columns:static_cast<NSUInteger>(K)
                                                     rowBytes:rowBytesA
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(N)
                                                      columns:static_cast<NSUInteger>(K)
                                                     rowBytes:rowBytesB
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                      columns:static_cast<NSUInteger>(N)
                                                     rowBytes:rowBytesC
                                                     dataType:MPSDataTypeFloat32];

            MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
            MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
            MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

            // C = A @ B^T
            MPSMatrixMultiplication* gemm =
                [[MPSMatrixMultiplication alloc] initWithDevice:impl_->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:static_cast<NSUInteger>(M)
                                                  resultColumns:static_cast<NSUInteger>(N)
                                                interiorColumns:static_cast<NSUInteger>(K)
                                                          alpha:1.0
                                                           beta:0.0];

            [gemm encodeToCommandBuffer:commandBuffer
                             leftMatrix:matrixA
                            rightMatrix:matrixB
                           resultMatrix:matrixC];
            [commandBuffer commit];
            return;
        }
    }

    // CPU fallback
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A.DataAs<float>(), K,
                B.DataAs<float>(), K, 0.0f, C->DataAs<float>(), N);
}

void MetalBackend::GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales,
                            const Tensor& zero_points, Tensor* C, int /*group_size*/) {
    if (!A.IsValid() || !W.IsValid() || !scales.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(C->shape[C->ndim - 1]);
    const int K = static_cast<int>(A.shape[A.ndim - 1]);
    const bool use_q4_1 = zero_points.IsValid() && zero_points.NumElements() > 0;

    // Decode phase (M == 1): Use custom quantized GEMV kernel on GPU
    if (M == 1) {
        @autoreleasepool {
            id<MTLComputePipelineState> pipeline =
                use_q4_1 ? impl_->gemvQ4_1Pipeline : impl_->gemvQ4_0Pipeline;

            if (pipeline) {
                id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

                id<MTLBuffer> bufferA =
                    impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
                id<MTLBuffer> bufferW =
                    impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());
                id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());
                if (!bufferA || !bufferW || !bufferC) {
                    std::cerr << "[MetalBackend] GemmInt4 GEMV: buffer creation failed"
                              << std::endl;
                    // Fall through to other paths.
                } else {
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];

                    // Buffer 0: input activations [K]
                    [encoder setBuffer:bufferA offset:0 atIndex:0];
                    // Buffer 1: packed weights (block_q4_0 or block_q4_1 structs)
                    [encoder setBuffer:bufferW offset:0 atIndex:1];
                    // Buffer 2: output [N]
                    [encoder setBuffer:bufferC offset:0 atIndex:2];
                    // Buffer 3: M (output rows)
                    uint M_u = static_cast<uint>(N);  // For GEMV, N is the output dim
                    [encoder setBytes:&M_u length:sizeof(uint) atIndex:3];
                    // Buffer 4: K (input dimension)
                    uint K_u = static_cast<uint>(K);
                    [encoder setBytes:&K_u length:sizeof(uint) atIndex:4];

                    // Dispatch: one threadgroup per output element
                    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                    MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);

                    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];

                    [commandBuffer commit];
                    // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                    return;
                }
            }
        }
    }

    // Prefill (M > 1): Prefer fused GPU INT4 GEMM to avoid a global FP32
    // dequantization pass before GEMM.
    @autoreleasepool {
        const bool useSimdgroupPrefill =
            impl_->supportsSimdgroupMatrix && impl_->enableSimdgroupInt4Gemm && M >= 4 && K >= 32;
        id<MTLComputePipelineState> simdgroupPipeline =
            use_q4_1 ? impl_->gemmQ4_1SimdgroupPipeline : impl_->gemmQ4_0SimdgroupPipeline;
        if (useSimdgroupPrefill && simdgroupPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferW =
                impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

            if (commandBuffer && bufferA && bufferW && bufferC) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (encoder) {
                    [encoder setComputePipelineState:simdgroupPipeline];
                    [encoder setBuffer:bufferA offset:0 atIndex:0];
                    [encoder setBuffer:bufferW offset:0 atIndex:1];
                    [encoder setBuffer:bufferC offset:0 atIndex:2];

                    uint M_u = static_cast<uint>(M);
                    uint N_u = static_cast<uint>(N);
                    uint K_u = static_cast<uint>(K);
                    [encoder setBytes:&M_u length:sizeof(uint) atIndex:3];
                    [encoder setBytes:&N_u length:sizeof(uint) atIndex:4];
                    [encoder setBytes:&K_u length:sizeof(uint) atIndex:5];

                    const MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
                    const MTLSize gridSize = MTLSizeMake((static_cast<NSUInteger>(N) + 7) / 8,
                                                         (static_cast<NSUInteger>(M) + 7) / 8, 1);
                    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];
                    [commandBuffer commit];
                    return;
                }
            }
        }

        id<MTLComputePipelineState> fusedPipeline =
            use_q4_1 ? impl_->gemmQ4_1FusedPipeline : impl_->gemmQ4_0FusedPipeline;
        if (fusedPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferW =
                impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

            if (commandBuffer && bufferA && bufferW && bufferC) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (encoder) {
                    [encoder setComputePipelineState:fusedPipeline];
                    [encoder setBuffer:bufferA offset:0 atIndex:0];
                    [encoder setBuffer:bufferW offset:0 atIndex:1];
                    [encoder setBuffer:bufferC offset:0 atIndex:2];

                    uint M_u = static_cast<uint>(M);
                    uint N_u = static_cast<uint>(N);
                    uint K_u = static_cast<uint>(K);
                    [encoder setBytes:&M_u length:sizeof(uint) atIndex:3];
                    [encoder setBytes:&N_u length:sizeof(uint) atIndex:4];
                    [encoder setBytes:&K_u length:sizeof(uint) atIndex:5];

                    const MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);
                    const MTLSize gridSize = MTLSizeMake((static_cast<NSUInteger>(N) + 7) / 8,
                                                         (static_cast<NSUInteger>(M) + 7) / 8, 1);
                    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];
                    [commandBuffer commit];
                    return;
                }
            }
        }
    }

    // Fallback prefill path: GPU dequantization + MPS GEMM
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

        // Step 1: Get scratch buffer for dequantized weights [N, K] (reuses pool)
        const size_t deq_size = static_cast<size_t>(N) * K * sizeof(float);
        id<MTLBuffer> dequantBuffer = impl_->GetScratchBuffer(deq_size);

        if (!dequantBuffer) {
            std::cerr << "[MetalBackend] GemmInt4: Failed to get scratch buffer" << std::endl;
            // Fall through to CPU fallback below
        } else {
            @autoreleasepool {
                id<MTLComputePipelineState> dequantPipeline =
                    use_q4_1 ? impl_->dequantizeQ4_1Pipeline : impl_->dequantizeQ4_0Pipeline;
                if (!dequantPipeline) {
                    std::cerr << "[MetalBackend] GemmInt4: Missing dequantize pipeline for "
                              << (use_q4_1 ? "Q4_1" : "Q4_0") << std::endl;
                } else {
                    // Step 2: Dispatch dequantization kernel
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    [encoder setComputePipelineState:dequantPipeline];

                    // Get or wrap buffer for packed weights (zero-copy)
                    id<MTLBuffer> weightBuffer =
                        impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());

                    [encoder setBuffer:weightBuffer offset:0 atIndex:0];
                    [encoder setBuffer:dequantBuffer offset:0 atIndex:1];
                    uint N_u = static_cast<uint>(N);
                    uint K_u = static_cast<uint>(K);
                    [encoder setBytes:&N_u length:sizeof(uint) atIndex:2];
                    [encoder setBytes:&K_u length:sizeof(uint) atIndex:3];

                    const int block_size = 32;
                    uint blocks_per_row =
                        static_cast<uint>((K + block_size - 1) / block_size);  // Ceiling division
                    MTLSize gridSize = MTLSizeMake(blocks_per_row, static_cast<NSUInteger>(N), 1);
                    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);  // One thread per block
                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];

                    // Step 3: MPS MatMul on dequantized weights
                    // Use GetOrWrapBuffer for zero-copy access
                    id<MTLBuffer> bufferA =
                        impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
                    id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

                    if (bufferA && bufferC) {
                        // A is [M, K], dequantized B is [N, K], we want C = A @ B^T = [M, N]
                        NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
                        NSUInteger rowBytesB = static_cast<NSUInteger>(K) * sizeof(float);
                        NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

                        MPSMatrixDescriptor* descA =
                            [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                                  columns:static_cast<NSUInteger>(K)
                                                                 rowBytes:rowBytesA
                                                                 dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor* descB =
                            [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(N)
                                                                  columns:static_cast<NSUInteger>(K)
                                                                 rowBytes:rowBytesB
                                                                 dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor* descC =
                            [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                                  columns:static_cast<NSUInteger>(N)
                                                                 rowBytes:rowBytesC
                                                                 dataType:MPSDataTypeFloat32];

                        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA
                                                                    descriptor:descA];
                        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:dequantBuffer
                                                                    descriptor:descB];
                        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC
                                                                    descriptor:descC];

                        // C = A @ B^T: [M,K] @ [N,K]^T = [M,N]
                        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                             initWithDevice:impl_->device
                              transposeLeft:NO
                             transposeRight:YES
                                 resultRows:static_cast<NSUInteger>(M)
                              resultColumns:static_cast<NSUInteger>(N)
                            interiorColumns:static_cast<NSUInteger>(K)
                                      alpha:1.0
                                       beta:0.0];

                        [gemm encodeToCommandBuffer:commandBuffer
                                         leftMatrix:matrixA
                                        rightMatrix:matrixB
                                       resultMatrix:matrixC];

                        [commandBuffer commit];
                        // Note: With zero-copy wrapping via newBufferWithBytesNoCopy,
                        // results are written directly to C->data, no copy needed

                        return;  // Success - exit early
                    }
                }
            }
        }
    }

    // CPU fallback only if GPU path failed
    std::cerr << "[MetalBackend] GemmInt4: GPU path failed, using CPU fallback" << std::endl;

    const float* a_data = A.DataAs<float>();
    const uint8_t* w_data = static_cast<const uint8_t*>(W.data);
    const float* scale_data = scales.DataAs<float>();
    float* c_data = C->DataAs<float>();

    const int block_size = 32;  // QK4_0 = QK4_1 = 32
    const int blocks_per_row = K / block_size;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a_data + m * K;
        float* c_row = c_data + m * N;

        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                float scale = scale_data[n * blocks_per_row + blk];
                int k_start = blk * block_size;
                const uint8_t* block_quants =
                    w_data + (n * blocks_per_row + blk) * (2 + block_size / 2) + 2;

                for (int i = 0; i < block_size; i += 2) {
                    uint8_t packed = block_quants[i / 2];
                    int8_t q0 = static_cast<int8_t>((packed & 0x0F)) - 8;
                    int8_t q1 = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;
                    sum += (static_cast<float>(q0) * scale) * a_row[k_start + i];
                    sum += (static_cast<float>(q1) * scale) * a_row[k_start + i + 1];
                }
            }
            c_row[n] = sum;
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Normalization
// ============================================================================

void MetalBackend::RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps) {
    if (!input.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
        return;
    }

    const int64_t dim = weight.shape[0];
    const int64_t n_tokens = input.NumElements() / dim;

    @autoreleasepool {
        if (impl_->rmsNormPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            id<MTLBuffer> inputBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(input.data), input.SizeBytes());
            id<MTLBuffer> weightBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(weight.data), weight.SizeBytes());
            id<MTLBuffer> outputBuffer = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());

            if (!inputBuffer || !weightBuffer || !outputBuffer) {
                std::cerr << "[MetalBackend] RMSNorm: GPU buffer setup failed, using CPU fallback"
                          << std::endl;
            } else {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl_->rmsNormPipeline];
                [encoder setBuffer:inputBuffer offset:0 atIndex:0];
                [encoder setBuffer:weightBuffer offset:0 atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];

                uint dim_u = static_cast<uint>(dim);
                [encoder setBytes:&dim_u length:sizeof(uint) atIndex:3];
                [encoder setBytes:&eps length:sizeof(float) atIndex:4];

                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(n_tokens), 1, 1);

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }

        // CPU fallback
        const float* x = input.DataAs<float>();
        const float* w = weight.DataAs<float>();
        float* out = output->DataAs<float>();

        for (int64_t t = 0; t < n_tokens; ++t) {
            const float* x_ptr = x + t * dim;
            float* out_ptr = out + t * dim;

            float sum_sq = 0.0f;
            for (int64_t i = 0; i < dim; ++i) {
                sum_sq += x_ptr[i] * x_ptr[i];
            }
            float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

            for (int64_t i = 0; i < dim; ++i) {
                out_ptr[i] = x_ptr[i] * rms * w[i];
            }
        }
    }
}

void MetalBackend::AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight,
                              Tensor* output, float eps) {
    if (!input.IsValid() || !residual.IsValid() || !weight.IsValid() || !output ||
        !output->IsValid()) {
        return;
    }

    const int64_t dim = weight.shape[0];
    if (dim <= 0) {
        return;
    }
    const int64_t n_tokens = input.NumElements() / dim;

    @autoreleasepool {
        if (impl_->addRmsNormPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            id<MTLBuffer> inputBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(input.data), input.SizeBytes());
            id<MTLBuffer> residualBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(residual.data), residual.SizeBytes());
            id<MTLBuffer> weightBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(weight.data), weight.SizeBytes());
            id<MTLBuffer> outputBuffer = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());

            if (inputBuffer && residualBuffer && weightBuffer && outputBuffer) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl_->addRmsNormPipeline];
                [encoder setBuffer:inputBuffer offset:0 atIndex:0];
                [encoder setBuffer:residualBuffer offset:0 atIndex:1];
                [encoder setBuffer:weightBuffer offset:0 atIndex:2];
                [encoder setBuffer:outputBuffer offset:0 atIndex:3];

                uint dim_u = static_cast<uint>(dim);
                [encoder setBytes:&dim_u length:sizeof(uint) atIndex:4];
                [encoder setBytes:&eps length:sizeof(float) atIndex:5];

                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(n_tokens), 1, 1);

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }

            std::cerr << "[MetalBackend] AddRMSNorm: GPU buffer setup failed, using fallback"
                      << std::endl;
        }
    }

    // Fallback: add on CPU, then normalize (GPU if available in RMSNorm).
    const int64_t n_elements = input.NumElements();
    float* out = output->DataAs<float>();
    const float* in = input.DataAs<float>();
    const float* res = residual.DataAs<float>();
    for (int64_t i = 0; i < n_elements; ++i) {
        out[i] = in[i] + res[i];
    }
    Tensor temp_input = *output;
    RMSNorm(temp_input, weight, output, eps);
}

// ============================================================================
// ComputeBackend Interface - Activation
// ============================================================================

void MetalBackend::Softmax(const Tensor& input, Tensor* output) {
    CopyToDevice(output->data, input.data, input.SizeBytes());
    SoftmaxInplace(output);
}

void MetalBackend::SoftmaxInplace(Tensor* data) {
    if (!data || !data->IsValid()) {
        return;
    }

    const int64_t n = data->shape[data->ndim - 1];
    int64_t batch_size = 1;
    for (int i = 0; i < data->ndim - 1; ++i) {
        batch_size *= data->shape[i];
    }

    @autoreleasepool {
        if (impl_->softmaxPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            id<MTLBuffer> dataBuffer = impl_->GetOrWrapBuffer(data->data, data->SizeBytes());

            if (commandBuffer && encoder && dataBuffer) {
                [encoder setComputePipelineState:impl_->softmaxPipeline];
                [encoder setBuffer:dataBuffer offset:0 atIndex:0];
                uint n_u = static_cast<uint>(n);
                uint stride_u = static_cast<uint>(n);
                uint rows_u = static_cast<uint>(std::max<int64_t>(1, batch_size));
                [encoder setBytes:&n_u length:sizeof(uint) atIndex:1];
                [encoder setBytes:&stride_u length:sizeof(uint) atIndex:2];
                [encoder setBytes:&rows_u length:sizeof(uint) atIndex:3];

                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(rows_u), 1, 1);

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }

        // CPU fallback
        {
            float* ptr = data->DataAs<float>();
            for (int64_t b = 0; b < batch_size; ++b) {
                float* row = ptr + b * n;

                // Find max
                float max_val = row[0];
                for (int64_t i = 1; i < n; ++i) {
                    if (row[i] > max_val)
                        max_val = row[i];
                }

                // Exp and sum
                float sum = 0.0f;
                for (int64_t i = 0; i < n; ++i) {
                    row[i] = std::exp(row[i] - max_val);
                    sum += row[i];
                }

                // Normalize
                float inv_sum = 1.0f / sum;
                for (int64_t i = 0; i < n; ++i) {
                    row[i] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Position Encoding
// ============================================================================

void MetalBackend::RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions,
                        Tensor* output, int rope_dim) {
    if (!input.IsValid() || !cos_sin.IsValid() || !positions || !output || !output->IsValid()) {
        return;
    }

    // Copy input to output first (RoPE is in-place on output)
    CopyToDevice(output->data, input.data, input.SizeBytes());

    // Parse dimensions
    uint n_tokens, head_dim, n_heads;
    if (input.ndim == 2) {
        n_tokens = static_cast<uint>(input.shape[0]);
        head_dim = static_cast<uint>(input.shape[1]);
        n_heads = 1;
    } else {
        n_heads = static_cast<uint>(input.shape[0]);
        n_tokens = static_cast<uint>(input.shape[1]);
        head_dim = static_cast<uint>(input.shape[2]);
    }

    uint rope_dim_u = (rope_dim < 0) ? head_dim : static_cast<uint>(rope_dim);

    // GPU path
    if (impl_->ropePipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->ropePipeline];

            // Get or wrap buffers - prefer zero-copy via GetOrWrapBuffer
            id<MTLBuffer> dataBuffer = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());
            id<MTLBuffer> cosSinBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(cos_sin.data), cos_sin.SizeBytes());

            // Create buffer for positions
            size_t pos_size = n_tokens * sizeof(int);
            id<MTLBuffer> posBuffer =
                [impl_->device newBufferWithBytes:positions
                                           length:pos_size
                                          options:MTLResourceStorageModeShared];

            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:cosSinBuffer offset:0 atIndex:1];
            [encoder setBuffer:posBuffer offset:0 atIndex:2];
            [encoder setBytes:&n_heads length:sizeof(uint) atIndex:3];
            [encoder setBytes:&n_tokens length:sizeof(uint) atIndex:4];
            [encoder setBytes:&head_dim length:sizeof(uint) atIndex:5];
            [encoder setBytes:&rope_dim_u length:sizeof(uint) atIndex:6];

            // Grid: (rope_dim/2, n_tokens, n_heads)
            MTLSize gridSize = MTLSizeMake(rope_dim_u / 2, n_tokens, n_heads);
            MTLSize threadgroupSize = MTLSizeMake(std::min(rope_dim_u / 2, 64u), 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // Zero-copy via GetOrWrapBuffer - no copy-back needed

            return;
        }
    }

    // CPU fallback (only if GPU pipeline not available)
    float* out = output->DataAs<float>();
    const float* cs = cos_sin.DataAs<float>();

    for (uint t = 0; t < n_tokens; ++t) {
        int pos = positions[t];
        const float* pos_cs = cs + pos * head_dim;

        for (uint h = 0; h < n_heads; ++h) {
            float* token = out + (h * n_tokens + t) * head_dim;

            for (uint d = 0; d < rope_dim_u / 2; ++d) {
                float cos_theta = pos_cs[2 * d];
                float sin_theta = pos_cs[2 * d + 1];

                float x0 = token[2 * d];
                float x1 = token[2 * d + 1];

                token[2 * d] = x0 * cos_theta - x1 * sin_theta;
                token[2 * d + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Fused Operations
// ============================================================================

void MetalBackend::FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk,
                                      const Tensor& wv, Tensor* q_out, Tensor* k_out,
                                      Tensor* v_out) {
    if (!input.IsValid() || !wq.IsValid() || !wk.IsValid() || !wv.IsValid() || !q_out || !k_out ||
        !v_out) {
        return;
    }

    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[input.ndim - 1]);
    const int hidden_q = static_cast<int>(wq.shape[0]);
    const int hidden_kv = static_cast<int>(wk.shape[0]);

    // Decode phase (M == 1): Use fused kernel for single dispatch
    if (M == 1 && impl_->fusedQKVPipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            id<MTLBuffer> inputBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(input.data), input.SizeBytes());
            id<MTLBuffer> wqBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(wq.data), wq.SizeBytes());
            id<MTLBuffer> wkBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(wk.data), wk.SizeBytes());
            id<MTLBuffer> wvBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(wv.data), wv.SizeBytes());
            id<MTLBuffer> qBuffer = impl_->GetOrWrapBuffer(q_out->data, q_out->SizeBytes());
            id<MTLBuffer> kBuffer = impl_->GetOrWrapBuffer(k_out->data, k_out->SizeBytes());
            id<MTLBuffer> vBuffer = impl_->GetOrWrapBuffer(v_out->data, v_out->SizeBytes());
            if (!inputBuffer || !wqBuffer || !wkBuffer || !wvBuffer || !qBuffer || !kBuffer ||
                !vBuffer) {
                std::cerr
                    << "[MetalBackend] FusedQKVProjection: buffer setup failed, using fallback"
                    << std::endl;
            } else {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl_->fusedQKVPipeline];

                // Set buffers
                [encoder setBuffer:inputBuffer offset:0 atIndex:0];
                [encoder setBuffer:wqBuffer offset:0 atIndex:1];
                [encoder setBuffer:wkBuffer offset:0 atIndex:2];
                [encoder setBuffer:wvBuffer offset:0 atIndex:3];
                [encoder setBuffer:qBuffer offset:0 atIndex:4];
                [encoder setBuffer:kBuffer offset:0 atIndex:5];
                [encoder setBuffer:vBuffer offset:0 atIndex:6];

                // Set constants
                uint K_u = static_cast<uint>(K);
                uint hidden_q_u = static_cast<uint>(hidden_q);
                uint hidden_kv_u = static_cast<uint>(hidden_kv);
                [encoder setBytes:&K_u length:sizeof(uint) atIndex:7];
                [encoder setBytes:&hidden_q_u length:sizeof(uint) atIndex:8];
                [encoder setBytes:&hidden_kv_u length:sizeof(uint) atIndex:9];

                // Total rows: hidden_q (Q) + hidden_kv (K) + hidden_kv (V)
                uint total_rows = hidden_q + 2 * hidden_kv;
                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(total_rows), 1, 1);

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }
    }

    // Prefill (M > 1) or no GPU pipeline: Fallback to three separate MatMuls
    MatMulTransB(input, wq, q_out);
    MatMulTransB(input, wk, k_out);
    MatMulTransB(input, wv, v_out);
}

void MetalBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output,
                                  float scale, bool causal, int n_head_kv, int sliding_window, float logit_softcap,
                                  uint32_t semantic_flags) {
    if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output || !output->IsValid()) {
        return;
    }

    (void)sliding_window;
    (void)logit_softcap;
    (void)semantic_flags;

    const int batch = static_cast<int>(Q.shape[0]);
    const int n_head = static_cast<int>(Q.shape[1]);
    const int seq_q = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int available_kv_heads = static_cast<int>(K.shape[1]);
    const int seq_kv = static_cast<int>(K.shape[2]);
    constexpr int kMaxFlashHeadDim = 128;

    if (n_head_kv <= 0)
        n_head_kv = available_kv_heads;

    const bool valid_gqa =
        n_head_kv > 0 && n_head_kv == available_kv_heads && (n_head % n_head_kv == 0);
    if (!valid_gqa) {
        std::cerr << "[MetalBackend] Warning: Invalid GQA config for FlashAttention (n_head="
                  << n_head << ", requested n_head_kv=" << n_head_kv
                  << ", tensor kv_heads=" << available_kv_heads << "), falling back to CPU"
                  << std::endl;
    }

    if (head_dim > kMaxFlashHeadDim) {
        static bool warned_head_dim = false;
        if (!warned_head_dim) {
            std::cerr << "[MetalBackend] Warning: FlashAttention head_dim=" << head_dim
                      << " exceeds Metal kernel limit (" << kMaxFlashHeadDim
                      << "). Falling back to CPU." << std::endl;
            warned_head_dim = true;
        }
    }

    // ==========================================================================
    // GPU Path: Use Metal FlashAttention for decode (seq_q == 1)
    // ==========================================================================
    if (valid_gqa && head_dim <= kMaxFlashHeadDim && seq_q == 1 &&
        impl_->flashAttentionDecodePipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->flashAttentionDecodePipeline];

            // Set buffers - use zero-copy MTLBuffer for UMA efficiency
            id<MTLBuffer> bufferQ =
                impl_->GetOrWrapBuffer(const_cast<void*>(Q.data), Q.SizeBytes());
            id<MTLBuffer> bufferK =
                impl_->GetOrWrapBuffer(const_cast<void*>(K.data), K.SizeBytes());
            id<MTLBuffer> bufferV =
                impl_->GetOrWrapBuffer(const_cast<void*>(V.data), V.SizeBytes());
            id<MTLBuffer> bufferOut = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());
            if (!commandBuffer || !encoder || !bufferQ || !bufferK || !bufferV || !bufferOut) {
                std::cerr << "[MetalBackend] FlashAttention decode: GPU buffer setup failed, "
                             "falling back to CPU"
                          << std::endl;
            } else {
                [encoder setBuffer:bufferQ offset:0 atIndex:0];
                [encoder setBuffer:bufferK offset:0 atIndex:1];
                [encoder setBuffer:bufferV offset:0 atIndex:2];
                [encoder setBuffer:bufferOut offset:0 atIndex:3];

                // Set constants
                [encoder setBytes:&scale length:sizeof(float) atIndex:4];
                uint seq_kv_u = static_cast<uint>(seq_kv);
                uint head_dim_u = static_cast<uint>(head_dim);
                uint n_heads_u = static_cast<uint>(n_head);
                uint n_kv_heads_u = static_cast<uint>(n_head_kv);
                [encoder setBytes:&seq_kv_u length:sizeof(uint) atIndex:5];
                [encoder setBytes:&head_dim_u length:sizeof(uint) atIndex:6];
                [encoder setBytes:&n_heads_u length:sizeof(uint) atIndex:7];
                [encoder setBytes:&n_kv_heads_u length:sizeof(uint) atIndex:8];

                // Dispatch: one threadgroup per (batch, head) pair
                // Threadgroup size is capped to reduce threadgroup memory footprint.
                MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
                MTLSize gridSize =
                    MTLSizeMake(static_cast<NSUInteger>(n_head), 1, static_cast<NSUInteger>(batch));

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }
    }

    // ==========================================================================
    // GPU Path: Use Metal FlashAttention for prefill (seq_q > 1)
    // ==========================================================================
    if (valid_gqa && head_dim <= kMaxFlashHeadDim && seq_q > 1 &&
        impl_->flashAttentionPrefillPipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->flashAttentionPrefillPipeline];

            // Set buffers - use zero-copy MTLBuffer for UMA efficiency
            id<MTLBuffer> bufferQ =
                impl_->GetOrWrapBuffer(const_cast<void*>(Q.data), Q.SizeBytes());
            id<MTLBuffer> bufferK =
                impl_->GetOrWrapBuffer(const_cast<void*>(K.data), K.SizeBytes());
            id<MTLBuffer> bufferV =
                impl_->GetOrWrapBuffer(const_cast<void*>(V.data), V.SizeBytes());
            id<MTLBuffer> bufferOut = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());
            if (!commandBuffer || !encoder || !bufferQ || !bufferK || !bufferV || !bufferOut) {
                std::cerr << "[MetalBackend] FlashAttention prefill: GPU buffer setup failed, "
                             "falling back to CPU"
                          << std::endl;
            } else {
                [encoder setBuffer:bufferQ offset:0 atIndex:0];
                [encoder setBuffer:bufferK offset:0 atIndex:1];
                [encoder setBuffer:bufferV offset:0 atIndex:2];
                [encoder setBuffer:bufferOut offset:0 atIndex:3];

                // Set constants
                [encoder setBytes:&scale length:sizeof(float) atIndex:4];
                uint seq_q_u = static_cast<uint>(seq_q);
                uint seq_kv_u = static_cast<uint>(seq_kv);
                uint head_dim_u = static_cast<uint>(head_dim);
                uint n_heads_u = static_cast<uint>(n_head);
                uint n_kv_heads_u = static_cast<uint>(n_head_kv);
                uint causal_u = causal ? 1 : 0;
                [encoder setBytes:&seq_q_u length:sizeof(uint) atIndex:5];
                [encoder setBytes:&seq_kv_u length:sizeof(uint) atIndex:6];
                [encoder setBytes:&head_dim_u length:sizeof(uint) atIndex:7];
                [encoder setBytes:&n_heads_u length:sizeof(uint) atIndex:8];
                [encoder setBytes:&n_kv_heads_u length:sizeof(uint) atIndex:9];
                [encoder setBytes:&causal_u length:sizeof(uint) atIndex:10];

                // Dispatch: one threadgroup per (head, q_block, batch) triple.
                // flash_attention_prefill uses BQ=16 queries per block.
                const int q_blocks = std::max(1, (seq_q + 16 - 1) / 16);
                MTLSize threadgroupSize = MTLSizeMake(128, 1, 1);
                MTLSize gridSize =
                    MTLSizeMake(static_cast<NSUInteger>(n_head), static_cast<NSUInteger>(q_blocks),
                                static_cast<NSUInteger>(batch));

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }
    }

    // ==========================================================================
    // CPU Fallback: Naive O(N^2) attention when GPU pipelines unavailable
    // ==========================================================================
    std::cerr << "[MetalBackend] Warning: FlashAttention falling back to CPU. seq_q=" << seq_q
              << std::endl;

    const int cpu_n_head_kv = std::max(1, available_kv_heads);
    const int n_rep = std::max(1, (n_head + cpu_n_head_kv - 1) / cpu_n_head_kv);

    const float* q_data = Q.DataAs<float>();
    const float* k_data = K.DataAs<float>();
    const float* v_data = V.DataAs<float>();
    float* o_data = output->DataAs<float>();

    // Allocate temporary scores
    std::vector<float> scores(seq_q * seq_kv);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_head; ++h) {
            int h_kv = std::min(h / n_rep, cpu_n_head_kv - 1);  // KV head index for GQA

            // Q @ K^T
            for (int i = 0; i < seq_q; ++i) {
                for (int j = 0; j < seq_kv; ++j) {
                    float dot = 0.0f;
                    const float* q_ptr = q_data + ((b * n_head + h) * seq_q + i) * head_dim;
                    const float* k_ptr =
                        k_data + ((b * cpu_n_head_kv + h_kv) * seq_kv + j) * head_dim;

                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_ptr[d] * k_ptr[d];
                    }

                    scores[i * seq_kv + j] = dot * scale;

                    // Causal mask
                    if (causal && j > i) {
                        scores[i * seq_kv + j] = -INFINITY;
                    }
                }
            }

            // Softmax per row
            for (int i = 0; i < seq_q; ++i) {
                float* row = scores.data() + i * seq_kv;

                float max_val = row[0];
                for (int j = 1; j < seq_kv; ++j) {
                    if (row[j] > max_val)
                        max_val = row[j];
                }

                float sum = 0.0f;
                for (int j = 0; j < seq_kv; ++j) {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }

                for (int j = 0; j < seq_kv; ++j) {
                    row[j] /= sum;
                }
            }

            // Scores @ V
            for (int i = 0; i < seq_q; ++i) {
                float* o_ptr = o_data + ((b * n_head + h) * seq_q + i) * head_dim;

                for (int d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_kv; ++j) {
                        const float* v_ptr =
                            v_data + ((b * cpu_n_head_kv + h_kv) * seq_kv + j) * head_dim;
                        sum += scores[i * seq_kv + j] * v_ptr[d];
                    }
                    o_ptr[d] = sum;
                }
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Synchronization
// ============================================================================

void MetalBackend::Synchronize() {
    @autoreleasepool {
        // Create a completion fence
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// ============================================================================
// Metal-Specific APIs
// ============================================================================

AppleSiliconChipInfo MetalBackend::GetDetailedChipInfo() const {
    return impl_->chipInfo;
}

bool MetalBackend::SupportsGPUFamily(int family) const {
    @autoreleasepool {
        return [impl_->device supportsFamily:static_cast<MTLGPUFamily>(family)];
    }
}

size_t MetalBackend::GetCurrentMemoryUsage() const {
    return impl_->currentMemoryUsage.load();
}

size_t MetalBackend::GetPeakMemoryUsage() const {
    return impl_->peakMemoryUsage.load();
}

void MetalBackend::EnableGPUCapture(const char* capture_path) {
    @autoreleasepool {
        MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
        MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = impl_->device;

        if (capture_path) {
            descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
            descriptor.outputURL =
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:capture_path]];
        } else {
            descriptor.destination = MTLCaptureDestinationDeveloperTools;
        }

        NSError* error = nil;
        if ([captureManager startCaptureWithDescriptor:descriptor error:&error]) {
            impl_->captureEnabled = true;
        } else {
            std::cerr << "[MetalBackend] Failed to start GPU capture: " <<
                [[error localizedDescription] UTF8String] << std::endl;
        }
    }
}

void MetalBackend::DisableGPUCapture() {
    @autoreleasepool {
        if (impl_->captureEnabled) {
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            impl_->captureEnabled = false;
        }
    }
}

bool MetalBackend::GemmInt4Grouped(const Tensor& A, const Tensor& W, const Tensor& scales,
                                   const Tensor& zero_points, Tensor* C, int group_size) {
    if (!A.IsValid() || !W.IsValid() || !scales.IsValid() || !C || !C->IsValid() ||
        group_size <= 0) {
        return false;
    }

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = W.shape[0];
    const bool has_zero_points = zero_points.IsValid() && zero_points.NumElements() > 0;

    if (M <= 0 || N <= 0 || K <= 0 || (K % 2) != 0 || (K % group_size) != 0) {
        return false;
    }
    if (!impl_->gemmInt4GroupedFusedPipeline && !impl_->gemmInt4GroupedSimdgroupPipeline) {
        return false;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        if (!commandBuffer) {
            return false;
        }

        id<MTLBuffer> bufferA = impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
        id<MTLBuffer> bufferW = impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());
        id<MTLBuffer> bufferScales =
            impl_->GetOrWrapBuffer(const_cast<void*>(scales.data), scales.SizeBytes());
        id<MTLBuffer> bufferZeros =
            has_zero_points ? impl_->GetOrWrapBuffer(const_cast<void*>(zero_points.data),
                                                     zero_points.SizeBytes())
                            : bufferScales;
        id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

        if (!bufferA || !bufferW || !bufferScales || !bufferZeros || !bufferC) {
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            return false;
        }

        const bool useSimdgroupPrefill = impl_->supportsSimdgroupMatrix &&
                                         impl_->enableSimdgroupInt4Gemm && M >= 4 && K >= 32 &&
                                         impl_->gemmInt4GroupedSimdgroupPipeline != nil;
        id<MTLComputePipelineState> pipeline = useSimdgroupPrefill
                                                   ? impl_->gemmInt4GroupedSimdgroupPipeline
                                                   : impl_->gemmInt4GroupedFusedPipeline;
        if (!pipeline) {
            [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferW offset:0 atIndex:1];
        [encoder setBuffer:bufferScales offset:0 atIndex:2];
        [encoder setBuffer:bufferZeros offset:0 atIndex:3];
        [encoder setBuffer:bufferC offset:0 atIndex:4];

        uint M_u = static_cast<uint>(M);
        uint N_u = static_cast<uint>(N);
        uint K_u = static_cast<uint>(K);
        uint group_size_u = static_cast<uint>(group_size);
        uint has_zero_points_u = has_zero_points ? 1u : 0u;
        [encoder setBytes:&M_u length:sizeof(uint) atIndex:5];
        [encoder setBytes:&N_u length:sizeof(uint) atIndex:6];
        [encoder setBytes:&K_u length:sizeof(uint) atIndex:7];
        [encoder setBytes:&group_size_u length:sizeof(uint) atIndex:8];
        [encoder setBytes:&has_zero_points_u length:sizeof(uint) atIndex:9];

        const MTLSize threadgroupSize =
            useSimdgroupPrefill ? MTLSizeMake(32, 1, 1) : MTLSizeMake(8, 8, 1);
        const MTLSize gridSize = MTLSizeMake((static_cast<NSUInteger>(N) + 7) / 8,
                                             (static_cast<NSUInteger>(M) + 7) / 8, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        return true;
    }
}

bool MetalBackend::DequantizeInt4Grouped(const void* weights_packed, const float* scales,
                                         const float* zeros, float* output, int64_t N, int64_t K,
                                         int group_size) {
    if (!weights_packed || !scales || !zeros || !output || N <= 0 || K <= 0 || group_size <= 0 ||
        (K % 2) != 0 || (K % group_size) != 0) {
        return false;
    }

    // Check if GPU kernel is available
    if (!impl_->dequantizeInt4GroupedPipeline) {
        return false;  // Fall back to CPU
    }

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        if (!commandBuffer) {
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            return false;
        }

        [encoder setComputePipelineState:impl_->dequantizeInt4GroupedPipeline];

        // Get or wrap buffers (zero-copy on UMA)
        const size_t weights_size = static_cast<size_t>(N) * (K / 2);
        const size_t scales_size = static_cast<size_t>(N) * (K / group_size) * sizeof(float);
        const size_t zeros_size = scales_size;
        const size_t output_size = static_cast<size_t>(N) * K * sizeof(float);

        id<MTLBuffer> weightsBuffer =
            impl_->GetOrWrapBuffer(const_cast<void*>(weights_packed), weights_size);
        id<MTLBuffer> scalesBuffer =
            impl_->GetOrWrapBuffer(const_cast<float*>(scales), scales_size);
        id<MTLBuffer> zerosBuffer = impl_->GetOrWrapBuffer(const_cast<float*>(zeros), zeros_size);
        id<MTLBuffer> outputBuffer = impl_->GetOrWrapBuffer(output, output_size);

        if (!weightsBuffer || !scalesBuffer || !zerosBuffer || !outputBuffer) {
            [encoder endEncoding];
            return false;
        }

        [encoder setBuffer:weightsBuffer offset:0 atIndex:0];
        [encoder setBuffer:scalesBuffer offset:0 atIndex:1];
        [encoder setBuffer:zerosBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];

        uint N_u = static_cast<uint>(N);
        uint K_u = static_cast<uint>(K);
        uint group_size_u = static_cast<uint>(group_size);
        [encoder setBytes:&N_u length:sizeof(uint) atIndex:4];
        [encoder setBytes:&K_u length:sizeof(uint) atIndex:5];
        [encoder setBytes:&group_size_u length:sizeof(uint) atIndex:6];

        // Grid size: (K/2, N) - each thread handles one packed byte (2 INT4 values)
        uint K_packed = static_cast<uint>(K / 2);
        MTLSize gridSize = MTLSizeMake(K_packed, static_cast<NSUInteger>(N), 1);
        MTLSize threadgroupSize =
            MTLSizeMake(std::min(K_packed, 256u), 1, 1);  // Up to 256 threads per threadgroup

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];  // Sync for correctness

        return true;
    }
}

// =============================================================================
// Activation Implementations (CPU Fallback)
// =============================================================================

void MetalBackend::LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                             Tensor* output, float eps) {
    // CPU fallback for LayerNorm
    // Note: In production, this should use a metal kernel or MPS
    const int64_t rows = input.shape[0];
    const int64_t cols = input.shape[1];

    const float* src = input.DataAs<float>();
    const float* g = gamma.DataAs<float>();
    const float* b = beta.DataAs<float>();
    float* dst = output->DataAs<float>();

    for (int64_t i = 0; i < rows; ++i) {
        const float* row_src = src + i * cols;
        float* row_dst = dst + i * cols;

        // Mean
        float sum = 0.0f;
        for (int64_t j = 0; j < cols; ++j)
            sum += row_src[j];
        float mean = sum / cols;

        // Variance
        float sum_sq_diff = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float diff = row_src[j] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / cols;
        float inv_std = 1.0f / std::sqrt(var + eps);

        // Normalize
        for (int64_t j = 0; j < cols; ++j) {
            row_dst[j] = (row_src[j] - mean) * inv_std * g[j] + b[j];
        }
    }
}

void MetalBackend::SiLU(const Tensor& input, Tensor* output) {
    // CPU fallback
    const int64_t n = input.NumElements();
    const float* src = input.DataAs<float>();
    float* dst = output->DataAs<float>();

    for (int64_t i = 0; i < n; ++i) {
        float x = src[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        dst[i] = x * sigmoid;
    }
}

void MetalBackend::GELU(const Tensor& input, Tensor* output) {
    // CPU fallback
    const int64_t n = input.NumElements();
    const float* src = input.DataAs<float>();
    float* dst = output->DataAs<float>();

    for (int64_t i = 0; i < n; ++i) {
        float x = src[i];
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

}  // namespace densecore

#endif  // __APPLE__
