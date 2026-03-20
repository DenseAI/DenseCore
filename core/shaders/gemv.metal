/**
 * @file gemv.metal
 * @brief Custom Metal shaders for GEMV (decode-phase) optimization
 *
 * These shaders are optimized for the LLM decode phase where batch_size = 1.
 * Standard GEMM implementations perform poorly for GEMV because:
 * 1. No parallelism along the batch dimension
 * 2. Memory-bound, not compute-bound
 *
 * Our approach:
 * - Parallelize across output dimension (M)
 * - Use threadgroup shared memory for K-dimension reduction
 * - Coalesce memory access patterns
 * - Support FP32, FP16, and INT4 weight formats
 *
 * Usage:
 *   Compile with: xcrun -sdk macosx metal -c gemv.metal -o gemv.air
 *   Link with: xcrun -sdk macosx metallib gemv.air -o densecore.metallib
 *
 * Copyright (c) 2024 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant uint THREADGROUP_SIZE = 256;
constant uint SIMD_SIZE = 32;  // Apple GPU SIMD width

// =============================================================================
// FP32 GEMV: output = weight @ input ([M,K] @ [K] = [M])
// =============================================================================

/**
 * @brief High-performance GEMV for FP32 weights
 *
 * Each threadgroup computes one output element via parallel reduction.
 * Uses shared memory to accumulate partial sums from each thread.
 *
 * Memory layout:
 * - input: [K] contiguous FP32
 * - weight: [M, K] row-major FP32
 * - output: [M] contiguous FP32
 */
kernel void gemv_f32(
    device const float* input [[buffer(0)]],      // [K]
    device const float* weight [[buffer(1)]],     // [M, K] row-major
    device float* output [[buffer(2)]],           // [M]
    constant uint& M [[buffer(3)]],               // Output dimension
    constant uint& K [[buffer(4)]],               // Input dimension
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each threadgroup handles one output row
    uint row = tgid;
    if (row >= M) return;

    // Shared memory for parallel reduction
    threadgroup float shared_sum[THREADGROUP_SIZE];

    // Each thread accumulates a portion of the dot product
    float sum = 0.0f;
    device const float* weight_row = weight + row * K;

    // Coalesced memory access: threads access consecutive K elements
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input[k], sum);
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    // Uses tree-based reduction for O(log N) complexity
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final result
    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

/**
 * @brief SIMD-optimized GEMV using simd_sum for faster reduction
 *
 * This version uses Metal's SIMD group operations for the final
 * reduction step, which is faster than shared memory on Apple GPUs.
 */
kernel void gemv_f32_simd(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    device const float* weight_row = weight + row * K;

    // Each thread accumulates partial dot product
    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input[k], sum);
    }

    // SIMD reduction within each simdgroup (32 threads)
    sum = simd_sum(sum);

    // One thread per simdgroup writes to shared memory
    threadgroup float simd_results[8];  // Max 256/32 = 8 simdgroups

    if (simd_lane == 0) {
        simd_results[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroups
    if (tid == 0) {
        uint num_simdgroups = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
        float final_sum = 0.0f;
        for (uint i = 0; i < num_simdgroups; ++i) {
            final_sum += simd_results[i];
        }
        output[row] = final_sum;
    }
}

// =============================================================================
// FP16 GEMV: For half-precision weights and activations
// =============================================================================

kernel void gemv_f16(
    device const half* input [[buffer(0)]],       // [K] FP16
    device const half* weight [[buffer(1)]],      // [M, K] FP16
    device half* output [[buffer(2)]],            // [M] FP16
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];

    // Accumulate in FP32 for precision
    float sum = 0.0f;
    device const half* weight_row = weight + row * K;

    // Process 2 elements at a time using half2
    uint k = tid * 2;
    for (; k + 1 < K; k += tg_size * 2) {
        half2 w = *reinterpret_cast<device const half2*>(weight_row + k);
        half2 x = *reinterpret_cast<device const half2*>(input + k);
        sum += float(w.x) * float(x.x) + float(w.y) * float(x.y);
    }

    // Handle remainder
    if (k < K) {
        sum += float(weight_row[k]) * float(input[k]);
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
        output[row] = half(shared_sum[0]);
    }
}

// =============================================================================
// INT4 GEMV: Quantized weights with dequantization
// =============================================================================

/**
 * @brief GEMV with INT4 packed weights and per-group quantization
 *
 * Weight format (Q4_0 style):
 * - 4-bit signed integers packed 2 per byte
 * - Per-group scale factor
 *
 * Memory layout:
 * - weight_packed: [M, K/2] uint8, two 4-bit weights per byte
 * - scales: [M, num_groups] float, one scale per group
 * - input: [K] float
 * - output: [M] float
 */
kernel void gemv_int4(
    device const float* input [[buffer(0)]],          // [K] FP32
    device const uint8_t* weight_packed [[buffer(1)]],// [M, K/2] packed INT4
    device const float* scales [[buffer(2)]],         // [M, num_groups]
    device float* output [[buffer(3)]],               // [M] FP32
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];

    uint num_groups = K / group_size;
    uint packed_K = K / 2;

    device const uint8_t* weight_row = weight_packed + row * packed_K;
    device const float* scale_row = scales + row * num_groups;

    float sum = 0.0f;

    // Each thread processes multiple groups
    for (uint g = tid; g < num_groups; g += tg_size) {
        float scale = scale_row[g];
        uint k_start = g * group_size;
        uint packed_start = k_start / 2;

        float group_sum = 0.0f;

        // Process group_size weights (packed_size bytes)
        for (uint i = 0; i < group_size; i += 2) {
            uint packed_idx = packed_start + i / 2;
            uint8_t packed_byte = weight_row[packed_idx];

            // Extract two 4-bit unsigned weights
            // Q4_0 format: unsigned 4-bit ints [0-15] centered by subtracting 8
            // This gives effective range [-8, +7] without sign extension
            uint8_t w0 = packed_byte & 0x0F;
            uint8_t w1 = (packed_byte >> 4) & 0x0F;

            // Q4_0 centering: subtract 8 to convert [0,15] -> [-8,+7]
            float f0 = (float(w0) - 8.0f) * scale;
            float f1 = (float(w1) - 8.0f) * scale;

            group_sum = fma(f0, input[k_start + i], group_sum);
            if (k_start + i + 1 < K) {
                group_sum = fma(f1, input[k_start + i + 1], group_sum);
            }
        }

        sum += group_sum;
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
// Batched GEMV: Multiple vectors at once
// =============================================================================

/**
 * @brief Batched GEMV for small batch sizes (2-8)
 *
 * For very small batches, it's still more efficient to use GEMV
 * rather than GEMM due to better cache utilization.
 */
kernel void gemv_f32_batched(
    device const float* input [[buffer(0)]],      // [B, K]
    device const float* weight [[buffer(1)]],     // [M, K]
    device float* output [[buffer(2)]],           // [B, M]
    constant uint& B [[buffer(3)]],               // Batch size
    constant uint& M [[buffer(4)]],               // Output dimension
    constant uint& K [[buffer(5)]],               // Input dimension
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_dim [[threads_per_threadgroup]])
{
    uint batch = tgid.y;
    uint row = tgid.x;
    uint tg_size = tg_dim.x;

    if (batch >= B || row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];

    device const float* input_batch = input + batch * K;
    device const float* weight_row = weight + row * K;
    device float* output_batch = output + batch * M;

    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input_batch[k], sum);
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
        output_batch[row] = shared_sum[0];
    }
}

// =============================================================================
// GEMV with Bias Addition
// =============================================================================

kernel void gemv_f32_bias(
    device const float* input [[buffer(0)]],      // [K]
    device const float* weight [[buffer(1)]],     // [M, K]
    device const float* bias [[buffer(2)]],       // [M]
    device float* output [[buffer(3)]],           // [M]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];

    float sum = 0.0f;
    device const float* weight_row = weight + row * K;

    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input[k], sum);
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
        output[row] = shared_sum[0] + bias[row];
    }
}
