/**
 * @file quantized_gemv.metal
 * @brief Quantized INT4/INT8 GEMV kernels for Metal (decode phase)
 *
 * LLM inference is memory-bandwidth bound, especially during decode phase.
 * Quantizing weights from FP16/FP32 to INT4/INT8 reduces memory transfer
 * by 2-4x, directly improving performance.
 *
 * Supported Quantization Formats:
 * 1. Q4_0: Simple 4-bit quantization with per-block scale
 * 2. Q4_1: 4-bit with scale and minimum (zero-point)
 * 3. Q4_K: K-quant format with super-blocks (better accuracy)
 * 4. Q8_0: 8-bit quantization with per-block scale
 *
 * Dequantization Formula:
 * - Q4_0: float_val = (int4_val - 8) * scale
 * - Q4_1: float_val = int4_val * scale + min
 * - Q8_0: float_val = int8_val * scale
 *
 * Performance Notes:
 * - INT4 unpacking is compute-intensive on GPU
 * - Group together unpacking operations for better throughput
 * - Use simdgroup operations when possible
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

// Block sizes for different quantization formats
constant uint QK4_0 = 32;   // Q4_0 block size
constant uint QK4_1 = 32;   // Q4_1 block size
constant uint QK8_0 = 32;   // Q8_0 block size

// Thread configuration
constant uint THREADGROUP_SIZE = 256;

// =============================================================================
// Quantized Block Structures
// =============================================================================

/**
 * @brief Q4_0 block: 32 values quantized to 4 bits
 *
 * Memory layout:
 * - scale: 2 bytes (FP16)
 * - quants: 16 bytes (32 x 4-bit values packed)
 * Total: 18 bytes for 32 values (0.5625 bytes/value)
 */
struct block_q4_0 {
    half scale;                    // Scale factor
    uint8_t quants[QK4_0 / 2];     // Packed 4-bit values (2 per byte)
};

/**
 * @brief Q4_1 block: 32 values with scale and minimum
 *
 * Memory layout:
 * - scale: 2 bytes (FP16)
 * - min: 2 bytes (FP16)
 * - quants: 16 bytes
 * Total: 20 bytes for 32 values
 */
struct block_q4_1 {
    half scale;
    half min;
    uint8_t quants[QK4_1 / 2];
};

/**
 * @brief Q8_0 block: 32 values quantized to 8 bits
 *
 * Memory layout:
 * - scale: 2 bytes (FP16)
 * - quants: 32 bytes (32 x 8-bit values)
 * Total: 34 bytes for 32 values
 */
struct block_q8_0 {
    half scale;
    int8_t quants[QK8_0];
};

// =============================================================================
// Dequantization Helper Functions
// =============================================================================

/**
 * @brief Extract 4-bit value from packed byte
 */
inline int8_t extract_q4(uint8_t packed, uint idx) {
    // idx = 0: low nibble, idx = 1: high nibble
    int8_t val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    // Q4_0 uses unsigned values 0-15, subtract 8 to center at 0
    return val - 8;
}

/**
 * @brief Extract unsigned 4-bit value (for Q4_1)
 */
inline uint8_t extract_q4_unsigned(uint8_t packed, uint idx) {
    return (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
}

/**
 * @brief Dequantize Q4_0 block to float
 */
inline void dequantize_q4_0(device const block_q4_0* block,
                            thread float* output,
                            uint count) {
    float scale = float(block->scale);

    for (uint i = 0; i < count; i += 2) {
        uint byte_idx = i / 2;
        uint8_t packed = block->quants[byte_idx];

        output[i] = float(extract_q4(packed, 0)) * scale;
        if (i + 1 < count) {
            output[i + 1] = float(extract_q4(packed, 1)) * scale;
        }
    }
}

/**
 * @brief Dequantize Q4_1 block to float
 */
inline void dequantize_q4_1(device const block_q4_1* block,
                            thread float* output,
                            uint count) {
    float scale = float(block->scale);
    float min_val = float(block->min);

    for (uint i = 0; i < count; i += 2) {
        uint byte_idx = i / 2;
        uint8_t packed = block->quants[byte_idx];

        output[i] = float(extract_q4_unsigned(packed, 0)) * scale + min_val;
        if (i + 1 < count) {
            output[i + 1] = float(extract_q4_unsigned(packed, 1)) * scale + min_val;
        }
    }
}

/**
 * @brief Dequantize Q8_0 block to float
 */
inline void dequantize_q8_0(device const block_q8_0* block,
                            thread float* output,
                            uint count) {
    float scale = float(block->scale);

    for (uint i = 0; i < count; ++i) {
        output[i] = float(block->quants[i]) * scale;
    }
}

// =============================================================================
// Q4_0 GEMV Kernel
// =============================================================================

/**
 * @brief GEMV with Q4_0 quantized weights
 *
 * Computes: output = weight @ input
 * where weight is [M, K] quantized to Q4_0 blocks
 *
 * Each threadgroup computes one output element.
 * Threads cooperatively dequantize and compute dot product.
 */
kernel void gemv_q4_0(
    device const float* input [[buffer(0)]],          // [K]
    device const block_q4_0* weight [[buffer(1)]],    // [M * K/QK4_0] blocks
    device float* output [[buffer(2)]],               // [M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    // Shared memory for parallel reduction
    threadgroup float shared_sum[THREADGROUP_SIZE];

    // Number of blocks per row
    uint blocks_per_row = K / QK4_0;

    // Each thread handles multiple blocks
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        // Get block for this row
        device const block_q4_0* block = &weight[row * blocks_per_row + block_idx];

        float scale = float(block->scale);
        uint k_start = block_idx * QK4_0;

        // Dequantize and compute dot product
        for (uint i = 0; i < QK4_0; i += 2) {
            uint byte_idx = i / 2;
            uint8_t packed = block->quants[byte_idx];

            float w0 = float(extract_q4(packed, 0)) * scale;
            float w1 = float(extract_q4(packed, 1)) * scale;

            sum = fma(w0, input[k_start + i], sum);
            sum = fma(w1, input[k_start + i + 1], sum);
        }
    }

    // Reduction
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
// Q4_1 GEMV Kernel
// =============================================================================

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

// =============================================================================
// Q8_0 GEMV Kernel
// =============================================================================

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
// Batched Quantized GEMV
// =============================================================================

/**
 * @brief Batched Q4_0 GEMV for small batch sizes
 */
kernel void gemv_q4_0_batched(
    device const float* input [[buffer(0)]],          // [B, K]
    device const block_q4_0* weight [[buffer(1)]],    // [M * K/QK4_0]
    device float* output [[buffer(2)]],               // [B, M]
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg_dim [[threads_per_threadgroup]])
{
    uint tg_size = tg_dim.x;
    uint batch_idx = tgid.y;
    uint row = tgid.x;

    if (batch_idx >= batch || row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];

    device const float* input_batch = input + batch_idx * K;
    device float* output_batch = output + batch_idx * M;

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

            sum = fma(w0, input_batch[k_start + i], sum);
            sum = fma(w1, input_batch[k_start + i + 1], sum);
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
        output_batch[row] = shared_sum[0];
    }
}

// =============================================================================
// Mixed Precision GEMM (FP16 accumulation)
// =============================================================================

/**
 * @brief Q4_0 GEMV with FP16 accumulation for better performance
 *
 * Uses half precision for intermediate accumulation, which is faster
 * on Apple GPUs. Final result is converted to FP32.
 */
kernel void gemv_q4_0_fp16(
    device const half* input [[buffer(0)]],           // [K] FP16 input
    device const block_q4_0* weight [[buffer(1)]],
    device half* output [[buffer(2)]],                // [M] FP16 output
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

        half scale_h = block->scale;
        uint k_start = block_idx * QK4_0;

        // Process 4 elements at a time using half4
        for (uint i = 0; i < QK4_0; i += 4) {
            uint byte_idx = i / 2;
            uint8_t packed0 = block->quants[byte_idx];
            uint8_t packed1 = block->quants[byte_idx + 1];

            half4 w;
            w.x = half(extract_q4(packed0, 0)) * scale_h;
            w.y = half(extract_q4(packed0, 1)) * scale_h;
            w.z = half(extract_q4(packed1, 0)) * scale_h;
            w.w = half(extract_q4(packed1, 1)) * scale_h;

            half4 x = half4(input[k_start + i],
                           input[k_start + i + 1],
                           input[k_start + i + 2],
                           input[k_start + i + 3]);

            sum += float(dot(w, x));
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
        output[row] = half(shared_sum[0]);
    }
}

// =============================================================================
// INT4 Group-Wise Dequantization Kernel (for ANE GEMM Fallback)
// =============================================================================
// Dequantizes packed INT4 weights to FP32 using group-wise scales and zeros.
// Matches the CPU dequantization logic in ane_backend.mm:GemmInt4().
//
// Weight layout: [N, K/2] packed - 2 signed INT4 values per byte
// Quantization: w_fp32 = scale * (w_int4_signed - zero)
// INT4 range: Signed [-8, 7] stored as unsigned [0, 15] with implicit offset
//
// Grid: (K/2, N, 1) - each thread handles one packed byte (2 INT4 values)
// =============================================================================

kernel void dequantize_int4_grouped(
    device const uint8_t* weights_packed [[buffer(0)]],   // [N, K/2] packed INT4
    device const float* scales [[buffer(1)]],             // [N, num_groups]
    device const float* zeros [[buffer(2)]],              // [N, num_groups]
    device float* output [[buffer(3)]],                   // [N, K] dequantized FP32
    constant uint& N [[buffer(4)]],                       // Output dimension
    constant uint& K [[buffer(5)]],                       // Input dimension
    constant uint& group_size [[buffer(6)]],              // Quantization group size
    uint2 tid [[thread_position_in_grid]])                // (k_pair_idx, n)
{
    uint k_pair = tid.x;  // Which pair of K values (byte index within row)
    uint n = tid.y;       // Which output row

    uint K_packed = K / 2;
    uint num_groups = K / group_size;

    if (k_pair >= K_packed || n >= N) return;

    // Get packed byte
    uint byte_idx = n * K_packed + k_pair;
    uint8_t packed = weights_packed[byte_idx];

    // Extract two INT4 values
    // Lower nibble: k = k_pair * 2
    // Upper nibble: k = k_pair * 2 + 1
    uint k0 = k_pair * 2;
    uint k1 = k0 + 1;

    // Extract unsigned 4-bit values
    int8_t val0 = int8_t(packed & 0x0F);
    int8_t val1 = int8_t((packed >> 4) & 0x0F);

    // Convert from unsigned 4-bit (0-15) to signed (-8 to 7)
    if (val0 > 7) val0 = val0 - 16;
    if (val1 > 7) val1 = val1 - 16;

    // Get group-wise scale and zero point
    uint group0 = k0 / group_size;
    uint group1 = k1 / group_size;

    float scale0 = scales[n * num_groups + group0];
    float zero0 = zeros[n * num_groups + group0];
    float scale1 = scales[n * num_groups + group1];
    float zero1 = zeros[n * num_groups + group1];

    // Dequantize: w_fp32 = scale * (w_int4 - zero)
    output[n * K + k0] = scale0 * (float(val0) - zero0);
    output[n * K + k1] = scale1 * (float(val1) - zero1);
}

