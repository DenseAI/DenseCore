/**
 * @file flash_attention.metal
 * @brief Memory-efficient FlashAttention implementation for Metal
 *
 * FlashAttention Algorithm:
 * Standard attention: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Problem: Materializing the N x N attention matrix requires O(N^2) memory.
 * For long sequences (e.g., 128K context), this becomes prohibitive.
 *
 * FlashAttention Solution:
 * 1. Tile Q, K, V into blocks that fit in SRAM (shared memory)
 * 2. Compute attention scores block-by-block
 * 3. Use online softmax with running max/sum
 * 4. Accumulate output incrementally
 *
 * Memory Complexity: O(N) instead of O(N^2)
 * SRAM Usage: ~100KB per threadgroup (tunable)
 *
 * Apple Silicon Considerations:
 * - M1/M2/M3 have 32KB threadgroup memory
 * - Use smaller tile sizes than NVIDIA GPUs
 * - Leverage simdgroup operations for reductions
 *
 * Tile Sizes (tuned for Apple Silicon):
 * - BLOCK_Q = 16 (queries per tile)
 * - BLOCK_K = 16 (keys per tile)
 * - HEAD_DIM = 64/128 (typical head dimensions)
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

constant uint BLOCK_Q = 16;          // Queries per block
constant uint BLOCK_K = 16;          // Keys per block
constant float NEG_INF = -1e9f;      // For masking

// =============================================================================
// Constants for Array Sizing
// =============================================================================
// MAX_HEAD_DIM must be a compile-time literal in this shader because Metal requires
// fixed-size threadgroup/thread arrays to use constant expressions during AIR compilation.
//
// Common head dimensions:
// - GPT-2/3: 64
// - Llama/Llama2: 128  
// - Llama3/future models: up to 256
// =============================================================================
constant uint MAX_HEAD_DIM = 128;
constant uint DECODE_THREADS = 64;

// =============================================================================
// Helper Structures
// =============================================================================

/**
 * @brief Online softmax state for incremental computation
 */
struct OnlineSoftmax {
  float max_val;      // Running maximum
  float sum_exp;      // Sum of exp(x - max)

  // Initialize
  static OnlineSoftmax init() {
    OnlineSoftmax s;
    s.max_val = NEG_INF;
    s.sum_exp = 0.0f;
    return s;
  }

  // Update with new value
  void update(float x) {
    float old_max = max_val;
    max_val = max(max_val, x);
    sum_exp = sum_exp * exp(old_max - max_val) + exp(x - max_val);
  }

  // Merge two states
  void merge(OnlineSoftmax other) {
    float new_max = max(max_val, other.max_val);
    sum_exp = sum_exp * exp(max_val - new_max) +
              other.sum_exp * exp(other.max_val - new_max);
    max_val = new_max;
  }
};

// =============================================================================
// FlashAttention Kernel - Forward Pass
// =============================================================================

/**
 * @brief Memory-efficient attention computation
 *
 * Computes: output = softmax(Q @ K^T / scale) @ V with O(N) memory
 *
 * Each threadgroup processes one query block (BLOCK_Q queries).
 * It iterates over all key blocks, computing partial attention
 * and accumulating the output.
 *
 * Memory layout (all row-major):
 * - Q: [batch, n_heads, seq_q, head_dim]
 * - K: [batch, n_kv_heads, seq_kv, head_dim]
 * - V: [batch, n_kv_heads, seq_kv, head_dim]
 * - output: [batch, n_heads, seq_q, head_dim]
 *
 * @param Q Query tensor
 * @param K Key tensor
 * @param V Value tensor
 * @param output Output tensor
 * @param scale Attention scale (1/sqrt(head_dim))
 * @param seq_q Query sequence length
 * @param seq_kv Key/Value sequence length
 * @param head_dim Head dimension
 * @param n_heads Number of query heads
 * @param n_kv_heads Number of KV heads (for GQA)
 * @param causal Whether to apply causal masking
 */
kernel void flash_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& n_heads [[buffer(8)]],
    constant uint& n_kv_heads [[buffer(9)]],
    constant uint& causal [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (head_dim == 0 || head_dim > MAX_HEAD_DIM) return;

    // ==========================================================================
    // Setup indices
    // ==========================================================================

    // Each threadgroup handles one (batch, head, query_block)
    uint batch_idx = tgid.z;
    uint head_idx = tgid.y;
    uint query_block_idx = tgid.x;

    // GQA: map query head to KV head
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);

    // Starting query position for this block
    uint q_start = query_block_idx * BLOCK_Q;
    if (q_start >= seq_q) return;

    // Number of queries in this block (handle boundary)
    uint q_count = min(BLOCK_Q, seq_q - q_start);

    // Thread's query index within the block
    uint local_q = tid.x;
    uint global_q = q_start + local_q;

    // ==========================================================================
    // Shared memory allocation
    // ==========================================================================

    // Shared memory for K and V tiles
    threadgroup float shared_K[BLOCK_K][MAX_HEAD_DIM];
    threadgroup float shared_V[BLOCK_K][MAX_HEAD_DIM];

    // Per-query state
    threadgroup float shared_max[BLOCK_Q];
    threadgroup float shared_sum[BLOCK_Q];
    threadgroup float shared_output[BLOCK_Q][MAX_HEAD_DIM];

    // ==========================================================================
    // Initialize output accumulator
    // ==========================================================================

    // Each thread initializes its query's output and softmax state
    if (local_q < q_count) {
        shared_max[local_q] = NEG_INF;
        shared_sum[local_q] = 0.0f;
        for (uint d = tid.y; d < head_dim; d += BLOCK_K) {
            shared_output[local_q][d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==========================================================================
    // Compute Q row for this thread
    // ==========================================================================

    // Pointers to Q, K, V for this batch and head
    device const float* Q_head = Q + (batch_idx * n_heads + head_idx) * seq_q * head_dim;
    device const float* K_head = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_head = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_head = output + (batch_idx * n_heads + head_idx) * seq_q * head_dim;

    // Load Q row for this thread's query
    float q_row[MAX_HEAD_DIM];
    if (local_q < q_count) {
        for (uint d = 0; d < head_dim; ++d) {
            q_row[d] = Q_head[global_q * head_dim + d];
        }
    }

    // ==========================================================================
    // Iterate over K/V blocks
    // ==========================================================================

    uint num_kv_blocks = (seq_kv + BLOCK_K - 1) / BLOCK_K;

    for (uint kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        uint k_start = kv_block * BLOCK_K;
        uint k_count = min(BLOCK_K, seq_kv - k_start);

        // ----------------------------------------------------------------------
        // Load K and V tiles into shared memory
        // ----------------------------------------------------------------------

        // Cooperative loading: each thread loads part of K and V
        for (uint k_offset = tid.x; k_offset < k_count; k_offset += BLOCK_Q) {
            for (uint d = tid.y; d < head_dim; d += BLOCK_K) {
                uint k_idx = k_start + k_offset;
                shared_K[k_offset][d] = K_head[k_idx * head_dim + d];
                shared_V[k_offset][d] = V_head[k_idx * head_dim + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ----------------------------------------------------------------------
        // Compute attention scores for this block
        // ----------------------------------------------------------------------

        if (local_q < q_count) {
            for (uint k_offset = 0; k_offset < k_count; ++k_offset) {
                uint global_k = k_start + k_offset;

                // Causal mask
                if (causal && global_k > global_q) {
                    continue;
                }

                // Compute dot product: Q[q] @ K[k]^T
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; ++d) {
                    dot = fma(q_row[d], shared_K[k_offset][d], dot);
                }
                float score = dot * scale;

                // Online softmax update
                float old_max = shared_max[local_q];
                float new_max = max(old_max, score);
                float scale_old = exp(old_max - new_max);
                float exp_score = exp(score - new_max);

                // Rescale running sum and add new term
                float new_sum = shared_sum[local_q] * scale_old + exp_score;

                // Rescale output accumulator and add V contribution
                float v_weight = exp_score / new_sum;
                float old_weight = (shared_sum[local_q] * scale_old) / new_sum;

                for (uint d = 0; d < head_dim; ++d) {
                    shared_output[local_q][d] = shared_output[local_q][d] * old_weight +
                                                 shared_V[k_offset][d] * v_weight;
                }

                // Update state
                shared_max[local_q] = new_max;
                shared_sum[local_q] = new_sum;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ==========================================================================
    // Write output
    // ==========================================================================

    if (local_q < q_count) {
        for (uint d = tid.y; d < head_dim; d += BLOCK_K) {
            O_head[global_q * head_dim + d] = shared_output[local_q][d];
        }
    }
}

// =============================================================================
// Simplified FlashAttention for single query (decode phase)
// =============================================================================

/**
 * @brief Optimized attention for decode phase (single query)
 *
 * For decode, seq_q = 1, so we can simplify the algorithm.
 * One threadgroup handles one (batch, head) pair.
 */
kernel void flash_attention_decode(
    device const float* Q [[buffer(0)]],      // [batch, n_heads, 1, head_dim]
    device const float* K [[buffer(1)]],      // [batch, n_kv_heads, seq_kv, head_dim]
    device const float* V [[buffer(2)]],      // [batch, n_kv_heads, seq_kv, head_dim]
    device float* output [[buffer(3)]],       // [batch, n_heads, 1, head_dim]
    constant float& scale [[buffer(4)]],
    constant uint& seq_kv [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_dim [[threads_per_threadgroup]])
{
    uint tg_size = tg_dim.x;
    if (tg_size > DECODE_THREADS || head_dim == 0 || head_dim > MAX_HEAD_DIM) return;

    uint batch_idx = tgid.z;
    uint head_idx = tgid.x;
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);

    // Pointers
    device const float* Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
    device const float* K_head = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_head = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_ptr = output + (batch_idx * n_heads + head_idx) * head_dim;

    // Load query into registers
    float q[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) {
        q[d] = Q_ptr[d];
    }

    // Shared memory for reduction
    // Note: For output reduction, we store per-thread scaled outputs then reduce
    threadgroup float shared_max[DECODE_THREADS];
    threadgroup float shared_sum[DECODE_THREADS];
    threadgroup float shared_output[DECODE_THREADS][MAX_HEAD_DIM];  // Per-thread output for reduction

    // Each thread handles a subset of keys
    float local_max = NEG_INF;
    float local_sum = 0.0f;
    float local_output[MAX_HEAD_DIM] = {0.0f};

    for (uint k = tid; k < seq_kv; k += tg_size) {
        // Compute Q @ K^T
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot = fma(q[d], K_head[k * head_dim + d], dot);
        }
        float score = dot * scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);
        float scale_factor = exp(old_max - local_max);
        float exp_score = exp(score - local_max);

        // Rescale and accumulate
        local_sum = local_sum * scale_factor + exp_score;
        for (uint d = 0; d < head_dim; ++d) {
            local_output[d] = local_output[d] * scale_factor +
                              V_head[k * head_dim + d] * exp_score;
        }
    }

    // Store local results
    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction across threads
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

    // Normalize each thread's local output and store in shared memory
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    float my_scale = exp(local_max - global_max) / global_sum;

    // Each thread stores its scaled local_output to its slot in shared memory
    for (uint d = 0; d < head_dim; ++d) {
        shared_output[tid][d] = local_output[d] * my_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===========================================================================
    // Parallel reduction of local_output across all threads
    // ===========================================================================
    // Tree reduction: sum shared_output[tid][d] across all active threads
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            for (uint d = 0; d < head_dim; ++d) {
                shared_output[tid][d] += shared_output[tid + stride][d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final output (result accumulated in shared_output[0] after reduction)
    if (tid == 0) {
        for (uint d = 0; d < head_dim; ++d) {
            O_ptr[d] = shared_output[0][d];
        }
    }
}

// =============================================================================
// Note on Grouped Query Attention (GQA)
// =============================================================================
// GQA is natively supported in flash_attention_forward and flash_attention_decode
// via the n_kv_heads parameter. The kernel computes:
//   kv_head_idx = head_idx / (n_heads / n_kv_heads)
// to map multiple query heads to the same KV head.
// No separate kernel is needed.
// =============================================================================
