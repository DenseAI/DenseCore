/**
 * @file window_attention.metal
 * @brief Metal implementation of Window Attention for Vision Transformers
 *
 * Optimized for Apple Silicon (M1/M2/M3).
 * Window Attention reduces O(N²) to O(W² × N/W²) by attention within windows.
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

constant uint MAX_WINDOW_SIZE = 14;      // Max window size supported
constant uint MAX_HEAD_DIM = 128;        // Max head dimension
constant float NEG_INF = -1e9f;

// =============================================================================
// Window Attention Kernel - No Shift
// =============================================================================

/**
 * @brief Window attention without cyclic shift
 *
 * Each threadgroup processes one window.
 * Memory layout: [B, H, W, C] where C = num_heads * head_dim
 *
 * @param Q Query tensor [B, num_windows, window_area, C]
 * @param K Key tensor [B, num_windows, window_area, C]
 * @param V Value tensor [B, num_windows, window_area, C]
 * @param output Output tensor [B, num_windows, window_area, C]
 * @param relative_bias [num_heads, window_area, window_area] - optional
 * @param scale Attention scale
 * @param window_area window_size * window_size
 * @param head_dim Dimension per head
 * @param num_heads Number of attention heads
 */
kernel void window_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* relative_bias [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& window_area [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& use_bias [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_dim [[threads_per_threadgroup]])
{
    // Indices
    uint batch_window = tgid.z;  // Combined batch and window index
    uint head = tgid.y;
    uint query_idx = tgid.x;     // Query position within window
    
    if (query_idx >= window_area) return;
    
    const uint C = num_heads * head_dim;
    const uint head_offset = head * head_dim;
    
    // Input/output pointers for this window
    device const float* Q_win = Q + batch_window * window_area * C + query_idx * C + head_offset;
    device const float* K_win = K + batch_window * window_area * C;
    device const float* V_win = V + batch_window * window_area * C;
    device float* O_win = output + batch_window * window_area * C + query_idx * C + head_offset;
    
    // Optional bias for this head
    device const float* bias_ptr = nullptr;
    if (use_bias) {
        bias_ptr = relative_bias + head * window_area * window_area + query_idx * window_area;
    }
    
    // Load query into registers
    float q[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) {
        q[d] = Q_win[d];
    }
    
    // Shared memory for softmax and output accumulation
    threadgroup float shared_max[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    threadgroup float shared_sum[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    
    // Compute attention scores
    float max_score = NEG_INF;
    float attn_scores[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    
    for (uint k = 0; k < window_area; ++k) {
        device const float* K_k = K_win + k * C + head_offset;
        
        // Dot product
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot = fma(q[d], K_k[d], dot);
        }
        
        float score = dot * scale;
        
        // Add relative position bias if present
        if (use_bias) {
            score += bias_ptr[k];
        }
        
        attn_scores[k] = score;
        max_score = max(max_score, score);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (uint k = 0; k < window_area; ++k) {
        attn_scores[k] = exp(attn_scores[k] - max_score);
        sum_exp += attn_scores[k];
    }
    
    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    
    // Compute output: weighted sum of values
    float out[MAX_HEAD_DIM] = {0.0f};
    
    for (uint k = 0; k < window_area; ++k) {
        float attn = attn_scores[k] * inv_sum;
        device const float* V_k = V_win + k * C + head_offset;
        
        for (uint d = 0; d < head_dim; ++d) {
            out[d] = fma(attn, V_k[d], out[d]);
        }
    }
    
    // Write output
    for (uint d = 0; d < head_dim; ++d) {
        O_win[d] = out[d];
    }
}

// =============================================================================
// Window Partition Kernel
// =============================================================================

/**
 * @brief Partition image into windows
 *
 * [B, H, W, C] -> [B * num_windows, window_size, window_size, C]
 */
kernel void window_partition(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& W [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& window_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint pos = gid.y;  // position within window (wh * window_size + ww)
    uint channel = gid.x;
    
    if (channel >= C) return;
    
    uint wh = pos / window_size;
    uint ww = pos % window_size;
    
    uint num_windows_w = W / window_size;
    uint num_windows_h = H / window_size;
    uint num_windows = num_windows_h * num_windows_w;
    
    // Output: [B * num_windows, window_size * window_size, C]
    // We need to figure out which window and which position
    // This kernel is called for each (batch, window_idx, pos_in_window, channel)
    
    for (uint window_idx = 0; window_idx < num_windows; ++window_idx) {
        uint window_h = window_idx / num_windows_w;
        uint window_w = window_idx % num_windows_w;
        
        uint src_h = window_h * window_size + wh;
        uint src_w = window_w * window_size + ww;
        
        uint src_idx = batch * H * W * C + src_h * W * C + src_w * C + channel;
        uint dst_idx = (batch * num_windows + window_idx) * (window_size * window_size) * C +
                       pos * C + channel;
        
        output[dst_idx] = input[src_idx];
    }
}

// =============================================================================
// Window Reverse Kernel
// =============================================================================

/**
 * @brief Merge windows back into image
 *
 * [B * num_windows, window_size, window_size, C] -> [B, H, W, C]
 */
kernel void window_reverse(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& W [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& window_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint hw = gid.y;  // h * W + w
    uint channel = gid.x;
    
    if (channel >= C) return;
    if (hw >= H * W) return;
    
    uint h = hw / W;
    uint w = hw % W;
    
    uint num_windows_w = W / window_size;
    uint num_windows_h = H / window_size;
    
    uint window_h = h / window_size;
    uint window_w = w / window_size;
    uint window_idx = window_h * num_windows_w + window_w;
    
    uint local_h = h % window_size;
    uint local_w = w % window_size;
    uint local_pos = local_h * window_size + local_w;
    
    uint src_idx = (batch * num_windows_h * num_windows_w + window_idx) * 
                   (window_size * window_size) * C + local_pos * C + channel;
    uint dst_idx = batch * H * W * C + hw * C + channel;
    
    output[dst_idx] = input[src_idx];
}

// =============================================================================
// Cyclic Shift Kernel
// =============================================================================

/**
 * @brief Cyclic shift for Swin-style cross-window attention
 */
kernel void cyclic_shift(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& W [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant int& shift_h [[buffer(6)]],
    constant int& shift_w [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint hw = gid.y;
    uint channel = gid.x;
    
    if (channel >= C) return;
    if (hw >= H * W) return;
    
    uint h = hw / W;
    uint w = hw % W;
    
    // Source position with cyclic wrap
    int src_h = ((int)h + shift_h + (int)H) % (int)H;
    int src_w = ((int)w + shift_w + (int)W) % (int)W;
    
    uint src_idx = batch * H * W * C + (uint)src_h * W * C + (uint)src_w * C + channel;
    uint dst_idx = batch * H * W * C + h * W * C + w * C + channel;
    
    output[dst_idx] = input[src_idx];
}
