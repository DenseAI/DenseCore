/**
 * @file temporal_attention.metal
 * @brief Metal implementation of Temporal Attention for Video Transformers
 *
 * Frame-to-frame attention for SORA, VideoMAE, and video diffusion models.
 * Supports causal masking and sliding window attention.
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

constant uint BLOCK_SIZE = 16;
constant uint MAX_HEAD_DIM = 128;
constant uint MAX_FRAMES = 256;
constant float NEG_INF = -1e9f;

// =============================================================================
// Temporal Attention Kernel
// =============================================================================

/**
 * @brief Attention across time dimension for video
 *
 * For each spatial position, performs attention across all frames.
 * Input layout: [B, T, N, H, D] where:
 *   B = batch, T = frames, N = spatial tokens, H = heads, D = head_dim
 *
 * Flattened as: [B*N, T, H*D] for attention along T dimension.
 *
 * @param Q Query [B*N, T, H*D]
 * @param K Key [B*N, T, H*D]  
 * @param V Value [B*N, T, H*D]
 * @param output Output [B*N, T, H*D]
 * @param scale Attention scale
 * @param T Number of frames
 * @param head_dim Head dimension
 * @param num_heads Number of heads
 * @param causal Whether to apply causal mask
 * @param window_size Sliding window size (0 = full attention)
 */
kernel void temporal_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& causal [[buffer(8)]],
    constant uint& window_size [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint batch_spatial = tgid.z;  // B * N combined
    uint head = tgid.y;
    uint query_frame = tgid.x;
    
    if (query_frame >= T) return;
    
    const uint D = num_heads * head_dim;
    const uint head_offset = head * head_dim;
    
    // Pointers
    device const float* Q_ptr = Q + batch_spatial * T * D + query_frame * D + head_offset;
    device const float* K_base = K + batch_spatial * T * D;
    device const float* V_base = V + batch_spatial * T * D;
    device float* O_ptr = output + batch_spatial * T * D + query_frame * D + head_offset;
    
    // Load query
    float q[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) {
        q[d] = Q_ptr[d];
    }
    
    // Compute attention scores
    float max_score = NEG_INF;
    float scores[MAX_FRAMES];  // Max frames supported
    
    uint k_start = 0;
    uint k_end = T;
    
    // Sliding window boundaries
    if (window_size > 0 && window_size < T) {
        k_start = (query_frame >= window_size) ? query_frame - window_size + 1 : 0;
        k_end = min(query_frame + window_size, T);
    }
    
    // Guard against T > MAX_FRAMES: Clamp attention to support buffer size
    // This effectively ignores frames beyond MAX_FRAMES index.
    if (k_end > MAX_FRAMES) k_end = MAX_FRAMES;
    if (k_start > k_end) k_start = k_end; // Should imply empty loop

    for (uint k = k_start; k < k_end; ++k) {
        // Causal mask
        if (causal && k > query_frame) {
            scores[k] = NEG_INF;
            continue;
        }
        
        device const float* K_k = K_base + k * D + head_offset;
        
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot = fma(q[d], K_k[d], dot);
        }
        
        scores[k] = dot * scale;
        max_score = max(max_score, scores[k]);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (uint k = k_start; k < k_end; ++k) {
        if (causal && k > query_frame) continue;
        scores[k] = exp(scores[k] - max_score);
        sum_exp += scores[k];
    }
    
    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    
    // Weighted sum of values
    float out[MAX_HEAD_DIM] = {0.0f};
    
    for (uint k = k_start; k < k_end; ++k) {
        if (causal && k > query_frame) continue;
        
        float attn = scores[k] * inv_sum;
        device const float* V_k = V_base + k * D + head_offset;
        
        for (uint d = 0; d < head_dim; ++d) {
            out[d] = fma(attn, V_k[d], out[d]);
        }
    }
    
    // Write output
    for (uint d = 0; d < head_dim; ++d) {
        O_ptr[d] = out[d];
    }
}

// =============================================================================
// Temporal RoPE Kernel
// =============================================================================

/**
 * @brief Apply temporal + spatial RoPE encoding
 *
 * 3D position encoding for video transformers.
 * Splits head_dim into temporal and spatial components.
 */
kernel void temporal_rope(
    device float* x [[buffer(0)]],           // [B, T, N, H, D] in-place
    device const float* temporal_pos [[buffer(1)]],  // [T]
    device const float* spatial_pos [[buffer(2)]],   // [N, 2] (h, w)
    constant uint& T [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& temporal_dim [[buffer(7)]],  // Dims for temporal
    constant uint& spatial_dim [[buffer(8)]],   // Dims for spatial
    constant float& theta [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint tn = gid.y;  // t * N + n
    uint hd = gid.x;  // h * head_dim + d
    
    if (tn >= T * N) return;
    if (hd >= num_heads * head_dim) return;
    
    uint t = tn / N;
    uint n = tn % N;
    uint h = hd / head_dim;
    uint d = hd % head_dim;
    
    // Only apply RoPE to even dimensions (pairs)
    if (d % 2 != 0) return;
    
    uint idx = batch * T * N * num_heads * head_dim + 
               tn * num_heads * head_dim + hd;
    
    float x0 = x[idx];
    float x1 = x[idx + 1];
    
    float angle = 0.0f;
    uint half_d = d / 2;
    
    if (d < temporal_dim) {
        // Temporal encoding
        float freq = 1.0f / pow(theta, 2.0f * float(half_d) / float(temporal_dim));
        angle = temporal_pos[t] * freq;
    } else if (d < temporal_dim + spatial_dim) {
        // Spatial encoding
        uint sp_d = (d - temporal_dim) / 2;
        float freq = 1.0f / pow(theta, 2.0f * float(sp_d) / float(spatial_dim));
        
        if ((d - temporal_dim) < spatial_dim / 2) {
            // Height dimension
            angle = spatial_pos[n * 2 + 0] * freq;
        } else {
            // Width dimension
            angle = spatial_pos[n * 2 + 1] * freq;
        }
    }
    
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    
    x[idx] = x0 * cos_a - x1 * sin_a;
    x[idx + 1] = x0 * sin_a + x1 * cos_a;
}
