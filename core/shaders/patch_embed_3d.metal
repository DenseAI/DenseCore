/**
 * @file patch_embed_3d.metal
 * @brief Metal implementation of 3D Patch Embedding for Video Transformers
 *
 * Converts video [B, C, T, H, W] to patch sequence [B, N, D].
 * Used in VideoMAE, SORA, and video diffusion models.
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Im2Col 3D Kernel
// =============================================================================

/**
 * @brief Extract 3D patches into columns for GEMM
 *
 * Converts [B, C, T, H, W] to [B * N_patches, C * Pt * Ph * Pw]
 * where N_patches = (T/Pt) * (H/Ph) * (W/Pw)
 */
kernel void im2col_3d(
    device const float* video [[buffer(0)]],
    device float* columns [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& W [[buffer(6)]],
    constant uint& Pt [[buffer(7)]],   // temporal patch size
    constant uint& Ph [[buffer(8)]],   // height patch size
    constant uint& Pw [[buffer(9)]],   // width patch size
    uint3 gid [[thread_position_in_grid]])
{
    // gid.z = batch
    // gid.y = patch index
    // gid.x = column index within patch (c * Pt * Ph * Pw + ...)
    
    uint batch = gid.z;
    uint patch_idx = gid.y;
    uint col_idx = gid.x;
    
    uint num_patches_t = T / Pt;
    uint num_patches_h = H / Ph;
    uint num_patches_w = W / Pw;
    uint num_patches = num_patches_t * num_patches_h * num_patches_w;
    uint patch_dim = C * Pt * Ph * Pw;
    
    if (patch_idx >= num_patches) return;
    if (col_idx >= patch_dim) return;
    
    // Decode patch position
    uint pt = patch_idx / (num_patches_h * num_patches_w);
    uint temp = patch_idx % (num_patches_h * num_patches_w);
    uint ph = temp / num_patches_w;
    uint pw = temp % num_patches_w;
    
    // Decode column position within patch
    uint c = col_idx / (Pt * Ph * Pw);
    uint rem = col_idx % (Pt * Ph * Pw);
    uint kt = rem / (Ph * Pw);
    rem = rem % (Ph * Pw);
    uint kh = rem / Pw;
    uint kw = rem % Pw;
    
    // Source position in video
    uint src_t = pt * Pt + kt;
    uint src_h = ph * Ph + kh;
    uint src_w = pw * Pw + kw;
    
    uint src_idx = batch * C * T * H * W +
                   c * T * H * W +
                   src_t * H * W +
                   src_h * W +
                   src_w;
    
    uint dst_idx = (batch * num_patches + patch_idx) * patch_dim + col_idx;
    
    columns[dst_idx] = video[src_idx];
}

// =============================================================================
// Patch Embed 3D with Fused GEMM
// =============================================================================

/**
 * @brief 3D Patch embedding with projection
 *
 * Combines Im2Col + GEMM projection.
 * Output = Im2Col(video) @ weights + bias
 */
kernel void patch_embed_3d(
    device const float* video [[buffer(0)]],
    device const float* weights [[buffer(1)]],  // [embed_dim, C * Pt * Ph * Pw]
    device const float* bias [[buffer(2)]],     // [embed_dim] or nullptr
    device float* output [[buffer(3)]],         // [B, N_patches, embed_dim]
    constant uint& B [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& W [[buffer(8)]],
    constant uint& Pt [[buffer(9)]],
    constant uint& Ph [[buffer(10)]],
    constant uint& Pw [[buffer(11)]],
    constant uint& embed_dim [[buffer(12)]],
    constant uint& use_bias [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint patch_idx = gid.y;
    uint embed_idx = gid.x;
    
    uint num_patches_t = T / Pt;
    uint num_patches_h = H / Ph;
    uint num_patches_w = W / Pw;
    uint num_patches = num_patches_t * num_patches_h * num_patches_w;
    uint patch_dim = C * Pt * Ph * Pw;
    
    if (patch_idx >= num_patches) return;
    if (embed_idx >= embed_dim) return;
    
    // Decode patch position
    uint pt = patch_idx / (num_patches_h * num_patches_w);
    uint temp = patch_idx % (num_patches_h * num_patches_w);
    uint ph = temp / num_patches_w;
    uint pw = temp % num_patches_w;
    
    // Compute output: dot product of patch with weight row
    float sum = 0.0f;
    
    for (uint c = 0; c < C; ++c) {
        for (uint kt = 0; kt < Pt; ++kt) {
            for (uint kh = 0; kh < Ph; ++kh) {
                for (uint kw = 0; kw < Pw; ++kw) {
                    uint src_t = pt * Pt + kt;
                    uint src_h = ph * Ph + kh;
                    uint src_w = pw * Pw + kw;
                    
                    uint video_idx = batch * C * T * H * W +
                                    c * T * H * W +
                                    src_t * H * W +
                                    src_h * W +
                                    src_w;
                    
                    uint weight_col = c * Pt * Ph * Pw + kt * Ph * Pw + kh * Pw + kw;
                    uint weight_idx = embed_idx * patch_dim + weight_col;
                    
                    sum = fma(video[video_idx], weights[weight_idx], sum);
                }
            }
        }
    }
    
    if (use_bias) {
        sum += bias[embed_idx];
    }
    
    uint out_idx = batch * num_patches * embed_dim + patch_idx * embed_dim + embed_idx;
    output[out_idx] = sum;
}

// =============================================================================
// Patchify Kernel (no projection)
// =============================================================================

/**
 * @brief Convert image to patches without projection
 *
 * [B, C, H, W] -> [B, N, P*P*C]
 */
kernel void patchify_2d(
    device const float* image [[buffer(0)]],
    device float* patches [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& W [[buffer(5)]],
    constant uint& P [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint patch_idx = gid.y;
    uint col_idx = gid.x;
    
    uint num_patches_h = H / P;
    uint num_patches_w = W / P;
    uint num_patches = num_patches_h * num_patches_w;
    uint patch_dim = C * P * P;
    
    if (patch_idx >= num_patches) return;
    if (col_idx >= patch_dim) return;
    
    uint ph = patch_idx / num_patches_w;
    uint pw = patch_idx % num_patches_w;
    
    // Decode position within patch
    uint c = col_idx / (P * P);
    uint rem = col_idx % (P * P);
    uint kh = rem / P;
    uint kw = rem % P;
    
    uint src_h = ph * P + kh;
    uint src_w = pw * P + kw;
    
    uint src_idx = batch * C * H * W + c * H * W + src_h * W + src_w;
    uint dst_idx = batch * num_patches * patch_dim + patch_idx * patch_dim + col_idx;
    
    patches[dst_idx] = image[src_idx];
}

// =============================================================================
// Unpatchify Kernel
// =============================================================================

/**
 * @brief Convert patches back to image
 *
 * [B, N, P*P*C] -> [B, C, H, W]
 */
kernel void unpatchify_2d(
    device const float* patches [[buffer(0)]],
    device float* image [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& W [[buffer(5)]],
    constant uint& P [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint hw = gid.y;
    uint c = gid.x;
    
    if (hw >= H * W) return;
    if (c >= C) return;
    
    uint h = hw / W;
    uint w = hw % W;
    
    uint num_patches_h = H / P;
    uint num_patches_w = W / P;
    uint patch_dim = C * P * P;
    
    uint ph = h / P;
    uint pw = w / P;
    uint patch_idx = ph * num_patches_w + pw;
    
    uint kh = h % P;
    uint kw = w % P;
    uint col_idx = c * P * P + kh * P + kw;
    
    uint src_idx = batch * num_patches_h * num_patches_w * patch_dim + 
                   patch_idx * patch_dim + col_idx;
    uint dst_idx = batch * C * H * W + c * H * W + h * W + w;
    
    image[dst_idx] = patches[src_idx];
}
