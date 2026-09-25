/**
 * @file triangular_attention.metal
 * @brief Triangular attention (AlphaFold) Metal kernel
 */

#include <metal_stdlib>
using namespace metal;

constant float NEG_INF = -1e9f;
constant uint MAX_HEAD_DIM = 128;

kernel void triangular_attention_forward(
    device const float* pair_repr [[buffer(0)]],   // [B, L, L, D]
    device const float* query_w [[buffer(1)]],     // [H, D_head, D]
    device const float* key_w [[buffer(2)]],       // [H, D_head, D]
    device const float* value_w [[buffer(3)]],     // [H, D_head, D]
    device float* output [[buffer(4)]],            // [B, L, L, D]
    constant uint& B [[buffer(5)]],
    constant uint& L [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& H [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& starting [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint j = gid.x;
    uint i = gid.y;
    uint hz = gid.z;

    if (i >= L || j >= L || H == 0) {
        return;
    }

    uint h = hz % H;
    uint b = hz / H;
    if (b >= B) {
        return;
    }

    uint D_head = head_dim;
    if (D_head == 0 || D_head > MAX_HEAD_DIM) {
        return;
    }

    uint pair_base = ((b * L + i) * L + j) * D;
    device const float* pair_ij = pair_repr + pair_base;

    thread float q_proj[MAX_HEAD_DIM];
    thread float proj_tmp[MAX_HEAD_DIM];
    thread float acc[MAX_HEAD_DIM];

    for (uint d = 0; d < D_head; ++d) {
        float sum = 0.0f;
        uint w_base = (h * D * D_head) + d * D;
        for (uint t = 0; t < D; ++t) {
            sum += pair_ij[t] * query_w[w_base + t];
        }
        q_proj[d] = sum;
        acc[d] = 0.0f;
    }

    float max_score = NEG_INF;
    float sum_exp = 0.0f;

    for (uint k = 0; k < L; ++k) {
        uint pair_k_base = starting != 0
            ? ((b * L + i) * L + k) * D
            : ((b * L + k) * L + j) * D;
        device const float* pair_k = pair_repr + pair_k_base;

        for (uint d = 0; d < D_head; ++d) {
            float sum = 0.0f;
            uint w_base = (h * D * D_head) + d * D;
            for (uint t = 0; t < D; ++t) {
                sum += pair_k[t] * key_w[w_base + t];
            }
            proj_tmp[d] = sum;
        }

        float dot = 0.0f;
        for (uint d = 0; d < D_head; ++d) {
            dot += q_proj[d] * proj_tmp[d];
        }
        float score = dot * scale;

        if (score > max_score) {
            float scale_down = exp(max_score - score);
            for (uint d = 0; d < D_head; ++d) {
                acc[d] *= scale_down;
            }
            sum_exp *= scale_down;
            max_score = score;
        }

        float w = exp(score - max_score);
        sum_exp += w;

        for (uint d = 0; d < D_head; ++d) {
            float sum = 0.0f;
            uint w_base = (h * D * D_head) + d * D;
            for (uint t = 0; t < D; ++t) {
                sum += pair_k[t] * value_w[w_base + t];
            }
            proj_tmp[d] = sum;
        }

        for (uint d = 0; d < D_head; ++d) {
            acc[d] += w * proj_tmp[d];
        }
    }

    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    uint out_base = pair_base + h * D_head;
    for (uint d = 0; d < D_head; ++d) {
        output[out_base + d] = pair_ij[h * D_head + d] + acc[d] * inv_sum;
    }
}
