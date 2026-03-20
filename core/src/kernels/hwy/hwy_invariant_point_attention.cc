/**
 * @file hwy_invariant_point_attention.cc
 * @brief Invariant Point Attention via Google Highway
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_invariant_point_attention.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// FastExp inline
#include "kernels/hwy/hwy_fastexp.h"

// ============================================================================
// Helper Functions
// ============================================================================

// Apply rotation R [3,3] to points P [N, 3] + translation T [3] -> Out [N, 3]
void ApplyRigidTransformHwy(const float* R, const float* T, const float* P, float* Out, int64_t N_points) {
    // Apply rigid transformation: Out = R * P + T
    // R: [3x3] rotation matrix (row-major)
    // P: [N, 3] input points
    // T: [3] translation vector

    const float r00 = R[0], r01 = R[1], r02 = R[2];
    const float r10 = R[3], r11 = R[4], r12 = R[5];
    const float r20 = R[6], r21 = R[7], r22 = R[8];
    const float t0 = T[0], t1 = T[1], t2 = T[2];

    for (int64_t i = 0; i < N_points; ++i) {
        float px = P[i * 3 + 0];
        float py = P[i * 3 + 1];
        float pz = P[i * 3 + 2];

        Out[i * 3 + 0] = (r00 * px + r01 * py + r02 * pz) + t0;
        Out[i * 3 + 1] = (r10 * px + r11 * py + r12 * pz) + t1;
        Out[i * 3 + 2] = (r20 * px + r21 * py + r22 * pz) + t2;
    }
}

// ============================================================================
// IPA Implementation
// ============================================================================

void InvariantPointAttentionImpl(const float* s_data, const float* pair_data, const float* R_data, const float* t_data,
                                 const float* qp_data, const float* vp_data, float* out_data, void* workspace,
                                 size_t workspace_size, int64_t B, int64_t L, int64_t D, int64_t D_pair, int num_heads,
                                 int num_query_points, int num_value_points, float scale) {
    if (num_heads <= 0) return;
    const hn::ScalableTag<float> d;
    auto N = hn::Lanes(d);

    const int64_t D_head = D / num_heads;
    const int64_t Q = num_query_points;
    const int64_t V = num_value_points;

    // Workspace Requirements:
    // global_q_pts: Q * 3
    // global_k_pts: Q * 3
    // attn_scores: L
    size_t required_bytes = (2 * Q * 3 + L) * sizeof(float);
    if (workspace_size < required_bytes) return;

    float* ws_ptr = reinterpret_cast<float*>(workspace);
    float* global_q_pts = ws_ptr;
    ws_ptr += Q * 3;
    float* global_k_pts = ws_ptr;
    ws_ptr += Q * 3;
    float* attn_scores = ws_ptr;
    // ws_ptr += L;

    // Initialize output with s_data (residual connection usually handled outside, but original code did memcpy)
    std::memcpy(out_data, s_data, B * L * D * sizeof(float));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < L; ++i) {
            const float* R_i = R_data + b * L * 9 + i * 9;
            const float* t_i = t_data + b * L * 3 + i * 3;
            const float* s_i = s_data + b * L * D + i * D;
            float* out_i = out_data + b * L * D + i * D;

            for (int64_t h = 0; h < num_heads; ++h) {
                const int64_t h_off = h * D_head;

                // 1. Transform Query Points (local frame i -> global)
                // qp_data is [H, Q, 3]
                ApplyRigidTransformHwy(R_i, t_i, qp_data + h * Q * 3, global_q_pts, Q);

                // 2. Compute Attention Scores
                float max_score = -1e30f;
                // Pre-load scalar query part
                // s_i is [D]
                // we need s_i[h_off : h_off+D_head]

                for (int64_t j = 0; j < L; ++j) {
                    const float* R_j = R_data + b * L * 9 + j * 9;
                    const float* t_j = t_data + b * L * 3 + j * 3;
                    const float* s_j = s_data + b * L * D + j * D;
                    const float* pair_ij = pair_data + b * L * L * D_pair + i * L * D_pair + j * D_pair;

                    // Transform Key Points (local frame j -> global)
                    ApplyRigidTransformHwy(R_j, t_j, qp_data + h * Q * 3, global_k_pts, Q);

                    // A. Scalar Attn: dot(s_i, s_j)
                    auto v_sum_scalar = hn::Zero(d);
                    size_t k = 0;
                    for (; k + N <= static_cast<size_t>(D_head); k += N) {
                        v_sum_scalar =
                            hn::MulAdd(hn::LoadU(d, s_i + h_off + k), hn::LoadU(d, s_j + h_off + k), v_sum_scalar);
                    }
                    float scalar_attn = hn::ReduceSum(d, v_sum_scalar);
                    for (; k < static_cast<size_t>(D_head); ++k) {
                        scalar_attn += s_i[h_off + k] * s_j[h_off + k];
                    }

                    // B. Point Attn: -sum(|q - k|^2)
                    float point_attn = 0.0f;
                    // Highway for points? Q is usually small (4-8). Scalar might be fine, but can vectorize loop over Q.
                    // Q*3 floats.
                    auto v_sum_dist = hn::Zero(d);
                    k = 0;
                    // Treat Q points as flattened array of 3*Q floats
                    // But calculations are (dx^2 + dy^2 + dz^2).
                    // Just unroll for small Q? Or scalar loop.
                    for (int64_t q = 0; q < Q; ++q) {
                        float dx = global_q_pts[q * 3 + 0] - global_k_pts[q * 3 + 0];
                        float dy = global_q_pts[q * 3 + 1] - global_k_pts[q * 3 + 1];
                        float dz = global_q_pts[q * 3 + 2] - global_k_pts[q * 3 + 2];
                        point_attn -= (dx * dx + dy * dy + dz * dz);
                    }

                    // C. Pair Bias
                    // sum(pair_ij[0:D_pair] * 0.1) -- example logic from original code
                    float pair_bias = 0.0f;
                    int64_t limit_d = D_pair < D_head ? D_pair : D_head;  // std::min

                    auto v_sum_pair = hn::Zero(d);
                    k = 0;
                    const auto v_factor = hn::Set(d, 0.1f);
                    for (; k + N <= static_cast<size_t>(limit_d); k += N) {
                        v_sum_pair = hn::MulAdd(hn::LoadU(d, pair_ij + k), v_factor, v_sum_pair);
                    }
                    pair_bias = hn::ReduceSum(d, v_sum_pair);
                    for (; k < static_cast<size_t>(limit_d); ++k) {
                        pair_bias += pair_ij[k] * 0.1f;
                    }

                    float score = (scalar_attn + point_attn * 0.5f + pair_bias) * scale;
                    attn_scores[j] = score;
                    if (score > max_score) max_score = score;
                }

                // 3. Softmax
                // ... same old softmax logic ...
                // Optimize with Highway
                auto v_max = hn::Set(d, max_score);
                auto v_sum_exp = hn::Zero(d);
                size_t j = 0;
                for (; j + N <= static_cast<size_t>(L); j += N) {
                    auto v = hn::LoadU(d, attn_scores + j);
                    auto v_exp = FastExpHwy(d, hn::Sub(v, v_max));
                    hn::StoreU(v_exp, d, attn_scores + j);
                    v_sum_exp = hn::Add(v_sum_exp, v_exp);
                }
                float sum_exp = hn::ReduceSum(d, v_sum_exp);
                for (; j < static_cast<size_t>(L); ++j) {
                    float val = std::exp(attn_scores[j] - max_score);
                    attn_scores[j] = val;
                    sum_exp += val;
                }

                float inv_sum = 1.0f / (sum_exp + 1e-9f);
                auto v_inv = hn::Set(d, inv_sum);
                j = 0;
                for (; j + N <= static_cast<size_t>(L); j += N) {
                    hn::StoreU(hn::Mul(hn::LoadU(d, attn_scores + j), v_inv), d, attn_scores + j);
                }
                for (; j < static_cast<size_t>(L); ++j) {
                    attn_scores[j] *= inv_sum;
                }

                // 4. Aggregate
                // Out_i = s_i + sum(attn[j] * s_j)  (Residual + Single Representation)
                // Out_i[points] += sum(attn[j] * Project(v_j)) (Point Representation)

                for (int64_t j = 0; j < L; ++j) {
                    float attn = attn_scores[j];
                    if (attn < 1e-9f) continue;

                    const float* s_j = s_data + b * L * D + j * D;
                    const float* R_j = R_data + b * L * 9 + j * 9;
                    const float* t_j = t_data + b * L * 3 + j * 3;

                    // Single Value Contribution
                    const auto v_attn = hn::Set(d, attn);
                    size_t k = 0;
                    for (; k + N <= static_cast<size_t>(D_head); k += N) {
                        auto v_out = hn::LoadU(d, out_i + h_off + k);
                        auto v_s = hn::LoadU(d, s_j + h_off + k);
                        hn::StoreU(hn::MulAdd(v_attn, v_s, v_out), d, out_i + h_off + k);
                    }
                    for (; k < static_cast<size_t>(D_head); ++k) {
                        out_i[h_off + k] += attn * s_j[h_off + k];
                    }

                    // Point Value Contribution
                    for (int64_t v = 0; v < V; ++v) {
                        const float* vp = vp_data + h * V * 3 + v * 3;
                        float r[3];
                        // Rotate to Global (frame j)
                        // RotatePoint(R_j, vp, r)
                        r[0] = R_j[0] * vp[0] + R_j[1] * vp[1] + R_j[2] * vp[2];
                        r[1] = R_j[3] * vp[0] + R_j[4] * vp[1] + R_j[5] * vp[2];
                        r[2] = R_j[6] * vp[0] + R_j[7] * vp[1] + R_j[8] * vp[2];

                        float gx = r[0] + t_j[0];
                        float gy = r[1] + t_j[1];
                        float gz = r[2] + t_j[2];

                        // Transform back to local frame i
                        float dx = gx - t_i[0];
                        float dy = gy - t_i[1];
                        float dz = gz - t_i[2];

                        // local = R_i^T * diff
                        float lx = R_i[0] * dx + R_i[3] * dy + R_i[6] * dz;
                        float ly = R_i[1] * dx + R_i[4] * dy + R_i[7] * dz;
                        float lz = R_i[2] * dx + R_i[5] * dy + R_i[8] * dz;

                        // Add to output
                        int64_t pt_offset = D_head - V * 3 + v * 3;
                        if (pt_offset >= 0 && pt_offset + 2 < D_head) {
                            out_i[h_off + pt_offset + 0] += attn * lx;
                            out_i[h_off + pt_offset + 1] += attn * ly;
                            out_i[h_off + pt_offset + 2] += attn * lz;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(InvariantPointAttentionImpl);

void InvariantPointAttention_Hwy(const float* s, const float* pair, const float* R, const float* t, const float* qp,
                                 const float* vp, float* out, void* workspace, size_t workspace_size, int64_t B,
                                 int64_t L, int64_t D, int64_t D_pair, int num_heads, int num_query_points,
                                 int num_value_points, float scale) {
    HWY_DYNAMIC_DISPATCH(InvariantPointAttentionImpl)(s, pair, R, t, qp, vp, out, workspace, workspace_size, B, L, D,
                                                      D_pair, num_heads, num_query_points, num_value_points, scale);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
