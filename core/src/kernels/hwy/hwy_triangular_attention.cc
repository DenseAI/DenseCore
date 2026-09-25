/**
 * @file hwy_triangular_attention.cc
 * @brief Triangular Attention via Google Highway
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_triangular_attention.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

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

// Project: output[D_out] = weight[D_out, D_in] @ input[D_in]
// Input: [D_in]
// Weight: [D_out, D_in] (row-major)
// Output: [D_out]
template <class D>
void ProjectVectorHwy(D d, const float* input, const float* weight, int64_t D_in, int64_t D_out, float* output) {
    auto N = hn::Lanes(d);

    for (int64_t r = 0; r < D_out; ++r) {
        const float* w_row = weight + r * D_in;
        auto v_sum = hn::Zero(d);

        size_t c = 0;
        for (; c + N <= static_cast<size_t>(D_in); c += N) {
            auto v_in = hn::LoadU(d, input + c);
            auto v_w = hn::LoadU(d, w_row + c);
            v_sum = hn::MulAdd(v_in, v_w, v_sum);
        }
        float sum = hn::ReduceSum(d, v_sum);
        for (; c < static_cast<size_t>(D_in); ++c) {
            sum += input[c] * w_row[c];
        }
        output[r] = sum;
    }
}

// ============================================================================
// Triangular Attention Implementation
// ============================================================================

void TriangularAttentionImpl(const float* pair_data, const float* qw, const float* kw, const float* vw, float* out_data,
                             void* workspace, size_t workspace_size, int64_t B, int64_t L, int64_t D, int num_heads,
                             float scale, bool starting, int64_t tile_row_start, int64_t tile_row_end,
                             int64_t tile_col_start, int64_t tile_col_end, bool output_is_tiled) {
    if (!pair_data || !qw || !kw || !vw || !out_data || !workspace) return;
    if (B <= 0 || L <= 0 || D <= 0 || num_heads <= 0) return;
    if ((D % num_heads) != 0) return;

    const hn::ScalableTag<float> d;
    auto N = hn::Lanes(d);

    const int64_t D_head = D / num_heads;
    if (D_head <= 0) return;
    auto fits_size_t = [](int64_t v) -> bool {
        if (v < 0) return false;
        return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    };
    if (!fits_size_t(B) || !fits_size_t(L) || !fits_size_t(D) || !fits_size_t(D_head)) {
        return;
    }
    const size_t B_sz = static_cast<size_t>(B);
    const size_t L_sz = static_cast<size_t>(L);
    const size_t D_sz = static_cast<size_t>(D);
    const size_t D_head_sz = static_cast<size_t>(D_head);

    auto checked_add = [](size_t a, size_t b, size_t* out) -> bool {
        if (!out) return false;
        if (a > std::numeric_limits<size_t>::max() - b) return false;
        *out = a + b;
        return true;
    };
    auto checked_mul = [](size_t a, size_t b, size_t* out) -> bool {
        if (!out) return false;
        if (a == 0 || b == 0) {
            *out = 0;
            return true;
        }
        if (a > std::numeric_limits<size_t>::max() / b) return false;
        *out = a * b;
        return true;
    };

    size_t row_stride = 0;
    size_t pair_stride = 0;
    size_t full_elements = 0;
    size_t full_bytes = 0;
    size_t head_weight_stride = 0;
    if (!checked_mul(L_sz, D_sz, &row_stride)) return;
    if (!checked_mul(L_sz, row_stride, &pair_stride)) return;
    if (!checked_mul(B_sz, pair_stride, &full_elements)) return;
    if (!checked_mul(full_elements, sizeof(float), &full_bytes)) return;
    if (!checked_mul(D_head_sz, D_sz, &head_weight_stride)) return;

    // Workspace Requirements:
    // q_proj: D_head
    // k_proj: D_head
    // v_proj: D_head
    // attn_accum: D_head
    // scores: L
    size_t proj_floats = 0;
    size_t required_floats = 0;
    size_t required_bytes = 0;
    if (!checked_mul(static_cast<size_t>(4), D_head_sz, &proj_floats)) return;
    if (!checked_add(proj_floats, L_sz, &required_floats)) return;
    if (!checked_mul(required_floats, sizeof(float), &required_bytes)) return;
    if (workspace_size < required_bytes) {
        fprintf(stderr, "[TriangularAttention] workspace too small: need %zu B, got %zu B\n",
                required_bytes, workspace_size);
        return;
    }

    float* ws_ptr = reinterpret_cast<float*>(workspace);
    float* q_proj_ptr = ws_ptr;
    ws_ptr += D_head_sz;
    float* k_proj_ptr = ws_ptr;
    ws_ptr += D_head_sz;
    float* v_proj_ptr = ws_ptr;
    ws_ptr += D_head_sz;
    float* attn_accum_ptr = ws_ptr;
    ws_ptr += D_head_sz;
    float* scores_ptr = ws_ptr;
    ws_ptr += L_sz;
    (void)ws_ptr;

    const int64_t row_start = std::max<int64_t>(0, std::min<int64_t>(L, tile_row_start));
    const int64_t row_end = std::max<int64_t>(row_start, std::min<int64_t>(L, tile_row_end));
    const int64_t col_start = std::max<int64_t>(0, std::min<int64_t>(L, tile_col_start));
    const int64_t col_end = std::max<int64_t>(col_start, std::min<int64_t>(L, tile_col_end));
    if (row_end <= row_start || col_end <= col_start) return;

    const int64_t tile_rows = row_end - row_start;
    const int64_t tile_cols = col_end - col_start;
    const size_t tile_rows_sz = static_cast<size_t>(tile_rows);
    const size_t tile_cols_sz = static_cast<size_t>(tile_cols);
    size_t tile_tokens = 0;
    size_t tile_stride = 0;
    if (!checked_mul(tile_rows_sz, tile_cols_sz, &tile_tokens)) return;
    if (!checked_mul(tile_tokens, D_sz, &tile_stride)) return;
    const bool full_tile = (row_start == 0 && row_end == L && col_start == 0 && col_end == L);

    // Full output path keeps previous residual behavior: start from pair_data.
    if (full_tile && !output_is_tiled) {
        std::memcpy(out_data, pair_data, full_bytes);
    }

    for (int64_t b = 0; b < B; ++b) {
        const size_t b_off = static_cast<size_t>(b) * pair_stride;
        const float* b_pair = pair_data + b_off;
        float* b_out_full = output_is_tiled ? nullptr : (out_data + b_off);
        float* b_out_tile = output_is_tiled ? (out_data + static_cast<size_t>(b) * tile_stride) : nullptr;

        for (int64_t i = row_start; i < row_end; ++i) {
            for (int64_t j = col_start; j < col_end; ++j) {
                const float* pair_ij =
                    b_pair + static_cast<size_t>(i) * row_stride + static_cast<size_t>(j) * D_sz;
                float* out_ij = nullptr;
                if (output_is_tiled) {
                    const int64_t ti = i - row_start;
                    const int64_t tj = j - col_start;
                    out_ij = b_out_tile + (static_cast<size_t>(ti) * tile_cols_sz + static_cast<size_t>(tj)) * D_sz;
                } else {
                    out_ij = b_out_full + static_cast<size_t>(i) * row_stride + static_cast<size_t>(j) * D_sz;
                }

                // Tiled updates only initialize residual for touched elements.
                if (!full_tile || output_is_tiled) {
                    std::memcpy(out_ij, pair_ij, D_sz * sizeof(float));
                }

                for (int h = 0; h < num_heads; ++h) {
                    const size_t h_off = static_cast<size_t>(h) * D_head_sz;
                    size_t h_weight_off = 0;
                    if (!checked_mul(static_cast<size_t>(h), head_weight_stride, &h_weight_off)) return;
                    const float* qw_h = qw + h_weight_off;
                    const float* kw_h = kw + h_weight_off;
                    const float* vw_h = vw + h_weight_off;

                    // Project Query from (i, j)
                    // W_q: [D_head, D] (row-major)
                    // Input: pair_ij [D]
                    // Output: q_proj [D_head]
                    ProjectVectorHwy(d, pair_ij, qw_h, D, D_head, q_proj_ptr);

                    float max_score = -1e30f;

                    // 1. Compute Scores
                    for (int64_t k = 0; k < L; ++k) {
                        const float* pair_ik = nullptr;
                        if (starting) {
                            // (i, k)
                            pair_ik = b_pair + static_cast<size_t>(i) * row_stride + static_cast<size_t>(k) * D_sz;
                        } else {
                            // (k, j)
                            pair_ik = b_pair + static_cast<size_t>(k) * row_stride + static_cast<size_t>(j) * D_sz;
                        }

                        // Project Key
                        ProjectVectorHwy(d, pair_ik, kw_h, D, D_head, k_proj_ptr);

                        // Dot product: q_proj . k_proj
                        auto v_dot = hn::Zero(d);
                        size_t c = 0;
                        for (; c + N <= static_cast<size_t>(D_head); c += N) {
                            v_dot = hn::MulAdd(hn::LoadU(d, q_proj_ptr + c), hn::LoadU(d, k_proj_ptr + c), v_dot);
                        }
                        float dot = hn::ReduceSum(d, v_dot);
                        for (; c < static_cast<size_t>(D_head); ++c) {
                            dot += q_proj_ptr[c] * k_proj_ptr[c];
                        }
                        scores_ptr[k] = dot * scale;
                        if (scores_ptr[k] > max_score) max_score = scores_ptr[k];
                    }

                    // 2. Softmax
                    float sum_exp = 0.0f;
                    auto v_max = hn::Set(d, max_score);
                    auto v_sum_exp = hn::Zero(d);
                    size_t k = 0;
                    for (; k + N <= static_cast<size_t>(L); k += N) {
                        auto v = hn::LoadU(d, scores_ptr + k);
                        auto v_exp = FastExpHwy(d, hn::Sub(v, v_max));
                        hn::StoreU(v_exp, d, scores_ptr + k);
                        v_sum_exp = hn::Add(v_sum_exp, v_exp);
                    }
                    sum_exp = hn::ReduceSum(d, v_sum_exp);
                    for (; k < static_cast<size_t>(L); ++k) {
                        float val = FastExpScalar(scores_ptr[k] - max_score);
                        scores_ptr[k] = val;
                        sum_exp += val;
                    }

                    float inv_sum = 1.0f / (sum_exp + 1e-9f);

                    // 3. Aggregate
                    std::memset(attn_accum_ptr, 0, D_head * sizeof(float));

                    for (int64_t k = 0; k < L; ++k) {
                        float prob = scores_ptr[k] * inv_sum;
                        if (prob < 1e-9f) continue;

                        const float* pair_ik = nullptr;
                        if (starting) {
                            pair_ik = b_pair + static_cast<size_t>(i) * row_stride + static_cast<size_t>(k) * D_sz;
                        } else {
                            pair_ik = b_pair + static_cast<size_t>(k) * row_stride + static_cast<size_t>(j) * D_sz;
                        }

                        // Project Value
                        ProjectVectorHwy(d, pair_ik, vw_h, D, D_head, v_proj_ptr);

                        // Accumulate: accum += prob * value
                        auto v_prob = hn::Set(d, prob);
                        size_t c = 0;
                        for (; c + N <= static_cast<size_t>(D_head); c += N) {
                            auto v_acc = hn::LoadU(d, attn_accum_ptr + c);
                            auto v_val = hn::LoadU(d, v_proj_ptr + c);
                            hn::StoreU(hn::MulAdd(v_prob, v_val, v_acc), d, attn_accum_ptr + c);
                        }
                        for (; c < static_cast<size_t>(D_head); ++c) {
                            attn_accum_ptr[c] += prob * v_proj_ptr[c];
                        }
                    }

                    // 4. Update Output (Residual + Attention)
                    // out_ij[h_off] += attn_accum
                    size_t c = 0;
                    for (; c + N <= static_cast<size_t>(D_head); c += N) {
                        auto v_out = hn::LoadU(d, out_ij + h_off + c);
                        auto v_acc = hn::LoadU(d, attn_accum_ptr + c);
                        hn::StoreU(hn::Add(v_out, v_acc), d, out_ij + h_off + c);
                    }
                    for (; c < static_cast<size_t>(D_head); ++c) {
                        out_ij[h_off + c] += attn_accum_ptr[c];
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

HWY_EXPORT(TriangularAttentionImpl);

void TriangularAttentionTiled_Hwy(const float* pair_data, const float* qw, const float* kw, const float* vw,
                                  float* out_data, void* workspace, size_t workspace_size, int64_t B, int64_t L,
                                  int64_t D, int num_heads, float scale, bool starting, int64_t tile_row_start,
                                  int64_t tile_row_end, int64_t tile_col_start, int64_t tile_col_end,
                                  bool output_is_tiled) {
    auto fn = HWY_DYNAMIC_DISPATCH(TriangularAttentionImpl);
    fn(pair_data, qw, kw, vw, out_data, workspace, workspace_size, B, L, D, num_heads, scale, starting, tile_row_start,
       tile_row_end, tile_col_start, tile_col_end, output_is_tiled);
}

void TriangularAttention_Hwy(const float* pair_data, const float* qw, const float* kw, const float* vw, float* out_data,
                             void* workspace, size_t workspace_size, int64_t B, int64_t L, int64_t D, int num_heads,
                             float scale, bool starting) {
    TriangularAttentionTiled_Hwy(pair_data, qw, kw, vw, out_data, workspace, workspace_size, B, L, D, num_heads, scale,
                                 starting, 0, L, 0, L, false);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
