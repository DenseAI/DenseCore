/**
 * @file hwy_attention_pair_bias.cc
 * @brief Highway implementation of Attention with Pair Bias
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "kernels/hwy/hwy_attention_pair_bias.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "kernels/hwy/hwy_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace densecore {
namespace hwy_kernels {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Helper: Vectorized Dot Product
template <class D> float DotProductHwy(D d, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, int64_t size) {
    auto sum_v = hn::Zero(d);
    size_t i = 0;
    size_t lanes = hn::Lanes(d);
    for (; i + lanes <= static_cast<size_t>(size); i += lanes) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        sum_v = hn::MulAdd(va, vb, sum_v);
    }
    float sum = hn::ReduceSum(d, sum_v);
    for (; i < static_cast<size_t>(size); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper: Vectorized MatMul (RowMajor * RowMajor = RowMajor)
// A: [M, K], B: [K, N], C: [M, N]
// We vectorize the inner loop over N (columns of B/C) for contiguous access.
void MatMul2DHwy(const float* HWY_RESTRICT A, const float* HWY_RESTRICT B, float* HWY_RESTRICT C, int M, int N, int K) {
    const hn::ScalableTag<float> d;
    size_t lanes = hn::Lanes(d);

    for (int i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;

        size_t j = 0;
        for (; j + lanes <= static_cast<size_t>(N); j += lanes) {
            auto sum_v = hn::Zero(d);
            for (int k = 0; k < K; ++k) {
                // Broadcast A[i, k]
                auto a_val = hn::Set(d, a_row[k]);
                // Load B[k, j...j+lanes]
                auto b_vals = hn::Load(d, B + k * N + j);
                sum_v = hn::MulAdd(a_val, b_vals, sum_v);
            }
            hn::Store(sum_v, d, c_row + j);
        }
        // Scalar fallback for remaining columns
        for (; j < static_cast<size_t>(N); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * B[k * N + j];
            }
            c_row[j] = sum;
        }
    }
}

void AttentionWithPairBiasImpl(const float* HWY_RESTRICT msa_input, const float* HWY_RESTRICT pair_data,
                               const float* HWY_RESTRICT wq_data, const float* HWY_RESTRICT wk_data,
                               const float* HWY_RESTRICT wv_data, const float* HWY_RESTRICT wo_data,
                               const float* HWY_RESTRICT pair_bias_w_data, float* HWY_RESTRICT output, int tokens,
                               int c_m, int c_z, int n_head, int head_dim, int pair_tokens, float scale_param) {
    const hn::ScalableTag<float> d;

    auto ZeroOutput = [output, tokens, c_m]() {
        if (!output || tokens <= 0 || c_m <= 0) return;
        std::fill_n(output, static_cast<size_t>(tokens) * static_cast<size_t>(c_m), 0.0f);
    };

    // Public API hardening: reject malformed shapes and prevent division by zero.
    if (!msa_input || !pair_data || !wq_data || !wk_data || !wv_data || !wo_data || !pair_bias_w_data || !output ||
        tokens <= 0 || c_m <= 0 || c_z <= 0 || n_head <= 0 || head_dim <= 0 || pair_tokens <= 0) {
        ZeroOutput();
        return;
    }

    // Derived shapes
    // pair_tokens = n_res * n_res
    int n_res = static_cast<int>(std::sqrt(static_cast<float>(pair_tokens)));
    if (n_res <= 0 || static_cast<int64_t>(n_res) * static_cast<int64_t>(n_res) != static_cast<int64_t>(pair_tokens) ||
        tokens % n_res != 0) {
        ZeroOutput();
        return;
    }

    int n_seq = tokens / n_res;
    float scale = (scale_param > 0) ? scale_param : 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Internal buffers
    // We allocate these on heap via std::vector as they can be large
    // In a real optimized kernel we might use a workspace buffer passed in
    std::vector<float> q(static_cast<size_t>(tokens) * c_m);
    std::vector<float> k(static_cast<size_t>(tokens) * c_m);
    std::vector<float> v(static_cast<size_t>(tokens) * c_m);

    // 1. Projections
    MatMul2DHwy(msa_input, wq_data, q.data(), tokens, c_m, c_m);
    MatMul2DHwy(msa_input, wk_data, k.data(), tokens, c_m, c_m);
    MatMul2DHwy(msa_input, wv_data, v.data(), tokens, c_m, c_m);

    // 2. Compute Pair Bias [n_res, n_res, n_head]
    // The loop structure: for each pair (i,j), dot product with each head's weight vector
    std::vector<float> pair_bias(static_cast<size_t>(n_res) * n_res * n_head);

    // 2. Compute Pair Bias [n_res, n_res, n_head]

    for (int i = 0; i < n_res; ++i) {
        for (int j = 0; j < n_res; ++j) {
            const float* pair_ij = pair_data + (static_cast<size_t>(i) * n_res + j) * c_z;
            for (int h = 0; h < n_head; ++h) {
                const float* w = pair_bias_w_data + h * c_z;
                // Vectorized dot product
                pair_bias[(static_cast<size_t>(i) * n_res + j) * n_head + h] = DotProductHwy(d, pair_ij, w, c_z);
            }
        }
    }

    // 3. Attention
    // Intermediate buffer for attention scores/probs per head: [n_res]
    // Reused per head
    std::vector<float> scores(n_res);
    // Buffer for projections before output matmul
    std::vector<float> attn_out(static_cast<size_t>(tokens) * c_m, 0.0f);

    for (int s = 0; s < n_seq; ++s) {
        for (int i = 0; i < n_res; ++i) {
            float* out_row = attn_out.data() + (static_cast<size_t>(s) * n_res + i) * c_m;  // [c_m] (broken into heads)

            for (int h = 0; h < n_head; ++h) {
                // Q vector for this token/head
                const float* q_ptr = q.data() + (static_cast<size_t>(s) * n_res + i) * c_m + h * head_dim;
                float* out_head_ptr = out_row + h * head_dim;

                float max_score = -std::numeric_limits<float>::infinity();

                // Score for each key j
                for (int j = 0; j < n_res; ++j) {
                    const float* k_ptr = k.data() + (static_cast<size_t>(s) * n_res + j) * c_m + h * head_dim;
                    float dot = DotProductHwy(d, q_ptr, k_ptr, head_dim);

                    dot *= scale;
                    dot += pair_bias[(static_cast<size_t>(i) * n_res + j) * n_head + h];

                    scores[j] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // Softmax
                float sum_exp = 0.0f;

                for (int jj = 0; jj < n_res; ++jj) {
                    float e = std::exp(scores[jj] - max_score);
                    scores[jj] = e;  // Store weight temporarily
                    sum_exp += e;
                }

                float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

                // Weighted sum of Values

                size_t lanes = hn::Lanes(d);
                std::memset(out_head_ptr, 0, sizeof(float) * head_dim);

                for (int jj = 0; jj < n_res; ++jj) {
                    float w = scores[jj] * inv_sum;
                    const float* v_ptr = v.data() + (static_cast<size_t>(s) * n_res + jj) * c_m + h * head_dim;
                    auto vw = hn::Set(d, w);

                    size_t dd = 0;
                    for (; dd + lanes <= static_cast<size_t>(head_dim); dd += lanes) {
                        auto val_out = hn::Load(d, out_head_ptr + dd);
                        auto val_v = hn::Load(d, v_ptr + dd);
                        val_out = hn::MulAdd(vw, val_v, val_out);
                        hn::Store(val_out, d, out_head_ptr + dd);
                    }
                    for (; dd < static_cast<size_t>(head_dim); ++dd) {
                        out_head_ptr[dd] += w * v_ptr[dd];
                    }
                }
            }
        }
    }

    // 4. Output Projection
    MatMul2DHwy(attn_out.data(), wo_data, output, tokens, c_m, c_m);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_kernels
}  // namespace densecore
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace densecore {
namespace hwy_kernels {

HWY_EXPORT(AttentionWithPairBiasImpl);

void AttentionWithPairBias_Hwy(const float* msa_input, const float* pair_data, const float* wq, const float* wk,
                               const float* wv, const float* wo, const float* pair_bias_w, float* output, int tokens,
                               int c_m, int c_z, int n_head, int head_dim, int pair_tokens, float scale) {
    HWY_DYNAMIC_DISPATCH(AttentionWithPairBiasImpl)
    (msa_input, pair_data, wq, wk, wv, wo, pair_bias_w, output, tokens, c_m, c_z, n_head, head_dim, pair_tokens, scale);
}

}  // namespace hwy_kernels
}  // namespace densecore
#endif  // HWY_ONCE
