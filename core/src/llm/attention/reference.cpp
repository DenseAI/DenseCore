#include "llm/attention/callback_ops.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

void ComputeFlashAttentionReference(const float* q, const float* k, const float* v, float* out, int n_head,
                                    int n_head_kv, int seq_q, int seq_kv, int head_dim, float scale, bool causal,
                                    int q_start_offset, int kv_start_offset, int sliding_window, float logit_softcap) {
    if (!q || !k || !v || !out || n_head <= 0 || n_head_kv <= 0 || seq_q <= 0 || seq_kv <= 0 || head_dim <= 0) {
        return;
    }

    const int n_rep = n_head / n_head_kv;
    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / n_rep;
        const float* q_head = q + static_cast<size_t>(h) * seq_q * head_dim;
        const float* k_head = k + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        const float* v_head = v + static_cast<size_t>(kv_head) * seq_kv * head_dim;
        float* out_head = out + static_cast<size_t>(h) * seq_q * head_dim;

        for (int tq = 0; tq < seq_q; ++tq) {
            const float* q_row = q_head + static_cast<size_t>(tq) * head_dim;
            float* out_row = out_head + static_cast<size_t>(tq) * head_dim;
            std::fill(out_row, out_row + head_dim, 0.0f);

            std::vector<float> scores(static_cast<size_t>(seq_kv), -INFINITY);
            float row_max = -INFINITY;
            for (int tk = 0; tk < seq_kv; ++tk) {
                const int query_pos = q_start_offset + tq;
                const int key_pos = kv_start_offset + tk;
                if (causal && key_pos > query_pos) {
                    continue;
                }
                if (sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                    continue;
                }
                const float* k_row = k_head + static_cast<size_t>(tk) * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;
                if (logit_softcap > 0.0f) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
                scores[static_cast<size_t>(tk)] = score;
                row_max = std::max(row_max, score);
            }

            if (!std::isfinite(row_max)) {
                continue;
            }

            float denom = 0.0f;
            for (int tk = 0; tk < seq_kv; ++tk) {
                float& score = scores[static_cast<size_t>(tk)];
                if (!std::isfinite(score)) {
                    score = 0.0f;
                    continue;
                }
                score = std::exp(score - row_max);
                denom += score;
            }
            if (!(denom > 0.0f) || !std::isfinite(denom)) {
                continue;
            }

            const float inv_denom = 1.0f / denom;
            for (int tk = 0; tk < seq_kv; ++tk) {
                const float p = scores[static_cast<size_t>(tk)] * inv_denom;
                if (!(p > 0.0f)) {
                    continue;
                }
                const float* v_row = v_head + static_cast<size_t>(tk) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_row[d] += p * v_row[d];
                }
            }
        }
    }
}
