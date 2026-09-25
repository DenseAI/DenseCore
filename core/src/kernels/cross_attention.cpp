/**
 * @file cross_attention.cpp
 * @brief Cross-Attention Kernel for Encoder-Decoder Models
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * ## Cross-Attention Overview
 *
 * Cross-Attention is a key operation in Encoder-Decoder architectures.
 * The Decoder Query extracts information by attending to the Encoder's Key/Value.
 *
 * **Target Models:**
 * - Whisper (Speech Recognition)
 * - T5/BART (Seq2Seq)
 * - mT5 (Multilingual Translation)
 *
 * **Optimization Strategy:**
 * - Encoder KV is fixed after encoding → Remove redundant computation via caching
 * - Parallel dot product using SIMD
 * - Improve cache locality with memory prefetching
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/transformer_ops.h"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "kernels/hwy/hwy_kernels.h"

namespace densecore {

// ============================================================================
// Cross-Attention Implementation
// ============================================================================

/**
 * @brief CPU Cross-Attention Kernel
 *
 * Computes attention between Decoder Query and Encoder Key/Value.
 * Used in Encoder-Decoder models (Whisper, T5).
 *
 * ## Mathematical Formulation
 *
 * Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Where:
 * - Q: Decoder queries [B, Lq, H, D]
 * - K: Encoder keys (cached) [B, Le, H, D]
 * - V: Encoder values (cached) [B, Le, H, D]
 * - Lq: Decoder sequence length
 * - Le: Encoder sequence length
 * - H: Number of heads
 * - D: Head dimension
 *
 * ## Optimization Notes
 *
 * - **No causal mask**: Cross-attention is not causal (decoder attends to entire encoder output)
 * - **Encoder KV Caching**: Encoder output is fixed, so computed once and cached
 * - **GQA Support**: Reuses key/value heads if n_head_kv < n_head_q
 *
 * Time Complexity: O(B * Lq * Le * D)
 */
class CpuCrossAttention : public DenseCoreOp {
public:
    CpuCrossAttention() = default;
    ~CpuCrossAttention() override = default;

    OpCapabilities GetCapabilities() const override {
        return OpCapabilities{
            .supports_fp16 = true,
            .supports_int8 = false,
            .supports_int4 = false,
            .max_batch_size = 256,
            .l2_cache_bytes = 0,
            .memory_bandwidth_gbps = 0,
            .priority = 50,
        };
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) {
            std::cerr << "[CrossAttention] Invalid input/output count" << std::endl;
            return;
        }

        const Tensor& Q = *inputs[0];  // Decoder query [B, Lq, H, D]
        const Tensor& K = *inputs[1];  // Encoder key [B, Le, H, D]
        const Tensor& V = *inputs[2];  // Encoder value [B, Le, H, D]
        Tensor& out = *outputs[0];

        float scale = 1.0f;
        if (params) {
            const auto* p = static_cast<const CrossAttentionParams*>(params);
            scale = p->scale;
        }

        CrossAttentionImpl(Q, K, V, &out, scale);
    }

private:
    /**
     * @brief Cross-Attention Implementation
     * Uses a scalar reference path for correctness and bounds safety.
     * Cross-attention is not on the DenseDiffusion hot path, so a safe
     * implementation is preferable to an unstable SIMD specialization.
     */
    void CrossAttentionImpl(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* out, float scale) {
        if (Q.ndim != 4 || K.ndim != 4 || V.ndim != 4 || !out || out->ndim != 4) {
            std::cerr << "[CrossAttention] Expected 4D Q/K/V/output tensors" << std::endl;
            return;
        }

        // Shape extraction: [batch, seq, n_head, head_dim]
        const int64_t batch = Q.shape[0];
        const int64_t seq_q = Q.shape[1];
        const int64_t n_head = Q.shape[2];
        const int64_t head_dim = Q.shape[3];
        const int64_t seq_k = K.shape[1];
        const int64_t n_kv_head = K.shape[2];

        if (batch <= 0 || seq_q <= 0 || seq_k <= 0 || n_head <= 0 || head_dim <= 0 || n_kv_head <= 0) {
            std::cerr << "[CrossAttention] Invalid tensor dimensions" << std::endl;
            return;
        }
        if (K.shape[0] != batch || V.shape[0] != batch || K.shape[1] != seq_k || V.shape[1] != seq_k ||
            K.shape[2] != n_kv_head || V.shape[2] != n_kv_head || K.shape[3] != head_dim || V.shape[3] != head_dim ||
            out->shape[0] != batch || out->shape[1] != seq_q || out->shape[2] != n_head || out->shape[3] != head_dim) {
            std::cerr << "[CrossAttention] Incompatible tensor shapes" << std::endl;
            return;
        }
        if (n_head % n_kv_head != 0) {
            std::cerr << "[CrossAttention] n_head must be divisible by n_kv_head" << std::endl;
            return;
        }

        const float* q_data = Q.DataAs<float>();
        const float* k_data = K.DataAs<float>();
        const float* v_data = V.DataAs<float>();
        float* o_data = out->DataAs<float>();
        if (!q_data || !k_data || !v_data || !o_data) {
            std::cerr << "[CrossAttention] Null tensor data" << std::endl;
            return;
        }

        const int64_t q_stride_b = Q.stride[0];
        const int64_t q_stride_s = Q.stride[1];
        const int64_t q_stride_h = Q.stride[2];
        const int64_t q_stride_d = Q.stride[3];
        const int64_t k_stride_b = K.stride[0];
        const int64_t k_stride_s = K.stride[1];
        const int64_t k_stride_h = K.stride[2];
        const int64_t k_stride_d = K.stride[3];
        const int64_t v_stride_b = V.stride[0];
        const int64_t v_stride_s = V.stride[1];
        const int64_t v_stride_h = V.stride[2];
        const int64_t v_stride_d = V.stride[3];
        const int64_t o_stride_b = out->stride[0];
        const int64_t o_stride_s = out->stride[1];
        const int64_t o_stride_h = out->stride[2];
        const int64_t o_stride_d = out->stride[3];

        std::vector<float> attn_scores(static_cast<size_t>(seq_k), 0.0f);
        const int64_t group_size = n_head / n_kv_head;

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t h_kv = h / group_size;
                for (int64_t q_pos = 0; q_pos < seq_q; ++q_pos) {
                    const float* q_ptr = q_data + b * q_stride_b + q_pos * q_stride_s + h * q_stride_h;
                    float max_score = -std::numeric_limits<float>::infinity();

                    for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                        const float* k_ptr = k_data + b * k_stride_b + k_pos * k_stride_s + h_kv * k_stride_h;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            dot += q_ptr[d * q_stride_d] * k_ptr[d * k_stride_d];
                        }
                        dot *= scale;
                        attn_scores[static_cast<size_t>(k_pos)] = dot;
                        max_score = std::max(max_score, dot);
                    }

                    float sum_exp = 0.0f;
                    for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                        float value = std::exp(attn_scores[static_cast<size_t>(k_pos)] - max_score);
                        attn_scores[static_cast<size_t>(k_pos)] = value;
                        sum_exp += value;
                    }
                    const float inv_sum = 1.0f / (sum_exp + 1e-9f);

                    float* out_ptr = o_data + b * o_stride_b + q_pos * o_stride_s + h * o_stride_h;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        out_ptr[d * o_stride_d] = 0.0f;
                    }

                    for (int64_t k_pos = 0; k_pos < seq_k; ++k_pos) {
                        const float weight = attn_scores[static_cast<size_t>(k_pos)] * inv_sum;
                        const float* v_ptr = v_data + b * v_stride_b + k_pos * v_stride_s + h_kv * v_stride_h;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            out_ptr[d * o_stride_d] += weight * v_ptr[d * v_stride_d];
                        }
                    }
                }
            }
        }
    }
};

// ============================================================================
// Auto-Registration (Unified Macro Pattern)
// ============================================================================

DENSECORE_REGISTER_OP(CpuCrossAttention, OpType::CrossAttention, DeviceType::CPU);

}  // namespace densecore
