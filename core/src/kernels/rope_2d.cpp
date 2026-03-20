/**
 * @file rope_2d.cpp
 * @brief CPU fallback for RoPE2D
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "simd_ops.h"

#include <cstring>
#include <vector>

namespace densecore {
namespace {

class CpuRoPE2DOp : public EmbeddingOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        (void)params;
        if (inputs.size() < 4 || outputs.empty()) {
            return;
        }
        RoPE2D(*inputs[0], *inputs[1], *inputs[2], *inputs[3], outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 0};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::SEQ || layout == TensorLayout::PATCH || layout == TensorLayout::UNKNOWN;
    }

    void PatchEmbed2D(const Tensor& image, const Tensor& conv_weight, const Tensor& conv_bias,
                      Tensor* patches) override {
        // Not implemented in this specialized kernel
        (void)image;
        (void)conv_weight;
        (void)conv_bias;
        (void)patches;
    }

    void RoPE2D(const Tensor& input, const Tensor& pos_h, const Tensor& pos_w, const Tensor& cos_sin,
                Tensor* output) override {
        if (!input.IsValid() || !output || !cos_sin.IsValid() || cos_sin.shape.size() < 2) {
            return;
        }

        const int64_t n_heads = input.shape[1];
        const int64_t n_tokens = input.shape[2];
        const int64_t head_dim = input.shape[3];

        if (head_dim % 2 != 0) {
            // Should not happen for standard models
            return;
        }

        const int half_dim = static_cast<int>(head_dim) / 2;
        const float* in = input.DataAs<float>();
        float* out = output->DataAs<float>();
        const float* cs_base = cos_sin.DataAs<float>();
        const int64_t max_seq_len = cos_sin.shape[0];

        // pos_h and pos_w are [N] tensors (one pos per token)
        const int* ph_data = pos_h.DataAs<int>();
        const int* pw_data = pos_w.DataAs<int>();
        if (!ph_data || !pw_data) {
            std::memcpy(out, in, input.SizeBytes());
            return;
        }

        const int zero_pos = 0;  // Wrapper for offset-based call

        // Stride for stepping to next token/head
        const int64_t token_stride = head_dim;
        const int64_t head_stride = n_tokens * head_dim;

        // We iterate over batch * heads (flattened in input[0]) manually if Batch > 1?
        // Tensor shape is [B, N_HEADS, SEQ, HEAD_DIM].
        // The total number of sequences to process is B * N_HEADS.
        // Wait, input.shape[0] is Batch.
        const int64_t batch = input.shape[0];
        const int64_t batch_stride = n_heads * head_stride;

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < n_heads; ++h) {
                const float* in_h = in + b * batch_stride + h * head_stride;
                float* out_h = out + b * batch_stride + h * head_stride;

                for (int64_t t = 0; t < n_tokens; ++t) {
                    const int y = ph_data[t];
                    const int x = pw_data[t];

                    const float* ptr_in = in_h + t * token_stride;
                    float* ptr_out = out_h + t * token_stride;

                    // Guard invalid positions and fall back to pass-through for safety.
                    if (y < 0 || x < 0 || y >= max_seq_len || x >= max_seq_len) {
                        std::memcpy(ptr_out, ptr_in, sizeof(float) * head_dim);
                        continue;
                    }

                    // Half 1: Height part
                    // Use cos_sin row 'y', first half columns
                    const float* cs_h = cs_base + y * head_dim;
                    simd::ApplyRoPE(ptr_out, ptr_in, cs_h, &zero_pos, 1, half_dim, half_dim, 1);

                    // Half 2: Width part
                    // Use cos_sin row 'x', second half columns
                    const float* cs_w = cs_base + x * head_dim + half_dim;
                    simd::ApplyRoPE(ptr_out + half_dim, ptr_in + half_dim, cs_w, &zero_pos, 1, half_dim, half_dim, 1);
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuRoPE2DOp, OpType::RoPE2D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
