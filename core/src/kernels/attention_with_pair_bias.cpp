/**
 * @file attention_with_pair_bias.cpp
 * @brief Multi-Head Attention with Pair Representation Bias (AlphaFold2/3)
 *
 * Standard multi-head attention where attention logits are augmented with
 * a bias term projected from the pair representation:
 *   attn_logits[s,i,h,j] = Q[s,i,h] . K[s,j,h] / sqrt(d) + pair_bias[i,j,h]
 *
 * Inputs:
 *   [0] msa_input    [N_seq*N_res, C_m]  - MSA representation (flattened)
 *   [1] pair_repr    [N_res*N_res, C_z]   - Pair representation (flattened)
 *   [2] wq           [C_m, C_m]           - Query projection
 *   [3] wk           [C_m, C_m]           - Key projection
 *   [4] wv           [C_m, C_m]           - Value projection
 *   [5] wo           [C_m, C_m]           - Output projection
 *   [6] pair_bias_w  [N_head, C_z]        - Pair bias projection weights
 *
 * Output:
 *   [0] output       [N_seq*N_res, C_m]   - Attention output
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/hwy/hwy_kernels.h"

namespace densecore {
namespace {

class CpuAttentionWithPairBiasOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 7 || outputs.empty()) return;

        const auto* p = static_cast<const AttentionWithPairBiasParams*>(params);
        AttentionWithPairBiasParams default_params;
        if (!p) p = &default_params;

        AttentionWithPairBias(*inputs[0], *inputs[1], *inputs[2], *inputs[3], *inputs[4], *inputs[5], *inputs[6], *p,
                              outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 8 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

private:
    void AttentionWithPairBias(const Tensor& msa_input, const Tensor& pair_repr, const Tensor& wq, const Tensor& wk,
                               const Tensor& wv, const Tensor& wo, const Tensor& pair_bias_w,
                               const AttentionWithPairBiasParams& params, Tensor* output) {
        if (!msa_input.IsValid() || !output) return;

        const int tokens = static_cast<int>(msa_input.shape[0]);
        const int c_m = static_cast<int>(msa_input.shape[1]);
        const int c_z = params.c_z;
        const int n_head = params.n_head;
        const int head_dim = c_m / n_head;

        // Infer n_res from pair_repr shape
        const int pair_tokens = static_cast<int>(pair_repr.shape[0]);

        const float* msa_data = msa_input.DataAs<float>();
        const float* pair_data = pair_repr.DataAs<float>();
        const float* pbw = pair_bias_w.DataAs<float>();
        const float* wq_d = wq.DataAs<float>();
        const float* wk_d = wk.DataAs<float>();
        const float* wv_d = wv.DataAs<float>();
        const float* wo_d = wo.DataAs<float>();
        float* out_data = output->DataAs<float>();

        hwy_kernels::AttentionWithPairBias_Hwy(msa_data, pair_data, wq_d, wk_d, wv_d, wo_d, pbw, out_data, tokens, c_m,
                                               c_z, n_head, head_dim, pair_tokens, params.scale);
    }
};

DENSECORE_REGISTER_OP(CpuAttentionWithPairBiasOp, OpType::AttentionWithPairBias, DeviceType::CPU);

}  // namespace
}  // namespace densecore
