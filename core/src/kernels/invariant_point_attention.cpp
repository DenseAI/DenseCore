/**
 * @file invariant_point_attention.cpp
 * @brief Invariant Point Attention (IPA) for Protein Structure (AlphaFold2/3)
 *
 * SE(3)-equivariant attention that operates on 3D point clouds.
 * Key innovation: attention scores are invariant to global rotation/translation.
 *
 * ## Key Concepts
 * - Points in local reference frame
 * - Distance-based attention bias
 * - Rotation-invariant features
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "kernels/hwy/hwy_kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// IPA Parameters
// ============================================================================

struct IPAParams {
    int num_heads = 12;
    int num_query_points = 4;
    int num_value_points = 8;
    float scale = 0.0f;  // 0 = auto
};


// ============================================================================
// CpuIPAOp
// ============================================================================

class CpuIPAOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 6 || outputs.empty()) return;

        const IPAParams* p = static_cast<const IPAParams*>(params);
        IPAParams default_params;
        if (!p) p = &default_params;

        InvariantPointAttention(*inputs[0], *inputs[1], *inputs[2], *inputs[3], *inputs[4], *inputs[5], *p, outputs[0]);
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

    /**
     * @brief Invariant Point Attention
     *
     * @param single_repr  [B, L, D] - Single (per-residue) representation
     * @param pair_repr    [B, L, L, D_pair] - Pair representation
     * @param rotations    [B, L, 3, 3] - Per-residue rotation matrices
     * @param translations [B, L, 3] - Per-residue translation vectors
     * @param query_points [H, Q, 3] - Query point offsets (learnable)
     * @param value_points [H, V, 3] - Value point offsets (learnable)
     * @param params       Configuration
     * @param output       [B, L, D] - Updated single representation
     */
    void InvariantPointAttention(const Tensor& single_repr, const Tensor& pair_repr, const Tensor& rotations,
                                 const Tensor& translations, const Tensor& query_points, const Tensor& value_points,
                                 const IPAParams& params, Tensor* output) {
        if (!single_repr.IsValid() || !pair_repr.IsValid() || !rotations.IsValid() || !translations.IsValid() ||
            !query_points.IsValid() || !value_points.IsValid() || !output) {
            return;
        }

        const int64_t B = single_repr.shape[0];
        const int64_t L = single_repr.shape[1];
        const int64_t D = single_repr.shape[2];
        const int64_t D_pair = pair_repr.shape[3];
        const int64_t H = params.num_heads;
        const int64_t Q = params.num_query_points;
        const int64_t V = params.num_value_points;

        const float* s_data = single_repr.DataAs<float>();
        const float* pair_data = pair_repr.DataAs<float>();
        const float* R_data = rotations.DataAs<float>();
        const float* t_data = translations.DataAs<float>();
        const float* qp_data = query_points.DataAs<float>();
        const float* vp_data = value_points.DataAs<float>();
        float* out_data = output->DataAs<float>();

        // Workspace allocation: (2 * Q * 3 + L) * sizeof(float)
        size_t workspace_size = (2 * Q * 3 + L) * sizeof(float);
        std::vector<uint8_t> workspace(workspace_size);

        hwy_kernels::InvariantPointAttention_Hwy(s_data, pair_data, R_data, t_data, qp_data, vp_data, out_data,
                                                 workspace.data(), workspace_size, B, L, D, D_pair, H, Q, V,
                                                 params.scale);
    }
};

DENSECORE_REGISTER_OP(CpuIPAOp, OpType::InvariantPointAttention, DeviceType::CPU);

}  // namespace
}  // namespace densecore
