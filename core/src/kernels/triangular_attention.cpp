/**
 * @file triangular_attention.cpp
 * @brief Triangular Attention for Protein Structure (AlphaFold2/3)
 *
 * Attention mechanism respecting triangular geometric constraints.
 * Uses axial attention pattern (row-wise or column-wise) on pair matrix.
 *
 * ## Key Insight
 * For pair (i,j) and (i,k), if we know edge (j,k) we can infer
 * geometric constraints - this is the "triangle inequality" in structure.
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include "kernels/hwy/hwy_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace densecore {
namespace {

// ============================================================================
// CpuTriangularAttentionOp
// ============================================================================

class CpuTriangularAttentionOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 4 || outputs.empty()) return;

        const TriangularAttentionParams* p = static_cast<const TriangularAttentionParams*>(params);
        TriangularAttentionParams default_params;
        if (!p) p = &default_params;

        TriangularAttention(*inputs[0], *inputs[1], *inputs[2], *inputs[3], *p, outputs[0]);
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
    static bool CheckedAdd(size_t a, size_t b, size_t* out) {
        if (!out) return false;
        if (a > std::numeric_limits<size_t>::max() - b) return false;
        *out = a + b;
        return true;
    }

    static bool CheckedMul(size_t a, size_t b, size_t* out) {
        if (!out) return false;
        if (a == 0 || b == 0) {
            *out = 0;
            return true;
        }
        if (a > std::numeric_limits<size_t>::max() / b) return false;
        *out = a * b;
        return true;
    }

    static bool FitsSizeT(int64_t v) {
        if (v < 0) return false;
        return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    }

public:
    /**
     * @brief Triangular self-attention on pair representation
     *
     * @param pair_repr [B, L, L, D] - Pair representation
     * @param query_w   [H, D, D_head] - Query projection per head
     * @param key_w     [H, D, D_head] - Key projection per head
     * @param value_w   [H, D, D_head] - Value projection per head
     * @param params    Configuration
     * @param output    [B, L, L, D] - Updated pair representation
     */
    void TriangularAttention(const Tensor& pair_repr, const Tensor& query_w, const Tensor& key_w, const Tensor& value_w,
                             const TriangularAttentionParams& params, Tensor* output) {
        if (!pair_repr.IsValid() || !query_w.IsValid() || !key_w.IsValid() || !value_w.IsValid() || !output) {
            return;
        }

        const int64_t B = pair_repr.shape[0];
        const int64_t L = pair_repr.shape[1];
        const int64_t D = pair_repr.shape[3];
        const int64_t H = params.num_heads;
        if (B <= 0 || L <= 0 || D <= 0 || H <= 0 || (D % H) != 0) {
            return;
        }
        if (H > std::numeric_limits<int>::max()) return;
        const int64_t D_head = D / H;
        if (D_head <= 0) return;

        float scale = params.scale > 0 ? params.scale : 1.0f / std::sqrt(static_cast<float>(D_head));

        const float* pair_data = pair_repr.DataAs<float>();
        const float* qw = query_w.DataAs<float>();
        const float* kw = key_w.DataAs<float>();
        const float* vw = value_w.DataAs<float>();
        float* out_data = output->DataAs<float>();

        const bool has_tile = params.tile_row_start >= 0 && params.tile_row_end > params.tile_row_start &&
                              params.tile_col_start >= 0 && params.tile_col_end > params.tile_col_start;
        if (!has_tile && (output->shape[1] != L || output->shape[2] != L)) {
            return;
        }

        int64_t row_start = 0;
        int64_t row_end = L;
        int64_t col_start = 0;
        int64_t col_end = L;
        bool output_is_tiled = false;
        if (has_tile) {
            row_start = std::max<int64_t>(0, static_cast<int64_t>(params.tile_row_start));
            row_end = std::min<int64_t>(L, static_cast<int64_t>(params.tile_row_end));
            col_start = std::max<int64_t>(0, static_cast<int64_t>(params.tile_col_start));
            col_end = std::min<int64_t>(L, static_cast<int64_t>(params.tile_col_end));
            if (row_end <= row_start || col_end <= col_start) {
                return;
            }

            const int64_t tile_rows = row_end - row_start;
            const int64_t tile_cols = col_end - col_start;
            output_is_tiled = output->shape[1] == tile_rows && output->shape[2] == tile_cols;
            const bool output_is_full = output->shape[1] == L && output->shape[2] == L;
            if (!output_is_tiled && !output_is_full) {
                return;
            }
        }

        if (!FitsSizeT(D_head) || !FitsSizeT(L)) {
            return;
        }

        // Workspace required: (4 * D_head + L) floats
        // q_proj, k_proj, v_proj, attn_accum, scores
        size_t proj_floats = 0;
        size_t workspace_floats = 0;
        size_t workspace_size = 0;
        if (!CheckedMul(static_cast<size_t>(D_head), static_cast<size_t>(4), &proj_floats)) return;
        if (!CheckedAdd(proj_floats, static_cast<size_t>(L), &workspace_floats)) return;
        if (!CheckedMul(workspace_floats, sizeof(float), &workspace_size)) return;
        std::vector<uint8_t> workspace(workspace_size);

        hwy_kernels::TriangularAttentionTiled_Hwy(pair_data, qw, kw, vw, out_data, workspace.data(), workspace_size, B,
                                                  L, D, static_cast<int>(H), scale, params.starting, row_start, row_end,
                                                  col_start, col_end, output_is_tiled);
    }
};

DENSECORE_REGISTER_OP(CpuTriangularAttentionOp, OpType::TriangularAttention, DeviceType::CPU);

}  // namespace
}  // namespace densecore
