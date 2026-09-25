/**
 * @file pair_representation.cpp
 * @brief Pair Representation Learning for Protein Structure (AlphaFold3, ESMFold)
 *
 * Computes pairwise relationships between sequence elements.
 * Core building block for protein structure prediction.
 *
 * ## Complexity
 * - O(N²) memory for pair matrix
 * - CPU advantage: Can handle very long sequences (>2000 residues)
 *   that exceed GPU memory
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace densecore {
namespace {

// ============================================================================
// Pair Representation Parameters
// ============================================================================

struct PairRepresentationParams {
    int num_heads = 8;
    float dropout = 0.0f;
    bool use_bias = true;
};

// ============================================================================
// CpuPairRepresentationOp
// ============================================================================

class CpuPairRepresentationOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;

        const PairRepresentationParams* p = static_cast<const PairRepresentationParams*>(params);
        PairRepresentationParams default_params;
        if (!p) p = &default_params;

        PairRepresentation(*inputs[0], *inputs[1], *inputs[2], *p, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 8 * 1024 * 1024,  // Large L2 helpful
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    /**
     * @brief Compute pair representation from MSA and existing pairs
     *
     * @param msa_repr    [B, N_seq, L, D_msa] - MSA representation
     * @param pair_repr   [B, L, L, D_pair] - Current pair representation
     * @param proj_weight [D_pair, D_msa*2] - Projection weights (left + right)
     * @param params      Configuration
     * @param output      [B, L, L, D_pair] - Updated pair representation
     */
    void PairRepresentation(const Tensor& msa_repr, const Tensor& pair_repr, const Tensor& proj_weight,
                            const PairRepresentationParams& params, Tensor* output) {
        (void)params;
        if (!msa_repr.IsValid() || !pair_repr.IsValid() || !proj_weight.IsValid() || !output) {
            return;
        }

        const int64_t B = pair_repr.shape[0];
        const int64_t L = pair_repr.shape[1];
        const int64_t D_pair = pair_repr.shape[3];
        const int64_t N_seq = msa_repr.shape[1];
        const int64_t D_msa = msa_repr.shape[3];

        const float* msa_data = msa_repr.DataAs<float>();
        const float* pair_data = pair_repr.DataAs<float>();
        const float* proj_data = proj_weight.DataAs<float>();
        float* out_data = output->DataAs<float>();

        // Initialize output with existing pair representation
        std::memcpy(out_data, pair_data, B * L * L * D_pair * sizeof(float));

        // Outer product mean from MSA to pairs
        // For each (i, j) pair, compute mean of outer products across sequences
        std::vector<float> left_proj(D_pair);
        std::vector<float> right_proj(D_pair);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < L; ++i) {
                for (int64_t j = 0; j < L; ++j) {
                    float* pair_ij = out_data + b * L * L * D_pair + i * L * D_pair + j * D_pair;

                    // Average over MSA sequences
                    std::memset(left_proj.data(), 0, D_pair * sizeof(float));
                    std::memset(right_proj.data(), 0, D_pair * sizeof(float));

                    for (int64_t s = 0; s < N_seq; ++s) {
                        const float* msa_i = msa_data + b * N_seq * L * D_msa + s * L * D_msa + i * D_msa;
                        const float* msa_j = msa_data + b * N_seq * L * D_msa + s * L * D_msa + j * D_msa;

                        // Project MSA features to pair space
                        // left_proj += W_left @ msa_i
                        // right_proj += W_right @ msa_j
                        for (int64_t d = 0; d < D_pair; ++d) {
                            const float* w_left = proj_data + d * D_msa * 2;
                            const float* w_right = proj_data + d * D_msa * 2 + D_msa;

                            float sum_left = 0.0f, sum_right = 0.0f;
#if defined(__AVX2__)
                            __m256 vl = _mm256_setzero_ps();
                            __m256 vr = _mm256_setzero_ps();
                            int64_t m = 0;
                            for (; m + 8 <= D_msa; m += 8) {
                                __m256 v_msa_i = _mm256_loadu_ps(msa_i + m);
                                __m256 v_msa_j = _mm256_loadu_ps(msa_j + m);
                                __m256 v_wl = _mm256_loadu_ps(w_left + m);
                                __m256 v_wr = _mm256_loadu_ps(w_right + m);
                                vl = _mm256_fmadd_ps(v_msa_i, v_wl, vl);
                                vr = _mm256_fmadd_ps(v_msa_j, v_wr, vr);
                            }
                            __m128 hi_l = _mm256_extractf128_ps(vl, 1);
                            __m128 lo_l = _mm256_castps256_ps128(vl);
                            __m128 sum_l = _mm_add_ps(lo_l, hi_l);
                            sum_l = _mm_hadd_ps(sum_l, sum_l);
                            sum_l = _mm_hadd_ps(sum_l, sum_l);
                            sum_left = _mm_cvtss_f32(sum_l);

                            __m128 hi_r = _mm256_extractf128_ps(vr, 1);
                            __m128 lo_r = _mm256_castps256_ps128(vr);
                            __m128 sum_r = _mm_add_ps(lo_r, hi_r);
                            sum_r = _mm_hadd_ps(sum_r, sum_r);
                            sum_r = _mm_hadd_ps(sum_r, sum_r);
                            sum_right = _mm_cvtss_f32(sum_r);

                            for (; m < D_msa; ++m) {
                                sum_left += msa_i[m] * w_left[m];
                                sum_right += msa_j[m] * w_right[m];
                            }
#else
                            for (int64_t m = 0; m < D_msa; ++m) {
                                sum_left += msa_i[m] * w_left[m];
                                sum_right += msa_j[m] * w_right[m];
                            }
#endif
                            left_proj[d] += sum_left;
                            right_proj[d] += sum_right;
                        }
                    }

                    // Normalize by sequence count and add outer product contribution
                    float inv_n = 1.0f / static_cast<float>(N_seq);
                    for (int64_t d = 0; d < D_pair; ++d) {
                        // Additive update: pair_ij += left * right (simplified)
                        pair_ij[d] += left_proj[d] * right_proj[d] * inv_n * inv_n;
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuPairRepresentationOp, OpType::PairRepresentation, DeviceType::CPU);

}  // namespace
}  // namespace densecore
