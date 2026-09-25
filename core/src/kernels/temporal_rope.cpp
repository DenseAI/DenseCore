/**
 * @file temporal_rope.cpp
 * @brief Temporal RoPE for Video/Autonomous Driving Transformers
 *
 * 3D Rotary Position Embedding for spatio-temporal sequences.
 * Extends RoPE to handle temporal dimension in video models.
 *
 * ## Use Cases
 * - Video DiT (temporal attention)
 * - Autonomous driving (temporal fusion)
 * - 4D scene understanding
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
// Temporal RoPE Parameters
// ============================================================================

struct TemporalRoPEParams {
    int temporal_dim = 0;      // Dimensions for temporal encoding
    int spatial_dim = 0;       // Dimensions for spatial encoding
    float theta = 10000.0f;    // Base frequency
    bool interleaved = false;  // Interleave temporal/spatial
};

// ============================================================================
// CpuTemporalRoPEOp
// ============================================================================

class CpuTemporalRoPEOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 3 || outputs.empty()) return;

        const TemporalRoPEParams* p = static_cast<const TemporalRoPEParams*>(params);
        TemporalRoPEParams default_params;
        if (!p) p = &default_params;

        TemporalRoPE(*inputs[0], *inputs[1], *inputs[2], *p, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};
    }

    /**
     * @brief Apply Temporal + Spatial RoPE
     *
     * @param input        [B, T, N, H, D] - Input (T=frames, N=spatial, H=heads, D=head_dim)
     * @param temporal_pos [T] - Temporal position indices
     * @param spatial_pos  [N, 2] - Spatial positions (h, w)
     * @param params       RoPE configuration
     * @param output       [B, T, N, H, D] - Output with position encoding
     */
    void TemporalRoPE(const Tensor& input, const Tensor& temporal_pos, const Tensor& spatial_pos,
                      const TemporalRoPEParams& params, Tensor* output) {
        if (!input.IsValid() || !temporal_pos.IsValid() || !spatial_pos.IsValid() || !output) {
            return;
        }

        const int64_t B = input.shape[0];
        const int64_t T = input.shape[1];
        const int64_t N = input.shape[2];
        const int64_t H = input.shape[3];
        const int64_t D = input.shape[4];

        const float* in_data = input.DataAs<float>();
        const float* t_pos = temporal_pos.DataAs<float>();
        const float* s_pos = spatial_pos.DataAs<float>();
        float* out_data = output->DataAs<float>();

        // Determine dimension split
        int64_t t_dim = params.temporal_dim > 0 ? params.temporal_dim : D / 4;
        int64_t s_dim = params.spatial_dim > 0 ? params.spatial_dim : D / 2;

        // Precompute frequency bands
        std::vector<float> t_freqs(t_dim / 2);
        std::vector<float> s_freqs(s_dim / 2);

        for (int64_t i = 0; i < t_dim / 2; ++i) {
            t_freqs[i] = 1.0f / std::pow(params.theta, 2.0f * i / t_dim);
        }
        for (int64_t i = 0; i < s_dim / 2; ++i) {
            s_freqs[i] = 1.0f / std::pow(params.theta, 2.0f * i / s_dim);
        }

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                float t_idx = t_pos[t];

                for (int64_t n = 0; n < N; ++n) {
                    float h_idx = s_pos[n * 2 + 0];
                    float w_idx = s_pos[n * 2 + 1];

                    for (int64_t h = 0; h < H; ++h) {
                        const float* inp = in_data + ((b * T + t) * N + n) * H * D + h * D;
                        float* outp = out_data + ((b * T + t) * N + n) * H * D + h * D;

                        int64_t d = 0;

                        // Temporal RoPE (first t_dim dimensions)
                        for (int64_t i = 0; i < t_dim / 2 && d + 1 < D; ++i, d += 2) {
                            float angle = t_idx * t_freqs[i];
                            float cos_a = std::cos(angle);
                            float sin_a = std::sin(angle);

                            float x0 = inp[d];
                            float x1 = inp[d + 1];
                            outp[d] = x0 * cos_a - x1 * sin_a;
                            outp[d + 1] = x0 * sin_a + x1 * cos_a;
                        }

                        // Spatial RoPE - Height (next s_dim/2 dimensions)
                        for (int64_t i = 0; i < s_dim / 4 && d + 1 < D; ++i, d += 2) {
                            float angle = h_idx * s_freqs[i];
                            float cos_a = std::cos(angle);
                            float sin_a = std::sin(angle);

                            float x0 = inp[d];
                            float x1 = inp[d + 1];
                            outp[d] = x0 * cos_a - x1 * sin_a;
                            outp[d + 1] = x0 * sin_a + x1 * cos_a;
                        }

                        // Spatial RoPE - Width (next s_dim/2 dimensions)
                        for (int64_t i = 0; i < s_dim / 4 && d + 1 < D; ++i, d += 2) {
                            float angle = w_idx * s_freqs[s_dim / 4 + i];
                            float cos_a = std::cos(angle);
                            float sin_a = std::sin(angle);

                            float x0 = inp[d];
                            float x1 = inp[d + 1];
                            outp[d] = x0 * cos_a - x1 * sin_a;
                            outp[d + 1] = x0 * sin_a + x1 * cos_a;
                        }

                        // Copy remaining dimensions unchanged
                        for (; d < D; ++d) {
                            outp[d] = inp[d];
                        }
                    }
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuTemporalRoPEOp, OpType::TemporalRoPE, DeviceType::CPU);

}  // namespace
}  // namespace densecore
