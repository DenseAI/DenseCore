/**
 * @file audio_conv1d.cpp
 * @brief 1D Convolution for Audio Processing (Whisper Frontend)
 *
 * Optimized 1D convolution for mel spectrogram processing.
 * Used in Whisper's audio encoder frontend.
 *
 * ## Features
 * - SIMD-optimized convolution
 * - Stride and padding support
 * - GELU activation fusion option
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"
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
// CpuAudioConv1DOp
// ============================================================================

class CpuAudioConv1DOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const AudioConv1DParams* p = static_cast<const AudioConv1DParams*>(params);
        AudioConv1DParams default_params;
        if (!p) p = &default_params;

        const Tensor* bias = (inputs.size() >= 3) ? inputs[2] : nullptr;
        Conv1D(*inputs[0], *inputs[1], bias, *p, outputs[0]);
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
     * @brief 1D Convolution
     *
     * @param input   [B, C_in, L_in] - Input audio/features
     * @param weight  [C_out, C_in, K] - Convolution weights
     * @param bias    [C_out] - Optional bias
     * @param params  Conv1D parameters (stride, padding)
     * @param output  [B, C_out, L_out] - Output features
     */
    void Conv1D(const Tensor& input, const Tensor& weight, const Tensor* bias, const AudioConv1DParams& params,
                Tensor* output) {
        if (!input.IsValid() || !weight.IsValid() || !output) return;

        const int64_t B = input.shape[0];
        const int64_t C_in = input.shape[1];
        const int64_t L_in = input.shape[2];

        const int64_t C_out = weight.shape[0];
        const int64_t K = weight.shape[2];
        const int64_t stride = params.stride;
        const int64_t padding = params.padding;

        const int64_t L_out = (L_in + 2 * padding - K) / stride + 1;

        const float* in_data = input.DataAs<float>();
        const float* w_data = weight.DataAs<float>();
        const float* b_data = bias && bias->IsValid() ? bias->DataAs<float>() : nullptr;
        float* out_data = output->DataAs<float>();

        std::memset(out_data, 0, B * C_out * L_out * sizeof(float));

        // Convolution: for each output position
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t co = 0; co < C_out; ++co) {
                const float* w_co = w_data + co * C_in * K;
                float* out_co = out_data + b * C_out * L_out + co * L_out;

                for (int64_t l_out = 0; l_out < L_out; ++l_out) {
                    const int64_t l_start = l_out * stride - padding;
                    float sum = 0.0f;

                    for (int64_t ci = 0; ci < C_in; ++ci) {
                        const float* in_ci = in_data + b * C_in * L_in + ci * L_in;
                        const float* w_ci = w_co + ci * K;

#if defined(__AVX2__)
                        __m256 sum_vec = _mm256_setzero_ps();
                        int64_t k = 0;

                        for (; k + 8 <= K; k += 8) {
                            int64_t l_in = l_start + k;

                            // Load input with bounds checking
                            float in_vals[8];
                            for (int i = 0; i < 8; ++i) {
                                int64_t idx = l_in + i;
                                in_vals[i] = (idx >= 0 && idx < L_in) ? in_ci[idx] : 0.0f;
                            }
                            __m256 v_in = _mm256_loadu_ps(in_vals);
                            __m256 v_w = _mm256_loadu_ps(w_ci + k);
                            sum_vec = _mm256_fmadd_ps(v_in, v_w, sum_vec);
                        }

                        // Horizontal sum
                        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                        __m128 lo = _mm256_castps256_ps128(sum_vec);
                        __m128 sum128 = _mm_add_ps(lo, hi);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        sum += _mm_cvtss_f32(sum128);

                        // Remainder
                        for (; k < K; ++k) {
                            int64_t l_in = l_start + k;
                            if (l_in >= 0 && l_in < L_in) {
                                sum += in_ci[l_in] * w_ci[k];
                            }
                        }
#else
                        for (int64_t k = 0; k < K; ++k) {
                            int64_t l_in = l_start + k;
                            if (l_in >= 0 && l_in < L_in) {
                                sum += in_ci[l_in] * w_ci[k];
                            }
                        }
#endif
                    }

                    if (b_data) {
                        sum += b_data[co];
                    }
                    out_co[l_out] = sum;
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuAudioConv1DOp, OpType::AudioConv1D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
