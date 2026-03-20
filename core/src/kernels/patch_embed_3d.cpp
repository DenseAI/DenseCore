/**
 * @file patch_embed_3d.cpp
 * @brief 3D Patch Embedding for Video Transformers (VideoMAE, SORA, Video DiT)
 *
 * Converts video input [B, C, T, H, W] to patch tokens [B, N, D]
 * where N = (T/Pt) * (H/Ph) * (W/Pw)
 *
 * ## Optimizations
 * - 3D Im2Col for efficient patch extraction
 * - AVX2/AVX-512 GEMM for projection
 * - Cache-friendly memory access patterns
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace densecore {
namespace {

struct PatchEmbed3DParams {
    int in_channels = 0;
    int embed_dim = 0;
    int temporal_patch = 0;
    int patch_h = 0;
    int patch_w = 0;
};

// ============================================================================
// Im2Col 3D: Extract video patches into matrix
// ============================================================================
// Input:  video[C, T, H, W]
// Output: col[N, K] where N = num_patches, K = C * Pt * Ph * Pw

static void Im2Col3D(const float* video, int64_t C, int64_t T, int64_t H, int64_t W, int64_t Pt, int64_t Ph,
                     int64_t Pw,  // patch sizes
                     int64_t num_t, int64_t num_h, int64_t num_w, float* col) {
    const int64_t K = C * Pt * Ph * Pw;

    for (int64_t pt = 0; pt < num_t; ++pt) {
        for (int64_t ph = 0; ph < num_h; ++ph) {
            for (int64_t pw = 0; pw < num_w; ++pw) {
                const int64_t patch_idx = pt * num_h * num_w + ph * num_w + pw;
                float* patch_out = col + patch_idx * K;

                const int64_t t_start = pt * Pt;
                const int64_t h_start = ph * Ph;
                const int64_t w_start = pw * Pw;

                // Flatten: C -> T -> H -> W
                for (int64_t c = 0; c < C; ++c) {
                    const float* vol_channel = video + c * T * H * W;
                    for (int64_t kt = 0; kt < Pt; ++kt) {
                        for (int64_t kh = 0; kh < Ph; ++kh) {
                            const float* row = vol_channel + (t_start + kt) * H * W + (h_start + kh) * W + w_start;
                            for (int64_t kw = 0; kw < Pw; ++kw) {
                                *patch_out++ = row[kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GEMM: [N, K] x [D, K]^T = [N, D]
// ============================================================================

#if defined(__AVX2__)
static void Gemm_AVX2(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
                      float* Out) {
    for (int64_t n = 0; n < N; ++n) {
        const float* col_row = Col + n * K;
        float* out_row = Out + n * D;

        for (int64_t d = 0; d < D; ++d) {
            const float* w_row = W + d * K;

            __m256 sum_vec = _mm256_setzero_ps();
            int64_t k = 0;

            for (; k + 8 <= K; k += 8) {
                __m256 v_col = _mm256_loadu_ps(col_row + k);
                __m256 v_w = _mm256_loadu_ps(w_row + k);
                sum_vec = _mm256_fmadd_ps(v_col, v_w, sum_vec);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            for (; k < K; ++k) {
                sum += col_row[k] * w_row[k];
            }

            if (Bias) sum += Bias[d];
            out_row[d] = sum;
        }
    }
}
#endif

[[maybe_unused]] static void Gemm_Scalar(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D,
                                         int64_t K, float* Out) {
    for (int64_t n = 0; n < N; ++n) {
        const float* col_row = Col + n * K;
        float* out_row = Out + n * D;

        for (int64_t d = 0; d < D; ++d) {
            const float* w_row = W + d * K;
            float sum = 0.0f;

            for (int64_t k = 0; k < K; ++k) {
                sum += col_row[k] * w_row[k];
            }

            if (Bias) sum += Bias[d];
            out_row[d] = sum;
        }
    }
}

static void Gemm_Dispatch(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
                          float* Out) {
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(N), static_cast<int>(D), static_cast<int>(K),
                1.0f, Col, static_cast<int>(K), W, static_cast<int>(K), 0.0f, Out, static_cast<int>(D));
#pragma clang diagnostic pop
    if (Bias) {
        for (int64_t n = 0; n < N; ++n) {
            vDSP_vadd(Out + n * D, 1, Bias, 1, Out + n * D, 1, static_cast<vDSP_Length>(D));
        }
    }
#elif defined(__AVX2__)
    Gemm_AVX2(Col, W, Bias, N, D, K, Out);
#else
    Gemm_Scalar(Col, W, Bias, N, D, K, Out);
#endif
}

// ============================================================================
// CpuPatchEmbed3DOp
// ============================================================================

class CpuPatchEmbed3DOp : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.size() < 2 || outputs.empty()) return;

        const PatchEmbed3DParams* p = static_cast<const PatchEmbed3DParams*>(params);
        PatchEmbed3DParams default_params;
        if (!p) p = &default_params;
        const Tensor* bias = (inputs.size() >= 3) ? inputs[2] : nullptr;
        PatchEmbed3D(*inputs[0], *inputs[1], bias, *p, outputs[0]);
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
     * @brief 3D Patch Embedding
     *
     * @param video       [B, C, T, H, W] - Video input
     * @param conv_weight [D, C, Pt, Ph, Pw] - 3D Conv weights
     * @param conv_bias   [D] - Optional bias
     * @param patches     [B, N, D] - Output patch embeddings
     */
    void PatchEmbed3D(const Tensor& video, const Tensor& conv_weight, const Tensor* conv_bias,
                      const PatchEmbed3DParams& params, Tensor* patches) {
        if (!video.IsValid() || !conv_weight.IsValid() || !patches) return;

        int64_t B = 0;
        int64_t C = 0;
        int64_t T = 0;
        int64_t H = 0;
        int64_t W = 0;
        if (video.ndim == 5) {
            B = video.shape[0];
            C = video.shape[1];
            T = video.shape[2];
            H = video.shape[3];
            W = video.shape[4];
        } else if (video.ndim == 4 && params.in_channels > 0) {
            B = video.shape[0];
            C = params.in_channels;
            H = video.shape[2];
            W = video.shape[3];
            if (video.shape[1] % C != 0) {
                return;
            }
            T = video.shape[1] / C;
        } else {
            return;
        }

        int64_t D = 0;
        int64_t Pt = 0;
        int64_t Ph = 0;
        int64_t Pw = 0;
        if (conv_weight.ndim == 5) {
            D = conv_weight.shape[0];
            Pt = conv_weight.shape[2];
            Ph = conv_weight.shape[3];
            Pw = conv_weight.shape[4];
        } else if (conv_weight.ndim == 2 && params.temporal_patch > 0 && params.patch_h > 0 && params.patch_w > 0) {
            D = conv_weight.shape[0];
            Pt = params.temporal_patch;
            Ph = params.patch_h;
            Pw = params.patch_w;
        } else {
            return;
        }

        if (Pt == 0 || Ph == 0 || Pw == 0) return;

        const int64_t num_t = T / Pt;
        const int64_t num_h = H / Ph;
        const int64_t num_w = W / Pw;
        const int64_t N = num_t * num_h * num_w;  // total patches
        const int64_t K = C * Pt * Ph * Pw;       // flattened patch size
        if (conv_weight.ndim == 2 && conv_weight.shape[1] != K) {
            return;
        }

        const float* video_data = video.DataAs<float>();
        const float* weight_data = conv_weight.DataAs<float>();
        const float* bias_data = conv_bias && conv_bias->IsValid() ? conv_bias->DataAs<float>() : nullptr;
        float* out_data = patches->DataAs<float>();

        // Scratch buffer for Im2Col
        std::vector<float> col_buffer(N * K);

        for (int64_t b = 0; b < B; ++b) {
            const float* video_b = video_data + b * C * T * H * W;
            float* out_b = out_data + b * N * D;

            // Step 1: Im2Col 3D
            Im2Col3D(video_b, C, T, H, W, Pt, Ph, Pw, num_t, num_h, num_w, col_buffer.data());

            // Step 2: GEMM
            Gemm_Dispatch(col_buffer.data(), weight_data, bias_data, N, D, K, out_b);
        }
    }
};

DENSECORE_REGISTER_OP(CpuPatchEmbed3DOp, OpType::PatchEmbed3D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
