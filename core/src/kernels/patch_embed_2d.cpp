/**
 * @file patch_embed_2d.cpp
 * @brief Optimized CPU PatchEmbed2D using Im2Col (N,K) + GEMM
 *
 * ## Performance Optimization
 *
 * Strategy: "Zero-Copy Im2Col + Fused GEMM"
 *
 * 1. Im2Col (Row-Major Patches):
 *    - Extract patches into a matrix [N, K] where N=num_patches, K=C*P*P.
 *    - This is cache-friendly for the subsequent GEMM (sequential access).
 *
 * 2. Fused GEMM (No Transpose):
 *    - Input:  Col [N, K]
 *    - Weight: W   [D, K] (Standard Conv2D layout: OutCh, InCh, KH, KW)
 *    - Output: Out [N, D]
 *    - Compute: Out = Col x W^T
 *    - Since W is already [D, K], we compute dot products of rows of Col and W.
 *    - Result is naturally [N, D] (patches, embed_dim), which is the target transformer layout.
 *    - ELIMINATES the explicit transpose step from previous implementations.
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

// ============================================================================
// Im2Col: Extract image patches into row-major matrix [N, K]
// ============================================================================
// Input:  image[C, H, W]
// Output: col[N, K] where N = num_patches, K = C * P * P
//
// Optimized for contiguous writes to 'col'.
// ============================================================================

static void Im2Col_NK(const float* image, int64_t C, int64_t H, int64_t W,
                      int64_t P,  // patch size (square)
                      int64_t num_patches_h, int64_t num_patches_w, float* col) {
    const int64_t K = C * P * P;

    // Iterate over patches (N dimension)
    for (int64_t ph = 0; ph < num_patches_h; ++ph) {
        for (int64_t pw = 0; pw < num_patches_w; ++pw) {
            // Output pointer for this patch (points to start of row in [N, K])
            float* patch_out = col + (ph * num_patches_w + pw) * K;

            const int64_t h_start = ph * P;
            const int64_t w_start = pw * P;

            // Flatten the patch: Channels -> Height -> Width
            for (int64_t c = 0; c < C; ++c) {
                const float* img_channel = image + c * H * W;
                for (int64_t kh = 0; kh < P; ++kh) {
                    const float* img_row = img_channel + (h_start + kh) * W + w_start;
                    // Copy row of pixels
                    // Optimize: Use memcpy for larger patches if needed, but simple loop allows compiler auto-vectorization
                    for (int64_t kw = 0; kw < P; ++kw) {
                        *patch_out++ = img_row[kw];
                    }
                }
            }
        }
    }
}

// ============================================================================
// GEMM: [N, K] x [D, K]^T = [N, D]
// ============================================================================
// Computes dot product of each patch vector (row in Col) with each filter (row in W).
// Adds bias if present.
// ============================================================================

#if defined(__AVX512F__)

static void Gemm_NK_DK_AVX512(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
                              float* Out) {
    // Col: [N, K]
    // W:   [D, K]
    // Out: [N, D]

    // Parallelize over N (patches) and D (output dims)
    // Here we use a simple blocked loop. For max performance, block D and N to fit in registers.

    // Simple implementation: Loop N, then D, then K (dot product)
    for (int64_t n = 0; n < N; ++n) {
        const float* col_row = Col + n * K;
        float* out_row = Out + n * D;

        for (int64_t d = 0; d < D; ++d) {
            const float* w_row = W + d * K;

            __m512 sum_vec = _mm512_setzero_ps();
            int64_t k = 0;

            // Unroll K loop
            for (; k + 16 <= K; k += 16) {
                __m512 v_col = _mm512_loadu_ps(col_row + k);
                __m512 v_w = _mm512_loadu_ps(w_row + k);
                sum_vec = _mm512_fmadd_ps(v_col, v_w, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            // Handle tail
            for (; k < K; ++k) {
                sum += col_row[k] * w_row[k];
            }

            if (Bias) {
                sum += Bias[d];
            }
            out_row[d] = sum;
        }
    }
}

#endif

#if defined(__AVX2__)

static void Gemm_NK_DK_AVX2(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
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
#if defined(__FMA__)
                sum_vec = _mm256_fmadd_ps(v_col, v_w, sum_vec);
#else
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v_col, v_w));
#endif
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);  // (a0+a1, a2+a3, ...)
            sum128 = _mm_hadd_ps(sum128, sum128);  // (Sum, Sum, ...)
            float sum = _mm_cvtss_f32(sum128);

            for (; k < K; ++k) {
                sum += col_row[k] * w_row[k];
            }

            if (Bias) {
                sum += Bias[d];
            }
            out_row[d] = sum;
        }
    }
}

#endif

static void Gemm_NK_DK_Scalar(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
                              float* Out) {
    for (int64_t n = 0; n < N; ++n) {
        const float* col_row = Col + n * K;
        float* out_row = Out + n * D;

        for (int64_t d = 0; d < D; ++d) {
            const float* w_row = W + d * K;
            float sum = 0.0f;

            for (int64_t k = 0; k < K; ++k) {
                sum += col_row[k] * w_row[k];
            }

            if (Bias) {
                sum += Bias[d];
            }
            out_row[d] = sum;
        }
    }
}

static void Gemm_Dispatch(const float* Col, const float* W, const float* Bias, int64_t N, int64_t D, int64_t K,
                          float* Out) {
#ifdef __APPLE__
    // Accelerate (cblas_sgemm)
    // Compute C = A * B
    // A = Col [N, K] (RowMajor)
    // B = W^T [K, D]. But W is [D, K] (RowMajor).
    // To treat W as [K, D], we say W is Transposed.
    // Result C is [N, D] (RowMajor)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(N), static_cast<int>(D), static_cast<int>(K),
                1.0f, Col, static_cast<int>(K), W, static_cast<int>(K),  // ldb is K because W is [D, K]
                0.0f, Out, static_cast<int>(D));

    if (Bias) {
        // Add bias vector [D] to every row of Out [N, D]
        // This can also be vectorized, but keeping it simple for now
        for (int64_t n = 0; n < N; ++n) {
            float* row = Out + n * D;
            vDSP_vadd(row, 1, Bias, 1, row, 1, static_cast<vDSP_Length>(D));
        }
    }
#elif defined(__AVX512F__)
    Gemm_NK_DK_AVX512(Col, W, Bias, N, D, K, Out);
#elif defined(__AVX2__)
    Gemm_NK_DK_AVX2(Col, W, Bias, N, D, K, Out);
#else
    Gemm_NK_DK_Scalar(Col, W, Bias, N, D, K, Out);
#endif
}

// ============================================================================
// CpuPatchEmbed2DOp: Optimized im2col + GEMM implementation
// ============================================================================

class CpuPatchEmbed2DOp : public EmbeddingOps {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        (void)params;
        if (inputs.size() < 2 || outputs.empty()) {
            return;
        }

        Tensor bias;
        if (inputs.size() >= 3 && inputs[2]) {
            bias = *inputs[2];
        }

        PatchEmbed2D(*inputs[0], *inputs[1], bias, outputs[0]);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};  // High priority
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::NCHW || layout == TensorLayout::UNKNOWN;
    }

    void PatchEmbed2D(const Tensor& image, const Tensor& conv_weight, const Tensor& conv_bias,
                      Tensor* patches) override {
        if (!image.IsValid() || !conv_weight.IsValid() || !patches) {
            return;
        }

        const int64_t B = image.shape[0];
        const int64_t C = image.shape[1];
        const int64_t H = image.shape[2];
        const int64_t W = image.shape[3];

        const int64_t D = conv_weight.shape[0];  // embed_dim
        const int64_t P = conv_weight.shape[2];  // patch_size

        if (P == 0) return;

        const int64_t num_patches_h = H / P;
        const int64_t num_patches_w = W / P;
        const int64_t N = num_patches_h * num_patches_w;  // total patches per image
        const int64_t K = C * P * P;                      // flattened patch size

        const float* img_data = image.DataAs<float>();
        const float* weight_data = conv_weight.DataAs<float>();
        const float* bias_data = conv_bias.IsValid() ? conv_bias.DataAs<float>() : nullptr;
        float* out_data = patches->DataAs<float>();

        // Scratch buffer for Im2Col [N, K]
        // This buffer is large: N * K floats.
        // For ViT-Base (224x224, P=16): N=196, K=3*16*16=768. 196*768*4 bytes = ~600KB. Safe.
        // For larger images (e.g. 1024x1024), this grows.
        // Optimization: We can process in blocks of N (e.g. 256 patches at a time) to stay in L2 cache.
        // For now, simple full allocation.
        std::vector<float> col_buffer(N * K);

        // Process each batch
        for (int64_t b = 0; b < B; ++b) {
            const float* img_b = img_data + b * (C * H * W);
            float* out_b = out_data + b * (N * D);

            // Step 1: Im2Col - extract patches into [N, K] matrix (row-major)
            Im2Col_NK(img_b, C, H, W, P, num_patches_h, num_patches_w, col_buffer.data());

            // Step 2: Fused GEMM - [N, K] x [D, K]^T = [N, D]
            // Direct computation, no transpose needed.
            Gemm_Dispatch(col_buffer.data(), weight_data, bias_data, N, D, K, out_b);
        }
    }

    void RoPE2D(const Tensor& input, const Tensor& pos_h, const Tensor& pos_w, const Tensor& cos_sin,
                Tensor* output) override {
        // Not implemented here
        (void)input;
        (void)pos_h;
        (void)pos_w;
        (void)cos_sin;
        (void)output;
    }
};

DENSECORE_REGISTER_OP(CpuPatchEmbed2DOp, OpType::PatchEmbed2D, DeviceType::CPU);

}  // namespace
}  // namespace densecore
