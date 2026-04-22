/**
 * @file optimization_bridge.cpp
 * @brief Runtime SIMD dispatch implementation
 *
 * Populates OpsRegistry with the best available kernel implementations
 * based on runtime CPU feature detection.
 */

#include "densecore/runtime/optimization_bridge.h"

#include <iostream>
#include <mutex>

#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"

namespace densecore {

// =============================================================================
// Scalar Fallback Implementations
// =============================================================================

/**
 * Scalar INT4 GEMM fallback (when AVX512/AVX2 not available at runtime)
 *
 * This is a standalone scalar implementation that's always available,
 * not guarded by __AVX512F__ or __AVX2__ preprocessor.
 */
static void GemmInt4Fp32_Scalar(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                                const float* zero_points, int M, int N, int K, int group_size) {
    if (!C || !A || !W_int4 || !scales || !zero_points || group_size <= 0) return;
    if ((group_size & 1) != 0) return;
    if (K % group_size != 0) return;

    const int num_groups = K / group_size;
    const int packed_k = (K + 1) / 2;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int g = 0; g < num_groups; g++) {
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];

                const int k_start = g * group_size;
                const uint8_t* w_packed = W_int4 + n * packed_k + g * (group_size / 2);

                for (int k = 0; k < group_size; k++) {
                    // Unpack 4-bit weight
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = w_packed[byte_idx];

                    int8_t q;
                    if (nibble_idx == 0) {
                        q = static_cast<int8_t>(packed_byte & 0x0F);
                    } else {
                        q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    }

                    // Sign extend from 4-bit to 8-bit
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    // Dequantize and accumulate
                    float w_dequant = scale * (static_cast<float>(q) - zero);
                    sum += A[m * K + k_start + k] * w_dequant;
                }
            }

            C[m * N + n] = sum;
        }
    }
}

// =============================================================================
// OpsRegistry Initialization
// =============================================================================

void OpsRegistry::Init() {
    static std::once_flag once;
    std::call_once(once, []() {
        auto& reg = Instance();

        // Detect CPU capabilities at runtime
        simd::SimdLevel level = simd::DetectSimdLevel();
        const char* level_name = simd::SimdLevelName(level);

        std::cout << "[OpsRegistry] Detected SIMD level: " << level_name << std::endl;

        // -----------------------------------------------------------------
        // RoPE Dispatch — Highway (portable SIMD, all ISAs)
        // -----------------------------------------------------------------
        reg.RoPE = hwy_kernels::ApplyRoPE_Hwy;
        std::cout << "  [RoPE] -> Highway (auto-dispatch)" << std::endl;

        // -----------------------------------------------------------------
        // FP32 GEMM Dispatch (small-batch split-N)
        // -----------------------------------------------------------------
        reg.GemmFP32 = hwy_kernels::GemmFP32_Hwy;
        std::cout << "  [GemmFP32] -> Highway (split-N)" << std::endl;

        // -----------------------------------------------------------------
        // Batched INT4 GEMM Dispatch
        // -----------------------------------------------------------------
        reg.GemmInt4Batched = hwy_kernels::GemmInt4Batched_Hwy;
        std::cout << "  [GemmInt4Batched] -> Highway (auto-dispatch)" << std::endl;

        // -----------------------------------------------------------------
        // GemmInt4 Dispatch
        // -----------------------------------------------------------------
        // Dispatch chain:
        //   x86:  AVX512 -> AVX2 -> Scalar
        //   ARM:  SVE/SVE2 -> NEON -> Scalar
#if defined(__AVX512F__)
        if (level >= simd::SimdLevel::AVX512) {
            reg.GemmInt4 = simd::GemmInt4Fp32_AVX512;
            std::cout << "  [GemmInt4] -> AVX-512" << std::endl;
        } else if (level >= simd::SimdLevel::AVX2) {
            reg.GemmInt4 = simd::GemmInt4Fp32_AVX2;
            std::cout << "  [GemmInt4] -> AVX2 (runtime: no AVX-512, build has AVX-512)" << std::endl;
        } else {
            reg.GemmInt4 = GemmInt4Fp32_Scalar;
            std::cout << "  [GemmInt4] -> Scalar (runtime: no AVX2)" << std::endl;
        }
#elif defined(__AVX2__)
        if (level >= simd::SimdLevel::AVX2) {
            reg.GemmInt4 = simd::GemmInt4Fp32_AVX2;
            std::cout << "  [GemmInt4] -> AVX2" << std::endl;
        } else {
            reg.GemmInt4 = GemmInt4Fp32_Scalar;
            std::cout << "  [GemmInt4] -> Scalar (runtime: no AVX2)" << std::endl;
        }
#elif defined(__ARM_FEATURE_SVE)
        // SVE available at compile time — use SVE kernel (works on SVE and SVE2)
        if (simd::IsArmFamily(level) && level >= simd::SimdLevel::SVE) {
            reg.GemmInt4 = simd::GemmInt4Fp32_SVE;
            std::cout << "  [GemmInt4] -> ARM SVE" << std::endl;
        } else if (simd::IsArmFamily(level)) {
            reg.GemmInt4 = simd::GemmInt4Fp32_NEON;
            std::cout << "  [GemmInt4] -> ARM NEON (runtime: no SVE)" << std::endl;
        } else {
            reg.GemmInt4 = GemmInt4Fp32_Scalar;
            std::cout << "  [GemmInt4] -> Scalar" << std::endl;
        }
#elif defined(DENSECORE_ARM) || defined(__aarch64__) || defined(_M_ARM64)
        // ARM build without SVE compile support — use NEON
        reg.GemmInt4 = simd::GemmInt4Fp32_NEON;
        std::cout << "  [GemmInt4] -> ARM NEON" << std::endl;
#else
        reg.GemmInt4 = GemmInt4Fp32_Scalar;
        std::cout << "  [GemmInt4] -> Scalar (build without AVX2/AVX-512/ARM)" << std::endl;
#endif

        // -----------------------------------------------------------------
        // Softmax / Dot Dispatch
        // -----------------------------------------------------------------
        reg.Softmax = simd::SoftmaxF32;
        std::cout << "  [Softmax] -> SIMD (internal dispatch)" << std::endl;

        reg.DotF32 = simd::DotF32;
        std::cout << "  [DotF32] -> SIMD (internal dispatch)" << std::endl;

        // Store selected ISA name
        reg.selected_isa = level_name;

        std::cout << "[OpsRegistry] Initialization complete. Using: " << level_name << std::endl;
    });
}

}  // namespace densecore
