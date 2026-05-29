/**
 * @file amx_matmul.cpp
 * @brief Intel AMX-INT8 tile GEMM kernel for CPU prefill matmul
 *
 * Uses AMX TDPBSSD (tile dot product of signed bytes, signed bytes, accumulated
 * to doubleword) for high-throughput INT8 matrix multiplication on Intel
 * Sapphire Rapids and newer (C4 instances).
 *
 * AMX processes 16×64 × 64×16 = 16×16 tiles in a single instruction,
 * computing 16384 INT8 multiply-adds. This is ~16x the throughput of
 * AVX-512 VNNI for large-batch (M>1) prefill matmul.
 *
 * Flow:
 *   1. Quantize FP32 activation matrix A[M,K] to INT8 on-the-fly
 *   2. Load pre-quantized INT8 weights B[K,N] (from Q8 GGUF)
 *   3. Tile-GEMM: C_i32[M,N] = A_i8[M,K] × B_i8[K,N]
 *   4. Dequantize: C_f32[M,N] = C_i32[M,N] * a_scale * b_scale
 *
 * Guard: only compiled on x86_64 with AMX-INT8 support.
 * Runtime CPUID check before use.
 */

#include "densecore/backend/cpu_backend.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include "densecore/simd/simd_platform.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AMX_INT8__) && defined(__AMX_TILE__) && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

namespace densecore {
namespace kernels {
namespace {

// AMX tile configuration: 8 tiles, each up to 16 rows × 64 bytes.
// For INT8 GEMM: tile rows=16, cols=64 (i.e. 64 INT8 elements per row).
//
// Tile layout for C[16,16] += A[16,64] × B[64,16]:
//   tmm0 = accumulator C
//   tmm1 = A tile (16 rows × 64 bytes = 16×64 int8)
//   tmm2 = B tile (16 rows × 64 bytes, transposed: 64×16 int8)

struct TileConfig {
    uint8_t palette_id = 1;
    uint8_t start_row = 0;
    uint8_t reserved[14] = {};
    uint16_t colsb[16] = {};  // columns in bytes per tile
    uint8_t rows[16] = {};    // rows per tile
};
static_assert(sizeof(TileConfig) == 64, "TileConfig must be 64 bytes");

// Configure tiles for the GEMM shape.
// tmm0: C accumulator [M_tile, N_tile] int32, colsb = N_tile * 4
// tmm1: A tile [M_tile, K_tile] int8, colsb = K_tile
// tmm2: B tile [K_tile, N_tile] int8, colsb = N_tile
void ConfigureTilesForGemm(int M_tile, int K_tile, int N_tile) {
    TileConfig config = {};
    config.palette_id = 1;

    // tmm0: accumulator (int32), M_tile rows × N_tile*4 bytes
    config.rows[0] = static_cast<uint8_t>(M_tile);
    config.colsb[0] = static_cast<uint16_t>(N_tile * 4);

    // tmm1: A input (int8), M_tile rows × K_tile bytes
    config.rows[1] = static_cast<uint8_t>(M_tile);
    config.colsb[1] = static_cast<uint16_t>(K_tile);

    // tmm2: B input (int8), K_tile/4 rows × N_tile*4 bytes
    // TDPBSSD expects B in vnni format: [K/4, N, 4]
    config.rows[2] = static_cast<uint8_t>(K_tile / 4);
    config.colsb[2] = static_cast<uint16_t>(N_tile * 4);

    _tile_loadconfig(&config);
}

// Quantize FP32 row to INT8 with per-row scale.
// Returns scale such that original ≈ int8_val * scale.
inline float QuantizeRowToInt8(const float* input, int8_t* output, int K) {
    // Find max absolute value
    float max_abs = 0.0f;
    int i = 0;
#if defined(__AVX512F__)
    __m512 vmax = _mm512_setzero_ps();
    for (; i + 16 <= K; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        vmax = _mm512_max_ps(
            vmax, _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_set1_epi32(0x80000000), _mm512_castps_si512(v))));
    }
    max_abs = _mm512_reduce_max_ps(vmax);
#endif
    for (; i < K; ++i) {
        float a = std::abs(input[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-10f) {
        std::memset(output, 0, static_cast<size_t>(K));
        return 0.0f;
    }

    const float scale = 127.0f / max_abs;
    i = 0;
#if defined(__AVX512F__)
    __m512 v_scale = _mm512_set1_ps(scale);
    for (; i + 16 <= K; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512i vi = _mm512_cvtps_epi32(_mm512_mul_ps(v, v_scale));
        __m128i packed = _mm512_cvtsepi32_epi8(vi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), packed);
    }
#endif
    for (; i < K; ++i) {
        float v = input[i] * scale;
        v = std::max(-127.0f, std::min(127.0f, v));
        output[i] = static_cast<int8_t>(v > 0 ? v + 0.5f : v - 0.5f);
    }

    return max_abs / 127.0f;  // inverse scale for dequantization
}

// AMX-INT8 GEMM: C[M,N] = A[M,K] × B[K,N]
// A is FP32 (quantized on-the-fly), B is already INT8 with per-column scales.
// Output C is FP32.
//
// This handles the core tile loop. Caller handles partitioning.
void AmxInt8GemmTiled(const float* A, int lda,              // [M, K] FP32 activations
                      const int8_t* B_int8, int ldb,        // [K, N] INT8 weights (row-major)
                      const float* b_scales, int N_scales,  // Per-column weight scales
                      float* C, int ldc,                    // [M, N] FP32 output
                      int M, int K, int N) {

    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    // Scratch for quantized A rows
    alignas(64) int8_t a_int8_buf[TILE_M * TILE_K];
    float a_scales[TILE_M];

    // Scratch for B in VNNI layout: [K/4, N, 4]
    alignas(64) int8_t b_vnni_buf[TILE_K * TILE_N];

    // Accumulator scratch
    alignas(64) int32_t c_int32[TILE_M * TILE_N];

    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        const int m_tile = std::min(TILE_M, M - m0);

        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            const int n_tile = std::min(TILE_N, N - n0);

            // Zero accumulator
            std::memset(c_int32, 0, sizeof(c_int32));

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                const int k_tile = std::min(TILE_K, K - k0);

                // Quantize A tile rows
                for (int mi = 0; mi < m_tile; ++mi) {
                    a_scales[mi] = QuantizeRowToInt8(A + (m0 + mi) * lda + k0, a_int8_buf + mi * TILE_K, k_tile);
                    // Zero-pad if k_tile < TILE_K
                    if (k_tile < TILE_K) {
                        std::memset(a_int8_buf + mi * TILE_K + k_tile, 0, static_cast<size_t>(TILE_K - k_tile));
                    }
                }

                // Repack B tile to VNNI layout: [K/4, N, 4]
                // Original B[k,n] → VNNI B[(k/4)*N*4 + n*4 + (k%4)]
                std::memset(b_vnni_buf, 0, sizeof(b_vnni_buf));
                for (int ki = 0; ki < k_tile; ++ki) {
                    for (int ni = 0; ni < n_tile; ++ni) {
                        b_vnni_buf[(ki / 4) * (TILE_N * 4) + ni * 4 + (ki % 4)] = B_int8[(k0 + ki) * ldb + (n0 + ni)];
                    }
                }

                // Configure and execute AMX tiles
                if (m_tile == TILE_M && k_tile == TILE_K && n_tile == TILE_N) {
                    ConfigureTilesForGemm(TILE_M, TILE_K, TILE_N);

                    _tile_loadd(0, c_int32, TILE_N * 4);     // tmm0 = C accumulator
                    _tile_loadd(1, a_int8_buf, TILE_K);      // tmm1 = A
                    _tile_loadd(2, b_vnni_buf, TILE_N * 4);  // tmm2 = B (VNNI format)
                    _tile_dpbssd(0, 1, 2);                   // tmm0 += tmm1 × tmm2
                    _tile_stored(0, c_int32, TILE_N * 4);    // Store result

                    _tile_release();
                } else {
                    // Scalar fallback for edge tiles
                    for (int mi = 0; mi < m_tile; ++mi) {
                        for (int ni = 0; ni < n_tile; ++ni) {
                            int32_t sum = 0;
                            for (int ki = 0; ki < k_tile; ++ki) {
                                sum += static_cast<int32_t>(a_int8_buf[mi * TILE_K + ki]) *
                                       static_cast<int32_t>(B_int8[(k0 + ki) * ldb + (n0 + ni)]);
                            }
                            c_int32[mi * TILE_N + ni] += sum;
                        }
                    }
                }
            }

            // Dequantize and write to output
            for (int mi = 0; mi < m_tile; ++mi) {
                const float a_dequant = a_scales[mi];
                for (int ni = 0; ni < n_tile; ++ni) {
                    const float b_dequant = (n0 + ni < N_scales) ? b_scales[n0 + ni] : 1.0f;
                    C[(m0 + mi) * ldc + (n0 + ni)] =
                        static_cast<float>(c_int32[mi * TILE_N + ni]) * a_dequant * b_dequant;
                }
            }
        }
    }
}

// OpRegistry wrapper for AMX INT8 MatMul
class CpuAmxInt8MatMulOp final : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        // This op is invoked by the HAL when AMX is available and the operand
        // types match (FP32 activation × Q8 weight → FP32 output).
        if (inputs.size() < 2 || outputs.empty()) return;
        const Tensor* A = inputs[0];  // [M, K] FP32 activations
        const Tensor* B = inputs[1];  // [K, N] INT8 weights (pre-quantized)
        Tensor* C = outputs[0];       // [M, N] FP32 output

        if (!A || !B || !C) return;
        if (!A->IsValid() || !B->IsValid() || !C->IsValid()) return;

        const int M = static_cast<int>(A->shape[0]);
        const int K = static_cast<int>(A->shape[1]);
        const int N = static_cast<int>(B->shape[1]);

        // Only use AMX for large prefill matmuls (M > 1)
        if (M <= 1) return;

        const float* a_data = A->DataAs<float>();
        const int8_t* b_data = B->DataAs<int8_t>();
        float* c_data = C->DataAs<float>();

        // Use uniform scale of 1.0 if no per-column scales provided
        std::vector<float> default_scales(static_cast<size_t>(N), 1.0f);
        const float* b_scales = default_scales.data();

        if (inputs.size() > 2 && inputs[2] && inputs[2]->IsValid()) {
            b_scales = inputs[2]->DataAs<float>();
        }

        AmxInt8GemmTiled(a_data, K, b_data, N, b_scales, N, c_data, N, M, K, N);
    }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 8 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 10};  // Higher priority than generic CPU MatMul
    }
};

DENSECORE_REGISTER_OP(CpuAmxInt8MatMulOp, OpType::MatMul, DeviceType::CPU);

}  // namespace
}  // namespace kernels
}  // namespace densecore

#endif  // __AMX_INT8__ && __AMX_TILE__
