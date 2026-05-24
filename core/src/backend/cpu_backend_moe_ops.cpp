#include "backend/cpu_backend_internal.h"
#include "ggml-cpu.h"  // For ggml_get_type_traits_cpu (vec_dot)
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/q4k_repacked_gemv.h"
#include "runtime/runtime_env.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace densecore {

namespace {

struct AlignedScratch {
    float* ptr = nullptr;
    size_t capacity = 0;

    ~AlignedScratch() {
        if (ptr) {
            free(ptr);
        }
    }

    void Resize(CpuBackend* b, size_t required) {
        if (required > capacity) {
            if (ptr) free(ptr);
            ptr = static_cast<float*>(b->AllocateDevice(required * sizeof(float)));
            capacity = required;
        }
    }

    bool HasCapacity(size_t required) const { return ptr != nullptr && required <= capacity; }
};

// Qwen3.5-35B-A3B: 256 experts x top-8 -> batch=4 yields 32 assignments
constexpr int kSmallDecodeMaxAssignments = 32;
constexpr int kSmallDecodeMaxSnapshotExperts = 512;

struct MoEInt4PathHistogram {
    std::atomic<uint64_t> direct_hwy{0};
    std::atomic<uint64_t> fused_swiglu_hwy{0};
    std::atomic<uint64_t> backend_gemm{0};
    std::atomic<uint64_t> f32_fallback{0};
};

MoEInt4PathHistogram& GetMoEInt4PathHistogram() {
    static MoEInt4PathHistogram histogram;
    return histogram;
}

struct QuantizedProjectionInputCache {
    const float* source = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    ggml_type type = GGML_TYPE_COUNT;
    size_t row_bytes = 0;
    std::vector<uint8_t> bytes;
};

constexpr int64_t kMoEQuantizedProjectionMaxBatch = 256;
constexpr int kMoEQ4KRawBatchedTileM = 8;

inline float GeluTanhApprox(float x);
const std::array<float, 1 << 16>& GetGeluF16LookupTable();

struct MoEBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(MoEBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "MoE Q8_K block layout must match ggml block_q8_K");

struct MoEBlockQ8Kx4 {
    float d[4];
    int8_t qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
};
static_assert(sizeof(MoEBlockQ8Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t),
              "MoE Q8_Kx4 block layout must match ggml block_q8_Kx4");

bool ComputeMoEQ4KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                      int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQ4KRawBatchedTileM || K <= 0 ||
        (K % QK_K) != 0) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    float lane_acc[kMoEQ4KRawBatchedTileM][8];
    float min_acc[kMoEQ4KRawBatchedTileM];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q4[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q4_blocks[bi];
        const uint8_t* q4 = xb.qs;
        int8_t* uq4 = unpacked_q4;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) {
                uq4[l] = static_cast<int8_t>(q4[l] & 0xF);
            }
            uq4 += 32;
            for (int l = 0; l < 32; ++l) {
                uq4[l] = static_cast<int8_t>(q4[l] >> 4);
            }
            uq4 += 32;
            q4 += 32;
        }

        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float q4_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q4_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q4;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]);
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float d = q4_d * yb.d;
            const float dmin = q4_dmin * yb.d;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        float sum = min_acc[m];
        for (int l = 0; l < 8; ++l) {
            sum += lane_acc[m][l];
        }
        out_sums[m] = sum;
    }
    return true;
}

#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
bool ComputeMoEQ4KQ8KBatchedRowDotprod(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride,
                                       int M, int K, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums || M <= 0 || M > kMoEQ4KRawBatchedTileM || K <= 0 ||
        (K % QK_K) != 0) {
        return false;
    }
    const int nb = K / QK_K;
    if (quant_row_stride < sizeof(MoEBlockQ8K) * static_cast<size_t>(nb)) {
        return false;
    }

    std::fill(out_sums, out_sums + M, 0.0f);
    const auto* q4_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = q4_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const auto* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const auto* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float q4_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float q4_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        int8x16_t q4_vec[QK_K / 32][2];
        const uint8_t* q4 = xb.qs;
        for (int chunk = 0; chunk < QK_K / 64; ++chunk) {
            const uint8x16_t packed0 = vld1q_u8(q4);
            const uint8x16_t packed1 = vld1q_u8(q4 + 16);
            q4 += 32;
            q4_vec[2 * chunk][0] = vreinterpretq_s8_u8(vandq_u8(packed0, low_mask));
            q4_vec[2 * chunk][1] = vreinterpretq_s8_u8(vandq_u8(packed1, low_mask));
            q4_vec[2 * chunk + 1][0] = vreinterpretq_s8_u8(vshrq_n_u8(packed0, 4));
            q4_vec[2 * chunk + 1][1] = vreinterpretq_s8_u8(vshrq_n_u8(packed1, 4));
        }

        for (int m = 0; m < M; ++m) {
            const auto* q8_blocks =
                reinterpret_cast<const MoEBlockQ8K*>(quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = q8_blocks[bi];

            int32_t min_dot = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                min_dot += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            int32_t dot_scaled = 0;
            const int8_t* q8 = yb.qs;
            for (int group = 0; group < QK_K / 32; ++group) {
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32), q4_vec[group][0]);
                acc = vdotq_s32(acc, vld1q_s8(q8 + group * 32 + 16), q4_vec[group][1]);
                dot_scaled += static_cast<int32_t>(scales[group]) * vaddvq_s32(acc);
            }

            const float d = q4_d * yb.d;
            const float dmin = q4_dmin * yb.d;
            out_sums[m] += d * static_cast<float>(dot_scaled) - dmin * static_cast<float>(min_dot);
        }
    }

    return true;
}
#endif

bool ComputeMoEQ4KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base, size_t quant_row_stride, int M,
                                int K, float* out_sums) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
    if (ComputeMoEQ4KQ8KBatchedRowDotprod(weight_row, quant_input_base, quant_row_stride, M, K, out_sums)) {
        return true;
    }
#endif
    return ComputeMoEQ4KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, K, out_sums);
}

bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel) {
    if (!backend || !weight_ptr || !qinput_data || !out_data || M <= 0 || M > kMoEQuantizedProjectionMaxBatch ||
        N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float sums[kMoEQ4KRawBatchedTileM];
        for (int n = n_start; n < n_end; ++n) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const void* weight_row = static_cast<const char*>(weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
                const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
                const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRow(weight_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sums[m];
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    return ok.load(std::memory_order_relaxed);
}

bool RunMoEQ4KRawBatchedFusedSwiGLU(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                    const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                    int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !out_data || M <= 0 ||
        M > kMoEQuantizedProjectionMaxBatch || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float gate_sums[kMoEQ4KRawBatchedTileM];
        alignas(64) float up_sums[kMoEQ4KRawBatchedTileM];
        for (int n = n_start; n < n_end; ++n) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const void* gate_row =
                static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            const void* up_row = static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
                const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
                const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRow(gate_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                gate_sums) ||
                    !ComputeMoEQ4KQ8KBatchedRow(up_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                up_sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    const float gate = gate_sums[m];
                    const float up = up_sums[m];
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        (gate / (1.0f + internal::FastExp(-gate))) * up;
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    return ok.load(std::memory_order_relaxed);
}

bool RunMoEQ4KRawBatchedFusedGEGLU(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                   const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                   int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !out_data || M <= 0 ||
        M > kMoEQuantizedProjectionMaxBatch || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    std::atomic<bool> ok{true};
    const auto compute_rows = [&](int n_start, int n_end) {
        alignas(64) float gate_sums[kMoEQ4KRawBatchedTileM];
        alignas(64) float up_sums[kMoEQ4KRawBatchedTileM];
        for (int n = n_start; n < n_end; ++n) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }
            const void* gate_row =
                static_cast<const char*>(gate_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            const void* up_row = static_cast<const char*>(up_weight_ptr) + static_cast<size_t>(n) * weight_row_bytes;
            for (int64_t m0 = 0; m0 < M; m0 += kMoEQ4KRawBatchedTileM) {
                const int tile_m = static_cast<int>(std::min<int64_t>(kMoEQ4KRawBatchedTileM, M - m0));
                const auto* q_tile = qinput_data + static_cast<size_t>(m0) * qinput_row_bytes;
                if (!ComputeMoEQ4KQ8KBatchedRow(gate_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                gate_sums) ||
                    !ComputeMoEQ4KQ8KBatchedRow(up_row, q_tile, qinput_row_bytes, tile_m, static_cast<int>(K),
                                                up_sums)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                for (int m = 0; m < tile_m; ++m) {
                    out_data[static_cast<size_t>(m0 + m) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                        GeluTanhApprox(gate_sums[m]) * up_sums[m];
                }
            }
        }
    };

    if (n_threads <= 1 || N < 64) {
        compute_rows(0, static_cast<int>(N));
    } else {
        pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_rows(n_start, n_end); });
    }
    return ok.load(std::memory_order_relaxed);
}

bool CanUseMoEQ4KRawBatchedScalar() {
    return true;
}

bool CanUseKQuantRowPairVecDotFastPath() {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
    return true;
#else
    return false;
#endif
}

bool CanUseQ4KRowPairVecDotFastPath() {
    return CanUseKQuantRowPairVecDotFastPath();
}

bool IsKQuantRowPairGatedProjectionType(ggml_type weight_type, ggml_type input_type) {
    return (weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q5_K || weight_type == GGML_TYPE_Q5_1) &&
           (input_type == GGML_TYPE_Q8_K || input_type == GGML_TYPE_Q8_1);
}

bool CanUseQ4KRepackedMoEGemvFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    return ggml_cpu_has_avx2();
#endif
}

bool CanUseQ4KRepackedMoEGEGLUFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    return ggml_cpu_has_avx2();
#endif
}

bool CanUseQ4KRepackedMoEPrefillFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    return ggml_cpu_has_avx2();
#endif
}

bool CanUseQ6KRepackedMoEGemvFastPath() {
    const char* enable_env = std::getenv("DENSECORE_MOE_ENABLE_Q6K_REPACKED_GEMV");
    if (!enable_env || enable_env[0] == '\0' || std::strcmp(enable_env, "0") == 0) {
        return false;
    }
    const char* env = std::getenv("DENSECORE_MOE_DISABLE_Q6K_REPACKED_GEMV");
    if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
        return false;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

bool CanUseQ5KRepackedMoEGemvFastPath() {
    const char* enable_env = std::getenv("DENSECORE_MOE_ENABLE_Q5K_REPACKED_GEMV");
    if (!enable_env || enable_env[0] == '\0' || std::strcmp(enable_env, "0") == 0) {
        return false;
    }
    const char* env = std::getenv("DENSECORE_MOE_DISABLE_Q5K_REPACKED_GEMV");
    if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
        return false;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

bool IsQwenA3BHybridMoEModel(const TransformerModel* model);

bool CanUseSmallDecodeQuantizedTileParallel(const TransformerModel* model) {
    const char* env = std::getenv("DENSECORE_MOE_ENABLE_SMALL_DECODE_QUANT_TILE_PARALLEL");
    if (env && env[0] != '\0') {
        return std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "OFF") != 0;
    }
    // Enable by default for Qwen A3B hybrid MoE models — tile-parallel GEMV
    // spreads expert work across all available cores during single-token decode.
    return IsQwenA3BHybridMoEModel(model);
}

using MoEQ4Kx8Block = densecore::kernels::Q4KRepackedGemvBlock;

struct MoEQ5Kx8Block {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qh[QK_K];
    uint8_t qs[QK_K * 4];
};
static_assert(sizeof(MoEQ5Kx8Block) == 16 * sizeof(uint16_t) + 8 * K_SCALE_SIZE + QK_K * 5,
              "MoE Q5_Kx8 block layout must match ggml block_q5_Kx8");

struct MoEQ6Kx8Block {
    uint16_t d[8];
    int8_t scales[QK_K / 16 * 8];
    uint8_t ql[QK_K / 2 * 8];
    uint8_t qh[QK_K / 4 * 8];
};
static_assert(sizeof(MoEQ6Kx8Block) == 8 * sizeof(uint16_t) + (QK_K / 16) * 8 + (3 * QK_K / 4) * 8,
              "MoE Q6_Kx8 block layout must match ggml block_q6_Kx8");

size_t GetRepackedMoECacheLimitBytes() {
    // Backward-compatible name: this cache limit now applies to the remaining
    // MoE-local Q5_K/Q6_K repack caches. Q4_K uses the shared single-flight
    // densecore::kernels::Q4KRepackedGemvWeight cache below.
    const char* env = std::getenv("DENSECORE_MOE_Q4K_REPACK_CACHE_MB");
    if (!env || env[0] == '\0') {
        constexpr size_t kMinBytes = 1024ull * 1024ull * 1024ull;
        constexpr size_t kMaxBytes = 4ull * 1024ull * 1024ull * 1024ull;
#if defined(__linux__)
        struct sysinfo info {};
        if (sysinfo(&info) == 0 && info.mem_unit > 0) {
            const uint64_t unit = static_cast<uint64_t>(info.mem_unit);
            const uint64_t free_bytes = (static_cast<uint64_t>(info.freeram) +
                                         static_cast<uint64_t>(info.bufferram)) *
                                        unit;
            const size_t target = static_cast<size_t>(free_bytes / 8);
            return std::clamp(target, kMinBytes, kMaxBytes);
        }
#endif
        return 2ull * 1024ull * 1024ull * 1024ull;
    }
    const long long mb = std::strtoll(env, nullptr, 10);
    if (mb <= 0) {
        return 0;
    }
    return static_cast<size_t>(mb) * 1024ull * 1024ull;
}

struct RepackedMoEKey {
    const void* weight_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const RepackedMoEKey& other) const {
        return weight_ptr == other.weight_ptr && rows == other.rows && cols == other.cols &&
               fingerprint == other.fingerprint;
    }
};

struct RepackedMoEKeyHash {
    size_t operator()(const RepackedMoEKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_ptr);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

uint64_t FingerprintRepackedMoEWeight(const void* weight_ptr, size_t bytes) {
    if (!weight_ptr || bytes == 0) {
        return 0;
    }
    const auto* data = static_cast<const uint8_t*>(weight_ptr);
    constexpr size_t kWindow = 4096;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const uint8_t* ptr, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<uint64_t>(ptr[i]);
            h *= 1099511628211ull;
        }
    };
    mix(reinterpret_cast<const uint8_t*>(&bytes), sizeof(bytes));
    if (bytes <= kWindow * 2) {
        mix(data, bytes);
    } else {
        mix(data, kWindow);
        mix(data + bytes - kWindow, kWindow);
    }
    return h;
}

using Q4KRepackedMoEWeight = densecore::kernels::Q4KRepackedGemvWeight;

std::shared_ptr<Q4KRepackedMoEWeight> GetOrCreateQ4KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !(CanUseQ4KRepackedMoEGemvFastPath() || CanUseQ4KRepackedMoEPrefillFastPath() ||
          CanUseQ4KRepackedMoEGEGLUFastPath())) {
        return nullptr;
    }
    return densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(weight_ptr, rows, cols);
}

struct Q5KRepackedMoEWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<MoEQ5Kx8Block> blocks;
};

std::shared_ptr<Q5KRepackedMoEWeight> GetOrCreateQ5KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !CanUseQ5KRepackedMoEGemvFastPath()) {
        return nullptr;
    }
    const size_t cache_limit = GetRepackedMoECacheLimitBytes();
    if (cache_limit == 0) {
        return nullptr;
    }

    static std::mutex mutex;
    static std::unordered_map<RepackedMoEKey, std::shared_ptr<Q5KRepackedMoEWeight>, RepackedMoEKeyHash> cache;
    static size_t cache_bytes = 0;
    static std::atomic<uint64_t> use_clock{0};

    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q5_K, cols);
    const RepackedMoEKey key{weight_ptr, rows, cols, FingerprintRepackedMoEWeight(weight_ptr, raw_bytes)};
    const uint64_t now = use_clock.fetch_add(1, std::memory_order_relaxed) + 1;

    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) {
        found->second->last_use = now;
        return found->second;
    }

    const int64_t blocks_per_row = cols / QK_K;
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * static_cast<size_t>(blocks_per_row);
    auto packed = std::make_shared<Q5KRepackedMoEWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = blocks_per_row;
    packed->blocks.resize(packed_blocks);
    packed->bytes = packed->blocks.size() * sizeof(MoEQ5Kx8Block);
    packed->last_use = now;

    if (ggml_repack_q5_K_8x8(weight_ptr, raw_bytes, rows, cols, packed->blocks.data(), packed->bytes) != 0) {
        return nullptr;
    }

    cache.emplace(key, packed);
    cache_bytes += packed->bytes;
    while (cache_bytes > cache_limit && cache.size() > 1) {
        auto oldest = cache.end();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == key) {
                continue;
            }
            if (oldest == cache.end() || it->second->last_use < oldest->second->last_use) {
                oldest = it;
            }
        }
        if (oldest == cache.end()) {
            break;
        }
        cache_bytes -= oldest->second->bytes;
        cache.erase(oldest);
    }

    return packed;
}

struct Q6KRepackedMoEWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    std::vector<MoEQ6Kx8Block> blocks;
};

std::shared_ptr<Q6KRepackedMoEWeight> GetOrCreateQ6KRepackedMoEWeight(const void* weight_ptr, int64_t rows,
                                                                      int64_t cols) {
    if (!weight_ptr || rows <= 0 || cols <= 0 || (rows % 8) != 0 || (cols % QK_K) != 0 ||
        !CanUseQ6KRepackedMoEGemvFastPath()) {
        return nullptr;
    }
    const size_t cache_limit = GetRepackedMoECacheLimitBytes();
    if (cache_limit == 0) {
        return nullptr;
    }

    static std::mutex mutex;
    static std::unordered_map<RepackedMoEKey, std::shared_ptr<Q6KRepackedMoEWeight>, RepackedMoEKeyHash> cache;
    static size_t cache_bytes = 0;
    static std::atomic<uint64_t> use_clock{0};

    const size_t raw_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q6_K, cols);
    const RepackedMoEKey key{weight_ptr, rows, cols, FingerprintRepackedMoEWeight(weight_ptr, raw_bytes)};
    const uint64_t now = use_clock.fetch_add(1, std::memory_order_relaxed) + 1;

    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) {
        found->second->last_use = now;
        return found->second;
    }

    const int64_t blocks_per_row = cols / QK_K;
    const size_t packed_blocks = static_cast<size_t>(rows / 8) * static_cast<size_t>(blocks_per_row);
    auto packed = std::make_shared<Q6KRepackedMoEWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = blocks_per_row;
    packed->blocks.resize(packed_blocks);
    packed->bytes = packed->blocks.size() * sizeof(MoEQ6Kx8Block);
    packed->last_use = now;

    if (ggml_repack_q6_K_8x8(weight_ptr, raw_bytes, rows, cols, packed->blocks.data(), packed->bytes) != 0) {
        return nullptr;
    }

    cache.emplace(key, packed);
    cache_bytes += packed->bytes;
    while (cache_bytes > cache_limit && cache.size() > 1) {
        auto oldest = cache.end();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == key) {
                continue;
            }
            if (oldest == cache.end() || it->second->last_use < oldest->second->last_use) {
                oldest = it;
            }
        }
        if (oldest == cache.end()) {
            break;
        }
        cache_bytes -= oldest->second->bytes;
        cache.erase(oldest);
    }

    return packed;
}

bool RunQ4KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const auto compute_tiles = [&](int tile_start, int tile_end) {
        return densecore::kernels::RunQ4KRepackedGemvRows(packed, qinput_data, qinput_row_bytes, output_data, rows,
                                                          cols, tile_start, tile_end);
    };
    if (n_threads > 1 && tile_count >= 2) {
        std::atomic<bool> ok{true};
        pool.ParallelFor(tile_count, [&](int start, int end, int) {
            if (!compute_tiles(start, end)) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
        return ok.load(std::memory_order_relaxed);
    }
    return compute_tiles(0, tile_count);
}

bool RunQ4KRepackedMoEGemmM4(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& packed,
                             const float* input_data, float* output_data, int64_t rows, int64_t cols,
                             int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !input_data || !output_data || rows < 4 || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0 || input_cols != packed->cols || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
            ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(packed->cols), out_tile + static_cast<size_t>(tile_start) * 8,
                                    static_cast<size_t>(cols), vx, q8x4_buf.data(), 4, (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    if (row < rows) {
        const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, input_cols);
        static thread_local std::vector<uint8_t> q8_tail_buf;
        const size_t tail_bytes = static_cast<size_t>(rows - row) * q8_row_bytes;
        if (q8_tail_buf.size() < tail_bytes) {
            q8_tail_buf.resize(tail_bytes);
        }
        const auto* iq_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        if (!iq_traits || !iq_traits->from_float) {
            return false;
        }
        for (int64_t m = row; m < rows; ++m) {
            iq_traits->from_float(input_data + static_cast<size_t>(m) * static_cast<size_t>(input_cols),
                                  q8_tail_buf.data() + static_cast<size_t>(m - row) * q8_row_bytes, input_cols);
        }
        return RunQ4KRepackedMoEGemv(backend, packed, q8_tail_buf.data(), q8_row_bytes,
                                     output_data + static_cast<size_t>(row) * static_cast<size_t>(cols), rows - row,
                                     cols, numa_node, allow_parallel);
    }

    return true;
}

bool RunQ4KRepackedMoEFusedSwiGLUM4(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& gate_packed,
                                    const std::shared_ptr<Q4KRepackedMoEWeight>& up_packed, const float* input_data,
                                    const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                    int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                    bool allow_parallel) {
    if (!backend || !gate_packed || !up_packed || !input_data || !qinput_data || !output_data || rows <= 0 ||
        gate_packed->rows != cols || up_packed->rows != cols || gate_packed->cols <= 0 ||
        gate_packed->cols != up_packed->cols || gate_packed->blocks_per_row != up_packed->blocks_per_row ||
        (cols % 8) != 0 || input_cols != gate_packed->cols || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(gate_packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);

        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 32> gate_tile{};
            std::array<float, 32> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 8, gate_vx,
                                        q8x4_buf.data(), 4, 8);
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 8, up_vx, q8x4_buf.data(), 4,
                                        8);
                for (int r = 0; r < 4; ++r) {
                    float* out_row =
                        out_tile + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(tile) * 8;
                    const float* gate_row = gate_tile.data() + static_cast<size_t>(r) * 8;
                    const float* up_row = up_tile.data() + static_cast<size_t>(r) * 8;
                    for (int c = 0; c < 8; ++c) {
                        const float gate_sum = gate_row[c];
                        out_row[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_row[c];
                    }
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    for (; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out_row = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 8> gate_tile{};
            std::array<float, 8> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 0, gate_vx, qi, 1, 8);
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 0, up_vx, qi, 1, 8);
                float* out = out_row + static_cast<size_t>(tile) * 8;
                for (int c = 0; c < 8; ++c) {
                    const float gate_sum = gate_tile[c];
                    out[c] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_tile[c];
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    return true;
}

bool RunQ4KRepackedMoEFusedGEGLU(CpuBackend* backend, const std::shared_ptr<Q4KRepackedMoEWeight>& gate_packed,
                                 const std::shared_ptr<Q4KRepackedMoEWeight>& up_packed, const float* input_data,
                                 const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                 int64_t rows, int64_t cols, int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !gate_packed || !up_packed || !input_data || !qinput_data || !output_data || rows <= 0 ||
        gate_packed->rows != cols || up_packed->rows != cols || gate_packed->cols <= 0 ||
        gate_packed->cols != up_packed->cols || gate_packed->blocks_per_row != up_packed->blocks_per_row ||
        (cols % 8) != 0 || input_cols != gate_packed->cols || (input_cols % QK_K) != 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(gate_packed->blocks_per_row);
    const size_t q8_tile_bytes = static_cast<size_t>(input_cols / QK_K) * sizeof(MoEBlockQ8Kx4);

    static thread_local std::vector<uint8_t> q8x4_buf;
    if (q8x4_buf.size() < q8_tile_bytes) {
        q8x4_buf.resize(q8_tile_bytes);
    }

    int64_t row = 0;
    for (; row + 3 < rows; row += 4) {
        const float* input_tile = input_data + static_cast<size_t>(row) * static_cast<size_t>(input_cols);
        ggml_quantize_mat_q8_K_4x8(input_tile, q8x4_buf.data(), input_cols);
        float* out_tile = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);

        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 32> gate_tile{};
            std::array<float, 32> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 8, gate_vx,
                                        q8x4_buf.data(), 4, 8);
                ggml_gemm_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 8, up_vx, q8x4_buf.data(), 4,
                                        8);
                for (int r = 0; r < 4; ++r) {
                    float* out_row =
                        out_tile + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(tile) * 8;
                    const float* gate_row = gate_tile.data() + static_cast<size_t>(r) * 8;
                    const float* up_row = up_tile.data() + static_cast<size_t>(r) * 8;
                    for (int c = 0; c < 8; ++c) {
                        out_row[c] = GeluTanhApprox(gate_row[c]) * up_row[c];
                    }
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    for (; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out_row = output_data + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            std::array<float, 8> gate_tile{};
            std::array<float, 8> up_tile{};
            for (int tile = tile_start; tile < tile_end; ++tile) {
                const void* gate_vx = gate_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                const void* up_vx = up_packed->blocks.data() + static_cast<size_t>(tile) * blocks_per_row;
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(gate_packed->cols), gate_tile.data(), 0, gate_vx, qi, 1, 8);
                ggml_gemv_q4_K_8x8_q8_K(static_cast<int>(up_packed->cols), up_tile.data(), 0, up_vx, qi, 1, 8);
                float* out = out_row + static_cast<size_t>(tile) * 8;
                for (int c = 0; c < 8; ++c) {
                    out[c] = GeluTanhApprox(gate_tile[c]) * up_tile[c];
                }
            }
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }

    return true;
}

bool RunQ5KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q5KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);

    for (int64_t row = 0; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out = output_data + static_cast<size_t>(row) * cols;
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
            ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(packed->cols), out + static_cast<size_t>(tile_start) * 8, 0, vx,
                                    qi, 1, (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }
    return true;
}

bool RunQ6KRepackedMoEGemv(CpuBackend* backend, const std::shared_ptr<Q6KRepackedMoEWeight>& packed,
                           const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data, int64_t rows,
                           int64_t cols, int numa_node, bool allow_parallel) {
    if (!backend || !packed || !qinput_data || !output_data || packed->rows != cols || (cols % 8) != 0 ||
        packed->cols <= 0) {
        return false;
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const int tile_count = static_cast<int>(cols / 8);
    const int blocks_per_row = static_cast<int>(packed->blocks_per_row);

    for (int64_t row = 0; row < rows; ++row) {
        const auto* qi = qinput_data + static_cast<size_t>(row) * qinput_row_bytes;
        float* out = output_data + static_cast<size_t>(row) * cols;
        const auto compute_tiles = [&](int tile_start, int tile_end) {
            if (tile_start >= tile_end) {
                return;
            }
            const void* vx = packed->blocks.data() + static_cast<size_t>(tile_start) * blocks_per_row;
            ggml_gemv_q6_K_8x8_q8_K(static_cast<int>(packed->cols), out + static_cast<size_t>(tile_start) * 8, 0, vx,
                                    qi, 1, (tile_end - tile_start) * 8);
        };
        if (n_threads > 1 && tile_count >= 2) {
            pool.ParallelFor(tile_count, [&](int start, int end, int) { compute_tiles(start, end); });
        } else {
            compute_tiles(0, tile_count);
        }
    }
    return true;
}

const std::array<float, 1 << 16>& GetGeluF16LookupTable() {
    static std::array<float, 1 << 16> table{};
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        for (uint32_t i = 0; i < table.size(); ++i) {
            const ggml_fp16_t fp16 = static_cast<ggml_fp16_t>(i);
            const float x = ggml_fp16_to_fp32(fp16);
            const float x3 = x * x * x;
            const float y = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
            table[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(y));
        }
    });
    return table;
}

inline float GeluTanhApprox(float x) {
    const ggml_fp16_t fp16 = ggml_fp32_to_fp16(x);
    return GetGeluF16LookupTable()[static_cast<uint16_t>(fp16)];
}

bool CanUsePackedInt4MoEFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // Keep the packed INT4 MoE route enabled on ARM, but route it through
    // CpuBackend::GemmInt4() instead of the direct Highway small-batch kernels.
    // That preserves the intended fast path while reusing the verified
    // runtime-selected ARM INT4 kernel selection.
    return true;
#else
    return true;
#endif
}

bool IsMoEDebugTimingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_MOE_DEBUG_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsMoEFFNDebugTimingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_FFN_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsMoESafeReferenceModeEnabled() {
    const char* env = std::getenv("DENSECORE_MOE_SAFE_REFERENCE");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool ShouldForceMoESafeReference(const CpuBackend::ExpertWeights* expert) {
    return expert && expert->force_safe_reference;
}

bool IsMoESafeReferenceModeEnabled(const CpuBackend::ExpertWeights* expert) {
    return IsMoESafeReferenceModeEnabled() || ShouldForceMoESafeReference(expert);
}

bool ShouldUseQwen36ShortPrefillSafeReference() {
    const auto mode = densecore::env::ParseRuntimeToggleMode("DENSECORE_QWEN36_SHORT_PREFILL_SAFE_REFERENCE",
#if defined(__aarch64__) || defined(_M_ARM64)
                                                             densecore::env::RuntimeToggleMode::Off
#else
                                                             densecore::env::RuntimeToggleMode::On
#endif
    );
    return mode == densecore::env::RuntimeToggleMode::On;
}

bool IsQwen36ShortSingleSeqPrefillSafeReferenceCandidate(const TransformerModel* model, const BatchSpec* batch,
                                                         int batch_size) {
    if (!model || !batch) {
        return false;
    }
    if (model->variant != ModelVariant::QWEN36 || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    if (!ShouldUseQwen36ShortPrefillSafeReference()) {
        return false;
    }
    if (batch->num_seqs != 1 || batch_size <= 1 || batch_size >= 64) {
        return false;
    }
    return true;
}

bool IsMoEMatmulPathDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_MOE_MATMUL_PATHS");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

#if defined(__aarch64__) || defined(_M_ARM64)
bool IsArmPackedInt4HwyProjectionSupported() {
    const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return level == densecore::simd::SimdLevel::SVE || level == densecore::simd::SimdLevel::SVE2;
}
#endif

bool IsMoECachePolicyDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_MOE_CACHE_POLICY");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

int GetSmallDecodeExpertWorkers() {
    const char* env = std::getenv("DENSECORE_MOE_SMALL_DECODE_EXPERT_WORKERS");
    if (!env || env[0] == '\0') {
        return 0;
    }
    const int parsed = std::atoi(env);
    return parsed > 0 ? parsed : 0;
}

bool IsGemma4PackedChecksumDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_PACKED_CHECKSUM");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

struct MoEExecutionTraceContext {
    int layer_idx = -1;
    int seq_id = -1;
    int token_idx = -1;
    int decode_step = -1;
    int n_past = -1;
    int expert_id = -1;
};

bool IsGemma4ParityTraceEnabled() {
    const char* env = std::getenv("DENSECORE_GEMMA4_PARITY_TRACE");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

int Gemma4ParityTraceLayerFilter() {
    const char* env = std::getenv("DENSECORE_GEMMA4_PARITY_TRACE_LAYER");
    if (!env || env[0] == '\0') {
        return -1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    return end == env ? -1 : static_cast<int>(parsed);
}

void LogGemma4MoETensorStats(const char* stage, const Tensor& tensor, const MoEExecutionTraceContext* trace_ctx) {
    if (!IsGemma4ParityTraceEnabled() || !stage || !trace_ctx || trace_ctx->layer_idx < 0 || !tensor.IsValid() ||
        tensor.dtype != DType::F32 || tensor.ndim != 2) {
        return;
    }
    const int layer_filter = Gemma4ParityTraceLayerFilter();
    if (layer_filter >= 0 && trace_ctx->layer_idx != layer_filter) {
        return;
    }
    static std::atomic<int> log_budget{0};
    const int current = log_budget.fetch_add(1, std::memory_order_relaxed);
    if (current >= 512) {
        return;
    }
    const float* data = tensor.DataAs<float>();
    if (!data) {
        return;
    }
    const size_t elems = static_cast<size_t>(tensor.shape[0]) * static_cast<size_t>(tensor.shape[1]);
    size_t nonfinite = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    double sum_sq = 0.0;
    uint64_t finite_hash = 1469598103934665603ull;
    for (size_t i = 0; i < elems; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) {
            ++nonfinite;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        finite_hash ^= static_cast<uint64_t>(bits);
        finite_hash *= 1099511628211ull;
    }
    if (elems == nonfinite) {
        min_v = std::numeric_limits<float>::quiet_NaN();
        max_v = std::numeric_limits<float>::quiet_NaN();
    }
    const double rms = elems > nonfinite ? std::sqrt(sum_sq / static_cast<double>(elems - nonfinite)) : NAN;
    std::fprintf(stderr,
                 "[GEMMA4_MOE_TRACE] layer=%d expert=%d token=%d seq=%d n_past=%d decode_step=%d stage=%s "
                 "shape=%lldx%lld nonfinite=%zu min=%g max=%g rms=%g hash=0x%llx\n",
                 trace_ctx->layer_idx, trace_ctx->expert_id, trace_ctx->token_idx, trace_ctx->seq_id, trace_ctx->n_past,
                 trace_ctx->decode_step, stage, static_cast<long long>(tensor.shape[0]),
                 static_cast<long long>(tensor.shape[1]), nonfinite, min_v, max_v, rms,
                 static_cast<unsigned long long>(finite_hash));
}

uint64_t Fnv1a64(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t ChecksumF32Tensor(const Tensor& tensor) {
    if (!tensor.IsValid() || tensor.dtype != DType::F32) {
        return 0;
    }
    const size_t elems = static_cast<size_t>(tensor.shape[0] * tensor.shape[1]);
    return Fnv1a64(tensor.DataAs<float>(), elems * sizeof(float));
}

uint64_t ChecksumRawTensorBytes(const ggml_tensor* tensor) {
    if (!tensor || !tensor->data) {
        return 0;
    }
    return Fnv1a64(tensor->data, ggml_nbytes(tensor));
}

uint64_t ChecksumPackedGateUpFallback(const CpuBackend::ExpertWeights& expert) {
    if (expert.gate_up_tensor && expert.gate_up_tensor->data) {
        return ChecksumRawTensorBytes(expert.gate_up_tensor);
    }
    uint64_t hash = 1469598103934665603ull;
    if (expert.w1_tensor && expert.w1_tensor->data) {
        hash ^= ChecksumRawTensorBytes(expert.w1_tensor);
        hash *= 1099511628211ull;
    }
    if (expert.w3_tensor && expert.w3_tensor->data) {
        hash ^= ChecksumRawTensorBytes(expert.w3_tensor);
        hash *= 1099511628211ull;
    }
    return hash;
}

void LogMoEMatmulPath(const char* path, int M, int K, int N, int group_size, bool allow_parallel) {
    if (!IsMoEMatmulPathDebugEnabled()) {
        return;
    }
    const MoEInt4PathHistogram& histogram = GetMoEInt4PathHistogram();
    std::fprintf(stderr,
                 "[MOE_MATMUL_PATH] path=%s M=%d K=%d N=%d group_size=%d allow_parallel=%d "
                 "moe_int4_direct_hwy_count=%llu moe_int4_fused_swiglu_hwy_count=%llu "
                 "moe_int4_backend_gemm_count=%llu moe_int4_f32_fallback_count=%llu\n",
                 path ? path : "unknown", M, K, N, group_size, allow_parallel ? 1 : 0,
                 static_cast<unsigned long long>(histogram.direct_hwy.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(histogram.fused_swiglu_hwy.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(histogram.backend_gemm.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(histogram.f32_fallback.load(std::memory_order_relaxed)));
}

void LogSmallDecodeExecutionPath(const char* path, int assignments, int workers, int batch_size) {
    if (!IsMoEMatmulPathDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "[MOE_SMALL_DECODE] path=%s assignments=%d workers=%d batch=%d\n", path ? path : "unknown",
                 assignments, workers, batch_size);
}

void LogSmallDecodeFallback(const char* reason, int assignments, int workers, int batch_size) {
    if (!IsMoEMatmulPathDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "[MOE_SMALL_DECODE] path=serial_fallback reason=%s assignments=%d workers=%d batch=%d\n",
                 reason ? reason : "unknown", assignments, workers, batch_size);
}

bool IsQwenA3BHybridMoEModel(const TransformerModel* model) {
    return model && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
           model->arch_flags.is_hybrid_ssm && model->hparams.n_experts > 0 &&
           model->hparams.n_experts_used > 1;
}

bool IsGemma4MoEModelForSmallDecodeParallel(const TransformerModel* model) {
    return model && model->arch_flags.is_gemma4 && model->hparams.n_experts > 0;
}

bool IsSmallDecodeExpertParallelAutoModel(const TransformerModel* model) {
    return IsQwenA3BHybridMoEModel(model) || IsGemma4MoEModelForSmallDecodeParallel(model);
}

bool IsSmallDecodeExpertParallelSimdLevel(densecore::simd::SimdLevel level) {
    return level == densecore::simd::SimdLevel::SVE || level == densecore::simd::SimdLevel::SVE2 ||
           densecore::simd::HasX86Avx2OrBetter(level);
}

densecore::env::RuntimeToggleMode ResolveSmallDecodeExpertParallelMode(const TransformerModel* model) {
    const densecore::env::RuntimeToggleMode generic_mode = densecore::env::ParseRuntimeToggleMode(
        "DENSECORE_MOE_SMALL_DECODE_EXPERT_PARALLEL", densecore::env::RuntimeToggleMode::Auto);
    if (std::getenv("DENSECORE_MOE_SMALL_DECODE_EXPERT_PARALLEL")) {
        return generic_mode;
    }
    if (model && model->variant == ModelVariant::QWEN35 && std::getenv("DENSECORE_QWEN35_MOE_PARALLEL")) {
        return densecore::env::ParseRuntimeToggleMode("DENSECORE_QWEN35_MOE_PARALLEL",
                                                      densecore::env::RuntimeToggleMode::Auto);
    }
    if (model && model->variant == ModelVariant::QWEN36 && std::getenv("DENSECORE_QWEN36_MOE_PARALLEL")) {
        return densecore::env::ParseRuntimeToggleMode("DENSECORE_QWEN36_MOE_PARALLEL",
                                                      densecore::env::RuntimeToggleMode::Auto);
    }
    return densecore::env::RuntimeToggleMode::Auto;
}

bool ResolveSmallDecodeExpertParallelAutoEligible(const TransformerModel* model, int physical_cores, int worker_cap,
                                                  densecore::simd::SimdLevel level) {
    const int effective_cores = std::max(physical_cores, worker_cap);
    return IsSmallDecodeExpertParallelAutoModel(model) && worker_cap > 1 && effective_cores >= 16 &&
           IsSmallDecodeExpertParallelSimdLevel(level);
}

bool ResolveSmallDecodeExpertParallelAutoEligible(const TransformerModel* model, int physical_cores,
                                                  densecore::simd::SimdLevel level) {
    return ResolveSmallDecodeExpertParallelAutoEligible(model, physical_cores, physical_cores, level);
}

int ResolveSmallDecodeExpertWorkers(int top_k, int worker_cap, int requested_override) {
    const int requested = requested_override > 0 ? requested_override : std::min(std::max(1, top_k), 8);
    return std::max(1, std::min(std::max(1, worker_cap), requested));
}

struct SmallDecodeExpertParallelDecision {
    bool requested = false;
    bool forced_on = false;
    bool enabled = false;
    int workers = 1;
    const char* reason = "disabled";
};

SmallDecodeExpertParallelDecision ResolveSmallDecodeExpertParallelDecision(const TransformerModel* model,
                                                                           int batch_size, int top_k,
                                                                           bool safe_reference_mode, int worker_cap) {
    SmallDecodeExpertParallelDecision decision;
    const int requested_override = GetSmallDecodeExpertWorkers();
    decision.workers = ResolveSmallDecodeExpertWorkers(top_k, worker_cap, requested_override);

    const densecore::env::RuntimeToggleMode mode = ResolveSmallDecodeExpertParallelMode(model);
    decision.forced_on = mode == densecore::env::RuntimeToggleMode::On;
    if (mode == densecore::env::RuntimeToggleMode::Off) {
        decision.reason = "mode_off";
        return decision;
    }

    if (!IsSmallDecodeExpertParallelAutoModel(model)) {
        decision.reason = "not_auto_moe_model";
        return decision;
    }

    int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
    if (physical_cores <= 0) {
        physical_cores = worker_cap;
    }
    const int effective_cores = std::max(physical_cores, worker_cap);
    const densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
    if (!IsSmallDecodeExpertParallelSimdLevel(simd_level)) {
        decision.reason = "simd_not_supported";
        return decision;
    }
    if (worker_cap <= 1 || effective_cores < 16) {
        decision.reason = "insufficient_worker_threads";
        return decision;
    }
    if (batch_size <= 0 || batch_size > 4) {
        decision.reason = "not_small_decode";
        return decision;
    }
    if (top_k <= 1) {
        decision.reason = "top_k_too_small";
        return decision;
    }

    decision.requested = true;
    if (safe_reference_mode) {
        decision.reason = "safe_reference";
        return decision;
    }
    if (decision.workers <= 1) {
        decision.reason = "insufficient_worker_threads";
        return decision;
    }

    decision.enabled = true;
    decision.reason = mode == densecore::env::RuntimeToggleMode::On ? "forced_on" : "auto";
    return decision;
}

CpuBackend::MoEProjectionPath ParseMoEProjectionPath(const char* path) {
    if (!path) {
        return CpuBackend::MoEProjectionPath::Unknown;
    }
    if (std::strcmp(path, "direct_hwy") == 0) {
        return CpuBackend::MoEProjectionPath::PackedInt4Fast;
    }
    if (std::strcmp(path, "backend_gemmint4") == 0) {
        return CpuBackend::MoEProjectionPath::RuntimeGemmInt4;
    }
    if (std::strcmp(path, "ggml_quantized_vecdot") == 0) {
        return CpuBackend::MoEProjectionPath::GgmlQuantizedVecDot;
    }
    if (std::strcmp(path, "reference_f32") == 0) {
        return CpuBackend::MoEProjectionPath::ReferenceF32;
    }
    if (std::strcmp(path, "dense_f32") == 0) {
        return CpuBackend::MoEProjectionPath::DenseF32;
    }
    return CpuBackend::MoEProjectionPath::Unknown;
}

void MaybeLogGemma4PackedChecksum(const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                                  const Tensor& w3, const Tensor& post_gate_up, const Tensor& post_down,
                                  const MoEExecutionTraceContext* trace_ctx) {
    if (!IsGemma4PackedChecksumDebugEnabled() || !trace_ctx || !expert.use_gelu_activation ||
        trace_ctx->layer_idx < 0) {
        return;
    }
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        return;
    }
    std::fprintf(stderr,
                 "[GEMMA4_PACKED] layer=%d expert=%d packed_gate_up=0x%llx gate=0x%llx up=0x%llx down=0x%llx "
                 "post_gate_up=0x%llx post_down=0x%llx\n",
                 trace_ctx->layer_idx, trace_ctx->expert_id,
                 static_cast<unsigned long long>(ChecksumPackedGateUpFallback(expert)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w1)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w3)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(w2)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(post_gate_up)),
                 static_cast<unsigned long long>(ChecksumF32Tensor(post_down)));
}

bool ShouldParallelizeExpertFFNInner(int64_t batch, int64_t hidden_dim, int64_t intermediate_dim) {
    const char* force_env = std::getenv("DENSECORE_MOE_FORCE_INNER_PARALLEL");
    if (force_env && force_env[0] != '\0') {
        return std::strcmp(force_env, "0") != 0;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    // Qwen3.5-35B-A3B batch=1 decode on C4A repeatedly executes tiny expert
    // GEMV/GEMM fragments (top-k experts, one token). Fanning each fragment out
    // across the whole thread pool costs more than the math itself and was the
    // dominant source of backend_us inflation after the crash fix.
    const int64_t work_items = batch * intermediate_dim;
    if (batch <= 1 && hidden_dim <= 4096 && intermediate_dim <= 8192 && work_items <= 8192) {
        return false;
    }
#endif
    return true;
}

bool ShouldForcePrefillInnerParallelExperts() {
    const char* env = std::getenv("DENSECORE_MOE_PREFILL_FORCE_INNER_PARALLEL");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool ShouldBalanceParallelExpertWork() {
    const char* env = std::getenv("DENSECORE_MOE_BALANCE_PARALLEL_EXPERT_WORK");
    if (!env || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0;
}

bool ShouldUseDynamicParallelExpertQueue() {
    const char* env = std::getenv("DENSECORE_MOE_DYNAMIC_PARALLEL_EXPERT_QUEUE");
    if (!env || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0;
}

int GetDynamicParallelExpertQueueChunk() {
    const char* env = std::getenv("DENSECORE_MOE_DYNAMIC_PARALLEL_EXPERT_QUEUE_CHUNK");
    if (!env || env[0] == '\0') {
        return 1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed <= 0) {
        return 1;
    }
    return static_cast<int>(std::min<long>(parsed, 32));
}

bool IsMoEReferenceCheckEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

float MoEReferenceTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE_TOL");
        if (!env || env[0] == '\0') {
            return 1e-2f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-2f;
        }
        return parsed;
    }();
    return tol;
}

bool ShouldRunMoEReferenceCheck() {
    if (!IsMoEReferenceCheckEnabled()) {
        return false;
    }
    static std::atomic<int> remaining_budget{[]() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_REFERENCE_MAX_CALLS");
        if (!env || env[0] == '\0') {
            return 1;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || parsed <= 0) {
            return 1;
        }
        return static_cast<int>(parsed);
    }()};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool DequantizePackedInt4ToF32(const CpuBackend::ExpertPackedInt4Weight& binding, int64_t rows, int64_t cols,
                               float* out) {
    if (!binding.IsValid() || !out || rows <= 0 || cols <= 0 || binding.K != cols || binding.N < rows) {
        return false;
    }

    const int packed_cols = static_cast<int>((cols + 1) / 2);
    const int num_full_groups = static_cast<int>(cols / binding.group_size);
    const int k_aligned = num_full_groups * binding.group_size;

    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* packed_row = binding.packed_weights + static_cast<size_t>(r) * packed_cols;
        float* out_row = out + r * cols;
        const float* row_scales = binding.scales + static_cast<size_t>(r) * num_full_groups;
        const float* row_zeros = binding.zeros + static_cast<size_t>(r) * num_full_groups;

        for (int g = 0; g < num_full_groups; ++g) {
            const float scale = row_scales[g];
            const float zero = row_zeros[g];
            const int k_start = g * binding.group_size;
            const uint8_t* packed_group = packed_row + (k_start / 2);
            for (int k = 0; k < binding.group_size; ++k) {
                const uint8_t packed = packed_group[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F) : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }
                out_row[k_start + k] = scale * (static_cast<float>(q) - zero);
            }
        }

        if (k_aligned < cols) {
            const float scale = (num_full_groups > 0) ? row_scales[num_full_groups - 1] : 1.0f;
            const float zero = (num_full_groups > 0) ? row_zeros[num_full_groups - 1] : 0.0f;
            for (int64_t k = k_aligned; k < cols; ++k) {
                const uint8_t packed = packed_row[k / 2];
                int8_t q = (k & 1) ? static_cast<int8_t>((packed >> 4) & 0x0F) : static_cast<int8_t>(packed & 0x0F);
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }
                out_row[k] = scale * (static_cast<float>(q) - zero);
            }
        }
    }

    return true;
}

struct MoEReferenceExpertMatrices {
    std::vector<float> w1;  // [intermediate, hidden]
    std::vector<float> w2;  // [hidden, intermediate]
    std::vector<float> w3;  // [intermediate, hidden]
    int hidden_dim = 0;
    int intermediate_dim = 0;
};

bool ApplyScaleSidecarInPlace(const ggml_tensor* scale_tensor, int64_t rows, int64_t cols, float* values,
                              std::string* reason) {
    if (!scale_tensor) {
        return true;
    }
    if (!values || !scale_tensor->data || scale_tensor->type != GGML_TYPE_F32) {
        if (reason) {
            *reason = "invalid Gemma4 scale sidecar";
        }
        return false;
    }
    const int64_t scale_cols = scale_tensor->ne[0];
    const int64_t scale_rows = scale_tensor->ne[1];
    if (scale_cols <= 0 || scale_rows <= 0) {
        if (reason) {
            *reason = "empty Gemma4 scale sidecar";
        }
        return false;
    }
    const char* base = reinterpret_cast<const char*>(scale_tensor->data);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t sr = (scale_rows == rows) ? r : (scale_rows == 1 ? 0 : -1);
        if (sr < 0) {
            if (reason) {
                *reason = "unsupported Gemma4 scale sidecar row shape";
            }
            return false;
        }
        const char* scale_row = base + static_cast<size_t>(sr) * static_cast<size_t>(scale_tensor->nb[1]);
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t sc = (scale_cols == cols) ? c : (scale_cols == 1 ? 0 : -1);
            if (sc < 0) {
                if (reason) {
                    *reason = "unsupported Gemma4 scale sidecar column shape";
                }
                return false;
            }
            const float scale =
                *reinterpret_cast<const float*>(scale_row + static_cast<size_t>(sc) * scale_tensor->nb[0]);
            values[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] *= scale;
        }
    }
    return true;
}

bool IsScalarScaleSidecar(const ggml_tensor* scale_tensor) {
    if (!scale_tensor) {
        return false;
    }
    if (!scale_tensor->data || scale_tensor->type != GGML_TYPE_F32) {
        return false;
    }
    return scale_tensor->ne[0] == 1 && scale_tensor->ne[1] == 1 && scale_tensor->ne[2] == 1 && scale_tensor->ne[3] == 1;
}

bool QuantizedProjectionInputTypeMatches(int ggml_type_id, ggml_type input_type) {
    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    if (wtype == GGML_TYPE_F32 || !ggml_is_quantized(wtype)) {
        return false;
    }
    const auto* traits = ggml_get_type_traits_cpu(wtype);
    return traits && traits->vec_dot && traits->vec_dot_type == input_type;
}

bool CanUseSharedDecodeInputCacheForExpert(const CpuBackend::ExpertWeights& expert, ggml_type input_type) {
    if (!QuantizedProjectionInputTypeMatches(expert.w1_type, input_type)) {
        return false;
    }
    if (expert.w3.ptr || expert.w3_int4.IsValid()) {
        return QuantizedProjectionInputTypeMatches(expert.w3_type, input_type);
    }
    return true;
}

float ReadScalarScaleSidecar(const ggml_tensor* scale_tensor) {
    if (!IsScalarScaleSidecar(scale_tensor)) {
        return 1.0f;
    }
    return *reinterpret_cast<const float*>(scale_tensor->data);
}

void ApplyScalarScaleToTensor(Tensor* tensor, float scale) {
    if (!tensor || !tensor->IsValid() || tensor->dtype != DType::F32 || scale == 1.0f) {
        return;
    }
    float* data = tensor->DataAs<float>();
    if (!data) {
        return;
    }
    size_t total = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        total *= static_cast<size_t>(std::max<int64_t>(1, tensor->shape[i]));
    }
    for (size_t i = 0; i < total; ++i) {
        data[i] *= scale;
    }
}

bool DequantExpertMatrixToF32(const CpuBackend::ExpertWeights& expert, const CpuBackend::ExpertWeight& weight,
                              int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding, int64_t rows,
                              int64_t cols, const ggml_tensor* scale_tensor, std::vector<float>* out,
                              std::string* reason) {
    (void)expert;
    if (!out) {
        return false;
    }
    out->clear();
    if (rows <= 0 || cols <= 0) {
        if (reason) {
            *reason = "missing expert weight";
        }
        return false;
    }
    if (int4_binding.IsValid()) {
        out->resize(static_cast<size_t>(rows * cols));
        if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, out->data())) {
            if (reason) {
                *reason = "packed-int4 MoE reference dequant failed";
            }
            out->clear();
            return false;
        }
        if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
            out->clear();
            return false;
        }
        return true;
    }
    if (!weight.ptr) {
        if (reason) {
            *reason = "missing expert weight";
        }
        return false;
    }
    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    const size_t total = static_cast<size_t>(rows * cols);
    out->resize(total);
    if (wtype == GGML_TYPE_F32) {
        std::memcpy(out->data(), weight.ptr, total * sizeof(float));
        if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
            out->clear();
            return false;
        }
        return true;
    }
    const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
    if (!traits || !traits->to_float) {
        if (reason) {
            *reason = "missing ggml to_float dequantizer";
        }
        return false;
    }
    const size_t row_bytes = ggml_row_size(wtype, cols);
    const char* src = static_cast<const char*>(weight.ptr);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out->data() + r * cols, cols);
    }
    if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, out->data(), reason)) {
        out->clear();
        return false;
    }
    return true;
}

bool BuildMoEReferenceExpertMatrices(const CpuBackend::ExpertWeights& expert, MoEReferenceExpertMatrices* out,
                                     std::string* reason) {
    if (!out) {
        return false;
    }
    out->w1.clear();
    out->w2.clear();
    out->w3.clear();
    out->hidden_dim = expert.hidden_dim;
    out->intermediate_dim = expert.intermediate_dim;
    if (expert.hidden_dim <= 0 || expert.intermediate_dim <= 0) {
        if (reason) {
            *reason = "invalid expert dimensions";
        }
        return false;
    }
    if (!DequantExpertMatrixToF32(expert, expert.w1, expert.w1_type, expert.w1_int4, expert.intermediate_dim,
                                  expert.hidden_dim, nullptr, &out->w1, reason)) {
        return false;
    }
    if (!DequantExpertMatrixToF32(expert, expert.w2, expert.w2_type, expert.w2_int4, expert.hidden_dim,
                                  expert.intermediate_dim, expert.w2_scale_tensor, &out->w2, reason)) {
        return false;
    }
    if ((expert.w3.ptr != nullptr || expert.w3_int4.IsValid()) &&
        !DequantExpertMatrixToF32(expert, expert.w3, expert.w3_type, expert.w3_int4, expert.intermediate_dim,
                                  expert.hidden_dim, nullptr, &out->w3, reason)) {
        return false;
    }
    return true;
}

void RunMoEReferenceCheck(const float* input_data, int batch_size, int hidden_dim, const moe::MoERouteResult& routing,
                          const CpuBackend::ExpertWeights* experts, int num_experts, const float* actual_output) {
    if (!input_data || !experts || !actual_output || batch_size <= 0 || hidden_dim <= 0 || num_experts <= 0) {
        return;
    }
    if (static_cast<int>(routing.expert_ids.size()) != batch_size * routing.top_k ||
        static_cast<int>(routing.weights.size()) != batch_size * routing.top_k) {
        return;
    }

    std::unordered_map<int, MoEReferenceExpertMatrices> cached;
    cached.reserve(static_cast<size_t>(std::min(num_experts, routing.top_k * batch_size)));
    std::vector<float> reference(static_cast<size_t>(batch_size * hidden_dim), 0.0f);
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::string failure_reason;

    for (int i = 0; i < static_cast<int>(routing.expert_ids.size()); ++i) {
        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
        const int token_idx =
            routing.token_indices.empty() ? (i / routing.top_k) : routing.token_indices[static_cast<size_t>(i)];
        const float route_weight = routing.weights[static_cast<size_t>(i)];
        if (expert_id < 0 || expert_id >= num_experts || token_idx < 0 || token_idx >= batch_size) {
            continue;
        }

        auto it = cached.find(expert_id);
        if (it == cached.end()) {
            MoEReferenceExpertMatrices matrices;
            if (!BuildMoEReferenceExpertMatrices(experts[static_cast<size_t>(expert_id)], &matrices, &failure_reason)) {
                std::fprintf(stderr, "[MoE_REF] skipped: expert=%d reason=%s\n", expert_id, failure_reason.c_str());
                return;
            }
            it = cached.emplace(expert_id, std::move(matrices)).first;
        }

        const MoEReferenceExpertMatrices& matrices = it->second;
        const bool use_gelu_activation = experts[static_cast<size_t>(expert_id)].use_gelu_activation;
        const float* token_in = input_data + static_cast<size_t>(token_idx) * hidden_dim;
        gate.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        if (matrices.w3.empty()) {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 1.0f);
        } else {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        }
        hidden.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);

        for (int r = 0; r < matrices.intermediate_dim; ++r) {
            const float* w1_row = matrices.w1.data() + static_cast<size_t>(r) * matrices.hidden_dim;
            const float gate_val = simd::DotF32(token_in, w1_row, matrices.hidden_dim);
            gate[static_cast<size_t>(r)] = gate_val;
            if (!matrices.w3.empty()) {
                const float* w3_row = matrices.w3.data() + static_cast<size_t>(r) * matrices.hidden_dim;
                up[static_cast<size_t>(r)] = simd::DotF32(token_in, w3_row, matrices.hidden_dim);
            }
            const float activated =
                use_gelu_activation ? GeluTanhApprox(gate_val) : (gate_val / (1.0f + std::exp(-gate_val)));
            hidden[static_cast<size_t>(r)] = activated * up[static_cast<size_t>(r)];
        }

        float* ref_out = reference.data() + static_cast<size_t>(token_idx) * hidden_dim;
        for (int r = 0; r < hidden_dim; ++r) {
            const float* w2_row = matrices.w2.data() + static_cast<size_t>(r) * matrices.intermediate_dim;
            ref_out[r] += route_weight * simd::DotF32(hidden.data(), w2_row, matrices.intermediate_dim);
        }
    }

    float max_abs_diff = 0.0f;
    size_t max_idx = 0;
    bool actual_nonfinite = false;
    bool ref_nonfinite = false;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i])) {
            ref_nonfinite = true;
            max_idx = i;
            break;
        }
        if (!std::isfinite(actual_output[i])) {
            actual_nonfinite = true;
            max_idx = i;
            break;
        }
        const float diff = std::fabs(reference[i] - actual_output[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }

    const int token_idx = hidden_dim > 0 ? static_cast<int>(max_idx / static_cast<size_t>(hidden_dim)) : -1;
    const int dim_idx = hidden_dim > 0 ? static_cast<int>(max_idx % static_cast<size_t>(hidden_dim)) : -1;
    std::fprintf(stderr,
                 "[MoE_REF] batch=%d top_k=%d max_abs_diff=%g token=%d dim=%d actual=%g ref=%g actual_nonfinite=%d "
                 "ref_nonfinite=%d tol=%g\n",
                 batch_size, routing.top_k, max_abs_diff, token_idx, dim_idx,
                 max_idx < reference.size() ? actual_output[max_idx] : 0.0f,
                 max_idx < reference.size() ? reference[max_idx] : 0.0f, actual_nonfinite ? 1 : 0,
                 ref_nonfinite ? 1 : 0, MoEReferenceTolerance());
    if (max_abs_diff > MoEReferenceTolerance()) {
        for (int b = 0; b < std::min(batch_size, 2); ++b) {
            std::fprintf(stderr, "[MoE_REF] token=%d routed:", b);
            for (int k = 0; k < routing.top_k; ++k) {
                const size_t idx = static_cast<size_t>(b * routing.top_k + k);
                std::fprintf(stderr, " (%d,%g)", routing.expert_ids[idx], routing.weights[idx]);
            }
            std::fprintf(stderr, "\n");
        }
    }
}

bool ExecuteMoEReferencePath(const float* input_data, int batch_size, int hidden_dim,
                             const moe::MoERouteResult& routing, const CpuBackend::ExpertWeights* experts,
                             int num_experts, float* out_data) {
    if (!input_data || !experts || !out_data || batch_size <= 0 || hidden_dim <= 0 || num_experts <= 0) {
        return false;
    }
    if (static_cast<int>(routing.expert_ids.size()) != batch_size * routing.top_k ||
        static_cast<int>(routing.weights.size()) != batch_size * routing.top_k) {
        return false;
    }

    std::unordered_map<int, MoEReferenceExpertMatrices> cached;
    cached.reserve(static_cast<size_t>(std::min(num_experts, routing.top_k * batch_size)));
    std::fill(out_data, out_data + static_cast<size_t>(batch_size) * static_cast<size_t>(hidden_dim), 0.0f);
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::string failure_reason;

    for (int i = 0; i < static_cast<int>(routing.expert_ids.size()); ++i) {
        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
        const int token_idx =
            routing.token_indices.empty() ? (i / routing.top_k) : routing.token_indices[static_cast<size_t>(i)];
        const float route_weight = routing.weights[static_cast<size_t>(i)];
        if (expert_id < 0 || expert_id >= num_experts || token_idx < 0 || token_idx >= batch_size ||
            route_weight == 0.0f) {
            continue;
        }

        auto it = cached.find(expert_id);
        if (it == cached.end()) {
            MoEReferenceExpertMatrices matrices;
            if (!BuildMoEReferenceExpertMatrices(experts[static_cast<size_t>(expert_id)], &matrices, &failure_reason)) {
                std::fprintf(stderr, "[MoE_REF_EXEC] skipped: expert=%d reason=%s\n", expert_id,
                             failure_reason.c_str());
                return false;
            }
            it = cached.emplace(expert_id, std::move(matrices)).first;
        }

        const MoEReferenceExpertMatrices& matrices = it->second;
        const bool use_gelu_activation = experts[static_cast<size_t>(expert_id)].use_gelu_activation;
        const float* token_in = input_data + static_cast<size_t>(token_idx) * hidden_dim;
        gate.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        if (matrices.w3.empty()) {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 1.0f);
        } else {
            up.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);
        }
        hidden.assign(static_cast<size_t>(matrices.intermediate_dim), 0.0f);

        for (int r = 0; r < matrices.intermediate_dim; ++r) {
            const float* w1_row = matrices.w1.data() + static_cast<size_t>(r) * matrices.hidden_dim;
            const float gate_val = simd::DotF32(token_in, w1_row, matrices.hidden_dim);
            gate[static_cast<size_t>(r)] = gate_val;
            if (!matrices.w3.empty()) {
                const float* w3_row = matrices.w3.data() + static_cast<size_t>(r) * matrices.hidden_dim;
                up[static_cast<size_t>(r)] = simd::DotF32(token_in, w3_row, matrices.hidden_dim);
            }
            const float activated =
                use_gelu_activation ? GeluTanhApprox(gate_val) : (gate_val / (1.0f + std::exp(-gate_val)));
            hidden[static_cast<size_t>(r)] = activated * up[static_cast<size_t>(r)];
        }

        float* ref_out = out_data + static_cast<size_t>(token_idx) * hidden_dim;
        for (int r = 0; r < hidden_dim; ++r) {
            const float* w2_row = matrices.w2.data() + static_cast<size_t>(r) * matrices.intermediate_dim;
            ref_out[r] += route_weight * simd::DotF32(hidden.data(), w2_row, matrices.intermediate_dim);
        }
    }

    return true;
}

void RecordMoEReferencePathTrace(CpuBackend* backend, int layer_idx, const BatchSpec* batch,
                                 const moe::MoERouteResult& routing, int num_experts) {
    if (!backend) {
        return;
    }
    const char projections[3] = {'1', '3', '2'};
    for (size_t i = 0; i < routing.expert_ids.size(); ++i) {
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int token_idx = routing.token_indices.empty() ? static_cast<int>(i / static_cast<size_t>(routing.top_k))
                                                            : routing.token_indices[i];
        for (char projection : projections) {
            CpuBackend::MoEPathTraceEntry entry;
            entry.layer_idx = layer_idx;
            entry.token_idx = token_idx;
            entry.expert_id = expert_id;
            entry.force_safe_reference = true;
            entry.safe_reference_mode = true;
            entry.projection[0] = projection;
            entry.projection[1] = '\0';
            entry.selected_path = CpuBackend::MoEProjectionPath::ReferenceF32;
            if (batch && token_idx >= 0 && token_idx < static_cast<int>(batch->seq_id.size())) {
                entry.seq_id = batch->seq_id[static_cast<size_t>(token_idx)];
                if (entry.seq_id >= 0 && entry.seq_id < static_cast<int>(batch->n_past.size())) {
                    entry.n_past = batch->n_past[static_cast<size_t>(entry.seq_id)];
                }
            }
            backend->RecordMoEPathTrace(entry);
        }
    }
}

template <size_t N> bool CopyIntVectorToFixedArray(const std::vector<int>& src, std::array<int, N>* dst, int* count) {
    if (!dst || !count) {
        return false;
    }
    if (src.size() > N) {
        *count = 0;
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i) {
        (*dst)[i] = src[i];
    }
    *count = static_cast<int>(src.size());
    return true;
}

template <size_t N> bool FixedArrayContains(const std::array<int, N>& values, int count, int target) {
    for (int i = 0; i < count; ++i) {
        if (values[static_cast<size_t>(i)] == target) {
            return true;
        }
    }
    return false;
}

bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node, bool allow_parallel,
                                CpuBackend::MoEProjectionPath* selected_path = nullptr);

bool ExpertUsesPackedInt4Only(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1_int4.IsValid() || !expert.w2_int4.IsValid()) {
        return false;
    }
    if (expert.w3.ptr != nullptr && !expert.w3_int4.IsValid()) {
        return false;
    }
    return true;
}

// Check if expert weights are in a ggml quantized format that supports vec_dot.
// Examples: Q4_K, Q4_0, Q6_K. This enables zero-dequantization GEMV.
bool ExpertHasGgmlQuantizedWeights(const CpuBackend::ExpertWeights& expert) {
    if (!expert.w1.ptr || !expert.w2.ptr) return false;
    const auto check_type = [](int type_id) -> bool {
        if (type_id == GGML_TYPE_F32) return false;
        const ggml_type wtype = static_cast<ggml_type>(type_id);
        if (!ggml_is_quantized(wtype)) return false;
        const auto* tc = ggml_get_type_traits_cpu(wtype);
        return tc && tc->vec_dot;
    };
    return check_type(expert.w1_type) && check_type(expert.w2_type) &&
           (expert.w3.ptr == nullptr || check_type(expert.w3_type));
}

// Run ggml vec_dot based GEMV for quantized expert weights.
// This bypasses F32 dequantization on the hot MoE path.
// Complexity: O(M * N * K) where expert matrices are [N, K].
bool TryRunGgmlQuantizedProjection(CpuBackend* backend, const void* weight_ptr, int ggml_type_id, const Tensor& input,
                                   Tensor* output, int64_t N, int64_t K, int numa_node, bool allow_parallel = true,
                                   QuantizedProjectionInputCache* input_cache = nullptr,
                                   bool allow_kquant_rowpair_vec_dot = true) {
    if (!backend || !weight_ptr || !output || !input.IsValid() || !output->IsValid()) return false;
    if (input.dtype != DType::F32 || output->dtype != DType::F32) return false;
    if (input.ndim != 2 || output->ndim != 2) return false;

    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    if (wtype == GGML_TYPE_F32 || !ggml_is_quantized(wtype)) return false;
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, wtype);
    const bool dispatch_census_enabled = []() {
        const char* env = std::getenv("DENSECORE_MATMUL_DISPATCH_CENSUS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "off") != 0;
    }();
    const auto dispatch_begin = dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                        : std::chrono::steady_clock::time_point{};
    const auto record_dispatch = [&](const char* path) {
        if (!dispatch_census_enabled) {
            return;
        }
        const uint64_t wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - dispatch_begin)
                .count());
        RecordMatmulDispatchCensus(census_ctx, GetCurrentExecutionPhase(), path, wtype, input.shape[0], N, K,
                                   wall_ns);
    };

    const auto* type_traits = ggml_get_type_traits(wtype);
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
    if (!type_traits || !type_traits_cpu || !type_traits_cpu->vec_dot) return false;

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) return false;

    // Resolve the input quantization type required by vec_dot
    const ggml_type iq_type = type_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) return false;

    const size_t w_row_bytes = ggml_row_size(wtype, K);
    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const char* w_data = static_cast<const char*>(weight_ptr);

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        // Quantize the active expert batch once, then reuse it for every expert row.
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }
    const bool q4k_repacked_candidate =
        wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const bool use_kquant_rowpair_vec_dot = allow_kquant_rowpair_vec_dot && CanUseKQuantRowPairVecDotFastPath() &&
                                            (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q6_K) &&
                                            iq_type == GGML_TYPE_Q8_K && M >= 2 && K % ggml_blck_size(wtype) == 0 &&
                                            N >= 2;
    if (CanUseMoEQ4KRawBatchedScalar() && wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && M > 1 &&
        K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedProjection(backend, weight_ptr, qinput_data, iq_row_bytes, out_data, M, N, K, numa_node,
                                          allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            record_dispatch("moe_expert");
            return true;
        }
    }
    if (CanUseQ4KRepackedMoEPrefillFastPath() && wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && M > 1 &&
        (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto packed = GetOrCreateQ4KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && M >= 4 &&
            RunQ4KRepackedMoEGemmM4(backend, packed, in_data, out_data, M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_prefill_gemm_m4", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
        if (packed && RunQ4KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_prefill_gemv", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
    }
    if (use_kquant_rowpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 16, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        }
        const char* path_name = wtype == GGML_TYPE_Q4_K ? "ggml_q4k_rowpair_m2_vecdot" : "ggml_q6k_rowpair_m2_vecdot";
        LogMoEMatmulPath(path_name, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        record_dispatch("moe_rowblock");
        return true;
    }

    const bool use_q5k_colpair_vec_dot = allow_kquant_rowpair_vec_dot && CanUseKQuantRowPairVecDotFastPath() &&
                                         wtype == GGML_TYPE_Q5_K && iq_type == GGML_TYPE_Q8_K &&
                                         K % ggml_blck_size(GGML_TYPE_Q5_K) == 0 && N >= 2;
    if (use_q5k_colpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 16, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        }
        LogMoEMatmulPath("ggml_q5k_colpair_vecdot", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                         allow_parallel);
        record_dispatch("moe_rowblock");
        return true;
    }

    if (wtype == GGML_TYPE_Q6_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q6_K)) == 0) {
        auto packed = GetOrCreateQ6KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && RunQ6KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q6k_repacked_gemv", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_dispatch("moe_expert");
            return true;
        }
    }

    if (wtype == GGML_TYPE_Q5_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q5_K)) == 0) {
        auto packed = GetOrCreateQ5KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && RunQ5KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q5k_repacked_gemv", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_dispatch("moe_expert");
            return true;
        }
    }

    if (wtype == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto packed = GetOrCreateQ4KRepackedMoEWeight(weight_ptr, N, K);
        if (packed && RunQ4KRepackedMoEGemv(backend, packed, qinput_data, iq_row_bytes, out_data, M, N, numa_node,
                                            allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_gemv", static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0,
                             allow_parallel);
            record_q4k_repacked(true, nullptr);
            record_dispatch("q4k_repacked_gemv");
            return true;
        }
    }

    const bool use_q4k_rowpair_vec_dot =
        allow_kquant_rowpair_vec_dot && CanUseQ4KRowPairVecDotFastPath() &&
        (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q5_K || wtype == GGML_TYPE_Q5_1 ||
         wtype == GGML_TYPE_Q8_0) &&
        (iq_type == GGML_TYPE_Q8_K || iq_type == GGML_TYPE_Q8_1 || iq_type == GGML_TYPE_Q8_0) && M == 1 &&
        K % ggml_blck_size(wtype) == 0 && N >= 4;
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    for (int64_t m = 0; m < M; ++m) {
        float* out_row = out_data + m * N;
        const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;

        if (use_q4k_rowpair_vec_dot) {
            const int64_t pair_count = N / 2;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    float sums[32] = {};
                    type_traits_cpu->vec_dot(static_cast<int>(K), sums, 16, w_row, w_row_bytes, qi, 0, 2);
                    out_row[n] = sums[0];
                    out_row[n + 1] = sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        } else if (n_threads <= 1) {
            for (int64_t n = 0; n < N; ++n) {
                const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
            }
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) {
                for (int n = n_start; n < n_end; ++n) {
                    const void* w_row = w_data + static_cast<size_t>(n) * w_row_bytes;
                    type_traits_cpu->vec_dot(static_cast<int>(K), out_row + n, 0, w_row, 0, qi, 0, 1);
                }
            });
        }
    }
    if (use_q4k_rowpair_vec_dot) {
        const char* path = wtype == GGML_TYPE_Q5_K   ? "ggml_q5k_rowpair_m1_vecdot"
                           : wtype == GGML_TYPE_Q5_1 ? "ggml_q5_1_rowpair_m1_vecdot"
                           : wtype == GGML_TYPE_Q8_0 ? "ggml_q8_0_rowpair_m1_vecdot"
                                                      : "ggml_q4k_rowpair_m1_vecdot";
        LogMoEMatmulPath(path, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
    }
    record_q4k_repacked(false, use_q4k_rowpair_vec_dot ? "rowpair_used" : "native_vecdot");
    record_dispatch(use_q4k_rowpair_vec_dot ? "moe_rowblock" : "moe_expert");
    return true;
}

bool TryRunGgmlQuantizedFusedGEGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, int gate_ggml_type_id,
                                              const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                              Tensor* output, int64_t N, int64_t K, int numa_node,
                                              bool allow_parallel = true,
                                              QuantizedProjectionInputCache* input_cache = nullptr) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !output || !input.IsValid() || !output->IsValid()) {
        return false;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    const ggml_type gate_type = static_cast<ggml_type>(gate_ggml_type_id);
    const ggml_type up_type = static_cast<ggml_type>(up_ggml_type_id);
    if (gate_type == GGML_TYPE_F32 || up_type == GGML_TYPE_F32 || !ggml_is_quantized(gate_type) ||
        !ggml_is_quantized(up_type)) {
        return false;
    }

    const auto* gate_traits = ggml_get_type_traits(gate_type);
    const auto* up_traits = ggml_get_type_traits(up_type);
    const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
    const auto* up_traits_cpu = ggml_get_type_traits_cpu(up_type);
    if (!gate_traits || !up_traits || !gate_traits_cpu || !up_traits_cpu || !gate_traits_cpu->vec_dot ||
        !up_traits_cpu->vec_dot) {
        return false;
    }
    if (gate_traits_cpu->vec_dot_type != up_traits_cpu->vec_dot_type) {
        return false;
    }

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) {
        return false;
    }

    const ggml_type iq_type = gate_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) {
        return false;
    }

    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const size_t gate_row_bytes = ggml_row_size(gate_type, K);
    const size_t up_row_bytes = ggml_row_size(up_type, K);
    const char* gate_data = static_cast<const char*>(gate_weight_ptr);
    const char* up_data = static_cast<const char*>(up_weight_ptr);
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, gate_type);
    RecordMoEExpertMatmulWeightType(census_ctx, up_type);
    const bool q4k_repacked_candidate =
        gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }

    if (CanUseMoEQ4KRawBatchedScalar() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedFusedGEGLU(backend, gate_weight_ptr, up_weight_ptr, qinput_data, iq_row_bytes, out_data,
                                          M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched_fused_geglu", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            return true;
        }
    }
    if (CanUseQ4KRepackedMoEGEGLUFastPath() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M == 1 && (N % 8) == 0 && (K % QK_K) == 0) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed &&
            RunQ4KRepackedMoEFusedGEGLU(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes, out_data,
                                        M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_repacked_fused_geglu", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(true, nullptr);
            return true;
        }
    }
    const bool use_kquant_rowpair_vec_dot =
        CanUseQ4KRowPairVecDotFastPath() && gate_type == up_type &&
        IsKQuantRowPairGatedProjectionType(gate_type, iq_type) &&
        gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M == 1 &&
        K % ggml_blck_size(gate_type) == 0 && N >= 4;
    if (use_kquant_rowpair_vec_dot) {
        auto& pool = backend->GetThreadPool(numa_node);
        const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
        const void* qi = qinput_data;
        float* out_row = out_data;
        const int64_t pair_count = N / 2;
        const auto compute_pair_range = [&](int pair_start, int pair_end) {
            for (int pair = pair_start; pair < pair_end; ++pair) {
                const int64_t n = static_cast<int64_t>(pair) * 2;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                float gate_sums[32] = {};
                float up_sums[32] = {};
                gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 16, gate_row, gate_row_bytes, qi, 0, 2);
                up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 16, up_row, up_row_bytes, qi, 0, 2);
                out_row[n] = GeluTanhApprox(gate_sums[0]) * up_sums[0];
                out_row[n + 1] = GeluTanhApprox(gate_sums[1]) * up_sums[1];
            }
        };
        if (n_threads <= 1) {
            compute_pair_range(0, static_cast<int>(pair_count));
        } else {
            pool.ParallelFor(static_cast<int>(pair_count),
                             [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
        }
        if ((N & 1) != 0) {
            const int64_t n = N - 1;
            const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
            const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
            up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
            out_row[n] = GeluTanhApprox(gate_sum) * up_sum;
        }
        const char* path = gate_type == GGML_TYPE_Q5_K   ? "ggml_q5k_rowpair_m1_fused_geglu"
                           : gate_type == GGML_TYPE_Q5_1 ? "ggml_q5_1_rowpair_m1_fused_geglu"
                                                         : "ggml_q4k_rowpair_m1_fused_geglu";
        LogMoEMatmulPath(path, static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        return true;
    }
    return false;
}

bool TryRunGgmlQuantizedFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, int gate_ggml_type_id,
                                              const void* up_weight_ptr, int up_ggml_type_id, const Tensor& input,
                                              Tensor* output, int64_t N, int64_t K, int numa_node,
                                              bool allow_parallel = true,
                                              QuantizedProjectionInputCache* input_cache = nullptr) {
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !output || !input.IsValid() || !output->IsValid()) {
        return false;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    const ggml_type gate_type = static_cast<ggml_type>(gate_ggml_type_id);
    const ggml_type up_type = static_cast<ggml_type>(up_ggml_type_id);
    if (gate_type == GGML_TYPE_F32 || up_type == GGML_TYPE_F32 || !ggml_is_quantized(gate_type) ||
        !ggml_is_quantized(up_type)) {
        return false;
    }

    const auto* gate_traits = ggml_get_type_traits(gate_type);
    const auto* up_traits = ggml_get_type_traits(up_type);
    const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
    const auto* up_traits_cpu = ggml_get_type_traits_cpu(up_type);
    if (!gate_traits || !up_traits || !gate_traits_cpu || !up_traits_cpu || !gate_traits_cpu->vec_dot ||
        !up_traits_cpu->vec_dot) {
        return false;
    }
    if (gate_traits_cpu->vec_dot_type != up_traits_cpu->vec_dot_type) {
        return false;
    }

    const int64_t M = input.shape[0];
    if (M <= 0 || M > kMoEQuantizedProjectionMaxBatch || K != input.shape[1] || N != output->shape[1]) {
        return false;
    }

    const ggml_type iq_type = gate_traits_cpu->vec_dot_type;
    const auto* iq_traits = ggml_get_type_traits_cpu(iq_type);
    if (!iq_traits || !iq_traits->from_float) {
        return false;
    }

    const size_t gate_row_bytes = ggml_row_size(gate_type, K);
    const size_t up_row_bytes = ggml_row_size(up_type, K);
    const size_t iq_row_bytes = ggml_row_size(iq_type, K);
    const float* in_data = input.DataAs<float>();
    float* out_data = output->DataAs<float>();
    const char* gate_data = static_cast<const char*>(gate_weight_ptr);
    const char* up_data = static_cast<const char*>(up_weight_ptr);
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    RecordMoEExpertMatmulWeightType(census_ctx, gate_type);
    RecordMoEExpertMatmulWeightType(census_ctx, up_type);
    const bool q4k_repacked_candidate =
        gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K && (N % 8) == 0 &&
        (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0;
    bool q4k_repacked_reported = false;
    const auto record_q4k_repacked = [&](bool used, const char* reject_reason) {
        if (q4k_repacked_candidate && !q4k_repacked_reported) {
            RecordMoEQ4KRepackedDecision(census_ctx, true, used, reject_reason);
            q4k_repacked_reported = true;
        }
    };

    const size_t total_qbytes = static_cast<size_t>(M) * iq_row_bytes;
    uint8_t* qinput_data = nullptr;
    if (input_cache) {
        const bool cache_hit = input_cache->source == in_data && input_cache->rows == M && input_cache->cols == K &&
                               input_cache->type == iq_type && input_cache->row_bytes == iq_row_bytes &&
                               input_cache->bytes.size() >= total_qbytes;
        if (!cache_hit) {
            input_cache->source = in_data;
            input_cache->rows = M;
            input_cache->cols = K;
            input_cache->type = iq_type;
            input_cache->row_bytes = iq_row_bytes;
            input_cache->bytes.resize(total_qbytes);
            for (int64_t m = 0; m < M; ++m) {
                iq_traits->from_float(in_data + m * K,
                                      input_cache->bytes.data() + static_cast<size_t>(m) * iq_row_bytes, K);
            }
        }
        qinput_data = input_cache->bytes.data();
    } else {
        static thread_local std::vector<uint8_t> qinput_buf;
        if (qinput_buf.size() < total_qbytes) qinput_buf.resize(total_qbytes);
        for (int64_t m = 0; m < M; ++m) {
            iq_traits->from_float(in_data + m * K, qinput_buf.data() + static_cast<size_t>(m) * iq_row_bytes, K);
        }
        qinput_data = qinput_buf.data();
    }

    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;
    const bool use_q4k_rowpair_m2_vec_dot = CanUseKQuantRowPairVecDotFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                            up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
                                            gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M >= 2 &&
                                            K % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && N >= 2;
    if (CanUseMoEQ4KRawBatchedScalar() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M > 1 && K % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
        if (RunMoEQ4KRawBatchedFusedSwiGLU(backend, gate_weight_ptr, up_weight_ptr, qinput_data, iq_row_bytes, out_data,
                                           M, N, K, numa_node, allow_parallel)) {
            LogMoEMatmulPath("ggml_q4k_raw_batched_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                             static_cast<int>(N), 0, allow_parallel);
            record_q4k_repacked(false, "raw_batched_used");
            return true;
        }
    }
    if (CanUseQ4KRepackedMoEPrefillFastPath() && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K &&
        iq_type == GGML_TYPE_Q8_K && M > 1 && (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed) {
            if (RunQ4KRepackedMoEFusedSwiGLUM4(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes,
                                               out_data, M, N, K, numa_node, allow_parallel)) {
                LogMoEMatmulPath(M >= 4 ? "ggml_q4k_repacked_prefill_gemm_m4_tile_fused_swiglu"
                                        : "ggml_q4k_repacked_prefill_tile_fused_swiglu",
                                 static_cast<int>(M), static_cast<int>(K), static_cast<int>(N), 0, allow_parallel);
                record_q4k_repacked(true, nullptr);
                return true;
            }
        }
    }
    if (use_q4k_rowpair_m2_vec_dot) {
        const int64_t pair_count = N / 2;
        for (int64_t m = 0; m < M; ++m) {
            float* out_row = out_data + static_cast<size_t>(m) * N;
            const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                    const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                    float gate_sums[32] = {};
                    float up_sums[32] = {};
                    gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 16, gate_row, gate_row_bytes, qi, 0, 2);
                    up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 16, up_row, up_row_bytes, qi, 0, 2);
                    out_row[n] = (gate_sums[0] / (1.0f + internal::FastExp(-gate_sums[0]))) * up_sums[0];
                    out_row[n + 1] = (gate_sums[1] / (1.0f + internal::FastExp(-gate_sums[1]))) * up_sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
        }
        LogMoEMatmulPath("ggml_q4k_rowpair_m2_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                         static_cast<int>(N), 0, allow_parallel);
        record_q4k_repacked(false, "rowpair_used");
        return true;
    }
    if (M == 1 && gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
        (N % 8) == 0 && (K % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
        auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, N, K);
        auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, N, K);
        if (gate_packed && up_packed) {
            if (RunQ4KRepackedMoEFusedSwiGLUM4(backend, gate_packed, up_packed, in_data, qinput_data, iq_row_bytes,
                                               out_data, M, N, K, numa_node, allow_parallel)) {
                LogMoEMatmulPath("ggml_q4k_repacked_tile_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                                 static_cast<int>(N), 0, allow_parallel);
                record_q4k_repacked(true, nullptr);
                return true;
            }
        }
    }
    const bool use_q4k_rowpair_vec_dot = CanUseQ4KRowPairVecDotFastPath() && gate_type == GGML_TYPE_Q4_K &&
                                         up_type == GGML_TYPE_Q4_K && iq_type == GGML_TYPE_Q8_K &&
                                         gate_traits_cpu->vec_dot == up_traits_cpu->vec_dot && M == 1 &&
                                         K % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && N >= 4;

    for (int64_t m = 0; m < M; ++m) {
        float* out_row = out_data + m * N;
        const void* qi = qinput_data + static_cast<size_t>(m) * iq_row_bytes;

        if (use_q4k_rowpair_vec_dot) {
            const int64_t pair_count = N / 2;
            const auto compute_pair_range = [&](int pair_start, int pair_end) {
                for (int pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t n = static_cast<int64_t>(pair) * 2;
                    const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                    const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                    float gate_sums[32] = {};
                    float up_sums[32] = {};
                    gate_traits_cpu->vec_dot(static_cast<int>(K), gate_sums, 16, gate_row, gate_row_bytes, qi, 0, 2);
                    up_traits_cpu->vec_dot(static_cast<int>(K), up_sums, 16, up_row, up_row_bytes, qi, 0, 2);
                    out_row[n] = (gate_sums[0] / (1.0f + internal::FastExp(-gate_sums[0]))) * up_sums[0];
                    out_row[n + 1] = (gate_sums[1] / (1.0f + internal::FastExp(-gate_sums[1]))) * up_sums[1];
                }
            };
            if (n_threads <= 1) {
                compute_pair_range(0, static_cast<int>(pair_count));
            } else {
                pool.ParallelFor(static_cast<int>(pair_count),
                                 [&](int pair_start, int pair_end, int) { compute_pair_range(pair_start, pair_end); });
            }
            if ((N & 1) != 0) {
                const int64_t n = N - 1;
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
            continue;
        }

        const auto compute_range = [&](int n_start, int n_end) {
            for (int n = n_start; n < n_end; ++n) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const void* gate_row = gate_data + static_cast<size_t>(n) * gate_row_bytes;
                const void* up_row = up_data + static_cast<size_t>(n) * up_row_bytes;
                gate_traits_cpu->vec_dot(static_cast<int>(K), &gate_sum, 0, gate_row, 0, qi, 0, 1);
                up_traits_cpu->vec_dot(static_cast<int>(K), &up_sum, 0, up_row, 0, qi, 0, 1);
                out_row[n] = (gate_sum / (1.0f + internal::FastExp(-gate_sum))) * up_sum;
            }
        };

        if (n_threads <= 1) {
            compute_range(0, static_cast<int>(N));
        } else {
            pool.ParallelFor(static_cast<int>(N), [&](int n_start, int n_end, int) { compute_range(n_start, n_end); });
        }
    }
    if (use_q4k_rowpair_vec_dot) {
        LogMoEMatmulPath("ggml_q4k_rowpair_m1_fused_swiglu", static_cast<int>(M), static_cast<int>(K),
                         static_cast<int>(N), 0, allow_parallel);
    }
    record_q4k_repacked(false, use_q4k_rowpair_vec_dot ? "rowpair_used" : "native_vecdot");
    return true;
}

bool TryRunPackedInt4ProjectionDirect(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                      const Tensor& input, Tensor* output, int numa_node, bool allow_parallel = true) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    // All platforms: Highway INT4 GEMV kernels with runtime ISA dispatch
    // (NEON/SVE2 on ARM, AVX2/AVX-512 on x86)
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || K != binding.K || N != binding.N) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    if (!IsArmPackedInt4HwyProjectionSupported()) {
        (void)backend;
        (void)numa_node;
        (void)allow_parallel;
        return false;
    }
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    if (M == 1) {
        if (n_threads <= 1) {
            densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                 binding.zeros, K, N, binding.group_size, 0, N);
        } else {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemvInt4_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                     binding.zeros, K, N, binding.group_size, n_start, n_end);
            });
        }
        return true;
    }

    if (M <= 4) {
        const size_t input_stride_bytes = static_cast<size_t>(K) * sizeof(float);
        const bool should_parallelize = n_threads > 1 && N >= 128;
        if (should_parallelize) {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights,
                                                            binding.scales, binding.zeros, M, K, N, binding.group_size,
                                                            0, M, n_start, n_end, input_stride_bytes);
            });
        } else {
            densecore::hwy_kernels::GemmInt4Batched_Hwy(output_data, input_data, binding.packed_weights, binding.scales,
                                                        binding.zeros, M, K, N, binding.group_size, 0, M, 0, N,
                                                        input_stride_bytes);
        }
        return true;
    }

    return false;
}

bool TryRunPackedInt4FusedSwiGLUProjectionDirect(CpuBackend* backend,
                                                 const CpuBackend::ExpertPackedInt4Weight& gate_binding,
                                                 const CpuBackend::ExpertPackedInt4Weight& up_binding,
                                                 const Tensor& input, Tensor* output, int numa_node,
                                                 bool allow_parallel = true) {
    if (!backend || !output || !gate_binding.IsValid() || !up_binding.IsValid() || !input.IsValid() ||
        !output->IsValid() || input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 ||
        output->ndim != 2) {
        return false;
    }

    // All platforms: Highway fused SwiGLU INT4 kernels with runtime ISA dispatch
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[1]);
    const int N = static_cast<int>(output->shape[1]);
    if (M <= 0 || M > 4 || gate_binding.K != K || up_binding.K != K || gate_binding.N != N || up_binding.N != N ||
        gate_binding.group_size != up_binding.group_size) {
        return false;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    if (!IsArmPackedInt4HwyProjectionSupported()) {
        (void)backend;
        (void)numa_node;
        (void)allow_parallel;
        return false;
    }
#endif

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();
    auto& pool = backend->GetThreadPool(numa_node);
    const int n_threads = allow_parallel ? pool.GetNumThreads() : 1;

    for (int m = 0; m < M; ++m) {
        const float* row_input = input_data + static_cast<size_t>(m) * K;
        float* row_output = output_data + static_cast<size_t>(m) * N;
        if (n_threads <= 1) {
            densecore::hwy_kernels::GemvInt4DualFusedSilu_Hwy(
                row_output, row_input, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                up_binding.packed_weights, up_binding.scales, up_binding.zeros, K, N, gate_binding.group_size, 0, N);
        } else {
            pool.ParallelFor(N, [&](int n_start, int n_end, int /*thread_id*/) {
                densecore::hwy_kernels::GemvInt4DualFusedSilu_Hwy(
                    row_output, row_input, gate_binding.packed_weights, gate_binding.scales, gate_binding.zeros,
                    up_binding.packed_weights, up_binding.scales, up_binding.zeros, K, N, gate_binding.group_size,
                    n_start, n_end);
            });
        }
    }
    return true;
}

void DispatchExpertFFNImpl(CpuBackend* backend, int numa_node, const Tensor& input,
                           const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                           const Tensor& w3, Tensor* output, bool allow_inner_parallel = true,
                           const MoEExecutionTraceContext* trace_ctx = nullptr,
                           QuantizedProjectionInputCache* shared_input_projection_cache = nullptr,
                           CpuBackend::MoEForwardProfile* profile = nullptr) {
    if (!backend || !output) {
        return;
    }

    const bool debug_ffn_timing = IsMoEFFNDebugTimingEnabled();
    const auto total_begin =
        debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;
    QuantizedProjectionInputCache local_input_projection_cache;
    QuantizedProjectionInputCache* input_projection_cache =
        shared_input_projection_cache ? shared_input_projection_cache : &local_input_projection_cache;

    const int64_t batch = input.shape[0];
    const int64_t intermediate_dim =
        expert.intermediate_dim > 0 ? static_cast<int64_t>(expert.intermediate_dim) : (w1.IsValid() ? w1.shape[0] : 0);
    if (intermediate_dim <= 0) return;
    const int64_t hidden_dim = input.shape[1];
    const size_t hidden_size = static_cast<size_t>(batch * intermediate_dim);
    const bool enable_inner_parallel =
        allow_inner_parallel && ShouldParallelizeExpertFFNInner(batch, hidden_dim, intermediate_dim);
    const bool safe_reference_mode = IsMoESafeReferenceModeEnabled(&expert);
    const bool ggml_quantized_vecdot_safe = batch == 1;

    hidden_scratch.Resize(backend, hidden_size);
    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);
    const bool profile_enabled = profile != nullptr;
    const auto w1w3_profile_begin =
        profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool used_fused_int4_swiglu =
        !safe_reference_mode && !expert.use_gelu_activation &&
        TryRunPackedInt4FusedSwiGLUProjectionDirect(backend, expert.w1_int4, expert.w3_int4, input, &hidden, numa_node,
                                                    enable_inner_parallel);
    if (used_fused_int4_swiglu) {
        GetMoEInt4PathHistogram().fused_swiglu_hwy.fetch_add(1, std::memory_order_relaxed);
        LogMoEMatmulPath("fused_swiglu_hwy", static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                         static_cast<int>(hidden.shape[1]), expert.w1_int4.group_size, enable_inner_parallel);
    }
	    const bool used_fused_ggml_quant_swiglu =
        !used_fused_int4_swiglu && !safe_reference_mode && ggml_quantized_vecdot_safe &&
        !expert.use_gelu_activation && expert.w1.ptr &&
        expert.w3.ptr &&
        TryRunGgmlQuantizedFusedSwiGLUProjection(backend, expert.w1.ptr, expert.w1_type, expert.w3.ptr, expert.w3_type,
                                                 input, &hidden, intermediate_dim, hidden_dim, numa_node,
                                                 enable_inner_parallel, input_projection_cache);
    if (used_fused_ggml_quant_swiglu) {
	        LogMoEMatmulPath("ggml_quantized_fused_swiglu", static_cast<int>(input.shape[0]),
	                         static_cast<int>(input.shape[1]), static_cast<int>(hidden.shape[1]), 0, enable_inner_parallel);
	    }
    const bool used_fused_ggml_quant_geglu =
        !used_fused_int4_swiglu && !used_fused_ggml_quant_swiglu && !safe_reference_mode &&
        ggml_quantized_vecdot_safe &&
        expert.use_gelu_activation && expert.w1.ptr && expert.w3.ptr &&
        TryRunGgmlQuantizedFusedGEGLUProjection(backend, expert.w1.ptr, expert.w1_type, expert.w3.ptr, expert.w3_type,
                                                input, &hidden, intermediate_dim, hidden_dim, numa_node,
                                                enable_inner_parallel, input_projection_cache);
    if (used_fused_ggml_quant_geglu) {
        LogMoEMatmulPath("ggml_quantized_fused_geglu", static_cast<int>(input.shape[0]),
                         static_cast<int>(input.shape[1]), static_cast<int>(hidden.shape[1]), 0, enable_inner_parallel);
    }
    const bool used_fused_gate_up =
        used_fused_int4_swiglu || used_fused_ggml_quant_swiglu || used_fused_ggml_quant_geglu;
	    if (!used_fused_gate_up &&
	        (w3.IsValid() || expert.w3_int4.IsValid() || expert.w3.ptr)) {
	        gate_scratch.Resize(backend, hidden_size);
	    }

    // Unified projection dispatch: packed INT4 -> ggml quantized GEMV -> F32 dense
    const auto run_projection = [&](const char projection_slot, const Tensor& src, const Tensor& dense_weight,
                                    const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                    const CpuBackend::ExpertWeight& raw_weight, int ggml_type_id, int64_t proj_rows,
                                    int64_t proj_cols, Tensor* dst) {
        auto record_path = [&](const char* path_name) {
            if (!trace_ctx) {
                return;
            }
            CpuBackend::MoEPathTraceEntry entry;
            entry.layer_idx = trace_ctx->layer_idx;
            entry.seq_id = trace_ctx->seq_id;
            entry.token_idx = trace_ctx->token_idx;
            entry.decode_step = trace_ctx->decode_step;
            entry.n_past = trace_ctx->n_past;
            entry.expert_id = trace_ctx->expert_id;
            entry.force_safe_reference = expert.force_safe_reference;
            entry.safe_reference_mode = safe_reference_mode;
            entry.projection[0] = projection_slot;
            entry.projection[1] = '\0';
            entry.selected_path = ParseMoEProjectionPath(path_name);
            backend->RecordMoEPathTrace(entry);
        };
        const bool projection_has_scale = projection_slot == '2' && expert.w2_scale_tensor != nullptr;
        const bool projection_scalar_scale = projection_has_scale && IsScalarScaleSidecar(expert.w2_scale_tensor);
        const bool quantized_scale_supported = !projection_has_scale || projection_scalar_scale;
        // Path 1: Packed INT4 (custom DenseCore format)
        CpuBackend::MoEProjectionPath packed_path = CpuBackend::MoEProjectionPath::Unknown;
        if (!safe_reference_mode && quantized_scale_supported &&
            TryRunPackedInt4Projection(backend, int4_binding, src, dst, numa_node, enable_inner_parallel,
                                       &packed_path)) {
            if (projection_scalar_scale) {
                ApplyScalarScaleToTensor(dst, ReadScalarScaleSidecar(expert.w2_scale_tensor));
            }
            record_path(packed_path == CpuBackend::MoEProjectionPath::PackedInt4Fast ? "direct_hwy"
                                                                                     : "backend_gemmint4");
            return;
        }
        // Path 2: Native ggml quantized GEMV (Q4_K, Q4_0, etc.) -> zero dequantization
        const bool allow_rowpair_vec_dot =
            !(expert.use_gelu_activation && projection_slot == '2' && ggml_type_id == GGML_TYPE_Q4_K);
        if (!safe_reference_mode && ggml_quantized_vecdot_safe && quantized_scale_supported && raw_weight.ptr &&
            ggml_type_id != GGML_TYPE_F32 &&
            TryRunGgmlQuantizedProjection(
                backend, raw_weight.ptr, ggml_type_id, src, dst, proj_rows, proj_cols, numa_node, enable_inner_parallel,
                src.DataAs<float>() == input.DataAs<float>() ? input_projection_cache : nullptr,
                allow_rowpair_vec_dot)) {
            if (projection_scalar_scale) {
                ApplyScalarScaleToTensor(dst, ReadScalarScaleSidecar(expert.w2_scale_tensor));
            }
            LogMoEMatmulPath("ggml_quantized_vecdot", static_cast<int>(src.shape[0]), static_cast<int>(src.shape[1]),
                             static_cast<int>(dst->shape[1]), int4_binding.group_size, enable_inner_parallel);
            record_path("ggml_quantized_vecdot");
            return;
        }
        // Path 3: F32 dense matmul fallback (requires pre-dequantized weight)
        if (dense_weight.IsValid()) {
            GetMoEInt4PathHistogram().f32_fallback.fetch_add(1, std::memory_order_relaxed);
            LogMoEMatmulPath(safe_reference_mode ? "reference_f32" : "dense_f32", static_cast<int>(src.shape[0]),
                             static_cast<int>(src.shape[1]), static_cast<int>(dst->shape[1]), int4_binding.group_size,
                             enable_inner_parallel);
            record_path(safe_reference_mode ? "reference_f32" : "dense_f32");
            backend->MatMulTransB(src, dense_weight, dst, numa_node);
        }
    };

    std::chrono::steady_clock::duration w1_duration{};
    std::chrono::steady_clock::duration gate_duration{};
    std::chrono::steady_clock::duration activation_duration{};
    std::chrono::steady_clock::duration w2_duration{};

	    if (!used_fused_gate_up) {
        const auto w1_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_projection('1', input, w1, expert.w1_int4, expert.w1, expert.w1_type, intermediate_dim, hidden_dim,
                       &hidden);
        LogGemma4MoETensorStats("w1", hidden, trace_ctx);
        if (debug_ffn_timing) {
            w1_duration += (std::chrono::steady_clock::now() - w1_begin);
        }
    }

	    if (!used_fused_gate_up &&
	        (w3.IsValid() || expert.w3_int4.IsValid() || expert.w3.ptr)) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        const auto gate_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        run_projection('3', input, w3, expert.w3_int4, expert.w3, expert.w3_type, intermediate_dim, hidden_dim, &gate);
        LogGemma4MoETensorStats("w3", gate, trace_ctx);
        if (debug_ffn_timing) {
            gate_duration += (std::chrono::steady_clock::now() - gate_begin);
        }

        auto& pool = backend->GetThreadPool(numa_node);
        const int total = static_cast<int>(hidden_size);
        float* h_ptr = hidden_scratch.ptr;
        const float* g_ptr = gate_scratch.ptr;

        const auto activation_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (enable_inner_parallel && pool.GetNumThreads() > 1) {
            pool.ParallelFor(total, [=](int start, int end, int) {
                for (int i = start; i < end; i++) {
                    float x = h_ptr[i];
                    const float activated =
                        expert.use_gelu_activation ? GeluTanhApprox(x) : (x / (1.0f + internal::FastExp(-x)));
                    h_ptr[i] = activated * g_ptr[i];
                }
            });
        } else {
            for (int i = 0; i < total; ++i) {
                float x = h_ptr[i];
                const float activated =
                    expert.use_gelu_activation ? GeluTanhApprox(x) : (x / (1.0f + internal::FastExp(-x)));
                h_ptr[i] = activated * g_ptr[i];
            }
        }
        if (debug_ffn_timing) {
            activation_duration += (std::chrono::steady_clock::now() - activation_begin);
        }
        LogGemma4MoETensorStats("activated_gate_up", hidden, trace_ctx);
    }
    if (profile_enabled) {
        profile->w1w3_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w1w3_profile_begin)
                .count());
    }

    const auto w2_begin = debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto w2_profile_begin =
        profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    run_projection('2', hidden, w2, expert.w2_int4, expert.w2, expert.w2_type, hidden_dim, intermediate_dim, output);
    LogGemma4MoETensorStats("w2", *output, trace_ctx);
    MaybeLogGemma4PackedChecksum(expert, w1, w2, w3, hidden, *output, trace_ctx);
    if (profile_enabled) {
        profile->w2_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w2_profile_begin)
                .count());
    }
    if (debug_ffn_timing) {
        w2_duration += (std::chrono::steady_clock::now() - w2_begin);
        static std::atomic<int> log_budget{0};
        int current = log_budget.load(std::memory_order_relaxed);
        while (current < 256 && !log_budget.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {}
        if (current < 256) {
            const auto total_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_begin).count();
            const auto w1_ms = std::chrono::duration<double, std::milli>(w1_duration).count();
            const auto gate_ms = std::chrono::duration<double, std::milli>(gate_duration).count();
            const auto activation_ms = std::chrono::duration<double, std::milli>(activation_duration).count();
            const auto w2_ms = std::chrono::duration<double, std::milli>(w2_duration).count();
            std::fprintf(stderr,
                         "[MOE_FFN] batch=%lld hidden=%lld inter=%lld gelu=%d fused=%d allow_inner_parallel=%d "
                         "effective_inner_parallel=%d "
                         "w1_ms=%.3f gate_ms=%.3f act_ms=%.3f w2_ms=%.3f total_ms=%.3f\n",
                         static_cast<long long>(batch), static_cast<long long>(hidden_dim),
                         static_cast<long long>(intermediate_dim), expert.use_gelu_activation ? 1 : 0,
                         used_fused_gate_up ? 1 : 0, allow_inner_parallel ? 1 : 0,
                         enable_inner_parallel ? 1 : 0, w1_ms, gate_ms, activation_ms, w2_ms, total_ms);
        }
    }
}

bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node, bool allow_parallel,
                                CpuBackend::MoEProjectionPath* selected_path) {
    if (!backend || !output || !binding.IsValid() || !input.IsValid() || !output->IsValid() ||
        input.dtype != DType::F32 || output->dtype != DType::F32 || input.ndim != 2 || output->ndim != 2) {
        return false;
    }

    if (!CanUsePackedInt4MoEFastPath()) {
        return false;
    }

    if (TryRunPackedInt4ProjectionDirect(backend, binding, input, output, numa_node, allow_parallel)) {
        GetMoEInt4PathHistogram().direct_hwy.fetch_add(1, std::memory_order_relaxed);
        LogMoEMatmulPath("direct_hwy", static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                         static_cast<int>(output->shape[1]), binding.group_size, allow_parallel);
        if (selected_path) {
            *selected_path = CpuBackend::MoEProjectionPath::PackedInt4Fast;
        }
        return true;
    }

    const int64_t m = input.shape[0];
    const int64_t k = input.shape[1];
    const int64_t n = output->shape[1];
    if (m <= 0 || k != binding.K || n != binding.N) {
        return false;
    }

    const int64_t groups_per_row = binding.K / binding.group_size;
    if (groups_per_row <= 0) {
        return false;
    }

    Tensor W = Tensor::Make2D(const_cast<uint8_t*>(binding.packed_weights), binding.N, binding.K, DType::INT8);
    Tensor S = Tensor::Make2D(const_cast<float*>(binding.scales), binding.N, groups_per_row);
    Tensor Z = Tensor::Make2D(const_cast<float*>(binding.zeros), binding.N, groups_per_row);
    backend->GemmInt4(input, W, S, Z, output, binding.group_size, numa_node);
    GetMoEInt4PathHistogram().backend_gemm.fetch_add(1, std::memory_order_relaxed);
    LogMoEMatmulPath("backend_gemmint4", static_cast<int>(input.shape[0]), static_cast<int>(input.shape[1]),
                     static_cast<int>(output->shape[1]), binding.group_size, allow_parallel);
    if (selected_path) {
        *selected_path = CpuBackend::MoEProjectionPath::RuntimeGemmInt4;
    }
    return true;
}

size_t GetExpertMatrixDequantBytes(int ggml_type_id, const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                   int64_t rows, int64_t cols, bool safe_reference_mode,
                                   const ggml_tensor* scale_tensor = nullptr) {
    if (rows <= 0 || cols <= 0) {
        return 0;
    }
    if (ggml_type_id == GGML_TYPE_F32) {
        return scale_tensor ? static_cast<size_t>(rows * cols * sizeof(float)) : 0;
    }
    if (int4_binding.IsValid()) {
        return safe_reference_mode ? static_cast<size_t>(rows * cols * sizeof(float)) : 0;
    }
    return static_cast<size_t>(rows * cols * sizeof(float));
}

}  // namespace

void CpuBackend::ApplyMultiLoRA(
    const Tensor& input, const std::string& layer_name,
    const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map, Tensor* output) {
    if (!output || !input.IsValid() || !output->IsValid()) {
        return;
    }
    if (adapter_token_map.empty()) {
        return;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32) {
        return;
    }
    if (input.ndim != 2 || output->ndim != 2) {
        return;
    }

    const int64_t total_tokens = input.shape[0];
    const int64_t input_dim = input.shape[1];
    const int64_t output_dim = output->shape[1];
    if (input_dim <= 0 || output_dim <= 0) {
        return;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();

    struct AdapterTask {
        const LoRAAdapter* adapter = nullptr;
        const std::vector<int>* indices = nullptr;
        int numa_node = -1;
    };

    std::unordered_map<int, std::vector<AdapterTask>> tasks_by_node;
    tasks_by_node.reserve(adapter_token_map.size());

    for (const auto& entry : adapter_token_map) {
        const LoRAAdapter* adapter = entry.first.get();
        const std::vector<int>& indices = entry.second;
        if (!adapter || indices.empty()) {
            continue;
        }

        int numa_node = -1;
        if (!adapter->weights.empty()) {
            const auto& weight = adapter->weights.begin()->second;
            if (weight.lora_a && weight.lora_a->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_a->data);
            }
            if (numa_node < 0 && weight.lora_b && weight.lora_b->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_b->data);
            }
        }

        tasks_by_node[numa_node].push_back(AdapterTask{adapter, &indices, numa_node});
    }

    for (auto& group : tasks_by_node) {
        auto& tasks = group.second;
        if (tasks.empty()) {
            continue;
        }

        auto& pool = GetThreadPool(group.first);
        const int task_count = static_cast<int>(tasks.size());

        pool.ParallelFor(task_count, [&](int start, int end, int) {
            static thread_local AlignedScratch input_scratch;
            static thread_local AlignedScratch down_scratch;
            static thread_local AlignedScratch out_scratch;
            static thread_local std::vector<float> lora_a_f32;
            static thread_local std::vector<float> lora_b_f32;

            for (int t = start; t < end; ++t) {
                const AdapterTask& task = tasks[t];
                const LoRAAdapter* adapter = task.adapter;
                const std::vector<int>& indices = *task.indices;
                if (!adapter || indices.empty()) {
                    continue;
                }

                auto it_w = adapter->weights.find(layer_name);
                if (it_w == adapter->weights.end()) {
                    continue;
                }
                const densecore::LoRALayerWeight* layer_weight = &it_w->second;
                if (!layer_weight || !layer_weight->lora_a || !layer_weight->lora_b) {
                    continue;
                }

                const ggml_tensor* lora_a = layer_weight->lora_a;
                const ggml_tensor* lora_b = layer_weight->lora_b;

                const int64_t lora_a_in = lora_a->ne[0];
                const int64_t lora_a_rank = lora_a->ne[1];
                const int64_t lora_b_rank = lora_b->ne[0];
                const int64_t lora_b_out = lora_b->ne[1];
                const int64_t rank = std::min<int64_t>({layer_weight->rank, lora_a_rank, lora_b_rank});

                if (lora_a_in != input_dim || lora_b_out != output_dim || rank <= 0) {
                    continue;
                }

                const float* lora_a_ptr = internal::GetLoRAWeightF32(lora_a, lora_a_f32);
                const float* lora_b_ptr = internal::GetLoRAWeightF32(lora_b, lora_b_f32);
                if (!lora_a_ptr || !lora_b_ptr) {
                    continue;
                }

                const int64_t token_count = static_cast<int64_t>(indices.size());
                if (token_count <= 0) {
                    continue;
                }

                input_scratch.Resize(this, static_cast<size_t>(token_count * input_dim));
                down_scratch.Resize(this, static_cast<size_t>(token_count * rank));
                out_scratch.Resize(this, static_cast<size_t>(token_count * output_dim));

                float* input_subset = input_scratch.ptr;
                float* down = down_scratch.ptr;
                float* out = out_scratch.ptr;

                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    std::memcpy(input_subset + i * input_dim, input_data + static_cast<int64_t>(idx) * input_dim,
                                static_cast<size_t>(input_dim) * sizeof(float));
                }

                native::GemmF32(down, input_subset, lora_a_ptr, static_cast<int>(token_count), static_cast<int>(rank),
                                static_cast<int>(input_dim));
                native::GemmF32(out, down, lora_b_ptr, static_cast<int>(token_count), static_cast<int>(output_dim),
                                static_cast<int>(rank));

                const float scale = adapter->scale;
                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    float* dst = output_data + static_cast<int64_t>(idx) * output_dim;
                    const float* src = out + i * output_dim;
                    if (scale == 1.0f) {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j];
                        }
                    } else {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j] * scale;
                        }
                    }
                }
            }
        });
    }
}

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const ExpertWeights& expert, const Tensor& w1,
                                   const Tensor& w2, const Tensor& w3, Tensor* output) {
    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const ExpertWeights& expert, const Tensor& w1, const Tensor& w2, const Tensor& w3,
                                   Tensor* output) {
    int numa_node = -1;
    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            numa_node = profiler->GetExpertNumaNode(expert_id);
        }
    }
    DispatchExpertFFNImpl(this, numa_node, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2,
                                   const Tensor& w3, Tensor* output) {
    ExpertWeights expert{};
    DispatchExpertFFN(nullptr, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const Tensor& w1, const Tensor& w2, const Tensor& w3, Tensor* output) {
    ExpertWeights expert{};
    DispatchExpertFFN(layer_key, expert_id, input, expert, w1, w2, w3, output);
}

void CpuBackend::ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, nullptr, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, layer_key, -1, nullptr, input, routing, experts.data(), static_cast<int>(experts.size()),
               output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, layer_key, layer_idx, batch, input, routing, experts.data(), static_cast<int>(experts.size()),
               output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output) {
    ForwardMoE(nullptr, layer_key, -1, nullptr, input, routing, experts, num_experts, output);
}

void CpuBackend::ForwardMoE(const TransformerModel* model, const TransformerLayer* layer_key, int layer_idx,
                            const BatchSpec* batch, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output, MoEForwardProfile* profile) {
    moe_forward_invocation_count_.fetch_add(1, std::memory_order_relaxed);
    const int batch_size = routing.batch_size;
    const int top_k = routing.top_k;
    const size_t hidden_dim = input.shape[1];
    const size_t assignment_count = routing.expert_ids.size();

    if (!experts || batch_size <= 0 || top_k <= 0 || num_experts <= 0 || assignment_count == 0) {
        return;
    }
    if (routing.weights.size() != assignment_count) {
        return;
    }
    if (!routing.token_indices.empty() && routing.token_indices.size() != assignment_count) {
        return;
    }
    if (routing.token_indices.empty() && assignment_count != static_cast<size_t>(batch_size * top_k)) {
        return;
    }

    auto registry = GetMoELayerRegistry(layer_key);
    RecordExpertAccess(layer_key, routing.expert_ids.data(), static_cast<int>(routing.expert_ids.size()));
    const bool arm_disable_registry_dequant_cache = false;

    float* out_data = output->DataAs<float>();
    std::memset(out_data, 0, batch_size * hidden_dim * sizeof(float));

    static thread_local AlignedScratch routing_scratch;
    static thread_local AlignedScratch expert_input_scratch;
    static thread_local AlignedScratch expert_output_scratch;
    static thread_local AlignedScratch w1_dequant;
    static thread_local AlignedScratch w2_dequant;
    static thread_local AlignedScratch w3_dequant;
    static thread_local AlignedScratch small_decode_output_scratch;
    const auto refresh_cached_tensors = [](MoELayerRegistry::DequantizedExpertCacheEntry* entry,
                                           const ExpertWeights& exp) {
        if (!entry) {
            return;
        }
        entry->w1_tensor = entry->w1.empty()
                               ? Tensor()
                               : Tensor::Make2D(entry->w1.data(), static_cast<int64_t>(exp.intermediate_dim),
                                                static_cast<int64_t>(exp.hidden_dim));
        entry->w2_tensor = entry->w2.empty() ? Tensor()
                                             : Tensor::Make2D(entry->w2.data(), static_cast<int64_t>(exp.hidden_dim),
                                                              static_cast<int64_t>(exp.intermediate_dim));
        entry->w3_tensor = entry->w3.empty()
                               ? Tensor()
                               : Tensor::Make2D(entry->w3.data(), static_cast<int64_t>(exp.intermediate_dim),
                                                static_cast<int64_t>(exp.hidden_dim));
    };

    const int total_assignments = static_cast<int>(assignment_count);
    if (total_assignments == 0 || num_experts == 0 || top_k <= 0) {
        return;
    }

    // Qwen3.5-35B-A3B can fan out into many one-token expert assignments on ARM.
    // The tiny decode-specialized path processes those assignments serially,
    // which underutilizes CPU and can strand the request in prefill. Force the
    // batched MoE path for very large expert pools on ARM so experts are grouped
    // and routed through the main reordered execution lane.
#if defined(__aarch64__) || defined(_M_ARM64)
    const bool arm_large_expert_pool = num_experts >= 128 && batch_size > 1;
#else
    const bool arm_large_expert_pool = false;
#endif
    const bool small_decode_candidate =
        batch_size <= 4 && total_assignments <= kSmallDecodeMaxAssignments && !arm_large_expert_pool;
    const bool has_token_indices = !routing.token_indices.empty();
    const float* input_data = input.DataAs<float>();
    const bool qwen36_short_prefill_safe_reference =
        IsQwen36ShortSingleSeqPrefillSafeReferenceCandidate(model, batch, batch_size);
    const bool safe_reference_mode =
        qwen36_short_prefill_safe_reference ||
        (num_experts > 0 ? IsMoESafeReferenceModeEnabled(&experts[0]) : IsMoESafeReferenceModeEnabled());
    const bool gemma4_safe_reference_mode = model && model->arch_flags.is_gemma4 && safe_reference_mode;
    if (gemma4_safe_reference_mode) {
        if (!ExecuteMoEReferencePath(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts,
                                     num_experts, out_data)) {
            std::fprintf(stderr, "[MoE_REF_EXEC] Gemma4 safe-reference execution failed; output left zeroed\n");
            return;
        }
        RecordMoEReferencePathTrace(this, layer_idx, batch, routing, num_experts);
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                                 out_data);
        }
        return;
    }
    if (qwen36_short_prefill_safe_reference &&
        ExecuteMoEReferencePath(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                                out_data)) {
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                                 out_data);
        }
        return;
    }
    bool small_decode_requires_general_path = false;
    if (small_decode_candidate) {
        for (size_t i = 0; i < assignment_count; ++i) {
            const int expert_id = routing.expert_ids[i];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            const auto& exp = experts[static_cast<size_t>(expert_id)];
            // Accept packed INT4 OR ggml quantized weights (Q4_K etc.)
            if (!ExpertUsesPackedInt4Only(exp) && !(ExpertHasGgmlQuantizedWeights(exp) && !safe_reference_mode)) {
                small_decode_requires_general_path = true;
                break;
            }
        }
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    std::array<int, kSmallDecodeMaxSnapshotExperts> small_step_local_hot_experts{};
    std::array<int, kSmallDecodeMaxSnapshotExperts> small_step_previous_batch_experts{};
    int small_step_local_hot_count = 0;
    int small_step_previous_batch_count = 0;
    bool small_step_snapshot_ok = true;
    std::vector<int> local_hot_experts;
    std::vector<int> previous_batch_experts;
    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
        if (small_decode_candidate) {
            small_step_snapshot_ok =
                CopyIntVectorToFixedArray(registry->local_expert_ids, &small_step_local_hot_experts,
                                          &small_step_local_hot_count) &&
                CopyIntVectorToFixedArray(registry->last_batch_experts, &small_step_previous_batch_experts,
                                          &small_step_previous_batch_count);
            if (!small_step_snapshot_ok) {
                local_hot_experts = registry->local_expert_ids;
                previous_batch_experts = registry->last_batch_experts;
            }
        } else {
            local_hot_experts = registry->local_expert_ids;
            previous_batch_experts = registry->last_batch_experts;
        }
    }

    std::array<int, kSmallDecodeMaxAssignments> small_step_current_batch_experts{};
    int small_step_current_batch_expert_count = 0;
    int small_step_max_expert_batch = 0;
    if (small_decode_candidate && small_step_snapshot_ok) {
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            bool seen = false;
            int expert_batch_count = 0;
            for (int j = 0; j < total_assignments; ++j) {
                if (routing.expert_ids[static_cast<size_t>(j)] == expert_id) {
                    ++expert_batch_count;
                }
            }
            small_step_max_expert_batch = std::max(small_step_max_expert_batch, expert_batch_count);
            for (int j = 0; j < small_step_current_batch_expert_count; ++j) {
                if (small_step_current_batch_experts[static_cast<size_t>(j)] == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen && small_step_current_batch_expert_count < kSmallDecodeMaxAssignments) {
                small_step_current_batch_experts[static_cast<size_t>(small_step_current_batch_expert_count++)] =
                    expert_id;
            }
        }
    }

    const bool small_decode_ready = small_decode_candidate && !small_decode_requires_general_path &&
                                    small_step_snapshot_ok && small_step_max_expert_batch <= 1 &&
                                    small_step_current_batch_expert_count > 0;
    if (!small_decode_ready) {
        const char* reject_reason = !small_decode_candidate        ? "not_small_decode"
                                    : small_decode_requires_general_path ? "requires_general_path"
                                    : !small_step_snapshot_ok          ? "snapshot_unavailable"
                                    : small_step_max_expert_batch > 1  ? "expert_batch_gt_1"
                                                                        : "no_active_experts";
        RecordMoESmallDecodeParallelDecision(GetCurrentWorkContext(), small_decode_candidate, false, reject_reason,
                                             small_step_current_batch_expert_count, top_k, total_assignments);
    }
    if (small_decode_ready) {
        int reuse_intersection = 0;
        if (small_step_previous_batch_count > 0) {
            for (int i = 0; i < small_step_current_batch_expert_count; ++i) {
                if (FixedArrayContains(small_step_previous_batch_experts, small_step_previous_batch_count,
                                       small_step_current_batch_experts[static_cast<size_t>(i)])) {
                    ++reuse_intersection;
                }
            }
        }
        const int reuse_union =
            small_step_current_batch_expert_count + small_step_previous_batch_count - reuse_intersection;
        int local_hot_count = 0;
        for (int i = 0; i < small_step_current_batch_expert_count; ++i) {
            if (FixedArrayContains(small_step_local_hot_experts, small_step_local_hot_count,
                                   small_step_current_batch_experts[static_cast<size_t>(i)])) {
                ++local_hot_count;
            }
        }

        moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
        moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(small_step_current_batch_expert_count),
                                                  std::memory_order_relaxed);
        moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(total_assignments), std::memory_order_relaxed);
        moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
        moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection),
                                                      std::memory_order_relaxed);
        moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)),
                                               std::memory_order_relaxed);
        moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(small_step_max_expert_batch),
                                                    std::memory_order_relaxed);

        if (registry) {
            std::lock_guard<std::mutex> lock(registry->mutex);
            registry->last_batch_experts.assign(small_step_current_batch_experts.begin(),
                                                small_step_current_batch_experts.begin() +
                                                    static_cast<ptrdiff_t>(small_step_current_batch_expert_count));
        }

        const bool dequant_cache_enabled =
            internal::IsMoEDequantCacheEnabled() && registry != nullptr && !arm_disable_registry_dequant_cache;
        const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
        const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();
        auto make_weight_f32_small = [&](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                                         const ggml_tensor* scale_tensor, int64_t rows, int64_t cols,
                                         AlignedScratch& scratch, size_t* dequantized_bytes,
                                         bool* dequantized_any) -> Tensor {
            if (int4_binding.IsValid()) {
                if (!safe_reference_mode) {
                    return Tensor();
                }
                scratch.Resize(this, static_cast<size_t>(rows * cols));
                if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, scratch.ptr)) {
                    return Tensor();
                }
                if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scratch.ptr, nullptr)) {
                    return Tensor();
                }
                if (dequantized_bytes) {
                    *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
                }
                if (dequantized_any) {
                    *dequantized_any = true;
                }
                return Tensor::Make2D(scratch.ptr, rows, cols);
            }
            if (!ptr || rows <= 0 || cols <= 0) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
            if (wtype == GGML_TYPE_F32 && !scale_tensor) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            if (wtype == GGML_TYPE_F32) {
                std::memcpy(scratch.ptr, ptr, static_cast<size_t>(rows * cols) * sizeof(float));
            } else {
                const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
                if (!traits || !traits->to_float) {
                    return Tensor::Make2D(ptr, rows, cols);
                }
                const size_t row_bytes = ggml_row_size(wtype, cols);
                const char* src = static_cast<const char*>(ptr);
                for (int64_t r = 0; r < rows; ++r) {
                    traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
                }
            }
            if (scale_tensor) {
                if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scratch.ptr, nullptr)) {
                    return Tensor();
                }
            }
            if (dequantized_bytes) {
                *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
            }
            if (dequantized_any) {
                *dequantized_any = true;
            }
            return Tensor::Make2D(scratch.ptr, rows, cols);
        };

        auto& small_decode_pool = GetThreadPool(-1);
        const int small_decode_worker_cap = std::max(1, small_decode_pool.GetNumThreads());
        const SmallDecodeExpertParallelDecision expert_parallel_decision = ResolveSmallDecodeExpertParallelDecision(
            model, batch_size, top_k, safe_reference_mode, small_decode_worker_cap);
        const int effective_small_decode_workers = std::min(total_assignments, expert_parallel_decision.workers);
        const bool use_small_decode_expert_parallel =
            expert_parallel_decision.enabled && effective_small_decode_workers > 1;
        RecordMoESmallDecodeParallelDecision(
            GetCurrentWorkContext(), true, use_small_decode_expert_parallel,
            use_small_decode_expert_parallel ? nullptr : expert_parallel_decision.reason,
            small_step_current_batch_expert_count, top_k, effective_small_decode_workers);
        int64_t small_decode_intermediate_dim = 0;
        for (int i = 0; i < total_assignments && small_decode_intermediate_dim <= 0; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id >= 0 && expert_id < num_experts) {
                small_decode_intermediate_dim =
                    static_cast<int64_t>(experts[static_cast<size_t>(expert_id)].intermediate_dim);
            }
        }
        const bool tile_par_can_use = CanUseSmallDecodeQuantizedTileParallel(model);
        const bool tile_par_ep_enabled = expert_parallel_decision.enabled;
        const bool use_small_decode_quantized_tile_parallel =
            tile_par_can_use && tile_par_ep_enabled && !safe_reference_mode &&
            batch_size == 1 && small_decode_worker_cap >= 16 && total_assignments >= 2 && hidden_dim >= 1024 &&
            small_decode_intermediate_dim >= 256;
        {
            static std::atomic<int> tile_par_diag_count{0};
            if (tile_par_diag_count.fetch_add(1, std::memory_order_relaxed) < 3) {
                std::fprintf(stderr,
                    "[MOE_TILE_DIAG] tile_parallel=%d can_use=%d ep_enabled=%d ep_reason=%s safe_ref=%d "
                    "batch=%d worker_cap=%d assignments=%d hidden=%lld intermediate=%lld\n",
                    use_small_decode_quantized_tile_parallel ? 1 : 0,
                    tile_par_can_use ? 1 : 0, tile_par_ep_enabled ? 1 : 0,
                    expert_parallel_decision.reason ? expert_parallel_decision.reason : "null",
                    safe_reference_mode ? 1 : 0, batch_size, small_decode_worker_cap,
                    total_assignments, static_cast<long long>(hidden_dim),
                    static_cast<long long>(small_decode_intermediate_dim));
            }
        }
        if (use_small_decode_quantized_tile_parallel) {
            bool all_assignments_supported = true;
            for (int i = 0; i < total_assignments; ++i) {
                const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                if (expert_id < 0 || expert_id >= num_experts) {
                    continue;
                }
                const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
                const bool down_scale_supported =
                    exp.w2_scale_tensor == nullptr || IsScalarScaleSidecar(exp.w2_scale_tensor);
                const auto gate_type = static_cast<ggml_type>(exp.w1_type);
                const auto up_type = static_cast<ggml_type>(exp.w3_type);
                const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
                const bool gelu_gate_up_supported =
                    exp.use_gelu_activation && gate_type == up_type && gate_traits_cpu && gate_traits_cpu->vec_dot &&
                    IsKQuantRowPairGatedProjectionType(gate_type, gate_traits_cpu->vec_dot_type) &&
                    (hidden_dim % ggml_blck_size(gate_type)) == 0;
                const bool has_quantized_gated_ffn =
                    exp.w1.ptr && exp.w3.ptr && exp.w2.ptr && exp.w1_type != GGML_TYPE_F32 &&
                    exp.w2_type != GGML_TYPE_F32 && exp.w3_type != GGML_TYPE_F32 &&
                    ggml_is_quantized(static_cast<ggml_type>(exp.w1_type)) &&
                    ggml_is_quantized(static_cast<ggml_type>(exp.w2_type)) &&
                    (!exp.use_gelu_activation || gelu_gate_up_supported) &&
                    ggml_is_quantized(static_cast<ggml_type>(exp.w3_type)) && down_scale_supported;
                if (!has_quantized_gated_ffn) {
                    all_assignments_supported = false;
                    static std::atomic<int> qgf_diag{0};
                    if (qgf_diag.fetch_add(1, std::memory_order_relaxed) < 2) {
                        std::fprintf(stderr,
                            "[MOE_TILE_DIAG] quantized_gated_ffn FAIL expert=%d w1=%d w3=%d w2=%d "
                            "gelu=%d down_scale=%d w1_ptr=%d w3_ptr=%d w2_ptr=%d\n",
                            expert_id, exp.w1_type, exp.w3_type, exp.w2_type,
                            exp.use_gelu_activation ? 1 : 0, down_scale_supported ? 1 : 0,
                            exp.w1.ptr ? 1 : 0, exp.w3.ptr ? 1 : 0, exp.w2.ptr ? 1 : 0);
                    }
                    break;
                }
            }

            if (all_assignments_supported) {
                // Scale splits to utilize available cores. With 2 assignments and
                // 16 cores, target_parallelism = 8, giving up to 8 gate splits
                // and 8 down splits per expert — each core gets a tile of the GEMV.
                const int target_parallelism = std::max(1, small_decode_worker_cap / std::max(1, total_assignments));
                const int max_gate_splits = std::min<int>(
                    target_parallelism, std::max<int>(1, static_cast<int>(small_decode_intermediate_dim / 64)));
                const int gate_splits = std::max<int>(1, std::min<int>(max_gate_splits, 8));
                const int down_splits = std::min<int>(
                    std::max<int>(1, small_decode_worker_cap / std::max(1, total_assignments)),
                    std::max<int>(1, static_cast<int>(hidden_dim / 256)));
                {
                    static std::atomic<int> tile_enter_diag{0};
                    if (tile_enter_diag.fetch_add(1, std::memory_order_relaxed) < 2) {
                        std::fprintf(stderr,
                            "[MOE_TILE_DIAG] ENTERING tile_parallel gate_splits=%d down_splits=%d "
                            "target_par=%d workers=%d assignments=%d\n",
                            gate_splits, down_splits, target_parallelism,
                            small_decode_worker_cap, total_assignments);
                    }
                }
                const size_t assignment_hidden_elems =
                    static_cast<size_t>(total_assignments) * static_cast<size_t>(small_decode_intermediate_dim);
                const size_t assignment_output_elems = static_cast<size_t>(total_assignments) * hidden_dim;
                const bool reused_assignment_scratch =
                    expert_input_scratch.HasCapacity(assignment_hidden_elems) &&
                    expert_output_scratch.HasCapacity(assignment_output_elems);
                expert_input_scratch.Resize(this, assignment_hidden_elems);
                expert_output_scratch.Resize(this, assignment_output_elems);
                float* assignment_hiddens = expert_input_scratch.ptr;
                float* assignment_outputs = expert_output_scratch.ptr;
                std::fill_n(assignment_hiddens, assignment_hidden_elems, 0.0f);
                std::fill_n(assignment_outputs, assignment_output_elems, 0.0f);
                if (profile && reused_assignment_scratch) {
                    profile->decode_scratch_reused += 2;
                    profile->decode_allocations_avoided += 2;
                }
                std::vector<QuantizedProjectionInputCache> down_input_projection_caches(
                    static_cast<size_t>(total_assignments));
                QuantizedProjectionInputCache shared_decode_input_projection_cache;
                QuantizedProjectionInputCache* shared_decode_input_projection_cache_ptr = nullptr;
                if (batch_size == 1) {
                    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
                    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, hidden_dim);
                    if (q8_traits && q8_traits->from_float && q8_row_bytes > 0) {
                        shared_decode_input_projection_cache.source = input_data;
                        shared_decode_input_projection_cache.rows = 1;
                        shared_decode_input_projection_cache.cols = hidden_dim;
                        shared_decode_input_projection_cache.type = GGML_TYPE_Q8_K;
                        shared_decode_input_projection_cache.row_bytes = q8_row_bytes;
                        shared_decode_input_projection_cache.bytes.resize(q8_row_bytes);
                        q8_traits->from_float(input_data, shared_decode_input_projection_cache.bytes.data(),
                                              hidden_dim);
                        shared_decode_input_projection_cache_ptr = &shared_decode_input_projection_cache;
                    }
                }
                std::atomic<bool> tile_ok{true};

                LogSmallDecodeExecutionPath("quantized_tile_parallel", total_assignments, small_decode_worker_cap,
                                            batch_size);
                small_decode_pool.ParallelFor(total_assignments * gate_splits, [&](int unit_start, int unit_end,
                                                                                   int /*thread_id*/) {
                    for (int unit = unit_start; unit < unit_end && tile_ok.load(std::memory_order_relaxed); ++unit) {
                        const int i = unit / gate_splits;
                        const int split = unit - i * gate_splits;
                        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                        if (expert_id < 0 || expert_id >= num_experts) {
                            continue;
                        }
                        const int token_idx =
                            has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
                        if (token_idx < 0 || token_idx >= batch_size) {
                            continue;
                        }

                        const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
                        const int64_t tile_start = (small_decode_intermediate_dim * split) / gate_splits;
                        const int64_t tile_end = (small_decode_intermediate_dim * (split + 1)) / gate_splits;
                        const int64_t tile_rows = tile_end - tile_start;
                        if (tile_rows <= 0) {
                            continue;
                        }
                        const size_t gate_row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w1_type), hidden_dim);
                        const size_t up_row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w3_type), hidden_dim);
                        const char* gate_ptr =
                            static_cast<const char*>(exp.w1.ptr) + static_cast<ptrdiff_t>(tile_start * gate_row_bytes);
                        const char* up_ptr =
                            static_cast<const char*>(exp.w3.ptr) + static_cast<ptrdiff_t>(tile_start * up_row_bytes);
                        Tensor expert_input =
                            Tensor::Make2D(const_cast<float*>(input_data + static_cast<size_t>(token_idx) * hidden_dim),
                                           1, static_cast<int64_t>(hidden_dim));
                        float* hidden_tile =
                            assignment_hiddens + static_cast<size_t>(i) *
                                                    static_cast<size_t>(small_decode_intermediate_dim) +
                            static_cast<size_t>(tile_start);
                        Tensor hidden_tensor = Tensor::Make2D(hidden_tile, 1, tile_rows);
                        QuantizedProjectionInputCache local_input_projection_cache;
                        QuantizedProjectionInputCache* unit_input_projection_cache =
                            (shared_decode_input_projection_cache_ptr &&
                             CanUseSharedDecodeInputCacheForExpert(exp, shared_decode_input_projection_cache_ptr->type))
                                ? shared_decode_input_projection_cache_ptr
                                : &local_input_projection_cache;
                        const bool gate_up_ok =
                            exp.use_gelu_activation
                                ? TryRunGgmlQuantizedFusedGEGLUProjection(
                                      this, gate_ptr, exp.w1_type, up_ptr, exp.w3_type, expert_input, &hidden_tensor,
                                      tile_rows, hidden_dim, profiler ? profiler->GetExpertNumaNode(expert_id) : -1,
                                      /*allow_parallel=*/false, unit_input_projection_cache)
                                : TryRunGgmlQuantizedFusedSwiGLUProjection(
                                      this, gate_ptr, exp.w1_type, up_ptr, exp.w3_type, expert_input, &hidden_tensor,
                                      tile_rows, hidden_dim, profiler ? profiler->GetExpertNumaNode(expert_id) : -1,
                                      /*allow_parallel=*/false, unit_input_projection_cache);
                        if (!gate_up_ok) {
                            tile_ok.store(false, std::memory_order_relaxed);
                        }
                    }
                });

                if (tile_ok.load(std::memory_order_relaxed)) {
                    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
                    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, small_decode_intermediate_dim);
                    if (!q8_traits || !q8_traits->from_float || q8_row_bytes == 0) {
                        tile_ok.store(false, std::memory_order_relaxed);
                    } else {
                        small_decode_pool.ParallelFor(total_assignments, [&](int start, int end, int /*thread_id*/) {
                            for (int i = start; i < end; ++i) {
                                const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                                if (expert_id < 0 || expert_id >= num_experts) {
                                    continue;
                                }
                                auto& cache = down_input_projection_caches[static_cast<size_t>(i)];
                                const float* hidden_row =
                                    assignment_hiddens +
                                    static_cast<size_t>(i) * static_cast<size_t>(small_decode_intermediate_dim);
                                cache.source = hidden_row;
                                cache.rows = 1;
                                cache.cols = small_decode_intermediate_dim;
                                cache.type = GGML_TYPE_Q8_K;
                                cache.row_bytes = q8_row_bytes;
                                cache.bytes.resize(q8_row_bytes);
                                q8_traits->from_float(hidden_row, cache.bytes.data(), small_decode_intermediate_dim);
                            }
                        });
                    }
                }

                if (tile_ok.load(std::memory_order_relaxed)) {
                    small_decode_pool.ParallelFor(total_assignments * down_splits, [&](int unit_start, int unit_end,
                                                                                       int /*thread_id*/) {
                        for (int unit = unit_start; unit < unit_end && tile_ok.load(std::memory_order_relaxed);
                             ++unit) {
                            const int i = unit / down_splits;
                            const int split = unit - i * down_splits;
                            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                            if (expert_id < 0 || expert_id >= num_experts) {
                                continue;
                            }
                            const int token_idx =
                                has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
                            if (token_idx < 0 || token_idx >= batch_size) {
                                continue;
                            }

                            const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
                            const int64_t tile_start = (hidden_dim * split) / down_splits;
                            const int64_t tile_end = (hidden_dim * (split + 1)) / down_splits;
                            const int64_t tile_rows = tile_end - tile_start;
                            if (tile_rows <= 0) {
                                continue;
                            }
                            const size_t down_row_bytes =
                                ggml_row_size(static_cast<ggml_type>(exp.w2_type), small_decode_intermediate_dim);
                            const char* down_ptr = static_cast<const char*>(exp.w2.ptr) +
                                                   static_cast<ptrdiff_t>(tile_start * down_row_bytes);
                            Tensor hidden_tensor = Tensor::Make2D(
                                assignment_hiddens +
                                    static_cast<size_t>(i) * static_cast<size_t>(small_decode_intermediate_dim),
                                1, small_decode_intermediate_dim);
                            float* output_tile = assignment_outputs + static_cast<size_t>(i) * hidden_dim +
                                                 static_cast<size_t>(tile_start);
                            Tensor output_tensor = Tensor::Make2D(output_tile, 1, tile_rows);
                            QuantizedProjectionInputCache local_down_input_projection_cache;
                            QuantizedProjectionInputCache* down_input_projection_cache =
                                QuantizedProjectionInputTypeMatches(exp.w2_type,
                                                                    down_input_projection_caches[static_cast<size_t>(i)].type)
                                    ? &down_input_projection_caches[static_cast<size_t>(i)]
                                    : &local_down_input_projection_cache;
                            if (!TryRunGgmlQuantizedProjection(this, down_ptr, exp.w2_type, hidden_tensor,
                                                               &output_tensor, tile_rows, small_decode_intermediate_dim,
                                                               profiler ? profiler->GetExpertNumaNode(expert_id) : -1,
                                                               /*allow_parallel=*/false,
                                                               down_input_projection_cache)) {
                                tile_ok.store(false, std::memory_order_relaxed);
                            } else if (exp.w2_scale_tensor != nullptr) {
                                ApplyScalarScaleToTensor(&output_tensor, ReadScalarScaleSidecar(exp.w2_scale_tensor));
                            }
                        }
                    });
                }

                if (tile_ok.load(std::memory_order_relaxed)) {
                    for (int i = 0; i < total_assignments; ++i) {
                        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                        if (expert_id < 0 || expert_id >= num_experts) {
                            continue;
                        }
                        const int token_idx =
                            has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
                        if (token_idx < 0 || token_idx >= batch_size) {
                            continue;
                        }
                        const float weight = routing.weights[static_cast<size_t>(i)];
                        if (weight == 0.0f) {
                            continue;
                        }
                        float* dst = out_data + static_cast<size_t>(token_idx) * hidden_dim;
                        const float* src = assignment_outputs + static_cast<size_t>(i) * hidden_dim;
                        for (size_t d = 0; d < hidden_dim; ++d) {
                            dst[d] += weight * src[d];
                        }
                    }
                    if (ShouldRunMoEReferenceCheck()) {
                        RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts,
                                             num_experts, out_data);
                    }
                    return;
                }
            }
        }
        if (use_small_decode_expert_parallel) {
            {
                static std::atomic<int> ep_diag{0};
                if (ep_diag.fetch_add(1, std::memory_order_relaxed) < 2) {
                    std::fprintf(stderr,
                        "[MOE_TILE_DIAG] FALLBACK to expert_parallel (NOT tile_parallel) "
                        "workers=%d assignments=%d\n",
                        effective_small_decode_workers, total_assignments);
                }
            }
            LogSmallDecodeExecutionPath("expert_parallel", total_assignments, effective_small_decode_workers,
                                        batch_size);
            const size_t assignment_output_elems = static_cast<size_t>(total_assignments) * hidden_dim;
            const bool reused_output_scratch = small_decode_output_scratch.HasCapacity(assignment_output_elems);
            small_decode_output_scratch.Resize(this, assignment_output_elems);
            if (profile && reused_output_scratch) {
                profile->decode_scratch_reused += 1;
                profile->decode_allocations_avoided += 1;
            }
            float* assignment_outputs = small_decode_output_scratch.ptr;
            QuantizedProjectionInputCache shared_decode_input_projection_cache;
            QuantizedProjectionInputCache* shared_decode_input_projection_cache_ptr = nullptr;
            if (batch_size == 1) {
                const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
                const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, hidden_dim);
                if (q8_traits && q8_traits->from_float && q8_row_bytes > 0) {
                    shared_decode_input_projection_cache.source = input_data;
                    shared_decode_input_projection_cache.rows = 1;
                    shared_decode_input_projection_cache.cols = hidden_dim;
                    shared_decode_input_projection_cache.type = GGML_TYPE_Q8_K;
                    shared_decode_input_projection_cache.row_bytes = q8_row_bytes;
                    shared_decode_input_projection_cache.bytes.resize(q8_row_bytes);
                    q8_traits->from_float(input_data, shared_decode_input_projection_cache.bytes.data(), hidden_dim);
                    shared_decode_input_projection_cache_ptr = &shared_decode_input_projection_cache;
                }
            }
            small_decode_pool.ParallelFor(effective_small_decode_workers, [&](int worker_start, int worker_end,
                                                                              int /*thread_id*/) {
                for (int worker = worker_start; worker < worker_end; ++worker) {
                    const int assignment_start = (worker * total_assignments) / effective_small_decode_workers;
                    const int assignment_end = ((worker + 1) * total_assignments) / effective_small_decode_workers;
                    for (int i = assignment_start; i < assignment_end; ++i) {
                        const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                        if (expert_id < 0 || expert_id >= num_experts) {
                            continue;
                        }
                        const int token_idx =
                            has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
                        if (token_idx < 0 || token_idx >= batch_size) {
                            continue;
                        }

                        const float weight = routing.weights[static_cast<size_t>(i)];
                        if (weight == 0.0f) {
                            continue;
                        }

                        const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
                        Tensor expert_input =
                            Tensor::Make2D(const_cast<float*>(input_data + static_cast<size_t>(token_idx) * hidden_dim),
                                           1, static_cast<int64_t>(hidden_dim));
                        float* assignment_output = assignment_outputs + static_cast<size_t>(i) * hidden_dim;
                        Tensor expert_out = Tensor::Make2D(assignment_output, 1, static_cast<int64_t>(hidden_dim));
                        QuantizedProjectionInputCache local_decode_input_projection_cache;
                        QuantizedProjectionInputCache* decode_input_projection_cache =
                            (shared_decode_input_projection_cache_ptr &&
                             CanUseSharedDecodeInputCacheForExpert(exp, shared_decode_input_projection_cache_ptr->type))
                                ? shared_decode_input_projection_cache_ptr
                                : &local_decode_input_projection_cache;
                        const int expert_numa_node = profiler ? profiler->GetExpertNumaNode(expert_id) : -1;
                        MoEExecutionTraceContext trace_ctx;
                        trace_ctx.layer_idx = layer_idx;
                        trace_ctx.expert_id = expert_id;
                        trace_ctx.token_idx = token_idx;
                        if (batch && trace_ctx.token_idx >= 0 &&
                            trace_ctx.token_idx < static_cast<int>(batch->seq_id.size())) {
                            trace_ctx.seq_id = batch->seq_id[static_cast<size_t>(trace_ctx.token_idx)];
                            if (trace_ctx.seq_id >= 0 && trace_ctx.seq_id < static_cast<int>(batch->n_past.size())) {
                                trace_ctx.n_past = batch->n_past[static_cast<size_t>(trace_ctx.seq_id)];
                                std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
                                const auto last_it = moe_decode_last_n_past_.find(trace_ctx.seq_id);
                                if (last_it == moe_decode_last_n_past_.end() || last_it->second != trace_ctx.n_past) {
                                    moe_decode_last_n_past_[trace_ctx.seq_id] = trace_ctx.n_past;
                                    trace_ctx.decode_step = moe_decode_step_ordinals_[trace_ctx.seq_id]++;
                                } else {
                                    trace_ctx.decode_step =
                                        std::max(0, moe_decode_step_ordinals_[trace_ctx.seq_id] - 1);
                                }
                            }
                        }
                        DispatchExpertFFNImpl(this, expert_numa_node, expert_input, exp, Tensor(), Tensor(), Tensor(),
                                              &expert_out, /*allow_inner_parallel=*/false, &trace_ctx,
                                              decode_input_projection_cache);
                    }
                }
            });

            for (int i = 0; i < total_assignments; ++i) {
                const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                if (expert_id < 0 || expert_id >= num_experts) {
                    continue;
                }
                const int token_idx = has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
                if (token_idx < 0 || token_idx >= batch_size) {
                    continue;
                }
                const float weight = routing.weights[static_cast<size_t>(i)];
                if (weight == 0.0f) {
                    continue;
                }

                float* dst = out_data + static_cast<size_t>(token_idx) * hidden_dim;
                const float* src = assignment_outputs + static_cast<size_t>(i) * hidden_dim;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    dst[d] += weight * src[d];
                }
            }
            if (ShouldRunMoEReferenceCheck()) {
                RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts,
                                     num_experts, out_data);
            }
            return;
        }
        if (expert_parallel_decision.forced_on) {
            LogSmallDecodeFallback(expert_parallel_decision.reason, total_assignments, effective_small_decode_workers,
                                   batch_size);
        }

        const bool reused_single_output_scratch = small_decode_output_scratch.HasCapacity(hidden_dim);
        small_decode_output_scratch.Resize(this, hidden_dim);
        if (profile && reused_single_output_scratch) {
            profile->decode_scratch_reused += 1;
            profile->decode_allocations_avoided += 1;
        }
        QuantizedProjectionInputCache small_decode_input_projection_cache;
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            const int token_idx = has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
            if (token_idx < 0 || token_idx >= batch_size) {
                continue;
            }

            const float weight = routing.weights[static_cast<size_t>(i)];
            if (weight == 0.0f) {
                continue;
            }

            const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
            const bool local_hot =
                FixedArrayContains(small_step_local_hot_experts, small_step_local_hot_count, expert_id);
            const bool reused_last_batch =
                FixedArrayContains(small_step_previous_batch_experts, small_step_previous_batch_count, expert_id);
            const bool should_try_cache =
                dequant_cache_enabled && (cache_all_active_experts || local_hot || reused_last_batch);
            const size_t cacheable_bytes =
                GetExpertMatrixDequantBytes(exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                            static_cast<int64_t>(exp.hidden_dim), safe_reference_mode) +
                GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                            exp.w2_scale_tensor) +
                ((exp.w3.ptr != nullptr || exp.w3_int4.IsValid())
                     ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                   static_cast<int64_t>(exp.hidden_dim), safe_reference_mode)
                     : 0);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> existing_entry;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(expert_id);
                    if (it != registry->dequant_cache.end()) {
                        existing_entry = it->second;
                        if (existing_entry) {
                            existing_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    }
                }

                if (existing_entry) {
                    cached_entry = std::move(existing_entry);
                } else {
                    auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
                    candidate->expert_id = expert_id;
                    candidate->bytes = cacheable_bytes;
                    if ((!exp.w1_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                        exp.w1_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w1_type));
                        if (exp.w1_int4.IsValid() && safe_reference_mode) {
                            candidate->w1.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            DequantizePackedInt4ToF32(exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                      static_cast<int64_t>(exp.hidden_dim), candidate->w1.data());
                        } else if (traits && traits->to_float) {
                            candidate->w1.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w1_type), exp.hidden_dim);
                            const char* src = static_cast<const char*>(exp.w1.ptr);
                            for (int64_t r = 0; r < exp.intermediate_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w1.data() + r * exp.hidden_dim, exp.hidden_dim);
                            }
                        }
                    }
                    if (exp.w2_scale_tensor ||
                        ((!exp.w2_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                         exp.w2_type != GGML_TYPE_F32)) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w2_type));
                        candidate->w2.resize(static_cast<size_t>(exp.hidden_dim) * exp.intermediate_dim);
                        if (exp.w2_int4.IsValid() && safe_reference_mode) {
                            DequantizePackedInt4ToF32(exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                                      static_cast<int64_t>(exp.intermediate_dim), candidate->w2.data());
                        } else if (exp.w2_type == GGML_TYPE_F32) {
                            std::memcpy(candidate->w2.data(), exp.w2.ptr, candidate->w2.size() * sizeof(float));
                        } else if (traits && traits->to_float) {
                            const size_t row_bytes =
                                ggml_row_size(static_cast<ggml_type>(exp.w2_type), exp.intermediate_dim);
                            const char* src = static_cast<const char*>(exp.w2.ptr);
                            for (int64_t r = 0; r < exp.hidden_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w2.data() + r * exp.intermediate_dim, exp.intermediate_dim);
                            }
                        }
                        ApplyScaleSidecarInPlace(exp.w2_scale_tensor, static_cast<int64_t>(exp.hidden_dim),
                                                 static_cast<int64_t>(exp.intermediate_dim), candidate->w2.data(),
                                                 nullptr);
                    }
                    if ((exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) &&
                        (!exp.w3_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                        exp.w3_type != GGML_TYPE_F32) {
                        const struct ggml_type_traits* traits =
                            ggml_get_type_traits(static_cast<ggml_type>(exp.w3_type));
                        if (exp.w3_int4.IsValid() && safe_reference_mode) {
                            candidate->w3.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            DequantizePackedInt4ToF32(exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                      static_cast<int64_t>(exp.hidden_dim), candidate->w3.data());
                        } else if (traits && traits->to_float) {
                            candidate->w3.resize(static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim);
                            const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w3_type), exp.hidden_dim);
                            const char* src = static_cast<const char*>(exp.w3.ptr);
                            for (int64_t r = 0; r < exp.intermediate_dim; ++r) {
                                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes),
                                                 candidate->w3.data() + r * exp.hidden_dim, exp.hidden_dim);
                            }
                        }
                    }
                    refresh_cached_tensors(candidate.get(), exp);

                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(expert_id);
                    if (it != registry->dequant_cache.end()) {
                        cached_entry = it->second;
                        if (cached_entry) {
                            cached_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    } else {
                        while (registry->dequant_cache_bytes + candidate->bytes > dequant_cache_budget &&
                               !registry->dequant_cache.empty()) {
                            auto evict_it = registry->dequant_cache.end();
                            uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
                            for (auto it_cache = registry->dequant_cache.begin();
                                 it_cache != registry->dequant_cache.end(); ++it_cache) {
                                if (!it_cache->second) {
                                    evict_it = it_cache;
                                    break;
                                }
                                if (it_cache->second->last_used < oldest_use) {
                                    oldest_use = it_cache->second->last_used;
                                    evict_it = it_cache;
                                }
                            }
                            if (evict_it == registry->dequant_cache.end()) {
                                break;
                            }
                            if (evict_it->second) {
                                registry->dequant_cache_bytes -=
                                    std::min(registry->dequant_cache_bytes, evict_it->second->bytes);
                            }
                            registry->dequant_cache.erase(evict_it);
                        }
                        if (registry->dequant_cache_bytes + candidate->bytes <= dequant_cache_budget) {
                            candidate->last_used = ++registry->dequant_cache_use_counter;
                            registry->dequant_cache_bytes += candidate->bytes;
                            registry->dequant_cache.emplace(expert_id, candidate);
                            cached_entry = std::move(candidate);
                        }
                    }
                }
            }

            // When expert weights are ggml-quantized (Q4_K etc.), skip F32 dequant entirely.
            // The quantized GEMV path in DispatchExpertFFNImpl handles these directly via vec_dot.
            const bool has_ggml_quant = !safe_reference_mode && ExpertHasGgmlQuantizedWeights(exp) &&
                                        (!exp.w2_scale_tensor || IsScalarScaleSidecar(exp.w2_scale_tensor));

            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1, w2, w3;
            if (!has_ggml_quant) {
                w1 = (cached_entry &&
                      (!exp.w1_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                      exp.w1_type != GGML_TYPE_F32)
                         ? cached_entry->w1_tensor
                         : make_weight_f32_small(exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                                 static_cast<int64_t>(exp.intermediate_dim),
                                                 static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes,
                                                 &dequantized_any);
                w2 = (cached_entry && (exp.w2_scale_tensor || ((!exp.w2_int4.IsValid() || safe_reference_mode ||
                                                                !CanUsePackedInt4MoEFastPath()) &&
                                                               exp.w2_type != GGML_TYPE_F32)))
                         ? cached_entry->w2_tensor
                         : make_weight_f32_small(exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                                 static_cast<int64_t>(exp.hidden_dim),
                                                 static_cast<int64_t>(exp.intermediate_dim), w2_dequant,
                                                 &dequantized_bytes, &dequantized_any);
                if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                    w3 = (cached_entry &&
                          (!exp.w3_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                          exp.w3_type != GGML_TYPE_F32)
                             ? cached_entry->w3_tensor
                             : make_weight_f32_small(exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                                     static_cast<int64_t>(exp.intermediate_dim),
                                                     static_cast<int64_t>(exp.hidden_dim), w3_dequant,
                                                     &dequantized_bytes, &dequantized_any);
                }
            }
            // else: w1/w2/w3 remain empty Tensors -> DispatchExpertFFNImpl uses quantized GEMV path
            if (dequantized_any) {
                moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
                moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                             std::memory_order_relaxed);
            }

            Tensor expert_input =
                Tensor::Make2D(const_cast<float*>(input_data + static_cast<size_t>(token_idx) * hidden_dim), 1,
                               static_cast<int64_t>(hidden_dim));
            Tensor expert_out = Tensor::Make2D(small_decode_output_scratch.ptr, 1, static_cast<int64_t>(hidden_dim));
            const int expert_numa_node = profiler ? profiler->GetExpertNumaNode(expert_id) : -1;
            MoEExecutionTraceContext trace_ctx;
            trace_ctx.layer_idx = layer_idx;
            trace_ctx.expert_id = expert_id;
            trace_ctx.token_idx = token_idx;
            if (batch && trace_ctx.token_idx >= 0 && trace_ctx.token_idx < static_cast<int>(batch->seq_id.size())) {
                trace_ctx.seq_id = batch->seq_id[static_cast<size_t>(trace_ctx.token_idx)];
                if (trace_ctx.seq_id >= 0 && trace_ctx.seq_id < static_cast<int>(batch->n_past.size())) {
                    trace_ctx.n_past = batch->n_past[static_cast<size_t>(trace_ctx.seq_id)];
                    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
                    const auto last_it = moe_decode_last_n_past_.find(trace_ctx.seq_id);
                    if (last_it == moe_decode_last_n_past_.end() || last_it->second != trace_ctx.n_past) {
                        moe_decode_last_n_past_[trace_ctx.seq_id] = trace_ctx.n_past;
                        trace_ctx.decode_step = moe_decode_step_ordinals_[trace_ctx.seq_id]++;
                    } else {
                        trace_ctx.decode_step = std::max(0, moe_decode_step_ordinals_[trace_ctx.seq_id] - 1);
                    }
                }
            }
            DispatchExpertFFNImpl(this, expert_numa_node, expert_input, exp, w1, w2, w3, &expert_out, true, &trace_ctx,
                                  &small_decode_input_projection_cache);

            float* dst = out_data + static_cast<size_t>(token_idx) * hidden_dim;
            const float* src = small_decode_output_scratch.ptr;
            for (size_t d = 0; d < hidden_dim; ++d) {
                dst[d] += weight * src[d];
            }
        }
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                                 out_data);
        }
        return;
    }

    const bool debug_timing = IsMoEDebugTimingEnabled();
    const auto forward_begin = std::chrono::steady_clock::now();
    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    routing_scratch.Resize(this, workspace_floats);

    moe::MoERoutingWorkspace ws;
    if (!moe::InitMoERoutingWorkspace(&ws, routing_scratch.ptr, routing_scratch.capacity * sizeof(float), batch_size,
                                      num_experts, top_k)) {
        return;
    }

    moe::MoEReorderMapView reorder_map;
    if (!moe::BuildMoEReorderMap(routing, num_experts, &reorder_map, &ws)) {
        return;
    }
    if (reorder_map.total_assignments == 0) {
        return;
    }

    const int hidden_dim_i = static_cast<int>(hidden_dim);
    const size_t packed_size = static_cast<size_t>(reorder_map.total_assignments) * hidden_dim;
    expert_input_scratch.Resize(this, packed_size);
    expert_output_scratch.Resize(this, packed_size);

    float* packed_input = expert_input_scratch.ptr;
    float* packed_output = expert_output_scratch.ptr;
    auto& reorder_pool = GetThreadPool(-1);
    const auto reorder_input_begin = std::chrono::steady_clock::now();
    moe::ReorderInputs(input_data, batch_size, hidden_dim_i, reorder_map, packed_input, &reorder_pool);
    const auto reorder_input_end = std::chrono::steady_clock::now();
    if (profile) {
        profile->reorder_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_input_end - reorder_input_begin).count());
    }

    std::unordered_set<int> local_hot_experts_set;
    local_hot_experts_set.reserve(local_hot_experts.size());
    local_hot_experts_set.insert(local_hot_experts.begin(), local_hot_experts.end());

    struct ActiveExpertWork {
        int expert_id = -1;
        int start = 0;
        int count = 0;
        int numa_node = -1;
        float ema_load = 0.0f;
        bool local_hot = false;
    };

    std::vector<ActiveExpertWork> active_work;
    active_work.reserve(static_cast<size_t>(num_experts));
    std::vector<int> current_batch_experts;
    current_batch_experts.reserve(static_cast<size_t>(num_experts));

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int start = reorder_map.expert_offsets[static_cast<size_t>(expert_id)];
        const int end = reorder_map.expert_offsets[static_cast<size_t>(expert_id + 1)];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        ActiveExpertWork work;
        work.expert_id = expert_id;
        work.start = start;
        work.count = count;
        work.local_hot = local_hot_experts_set.find(expert_id) != local_hot_experts_set.end();
        if (profiler) {
            work.numa_node = profiler->GetExpertNumaNode(expert_id);
            work.ema_load = profiler->GetEmaLoad(expert_id);
        }
        active_work.push_back(work);
        current_batch_experts.push_back(expert_id);
    }

    if (active_work.empty()) {
        return;
    }

    if (IsMoEMatmulPathDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_REORDER] assignments=%d active_experts=%zu token_indices=", reorder_map.total_assignments,
                     active_work.size());
        const int trace_count = std::min(reorder_map.total_assignments, 32);
        for (int i = 0; i < trace_count; ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", reorder_map.token_indices[i]);
        }
        std::fprintf(stderr, "\n[MOE_REORDER] expert_batches=");
        for (size_t i = 0; i < active_work.size(); ++i) {
            std::fprintf(stderr, "%s(%d:%d)", i == 0 ? "" : ",", active_work[i].expert_id, active_work[i].count);
        }
        std::fputc('\n', stderr);
    }

    int reuse_intersection = 0;
    std::unordered_set<int> previous_batch_set;
    if (!previous_batch_experts.empty()) {
        previous_batch_set.insert(previous_batch_experts.begin(), previous_batch_experts.end());
        for (int expert_id : current_batch_experts) {
            if (previous_batch_set.find(expert_id) != previous_batch_set.end()) {
                ++reuse_intersection;
            }
        }
    }
    if (!small_decode_ready) {
        const SmallDecodeExpertParallelDecision expert_parallel_decision =
            ResolveSmallDecodeExpertParallelDecision(model, batch_size, top_k, safe_reference_mode, /*worker_cap=*/1);
        if (expert_parallel_decision.forced_on) {
            const char* fallback_reason = expert_parallel_decision.reason;
            if (!small_decode_candidate) {
                fallback_reason = "shape_not_small_decode";
            } else if (small_decode_requires_general_path) {
                fallback_reason = "weights_require_general_path";
            } else if (!small_step_snapshot_ok) {
                fallback_reason = "snapshot_unavailable";
            } else if (small_step_max_expert_batch > 1) {
                fallback_reason = "expert_batch_gt_one";
            } else if (small_step_current_batch_expert_count <= 0) {
                fallback_reason = "no_active_experts";
            }
            LogSmallDecodeFallback(fallback_reason, total_assignments, 1, batch_size);
        }
    }
    const int reuse_union =
        static_cast<int>(current_batch_experts.size() + previous_batch_experts.size() - reuse_intersection);
    const int max_expert_batch = reorder_map.max_expert_batch;
    const int local_hot_count = static_cast<int>(std::count_if(
        active_work.begin(), active_work.end(), [](const ActiveExpertWork& work) { return work.local_hot; }));
    const int worker_threads = std::max(1, reorder_pool.GetNumThreads());
    const bool small_decode_step = batch_size <= 4 && total_assignments <= 8 && max_expert_batch <= 1;

    moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
    moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(active_work.size()), std::memory_order_relaxed);
    moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(reorder_map.total_assignments),
                                           std::memory_order_relaxed);
    moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
    moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection), std::memory_order_relaxed);
    moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)), std::memory_order_relaxed);
    moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(std::max(0, max_expert_batch)),
                                                std::memory_order_relaxed);

    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        registry->last_batch_experts = current_batch_experts;
    }

    if (!small_decode_step && internal::IsMoELocalityOrderingEnabled()) {
        moe_stats_total_ordering_considered_.fetch_add(1, std::memory_order_relaxed);

        const bool enough_active_experts =
            static_cast<int>(active_work.size()) >= internal::GetMoELocalityOrderingMinActiveExperts();
        const bool enough_reuse_signal =
            reuse_intersection >= internal::GetMoELocalityOrderingMinReuseIntersection() || local_hot_count > 0;

        auto count_numa_switches = [](const std::vector<ActiveExpertWork>& work_items) -> uint64_t {
            uint64_t switches = 0;
            for (size_t i = 1; i < work_items.size(); ++i) {
                const int prev = work_items[i - 1].numa_node;
                const int cur = work_items[i].numa_node;
                if (prev >= 0 && cur >= 0 && prev != cur) {
                    ++switches;
                }
            }
            return switches;
        };

        if (!enough_active_experts) {
            moe_stats_total_ordering_skipped_small_batch_.fetch_add(1, std::memory_order_relaxed);
        } else if (!enough_reuse_signal) {
            moe_stats_total_ordering_skipped_low_reuse_.fetch_add(1, std::memory_order_relaxed);
        } else {
            const uint64_t switches_before = count_numa_switches(active_work);
            moe_stats_total_ordering_numa_switches_before_.fetch_add(switches_before, std::memory_order_relaxed);

            auto ordering_score = [&previous_batch_set](const ActiveExpertWork& work) -> int {
                const bool reused = previous_batch_set.find(work.expert_id) != previous_batch_set.end();
                int score = 0;
                if (work.local_hot) score += 32;
                if (reused) score += 24;
                if (work.numa_node >= 0) score += 4;
                score += std::min(work.count, 4) * 3;
                return score;
            };

            std::stable_sort(active_work.begin(), active_work.end(),
                             [&ordering_score](const ActiveExpertWork& lhs, const ActiveExpertWork& rhs) {
                                 const int lhs_score = ordering_score(lhs);
                                 const int rhs_score = ordering_score(rhs);
                                 if (lhs_score != rhs_score) return lhs_score > rhs_score;
                                 if (lhs.ema_load != rhs.ema_load) return lhs.ema_load < rhs.ema_load;
                                 if (lhs.count != rhs.count) return lhs.count < rhs.count;
                                 if (lhs.numa_node != rhs.numa_node) return lhs.numa_node < rhs.numa_node;
                                 return lhs.expert_id < rhs.expert_id;
                             });

            const uint64_t switches_after = count_numa_switches(active_work);
            moe_stats_total_ordering_applied_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_ordering_numa_switches_after_.fetch_add(switches_after, std::memory_order_relaxed);
        }
    }

    const bool dequant_cache_enabled =
        internal::IsMoEDequantCacheEnabled() && registry != nullptr && !arm_disable_registry_dequant_cache;
    const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
    const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

    auto make_weight_f32 = [&](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                               const ggml_tensor* scale_tensor, int64_t rows, int64_t cols, AlignedScratch& scratch,
                               size_t* dequantized_bytes, bool* dequantized_any) -> Tensor {
        if (int4_binding.IsValid()) {
            if (!safe_reference_mode) {
                return Tensor();
            }
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, scratch.ptr)) {
                return Tensor();
            }
            if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scratch.ptr, nullptr)) {
                return Tensor();
            }
            if (dequantized_bytes) {
                *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
            }
            if (dequantized_any) {
                *dequantized_any = true;
            }
            return Tensor::Make2D(scratch.ptr, rows, cols);
        }
        if (!ptr || rows <= 0 || cols <= 0) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32 && !scale_tensor) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        scratch.Resize(this, static_cast<size_t>(rows * cols));
        if (wtype == GGML_TYPE_F32) {
            std::memcpy(scratch.ptr, ptr, static_cast<size_t>(rows * cols) * sizeof(float));
        } else {
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
            }
        }
        if (scale_tensor) {
            if (!ApplyScaleSidecarInPlace(scale_tensor, rows, cols, scratch.ptr, nullptr)) {
                return Tensor();
            }
        }
        if (dequantized_bytes) {
            *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
        }
        if (dequantized_any) {
            *dequantized_any = true;
        }
        return Tensor::Make2D(scratch.ptr, rows, cols);
    };

    auto dequantize_into_buffer = [](void* ptr, int ggml_type_id, const ExpertPackedInt4Weight& int4_binding,
                                     const ggml_tensor* scale_tensor, int64_t rows, int64_t cols,
                                     simd::AlignedVector<float>* dst) -> bool {
        if (!dst) return false;
        dst->clear();
        if (int4_binding.IsValid()) {
            if (!IsMoESafeReferenceModeEnabled()) return false;
            dst->resize(static_cast<size_t>(rows * cols));
            if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, dst->data())) {
                dst->clear();
                return false;
            }
            return ApplyScaleSidecarInPlace(scale_tensor, rows, cols, dst->data(), nullptr);
        }
        if (!ptr || rows <= 0 || cols <= 0) return false;
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        const size_t total = static_cast<size_t>(rows * cols);
        dst->resize(total);
        float* out = dst->data();
        if (wtype == GGML_TYPE_F32) {
            std::memcpy(out, ptr, total * sizeof(float));
        } else {
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                dst->clear();
                return false;
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out + r * cols, cols);
            }
        }
        return ApplyScaleSidecarInPlace(scale_tensor, rows, cols, dst->data(), nullptr);
    };

    uint64_t cached_experts_this_step = 0;
    std::chrono::steady_clock::duration dequant_duration{};
    std::chrono::steady_clock::duration expert_duration{};
    std::mutex work_stats_mutex;
    const bool force_inner_parallel_prefill =
        ShouldForcePrefillInnerParallelExperts() && !small_decode_step && batch_size > 1;
    const bool prefer_inner_parallel_prefill =
        force_inner_parallel_prefill ||
        (!small_decode_step && batch_size > 1 && active_work.size() < static_cast<size_t>(worker_threads));
    const bool parallelize_experts = !prefer_inner_parallel_prefill && !small_decode_step && batch_size > 1 &&
                                     active_work.size() >= static_cast<size_t>(std::max(4, worker_threads / 2)) &&
                                     registry != nullptr;
    if (IsMoECachePolicyDebugEnabled()) {
        std::fprintf(
            stderr,
            "[MOE_CACHE_POLICY] arm_disable_registry_dequant_cache=%d registry_present=%d "
            "dequant_cache_enabled=%d parallelize_experts=%d reason=%s\n",
            arm_disable_registry_dequant_cache ? 1 : 0, registry ? 1 : 0, dequant_cache_enabled ? 1 : 0,
            parallelize_experts ? 1 : 0,
            force_inner_parallel_prefill ? "prefill_inner_parallel_forced"
            : prefer_inner_parallel_prefill ? "prefill_inner_parallel"
                                            : "cache_lane_available");
    }

    if (parallelize_experts && ShouldBalanceParallelExpertWork() && active_work.size() > 1) {
        const int active_threads = std::max(1, std::min(worker_threads, static_cast<int>(active_work.size())));
        const int work_per_thread = (static_cast<int>(active_work.size()) + active_threads - 1) / active_threads;
        std::vector<ActiveExpertWork> by_cost = active_work;
        std::stable_sort(by_cost.begin(), by_cost.end(), [](const ActiveExpertWork& lhs, const ActiveExpertWork& rhs) {
            if (lhs.count != rhs.count) {
                return lhs.count > rhs.count;
            }
            return lhs.expert_id < rhs.expert_id;
        });

        std::vector<std::vector<ActiveExpertWork>> buckets(static_cast<size_t>(active_threads));
        std::vector<int64_t> bucket_cost(static_cast<size_t>(active_threads), 0);
        for (const ActiveExpertWork& work : by_cost) {
            int best_bucket = -1;
            for (int bucket = 0; bucket < active_threads; ++bucket) {
                if (static_cast<int>(buckets[static_cast<size_t>(bucket)].size()) >= work_per_thread) {
                    continue;
                }
                if (best_bucket < 0 ||
                    bucket_cost[static_cast<size_t>(bucket)] < bucket_cost[static_cast<size_t>(best_bucket)]) {
                    best_bucket = bucket;
                }
            }
            if (best_bucket < 0) {
                best_bucket = active_threads - 1;
            }
            buckets[static_cast<size_t>(best_bucket)].push_back(work);
            bucket_cost[static_cast<size_t>(best_bucket)] += std::max(1, work.count);
        }

        active_work.clear();
        active_work.reserve(by_cost.size());
        for (auto& bucket : buckets) {
            active_work.insert(active_work.end(), bucket.begin(), bucket.end());
        }
    }

    auto run_active_work_range = [&](int start_idx, int end_idx, bool allow_inner_parallel) {
        uint64_t local_cached_experts = 0;
        std::chrono::steady_clock::duration local_dequant_duration{};
        std::chrono::steady_clock::duration local_expert_duration{};

        for (int raw_idx = start_idx; raw_idx < end_idx; ++raw_idx) {
            const size_t idx = static_cast<size_t>(raw_idx);
            const ActiveExpertWork& work = active_work[idx];
            const ExpertWeights& exp = experts[static_cast<size_t>(work.expert_id)];

            if (!small_decode_step && internal::IsMoENextExpertPrefetchEnabled() && idx + 1 < active_work.size()) {
                moe_stats_total_prefetch_candidates_.fetch_add(1, std::memory_order_relaxed);
                const ActiveExpertWork& next_work = active_work[idx + 1];
                const bool next_reused = previous_batch_set.find(next_work.expert_id) != previous_batch_set.end();
                const int reuse_signals =
                    (next_work.local_hot ? 1 : 0) + (next_reused ? 1 : 0) + (next_work.count > 1 ? 1 : 0);
                const bool enough_compute_distance = work.count >= internal::GetMoEPrefetchMinCurrentExpertTokens();
                const bool under_pressure_budget =
                    static_cast<int>(active_work.size()) <= internal::GetMoEPrefetchMaxActiveExperts() &&
                    worker_threads <= internal::GetMoEPrefetchMaxThreadCount();

                if (!enough_compute_distance) {
                    moe_stats_total_prefetch_skipped_distance_.fetch_add(1, std::memory_order_relaxed);
                } else if (!under_pressure_budget) {
                    moe_stats_total_prefetch_skipped_pressure_.fetch_add(1, std::memory_order_relaxed);
                } else if (reuse_signals <= 0) {
                    moe_stats_total_prefetch_skipped_signal_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    const ExpertWeights& next_exp = experts[static_cast<size_t>(next_work.expert_id)];
                    size_t prefetch_budget = internal::GetMoEPrefetchBytes();
                    if (reuse_signals == 1) {
                        prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                    }
                    if (static_cast<int>(active_work.size()) >=
                        std::max(2, internal::GetMoEPrefetchMaxActiveExperts() / 2)) {
                        prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                    }

                    const int weight_count = (next_exp.w3.ptr != nullptr || next_exp.w3_int4.IsValid()) ? 3 : 2;
                    const size_t per_weight_budget =
                        std::max<size_t>(64, prefetch_budget / static_cast<size_t>(weight_count));
                    auto prefetch_weight = [per_weight_budget](const ExpertWeight& weight) {
                        if (!weight.ptr || weight.size == 0) return static_cast<size_t>(0);
                        const size_t bytes = std::min(per_weight_budget, weight.size);
                        densecore::simd::PrefetchRange(weight.ptr, bytes);
                        return bytes;
                    };
                    const size_t prefetched =
                        prefetch_weight(next_exp.w1) + prefetch_weight(next_exp.w2) + prefetch_weight(next_exp.w3);
                    if (prefetched > 0) {
                        moe_stats_total_prefetch_calls_.fetch_add(1, std::memory_order_relaxed);
                        moe_stats_total_prefetch_bytes_.fetch_add(static_cast<uint64_t>(prefetched),
                                                                  std::memory_order_relaxed);
                    }
                }
            }

            const auto dequant_begin =
                debug_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
            const bool should_try_cache =
                dequant_cache_enabled &&
                (cache_all_active_experts || work.local_hot ||
                 previous_batch_set.find(work.expert_id) != previous_batch_set.end() || work.count > 1);
            const size_t cacheable_bytes =
                GetExpertMatrixDequantBytes(exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                            static_cast<int64_t>(exp.hidden_dim), safe_reference_mode) +
                GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                            exp.w2_scale_tensor) +
                ((exp.w3.ptr != nullptr || exp.w3_int4.IsValid())
                     ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                                   static_cast<int64_t>(exp.hidden_dim), safe_reference_mode)
                     : 0);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> existing_entry;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(work.expert_id);
                    if (it != registry->dequant_cache.end()) {
                        existing_entry = it->second;
                        if (existing_entry) {
                            existing_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    }
                }

                if (!existing_entry) {
                    auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
                    candidate->expert_id = work.expert_id;
                    candidate->bytes = cacheable_bytes;
                    dequantize_into_buffer(exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                           static_cast<int64_t>(exp.intermediate_dim),
                                           static_cast<int64_t>(exp.hidden_dim), &candidate->w1);
                    dequantize_into_buffer(exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                           static_cast<int64_t>(exp.hidden_dim),
                                           static_cast<int64_t>(exp.intermediate_dim), &candidate->w2);
                    if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                        dequantize_into_buffer(exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                               static_cast<int64_t>(exp.intermediate_dim),
                                               static_cast<int64_t>(exp.hidden_dim), &candidate->w3);
                    }
                    refresh_cached_tensors(candidate.get(), exp);

                    std::lock_guard<std::mutex> lock(registry->mutex);
                    auto it = registry->dequant_cache.find(work.expert_id);
                    if (it != registry->dequant_cache.end()) {
                        cached_entry = it->second;
                        if (cached_entry) {
                            cached_entry->last_used = ++registry->dequant_cache_use_counter;
                        }
                    } else if (candidate->bytes <= dequant_cache_budget) {
                        while (registry->dequant_cache_bytes + candidate->bytes > dequant_cache_budget &&
                               !registry->dequant_cache.empty()) {
                            auto evict_it = registry->dequant_cache.end();
                            uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
                            for (auto it_cache = registry->dequant_cache.begin();
                                 it_cache != registry->dequant_cache.end(); ++it_cache) {
                                if (!it_cache->second) {
                                    evict_it = it_cache;
                                    break;
                                }
                                if (it_cache->second->last_used < oldest_use) {
                                    oldest_use = it_cache->second->last_used;
                                    evict_it = it_cache;
                                }
                            }
                            if (evict_it == registry->dequant_cache.end()) {
                                break;
                            }
                            if (evict_it->second) {
                                registry->dequant_cache_bytes -=
                                    std::min(registry->dequant_cache_bytes, evict_it->second->bytes);
                            }
                            registry->dequant_cache.erase(evict_it);
                        }
                        if (registry->dequant_cache_bytes + candidate->bytes <= dequant_cache_budget) {
                            candidate->last_used = ++registry->dequant_cache_use_counter;
                            registry->dequant_cache_bytes += candidate->bytes;
                            registry->dequant_cache.emplace(work.expert_id, candidate);
                            cached_entry = std::move(candidate);
                        }
                    }
                } else {
                    cached_entry = std::move(existing_entry);
                }
            }

            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1;
            Tensor w2;
            Tensor w3;
            if (cached_entry) {
                w1 = (exp.w1_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) ? Tensor()
                     : (exp.w1_type == GGML_TYPE_F32)
                         ? Tensor::Make2D(exp.w1.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                          static_cast<int64_t>(exp.hidden_dim))
                         : cached_entry->w1_tensor;
                w2 = ((exp.w2_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) &&
                      !exp.w2_scale_tensor)
                         ? Tensor()
                     : ((exp.w2_type == GGML_TYPE_F32) && !exp.w2_scale_tensor)
                         ? Tensor::Make2D(exp.w2.ptr, static_cast<int64_t>(exp.hidden_dim),
                                          static_cast<int64_t>(exp.intermediate_dim))
                         : cached_entry->w2_tensor;
                if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                    w3 = (exp.w3_int4.IsValid() && !safe_reference_mode && CanUsePackedInt4MoEFastPath()) ? Tensor()
                         : (exp.w3_type == GGML_TYPE_F32)
                             ? Tensor::Make2D(exp.w3.ptr, static_cast<int64_t>(exp.intermediate_dim),
                                              static_cast<int64_t>(exp.hidden_dim))
                             : cached_entry->w3_tensor;
                }
                ++local_cached_experts;
            } else {
                // The ggml vec_dot projection path is validated for small prefill
                // expert batches; avoid building dense weights when the quantized
                // dispatch can consume the raw ggml rows directly.
                const bool can_use_ggml_quant_gen =
                    !safe_reference_mode && ExpertHasGgmlQuantizedWeights(exp) &&
                    (!exp.w2_scale_tensor || IsScalarScaleSidecar(exp.w2_scale_tensor)) &&
                    work.count == 1 && work.count <= kMoEQuantizedProjectionMaxBatch;
                if (!can_use_ggml_quant_gen) {
                    w1 = make_weight_f32(
                        exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr, static_cast<int64_t>(exp.intermediate_dim),
                        static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes, &dequantized_any);
                    w2 = make_weight_f32(
                        exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor, static_cast<int64_t>(exp.hidden_dim),
                        static_cast<int64_t>(exp.intermediate_dim), w2_dequant, &dequantized_bytes, &dequantized_any);
                    if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                        w3 = make_weight_f32(
                            exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr, static_cast<int64_t>(exp.intermediate_dim),
                            static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes, &dequantized_any);
                    }
                }
                // else: w1/w2/w3 remain empty — DispatchExpertFFNImpl uses quantized GEMV
            }
            if (dequantized_any) {
                moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
                moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                             std::memory_order_relaxed);
            }
            if (debug_timing) {
                local_dequant_duration += (std::chrono::steady_clock::now() - dequant_begin);
            }

            Tensor expert_input = Tensor::Make2D(packed_input + static_cast<size_t>(work.start) * hidden_dim,
                                                 static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));
            Tensor expert_out = Tensor::Make2D(packed_output + static_cast<size_t>(work.start) * hidden_dim,
                                               static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));

            const auto expert_begin =
                debug_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto expert_profile_begin =
                profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            MoEExecutionTraceContext trace_ctx;
            trace_ctx.layer_idx = layer_idx;
            trace_ctx.expert_id = work.expert_id;
            trace_ctx.token_idx = (work.count == 1) ? reorder_map.token_indices[static_cast<size_t>(work.start)] : -1;
            if (batch && trace_ctx.token_idx >= 0 && trace_ctx.token_idx < static_cast<int>(batch->seq_id.size())) {
                trace_ctx.seq_id = batch->seq_id[static_cast<size_t>(trace_ctx.token_idx)];
                if (trace_ctx.seq_id >= 0 && trace_ctx.seq_id < static_cast<int>(batch->n_past.size())) {
                    trace_ctx.n_past = batch->n_past[static_cast<size_t>(trace_ctx.seq_id)];
                    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
                    const auto last_it = moe_decode_last_n_past_.find(trace_ctx.seq_id);
                    if (last_it == moe_decode_last_n_past_.end() || last_it->second != trace_ctx.n_past) {
                        moe_decode_last_n_past_[trace_ctx.seq_id] = trace_ctx.n_past;
                        trace_ctx.decode_step = moe_decode_step_ordinals_[trace_ctx.seq_id]++;
                    } else {
                        trace_ctx.decode_step = std::max(0, moe_decode_step_ordinals_[trace_ctx.seq_id] - 1);
                    }
                }
            }
            DispatchExpertFFNImpl(this, work.numa_node, expert_input, exp, w1, w2, w3, &expert_out,
                                  allow_inner_parallel, &trace_ctx, nullptr, profile);
            if (profile) {
                profile->expert_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::steady_clock::now() - expert_profile_begin)
                                                                .count());
            }
            if (debug_timing) {
                local_expert_duration += (std::chrono::steady_clock::now() - expert_begin);
            }
        }

        if (parallelize_experts) {
            std::lock_guard<std::mutex> lock(work_stats_mutex);
            cached_experts_this_step += local_cached_experts;
            dequant_duration += local_dequant_duration;
            expert_duration += local_expert_duration;
        } else {
            cached_experts_this_step += local_cached_experts;
            dequant_duration += local_dequant_duration;
            expert_duration += local_expert_duration;
        }
    };

    if (parallelize_experts) {
        const int active_threads = std::max(1, std::min(worker_threads, static_cast<int>(active_work.size())));
        if (ShouldUseDynamicParallelExpertQueue() && active_work.size() > static_cast<size_t>(active_threads)) {
            std::atomic<int> next_expert{0};
            const int queue_chunk =
                std::max(1, std::min(GetDynamicParallelExpertQueueChunk(), static_cast<int>(active_work.size())));
            reorder_pool.ParallelFor(active_threads, [&](int, int, int) {
                for (;;) {
                    const int idx = next_expert.fetch_add(queue_chunk, std::memory_order_relaxed);
                    if (idx >= static_cast<int>(active_work.size())) {
                        break;
                    }
                    run_active_work_range(idx, std::min(idx + queue_chunk, static_cast<int>(active_work.size())),
                                          false);
                }
            });
        } else {
            reorder_pool.ParallelFor(static_cast<int>(active_work.size()),
                                     [&](int start, int end, int) { run_active_work_range(start, end, false); });
        }
    } else {
        run_active_work_range(0, static_cast<int>(active_work.size()), true);
    }
    if (cached_experts_this_step > 0) {
        moe_stats_total_cached_experts_.fetch_add(cached_experts_this_step, std::memory_order_relaxed);
    }

    const auto reorder_output_begin = std::chrono::steady_clock::now();
    moe::ReorderOutputs(packed_output, hidden_dim_i, reorder_map, out_data, &reorder_pool);
    const auto reorder_output_end = std::chrono::steady_clock::now();
    if (profile) {
        profile->reduce_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(reorder_output_end - reorder_output_begin).count());
    }

    if (ShouldRunMoEReferenceCheck()) {
        RunMoEReferenceCheck(input_data, batch_size, static_cast<int>(hidden_dim), routing, experts, num_experts,
                             out_data);
    }

    if (debug_timing && batch_size >= 2 && batch_size <= 4) {
        static std::atomic<int> timing_log_count{0};
        const int log_idx = timing_log_count.fetch_add(1, std::memory_order_relaxed);
        if (log_idx < 24) {
            const auto total_ms = std::chrono::duration<double, std::milli>(reorder_output_end - forward_begin).count();
            const auto reorder_in_ms =
                std::chrono::duration<double, std::milli>(reorder_input_end - reorder_input_begin).count();
            const auto dequant_ms = std::chrono::duration<double, std::milli>(dequant_duration).count();
            const auto expert_ms = std::chrono::duration<double, std::milli>(expert_duration).count();
            const auto reorder_out_ms =
                std::chrono::duration<double, std::milli>(reorder_output_end - reorder_output_begin).count();
            std::fprintf(stderr,
                         "[MoE_TIMING #%d] batch=%d assignments=%d active_experts=%zu reorder_in=%.2fms "
                         "dequant=%.2fms expert_ffn=%.2fms reorder_out=%.2fms total=%.2fms\n",
                         log_idx, batch_size, total_assignments, active_work.size(), reorder_in_ms, dequant_ms,
                         expert_ms, reorder_out_ms, total_ms);
        }
    }
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, int layer_idx, const BatchSpec* batch,
                            const Tensor& input, const moe::MoERouteResult& routing, const ExpertWeights* experts,
                            int num_experts, Tensor* output) {
    ForwardMoE(nullptr, layer_key, layer_idx, batch, input, routing, experts, num_experts, output);
}

void CpuBackend::ResetMoEPathTrace() {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    moe_path_trace_.clear();
    moe_forward_invocation_count_.store(0, std::memory_order_relaxed);
    moe_decode_step_ordinals_.clear();
    moe_decode_last_n_past_.clear();
}

std::vector<CpuBackend::MoEPathTraceEntry> CpuBackend::GetMoEPathTraceSnapshot() const {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    return moe_path_trace_;
}

void CpuBackend::RecordMoEPathTrace(const MoEPathTraceEntry& entry) {
    std::lock_guard<std::mutex> lock(moe_path_trace_mutex_);
    moe_path_trace_.push_back(entry);
    if (moe_path_trace_.size() > 512) {
        moe_path_trace_.erase(moe_path_trace_.begin(),
                              moe_path_trace_.begin() + static_cast<std::ptrdiff_t>(moe_path_trace_.size() - 512));
    }
}

uint64_t CpuBackend::GetMoEForwardInvocationCount() const {
    return moe_forward_invocation_count_.load(std::memory_order_relaxed);
}

namespace testing {

bool ResolveQwen36SmallDecodeExpertParallelAutoEligibleForTest(bool is_qwen36_hybrid_moe, int physical_cores,
                                                               int simd_level) {
    TransformerModel model{};
    if (is_qwen36_hybrid_moe) {
        model.variant = ModelVariant::QWEN36;
        model.arch_flags.is_hybrid_ssm = true;
        model.hparams.n_experts = 256;
        model.hparams.n_experts_used = 8;
    }
    return ResolveSmallDecodeExpertParallelAutoEligible(is_qwen36_hybrid_moe ? &model : nullptr, physical_cores,
                                                        static_cast<densecore::simd::SimdLevel>(simd_level));
}

bool ResolveGemma4SmallDecodeExpertParallelAutoEligibleForTest(int physical_cores, int simd_level) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    return ResolveSmallDecodeExpertParallelAutoEligible(&model, physical_cores,
                                                        static_cast<densecore::simd::SimdLevel>(simd_level));
}

int ResolveQwen36SmallDecodeExpertWorkersForTest(int top_k, int worker_cap, int requested_override) {
    return ResolveSmallDecodeExpertWorkers(top_k, worker_cap, requested_override);
}

bool RunGgmlQuantizedProjectionForTest(CpuBackend* backend, const void* weight_ptr, int ggml_type_id,
                                       const Tensor& input, Tensor* output, int64_t N, int64_t K) {
    return TryRunGgmlQuantizedProjection(backend, weight_ptr, ggml_type_id, input, output, N, K, /*numa_node=*/0,
                                         /*allow_parallel=*/false, nullptr);
}

bool RunGgmlQuantizedFusedSwiGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                  int gate_ggml_type_id, const void* up_weight_ptr, int up_ggml_type_id,
                                                  const Tensor& input, Tensor* output, int64_t N, int64_t K,
                                                  bool use_gelu_activation) {
    if (use_gelu_activation) {
        return false;
    }
    return TryRunGgmlQuantizedFusedSwiGLUProjection(backend, gate_weight_ptr, gate_ggml_type_id, up_weight_ptr,
                                                    up_ggml_type_id, input, output, N, K, /*numa_node=*/0,
                                                    /*allow_parallel=*/false, nullptr);
}

bool RunGgmlQuantizedFusedGEGLUProjectionForTest(CpuBackend* backend, const void* gate_weight_ptr,
                                                 int gate_ggml_type_id, const void* up_weight_ptr, int up_ggml_type_id,
                                                 const Tensor& input, Tensor* output, int64_t N, int64_t K) {
    return TryRunGgmlQuantizedFusedGEGLUProjection(backend, gate_weight_ptr, gate_ggml_type_id, up_weight_ptr,
                                                   up_ggml_type_id, input, output, N, K, /*numa_node=*/0,
                                                   /*allow_parallel=*/false, nullptr);
}

}  // namespace testing

}  // namespace densecore
