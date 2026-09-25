#include "llm/matmul/internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include "densecore/backend/cpu_backend.h"
#include "densecore/runtime/optimization_bridge.h"
#include "llm/config/runtime_config.h"
#include "llm/matmul/diagnostics.h"
#include "llm/runtime/spin_wait.h"

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif
#include <mutex>
#include <vector>

#include "densecore/backend/matmul_backend.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/work_context.h"

using densecore::llm::matmul::FP8MatmulCustomParams;
using densecore::llm::matmul::FP8MatmulOpData;
using densecore::llm::matmul::HalMatmulCustomParams;
using densecore::llm::runtime::ResolveBackendRegistry;

void cb_matmul_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    ScopedInferenceWorkContext callback_context(static_cast<InferenceWorkContext*>(userdata));
    if (ith != 0 || nth <= 0 || !dst || !dst->src[0] || !dst->src[1]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalMatmulCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }
    const struct ggml_tensor* input = dst->src[0];
    const struct ggml_tensor* weight = dst->src[1];
    if (!input->data || !weight->data || !dst->data) {
        return;
    }

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(input->ne[0]) * sizeof(float);
    const bool weight_contig =
        weight->nb[0] == sizeof(float) && weight->nb[1] == static_cast<size_t>(weight->ne[0]) * sizeof(float);
    const bool output_contig =
        dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(dst->ne[0]) * sizeof(float);

    if (input->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !input_contig ||
        !weight_contig || !output_contig) {
        // Fallback to local backend matmul when HAL path is inapplicable.
        densecore::MatmulParams matmul_params;
        matmul_params.a = input->data;
        matmul_params.b = weight->data;
        matmul_params.c = dst->data;
        matmul_params.M = input->ne[1];
        matmul_params.N = weight->ne[1];
        matmul_params.K = input->ne[0];
        matmul_params.lda = input->nb[1] / ggml_type_size(input->type);
        matmul_params.ldb = weight->nb[1] / ggml_type_size(weight->type);
        matmul_params.ldc = dst->nb[1] / ggml_type_size(dst->type);
        matmul_params.trans_b = true;
        matmul_params.a_type = densecore::DType::F32;
        matmul_params.b_type = densecore::DType::F32;
        matmul_params.c_type = densecore::DType::F32;
        densecore::GetDenseCoreMatmulBackend().Execute(matmul_params);
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    densecore::BackendRegistry& registry = ResolveBackendRegistry(batch);

    densecore::ComputeBackend* backend = registry.Get(params->data.preferred_device);
    if (!backend && params->data.preferred_device != densecore::DeviceType::CPU) {
        backend = registry.Get(densecore::DeviceType::CPU);
    }
    if (!backend) {
        return;
    }

    densecore::Tensor A = densecore::Tensor::Make2D(const_cast<void*>(input->data), input->ne[1], input->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor B = densecore::Tensor::Make2D(const_cast<void*>(weight->data), weight->ne[1], weight->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor C =
        densecore::Tensor::Make2D(dst->data, dst->ne[1], dst->ne[0], densecore::DType::F32, densecore::DeviceType::CPU);
    backend->MatMulTransB(A, B, &C);
}

void cb_matmul_fp8_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const FP8MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const FP8MatmulOpData& ud = params->data;
    if (!ud.packed_weights || ud.K <= 0 || ud.N <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    static std::once_flag fp8_lut_once;
    std::call_once(fp8_lut_once, []() { densecore::hwy_kernels::InitFP8LUTs_Hwy(); });

    // =========================================================================
    // FP8 Batched GEMM: Tile-based dequant for weight reuse across M tokens
    // =========================================================================
    // For M>1, dequantizing FP8 weights tile-by-tile and computing F32 GEMM
    // on tiles avoids reloading the full weight matrix for each token.
    //
    // Tile size: TILE_N weight rows × K elements dequantized to F32.
    // The F32 tile stays in L2 cache while all M activation rows access it.
    // =========================================================================
    if (M > 1) {
        const bool input_contig =
            input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
        const bool output_contig =
            dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);

        // Thread-local dequant buffer: TILE_N × K floats
        constexpr int FP8_TILE_N = 8;  // 8 × 4096 × 4 = 128KB (fits L2)
        const int total_tiles = (ud.N + FP8_TILE_N - 1) / FP8_TILE_N;
        const int tiles_per_task = (total_tiles + nth - 1) / nth;
        const int tile_start = ith * tiles_per_task;
        const int tile_end = std::min(total_tiles, tile_start + tiles_per_task);
        if (tile_start >= tile_end) return;

        static thread_local std::vector<float> fp8_dequant_buf;
        fp8_dequant_buf.resize(static_cast<size_t>(FP8_TILE_N) * ud.K);

        const uint8_t* w_fp8 = ud.packed_weights;

        for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const int n = tile_idx * FP8_TILE_N;
            const int tile_n = std::min(FP8_TILE_N, ud.N - n);

            // Dequantize tile_n weight rows: FP8 → F32 (done ONCE per tile)
            for (int i = 0; i < tile_n; i++) {
                const uint8_t* w_row = w_fp8 + static_cast<size_t>(n + i) * ud.K;
                float* f32_row = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                if (ud.format == TransformerModel::FP8Format::E5M2) {
                    densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                } else {
                    densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                }
            }

            // Tiled GEMM: C[:, n:n+tile_n] += A × W_tile^T
            // Weight tile in L2, activation rows cycle through L1
            for (int64_t m = 0; m < M; m++) {
                const float* a_row = nullptr;
                if (input_contig) {
                    a_row = reinterpret_cast<const float*>(input->data) + m * ud.K;
                } else {
                    a_row =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
                }
                float* c_row = nullptr;
                if (output_contig) {
                    c_row = reinterpret_cast<float*>(dst->data) + m * ud.N;
                } else {
                    c_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
                }
                for (int i = 0; i < tile_n; i++) {
                    const float* w_row_f32 = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                    c_row[n + i] = densecore::simd::DotF32(a_row, w_row_f32, static_cast<size_t>(ud.K));
                }
            }
        }
        return;
    }

    // =========================================================================
    // M=1 DECODE: Single-token GEMV (existing optimized path)
    // =========================================================================
    const int64_t rows_per_task = (M + nth - 1) / nth;
    const int64_t m_start = static_cast<int64_t>(ith) * rows_per_task;
    const int64_t m_end = std::min<int64_t>(M, m_start + rows_per_task);
    if (m_start >= m_end) return;

    for (int64_t m = m_start; m < m_end; ++m) {
        const float* x_row =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
        float* y_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
        if (ud.format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::Gemv_FP8_E5M2_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights, x_row,
                                                      /*beta=*/0.0f, y_row);
        } else {
            densecore::hwy_kernels::Gemv_FP8_E4M3FN_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights,
                                                        x_row,
                                                        /*beta=*/0.0f, y_row);
        }
    }
}

using densecore::llm::matmul::Int4MatmulCustomParams;
using densecore::llm::matmul::Int4MatmulOpData;
using densecore::llm::runtime::SpinPause;

[[maybe_unused]] static void ComputeInt4MatmulReference(float* output, const float* input,
                                                        const uint8_t* packed_weights, const float* scales,
                                                        const float* zeros, int M, int K, int N, int group_size) {
    if (!output || !input || !packed_weights || !scales || M <= 0 || K <= 0 || N <= 0 || group_size <= 0 ||
        (K % group_size) != 0) {
        return;
    }

    const int num_groups = K / group_size;
    const int packed_K = (K + 1) / 2;

    for (int m = 0; m < M; ++m) {
        const float* a_row = input + static_cast<size_t>(m) * static_cast<size_t>(K);
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const float scale = scales[static_cast<size_t>(n) * static_cast<size_t>(num_groups) + g];
                const float zero = zeros ? zeros[static_cast<size_t>(n) * static_cast<size_t>(num_groups) + g] : 0.0f;
                const uint8_t* w_ptr = packed_weights + static_cast<size_t>(n) * static_cast<size_t>(packed_K) +
                                       static_cast<size_t>(g) * static_cast<size_t>(group_size / 2);
                const float* a_ptr = a_row + static_cast<size_t>(g) * static_cast<size_t>(group_size);

                for (int k = 0; k < group_size; ++k) {
                    int q = (w_ptr[k / 2] >> ((k & 1) ? 4 : 0)) & 0x0F;
                    if (q > 7) q -= 16;
                    acc += scale * (static_cast<float>(q) - zero) * a_ptr[k];
                }
            }
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = acc;
        }
    }
}

static bool ShouldUseArmInt4DirectFastPath(const ggml_tensor* input, const Int4MatmulOpData& ud, int ith) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!input || !ud.packed_weights || !ud.scales || !ud.zeros || ud.K <= 0 || ud.N <= 0 || ud.group_size <= 0 ||
        input->type != GGML_TYPE_F32 || input->ne[0] != ud.K || input->ne[1] <= 0) {
        return false;
    }

    const auto mode = densecore::llm::config::LoadArmInt4DirectFastPathMode();
    if (mode == densecore::env::RuntimeToggleMode::Off) {
        return false;
    }
    if (mode == densecore::env::RuntimeToggleMode::On) {
        return true;
    }

    static std::atomic<int> state{0};  // 0=unknown, 1=enabled, 2=disabled
    int current = state.load(std::memory_order_acquire);
    if (current != 0) {
        return current == 1;
    }

    if (ith != 0) {
        int spin_count = 0;
        while ((current = state.load(std::memory_order_acquire)) == 0) {
            SpinPause(spin_count++);
        }
        return current == 1;
    }

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    current = state.load(std::memory_order_relaxed);
    if (current == 0) {
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }
        auto& reg = densecore::OpsRegistry::Instance();

        bool ok = false;
        float max_abs_diff = std::numeric_limits<float>::infinity();

        if (reg.GemmInt4Batched) {
            const int M_check = std::min<int>(static_cast<int>(input->ne[1]), 2);
            const int N_check = std::min(ud.N, 32);
            const int K_check = ud.K;
            const int num_groups = K_check / ud.group_size;
            const int packed_K = (K_check + 1) / 2;

            std::vector<float> sample_input(static_cast<size_t>(M_check) * static_cast<size_t>(K_check));
            for (int m = 0; m < M_check; ++m) {
                const char* src_row =
                    reinterpret_cast<const char*>(input->data) + static_cast<size_t>(m) * input->nb[1];
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(sample_input.data() + static_cast<size_t>(m) * static_cast<size_t>(K_check), src_row,
                                static_cast<size_t>(K_check) * sizeof(float));
                } else {
                    for (int k = 0; k < K_check; ++k) {
                        sample_input[static_cast<size_t>(m) * static_cast<size_t>(K_check) + static_cast<size_t>(k)] =
                            *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
                    }
                }
            }

            std::vector<uint8_t> sample_weights(static_cast<size_t>(N_check) * static_cast<size_t>(packed_K));
            std::vector<float> sample_scales(static_cast<size_t>(N_check) * static_cast<size_t>(num_groups));
            std::vector<float> sample_zeros(static_cast<size_t>(N_check) * static_cast<size_t>(num_groups));
            for (int n = 0; n < N_check; ++n) {
                std::memcpy(sample_weights.data() + static_cast<size_t>(n) * static_cast<size_t>(packed_K),
                            ud.packed_weights + static_cast<size_t>(n) * static_cast<size_t>(packed_K),
                            static_cast<size_t>(packed_K));
                std::memcpy(sample_scales.data() + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            ud.scales + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            static_cast<size_t>(num_groups) * sizeof(float));
                std::memcpy(sample_zeros.data() + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            ud.zeros + static_cast<size_t>(n) * static_cast<size_t>(num_groups),
                            static_cast<size_t>(num_groups) * sizeof(float));
            }

            std::vector<float> fast_output(static_cast<size_t>(M_check) * static_cast<size_t>(N_check), 0.0f);
            std::vector<float> ref_output(static_cast<size_t>(M_check) * static_cast<size_t>(N_check), 0.0f);

            reg.GemmInt4Batched(fast_output.data(), sample_input.data(), sample_weights.data(), sample_scales.data(),
                                sample_zeros.data(), M_check, K_check, N_check, ud.group_size, 0, M_check, 0, N_check,
                                static_cast<size_t>(K_check) * sizeof(float));
            ComputeInt4MatmulReference(ref_output.data(), sample_input.data(), sample_weights.data(),
                                       sample_scales.data(), sample_zeros.data(), M_check, K_check, N_check,
                                       ud.group_size);

            ok = true;
            max_abs_diff = 0.0f;
            for (size_t i = 0; i < fast_output.size(); ++i) {
                const float diff = std::fabs(fast_output[i] - ref_output[i]);
                max_abs_diff = std::max(max_abs_diff, diff);
                const float tol = std::max(1e-4f, 1e-4f * std::fabs(ref_output[i]));
                if (!std::isfinite(fast_output[i]) || diff > tol) {
                    ok = false;
                    break;
                }
            }
        }

        state.store(ok ? 1 : 2, std::memory_order_release);
        LogMatmulValidationOnce("arm_int4_direct_fastpath", ok, max_abs_diff);
        current = ok ? 1 : 2;
    }

    return current == 1;
#else
    (void)input;
    (void)ud;
    (void)ith;
    return true;
#endif
}

void cb_matmul_int4_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    ScopedInferenceWorkContext callback_context(static_cast<InferenceWorkContext*>(userdata));
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const Int4MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const Int4MatmulOpData& ud = params->data;
    if (!ud.packed_weights || !ud.scales || !ud.zeros || ud.K <= 0 || ud.N <= 0 || ud.group_size <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);
    constexpr bool single_threading_layer = DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER != 0;
    const bool allow_direct_fast_path = ShouldUseArmInt4DirectFastPath(input, ud, ith);
    InferenceWorkContext* gemma4_profile_ctx =
        (ud.is_gemma4 && IsQwen36ProfilingEnabled()) ? GetCurrentWorkContext() : nullptr;
    auto record_gemma4_int4_used = [&](uint64_t elapsed_ns) {
        if (!gemma4_profile_ctx) {
            return;
        }
        RecordGemma4NativeInt4Timing(gemma4_profile_ctx, elapsed_ns);
    };

    // Legacy escape hatch: keep DenseCore backend threadpool path for
    // platform-specific tuning. This path is intentionally serialized at GGML
    // level to avoid nested parallelism.
    if (!allow_direct_fast_path || !single_threading_layer) {
        if (ith != 0) return;

        static thread_local std::vector<float> legacy_contig_input;
        static thread_local std::vector<float> legacy_contig_output;
        const float* input_ptr = reinterpret_cast<const float*>(input->data);
        float* output_ptr = reinterpret_cast<float*>(dst->data);

        if (!input_contig || !output_contig) {
            legacy_contig_input.resize(static_cast<size_t>(M * ud.K));
            legacy_contig_output.resize(static_cast<size_t>(M * ud.N));

            for (int64_t m = 0; m < M; ++m) {
                const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(legacy_contig_input.data() + m * ud.K, src_row,
                                static_cast<size_t>(ud.K) * sizeof(float));
                } else {
                    for (int k = 0; k < ud.K; ++k) {
                        legacy_contig_input[static_cast<size_t>(m) * ud.K + k] =
                            *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
                    }
                }
            }
            input_ptr = legacy_contig_input.data();
            output_ptr = legacy_contig_output.data();
        }

        densecore::Tensor A = densecore::Tensor::Make2D(const_cast<float*>(input_ptr), M, ud.K);
        densecore::Tensor W =
            densecore::Tensor::Make2D(const_cast<uint8_t*>(ud.packed_weights), ud.N, ud.K, densecore::DType::INT8);
        const int64_t groups_per_row = ud.K / ud.group_size;
        densecore::Tensor S = densecore::Tensor::Make2D(const_cast<float*>(ud.scales), ud.N, groups_per_row);
        densecore::Tensor Z = densecore::Tensor::Make2D(const_cast<float*>(ud.zeros), ud.N, groups_per_row);
        densecore::Tensor C = densecore::Tensor::Make2D(output_ptr, M, ud.N);
        const auto hwy_begin =
            gemma4_profile_ctx ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch()).GemmInt4(A, W, S, Z, &C, ud.group_size);
        if (gemma4_profile_ctx) {
            const auto hwy_end = std::chrono::steady_clock::now();
            record_gemma4_int4_used(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(hwy_end - hwy_begin).count()));
        }

        if (!input_contig || !output_contig) {
            for (int64_t m = 0; m < M; ++m) {
                char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
                if (dst->nb[0] == sizeof(float)) {
                    std::memcpy(dst_row, legacy_contig_output.data() + m * ud.N,
                                static_cast<size_t>(ud.N) * sizeof(float));
                } else {
                    for (int n = 0; n < ud.N; ++n) {
                        *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                            legacy_contig_output[static_cast<size_t>(m) * ud.N + n];
                    }
                }
            }
        }
        return;
    }

    // GGML-thread partitioning over N tiles: each task writes a disjoint output
    // column range [n_start, n_end), so no synchronization is required.
    const int n_per_task = (ud.N + nth - 1) / nth;
    const int n_start = ith * n_per_task;
    const int n_end = std::min(ud.N, n_start + n_per_task);
    if (n_start >= n_end) return;

    // Use batched kernel whenever input elements are contiguous within each row
    // (nb[0] == sizeof(float)). The kernel handles row padding via input_stride_bytes,
    // eliminating the per-row gather/scatter fallback for padded GGML tensors.
    const bool input_elements_contig = input->nb[0] == sizeof(float);
    if (input_elements_contig && output_contig) {
        const float* in_ptr = reinterpret_cast<const float*>(input->data);
        float* out_ptr = reinterpret_cast<float*>(dst->data);
        const auto hwy_begin =
            gemma4_profile_ctx ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        densecore::Ops::GemmInt4Batched(out_ptr, in_ptr, ud.packed_weights, ud.scales, ud.zeros, static_cast<int>(M),
                                        ud.K, ud.N, ud.group_size, 0, static_cast<int>(M), n_start, n_end,
                                        input->nb[1]);
        if (gemma4_profile_ctx) {
            const auto hwy_end = std::chrono::steady_clock::now();
            record_gemma4_int4_used(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(hwy_end - hwy_begin).count()));
        }
        return;
    }

    // Strided fallback: gather one input row at a time, compute assigned output
    // tile, scatter back. This path avoids backend threadpool usage entirely.
    static thread_local std::vector<float> gathered_input;
    static thread_local std::vector<float> partial_output;
    const int n_count = n_end - n_start;
    gathered_input.resize(static_cast<size_t>(ud.K));
    // GemvInt4_Hwy writes absolute output indices [n_start, n_end).
    // Keep that indexing contract even though this task scatters only its slice.
    partial_output.resize(static_cast<size_t>(ud.N));

    for (int64_t m = 0; m < M; ++m) {
        const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
        if (input->nb[0] == sizeof(float)) {
            std::memcpy(gathered_input.data(), src_row, static_cast<size_t>(ud.K) * sizeof(float));
        } else {
            for (int k = 0; k < ud.K; ++k) {
                gathered_input[k] = *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
            }
        }

        const auto hwy_begin =
            gemma4_profile_ctx ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        densecore::hwy_kernels::GemvInt4_Hwy(partial_output.data(), gathered_input.data(), ud.packed_weights, ud.scales,
                                             ud.zeros, ud.K, ud.N, ud.group_size, n_start, n_end);
        if (gemma4_profile_ctx) {
            const auto hwy_end = std::chrono::steady_clock::now();
            record_gemma4_int4_used(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(hwy_end - hwy_begin).count()));
        }

        char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
        if (dst->nb[0] == sizeof(float)) {
            float* dst_row_f32 = reinterpret_cast<float*>(dst_row);
            std::memcpy(dst_row_f32 + n_start, partial_output.data() + n_start,
                        static_cast<size_t>(n_count) * sizeof(float));
        } else {
            for (int n = n_start; n < n_end; ++n) {
                *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                    partial_output[static_cast<size_t>(n)];
            }
        }
    }
}
