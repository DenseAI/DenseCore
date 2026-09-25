#include "backend/cpu_moe_execution.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/memory/memory_pool.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/kernel_caps.h"
#include "llm/attention/internal.h"
#include "llm/matmul/diagnostics.h"
#include "llm/matmul/execution_policy.h"
#include "llm/matmul/q8_kernels.h"
#include "llm/matmul/quant_cache.h"
#include "llm/matmul/userdata.h"
#include "llm/models/common/family_internal.h"
#include "llm/runtime/cpu_execution.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/spin_wait.h"
#include "llm/runtime/work_context.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

using densecore::llm::models::IsHybridSSMQkvWeightName;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;
using densecore::llm::runtime::SpinPause;

#include "densecore/backend/matmul_backend.h"
#include "densecore/models/model_execution_contract.h"
#include "llm/matmul/arm_m4.h"
#include "llm/matmul/kquant_batched_kernels.h"
#include "llm/matmul/q6_small_batch.h"
#include "llm/matmul/q8_small_batch.h"

void MarkQwen36ProfileFlag(std::atomic<int>& counter);


static int BatchedDecodeRowMajorOverride() {
    static const int override_value = [] {
        if (std::getenv("DENSECORE_BATCHED_DECODE_ROW_MAJOR") == nullptr) return -1;
        return densecore::env::ParseBoolEnv("DENSECORE_BATCHED_DECODE_ROW_MAJOR", false) ? 1 : 0;
    }();
    return override_value;
}

#ifdef DENSECORE_TEST_BUILD
static std::atomic<uint64_t> batched_decode_row_major_test_ops{0};
static std::atomic<uint64_t> batched_decode_native_m4_test_ops{0};
#endif

void cb_gemv_batched_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto profile_begin = (ith == 0 && IsQwen36ProfilingEnabled()) ? std::chrono::steady_clock::now()
                                                                        : std::chrono::steady_clock::time_point{};
    auto* ud = static_cast<GemvBatchedUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(ud ? ud->work_ctx : nullptr);
    if (!ud || !ud->weight_tensor) return;
    if (nth <= 0) return;
    if (!dst || !dst->src[0] || !dst->src[1]) return;

    const struct ggml_tensor* src = dst->src[0];
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!src->data || !weight_tensor->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;

    const int N = static_cast<int>(src->ne[0]);
    const int M = static_cast<int>(src->ne[1]);
    const int K = static_cast<int>(weight_tensor->ne[1]);
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (M != ud->M || K != ud->K || N != ud->N) return;
    if (static_cast<int>(dst->ne[0]) != K || static_cast<int>(dst->ne[1]) != M) return;

    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(K, k_start + k_per_thread);
    if (k_start >= k_end) return;

    const bool input_contig = src->nb[0] == sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float);
    const char* input_base = reinterpret_cast<const char*>(src->data);
    char* output_base = reinterpret_cast<char*>(dst->data);
    const size_t input_col_stride = static_cast<size_t>(src->nb[1]);
    const size_t output_col_stride = static_cast<size_t>(dst->nb[1]);
    const int64_t output_col_stride_floats = dst->nb[1] / static_cast<int64_t>(sizeof(float));
    const size_t weight_row_stride = static_cast<size_t>(weight_tensor->nb[1]);
    const ggml_type weight_type = weight_tensor->type;
    const char* weight_name = weight_tensor->name[0] ? weight_tensor->name : "(unnamed)";
    if ((dst->nb[1] % static_cast<int64_t>(sizeof(float))) != 0) {
        throw densecore::InvalidArgumentException(
            std::string("batched GEMV output column stride is not float-aligned for ") + weight_name);
    }
    InferenceWorkContext* callback_work_ctx = ud->work_ctx ? ud->work_ctx : GetCurrentWorkContext();
    auto* callback_matmul_state = callback_work_ctx ? &GetMatmulWorkState(callback_work_ctx) : nullptr;
    const BatchSpec* callback_batch = (callback_work_ctx && GetInferenceWorkContextBatch(callback_work_ctx))
                                          ? GetInferenceWorkContextBatch(callback_work_ctx)
                                          : GetCurrentBatch();
    const InferenceExecutionPhase callback_phase =
        (callback_work_ctx && GetInferenceWorkContextPhase(callback_work_ctx) != InferenceExecutionPhase::Unknown)
            ? GetInferenceWorkContextPhase(callback_work_ctx)
            : GetCurrentExecutionPhase();
    const bool matmul_dispatch_census_enabled =
        ith == 0 && callback_work_ctx && ResolveFastPathRuntimeConfig(callback_batch).matmul_dispatch_census;
    const auto record_batched_dispatch_census = [&](const char* dispatch_path, int actual_m,
                                                    std::chrono::steady_clock::time_point begin) {
        if (!matmul_dispatch_census_enabled || !dispatch_path || dispatch_path[0] == '\0' ||
            begin == std::chrono::steady_clock::time_point{}) {
            return;
        }
        const uint64_t wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
        RecordMatmulDispatchCensus(callback_work_ctx, callback_phase, dispatch_path, weight_type, std::max(1, actual_m),
                                   K, N, wall_ns);
    };
    const auto gemma4_dense_prefill_begin = (ith == 0 && ud->gemma4_dense_prefill_native)
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    const auto record_quant_profile = [&](bool used_quantized, bool used_true_batched) {
        if (ith != 0) {
            return;
        }
        if (gemma4_dense_prefill_begin != std::chrono::steady_clock::time_point{}) {
            const uint64_t wall_ns =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - gemma4_dense_prefill_begin)
                                          .count());
            RecordGemma4DensePrefillNativeTiming(callback_work_ctx, wall_ns);
        }
        if (profile_begin == std::chrono::steady_clock::time_point()) {
            return;
        }
        InferenceWorkContext* work_ctx = callback_work_ctx;
        if (!work_ctx) {
            return;
        }
        AddQwen36ProfileNs((*GetInferenceWorkContextProfile(work_ctx)).quant_matmul_ns,
                           static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - profile_begin)
                                                     .count()));
        if (used_quantized && densecore::simd::IsArmFamily(GetRuntimeSimdLevel())) {
            MarkQwen36ProfileFlag((*GetInferenceWorkContextProfile(work_ctx)).arm_batched_quant_used);
        }
        if (used_true_batched) {
            MarkQwen36ProfileFlag((*GetInferenceWorkContextProfile(work_ctx)).q4k_true_batched_used);
        }
    };
    bool qwen36_probe_local_failed = false;
    bool qwen36_probe_local_internal_error = false;
    float qwen36_probe_local_max_abs_error = 0.0f;
    constexpr bool qwen36_shadow_probe_enabled = false;
    const auto finalize_qwen36_probe = [&]() {
        if (!qwen36_shadow_probe_enabled || !ud->qwen36_prefill_q4k_probe || !ud->qwen36_prefill_q4k_admission_key) {
            return;
        }
        InferenceWorkContext* probe_work_ctx = callback_work_ctx;
        if (probe_work_ctx) {
            (*GetInferenceWorkContextProfile(probe_work_ctx))
                .qwen36_prefill_q4k_probe_participants.fetch_add(1, std::memory_order_relaxed);
            if (qwen36_probe_local_failed || qwen36_probe_local_internal_error) {
                (*GetInferenceWorkContextProfile(probe_work_ctx))
                    .qwen36_prefill_q4k_probe_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
        AtomicMaxFloatBits(ud->qwen36_prefill_q4k_probe_max_abs_error_bits, qwen36_probe_local_max_abs_error);
        if (qwen36_probe_local_internal_error) {
            ud->qwen36_prefill_q4k_probe_internal_errors.fetch_add(1, std::memory_order_relaxed);
        }
        if (qwen36_probe_local_failed || qwen36_probe_local_internal_error) {
            ud->qwen36_prefill_q4k_probe_failures.fetch_add(1, std::memory_order_relaxed);
        }
        // Only row-owning partitions participate here. Threads with
        // k_start >= K return before finalize(), so this is intentionally not nth.
        const int participants = std::max(1, (K + k_per_thread - 1) / k_per_thread);
        const int done = ud->qwen36_prefill_q4k_probe_done.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done == participants) {
            const int failures = ud->qwen36_prefill_q4k_probe_failures.load(std::memory_order_acquire);
            const int internal_errors = ud->qwen36_prefill_q4k_probe_internal_errors.load(std::memory_order_acquire);
            const uint32_t bits = ud->qwen36_prefill_q4k_probe_max_abs_error_bits.load(std::memory_order_acquire);
            float max_abs_error = 0.0f;
            std::memcpy(&max_abs_error, &bits, sizeof(float));
            const bool pass = failures == 0 && internal_errors == 0;
            const auto reason = pass ? Qwen36PrefillQ4KBatchedRejectReason::Admitted
                                     : (internal_errors > 0 ? Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError
                                                            : Qwen36PrefillQ4KBatchedRejectReason::ProbeMismatch);
            StoreQwen36Q4KBatchedAdmission(ud->qwen36_prefill_q4k_admission_key,
                                           pass ? Qwen36Q4KBatchedAdmissionState::Pass
                                                : Qwen36Q4KBatchedAdmissionState::Reject,
                                           max_abs_error, reason);
            RecordQwen36Q4KBatchedProbeResult(probe_work_ctx, pass, max_abs_error, reason);
        }
    };

    thread_local std::vector<const float*> x_rows;
    thread_local std::vector<float> gathered_inputs;
    thread_local std::vector<float> sums;
    x_rows.resize(static_cast<size_t>(M));
    sums.resize(static_cast<size_t>(M));

    if (input_contig) {
        for (int m = 0; m < M; ++m) {
            x_rows[static_cast<size_t>(m)] =
                reinterpret_cast<const float*>(input_base + static_cast<size_t>(m) * input_col_stride);
        }
    } else {
        gathered_inputs.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * input_col_stride;
            float* dst_col = gathered_inputs.data() + static_cast<size_t>(m) * static_cast<size_t>(N);
            for (int i = 0; i < N; ++i) {
                dst_col[static_cast<size_t>(i)] =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * src->nb[0]);
            }
            x_rows[static_cast<size_t>(m)] = dst_col;
        }
    }

    const char* weight_base = reinterpret_cast<const char*>(weight_tensor->data);

    auto store_out = [&](int m, int k, float value) {
        if (ud->gemma4_prefill_safe_batched && !std::isfinite(value)) {
            value = 0.0f;
        }
        char* out_col = output_base + static_cast<size_t>(m) * output_col_stride;
        if (output_contig) {
            reinterpret_cast<float*>(out_col)[k] = value;
        } else {
            *reinterpret_cast<float*>(out_col + static_cast<size_t>(k) * dst->nb[0]) = value;
        }
    };
    auto load_out = [&](int m, int k) -> float {
        const char* out_col = output_base + static_cast<size_t>(m) * output_col_stride;
        if (output_contig) {
            return reinterpret_cast<const float*>(out_col)[k];
        }
        return *reinterpret_cast<const float*>(out_col + static_cast<size_t>(k) * dst->nb[0]);
    };
    auto maybe_log_output_partition = [&](const char* path) {
        static const bool debug_out = densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_GEMV_BATCHED_OUT", false);
        if (!debug_out) {
            return;
        }
        static const char* target_weight = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GEMV_BATCHED_OUT_WEIGHT");
        if (target_weight && target_weight[0] != '\0' && !std::strstr(weight_name, target_weight)) {
            return;
        }
        static const int target_token =
            densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_GEMV_BATCHED_OUT_TOKEN", -1);
        const int m_debug = (target_token >= 0) ? target_token : (M - 1);
        if (m_debug < 0 || m_debug >= M) {
            return;
        }
        static const int row_begin_env =
            densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_GEMV_BATCHED_OUT_ROW_BEGIN", -1);
        static const int row_end_env =
            densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_GEMV_BATCHED_OUT_ROW_END", -1);
        const int stat_begin = std::max(k_start, row_begin_env >= 0 ? row_begin_env : k_start);
        const int stat_end = std::min(k_end, row_end_env >= 0 ? row_end_env : k_end);
        if (stat_begin >= stat_end) {
            return;
        }
        static std::atomic<int> remaining{
            densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_GEMV_BATCHED_OUT_MAX_CALLS", 64)};
        int budget = remaining.load(std::memory_order_relaxed);
        while (budget > 0 && !remaining.compare_exchange_weak(budget, budget - 1, std::memory_order_relaxed,
                                                              std::memory_order_relaxed)) {}
        if (budget <= 0) {
            return;
        }

        float min_v = std::numeric_limits<float>::infinity();
        float max_v = -std::numeric_limits<float>::infinity();
        float max_abs = 0.0f;
        double sum = 0.0;
        double sum_sq = 0.0;
        int finite = 0;
        int nonzero = 0;
        for (int k = stat_begin; k < stat_end; ++k) {
            const float v = load_out(m_debug, k);
            if (!std::isfinite(v)) {
                continue;
            }
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            max_abs = std::max(max_abs, std::fabs(v));
            sum += v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            finite++;
            if (std::fabs(v) > 1.0e-12f) {
                nonzero++;
            }
        }
        if (!std::isfinite(min_v)) {
            min_v = 0.0f;
        }
        if (!std::isfinite(max_v)) {
            max_v = 0.0f;
        }

        const float* x_row = x_rows[static_cast<size_t>(m_debug)];
        float input_max_abs = 0.0f;
        double input_sum_sq = 0.0;
        int input_finite = 0;
        for (int i = 0; i < N; ++i) {
            const float v = x_row[i];
            if (!std::isfinite(v)) {
                continue;
            }
            input_max_abs = std::max(input_max_abs, std::fabs(v));
            input_sum_sq += static_cast<double>(v) * static_cast<double>(v);
            input_finite++;
        }

        float sample0 = 0.0f;
        float sample_mid = 0.0f;
        float sample_last = 0.0f;
        const int sample_mid_k = stat_begin + (stat_end - stat_begin) / 2;
        sample0 = load_out(m_debug, stat_begin);
        sample_mid = load_out(m_debug, sample_mid_k);
        sample_last = load_out(m_debug, stat_end - 1);

        float w0_max_abs = 0.0f;
        float wmid_max_abs = 0.0f;
        float wlast_max_abs = 0.0f;
        double w0_sum_sq = 0.0;
        double wmid_sum_sq = 0.0;
        double wlast_sum_sq = 0.0;
        int w0_finite = 0;
        int wmid_finite = 0;
        int wlast_finite = 0;
        const auto* debug_traits = ggml_get_type_traits(weight_type);
        if (debug_traits && debug_traits->to_float) {
            thread_local std::vector<float> debug_weight_row;
            debug_weight_row.resize(static_cast<size_t>(N));
            auto scan_weight = [&](int k, float& row_max_abs, double& row_sum_sq, int& row_finite) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                debug_traits->to_float(row_ptr, debug_weight_row.data(), N);
                for (int i = 0; i < N; ++i) {
                    const float v = debug_weight_row[static_cast<size_t>(i)];
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    row_max_abs = std::max(row_max_abs, std::fabs(v));
                    row_sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    row_finite++;
                }
            };
            scan_weight(stat_begin, w0_max_abs, w0_sum_sq, w0_finite);
            scan_weight(sample_mid_k, wmid_max_abs, wmid_sum_sq, wmid_finite);
            scan_weight(stat_end - 1, wlast_max_abs, wlast_sum_sq, wlast_finite);
        }

        std::fprintf(stderr,
                     "[GEMV_BATCHED_OUT] path=%s w=%s ith=%d nth=%d M=%d N=%d K=%d token=%d "
                     "rows=[%d,%d) stat_rows=[%d,%d) type=%d finite=%d nonzero=%d min=%.8g max=%.8g "
                     "max_abs=%.8g mean=%.8g rms=%.8g sample={%d:%.8g,%d:%.8g,%d:%.8g} "
                     "input_finite=%d input_max_abs=%.8g input_rms=%.8g "
                     "w0={finite:%d,max_abs:%.8g,rms:%.8g} wmid={finite:%d,max_abs:%.8g,rms:%.8g} "
                     "wlast={finite:%d,max_abs:%.8g,rms:%.8g}\n",
                     path ? path : "(unknown)", weight_name, ith, nth, M, N, K, m_debug, k_start, k_end, stat_begin,
                     stat_end, static_cast<int>(weight_type), finite, nonzero, min_v, max_v, max_abs,
                     finite ? sum / finite : 0.0, finite ? std::sqrt(sum_sq / finite) : 0.0, stat_begin, sample0,
                     sample_mid_k, sample_mid, stat_end - 1, sample_last, input_finite, input_max_abs,
                     input_finite ? std::sqrt(input_sum_sq / input_finite) : 0.0, w0_finite, w0_max_abs,
                     w0_finite ? std::sqrt(w0_sum_sq / w0_finite) : 0.0, wmid_finite, wmid_max_abs,
                     wmid_finite ? std::sqrt(wmid_sum_sq / wmid_finite) : 0.0, wlast_finite, wlast_max_abs,
                     wlast_finite ? std::sqrt(wlast_sum_sq / wlast_finite) : 0.0);
    };

    static const bool debug_batched_io = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GEMV_BATCHED_IO");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (debug_batched_io && ith == 0 && std::strstr(weight_name, "ssm_out")) {
        static std::atomic<int> debug_count{0};
        const int idx = debug_count.fetch_add(1, std::memory_order_relaxed);
        if (idx < 16) {
            auto log_input_col = [&](int m) {
                if (m < 0 || m >= M) return;
                const float* row = x_rows[static_cast<size_t>(m)];
                float min_v = std::numeric_limits<float>::infinity();
                float max_v = -std::numeric_limits<float>::infinity();
                float max_abs = 0.0f;
                double sum_sq = 0.0;
                double sum = 0.0;
                int finite = 0;
                for (int i = 0; i < N; ++i) {
                    const float v = row[i];
                    if (!std::isfinite(v)) continue;
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    max_abs = std::max(max_abs, std::fabs(v));
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    finite++;
                }
                if (!std::isfinite(min_v)) min_v = 0.0f;
                if (!std::isfinite(max_v)) max_v = 0.0f;
                std::fprintf(stderr,
                             "[GEMV_BATCHED_IO] w=%s call=%d token=%d M=%d N=%d K=%d finite=%d min=%.8g max=%.8g "
                             "max_abs=%.8g mean=%.8g rms=%.8g first=%.8g\n",
                             weight_name, idx, m, M, N, K, finite, min_v, max_v, max_abs, finite ? sum / finite : 0.0,
                             finite ? std::sqrt(sum_sq / finite) : 0.0, N > 0 ? row[0] : 0.0f);
            };
            log_input_col(0);
            if (M > 1) {
                log_input_col(M - 1);
            }
        }
    }
    if (weight_type == GGML_TYPE_F32) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "BATCHED_F32_CALLBACK", false, false, false);
        }
        // Fast path: contiguous layout -> Highway SIMD GEMM with Split-N parallelism.
        // GemmFP32_Hwy computes C[:, n_start:n_end) = A[M,K_gemm] x B[N_gemm,K_gemm]^T
        // Our mapping: A=input[M, N_input], B=weight[K_output, N_input], C=output[M, K_output]
        // GEMM K_gemm = N (input dim), GEMM N_gemm = K (output dim)
        // Split-N over K_output (the output dimension, which IS GEMM's N).
        const bool weight_contig = (weight_row_stride == static_cast<size_t>(N) * sizeof(float));
        if (input_contig && output_contig && weight_contig && (K % 8) == 0) {
            const auto branch_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                     : std::chrono::steady_clock::time_point{};
            const float* A = reinterpret_cast<const float*>(input_base);
            const float* B = reinterpret_cast<const float*>(weight_base);
            float* C = reinterpret_cast<float*>(output_base);
            // k_start/k_end map to n_start/n_end in GEMM Split-N convention.
            densecore::hwy_kernels::GemmFP32_Hwy(C, A, B, M, K, N, k_start, k_end);
            if (ud->gemma4_prefill_safe_batched) {
                for (int m = 0; m < M; ++m) {
                    float* out_row = reinterpret_cast<float*>(output_base + static_cast<size_t>(m) * output_col_stride);
                    for (int k = k_start; k < k_end; ++k) {
                        if (!std::isfinite(out_row[k])) {
                            out_row[k] = 0.0f;
                        }
                    }
                }
            }
            maybe_log_output_partition("f32_hwy");
            record_quant_profile(false, false);
            record_batched_dispatch_census("batched_f32_hwy", M, branch_begin);
            return;
        }

        // Strided fallback: scalar with weight row reuse
        const auto branch_begin =
            matmul_dispatch_census_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (int k = k_start; k < k_end; ++k) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            const float* w_row =
                reinterpret_cast<const float*>(weight_base + static_cast<size_t>(k) * weight_row_stride);
            for (int i = 0; i < N; ++i) {
                const float w = w_row[i];
                if (ud->gemma4_prefill_safe_batched && !std::isfinite(w)) {
                    continue;
                }
                for (int m = 0; m < M; ++m) {
                    const float x = x_rows[static_cast<size_t>(m)][i];
                    if (ud->gemma4_prefill_safe_batched && !std::isfinite(x)) {
                        continue;
                    }
                    sums[static_cast<size_t>(m)] += x * w;
                }
            }
            for (int m = 0; m < M; ++m) {
                store_out(m, k, sums[static_cast<size_t>(m)]);
            }
        }
        maybe_log_output_partition("f32_scalar");
        record_quant_profile(false, false);
        record_batched_dispatch_census("batched_f32_scalar", M, branch_begin);
        return;
    }

    if (ud->qwen36_ssm_q8_direct_batched && weight_type == GGML_TYPE_Q8_0 && input_contig && output_contig &&
        (N % QK8_0) == 0) {
        const auto branch_begin =
            matmul_dispatch_census_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (q8_traits && q8_traits->from_float) {
            const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, static_cast<int64_t>(N));
            const size_t q8_row_stride = densecore::AlignUp(q8_row_size, static_cast<size_t>(64));
            const size_t q8_total_size = q8_row_stride * static_cast<size_t>(M);
            const int64_t token_pos = (callback_batch && callback_batch->num_seqs == 1 && !callback_batch->pos.empty())
                                          ? callback_batch->pos.front()
                                          : std::numeric_limits<int64_t>::min();
            const bool can_share_q8_inputs =
                callback_work_ctx && ud->quantized_stamp && ud->slot_id >= 0 && q8_total_size > 0;
            const uint64_t q8_inputs_stamp =
                can_share_q8_inputs
                    ? ComputeGemvBatchedQuantStamp(callback_batch, M, ud->slot_id, src->data, weight_tensor->data)
                    : 0;
            const uint8_t* q8_input_base = nullptr;
            bool used_shared_q8_inputs = false;
            thread_local std::vector<uint8_t> q8_inputs;
            if (can_share_q8_inputs) {
                if (ith == 0) {
                    q8_input_base = GetOrFillBatchedQuantizedActivationCache(callback_work_ctx, src, src->data, x_rows,
                                                                             M, N, GGML_TYPE_Q8_0, q8_row_stride,
                                                                             q8_total_size, token_pos, q8_traits);
                    if (q8_input_base) {
                        ud->quantized_stamp->store(q8_inputs_stamp, std::memory_order_release);
                        used_shared_q8_inputs = true;
                    }
                } else {
                    int spin_count = 0;
                    while (ud->quantized_stamp->load(std::memory_order_acquire) != q8_inputs_stamp) {
                        if (spin_count >= 16384) {
                            break;
                        }
                        SpinPause(spin_count++);
                    }
                    if (ud->quantized_stamp->load(std::memory_order_acquire) == q8_inputs_stamp) {
                        q8_input_base = FindQuantizedActivationCacheEntry(
                            callback_work_ctx, src, src->data, static_cast<int64_t>(M) * static_cast<int64_t>(N),
                            GGML_TYPE_Q8_0, q8_total_size, -1, token_pos);
                    }
                    if (q8_input_base) {
                        used_shared_q8_inputs = true;
                    }
                }
            }
            if (!q8_input_base) {
                q8_inputs.resize(q8_total_size);
                for (int m = 0; m < M; ++m) {
                    q8_traits->from_float(x_rows[static_cast<size_t>(m)],
                                          q8_inputs.data() + static_cast<size_t>(m) * q8_row_stride,
                                          static_cast<int64_t>(N));
                }
                q8_input_base = q8_inputs.data();
            }

            auto run_scalar_cols = [&](int m, int col_begin, int col_end) {
                if (col_begin >= col_end) {
                    return;
                }
                const auto* x_blocks =
                    reinterpret_cast<const block_q8_0*>(q8_input_base + static_cast<size_t>(m) * q8_row_stride);
                for (int k = col_begin; k < col_end; ++k) {
                    const auto* w_blocks =
                        reinterpret_cast<const block_q8_0*>(weight_base + static_cast<size_t>(k) * weight_row_stride);
                    store_out(m, k, DenseCoreQ8_0BlockDot(w_blocks, x_blocks, N));
                }
            };

            if ((K % 4) == 0) {
                auto packed = GetOrCreateQ8RepackedGemvWeight(weight_base, K, N, /*force_enable=*/true);
                if (packed && packed->blocks_per_row > 0) {
                    const size_t q8_4x8_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
                    const int k_aligned_start = (k_start + 3) & ~3;
                    const int k_aligned_end = k_end & ~3;
                    int m = 0;
                    for (; m < M; ++m) {
                        const uint8_t* q_ptr = q8_input_base + static_cast<size_t>(m) * q8_row_stride;
                        run_scalar_cols(m, k_start, std::min(k_aligned_start, k_end));
                        if (k_aligned_start < k_aligned_end) {
                            const size_t packed_offset = static_cast<size_t>(k_aligned_start / 4) *
                                                         static_cast<size_t>(packed->blocks_per_row) *
                                                         q8_4x8_block_bytes;
                            float* out_group =
                                reinterpret_cast<float*>(output_base + static_cast<size_t>(m) * output_col_stride) +
                                k_aligned_start;
                            DenseCoreGemvQ8_0_4x8Q8_0Generic(N, out_group, packed->data.data() + packed_offset, q_ptr,
                                                             k_aligned_end - k_aligned_start);
                        }
                        run_scalar_cols(m, std::max(k_aligned_end, k_start), k_end);
                    }
                    LogMatmulPathOnce("qwen36_ssm_q8_0_repacked_direct_batched");
                    if (ith == 0) {
                        densecore::llm::attention::RecordSharedQuantReuse(used_shared_q8_inputs);
                    }
                    maybe_log_output_partition("q8_repacked_direct");
                    record_quant_profile(true, false);
                    record_batched_dispatch_census("qwen36_ssm_q8_0_repacked_direct_batched", M, branch_begin);
                    return;
                }
            }

            for (int k = k_start; k < k_end; ++k) {
                for (int m = 0; m < M; ++m) {
                    run_scalar_cols(m, k, k + 1);
                }
            }
            LogMatmulPathOnce("qwen36_ssm_q8_0_direct_batched");
            if (ith == 0) {
                densecore::llm::attention::RecordSharedQuantReuse(used_shared_q8_inputs);
            }
            maybe_log_output_partition("q8_direct_scalar");
            record_quant_profile(true, false);
            record_batched_dispatch_census("qwen36_ssm_q8_0_direct_batched", M, branch_begin);
            return;
        }
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type =
            (ud->input_quant_type != GGML_TYPE_F32) ? ud->input_quant_type : type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        const size_t quant_row_stride = (ud->quant_row_stride > 0)
                                            ? ud->quant_row_stride
                                            : densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
        // Some x86 traits advertise nrc=2 although K-quant kernels write one
        // result. Use the shared kernel capability before trusting row counts.
        if ((weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q5_K || weight_type == GGML_TYPE_Q6_K) &&
            !densecore::kernels::KQuantVecDotRowPairSupported()) {
            vec_dot_nrows = 1;
        }
        const bool can_quantize_inputs = input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
                                         quant_row_stride <= kMaxQuantInputBufferSize;
        const bool can_use_q4k_true_batched = can_quantize_inputs && weight_type == GGML_TYPE_Q4_K &&
                                              vec_dot_type == GGML_TYPE_Q8_K && IsQ4KTrueBatchedKernelEnabled() &&
                                              (N % QK_K == 0);
        const bool can_use_q5k_true_batched =
            can_quantize_inputs && weight_type == GGML_TYPE_Q5_K && vec_dot_type == GGML_TYPE_Q8_K && (N % QK_K == 0);
        if (ud->require_q4k_true_batched && !can_use_q4k_true_batched) {
            throw densecore::InvalidArgumentException(
                std::string("LFM2 prefill Q4_K true-batched callback rejected for ") + weight_name);
        }
        static constexpr size_t kMaxFullBatchedQActCacheBytes = 32ull * 1024ull * 1024ull;
        const int quant_tile_cols = ResolveQuantBatchedTileCols(kMaxSmallBatchColsHard, vec_dot_nrows,
                                                                can_use_q4k_true_batched || can_use_q5k_true_batched);

        const bool can_use_q8_0_repacked_batched =
            (ud->gemma4_dense_prefill_native || ud->lfm2_q8_repacked_batched || ud->qwen36_ssm_q8_repacked_batched) &&
            !ud->gemma4_prefill_safe_batched && weight_type == GGML_TYPE_Q8_0 && input_contig && output_contig &&
            type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_0 && input_type_traits && input_type_traits->from_float &&
            (N % QK8_0) == 0 && (K % 4) == 0 && M >= 4;
        if (can_use_q8_0_repacked_batched) {
            const auto branch_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                     : std::chrono::steady_clock::time_point{};
            auto* q8_profile_ctx = callback_work_ctx;
            const auto q8_now = []() { return std::chrono::steady_clock::now(); };
            const auto q8_elapsed_ns = [](std::chrono::steady_clock::time_point begin) -> uint64_t {
                return static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
                        .count());
            };
            const auto q8_weight_begin = q8_now();
            auto packed = GetOrCreateQ8RepackedGemvWeight(weight_base, K, N, /*force_enable=*/true);
            if (q8_profile_ctx) {
                (*GetInferenceWorkContextProfile(q8_profile_ctx))
                    .q8_batched_weight_cache_ns.fetch_add(q8_elapsed_ns(q8_weight_begin), std::memory_order_relaxed);
            }
            if (packed && packed->blocks_per_row > 0) {
                const size_t q8_4x8_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
                const size_t q8_row_stride = quant_row_stride;
                const size_t q8_total_size = q8_row_stride * static_cast<size_t>(M);
                const int64_t token_pos =
                    (callback_batch && callback_batch->num_seqs == 1 && !callback_batch->pos.empty())
                        ? callback_batch->pos.front()
                        : std::numeric_limits<int64_t>::min();
                const int k_aligned_start = (k_start + 3) & ~3;
                const int k_aligned_end = k_end & ~3;
                const int m_aligned_end = M & ~3;
                // The 4x8 Q8_0 GEMM repacks activation rows a second time, so it
                // only wins for wide projections. The per-N threshold keeps small
                // hybrid-SSM projections on the row-GEMV lane (where the second
                // repack does not pay for itself) while admitting wide ones.
                //
                // Qwen hybrid-SSM Q8_0 prefill: on x86 C4 these go through the
                // dedicated AMX tiled path (qwen36_ssm_q8_prefill_amx), so this
                // block is never reached there. On C4A (no AMX) the fused
                // ssm_qkv_gate projection is N=12288 and was degrading to per-token
                // GEMV (q8_batched_gemv_ops dominated, ~282s compute), which is the
                // ARM prefill bottleneck. Admit qwen36_ssm_q8_repacked_batched here
                // so those wide SSM projections use the tiled 4x8 GEMM — the ARM
                // analogue of the x86 AMX path — while N<threshold projections stay
                // on GEMV.
                bool q8_true_gemm_candidate = ud->gemma4_dense_prefill_native || ud->gemma4_prefill_safe_batched;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                constexpr int kQ8TrueGemmMinN = 2048;
                // Keep the DenseCore callback viable for direct callers even
                // though maintained Qwen prefill dispatches earlier.
                q8_true_gemm_candidate = q8_true_gemm_candidate || ud->qwen36_ssm_q8_repacked_batched;
#else
                constexpr int kQ8TrueGemmMinN = 4096;
                // C4A has no AMX SSM path, so this block is the primary lane for
                // Qwen hybrid-SSM Q8 prefill. Admit it to the tiled 4x8 GEMM (the
                // ARM analogue of the x86 AMX path) for wide projections.
                q8_true_gemm_candidate = q8_true_gemm_candidate || ud->qwen36_ssm_q8_repacked_batched;
#endif
                const bool large_qwen36_prefill = ShouldUseArmQwen36LargeQ8Prefill(
                    ud->qwen36_ssm_q8_repacked_batched && callback_work_ctx &&
                        GetInferenceWorkContextModelVariant(callback_work_ctx) == ModelVariant::QWEN36 &&
                        callback_phase == InferenceExecutionPhase::Prefill,
                    M, N, K);
                const bool q8_true_gemm_shape_profitable =
                    q8_true_gemm_candidate && (N >= kQ8TrueGemmMinN || large_qwen36_prefill);
                const bool can_use_q8_true_gemm =
                    q8_true_gemm_shape_profitable && DenseCoreQ8_0Gemm4x8FastBackendCompiled() && m_aligned_end >= 4 &&
                    k_aligned_start < k_aligned_end && (output_col_stride % sizeof(float)) == 0;
                const size_t q8_gemm_packed_bytes = DenseCoreQ8_0RowsTo4x8PackedBytes(m_aligned_end, N);
                const bool can_share_q8_gemm_inputs = can_use_q8_true_gemm && callback_work_ctx &&
                                                      q8_gemm_packed_bytes > 0 &&
                                                      q8_gemm_packed_bytes <= kMaxFullBatchedQActCacheBytes;
                const uint64_t q8_gemm_inputs_stamp = can_share_q8_gemm_inputs ? [&]() {
                    uint64_t stamp =
                        ComputeGemvBatchedQuantStamp(callback_batch, m_aligned_end, ud->slot_id, src->data, nullptr);
                    stamp ^= 0x8a5cd789635d2dffull;
                    stamp *= 1099511628211ull;
                    stamp ^= static_cast<uint64_t>(static_cast<uint32_t>(N));
                    stamp *= 1099511628211ull;
                    return stamp == 0 ? 1 : stamp;
                }()
                                                                               : 0;
                thread_local std::vector<uint8_t> q8_gemm_inputs;
                const bool can_share_q8_inputs = callback_work_ctx && ud->quantized_stamp && q8_total_size > 0 &&
                                                 q8_total_size <= kMaxFullBatchedQActCacheBytes;
                const uint64_t q8_inputs_stamp =
                    can_share_q8_inputs
                        ? ComputeGemvBatchedQuantStamp(callback_batch, M, ud->slot_id, src->data, weight_tensor->data)
                        : 0;
                const uint8_t* q8_input_base = nullptr;
                bool used_shared_q8_inputs = false;
                thread_local std::vector<uint8_t> q8_inputs;
                if (can_share_q8_inputs) {
                    if (ith == 0) {
                        const auto quant_begin = q8_now();
                        q8_input_base = GetOrFillBatchedQuantizedActivationCache(
                            callback_work_ctx, src, src->data, x_rows, M, N, GGML_TYPE_Q8_0, q8_row_stride,
                            q8_total_size, token_pos, input_type_traits);
                        if (q8_profile_ctx) {
                            (*GetInferenceWorkContextProfile(q8_profile_ctx))
                                .q8_batched_activation_quant_ns.fetch_add(q8_elapsed_ns(quant_begin),
                                                                          std::memory_order_relaxed);
                        }
                        if (q8_input_base) {
                            ud->quantized_stamp->store(q8_inputs_stamp, std::memory_order_release);
                            used_shared_q8_inputs = true;
                        }
                    } else {
                        const auto wait_begin = q8_now();
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != q8_inputs_stamp) {
                            if (spin_count >= 16384) {
                                break;
                            }
                            SpinPause(spin_count++);
                        }
                        if (ud->quantized_stamp->load(std::memory_order_acquire) == q8_inputs_stamp) {
                            q8_input_base = FindQuantizedActivationCacheEntry(
                                callback_work_ctx, src, src->data, static_cast<int64_t>(M) * static_cast<int64_t>(N),
                                GGML_TYPE_Q8_0, q8_total_size, -1, token_pos);
                        }
                        if (q8_input_base) {
                            used_shared_q8_inputs = true;
                        }
                        if (q8_profile_ctx) {
                            (*GetInferenceWorkContextProfile(q8_profile_ctx))
                                .q8_batched_activation_wait_ns.fetch_add(q8_elapsed_ns(wait_begin),
                                                                         std::memory_order_relaxed);
                        }
                    }
                }
                if (!q8_input_base) {
                    const auto quant_begin = q8_now();
                    q8_inputs.resize(q8_total_size);
                    for (int m = 0; m < M; ++m) {
                        input_type_traits->from_float(x_rows[static_cast<size_t>(m)],
                                                      q8_inputs.data() + static_cast<size_t>(m) * q8_row_stride,
                                                      static_cast<int64_t>(N));
                    }
                    q8_input_base = q8_inputs.data();
                    if (q8_profile_ctx) {
                        (*GetInferenceWorkContextProfile(q8_profile_ctx))
                            .q8_batched_activation_quant_ns.fetch_add(q8_elapsed_ns(quant_begin),
                                                                      std::memory_order_relaxed);
                    }
                }

                const auto run_scalar_cols = [&](int m, int col_begin, int col_end) {
                    if (col_begin >= col_end) {
                        return;
                    }
                    const uint8_t* q8_row = q8_input_base + static_cast<size_t>(m) * q8_row_stride;
                    for (int k = col_begin; k < col_end; ++k) {
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                        float sum = 0.0f;
                        type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q8_row, 0, 1);
                        store_out(m, k, sum);
                    }
                };

                const uint8_t* q8_gemm_input_base = nullptr;
                if (can_share_q8_gemm_inputs) {
                    if (ith == 0) {
                        const auto pack_begin = q8_now();
                        q8_gemm_input_base =
                            GetOrFillQ8_0RowsTo4x8ActivationCache(callback_work_ctx, src, src->data, q8_input_base,
                                                                  q8_row_stride, m_aligned_end, N, token_pos);
                        if (q8_profile_ctx) {
                            (*GetInferenceWorkContextProfile(q8_profile_ctx))
                                .q8_batched_activation_pack_ns.fetch_add(q8_elapsed_ns(pack_begin),
                                                                         std::memory_order_relaxed);
                        }
                        if (q8_gemm_input_base) {
                            callback_matmul_state->q8_gemm_packed_stamp.store(q8_gemm_inputs_stamp,
                                                                              std::memory_order_release);
                        }
                    } else {
                        const auto pack_wait_begin = q8_now();
                        int spin_count = 0;
                        while (callback_matmul_state->q8_gemm_packed_stamp.load(std::memory_order_acquire) !=
                               q8_gemm_inputs_stamp) {
                            if (spin_count >= 16384) {
                                break;
                            }
                            SpinPause(spin_count++);
                        }
                        if (callback_matmul_state->q8_gemm_packed_stamp.load(std::memory_order_acquire) ==
                                q8_gemm_inputs_stamp &&
                            DenseCoreValidateQ8_0RowsTo4x8ActivationCache(
                                callback_work_ctx, src, src->data, m_aligned_end, N, q8_gemm_packed_bytes, token_pos)) {
                            q8_gemm_input_base = callback_matmul_state->q8_gemm_packed_buffer.data();
                        }
                        if (q8_profile_ctx) {
                            (*GetInferenceWorkContextProfile(q8_profile_ctx))
                                .q8_batched_activation_pack_ns.fetch_add(q8_elapsed_ns(pack_wait_begin),
                                                                         std::memory_order_relaxed);
                        }
                    }
                }
                int m = 0;
                if (can_use_q8_true_gemm) {
                    if (!q8_gemm_input_base) {
                        const auto pack_begin = q8_now();
                        const bool packed_inputs =
                            DenseCorePackQ8_0RowsTo4x8(q8_input_base, q8_row_stride, m_aligned_end, N, q8_gemm_inputs);
                        if (q8_profile_ctx) {
                            (*GetInferenceWorkContextProfile(q8_profile_ctx))
                                .q8_batched_activation_pack_ns.fetch_add(q8_elapsed_ns(pack_begin),
                                                                         std::memory_order_relaxed);
                        }
                        if (packed_inputs) {
                            q8_gemm_input_base = q8_gemm_inputs.data();
                        }
                    }
                }
                if (q8_profile_ctx) {
                    (*GetInferenceWorkContextProfile(q8_profile_ctx))
                        .q8_batched_used_ops.fetch_add(1, std::memory_order_relaxed);
                    if (q8_gemm_input_base) {
                        (*GetInferenceWorkContextProfile(q8_profile_ctx))
                            .q8_batched_true_gemm_ops.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        (*GetInferenceWorkContextProfile(q8_profile_ctx))
                            .q8_batched_gemv_ops.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                const auto compute_begin = q8_now();
                if (q8_gemm_input_base) {
                    for (; m < m_aligned_end; m += 4) {
                        for (int r = 0; r < 4; ++r) {
                            run_scalar_cols(m + r, k_start, std::min(k_aligned_start, k_end));
                        }
                        const size_t packed_offset = static_cast<size_t>(k_aligned_start / 4) *
                                                     static_cast<size_t>(packed->blocks_per_row) * q8_4x8_block_bytes;
                        const size_t input_packed_offset = static_cast<size_t>(m / 4) *
                                                           static_cast<size_t>(packed->blocks_per_row) *
                                                           q8_4x8_block_bytes;
                        float* out_group =
                            reinterpret_cast<float*>(output_base + static_cast<size_t>(m) * output_col_stride) +
                            k_aligned_start;
                        DenseCoreGemmQ8_0_4x8x4Q8_0Generic(
                            N, out_group, output_col_stride / sizeof(float), packed->data.data() + packed_offset,
                            q8_gemm_input_base + input_packed_offset, k_aligned_end - k_aligned_start);
                        for (int r = 0; r < 4; ++r) {
                            run_scalar_cols(m + r, std::max(k_aligned_end, k_start), k_end);
                        }
                    }
                }
                for (; m < M; ++m) {
                    run_scalar_cols(m, k_start, std::min(k_aligned_start, k_end));
                    if (k_aligned_start < k_aligned_end) {
                        const size_t packed_offset = static_cast<size_t>(k_aligned_start / 4) *
                                                     static_cast<size_t>(packed->blocks_per_row) * q8_4x8_block_bytes;
                        float* out_group =
                            reinterpret_cast<float*>(output_base + static_cast<size_t>(m) * output_col_stride) +
                            k_aligned_start;
                        const uint8_t* q8_row = q8_input_base + static_cast<size_t>(m) * q8_row_stride;
                        if (ud->qwen36_ssm_q8_repacked_batched || ud->lfm2_q8_repacked_batched) {
                            DenseCoreGemvQ8_0_4x8Q8_0Generic(N, out_group, packed->data.data() + packed_offset, q8_row,
                                                             k_aligned_end - k_aligned_start);
                        } else {
                            ggml_gemv_q8_0_4x8_q8_0(N, out_group, 0, packed->data.data() + packed_offset, q8_row, 1,
                                                    k_aligned_end - k_aligned_start);
                        }
                    }
                    run_scalar_cols(m, std::max(k_aligned_end, k_start), k_end);
                }
                if (ud->gemma4_prefill_safe_batched) {
                    for (int row = 0; row < M; ++row) {
                        float* out_row =
                            reinterpret_cast<float*>(output_base + static_cast<size_t>(row) * output_col_stride);
                        for (int col = k_start; col < k_end; ++col) {
                            if (!std::isfinite(out_row[col])) {
                                out_row[col] = 0.0f;
                            }
                        }
                    }
                }
                LogMatmulPathOnce(ud->qwen36_ssm_q8_repacked_batched
                                      ? "qwen36_ssm_q8_0_repacked_batched"
                                      : (ud->lfm2_q8_repacked_batched ? "lfm2_q8_0_repacked_batched"
                                                                      : "gemma4_q8_0_repacked_batched"));
                if (q8_profile_ctx) {
                    (*GetInferenceWorkContextProfile(q8_profile_ctx))
                        .q8_batched_compute_ns.fetch_add(q8_elapsed_ns(compute_begin), std::memory_order_relaxed);
                }
                record_quant_profile(true, false);
                if (ith == 0) {
                    densecore::llm::attention::RecordSharedQuantReuse(used_shared_q8_inputs);
                }
                record_batched_dispatch_census(ud->qwen36_ssm_q8_repacked_batched
                                                   ? "qwen36_ssm_q8_0_repacked_batched"
                                                   : (ud->lfm2_q8_repacked_batched ? "lfm2_q8_0_repacked_batched"
                                                                                   : "gemma4_q8_0_repacked_batched"),
                                               M, branch_begin);
                return;
            }
        }

        if (can_quantize_inputs) {
            thread_local std::vector<uint8_t> quant_inputs_tls;
            // Four native-layout input rows share weight decoding in registers.
            // Keep prefill, other models, and explicit reference routes unchanged.
            const bool native_m4_type =
                (weight_type == GGML_TYPE_Q8_0 && vec_dot_type == GGML_TYPE_Q8_0 && Q8SmallBatchDot4Supported()) ||
                (weight_type == GGML_TYPE_Q6_K && vec_dot_type == GGML_TYPE_Q8_K && Q6KQ8KM4NativeAvailable());
            if (native_m4_type && callback_work_ctx &&
                GetInferenceWorkContextModelVariant(callback_work_ctx) == ModelVariant::QWEN36 &&
                callback_phase == InferenceExecutionPhase::Decode && M == 4 && N >= 256 &&
                N % ggml_blck_size(weight_type) == 0 && quant_row_stride >= quant_row_size && input_contig &&
                output_contig && !ud->force_reference_scalar && !ud->disable_quant_nrc_fast &&
                !ud->require_q4k_true_batched && BatchedDecodeRowMajorOverride() != 0) {
                const auto begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                  : std::chrono::steady_clock::time_point{};
                quant_inputs_tls.resize(quant_row_stride * 4);
                for (int m = 0; m < 4; ++m) {
                    input_type_traits->from_float(x_rows[m], quant_inputs_tls.data() + m * quant_row_stride, N);
                }
                bool completed = true;
                for (int k = k_start; k < k_end; ++k) {
                    const void* row = weight_base + static_cast<size_t>(k) * weight_row_stride;
                    float* out = reinterpret_cast<float*>(output_base) + k;
                    if (weight_type == GGML_TYPE_Q8_0) {
                        Q8SmallBatchDot4(N, row, quant_inputs_tls.data(), quant_row_stride, out,
                                         output_col_stride_floats);
                    } else {
                        float sums[4];
                        if (!ComputeQ6KQ8KM4Native(row, quant_inputs_tls.data(), quant_row_stride, 4, N, sums)) {
                            completed = false;
                            break;
                        }
                        for (int m = 0; m < 4; ++m) out[m * output_col_stride_floats] = sums[m];
                    }
                }
                if (completed) {
                    const char* path = weight_type == GGML_TYPE_Q8_0 ? "q8_native_m4" : "q6k_native_m4";
                    if (ith == 0) {
                        densecore::llm::attention::RecordSharedQuantReuse(false);
#ifdef DENSECORE_TEST_BUILD
                        batched_decode_native_m4_test_ops.fetch_add(1, std::memory_order_relaxed);
#endif
                    }
                    LogMatmulPathOnce(path);
                    record_batched_dispatch_census(path, M, begin);
                    maybe_log_output_partition(path);
                    record_quant_profile(true, false);
                    finalize_qwen36_probe();
                    return;
                }
                // A rejected row falls through and the existing path rewrites
                // this worker's entire output partition.
            }
            const auto& qact_config = ResolveFastPathRuntimeConfig(callback_batch);
            const bool gemma4_safe_q4k_prefill_qact_cache =
                ud->gemma4_prefill_safe_batched && weight_type == GGML_TYPE_Q4_K && vec_dot_type == GGML_TYPE_Q8_K &&
                input_contig && can_use_q4k_true_batched;
            const bool qact_cache_enabled_for_batched = ud->require_q4k_true_batched ||
                                                        gemma4_safe_q4k_prefill_qact_cache ||
                                                        QuantizedActivationCacheEnabled(qact_config);
            const size_t full_quant_total_size = quant_row_stride * static_cast<size_t>(M);
            const int64_t token_pos = (callback_batch && callback_batch->num_seqs == 1 && !callback_batch->pos.empty())
                                          ? callback_batch->pos.front()
                                          : std::numeric_limits<int64_t>::min();
            const bool can_use_full_batched_qact =
                qact_cache_enabled_for_batched && callback_work_ctx && ud->quantized_stamp && M > quant_tile_cols &&
                full_quant_total_size > 0 && full_quant_total_size <= kMaxFullBatchedQActCacheBytes &&
                can_use_q4k_true_batched;
            const uint64_t full_batched_qact_stamp =
                can_use_full_batched_qact
                    ? ComputeGemvBatchedQuantStamp(callback_batch, M, ud->slot_id, src->data, weight_tensor->data)
                    : 0;
            const uint8_t* full_batched_qact_base = nullptr;
            bool used_full_batched_qact_buffer = false;
            if (can_use_full_batched_qact) {
                if (ith == 0) {
                    full_batched_qact_base = GetOrFillBatchedQuantizedActivationCache(
                        callback_work_ctx, src, src->data, x_rows, M, N, vec_dot_type, quant_row_stride,
                        full_quant_total_size, token_pos, input_type_traits);
                    if (full_batched_qact_base) {
                        ud->quantized_stamp->store(full_batched_qact_stamp, std::memory_order_release);
                        used_full_batched_qact_buffer = true;
                    }
                } else {
                    int spin_count = 0;
                    while (ud->quantized_stamp->load(std::memory_order_acquire) != full_batched_qact_stamp) {
                        if (spin_count >= 4096) {
                            break;
                        }
                        SpinPause(spin_count++);
                    }
                    if (ud->quantized_stamp->load(std::memory_order_acquire) == full_batched_qact_stamp) {
                        full_batched_qact_base = FindQuantizedActivationCacheEntry(
                            callback_work_ctx, src, src->data, static_cast<int64_t>(M) * static_cast<int64_t>(N),
                            vec_dot_type, full_quant_total_size, -1, token_pos);
                    }
                    if (full_batched_qact_base) {
                        used_full_batched_qact_buffer = true;
                    }
                }
            }
            const bool can_sync_on_stamp =
                M <= quant_tile_cols && nth > 1 && ud->slot_id >= 0 && ud->quant_input_shared && ud->quantized_stamp;
            const uint64_t expected_stamp =
                can_sync_on_stamp
                    ? ComputeGemvBatchedQuantStamp(callback_batch, M, ud->slot_id, src->data, weight_tensor->data)
                    : 0;
            bool logged_quant_reuse = false;

            // Repacked 8x8 GEMM for batched decode.
            //
            // At M==1 Q4_K already runs through the interleaved q4_K_8x8 layout
            // (ggml_gemv_q4_K_8x8_q8_K on the runtime repack cache, see
            // inference_matmul_gemv_custom.inl), but every M>1 step dropped that
            // layout and stayed on raw GGUF blocks. The raw-block true-batched row
            // kernel below does amortize the Q4_K unpack across the batch, so the
            // gap is not weight traffic -- it is that it works one weight row at a
            // time, while ggml_gemm_*_8x8_q8_K takes 8 interleaved weight rows
            // against 4 activation rows in one register-blocked pass. That is the
            // same kernel llama.cpp reaches at nrows>3 (repack.cpp
            // forward_mul_mat_one_chunk) and is where its concurrency scaling comes
            // from.
            //
            // Covers Q4_K, Q5_K and Q6_K, which is deliberate: by weight bytes a
            // Q4_K_M build of Qwen3.5-0.8B is 47% Q6_K (the 246 MB tied
            // token_embd/lm_head), 32% Q4_K and 20% Q5_K, so a Q4_K-only version
            // reaches barely a third of what a decode step reads.
            //
            // This branch owns exactly the same output rows as the fallback
            // below ([k_start, k_end)), so a thread that rejects here (e.g. the
            // repack cache declines the weight for budget reasons) still writes the
            // same range and no output row can be left uncovered.
            static const bool batched_decode_repack_gemm_enabled =
                densecore::env::ParseBoolEnv("DENSECORE_BATCHED_DECODE_REPACK_GEMM", true);
            // The kernels consume 4 activation rows per call. A batch that is not a
            // multiple of 4 has to pad the final group, and the padded arithmetic is
            // pure waste: measured on Qwen3.5-0.8B-Q4_K_M, padding a 2-wide batch up
            // to 4 cost ~3.5% against the existing true-batched row kernels, while an
            // exact 4-wide batch gained ~4%. So take this path only for exact
            // multiples of 4 unless the operator opts into padding.
            static const bool batched_decode_repack_gemm_pad =
                densecore::env::ParseBoolEnv("DENSECORE_BATCHED_DECODE_REPACK_GEMM_PAD", false);
            const bool batched_decode_repack_gemm_width_ok =
                batched_decode_repack_gemm_pad ? (M >= 2) : (M >= 4 && (M % 4) == 0);
            // Q4_K is the measured default. Q5_K/Q6_K coverage is opt-in: the
            // kernels only exist vectorized on ARM (RepackedDenseGemmTypeSupported
            // enforces that), and even there the win is unmeasured while the
            // repacked copies cost ~350 MB extra resident for this model, so a
            // platform A/B has to justify it before it becomes a default.
            static const bool batched_decode_repack_gemm_kquants =
                densecore::env::ParseBoolEnv("DENSECORE_BATCHED_DECODE_REPACK_GEMM_KQUANTS", false);
            const bool batched_decode_repack_gemm_type_ok =
                weight_type == GGML_TYPE_Q4_K ||
                (batched_decode_repack_gemm_kquants &&
                 densecore::kernels::RepackedDenseGemmTypeSupported(static_cast<int32_t>(weight_type)));
            const bool qwen38_prefill_q4k_8x8 = callback_phase == InferenceExecutionPhase::Prefill &&
                                                weight_type == GGML_TYPE_Q4_K && ud->qwen38_q4k_8x8_weight &&
                                                ud->qwen38_q4k_8x8_weight->data && ud->qwen38_q4k_8x8_rows == K &&
                                                ud->qwen38_q4k_8x8_cols == N;
            if (batched_decode_repack_gemm_enabled && batched_decode_repack_gemm_width_ok &&
                batched_decode_repack_gemm_type_ok && vec_dot_type == GGML_TYPE_Q8_K && output_contig &&
                !ud->force_reference_scalar && (!ud->require_q4k_true_batched || qwen38_prefill_q4k_8x8) &&
                !ud->gemma4_prefill_safe_batched && (!callback_batch || callback_batch->lora_map.empty()) &&
                (N % QK_K) == 0 && (K % 8) == 0 && (k_start % 8) == 0 && (k_end % 8) == 0) {
                const auto& repack_config = ResolveFastPathRuntimeConfig(callback_batch);
                Q4KRepackedGemvRejectReason repack_reject = Q4KRepackedGemvRejectReason::None;
                const bool repack_enabled = Q4KRepackedGemvEnabled(repack_config, &repack_reject);
                const bool repack_phase_ok = callback_phase == InferenceExecutionPhase::Decode ||
                                             (repack_config.q4k_repacked_gemv_allow_prefill &&
                                              callback_phase == InferenceExecutionPhase::Prefill);
                if (repack_enabled && repack_phase_ok && repack_reject == Q4KRepackedGemvRejectReason::None) {
                    const auto repack_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                             : std::chrono::steady_clock::time_point{};
                    const bool repack_is_q4k = weight_type == GGML_TYPE_Q4_K;
                    densecore::kernels::Q4KRepackedGemvCacheLookup repack_lookup;
                    // Q4_K keeps the established typed cache; the other K-quants use
                    // the byte-addressed twin. Both share one memory budget.
                    std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight> packed_q4k;
                    std::shared_ptr<densecore::kernels::RepackedDenseGemmWeight> packed_other;
                    if (repack_is_q4k && !qwen38_prefill_q4k_8x8) {
                        packed_q4k = densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(weight_tensor->data, K, N,
                                                                                          &repack_lookup);
                    } else {
                        packed_other = densecore::kernels::GetOrCreateRepackedDenseGemmWeight(
                            static_cast<int32_t>(weight_type), weight_tensor->data, K, N, &repack_lookup);
                    }
                    if (!qwen38_prefill_q4k_8x8) {
                        RecordQ4KRepackedGemvCacheLookup(callback_work_ctx, repack_lookup);
                        Q4KRepackedGemvShouldAutoDisableForLookup(callback_work_ctx, repack_config, repack_lookup,
                                                                  &repack_reject);
                    }
                    const bool repack_shape_ok =
                        repack_is_q4k
                            ? (qwen38_prefill_q4k_8x8 || (packed_q4k && packed_q4k->rows == static_cast<int64_t>(K) &&
                                                          packed_q4k->cols == static_cast<int64_t>(N)))
                            : (packed_other && packed_other->rows == static_cast<int64_t>(K) &&
                               packed_other->cols == static_cast<int64_t>(N));
                    if (repack_shape_ok && repack_reject == Q4KRepackedGemvRejectReason::None) {
                        const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, static_cast<int64_t>(N));
                        const int tile_start = k_start / 8;
                        const int tile_end = k_end / 8;
                        const int tile_cols = (tile_end - tile_start) * 8;
                        // Only reachable with a padded width when the operator opted
                        // in above; the default gate admits exact multiples of 4, so
                        // group_count * 4 == M and nothing is discarded.
                        const int padded_m = ((M + 3) / 4) * 4;
                        const int group_count = padded_m / 4;
                        thread_local std::vector<float> repack_gather;
                        thread_local std::vector<uint8_t> repack_q8x4;
                        thread_local std::vector<float> repack_pad_out;
                        bool repack_ok = q8_row_bytes > 0 && tile_cols > 0 && group_count > 0;
                        densecore::runtime::BatchedActivationPackCache::Snapshot shared_q8x4;
                        const uint8_t* q8x4_base = nullptr;
                        if (repack_ok) {
                            const auto fill_q8x4 = [&](std::vector<uint8_t>& packed) {
                                repack_gather.resize(static_cast<size_t>(4) * static_cast<size_t>(N));
                                packed.resize(static_cast<size_t>(group_count) * 4 * q8_row_bytes);
                                for (int group = 0; group < group_count; ++group) {
                                    for (int r = 0; r < 4; ++r) {
                                        const int m = std::min(group * 4 + r, M - 1);
                                        std::memcpy(
                                            repack_gather.data() + static_cast<size_t>(r) * static_cast<size_t>(N),
                                            x_rows[static_cast<size_t>(m)], static_cast<size_t>(N) * sizeof(float));
                                    }
                                    ggml_quantize_mat_q8_K_4x8(repack_gather.data(),
                                                               packed.data() +
                                                                   static_cast<size_t>(group) * 4 * q8_row_bytes,
                                                               static_cast<int64_t>(N));
                                }
                            };
                            // Userdata identifies the graph operation across worker-local map3
                            // tensor views and distinguishes consumers of reused input storage;
                            // execution_generation invalidates cached graphs on their next run.
                            // Packing is synchronized, but kernels run outside the cache lock.
                            if (callback_work_ctx && nth > 1) {
                                shared_q8x4 = callback_matmul_state->q8k_batched_pack_cache.GetOrFill(
                                    {GetInferenceWorkContextExecutionGeneration(callback_work_ctx), ud, src->data, M,
                                     N},
                                    fill_q8x4);
                            }
                            if (shared_q8x4) {
                                q8x4_base = shared_q8x4->data();
                            } else {
                                fill_q8x4(repack_q8x4);
                                q8x4_base = repack_q8x4.data();
                            }
                        }
                        const auto run_group = [&](const void* q8x4_group, float* out, size_t out_stride_floats) {
                            if (repack_is_q4k) {
                                const auto* packed_base =
                                    qwen38_prefill_q4k_8x8
                                        ? static_cast<const densecore::kernels::Q4KRepackedGemvBlock*>(
                                              ud->qwen38_q4k_8x8_weight->data)
                                        : packed_q4k->blocks.data();
                                ggml_gemm_q4_K_8x8_q8_K(N, out, out_stride_floats,
                                                        packed_base + static_cast<size_t>(tile_start) *
                                                                          static_cast<size_t>(N / QK_K),
                                                        q8x4_group, 4, tile_cols);
                                return true;
                            }
                            return densecore::kernels::RunRepackedDenseGemmGroup(
                                *packed_other, q8x4_group, out, out_stride_floats, tile_start, tile_end);
                        };
                        if (repack_ok) {
                            for (int group = 0; group < group_count && repack_ok; ++group) {
                                const int group_rows = std::min(4, M - group * 4);
                                const void* q8x4_group = q8x4_base + static_cast<size_t>(group) * 4 * q8_row_bytes;
                                if (group_rows == 4) {
                                    float* out_tile = reinterpret_cast<float*>(
                                        output_base + static_cast<size_t>(group * 4) * output_col_stride +
                                        static_cast<size_t>(k_start) * sizeof(float));
                                    repack_ok =
                                        run_group(q8x4_group, out_tile, static_cast<size_t>(output_col_stride_floats));
                                    continue;
                                }
                                // Padded final group: land it in scratch so the
                                // duplicate rows never touch the real output.
                                repack_pad_out.resize(static_cast<size_t>(4) * static_cast<size_t>(tile_cols));
                                repack_ok =
                                    run_group(q8x4_group, repack_pad_out.data(), static_cast<size_t>(tile_cols));
                                if (!repack_ok) {
                                    break;
                                }
                                for (int r = 0; r < group_rows; ++r) {
                                    float* out_row = reinterpret_cast<float*>(
                                        output_base + static_cast<size_t>(group * 4 + r) * output_col_stride +
                                        static_cast<size_t>(k_start) * sizeof(float));
                                    std::memcpy(out_row,
                                                repack_pad_out.data() +
                                                    static_cast<size_t>(r) * static_cast<size_t>(tile_cols),
                                                static_cast<size_t>(tile_cols) * sizeof(float));
                                }
                            }
                        }
                        if (repack_ok) {
                            if (ith == 0 && callback_work_ctx) {
                                (*GetInferenceWorkContextProfile(callback_work_ctx))
                                    .q4k_repacked_gemv_used.store(1, std::memory_order_relaxed);
                                (*GetInferenceWorkContextProfile(callback_work_ctx))
                                    .q4k_repacked_gemv_used_ops.fetch_add(1, std::memory_order_relaxed);
                                if (qwen38_prefill_q4k_8x8) {
                                    (*GetInferenceWorkContextProfile(callback_work_ctx))
                                        .qwen36_prefill_q4k_batched_used.store(1, std::memory_order_relaxed);
                                }
                            }
                            LogMatmulPathOnce("repacked_batched_gemm_8x8");
                            record_batched_dispatch_census("repacked_batched_gemm_8x8", M, repack_begin);
                            maybe_log_output_partition("repacked_batched_gemm_8x8");
                            record_quant_profile(true, false);
                            finalize_qwen36_probe();
                            return;
                        }
                        // A partially written group would corrupt this thread's rows,
                        // so a mid-group failure has to be impossible rather than
                        // recovered: run_group only fails on a rejected shape, which
                        // is checked before the first call.
                    }
                }
            }

            // Row-major traversal keeps each weight row hot while all token
            // tiles consume it. Each vec_dot still unpacks the quantized blocks;
            // its operands and per-output accumulation order remain unchanged.
            // The C4A profile uses this traversal for Qwen36 four-request decode by default.
            // Explicit 0/1 overrides retain their behavior for every model.
            const bool batched_decode_row_major_enabled = ShouldUseBatchedDecodeRowMajor(
                callback_work_ctx ? GetInferenceWorkContextModelVariant(callback_work_ctx) : ModelVariant::UNKNOWN,
                callback_phase, M, BatchedDecodeRowMajorOverride());
            if (batched_decode_row_major_enabled && M > quant_tile_cols && quant_tile_cols >= 1 && output_contig &&
                vec_dot_nrows >= quant_tile_cols && !ud->force_reference_scalar && !ud->disable_quant_nrc_fast &&
                !ud->require_q4k_true_batched) {
                const auto batch_reuse_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                              : std::chrono::steady_clock::time_point{};
                const size_t batch_quant_total_size = quant_row_stride * static_cast<size_t>(M);
                quant_inputs_tls.resize(batch_quant_total_size);
                for (int m = 0; m < M; ++m) {
                    input_type_traits->from_float(x_rows[static_cast<size_t>(m)],
                                                  quant_inputs_tls.data() + static_cast<size_t>(m) * quant_row_stride,
                                                  static_cast<int64_t>(N));
                }
                const void* reuse_sample_row_ptr = weight_base + static_cast<size_t>(k_start) * weight_row_stride;
                if (ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, weight_name,
                                                         reuse_sample_row_ptr, quant_inputs_tls.data(), x_rows[0], N)) {
                    if (ith == 0) {
                        densecore::llm::attention::RecordSharedQuantReuse(false);
                    }
#if defined(DENSECORE_TARGET_C4A) && defined(__aarch64__)
                    static const bool arm_m4_tile_enabled = densecore::env::ParseBoolEnv("DENSECORE_ARM_M4_TILE", true);
                    const bool use_arm_m4 =
                        arm_m4_tile_enabled && M == 4 && callback_work_ctx &&
                        GetInferenceWorkContextModelVariant(callback_work_ctx) == ModelVariant::QWEN36 &&
                        callback_phase == InferenceExecutionPhase::Decode &&
                        ((weight_type == GGML_TYPE_Q8_0 && vec_dot_type == GGML_TYPE_Q8_0) ||
                         (weight_type == GGML_TYPE_Q6_K && vec_dot_type == GGML_TYPE_Q8_K));
#endif
                    for (int k = k_start; k < k_end; ++k) {
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
#if defined(DENSECORE_TARGET_C4A) && defined(__aarch64__)
                        if (use_arm_m4) {
                            const bool pair = k + 1 < k_end;
                            float tail[8];
                            float* dst = pair ? reinterpret_cast<float*>(output_base) + k : tail;
                            if (densecore_arm_m4::Compute(weight_type, N, row_ptr, pair ? weight_row_stride : 0,
                                                          quant_inputs_tls.data(), quant_row_stride, dst,
                                                          pair ? output_col_stride_floats : 2)) {
                                if (!pair) {
                                    for (int m = 0; m < 4; ++m)
                                        *reinterpret_cast<float*>(output_base + size_t(m) * output_col_stride +
                                                                  size_t(k) * sizeof(float)) = tail[2 * m];
                                }
                                k += pair ? 1 : 0;
                                continue;
                            }
                        }
#endif
                        int weight_rows = 1;
#if defined(__aarch64__) || defined(_M_ARM64)
                        if (quant_tile_cols == 2 && k + 1 < k_end) weight_rows = 2;
#endif
                        for (int tile_start = 0; tile_start < M; tile_start += quant_tile_cols) {
                            const int tile_m = std::min(quant_tile_cols, M - tile_start);
                            float* out_ptr = reinterpret_cast<float*>(
                                output_base + static_cast<size_t>(tile_start) * output_col_stride +
                                static_cast<size_t>(k) * sizeof(float));
                            const auto* quant_rows =
                                quant_inputs_tls.data() + static_cast<size_t>(tile_start) * quant_row_stride;
                            if (tile_m == 2 && weight_rows == 2) {
                                type_traits_cpu->vec_dot(N, out_ptr, output_col_stride_floats, row_ptr,
                                                         weight_row_stride, quant_rows, quant_row_stride, 2);
                            } else if (tile_m == 2) {
                                // nrc=2 writes a 2x2 tile; bx=0 repeats the weight
                                // row. Copy only its first result for each token.
                                float tile[4] = {};
                                type_traits_cpu->vec_dot(N, tile, 2, row_ptr, 0, quant_rows, quant_row_stride, 2);
                                out_ptr[0] = tile[0];
                                out_ptr[output_col_stride_floats] = tile[2];
                            } else {
                                for (int r = 0; r < weight_rows; ++r) {
                                    const void* tail_weight =
                                        weight_base + static_cast<size_t>(k + r) * weight_row_stride;
                                    type_traits_cpu->vec_dot(N, out_ptr + r, 0, tail_weight, 0, quant_rows, 0, 1);
                                }
                            }
                        }
                        k += weight_rows - 1;
                    }
#ifdef DENSECORE_TEST_BUILD
                    if (ith == 0) batched_decode_row_major_test_ops.fetch_add(1, std::memory_order_relaxed);
#endif
                    LogMatmulPathOnce("gemv_batched_quant_row_major");
                    record_batched_dispatch_census("gemv_batched_quant_row_major", M, batch_reuse_begin);
                    maybe_log_output_partition("gemv_batched_quant_row_major");
                    record_quant_profile(true, false);
                    finalize_qwen36_probe();
                    return;
                }
            }

            for (int tile_start = 0; tile_start < M; tile_start += quant_tile_cols) {
                const int tile_m = std::min(quant_tile_cols, M - tile_start);
                const auto tile_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                                       : std::chrono::steady_clock::time_point{};
                const size_t quant_total_size = quant_row_stride * static_cast<size_t>(tile_m);
                const bool use_shared_quant_buffer = can_sync_on_stamp && tile_start == 0;
                const uint8_t* quant_input_base = nullptr;
                bool used_local_quant_buffer = false;

                if (used_full_batched_qact_buffer && full_batched_qact_base) {
                    quant_input_base = full_batched_qact_base + static_cast<size_t>(tile_start) * quant_row_stride;
                    if (ith == 0 && !logged_quant_reuse) {
                        densecore::llm::attention::RecordSharedQuantReuse(true);
                        logged_quant_reuse = true;
                    }
                } else if (use_shared_quant_buffer) {
                    if (ith == 0) {
                        for (int m = 0; m < tile_m; ++m) {
                            uint8_t* q_ptr = ud->quant_input_shared + static_cast<size_t>(m) * quant_row_stride;
                            input_type_traits->from_float(x_rows[static_cast<size_t>(tile_start + m)], q_ptr,
                                                          static_cast<int64_t>(N));
                        }
                        ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                    } else {
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                            if (spin_count >= 4096) {
                                quant_inputs_tls.resize(quant_total_size);
                                for (int m = 0; m < tile_m; ++m) {
                                    uint8_t* q_ptr =
                                        quant_inputs_tls.data() + static_cast<size_t>(m) * quant_row_stride;
                                    input_type_traits->from_float(x_rows[static_cast<size_t>(tile_start + m)], q_ptr,
                                                                  static_cast<int64_t>(N));
                                }
                                used_local_quant_buffer = true;
                                break;
                            }
                            SpinPause(spin_count++);
                        }
                    }
                    quant_input_base = used_local_quant_buffer ? quant_inputs_tls.data() : ud->quant_input_shared;
                    if (quant_input_base == ud->quant_input_shared && ith == 0 && !logged_quant_reuse) {
                        densecore::llm::attention::RecordSharedQuantReuse(true);
                        logged_quant_reuse = true;
                    }
                } else {
                    quant_inputs_tls.resize(quant_total_size);
                    for (int m = 0; m < tile_m; ++m) {
                        uint8_t* q_ptr = quant_inputs_tls.data() + static_cast<size_t>(m) * quant_row_stride;
                        input_type_traits->from_float(x_rows[static_cast<size_t>(tile_start + m)], q_ptr,
                                                      static_cast<int64_t>(N));
                    }
                    quant_input_base = quant_inputs_tls.data();
                    if (ith == 0 && !logged_quant_reuse) {
                        densecore::llm::attention::RecordSharedQuantReuse(false);
                        logged_quant_reuse = true;
                    }
                }

                if (ud->require_q4k_true_batched && !quant_input_base) {
                    throw densecore::InvalidArgumentException(
                        std::string("LFM2 prefill Q4_K true-batched quant input unavailable for ") + weight_name);
                }
                const bool can_use_quant_nrc_fast = output_contig && vec_dot_nrows >= tile_m;
                const void* sample_row_ptr = weight_base + static_cast<size_t>(k_start) * weight_row_stride;
                const bool allow_native_q4k_vecdot =
                    ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, weight_name, sample_row_ptr,
                                                         quant_input_base, x_rows[static_cast<size_t>(tile_start)], N);

                if (!ud->force_reference_scalar && !ud->disable_quant_nrc_fast && !ud->require_q4k_true_batched &&
                    can_use_quant_nrc_fast && quant_input_base && allow_native_q4k_vecdot) {
                    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0 &&
                        tile_start == 0) {
                        LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "GGML_QUANT_NRC_M_CALLBACK", true,
                                                true, false);
                    }
                    for (int k = k_start; k < k_end; ++k) {
                        char* out_col = output_base + static_cast<size_t>(tile_start) * output_col_stride;
                        float* out_ptr = reinterpret_cast<float*>(out_col + static_cast<size_t>(k) * sizeof(float));
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
#if defined(__aarch64__) || defined(_M_ARM64)
                        if (tile_m == 2 && k + 1 < k_end) {
                            // Consume both weights and both tokens from the native
                            // 2x2 tile. Never pair across this worker's row boundary.
                            type_traits_cpu->vec_dot(N, out_ptr, output_col_stride_floats, row_ptr, weight_row_stride,
                                                     quant_input_base, quant_row_stride, 2);
                            ++k;
                            continue;
                        }
#endif
                        if (tile_m == 2) {
                            // Native nrc=2 produces two weights by two tokens.
                            // bx=0 duplicates a weight; direct output would also
                            // overwrite the next weight row (and the final guard).
                            float tile[4] = {};
                            type_traits_cpu->vec_dot(N, tile, 2, row_ptr, 0, quant_input_base, quant_row_stride, 2);
                            out_ptr[0] = tile[0];
                            out_ptr[output_col_stride_floats] = tile[2];
                        } else {
                            type_traits_cpu->vec_dot(N, out_ptr, 0, row_ptr, 0, quant_input_base, 0, 1);
                        }
                    }
                    LogMatmulPathOnce("gemv_batched_quant_nrc");
                    record_batched_dispatch_census("gemv_batched_quant_nrc", tile_m, tile_begin);
                    continue;
                }

                if (!ud->force_reference_scalar && can_use_q4k_true_batched && quant_input_base &&
                    (!ud->qwen36_prefill_q4k_admission_key || ud->qwen36_prefill_q4k_admitted)) {
                    alignas(64) std::array<float, kMaxSmallBatchColsHard> row_sums{};
                    alignas(64) std::array<float, kMaxSmallBatchColsHard> row_sums_next{};
                    static const bool debug_q4k_kernel_check = []() {
                        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_CHECK");
                        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                    }();
                    static const float kDebugKernelWarnDiff = []() {
                        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_TOL");
                        if (!env || env[0] == '\0') return 1e-3f;
                        char* end = nullptr;
                        const float v = std::strtof(env, &end);
                        if (end == env || !std::isfinite(v) || v < 0.0f) return 1e-3f;
                        return v;
                    }();
                    static std::atomic<int> q4k_kernel_warn_count{0};
                    float max_abs_diff = 0.0f;
                    int max_diff_k = -1;
                    int max_diff_m = -1;
                    float max_diff_batched = 0.0f;
                    float max_diff_ref = 0.0f;
                    bool all_rows_ok = true;
                    bool probe_failed = false;
                    const bool run_qwen36_probe = qwen36_shadow_probe_enabled && ud->qwen36_prefill_q4k_probe &&
                                                  ud->qwen36_prefill_q4k_admission_key;
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
                    const bool allow_q4k_row_pair =
                        !debug_q4k_kernel_check && !run_qwen36_probe && ggml_cpu_has_avx2() && N >= 1024;
#else
                    const bool allow_q4k_row_pair = false;
#endif
                    constexpr float kQwen36Q4KBatchedProbeTol = 1e-3f;
                    for (int k = k_start; k < k_end; ++k) {
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
                        if (allow_q4k_row_pair && k + 1 < k_end) {
                            const void* row_next_ptr = weight_base + static_cast<size_t>(k + 1) * weight_row_stride;
                            if (ComputeQ4KQ8KBatchedRow2Avx2(row_ptr, row_next_ptr, quant_input_base, quant_row_stride,
                                                             tile_m, N, row_sums.data(), row_sums_next.data())) {
                                for (int m = 0; m < tile_m; ++m) {
                                    store_out(tile_start + m, k, row_sums[static_cast<size_t>(m)]);
                                    store_out(tile_start + m, k + 1, row_sums_next[static_cast<size_t>(m)]);
                                }
                                ++k;
                                continue;
                            }
                        }
#else
                        (void)allow_q4k_row_pair;
#endif
                        if (!ComputeQ4KQ8KBatchedRow(row_ptr, quant_input_base, quant_row_stride, tile_m, N,
                                                     row_sums.data())) {
                            all_rows_ok = false;
                            if (run_qwen36_probe) {
                                qwen36_probe_local_internal_error = true;
                                for (int m = 0; m < tile_m; ++m) {
                                    float ref = 0.0f;
                                    const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_row_stride;
                                    type_traits_cpu->vec_dot(N, &ref, 0, row_ptr, 0, q_ptr, 0, 1);
                                    store_out(tile_start + m, k, ref);
                                }
                                continue;
                            }
                            break;
                        }

                        if (debug_q4k_kernel_check || run_qwen36_probe) {
                            for (int m = 0; m < tile_m; ++m) {
                                float ref = 0.0f;
                                const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_row_stride;
                                type_traits_cpu->vec_dot(N, &ref, 0, row_ptr, 0, q_ptr, 0, 1);
                                float probe_value = row_sums[static_cast<size_t>(m)];
                                if (run_qwen36_probe &&
                                    Qwen36Q4KBatchedProbeForceFailThreadForTest().load(std::memory_order_relaxed) ==
                                        ith &&
                                    k == k_start && m == 0) {
                                    probe_value += 1.0f;
                                }
                                const float diff = std::fabs(probe_value - ref);
                                if (run_qwen36_probe && (!std::isfinite(diff) || diff > kQwen36Q4KBatchedProbeTol)) {
                                    probe_failed = true;
                                    qwen36_probe_local_failed = true;
                                }
                                if (diff > max_abs_diff) {
                                    max_abs_diff = diff;
                                    max_diff_k = k;
                                    max_diff_m = tile_start + m;
                                    max_diff_batched = row_sums[static_cast<size_t>(m)];
                                    max_diff_ref = ref;
                                }
                                if (run_qwen36_probe) {
                                    store_out(tile_start + m, k, ref);
                                }
                            }
                        }
                        if (run_qwen36_probe) {
                            qwen36_probe_local_max_abs_error = std::max(qwen36_probe_local_max_abs_error, max_abs_diff);
                            continue;
                        }
                        if (probe_failed) {
                            all_rows_ok = false;
                            break;
                        }

                        for (int m = 0; m < tile_m; ++m) {
                            store_out(tile_start + m, k, row_sums[static_cast<size_t>(m)]);
                        }
                    }
                    if (run_qwen36_probe) {
                        qwen36_probe_local_max_abs_error = std::max(qwen36_probe_local_max_abs_error, max_abs_diff);
                        if (probe_failed || !all_rows_ok) {
                            qwen36_probe_local_failed = true;
                        }
                        continue;
                    }
                    if (debug_q4k_kernel_check && max_abs_diff > kDebugKernelWarnDiff) {
                        const int warn_idx = q4k_kernel_warn_count.fetch_add(1, std::memory_order_relaxed);
                        if (warn_idx < 32) {
                            std::cerr << "[Q4K_BATCHED_CHECK] WARN ith=" << ith << " max_abs_diff=" << max_abs_diff
                                      << " at(k,m)=" << max_diff_k << "," << max_diff_m
                                      << " batched=" << max_diff_batched << " ref=" << max_diff_ref << std::endl;
                        }
                    }
                    if (!all_rows_ok && ud->qwen36_prefill_q4k_admitted && ud->qwen36_prefill_q4k_admission_key) {
                        InferenceWorkContext* downgrade_ctx = callback_work_ctx;
                        DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(
                            ud->qwen36_prefill_q4k_admission_key, max_abs_diff,
                            Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError, downgrade_ctx);
                    }
                    if (all_rows_ok) {
                        if (ith == 0 && ud->qwen36_prefill_q4k_admission_key) {
                            InferenceWorkContext* work_ctx = callback_work_ctx;
                            if (work_ctx) {
                                (*GetInferenceWorkContextProfile(work_ctx))
                                    .qwen36_prefill_q4k_batched_used.store(1, std::memory_order_relaxed);
                            }
                        }
                        record_quant_profile(true, true);
                        record_batched_dispatch_census("q4k_q8k_true_batched", tile_m, tile_begin);
                        continue;
                    }
                }
                if (!ud->force_reference_scalar && can_use_q5k_true_batched && quant_input_base) {
                    alignas(64) std::array<float, kMaxSmallBatchColsHard> row_sums{};
                    bool all_rows_ok = true;
                    for (int k = k_start; k < k_end; ++k) {
                        const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                        if (!ComputeQ5KQ8KBatchedRow(row_ptr, quant_input_base, quant_row_stride, tile_m, N,
                                                     row_sums.data())) {
                            all_rows_ok = false;
                            break;
                        }
                        for (int m = 0; m < tile_m; ++m) {
                            store_out(tile_start + m, k, row_sums[static_cast<size_t>(m)]);
                        }
                    }
                    if (all_rows_ok) {
                        record_quant_profile(true, true);
                        record_batched_dispatch_census("q5k_q8k_true_batched", tile_m, tile_begin);
                        continue;
                    }
                }
                if (ud->require_q4k_true_batched) {
                    throw densecore::InvalidArgumentException(
                        std::string("LFM2 prefill Q4_K true-batched kernel failed for ") + weight_name);
                }

                for (int k = k_start; k < k_end; ++k) {
                    const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                    for (int m = 0; m < tile_m; ++m) {
                        float sum = 0.0f;
                        const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_row_stride;
                        type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                        store_out(tile_start + m, k, sum);
                    }
                }
                record_batched_dispatch_census("quant_vecdot_scalar", tile_m, tile_begin);
            }
            maybe_log_output_partition("quant_vecdot_scalar");
            record_quant_profile(true, false);
            finalize_qwen36_probe();
            return;
        }
    }
    if (ud->require_q4k_true_batched) {
        throw densecore::InvalidArgumentException(
            std::string("LFM2 prefill Q4_K true-batched callback could not quantize input for ") + weight_name);
    }

    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
        LogHybridSSMQkvDispatch(weight_name, weight_type, M, K, N, "BATCHED_DEQUANT_REFERENCE_CALLBACK", false, false,
                                false);
    }

    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float) {
        for (int k = k_start; k < k_end; ++k) {
            for (int m = 0; m < M; ++m) {
                store_out(m, k, 0.0f);
            }
        }
        return;
    }

    thread_local std::vector<float> dequant_row;
    dequant_row.resize(static_cast<size_t>(N));
    const auto branch_begin =
        matmul_dispatch_census_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (int k = k_start; k < k_end; ++k) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const void* row_ptr =
            reinterpret_cast<const char*>(weight_tensor->data) + static_cast<size_t>(k) * weight_row_stride;
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int i = 0; i < N; ++i) {
            const float w = dequant_row[static_cast<size_t>(i)];
            if (ud->gemma4_prefill_safe_batched && !std::isfinite(w)) {
                continue;
            }
            for (int m = 0; m < M; ++m) {
                const float x = x_rows[static_cast<size_t>(m)][i];
                if (ud->gemma4_prefill_safe_batched && !std::isfinite(x)) {
                    continue;
                }
                sums[static_cast<size_t>(m)] += x * w;
            }
        }
        for (int m = 0; m < M; ++m) {
            store_out(m, k, sums[static_cast<size_t>(m)]);
        }
    }
    maybe_log_output_partition("dequant_reference_scalar");
    record_quant_profile(ggml_is_quantized(weight_type), false);
    record_batched_dispatch_census("dequant_reference_scalar", M, branch_begin);
}

void cb_gemv_batched_custom_map3(struct ggml_tensor* dst, const struct ggml_tensor* shape,
                                 const struct ggml_tensor* input, const struct ggml_tensor* weight, int ith, int nth,
                                 void* userdata) {
    (void)shape;
    if (!dst || !input || !weight) {
        return;
    }
    struct ggml_tensor view = *dst;
    view.src[0] = const_cast<struct ggml_tensor*>(input);
    view.src[1] = const_cast<struct ggml_tensor*>(weight);
    cb_gemv_batched_custom(&view, ith, nth, userdata);
}


#ifdef DENSECORE_TEST_BUILD
uint64_t BatchedDecodeRowMajorOpsForTest() {
    return batched_decode_row_major_test_ops.load();
}
uint64_t BatchedDecodeNativeM4OpsForTest() {
    return batched_decode_native_m4_test_ops.load();
}
#endif
