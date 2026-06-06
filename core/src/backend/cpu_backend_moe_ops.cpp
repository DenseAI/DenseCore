#include "backend/cpu_backend_internal.h"
#include "backend/cpu_backend_moe_forward_plan.h"
#include "backend/cpu_backend_moe_projection.h"
#include "ggml-cpu.h"  // For ggml_get_type_traits_cpu (vec_dot)
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/kernel_caps.h"
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
#include <stdexcept>
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

extern "C" void ggml_gemm_q5_K_8x8_q8_K(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);

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

constexpr int64_t kMoEQuantizedProjectionMaxBatch = 256;
constexpr int kMoEQ4KRawBatchedTileM = 8;

inline float GeluTanhApprox(float x);
const std::array<float, 1 << 16>& GetGeluF16LookupTable();
bool ExpertHasGgmlQuantizedWeights(const CpuBackend::ExpertWeights& expert);
bool IsScalarScaleSidecar(const ggml_tensor* scale_tensor);
void LogMoEMatmulPath(const char* path, int M, int K, int N, int group_size, bool allow_parallel);

#include "backend/cpu_backend_moe_raw_batched.inl"

bool CanUseMoEQ4KRawBatchedScalar() {
    return true;
}

bool CanUseKQuantRowPairVecDotFastPath() {
    // Phase 0: single source of truth in kernels/kernel_caps.h.
    return densecore::kernels::KQuantVecDotRowPairSupported();
}

bool CanUseQ4KRowPairVecDotFastPath() {
    return CanUseKQuantRowPairVecDotFastPath();
}

bool IsKQuantRowPairGatedProjectionType(ggml_type weight_type, ggml_type input_type) {
    return (weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q5_K || weight_type == GGML_TYPE_Q5_1) &&
           (input_type == GGML_TYPE_Q8_K || input_type == GGML_TYPE_Q8_1);
}

bool CanUseQ4KRepackedMoEGemvFastPath() {
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable();
}

bool CanUseQ4KRepackedMoEGEGLUFastPath() {
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable();
}

bool CanUseQ4KRepackedMoEPrefillFastPath() {
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable();
}

bool Gemma4MoEPrefillQuantBatchX86Supported() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return densecore::simd::HasX86Avx2OrBetter(level) && ggml_cpu_has_avx2();
#endif
}

bool IsGemma4MoEPrefillQuantBatchEnabled(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4 || model->hparams.n_experts <= 0) {
        return false;
    }
    return Gemma4MoEPrefillQuantBatchX86Supported();
}

bool IsGemma4MoEPrefillQuantBatchForcedOn(const TransformerModel* model) {
    (void)model;
    return false;
}

bool CanUseGgmlQuantizedMoEPrefillBatch(const CpuBackend::ExpertWeights& exp, int64_t batch, int64_t hidden_dim,
                                        int64_t intermediate_dim, bool safe_reference_mode,
                                        const char** reject_reason = nullptr) {
    auto reject = [&](const char* reason) {
        if (reject_reason) {
            *reject_reason = reason;
        }
        return false;
    };
    if (safe_reference_mode) return reject("safe_reference_mode");
    if (batch <= 0 || batch > kMoEQuantizedProjectionMaxBatch) return reject("unsupported_batch");
    if (!ExpertHasGgmlQuantizedWeights(exp)) return reject("not_ggml_quantized");
    if (exp.w2_scale_tensor && !IsScalarScaleSidecar(exp.w2_scale_tensor)) return reject("unsupported_scale_sidecar");

    const auto w1_type = static_cast<ggml_type>(exp.w1_type);
    const auto w2_type = static_cast<ggml_type>(exp.w2_type);
    const auto w3_type = static_cast<ggml_type>(exp.w3_type);

    if (!ggml_is_quantized(w1_type) || !ggml_is_quantized(w2_type)) return reject("w1_w2_not_quantized");
    if (exp.w3.ptr && !ggml_is_quantized(w3_type)) return reject("w3_not_quantized");

    const bool q4k_gate_up = exp.w1.ptr && exp.w3.ptr && w1_type == GGML_TYPE_Q4_K && w3_type == GGML_TYPE_Q4_K &&
                             hidden_dim % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && intermediate_dim % 8 == 0;

    const bool q4k_down = exp.w2.ptr && w2_type == GGML_TYPE_Q4_K &&
                          intermediate_dim % ggml_blck_size(GGML_TYPE_Q4_K) == 0 && hidden_dim % 8 == 0;

    if (!q4k_gate_up) return reject("unsupported_gate_up_shape_or_type");
    if (!q4k_down) return reject("unsupported_down_shape_or_type");
    if (reject_reason) {
        *reject_reason = "none";
    }
    return true;
}

bool CanUseQ5KRepackedMoEGemvFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return ggml_cpu_has_avx2();
#endif
}

bool CanUseQ6KRepackedMoEGemvFastPath() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

bool IsQwenA3BHybridMoEModel(const TransformerModel* model);
bool IsLFM2MoEModelForSmallDecodeParallel(const TransformerModel* model);

bool CanUseSmallDecodeQuantizedTileParallel(const TransformerModel* model) {
    return IsQwenA3BHybridMoEModel(model) || IsLFM2MoEModelForSmallDecodeParallel(model);
}

#include "backend/cpu_backend_moe_repacked.inl"

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

#include "backend/cpu_backend_moe_reference.inl"

#include "backend/cpu_backend_moe_projection_plan.inl"

#include "backend/cpu_backend_moe_quant_projection.inl"

#include "backend/cpu_backend_moe_int4_projection.inl"

void DispatchExpertFFNImpl(CpuBackend* backend, int numa_node, const Tensor& input,
                           const CpuBackend::ExpertWeights& expert, const Tensor& w1, const Tensor& w2,
                           const Tensor& w3, Tensor* output, bool allow_inner_parallel = true,
                           const MoEExecutionTraceContext* trace_ctx = nullptr,
                           QuantizedProjectionInputCache* shared_input_projection_cache = nullptr,
                           CpuBackend::MoEForwardProfile* profile = nullptr,
                           bool allow_gemma4_quant_prefill_batch = false,
                           bool force_gemma4_quant_prefill_batch = false) {
    if (!backend || !output) {
        return;
    }

    const bool debug_ffn_timing = IsMoEFFNDebugTimingEnabled();
    const auto total_begin =
        debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;
    QuantizedProjectionInputCache local_input_projection_cache;
    QuantizedProjectionInputCache local_down_projection_cache;
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
    const char* gemma4_quant_prefill_reject_reason = "none";
    const bool gemma4_quant_prefill_batch_safe =
        allow_gemma4_quant_prefill_batch &&
        CanUseGgmlQuantizedMoEPrefillBatch(expert, batch, hidden_dim, intermediate_dim, safe_reference_mode,
                                           &gemma4_quant_prefill_reject_reason);
    const bool ggml_quantized_vecdot_safe = batch == 1 || gemma4_quant_prefill_batch_safe;
    const bool force_gemma4_quant_prefill_fast_path =
        force_gemma4_quant_prefill_batch && gemma4_quant_prefill_batch_safe;
    InferenceWorkContext* gemma4_quant_prefill_ctx = GetCurrentWorkContext();

    hidden_scratch.Resize(backend, hidden_size);
    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);
    const bool profile_enabled = profile != nullptr;
    const auto w1w3_profile_begin =
        profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    MoEFusedGateUpRequest fused_gate_up_request;
    fused_gate_up_request.backend = backend;
    fused_gate_up_request.numa_node = numa_node;
    fused_gate_up_request.input = &input;
    fused_gate_up_request.dense_gate_weight = &w3;
    fused_gate_up_request.hidden = &hidden;
    fused_gate_up_request.expert = &expert;
    fused_gate_up_request.input_projection_cache = input_projection_cache;
    fused_gate_up_request.gemma4_quant_prefill_ctx = gemma4_quant_prefill_ctx;
    fused_gate_up_request.intermediate_dim = intermediate_dim;
    fused_gate_up_request.hidden_dim = hidden_dim;
    fused_gate_up_request.safe_reference_mode = safe_reference_mode;
    fused_gate_up_request.ggml_quantized_vecdot_safe = ggml_quantized_vecdot_safe;
    fused_gate_up_request.gemma4_quant_prefill_batch_safe = gemma4_quant_prefill_batch_safe;
    fused_gate_up_request.force_gemma4_quant_prefill_fast_path = force_gemma4_quant_prefill_fast_path;
    fused_gate_up_request.enable_inner_parallel = enable_inner_parallel;

    const MoEFusedGateUpPlan fused_gate_up_plan = ResolveMoEFusedGateUpPlan(fused_gate_up_request);
    const bool used_fused_gate_up = EmitMoEFusedGateUpFromPlan(fused_gate_up_request, fused_gate_up_plan);
    if (!used_fused_gate_up && fused_gate_up_plan.requires_unfused_gate_projection) {
        if (fused_gate_up_plan.force_quant_prefill_failure) {
            RecordGemma4MoEPrefillQuantBatchDecision(gemma4_quant_prefill_ctx, false, false, "gate_up_fast_path_failed",
                                                     true, false);
            throw std::runtime_error("Gemma4 MoE prefill quant batch gate/up fast path failed");
        }
        gate_scratch.Resize(backend, hidden_size);
    }

    MoEProjectionRuntimeContext projection_context;
    projection_context.backend = backend;
    projection_context.numa_node = numa_node;
    projection_context.original_input = &input;
    projection_context.input_projection_cache = input_projection_cache;
    projection_context.down_projection_cache = &local_down_projection_cache;
    projection_context.expert = &expert;
    projection_context.trace_ctx = trace_ctx;
    projection_context.gemma4_quant_prefill_ctx = gemma4_quant_prefill_ctx;
    projection_context.safe_reference_mode = safe_reference_mode;
    projection_context.ggml_quantized_vecdot_safe = ggml_quantized_vecdot_safe;
    projection_context.force_gemma4_quant_prefill_fast_path = force_gemma4_quant_prefill_fast_path;
    projection_context.gemma4_quant_prefill_batch_safe = gemma4_quant_prefill_batch_safe;
    projection_context.enable_inner_parallel = enable_inner_parallel;

    std::chrono::steady_clock::duration w1_duration{};
    std::chrono::steady_clock::duration gate_duration{};
    std::chrono::steady_clock::duration activation_duration{};
    std::chrono::steady_clock::duration w2_duration{};

    if (!used_fused_gate_up) {
        const auto w1_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        RunMoEProjectionFromContext(projection_context, '1', input, w1, expert.w1_int4, expert.w1, expert.w1_type,
                                    intermediate_dim, hidden_dim, &hidden);
        LogGemma4MoETensorStats("w1", hidden, trace_ctx);
        if (debug_ffn_timing) {
            w1_duration += (std::chrono::steady_clock::now() - w1_begin);
        }
    }

    if (!used_fused_gate_up && (w3.IsValid() || expert.w3_int4.IsValid() || expert.w3.ptr)) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        const auto gate_begin =
            debug_ffn_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        RunMoEProjectionFromContext(projection_context, '3', input, w3, expert.w3_int4, expert.w3, expert.w3_type,
                                    intermediate_dim, hidden_dim, &gate);
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
    RunMoEProjectionFromContext(projection_context, '2', hidden, w2, expert.w2_int4, expert.w2, expert.w2_type,
                                hidden_dim, intermediate_dim, output);
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
                         used_fused_gate_up ? 1 : 0, allow_inner_parallel ? 1 : 0, enable_inner_parallel ? 1 : 0, w1_ms,
                         gate_ms, activation_ms, w2_ms, total_ms);
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

size_t GetMoEExpertDequantCacheBytes(const CpuBackend::ExpertWeights& exp, bool safe_reference_mode) {
    return GetExpertMatrixDequantBytes(exp.w1_type, exp.w1_int4, static_cast<int64_t>(exp.intermediate_dim),
                                       static_cast<int64_t>(exp.hidden_dim), safe_reference_mode) +
           GetExpertMatrixDequantBytes(exp.w2_type, exp.w2_int4, static_cast<int64_t>(exp.hidden_dim),
                                       static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                       exp.w2_scale_tensor) +
           ((exp.w3.ptr != nullptr || exp.w3_int4.IsValid())
                ? GetExpertMatrixDequantBytes(exp.w3_type, exp.w3_int4, static_cast<int64_t>(exp.intermediate_dim),
                                              static_cast<int64_t>(exp.hidden_dim), safe_reference_mode)
                : 0);
}

Tensor MakeMoEWeightF32(CpuBackend* backend, void* ptr, int ggml_type_id,
                        const CpuBackend::ExpertPackedInt4Weight& int4_binding, const ggml_tensor* scale_tensor,
                        int64_t rows, int64_t cols, bool safe_reference_mode, AlignedScratch& scratch,
                        size_t* dequantized_bytes, bool* dequantized_any) {
    if (int4_binding.IsValid()) {
        if (!safe_reference_mode) {
            return Tensor();
        }
        scratch.Resize(backend, static_cast<size_t>(rows * cols));
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
    scratch.Resize(backend, static_cast<size_t>(rows * cols));
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
}

bool DequantizeMoEWeightToVector(void* ptr, int ggml_type_id,
                                 const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                 const ggml_tensor* scale_tensor, int64_t rows, int64_t cols,
                                 bool safe_reference_mode, bool materialize_f32,
                                 simd::AlignedVector<float>* dst) {
    if (!dst) return false;
    dst->clear();
    if (int4_binding.IsValid()) {
        if (!safe_reference_mode) return false;
        dst->resize(static_cast<size_t>(rows * cols));
        if (!DequantizePackedInt4ToF32(int4_binding, rows, cols, dst->data())) {
            dst->clear();
            return false;
        }
        return ApplyScaleSidecarInPlace(scale_tensor, rows, cols, dst->data(), nullptr);
    }
    if (!ptr || rows <= 0 || cols <= 0) return false;
    const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
    if (wtype == GGML_TYPE_F32 && !materialize_f32 && !scale_tensor) {
        return true;
    }
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
}

template <typename CacheEntry>
void RefreshMoEDequantCacheTensors(CacheEntry* entry, const CpuBackend::ExpertWeights& exp) {
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
}

template <typename CacheEntry>
void PopulateMoEDequantCacheEntry(CacheEntry* entry, const CpuBackend::ExpertWeights& exp, bool safe_reference_mode,
                                  bool materialize_f32) {
    if (!entry) {
        return;
    }
    DequantizeMoEWeightToVector(exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                static_cast<int64_t>(exp.intermediate_dim), static_cast<int64_t>(exp.hidden_dim),
                                safe_reference_mode, materialize_f32, &entry->w1);
    DequantizeMoEWeightToVector(exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                static_cast<int64_t>(exp.hidden_dim), static_cast<int64_t>(exp.intermediate_dim),
                                safe_reference_mode, materialize_f32, &entry->w2);
    if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
        DequantizeMoEWeightToVector(exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                    static_cast<int64_t>(exp.intermediate_dim), static_cast<int64_t>(exp.hidden_dim),
                                    safe_reference_mode, materialize_f32, &entry->w3);
    }
    RefreshMoEDequantCacheTensors(entry, exp);
}

#include "backend/cpu_backend_moe_small_decode.inl"

}  // namespace

bool RunQ5KRepackedMoEFusedSwiGLURawProjection(CpuBackend* backend, const void* gate_weight_ptr,
                                               const void* up_weight_ptr, const float* input_data,
                                               const uint8_t* qinput_data, size_t qinput_row_bytes,
                                               float* output_data, int64_t rows, int64_t cols, int64_t input_cols,
                                               int numa_node, bool allow_parallel) {
    return RunQ5KRepackedMoEFusedSwiGLURawProjectionImpl(backend, gate_weight_ptr, up_weight_ptr, input_data,
                                                         qinput_data, qinput_row_bytes, output_data, rows, cols,
                                                         input_cols, numa_node, allow_parallel);
}

bool RunQ4KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel) {
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    if (!backend || !weight_ptr || !qinput_data || !output_data || rows <= 0 || cols <= 0 || input_cols <= 0 ||
        (cols % 8) != 0 || (input_cols % ggml_blck_size(GGML_TYPE_Q4_K)) != 0) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "unsupported_shape");
        return false;
    }
    auto packed = GetOrCreateQ4KRepackedMoEWeight(weight_ptr, cols, input_cols);
    if (!packed) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "pack_failed");
        return false;
    }
    if (!RunQ4KRepackedMoEGemv(backend, packed, qinput_data, qinput_row_bytes, output_data, rows, cols, numa_node,
                               allow_parallel)) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "run_failed");
        return false;
    }
    RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/true, nullptr);
    LogMoEMatmulPath("ggml_q4k_repacked_moe_projection", static_cast<int>(rows), static_cast<int>(input_cols),
                     static_cast<int>(cols), 0, allow_parallel);
    return true;
}

bool RunQ6KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel) {
    if (!backend || !weight_ptr || !qinput_data || !output_data || rows <= 0 || cols <= 0 || input_cols <= 0 ||
        (cols % 8) != 0 || (input_cols % ggml_blck_size(GGML_TYPE_Q6_K)) != 0) {
        return false;
    }
    auto packed = GetOrCreateQ6KRepackedMoEWeight(weight_ptr, cols, input_cols);
    if (!packed) {
        return false;
    }
    if (!RunQ6KRepackedMoEGemv(backend, packed, qinput_data, qinput_row_bytes, output_data, rows, cols, numa_node,
                               allow_parallel)) {
        return false;
    }
    LogMoEMatmulPath("ggml_q6k_repacked_moe_projection", static_cast<int>(rows), static_cast<int>(input_cols),
                     static_cast<int>(cols), 0, allow_parallel);
    return true;
}

bool RunQ4KRepackedMoEFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr,
                                            const void* up_weight_ptr, const float* input_data,
                                            const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                            int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                            bool allow_parallel) {
    InferenceWorkContext* census_ctx = GetCurrentWorkContext();
    if (!backend || !gate_weight_ptr || !up_weight_ptr || !qinput_data || !output_data || rows <= 0 ||
        cols <= 0 || input_cols <= 0 || (cols % 8) != 0 ||
        (input_cols % ggml_blck_size(GGML_TYPE_Q4_K)) != 0 || (rows >= 4 && !input_data)) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "unsupported_shape");
        return false;
    }
    auto gate_packed = GetOrCreateQ4KRepackedMoEWeight(gate_weight_ptr, cols, input_cols);
    auto up_packed = GetOrCreateQ4KRepackedMoEWeight(up_weight_ptr, cols, input_cols);
    if (!gate_packed || !up_packed) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "pack_failed");
        return false;
    }
    if (!RunQ4KRepackedMoEFusedSwiGLUM4(backend, gate_packed, up_packed, input_data, qinput_data, qinput_row_bytes,
                                        output_data, rows, cols, input_cols, numa_node, allow_parallel)) {
        RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/false, "run_failed");
        return false;
    }
    RecordMoEQ4KRepackedDecision(census_ctx, /*candidate=*/true, /*used=*/true, nullptr);
    LogMoEMatmulPath(rows >= 4 ? "ggml_q4k_repacked_moe_gemm_m4_fused_swiglu"
                               : "ggml_q4k_repacked_moe_gemv_fused_swiglu",
                     static_cast<int>(rows), static_cast<int>(input_cols), static_cast<int>(cols), 0,
                     allow_parallel);
    return true;
}

bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel) {
    return RunMoEQ4KRawBatchedProjectionImpl(backend, weight_ptr, qinput_data, qinput_row_bytes, out_data, M, N, K,
                                            numa_node, allow_parallel);
}

bool RunMoEKQuantRawBatchedProjection(CpuBackend* backend, int ggml_type_id, const void* weight_ptr,
                                      const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data,
                                      int64_t M, int64_t N, int64_t K, int numa_node, bool allow_parallel) {
    return RunMoEKQuantRawBatchedProjectionImpl(backend, static_cast<ggml_type>(ggml_type_id), weight_ptr, qinput_data,
                                               qinput_row_bytes, out_data, M, N, K, numa_node, allow_parallel);
}

bool RunMoEKQuantRawBatchedFusedSwiGLU(CpuBackend* backend, int ggml_type_id, const void* gate_weight_ptr,
                                       const void* up_weight_ptr, const uint8_t* qinput_data,
                                       size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                       int numa_node, bool allow_parallel) {
    if (static_cast<ggml_type>(ggml_type_id) == GGML_TYPE_Q4_K) {
        return RunMoEQ4KRawBatchedFusedSwiGLUImpl(backend, gate_weight_ptr, up_weight_ptr, qinput_data,
                                                 qinput_row_bytes, out_data, M, N, K, numa_node, allow_parallel);
    }
    return RunMoEKQuantRawBatchedFusedSwiGLUImpl(backend, static_cast<ggml_type>(ggml_type_id), gate_weight_ptr,
                                                up_weight_ptr, qinput_data, qinput_row_bytes, out_data, M, N, K,
                                                numa_node, allow_parallel);
}

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
    if (TryExecuteMoESafeReferenceFastPath(this, model, layer_idx, batch, input_data, batch_size,
                                           static_cast<int>(hidden_dim), routing, experts, num_experts, out_data,
                                           qwen36_short_prefill_safe_reference, safe_reference_mode)) {
        return;
    }

    const bool dequant_cache_enabled =
        internal::IsMoEDequantCacheEnabled() && registry != nullptr && !arm_disable_registry_dequant_cache;
    const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
    const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();
    const auto get_or_create_dequant_cache_entry =
        [&](int expert_id, const ExpertWeights& exp, size_t cacheable_bytes, bool materialize_f32) {
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
            if (!registry || cacheable_bytes == 0 || cacheable_bytes > dequant_cache_budget) {
                return cached_entry;
            }

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
                return existing_entry;
            }

            auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
            candidate->expert_id = expert_id;
            candidate->bytes = cacheable_bytes;
            PopulateMoEDequantCacheEntry(candidate.get(), exp, safe_reference_mode, materialize_f32);

            std::lock_guard<std::mutex> lock(registry->mutex);
            auto it = registry->dequant_cache.find(expert_id);
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
                    for (auto it_cache = registry->dequant_cache.begin(); it_cache != registry->dequant_cache.end();
                         ++it_cache) {
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
            return cached_entry;
        };

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

    // Gather registry hot-expert snapshot + this call's active-expert set. The
    // returned struct owns the data; we bind the original local names to its fields
    // so the downstream small-decode logic is unchanged.
    MoESmallDecodeState small_decode_state =
        GatherMoESmallDecodeState(registry, routing, num_experts, total_assignments, small_decode_candidate);
    std::shared_ptr<moe::ExpertProfiler> profiler = std::move(small_decode_state.profiler);
    const std::array<int, kSmallDecodeMaxSnapshotExperts>& small_step_local_hot_experts =
        small_decode_state.local_hot_experts;
    const int small_step_local_hot_count = small_decode_state.local_hot_count;
    const std::array<int, kSmallDecodeMaxSnapshotExperts>& small_step_previous_batch_experts =
        small_decode_state.previous_batch_experts;
    const int small_step_previous_batch_count = small_decode_state.previous_batch_count;
    const bool small_step_snapshot_ok = small_decode_state.snapshot_ok;
    const std::vector<int>& local_hot_experts = small_decode_state.local_hot_experts_overflow;
    const std::vector<int>& previous_batch_experts = small_decode_state.previous_batch_experts_overflow;
    const std::array<int, kSmallDecodeMaxAssignments>& small_step_current_batch_experts =
        small_decode_state.current_batch_experts;
    const int small_step_current_batch_expert_count = small_decode_state.current_batch_expert_count;
    const int small_step_max_expert_batch = small_decode_state.max_expert_batch;

    const bool small_decode_ready = small_decode_candidate && !small_decode_requires_general_path &&
                                    small_step_snapshot_ok && small_step_max_expert_batch <= 1 &&
                                    small_step_current_batch_expert_count > 0;
    if (!small_decode_ready) {
        const char* reject_reason = !small_decode_candidate              ? "not_small_decode"
                                    : small_decode_requires_general_path ? "requires_general_path"
                                    : !small_step_snapshot_ok            ? "snapshot_unavailable"
                                    : small_step_max_expert_batch > 1    ? "expert_batch_gt_1"
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
            tile_par_can_use && tile_par_ep_enabled && !safe_reference_mode && batch_size == 1 &&
            small_decode_worker_cap >= 16 && total_assignments >= 2 && hidden_dim >= 1024 &&
            small_decode_intermediate_dim >= 256;
        if (use_small_decode_quantized_tile_parallel) {
            MoESmallDecodeTileParallelRequest tile_request;
            tile_request.backend = this;
            tile_request.pool = &small_decode_pool;
            tile_request.routing = &routing;
            tile_request.experts = experts;
            tile_request.profiler = profiler.get();
            tile_request.profile = profile;
            tile_request.hidden_scratch = &expert_input_scratch;
            tile_request.output_scratch = &expert_output_scratch;
            tile_request.input_data = input_data;
            tile_request.out_data = out_data;
            tile_request.batch_size = batch_size;
            tile_request.top_k = top_k;
            tile_request.total_assignments = total_assignments;
            tile_request.num_experts = num_experts;
            tile_request.worker_cap = small_decode_worker_cap;
            tile_request.hidden_dim = static_cast<int64_t>(hidden_dim);
            tile_request.intermediate_dim = small_decode_intermediate_dim;
            tile_request.has_token_indices = has_token_indices;
            if (TryExecuteMoESmallDecodeQuantizedTileParallel(tile_request)) {
                return;
            }
        }
        if (use_small_decode_expert_parallel) {
            if (IsMoEMatmulPathDebugEnabled()) {
                std::fprintf(stderr,
                             "[MOE_TILE_DIAG] FALLBACK to expert_parallel (NOT tile_parallel) "
                             "workers=%d assignments=%d\n",
                             effective_small_decode_workers, total_assignments);
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
            const size_t cacheable_bytes = GetMoEExpertDequantCacheBytes(exp, safe_reference_mode);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                cached_entry = get_or_create_dequant_cache_entry(expert_id, exp, cacheable_bytes,
                                                                 /*materialize_f32=*/false);
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
                         : MakeMoEWeightF32(this, exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                            static_cast<int64_t>(exp.intermediate_dim),
                                            static_cast<int64_t>(exp.hidden_dim), safe_reference_mode, w1_dequant,
                                            &dequantized_bytes, &dequantized_any);
                w2 = (cached_entry && (exp.w2_scale_tensor || ((!exp.w2_int4.IsValid() || safe_reference_mode ||
                                                                !CanUsePackedInt4MoEFastPath()) &&
                                                               exp.w2_type != GGML_TYPE_F32)))
                         ? cached_entry->w2_tensor
                         : MakeMoEWeightF32(this, exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                            static_cast<int64_t>(exp.hidden_dim),
                                            static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode, w2_dequant,
                                            &dequantized_bytes, &dequantized_any);
                if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                    w3 = (cached_entry &&
                          (!exp.w3_int4.IsValid() || safe_reference_mode || !CanUsePackedInt4MoEFastPath()) &&
                          exp.w3_type != GGML_TYPE_F32)
                             ? cached_entry->w3_tensor
                             : MakeMoEWeightF32(this, exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                                static_cast<int64_t>(exp.intermediate_dim),
                                                static_cast<int64_t>(exp.hidden_dim), safe_reference_mode, w3_dequant,
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

    MoEForwardExecutionPlan execution_plan = BuildMoEForwardExecutionPlan(
        reorder_map, num_experts, batch_size, total_assignments, std::max(1, reorder_pool.GetNumThreads()),
        registry != nullptr, local_hot_experts_set, previous_batch_experts, profiler);
    std::vector<MoEActiveExpertWork>& active_work = execution_plan.active_work;
    const std::vector<int>& current_batch_experts = execution_plan.current_batch_experts;
    const std::unordered_set<int>& previous_batch_set = execution_plan.previous_batch_set;

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

    const int reuse_intersection = execution_plan.reuse_intersection;
    const int reuse_union = execution_plan.reuse_union;
    const int max_expert_batch = execution_plan.max_expert_batch;
    const int local_hot_count = execution_plan.local_hot_count;
    const int worker_threads = execution_plan.worker_threads;
    const bool small_decode_step = execution_plan.small_decode_step;

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

    if (execution_plan.ordering.considered) {
        moe_stats_total_ordering_considered_.fetch_add(1, std::memory_order_relaxed);
        if (execution_plan.ordering.skipped_small_batch) {
            moe_stats_total_ordering_skipped_small_batch_.fetch_add(1, std::memory_order_relaxed);
        } else if (execution_plan.ordering.skipped_low_reuse) {
            moe_stats_total_ordering_skipped_low_reuse_.fetch_add(1, std::memory_order_relaxed);
        }
        if (execution_plan.ordering.applied) {
            moe_stats_total_ordering_numa_switches_before_.fetch_add(
                execution_plan.ordering.numa_switches_before, std::memory_order_relaxed);
            moe_stats_total_ordering_applied_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_ordering_numa_switches_after_.fetch_add(execution_plan.ordering.numa_switches_after,
                                                                    std::memory_order_relaxed);
        }
    }

    uint64_t cached_experts_this_step = 0;
    std::chrono::steady_clock::duration dequant_duration{};
    std::chrono::steady_clock::duration expert_duration{};
    std::mutex work_stats_mutex;
    const bool prefer_inner_parallel_prefill = execution_plan.prefer_inner_parallel_prefill;
    const bool parallelize_experts = execution_plan.parallelize_experts;
    if (IsMoECachePolicyDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_CACHE_POLICY] arm_disable_registry_dequant_cache=%d registry_present=%d "
                     "dequant_cache_enabled=%d parallelize_experts=%d reason=%s\n",
                     arm_disable_registry_dequant_cache ? 1 : 0, registry ? 1 : 0, dequant_cache_enabled ? 1 : 0,
                     parallelize_experts ? 1 : 0,
                     prefer_inner_parallel_prefill ? "prefill_inner_parallel" : "cache_lane_available");
    }

    auto run_active_work_range = [&](int start_idx, int end_idx, bool allow_inner_parallel) {
        uint64_t local_cached_experts = 0;
        std::chrono::steady_clock::duration local_dequant_duration{};
        std::chrono::steady_clock::duration local_expert_duration{};

        for (int raw_idx = start_idx; raw_idx < end_idx; ++raw_idx) {
            const size_t idx = static_cast<size_t>(raw_idx);
            const MoEActiveExpertWork& work = active_work[idx];
            const ExpertWeights& exp = experts[static_cast<size_t>(work.expert_id)];

            if (!small_decode_step && internal::IsMoENextExpertPrefetchEnabled() && idx + 1 < active_work.size()) {
                moe_stats_total_prefetch_candidates_.fetch_add(1, std::memory_order_relaxed);
                const MoEActiveExpertWork& next_work = active_work[idx + 1];
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
            const bool prefill_phase =
                GetCurrentExecutionPhase() == InferenceExecutionPhase::Prefill || GetCurrentWorkContext() == nullptr;
            const bool gemma4_quant_prefill_enabled =
                prefill_phase && model && model->arch_flags.is_gemma4 && IsGemma4MoEPrefillQuantBatchEnabled(model);
            const bool gemma4_quant_prefill_forced = prefill_phase && IsGemma4MoEPrefillQuantBatchForcedOn(model);
            const char* gemma4_quant_prefill_reject_reason = "none";
            const bool can_use_gemma4_quant_prefill_batch =
                gemma4_quant_prefill_enabled &&
                CanUseGgmlQuantizedMoEPrefillBatch(exp, work.count, static_cast<int64_t>(hidden_dim),
                                                   static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode,
                                                   &gemma4_quant_prefill_reject_reason);
            if (gemma4_quant_prefill_enabled && work.count > 1) {
                RecordGemma4MoEPrefillQuantBatchDecision(
                    GetCurrentWorkContext(), true, false,
                    can_use_gemma4_quant_prefill_batch ? nullptr : gemma4_quant_prefill_reject_reason, false, false);
                if (gemma4_quant_prefill_forced && !can_use_gemma4_quant_prefill_batch) {
                    throw std::runtime_error(std::string("Gemma4 MoE prefill quant batch rejected: ") +
                                             gemma4_quant_prefill_reject_reason);
                }
            }
            const bool should_try_cache =
                dequant_cache_enabled && !can_use_gemma4_quant_prefill_batch &&
                (cache_all_active_experts || work.local_hot ||
                 previous_batch_set.find(work.expert_id) != previous_batch_set.end() || work.count > 1);
            const size_t cacheable_bytes = GetMoEExpertDequantCacheBytes(exp, safe_reference_mode);
            if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
                cached_entry = get_or_create_dequant_cache_entry(work.expert_id, exp, cacheable_bytes,
                                                                 /*materialize_f32=*/true);
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
                // Avoid building dense weights when quantized dispatch can consume
                // the raw ggml rows directly.
                const bool can_use_ggml_quant_gen =
                    !safe_reference_mode && ExpertHasGgmlQuantizedWeights(exp) &&
                    (!exp.w2_scale_tensor || IsScalarScaleSidecar(exp.w2_scale_tensor)) &&
                    work.count <= kMoEQuantizedProjectionMaxBatch &&
                    (work.count == 1 || can_use_gemma4_quant_prefill_batch);
                if (!can_use_ggml_quant_gen) {
                    w1 = MakeMoEWeightF32(this, exp.w1.ptr, exp.w1_type, exp.w1_int4, nullptr,
                                          static_cast<int64_t>(exp.intermediate_dim),
                                          static_cast<int64_t>(exp.hidden_dim), safe_reference_mode, w1_dequant,
                                          &dequantized_bytes, &dequantized_any);
                    w2 = MakeMoEWeightF32(this, exp.w2.ptr, exp.w2_type, exp.w2_int4, exp.w2_scale_tensor,
                                          static_cast<int64_t>(exp.hidden_dim),
                                          static_cast<int64_t>(exp.intermediate_dim), safe_reference_mode, w2_dequant,
                                          &dequantized_bytes, &dequantized_any);
                    if (exp.w3.ptr != nullptr || exp.w3_int4.IsValid()) {
                        w3 = MakeMoEWeightF32(this, exp.w3.ptr, exp.w3_type, exp.w3_int4, nullptr,
                                              static_cast<int64_t>(exp.intermediate_dim),
                                              static_cast<int64_t>(exp.hidden_dim), safe_reference_mode, w3_dequant,
                                              &dequantized_bytes, &dequantized_any);
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
                                  allow_inner_parallel, &trace_ctx, nullptr, profile,
                                  can_use_gemma4_quant_prefill_batch, gemma4_quant_prefill_forced);
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
        if (active_work.size() > static_cast<size_t>(active_threads)) {
            std::atomic<int> next_expert{0};
            reorder_pool.ParallelFor(active_threads, [&](int, int, int) {
                for (;;) {
                    const int idx = next_expert.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= static_cast<int>(active_work.size())) {
                        break;
                    }
                    run_active_work_range(idx, std::min(idx + 1, static_cast<int>(active_work.size())), false);
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

bool ResolveLFM2SmallDecodeExpertParallelAutoEligibleForTest(int physical_cores, int simd_level) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 32;
    model.hparams.n_experts_used = 32;
    return ResolveSmallDecodeExpertParallelAutoEligible(&model, physical_cores,
                                                        static_cast<densecore::simd::SimdLevel>(simd_level));
}

int ResolveQwen36SmallDecodeExpertWorkersForTest(int top_k, int worker_cap) {
    return ResolveSmallDecodeExpertWorkers(top_k, worker_cap);
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

bool CanUseGgmlQuantizedMoEPrefillBatchForTest(const CpuBackend::ExpertWeights& exp, int64_t batch, int64_t hidden_dim,
                                               int64_t intermediate_dim, bool safe_reference_mode,
                                               const char** reject_reason) {
    return CanUseGgmlQuantizedMoEPrefillBatch(exp, batch, hidden_dim, intermediate_dim, safe_reference_mode,
                                              reject_reason);
}

}  // namespace testing

}  // namespace densecore
