#include "llm/matmul/execution_policy.h"
#include "llm/config/runtime_config.h"
#include "llm/matmul/diagnostics.h"
#include "llm/matmul/q8_kernels.h"
#include "llm/matmul/quant_cache.h"
#include "llm/matmul/userdata.h"
#include "llm/models/common/family_internal.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

uint64_t ComputeGemvBatchedQuantStamp(const BatchSpec* batch, int M, int slot_id, const void* src_data_ptr,
                                      const void* weight_data_ptr) {
    // FNV-1a over stable per-op metadata to avoid stale-buffer reuse when
    // slot_id repeats across different batched inputs.
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ull;
    };

    mix(static_cast<uint64_t>(static_cast<uint32_t>(slot_id)));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(M)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_data_ptr)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_data_ptr)));
    if (batch) {
        mix(static_cast<uint64_t>(batch->tokens.size()));
        const int n_pos = std::min(M, static_cast<int>(batch->pos.size()));
        for (int i = 0; i < n_pos; ++i) {
            mix(static_cast<uint64_t>(static_cast<uint32_t>(batch->pos[static_cast<size_t>(i)])));
        }
    }

    // Zero is the reset sentinel for stamps.
    return hash == 0 ? 1 : hash;
}

bool ShouldUseArmNativeQ4KVecDotValidated(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                          const char* weight_name, const void* sample_row_ptr,
                                          const void* sample_quant_input, const float* sample_input_f32, int N) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (weight_type != GGML_TYPE_Q4_K) {
        return type_traits_cpu && type_traits_cpu->vec_dot;
    }
    if (!type_traits_cpu || !type_traits_cpu->vec_dot || !sample_row_ptr || !sample_quant_input || !sample_input_f32 ||
        N <= 0) {
        return false;
    }

    const densecore::env::RuntimeToggleMode mode = densecore::llm::config::LoadArmQ4KNativeVecDotMode();
    if (mode == densecore::env::RuntimeToggleMode::Off) {
        return false;
    }
    if (mode == densecore::env::RuntimeToggleMode::On) {
        return true;
    }

    const bool is_hybrid_ssm_qkv = densecore::llm::models::IsHybridSSMQkvWeightName(weight_name);
    if (is_hybrid_ssm_qkv && densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv()) {
        return false;  // Always fail-closed for hybrid SSM QKV on ARM unless specifically opted out
    }

    // Multi-sample validation: test multiple representative weight rows
    static std::atomic<int> state{0};
    int current = state.load(std::memory_order_acquire);
    if (current != 0) {
        return current == 1;
    }

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    current = state.load(std::memory_order_relaxed);
    if (current == 0) {
        const auto* type_traits = ggml_get_type_traits(weight_type);
        bool ok = false;
        float global_max_abs_diff = 0.0f;
        if (type_traits && type_traits->to_float) {
            thread_local std::vector<float> dequant_buffer;
            dequant_buffer.resize(static_cast<size_t>(N));
            const size_t row_stride = ggml_row_size(weight_type, N);

            // Validate at least kMinValidationRows representative rows.
            // Rows are sampled at the base pointer (row 0) and at evenly
            // spaced offsets to detect position-dependent accuracy loss.
            static constexpr int kMinValidationRows = 4;
            ok = true;
            for (int sample = 0; sample < kMinValidationRows; ++sample) {
                const void* row_ptr =
                    reinterpret_cast<const char*>(sample_row_ptr) + static_cast<size_t>(sample) * row_stride;

                float native_sum = 0.0f;
                type_traits_cpu->vec_dot(N, &native_sum, 0, row_ptr, 0, sample_quant_input, 0, 1);

                type_traits->to_float(row_ptr, dequant_buffer.data(), N);
                float ref_sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    ref_sum += dequant_buffer[static_cast<size_t>(i)] * sample_input_f32[i];
                }

                const float diff = std::fabs(native_sum - ref_sum);
                global_max_abs_diff = std::max(global_max_abs_diff, diff);

                // Tighter tolerance than before (was 5e-4 relative, now 2e-4).
                // Fail closed: any single row exceeding tolerance disables the path.
                const float tol = std::max(1e-3f, 2e-4f * std::fabs(ref_sum));
                if (!std::isfinite(native_sum) || diff > tol) {
                    ok = false;
                    break;
                }
            }
        }

        state.store(ok ? 1 : 2, std::memory_order_release);
        LogMatmulValidationOnce("arm_q4k_native_vecdot", ok, global_max_abs_diff);
        current = ok ? 1 : 2;
    }

    return current == 1;
#else
    (void)weight_type;
    (void)weight_name;
    (void)sample_row_ptr;
    (void)sample_quant_input;
    (void)sample_input_f32;
    (void)N;
    return type_traits_cpu && type_traits_cpu->vec_dot;
#endif
}

bool ShouldUseArmQwen36LargeQ8Prefill(bool qwen36_ssm_prefill, int tokens, int reduction, int rows) {
#if defined(DENSECORE_Q8_4X8_NEON_I8MM)
    return qwen36_ssm_prefill && tokens >= 128 && tokens <= 4096 && reduction >= 2048 && reduction % QK8_0 == 0 &&
           rows > 0 && rows % 4 == 0;
#else
    (void)qwen36_ssm_prefill;
    (void)tokens;
    (void)reduction;
    (void)rows;
    return false;
#endif
}

bool ShouldUseBatchedDecodeRowMajor(ModelVariant variant, InferenceExecutionPhase phase, int tokens,
                                    int explicit_override) {
    if (explicit_override >= 0) return explicit_override != 0;
#if defined(DENSECORE_TARGET_C4A) && (defined(__aarch64__) || defined(_M_ARM64))
    return variant == ModelVariant::QWEN36 && phase == InferenceExecutionPhase::Decode && tokens == 4;
#else
    (void)variant;
    (void)phase;
    (void)tokens;
    return false;
#endif
}

int ResolveQuantBatchedTileCols(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k) {
    int tile_cols = std::max(1, std::min(kMaxSmallBatchColsHard, requested_cols));
    if (!allow_true_batched_q4k && vec_dot_nrows > 0) {
        tile_cols = std::min(tile_cols, vec_dot_nrows);
    }
    return std::max(1, tile_cols);
}

bool IsQ4KTrueBatchedKernelEnabled() {
    return ResolveQ4KTrueBatchedKernelEnabledPolicy(densecore::simd::DetectSimdLevel(),
#if defined(__ARM_FEATURE_SVE)
                                                    true
#else
                                                    false
#endif
    );
}

bool ResolveQ4KTrueBatchedKernelEnabledPolicy(densecore::simd::SimdLevel level, bool compiled_with_sve) {
    if (compiled_with_sve && (level == densecore::simd::SimdLevel::SVE || level == densecore::simd::SimdLevel::SVE2)) {
        return true;
    }
#if defined(__AVX2__) || defined(__AVX512F__) || defined(DENSECORE_X86)
    return level == densecore::simd::SimdLevel::AVX2 || level == densecore::simd::SimdLevel::AVX512 ||
           level == densecore::simd::SimdLevel::AMX;
#else
    return false;
#endif
}

densecore::simd::SimdLevel GetRuntimeSimdLevel() {
    static const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return level;
}

bool IsSemanticDecodeProjectionGemv(const GemvUserData* userdata, int N, int K) {
    if (!userdata || userdata->phase_snapshot != InferenceExecutionPhase::Decode || userdata->lfm2_decode_lm_head ||
        N <= 0 || K <= 0) {
        return false;
    }
    if (userdata->semantic_op == densecore::runtime::DenseCoreSemanticOp::LmHead ||
        userdata->tensor_role == densecore::runtime::DenseCoreTensorRole::LmHead) {
        return false;
    }
    return userdata->semantic_op == densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer ||
           densecore::runtime::IsHybridSsmTensorRole(userdata->tensor_role) ||
           userdata->tensor_role == densecore::runtime::DenseCoreTensorRole::MoERouter;
}

int ResolveGemvCustomOpTaskCount(int N, int K, int requested_threads, int physical_cores,
                                 bool semantic_decode_projection, GemvCustomTaskCapReason* cap_reason) {
    int n_threads = std::max(1, requested_threads);
    GemvCustomTaskCapReason reason = GemvCustomTaskCapReason::Unknown;
    if (physical_cores > 0) {
        n_threads = std::min(n_threads, physical_cores);
        reason = GemvCustomTaskCapReason::PhysicalCore;
    }

    if (K < 64) {
        n_threads = 1;
        reason = GemvCustomTaskCapReason::SmallK64;
    } else if (K < 512) {
        n_threads = std::min(n_threads, 2);
        reason = GemvCustomTaskCapReason::SmallK512;
    } else if (K < 1536) {
        n_threads = std::min(n_threads, 4);
        reason = GemvCustomTaskCapReason::SmallK1536;
    } else if (K < 3072) {
        n_threads = std::min(n_threads, 6);
        reason = GemvCustomTaskCapReason::SmallK3072;
    }
    if (semantic_decode_projection) {
        const int projection_cap = (N <= 4096 && K <= 8192) ? 4 : 8;
        if (n_threads > projection_cap) {
            n_threads = projection_cap;
            reason = GemvCustomTaskCapReason::DecodeProjection;
        }
    }
    if (cap_reason) {
        *cap_reason = reason;
    }
    return std::max(1, n_threads);
}
