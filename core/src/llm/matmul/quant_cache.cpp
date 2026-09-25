#include "llm/matmul/quant_cache.h"
#include "llm/matmul/q8_kernels.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/work_context.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>


void DenseCoreClearQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    GetMatmulWorkState(ctx).q8_gemm_packed_generation = 0;
    GetMatmulWorkState(ctx).q8_gemm_packed_tensor = nullptr;
    GetMatmulWorkState(ctx).q8_gemm_packed_source = nullptr;
    GetMatmulWorkState(ctx).q8_gemm_packed_rows = 0;
    GetMatmulWorkState(ctx).q8_gemm_packed_cols = 0;
    GetMatmulWorkState(ctx).q8_gemm_packed_bytes = 0;
    GetMatmulWorkState(ctx).q8_gemm_packed_token_pos = std::numeric_limits<int64_t>::min();
    GetMatmulWorkState(ctx).q8_gemm_packed_buffer.clear();
}

bool DenseCoreValidateQ8_0RowsTo4x8ActivationCache(const InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                   const void* source, int rows, int n, size_t packed_bytes,
                                                   int64_t token_pos) {
    return ctx && src_tensor && source && rows > 0 && n > 0 && packed_bytes > 0 &&
           GetMatmulWorkState(ctx).q8_gemm_packed_generation == GetInferenceWorkContextExecutionGeneration(ctx) &&
           GetMatmulWorkState(ctx).q8_gemm_packed_tensor == src_tensor &&
           GetMatmulWorkState(ctx).q8_gemm_packed_source == source &&
           GetMatmulWorkState(ctx).q8_gemm_packed_rows == rows && GetMatmulWorkState(ctx).q8_gemm_packed_cols == n &&
           GetMatmulWorkState(ctx).q8_gemm_packed_bytes == packed_bytes &&
           GetMatmulWorkState(ctx).q8_gemm_packed_token_pos == token_pos &&
           GetMatmulWorkState(ctx).q8_gemm_packed_buffer.size() == packed_bytes;
}

const uint8_t* GetOrFillQ8_0RowsTo4x8ActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                     const void* source, const uint8_t* q8_input_base,
                                                     size_t q8_row_stride, int rows, int n, int64_t token_pos) {
    const size_t packed_bytes = DenseCoreQ8_0RowsTo4x8PackedBytes(rows, n);
    if (!ctx || !src_tensor || !source || !q8_input_base || q8_row_stride == 0 || packed_bytes == 0) {
        return nullptr;
    }
    if (DenseCoreValidateQ8_0RowsTo4x8ActivationCache(ctx, src_tensor, source, rows, n, packed_bytes, token_pos)) {
        return GetMatmulWorkState(ctx).q8_gemm_packed_buffer.data();
    }
    if (!DenseCorePackQ8_0RowsTo4x8(q8_input_base, q8_row_stride, rows, n,
                                    GetMatmulWorkState(ctx).q8_gemm_packed_buffer)) {
        DenseCoreClearQ8_0RowsTo4x8ActivationCache(ctx);
        return nullptr;
    }
    GetMatmulWorkState(ctx).q8_gemm_packed_generation = GetInferenceWorkContextExecutionGeneration(ctx);
    GetMatmulWorkState(ctx).q8_gemm_packed_tensor = src_tensor;
    GetMatmulWorkState(ctx).q8_gemm_packed_source = source;
    GetMatmulWorkState(ctx).q8_gemm_packed_rows = rows;
    GetMatmulWorkState(ctx).q8_gemm_packed_cols = n;
    GetMatmulWorkState(ctx).q8_gemm_packed_bytes = packed_bytes;
    GetMatmulWorkState(ctx).q8_gemm_packed_token_pos = token_pos;
    return GetMatmulWorkState(ctx).q8_gemm_packed_buffer.data();
}


const char* Q4KRepackedGemvRejectReasonName(int reason) {
    switch (static_cast<Q4KRepackedGemvRejectReason>(reason)) {
    case Q4KRepackedGemvRejectReason::None: return "none";
    case Q4KRepackedGemvRejectReason::EnvOff: return "env_off";
    case Q4KRepackedGemvRejectReason::UnsupportedIsa: return "unsupported_isa";
    case Q4KRepackedGemvRejectReason::NotDecode: return "not_decode";
    case Q4KRepackedGemvRejectReason::NotQ4K: return "not_q4k";
    case Q4KRepackedGemvRejectReason::DynamicLora: return "dynamic_lora";
    case Q4KRepackedGemvRejectReason::Shape: return "shape";
    case Q4KRepackedGemvRejectReason::MissingVecDot: return "missing_vec_dot";
    case Q4KRepackedGemvRejectReason::Cache: return "cache";
    case Q4KRepackedGemvRejectReason::RealKernelUnavailable: return "real_kernel_unavailable";
    case Q4KRepackedGemvRejectReason::CopiedExperimentDisabled: return "copied_experiment_disabled";
    case Q4KRepackedGemvRejectReason::ProbeFailed: return "probe_failed";
    case Q4KRepackedGemvRejectReason::ReferenceForced: return "reference_forced";
    case Q4KRepackedGemvRejectReason::CacheThrashing: return "cache_thrashing";
    case Q4KRepackedGemvRejectReason::CacheLimitTooSmall: return "cache_limit_too_small";
    case Q4KRepackedGemvRejectReason::WorkingSetExceedsCache: return "working_set_exceeds_cache";
    case Q4KRepackedGemvRejectReason::RepeatedRepack: return "repeated_repack";
    case Q4KRepackedGemvRejectReason::EvictionRatioHigh: return "eviction_ratio_high";
    case Q4KRepackedGemvRejectReason::RepackBytesHigh: return "repack_bytes_high";
    }
    return "unknown";
}

bool Q4KRepackedGemvRejectReasonIsCacheThrash(int reason) {
    switch (static_cast<Q4KRepackedGemvRejectReason>(reason)) {
    case Q4KRepackedGemvRejectReason::CacheThrashing:
    case Q4KRepackedGemvRejectReason::CacheLimitTooSmall:
    case Q4KRepackedGemvRejectReason::WorkingSetExceedsCache:
    case Q4KRepackedGemvRejectReason::RepeatedRepack:
    case Q4KRepackedGemvRejectReason::EvictionRatioHigh:
    case Q4KRepackedGemvRejectReason::RepackBytesHigh: return true;
    default: return false;
    }
}


const char* GemvCustomTaskCapReasonName(int reason) {
    switch (static_cast<GemvCustomTaskCapReason>(reason)) {
    case GemvCustomTaskCapReason::Unknown: return "unknown";
    case GemvCustomTaskCapReason::PhysicalCore: return "physical_cores";
    case GemvCustomTaskCapReason::PerformanceProfileConfiguredThreads: return "perf_profile_configured_threads";
    case GemvCustomTaskCapReason::SmallK64: return "small_k_lt_64";
    case GemvCustomTaskCapReason::SmallK512: return "small_k_lt_512";
    case GemvCustomTaskCapReason::SmallK1536: return "small_k_lt_1536";
    case GemvCustomTaskCapReason::SmallK3072: return "small_k_lt_3072";
    case GemvCustomTaskCapReason::DecodeProjection: return "decode_projection";
    }
    return "unknown";
}


const char* Qwen36SSMQ8PrefillAMXRejectReasonName(int reason) {
    switch (static_cast<Qwen36SSMQ8PrefillAMXRejectReason>(reason)) {
    case Qwen36SSMQ8PrefillAMXRejectReason::None: return "none";
    case Qwen36SSMQ8PrefillAMXRejectReason::EnvOff: return "env_off";
    case Qwen36SSMQ8PrefillAMXRejectReason::NotQwen36HybridSsm: return "not_qwen36_hybrid_ssm";
    case Qwen36SSMQ8PrefillAMXRejectReason::NotPrefill: return "not_prefill";
    case Qwen36SSMQ8PrefillAMXRejectReason::NotSSMProjection: return "not_ssm_projection";
    case Qwen36SSMQ8PrefillAMXRejectReason::NotQ8_0: return "not_q8_0";
    case Qwen36SSMQ8PrefillAMXRejectReason::DynamicLora: return "dynamic_lora";
    case Qwen36SSMQ8PrefillAMXRejectReason::BackendUnavailable: return "backend_unavailable";
    case Qwen36SSMQ8PrefillAMXRejectReason::AliasUnavailable: return "alias_unavailable";
    case Qwen36SSMQ8PrefillAMXRejectReason::ProbeUnavailable: return "probe_unavailable";
    case Qwen36SSMQ8PrefillAMXRejectReason::Admitted: return "admitted";
    case Qwen36SSMQ8PrefillAMXRejectReason::DecodeOriginalQ8: return "decode_original_q8";
    case Qwen36SSMQ8PrefillAMXRejectReason::ResidentDecodeRegressionRisk: return "resident_decode_regression_risk";
    case Qwen36SSMQ8PrefillAMXRejectReason::PhaseUnknown: return "phase_unknown";
    }
    return "unknown";
}

Qwen36SSMQ8PrefillAMXRejectReason
ResolveQwen36SSMQ8PrefillAMXReason(densecore::llm::config::Qwen36SSMQ8PrefillAMXMode mode,
                                   InferenceExecutionPhase phase, bool lora_active) {
    if (mode == densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off) {
        return Qwen36SSMQ8PrefillAMXRejectReason::EnvOff;
    }
    if (phase == InferenceExecutionPhase::Unknown) {
        return Qwen36SSMQ8PrefillAMXRejectReason::PhaseUnknown;
    }
    if (phase != InferenceExecutionPhase::Prefill) {
        return Qwen36SSMQ8PrefillAMXRejectReason::NotPrefill;
    }
    if (lora_active) {
        return Qwen36SSMQ8PrefillAMXRejectReason::DynamicLora;
    }
    return Qwen36SSMQ8PrefillAMXRejectReason::None;
}

void RecordQwen36SSMQ8PrefillAMXReject(InferenceWorkContext* ctx, Qwen36SSMQ8PrefillAMXRejectReason reason) {
    if (!ctx || reason == Qwen36SSMQ8PrefillAMXRejectReason::None) {
        return;
    }
    (*GetInferenceWorkContextProfile(ctx))
        .qwen36_ssm_q8_prefill_amx_last_reject_reason.store(static_cast<int>(reason), std::memory_order_relaxed);
}

void RecordQwen36SSMQ8PrefillAMXUsed(InferenceWorkContext* ctx, int projection_kind) {
    if (!ctx) {
        return;
    }
    (*GetInferenceWorkContextProfile(ctx)).qwen36_ssm_q8_prefill_amx_used.store(1, std::memory_order_relaxed);
    (*GetInferenceWorkContextProfile(ctx)).qwen36_ssm_q8_prefill_amx_used_ops.fetch_add(1, std::memory_order_relaxed);
    switch (projection_kind) {
    case 1:
        (*GetInferenceWorkContextProfile(ctx))
            .qwen36_ssm_q8_prefill_amx_qkv_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case 2:
        (*GetInferenceWorkContextProfile(ctx))
            .qwen36_ssm_q8_prefill_amx_gate_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case 3:
        (*GetInferenceWorkContextProfile(ctx))
            .qwen36_ssm_q8_prefill_amx_out_count.fetch_add(1, std::memory_order_relaxed);
        break;
    default: break;
    }
}

bool Q4KRepackedGemvEnabled(const densecore::llm::config::FastPathRuntimeConfig& config,
                            Q4KRepackedGemvRejectReason* reject_reason) {
    if (config.q4k_repacked_gemv == densecore::env::RuntimeToggleMode::Off) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::EnvOff;
        return false;
    }
    const bool supported = densecore::kernels::Q4KRepackedGemvIsaSupported();
    if (!supported) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::UnsupportedIsa;
        return false;
    }
    if (!densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
        if (reject_reason) *reject_reason = Q4KRepackedGemvRejectReason::RealKernelUnavailable;
        return false;
    }
    return true;
}

void RecordQ4KRepackedGemvCacheLookup(InferenceWorkContext* work_ctx,
                                      const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup) {
    if (!work_ctx) {
        return;
    }
    auto& profile = (*GetInferenceWorkContextProfile(work_ctx));
    if (lookup.waited) {
        profile.q4k_repacked_gemv_cache_waited_hits.fetch_add(1, std::memory_order_relaxed);
    } else if (lookup.cache_hit) {
        profile.q4k_repacked_gemv_cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        profile.q4k_repacked_gemv_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    if (lookup.cache_evictions != 0) {
        profile.q4k_repacked_gemv_cache_evictions.fetch_add(lookup.cache_evictions, std::memory_order_relaxed);
    }
    if (lookup.cache_evicted_bytes != 0) {
        profile.q4k_repacked_gemv_cache_evicted_bytes.fetch_add(lookup.cache_evicted_bytes, std::memory_order_relaxed);
    }
    if (lookup.repack_bytes != 0) {
        profile.q4k_repacked_gemv_repack_bytes.fetch_add(lookup.repack_bytes, std::memory_order_relaxed);
    }
    if (lookup.resident_bytes != 0) {
        profile.q4k_repacked_gemv_resident_bytes.store(lookup.resident_bytes, std::memory_order_relaxed);
    }
    if (lookup.weight_key != 0) {
        std::lock_guard<std::mutex> lock(GetMatmulWorkState(work_ctx).q4k_repacked_gemv_request_mutex);
        auto [it, inserted] =
            GetMatmulWorkState(work_ctx).q4k_repacked_gemv_repack_counts.emplace(lookup.weight_key, 0);
        if (inserted) {
            profile.q4k_repacked_gemv_distinct_weights_seen.store(
                static_cast<uint64_t>(GetMatmulWorkState(work_ctx).q4k_repacked_gemv_repack_counts.size()),
                std::memory_order_relaxed);
        }
        if (lookup.repacked) {
            ++it->second;
            if (it->second > 1) {
                profile.q4k_repacked_gemv_repeated_repack_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

void SetQ4KRepackedGemvAutoDisable(InferenceWorkContext* work_ctx, Q4KRepackedGemvRejectReason local_reason,
                                   Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || local_reason == Q4KRepackedGemvRejectReason::None) {
        return;
    }
    const int encoded = static_cast<int>(local_reason);
    int expected = 0;
    (*GetInferenceWorkContextProfile(work_ctx))
        .q4k_repacked_gemv_primary_disable_reason.compare_exchange_strong(expected, encoded, std::memory_order_relaxed);
    (*GetInferenceWorkContextProfile(work_ctx))
        .q4k_repacked_gemv_last_reject_reason.store(encoded, std::memory_order_relaxed);
    if (reason) {
        *reason = local_reason;
    }
}

bool Q4KRepackedGemvShouldAutoDisableForLookup(InferenceWorkContext* work_ctx,
                                               const densecore::llm::config::FastPathRuntimeConfig& config,
                                               const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup,
                                               Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || !config.q4k_repacked_gemv_disable_on_thrash ||
        config.q4k_repacked_gemv != densecore::env::RuntimeToggleMode::Auto) {
        return false;
    }
    Q4KRepackedGemvRejectReason local_reason = Q4KRepackedGemvRejectReason::None;
    const uint64_t repeated_repack_count = (*GetInferenceWorkContextProfile(work_ctx))
                                               .q4k_repacked_gemv_repeated_repack_count.load(std::memory_order_relaxed);
    const uint64_t cache_evictions =
        (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed);
    const uint64_t used_ops =
        (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_used_ops.load(std::memory_order_relaxed);
    const uint64_t repack_bytes =
        (*GetInferenceWorkContextProfile(work_ctx)).q4k_repacked_gemv_repack_bytes.load(std::memory_order_relaxed);
    const uint64_t repack_threshold_bytes =
        static_cast<uint64_t>(std::max(1, config.q4k_repacked_gemv_thrash_repack_mb)) * 1024ULL * 1024ULL;
    const uint64_t cache_fraction_threshold =
        lookup.cache_limit_bytes == 0 ? 0
                                      : static_cast<uint64_t>(static_cast<double>(lookup.cache_limit_bytes) *
                                                              config.q4k_repacked_gemv_thrash_repack_cache_fraction);
    const uint64_t eviction_ratio_denominator = std::max<uint64_t>(1, used_ops);

    if (repeated_repack_count > 0) {
        local_reason = Q4KRepackedGemvRejectReason::RepeatedRepack;
    } else if (lookup.working_set_exceeds_cache) {
        local_reason = Q4KRepackedGemvRejectReason::WorkingSetExceedsCache;
    } else if (lookup.cache_limit_too_small) {
        local_reason = Q4KRepackedGemvRejectReason::CacheLimitTooSmall;
    } else if (cache_evictions > 0 && repack_bytes >= repack_threshold_bytes &&
               static_cast<double>(cache_evictions) / static_cast<double>(eviction_ratio_denominator) >
                   config.q4k_repacked_gemv_thrash_eviction_ratio) {
        local_reason = Q4KRepackedGemvRejectReason::EvictionRatioHigh;
    } else if (repack_bytes >= repack_threshold_bytes && cache_fraction_threshold > 0 &&
               repack_bytes > cache_fraction_threshold) {
        local_reason = Q4KRepackedGemvRejectReason::RepackBytesHigh;
    } else if (lookup.repacked && lookup.cache_evictions != 0 &&
               (*GetInferenceWorkContextProfile(work_ctx))
                       .q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed) > lookup.cache_evictions) {
        local_reason = Q4KRepackedGemvRejectReason::CacheThrashing;
    }
    if (local_reason == Q4KRepackedGemvRejectReason::None) {
        return false;
    }
    SetQ4KRepackedGemvAutoDisable(work_ctx, local_reason, reason);
    return true;
}


bool RunQ4KRepackedGemvProbe(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                             const void* weight_data, const void* quant_input, int64_t rows, int64_t cols) {
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!packed || !type_traits_cpu || !type_traits_cpu->vec_dot || !weight_data || !quant_input || rows <= 0 ||
        cols <= 0) {
        return false;
    }
    const int tile_count = static_cast<int>(rows / 8);
    if (tile_count <= 0) {
        return false;
    }
    std::array<int, 3> sample_tiles{0, tile_count / 2, tile_count - 1};
    std::vector<float> probe(static_cast<size_t>(rows), 0.0f);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const auto* weight_base = static_cast<const uint8_t*>(weight_data);
    for (const int tile : sample_tiles) {
        const int clamped_tile = std::clamp(tile, 0, tile_count - 1);
        if (!densecore::kernels::RunQ4KRepackedGemv(packed, static_cast<const uint8_t*>(quant_input), q8_row_bytes,
                                                    probe.data(), rows, clamped_tile, clamped_tile + 1)) {
            return false;
        }
        const int row_begin = clamped_tile * 8;
        const int row_end = std::min(row_begin + 8, static_cast<int>(rows));
        for (int row = row_begin; row < row_end; ++row) {
            const void* row_ptr = weight_base + static_cast<size_t>(row) * q4_row_bytes;
            float reference = 0.0f;
            type_traits_cpu->vec_dot(cols, &reference, 0, row_ptr, 0, quant_input, 0, 1);
            const float diff = std::fabs(reference - probe[static_cast<size_t>(row)]);
            const float tol = std::max(1.0e-3f, 1.0e-3f * std::fabs(reference));
            if (!std::isfinite(reference) || !std::isfinite(probe[static_cast<size_t>(row)]) || diff > tol) {
                return false;
            }
        }
    }
    return true;
}

bool Q4KRepackedGemvProbePassed(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
                                const void* weight_data, const void* quant_input, int64_t rows, int64_t cols) {
    if (!packed) {
        return false;
    }
    static std::mutex mutex;
    static std::unordered_map<Q4KRepackedGemvProbeKey, std::shared_ptr<Q4KRepackedGemvProbeEntry>,
                              Q4KRepackedGemvProbeKeyHash>
        decisions;
    static std::vector<Q4KRepackedGemvProbeKey> insertion_order;
    constexpr size_t max_entries = 4096;
    const Q4KRepackedGemvProbeKey key{
        weight_data,
        rows,
        cols,
        densecore::kernels::Q4KRepackedGemvWeightFingerprint(weight_data, rows, cols),
    };

    std::shared_ptr<Q4KRepackedGemvProbeEntry> entry;
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto found = decisions.find(key);
        if (found != decisions.end()) {
            entry = found->second;
            if (entry->done) {
                return entry->passed;
            }
            entry->cv.wait(lock, [&]() { return entry->done; });
            return entry->passed;
        }
        entry = std::make_shared<Q4KRepackedGemvProbeEntry>();
        entry->running = true;
        decisions.emplace(key, entry);
        insertion_order.push_back(key);
        while (decisions.size() > max_entries && !insertion_order.empty()) {
            auto victim = decisions.find(insertion_order.front());
            if (victim != decisions.end() && victim->second && victim->second->running) {
                break;
            }
            if (victim != decisions.end()) {
                decisions.erase(victim);
            }
            insertion_order.erase(insertion_order.begin());
        }
    }

    const bool passed = RunQ4KRepackedGemvProbe(packed, weight_data, quant_input, rows, cols);
    {
        std::lock_guard<std::mutex> lock(mutex);
        entry->passed = passed;
        entry->done = true;
        entry->running = false;
        entry->cv.notify_all();
    }
    return passed;
}

void RecordQActCacheHit(InferenceWorkContext* ctx, size_t bytes) {
    if (!ctx) return;
    (*GetInferenceWorkContextProfile(ctx)).qact_cache_hits.fetch_add(1, std::memory_order_relaxed);
    (*GetInferenceWorkContextProfile(ctx)).qact_cache_reused_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void RecordQActCacheMiss(InferenceWorkContext* ctx) {
    if (!ctx) return;
    (*GetInferenceWorkContextProfile(ctx)).qact_cache_misses.fetch_add(1, std::memory_order_relaxed);
}

bool QuantizedActivationCacheEnabled(const densecore::llm::config::FastPathRuntimeConfig& config) {
    if (config.qact_cache == densecore::env::RuntimeToggleMode::Off) {
        return false;
    }
    return config.qact_cache == densecore::env::RuntimeToggleMode::On ||
           config.qact_cache == densecore::env::RuntimeToggleMode::Auto;
}

bool QuantizedActivationKeyMatches(uint64_t generation, const ggml_tensor* tensor, const void* source, int64_t len,
                                   ggml_type type, size_t bytes, int slot_id, int64_t token_pos, size_t buffer_size,
                                   const InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                   const void* expected_source, int64_t expected_len, ggml_type expected_type,
                                   size_t expected_bytes, int expected_slot_id, int64_t expected_token_pos) {
    return ctx && generation == GetInferenceWorkContextExecutionGeneration(ctx) && tensor == src_tensor &&
           source == expected_source && len == expected_len && type == expected_type && bytes == expected_bytes &&
           slot_id == expected_slot_id && token_pos == expected_token_pos && buffer_size == expected_bytes;
}

const uint8_t* FindQuantizedActivationCacheEntry(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                 const void* source, int64_t len, ggml_type quant_type,
                                                 size_t quant_bytes, int slot_id, int64_t token_pos) {
    if (!ctx) {
        return nullptr;
    }
    if (QuantizedActivationKeyMatches(GetMatmulWorkState(ctx).qact_generation, GetMatmulWorkState(ctx).qact_tensor,
                                      GetMatmulWorkState(ctx).qact_source, GetMatmulWorkState(ctx).qact_len,
                                      GetMatmulWorkState(ctx).qact_type, GetMatmulWorkState(ctx).qact_bytes,
                                      GetMatmulWorkState(ctx).qact_slot_id, GetMatmulWorkState(ctx).qact_token_pos,
                                      GetMatmulWorkState(ctx).qact_buffer.size(), ctx, src_tensor, source, len,
                                      quant_type, quant_bytes, slot_id, token_pos)) {
        RecordQActCacheHit(ctx, quant_bytes);
        return GetMatmulWorkState(ctx).qact_buffer.data();
    }
    for (auto& slot : GetMatmulWorkState(ctx).qact_extra_slots) {
        if (QuantizedActivationKeyMatches(slot.generation, slot.tensor, slot.source, slot.len, slot.type, slot.bytes,
                                          slot.slot_id, slot.token_pos, slot.buffer.size(), ctx, src_tensor, source,
                                          len, quant_type, quant_bytes, slot_id, token_pos)) {
            RecordQActCacheHit(ctx, quant_bytes);
            return slot.buffer.data();
        }
    }
    return nullptr;
}

QuantizedActivationCacheEntry* SelectExtraQuantizedActivationSlot(InferenceWorkContext* ctx) {
    if (!ctx) {
        return nullptr;
    }
    for (auto& slot : GetMatmulWorkState(ctx).qact_extra_slots) {
        if (slot.generation != GetInferenceWorkContextExecutionGeneration(ctx) || slot.buffer.empty()) {
            return &slot;
        }
    }
    int idx = GetMatmulWorkState(ctx).qact_extra_next_slot;
    if (idx < 0 || idx >= kExtraQuantizedActivationCacheSlots) {
        idx = 0;
    }
    GetMatmulWorkState(ctx).qact_extra_next_slot = (idx + 1) % kExtraQuantizedActivationCacheSlots;
    return &GetMatmulWorkState(ctx).qact_extra_slots[static_cast<size_t>(idx)];
}

bool PrimaryQuantizedActivationSlotAvailable(const InferenceWorkContext* ctx) {
    return !ctx || GetMatmulWorkState(ctx).qact_generation != GetInferenceWorkContextExecutionGeneration(ctx) ||
           GetMatmulWorkState(ctx).qact_buffer.empty();
}

void StoreExtraQuantizedActivationMetadata(QuantizedActivationCacheEntry* slot, InferenceWorkContext* ctx,
                                           const ggml_tensor* src_tensor, const void* source, int64_t len,
                                           ggml_type quant_type, size_t quant_bytes, int slot_id, int64_t token_pos) {
    if (!slot || !ctx) {
        return;
    }
    slot->generation = GetInferenceWorkContextExecutionGeneration(ctx);
    slot->tensor = src_tensor;
    slot->source = source;
    slot->len = len;
    slot->type = quant_type;
    slot->bytes = quant_bytes;
    slot->slot_id = slot_id;
    slot->token_pos = token_pos;
}

const uint8_t* GetOrFillQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                 const void* source, const float* x_f32, int64_t len,
                                                 ggml_type quant_type, size_t quant_bytes, int slot_id,
                                                 int64_t token_pos, const ggml_type_traits_cpu* input_type_traits) {
    if (!ctx || !src_tensor || !source || !x_f32 || len <= 0 || quant_type == GGML_TYPE_F32 || quant_bytes == 0 ||
        !input_type_traits || !input_type_traits->from_float) {
        return nullptr;
    }
#ifndef NDEBUG
    const bool data_pointer_matches_different_tensor = GetMatmulWorkState(ctx).qact_source == source &&
                                                       GetMatmulWorkState(ctx).qact_tensor &&
                                                       GetMatmulWorkState(ctx).qact_tensor != src_tensor;
    if (data_pointer_matches_different_tensor) {
        static const bool strict_qact_cache_assert = []() {
            const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_QACT_CACHE_ASSERT");
            return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        }();
        if (strict_qact_cache_assert) {
            assert(GetMatmulWorkState(ctx).qact_tensor == src_tensor &&
                   "qact cache data pointer reused by a different tensor");
        }
    }
#endif
    if (const uint8_t* cached = FindQuantizedActivationCacheEntry(ctx, src_tensor, source, len, quant_type, quant_bytes,
                                                                  slot_id, token_pos)) {
        return cached;
    }

    if (PrimaryQuantizedActivationSlotAvailable(ctx)) {
        GetMatmulWorkState(ctx).qact_buffer.resize(quant_bytes);
        input_type_traits->from_float(x_f32, GetMatmulWorkState(ctx).qact_buffer.data(), len);
        GetMatmulWorkState(ctx).qact_tensor = src_tensor;
        GetMatmulWorkState(ctx).qact_generation = GetInferenceWorkContextExecutionGeneration(ctx);
        GetMatmulWorkState(ctx).qact_source = source;
        GetMatmulWorkState(ctx).qact_len = len;
        GetMatmulWorkState(ctx).qact_type = quant_type;
        GetMatmulWorkState(ctx).qact_bytes = quant_bytes;
        GetMatmulWorkState(ctx).qact_slot_id = slot_id;
        GetMatmulWorkState(ctx).qact_token_pos = token_pos;
        RecordQActCacheMiss(ctx);
        return GetMatmulWorkState(ctx).qact_buffer.data();
    }

    QuantizedActivationCacheEntry* slot = SelectExtraQuantizedActivationSlot(ctx);
    if (!slot) {
        return nullptr;
    }
    slot->buffer.resize(quant_bytes);
    input_type_traits->from_float(x_f32, slot->buffer.data(), len);
    StoreExtraQuantizedActivationMetadata(slot, ctx, src_tensor, source, len, quant_type, quant_bytes, slot_id,
                                          token_pos);
    RecordQActCacheMiss(ctx);
    return slot->buffer.data();
}

const uint8_t* GetOrFillBatchedQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                        const void* source, const std::vector<const float*>& x_rows,
                                                        int M, int N, ggml_type quant_type, size_t quant_row_stride,
                                                        size_t quant_bytes, int64_t token_pos,
                                                        const ggml_type_traits_cpu* input_type_traits) {
    if (!ctx || !src_tensor || !source || M <= 0 || N <= 0 || quant_type == GGML_TYPE_F32 || quant_row_stride == 0 ||
        quant_bytes == 0 || !input_type_traits || !input_type_traits->from_float ||
        x_rows.size() < static_cast<size_t>(M)) {
        return nullptr;
    }
    for (int m = 0; m < M; ++m) {
        if (!x_rows[static_cast<size_t>(m)]) {
            return nullptr;
        }
    }
    const int64_t len = static_cast<int64_t>(M) * static_cast<int64_t>(N);
    if (const uint8_t* cached =
            FindQuantizedActivationCacheEntry(ctx, src_tensor, source, len, quant_type, quant_bytes, -1, token_pos)) {
        return cached;
    }

    if (PrimaryQuantizedActivationSlotAvailable(ctx)) {
        GetMatmulWorkState(ctx).qact_buffer.resize(quant_bytes);
        for (int m = 0; m < M; ++m) {
            uint8_t* q_ptr = GetMatmulWorkState(ctx).qact_buffer.data() + static_cast<size_t>(m) * quant_row_stride;
            input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
        }
        GetMatmulWorkState(ctx).qact_tensor = src_tensor;
        GetMatmulWorkState(ctx).qact_generation = GetInferenceWorkContextExecutionGeneration(ctx);
        GetMatmulWorkState(ctx).qact_source = source;
        GetMatmulWorkState(ctx).qact_len = len;
        GetMatmulWorkState(ctx).qact_type = quant_type;
        GetMatmulWorkState(ctx).qact_bytes = quant_bytes;
        GetMatmulWorkState(ctx).qact_slot_id = -1;
        GetMatmulWorkState(ctx).qact_token_pos = token_pos;
        RecordQActCacheMiss(ctx);
        return GetMatmulWorkState(ctx).qact_buffer.data();
    }

    QuantizedActivationCacheEntry* slot = SelectExtraQuantizedActivationSlot(ctx);
    if (!slot) {
        return nullptr;
    }
    slot->buffer.resize(quant_bytes);
    for (int m = 0; m < M; ++m) {
        uint8_t* q_ptr = slot->buffer.data() + static_cast<size_t>(m) * quant_row_stride;
        input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
    }
    StoreExtraQuantizedActivationMetadata(slot, ctx, src_tensor, source, len, quant_type, quant_bytes, -1, token_pos);
    RecordQActCacheMiss(ctx);
    return slot->buffer.data();
}


const char* Qwen36PrefillQ4KBatchedRejectReasonName(int reason) {
    switch (static_cast<Qwen36PrefillQ4KBatchedRejectReason>(reason)) {
    case Qwen36PrefillQ4KBatchedRejectReason::None: return "none";
    case Qwen36PrefillQ4KBatchedRejectReason::EnvOff: return "env_off";
    case Qwen36PrefillQ4KBatchedRejectReason::LoraActive: return "lora_active";
    case Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape: return "unsupported_shape";
    case Qwen36PrefillQ4KBatchedRejectReason::MissingVecDot: return "missing_vec_dot";
    case Qwen36PrefillQ4KBatchedRejectReason::KernelUnavailable: return "kernel_unavailable";
    case Qwen36PrefillQ4KBatchedRejectReason::ProbeMismatch: return "probe_mismatch";
    case Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError: return "probe_internal_error";
    case Qwen36PrefillQ4KBatchedRejectReason::Admitted: return "admitted";
    case Qwen36PrefillQ4KBatchedRejectReason::RejectedCached: return "rejected_cached";
    case Qwen36PrefillQ4KBatchedRejectReason::NotQ4K: return "not_q4k";
    }
    return "unknown";
}

Qwen36PrefillQ4KBatchedRejectReason
ResolveQwen36PrefillQ4KBatchedReason(bool relevant, bool mode_off, bool lora_active, bool weight_is_q4k,
                                     bool shape_supported, bool kernel_available, bool has_vec_dot,
                                     bool candidate_ready, bool mode_on, bool mode_probe,
                                     Qwen36Q4KBatchedAdmissionState admission_state) {
    if (!relevant) {
        return Qwen36PrefillQ4KBatchedRejectReason::None;
    }
    if (mode_off) {
        return Qwen36PrefillQ4KBatchedRejectReason::EnvOff;
    }
    if (lora_active) {
        return Qwen36PrefillQ4KBatchedRejectReason::LoraActive;
    }
    if (!weight_is_q4k) {
        return Qwen36PrefillQ4KBatchedRejectReason::NotQ4K;
    }
    if (!shape_supported) {
        return Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape;
    }
    if (!kernel_available) {
        return Qwen36PrefillQ4KBatchedRejectReason::KernelUnavailable;
    }
    if (!has_vec_dot) {
        return Qwen36PrefillQ4KBatchedRejectReason::MissingVecDot;
    }
    if (!candidate_ready) {
        return Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape;
    }
    if (mode_probe && admission_state == Qwen36Q4KBatchedAdmissionState::Reject) {
        return Qwen36PrefillQ4KBatchedRejectReason::RejectedCached;
    }
    if (mode_on || (mode_probe && admission_state == Qwen36Q4KBatchedAdmissionState::Pass)) {
        return Qwen36PrefillQ4KBatchedRejectReason::Admitted;
    }
    return Qwen36PrefillQ4KBatchedRejectReason::None;
}


std::mutex& Qwen36Q4KBatchedAdmissionMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue>& Qwen36Q4KBatchedAdmissionMap() {
    static std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue> map;
    return map;
}

std::atomic<int>& Qwen36Q4KBatchedProbeForceFailThreadForTest() {
    static std::atomic<int> value{-1};
    return value;
}

uint64_t HashQwen36Q4KBatchedAdmissionKey(const TransformerModel* model, const ggml_tensor* weight,
                                          const ggml_tensor* input, int M, int N, int K) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(reinterpret_cast<uintptr_t>(model));
    mix(model ? static_cast<uint64_t>(model->variant) : 0);
    mix(model ? static_cast<uint64_t>(model->arch) : 0);
    mix(reinterpret_cast<uintptr_t>(weight));
    mix(reinterpret_cast<uintptr_t>(weight ? weight->data : nullptr));
    mix(weight ? static_cast<uint64_t>(weight->type) : 0);
    mix(input ? static_cast<uint64_t>(input->type) : 0);
    mix(static_cast<uint64_t>(M));
    mix(static_cast<uint64_t>(N));
    mix(static_cast<uint64_t>(K));
    if (weight && weight->name[0]) {
        for (const char* p = weight->name; *p; ++p) {
            mix(static_cast<unsigned char>(*p));
        }
    }
    return h;
}

Qwen36Q4KBatchedAdmissionValue LookupQwen36Q4KBatchedAdmission(uint64_t key) {
    std::lock_guard<std::mutex> lock(Qwen36Q4KBatchedAdmissionMutex());
    auto& map = Qwen36Q4KBatchedAdmissionMap();
    auto it = map.find(key);
    return it == map.end() ? Qwen36Q4KBatchedAdmissionValue{} : it->second;
}

void StoreQwen36Q4KBatchedAdmission(uint64_t key, Qwen36Q4KBatchedAdmissionState state, float max_abs_error,
                                    Qwen36PrefillQ4KBatchedRejectReason reason) {
    std::lock_guard<std::mutex> lock(Qwen36Q4KBatchedAdmissionMutex());
    auto& value = Qwen36Q4KBatchedAdmissionMap()[key];
    if (value.state == Qwen36Q4KBatchedAdmissionState::Reject && state != Qwen36Q4KBatchedAdmissionState::Reject) {
        return;
    }
    value.state = state;
    value.max_abs_error = max_abs_error;
    value.reject_reason = reason;
}

void RecordQwen36Q4KBatchedProbeResult(InferenceWorkContext* ctx, bool pass, float max_abs_error,
                                       Qwen36PrefillQ4KBatchedRejectReason reason) {
    if (!ctx) return;
    (*GetInferenceWorkContextProfile(ctx))
        .qwen36_prefill_q4k_batched_probe_pass.store(pass ? 1 : 0, std::memory_order_relaxed);
    uint32_t bits = 0;
    std::memcpy(&bits, &max_abs_error, sizeof(float));
    (*GetInferenceWorkContextProfile(ctx))
        .qwen36_prefill_q4k_batched_max_abs_error_bits.store(bits, std::memory_order_relaxed);
    (*GetInferenceWorkContextProfile(ctx))
        .qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason), std::memory_order_relaxed);
}

void DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(uint64_t key, float max_abs_error,
                                                        Qwen36PrefillQ4KBatchedRejectReason reason,
                                                        InferenceWorkContext* ctx) {
    if (!key) {
        return;
    }
    StoreQwen36Q4KBatchedAdmission(key, Qwen36Q4KBatchedAdmissionState::Reject, max_abs_error, reason);
    if (ctx) {
        (*GetInferenceWorkContextProfile(ctx))
            .qwen36_prefill_q4k_admission_downgraded.fetch_add(1, std::memory_order_relaxed);
        (*GetInferenceWorkContextProfile(ctx))
            .qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason), std::memory_order_relaxed);
    }
}

void AtomicMaxFloatBits(std::atomic<uint32_t>& target, float value) {
    if (!std::isfinite(value) || value < 0.0f) {
        value = std::numeric_limits<float>::infinity();
    }
    uint32_t desired = 0;
    std::memcpy(&desired, &value, sizeof(float));
    uint32_t current = target.load(std::memory_order_relaxed);
    float current_value = 0.0f;
    std::memcpy(&current_value, &current, sizeof(float));
    while (value > current_value &&
           !target.compare_exchange_weak(current, desired, std::memory_order_relaxed, std::memory_order_relaxed)) {
        std::memcpy(&current_value, &current, sizeof(float));
    }
}

bool IsGemma4SharedDenseFfnWeightName(const char* weight_name) {
    if (!weight_name) {
        return false;
    }
    return std::strstr(weight_name, ".ffn_gate.weight") || std::strstr(weight_name, ".ffn_up.weight") ||
           std::strstr(weight_name, ".ffn_down.weight");
}

bool IsQ8RepackedGemvEnabled() {
    static const bool enabled = []() -> bool {
#if defined(__aarch64__) || defined(_M_ARM64)
        return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
        return false;
#endif
    }();
    return enabled;
}

std::shared_ptr<Q8RepackedGemvWeight> GetOrCreateQ8RepackedGemvWeight(const void* weight_data, int64_t rows,
                                                                      int64_t cols, bool force_enable) {
    if (!weight_data || rows <= 0 || cols <= 0 || (rows % 4) != 0 || (cols % QK8_0) != 0 ||
        (!force_enable && !IsQ8RepackedGemvEnabled())) {
        return nullptr;
    }
    const Q8RepackedGemvKey key{weight_data, rows, cols};
    thread_local Q8RepackedGemvKey tls_key{};
    thread_local std::shared_ptr<Q8RepackedGemvWeight> tls_packed;
    if (tls_packed && tls_key == key) {
        return tls_packed;
    }
    static std::mutex mutex;
    static std::unordered_map<Q8RepackedGemvKey, std::shared_ptr<Q8RepackedGemvWeight>, Q8RepackedGemvKeyHash> cache;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        tls_key = key;
        tls_packed = it->second;
        return tls_packed;
    }

    const size_t src_bytes = static_cast<size_t>(rows) * ggml_row_size(GGML_TYPE_Q8_0, cols);
    const size_t block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const size_t dst_bytes = static_cast<size_t>(rows / 4) * static_cast<size_t>(cols / QK8_0) * block_bytes;
    auto packed = std::make_shared<Q8RepackedGemvWeight>();
    packed->rows = rows;
    packed->cols = cols;
    packed->blocks_per_row = cols / QK8_0;
    packed->bytes = dst_bytes;
    packed->data.resize(dst_bytes);
    if (ggml_repack_q8_0_4x8(weight_data, src_bytes, rows, cols, packed->data.data(), packed->data.size()) != 0) {
        return nullptr;
    }

    auto [insert_it, inserted] = cache.emplace(key, packed);
    tls_key = key;
    tls_packed = inserted ? packed : insert_it->second;
    return tls_packed;
}
