// Quantized GEMV cache, probes, and admission helpers used by matmul callbacks.
struct Q8RepackedGemvWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    size_t bytes = 0;
    std::vector<uint8_t> data;
};

struct Q8RepackedGemvKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    bool operator==(const Q8RepackedGemvKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols;
    }
};

struct Q8RepackedGemvKeyHash {
    size_t operator()(const Q8RepackedGemvKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

constexpr int kQ8RepackedGemvMinOutputRows = 4096;

static void DenseCoreGemvQ8_0_4x8Q8_0Generic(int n, float* out, const void* packed_weight, const void* q8_input,
                                             int nc) {
    if (!out || !packed_weight || !q8_input || n <= 0 || (n % QK8_0) != 0 || (nc % 4) != 0) {
        return;
    }
    const int nb = n / QK8_0;
    const size_t packed_block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
    const auto* packed_base = static_cast<const uint8_t*>(packed_weight);
    const auto* input_blocks = static_cast<const block_q8_0*>(q8_input);
    const auto fp16_to_f32 = [](ggml_fp16_t value) {
        float out_value = 0.0f;
        ggml_cpu_fp16_to_fp32(&value, &out_value, 1);
        return out_value;
    };
    for (int group = 0; group < nc / 4; ++group) {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const auto* group_base = packed_base + static_cast<size_t>(group) * static_cast<size_t>(nb) * packed_block_bytes;
        for (int b = 0; b < nb; ++b) {
            const auto* block_base = group_base + static_cast<size_t>(b) * packed_block_bytes;
            const auto* scales = reinterpret_cast<const ggml_fp16_t*>(block_base);
            const auto* qs = reinterpret_cast<const int8_t*>(block_base + 4 * sizeof(ggml_fp16_t));
            const block_q8_0& x = input_blocks[b];
            const float input_scale = fp16_to_f32(x.d);
            for (int row = 0; row < 4; ++row) {
                int acc = 0;
                for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                    const int q_base = chunk * 4 * 8 + row * 8;
                    const int x_base = chunk * 8;
                    for (int i = 0; i < 8; ++i) {
                        acc += static_cast<int>(qs[q_base + i]) * static_cast<int>(x.qs[x_base + i]);
                    }
                }
                sum[row] += static_cast<float>(acc) * fp16_to_f32(scales[row]) * input_scale;
            }
        }
        for (int row = 0; row < 4; ++row) {
            out[static_cast<size_t>(group) * 4 + row] = sum[row];
        }
    }
}

struct DenseCoreQ8_0x4Block {
    ggml_fp16_t d[4];
    int8_t qs[QK8_0 * 4];
};
static_assert(sizeof(DenseCoreQ8_0x4Block) == 4 * sizeof(ggml_fp16_t) + QK8_0 * 4,
              "Q8_0x4 block layout must match ggml repack layout");

static bool DenseCoreQ8_0_4x8GemmM4FromQ8(int n, float* out, size_t out_stride_floats, const void* packed_weight,
                                          const uint8_t* q8_input_base, size_t q8_row_stride,
                                          std::vector<DenseCoreQ8_0x4Block>& qtile, int nc) {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (!out || !packed_weight || !q8_input_base || n <= 0 || (n % QK8_0) != 0 || (nc % 4) != 0 ||
        q8_row_stride < ggml_row_size(GGML_TYPE_Q8_0, n)) {
        return false;
    }
    const int nb = n / QK8_0;
    qtile.resize(static_cast<size_t>(nb));
    for (int b = 0; b < nb; ++b) {
        DenseCoreQ8_0x4Block& block = qtile[static_cast<size_t>(b)];
        for (int row = 0; row < 4; ++row) {
            const auto* src = reinterpret_cast<const block_q8_0*>(
                q8_input_base + static_cast<size_t>(row) * q8_row_stride +
                static_cast<size_t>(b) * sizeof(block_q8_0));
            block.d[row] = src->d;
            for (int group = 0; group < 4; ++group) {
                for (int lane = 0; lane < 8; ++lane) {
                    block.qs[group * 32 + row * 8 + lane] = src->qs[group * 8 + lane];
                }
            }
        }
    }

    const auto* b_ptr_base = static_cast<const DenseCoreQ8_0x4Block*>(packed_weight);
    const DenseCoreQ8_0x4Block* a_ptr_base = qtile.data();
    for (int x = 0; x < nc; x += 4) {
        const DenseCoreQ8_0x4Block* b_ptr = b_ptr_base + static_cast<size_t>(x / 4) * static_cast<size_t>(nb);
        const DenseCoreQ8_0x4Block* a_ptr = a_ptr_base;

        float32x4_t acc_f32[4];
        for (int i = 0; i < 4; ++i) {
            acc_f32[i] = vdupq_n_f32(0);
        }
        for (int b = 0; b < nb; ++b) {
            int32x4_t acc[4];
            for (int i = 0; i < 4; ++i) {
                acc[i] = vdupq_n_s32(0);
            }
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int8x16_t a01 = vld1q_s8(a_ptr->qs + chunk * 32);
                const int8x16_t a23 = vld1q_s8(a_ptr->qs + chunk * 32 + 16);
                const int8x16_t b01 = vld1q_s8(b_ptr->qs + chunk * 32);
                const int8x16_t b23 = vld1q_s8(b_ptr->qs + chunk * 32 + 16);
                acc[0] = vmmlaq_s32(acc[0], a01, b01);
                acc[1] = vmmlaq_s32(acc[1], a01, b23);
                acc[2] = vmmlaq_s32(acc[2], a23, b01);
                acc[3] = vmmlaq_s32(acc[3], a23, b23);
            }

            const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
            const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
            const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
            const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
            const float32x4_t a_d = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(a_ptr->d)));
            const float32x4_t b_d = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(b_ptr->d)));
            acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
            acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
            acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
            acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
            ++a_ptr;
            ++b_ptr;
        }
        for (int row = 0; row < 4; ++row) {
            vst1q_f32(out + static_cast<size_t>(row) * out_stride_floats + x, acc_f32[row]);
        }
    }
    return true;
#else
    (void)n;
    (void)out;
    (void)out_stride_floats;
    (void)packed_weight;
    (void)q8_input_base;
    (void)q8_row_stride;
    (void)qtile;
    (void)nc;
    return false;
#endif
}

static inline float DenseCoreQ8_0BlockDot(const block_q8_0* weight_blocks, const block_q8_0* input_blocks,
                                          int n) {
    if (!weight_blocks || !input_blocks || n <= 0 || (n % QK8_0) != 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    const int nb = n / QK8_0;
    for (int b = 0; b < nb; ++b) {
        const block_q8_0& w = weight_blocks[b];
        const block_q8_0& x = input_blocks[b];
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(w.qs + 0), vld1q_s8(x.qs + 0));
        acc = vdotq_s32(acc, vld1q_s8(w.qs + 16), vld1q_s8(x.qs + 16));
        const int dot = vaddvq_s32(acc);
#else
        int dot = 0;
        for (int i = 0; i < QK8_0; ++i) {
            dot += static_cast<int>(w.qs[i]) * static_cast<int>(x.qs[i]);
        }
#endif
        sum += static_cast<float>(dot) * ggml_fp16_to_fp32(w.d) * ggml_fp16_to_fp32(x.d);
    }
    return sum;
}
constexpr int64_t kQwen36C4AmxSmartMatmulMaxTokens = 128;

enum class Q4KRepackedGemvRejectReason : int {
    None = 0,
    EnvOff = 1,
    UnsupportedIsa = 2,
    NotDecode = 3,
    NotQ4K = 4,
    DynamicLora = 5,
    Shape = 6,
    MissingVecDot = 7,
    Cache = 8,
    RealKernelUnavailable = 9,
    CopiedExperimentDisabled = 10,
    ProbeFailed = 11,
    ReferenceForced = 12,
    CacheThrashing = 13,
    CacheLimitTooSmall = 14,
    WorkingSetExceedsCache = 15,
    RepeatedRepack = 16,
    EvictionRatioHigh = 17,
    RepackBytesHigh = 18,
};

const char* Q4KRepackedGemvRejectReasonName(int reason) {
    switch (static_cast<Q4KRepackedGemvRejectReason>(reason)) {
        case Q4KRepackedGemvRejectReason::None:
            return "none";
        case Q4KRepackedGemvRejectReason::EnvOff:
            return "env_off";
        case Q4KRepackedGemvRejectReason::UnsupportedIsa:
            return "unsupported_isa";
        case Q4KRepackedGemvRejectReason::NotDecode:
            return "not_decode";
        case Q4KRepackedGemvRejectReason::NotQ4K:
            return "not_q4k";
        case Q4KRepackedGemvRejectReason::DynamicLora:
            return "dynamic_lora";
        case Q4KRepackedGemvRejectReason::Shape:
            return "shape";
        case Q4KRepackedGemvRejectReason::MissingVecDot:
            return "missing_vec_dot";
        case Q4KRepackedGemvRejectReason::Cache:
            return "cache";
        case Q4KRepackedGemvRejectReason::RealKernelUnavailable:
            return "real_kernel_unavailable";
        case Q4KRepackedGemvRejectReason::CopiedExperimentDisabled:
            return "copied_experiment_disabled";
        case Q4KRepackedGemvRejectReason::ProbeFailed:
            return "probe_failed";
        case Q4KRepackedGemvRejectReason::ReferenceForced:
            return "reference_forced";
        case Q4KRepackedGemvRejectReason::CacheThrashing:
            return "cache_thrashing";
        case Q4KRepackedGemvRejectReason::CacheLimitTooSmall:
            return "cache_limit_too_small";
        case Q4KRepackedGemvRejectReason::WorkingSetExceedsCache:
            return "working_set_exceeds_cache";
        case Q4KRepackedGemvRejectReason::RepeatedRepack:
            return "repeated_repack";
        case Q4KRepackedGemvRejectReason::EvictionRatioHigh:
            return "eviction_ratio_high";
        case Q4KRepackedGemvRejectReason::RepackBytesHigh:
            return "repack_bytes_high";
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
        case Q4KRepackedGemvRejectReason::RepackBytesHigh:
            return true;
        default:
            return false;
    }
}

enum class GemvCustomTaskCapReason : int {
    Unknown = 0,
    PhysicalCore = 1,
    PerformanceProfileConfiguredThreads = 2,
    SmallK64 = 3,
    SmallK512 = 4,
    SmallK1536 = 5,
    SmallK3072 = 6,
    DecodeProjection = 7,
};

const char* GemvCustomTaskCapReasonName(int reason) {
    switch (static_cast<GemvCustomTaskCapReason>(reason)) {
        case GemvCustomTaskCapReason::Unknown:
            return "unknown";
        case GemvCustomTaskCapReason::PhysicalCore:
            return "physical_cores";
        case GemvCustomTaskCapReason::PerformanceProfileConfiguredThreads:
            return "perf_profile_configured_threads";
        case GemvCustomTaskCapReason::SmallK64:
            return "small_k_lt_64";
        case GemvCustomTaskCapReason::SmallK512:
            return "small_k_lt_512";
        case GemvCustomTaskCapReason::SmallK1536:
            return "small_k_lt_1536";
        case GemvCustomTaskCapReason::SmallK3072:
            return "small_k_lt_3072";
        case GemvCustomTaskCapReason::DecodeProjection:
            return "decode_projection";
    }
    return "unknown";
}

enum class Qwen36SSMQ8PrefillAMXRejectReason : int {
    None = 0,
    EnvOff = 1,
    NotQwen36HybridSsm = 2,
    NotPrefill = 3,
    NotSSMProjection = 4,
    NotQ8_0 = 5,
    DynamicLora = 6,
    BackendUnavailable = 7,
    AliasUnavailable = 8,
    ProbeUnavailable = 9,
    Admitted = 10,
    DecodeOriginalQ8 = 11,
    ResidentDecodeRegressionRisk = 12,
    PhaseUnknown = 13,
};

const char* Qwen36SSMQ8PrefillAMXRejectReasonName(int reason) {
    switch (static_cast<Qwen36SSMQ8PrefillAMXRejectReason>(reason)) {
        case Qwen36SSMQ8PrefillAMXRejectReason::None:
            return "none";
        case Qwen36SSMQ8PrefillAMXRejectReason::EnvOff:
            return "env_off";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotQwen36HybridSsm:
            return "not_qwen36_hybrid_ssm";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotPrefill:
            return "not_prefill";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotSSMProjection:
            return "not_ssm_projection";
        case Qwen36SSMQ8PrefillAMXRejectReason::NotQ8_0:
            return "not_q8_0";
        case Qwen36SSMQ8PrefillAMXRejectReason::DynamicLora:
            return "dynamic_lora";
        case Qwen36SSMQ8PrefillAMXRejectReason::BackendUnavailable:
            return "backend_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::AliasUnavailable:
            return "alias_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::ProbeUnavailable:
            return "probe_unavailable";
        case Qwen36SSMQ8PrefillAMXRejectReason::Admitted:
            return "admitted";
        case Qwen36SSMQ8PrefillAMXRejectReason::DecodeOriginalQ8:
            return "decode_original_q8";
        case Qwen36SSMQ8PrefillAMXRejectReason::ResidentDecodeRegressionRisk:
            return "resident_decode_regression_risk";
        case Qwen36SSMQ8PrefillAMXRejectReason::PhaseUnknown:
            return "phase_unknown";
    }
    return "unknown";
}

static Qwen36SSMQ8PrefillAMXRejectReason ResolveQwen36SSMQ8PrefillAMXReason(
    densecore::llm::config::Qwen36SSMQ8PrefillAMXMode mode, InferenceExecutionPhase phase, bool lora_active) {
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

static void RecordQwen36SSMQ8PrefillAMXReject(InferenceWorkContext* ctx,
                                               Qwen36SSMQ8PrefillAMXRejectReason reason) {
    if (!ctx || reason == Qwen36SSMQ8PrefillAMXRejectReason::None) {
        return;
    }
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_last_reject_reason.store(static_cast<int>(reason),
                                                                           std::memory_order_relaxed);
}

static void RecordQwen36SSMQ8PrefillAMXUsed(InferenceWorkContext* ctx, int projection_kind) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_used.store(1, std::memory_order_relaxed);
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_used_ops.fetch_add(1, std::memory_order_relaxed);
    switch (projection_kind) {
        case 1:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_qkv_count.fetch_add(1, std::memory_order_relaxed);
            break;
        case 2:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_gate_count.fetch_add(1, std::memory_order_relaxed);
            break;
        case 3:
            ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_out_count.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

static bool Q4KRepackedGemvEnabled(const densecore::llm::config::FastPathRuntimeConfig& config,
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

static inline void RecordQ4KRepackedGemvCacheLookup(InferenceWorkContext* work_ctx,
                                                    const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup) {
    if (!work_ctx) {
        return;
    }
    auto& profile = work_ctx->qwen36_profile;
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
        profile.q4k_repacked_gemv_cache_evicted_bytes.fetch_add(lookup.cache_evicted_bytes,
                                                                std::memory_order_relaxed);
    }
    if (lookup.repack_bytes != 0) {
        profile.q4k_repacked_gemv_repack_bytes.fetch_add(lookup.repack_bytes, std::memory_order_relaxed);
    }
    if (lookup.resident_bytes != 0) {
        profile.q4k_repacked_gemv_resident_bytes.store(lookup.resident_bytes, std::memory_order_relaxed);
    }
    if (lookup.weight_key != 0) {
        std::lock_guard<std::mutex> lock(work_ctx->q4k_repacked_gemv_request_mutex);
        auto [it, inserted] = work_ctx->q4k_repacked_gemv_repack_counts.emplace(lookup.weight_key, 0);
        if (inserted) {
            profile.q4k_repacked_gemv_distinct_weights_seen.store(
                static_cast<uint64_t>(work_ctx->q4k_repacked_gemv_repack_counts.size()), std::memory_order_relaxed);
        }
        if (lookup.repacked) {
            ++it->second;
            if (it->second > 1) {
                profile.q4k_repacked_gemv_repeated_repack_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

static inline void SetQ4KRepackedGemvAutoDisable(InferenceWorkContext* work_ctx, Q4KRepackedGemvRejectReason local_reason,
                                                 Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || local_reason == Q4KRepackedGemvRejectReason::None) {
        return;
    }
    const int encoded = static_cast<int>(local_reason);
    int expected = 0;
    work_ctx->qwen36_profile.q4k_repacked_gemv_primary_disable_reason.compare_exchange_strong(
        expected, encoded, std::memory_order_relaxed);
    work_ctx->qwen36_profile.q4k_repacked_gemv_last_reject_reason.store(encoded, std::memory_order_relaxed);
    if (reason) {
        *reason = local_reason;
    }
}

static inline bool Q4KRepackedGemvShouldAutoDisableForLookup(
    InferenceWorkContext* work_ctx, const densecore::llm::config::FastPathRuntimeConfig& config,
    const densecore::kernels::Q4KRepackedGemvCacheLookup& lookup, Q4KRepackedGemvRejectReason* reason) {
    if (!work_ctx || !config.q4k_repacked_gemv_disable_on_thrash ||
        config.q4k_repacked_gemv != densecore::env::RuntimeToggleMode::Auto) {
        return false;
    }
    Q4KRepackedGemvRejectReason local_reason = Q4KRepackedGemvRejectReason::None;
    const uint64_t repeated_repack_count =
        work_ctx->qwen36_profile.q4k_repacked_gemv_repeated_repack_count.load(std::memory_order_relaxed);
    const uint64_t cache_evictions =
        work_ctx->qwen36_profile.q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed);
    const uint64_t used_ops = work_ctx->qwen36_profile.q4k_repacked_gemv_used_ops.load(std::memory_order_relaxed);
    const uint64_t repack_bytes =
        work_ctx->qwen36_profile.q4k_repacked_gemv_repack_bytes.load(std::memory_order_relaxed);
    const uint64_t repack_threshold_bytes =
        static_cast<uint64_t>(std::max(1, config.q4k_repacked_gemv_thrash_repack_mb)) * 1024ULL * 1024ULL;
    const uint64_t cache_fraction_threshold =
        lookup.cache_limit_bytes == 0
            ? 0
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
    } else if (repack_bytes >= repack_threshold_bytes &&
               cache_fraction_threshold > 0 && repack_bytes > cache_fraction_threshold) {
        local_reason = Q4KRepackedGemvRejectReason::RepackBytesHigh;
    } else if (lookup.repacked && lookup.cache_evictions != 0 &&
               work_ctx->qwen36_profile.q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed) >
                   lookup.cache_evictions) {
        local_reason = Q4KRepackedGemvRejectReason::CacheThrashing;
    }
    if (local_reason == Q4KRepackedGemvRejectReason::None) {
        return false;
    }
    SetQ4KRepackedGemvAutoDisable(work_ctx, local_reason, reason);
    return true;
}

struct Q4KRepackedGemvProbeKey {
    const void* weight = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t fingerprint = 0;

    bool operator==(const Q4KRepackedGemvProbeKey& other) const {
        return weight == other.weight && rows == other.rows && cols == other.cols && fingerprint == other.fingerprint;
    }
};

struct Q4KRepackedGemvProbeKeyHash {
    size_t operator()(const Q4KRepackedGemvProbeKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.cols) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.fingerprint) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct Q4KRepackedGemvProbeEntry {
    bool running = false;
    bool done = false;
    bool passed = false;
    std::condition_variable cv;
};

static bool RunQ4KRepackedGemvProbe(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
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

static bool Q4KRepackedGemvProbePassed(const std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight>& packed,
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
        weight_data, rows, cols,
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

static void RecordQActCacheHit(InferenceWorkContext* ctx, size_t bytes) {
    if (!ctx) return;
    ctx->qwen36_profile.qact_cache_hits.fetch_add(1, std::memory_order_relaxed);
    ctx->qwen36_profile.qact_cache_reused_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static void RecordQActCacheMiss(InferenceWorkContext* ctx) {
    if (!ctx) return;
    ctx->qwen36_profile.qact_cache_misses.fetch_add(1, std::memory_order_relaxed);
}

static bool QuantizedActivationCacheEnabled(const densecore::llm::config::FastPathRuntimeConfig& config) {
    if (config.qact_cache == densecore::env::RuntimeToggleMode::Off) {
        return false;
    }
    return config.qact_cache == densecore::env::RuntimeToggleMode::On ||
           config.qact_cache == densecore::env::RuntimeToggleMode::Auto;
}

static const uint8_t* GetOrFillQuantizedActivationCache(InferenceWorkContext* ctx, const ggml_tensor* src_tensor,
                                                        const void* source, const float* x_f32, int64_t len,
                                                        ggml_type quant_type, size_t quant_bytes, int slot_id,
                                                        int64_t token_pos,
                                                        const ggml_type_traits_cpu* input_type_traits) {
    if (!ctx || !src_tensor || !source || !x_f32 || len <= 0 || quant_type == GGML_TYPE_F32 || quant_bytes == 0 ||
        !input_type_traits || !input_type_traits->from_float) {
        return nullptr;
    }
    const bool data_pointer_matches_different_tensor =
        ctx->qact_source == source && ctx->qact_tensor && ctx->qact_tensor != src_tensor;
#ifndef NDEBUG
    if (data_pointer_matches_different_tensor) {
        static const bool strict_qact_cache_assert = []() {
            const char* env = std::getenv("DENSECORE_DEBUG_QACT_CACHE_ASSERT");
            return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        }();
        if (strict_qact_cache_assert) {
            assert(ctx->qact_tensor == src_tensor && "qact cache data pointer reused by a different tensor");
        }
    }
#endif
    if (ctx->qact_generation == ctx->execution_generation && ctx->qact_tensor == src_tensor &&
        ctx->qact_source == source && ctx->qact_len == len && ctx->qact_type == quant_type &&
        ctx->qact_bytes == quant_bytes && ctx->qact_slot_id == slot_id && ctx->qact_token_pos == token_pos &&
        ctx->qact_buffer.size() == quant_bytes) {
        RecordQActCacheHit(ctx, quant_bytes);
        return ctx->qact_buffer.data();
    }
    ctx->qact_buffer.resize(quant_bytes);
    input_type_traits->from_float(x_f32, ctx->qact_buffer.data(), len);
    ctx->qact_tensor = src_tensor;
    ctx->qact_generation = ctx->execution_generation;
    ctx->qact_source = source;
    ctx->qact_len = len;
    ctx->qact_type = quant_type;
    ctx->qact_bytes = quant_bytes;
    ctx->qact_slot_id = slot_id;
    ctx->qact_token_pos = token_pos;
    RecordQActCacheMiss(ctx);
    return ctx->qact_buffer.data();
}

static const uint8_t* GetOrFillBatchedQuantizedActivationCache(
    InferenceWorkContext* ctx, const ggml_tensor* src_tensor, const void* source,
    const std::vector<const float*>& x_rows, int M, int N, ggml_type quant_type, size_t quant_row_stride,
    size_t quant_bytes, int64_t token_pos, const ggml_type_traits_cpu* input_type_traits) {
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
    if (ctx->qact_generation == ctx->execution_generation && ctx->qact_tensor == src_tensor &&
        ctx->qact_source == source && ctx->qact_len == len && ctx->qact_type == quant_type &&
        ctx->qact_bytes == quant_bytes && ctx->qact_slot_id == -1 && ctx->qact_token_pos == token_pos &&
        ctx->qact_buffer.size() == quant_bytes) {
        RecordQActCacheHit(ctx, quant_bytes);
        return ctx->qact_buffer.data();
    }

    ctx->qact_buffer.resize(quant_bytes);
    for (int m = 0; m < M; ++m) {
        uint8_t* q_ptr = ctx->qact_buffer.data() + static_cast<size_t>(m) * quant_row_stride;
        input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
    }
    ctx->qact_tensor = src_tensor;
    ctx->qact_generation = ctx->execution_generation;
    ctx->qact_source = source;
    ctx->qact_len = len;
    ctx->qact_type = quant_type;
    ctx->qact_bytes = quant_bytes;
    ctx->qact_slot_id = -1;
    ctx->qact_token_pos = token_pos;
    RecordQActCacheMiss(ctx);
    return ctx->qact_buffer.data();
}

enum class Qwen36Q4KBatchedAdmissionState : int { Unknown = 0, Pass = 1, Reject = 2 };

enum class Qwen36PrefillQ4KBatchedRejectReason : int {
    None = 0,
    EnvOff = 1,
    LoraActive = 2,
    UnsupportedShape = 3,
    MissingVecDot = 4,
    KernelUnavailable = 5,
    ProbeMismatch = 6,
    ProbeInternalError = 7,
    Admitted = 8,
    RejectedCached = 9,
    NotQ4K = 10,
};

const char* Qwen36PrefillQ4KBatchedRejectReasonName(int reason) {
    switch (static_cast<Qwen36PrefillQ4KBatchedRejectReason>(reason)) {
        case Qwen36PrefillQ4KBatchedRejectReason::None:
            return "none";
        case Qwen36PrefillQ4KBatchedRejectReason::EnvOff:
            return "env_off";
        case Qwen36PrefillQ4KBatchedRejectReason::LoraActive:
            return "lora_active";
        case Qwen36PrefillQ4KBatchedRejectReason::UnsupportedShape:
            return "unsupported_shape";
        case Qwen36PrefillQ4KBatchedRejectReason::MissingVecDot:
            return "missing_vec_dot";
        case Qwen36PrefillQ4KBatchedRejectReason::KernelUnavailable:
            return "kernel_unavailable";
        case Qwen36PrefillQ4KBatchedRejectReason::ProbeMismatch:
            return "probe_mismatch";
        case Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError:
            return "probe_internal_error";
        case Qwen36PrefillQ4KBatchedRejectReason::Admitted:
            return "admitted";
        case Qwen36PrefillQ4KBatchedRejectReason::RejectedCached:
            return "rejected_cached";
        case Qwen36PrefillQ4KBatchedRejectReason::NotQ4K:
            return "not_q4k";
    }
    return "unknown";
}

static Qwen36PrefillQ4KBatchedRejectReason ResolveQwen36PrefillQ4KBatchedReason(
    bool relevant, bool mode_off, bool lora_active, bool weight_is_q4k, bool shape_supported, bool kernel_available,
    bool has_vec_dot, bool candidate_ready, bool mode_on, bool mode_probe,
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

struct Qwen36Q4KBatchedAdmissionValue {
    Qwen36Q4KBatchedAdmissionState state = Qwen36Q4KBatchedAdmissionState::Unknown;
    float max_abs_error = 0.0f;
    Qwen36PrefillQ4KBatchedRejectReason reject_reason = Qwen36PrefillQ4KBatchedRejectReason::None;
};

static std::mutex& Qwen36Q4KBatchedAdmissionMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue>& Qwen36Q4KBatchedAdmissionMap() {
    static std::unordered_map<uint64_t, Qwen36Q4KBatchedAdmissionValue> map;
    return map;
}

static std::atomic<int>& Qwen36Q4KBatchedProbeForceFailThreadForTest() {
    static std::atomic<int> value{-1};
    return value;
}

static uint64_t HashQwen36Q4KBatchedAdmissionKey(const TransformerModel* model, const ggml_tensor* weight,
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

static Qwen36Q4KBatchedAdmissionValue LookupQwen36Q4KBatchedAdmission(uint64_t key) {
    std::lock_guard<std::mutex> lock(Qwen36Q4KBatchedAdmissionMutex());
    auto& map = Qwen36Q4KBatchedAdmissionMap();
    auto it = map.find(key);
    return it == map.end() ? Qwen36Q4KBatchedAdmissionValue{} : it->second;
}

static void StoreQwen36Q4KBatchedAdmission(uint64_t key, Qwen36Q4KBatchedAdmissionState state, float max_abs_error,
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

static void RecordQwen36Q4KBatchedProbeResult(InferenceWorkContext* ctx, bool pass, float max_abs_error,
                                              Qwen36PrefillQ4KBatchedRejectReason reason) {
    if (!ctx) return;
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_probe_pass.store(pass ? 1 : 0, std::memory_order_relaxed);
    uint32_t bits = 0;
    std::memcpy(&bits, &max_abs_error, sizeof(float));
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_max_abs_error_bits.store(bits, std::memory_order_relaxed);
    ctx->qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason),
                                                                            std::memory_order_relaxed);
}

static void DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(
    uint64_t key, float max_abs_error, Qwen36PrefillQ4KBatchedRejectReason reason, InferenceWorkContext* ctx) {
    if (!key) {
        return;
    }
    StoreQwen36Q4KBatchedAdmission(key, Qwen36Q4KBatchedAdmissionState::Reject, max_abs_error, reason);
    if (ctx) {
        ctx->qwen36_profile.qwen36_prefill_q4k_admission_downgraded.fetch_add(1, std::memory_order_relaxed);
        ctx->qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason.store(static_cast<int>(reason),
                                                                                std::memory_order_relaxed);
    }
}

static void AtomicMaxFloatBits(std::atomic<uint32_t>& target, float value) {
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

static bool IsGemma4SharedDenseFfnWeightName(const char* weight_name) {
    if (!weight_name) {
        return false;
    }
    return std::strstr(weight_name, ".ffn_gate.weight") || std::strstr(weight_name, ".ffn_up.weight") ||
           std::strstr(weight_name, ".ffn_down.weight");
}

static bool IsQ8RepackedGemvEnabled() {
    static const bool enabled = []() -> bool {
        const char* env = std::getenv("DENSECORE_DISABLE_Q8_REPACKED_GEMV");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0) {
            return false;
        }
#if defined(__aarch64__) || defined(_M_ARM64)
        return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
        const char* enable_env = std::getenv("DENSECORE_ENABLE_Q8_REPACKED_GEMV");
        return enable_env && enable_env[0] != '\0' && std::strcmp(enable_env, "0") != 0 &&
               std::strcmp(enable_env, "false") != 0 && std::strcmp(enable_env, "off") != 0;
#endif
    }();
    return enabled;
}

static std::shared_ptr<Q8RepackedGemvWeight> GetOrCreateQ8RepackedGemvWeight(const void* weight_data, int64_t rows,
                                                                              int64_t cols,
                                                                              bool force_enable = false) {
    if (!weight_data || rows <= 0 || cols <= 0 || (rows % 4) != 0 || (cols % QK8_0) != 0 ||
        (!force_enable && !IsQ8RepackedGemvEnabled())) {
        return nullptr;
    }
    static std::mutex mutex;
    static std::unordered_map<Q8RepackedGemvKey, std::shared_ptr<Q8RepackedGemvWeight>, Q8RepackedGemvKeyHash> cache;

    const Q8RepackedGemvKey key{weight_data, rows, cols};
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
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
    return inserted ? packed : insert_it->second;
}
