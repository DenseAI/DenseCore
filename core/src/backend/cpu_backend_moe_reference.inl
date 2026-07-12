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

bool IsMoESafeReferenceModeEnabled(const CpuBackend::ExpertWeights* expert) {
    return expert && expert->force_safe_reference;
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

bool IsGemma4PackedChecksumDebugEnabled() {
    const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_PACKED_CHECKSUM");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

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

bool IsQwenA3BHybridMoEModel(const TransformerModel* model) {
    return model && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
           model->arch_flags.is_hybrid_ssm && model->hparams.n_experts > 0 && model->hparams.n_experts_used > 1;
}

bool IsGemma4MoEModelForSmallDecodeParallel(const TransformerModel* model) {
    return model && model->arch_flags.is_gemma4 && model->hparams.n_experts > 0;
}

bool IsLFM2MoEModelForSmallDecodeParallel(const TransformerModel* model) {
    return model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv &&
           model->hparams.n_experts > 0 && model->hparams.n_experts_used > 1;
}

bool IsSmallDecodeExpertParallelAutoModel(const TransformerModel* model) {
    return IsQwenA3BHybridMoEModel(model) || IsGemma4MoEModelForSmallDecodeParallel(model) ||
           IsLFM2MoEModelForSmallDecodeParallel(model);
}

bool IsSmallDecodeExpertParallelSimdLevel(densecore::simd::SimdLevel level) {
    return level == densecore::simd::SimdLevel::SVE || level == densecore::simd::SimdLevel::SVE2 ||
           densecore::simd::HasX86Avx2OrBetter(level);
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

int ResolveSmallDecodeExpertWorkers(int top_k, int worker_cap) {
    const int requested = std::min(std::max(1, top_k), 8);
    return std::max(1, std::min(std::max(1, worker_cap), requested));
}

struct SmallDecodeExpertParallelDecision {
    bool requested = false;
    bool enabled = false;
    int workers = 1;
    const char* reason = "disabled";
};

SmallDecodeExpertParallelDecision ResolveSmallDecodeExpertParallelDecision(const TransformerModel* model,
                                                                           int batch_size, int top_k,
                                                                           bool safe_reference_mode, int worker_cap) {
    SmallDecodeExpertParallelDecision decision;
    decision.workers = ResolveSmallDecodeExpertWorkers(top_k, worker_cap);

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
    decision.reason = "auto";
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
#if defined(__aarch64__) || defined(_M_ARM64)
    // Qwen3.5-35B-A3B batch=1 decode on C4A repeatedly executes tiny expert
    // GEMV/GEMM fragments (top-k experts, one token). Fanning each fragment out
    // across the whole thread pool costs more than the math itself and was the
    // dominant source of backend_us inflation after the crash fix.
    const int64_t work_items = batch * intermediate_dim;
    if (batch <= 1 && hidden_dim <= 4096 && intermediate_dim <= 8192 && work_items <= 8192) {
        return false;
    }
#else
    (void)batch;
    (void)hidden_dim;
    (void)intermediate_dim;
#endif
    return true;
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
