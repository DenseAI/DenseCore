bool TryRunPackedInt4Projection(CpuBackend* backend, const CpuBackend::ExpertPackedInt4Weight& binding,
                                const Tensor& input, Tensor* output, int numa_node, bool allow_parallel,
                                CpuBackend::MoEProjectionPath* selected_path = nullptr);
bool TryRunGgmlQuantizedProjection(CpuBackend* backend, const void* weight_ptr, int ggml_type_id, const Tensor& input,
                                   Tensor* output, int64_t N, int64_t K, int numa_node, bool allow_parallel,
                                   QuantizedProjectionInputCache* input_cache, bool allow_kquant_rowpair_vec_dot,
                                   bool prefer_q4k_repacked_prefill);

struct MoEProjectionRequest {
    CpuBackend* backend = nullptr;
    int numa_node = 0;
    char projection_slot = '?';
    const Tensor* src = nullptr;
    const Tensor* dense_weight = nullptr;
    const CpuBackend::ExpertPackedInt4Weight* int4_binding = nullptr;
    const CpuBackend::ExpertWeight* raw_weight = nullptr;
    int ggml_type_id = GGML_TYPE_COUNT;
    int64_t proj_rows = 0;
    int64_t proj_cols = 0;
    Tensor* dst = nullptr;
    const Tensor* original_input = nullptr;
    QuantizedProjectionInputCache* input_projection_cache = nullptr;
    QuantizedProjectionInputCache* down_projection_cache = nullptr;
    const CpuBackend::ExpertWeights* expert = nullptr;
    const MoEExecutionTraceContext* trace_ctx = nullptr;
    InferenceWorkContext* gemma4_quant_prefill_ctx = nullptr;
    bool safe_reference_mode = false;
    bool ggml_quantized_vecdot_safe = false;
    bool force_gemma4_quant_prefill_fast_path = false;
    bool gemma4_quant_prefill_batch_safe = false;
    bool enable_inner_parallel = false;
};

struct MoEProjectionPlan {
    bool projection_has_scale = false;
    bool projection_scalar_scale = false;
    bool quantized_scale_supported = true;
    bool try_packed_int4 = false;
    bool try_ggml_quantized = false;
    bool force_quant_prefill_failure = false;
    bool allow_dense_fallback = false;
    bool use_input_projection_cache = false;
    bool allow_rowpair_vec_dot = true;
};

MoEProjectionPlan ResolveMoEProjectionPlan(const MoEProjectionRequest& req) {
    MoEProjectionPlan plan;
    plan.projection_has_scale =
        req.expert && req.projection_slot == '2' && req.expert->w2_scale_tensor != nullptr;
    plan.projection_scalar_scale = plan.projection_has_scale && IsScalarScaleSidecar(req.expert->w2_scale_tensor);
    plan.quantized_scale_supported = !plan.projection_has_scale || plan.projection_scalar_scale;
    plan.try_packed_int4 = !req.safe_reference_mode && plan.quantized_scale_supported;
    plan.allow_rowpair_vec_dot = !(req.expert && req.expert->use_gelu_activation && req.projection_slot == '2' &&
                                   req.ggml_type_id == GGML_TYPE_Q4_K);
    plan.try_ggml_quantized = !req.safe_reference_mode && req.ggml_quantized_vecdot_safe &&
                              plan.quantized_scale_supported && req.raw_weight && req.raw_weight->ptr &&
                              req.ggml_type_id != GGML_TYPE_F32;
    plan.force_quant_prefill_failure = req.force_gemma4_quant_prefill_fast_path && req.projection_slot == '2' &&
                                       plan.quantized_scale_supported && req.raw_weight && req.raw_weight->ptr &&
                                       req.ggml_type_id != GGML_TYPE_F32;
    plan.allow_dense_fallback = req.dense_weight && req.dense_weight->IsValid();
    plan.use_input_projection_cache = req.src && req.original_input &&
                                      req.src->DataAs<float>() == req.original_input->DataAs<float>();
    return plan;
}

void RecordMoEProjectionPath(CpuBackend* backend, const MoEExecutionTraceContext* trace_ctx,
                             const CpuBackend::ExpertWeights* expert, char projection_slot,
                             bool safe_reference_mode, const char* path_name) {
    if (!backend || !trace_ctx) {
        return;
    }
    CpuBackend::MoEPathTraceEntry entry;
    entry.layer_idx = trace_ctx->layer_idx;
    entry.seq_id = trace_ctx->seq_id;
    entry.token_idx = trace_ctx->token_idx;
    entry.decode_step = trace_ctx->decode_step;
    entry.n_past = trace_ctx->n_past;
    entry.expert_id = trace_ctx->expert_id;
    entry.force_safe_reference = expert ? expert->force_safe_reference : false;
    entry.safe_reference_mode = safe_reference_mode;
    entry.projection[0] = projection_slot;
    entry.projection[1] = '\0';
    entry.selected_path = ParseMoEProjectionPath(path_name);
    backend->RecordMoEPathTrace(entry);
}

void ApplyMoEProjectionScaleIfNeeded(const MoEProjectionRequest& req, const MoEProjectionPlan& plan) {
    if (plan.projection_scalar_scale && req.expert && req.dst) {
        ApplyScalarScaleToTensor(req.dst, ReadScalarScaleSidecar(req.expert->w2_scale_tensor));
    }
}

bool EmitMoEProjectionFromPlan(const MoEProjectionRequest& req, const MoEProjectionPlan& plan) {
    if (!req.backend || !req.src || !req.dst || !req.int4_binding || !req.raw_weight) {
        return false;
    }

    CpuBackend::MoEProjectionPath packed_path = CpuBackend::MoEProjectionPath::Unknown;
    if (plan.try_packed_int4 &&
        TryRunPackedInt4Projection(req.backend, *req.int4_binding, *req.src, req.dst, req.numa_node,
                                   req.enable_inner_parallel, &packed_path)) {
        ApplyMoEProjectionScaleIfNeeded(req, plan);
        RecordMoEProjectionPath(req.backend, req.trace_ctx, req.expert, req.projection_slot, req.safe_reference_mode,
                                packed_path == CpuBackend::MoEProjectionPath::PackedInt4Fast ? "direct_hwy"
                                                                                              : "backend_gemmint4");
        return true;
    }

    if (plan.try_ggml_quantized &&
        TryRunGgmlQuantizedProjection(
            req.backend, req.raw_weight->ptr, req.ggml_type_id, *req.src, req.dst, req.proj_rows, req.proj_cols,
            req.numa_node, req.enable_inner_parallel,
            plan.use_input_projection_cache ? req.input_projection_cache : req.down_projection_cache,
            plan.allow_rowpair_vec_dot, req.gemma4_quant_prefill_batch_safe)) {
        ApplyMoEProjectionScaleIfNeeded(req, plan);
        LogMoEMatmulPath("ggml_quantized_vecdot", static_cast<int>(req.src->shape[0]),
                         static_cast<int>(req.src->shape[1]), static_cast<int>(req.dst->shape[1]),
                         req.int4_binding->group_size, req.enable_inner_parallel);
        RecordMoEProjectionPath(req.backend, req.trace_ctx, req.expert, req.projection_slot, req.safe_reference_mode,
                                "ggml_quantized_vecdot");
        if (req.gemma4_quant_prefill_batch_safe && req.projection_slot == '2') {
            RecordGemma4MoEPrefillQuantBatchDecision(req.gemma4_quant_prefill_ctx, false, true, nullptr, false, true);
        }
        return true;
    }

    if (plan.force_quant_prefill_failure) {
        RecordGemma4MoEPrefillQuantBatchDecision(req.gemma4_quant_prefill_ctx, false, false,
                                                 "down_fast_path_failed", false, true);
        throw std::runtime_error("Gemma4 MoE prefill quant batch down fast path failed");
    }

    if (plan.allow_dense_fallback) {
        GetMoEInt4PathHistogram().f32_fallback.fetch_add(1, std::memory_order_relaxed);
        const char* path_name = req.safe_reference_mode ? "reference_f32" : "dense_f32";
        LogMoEMatmulPath(path_name, static_cast<int>(req.src->shape[0]), static_cast<int>(req.src->shape[1]),
                         static_cast<int>(req.dst->shape[1]), req.int4_binding->group_size,
                         req.enable_inner_parallel);
        RecordMoEProjectionPath(req.backend, req.trace_ctx, req.expert, req.projection_slot, req.safe_reference_mode,
                                path_name);
        req.backend->MatMulTransB(*req.src, *req.dense_weight, req.dst, req.numa_node);
        return true;
    }

    return false;
}

bool RunMoEProjectionFromContext(const MoEProjectionRuntimeContext& ctx, char projection_slot, const Tensor& src,
                                 const Tensor& dense_weight,
                                 const CpuBackend::ExpertPackedInt4Weight& int4_binding,
                                 const CpuBackend::ExpertWeight& raw_weight, int ggml_type_id, int64_t proj_rows,
                                 int64_t proj_cols, Tensor* dst) {
    MoEProjectionRequest request;
    request.backend = ctx.backend;
    request.numa_node = ctx.numa_node;
    request.projection_slot = projection_slot;
    request.src = &src;
    request.dense_weight = &dense_weight;
    request.int4_binding = &int4_binding;
    request.raw_weight = &raw_weight;
    request.ggml_type_id = ggml_type_id;
    request.proj_rows = proj_rows;
    request.proj_cols = proj_cols;
    request.dst = dst;
    request.original_input = ctx.original_input;
    request.input_projection_cache = ctx.input_projection_cache;
    request.down_projection_cache = ctx.down_projection_cache;
    request.expert = ctx.expert;
    request.trace_ctx = ctx.trace_ctx;
    request.gemma4_quant_prefill_ctx = ctx.gemma4_quant_prefill_ctx;
    request.safe_reference_mode = ctx.safe_reference_mode;
    request.ggml_quantized_vecdot_safe = ctx.ggml_quantized_vecdot_safe;
    request.force_gemma4_quant_prefill_fast_path = ctx.force_gemma4_quant_prefill_fast_path;
    request.gemma4_quant_prefill_batch_safe = ctx.gemma4_quant_prefill_batch_safe;
    request.enable_inner_parallel = ctx.enable_inner_parallel;

    const MoEProjectionPlan plan = ResolveMoEProjectionPlan(request);
    return EmitMoEProjectionFromPlan(request, plan);
}

