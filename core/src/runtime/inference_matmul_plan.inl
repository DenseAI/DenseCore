// Smart matmul planning, kernel-resolution, and fallback-free emit helpers.
struct SmartMatmulAliasSelection {
    ggml_tensor* weight = nullptr;
    bool using_cpu_repack_alias = false;
};

struct SmartMatmulExecutionPlan {
    ggml_tensor* weight = nullptr;
    SmartMatmulOriginalTensorState original{};
    const BatchSpec* current_batch = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    InferenceExecutionPhase dispatch_phase = InferenceExecutionPhase::Unknown;
    bool dispatch_is_prefill_phase = false;
    bool using_cpu_repack_alias = false;
    bool using_qwen36_ssm_q8_prefill_amx_alias = false;
};

static SmartMatmulAliasSelection ResolveSmartMatmulAliasSelection(
    TransformerModel* model, ggml_tensor* weight, const ggml_tensor* input,
    const SmartMatmulOriginalTensorState& original, bool dispatch_is_prefill_phase) {
    SmartMatmulAliasSelection selection;
    selection.weight = weight;
    if (!model || !weight || !input || input->type != GGML_TYPE_F32) {
        return selection;
    }

    constexpr int64_t kQwenHybridSmallPrefillQ4KMaxCols = 256;
    const bool qwen_hybrid_small_prefill_q4k =
        dispatch_is_prefill_phase && original.input_cols > 1 &&
        original.input_cols <= kQwenHybridSmallPrefillQ4KMaxCols;
    const bool prefer_qwen_hybrid_ssm_quant_batched_over_repack =
        original.qwen_hybrid_ssm_projection && qwen_hybrid_small_prefill_q4k &&
        selection.weight->type == GGML_TYPE_Q4_K && input->ne[0] == selection.weight->ne[0] &&
        (input->ne[0] % QK_K == 0) && IsQ4KTrueBatchedKernelEnabled();
    const bool prefer_gemma4_quant_prefill_densecore_path =
        dispatch_is_prefill_phase && model->arch_flags.is_gemma4 &&
        (selection.weight->type == GGML_TYPE_Q4_K || selection.weight->type == GGML_TYPE_Q8_0) &&
        input->ne[1] > 1 && input->ne[0] == selection.weight->ne[0] &&
        (selection.weight->type != GGML_TYPE_Q4_K || IsQ4KTrueBatchedKernelEnabled());
    const bool prefer_gemma4_quant_decode_densecore_path =
        model->arch_flags.is_gemma4 && ggml_is_quantized(selection.weight->type) &&
        input->ne[1] <= 1 && input->ne[0] == selection.weight->ne[0];
    const bool prefer_lfm2_quant_prefill_densecore_path =
        dispatch_is_prefill_phase && original.lfm2_shortconv_semantic &&
        ggml_is_quantized(selection.weight->type) && input->ne[1] > 1 &&
        input->ne[0] == selection.weight->ne[0];
    const bool prefer_lfm2_quant_decode_densecore_path =
        original.lfm2_shortconv_semantic && ggml_is_quantized(selection.weight->type) &&
        input->ne[1] <= 1 && input->ne[0] == selection.weight->ne[0];

    if (input->ne[1] <= 1) {
        auto it_decode_repack = model->cpu_decode_repack_aliases.find(selection.weight);
        if (!prefer_gemma4_quant_decode_densecore_path && !prefer_lfm2_quant_decode_densecore_path &&
            it_decode_repack != model->cpu_decode_repack_aliases.end() && it_decode_repack->second) {
            selection.weight = it_decode_repack->second;
            selection.using_cpu_repack_alias = true;
        }
    }
    auto it_repack = model->cpu_repack_aliases.find(selection.weight);
    if (!selection.using_cpu_repack_alias && it_repack != model->cpu_repack_aliases.end() && it_repack->second) {
        const bool amx_alias = model->cpu_amx_aliases.find(it_repack->second) != model->cpu_amx_aliases.end();
        const bool amx_decode_regression_risk =
            (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) && amx_alias &&
            input->ne[1] == 1;
        if (!prefer_gemma4_quant_prefill_densecore_path && !prefer_lfm2_quant_prefill_densecore_path &&
            !prefer_gemma4_quant_decode_densecore_path && !prefer_lfm2_quant_decode_densecore_path &&
            !prefer_qwen_hybrid_ssm_quant_batched_over_repack && !amx_decode_regression_risk) {
            selection.weight = it_repack->second;
            selection.using_cpu_repack_alias = true;
        }
    }
    return selection;
}

static SmartMatmulOriginalTensorState CaptureSmartMatmulOriginalTensorState(const TransformerModel* model,
                                                                            ggml_tensor* weight,
                                                                            const ggml_tensor* input) {
    SmartMatmulOriginalTensorState state;
    state.weight = weight;
    state.input_cols = input ? static_cast<int>(input->ne[1]) : 0;
    state.weight_name = SmartMatmulTensorName(weight);
    state.requirement = ResolveSmartMatmulRequirement(model, weight);
    state.hybrid_ssm_projection = densecore::runtime::IsHybridSsmTensorRole(state.requirement.tensor_role);
    state.hybrid_ssm_projection_kind = densecore::runtime::HybridSsmProjectionKind(state.requirement.tensor_role);
    state.qwen_target_model = model && densecore::runtime::IsQwenTargetVariant(model->variant);
    state.qwen_hybrid_ssm_projection = state.qwen_target_model && state.hybrid_ssm_projection;
    state.lfm2_shortconv_semantic =
        state.requirement.semantic_op == densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer ||
        IsLFM2TargetProjectionWeight(model, weight, model && model->output == weight);
    return state;
}

static bool ResolveSmartMatmulPrefillPhase(InferenceExecutionPhase dispatch_phase,
                                           const SmartMatmulOriginalTensorState& original) {
    return dispatch_phase == InferenceExecutionPhase::Prefill ||
           (dispatch_phase == InferenceExecutionPhase::Unknown && original.input_cols > 1);
}

static densecore::runtime::DenseCoreMatmulPhase ResolveSmartMatmulPlanPhase(
    InferenceExecutionPhase dispatch_phase, bool dispatch_is_prefill_phase) {
    if (dispatch_is_prefill_phase) {
        return densecore::runtime::DenseCoreMatmulPhase::Prefill;
    }
    if (dispatch_phase == InferenceExecutionPhase::Decode) {
        return densecore::runtime::DenseCoreMatmulPhase::Decode;
    }
    return densecore::runtime::DenseCoreMatmulPhase::Unknown;
}

static SmartMatmulExecutionPlan ResolveSmartMatmulExecutionPlan(TransformerModel* model, ggml_tensor* weight,
                                                               ggml_tensor* input) {
    SmartMatmulExecutionPlan plan;
    plan.weight = ResolveSmartMatmulInitialWeight(model, weight, input);
    plan.original = CaptureSmartMatmulOriginalTensorState(model, plan.weight, input);
    plan.current_batch = GetCurrentBatch();
    plan.work_ctx = GetCurrentWorkContext();
    plan.dispatch_phase = plan.work_ctx ? plan.work_ctx->phase : InferenceExecutionPhase::Unknown;
    plan.dispatch_is_prefill_phase = ResolveSmartMatmulPrefillPhase(plan.dispatch_phase, plan.original);

    const SmartMatmulAliasSelection alias_selection =
        ResolveSmartMatmulAliasSelection(model, plan.weight, input, plan.original, plan.dispatch_is_prefill_phase);
    plan.weight = alias_selection.weight;
    plan.using_cpu_repack_alias = alias_selection.using_cpu_repack_alias;
    return plan;
}

static void RecordSmartMatmulRuntimeModes(
    InferenceWorkContext* work_ctx, const densecore::llm::config::FastPathRuntimeConfig& fast_path_config) {
    if (!work_ctx) {
        return;
    }
    work_ctx->qwen36_profile.qwen36_prefill_q4k_batched_mode.store(
        static_cast<int>(fast_path_config.qwen36_prefill_q4k_batched), std::memory_order_relaxed);
    work_ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_mode.store(
        static_cast<int>(fast_path_config.qwen36_ssm_q8_prefill_amx), std::memory_order_relaxed);
}

static void ApplySmartMatmulQwen36Q8PrefillAlias(
    TransformerModel* model, const ggml_tensor* input,
    const densecore::llm::config::FastPathRuntimeConfig& fast_path_config, SmartMatmulExecutionPlan* plan) {
    if (!plan || !model || !input || input->type != GGML_TYPE_F32) {
        return;
    }
    ggml_tensor* const original_weight = plan->original.weight;
    const int qwen36_ssm_projection_kind = plan->original.hybrid_ssm_projection_kind;
    if (plan->work_ctx && plan->original.qwen_hybrid_ssm_projection && qwen36_ssm_projection_kind != 0 &&
        original_weight) {
        RecordQwen36SSMProjectionWeightType(plan->work_ctx, original_weight->type);
    }
    const bool original_qwen36_ssm_q8_projection =
        model->variant == ModelVariant::QWEN36 && plan->original.hybrid_ssm_projection &&
        qwen36_ssm_projection_kind != 0 && original_weight && original_weight->type == GGML_TYPE_Q8_0 &&
        original_weight->ne[0] == input->ne[0];
    if (original_qwen36_ssm_q8_projection && plan->original.input_cols <= 1 && plan->weight == original_weight) {
        plan->work_ctx = plan->work_ctx ? plan->work_ctx : GetCurrentWorkContext();
        if (plan->work_ctx) {
            plan->work_ctx->qwen36_profile.qwen36_ssm_q8_decode_used_original_q8_path.store(
                1, std::memory_order_relaxed);
            RecordQwen36SSMQ8PrefillAMXReject(
                plan->work_ctx, ResolveQwen36SSMQ8PrefillAMXReason(
                                    fast_path_config.qwen36_ssm_q8_prefill_amx, plan->dispatch_phase,
                                    plan->current_batch && !plan->current_batch->lora_map.empty()));
        }
    }
    if (!plan->dispatch_is_prefill_phase || !original_qwen36_ssm_q8_projection || plan->original.input_cols <= 1) {
        return;
    }

    plan->work_ctx = plan->work_ctx ? plan->work_ctx : GetCurrentWorkContext();
    auto reject_q8_prefill = [&](Qwen36SSMQ8PrefillAMXRejectReason reason, bool candidate) {
        if (candidate && plan->work_ctx) {
            plan->work_ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_rejected_ops.fetch_add(
                1, std::memory_order_relaxed);
        }
        RecordQwen36SSMQ8PrefillAMXReject(plan->work_ctx, reason);
    };
    const Qwen36SSMQ8PrefillAMXRejectReason q8_prefill_gate = ResolveQwen36SSMQ8PrefillAMXReason(
        fast_path_config.qwen36_ssm_q8_prefill_amx, plan->dispatch_phase,
        plan->current_batch && !plan->current_batch->lora_map.empty());
    if (q8_prefill_gate != Qwen36SSMQ8PrefillAMXRejectReason::None) {
        reject_q8_prefill(q8_prefill_gate, false);
        return;
    }
    if (plan->work_ctx) {
        plan->work_ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_candidate_ops.fetch_add(
            1, std::memory_order_relaxed);
    }
    auto it_prefill_amx = model->qwen36_ssm_q8_prefill_amx_aliases.find(original_weight);
    if (it_prefill_amx == model->qwen36_ssm_q8_prefill_amx_aliases.end() || !it_prefill_amx->second) {
        reject_q8_prefill(Qwen36SSMQ8PrefillAMXRejectReason::AliasUnavailable, true);
    } else if (model->cpu_amx_aliases.find(it_prefill_amx->second) == model->cpu_amx_aliases.end()) {
        reject_q8_prefill(Qwen36SSMQ8PrefillAMXRejectReason::BackendUnavailable, true);
    } else {
        plan->weight = it_prefill_amx->second;
        plan->using_qwen36_ssm_q8_prefill_amx_alias = true;
        plan->using_cpu_repack_alias = false;
        RecordQwen36SSMQ8PrefillAMXUsed(plan->work_ctx, qwen36_ssm_projection_kind);
    }
}

struct SmartMatmulDispatchState {
    int m = 0;
    int k = 0;
    int n = 0;
    const char* weight_name = "(unnamed)";
    const char* input_name = "(unnamed)";
    bool gemma4_lm_head = false;
    bool compatible = false;
    bool prefill_phase = false;
    densecore::runtime::DenseCoreMatmulPhase plan_phase = densecore::runtime::DenseCoreMatmulPhase::Unknown;
    densecore::runtime::HostKernelCapabilities host_caps{};
    densecore::models::ModelTensorExecutionRequirement requirement{};
    densecore::runtime::KernelResolution kernel_resolution{};
    densecore::runtime::DenseCoreMatmulPlan matmul_plan{};
    densecore::runtime::DenseCoreTensorRole tensor_role = densecore::runtime::DenseCoreTensorRole::Unknown;
    densecore::runtime::DenseCoreSemanticOp semantic_op = densecore::runtime::DenseCoreSemanticOp::Unknown;
    bool hybrid_ssm_qkv = false;
    bool hybrid_ssm_gate = false;
    bool hybrid_ssm_out = false;
    bool hybrid_ssm_projection = false;
    bool moe_router = false;
    bool lm_head_role = false;
    bool lfm2_shortconv_semantic = false;
    bool fallback_free_target = false;
    bool qwen_target = false;
    bool qwen35_target = false;
    bool qwen36_target = false;
    bool qwen35_hybrid_ssm = false;
    bool qwen36_hybrid_ssm = false;
    bool qwen35_hybrid_ssm_gate = false;
    bool qwen35_hybrid_ssm_out = false;
    bool qwen36_hybrid_ssm_qkv = false;
    bool qwen36_hybrid_ssm_gate = false;
    bool qwen36_hybrid_ssm_out = false;
    bool qwen36_lm_head = false;
};

struct SmartMatmulPrefillProjectionPlan {
    bool qwen_hybrid_ssm_q8_prefill = false;
    bool qwen_hybrid_ssm_quant_prefill_fast_path_eligible = false;
    bool qwen36_hybrid_ssm_q4k_prefill_relevant = false;
    bool qwen36_lm_head_q4k_prefill_relevant = false;
    bool lfm2_prefill_q4k_relevant = false;
    bool lfm2_prefill_q6k_relevant = false;
    bool lfm2_prefill_quant_nrc_unsafe = false;
    bool q4k_batched_prefill_relevant = false;
    bool qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k = false;
    bool qwen36_hybrid_ssm_q4k_prefill_shape_supported = false;
    bool qwen36_hybrid_ssm_q4k_prefill_kernel_available = false;
    bool qwen36_hybrid_ssm_q4k_prefill_lora_active = false;
    bool qwen36_hybrid_ssm_q4k_prefill_probe_candidate = false;
};

static SmartMatmulDispatchState ResolveSmartMatmulDispatchState(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input,
    const ggml_tensor* original_weight, InferenceExecutionPhase dispatch_phase, bool dispatch_is_prefill_phase,
    bool using_cpu_repack_alias) {
    SmartMatmulDispatchState state;
    state.m = input ? static_cast<int>(input->ne[1]) : 0;
    state.k = weight ? static_cast<int>(weight->ne[0]) : 0;
    state.n = weight ? static_cast<int>(weight->ne[1]) : 0;
    state.weight_name = SmartMatmulTensorName(weight);
    state.input_name = SmartMatmulTensorName(input);
    state.gemma4_lm_head = model && (model->output == weight || model->output == original_weight);
    state.compatible = weight && input && weight->ne[0] == input->ne[0];
    state.prefill_phase = dispatch_is_prefill_phase;
    state.plan_phase = ResolveSmartMatmulPlanPhase(dispatch_phase, dispatch_is_prefill_phase);
    state.host_caps = densecore::runtime::CompileTimeHostKernelCapabilities(IsQ4KTrueBatchedKernelEnabled(),
                                                                            using_cpu_repack_alias);
    state.requirement = ResolveSmartMatmulRequirement(model, weight);
    state.kernel_resolution = densecore::runtime::ResolveKernelResolution(
        model, weight ? weight->type : GGML_TYPE_COUNT, input ? input->type : GGML_TYPE_COUNT, state.m, state.n,
        state.k, state.plan_phase, state.weight_name, model && model->output == weight, state.compatible,
        state.host_caps, state.requirement.semantic_op, state.requirement.tensor_role,
        densecore::runtime::DenseCoreKernelFamily::None,
        state.requirement.fallback_policy, /*has_fallback_policy_override=*/true);
    state.matmul_plan = state.kernel_resolution.matmul_plan;
    state.tensor_role = state.kernel_resolution.tensor_role;
    state.semantic_op = state.kernel_resolution.semantic_op;
    state.hybrid_ssm_qkv = state.tensor_role == densecore::runtime::DenseCoreTensorRole::HybridSSMQkv;
    state.hybrid_ssm_gate = state.tensor_role == densecore::runtime::DenseCoreTensorRole::HybridSSMGate;
    state.hybrid_ssm_out = state.tensor_role == densecore::runtime::DenseCoreTensorRole::SSMOut;
    state.hybrid_ssm_projection = densecore::runtime::IsHybridSsmTensorRole(state.tensor_role);
    state.moe_router = state.tensor_role == densecore::runtime::DenseCoreTensorRole::MoERouter;
    state.lm_head_role = state.tensor_role == densecore::runtime::DenseCoreTensorRole::LmHead;
    state.lfm2_shortconv_semantic =
        state.semantic_op == densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer;
    state.fallback_free_target =
        state.requirement.fallback_policy == densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget;
    state.qwen_target = model && densecore::runtime::IsQwenTargetVariant(model->variant);
    state.qwen35_target = model && model->variant == ModelVariant::QWEN35;
    state.qwen36_target = model && model->variant == ModelVariant::QWEN36;
    state.qwen35_hybrid_ssm = state.qwen35_target && state.hybrid_ssm_projection;
    state.qwen36_hybrid_ssm = state.qwen36_target && state.hybrid_ssm_projection;
    state.qwen35_hybrid_ssm_gate = state.qwen35_target && state.hybrid_ssm_gate;
    state.qwen35_hybrid_ssm_out = state.qwen35_target && state.hybrid_ssm_out;
    state.qwen36_hybrid_ssm_qkv = state.qwen36_target && state.hybrid_ssm_qkv;
    state.qwen36_hybrid_ssm_gate = state.qwen36_target && state.hybrid_ssm_gate;
    state.qwen36_hybrid_ssm_out = state.qwen36_target && state.hybrid_ssm_out;
    state.qwen36_lm_head = state.qwen36_target && state.lm_head_role;
    state.lfm2_shortconv_semantic =
        state.lfm2_shortconv_semantic ||
        IsLFM2TargetProjectionWeight(model, weight, model && (model->output == weight || model->output == original_weight));
    return state;
}

static SmartMatmulPrefillProjectionPlan ResolveSmartMatmulPrefillProjectionPlan(
    const SmartMatmulDispatchState& dispatch, const ggml_tensor* weight, const ggml_tensor* input,
    const BatchSpec* current_batch) {
    SmartMatmulPrefillProjectionPlan plan;
    if (!weight || !input) {
        return plan;
    }

    const bool q4k_batched_prefill_targeted =
        densecore::runtime::KernelResolutionTargetsQ4KBatchedPrefill(dispatch.kernel_resolution);
    const bool q4k_batched_prefill_selected =
        densecore::runtime::KernelResolutionSelectsQ4KBatchedPrefill(dispatch.kernel_resolution);
    const bool q4k_batched_prefill_shape_ready =
        q4k_batched_prefill_selected && (input->ne[0] % QK_K == 0);
    const bool q4k_batched_prefill_kernel_ready =
        q4k_batched_prefill_shape_ready && dispatch.host_caps.q4k_true_batched;

    plan.qwen_hybrid_ssm_q8_prefill =
        ((dispatch.qwen35_hybrid_ssm && dispatch.hybrid_ssm_qkv) || dispatch.qwen35_hybrid_ssm_gate ||
         dispatch.qwen36_hybrid_ssm_qkv || dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
        dispatch.prefill_phase && weight->type == GGML_TYPE_Q8_0 && input->type == GGML_TYPE_F32 &&
        dispatch.m > 1 && weight->ne[0] == input->ne[0];
    plan.qwen_hybrid_ssm_quant_prefill_fast_path_eligible =
        (dispatch.hybrid_ssm_qkv || dispatch.qwen35_hybrid_ssm_gate || dispatch.qwen35_hybrid_ssm_out ||
         dispatch.qwen36_hybrid_ssm_qkv || dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
        q4k_batched_prefill_kernel_ready;
    plan.qwen36_hybrid_ssm_q4k_prefill_relevant =
        dispatch.prefill_phase &&
        (dispatch.qwen36_hybrid_ssm_qkv || dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
        dispatch.m > 1 && input->type == GGML_TYPE_F32;
    plan.qwen36_lm_head_q4k_prefill_relevant =
        dispatch.qwen36_lm_head && q4k_batched_prefill_targeted;
    plan.lfm2_prefill_q4k_relevant =
        dispatch.lfm2_shortconv_semantic && q4k_batched_prefill_targeted;
    plan.lfm2_prefill_q6k_relevant =
        dispatch.prefill_phase && dispatch.lfm2_shortconv_semantic && dispatch.m > 1 &&
        input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q6_K;
    plan.lfm2_prefill_quant_nrc_unsafe =
        dispatch.prefill_phase && dispatch.lfm2_shortconv_semantic && dispatch.m > 1 &&
        input->type == GGML_TYPE_F32 && ggml_is_quantized(weight->type) &&
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        true;
#else
        weight->type == GGML_TYPE_Q6_K;
#endif
    plan.q4k_batched_prefill_relevant = plan.qwen36_hybrid_ssm_q4k_prefill_relevant ||
                                        plan.qwen36_lm_head_q4k_prefill_relevant ||
                                        plan.lfm2_prefill_q4k_relevant;
    plan.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k =
        plan.q4k_batched_prefill_relevant && weight->type == GGML_TYPE_Q4_K;
    plan.qwen36_hybrid_ssm_q4k_prefill_shape_supported =
        plan.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k && q4k_batched_prefill_shape_ready;
    plan.qwen36_hybrid_ssm_q4k_prefill_kernel_available = dispatch.host_caps.q4k_true_batched;
    plan.qwen36_hybrid_ssm_q4k_prefill_lora_active = current_batch && !current_batch->lora_map.empty();
    plan.qwen36_hybrid_ssm_q4k_prefill_probe_candidate =
        plan.q4k_batched_prefill_relevant && q4k_batched_prefill_kernel_ready &&
        !plan.qwen36_hybrid_ssm_q4k_prefill_lora_active;
    return plan;
}

static void RecordSmartMatmulGraphCensus(InferenceWorkContext* dispatch_work_ctx,
                                         InferenceExecutionPhase dispatch_phase,
                                         const SmartMatmulOriginalTensorState& original,
                                         const SmartMatmulDispatchState& dispatch,
                                         ggml_type weight_type,
                                         bool matmul_expected_decode,
                                         const char* selected_path) {
    const char* census_path = selected_path;
    if (original.qwen_hybrid_ssm_projection || (dispatch.qwen36_target && original.hybrid_ssm_projection)) {
        census_path = "ssm_projection";
    }
    RecordGraphBuildMatmulCensus(dispatch_work_ctx ? dispatch_work_ctx : GetCurrentWorkContext(), dispatch_phase,
                                 census_path, weight_type, dispatch.m, dispatch.n, dispatch.k, dispatch.weight_name,
                                 dispatch.input_name, matmul_expected_decode);
}

static ggml_tensor* EmitMatmulFromPlan(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                                       TransformerModel* model, InferenceWorkContext* dispatch_work_ctx,
                                       InferenceExecutionPhase dispatch_phase,
                                       const SmartMatmulDispatchState& dispatch,
                                       densecore::runtime::GgmlComputeOp op, const char* reason) {
    InferenceWorkContext* work_ctx = dispatch_work_ctx ? dispatch_work_ctx : GetCurrentWorkContext();
    RecordQwenTargetGgmlComputeFallback(work_ctx, model, op, reason, dispatch.weight_name, dispatch_phase);
    if (dispatch.kernel_resolution.fallback_policy ==
        densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget) {
        throw densecore::InvalidArgumentException(
            std::string("Target fallback-free matmul cannot fall back to GGML: tensor_role=") +
            densecore::runtime::DenseCoreTensorRoleName(dispatch.kernel_resolution.tensor_role) +
            " semantic_op=" +
            densecore::runtime::DenseCoreSemanticOpName(dispatch.kernel_resolution.semantic_op) +
            " reason=" + (reason ? reason : "unknown"));
    }
    return ggml_mul_mat(ctx, weight, input);
}
