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
    const bool prefer_qwen_target_quant_prefill_densecore_path =
        dispatch_is_prefill_phase && original.qwen_target_model && ggml_is_quantized(selection.weight->type) &&
        input->ne[1] > 1 && input->ne[0] == selection.weight->ne[0];

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
            !prefer_qwen_target_quant_prefill_densecore_path &&
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
    plan.dispatch_phase = plan.work_ctx ? plan.work_ctx->phase : GetCurrentExecutionPhase();
    if (plan.dispatch_phase == InferenceExecutionPhase::Unknown) {
        plan.dispatch_phase = GetCurrentExecutionPhase();
    }
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
    const int qwen_ssm_projection_kind = plan->original.hybrid_ssm_projection_kind;
    if (plan->work_ctx && plan->original.qwen_hybrid_ssm_projection && qwen_ssm_projection_kind != 0 &&
        original_weight) {
        RecordQwen36SSMProjectionWeightType(plan->work_ctx, original_weight->type);
    }
    const bool qwen_hybrid_ssm_model =
        (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
        model->arch_flags.is_hybrid_ssm;
    const bool original_qwen_ssm_q8_projection =
        qwen_hybrid_ssm_model && plan->original.hybrid_ssm_projection &&
        qwen_ssm_projection_kind != 0 && original_weight && original_weight->type == GGML_TYPE_Q8_0 &&
        original_weight->ne[0] == input->ne[0];
    if (original_qwen_ssm_q8_projection && plan->original.input_cols <= 1 && plan->weight == original_weight) {
        plan->work_ctx = plan->work_ctx ? plan->work_ctx : GetCurrentWorkContext();
        if (plan->work_ctx) {
            plan->work_ctx->qwen36_profile.qwen36_ssm_q8_decode_used_original_q8_path.store(
                1, std::memory_order_relaxed);
            const Qwen36SSMQ8PrefillAMXRejectReason decode_alias_reason =
                ResolveQwen36SSMQ8PrefillAMXReason(fast_path_config.qwen36_ssm_q8_prefill_amx,
                                                   plan->dispatch_phase,
                                                   plan->current_batch && !plan->current_batch->lora_map.empty());
            if (decode_alias_reason != Qwen36SSMQ8PrefillAMXRejectReason::EnvOff) {
                RecordQwen36SSMQ8PrefillAMXReject(plan->work_ctx, decode_alias_reason);
            }
        }
    }
    if (!plan->dispatch_is_prefill_phase || !original_qwen_ssm_q8_projection || plan->original.input_cols <= 1) {
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
        if (q8_prefill_gate != Qwen36SSMQ8PrefillAMXRejectReason::EnvOff) {
            reject_q8_prefill(q8_prefill_gate, false);
        }
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
        RecordQwen36SSMQ8PrefillAMXUsed(plan->work_ctx, qwen_ssm_projection_kind);
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
    bool qwen36_lm_head_q4k_prefill_fast_path_eligible = false;
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

struct SmartMatmulQuantCapabilityPlan {
    bool has_quant_vec_dot = false;
    bool has_quant_from_float = false;
    bool quant_input_size_ok = false;
    bool quant_nrc_batch_ready = false;
    bool quant_true_batched_kernel_ready = false;
};

struct SmartMatmulQ4KBatchedAdmissionPlan {
    uint64_t admission_key = 0;
    Qwen36Q4KBatchedAdmissionValue admission{};
    Qwen36PrefillQ4KBatchedRejectReason reject_reason = Qwen36PrefillQ4KBatchedRejectReason::None;
    bool mode_off = false;
    bool mode_on = false;
    bool mode_probe = false;
    bool probe_rejected = false;
    bool candidate_ready = false;
    bool probe_admitted = false;
    bool qwen36_prefers_ggml_quant = false;
    bool lfm2_prefers_ggml_quant = false;
    bool lfm2_q4k_true_batched_admitted = false;
};

struct SmartMatmulAliasRoutingPlan {
    bool defer_cpu_repack_alias_to_batched_q4k = false;
    bool use_cpu_repack_alias_fallback = false;
};

struct SmartMatmulGemma4NativePlan {
    bool dense_prefill_candidate = false;
    bool dense_prefill_allowed = false;
    Gemma4NativeMatmulReject dense_prefill_reject = Gemma4NativeMatmulReject::None;
    bool decode_candidate = false;
    bool decode_allowed = false;
    Gemma4NativeMatmulReject decode_reject = Gemma4NativeMatmulReject::None;
};

struct SmartMatmulLfm2PrefillCompatibilityPlan {
    bool q5k_q6k_or_q8_relevant = false;
    bool force_non_q4k_reference_on_arm = false;
};

struct SmartMatmulGgmlCompatibilityPlan {
    bool plain_hybrid_ssm_prefill = false;
    bool qwen36_hybrid_ssm_q4k_native_prefill = false;
    bool qwen36_lm_head_prefill = false;
};

struct SmartMatmulGemvUserDataPlan {
    bool gemma4_model = false;
    bool gemma4_maintained_q8_decode_repacked = false;
    bool lfm2_tied_lm_head = false;
    bool qwen_c4a_q6k_lm_head = false;
};

struct SmartMatmulCandidatePlan {
    bool gemv = false;
    bool small_batch_f32 = false;
    bool moe_router_f32_prefill = false;
    bool small_batch_quant = false;
    bool small_batch = false;
    bool true_batched_q4k = false;
    const char* quant_reject_reason = "unknown";
};

struct SmartMatmulGgmlEmitPlan {
    bool active = false;
    const char* type_label = nullptr;
    const char* selected_path = "GGML_NATIVE";
    const char* dispatch_reason = nullptr;
    const char* hybrid_path = nullptr;
    const char* graph_path = "ggml_mul_mat";
    const char* emit_reason = "temporary_reference_generic_matmul_fallback";
    bool log_hybrid = false;
};

struct SmartMatmulPathPlan {
    SmartMatmulPrefillProjectionPlan prefill{};
    SmartMatmulQuantCapabilityPlan quant{};
    SmartMatmulQ4KBatchedAdmissionPlan q4k{};
    SmartMatmulAliasRoutingPlan alias{};
    SmartMatmulGemma4NativePlan gemma4_native{};
    SmartMatmulCandidatePlan candidate{};
    SmartMatmulGgmlEmitPlan ggml_emit{};
    Qwen36PrefillQ4KBatchedRejectReason q4k_reject_reason = Qwen36PrefillQ4KBatchedRejectReason::None;
};

constexpr int kMaxSmallBatchQuantCols = 4096;

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
    state.kernel_resolution = densecore::models::ResolveModelTensorKernelResolution(
        model, state.requirement, weight ? weight->type : GGML_TYPE_COUNT, input ? input->type : GGML_TYPE_COUNT,
        state.m, state.n, state.k, state.plan_phase, state.weight_name, model && model->output == weight,
        state.compatible, state.host_caps);
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
        state.kernel_resolution.fallback_policy == densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget;
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
         dispatch.qwen35_hybrid_ssm_out || dispatch.qwen36_hybrid_ssm_qkv ||
         dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
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
    if (plan.qwen36_lm_head_q4k_prefill_relevant && weight->type == GGML_TYPE_Q4_K &&
        input->type == GGML_TYPE_F32 && weight->ne[0] == input->ne[0] && dispatch.host_caps.q4k_true_batched) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        plan.qwen36_lm_head_q4k_prefill_fast_path_eligible =
            type_traits_cpu && type_traits_cpu->vec_dot && type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_K &&
            (input->ne[0] % QK_K == 0);
    }
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

static SmartMatmulQuantCapabilityPlan ResolveSmartMatmulQuantCapabilityPlan(
    const ggml_tensor* weight, const ggml_tensor* input, int input_cols) {
    SmartMatmulQuantCapabilityPlan plan;
    if (!weight || !input || !ggml_is_quantized(weight->type) || input->type != GGML_TYPE_F32) {
        return plan;
    }
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
    if (!type_traits_cpu || !type_traits_cpu->vec_dot) {
        return plan;
    }

    plan.has_quant_vec_dot = true;
    const auto* input_type_traits = ggml_get_type_traits_cpu(type_traits_cpu->vec_dot_type);
    if (input_type_traits && input_type_traits->from_float) {
        plan.has_quant_from_float = true;
        const size_t quant_row_size = ggml_row_size(type_traits_cpu->vec_dot_type, input->ne[0]);
        plan.quant_input_size_ok = quant_row_size > 0 && quant_row_size <= kMaxQuantInputBufferSize;
    }
    const int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
    plan.quant_nrc_batch_ready = vec_dot_nrows >= std::min(input_cols, kMaxSmallBatchColsHard);
    plan.quant_true_batched_kernel_ready = IsQ4KTrueBatchedKernelEnabled() && weight->type == GGML_TYPE_Q4_K &&
                                           type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_K &&
                                           (input->ne[0] % QK_K == 0);
    return plan;
}

static SmartMatmulQ4KBatchedAdmissionPlan ResolveSmartMatmulQ4KBatchedAdmissionPlan(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input,
    const SmartMatmulDispatchState& dispatch, const SmartMatmulPrefillProjectionPlan& prefill,
    const SmartMatmulQuantCapabilityPlan& quant,
    const densecore::llm::config::FastPathRuntimeConfig& fast_path_config) {
    SmartMatmulQ4KBatchedAdmissionPlan plan;
    if (!weight || !input) {
        return plan;
    }

    plan.admission_key = prefill.qwen36_hybrid_ssm_q4k_prefill_probe_candidate
                             ? HashQwen36Q4KBatchedAdmissionKey(model, weight, input, dispatch.m, dispatch.n,
                                                               dispatch.k)
                             : 0;
    plan.admission = plan.admission_key ? LookupQwen36Q4KBatchedAdmission(plan.admission_key)
                                        : Qwen36Q4KBatchedAdmissionValue{};
    const bool qwen36_q4k_relevant =
        prefill.qwen36_hybrid_ssm_q4k_prefill_relevant ||
        prefill.qwen36_lm_head_q4k_prefill_relevant;
    plan.mode_off = qwen36_q4k_relevant &&
                    fast_path_config.qwen36_prefill_q4k_batched ==
                        densecore::llm::config::Qwen36PrefillQ4KBatchedMode::Off;
    plan.mode_on = qwen36_q4k_relevant &&
                   fast_path_config.qwen36_prefill_q4k_batched ==
                       densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On;
    // Qwen3.6 and LFM2 target prefill Q4_K go directly to the DenseCore
    // true-batched kernel by default. "probe"/"auto" are parsed as On now; the
    // probe enum remains only for older diagnostic callers that construct the
    // config directly.
    plan.mode_probe =
        (qwen36_q4k_relevant &&
         fast_path_config.qwen36_prefill_q4k_batched ==
             densecore::llm::config::Qwen36PrefillQ4KBatchedMode::Probe) ||
        prefill.lfm2_prefill_q4k_relevant;
    plan.probe_rejected =
        plan.mode_probe && plan.admission.state == Qwen36Q4KBatchedAdmissionState::Reject;
    plan.candidate_ready = prefill.q4k_batched_prefill_relevant && quant.has_quant_vec_dot &&
                           quant.has_quant_from_float && quant.quant_input_size_ok &&
                           quant.quant_true_batched_kernel_ready &&
                           !prefill.qwen36_hybrid_ssm_q4k_prefill_lora_active;
    plan.reject_reason = ResolveQwen36PrefillQ4KBatchedReason(
        prefill.q4k_batched_prefill_relevant, plan.mode_off,
        prefill.qwen36_hybrid_ssm_q4k_prefill_lora_active,
        prefill.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k,
        prefill.qwen36_hybrid_ssm_q4k_prefill_shape_supported,
        prefill.qwen36_hybrid_ssm_q4k_prefill_kernel_available, quant.has_quant_vec_dot,
        plan.candidate_ready, plan.mode_on, plan.mode_probe, plan.admission.state);
    plan.probe_admitted = plan.candidate_ready &&
                          (plan.mode_on || prefill.lfm2_prefill_q4k_relevant ||
                           (plan.mode_probe && plan.admission.state != Qwen36Q4KBatchedAdmissionState::Reject));
    plan.qwen36_prefers_ggml_quant =
        dispatch.prefill_phase && dispatch.qwen36_hybrid_ssm && dispatch.m > 1 &&
        !prefill.qwen_hybrid_ssm_q8_prefill &&
        (!plan.probe_admitted || plan.mode_off || plan.probe_rejected);
    plan.lfm2_prefers_ggml_quant =
        dispatch.prefill_phase && dispatch.lfm2_shortconv_semantic && dispatch.m > 1 &&
        weight->type == GGML_TYPE_Q4_K && (!plan.probe_admitted || plan.probe_rejected);
    plan.lfm2_q4k_true_batched_admitted =
        prefill.lfm2_prefill_q4k_relevant && dispatch.compatible &&
        dispatch.m <= kMaxSmallBatchQuantCols && plan.candidate_ready && plan.probe_admitted;
    return plan;
}

static SmartMatmulGemma4NativePlan ResolveSmartMatmulGemma4NativePlan(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input, const BatchSpec* batch,
    const SmartMatmulDispatchState& dispatch, InferenceExecutionPhase dispatch_phase) {
    SmartMatmulGemma4NativePlan plan;
    if (!weight || !input) {
        return plan;
    }
    plan.dense_prefill_candidate =
        model && model->arch_flags.is_gemma4 && input->type == GGML_TYPE_F32 && dispatch.m > 1 &&
        ggml_is_quantized(weight->type) && IsGemma4DenseWeightNameCandidate(dispatch.weight_name,
                                                                            dispatch.gemma4_lm_head);
    plan.dense_prefill_allowed =
        CanUseGemma4DensePrefillNative(model, weight, input, batch, &plan.dense_prefill_reject);
    plan.decode_allowed =
        CanUseGemma4DecodeNative(model, weight, input, batch, dispatch_phase, &plan.decode_reject);
    plan.decode_candidate =
        model && model->arch_flags.is_gemma4 && input->type == GGML_TYPE_F32 && dispatch.m <= 1 &&
        (weight->type == GGML_TYPE_F32 || ggml_is_quantized(weight->type)) &&
        IsGemma4DenseWeightNameCandidate(dispatch.weight_name, dispatch.gemma4_lm_head);
    return plan;
}

static void RecordSmartMatmulGemma4NativeCandidates(InferenceWorkContext* work_ctx, const ggml_tensor* weight,
                                                    const SmartMatmulDispatchState& dispatch,
                                                    const SmartMatmulGemma4NativePlan& plan) {
    if (!weight) {
        return;
    }
    InferenceWorkContext* ctx = work_ctx ? work_ctx : GetCurrentWorkContext();
    if (plan.dense_prefill_candidate) {
        RecordGemma4DensePrefillNativeDecision(
            ctx, /*candidate=*/true, /*used=*/false,
            plan.dense_prefill_allowed ? nullptr : Gemma4NativeMatmulRejectName(plan.dense_prefill_reject),
            weight->type, 0, false);
    }
    if (plan.decode_candidate) {
        RecordGemma4DecodeNativeDecision(
            ctx, /*candidate=*/true, /*used=*/false,
            plan.decode_allowed ? nullptr : Gemma4NativeMatmulRejectName(plan.decode_reject),
            /*moe_used=*/false, /*dense_used=*/false, /*lm_head_used=*/dispatch.gemma4_lm_head, 0, 0, 0, false);
    }
}

static SmartMatmulLfm2PrefillCompatibilityPlan ResolveSmartMatmulLfm2PrefillCompatibilityPlan(
    const ggml_tensor* weight, const ggml_tensor* input, const SmartMatmulDispatchState& dispatch) {
    SmartMatmulLfm2PrefillCompatibilityPlan plan;
    if (!weight || !input) {
        return plan;
    }
    plan.q5k_q6k_or_q8_relevant =
        dispatch.prefill_phase && dispatch.lfm2_shortconv_semantic && input->type == GGML_TYPE_F32 &&
        dispatch.m > 1 &&
        (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_Q6_K || weight->type == GGML_TYPE_Q8_0);
    plan.force_non_q4k_reference_on_arm =
        dispatch.prefill_phase && dispatch.lfm2_shortconv_semantic && ggml_is_quantized(weight->type) &&
        weight->type != GGML_TYPE_Q4_K && !plan.q5k_q6k_or_q8_relevant && input->type == GGML_TYPE_F32 &&
        dispatch.m > 1 &&
#if defined(__aarch64__) || defined(_M_ARM64)
        true;
#else
        false;
#endif
    return plan;
}

static SmartMatmulAliasRoutingPlan ResolveSmartMatmulAliasRoutingPlan(const ggml_tensor* weight,
                                                                      const ggml_tensor* input,
                                                                      const SmartMatmulDispatchState& dispatch,
                                                                      bool using_cpu_repack_alias) {
    SmartMatmulAliasRoutingPlan plan;
    if (!weight || !input || !using_cpu_repack_alias) {
        return plan;
    }
    plan.defer_cpu_repack_alias_to_batched_q4k = dispatch.m > 1 && input->type == GGML_TYPE_F32 &&
                                                 weight->type == GGML_TYPE_Q4_K && dispatch.compatible &&
                                                 dispatch.host_caps.q4k_true_batched;
    plan.use_cpu_repack_alias_fallback = !plan.defer_cpu_repack_alias_to_batched_q4k;
    return plan;
}

static SmartMatmulGgmlCompatibilityPlan ResolveSmartMatmulGgmlCompatibilityPlan(
    const ggml_tensor* weight, const ggml_tensor* input, const SmartMatmulDispatchState& dispatch,
    const SmartMatmulPrefillProjectionPlan& prefill, const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission) {
    SmartMatmulGgmlCompatibilityPlan plan;
    if (!weight || !input) {
        return plan;
    }
    plan.qwen36_hybrid_ssm_q4k_native_prefill =
        (dispatch.qwen36_hybrid_ssm_qkv || dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
        prefill.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k && q4k_admission.mode_off;
    plan.plain_hybrid_ssm_prefill =
        dispatch.prefill_phase &&
        ((dispatch.hybrid_ssm_qkv && !prefill.qwen_hybrid_ssm_quant_prefill_fast_path_eligible &&
          !dispatch.qwen_target) ||
         dispatch.qwen36_hybrid_ssm_qkv || dispatch.qwen36_hybrid_ssm_gate || dispatch.qwen36_hybrid_ssm_out) &&
        !prefill.qwen_hybrid_ssm_q8_prefill &&
        (!prefill.qwen_hybrid_ssm_quant_prefill_fast_path_eligible ||
         plan.qwen36_hybrid_ssm_q4k_native_prefill) &&
        ggml_is_quantized(weight->type) && input->type == GGML_TYPE_F32 && dispatch.m > 1;
    plan.qwen36_lm_head_prefill =
        dispatch.prefill_phase && dispatch.qwen36_lm_head && ggml_is_quantized(weight->type) && dispatch.m > 1 &&
        !prefill.qwen36_lm_head_q4k_prefill_fast_path_eligible;
    return plan;
}

static const char* ResolveSmartMatmulQuantRejectReason(const SmartMatmulQuantCapabilityPlan& quant,
                                                       const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission,
                                                       int input_cols) {
    if (input_cols > kMaxSmallBatchQuantCols) {
        return "M>max_quant_cols";
    }
    if (q4k_admission.qwen36_prefers_ggml_quant) {
        return "qwen36_prefill_ggml_quant";
    }
    if (q4k_admission.lfm2_prefers_ggml_quant) {
        return "lfm2_prefill_q4k_probe_rejected";
    }
    if (!quant.has_quant_vec_dot) {
        return "no_vec_dot";
    }
    if (!quant.has_quant_from_float) {
        return "no_from_float";
    }
    if (!quant.quant_input_size_ok) {
        return "quant_input_too_large";
    }
    if (!quant.quant_nrc_batch_ready && !quant.quant_true_batched_kernel_ready) {
        return "quant_batched_kernel_unavailable";
    }
    return "unknown";
}

static SmartMatmulCandidatePlan ResolveSmartMatmulCandidatePlan(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input,
    const SmartMatmulDispatchState& dispatch, const SmartMatmulPrefillProjectionPlan& prefill,
    const SmartMatmulQuantCapabilityPlan& quant, const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission) {
    SmartMatmulCandidatePlan plan;
    if (!weight || !input) {
        return plan;
    }
    plan.gemv = dispatch.m == 1 && (weight->type == GGML_TYPE_F32 || ggml_is_quantized(weight->type));
    plan.small_batch_f32 = dispatch.m > 1 && dispatch.m <= kMaxSmallBatchColsHard &&
                           input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32;
    plan.moe_router_f32_prefill =
        model && dispatch.m > 1 && input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 &&
        dispatch.moe_router && weight->ne[0] == input->ne[0] && weight->ne[1] <= 512 &&
        weight->ne[0] >= 512 && dispatch.fallback_free_target;
    plan.small_batch_quant =
        dispatch.m > 1 && (dispatch.m <= kMaxSmallBatchQuantCols || prefill.lfm2_prefill_q6k_relevant) &&
        input->type == GGML_TYPE_F32 && ggml_is_quantized(weight->type) &&
        !q4k_admission.qwen36_prefers_ggml_quant && !q4k_admission.lfm2_prefers_ggml_quant &&
        quant.has_quant_vec_dot && quant.has_quant_from_float && quant.quant_input_size_ok;
    plan.small_batch = plan.small_batch_f32 || plan.moe_router_f32_prefill || plan.small_batch_quant;
    plan.true_batched_q4k = plan.small_batch_quant && quant.quant_true_batched_kernel_ready &&
                            weight->type == GGML_TYPE_Q4_K;
    plan.quant_reject_reason = ResolveSmartMatmulQuantRejectReason(quant, q4k_admission, dispatch.m);
    return plan;
}

static SmartMatmulGgmlEmitPlan ResolveSmartMatmulGgmlEmitPlan(
    const SmartMatmulAliasRoutingPlan& alias, const SmartMatmulGgmlCompatibilityPlan& compatibility,
    const SmartMatmulLfm2PrefillCompatibilityPlan& lfm2_compatibility,
    bool force_plain_hybrid_ssm_qkv, bool using_qwen36_ssm_q8_prefill_amx_alias) {
    SmartMatmulGgmlEmitPlan plan;
    if (force_plain_hybrid_ssm_qkv) {
        plan.active = true;
        plan.type_label = "PLAIN_GGML";
        plan.selected_path = "CONSERVATIVE_FALLBACK";
        plan.hybrid_path = "PLAIN_GGML_CONSERVATIVE_FALLBACK";
        plan.emit_reason = "temporary_reference_forced_hybrid_ssm_qkv";
        plan.log_hybrid = true;
        return plan;
    }
    if (alias.use_cpu_repack_alias_fallback) {
        plan.active = true;
        plan.selected_path = "GGML_CPU_REPACK";
        plan.emit_reason = "temporary_reference_cpu_repack_alias";
        return plan;
    }
    if (using_qwen36_ssm_q8_prefill_amx_alias) {
        plan.active = true;
        plan.selected_path = "DENSECORE_AMX_Q8_PREFILL";
        plan.dispatch_reason = "qwen36_ssm_q8_prefill_amx";
        plan.graph_path = "ssm_projection";
        plan.emit_reason = "maintained_qwen36_ssm_q8_amx_prefill";
        return plan;
    }
    if (compatibility.plain_hybrid_ssm_prefill) {
        plan.active = true;
        plan.selected_path = "GGML_NATIVE";
        plan.dispatch_reason = "hybrid_ssm_prefill_correctness";
        plan.emit_reason = "temporary_reference_hybrid_ssm_prefill_correctness";
        return plan;
    }
    if (lfm2_compatibility.force_non_q4k_reference_on_arm) {
        plan.active = true;
        plan.selected_path = "GGML_NATIVE";
        plan.dispatch_reason = "lfm2_arm_prefill_non_q4k_reference";
        plan.emit_reason = "temporary_reference_lfm2_arm_prefill_non_q4k";
        return plan;
    }
    if (compatibility.qwen36_lm_head_prefill) {
        plan.active = true;
        plan.selected_path = "GGML_NATIVE";
        plan.dispatch_reason = "qwen36_lm_head_prefill_correctness";
        plan.emit_reason = "temporary_reference_qwen36_lm_head_prefill_correctness";
        return plan;
    }
    return plan;
}

static Qwen36PrefillQ4KBatchedRejectReason ResolveSmartMatmulEffectiveQ4KRejectReason(
    const SmartMatmulPrefillProjectionPlan& prefill, const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission) {
    if (prefill.qwen36_hybrid_ssm_q4k_prefill_relevant &&
        !prefill.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k) {
        return Qwen36PrefillQ4KBatchedRejectReason::NotQ4K;
    }
    return q4k_admission.reject_reason;
}

static const char* ResolveSmartMatmulLfm2Q4KRequiredRejectReason(
    bool compatible, int input_cols, const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission,
    Qwen36PrefillQ4KBatchedRejectReason reject_reason) {
    const char* reason = Qwen36PrefillQ4KBatchedRejectReasonName(static_cast<int>(reject_reason));
    if (reject_reason != Qwen36PrefillQ4KBatchedRejectReason::None &&
        reject_reason != Qwen36PrefillQ4KBatchedRejectReason::Admitted) {
        return reason;
    }
    if (!compatible) {
        return "incompatible_shape";
    }
    if (input_cols > kMaxSmallBatchQuantCols) {
        return "M>max_quant_cols";
    }
    if (!q4k_admission.candidate_ready) {
        return "candidate_not_ready";
    }
    return "not_admitted";
}

static SmartMatmulPathPlan ResolveSmartMatmulPathPlan(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input,
    const BatchSpec* current_batch, const SmartMatmulDispatchState& dispatch,
    const densecore::llm::config::FastPathRuntimeConfig& fast_path_config, bool using_cpu_repack_alias,
    InferenceExecutionPhase dispatch_phase, bool force_plain_hybrid_ssm_qkv,
    bool using_qwen36_ssm_q8_prefill_amx_alias) {
    SmartMatmulPathPlan plan;
    plan.prefill = ResolveSmartMatmulPrefillProjectionPlan(dispatch, weight, input, current_batch);
    plan.quant = ResolveSmartMatmulQuantCapabilityPlan(weight, input, dispatch.m);
    plan.q4k = ResolveSmartMatmulQ4KBatchedAdmissionPlan(model, weight, input, dispatch, plan.prefill, plan.quant,
                                                         fast_path_config);
    plan.q4k_reject_reason = ResolveSmartMatmulEffectiveQ4KRejectReason(plan.prefill, plan.q4k);
    plan.alias = ResolveSmartMatmulAliasRoutingPlan(weight, input, dispatch, using_cpu_repack_alias);
    plan.gemma4_native = ResolveSmartMatmulGemma4NativePlan(model, weight, input, current_batch, dispatch,
                                                            dispatch_phase);
    const SmartMatmulLfm2PrefillCompatibilityPlan lfm2_compatibility =
        ResolveSmartMatmulLfm2PrefillCompatibilityPlan(weight, input, dispatch);
    const SmartMatmulGgmlCompatibilityPlan ggml_compatibility =
        ResolveSmartMatmulGgmlCompatibilityPlan(weight, input, dispatch, plan.prefill, plan.q4k);
    plan.candidate = ResolveSmartMatmulCandidatePlan(model, weight, input, dispatch, plan.prefill, plan.quant,
                                                     plan.q4k);
    plan.ggml_emit = ResolveSmartMatmulGgmlEmitPlan(plan.alias, ggml_compatibility, lfm2_compatibility,
                                                    force_plain_hybrid_ssm_qkv,
                                                    using_qwen36_ssm_q8_prefill_amx_alias);
    return plan;
}

static void RecordSmartMatmulQ4KBatchedRejectReason(InferenceWorkContext* work_ctx,
                                                    Qwen36PrefillQ4KBatchedRejectReason reason) {
    if (!work_ctx || reason == Qwen36PrefillQ4KBatchedRejectReason::None) {
        return;
    }
    work_ctx->qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason.store(
        static_cast<int>(reason), std::memory_order_relaxed);
}

static bool ShouldUseQwenHybridSSMQ8RepackedBatched(bool qwen_hybrid_ssm_q8_prefill, int tokens) {
    return qwen_hybrid_ssm_q8_prefill && tokens >= 4;
}

static bool ShouldUseQwenHybridSSMQ8DirectBatched(bool qwen_hybrid_ssm_q8_prefill, int tokens) {
    return qwen_hybrid_ssm_q8_prefill && tokens > 0 && tokens < 4;
}

static void ConfigureSmartMatmulBatchedUserData(GemvBatchedUserData* ud, const ggml_tensor* weight,
                                                const SmartMatmulDispatchState& dispatch,
                                                const SmartMatmulPrefillProjectionPlan& prefill,
                                                const SmartMatmulQ4KBatchedAdmissionPlan& q4k_admission,
                                                bool gemma4_dense_prefill_native,
                                                bool gemma4_prefill_safe_batched) {
    if (!ud || !weight) {
        return;
    }
    ud->force_reference_scalar = false;
    ud->qwen36_prefill_q4k_admission_key = q4k_admission.admission_key;
    ud->qwen36_prefill_q4k_probe =
        !prefill.lfm2_prefill_q4k_relevant && q4k_admission.mode_probe &&
        !prefill.qwen36_lm_head_q4k_prefill_relevant &&
        q4k_admission.admission.state == Qwen36Q4KBatchedAdmissionState::Unknown;
    ud->qwen36_prefill_q4k_admitted = q4k_admission.probe_admitted;
    ud->require_q4k_true_batched = prefill.lfm2_prefill_q4k_relevant ||
                                   prefill.qwen36_hybrid_ssm_q4k_prefill_weight_is_q4k ||
                                   prefill.qwen36_lm_head_q4k_prefill_relevant;
    ud->gemma4_prefill_safe_batched = gemma4_prefill_safe_batched;
    ud->disable_quant_nrc_fast = prefill.lfm2_prefill_quant_nrc_unsafe ||
                                 prefill.lfm2_prefill_q4k_relevant ||
                                 gemma4_prefill_safe_batched;
    ud->gemma4_dense_prefill_native = gemma4_dense_prefill_native;
    ud->lfm2_q8_repacked_batched =
        dispatch.lfm2_shortconv_semantic && weight->type == GGML_TYPE_Q8_0 && dispatch.m > 1;
    // Qwen hybrid-SSM Q8_0 prefill must stay inside DenseCore's owned path.
    // Batches of four or more use DenseCore's ISA-specific 4x8 GEMM. Tiny
    // batches keep row-GEMV because activation packing cannot amortize there.
    ud->qwen36_ssm_q8_repacked_batched =
        ShouldUseQwenHybridSSMQ8RepackedBatched(prefill.qwen_hybrid_ssm_q8_prefill, dispatch.m);
    ud->qwen36_ssm_q8_direct_batched =
        ShouldUseQwenHybridSSMQ8DirectBatched(prefill.qwen_hybrid_ssm_q8_prefill, dispatch.m);
}

static SmartMatmulGemvUserDataPlan ResolveSmartMatmulGemvUserDataPlan(
    const TransformerModel* model, const ggml_tensor* weight, const ggml_tensor* input,
    const SmartMatmulDispatchState& dispatch) {
    SmartMatmulGemvUserDataPlan plan;
    if (!weight || !input) {
        return plan;
    }
    plan.gemma4_model = model && model->arch_flags.is_gemma4;
    plan.gemma4_maintained_q8_decode_repacked =
        plan.gemma4_model && weight->type == GGML_TYPE_Q8_0 && input->type == GGML_TYPE_F32 && dispatch.m == 1 &&
        (dispatch.gemma4_lm_head || IsGemma4SharedDenseFfnWeightName(dispatch.weight_name));
    plan.lfm2_tied_lm_head =
        model && model->arch_flags.is_lfm2_shortconv &&
        (model->output == weight || std::strcmp(dispatch.weight_name, "token_embd.weight") == 0 ||
         std::strcmp(dispatch.weight_name, "output.weight") == 0 ||
         std::strcmp(dispatch.weight_name, "output") == 0);
    plan.qwen_c4a_q6k_lm_head =
#if defined(__aarch64__) || defined(_M_ARM64)
        dispatch.qwen35_target && weight->type == GGML_TYPE_Q6_K &&
        (model->output == weight || std::strcmp(dispatch.weight_name, "output.weight") == 0 ||
         std::strcmp(dispatch.weight_name, "lm_head.weight") == 0 || std::strcmp(dispatch.weight_name, "output") == 0);
#else
        false;
#endif
    return plan;
}

static void ConfigureSmartMatmulGemvUserData(GemvUserData* ud, const TransformerModel* model,
                                             const SmartMatmulDispatchState& dispatch,
                                             const SmartMatmulGemma4NativePlan& gemma4_plan,
                                             const SmartMatmulGemvUserDataPlan& gemv_plan,
                                             bool matmul_expected_decode) {
    if (!ud) {
        return;
    }
    ud->force_reference_scalar = false;
    ud->model_identity = reinterpret_cast<uintptr_t>(model);
    ud->force_q8_repacked_gemv = gemv_plan.gemma4_maintained_q8_decode_repacked;
    ud->disable_q8_repacked_gemv = gemv_plan.gemma4_model && !gemv_plan.gemma4_maintained_q8_decode_repacked;
    ud->gemma4_decode_native = gemma4_plan.decode_allowed;
    ud->gemma4_decode_lm_head = dispatch.gemma4_lm_head;
    ud->semantic_op = dispatch.semantic_op;
    ud->tensor_role = dispatch.tensor_role;
    ud->lfm2_decode_lm_head =
        (gemv_plan.lfm2_tied_lm_head || gemv_plan.qwen_c4a_q6k_lm_head) && matmul_expected_decode;
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
    const bool maintained_qwen_ssm_q8_amx_prefill =
        reason && std::strcmp(reason, "maintained_qwen36_ssm_q8_amx_prefill") == 0 && model &&
        (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
        model->arch_flags.is_hybrid_ssm && dispatch_phase == InferenceExecutionPhase::Prefill &&
        (dispatch.qwen35_hybrid_ssm || dispatch.qwen36_hybrid_ssm) &&
        dispatch.hybrid_ssm_projection && weight && model->cpu_amx_aliases.find(weight) != model->cpu_amx_aliases.end();
    if (maintained_qwen_ssm_q8_amx_prefill) {
        return ggml_mul_mat(ctx, weight, input);
    }
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
