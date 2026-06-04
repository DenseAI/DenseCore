// ============================================================================
// EXPLICIT INFERENCE WORK CONTEXT (per-thread, no implicit TLS pools)
// ============================================================================
struct Qwen36ProfileCounters {
    std::atomic<uint64_t> attention_ns{0};
    std::atomic<uint64_t> paged_attention_ns{0};
    std::atomic<uint64_t> standard_attention_ns{0};
    std::atomic<uint64_t> portable_flash_attention_ns{0};
    std::atomic<uint64_t> native_flash_attention_ns{0};
    std::atomic<uint64_t> hal_attention_ns{0};
    std::atomic<uint64_t> attention_repack_ns{0};
    std::atomic<uint64_t> moe_forward_ns{0};
    std::atomic<uint64_t> moe_route_ns{0};
    std::atomic<uint64_t> moe_reorder_ns{0};
    std::atomic<uint64_t> moe_expert_ns{0};
    std::atomic<uint64_t> moe_reduce_ns{0};
    std::atomic<uint64_t> moe_w1w3_ns{0};
    std::atomic<uint64_t> moe_w2_ns{0};
    std::atomic<uint64_t> moe_rowblock_ns{0};
    std::atomic<uint64_t> moe_rowblock_w1w3_ns{0};
    std::atomic<uint64_t> moe_rowblock_w2_ns{0};
    std::atomic<uint64_t> shared_expert_ns{0};
    std::atomic<uint64_t> quant_matmul_ns{0};
    std::atomic<uint64_t> ssm_qkv_wall_ns{0};
    std::atomic<uint64_t> ssm_gate_wall_ns{0};
    std::atomic<uint64_t> ssm_delta_wall_ns{0};
    std::atomic<uint64_t> ssm_out_wall_ns{0};
    std::atomic<uint64_t> ssm_conv1d_ns{0};
    std::atomic<uint64_t> ssm_delta_ns{0};
    std::atomic<uint64_t> kv_update_ns{0};
    std::atomic<uint64_t> sample_ns{0};
    std::atomic<uint64_t> kleidiai_candidate_ops{0};
    std::atomic<uint64_t> kleidiai_allowed_ops{0};
    std::atomic<uint64_t> kleidiai_rejected_ops{0};
    std::atomic<uint64_t> graph_cache_hits{0};
    std::atomic<uint64_t> graph_cache_misses{0};
    std::atomic<uint64_t> q4k_repacked_gemv_cache_hits{0};
    std::atomic<uint64_t> q4k_repacked_gemv_cache_waited_hits{0};
    std::atomic<uint64_t> q4k_repacked_gemv_cache_misses{0};
    std::atomic<uint64_t> q4k_repacked_gemv_cache_evictions{0};
    std::atomic<uint64_t> q4k_repacked_gemv_cache_evicted_bytes{0};
    std::atomic<uint64_t> q4k_repacked_gemv_repack_bytes{0};
    std::atomic<uint64_t> q4k_repacked_gemv_probe_ns{0};
    std::atomic<uint64_t> q4k_repacked_gemv_resident_bytes{0};
    std::atomic<uint64_t> q4k_repacked_gemv_distinct_weights_seen{0};
    std::atomic<uint64_t> q4k_repacked_gemv_repeated_repack_count{0};
    std::atomic<uint64_t> q4k_copied_gemv_experiment_cache_hits{0};
    std::atomic<uint64_t> q4k_copied_gemv_experiment_cache_misses{0};
    std::atomic<uint64_t> qact_cache_hits{0};
    std::atomic<uint64_t> qact_cache_misses{0};
    std::atomic<uint64_t> qact_cache_reused_bytes{0};
    std::atomic<uint64_t> moe_decode_scratch_reused{0};
    std::atomic<uint64_t> moe_decode_allocations_avoided{0};
    std::atomic<int> moe_task_count{0};
    std::atomic<int> moe_rowblock_used{0};
    std::atomic<int> moe_rowblock_tasks{0};
    std::atomic<int> selected_expert_count{0};
    std::atomic<int> ssm_conv1d_calls{0};
    std::atomic<int> ssm_delta_calls{0};
    std::atomic<int> q4k_true_batched_used{0};
    std::atomic<int> qwen36_prefill_q4k_batched_mode{1};
    std::atomic<int> qwen36_prefill_q4k_batched_used{0};
    std::atomic<int> qwen36_prefill_q4k_batched_probe_pass{0};
    std::atomic<uint32_t> qwen36_prefill_q4k_batched_max_abs_error_bits{0};
    std::atomic<int> qwen36_prefill_q4k_batched_last_reject_reason{0};
    std::atomic<uint64_t> qwen36_prefill_q4k_probe_participants{0};
    std::atomic<uint64_t> qwen36_prefill_q4k_probe_failures{0};
    std::atomic<uint64_t> qwen36_prefill_q4k_admission_downgraded{0};
    std::atomic<int> qwen36_ssm_q8_prefill_amx_mode{0};
    std::atomic<int> qwen36_ssm_q8_prefill_amx_prepared{0};
    std::atomic<int> qwen36_ssm_q8_prefill_amx_used{0};
    std::atomic<int> qwen36_ssm_q8_prefill_amx_last_reject_reason{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_qkv_count{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_gate_count{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_out_count{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_candidate_ops{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_used_ops{0};
    std::atomic<uint64_t> qwen36_ssm_q8_prefill_amx_rejected_ops{0};
    std::atomic<int> qwen36_ssm_q8_decode_used_original_q8_path{0};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> qwen36_ssm_projection_weight_type_hist{};
    std::atomic<int> q4k_repacked_gemv_used{0};
    std::atomic<uint64_t> q4k_repacked_gemv_seen_ops{0};
    std::atomic<uint64_t> q4k_repacked_gemv_candidate_ops{0};
    std::atomic<uint64_t> q4k_repacked_gemv_used_ops{0};
    std::atomic<uint64_t> q4k_repacked_gemv_rejected_ops{0};
    std::atomic<int> q4k_repacked_gemv_last_reject_reason{0};
    std::atomic<int> q4k_repacked_gemv_primary_disable_reason{0};
    std::atomic<uint64_t> gemv_custom_total_ops{0};
    std::atomic<uint64_t> gemv_custom_decode_ops{0};
    std::atomic<uint64_t> gemv_custom_prefill_ops{0};
    std::atomic<uint64_t> gemv_custom_q4k_seen_ops{0};
    std::atomic<uint64_t> gemv_custom_non_q4k_ops{0};
    std::atomic<uint64_t> gemv_custom_quant_input_null_ops{0};
    std::atomic<uint64_t> gemv_custom_shape_reject_ops{0};
    std::atomic<uint64_t> gemv_custom_phase_unknown_ops{0};
    std::atomic<uint64_t> gemv_custom_force_reference_ops{0};
    std::atomic<uint64_t> gemv_custom_dynamic_lora_ops{0};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> gemv_custom_weight_type_hist{};
    std::array<std::atomic<uint64_t>, kMatmulQuantInputTypeHistCount> gemv_custom_quant_input_type_hist{};
    std::atomic<uint64_t> lfm2_decode_lm_head_custom_gemv_used_ops{0};
    std::atomic<uint64_t> lfm2_decode_lm_head_custom_gemv_ns{0};
    std::atomic<int> gemv_custom_tasks_effective{0};
    std::atomic<int> gemv_custom_tasks_cap_reason{0};
    std::atomic<uint64_t> decode_matmul_created_ops{0};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> decode_matmul_weight_type_hist{};
    std::array<std::atomic<uint64_t>, kMatmulPathHistCount> decode_matmul_path_hist{};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> prefill_matmul_weight_type_hist{};
    std::array<std::atomic<uint64_t>, kMatmulPathHistCount> prefill_matmul_path_hist{};
    std::atomic<uint64_t> qwen_target_ggml_compute_ops{0};
    std::atomic<uint64_t> qwen_target_ggml_matmul_ops{0};
    std::atomic<uint64_t> qwen_target_ggml_matmul_id_ops{0};
    std::atomic<uint64_t> qwen_target_ggml_quant_vecdot_ops{0};
    std::atomic<uint64_t> qwen_target_ggml_quantize_kv_ops{0};
    std::atomic<uint64_t> qwen_target_ggml_attention_ops{0};
    std::atomic<uint64_t> q6k_gemv_seen_ops{0};
    std::atomic<uint64_t> q6k_gemv_candidate_ops{0};
    std::atomic<uint64_t> q6k_gemv_used_ops{0};
    std::atomic<uint64_t> q6k_gemv_rejected_ops{0};
    std::atomic<uint64_t> q6k_gemv_reject_quant_input_null_ops{0};
    std::atomic<uint64_t> q6k_gemv_reject_unsupported_quant_input_ops{0};
    std::atomic<uint64_t> q6k_gemv_reject_shape_ops{0};
    std::atomic<uint64_t> q6k_gemv_reject_phase_ops{0};
    std::atomic<uint64_t> q6k_gemv_reject_kernel_unavailable_ops{0};
    std::atomic<int> q6k_gemv_last_reject_reason{0};
    std::atomic<uint64_t> q6k_gemv_total_ns{0};
    std::atomic<uint64_t> moe_small_decode_parallel_candidate_ops{0};
    std::atomic<uint64_t> moe_small_decode_parallel_used_ops{0};
    std::atomic<uint64_t> moe_small_decode_parallel_rejected_ops{0};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> moe_expert_matmul_weight_type_hist{};
    std::atomic<uint64_t> moe_q4k_repacked_candidate_ops{0};
    std::atomic<uint64_t> moe_q4k_repacked_used_ops{0};
    std::atomic<uint64_t> moe_q4k_repacked_rejected_ops{0};
    std::atomic<uint64_t> moe_q5k_repacked_candidate_ops{0};
    std::atomic<uint64_t> moe_q5k_repacked_used_ops{0};
    std::atomic<uint64_t> moe_q5k_repacked_rejected_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_candidate_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_used_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_rejected_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_gate_up_used{0};
    std::atomic<uint64_t> gemma4_moe_prefill_quant_batch_down_used{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_candidate_layers{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_used_layers{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_rejected_layers{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_gate_up_ns{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_down_ns{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_total_ns{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_duplicate_work_detected{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_candidate_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_used_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_rejected_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_q4k_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_q8_0_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_native_ns{0};
    std::atomic<uint64_t> gemma4_dense_prefill_replaced_ggml_mul_mat_ops{0};
    std::atomic<uint64_t> gemma4_dense_prefill_duplicate_work_detected{0};
    std::atomic<uint64_t> gemma4_fast_gelu_enabled{0};
    std::atomic<uint64_t> gemma4_fast_gelu_used{0};
    std::atomic<uint64_t> gemma4_fast_gelu_ns{0};
    std::atomic<uint64_t> gemma4_native_moe_prefill_gate_up_fast_gelu_ns{0};
    std::atomic<uint64_t> gemma4_decode_native_candidate_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_used_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_rejected_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_moe_used_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_dense_used_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_lm_head_used_ops{0};
    std::atomic<uint64_t> gemma4_decode_native_ns{0};
    std::atomic<uint64_t> gemma4_decode_replaced_ggml_mul_mat_ops{0};
    std::atomic<uint64_t> gemma4_decode_replaced_ggml_mul_mat_id_ops{0};
    std::atomic<uint64_t> gemma4_decode_duplicate_work_detected{0};
    std::atomic<uint64_t> gemma4_native_int4_gemv_candidate_ops{0};
    std::atomic<uint64_t> gemma4_native_int4_gemv_used_ops{0};
    std::atomic<uint64_t> gemma4_native_int4_gemv_ns{0};
    std::atomic<uint64_t> gemma4_native_int4_repacked_weight_count{0};
    std::atomic<uint64_t> gemma4_native_int4_repacked_bytes{0};
    std::atomic<uint64_t> gemma4_native_fused_gateup_used_ops{0};
    std::atomic<uint64_t> ggml_delegated_quant_gemv_ops{0};
    std::atomic<uint64_t> gemma4_native_paged_attention_candidate_ops{0};
    std::atomic<uint64_t> gemma4_native_paged_attention_used_ops{0};
    std::atomic<uint64_t> gemma4_native_paged_attention_ns{0};
    std::atomic<uint64_t> gemma4_ggml_attention_fallback_ops{0};
    std::atomic<int> gemma4_paged_attention_cache_type{-1};
    std::atomic<int> gemma4_paged_attention_context_len{0};
    std::atomic<uint64_t> gemma4_paged_attention_head_range{0};
    std::atomic<uint64_t> native_moe_fast_decode_candidate_ops{0};
    std::atomic<uint64_t> native_moe_fast_decode_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_decode_rejected_ops{0};
    std::atomic<uint64_t> native_moe_fast_decode_w1w3_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_decode_w2_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_decode_ns{0};
    std::atomic<uint64_t> native_moe_fast_w1w3_ns{0};
    std::atomic<uint64_t> native_moe_fast_w2_ns{0};
    std::atomic<uint64_t> native_moe_fast_reduce_ns{0};
    std::atomic<uint64_t> native_moe_fast_total_ns{0};
    std::atomic<uint64_t> native_moe_fast_w1w3_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_w2_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_w2_q5k_candidate_ops{0};
    std::atomic<uint64_t> native_moe_fast_w2_q5k_used_ops{0};
    std::atomic<uint64_t> native_moe_fast_w2_q5k_rejected_ops{0};
    std::atomic<uint64_t> native_moe_fast_w2_q5k_ns{0};
    std::atomic<int> qwen35_moe_path{0};
    std::atomic<uint64_t> qwen35_moe_layers_seen{0};
    std::atomic<uint64_t> qwen35_moe_forward_calls{0};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> qwen35_moe_w1w3_weight_type_hist{};
    std::array<std::atomic<uint64_t>, kMatmulWeightTypeHistCount> qwen35_moe_w2_weight_type_hist{};
    std::atomic<int> qwen35_moe_selected_expert_count{0};
    std::atomic<int> qwen35_moe_top_k{0};
    std::atomic<int> qwen35_moe_instrumentation_missing{0};
    std::atomic<uint64_t> qwen36_prefill_total_ns{0};
    std::atomic<uint64_t> qwen36_prefill_ssm_projection_ns{0};
    std::atomic<uint64_t> qwen36_prefill_ssm_delta_state_ns{0};
    std::atomic<uint64_t> qwen36_prefill_attention_ns{0};
    std::atomic<uint64_t> qwen36_prefill_mlp_or_moe_ns{0};
    std::atomic<uint64_t> qwen36_prefill_graph_build_ns{0};
    std::atomic<uint64_t> qwen36_prefill_graph_execute_ns{0};
    std::atomic<int> moe_selected_expert_count{0};
    std::atomic<int> moe_top_k{0};
    std::atomic<int> moe_expert_parallel_tasks{0};
    std::atomic<int> q4k_copied_gemv_experiment_used{0};
    std::atomic<int> q4k_copied_gemv_experiment_last_reject_reason{0};
    std::atomic<int> paged_attn_decode_head_tile_effective{0};
    std::atomic<int> arm_batched_quant_used{0};
    std::atomic<int> attention_path_paged{0};
    std::atomic<int> attention_path_standard{0};
    std::atomic<int> attention_path_portable_flash{0};
    std::atomic<int> attention_path_native_flash{0};
    std::atomic<int> attention_path_hal{0};
    std::atomic<uint64_t> flash_attention_headseq_prefill_calls{0};
    std::atomic<uint64_t> flash_attention_native_decode_calls{0};
    std::atomic<uint64_t> flash_attention_reference_calls{0};
    std::atomic<uint64_t> flash_attention_non_avx512_tiled_calls{0};
    std::atomic<uint64_t> flash_attention_avx512_tiled_calls{0};
    std::atomic<int> flash_attention_last_nth{0};
    std::atomic<int> flash_attention_last_active_threads{0};
    std::atomic<int> kleidiai_compiled_enabled{0};
    std::atomic<int> kleidiai_last_reject_reason{0};
};

struct InferenceWorkContext {
    const BatchSpec* batch = nullptr;
    ModelVariant model_variant = ModelVariant::UNKNOWN;
    InferenceExecutionPhase phase = InferenceExecutionPhase::Unknown;
    bool graph_build_no_alloc = false;
    Qwen36ProfileCounters qwen36_profile;
    KVCacheUserData kv_pool[256];
    std::vector<Gemma4SharedKVState> gemma4_shared_kv_states;
    QKVUserData qkv_pool[256];
    KVUpdateGatherUserData kv_update_gather_pool[kMaxKVUpdateGatherSlots];
    int qkv_index = 0;
    AddRMSNormUserData add_rmsnorm_pool[kMaxAddRMSNormSlots];
    int add_rmsnorm_index = 0;
    alignas(64) std::array<uint8_t, kMaxQuantInputBufferSize> gemv_quant_input_shared{};
    std::atomic<uint64_t> gemv_quantized_stamp{0};
    uint64_t execution_generation = 0;
    uint64_t qact_generation = 0;
    const ggml_tensor* qact_tensor = nullptr;
    const void* qact_source = nullptr;
    int64_t qact_len = 0;
    ggml_type qact_type = GGML_TYPE_COUNT;
    size_t qact_bytes = 0;
    int qact_slot_id = -1;
    int64_t qact_token_pos = std::numeric_limits<int64_t>::min();
    std::vector<uint8_t> qact_buffer;
    std::mutex q4k_repacked_gemv_request_mutex;
    std::unordered_map<uint64_t, uint32_t> q4k_repacked_gemv_repack_counts;
    alignas(
        64) std::array<uint8_t, kMaxQuantInputBufferSize * kMaxSmallBatchColsHard> gemv_batched_quant_input_shared{};
    std::atomic<uint64_t> gemv_batched_quantized_stamp{0};
    GemvUserData gemv_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_userdata_index = 0;
    GemvBatchedUserData gemv_batched_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_batched_userdata_index = 0;
    PagedAttentionUserData paged_attention_userdata_pool[kMaxPagedAttentionUserDataSlots];
    int paged_attention_userdata_index = 0;
    std::vector<const void*> paged_attention_shared_k_block_ptrs;
    std::vector<const void*> paged_attention_shared_v_block_ptrs;
    mutable std::mutex profile_string_mutex;
    std::string moe_small_decode_parallel_last_reject_reason;
    std::string moe_q4k_repacked_last_reject_reason;
    std::string moe_q5k_repacked_last_reject_reason;
    std::string gemma4_moe_prefill_quant_batch_last_reject_reason;
    std::string gemma4_native_moe_prefill_last_reject_reason;
    std::string gemma4_dense_prefill_native_last_reject_reason;
    std::string gemma4_decode_native_last_reject_reason;
    std::string qwen_target_ggml_compute_last_reason;
    std::string qwen_target_ggml_compute_last_op;
    std::string qwen_target_ggml_compute_target;
    std::string native_moe_fast_decode_last_reject_reason;
    std::string native_moe_fast_w2_q5k_last_reject_reason;
    std::string q6k_gemv_effective_phase;
    std::string q6k_gemv_graph_phase;
    std::string q6k_gemv_callback_phase;
    mutable std::mutex matmul_dispatch_census_mutex;
    std::vector<MatmulDispatchCensusEntry> matmul_dispatch_census_entries;
    std::vector<MatmulShapeCensusEntry> decode_matmul_shape_entries;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_shape_entries;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_ggml_shape_entries;
    std::vector<MatmulShapeCensusEntry> q6k_gemv_shape_entries;
    std::vector<MatmulShapeCensusEntry> qwen36_prefill_slow_entries;
    SSMConv1DUserData ssm_conv1d_pool[128];
    int ssm_conv1d_index = 0;
    LFM2ShortConvUserData lfm2_shortconv_pool[128];
    int lfm2_shortconv_index = 0;
    ProjectionReferenceUserData projection_reference_pool[384];
    int projection_reference_index = 0;
    RmsNormReferenceUserData rmsnorm_reference_pool[256];
    int rmsnorm_reference_index = 0;
    AttentionCoreReferenceUserData attention_core_reference_pool[128];
    int attention_core_reference_index = 0;
    SSMQwen35DeltaUserData ssm_qwen35_delta_pool[128];
    int ssm_qwen35_delta_index = 0;
    std::vector<ggml_bf16_t> bf16_buffer;
};

bool IsQwen36ProfilingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_QWEN36_PROFILE");
        if (env && env[0] != '\0') {
            return std::strcmp(env, "0") != 0;
        }
        env = std::getenv("DENSECORE_QWEN36_PREFILL_PROFILE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static std::size_t MatmulWeightTypeHistIndex(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return 0;
        case GGML_TYPE_Q5_K:
            return 1;
        case GGML_TYPE_Q6_K:
            return 2;
        case GGML_TYPE_Q8_0:
            return 3;
        case GGML_TYPE_F16:
            return 4;
        case GGML_TYPE_F32:
            return 5;
        default:
            return 6;
    }
}

static std::size_t MatmulQuantInputTypeHistIndex(ggml_type type, bool has_quant_input) {
    if (!has_quant_input) {
        return 2;
    }
    switch (type) {
        case GGML_TYPE_Q8_K:
            return 0;
        case GGML_TYPE_Q8_0:
            return 1;
        default:
            return 3;
    }
}

static const char* MatmulPhaseName(InferenceExecutionPhase phase) {
    switch (phase) {
        case InferenceExecutionPhase::Prefill:
            return "prefill";
        case InferenceExecutionPhase::Decode:
            return "decode";
        case InferenceExecutionPhase::Unknown:
        default:
            return "unknown";
    }
}

static const char* MatmulModelFamilyName(ModelVariant variant) {
    switch (variant) {
        case ModelVariant::QWEN35:
            return "qwen35";
        case ModelVariant::QWEN36:
            return "qwen36";
        case ModelVariant::GEMMA4:
            return "gemma4";
        case ModelVariant::LFM2MOE:
            return "lfm2";
        case ModelVariant::GEMMA:
            return "gemma";
        case ModelVariant::QWEN3:
            return "qwen3";
        case ModelVariant::QWEN3NEXT:
            return "qwen3next";
        default:
            return "unknown";
    }
}

static const char* MatmulWeightTypeHistLabel(std::size_t index) {
    switch (index) {
        case 0:
            return "q4_k";
        case 1:
            return "q5_k";
        case 2:
            return "q6_k";
        case 3:
            return "q8_0";
        case 4:
            return "f16";
        case 5:
            return "f32";
        default:
            return "other";
    }
}

static std::size_t MatmulPathHistIndex(const char* path) {
    if (!path || path[0] == '\0') {
        return 6;
    }
    if (std::strcmp(path, "ggml_mul_mat") == 0 || std::strcmp(path, "GGML_NATIVE") == 0) {
        return 0;
    }
    if (std::strcmp(path, "ggml_mul_mat_id") == 0) {
        return 1;
    }
    if (std::strcmp(path, "custom_gemv") == 0 || std::strcmp(path, "GEMV_QUANT") == 0 ||
        std::strcmp(path, "GEMV_F32") == 0) {
        return 2;
    }
    if (std::strcmp(path, "custom_batched_gemv") == 0 || std::strcmp(path, "GGML_QUANT_NRC_M") == 0 ||
        std::strcmp(path, "GGML_QUANT_Q4K_TRUE_BATCHED") == 0 || std::strcmp(path, "BATCHED_F32") == 0) {
        return 3;
    }
    if (std::strcmp(path, "moe_native") == 0) {
        return 4;
    }
    if (std::strcmp(path, "ssm_projection") == 0) {
        return 5;
    }
    return 6;
}

static const char* MatmulPathHistLabel(std::size_t index) {
    switch (index) {
        case 0:
            return "ggml_mul_mat";
        case 1:
            return "ggml_mul_mat_id";
        case 2:
            return "custom_gemv";
        case 3:
            return "custom_batched_gemv";
        case 4:
            return "moe_native";
        case 5:
            return "ssm_projection";
        default:
            return "other";
    }
}

static int Qwen35MoEPathCode(const char* path) {
    if (!path || path[0] == '\0' || std::strcmp(path, "none") == 0) return 0;
    if (std::strcmp(path, "native_graph") == 0) return 1;
    if (std::strcmp(path, "cb_moe_forward") == 0) return 2;
    if (std::strcmp(path, "ggml_mul_mat_id") == 0) return 3;
    if (std::strcmp(path, "fused_rowblock") == 0) return 4;
    return 5;
}

const char* Qwen35MoEPathName(int code) {
    switch (code) {
        case 1:
            return "native_graph";
        case 2:
            return "cb_moe_forward";
        case 3:
            return "ggml_mul_mat_id";
        case 4:
            return "fused_rowblock";
        case 5:
            return "other";
        case 0:
        default:
            return "none";
    }
}

static int Q6KGemvRejectReasonCode(const char* reason) {
    if (!reason || reason[0] == '\0' || std::strcmp(reason, "none") == 0) return 0;
    if (std::strcmp(reason, "not_q6_k") == 0) return 1;
    if (std::strcmp(reason, "not_decode") == 0) return 2;
    if (std::strcmp(reason, "missing_vec_dot") == 0) return 3;
    if (std::strcmp(reason, "kernel_unavailable") == 0) return 3;
    if (std::strcmp(reason, "shape") == 0) return 4;
    if (std::strcmp(reason, "unsupported_shape") == 0) return 4;
    if (std::strcmp(reason, "fallback") == 0) return 5;
    if (std::strcmp(reason, "disabled") == 0) return 6;
    if (std::strcmp(reason, "dynamic_lora") == 0) return 7;
    if (std::strcmp(reason, "reference_forced") == 0) return 8;
    if (std::strcmp(reason, "missing_quant_input") == 0) return 9;
    if (std::strcmp(reason, "quant_input_null") == 0) return 9;
    if (std::strcmp(reason, "input_type") == 0) return 10;
    if (std::strcmp(reason, "unsupported_quant_input") == 0) return 10;
    return 11;
}

const char* Q6KGemvRejectReasonName(int code) {
    switch (code) {
        case 1:
            return "not_q6_k";
        case 2:
            return "not_decode";
        case 3:
            return "kernel_unavailable";
        case 4:
            return "unsupported_shape";
        case 5:
            return "fallback";
        case 6:
            return "disabled";
        case 7:
            return "dynamic_lora";
        case 8:
            return "reference_forced";
        case 9:
            return "quant_input_null";
        case 10:
            return "unsupported_quant_input";
        case 11:
            return "other";
        case 0:
        default:
            return "none";
    }
}

static std::string MatmulShapeBucket(int64_t m, int64_t n, int64_t k) {
    std::ostringstream oss;
    if (m > 0) {
        oss << "M=" << m << ",";
    }
    oss << "N=" << n << ",K=" << k;
    return oss.str();
}

static void AddMatmulShapeCensusEntry(std::vector<MatmulShapeCensusEntry>* entries,
                                      const MatmulShapeCensusEntry& entry) {
    if (!entries || entry.ops == 0) {
        return;
    }
    for (auto& existing : *entries) {
        if (existing.model_family == entry.model_family && existing.phase == entry.phase &&
            existing.op_type == entry.op_type && existing.dispatch_path == entry.dispatch_path &&
            existing.weight_type == entry.weight_type && existing.weight_class == entry.weight_class &&
            existing.shape_bucket == entry.shape_bucket && existing.left_name == entry.left_name &&
            existing.right_name == entry.right_name) {
            existing.ops += entry.ops;
            existing.wall_ns += entry.wall_ns;
            existing.calls += entry.calls;
            existing.active_threads = std::max(existing.active_threads, entry.active_threads);
            existing.contiguous_or_copy_input =
                std::max(existing.contiguous_or_copy_input, entry.contiguous_or_copy_input);
            return;
        }
    }
    entries->push_back(entry);
}

static void SortAndTrimMatmulShapeCensusEntries(std::vector<MatmulShapeCensusEntry>* entries) {
    if (!entries) {
        return;
    }
    std::sort(entries->begin(), entries->end(), [](const auto& a, const auto& b) {
        const uint64_t a_cost = a.wall_ns != 0 ? a.wall_ns : a.ops;
        const uint64_t b_cost = b.wall_ns != 0 ? b.wall_ns : b.ops;
        if (a_cost != b_cost) {
            return a_cost > b_cost;
        }
        return a.shape_bucket < b.shape_bucket;
    });
    if (entries->size() > kMatmulTopShapeCount) {
        entries->resize(kMatmulTopShapeCount);
    }
}

template <typename THist>
static void ResetAtomicHistogram(THist& hist) {
    for (auto& value : hist) {
        value.store(0, std::memory_order_relaxed);
    }
}

template <typename TDestHist, typename TSrcHist>
static void SnapshotAtomicHistogram(TDestHist& dst, const TSrcHist& src) {
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = src[i].load(std::memory_order_relaxed);
    }
}

static void AddMatmulDispatchCensusEntry(std::vector<MatmulDispatchCensusEntry>* entries,
                                         const MatmulDispatchCensusEntry& entry) {
    if (!entries || entry.wall_ns == 0 || entry.ops == 0) {
        return;
    }
    for (auto& existing : *entries) {
        if (existing.model_family == entry.model_family && existing.phase == entry.phase &&
            existing.dispatch_path == entry.dispatch_path &&
            existing.weight_type == entry.weight_type && existing.shape_bucket == entry.shape_bucket) {
            existing.wall_ns += entry.wall_ns;
            existing.ops += entry.ops;
            return;
        }
    }
    entries->push_back(entry);
}

static void SortAndTrimMatmulDispatchCensusEntries(std::vector<MatmulDispatchCensusEntry>* entries) {
    if (!entries) {
        return;
    }
    std::sort(entries->begin(), entries->end(), [](const auto& a, const auto& b) {
        if (a.wall_ns != b.wall_ns) {
            return a.wall_ns > b.wall_ns;
        }
        return a.ops > b.ops;
    });
    if (entries->size() > kMatmulDispatchTopSlowCount) {
        entries->resize(kMatmulDispatchTopSlowCount);
    }
}

void ResetQwen36Profile(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->model_variant = ModelVariant::UNKNOWN;
    auto& p = ctx->qwen36_profile;
    p.attention_ns.store(0, std::memory_order_relaxed);
    p.paged_attention_ns.store(0, std::memory_order_relaxed);
    p.standard_attention_ns.store(0, std::memory_order_relaxed);
    p.portable_flash_attention_ns.store(0, std::memory_order_relaxed);
    p.native_flash_attention_ns.store(0, std::memory_order_relaxed);
    p.hal_attention_ns.store(0, std::memory_order_relaxed);
    p.attention_repack_ns.store(0, std::memory_order_relaxed);
    p.moe_forward_ns.store(0, std::memory_order_relaxed);
    p.moe_route_ns.store(0, std::memory_order_relaxed);
    p.moe_reorder_ns.store(0, std::memory_order_relaxed);
    p.moe_expert_ns.store(0, std::memory_order_relaxed);
    p.moe_reduce_ns.store(0, std::memory_order_relaxed);
    p.moe_w1w3_ns.store(0, std::memory_order_relaxed);
    p.moe_w2_ns.store(0, std::memory_order_relaxed);
    p.moe_rowblock_ns.store(0, std::memory_order_relaxed);
    p.moe_rowblock_w1w3_ns.store(0, std::memory_order_relaxed);
    p.moe_rowblock_w2_ns.store(0, std::memory_order_relaxed);
    p.shared_expert_ns.store(0, std::memory_order_relaxed);
    p.quant_matmul_ns.store(0, std::memory_order_relaxed);
    p.ssm_qkv_wall_ns.store(0, std::memory_order_relaxed);
    p.ssm_gate_wall_ns.store(0, std::memory_order_relaxed);
    p.ssm_delta_wall_ns.store(0, std::memory_order_relaxed);
    p.ssm_out_wall_ns.store(0, std::memory_order_relaxed);
    p.ssm_conv1d_ns.store(0, std::memory_order_relaxed);
    p.ssm_delta_ns.store(0, std::memory_order_relaxed);
    p.kv_update_ns.store(0, std::memory_order_relaxed);
    p.sample_ns.store(0, std::memory_order_relaxed);
    p.kleidiai_candidate_ops.store(0, std::memory_order_relaxed);
    p.kleidiai_allowed_ops.store(0, std::memory_order_relaxed);
    p.kleidiai_rejected_ops.store(0, std::memory_order_relaxed);
    p.graph_cache_hits.store(0, std::memory_order_relaxed);
    p.graph_cache_misses.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_cache_hits.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_cache_waited_hits.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_cache_misses.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_cache_evictions.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_cache_evicted_bytes.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_repack_bytes.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_probe_ns.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_resident_bytes.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_distinct_weights_seen.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_repeated_repack_count.store(0, std::memory_order_relaxed);
    p.q4k_copied_gemv_experiment_cache_hits.store(0, std::memory_order_relaxed);
    p.q4k_copied_gemv_experiment_cache_misses.store(0, std::memory_order_relaxed);
    p.qact_cache_hits.store(0, std::memory_order_relaxed);
    p.qact_cache_misses.store(0, std::memory_order_relaxed);
    p.qact_cache_reused_bytes.store(0, std::memory_order_relaxed);
    p.moe_decode_scratch_reused.store(0, std::memory_order_relaxed);
    p.moe_decode_allocations_avoided.store(0, std::memory_order_relaxed);
    p.moe_task_count.store(0, std::memory_order_relaxed);
    p.moe_rowblock_used.store(0, std::memory_order_relaxed);
    p.moe_rowblock_tasks.store(0, std::memory_order_relaxed);
    p.selected_expert_count.store(0, std::memory_order_relaxed);
    p.ssm_conv1d_calls.store(0, std::memory_order_relaxed);
    p.ssm_delta_calls.store(0, std::memory_order_relaxed);
    p.q4k_true_batched_used.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_batched_mode.store(1, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_batched_used.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_batched_probe_pass.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_batched_max_abs_error_bits.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_batched_last_reject_reason.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_probe_participants.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_probe_failures.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_q4k_admission_downgraded.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_mode.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_prepared.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_used.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_last_reject_reason.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_qkv_count.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_gate_count.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_out_count.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_candidate_ops.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_used_ops.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_prefill_amx_rejected_ops.store(0, std::memory_order_relaxed);
    p.qwen36_ssm_q8_decode_used_original_q8_path.store(0, std::memory_order_relaxed);
    ResetAtomicHistogram(p.qwen36_ssm_projection_weight_type_hist);
    p.q4k_repacked_gemv_used.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_seen_ops.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_candidate_ops.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_used_ops.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_rejected_ops.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_last_reject_reason.store(0, std::memory_order_relaxed);
    p.q4k_repacked_gemv_primary_disable_reason.store(0, std::memory_order_relaxed);
    p.gemv_custom_total_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_decode_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_prefill_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_q4k_seen_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_non_q4k_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_quant_input_null_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_shape_reject_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_phase_unknown_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_force_reference_ops.store(0, std::memory_order_relaxed);
    p.gemv_custom_dynamic_lora_ops.store(0, std::memory_order_relaxed);
    p.lfm2_decode_lm_head_custom_gemv_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_decode_lm_head_custom_gemv_ns.store(0, std::memory_order_relaxed);
    ResetAtomicHistogram(p.gemv_custom_weight_type_hist);
    ResetAtomicHistogram(p.gemv_custom_quant_input_type_hist);
    p.gemv_custom_tasks_effective.store(0, std::memory_order_relaxed);
    p.gemv_custom_tasks_cap_reason.store(0, std::memory_order_relaxed);
    p.decode_matmul_created_ops.store(0, std::memory_order_relaxed);
    ResetAtomicHistogram(p.decode_matmul_weight_type_hist);
    ResetAtomicHistogram(p.decode_matmul_path_hist);
    ResetAtomicHistogram(p.prefill_matmul_weight_type_hist);
    ResetAtomicHistogram(p.prefill_matmul_path_hist);
    p.qwen_target_ggml_compute_ops.store(0, std::memory_order_relaxed);
    p.qwen_target_ggml_matmul_ops.store(0, std::memory_order_relaxed);
    p.qwen_target_ggml_matmul_id_ops.store(0, std::memory_order_relaxed);
    p.qwen_target_ggml_quant_vecdot_ops.store(0, std::memory_order_relaxed);
    p.qwen_target_ggml_quantize_kv_ops.store(0, std::memory_order_relaxed);
    p.qwen_target_ggml_attention_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_seen_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_candidate_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_used_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_rejected_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_reject_quant_input_null_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_reject_unsupported_quant_input_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_reject_shape_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_reject_phase_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_reject_kernel_unavailable_ops.store(0, std::memory_order_relaxed);
    p.q6k_gemv_last_reject_reason.store(0, std::memory_order_relaxed);
    p.q6k_gemv_total_ns.store(0, std::memory_order_relaxed);
    p.moe_small_decode_parallel_candidate_ops.store(0, std::memory_order_relaxed);
    p.moe_small_decode_parallel_used_ops.store(0, std::memory_order_relaxed);
    p.moe_small_decode_parallel_rejected_ops.store(0, std::memory_order_relaxed);
    ResetAtomicHistogram(p.moe_expert_matmul_weight_type_hist);
    p.moe_q4k_repacked_candidate_ops.store(0, std::memory_order_relaxed);
    p.moe_q4k_repacked_used_ops.store(0, std::memory_order_relaxed);
    p.moe_q4k_repacked_rejected_ops.store(0, std::memory_order_relaxed);
    p.moe_q5k_repacked_candidate_ops.store(0, std::memory_order_relaxed);
    p.moe_q5k_repacked_used_ops.store(0, std::memory_order_relaxed);
    p.moe_q5k_repacked_rejected_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_candidate_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_rejected_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_gate_up_used.store(0, std::memory_order_relaxed);
    p.gemma4_moe_prefill_quant_batch_down_used.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_candidate_layers.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_used_layers.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_rejected_layers.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_gate_up_ns.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_down_ns.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_total_ns.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_duplicate_work_detected.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_candidate_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_rejected_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_q4k_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_q8_0_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_native_ns.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_replaced_ggml_mul_mat_ops.store(0, std::memory_order_relaxed);
    p.gemma4_dense_prefill_duplicate_work_detected.store(0, std::memory_order_relaxed);
    p.gemma4_fast_gelu_enabled.store(0, std::memory_order_relaxed);
    p.gemma4_fast_gelu_used.store(0, std::memory_order_relaxed);
    p.gemma4_fast_gelu_ns.store(0, std::memory_order_relaxed);
    p.gemma4_native_moe_prefill_gate_up_fast_gelu_ns.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_candidate_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_rejected_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_moe_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_dense_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_lm_head_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_native_ns.store(0, std::memory_order_relaxed);
    p.gemma4_decode_replaced_ggml_mul_mat_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_replaced_ggml_mul_mat_id_ops.store(0, std::memory_order_relaxed);
    p.gemma4_decode_duplicate_work_detected.store(0, std::memory_order_relaxed);
    p.gemma4_native_int4_gemv_candidate_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_int4_gemv_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_int4_gemv_ns.store(0, std::memory_order_relaxed);
    p.gemma4_native_int4_repacked_weight_count.store(0, std::memory_order_relaxed);
    p.gemma4_native_int4_repacked_bytes.store(0, std::memory_order_relaxed);
    p.gemma4_native_fused_gateup_used_ops.store(0, std::memory_order_relaxed);
    p.ggml_delegated_quant_gemv_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_paged_attention_candidate_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_paged_attention_used_ops.store(0, std::memory_order_relaxed);
    p.gemma4_native_paged_attention_ns.store(0, std::memory_order_relaxed);
    p.gemma4_ggml_attention_fallback_ops.store(0, std::memory_order_relaxed);
    p.gemma4_paged_attention_cache_type.store(-1, std::memory_order_relaxed);
    p.gemma4_paged_attention_context_len.store(0, std::memory_order_relaxed);
    p.gemma4_paged_attention_head_range.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_candidate_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_rejected_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_w1w3_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_w2_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_decode_ns.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w1w3_ns.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_ns.store(0, std::memory_order_relaxed);
    p.native_moe_fast_reduce_ns.store(0, std::memory_order_relaxed);
    p.native_moe_fast_total_ns.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w1w3_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_q5k_candidate_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_q5k_used_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_q5k_rejected_ops.store(0, std::memory_order_relaxed);
    p.native_moe_fast_w2_q5k_ns.store(0, std::memory_order_relaxed);
    p.qwen35_moe_path.store(0, std::memory_order_relaxed);
    p.qwen35_moe_layers_seen.store(0, std::memory_order_relaxed);
    p.qwen35_moe_forward_calls.store(0, std::memory_order_relaxed);
    ResetAtomicHistogram(p.qwen35_moe_w1w3_weight_type_hist);
    ResetAtomicHistogram(p.qwen35_moe_w2_weight_type_hist);
    p.qwen35_moe_selected_expert_count.store(0, std::memory_order_relaxed);
    p.qwen35_moe_top_k.store(0, std::memory_order_relaxed);
    p.qwen35_moe_instrumentation_missing.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_total_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_ssm_projection_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_ssm_delta_state_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_attention_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_mlp_or_moe_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_graph_build_ns.store(0, std::memory_order_relaxed);
    p.qwen36_prefill_graph_execute_ns.store(0, std::memory_order_relaxed);
    p.moe_selected_expert_count.store(0, std::memory_order_relaxed);
    p.moe_top_k.store(0, std::memory_order_relaxed);
    p.moe_expert_parallel_tasks.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        ctx->moe_small_decode_parallel_last_reject_reason.clear();
        ctx->moe_q4k_repacked_last_reject_reason.clear();
        ctx->moe_q5k_repacked_last_reject_reason.clear();
        ctx->gemma4_moe_prefill_quant_batch_last_reject_reason.clear();
        ctx->gemma4_native_moe_prefill_last_reject_reason.clear();
        ctx->gemma4_dense_prefill_native_last_reject_reason.clear();
        ctx->gemma4_decode_native_last_reject_reason.clear();
        ctx->qwen_target_ggml_compute_last_reason.clear();
        ctx->qwen_target_ggml_compute_last_op.clear();
        ctx->qwen_target_ggml_compute_target.clear();
        ctx->native_moe_fast_decode_last_reject_reason.clear();
        ctx->native_moe_fast_w2_q5k_last_reject_reason.clear();
        ctx->q6k_gemv_effective_phase.clear();
        ctx->q6k_gemv_graph_phase.clear();
        ctx->q6k_gemv_callback_phase.clear();
    }
    {
        std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
        ctx->matmul_dispatch_census_entries.clear();
        ctx->decode_matmul_shape_entries.clear();
        ctx->q6k_gemv_shape_entries.clear();
        ctx->qwen36_prefill_slow_entries.clear();
    }
    p.q4k_copied_gemv_experiment_used.store(0, std::memory_order_relaxed);
    p.q4k_copied_gemv_experiment_last_reject_reason.store(0, std::memory_order_relaxed);
    p.paged_attn_decode_head_tile_effective.store(0, std::memory_order_relaxed);
    p.arm_batched_quant_used.store(0, std::memory_order_relaxed);
    p.attention_path_paged.store(0, std::memory_order_relaxed);
    p.attention_path_standard.store(0, std::memory_order_relaxed);
    p.attention_path_portable_flash.store(0, std::memory_order_relaxed);
    p.attention_path_native_flash.store(0, std::memory_order_relaxed);
    p.attention_path_hal.store(0, std::memory_order_relaxed);
    p.flash_attention_headseq_prefill_calls.store(0, std::memory_order_relaxed);
    p.flash_attention_native_decode_calls.store(0, std::memory_order_relaxed);
    p.flash_attention_reference_calls.store(0, std::memory_order_relaxed);
    p.flash_attention_non_avx512_tiled_calls.store(0, std::memory_order_relaxed);
    p.flash_attention_avx512_tiled_calls.store(0, std::memory_order_relaxed);
    p.flash_attention_last_nth.store(0, std::memory_order_relaxed);
    p.flash_attention_last_active_threads.store(0, std::memory_order_relaxed);
    p.kleidiai_compiled_enabled.store(densecore::runtime::KleidiAICompiledEnabled() ? 1 : 0,
                                      std::memory_order_relaxed);
    p.kleidiai_last_reject_reason.store(
        static_cast<int>(densecore::runtime::KernelAdmissionRejectReason::None), std::memory_order_relaxed);
}

Qwen36ProfileSnapshot GetQwen36ProfileSnapshot(const InferenceWorkContext* ctx) {
    Qwen36ProfileSnapshot snapshot;
    if (!ctx) {
        return snapshot;
    }
    const auto& p = ctx->qwen36_profile;
    snapshot.attention_ns = p.attention_ns.load(std::memory_order_relaxed);
    snapshot.paged_attention_ns = p.paged_attention_ns.load(std::memory_order_relaxed);
    snapshot.standard_attention_ns = p.standard_attention_ns.load(std::memory_order_relaxed);
    snapshot.portable_flash_attention_ns = p.portable_flash_attention_ns.load(std::memory_order_relaxed);
    snapshot.native_flash_attention_ns = p.native_flash_attention_ns.load(std::memory_order_relaxed);
    snapshot.hal_attention_ns = p.hal_attention_ns.load(std::memory_order_relaxed);
    snapshot.attention_repack_ns = p.attention_repack_ns.load(std::memory_order_relaxed);
    snapshot.moe_forward_ns = p.moe_forward_ns.load(std::memory_order_relaxed);
    snapshot.moe_route_ns = p.moe_route_ns.load(std::memory_order_relaxed);
    snapshot.moe_reorder_ns = p.moe_reorder_ns.load(std::memory_order_relaxed);
    snapshot.moe_expert_ns = p.moe_expert_ns.load(std::memory_order_relaxed);
    snapshot.moe_reduce_ns = p.moe_reduce_ns.load(std::memory_order_relaxed);
    snapshot.moe_w1w3_ns = p.moe_w1w3_ns.load(std::memory_order_relaxed);
    snapshot.moe_w2_ns = p.moe_w2_ns.load(std::memory_order_relaxed);
    snapshot.moe_rowblock_ns = p.moe_rowblock_ns.load(std::memory_order_relaxed);
    snapshot.moe_rowblock_w1w3_ns = p.moe_rowblock_w1w3_ns.load(std::memory_order_relaxed);
    snapshot.moe_rowblock_w2_ns = p.moe_rowblock_w2_ns.load(std::memory_order_relaxed);
    snapshot.shared_expert_ns = p.shared_expert_ns.load(std::memory_order_relaxed);
    snapshot.quant_matmul_ns = p.quant_matmul_ns.load(std::memory_order_relaxed);
    snapshot.ssm_qkv_wall_ns = p.ssm_qkv_wall_ns.load(std::memory_order_relaxed);
    snapshot.ssm_gate_wall_ns = p.ssm_gate_wall_ns.load(std::memory_order_relaxed);
    snapshot.ssm_delta_wall_ns = p.ssm_delta_wall_ns.load(std::memory_order_relaxed);
    snapshot.ssm_out_wall_ns = p.ssm_out_wall_ns.load(std::memory_order_relaxed);
    snapshot.ssm_conv1d_ns = p.ssm_conv1d_ns.load(std::memory_order_relaxed);
    snapshot.ssm_delta_ns = p.ssm_delta_ns.load(std::memory_order_relaxed);
    snapshot.kv_update_ns = p.kv_update_ns.load(std::memory_order_relaxed);
    snapshot.sample_ns = p.sample_ns.load(std::memory_order_relaxed);
    snapshot.kleidiai_candidate_ops = p.kleidiai_candidate_ops.load(std::memory_order_relaxed);
    snapshot.kleidiai_allowed_ops = p.kleidiai_allowed_ops.load(std::memory_order_relaxed);
    snapshot.kleidiai_rejected_ops = p.kleidiai_rejected_ops.load(std::memory_order_relaxed);
    snapshot.graph_cache_hits = p.graph_cache_hits.load(std::memory_order_relaxed);
    snapshot.graph_cache_misses = p.graph_cache_misses.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_cache_hits = p.q4k_repacked_gemv_cache_hits.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_cache_waited_hits =
        p.q4k_repacked_gemv_cache_waited_hits.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_cache_misses = p.q4k_repacked_gemv_cache_misses.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_cache_evictions =
        p.q4k_repacked_gemv_cache_evictions.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_cache_evicted_bytes =
        p.q4k_repacked_gemv_cache_evicted_bytes.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_repack_bytes = p.q4k_repacked_gemv_repack_bytes.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_probe_ns = p.q4k_repacked_gemv_probe_ns.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_resident_bytes =
        p.q4k_repacked_gemv_resident_bytes.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_distinct_weights_seen =
        p.q4k_repacked_gemv_distinct_weights_seen.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_repeated_repack_count =
        p.q4k_repacked_gemv_repeated_repack_count.load(std::memory_order_relaxed);
    snapshot.q4k_copied_gemv_experiment_cache_hits =
        p.q4k_copied_gemv_experiment_cache_hits.load(std::memory_order_relaxed);
    snapshot.q4k_copied_gemv_experiment_cache_misses =
        p.q4k_copied_gemv_experiment_cache_misses.load(std::memory_order_relaxed);
    snapshot.qact_cache_hits = p.qact_cache_hits.load(std::memory_order_relaxed);
    snapshot.qact_cache_misses = p.qact_cache_misses.load(std::memory_order_relaxed);
    snapshot.qact_cache_reused_bytes = p.qact_cache_reused_bytes.load(std::memory_order_relaxed);
    snapshot.moe_decode_scratch_reused = p.moe_decode_scratch_reused.load(std::memory_order_relaxed);
    snapshot.moe_decode_allocations_avoided = p.moe_decode_allocations_avoided.load(std::memory_order_relaxed);
    snapshot.moe_task_count = p.moe_task_count.load(std::memory_order_relaxed);
    snapshot.moe_rowblock_used = p.moe_rowblock_used.load(std::memory_order_relaxed);
    snapshot.moe_rowblock_tasks = p.moe_rowblock_tasks.load(std::memory_order_relaxed);
    snapshot.selected_expert_count = p.selected_expert_count.load(std::memory_order_relaxed);
    snapshot.ssm_conv1d_calls = p.ssm_conv1d_calls.load(std::memory_order_relaxed);
    snapshot.ssm_delta_calls = p.ssm_delta_calls.load(std::memory_order_relaxed);
    snapshot.q4k_true_batched_used = p.q4k_true_batched_used.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_batched_mode = p.qwen36_prefill_q4k_batched_mode.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_batched_used = p.qwen36_prefill_q4k_batched_used.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_batched_probe_pass =
        p.qwen36_prefill_q4k_batched_probe_pass.load(std::memory_order_relaxed);
    const uint32_t max_abs_bits = p.qwen36_prefill_q4k_batched_max_abs_error_bits.load(std::memory_order_relaxed);
    std::memcpy(&snapshot.qwen36_prefill_q4k_batched_max_abs_error, &max_abs_bits, sizeof(float));
    snapshot.qwen36_prefill_q4k_batched_last_reject_reason =
        p.qwen36_prefill_q4k_batched_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_probe_participants =
        p.qwen36_prefill_q4k_probe_participants.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_probe_failures =
        p.qwen36_prefill_q4k_probe_failures.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_q4k_admission_downgraded =
        p.qwen36_prefill_q4k_admission_downgraded.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_mode = p.qwen36_ssm_q8_prefill_amx_mode.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_prepared =
        p.qwen36_ssm_q8_prefill_amx_prepared.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_used = p.qwen36_ssm_q8_prefill_amx_used.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_last_reject_reason =
        p.qwen36_ssm_q8_prefill_amx_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_qkv_count =
        p.qwen36_ssm_q8_prefill_amx_qkv_count.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_gate_count =
        p.qwen36_ssm_q8_prefill_amx_gate_count.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_out_count =
        p.qwen36_ssm_q8_prefill_amx_out_count.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_candidate_ops =
        p.qwen36_ssm_q8_prefill_amx_candidate_ops.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_used_ops =
        p.qwen36_ssm_q8_prefill_amx_used_ops.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_prefill_amx_rejected_ops =
        p.qwen36_ssm_q8_prefill_amx_rejected_ops.load(std::memory_order_relaxed);
    snapshot.qwen36_ssm_q8_decode_used_original_q8_path =
        p.qwen36_ssm_q8_decode_used_original_q8_path.load(std::memory_order_relaxed);
    SnapshotAtomicHistogram(snapshot.qwen36_ssm_projection_weight_type_hist,
                            p.qwen36_ssm_projection_weight_type_hist);
    snapshot.q4k_repacked_gemv_used = p.q4k_repacked_gemv_used.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_seen_ops = p.q4k_repacked_gemv_seen_ops.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_candidate_ops =
        p.q4k_repacked_gemv_candidate_ops.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_used_ops = p.q4k_repacked_gemv_used_ops.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_rejected_ops =
        p.q4k_repacked_gemv_rejected_ops.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_last_reject_reason =
        p.q4k_repacked_gemv_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.q4k_repacked_gemv_primary_disable_reason =
        p.q4k_repacked_gemv_primary_disable_reason.load(std::memory_order_relaxed);
    snapshot.gemv_custom_total_ops = p.gemv_custom_total_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_decode_ops = p.gemv_custom_decode_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_prefill_ops = p.gemv_custom_prefill_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_q4k_seen_ops = p.gemv_custom_q4k_seen_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_non_q4k_ops = p.gemv_custom_non_q4k_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_quant_input_null_ops = p.gemv_custom_quant_input_null_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_shape_reject_ops = p.gemv_custom_shape_reject_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_phase_unknown_ops = p.gemv_custom_phase_unknown_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_force_reference_ops = p.gemv_custom_force_reference_ops.load(std::memory_order_relaxed);
    snapshot.gemv_custom_dynamic_lora_ops = p.gemv_custom_dynamic_lora_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_decode_lm_head_custom_gemv_used_ops =
        p.lfm2_decode_lm_head_custom_gemv_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_decode_lm_head_custom_gemv_ns =
        p.lfm2_decode_lm_head_custom_gemv_ns.load(std::memory_order_relaxed);
    SnapshotAtomicHistogram(snapshot.gemv_custom_weight_type_hist, p.gemv_custom_weight_type_hist);
    SnapshotAtomicHistogram(snapshot.gemv_custom_quant_input_type_hist, p.gemv_custom_quant_input_type_hist);
    snapshot.gemv_custom_tasks_effective = p.gemv_custom_tasks_effective.load(std::memory_order_relaxed);
    snapshot.gemv_custom_tasks_cap_reason = p.gemv_custom_tasks_cap_reason.load(std::memory_order_relaxed);
    snapshot.decode_matmul_created_ops = p.decode_matmul_created_ops.load(std::memory_order_relaxed);
    SnapshotAtomicHistogram(snapshot.decode_matmul_weight_type_hist, p.decode_matmul_weight_type_hist);
    SnapshotAtomicHistogram(snapshot.decode_matmul_path_hist, p.decode_matmul_path_hist);
    SnapshotAtomicHistogram(snapshot.prefill_matmul_weight_type_hist, p.prefill_matmul_weight_type_hist);
    SnapshotAtomicHistogram(snapshot.prefill_matmul_path_hist, p.prefill_matmul_path_hist);
    snapshot.qwen_target_ggml_compute_ops = p.qwen_target_ggml_compute_ops.load(std::memory_order_relaxed);
    snapshot.qwen_target_ggml_matmul_ops = p.qwen_target_ggml_matmul_ops.load(std::memory_order_relaxed);
    snapshot.qwen_target_ggml_matmul_id_ops = p.qwen_target_ggml_matmul_id_ops.load(std::memory_order_relaxed);
    snapshot.qwen_target_ggml_quant_vecdot_ops =
        p.qwen_target_ggml_quant_vecdot_ops.load(std::memory_order_relaxed);
    snapshot.qwen_target_ggml_quantize_kv_ops =
        p.qwen_target_ggml_quantize_kv_ops.load(std::memory_order_relaxed);
    snapshot.qwen_target_ggml_attention_ops = p.qwen_target_ggml_attention_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_seen_ops = p.q6k_gemv_seen_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_candidate_ops = p.q6k_gemv_candidate_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_used_ops = p.q6k_gemv_used_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_rejected_ops = p.q6k_gemv_rejected_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_reject_quant_input_null_ops =
        p.q6k_gemv_reject_quant_input_null_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_reject_unsupported_quant_input_ops =
        p.q6k_gemv_reject_unsupported_quant_input_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_reject_shape_ops = p.q6k_gemv_reject_shape_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_reject_phase_ops = p.q6k_gemv_reject_phase_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_reject_kernel_unavailable_ops =
        p.q6k_gemv_reject_kernel_unavailable_ops.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_last_reject_reason = p.q6k_gemv_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.q6k_gemv_total_ns = p.q6k_gemv_total_ns.load(std::memory_order_relaxed);
    snapshot.moe_small_decode_parallel_candidate_ops =
        p.moe_small_decode_parallel_candidate_ops.load(std::memory_order_relaxed);
    snapshot.moe_small_decode_parallel_used_ops =
        p.moe_small_decode_parallel_used_ops.load(std::memory_order_relaxed);
    snapshot.moe_small_decode_parallel_rejected_ops =
        p.moe_small_decode_parallel_rejected_ops.load(std::memory_order_relaxed);
    SnapshotAtomicHistogram(snapshot.moe_expert_matmul_weight_type_hist, p.moe_expert_matmul_weight_type_hist);
    snapshot.moe_q4k_repacked_candidate_ops = p.moe_q4k_repacked_candidate_ops.load(std::memory_order_relaxed);
    snapshot.moe_q4k_repacked_used_ops = p.moe_q4k_repacked_used_ops.load(std::memory_order_relaxed);
    snapshot.moe_q4k_repacked_rejected_ops = p.moe_q4k_repacked_rejected_ops.load(std::memory_order_relaxed);
    snapshot.moe_q5k_repacked_candidate_ops = p.moe_q5k_repacked_candidate_ops.load(std::memory_order_relaxed);
    snapshot.moe_q5k_repacked_used_ops = p.moe_q5k_repacked_used_ops.load(std::memory_order_relaxed);
    snapshot.moe_q5k_repacked_rejected_ops = p.moe_q5k_repacked_rejected_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_candidate_ops =
        p.gemma4_moe_prefill_quant_batch_candidate_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_used_ops =
        p.gemma4_moe_prefill_quant_batch_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_rejected_ops =
        p.gemma4_moe_prefill_quant_batch_rejected_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops =
        p.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops =
        p.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_gate_up_used =
        p.gemma4_moe_prefill_quant_batch_gate_up_used.load(std::memory_order_relaxed);
    snapshot.gemma4_moe_prefill_quant_batch_down_used =
        p.gemma4_moe_prefill_quant_batch_down_used.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_candidate_layers =
        p.gemma4_native_moe_prefill_candidate_layers.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_used_layers =
        p.gemma4_native_moe_prefill_used_layers.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_rejected_layers =
        p.gemma4_native_moe_prefill_rejected_layers.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_gate_up_ns =
        p.gemma4_native_moe_prefill_gate_up_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_down_ns =
        p.gemma4_native_moe_prefill_down_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_total_ns =
        p.gemma4_native_moe_prefill_total_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops =
        p.gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_duplicate_work_detected =
        p.gemma4_native_moe_prefill_duplicate_work_detected.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_candidate_ops =
        p.gemma4_dense_prefill_native_candidate_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_used_ops =
        p.gemma4_dense_prefill_native_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_rejected_ops =
        p.gemma4_dense_prefill_native_rejected_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_q4k_ops =
        p.gemma4_dense_prefill_native_q4k_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_q8_0_ops =
        p.gemma4_dense_prefill_native_q8_0_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_native_ns =
        p.gemma4_dense_prefill_native_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_replaced_ggml_mul_mat_ops =
        p.gemma4_dense_prefill_replaced_ggml_mul_mat_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_dense_prefill_duplicate_work_detected =
        p.gemma4_dense_prefill_duplicate_work_detected.load(std::memory_order_relaxed);
    snapshot.gemma4_fast_gelu_enabled = p.gemma4_fast_gelu_enabled.load(std::memory_order_relaxed);
    snapshot.gemma4_fast_gelu_used = p.gemma4_fast_gelu_used.load(std::memory_order_relaxed);
    snapshot.gemma4_fast_gelu_ns = p.gemma4_fast_gelu_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_native_moe_prefill_gate_up_fast_gelu_ns =
        p.gemma4_native_moe_prefill_gate_up_fast_gelu_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_candidate_ops =
        p.gemma4_decode_native_candidate_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_used_ops =
        p.gemma4_decode_native_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_rejected_ops =
        p.gemma4_decode_native_rejected_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_moe_used_ops =
        p.gemma4_decode_native_moe_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_dense_used_ops =
        p.gemma4_decode_native_dense_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_lm_head_used_ops =
        p.gemma4_decode_native_lm_head_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_native_ns = p.gemma4_decode_native_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_replaced_ggml_mul_mat_ops =
        p.gemma4_decode_replaced_ggml_mul_mat_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_replaced_ggml_mul_mat_id_ops =
        p.gemma4_decode_replaced_ggml_mul_mat_id_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_decode_duplicate_work_detected =
        p.gemma4_decode_duplicate_work_detected.load(std::memory_order_relaxed);
    snapshot.gemma4_native_int4_gemv_candidate_ops =
        p.gemma4_native_int4_gemv_candidate_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_int4_gemv_used_ops =
        p.gemma4_native_int4_gemv_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_int4_gemv_ns = p.gemma4_native_int4_gemv_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_native_int4_repacked_weight_count =
        p.gemma4_native_int4_repacked_weight_count.load(std::memory_order_relaxed);
    snapshot.gemma4_native_int4_repacked_bytes =
        p.gemma4_native_int4_repacked_bytes.load(std::memory_order_relaxed);
    snapshot.gemma4_native_fused_gateup_used_ops =
        p.gemma4_native_fused_gateup_used_ops.load(std::memory_order_relaxed);
    snapshot.ggml_delegated_quant_gemv_ops = p.ggml_delegated_quant_gemv_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_paged_attention_candidate_ops =
        p.gemma4_native_paged_attention_candidate_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_paged_attention_used_ops =
        p.gemma4_native_paged_attention_used_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_native_paged_attention_ns =
        p.gemma4_native_paged_attention_ns.load(std::memory_order_relaxed);
    snapshot.gemma4_ggml_attention_fallback_ops =
        p.gemma4_ggml_attention_fallback_ops.load(std::memory_order_relaxed);
    snapshot.gemma4_paged_attention_cache_type =
        p.gemma4_paged_attention_cache_type.load(std::memory_order_relaxed);
    snapshot.gemma4_paged_attention_context_len =
        p.gemma4_paged_attention_context_len.load(std::memory_order_relaxed);
    snapshot.gemma4_paged_attention_head_range =
        p.gemma4_paged_attention_head_range.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_candidate_ops =
        p.native_moe_fast_decode_candidate_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_used_ops =
        p.native_moe_fast_decode_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_rejected_ops =
        p.native_moe_fast_decode_rejected_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_w1w3_used_ops =
        p.native_moe_fast_decode_w1w3_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_w2_used_ops =
        p.native_moe_fast_decode_w2_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_decode_ns = p.native_moe_fast_decode_ns.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w1w3_ns = p.native_moe_fast_w1w3_ns.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_ns = p.native_moe_fast_w2_ns.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_reduce_ns = p.native_moe_fast_reduce_ns.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_total_ns = p.native_moe_fast_total_ns.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w1w3_used_ops =
        p.native_moe_fast_w1w3_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_used_ops = p.native_moe_fast_w2_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_q5k_candidate_ops =
        p.native_moe_fast_w2_q5k_candidate_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_q5k_used_ops =
        p.native_moe_fast_w2_q5k_used_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_q5k_rejected_ops =
        p.native_moe_fast_w2_q5k_rejected_ops.load(std::memory_order_relaxed);
    snapshot.native_moe_fast_w2_q5k_ns = p.native_moe_fast_w2_q5k_ns.load(std::memory_order_relaxed);
    snapshot.qwen35_moe_path = p.qwen35_moe_path.load(std::memory_order_relaxed);
    snapshot.qwen35_moe_layers_seen = p.qwen35_moe_layers_seen.load(std::memory_order_relaxed);
    snapshot.qwen35_moe_forward_calls = p.qwen35_moe_forward_calls.load(std::memory_order_relaxed);
    SnapshotAtomicHistogram(snapshot.qwen35_moe_w1w3_weight_type_hist, p.qwen35_moe_w1w3_weight_type_hist);
    SnapshotAtomicHistogram(snapshot.qwen35_moe_w2_weight_type_hist, p.qwen35_moe_w2_weight_type_hist);
    snapshot.qwen35_moe_selected_expert_count =
        p.qwen35_moe_selected_expert_count.load(std::memory_order_relaxed);
    snapshot.qwen35_moe_top_k = p.qwen35_moe_top_k.load(std::memory_order_relaxed);
    snapshot.qwen35_moe_instrumentation_missing =
        p.qwen35_moe_instrumentation_missing.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_total_ns = p.qwen36_prefill_total_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_ssm_projection_ns =
        p.qwen36_prefill_ssm_projection_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_ssm_delta_state_ns =
        p.qwen36_prefill_ssm_delta_state_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_attention_ns = p.qwen36_prefill_attention_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_mlp_or_moe_ns = p.qwen36_prefill_mlp_or_moe_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_graph_build_ns = p.qwen36_prefill_graph_build_ns.load(std::memory_order_relaxed);
    snapshot.qwen36_prefill_graph_execute_ns = p.qwen36_prefill_graph_execute_ns.load(std::memory_order_relaxed);
    snapshot.moe_selected_expert_count = p.moe_selected_expert_count.load(std::memory_order_relaxed);
    snapshot.moe_top_k = p.moe_top_k.load(std::memory_order_relaxed);
    snapshot.moe_expert_parallel_tasks = p.moe_expert_parallel_tasks.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        snapshot.moe_small_decode_parallel_last_reject_reason =
            ctx->moe_small_decode_parallel_last_reject_reason;
        snapshot.moe_q4k_repacked_last_reject_reason = ctx->moe_q4k_repacked_last_reject_reason;
        snapshot.moe_q5k_repacked_last_reject_reason = ctx->moe_q5k_repacked_last_reject_reason;
        snapshot.gemma4_moe_prefill_quant_batch_last_reject_reason =
            ctx->gemma4_moe_prefill_quant_batch_last_reject_reason;
        snapshot.gemma4_native_moe_prefill_last_reject_reason =
            ctx->gemma4_native_moe_prefill_last_reject_reason;
        snapshot.gemma4_dense_prefill_native_last_reject_reason =
            ctx->gemma4_dense_prefill_native_last_reject_reason;
        snapshot.gemma4_decode_native_last_reject_reason =
            ctx->gemma4_decode_native_last_reject_reason;
        snapshot.qwen_target_ggml_compute_last_reason =
            ctx->qwen_target_ggml_compute_last_reason;
        snapshot.qwen_target_ggml_compute_last_op =
            ctx->qwen_target_ggml_compute_last_op;
        snapshot.qwen_target_ggml_compute_target =
            ctx->qwen_target_ggml_compute_target;
        snapshot.native_moe_fast_decode_last_reject_reason =
            ctx->native_moe_fast_decode_last_reject_reason;
        snapshot.native_moe_fast_w2_q5k_last_reject_reason =
            ctx->native_moe_fast_w2_q5k_last_reject_reason;
        snapshot.q6k_gemv_effective_phase = ctx->q6k_gemv_effective_phase;
        snapshot.q6k_gemv_graph_phase = ctx->q6k_gemv_graph_phase;
        snapshot.q6k_gemv_callback_phase = ctx->q6k_gemv_callback_phase;
    }
    {
        std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
        snapshot.matmul_dispatch_top_slow_entries = ctx->matmul_dispatch_census_entries;
        snapshot.decode_matmul_top_shapes = ctx->decode_matmul_shape_entries;
        snapshot.prefill_matmul_top_shapes = ctx->prefill_matmul_shape_entries;
        snapshot.prefill_matmul_ggml_top_shapes = ctx->prefill_matmul_ggml_shape_entries;
        snapshot.q6k_gemv_weight_shapes = ctx->q6k_gemv_shape_entries;
        snapshot.qwen36_prefill_top_slow_ops = ctx->qwen36_prefill_slow_entries;
    }
    SortAndTrimMatmulDispatchCensusEntries(&snapshot.matmul_dispatch_top_slow_entries);
    SortAndTrimMatmulShapeCensusEntries(&snapshot.decode_matmul_top_shapes);
    SortAndTrimMatmulShapeCensusEntries(&snapshot.prefill_matmul_top_shapes);
    SortAndTrimMatmulShapeCensusEntries(&snapshot.prefill_matmul_ggml_top_shapes);
    SortAndTrimMatmulShapeCensusEntries(&snapshot.q6k_gemv_weight_shapes);
    SortAndTrimMatmulShapeCensusEntries(&snapshot.qwen36_prefill_top_slow_ops);
    snapshot.q4k_copied_gemv_experiment_used =
        p.q4k_copied_gemv_experiment_used.load(std::memory_order_relaxed);
    snapshot.q4k_copied_gemv_experiment_last_reject_reason =
        p.q4k_copied_gemv_experiment_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.paged_attn_decode_head_tile_effective =
        p.paged_attn_decode_head_tile_effective.load(std::memory_order_relaxed);
    snapshot.arm_batched_quant_used = p.arm_batched_quant_used.load(std::memory_order_relaxed);
    snapshot.attention_path_paged = p.attention_path_paged.load(std::memory_order_relaxed);
    snapshot.attention_path_standard = p.attention_path_standard.load(std::memory_order_relaxed);
    snapshot.attention_path_portable_flash = p.attention_path_portable_flash.load(std::memory_order_relaxed);
    snapshot.attention_path_native_flash = p.attention_path_native_flash.load(std::memory_order_relaxed);
    snapshot.attention_path_hal = p.attention_path_hal.load(std::memory_order_relaxed);
    snapshot.flash_attention_headseq_prefill_calls =
        p.flash_attention_headseq_prefill_calls.load(std::memory_order_relaxed);
    snapshot.flash_attention_native_decode_calls =
        p.flash_attention_native_decode_calls.load(std::memory_order_relaxed);
    snapshot.flash_attention_reference_calls = p.flash_attention_reference_calls.load(std::memory_order_relaxed);
    snapshot.flash_attention_non_avx512_tiled_calls =
        p.flash_attention_non_avx512_tiled_calls.load(std::memory_order_relaxed);
    snapshot.flash_attention_avx512_tiled_calls =
        p.flash_attention_avx512_tiled_calls.load(std::memory_order_relaxed);
    snapshot.flash_attention_last_nth = p.flash_attention_last_nth.load(std::memory_order_relaxed);
    snapshot.flash_attention_last_active_threads =
        p.flash_attention_last_active_threads.load(std::memory_order_relaxed);
    snapshot.kleidiai_compiled_enabled = p.kleidiai_compiled_enabled.load(std::memory_order_relaxed);
    snapshot.kleidiai_last_reject_reason = p.kleidiai_last_reject_reason.load(std::memory_order_relaxed);
    return snapshot;
}

static inline void RecordKleidiAIAdmissionDecision(
    InferenceWorkContext* ctx, const densecore::runtime::KernelAdmissionDecision& decision) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    p.kleidiai_compiled_enabled.store(densecore::runtime::KleidiAICompiledEnabled() ? 1 : 0,
                                      std::memory_order_relaxed);
    if (!decision.candidate) {
        return;
    }
    p.kleidiai_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    if (decision.allowed) {
        p.kleidiai_allowed_ops.fetch_add(1, std::memory_order_relaxed);
    } else {
        p.kleidiai_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        p.kleidiai_last_reject_reason.store(static_cast<int>(decision.reject_reason), std::memory_order_relaxed);
    }
}

void AddQwen36SSMProjectionWallProfile(InferenceWorkContext* ctx, uint64_t qkv_ns, uint64_t gate_ns,
                                        uint64_t out_ns) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    p.ssm_qkv_wall_ns.fetch_add(qkv_ns, std::memory_order_relaxed);
    p.ssm_gate_wall_ns.fetch_add(gate_ns, std::memory_order_relaxed);
    p.ssm_out_wall_ns.fetch_add(out_ns, std::memory_order_relaxed);
}

void RecordQwen36SSMQ8PrefillAMXPrepared(InferenceWorkContext* ctx, int mode) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_prepared.store(1, std::memory_order_relaxed);
    ctx->qwen36_profile.qwen36_ssm_q8_prefill_amx_mode.store(mode, std::memory_order_relaxed);
}

void RecordQwen36SSMProjectionWeightType(InferenceWorkContext* ctx, ggml_type weight_type) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile
        .qwen36_ssm_projection_weight_type_hist[MatmulWeightTypeHistIndex(weight_type)]
        .fetch_add(1, std::memory_order_relaxed);
}

void RecordMoESmallDecodeParallelDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                          const char* reject_reason, int selected_expert_count, int top_k,
                                          int task_count) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.moe_small_decode_parallel_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.moe_small_decode_parallel_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (candidate) {
        p.moe_small_decode_parallel_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        if (reject_reason && reject_reason[0] != '\0') {
            std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
            ctx->moe_small_decode_parallel_last_reject_reason = reject_reason;
        }
    }
    const auto update_max = [](std::atomic<int>& counter, int value) {
        int current = counter.load(std::memory_order_relaxed);
        while (value > current && !counter.compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
    };
    if (selected_expert_count > 0) update_max(p.moe_selected_expert_count, selected_expert_count);
    if (top_k > 0) update_max(p.moe_top_k, top_k);
    if (task_count > 0) update_max(p.moe_expert_parallel_tasks, task_count);
}

void RecordMoEExpertMatmulWeightType(InferenceWorkContext* ctx, ggml_type weight_type) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile
        .moe_expert_matmul_weight_type_hist[MatmulWeightTypeHistIndex(weight_type)]
        .fetch_add(1, std::memory_order_relaxed);
}

void RecordMoEQ4KRepackedDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.moe_q4k_repacked_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.moe_q4k_repacked_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (candidate) {
        p.moe_q4k_repacked_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        if (reject_reason && reject_reason[0] != '\0') {
            std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
            ctx->moe_q4k_repacked_last_reject_reason = reject_reason;
        }
    }
}

void RecordMoEQ5KRepackedDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.moe_q5k_repacked_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.moe_q5k_repacked_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (candidate) {
        p.moe_q5k_repacked_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        if (reject_reason && reject_reason[0] != '\0') {
            std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
            ctx->moe_q5k_repacked_last_reject_reason = reject_reason;
        }
    }
}

void RecordGemma4MoEPrefillQuantBatchDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                              const char* reject_reason, bool gate_up_used, bool down_used) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.gemma4_moe_prefill_quant_batch_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.gemma4_moe_prefill_quant_batch_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (candidate && reject_reason && reject_reason[0] != '\0') {
        p.gemma4_moe_prefill_quant_batch_rejected_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (reject_reason && reject_reason[0] != '\0') {
        p.gemma4_moe_prefill_quant_batch_rejected_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (reject_reason && std::strcmp(reject_reason, "unsupported_gate_up_shape_or_type") == 0) {
        p.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (reject_reason && std::strcmp(reject_reason, "unsupported_down_shape_or_type") == 0) {
        p.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (gate_up_used) {
        p.gemma4_moe_prefill_quant_batch_gate_up_used.fetch_add(1, std::memory_order_relaxed);
    }
    if (down_used) {
        p.gemma4_moe_prefill_quant_batch_down_used.fetch_add(1, std::memory_order_relaxed);
    }
    if (reject_reason && reject_reason[0] != '\0') {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        ctx->gemma4_moe_prefill_quant_batch_last_reject_reason = reject_reason;
    }
}

void RecordGemma4NativeMoEPrefillDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                          const char* reject_reason, uint64_t replaced_mul_mat_id_ops,
                                          bool duplicate_work_detected) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.gemma4_native_moe_prefill_candidate_layers.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.gemma4_native_moe_prefill_used_layers.fetch_add(1, std::memory_order_relaxed);
    } else if (candidate && reject_reason && reject_reason[0] != '\0') {
        p.gemma4_native_moe_prefill_rejected_layers.fetch_add(1, std::memory_order_relaxed);
    }
    if (replaced_mul_mat_id_ops > 0) {
        p.gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops.fetch_add(replaced_mul_mat_id_ops,
                                                                           std::memory_order_relaxed);
    }
    if (duplicate_work_detected) {
        p.gemma4_native_moe_prefill_duplicate_work_detected.fetch_add(1, std::memory_order_relaxed);
    }
    if (reject_reason && reject_reason[0] != '\0') {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        ctx->gemma4_native_moe_prefill_last_reject_reason = reject_reason;
    }
}

void RecordGemma4NativeMoEPrefillTiming(InferenceWorkContext* ctx, uint64_t gate_up_ns, uint64_t down_ns,
                                        uint64_t total_ns) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (gate_up_ns > 0) {
        p.gemma4_native_moe_prefill_gate_up_ns.fetch_add(gate_up_ns, std::memory_order_relaxed);
    }
    if (down_ns > 0) {
        p.gemma4_native_moe_prefill_down_ns.fetch_add(down_ns, std::memory_order_relaxed);
    }
    if (total_ns > 0) {
        p.gemma4_native_moe_prefill_total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    }
}

void RecordGemma4DensePrefillNativeDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                            const char* reject_reason, ggml_type weight_type,
                                            uint64_t replaced_mul_mat_ops, bool duplicate_work_detected) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.gemma4_dense_prefill_native_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.gemma4_dense_prefill_native_used_ops.fetch_add(1, std::memory_order_relaxed);
        if (weight_type == GGML_TYPE_Q4_K) {
            p.gemma4_dense_prefill_native_q4k_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (weight_type == GGML_TYPE_Q8_0) {
            p.gemma4_dense_prefill_native_q8_0_ops.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (candidate && reject_reason && reject_reason[0] != '\0') {
        p.gemma4_dense_prefill_native_rejected_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (replaced_mul_mat_ops > 0) {
        p.gemma4_dense_prefill_replaced_ggml_mul_mat_ops.fetch_add(replaced_mul_mat_ops,
                                                                   std::memory_order_relaxed);
    }
    if (duplicate_work_detected) {
        p.gemma4_dense_prefill_duplicate_work_detected.fetch_add(1, std::memory_order_relaxed);
    }
    if (reject_reason && reject_reason[0] != '\0') {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        ctx->gemma4_dense_prefill_native_last_reject_reason = reject_reason;
    }
}

void RecordGemma4DensePrefillNativeTiming(InferenceWorkContext* ctx, uint64_t wall_ns) {
    if (!ctx || wall_ns == 0) {
        return;
    }
    ctx->qwen36_profile.gemma4_dense_prefill_native_ns.fetch_add(wall_ns, std::memory_order_relaxed);
}

void RecordGemma4FastGeluDecision(InferenceWorkContext* ctx, bool enabled, bool used, uint64_t wall_ns) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (enabled) {
        p.gemma4_fast_gelu_enabled.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.gemma4_fast_gelu_used.fetch_add(1, std::memory_order_relaxed);
    }
    if (wall_ns > 0) {
        p.gemma4_fast_gelu_ns.fetch_add(wall_ns, std::memory_order_relaxed);
        p.gemma4_native_moe_prefill_gate_up_fast_gelu_ns.fetch_add(wall_ns, std::memory_order_relaxed);
    }
}

void RecordGemma4DecodeNativeDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                      const char* reject_reason, bool moe_used, bool dense_used,
                                      bool lm_head_used, uint64_t wall_ns, uint64_t replaced_mul_mat_ops,
                                      uint64_t replaced_mul_mat_id_ops, bool duplicate_work_detected) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.gemma4_decode_native_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.gemma4_decode_native_used_ops.fetch_add(1, std::memory_order_relaxed);
        if (moe_used) {
            p.gemma4_decode_native_moe_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (dense_used) {
            p.gemma4_decode_native_dense_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (lm_head_used) {
            p.gemma4_decode_native_lm_head_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (candidate && reject_reason && reject_reason[0] != '\0') {
        p.gemma4_decode_native_rejected_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (wall_ns > 0) {
        p.gemma4_decode_native_ns.fetch_add(wall_ns, std::memory_order_relaxed);
    }
    if (replaced_mul_mat_ops > 0) {
        p.gemma4_decode_replaced_ggml_mul_mat_ops.fetch_add(replaced_mul_mat_ops, std::memory_order_relaxed);
    }
    if (replaced_mul_mat_id_ops > 0) {
        p.gemma4_decode_replaced_ggml_mul_mat_id_ops.fetch_add(replaced_mul_mat_id_ops,
                                                               std::memory_order_relaxed);
    }
    if (duplicate_work_detected) {
        p.gemma4_decode_duplicate_work_detected.fetch_add(1, std::memory_order_relaxed);
    }
    if (reject_reason && reject_reason[0] != '\0') {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        ctx->gemma4_decode_native_last_reject_reason = reject_reason;
    }
}

void RecordNativeMoEFastDecodeDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                       const char* reject_reason, bool w1w3_used, bool w2_used,
                                       uint64_t wall_ns) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.native_moe_fast_decode_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.native_moe_fast_decode_used_ops.fetch_add(1, std::memory_order_relaxed);
        if (w1w3_used) {
            p.native_moe_fast_decode_w1w3_used_ops.fetch_add(1, std::memory_order_relaxed);
            p.native_moe_fast_w1w3_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (w2_used) {
            p.native_moe_fast_decode_w2_used_ops.fetch_add(1, std::memory_order_relaxed);
            p.native_moe_fast_w2_used_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (wall_ns != 0) {
            p.native_moe_fast_decode_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            p.native_moe_fast_total_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            if (w1w3_used) {
                p.native_moe_fast_w1w3_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            }
        }
    } else if (candidate) {
        p.native_moe_fast_decode_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        if (reject_reason && reject_reason[0] != '\0') {
            std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
            ctx->native_moe_fast_decode_last_reject_reason = reject_reason;
        }
    }
}

void RecordNativeMoEFastW2Q5KDecision(InferenceWorkContext* ctx, bool candidate, bool used,
                                      const char* reject_reason, uint64_t wall_ns) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (candidate) {
        p.native_moe_fast_w2_q5k_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.native_moe_fast_w2_q5k_used_ops.fetch_add(1, std::memory_order_relaxed);
        if (wall_ns != 0) {
            p.native_moe_fast_w2_q5k_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            p.native_moe_fast_w2_ns.fetch_add(wall_ns, std::memory_order_relaxed);
        }
    } else if (candidate) {
        p.native_moe_fast_w2_q5k_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        if (reject_reason && reject_reason[0] != '\0') {
            std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
            ctx->native_moe_fast_w2_q5k_last_reject_reason = reject_reason;
        }
    }
}

void RecordMatmulDispatchCensus(InferenceWorkContext* ctx, InferenceExecutionPhase phase, const char* dispatch_path,
                                ggml_type weight_type, int64_t m, int64_t n, int64_t k, uint64_t wall_ns) {
    if (!ctx || wall_ns == 0 || !dispatch_path || dispatch_path[0] == '\0') {
        return;
    }
    const BatchSpec* batch = ctx->batch ? ctx->batch : GetCurrentBatch();
    if (!ResolveFastPathRuntimeConfig(batch).matmul_dispatch_census) {
        return;
    }
    MatmulDispatchCensusEntry entry;
    entry.phase = MatmulPhaseName(phase);
    entry.model_family = MatmulModelFamilyName(ctx->model_variant);
    entry.dispatch_path = dispatch_path;
    entry.weight_type = MatmulWeightTypeHistLabel(MatmulWeightTypeHistIndex(weight_type));
    entry.shape_bucket = MatmulShapeBucket(m, n, k);
    entry.wall_ns = wall_ns;
    entry.ops = 1;
    std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
    AddMatmulDispatchCensusEntry(&ctx->matmul_dispatch_census_entries, entry);
}

void RecordGraphBuildMatmulCensus(InferenceWorkContext* ctx, InferenceExecutionPhase phase, const char* dispatch_path,
                                  ggml_type weight_type, int64_t m, int64_t n, int64_t k, const char* left_name,
                                  const char* right_name, bool expected_decode) {
    if (!ctx || !dispatch_path || dispatch_path[0] == '\0') {
        return;
    }
    auto& p = ctx->qwen36_profile;
    const std::size_t weight_idx = MatmulWeightTypeHistIndex(weight_type);
    const std::size_t path_idx = MatmulPathHistIndex(dispatch_path);
    if (phase == InferenceExecutionPhase::Decode || expected_decode) {
        p.decode_matmul_created_ops.fetch_add(1, std::memory_order_relaxed);
        p.decode_matmul_weight_type_hist[weight_idx].fetch_add(1, std::memory_order_relaxed);
        p.decode_matmul_path_hist[path_idx].fetch_add(1, std::memory_order_relaxed);
        MatmulShapeCensusEntry entry;
        entry.phase = "decode";
        entry.model_family = MatmulModelFamilyName(ctx->model_variant);
        entry.dispatch_path = MatmulPathHistLabel(path_idx);
        entry.weight_type = MatmulWeightTypeHistLabel(weight_idx);
        entry.shape_bucket = MatmulShapeBucket(m, n, k);
        entry.left_name = left_name && left_name[0] ? left_name : "unnamed_weight";
        entry.right_name = right_name && right_name[0] ? right_name : "unnamed_input";
        entry.ops = 1;
        std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
        AddMatmulShapeCensusEntry(&ctx->decode_matmul_shape_entries, entry);
    } else if (phase == InferenceExecutionPhase::Prefill) {
        p.prefill_matmul_weight_type_hist[weight_idx].fetch_add(1, std::memory_order_relaxed);
        p.prefill_matmul_path_hist[path_idx].fetch_add(1, std::memory_order_relaxed);
        MatmulShapeCensusEntry entry;
        entry.phase = "prefill";
        entry.model_family = MatmulModelFamilyName(ctx->model_variant);
        entry.dispatch_path = MatmulPathHistLabel(path_idx);
        entry.weight_type = MatmulWeightTypeHistLabel(weight_idx);
        entry.shape_bucket = MatmulShapeBucket(m, n, k);
        entry.left_name = left_name && left_name[0] ? left_name : "unnamed_weight";
        entry.right_name = right_name && right_name[0] ? right_name : "unnamed_input";
        entry.ops = 1;
        std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
        AddMatmulShapeCensusEntry(&ctx->prefill_matmul_shape_entries, entry);
        if (path_idx == 0 || path_idx == 1) {
            AddMatmulShapeCensusEntry(&ctx->prefill_matmul_ggml_shape_entries, entry);
        }
    }
}

void RecordQwenTargetGgmlComputeFallback(InferenceWorkContext* ctx, const TransformerModel* model,
                                         densecore::runtime::GgmlComputeOp op, const char* reason,
                                         const char* tensor_name, InferenceExecutionPhase phase) {
    const densecore::runtime::QwenHotPathPlan plan = densecore::runtime::ResolveQwenHotPathPlan(model);
    if (!plan.target_model) {
        return;
    }

    if (densecore::runtime::ShouldRejectQwenGgmlCompute(plan, reason)) {
        std::string message = "Qwen target GGML compute rejected: op=";
        message += densecore::runtime::GgmlComputeOpName(op);
        message += " target=";
        message += densecore::runtime::QwenHotPathTargetLabel(plan);
        message += " phase=";
        message += MatmulPhaseName(phase);
        message += " reason=";
        message += (reason && reason[0] ? reason : "unclassified");
        if (tensor_name && tensor_name[0]) {
            message += " tensor=";
            message += tensor_name;
        }
        throw densecore::InvalidArgumentException(message);
    }

    if (!ctx) {
        return;
    }

    auto& p = ctx->qwen36_profile;
    p.qwen_target_ggml_compute_ops.fetch_add(1, std::memory_order_relaxed);
    switch (op) {
    case densecore::runtime::GgmlComputeOp::Matmul:
        p.qwen_target_ggml_matmul_ops.fetch_add(1, std::memory_order_relaxed);
        break;
    case densecore::runtime::GgmlComputeOp::MatmulId:
        p.qwen_target_ggml_matmul_id_ops.fetch_add(1, std::memory_order_relaxed);
        break;
    case densecore::runtime::GgmlComputeOp::QuantVecDot:
        p.qwen_target_ggml_quant_vecdot_ops.fetch_add(1, std::memory_order_relaxed);
        break;
    case densecore::runtime::GgmlComputeOp::QuantizeKv:
        p.qwen_target_ggml_quantize_kv_ops.fetch_add(1, std::memory_order_relaxed);
        break;
    case densecore::runtime::GgmlComputeOp::Attention:
        p.qwen_target_ggml_attention_ops.fetch_add(1, std::memory_order_relaxed);
        break;
    }

    std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
    ctx->qwen_target_ggml_compute_last_reason = reason && reason[0] ? reason : "unclassified";
    ctx->qwen_target_ggml_compute_last_op = densecore::runtime::GgmlComputeOpName(op);
    ctx->qwen_target_ggml_compute_target = densecore::runtime::QwenHotPathTargetLabel(plan);
}

void RecordGemma4GgmlAttentionFallback(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile.gemma4_ggml_attention_fallback_ops.fetch_add(1, std::memory_order_relaxed);
}

void RecordGemma4NativeFusedGateUpUsed(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->qwen36_profile.gemma4_native_fused_gateup_used_ops.fetch_add(1, std::memory_order_relaxed);
}

void RecordQ6KGemvDecision(InferenceWorkContext* ctx, bool candidate, bool used, const char* reject_reason,
                           const char* weight_name, int64_t m, int64_t n, int64_t k, uint64_t wall_ns,
                           const char* effective_phase, const char* graph_phase, const char* callback_phase,
                           const char* dispatch_path) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    p.q6k_gemv_seen_ops.fetch_add(1, std::memory_order_relaxed);
    if (candidate) {
        p.q6k_gemv_candidate_ops.fetch_add(1, std::memory_order_relaxed);
    }
    if (used) {
        p.q6k_gemv_used_ops.fetch_add(1, std::memory_order_relaxed);
        p.q6k_gemv_total_ns.fetch_add(wall_ns, std::memory_order_relaxed);
    } else {
        p.q6k_gemv_rejected_ops.fetch_add(1, std::memory_order_relaxed);
        p.q6k_gemv_last_reject_reason.store(Q6KGemvRejectReasonCode(reject_reason), std::memory_order_relaxed);
        if (std::strcmp(reject_reason ? reject_reason : "", "quant_input_null") == 0 ||
            std::strcmp(reject_reason ? reject_reason : "", "missing_quant_input") == 0) {
            p.q6k_gemv_reject_quant_input_null_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(reject_reason ? reject_reason : "", "unsupported_quant_input") == 0 ||
                   std::strcmp(reject_reason ? reject_reason : "", "input_type") == 0) {
            p.q6k_gemv_reject_unsupported_quant_input_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(reject_reason ? reject_reason : "", "unsupported_shape") == 0 ||
                   std::strcmp(reject_reason ? reject_reason : "", "shape") == 0) {
            p.q6k_gemv_reject_shape_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(reject_reason ? reject_reason : "", "not_decode") == 0) {
            p.q6k_gemv_reject_phase_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(reject_reason ? reject_reason : "", "kernel_unavailable") == 0 ||
                   std::strcmp(reject_reason ? reject_reason : "", "missing_vec_dot") == 0) {
            p.q6k_gemv_reject_kernel_unavailable_ops.fetch_add(1, std::memory_order_relaxed);
        }
    }
    {
        MatmulShapeCensusEntry entry;
        entry.model_family = MatmulModelFamilyName(ctx->model_variant);
        entry.phase = effective_phase && effective_phase[0] ? effective_phase : MatmulPhaseName(ctx->phase);
        entry.dispatch_path =
            used ? (dispatch_path && dispatch_path[0] ? dispatch_path : "q6k_direct_vecdot")
                 : (candidate ? "candidate_rejected" : "pre_candidate_rejected");
        entry.weight_type = "q6_k";
        entry.shape_bucket = MatmulShapeBucket(m, n, k);
        entry.left_name = weight_name && weight_name[0] ? weight_name : "unnamed_weight";
        entry.right_name = "gemv_input";
        entry.ops = 1;
        std::lock_guard<std::mutex> lock(ctx->matmul_dispatch_census_mutex);
        AddMatmulShapeCensusEntry(&ctx->q6k_gemv_shape_entries, entry);
    }
    if ((effective_phase && effective_phase[0]) || (graph_phase && graph_phase[0]) ||
        (callback_phase && callback_phase[0])) {
        std::lock_guard<std::mutex> lock(ctx->profile_string_mutex);
        if (effective_phase && effective_phase[0]) {
            ctx->q6k_gemv_effective_phase = effective_phase;
        }
        if (graph_phase && graph_phase[0]) {
            ctx->q6k_gemv_graph_phase = graph_phase;
        }
        if (callback_phase && callback_phase[0]) {
            ctx->q6k_gemv_callback_phase = callback_phase;
        }
    }
}

void RecordQwen35MoEGraphPath(InferenceWorkContext* ctx, const char* path, int top_k, int selected_expert_count,
                              ggml_type w1w3_type, ggml_type w2_type) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    p.qwen35_moe_path.store(Qwen35MoEPathCode(path), std::memory_order_relaxed);
    p.qwen35_moe_layers_seen.fetch_add(1, std::memory_order_relaxed);
    p.qwen35_moe_forward_calls.fetch_add(1, std::memory_order_relaxed);
    p.qwen35_moe_w1w3_weight_type_hist[MatmulWeightTypeHistIndex(w1w3_type)].fetch_add(1, std::memory_order_relaxed);
    p.qwen35_moe_w2_weight_type_hist[MatmulWeightTypeHistIndex(w2_type)].fetch_add(1, std::memory_order_relaxed);
    int current_selected = p.qwen35_moe_selected_expert_count.load(std::memory_order_relaxed);
    while (current_selected < selected_expert_count &&
           !p.qwen35_moe_selected_expert_count.compare_exchange_weak(
               current_selected, selected_expert_count, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    int current_top_k = p.qwen35_moe_top_k.load(std::memory_order_relaxed);
    while (current_top_k < top_k &&
           !p.qwen35_moe_top_k.compare_exchange_weak(current_top_k, top_k, std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
    }
}

static inline void AddQwen36ProfileNs(std::atomic<uint64_t>& counter, uint64_t value) {
    if (!IsQwen36ProfilingEnabled() || value == 0) {
        return;
    }
    counter.fetch_add(value, std::memory_order_relaxed);
}

static inline void SetQwen36ProfileMax(std::atomic<int>& counter, int value) {
    if (!IsQwen36ProfilingEnabled() || value <= 0) {
        return;
    }
    int current = counter.load(std::memory_order_relaxed);
    while (current < value &&
           !counter.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

static inline void MarkQwen36ProfileFlag(std::atomic<int>& counter) {
    if (!IsQwen36ProfilingEnabled()) {
        return;
    }
    counter.store(1, std::memory_order_relaxed);
}

InferenceWorkContext* CreateInferenceWorkContext() {
    return new InferenceWorkContext();
}

void DestroyInferenceWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    const BatchSpec* batch = ctx->batch;
    if (batch && g_shared_batch.load(std::memory_order_acquire) == batch) {
        g_shared_batch.store(nullptr, std::memory_order_release);
    }
    if (tls_work_ctx == ctx) {
        tls_work_ctx = nullptr;
    }
    delete ctx;
}

void ResetInferenceWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) return;
    static std::atomic<uint64_t> generation_counter{1};
    ctx->batch = nullptr;
    ctx->phase = InferenceExecutionPhase::Unknown;
    ctx->graph_build_no_alloc = false;
    ctx->execution_generation = generation_counter.fetch_add(1, std::memory_order_relaxed);
    ResetQwen36Profile(ctx);
    ctx->gemma4_shared_kv_states.clear();
    ctx->qkv_index = 0;
    ctx->add_rmsnorm_index = 0;
    ctx->gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->qact_generation = 0;
    ctx->qact_tensor = nullptr;
    ctx->qact_source = nullptr;
    ctx->qact_len = 0;
    ctx->qact_type = GGML_TYPE_COUNT;
    ctx->qact_bytes = 0;
    ctx->qact_slot_id = -1;
    ctx->qact_token_pos = std::numeric_limits<int64_t>::min();
    ctx->qact_buffer.clear();
    {
        std::lock_guard<std::mutex> lock(ctx->q4k_repacked_gemv_request_mutex);
        ctx->q4k_repacked_gemv_repack_counts.clear();
    }
    ctx->gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->gemv_userdata_index = 0;
    ctx->gemv_batched_userdata_index = 0;
    ctx->paged_attention_userdata_index = 0;
    ctx->paged_attention_shared_k_block_ptrs.clear();
    ctx->paged_attention_shared_v_block_ptrs.clear();
    ctx->ssm_conv1d_index = 0;
    ctx->lfm2_shortconv_index = 0;
    ctx->projection_reference_index = 0;
    ctx->rmsnorm_reference_index = 0;
    ctx->attention_core_reference_index = 0;
    ctx->ssm_qwen35_delta_index = 0;
    g_shared_batch.store(nullptr, std::memory_order_release);
}

void ResetCachedDecodeGraphWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) return;
    static std::atomic<uint64_t> generation_counter{1000000000ull};
    ctx->phase = InferenceExecutionPhase::Decode;
    ctx->graph_build_no_alloc = false;
    ctx->execution_generation = generation_counter.fetch_add(1, std::memory_order_relaxed);
    ResetQwen36Profile(ctx);
    ctx->paged_attention_shared_k_block_ptrs.clear();
    ctx->paged_attention_shared_v_block_ptrs.clear();
    ctx->gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->qact_generation = 0;
    ctx->qact_tensor = nullptr;
    ctx->qact_source = nullptr;
    ctx->qact_len = 0;
    ctx->qact_type = GGML_TYPE_COUNT;
    ctx->qact_bytes = 0;
    ctx->qact_slot_id = -1;
    ctx->qact_token_pos = std::numeric_limits<int64_t>::min();
    ctx->qact_buffer.clear();
    {
        std::lock_guard<std::mutex> lock(ctx->q4k_repacked_gemv_request_mutex);
        ctx->q4k_repacked_gemv_repack_counts.clear();
    }
    ctx->gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
}

void SetCurrentWorkContext(InferenceWorkContext* ctx) {
    tls_work_ctx = ctx;
}

InferenceWorkContext* GetCurrentWorkContext() {
    return tls_work_ctx;
}

void SetInferenceWorkContextGraphBuildNoAlloc(InferenceWorkContext* ctx, bool no_alloc) {
    if (!ctx) {
        return;
    }
    ctx->graph_build_no_alloc = no_alloc;
}

bool IsCurrentGraphBuildNoAlloc() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    return ctx && ctx->graph_build_no_alloc;
}

void SetInferenceWorkContextModelVariant(InferenceWorkContext* ctx, ModelVariant variant) {
    if (!ctx) {
        return;
    }
    ctx->model_variant = variant;
}

void SetCurrentExecutionPhase(InferenceExecutionPhase phase) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        return;
    }
    ctx->phase = phase;
}

InferenceExecutionPhase GetCurrentExecutionPhase() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    return ctx ? ctx->phase : InferenceExecutionPhase::Unknown;
}

void SetCurrentBatch(const BatchSpec* batch) {
    if (!batch) {
        throw densecore::InvalidArgumentException("SetCurrentBatch called with null batch");
    }
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("SetCurrentBatch called without active InferenceWorkContext");
    }
    ctx->batch = batch;
    g_shared_batch.store(batch, std::memory_order_release);
}

const BatchSpec* GetCurrentBatch() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (ctx && ctx->batch) {
        return ctx->batch;
    }
    return g_shared_batch.load(std::memory_order_acquire);
}

void ClearCurrentBatch() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (ctx) {
        ctx->batch = nullptr;
    }
    g_shared_batch.store(nullptr, std::memory_order_release);
}

static Gemma4SharedKVState* GetGemma4SharedKVStateSlot(int layer) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemma4SharedKVStateSlot called without active InferenceWorkContext");
    }
    if (layer < 0) {
        throw densecore::InvalidArgumentException("GetGemma4SharedKVStateSlot called with negative layer");
    }
    if (ctx->gemma4_shared_kv_states.size() <= static_cast<size_t>(layer)) {
        ctx->gemma4_shared_kv_states.resize(static_cast<size_t>(layer) + 1);
    }
    return &ctx->gemma4_shared_kv_states[static_cast<size_t>(layer)];
}

static void SetGemma4SharedKVState(int source_layer, ggml_tensor* k, ggml_tensor* v) {
    Gemma4SharedKVState* slot = GetGemma4SharedKVStateSlot(source_layer);
    slot->k = k;
    slot->v = v;
    slot->source_layer = source_layer;
}

static const Gemma4SharedKVState* GetGemma4SharedKVState(int source_layer) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx || source_layer < 0 || static_cast<size_t>(source_layer) >= ctx->gemma4_shared_kv_states.size()) {
        return nullptr;
    }
    const Gemma4SharedKVState& slot = ctx->gemma4_shared_kv_states[static_cast<size_t>(source_layer)];
    if (!slot.k || !slot.v || slot.source_layer != source_layer) {
        return nullptr;
    }
    return &slot;
}

// NOTE: KVCacheUserData is defined in runtime/inference_types_internal.h
// Pool of KVCacheUserData to avoid allocation per layer
// Max layers supported: 128 (enough for any current model)
// Each layer needs 2 entries (K and V), so 256 total slots
static constexpr int kMaxKVCacheUserDataSlots = 256;

// Helper to get a userdata slot (no allocation, no leak)
// NOTE: Not inline - needs external linkage for graph_builders/
KVCacheUserData* GetKVCacheUserData(int layer, bool is_k) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetKVCacheUserData called without active InferenceWorkContext");
    }
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVCacheUserDataSlots) {
        idx = idx % kMaxKVCacheUserDataSlots;  // Wrap for safety
    }
    return &ctx->kv_pool[idx];
}

KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetKVUpdateGatherUserData called without active InferenceWorkContext");
    }
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVUpdateGatherSlots) {
        idx = idx % kMaxKVUpdateGatherSlots;
    }
    return &ctx->kv_update_gather_pool[idx];
}

AddRMSNormUserData* GetAddRMSNormUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetAddRMSNormUserData called without active InferenceWorkContext");
    }
    int idx = ctx->add_rmsnorm_index++;
    if (idx >= kMaxAddRMSNormSlots) {
        // Reset index for next iteration but throw for current overflow
        ctx->add_rmsnorm_index = 0;
        throw densecore::OutOfMemoryException(
            "AddRMSNormUserData pool exhausted (max=" + std::to_string(kMaxAddRMSNormSlots) +
            "). Consider increasing kMaxAddRMSNormSlots for deep models.");
    }
    return &ctx->add_rmsnorm_pool[idx];
}

// Thread-local pool for QKV userdata
static constexpr int kMaxQKVUserDataSlots = 256;

inline QKVUserData* GetQKVUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetQKVUserData called without active InferenceWorkContext");
    }
    int idx = ctx->qkv_index++;
    if (idx >= kMaxQKVUserDataSlots) {
        // Reset index for next iteration but throw for current overflow
        ctx->qkv_index = 0;
        throw densecore::OutOfMemoryException(
            "QKVUserData pool exhausted (max=" + std::to_string(kMaxQKVUserDataSlots) +
            "). Consider increasing kMaxQKVUserDataSlots for deep models.");
    }
    return &ctx->qkv_pool[idx];
}

PagedAttentionUserData* GetPagedAttentionUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetPagedAttentionUserData called without active InferenceWorkContext");
    }
    int idx = ctx->paged_attention_userdata_index++;
    if (idx >= kMaxPagedAttentionUserDataSlots) {
        ctx->paged_attention_userdata_index = 0;
        idx = 0;
    }
    PagedAttentionUserData* ud = &ctx->paged_attention_userdata_pool[idx];
    ud->shared_k_block_ptrs = &ctx->paged_attention_shared_k_block_ptrs;
    ud->shared_v_block_ptrs = &ctx->paged_attention_shared_v_block_ptrs;
    ud->shared_block_ptrs_ready.store(0, std::memory_order_relaxed);
    return ud;
}

inline SSMConv1DUserData* GetSSMConv1DUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetSSMConv1DUserData called without active InferenceWorkContext");
    }
    int idx = ctx->ssm_conv1d_index++;
    if (idx >= 128) {
        ctx->ssm_conv1d_index = 0;
        throw densecore::OutOfMemoryException("SSMConv1DUserData pool exhausted");
    }
    return &ctx->ssm_conv1d_pool[idx];
}

inline LFM2ShortConvUserData* GetLFM2ShortConvUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetLFM2ShortConvUserData called without active InferenceWorkContext");
    }
    int idx = ctx->lfm2_shortconv_index++;
    if (idx >= 128) {
        ctx->lfm2_shortconv_index = 0;
        throw densecore::OutOfMemoryException("LFM2ShortConvUserData pool exhausted");
    }
    return &ctx->lfm2_shortconv_pool[idx];
}

inline ProjectionReferenceUserData* GetProjectionReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetProjectionReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->projection_reference_index++;
    if (idx >= 384) {
        ctx->projection_reference_index = 0;
        idx = 0;
    }
    ctx->projection_reference_pool[idx] = ProjectionReferenceUserData{};
    return &ctx->projection_reference_pool[idx];
}

inline RmsNormReferenceUserData* GetRmsNormReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetRmsNormReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->rmsnorm_reference_index++;
    if (idx >= 256) {
        ctx->rmsnorm_reference_index = 0;
        idx = 0;
    }
    return &ctx->rmsnorm_reference_pool[idx];
}

inline SharedScalarGateReferenceUserData* GetSharedScalarGateReferenceUserData() {
    static thread_local SharedScalarGateReferenceUserData pool[64];
    static thread_local int index = 0;
    if (index >= 64) {
        index = 0;
    }
    return &pool[index++];
}

inline AttentionCoreReferenceUserData* GetAttentionCoreReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetAttentionCoreReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->attention_core_reference_index++;
    if (idx >= 128) {
        ctx->attention_core_reference_index = 0;
        idx = 0;
    }
    return &ctx->attention_core_reference_pool[idx];
}

inline SSMQwen35DeltaUserData* GetSSMQwen35DeltaUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetSSMQwen35DeltaUserData called without active InferenceWorkContext");
    }
    int idx = ctx->ssm_qwen35_delta_index++;
    if (idx >= 128) {
        ctx->ssm_qwen35_delta_index = 0;
        throw densecore::OutOfMemoryException("SSMQwen35DeltaUserData pool exhausted");
    }
    return &ctx->ssm_qwen35_delta_pool[idx];
}

/**
 * Custom callback for fused Q/K/V projection with HYBRID parallelism
 *
 * Implements a smart dispatch strategy based on batch size:
 *
 * CASE A - SINGLE-TOKEN DECODE (n_tokens == 1):
 *   - All threads iterate through ALL tokens
 *   - Pass real ith/nth to ComputeQKV for TENSOR PARALLELISM
 *   - Multiple threads collaborate on each token's output dimensions
 *   - Only used for M=1 where column-splitting is the sole parallelism option
 *
 * CASE B - BATCHED DECODE / PREFILL (n_tokens >= 2):
 *   - Partition tokens across threads (TOKEN PARALLELISM)
 *   - Each thread computes FULL dimensions for its token subset
 *   - Pass ith=0, nth=1 to ComputeQKV to disable dimension splitting
 *   - More cache-friendly: each thread keeps weights hot in L1/L2
 *   - Avoids synchronization overhead of tensor parallelism for small batches
 */
void cb_compute_qkv(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    // ===========================================================================
    // BARRIER SAFETY CONTRACT:
    // ===========================================================================
    // This callback is invoked by GGML's thread pool. ALL threads must reach the
    // end of this function cleanly, even if they have no work to do.
    //
    // - ComputeQKV handles `start >= end` by doing nothing and returning early
    // - Early returns here are safe ONLY in PREFILL case (token partitioning)
    // - In DECODE case, all threads iterate all tokens (no early return)
    // ===========================================================================
    auto* ud = static_cast<QKVUserData*>(userdata);
    if (!ud || !dst || !src || !dst->data || !src->data || !ud->w_q || !ud->w_k || !ud->w_v) return;
    if (dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) return;

    const int n_embd = static_cast<int>(src->ne[0]);
    const int n_tokens = static_cast<int>(src->ne[1]);
    const int dim_q = ud->dim_q;
    const int dim_k = ud->dim_k;
    const int dim_v = ud->dim_v;
    const int total_dim = dim_q + dim_k + dim_v;

    if (n_embd != ud->n_embd || n_tokens <= 0 || total_dim <= 0) {
        return;
    }
    if (static_cast<int>(dst->ne[0]) != total_dim || static_cast<int>(dst->ne[1]) != n_tokens) {
        return;
    }

    const float* x = reinterpret_cast<const float*>(src->data);
    float* qkv_out = reinterpret_cast<float*>(dst->data);

    // ==========================================================================
    // HYBRID DISPATCH: Choose parallelism strategy based on batch size
    // ==========================================================================
    // Token parallelism preferred when n_tokens >= 2: each thread computes FULL
    // output dimensions for its subset of tokens. This avoids the synchronization
    // overhead of column-splitting and keeps weight data hot in L1/L2 per-thread.
    // Tensor parallelism reserved for single-token decode (M=1) where we MUST
    // split output columns to utilize multiple threads.
    // ==========================================================================

    if (n_tokens < 2) {
        // ========================================================================
        // CASE A: SINGLE-TOKEN DECODE (Tensor Parallelism)
        // ========================================================================
        // Only 1 token: all threads collaborate, each computing a SLICE of dims
        // ========================================================================
        for (int t = 0; t < n_tokens; t++) {
            const float* x_t = x + static_cast<size_t>(t) * n_embd;
            float* token_out = qkv_out + static_cast<size_t>(t) * total_dim;
            float* q_t = token_out;
            float* k_t = token_out + dim_q;
            float* v_t = token_out + dim_q + dim_k;

            // Each thread computes slice [start_col, end_col) of output dimensions
            // ComputeQKV internally partitions: total_cols = dim_q + dim_k + dim_v
            densecore::simd::ComputeQKV(q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, n_embd, dim_q, dim_k, dim_v, ith,
                                        nth  // Enable tensor parallelism
            );
        }
    } else {
        // ========================================================================
        // CASE B: PREFILL (Token Parallelism)
        // ========================================================================
        // Many tokens (prompt processing), partition tokens across threads
        // Each thread computes FULL dimensions for its subset of tokens
        // More cache-friendly: each thread touches contiguous weight rows
        // ========================================================================
        const int tokens_per_thread = (n_tokens + nth - 1) / nth;  // Ceiling div
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

        // Early exit if this thread has no tokens to process
        if (t_start >= n_tokens) return;

        for (int t = t_start; t < t_end; t++) {
            const float* x_t = x + static_cast<size_t>(t) * n_embd;
            float* token_out = qkv_out + static_cast<size_t>(t) * total_dim;
            float* q_t = token_out;
            float* k_t = token_out + dim_q;
            float* v_t = token_out + dim_q + dim_k;

            // Compute FULL dimensions for this token (no dimension splitting)
            // Pass ith=0, nth=1 to disable tensor parallelism within ComputeQKV
            densecore::simd::ComputeQKV(q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, n_embd, dim_q, dim_k, dim_v, 0,
                                        1  // Disable tensor parallelism (single-threaded kernel call)
            );
        }
    }
}

void cb_compute_qkv_map2(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                         int nth, void* userdata) {
    (void)a;  // Placeholder output tensor used only for dst shape allocation.
    cb_compute_qkv(dst, b, ith, nth, userdata);
}

// ============================================================================
// RoPE Table Initialization
// ============================================================================

/**
 * @brief Initialize pre-computed RoPE cos/sin table for the model
 *
 * Populates model->rope_cos_sin with values for all positions and dimensions.
 * Layout: [pos * head_dim + d] = cos/sin pair for position 'pos', dimension 'd'
 * Interleaved format: [cos0, sin0, cos1, sin1, ...]
 *
 * @param model Model to initialize RoPE table for
 */
void InitRoPETable(TransformerModel* model) {
    if (!model) return;

    const int n_ctx = model->hparams.n_ctx;
    int head_dim = model->hparams.n_embd / model->hparams.n_head;
    if (model->hparams.n_embd_head_k > 0) {
        head_dim = model->hparams.n_embd_head_k;
    }
    const float freq_base = model->hparams.rope_freq_base;

    // Reuse RoPETable from densecore/simd/simd_ops.h to avoid code duplication
    densecore::simd::RoPETable table;
    table.Init(n_ctx, head_dim, freq_base);

    // Move the computed data to the model
    model->rope_cos_sin = std::move(table.cos_sin);
    model->rope_head_dim = head_dim;
}

struct RopeCustomOpData {
    const float* cos_sin = nullptr;
    int max_seq_len = 0;
    int head_dim = 0;
    int rope_dim = 0;
};

struct RopeCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    RopeCustomOpData data;
};

void cb_rope_precomputed_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0 || !dst || !dst->src[0] || !dst->src[1]) return;

    const auto* params = reinterpret_cast<const RopeCustomParams*>(dst->op_params);
    if (!params) return;
    const RopeCustomOpData& ud = params->data;
    if (!ud.cos_sin || ud.max_seq_len <= 0 || ud.head_dim <= 0 || ud.rope_dim <= 0) return;

    const struct ggml_tensor* src = dst->src[0];
    const struct ggml_tensor* pos = dst->src[1];
    if (!src->data || !dst->data || !pos->data || pos->type != GGML_TYPE_I32 || src->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return;
    }

    const int head_dim = static_cast<int>(src->ne[0]);
    const int n_heads = static_cast<int>(src->ne[1]);
    const int n_tokens = static_cast<int>(src->ne[2]);
    if (head_dim != ud.head_dim || n_heads <= 0 || n_tokens <= 0) return;

    const int rope_dim = std::min(ud.rope_dim, head_dim);
    const int total_rows = n_heads * n_tokens;
    const int rows_per_task = (total_rows + nth - 1) / nth;
    const int row_start = ith * rows_per_task;
    const int row_end = std::min(total_rows, row_start + rows_per_task);
    if (row_start >= row_end) return;

    const int* pos_data = reinterpret_cast<const int*>(pos->data);
    const bool dense_row = (src->nb[0] == sizeof(float)) && (dst->nb[0] == sizeof(float));

    static thread_local std::vector<float> row_in;
    static thread_local std::vector<float> row_out;
    if (!dense_row) {
        row_in.resize(static_cast<size_t>(head_dim));
        row_out.resize(static_cast<size_t>(head_dim));
    }

    for (int row = row_start; row < row_end; ++row) {
        const int token_idx = row / n_heads;
        const int head_idx = row % n_heads;
        const int pos_value = pos_data[token_idx];

        const char* src_ptr = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2] +
                              static_cast<size_t>(head_idx) * src->nb[1];
        char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2] +
                        static_cast<size_t>(head_idx) * dst->nb[1];

        if (dense_row) {
            densecore::simd::ApplyRoPE(reinterpret_cast<float*>(dst_ptr), reinterpret_cast<const float*>(src_ptr),
                                       ud.cos_sin, &pos_value, 1, head_dim, rope_dim, ud.max_seq_len);
            continue;
        }

        for (int d = 0; d < head_dim; ++d) {
            row_in[d] = *reinterpret_cast<const float*>(src_ptr + static_cast<size_t>(d) * src->nb[0]);
        }
        densecore::simd::ApplyRoPE(row_out.data(), row_in.data(), ud.cos_sin, &pos_value, 1, head_dim, rope_dim,
                                   ud.max_seq_len);
        for (int d = 0; d < head_dim; ++d) {
            *reinterpret_cast<float*>(dst_ptr + static_cast<size_t>(d) * dst->nb[0]) = row_out[d];
        }
    }
}

static inline bool ModelUsesMRoPE(const TransformerModel* model);
static inline int PositionIdsPerToken(const TransformerModel* model);

inline struct ggml_tensor* ggml_rope_precomputed_table(struct ggml_context* ctx, struct ggml_tensor* input,
                                                       struct ggml_tensor* pos, const TransformerModel* model,
                                                       int rope_dim, const BatchSpec* batch) {
    if (!ctx || !input || !pos || !model) return nullptr;
    if (input->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) return nullptr;
    if (ModelUsesMRoPE(model)) return nullptr;
    if (model->rope_cos_sin.empty() || model->rope_head_dim <= 0) return nullptr;

    const int head_dim = static_cast<int>(input->ne[0]);
    const int n_heads = static_cast<int>(input->ne[1]);
    const int n_tokens = static_cast<int>(input->ne[2]);
    if (head_dim <= 0 || n_heads <= 0 || n_tokens <= 0) return nullptr;
    if (model->rope_head_dim < head_dim) return nullptr;

    const int clamped_rope_dim = std::max(0, std::min(rope_dim, head_dim));
    if (clamped_rope_dim == 0) return nullptr;

    struct ggml_tensor* result = ggml_dup_tensor(ctx, input);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = pos;

    RopeCustomParams params = {};
    params.fun = cb_rope_precomputed_custom;
    params.n_tasks = ResolveTaskCount(batch, std::max(1, n_heads * n_tokens));
    params.userdata = nullptr;
    params.data.cos_sin = model->rope_cos_sin.data();
    params.data.max_seq_len = model->hparams.n_ctx;
    params.data.head_dim = head_dim;
    params.data.rope_dim = clamped_rope_dim;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

static inline bool IsTokenHeadDense(const struct ggml_tensor* t, int head_dim) {
    return t && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(head_dim) * sizeof(float);
}

static inline bool IsTokenSpanDense(const struct ggml_tensor* t, int head_dim, int n_head_kv) {
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    return IsTokenHeadDense(t, head_dim) && t->nb[2] == head_block_bytes;
}

static inline void GatherTokenHeadContiguous(const struct ggml_tensor* src, int token_idx, int head_dim, int n_head_kv,
                                             float* out) {
    const char* token_base = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2];
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * n_head_kv * sizeof(float);
    if (IsTokenHeadDense(src, head_dim)) {
        std::memcpy(out, token_base, head_block_bytes);
        return;
    }

    const size_t nb0 = src->nb[0];
    const size_t nb1 = src->nb[1];
    for (int h = 0; h < n_head_kv; ++h) {
        const char* head_base = token_base + static_cast<size_t>(h) * nb1;
        float* out_head = out + static_cast<size_t>(h) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out_head[d] = *reinterpret_cast<const float*>(head_base + static_cast<size_t>(d) * nb0);
        }
    }
}

static inline void ScatterTokenHeadContiguous(const float* in, struct ggml_tensor* dst, int token_idx, int head_dim,
                                              int n_head_kv) {
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * n_head_kv * sizeof(float);
    if (IsTokenHeadDense(dst, head_dim)) {
        std::memcpy(token_base, in, head_block_bytes);
        return;
    }

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    for (int h = 0; h < n_head_kv; ++h) {
        char* head_base = token_base + static_cast<size_t>(h) * nb1;
        const float* in_head = in + static_cast<size_t>(h) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            *reinterpret_cast<float*>(head_base + static_cast<size_t>(d) * nb0) = in_head[d];
        }
    }
}

static inline bool ModelUsesMRoPE(const TransformerModel* model) {
    static const bool force_standard_qwen35_rope = ParseTruthyEnv("DENSECORE_QWEN35_FORCE_STANDARD_ROPE", false);
    if (force_standard_qwen35_rope && model && model->arch == ModelArch::QWEN35) {
        return false;
    }
    return model && model->hparams.rope_sections[0] > 0 && model->hparams.rope_sections[1] > 0;
}

static inline int ModelMRoPEMode(const TransformerModel* model) {
    if (!ModelUsesMRoPE(model)) {
        return GGML_ROPE_TYPE_NORMAL;
    }
    return model->hparams.rope_mrope_interleaved ? GGML_ROPE_TYPE_IMROPE : GGML_ROPE_TYPE_MROPE;
}

static inline int PositionIdsPerToken(const TransformerModel* model) {
    return ModelUsesMRoPE(model) ? GGML_MROPE_SECTIONS : 1;
}

bool PopulatePositionTensor(TransformerModel* model, const BatchSpec& batch, struct ggml_tensor* pos) {
    if (!model || !pos || !pos->data || pos->type != GGML_TYPE_I32) {
        return false;
    }

    const int n_tokens = static_cast<int>(batch.pos.size());
    if (n_tokens <= 0) {
        return false;
    }

    const int ids_per_token = PositionIdsPerToken(model);
    const int expected = n_tokens * ids_per_token;
    if (pos->ne[0] != expected) {
        return false;
    }

    int32_t* dst = reinterpret_cast<int32_t*>(pos->data);
    if (ids_per_token == 1) {
        std::memcpy(dst, batch.pos.data(), static_cast<size_t>(n_tokens) * sizeof(int32_t));
        return true;
    }

    for (int i = 0; i < n_tokens; ++i) {
        const int32_t p = batch.pos[static_cast<size_t>(i)];
        for (int j = 0; j < ids_per_token; ++j) {
            const size_t offset = static_cast<size_t>(j) * static_cast<size_t>(n_tokens) + static_cast<size_t>(i);
            dst[offset] = (ids_per_token == GGML_MROPE_SECTIONS && j == 3) ? 0 : p;
        }
    }

    return true;
}

static inline void GatherTokenSpanHeadContiguous(const struct ggml_tensor* src, int token_idx, int token_count,
                                                 int head_dim, int n_head_kv, float* out) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    const char* token_base = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2];
    if (IsTokenSpanDense(src, head_dim, n_head_kv)) {
        std::memcpy(out, token_base, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }
    for (int i = 0; i < token_count; ++i) {
        GatherTokenHeadContiguous(src, token_idx + i, head_dim, n_head_kv,
                                  out + static_cast<size_t>(i) * static_cast<size_t>(head_dim) * n_head_kv);
    }
}

static inline void ScatterTokenSpanHeadContiguous(const float* in, struct ggml_tensor* dst, int token_idx,
                                                  int token_count, int head_dim, int n_head_kv) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        std::memcpy(token_base, in, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }
    for (int i = 0; i < token_count; ++i) {
        ScatterTokenHeadContiguous(in + static_cast<size_t>(i) * static_cast<size_t>(head_dim) * n_head_kv, dst,
                                   token_idx + i, head_dim, n_head_kv);
    }
}

static inline void ZeroTokenSpanHead(struct ggml_tensor* dst, int token_idx, int token_count, int head_dim,
                                     int n_head_kv) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        std::memset(token_base, 0, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    for (int i = 0; i < token_count; ++i) {
        char* token_ptr = token_base + static_cast<size_t>(i) * dst->nb[2];
        for (int h = 0; h < n_head_kv; ++h) {
            char* head_ptr = token_ptr + static_cast<size_t>(h) * nb1;
            for (int d = 0; d < head_dim; ++d) {
                *reinterpret_cast<float*>(head_ptr + static_cast<size_t>(d) * nb0) = 0.0f;
            }
        }
    }
}

static inline void WriteCurrentBatchKvToCache(const BatchSpec* batch, const struct ggml_tensor* src,
                                              PagedKVCache* cache, int layer, int head_dim, int n_head_kv, bool is_k,
                                              int token_begin, int token_end, int* writes_ok, int* writes_skipped) {
    if (!batch || !src || !cache || !src->data || head_dim <= 0 || n_head_kv <= 0) {
        return;
    }

    const int N = static_cast<int>(batch->tokens.size());
    if (N <= 0) {
        return;
    }

    const int begin = std::max(0, token_begin);
    const int end = (token_end < 0) ? N : std::min(N, token_end);
    if (begin >= end) {
        return;
    }

    const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    const bool src_dense = IsTokenSpanDense(src, head_dim, n_head_kv);
    std::vector<float> packed;
    if (!src_dense) {
        packed.reserve(static_cast<size_t>(std::min(end - begin, BLOCK_SIZE)) * head_block_size);
    }

    const bool debug_gemma4_decode_write = [&]() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0 && batch->tokens.size() == 1 &&
               (layer == 13 || layer == 14);
    }();

    for (int i = begin; i < end;) {
        if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const int seq_id = batch->seq_id[i];
        const int pos = batch->pos[i];
        if (seq_id < 0 || seq_id >= static_cast<int>(batch->block_tables.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const auto& block_table = batch->block_tables[seq_id];
        const int logical_block = pos / BLOCK_SIZE;
        const int slot = pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const int block_id = block_table[logical_block];
        int run = 1;
        while (i + run < end && i + run < static_cast<int>(batch->seq_id.size()) &&
               i + run < static_cast<int>(batch->pos.size())) {
            const int next_seq_id = batch->seq_id[i + run];
            const int next_pos = batch->pos[i + run];
            if (next_seq_id != seq_id || next_pos != pos + run) {
                break;
            }
            if ((next_pos / BLOCK_SIZE) != logical_block) {
                break;
            }
            ++run;
        }

        packed.resize(static_cast<size_t>(run) * head_block_size);
        const float* run_data = nullptr;
        if (src_dense) {
            run_data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                      static_cast<size_t>(i) * src->nb[2]);
        } else {
            GatherTokenSpanHeadContiguous(src, i, run, head_dim, n_head_kv, packed.data());
            run_data = packed.data();
        }
        if (is_k) {
            cache->WriteKSlots(block_id, layer, slot, run, run_data);
        } else {
            cache->WriteVSlots(block_id, layer, slot, run, run_data);
        }
        if (debug_gemma4_decode_write && run == 1) {
            std::vector<float> roundtrip(static_cast<size_t>(run) * head_block_size, 0.0f);
            if (is_k) {
                cache->ReadKSlots(block_id, layer, slot, run, roundtrip.data());
            } else {
                cache->ReadVSlots(block_id, layer, slot, run, roundtrip.data());
            }

            auto log_slice = [&](const char* stage, const float* values) {
                float min_v = std::numeric_limits<float>::infinity();
                float max_v = -std::numeric_limits<float>::infinity();
                float max_abs = 0.0f;
                double sum = 0.0;
                double sum_sq = 0.0;
                double checksum = 0.0;
                int finite_ct = 0;
                int nan_ct = 0;
                int inf_ct = 0;
                for (size_t vi = 0; vi < head_block_size; ++vi) {
                    const float v = values[vi];
                    if (std::isnan(v)) {
                        nan_ct++;
                        continue;
                    }
                    if (!std::isfinite(v)) {
                        inf_ct++;
                        continue;
                    }
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    max_abs = std::max(max_abs, std::fabs(v));
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    checksum += static_cast<double>(vi + 1) * static_cast<double>(v);
                    finite_ct++;
                }
                if (!std::isfinite(min_v)) min_v = 0.0f;
                if (!std::isfinite(max_v)) max_v = 0.0f;
                std::fprintf(stderr,
                             "[GEMMA4_SHARED_KV] action=%s kind=%s layer=%d source_layer=%d seq=%d pos=%d block=%d "
                             "slot=%d head_dim=%d heads=%d nan=%d inf=%d min=%.8g max=%.8g max_abs=%.8g "
                             "mean=%.8g rms=%.8g checksum=%.12g first8=",
                             stage, is_k ? "K" : "V", layer, layer, seq_id, pos, block_id, slot, head_dim, n_head_kv,
                             nan_ct, inf_ct, min_v, max_v, max_abs, finite_ct > 0 ? (sum / finite_ct) : 0.0,
                             finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0, checksum);
                const size_t preview = std::min<size_t>(8, head_block_size);
                for (size_t vi = 0; vi < preview; ++vi) {
                    std::fprintf(stderr, "%s%.8g", vi == 0 ? "" : ",", values[vi]);
                }
                std::fprintf(stderr, "\n");
            };

            log_slice("pre-cache-write", run_data);
            log_slice("post-cache-readback", roundtrip.data());
        }
        if (ShouldRunKvRoundTripProbe(layer, is_k)) {
            std::vector<float> roundtrip(static_cast<size_t>(run) * head_block_size, 0.0f);
            if (is_k) {
                cache->ReadKSlots(block_id, layer, slot, run, roundtrip.data());
            } else {
                cache->ReadVSlots(block_id, layer, slot, run, roundtrip.data());
            }

            float max_abs_diff = 0.0f;
            int max_idx = -1;
            for (size_t ri = 0; ri < roundtrip.size(); ++ri) {
                const float diff = std::fabs(roundtrip[ri] - run_data[ri]);
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    max_idx = static_cast<int>(ri);
                }
            }
            std::fprintf(stderr,
                         "[KV_ROUNDTRIP] layer=%d kind=%s seq=%d pos=%d block=%d slot=%d run=%d cache_type=%s "
                         "max_abs_diff=%g max_idx=%d src=%g dst=%g\n",
                         layer, is_k ? "K" : "V", seq_id, pos, block_id, slot, run, ggml_type_name(cache->cache_type),
                         static_cast<double>(max_abs_diff), max_idx,
                         max_idx >= 0 ? static_cast<double>(run_data[max_idx]) : 0.0,
                         max_idx >= 0 ? static_cast<double>(roundtrip[static_cast<size_t>(max_idx)]) : 0.0);
        }
        if (writes_ok) {
            *writes_ok += run;
        }
        i += run;
    }
}

static inline bool ReadSingleCurrentBatchKvFromCache(const BatchSpec* batch, PagedKVCache* cache, int layer,
                                                     int head_dim, int n_head_kv, bool is_k, int batch_token_idx,
                                                     float* out) {
    if (!batch || !cache || !out || batch_token_idx < 0 || batch_token_idx >= static_cast<int>(batch->tokens.size()) ||
        batch_token_idx >= static_cast<int>(batch->seq_id.size()) ||
        batch_token_idx >= static_cast<int>(batch->pos.size())) {
        return false;
    }

    const int seq_id = batch->seq_id[batch_token_idx];
    const int pos = batch->pos[batch_token_idx];
    if (seq_id < 0 || seq_id >= static_cast<int>(batch->block_tables.size()) || pos < 0) {
        return false;
    }

    const auto& block_table = batch->block_tables[static_cast<size_t>(seq_id)];
    const int logical_block = pos / BLOCK_SIZE;
    const int slot = pos % BLOCK_SIZE;
    if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
        return false;
    }

    const int block_id = block_table[static_cast<size_t>(logical_block)];
    if (block_id < 0) {
        return false;
    }

    if (is_k) {
        cache->ReadKSlots(block_id, layer, slot, 1, out);
    } else {
        cache->ReadVSlots(block_id, layer, slot, 1, out);
    }
    return true;
}

// Custom callback to load K/V history from cache and append current K/V.
// This implementation is stride-safe for both contiguous and view tensors.
void cb_kv_manage(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !src || !dst || !src->data || !dst->data) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    const int n_total = static_cast<int>(src->ne[2]);
    const int n_past = n_total - N;
    if (N < 0 || n_total < 0 || n_past < 0) return;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    int writes_ok = 0;
    int writes_skipped = 0;

    // 1) Write current tokens to cache (parallel over token ranges).
    if (N > 0) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                       &writes_ok, &writes_skipped);
        }
    }

    // 2) Read history [0, n_past) from cache into dst (parallel over tokens).
    if (n_past > 0) {
        const int seq_id = batch->seq_id.empty() ? -1 : batch->seq_id[0];
        const bool has_valid_seq = (seq_id >= 0 && seq_id < static_cast<int>(batch->block_tables.size()));
        const auto* block_table = has_valid_seq ? &batch->block_tables[seq_id] : nullptr;
        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
        if (ud->force_full_history) {
            retention_policy.enabled = false;
            retention_policy.sliding_window = -1;
            retention_policy.sink_tokens = 0;
        }
        KVRetentionSpan retained_history;
        if (has_valid_seq && seq_id < static_cast<int>(batch->n_past.size())) {
            const int seq_n_past = std::max(0, batch->n_past[static_cast<size_t>(seq_id)]);
            retained_history = densecore::llm::config::ComputeKVRetentionSpan(seq_n_past, retention_policy);
        }

        const int tokens_per_thread = (n_past + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_past);

        std::vector<float> packed;
        packed.reserve(static_cast<size_t>(std::max(1, BLOCK_SIZE)) * head_block_size);
        for (int t = t_start; t < t_end;) {
            if (!has_valid_seq) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            if (!block_table || t >= retained_history.history_kept) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            const int token_pos = densecore::llm::config::MapRetainedHistoryIndex(retained_history, t);
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table->size())) {
                ZeroTokenSpanHead(dst, t, 1, head_dim, n_head_kv);
                ++t;
                continue;
            }

            const int block_id = (*block_table)[logical_block];
            int run = 1;
            while (t + run < t_end && t + run < retained_history.history_kept) {
                const int next_token_pos = densecore::llm::config::MapRetainedHistoryIndex(retained_history, t + run);
                if (next_token_pos != token_pos + run) {
                    break;
                }
                if ((next_token_pos / BLOCK_SIZE) != logical_block) {
                    break;
                }
                ++run;
            }
            packed.resize(static_cast<size_t>(run) * head_block_size);
            if (ud->is_k) {
                ud->cache->ReadKSlots(block_id, ud->layer, slot, run, packed.data());
            } else {
                ud->cache->ReadVSlots(block_id, ud->layer, slot, run, packed.data());
            }
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t, run, head_dim, n_head_kv);
            t += run;
        }
    }

    // 3) Append current tokens to dst [n_past, n_total) (parallel over tokens).
    if (N > 0) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);

        if (t_end > t_start) {
            std::vector<float> packed;
            packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
            GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
            ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, t_end - t_start, head_dim, n_head_kv);
        }
    }

    if (IsDebugInferenceStatsEnabled() && ith == 0 && ud->is_k && (ud->layer == 0 || ud->layer == 3)) {
        static int kv_layout_dbg = 0;
        if (kv_layout_dbg < 6) {
            fprintf(stderr,
                    "[KV_LAYOUT #%d] layer=%d N=%d n_past=%d src.nb=[%zu,%zu,%zu] dst.nb=[%zu,%zu,%zu] "
                    "dense(src,dst)=(%d,%d) writes_ok=%d writes_skipped=%d\n",
                    kv_layout_dbg, ud->layer, N, n_past, src->nb[0], src->nb[1], src->nb[2], dst->nb[0], dst->nb[1],
                    dst->nb[2], IsTokenHeadDense(src, head_dim) ? 1 : 0, IsTokenHeadDense(dst, head_dim) ? 1 : 0,
                    writes_ok, writes_skipped);
            kv_layout_dbg++;
        }
    }
}

void cb_kv_write_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !src || !dst || !src->data || !dst->data) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    if (N <= 0) {
        return;
    }

    const int tokens_per_thread = (N + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, N);

    if (t_end > t_start) {
        WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                   nullptr, nullptr);
    }

    const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    if (t_end > t_start) {
        if (IsTokenSpanDense(src, head_dim, n_head_kv) && IsTokenSpanDense(dst, head_dim, n_head_kv)) {
            const char* src_ptr = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(t_start) * src->nb[2];
            char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(t_start) * dst->nb[2];
            std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(t_end - t_start) * head_block_size * sizeof(float));
        } else {
            std::vector<float> packed;
            packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
            GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t_start, t_end - t_start, head_dim, n_head_kv);
        }
    }
}

void cb_kv_update_and_gather_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data || !dst->src[0] || !dst->src[0]->data || nth <= 0) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const struct ggml_tensor* src = dst->src[0];
    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    const int n_total = static_cast<int>(dst->ne[2]);
    const int n_past = n_total - N;
    if (N < 0 || n_total < 0 || n_past < 0) return;

    const bool read_only_shared_kv = ud->read_only_shared_kv;
    const int history_tokens = read_only_shared_kv ? n_total : n_past;
    const char* debug_shared_env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
    const bool debug_gemma4_decode = debug_shared_env && debug_shared_env[0] != '\0' &&
                                     std::strcmp(debug_shared_env, "0") != 0 && (ud->layer == 13 || ud->layer == 14);
    if (debug_gemma4_decode && ith == 0) {
        std::fprintf(stderr,
                     "[GEMMA4_SHARED_KV] action=callback-state kind=%s layer=%d source_layer=%d N=%d n_total=%d "
                     "n_past=%d read_only=%d force_full_history=%d seqs=%d pos0=%d\n",
                     ud->is_k ? "K" : "V", ud->layer, ud->layer, N, n_total, n_past, read_only_shared_kv ? 1 : 0,
                     ud->force_full_history ? 1 : 0, batch->num_seqs,
                     batch->pos.empty() ? -1 : batch->pos[0]);
    }

    if (N > 0 && !read_only_shared_kv) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                       nullptr, nullptr);
        }
    }

    if (history_tokens > 0) {
        const int seq_id = batch->seq_id.empty() ? -1 : batch->seq_id[0];
        const bool has_valid_seq = (seq_id >= 0 && seq_id < static_cast<int>(batch->block_tables.size()));
        const auto* block_table = has_valid_seq ? &batch->block_tables[seq_id] : nullptr;
        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
        if (ud->force_full_history) {
            retention_policy.enabled = false;
            retention_policy.sliding_window = -1;
            retention_policy.sink_tokens = 0;
        }
        KVRetentionSpan retained_history;
        if (has_valid_seq && seq_id < static_cast<int>(batch->n_past.size())) {
            const int seq_n_past = read_only_shared_kv ? history_tokens
                                                       : std::max(0, batch->n_past[static_cast<size_t>(seq_id)]);
            retained_history = densecore::llm::config::ComputeKVRetentionSpan(seq_n_past, retention_policy);
        }

        const int tokens_per_thread = (history_tokens + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, history_tokens);
        std::vector<float> packed;
        const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
        packed.reserve(static_cast<size_t>(std::max(1, BLOCK_SIZE)) * head_block_size);
        for (int t = t_start; t < t_end;) {
            if (!has_valid_seq) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            if (!block_table || t >= retained_history.history_kept) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            const int token_pos = densecore::llm::config::MapRetainedHistoryIndex(retained_history, t);
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table->size())) {
                ZeroTokenSpanHead(dst, t, 1, head_dim, n_head_kv);
                ++t;
                continue;
            }

            const int block_id = (*block_table)[logical_block];
            int run = 1;
            while (t + run < t_end && t + run < retained_history.history_kept) {
                const int next_token_pos = densecore::llm::config::MapRetainedHistoryIndex(retained_history, t + run);
                if (next_token_pos != token_pos + run) {
                    break;
                }
                if ((next_token_pos / BLOCK_SIZE) != logical_block) {
                    break;
                }
                ++run;
            }
            packed.resize(static_cast<size_t>(run) * head_block_size);
            if (ud->is_k) {
                ud->cache->ReadKSlots(block_id, ud->layer, slot, run, packed.data());
            } else {
                ud->cache->ReadVSlots(block_id, ud->layer, slot, run, packed.data());
            }
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t, run, head_dim, n_head_kv);
            t += run;
        }
    }

    if (N > 0 && !read_only_shared_kv) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
            if (N == 1) {
                std::vector<float> packed(head_block_size, 0.0f);
                if (!ReadSingleCurrentBatchKvFromCache(batch, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k,
                                                       t_start, packed.data())) {
                    GatherTokenSpanHeadContiguous(src, t_start, 1, head_dim, n_head_kv, packed.data());
                }
                ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, 1, head_dim, n_head_kv);
            } else if (IsTokenSpanDense(src, head_dim, n_head_kv) && IsTokenSpanDense(dst, head_dim, n_head_kv)) {
                const char* src_ptr =
                    reinterpret_cast<const char*>(src->data) + static_cast<size_t>(t_start) * src->nb[2];
                char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(n_past + t_start) * dst->nb[2];
                std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(t_end - t_start) * head_block_size * sizeof(float));
            } else {
                std::vector<float> packed;
                packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
                GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
                ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, t_end - t_start, head_dim,
                                               n_head_kv);
            }
        }
    }
}

inline struct ggml_tensor* ggml_kv_update_and_gather(struct ggml_context* ctx, struct ggml_tensor* src, int n_total,
                                                     int n_tasks, KVCacheUserData* userdata) {
    if (!ctx || !src || n_total <= 0) {
        return src;
    }
    struct ggml_tensor* args[1] = {src};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, src->ne[0], src->ne[1], n_total, 1, args, 1,
                          cb_kv_update_and_gather_custom, n_tasks, userdata);
}

// ============================================================================
// NEW: Robust KV Cache Update and Gather Callback
// ============================================================================
/**
 * @brief Unified KV Cache Update and Gather Callback
 *
 * This callback performs three operations atomically:
 *   Step A: Write current K/V tokens into PagedKVCache
 *   Step B: Read historical K/V from cache into destination tensor
 *   Step C: Append current K/V to destination tensor after history
 *
 * Key improvements over cb_kv_manage:
 *   - Does NOT rely on ggml_pad assumptions about data placement
 *   - Explicitly controls all memory operations
 *   - Uses src_data pointer from userdata (not src tensor)
 *   - Clear, sequential steps with bounds checking
 *
 * Threading: ith == 0 only to avoid race conditions on cache writes.
 *
 * Input: dst is pre-allocated [head_dim * n_head_kv, n_past + N]
 * Output: dst filled with [history (0..n_past) | current (n_past..n_total)]
 */
void cb_kv_update_and_gather(struct ggml_tensor* dst, const struct ggml_tensor* /* src - unused */, int ith, int nth,
                             void* userdata) {
    (void)nth;  // Unused - single thread execution

    // Only thread 0 performs the work to avoid race conditions
    if (ith != 0) return;

    auto* ud = static_cast<KVUpdateGatherUserData*>(userdata);
    if (!ud || !ud->cache || !ud->batch || !ud->src_tensor) return;

    // Get source data pointer from tensor at runtime (after GGML backend
    // allocates memory)
    const float* src_data = reinterpret_cast<const float*>(ud->src_tensor->data);
    if (!src_data) return;

    const int layer = ud->layer;
    const int head_dim = ud->head_dim;
    const int n_head_kv = ud->n_head_kv;
    const int N = ud->N;
    const int n_past = ud->n_past;
    const bool is_k = ud->is_k;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    const size_t head_block_bytes = head_block_size * sizeof(float);
    const int n_total = n_past + N;

    // ===========================================================================
    // BOUNDS CHECK: Verify destination tensor has sufficient size
    // ===========================================================================
    const size_t expected_bytes = head_block_size * n_total * sizeof(float);
    if (ggml_nbytes(dst) < expected_bytes) {
        // Tensor too small - this indicates a graph construction error
        // Log and return to avoid buffer overflow
        return;
    }

    // ===========================================================================
    // STEP A: Write current tokens to cache
    // ===========================================================================
    // For each new token in the batch, write its K/V to the PagedKVCache
    // ===========================================================================
    for (int i = 0; i < N;) {
        if (i >= static_cast<int>(ud->batch->seq_id.size()) || i >= static_cast<int>(ud->batch->pos.size())) {
            ++i;
            continue;
        }

        const int seq_id = ud->batch->seq_id[i];
        const int pos = ud->batch->pos[i];
        if (seq_id < 0 || seq_id >= static_cast<int>(ud->batch->block_tables.size())) {
            ++i;
            continue;
        }

        const auto& block_table = ud->batch->block_tables[seq_id];
        const int logical_block = pos / BLOCK_SIZE;
        const int slot = pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            ++i;
            continue;
        }

        int run = 1;
        while (i + run < N && i + run < static_cast<int>(ud->batch->seq_id.size()) &&
               i + run < static_cast<int>(ud->batch->pos.size())) {
            const int next_seq_id = ud->batch->seq_id[i + run];
            const int next_pos = ud->batch->pos[i + run];
            if (next_seq_id != seq_id || next_pos != pos + run) {
                break;
            }
            if ((next_pos / BLOCK_SIZE) != logical_block) {
                break;
            }
            ++run;
        }

        const int block_id = block_table[logical_block];
        const float* token_data = src_data + static_cast<size_t>(i) * head_block_size;
        if (is_k) {
            ud->cache->WriteKSlots(block_id, layer, slot, run, token_data);
        } else {
            ud->cache->WriteVSlots(block_id, layer, slot, run, token_data);
        }
        i += run;
    }

    // ===========================================================================
    // STEP B: Gather history from cache into destination tensor [0, n_past)
    // ===========================================================================
    // Read all historical K/V values from the PagedKVCache into dst
    // ===========================================================================
    if (n_past > 0) {
        // Use seq_id from first token (all tokens in batch share same sequence for
        // decode)
        int seq_id = ud->batch->seq_id[0];

        if (seq_id >= 0 && seq_id < static_cast<int>(ud->batch->block_tables.size())) {
            const auto& block_table = ud->batch->block_tables[seq_id];
            const bool dst_dense = IsTokenSpanDense(dst, head_dim, n_head_kv);
            std::vector<float> packed;
            packed.reserve(static_cast<size_t>(BLOCK_SIZE) * head_block_size);

            for (int i = 0; i < n_past;) {
                const int logical_block = i / BLOCK_SIZE;
                const int slot = i % BLOCK_SIZE;
                const int run = std::min(BLOCK_SIZE - slot, n_past - i);

                if (logical_block >= 0 && logical_block < static_cast<int>(block_table.size())) {
                    const int block_id = block_table[logical_block];
                    float* dst_slot =
                        dst_dense ? (reinterpret_cast<float*>(dst->data) + static_cast<size_t>(i) * head_block_size)
                                  : nullptr;
                    if (!dst_dense) {
                        packed.resize(static_cast<size_t>(run) * head_block_size);
                        dst_slot = packed.data();
                    }
                    if (is_k) {
                        ud->cache->ReadKSlots(block_id, layer, slot, run, dst_slot);
                    } else {
                        ud->cache->ReadVSlots(block_id, layer, slot, run, dst_slot);
                    }
                    if (!dst_dense) {
                        ScatterTokenSpanHeadContiguous(dst_slot, dst, i, run, head_dim, n_head_kv);
                    }
                } else {
                    ZeroTokenSpanHead(dst, i, run, head_dim, n_head_kv);
                }
                i += run;
            }
        } else {
            // Invalid sequence - zero-fill entire history section
            ZeroTokenSpanHead(dst, 0, n_past, head_dim, n_head_kv);
        }
    }

    // ===========================================================================
    // STEP C: Append current tokens to destination tensor [n_past, n_total)
    // ===========================================================================
    // Copy the current K/V data after the history section
    // ===========================================================================
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        float* dst_current = reinterpret_cast<float*>(dst->data) + n_past * head_block_size;
        memcpy(dst_current, src_data, N * head_block_bytes);
    } else {
        ScatterTokenSpanHeadContiguous(src_data, dst, n_past, N, head_dim, n_head_kv);
    }
}

// ============================================================================
// Parallel GEMV Callback for Decode-Phase (N=1)
// ============================================================================
// GGML's ggml_mul_mat parallelizes along batch dimension.
// During decode (batch_size=1), there's NO parallelism opportunity.
// This callback uses GemvParallel to parallelize along output dimension.
// ============================================================================

inline GemvUserData* GetGemvUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvUserData called without active InferenceWorkContext");
    }
    int idx = ctx->gemv_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->gemv_userdata_index = 0;
        idx = 0;
    }
    GemvUserData* ud = &ctx->gemv_userdata_pool[idx];
    ud->slot_id = idx;
    ud->quant_input_shared = ctx->gemv_quant_input_shared.data();
    ud->quantized_stamp = &ctx->gemv_quantized_stamp;
    return ud;
}

inline GemvBatchedUserData* GetGemvBatchedUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvBatchedUserData called without active InferenceWorkContext");
    }
    int idx = ctx->gemv_batched_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->gemv_batched_userdata_index = 0;
        idx = 0;
    }
    GemvBatchedUserData* ud = &ctx->gemv_batched_userdata_pool[idx];
    ud->slot_id = idx;
    ud->input_quant_type = GGML_TYPE_F32;
    ud->quant_input_shared = ctx->gemv_batched_quant_input_shared.data();
    ud->quantized_stamp = &ctx->gemv_batched_quantized_stamp;
    ud->work_ctx = ctx;
    ud->qwen36_prefill_q4k_admission_key = 0;
    ud->qwen36_prefill_q4k_probe = false;
    ud->qwen36_prefill_q4k_admitted = false;
    ud->require_q4k_true_batched = false;
    ud->disable_quant_nrc_fast = false;
    ud->gemma4_dense_prefill_native = false;
    ud->lfm2_q8_repacked_batched = false;
    ud->qwen36_ssm_q8_repacked_batched = false;
    ud->qwen36_ssm_q8_direct_batched = false;
    ud->qwen36_prefill_q4k_probe_done.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_failures.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_internal_errors.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_max_abs_error_bits.store(0, std::memory_order_relaxed);
    return ud;
}
