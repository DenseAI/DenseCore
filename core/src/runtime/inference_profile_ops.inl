// Qwen3.6 profiling and telemetry helpers. Requires InferenceWorkContext.
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
    p.qact_cache_hits.store(0, std::memory_order_relaxed);
    p.qact_cache_misses.store(0, std::memory_order_relaxed);
    p.qact_cache_reused_bytes.store(0, std::memory_order_relaxed);
    p.q8_batched_weight_cache_ns.store(0, std::memory_order_relaxed);
    p.q8_batched_activation_quant_ns.store(0, std::memory_order_relaxed);
    p.q8_batched_activation_wait_ns.store(0, std::memory_order_relaxed);
    p.q8_batched_activation_pack_ns.store(0, std::memory_order_relaxed);
    p.q8_batched_compute_ns.store(0, std::memory_order_relaxed);
    p.q8_batched_used_ops.store(0, std::memory_order_relaxed);
    p.q8_batched_true_gemm_ops.store(0, std::memory_order_relaxed);
    p.q8_batched_gemv_ops.store(0, std::memory_order_relaxed);
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
    p.lfm2_greedy_lm_head_argmax_candidate_ops.store(0, std::memory_order_relaxed);
    p.lfm2_greedy_lm_head_argmax_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_greedy_lm_head_argmax_rejected_ops.store(0, std::memory_order_relaxed);
    p.lfm2_greedy_lm_head_argmax_ns.store(0, std::memory_order_relaxed);
    p.lfm2_greedy_lm_head_argmax_last_reject_reason.store(0, std::memory_order_relaxed);
    p.lfm2_w1w3_q4k_vecdot_rowpair_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_w1w3_q4k_vecdot_scalar_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_w1w3_q4k_hwy_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_w1w3_q4k_repacked_used_ops.store(0, std::memory_order_relaxed);
    p.lfm2_w1w3_q5k_hwy_used_ops.store(0, std::memory_order_relaxed);
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
    snapshot.qact_cache_hits = p.qact_cache_hits.load(std::memory_order_relaxed);
    snapshot.qact_cache_misses = p.qact_cache_misses.load(std::memory_order_relaxed);
    snapshot.qact_cache_reused_bytes = p.qact_cache_reused_bytes.load(std::memory_order_relaxed);
    snapshot.q8_batched_weight_cache_ns = p.q8_batched_weight_cache_ns.load(std::memory_order_relaxed);
    snapshot.q8_batched_activation_quant_ns =
        p.q8_batched_activation_quant_ns.load(std::memory_order_relaxed);
    snapshot.q8_batched_activation_wait_ns = p.q8_batched_activation_wait_ns.load(std::memory_order_relaxed);
    snapshot.q8_batched_activation_pack_ns = p.q8_batched_activation_pack_ns.load(std::memory_order_relaxed);
    snapshot.q8_batched_compute_ns = p.q8_batched_compute_ns.load(std::memory_order_relaxed);
    snapshot.q8_batched_used_ops = p.q8_batched_used_ops.load(std::memory_order_relaxed);
    snapshot.q8_batched_true_gemm_ops = p.q8_batched_true_gemm_ops.load(std::memory_order_relaxed);
    snapshot.q8_batched_gemv_ops = p.q8_batched_gemv_ops.load(std::memory_order_relaxed);
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
    snapshot.lfm2_greedy_lm_head_argmax_candidate_ops =
        p.lfm2_greedy_lm_head_argmax_candidate_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_greedy_lm_head_argmax_used_ops =
        p.lfm2_greedy_lm_head_argmax_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_greedy_lm_head_argmax_rejected_ops =
        p.lfm2_greedy_lm_head_argmax_rejected_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_greedy_lm_head_argmax_ns =
        p.lfm2_greedy_lm_head_argmax_ns.load(std::memory_order_relaxed);
    snapshot.lfm2_greedy_lm_head_argmax_last_reject_reason =
        p.lfm2_greedy_lm_head_argmax_last_reject_reason.load(std::memory_order_relaxed);
    snapshot.lfm2_w1w3_q4k_vecdot_rowpair_used_ops =
        p.lfm2_w1w3_q4k_vecdot_rowpair_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_w1w3_q4k_vecdot_scalar_used_ops =
        p.lfm2_w1w3_q4k_vecdot_scalar_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_w1w3_q4k_hwy_used_ops = p.lfm2_w1w3_q4k_hwy_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_w1w3_q4k_repacked_used_ops =
        p.lfm2_w1w3_q4k_repacked_used_ops.load(std::memory_order_relaxed);
    snapshot.lfm2_w1w3_q5k_hwy_used_ops = p.lfm2_w1w3_q5k_hwy_used_ops.load(std::memory_order_relaxed);
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

void RecordLFM2NativeMoEW1W3Kernel(InferenceWorkContext* ctx, const char* kernel_name) {
    if (!ctx || !kernel_name) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    if (std::strcmp(kernel_name, "q4k_vecdot_rowpair") == 0) {
        p.lfm2_w1w3_q4k_vecdot_rowpair_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (std::strcmp(kernel_name, "q4k_vecdot_scalar") == 0) {
        p.lfm2_w1w3_q4k_vecdot_scalar_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (std::strcmp(kernel_name, "q4k_hwy") == 0) {
        p.lfm2_w1w3_q4k_hwy_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (std::strcmp(kernel_name, "q4k_repacked") == 0) {
        p.lfm2_w1w3_q4k_repacked_used_ops.fetch_add(1, std::memory_order_relaxed);
    } else if (std::strcmp(kernel_name, "q5k_hwy") == 0) {
        p.lfm2_w1w3_q5k_hwy_used_ops.fetch_add(1, std::memory_order_relaxed);
    }
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
    const densecore::runtime::TargetFastPathPlan plan = densecore::runtime::ResolveTargetFastPathPlan(model);
    if (!plan.target_model) {
        return;
    }

    if (densecore::runtime::ShouldRejectTargetGgmlCompute(plan, reason)) {
        std::string message = "Target fallback-free GGML compute rejected: op=";
        message += densecore::runtime::GgmlComputeOpName(op);
        message += " target=";
        message += densecore::runtime::TargetFastPathLabel(plan);
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
    ctx->qwen_target_ggml_compute_target = densecore::runtime::TargetFastPathLabel(plan);
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
                              int task_count, ggml_type w1w3_type, ggml_type w2_type) {
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
    auto update_max = [](std::atomic<int>& counter, int value) {
        int current = counter.load(std::memory_order_relaxed);
        while (value > current && !counter.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                                                 std::memory_order_relaxed)) {
        }
    };
    if (selected_expert_count > 0) {
        update_max(p.moe_selected_expert_count, selected_expert_count);
        update_max(p.selected_expert_count, selected_expert_count);
    }
    if (top_k > 0) {
        update_max(p.moe_top_k, top_k);
    }
    if (task_count > 0) {
        update_max(p.moe_task_count, task_count);
        update_max(p.moe_expert_parallel_tasks, task_count);
    }
}

void RecordNativeMoEGraphCallbackExecution(InferenceWorkContext* ctx, int selected_expert_count, int task_count) {
    if (!ctx) {
        return;
    }
    auto& p = ctx->qwen36_profile;
    auto update_max = [](std::atomic<int>& counter, int value) {
        int current = counter.load(std::memory_order_relaxed);
        while (value > current && !counter.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                                                 std::memory_order_relaxed)) {
        }
    };
    if (selected_expert_count > 0) {
        update_max(p.selected_expert_count, selected_expert_count);
        update_max(p.moe_selected_expert_count, selected_expert_count);
    }
    if (task_count > 0) {
        update_max(p.moe_task_count, task_count);
        update_max(p.moe_expert_parallel_tasks, task_count);
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
