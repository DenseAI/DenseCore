// Decode GEMV custom callback and local scoring helpers.
static bool IsBetterArgmaxCandidate(int token, float value, int best_token, float best_value) {
    if (!std::isfinite(value)) {
        return false;
    }
    if (best_token < 0) {
        return true;
    }
    return value > best_value || (value == best_value && token < best_token);
}

static float ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(float value, int token,
                                                         const std::vector<int>& repeated_tokens, size_t& repeat_pos,
                                                         float repetition_penalty, bool repetition_penalty_active) {
    if (!repetition_penalty_active || repeat_pos >= repeated_tokens.size()) {
        return value;
    }
    if (repeated_tokens[repeat_pos] > token) {
        return value;
    }
    while (repeat_pos < repeated_tokens.size() && repeated_tokens[repeat_pos] < token) {
        ++repeat_pos;
    }
    if (repeat_pos >= repeated_tokens.size() || repeated_tokens[repeat_pos] != token) {
        return value;
    }
    if (!std::isfinite(value)) {
        return value;
    }
    size_t repeat_end = repeat_pos + 1;
    while (repeat_end < repeated_tokens.size() && repeated_tokens[repeat_end] == token) {
        ++repeat_end;
    }
    const size_t count = repeat_end - repeat_pos;
    for (size_t rep = 0; rep < count; ++rep) {
        if (value < 0.0f) {
            value *= repetition_penalty;
        } else {
            value /= repetition_penalty;
        }
    }
    repeat_pos = repeat_end;
    return value;
}

/**
 * Custom callback for parallel GEMV (decode-phase)
 *
 * REFACTORED: Uses GGML_OP_CUSTOM signature to allow output tensor shape
 * to be independent of input tensor shape. This fixes memory corruption
 * when Qwen3 projections change dimensions (e.g., 1024 -> 2048).
 *
 * Signature: void (*)(struct ggml_tensor *dst, int ith, int nth, void
 * *userdata) Input tensor accessed via dst->src[0]
 */
void cb_gemv_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<GemvUserData*>(userdata);
    if (!ud || !ud->weight_tensor) return;
    InferenceWorkContext* callback_work_ctx = ud->work_ctx ? ud->work_ctx : GetCurrentWorkContext();
    const BatchSpec* callback_batch =
        (callback_work_ctx && callback_work_ctx->batch) ? callback_work_ctx->batch : GetCurrentBatch();
    const InferenceExecutionPhase callback_phase =
        callback_work_ctx ? callback_work_ctx->phase : ud->phase_snapshot;

    // Extract input tensor from dst->src[0] (GGML_OP_CUSTOM convention)
    const struct ggml_tensor* src = dst->src[0];
    if (!src) return;

    // Extract weight tensor from dst->src[1] (reliable graph topology, not
    // userdata)
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!weight_tensor || !weight_tensor->data) {
        fprintf(stderr, "CRITICAL: GEMV weight tensor is null or has no data\n");
        return;
    }

    // Get data pointers at runtime (guaranteed valid after GGML allocates)
    const float* x_f32 = reinterpret_cast<const float*>(src->data);
    const void* weight_data = weight_tensor->data;
    float* output = reinterpret_cast<float*>(dst->data);

    if (!x_f32 || !weight_data || !output) return;

    // ==========================================================================
    // DIMENSION VALIDATION (using weight tensor from graph, not stale userdata)
    // ==========================================================================
    const int K = static_cast<int>(weight_tensor->ne[1]);  // Output dimension from weight
    const int N = static_cast<int>(weight_tensor->ne[0]);  // Input dimension from weight

    // Validate output tensor matches expected K
    if (dst->ne[0] != K) {
        fprintf(stderr,
                "CRITICAL: GEMV buffer mismatch! dst->ne[0](%ld) != weight->ne[1](%d). "
                "Output tensor was sized incorrectly.\n",
                (long)dst->ne[0], K);
        return;
    }

    const ggml_type weight_type = weight_tensor->type;
    const char* weight_name = weight_tensor->name[0] ? weight_tensor->name : "(unnamed)";
    if (ith == 0 && callback_work_ctx) {
        auto& profile = callback_work_ctx->qwen36_profile;
        profile.gemv_custom_total_ops.fetch_add(1, std::memory_order_relaxed);
        profile.gemv_custom_weight_type_hist[MatmulWeightTypeHistIndex(weight_type)].fetch_add(
            1, std::memory_order_relaxed);
        if (callback_phase == InferenceExecutionPhase::Decode) {
            profile.gemv_custom_decode_ops.fetch_add(1, std::memory_order_relaxed);
        } else if (callback_phase == InferenceExecutionPhase::Prefill) {
            profile.gemv_custom_prefill_ops.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.gemv_custom_phase_unknown_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (weight_type == GGML_TYPE_Q4_K) {
            profile.gemv_custom_q4k_seen_ops.fetch_add(1, std::memory_order_relaxed);
            profile.q4k_repacked_gemv_seen_ops.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.gemv_custom_non_q4k_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (ud->dynamic_lora_active) {
            profile.gemv_custom_dynamic_lora_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (ud->force_reference_scalar) {
            profile.gemv_custom_force_reference_ops.fetch_add(1, std::memory_order_relaxed);
        }
    }
    static const bool debug_gemv_timing = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMV_TIMING");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    const auto gemv_begin =
        (debug_gemv_timing && ith == 0) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool matmul_dispatch_census_enabled =
        ith == 0 && callback_work_ctx && ResolveFastPathRuntimeConfig(callback_batch).matmul_dispatch_census;
    const auto census_begin = matmul_dispatch_census_enabled ? std::chrono::steady_clock::now()
                                                             : std::chrono::steady_clock::time_point{};
    const auto gemma4_decode_begin =
        (ith == 0 && ud->gemma4_decode_native) ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    const auto lfm2_lm_head_begin =
        (ith == 0 && ud->lfm2_decode_lm_head) ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
    const auto q6k_begin = (ith == 0 && weight_type == GGML_TYPE_Q6_K) ? std::chrono::steady_clock::now()
                                                                       : std::chrono::steady_clock::time_point{};
    bool q6k_decision_recorded = false;
    const auto maybe_log_gemv_timing = [&](const char* path) {
        if (!debug_gemv_timing || ith != 0) {
            return;
        }
        static std::atomic<int> log_budget{0};
        const int current = log_budget.fetch_add(1, std::memory_order_relaxed);
        if (current >= 256) {
            return;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gemv_begin).count();
        std::fprintf(stderr,
                     "[GEMV_TIMING] weight=%s type=%s path=%s K_out=%d N_in=%d nth=%d ith0_ms=%.3f\n",
                     weight_name, ggml_type_name(weight_type), path ? path : "unknown", K, N, nth, ms);
    };
    const auto record_gemv_dispatch_census = [&](const char* dispatch_path) {
        uint64_t wall_ns = 0;
        if (census_begin != std::chrono::steady_clock::time_point{}) {
            wall_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - census_begin)
                    .count());
        }
        if (ith == 0 && weight_type == GGML_TYPE_Q6_K && callback_work_ctx && !q6k_decision_recorded) {
            if (wall_ns == 0 && q6k_begin != std::chrono::steady_clock::time_point{}) {
                wall_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - q6k_begin)
                        .count());
            }
            const auto* q6_type_traits_cpu_for_census = ggml_get_type_traits_cpu(weight_type);
            const bool q6_shape_decode_for_census = N > 0 && K > 0 && (N % QK_K) == 0 && K >= 1024 && N >= 1024;
            const bool q6_effective_decode_for_census =
                callback_phase == InferenceExecutionPhase::Decode || q6_shape_decode_for_census;
            const bool q6_direct_path =
                dispatch_path && (std::strcmp(dispatch_path, "q6k_direct_vecdot") == 0 ||
                                  std::strcmp(dispatch_path, "q6k_direct_vecdot_rowpair") == 0);
            const bool q6_decode_candidate = q6_effective_decode_for_census && q6_direct_path;
            const bool q6_used =
                q6_decode_candidate && q6_type_traits_cpu_for_census && q6_type_traits_cpu_for_census->vec_dot;
            const char* reject_reason = "none";
            if (!q6_decode_candidate) {
                reject_reason = "not_decode";
            } else if (!q6_type_traits_cpu_for_census || !q6_type_traits_cpu_for_census->vec_dot) {
                reject_reason = "kernel_unavailable";
            } else if (!q6_used) {
                reject_reason = "fallback";
            }
            const char* callback_phase_name = MatmulPhaseName(callback_phase);
            const char* graph_phase_name = q6_shape_decode_for_census ? "decode_shape" : callback_phase_name;
            const char* effective_phase_name = q6_effective_decode_for_census ? "decode" : callback_phase_name;
            RecordQ6KGemvDecision(callback_work_ctx, q6_decode_candidate, q6_used, reject_reason, weight_name, 1, K, N,
                                  wall_ns, effective_phase_name, graph_phase_name, callback_phase_name, dispatch_path);
            q6k_decision_recorded = true;
        }
        if (ith == 0 && callback_work_ctx && ud->lfm2_decode_lm_head) {
            if (wall_ns == 0 && lfm2_lm_head_begin != std::chrono::steady_clock::time_point{}) {
                wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::steady_clock::now() - lfm2_lm_head_begin)
                                                    .count());
            }
            auto& profile = callback_work_ctx->qwen36_profile;
            profile.lfm2_decode_lm_head_custom_gemv_used_ops.fetch_add(1, std::memory_order_relaxed);
            if (wall_ns > 0) {
                profile.lfm2_decode_lm_head_custom_gemv_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            }
        }
        if (!matmul_dispatch_census_enabled) {
            if (gemma4_decode_begin != std::chrono::steady_clock::time_point{}) {
                wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::steady_clock::now() - gemma4_decode_begin)
                                                    .count());
                RecordGemma4DecodeNativeDecision(callback_work_ctx, /*candidate=*/false, /*used=*/true, nullptr,
                                                 /*moe_used=*/false, /*dense_used=*/true,
                                                 /*lm_head_used=*/ud->gemma4_decode_lm_head, wall_ns,
                                                 /*replaced_mul_mat_ops=*/1, /*replaced_mul_mat_id_ops=*/0,
                                                 /*duplicate_work_detected=*/false);
            }
            return;
        }
        if (gemma4_decode_begin != std::chrono::steady_clock::time_point{}) {
            wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - gemma4_decode_begin)
                                                .count());
            RecordGemma4DecodeNativeDecision(callback_work_ctx, /*candidate=*/false, /*used=*/true, nullptr,
                                             /*moe_used=*/false, /*dense_used=*/true,
                                             /*lm_head_used=*/ud->gemma4_decode_lm_head, wall_ns,
                                             /*replaced_mul_mat_ops=*/1, /*replaced_mul_mat_id_ops=*/0,
                                             /*duplicate_work_detected=*/false);
        }
        RecordMatmulDispatchCensus(callback_work_ctx, callback_phase, dispatch_path, weight_type, 1, K, N, wall_ns);
    };

    // ==========================================================================
    // SHARED PRE-QUANTIZATION (token-position synchronized):
    // Thread 0 quantizes once per decode token position, other threads wait for
    // that position marker and reuse the same quantized buffer.
    // ==========================================================================
    const void* quant_input = nullptr;
    bool used_shared_quant_buffer = false;
    if (ud->input_quant_type != GGML_TYPE_F32) {
        const auto* input_type_traits = ggml_get_type_traits_cpu(ud->input_quant_type);
        if (input_type_traits && input_type_traits->from_float) {
            const size_t quant_input_size = ggml_row_size(ud->input_quant_type, N);
            if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                const BatchSpec* batch = callback_batch;
                const bool has_valid_stamp = ud->slot_id >= 0;
                const uint64_t expected_stamp =
                    ComputeGemvBatchedQuantStamp(batch, 1, ud->slot_id, src->data, weight_tensor->data);

                if (nth <= 1 || !has_valid_stamp) {
                    alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                    quant_input_tls.resize(quant_input_size);
                    input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                    quant_input = quant_input_tls.data();
                } else {
                    if (!ud->quant_input_shared || !ud->quantized_stamp) {
                        alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                        quant_input_tls.resize(quant_input_size);
                        input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                        quant_input = quant_input_tls.data();
                    } else if (ith == 0) {
                        InferenceWorkContext* work_ctx = callback_work_ctx;
                        const auto& qact_config = ResolveFastPathRuntimeConfig(batch);
                        const int64_t token_pos =
                            (batch && batch->num_seqs == 1 && !batch->pos.empty()) ? batch->pos.front()
                                                                                   : std::numeric_limits<int64_t>::min();
                        const uint8_t* cached_qact =
                            QuantizedActivationCacheEnabled(qact_config)
                                ? GetOrFillQuantizedActivationCache(work_ctx, src, src->data, x_f32,
                                                                    static_cast<int64_t>(N), ud->input_quant_type,
                                                                    quant_input_size, ud->slot_id, token_pos,
                                                                    input_type_traits)
                                : nullptr;
                        if (cached_qact) {
                            std::memcpy(ud->quant_input_shared, cached_qact, quant_input_size);
                        } else {
                            input_type_traits->from_float(x_f32, ud->quant_input_shared, static_cast<int64_t>(N));
                        }
                        ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                        quant_input = ud->quant_input_shared;
                        used_shared_quant_buffer = true;
                    } else {
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                            if (spin_count >= 4096) {
                                alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                                quant_input_tls.resize(quant_input_size);
                                input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                                quant_input = quant_input_tls.data();
                                break;
                            }
                            SpinPause(spin_count++);
                        }
                        if (!quant_input) {
                            quant_input = ud->quant_input_shared;
                            used_shared_quant_buffer = true;
                        }
                    }
                }
            }
        }
    }
    if (ith == 0 && quant_input) {
        RecordSharedQuantReuse(used_shared_quant_buffer);
    }
    const bool q4k_candidate_shape_ok = N > 0 && K > 0 && (N % QK_K) == 0 && (K % 8) == 0;
    const bool q4k_repacked_candidate =
        weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K && quant_input &&
        q4k_candidate_shape_ok;
    const bool q6k_candidate_shape_ok = N > 0 && K > 0 && (N % QK_K) == 0 && K >= 1024 && N >= 1024;
    const bool q6k_effective_decode_gemv =
        callback_phase == InferenceExecutionPhase::Decode || q6k_candidate_shape_ok;
    const bool q6k_decode_candidate =
        weight_type == GGML_TYPE_Q6_K && q6k_effective_decode_gemv && ud->input_quant_type == GGML_TYPE_Q8_K &&
        quant_input && q6k_candidate_shape_ok &&
        !ud->dynamic_lora_active && !ud->force_reference_scalar;
    if (ith == 0 && callback_work_ctx) {
        auto& profile = callback_work_ctx->qwen36_profile;
        profile.gemv_custom_quant_input_type_hist[MatmulQuantInputTypeHistIndex(ud->input_quant_type, quant_input != nullptr)]
            .fetch_add(1, std::memory_order_relaxed);
        if (!quant_input) {
            profile.gemv_custom_quant_input_null_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K && quant_input &&
            !q4k_candidate_shape_ok) {
            profile.gemv_custom_shape_reject_ops.fetch_add(1, std::memory_order_relaxed);
        }
        if (q4k_repacked_candidate) {
            profile.q4k_repacked_gemv_candidate_ops.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    if (k_start >= K) return;

    const size_t row_stride = weight_tensor->nb[1];  // Bytes per row

    if (weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K && quant_input &&
        (ud->dynamic_lora_active || ud->force_reference_scalar) && ith == 0) {
        if (InferenceWorkContext* work_ctx = callback_work_ctx) {
            if (q4k_repacked_candidate) {
                work_ctx->qwen36_profile.q4k_repacked_gemv_rejected_ops.fetch_add(1, std::memory_order_relaxed);
            }
            work_ctx->qwen36_profile.q4k_repacked_gemv_last_reject_reason.store(
                static_cast<int>(ud->dynamic_lora_active ? Q4KRepackedGemvRejectReason::DynamicLora
                                                         : Q4KRepackedGemvRejectReason::ReferenceForced),
                std::memory_order_relaxed);
        }
    }
    if (weight_type == GGML_TYPE_Q6_K) {
        const auto* q6_type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
        const bool q6_can_use_direct = q6k_decode_candidate && q6_type_traits_cpu && q6_type_traits_cpu->vec_dot;
        const char* q6_reject_reason = "none";
        if (!q6k_decode_candidate) {
            if (!q6k_effective_decode_gemv) {
                q6_reject_reason = "not_decode";
            } else if (ud->dynamic_lora_active) {
                q6_reject_reason = "dynamic_lora";
            } else if (ud->force_reference_scalar) {
                q6_reject_reason = "reference_forced";
            } else if (!quant_input) {
                q6_reject_reason = "quant_input_null";
            } else if (ud->input_quant_type != GGML_TYPE_Q8_K) {
                q6_reject_reason = "unsupported_quant_input";
            } else {
                q6_reject_reason = "unsupported_shape";
            }
        } else if (!q6_type_traits_cpu || !q6_type_traits_cpu->vec_dot) {
            q6_reject_reason = "kernel_unavailable";
        }
        if (q6_can_use_direct) {
            // Phase 0: the nrc=2 row-pair shape is only correct where ggml
            // implements it (ARM i8mm). kernels::KQuantVecDotRowPairSupported() is
            // the single source of truth; do NOT gate on traits->nrows (some x86
            // builds advertise nrows==2 without honoring nrc==2 -> corrupted logits).
            const bool q6_can_use_rowpair = densecore::kernels::KQuantVecDotRowPairSupported() &&
                                            q6_type_traits_cpu->nrows >= 2 && row_stride > 0 &&
                                            (N % ggml_blck_size(weight_type)) == 0;
            const bool lfm2_argmax_candidate = ud->lfm2_decode_lm_head && callback_work_ctx;
            if (lfm2_argmax_candidate && ith == 0) {
                auto& profile = callback_work_ctx->qwen36_profile;
                profile.lfm2_greedy_lm_head_argmax_candidate_ops.fetch_add(1, std::memory_order_relaxed);
                if (!callback_work_ctx->lfm2_greedy_lm_head_argmax_allowed) {
                    profile.lfm2_greedy_lm_head_argmax_rejected_ops.fetch_add(1, std::memory_order_relaxed);
                    profile.lfm2_greedy_lm_head_argmax_last_reject_reason.store(
                        static_cast<int>(callback_work_ctx->lfm2_greedy_lm_head_argmax_reject_reason),
                        std::memory_order_relaxed);
                    callback_work_ctx->lfm2_greedy_lm_head_argmax_generation = 0;
                    callback_work_ctx->lfm2_greedy_lm_head_argmax_token = -1;
                    callback_work_ctx->lfm2_greedy_lm_head_argmax_value = -std::numeric_limits<float>::infinity();
                    callback_work_ctx->lfm2_greedy_lm_head_argmax_vocab_size = 0;
                    callback_work_ctx->lfm2_greedy_lm_head_sparse_logits_ptr = nullptr;
                    callback_work_ctx->lfm2_greedy_lm_head_sparse_logits_vocab_size = 0;
                    callback_work_ctx->lfm2_greedy_lm_head_sparse_logits_token = -1;
                }
            }
            if (lfm2_argmax_candidate && callback_work_ctx->lfm2_greedy_lm_head_argmax_allowed) {
                const uintptr_t argmax_key =
                    reinterpret_cast<uintptr_t>(dst->data) ^
                    (reinterpret_cast<uintptr_t>(src->data) << 1) ^
                    (reinterpret_cast<uintptr_t>(weight_data) << 7) ^
                    (static_cast<uintptr_t>(callback_work_ctx->execution_generation) * 0x9e3779b97f4a7c15ull);
                {
                    std::lock_guard<std::mutex> lock(ud->lfm2_argmax_mutex);
                    if (ud->lfm2_argmax_key != argmax_key || ud->lfm2_argmax_nth != nth) {
                        ud->lfm2_argmax_key = argmax_key;
                        ud->lfm2_argmax_done = 0;
                        ud->lfm2_argmax_nth = nth;
                        ud->lfm2_argmax_best_token = -1;
                        ud->lfm2_argmax_best_value = -std::numeric_limits<float>::infinity();
                        ud->lfm2_argmax_begin = std::chrono::steady_clock::now();
                    }
                    const uintptr_t q6_cache_weight = reinterpret_cast<uintptr_t>(weight_data);
                    if (ud->lfm2_argmax_q6k_cache_weight != q6_cache_weight ||
                        ud->lfm2_argmax_q6k_cache_nth != nth ||
                        ud->lfm2_argmax_q6k_cache_k != K ||
                        ud->lfm2_argmax_q6k_cache_n != N) {
                        ud->lfm2_argmax_q6k_cache_weight = q6_cache_weight;
                        ud->lfm2_argmax_q6k_cache_nth = nth;
                        ud->lfm2_argmax_q6k_cache_k = K;
                        ud->lfm2_argmax_q6k_cache_n = N;
                        ud->lfm2_argmax_q6k_slice_cache.assign(static_cast<size_t>(std::max(1, nth)), {});
                    }
                }

                int local_best_token = -1;
                float local_best_value = -std::numeric_limits<float>::infinity();
                const auto& repeated_tokens = callback_work_ctx->lfm2_greedy_lm_head_argmax_repeated_tokens;
                const float repetition_penalty = callback_work_ctx->lfm2_greedy_lm_head_argmax_repetition_penalty;
                const bool repetition_penalty_active = repetition_penalty != 1.0f && !repeated_tokens.empty();
                size_t repeat_pos =
                    repetition_penalty_active
                        ? static_cast<size_t>(std::lower_bound(repeated_tokens.begin(), repeated_tokens.end(), k_start) -
                                              repeated_tokens.begin())
                        : repeated_tokens.size();
                int k = k_start;
                const bool q6_can_use_repacked_argmax =
                    row_stride > 0 && (N % ggml_blck_size(weight_type)) == 0 && (k_start % 8) == 0 &&
                    ((k_end - k_start) % 8) == 0;
                bool used_repacked_argmax = false;
                if (q6_can_use_repacked_argmax) {
                    const int row_count = k_end - k_start;
                    alignas(64) thread_local std::vector<float> repacked_argmax_tile;
                    repacked_argmax_tile.resize(static_cast<size_t>(row_count));
                    const void* row_ptr =
                        reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k_start) * row_stride;
                    std::shared_ptr<void>* slice_cache = nullptr;
                    if (ith >= 0 && ith < static_cast<int>(ud->lfm2_argmax_q6k_slice_cache.size())) {
                        slice_cache = &ud->lfm2_argmax_q6k_slice_cache[static_cast<size_t>(ith)];
                    }
                    const bool projected = densecore::RunQ6KRepackedMoEProjectionCached(
                        &densecore::GetCpuBackend(), row_ptr, static_cast<const uint8_t*>(quant_input),
                        ggml_row_size(GGML_TYPE_Q8_K, N), repacked_argmax_tile.data(), 1, row_count, N,
                        /*numa_node=*/0, /*allow_parallel=*/false, slice_cache);
                    if (projected) {
                        used_repacked_argmax = true;
                        for (int offset = 0; offset < row_count; ++offset) {
                            const int token = k_start + offset;
                            float v = repacked_argmax_tile[static_cast<size_t>(offset)];
                            v = ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(
                                v, token, repeated_tokens, repeat_pos, repetition_penalty, repetition_penalty_active);
                            if (IsBetterArgmaxCandidate(token, v, local_best_token, local_best_value)) {
                                local_best_token = token;
                                local_best_value = v;
                            }
                        }
                        k = k_end;
                    }
                }
                if (q6_can_use_rowpair && (k & 1)) {
                    const void* row_ptr =
                        reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    float v = 0.0f;
                    q6_type_traits_cpu->vec_dot(N, &v, 0, row_ptr, 0, quant_input, 0, 1);
                    v = ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(v, k, repeated_tokens, repeat_pos,
                                                                     repetition_penalty, repetition_penalty_active);
                    if (IsBetterArgmaxCandidate(k, v, local_best_token, local_best_value)) {
                        local_best_token = k;
                        local_best_value = v;
                    }
                    ++k;
                }
                if (q6_can_use_rowpair) {
                    for (; k + 1 < k_end; k += 2) {
                        const void* row_ptr =
                            reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        q6_type_traits_cpu->vec_dot(N, sums, 2, row_ptr, row_stride, quant_input, 0, 2);
                        sums[0] = ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(
                            sums[0], k, repeated_tokens, repeat_pos, repetition_penalty, repetition_penalty_active);
                        if (IsBetterArgmaxCandidate(k, sums[0], local_best_token, local_best_value)) {
                            local_best_token = k;
                            local_best_value = sums[0];
                        }
                        sums[1] = ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(
                            sums[1], k + 1, repeated_tokens, repeat_pos, repetition_penalty,
                            repetition_penalty_active);
                        if (IsBetterArgmaxCandidate(k + 1, sums[1], local_best_token, local_best_value)) {
                            local_best_token = k + 1;
                            local_best_value = sums[1];
                        }
                    }
                }
                for (; k < k_end; ++k) {
                    const void* row_ptr =
                        reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    float v = 0.0f;
                    q6_type_traits_cpu->vec_dot(N, &v, 0, row_ptr, 0, quant_input, 0, 1);
                    v = ApplyLFM2GreedyArgmaxRepetitionPenaltyIfDue(v, k, repeated_tokens, repeat_pos,
                                                                     repetition_penalty, repetition_penalty_active);
                    if (IsBetterArgmaxCandidate(k, v, local_best_token, local_best_value)) {
                        local_best_token = k;
                        local_best_value = v;
                    }
                }

                bool last_thread = false;
                int best_token = -1;
                float best_value = -std::numeric_limits<float>::infinity();
                std::chrono::steady_clock::time_point begin;
                {
                    std::lock_guard<std::mutex> lock(ud->lfm2_argmax_mutex);
                    if (ud->lfm2_argmax_key != argmax_key) {
                        callback_work_ctx->qwen36_profile.lfm2_greedy_lm_head_argmax_rejected_ops.fetch_add(
                            1, std::memory_order_relaxed);
                        callback_work_ctx->qwen36_profile.lfm2_greedy_lm_head_argmax_last_reject_reason.store(
                            static_cast<int>(LFM2GreedyLMHeadArgmaxRejectReason::Sync), std::memory_order_relaxed);
                    } else {
                        if (IsBetterArgmaxCandidate(local_best_token, local_best_value, ud->lfm2_argmax_best_token,
                                                    ud->lfm2_argmax_best_value)) {
                            ud->lfm2_argmax_best_token = local_best_token;
                            ud->lfm2_argmax_best_value = local_best_value;
                        }
                        ud->lfm2_argmax_done += 1;
                        if (ud->lfm2_argmax_done == nth) {
                            last_thread = true;
                            best_token = ud->lfm2_argmax_best_token;
                            best_value = ud->lfm2_argmax_best_value;
                            begin = ud->lfm2_argmax_begin;
                        }
                    }
                }
                if (last_thread) {
                    WriteInferenceWorkContextLFM2GreedyLMHeadSparseLogits(callback_work_ctx, output, K, best_token,
                                                                          best_value);
                    RecordInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(
                        callback_work_ctx, callback_work_ctx->execution_generation, best_token, best_value, K);
                    auto& profile = callback_work_ctx->qwen36_profile;
                    profile.lfm2_decode_lm_head_custom_gemv_used_ops.fetch_add(1, std::memory_order_relaxed);
                    profile.lfm2_greedy_lm_head_argmax_used_ops.fetch_add(1, std::memory_order_relaxed);
                    profile.lfm2_greedy_lm_head_argmax_last_reject_reason.store(
                        static_cast<int>(LFM2GreedyLMHeadArgmaxRejectReason::None), std::memory_order_relaxed);
                    uint64_t wall_ns = 0;
                    if (begin != std::chrono::steady_clock::time_point{}) {
                        wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            std::chrono::steady_clock::now() - begin)
                                                            .count());
                        profile.lfm2_decode_lm_head_custom_gemv_ns.fetch_add(wall_ns, std::memory_order_relaxed);
                        profile.lfm2_greedy_lm_head_argmax_ns.fetch_add(wall_ns, std::memory_order_relaxed);
                    }
                    RecordQ6KGemvDecision(callback_work_ctx, /*candidate=*/true, /*used=*/true, "none", weight_name,
                                          1, K, N, wall_ns, "decode", "decode_shape",
                                          MatmulPhaseName(callback_phase),
                                          used_repacked_argmax
                                              ? "lfm2_q6k_argmax_repacked"
                                              : (q6_can_use_rowpair ? "lfm2_q6k_argmax_rowpair"
                                                                    : "lfm2_q6k_argmax"));
                    q6k_decision_recorded = true;
                }
                return;
            }
            int k = k_start;
            if (q6_can_use_rowpair && (k & 1)) {
                const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                q6_type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
                ++k;
            }
            if (q6_can_use_rowpair) {
                for (; k + 1 < k_end; k += 2) {
                    const void* row_ptr =
                        reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    q6_type_traits_cpu->vec_dot(N, sums, 2, row_ptr, row_stride, quant_input, 0, 2);
                    output[k] = sums[0];
                    output[k + 1] = sums[1];
                }
            }
            for (; k < k_end; ++k) {
                const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                q6_type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
            }
            maybe_log_gemv_timing(q6_can_use_rowpair ? "q6k_direct_vecdot_rowpair" : "q6k_direct_vecdot");
            record_gemv_dispatch_census(q6_can_use_rowpair ? "q6k_direct_vecdot_rowpair" : "q6k_direct_vecdot");
            return;
        }
        if (ith == 0 && callback_work_ctx) {
            const char* callback_phase_name = MatmulPhaseName(callback_phase);
            const char* graph_phase_name = q6k_candidate_shape_ok ? "decode_shape" : callback_phase_name;
            const char* effective_phase_name = q6k_effective_decode_gemv ? "decode" : callback_phase_name;
            RecordQ6KGemvDecision(callback_work_ctx, q6k_decode_candidate, /*used=*/false, q6_reject_reason,
                                  weight_name, 1, K, N, 0, effective_phase_name, graph_phase_name,
                                  callback_phase_name, nullptr);
            q6k_decision_recorded = true;
        }
    }
    if (weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K && quant_input &&
        !ud->dynamic_lora_active && !ud->force_reference_scalar) {
        Q4KRepackedGemvRejectReason reject = Q4KRepackedGemvRejectReason::None;
        InferenceWorkContext* dispatch_work_ctx = callback_work_ctx;
        const BatchSpec* batch = (dispatch_work_ctx && dispatch_work_ctx->batch) ? dispatch_work_ctx->batch : callback_batch;
        const auto& fast_config = ResolveFastPathRuntimeConfig(batch);
        const bool real_repacked_enabled = Q4KRepackedGemvEnabled(fast_config, &reject);
        if (dispatch_work_ctx) {
            const int disable_reason =
                dispatch_work_ctx->qwen36_profile.q4k_repacked_gemv_primary_disable_reason.load(
                    std::memory_order_relaxed);
            if (disable_reason != 0 && fast_config.q4k_repacked_gemv == densecore::env::RuntimeToggleMode::Auto) {
                reject = static_cast<Q4KRepackedGemvRejectReason>(disable_reason);
            }
        }
        const bool phase_allows_repacked =
            dispatch_work_ctx && batch &&
            (callback_phase == InferenceExecutionPhase::Decode ||
             (fast_config.q4k_repacked_gemv_allow_prefill &&
              callback_phase == InferenceExecutionPhase::Prefill));
        if (real_repacked_enabled && !phase_allows_repacked) {
            reject = Q4KRepackedGemvRejectReason::NotDecode;
        }
        if (real_repacked_enabled && phase_allows_repacked && reject == Q4KRepackedGemvRejectReason::None) {
            const bool shape_ok = q4k_candidate_shape_ok;
            if (shape_ok) {
                densecore::kernels::Q4KRepackedGemvCacheLookup cache_lookup;
                auto packed = densecore::kernels::GetOrCreateQ4KRepackedGemvWeight(weight_data, K, N, &cache_lookup);
                RecordQ4KRepackedGemvCacheLookup(dispatch_work_ctx, cache_lookup);
                Q4KRepackedGemvShouldAutoDisableForLookup(dispatch_work_ctx, fast_config, cache_lookup, &reject);
                const int tile_count = K / 8;
                const int tile_start = (tile_count * ith) / nth;
                const int tile_end = (tile_count * (ith + 1)) / nth;
                const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, N);
                bool probe_ok = true;
                if (reject == Q4KRepackedGemvRejectReason::None && packed && fast_config.q4k_repacked_gemv_probe) {
                    const auto probe_begin = std::chrono::steady_clock::now();
                    probe_ok = Q4KRepackedGemvProbePassed(packed, weight_data, quant_input, K, N);
                    if (dispatch_work_ctx) {
                        AddQwen36ProfileNs(
                            dispatch_work_ctx->qwen36_profile.q4k_repacked_gemv_probe_ns,
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - probe_begin)
                                    .count()));
                    }
                }
                if (reject == Q4KRepackedGemvRejectReason::None && packed && probe_ok &&
                    densecore::kernels::RunQ4KRepackedGemv(packed, static_cast<const uint8_t*>(quant_input),
                                                           q8_row_bytes, output, K, tile_start, tile_end)) {
                    if (ith == 0) {
                        InferenceWorkContext* work_ctx = callback_work_ctx;
                        if (work_ctx) {
                            work_ctx->qwen36_profile.q4k_repacked_gemv_used.store(1, std::memory_order_relaxed);
                            work_ctx->qwen36_profile.q4k_repacked_gemv_used_ops.fetch_add(
                                1, std::memory_order_relaxed);
                            work_ctx->qwen36_profile.q4k_repacked_gemv_last_reject_reason.store(
                                static_cast<int>(Q4KRepackedGemvRejectReason::None), std::memory_order_relaxed);
                        }
                    }
                    maybe_log_gemv_timing("q4k_repacked_gemv");
                    record_gemv_dispatch_census("q4k_repacked_gemv");
                    return;
                }
                if (reject == Q4KRepackedGemvRejectReason::None) {
                    if (cache_lookup.working_set_exceeds_cache) {
                        reject = Q4KRepackedGemvRejectReason::WorkingSetExceedsCache;
                    } else if (cache_lookup.cache_limit_too_small) {
                        reject = Q4KRepackedGemvRejectReason::CacheLimitTooSmall;
                    } else {
                        reject = packed && !probe_ok ? Q4KRepackedGemvRejectReason::ProbeFailed
                                                     : Q4KRepackedGemvRejectReason::Cache;
                    }
                }
            } else {
                reject = Q4KRepackedGemvRejectReason::Shape;
            }
        }
        if (ith == 0) {
            InferenceWorkContext* work_ctx = callback_work_ctx;
            if (work_ctx) {
                if (reject != Q4KRepackedGemvRejectReason::None) {
                    if (q4k_repacked_candidate) {
                        work_ctx->qwen36_profile.q4k_repacked_gemv_rejected_ops.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
                work_ctx->qwen36_profile.q4k_repacked_gemv_last_reject_reason.store(
                    static_cast<int>(reject), std::memory_order_relaxed);
            }
        }
    }

    // ==========================================================================
    // CASE A: FP32 weights - use optimized simd::GemvParallel
    // ==========================================================================
    if (weight_type == GGML_TYPE_F32) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_F32_CALLBACK", false, false, false);
        }
        const float* weight = reinterpret_cast<const float*>(weight_data);
        densecore::simd::GemvParallel(output, x_f32, weight, N, K, ith, nth);
        maybe_log_gemv_timing("f32");
        record_gemv_dispatch_census("generic_gemv");
        return;
    }

    // ==========================================================================
    // CASE B: Quantized weights with pre-quantized input - use native vec_dot
    // ==========================================================================
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
    const void* sample_row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k_start) * row_stride;
    const bool allow_native_q4k_vecdot =
        ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, weight_name, sample_row_ptr, quant_input,
                                             x_f32, N);

    if (!ud->force_reference_scalar && quant_input && type_traits_cpu && type_traits_cpu->vec_dot &&
        allow_native_q4k_vecdot) {
        if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
            LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_NATIVE_VECDOT_CALLBACK", false, true,
                                    false);
        }
        const int q8_repacked_min_rows =
            IsGemma4SharedDenseFfnWeightName(weight_name)
                ? ParsePositiveEnvInt("DENSECORE_GEMMA4_SHARED_Q8_REPACKED_MIN_ROWS",
                                      kQ8RepackedGemvMinOutputRows)
                : kQ8RepackedGemvMinOutputRows;
        if (!ud->disable_q8_repacked_gemv && weight_type == GGML_TYPE_Q8_0 && ud->input_quant_type == GGML_TYPE_Q8_0 &&
            (K >= q8_repacked_min_rows || ud->force_q8_repacked_gemv) &&
            (N % QK8_0) == 0 && (K % 4) == 0) {
            auto packed = GetOrCreateQ8RepackedGemvWeight(weight_data, K, N, ud->force_q8_repacked_gemv);
            if (packed && packed->blocks_per_row > 0) {
                const size_t block_bytes = 4 * sizeof(ggml_fp16_t) + QK8_0 * 4;
                int k = k_start;
                for (; k < k_end && (k & 3); ++k) {
                    const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
                }
                const int packed_end = k_end & ~3;
                if (k < packed_end) {
                    const size_t packed_offset =
                        static_cast<size_t>(k / 4) * static_cast<size_t>(packed->blocks_per_row) * block_bytes;
                    ggml_gemv_q8_0_4x8_q8_0(N, output + k, 0, packed->data.data() + packed_offset, quant_input, 1,
                                            packed_end - k);
                    k = packed_end;
                }
                for (; k < k_end; ++k) {
                    const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                    type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
                }
                maybe_log_gemv_timing("q8_0_repacked_4x8");
                record_gemv_dispatch_census("q8_0_repacked_4x8");
                return;
            }
        }
        const bool can_use_rowpair =
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
            weight_type == GGML_TYPE_Q4_K && ud->input_quant_type == GGML_TYPE_Q8_K &&
            (N % ggml_blck_size(weight_type)) == 0;
#else
            false;
#endif
        int k = k_start;
        if (can_use_rowpair && (k & 1)) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
            ++k;
        }
        if (can_use_rowpair) {
            for (; k + 1 < k_end; k += 2) {
                const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
                float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                type_traits_cpu->vec_dot(N, sums, 2, row_ptr, row_stride, quant_input, 0, 2);
                output[k] = sums[0];
                output[k + 1] = sums[1];
            }
        }
        for (; k < k_end; ++k) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + static_cast<size_t>(k) * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
        }
        maybe_log_gemv_timing(can_use_rowpair ? "quant_vecdot_rowpair" : "quant_vecdot");
        record_gemv_dispatch_census("generic_gemv");
        return;
    }

    // ==========================================================================
    // CASE C: Fallback - dequantize weights (no pre-quantized input available)
    // ==========================================================================
    thread_local std::vector<float> dequant_buffer_tls;
    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float || N > static_cast<int>(kMaxDequantBufferSize)) {
        for (int k = k_start; k < k_end; k++) output[k] = 0.0f;
        record_gemv_dispatch_census("other");
        return;
    }
    if (IsHybridSSMQkvWeightName(weight_name) && IsDebugMatmulDispatchEnabled() && ith == 0) {
        LogHybridSSMQkvDispatch(weight_name, weight_type, 1, K, N, "GEMV_DEQUANT_REFERENCE_CALLBACK", false, false,
                                false);
    }
    dequant_buffer_tls.resize(static_cast<size_t>(N));
    float* dequant_buffer = dequant_buffer_tls.data();

    for (int k = k_start; k < k_end; k++) {
        const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
        type_traits->to_float(row_ptr, dequant_buffer, N);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x_f32[i] * dequant_buffer[i];
        }
        output[k] = sum;
    }
    maybe_log_gemv_timing("dequant_reference");
    record_gemv_dispatch_census("generic_gemv");
}
