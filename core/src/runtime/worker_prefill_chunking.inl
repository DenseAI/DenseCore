// Model-specific prefill chunking and graph-context downgrade policy.
int RequestPromptTokenCountForChunking(const Request* req) {
    if (!req) {
        return 0;
    }
    if (!req->original_prompt_tokens_for_cache.empty()) {
        return static_cast<int>(req->original_prompt_tokens_for_cache.size());
    }
    if (!req->prompt_tokens_for_cache.empty()) {
        return static_cast<int>(req->prompt_tokens_for_cache.size());
    }
    if (req->prompt_token_count > 0) {
        return req->prompt_token_count;
    }
    return static_cast<int>(req->tokens.size());
}

int ResolveQwen36PrefillChunkTokensImpl(const TransformerModel* model, const Request* req) {
    if (!model || !req) {
        return -1;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::QWEN35 && descriptor.variant != ModelVariant::QWEN36) {
        return -1;
    }
    const bool qwen35_dense = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts <= 0;
    const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
    const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
    const int hybrid_ssm_chunk_tokens =
#if defined(__aarch64__) || defined(_M_ARM64)
        64;
#else
        320;
#endif
    const int base_chunk_tokens =
        qwen35_dense ? 768 : ((qwen35_moe || qwen_hybrid_ssm) ? hybrid_ssm_chunk_tokens : 192);
    const int base_auto_min_tokens = qwen35_dense ? 1024 : ((qwen35_moe || qwen_hybrid_ssm) ? 1280 : 1536);
    const char* chunk_env = "DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS";
    const char* default_env = "DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS";
    const char* auto_min_env = "DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS";
    const int configured_default_chunk_tokens = densecore::env::ParsePositiveEnvInt(default_env, base_chunk_tokens);
    const int default_chunk_tokens = std::min(configured_default_chunk_tokens, base_chunk_tokens);

    const int explicit_tokens = densecore::env::ParsePositiveEnvInt(chunk_env, 0);
    if (explicit_tokens > 0) {
        return explicit_tokens;
    }

    const char* env_value = std::getenv(chunk_env);
    if (env_value && env_value[0] != '\0') {
        const std::string lowered = densecore::env::AsciiLowerCopy(env_value);
        if (lowered == "off" || lowered == "false" || lowered == "no") {
            const int prompt_tokens = RequestPromptTokenCountForChunking(req);
            const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, base_auto_min_tokens);
            if (qwen_hybrid_ssm && prompt_tokens >= auto_min_tokens) {
                return default_chunk_tokens;
            }
            return -1;
        }
        if (lowered == "on" || lowered == "true" || lowered == "yes" || lowered == "force") {
            return default_chunk_tokens;
        }
    }

    const int prompt_tokens = RequestPromptTokenCountForChunking(req);
    if (prompt_tokens <= 0) {
        return default_chunk_tokens;
    }
    const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, base_auto_min_tokens);
    return prompt_tokens >= auto_min_tokens ? default_chunk_tokens : -1;
}

int ResolveGemma4PrefillChunkTokensImpl(const TransformerModel* model, const Request* req) {
    if (!model || !req) {
        return -1;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::GEMMA4 || model->hparams.n_experts <= 0) {
        return -1;
    }
    const char* chunk_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS";
    const char* default_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_DEFAULT_TOKENS";
    const char* auto_min_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS";
    const int explicit_tokens = densecore::env::ParsePositiveEnvInt(chunk_env, 0);
    if (explicit_tokens > 0) {
        return explicit_tokens;
    }

    const auto resolve_manual_default = [&]() { return densecore::env::ParsePositiveEnvInt(default_env, 128); };
    const char* env_value = std::getenv(chunk_env);
    bool explicit_auto = false;
    if (env_value && env_value[0] != '\0') {
        const std::string lowered = densecore::env::AsciiLowerCopy(env_value);
        if (lowered == "off" || lowered == "false" || lowered == "no") {
            return -1;
        }
        if (lowered == "on" || lowered == "true" || lowered == "yes" || lowered == "force") {
            return resolve_manual_default();
        }
        explicit_auto = (lowered == "0" || lowered == "auto");
    }
    if (!explicit_auto) {
        const densecore::env::RuntimeToggleMode mode =
            densecore::env::ParseRuntimeToggleMode(chunk_env, densecore::env::RuntimeToggleMode::Auto);
        if (mode == densecore::env::RuntimeToggleMode::Off) {
            return -1;
        }
        if (mode == densecore::env::RuntimeToggleMode::On) {
            return resolve_manual_default();
        }
    }

    const int prompt_tokens = RequestPromptTokenCountForChunking(req);
    if (prompt_tokens <= 0) {
        return resolve_manual_default();
    }
    const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, 1024);
    if (prompt_tokens < auto_min_tokens) {
        return -1;
    }
    if (densecore::env::ParsePositiveEnvInt(default_env, 0) > 0) {
        return resolve_manual_default();
    }

    // Gemma4 MoE prefill uses the transient ggml graph allocator, so the
    // default path can run the whole prompt without the KV history gathers
    // introduced by chunking.
    return -1;
}

int ResolveModelPrefillChunkTokens(const TransformerModel* model, const Request* req) {
    const int qwen36_tokens = ResolveQwen36PrefillChunkTokensImpl(model, req);
    if (qwen36_tokens != -1) {
        return qwen36_tokens;
    }
    return ResolveGemma4PrefillChunkTokensImpl(model, req);
}

size_t EstimatePrefillGraphContextReservationBytes(const TransformerModel* model, size_t prompt_tokens,
                                                   size_t chunk_tokens) {
    const auto estimate =
        EngineState::EstimateGraphContextSize(model, prompt_tokens, /*num_seqs_hint=*/1, chunk_tokens);
    size_t bytes = estimate.total_bytes;
    if (bytes > 0 && model &&
        (model->arch_flags.is_hybrid_ssm || model->arch_flags.is_gemma4 || model->hparams.n_experts > 0)) {
        constexpr size_t MB = 1024ULL * 1024ULL;
        const size_t object_pool_margin =
            std::max<size_t>(64ULL * MB, std::max(bytes / 32, estimate.long_context_safety_pad_bytes / 4));
        bytes = AlignUpBytes(bytes + object_pool_margin, 64ULL * MB);
    }
    return bytes;
}

size_t GraphContextSafetyMarginBytes() {
    // The graph-size estimate already includes long-context/object safety pads.
    // This margin is only admission headroom against live runtime pressure after
    // weights, KV, and repacked caches are resident; a fixed 512 MiB guard was
    // conservative enough to reject Qwen3.5 35B Q5 long QA even though the
    // estimated graph pool itself fit in available memory.
    return ParseSizeEnvMb("DENSECORE_GRAPH_CTX_SAFETY_MARGIN_MB", /*default_mb=*/256, /*min_mb=*/0,
                          /*max_mb=*/65536) *
           1024ULL * 1024ULL;
}

size_t GraphContextPrefillAdmissionBaseMarginBytes(const TransformerModel* model, size_t active_prefill_tokens,
                                                   size_t available_bytes) {
    const size_t configured_margin_bytes = GraphContextSafetyMarginBytes();
#if defined(__x86_64__) || defined(_M_X64)
    if (model && active_prefill_tokens > 0 && available_bytes > 0 && model->arch_flags.is_hybrid_ssm &&
        ModelHasMoEGraphLayers(model)) {
        const auto descriptor = densecore::models::DescribeModel(model);
        if (descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36) {
            constexpr size_t MB = 1024ULL * 1024ULL;
            const size_t dynamic_margin_bytes = std::max<size_t>(
                std::min<size_t>(32ULL * MB, configured_margin_bytes), available_bytes / 128);
            return std::min(configured_margin_bytes, dynamic_margin_bytes);
        }
    }
#else
    (void)model;
    (void)active_prefill_tokens;
    (void)available_bytes;
#endif
    return configured_margin_bytes;
}

size_t EstimateQwenMoEQ4KPrefillFutureReserveBytes(const TransformerModel* model, size_t active_prefill_tokens = 0,
                                                   size_t available_bytes = 0) {
#if defined(__aarch64__) || defined(_M_ARM64)
    (void)model;
    (void)active_prefill_tokens;
    (void)available_bytes;
    return 0;
#else
    if (!model || !densecore::kernels::Q4KRealPackedGemvKernelAvailable()) {
        return 0;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    constexpr size_t MB = 1024ULL * 1024ULL;
    const auto& hp = model->hparams;
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    const bool qwen_moe = (descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36) &&
                          (ModelHasMoEGraphLayers(model) || contract.has_moe);
    if (!qwen_moe) {
        return 0;
    }

    size_t contract_moe_layers = 0;
    int contract_num_experts = 0;
    int contract_top_k = 0;
    for (const auto& layer : contract.layers) {
        if (!layer.has_moe) {
            continue;
        }
        ++contract_moe_layers;
        contract_num_experts = std::max(contract_num_experts, layer.moe_num_experts);
        contract_top_k = std::max(contract_top_k, layer.moe_top_k);
    }
    const size_t layer_count =
        contract_moe_layers > 0
            ? contract_moe_layers
            : static_cast<size_t>(
                  std::max<int>(1, static_cast<int>(hp.n_layer) - std::max(0, model->moe_first_k_dense_replace)));
    uint32_t experts_used = hp.n_experts_used;
    uint32_t total_experts = hp.n_experts;
    if (experts_used == 0 && contract_top_k > 0) {
        experts_used = static_cast<uint32_t>(contract_top_k);
    }
    if (total_experts == 0 && contract_num_experts > 0) {
        total_experts = static_cast<uint32_t>(contract_num_experts);
    }
    const size_t top_k = static_cast<size_t>(std::max<uint32_t>(1, experts_used));
    const bool expert_count_known = total_experts > experts_used || contract_num_experts > contract_top_k;
    const size_t experts_total =
        expert_count_known ? static_cast<size_t>(std::max<uint32_t>(1, std::max(total_experts, experts_used))) : top_k;
    size_t routed_experts_per_layer = top_k;
    if (active_prefill_tokens > 0) {
        const size_t cache_warmup_tokens = std::min<size_t>(active_prefill_tokens, 8);
        if (cache_warmup_tokens > std::numeric_limits<size_t>::max() / top_k) {
            routed_experts_per_layer = expert_count_known ? experts_total : std::numeric_limits<size_t>::max();
        } else if (expert_count_known) {
            routed_experts_per_layer = std::min(experts_total, std::max(top_k, cache_warmup_tokens * top_k));
        } else {
            routed_experts_per_layer = std::max(top_k, cache_warmup_tokens * top_k);
        }
    } else if (total_experts > 0 && experts_used > 0) {
        routed_experts_per_layer = std::min(experts_total, top_k);
    }
    size_t inferred_ffn_rows = 0;
    size_t inferred_hidden_cols = static_cast<size_t>(hp.n_embd);
    const size_t model_hidden = static_cast<size_t>(hp.n_embd);
    const size_t model_experts = static_cast<size_t>(std::max<uint32_t>(total_experts, 0));
    auto inspect_moe_tensor_shape = [&](const ggml_tensor* tensor) {
        if (!tensor) {
            return;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] <= 1) {
                continue;
            }
            const size_t value = static_cast<size_t>(tensor->ne[dim]);
            if (model_hidden > 0 && value == model_hidden) {
                inferred_hidden_cols = std::max(inferred_hidden_cols, value);
                continue;
            }
            if (model_experts > 0 && value == model_experts) {
                continue;
            }
            if (value <= top_k) {
                continue;
            }
            inferred_ffn_rows = std::max(inferred_ffn_rows, value);
        }
    };
    for (const TransformerLayer& layer : model->layers) {
        inspect_moe_tensor_shape(layer.Get("ffn_down_exps.weight"));
        inspect_moe_tensor_shape(layer.Get("ffn_down_exps"));
        inspect_moe_tensor_shape(layer.Get("experts.down_proj.weight"));
        inspect_moe_tensor_shape(layer.Get("ffn_gate_up_exps.weight"));
        inspect_moe_tensor_shape(layer.Get("ffn_gate_up_exps"));
        inspect_moe_tensor_shape(layer.Get("experts.gate_up_proj.weight"));
    }
    const size_t ffn_rows =
        AlignUpBytes(std::max<size_t>(8, std::max(static_cast<size_t>(hp.n_ff), inferred_ffn_rows)), 8);
    const size_t hidden_cols =
        AlignUpBytes(std::max<size_t>(densecore::kernels::kQ4KSuperBlock,
                                      std::max(static_cast<size_t>(hp.n_embd), inferred_hidden_cols)),
                     densecore::kernels::kQ4KSuperBlock);
    const size_t packed_blocks =
        (ffn_rows / 8ULL) * (hidden_cols / static_cast<size_t>(densecore::kernels::kQ4KSuperBlock));
    const size_t packed_matrix_bytes = packed_blocks * sizeof(densecore::kernels::Q4KRepackedGemvBlock);
    if (packed_matrix_bytes == 0) {
        return 0;
    }

    auto saturating_mul = [](size_t a, size_t b) -> size_t {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            return std::numeric_limits<size_t>::max();
        }
        return a * b;
    };
    auto saturating_add = [](size_t a, size_t b) -> size_t {
        if (b > std::numeric_limits<size_t>::max() - a) {
            return std::numeric_limits<size_t>::max();
        }
        return a + b;
    };

    // Qwen MoE Q4_K prefill uses two repacked expert matrices per routed expert
    // (gate and up). Reserve for the current active prefill chunk, not for the
    // whole user prompt, so long-context requests scale with host headroom
    // instead of being rejected by a worst-case all-prompt cache assumption.
    size_t estimated_working_set = saturating_mul(layer_count, routed_experts_per_layer);
    estimated_working_set = saturating_mul(estimated_working_set, 2ULL);
    estimated_working_set = saturating_mul(estimated_working_set, packed_matrix_bytes);
    const size_t jitter = std::max(256ULL * MB, estimated_working_set / 4ULL);
    const size_t estimated_with_jitter =
        AlignUpBytes(saturating_add(estimated_working_set, jitter), 64ULL * MB);
    const auto stats = densecore::kernels::Q4KRepackedGemvCacheStatsSnapshot();
    const size_t resident_bytes = static_cast<size_t>(stats.resident_bytes);
    size_t reserve_bytes = estimated_with_jitter > resident_bytes ? estimated_with_jitter - resident_bytes : 0;
    if (available_bytes > 0) {
        size_t dynamic_headroom_cap = std::max<size_t>(64ULL * MB, available_bytes / 48);
        if (available_bytes > 128ULL * MB) {
            dynamic_headroom_cap = std::min<size_t>(dynamic_headroom_cap, available_bytes - 64ULL * MB);
        } else {
            dynamic_headroom_cap = std::min<size_t>(dynamic_headroom_cap, available_bytes / 2);
        }
        reserve_bytes = std::min(reserve_bytes, dynamic_headroom_cap);
        if (active_prefill_tokens > 1) {
            std::cerr << "[DenseCore] QwenMoEQ4KPrefillReserveEstimate"
                      << " hp_experts=" << hp.n_experts
                      << " hp_experts_used=" << hp.n_experts_used
                      << " contract_experts=" << contract_num_experts
                      << " contract_top_k=" << contract_top_k
                      << " expert_count_known=" << (expert_count_known ? 1 : 0)
                      << " top_k=" << top_k
                      << " routed_experts_per_layer=" << routed_experts_per_layer
                      << " layer_count=" << layer_count
                      << " inferred_ffn_rows=" << inferred_ffn_rows
                      << " inferred_hidden_cols=" << inferred_hidden_cols
                      << " packed_matrix_mb=" << (packed_matrix_bytes / MB)
                      << " estimated_mb=" << (estimated_with_jitter / MB)
                      << " resident_mb=" << (resident_bytes / MB)
                      << " cap_mb=" << (dynamic_headroom_cap / MB)
                      << " reserve_mb=" << (reserve_bytes / MB)
                      << " available_mb=" << (available_bytes / MB)
                      << std::endl;
        }
    }
    return reserve_bytes;
#endif
}

size_t GraphContextPrefillAdmissionMarginBytes(const TransformerModel* model, size_t active_prefill_tokens = 0,
                                               size_t available_bytes = 0) {
    return GraphContextPrefillAdmissionBaseMarginBytes(model, active_prefill_tokens, available_bytes) +
           EstimateQwenMoEQ4KPrefillFutureReserveBytes(model, active_prefill_tokens, available_bytes);
}

int SelectLargestQwenPrefillChunkThatFits(const TransformerModel* model, size_t prompt_tokens, size_t available_bytes,
                                          size_t safety_margin_bytes, int current_chunk_tokens) {
    if (!model || prompt_tokens <= 0 || available_bytes == 0 || current_chunk_tokens <= 1) {
        return current_chunk_tokens;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    const bool model_has_moe = ModelHasMoEGraphLayers(model);
    const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model_has_moe;
    const bool qwen36_moe = descriptor.variant == ModelVariant::QWEN36 && model_has_moe;
    const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
    if (!qwen35_moe && !qwen36_moe && !qwen_hybrid_ssm) {
        return current_chunk_tokens;
    }

    // Larger chunks reduce graph rebuilds but Qwen hybrid-SSM Q8 projections
    // become less efficient past 128 tokens on C4A: the graph-build savings are
    // outweighed by larger Q8 GEMM work and weaker quantized-activation reuse.
    // Keep automatic growth inside the measured fast range and let graph-context
    // downgrade handle memory pressure below it.
    constexpr int kCandidates[] = {128, 96, 64, 48, 32};
    for (int candidate : kCandidates) {
        if (candidate <= current_chunk_tokens || static_cast<size_t>(candidate) > prompt_tokens) {
            continue;
        }
        const size_t candidate_reservation_bytes =
            EstimatePrefillGraphContextReservationBytes(model, prompt_tokens, static_cast<size_t>(candidate));
        if (candidate_reservation_bytes + safety_margin_bytes <= available_bytes) {
            return candidate;
        }
    }
    return current_chunk_tokens;
}

bool IsGraphContextAutoDowngradeEnabled() {
    return ParseBoolEnvDefault("DENSECORE_GRAPH_CTX_AUTO_DOWNGRADE", true);
}

bool IsGraphContextFailClosedEnabled() {
    return ParseBoolEnvDefault("DENSECORE_GRAPH_CTX_FAIL_CLOSED", true);
}

int ApplyGraphContextPrefillChunkDowngrade(const TransformerModel* model, Request* req, int chunk_tokens) {
    if (!model || !req || !IsGraphContextAutoDowngradeEnabled()) {
        return chunk_tokens;
    }
    const size_t available_bytes = ReadAvailableMemoryBytesForRuntimePools();
    if (available_bytes == 0) {
        return chunk_tokens;
    }
    const size_t prompt_tokens = static_cast<size_t>(std::max(1, RequestPromptTokenCountForChunking(req)));
    int effective_chunk = chunk_tokens;
    const bool originally_unchunked = effective_chunk <= 1;
    const size_t active_prefill_tokens =
        originally_unchunked ? prompt_tokens : static_cast<size_t>(std::max(1, effective_chunk));
    const size_t safety_margin_bytes =
        GraphContextPrefillAdmissionMarginBytes(model, active_prefill_tokens, available_bytes);
    const size_t initial_reservation_bytes = EstimatePrefillGraphContextReservationBytes(
        model, prompt_tokens, originally_unchunked ? 0 : static_cast<size_t>(effective_chunk));
    req->graph_ctx_requested_mb = initial_reservation_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_available_mb = available_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_safety_margin_mb = safety_margin_bytes / (1024ULL * 1024ULL);
    if (initial_reservation_bytes + safety_margin_bytes <= available_bytes) {
        const int expanded_chunk =
            SelectLargestQwenPrefillChunkThatFits(model, prompt_tokens, available_bytes, safety_margin_bytes,
                                                  chunk_tokens);
        if (expanded_chunk != chunk_tokens) {
            const size_t expanded_reservation_bytes =
                EstimatePrefillGraphContextReservationBytes(model, prompt_tokens, static_cast<size_t>(expanded_chunk));
            req->graph_ctx_requested_mb = expanded_reservation_bytes / (1024ULL * 1024ULL);
            req->graph_ctx_downgraded_chunk_tokens = expanded_chunk;
            req->graph_ctx_fail_reason.clear();
            std::cerr << "[DenseCore] GraphCtxAutoUpgrade"
                      << " req=" << req->id << " original_chunk_tokens=" << chunk_tokens
                      << " expanded_chunk_tokens=" << expanded_chunk
                      << " requested_mb=" << (initial_reservation_bytes / (1024ULL * 1024ULL))
                      << " expanded_mb=" << (expanded_reservation_bytes / (1024ULL * 1024ULL))
                      << " available_mb=" << req->graph_ctx_available_mb
                      << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << std::endl;
            return expanded_chunk;
        }
        return chunk_tokens;
    }
    if (originally_unchunked) {
        const auto descriptor = densecore::models::DescribeModel(model);
        const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
        const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
        effective_chunk = (qwen35_moe || qwen_hybrid_ssm) ? 384 : 512;
        effective_chunk = std::max(1, std::min<int>(effective_chunk, static_cast<int>(prompt_tokens)));
    }
    constexpr int kPressureCandidates[] = {384, 320, 288, 256, 224, 192, 160, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1};
    for (int candidate : kPressureCandidates) {
        if (candidate > effective_chunk || static_cast<size_t>(candidate) > prompt_tokens) {
            continue;
        }
        const size_t candidate_reservation_bytes =
            EstimatePrefillGraphContextReservationBytes(model, prompt_tokens, static_cast<size_t>(candidate));
        const size_t candidate_safety_margin_bytes =
            GraphContextPrefillAdmissionMarginBytes(model, static_cast<size_t>(candidate), available_bytes);
        if (candidate_reservation_bytes + candidate_safety_margin_bytes <= available_bytes) {
            req->graph_ctx_downgraded_chunk_tokens = candidate;
            req->graph_ctx_requested_mb = candidate_reservation_bytes / (1024ULL * 1024ULL);
            req->graph_ctx_safety_margin_mb = candidate_safety_margin_bytes / (1024ULL * 1024ULL);
            req->graph_ctx_fail_reason.clear();
            std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
                      << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
                      << " downgraded_chunk_tokens=" << candidate
                      << " requested_mb=" << (initial_reservation_bytes / (1024ULL * 1024ULL))
                      << " downgraded_mb=" << (candidate_reservation_bytes / (1024ULL * 1024ULL))
                      << " available_mb=" << req->graph_ctx_available_mb
                      << " safety_margin_mb=" << (candidate_safety_margin_bytes / (1024ULL * 1024ULL))
                      << std::endl;
            return candidate;
        }
    }

    req->graph_ctx_downgraded_chunk_tokens = 1;
    req->graph_ctx_fail_reason = "prefill_graph_ctx_exceeds_available_after_min_chunk";
    std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
              << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
              << " downgraded_chunk_tokens=1"
              << " requested_mb=" << (initial_reservation_bytes / (1024ULL * 1024ULL)) << " downgraded_mb=unknown"
              << " available_mb=" << req->graph_ctx_available_mb
              << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << " reason=min_chunk_still_too_large"
              << std::endl;
    return 1;
}
