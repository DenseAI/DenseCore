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
        256;
#else
        384;
#endif
    const int base_chunk_tokens =
        qwen35_dense ? 768 : ((qwen35_moe || qwen_hybrid_ssm) ? hybrid_ssm_chunk_tokens : 192);
    const int base_auto_min_tokens = qwen35_dense ? 1024 : (qwen35_moe ? 1024 : (qwen_hybrid_ssm ? 1280 : 1536));
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
    const size_t safety_margin_bytes = GraphContextSafetyMarginBytes();
    const size_t prompt_tokens = static_cast<size_t>(std::max(1, RequestPromptTokenCountForChunking(req)));
    int effective_chunk = chunk_tokens;
    const bool originally_unchunked = effective_chunk <= 1;
    const auto initial_estimate = EngineState::EstimateGraphContextSize(
        model, prompt_tokens, /*num_seqs_hint=*/1, originally_unchunked ? 0 : static_cast<size_t>(effective_chunk));
    req->graph_ctx_requested_mb = initial_estimate.total_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_available_mb = available_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_safety_margin_mb = safety_margin_bytes / (1024ULL * 1024ULL);
    if (initial_estimate.total_bytes + safety_margin_bytes <= available_bytes) {
        return chunk_tokens;
    }
    if (originally_unchunked) {
        const auto descriptor = densecore::models::DescribeModel(model);
        const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
        const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
        effective_chunk = (qwen35_moe || qwen_hybrid_ssm) ? 384 : 512;
        effective_chunk = std::max(1, std::min<int>(effective_chunk, static_cast<int>(prompt_tokens)));
    }
    const int best_effort_pressure_chunk = std::max(1, effective_chunk);

    while (effective_chunk > 1) {
        const auto estimate = EngineState::EstimateGraphContextSize(model, prompt_tokens, /*num_seqs_hint=*/1,
                                                                    static_cast<size_t>(effective_chunk));
        if (estimate.total_bytes + safety_margin_bytes <= available_bytes) {
            req->graph_ctx_downgraded_chunk_tokens = effective_chunk;
            req->graph_ctx_fail_reason.clear();
            std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
                      << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
                      << " downgraded_chunk_tokens=" << effective_chunk
                      << " requested_mb=" << (initial_estimate.total_bytes / (1024ULL * 1024ULL))
                      << " downgraded_mb=" << (estimate.total_bytes / (1024ULL * 1024ULL))
                      << " available_mb=" << req->graph_ctx_available_mb
                      << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << std::endl;
            return effective_chunk;
        }
        effective_chunk = std::max(1, effective_chunk / 2);
    }

    effective_chunk = std::max(1, best_effort_pressure_chunk);
    req->graph_ctx_downgraded_chunk_tokens = effective_chunk;
    req->graph_ctx_fail_reason.clear();
    std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
              << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
              << " downgraded_chunk_tokens=" << effective_chunk
              << " requested_mb=" << (initial_estimate.total_bytes / (1024ULL * 1024ULL)) << " downgraded_mb=unknown"
              << " available_mb=" << req->graph_ctx_available_mb
              << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << " reason=best_effort_pressure_chunk"
              << std::endl;
    return effective_chunk;
}
