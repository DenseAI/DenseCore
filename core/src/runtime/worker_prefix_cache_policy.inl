// Prefix-cache and recurrent-state snapshot policy helpers.
bool IsPrefixCacheReuseDisabled() {
    return GetWorkerRuntimeConfig().prefix_cache_reuse_disabled;
}

bool IsHybridSSMSnapshotRestoreDisabled() {
    return GetWorkerRuntimeConfig().hybrid_ssm_snapshot_restore_disabled;
}

bool IsQwen36PrefixCacheReuseEnabled() {
    return GetWorkerRuntimeConfig().qwen36_prefix_cache_reuse_enabled;
}

bool IsQwen36HybridSSMSnapshotRestoreEnabled() {
    return GetWorkerRuntimeConfig().qwen36_hybrid_ssm_snapshot_restore_enabled;
}

bool IsQwen36HybridSSMModel(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    return densecore::models::DescribeModel(model).variant == ModelVariant::QWEN36;
}

bool IsQwenHybridSSMModel(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const ModelVariant variant = densecore::models::DescribeModel(model).variant;
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36;
}

bool Qwen36HybridSSMProjectionWeightsAreNotQ4K(const TransformerModel* model) {
    if (!IsQwen36HybridSSMModel(model)) {
        return false;
    }
    bool saw_ssm_projection = false;
    for (const auto& layer : model->layers) {
        for (const char* key : {model_keys::kAttnQkvWeight, model_keys::kAttnGate, model_keys::kSSMOut}) {
            const ggml_tensor* tensor = layer.Get(key);
            if (!tensor) {
                continue;
            }
            saw_ssm_projection = true;
            if (tensor->type == GGML_TYPE_Q4_K) {
                return false;
            }
        }
    }
    return saw_ssm_projection;
}

struct Qwen36SSMProjectionTypeSummary {
    std::string actual_types;
    int quant_preserved = 0;
    int dequantized_count = 0;
};

Qwen36SSMProjectionTypeSummary SummarizeQwen36SSMProjectionTypes(const TransformerModel* model) {
    Qwen36SSMProjectionTypeSummary summary;
    if (!IsQwenHybridSSMModel(model)) {
        return summary;
    }
    std::map<std::string, int> type_counts;
    int saw = 0;
    int quantized = 0;
    int dense = 0;
    for (const auto& layer : model->layers) {
        for (const auto& item : {std::pair<const char*, const char*>("ssm_qkv", model_keys::kAttnQkvWeight),
                                 std::pair<const char*, const char*>("ssm_gate", model_keys::kAttnGate),
                                 std::pair<const char*, const char*>("ssm_out", model_keys::kSSMOut)}) {
            const ggml_tensor* tensor = layer.Get(item.second);
            if (!tensor) {
                continue;
            }
            ++saw;
            if (ggml_is_quantized(tensor->type)) {
                ++quantized;
            } else if (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_F32 ||
                       tensor->type == GGML_TYPE_BF16) {
                ++dense;
            }
            std::string key = std::string(item.first) + ":" + ggml_type_name(tensor->type);
            type_counts[key]++;
        }
    }
    std::ostringstream out;
    bool first = true;
    for (const auto& [key, count] : type_counts) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << key << ":" << count;
    }
    summary.actual_types = out.str();
    summary.quant_preserved = saw > 0 && quantized == saw ? 1 : 0;
    summary.dequantized_count = dense;
    return summary;
}

std::string SummarizeQwen36SSMQ8ProjectionCounts(const TransformerModel* model) {
    if (!IsQwen36HybridSSMModel(model)) {
        return "none";
    }
    uint64_t qkv = 0;
    uint64_t gate = 0;
    uint64_t out = 0;
    for (const auto& layer : model->layers) {
        const ggml_tensor* qkv_tensor = layer.Get(model_keys::kAttnQkvWeight);
        const ggml_tensor* gate_tensor = layer.Get(model_keys::kAttnGate);
        const ggml_tensor* out_tensor = layer.Get(model_keys::kSSMOut);
        if (qkv_tensor && qkv_tensor->type == GGML_TYPE_Q8_0) {
            ++qkv;
        }
        if (gate_tensor && gate_tensor->type == GGML_TYPE_Q8_0) {
            ++gate;
        }
        if (out_tensor && out_tensor->type == GGML_TYPE_Q8_0) {
            ++out;
        }
    }
    if (qkv == 0 && gate == 0 && out == 0) {
        return "none";
    }
    std::ostringstream counts;
    counts << "ssm_qkv:" << qkv << ",ssm_gate:" << gate << ",ssm_out:" << out;
    return counts.str();
}

bool IsPrefixCacheAllowedForModel(const TransformerModel* model) {
    if (IsPrefixCacheReuseDisabled()) {
        return false;
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36PrefixCacheReuseEnabled()) {
        return false;
    }
    return true;
}

const char* PrefixCacheSkipReasonForModel(const TransformerModel* model) {
    if (IsPrefixCacheReuseDisabled()) {
        return "disabled_by_env";
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36PrefixCacheReuseEnabled()) {
        return "qwen36_prefix_cache_disabled";
    }
    return "none";
}

// Models whose layers carry per-sequence recurrent/conv state (hybrid SSM and
// LFM2 short-conv) cannot reconstruct that state from cached KV blocks alone.
// Prefix-cache reuse for them is only correct when the boundary state is
// snapshotted at block registration and restored on a cache hit. This predicate
// gates that snapshot/restore lifecycle (the machinery itself is state-agnostic;
// it copies SSMSequenceRuntimeState, which for LFM2 holds conv_state only).
bool ModelRequiresPrefixStateSnapshot(const TransformerModel* model) {
    return model && (model->arch_flags.is_hybrid_ssm || model->arch_flags.is_lfm2_shortconv);
}

bool IsHybridSSMSnapshotRestoreAllowedForModel(const TransformerModel* model) {
    if (IsHybridSSMSnapshotRestoreDisabled()) {
        return false;
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36HybridSSMSnapshotRestoreEnabled()) {
        return false;
    }
    return true;
}

BlockManager::HybridSSMSnapshotValidator BuildHybridSSMSnapshotValidatorForRequest(const TransformerModel* model,
                                                                                   const Request* req) {
    if (!ModelRequiresPrefixStateSnapshot(model) || !req || req->ssm_runtime_states.empty()) {
        return {};
    }

    const size_t expected_layers = req->ssm_runtime_states.size();
    size_t expected_conv_elems = 0;
    size_t expected_ssm_elems = 0;
    if (model->arch_flags.is_lfm2_shortconv) {
        // LFM2 short-conv layers keep only a conv-state ring (no SSM recurrent state).
        expected_conv_elems = TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(
            static_cast<int>(model->hparams.n_embd), model->lfm2_conv_kernel);
        expected_ssm_elems = 0;
    } else {
        const int expected_conv = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
        const int expected_head_dim = model->ssm_inner_size / std::max(1, model->ssm_time_step_rank);
        expected_conv_elems =
            TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(expected_conv, model->ssm_conv_kernel);
        expected_ssm_elems = TransformerModel::SSMSequenceRuntimeState::ExpectedStateElements(
            model->ssm_time_step_rank, expected_head_dim, model->ssm_state_size);
    }

    return [expected_layers, expected_conv_elems,
            expected_ssm_elems](const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
        if (states.size() != expected_layers) {
            return false;
        }
        for (const auto& state : states) {
            if (state.conv_state.size() != expected_conv_elems || state.ssm_state.size() != expected_ssm_elems) {
                return false;
            }
        }
        return true;
    };
}

void InitializeRequestPrefixCacheState(Request* req, const TransformerModel* model) {
    if (!req) {
        return;
    }
    if (req->original_prompt_tokens_for_cache.empty()) {
        req->original_prompt_tokens_for_cache = req->tokens;
    }
    if (req->prompt_tokens_for_cache.empty()) {
        req->prompt_tokens_for_cache = req->original_prompt_tokens_for_cache;
    }
    if (req->prompt_token_count <= 0) {
        req->prompt_token_count = static_cast<int>(req->original_prompt_tokens_for_cache.size());
    }
    req->prefix_cache_allowed = IsPrefixCacheAllowedForModel(model);
    if (!req->prefix_cache_allowed) {
        req->prefix_cache_skip_reason = PrefixCacheSkipReasonForModel(model);
    } else if (req->prefix_cache_skip_reason.empty()) {
        req->prefix_cache_skip_reason = "none";
    }
}

BlockManager::PrefixCacheMatch ProbeReusablePrefixCacheForRequest(PagedKVCache* kv_cache, const TransformerModel* model,
                                                                  const Request* req) {
    BlockManager::PrefixCacheMatch match;
    if (!kv_cache || !kv_cache->block_manager || !req || !req->prefix_cache_allowed ||
        req->original_prompt_tokens_for_cache.empty()) {
        return match;
    }
    const bool require_snapshot = ModelRequiresPrefixStateSnapshot(model);
    const auto snapshot_validator = BuildHybridSSMSnapshotValidatorForRequest(model, req);
    match = kv_cache->block_manager->FindLongestCachedPrefixWithVerification(
        req->original_prompt_tokens_for_cache.data(), static_cast<int>(req->original_prompt_tokens_for_cache.size()),
        require_snapshot, snapshot_validator);
    if (!match.cached_block_ids.empty()) {
        kv_cache->block_manager->Free(match.cached_block_ids);
    }
    return match;
}
