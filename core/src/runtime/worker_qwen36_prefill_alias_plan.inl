struct WorkerQwen36SSMQ8PrefillAliasPlan {
    bool clear_decode_aliases = false;
    bool prepare_prefill_aliases = false;
    densecore::llm::config::Qwen36SSMQ8PrefillAMXMode mode =
        densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off;
};

static WorkerQwen36SSMQ8PrefillAliasPlan ResolveWorkerQwen36SSMQ8PrefillAliasPlan(
    const TransformerModel* model, bool is_decode_batch, bool is_prefill_batch, int prompt_token_count,
    const densecore::llm::config::FastPathRuntimeConfig& fast_path_config) {
    WorkerQwen36SSMQ8PrefillAliasPlan plan;
    plan.clear_decode_aliases = is_decode_batch;
    plan.mode = fast_path_config.qwen36_ssm_q8_prefill_amx;
    const bool qwen_hybrid_ssm =
        model && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
        model->arch_flags.is_hybrid_ssm;
    plan.prepare_prefill_aliases = is_prefill_batch && qwen_hybrid_ssm &&
                                   plan.mode != densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off &&
                                   prompt_token_count >= fast_path_config.qwen36_ssm_q8_prefill_amx_min_tokens;
    return plan;
}

static void ApplyWorkerQwen36SSMQ8PrefillAliasPlan(
    TransformerModel* model, const WorkerQwen36SSMQ8PrefillAliasPlan& plan, std::vector<Request*>& batch_requests) {
    if (plan.clear_decode_aliases) {
        ClearQwen36SSMQ8PrefillAMXAliases(model);
        if (!plan.prepare_prefill_aliases) {
            return;
        }
    }
    if (!plan.prepare_prefill_aliases || !PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(model)) {
        return;
    }
    const std::string projection_counts = SummarizeQwen36SSMQ8ProjectionCounts(model);
    for (Request* req : batch_requests) {
        if (!req) {
            continue;
        }
        req->qwen36_ssm_q8_prefill_amx_mode = static_cast<int>(plan.mode);
        req->qwen36_ssm_q8_prefill_amx_prepared = 1;
        req->qwen36_ssm_q8_prefill_amx_prepared_projection_counts = projection_counts;
    }
}
