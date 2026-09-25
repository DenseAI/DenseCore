// The worker publishes request telemetry; model/ISA preparation belongs to
// the model-bound prepared-weight owner.
static void PrepareWorkerWeightsForExecution(TransformerModel* model, bool is_decode_batch, bool is_prefill_batch,
                                             int prompt_token_count,
                                             const densecore::llm::config::FastPathRuntimeConfig& config,
                                             std::vector<Request*>& batch_requests) {
    if (!model) return;
    using densecore::llm::weights::WeightExecutionPhase;
    const auto phase = is_decode_batch
                           ? WeightExecutionPhase::Decode
                           : (is_prefill_batch ? WeightExecutionPhase::Prefill : WeightExecutionPhase::Embedding);
    const auto preparation = model->prepared_weights.PrepareForExecution(*model, phase, prompt_token_count, config);
    if (!preparation.prefill_aliases_prepared) return;
    const std::string projection_counts = SummarizeQwen36SSMQ8ProjectionCounts(model);
    for (Request* req : batch_requests) {
        if (!req) continue;
        req->qwen36_ssm_q8_prefill_amx_mode = preparation.prefill_alias_mode;
        req->qwen36_ssm_q8_prefill_amx_prepared = 1;
        req->qwen36_ssm_q8_prefill_amx_prepared_projection_counts = projection_counts;
    }
}
