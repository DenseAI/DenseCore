namespace {

const char* GraphExecutionRouteName(densecore::TransformerGraphExecutionRoute route) {
    switch (route) {
    case densecore::TransformerGraphExecutionRoute::RegistryBuilder: return "RegistryBuilder";
    case densecore::TransformerGraphExecutionRoute::InlineDenseAttention: return "InlineDenseAttention";
    case densecore::TransformerGraphExecutionRoute::InlineHybridSSM: return "InlineHybridSSM";
    case densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV: return "InlineSlidingWindowSharedKV";
    case densecore::TransformerGraphExecutionRoute::Reject:
    default: return "Reject";
    }
}

class InlineDenseDecoderRegistryBuilder : public densecore::TransformerGraphBuilder {
public:
    struct ggml_tensor* Build(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx,
                              const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                              struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) override {
        return BuildTransformerGraphInlineImpl(model, cache, ctx, batch, embedding_mode, gf, out_embd, out_pos);
    }

    const char* Name() const override { return "densecore_inline_dense_decoder"; }
};

struct InlineDenseDecoderRegistryBuilderRegistrar {
    InlineDenseDecoderRegistryBuilderRegistrar() {
        densecore::TransformerGraphRegistry::Instance().RegisterExact(
            densecore::kDenseDecoderGenericBuilderKey, "densecore_inline_dense_decoder",
            densecore::models::MakeDenseDecoderGenericSupport("densecore_inline_dense_decoder"),
            []() { return std::make_unique<InlineDenseDecoderRegistryBuilder>(); });
    }
};

static InlineDenseDecoderRegistryBuilderRegistrar g_inline_dense_decoder_registry_builder_registrar;

}  // namespace

struct ggml_tensor* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx_c,
                                          const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                                          struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) {
    const densecore::TransformerGraphExecutionPlan* bound_plan =
        (batch.deps && batch.deps->transformer_execution_plan) ? batch.deps->transformer_execution_plan : nullptr;
    const densecore::TransformerGraphExecutionPlan fallback_plan =
        bound_plan ? densecore::TransformerGraphExecutionPlan{}
                   : densecore::ResolveTransformerGraphExecutionPlan(model);
    const densecore::TransformerGraphExecutionPlan& plan = bound_plan ? *bound_plan : fallback_plan;

    if (IsVerboseGraphBuildLoggingEnabled()) {
        std::cerr << "[BuildTransformerGraph] Resolved capabilities: "
                  << densecore::models::FormatModelGraphCapabilities(plan.resolution.capabilities) << std::endl;
        std::cerr << "[BuildTransformerGraph] Selected graph family: "
                  << densecore::models::FormatGraphFamilyResolution(plan.resolution) << std::endl;
        if (!plan.debug_reason.empty()) {
            std::cerr << "[BuildTransformerGraph] Dispatch detail: " << plan.debug_reason << std::endl;
        }
        std::cerr << "[BuildTransformerGraph] Execution route: " << GraphExecutionRouteName(plan.route)
                  << (plan.selected_builder_name.empty() ? "" : " via ")
                  << (plan.selected_builder_name.empty() ? "" : plan.selected_builder_name.c_str()) << std::endl;
    }

    switch (plan.route) {
    case densecore::TransformerGraphExecutionRoute::RegistryBuilder: {
        std::string execution_error;
        auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &execution_error);
        if (!builder) {
            throw densecore::GraphBuildException("BuildTransformerGraph dispatch selected registry builder key '" +
                                                 plan.registry_builder_key +
                                                 "' but execution failed closed: " + execution_error);
        }
        return builder->Build(model, cache, ctx_c, batch, embedding_mode, gf, out_embd, out_pos);
    }
    case densecore::TransformerGraphExecutionRoute::InlineDenseAttention:
    case densecore::TransformerGraphExecutionRoute::InlineHybridSSM:
    case densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV:
        return BuildTransformerGraphInlineImpl(model, cache, ctx_c, batch, embedding_mode, gf, out_embd, out_pos);
    case densecore::TransformerGraphExecutionRoute::Reject:
    default: throw densecore::GraphBuildException("BuildTransformerGraph fail-closed: " + plan.debug_reason);
    }
}
