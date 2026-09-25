#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include "densecore/exceptions.h"
#include "ggml-cpu.h"
#include <memory>
#include <string>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/graph_registry.h"
#include "densecore/models/model_graph_bridge.h"
#include "densecore/models/transformer_graph_builder.h"
#include "densecore/runtime/inference.h"
#include "densecore/models/model_types.h"

namespace {

class ScopedEnvOverride {
public:
    ScopedEnvOverride(const char* name, const char* value) : name_(name ? name : "") {
        const char* previous = std::getenv(name_.c_str());
        if (previous) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~ScopedEnvOverride() {
        if (had_previous_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    bool had_previous_ = false;
    std::string previous_;
};

TEST(BuildTransformerGraphDispatchTest, DenseDecoderSelectsRegistryBuilderRoute) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderDenseAttention);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);
    EXPECT_EQ(plan.registry_builder_key, densecore::kDenseDecoderGenericBuilderKey);
    EXPECT_EQ(plan.selected_builder_name, "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, Qwen35SelectsInlineHybridSsmRoute) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.arch_flags.requires_q_norm = true;
    model.arch_flags.requires_k_norm = true;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineHybridSSM);
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, HybridSsmMoeGraphPlanCarriesExecutionContractFacts) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 1;
    model.hparams.n_experts = 4;
    model.hparams.n_experts_used = 2;
    model.hybrid_layer_is_ssm = {1};
    model.ssm_time_step_rank = 8;
    model.ssm_inner_size = 64;
    model.ssm_group_count = 1;
    model.ssm_state_size = 16;
    model.ssm_conv_kernel = 4;
    model.layers.resize(1);
    model.layers[0].is_moe = true;
    model.layers[0].experts.resize(4);

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineHybridSSM);
    EXPECT_TRUE(plan.resolution.capabilities.requires_decode_graph_runtime_rebind);
    EXPECT_TRUE(plan.resolution.capabilities.requires_fallback_free_fast_path);
    EXPECT_TRUE(plan.resolution.capabilities.requires_native_moe_fast_path);
    EXPECT_EQ(plan.resolution.capabilities.native_moe_max_direct_tokens, 4096);
}

TEST(BuildTransformerGraphDispatchTest, Qwen38SelectsInlineHybridSsmRoute) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN38;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_layer = 64;
    model.gguf_declared_layer_count = 65;
    model.skipped_mtp_layer_count = 1;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineHybridSSM);
    EXPECT_TRUE(plan.resolution.capabilities.requires_fallback_free_fast_path);
}

TEST(BuildTransformerGraphDispatchTest, Gemma4SelectsInlineSlidingSharedKvRoute) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_is_sliding = {1, 0};
    model.gemma4_layer_kv_source = {0, 0};
    model.gemma4_layer_n_head_kv = {4, 2};
    model.hparams.n_head_kv = 4;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderSlidingWindowSharedKV);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineSlidingWindowSharedKV);
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, Gemma4GraphCapabilitiesUseMaintainedDecoderSpecSemantics) {
    ScopedEnvOverride disable_sliding("DENSECORE_GEMMA4_DISABLE_SLIDING_WINDOW", "1");
    ScopedEnvOverride disable_shared_kv("DENSECORE_GEMMA4_DISABLE_SHARED_KV", "1");

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_is_sliding = {1, 0};
    model.gemma4_layer_kv_source = {0, 0};
    model.gemma4_layer_n_head_kv = {4, 2};
    model.hparams.n_head_kv = 4;

    const auto spec = densecore::models::BuildDecoderModelSpec(&model);
    const auto capabilities = densecore::models::ResolveModelGraphCapabilities(&model);

    EXPECT_TRUE(spec.has_sliding_window_attention);
    EXPECT_TRUE(spec.has_shared_kv);
    EXPECT_TRUE(capabilities.has_sliding_window_attention);
    EXPECT_TRUE(capabilities.has_shared_kv_source);
}

TEST(BuildTransformerGraphDispatchTest, ExecutionPlanFollowsAttachedDecoderSpecBeforeRawArchDefaults) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    densecore::models::DecoderModelSpec spec{};
    spec.arch = ModelArch::LLAMA;
    spec.has_hybrid_ssm_mixer = true;
    spec.output.prefill_logits_policy = densecore::models::DecoderPrefillLogitsPolicy::FullSequence;
    model.decoder_spec = std::make_shared<const densecore::models::DecoderModelSpec>(spec);

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_TRUE(plan.resolution.capabilities.has_hybrid_ssm_mixer);
    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderHybridSSM);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineHybridSSM);
}

TEST(BuildTransformerGraphDispatchTest, AdmissionRejectsMissingFineGrainedMoERouterContract) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 4;
    model.hparams.n_experts_used = 2;
    model.layers.resize(1);
    model.layers[0].is_moe = true;

    densecore::models::GraphBuilderSupport support{};
    support.builder_name = "unit-test-partial-moe";
    support.supported_families = {densecore::models::GraphFamily::DecoderDenseAttention};
    support.supports_moe = true;
    support.supported_ffn_activations = {densecore::models::DecoderActivation::GeluPytorchTanh};
    support.supported_rope_kinds = {densecore::models::DecoderRopeKind::Proportional};
    support.supported_prefill_logits_policies = {densecore::models::DecoderPrefillLogitsPolicy::LastTokenForMoE};
    support.supported_semantic_ops = {
        densecore::models::DecoderSemanticOpKind::AttentionNorm,
        densecore::models::DecoderSemanticOpKind::AttentionProjection,
        densecore::models::DecoderSemanticOpKind::AttentionCore,
        densecore::models::DecoderSemanticOpKind::AttentionOutputProjection,
        densecore::models::DecoderSemanticOpKind::FfnNorm,
        densecore::models::DecoderSemanticOpKind::MoERouter,
        densecore::models::DecoderSemanticOpKind::MoEExpertDispatch,
        densecore::models::DecoderSemanticOpKind::ResidualAdd,
    };

    const auto admission = densecore::models::AdmitGraphBuilder(densecore::models::ResolveGraphFamily(&model), support);

    EXPECT_FALSE(admission.admitted);
    EXPECT_NE(admission.Summary().find("MoE router kinds [gemma4_softmax_top_k]"), std::string::npos);
}

TEST(BuildTransformerGraphDispatchTest, UnsupportedMultimodalProjectionFailsClosed) {
    TransformerModel model{};
    model.arch = ModelArch::LLAVA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::MultimodalProjectedDecoder);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::Reject);
    EXPECT_NE(plan.debug_reason.find("does not implement graph family MultimodalProjectedDecoder"), std::string::npos);
}

TEST(BuildTransformerGraphDispatchTest, Qwen3DoesNotUseLegacyArchOnlyRegistryFallback) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN3;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.preferred_family, densecore::models::GraphFamily::DecoderDenseAttention);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::InlineDenseAttention);
    EXPECT_TRUE(plan.registry_builder_key.empty());
    EXPECT_TRUE(plan.selected_builder_name.empty());
}

TEST(BuildTransformerGraphDispatchTest, ExactRegistryKeyIsPresentAndLlamaCompatibilityKeyIsNotUsed) {
    auto exact = densecore::TransformerGraphRegistry::Instance().GetBuilderExact(densecore::kDenseDecoderGenericBuilderKey);
    auto legacy = densecore::TransformerGraphRegistry::Instance().GetBuilderExact("llama");
    auto exact_descriptor =
        densecore::TransformerGraphRegistry::Instance().GetBuilderDescriptorExact(densecore::kDenseDecoderGenericBuilderKey);
    auto legacy_descriptor = densecore::TransformerGraphRegistry::Instance().GetBuilderDescriptorExact("llama");

    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(std::string(exact->Name()), "densecore_inline_dense_decoder");
    ASSERT_TRUE(exact_descriptor.has_value());
    EXPECT_EQ(exact_descriptor->display_name, "densecore_inline_dense_decoder");
    EXPECT_EQ(legacy, nullptr);
    EXPECT_FALSE(legacy_descriptor.has_value());
}

TEST(BuildTransformerGraphDispatchTest, ExactBuilderIdentityDoesNotDependOnFallbackOrder) {
    auto builder = densecore::TransformerGraphRegistry::Instance().GetBuilderExact(densecore::kDenseDecoderGenericBuilderKey);
    ASSERT_NE(builder, nullptr);
    EXPECT_EQ(std::string(builder->Name()), "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, UnknownTopologyFailsClosed) {
    TransformerModel model{};
    model.arch = ModelArch::VIT;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);

    EXPECT_EQ(plan.resolution.capabilities.topology, densecore::models::GraphTopology::UNKNOWN);
    EXPECT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::Reject);
}

TEST(BuildTransformerGraphDispatchTest, ExecutionHelperInstantiatesExactBuilderFromPlan) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    const auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);
    ASSERT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);

    std::string error_reason;
    auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &error_reason);

    ASSERT_NE(builder, nullptr);
    EXPECT_TRUE(error_reason.empty());
    EXPECT_EQ(std::string(builder->Name()), "densecore_inline_dense_decoder");
}

TEST(BuildTransformerGraphDispatchTest, ExecutionHelperFailsClosedWhenExactKeyIsUnavailable) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    auto plan = densecore::ResolveTransformerGraphExecutionPlan(&model);
    ASSERT_EQ(plan.route, densecore::TransformerGraphExecutionRoute::RegistryBuilder);
    plan.registry_builder_key = "llama";

    std::string error_reason;
    auto builder = densecore::InstantiateRegistryBuilderForExecutionPlan(plan, &error_reason);

    EXPECT_EQ(builder, nullptr);
    EXPECT_NE(error_reason.find("exact registry builder key 'llama' is not available for execution"),
              std::string::npos);
}

TEST(ModelGraphBridgeContractTest, OpenVlaHeuristicRequiresExplicitContractName) {
    static TransformerModel model{};
    model.arch = ModelArch::LLAMA;

    ASSERT_TRUE(densecore::ModelGraphBridge::RegisterFromModel(&model));
    EXPECT_EQ(densecore::GraphRegistry::Instance().GetBuilder("openvla"), nullptr);
    EXPECT_NE(densecore::GraphRegistry::Instance().GetBuilder("openvla_contract"), nullptr);
}

}  // namespace

TEST(BuildTransformerGraphDispatchTest, BertRejectsGenerationBeforeBuildingGraph) {
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    BatchSpec batch{};
    EXPECT_THROW(BuildTransformerGraph(&model, nullptr, nullptr, batch, false, nullptr, nullptr, nullptr),
                 densecore::GraphBuildException);
}

TEST(BuildTransformerGraphDispatchTest, BertEmbeddingGraphRetainsTokenAndPositionComputation) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    ASSERT_NE(context, nullptr);
    TransformerModel model{};
    model.arch = ModelArch::BERT;
    model.hparams.n_layer = 0;
    model.tok_embeddings = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 2, 3);
    model.position_embeddings = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 2, 3);
    const float tokens[] = {1, 2, 3, 4, 5, 6};
    const float positions[] = {10, 20, 30, 40, 50, 60};
    std::memcpy(model.tok_embeddings->data, tokens, sizeof(tokens));
    std::memcpy(model.position_embeddings->data, positions, sizeof(positions));
    BatchSpec batch{};
    batch.tokens = {2, 0};
    batch.pos = {0, 1};
    auto* graph = ggml_new_graph(context.get());
    ggml_tensor* token_ids = nullptr;
    ggml_tensor* position_ids = nullptr;
    auto* output = BuildTransformerGraph(&model, nullptr, context.get(), batch, true, graph,
                                        &token_ids, &position_ids);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(ggml_graph_compute_with_ctx(context.get(), graph, 1), GGML_STATUS_SUCCESS);
    EXPECT_STREQ(ggml_get_name(output), "bert_encoder_hidden_states");
    EXPECT_EQ(ggml_get_i32_1d(token_ids, 0), 2);
    EXPECT_EQ(ggml_get_i32_1d(position_ids, 1), 1);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 0), 15);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 1), 26);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 2), 31);
    EXPECT_FLOAT_EQ(ggml_get_f32_1d(output, 3), 42);
}
