#include <gtest/gtest.h>

#include <ggml.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "densecore/models/model_execution_contract.h"
#include "densecore/runtime/ggml_compute_policy.h"
#include "runtime/worker_internal.h"
#include "runtime/kernel_admission.h"

namespace {

struct CallbackCapture {
    std::mutex mu;
    std::vector<std::string> tokens;
    std::vector<int> token_ids;
    std::vector<int> finished_flags;
};

void TokenCallbackCapture(const char* token, int is_final, void* user_data) {
    auto* capture = static_cast<CallbackCapture*>(user_data);
    ASSERT_NE(capture, nullptr);
    std::lock_guard<std::mutex> lock(capture->mu);
    capture->tokens.emplace_back(token ? token : "");
    capture->finished_flags.push_back(is_final);
}

void TokenCallbackExCapture(const char* token, int len, int token_id, int is_final, void* user_data) {
    auto* capture = static_cast<CallbackCapture*>(user_data);
    ASSERT_NE(capture, nullptr);
    std::lock_guard<std::mutex> lock(capture->mu);
    capture->tokens.emplace_back(token && len > 0 ? std::string(token, token + len) : "");
    capture->token_ids.push_back(token_id);
    capture->finished_flags.push_back(is_final);
}

void TokenResultCallbackCapture(const TokenResult* result, void* user_data) {
    auto* capture = static_cast<CallbackCapture*>(user_data);
    ASSERT_NE(capture, nullptr);
    ASSERT_NE(result, nullptr);
    std::lock_guard<std::mutex> lock(capture->mu);
    capture->tokens.emplace_back(result->text ? result->text : "");
    capture->token_ids.push_back(result->token_id);
    capture->finished_flags.push_back(result->is_finished);
}

void InitDecodeSummaryRequest(Request* req) {
    ASSERT_NE(req, nullptr);
    req->id = 7;
    req->prompt_token_count = 11;
    req->sampled_token_count = 3;
    req->visible_emitted_token_count = 3;
    req->decode_finish_cause = DecodeFinishCause::MaxTokens;
    req->start_time = std::chrono::steady_clock::now();
    req->first_token_time = req->start_time + std::chrono::milliseconds(10);
    req->last_external_emit_time = req->first_token_time + std::chrono::milliseconds(20);
}

}  // namespace

TEST(WorkerResultDispatchTest, DirectTokenCallbackInvokesCallbackImmediately) {
    CallbackCapture capture;
    Request req{};
    req.callback = TokenCallbackCapture;
    req.user_data = &capture;

    EmitRequestResult(nullptr, &req, "hello", 17, false, false, /*use_direct_callback=*/true);

    ASSERT_EQ(capture.tokens.size(), 1u);
    EXPECT_EQ(capture.tokens[0], "hello");
    ASSERT_EQ(capture.finished_flags.size(), 1u);
    EXPECT_EQ(capture.finished_flags[0], 0);
}

TEST(WorkerResultDispatchTest, SuppressesQwenVisibleControlMarkers) {
    std::string text = "cobalt-river-913 /no_think";
    SuppressQwenVisibleControlMarkers(&text);
    EXPECT_EQ(text, "cobalt-river-913");

    text = "/nothink";
    SuppressQwenVisibleControlMarkers(&text);
    EXPECT_TRUE(text.empty());

    text = "plain answer";
    SuppressQwenVisibleControlMarkers(&text);
    EXPECT_EQ(text, "plain answer");
}

TEST(WorkerResultDispatchTest, DirectTokenCallbackExReceivesLengthAndTokenID) {
    CallbackCapture capture;
    Request req{};
    req.callback_ex = TokenCallbackExCapture;
    req.user_data = &capture;

    EmitRequestResult(nullptr, &req, std::string("he\0llo", 6), 123, false, false, /*use_direct_callback=*/true);

    ASSERT_EQ(capture.tokens.size(), 1u);
    EXPECT_EQ(capture.tokens[0], std::string("he\0llo", 6));
    ASSERT_EQ(capture.token_ids.size(), 1u);
    EXPECT_EQ(capture.token_ids[0], 123);
    ASSERT_EQ(capture.finished_flags.size(), 1u);
    EXPECT_EQ(capture.finished_flags[0], 0);
}

TEST(WorkerResultDispatchTest, DirectTokenResultCallbackInvokesStructuredCallbackImmediately) {
    CallbackCapture capture;
    Request req{};
    req.token_result_callback = TokenResultCallbackCapture;
    req.user_data = &capture;

    EmitRequestResult(nullptr, &req, "", 42, true, false, /*use_direct_callback=*/true);

    ASSERT_EQ(capture.tokens.size(), 1u);
    EXPECT_EQ(capture.tokens[0], "");
    ASSERT_EQ(capture.token_ids.size(), 1u);
    EXPECT_EQ(capture.token_ids[0], 42);
    ASSERT_EQ(capture.finished_flags.size(), 1u);
    EXPECT_EQ(capture.finished_flags[0], 1);
}

TEST(WorkerResultDispatchTest, QueuedDispatchEnqueuesResultEvent) {
    EngineState state;
    state.status = EngineStatus::STOPPED;

    CallbackCapture capture;
    Request req{};
    req.id = 99;
    req.callback = TokenCallbackCapture;
    req.user_data = &capture;

    EmitRequestResult(&state, &req, "queued", 13, false, false, /*use_direct_callback=*/false);

    std::lock_guard<std::mutex> lock(state.result_mu);
    ASSERT_EQ(state.result_queue.size(), 1u);
    const ResultEvent& event = state.result_queue.front();
    EXPECT_EQ(event.request_id, 99);
    EXPECT_EQ(event.token_str, "queued");
    EXPECT_EQ(event.token_id, 13);
    EXPECT_FALSE(event.finished);
    EXPECT_FALSE(event.error);
    EXPECT_EQ(event.callback, TokenCallbackCapture);
}

TEST(WorkerResultDispatchTest, Qwen35DecodeSummaryUsesDedicatedTag) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    InitDecodeSummaryRequest(&req);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Qwen35DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("prompt_tokens=11"), std::string::npos);
    EXPECT_NE(captured.find("attention_ms="), std::string::npos);
    EXPECT_NE(captured.find("moe_forward_ms="), std::string::npos);
    EXPECT_NE(captured.find("kleidiai_compiled_enabled="), std::string::npos);
    EXPECT_NE(captured.find("kleidiai_candidate_ops="), std::string::npos);
    EXPECT_NE(captured.find("kleidiai_last_reject_reason="), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_valid=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_fast_path_class=qwen_hybrid_ssm_moe"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_requires_fallback_free_fast_path=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_requires_native_moe_fast_path=1"), std::string::npos);
    EXPECT_NE(captured.find("graph_plan_route=inline_hybrid_ssm"), std::string::npos);
    EXPECT_NE(captured.find("graph_plan_family=DecoderHybridSSM"), std::string::npos);
    EXPECT_NE(captured.find("qwen_hot_path_target=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_hot_path_moe_lane=1"), std::string::npos);
}

TEST(WorkerResultDispatchTest, DecodeSummaryIncludesPrefixCacheMetrics) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.prefix_cache_allowed = true;
    req.prefix_cache_hit = true;
    req.prefix_cache_skipped_tokens = 32;
    req.prefix_cache_hit_blocks = 2;
    req.prefix_cache_registered_blocks = 3;
    req.prefix_cache_extended_blocks = 1;
    req.hybrid_ssm_snapshot_restore_attempted = true;
    req.hybrid_ssm_snapshot_restore_applied = true;
    req.prefix_cache_skip_reason = "none";

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("prefix_cache_allowed=1"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_hit=1"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_skipped_tokens=32"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_hit_blocks=2"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_registered_blocks=3"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_extended_blocks=1"), std::string::npos);
    EXPECT_NE(captured.find("hybrid_ssm_snapshot_restore_attempted=1"), std::string::npos);
    EXPECT_NE(captured.find("hybrid_ssm_snapshot_restore_applied=1"), std::string::npos);
    EXPECT_NE(captured.find("prefix_cache_skip_reason=none"), std::string::npos);
}

TEST(WorkerResultDispatchTest, DecodeSummaryIncludesQ5MoeTelemetry) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.moe_q5k_repacked_candidate_ops = 3;
    req.moe_q5k_repacked_used_ops = 1;
    req.moe_q5k_repacked_rejected_ops = 2;
    req.moe_q5k_repacked_last_reject_reason = "kernel_unavailable";
    req.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops = 4;
    req.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops = 5;
    req.qwen35_moe_w1w3_weight_type_hist[2] = 6;
    req.qwen35_moe_w2_weight_type_hist[3] = 7;
    MatmulDispatchCensusEntry census{};
    census.model_family = "qwen36";
    census.phase = "decode";
    census.dispatch_path = "ggml_mul_mat_id";
    census.weight_type = "q5_k";
    census.shape_bucket = "m1xn4096xk4096";
    census.wall_ns = 2000000;
    census.ops = 8;
    req.matmul_dispatch_top_slow_entries.push_back(census);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("moe_q5k_repacked_candidate_ops=3"), std::string::npos);
    EXPECT_NE(captured.find("moe_q5k_repacked_used_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("moe_q5k_repacked_rejected_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("moe_q5k_repacked_last_reject_reason=kernel_unavailable"), std::string::npos);
    EXPECT_NE(captured.find("gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops=4"),
              std::string::npos);
    EXPECT_NE(captured.find("gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops=5"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w1w3_weight_type_hist="), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w2_weight_type_hist="), std::string::npos);
    EXPECT_NE(captured.find("matmul_dispatch_top_slow=qwen36:decode:ggml_mul_mat_id:q5_k:m1xn4096xk4096"),
              std::string::npos);
}

TEST(WorkerResultDispatchTest, QwenDecodeSummaryMarksFallbackAsFastPathFailure) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.qwen_target_ggml_compute_ops = 2;
    req.qwen_target_ggml_matmul_ops = 2;
    req.qwen_target_ggml_compute_last_reason = "temporary_reference_generic_matmul_fallback";
    req.qwen_target_ggml_compute_last_op = "ggml_mul_mat";
    req.qwen_target_ggml_compute_target = "qwen35_35b_a3b";
    req.native_moe_fallback_w2_ops = 1;
    req.native_moe_fast_w2_q5k_rejected_ops = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("qwen_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_failure_reason=ggml_compute_or_matmul_path"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=ggml_compute_or_matmul_path"), std::string::npos);
    EXPECT_NE(captured.find("qwen_target_ggml_compute_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("native_moe_fallback_w2_ops=1"), std::string::npos);
}

TEST(WorkerResultDispatchTest, QwenTargetRejectsTemporaryReferenceGgmlComputeByDefault) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const auto plan = densecore::runtime::ResolveQwenHotPathPlan(&model);
    ASSERT_TRUE(plan.target_model);
    EXPECT_TRUE(densecore::runtime::ShouldRejectQwenGgmlCompute(
        plan, "temporary_reference_generic_matmul_fallback"));
    EXPECT_TRUE(densecore::runtime::ShouldRejectQwenGgmlCompute(plan, "native_moe_w2_fast_node_missing"));

    const auto target_plan = densecore::runtime::ResolveTargetFastPathPlan(&model);
    ASSERT_TRUE(target_plan.target_model);
    EXPECT_TRUE(target_plan.qwen_target);
    EXPECT_TRUE(densecore::runtime::ShouldRejectTargetGgmlCompute(
        target_plan, "temporary_reference_generic_matmul_fallback"));
}

TEST(WorkerResultDispatchTest, Qwen35Point8BUsesDedicatedHotPathLabel) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 1024;
    model.hparams.n_layer = 24;
    model.hparams.n_experts = 0;

    const auto qwen_plan = densecore::runtime::ResolveQwenHotPathPlan(&model);
    ASSERT_TRUE(qwen_plan.target_model);
    EXPECT_TRUE(qwen_plan.qwen35_0_8b);
    EXPECT_STREQ(densecore::runtime::QwenHotPathTargetLabel(qwen_plan), "qwen35_0.8b_dense");

    const auto target_plan = densecore::runtime::ResolveTargetFastPathPlan(&model);
    ASSERT_TRUE(target_plan.qwen_target);
    EXPECT_TRUE(target_plan.qwen35_0_8b);
    EXPECT_STREQ(densecore::runtime::TargetFastPathLabel(target_plan), "qwen35_0.8b_dense");
}

TEST(WorkerResultDispatchTest, GemmaAndLfmTargetsRejectTemporaryReferenceGgmlComputeByDefault) {
    TransformerModel gemma{};
    gemma.arch = ModelArch::GEMMA;
    gemma.variant = ModelVariant::GEMMA4;
    gemma.arch_flags.is_gemma4 = true;

    const auto gemma_plan = densecore::runtime::ResolveTargetFastPathPlan(&gemma);
    ASSERT_TRUE(gemma_plan.target_model);
    EXPECT_TRUE(gemma_plan.gemma4_target);
    EXPECT_STREQ(densecore::runtime::TargetFastPathLabel(gemma_plan), "gemma4_26b_a4b");
    EXPECT_TRUE(densecore::runtime::ShouldRejectTargetGgmlCompute(
        gemma_plan, "temporary_reference_generic_matmul_fallback"));
    EXPECT_TRUE(densecore::runtime::ShouldRejectTargetGgmlCompute(
        gemma_plan, "gemma4_maintained_prefill_quant_native"));
    EXPECT_TRUE(densecore::runtime::ShouldRejectTargetGgmlCompute(
        gemma_plan, "temporary_reference_gemma4_prefill_quant_native"));

    TransformerModel lfm2{};
    lfm2.arch = ModelArch::LFM2;
    lfm2.variant = ModelVariant::LFM2MOE;
    lfm2.arch_flags.is_lfm2_shortconv = true;

    const auto lfm2_plan = densecore::runtime::ResolveTargetFastPathPlan(&lfm2);
    ASSERT_TRUE(lfm2_plan.target_model);
    EXPECT_TRUE(lfm2_plan.lfm2_target);
    EXPECT_TRUE(lfm2_plan.lfm2_shortconv_lane);
    EXPECT_STREQ(densecore::runtime::TargetFastPathLabel(lfm2_plan), "lfm2_8b_a1b");
    EXPECT_TRUE(densecore::runtime::ShouldRejectTargetGgmlCompute(
        lfm2_plan, "temporary_reference_generic_matmul_fallback"));
}

TEST(KernelResolutionPolicyTest, QwenHybridSSMQ4PrefillResolvesSemanticFallbackFreePath) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    densecore::runtime::HostKernelCapabilities caps{};
    caps.arm_sve2 = true;
    caps.q4k_true_batched = true;
    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/192, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Prefill, "blk.0.attn_qkv.weight",
        /*is_lm_head=*/false, /*compatible=*/true, caps);

    EXPECT_EQ(resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::HybridSsmMixer);
    EXPECT_EQ(resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::HybridSSMQkv);
    EXPECT_EQ(resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    EXPECT_EQ(resolution.host_backend, densecore::runtime::DenseCoreHostBackend::ArmSve2);
    EXPECT_EQ(resolution.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    EXPECT_EQ(resolution.rejected_count, 0);
    EXPECT_STREQ(densecore::runtime::DenseCoreSemanticOpName(resolution.semantic_op), "hybrid_ssm_mixer");
    EXPECT_STREQ(densecore::runtime::DenseCoreTensorRoleName(resolution.tensor_role), "hybrid_ssm_qkv");
}

TEST(KernelResolutionPolicyTest, ContractFastPathCounterKindsMirrorEmittedNames) {
    struct Case {
        ModelArch arch;
        ModelVariant variant;
        bool hybrid_ssm;
        bool gemma4;
        bool lfm2_shortconv;
        int experts;
    };
    const Case cases[] = {
        {ModelArch::QWEN35, ModelVariant::QWEN36, true, false, false, 128},
        {ModelArch::GEMMA, ModelVariant::GEMMA4, false, true, false, 128},
        {ModelArch::LFM2, ModelVariant::LFM2MOE, false, false, true, 32},
    };

    for (const Case& c : cases) {
        TransformerModel model{};
        model.arch = c.arch;
        model.variant = c.variant;
        model.arch_flags.is_hybrid_ssm = c.hybrid_ssm;
        model.arch_flags.is_gemma4 = c.gemma4;
        model.arch_flags.is_lfm2_shortconv = c.lfm2_shortconv;
        model.hparams.n_experts = c.experts;

        const auto contract = densecore::models::BuildModelExecutionContract(&model);
        ASSERT_EQ(contract.required_fast_path_counter_kinds.size(), contract.required_fast_path_counters.size());
        ASSERT_FALSE(contract.required_fast_path_counter_kinds.empty());
        const auto required_kinds = densecore::models::ModelExecutionContractRequiredFastPathCounterKinds(contract);
        ASSERT_EQ(required_kinds.size(), contract.required_fast_path_counter_kinds.size());
        for (std::size_t i = 0; i < contract.required_fast_path_counter_kinds.size(); ++i) {
            const auto kind = contract.required_fast_path_counter_kinds[i];
            EXPECT_EQ(required_kinds[i], kind);
            EXPECT_EQ(contract.required_fast_path_counters[i],
                      densecore::models::ExecutionFastPathCounterKindName(kind));
            EXPECT_EQ(densecore::models::ExecutionFastPathCounterKindFromName(contract.required_fast_path_counters[i]),
                      kind);
        }
        auto legacy_contract = contract;
        legacy_contract.required_fast_path_counter_kinds.clear();
        const auto legacy_required_kinds =
            densecore::models::ModelExecutionContractRequiredFastPathCounterKinds(legacy_contract);
        ASSERT_EQ(legacy_required_kinds.size(), contract.required_fast_path_counter_kinds.size());
        EXPECT_EQ(densecore::models::FormatModelExecutionContractRequiredFastPathCounters(legacy_contract),
                  densecore::models::FormatModelExecutionContractRequiredFastPathCounters(contract));
        for (std::size_t i = 0; i < legacy_required_kinds.size(); ++i) {
            EXPECT_EQ(legacy_required_kinds[i], contract.required_fast_path_counter_kinds[i]);
        }

        const std::string formatted = densecore::models::FormatModelExecutionContract(contract);
        EXPECT_NE(formatted.find("required_fast_path_counters=["), std::string::npos);
        EXPECT_NE(formatted.find(contract.required_fast_path_counters.front()), std::string::npos);
    }
}

TEST(KernelResolutionPolicyTest, QwenTargetGenericFallbackCarriesRejectionEvidence) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F16, /*m=*/1, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Decode, "blk.0.ssm_out.weight",
        /*is_lm_head=*/false, /*compatible=*/false);

    EXPECT_EQ(resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::HybridSsmMixer);
    EXPECT_EQ(resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::SSMOut);
    EXPECT_EQ(resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::TemporaryReferenceGgml);
    EXPECT_EQ(resolution.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    ASSERT_GE(resolution.rejected_count, 2);
    EXPECT_STREQ(resolution.rejected[0], "incompatible_shape");
    EXPECT_STREQ(resolution.rejected[1], "unsupported_input_type");
}

TEST(KernelResolutionPolicyTest, LFM2ShortConvAndLmHeadExposeSemanticRoles) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    const auto shortconv_resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q5_K, GGML_TYPE_F32, /*m=*/1, /*n=*/2048, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Decode, "blk.0.shortconv.out.weight",
        /*is_lm_head=*/false, /*compatible=*/true);
    EXPECT_EQ(shortconv_resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer);
    EXPECT_EQ(shortconv_resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::ShortConvOut);
    EXPECT_EQ(shortconv_resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQuantGemv);
    EXPECT_EQ(shortconv_resolution.fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    EXPECT_TRUE(shortconv_resolution.matmul_plan.target_fallback_free_path);
    EXPECT_TRUE(shortconv_resolution.matmul_plan.target.lfm2_target);

    const auto lm_head_resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q6_K, GGML_TYPE_F32, /*m=*/1, /*n=*/32000, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Decode, "output.weight",
        /*is_lm_head=*/true, /*compatible=*/true);
    EXPECT_EQ(lm_head_resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::LmHead);
    EXPECT_EQ(lm_head_resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::LmHead);
    EXPECT_EQ(lm_head_resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQuantGemv);
    EXPECT_EQ(lm_head_resolution.fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    EXPECT_STREQ(densecore::runtime::TargetFastPathLabel(lm_head_resolution.matmul_plan.target), "lfm2_8b_a1b");
}

TEST(KernelResolutionPolicyTest, Gemma4DefaultResolutionIsFallbackFreeTarget) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 4;

    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/1, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Decode, "blk.0.ffn_down_exps.weight",
        /*is_lm_head=*/false, /*compatible=*/true);

    EXPECT_EQ(resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);
    EXPECT_EQ(resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::MoEDown);
    EXPECT_EQ(resolution.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    EXPECT_TRUE(resolution.matmul_plan.target_fallback_free_path);
    EXPECT_TRUE(resolution.matmul_plan.target.gemma4_target);
    EXPECT_STREQ(densecore::runtime::TargetFastPathLabel(resolution.matmul_plan.target), "gemma4_26b_a4b");
}

TEST(KernelResolutionPolicyTest, ContractRequirementOverridesKernelAndFallbackPolicy) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 4;

    const auto requirement = densecore::models::ResolveModelTensorExecutionRequirement(
        &model, "blk.0.ffn_gate_up_exps.weight", /*is_lm_head=*/false, GGML_TYPE_Q4_K);
    ASSERT_EQ(requirement.tensor_role, densecore::runtime::DenseCoreTensorRole::MoEGateUp);
    ASSERT_EQ(requirement.semantic_op, densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);
    ASSERT_EQ(requirement.prefill_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQwenMoeDirect);
    ASSERT_EQ(requirement.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);

    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/192, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Prefill, "blk.0.ffn_gate_up_exps.weight",
        /*is_lm_head=*/false, /*compatible=*/true, densecore::runtime::HostKernelCapabilities{},
        requirement.semantic_op, requirement.tensor_role, requirement.prefill_kernel, requirement.fallback_policy,
        /*has_fallback_policy_override=*/true);

    EXPECT_EQ(resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);
    EXPECT_EQ(resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::MoEGateUp);
    EXPECT_EQ(resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQwenMoeDirect);
    EXPECT_EQ(resolution.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
}

TEST(KernelResolutionPolicyTest, LoaderTensorRoleOverridesNameFallback) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 4;
    model.layers.resize(1);

    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 16);
    ggml_set_name(tensor, "blk.0.plain_projection.weight");
    model.layers[0].Set("loader.bound.moe_down", tensor, densecore::runtime::DenseCoreTensorRole::MoEDown);

    const auto requirement = densecore::models::ResolveModelTensorExecutionRequirement(
        &model, tensor, tensor->name, /*is_lm_head=*/false, tensor->type);
    EXPECT_EQ(requirement.tensor_role, densecore::runtime::DenseCoreTensorRole::MoEDown);
    EXPECT_EQ(requirement.semantic_op, densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);
    EXPECT_EQ(requirement.prefill_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQwenMoeDirect);
    EXPECT_EQ(requirement.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);

    const auto fallback_only = densecore::models::ResolveModelTensorExecutionRequirement(
        &model, tensor->name, /*is_lm_head=*/false, tensor->type);
    EXPECT_EQ(fallback_only.tensor_role, densecore::runtime::DenseCoreTensorRole::DenseProjection);

    ggml_free(ctx);
}

TEST(KernelResolutionPolicyTest, TensorRoleOverrideSelectsMaintainedKernelWithoutNameOrKernelOverride) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 4;

    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/128, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Prefill, "blk.0.plain_projection.weight",
        /*is_lm_head=*/false, /*compatible=*/true, densecore::runtime::HostKernelCapabilities{},
        densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch,
        densecore::runtime::DenseCoreTensorRole::MoEDown);

    EXPECT_EQ(resolution.semantic_op, densecore::runtime::DenseCoreSemanticOp::MoeExpertDispatch);
    EXPECT_EQ(resolution.tensor_role, densecore::runtime::DenseCoreTensorRole::MoEDown);
    EXPECT_EQ(resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQwenMoeDirect);
}

TEST(KernelResolutionPolicyTest, ContractRequirementPhaseKernelDrivesResolutionMatrix) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const auto requirement = densecore::models::ResolveModelTensorExecutionRequirement(
        &model, "blk.0.attn_qkv.weight", /*is_lm_head=*/false, GGML_TYPE_Q4_K);
    ASSERT_EQ(requirement.semantic_op, densecore::runtime::DenseCoreSemanticOp::HybridSsmMixer);
    ASSERT_EQ(requirement.tensor_role, densecore::runtime::DenseCoreTensorRole::HybridSSMQkv);
    ASSERT_EQ(requirement.prefill_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    ASSERT_EQ(requirement.decode_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQuantGemv);
    EXPECT_EQ(densecore::models::ModelTensorExecutionRequirementKernelForPhase(
                  requirement, densecore::runtime::DenseCoreMatmulPhase::Prefill),
              densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    EXPECT_EQ(densecore::models::ModelTensorExecutionRequirementKernelForPhase(
                  requirement, densecore::runtime::DenseCoreMatmulPhase::Decode),
              densecore::runtime::DenseCoreKernelFamily::DenseCoreQuantGemv);

    densecore::runtime::HostKernelCapabilities caps{};
    caps.arm_sve2 = true;
    caps.q4k_true_batched = true;
    const auto prefill_resolution = densecore::models::ResolveModelTensorKernelResolution(
        &model, requirement, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/128, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Prefill, "blk.0.attn_qkv.weight", /*is_lm_head=*/false,
        /*compatible=*/true, caps);
    const auto decode_resolution = densecore::models::ResolveModelTensorKernelResolution(
        &model, requirement, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/1, /*n=*/4096, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Decode, "blk.0.attn_qkv.weight", /*is_lm_head=*/false,
        /*compatible=*/true, caps);

    EXPECT_EQ(prefill_resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    EXPECT_EQ(decode_resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQuantGemv);
    EXPECT_EQ(prefill_resolution.host_backend, densecore::runtime::DenseCoreHostBackend::ArmSve2);
    EXPECT_EQ(decode_resolution.host_backend, densecore::runtime::DenseCoreHostBackend::ArmSve2);
    EXPECT_EQ(prefill_resolution.fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
    EXPECT_EQ(decode_resolution.fallback_policy,
              densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
}

TEST(KernelResolutionPolicyTest, Q4KBatchedPrefillHelpersKeepTargetSeparateFromShapeReadiness) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    const auto resolution = densecore::runtime::ResolveKernelResolution(
        &model, GGML_TYPE_Q4_K, GGML_TYPE_F32, /*m=*/128, /*n=*/2048, /*k=*/2048,
        densecore::runtime::DenseCoreMatmulPhase::Prefill, "blk.0.shortconv_in_proj.weight",
        /*is_lm_head=*/false, /*compatible=*/false, densecore::runtime::HostKernelCapabilities{},
        densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer,
        densecore::runtime::DenseCoreTensorRole::ShortConvIn);

    EXPECT_EQ(resolution.selected_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    EXPECT_TRUE(densecore::runtime::KernelResolutionTargetsQ4KBatchedPrefill(resolution));
    EXPECT_FALSE(densecore::runtime::KernelResolutionSelectsQ4KBatchedPrefill(resolution));
    ASSERT_GE(resolution.rejected_count, 1);
    EXPECT_STREQ(resolution.rejected[0], "incompatible_shape");
}

TEST(KernelResolutionPolicyTest, TensorRoleHelpersClassifyProjectionFamilies) {
    using densecore::runtime::DenseCoreTensorRole;

    EXPECT_TRUE(densecore::runtime::IsHybridSsmTensorRole(DenseCoreTensorRole::HybridSSMQkv));
    EXPECT_TRUE(densecore::runtime::IsHybridSsmTensorRole(DenseCoreTensorRole::HybridSSMGate));
    EXPECT_TRUE(densecore::runtime::IsHybridSsmTensorRole(DenseCoreTensorRole::SSMOut));
    EXPECT_FALSE(densecore::runtime::IsHybridSsmTensorRole(DenseCoreTensorRole::ShortConvIn));
    EXPECT_EQ(densecore::runtime::HybridSsmProjectionKind(DenseCoreTensorRole::HybridSSMQkv), 1);
    EXPECT_EQ(densecore::runtime::HybridSsmProjectionKind(DenseCoreTensorRole::HybridSSMGate), 2);
    EXPECT_EQ(densecore::runtime::HybridSsmProjectionKind(DenseCoreTensorRole::SSMOut), 3);
    EXPECT_EQ(densecore::runtime::HybridSsmProjectionKind(DenseCoreTensorRole::LmHead), 0);

    EXPECT_TRUE(densecore::runtime::IsShortConvTensorRole(DenseCoreTensorRole::ShortConvIn));
    EXPECT_TRUE(densecore::runtime::IsShortConvTensorRole(DenseCoreTensorRole::ShortConvOut));
    EXPECT_TRUE(densecore::runtime::IsMoeExpertTensorRole(DenseCoreTensorRole::MoEGateUp));
    EXPECT_TRUE(densecore::runtime::IsMoeExpertTensorRole(DenseCoreTensorRole::MoEDown));
    EXPECT_FALSE(densecore::runtime::IsMoeExpertTensorRole(DenseCoreTensorRole::MoERouter));
}

TEST(KernelResolutionPolicyTest, ContractRequirementMakesLFM2ShortConvFallbackFree) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;

    const auto requirement = densecore::models::ResolveModelTensorExecutionRequirement(
        &model, "blk.0.shortconv_in_proj.weight", /*is_lm_head=*/false, GGML_TYPE_Q4_K);

    EXPECT_EQ(requirement.semantic_op, densecore::runtime::DenseCoreSemanticOp::Lfm2ShortConvMixer);
    EXPECT_EQ(requirement.tensor_role, densecore::runtime::DenseCoreTensorRole::ShortConvIn);
    EXPECT_EQ(requirement.prefill_kernel, densecore::runtime::DenseCoreKernelFamily::DenseCoreQ4KBatched);
    EXPECT_EQ(requirement.fallback_policy, densecore::runtime::DenseCoreFallbackPolicyKind::FallbackFreeTarget);
}

TEST(WorkerResultDispatchTest, DenseQwenDecodeSummaryRequiresTargetFastPath) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = false;
    model.hparams.n_experts = 0;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.qwen_target_ggml_compute_ops = 1;
    req.qwen_target_ggml_matmul_ops = 1;
    req.qwen_target_ggml_compute_last_reason = "temporary_reference_dense_projection";
    req.qwen_target_ggml_compute_last_op = "ggml_mul_mat";
    req.qwen_target_ggml_compute_target = "qwen35_9b_dense";

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("qwen_hot_path_target=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_hot_path_dense_lane=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_fast_path_class=qwen_dense"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_requires_fallback_free_fast_path=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_requires_native_moe_fast_path=0"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_failure_reason=ggml_compute_or_matmul_path"), std::string::npos);
    EXPECT_NE(captured.find("qwen_target_ggml_compute_target=qwen35_9b_dense"), std::string::npos);
}

TEST(WorkerResultDispatchTest, QwenDecodeSummaryAcceptsNativeMoeFastGateUpAndDown) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.native_moe_fast_decode_used_ops = 2;
    req.native_moe_fast_decode_w1w3_used_ops = 1;
    req.native_moe_fast_decode_w2_used_ops = 1;
    req.native_moe_fast_w1w3_used_ops = 1;
    req.native_moe_fast_w2_used_ops = 1;
    req.native_moe_fast_w2_q5k_used_ops = 1;
    req.qwen_native_moe_w2_q5k_raw_batched_used_ops = 1;
    req.qwen_native_moe_w2_q5k_raw_batched_ns = 2'000'000;
    req.qwen_native_moe_fused_router_used_ops = 40;
    req.qwen_native_moe_fused_router_ns = 1'500'000;
    req.qwen_native_moe_q4_gateup_rowpair_used_ops = 20;
    req.qwen_native_moe_q4_gateup_rowpair_ns = 2'500'000;
    req.moe_kquant_raw_batched_q4k_used_ops = 2;
    req.moe_kquant_raw_batched_q4k_ns = 3'000'000;
    req.moe_kquant_raw_batched_q5k_used_ops = 3;
    req.moe_kquant_raw_batched_q5k_ns = 4'000'000;
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.qwen35_moe_w1w3_weight_type_hist[0] = 1;
    req.qwen35_moe_w2_weight_type_hist[1] = 1;
    req.ssm_conv1d_calls = 1;
    req.ssm_delta_calls = 1;
    req.ssm_delta_fast_default_used_ops = 1;
    req.ssm_delta_fast_default_wall_ns = 5'000'000;
    req.q6k_gemv_candidate_ops = 2;
    req.q6k_gemv_used_ops = 2;
    req.q6k_gemv_effective_phase = "decode";
    req.decode_graph_node_custom_lm_head_count = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("graph_kernel_coverage=100"), std::string::npos);
    EXPECT_NE(captured.find("graph_kernel_coverage_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("graph_kernel_missing=none"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
    EXPECT_NE(captured.find("target_no_ggml_path=1"), std::string::npos);
    EXPECT_NE(captured.find("target_no_native_moe_fallback=1"), std::string::npos);
    EXPECT_NE(captured.find("target_no_native_moe_reject=1"), std::string::npos);
    EXPECT_NE(captured.find("target_native_moe_fast_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_stateful_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=none"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w2_q5k_raw_batched_used_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w2_q5k_raw_batched_ms=2"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_fused_router_used_ops=40"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_fused_router_ms=1.5"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_q4_gateup_rowpair_used_ops=20"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_q4_gateup_rowpair_ms=2.5"), std::string::npos);
    EXPECT_NE(captured.find("moe_kquant_raw_batched_q4k_used_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("moe_kquant_raw_batched_q4k_ms=3"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w1w3_q4k_raw_batched_used_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("qwen_native_moe_w1w3_q4k_raw_batched_ms=3"), std::string::npos);
    EXPECT_NE(captured.find("moe_kquant_raw_batched_q5k_used_ops=3"), std::string::npos);
    EXPECT_NE(captured.find("moe_kquant_raw_batched_q5k_ms=4"), std::string::npos);
    EXPECT_NE(captured.find("qwen_decode_q6k_lm_head_overlap_candidate_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_decode_q6k_lm_head_overlap_used_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("qwen_decode_custom_lm_head_count=1"), std::string::npos);
    EXPECT_NE(captured.find("ssm_delta_fast_default_used_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("ssm_delta_fast_default_wall_ms=5"), std::string::npos);
}

TEST(WorkerResultDispatchTest, QwenDecodeSummaryRejectsMissingNativeMoeFastOps) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.ssm_conv1d_calls = 1;
    req.ssm_delta_calls = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=native_moe_fast_ops_missing"), std::string::npos);
    EXPECT_NE(captured.find("target_native_moe_fast_ops_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_stateful_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters="
                            "native_moe_fast_w1w3_used_ops,native_moe_fast_w2_used_ops"),
              std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_failure_reason=native_moe_fast_ops_missing"), std::string::npos);
}

TEST(WorkerResultDispatchTest, LFM2DecodeSummaryIncludesDedicatedFastPathAliases) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 32;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.native_moe_fast_decode_used_ops = 5;
    req.native_moe_fast_decode_w1w3_used_ops = 2;
    req.native_moe_fast_decode_w2_used_ops = 3;
    req.native_moe_fast_w1w3_used_ops = 2;
    req.native_moe_fast_w2_used_ops = 3;
    req.lfm2_w1w3_q4k_vecdot_rowpair_used_ops = 2;
    req.lfm2_w1w3_q4k_vecdot_scalar_used_ops = 1;
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.qwen35_moe_w1w3_weight_type_hist[0] = 2;
    req.qwen35_moe_w2_weight_type_hist[0] = 3;
    req.graph_cache_miss_count = 1;
    req.decode_graph_node_custom_moe_count = 2;
    req.decode_graph_node_custom_moe_ns = 1100000;
    req.decode_graph_node_custom_ssm_count = 4;
    req.decode_graph_node_custom_ssm_ns = 2200000;
    req.decode_graph_node_custom_projection_count = 6;
    req.decode_graph_node_custom_projection_ns = 4400000;
    req.decode_graph_node_custom_lm_head_count = 1;
    req.decode_graph_node_custom_lm_head_ns = 3300000;
    req.lfm2_greedy_lm_head_argmax_candidate_ops = 1;
    req.lfm2_greedy_lm_head_argmax_rejected_ops = 1;
    req.lfm2_greedy_lm_head_argmax_last_reject_reason =
        static_cast<int>(LFM2GreedyLMHeadArgmaxRejectReason::ParityUnverified);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[LFM2DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_native_moe_decode_used_ops=5"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w1w3_q4k_repacked_used_ops=0"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w1w3_q4k_vecdot_rowpair_used_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w1w3_q4k_vecdot_scalar_used_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w2_q4k_repacked_used_ops=3"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_shortconv_sequence_fast_used_ops=4"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_greedy_lm_head_argmax_candidate_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_greedy_lm_head_argmax_used_ops=0"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_greedy_lm_head_argmax_rejected_ops=1"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_greedy_lm_head_argmax_last_reject_reason=parity_unverified"),
              std::string::npos);
    EXPECT_NE(captured.find("lfm2_decode_graph_rebuilds=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_has_lfm2_shortconv=1"), std::string::npos);
    EXPECT_NE(captured.find("decode_graph_custom_node_hist=moe:count=2:ms=1.1,ssm_stateful:count=4:ms=2.2,"
                            "projection:count=6:ms=4.4,lm_head:count=1:ms=3.3"),
              std::string::npos);
    EXPECT_NE(captured.find("graph_plan_route=inline_hybrid_ssm"), std::string::npos);
    EXPECT_NE(captured.find("graph_plan_family=DecoderHybridSSM"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
    EXPECT_NE(captured.find("target_native_moe_fast_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_stateful_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=none"), std::string::npos);
}

TEST(LFM2GreedyLMHeadArgmax, RejectingSamplingInvalidatesPrecomputedToken) {
    InferenceWorkContext* ctx = CreateInferenceWorkContext();
    ASSERT_NE(ctx, nullptr);
    ResetInferenceWorkContext(ctx);

    SetInferenceWorkContextLFM2GreedyLMHeadArgmaxSampling(
        ctx, true, LFM2GreedyLMHeadArgmaxRejectReason::None, 1.0f, 0.0f, nullptr, nullptr);
    RecordInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(
        ctx, GetInferenceWorkContextExecutionGenerationForTest(ctx), 7, 3.5f, 16);

    int token = -1;
    float value = 0.0f;
    EXPECT_TRUE(TryGetInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(ctx, 16, &token, &value));
    EXPECT_EQ(token, 7);
    EXPECT_FLOAT_EQ(value, 3.5f);

    SetInferenceWorkContextLFM2GreedyLMHeadArgmaxSampling(
        ctx, false, LFM2GreedyLMHeadArgmaxRejectReason::ParityUnverified, 1.0f, 0.0f, nullptr, nullptr);

    token = -1;
    value = 0.0f;
    EXPECT_FALSE(TryGetInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(ctx, 16, &token, &value));
    DestroyInferenceWorkContext(ctx);
}

TEST(LFM2GreedyLMHeadArgmax, DisallowedTokenFilterDoesNotInvalidatePrecomputedToken) {
    InferenceWorkContext* ctx = CreateInferenceWorkContext();
    ASSERT_NE(ctx, nullptr);
    ResetInferenceWorkContext(ctx);

    const std::vector<int> disallowed_tokens{3, 5, 11};
    SetInferenceWorkContextLFM2GreedyLMHeadArgmaxSampling(
        ctx, true, LFM2GreedyLMHeadArgmaxRejectReason::None, 1.0f, 0.0f, nullptr, &disallowed_tokens);
    RecordInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(
        ctx, GetInferenceWorkContextExecutionGenerationForTest(ctx), 7, 4.25f, 16);

    int token = -1;
    float value = 0.0f;
    EXPECT_TRUE(TryGetInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(ctx, 16, &token, &value));
    EXPECT_EQ(token, 7);
    EXPECT_FLOAT_EQ(value, 4.25f);
    DestroyInferenceWorkContext(ctx);
}

TEST(WorkerResultDispatchTest, LFM2DecodeSummaryRejectsMissingNativeMoeFastOps) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 32;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.ssm_conv1d_calls = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[LFM2DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=native_moe_fast_ops_missing"), std::string::npos);
    EXPECT_NE(captured.find("target_native_moe_fast_ops_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_stateful_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters="
                            "native_moe_fast_w1w3_used_ops,native_moe_fast_w2_used_ops"),
              std::string::npos);
}

TEST(WorkerResultDispatchTest, LFM2DecodeSummaryRejectsMissingShortConvFastOps) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 32;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.native_moe_fast_decode_used_ops = 2;
    req.native_moe_fast_decode_w1w3_used_ops = 1;
    req.native_moe_fast_decode_w2_used_ops = 1;
    req.native_moe_fast_w1w3_used_ops = 1;
    req.native_moe_fast_w2_used_ops = 1;
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[LFM2DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=required_fast_path_counters_missing"),
              std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=lfm2_shortconv_sequence_fast_used_ops"),
              std::string::npos);
}

TEST(WorkerResultDispatchTest, LFM2DecodeSummaryMarksNativeMoeFallbackAsFastPathFailure) {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_experts = 32;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.native_moe_fallback_w1w3_ops = 1;
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.qwen35_moe_w1w3_weight_type_hist[0] = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[LFM2DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=native_moe_w1w3_or_w2_fallback"), std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryRejectsMissingFastPathCounters) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_fast_path_class=gemma4_moe"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_required_fast_path_counters="
                            "target_no_ggml_path,gemma4_prefill_maintained_fast_ops,"
                            "gemma4_decode_maintained_fast_ops"),
              std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_forbidden_fast_paths="
                            "gemma4_arm_native_moe_prefill_quality_failed"),
              std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_gemma4_moe_fast_ops_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters="
                            "gemma4_prefill_maintained_fast_ops,gemma4_decode_maintained_fast_ops"),
              std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=gemma4_moe_prefill_fast_ops_missing"),
              std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryAcceptsPrefillAndDecodeFastCounters) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.gemma4_native_moe_prefill_used_layers = 29;
    req.decode_matmul_path_hist[2] = 3;    // custom_gemv

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_fast_path_class=gemma4_moe"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_required_fast_path_counters="
                            "target_no_ggml_path,gemma4_prefill_maintained_fast_ops,"
                            "gemma4_decode_maintained_fast_ops"),
              std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_forbidden_fast_paths="
                            "gemma4_arm_native_moe_prefill_quality_failed"),
              std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_gemma4_moe_fast_ops_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=none"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryAcceptsCustomGemvDespiteOptionalDecodeNativeReject) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.gemma4_native_moe_prefill_used_layers = 29;
    req.decode_matmul_path_hist[2] = 3;  // maintained custom_gemv decode path
    req.gemma4_decode_native_rejected_ops = 4;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("gemma4_decode_native_rejected_ops=4"), std::string::npos);
    EXPECT_NE(captured.find("target_no_native_moe_reject=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryAcceptsX86CustomBatchedPrefill) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.prefill_matmul_path_hist[3] = 98;  // custom_batched_gemv
    req.decode_matmul_path_hist[2] = 3;    // custom_gemv
    req.gemma4_decode_native_rejected_ops = 3;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("prefill_matmul_path_hist=ggml_mul_mat:0,ggml_mul_mat_id:0,custom_gemv:0,"
                            "custom_batched_gemv:98"),
              std::string::npos);
    EXPECT_NE(captured.find("target_no_ggml_path=1"), std::string::npos);
    EXPECT_NE(captured.find("target_no_native_moe_reject=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=none"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryRejectsMaintainedQ8PrefillNativeMatmul) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.gemma4_native_moe_prefill_used_layers = 29;
    req.decode_matmul_path_hist[2] = 3;  // custom_gemv
    req.prefill_matmul_path_hist[0] = 69;
    req.qwen_target_ggml_compute_ops = 69;
    req.qwen_target_ggml_matmul_ops = 69;
    req.qwen_target_ggml_compute_last_reason = "gemma4_maintained_prefill_quant_native";

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("qwen_target_ggml_compute_ops=69"), std::string::npos);
    EXPECT_NE(captured.find("qwen_target_ggml_compute_last_reason=gemma4_maintained_prefill_quant_native"),
              std::string::npos);
    EXPECT_NE(captured.find("target_no_ggml_path=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=ggml_compute_or_matmul_path"), std::string::npos);
}

TEST(WorkerResultDispatchTest, Gemma4MoESummaryRejectsDecodeWhenCustomGemvIsMissing) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.variant = ModelVariant::GEMMA4;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 4;

    Request req{};
    InitDecodeSummaryRequest(&req);
    req.gemma4_native_moe_prefill_used_layers = 29;
    req.decode_matmul_path_hist[2] = 0;    // custom_gemv
    req.gemma4_decode_native_moe_used_ops = 0;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Gemma4DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_gemma4_moe_fast_ops_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_required_fast_path_counters_ok=0"), std::string::npos);
    EXPECT_NE(captured.find("target_missing_required_fast_path_counters=gemma4_decode_maintained_fast_ops"),
              std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=gemma4_moe_decode_fast_ops_missing"),
              std::string::npos);
}

TEST(WorkerResultDispatchTest, DecodeSummaryTagsQwen36AndGemma4Variants) {
    struct Case {
        ModelArch arch;
        ModelVariant variant;
        const char* tag;
    };
    const Case cases[] = {
        {ModelArch::QWEN35, ModelVariant::QWEN36, "[Qwen36DecodeSummary]"},
        {ModelArch::GEMMA, ModelVariant::GEMMA4, "[Gemma4DecodeSummary]"},
    };

    for (const Case& test_case : cases) {
        TransformerModel model{};
        model.arch = test_case.arch;
        model.variant = test_case.variant;
        Request req{};
        InitDecodeSummaryRequest(&req);

        ::testing::internal::CaptureStderr();
        LogRequestDecodeSummary(&req, &model);
        const std::string captured = ::testing::internal::GetCapturedStderr();

        EXPECT_NE(captured.find(test_case.tag), std::string::npos) << test_case.tag;
        EXPECT_NE(captured.find("prompt_tokens=11"), std::string::npos) << test_case.tag;
    }
}

TEST(WorkerResultDispatchTest, DecodeSummarySkipsGenericDecoderVariants) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.variant = ModelVariant::LLAMA;
    Request req{};
    InitDecodeSummaryRequest(&req);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_TRUE(captured.empty());
}

TEST(KernelAdmissionTest, KleidiAIDefaultsToQualityGateDenyForKnownModels) {
    densecore::runtime::KernelAdmissionDescriptor desc{};
    desc.model_variant = ModelVariant::GEMMA4;
    desc.is_gemma4 = true;
    desc.weight_name = "blk.0.ffn_gate.weight";
    desc.weight_type = GGML_TYPE_Q4_0;
    desc.input_type = GGML_TYPE_F32;
    desc.m = 128;
    desc.n = 4096;
    desc.k = 4096;

    const auto decision = densecore::runtime::EvaluateKleidiAIAdmission(desc);

    EXPECT_TRUE(decision.candidate);
    EXPECT_FALSE(decision.allowed);
    if (densecore::runtime::KleidiAICompiledEnabled()) {
        EXPECT_EQ(decision.reject_reason, densecore::runtime::KernelAdmissionRejectReason::QualityGateUnpromoted);
    } else {
        EXPECT_EQ(decision.reject_reason, densecore::runtime::KernelAdmissionRejectReason::BackendNotCompiled);
    }
    EXPECT_EQ(decision.op_kind, densecore::runtime::KernelOpKind::FfnGate);
}

TEST(KernelAdmissionTest, KleidiAIRejectsQ4KUntilSeparateParityGateExists) {
    densecore::runtime::KernelAdmissionDescriptor desc{};
    desc.model_variant = ModelVariant::QWEN36;
    desc.is_hybrid_ssm = true;
    desc.weight_name = "output.weight";
    desc.weight_type = GGML_TYPE_Q4_K;
    desc.input_type = GGML_TYPE_F32;
    desc.m = 128;
    desc.n = 4096;
    desc.k = 4096;

    const auto decision = densecore::runtime::EvaluateKleidiAIAdmission(desc);

    EXPECT_FALSE(decision.candidate);
    EXPECT_FALSE(decision.allowed);
    EXPECT_EQ(decision.reject_reason, densecore::runtime::KernelAdmissionRejectReason::UnsupportedTensorType);
}
