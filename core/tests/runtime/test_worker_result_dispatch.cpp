#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

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
    req.moe_q5k_repacked_last_reject_reason = "env_or_kernel_disabled";
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
    EXPECT_NE(captured.find("moe_q5k_repacked_last_reject_reason=env_or_kernel_disabled"), std::string::npos);
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
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.qwen35_moe_w1w3_weight_type_hist[0] = 1;
    req.qwen35_moe_w2_weight_type_hist[1] = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
    EXPECT_NE(captured.find("qwen_fast_path_ok=1"), std::string::npos);
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
    req.qwen35_moe_forward_calls = 1;
    req.qwen35_moe_path = "native_graph";
    req.qwen35_moe_w1w3_weight_type_hist[0] = 2;
    req.qwen35_moe_w2_weight_type_hist[0] = 3;
    req.ssm_conv1d_calls = 4;
    req.graph_cache_miss_count = 1;

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[LFM2DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_native_moe_decode_used_ops=5"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w1w3_q4k_repacked_used_ops=2"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_w2_q4k_repacked_used_ops=3"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_shortconv_sequence_fast_used_ops=4"), std::string::npos);
    EXPECT_NE(captured.find("lfm2_decode_graph_rebuilds=1"), std::string::npos);
    EXPECT_NE(captured.find("model_execution_contract_has_lfm2_shortconv=1"), std::string::npos);
    EXPECT_NE(captured.find("graph_plan_route=inline_hybrid_ssm"), std::string::npos);
    EXPECT_NE(captured.find("graph_plan_family=DecoderHybridSSM"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_required=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_ok=1"), std::string::npos);
    EXPECT_NE(captured.find("target_fast_path_failure_reason=none"), std::string::npos);
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
