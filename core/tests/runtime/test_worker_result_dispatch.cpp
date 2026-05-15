#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

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
