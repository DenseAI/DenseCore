#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/worker_internal.h"

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
    req.id = 7;
    req.prompt_token_count = 11;
    req.sampled_token_count = 3;
    req.visible_emitted_token_count = 3;
    req.decode_finish_cause = DecodeFinishCause::MaxTokens;
    req.start_time = std::chrono::steady_clock::now();
    req.first_token_time = req.start_time + std::chrono::milliseconds(10);
    req.last_external_emit_time = req.first_token_time + std::chrono::milliseconds(20);

    ::testing::internal::CaptureStderr();
    LogRequestDecodeSummary(&req, &model);
    const std::string captured = ::testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("[Qwen35DecodeSummary]"), std::string::npos);
    EXPECT_NE(captured.find("prompt_tokens=11"), std::string::npos);
    EXPECT_NE(captured.find("attention_ms="), std::string::npos);
    EXPECT_NE(captured.find("moe_forward_ms="), std::string::npos);
}
