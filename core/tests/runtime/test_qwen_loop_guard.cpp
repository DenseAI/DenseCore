#include <gtest/gtest.h>

#include "runtime/worker_internal.h"

TEST(QwenLoopGuardTest, StopsEightIdenticalSuffixTokens) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {1, 2, 3, 9, 9, 9, 9, 9, 9, 9, 9};

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, StopsAlternatingTwoTokenLoop) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5};

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, IgnoresNormalShortSuffixes) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;

    Request req{};
    req.token_history = {10, 11, 12, 13, 13, 13, 13, 13, 13, 13};

    EXPECT_FALSE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, Qwen36DoesNotTripBeforeThirtyTwoGeneratedTokens) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;

    Request req{};
    req.generated_count = 16;
    req.token_history = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

    EXPECT_FALSE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, Qwen36StopsSixteenIdenticalSuffixTokensAfterGate) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;

    Request req{};
    req.generated_count = 32;
    req.token_history = {1, 2, 3, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, Qwen36StopsTwentyFourTokenAlternatingLoopAfterGate) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;

    Request req{};
    req.generated_count = 32;
    req.token_history = {
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    };

    EXPECT_TRUE(ShouldTerminateRepetitiveLoop(&model, &req));
}

TEST(QwenLoopGuardTest, DecodeVisibleProgressWatchdogTripsAfterSilentSteps) {
    Request req{};
    const auto now = std::chrono::steady_clock::now();
    req.last_external_emit_time = now - std::chrono::milliseconds(DecodeVisibleProgressTimeoutMs() + 1);
    req.last_sampled_token_time = now;
    req.decode_no_output_steps = static_cast<uint64_t>(DecodeVisibleProgressMaxSilentSteps());

    EXPECT_TRUE(HasDecodeVisibleProgressStalled(&req, now));
}

TEST(QwenLoopGuardTest, DecodeVisibleProgressWatchdogResetsAfterExternalEmit) {
    Request req{};
    const auto now = std::chrono::steady_clock::now();

    NoteDecodeSampleProgress(&req, now, /*token_id=*/17);
    EXPECT_FALSE(HasDecodeVisibleProgressStalled(&req, now));

    req.decode_no_output_steps = static_cast<uint64_t>(DecodeVisibleProgressMaxSilentSteps());
    req.last_external_emit_time = now - std::chrono::milliseconds(DecodeVisibleProgressTimeoutMs() + 1);
    EXPECT_TRUE(HasDecodeVisibleProgressStalled(&req, now));

    NoteVisibleEmitProgress(&req, now, /*token_id=*/17);
    EXPECT_FALSE(HasDecodeVisibleProgressStalled(&req, now));
}

TEST(QwenLoopGuardTest, SilentFinishReasonPrefersReasoningSuppression) {
    Request req{};
    req.suppressed_token_count = 4;

    FinalizeDecodeSilentFinishReason(&req);

    EXPECT_EQ(req.decode_silent_finish_reason, DecodeSilentFinishReason::ReasoningSuppressedOnly);
}

TEST(QwenLoopGuardTest, SilentFinishReasonFallsBackToUtf8PendingOnly) {
    Request req{};
    req.utf8_pending = "\xF0\x9F";

    FinalizeDecodeSilentFinishReason(&req);

    EXPECT_EQ(req.decode_silent_finish_reason, DecodeSilentFinishReason::Utf8PendingOnly);
}
