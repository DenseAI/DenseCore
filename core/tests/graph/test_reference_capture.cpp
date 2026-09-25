#include "llm/graph/support_internal.h"
#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

TEST(GraphReferenceCapture, AssemblyScopeDoesNotLeakAcrossThreads) {
    std::vector<float> output;
    const auto previous = ExchangeAttentionCaptureForTest({&output, 7});
    EXPECT_TRUE(ShouldCaptureAttentionForTest(7));
    EXPECT_EQ(AttentionCaptureUserDataForTest(), &output);
    std::atomic<bool> isolated{false};
    std::thread other(
        [&] { isolated.store(!ShouldCaptureAttentionForTest(7) && AttentionCaptureUserDataForTest() == nullptr); });
    other.join();
    EXPECT_TRUE(isolated.load());
    ExchangeAttentionCaptureForTest(previous);
}

TEST(GraphReferenceCapture, SecondaryWorkerUsesCapturedUserdataAfterAssemblyScopeEnds) {
    auto* context = ggml_init({16384, nullptr, false});
    ASSERT_NE(context, nullptr);
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(context, ggml_free);
    auto* source = ggml_new_tensor_1d(context, GGML_TYPE_F32, 3);
    auto* destination = ggml_new_tensor_1d(context, GGML_TYPE_F32, 3);
    const float expected[] = {1.0f, -2.5f, 7.25f};
    std::copy(std::begin(expected), std::end(expected), static_cast<float*>(source->data));
    std::vector<float> output;
    const auto previous = ExchangeAttentionCaptureForTest({&output, 3});
    void* captured = AttentionCaptureUserDataForTest();
    ExchangeAttentionCaptureForTest(previous);
    std::thread worker([&] { cb_test_capture_attention_tensor(destination, source, 0, 1, captured); });
    worker.join();
    EXPECT_EQ(output, std::vector<float>(std::begin(expected), std::end(expected)));
    EXPECT_EQ(static_cast<float*>(destination->data)[1], -2.5f);
}
