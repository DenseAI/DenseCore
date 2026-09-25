#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "densecore.h"
#include "runtime/engine_internal.h"

namespace densecore {
namespace testing {
void EnablePauseBeforePendingDequeueForTest(bool enabled);
bool WaitForPauseBeforePendingDequeueForTest(int timeout_ms);
void ReleasePauseBeforePendingDequeueForTest();
void EnablePauseAfterPendingPublishForTest(bool enabled);
bool WaitForPauseAfterPendingPublishForTest(int timeout_ms);
void ReleasePauseAfterPendingPublishForTest();
void EnablePauseAfterSchedulerAdmissionForTest(bool enabled);
bool WaitForPauseAfterSchedulerAdmissionForTest(int timeout_ms);
void ReleasePauseAfterSchedulerAdmissionForTest();
}  // namespace testing
}  // namespace densecore

namespace {

using namespace std::chrono_literals;

struct CallbackState {
    std::mutex mu;
    std::condition_variable cv;
    DenseCoreHandle engine = nullptr;
    int request_id = -1;
    bool cancel_on_first_non_final = false;
    bool cancel_issued = false;
    int callback_count = 0;
    int final_count = 0;
    bool completed = false;
    std::thread::id final_thread_id;
    std::vector<std::string> final_texts;
    std::vector<std::string> tokens;

    bool WaitForFinal(std::chrono::milliseconds timeout = 10s) {
        std::unique_lock<std::mutex> lock(mu);
        return cv.wait_for(lock, timeout, [&] { return completed; });
    }
};

struct EmbeddingCallbackState {
    std::mutex mu;
    std::condition_variable cv;
    int callback_count = 0;
    int terminal_size = 0;
    std::vector<float> embedding;

    bool Wait(std::chrono::milliseconds timeout = 10s) {
        std::unique_lock<std::mutex> lock(mu);
        return cv.wait_for(lock, timeout, [&] { return callback_count > 0; });
    }
};

void EmbeddingLifecycleCallback(const float* embedding, int size, void* user_data) {
    auto* state = static_cast<EmbeddingCallbackState*>(user_data);
    ASSERT_NE(state, nullptr);
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->callback_count++;
        state->terminal_size = size;
        if (embedding && size > 0) {
            state->embedding.assign(embedding, embedding + size);
        }
    }
    state->cv.notify_all();
}

void LifecycleCallbackEx(const char* token, int len, int /*token_id*/, int is_final, void* user_data) {
    auto* state = static_cast<CallbackState*>(user_data);
    ASSERT_NE(state, nullptr);

    bool should_cancel = false;
    std::string token_text = token && len > 0 ? std::string(token, token + len) : "";
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->callback_count++;
        if (is_final) {
            state->final_count++;
            state->completed = true;
            state->final_thread_id = std::this_thread::get_id();
            state->final_texts.push_back(token_text);
        } else {
            state->tokens.push_back(token_text);
            if (state->cancel_on_first_non_final && !state->cancel_issued) {
                state->cancel_issued = true;
                should_cancel = true;
            }
        }
    }

    if (should_cancel) {
        EXPECT_EQ(CancelRequest(state->engine, state->request_id), 0) << DenseCoreGetLastError();
    }
    state->cv.notify_all();
}

bool WaitForEngineShutdownStarted(EngineState* state, std::chrono::milliseconds timeout = 5s) {
    // The paused worker keeps the engine alive until the test releases it.
    // DRAINING can be transient when there are no active requests, so observe
    // admission closure rather than requiring the poll to catch that instant.
    const auto shutdown_started = [&] {
        const auto status = state->status.load(std::memory_order_acquire);
        return status == EngineStatus::DRAINING || status == EngineStatus::STOPPED;
    };
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (shutdown_started()) {
            return true;
        }
        std::this_thread::yield();
    }
    return shutdown_started();
}

class RequestLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = InitEngine("mock", nullptr, 2);
        ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();
        densecore::testing::EnablePauseBeforePendingDequeueForTest(false);
        densecore::testing::EnablePauseAfterPendingPublishForTest(false);
        densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(false);
    }

    void TearDown() override {
        densecore::testing::ReleasePauseBeforePendingDequeueForTest();
        densecore::testing::EnablePauseBeforePendingDequeueForTest(false);
        densecore::testing::ReleasePauseAfterPendingPublishForTest();
        densecore::testing::EnablePauseAfterPendingPublishForTest(false);
        densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
        densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(false);
        if (engine_) {
            FreeEngine(engine_);
            engine_ = nullptr;
        }
    }

    DenseCoreHandle engine_ = nullptr;
};

struct EngineCloseGuard {
    DenseCoreHandle* handle = nullptr;

    ~EngineCloseGuard() {
        if (!handle || !*handle) {
            return;
        }
        densecore::testing::ReleasePauseBeforePendingDequeueForTest();
        densecore::testing::ReleasePauseAfterPendingPublishForTest();
        densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
        FreeEngine(*handle);
        *handle = nullptr;
    }
};

TEST_F(RequestLifecycleTest, RejectsSubmitAfterDrainingBegins) {
    auto* state = static_cast<EngineState*>(engine_);
    state->status.store(EngineStatus::DRAINING, std::memory_order_release);

    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int rc = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "hello", 8, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);

    EXPECT_EQ(rc, DENSECORE_STATUS_ENGINE_STOPPED);
    EXPECT_EQ(state->pending_requests.Size(), 0u);
    EXPECT_EQ(callback_state.callback_count, 0);

    state->status.store(EngineStatus::RUNNING, std::memory_order_release);
}

TEST(PendingRequestBookkeepingTest, RapidPublishAndDequeueCannotLeaveStalePendingId) {
    EngineState state;
    Request* request = state.request_pool.Acquire();
    ASSERT_NE(request, nullptr);
    request->id = 7001;
    request->tier = "standard";

    densecore::testing::EnablePauseAfterPendingPublishForTest(true);
    bool admitted = false;
    std::thread submitter([&] { admitted = state.TryAdmitPendingRequest(request); });
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterPendingPublishForTest(5000));

    Request* dequeued = state.pending_requests.Pop();
    ASSERT_EQ(dequeued, request);
    state.RemovePendingRequest(dequeued->id);
    densecore::testing::ReleasePauseAfterPendingPublishForTest();
    submitter.join();

    EXPECT_TRUE(admitted);
    EXPECT_FALSE(state.IsPendingRequest(request->id));
    state.request_pool.Release(request);
    state.Shutdown();
}

TEST_F(RequestLifecycleTest, SuccessfulEmbeddingCompletionEmitsExactlyOnce) {
    EmbeddingCallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitEmbeddingRequest(engine_, "embedding-success", EmbeddingLifecycleCallback,
                                                  &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(callback_state.Wait(15s));
    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_GT(callback_state.terminal_size, 0);
    EXPECT_EQ(callback_state.embedding.size(), static_cast<size_t>(callback_state.terminal_size));
}

TEST_F(RequestLifecycleTest, PendingEmbeddingCancellationEmitsTerminalFailureOnce) {
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    EmbeddingCallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitEmbeddingRequest(engine_, "embedding-cancel", EmbeddingLifecycleCallback,
                                                  &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    EXPECT_EQ(CancelRequest(engine_, request_id), 0) << DenseCoreGetLastError();
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();

    ASSERT_TRUE(callback_state.Wait());
    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_LT(callback_state.terminal_size, 0);
    EXPECT_TRUE(callback_state.embedding.empty());
}

TEST_F(RequestLifecycleTest, ShutdownFinalizesPendingEmbeddingOnce) {
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    EmbeddingCallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitEmbeddingRequest(engine_, "embedding-pending-shutdown", EmbeddingLifecycleCallback,
                                                  &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));

    DenseCoreHandle handle = engine_;
    engine_ = nullptr;
    std::thread shutdown_thread([handle] { FreeEngine(handle); });
    ASSERT_TRUE(WaitForEngineShutdownStarted(static_cast<EngineState*>(handle)));
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(callback_state.Wait());
    shutdown_thread.join();

    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_EQ(callback_state.terminal_size, DENSECORE_STATUS_ENGINE_STOPPED);
}

TEST_F(RequestLifecycleTest, ShutdownFinalizesActiveEmbeddingOnce) {
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    EmbeddingCallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitEmbeddingRequest(engine_, "embedding-active-shutdown", EmbeddingLifecycleCallback,
                                                  &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));

    DenseCoreHandle handle = engine_;
    engine_ = nullptr;
    std::thread shutdown_thread([handle] { FreeEngine(handle); });
    ASSERT_TRUE(WaitForEngineShutdownStarted(static_cast<EngineState*>(handle)));
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    ASSERT_TRUE(callback_state.Wait(15s));
    shutdown_thread.join();

    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_TRUE(callback_state.terminal_size > 0 || callback_state.terminal_size == DENSECORE_STATUS_ENGINE_STOPPED);
}

TEST(EmbeddingFinalizerTest, DuplicateTerminalAttemptsDispatchOnlyOnce) {
    EngineState state;
    Request* request = state.request_pool.Acquire();
    ASSERT_NE(request, nullptr);
    request->id = 7002;
    request->is_embedding = true;
    EmbeddingCallbackState callback_state;
    request->embedding_callback = EmbeddingLifecycleCallback;
    request->user_data = &callback_state;

    EXPECT_TRUE(FinalizeEmbeddingRequestOnce(&state, request, {}, DENSECORE_STATUS_ENGINE_STOPPED, true));
    EXPECT_FALSE(FinalizeEmbeddingRequestOnce(&state, request, {1.0f}, DENSECORE_STATUS_OK, true));
    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_EQ(callback_state.terminal_size, DENSECORE_STATUS_ENGINE_STOPPED);

    state.request_pool.Release(request);
    state.Shutdown();
}

TEST_F(RequestLifecycleTest, PendingCancellationEmitsExactlyOneTerminalCallback) {
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);

    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "pending-cancel", 16, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));

    EXPECT_EQ(CancelRequest(engine_, request_id), 0) << DenseCoreGetLastError();
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();

    ASSERT_TRUE(callback_state.WaitForFinal());
    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "Error: request canceled");
}

TEST_F(RequestLifecycleTest, ShutdownFinalizesPendingRequestExactlyOnce) {
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);

    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "pending-shutdown", 16, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));

    DenseCoreHandle handle = engine_;
    engine_ = nullptr;
    std::thread shutdown_thread([handle]() { FreeEngine(handle); });
    const bool shutdown_started = WaitForEngineShutdownStarted(static_cast<EngineState*>(handle));
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();

    const bool completed = callback_state.WaitForFinal();
    shutdown_thread.join();

    EXPECT_TRUE(shutdown_started);
    ASSERT_TRUE(completed);
    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "Error: Engine shutdown");
}

TEST_F(RequestLifecycleTest, ActiveCancellationEmitsExactlyOneTerminalCallback) {
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    callback_state.engine = engine_;

    callback_state.request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "active-cancel", 64, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);
    ASSERT_GT(callback_state.request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    EXPECT_EQ(CancelRequest(engine_, callback_state.request_id), 0) << DenseCoreGetLastError();
    callback_state.cancel_issued = true;
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();

    ASSERT_TRUE(callback_state.WaitForFinal(15s));
    EXPECT_TRUE(callback_state.cancel_issued);
    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "Error: request canceled");
}

TEST_F(RequestLifecycleTest, SuccessfulCompletionEmitsExactlyOneTerminalCallback) {
    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};

    const int request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "success", 8, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    (void)request_id;

    ASSERT_TRUE(callback_state.WaitForFinal(15s));
    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "");
}

TEST_F(RequestLifecycleTest, PendingCancelShutdownRaceStillEmitsOneTerminalCallback) {
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);

    CallbackState callback_state;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int request_id = SubmitRequestWithSamplingConstraintsCallbackEx(
        engine_, "cancel-shutdown-race", 16, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
        LifecycleCallbackEx, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));

    EXPECT_EQ(CancelRequest(engine_, request_id), 0) << DenseCoreGetLastError();
    DenseCoreHandle handle = engine_;
    engine_ = nullptr;
    std::thread shutdown_thread([handle]() { FreeEngine(handle); });
    const bool shutdown_started = WaitForEngineShutdownStarted(static_cast<EngineState*>(handle));
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();

    const bool completed = callback_state.WaitForFinal();
    shutdown_thread.join();

    EXPECT_TRUE(shutdown_started);
    ASSERT_TRUE(completed);
    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_TRUE(callback_state.final_texts.front() == "Error: request canceled" ||
                callback_state.final_texts.front() == "Error: Engine shutdown");
}

TEST(RequestLifecycleCallbackLoopTest, CallbackLoopWaitsForProducerCompletionAfterStopped) {
    EngineState state;
    state.status.store(EngineStatus::STOPPED, std::memory_order_release);
    state.result_producers_done.store(false, std::memory_order_release);

    CallbackState callback_state;
    std::thread callback_thread([&state]() { CallbackLoop(&state); });

    PushResultEvent(&state, 77, "late-final", -1, true, true, nullptr, LifecycleCallbackEx, nullptr, &callback_state);
    state.result_producers_done.store(true, std::memory_order_release);
    state.result_cv.notify_all();

    ASSERT_TRUE(callback_state.WaitForFinal());
    callback_thread.join();

    EXPECT_EQ(callback_state.final_count, 1);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "late-final");
}

TEST(RequestLifecycleCallbackModeTest, ShutdownPendingRequestUsesConfiguredDirectCallbackMode) {
    EngineState state;
    state.fast_path_config.worker.callback_mode =
        densecore::llm::config::WorkerRuntimeConfig::CallbackMode::Direct;

    CallbackState callback_state;
    Request* request = state.request_pool.Acquire();
    ASSERT_NE(request, nullptr);
    request->id = 91;
    request->callback_ex = LifecycleCallbackEx;
    request->user_data = &callback_state;
    ASSERT_TRUE(state.TryAdmitPendingRequest(request));

    const std::thread::id shutdown_thread_id = std::this_thread::get_id();
    state.Shutdown();

    ASSERT_TRUE(callback_state.WaitForFinal());
    EXPECT_EQ(callback_state.final_count, 1);
    EXPECT_EQ(callback_state.final_thread_id, shutdown_thread_id);
    ASSERT_EQ(callback_state.final_texts.size(), 1u);
    EXPECT_EQ(callback_state.final_texts.front(), "Error: Engine shutdown");
}

}  // namespace
