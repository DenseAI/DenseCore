#include <gtest/gtest.h>

#include <atomic>
#include <climits>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "densecore.h"
#include "runtime/engine_internal.h"
#include "runtime/request_builder.h"

namespace densecore {
namespace testing {
void EnablePauseBeforePendingDequeueForTest(bool enabled);
bool WaitForPauseBeforePendingDequeueForTest(int timeout_ms);
void ReleasePauseBeforePendingDequeueForTest();
void EnablePauseAfterPendingPublishForTest(bool enabled);
bool WaitForPauseAfterPendingPublishForTest(int timeout_ms);
void ReleasePauseAfterPendingPublishForTest();
void EnablePauseAfterSchedulerAdmissionForTest(bool enabled);
void SetSchedulerAdmissionPauseCountForTest(int count);
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

TEST_F(RequestLifecycleTest, WorkerExceptionTerminatesActiveAndPendingRequestsAndRejectsAdmission) {
    auto* state = static_cast<EngineState*>(engine_);
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    CallbackState active, pending;
    EmbeddingCallbackState embedding;
    EngineCloseGuard close_before_callback_state{&engine_};
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_, "active", 100, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0, LifecycleCallbackEx, &active), 0);
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_, "pending", 100, nullptr, 0.7f, 0.9f, 20, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0, LifecycleCallbackEx, &pending), 0);
    ASSERT_GT(SubmitEmbeddingRequest(engine_, "pending embedding", EmbeddingLifecycleCallback, &embedding), 0);
    state->fail_worker_for_test.store(true);
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    ASSERT_TRUE(active.WaitForFinal());
    ASSERT_TRUE(pending.WaitForFinal());
    ASSERT_TRUE(embedding.Wait());
    EXPECT_EQ(active.final_count, 1);
    EXPECT_EQ(pending.final_count, 1);
    EXPECT_EQ(embedding.callback_count, 1);
    EXPECT_LT(embedding.terminal_size, 0);
    EXPECT_NE(active.final_texts.front().find("worker failed"), std::string::npos);
    EXPECT_EQ(state->status.load(), EngineStatus::STOPPED);
    EXPECT_EQ(SubmitEmbeddingRequest(engine_, "rejected", EmbeddingLifecycleCallback, &embedding),
              DENSECORE_STATUS_ENGINE_STOPPED);
    FreeEngine(engine_);
    engine_ = nullptr;
    EXPECT_EQ(active.final_count, 1);
    EXPECT_EQ(pending.final_count, 1);
    EXPECT_EQ(embedding.callback_count, 1);
}

TEST_F(RequestLifecycleTest, WorkerExceptionWhileDequeuedTerminatesEmbeddingBeforeSchedulerAdmission) {
    auto* state = static_cast<EngineState*>(engine_);
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    EmbeddingCallbackState embedding;
    EngineCloseGuard close_before_callback_state{&engine_};
    ASSERT_GT(SubmitEmbeddingRequest(engine_, "dequeued embedding", EmbeddingLifecycleCallback, &embedding), 0);
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    state->fail_worker_for_test.store(true);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(embedding.Wait());
    EXPECT_EQ(embedding.callback_count, 1);
    EXPECT_EQ(embedding.terminal_size, DENSECORE_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(state->pending_requests.Size(), 0u);
}

TEST(RequestTerminalPublicationTest, GenerationQueueFailureFallsBackOnceAndWakesCaller) {
    EngineState state;
    Request req;
    CallbackState callback;
    req.Reset();
    req.callback_ex = LifecycleCallbackEx;
    req.user_data = &callback;
    state.fail_result_publication_for_test.store(true);
    EXPECT_TRUE(FinalizeGenerationRequestOnce(&state, &req, "failure", -1, true, false));
    EXPECT_TRUE(callback.WaitForFinal(10ms));
    EXPECT_EQ(callback.final_count, 1);
    EXPECT_TRUE(state.result_queue.empty());
    EXPECT_FALSE(FinalizeGenerationRequestOnce(&state, &req, "duplicate", -1, true, false));
    EXPECT_EQ(callback.final_count, 1);
}

TEST(RequestTerminalPublicationTest, EmbeddingQueueFailurePreservesSuccessPayloadAndCompletesOnce) {
    EngineState state;
    Request req;
    EmbeddingCallbackState callback;
    req.Reset();
    req.is_embedding = true;
    req.embedding_callback = EmbeddingLifecycleCallback;
    req.user_data = &callback;
    state.fail_result_publication_for_test.store(true);
    EXPECT_TRUE(FinalizeEmbeddingRequestOnce(&state, &req, {1.0f, 2.0f}, DENSECORE_STATUS_OK, false));
    EXPECT_TRUE(callback.Wait(10ms));
    EXPECT_EQ(callback.callback_count, 1);
    EXPECT_EQ(callback.embedding, (std::vector<float>{1.0f, 2.0f}));
    EXPECT_TRUE(state.result_queue.empty());
    EXPECT_FALSE(FinalizeEmbeddingRequestOnce(&state, &req, {}, DENSECORE_STATUS_INTERNAL_ERROR, false));
    EXPECT_EQ(callback.callback_count, 1);
}

TEST_F(RequestLifecycleTest, AutoEmbeddingRejectsUnsupportedMetadataBeforeAdmissionAndKeepsExplicitLast) {
    auto* state = static_cast<EngineState*>(engine_);
    {
        std::lock_guard<std::mutex> lock(state->construction_mu);
        auto* entry = state->GetDefaultModel();
        ASSERT_TRUE(entry && entry->model);
        entry->model->has_pooling_metadata = true;
        entry->model->gguf_pooling_type = 4; // GGUF rank pooling is not a sentence vector.
    }
    EmbeddingCallbackState rejected;
    EmbeddingCallbackState explicit_last;
    EngineCloseGuard close_before_callback_state{&engine_};
    EXPECT_EQ(SubmitEmbeddingRequestEx(engine_, "unsupported-auto", -1, 1, EmbeddingLifecycleCallback, &rejected),
              DENSECORE_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(DenseCoreGetLastError()).find("pooling metadata"), std::string::npos);
    EXPECT_EQ(rejected.callback_count, 0);
    ASSERT_GT(SubmitEmbeddingRequestEx(engine_, "explicit-last", 2, 1, EmbeddingLifecycleCallback, &explicit_last), 0);
    ASSERT_TRUE(explicit_last.Wait(15s));
    EXPECT_EQ(explicit_last.callback_count, 1);
    EXPECT_GT(explicit_last.terminal_size, 0);
}

TEST_F(RequestLifecycleTest, RecurrentAdmissionCapsEightSlotsAndCancellationBypassesTheRawQueue) {
    constexpr int capacity = 8, count = 64;
    auto* state = static_cast<EngineState*>(engine_);
    std::array<CallbackState, count> callbacks;
    std::array<int, count> ids;
    EngineCloseGuard close_before_callbacks{&engine_};
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    for (int i = 0; i < count; ++i) {
        ids[i] = SubmitRequestWithSamplingConstraintsCallbackEx(
            engine_, "queued-state-budget", 32, nullptr, 0.0f, 1.0f, 1, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
            LifecycleCallbackEx, &callbacks[i]);
        ASSERT_GT(ids[i], 0) << DenseCoreGetLastError();
    }
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    // The dequeue pause precedes recurrent allocation. Use a tiny synthetic SSM
    // to prove actual state allocation without any external model artifact.
    auto* entry = state->GetDefaultModel();
    ASSERT_NE(entry, nullptr);
    entry->kv_config.max_num_seqs = capacity;
    state->fast_path_config.worker.prefix_cache_reuse_disabled = true;
    auto* model = entry->model.get();
    model->arch_flags.is_hybrid_ssm = true;
    model->ssm_inner_size = 2;
    model->ssm_group_count = 1;
    model->ssm_state_size = 2;
    model->ssm_time_step_rank = 1;
    model->ssm_conv_kernel = 2;
    model->ssm_layer_states.resize(1);
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    densecore::testing::SetSchedulerAdmissionPauseCountForTest(capacity);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        ASSERT_EQ(state->active_requests.size(), capacity);
        for (const auto* active : state->active_requests) {
            ASSERT_EQ(active->ssm_runtime_states.size(), 1u);
            EXPECT_EQ(active->ssm_runtime_states[0].conv_state.size(), 6u);
            EXPECT_EQ(active->ssm_runtime_states[0].ssm_state.size(), 4u);
        }
    }
    EXPECT_EQ(state->pending_requests.Size(), count - capacity);
    Request* canceled_raw = nullptr;
    int raw_count = 0;
    EXPECT_EQ(state->pending_requests.PopMatching([&](Request* pending) {
        ++raw_count;
        EXPECT_TRUE(pending->ssm_runtime_states.empty());
        if (pending->id == ids.back()) canceled_raw = pending;
        return false;
    }), nullptr);
    EXPECT_EQ(raw_count, count - capacity);
    ASSERT_NE(canceled_raw, nullptr);
    // Select a cancelled request at the tail even while all eight slots remain
    // occupied. Its ownership is still covered by fatal dequeue cleanup.
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    ASSERT_EQ(CancelRequest(engine_, ids.back()), DENSECORE_STATUS_OK);
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    EXPECT_TRUE(canceled_raw->cancelled.load());
    EXPECT_TRUE(canceled_raw->ssm_runtime_states.empty());
    EXPECT_EQ(state->pending_requests.Size(), count - capacity - 1);
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        EXPECT_EQ(state->active_requests.size(), capacity);
    }
    state->fail_worker_for_test.store(true);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    for (auto& callback : callbacks) {
        ASSERT_TRUE(callback.WaitForFinal());
        EXPECT_EQ(callback.final_count, 1);
    }
    FreeEngine(engine_);
    engine_ = nullptr;
    for (const auto& callback : callbacks) EXPECT_EQ(callback.final_count, 1);
}

TEST_F(RequestLifecycleTest, FullBudgetPreservesNormalCompletionAndPendingUserCancellation) {
    constexpr int count = 24;
    auto* state = static_cast<EngineState*>(engine_);
    std::array<CallbackState, count> callbacks;
    std::array<int, count> ids;
    EngineCloseGuard close_before_callbacks{&engine_};
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    for (int i = 0; i < count; ++i) {
        ids[i] = SubmitRequestWithSamplingConstraintsCallbackEx(
            engine_, "bounded-normal-completion", 2, nullptr, 0.0f, 1.0f, 1, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
            LifecycleCallbackEx, &callbacks[i]);
        ASSERT_GT(ids[i], 0) << DenseCoreGetLastError();
    }
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    state->GetDefaultModel()->kv_config.max_num_seqs = 1;
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    {
        std::lock_guard<std::mutex> lock(state->active_mu);
        EXPECT_EQ(state->active_requests.size(), 1u);
    }
    EXPECT_EQ(state->pending_requests.Size(), count - 1);
    ASSERT_EQ(CancelRequest(engine_, ids.back()), DENSECORE_STATUS_OK);
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    for (auto& callback : callbacks) {
        ASSERT_TRUE(callback.WaitForFinal());
        EXPECT_EQ(callback.final_count, 1);
    }
    ASSERT_EQ(callbacks.back().final_texts.size(), 1u);
    EXPECT_EQ(callbacks.back().final_texts[0], "Error: request canceled");
    for (int i = 0; i < count - 1; ++i) {
        ASSERT_EQ(callbacks[i].final_texts.size(), 1u);
        EXPECT_TRUE(callbacks[i].final_texts[0].empty());
    }
}

TEST_F(RequestLifecycleTest, FullBudgetShutdownFinalizesActiveAndRawPendingRequestsOnce) {
    constexpr int count = 24;
    auto* state = static_cast<EngineState*>(engine_);
    std::array<CallbackState, count> callbacks;
    EngineCloseGuard close_before_callbacks{&engine_};
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    for (auto& callback : callbacks) {
        ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(
            engine_, "bounded-shutdown", 2, nullptr, 0.0f, 1.0f, 1, 1.0f, nullptr, 0, nullptr, 0, 0, nullptr, 0,
            LifecycleCallbackEx, &callback), 0);
    }
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    state->GetDefaultModel()->kv_config.max_num_seqs = 1;
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    EXPECT_EQ(state->pending_requests.Size(), count - 1);
    std::thread shutdown([&] { BeginEngineShutdown(engine_); });
    const bool began = WaitForEngineShutdownStarted(state);
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    shutdown.join();
    ASSERT_TRUE(began);
    for (auto& callback : callbacks) {
        ASSERT_TRUE(callback.WaitForFinal());
        EXPECT_EQ(callback.final_count, 1);
    }
    EXPECT_TRUE(state->pending_requests.Empty());
    EXPECT_EQ(state->metrics.active_requests.load(), 0);
    FreeEngine(engine_);
    engine_ = nullptr;
    for (const auto& callback : callbacks) EXPECT_EQ(callback.final_count, 1);
}

TEST_F(RequestLifecycleTest, PendingPayloadBudgetRejectsBeforeCopyForTextIdsAndEmbedding) {
    using namespace densecore::runtime;
    CallbackState generation;
    EmbeddingCallbackState embedding;
    EngineCloseGuard close_before_callback_state{&engine_};
    const int token = 1;
    EXPECT_EQ(SubmitRequestIds(engine_,&token,INT_MAX,8,nullptr,nullptr),DENSECORE_STATUS_OUT_OF_MEMORY);
    DenseCoreTensorInput graph_input{};
    EXPECT_EQ(SubmitGraphRequest(engine_,&graph_input,INT_MAX,"huge-input-count",nullptr,nullptr),DENSECORE_STATUS_OUT_OF_MEMORY);
    const std::string oversized(ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission)/256+1,'x');
    EXPECT_EQ(SubmitRequest(engine_,oversized.c_str(),8,nullptr,nullptr,nullptr),DENSECORE_STATUS_OUT_OF_MEMORY);
    auto blocker = ResourceAllocationLease::Acquire(AllocationCategory::Admission,
                          ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission));
    ASSERT_NE(blocker,nullptr);
    EXPECT_EQ(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"budget-reject",8,nullptr,0,1,1,1,
                      nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&generation),DENSECORE_STATUS_OUT_OF_MEMORY);
    EXPECT_NE(std::string(DenseCoreGetLastError()).find("admission"),std::string::npos);
    // This deliberately tiny buffer must never be read using the hostile count.
    EXPECT_EQ(SubmitRequestIds(engine_,&token,INT_MAX,8,nullptr,nullptr),DENSECORE_STATUS_OUT_OF_MEMORY);
    EXPECT_EQ(SubmitEmbeddingRequest(engine_,"budget-reject",EmbeddingLifecycleCallback,&embedding),DENSECORE_STATUS_OUT_OF_MEMORY);
    const char* first_prompt="batch-reject";
    EXPECT_EQ(SubmitBatchEmbeddingRequest(engine_,&first_prompt,INT_MAX,-1,1,EmbeddingLifecycleCallback,&embedding),DENSECORE_STATUS_OUT_OF_MEMORY);
    EXPECT_EQ(generation.callback_count,0);
    EXPECT_EQ(embedding.callback_count,0);
    EXPECT_TRUE(static_cast<EngineState*>(engine_)->pending_requests.Empty());
    blocker.reset();
    EXPECT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"budget-recovers",2,nullptr,0,1,1,1,
                      nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&generation),0);
    ASSERT_TRUE(generation.WaitForFinal());
    EXPECT_EQ(generation.final_count,1);
    ASSERT_GT(SubmitEmbeddingRequest(engine_,"embedding-recovers",EmbeddingLifecycleCallback,&embedding),0);
    ASSERT_TRUE(embedding.Wait());
    EXPECT_EQ(embedding.callback_count,1);
    EXPECT_GT(embedding.terminal_size,0);
}

TEST_F(RequestLifecycleTest, RequestPoolRetirementFreesPayloadButKeepsMetadataCharged) {
    using namespace densecore::runtime;
    auto* state=static_cast<EngineState*>(engine_);
    const auto before=ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission);
    auto* req=state->request_pool.Acquire();
    req->Reset();
    req->prompt.assign(1024*1024,'x');
    req->tokens.resize(1024*1024);
    req->grammar.accumulated.assign(1024*1024,'x');
    req->admission_payload_lease=ResourceAllocationLease::Acquire(AllocationCategory::Admission,8*1024*1024);
    ASSERT_NE(req->admission_payload_lease,nullptr);
    state->request_pool.Release(req);
    EXPECT_EQ(ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission),before-sizeof(Request)-128);
    auto* reused=state->request_pool.Acquire();
    EXPECT_EQ(reused,req);
    EXPECT_LT(reused->prompt.capacity(),1024u);
    EXPECT_EQ(reused->tokens.capacity(),0u);
    EXPECT_LT(reused->grammar.accumulated.capacity(),1024u);
    EXPECT_FALSE(reused->admission_payload_lease);
    state->request_pool.Release(reused);
}

TEST_F(RequestLifecycleTest, AcceptedPendingPayloadReservationsReleaseAfterCancellationAndShutdown) {
    using namespace densecore::runtime;
    std::array<CallbackState,8> callbacks;
    EmbeddingCallbackState embedding;
    EngineCloseGuard close_before_callbacks{&engine_};
    auto* state=static_cast<EngineState*>(engine_);
    const auto before=ResourceLedgerUsageSnapshot().admission_bytes;
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    int last=-1;
    for (auto& callback:callbacks) {
        last=SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"pending-payload",2,nullptr,0,1,1,1,
                      nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&callback);
        ASSERT_GT(last,0) << DenseCoreGetLastError();
    }
    const int embedding_id=SubmitEmbeddingRequest(engine_,"pending-embedding",EmbeddingLifecycleCallback,&embedding);
    ASSERT_GT(embedding_id,0) << DenseCoreGetLastError();
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    EXPECT_GT(ResourceLedgerUsageSnapshot().admission_bytes,before+9*sizeof(Request));
    auto blocker=ResourceAllocationLease::Acquire(AllocationCategory::Admission,
                         ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission));
    ASSERT_NE(blocker,nullptr);
    EXPECT_EQ(SubmitRequest(engine_,"refused",2,nullptr,nullptr,nullptr),DENSECORE_STATUS_OUT_OF_MEMORY);
    ASSERT_EQ(CancelRequest(engine_,last),DENSECORE_STATUS_OK);
    ASSERT_EQ(CancelRequest(engine_,embedding_id),DENSECORE_STATUS_OK);
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    for (auto& callback:callbacks) {ASSERT_TRUE(callback.WaitForFinal());EXPECT_EQ(callback.final_count,1);}
    ASSERT_TRUE(embedding.Wait());
    EXPECT_EQ(embedding.callback_count,1);
    EXPECT_LT(embedding.terminal_size,0);
    blocker.reset();
    FreeEngine(engine_);
    engine_=nullptr;
    EXPECT_EQ(ResourceLedgerUsageSnapshot().admission_bytes,before);
}

TEST(RequestAdmissionBudgetTest, QueuedResultRetainsPayloadReservationAfterRequestRetirement) {
    using namespace densecore::runtime;
    RuntimeMemoryPlan plan;
    plan.available_bytes=plan.allocator_bytes=256*1024;
    ResourcePlanLease deterministic_plan(plan);
    EngineState state;
    Request req;
    req.Reset();
    const auto before=ResourceLedgerUsageSnapshot().admission_bytes;
    req.admission_payload_lease=ResourceAllocationLease::Acquire(AllocationCategory::Admission,64*1024);
    ASSERT_NE(req.admission_payload_lease,nullptr);
    PushResultEvent(&state,1,"owned-output",1,true,false,nullptr,nullptr,nullptr,nullptr,nullptr,1,
                    DENSECORE_GENERATION_STOP,req.admission_payload_lease);
    req.ReleaseExecutionState();
    EXPECT_EQ(ResourceLedgerUsageSnapshot().admission_bytes,before+64*1024);
    auto blocker=ResourceAllocationLease::Acquire(AllocationCategory::Admission,
                          ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission));
    ASSERT_NE(blocker,nullptr);
    EXPECT_THROW(AcquireBoundedTextRequest(&state,"must-wait-for-output",0,1),densecore::OutOfMemoryException);
    {
        // Simulate a callback thread that has not consumed its owned result yet.
        std::lock_guard<std::mutex> lock(state.result_mu);
        ASSERT_EQ(state.result_queue.size(),1u);
        state.result_queue.clear();
    }
    auto* recovered=AcquireBoundedTextRequest(&state,"output-drained",0,1);
    ASSERT_NE(recovered,nullptr);
    state.request_pool.Release(recovered);
    blocker.reset();
    EXPECT_EQ(ResourceLedgerUsageSnapshot().admission_bytes,before+sizeof(Request)+128);
}

TEST(RequestAdmissionBudgetTest, StandaloneBudgetPreservesUnknownExhaustedAndUnlimitedStates) {
    using namespace densecore::runtime;
    ResourceBudget budget;
    EXPECT_EQ(StandaloneAdmissionCapacity(budget),0u);
    budget.memory={ResourceState::Finite,0};
    EXPECT_EQ(StandaloneAdmissionCapacity(budget),0u);
    budget.memory={ResourceState::Finite,64*1024*1024};
    EXPECT_EQ(StandaloneAdmissionCapacity(budget),2*1024*1024u);
    budget.memory={ResourceState::Unlimited,0};
    EXPECT_EQ(StandaloneAdmissionCapacity(budget),128*1024*1024u);
}

TEST(RequestAdmissionBudgetTest, TerminalFallbackCallbacksKeepTheirPayloadReservation) {
    using namespace densecore::runtime;
    RuntimeMemoryPlan plan;
    plan.available_bytes=plan.allocator_bytes=64*1024;
    ResourcePlanLease deterministic_plan(plan);
    EngineState state;
    const auto before=ResourceLedgerUsageSnapshot().admission_bytes;
    struct Probe {uint64_t expected; int calls=0; bool charged=false;};
    Probe generation{before+1024},embedding{before+1024};
    auto generation_callback=[](const char*,int finished,void* user) {
        auto& probe=*static_cast<Probe*>(user);
        ++probe.calls;
        probe.charged=finished && ResourceLedgerUsageSnapshot().admission_bytes>=probe.expected;
    };
    auto embedding_callback=[](const float*,int size,void* user) {
        auto& probe=*static_cast<Probe*>(user);
        ++probe.calls;
        probe.charged=size==1 && ResourceLedgerUsageSnapshot().admission_bytes>=probe.expected;
    };
    state.fail_result_publication_for_test.store(true);
    PushResultEvent(&state,1,"terminal",1,true,false,generation_callback,nullptr,nullptr,&generation,nullptr,1,
                    DENSECORE_GENERATION_STOP,ResourceAllocationLease::Acquire(AllocationCategory::Admission,1024));
    EXPECT_EQ(generation.calls,1);
    EXPECT_TRUE(generation.charged);
    EXPECT_EQ(ResourceLedgerUsageSnapshot().admission_bytes,before);
    state.fail_result_publication_for_test.store(true);
    PushEmbeddingResultEvent(&state,2,{1.0f},DENSECORE_STATUS_OK,embedding_callback,&embedding,
                             ResourceAllocationLease::Acquire(AllocationCategory::Admission,1024));
    EXPECT_EQ(embedding.calls,1);
    EXPECT_TRUE(embedding.charged);
    EXPECT_EQ(ResourceLedgerUsageSnapshot().admission_bytes,before);
}

TEST_F(RequestLifecycleTest, PartiallyAdmittedEmbeddingBatchTerminatesEveryChild) {
    using namespace densecore::runtime;
    struct BatchProbe {std::mutex mu; std::condition_variable cv; std::vector<int> sizes;};
    BatchProbe batch;
    EmbeddingCallbackState initial;
    EngineCloseGuard close_before_callbacks{&engine_};
    auto callback=[](const float*,int size,void* user) {
        auto& probe=*static_cast<BatchProbe*>(user);
        {std::lock_guard<std::mutex> lock(probe.mu);probe.sizes.push_back(size);}
        probe.cv.notify_all();
    };
    densecore::testing::EnablePauseBeforePendingDequeueForTest(true);
    const auto before=ResourceLedgerUsageSnapshot().admission_bytes;
    ASSERT_GT(SubmitEmbeddingRequest(engine_,"same-text",EmbeddingLifecycleCallback,&initial),0);
    ASSERT_TRUE(densecore::testing::WaitForPauseBeforePendingDequeueForTest(5000));
    const auto one_request=ResourceLedgerUsageSnapshot().admission_bytes-before;
    const auto remaining=ResourceAllocationLease::AvailableBytes(AllocationCategory::Admission);
    ASSERT_GT(remaining,one_request);
    auto blocker=ResourceAllocationLease::Acquire(AllocationCategory::Admission,remaining-one_request);
    ASSERT_NE(blocker,nullptr);
    const char* prompts[]={"same-text","same-text"};
    ASSERT_GT(SubmitBatchEmbeddingRequest(engine_,prompts,2,-1,1,callback,&batch),0);
    {
        std::lock_guard<std::mutex> lock(batch.mu);
        ASSERT_EQ(batch.sizes.size(),1u);
        EXPECT_EQ(batch.sizes[0],DENSECORE_STATUS_OUT_OF_MEMORY);
    }
    densecore::testing::ReleasePauseBeforePendingDequeueForTest();
    ASSERT_TRUE(initial.Wait());
    {
        std::unique_lock<std::mutex> lock(batch.mu);
        ASSERT_TRUE(batch.cv.wait_for(lock,10s,[&]{return batch.sizes.size()==2;}));
        EXPECT_GT(batch.sizes[1],0);
    }
    blocker.reset();
    FreeEngine(engine_);
    engine_=nullptr;
    EXPECT_EQ(batch.sizes.size(),2u);
}

TEST(EngineShutdownPolicyTest, ConfiguredDeadlineExtendsLegacyWindowWithoutRestartingIt) {
    const auto now=EngineShutdownClock::time_point(std::chrono::seconds(100));
    const auto legacy=EngineShutdownDeadline(5000,now);
    const auto configured=EngineShutdownDeadline(30000,now);
    EXPECT_TRUE(EngineShutdownGraceExpired(legacy,now+std::chrono::seconds(6)));
    EXPECT_FALSE(EngineShutdownGraceExpired(configured,now+std::chrono::seconds(6)));
    EXPECT_TRUE(EngineShutdownGraceExpired(configured,now+std::chrono::seconds(30)));
    EXPECT_EQ(EngineShutdownDeadline(0,now),now);
    EXPECT_THROW(EngineShutdownDeadline(-1,now),std::invalid_argument);
    EXPECT_THROW(EngineShutdownDeadline(INT64_MAX,now),std::invalid_argument);
    EXPECT_THROW(EngineShutdownDeadline(1,EngineShutdownClock::time_point::max()),std::invalid_argument);
}

TEST_F(RequestLifecycleTest, InvalidShutdownOptionsKeepAdmissionOpen) {
    DenseCoreShutdownOptions options{sizeof(DenseCoreShutdownOptions),2,30000};
    EXPECT_EQ(DenseCoreShutdownEngineWithOptions(engine_,&options),DENSECORE_STATUS_INVALID_ARGUMENT);
    options.version=1;
    options.struct_size=sizeof(DenseCoreShutdownOptions)-1;
    EXPECT_EQ(DenseCoreShutdownEngineWithOptions(engine_,&options),DENSECORE_STATUS_INVALID_ARGUMENT);
    options.struct_size=sizeof(DenseCoreShutdownOptions);
    options.grace_timeout_ms=-1;
    EXPECT_EQ(DenseCoreShutdownEngineWithOptions(engine_,&options),DENSECORE_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(static_cast<EngineState*>(engine_)->status.load(),EngineStatus::RUNNING);
    CallbackState callback;
    EngineCloseGuard close_before_callback{&engine_};
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"valid-after-invalid-shutdown",2,nullptr,0,1,1,1,
                         nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&callback),0);
    ASSERT_TRUE(callback.WaitForFinal());
    EXPECT_EQ(callback.final_count,1);
}

TEST_F(RequestLifecycleTest, ExpiredConfiguredGraceCancelsHeldRequestExactlyOnce) {
    auto* state=static_cast<EngineState*>(engine_);
    CallbackState callback;
    EngineCloseGuard close_before_callback{&engine_};
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"held-through-deadline",32,nullptr,0,1,1,1,
                         nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&callback),0);
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    DenseCoreShutdownOptions options{sizeof(DenseCoreShutdownOptions),1,0};
    int shutdown_status=DENSECORE_STATUS_INTERNAL_ERROR;
    std::thread shutdown([&]{shutdown_status=DenseCoreShutdownEngineWithOptions(engine_,&options);});
    const auto deadline=EngineShutdownClock::now()+5s;
    while (state->status.load()!=EngineStatus::STOPPED && EngineShutdownClock::now()<deadline) std::this_thread::yield();
    const bool stopped=state->status.load()==EngineStatus::STOPPED;
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    shutdown.join();
    ASSERT_TRUE(stopped);
    EXPECT_EQ(shutdown_status,DENSECORE_STATUS_OK);
    ASSERT_TRUE(callback.WaitForFinal());
    EXPECT_EQ(callback.final_count,1);
    ASSERT_EQ(callback.final_texts.size(),1u);
    EXPECT_FALSE(callback.final_texts[0].empty());
    EXPECT_EQ(DenseCoreShutdownEngineWithOptions(engine_,&options),DENSECORE_STATUS_OK);
    FreeEngine(engine_);engine_=nullptr;
    EXPECT_EQ(callback.final_count,1);
}

TEST_F(RequestLifecycleTest, ConfiguredGraceLetsActiveRequestCompleteAndRejectsNewAdmission) {
    auto* state=static_cast<EngineState*>(engine_);
    CallbackState callback;
    EngineCloseGuard close_before_callback{&engine_};
    densecore::testing::EnablePauseAfterSchedulerAdmissionForTest(true);
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"finish-within-grace",2,nullptr,0,1,1,1,
                         nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&callback),0);
    ASSERT_TRUE(densecore::testing::WaitForPauseAfterSchedulerAdmissionForTest(5000));
    DenseCoreShutdownOptions options{sizeof(DenseCoreShutdownOptions),1,30000};
    int shutdown_status=DENSECORE_STATUS_INTERNAL_ERROR;
    std::thread shutdown([&]{shutdown_status=DenseCoreShutdownEngineWithOptions(engine_,&options);});
    const bool draining=WaitForEngineShutdownStarted(state);
    EXPECT_EQ(SubmitRequest(engine_,"new-during-drain",2,nullptr,nullptr,nullptr),DENSECORE_STATUS_ENGINE_STOPPED);
    densecore::testing::ReleasePauseAfterSchedulerAdmissionForTest();
    shutdown.join();
    ASSERT_TRUE(draining);
    EXPECT_EQ(shutdown_status,DENSECORE_STATUS_OK);
    ASSERT_TRUE(callback.WaitForFinal());
    EXPECT_EQ(callback.final_count,1);
    ASSERT_EQ(callback.final_texts.size(),1u);
    EXPECT_TRUE(callback.final_texts[0].empty());
}

TEST_F(RequestLifecycleTest, FinalTeardownWaitsForBorrowedModelPreparationBeforeDestroyingOwners) {
    auto* state=static_cast<EngineState*>(engine_);
    CallbackState rejected,warmup;
    EngineCloseGuard close_before_callback{&engine_};
    // Worker startup itself queries model policy under construction_mu. Finish
    // one request before deliberately holding that mutex across shutdown.
    ASSERT_GT(SubmitRequestWithSamplingConstraintsCallbackEx(engine_,"startup-ready",1,nullptr,0,1,1,1,
                         nullptr,0,nullptr,0,0,nullptr,0,LifecycleCallbackEx,&warmup),0);
    ASSERT_TRUE(warmup.WaitForFinal());
    std::unique_lock<std::mutex> construction(state->construction_mu);
    auto* entry=state->GetDefaultModel();
    ASSERT_NE(entry,nullptr);
    ASSERT_NE(entry->model,nullptr);
    auto* borrowed_model=entry->model.get();
    auto* prepared=AcquireBoundedTextRequest(state,"borrowed-preparation",0,2);
    RequestGuard prepared_guard(state,prepared);
    prepared->max_tokens=2;
    prepared->callback_ex=LifecycleCallbackEx;
    prepared->user_data=&rejected;
    std::thread shutdown([&]{BeginEngineShutdown(engine_);});
    const auto deadline=EngineShutdownClock::now()+5s;
    while (!state->construction_teardown_waiting_for_test.load(std::memory_order_acquire) &&
           EngineShutdownClock::now()<deadline) std::this_thread::yield();
    const bool blocked=state->construction_teardown_waiting_for_test.load(std::memory_order_acquire);
    if (blocked) {
        // The signal comes after both joins and a contended lock attempt. The
        // borrowed owner is therefore protected at the exact old race boundary.
        EXPECT_EQ(state->status.load(),EngineStatus::STOPPED);
        EXPECT_TRUE(state->result_producers_done.load());
        EXPECT_EQ(state->GetDefaultModel(),entry);
        EXPECT_EQ(entry->model.get(),borrowed_model);
        prepared->tokens=Tokenizer::Tokenize(borrowed_model,"borrowed-preparation",false);
    }
    construction.unlock();
    const int published=PublishPreparedRequest(state,prepared,prepared_guard,"preparation finished during shutdown");
    shutdown.join();
    EXPECT_TRUE(blocked);
    EXPECT_EQ(published,DENSECORE_STATUS_ENGINE_STOPPED);
    EXPECT_EQ(rejected.callback_count,0);
    EXPECT_TRUE(state->pending_requests.Empty());
    EXPECT_EQ(state->GetDefaultModel(),nullptr);
}

TEST_F(RequestLifecycleTest, StoppedEngineRejectsModelPreparationAndKeepsReadQueriesSafe) {
    BeginEngineShutdown(engine_);
    EXPECT_EQ(LoadModel(engine_,"late-model","mock",2),DENSECORE_STATUS_ENGINE_STOPPED);
    DenseCoreRuntimeOptimizationState runtime{};
    EXPECT_EQ(DenseCoreGetRuntimeOptimizationState(engine_,&runtime),DENSECORE_STATUS_OK);
    EXPECT_EQ(GetDetailedMetrics(engine_).kv_cache_total_blocks,0);
    DenseCoreTensor tensor{};
    EXPECT_EQ(DenseCoreSetWeight(engine_,"missing-weight",&tensor),DENSECORE_STATUS_MODEL_LOAD_FAILED);
}
