#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "densecore.h"
#include "runtime/engine_internal.h"

namespace {

// No worker is started: inspect exactly what each public entry point admits,
// without sampling/execution masking differences in legacy preparation policy.
struct ActiveRequestGuard {
    EngineState& state;
    Request request{};
    explicit ActiveRequestGuard(EngineState& state) : state(state) { state.active_requests.push_back(&request); }
    ~ActiveRequestGuard() { state.active_requests.clear(); }
};

class RequestSubmissionTest : public ::testing::Test {
protected:
    EngineState state;

    Request* Take(int id) {
        EXPECT_GT(id, 0) << DenseCoreGetLastError();
        auto* request = state.pending_requests.Pop();
        if (request) {
            EXPECT_EQ(request->id, id);
            state.RemovePendingRequest(id);
        }
        return request;
    }
};

TEST_F(RequestSubmissionTest, LegacyIdsDoNotAcquireSamplingPreparation) {
    const int ids[] = {17, 23, 42};
    auto* request = Take(SubmitRequestIds(&state, ids, 3, 11, nullptr, nullptr));
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->tokens, (std::vector<int>{17, 23, 42}));
    EXPECT_EQ(request->token_history, request->tokens);
    EXPECT_TRUE(request->prompt.empty());
    EXPECT_FALSE(request->token_id_submit_used);
    EXPECT_TRUE(request->parity_debug_submit_api.empty());
    EXPECT_EQ(request->max_tokens, 11);
    state.request_pool.Release(request);
}

TEST_F(RequestSubmissionTest, SamplingIdsPreserveCallerOptionsAndEntryPointIdentity) {
    const int ids[] = {17, 23, 42};
    const char* stops[] = {"END", nullptr};
    auto* request = Take(SubmitRequestIdsWithSamplingEx(&state, ids, 3, 19, "adapter", 0.25f, 0.8f, 13, 1.2f, stops, 0,
                                                        nullptr, nullptr));
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->tokens, (std::vector<int>{17, 23, 42}));
    EXPECT_TRUE(request->token_id_submit_used);
    EXPECT_EQ(request->parity_debug_submit_api, "SubmitRequestIdsWithSamplingConstraintsEx");
    EXPECT_EQ(request->lora_name, "adapter");
    EXPECT_FLOAT_EQ(request->sampling_params.temperature, 0.25f);
    EXPECT_FLOAT_EQ(request->sampling_params.top_p, 0.8f);
    EXPECT_EQ(request->sampling_params.top_k, 13);
    EXPECT_EQ(request->stop_sequences, (std::vector<std::string>{"END"}));
    state.request_pool.Release(request);
}

TEST_F(RequestSubmissionTest, LegacyFormatIdsKeepDefaultLoraAndJsonMode) {
    state.default_lora_adapter = "default-adapter";
    const int ids[] = {17, 23};
    auto* request = Take(SubmitRequestIdsWithFormat(&state, ids, 2, 9, 1, nullptr, nullptr));
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->tokens, (std::vector<int>{17, 23}));
    EXPECT_TRUE(request->json_mode);
    EXPECT_EQ(request->lora_name, "default-adapter");
    EXPECT_FALSE(request->token_id_submit_used);
    state.request_pool.Release(request);
}

TEST_F(RequestSubmissionTest, IdEntryPointsRejectDrainingWithoutCallbacksOrPendingStorage) {
    state.status.store(EngineStatus::DRAINING);
    int callbacks = 0;
    const auto callback = [](const char*, int, void* data) { ++*static_cast<int*>(data); };
    const int ids[] = {17};
    EXPECT_EQ(SubmitRequestIds(&state, ids, 1, 9, callback, &callbacks), DENSECORE_STATUS_ENGINE_STOPPED);
    EXPECT_EQ(SubmitRequestIdsWithFormat(&state, ids, 1, 9, 0, callback, &callbacks), DENSECORE_STATUS_ENGINE_STOPPED);
    EXPECT_EQ(SubmitRequestIdsWithSampling(&state, ids, 1, 9, 0.25f, 0.8f, 13, 1.2f, nullptr, 0, callback, &callbacks),
              DENSECORE_STATUS_ENGINE_STOPPED);
    EXPECT_EQ(callbacks, 0);
    EXPECT_TRUE(state.pending_requests.Empty());
}

TEST_F(RequestSubmissionTest, ConcurrentSubmitCopiesInputsAndAssignsUniqueIds) {
    constexpr int kSubmitters = 8;
    std::vector<int> request_ids(kSubmitters);
    std::vector<std::thread> submitters;
    for (int i = 0; i < kSubmitters; ++i) {
        submitters.emplace_back([&, i] {
            int ids[] = {i, 23, 42};
            request_ids[i] =
                SubmitRequestIdsWithSampling(&state, ids, 3, 9, 0.25f, 0.8f, 13, 1.2f, nullptr, 0, nullptr, nullptr);
            ids[0] = -1;
        });
    }
    for (auto& submitter : submitters) submitter.join();
    EXPECT_EQ(std::set<int>(request_ids.begin(), request_ids.end()).size(), kSubmitters);
    std::set<int> input_values;
    for (int i = 0; i < kSubmitters; ++i) {
        auto* request = state.pending_requests.Pop();
        ASSERT_NE(request, nullptr);
        ASSERT_EQ(request->tokens.size(), 3u);
        input_values.insert(request->tokens[0]);
        state.RemovePendingRequest(request->id);
        state.request_pool.Release(request);
    }
    EXPECT_EQ(input_values, (std::set<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(RequestSubmissionTest, ExclusiveActivationWaitsForRetirementNotTerminalDelivery) {
    ActiveRequestGuard active(state);
    active.request.id = 101;
    active.request.finished = true;
    active.request.generation_terminal_dispatched.store(true);
    const int tokens[] = {17};
    const int id = SubmitRequestIds(&state, tokens, 1, 4, nullptr, nullptr);
    ASSERT_GT(id, 0);
    EXPECT_EQ(state.PopReadyPendingRequest(true), nullptr);
    EXPECT_TRUE(state.IsPendingRequest(id));
    // Only the worker's retirement/removal releases the execution slot.
    state.active_requests.clear();
    auto* request = state.PopReadyPendingRequest(true);
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->id, id);
    state.RemovePendingRequest(id);
    state.request_pool.Release(request);
}

TEST_F(RequestSubmissionTest, CancellationBypassesExclusiveGateWithoutReorderingSurvivors) {
    ActiveRequestGuard active(state);
    const int tokens[] = {17};
    const int first = SubmitRequestIds(&state, tokens, 1, 4, nullptr, nullptr);
    const int canceled = SubmitRequestIds(&state, tokens, 1, 4, nullptr, nullptr);
    const int last = SubmitRequestIds(&state, tokens, 1, 4, nullptr, nullptr);
    state.RecordPendingCancellation(canceled);
    auto* request = state.PopReadyPendingRequest(true);
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->id, canceled);
    EXPECT_TRUE(request->cancelled.load());
    state.RemovePendingRequest(canceled);
    state.request_pool.Release(request);
    EXPECT_EQ(state.PopReadyPendingRequest(true), nullptr);
    state.active_requests.clear();
    for (int id : {first, last}) {
        request = state.PopReadyPendingRequest(true);
        ASSERT_NE(request, nullptr);
        EXPECT_EQ(request->id, id);
        state.RemovePendingRequest(id);
        state.request_pool.Release(request);
    }
}

TEST_F(RequestSubmissionTest, NonexclusiveActivationPreservesConcurrentAdmission) {
    ActiveRequestGuard active(state);
    const int tokens[] = {17};
    const int id = SubmitRequestIds(&state, tokens, 1, 4, nullptr, nullptr);
    auto* request = state.PopReadyPendingRequest(false);
    EXPECT_NE(request, nullptr);
    state.active_requests.clear();
    if (request) {
        EXPECT_EQ(request->id, id);
        state.RemovePendingRequest(id);
        state.request_pool.Release(request);
    }
}

TEST_F(RequestSubmissionTest, FailedPublicationRollsBackPendingIdentityWithoutCallback) {
    state.fail_pending_publication_for_test.store(true);
    int callbacks = 0;
    const auto callback = [](const char*, int, void* data) { ++*static_cast<int*>(data); };
    const int ids[] = {17};
    EXPECT_EQ(SubmitRequestIds(&state, ids, 1, 4, callback, &callbacks), DENSECORE_STATUS_OUT_OF_MEMORY);
    EXPECT_TRUE(state.pending_requests.Empty());
    EXPECT_TRUE(state.pending_request_ids.empty());
    EXPECT_EQ(callbacks, 0);
    // The same pooled object remains usable after the failed ownership transfer.
    auto* request = Take(SubmitRequestIds(&state, ids, 1, 4, callback, &callbacks));
    ASSERT_NE(request, nullptr);
    state.request_pool.Release(request);
}

TEST(PendingCancellationQueueTest, PredicateFailurePreservesBookkeepingAndFifo) {
    densecore::ShardedPriorityQueue<int> queue;
    int first = 1, second = 2, third = 3;
    queue.Push(&first, "standard");
    queue.Push(&second, "standard");
    queue.Push(&third, "batch");
    EXPECT_THROW(queue.PopMatching([&](int* item) {
        if (item == &second) throw std::bad_alloc();
        return false;
    }),
                 std::bad_alloc);
    EXPECT_EQ(queue.Size(), 3u);
    EXPECT_EQ(queue.Pop(), &first);
    EXPECT_EQ(queue.Pop(), &second);
    EXPECT_EQ(queue.Pop(), &third);
    EXPECT_TRUE(queue.Empty());
}

// Hold a wait predicate open while shutdown changes its completion condition.
// This recreates the check-to-wait window without production test hooks.
void CheckShutdownWakeup(bool callback_wait) {
    EngineState state;
    auto& wait_mutex = callback_wait ? state.result_mu : state.cv_mu;
    auto& wake = callback_wait ? state.result_cv : state.queue_cv;
    std::mutex gate_mutex;
    std::condition_variable gate;
    bool release_predicate = false;
    std::promise<void> checked, stopped;
    auto checked_future = checked.get_future();
    auto stopped_future = stopped.get_future();
    auto finished = [&] {
        return callback_wait ? state.result_producers_done.load(std::memory_order_acquire)
                             : state.status.load() != EngineStatus::RUNNING;
    };
    std::thread waiter([&] {
        std::unique_lock<std::mutex> lock(wait_mutex);
        bool announced = false;
        wake.wait(lock, [&] {
            const bool done = finished();
            if (!done && !announced) {
                announced = true;
                checked.set_value();
                std::unique_lock<std::mutex> gate_lock(gate_mutex);
                gate.wait(gate_lock, [&] { return release_predicate; });
            }
            return done;
        });
    });
    if (callback_wait)
        state.callback_thread = std::move(waiter);
    else
        state.worker_thread = std::move(waiter);
    checked_future.wait();
    std::thread shutdown([&] {
        state.Shutdown();
        stopped.set_value();
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!finished() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    // With the old unlocked publication, allow its notification to finish while
    // the waiter still holds its stale false predicate value.
    if (finished()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_predicate = true;
    }
    gate.notify_one();
    const bool shutdown_woke_waiter = stopped_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    // Recover the deliberately recreated lost wakeup so a failing test cleans up.
    while (stopped_future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) wake.notify_all();
    shutdown.join();
    EXPECT_TRUE(shutdown_woke_waiter);
}

TEST(EngineShutdownWakeupTest, WorkerStatusTransitionCannotBeLostBeforeWait) {
    CheckShutdownWakeup(false);
}

TEST(EngineShutdownWakeupTest, ProducerCompletionCannotBeLostBeforeCallbackWait) {
    CheckShutdownWakeup(true);
}

}  // namespace
