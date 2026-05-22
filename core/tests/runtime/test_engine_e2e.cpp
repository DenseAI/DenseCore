/**
 * @file test_engine_e2e.cpp
 * @brief End-to-End smoke tests for the DenseCore inference pipeline
 *
 * Tests the full lifecycle: InitEngine -> SubmitRequest -> TokenCallback -> FreeEngine
 * Uses mock model (DENSECORE_TEST_BUILD) to avoid requiring a real GGUF file.
 *
 * These tests verify that the entire pipeline is wired correctly:
 *   1. Engine initialization (model loading, KV cache, scheduler, worker thread)
 *   2. Request submission and tokenization
 *   3. Graph building and inference execution
 *   4. Token sampling and callback delivery
 *   5. Engine shutdown and resource cleanup
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "densecore.h"

namespace {

// ============================================================================
// Callback helpers
// ============================================================================

struct CallbackState {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::string> tokens;
    std::atomic<bool> finished{false};
    std::atomic<int> callback_count{0};

    void WaitForCompletion(int timeout_ms = 30000) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return finished.load(); });
    }

    std::string FullOutput() const {
        std::string result;
        for (const auto& t : tokens) {
            result += t;
        }
        return result;
    }
};

struct TokenResultState {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int> token_ids;
    std::vector<std::string> token_text;
    std::atomic<bool> finished{false};

    void WaitForCompletion(int timeout_ms = 30000) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return finished.load(); });
    }
};

void TokenCallbackFn(const char* token, int is_final, void* user_data) {
    auto* state = static_cast<CallbackState*>(user_data);
    if (!state) return;

    {
        std::lock_guard<std::mutex> lock(state->mu);
        if (token) {
            state->tokens.emplace_back(token);
        }
        state->callback_count.fetch_add(1, std::memory_order_relaxed);
        if (is_final) {
            state->finished.store(true, std::memory_order_release);
        }
    }
    state->cv.notify_one();
}

void TokenResultCallbackFn(const TokenResult* result, void* user_data) {
    auto* state = static_cast<TokenResultState*>(user_data);
    if (!state || !result) return;

    {
        std::lock_guard<std::mutex> lock(state->mu);
        if (!result->is_finished && result->token_id >= 0) {
            state->token_ids.push_back(result->token_id);
            state->token_text.emplace_back(result->text ? result->text : "");
        }
        if (result->is_finished) {
            state->finished.store(true, std::memory_order_release);
        }
    }
    state->cv.notify_one();
}

// ============================================================================
// Test fixture
// ============================================================================

class EngineE2ETest : public ::testing::Test {
protected:
    void SetUp() override { engine_ = nullptr; }

    void TearDown() override {
        if (engine_) {
            FreeEngine(engine_);
            engine_ = nullptr;
        }
    }

    DenseCoreHandle engine_ = nullptr;
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(EngineE2ETest, InitAndFree_MockModel) {
    // Verify engine can initialize with mock model and shut down cleanly
    engine_ = InitEngine("mock", nullptr, 0);
    ASSERT_NE(engine_, nullptr) << "InitEngine failed: " << DenseCoreGetLastError();

    // Verify we can query basic info
    const char* version = GetLibraryVersionString();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST_F(EngineE2ETest, InitEngineEx_MockModel) {
    // Test extended init with explicit thread/NUMA config
    engine_ = InitEngineEx("mock", nullptr, 2, -1, 0);
    ASSERT_NE(engine_, nullptr) << "InitEngineEx failed: " << DenseCoreGetLastError();
}

TEST_F(EngineE2ETest, RequestSnapshotCAPIABIAndOwnedFree) {
    static_assert(std::is_standard_layout<DenseCoreRequestSnapshot>::value,
                  "DenseCoreRequestSnapshot must remain CGO-compatible");
    static_assert(std::is_standard_layout<DenseCoreRuntimeOptimizationState>::value,
                  "DenseCoreRuntimeOptimizationState must remain CGO-compatible");
    static_assert(offsetof(DenseCoreRequestSnapshot, token_ids) > offsetof(DenseCoreRequestSnapshot, rendered_prompt),
                  "DenseCoreRequestSnapshot field order changed unexpectedly");
    static_assert(offsetof(DenseCoreRequestSnapshot, caller_owns_buffers) >
                      offsetof(DenseCoreRequestSnapshot, prompt_family),
                  "DenseCoreRequestSnapshot ownership marker must stay after owned pointer fields");
    static_assert(offsetof(DenseCoreRuntimeOptimizationState, active_thread_policy_label) >
                      offsetof(DenseCoreRuntimeOptimizationState, moe_dequant_cache_mb),
                  "Runtime optimization state label must stay after scalar fields");

    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << "InitEngine failed: " << DenseCoreGetLastError();

    DenseCoreRequestSnapshot snapshot{};
    int rc = DenseCoreBuildRequestSnapshot(engine_, "hello", 8, 0.7f, 0.9f, 40, 1.1f, 0, &snapshot);
    ASSERT_EQ(rc, DENSECORE_STATUS_OK) << DenseCoreGetLastError();
    EXPECT_NE(snapshot.rendered_prompt, nullptr);
    EXPECT_EQ(snapshot.caller_owns_buffers, 1);
    EXPECT_GE(snapshot.num_token_ids, 0);
    EXPECT_NE(snapshot.model_variant, nullptr);
    EXPECT_NE(snapshot.prompt_family, nullptr);
    DenseCoreFreeRequestSnapshot(&snapshot);
    EXPECT_EQ(snapshot.rendered_prompt, nullptr);
    EXPECT_EQ(snapshot.token_ids, nullptr);
    EXPECT_EQ(snapshot.num_token_ids, 0);
    EXPECT_EQ(snapshot.tokenizer_type, nullptr);
    EXPECT_EQ(snapshot.model_variant, nullptr);
    EXPECT_EQ(snapshot.prompt_family, nullptr);
    EXPECT_EQ(snapshot.caller_owns_buffers, 0);

    rc = DenseCoreBuildRenderedRequestSnapshot(engine_, "<rendered>hello</rendered>", 8, 0.7f, 0.9f, 40, 1.1f, 0,
                                               &snapshot);
    ASSERT_EQ(rc, DENSECORE_STATUS_OK) << DenseCoreGetLastError();
    EXPECT_NE(snapshot.rendered_prompt, nullptr);
    EXPECT_EQ(snapshot.template_applied, 0);
    EXPECT_EQ(snapshot.text_primed, 0);
    EXPECT_EQ(snapshot.token_primed, 0);
    EXPECT_EQ(snapshot.caller_owns_buffers, 1);
    DenseCoreFreeRequestSnapshot(&snapshot);
    EXPECT_EQ(snapshot.rendered_prompt, nullptr);
    EXPECT_EQ(snapshot.token_ids, nullptr);
    EXPECT_EQ(snapshot.caller_owns_buffers, 0);

    DenseCoreRuntimeOptimizationState runtime_state{};
    rc = DenseCoreGetRuntimeOptimizationState(engine_, &runtime_state);
    ASSERT_EQ(rc, DENSECORE_STATUS_OK) << DenseCoreGetLastError();
    EXPECT_EQ(runtime_state.struct_size, static_cast<int>(sizeof(DenseCoreRuntimeOptimizationState)));
    EXPECT_EQ(runtime_state.token_id_submit_supported, 1);
    EXPECT_GE(runtime_state.decode_graph_cache_max_batch, 1);
    EXPECT_GT(strlen(runtime_state.active_thread_policy_label), 0u);
}

TEST_F(EngineE2ETest, InitEngine_NullPath) {
    engine_ = InitEngine(nullptr, nullptr, 0);
    EXPECT_EQ(engine_, nullptr);
    // Should have set an error
    const char* err = DenseCoreGetLastError();
    EXPECT_NE(err, nullptr);
}

TEST_F(EngineE2ETest, InitEngine_EmptyPath) {
    engine_ = InitEngine("", nullptr, 0);
    EXPECT_EQ(engine_, nullptr);
}

TEST_F(EngineE2ETest, InitEngine_PathTraversal) {
    engine_ = InitEngine("../../etc/passwd", nullptr, 0);
    EXPECT_EQ(engine_, nullptr);
    const char* err = DenseCoreGetLastError();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err).find("path traversal"), std::string::npos);
}

TEST_F(EngineE2ETest, InitEngine_PathTraversalInMiddleSegment) {
    engine_ = InitEngine("core/../README.md", nullptr, 0);
    EXPECT_EQ(engine_, nullptr);
    const char* err = DenseCoreGetLastError();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err).find("path traversal"), std::string::npos);
}

TEST_F(EngineE2ETest, SubmitRequest_BasicGeneration) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    CallbackState state;
    int req_id = SubmitRequest(engine_, "Hello", 10, nullptr, TokenCallbackFn, &state);
    EXPECT_GE(req_id, 0) << "SubmitRequest failed: " << DenseCoreGetLastError();

    // Wait for generation to complete (mock model should finish quickly)
    state.WaitForCompletion(15000);

    EXPECT_TRUE(state.finished.load()) << "Generation did not complete within timeout";
    EXPECT_GT(state.callback_count.load(), 0) << "No callbacks received";
}

TEST_F(EngineE2ETest, SubmitRequest_WithSampling) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    CallbackState state;
    int req_id = SubmitRequestWithSampling(engine_, "Test prompt", 5,
                                           /* temperature */ 0.7f,
                                           /* top_p */ 0.9f,
                                           /* top_k */ 40,
                                           /* repetition_penalty */ 1.1f,
                                           /* stop_sequences */ nullptr,
                                           /* json_mode */ 0, TokenCallbackFn, &state);
    EXPECT_GE(req_id, 0) << "SubmitRequestWithSampling failed: " << DenseCoreGetLastError();

    state.WaitForCompletion(15000);
    EXPECT_TRUE(state.finished.load()) << "Generation with sampling did not complete";
    EXPECT_GT(state.callback_count.load(), 0);
}

TEST_F(EngineE2ETest, SubmitRequest_WithTokenResults) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    TokenResultState state;
    int req_id = SubmitRequestWithTokenResults(engine_, "Test prompt", 4,
                                               /*temperature=*/1.0f,
                                               /*top_p=*/1.0f,
                                               /*top_k=*/1,
                                               /*repetition_penalty=*/1.0f,
                                               TokenResultCallbackFn, &state);
    EXPECT_GE(req_id, 0) << DenseCoreGetLastError();

    state.WaitForCompletion(15000);
    EXPECT_TRUE(state.finished.load()) << "Structured token generation did not complete";
    EXPECT_FALSE(state.token_ids.empty()) << "Structured token callback emitted no token IDs";
}

TEST_F(EngineE2ETest, SubmitRequest_TokenIds) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    // Submit with pre-tokenized input
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    CallbackState state;
    int req_id = SubmitRequestIds(engine_, tokens.data(), static_cast<int>(tokens.size()), 5, TokenCallbackFn, &state);
    EXPECT_GE(req_id, 0) << "SubmitRequestIds failed: " << DenseCoreGetLastError();

    state.WaitForCompletion(15000);
    EXPECT_TRUE(state.finished.load()) << "Token ID generation did not complete";
}

TEST_F(EngineE2ETest, SubmitRequest_JsonMode) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    CallbackState state;
    int req_id = SubmitRequestWithFormat(engine_, "Generate JSON", 20, /* json_mode */ 1, TokenCallbackFn, &state);
    EXPECT_GE(req_id, 0) << DenseCoreGetLastError();

    state.WaitForCompletion(15000);
    EXPECT_TRUE(state.finished.load()) << "JSON mode generation did not complete";
}

TEST_F(EngineE2ETest, MultipleRequests_Sequential) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    for (int i = 0; i < 3; ++i) {
        CallbackState state;
        std::string prompt = "Request " + std::to_string(i);
        int req_id = SubmitRequest(engine_, prompt.c_str(), 5, nullptr, TokenCallbackFn, &state);
        EXPECT_GE(req_id, 0) << "Request " << i << " failed";

        state.WaitForCompletion(15000);
        EXPECT_TRUE(state.finished.load()) << "Request " << i << " did not complete";
    }
}

TEST_F(EngineE2ETest, MultipleRequests_Concurrent) {
    engine_ = InitEngine("mock", nullptr, 4);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    constexpr int NUM_REQUESTS = 5;
    std::vector<CallbackState> states(NUM_REQUESTS);
    std::vector<int> req_ids(NUM_REQUESTS);

    // Submit all at once (continuous batching)
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        std::string prompt = "Concurrent " + std::to_string(i);
        req_ids[i] = SubmitRequest(engine_, prompt.c_str(), 5, nullptr, TokenCallbackFn, &states[i]);
        EXPECT_GE(req_ids[i], 0) << "Concurrent request " << i << " submission failed";
    }

    // Wait for all
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        states[i].WaitForCompletion(30000);
        EXPECT_TRUE(states[i].finished.load()) << "Concurrent request " << i << " did not complete";
    }
}

TEST_F(EngineE2ETest, SubmitRequest_NullCallback) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    // Null callback should be rejected gracefully
    int req_id = SubmitRequest(engine_, "Hello", 10, nullptr, nullptr, nullptr);
    // Engine may reject or accept - either way it should not crash
    (void)req_id;
}

TEST_F(EngineE2ETest, SubmitRequest_EmptyPrompt) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    CallbackState state;
    int req_id = SubmitRequest(engine_, "", 10, nullptr, TokenCallbackFn, &state);
    // Empty prompt may be rejected or produce empty output - should not crash
    if (req_id >= 0) {
        state.WaitForCompletion(10000);
    }
}

TEST_F(EngineE2ETest, SubmitRequest_ZeroMaxTokens) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    CallbackState state;
    int req_id = SubmitRequest(engine_, "Hello", 0, nullptr, TokenCallbackFn, &state);
    // Zero max_tokens should complete immediately or be rejected
    if (req_id >= 0) {
        state.WaitForCompletion(10000);
    }
}

TEST_F(EngineE2ETest, EngineMetrics) {
    engine_ = InitEngine("mock", nullptr, 2);
    ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();

    // Submit a request first
    CallbackState state;
    SubmitRequest(engine_, "Metrics test", 5, nullptr, TokenCallbackFn, &state);
    state.WaitForCompletion(15000);

    // Query metrics
    DetailedMetrics metrics = GetDetailedMetrics(engine_);
    // Verify metrics struct has reasonable values after generation
    EXPECT_GE(metrics.completed_requests, 0);
}

}  // namespace
