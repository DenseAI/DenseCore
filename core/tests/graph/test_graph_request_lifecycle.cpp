#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "densecore.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/models/graph_registry.h"

namespace {

using namespace std::chrono_literals;

struct GraphCallbackState {
    std::mutex mu;
    std::condition_variable cv;
    int callback_count = 0;
    int num_outputs = 0;
    float first_value = 0.0f;
    bool completed = false;
};

void CaptureGraphResult(const DenseCoreTensorOutput* outputs, int num_outputs, void* user_data) {
    auto* state = static_cast<GraphCallbackState*>(user_data);
    if (!state) return;

    {
        std::lock_guard<std::mutex> lock(state->mu);
        ++state->callback_count;
        state->num_outputs = num_outputs;
        if (outputs && num_outputs > 0 && outputs[0].data) {
            state->first_value = static_cast<const float*>(outputs[0].data)[0];
        }
        state->completed = true;
    }
    state->cv.notify_all();
}

bool WaitForGraphCallback(GraphCallbackState* state, std::chrono::milliseconds timeout = 5s) {
    std::unique_lock<std::mutex> lock(state->mu);
    return state->cv.wait_for(lock, timeout, [&] { return state->completed; });
}

struct BlockingBuilderState {
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false;
    bool proceed = false;
};

class BlockingPassthroughBuilder final : public densecore::GraphBuilder {
public:
    explicit BlockingPassthroughBuilder(std::shared_ptr<BlockingBuilderState> state) : state_(std::move(state)) {}

    std::unique_ptr<densecore::OperationGraph> Build(const std::vector<densecore::Tensor>& inputs,
                                                     const std::string&) override {
        {
            std::unique_lock<std::mutex> lock(state_->mu);
            state_->entered = true;
            state_->cv.notify_all();
            state_->cv.wait(lock, [&] { return state_->proceed; });
        }

        if (inputs.empty()) return nullptr;
        auto graph = std::make_unique<densecore::OperationGraph>();
        const size_t output = graph->RegisterTensor(inputs.front());
        graph->MarkOutput(output);
        return graph;
    }

private:
    std::shared_ptr<BlockingBuilderState> state_;
};

class GraphRequestLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = InitEngine("mock", nullptr, 2);
        ASSERT_NE(engine_, nullptr) << DenseCoreGetLastError();
    }

    void TearDown() override {
        if (engine_) {
            FreeEngine(engine_);
            engine_ = nullptr;
        }
    }

    DenseCoreHandle engine_ = nullptr;
};

TEST_F(GraphRequestLifecycleTest, AsyncSubmissionOwnsInputBytesUntilCallback) {
    auto builder_state = std::make_shared<BlockingBuilderState>();
    densecore::GraphRegistry::Instance().Register("graph_lifecycle_owned_input", [builder_state] {
        return std::make_unique<BlockingPassthroughBuilder>(builder_state);
    });

    std::vector<float> caller_data = {1.25f, 2.5f, 3.75f, 5.0f};
    DenseCoreTensorInput input{};
    input.name = "caller_owned_input";
    input.data = caller_data.data();
    input.ndim = 2;
    input.shape[0] = 2;
    input.shape[1] = 2;
    input.dtype = DENSECORE_DTYPE_F32;

    GraphCallbackState callback_state;
    const int request_id =
        SubmitGraphRequest(engine_, &input, 1, "graph_lifecycle_owned_input", CaptureGraphResult, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();

    bool builder_entered = false;
    {
        std::unique_lock<std::mutex> lock(builder_state->mu);
        builder_entered = builder_state->cv.wait_for(lock, 5s, [&] { return builder_state->entered; });
    }

    if (builder_entered) {
        caller_data.assign(caller_data.size(), 99.0f);
    }
    {
        std::lock_guard<std::mutex> lock(builder_state->mu);
        builder_state->proceed = true;
    }
    builder_state->cv.notify_all();
    ASSERT_TRUE(builder_entered);

    ASSERT_TRUE(WaitForGraphCallback(&callback_state));
    std::lock_guard<std::mutex> lock(callback_state.mu);
    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_EQ(callback_state.num_outputs, 1);
    EXPECT_FLOAT_EQ(callback_state.first_value, 1.25f);
}

TEST_F(GraphRequestLifecycleTest, AsyncFailureCompletesExactlyOnceWithErrorStatus) {
    float input_data = 1.0f;
    DenseCoreTensorInput input{};
    input.name = "input";
    input.data = &input_data;
    input.ndim = 1;
    input.shape[0] = 1;
    input.dtype = DENSECORE_DTYPE_F32;

    GraphCallbackState callback_state;
    const int request_id =
        SubmitGraphRequest(engine_, &input, 1, "graph_lifecycle_missing_builder", CaptureGraphResult, &callback_state);
    ASSERT_GT(request_id, 0) << DenseCoreGetLastError();
    ASSERT_TRUE(WaitForGraphCallback(&callback_state));

    std::this_thread::sleep_for(50ms);
    std::lock_guard<std::mutex> lock(callback_state.mu);
    EXPECT_EQ(callback_state.callback_count, 1);
    EXPECT_EQ(callback_state.num_outputs, DENSECORE_STATUS_UNSUPPORTED_OPERATION);
}

TEST_F(GraphRequestLifecycleTest, SyncFailureReturnsInsteadOfWaitingForever) {
    struct SyncCallState {
        float input_data = 1.0f;
        float output_data = 0.0f;
        DenseCoreTensorInput input{};
        DenseCoreTensorOutput output{};
        std::promise<int> result;

        SyncCallState() {
            input.name = "input";
            input.data = &input_data;
            input.ndim = 1;
            input.shape[0] = 1;
            input.dtype = DENSECORE_DTYPE_F32;
            output.data = &output_data;
        }
    };

    auto state = std::make_shared<SyncCallState>();
    auto result = state->result.get_future();
    std::thread execution([engine = engine_, state] {
        state->result.set_value(
            ExecuteGraphSync(engine, &state->input, 1, "graph_lifecycle_missing_sync_builder", &state->output, 1));
    });

    if (result.wait_for(5s) != std::future_status::ready) {
        execution.detach();
        FAIL() << "ExecuteGraphSync did not return after graph execution failure";
    }
    EXPECT_EQ(result.get(), DENSECORE_STATUS_UNSUPPORTED_OPERATION);
    execution.join();
}

TEST_F(GraphRequestLifecycleTest, RejectsInputShapeByteSizeOverflowBeforeEnqueue) {
    float input_data = 1.0f;
    DenseCoreTensorInput input{};
    input.name = "overflow";
    input.data = &input_data;
    input.ndim = 2;
    input.shape[0] = std::numeric_limits<int64_t>::max();
    input.shape[1] = 3;
    input.dtype = DENSECORE_DTYPE_F32;

    GraphCallbackState callback_state;
    EXPECT_EQ(SubmitGraphRequest(engine_, &input, 1, "unused", CaptureGraphResult, &callback_state),
              DENSECORE_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(callback_state.callback_count, 0);
}

}  // namespace
