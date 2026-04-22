/**
 * @file test_hal_abstraction.cpp
 * @brief Tests for HAL abstraction layer contracts
 *
 * Verifies:
 * - OperationGraph::Execute() polymorphism (no dynamic_cast needed)
 * - OpRegistry dependency injection for test isolation
 * - Stream/Event RAII behavior
 */

#include <gtest/gtest.h>

#include "densecore/hal/compute_backend.h"
#include "densecore/hal/device_interface.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"

namespace densecore {
namespace testing {

// =============================================================================
// Mock Graph for Testing Execute() Polymorphism
// =============================================================================

class TestGraph : public OperationGraph {
public:
    void Execute() const override { executed_ = true; }

    bool WasExecuted() const { return executed_; }
    void Reset() { executed_ = false; }

private:
    mutable bool executed_ = false;
};

// =============================================================================
// Mock Op for Testing OpRegistry
// =============================================================================

class MockOp : public DenseCoreOp {
public:
    explicit MockOp(int priority = 100) : priority_(priority) {}

    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                 const void* params) override {
        (void)inputs;
        (void)outputs;
        (void)params;
        execute_count_++;
    }

    bool Supports(DeviceType device) const override {
        (void)device;
        return true;
    }

    OpCapabilities GetCapabilities() const override {
        OpCapabilities caps;
        caps.priority = priority_;
        return caps;
    }

    int GetExecuteCount() const { return execute_count_; }

private:
    int priority_;
    int execute_count_ = 0;
};

// =============================================================================
// Mock Device/Stream/Event for RAII Testing
// =============================================================================

class MockStream : public Stream {
public:
    void Synchronize() override {}
    void RecordEvent(Event* /*event*/) override {}
    void WaitEvent(Event* /*event*/) override {}
};

class MockEvent : public Event {
public:
    void Synchronize() override {}
    float ElapsedTime(const Event* /*start*/, const Event* /*end*/) const override { return 0.0f; }
};

class MockDevice : public Device {
public:
    Allocator* GetAllocator() override { return nullptr; }

    Stream* CreateStream() override {
        streams_created_++;
        return new MockStream();
    }

    Event* CreateEvent(bool /*enable_timing*/) override {
        events_created_++;
        return new MockEvent();
    }

    void DestroyStream(Stream* stream) override {
        streams_destroyed_++;
        delete stream;
    }

    void DestroyEvent(Event* event) override {
        events_destroyed_++;
        delete event;
    }

    int GetStreamsCreated() const { return streams_created_; }
    int GetStreamsDestroyed() const { return streams_destroyed_; }
    int GetEventsCreated() const { return events_created_; }
    int GetEventsDestroyed() const { return events_destroyed_; }

private:
    int streams_created_ = 0;
    int streams_destroyed_ = 0;
    int events_created_ = 0;
    int events_destroyed_ = 0;
};

// =============================================================================
// Test: OperationGraph Execute() Polymorphism
// =============================================================================

TEST(HALAbstraction, GraphExecutePolymorphism) {
    TestGraph graph;
    EXPECT_FALSE(graph.WasExecuted());

    // Call Execute() through base class pointer (no dynamic_cast needed)
    const OperationGraph* base_ptr = &graph;
    base_ptr->Execute();

    EXPECT_TRUE(graph.WasExecuted());
}

TEST(HALAbstraction, ImmediateModeGraphExecute) {
    ImmediateModeGraph graph;
    int counter = 0;

    graph.RecordOperation([&counter]() { counter++; });
    graph.RecordOperation([&counter]() { counter += 10; });

    EXPECT_EQ(counter, 0);

    // Execute through polymorphic interface
    const OperationGraph* base_ptr = &graph;
    base_ptr->Execute();

    EXPECT_EQ(counter, 11);

    // Execute again
    base_ptr->Execute();
    EXPECT_EQ(counter, 22);
}

// =============================================================================
// Test: OpRegistry Dependency Injection
// =============================================================================

TEST(HALAbstraction, OpRegistryDependencyInjection) {
    // Create isolated registry for testing
    auto test_registry = std::make_unique<OpRegistry>();

    // Register mock op
    test_registry->Register<MockOp>(OpType::MatMul, DeviceType::CPU);

    // Set as test instance
    OpRegistry::SetTestInstance(std::move(test_registry));
    EXPECT_TRUE(OpRegistry::HasTestInstance());

    // Instance() should now return test instance
    auto* op = OpRegistry::Instance().Get(OpType::MatMul, DeviceType::CPU);
    EXPECT_NE(op, nullptr);

    // Reset to global instance
    OpRegistry::ResetToGlobalInstance();
    EXPECT_FALSE(OpRegistry::HasTestInstance());
}

TEST(HALAbstraction, OpRegistryGetBestCaching) {
    auto test_registry = std::make_unique<OpRegistry>();

    // Register two ops with different priorities
    auto low_priority = std::make_shared<MockOp>(50);
    auto high_priority = std::make_shared<MockOp>(100);

    test_registry->RegisterImpl(OpType::RMSNorm, DeviceType::CPU, low_priority);
    test_registry->RegisterImpl(OpType::RMSNorm, DeviceType::CPU, high_priority);

    OpRegistry::SetTestInstance(std::move(test_registry));

    // First call should compute and cache
    auto* best1 = OpRegistry::Instance().GetBest(OpType::RMSNorm, DeviceType::CPU);
    EXPECT_NE(best1, nullptr);
    EXPECT_EQ(best1->GetCapabilities().priority, 100);  // Higher priority

    // Second call should return cached result
    auto* best2 = OpRegistry::Instance().GetBest(OpType::RMSNorm, DeviceType::CPU);
    EXPECT_EQ(best1, best2);  // Same pointer (cached)

    OpRegistry::ResetToGlobalInstance();
}

// =============================================================================
// Test: Stream/Event RAII
// =============================================================================

TEST(HALAbstraction, StreamRAII) {
    MockDevice device;

    EXPECT_EQ(device.GetStreamsCreated(), 0);
    EXPECT_EQ(device.GetStreamsDestroyed(), 0);

    {
        StreamPtr stream = MakeStream(&device);
        EXPECT_EQ(device.GetStreamsCreated(), 1);
        EXPECT_EQ(device.GetStreamsDestroyed(), 0);
        EXPECT_NE(stream.get(), nullptr);
    }
    // Stream should be destroyed when StreamPtr goes out of scope
    EXPECT_EQ(device.GetStreamsDestroyed(), 1);
}

TEST(HALAbstraction, EventRAII) {
    MockDevice device;

    EXPECT_EQ(device.GetEventsCreated(), 0);
    EXPECT_EQ(device.GetEventsDestroyed(), 0);

    {
        EventPtr event = MakeEvent(&device);
        EXPECT_EQ(device.GetEventsCreated(), 1);
        EXPECT_EQ(device.GetEventsDestroyed(), 0);
        EXPECT_NE(event.get(), nullptr);
    }
    // Event should be destroyed when EventPtr goes out of scope
    EXPECT_EQ(device.GetEventsDestroyed(), 1);
}

TEST(HALAbstraction, StreamRAIIMultiple) {
    MockDevice device;

    {
        StreamPtr stream1 = MakeStream(&device);
        StreamPtr stream2 = MakeStream(&device);
        EXPECT_EQ(device.GetStreamsCreated(), 2);
        EXPECT_EQ(device.GetStreamsDestroyed(), 0);

        // Move stream1 to stream3
        StreamPtr stream3 = std::move(stream1);
        EXPECT_EQ(stream1.get(), nullptr);
        EXPECT_NE(stream3.get(), nullptr);
    }
    // Both streams should be destroyed
    EXPECT_EQ(device.GetStreamsDestroyed(), 2);
}

TEST(HALAbstraction, StreamDeleterWithNullDevice) {
    // Test that StreamDeleter handles null device gracefully
    StreamDeleter deleter(nullptr);
    MockStream* stream = new MockStream();

    // Should not crash - just doesn't delete
    deleter(stream);

    // Manual cleanup since deleter didn't work
    delete stream;
}

}  // namespace testing
}  // namespace densecore
