/**
 * @file test_op_registry.cpp
 * @brief Unit tests for OpRegistry
 *
 * Verifies vendor plugin registration/retrieval system.
 */

#include <gtest/gtest.h>

#include <cstdlib>

#include "densecore/hal/op_registry.h"
#include "densecore/hal/operation_graph.h"

namespace densecore {
namespace {

// ============================================================================
// Mock Op Implementation for Testing
// ============================================================================

// ============================================================================
// Mock Op Implementation for Testing
// ============================================================================

class MockEmbeddingOp : public EmbeddingOps {
public:
    MockEmbeddingOp(int priority = 0) : priority_(priority) {}

    void Execute(const std::vector<Tensor*>& /*inputs*/, const std::vector<Tensor*>& /*outputs*/, const void* /*params*/
                 ) override {}

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = priority_};
    }

    std::tuple<int, int, int> GetOptimalTileConfig(const std::vector<int64_t>& /*input_shape*/,
                                                   DeviceType /*device*/) const override {
        return {1, 1, 1};
    }

    void PatchEmbed2D(const Tensor& /*image*/, const Tensor& /*conv_weight*/, const Tensor& /*conv_bias*/,
                      Tensor* /*patches*/
                      ) override {}

    void RoPE2D(const Tensor& /*input*/, const Tensor& /*pos_h*/, const Tensor& /*pos_w*/, const Tensor& /*cos_sin*/,
                Tensor* /*output*/
                ) override {}

private:
    int priority_;
};

class MockSelectableOp : public EmbeddingOps {
public:
    MockSelectableOp(int priority, bool supports_int8, TensorLayout layout_ok)
        : priority_(priority), supports_int8_(supports_int8), layout_ok_(layout_ok) {}

    void Execute(const std::vector<Tensor*>& /*inputs*/, const std::vector<Tensor*>& /*outputs*/, const void* /*params*/
                 ) override {}

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = supports_int8_,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = priority_};
    }

    std::tuple<int, int, int> GetOptimalTileConfig(const std::vector<int64_t>& /*input_shape*/,
                                                   DeviceType /*device*/) const override {
        return {1, 1, 1};
    }

    bool SupportsLayout(TensorLayout layout) const override {
        return layout == TensorLayout::UNKNOWN || layout == layout_ok_;
    }

    void PatchEmbed2D(const Tensor& /*image*/, const Tensor& /*conv_weight*/, const Tensor& /*conv_bias*/,
                      Tensor* /*patches*/
                      ) override {}

    void RoPE2D(const Tensor& /*input*/, const Tensor& /*pos_h*/, const Tensor& /*pos_w*/, const Tensor& /*cos_sin*/,
                Tensor* /*output*/
                ) override {}

private:
    int priority_;
    bool supports_int8_;
    TensorLayout layout_ok_;
};

class MockAnyDeviceOp : public EmbeddingOps {
public:
    explicit MockAnyDeviceOp(int priority = 0) : priority_(priority) {}

    void Execute(const std::vector<Tensor*>& /*inputs*/, const std::vector<Tensor*>& /*outputs*/, const void* /*params*/
                 ) override {}

    bool Supports(DeviceType /*device*/) const override { return true; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = true,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = priority_};
    }

    std::tuple<int, int, int> GetOptimalTileConfig(const std::vector<int64_t>& /*input_shape*/,
                                                   DeviceType /*device*/) const override {
        return {1, 1, 1};
    }

    void PatchEmbed2D(const Tensor& /*image*/, const Tensor& /*conv_weight*/, const Tensor& /*conv_bias*/,
                      Tensor* /*patches*/
                      ) override {}

    void RoPE2D(const Tensor& /*input*/, const Tensor& /*pos_h*/, const Tensor& /*pos_w*/, const Tensor& /*cos_sin*/,
                Tensor* /*output*/
                ) override {}

private:
    int priority_;
};

// ============================================================================
// Tests
// ============================================================================

class OpRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Registry is a global singleton, so be careful
    }
};

TEST_F(OpRegistryTest, SingletonConsistency) {
    // Check if Instance() always returns the same object
    auto& instance1 = OpRegistry::Instance();
    auto& instance2 = OpRegistry::Instance();

    EXPECT_EQ(&instance1, &instance2);
    EXPECT_TRUE(OpRegistry::IsInitialized());
}

TEST_F(OpRegistryTest, RegisterAndRetrieve) {
    // Register and retrieve kernel
    OpRegistry::Instance().Register<MockEmbeddingOp>(OpType::PatchEmbed2D, DeviceType::CPU);

    auto* op = OpRegistry::Instance().Get(OpType::PatchEmbed2D, DeviceType::CPU);
    EXPECT_NE(op, nullptr);

    // Check support
    EXPECT_TRUE(op->Supports(DeviceType::CPU));
}

TEST_F(OpRegistryTest, GetBestReturnsHighestPriority) {
    // Register kernels with multiple priorities
    auto low_priority = std::make_shared<MockEmbeddingOp>(10);
    auto high_priority = std::make_shared<MockEmbeddingOp>(100);

    OpRegistry::Instance().RegisterImpl(OpType::RoPE2D, DeviceType::CPU, low_priority);
    OpRegistry::Instance().RegisterImpl(OpType::RoPE2D, DeviceType::CPU, high_priority);

    auto* best = OpRegistry::Instance().GetBest(OpType::RoPE2D, DeviceType::CPU);
    ASSERT_NE(best, nullptr);

    // Check if the highest priority kernel is returned
    EXPECT_EQ(best->GetCapabilities().priority, 100);
}

TEST_F(OpRegistryTest, GetBestFallsBackToCPU) {
    // CPU 커널만 등록된 상태에서 NPU 요청 시 CPU fallback
    OpRegistry::Instance().Register<MockEmbeddingOp>(OpType::PatchEmbed3D, DeviceType::CPU);

    // Requesting NPU (which doesn't exist, so CPU fallback)
    auto* op = OpRegistry::Instance().GetBest(OpType::PatchEmbed3D, DeviceType::NPU);
    EXPECT_NE(op, nullptr);  // CPU fallback successful
}

TEST_F(OpRegistryTest, GetBestPrefersMetalBeforeCpuForNpuRequest) {
    auto cpu = std::make_shared<MockAnyDeviceOp>(10);
    auto metal = std::make_shared<MockAnyDeviceOp>(20);

    OpRegistry::Instance().RegisterImpl(OpType::Custom, DeviceType::CPU, cpu);
    OpRegistry::Instance().RegisterImpl(OpType::Custom, DeviceType::METAL, metal);

    auto* best = OpRegistry::Instance().GetBest(OpType::Custom, DeviceType::NPU);
    ASSERT_NE(best, nullptr);
    EXPECT_EQ(best->GetCapabilities().priority, 20);
}

#if defined(__APPLE__)
TEST_F(OpRegistryTest, AppleVideo3DOpsHaveExplicitNpuRegistrations) {
    OpRegistry& registry = OpRegistry::Instance();
    EXPECT_NE(registry.Get(OpType::TemporalAttention, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::PatchEmbed3D, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::Patchify, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::Unpatchify, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::PointCloudPatchify, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::PointCloudUnpatchify, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::DeformableAttention, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::FarthestPointSampling, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::KNNQuery, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::BallQuery, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::NeRFPositionalEncoding, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::GaussianFourierFeatures, DeviceType::NPU), nullptr);
    EXPECT_NE(registry.Get(OpType::GridSample, DeviceType::NPU), nullptr);
}
#endif

TEST_F(OpRegistryTest, GetBestRespectsLayoutCriteria) {
    auto high_priority = std::make_shared<MockSelectableOp>(100, true, TensorLayout::NHWC);
    auto low_priority = std::make_shared<MockSelectableOp>(10, true, TensorLayout::NCHW);

    OpRegistry::Instance().RegisterImpl(OpType::Patchify, DeviceType::CPU, high_priority);
    OpRegistry::Instance().RegisterImpl(OpType::Patchify, DeviceType::CPU, low_priority);

    OpRegistry::SelectionCriteria criteria;
    criteria.layout = TensorLayout::NCHW;
    auto* best = OpRegistry::Instance().GetBest(OpType::Patchify, DeviceType::CPU, criteria);
    ASSERT_NE(best, nullptr);
    EXPECT_EQ(best->GetCapabilities().priority, 10);
}

TEST_F(OpRegistryTest, GetBestRespectsDTypeCriteria) {
    auto high_priority = std::make_shared<MockSelectableOp>(100, false, TensorLayout::SEQ);
    auto low_priority = std::make_shared<MockSelectableOp>(10, true, TensorLayout::SEQ);

    OpRegistry::Instance().RegisterImpl(OpType::Unpatchify, DeviceType::CPU, high_priority);
    OpRegistry::Instance().RegisterImpl(OpType::Unpatchify, DeviceType::CPU, low_priority);

    OpRegistry::SelectionCriteria criteria;
    criteria.dtype = DType::INT8;
    auto* best = OpRegistry::Instance().GetBest(OpType::Unpatchify, DeviceType::CPU, criteria);
    ASSERT_NE(best, nullptr);
    EXPECT_EQ(best->GetCapabilities().priority, 10);
}

TEST_F(OpRegistryTest, GetReturnsNullForUnregistered) {
    // Retrieve unregistered OpType
    auto* op = OpRegistry::Instance().Get(OpType::Custom, DeviceType::ASIC);
    EXPECT_EQ(op, nullptr);
}

TEST_F(OpRegistryTest, ListRegisteredContainsOps) {
    // Check list for debugging
    auto list = OpRegistry::Instance().ListRegistered();
    // Should have at least one registered Op (registered in previous tests)
    EXPECT_GT(list.size(), 0);
}

TEST_F(OpRegistryTest, DetectDeviceTypeReturnsValid) {
    auto device = OpRegistry::DetectDeviceType();

    // Should return one of known runtime-selectable devices
    EXPECT_TRUE(device == DeviceType::CPU || device == DeviceType::METAL || device == DeviceType::NPU ||
                device == DeviceType::ASIC);
}

TEST_F(OpRegistryTest, DetectDeviceTypeEnvOverride) {
#if defined(_WIN32)
    _putenv_s("DENSECORE_PREFERRED_DEVICE", "CPU");
#else
    setenv("DENSECORE_PREFERRED_DEVICE", "CPU", 1);
#endif
    EXPECT_EQ(OpRegistry::DetectDeviceType(), DeviceType::CPU);

#if defined(_WIN32)
    _putenv_s("DENSECORE_PREFERRED_DEVICE", "");
#else
    unsetenv("DENSECORE_PREFERRED_DEVICE");
#endif
}

TEST_F(OpRegistryTest, ProfileAdmissionDetectsMissingOps) {
    OpRegistry registry;
    OpRegistry::AdmissionPolicy policy;
    policy.allow_fallback = true;

    auto report = registry.CheckProfileAdmission(InferenceProfile::Embedding, DeviceType::CPU, policy);
    EXPECT_FALSE(report.admitted);
    EXPECT_GT(report.missing_ops.size(), 0u);
}

TEST_F(OpRegistryTest, ProfileAdmissionSupportsFallbackPolicy) {
    OpRegistry registry;
    for (OpType op : OpRegistry::GetRequiredOpsForProfile(InferenceProfile::Embedding)) {
        registry.RegisterImpl(op, DeviceType::CPU, std::make_shared<MockAnyDeviceOp>(100));
    }

    OpRegistry::AdmissionPolicy allow_fallback;
    allow_fallback.allow_fallback = true;
    auto ok_with_fallback = registry.CheckProfileAdmission(InferenceProfile::Embedding, DeviceType::NPU, allow_fallback);
    EXPECT_TRUE(ok_with_fallback.admitted);
    EXPECT_TRUE(ok_with_fallback.used_fallback);
    EXPECT_GT(ok_with_fallback.fallback_ops.size(), 0u);

    OpRegistry::AdmissionPolicy strict_native;
    strict_native.allow_fallback = false;
    auto strict = registry.CheckProfileAdmission(InferenceProfile::Embedding, DeviceType::NPU, strict_native);
    EXPECT_FALSE(strict.admitted);
    EXPECT_GT(strict.fallback_ops.size(), 0u);
}

TEST_F(OpRegistryTest, GraphAdmissionFindsMissingNodeOps) {
    OpRegistry registry;
    registry.RegisterImpl(OpType::MatMul, DeviceType::CPU, std::make_shared<MockAnyDeviceOp>(100));

    float a_data[4] = {0};
    float b_data[4] = {0};
    float c_data[4] = {0};
    OperationGraph graph;
    const size_t a = graph.RegisterTensor(Tensor::Make2D(a_data, 2, 2));
    const size_t b = graph.RegisterTensor(Tensor::Make2D(b_data, 2, 2));
    const size_t c = graph.RegisterTensor(Tensor::Make2D(c_data, 2, 2));
    graph.AddNode(OpType::MatMul, {a, b}, {c}, "matmul");
    graph.AddNode(OpType::Custom, {a}, {c}, "custom_missing");

    auto report = registry.CheckGraphAdmission(graph, DeviceType::CPU);
    EXPECT_FALSE(report.admitted);
    ASSERT_EQ(report.missing_ops.size(), 1u);
    EXPECT_EQ(report.missing_ops[0], OpType::Custom);
}

TEST_F(OpRegistryTest, DispatchTelemetryCapturesFallbacks) {
    OpRegistry& registry = OpRegistry::Instance();
    registry.ResetDispatchTelemetry();
    registry.RegisterImpl(OpType::AudioConv1D, DeviceType::CPU, std::make_shared<MockAnyDeviceOp>(100));

    auto* best = registry.GetBest(OpType::AudioConv1D, DeviceType::NPU);
    ASSERT_NE(best, nullptr);

    auto snapshot = registry.GetDispatchTelemetry();
    EXPECT_GE(snapshot.total_requests, 1u);
    EXPECT_GE(snapshot.fallbacks, 1u);
    ASSERT_FALSE(snapshot.fallback_records.empty());
    EXPECT_EQ(snapshot.fallback_records.front().op, OpType::AudioConv1D);
    EXPECT_EQ(snapshot.fallback_records.front().requested_device, DeviceType::NPU);
}

}  // namespace
}  // namespace densecore
