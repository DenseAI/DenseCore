#include <gtest/gtest.h>
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"

namespace densecore {
namespace {

class MockOpDI : public DenseCoreOp {
public:
    void Execute(const std::vector<Tensor*>&, const std::vector<Tensor*>&, const void*) override {}
    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }
    OpCapabilities GetCapabilities() const override { return {.priority = 100}; }
};

TEST(OpRegistryDITest, IndependentInstances) {
    OpRegistry registry1;
    OpRegistry registry2;

    registry1.Register<MockOpDI>(OpType::Custom, DeviceType::CPU);

    // Registry 1 should have it
    EXPECT_NE(registry1.Get(OpType::Custom, DeviceType::CPU), nullptr);
    
    // Registry 2 should NOT have it (isolated)
    EXPECT_EQ(registry2.Get(OpType::Custom, DeviceType::CPU), nullptr);
}

TEST(OpRegistryDITest, GlobalRegistrarIntegration) {
    // Add a global registrar
    OpRegistry::AddGlobalRegistrar([](OpRegistry& r) {
        r.Register<MockOpDI>(OpType::Softmax, DeviceType::CPU);
    });

    // New instance should automatically have the global registration
    OpRegistry registry;
    EXPECT_NE(registry.Get(OpType::Softmax, DeviceType::CPU), nullptr);
}

} // namespace
} // namespace densecore
