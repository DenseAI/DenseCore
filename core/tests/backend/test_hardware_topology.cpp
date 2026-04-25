#include <gtest/gtest.h>

#include "densecore/backend/hardware_topology.h"
#include "densecore/simd/simd_ops.h"

namespace {

TEST(HardwareTopology, DetectsPositiveCoreCount) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    EXPECT_GT(densecore::simd::GetNumCores(), 0);
    EXPECT_GE(topo.GetLogicalCoreCount(), 0);
    EXPECT_GE(topo.GetPhysicalCoreCount(), 0);
}

TEST(HardwareTopology, SetupComputeThreadAffinityFromCoreIdsScatter) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    topo.SetupComputeThreadAffinityFromCoreIds({7, 3}, 3, densecore::PinningPolicy::SCATTER);

    EXPECT_TRUE(topo.IsComputeAffinityConfigured());
    EXPECT_EQ(topo.GetAssignedCore(0), 3);
    EXPECT_EQ(topo.GetAssignedCore(1), 7);
    EXPECT_EQ(topo.GetAssignedCore(2), 3);
}

TEST(HardwareTopology, SetupComputeThreadAffinityFromCoreIdsCompact) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    topo.SetupComputeThreadAffinityFromCoreIds({5, 2}, 3, densecore::PinningPolicy::COMPACT);

    EXPECT_TRUE(topo.IsComputeAffinityConfigured());
    EXPECT_EQ(topo.GetAssignedCore(0), 2);
    EXPECT_EQ(topo.GetAssignedCore(1), 5);
    EXPECT_EQ(topo.GetAssignedCore(2), 5);
}

}  // namespace
