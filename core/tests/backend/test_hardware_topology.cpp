#include <gtest/gtest.h>

#include <set>

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

// Regression: SetupComputeThreadAffinity used to coerce the "no node requested"
// sentinel (-1) to node 0 and build the core list from that node alone. Callers
// size n_threads from the machine-wide physical core count, so on a multi-socket
// host every thread was packed onto one node's cores -- several threads per core
// -- while the other socket idled. With a machine-wide request, a thread count
// that fits in the machine must map to that many *distinct* cores.
TEST(HardwareTopology, SetupComputeThreadAffinityMachineWideUsesDistinctCores) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    const int physical = topo.GetPhysicalCoreCount();
    if (physical < 2) {
        GTEST_SKIP() << "needs at least 2 physical cores";
    }

    topo.SetupComputeThreadAffinity(-1, physical, densecore::PinningPolicy::SCATTER);
    ASSERT_TRUE(topo.IsComputeAffinityConfigured());

    std::set<int> assigned;
    for (int i = 0; i < physical; ++i) {
        const int core = topo.GetAssignedCore(i);
        EXPECT_GE(core, 0) << "thread " << i << " has no core";
        assigned.insert(core);
    }
    EXPECT_EQ(static_cast<int>(assigned.size()), physical)
        << "compute threads share cores even though the machine has one per thread";
}

// Regression for the hwloc 2 NUMA-node mislabel.
//
// hwloc 2.0 moved NUMA nodes out of the CPU hierarchy, so the old
// hwloc_get_ancestor_obj_by_type(HWLOC_OBJ_NUMANODE, pu) lookup returned NULL
// for every PU and every core was labelled node 0. numa_node_count_ comes from
// a different query and stayed correct at 2, so nothing in the node count gave
// it away -- but GetPhysicalCoreIds(1) came back empty, which silently turned
// every node-1 thread pin into a no-op. If the host says it has N>1 nodes, the
// per-core labels have to use more than one of them.
TEST(HardwareTopology, PhysicalCoresAreNotAllCollapsedOntoOneNumaNode) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    const int node_count = topo.GetNumaNodeCount();
    if (node_count < 2) {
        GTEST_SKIP() << "single-node host";
    }

    std::set<int> labelled_nodes;
    int covered = 0;
    for (int node = 0; node < node_count; ++node) {
        const auto ids = topo.GetPhysicalCoreIds(node);
        if (!ids.empty()) {
            labelled_nodes.insert(node);
            covered += static_cast<int>(ids.size());
        }
    }
    EXPECT_GE(static_cast<int>(labelled_nodes.size()), 2)
        << "host reports " << node_count << " NUMA nodes but every physical core is labelled with one of them";
    EXPECT_EQ(covered, topo.GetPhysicalCoreCount())
        << "per-node core lists must partition the machine's physical cores";
}

// The interleaved list must cover every physical core exactly once, and any
// prefix of it must touch every NUMA node it can. A prefix that stayed on one
// node is what confined the compute pool to a single socket.
TEST(HardwareTopology, InterleavedPhysicalCoresSpanNodesInPrefixOrder) {
    auto& topo = densecore::HardwareTopology::GetInstance();
    const auto grouped = topo.GetPhysicalCoreIds(-1);
    const auto interleaved = topo.GetPhysicalCoreIdsInterleavedByNode();

    std::multiset<int> grouped_set(grouped.begin(), grouped.end());
    std::multiset<int> interleaved_set(interleaved.begin(), interleaved.end());
    EXPECT_EQ(grouped_set, interleaved_set) << "interleaving must not add or drop cores";

    const int node_count = topo.GetNumaNodeCount();
    if (node_count < 2 || static_cast<int>(interleaved.size()) < node_count) {
        GTEST_SKIP() << "needs a multi-node host with a core per node";
    }
    std::set<int> nodes_in_prefix;
    for (int i = 0; i < node_count; ++i) {
        nodes_in_prefix.insert(topo.GetNumaNodeOfCore(interleaved[static_cast<size_t>(i)]));
    }
    EXPECT_EQ(static_cast<int>(nodes_in_prefix.size()), node_count)
        << "the first " << node_count << " cores must cover every node";
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
