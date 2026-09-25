#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "densecore/backend/cpu_backend.h"
#include "densecore/models/model_types.h"

namespace densecore::testing {
std::vector<int> SelectLocalExpertsWithHysteresisForTest(const std::vector<int>& previous_local,
                                                          const std::vector<int>& hot_expert_ids,
                                                          const std::vector<float>& ema_loads, int n_experts);
float ComputeHotSetChurnForTest(const std::vector<int>& previous_hot, const std::vector<int>& current_hot);
int NextRebalanceIntervalMsForTest(int current_interval_ms, int base_interval_ms, bool any_high, bool all_low);
int NextMigrationStableCyclesForTest(int current_cycles, bool has_previous_hot, float churn, uint64_t delta_hits);
}  // namespace densecore::testing

namespace {

using densecore::CpuBackend;

struct FreeDeleter {
    void operator()(void* ptr) const { std::free(ptr); }
};
using PageBuffer = std::unique_ptr<void, FreeDeleter>;

PageBuffer AllocatePageBuffer(size_t page_size, size_t page_count) {
    const size_t size = page_size * page_count;
    void* ptr = std::aligned_alloc(page_size, size);
    if (ptr) std::memset(ptr, 0, size);
    return PageBuffer(ptr);
}

std::vector<CpuBackend::ExpertWeights> MakeExpertWeights(std::vector<PageBuffer>* buffers, int n_experts,
                                                          size_t page_size, size_t page_count = 3) {
    std::vector<CpuBackend::ExpertWeights> experts(static_cast<size_t>(n_experts));
    buffers->reserve(static_cast<size_t>(n_experts) * 3);
    const size_t bytes = page_size * page_count;
    for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
        buffers->push_back(AllocatePageBuffer(page_size, page_count));
        buffers->push_back(AllocatePageBuffer(page_size, page_count));
        buffers->push_back(AllocatePageBuffer(page_size, page_count));
        const size_t base = static_cast<size_t>(expert_id) * 3;
        EXPECT_NE((*buffers)[base].get(), nullptr);
        EXPECT_NE((*buffers)[base + 1].get(), nullptr);
        EXPECT_NE((*buffers)[base + 2].get(), nullptr);
        experts[static_cast<size_t>(expert_id)].w1 = {(*buffers)[base].get(), bytes};
        experts[static_cast<size_t>(expert_id)].w2 = {(*buffers)[base + 1].get(), bytes};
        experts[static_cast<size_t>(expert_id)].w3 = {(*buffers)[base + 2].get(), bytes};
    }
    return experts;
}

template <typename Predicate>
bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
}

TEST(NumaRebalancePolicy, ChurnUsesJaccardDistance) {
    EXPECT_FLOAT_EQ(densecore::testing::ComputeHotSetChurnForTest({}, {}), 0.0f);
    EXPECT_FLOAT_EQ(densecore::testing::ComputeHotSetChurnForTest({0, 1}, {0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(densecore::testing::ComputeHotSetChurnForTest({0, 1}, {2, 3}), 1.0f);
    EXPECT_NEAR(densecore::testing::ComputeHotSetChurnForTest({0, 1, 2, 3}, {2, 3, 4, 5}),
                2.0f / 3.0f, 1e-6f);
}

TEST(NumaRebalancePolicy, AdaptiveIntervalHonorsDirectionAndBounds) {
    EXPECT_EQ(densecore::testing::NextRebalanceIntervalMsForTest(1000, 1000, true, false), 700);
    EXPECT_EQ(densecore::testing::NextRebalanceIntervalMsForTest(1000, 1000, false, true), 1300);
    EXPECT_EQ(densecore::testing::NextRebalanceIntervalMsForTest(1000, 1000, false, false), 1000);
    EXPECT_EQ(densecore::testing::NextRebalanceIntervalMsForTest(250, 1000, true, false), 250);
    EXPECT_EQ(densecore::testing::NextRebalanceIntervalMsForTest(4000, 1000, false, true), 4000);
}

TEST(NumaRebalancePolicy, HysteresisRequiresStrictlyMoreThanFifteenPercent) {
    const std::vector<int> previous{0};
    const std::vector<int> candidate{1};
    EXPECT_EQ(densecore::testing::SelectLocalExpertsWithHysteresisForTest(previous, candidate, {100.0f, 114.9f}, 2),
              previous);
    EXPECT_EQ(densecore::testing::SelectLocalExpertsWithHysteresisForTest(previous, candidate, {100.0f, 115.0f}, 2),
              previous);
    EXPECT_EQ(densecore::testing::SelectLocalExpertsWithHysteresisForTest(previous, candidate, {100.0f, 115.1f}, 2),
              candidate);
}

TEST(NumaRebalancePolicy, MigrationRequiresTwoStableActiveObservations) {
    using densecore::testing::NextMigrationStableCyclesForTest;

    EXPECT_EQ(NextMigrationStableCyclesForTest(0, false, 0.0f, 8), 0);
    EXPECT_EQ(NextMigrationStableCyclesForTest(0, true, 0.0f, 0), 0);
    EXPECT_EQ(NextMigrationStableCyclesForTest(0, true, 0.0f, 8), 1);
    EXPECT_EQ(NextMigrationStableCyclesForTest(1, true, 0.10f, 8), 2);
    EXPECT_EQ(NextMigrationStableCyclesForTest(2, true, 0.0f, 8), 2);
    EXPECT_EQ(NextMigrationStableCyclesForTest(2, true, 0.11f, 8), 0);
}

#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
TEST(NumaRebalancePolicy, MigrationCommitsOnlyFullyVerifiedExpert) {
    CpuBackend backend;
    TransformerLayer layer{};
    constexpr int kTargetNode = 1;
    constexpr size_t kPageSize = 4096;
    std::vector<PageBuffer> buffers;
    auto experts = MakeExpertWeights(&buffers, 2, kPageSize);
    ASSERT_EQ(buffers.size(), 6u);

    backend.InitMoEProfiler(&layer, 2, 1.0f);
    backend.SetNumaRebalanceTestHooks(
        [](size_t count, void**, const int* nodes, int* status) {
            for (size_t i = 0; i < count; ++i) status[i] = nodes[i];
            return 0L;
        },
        [](const void*, size_t) { return kTargetNode; });

    EXPECT_GT(backend.RebalanceExperts(&layer, {0}, experts, 2, kTargetNode), 0);
    EXPECT_EQ(backend.GetExpertNumaNode(&layer, 0), kTargetNode);
    EXPECT_EQ(backend.GetLocalExpertIdsForTest(&layer), (std::vector<int>{0}));

    auto* profiler = backend.GetProfiler(&layer);
    ASSERT_NE(profiler, nullptr);
    profiler->RecordHit(1);
    backend.SetNumaRebalanceTestHooks(
        [](size_t, void**, const int*, int*) {
            errno = EIO;
            return -1L;
        },
        [](const void*, size_t) { return -1; });

    EXPECT_EQ(backend.RebalanceExperts(&layer, {1}, experts, 2, kTargetNode), 0);
    EXPECT_EQ(backend.GetExpertNumaNode(&layer, 1), -1);
    EXPECT_EQ(backend.GetLocalExpertIdsForTest(&layer), (std::vector<int>{0}));
}

TEST(NumaRebalancePolicy, ResidentExpertIsNotRemigratedAfterHotSetReentry) {
    CpuBackend backend;
    TransformerLayer layer{};
    constexpr int kTargetNode = 1;
    constexpr size_t kPageSize = 4096;
    std::vector<PageBuffer> buffers;
    auto experts = MakeExpertWeights(&buffers, 2, kPageSize);
    ASSERT_EQ(buffers.size(), 6u);

    backend.InitMoEProfiler(&layer, 2, 1.0f);
    int move_pages_calls = 0;
    backend.SetNumaRebalanceTestHooks(
        [&move_pages_calls](size_t count, void**, const int* nodes, int* status) {
            ++move_pages_calls;
            for (size_t i = 0; i < count; ++i) status[i] = nodes[i];
            return 0L;
        },
        [](const void*, size_t) { return kTargetNode; });

    auto* profiler = backend.GetProfiler(&layer);
    ASSERT_NE(profiler, nullptr);

    profiler->RecordHit(0);
    EXPECT_GT(backend.RebalanceExperts(&layer, {0}, experts, 2, kTargetNode), 0);
    EXPECT_EQ(move_pages_calls, 3);

    profiler->ApplyDecay();
    profiler->RecordHit(1);
    EXPECT_GT(backend.RebalanceExperts(&layer, {1}, experts, 2, kTargetNode), 0);
    EXPECT_EQ(move_pages_calls, 6);

    profiler->ApplyDecay();
    profiler->RecordHit(0);
    EXPECT_EQ(backend.RebalanceExperts(&layer, {0}, experts, 2, kTargetNode), 0);
    EXPECT_EQ(move_pages_calls, 6);
    EXPECT_EQ(backend.GetExpertNumaNode(&layer, 0), kTargetNode);
    EXPECT_EQ(backend.GetLocalExpertIdsForTest(&layer), (std::vector<int>{0}));
}

TEST(NumaRebalancePolicy, PermissionFailureDisablesFutureRebalancing) {
    CpuBackend backend;
    TransformerLayer layer{};
    constexpr size_t kPageSize = 4096;
    std::vector<PageBuffer> buffers;
    auto experts = MakeExpertWeights(&buffers, 1, kPageSize);
    ASSERT_EQ(buffers.size(), 3u);

    backend.InitMoEProfiler(&layer, 1);
    backend.SetNumaRebalanceTestHooks(
        [](size_t, void**, const int*, int*) {
            errno = EPERM;
            return -1L;
        },
        [](const void*, size_t) { return -1; });

    EXPECT_EQ(backend.RebalanceExperts(&layer, {0}, experts, 1, 0), 0);
    EXPECT_TRUE(backend.IsNumaRebalanceDisabledForTest());
    backend.StartRebalanceThread(250, 1, true);
    EXPECT_FALSE(backend.IsRebalanceThreadRunning());
}

TEST(NumaRebalanceIntegration, MigratesAllThreeWeightRangesAcrossNodes) {
    CpuBackend backend;
    const int node_count = backend.GetNumaNodeCount();
    if (node_count < 2) {
        GTEST_SKIP() << "requires at least two NUMA nodes";
    }

    TransformerLayer layer{};
    constexpr size_t kPageSize = 4096;
    constexpr size_t kPageCount = 4;
    std::vector<PageBuffer> buffers;
    auto experts = MakeExpertWeights(&buffers, 1, kPageSize, kPageCount);
    ASSERT_EQ(buffers.size(), 3u);

    const int source_node = 0;
    const int target_node = 1;
    const size_t bytes = kPageSize * kPageCount;
    for (const auto& buffer : buffers) {
        ASSERT_TRUE(backend.BindMemoryToNumaNode(buffer.get(), bytes, source_node));
        ASSERT_EQ(backend.QueryMemoryNumaNodeRange(buffer.get(), bytes), source_node);
    }

    backend.InitMoEProfiler(&layer, 1);
    const int migrated = backend.RebalanceExperts(&layer, {0}, experts, 1, target_node);
    EXPECT_EQ(migrated, static_cast<int>(3 * kPageCount));
    for (const auto& buffer : buffers) {
        EXPECT_EQ(backend.QueryMemoryNumaNodeRange(buffer.get(), bytes), target_node);
    }
    EXPECT_EQ(backend.GetExpertNumaNode(&layer, 0), target_node);
}
#endif

TEST(NumaRebalanceThread, ObservesHighThenLowChurnAndStopsCleanly) {
    CpuBackend backend;
    TransformerLayer layer{};
    backend.InitMoEProfiler(&layer, 4);

    backend.StartRebalanceThread(300, 2, false);
    ASSERT_TRUE(backend.IsRebalanceThreadRunning());
    ASSERT_TRUE(WaitUntil(
        [&] { return backend.GetNumaRebalanceTestStats().interval_decreases >= 1; },
        std::chrono::milliseconds(1000)));
    ASSERT_TRUE(WaitUntil(
        [&] { return backend.GetNumaRebalanceTestStats().interval_increases >= 1; },
        std::chrono::milliseconds(1000)));

    backend.StopRebalanceThread();
    const auto stats = backend.GetNumaRebalanceTestStats();
    EXPECT_FALSE(backend.IsRebalanceThreadRunning());
    EXPECT_GE(stats.cycles, 2u);
    EXPECT_GE(stats.interval_decreases, 1u);
    EXPECT_GE(stats.interval_increases, 1u);
    EXPECT_GT(stats.current_interval_ms, 0);
}

}  // namespace
