#include "ggml.h"
#include "runtime/batched_activation_pack_cache.h"
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <thread>

using densecore::runtime::BatchedActivationPackCache;

TEST(BatchedActivationPackCacheTest, ConcurrentWorkersFillOnceAndKeepSnapshotAlive) {
    BatchedActivationPackCache cache;
    const int op = 0, source = 0;
    const BatchedActivationPackCache::Key key{1, &op, &source, 4, 256};
    std::atomic<int> fills{0};
    std::vector<std::thread> workers;
    std::vector<BatchedActivationPackCache::Snapshot> snapshots(8);
    for (size_t i = 0; i < snapshots.size(); ++i) {
        workers.emplace_back([&, i] {
            snapshots[i] = cache.GetOrFill(key, [&](auto& bytes) {
                ++fills;
                bytes.assign(64, 7);
            });
        });
    }
    for (auto& worker : workers) worker.join();
    EXPECT_EQ(fills.load(), 1);
    for (const auto& snapshot : snapshots) EXPECT_EQ(snapshot, snapshots[0]);
    auto next_key = key;
    ++next_key.generation;
    auto next = cache.GetOrFill(next_key, [](auto& bytes) { bytes.assign(64, 9); });
    EXPECT_EQ(next->front(), 9);
    EXPECT_EQ(snapshots[0]->front(), 7);
}

TEST(BatchedActivationPackCacheTest, IdentityShapeAndGenerationInvalidate) {
    BatchedActivationPackCache cache;
    int op = 0, other_op = 0, source = 0, other_source = 0;
    BatchedActivationPackCache::Key key{1, &op, &source, 4, 256};
    int fills = 0;
    auto fill = [&](auto& bytes) { bytes.assign(8, static_cast<uint8_t>(++fills)); };
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 1);
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 1);
    ++key.generation;  // same memory, new graph execution
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 2);
    key.operation = &other_op;  // same source reused after an in-place producer
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 3);
    key.source = &other_source;
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 4);
    key.rows = 5;  // padded tail width is part of the key
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 5);
    key.cols = 512;
    EXPECT_EQ(cache.GetOrFill(key, fill)->front(), 6);
    key.generation = 0;
    EXPECT_FALSE(cache.GetOrFill(key, fill));
    EXPECT_EQ(fills, 6);
}

namespace densecore::testing {
int RunQ4KSharedBatchedPackForTest(int tokens);
int RunQ4KMap3SharedBatchedPackForTest(int tokens);
}

TEST(BatchedActivationPackCacheTest, Q4KConcurrentCallbackMatchesVecDotAcrossExecutions) {
    for (int tokens : {4, 8}) {
        const int result = densecore::testing::RunQ4KSharedBatchedPackForTest(tokens);
        if (result < 0) GTEST_SKIP() << "Q4K repacked kernel unavailable on this ISA";
        EXPECT_EQ(result, 1) << "tokens=" << tokens;
    }
}

TEST(BatchedActivationPackCacheTest, Q4KPaddedTailMatchesVecDotAcrossExecutions) {
    const char* pad = std::getenv("DENSECORE_BATCHED_DECODE_REPACK_GEMM_PAD");
    if (!pad || std::string(pad) != "1") {
        GTEST_SKIP() << "Run with DENSECORE_BATCHED_DECODE_REPACK_GEMM_PAD=1 to cover optional padding";
    }
    for (const int tokens : {3, 5, 7}) {
        const int result = densecore::testing::RunQ4KSharedBatchedPackForTest(tokens);
        if (result < 0) GTEST_SKIP() << "Q4K repacked kernel unavailable on this ISA";
        EXPECT_EQ(result, 1) << "tokens=" << tokens;
    }
}

TEST(BatchedActivationPackCacheTest, Map3OperationIdentityIsolatesReusedInputWithinExecution) {
    for (int tokens : {4, 8}) {
        const int result = densecore::testing::RunQ4KMap3SharedBatchedPackForTest(tokens);
        if (result < 0) GTEST_SKIP() << "Q4K repacked kernel unavailable on this ISA";
        EXPECT_EQ(result, 1) << "tokens=" << tokens;
    }
}

namespace densecore::testing {
bool RunQuantBatchedColumnsForTest(ggml_type type, int tokens);
}

TEST(BatchedActivationPackCacheTest, QuantizedDecodeWritesEveryColumn) {
    for (const auto type : {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        for (const int tokens : {2, 3, 4, 5, 7}) {
            EXPECT_TRUE(densecore::testing::RunQuantBatchedColumnsForTest(type, tokens))
                << "type=" << type << " tokens=" << tokens;
        }
    }
}
