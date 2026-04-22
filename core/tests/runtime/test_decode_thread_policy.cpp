#include <gtest/gtest.h>

#include "worker_internal.h"

namespace {

using densecore::simd::SimdLevel;

TEST(DecodeThreadPolicy, AVX2SingleSequenceDoesNotGrabAllCores) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(2, 64, 64, SimdLevel::AVX2), 8);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(4, 64, 64, SimdLevel::AVX2), 16);
}

TEST(DecodeThreadPolicy, AVX2StillScalesUpOnHighCoreHosts) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX2), 32);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(16, 64, 64, SimdLevel::AVX2), 48);
}

TEST(DecodeThreadPolicy, WideSimdGetsLargerPerSequenceBudget) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX512), 8);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX512), 64);
}

TEST(DecodeThreadPolicy, RespectsConfiguredBaseThreadCap) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX512), 24);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX2), 24);
}

TEST(DecodeThreadPolicy, SmallHostsKeepAvailableThreads) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 4, 4, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 2, 2, SimdLevel::AVX2), 2);
}

}  // namespace
