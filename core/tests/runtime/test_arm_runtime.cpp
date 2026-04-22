/**
 * @file test_arm_runtime.cpp
 * @brief Unit tests for ARM runtime utility helpers
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "densecore/arm_runtime.h"

namespace densecore {
namespace arm_runtime {
namespace {

TEST(ArmRuntimeTest, ParseCpuListBasic) {
    std::vector<int> cores;
    ASSERT_TRUE(ParseCpuList("0-3,6,8-9", &cores));
    const std::vector<int> expected = {0, 1, 2, 3, 6, 8, 9};
    EXPECT_EQ(cores, expected);
}

TEST(ArmRuntimeTest, ParseCpuListRejectsInvalidInput) {
    std::vector<int> cores;
    EXPECT_FALSE(ParseCpuList("3-1", &cores));
    EXPECT_FALSE(ParseCpuList("abc", &cores));
    EXPECT_FALSE(ParseCpuList("", &cores));
}

TEST(ArmRuntimeTest, PartitionByFrequencySplitsBigLittle) {
    const std::vector<std::pair<int, int>> freq = {
        {0, 1200000},
        {1, 1200000},
        {2, 2200000},
        {3, 2200000},
    };

    CoreClusters clusters = PartitionCoresByMaxFrequency(freq);
    const std::vector<int> expected_little = {0, 1};
    const std::vector<int> expected_big = {2, 3};
    EXPECT_EQ(clusters.little_cores, expected_little);
    EXPECT_EQ(clusters.big_cores, expected_big);
}

TEST(ArmRuntimeTest, PartitionFallsBackToAllBigWhenUniform) {
    const std::vector<std::pair<int, int>> freq = {
        {0, 1800000},
        {1, 1800000},
        {2, 1800000},
    };

    CoreClusters clusters = PartitionCoresByMaxFrequency(freq);
    EXPECT_TRUE(clusters.little_cores.empty());
    const std::vector<int> expected_big = {0, 1, 2};
    EXPECT_EQ(clusters.big_cores, expected_big);
}

}  // namespace
}  // namespace arm_runtime
}  // namespace densecore
