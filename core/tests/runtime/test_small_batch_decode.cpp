#include "densecore/models/model_types.h"
#include <cstdint>
#include <gtest/gtest.h>

namespace densecore::testing {
bool NativeMoELegacyDispatchBatchForTest(int exit_mode, int threads, int calls);
bool NativeM4AvailableForTest(bool q6);
bool RunNativeM4CallbackForTest(bool q6, int tokens, int phase, bool reference_only, int threads, bool* exact,
                                uint64_t* used, bool qwen36 = true, bool padded = false);
bool RunQwenNativeM4GraphForTest(bool fused, bool* matches, uint64_t* ops);
bool QwenQ4DownPairAvailableForTest();
bool RunQwenQ4DownPairCallbackForTest(bool qwen36, int tokens, int phase, bool* matches, uint64_t* used);
int ModelVariantAfterWorkContextResetForTest(int kind);
}  // namespace densecore::testing

TEST(SmallBatchDecodeTest, CachedGraphAndProfileResetPreserveModelIdentity) {
    EXPECT_EQ(densecore::testing::ModelVariantAfterWorkContextResetForTest(0), static_cast<int>(ModelVariant::QWEN36));
    EXPECT_EQ(densecore::testing::ModelVariantAfterWorkContextResetForTest(1), static_cast<int>(ModelVariant::QWEN36));
    EXPECT_EQ(densecore::testing::ModelVariantAfterWorkContextResetForTest(2), static_cast<int>(ModelVariant::UNKNOWN));
}

TEST(SmallBatchDecodeTest, Q4DownPairPreservesWeightedOutputsTailsAndDispatchCounts) {
    for (bool qwen36 : {false, true})
        for (int tokens : {1, 4})
            for (int phase : {1, 2}) {
                bool matches = false;
                uint64_t used = 0;
                ASSERT_TRUE(
                    densecore::testing::RunQwenQ4DownPairCallbackForTest(qwen36, tokens, phase, &matches, &used));
                EXPECT_TRUE(matches);
                const bool enabled =
                    qwen36 && tokens == 4 && phase == 2 && densecore::testing::QwenQ4DownPairAvailableForTest();
                EXPECT_EQ(used, enabled ? 16u : 0u);
            }
}

TEST(SmallBatchDecodeTest, NativeTilesMatchDotAcrossThreadPartitionsWithoutTLS) {
    for (bool q6 : {false, true}) {
        for (int threads : {1, 16}) {
            bool exact = false;
            uint64_t used = 0;
            ASSERT_TRUE(densecore::testing::RunNativeM4CallbackForTest(q6, 4, 2, false, threads, &exact, &used));
            EXPECT_TRUE(exact);
            EXPECT_EQ(used, densecore::testing::NativeM4AvailableForTest(q6) ? 1u : 0u);
        }
    }
}

TEST(SmallBatchDecodeTest, QwenDecodeGraphSelectsNativeQ8Tile) {
    for (bool fused : {false, true}) {
        bool matches = false;
        uint64_t used = 0;
        ASSERT_TRUE(densecore::testing::RunQwenNativeM4GraphForTest(fused, &matches, &used));
        EXPECT_TRUE(matches);
        EXPECT_EQ(used, densecore::testing::NativeM4AvailableForTest(false) ? 1u : 0u);
    }
}

TEST(SmallBatchDecodeTest, NativeTilesKeepOtherBatchSizesPrefillAndReferenceOnExistingPath) {
    for (bool q6 : {false, true}) {
        for (int tokens : {1, 2, 3, 4}) {
            for (int phase : {1, 2}) {
                const bool reference_only = tokens == 4 && phase == 2;
                bool exact = false;
                uint64_t used = 0;
                ASSERT_TRUE(densecore::testing::RunNativeM4CallbackForTest(q6, tokens, phase, reference_only, 2, &exact,
                                                                           &used));
                EXPECT_TRUE(exact);
                EXPECT_EQ(used, 0u);
            }
        }
    }
}

TEST(SmallBatchDecodeTest, LegacyDispatchCountsSurviveFlushReturnAndException) {
    for (int exit_mode : {0, 1, 2}) {
        for (int threads : {1, 16}) {
            EXPECT_TRUE(densecore::testing::NativeMoELegacyDispatchBatchForTest(exit_mode, threads, 32));
        }
    }
}

TEST(SmallBatchDecodeTest, OtherModelDoesNotUseNativeM4Tile) {
    for (bool q6 : {false, true}) {
        bool exact = false;
        uint64_t used = 0;
        ASSERT_TRUE(densecore::testing::RunNativeM4CallbackForTest(q6, 4, 2, false, 2, &exact, &used, false));
        EXPECT_TRUE(exact);
        EXPECT_EQ(used, 0u);
    }
}

TEST(SmallBatchDecodeTest, NativeTilesRespectPaddedInputAndOutputColumns) {
    for (bool q6 : {false, true}) {
        bool exact = false;
        uint64_t used = 0;
        ASSERT_TRUE(densecore::testing::RunNativeM4CallbackForTest(q6, 4, 2, false, 16, &exact, &used, true, true));
        EXPECT_TRUE(exact);
        EXPECT_EQ(used, densecore::testing::NativeM4AvailableForTest(q6) ? 1u : 0u);
    }
}
