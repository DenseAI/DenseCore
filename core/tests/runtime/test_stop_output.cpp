#include <gtest/gtest.h>
#include <array>
#include "runtime/stop_output.h"

namespace {
using densecore::runtime::FilterStopOutput;
using densecore::runtime::FinishStopOutput;

TEST(StopOutput, MarkerInsideOnePieceIsExcludedWithFollowingText) {
    std::string pending, piece = "Paris<END>ignore";
    ASSERT_TRUE(FilterStopOutput(pending, piece, {"<END>"}));
    EXPECT_EQ(piece, "Paris");
    EXPECT_TRUE(pending.empty());
}
TEST(StopOutput, SplitMarkerIsHeldAndNeverLeaks) {
    std::string pending, first = "Paris<EN", second = "D>ignore";
    EXPECT_FALSE(FilterStopOutput(pending, first, {"<END>"}));
    EXPECT_EQ(first, "Paris"); EXPECT_EQ(pending, "<EN");
    EXPECT_TRUE(FilterStopOutput(pending, second, {"<END>"}));
    EXPECT_TRUE(second.empty()); EXPECT_TRUE(pending.empty());
}
TEST(StopOutput, EarlierIncompletePrefixDoesNotBlockAnAvailableFullMatch) {
    std::string pending, piece = "abc";
    EXPECT_TRUE(FilterStopOutput(pending, piece, {"abcd", "bc"}));
    EXPECT_EQ(piece, "a");
}
TEST(StopOutput, EarliestFullMatchWinsIndependentlyOfListOrder) {
    for (const auto& stops : {std::vector<std::string>{"later", "END"}, {"END", "later"}}) {
        std::string pending, piece = "beforeEND later";
        EXPECT_TRUE(FilterStopOutput(pending, piece, stops));
        EXPECT_EQ(piece, "before");
    }
}
TEST(StopOutput, OverlapAndLongPrefixRemainBoundedAndFlushOnSuccess) {
    std::string pending;
    const std::vector<std::string> stops{"ababababZ"};
    for (int i = 0; i < 50; ++i) {
        std::string piece = "ab";
        EXPECT_FALSE(FilterStopOutput(pending, piece, stops));
        EXPECT_LE(pending.size(), stops[0].size() - 1);
    }
    EXPECT_EQ(FinishStopOutput(pending, false), "abababab");
    EXPECT_TRUE(pending.empty());
    EXPECT_TRUE(FinishStopOutput(pending, false).empty());
}
TEST(StopOutput, EmptyStopsAndNoStopsDoNotDelayOrModifyOutput) {
    for (const auto& stops : {std::vector<std::string>{}, {"", ""}}) {
        std::string pending, piece = "café";
        const auto* original = piece.data();
        EXPECT_FALSE(FilterStopOutput(pending, piece, stops));
        EXPECT_EQ(piece, "café"); EXPECT_EQ(piece.data(), original);
    }
}
TEST(StopOutput, UnicodeStopAndPrecedingTextKeepWholeCodePoints) {
    std::string pending, first = "café🌟", second = "ENDtrailing";
    EXPECT_FALSE(FilterStopOutput(pending, first, {"🌟END"}));
    EXPECT_EQ(first, "café"); EXPECT_EQ(pending, "🌟");
    EXPECT_TRUE(FilterStopOutput(pending, second, {"🌟END"}));
    EXPECT_TRUE(second.empty());
}
TEST(StopOutput, InvalidByteMarkerCannotSplitAValidCharacter) {
    std::string pending, piece = "café.";
    EXPECT_FALSE(FilterStopOutput(pending, piece, {std::string("\xa9", 1)}));
    EXPECT_EQ(piece, "café.");
}
TEST(StopOutput, FourInterleavedRequestsKeepIndependentStopsAndCancelReuseClearsTail) {
    std::array<std::string, 4> pending;
    for (size_t i = 0; i < pending.size(); ++i) {
        std::string piece = "body<";
        EXPECT_FALSE(FilterStopOutput(pending[i], piece, {"<" + std::to_string(i) + ">"}));
        EXPECT_EQ(piece, "body");
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        if (i == 1) {
            EXPECT_TRUE(FinishStopOutput(pending[i], true).empty());
            std::string piece = "reuse";
            EXPECT_FALSE(FilterStopOutput(pending[i], piece, {"OTHER"}));
            EXPECT_EQ(piece, "reuse");
        } else {
            std::string piece = std::to_string(i) + ">ignored";
            EXPECT_TRUE(FilterStopOutput(pending[i], piece, {"<" + std::to_string(i) + ">"}));
            EXPECT_TRUE(piece.empty());
        }
    }
}
} // namespace
