#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "kernels/q4k_repacked_gemv.h"

namespace {
using densecore::kernels::FingerprintQ4KRepackedGemvWeight;

TEST(Q4KFingerprintTest, EmptyInputIsZero) {
    const uint8_t byte = 1;
    EXPECT_EQ(FingerprintQ4KRepackedGemvWeight(nullptr, 100), 0u);
    EXPECT_EQ(FingerprintQ4KRepackedGemvWeight(&byte, 0), 0u);
}

TEST(Q4KFingerprintTest, CoversEverySampledByteAndUnalignedInput) {
    for (const size_t size : {1u, 3u, 4u, 7u, 8u, 31u, 4095u, 4096u, 4097u, 8191u, 8192u, 8193u, 16385u}) {
        std::vector<uint8_t> storage(size + 1);
        auto* data = storage.data() + 1;
        for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint8_t>(i * 37 + 11);
        const auto original = FingerprintQ4KRepackedGemvWeight(data, size);
        const std::vector<uint8_t> copy(data, data + size);
        EXPECT_EQ(original, FingerprintQ4KRepackedGemvWeight(copy.data(), size));
        for (size_t i = 0; i < size; ++i) {
            if (size > 8192 && i >= 4096 && i < size - 4096) continue;
            data[i] ^= 0x80;
            ASSERT_NE(original, FingerprintQ4KRepackedGemvWeight(data, size)) << size << ':' << i;
            data[i] ^= 0x80;
        }
        EXPECT_EQ(original, FingerprintQ4KRepackedGemvWeight(data, size));
    }
}

TEST(Q4KFingerprintTest, LengthParticipatesAndSampleWindowIsUnchanged) {
    std::vector<uint8_t> data(16385, 0);
    EXPECT_NE(FingerprintQ4KRepackedGemvWeight(data.data(), 16384),
              FingerprintQ4KRepackedGemvWeight(data.data(), 16385));
    const auto original = FingerprintQ4KRepackedGemvWeight(data.data(), data.size());
    data[8192] = 1;  // The existing contract samples the first and last 4 KiB only.
    EXPECT_EQ(original, FingerprintQ4KRepackedGemvWeight(data.data(), data.size()));
}

TEST(Q4KFingerprintTest, LaneOrderAndRepeatedPatternsRemainDistinct) {
    std::array<uint8_t, 64> data{};
    for (size_t first = 0; first < 8; ++first) {
        data.fill(0);
        data[first] = 1;
        const auto original = FingerprintQ4KRepackedGemvWeight(data.data(), data.size());
        for (size_t second = first + 1; second < 8; ++second) {
            std::swap(data[first], data[second]);
            EXPECT_NE(original, FingerprintQ4KRepackedGemvWeight(data.data(), data.size()));
            std::swap(data[first], data[second]);
        }
    }
}
}  // namespace
