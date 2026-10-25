#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <future>
#include <vector>

#include "kernels/q4k_repacked_gemv.h"
#include "densecore.h"
#include "densecore/runtime/resource_ledger.h"

namespace densecore::testing {
bool Q5RepackedMoESupportedForTest();
std::shared_ptr<void> Q5RepackedMoEViewForTest(const void* source, int64_t rows, int64_t cols,
                                            const void** data, size_t* bytes);
}

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

TEST(Q4KRepackedLifetimeTest, EngineRetirementInvalidatesReusedStorageWithIdenticalEdges) {
    using namespace densecore::kernels;
    if (!Q4KRealPackedGemvKernelAvailable()) GTEST_SKIP() << "Host has no packed Q4_K kernel";
    constexpr int64_t rows = 128;
    constexpr int64_t cols = 256;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    std::vector<float> source(rows * cols);
    for (size_t i = 0; i < source.size(); ++i) source[i] = std::sin(static_cast<float>(i) * 0.017f);
    std::vector<uint8_t> reused_storage(rows * row_bytes);
    for (int64_t row = 0; row < rows; ++row)
        ggml_quantize_chunk(GGML_TYPE_Q4_K, source.data() + row * cols, reused_storage.data() + row * row_bytes,
                            0, 1, cols, nullptr);
    const auto fingerprint = FingerprintQ4KRepackedGemvWeight(reused_storage.data(), reused_storage.size());
    const auto original = GetOrCreateQ4KRepackedGemvWeight(reused_storage.data(), rows, cols);
    ASSERT_NE(original, nullptr);
    const auto original_blocks = original->blocks;
    // A new allocation/model can reuse this address, shape and sampled edges.
    // Real engine teardown must retire its process-global packed-weight views.
    DenseCoreHandle engine = InitEngine("mock", nullptr, 1);
    ASSERT_NE(engine, nullptr) << DenseCoreGetLastError();
    FreeEngine(engine);
    reused_storage[reused_storage.size() / 2 + 64] ^= 0x11;
    ASSERT_EQ(fingerprint, FingerprintQ4KRepackedGemvWeight(reused_storage.data(), reused_storage.size()));
    Q4KRepackedGemvCacheLookup lookup;
    const auto replacement = GetOrCreateQ4KRepackedGemvWeight(reused_storage.data(), rows, cols, &lookup);
    ASSERT_NE(replacement, nullptr);
    EXPECT_FALSE(lookup.cache_hit);
    EXPECT_NE(replacement, original);
    // The old inflight shared owner remains valid across retirement.
    EXPECT_EQ(std::memcmp(original->blocks.data(), original_blocks.data(), original->bytes), 0);
    std::vector<Q4KRepackedGemvBlock> expected(original_blocks.size());
    ASSERT_EQ(ggml_repack_q4_K_8x8(reused_storage.data(), reused_storage.size(), rows, cols,
                                  expected.data(), original->bytes), 0);
    EXPECT_EQ(std::memcmp(replacement->blocks.data(), expected.data(), replacement->bytes), 0);
    EXPECT_NE(std::memcmp(original->blocks.data(), expected.data(), original->bytes), 0);
}

TEST(Q4KRepackedLifetimeTest, EngineRetirementAlsoInvalidatesMoEQ5PackedViews) {
    using namespace densecore::testing;
    if (!Q5RepackedMoESupportedForTest()) GTEST_SKIP() << "Host has no packed MoE Q5_K kernel";
    constexpr int64_t rows = 128;
    constexpr int64_t cols = 256;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_K, cols);
    std::vector<float> source(rows * cols);
    for (size_t i = 0; i < source.size(); ++i) source[i] = std::sin(static_cast<float>(i) * 0.013f);
    std::vector<uint8_t> reused_storage(rows * row_bytes);
    for (int64_t row = 0; row < rows; ++row)
        ggml_quantize_chunk(GGML_TYPE_Q5_K, source.data() + row * cols, reused_storage.data() + row * row_bytes,
                            0, 1, cols, nullptr);
    const void* old_data = nullptr;
    size_t old_bytes = 0;
    const auto old_owner = Q5RepackedMoEViewForTest(reused_storage.data(), rows, cols, &old_data, &old_bytes);
    ASSERT_NE(old_owner, nullptr);
    const auto* old_begin = static_cast<const uint8_t*>(old_data);
    const std::vector<uint8_t> old_copy(old_begin, old_begin + old_bytes);
    DenseCoreHandle engine = InitEngine("mock", nullptr, 1);
    ASSERT_NE(engine, nullptr) << DenseCoreGetLastError();
    FreeEngine(engine);
    reused_storage[reused_storage.size() / 2 + 64] ^= 0x11;
    const void* new_data = nullptr;
    size_t new_bytes = 0;
    const auto replacement = Q5RepackedMoEViewForTest(reused_storage.data(), rows, cols, &new_data, &new_bytes);
    ASSERT_NE(replacement, nullptr);
    EXPECT_NE(replacement, old_owner);
    EXPECT_EQ(old_bytes, new_bytes);
    EXPECT_EQ(std::memcmp(old_data, old_copy.data(), old_bytes), 0);
    std::vector<uint8_t> expected(old_bytes);
    ASSERT_EQ(ggml_repack_q5_K_8x8(reused_storage.data(), reused_storage.size(), rows, cols,
                                  expected.data(), expected.size()), 0);
    EXPECT_EQ(std::memcmp(new_data, expected.data(), new_bytes), 0);
    EXPECT_NE(std::memcmp(old_data, expected.data(), old_bytes), 0);
}

TEST(Q4KRepackedLifetimeTest, AllocationFailureDoesNotStrandBuilderOrBudget) {
    using namespace densecore::kernels;
    if (!Q4KRealPackedGemvKernelAvailable()) GTEST_SKIP() << "Host has no packed Q4_K kernel";
    struct Cleanup {
        ~Cleanup() { Q4KRepackedGemvFailNextAllocationForTest(false); RetireRepackedWeightCaches(); }
    } cleanup;
    RetireRepackedWeightCaches();
    constexpr int64_t rows = 128;
    constexpr int64_t cols = 256;
    std::vector<uint8_t> source(rows * ggml_row_size(GGML_TYPE_Q4_K, cols), 0);
    const auto before = densecore::runtime::ResourceLedgerUsageSnapshot().repack_bytes;
    Q4KRepackedGemvFailNextAllocationForTest();
    EXPECT_THROW(GetOrCreateQ4KRepackedGemvWeight(source.data(), rows, cols), std::bad_alloc);
    Q4KRepackedGemvFailNextAllocationForTest(false);
    ASSERT_EQ(Q4KRepackedGemvBuildersForTest(), 0u);
    EXPECT_EQ(densecore::runtime::ResourceLedgerUsageSnapshot().repack_bytes, before);
    const auto recovered = GetOrCreateQ4KRepackedGemvWeight(source.data(), rows, cols);
    ASSERT_NE(recovered, nullptr);
    EXPECT_EQ(Q4KRepackedGemvBuildersForTest(), 0u);
}

void CheckRetirementDuringBuild(bool fail_detached_builder) {
    using namespace densecore::kernels;
    constexpr int64_t rows = 128;
    constexpr int64_t cols = 256;
    std::vector<uint8_t> source(rows * ggml_row_size(GGML_TYPE_Q4_K, cols), 0);
    std::future<std::shared_ptr<Q4KRepackedGemvWeight>> delayed;
    struct Cleanup {
        ~Cleanup() {
            Q4KRepackedGemvFailNextAllocationForTest(false);
            Q4KRepackedGemvPauseNextBuildForTest(false);
            RetireRepackedWeightCaches();
        }
    } cleanup;  // Releases a paused future before its destructor waits.
    RetireRepackedWeightCaches();
    Q4KRepackedGemvPauseNextBuildForTest(true);
    delayed = std::async(std::launch::async, [&]() {
        return GetOrCreateQ4KRepackedGemvWeight(source.data(), rows, cols);
    });
    ASSERT_TRUE(Q4KRepackedGemvWaitForPausedBuildForTest(5000));
    RetireRepackedWeightCaches();
    const auto replacement = GetOrCreateQ4KRepackedGemvWeight(source.data(), rows, cols);
    ASSERT_NE(replacement, nullptr);
    if (fail_detached_builder) Q4KRepackedGemvFailNextAllocationForTest();
    Q4KRepackedGemvPauseNextBuildForTest(false);
    if (fail_detached_builder) {
        EXPECT_THROW(delayed.get(), std::bad_alloc);
    } else {
        const auto detached = delayed.get();
        ASSERT_NE(detached, nullptr);
        EXPECT_NE(detached, replacement);
    }
    const auto current = GetOrCreateQ4KRepackedGemvWeight(source.data(), rows, cols);
    EXPECT_EQ(current, replacement);
    EXPECT_EQ(Q4KRepackedGemvCacheStatsSnapshot().resident_bytes, replacement->bytes);
    EXPECT_EQ(Q4KRepackedGemvBuildersForTest(), 0u);
}

TEST(Q4KRepackedLifetimeTest, DetachedBuilderCannotChargeRetiredCacheEntry) {
    if (!densecore::kernels::Q4KRealPackedGemvKernelAvailable()) GTEST_SKIP() << "Host has no packed Q4_K kernel";
    CheckRetirementDuringBuild(false);
}

TEST(Q4KRepackedLifetimeTest, FailedDetachedBuilderCannotEraseReplacementEntry) {
    if (!densecore::kernels::Q4KRealPackedGemvKernelAvailable()) GTEST_SKIP() << "Host has no packed Q4_K kernel";
    CheckRetirementDuringBuild(true);
}
}  // namespace
