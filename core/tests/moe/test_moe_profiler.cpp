/**
 * @file test_moe_profiler.cpp
 * @brief Unit tests for MoE ExpertProfiler
 *
 * Tests thread-safety, EMA tracking, and GetHotExperts functionality.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include "densecore/moe/profiler.h"

using namespace densecore::moe;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(MoEProfiler, Construction) {
    ExpertProfiler profiler(8);
    EXPECT_EQ(profiler.GetNumExperts(), 8);
    EXPECT_EQ(profiler.GetTotalHits(), 0);
}

TEST(MoEProfiler, RecordHit_Basic) {
    ExpertProfiler profiler(4);

    profiler.RecordHit(0);
    profiler.RecordHit(1);
    profiler.RecordHit(1);
    profiler.RecordHit(2);

    EXPECT_EQ(profiler.GetTotalHits(), 4);

    auto stats0 = profiler.GetExpertStats(0);
    auto stats1 = profiler.GetExpertStats(1);
    auto stats2 = profiler.GetExpertStats(2);
    auto stats3 = profiler.GetExpertStats(3);

    EXPECT_EQ(stats0.hit_count.load(), 1);
    EXPECT_EQ(stats1.hit_count.load(), 2);
    EXPECT_EQ(stats2.hit_count.load(), 1);
    EXPECT_EQ(stats3.hit_count.load(), 0);
}

TEST(MoEProfiler, RecordHit_OutOfBounds) {
    ExpertProfiler profiler(4);

    // Should silently ignore out-of-bounds
    profiler.RecordHit(-1);
    profiler.RecordHit(4);
    profiler.RecordHit(100);

    EXPECT_EQ(profiler.GetTotalHits(), 0);
}

TEST(MoEProfiler, RecordHitBatch) {
    ExpertProfiler profiler(8);

    std::vector<int> experts = {0, 1, 2, 1, 3, 2, 2};
    profiler.RecordHitBatch(experts.data(), experts.size());

    EXPECT_EQ(profiler.GetTotalHits(), 7);
    EXPECT_EQ(profiler.GetExpertStats(0).hit_count.load(), 1);
    EXPECT_EQ(profiler.GetExpertStats(1).hit_count.load(), 2);
    EXPECT_EQ(profiler.GetExpertStats(2).hit_count.load(), 3);
    EXPECT_EQ(profiler.GetExpertStats(3).hit_count.load(), 1);
}

// =============================================================================
// EMA Tracking Tests
// =============================================================================

TEST(MoEProfiler, EMA_SingleHit) {
    ExpertProfiler profiler(4);

    profiler.RecordHit(0);

    auto stats = profiler.GetExpertStats(0);
    // After one hit: EMA = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
    EXPECT_NEAR(stats.load_ema.load(), 0.1f, 0.01f);
}

TEST(MoEProfiler, EMA_MultipleHits) {
    ExpertProfiler profiler(4);

    // Hit expert 0 multiple times
    for (int i = 0; i < 10; i++) {
        profiler.RecordHit(0);
    }

    auto stats = profiler.GetExpertStats(0);
    // After 10 hits: EMA converges toward 1.0
    // EMA(n) = α + (1-α) * EMA(n-1)
    // EMA(10) ≈ 1 - (1-0.1)^10 ≈ 0.65
    EXPECT_GT(stats.load_ema.load(), 0.5f);
    EXPECT_LT(stats.load_ema.load(), 1.0f);
}

TEST(MoEProfiler, ApplyDecay) {
    ExpertProfiler profiler(4);

    // Hit expert 0 many times to build up EMA
    for (int i = 0; i < 20; i++) {
        profiler.RecordHit(0);
    }

    float ema_before = profiler.GetExpertStats(0).load_ema.load();
    EXPECT_GT(ema_before, 0.5f);

    // Apply decay
    profiler.ApplyDecay();

    float ema_after = profiler.GetExpertStats(0).load_ema.load();
    // EMA should decrease by factor (1 - α) = 0.9
    EXPECT_NEAR(ema_after, ema_before * 0.9f, 0.01f);
}

// =============================================================================
// GetHotExperts Tests
// =============================================================================

TEST(MoEProfiler, GetHotExperts_Basic) {
    ExpertProfiler profiler(8);

    // Create distinct hotness levels
    for (int i = 0; i < 50; i++) profiler.RecordHit(3);  // Hottest
    for (int i = 0; i < 30; i++) profiler.RecordHit(1);  // 2nd
    for (int i = 0; i < 20; i++) profiler.RecordHit(5);  // 3rd
    for (int i = 0; i < 10; i++) profiler.RecordHit(0);  // 4th

    auto hot = profiler.GetHotExperts(3);

    ASSERT_EQ(hot.size(), 3);
    EXPECT_EQ(hot[0], 3);  // Hottest
    EXPECT_EQ(hot[1], 1);  // 2nd
    EXPECT_EQ(hot[2], 5);  // 3rd
}

TEST(MoEProfiler, GetHotExperts_MoreThanAvailable) {
    ExpertProfiler profiler(4);

    profiler.RecordHit(0);
    profiler.RecordHit(1);

    auto hot = profiler.GetHotExperts(10);  // Request more than n_experts

    EXPECT_EQ(hot.size(), 4);  // Should return all 4
}

TEST(MoEProfiler, GetHotExperts_ZeroK) {
    ExpertProfiler profiler(4);

    profiler.RecordHit(0);

    auto hot = profiler.GetHotExperts(0);

    EXPECT_TRUE(hot.empty());
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(MoEProfiler, RecordHit_Concurrent) {
    constexpr int n_experts = 16;
    constexpr int n_threads = 8;
    constexpr int hits_per_thread = 10000;

    ExpertProfiler profiler(n_experts);
    std::vector<std::thread> threads;

    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&profiler, t]() {
            std::mt19937 rng(t);
            std::uniform_int_distribution<> dist(0, n_experts - 1);

            for (int i = 0; i < hits_per_thread; i++) {
                profiler.RecordHit(dist(rng));
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Verify total hits
    EXPECT_EQ(profiler.GetTotalHits(), static_cast<uint64_t>(n_threads * hits_per_thread));
}

TEST(MoEProfiler, GetHotExperts_WhileRecording) {
    constexpr int n_experts = 16;
    constexpr int n_recording_threads = 4;
    constexpr int n_reading_threads = 2;

    ExpertProfiler profiler(n_experts);
    std::atomic<bool> stop{false};
    std::atomic<int> read_count{0};

    std::vector<std::thread> threads;

    // Recording threads
    for (int t = 0; t < n_recording_threads; t++) {
        threads.emplace_back([&profiler, &stop, t]() {
            std::mt19937 rng(t);
            std::uniform_int_distribution<> dist(0, n_experts - 1);

            while (!stop.load()) {
                profiler.RecordHit(dist(rng));
            }
        });
    }

    // Reading threads
    for (int t = 0; t < n_reading_threads; t++) {
        threads.emplace_back([&profiler, &stop, &read_count]() {
            while (!stop.load()) {
                auto hot = profiler.GetHotExperts(4);
                EXPECT_LE(hot.size(), 4);
                read_count.fetch_add(1);
            }
        });
    }

    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_GT(read_count.load(), 0);
    EXPECT_GT(profiler.GetTotalHits(), 0);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST(MoEProfiler, Reset) {
    ExpertProfiler profiler(4);

    for (int i = 0; i < 100; i++) {
        profiler.RecordHit(0);
    }

    EXPECT_GT(profiler.GetTotalHits(), 0);

    profiler.Reset();

    EXPECT_EQ(profiler.GetTotalHits(), 0);
    EXPECT_FLOAT_EQ(profiler.GetExpertStats(0).load_ema.load(), 0.0f);
}
