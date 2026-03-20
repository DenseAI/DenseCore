/**
 * @file test_memory_pool.cpp
 * @brief Unit tests for memory pool components
 */

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "memory_pool.h"

using namespace densecore;

// =============================================================================
// ArenaAllocator Tests
// =============================================================================

TEST(ArenaAllocator, BasicAllocation) {
    ArenaAllocator arena(1024 * 1024);  // 1MB

    void* ptr = arena.Allocate(1024);
    ASSERT_NE(ptr, nullptr);

    // Should be able to write to it
    memset(ptr, 0xAB, 1024);
}

TEST(ArenaAllocator, Alignment) {
    ArenaAllocator arena(1024 * 1024);

    for (int i = 0; i < 10; i++) {
        void* ptr = arena.Allocate(i + 1);  // Various sizes
        ASSERT_NE(ptr, nullptr);

        // Check 64-byte alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % MEMORY_ALIGNMENT, 0) << "Allocation " << i << " not aligned";
    }
}

TEST(ArenaAllocator, Reset) {
    ArenaAllocator arena(1024 * 1024);

    // Allocate some memory
    void* ptr1 = arena.Allocate(1024);
    arena.Allocate(2048);
    arena.Allocate(4096);

    size_t used_before = arena.Used();
    EXPECT_GT(used_before, 0);

    // Reset
    arena.Reset();

    EXPECT_EQ(arena.Used(), 0);

    // Allocate again, should get same or similar address
    void* ptr2 = arena.Allocate(1024);
    EXPECT_EQ(ptr1, ptr2);
}

TEST(ArenaAllocator, UsageTracking) {
    ArenaAllocator arena(1024 * 1024);

    EXPECT_EQ(arena.Used(), 0);
    EXPECT_GE(arena.Capacity(), 1024 * 1024);

    arena.Allocate(100);
    EXPECT_GE(arena.Used(), 100);

    arena.Allocate(200);
    EXPECT_GE(arena.Used(), 300);
}

TEST(ArenaAllocator, TypedAllocation) {
    ArenaAllocator arena(1024 * 1024);

    float* floats = arena.Allocate<float>(100);
    ASSERT_NE(floats, nullptr);

    // Write and read back
    for (int i = 0; i < 100; i++) {
        floats[i] = static_cast<float>(i);
    }

    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(floats[i], static_cast<float>(i));
    }
}

TEST(ArenaAllocator, GrowOnDemand) {
    ArenaAllocator arena(64);  // Very small initial size

    // Allocate more than initial capacity
    void* ptr = arena.Allocate(1024);
    ASSERT_NE(ptr, nullptr);

    // Capacity should have grown
    EXPECT_GE(arena.Capacity(), 1024);
}

TEST(ArenaAllocator, MultipleResets) {
    ArenaAllocator arena(1024 * 1024);

    for (int round = 0; round < 5; round++) {
        // Allocate various sizes
        for (int i = 0; i < 100; i++) {
            void* ptr = arena.Allocate(64 + i * 16);
            ASSERT_NE(ptr, nullptr);
        }

        // Reset for next round
        arena.Reset();
        EXPECT_EQ(arena.Used(), 0);
    }
}

// =============================================================================
// BlockPool Tests
// =============================================================================

TEST(BlockPool, BasicAllocateFree) {
    BlockPool pool(1024, 8);  // 1KB blocks, 8 initial

    EXPECT_EQ(pool.BlockSize(), AlignUp(1024, MEMORY_ALIGNMENT));
    EXPECT_EQ(pool.TotalBlocks(), 8);
    EXPECT_EQ(pool.FreeBlocks(), 8);

    void* block = pool.Allocate();
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(pool.FreeBlocks(), 7);

    pool.Free(block);
    EXPECT_EQ(pool.FreeBlocks(), 8);
}

TEST(BlockPool, Reuse) {
    BlockPool pool(1024, 4);

    void* block1 = pool.Allocate();
    pool.Free(block1);
    void* block2 = pool.Allocate();

    EXPECT_EQ(block1, block2);  // Should reuse same block
}

TEST(BlockPool, AllocateAll) {
    BlockPool pool(256, 4);

    std::vector<void*> blocks;
    for (int i = 0; i < 4; i++) {
        void* block = pool.Allocate();
        ASSERT_NE(block, nullptr);
        blocks.push_back(block);
    }

    EXPECT_EQ(pool.FreeBlocks(), 0);

    // Allocate more - should grow
    void* extra = pool.Allocate();
    ASSERT_NE(extra, nullptr);
    EXPECT_GT(pool.TotalBlocks(), 4);

    // Clean up
    for (void* b : blocks) {
        pool.Free(b);
    }
    pool.Free(extra);
}

TEST(BlockPool, Alignment) {
    BlockPool pool(100, 4);  // Non-aligned size

    void* block = pool.Allocate();
    ASSERT_NE(block, nullptr);

    uintptr_t addr = reinterpret_cast<uintptr_t>(block);
    EXPECT_EQ(addr % MEMORY_ALIGNMENT, 0);

    pool.Free(block);
}

TEST(BlockPool, ThreadSafety) {
    BlockPool pool(256, 16);

    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 100;

    std::atomic<int> successful_allocs{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&pool, &successful_allocs]() {
            std::vector<void*> my_blocks;

            for (int i = 0; i < OPS_PER_THREAD; i++) {
                void* block = pool.Allocate();
                if (block) {
                    successful_allocs++;
                    my_blocks.push_back(block);
                }

                // Randomly free some blocks
                if (!my_blocks.empty() && (i % 3 == 0)) {
                    pool.Free(my_blocks.back());
                    my_blocks.pop_back();
                }
            }

            // Free remaining blocks
            for (void* b : my_blocks) {
                pool.Free(b);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All allocations should have succeeded
    EXPECT_EQ(successful_allocs.load(), NUM_THREADS * OPS_PER_THREAD);

    // All blocks should be returned to pool
    EXPECT_EQ(pool.FreeBlocks(), pool.TotalBlocks());
}

// =============================================================================
// InferenceScratch Tests
// =============================================================================

TEST(InferenceScratch, Initialize) {
    InferenceScratch scratch;

    scratch.Initialize(512,   // n_embd
                       8,     // n_head
                       8,     // n_head_kv
                       12,    // n_layer
                       2048,  // max_batch_tokens
                       4096   // max_seq_len
    );

    EXPECT_GT(scratch.MemoryCapacity(), 0);
}

TEST(InferenceScratch, AllocateReset) {
    InferenceScratch scratch;
    scratch.Initialize(512, 8, 8, 12, 2048, 4096);

    float* buf1 = scratch.Allocate<float>(1024);
    ASSERT_NE(buf1, nullptr);

    size_t used1 = scratch.MemoryUsed();
    EXPECT_GT(used1, 0);

    float* buf2 = scratch.Allocate<float>(512);
    ASSERT_NE(buf2, nullptr);
    EXPECT_NE(buf1, buf2);

    size_t used2 = scratch.MemoryUsed();
    EXPECT_GT(used2, used1);

    scratch.Reset();
    EXPECT_EQ(scratch.MemoryUsed(), 0);

    // After reset, should get same address as before
    float* buf3 = scratch.Allocate<float>(1024);
    EXPECT_EQ(buf1, buf3);
}

// =============================================================================
// AlignUp Utility Test
// =============================================================================

TEST(MemoryUtils, AlignUp) {
    EXPECT_EQ(AlignUp(0, 64), 0);
    EXPECT_EQ(AlignUp(1, 64), 64);
    EXPECT_EQ(AlignUp(63, 64), 64);
    EXPECT_EQ(AlignUp(64, 64), 64);
    EXPECT_EQ(AlignUp(65, 64), 128);
    EXPECT_EQ(AlignUp(100, 16), 112);
}

// =============================================================================
// MemoryPoolManager Tests
// =============================================================================

TEST(MemoryPoolManager, Singleton) {
    auto& manager1 = MemoryPoolManager::Instance();
    auto& manager2 = MemoryPoolManager::Instance();

    EXPECT_EQ(&manager1, &manager2);
}

TEST(MemoryPoolManager, GetBlockPool) {
    auto& manager = MemoryPoolManager::Instance();

    BlockPool& pool1 = manager.GetBlockPool(1024);
    BlockPool& pool2 = manager.GetBlockPool(1024);

    // Same size should return same pool
    EXPECT_EQ(&pool1, &pool2);

    // Different size should return different pool
    BlockPool& pool3 = manager.GetBlockPool(2048);
    EXPECT_NE(&pool1, &pool3);
}
