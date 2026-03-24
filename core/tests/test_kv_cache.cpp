/**
 * @file test_kv_cache.cpp
 * @brief Unit tests for KV cache components
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <set>

#include "kv_cache.h"
#include "model_types.h"

// =============================================================================
// SequenceBlockTable Tests
// =============================================================================

TEST(SequenceBlockTable, GetPhysicalBlockId) {
    SequenceBlockTable table;
    table.block_ids = {10, 20, 30, 40};  // 4 blocks

    // Token 0-15 in block 0 (id=10)
    EXPECT_EQ(table.GetPhysicalBlockId(0), 10);
    EXPECT_EQ(table.GetPhysicalBlockId(15), 10);

    // Token 16-31 in block 1 (id=20)
    EXPECT_EQ(table.GetPhysicalBlockId(16), 20);
    EXPECT_EQ(table.GetPhysicalBlockId(31), 20);

    // Token 32-47 in block 2 (id=30)
    EXPECT_EQ(table.GetPhysicalBlockId(32), 30);
}

TEST(SequenceBlockTable, GetPhysicalBlockId_InvalidPos) {
    SequenceBlockTable table;
    table.block_ids = {10, 20};  // 2 blocks = 32 tokens max

    // Token beyond allocated blocks
    EXPECT_EQ(table.GetPhysicalBlockId(32), -1);
    EXPECT_EQ(table.GetPhysicalBlockId(100), -1);
}

TEST(SequenceBlockTable, GetSlotIndex) {
    SequenceBlockTable table;

    // BLOCK_SIZE is 16
    EXPECT_EQ(table.GetSlotIndex(0), 0);
    EXPECT_EQ(table.GetSlotIndex(1), 1);
    EXPECT_EQ(table.GetSlotIndex(15), 15);
    EXPECT_EQ(table.GetSlotIndex(16), 0);  // Wraps to next block
    EXPECT_EQ(table.GetSlotIndex(17), 1);
    EXPECT_EQ(table.GetSlotIndex(32), 0);
}

TEST(SequenceBlockTable, GetNumBlocks) {
    SequenceBlockTable table;
    EXPECT_EQ(table.GetNumBlocks(), 0);

    table.block_ids = {1, 2, 3};
    EXPECT_EQ(table.GetNumBlocks(), 3);
}

TEST(SequenceBlockTable, BlocksNeeded) {
    // BLOCK_SIZE is 16
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(0), 0);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(1), 1);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(16), 1);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(17), 2);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(32), 2);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(33), 3);
}

// =============================================================================
// PhysicalBlock Tests
// =============================================================================

TEST(PhysicalBlock, IsFull) {
    PhysicalBlock block;
    block.num_filled_slots = 0;

    EXPECT_FALSE(block.is_full());

    block.num_filled_slots = BLOCK_SIZE - 1;
    EXPECT_FALSE(block.is_full());

    block.num_filled_slots = BLOCK_SIZE;
    EXPECT_TRUE(block.is_full());
}

TEST(PhysicalBlock, RefCount) {
    PhysicalBlock block;
    block.ref_count = 1;

    block.ref_count++;
    EXPECT_EQ(block.ref_count, 2);

    block.ref_count--;
    EXPECT_EQ(block.ref_count, 1);
}

// =============================================================================
// BlockManager Tests
// =============================================================================

class BlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override { manager = std::make_unique<BlockManager>(32, BLOCK_SIZE); }

    void TearDown() override { manager.reset(); }

    std::unique_ptr<BlockManager> manager;
};

static TransformerModel MakeTestModel() {
    TransformerModel model;
    model.hparams.n_layer = 2;
    model.hparams.n_head_kv = 1;
    model.hparams.n_embd_head_k = 4;
    model.hparams.n_embd = 4;
    model.hparams.n_head = 1;
    model.hparams.n_ctx = 32;
    return model;
}

static TransformerModel MakeGLM5TestModel() {
    TransformerModel model = MakeTestModel();
    model.arch = ModelArch::GLM5_DSA;
    model.arch_flags.is_glm_dsa = true;
    model.hparams.n_embd_head_v = 4;
    model.glm_index_head_dim = 6;
    return model;
}

TEST_F(BlockManagerTest, AllocateSingleBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_GE(block_id, 0);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 1);
    EXPECT_EQ(free, 31);
}

TEST_F(BlockManagerTest, FreeSingleBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_GE(block_id, 0);

    manager->FreeSingle(block_id);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 0);
    EXPECT_EQ(free, 32);
}

TEST_F(BlockManagerTest, AllocateMultipleBlocks) {
    std::vector<int> blocks = manager->Allocate(5);
    EXPECT_EQ(blocks.size(), 5);

    // All block IDs should be unique
    std::set<int> unique_ids(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_ids.size(), 5);

    int used = manager->GetUsedBlockCount();
    EXPECT_EQ(used, 5);
}

TEST_F(BlockManagerTest, FreeMultipleBlocks) {
    std::vector<int> blocks = manager->Allocate(5);
    EXPECT_EQ(blocks.size(), 5);

    manager->Free(blocks);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 0);
    EXPECT_EQ(free, 32);
}

TEST_F(BlockManagerTest, AllocateAllBlocks) {
    std::vector<int> blocks = manager->Allocate(32);
    EXPECT_EQ(blocks.size(), 32);

    // All blocks allocated
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(free, 0);

    // Next single allocation should fail
    int exhausted = manager->AllocateSingle();
    EXPECT_EQ(exhausted, -1);

    // Free one block
    manager->FreeSingle(blocks[0]);

    // Now allocation should succeed
    int reclaimed = manager->AllocateSingle();
    EXPECT_GE(reclaimed, 0);
}

TEST_F(BlockManagerTest, Fork_IncrementRefCount) {
    int block_id = manager->AllocateSingle();
    EXPECT_EQ(manager->GetRefCount(block_id), 1);

    // Fork (increment ref count for CoW)
    int forked_id = manager->Fork(block_id);
    EXPECT_EQ(forked_id, block_id);
    EXPECT_EQ(manager->GetRefCount(block_id), 2);

    // IsShared should return true
    EXPECT_TRUE(manager->IsShared(block_id));
}

TEST_F(BlockManagerTest, CopyOnWrite_SharedBlock) {
    int block_id = manager->AllocateSingle();
    manager->Fork(block_id);  // Now ref_count = 2

    // CoW should allocate a new block
    // Fix for CI: Explicitly pass callback (ARM64 build failure was due to missing arg in old version)
    int cow_id = manager->CopyOnWrite(block_id, [](int, int) {});
    EXPECT_GE(cow_id, 0);
    EXPECT_NE(cow_id, block_id);  // Different block

    // Original still has ref_count = 1 (decremented from 2)
    // The CoW returns a new block with ref_count = 1
    EXPECT_EQ(manager->GetRefCount(block_id), 1);
}

// =============================================================================
// PagedKVCache Swap Persistence Tests
// =============================================================================

TEST(PagedKVCache, CopyRestoreBlocksRoundTrip) {
    TransformerModel model = MakeTestModel();
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&model, 1, 32, GGML_TYPE_F32, -1));
    ASSERT_NE(cache, nullptr);

    std::vector<int> blocks = cache->block_manager->Allocate(2);
    ASSERT_EQ(blocks.size(), 2u);

    const int elements_per_slot = cache->GetElementsPerSlot();
    std::vector<float> slot(elements_per_slot);

    for (size_t b = 0; b < blocks.size(); ++b) {
        for (int layer = 0; layer < cache->n_layer; ++layer) {
            for (int slot_idx = 0; slot_idx < BLOCK_SIZE; ++slot_idx) {
                float base = static_cast<float>(b * 1000 + layer * 100 + slot_idx);
                for (int i = 0; i < elements_per_slot; ++i) {
                    slot[i] = base + static_cast<float>(i) * 0.25f;
                }
                cache->WriteKSlot(blocks[b], layer, slot_idx, slot.data());
                for (int i = 0; i < elements_per_slot; ++i) {
                    slot[i] = base + 5000.0f + static_cast<float>(i) * 0.5f;
                }
                cache->WriteVSlot(blocks[b], layer, slot_idx, slot.data());
            }
        }
    }

    size_t bytes_per_block = cache->GetBytesPerBlock();
    size_t total_bytes = blocks.size() * static_cast<size_t>(cache->n_layer) * bytes_per_block;
    std::vector<uint8_t> expected_k(total_bytes, 0);
    std::vector<uint8_t> expected_v(total_bytes, 0);

    for (size_t b = 0; b < blocks.size(); ++b) {
        for (int layer = 0; layer < cache->n_layer; ++layer) {
            size_t offset = (b * static_cast<size_t>(cache->n_layer) + static_cast<size_t>(layer)) * bytes_per_block;
            const void* k_ptr = cache->GetKBlockPtr(blocks[b], layer);
            const void* v_ptr = cache->GetVBlockPtr(blocks[b], layer);
            ASSERT_NE(k_ptr, nullptr);
            ASSERT_NE(v_ptr, nullptr);
            std::memcpy(expected_k.data() + offset, k_ptr, bytes_per_block);
            std::memcpy(expected_v.data() + offset, v_ptr, bytes_per_block);
        }
    }

    std::vector<uint8_t> k_dump;
    std::vector<uint8_t> v_dump;
    cache->CopyBlocksToHost(blocks, &k_dump, &v_dump);
    ASSERT_EQ(k_dump.size(), total_bytes);
    ASSERT_EQ(v_dump.size(), total_bytes);
    EXPECT_EQ(std::memcmp(k_dump.data(), expected_k.data(), total_bytes), 0);
    EXPECT_EQ(std::memcmp(v_dump.data(), expected_v.data(), total_bytes), 0);

    for (size_t b = 0; b < blocks.size(); ++b) {
        for (int layer = 0; layer < cache->n_layer; ++layer) {
            void* k_ptr = cache->GetKBlockPtr(blocks[b], layer);
            void* v_ptr = cache->GetVBlockPtr(blocks[b], layer);
            std::memset(k_ptr, 0, bytes_per_block);
            std::memset(v_ptr, 0, bytes_per_block);
        }
    }

    cache->RestoreBlocksFromHost(blocks, k_dump, v_dump);

    for (size_t b = 0; b < blocks.size(); ++b) {
        for (int layer = 0; layer < cache->n_layer; ++layer) {
            size_t offset = (b * static_cast<size_t>(cache->n_layer) + static_cast<size_t>(layer)) * bytes_per_block;
            const void* k_ptr = cache->GetKBlockPtr(blocks[b], layer);
            const void* v_ptr = cache->GetVBlockPtr(blocks[b], layer);
            ASSERT_NE(k_ptr, nullptr);
            ASSERT_NE(v_ptr, nullptr);
            EXPECT_EQ(std::memcmp(expected_k.data() + offset, k_ptr, bytes_per_block), 0);
            EXPECT_EQ(std::memcmp(expected_v.data() + offset, v_ptr, bytes_per_block), 0);
        }
    }
}

TEST(PagedKVCache, CopyRestoreBlocksRoundTrip_WithGLM5IndexerCache) {
    TransformerModel model = MakeGLM5TestModel();
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&model, 1, 32, GGML_TYPE_F16, -1));
    ASSERT_NE(cache, nullptr);
    ASSERT_TRUE(cache->has_index_cache);

    std::vector<int> blocks = cache->block_manager->Allocate(1);
    ASSERT_EQ(blocks.size(), 1u);

    const int k_elements_per_slot = cache->GetElementsPerSlot();
    const int v_elements_per_slot = cache->GetVElementsPerSlot();
    std::vector<float> k_slot(k_elements_per_slot);
    std::vector<float> v_slot(v_elements_per_slot);
    std::vector<float> index_slot(static_cast<size_t>(cache->index_head_dim));

    for (int layer = 0; layer < cache->n_layer; ++layer) {
        for (int slot_idx = 0; slot_idx < BLOCK_SIZE; ++slot_idx) {
            for (int i = 0; i < k_elements_per_slot; ++i) {
                k_slot[i] = static_cast<float>(1000 + layer * 100 + slot_idx * 10 + i);
            }
            for (int i = 0; i < v_elements_per_slot; ++i) {
                v_slot[i] = static_cast<float>(2000 + layer * 100 + slot_idx * 10 + i);
            }
            for (int i = 0; i < cache->index_head_dim; ++i) {
                index_slot[static_cast<size_t>(i)] = static_cast<float>(100 + layer * 10 + slot_idx + i);
            }
            cache->WriteKSlot(blocks[0], layer, slot_idx, k_slot.data());
            cache->WriteVSlot(blocks[0], layer, slot_idx, v_slot.data());
            cache->WriteIndexSlot(blocks[0], layer, slot_idx, index_slot.data());
        }
    }

    std::vector<uint8_t> k_dump;
    std::vector<uint8_t> v_dump;
    cache->CopyBlocksToHost(blocks, &k_dump, &v_dump);
    ASSERT_GT(k_dump.size(), 0u);
    ASSERT_GT(v_dump.size(), 0u);

    const size_t k_bytes = cache->GetBytesPerBlock() * static_cast<size_t>(cache->n_layer) * blocks.size();
    const size_t index_bytes = cache->GetIndexBytesPerBlock() * static_cast<size_t>(cache->n_layer) * blocks.size();
    ASSERT_EQ(k_dump.size(), k_bytes + index_bytes);

    for (int layer = 0; layer < cache->n_layer; ++layer) {
        std::memset(cache->GetKBlockPtr(blocks[0], layer), 0, cache->GetBytesPerBlock());
        std::memset(cache->GetVBlockPtr(blocks[0], layer), 0, cache->GetVBytesPerBlock());
        std::memset(cache->GetIndexBlockPtr(blocks[0], layer), 0, cache->GetIndexBytesPerBlock());
    }

    cache->RestoreBlocksFromHost(blocks, k_dump, v_dump);

    std::vector<float> index_read(static_cast<size_t>(cache->index_head_dim), 0.0f);
    for (int layer = 0; layer < cache->n_layer; ++layer) {
        for (int slot_idx = 0; slot_idx < BLOCK_SIZE; ++slot_idx) {
            cache->ReadIndexSlot(blocks[0], layer, slot_idx, index_read.data());
            for (int i = 0; i < cache->index_head_dim; ++i) {
                const float expected = static_cast<float>(100 + layer * 10 + slot_idx + i);
                EXPECT_NEAR(index_read[static_cast<size_t>(i)], expected, 1e-2f);
            }
        }
    }
}

TEST_F(BlockManagerTest, CopyOnWrite_UniqueBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_EQ(manager->GetRefCount(block_id), 1);

    // CoW on unique block should return same block
    int cow_id = manager->CopyOnWrite(block_id, [](int, int) {});
    EXPECT_EQ(cow_id, block_id);
}

TEST_F(BlockManagerTest, SlotManagement) {
    int block_id = manager->AllocateSingle();

    EXPECT_EQ(manager->GetFilledSlots(block_id), 0);

    manager->SetFilledSlots(block_id, 10);
    EXPECT_EQ(manager->GetFilledSlots(block_id), 10);

    manager->SetFilledSlots(block_id, BLOCK_SIZE);
    EXPECT_EQ(manager->GetFilledSlots(block_id), BLOCK_SIZE);
}

TEST_F(BlockManagerTest, ComputeTokenHash) {
    std::vector<int> tokens1 = {1, 2, 3, 4};
    std::vector<int> tokens2 = {1, 2, 3, 4};
    std::vector<int> tokens3 = {1, 2, 3, 5};

    uint64_t hash1 = BlockManager::ComputeTokenHash(tokens1.data(), tokens1.size());
    uint64_t hash2 = BlockManager::ComputeTokenHash(tokens2.data(), tokens2.size());
    uint64_t hash3 = BlockManager::ComputeTokenHash(tokens3.data(), tokens3.size());

    EXPECT_EQ(hash1, hash2);  // Same tokens -> same hash
    EXPECT_NE(hash1, hash3);  // Different tokens -> different hash
}

TEST_F(BlockManagerTest, PrefixCaching) {
    int block_id = manager->AllocateSingle();
    uint64_t hash = 0x12345678;

    // Register block in prefix cache
    manager->RegisterPrefixBlock(block_id, hash);

    // Should find the cached block
    int cached_id = manager->FindCachedBlock(hash);
    EXPECT_EQ(cached_id, block_id);

    // Unregister and verify not found
    manager->UnregisterPrefixBlock(block_id);
    int not_found = manager->FindCachedBlock(hash);
    EXPECT_EQ(not_found, -1);
}

// =============================================================================
// Prefix Caching with Multi-Stage Collision Verification E2E Test
// =============================================================================
TEST_F(BlockManagerTest, PrefixCachingWithVerification) {
    // Test tokens for prefix caching
    std::vector<int> tokens = {100, 200, 300, 400, 500, 600, 700, 800};
    int n_tokens = static_cast<int>(tokens.size());

    // Allocate block for tokens
    int block_id = manager->AllocateSingle();
    ASSERT_GE(block_id, 0);
    EXPECT_EQ(manager->GetRefCount(block_id), 1);

    // Compute hash and register with tokens (simulates prefill completion)
    uint64_t hash = BlockManager::ComputeTokenHash(tokens.data(), n_tokens);
    manager->RegisterPrefixBlockWithTokens(block_id, hash, tokens.data(), n_tokens);

    // Verify cache hit with same tokens (simulates new request with same prompt)
    int found = manager->FindCachedBlockWithVerification(hash, tokens.data(), n_tokens);
    EXPECT_EQ(found, block_id);

    // Verify ref_count increased (Fork semantics for cache hit)
    EXPECT_EQ(manager->GetRefCount(block_id), 2);

    // Verify collision detection: different tokens with same hash should not match
    std::vector<int> different_tokens = {100, 200, 300, 400, 500, 600, 700, 999};
    int not_found = manager->FindCachedBlockWithVerification(hash, different_tokens.data(),
                                                             static_cast<int>(different_tokens.size()));
    EXPECT_EQ(not_found, -1);  // Should not match due to token verification stage

    // Verify length mismatch rejection (fast-path)
    std::vector<int> shorter_tokens = {100, 200, 300, 400};
    int not_found_short =
        manager->FindCachedBlockWithVerification(hash, shorter_tokens.data(), static_cast<int>(shorter_tokens.size()));
    EXPECT_EQ(not_found_short, -1);  // Should not match due to length check

    // Cleanup: free the block (ref_count goes down)
    manager->FreeSingle(block_id);
    EXPECT_EQ(manager->GetRefCount(block_id), 1);  // Still 1 from first ForFind
    manager->FreeSingle(block_id);
    EXPECT_EQ(manager->GetRefCount(block_id), 0);  // Now fully freed
}

// =============================================================================
// BLOCK_SIZE constant test
// =============================================================================

TEST(KVCache, BlockSizeConstant) {
    EXPECT_EQ(BLOCK_SIZE, 16);  // Verify expected block size
}

TEST(PagedKVCache, InitRejectsInvalidDimensions) {
    TransformerModel model = MakeTestModel();

    std::unique_ptr<PagedKVCache> zero_seqs(InitPagedKVCache(&model, 0, 32, GGML_TYPE_F16, -1));
    EXPECT_EQ(zero_seqs, nullptr);

    std::unique_ptr<PagedKVCache> zero_len(InitPagedKVCache(&model, 1, 0, GGML_TYPE_F16, -1));
    EXPECT_EQ(zero_len, nullptr);
}

TEST(PagedKVCache, Q40RoundTripReadWrite) {
    TransformerModel model = MakeTestModel();
    model.hparams.n_embd_head_k = 64;
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&model, 1, 32, GGML_TYPE_Q4_0, -1));
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->cache_type, GGML_TYPE_Q4_0);

    std::vector<int> blocks = cache->block_manager->Allocate(1);
    ASSERT_EQ(blocks.size(), 1u);

    const int elements = cache->GetElementsPerSlot();
    std::vector<float> input(elements, 0.0f);
    std::vector<float> output(elements, 0.0f);
    for (int i = 0; i < elements; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.25f);
    }

    cache->WriteKSlot(blocks[0], 0, 0, input.data());
    cache->ReadKSlot(blocks[0], 0, 0, output.data());

    for (int i = 0; i < elements; ++i) {
        EXPECT_NEAR(output[i], input[i], 0.35f) << "i=" << i;
    }
}

TEST(PagedKVCache, QuantizedCacheFallsBackWhenHeadDimMisaligned) {
    TransformerModel model = MakeTestModel();
    model.hparams.n_embd_head_k = 48;  // Not divisible by Q4_0/Q8_0 block size.

    std::unique_ptr<PagedKVCache> q4_cache(InitPagedKVCache(&model, 1, 32, GGML_TYPE_Q4_0, -1));
    ASSERT_NE(q4_cache, nullptr);
    EXPECT_EQ(q4_cache->cache_type, GGML_TYPE_F16);

    std::unique_ptr<PagedKVCache> q8_cache(InitPagedKVCache(&model, 1, 32, GGML_TYPE_Q8_0, -1));
    ASSERT_NE(q8_cache, nullptr);
    EXPECT_EQ(q8_cache->cache_type, GGML_TYPE_F16);
}
