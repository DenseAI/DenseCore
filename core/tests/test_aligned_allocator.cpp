/**
 * @file test_aligned_allocator.cpp
 * @brief Unit tests for aligned memory allocator
 */

#include <gtest/gtest.h>
#include <vector>

#include "aligned_allocator.h"

using namespace densecore;

TEST(AlignedAllocator, RawAllocation) {
    void* ptr = aligned_alloc_raw(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    aligned_free(ptr);
}

TEST(AlignedAllocator, VectorAlignment) {
    // Verify that std::vector uses the allocator correctly
    std::vector<float, AlignedAllocator<float>> vec(1024);

    // Check data pointer alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % 64, 0);

    // Verify we can write/read
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(i);
    }
    EXPECT_EQ(vec[100], 100.0f);
}

TEST(AlignedAllocator, UniquePtrAlignment) {
    auto ptr = make_aligned<float>(1024);
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr.get()) % 64, 0);

    ptr[0] = 123.0f;
    EXPECT_EQ(ptr[0], 123.0f);
}

TEST(AlignedAllocator, CustomAlignment) {
    // Try a larger power-of-2 alignment (e.g. page size 4096)
    size_t large_align = 4096;
    void* ptr = aligned_alloc_raw(1024, large_align);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % large_align, 0);
    aligned_free(ptr);
}

TEST(AlignedAllocator, BadAlloc) {
    // Allocate too much memory
    EXPECT_THROW(
        { auto ptr = make_aligned<double>(static_cast<size_t>(-1) / sizeof(double)); },
        std::bad_alloc);
}
