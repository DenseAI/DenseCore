/**
 * @file test_kernel_context.cpp
 * @brief Unit tests for KernelContext
 */

#include <gtest/gtest.h>

#include "kernel_context.h"

using namespace densecore;

TEST(KernelContext, Initialization) {
    KernelContext ctx(1024);
    EXPECT_NE(ctx.ScratchpadData(), nullptr);
    EXPECT_EQ(ctx.ScratchpadSize(), 1024);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ctx.ScratchpadData()) % 64, 0);
}

TEST(KernelContext, Resize) {
    KernelContext ctx(1024);
    void* original_ptr = ctx.ScratchpadData();

    // Shrinking/same size shouldn't reallocate (implementation choice, but
    // allowed to) Our implementation: resize only if growing
    ctx.ResizeScratchpad(512);
    EXPECT_EQ(ctx.ScratchpadSize(), 1024);
    EXPECT_EQ(ctx.ScratchpadData(), original_ptr);

    // Growing should reallocate
    ctx.ResizeScratchpad(2048);
    EXPECT_EQ(ctx.ScratchpadSize(), 2048);
    EXPECT_NE(ctx.ScratchpadData(), original_ptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ctx.ScratchpadData()) % 64, 0);
}

TEST(KernelContext, AMXConfiguration) {
    KernelContext ctx;

    // First config
    ctx.ConfigureAMX(16, 64);

    // Second config (same) - should be no-op
    ctx.ConfigureAMX(16, 64);

    // Different config - should update
    ctx.ConfigureAMX(32, 64);
}

TEST(KernelContext, MoveSemantics) {
    KernelContext ctx1(1024);
    void* ptr1 = ctx1.ScratchpadData();

    // Move construction
    KernelContext ctx2(std::move(ctx1));
    EXPECT_EQ(ctx2.ScratchpadSize(), 1024);
    EXPECT_EQ(ctx2.ScratchpadData(), ptr1);
    EXPECT_EQ(ctx1.ScratchpadSize(), 0);  // Moved from

    // Move assignment
    KernelContext ctx3;
    ctx3 = std::move(ctx2);
    EXPECT_EQ(ctx3.ScratchpadSize(), 1024);
    EXPECT_EQ(ctx3.ScratchpadData(), ptr1);
}
