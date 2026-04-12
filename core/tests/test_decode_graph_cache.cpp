#include <gtest/gtest.h>

#include "model_types.h"
#include "worker_internal.h"

TEST(DecodeGraphCachePolicyTest, HybridSSMModelsRemainCacheEligible) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
}

TEST(DecodeGraphCachePolicyTest, HybridSSMModelsRequireRuntimeRebind) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    EXPECT_TRUE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
}

TEST(DecodeGraphCachePolicyTest, DenseTransformerModelsDoNotRequireRuntimeRebind) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.arch_flags.is_hybrid_ssm = false;

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
    EXPECT_FALSE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
}
