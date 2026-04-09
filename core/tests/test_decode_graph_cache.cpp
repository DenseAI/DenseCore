#include <gtest/gtest.h>

#include "model_types.h"
#include "worker_internal.h"

TEST(DecodeGraphCachePolicyTest, RejectsHybridSSMModels) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    EXPECT_FALSE(IsDecodeGraphCacheSafeForModel(&model));
}

TEST(DecodeGraphCachePolicyTest, KeepsDenseTransformerModelsEligible) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.arch_flags.is_hybrid_ssm = false;

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
}
