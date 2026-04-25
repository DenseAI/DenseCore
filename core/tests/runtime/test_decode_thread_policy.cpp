#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "runtime/worker_internal.h"

namespace {

using densecore::simd::SimdLevel;

class ScopedEnvVar {
  public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        Set(value);
    }

    ~ScopedEnvVar() {
        if (had_prev_) {
            Set(prev_value_.c_str());
        } else {
            Set(nullptr);
        }
    }

  private:
    void Set(const char* value) {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

TEST(DecodeThreadPolicy, AVX2SingleSequenceDoesNotGrabAllCores) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(2, 64, 64, SimdLevel::AVX2), 8);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(4, 64, 64, SimdLevel::AVX2), 16);
}

TEST(DecodeThreadPolicy, AVX2StillScalesUpOnHighCoreHosts) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX2), 32);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(16, 64, 64, SimdLevel::AVX2), 48);
}

TEST(DecodeThreadPolicy, WideSimdGetsLargerPerSequenceBudget) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 64, 64, SimdLevel::AVX512), 8);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 64, SimdLevel::AVX512), 64);
}

TEST(DecodeThreadPolicy, RespectsConfiguredBaseThreadCap) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX512), 24);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(8, 64, 24, SimdLevel::AVX2), 24);
}

TEST(DecodeThreadPolicy, SmallHostsKeepAvailableThreads) {
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 4, 4, SimdLevel::AVX2), 4);
    EXPECT_EQ(ResolveAutoDecodeThreadsForBatchWithSimd(1, 2, 2, SimdLevel::AVX2), 2);
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesShortPromptProfileOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 8);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_short_prompt");
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesMediumPromptProfileOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 96, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 12);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_medium_prompt");
}

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesLongPromptProfileOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 256, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_long_prompt");
}

TEST(DecodeThreadPolicy, Qwen36DenseSingleDecodeUsesC4SweetSpotOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 10);
    EXPECT_STREQ(selection.label, "decode_qwen36_dense27_c4_sweet_spot");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesC4SweetSpotOnWideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 8);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4_sweet_spot");
}

TEST(DecodeThreadPolicy, Qwen36SingleDecodeEnvOverrideStillWins) {
    ScopedEnvVar decode_override("DENSECORE_QWEN36_SINGLE_DECODE_THREADS", "11");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 11);
    EXPECT_STREQ(selection.label, "decode_qwen36_single_env_override");
}

TEST(DecodeThreadPolicy, NonQwen36SingleDecodeKeepsGenericAutoPolicy) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 64, 64, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 8);
    EXPECT_STREQ(selection.label, "decode_batch_auto");
}

TEST(DecodeThreadPolicy, LongHybridSsmPromptBypassesSingleRequestFastPath) {
    ScopedEnvVar disable_override("DENSECORE_DISABLE_SINGLE_REQ_FAST_PATH_FOR_LONG_HYBRID_SSM", nullptr);
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens.resize(BLOCK_SIZE + 1, 1);

    EXPECT_TRUE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

TEST(DecodeThreadPolicy, HybridSsmDecodeReentryBypassesSingleRequestFastPath) {
    ScopedEnvVar disable_override("DENSECORE_DISABLE_SINGLE_REQ_FAST_PATH_FOR_LONG_HYBRID_SSM", nullptr);
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens = {1};
    req.n_past = 1024;

    EXPECT_TRUE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

TEST(DecodeThreadPolicy, ShortFreshHybridSsmPromptKeepsSingleRequestFastPathEligible) {
    ScopedEnvVar disable_override("DENSECORE_DISABLE_SINGLE_REQ_FAST_PATH_FOR_LONG_HYBRID_SSM", nullptr);
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    Request req{};
    req.tokens.resize(std::max(1, BLOCK_SIZE / 2), 1);

    EXPECT_FALSE(ShouldBypassSingleRequestFastPathForLongHybridSSM(&model, &req));
}

}  // namespace
