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

TEST(DecodeThreadPolicy, Qwen36SinglePrefillUsesFullCoreProfileOnC4A) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;

    const PrefillThreadPolicySelection selection =
        ResolvePrefillThreadPolicySelection(&model, 1, 32, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "prefill_qwen36_single_c4a_full_core");
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
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_safe_cap");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesSafeCapOnX86WideSimd) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::AVX512);

    EXPECT_EQ(selection.threads, 8);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_safe_cap");
}

TEST(DecodeThreadPolicy, Qwen36A3BSingleDecodeUsesC4AWideSimdPolicyOnArm) {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    const DecodeThreadPolicySelection selection =
        ResolveDecodeThreadPolicySelection(&model, 1, 16, 16, SimdLevel::SVE2);

    EXPECT_EQ(selection.threads, 16);
    EXPECT_STREQ(selection.label, "decode_qwen36_a3b_c4a_moe_16");
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

TEST(DecodeThreadPolicy, Qwen36PrefillChunkEnvFeedsSchedulerAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "256");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 520;
    req.prompt_tokens_for_cache.resize(520, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 256);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkZeroKeepsAutoLongPromptAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", "512");
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 520;
    req.prompt_tokens_for_cache.resize(520, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 192);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoKeepsOneKPromptsUnchunked) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1024;
    req.prompt_tokens_for_cache.resize(1024, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkAutoStillChunksVeryLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 192);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkDefaultTokensEnvTunesAutoAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS", "512");
    ScopedEnvVar default_tokens("DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS", "768");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), 768);
}

TEST(DecodeThreadPolicy, Qwen36PrefillChunkOffDisablesLongPromptAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", "off");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 520;
    req.prompt_tokens_for_cache.resize(520, 1);

    EXPECT_EQ(ResolveQwen36PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkAutoChunksLongPrompts) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);
    ScopedEnvVar default_tokens("DENSECORE_GEMMA4_PREFILL_CHUNK_DEFAULT_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 1536;
    req.prompt_tokens_for_cache.resize(1536, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), 256);
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkAutoKeepsShortPromptsUnchunked) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "0");
    ScopedEnvVar auto_min("DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS", nullptr);

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 512;
    req.prompt_tokens_for_cache.resize(512, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), -1);
}

TEST(DecodeThreadPolicy, Gemma4MoEPrefillChunkEnvTunesSchedulerAdmission) {
    ScopedEnvVar chunk_override("DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS", "384");

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_experts = 128;

    Request req{};
    req.prompt_token_count = 520;
    req.prompt_tokens_for_cache.resize(520, 1);

    EXPECT_EQ(ResolveGemma4PrefillChunkTokens(&model, &req), 384);
}

}  // namespace
