#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "densecore/runtime/inference.h"
#include "densecore/exceptions.h"
#include "ggml-cpu.h"
#include "densecore/models/model_types.h"
#include "kernels/q4k_repacked_gemv.h"
#include "models/model_inference_policy.h"
#include "densecore/memory/kv_cache.h"
#include "runtime/runtime_env.h"
#include "runtime/worker_internal.h"
#include "llm/graph/cache.h"

namespace densecore::testing {
extern bool ShouldUsePagedDecodeAttentionForBatchTest(const TransformerModel* model, const PagedKVCache* cache,
                                                      const BatchSpec& batch);
extern int ResolvePagedAttentionDecodeHeadTileForTest(int n_head, int n_tokens, int n_tasks);
extern uint64_t HashQwen36Q4KBatchedAdmissionKeyForTest(const TransformerModel* model, const ggml_tensor* weight,
                                                        const ggml_tensor* input, int M, int N, int K);
extern void StoreQwen36Q4KBatchedAdmissionForTest(uint64_t key, bool pass, float max_abs_error, const char* reason);
extern int LookupQwen36Q4KBatchedAdmissionForTest(uint64_t key);
extern void DowngradeQwen36Q4KBatchedAdmissionForTest(uint64_t key, float max_abs_error);
extern bool Q4KRepackedGemvEnabledForTest(densecore::env::RuntimeToggleMode mode, int* reject_reason);
extern bool QActCacheSharedDataDifferentTensorMissesForTest();
extern bool QActCacheSameTensorDifferentTokenOrSlotMissesForTest(bool change_token_pos);
extern bool QActCacheResetAcrossCachedDecodeReuseForTest();
extern bool QActBatchedCacheReusesSameTensorForTest();
extern bool QActCacheKeepsMultipleTensorEntriesForTest();
extern bool RunQwen36Q4KBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle, int* admission_state,
                                             int* reject_reason);
extern bool RunQwen36Q4KBatchedDirectDispatchCensusForTest(int nth,
                                                           std::vector<MatmulDispatchCensusEntry>* dispatch_entries,
                                                           bool* output_matches_vecdot_oracle);
extern bool RunQwen36SSMQ8RepackedBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle);
extern bool RunQwen36SSMQ8RepackedBatchedC4ProjectionForTest(int nth, bool* output_matches_vecdot_oracle,
                                                             uint64_t* true_gemm_ops, uint64_t* gemv_ops);
extern int ResolveQwen36PrefillQ4KBatchedReasonForTest(bool relevant, bool mode_off, bool lora_active,
                                                       bool weight_is_q4k, bool shape_supported,
                                                       bool kernel_available, bool has_vec_dot, bool candidate_ready,
                                                       bool mode_on, bool mode_probe, int admission_state);
extern const char* Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(int reason);
extern int ResolveQwen36SSMQ8PrefillAMXReasonForTest(int mode, int phase, bool lora_active);
extern bool ShouldUseQwenHybridSSMQ8RepackedBatchedForTest(bool relevant, int tokens);
extern bool ShouldUseQwenHybridSSMQ8DirectBatchedForTest(bool relevant, int tokens);
}

namespace {

TransformerModel MakeDecodeModel(bool gemma4) {
    TransformerModel model{};
    model.arch = gemma4 ? ModelArch::GEMMA : ModelArch::LLAMA;
    model.arch_flags.is_gemma4 = gemma4;
    model.hparams.n_embd = 1024;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    return model;
}

ggml_tensor* NewTensor2D(ggml_context* ctx, int rows, int cols) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
}

TransformerModel MakeQwen36HybridDecodeModel() {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    return model;
}

TransformerModel MakeQwen35HybridDecodeModel() {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    return model;
}

TransformerModel MakeLFM2ShortConvDecodeModel() {
    TransformerModel model{};
    model.arch = ModelArch::LFM2;
    model.variant = ModelVariant::LFM2MOE;
    model.arch_flags.is_hybrid_ssm = true;
    model.arch_flags.is_lfm2_shortconv = true;
    model.hparams.n_embd = 4096;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    return model;
}

BatchSpec MakeDecodeOnlyBatch(int num_seqs, int n_past) {
    BatchSpec batch{};
    batch.num_seqs = num_seqs;
    batch.tokens.assign(static_cast<size_t>(num_seqs), 1);
    batch.seq_id.resize(static_cast<size_t>(num_seqs));
    batch.pos.resize(static_cast<size_t>(num_seqs));
    batch.block_tables.resize(static_cast<size_t>(num_seqs));
    batch.n_past.assign(static_cast<size_t>(num_seqs), n_past);
    const int blocks_per_seq = std::max(1, (n_past + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int next_block_id = 0;
    for (int i = 0; i < num_seqs; ++i) {
        batch.seq_id[static_cast<size_t>(i)] = i;
        batch.pos[static_cast<size_t>(i)] = n_past;
        auto& block_table = batch.block_tables[static_cast<size_t>(i)];
        block_table.resize(static_cast<size_t>(blocks_per_seq));
        for (int block = 0; block < blocks_per_seq; ++block) {
            block_table[static_cast<size_t>(block)] = next_block_id++;
        }
    }
    return batch;
}

class ScopedEnvOverride {
public:
    ScopedEnvOverride(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        Set(value);
    }

    ~ScopedEnvOverride() {
        if (had_prev_) {
            Set(prev_value_.c_str());
        } else {
            Set(nullptr);
        }
    }

private:
    void Set(const char* value) {
#ifdef _WIN32
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

}  // namespace

TEST(DecodeGraphCachePolicyTest, ServerBatchingDefaultsUseSixteenSeqCacheWindow) {
    EXPECT_GE(DecodeGraphCacheMaxBatch(), 16);
    EXPECT_GE(DecodeGraphCacheLruSize(), 32);
    EXPECT_GE(DecodeGraphCacheCtxBytes(), 192ULL * 1024ULL * 1024ULL);
}

TEST(DecodeGraphCachePolicyTest, AdaptivePagedDecodeHeadTileKeepsSingleTokenBusy) {
    ScopedEnvOverride tile_env("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", nullptr);
    EXPECT_LE(densecore::testing::ResolvePagedAttentionDecodeHeadTileForTest(32, 1, 16), 2);
    EXPECT_GE(densecore::testing::ResolvePagedAttentionDecodeHeadTileForTest(8, 1, 16), 1);
}

TEST(DecodeGraphCachePolicyTest, PagedDecodeHeadTileEnvOverrideWins) {
    ScopedEnvOverride tile_env("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", "8");
    EXPECT_EQ(densecore::testing::ResolvePagedAttentionDecodeHeadTileForTest(32, 1, 16), 8);
}

TEST(DecodeGraphCachePolicyTest, Qwen36PrefillQ4KAdmissionKeySeparatesShapeAndTensor) {
    ggml_init_params params{16 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);
    TransformerModel model = MakeQwen36HybridDecodeModel();
    ggml_tensor* w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, QK_K, 4);
    ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, QK_K, 4);
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK_K, 2);
    ASSERT_NE(w0, nullptr);
    ASSERT_NE(w1, nullptr);
    ASSERT_NE(input, nullptr);
    const uint64_t key0 = densecore::testing::HashQwen36Q4KBatchedAdmissionKeyForTest(&model, w0, input, 2, 4, QK_K);
    const uint64_t key1 = densecore::testing::HashQwen36Q4KBatchedAdmissionKeyForTest(&model, w1, input, 2, 4, QK_K);
    const uint64_t key2 = densecore::testing::HashQwen36Q4KBatchedAdmissionKeyForTest(&model, w0, input, 3, 4, QK_K);
    EXPECT_NE(key0, key1);
    EXPECT_NE(key0, key2);
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key0, false, 0.25f, "probe_mismatch");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key0), 2);
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key0, true, 0.0f, "pass");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key0), 2);
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key1), 0);
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key1, true, 0.0f, "pass");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key1), 1);
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key1, false, 0.5f, "probe_mismatch");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key1), 2);
    ggml_free(ctx);
}

TEST(DecodeGraphCachePolicyTest, Qwen36PrefillQ4KRuntimeFailureDowngradesAdmittedKey) {
    constexpr uint64_t key = 0x5157333651344bULL;
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key, true, 0.0f, "pass");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key), 1);
    densecore::testing::DowngradeQwen36Q4KBatchedAdmissionForTest(key, 1.0f);
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key), 2);
    densecore::testing::StoreQwen36Q4KBatchedAdmissionForTest(key, true, 0.0f, "pass");
    EXPECT_EQ(densecore::testing::LookupQwen36Q4KBatchedAdmissionForTest(key), 2);
}

TEST(DecodeGraphCachePolicyTest, Q4KRepackedGateAdmitsWhenRealKernelIsAvailable) {
    int reject_reason = 0;
    EXPECT_FALSE(densecore::testing::Q4KRepackedGemvEnabledForTest(densecore::env::RuntimeToggleMode::Off,
                                                                   &reject_reason));
    EXPECT_EQ(reject_reason, 1);
    const bool available = densecore::kernels::Q4KRealPackedGemvKernelAvailable();
    EXPECT_EQ(densecore::testing::Q4KRepackedGemvEnabledForTest(densecore::env::RuntimeToggleMode::On,
                                                                &reject_reason),
              available);
    EXPECT_EQ(reject_reason == 0, available);
}

TEST(DecodeGraphCachePolicyTest, Q4KRepackedColdFloorDoesNotReplaceAutomaticBudget) {
    using namespace densecore::kernels;
    Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();
    constexpr size_t kFloorBytes = 1;
    if (Q4KRepackedGemvAutoCacheLimitBytes(0) <= kFloorBytes) {
        GTEST_SKIP() << "No automatic cache headroom on this host";
    }
    // A fresh long prefill raises a working-set floor before the first cache
    // lookup. It must retain the automatic memory budget, just as a warmed
    // process does, rather than turning the lower bound into a hard limit.
    const size_t limit = Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(kFloorBytes);
    EXPECT_GT(limit, kFloorBytes);
    EXPECT_EQ(Q4KRepackedGemvRuntimeCacheBudgetFloorBytes(), kFloorBytes);
    EXPECT_EQ(Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(kFloorBytes), limit);
    Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();
}

TEST(DecodeGraphCachePolicyTest, Q4KRepackedCacheFloorSurvivesRuntimeBudgetRefresh) {
    densecore::kernels::Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();
    constexpr size_t kFloorBytes = 768ULL * 1024ULL * 1024ULL;
    EXPECT_GE(densecore::kernels::Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(kFloorBytes), kFloorBytes);
    EXPECT_GE(densecore::kernels::Q4KRepackedGemvRefreshRuntimeCacheBudget(
                  std::numeric_limits<size_t>::max() / 4),
              kFloorBytes);
    EXPECT_GE(densecore::kernels::Q4KRepackedGemvCacheLimitBytes(), kFloorBytes);
    EXPECT_GE(densecore::kernels::Q4KRepackedGemvRuntimeCacheBudgetFloorBytes(), kFloorBytes);
    densecore::kernels::Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();
}

TEST(DecodeGraphCachePolicyTest, QActCacheDoesNotReuseWhenDifferentTensorsShareDataPointer) {
    EXPECT_TRUE(densecore::testing::QActCacheSharedDataDifferentTensorMissesForTest());
}

TEST(DecodeGraphCachePolicyTest, QActCacheMissesWhenTokenPositionChanges) {
    EXPECT_TRUE(densecore::testing::QActCacheSameTensorDifferentTokenOrSlotMissesForTest(/*change_token_pos=*/true));
}

TEST(DecodeGraphCachePolicyTest, QActCacheMissesWhenSlotIdChanges) {
    EXPECT_TRUE(densecore::testing::QActCacheSameTensorDifferentTokenOrSlotMissesForTest(/*change_token_pos=*/false));
}

TEST(DecodeGraphCachePolicyTest, QActCacheResetsAcrossCachedDecodeGraphReuse) {
    EXPECT_TRUE(densecore::testing::QActCacheResetAcrossCachedDecodeReuseForTest());
}

TEST(DecodeGraphCachePolicyTest, QActBatchedCacheReusesSameTensorWithinExecution) {
    EXPECT_TRUE(densecore::testing::QActBatchedCacheReusesSameTensorForTest());
}

TEST(DecodeGraphCachePolicyTest, QActCacheKeepsMultipleTensorEntriesWithinExecution) {
    EXPECT_TRUE(densecore::testing::QActCacheKeepsMultipleTensorEntriesForTest());
}

TEST(DecodeGraphCachePolicyTest, Qwen36PrefillQ4KReasonDoesNotAdmitInvalidCandidates) {
    constexpr int kPass = 1;
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, true, true, true, true, true, true, true, false, kPass),
              2);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, false, true, true, true, true, false, kPass),
              3);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, true, false, true, true, true, false, kPass),
              5);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, true, true, false, true, true, false, kPass),
              4);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, false, false, true, true, true, true, false, kPass),
              10);
}

TEST(DecodeGraphCachePolicyTest, Qwen36PrefillQ4KReasonAdmitsOnlyValidOnOrPassedProbe) {
    constexpr int kUnknown = 0;
    constexpr int kPass = 1;
    constexpr int kReject = 2;
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, true, true, true, true, true, false, kUnknown),
              8);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, true, true, true, true, false, true, kPass),
              8);
    EXPECT_EQ(densecore::testing::ResolveQwen36PrefillQ4KBatchedReasonForTest(
                  true, false, false, true, true, true, true, true, false, true, kReject),
              9);
}

TEST(DecodeGraphCachePolicyTest, Qwen36SSMQ8PrefillAMXRejectReasonsAreStableStrings) {
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(1), "env_off");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(6), "dynamic_lora");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(9), "probe_unavailable");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(10), "admitted");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(11), "decode_original_q8");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(12),
                 "resident_decode_regression_risk");
    EXPECT_STREQ(densecore::testing::Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(13), "phase_unknown");
}

TEST(DecodeGraphCachePolicyTest, Qwen36SSMQ8PrefillAMXAdmissionRequiresExplicitPrefillPhase) {
    constexpr int kModeOn = 2;
    constexpr int kUnknown = 0;
    constexpr int kPrefill = 1;
    constexpr int kDecode = 2;
    EXPECT_EQ(densecore::testing::ResolveQwen36SSMQ8PrefillAMXReasonForTest(kModeOn, kUnknown, false), 13);
    EXPECT_EQ(densecore::testing::ResolveQwen36SSMQ8PrefillAMXReasonForTest(kModeOn, kDecode, false), 3);
    EXPECT_EQ(densecore::testing::ResolveQwen36SSMQ8PrefillAMXReasonForTest(kModeOn, kPrefill, true), 6);
    EXPECT_EQ(densecore::testing::ResolveQwen36SSMQ8PrefillAMXReasonForTest(kModeOn, kPrefill, false), 0);
}

TEST(DecodeGraphCachePolicyTest, Qwen36DirectBatchedPathPublishesKernelOutput) {
    bool output_matches_vecdot_oracle = false;
    int admission_state = 0;
    int reject_reason = 0;
    ASSERT_TRUE(densecore::testing::RunQwen36Q4KBatchedDirectForTest(
        /*nth=*/2, &output_matches_vecdot_oracle, &admission_state, &reject_reason));
    EXPECT_TRUE(output_matches_vecdot_oracle);
    EXPECT_EQ(admission_state, 0);
    EXPECT_EQ(reject_reason, 0);
}

TEST(DecodeGraphCachePolicyTest, Qwen36DirectBatchedPathDoesNotShadowRejectPartitions) {
    bool output_matches_vecdot_oracle = false;
    int admission_state = 0;
    int reject_reason = 0;
    ASSERT_TRUE(densecore::testing::RunQwen36Q4KBatchedDirectForTest(
        /*nth=*/2, &output_matches_vecdot_oracle, &admission_state, &reject_reason));
    EXPECT_TRUE(output_matches_vecdot_oracle);
    EXPECT_EQ(admission_state, 0);
    EXPECT_EQ(reject_reason, 0);
}

TEST(DecodeGraphCachePolicyTest, Qwen36DirectBatchedPathPublishesRuntimeDispatchCensus) {
    ScopedEnvOverride enable_census("DENSECORE_MATMUL_DISPATCH_CENSUS", "1");
    bool output_matches_vecdot_oracle = false;
    std::vector<MatmulDispatchCensusEntry> dispatch_entries;
    ASSERT_TRUE(densecore::testing::RunQwen36Q4KBatchedDirectDispatchCensusForTest(
        /*nth=*/2, &dispatch_entries, &output_matches_vecdot_oracle));
    ASSERT_TRUE(output_matches_vecdot_oracle);
    ASSERT_FALSE(dispatch_entries.empty());
    const auto it = std::find_if(dispatch_entries.begin(), dispatch_entries.end(),
                                 [](const auto& entry) { return entry.dispatch_path == "q4k_q8k_true_batched"; });
    ASSERT_NE(it, dispatch_entries.end());
    EXPECT_EQ(it->phase, "decode");
    EXPECT_EQ(it->shape_bucket, "M=2,N=8,K=256");
    EXPECT_GT(it->wall_ns, 0u);
    EXPECT_GT(it->ops, 0u);
}

TEST(DecodeGraphCachePolicyTest, Qwen36SSMQ8RepackedBatchedPathMatchesVecDotOracle) {
    bool output_matches_vecdot_oracle = false;
    ASSERT_TRUE(densecore::testing::RunQwen36SSMQ8RepackedBatchedDirectForTest(
        /*nth=*/4, &output_matches_vecdot_oracle));
    EXPECT_TRUE(output_matches_vecdot_oracle);
}

TEST(DecodeGraphCachePolicyTest, QwenHybridSSMQ8UsesGemmFromFourTokens) {
    EXPECT_FALSE(densecore::testing::ShouldUseQwenHybridSSMQ8RepackedBatchedForTest(true, 1));
    EXPECT_TRUE(densecore::testing::ShouldUseQwenHybridSSMQ8DirectBatchedForTest(true, 1));
    EXPECT_TRUE(densecore::testing::ShouldUseQwenHybridSSMQ8RepackedBatchedForTest(true, 4));
    EXPECT_FALSE(densecore::testing::ShouldUseQwenHybridSSMQ8DirectBatchedForTest(true, 4));
    EXPECT_FALSE(densecore::testing::ShouldUseQwenHybridSSMQ8RepackedBatchedForTest(false, 192));
}

TEST(DecodeGraphCachePolicyTest, Qwen36SSMQ8C4ProjectionUsesArchitectureSpecificKernel) {
    bool output_matches_vecdot_oracle = false;
    uint64_t true_gemm_ops = 0;
    uint64_t gemv_ops = 0;
    ASSERT_TRUE(densecore::testing::RunQwen36SSMQ8RepackedBatchedC4ProjectionForTest(
        /*nth=*/4, &output_matches_vecdot_oracle, &true_gemm_ops, &gemv_ops));
    EXPECT_TRUE(output_matches_vecdot_oracle);
#if defined(__aarch64__) || defined(_M_ARM64)
    // This 2048-wide fixture is below the maintained ARM GEMM threshold.
    EXPECT_EQ(true_gemm_ops, 0u);
    EXPECT_GT(gemv_ops, 0u);
#else
    EXPECT_GT(true_gemm_ops, 0u);
    EXPECT_EQ(gemv_ops, 0u);
#endif
}

TEST(DecodeGraphCachePolicyTest, UnqualifiedHybridSSMModelsAreNotDecodeGraphCacheSafeYet) {
    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.variant = ModelVariant::UNKNOWN;
    model.arch_flags.is_hybrid_ssm = true;

    EXPECT_FALSE(IsDecodeGraphCacheSafeForModel(&model));
}

TEST(DecodeGraphCachePolicyTest, Qwen35HybridSSMSingleDecodeCanUseDecodeGraphCache) {
    const TransformerModel model = MakeQwen35HybridDecodeModel();

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
    EXPECT_TRUE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
}

TEST(DecodeGraphCachePolicyTest, Qwen36HybridSSMSingleDecodeCanUseDecodeGraphCache) {
    const TransformerModel model = MakeQwen36HybridDecodeModel();

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
    EXPECT_TRUE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
}

TEST(DecodeGraphCachePolicyTest, LFM2ShortConvSingleDecodeCanUseDecodeGraphCache) {
    const TransformerModel model = MakeLFM2ShortConvDecodeModel();

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
    EXPECT_TRUE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
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

TEST(DecodeGraphCachePolicyTest, Gemma4ModelsUseDecodeGraphCacheOncePagedDecodeIsDefault) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    ScopedEnvOverride gemma_cache_env("DENSECORE_GEMMA4_DECODE_GRAPH_CACHE", nullptr);
    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
    EXPECT_FALSE(DoesDecodeGraphCacheRequireRuntimeRebind(&model));
}

TEST(DecodeGraphCachePolicyTest, Gemma4PagedDecodeAdmitsGraphCacheWithoutEnvOverride) {
    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;

    EXPECT_TRUE(IsDecodeGraphCacheSafeForModel(&model));
}

namespace {

bool ShouldUsePagedDecodeAttentionForTestBatch(const TransformerModel* model, int num_seqs = 2,
                                               int n_past = BLOCK_SIZE) {
    const BatchSpec batch = MakeDecodeOnlyBatch(num_seqs, n_past);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(model->hparams.n_head);
    const int n_head_kv = static_cast<int>(model->hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(model->hparams.n_embd) / n_head;
    const int head_dim_kv = head_dim_q;
    EXPECT_TRUE(IsPagedDecodeCandidate(&cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv));
    return densecore::testing::ShouldUsePagedDecodeAttentionForBatchTest(model, &cache, batch);
}

}  // namespace

TEST(DecodeGraphCachePolicyTest, Gemma4PagedDecodeSupportIsEnabledByDefault) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    const TransformerModel llama = MakeDecodeModel(false);

    EXPECT_TRUE(densecore::models::SupportsPagedDecodeAttention(&gemma4));
    EXPECT_TRUE(densecore::models::SupportsPagedDecodeAttention(&llama));
}

TEST(DecodeGraphCachePolicyTest, Gemma4PagedDecodeSupportIgnoresLegacyDisableEnv) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    ScopedEnvOverride gemma_paged_env("DENSECORE_GEMMA4_ENABLE_PAGED_DECODE", "0");
    ScopedEnvOverride dense_baseline_env("DENSECORE_GEMMA4_FORCE_DENSE_BASELINE", "1");

    EXPECT_TRUE(densecore::models::SupportsPagedDecodeAttention(&gemma4));
}

TEST(DecodeGraphCachePolicyTest, Gemma4BatchedDecodeTopologyIsStableByDefault) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/2, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&gemma4, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, Gemma4BatchedDecodeTopologyUsesMaintainedPagedDecode) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/2, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&gemma4, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, Gemma4LayerHeadKvPatternCanStillUseStablePagedDecodeCache) {
    TransformerModel gemma4 = MakeDecodeModel(true);
    gemma4.gemma4_layer_n_head_kv = {4, 8, 4, 8};
    gemma4.hparams.n_embd_head_k = 512;
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/1, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&gemma4, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, Gemma4InferenceDecisionUsesPagedDecodeByDefault) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&gemma4));
}

TEST(DecodeGraphCachePolicyTest, Gemma4InferenceDecisionUsesMaintainedPagedDecodeDespiteLegacyEnv) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    ScopedEnvOverride gemma_paged_env("DENSECORE_GEMMA4_ENABLE_PAGED_DECODE", "0");
    ScopedEnvOverride dense_baseline_env("DENSECORE_GEMMA4_FORCE_DENSE_BASELINE", "1");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&gemma4));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaBatchedDecodeTopologyRemainsStablePagedDecode) {
    const TransformerModel llama = MakeDecodeModel(false);
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/2, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&llama, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionKeepsPagedDecodeOn) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionDoesNotUseEnvModeDecline) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "9999");

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionUsesPagedDecodeWhenPolicyRequestsIt) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(RuntimeEnvTest, ParseBoolEnvKeepsDefaultForUnknownPresentValue) {
    ScopedEnvOverride mode_env("DENSECORE_TEST_TRUTHY_ENV", "maybe");
    EXPECT_TRUE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", true));
    EXPECT_FALSE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", false));
}

TEST(RuntimeEnvTest, ParseBoolEnvDisablesOnEveryFalseSpelling) {
    for (const char* spelling : {"0", "false", "FALSE", "no", "off", "Off"}) {
        ScopedEnvOverride env("DENSECORE_TEST_TRUTHY_ENV", spelling);
        EXPECT_FALSE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", true))
            << "spelling: " << spelling;
    }
}

TEST(RuntimeEnvTest, ParseBoolEnvEnablesOnEveryTrueSpelling) {
    for (const char* spelling : {"1", "true", "TRUE", "yes", "on", "On"}) {
        ScopedEnvOverride env("DENSECORE_TEST_TRUTHY_ENV", spelling);
        EXPECT_TRUE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", false))
            << "spelling: " << spelling;
    }
}

TEST(RuntimeEnvTest, ParseBoolEnvUsesDefaultWhenUnsetOrEmpty) {
    ScopedEnvOverride unset_env("DENSECORE_TEST_TRUTHY_ENV", nullptr);
    EXPECT_TRUE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", true));
    EXPECT_FALSE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", false));

    ScopedEnvOverride empty_env("DENSECORE_TEST_TRUTHY_ENV", "");
    EXPECT_TRUE(densecore::env::ParseBoolEnv("DENSECORE_TEST_TRUTHY_ENV", true));
}

TEST(RuntimeEnvTest, ParseRuntimeToggleModeNormalizesCommonAliases) {
    ScopedEnvOverride on_env("DENSECORE_TEST_TOGGLE_ENV", "FORCE");
    EXPECT_EQ(densecore::env::ParseRuntimeToggleMode("DENSECORE_TEST_TOGGLE_ENV",
                                                     densecore::env::RuntimeToggleMode::Auto),
              densecore::env::RuntimeToggleMode::On);
}

TEST(RuntimeEnvTest, ParsePositiveEnvIntRejectsZeroAndGarbage) {
    ScopedEnvOverride zero_env("DENSECORE_TEST_INT_ENV", "0");
    EXPECT_EQ(densecore::env::ParsePositiveEnvInt("DENSECORE_TEST_INT_ENV", 17), 17);

    ScopedEnvOverride garbage_env("DENSECORE_TEST_INT_ENV", "abc");
    EXPECT_EQ(densecore::env::ParsePositiveEnvInt("DENSECORE_TEST_INT_ENV", 23), 23);
}

TEST(DecodeGraphCachePolicyTest, PagedDecodeAllowsShortBatchedDecode) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama, /*num_seqs=*/2, /*n_past=*/63));
}

TEST(DecodeGraphCachePolicyTest, PagedDecodeAllowsShortSingleDecode) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama, /*num_seqs=*/1, /*n_past=*/63));
}

TEST(DecodeGraphCachePolicyTest, QwenHybridSingleDecodeTopologyIsCacheStableWithoutForcedPagedDecode) {
    TransformerModel qwen36 = MakeQwen36HybridDecodeModel();
    qwen36.hparams.n_head_kv = 2;
    qwen36.hparams.n_embd_head_k = 256;
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/1, /*n_past=*/63);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&qwen36, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, LFM2ShortConvSingleDecodeTopologyIsCacheStableWithoutForcedPagedDecode) {
    TransformerModel lfm2 = MakeLFM2ShortConvDecodeModel();
    lfm2.hparams.n_head_kv = 2;
    lfm2.hparams.n_embd_head_k = 256;
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/1, /*n_past=*/63);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&lfm2, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, Qwen35SingleDecodeTopologyUsesAttentionWeightHeadDims) {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel qwen35 = MakeQwen35HybridDecodeModel();
    qwen35.hparams.n_embd = 128;
    qwen35.hparams.n_head_kv = 2;
    qwen35.hparams.n_embd_head_k = 16;
    qwen35.layers.resize(1);
    qwen35.layers[0].Set(model_keys::kAttnQWeight, NewTensor2D(ctx, 128, 256));
    qwen35.layers[0].Set(model_keys::kAttnKWeight, NewTensor2D(ctx, 128, 16));

    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/1, /*n_past=*/63);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&qwen35, &cache, batch));

    ggml_free(ctx);
}

namespace densecore::testing {
bool ArmQwen36LargeQ8PrefillAdmittedForTest(bool relevant, int tokens, int reduction, int rows);
bool RunQwenQ8PrefillGraphForTest(int tokens, bool fused, ModelVariant variant, bool* owned_callback,
                                 bool* output_matches, uint64_t* true_gemm_ops);
}

TEST(DecodeGraphCachePolicyTest, Qwen36LargeQ8PrefillAdmissionKeepsSmallAndUnrelatedShapes) {
    using densecore::testing::ArmQwen36LargeQ8PrefillAdmittedForTest;
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(false, 390, 2048, 8192));
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, 127, 2048, 8192));
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, 390, 1024, 8192));
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, 390, 2049, 8192));
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, 390, 2048, 8193));
    EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, 4097, 2048, 8192));
    for (const int tokens : {128, 129, 390}) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
        EXPECT_TRUE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, tokens, 2048, 8192));
#else
        EXPECT_FALSE(ArmQwen36LargeQ8PrefillAdmittedForTest(true, tokens, 2048, 8192));
#endif
    }
}

TEST(DecodeGraphCachePolicyTest, Qwen36LargeQ8PrefillGraphMatchesMaintainedGgmlWithPackedTails) {
    for (const bool fused : {false, true}) {
        for (const int tokens : {128, 129, 390}) {
            bool owned_callback = false, matches = false;
            uint64_t true_gemm_ops = 0;
            ASSERT_TRUE(densecore::testing::RunQwenQ8PrefillGraphForTest(
                tokens, fused, ModelVariant::QWEN36, &owned_callback, &matches, &true_gemm_ops));
            EXPECT_TRUE(matches) << "tokens=" << tokens << " fused=" << fused;
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8)
            EXPECT_TRUE(owned_callback) << "tokens=" << tokens << " fused=" << fused;
            EXPECT_GT(true_gemm_ops, 0u) << "tokens=" << tokens << " fused=" << fused;
#else
            EXPECT_FALSE(owned_callback);
            EXPECT_EQ(true_gemm_ops, 0u);
#endif
        }
    }
}

TEST(DecodeGraphCachePolicyTest, QwenLargeQ8PrefillKeepsShortAndOtherModelGraphRoutes) {
    for (const auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36}) {
        const int tokens = variant == ModelVariant::QWEN35 ? 128 : 4;
        bool owned_callback = false, matches = false;
        uint64_t true_gemm_ops = 0;
        ASSERT_TRUE(densecore::testing::RunQwenQ8PrefillGraphForTest(
            tokens, true, variant, &owned_callback, &matches, &true_gemm_ops));
        EXPECT_TRUE(matches);
        EXPECT_FALSE(owned_callback);
        EXPECT_EQ(true_gemm_ops, 0u);
    }
}

namespace densecore::testing {
bool BatchedDecodeRowMajorPolicyForTest(ModelVariant variant, int phase, int tokens, int override_value);
bool RunQwenRowMajorGraphForTest(int tokens, bool fused, ModelVariant variant, int phase,
                                bool* matches, uint64_t* row_major_ops);
}

TEST(DecodeGraphCachePolicyTest, RowMajorDefaultIsC4AQwen36DecodeM4WithExplicitOverrides) {
    const int decode = static_cast<int>(InferenceExecutionPhase::Decode);
    const int prefill = static_cast<int>(InferenceExecutionPhase::Prefill);
    for (const auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36, ModelVariant::QWEN38}) {
        for (int phase : {decode, prefill}) for (int tokens : {1, 2, 3, 4}) {
            bool expected = false;
#if defined(DENSECORE_TARGET_C4A) && (defined(__aarch64__) || defined(_M_ARM64))
            expected = variant == ModelVariant::QWEN36 && phase == decode && tokens == 4;
#endif
            EXPECT_EQ(densecore::testing::BatchedDecodeRowMajorPolicyForTest(variant, phase, tokens, -1), expected);
            EXPECT_FALSE(densecore::testing::BatchedDecodeRowMajorPolicyForTest(variant, phase, tokens, 0));
            EXPECT_TRUE(densecore::testing::BatchedDecodeRowMajorPolicyForTest(variant, phase, tokens, 1));
        }
    }
}

TEST(DecodeGraphCachePolicyTest, RowMajorDefaultGraphMatchesGgmlAndRetainsControlRoutes) {
    if (std::getenv("DENSECORE_BATCHED_DECODE_ROW_MAJOR")) GTEST_SKIP() << "Default-route test needs no override";
    const int decode = static_cast<int>(InferenceExecutionPhase::Decode);
    const int prefill = static_cast<int>(InferenceExecutionPhase::Prefill);
    for (const auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36, ModelVariant::QWEN38}) {
        for (bool fused : {false, true}) for (int phase : {decode, prefill}) for (int tokens : {1, 4}) {
            SCOPED_TRACE(::testing::Message() << "variant=" << static_cast<int>(variant)
                         << " phase=" << phase << " tokens=" << tokens << " fused=" << fused);
            bool matches = false;
            uint64_t ops = 0;
            if (variant == ModelVariant::QWEN38 && phase == prefill && tokens == 4) {
                // This fixture has no maintained Qwen38 AMX alias. Its Q8
                // prefill route is intentionally rejected on every architecture.
                try {
                    densecore::testing::RunQwenRowMajorGraphForTest(tokens, fused, variant, phase, &matches, &ops);
                    ADD_FAILURE() << "Qwen38 Q8 prefill must retain its fail-closed policy";
                } catch (const densecore::InvalidArgumentException& error) {
                    EXPECT_NE(std::string(error.what()).find("maintained_qwen_hybrid_ssm_q8_native_prefill"),
                              std::string::npos);
                }
                continue;
            }
            ASSERT_TRUE(densecore::testing::RunQwenRowMajorGraphForTest(tokens, fused, variant, phase, &matches, &ops));
            EXPECT_TRUE(matches);
            bool expected = false;
#if defined(DENSECORE_TARGET_C4A) && (defined(__aarch64__) || defined(_M_ARM64))
            expected = variant == ModelVariant::QWEN36 && phase == decode && tokens == 4;
#endif
            EXPECT_EQ(ops, expected ? 1u : 0u);
        }
    }
}

namespace {
thread_local int released_cache_contexts = 0;
void CountCacheContextRelease(InferenceWorkContext* context) {
    ++released_cache_contexts;
    DestroyInferenceWorkContext(context);
}

densecore::llm::graph::DecodeGraphCacheEntry MakeOwnedCacheEntry() {
    densecore::llm::graph::DecodeGraphCacheEntry entry;
    entry.ctx = ggml_init({4096, nullptr, true});
    entry.work_ctx = {CreateInferenceWorkContext(), CountCacheContextRelease};
    return entry;
}
}  // namespace

TEST(GraphCacheOwnershipTest, MoveAndRepeatedResetReleaseEachContextOnce) {
    released_cache_contexts = 0;
    auto first = MakeOwnedCacheEntry();
    ASSERT_NE(first.ctx, nullptr);
    auto* original = first.ctx;
    auto second = std::move(first);
    EXPECT_EQ(first.ctx, nullptr);
    EXPECT_EQ(second.ctx, original);
    auto third = MakeOwnedCacheEntry();
    third = std::move(second);
    EXPECT_EQ(released_cache_contexts, 1);
    EXPECT_EQ(second.ctx, nullptr);
    EXPECT_EQ(third.ctx, original);
    third.Reset();
    third.Reset();
    EXPECT_EQ(released_cache_contexts, 2);
    EXPECT_EQ(third.ctx, nullptr);
    EXPECT_EQ(third.work_ctx, nullptr);
}

TEST(GraphCacheOwnershipTest, ExceptionUnwindReleasesCachedAndUninsertedEntries) {
    using namespace densecore::llm::graph;
    released_cache_contexts = 0;
    try {
        GraphCache cache;
        cache.InsertDecode(DecodeGraphCacheKey{}, MakeOwnedCacheEntry(), 2);
        auto pending = MakeOwnedCacheEntry();
        throw std::runtime_error("graph construction failed");
    } catch (const std::runtime_error&) {}
    EXPECT_EQ(released_cache_contexts, 2);
}

TEST(GraphCacheOwnershipTest, TouchPreservesEntriesAndClearResetsRejectionAndBudgetState) {
    using namespace densecore::llm::graph;
    GraphCache cache;
    DecodeGraphCacheKey first_key{};
    DecodeGraphCacheKey second_key{};
    second_key.stateful_request_identity = 7;
    auto& first = cache.InsertDecode(first_key, MakeOwnedCacheEntry(), 2);
    cache.InsertDecode(second_key, MakeOwnedCacheEntry(), 2);
    cache.Touch(&first);
    cache.MakeDecodeRoom(2);
    EXPECT_NE(cache.FindDecode(first_key), nullptr);
    EXPECT_EQ(cache.FindDecode(second_key), nullptr);
    cache.MarkUncacheable(second_key, 2);
    PrefillGraphCacheEntry prefill;
    prefill.ctx_bytes = 1024;
    cache.InsertPrefill({}, std::move(prefill), 2, 2048);
    cache.Clear();
    EXPECT_EQ(cache.DecodeSize(), 0u);
    EXPECT_EQ(cache.UncacheableSize(), 0u);
    EXPECT_EQ(cache.PrefillBytes(), 0u);
}

TEST(GraphCacheOwnershipTest, ReusableArenaResetsMetadataAndInvalidBackendPreservesOwnership) {
    using namespace densecore::llm::graph;
    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    {
        ScopedGgmlGraphArena arena;
        ASSERT_TRUE(arena.Ensure(4096, backend));
        auto* context = arena.ctx();
        ASSERT_NE(ggml_new_tensor_1d(context, GGML_TYPE_F32, 1), nullptr);
        ASSERT_TRUE(arena.Ensure(4096, backend));
        EXPECT_EQ(arena.ctx(), context);
        EXPECT_EQ(ggml_get_first_tensor(context), nullptr);
        EXPECT_FALSE(arena.Ensure(4096, nullptr));
        EXPECT_EQ(arena.ctx(), context);
        arena.Reset();
        arena.Reset();
        EXPECT_EQ(arena.ctx(), nullptr);
    }
    ggml_backend_free(backend);
}

TEST(GraphCacheOwnershipTest, PrefillEvictionEnforcesCountAndByteBudgetsAfterTouch) {
    using namespace densecore::llm::graph;
    GraphCache cache;
    PrefillGraphCacheKey a{}, b{}, c{};
    b.n_past = 1;
    c.n_past = 2;
    auto insert = [&](PrefillGraphCacheKey key, size_t bytes) -> PrefillGraphCacheEntry& {
        PrefillGraphCacheEntry entry;
        entry.ctx_bytes = bytes;
        return cache.InsertPrefill(key, std::move(entry), 2, 100);
    };
    auto& first = insert(a, 40);
    insert(b, 40);
    cache.Touch(&first);
    insert(c, 50);
    EXPECT_NE(cache.FindPrefill(a), nullptr);
    EXPECT_EQ(cache.FindPrefill(b), nullptr);
    EXPECT_NE(cache.FindPrefill(c), nullptr);
    EXPECT_EQ(cache.PrefillBytes(), 90u);
    cache.ErasePrefill(a);
    cache.ErasePrefill(a);
    EXPECT_EQ(cache.PrefillBytes(), 50u);
    EXPECT_EQ(cache.PrefillSize(), 1u);
}

TEST(GraphCacheOwnershipTest, InvalidBudgetDoesNotEvictAndReplacingKeyAccountsOnce) {
    using namespace densecore::llm::graph;
    GraphCache cache;
    PrefillGraphCacheEntry first;
    first.ctx_bytes = 40;
    cache.InsertPrefill({}, std::move(first), 2, 100);
    PrefillGraphCacheEntry oversized;
    oversized.ctx_bytes = 101;
    EXPECT_THROW(cache.InsertPrefill({}, std::move(oversized), 2, 100), std::invalid_argument);
    EXPECT_EQ(cache.PrefillBytes(), 40u);
    PrefillGraphCacheEntry replacement;
    replacement.ctx_bytes = 60;
    cache.InsertPrefill({}, std::move(replacement), 2, 100);
    EXPECT_EQ(cache.PrefillBytes(), 60u);
    EXPECT_EQ(cache.PrefillSize(), 1u);
    EXPECT_THROW(cache.InsertDecode({}, MakeOwnedCacheEntry(), 0), std::invalid_argument);
    EXPECT_EQ(cache.DecodeSize(), 0u);
}

TEST(GraphCacheOwnershipTest, UncacheableKeysUseBoundedInsertionOrderWithoutDuplicateRefresh) {
    using namespace densecore::llm::graph;
    GraphCache cache;
    DecodeGraphCacheKey a{}, b{}, c{};
    b.batch_size = 2;
    c.batch_size = 3;
    EXPECT_TRUE(cache.MarkUncacheable(a, 2));
    EXPECT_TRUE(cache.MarkUncacheable(b, 2));
    EXPECT_FALSE(cache.MarkUncacheable(a, 2));
    EXPECT_TRUE(cache.MarkUncacheable(c, 2));
    EXPECT_FALSE(cache.IsUncacheable(a));
    EXPECT_TRUE(cache.IsUncacheable(b));
    EXPECT_TRUE(cache.IsUncacheable(c));
    EXPECT_EQ(cache.UncacheableSize(), 2u);
}

TEST(GraphCacheOwnershipTest, ArenaAcquireReportsHitsAndRetainsMostRecentlyUsedKey) {
    using namespace densecore::llm::graph;
    GraphCache cache;
    PrefillGraphCacheKey a{}, b{}, c{};
    b.n_past = 1;
    c.n_past = 2;
    bool reused = true;
    auto* first = cache.AcquireArena(a, 4096, 2, &reused).arena.get();
    EXPECT_FALSE(reused);
    cache.AcquireArena(b, 4096, 2, &reused);
    EXPECT_FALSE(reused);
    EXPECT_EQ(cache.AcquireArena(a, 4096, 2, &reused).arena.get(), first);
    EXPECT_TRUE(reused);
    cache.AcquireArena(c, 4096, 2, &reused);
    EXPECT_FALSE(reused);
    EXPECT_EQ(cache.AcquireArena(a, 4096, 2, &reused).arena.get(), first);
    EXPECT_TRUE(reused);
    cache.AcquireArena(b, 4096, 2, &reused);
    EXPECT_FALSE(reused);
}
