#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "densecore/runtime/inference.h"
#include "densecore/models/model_types.h"
#include "models/model_inference_policy.h"
#include "densecore/memory/kv_cache.h"
#include "runtime/runtime_env.h"
#include "runtime/worker_internal.h"

namespace densecore::testing {
extern bool ShouldUsePagedDecodeAttentionForBatchTest(const TransformerModel* model, const PagedKVCache* cache,
                                                      const BatchSpec& batch);
}

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

TEST(DecodeGraphCachePolicyTest, Gemma4BatchedDecodeTopologyIsStableByDefault) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/2, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&gemma4, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, Gemma4InferenceDecisionUsesPagedDecodeByDefault) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "on");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&gemma4));
}

TEST(DecodeGraphCachePolicyTest, Gemma4InferenceDecisionCanFallBackToDenseBaselineEscapeHatch) {
    const TransformerModel gemma4 = MakeDecodeModel(true);
    ScopedEnvOverride dense_baseline_env("DENSECORE_GEMMA4_FORCE_DENSE_BASELINE", "1");
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "on");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_FALSE(ShouldUsePagedDecodeAttentionForTestBatch(&gemma4));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaBatchedDecodeTopologyRemainsStablePagedDecode) {
    const TransformerModel llama = MakeDecodeModel(false);
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/2, /*n_past=*/BLOCK_SIZE);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&llama, &cache, batch));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionRespectsExplicitPolicyOff) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "off");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_FALSE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionRespectsAutoModeDecline) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "auto");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "9999");

    EXPECT_FALSE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(DecodeGraphCachePolicyTest, NonGemmaInferenceDecisionUsesPagedDecodeWhenPolicyRequestsIt) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "on");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", nullptr);

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama));
}

TEST(RuntimeEnvTest, ParseTruthyEnvTreatsUnknownPresentValueAsFalse) {
    ScopedEnvOverride mode_env("DENSECORE_TEST_TRUTHY_ENV", "maybe");
    EXPECT_FALSE(densecore::env::ParseTruthyEnv("DENSECORE_TEST_TRUTHY_ENV", true));
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

TEST(DecodeGraphCachePolicyTest, AutoModeAllowsShortBatchedDecodeWithSeparateThreshold) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "auto");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(ShouldUsePagedDecodeAttentionForTestBatch(&llama, /*num_seqs=*/2, /*n_past=*/63));
}

TEST(DecodeGraphCachePolicyTest, AutoModeKeepsShortSingleDecodeOnStandardPath) {
    const TransformerModel llama = MakeDecodeModel(false);
    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "auto");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_FALSE(ShouldUsePagedDecodeAttentionForTestBatch(&llama, /*num_seqs=*/1, /*n_past=*/63));
}

TEST(DecodeGraphCachePolicyTest, Qwen36SingleDecodeTopologyIsCacheStableWithoutForcedPagedDecode) {
    const TransformerModel qwen36 = MakeQwen36HybridDecodeModel();
    const BatchSpec batch = MakeDecodeOnlyBatch(/*num_seqs=*/1, /*n_past=*/63);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    ScopedEnvOverride force_env("DENSECORE_FORCE_PAGED_DECODE", nullptr);
    ScopedEnvOverride legacy_env("DENSECORE_ENABLE_PAGED_ATTN_DECODE", nullptr);
    ScopedEnvOverride mode_env("DENSECORE_PAGED_ATTN_DECODE_MODE", "auto");
    ScopedEnvOverride min_ctx_env("DENSECORE_PAGED_DECODE_MIN_CONTEXT", "128");
    ScopedEnvOverride min_batch_ctx_env("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", "64");

    EXPECT_TRUE(IsStablePagedDecodeTopologyForCache(&qwen36, &cache, batch));
}
