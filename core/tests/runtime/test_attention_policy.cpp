#include <gtest/gtest.h>

#include "llm/attention/internal.h"

namespace densecore {
namespace testing {
extern int ResolveQuantBatchedTileColsForTest(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k);
}  // namespace testing
namespace llm::attention::testing {
extern void ResetSharedPrefillFlashMaskBuildsForTest();
extern uint64_t GetSharedPrefillFlashMaskBuildsForTest();
extern void ExerciseSharedPrefillFlashMaskBuildForTest(bool native_flash_selected, int n_total_tokens, int n_tokens,
                                                       int attn_query_base_pos, bool decode_only_batch,
                                                       int fast_attn_sliding_window);
}  // namespace llm::attention::testing
}  // namespace densecore

namespace {

TransformerModel MakeModel(ModelArch arch = ModelArch::LLAMA, bool gemma4 = false) {
    TransformerModel model{};
    model.arch = arch;
    model.arch_flags.is_gemma4 = gemma4;
    model.hparams.n_embd = 1024;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    return model;
}

BatchSpec MakeDecodeBatch(int num_seqs, int n_past) {
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

}  // namespace

TEST(AttentionPolicyTest, Gemma4SoftcapDefaultsToFifty) {
    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    EXPECT_FLOAT_EQ(densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(&gemma4), 50.0f);
}

TEST(AttentionPolicyTest, PagedFallbackReasonNameMatchesPolicyOff) {
    EXPECT_STREQ(densecore::llm::attention::DecodePagedFallbackReasonName(
                     densecore::llm::attention::DecodePagedFallbackReason::PolicyOff),
                 "policy_off");
}

TEST(AttentionPolicyTest, BaseDecisionRejectsInvalidDecodeLayout) {
    TransformerModel llama = MakeModel();
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();
    BatchSpec batch = MakeDecodeBatch(2, BLOCK_SIZE);
    batch.seq_id[1] = 0;

    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(llama.hparams.n_head);
    const int n_head_kv = static_cast<int>(llama.hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(llama.hparams.n_embd) / n_head;
    const int head_dim_kv = head_dim_q;

    const auto decision = densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
        policy, &llama, &cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv);

    EXPECT_FALSE(decision.use_paged_decode_attention);
    EXPECT_EQ(decision.paged_decode_decision.reason,
              densecore::llm::attention::DecodePagedFallbackReason::NonDecodeOnlyLayout);
}

TEST(AttentionPolicyTest, DispatchDecisionPrefersPortableCpuFlashForSafeDecodeOnCpu) {
    TransformerModel llama = MakeModel();
    densecore::llm::attention::BasePagedDecodeExecutionDecision base{};
    base.paged_decode_decision.candidate = true;
    base.paged_decode_decision.requested = false;
    base.paged_decode_supported = true;
    base.use_paged_decode_attention = false;

    const auto decision = densecore::llm::attention::ResolveDecodeAttentionDispatchDecision(
        &llama, base, /*use_cache=*/true, /*layer_idx=*/0, /*n_tokens_in_batch=*/1, /*n_past_val=*/64, /*n_head=*/16,
        /*n_head_kv=*/8, /*head_dim_q=*/64, /*head_dim_kv=*/64, /*head_dim_v=*/64,
        densecore::DeviceType::CPU, /*flash_attention_disabled=*/false, /*flash_attention_forced=*/false,
        /*flash_attention_isa_supported=*/false, /*portable_cpu_flash_attention_enabled=*/true,
        /*ops_registry_initialized=*/true, /*force_safe_gqa_decode_enabled=*/true,
        /*debug_disable_fast_attn_from_layer=*/-1, /*qcur_contiguous=*/true, /*k_contiguous=*/true,
        /*v_contiguous=*/true, /*single_seq_layout=*/true);

    EXPECT_TRUE(decision.force_safe_gqa_decode);
    EXPECT_TRUE(decision.prefer_portable_cpu_flash_safe_decode);
    EXPECT_TRUE(decision.use_portable_cpu_flash_attention);
    EXPECT_FALSE(decision.use_paged_decode_attention);
    EXPECT_EQ(decision.attention_path_kind, densecore::llm::attention::DecodeAttentionPathKind::PortableCpuFlash);
}

TEST(AttentionPolicyTest, DispatchDecisionUsesHalForNonCpuPreferredDevice) {
    TransformerModel llama = MakeModel();
    densecore::llm::attention::BasePagedDecodeExecutionDecision base{};
    base.use_paged_decode_attention = false;

    const auto decision = densecore::llm::attention::ResolveDecodeAttentionDispatchDecision(
        &llama, base, /*use_cache=*/true, /*layer_idx=*/0, /*n_tokens_in_batch=*/2, /*n_past_val=*/0, /*n_head=*/16,
        /*n_head_kv=*/8, /*head_dim_q=*/64, /*head_dim_kv=*/64, /*head_dim_v=*/64,
        densecore::DeviceType::METAL, /*flash_attention_disabled=*/false, /*flash_attention_forced=*/false,
        /*flash_attention_isa_supported=*/true, /*portable_cpu_flash_attention_enabled=*/true,
        /*ops_registry_initialized=*/true, /*force_safe_gqa_decode_enabled=*/true,
        /*debug_disable_fast_attn_from_layer=*/-1, /*qcur_contiguous=*/false, /*k_contiguous=*/false,
        /*v_contiguous=*/false, /*single_seq_layout=*/false);

    EXPECT_TRUE(decision.use_hal_attention_dispatch);
    EXPECT_EQ(decision.attention_path_kind, densecore::llm::attention::DecodeAttentionPathKind::Hal);
}

TEST(AttentionPolicyTest, Qwen3AutoPolicyRequestsPagedDecodeForShortInteractiveContext) {
    TransformerModel qwen3 = MakeModel(ModelArch::QWEN3);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();
    policy.mode = densecore::llm::config::DecodePagedAttentionMode::Auto;
    policy.min_context_tokens = 128;
    policy.min_batched_context_tokens = 64;

    BatchSpec batch = MakeDecodeBatch(1, 27);
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(qwen3.hparams.n_head);
    const int n_head_kv = static_cast<int>(qwen3.hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(qwen3.hparams.n_embd) / n_head;

    const auto decision = densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
        policy, &qwen3, &cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_q);

    EXPECT_TRUE(decision.paged_decode_decision.candidate);
    EXPECT_TRUE(decision.paged_decode_decision.requested);
    EXPECT_TRUE(decision.use_paged_decode_attention);
}

TEST(AttentionPolicyTest, Qwen36AutoPolicyRequestsPagedDecodeForShortInteractiveContext) {
    TransformerModel qwen36 = MakeModel(ModelArch::QWEN35);
    qwen36.variant = ModelVariant::QWEN36;
    qwen36.arch_flags.is_hybrid_ssm = true;
    qwen36.hparams.n_embd = 2048;
    qwen36.hparams.n_head = 16;
    qwen36.hparams.n_head_kv = 2;

    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();
    policy.mode = densecore::llm::config::DecodePagedAttentionMode::Auto;
    policy.min_context_tokens = 128;
    policy.min_batched_context_tokens = 64;

    BatchSpec batch = MakeDecodeBatch(1, 27);
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(qwen36.hparams.n_head);
    const int n_head_kv = static_cast<int>(qwen36.hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(qwen36.hparams.n_embd) / n_head;

    const auto decision = densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
        policy, &qwen36, &cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_q);

    EXPECT_TRUE(decision.paged_decode_decision.candidate);
    EXPECT_TRUE(decision.paged_decode_decision.requested);
    EXPECT_TRUE(decision.use_paged_decode_attention);
}

TEST(AttentionPolicyTest, QuantTileClampsToNativeVecDotLaneWidthWhenTrueBatchedUnavailable) {
    EXPECT_EQ(densecore::testing::ResolveQuantBatchedTileColsForTest(/*requested_cols=*/16, /*vec_dot_nrows=*/2,
                                                                     /*allow_true_batched_q4k=*/false),
              2);
    EXPECT_EQ(densecore::testing::ResolveQuantBatchedTileColsForTest(/*requested_cols=*/12, /*vec_dot_nrows=*/4,
                                                                     /*allow_true_batched_q4k=*/false),
              4);
}

TEST(AttentionPolicyTest, QuantTileKeepsRequestedWidthWhenTrueBatchedPathExists) {
    EXPECT_EQ(densecore::testing::ResolveQuantBatchedTileColsForTest(/*requested_cols=*/16, /*vec_dot_nrows=*/2,
                                                                     /*allow_true_batched_q4k=*/true),
              16);
}

TEST(AttentionPolicyTest, SharedPrefillFlashMaskBuildsOnlyWhenNativeFlashPathRuns) {
    densecore::llm::attention::testing::ResetSharedPrefillFlashMaskBuildsForTest();
    densecore::llm::attention::testing::ExerciseSharedPrefillFlashMaskBuildForTest(
        /*native_flash_selected=*/false, /*n_total_tokens=*/32, /*n_tokens=*/8, /*attn_query_base_pos=*/24,
        /*decode_only_batch=*/false, /*fast_attn_sliding_window=*/-1);
    EXPECT_EQ(densecore::llm::attention::testing::GetSharedPrefillFlashMaskBuildsForTest(), 0u);

    densecore::llm::attention::testing::ResetSharedPrefillFlashMaskBuildsForTest();
    densecore::llm::attention::testing::ExerciseSharedPrefillFlashMaskBuildForTest(
        /*native_flash_selected=*/true, /*n_total_tokens=*/32, /*n_tokens=*/8, /*attn_query_base_pos=*/24,
        /*decode_only_batch=*/false, /*fast_attn_sliding_window=*/-1);
    EXPECT_EQ(densecore::llm::attention::testing::GetSharedPrefillFlashMaskBuildsForTest(), 1u);
}
