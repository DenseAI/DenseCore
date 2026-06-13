#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/runtime/inference.h"
#include "densecore/simd/simd_ops.h"
#include "llm/attention/internal.h"
#include "runtime/runtime_env.h"

namespace densecore {
namespace testing {
extern struct ggml_tensor* SmartMulMatTest(struct ggml_context* ctx, struct ggml_tensor* weight,
                                           struct ggml_tensor* input, TransformerModel* model);
extern struct ggml_tensor* SmartMulMatWithPhaseTest(struct ggml_context* ctx, struct ggml_tensor* weight,
                                                   struct ggml_tensor* input, TransformerModel* model, int phase);
extern bool ShouldUsePrefillLastLogitsOnlyForTest(const TransformerModel* model, int num_seqs, int n_tokens);
extern int ResolveQuantBatchedTileColsForTest(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k);
extern bool ShouldUseGemma4PrefillSafeBatchedForTest(bool is_gemma4, bool prefill_phase, int m,
                                                     bool dense_prefill_allowed);
extern bool ResolveQ4KTrueBatchedKernelPolicyForTest(int simd_level, bool compiled_with_sve);
extern bool ShouldUsePortableFlashHeadSeqReferenceFallbackForTest(bool explicit_debug_reference);
extern bool CompiledWithX86Avx512ForFlashAttentionForTest();
extern bool RunGemma4SafeF32BatchedForTest(int nth, bool* output_matches_oracle, bool* output_all_finite);
extern bool RunGemma4SafeF32MsplitBatchedForTest(int nth, bool* output_matches_oracle, bool* output_all_finite);
extern bool RunGemma4SafeQ8BatchedForTest(int nth, bool* output_matches_vecdot_oracle, uint64_t* q8_batched_used_ops);
extern bool RunGemma4SafeQ4KPrefillBatchedUsesQActCacheForTest(int nth, bool* output_matches_vecdot_oracle,
                                                               uint64_t* qact_hits, uint64_t* qact_misses);
extern bool RunDenseCoreQ8RepackedGemvForTest(bool* output_matches_vecdot_oracle);
extern bool RunGemma4NativeQ8PrefillTrueGemmForTest(int nth, bool* output_matches_vecdot_oracle,
                                                    uint64_t* true_gemm_ops, uint64_t* gemv_ops);
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
        Set(had_prev_ ? prev_value_.c_str() : nullptr);
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

TransformerModel MakeGemma4LayerPolicyModel() {
    TransformerModel model = MakeModel(ModelArch::GEMMA, true);
    model.gemma4_layer_is_sliding = {1, 0};
    return model;
}

densecore::llm::attention::DecodeAttentionDispatchDecision ResolveLayerDispatch(const TransformerModel& model,
                                                                                 int layer_idx) {
    densecore::llm::attention::BasePagedDecodeExecutionDecision base{};
    base.paged_decode_supported = true;
    base.decode_only_batch = true;
    base.use_paged_decode_attention = true;
    base.paged_decode_decision.candidate = true;
    base.paged_decode_decision.requested = true;
    return densecore::llm::attention::ResolveDecodeAttentionDispatchDecision(
        &model, base, /*use_cache=*/true, layer_idx, /*n_tokens_in_batch=*/1, /*n_past_val=*/32,
        /*n_head=*/16, /*n_head_kv=*/8, /*head_dim_q=*/256, /*head_dim_kv=*/256, /*head_dim_v=*/256,
        densecore::DeviceType::CPU, /*flash_attention_disabled=*/true, /*flash_attention_forced=*/false,
        /*flash_attention_isa_supported=*/false, /*portable_cpu_flash_attention_enabled=*/false,
        /*ops_registry_initialized=*/false, /*force_safe_gqa_decode_enabled=*/false,
        /*debug_disable_fast_attn_from_layer=*/-1, /*qcur_contiguous=*/true, /*k_contiguous=*/true,
        /*v_contiguous=*/true, /*single_seq_layout=*/true);
}

}  // namespace

TEST(AttentionPolicyTest, Gemma4TextAttentionSoftcapMatchesLoadedMetadata) {
    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    gemma4.gemma4_attention_logit_softcapping = 50.0f;
    EXPECT_FLOAT_EQ(densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(&gemma4), 50.0f);
}

TEST(AttentionPolicyTest, PagedFallbackReasonNameMatchesPolicyOff) {
    EXPECT_STREQ(densecore::llm::attention::DecodePagedFallbackReasonName(
                     densecore::llm::attention::DecodePagedFallbackReason::PolicyOff),
                 "policy_off");
}

TEST(AttentionPolicyTest, Gemma4PagedDecodeUsesSlidingLayersOnly) {
    const TransformerModel gemma4 = MakeGemma4LayerPolicyModel();

    EXPECT_TRUE(ResolveLayerDispatch(gemma4, 0).use_paged_decode_attention);
    EXPECT_FALSE(ResolveLayerDispatch(gemma4, 1).use_paged_decode_attention);
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

TEST(AttentionPolicyTest, Gemma4PrefillUsesPortableCpuFlashForGqaSoftcapSemantics) {
    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    densecore::llm::attention::BasePagedDecodeExecutionDecision base{};
    base.use_paged_decode_attention = false;
    base.paged_decode_supported = true;

    const auto decision = densecore::llm::attention::ResolveDecodeAttentionDispatchDecision(
        &gemma4, base, /*use_cache=*/true, /*layer_idx=*/0, /*n_tokens_in_batch=*/22, /*n_past_val=*/0,
        /*n_head=*/16, /*n_head_kv=*/8, /*head_dim_q=*/256, /*head_dim_kv=*/256, /*head_dim_v=*/256,
        densecore::DeviceType::CPU, /*flash_attention_disabled=*/false, /*flash_attention_forced=*/false,
        /*flash_attention_isa_supported=*/false, /*portable_cpu_flash_attention_enabled=*/true,
        /*ops_registry_initialized=*/true, /*force_safe_gqa_decode_enabled=*/true,
        /*debug_disable_fast_attn_from_layer=*/-1, /*qcur_contiguous=*/false, /*k_contiguous=*/false,
        /*v_contiguous=*/false, /*single_seq_layout=*/false);

    EXPECT_TRUE(decision.use_portable_cpu_flash_attention);
    EXPECT_EQ(decision.attention_path_kind, densecore::llm::attention::DecodeAttentionPathKind::PortableCpuFlash);
}

TEST(AttentionPolicyTest, HeadSeqPortableFlashReferenceFallbackRequiresExplicitDebugOptIn) {
    EXPECT_FALSE(densecore::testing::ShouldUsePortableFlashHeadSeqReferenceFallbackForTest(false));
    EXPECT_TRUE(densecore::testing::ShouldUsePortableFlashHeadSeqReferenceFallbackForTest(true));
    (void)densecore::testing::CompiledWithX86Avx512ForFlashAttentionForTest();
}

TEST(AttentionPolicyTest, Gemma4QuantizedPrefillProjectionUsesDenseCoreBatchedPath) {
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, /*ne0=*/2816, /*ne1=*/4096);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2816, /*ne1=*/613);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatTest(ctx, weight, input, &gemma4);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);

    ggml_free(ctx);
}

TEST(AttentionPolicyTest, Gemma4F32RouterPrefillUsesCustomBatchedPath) {
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2816, /*ne1=*/128);
    ASSERT_NE(weight, nullptr);
    ggml_set_name(weight, "blk.0.ffn_gate_inp.weight");
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2816, /*ne1=*/512);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatWithPhaseTest(
        ctx, weight, input, &gemma4, static_cast<int>(InferenceExecutionPhase::Prefill));
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);

    ggml_free(ctx);
}

TEST(AttentionPolicyTest, Gemma4SafeF32RouterBatchedMatchesFiniteOracle) {
    bool matches = false;
    bool finite = false;
    ASSERT_TRUE(densecore::testing::RunGemma4SafeF32BatchedForTest(/*nth=*/4, &matches, &finite));
    EXPECT_TRUE(matches);
    EXPECT_TRUE(finite);
}

TEST(AttentionPolicyTest, Gemma4SafeF32RouterMsplitBatchedMatchesOracle) {
    bool matches = false;
    bool finite = false;
    ASSERT_TRUE(densecore::testing::RunGemma4SafeF32MsplitBatchedForTest(/*nth=*/4, &matches, &finite));
    EXPECT_TRUE(matches);
    EXPECT_TRUE(finite);
}

TEST(AttentionPolicyTest, Gemma4SafeQ8PrefillBatchedUsesRowVecDotOracle) {
    bool matches = false;
    uint64_t q8_batched_used_ops = 0;
    ASSERT_TRUE(densecore::testing::RunGemma4SafeQ8BatchedForTest(/*nth=*/4, &matches, &q8_batched_used_ops));
    EXPECT_TRUE(matches);
    EXPECT_EQ(q8_batched_used_ops, 0u);
}

TEST(AttentionPolicyTest, Gemma4SafeQ4KPrefillBatchedUsesQActCache) {
    bool matches = false;
    uint64_t qact_hits = 0;
    uint64_t qact_misses = 0;
    ASSERT_TRUE(densecore::testing::RunGemma4SafeQ4KPrefillBatchedUsesQActCacheForTest(
        /*nth=*/4, &matches, &qact_hits, &qact_misses));
    EXPECT_TRUE(matches);
    EXPECT_GE(qact_hits, 1u);
    EXPECT_EQ(qact_misses, 1u);
}

TEST(AttentionPolicyTest, DenseCoreQ8RepackedGemvMatchesVecDotOracle) {
    bool matches = false;
    ASSERT_TRUE(densecore::testing::RunDenseCoreQ8RepackedGemvForTest(&matches));
    EXPECT_TRUE(matches);
}

TEST(AttentionPolicyTest, Gemma4NativeQ8PrefillTrueGemmMatchesVecDotOracle) {
    bool matches = false;
    uint64_t true_gemm_ops = 0;
    uint64_t gemv_ops = 0;
    ASSERT_TRUE(densecore::testing::RunGemma4NativeQ8PrefillTrueGemmForTest(
        /*nth=*/4, &matches, &true_gemm_ops, &gemv_ops));
    EXPECT_TRUE(matches);
    EXPECT_GT(true_gemm_ops, 0u);
    EXPECT_EQ(gemv_ops, 0u);
}

TEST(AttentionPolicyTest, Gemma4Q8PrefillCpuRepackAliasUsesRawCustomBatchedPath) {
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    ggml_tensor* original = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, /*ne0=*/2816, /*ne1=*/1024);
    ASSERT_NE(original, nullptr);
    ggml_set_name(original, "blk.0.attn_k.weight");
    ggml_tensor* alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, /*ne0=*/2816, /*ne1=*/1024);
    ASSERT_NE(alias, nullptr);
    ggml_set_name(alias, "blk.0.attn_k.weight.cpu_repack_2d");
    gemma4.cpu_repack_aliases[original] = alias;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2816, /*ne1=*/512);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatWithPhaseTest(
        ctx, original, input, &gemma4, static_cast<int>(InferenceExecutionPhase::Prefill));
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_EQ(result->src[1], original);

    ggml_free(ctx);
}

TEST(AttentionPolicyTest, Gemma4Q8DecodeCpuRepackAliasUsesRawCustomGemvPath) {
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel gemma4 = MakeModel(ModelArch::GEMMA, true);
    ggml_tensor* original = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, /*ne0=*/2112, /*ne1=*/2816);
    ASSERT_NE(original, nullptr);
    ggml_set_name(original, "blk.0.ffn_down.weight");
    ggml_tensor* alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, /*ne0=*/2112, /*ne1=*/2816);
    ASSERT_NE(alias, nullptr);
    ggml_set_name(alias, "blk.0.ffn_down.weight.cpu_repack_2d");
    gemma4.cpu_decode_repack_aliases[original] = alias;
    gemma4.cpu_repack_aliases[original] = alias;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2112, /*ne1=*/1);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatWithPhaseTest(
        ctx, original, input, &gemma4, static_cast<int>(InferenceExecutionPhase::Decode));
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_EQ(result->src[1], original);

    ggml_free(ctx);
}

TEST(AttentionPolicyTest, Qwen35MoEPrefillCpuRepackAliasUsesRawQ4KBatchedPath) {
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    TransformerModel qwen35 = MakeModel(ModelArch::QWEN35);
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.hparams.n_experts = 128;

    ggml_tensor* original = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, /*ne0=*/2048, /*ne1=*/4096);
    ASSERT_NE(original, nullptr);
    ggml_set_name(original, "blk.0.attn_qkv.weight");
    ggml_tensor* alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, /*ne0=*/2048, /*ne1=*/4096);
    ASSERT_NE(alias, nullptr);
    ggml_set_name(alias, "blk.0.attn_qkv.weight.fast_matmul");
    qwen35.cpu_repack_aliases[original] = alias;
    qwen35.cpu_repack_alias_sources[alias] = original;

    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, /*ne0=*/2048, /*ne1=*/16);
    ASSERT_NE(input, nullptr);

    ggml_tensor* result = densecore::testing::SmartMulMatWithPhaseTest(
        ctx, alias, input, &qwen35, static_cast<int>(InferenceExecutionPhase::Prefill));
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->op, GGML_OP_CUSTOM);
    EXPECT_EQ(result->src[1], original);

    ggml_free(ctx);
}

TEST(AttentionPolicyTest, Gemma4PrefillProjectsOnlyPromptEndLogits) {
    TransformerModel gemma4_moe = MakeModel(ModelArch::GEMMA, true);
    gemma4_moe.hparams.n_experts = 128;

    EXPECT_TRUE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_moe, /*num_seqs=*/1,
                                                                          /*n_tokens=*/384));
    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_moe, /*num_seqs=*/1,
                                                                           /*n_tokens=*/1));
    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_moe, /*num_seqs=*/2,
                                                                           /*n_tokens=*/384));

    TransformerModel gemma4_dense = MakeModel(ModelArch::GEMMA, true);
    EXPECT_TRUE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_dense, /*num_seqs=*/1,
                                                                          /*n_tokens=*/384));
    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_dense, /*num_seqs=*/2,
                                                                           /*n_tokens=*/384));
}

TEST(AttentionPolicyTest, PrefillLastLogitsPolicyUsesAttachedDecoderSpecBeforeRawModelFlags) {
    TransformerModel gemma4_moe = MakeModel(ModelArch::GEMMA, true);
    gemma4_moe.hparams.n_experts = 128;

    densecore::models::DecoderModelSpec full_sequence_spec{};
    full_sequence_spec.output.prefill_logits_policy = densecore::models::DecoderPrefillLogitsPolicy::FullSequence;
    gemma4_moe.decoder_spec = std::make_shared<const densecore::models::DecoderModelSpec>(full_sequence_spec);

    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&gemma4_moe, /*num_seqs=*/1,
                                                                           /*n_tokens=*/384));

    TransformerModel dense = MakeModel(ModelArch::LLAMA);
    densecore::models::DecoderModelSpec last_token_spec{};
    last_token_spec.output.prefill_logits_policy = densecore::models::DecoderPrefillLogitsPolicy::LastTokenForMoE;
    dense.decoder_spec = std::make_shared<const densecore::models::DecoderModelSpec>(last_token_spec);

    EXPECT_TRUE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&dense, /*num_seqs=*/1, /*n_tokens=*/384));
}

TEST(AttentionPolicyTest, Qwen35PrefillLastLogitsOnlyDefaultsOn) {
    TransformerModel qwen35 = MakeModel(ModelArch::QWEN35);
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.arch_flags.is_hybrid_ssm = true;

    EXPECT_TRUE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&qwen35, /*num_seqs=*/1,
                                                                          /*n_tokens=*/384));
    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&qwen35, /*num_seqs=*/1,
                                                                           /*n_tokens=*/1));
    EXPECT_FALSE(densecore::testing::ShouldUsePrefillLastLogitsOnlyForTest(&qwen35, /*num_seqs=*/2,
                                                                           /*n_tokens=*/384));
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

TEST(AttentionPolicyTest, Qwen35AutoPolicyKeepsGenericShortContextFloor) {
    TransformerModel qwen35 = MakeModel(ModelArch::QWEN35);
    qwen35.variant = ModelVariant::QWEN35;
    qwen35.arch_flags.is_hybrid_ssm = true;
    qwen35.hparams.n_embd = 2048;
    qwen35.hparams.n_head = 16;
    qwen35.hparams.n_head_kv = 2;

    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();
    policy.mode = densecore::llm::config::DecodePagedAttentionMode::Auto;
    policy.min_context_tokens = 128;
    policy.min_batched_context_tokens = 64;

    BatchSpec batch = MakeDecodeBatch(1, 27);
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(qwen35.hparams.n_head);
    const int n_head_kv = static_cast<int>(qwen35.hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(qwen35.hparams.n_embd) / n_head;

    const auto decision = densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
        policy, &qwen35, &cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_q);

    EXPECT_TRUE(decision.paged_decode_decision.candidate);
    EXPECT_FALSE(decision.paged_decode_decision.requested);
    EXPECT_EQ(decision.paged_decode_decision.reason,
              densecore::llm::attention::DecodePagedFallbackReason::AutoContextShort);
    EXPECT_FALSE(decision.use_paged_decode_attention);
}

TEST(AttentionPolicyTest, LlamaAutoPolicyKeepsGenericShortContextFloor) {
    TransformerModel llama = MakeModel(ModelArch::LLAMA);
    PagedKVCache cache{};
    cache.cache_type = GGML_TYPE_F16;
    cache.max_blocks = 8;

    auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();
    policy.mode = densecore::llm::config::DecodePagedAttentionMode::Auto;
    policy.min_context_tokens = 128;
    policy.min_batched_context_tokens = 64;

    BatchSpec batch = MakeDecodeBatch(1, 27);
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(llama.hparams.n_head);
    const int n_head_kv = static_cast<int>(llama.hparams.n_head_kv);
    const int head_dim_q = static_cast<int>(llama.hparams.n_embd) / n_head;

    const auto decision = densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
        policy, &llama, &cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_q);

    EXPECT_TRUE(decision.paged_decode_decision.candidate);
    EXPECT_FALSE(decision.paged_decode_decision.requested);
    EXPECT_EQ(decision.paged_decode_decision.reason,
              densecore::llm::attention::DecodePagedFallbackReason::AutoContextShort);
    EXPECT_FALSE(decision.use_paged_decode_attention);
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

TEST(AttentionPolicyTest, Gemma4PrefillSafeBatchedStaysOnExceptAdmittedDenseNativeOp) {
    EXPECT_TRUE(densecore::testing::ShouldUseGemma4PrefillSafeBatchedForTest(
        /*is_gemma4=*/true, /*prefill_phase=*/true, /*m=*/16, /*dense_prefill_allowed=*/false));
    EXPECT_FALSE(densecore::testing::ShouldUseGemma4PrefillSafeBatchedForTest(
        /*is_gemma4=*/true, /*prefill_phase=*/true, /*m=*/16, /*dense_prefill_allowed=*/true));
    EXPECT_FALSE(densecore::testing::ShouldUseGemma4PrefillSafeBatchedForTest(
        /*is_gemma4=*/true, /*prefill_phase=*/false, /*m=*/1, /*dense_prefill_allowed=*/false));
    EXPECT_FALSE(densecore::testing::ShouldUseGemma4PrefillSafeBatchedForTest(
        /*is_gemma4=*/false, /*prefill_phase=*/true, /*m=*/16, /*dense_prefill_allowed=*/false));
}

TEST(AttentionPolicyTest, Q4KTrueBatchedAutoPolicy) {
    using densecore::simd::SimdLevel;

    EXPECT_TRUE(densecore::testing::ResolveQ4KTrueBatchedKernelPolicyForTest(
        static_cast<int>(SimdLevel::AVX2), /*compiled_with_sve=*/false));
    EXPECT_TRUE(densecore::testing::ResolveQ4KTrueBatchedKernelPolicyForTest(
        static_cast<int>(SimdLevel::AVX512), /*compiled_with_sve=*/false));
    EXPECT_TRUE(densecore::testing::ResolveQ4KTrueBatchedKernelPolicyForTest(
        static_cast<int>(SimdLevel::AMX), /*compiled_with_sve=*/false));
    EXPECT_TRUE(densecore::testing::ResolveQ4KTrueBatchedKernelPolicyForTest(
        static_cast<int>(SimdLevel::SVE2), /*compiled_with_sve=*/true));
    EXPECT_FALSE(densecore::testing::ResolveQ4KTrueBatchedKernelPolicyForTest(
        static_cast<int>(SimdLevel::NEON), /*compiled_with_sve=*/false));
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
