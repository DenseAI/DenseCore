#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "runtime/engine_internal.h"

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }

        if (value) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), value);
#else
            setenv(name_.c_str(), value, 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), "");
#else
            unsetenv(name_.c_str());
#endif
        }
    }

    ~ScopedEnvVar() {
        if (had_prev_) {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), prev_value_.c_str());
#else
            setenv(name_.c_str(), prev_value_.c_str(), 1);
#endif
            return;
        }

#if defined(_WIN32)
        _putenv_s(name_.c_str(), "");
#else
        unsetenv(name_.c_str());
#endif
    }

private:
    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

TransformerModel MakeModel(int head_dim_k, int n_head_kv = 8, int n_layer = 4, int head_dim_v = 0) {
    TransformerModel model{};
    model.hparams.n_embd_head_k = head_dim_k;
    model.hparams.n_embd_head_v = head_dim_v;
    model.hparams.n_head_kv = n_head_kv;
    model.hparams.n_layer = n_layer;
    model.hparams.n_head = std::max(1, n_head_kv);
    const int effective_v = head_dim_v > 0 ? head_dim_v : head_dim_k;
    model.hparams.n_embd = std::max(head_dim_k, effective_v) * model.hparams.n_head;
    return model;
}

size_t ExpectedKVBytesPerToken(const TransformerModel& model, ggml_type cache_type) {
    const int k_head_dim = model.hparams.n_embd_head_k;
    const int v_head_dim = model.hparams.n_embd_head_v > 0 ? model.hparams.n_embd_head_v : model.hparams.n_embd_head_k;
    const int n_head_kv = model.hparams.n_head_kv;
    const int n_layer = model.hparams.n_layer;

    auto bytes_per_slot_for_dim = [&](ggml_type type, int head_dim) -> size_t {
        if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
            return ggml_row_size(type, static_cast<int64_t>(head_dim)) * static_cast<size_t>(n_head_kv);
        }
        return ggml_row_size(type, static_cast<int64_t>(head_dim) * static_cast<int64_t>(n_head_kv));
    };

    const size_t k_bytes = bytes_per_slot_for_dim(cache_type, k_head_dim);
    const size_t v_bytes = bytes_per_slot_for_dim(cache_type, v_head_dim);
    const size_t index_bytes = (model.arch_flags.is_glm_dsa && model.glm_index_head_dim > 0)
                                   ? ggml_row_size(GGML_TYPE_F16, static_cast<int64_t>(model.glm_index_head_dim))
                                   : 0;
    return (k_bytes + v_bytes + index_bytes) * static_cast<size_t>(n_layer);
}

}  // namespace

TEST(EngineKVCacheConfig, MisalignedInt8FallsBackToF16Budgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim=*/48);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_F16);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((32ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.requested_cache_type, GGML_TYPE_Q8_0);
    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_F16);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, MisalignedInt4FallsBackToF16Budgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim=*/48);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q4_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_F16);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((32ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.requested_cache_type, GGML_TYPE_Q4_0);
    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_F16);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, AlignedInt8KeepsQuantizedBudgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "8");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim=*/64);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_Q8_0);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((8ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q8_0);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, AlignedInt4KeepsQuantizedBudgetingWithoutBudgetInflation) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "8");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim=*/64);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q4_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_Q4_0);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((8ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q4_0);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, MisalignedValueHeadFallsBackToF16Budgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim_k=*/64, /*n_head_kv=*/8, /*n_layer=*/4, /*head_dim_v=*/48);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_F16);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((32ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.requested_cache_type, GGML_TYPE_Q8_0);
    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_F16);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, AsymmetricAlignedKVUsesBothHeadDimsForBudgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "8");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim_k=*/64, /*n_head_kv=*/8, /*n_layer=*/4, /*head_dim_v=*/96);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_Q8_0);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((8ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q8_0);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, GLMIndexCacheContributesToBudgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "8");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim_k=*/64);
    model.arch_flags.is_glm_dsa = true;
    model.glm_index_head_dim = 32;
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token = ExpectedKVBytesPerToken(model, GGML_TYPE_Q8_0);
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((8ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q8_0);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}

TEST(EngineKVCacheConfig, AutoKVTargetUsesAvailableMemoryWhenNoEnvTargetIsSet) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", nullptr);
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", nullptr);
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar available_hint("DENSECORE_KV_AVAILABLE_MB_HINT", "65536");

    TransformerModel model = MakeModel(/*head_dim_k=*/256, /*n_head_kv=*/10, /*n_layer=*/48);
    model.hparams.n_ctx = 131072;
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.max_num_seqs, 4);
    EXPECT_GT(config.target_kv_memory / (1024ULL * 1024ULL), 7168ULL)
        << "Auto KV sizing should use current available capacity after generic runtime headroom, not the old "
           "fixed half-of-available split";
    EXPECT_LT(config.target_kv_memory / (1024ULL * 1024ULL), 65536ULL / 4ULL);
    EXPECT_GT(config.max_seq_len, 15291);
    EXPECT_LE(config.max_seq_len, model.hparams.n_ctx);
}

TEST(EngineKVCacheConfig, LargeHybridSsmDefaultsToTwoWeightSharedSequences) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "4096");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", nullptr);

    TransformerModel model = MakeModel(/*head_dim_k=*/256, /*n_head_kv=*/8, /*n_layer=*/48);
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_ctx = 262144;

    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.max_num_seqs, 2)
        << "Large 35B-class hybrid-SSM serving should reserve KV for c=2 by default instead of over-reserving c=4";
}

TEST(EngineKVCacheConfig, ExplicitMaxNumSeqsOverridesLargeModelDefault) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "4096");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim_k=*/256, /*n_head_kv=*/8, /*n_layer=*/48);
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_ctx = 262144;

    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.max_num_seqs, 4);
}

TEST(EngineKVCacheConfig, SmallModelKeepsBatchFourDefault) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "512");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", nullptr);

    TransformerModel model = MakeModel(/*head_dim_k=*/64, /*n_head_kv=*/8, /*n_layer=*/16);
    model.hparams.n_ctx = 8192;

    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.max_num_seqs, 4);
}

TEST(EngineKVCacheConfig, AutoKVTargetKeepsLargeGemma4ContextUsableWithoutEnvOverride) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", nullptr);
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", nullptr);
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar available_hint("DENSECORE_KV_AVAILABLE_MB_HINT", "65536");

    TransformerModel model = MakeModel(/*head_dim_k=*/512, /*n_head_kv=*/16, /*n_layer=*/60, /*head_dim_v=*/512);
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_ctx = 262144;

    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    EXPECT_EQ(config.max_num_seqs, 4);
    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q8_0);
    EXPECT_GT(config.max_seq_len, 4096)
        << "Large Gemma4 models should derive usable long-context capacity from current memory and bytes/token "
           "instead of requiring a model-specific DENSECORE_MAX_SEQ_LEN override";
    EXPECT_LE(config.max_seq_len, model.hparams.n_ctx);
}

TEST(EngineKVCacheConfig, ExplicitKVTargetStillActsAsOverride) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "512");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", nullptr);
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar available_hint("DENSECORE_KV_AVAILABLE_MB_HINT", "65536");

    TransformerModel model = MakeModel(/*head_dim_k=*/256, /*n_head_kv=*/10, /*n_layer=*/48);
    model.hparams.n_ctx = 131072;
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.target_kv_memory / (1024ULL * 1024ULL), 512ULL);
    EXPECT_EQ(config.max_seq_len, 1092);
}

TEST(EngineKVCacheConfig, HybridSsmGraphContextKeepsBatchFourHeadroom) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "512");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "8192");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 24;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;

    const size_t graph_ctx_bytes = EngineState::CalculateGraphContextSize(&model);
    EXPECT_GE(graph_ctx_bytes, static_cast<size_t>(4096) * 1024 * 1024)
        << "Hybrid Qwen3.5 batch decode graph context regressed below the known-safe 4 GB floor";
}

TEST(EngineKVCacheConfig, HybridSsmGraphContextCanGrowBeyondLaptopCeilingWhenMemoryAllows) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "32768");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 24;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;

    const size_t graph_ctx_bytes = EngineState::CalculateGraphContextSize(&model);
    EXPECT_GT(graph_ctx_bytes, static_cast<size_t>(16) * 1024 * 1024 * 1024)
        << "Adaptive graph context sizing should no longer stop at the old half-free-memory 16 GB ceiling";
    EXPECT_LE(graph_ctx_bytes, static_cast<size_t>(32) * 1024 * 1024 * 1024)
        << "Adaptive graph context sizing must still leave headroom on 32 GB available-memory hosts";
}

TEST(EngineKVCacheConfig, GraphContextHonorsExtraHeadroomEnv) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "1024");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", "8192");
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "16384");
    ScopedEnvVar graph_ctx_extra_zero("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::LLAMA;
    model.hparams.n_embd = 1024;
    model.hparams.n_layer = 8;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 8192;

    const size_t without_extra = EngineState::CalculateGraphContextSize(&model);

    ScopedEnvVar graph_ctx_extra_64("DENSECORE_GRAPH_CTX_EXTRA_MB", "64");
    const size_t with_extra = EngineState::CalculateGraphContextSize(&model);

    EXPECT_GE(with_extra, without_extra + static_cast<size_t>(64) * 1024 * 1024)
        << "DENSECORE_GRAPH_CTX_EXTRA_MB should increase graph context budget";
}

TEST(EngineKVCacheConfig, HybridSsmGraphContextGrowsForLongContextHint) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", "16384");
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "65536");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 24;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;

    const auto default_estimate = EngineState::EstimateGraphContextSize(&model);
    const auto long_hint_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/16384,
                                                                          /*num_seqs_hint=*/1,
                                                                          /*chunk_token_hint=*/2048);

    EXPECT_GT(long_hint_estimate.effective_seq_len, default_estimate.effective_seq_len);
    EXPECT_EQ(long_hint_estimate.effective_query_len, 2048U);
    EXPECT_EQ(long_hint_estimate.effective_num_seqs, 1U);
    EXPECT_GT(long_hint_estimate.total_bytes, static_cast<size_t>(1024) * 1024 * 1024)
        << "Long-context chunked graph builds still need GB-scale scratch even when the actual batch is smaller than "
           "the runtime max batch";
    EXPECT_GE(long_hint_estimate.long_context_safety_pad_bytes, static_cast<size_t>(2528) * 1024 * 1024)
        << "Hybrid Qwen3.5 long-prefill graph sizing regressed below the safety headroom needed to avoid "
           "ggml_new_object aborts on the Go server path";
}

TEST(EngineKVCacheConfig, Gemma4DecodeGraphContextUsesActualQueryShape) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "1024");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", "57344");
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "65536");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_embd = 4096;
    model.hparams.n_layer = 46;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    model.hparams.n_embd_head_k = 256;
    model.hparams.n_embd_head_v = 256;
    model.hparams.n_ctx = 131072;

    const auto decode_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/1264,
                                                                       /*num_seqs_hint=*/1,
                                                                       /*chunk_token_hint=*/1);

    EXPECT_EQ(decode_estimate.effective_seq_len, 1264U);
    EXPECT_EQ(decode_estimate.effective_query_len, 1U);
    EXPECT_EQ(decode_estimate.effective_num_seqs, 1U);
    EXPECT_LT(decode_estimate.total_bytes, static_cast<size_t>(2) * 1024 * 1024 * 1024)
        << "Single-token Gemma4 decode must not be estimated like a 4-sequence long prefill graph";
}

TEST(EngineKVCacheConfig, Gemma4PrefillGraphContextDoesNotUseRuntimeBatchMaxForSingleRequest) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "1024");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", "57344");
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "65536");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::GEMMA;
    model.arch_flags.is_gemma4 = true;
    model.hparams.n_embd = 4096;
    model.hparams.n_layer = 46;
    model.hparams.n_head = 16;
    model.hparams.n_head_kv = 8;
    model.hparams.n_embd_head_k = 256;
    model.hparams.n_embd_head_v = 256;
    model.hparams.n_ctx = 131072;

    const auto prefill_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/1641,
                                                                        /*num_seqs_hint=*/1,
                                                                        /*chunk_token_hint=*/1641);

    EXPECT_EQ(prefill_estimate.effective_seq_len, 1641U);
    EXPECT_EQ(prefill_estimate.effective_query_len, 1641U);
    EXPECT_EQ(prefill_estimate.effective_num_seqs, 1U);
    EXPECT_LT(prefill_estimate.total_bytes, static_cast<size_t>(48) * 1024 * 1024 * 1024)
        << "Single-request Gemma4 prefill should not inherit DENSECORE_MAX_NUM_SEQS=4 graph scratch";
}
