#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "runtime/engine_internal.h"
#include "runtime/worker_internal.h"
#include "runtime/speculation_config.h"
#include "llm/speculative/head_memory.h"
#include "llm/speculative/nextn_graph.h"
#include "densecore/exceptions.h"
#include <limits>

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

#if defined(__linux__)
TEST(EngineKVCacheConfig, KVCapacityUsesDemandPagedArenaOnLinux) {
    constexpr size_t kArenaBytes = 64ULL * 1024ULL * 1024ULL;
    densecore::KVBlockAllocator allocator(/*num_blocks=*/1024, /*block_stride=*/kArenaBytes / 1024);

    ASSERT_TRUE(allocator.IsValid());
    EXPECT_EQ(allocator.ArenaSize(), kArenaBytes);
    EXPECT_EQ(allocator.AllocType(), densecore::AllocationType::Mmap)
        << "The logical KV capacity must not be eagerly committed as resident RAM";
}
#endif

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

TEST(EngineKVCacheConfig, AutoKVTargetDoesNotOverrideConstrainedContainerBudget) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", nullptr);
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", nullptr);
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");
    ScopedEnvVar available_hint("DENSECORE_KV_AVAILABLE_MB_HINT", "256");

    TransformerModel model = MakeModel(/*head_dim_k=*/64, /*n_head_kv=*/8, /*n_layer=*/16);
    model.hparams.n_ctx = 8192;
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    const size_t target_mb_value = config.target_kv_memory / (1024ULL * 1024ULL);
    EXPECT_GT(target_mb_value, 0ULL);
    EXPECT_LE(target_mb_value, 64ULL) << "Per-sequence auto KV sizing must stay within the detected container budget";
}

TEST(EngineKVCacheConfig, LargeHybridSsmDefaultsToFourConcurrentSequences) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "4096");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", nullptr);

    TransformerModel model = MakeModel(/*head_dim_k=*/256, /*n_head_kv=*/8, /*n_layer=*/48);
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_ctx = 262144;

    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_F16);

    EXPECT_EQ(config.max_num_seqs, 4)
        << "Large 35B-class hybrid-SSM serving should default to pod-level concurrency instead of serial c=2 admission";
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

TEST(EngineKVCacheConfig, RuntimeSchedulerConfigRaisesActiveSeqsAndBoundsParallelPrefillChunks) {
    KVCacheConfig kv_config;
    kv_config.max_num_seqs = 4;
    kv_config.max_seq_len = 4096;

    const densecore::SchedulerConfig scheduler_config = BuildRuntimeSchedulerConfig(kv_config);

    EXPECT_EQ(scheduler_config.max_num_seqs, 4);
    EXPECT_EQ(scheduler_config.max_prefill_seqs, 1);
    EXPECT_EQ(scheduler_config.max_parallel_prefill_chunk_tokens, 256);
    EXPECT_TRUE(scheduler_config.enable_mixed_prefill_decode);
    EXPECT_GT(scheduler_config.max_mixed_prefill_tokens, 0);
    EXPECT_LE(scheduler_config.max_mixed_prefill_tokens, scheduler_config.max_prefill_tokens);
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

TEST(EngineKVCacheConfig, GraphContextMinEnvCanScaleBeyondOldFixed48GbCeilingWhenMemoryAllows) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "1");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", "65536");
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "98304");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 24;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;

    const size_t graph_ctx_bytes = EngineState::CalculateGraphContextSize(&model);
    EXPECT_GE(graph_ctx_bytes, 65536ULL * 1024ULL * 1024ULL)
        << "Graph context sizing must not retain the old fixed 48 GB ceiling when live host memory allows more";
    EXPECT_LE(graph_ctx_bytes, 98304ULL * 1024ULL * 1024ULL)
        << "Env-expanded graph context should still respect current live memory headroom";
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

TEST(EngineKVCacheConfig, QwenHybridMoeLongPrefillUsesChunkShapeNotWholePromptGraph) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "32768");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "1");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "1048576");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 48;
    model.hparams.n_head = 8;
    model.hparams.n_head_kv = 8;
    model.hparams.n_embd_head_k = 256;
    model.hparams.n_embd_head_v = 256;
    model.hparams.n_ff = 12288;
    model.hparams.n_ctx = 262144;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 8;
    model.ssm_inner_size = 2048;

    const auto chunked_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/32768,
                                                                        /*num_seqs_hint=*/1,
                                                                        /*chunk_token_hint=*/320);
    const auto whole_prompt_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/32768,
                                                                             /*num_seqs_hint=*/1,
                                                                             /*chunk_token_hint=*/32768);

    EXPECT_EQ(chunked_estimate.effective_seq_len, 32768U);
    EXPECT_EQ(chunked_estimate.effective_query_len, 320U);
    EXPECT_EQ(chunked_estimate.effective_num_seqs, 1U);
    EXPECT_LT(chunked_estimate.total_bytes, whole_prompt_estimate.total_bytes)
        << "Long prefill admission should size the active graph from the current chunk, not the whole prompt";
    EXPECT_LE(chunked_estimate.total_bytes, 1048576ULL * 1024ULL * 1024ULL)
        << "32k chunked prefill should remain bounded by live memory instead of a machine-class constant";
}

TEST(EngineKVCacheConfig, QwenHybridMoePrefillAccountsForNativeW2DirectOutputScratch) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "1");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "65536");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 40;
    model.hparams.n_head = 8;
    model.hparams.n_head_kv = 8;
    model.hparams.n_experts = 256;
    model.hparams.n_experts_used = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;

    const auto moe = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/372,
                                                           /*num_seqs_hint=*/1,
                                                           /*chunk_token_hint=*/320);
    model.hparams.n_experts = 0;
    model.hparams.n_experts_used = 0;
    const auto dense = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/372,
                                                             /*num_seqs_hint=*/1,
                                                             /*chunk_token_hint=*/320);

    constexpr size_t kNativeW2ScratchBytes = 8ULL * 320ULL * 2048ULL * sizeof(float) * 40ULL;
    EXPECT_GE(moe.hybrid_ssm_extra_bytes, dense.hybrid_ssm_extra_bytes + kNativeW2ScratchBytes);
    EXPECT_GT(moe.total_bytes, dense.total_bytes);
}

TEST(EngineKVCacheConfig, QwenHybridMoe32kPrefillAdmissionUsesLiveMemoryEnvelope) {
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "32768");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "1");
    ScopedEnvVar graph_ctx_min("DENSECORE_GRAPH_CTX_MIN_MB", nullptr);
    ScopedEnvVar graph_ctx_max("DENSECORE_GRAPH_CTX_MAX_MB", nullptr);
    ScopedEnvVar graph_ctx_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "65536");
    ScopedEnvVar graph_ctx_extra("DENSECORE_GRAPH_CTX_EXTRA_MB", "0");

    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 2048;
    model.hparams.n_layer = 48;
    model.hparams.n_head = 8;
    model.hparams.n_head_kv = 8;
    model.hparams.n_embd_head_k = 256;
    model.hparams.n_embd_head_v = 256;
    model.hparams.n_ff = 12288;
    model.hparams.n_ctx = 262144;
    model.hparams.n_experts = 128;
    model.hparams.n_experts_used = 8;
    model.ssm_inner_size = 2048;

    const auto chunked_estimate = EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/32768,
                                                                        /*num_seqs_hint=*/1,
                                                                        /*chunk_token_hint=*/320);
    const auto larger_host_estimate = [&]() {
        ScopedEnvVar larger_available("DENSECORE_GRAPH_CTX_AVAILABLE_MB_HINT", "131072");
        return EngineState::EstimateGraphContextSize(&model, /*seq_len_hint=*/32768, /*num_seqs_hint=*/1,
                                                     /*chunk_token_hint=*/320);
    }();

    EXPECT_EQ(chunked_estimate.effective_seq_len, 32768U);
    EXPECT_EQ(chunked_estimate.effective_query_len, 320U);
    EXPECT_LE(chunked_estimate.total_bytes, 65536ULL * 1024ULL * 1024ULL)
        << "32k prefill graph admission must be capped by current live memory, not by a fixed C4/C4A RAM table";
    EXPECT_LE(chunked_estimate.total_bytes, larger_host_estimate.total_bytes)
        << "Graph budget should scale with live host headroom when the request shape is unchanged";
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


TEST(EngineKVCacheConfig, C4AQwen36AdmitsWaitingPrefillBeforeAdvancingDecodeCohorts) {
    KVCacheConfig config{};
    config.max_num_seqs = 4;
    config.max_seq_len = 4096;
    TransformerModel model{};
    model.arch_flags.is_hybrid_ssm = true;
    for (auto variant : {ModelVariant::QWEN35, ModelVariant::QWEN36, ModelVariant::QWEN38}) {
        model.variant = variant;
        const auto scheduler = BuildRuntimeSchedulerConfig(config, &model);
        bool expected = false;
#if defined(DENSECORE_TARGET_C4A)
        if (variant == ModelVariant::QWEN36) expected = true;
#endif
        EXPECT_EQ(scheduler.fill_slots_before_first_decode, expected);
        EXPECT_EQ(scheduler.max_consecutive_decode_batches, 8);
        EXPECT_EQ(scheduler.max_prefill_seqs, 1);
        EXPECT_EQ(scheduler.max_num_seqs, 4);
        EXPECT_TRUE(scheduler.isolate_prefill_decode);
    }
    model.arch_flags.is_hybrid_ssm = false;
    model.variant = ModelVariant::QWEN36;
    EXPECT_FALSE(BuildRuntimeSchedulerConfig(config, &model).fill_slots_before_first_decode);
    EXPECT_FALSE(BuildRuntimeSchedulerConfig(config).fill_slots_before_first_decode);
}

TEST(EngineKVCacheConfig, SafeAutoConcurrencyKeepsLegacyEffectiveContext) {
    ScopedEnvVar target("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", nullptr);
    ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS", nullptr);
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "16384");
    auto model = MakeModel(64, 4, 4);
    model.hparams.n_ctx = 8192;
    const auto legacy = ComputeKVCacheConfig(&model, GGML_TYPE_F16);
    const auto safe = ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
    EXPECT_EQ(safe.max_seq_len, legacy.max_seq_len);
    EXPECT_EQ(safe.max_num_seqs, 8);
    EXPECT_TRUE(safe.memory_plan.Fits());
    EXPECT_GT(safe.memory_plan.graph_bytes, 0u);
    EXPECT_GT(safe.memory_plan.repack_bytes, 0u);
}
TEST(EngineKVCacheConfig, SafeExplicitContextRemainsExactAndCapacityFailureIsExplicit) {
    ScopedEnvVar target("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS", "2");
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "16384");
    auto model = MakeModel(64, 4, 4);
    model.hparams.n_ctx = 8192;
    auto safe = ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
    EXPECT_EQ(safe.max_seq_len, 8192);
    EXPECT_EQ(safe.max_num_seqs, 2);
    {
        ScopedEnvVar exhausted("DENSECORE_KV_AVAILABLE_MB_HINT", "0");
        try {
            ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
            FAIL() << "exhausted budget must fail initialization";
        } catch (const std::exception& ex) {
            EXPECT_NE(std::string(ex.what()).find("required_bytes="), std::string::npos);
            EXPECT_NE(std::string(ex.what()).find("available_bytes=0"), std::string::npos);
        }
    }
}
TEST(EngineKVCacheConfig, SafeAutoConcurrencyReducesSequencesInsteadOfEffectiveContext) {
    ScopedEnvVar target("DENSECORE_KV_TARGET_MB", "128");
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS", nullptr);
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "4096");
    auto model = MakeModel(128, 16, 8);
    model.hparams.n_ctx = 4096;
    auto safe = ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
    EXPECT_EQ(safe.max_seq_len, 4096);
    EXPECT_GE(safe.max_num_seqs, 1);
    EXPECT_LT(safe.max_num_seqs, 8);
    EXPECT_TRUE(safe.memory_plan.Fits());
}

TEST(EngineKVCacheConfig, SafeBudgetRejectsWhenEvenOneSequenceCannotFit) {
    ScopedEnvVar target("DENSECORE_KV_TARGET_MB", "128");
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", "4096");
    ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS", nullptr);
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "1024");
    auto model = MakeModel(128, 16, 8);
    model.hparams.n_ctx = 4096;
    try {
        ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
        FAIL() << "one-sequence capacity shortfall must fail instead of shrinking context";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("context=4096"),std::string::npos);
        EXPECT_NE(std::string(error.what()).find("available_bytes="),std::string::npos);
    }
}

TEST(EngineKVCacheConfig, Qwen35TwoThousandTokenContextUsesScheduledPrefillShape) {
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", "2048");
    ScopedEnvVar target("DENSECORE_KV_TARGET_MB", "64");
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "8192");
    TransformerModel model = MakeModel(256,2,24);
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = 1024;
    model.hparams.n_head = 8;
    model.hparams.n_ctx = 262144;
    model.ssm_inner_size = 2048;
    model.ssm_time_step_rank = 16;
    model.ssm_state_size = 128;
    model.ssm_group_count = 16;
    model.ssm_conv_kernel = 4;
    model.ssm_layer_states.resize(18);
    for (const char* concurrent : {"1","4"}) {
        ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS",concurrent);
        const auto config = ComputeSafeKVCacheConfig(&model,GGML_TYPE_F16);
        EXPECT_EQ(config.max_seq_len,2048);
        EXPECT_EQ(config.max_num_seqs,std::atoi(concurrent));
        EXPECT_GT(config.max_prefill_chunk_tokens,0);
        EXPECT_LE(config.max_prefill_chunk_tokens,768);
        EXPECT_TRUE(config.memory_plan.Fits());
        EXPECT_LT(config.memory_plan.RequiredBytes(),8192ULL*1024*1024)
            << "serial bounded prefill must not charge c*2048 query tokens or two full arenas";
        const auto scheduler = BuildRuntimeSchedulerConfig(config,&model);
        EXPECT_EQ(scheduler.max_prefill_seqs,1);
        EXPECT_EQ(scheduler.max_prefill_tokens,config.max_prefill_chunk_tokens);
    }
}

namespace {
TransformerModel MakeMtpBudgetModel() {
    auto model = MakeModel(64, 4, 4);
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN35;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_vocab = 5000000; // Shape only: include head-logit arenas within the 4GiB test budget.
    model.hparams.n_ctx = 16;
    model.ssm_inner_size = 64;
    model.ssm_time_step_rank = 4;
    model.ssm_state_size = 16;
    model.ssm_group_count = 4;
    model.ssm_conv_kernel = 4;
    model.ssm_full_attn_interval = 4;
    model.ssm_layer_states.resize(3);
    return model;
}
}

TEST(EngineKVCacheConfig, AutoConcurrencyIncludesMtpBeforeKvAllocationAndKeepsContext) {
    ScopedEnvVar context("DENSECORE_MAX_SEQ_LEN", "16");
    ScopedEnvVar concurrency("DENSECORE_MAX_NUM_SEQS", nullptr);
    ScopedEnvVar memory("DENSECORE_KV_AVAILABLE_MB_HINT", "4096");
    auto model = MakeMtpBudgetModel();
    const auto plain = ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16);
    const auto mtp = ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16, DENSECORE_SPEC_MTP, 2);
    EXPECT_EQ(plain.max_num_seqs, 8);
    EXPECT_GE(mtp.max_num_seqs, 1);
    EXPECT_LT(mtp.max_num_seqs, plain.max_num_seqs);
    EXPECT_EQ(mtp.max_seq_len, 16);
    EXPECT_TRUE(mtp.memory_plan.Fits());
    EXPECT_EQ(mtp.memory_plan.mtp_checkpoint_bytes,
              CalculateSpeculationMemory(model, 16, mtp.max_num_seqs, 2).Total());
    ScopedEnvVar explicit_eight("DENSECORE_MAX_NUM_SEQS", "8");
    EXPECT_THROW(ComputeSafeKVCacheConfig(&model, GGML_TYPE_F16, DENSECORE_SPEC_MTP, 2),
                 densecore::OutOfMemoryException);
}

TEST(EngineKVCacheConfig, MtpMemoryCostChecksOverflowAndResolvesValueHeadDefault) {
    auto model = MakeMtpBudgetModel();
    const auto cost = CalculateSpeculationMemory(model, 16, 2, 2);
    EXPECT_EQ(cost.head_kv, ComputeKVCacheBytesPerToken(GGML_TYPE_F16, 64, 64, 4, 1) * 16 * 2);
    EXPECT_GT(cost.checkpoints, 0u);
    EXPECT_THROW(CalculateSpeculationMemory(model, UINT64_MAX, 1, 2), std::overflow_error);
    EXPECT_THROW(CalculateSpeculationMemory(model, 16, 0, 2), std::invalid_argument);
    EXPECT_THROW(CalculateSpeculationMemory(model, 16, 1, 4), std::invalid_argument);
    EXPECT_EQ(cost.prompt_tokens, 16u * sizeof(int) * 2);
}

TEST(EngineKVCacheConfig, MtpResourceCostRejectsUnsignedDimensionsBeforeNarrowing) {
    auto model = MakeMtpBudgetModel();
    model.hparams.n_head_kv = std::numeric_limits<decltype(model.hparams.n_head_kv)>::max();
    EXPECT_THROW(CalculateSpeculationMemory(model, 16, 1, 2), std::invalid_argument);
    model = MakeMtpBudgetModel();
    model.hparams.n_embd_head_k = std::numeric_limits<decltype(model.hparams.n_embd_head_k)>::max();
    EXPECT_THROW(CalculateSpeculationMemory(model, 16, 1, 2), std::invalid_argument);
}

TEST(MtpHeadMemory, OddKeyValueBlockStridesMatchActualKVAllocator) {
    using namespace densecore::llm::speculative;
    auto model=MakeMtpBudgetModel();
    model.arch_flags.is_hybrid_ssm=false;
    model.hparams.n_layer=1;
    model.hparams.n_head=1;
    model.hparams.n_head_kv=1;
    model.hparams.n_embd_head_k=3;
    model.hparams.n_embd_head_v=5;
    std::unique_ptr<PagedKVCache> cache(InitPagedKVCache(&model,1,17,GGML_TYPE_F16,-1));
    ASSERT_NE(cache,nullptr);
    const uint64_t actual=cache->k_allocator->ArenaSize()+cache->v_allocator->ArenaSize();
    EXPECT_EQ(HeadKVCacheBytes(model,17),actual);
    EXPECT_EQ(actual,640u); // Two K blocks128bytes each, two V blocks192bytes each.
    EXPECT_GT(actual,ComputeKVCacheBytesPerToken(GGML_TYPE_F16,3,5,1,1)*32);
}

TEST(MtpHeadMemory, EveryBoundedRuntimeShapeFitsNonPowerOfTwoContextPlan) {
    using namespace densecore::llm::speculative;
    auto model=MakeMtpBudgetModel();
    for (uint64_t context:{1,17,2049}) {
        const auto planned=CalculateSpeculationMemory(model,context,1,3);
        const auto width=std::min(context,kHeadBootstrapTokens);
        EXPECT_EQ(planned.graph,CalculateHeadGraphMemory(model,width,context).Total());
        for (uint64_t queries=1;queries<=width;++queries) {
            for (uint64_t keys:{queries,std::max(queries,context/2),context})
                EXPECT_LE(CalculateHeadGraphMemory(model,queries,keys).Total(),planned.graph);
        }
    }
    EXPECT_THROW(CalculateHeadGraphMemory(model,17,2049),std::invalid_argument);
    EXPECT_THROW(CalculateHeadGraphMemory(model,4,3),std::invalid_argument);
}

TEST(MtpHeadMemory, HugeVocabularyAndActualHeadFfnAreReservedBeforeExecution) {
    using namespace densecore::llm::speculative;
    auto model=MakeMtpBudgetModel();
    const auto small=CalculateHeadGraphMemory(model,16,2049);
    model.hparams.n_vocab=100000000;
    const auto large=CalculateHeadGraphMemory(model,16,2049);
    EXPECT_GE(large.logits,16ULL*100000000*sizeof(float));
    EXPECT_GT(large.Total()-small.Total(),4ULL*1024*1024*1024);
    ggml_tensor gate{};
    gate.ne[1]=17;
    model.mtp_heads.resize(1);
    model.mtp_heads[0].tensors["ffn_gate.weight"]=&gate;
    const auto actual_ffn=CalculateHeadGraphMemory(model,16,2049).feed_forward;
    model.hparams.n_ff=1000000; // A trunk width must not replace the actual head width.
    EXPECT_EQ(CalculateHeadGraphMemory(model,16,2049).feed_forward,actual_ffn);
    gate.ne[1]=170;
    EXPECT_GT(CalculateHeadGraphMemory(model,16,2049).feed_forward,actual_ffn);
    EXPECT_THROW(CalculateHeadGraphMemory(model,16,UINT64_MAX),std::overflow_error);
    EXPECT_THROW(HeadKVCacheBytes(model,UINT64_MAX),std::overflow_error);
}

TEST(MtpHeadMemory, ExpertMetadataScalesPerSessionButGraphAndBankRemainShared) {
    using namespace densecore::llm::speculative;
    auto model=MakeMtpBudgetModel();
    model.hparams.n_embd=2048;
    model.hparams.n_experts=256;
    model.hparams.n_experts_used=8;
    ggml_tensor expert{},shared{},router{},shared_router{};
    auto set_router=[&](ggml_tensor& tensor,int64_t rows) {
        tensor.type=GGML_TYPE_BF16;
        tensor.ne[0]=model.hparams.n_embd;
        tensor.ne[1]=rows;
        tensor.ne[2]=tensor.ne[3]=1;
        tensor.nb[0]=sizeof(uint16_t);
        for(int i=1;i<4;++i) tensor.nb[i]=tensor.nb[i-1]*tensor.ne[i-1];
    };
    set_router(router,256);
    set_router(shared_router,1);
    expert.ne[1]=512;
    shared.ne[1]=1024;
    model.mtp_heads.resize(1);
    model.mtp_heads[0].tensors["ffn_gate_exps.weight"]=&expert;
    model.mtp_heads[0].tensors["ffn_gate_shexp.weight"]=&shared;
    model.mtp_heads[0].tensors["ffn_gate_inp.weight"]=&router;
    model.mtp_heads[0].tensors["ffn_gate_inp_shexp.weight"]=&shared_router;
    const uint64_t metadata=NextNGraph::RequiredExpertMetadataBytes(model);
    const auto one=CalculateSpeculationMemory(model,2049,1,3);
    const auto eight=CalculateSpeculationMemory(model,2049,8,3);
    RecordProperty("expert_metadata_bytes_35b_router_shape",std::to_string(metadata));
    EXPECT_GT(metadata,3*1024*1024u);
    EXPECT_EQ(one.head_metadata,metadata);
    EXPECT_EQ(eight.head_metadata,8*metadata);
    EXPECT_EQ(eight.graph,one.graph);
    EXPECT_EQ(eight.checkpoints,one.checkpoints);
    EXPECT_EQ(eight.Total()-one.Total(),7*(one.head_kv+one.prompt_hidden+one.prompt_tokens+one.distributions+one.head_metadata));
    const auto routed=CalculateHeadGraphMemory(model,16,2049).feed_forward;
    model.hparams.n_experts_used=1;
    EXPECT_LT(CalculateHeadGraphMemory(model,16,2049).feed_forward,routed);
    model.hparams.n_experts_used=257;
    EXPECT_THROW(CalculateHeadGraphMemory(model,16,2049),std::invalid_argument);
}
