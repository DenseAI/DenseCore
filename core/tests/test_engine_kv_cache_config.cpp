#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "engine_internal.h"

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

TransformerModel MakeModel(int head_dim, int n_head_kv = 8, int n_layer = 4) {
    TransformerModel model{};
    model.hparams.n_embd_head_k = head_dim;
    model.hparams.n_head_kv = n_head_kv;
    model.hparams.n_layer = n_layer;
    model.hparams.n_head = std::max(1, n_head_kv);
    model.hparams.n_embd = head_dim * model.hparams.n_head;
    return model;
}

}  // namespace

TEST(EngineKVCacheConfig, MisalignedInt8FallsBackToF16Budgeting) {
    ScopedEnvVar target_mb("DENSECORE_KV_TARGET_MB", "32");
    ScopedEnvVar max_seq_len("DENSECORE_MAX_SEQ_LEN", "8192");
    ScopedEnvVar max_num_seqs("DENSECORE_MAX_NUM_SEQS", "4");

    TransformerModel model = MakeModel(/*head_dim=*/48);
    const KVCacheConfig config = ComputeKVCacheConfig(&model, GGML_TYPE_Q8_0);

    const size_t expected_bytes_per_token =
        ggml_row_size(GGML_TYPE_F16, static_cast<int64_t>(model.hparams.n_embd_head_k) * model.hparams.n_head_kv) *
        static_cast<size_t>(model.hparams.n_layer) * 2;
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

    const size_t expected_bytes_per_token =
        ggml_row_size(GGML_TYPE_F16, static_cast<int64_t>(model.hparams.n_embd_head_k) * model.hparams.n_head_kv) *
        static_cast<size_t>(model.hparams.n_layer) * 2;
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

    const size_t expected_bytes_per_token =
        ggml_row_size(GGML_TYPE_Q8_0, model.hparams.n_embd_head_k) * static_cast<size_t>(model.hparams.n_head_kv) *
        static_cast<size_t>(model.hparams.n_layer) * 2;
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

    const size_t expected_bytes_per_token =
        ggml_row_size(GGML_TYPE_Q4_0, model.hparams.n_embd_head_k) * static_cast<size_t>(model.hparams.n_head_kv) *
        static_cast<size_t>(model.hparams.n_layer) * 2;
    const int expected_seq_len =
        std::max(256, std::min(static_cast<int>((8ULL * 1024ULL * 1024ULL) / expected_bytes_per_token), 8192));

    EXPECT_EQ(config.effective_cache_type, GGML_TYPE_Q4_0);
    EXPECT_EQ(config.bytes_per_token, expected_bytes_per_token);
    EXPECT_EQ(config.max_seq_len, expected_seq_len);
}
