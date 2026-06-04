#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "llm/config/runtime_config.h"

namespace {

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

}  // namespace

TEST(LLMRuntimeConfigTest, WorkerRuntimeFlagsUseCentralizedParsing) {
    ScopedEnvVar runtime_path("DENSECORE_DEBUG_RUNTIME_PATH", "1");
    ScopedEnvVar disable_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", "1");
    ScopedEnvVar enable_qwen36_prefix("DENSECORE_QWEN36_ENABLE_PREFIX_CACHE_REUSE", "1");
    ScopedEnvVar enable_qwen36_snapshot("DENSECORE_QWEN36_ENABLE_HYBRID_SSM_SNAPSHOT_RESTORE", "1");
    ScopedEnvVar prefill_threads("DENSECORE_DEBUG_PREFILL_THREADS", "7");

    const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();

    EXPECT_TRUE(config.runtime_path_logging);
    EXPECT_TRUE(config.prefix_cache_reuse_disabled);
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_TRUE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);
    EXPECT_EQ(config.prefill_thread_override, 7);
}

TEST(LLMRuntimeConfigTest, Qwen36PrefixAndSnapshotRestoreDefaultToAutoWithKillSwitches) {
    ScopedEnvVar disable_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar disable_snapshot("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", nullptr);
    ScopedEnvVar legacy_prefix("DENSECORE_QWEN36_ENABLE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar legacy_snapshot("DENSECORE_QWEN36_ENABLE_HYBRID_SSM_SNAPSHOT_RESTORE", nullptr);

    auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_TRUE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_TRUE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);

    ScopedEnvVar kill_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", "1");
    ScopedEnvVar kill_snapshot("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", "1");
    config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_FALSE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);
}

TEST(LLMRuntimeConfigTest, KVRetentionPolicyHonorsPrimaryEnvNames) {
    ScopedEnvVar legacy_window("DENSECORE_SLIDING_WINDOW_SIZE", "128");
    ScopedEnvVar primary_window("DENSECORE_KV_SLIDING_WINDOW", "32");
    ScopedEnvVar legacy_sink("DENSECORE_SINK_TOKENS", "9");
    ScopedEnvVar primary_sink("DENSECORE_KV_SINK_TOKENS", "4");

    const auto policy = densecore::llm::config::LoadKVRetentionPolicy();
    const auto span = densecore::llm::config::ComputeKVRetentionSpan(40, policy);

    EXPECT_TRUE(policy.enabled);
    EXPECT_EQ(policy.sliding_window, 32);
    EXPECT_EQ(policy.sink_tokens, 4);
    EXPECT_EQ(span.sink_kept, 4);
    EXPECT_EQ(span.tail_start, 8);
    EXPECT_EQ(span.history_kept, 36);
}

TEST(LLMRuntimeConfigTest, DecodePagedAttentionPolicyKeepsPagedDecodeOnAndRespectsQuantizedAlias) {
    ScopedEnvVar legacy_quantized("DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8", "0");
    ScopedEnvVar primary_quantized("DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED", nullptr);
    ScopedEnvVar debug_log("DENSECORE_DEBUG_PAGED_ATTN_DECODE", "1");

    const auto policy = densecore::llm::config::LoadDecodePagedAttentionPolicy();

    EXPECT_EQ(policy.mode, densecore::llm::config::DecodePagedAttentionMode::On);
    EXPECT_FALSE(policy.allow_quantized_auto);
    EXPECT_TRUE(policy.debug_log);
}

TEST(LLMRuntimeConfigTest, KVCacheRuntimeConfigUsesCentralizedBulkPathPolicy) {
    ScopedEnvVar bulk_path("DENSECORE_KV_USE_BULK_PATH", "0");
    const auto config = densecore::llm::config::LoadKVCacheRuntimeConfig();
    EXPECT_FALSE(config.use_bulk_slot_path);
}

TEST(LLMRuntimeConfigTest, EngineHelpersPreservePositiveIntAndBoolParsing) {
    ScopedEnvVar positive("DENSECORE_TEST_POSITIVE", "17");
    ScopedEnvVar invalid_positive("DENSECORE_TEST_INVALID_POSITIVE", "-3");
    ScopedEnvVar truthy("DENSECORE_TEST_BOOL", "yes");

    bool was_set = false;
    EXPECT_EQ(densecore::llm::config::ReadPositiveIntEnv("DENSECORE_TEST_POSITIVE", 5, &was_set), 17);
    EXPECT_TRUE(was_set);
    EXPECT_EQ(densecore::llm::config::ReadPositiveIntEnv("DENSECORE_TEST_INVALID_POSITIVE", 5, &was_set), 5);
    EXPECT_FALSE(was_set);
    EXPECT_TRUE(densecore::llm::config::ReadBoolEnv("DENSECORE_TEST_BOOL", false));
}

TEST(LLMRuntimeConfigTest, FastPathRuntimeConfigAggregatesHotLoopPolicies) {
    ScopedEnvVar runtime_path("DENSECORE_DEBUG_RUNTIME_PATH", "1");
    ScopedEnvVar bench_respect("DENSECORE_BENCH_RESPECT_THREADS", "1");
    ScopedEnvVar prefill_graph_cache("DENSECORE_PREFILL_GRAPH_CACHE", "0");
    ScopedEnvVar prefill_graph_cache_lru("DENSECORE_PREFILL_GRAPH_CACHE_LRU", "9");
    ScopedEnvVar prefill_graph_cache_mb("DENSECORE_PREFILL_GRAPH_CACHE_MAX_MB", "256");
    ScopedEnvVar sink_tokens("DENSECORE_KV_SINK_TOKENS", "6");
    ScopedEnvVar qwen36_ssm_q8_amx("DENSECORE_QWEN36_SSM_Q8_AMX_ALIAS", "on");
    ScopedEnvVar qwen36_ssm_q8_prefill_amx("DENSECORE_QWEN36_SSM_Q8_PREFILL_AMX", "on");
    ScopedEnvVar qwen36_expert_repack("DENSECORE_QWEN36_EXPERT_CPU_REPACK", "off");
    ScopedEnvVar qact_cache("DENSECORE_ENABLE_QACT_CACHE", "on");

    const auto config = densecore::llm::config::LoadFastPathRuntimeConfig();

    EXPECT_TRUE(config.worker.runtime_path_logging);
    EXPECT_TRUE(config.engine_debug.runtime_path_logging);
    EXPECT_TRUE(config.bench_respect_threads);
    EXPECT_FALSE(config.prefill_graph_cache.enabled);
    EXPECT_EQ(config.prefill_graph_cache.lru_size, 9);
    EXPECT_EQ(config.prefill_graph_cache.max_bytes, 256ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(config.kv_retention.sink_tokens, 6);
    EXPECT_EQ(config.qwen36_prefill_q4k_batched, densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On);
    EXPECT_EQ(config.qwen36_ssm_q8_amx_alias, densecore::env::RuntimeToggleMode::On);
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::On);
    EXPECT_EQ(config.qwen36_expert_cpu_repack, densecore::env::RuntimeToggleMode::Off);
    EXPECT_EQ(config.q4k_repacked_gemv, densecore::env::RuntimeToggleMode::On);
    EXPECT_EQ(config.qact_cache, densecore::env::RuntimeToggleMode::On);
}

TEST(LLMRuntimeConfigTest, PromotedQwenFastPathsDefaultOn) {
    ScopedEnvVar qwen36_ssm_q8_amx("DENSECORE_QWEN36_SSM_Q8_AMX_ALIAS", nullptr);
    ScopedEnvVar qwen36_ssm_q8_prefill_amx("DENSECORE_QWEN36_SSM_Q8_PREFILL_AMX", nullptr);
    ScopedEnvVar qwen36_expert_repack("DENSECORE_QWEN36_EXPERT_CPU_REPACK", nullptr);
    ScopedEnvVar qact_cache("DENSECORE_ENABLE_QACT_CACHE", nullptr);
    auto config = densecore::llm::config::LoadFastPathRuntimeConfig();
    EXPECT_EQ(config.qwen36_prefill_q4k_batched, densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On);
    EXPECT_EQ(config.qwen36_ssm_q8_amx_alias, densecore::env::RuntimeToggleMode::Off);
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Probe);
    EXPECT_EQ(config.qwen36_expert_cpu_repack, densecore::env::RuntimeToggleMode::Auto);
    EXPECT_EQ(config.q4k_repacked_gemv, densecore::env::RuntimeToggleMode::On);
    EXPECT_EQ(config.qact_cache, densecore::env::RuntimeToggleMode::Off);

    ScopedEnvVar invalid_qwen36_ssm_q8_amx("DENSECORE_QWEN36_SSM_Q8_AMX_ALIAS", "garbage");
    ScopedEnvVar invalid_qwen36_ssm_q8_prefill_amx("DENSECORE_QWEN36_SSM_Q8_PREFILL_AMX", "garbage");
    ScopedEnvVar invalid_qwen36_expert_repack("DENSECORE_QWEN36_EXPERT_CPU_REPACK", "garbage");
    ScopedEnvVar invalid_qact_cache("DENSECORE_ENABLE_QACT_CACHE", "garbage");
    config = densecore::llm::config::LoadFastPathRuntimeConfig();
    EXPECT_EQ(config.qwen36_prefill_q4k_batched, densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On);
    EXPECT_EQ(config.qwen36_ssm_q8_amx_alias, densecore::env::RuntimeToggleMode::Off);
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off);
    EXPECT_EQ(config.qwen36_expert_cpu_repack, densecore::env::RuntimeToggleMode::Off);
    EXPECT_EQ(config.q4k_repacked_gemv, densecore::env::RuntimeToggleMode::On);
    EXPECT_EQ(config.qact_cache, densecore::env::RuntimeToggleMode::Off);
}

TEST(LLMRuntimeConfigTest, FastPathRuntimeConfigIsASnapshotNotALiveEnvView) {
    ScopedEnvVar bench_respect("DENSECORE_BENCH_RESPECT_THREADS", "1");
    const auto snapshot = densecore::llm::config::LoadFastPathRuntimeConfig();

    {
        ScopedEnvVar flipped_bench_respect("DENSECORE_BENCH_RESPECT_THREADS", "0");
        const auto updated = densecore::llm::config::LoadFastPathRuntimeConfig();
        EXPECT_TRUE(snapshot.bench_respect_threads);
        EXPECT_FALSE(updated.bench_respect_threads);
    }

    EXPECT_TRUE(snapshot.bench_respect_threads);
}
