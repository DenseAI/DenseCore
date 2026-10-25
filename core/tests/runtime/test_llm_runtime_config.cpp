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
    ScopedEnvVar prefill_threads("DENSECORE_DEBUG_PREFILL_THREADS", "7");

    const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();

    EXPECT_TRUE(config.runtime_path_logging);
    EXPECT_TRUE(config.prefix_cache_reuse_disabled);
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_FALSE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);
    EXPECT_EQ(config.prefill_thread_override, 7);
}

TEST(LLMRuntimeConfigTest, Qwen36PrefixAndSnapshotRestoreDefaultFailClosedWithExplicitOptIn) {
    ScopedEnvVar prefix_reuse("DENSECORE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar snapshot_restore("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", nullptr);
    ScopedEnvVar qwen36_prefix("DENSECORE_QWEN36_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar qwen36_snapshot("DENSECORE_QWEN36_HYBRID_SSM_SNAPSHOT_RESTORE", nullptr);
    ScopedEnvVar disable_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar disable_snapshot("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", nullptr);

    auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_FALSE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);

    ScopedEnvVar enable_prefix("DENSECORE_QWEN36_PREFIX_CACHE_REUSE", "1");
    config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_FALSE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);

    ScopedEnvVar enable_snapshot("DENSECORE_QWEN36_HYBRID_SSM_SNAPSHOT_RESTORE", "1");
    config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_TRUE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_TRUE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);
}

TEST(LLMRuntimeConfigTest, PrefixAndSnapshotOperationalControlsWorkInReleaseBuilds) {
    ScopedEnvVar prefix_reuse("DENSECORE_PREFIX_CACHE_REUSE", "0");
    ScopedEnvVar snapshot_restore("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", "false");
    ScopedEnvVar qwen36_prefix("DENSECORE_QWEN36_PREFIX_CACHE_REUSE", "1");
    ScopedEnvVar qwen36_snapshot("DENSECORE_QWEN36_HYBRID_SSM_SNAPSHOT_RESTORE", "1");
    ScopedEnvVar debug_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar debug_snapshot("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", nullptr);

    const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_TRUE(config.prefix_cache_reuse_disabled);
    EXPECT_TRUE(config.hybrid_ssm_snapshot_restore_disabled);
    EXPECT_FALSE(config.qwen36_prefix_cache_reuse_enabled);
    EXPECT_FALSE(config.qwen36_hybrid_ssm_snapshot_restore_enabled);
}

TEST(LLMRuntimeConfigTest, KVRetentionPolicyUsesCanonicalEnvNamesOnly) {
    ScopedEnvVar canonical_window("DENSECORE_SLIDING_WINDOW_SIZE", "32");
    ScopedEnvVar removed_window_alias("DENSECORE_KV_SLIDING_WINDOW", "128");
    ScopedEnvVar canonical_sink("DENSECORE_SINK_TOKENS", "4");
    ScopedEnvVar removed_sink_alias("DENSECORE_KV_SINK_TOKENS", "9");

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
    ScopedEnvVar prefill_graph_cache_lru("DENSECORE_PREFILL_GRAPH_CACHE_LRU", "9");
    ScopedEnvVar prefill_graph_cache_mb("DENSECORE_PREFILL_GRAPH_CACHE_MAX_MB", "256");
    ScopedEnvVar sink_tokens("DENSECORE_SINK_TOKENS", "6");

    const auto config = densecore::llm::config::LoadFastPathRuntimeConfig();

    EXPECT_TRUE(config.worker.runtime_path_logging);
    EXPECT_TRUE(config.engine_debug.runtime_path_logging);
    EXPECT_TRUE(config.bench_respect_threads);
    EXPECT_TRUE(config.prefill_graph_cache.enabled);
    EXPECT_EQ(config.prefill_graph_cache.lru_size, 9);
    EXPECT_EQ(config.prefill_graph_cache.max_bytes, 256ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(config.kv_retention.sink_tokens, 6);
    EXPECT_EQ(config.qwen36_prefill_q4k_batched, densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On);
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__aarch64__)
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::On);
#else
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off);
#endif
    EXPECT_EQ(config.q4k_repacked_gemv, densecore::env::RuntimeToggleMode::On);
    EXPECT_EQ(config.qact_cache, densecore::env::RuntimeToggleMode::Off);
}

TEST(LLMRuntimeConfigTest, PromotedQwenFastPathsDefaultOn) {
    auto config = densecore::llm::config::LoadFastPathRuntimeConfig();
    EXPECT_EQ(config.qwen36_prefill_q4k_batched, densecore::llm::config::Qwen36PrefillQ4KBatchedMode::On);
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__aarch64__)
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::On);
#else
    EXPECT_EQ(config.qwen36_ssm_q8_prefill_amx, densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off);
#endif
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

TEST(LLMRuntimeConfigTest, WorkerRuntimeConfigUsesApprovedOutputAliases) {
    ScopedEnvVar sample("DENSECORE_DEBUG_SAMPLE", "0");
    ScopedEnvVar sample_top("DENSECORE_DEBUG_SAMPLE_TOP", "0");
    ScopedEnvVar sample_trace("DENSECORE_DEBUG_SAMPLER_TRACE", "off");
    ScopedEnvVar first_token_raw("DENSECORE_DEBUG_FIRST_TOKEN_RAW", "0");

    const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
    EXPECT_TRUE(config.debug_sample);
    EXPECT_TRUE(config.debug_sample_top);
    EXPECT_TRUE(config.debug_sampler_trace);
    EXPECT_TRUE(config.debug_first_token_raw);
}

TEST(LLMRuntimeConfigTest, WorkerRuntimeConfigUsesCanonicalCallbackModeAliasOnly) {
    {
        ScopedEnvVar direct_mode("DENSECORE_DIRECT_CALLBACK", "1");
        const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
        EXPECT_EQ(config.callback_mode, densecore::llm::config::WorkerRuntimeConfig::CallbackMode::Async);
    }

    {
        ScopedEnvVar callback_mode("DENSECORE_CALLBACK_MODE", "direct");
        const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
        EXPECT_EQ(config.callback_mode, densecore::llm::config::WorkerRuntimeConfig::CallbackMode::Direct);
    }

    {
        ScopedEnvVar callback_mode("DENSECORE_CALLBACK_MODE", "0");
        const auto config = densecore::llm::config::LoadWorkerRuntimeConfig();
        EXPECT_EQ(config.callback_mode, densecore::llm::config::WorkerRuntimeConfig::CallbackMode::Async);
    }
}

TEST(LLMRuntimeConfigTest, RemovedEnvironmentAliasesAreRejectedExplicitly) {
    ScopedEnvVar removed_alias("DENSECORE_DIRECT_CALLBACK", "1");

    const std::string error = densecore::llm::config::RemovedEnvironmentAliasError();

    EXPECT_NE(error.find("DENSECORE_DIRECT_CALLBACK"), std::string::npos);
    EXPECT_NE(error.find("DENSECORE_CALLBACK_MODE"), std::string::npos);
}

#include <unordered_map>
#include "densecore.h"
#include "densecore/runtime/resource_budget.h"
#include "densecore/models/embedding.h"

namespace {
using ResourceFiles = std::unordered_map<std::string, std::string>;
densecore::runtime::ResourceBudget FixtureBudget(const ResourceFiles& files, int affinity = 16) {
    return densecore::runtime::DetectResourceBudget([&](const std::string& path, std::string& value) {
        auto it = files.find(path); if (it == files.end()) { errno = ENOENT; return false; } value = it->second; return true;
    }, affinity);
}
ResourceFiles V2Resources() {
    return {{"/proc/meminfo","MemAvailable: 10000 kB\n"}, {"/proc/self/cgroup","0::/pods/one\n"},
            {"/proc/self/mountinfo","21 20 0:1 / /cg rw - cgroup2 cgroup rw\n"},
            {"/cg/memory.max","max"}, {"/cg/cpu.max","max 100000"},
            {"/cg/pods/memory.max","9000000"}, {"/cg/pods/memory.current","3000000"},
            {"/cg/pods/cpu.max","150000 100000"},
            {"/cg/pods/one/memory.max","8000000"}, {"/cg/pods/one/memory.current","1000000"},
            {"/cg/pods/one/cpu.max","250000 100000"}, {"/cg/pods/one/cpuset.cpus.effective","0-3,8-9"}};
}
}
TEST(ResourceBudgetTest, NestedV2UsesTightestRemainingMemoryAndCeilsFractionalCPU) {
    auto result = FixtureBudget(V2Resources());
    EXPECT_EQ(result.memory.state, densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes, 6000000u);
    EXPECT_EQ(result.cgroup_memory_limit.bytes, 8000000u);
    EXPECT_DOUBLE_EQ(result.cpu_quota, 1.5);
    EXPECT_EQ(result.effective_threads, 2);
}
TEST(ResourceBudgetTest, ExhaustedParentIsFiniteZeroAndNeverUnlimited) {
    auto files = V2Resources(); files["/cg/pods/memory.current"] = "10000000";
    auto result = FixtureBudget(files);
    EXPECT_EQ(result.memory.state, densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes, 0u);
}
TEST(ResourceBudgetTest, MissingUsageIsUnknownInsteadOfFullLimit) {
    auto files = V2Resources(); files.erase("/cg/pods/memory.current");
    EXPECT_EQ(FixtureBudget(files).memory.state, densecore::runtime::ResourceState::Unknown);
}
TEST(ResourceBudgetTest, UnlimitedRemainsDistinctFromDetectionFailure) {
    auto files = V2Resources();
    files["/cg/pods/memory.max"] = "max"; files["/cg/pods/one/memory.max"] = "max";
    auto result = FixtureBudget(files);
    EXPECT_EQ(result.cgroup_memory_limit.state, densecore::runtime::ResourceState::Unlimited);
    EXPECT_EQ(result.memory.bytes, 10240000u);
    EXPECT_EQ(FixtureBudget({}).memory.state, densecore::runtime::ResourceState::Unknown);
}
TEST(ResourceBudgetTest, NamespacedV1MountRootAndParentAreResolved) {
    ResourceFiles files{{"/proc/meminfo","MemAvailable: 10000 kB\n"},
        {"/proc/self/cgroup","5:memory:/slice/task\n4:cpu,cpuacct:/slice/task\n3:cpuset:/slice/task\n"},
        {"/proc/self/mountinfo","1 0 0:1 /slice /memory rw - cgroup cgroup rw,memory\n2 0 0:2 /slice /cpu rw - cgroup cgroup rw,cpu,cpuacct\n3 0 0:3 /slice /sets rw - cgroup cgroup rw,cpuset\n"},
        {"/memory/memory.limit_in_bytes","6000000"}, {"/memory/memory.usage_in_bytes","2000000"},
        {"/memory/task/memory.limit_in_bytes","5000000"}, {"/memory/task/memory.usage_in_bytes","4000000"},
        {"/cpu/cpu.cfs_quota_us","-1"}, {"/cpu/task/cpu.cfs_quota_us","50000"}, {"/cpu/task/cpu.cfs_period_us","100000"},
        {"/sets/task/cpuset.cpus","2-3"}};
    auto result = FixtureBudget(files, 8);
    EXPECT_EQ(result.memory.bytes, 1000000u);
    EXPECT_EQ(result.effective_threads, 1);
    EXPECT_DOUBLE_EQ(result.cpu_quota, 0.5);
}
TEST(ResourceBudgetTest, CpusetAndAffinityRestrictUnlimitedCPU) {
    auto files = V2Resources(); files["/cg/pods/cpu.max"] = "max 100000"; files["/cg/pods/one/cpu.max"] = "max 100000";
    EXPECT_EQ(FixtureBudget(files, 4).effective_threads, 4);
    EXPECT_EQ(FixtureBudget(files, 16).effective_threads, 6);
}
TEST(EmbeddingPoolingTest, AutoConvertsGGUFEnumsAndPreservesExplicitValues) {
    using densecore::PoolingStrategy;
    PoolingStrategy result;
    ASSERT_TRUE(densecore::ResolveEmbeddingPooling(-1, true, 3, result));
    EXPECT_EQ(result, PoolingStrategy::LAST);
    ASSERT_TRUE(densecore::ResolveEmbeddingPooling(-1, true, 1, result));
    EXPECT_EQ(result, PoolingStrategy::MEAN);
    ASSERT_TRUE(densecore::ResolveEmbeddingPooling(-1, true, 2, result));
    EXPECT_EQ(result, PoolingStrategy::CLS);
    ASSERT_TRUE(densecore::ResolveEmbeddingPooling(-1, false, -1, result));
    EXPECT_EQ(result, PoolingStrategy::MEAN);
    EXPECT_FALSE(densecore::ResolveEmbeddingPooling(-1, true, 0, result));
    EXPECT_FALSE(densecore::ResolveEmbeddingPooling(-1, true, 4, result));
    EXPECT_FALSE(densecore::ResolveEmbeddingPooling(-1, true, -1, result));
    ASSERT_TRUE(densecore::ResolveEmbeddingPooling(2, true, 4, result));
    EXPECT_EQ(result, PoolingStrategy::LAST);
    EXPECT_FALSE(densecore::ResolveEmbeddingPooling(7, false, 0, result));
}
TEST(ResourceBudgetTest, SizedQueryRejectsUnknownVersionAndShortStruct) {
    DenseCoreResourceBudget budget{};
    EXPECT_EQ(DenseCoreGetResourceBudget(&budget), DENSECORE_STATUS_INVALID_ARGUMENT);
    budget.struct_size = sizeof(budget); budget.version = 2;
    EXPECT_EQ(DenseCoreGetResourceBudget(&budget), DENSECORE_STATUS_INVALID_ARGUMENT);
    budget.version = 1;
    EXPECT_EQ(DenseCoreGetResourceBudget(&budget), DENSECORE_STATUS_OK);
    EXPECT_GE(budget.effective_threads, 1u);
}

TEST(ResourceBudgetTest, ExhaustionRemainsKnownEvenWhenAnotherConstraintIsUnreadable) {
    auto files = V2Resources();
    files["/cg/pods/memory.current"] = "9000000";
    files.erase("/cg/pods/one/memory.current");
    auto result = FixtureBudget(files);
    EXPECT_EQ(result.memory.state, densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes, 0u);
}
TEST(ResourceBudgetTest, HierarchyRootMissingLimitIsUnlimitedButConstrainedMissingLimitIsUnknown) {
    auto files = V2Resources(); files.erase("/cg/memory.max"); files.erase("/cg/cpu.max");
    EXPECT_EQ(FixtureBudget(files).memory.state, densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(FixtureBudget(files).cpu_quota_state, densecore::runtime::ResourceState::Finite);
    files.erase("/cg/pods/memory.max");
    EXPECT_EQ(FixtureBudget(files).memory.state, densecore::runtime::ResourceState::Unknown);
}
TEST(ResourceBudgetTest, CPUParentReadFailureIsUnknownAndConservative) {
    auto files = V2Resources(); files.erase("/cg/pods/cpu.max");
    auto result = FixtureBudget(files);
    EXPECT_EQ(result.cpu_quota_state, densecore::runtime::ResourceState::Unknown);
    EXPECT_EQ(result.effective_threads, 1);
}
TEST(ResourceBudgetTest, UnreadableRootLimitIsNotMistakenForUnlimited) {
    const auto files = V2Resources();
    const auto result = densecore::runtime::DetectResourceBudget([&](const std::string& path, std::string& text) {
        if (path == "/cg/memory.max") { errno = EACCES; return false; }
        auto it = files.find(path);
        if (it == files.end()) { errno = ENOENT; return false; }
        text = it->second;
        return true;
    }, 16);
    EXPECT_EQ(result.memory.state, densecore::runtime::ResourceState::Unknown);
}

TEST(LLMRuntimeConfigTest, Qwen35PrefixQualificationOptInHonorsBothGlobalControls) {
    ScopedEnvVar prefix_reuse("DENSECORE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar snapshot_restore("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", nullptr);
    ScopedEnvVar opt_in("DENSECORE_QWEN35_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar debug_prefix("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", nullptr);
    ScopedEnvVar debug_snapshot("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", nullptr);
    EXPECT_FALSE(densecore::llm::config::LoadWorkerRuntimeConfig().qwen35_prefix_cache_reuse_enabled);
    {
        ScopedEnvVar enabled("DENSECORE_QWEN35_PREFIX_CACHE_REUSE", "1");
        EXPECT_TRUE(densecore::llm::config::LoadWorkerRuntimeConfig().qwen35_prefix_cache_reuse_enabled);
        {
            ScopedEnvVar disabled("DENSECORE_PREFIX_CACHE_REUSE", "0");
            EXPECT_FALSE(densecore::llm::config::LoadWorkerRuntimeConfig().qwen35_prefix_cache_reuse_enabled);
        }
        {
            ScopedEnvVar disabled("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", "0");
            EXPECT_FALSE(densecore::llm::config::LoadWorkerRuntimeConfig().qwen35_prefix_cache_reuse_enabled);
        }
    }
}

#include "densecore/runtime/resource_ledger.h"
TEST(ResourceLedgerTest, ConcurrentPlansCannotAllocateTheSameRemainingBudgetTwice) {
    using namespace densecore::runtime;
    const auto before = ResourceLedgerUsageSnapshot();
    RuntimeMemoryPlan plan;
    plan.resource_capacity_bytes = before.planned_bytes + 100;
    plan.available_bytes = 100;
    plan.graph_bytes = 60;
    ResourcePlanLease first(plan);
    EXPECT_THROW(ResourcePlanLease second(plan), densecore::OutOfMemoryException);
    EXPECT_EQ(ProcessPlannedMemoryBytes(), before.planned_bytes + 60);
    std::string error;
    EXPECT_TRUE(first.ExtendMTP(40, &error));
    EXPECT_FALSE(first.ExtendMTP(1, &error));
    EXPECT_NE(error.find("required_bytes="), std::string::npos);
}
TEST(ResourceLedgerTest, AllocationLifetimesRemainChargedAfterCacheReferenceIsReleased) {
    using namespace densecore::runtime;
    const auto before = ResourceLedgerUsageSnapshot();
    RuntimeMemoryPlan plan;
    plan.graph_bytes = before.graph_bytes + 64;
    plan.repack_bytes = before.repack_bytes + 64;
    plan.available_bytes = plan.RequiredBytes();
    plan.resource_capacity_bytes = resource_detail::Add(before.planned_bytes,plan.RequiredBytes());
    ResourcePlanLease reservation(plan);
    auto cached = ResourceAllocationLease::Acquire(AllocationCategory::Repack,64);
    ASSERT_TRUE(cached);
    auto in_flight = cached;
    cached.reset();
    EXPECT_FALSE(ResourceAllocationLease::Acquire(AllocationCategory::Repack,1));
    in_flight.reset();
    EXPECT_TRUE(ResourceAllocationLease::Acquire(AllocationCategory::Repack,1));
    auto graph = ResourceAllocationLease::Acquire(AllocationCategory::Graph,64);
    ASSERT_TRUE(graph);
    EXPECT_FALSE(ResourceAllocationLease::Acquire(AllocationCategory::Graph,1));
    graph.reset();
    EXPECT_EQ(ResourceLedgerUsageSnapshot().graph_bytes,before.graph_bytes);
}
TEST(ResourceLedgerTest, FailedWeightConstructionDoesNotStrandAnAllocationCharge) {
    using namespace densecore::runtime;
    struct FailedWeight { FailedWeight() { throw std::runtime_error("construction failed"); } };
    const auto before = ResourceLedgerUsageSnapshot();
    EXPECT_THROW(MakeBudgetedRepackedWeight<FailedWeight>(16),std::runtime_error);
    EXPECT_EQ(ResourceLedgerUsageSnapshot().repack_bytes,before.repack_bytes);
}

TEST(ResourceLedgerTest, ActualKVReconciliationGrowsAtomicallyAndRejectsWithoutChangingReservation) {
    using namespace densecore::runtime;
    const auto before = ResourceLedgerUsageSnapshot();
    RuntimeMemoryPlan plan;
    plan.kv_bytes = 50;
    plan.available_bytes = 100;
    plan.resource_capacity_bytes = before.planned_bytes + 100;
    ResourcePlanLease lease(plan);
    std::string error;
    EXPECT_TRUE(lease.EnsureKVBytes(60, &error));
    EXPECT_EQ(ProcessPlannedMemoryBytes(),before.planned_bytes+60);
    EXPECT_FALSE(lease.EnsureKVBytes(101, &error));
    EXPECT_NE(error.find("required_bytes="),std::string::npos);
    EXPECT_EQ(ProcessPlannedMemoryBytes(),before.planned_bytes+60);
    EXPECT_TRUE(lease.EnsureKVBytes(40, &error));
    EXPECT_EQ(ProcessPlannedMemoryBytes(),before.planned_bytes+60);
}

TEST(ResourceBudgetTest, UnreadableHierarchyDiscoveryNeverFallsBackToHostCapacity) {
    for (const auto* unavailable : {"/proc/self/cgroup","/proc/self/mountinfo"}) {
        auto files = V2Resources();
        files.erase(unavailable);
        const auto result = FixtureBudget(files);
        EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Unknown) << unavailable;
        EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Unknown) << unavailable;
        EXPECT_EQ(result.effective_threads,1) << unavailable;
    }
}
TEST(ResourceBudgetTest, MalformedHierarchyDiscoveryIsUnknownInsteadOfUnlimited) {
    for (const auto* corrupt : {"/proc/self/cgroup","/proc/self/mountinfo"}) {
        auto files = V2Resources();
        files[corrupt] = "malformed record";
        const auto result = FixtureBudget(files);
        EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Unknown) << corrupt;
        EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Unknown) << corrupt;
        EXPECT_EQ(result.effective_threads,1) << corrupt;
    }
}
TEST(ResourceBudgetTest, ExplicitlyAbsentHierarchyUsesHostMemoryAndAffinity) {
    ResourceFiles files{{"/proc/meminfo","MemAvailable: 10000 kB\n"},
                        {"/proc/self/cgroup",""},
                        {"/proc/self/mountinfo","1 0 0:1 / / rw - ext4 /dev/root rw\n"}};
    const auto result = FixtureBudget(files,4);
    EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes,10240000u);
    EXPECT_EQ(result.cgroup_memory_limit.state,densecore::runtime::ResourceState::Unlimited);
    EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Unlimited);
    EXPECT_EQ(result.effective_threads,4);
}
TEST(ResourceBudgetTest, EmptyMountInfoOrInconsistentEmptyMembershipIsUnknown) {
    for (const auto* empty : {"/proc/self/cgroup","/proc/self/mountinfo"}) {
        auto files = V2Resources();
        files[empty] = "";
        const auto result = FixtureBudget(files);
        EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Unknown);
        EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Unknown);
        EXPECT_EQ(result.effective_threads,1);
    }
}

TEST(ResourceBudgetTest, HybridHierarchyUsesAuthoritativeV1Controllers) {
    ResourceFiles files{{"/proc/meminfo","MemAvailable: 10000 kB\n"},
        {"/proc/self/cgroup","5:memory:/slice/task\n4:cpu,cpuacct:/slice/task\n0::/slice/task\n"},
        {"/proc/self/mountinfo","1 0 0:1 /slice /memory rw - cgroup cgroup rw,memory\n2 0 0:2 /slice /cpu rw - cgroup cgroup rw,cpu,cpuacct\n3 0 0:3 / /unified rw - cgroup2 cgroup rw\n"},
        {"/memory/memory.limit_in_bytes","6000000"}, {"/memory/memory.usage_in_bytes","2000000"},
        {"/memory/task/memory.limit_in_bytes","5000000"}, {"/memory/task/memory.usage_in_bytes","4000000"},
        {"/cpu/cpu.cfs_quota_us","-1"}, {"/cpu/task/cpu.cfs_quota_us","50000"}, {"/cpu/task/cpu.cfs_period_us","100000"},
        {"/unified/cgroup.controllers",""}};
    const auto result = FixtureBudget(files,8);
    EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes,1000000u);
    EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Finite);
    EXPECT_DOUBLE_EQ(result.cpu_quota,0.5);
    EXPECT_EQ(result.effective_threads,1);
}
TEST(ResourceBudgetTest, UnavailableChildControllersStillHonorControllingAncestors) {
    auto files = V2Resources();
    files["/cg/cgroup.controllers"] = "cpu memory cpuset";
    files["/cg/pods/one/cgroup.controllers"] = "";
    files.erase("/cg/pods/one/memory.max");
    files.erase("/cg/pods/one/cpu.max");
    const auto result = FixtureBudget(files);
    EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes,6000000u);
    EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Finite);
    EXPECT_DOUBLE_EQ(result.cpu_quota,1.5);
    EXPECT_EQ(result.effective_threads,2);
}
TEST(ResourceBudgetTest, ValidV2HierarchyWithoutResourceControllersUsesHostBudget) {
    auto files = V2Resources();
    files["/cg/cgroup.controllers"] = "";
    const auto result = FixtureBudget(files,4);
    EXPECT_EQ(result.memory.state,densecore::runtime::ResourceState::Finite);
    EXPECT_EQ(result.memory.bytes,10240000u);
    EXPECT_EQ(result.cgroup_memory_limit.state,densecore::runtime::ResourceState::Unlimited);
    EXPECT_EQ(result.cpu_quota_state,densecore::runtime::ResourceState::Unlimited);
    EXPECT_EQ(result.effective_threads,4);
}
