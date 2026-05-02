#ifndef DENSECORE_LLM_RUNTIME_CONFIG_H
#define DENSECORE_LLM_RUNTIME_CONFIG_H

#include <cstddef>
#include <string>

#include "runtime/runtime_env.h"

namespace densecore::llm::config {

struct WorkerRuntimeConfig {
    bool validate_mul = false;
    bool runtime_path_logging = false;
    bool scheduler_stall_debug = false;
    bool moe_trace_dump = false;
    bool sampler_trace_dump = false;
    bool determinism_boundary_debug = false;
    bool prefix_cache_reuse_disabled = false;
    bool hybrid_ssm_snapshot_restore_disabled = false;
    bool qwen36_prefix_cache_reuse_enabled = false;
    bool qwen36_hybrid_ssm_snapshot_restore_enabled = false;
    bool graph_cache_reuse_disabled = false;
    bool moe_trace_plumbing_disabled = false;
    bool moe_graph_summary = false;
    bool zero_fill_prefill_input_buffer = false;
    bool zero_fill_prefill_graph_buffer = false;
    bool zero_fill_prefill_kv_blocks = false;
    int prefill_thread_override = 0;
};

WorkerRuntimeConfig LoadWorkerRuntimeConfig();

struct EngineRuntimeDebugConfig {
    bool verbose_token_trace = false;
    bool runtime_path_logging = false;
    bool runtime_path_token_logging = false;
    bool parity_debug = false;
};

EngineRuntimeDebugConfig LoadEngineRuntimeDebugConfig();

struct EngineAneBootstrapConfig {
    bool bootstrap_buckets = true;
    std::string layer_prefix = "layer_0";
};

EngineAneBootstrapConfig LoadEngineAneBootstrapConfig();

int ReadPositiveIntEnv(const char* name, int default_value, bool* was_set = nullptr);
std::string ReadStringEnv(const char* name);
bool ReadBoolEnv(const char* name, bool default_value);

struct KVCacheRuntimeConfig {
    bool use_bulk_slot_path = true;
    bool use_hugepages = false;
};

KVCacheRuntimeConfig LoadKVCacheRuntimeConfig();

enum class DecodePagedAttentionMode { Off = 0, Auto = 1, On = 2 };

struct DecodePagedAttentionPolicy {
    DecodePagedAttentionMode mode = DecodePagedAttentionMode::Auto;
    int min_context_tokens = 256;
    int min_batched_context_tokens = 256;
    int min_head_dim = 64;
    int min_heads = 8;
    bool allow_quantized_auto = false;
    bool debug_log = false;
};

DecodePagedAttentionMode LoadDecodePagedAttentionMode();
DecodePagedAttentionPolicy LoadDecodePagedAttentionPolicy();

struct KVRetentionPolicy {
    int sliding_window = -1;
    int sink_tokens = 0;
    bool enabled = false;
};

struct KVRetentionSpan {
    int history_kept = 0;
    int sink_kept = 0;
    int tail_start = 0;
};

KVRetentionPolicy LoadKVRetentionPolicy();
KVRetentionSpan ComputeKVRetentionSpan(int n_past, const KVRetentionPolicy& policy);
int MapRetainedHistoryIndex(const KVRetentionSpan& span, int retained_index);

densecore::env::RuntimeToggleMode LoadArmQ4KNativeVecDotMode();
densecore::env::RuntimeToggleMode LoadArmInt4DirectFastPathMode();

struct PrefillGraphCachePolicy {
    bool enabled = true;
    int lru_size = 16;
    std::size_t max_bytes = 1024ULL * 1024ULL * 1024ULL;
};

struct FastPathRuntimeConfig {
    WorkerRuntimeConfig worker{};
    EngineRuntimeDebugConfig engine_debug{};
    DecodePagedAttentionPolicy decode_paged_attention{};
    KVRetentionPolicy kv_retention{};
    PrefillGraphCachePolicy prefill_graph_cache{};
    bool bench_respect_threads = false;
};

FastPathRuntimeConfig LoadFastPathRuntimeConfig();

}  // namespace densecore::llm::config

#endif  // DENSECORE_LLM_RUNTIME_CONFIG_H
