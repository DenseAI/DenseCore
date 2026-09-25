#include "llm/config/runtime_config.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "densecore/simd/simd_ops.h"
#include "runtime/runtime_env.h"

namespace densecore::llm::config {
namespace {

bool IsEnvPresent(const char* name) {
    return std::getenv(name) != nullptr;
}

// Presence-only diagnostic knobs; compiled out of release builds like the rest
// of the DENSECORE_DEBUG_* namespace.
bool IsDiagnosticEnvPresent(const char* name) {
    return env::GetDiagnosticEnv(name) != nullptr;
}

Qwen36SSMQ8PrefillAMXMode DefaultQwen36SSMQ8PrefillAMXMode() {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__aarch64__)
    return Qwen36SSMQ8PrefillAMXMode::On;
#else
    return Qwen36SSMQ8PrefillAMXMode::Off;
#endif
}

WorkerRuntimeConfig::CallbackMode ParseCallbackMode() {
    const std::string mode = env::AsciiLowerCopy(std::getenv("DENSECORE_CALLBACK_MODE"));
    if (mode == "direct" || mode == "1" || mode == "true" || mode == "yes" || mode == "on") {
        return WorkerRuntimeConfig::CallbackMode::Direct;
    }
    return WorkerRuntimeConfig::CallbackMode::Async;
}

std::size_t ReadAvailableMemoryMbForAutoRuntimeCache() {
#if defined(__linux__)
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (!file) {
        return 0;
    }
    char line[256] = {};
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            std::fclose(file);
            return static_cast<std::size_t>(kb / 1024ULL);
        }
    }
    std::fclose(file);
#endif
    return 0;
}

}  // namespace

WorkerRuntimeConfig LoadWorkerRuntimeConfig() {
    WorkerRuntimeConfig config;
    config.validate_mul = env::ParseDiagnosticEnv("DENSECORE_DEBUG_VALIDATE_MUL", false);
    config.runtime_path_logging = env::ParseDiagnosticEnv("DENSECORE_DEBUG_RUNTIME_PATH", false);
    config.scheduler_stall_debug = env::ParseDiagnosticEnv("DENSECORE_SCHEDULER_STALL_DEBUG", false);
    config.moe_trace_dump = env::ParseDiagnosticEnv("DENSECORE_DEBUG_MOE_TRACE_DUMP", false);
    config.sampler_trace_dump = env::ParseDiagnosticEnv("DENSECORE_DEBUG_SAMPLER_TRACE_DUMP", false);
    config.determinism_boundary_debug = env::ParseDiagnosticEnv("DENSECORE_DEBUG_DETERMINISM_BOUNDARY", false);
    config.prefix_cache_reuse_disabled =
        !env::ParseBoolEnv("DENSECORE_PREFIX_CACHE_REUSE", true) ||
        env::ParseDiagnosticEnv("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", false);
    config.hybrid_ssm_snapshot_restore_disabled =
        !env::ParseBoolEnv("DENSECORE_HYBRID_SSM_SNAPSHOT_RESTORE", true) ||
        env::ParseDiagnosticEnv("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", false);
    // Qwen3.6 recurrent-state restore is fail-closed until cache-hit token
    // parity is proven. The global controls remain enabled for qualified
    // model families; Qwen3.6 requires an explicit experimental opt-in.
    config.qwen36_hybrid_ssm_snapshot_restore_enabled =
        !config.hybrid_ssm_snapshot_restore_disabled &&
        env::ParseBoolEnv("DENSECORE_QWEN36_HYBRID_SSM_SNAPSHOT_RESTORE", false);
    config.qwen36_prefix_cache_reuse_enabled =
        !config.prefix_cache_reuse_disabled && config.qwen36_hybrid_ssm_snapshot_restore_enabled &&
        env::ParseBoolEnv("DENSECORE_QWEN36_PREFIX_CACHE_REUSE", false);
    config.graph_cache_reuse_disabled = env::ParseDiagnosticEnv("DENSECORE_DEBUG_DISABLE_GRAPH_CACHE_REUSE", false);
    config.moe_trace_plumbing_disabled = env::ParseDiagnosticEnv("DENSECORE_DEBUG_DISABLE_MOE_TRACE_PLUMBING", false);
    config.moe_graph_summary = env::ParseDiagnosticEnv("DENSECORE_DEBUG_MOE_GRAPH_SUMMARY", false);
    config.zero_fill_prefill_input_buffer =
        env::ParseDiagnosticEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_INPUT_BUFFER", false);
    config.zero_fill_prefill_graph_buffer =
        env::ParseDiagnosticEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_GRAPH_BUFFER", false);
    config.zero_fill_prefill_kv_blocks = env::ParseDiagnosticEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_KV_BLOCKS", false);
    config.debug_sample = IsDiagnosticEnvPresent("DENSECORE_DEBUG_SAMPLE");
    config.debug_sample_top = IsDiagnosticEnvPresent("DENSECORE_DEBUG_SAMPLE_TOP");
    config.debug_sampler_trace = IsDiagnosticEnvPresent("DENSECORE_DEBUG_SAMPLER_TRACE");
    config.debug_first_token_raw = IsDiagnosticEnvPresent("DENSECORE_DEBUG_FIRST_TOKEN_RAW");
    config.prefill_thread_override = env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_PREFILL_THREADS", 0);
    config.callback_mode = ParseCallbackMode();
    return config;
}

EngineRuntimeDebugConfig LoadEngineRuntimeDebugConfig() {
    EngineRuntimeDebugConfig config;
    config.verbose_token_trace = env::ParseDiagnosticEnv("DENSECORE_VERBOSE_TOKEN_TRACE", false);
    config.runtime_path_logging = env::ParseDiagnosticEnv("DENSECORE_DEBUG_RUNTIME_PATH", false);
    config.runtime_path_token_logging = env::ParseDiagnosticEnv("DENSECORE_DEBUG_RUNTIME_PATH_TOKENS", false);
    config.parity_debug = env::ParseDiagnosticEnv("DENSECORE_PARITY_DEBUG", false);
    return config;
}

EngineAneBootstrapConfig LoadEngineAneBootstrapConfig() {
    EngineAneBootstrapConfig config;
    config.bootstrap_buckets = true;
    const std::string layer_prefix = ReadStringEnv("DENSECORE_ANE_LAYER_PREFIX");
    if (!layer_prefix.empty()) {
        config.layer_prefix = layer_prefix;
    }
    return config;
}

int ReadPositiveIntEnv(const char* name, int default_value, bool* was_set) {
    const char* env_value = std::getenv(name);
    if (!env_value || env_value[0] == '\0') {
        if (was_set) {
            *was_set = false;
        }
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env_value, &end, 10);
    if (end == env_value || end == nullptr || *end != '\0' || parsed <= 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        if (was_set) {
            *was_set = false;
        }
        return default_value;
    }

    if (was_set) {
        *was_set = true;
    }
    return static_cast<int>(parsed);
}

std::string ReadStringEnv(const char* name) {
    const char* env_value = std::getenv(name);
    if (!env_value || env_value[0] == '\0') {
        return {};
    }
    return std::string(env_value);
}

bool ReadBoolEnv(const char* name, bool default_value) {
    return env::ParseBoolEnv(name, default_value);
}

std::string RemovedEnvironmentAliasError() {
    struct RemovedAlias {
        const char* alias;
        const char* replacement;
    };
    static constexpr RemovedAlias kRemovedAliases[] = {
        {"DENSECORE_ALLOW_EXCEED_BASE_THREADS", "DENSECORE_ALLOW_DECODE_THREADS_OVER_BASE"},
        {"DENSECORE_PAGED_ATTN_DECODE_MIN_CONTEXT", "DENSECORE_PAGED_DECODE_MIN_CONTEXT"},
        {"DENSECORE_KV_SLIDING_WINDOW", "DENSECORE_SLIDING_WINDOW_SIZE"},
        {"DENSECORE_KV_SINK_TOKENS", "DENSECORE_SINK_TOKENS"},
        {"DENSECORE_DIRECT_CALLBACK", "DENSECORE_CALLBACK_MODE"},
        {"DENSECORE_PAGED_ATTN_DECODE_MIN_BATCH_CONTEXT", "DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT"},
    };

    for (const RemovedAlias& removed : kRemovedAliases) {
        if (std::getenv(removed.alias) != nullptr) {
            return std::string("environment variable ") + removed.alias + " was removed; use " + removed.replacement;
        }
    }
    return {};
}

KVCacheRuntimeConfig LoadKVCacheRuntimeConfig() {
    KVCacheRuntimeConfig config;
    config.use_bulk_slot_path = env::ParseBoolEnv("DENSECORE_KV_USE_BULK_PATH", true);
#if defined(__linux__)
    // Transparent hugepages measurably cut KV-cache TLB pressure and are the
    // qualified default. Hosts that need them off should set the kernel's THP
    // policy rather than a DenseCore-specific switch.
    config.use_hugepages = true;
#else
    config.use_hugepages = false;
#endif
    return config;
}

DecodePagedAttentionPolicy LoadDecodePagedAttentionPolicy() {
    DecodePagedAttentionPolicy policy;
    policy.mode = DecodePagedAttentionMode::On;

    const densecore::simd::SimdLevel simd = densecore::simd::DetectSimdLevel();
    const bool has_avx2_or_better = densecore::simd::HasX86Avx2OrBetter(simd);
    const bool has_avx512_or_better = densecore::simd::HasX86Avx512OrBetter(simd);
    const bool is_arm = densecore::simd::IsArmFamily(simd);
    const int default_min_context = has_avx2_or_better ? 128 : (is_arm ? 64 : 256);
    const int default_min_batched_context = (has_avx2_or_better && !has_avx512_or_better) ? 64 : default_min_context;
    policy.min_context_tokens = env::ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_CONTEXT", default_min_context);
    policy.min_batched_context_tokens =
        env::ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", default_min_batched_context);
    // Below these shapes the paged decode kernel loses to the standard path.
    policy.min_head_dim = 64;
    policy.min_heads = 8;

    const bool legacy_allow_q8 = env::ParseBoolEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8", true);
    policy.allow_quantized_auto = env::ParseBoolEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED", legacy_allow_q8);
    policy.debug_log = env::ParseDiagnosticEnv("DENSECORE_DEBUG_PAGED_ATTN_DECODE", false);
    return policy;
}

KVRetentionPolicy LoadKVRetentionPolicy() {
    KVRetentionPolicy policy;
    policy.sliding_window = env::ParseIntEnv("DENSECORE_SLIDING_WINDOW_SIZE", -1);
    policy.sink_tokens = env::ParseIntEnv("DENSECORE_SINK_TOKENS", 0);

    if (policy.sliding_window < 0) {
        policy.sliding_window = -1;
    }
    if (policy.sink_tokens < 0) {
        policy.sink_tokens = 0;
    }
    policy.enabled = policy.sliding_window >= 0;
    return policy;
}

KVRetentionSpan ComputeKVRetentionSpan(int n_past, const KVRetentionPolicy& policy) {
    KVRetentionSpan span;
    if (n_past <= 0) {
        return span;
    }

    if (!policy.enabled || policy.sliding_window < 0) {
        span.history_kept = n_past;
        span.sink_kept = n_past;
        span.tail_start = n_past;
        return span;
    }

    span.sink_kept = std::clamp(policy.sink_tokens, 0, n_past);
    span.tail_start = std::max(span.sink_kept, n_past - std::max(0, policy.sliding_window));
    span.history_kept = span.sink_kept + std::max(0, n_past - span.tail_start);
    return span;
}

int MapRetainedHistoryIndex(const KVRetentionSpan& span, int retained_index) {
    if (retained_index < span.sink_kept) {
        return retained_index;
    }
    return span.tail_start + (retained_index - span.sink_kept);
}

env::RuntimeToggleMode LoadArmQ4KNativeVecDotMode() {
    return env::RuntimeToggleMode::On;
}

env::RuntimeToggleMode LoadArmInt4DirectFastPathMode() {
    return env::RuntimeToggleMode::On;
}

FastPathRuntimeConfig LoadFastPathRuntimeConfig() {
    FastPathRuntimeConfig config;
    config.worker = LoadWorkerRuntimeConfig();
    config.engine_debug = LoadEngineRuntimeDebugConfig();
    config.decode_paged_attention = LoadDecodePagedAttentionPolicy();
    config.kv_retention = LoadKVRetentionPolicy();
    config.bench_respect_threads = env::ParseBoolEnv("DENSECORE_BENCH_RESPECT_THREADS", false);
    config.prefill_graph_cache.enabled = true;
    config.prefill_graph_cache.lru_size =
        std::max(1, env::ParsePositiveEnvInt("DENSECORE_PREFILL_GRAPH_CACHE_LRU", 16));
    const std::size_t available_mb = ReadAvailableMemoryMbForAutoRuntimeCache();
    const int auto_prefill_graph_cache_mb =
        available_mb > 0 ? static_cast<int>(std::max<std::size_t>(128, available_mb / 32)) : 1024;
    const int prefill_graph_cache_mb =
        std::max(128, ReadPositiveIntEnv("DENSECORE_PREFILL_GRAPH_CACHE_MAX_MB", auto_prefill_graph_cache_mb));
    config.prefill_graph_cache.max_bytes = static_cast<std::size_t>(prefill_graph_cache_mb) * 1024ULL * 1024ULL;
    config.qwen36_prefill_q4k_batched = Qwen36PrefillQ4KBatchedMode::On;
    config.qwen36_ssm_q8_prefill_amx = DefaultQwen36SSMQ8PrefillAMXMode();
    config.qwen36_ssm_q8_prefill_amx_min_tokens = 256;
    config.native_moe_fast_decode = env::RuntimeToggleMode::On;
    config.q4k_repacked_gemv = env::RuntimeToggleMode::On;
    config.q4k_repacked_gemv_allow_prefill = true;
    config.q4k_repacked_gemv_probe = false;
    config.q4k_repacked_gemv_disable_on_thrash = false;
    config.q4k_repacked_gemv_thrash_repack_mb = 256;
    config.q4k_repacked_gemv_thrash_eviction_ratio = 0.25;
    config.q4k_repacked_gemv_thrash_repack_cache_fraction = 0.50;
    // Optional global qact-cache promotion is deliberately fail-closed. The
    // validated Gemma4/LFM2 batched paths enable their required activation cache
    // through model/path admission, not through an environment knob.
    config.qact_cache = env::RuntimeToggleMode::Off;
    config.matmul_dispatch_census = env::ParseDiagnosticEnv("DENSECORE_MATMUL_DISPATCH_CENSUS", false);
    return config;
}

}  // namespace densecore::llm::config
