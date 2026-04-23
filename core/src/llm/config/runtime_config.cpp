#include "llm/config/runtime_config.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "densecore/simd/simd_ops.h"

namespace densecore::llm::config {
namespace {

bool IsLinuxHugepagesDisabled() {
    const char* value = std::getenv("DENSECORE_DISABLE_HUGEPAGES");
    return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0);
}

bool ParseLegacyEnabledBool(const char* name, bool default_value) {
    const char* env_value = std::getenv(name);
    if (!env_value || env_value[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env_value, "0") != 0;
}

}  // namespace

WorkerRuntimeConfig LoadWorkerRuntimeConfig() {
    WorkerRuntimeConfig config;
    config.validate_mul = env::ParseNonZeroEnv("DENSECORE_DEBUG_VALIDATE_MUL", false);
    config.runtime_path_logging = env::ParseNonZeroEnv("DENSECORE_DEBUG_RUNTIME_PATH", false);
    config.scheduler_stall_debug = env::ParseNonZeroEnv("DENSECORE_SCHEDULER_STALL_DEBUG", false);
    config.moe_trace_dump = env::ParseNonZeroEnv("DENSECORE_DEBUG_MOE_TRACE_DUMP", false);
    config.sampler_trace_dump = env::ParseNonZeroEnv("DENSECORE_DEBUG_SAMPLER_TRACE_DUMP", false);
    config.determinism_boundary_debug = env::ParseNonZeroEnv("DENSECORE_DEBUG_DETERMINISM_BOUNDARY", false);
    config.prefix_cache_reuse_disabled = env::ParseNonZeroEnv("DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE", false);
    config.hybrid_ssm_snapshot_restore_disabled =
        env::ParseNonZeroEnv("DENSECORE_DEBUG_DISABLE_HYBRID_SSM_RESTORE", false);
    config.graph_cache_reuse_disabled = env::ParseNonZeroEnv("DENSECORE_DEBUG_DISABLE_GRAPH_CACHE_REUSE", false);
    config.moe_trace_plumbing_disabled = env::ParseNonZeroEnv("DENSECORE_DEBUG_DISABLE_MOE_TRACE_PLUMBING", false);
    config.moe_graph_summary = env::ParseNonZeroEnv("DENSECORE_DEBUG_MOE_GRAPH_SUMMARY", false);
    config.zero_fill_prefill_input_buffer =
        env::ParseNonZeroEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_INPUT_BUFFER", false);
    config.zero_fill_prefill_graph_buffer =
        env::ParseNonZeroEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_GRAPH_BUFFER", false);
    config.zero_fill_prefill_kv_blocks = env::ParseNonZeroEnv("DENSECORE_DEBUG_ZERO_FILL_PREFILL_KV_BLOCKS", false);
    config.prefill_thread_override = env::ParsePositiveEnvInt("DENSECORE_DEBUG_PREFILL_THREADS", 0);
    return config;
}

EngineRuntimeDebugConfig LoadEngineRuntimeDebugConfig() {
    EngineRuntimeDebugConfig config;
    config.verbose_token_trace = env::ParseNonZeroEnv("DENSECORE_VERBOSE_TOKEN_TRACE", false);
    config.runtime_path_logging = env::ParseNonZeroEnv("DENSECORE_DEBUG_RUNTIME_PATH", false);
    config.runtime_path_token_logging = env::ParseNonZeroEnv("DENSECORE_DEBUG_RUNTIME_PATH_TOKENS", false);
    config.parity_debug = env::ParseNonZeroEnv("DENSECORE_PARITY_DEBUG", false);
    return config;
}

EngineAneBootstrapConfig LoadEngineAneBootstrapConfig() {
    EngineAneBootstrapConfig config;
    config.bootstrap_buckets = env::ParseNonZeroEnv("DENSECORE_ANE_BUCKET_BOOTSTRAP", true);
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
    const std::string lowered = env::AsciiLowerCopy(std::getenv(name));
    if (lowered.empty()) {
        return default_value;
    }
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

KVCacheRuntimeConfig LoadKVCacheRuntimeConfig() {
    KVCacheRuntimeConfig config;
    config.use_bulk_slot_path = env::ParseNonZeroEnv("DENSECORE_KV_USE_BULK_PATH", true);
#if defined(__linux__)
    config.use_hugepages = !IsLinuxHugepagesDisabled();
#else
    config.use_hugepages = env::ParseTruthyEnv("DENSECORE_USE_HUGEPAGES", false);
#endif
    return config;
}

DecodePagedAttentionMode LoadDecodePagedAttentionMode() {
    if (env::ParseTruthyEnv("DENSECORE_FORCE_PAGED_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    const char* explicit_mode_env = std::getenv("DENSECORE_PAGED_ATTN_DECODE_MODE");
    switch (env::ParseRuntimeToggleMode("DENSECORE_PAGED_ATTN_DECODE_MODE", env::RuntimeToggleMode::Auto)) {
    case env::RuntimeToggleMode::Off: return DecodePagedAttentionMode::Off;
    case env::RuntimeToggleMode::On: return DecodePagedAttentionMode::On;
    case env::RuntimeToggleMode::Auto:
        if (explicit_mode_env && explicit_mode_env[0] != '\0') {
            return DecodePagedAttentionMode::Auto;
        }
        break;
    }

    if (env::ParseTruthyEnv("DENSECORE_ENABLE_PAGED_ATTN_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    return DecodePagedAttentionMode::On;
}

DecodePagedAttentionPolicy LoadDecodePagedAttentionPolicy() {
    DecodePagedAttentionPolicy policy;
    policy.mode = LoadDecodePagedAttentionMode();

    const densecore::simd::SimdLevel simd = densecore::simd::DetectSimdLevel();
    const bool has_avx2_or_better = densecore::simd::HasX86Avx2OrBetter(simd);
    const bool has_avx512_or_better = densecore::simd::HasX86Avx512OrBetter(simd);
    const bool is_arm = densecore::simd::IsArmFamily(simd);
    const int default_min_context = has_avx2_or_better ? 128 : (is_arm ? 64 : 256);
    const int default_min_batched_context = (has_avx2_or_better && !has_avx512_or_better) ? 64 : default_min_context;
    const int legacy_min_context =
        env::ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_CONTEXT", default_min_context);

    policy.min_context_tokens = env::ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_CONTEXT", legacy_min_context);
    const int legacy_min_batched_context =
        env::ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_BATCH_CONTEXT", default_min_batched_context);
    policy.min_batched_context_tokens =
        env::ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_BATCH_CONTEXT", legacy_min_batched_context);
    policy.min_head_dim = env::ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEAD_DIM", 64);
    policy.min_heads = env::ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEADS", 8);

    const bool legacy_allow_q8 = env::ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8", true);
    policy.allow_quantized_auto = env::ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED", legacy_allow_q8);
    policy.debug_log = env::ParseTruthyEnv("DENSECORE_DEBUG_PAGED_ATTN_DECODE", false);
    return policy;
}

KVRetentionPolicy LoadKVRetentionPolicy() {
    KVRetentionPolicy policy;
    policy.sliding_window =
        env::ParseIntEnv("DENSECORE_KV_SLIDING_WINDOW", env::ParseIntEnv("DENSECORE_SLIDING_WINDOW_SIZE", -1));
    policy.sink_tokens = env::ParseIntEnv("DENSECORE_KV_SINK_TOKENS", env::ParseIntEnv("DENSECORE_SINK_TOKENS", 0));

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
#if defined(__aarch64__) || defined(_M_ARM64)
    const char* legacy = std::getenv("DENSECORE_ARM_ALLOW_Q4K_NATIVE_VECDOT");
    if (legacy && legacy[0] != '\0') {
        if (std::strcmp(legacy, "0") == 0 || std::strcmp(legacy, "false") == 0 || std::strcmp(legacy, "FALSE") == 0) {
            return env::RuntimeToggleMode::Off;
        }
        return env::RuntimeToggleMode::On;
    }

    return env::ParseRuntimeToggleMode("DENSECORE_ARM_Q4K_NATIVE_VECDOT_MODE", env::RuntimeToggleMode::Auto);
#else
    return env::RuntimeToggleMode::On;
#endif
}

env::RuntimeToggleMode LoadArmInt4DirectFastPathMode() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return env::ParseRuntimeToggleMode("DENSECORE_ARM_INT4_DIRECT_FASTPATH_MODE", env::RuntimeToggleMode::Auto);
#else
    return env::RuntimeToggleMode::On;
#endif
}

FastPathRuntimeConfig LoadFastPathRuntimeConfig() {
    FastPathRuntimeConfig config;
    config.worker = LoadWorkerRuntimeConfig();
    config.engine_debug = LoadEngineRuntimeDebugConfig();
    config.decode_paged_attention = LoadDecodePagedAttentionPolicy();
    config.kv_retention = LoadKVRetentionPolicy();
    config.bench_respect_threads = env::ParseTruthyEnv("DENSECORE_BENCH_RESPECT_THREADS", false);
    config.prefill_graph_cache.enabled = ParseLegacyEnabledBool("DENSECORE_PREFILL_GRAPH_CACHE", true);
    config.prefill_graph_cache.lru_size = std::max(1, env::ParsePositiveEnvInt("DENSECORE_PREFILL_GRAPH_CACHE_LRU", 16));
    const int prefill_graph_cache_mb =
        std::max(128, env::ParsePositiveEnvInt("DENSECORE_PREFILL_GRAPH_CACHE_MAX_MB", 1024));
    config.prefill_graph_cache.max_bytes = static_cast<std::size_t>(prefill_graph_cache_mb) * 1024ULL * 1024ULL;
    return config;
}

}  // namespace densecore::llm::config
