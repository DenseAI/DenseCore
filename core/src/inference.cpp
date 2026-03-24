#include "inference.h"

#include "densecore/inference_types_internal.h"          // Shared internal types
#include "densecore/models/transformer_graph_builder.h"  // Strategy Pattern for graph building
#include "flash_attention.h"
#include "ggml-cpu.h"           // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"               // Required for ggml_tensor definition
#include "hardware_topology.h"  // For compute thread affinity
#include "matmul_backend.h"
#include "optimization_bridge.h"  // Runtime SIMD dispatch

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "dtype_utils.h"  // For GgmlTypeToDType
#include "kernels/hwy/hwy_kernels.h"
#include "kv_cache.h"  // Added for KV cache
#include "memory_pool.h"
#include "moe/moe_routing.h"
#include "quantization/int4_types.h"  // For TensorInt4
#include "scheduler.h"
#include "simd_ops.h"

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif

#ifndef DENSECORE_DEFAULT_PRECOMPUTED_ROPE
#define DENSECORE_DEFAULT_PRECOMPUTED_ROPE 1
#endif

#ifndef DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
#define DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM 0
#endif

#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

namespace {
static const InferenceConfig& ResolveInferenceConfig(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->config) {
        return *batch->deps->config;
    }
    return InferenceConfig::Instance();
}

static densecore::HardwareTopology& ResolveHardwareTopology(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->hardware_topology) {
        return *batch->deps->hardware_topology;
    }
    return densecore::HardwareTopology::GetInstance();
}

static densecore::BackendRegistry& ResolveBackendRegistry(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->backend_registry) {
        return *batch->deps->backend_registry;
    }
    return densecore::BackendRegistry::Instance();
}

static densecore::DeviceType ResolvePreferredDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredMatmulDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_matmul_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredAttentionDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_attention_device;
    }
    return densecore::DeviceType::CPU;
}

static densecore::DeviceType ResolvePreferredNormDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_norm_device;
    }
    return densecore::DeviceType::CPU;
}

static bool IsMixedRoutingEnabled(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->mixed_operation_routing;
    }
    return false;
}

static bool IsVerboseGraphBuildLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_VERBOSE_GRAPH_BUILD");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugInferenceStatsEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_INFERENCE_STATS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugGemvSelectionEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMV_SELECTION");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulDispatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_DISPATCH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Dispatch path labels for instrumentation
static const char* MatmulWeightTypeLabel(ggml_type wtype, bool is_packed_int4, bool is_packed_fp8) {
    if (is_packed_int4) return "PACKED_INT4";
    if (is_packed_fp8) return "PACKED_FP8";
    if (ggml_is_quantized(wtype)) return "GGML_QUANT";
    if (wtype == GGML_TYPE_F32) return "FLOAT_F32";
    if (wtype == GGML_TYPE_F16) return "FLOAT_F16";
    if (wtype == GGML_TYPE_BF16) return "FLOAT_BF16";
    return "UNKNOWN";
}

static const char* DetectedISATier() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

static void LogMatmulDispatch(const char* weight_name, const char* weight_type_label, int M, int N, int K,
                              const char* path_label, const char* fallback_reason = nullptr) {
    if (!IsDebugMatmulDispatchEnabled()) return;
    if (fallback_reason) {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s fallback=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label,
                fallback_reason);
    } else {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label);
    }
}

// Env-tunable thresholds declared here, defined after ParsePositiveEnvInt.
static int GetBatchedMinM();
static int GetBatchedMinN();
static int GetBatchedMinK();

static void LogMatmulPathOnce(const char* path) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::atomic<bool> logged_ggml_mul_mat{false};
    static std::atomic<bool> logged_gemv_batched_quant_nrc{false};
    std::atomic<bool>* once_flag = nullptr;

    if (std::strcmp(path, "ggml_mul_mat") == 0) {
        once_flag = &logged_ggml_mul_mat;
    } else if (std::strcmp(path, "gemv_batched_quant_nrc") == 0) {
        once_flag = &logged_gemv_batched_quant_nrc;
    } else {
        return;
    }

    bool expected = false;
    if (once_flag->compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        fprintf(stderr, "[DenseCore][MatmulPath] %s\n", path);
    }
}

static bool IsCustomGemvDisabled() {
    static const bool disabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_DISABLE_CUSTOM_GEMV");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const char* mode_env = std::getenv("DENSECORE_CUSTOM_GEMV_MODE");
        if (mode_env && mode_env[0] != '\0') {
            std::string mode(mode_env);
            for (char& c : mode) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (mode == "on" || mode == "1" || mode == "true" || mode == "force") {
                return false;
            }
            if (mode == "off" || mode == "0" || mode == "false") {
                return true;
            }
        }

        // Auto mode: keep custom GEMV enabled unless architecture-specific
        // logic in smart_mul_mat selects native GGML GEMV for better throughput.
        return false;
    }();
    return disabled;
}

// Legacy opt-in (kept for backward compat, but batched quant is now always-on by default)
static bool IsSmallBatchQuantGemmEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_SMALL_BATCH_GEMV_QUANT");
        if (!env || env[0] == '\0') {
            return true;
        }
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0 ||
            std::strcmp(env, "off") == 0 || std::strcmp(env, "OFF") == 0 || std::strcmp(env, "no") == 0 ||
            std::strcmp(env, "NO") == 0) {
            return false;
        }
        return true;
    }();
    return enabled;
}

// Primary opt-out: DENSECORE_DISABLE_BATCHED_QUANT=1 disables the shared-quant nrc=M fast path.
// Batched quant is ON by default for all quant weights with M>=2.
static bool IsBatchedQuantDisabled() {
    static const bool disabled = []() {
        const char* env = std::getenv("DENSECORE_DISABLE_BATCHED_QUANT");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        // Honor legacy env as well
        return !IsSmallBatchQuantGemmEnabled();
    }();
    return disabled;
}

// Enable nrc-batched quant path by default and rely on per-type runtime checks
// (vec_dot nrows >= M) before dispatching to the fast path.
// DENSECORE_ENABLE_QUANT_NRC_BATCH=0 can be used to force-disable it.
static bool IsQuantNrcBatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_ENABLE_QUANT_NRC_BATCH");
        if (env && env[0] != '\0') {
            return std::strcmp(env, "0") != 0;
        }
        return true;
    }();
    return enabled;
}

// Deprecated path: keep disabled until a correctness/perf-positive
// implementation is available.
static bool IsQ4KTrueBatchedKernelEnabled() {
    return false;
}

static bool IsQ4KTrueBatchedAvx2Enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_ENABLE_Q4K_BATCHED_AVX2");
        if (!env || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static inline void SpinPause(int spin_count) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if ((spin_count & 0x3F) != 0) {
        _mm_pause();
        return;
    }
#elif defined(__aarch64__)
    if ((spin_count & 0x3F) != 0) {
        asm volatile("yield");
        return;
    }
#endif
    std::this_thread::yield();
}

static uint64_t ComputeGemvBatchedQuantStamp(const BatchSpec* batch, int M, int slot_id, const void* src_data_ptr,
                                             const void* weight_data_ptr) {
    // FNV-1a over stable per-op metadata to avoid stale-buffer reuse when
    // slot_id repeats across different batched inputs.
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ull;
    };

    mix(static_cast<uint64_t>(static_cast<uint32_t>(slot_id)));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(M)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_data_ptr)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_data_ptr)));
    if (batch) {
        mix(static_cast<uint64_t>(batch->tokens.size()));
        const int n_pos = std::min(M, static_cast<int>(batch->pos.size()));
        for (int i = 0; i < n_pos; ++i) {
            mix(static_cast<uint64_t>(static_cast<uint32_t>(batch->pos[static_cast<size_t>(i)])));
        }
    }

    // Zero is the reset sentinel for stamps.
    return hash == 0 ? 1 : hash;
}

enum class DecodePagedAttentionMode { Off = 0, Auto = 1, On = 2 };

struct DecodePagedAttentionPolicy {
    DecodePagedAttentionMode mode = DecodePagedAttentionMode::Auto;
    int min_context_tokens = 256;
    int min_head_dim = 64;
    int min_heads = 8;
    bool allow_quantized_auto = false;
    bool debug_log = false;
};

static int ParsePositiveEnvInt(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static int ParseIntEnv(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        return default_value;
    }
    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

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

static KVRetentionPolicy LoadKVRetentionPolicy() {
    KVRetentionPolicy policy;
    policy.sliding_window =
        ParseIntEnv("DENSECORE_KV_SLIDING_WINDOW", ParseIntEnv("DENSECORE_SLIDING_WINDOW_SIZE", -1));
    policy.sink_tokens = ParseIntEnv("DENSECORE_KV_SINK_TOKENS", ParseIntEnv("DENSECORE_SINK_TOKENS", 0));

    if (policy.sliding_window < 0) {
        policy.sliding_window = -1;
    }
    if (policy.sink_tokens < 0) {
        policy.sink_tokens = 0;
    }
    policy.enabled = (policy.sliding_window >= 0);
    return policy;
}

static const KVRetentionPolicy& GetKVRetentionPolicy() {
    static const KVRetentionPolicy policy = LoadKVRetentionPolicy();
    return policy;
}

static KVRetentionSpan ComputeKVRetentionSpan(int n_past, const KVRetentionPolicy& policy) {
    KVRetentionSpan span;
    if (n_past <= 0) {
        return span;
    }

    if (!policy.enabled || policy.sliding_window < 0) {
        span.history_kept = n_past;
        // No sliding-window retention: preserved history must remain in
        // identity order [0, n_past) for all downstream KV readers.
        span.sink_kept = n_past;
        span.tail_start = n_past;
        return span;
    }

    span.sink_kept = std::clamp(policy.sink_tokens, 0, n_past);
    span.tail_start = std::max(span.sink_kept, n_past - std::max(0, policy.sliding_window));
    span.history_kept = span.sink_kept + std::max(0, n_past - span.tail_start);
    return span;
}

static int MapRetainedHistoryIndex(const KVRetentionSpan& span, int retained_index) {
    if (retained_index < span.sink_kept) {
        return retained_index;
    }
    return span.tail_start + (retained_index - span.sink_kept);
}

// Env-tunable thresholds for batched GEMM/GEMV routing
static int GetBatchedMinM() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_M", 2);
    return val;
}

static int GetBatchedMinN() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_N", 32);
    return val;
}

static int GetBatchedMinK() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_K", 32);
    return val;
}

static bool ParseTruthyEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0 ||
           std::strcmp(env, "yes") == 0 || std::strcmp(env, "YES") == 0 || std::strcmp(env, "on") == 0 ||
           std::strcmp(env, "ON") == 0;
}

static std::string AsciiLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

enum class RuntimeToggleMode { Off = 0, Auto = 1, On = 2 };

static RuntimeToggleMode ParseRuntimeToggleMode(const char* name, RuntimeToggleMode default_mode) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_mode;
    }

    const std::string mode = AsciiLower(env);
    if (mode == "off" || mode == "0" || mode == "false" || mode == "no") {
        return RuntimeToggleMode::Off;
    }
    if (mode == "on" || mode == "1" || mode == "true" || mode == "yes" || mode == "force") {
        return RuntimeToggleMode::On;
    }
    if (mode == "auto") {
        return RuntimeToggleMode::Auto;
    }
    return default_mode;
}

static int ResolveTaskCount(const BatchSpec* batch, int work_items) {
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    if (work_items > 0) {
        n_tasks = std::min(n_tasks, work_items);
    }
    return std::max(1, n_tasks);
}

static densecore::simd::SimdLevel GetRuntimeSimdLevel() {
    static const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return level;
}

static DecodePagedAttentionMode ParseDecodePagedAttentionMode() {
    if (ParseTruthyEnv("DENSECORE_FORCE_PAGED_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    const char* mode_env = std::getenv("DENSECORE_PAGED_ATTN_DECODE_MODE");
    if (mode_env && mode_env[0] != '\0') {
        const std::string mode = AsciiLower(mode_env);
        if (mode == "off" || mode == "0" || mode == "false") {
            return DecodePagedAttentionMode::Off;
        }
        if (mode == "on" || mode == "1" || mode == "true" || mode == "force") {
            return DecodePagedAttentionMode::On;
        }
        if (mode == "auto") {
            return DecodePagedAttentionMode::Auto;
        }
    }

    // Backward-compatible env fallback.
    if (ParseTruthyEnv("DENSECORE_ENABLE_PAGED_ATTN_DECODE", false)) {
        return DecodePagedAttentionMode::On;
    }

    return DecodePagedAttentionMode::Auto;
}

static DecodePagedAttentionPolicy LoadDecodePagedAttentionPolicy() {
    DecodePagedAttentionPolicy policy;
    policy.mode = ParseDecodePagedAttentionMode();
    const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
    const bool has_avx2_or_better = densecore::simd::HasX86Avx2OrBetter(simd);
    const bool is_arm = densecore::simd::IsArmFamily(simd);
    const int default_min_context = has_avx2_or_better ? 128 : (is_arm ? 64 : 256);
    const int legacy_min_context = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_CONTEXT", default_min_context);
    policy.min_context_tokens = ParsePositiveEnvInt("DENSECORE_PAGED_DECODE_MIN_CONTEXT", legacy_min_context);
    policy.min_head_dim = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEAD_DIM", 64);
    policy.min_heads = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_MIN_HEADS", 8);
    const bool legacy_allow_q8 = ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_Q8", false);
    policy.allow_quantized_auto = ParseTruthyEnv("DENSECORE_PAGED_ATTN_DECODE_ALLOW_QUANTIZED", legacy_allow_q8);
    policy.debug_log = ParseTruthyEnv("DENSECORE_DEBUG_PAGED_ATTN_DECODE", false);
    return policy;
}

struct DecodeContextSummary {
    int min_context = 0;
    int max_context = 0;
    int avg_context = 0;
    bool valid = false;
};

static bool IsBatchedPagedDecodeEnabled() {
    static const bool enabled = ParseTruthyEnv("DENSECORE_ENABLE_BATCHED_PAGED_DECODE", false);
    return enabled;
}

static bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch) {
    if (n_tokens_in_batch <= 0) {
        return false;
    }
    if (batch.num_seqs != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.tokens.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.seq_id.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.pos.size()) != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.block_tables.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.n_past.size()) != n_tokens_in_batch) {
        return false;
    }

    std::vector<uint8_t> seen(static_cast<size_t>(n_tokens_in_batch), 0);
    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch) {
            return false;
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(seq_idx)] = 1;
    }

    return true;
}

static DecodeContextSummary SummarizeDecodeContext(const BatchSpec& batch, int n_tokens_in_batch) {
    DecodeContextSummary summary;
    if (!IsDecodeOnlyBatchLayout(batch, n_tokens_in_batch)) {
        return summary;
    }

    summary.min_context = std::numeric_limits<int>::max();
    summary.max_context = 0;
    int64_t sum_context = 0;
    std::vector<uint8_t> seen(static_cast<size_t>(batch.num_seqs), 0);

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
            return DecodeContextSummary{};
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return DecodeContextSummary{};
        }
        seen[static_cast<size_t>(seq_idx)] = 1;

        const int pos_i = batch.pos[static_cast<size_t>(i)];
        if (pos_i < 0) {
            return DecodeContextSummary{};
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodeContextSummary{};
        }
        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodeContextSummary{};
        }

        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (n_past_i < 0) {
            return DecodeContextSummary{};
        }

        const KVRetentionSpan retained = ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const int max_context_i = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len_i = std::max(1, std::min(retained.history_kept + 1, max_context_i));
        summary.min_context = std::min(summary.min_context, context_len_i);
        summary.max_context = std::max(summary.max_context, context_len_i);
        sum_context += context_len_i;
    }

    summary.avg_context = static_cast<int>((sum_context + n_tokens_in_batch - 1) / n_tokens_in_batch);
    summary.valid = true;
    return summary;
}

static bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                                   int n_head_kv, int head_dim_q, int head_dim_kv) {
    if (!cache) {
        return false;
    }
    if (!IsDecodeOnlyBatchLayout(batch, n_tokens_in_batch)) {
        return false;
    }
    if (n_head_kv <= 0 || n_head <= 0) {
        return false;
    }
    if (n_head % n_head_kv != 0) {
        return false;
    }
    if (head_dim_q != head_dim_kv) {
        return false;
    }

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        const int pos_i = batch.pos[static_cast<size_t>(i)];
        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (pos_i < 0 || n_past_i < 0) {
            return false;
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return false;
        }

        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return false;
        }
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= cache->max_blocks) {
            return false;
        }
    }

    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        return false;
    }

    return true;
}

static bool ShouldUsePagedDecodeAttention(const DecodePagedAttentionPolicy& policy, const PagedKVCache* cache,
                                          const BatchSpec& batch, int n_tokens_in_batch, int n_head, int n_head_kv,
                                          int head_dim_q, int head_dim_kv) {
    if (policy.mode == DecodePagedAttentionMode::Off) return false;
    if (!IsPagedDecodeCandidate(cache, batch, n_tokens_in_batch, n_head, n_head_kv, head_dim_q, head_dim_kv)) {
        return false;
    }

    if (policy.mode == DecodePagedAttentionMode::On) {
        return true;
    }

    // Auto mode: enable only when context is long enough to amortize callback/setup overhead.
    if (!policy.allow_quantized_auto && ggml_is_quantized(cache->cache_type)) {
        return false;
    }
    if (n_head < policy.min_heads) {
        return false;
    }
    if (head_dim_q < policy.min_head_dim) {
        return false;
    }
    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, n_tokens_in_batch);
    if (!context_summary.valid) {
        return false;
    }
    // Use the shortest context in the decode micro-batch as representative.
    // This prevents Auto mode from enabling paged decode only because a subset
    // of sequences has long history.
    const int context_len = context_summary.min_context;
    if (context_len < policy.min_context_tokens) {
        return false;
    }

    return true;
}

static bool IsFlashAttentionDisabled() {
    static const bool disabled = []() {
        if (ParseRuntimeToggleMode("DENSECORE_FLASH_ATTN_MODE", RuntimeToggleMode::Auto) == RuntimeToggleMode::Off) {
            return true;
        }
        const char* legacy_env = std::getenv("DENSECORE_DISABLE_FLASH_ATTN");
        return legacy_env && legacy_env[0] != '\0' && std::strcmp(legacy_env, "0") != 0;
    }();
    return disabled;
}

static bool IsPagedAttentionHwyEnabled() {
    static const bool enabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_PAGED_ATTN_USE_HWY");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const RuntimeToggleMode mode = ParseRuntimeToggleMode("DENSECORE_PAGED_ATTN_HWY_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        // Auto mode: use Highway on SIMD-capable hosts and scalar on true
        // scalar-only CPUs.
        return GetRuntimeSimdLevel() != densecore::simd::SimdLevel::NONE;
    }();
    return enabled;
}

static bool IsDecodeProfileEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_PROFILE_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsForceSafeGqaDecodeEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_FORCE_SAFE_GQA_DECODE");
        if (!env || env[0] == '\0') return true;
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsPrefillAttentionSkipContEnabled() {
    static const bool enabled = []() {
        const char* legacy_env = std::getenv("DENSECORE_PREFILL_ATTN_SKIP_CONT");
        if (legacy_env && legacy_env[0] != '\0') {
            return std::strcmp(legacy_env, "0") != 0;
        }

        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_PREFILL_ATTN_SKIP_CONT_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        // Auto mode: keep OFF on AVX2 where regression was observed, enable on
        // AVX-512/AMX class x86 where memory bandwidth pressure is lower.
        return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel());
    }();
    return enabled;
}

static bool IsFlashAttentionForced() {
    static const bool forced = []() { return ParseTruthyEnv("DENSECORE_FORCE_FLASH_ATTN", false); }();
    return forced;
}

static bool IsFlashAttentionIsaSupported() {
    static const bool supported = []() { return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel()); }();
    return supported;
}

static bool IsPortableCpuFlashAttentionEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_PORTABLE_FLASH_ATTN_MODE", RuntimeToggleMode::Auto);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }
        return densecore::simd::IsArmFamily(GetRuntimeSimdLevel());
    }();
    return enabled;
}

static bool IsInt4SingleThreadingLayerEnabled() {
    static const bool enabled = []() {
        const char* legacy = std::getenv("DENSECORE_INT4_USE_BACKEND_THREADPOOL");
        if (legacy && legacy[0] != '\0') {
            return std::strcmp(legacy, "0") == 0;
        }
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_INT4_THREADING_MODE",
            DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsPrecomputedRoPEEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode =
            ParseRuntimeToggleMode("DENSECORE_ROPE_PRECOMPUTED_MODE",
                                   DENSECORE_DEFAULT_PRECOMPUTED_ROPE ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsFusedResidualRmsNormEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_FUSED_RESIDUAL_RMSNORM_MODE",
            DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM ? RuntimeToggleMode::On : RuntimeToggleMode::Off);
        return mode != RuntimeToggleMode::Off;
    }();
    return enabled;
}

static bool IsFusedQKVEnabled() {
    static const bool enabled = []() {
        const RuntimeToggleMode mode = ParseRuntimeToggleMode(
            "DENSECORE_FUSED_QKV_MODE", DENSECORE_DEFAULT_FUSED_QKV ? RuntimeToggleMode::Auto : RuntimeToggleMode::Off);
        if (mode == RuntimeToggleMode::On) {
            return true;
        }
        if (mode == RuntimeToggleMode::Off) {
            return false;
        }

        const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
        return densecore::simd::HasX86Avx2OrBetter(simd) || densecore::simd::IsArmFamily(simd);
    }();
    return enabled;
}
}  // namespace

// ============================================================================
// InferenceContext Implementation ("Rebuild Graph, Reuse Memory" Strategy)
// ============================================================================

void InferenceContext::Init(size_t buffer_size) {
    if (initialized) {
        return;  // Already initialized
    }

    // Allocate aligned buffer for GGML context
    // Use 64-byte alignment for AVX-512 compatibility
    compute_buffer.resize(buffer_size);

    struct ggml_init_params params = {
        .mem_size = buffer_size,
        .mem_buffer = compute_buffer.data(),
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);

    if (ctx_compute) {
        initialized = true;
        std::cerr << "[InferenceContext] Initialized with " << (buffer_size / (1024 * 1024)) << " MB persistent buffer"
                  << std::endl;
    } else {
        throw densecore::OutOfMemoryException("InferenceContext: ggml_init failed");
    }
}

void InferenceContext::Reset() {
    if (!initialized || compute_buffer.empty()) {
        return;
    }

    // GGML doesn't expose a public ggml_reset_pool() API.
    // Workaround: Free and re-init with the SAME memory buffer.
    // This is effectively O(1) since:
    //   - No malloc/free syscalls (buffer is reused)
    //   - ggml_init just sets up internal allocator state
    if (ctx_compute) {
        ggml_free(ctx_compute);
    }

    struct ggml_init_params params = {
        .mem_size = compute_buffer.size(),
        .mem_buffer = compute_buffer.data(),
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);
    if (!ctx_compute) {
        throw densecore::OutOfMemoryException("InferenceContext::Reset: ggml_init failed");
    }
}

void InferenceContext::Free() {
    if (ctx_compute) {
        ggml_free(ctx_compute);
        ctx_compute = nullptr;
    }
    compute_buffer.clear();
    compute_buffer.shrink_to_fit();
    initialized = false;
}

// Forward declaration for explicit work context
struct InferenceWorkContext;
static thread_local InferenceWorkContext* tls_work_ctx = nullptr;
static std::atomic<const BatchSpec*> g_shared_batch{nullptr};


// ============================================================================
// NEW: Robust KV Cache Update and Gather UserData
// ============================================================================
// This replaces the fragile cb_kv_manage approach that relied on ggml_pad
// assumptions. The new approach explicitly:
//   1. Writes current K/V to the PagedKVCache
//   2. Reads history from the cache into destination tensor
//   3. Appends current K/V to destination tensor
// ============================================================================

struct KVUpdateGatherUserData {
    PagedKVCache* cache;             // KV cache instance
    const BatchSpec* batch;          // Batch specification with block tables
    int layer;                       // Current transformer layer
    int head_dim;                    // Dimension per head
    int n_head_kv;                   // Number of KV heads
    int N;                           // Current batch size (new tokens)
    int n_past;                      // Number of past/history tokens
    bool is_k;                       // True for K tensor, false for V tensor
    struct ggml_tensor* src_tensor;  // Pointer to Kcur/Vcur tensor (data accessed at runtime)
};

// Pool size for KVUpdateGatherUserData
static constexpr int kMaxKVUpdateGatherSlots = 256;

KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k);

// ============================================================================
// Multi-LoRA Batching Callback
// ============================================================================
// Applies per-request LoRA adapters during inference graph execution.
// Uses thread-local batch context to access the adapter-to-token mapping.
// The tensor name (e.g., "blk.0.attn_q") identifies which layer weights to use.
// ============================================================================

static const BatchSpec* GetCurrentBatch();

/**
 * @brief GGML callback for Multi-LoRA application during inference.
 *
 * 각 LoRA 가능 레이어(QKV, Output Projection, FFN)에서 호출됩니다.
 * Gather-Compute-Scatter 패턴으로 배치 내 각 토큰에 해당하는 LoRA 어댑터를 적용합니다.
 *
 * Time Complexity: O(N * R * D) where N=tokens, R=rank, D=hidden_dim
 * Space Complexity: O(N * max(R, D)) for scratch buffers (thread-local)
 *
 * @param dst Output tensor (base projection result, LoRA delta added in-place)
 * @param src0 First input (projection output tensor, same as dst)
 * @param src1 Second input (pre-projection hidden states, used for Gather)
 * @param ith Thread index (only thread 0 executes to avoid races)
 * @param nth Total threads
 * @param userdata Unused (uses GetCurrentBatch() for batch context)
 */
void cb_apply_multi_lora(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                         int ith, int nth, void* userdata) {
    (void)nth;
    (void)userdata;

    // Single-threaded execution: LoRA application is already parallelized internally
    if (ith != 0) return;

    if (!dst || !src0 || !dst->data || !src0->data) return;

    // Get current batch from explicit work context
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    // NOTE: ggml_map_custom2() allocates dst with ggml_dup_tensor(a), which only
    // duplicates metadata (shape/type) and does not copy payload. We must copy
    // base projection output (src0) into dst first, then apply LoRA deltas.
    if (dst->data != src0->data) {
        if (src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->ne[0] == dst->ne[0] &&
            src0->ne[1] == dst->ne[1]) {
            const size_t row_bytes = static_cast<size_t>(src0->ne[0]) * sizeof(float);
            for (int64_t row = 0; row < src0->ne[1]; ++row) {
                const auto* src_row = reinterpret_cast<const uint8_t*>(src0->data) + row * src0->nb[1];
                auto* dst_row = reinterpret_cast<uint8_t*>(dst->data) + row * dst->nb[1];
                memcpy(dst_row, src_row, row_bytes);
            }
        } else {
            memcpy(dst->data, src0->data, std::min(ggml_nbytes(dst), ggml_nbytes(src0)));
        }
    }

    // Early exit if no LoRA adapters in this batch
    if (batch->lora_map.empty()) return;

    // Validate tensor data - src1 is the pre-projection hidden states
    if (!src1 || !src1->data) return;

    // Extract layer name from tensor name (e.g., "blk.0.attn_q" -> "blk.0.attn_q")
    // The tensor name is set by apply_lora lambda in BuildTransformerGraph
    const char* layer_name = dst->name;
    if (!layer_name || layer_name[0] == '\0') return;

    // Convert GGML tensors to DenseCore Tensor format
    // src1 = pre-projection hidden states (input to LoRA)
    densecore::Tensor t_input;
    t_input.data = const_cast<void*>(src1->data);
    t_input.dtype = densecore::GgmlTypeToDType(src1->type);
    t_input.ndim = 2;
    t_input.shape[0] = src1->ne[1];  // tokens (GGML: ne[1] = rows after mul_mat)
    t_input.shape[1] = src1->ne[0];  // hidden_dim
    t_input.stride[0] = src1->nb[1];
    t_input.stride[1] = src1->nb[0];

    // dst = projection output (LoRA delta added in-place)
    densecore::Tensor t_output;
    t_output.data = dst->data;
    t_output.dtype = densecore::GgmlTypeToDType(dst->type);
    t_output.ndim = 2;
    t_output.shape[0] = dst->ne[1];  // tokens
    t_output.shape[1] = dst->ne[0];  // output_dim
    t_output.stride[0] = dst->nb[1];
    t_output.stride[1] = dst->nb[0];

    // Dispatch to CpuBackend (NUMA-aware, parallelized internally)
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ApplyMultiLoRA(t_input, std::string(layer_name), batch->lora_map, &t_output);
}

// ============================================================================
// Fused Add + RMSNorm Callback (AVX-512 Optimized)
// ============================================================================
// Combines residual connection (x += residual) and RMSNorm in a single pass
// to reduce memory bandwidth by loading/storing data once instead of twice.
// ============================================================================

/**
 * User data for fused Add+RMSNorm operation
 */
struct AddRMSNormUserData {
    const float* residual;    ///< Residual tensor data [n_embd, N]
    const float* rms_weight;  ///< RMSNorm weight [n_embd]
    int n_embd;               ///< Embedding dimension
    int n_tokens;             ///< Number of tokens
    float eps;                ///< RMSNorm epsilon
};

// Thread-local pool for AddRMSNorm user data
static constexpr int kMaxAddRMSNormSlots = 256;

AddRMSNormUserData* GetAddRMSNormUserData();

// =============================================================================
// Parallel GEMV User Data + Buffers
// =============================================================================
static constexpr int kMaxGemvUserDataSlots = 2048;
static constexpr size_t kMaxQuantInputBufferSize = 65536;  // 64KB for large N
static constexpr int kMaxSmallBatchColsHard = 8;
static constexpr size_t kMaxDequantBufferSize = 16384;
static constexpr int kMaxPagedAttentionUserDataSlots = 256;

/**
 * User data for parallel GEMV operation
 */
struct GemvUserData {
    struct ggml_tensor* weight_tensor;  // Weight tensor (data accessed at runtime)
    int N;                              // Input dimension
    int K;                              // Output dimension
    ggml_type weight_type;              // Tensor type (F32, Q4_K, Q8_0, etc.)
    ggml_type input_quant_type;         // Quantization type for input (Q8_K, Q8_0, or F32)
    int slot_id = -1;
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
};

struct GemvBatchedUserData {
    struct ggml_tensor* weight_tensor = nullptr;  // Weight tensor (data accessed at runtime)
    int N = 0;                                    // Input dimension
    int K = 0;                                    // Output dimension
    int M = 0;                                    // Number of input columns (tokens)
    ggml_type weight_type = GGML_TYPE_F32;
    int slot_id = -1;
    ggml_type input_quant_type = GGML_TYPE_F32;
    size_t quant_row_stride = 0;  // Pre-computed aligned row stride for quantized input
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
};

struct PagedAttentionUserData {
    PagedKVCache* cache = nullptr;
    int layer = 0;
    int head_dim = 0;
    int v_head_dim = 0;
    int n_head = 0;
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    std::atomic<uint64_t> epoch_started{0};
    std::atomic<uint64_t> epoch_done{0};
    std::atomic<int> kv_writers_done{0};
};

/**
 * Custom callback for fused Add + RMSNorm
 *
 * Input tensor (src): Current tensor to add residual to and normalize
 * Output tensor (dst): Result of (src + residual) normalized with RMSNorm
 *
 * Uses AVX-512 fused kernel for optimal memory bandwidth utilization.
 */
void cb_residual_rmsnorm_fused(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                               void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->rms_weight) return;

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    const float eps = ud->eps;
    const bool has_residual = ud->residual != nullptr;

    // Partition work across tokens
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens) return;

    // Process assigned tokens
    for (int t = t_start; t < t_end; t++) {
        const float* x_ptr = (const float*)src->data + t * n_embd;
        float* out_ptr = (float*)dst->data + t * n_embd;

        if (has_residual) {
            const float* res_ptr = ud->residual + t * n_embd;
            // Use unified AddRMSNorm dispatcher (Runtime AVX512/AVX2/Scalar)
            densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);
        } else {
            densecore::simd::RMSNorm(x_ptr, ud->rms_weight, out_ptr, static_cast<size_t>(n_embd), eps);
        }
    }
}

// ============================================================================
// Fused SiLU×Mul Callback (SwiGLU FFN Optimization)
// ============================================================================
// Computes: out = silu(gate) * up in a single pass
// Saves memory by avoiding intermediate silu(gate) tensor allocation.
// ============================================================================

/**
 * Custom callback for fused SiLU×Mul (SwiGLU FFN)
 *
 * Signature compatible with ggml_map_custom2:
 *   void (*)(struct ggml_tensor *dst, const struct ggml_tensor *a,
 *            const struct ggml_tensor *b, int ith, int nth, void *userdata)
 */
void cb_silu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;  // Not needed since we use a/b directly

    // a = gate tensor (w1 output, SiLU input)
    // b = up tensor (w3 output)
    // dst = output tensor

    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;

    const float* gate = reinterpret_cast<const float*>(a->data);
    const float* up = reinterpret_cast<const float*>(b->data);
    float* out = reinterpret_cast<float*>(dst->data);

    // Total elements (flattened)
    const size_t size = ggml_nelements(a);

    // Use parallel SiLU×Mul from simd_ops.h
    densecore::simd::SiLUMulParallel(out, gate, up, size, ith, nth);
}

// ============================================================================
// Fused QKV Projection Callback (Tensor-Level Parallelism)
// ============================================================================

// Computes Q, K, V projections in a single pass with intra-operator parallelism
// across the output dimension (dim_q + dim_k + dim_v). This enables
// multi-thread utilization during decode when batch_size=1.
// ============================================================================

/**
 * User data for fused Q/K/V projection operation
 */
struct QKVUserData {
    const float* w_q;  ///< Q weight [dim_q, n_embd] (row-major)
    const float* w_k;  ///< K weight [dim_k, n_embd] (row-major)
    const float* w_v;  ///< V weight [dim_v, n_embd] (row-major)
    int n_embd;        ///< Input embedding dimension
    int dim_q;         ///< Q output dimension (n_head * head_dim)
    int dim_k;         ///< K output dimension (n_head_kv * head_dim)
    int dim_v;         ///< V output dimension (n_head_kv * head_dim)
};

// ============================================================================
// SSM (Mamba2) Callback UserData Structs
// ============================================================================
struct SSMConv1DUserData {
    float* conv_state;
    const float* weight;
    int channels;
    int kernel_size;
};

struct SSMQwen35DeltaUserData {
    const float* alpha_weight;  // [n_heads, n_embd]
    const float* beta_weight;   // [n_heads, n_embd]
    const float* dt_bias;       // [n_heads]
    const float* ssm_a;         // [n_heads]
    const float* norm_weight;   // [head_dim_v]
    float* ssm_state;           // [n_heads * head_dim_k * head_dim_v]
    int n_embd;
    int d_inner;
    int n_heads;
    int head_dim_v;
    int head_dim_k;
    int n_groups;
    float norm_eps;
};

struct GLMDSAPackUserData {
    int n_heads;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int v_head_dim;
};

// ============================================================================
// EXPLICIT INFERENCE WORK CONTEXT (per-thread, no implicit TLS pools)
// ============================================================================
struct InferenceWorkContext {
    const BatchSpec* batch = nullptr;
    KVCacheUserData kv_pool[256];
    QKVUserData qkv_pool[256];
    KVUpdateGatherUserData kv_update_gather_pool[kMaxKVUpdateGatherSlots];
    int qkv_index = 0;
    AddRMSNormUserData add_rmsnorm_pool[kMaxAddRMSNormSlots];
    int add_rmsnorm_index = 0;
    alignas(64) std::array<uint8_t, kMaxQuantInputBufferSize> gemv_quant_input_shared{};
    std::atomic<uint64_t> gemv_quantized_stamp{0};
    alignas(
        64) std::array<uint8_t, kMaxQuantInputBufferSize * kMaxSmallBatchColsHard> gemv_batched_quant_input_shared{};
    std::atomic<uint64_t> gemv_batched_quantized_stamp{0};
    GemvUserData gemv_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_userdata_index = 0;
    GemvBatchedUserData gemv_batched_userdata_pool[kMaxGemvUserDataSlots];
    int gemv_batched_userdata_index = 0;
    PagedAttentionUserData paged_attention_userdata_pool[kMaxPagedAttentionUserDataSlots];
    int paged_attention_userdata_index = 0;
    SSMConv1DUserData ssm_conv1d_pool[128];
    int ssm_conv1d_index = 0;
    SSMQwen35DeltaUserData ssm_qwen35_delta_pool[128];
    int ssm_qwen35_delta_index = 0;
    std::vector<ggml_bf16_t> bf16_buffer;
};

InferenceWorkContext* CreateInferenceWorkContext() {
    return new InferenceWorkContext();
}

void DestroyInferenceWorkContext(InferenceWorkContext* ctx) {
    delete ctx;
}

void ResetInferenceWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) return;
    ctx->batch = nullptr;
    ctx->qkv_index = 0;
    ctx->add_rmsnorm_index = 0;
    ctx->gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->gemv_userdata_index = 0;
    ctx->gemv_batched_userdata_index = 0;
    ctx->paged_attention_userdata_index = 0;
    ctx->ssm_conv1d_index = 0;
    ctx->ssm_qwen35_delta_index = 0;
    g_shared_batch.store(nullptr, std::memory_order_release);
}

void SetCurrentWorkContext(InferenceWorkContext* ctx) {
    tls_work_ctx = ctx;
}

InferenceWorkContext* GetCurrentWorkContext() {
    return tls_work_ctx;
}

void SetCurrentBatch(const BatchSpec* batch) {
    if (!batch) {
        throw densecore::InvalidArgumentException("SetCurrentBatch called with null batch");
    }
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("SetCurrentBatch called without active InferenceWorkContext");
    }
    ctx->batch = batch;
    g_shared_batch.store(batch, std::memory_order_release);
}

static const BatchSpec* GetCurrentBatch() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (ctx && ctx->batch) {
        return ctx->batch;
    }
    return g_shared_batch.load(std::memory_order_acquire);
}

// NOTE: KVCacheUserData is defined in densecore/inference_types_internal.h
// Pool of KVCacheUserData to avoid allocation per layer
// Max layers supported: 128 (enough for any current model)
// Each layer needs 2 entries (K and V), so 256 total slots
static constexpr int kMaxKVCacheUserDataSlots = 256;

// Helper to get a userdata slot (no allocation, no leak)
// NOTE: Not inline - needs external linkage for graph_builders/
KVCacheUserData* GetKVCacheUserData(int layer, bool is_k) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetKVCacheUserData called without active InferenceWorkContext");
    }
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVCacheUserDataSlots) {
        idx = idx % kMaxKVCacheUserDataSlots;  // Wrap for safety
    }
    return &ctx->kv_pool[idx];
}

KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetKVUpdateGatherUserData called without active InferenceWorkContext");
    }
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVUpdateGatherSlots) {
        idx = idx % kMaxKVUpdateGatherSlots;
    }
    return &ctx->kv_update_gather_pool[idx];
}

AddRMSNormUserData* GetAddRMSNormUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetAddRMSNormUserData called without active InferenceWorkContext");
    }
    int idx = ctx->add_rmsnorm_index++;
    if (idx >= kMaxAddRMSNormSlots) {
        // Reset index for next iteration but throw for current overflow
        ctx->add_rmsnorm_index = 0;
        throw densecore::OutOfMemoryException(
            "AddRMSNormUserData pool exhausted (max=" + std::to_string(kMaxAddRMSNormSlots) +
            "). Consider increasing kMaxAddRMSNormSlots for deep models.");
    }
    return &ctx->add_rmsnorm_pool[idx];
}

// Thread-local pool for QKV userdata
static constexpr int kMaxQKVUserDataSlots = 256;

inline QKVUserData* GetQKVUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetQKVUserData called without active InferenceWorkContext");
    }
    int idx = ctx->qkv_index++;
    if (idx >= kMaxQKVUserDataSlots) {
        // Reset index for next iteration but throw for current overflow
        ctx->qkv_index = 0;
        throw densecore::OutOfMemoryException(
            "QKVUserData pool exhausted (max=" + std::to_string(kMaxQKVUserDataSlots) +
            "). Consider increasing kMaxQKVUserDataSlots for deep models.");
    }
    return &ctx->qkv_pool[idx];
}

inline PagedAttentionUserData* GetPagedAttentionUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetPagedAttentionUserData called without active InferenceWorkContext");
    }
    int idx = ctx->paged_attention_userdata_index++;
    if (idx >= kMaxPagedAttentionUserDataSlots) {
        ctx->paged_attention_userdata_index = 0;
        idx = 0;
    }
    return &ctx->paged_attention_userdata_pool[idx];
}

inline SSMConv1DUserData* GetSSMConv1DUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetSSMConv1DUserData called without active InferenceWorkContext");
    }
    int idx = ctx->ssm_conv1d_index++;
    if (idx >= 128) {
        ctx->ssm_conv1d_index = 0;
        throw densecore::OutOfMemoryException("SSMConv1DUserData pool exhausted");
    }
    return &ctx->ssm_conv1d_pool[idx];
}

inline SSMQwen35DeltaUserData* GetSSMQwen35DeltaUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetSSMQwen35DeltaUserData called without active InferenceWorkContext");
    }
    int idx = ctx->ssm_qwen35_delta_index++;
    if (idx >= 128) {
        ctx->ssm_qwen35_delta_index = 0;
        throw densecore::OutOfMemoryException("SSMQwen35DeltaUserData pool exhausted");
    }
    return &ctx->ssm_qwen35_delta_pool[idx];
}

/**
 * Custom callback for fused Q/K/V projection with HYBRID parallelism
 *
 * Implements a smart dispatch strategy based on batch size:
 *
 * CASE A - SINGLE-TOKEN DECODE (n_tokens == 1):
 *   - All threads iterate through ALL tokens
 *   - Pass real ith/nth to ComputeQKV for TENSOR PARALLELISM
 *   - Multiple threads collaborate on each token's output dimensions
 *   - Only used for M=1 where column-splitting is the sole parallelism option
 *
 * CASE B - BATCHED DECODE / PREFILL (n_tokens >= 2):
 *   - Partition tokens across threads (TOKEN PARALLELISM)
 *   - Each thread computes FULL dimensions for its token subset
 *   - Pass ith=0, nth=1 to ComputeQKV to disable dimension splitting
 *   - More cache-friendly: each thread keeps weights hot in L1/L2
 *   - Avoids synchronization overhead of tensor parallelism for small batches
 */
void cb_compute_qkv(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    // ===========================================================================
    // BARRIER SAFETY CONTRACT:
    // ===========================================================================
    // This callback is invoked by GGML's thread pool. ALL threads must reach the
    // end of this function cleanly, even if they have no work to do.
    //
    // - ComputeQKV handles `start >= end` by doing nothing and returning early
    // - Early returns here are safe ONLY in PREFILL case (token partitioning)
    // - In DECODE case, all threads iterate all tokens (no early return)
    // ===========================================================================
    auto* ud = static_cast<QKVUserData*>(userdata);
    if (!ud || !dst || !src || !dst->data || !src->data || !ud->w_q || !ud->w_k || !ud->w_v) return;
    if (dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) return;

    const int n_embd = static_cast<int>(src->ne[0]);
    const int n_tokens = static_cast<int>(src->ne[1]);
    const int dim_q = ud->dim_q;
    const int dim_k = ud->dim_k;
    const int dim_v = ud->dim_v;
    const int total_dim = dim_q + dim_k + dim_v;

    if (n_embd != ud->n_embd || n_tokens <= 0 || total_dim <= 0) {
        return;
    }
    if (static_cast<int>(dst->ne[0]) != total_dim || static_cast<int>(dst->ne[1]) != n_tokens) {
        return;
    }

    const float* x = reinterpret_cast<const float*>(src->data);
    float* qkv_out = reinterpret_cast<float*>(dst->data);

    // ==========================================================================
    // HYBRID DISPATCH: Choose parallelism strategy based on batch size
    // ==========================================================================
    // Token parallelism preferred when n_tokens >= 2: each thread computes FULL
    // output dimensions for its subset of tokens. This avoids the synchronization
    // overhead of column-splitting and keeps weight data hot in L1/L2 per-thread.
    // Tensor parallelism reserved for single-token decode (M=1) where we MUST
    // split output columns to utilize multiple threads.
    // ==========================================================================

    if (n_tokens < 2) {
        // ========================================================================
        // CASE A: SINGLE-TOKEN DECODE (Tensor Parallelism)
        // ========================================================================
        // Only 1 token: all threads collaborate, each computing a SLICE of dims
        // ========================================================================
        for (int t = 0; t < n_tokens; t++) {
            const float* x_t = x + static_cast<size_t>(t) * n_embd;
            float* token_out = qkv_out + static_cast<size_t>(t) * total_dim;
            float* q_t = token_out;
            float* k_t = token_out + dim_q;
            float* v_t = token_out + dim_q + dim_k;

            // Each thread computes slice [start_col, end_col) of output dimensions
            // ComputeQKV internally partitions: total_cols = dim_q + dim_k + dim_v
            densecore::simd::ComputeQKV(q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, n_embd, dim_q, dim_k, dim_v, ith,
                                        nth  // Enable tensor parallelism
            );
        }
    } else {
        // ========================================================================
        // CASE B: PREFILL (Token Parallelism)
        // ========================================================================
        // Many tokens (prompt processing), partition tokens across threads
        // Each thread computes FULL dimensions for its subset of tokens
        // More cache-friendly: each thread touches contiguous weight rows
        // ========================================================================
        const int tokens_per_thread = (n_tokens + nth - 1) / nth;  // Ceiling div
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

        // Early exit if this thread has no tokens to process
        if (t_start >= n_tokens) return;

        for (int t = t_start; t < t_end; t++) {
            const float* x_t = x + static_cast<size_t>(t) * n_embd;
            float* token_out = qkv_out + static_cast<size_t>(t) * total_dim;
            float* q_t = token_out;
            float* k_t = token_out + dim_q;
            float* v_t = token_out + dim_q + dim_k;

            // Compute FULL dimensions for this token (no dimension splitting)
            // Pass ith=0, nth=1 to disable tensor parallelism within ComputeQKV
            densecore::simd::ComputeQKV(q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, n_embd, dim_q, dim_k, dim_v, 0,
                                        1  // Disable tensor parallelism (single-threaded kernel call)
            );
        }
    }
}

void cb_compute_qkv_map2(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                         int nth, void* userdata) {
    (void)a;  // Placeholder output tensor used only for dst shape allocation.
    cb_compute_qkv(dst, b, ith, nth, userdata);
}

// ============================================================================
// RoPE Table Initialization
// ============================================================================

/**
 * @brief Initialize pre-computed RoPE cos/sin table for the model
 *
 * Populates model->rope_cos_sin with values for all positions and dimensions.
 * Layout: [pos * head_dim + d] = cos/sin pair for position 'pos', dimension 'd'
 * Interleaved format: [cos0, sin0, cos1, sin1, ...]
 *
 * @param model Model to initialize RoPE table for
 */
void InitRoPETable(TransformerModel* model) {
    if (!model) return;

    const int n_ctx = model->hparams.n_ctx;
    int head_dim = model->hparams.n_embd / model->hparams.n_head;
    if (model->hparams.n_embd_head_k > 0) {
        head_dim = model->hparams.n_embd_head_k;
    }
    const float freq_base = model->hparams.rope_freq_base;

    // Reuse RoPETable from simd_ops.h to avoid code duplication
    densecore::simd::RoPETable table;
    table.Init(n_ctx, head_dim, freq_base);

    // Move the computed data to the model
    model->rope_cos_sin = std::move(table.cos_sin);
    model->rope_head_dim = head_dim;
}

struct RopeCustomOpData {
    const float* cos_sin = nullptr;
    int max_seq_len = 0;
    int head_dim = 0;
    int rope_dim = 0;
};

struct RopeCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    RopeCustomOpData data;
};

void cb_rope_precomputed_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0 || !dst || !dst->src[0] || !dst->src[1]) return;

    const auto* params = reinterpret_cast<const RopeCustomParams*>(dst->op_params);
    if (!params) return;
    const RopeCustomOpData& ud = params->data;
    if (!ud.cos_sin || ud.max_seq_len <= 0 || ud.head_dim <= 0 || ud.rope_dim <= 0) return;

    const struct ggml_tensor* src = dst->src[0];
    const struct ggml_tensor* pos = dst->src[1];
    if (!src->data || !dst->data || !pos->data || pos->type != GGML_TYPE_I32 || src->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return;
    }

    const int head_dim = static_cast<int>(src->ne[0]);
    const int n_heads = static_cast<int>(src->ne[1]);
    const int n_tokens = static_cast<int>(src->ne[2]);
    if (head_dim != ud.head_dim || n_heads <= 0 || n_tokens <= 0) return;

    const int rope_dim = std::min(ud.rope_dim, head_dim);
    const int total_rows = n_heads * n_tokens;
    const int rows_per_task = (total_rows + nth - 1) / nth;
    const int row_start = ith * rows_per_task;
    const int row_end = std::min(total_rows, row_start + rows_per_task);
    if (row_start >= row_end) return;

    const int* pos_data = reinterpret_cast<const int*>(pos->data);
    const bool dense_row = (src->nb[0] == sizeof(float)) && (dst->nb[0] == sizeof(float));

    static thread_local std::vector<float> row_in;
    static thread_local std::vector<float> row_out;
    if (!dense_row) {
        row_in.resize(static_cast<size_t>(head_dim));
        row_out.resize(static_cast<size_t>(head_dim));
    }

    for (int row = row_start; row < row_end; ++row) {
        const int token_idx = row / n_heads;
        const int head_idx = row % n_heads;
        const int pos_value = pos_data[token_idx];

        const char* src_ptr = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2] +
                              static_cast<size_t>(head_idx) * src->nb[1];
        char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2] +
                        static_cast<size_t>(head_idx) * dst->nb[1];

        if (dense_row) {
            densecore::simd::ApplyRoPE(reinterpret_cast<float*>(dst_ptr), reinterpret_cast<const float*>(src_ptr),
                                       ud.cos_sin, &pos_value, 1, head_dim, rope_dim, ud.max_seq_len);
            continue;
        }

        for (int d = 0; d < head_dim; ++d) {
            row_in[d] = *reinterpret_cast<const float*>(src_ptr + static_cast<size_t>(d) * src->nb[0]);
        }
        densecore::simd::ApplyRoPE(row_out.data(), row_in.data(), ud.cos_sin, &pos_value, 1, head_dim, rope_dim,
                                   ud.max_seq_len);
        for (int d = 0; d < head_dim; ++d) {
            *reinterpret_cast<float*>(dst_ptr + static_cast<size_t>(d) * dst->nb[0]) = row_out[d];
        }
    }
}

inline struct ggml_tensor* ggml_rope_precomputed_table(struct ggml_context* ctx, struct ggml_tensor* input,
                                                       struct ggml_tensor* pos, const TransformerModel* model,
                                                       int rope_dim, const BatchSpec* batch) {
    if (!ctx || !input || !pos || !model) return nullptr;
    if (input->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) return nullptr;
    if (model->rope_cos_sin.empty() || model->rope_head_dim <= 0) return nullptr;

    const int head_dim = static_cast<int>(input->ne[0]);
    const int n_heads = static_cast<int>(input->ne[1]);
    const int n_tokens = static_cast<int>(input->ne[2]);
    if (head_dim <= 0 || n_heads <= 0 || n_tokens <= 0) return nullptr;
    if (model->rope_head_dim < head_dim) return nullptr;

    const int clamped_rope_dim = std::max(0, std::min(rope_dim, head_dim));
    if (clamped_rope_dim == 0) return nullptr;

    struct ggml_tensor* result = ggml_dup_tensor(ctx, input);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = pos;

    RopeCustomParams params = {};
    params.fun = cb_rope_precomputed_custom;
    params.n_tasks = ResolveTaskCount(batch, std::max(1, n_heads * n_tokens));
    params.userdata = nullptr;
    params.data.cos_sin = model->rope_cos_sin.data();
    params.data.max_seq_len = model->hparams.n_ctx;
    params.data.head_dim = head_dim;
    params.data.rope_dim = clamped_rope_dim;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

static inline bool IsTokenHeadDense(const struct ggml_tensor* t, int head_dim) {
    return t && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) &&
           t->nb[1] == static_cast<size_t>(head_dim) * sizeof(float);
}

static inline void GatherTokenHeadContiguous(const struct ggml_tensor* src, int token_idx, int head_dim, int n_head_kv,
                                             float* out) {
    const char* token_base = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2];
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * n_head_kv * sizeof(float);
    if (IsTokenHeadDense(src, head_dim)) {
        std::memcpy(out, token_base, head_block_bytes);
        return;
    }

    const size_t nb0 = src->nb[0];
    const size_t nb1 = src->nb[1];
    for (int h = 0; h < n_head_kv; ++h) {
        const char* head_base = token_base + static_cast<size_t>(h) * nb1;
        float* out_head = out + static_cast<size_t>(h) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out_head[d] = *reinterpret_cast<const float*>(head_base + static_cast<size_t>(d) * nb0);
        }
    }
}

static inline void ScatterTokenHeadContiguous(const float* in, struct ggml_tensor* dst, int token_idx, int head_dim,
                                              int n_head_kv) {
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * n_head_kv * sizeof(float);
    if (IsTokenHeadDense(dst, head_dim)) {
        std::memcpy(token_base, in, head_block_bytes);
        return;
    }

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    for (int h = 0; h < n_head_kv; ++h) {
        char* head_base = token_base + static_cast<size_t>(h) * nb1;
        const float* in_head = in + static_cast<size_t>(h) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            *reinterpret_cast<float*>(head_base + static_cast<size_t>(d) * nb0) = in_head[d];
        }
    }
}

// Custom callback to load K/V history from cache and append current K/V.
// This implementation is stride-safe for both contiguous and view tensors.
void cb_kv_manage(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !src || !dst || !src->data || !dst->data) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->n_head_kv;
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    const int n_total = static_cast<int>(src->ne[2]);
    const int n_past = n_total - N;
    if (N < 0 || n_total < 0 || n_past < 0) return;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    const size_t head_block_bytes = head_block_size * sizeof(float);

    int writes_ok = 0;
    int writes_skipped = 0;

    // 1) Write current tokens to cache (single-threaded cache update).
    if (ith == 0 && N > 0) {
        std::vector<float> packed(head_block_size);
        for (int i = 0; i < N; ++i) {
            if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) continue;
            const int seq_id = batch->seq_id[i];
            const int pos = batch->pos[i];
            if (seq_id < 0 || seq_id >= static_cast<int>(batch->block_tables.size())) continue;

            const auto& block_table = batch->block_tables[seq_id];
            const int logical_block = pos / BLOCK_SIZE;
            const int slot = pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                writes_skipped++;
                continue;
            }

            const int block_id = block_table[logical_block];
            GatherTokenHeadContiguous(src, i, head_dim, n_head_kv, packed.data());
            if (ud->is_k) {
                ud->cache->WriteKSlot(block_id, ud->layer, slot, packed.data());
            } else {
                ud->cache->WriteVSlot(block_id, ud->layer, slot, packed.data());
            }
            writes_ok++;
        }
    }

    // 2) Read history [0, n_past) from cache into dst (parallel over tokens).
    if (n_past > 0) {
        const int seq_id = batch->seq_id.empty() ? -1 : batch->seq_id[0];
        const bool has_valid_seq = (seq_id >= 0 && seq_id < static_cast<int>(batch->block_tables.size()));
        const auto* block_table = has_valid_seq ? &batch->block_tables[seq_id] : nullptr;
        const KVRetentionPolicy& retention_policy = GetKVRetentionPolicy();
        KVRetentionSpan retained_history;
        if (has_valid_seq && seq_id < static_cast<int>(batch->n_past.size())) {
            const int seq_n_past = std::max(0, batch->n_past[static_cast<size_t>(seq_id)]);
            retained_history = ComputeKVRetentionSpan(seq_n_past, retention_policy);
        }

        const int tokens_per_thread = (n_past + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_past);

        std::vector<float> packed(head_block_size, 0.0f);
        for (int t = t_start; t < t_end; ++t) {
            if (!has_valid_seq) {
                std::fill(packed.begin(), packed.end(), 0.0f);
                ScatterTokenHeadContiguous(packed.data(), dst, t, head_dim, n_head_kv);
                continue;
            }

            if (!block_table || t >= retained_history.history_kept) {
                std::fill(packed.begin(), packed.end(), 0.0f);
                ScatterTokenHeadContiguous(packed.data(), dst, t, head_dim, n_head_kv);
                continue;
            }

            const int token_pos = MapRetainedHistoryIndex(retained_history, t);
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table->size())) {
                std::fill(packed.begin(), packed.end(), 0.0f);
                ScatterTokenHeadContiguous(packed.data(), dst, t, head_dim, n_head_kv);
                continue;
            }

            const int block_id = (*block_table)[logical_block];
            if (ud->is_k) {
                ud->cache->ReadKSlot(block_id, ud->layer, slot, packed.data());
            } else {
                ud->cache->ReadVSlot(block_id, ud->layer, slot, packed.data());
            }
            ScatterTokenHeadContiguous(packed.data(), dst, t, head_dim, n_head_kv);
        }
    }

    // 3) Append current tokens to dst [n_past, n_total) (parallel over tokens).
    if (N > 0) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);

        std::vector<float> packed(head_block_size);
        for (int t = t_start; t < t_end; ++t) {
            GatherTokenHeadContiguous(src, t, head_dim, n_head_kv, packed.data());
            ScatterTokenHeadContiguous(packed.data(), dst, n_past + t, head_dim, n_head_kv);
        }
    }

    if (IsDebugInferenceStatsEnabled() && ith == 0 && ud->is_k && (ud->layer == 0 || ud->layer == 3)) {
        static int kv_layout_dbg = 0;
        if (kv_layout_dbg < 6) {
            fprintf(stderr,
                    "[KV_LAYOUT #%d] layer=%d N=%d n_past=%d src.nb=[%zu,%zu,%zu] dst.nb=[%zu,%zu,%zu] "
                    "dense(src,dst)=(%d,%d) writes_ok=%d writes_skipped=%d\n",
                    kv_layout_dbg, ud->layer, N, n_past, src->nb[0], src->nb[1], src->nb[2], dst->nb[0], dst->nb[1],
                    dst->nb[2], IsTokenHeadDense(src, head_dim) ? 1 : 0, IsTokenHeadDense(dst, head_dim) ? 1 : 0,
                    writes_ok, writes_skipped);
            kv_layout_dbg++;
        }
    }
}

// ============================================================================
// NEW: Robust KV Cache Update and Gather Callback
// ============================================================================
/**
 * @brief Unified KV Cache Update and Gather Callback
 *
 * This callback performs three operations atomically:
 *   Step A: Write current K/V tokens into PagedKVCache
 *   Step B: Read historical K/V from cache into destination tensor
 *   Step C: Append current K/V to destination tensor after history
 *
 * Key improvements over cb_kv_manage:
 *   - Does NOT rely on ggml_pad assumptions about data placement
 *   - Explicitly controls all memory operations
 *   - Uses src_data pointer from userdata (not src tensor)
 *   - Clear, sequential steps with bounds checking
 *
 * Threading: ith == 0 only to avoid race conditions on cache writes.
 *
 * Input: dst is pre-allocated [head_dim * n_head_kv, n_past + N]
 * Output: dst filled with [history (0..n_past) | current (n_past..n_total)]
 */
void cb_kv_update_and_gather(struct ggml_tensor* dst, const struct ggml_tensor* /* src - unused */, int ith, int nth,
                             void* userdata) {
    (void)nth;  // Unused - single thread execution

    // Only thread 0 performs the work to avoid race conditions
    if (ith != 0) return;

    auto* ud = static_cast<KVUpdateGatherUserData*>(userdata);
    if (!ud || !ud->cache || !ud->batch || !ud->src_tensor) return;

    // Get source data pointer from tensor at runtime (after GGML backend
    // allocates memory)
    const float* src_data = reinterpret_cast<const float*>(ud->src_tensor->data);
    if (!src_data) return;

    const int layer = ud->layer;
    const int head_dim = ud->head_dim;
    const int n_head_kv = ud->n_head_kv;
    const int N = ud->N;
    const int n_past = ud->n_past;
    const bool is_k = ud->is_k;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    const size_t head_block_bytes = head_block_size * sizeof(float);
    const int n_total = n_past + N;

    // ===========================================================================
    // BOUNDS CHECK: Verify destination tensor has sufficient size
    // ===========================================================================
    const size_t expected_bytes = head_block_size * n_total * sizeof(float);
    if (ggml_nbytes(dst) < expected_bytes) {
        // Tensor too small - this indicates a graph construction error
        // Log and return to avoid buffer overflow
        return;
    }

    // ===========================================================================
    // STEP A: Write current tokens to cache
    // ===========================================================================
    // For each new token in the batch, write its K/V to the PagedKVCache
    // ===========================================================================
    for (int i = 0; i < N; i++) {
        // Validate seq_id bounds
        if (i >= static_cast<int>(ud->batch->seq_id.size())) continue;

        int seq_id = ud->batch->seq_id[i];
        int pos = ud->batch->pos[i];

        // Validate block_table bounds
        if (seq_id < 0 || seq_id >= static_cast<int>(ud->batch->block_tables.size())) continue;

        const auto& block_table = ud->batch->block_tables[seq_id];
        int logical_block = pos / BLOCK_SIZE;
        int slot = pos % BLOCK_SIZE;

        // Validate logical block exists
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;

        int block_id = block_table[logical_block];
        const float* token_data = src_data + i * head_block_size;

        if (is_k) {
            ud->cache->WriteKSlot(block_id, layer, slot, token_data);
        } else {
            ud->cache->WriteVSlot(block_id, layer, slot, token_data);
        }
    }

    // ===========================================================================
    // STEP B: Gather history from cache into destination tensor [0, n_past)
    // ===========================================================================
    // Read all historical K/V values from the PagedKVCache into dst
    // ===========================================================================
    if (n_past > 0) {
        // Use seq_id from first token (all tokens in batch share same sequence for
        // decode)
        int seq_id = ud->batch->seq_id[0];

        if (seq_id >= 0 && seq_id < static_cast<int>(ud->batch->block_tables.size())) {
            const auto& block_table = ud->batch->block_tables[seq_id];

            for (int i = 0; i < n_past; i++) {
                int logical_block = i / BLOCK_SIZE;
                int slot = i % BLOCK_SIZE;

                float* dst_slot = reinterpret_cast<float*>(dst->data) + i * head_block_size;

                if (logical_block >= 0 && logical_block < static_cast<int>(block_table.size())) {
                    int block_id = block_table[logical_block];
                    if (is_k) {
                        ud->cache->ReadKSlot(block_id, layer, slot, dst_slot);
                    } else {
                        ud->cache->ReadVSlot(block_id, layer, slot, dst_slot);
                    }
                } else {
                    // Block doesn't exist - zero-fill this slot
                    memset(dst_slot, 0, head_block_bytes);
                }
            }
        } else {
            // Invalid sequence - zero-fill entire history section
            memset(dst->data, 0, n_past * head_block_bytes);
        }
    }

    // ===========================================================================
    // STEP C: Append current tokens to destination tensor [n_past, n_total)
    // ===========================================================================
    // Copy the current K/V data after the history section
    // ===========================================================================
    float* dst_current = reinterpret_cast<float*>(dst->data) + n_past * head_block_size;
    memcpy(dst_current, src_data, N * head_block_bytes);
}

// ============================================================================
// Parallel GEMV Callback for Decode-Phase (N=1)
// ============================================================================
// GGML's ggml_mul_mat parallelizes along batch dimension.
// During decode (batch_size=1), there's NO parallelism opportunity.
// This callback uses GemvParallel to parallelize along output dimension.
// ============================================================================

inline GemvUserData* GetGemvUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvUserData called without active InferenceWorkContext");
    }
    int idx = ctx->gemv_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->gemv_userdata_index = 0;
        idx = 0;
    }
    GemvUserData* ud = &ctx->gemv_userdata_pool[idx];
    ud->slot_id = idx;
    ud->quant_input_shared = ctx->gemv_quant_input_shared.data();
    ud->quantized_stamp = &ctx->gemv_quantized_stamp;
    return ud;
}

inline GemvBatchedUserData* GetGemvBatchedUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvBatchedUserData called without active InferenceWorkContext");
    }
    int idx = ctx->gemv_batched_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->gemv_batched_userdata_index = 0;
        idx = 0;
    }
    GemvBatchedUserData* ud = &ctx->gemv_batched_userdata_pool[idx];
    ud->slot_id = idx;
    ud->input_quant_type = GGML_TYPE_F32;
    ud->quant_input_shared = ctx->gemv_batched_quant_input_shared.data();
    ud->quantized_stamp = &ctx->gemv_batched_quantized_stamp;
    return ud;
}

/**
 * Custom callback for parallel GEMV (decode-phase)
 *
 * REFACTORED: Uses GGML_OP_CUSTOM signature to allow output tensor shape
 * to be independent of input tensor shape. This fixes memory corruption
 * when Qwen3 projections change dimensions (e.g., 1024 -> 2048).
 *
 * Signature: void (*)(struct ggml_tensor *dst, int ith, int nth, void
 * *userdata) Input tensor accessed via dst->src[0]
 */
void cb_gemv_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<GemvUserData*>(userdata);
    if (!ud || !ud->weight_tensor) return;

    // Extract input tensor from dst->src[0] (GGML_OP_CUSTOM convention)
    const struct ggml_tensor* src = dst->src[0];
    if (!src) return;

    // Extract weight tensor from dst->src[1] (reliable graph topology, not
    // userdata)
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!weight_tensor || !weight_tensor->data) {
        fprintf(stderr, "CRITICAL: GEMV weight tensor is null or has no data\n");
        return;
    }

    // Get data pointers at runtime (guaranteed valid after GGML allocates)
    const float* x_f32 = reinterpret_cast<const float*>(src->data);
    const void* weight_data = weight_tensor->data;
    float* output = reinterpret_cast<float*>(dst->data);

    if (!x_f32 || !weight_data || !output) return;

    // ==========================================================================
    // DIMENSION VALIDATION (using weight tensor from graph, not stale userdata)
    // ==========================================================================
    const int K = static_cast<int>(weight_tensor->ne[1]);  // Output dimension from weight
    const int N = static_cast<int>(weight_tensor->ne[0]);  // Input dimension from weight

    // Validate output tensor matches expected K
    if (dst->ne[0] != K) {
        fprintf(stderr,
                "CRITICAL: GEMV buffer mismatch! dst->ne[0](%ld) != weight->ne[1](%d). "
                "Output tensor was sized incorrectly.\n",
                (long)dst->ne[0], K);
        return;
    }

    const ggml_type weight_type = weight_tensor->type;

    // ==========================================================================
    // SHARED PRE-QUANTIZATION (token-position synchronized):
    // Thread 0 quantizes once per decode token position, other threads wait for
    // that position marker and reuse the same quantized buffer.
    // ==========================================================================
    const void* quant_input = nullptr;
    if (ud->input_quant_type != GGML_TYPE_F32) {
        const auto* input_type_traits = ggml_get_type_traits_cpu(ud->input_quant_type);
        if (input_type_traits && input_type_traits->from_float) {
            const size_t quant_input_size = ggml_row_size(ud->input_quant_type, N);
            if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                const BatchSpec* batch = GetCurrentBatch();
                const bool has_valid_stamp = ud->slot_id >= 0;
                const uint64_t expected_stamp =
                    ComputeGemvBatchedQuantStamp(batch, 1, ud->slot_id, src->data, weight_tensor->data);

                if (nth <= 1 || !has_valid_stamp) {
                    alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                    quant_input_tls.resize(quant_input_size);
                    input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                    quant_input = quant_input_tls.data();
                } else {
                    if (!ud->quant_input_shared || !ud->quantized_stamp) {
                        alignas(64) thread_local std::vector<uint8_t> quant_input_tls;
                        quant_input_tls.resize(quant_input_size);
                        input_type_traits->from_float(x_f32, quant_input_tls.data(), static_cast<int64_t>(N));
                        quant_input = quant_input_tls.data();
                    } else if (ith == 0) {
                        input_type_traits->from_float(x_f32, ud->quant_input_shared, static_cast<int64_t>(N));
                        ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                        quant_input = ud->quant_input_shared;
                    } else {
                        int spin_count = 0;
                        while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                            SpinPause(spin_count++);
                        }
                        quant_input = ud->quant_input_shared;
                    }
                }
            }
        }
    }

    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    if (k_start >= K) return;

    // ==========================================================================
    // CASE A: FP32 weights - use optimized simd::GemvParallel
    // ==========================================================================
    if (weight_type == GGML_TYPE_F32) {
        const float* weight = reinterpret_cast<const float*>(weight_data);
        densecore::simd::GemvParallel(output, x_f32, weight, N, K, ith, nth);
        return;
    }

    // ==========================================================================
    // CASE B: Quantized weights with pre-quantized input - use native vec_dot
    // ==========================================================================
    const size_t row_stride = weight_tensor->nb[1];  // Bytes per row
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);

    if (quant_input && type_traits_cpu && type_traits_cpu->vec_dot) {
        for (int k = k_start; k < k_end; k++) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
        }
        return;
    }

    // ==========================================================================
    // CASE C: Fallback - dequantize weights (no pre-quantized input available)
    // ==========================================================================
    thread_local std::vector<float> dequant_buffer_tls;
    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float || N > static_cast<int>(kMaxDequantBufferSize)) {
        for (int k = k_start; k < k_end; k++) output[k] = 0.0f;
        return;
    }
    dequant_buffer_tls.resize(static_cast<size_t>(N));
    float* dequant_buffer = dequant_buffer_tls.data();

    for (int k = k_start; k < k_end; k++) {
        const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
        type_traits->to_float(row_ptr, dequant_buffer, N);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x_f32[i] * dequant_buffer[i];
        }
        output[k] = sum;
    }
}

struct DensecoreBlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(DensecoreBlockQ8K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t),
              "DensecoreBlockQ8K layout mismatch");

// True-batched Q4_K x Q8_K row dot:
// - Reuses Q4_K decode/scales once per weight row.
// - Computes all M column dots in one pass.
static inline bool ComputeQ4KQ8KBatchedRowScalar(const void* weight_row, const uint8_t* quant_input_base,
                                                 size_t quant_row_stride, int M, int N, float* out_sums) {
    if (!weight_row || !quant_input_base || !out_sums) return false;
    if (M <= 0 || M > kMaxSmallBatchColsHard) return false;
    if (N <= 0 || (N % QK_K) != 0) return false;
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        return false;
    }

    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    float lane_acc[kMaxSmallBatchColsHard][8];
    float min_acc[kMaxSmallBatchColsHard];
    std::memset(lane_acc, 0, sizeof(lane_acc));
    std::memset(min_acc, 0, sizeof(min_acc));

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;

    int8_t unpacked_q4[QK_K];
    uint32_t utmp[4];
    int32_t dot_chunks[8];

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];
        const uint8_t* q4 = xb.qs;
        int8_t* uq4 = unpacked_q4;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] & 0xF);
            uq4 += 32;
            for (int l = 0; l < 32; ++l) uq4[l] = static_cast<int8_t>(q4[l] >> 4);
            uq4 += 32;
            q4 += 32;
        }

        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
        const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            int32_t sumi = 0;
            for (int j = 0; j < QK_K / 16; ++j) {
                sumi += static_cast<int32_t>(yb.bsums[j]) * static_cast<int32_t>(mins[j / 2]);
            }

            std::memset(dot_chunks, 0, sizeof(dot_chunks));
            const int8_t* q8 = yb.qs;
            const int8_t* uq = unpacked_q4;
            int is = 0;
            for (int j = 0; j < QK_K / 32; ++j) {
                const int32_t scale = static_cast<int32_t>(scales[is++]);
                for (int rep = 0; rep < 4; ++rep) {
                    for (int l = 0; l < 8; ++l) {
                        dot_chunks[l] += scale * (static_cast<int32_t>(q8[l]) * static_cast<int32_t>(uq[l]));
                    }
                    q8 += 8;
                    uq += 8;
                }
            }

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = x_dmin * yd;
            for (int l = 0; l < 8; ++l) {
                lane_acc[m][l] += d * static_cast<float>(dot_chunks[l]);
            }
            min_acc[m] -= dmin * static_cast<float>(sumi);
        }
    }

    for (int m = 0; m < M; ++m) {
        float sum = min_acc[m];
        for (int l = 0; l < 8; ++l) {
            sum += lane_acc[m][l];
        }
        out_sums[m] = sum;
    }

    return true;
}

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
static inline __m256i M256SetM128i(const __m128i hi, const __m128i lo) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline float HSumFloat8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i GetScaleShuffleK4(int i) {
    static const uint8_t k_shuffle[256] = {
        0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
        0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
        2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
        4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
        6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
        8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
        8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_shuffle) + i);
}

static inline bool ComputeQ4KQ8KBatchedRowAvx2(const void* weight_row, const uint8_t* quant_input_base,
                                               size_t quant_row_stride, int M, int N, float* out_sums) {
    const bool debug_q4k_path = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    static std::atomic<bool> logged_invalid_args{false};
    static std::atomic<bool> logged_invalid_m{false};
    static std::atomic<bool> logged_invalid_n{false};
    static std::atomic<bool> logged_stride{false};
    static std::atomic<bool> logged_qkk{false};

    if (!weight_row || !quant_input_base || !out_sums) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_args.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_args" << std::endl;
            }
        }
        return false;
    }
    if (M <= 0 || M > kMaxSmallBatchColsHard) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_m.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_m M=" << M << std::endl;
            }
        }
        return false;
    }
    if (N <= 0 || (N % QK_K) != 0) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_invalid_n.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable invalid_n N=" << N << std::endl;
            }
        }
        return false;
    }
    if (quant_row_stride < static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_stride.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable stride stride=" << quant_row_stride << " need="
                          << (static_cast<size_t>(sizeof(DensecoreBlockQ8K)) * static_cast<size_t>(N / QK_K))
                          << std::endl;
            }
        }
        return false;
    }
    if (QK_K != 256) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_qkk.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2_unavailable QK_K=" << QK_K << std::endl;
            }
        }
        return false;
    }
    const auto* x_blocks = reinterpret_cast<const block_q4_K*>(weight_row);
    const int nb = N / QK_K;
    alignas(64) std::array<float, kMaxSmallBatchColsHard> sums{};

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);

    for (int bi = 0; bi < nb; ++bi) {
        const auto& xb = x_blocks[bi];

        uint32_t utmp[4];
        std::memcpy(utmp, xb.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = M256SetM128i(sc128, sc128);

        const float x_d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.d));
        const float x_dmin = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(xb.dmin));

        // Reuse decoded q4 nibble vectors/scales for all M columns.
        __m256i q4l[QK_K / 64];
        __m256i q4h[QK_K / 64];
        __m256i scale_l[QK_K / 64];
        __m256i scale_h[QK_K / 64];
        const uint8_t* q4 = xb.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }

        for (int m = 0; m < M; ++m) {
            const auto* y_blocks = reinterpret_cast<const DensecoreBlockQ8K*>(
                quant_input_base + static_cast<size_t>(m) * quant_row_stride);
            const auto& yb = y_blocks[bi];

            const float yd = yb.d;
            const float d = x_d * yd;
            const float dmin = -x_dmin * yd;

            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            __m128i sum32 = _mm_hadd_epi32(prod, prod);
            sum32 = _mm_hadd_epi32(sum32, sum32);
            const int32_t min_dot = _mm_cvtsi128_si32(sum32);

            const int8_t* q8 = yb.qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);

                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            sums[static_cast<size_t>(m)] +=
                d * HSumFloat8(_mm256_cvtepi32_ps(sumi)) + dmin * static_cast<float>(min_dot);
        }
    }

    for (int m = 0; m < M; ++m) {
        out_sums[m] = sums[static_cast<size_t>(m)];
    }
    return true;
}
#endif

static inline bool ComputeQ4KQ8KBatchedRow(const void* weight_row, const uint8_t* quant_input_base,
                                           size_t quant_row_stride, int M, int N, float* out_sums) {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    static const bool debug_q4k_path = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    static std::atomic<bool> logged_avx2{false};
    static std::atomic<bool> logged_scalar{false};
    if (IsQ4KTrueBatchedAvx2Enabled() &&
        ComputeQ4KQ8KBatchedRowAvx2(weight_row, quant_input_base, quant_row_stride, M, N, out_sums)) {
        if (debug_q4k_path) {
            bool expected = false;
            if (logged_avx2.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                std::cerr << "[Q4K_BATCHED_PATH] avx2" << std::endl;
            }
        }
        return true;
    }
    if (debug_q4k_path) {
        bool expected = false;
        if (logged_scalar.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
            std::cerr << "[Q4K_BATCHED_PATH] scalar" << std::endl;
        }
    }
#endif
    return ComputeQ4KQ8KBatchedRowScalar(weight_row, quant_input_base, quant_row_stride, M, N, out_sums);
}

/**
 * Custom callback for small-batch GEMV/GEMM hybrid (2 <= M <= 8).
 *
 * Computes output[:, m] = weight @ input[:, m] for all m in [0, M) while
 * reusing each weight row across M tokens before evicting it from cache.
 */
void cb_gemv_batched_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<GemvBatchedUserData*>(userdata);
    if (!ud || !ud->weight_tensor) return;
    if (nth <= 0) return;
    if (!dst || !dst->src[0] || !dst->src[1]) return;

    const struct ggml_tensor* src = dst->src[0];
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!src->data || !weight_tensor->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;

    const int N = static_cast<int>(src->ne[0]);
    const int M = static_cast<int>(src->ne[1]);
    const int K = static_cast<int>(weight_tensor->ne[1]);
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (M != ud->M || K != ud->K || N != ud->N) return;
    if (static_cast<int>(dst->ne[0]) != K || static_cast<int>(dst->ne[1]) != M) return;

    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(K, k_start + k_per_thread);
    if (k_start >= k_end) return;

    const bool input_contig = src->nb[0] == sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float);
    const char* input_base = reinterpret_cast<const char*>(src->data);
    char* output_base = reinterpret_cast<char*>(dst->data);
    const size_t input_col_stride = static_cast<size_t>(src->nb[1]);
    const size_t output_col_stride = static_cast<size_t>(dst->nb[1]);
    const size_t weight_row_stride = static_cast<size_t>(weight_tensor->nb[1]);
    const ggml_type weight_type = weight_tensor->type;

    thread_local std::vector<const float*> x_rows;
    thread_local std::vector<float> gathered_inputs;
    thread_local std::vector<float> sums;
    x_rows.resize(static_cast<size_t>(M));
    sums.resize(static_cast<size_t>(M));

    if (input_contig) {
        for (int m = 0; m < M; ++m) {
            x_rows[static_cast<size_t>(m)] =
                reinterpret_cast<const float*>(input_base + static_cast<size_t>(m) * input_col_stride);
        }
    } else {
        gathered_inputs.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * input_col_stride;
            float* dst_col = gathered_inputs.data() + static_cast<size_t>(m) * static_cast<size_t>(N);
            for (int i = 0; i < N; ++i) {
                dst_col[static_cast<size_t>(i)] =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * src->nb[0]);
            }
            x_rows[static_cast<size_t>(m)] = dst_col;
        }
    }

    auto store_out = [&](int m, int k, float value) {
        char* out_col = output_base + static_cast<size_t>(m) * output_col_stride;
        if (output_contig) {
            reinterpret_cast<float*>(out_col)[k] = value;
        } else {
            *reinterpret_cast<float*>(out_col + static_cast<size_t>(k) * dst->nb[0]) = value;
        }
    };

    const char* weight_base = reinterpret_cast<const char*>(weight_tensor->data);

    if (weight_type == GGML_TYPE_F32) {
        // Fast path: contiguous layout -> Highway SIMD GEMM with Split-N parallelism.
        // GemmFP32_Hwy computes C[:, n_start:n_end) = A[M,K_gemm] x B[N_gemm,K_gemm]^T
        // Our mapping: A=input[M, N_input], B=weight[K_output, N_input], C=output[M, K_output]
        // GEMM K_gemm = N (input dim), GEMM N_gemm = K (output dim)
        // Split-N over K_output (the output dimension, which IS GEMM's N).
        const bool weight_contig = (weight_row_stride == static_cast<size_t>(N) * sizeof(float));
        if (input_contig && output_contig && weight_contig) {
            const float* A = reinterpret_cast<const float*>(input_base);
            const float* B = reinterpret_cast<const float*>(weight_base);
            float* C = reinterpret_cast<float*>(output_base);
            // k_start/k_end map to n_start/n_end in GEMM Split-N convention
            densecore::hwy_kernels::GemmFP32_Hwy(C, A, B, M, K, N, k_start, k_end);
            return;
        }

        // Strided fallback: scalar with weight row reuse
        for (int k = k_start; k < k_end; ++k) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            const float* w_row =
                reinterpret_cast<const float*>(weight_base + static_cast<size_t>(k) * weight_row_stride);
            for (int i = 0; i < N; ++i) {
                const float w = w_row[i];
                for (int m = 0; m < M; ++m) {
                    sums[static_cast<size_t>(m)] += x_rows[static_cast<size_t>(m)][i] * w;
                }
            }
            for (int m = 0; m < M; ++m) {
                store_out(m, k, sums[static_cast<size_t>(m)]);
            }
        }
        return;
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type =
            (ud->input_quant_type != GGML_TYPE_F32) ? ud->input_quant_type : type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        // Use pre-computed stride from graph-build if available, else compute now
        const size_t quant_row_stride = (ud->quant_row_stride > 0)
                                            ? ud->quant_row_stride
                                            : densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        const size_t quant_buffer_capacity =
            static_cast<size_t>(kMaxQuantInputBufferSize) * static_cast<size_t>(kMaxSmallBatchColsHard);
        const int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
        const bool vec_dot_supports_nrc_batch = vec_dot_nrows >= M;

        const bool can_quantize_inputs = input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
                                         M <= kMaxSmallBatchColsHard && quant_row_stride <= kMaxQuantInputBufferSize &&
                                         quant_row_stride * static_cast<size_t>(M) <= quant_buffer_capacity;

        const bool can_use_quant_nrc_fast = can_quantize_inputs && output_contig && vec_dot_supports_nrc_batch;
        const bool can_use_q4k_true_batched = can_quantize_inputs && weight_type == GGML_TYPE_Q4_K &&
                                              vec_dot_type == GGML_TYPE_Q8_K && IsQ4KTrueBatchedKernelEnabled() &&
                                              (N % QK_K == 0);

        const uint8_t* quant_input_base = nullptr;
        size_t quant_input_row_stride = quant_row_stride;
        if (can_quantize_inputs) {
            const BatchSpec* batch = GetCurrentBatch();
            const bool can_sync_on_stamp = nth > 1 && ud->slot_id >= 0 && ud->quant_input_shared && ud->quantized_stamp;
            const uint64_t expected_stamp =
                ComputeGemvBatchedQuantStamp(batch, M, ud->slot_id, src->data, weight_tensor->data);

            const size_t quant_total_size = quant_row_stride * static_cast<size_t>(M);
            thread_local std::vector<uint8_t> quant_inputs_tls;
            if (!can_sync_on_stamp) {
                quant_inputs_tls.resize(quant_total_size);
                for (int m = 0; m < M; ++m) {
                    uint8_t* q_ptr = quant_inputs_tls.data() + static_cast<size_t>(m) * quant_row_stride;
                    input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
                }
                quant_input_base = quant_inputs_tls.data();
            } else if (ith == 0) {
                for (int m = 0; m < M; ++m) {
                    uint8_t* q_ptr = ud->quant_input_shared + static_cast<size_t>(m) * quant_row_stride;
                    input_type_traits->from_float(x_rows[static_cast<size_t>(m)], q_ptr, static_cast<int64_t>(N));
                }
                ud->quantized_stamp->store(expected_stamp, std::memory_order_release);
                quant_input_base = ud->quant_input_shared;
            } else {
                int spin_count = 0;
                while (ud->quantized_stamp->load(std::memory_order_acquire) != expected_stamp) {
                    SpinPause(spin_count++);
                }
                quant_input_base = ud->quant_input_shared;
            }
        }

        // Fast path: one-time quantization + multi-column vec_dot (nrc=M).
        if (can_use_quant_nrc_fast && quant_input_base) {
            for (int k = k_start; k < k_end; ++k) {
                float* out_ptr = reinterpret_cast<float*>(output_base + static_cast<size_t>(k) * sizeof(float));
                const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                type_traits_cpu->vec_dot(N, out_ptr, dst->nb[1], row_ptr, 0, quant_input_base, quant_input_row_stride,
                                         M);
            }
            LogMatmulPathOnce("gemv_batched_quant_nrc");
            return;
        }

        // True-batched custom kernel for Q4_K x Q8_K on x86:
        // unpack/scale once per weight row, compute all M columns together.
        if (can_use_q4k_true_batched && quant_input_base) {
            alignas(64) std::array<float, kMaxSmallBatchColsHard> row_sums{};
            static const bool debug_q4k_kernel_check = []() {
                const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_CHECK");
                return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
            }();
            static const float kDebugKernelWarnDiff = []() {
                const char* env = std::getenv("DENSECORE_DEBUG_Q4K_BATCHED_KERNEL_TOL");
                if (!env || env[0] == '\0') return 1e-3f;
                char* end = nullptr;
                const float v = std::strtof(env, &end);
                if (end == env || !std::isfinite(v) || v < 0.0f) return 1e-3f;
                return v;
            }();
            static std::atomic<int> q4k_kernel_warn_count{0};
            float max_abs_diff = 0.0f;
            int max_diff_k = -1;
            int max_diff_m = -1;
            float max_diff_batched = 0.0f;
            float max_diff_ref = 0.0f;
            bool all_rows_ok = true;
            for (int k = k_start; k < k_end; ++k) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                if (!ComputeQ4KQ8KBatchedRow(row_ptr, quant_input_base, quant_input_row_stride, M, N,
                                             row_sums.data())) {
                    all_rows_ok = false;
                    break;
                }

                if (debug_q4k_kernel_check) {
                    for (int m = 0; m < M; ++m) {
                        float ref = 0.0f;
                        const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_input_row_stride;
                        type_traits_cpu->vec_dot(N, &ref, 0, row_ptr, 0, q_ptr, 0, 1);
                        const float diff = std::fabs(row_sums[static_cast<size_t>(m)] - ref);
                        if (diff > max_abs_diff) {
                            max_abs_diff = diff;
                            max_diff_k = k;
                            max_diff_m = m;
                            max_diff_batched = row_sums[static_cast<size_t>(m)];
                            max_diff_ref = ref;
                        }
                    }
                }

                for (int m = 0; m < M; ++m) {
                    store_out(m, k, row_sums[static_cast<size_t>(m)]);
                }
            }
            if (debug_q4k_kernel_check && max_abs_diff > kDebugKernelWarnDiff) {
                const int warn_idx = q4k_kernel_warn_count.fetch_add(1, std::memory_order_relaxed);
                if (warn_idx < 32) {
                    std::cerr << "[Q4K_BATCHED_CHECK] WARN ith=" << ith << " max_abs_diff=" << max_abs_diff
                              << " at(k,m)=" << max_diff_k << "," << max_diff_m << " batched=" << max_diff_batched
                              << " ref=" << max_diff_ref << std::endl;
                }
            }
            if (all_rows_ok) {
                return;
            }
        }

        // Robust fallback: reuse the quantized input buffer and run vec_dot with
        // nrc=1. This keeps parity with single-token quant GEMV math when the
        // backend has no true nrc=M support for this weight/input type pair.
        if (quant_input_base) {
            for (int k = k_start; k < k_end; ++k) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * weight_row_stride;
                for (int m = 0; m < M; ++m) {
                    float sum = 0.0f;
                    const void* q_ptr = quant_input_base + static_cast<size_t>(m) * quant_input_row_stride;
                    type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                    store_out(m, k, sum);
                }
            }
            return;
        }
    }

    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float) {
        for (int k = k_start; k < k_end; ++k) {
            for (int m = 0; m < M; ++m) {
                store_out(m, k, 0.0f);
            }
        }
        return;
    }

    thread_local std::vector<float> dequant_row;
    dequant_row.resize(static_cast<size_t>(N));
    for (int k = k_start; k < k_end; ++k) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const void* row_ptr =
            reinterpret_cast<const char*>(weight_tensor->data) + static_cast<size_t>(k) * weight_row_stride;
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int i = 0; i < N; ++i) {
            const float w = dequant_row[static_cast<size_t>(i)];
            for (int m = 0; m < M; ++m) {
                sums[static_cast<size_t>(m)] += x_rows[static_cast<size_t>(m)][i] * w;
            }
        }
        for (int m = 0; m < M; ++m) {
            store_out(m, k, sums[static_cast<size_t>(m)]);
        }
    }
}

static void ComputePagedAttentionScalarHeads(const PagedAttentionUserData* ud, const std::vector<int>& block_table,
                                             int context_len, const KVRetentionSpan& retained_span, int current_pos,
                                             float scale, const float* q_data, float* out_data, int h_start,
                                             int h_end) {
    if (!ud || !ud->cache || !q_data || !out_data) {
        return;
    }

    const int n_head_kv = ud->cache->n_head_kv;
    const int n_head_total = ud->n_head;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (head_dim <= 0 || v_head_dim <= 0 || n_head_kv <= 0 || n_head_total <= 0 || (n_head_total % n_head_kv) != 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    const int kv_group_size = n_head_total / n_head_kv;
    const auto k_layout = ud->cache->GetBlockLayout();
    const auto v_layout = ud->cache->GetVBlockLayout();
    if (k_layout.head_stride_bytes == 0 || k_layout.slot_stride_bytes == 0 || v_layout.head_stride_bytes == 0 ||
        v_layout.slot_stride_bytes == 0) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
        return;
    }

    thread_local std::vector<const uint8_t*> k_blocks;
    thread_local std::vector<const uint8_t*> v_blocks;
    k_blocks.resize(block_table.size(), nullptr);
    v_blocks.resize(block_table.size(), nullptr);
    for (size_t bi = 0; bi < block_table.size(); ++bi) {
        const int block_id = block_table[bi];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) {
            continue;
        }
        k_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetKBlockPtr(block_id, ud->layer));
        v_blocks[bi] = reinterpret_cast<const uint8_t*>(ud->cache->GetVBlockPtr(block_id, ud->layer));
    }

    thread_local std::vector<float> k_head_scratch;
    thread_local std::vector<float> v_head_scratch;
    thread_local std::vector<float> scores;
    k_head_scratch.resize(static_cast<size_t>(head_dim));
    v_head_scratch.resize(static_cast<size_t>(v_head_dim));
    scores.resize(static_cast<size_t>(context_len));

    const auto* quant_traits =
        ggml_is_quantized(k_layout.cache_type) ? ggml_get_type_traits(k_layout.cache_type) : nullptr;

    for (int h = h_start; h < h_end; ++h) {
        const float* q_head = q_data + static_cast<size_t>(h) * head_dim;
        float* out_head = out_data + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        int kv_head = h / kv_group_size;
        if (kv_head < 0) kv_head = 0;
        if (kv_head >= n_head_kv) kv_head = n_head_kv - 1;
        const size_t k_head_offset_bytes = static_cast<size_t>(kv_head) * k_layout.head_stride_bytes;
        const size_t v_head_offset_bytes = static_cast<size_t>(kv_head) * v_layout.head_stride_bytes;

        float max_score = -INFINITY;
        for (int t = 0; t < context_len; ++t) {
            const int token_pos =
                (t < retained_span.history_kept) ? MapRetainedHistoryIndex(retained_span, t) : current_pos;
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            const uint8_t* k_block_base = k_blocks[static_cast<size_t>(logical_block)];
            if (!k_block_base) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            const uint8_t* k_ptr =
                k_block_base + static_cast<size_t>(slot_idx) * k_layout.slot_stride_bytes + k_head_offset_bytes;
            const float* k_head = nullptr;
            if (k_layout.cache_type == GGML_TYPE_F32) {
                k_head = reinterpret_cast<const float*>(k_ptr);
            } else if (k_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(k_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(k_ptr),
                                                 head_dim);
                k_head = k_head_scratch.data();
            } else if (ggml_is_quantized(k_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(k_ptr, k_head_scratch.data(), head_dim);
                k_head = k_head_scratch.data();
            }
            if (!k_head) {
                scores[static_cast<size_t>(t)] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            const float score = dot * scale;
            scores[static_cast<size_t>(t)] = score;
            if (std::isfinite(score) && score > max_score) {
                max_score = score;
            }
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            const float score = scores[static_cast<size_t>(t)];
            if (!std::isfinite(score)) continue;

            const float weight = std::exp(score - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;

            const int token_pos =
                (t < retained_span.history_kept) ? MapRetainedHistoryIndex(retained_span, t) : current_pos;
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot_idx = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
            const uint8_t* v_block_base = v_blocks[static_cast<size_t>(logical_block)];
            if (!v_block_base) continue;

            const uint8_t* v_ptr =
                v_block_base + static_cast<size_t>(slot_idx) * v_layout.slot_stride_bytes + v_head_offset_bytes;
            const float* v_head = nullptr;
            if (v_layout.cache_type == GGML_TYPE_F32) {
                v_head = reinterpret_cast<const float*>(v_ptr);
            } else if (v_layout.cache_type == GGML_TYPE_F16) {
                densecore::simd::ConvertF16ToF32(v_head_scratch.data(), reinterpret_cast<const ggml_fp16_t*>(v_ptr),
                                                 v_head_dim);
                v_head = v_head_scratch.data();
            } else if (ggml_is_quantized(v_layout.cache_type) && quant_traits && quant_traits->to_float) {
                quant_traits->to_float(v_ptr, v_head_scratch.data(), v_head_dim);
                v_head = v_head_scratch.data();
            }
            if (!v_head) continue;

            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
            denom += weight;
        }

        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }

        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}

void cb_paged_attention_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data) return;
    if (nth <= 0) return;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    if (q_tokens <= 0 || ud->head_dim <= 0 || v_head_dim <= 0 || ud->n_head <= 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const int n_head_kv = ud->cache->n_head_kv;
    if (n_head_kv <= 0 || (ud->n_head % n_head_kv) != 0) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (q_tensor->type != GGML_TYPE_F32 || k_tensor->type != GGML_TYPE_F32 || v_tensor->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    if (static_cast<int>(q_tensor->ne[0]) != ud->head_dim || static_cast<int>(q_tensor->ne[1]) != ud->n_head ||
        static_cast<int>(k_tensor->ne[0]) != ud->head_dim || static_cast<int>(k_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(v_tensor->ne[0]) != v_head_dim || static_cast<int>(v_tensor->ne[1]) != n_head_kv ||
        static_cast<int>(k_tensor->ne[2]) != q_tokens || static_cast<int>(v_tensor->ne[2]) != q_tokens ||
        static_cast<int>(dst->ne[0]) != v_head_dim || static_cast<int>(dst->ne[1]) != ud->n_head ||
        static_cast<int>(dst->ne[2]) != q_tokens) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const bool layout_ok = q_tensor->nb[0] == sizeof(float) && k_tensor->nb[0] == sizeof(float) &&
                           v_tensor->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
                           q_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           k_tensor->nb[1] == static_cast<size_t>(ud->head_dim) * sizeof(float) &&
                           v_tensor->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float) &&
                           dst->nb[1] == static_cast<size_t>(v_head_dim) * sizeof(float);
    if (!layout_ok) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch || batch->num_seqs != q_tokens || static_cast<int>(batch->tokens.size()) != q_tokens ||
        static_cast<int>(batch->seq_id.size()) != q_tokens || static_cast<int>(batch->pos.size()) != q_tokens ||
        static_cast<int>(batch->block_tables.size()) != batch->num_seqs ||
        static_cast<int>(batch->n_past.size()) != batch->num_seqs) {
        if (ith == 0) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    // Token-parallel fast path: when nth == q_tokens > 1, each thread owns
    // exactly one token. Write KV for its token, then run all heads — no barrier.
    const bool token_parallel_mode = (nth == q_tokens && q_tokens > 1);

    // Phase 1: Write current decode K/V to paged cache (parallel over tokens).
    // Each thread handles a disjoint subset of decode tokens, then synchronizes
    // via an epoch barrier before any thread starts attention reads.
    const bool do_profile = (ith == 0) && IsDecodeProfileEnabled();
    std::chrono::steady_clock::time_point kv_begin, kv_end;
    if (do_profile) kv_begin = std::chrono::steady_clock::now();

    const char* k_base = reinterpret_cast<const char*>(k_tensor->data);
    const char* v_base = reinterpret_cast<const char*>(v_tensor->data);
    const size_t k_token_stride = static_cast<size_t>(k_tensor->nb[2]);
    const size_t v_token_stride = static_cast<size_t>(v_tensor->nb[2]);

    if (token_parallel_mode) {
        // Each thread writes KV for exactly token_idx == ith. No barrier needed
        // because each thread only reads from its own token's KV cache slot.
        const int token_idx = ith;
        const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
        if (seq_idx >= 0 && seq_idx < batch->num_seqs) {
            const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
            if (!block_table.empty()) {
                const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
                if (pos_i >= 0) {
                    const int logical_block = pos_i / BLOCK_SIZE;
                    const int slot = pos_i % BLOCK_SIZE;
                    if (logical_block >= 0 && logical_block < static_cast<int>(block_table.size())) {
                        const int block_id = block_table[static_cast<size_t>(logical_block)];
                        if (block_id >= 0 && block_id < ud->cache->max_blocks) {
                            const float* k_src = reinterpret_cast<const float*>(
                                k_base + static_cast<size_t>(token_idx) * k_token_stride);
                            const float* v_src = reinterpret_cast<const float*>(
                                v_base + static_cast<size_t>(token_idx) * v_token_stride);
                            ud->cache->WriteKSlot(block_id, ud->layer, slot, k_src);
                            ud->cache->WriteVSlot(block_id, ud->layer, slot, v_src);
                        }
                    }
                }
            }
        }
    } else {
        // Standard barrier-based path for n_tokens==1 or head-tiled parallelism
        uint64_t epoch = 0;
        if (ith == 0) {
            ud->kv_writers_done.store(0, std::memory_order_release);
            epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
        } else {
            int spin_count = 0;
            while (true) {
                const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
                const uint64_t done = ud->epoch_done.load(std::memory_order_acquire);
                if (started > done) {
                    epoch = started;
                    break;
                }
                SpinPause(spin_count++);
            }
        }

        for (int i = ith; i < q_tokens; i += nth) {
            const int seq_idx = batch->seq_id[static_cast<size_t>(i)];
            if (seq_idx < 0 || seq_idx >= batch->num_seqs) {
                continue;
            }
            const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
            if (block_table.empty()) {
                continue;
            }

            const int pos_i = batch->pos[static_cast<size_t>(i)];
            if (pos_i < 0) {
                continue;
            }
            const int logical_block = pos_i / BLOCK_SIZE;
            const int slot = pos_i % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                continue;
            }

            const int block_id = block_table[static_cast<size_t>(logical_block)];
            if (block_id < 0 || block_id >= ud->cache->max_blocks) {
                continue;
            }

            const float* k_src = reinterpret_cast<const float*>(k_base + static_cast<size_t>(i) * k_token_stride);
            const float* v_src = reinterpret_cast<const float*>(v_base + static_cast<size_t>(i) * v_token_stride);
            ud->cache->WriteKSlot(block_id, ud->layer, slot, k_src);
            ud->cache->WriteVSlot(block_id, ud->layer, slot, v_src);
        }

        const int writers_done = ud->kv_writers_done.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (writers_done == nth) {
            ud->epoch_done.store(epoch, std::memory_order_release);
        } else {
            int spin_count = 0;
            while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
                SpinPause(spin_count++);
            }
        }
    }
    if (do_profile && ith == 0) kv_end = std::chrono::steady_clock::now();

    // Phase 2: Attention computation
    const int head_tile = std::max(1, ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", 8));
    const int tiles_per_token = std::max(1, (ud->n_head + head_tile - 1) / head_tile);
    int tile_start, tile_end;
    if (token_parallel_mode) {
        // Each thread handles all heads for its token (token_idx == ith)
        tile_start = ith * tiles_per_token;
        tile_end = tile_start + tiles_per_token;
    } else {
        const int total_tiles = q_tokens * tiles_per_token;
        if (total_tiles <= 0) {
            return;
        }
        tile_start = (total_tiles * ith) / nth;
        tile_end = (total_tiles * (ith + 1)) / nth;
    }
    if (tile_start >= tile_end) {
        return;
    }

    const bool use_hwy = IsPagedAttentionHwyEnabled();
    const float scale = 1.0f / std::sqrt(static_cast<float>(ud->head_dim));
    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);
    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);

    // Precompute cache_type_id once to skip per-tile Tensor wrapper + validation
    int32_t cache_type_id = -1;
    if (use_hwy) {
        if (ud->cache->cache_type == GGML_TYPE_F32)
            cache_type_id = 0;
        else if (ud->cache->cache_type == GGML_TYPE_F16)
            cache_type_id = 1;
        else if (ud->cache->cache_type == GGML_TYPE_Q4_0)
            cache_type_id = 4;
        else if (ud->cache->cache_type == GGML_TYPE_Q8_0)
            cache_type_id = 8;
    }
    const auto k_cache_layout = ud->cache->GetBlockLayout();
    const auto v_cache_layout = ud->cache->GetVBlockLayout();
    const bool hwy_ready = use_hwy && cache_type_id >= 0 && k_cache_layout.head_stride_bytes > 0 &&
                           k_cache_layout.slot_stride_bytes > 0 && v_cache_layout.head_stride_bytes > 0 &&
                           v_cache_layout.slot_stride_bytes > 0;
    thread_local std::vector<const void*> k_block_ptrs;
    thread_local std::vector<const void*> v_block_ptrs;
    int cached_token_idx = -1;
    int cached_seq_idx = -1;
    const std::vector<int>* cached_block_table = nullptr;

    auto zero_token_heads = [&](float* out_token, int h_start, int h_end) {
        for (int h = h_start; h < h_end; ++h) {
            float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
            std::fill(out_head, out_head + v_head_dim, 0.0f);
        }
    };

    std::chrono::steady_clock::time_point attn_begin;
    if (do_profile) attn_begin = std::chrono::steady_clock::now();

    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int token_idx = tile / tiles_per_token;
        const int head_tile_idx = tile % tiles_per_token;
        const int h_start = head_tile_idx * head_tile;
        const int h_end = std::min(ud->n_head, h_start + head_tile);
        if (h_start >= h_end) {
            continue;
        }

        const float* q_token = reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
        float* out_token = reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);

        if (token_idx < 0 || token_idx >= q_tokens) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
        if (seq_idx < 0 || seq_idx >= batch->num_seqs) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
        const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
        if (pos_i < 0 || n_past_i < 0) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        const KVRetentionSpan retained_span = ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const bool retention_truncated = retained_span.history_kept < n_past_i;
        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len = std::max(1, std::min(retained_span.history_kept + 1, max_context));
        if (context_len <= 0) {
            zero_token_heads(out_token, h_start, h_end);
            continue;
        }

        if (!hwy_ready || retention_truncated) {
            ComputePagedAttentionScalarHeads(ud, block_table, context_len, retained_span, pos_i, scale, q_token,
                                             out_token, h_start, h_end);
            continue;
        }

        if (token_idx != cached_token_idx || seq_idx != cached_seq_idx || cached_block_table != &block_table) {
            k_block_ptrs.resize(block_table.size(), nullptr);
            v_block_ptrs.resize(block_table.size(), nullptr);
            for (size_t bi = 0; bi < block_table.size(); ++bi) {
                const int block_id = block_table[bi];
                if (block_id < 0 || block_id >= ud->cache->max_blocks) {
                    continue;
                }
                k_block_ptrs[bi] = ud->cache->GetKBlockPtr(block_id, ud->layer);
                v_block_ptrs[bi] = ud->cache->GetVBlockPtr(block_id, ud->layer);
            }
            cached_token_idx = token_idx;
            cached_seq_idx = seq_idx;
            cached_block_table = &block_table;
        }

        densecore::hwy_kernels::PagedAttention_Hwy(
            q_token, k_block_ptrs.data(), v_block_ptrs.data(), cache_type_id, ud->n_head, ud->head_dim, v_head_dim,
            ud->cache->n_head_kv, static_cast<int32_t>(block_table.size()), context_len,
            static_cast<int64_t>(k_cache_layout.head_stride_bytes),
            static_cast<int64_t>(k_cache_layout.slot_stride_bytes),
            static_cast<int64_t>(v_cache_layout.head_stride_bytes),
            static_cast<int64_t>(v_cache_layout.slot_stride_bytes), scale, out_token, h_start, h_end, ud->n_head);
    }

    // Decode profiling: log KV write + attention compute timing (thread 0, every 100th call)
    if (do_profile) {
        const auto attn_end = std::chrono::steady_clock::now();
        static thread_local int profile_cb_count = 0;
        if (++profile_cb_count % 100 == 0) {
            const long kv_us = std::chrono::duration_cast<std::chrono::microseconds>(kv_end - kv_begin).count();
            const long attn_us = std::chrono::duration_cast<std::chrono::microseconds>(attn_end - attn_begin).count();
            fprintf(stderr, "[DecodeProfile] bs=%d threads=%d layer=%d kv_us=%ld attn_us=%ld\n", q_tokens, nth,
                    ud->layer, kv_us, attn_us);
        }
    }
}

void cb_glm_dsa_attention_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<PagedAttentionUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data || nth <= 0) return;
    if (!dst->src[0] || !dst->src[1] || !dst->src[2] || !dst->src[3] || !dst->src[4] || !dst->src[5]) return;

    const auto* q_tensor = dst->src[0];
    const auto* k_tensor = dst->src[1];
    const auto* v_tensor = dst->src[2];
    const auto* index_q_tensor = dst->src[3];
    const auto* index_weights_tensor = dst->src[4];
    const auto* index_k_tensor = dst->src[5];
    if (!q_tensor->data || !k_tensor->data || !v_tensor->data || !index_q_tensor->data || !index_weights_tensor->data ||
        !index_k_tensor->data) {
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int q_tokens = static_cast<int>(q_tensor->ne[2]);
    const int n_head = ud->n_head;
    const int n_head_kv = ud->cache->n_head_kv;
    const int head_dim = ud->head_dim;
    const int v_head_dim = ud->v_head_dim > 0 ? ud->v_head_dim : ud->head_dim;
    const int index_n_heads = ud->index_n_heads;
    const int index_head_dim = ud->index_head_dim;
    const int index_topk = ud->index_topk;
    if (q_tokens <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || v_head_dim <= 0 || index_n_heads <= 0 ||
        index_head_dim <= 0 || (n_head % n_head_kv) != 0) {
        return;
    }

    const char* q_base = reinterpret_cast<const char*>(q_tensor->data);
    const char* k_base = reinterpret_cast<const char*>(k_tensor->data);
    const char* v_base = reinterpret_cast<const char*>(v_tensor->data);
    const char* index_q_base = reinterpret_cast<const char*>(index_q_tensor->data);
    const char* index_weights_base = reinterpret_cast<const char*>(index_weights_tensor->data);
    const char* index_k_base = reinterpret_cast<const char*>(index_k_tensor->data);
    char* out_base = reinterpret_cast<char*>(dst->data);

    const size_t q_token_stride = static_cast<size_t>(q_tensor->nb[2]);
    const size_t k_token_stride = static_cast<size_t>(k_tensor->nb[2]);
    const size_t v_token_stride = static_cast<size_t>(v_tensor->nb[2]);
    const size_t index_q_token_stride = static_cast<size_t>(index_q_tensor->nb[2]);
    const size_t index_weights_token_stride = static_cast<size_t>(index_weights_tensor->nb[1]);
    const size_t index_k_token_stride = static_cast<size_t>(index_k_tensor->nb[1]);
    const size_t out_token_stride = static_cast<size_t>(dst->nb[2]);
    const int kv_group_size = n_head / n_head_kv;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float index_scale = 1.0f / std::sqrt(static_cast<float>(index_head_dim));

    uint64_t epoch = 0;
    if (ith == 0) {
        ud->kv_writers_done.store(0, std::memory_order_release);
        epoch = ud->epoch_started.fetch_add(1, std::memory_order_acq_rel) + 1;
    } else {
        int spin_count = 0;
        while (true) {
            const uint64_t started = ud->epoch_started.load(std::memory_order_acquire);
            const uint64_t done = ud->epoch_done.load(std::memory_order_acquire);
            if (started > done) {
                epoch = started;
                break;
            }
            SpinPause(spin_count++);
        }
    }

    for (int i = ith; i < q_tokens; i += nth) {
        if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) continue;
        const int seq_idx = batch->seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= static_cast<int>(batch->block_tables.size())) continue;
        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(i)];
        if (pos_i < 0) continue;

        const int logical_block = pos_i / BLOCK_SIZE;
        const int slot = pos_i % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
        const int block_id = block_table[static_cast<size_t>(logical_block)];
        if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

        const float* k_src = reinterpret_cast<const float*>(k_base + static_cast<size_t>(i) * k_token_stride);
        const float* v_src = reinterpret_cast<const float*>(v_base + static_cast<size_t>(i) * v_token_stride);
        const float* index_k_src =
            reinterpret_cast<const float*>(index_k_base + static_cast<size_t>(i) * index_k_token_stride);
        ud->cache->WriteKSlot(block_id, ud->layer, slot, k_src);
        ud->cache->WriteVSlot(block_id, ud->layer, slot, v_src);
        ud->cache->WriteIndexSlot(block_id, ud->layer, slot, index_k_src);
    }

    const int writers_done = ud->kv_writers_done.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (writers_done == nth) {
        ud->epoch_done.store(epoch, std::memory_order_release);
    } else {
        int spin_count = 0;
        while (ud->epoch_done.load(std::memory_order_acquire) < epoch) {
            SpinPause(spin_count++);
        }
    }

    thread_local std::vector<float> index_key_scratch;
    thread_local std::vector<float> k_slot_scratch;
    thread_local std::vector<float> v_slot_scratch;
    thread_local std::vector<std::pair<float, int>> index_scores;
    thread_local std::vector<float> attn_scores;
    thread_local std::vector<int> selected_positions;
    thread_local std::map<int, std::vector<int>> token_selected_positions_cache;
    token_selected_positions_cache.clear();
    index_key_scratch.resize(static_cast<size_t>(index_head_dim));
    k_slot_scratch.resize(static_cast<size_t>(ud->cache->GetElementsPerSlot()));
    v_slot_scratch.resize(static_cast<size_t>(ud->cache->GetVElementsPerSlot()));

    // Flatten token and head loops for 2D parallelization
    const int total_work = q_tokens * n_head;
    for (int work_idx = ith; work_idx < total_work; work_idx += nth) {
        const int token_idx = work_idx / n_head;
        const int h = work_idx % n_head;

        if (token_idx >= static_cast<int>(batch->seq_id.size()) || token_idx >= static_cast<int>(batch->pos.size())) {
            continue;
        }

        const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
        if (seq_idx < 0 || seq_idx >= batch->num_seqs || seq_idx >= static_cast<int>(batch->block_tables.size()) ||
            seq_idx >= static_cast<int>(batch->n_past.size())) {
            continue;
        }

        const auto& block_table = batch->block_tables[static_cast<size_t>(seq_idx)];
        const int pos_i = batch->pos[static_cast<size_t>(token_idx)];
        const int n_past_i = batch->n_past[static_cast<size_t>(seq_idx)];
        if (block_table.empty() || pos_i < 0 || n_past_i < 0) {
            continue;
        }

        const KVRetentionSpan retained_span = ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy());
        const int max_context = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len = std::max(1, std::min(retained_span.history_kept + 1, max_context));
        const int effective_topk = index_topk > 0 ? std::min(index_topk, context_len) : context_len;

        // Compute selected positions once per token (with lazy cache)
        if (token_selected_positions_cache.find(token_idx) == token_selected_positions_cache.end()) {
            const float* index_weights_token = reinterpret_cast<const float*>(
                index_weights_base + static_cast<size_t>(token_idx) * index_weights_token_stride);
            const char* index_q_token_base = index_q_base + static_cast<size_t>(token_idx) * index_q_token_stride;
            index_scores.clear();
            index_scores.reserve(static_cast<size_t>(context_len));
            for (int t = 0; t < context_len; ++t) {
                const int token_pos =
                    (t < retained_span.history_kept) ? MapRetainedHistoryIndex(retained_span, t) : pos_i;
                const int logical_block = token_pos / BLOCK_SIZE;
                const int slot = token_pos % BLOCK_SIZE;
                if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) continue;
                const int block_id = block_table[static_cast<size_t>(logical_block)];
                if (block_id < 0 || block_id >= ud->cache->max_blocks) continue;

                ud->cache->ReadIndexSlot(block_id, ud->layer, slot, index_key_scratch.data());
                float score = 0.0f;
                for (int ih = 0; ih < index_n_heads; ++ih) {
                    const float* q_index_head = reinterpret_cast<const float*>(
                        index_q_token_base + static_cast<size_t>(ih) * index_q_tensor->nb[1]);
                    float dot = 0.0f;
                    for (int d = 0; d < index_head_dim; ++d) {
                        dot += q_index_head[d] * index_key_scratch[static_cast<size_t>(d)];
                    }
                    score += index_weights_token[ih] * (dot * index_scale);
                }
                index_scores.emplace_back(score, token_pos);
            }

            if (!index_scores.empty()) {
                if (static_cast<int>(index_scores.size()) > effective_topk) {
                    std::partial_sort(index_scores.begin(), index_scores.begin() + effective_topk, index_scores.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                }
                const int selected_count = std::min(effective_topk, static_cast<int>(index_scores.size()));
                std::vector<int>& cached = token_selected_positions_cache[token_idx];
                cached.resize(static_cast<size_t>(selected_count));
                for (int i = 0; i < selected_count; ++i) {
                    cached[static_cast<size_t>(i)] = index_scores[static_cast<size_t>(i)].second;
                }
            }
        }

        const auto& cached_it = token_selected_positions_cache.find(token_idx);
        if (cached_it == token_selected_positions_cache.end() || cached_it->second.empty()) {
            continue;
        }
        const std::vector<int>& selected_positions_ref = cached_it->second;
        const int selected_count = static_cast<int>(selected_positions_ref.size());

        float* out_token = reinterpret_cast<float*>(out_base + static_cast<size_t>(token_idx) * out_token_stride);
        const float* q_token = reinterpret_cast<const float*>(q_base + static_cast<size_t>(token_idx) * q_token_stride);
        const float* q_head = q_token + static_cast<size_t>(h) * head_dim;
        float* out_head = out_token + static_cast<size_t>(h) * v_head_dim;
        std::fill(out_head, out_head + v_head_dim, 0.0f);

        const int kv_head = std::min(n_head_kv - 1, std::max(0, h / kv_group_size));
        float max_score = -std::numeric_limits<float>::infinity();
        attn_scores.resize(static_cast<size_t>(selected_count));
        for (int i = 0; i < selected_count; ++i) {
            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadKSlot(block_id, ud->layer, slot, k_slot_scratch.data());
            const float* k_head = k_slot_scratch.data() + static_cast<size_t>(kv_head) * head_dim;

            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_head[d] * k_head[d];
            }
            score *= attn_scale;
            attn_scores[static_cast<size_t>(i)] = score;
            if (score > max_score) max_score = score;
        }

        if (!std::isfinite(max_score)) {
            continue;
        }

        float denom = 0.0f;
        for (int i = 0; i < selected_count; ++i) {
            const float weight = std::exp(attn_scores[static_cast<size_t>(i)] - max_score);
            if (!(weight > 0.0f) || !std::isfinite(weight)) continue;
            denom += weight;

            const int token_pos = selected_positions_ref[static_cast<size_t>(i)];
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            const int block_id = block_table[static_cast<size_t>(logical_block)];
            ud->cache->ReadVSlot(block_id, ud->layer, slot, v_slot_scratch.data());
            const float* v_head = v_slot_scratch.data() + static_cast<size_t>(kv_head) * v_head_dim;
            for (int d = 0; d < v_head_dim; ++d) {
                out_head[d] += weight * v_head[d];
            }
        }

        if (!(denom > 0.0f) || !std::isfinite(denom)) {
            std::fill(out_head, out_head + v_head_dim, 0.0f);
            continue;
        }
        const float inv = 1.0f / denom;
        for (int d = 0; d < v_head_dim; ++d) {
            out_head[d] *= inv;
        }
    }
}

inline struct ggml_tensor* ggml_glm_dsa_attention(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                  struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                  struct ggml_tensor* index_q, struct ggml_tensor* index_weights,
                                                  struct ggml_tensor* index_k, PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;
    result->src[3] = index_q;
    result->src[4] = index_weights;
    result->src[5] = index_k;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_head = std::max(1, static_cast<int>(q_cur->ne[1]));
    const int q_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    n_tasks = std::max(1, std::min(n_tasks, q_tokens * n_head));

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_glm_dsa_attention_custom, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

/**
 * Create a custom GGML operation for parallel GEMV
 *
 * REFACTORED: Uses GGML_OP_CUSTOM instead of GGML_OP_MAP_CUSTOM1.
 * GGML_OP_MAP_CUSTOM1 assumes output shape == input shape, which causes
 * buffer overflows when Qwen3 projections change dimensions (e.g., 1024->2048).
 * GGML_OP_CUSTOM allows the output tensor shape to be independent of inputs.
 */
inline struct ggml_tensor* ggml_mul_mat_gemv(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input, GemvUserData* userdata) {
    const int K = weight->ne[1];  // Output dimension
    const int N = weight->ne[0];  // Input dimension

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;

    const ggml_type wtype = weight->type;
    if (ggml_is_quantized(wtype)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
            const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                const size_t quant_input_size = ggml_row_size(vec_dot_type, N);
                if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                    userdata->input_quant_type = vec_dot_type;
                }
            }
        }
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }

    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;

    n_threads = std::min(n_threads, physical_cores);

    if (K < 64) {
        n_threads = 1;
    } else if (K < 512) {
        n_threads = std::min(n_threads, 2);
    } else if (K < 1536) {
        n_threads = std::min(n_threads, 4);
    } else if (K < 3072) {
        n_threads = std::min(n_threads, 6);
    }

    // ===========================================================================
    // Create output tensor with correct dimension K (INDEPENDENT of input shape)
    // This is the critical fix: GGML_OP_CUSTOM allows explicit output dimensions
    // ===========================================================================
    const int64_t ne_res[4] = {K, 1, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    // ===========================================================================
    // Configure GGML_OP_CUSTOM (NOT MAP_CUSTOM1 which assumes shape preservation)
    // ===========================================================================
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;   // Input tensor accessible via dst->src[0] in callback
    result->src[1] = weight;  // Weight tensor accessible via dst->src[1] in callback

    // Custom op params (layout must match ggml_custom_op_params)
    // Signature: { ggml_custom_op_t fun, int n_tasks, void *userdata }
    // NOTE: userdata is still passed for pre-quantized input buffer pointer,
    //       but dimensions/weight are read directly from dst->src[1] for
    //       reliability
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

inline struct ggml_tensor* ggml_mul_mat_gemv_batched(struct ggml_context* ctx, struct ggml_tensor* weight,
                                                     struct ggml_tensor* input, GemvBatchedUserData* userdata) {
    const int K = static_cast<int>(weight->ne[1]);  // Output dimension
    const int N = static_cast<int>(weight->ne[0]);  // Input dimension
    const int M = static_cast<int>(input->ne[1]);   // Batch columns
    if (K <= 0 || N <= 0 || M <= 0) {
        return ggml_mul_mat(ctx, weight, input);
    }

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->M = M;
    userdata->weight_type = weight->type;
    userdata->input_quant_type = GGML_TYPE_F32;

    if (ggml_is_quantized(weight->type)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        if (!type_traits_cpu || !type_traits_cpu->vec_dot || M > kMaxSmallBatchColsHard) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        if (!input_type_traits || !input_type_traits->from_float) {
            return ggml_mul_mat(ctx, weight, input);
        }

        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        const size_t quant_row_stride = densecore::AlignUp(quant_row_size, static_cast<size_t>(64));
        const size_t quant_buffer_capacity =
            static_cast<size_t>(kMaxQuantInputBufferSize) * static_cast<size_t>(kMaxSmallBatchColsHard);

        if (quant_row_size == 0 || quant_row_stride > kMaxQuantInputBufferSize ||
            quant_row_stride * static_cast<size_t>(M) > quant_buffer_capacity) {
            return ggml_mul_mat(ctx, weight, input);
        }

        userdata->input_quant_type = vec_dot_type;
        userdata->quant_row_stride = quant_row_stride;
    }

    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_threads = std::min(n_threads, physical_cores);
    }
    if (K < 256) {
        n_threads = std::min(n_threads, 2);
    } else if (K < 1024) {
        n_threads = std::min(n_threads, 4);
    }
    n_threads = std::max(1, std::min(n_threads, K));

    const int64_t ne_res[4] = {K, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_batched_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

inline struct ggml_tensor* ggml_paged_attention_decode(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                       struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                       PagedAttentionUserData* userdata) {
    const int64_t ne_res[4] = {v_cur->ne[0], q_cur->ne[1], q_cur->ne[2], 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = q_cur;
    result->src[1] = k_cur;
    result->src[2] = v_cur;

    const BatchSpec* batch = GetCurrentBatch();
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    const int n_heads = static_cast<int>(q_cur->ne[1]);
    const int n_tokens = std::max(1, static_cast<int>(q_cur->ne[2]));
    const int head_tile = std::max(1, ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", 8));
    const int tiles_per_token = std::max(1, (std::max(1, n_heads) + head_tile - 1) / head_tile);
    const int total_tiles = std::max(1, n_tokens * tiles_per_token);
    // Token-parallel mode: when batch has multiple tokens and threads >= tokens,
    // assign 1 thread per token (each handles all heads). Eliminates KV write
    // barrier overhead and improves cache locality for small batches.
    const bool token_parallel = (n_tokens > 1 && n_tokens <= n_tasks);
    if (token_parallel) {
        n_tasks = n_tokens;
    } else {
        n_tasks = std::max(1, std::min(n_tasks, total_tiles));
    }
    // Scalar fallback remains correctness-first, but we keep the same token/head
    // tiling to preserve parallelism on non-Highway hosts.
    if (!token_parallel && !IsPagedAttentionHwyEnabled()) {
        const int scalar_tasks = ParsePositiveEnvInt("DENSECORE_PAGED_ATTN_SCALAR_TASKS", n_tasks);
        n_tasks = std::max(1, std::min(scalar_tasks, total_tiles));
    }
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_paged_attention_decode, n_tasks, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));

    return result;
}

// ============================================================================
// oneDNN MatMul Custom Op (Prefill-only acceleration)
// ============================================================================

struct MatmulOpData {
    densecore::MatmulBackendKind backend;
    densecore::DType a_type;
    densecore::DType b_type;
    densecore::DType c_type;
    bool convert_a_f32_to_bf16 = false;
};

struct MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    MatmulOpData data;
};

// GgmlTypeToDType is now defined in dtype_utils.h

void cb_matmul_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (ith != 0 || nth <= 0) {
        return;
    }

    const auto* params = reinterpret_cast<const MatmulCustomParams*>(dst->op_params);
    if (!params || !dst || !dst->src[0] || !dst->src[1]) return;
    const MatmulOpData& ud = params->data;

    const struct ggml_tensor* input = dst->src[0];
    const struct ggml_tensor* weight = dst->src[1];
    if (!input->data || !weight->data || !dst->data) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    const int64_t N = weight->ne[1];

    const int64_t lda = input->nb[1] / ggml_type_size(input->type);
    const int64_t ldb = weight->nb[1] / ggml_type_size(weight->type);
    const int64_t ldc = dst->nb[1] / ggml_type_size(dst->type);

    densecore::MatmulParams matmul_params;
    const void* a_ptr = input->data;
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        throw densecore::InvalidArgumentException("cb_matmul_custom called without active InferenceWorkContext");
    }
    std::vector<ggml_bf16_t>& bf16_buffer = work_ctx->bf16_buffer;
    bool used_bf16_buffer = false;
    if (ud.convert_a_f32_to_bf16 && input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_BF16 && M > 1) {
        const size_t total = static_cast<size_t>(M * K);
        if (bf16_buffer.size() < total) {
            bf16_buffer.resize(total);
        }
        for (int64_t m = 0; m < M; ++m) {
            const float* src =
                reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
            ggml_fp32_to_bf16_row(src, bf16_buffer.data() + m * K, K);
        }
        a_ptr = bf16_buffer.data();
        used_bf16_buffer = true;
    }
    matmul_params.a = a_ptr;
    matmul_params.b = weight->data;
    matmul_params.c = dst->data;
    matmul_params.M = M;
    matmul_params.N = N;
    matmul_params.K = K;
    matmul_params.lda = used_bf16_buffer ? K : lda;
    matmul_params.ldb = ldb;
    matmul_params.ldc = ldc;
    matmul_params.trans_b = true;
    matmul_params.a_type = ud.a_type;
    matmul_params.b_type = ud.b_type;
    matmul_params.c_type = ud.c_type;

    densecore::MatmulBackend& backend = (ud.backend == densecore::MatmulBackendKind::OneDNN)
                                            ? densecore::GetOneDnnMatmulBackend()
                                            : densecore::GetDenseCoreMatmulBackend();
    backend.Execute(matmul_params);
}

inline struct ggml_tensor* ggml_mul_mat_onednn(struct ggml_context* ctx, struct ggml_tensor* weight,
                                               struct ggml_tensor* input, const MatmulOpData& data) {
    const int64_t N = weight->ne[1];  // Output dimension
    const int64_t M = input->ne[1];   // Tokens

    const int64_t ne_res[4] = {N, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    MatmulCustomParams params = {cb_matmul_custom, 1, nullptr, data};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

struct HalMatmulOpData {
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
};

struct HalMatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalMatmulOpData data;
};

void cb_matmul_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (ith != 0 || nth <= 0 || !dst || !dst->src[0] || !dst->src[1]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalMatmulCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }
    const struct ggml_tensor* input = dst->src[0];
    const struct ggml_tensor* weight = dst->src[1];
    if (!input->data || !weight->data || !dst->data) {
        return;
    }

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(input->ne[0]) * sizeof(float);
    const bool weight_contig =
        weight->nb[0] == sizeof(float) && weight->nb[1] == static_cast<size_t>(weight->ne[0]) * sizeof(float);
    const bool output_contig =
        dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(dst->ne[0]) * sizeof(float);

    if (input->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !input_contig ||
        !weight_contig || !output_contig) {
        // Fallback to local backend matmul when HAL path is inapplicable.
        densecore::MatmulParams matmul_params;
        matmul_params.a = input->data;
        matmul_params.b = weight->data;
        matmul_params.c = dst->data;
        matmul_params.M = input->ne[1];
        matmul_params.N = weight->ne[1];
        matmul_params.K = input->ne[0];
        matmul_params.lda = input->nb[1] / ggml_type_size(input->type);
        matmul_params.ldb = weight->nb[1] / ggml_type_size(weight->type);
        matmul_params.ldc = dst->nb[1] / ggml_type_size(dst->type);
        matmul_params.trans_b = true;
        matmul_params.a_type = densecore::DType::F32;
        matmul_params.b_type = densecore::DType::F32;
        matmul_params.c_type = densecore::DType::F32;
        densecore::GetDenseCoreMatmulBackend().Execute(matmul_params);
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    densecore::BackendRegistry& registry = ResolveBackendRegistry(batch);

    densecore::ComputeBackend* backend = registry.Get(params->data.preferred_device);
    if (!backend && params->data.preferred_device != densecore::DeviceType::CPU) {
        backend = registry.Get(densecore::DeviceType::CPU);
    }
    if (!backend) {
        return;
    }

    densecore::Tensor A = densecore::Tensor::Make2D(const_cast<void*>(input->data), input->ne[1], input->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor B = densecore::Tensor::Make2D(const_cast<void*>(weight->data), weight->ne[1], weight->ne[0],
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor C =
        densecore::Tensor::Make2D(dst->data, dst->ne[1], dst->ne[0], densecore::DType::F32, densecore::DeviceType::CPU);
    backend->MatMulTransB(A, B, &C);
}

inline struct ggml_tensor* ggml_mul_mat_hal(struct ggml_context* ctx, struct ggml_tensor* weight,
                                            struct ggml_tensor* input, densecore::DeviceType preferred_device) {
    const int64_t N = weight->ne[1];
    const int64_t M = input->ne[1];

    const int64_t ne_res[4] = {N, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    HalMatmulCustomParams params = {cb_matmul_hal_custom, 1, nullptr, {preferred_device}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct HalAttentionOpData {
    struct ggml_tensor* k_tensor = nullptr;
    struct ggml_tensor* v_tensor = nullptr;
    float scale = 1.0f;
    int n_head_kv = -1;
    uint8_t causal = 1;
    densecore::DeviceType preferred_device = densecore::DeviceType::CPU;
};

struct HalAttentionCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    HalAttentionOpData data;
};

void cb_flash_attention_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (ith != 0 || nth <= 0 || !dst || !dst->src[0]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalAttentionCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }

    const struct ggml_tensor* q = dst->src[0];
    const struct ggml_tensor* k = params->data.k_tensor;
    const struct ggml_tensor* v = params->data.v_tensor;
    if (!q || !k || !v || !q->data || !k->data || !v->data || !dst->data) {
        return;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return;
    }
    if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || !ggml_is_contiguous(dst)) {
        return;
    }
    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || k->ne[0] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
        v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0) {
        return;
    }
    if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2]) {
        return;
    }

    const int head_dim = static_cast<int>(q->ne[0]);
    const int seq_q = static_cast<int>(q->ne[1]);
    const int n_head = static_cast<int>(q->ne[2]);
    const int seq_kv = static_cast<int>(k->ne[1]);
    const int inferred_n_head_kv = static_cast<int>(k->ne[2]);
    const int n_head_kv = params->data.n_head_kv > 0 ? params->data.n_head_kv : inferred_n_head_kv;

    if (n_head_kv <= 0 || n_head <= 0 || seq_q <= 0 || seq_kv <= 0 || head_dim <= 0) {
        return;
    }
    if (n_head_kv != inferred_n_head_kv) {
        return;
    }
    if (n_head % n_head_kv != 0) {
        return;
    }

    const BatchSpec* batch = GetCurrentBatch();
    densecore::BackendRegistry& registry = ResolveBackendRegistry(batch);
    densecore::ComputeBackend* backend = registry.Get(params->data.preferred_device);
    if (!backend && params->data.preferred_device != densecore::DeviceType::CPU) {
        backend = registry.Get(densecore::DeviceType::CPU);
    }
    if (!backend) {
        return;
    }

    densecore::Tensor Q = densecore::Tensor::Make4D(const_cast<void*>(q->data), 1, n_head, seq_q, head_dim,
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor K = densecore::Tensor::Make4D(const_cast<void*>(k->data), 1, n_head_kv, seq_kv, head_dim,
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor V = densecore::Tensor::Make4D(const_cast<void*>(v->data), 1, n_head_kv, seq_kv, head_dim,
                                                    densecore::DType::F32, densecore::DeviceType::CPU);
    densecore::Tensor O = densecore::Tensor::Make4D(dst->data, 1, n_head, seq_q, head_dim, densecore::DType::F32,
                                                    densecore::DeviceType::CPU);

    try {
        backend->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv);
    } catch (...) {
        densecore::ComputeBackend* cpu = registry.Get(densecore::DeviceType::CPU);
        if (cpu && cpu != backend) {
            cpu->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv);
        }
    }
}

inline struct ggml_tensor* ggml_flash_attention_hal(struct ggml_context* ctx, struct ggml_tensor* Q,
                                                    struct ggml_tensor* K, struct ggml_tensor* V, float scale,
                                                    bool causal, int n_head_kv,
                                                    densecore::DeviceType preferred_device) {
    const int64_t ne_res[4] = {Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);
    result->op = GGML_OP_CUSTOM;
    result->src[0] = Q;

    HalAttentionCustomParams params = {
        cb_flash_attention_hal_custom,
        1,
        nullptr,
        {K, V, scale, n_head_kv, static_cast<uint8_t>(causal ? 1 : 0), preferred_device}};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    std::memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct Int4MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    const float* scales = nullptr;
    const float* zeros = nullptr;
    int K = 0;
    int N = 0;
    int group_size = 0;
};

struct Int4MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    Int4MatmulOpData data;
};

void cb_matmul_int4_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const Int4MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const Int4MatmulOpData& ud = params->data;
    if (!ud.packed_weights || !ud.scales || !ud.zeros || ud.K <= 0 || ud.N <= 0 || ud.group_size <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    const bool input_contig =
        input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
    const bool output_contig = dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);
    const bool single_threading_layer = IsInt4SingleThreadingLayerEnabled();

    // Legacy escape hatch: keep DenseCore backend threadpool path for
    // platform-specific tuning. This path is intentionally serialized at GGML
    // level to avoid nested parallelism.
    if (!single_threading_layer) {
        if (ith != 0) return;

        static thread_local std::vector<float> legacy_contig_input;
        static thread_local std::vector<float> legacy_contig_output;
        const float* input_ptr = reinterpret_cast<const float*>(input->data);
        float* output_ptr = reinterpret_cast<float*>(dst->data);

        if (!input_contig || !output_contig) {
            legacy_contig_input.resize(static_cast<size_t>(M * ud.K));
            legacy_contig_output.resize(static_cast<size_t>(M * ud.N));

            for (int64_t m = 0; m < M; ++m) {
                const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(legacy_contig_input.data() + m * ud.K, src_row,
                                static_cast<size_t>(ud.K) * sizeof(float));
                } else {
                    for (int k = 0; k < ud.K; ++k) {
                        legacy_contig_input[static_cast<size_t>(m) * ud.K + k] =
                            *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
                    }
                }
            }
            input_ptr = legacy_contig_input.data();
            output_ptr = legacy_contig_output.data();
        }

        densecore::Tensor A = densecore::Tensor::Make2D(const_cast<float*>(input_ptr), M, ud.K);
        densecore::Tensor W =
            densecore::Tensor::Make2D(const_cast<uint8_t*>(ud.packed_weights), ud.N, ud.K, densecore::DType::INT8);
        const int64_t groups_per_row = ud.K / ud.group_size;
        densecore::Tensor S = densecore::Tensor::Make2D(const_cast<float*>(ud.scales), ud.N, groups_per_row);
        densecore::Tensor Z = densecore::Tensor::Make2D(const_cast<float*>(ud.zeros), ud.N, groups_per_row);
        densecore::Tensor C = densecore::Tensor::Make2D(output_ptr, M, ud.N);
        densecore::GetCpuBackend().GemmInt4(A, W, S, Z, &C, ud.group_size);

        if (!input_contig || !output_contig) {
            for (int64_t m = 0; m < M; ++m) {
                char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
                if (dst->nb[0] == sizeof(float)) {
                    std::memcpy(dst_row, legacy_contig_output.data() + m * ud.N,
                                static_cast<size_t>(ud.N) * sizeof(float));
                } else {
                    for (int n = 0; n < ud.N; ++n) {
                        *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                            legacy_contig_output[static_cast<size_t>(m) * ud.N + n];
                    }
                }
            }
        }
        return;
    }

    // GGML-thread partitioning over N tiles: each task writes a disjoint output
    // column range [n_start, n_end), so no synchronization is required.
    const int n_per_task = (ud.N + nth - 1) / nth;
    const int n_start = ith * n_per_task;
    const int n_end = std::min(ud.N, n_start + n_per_task);
    if (n_start >= n_end) return;

    // Use batched kernel whenever input elements are contiguous within each row
    // (nb[0] == sizeof(float)). The kernel handles row padding via input_stride_bytes,
    // eliminating the per-row gather/scatter fallback for padded GGML tensors.
    const bool input_elements_contig = input->nb[0] == sizeof(float);
    if (input_elements_contig && output_contig) {
        const float* in_ptr = reinterpret_cast<const float*>(input->data);
        float* out_ptr = reinterpret_cast<float*>(dst->data);
        densecore::Ops::GemmInt4Batched(out_ptr, in_ptr, ud.packed_weights, ud.scales, ud.zeros, static_cast<int>(M),
                                        ud.K, ud.N, ud.group_size, 0, static_cast<int>(M), n_start, n_end,
                                        input->nb[1]);
        return;
    }

    // Strided fallback: gather one input row at a time, compute assigned output
    // tile, scatter back. This path avoids backend threadpool usage entirely.
    static thread_local std::vector<float> gathered_input;
    static thread_local std::vector<float> partial_output;
    const int n_count = n_end - n_start;
    gathered_input.resize(static_cast<size_t>(ud.K));
    partial_output.resize(static_cast<size_t>(n_count));

    for (int64_t m = 0; m < M; ++m) {
        const char* src_row = reinterpret_cast<const char*>(input->data) + m * input->nb[1];
        if (input->nb[0] == sizeof(float)) {
            std::memcpy(gathered_input.data(), src_row, static_cast<size_t>(ud.K) * sizeof(float));
        } else {
            for (int k = 0; k < ud.K; ++k) {
                gathered_input[k] = *reinterpret_cast<const float*>(src_row + static_cast<size_t>(k) * input->nb[0]);
            }
        }

        densecore::hwy_kernels::GemvInt4_Hwy(partial_output.data(), gathered_input.data(), ud.packed_weights, ud.scales,
                                             ud.zeros, ud.K, ud.N, ud.group_size, n_start, n_end);

        char* dst_row = reinterpret_cast<char*>(dst->data) + m * dst->nb[1];
        if (dst->nb[0] == sizeof(float)) {
            float* dst_row_f32 = reinterpret_cast<float*>(dst_row);
            std::memcpy(dst_row_f32 + n_start, partial_output.data(), static_cast<size_t>(n_count) * sizeof(float));
        } else {
            for (int n = n_start; n < n_end; ++n) {
                *reinterpret_cast<float*>(dst_row + static_cast<size_t>(n) * dst->nb[0]) =
                    partial_output[static_cast<size_t>(n - n_start)];
            }
        }
    }
}

inline struct ggml_tensor* ggml_mul_mat_int4(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input,
                                             const TransformerModel::Int4WeightBinding& binding) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    Int4MatmulCustomParams params = {};
    params.fun = cb_matmul_int4_custom;
    const BatchSpec* batch = GetCurrentBatch();
    if (IsInt4SingleThreadingLayerEnabled()) {
        params.n_tasks = ResolveTaskCount(batch, static_cast<int>(std::max<int64_t>(1, binding.n)));
    } else {
        params.n_tasks = 1;
    }
    params.userdata = nullptr;
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.scales = binding.scales ? reinterpret_cast<const float*>(binding.scales->data) : nullptr;
    params.data.zeros = binding.zeros ? reinterpret_cast<const float*>(binding.zeros->data) : nullptr;
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.group_size = binding.group_size;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

struct FP8MatmulOpData {
    const uint8_t* packed_weights = nullptr;
    int K = 0;
    int N = 0;
    TransformerModel::FP8Format format = TransformerModel::FP8Format::E4M3FN;
};

struct FP8MatmulCustomParams {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
    FP8MatmulOpData data;
};

void cb_matmul_fp8_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (nth <= 0) return;
    if (!dst || !dst->src[0]) return;

    const auto* params = reinterpret_cast<const FP8MatmulCustomParams*>(dst->op_params);
    if (!params) return;
    const FP8MatmulOpData& ud = params->data;
    if (!ud.packed_weights || ud.K <= 0 || ud.N <= 0) return;

    const struct ggml_tensor* input = dst->src[0];
    if (!input || !input->data || !dst->data || input->type != GGML_TYPE_F32) return;

    const int64_t M = input->ne[1];
    const int64_t K = input->ne[0];
    if (K != ud.K || M <= 0) return;

    static std::once_flag fp8_lut_once;
    std::call_once(fp8_lut_once, []() { densecore::hwy_kernels::InitFP8LUTs_Hwy(); });

    // =========================================================================
    // FP8 Batched GEMM: Tile-based dequant for weight reuse across M tokens
    // =========================================================================
    // For M>1, dequantizing FP8 weights tile-by-tile and computing F32 GEMM
    // on tiles avoids reloading the full weight matrix for each token.
    //
    // Tile size: TILE_N weight rows × K elements dequantized to F32.
    // The F32 tile stays in L2 cache while all M activation rows access it.
    // =========================================================================
    if (M > 1) {
        const bool input_contig =
            input->nb[0] == sizeof(float) && input->nb[1] == static_cast<size_t>(ud.K) * sizeof(float);
        const bool output_contig =
            dst->nb[0] == sizeof(float) && dst->nb[1] == static_cast<size_t>(ud.N) * sizeof(float);

        // Thread-local dequant buffer: TILE_N × K floats
        constexpr int FP8_TILE_N = 8;  // 8 × 4096 × 4 = 128KB (fits L2)
        const int total_tiles = (ud.N + FP8_TILE_N - 1) / FP8_TILE_N;
        const int tiles_per_task = (total_tiles + nth - 1) / nth;
        const int tile_start = ith * tiles_per_task;
        const int tile_end = std::min(total_tiles, tile_start + tiles_per_task);
        if (tile_start >= tile_end) return;

        static thread_local std::vector<float> fp8_dequant_buf;
        fp8_dequant_buf.resize(static_cast<size_t>(FP8_TILE_N) * ud.K);

        const uint8_t* w_fp8 = ud.packed_weights;

        for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const int n = tile_idx * FP8_TILE_N;
            const int tile_n = std::min(FP8_TILE_N, ud.N - n);

            // Dequantize tile_n weight rows: FP8 → F32 (done ONCE per tile)
            for (int i = 0; i < tile_n; i++) {
                const uint8_t* w_row = w_fp8 + static_cast<size_t>(n + i) * ud.K;
                float* f32_row = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                if (ud.format == TransformerModel::FP8Format::E5M2) {
                    densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                } else {
                    densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(w_row, f32_row, static_cast<int64_t>(ud.K));
                }
            }

            // Tiled GEMM: C[:, n:n+tile_n] += A × W_tile^T
            // Weight tile in L2, activation rows cycle through L1
            for (int64_t m = 0; m < M; m++) {
                const float* a_row = nullptr;
                if (input_contig) {
                    a_row = reinterpret_cast<const float*>(input->data) + m * ud.K;
                } else {
                    a_row =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
                }
                float* c_row = nullptr;
                if (output_contig) {
                    c_row = reinterpret_cast<float*>(dst->data) + m * ud.N;
                } else {
                    c_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
                }
                for (int i = 0; i < tile_n; i++) {
                    const float* w_row_f32 = fp8_dequant_buf.data() + static_cast<size_t>(i) * ud.K;
                    c_row[n + i] = densecore::simd::DotF32(a_row, w_row_f32, static_cast<size_t>(ud.K));
                }
            }
        }
        return;
    }

    // =========================================================================
    // M=1 DECODE: Single-token GEMV (existing optimized path)
    // =========================================================================
    const int64_t rows_per_task = (M + nth - 1) / nth;
    const int64_t m_start = static_cast<int64_t>(ith) * rows_per_task;
    const int64_t m_end = std::min<int64_t>(M, m_start + rows_per_task);
    if (m_start >= m_end) return;

    for (int64_t m = m_start; m < m_end; ++m) {
        const float* x_row =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(input->data) + m * input->nb[1]);
        float* y_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + m * dst->nb[1]);
        if (ud.format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::Gemv_FP8_E5M2_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights, x_row,
                                                      /*beta=*/0.0f, y_row);
        } else {
            densecore::hwy_kernels::Gemv_FP8_E4M3FN_Hwy(/*M=*/ud.N, /*N=*/ud.K, /*alpha=*/1.0f, ud.packed_weights,
                                                        x_row,
                                                        /*beta=*/0.0f, y_row);
        }
    }
}

inline struct ggml_tensor* ggml_mul_mat_fp8(struct ggml_context* ctx, struct ggml_tensor* weight,
                                            struct ggml_tensor* input,
                                            const TransformerModel::FP8WeightBinding& binding) {
    const int64_t M = input->ne[1];
    const int64_t ne_res[4] = {binding.n, M, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;
    result->src[1] = weight;

    FP8MatmulCustomParams params = {};
    params.fun = cb_matmul_fp8_custom;
    const BatchSpec* batch = GetCurrentBatch();
    int n_threads = ResolveInferenceConfig(batch).num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores <= 0) physical_cores = 4;
    n_threads = std::min(n_threads, physical_cores);
    constexpr int FP8_TILE_N = 8;
    int64_t max_parallel_tasks = M;
    if (M > 1) {
        max_parallel_tasks = std::max<int64_t>(1, (binding.n + FP8_TILE_N - 1) / FP8_TILE_N);
    }
    const int64_t capped_tasks = std::min<int64_t>(static_cast<int64_t>(n_threads), max_parallel_tasks);
    n_threads = static_cast<int>(std::max<int64_t>(1, capped_tasks));
    params.n_tasks = n_threads;
    params.userdata = nullptr;
    params.data.packed_weights = reinterpret_cast<const uint8_t*>(binding.packed ? binding.packed->data : nullptr);
    params.data.K = static_cast<int>(binding.k);
    params.data.N = static_cast<int>(binding.n);
    params.data.format = binding.format;

    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));
    return result;
}

/**
 * Smart matrix multiplication dispatcher
 *
 * CRITICAL:
 * - For decode-phase (batch=1), uses optimized GEMV path when dimensions are
 *   compatible.
 * - For small decode micro-batches (2 <= M <= 8), uses a custom batched path
 *   that reuses each weight row across M tokens.
 * - If incompatible (e.g., transposed layout mismatch), falls back to
 *   ggml_mul_mat which handles stride/transpose correctly.
 */
// NOTE: Not inline - needs external linkage for graph_builders/
struct ggml_tensor* smart_mul_mat(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                  TransformerModel* model) {
    const int M = static_cast<int>(input->ne[1]);
    const int K_dim = static_cast<int>(weight->ne[0]);
    const int N_dim = static_cast<int>(weight->ne[1]);
    const char* w_name = (weight->name[0] ? weight->name : "(unnamed)");

    // ========================================================================
    // PATH 1: DenseCore-packed INT4 (highest priority)
    // ========================================================================
    if (model) {
        auto it_int4 = model->int4_weight_bindings.find(weight);
        if (it_int4 != model->int4_weight_bindings.end()) {
            const auto& binding = it_int4->second;
            if (binding.k > 0 && binding.n > 0 && binding.group_size > 0 && binding.packed && binding.scales &&
                binding.zeros) {
                LogMatmulDispatch(w_name, "PACKED_INT4", M, N_dim, K_dim,
                                  M == 1 ? "HWY_INT4_GEMV" : "HWY_INT4_BATCHED");
                return ggml_mul_mat_int4(ctx, weight, input, binding);
            }
        }

        // Custom FP8 packed path (optional metadata-driven route).
        auto it_fp8 = model->fp8_weight_bindings.find(weight);
        if (it_fp8 != model->fp8_weight_bindings.end()) {
            const auto& binding = it_fp8->second;
            if (binding.k > 0 && binding.n > 0 && binding.packed) {
                LogMatmulDispatch(w_name, "PACKED_FP8", M, N_dim, K_dim, M == 1 ? "FP8_GEMV" : "FP8_TILED_GEMM");
                return ggml_mul_mat_fp8(ctx, weight, input, binding);
            }
        }
    }

    const int input_cols = M;

    // STRICT COMPATIBILITY CHECK (zero-overhead: evaluated at graph build time)
    const bool is_compatible = (weight->ne[0] == input->ne[0]);
    const BatchSpec* current_batch = GetCurrentBatch();
    const densecore::DeviceType preferred_matmul_device = ResolvePreferredMatmulDevice(current_batch);
    const bool hal_matmul_candidate = preferred_matmul_device != densecore::DeviceType::CPU && is_compatible &&
                                      weight->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 &&
                                      weight->nb[0] == static_cast<int64_t>(sizeof(float)) &&
                                      input->nb[0] == static_cast<int64_t>(sizeof(float));
    if (hal_matmul_candidate) {
        LogMatmulDispatch(w_name, "FLOAT_F32", M, N_dim, K_dim, "HAL_MATMUL_ROUTE");
        return ggml_mul_mat_hal(ctx, weight, input, preferred_matmul_device);
    }

    const int max_small_batch_cols = ParsePositiveEnvInt("DENSECORE_SMALL_BATCH_GEMV_MAX_COLS", 8);
    const int max_small_batch_quant_cols =
        ParsePositiveEnvInt("DENSECORE_SMALL_BATCH_GEMV_QUANT_MAX_COLS", kMaxSmallBatchColsHard);
    const bool is_gemv_candidate =
        (input_cols == 1) && (weight->type == GGML_TYPE_F32 || ggml_is_quantized(weight->type));
    const bool is_small_batch_f32_candidate = (input_cols > 1 && input_cols <= max_small_batch_cols &&
                                               input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    bool has_quant_vec_dot = false;
    bool has_quant_from_float = false;
    bool quant_input_size_ok = false;
    bool quant_nrc_batch_ready = false;
    bool quant_true_batched_kernel_ready = false;
    if (ggml_is_quantized(weight->type) && input->type == GGML_TYPE_F32) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            has_quant_vec_dot = true;
            const auto* input_type_traits = ggml_get_type_traits_cpu(type_traits_cpu->vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                has_quant_from_float = true;
                const size_t quant_row_size = ggml_row_size(type_traits_cpu->vec_dot_type, input->ne[0]);
                quant_input_size_ok = (quant_row_size > 0 && quant_row_size <= kMaxQuantInputBufferSize);
            }
            const int vec_dot_nrows = std::max<int>(1, static_cast<int>(type_traits_cpu->nrows));
            quant_nrc_batch_ready =
                IsQuantNrcBatchEnabled() && vec_dot_nrows >= input_cols && input_cols <= kMaxSmallBatchColsHard;
            quant_true_batched_kernel_ready = IsQ4KTrueBatchedKernelEnabled() && (weight->type == GGML_TYPE_Q4_K) &&
                                              (type_traits_cpu->vec_dot_type == GGML_TYPE_Q8_K) &&
                                              (input->ne[0] % QK_K == 0) && input_cols <= kMaxSmallBatchColsHard;
        }
    }
    const bool is_small_batch_quant_candidate =
        (input_cols > 1 && input_cols <= max_small_batch_quant_cols && input->type == GGML_TYPE_F32 &&
         ggml_is_quantized(weight->type) && !IsBatchedQuantDisabled() && has_quant_vec_dot && has_quant_from_float &&
         quant_input_size_ok);
    const bool is_small_batch_candidate = is_small_batch_f32_candidate || is_small_batch_quant_candidate;

    const char* wtype_label = MatmulWeightTypeLabel(weight->type, false, false);

    if (is_gemv_candidate && IsDebugGemvSelectionEnabled()) {
        static int dbg_gemv_ct = 0;
        if (dbg_gemv_ct < 64) {
            fprintf(stderr, "[GEMV_SEL #%d] w=%s w.ne=[%ld,%ld] in.ne=[%ld,%ld] compat=%d q=%d\n", dbg_gemv_ct, w_name,
                    (long)weight->ne[0], (long)weight->ne[1], (long)input->ne[0], (long)input->ne[1],
                    is_compatible ? 1 : 0, ggml_is_quantized(weight->type) ? 1 : 0);
            dbg_gemv_ct++;
        }
    }

    // ========================================================================
    // PATH 2: M==1 GEMV (decode single-token)
    // ========================================================================
    if (!IsCustomGemvDisabled() && is_gemv_candidate && is_compatible) {
        if (input->type != GGML_TYPE_F32) {
            fprintf(stderr, "CRITICAL: smart_mul_mat input type is %d! Tensor name: %s\n", input->type, input->name);
        }
        LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim,
                          ggml_is_quantized(weight->type) ? "GEMV_QUANT" : "GEMV_F32");
        GemvUserData* ud = GetGemvUserData();
        return ggml_mul_mat_gemv(ctx, weight, input, ud);
    }

    // ========================================================================
    // PATH 3: Small-batch (2<=M<=8) custom batched path
    //   - GGML_QUANT: shared-quant + nrc=M vec_dot (weight row reuse)
    //   - FP32: batched dot with weight row reuse
    // ========================================================================
    if (!IsCustomGemvDisabled() && is_small_batch_candidate && is_compatible) {
        if (is_small_batch_quant_candidate) {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_QUANT_NRC_M");
        } else {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "BATCHED_F32");
        }
        GemvBatchedUserData* ud = GetGemvBatchedUserData();
        return ggml_mul_mat_gemv_batched(ctx, weight, input, ud);
    }

    // Log fallback reasons for quant weights that didn't take the batched path
    if (IsDebugMatmulDispatchEnabled() && input_cols > 1 && ggml_is_quantized(weight->type) && is_compatible &&
        !is_small_batch_quant_candidate) {
        const char* reason = "unknown";
        if (IsCustomGemvDisabled())
            reason = "custom_gemv_disabled";
        else if (input_cols > max_small_batch_quant_cols)
            reason = "M>max_quant_cols";
        else if (IsBatchedQuantDisabled())
            reason = "batched_quant_disabled";
        else if (!has_quant_vec_dot)
            reason = "no_vec_dot";
        else if (!has_quant_from_float)
            reason = "no_from_float";
        else if (!quant_input_size_ok)
            reason = "quant_input_too_large";
        else if (!quant_nrc_batch_ready && !quant_true_batched_kernel_ready)
            reason = "quant_batched_kernel_unavailable";
        LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_FALLBACK", reason);
    }

    // ========================================================================
    // PATH 4: OneDNN for large batches (prefill)
    // ========================================================================
    if (is_compatible && input_cols > 1) {
        densecore::MatmulParams params;
        params.M = input->ne[1];
        params.K = input->ne[0];
        params.N = weight->ne[1];
        params.lda = input->nb[1] / ggml_type_size(input->type);
        params.ldb = weight->nb[1] / ggml_type_size(weight->type);
        params.ldc = params.N;
        params.trans_b = true;
        params.a_type = densecore::GgmlTypeToDType(input->type);
        params.b_type = densecore::GgmlTypeToDType(weight->type);
        params.c_type = densecore::DType::F32;

        const bool convert_f32_to_bf16 =
            (input->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_BF16 && input_cols > 1);
        if (convert_f32_to_bf16) {
            params.a_type = densecore::DType::BF16;
        }

        if (densecore::SelectMatmulBackend(params, true) == densecore::MatmulBackendKind::OneDNN) {
            LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "ONEDNN_GEMM");
            MatmulOpData data;
            data.backend = densecore::MatmulBackendKind::OneDNN;
            data.a_type = params.a_type;
            data.b_type = params.b_type;
            data.c_type = params.c_type;
            data.convert_a_f32_to_bf16 = convert_f32_to_bf16;
            return ggml_mul_mat_onednn(ctx, weight, input, data);
        }
    }

    // Standard GGML fallback (handles transpose/stride correctly)
    LogMatmulDispatch(w_name, wtype_label, M, N_dim, K_dim, "GGML_NATIVE");
    LogMatmulPathOnce("ggml_mul_mat");
    return ggml_mul_mat(ctx, weight, input);
}

// ============================================================================
// SIMPLIFIED UNIVERSAL ATTENTION (llama.cpp style)
// This version trades the complex paged KV cache for correctness and
// clarity. Once working, KV cache can be re-added following the proven
// llama.cpp pattern.
// ============================================================================

// ============================================================================
// MoE Forward Callbacks
// ============================================================================

// Helper: Convert GGML tensor to DenseCore HAL Tensor
inline densecore::Tensor GgmlToTensor(const struct ggml_tensor* t) {
    densecore::Tensor out;
    out.data = t->data;
    out.dtype = densecore::GgmlTypeToDType(t->type);
    for (int i = 0; i < 4; ++i) {
        out.shape[i] = t->ne[i];
        out.stride[i] = t->nb[i];
    }
    return out;
}

// NOTE: MoEUserData is defined in densecore/inference_types_internal.h

MoEUserData* AllocateMoEUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(MoEUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    return reinterpret_cast<MoEUserData*>(storage->data);
}

static GLMDSAPackUserData* AllocateGLMDSAPackUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(GLMDSAPackUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    return reinterpret_cast<GLMDSAPackUserData*>(storage->data);
}

void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int q_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * q_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * q_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim,
                   static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, src_head,
                   static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !src2 || !dst->data || !src1->data || !src2->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int k_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t kv_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t rope_row_stride = static_cast<size_t>(src2->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* kv_src = reinterpret_cast<const float*>(src1->data);
    const float* rope_src = reinterpret_cast<const float*>(src2->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* kv_row = kv_src + static_cast<size_t>(t) * kv_row_stride;
        const float* rope_row = rope_src + static_cast<size_t>(t) * rope_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* kv_head = kv_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * k_head_dim;
            memcpy(dst_head, rope_row, static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, kv_head, static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * ud->v_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim, static_cast<size_t>(ud->v_head_dim) * sizeof(float));
        }
    }
}

static std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeights(const TransformerLayer* layer) {
    using ExpertWeights = densecore::CpuBackend::ExpertWeights;

    std::vector<ExpertWeights> experts;
    if (!layer) {
        return experts;
    }

    size_t n_experts = layer->NumExperts();
    experts.reserve(n_experts);

    for (size_t i = 0; i < n_experts; ++i) {
        ExpertWeights w;
        w.w1 = {nullptr, 0};
        w.w2 = {nullptr, 0};
        w.w3 = {nullptr, 0};
        w.hidden_dim = 0;
        w.intermediate_dim = 0;

        auto* gw1 = layer->GetExpert(i, model_keys::kFfnGate);
        if (gw1) {
            w.w1.ptr = gw1->data;
            w.w1.size = ggml_nbytes(gw1);
            w.hidden_dim = static_cast<int>(gw1->ne[0]);
            w.intermediate_dim = static_cast<int>(gw1->ne[1]);
        }

        auto* gw2 = layer->GetExpert(i, model_keys::kFfnDown);
        if (gw2) {
            w.w2.ptr = gw2->data;
            w.w2.size = ggml_nbytes(gw2);
        }

        auto* gw3 = layer->GetExpert(i, model_keys::kFfnUp);
        if (gw3) {
            w.w3.ptr = gw3->data;
            w.w3.size = ggml_nbytes(gw3);
        }

        experts.push_back(w);
    }

    return experts;
}

static int GetMoERebalanceIntervalMs() {
    static const int interval_ms = std::max(250, ParsePositiveEnvInt("DENSECORE_MOE_REBALANCE_INTERVAL_MS", 5000));
    return interval_ms;
}

static int GetMoERebalanceTopK() {
    static const int top_k = std::max(1, ParsePositiveEnvInt("DENSECORE_MOE_REBALANCE_TOP_K", 4));
    return top_k;
}

static bool IsMoEPageMigrationEnabled() {
    static const bool enabled = ParseTruthyEnv("DENSECORE_MOE_ENABLE_PAGE_MIGRATION", false);
    return enabled;
}

static void EnsureMoERebalanceThread(densecore::CpuBackend* backend) {
    if (!backend || backend->IsRebalanceThreadRunning()) {
        return;
    }
    backend->StartRebalanceThread(GetMoERebalanceIntervalMs(), GetMoERebalanceTopK(), IsMoEPageMigrationEnabled());
}

static void UpdateSchedulerExperts(const MoEUserData* ud, const densecore::moe::MoERouteResult& routing) {
    if (!ud || !ud->scheduler || !ud->batch) {
        return;
    }
    const BatchSpec* batch = ud->batch;
    if (batch->seq_id.empty() || batch->scheduler_seq_ids.empty()) {
        return;
    }

    const int num_seqs = static_cast<int>(batch->scheduler_seq_ids.size());
    std::vector<std::vector<int>> per_seq_experts(static_cast<size_t>(num_seqs));

    const bool has_token_indices = !routing.token_indices.empty();
    const int total_assignments = static_cast<int>(routing.expert_ids.size());
    for (int i = 0; i < total_assignments; ++i) {
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= static_cast<int>(batch->seq_id.size())) {
            continue;
        }
        const int seq_idx = batch->seq_id[token_idx];
        if (seq_idx < 0 || seq_idx >= num_seqs) {
            continue;
        }
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0) {
            continue;
        }
        per_seq_experts[static_cast<size_t>(seq_idx)].push_back(expert_id);
    }

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
        auto& experts = per_seq_experts[static_cast<size_t>(seq_idx)];
        if (experts.empty()) {
            continue;
        }
        const int sched_seq_id = batch->scheduler_seq_ids[static_cast<size_t>(seq_idx)];
        if (sched_seq_id < 0) {
            continue;
        }

        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        ud->scheduler->SetPredictedExperts(sched_seq_id, experts);
    }
}

static densecore::moe::MoERouteResult RouteMoEGroupedSigmoid(const struct ggml_tensor* gate_logits,
                                                             const MoEUserData* ud) {
    densecore::moe::MoERouteResult routing;
    if (!gate_logits || !ud || !ud->model) {
        return routing;
    }

    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    routing.batch_size = batch_size;
    routing.top_k = top_k;
    routing.expert_ids.assign(static_cast<size_t>(batch_size * top_k), -1);
    routing.weights.assign(static_cast<size_t>(batch_size * top_k), 0.0f);
    routing.token_indices.assign(static_cast<size_t>(batch_size * top_k), 0);

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return routing;
    }

    const struct ggml_tensor* bias_t = ud->layer ? ud->layer->Get(model_keys::kMoeCorrectionBias) : nullptr;
    const float* correction_bias =
        (bias_t && bias_t->type == GGML_TYPE_F32) ? reinterpret_cast<const float*>(bias_t->data) : nullptr;

    const int n_group = std::max(1, ud->model->moe_n_group);
    const int group_size = std::max(1, n_experts / n_group);
    const int topk_group = std::max(1, std::min(ud->model->moe_topk_group, n_group));

    std::vector<float> probs(static_cast<size_t>(n_experts), 0.0f);
    std::vector<float> choice_scores(static_cast<size_t>(n_experts), 0.0f);
    std::vector<float> group_scores(static_cast<size_t>(n_group), -std::numeric_limits<float>::infinity());
    std::vector<int> active_groups(static_cast<size_t>(n_group), 0);
    std::vector<int> selected(static_cast<size_t>(top_k), -1);
    const auto stable_sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };

    for (int token_idx = 0; token_idx < batch_size; ++token_idx) {
        const float* row = logits + static_cast<size_t>(token_idx) * n_experts;
        for (int e = 0; e < n_experts; ++e) {
            const float p = stable_sigmoid(row[e]);
            probs[static_cast<size_t>(e)] = p;
            choice_scores[static_cast<size_t>(e)] = p + (correction_bias ? correction_bias[e] : 0.0f);
        }

        for (int g = 0; g < n_group; ++g) {
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            float best = -std::numeric_limits<float>::infinity();
            float second = -std::numeric_limits<float>::infinity();
            for (int e = begin; e < end; ++e) {
                const float score = choice_scores[static_cast<size_t>(e)];
                if (score > best) {
                    second = best;
                    best = score;
                } else if (score > second) {
                    second = score;
                }
            }
            group_scores[static_cast<size_t>(g)] = best + ((end - begin) > 1 ? second : 0.0f);
        }

        std::iota(active_groups.begin(), active_groups.end(), 0);
        std::partial_sort(
            active_groups.begin(), active_groups.begin() + topk_group, active_groups.end(),
            [&](int a, int b) { return group_scores[static_cast<size_t>(a)] > group_scores[static_cast<size_t>(b)]; });

        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(static_cast<size_t>(topk_group * group_size));
        for (int group_rank = 0; group_rank < topk_group; ++group_rank) {
            const int g = active_groups[static_cast<size_t>(group_rank)];
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            for (int e = begin; e < end; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }
        if (static_cast<int>(candidates.size()) < top_k) {
            candidates.clear();
            candidates.reserve(static_cast<size_t>(n_experts));
            for (int e = 0; e < n_experts; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }

        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int expert_id = candidates[static_cast<size_t>(k)].second;
            selected[static_cast<size_t>(k)] = expert_id;
            weight_sum += probs[static_cast<size_t>(expert_id)];
        }

        const float scale = ud->model->moe_routed_scaling_factor;
        for (int k = 0; k < top_k; ++k) {
            const int expert_id = selected[static_cast<size_t>(k)];
            float weight = probs[static_cast<size_t>(expert_id)];
            if (ud->model->moe_norm_topk_prob && weight_sum > 1e-20f) {
                weight /= weight_sum;
            }
            weight *= scale;
            routing.expert_ids[static_cast<size_t>(token_idx * top_k + k)] = expert_id;
            routing.weights[static_cast<size_t>(token_idx * top_k + k)] = weight;
            routing.token_indices[static_cast<size_t>(token_idx * top_k + k)] = token_idx;
        }
    }

    return routing;
}

void cb_moe_forward(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                    int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<MoEUserData*>(userdata);
    if (!ud || !ud->layer || !ud->backend) return;

    std::vector<densecore::CpuBackend::ExpertWeights> experts = ud->backend->GetRegisteredExperts(ud->layer);
    if (experts.empty()) {
        experts = BuildExpertWeights(ud->layer);
        ud->backend->InitMoEProfiler(ud->layer, static_cast<int>(experts.size()));
        ud->backend->RegisterMoEExperts(ud->layer, experts);
        EnsureMoERebalanceThread(ud->backend);
    }

    // Routing (src1 = gate_logits)
    densecore::moe::MoERouteResult routing;
    if (ud->model && ud->model->arch_flags.is_glm_moe) {
        routing = RouteMoEGroupedSigmoid(src1, ud);
    } else {
        densecore::Tensor t_gate_logits = GgmlToTensor(src1);
        routing = densecore::moe::MoETopKRoute(t_gate_logits, ud->k);
    }
    UpdateSchedulerExperts(ud, routing);

    // Forward (src0 = input, dst = output)
    densecore::Tensor t_input = GgmlToTensor(src0);
    densecore::Tensor t_output = GgmlToTensor(dst);

    ud->backend->ForwardMoE(ud->layer, t_input, routing, experts, &t_output);
}

// ============================================================================
// SSM (Mamba2) Callback Functions
// ============================================================================

// Conv1D decode callback: processes N tokens sequentially through the ring buffer
static void cb_ssm_conv1d(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<SSMConv1DUserData*>(userdata);
    const float* input = reinterpret_cast<const float*>(src->data);
    float* output = reinterpret_cast<float*>(dst->data);
    if (!ud || !input || !output) return;

    static const bool ssm_passthrough = (std::getenv("SSM_PASSTHROUGH") != nullptr);
    const int N = static_cast<int>(src->ne[1]);
    if (ssm_passthrough) {
        // Pass-through: copy input → output, skip conv and state update
        std::memcpy(output, input, static_cast<size_t>(N) * ud->channels * sizeof(float));
        return;
    }
    for (int t = 0; t < N; ++t) {
        densecore::hwy_kernels::SSMConv1DDecode_Hwy(ud->conv_state, &input[t * ud->channels], ud->weight,
                                                    &output[t * ud->channels], ud->channels, ud->kernel_size);
    }
}

static inline float SoftplusStable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

static inline float SigmoidStable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

// Qwen3.5 recurrent delta-net callback.
//
// Inputs:
//   a = convolved qkv_mixed after SiLU [conv_channels, N]
//   b = z projection before SiLU       [d_inner, N]
//   c = normalized layer input         [n_embd, N]
//
// Output:
//   dst = scratch buffer whose first d_inner rows contain the recurrent
//         linear-attention output after gated RMSNorm
static void cb_ssm_qwen35_delta(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                                const struct ggml_tensor* c, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    const float* qkv_conv = reinterpret_cast<const float*>(a->data);
    const float* z_proj = reinterpret_cast<const float*>(b->data);
    const float* input = reinterpret_cast<const float*>(c->data);
    float* y_out = reinterpret_cast<float*>(dst->data);
    if (!ud || !qkv_conv || !z_proj || !input || !y_out) return;

    static const bool ssm_passthrough = (std::getenv("SSM_PASSTHROUGH") != nullptr);
    const int N = static_cast<int>(a->ne[1]);
    const int out_elems = static_cast<int>(dst->ne[0]) * N;
    const ptrdiff_t qkv_stride = static_cast<ptrdiff_t>(a->nb[1] / sizeof(float));
    const ptrdiff_t z_stride = static_cast<ptrdiff_t>(b->nb[1] / sizeof(float));
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(c->nb[1] / sizeof(float));
    const ptrdiff_t out_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    if (ssm_passthrough) {
        std::memset(y_out, 0, static_cast<size_t>(out_elems) * sizeof(float));
        return;
    }

    static const bool ssm_debug = (std::getenv("SSM_DEBUG") != nullptr);
    const int num_k_heads = ud->n_groups;
    const int num_v_heads = ud->n_heads;
    const int head_k_dim = ud->head_dim_k;
    const int head_v_dim = ud->head_dim_v;
    const int qk_total = head_k_dim * num_k_heads;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_v_dim));
    const int state_stride = head_k_dim * head_v_dim;

    std::vector<float> q_norm(static_cast<size_t>(head_k_dim));
    std::vector<float> k_norm(static_cast<size_t>(head_k_dim));
    std::vector<float> kv_mem(static_cast<size_t>(head_v_dim));
    std::vector<float> delta(static_cast<size_t>(head_v_dim));
    std::vector<float> y(static_cast<size_t>(ud->d_inner));

    for (int t = 0; t < N; ++t) {
        const float* input_t = input + static_cast<ptrdiff_t>(t) * input_stride;
        const float* qkv_t = qkv_conv + static_cast<ptrdiff_t>(t) * qkv_stride;
        const float* z_t = z_proj + static_cast<ptrdiff_t>(t) * z_stride;
        const float* q_base = qkv_t;
        const float* k_base = qkv_t + qk_total;
        const float* v_base = qkv_t + 2 * qk_total;

        for (int h = 0; h < num_v_heads; ++h) {
            const int src_k_head = (num_k_heads == num_v_heads) ? h : (h % num_k_heads);
            const float* q_head = q_base + src_k_head * head_k_dim;
            const float* k_head = k_base + src_k_head * head_k_dim;
            const float* v_head = v_base + h * head_v_dim;
            const float* alpha_row = ud->alpha_weight + static_cast<size_t>(h) * ud->n_embd;
            const float* beta_row = ud->beta_weight + static_cast<size_t>(h) * ud->n_embd;
            float* state = ud->ssm_state + static_cast<size_t>(h) * state_stride;
            float* y_head = y.data() + static_cast<size_t>(h) * head_v_dim;

            float alpha = ud->dt_bias[h];
            float beta = 0.0f;
            for (int i = 0; i < ud->n_embd; ++i) {
                alpha += alpha_row[i] * input_t[i];
                beta += beta_row[i] * input_t[i];
            }
            const float decay_gate = SoftplusStable(alpha) * ud->ssm_a[h];
            const float beta_gate = SigmoidStable(beta);
            const float decay = std::exp(decay_gate);

            float q_sum_sq = 0.0f;
            float k_sum_sq = 0.0f;
            for (int i = 0; i < head_k_dim; ++i) {
                q_sum_sq += q_head[i] * q_head[i];
                k_sum_sq += k_head[i] * k_head[i];
            }
            const float q_inv_norm = q_scale / std::sqrt(q_sum_sq + ud->norm_eps);
            const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + ud->norm_eps);
            for (int i = 0; i < head_k_dim; ++i) {
                q_norm[i] = q_head[i] * q_inv_norm;
                k_norm[i] = k_head[i] * k_inv_norm;
            }

            for (int idx = 0; idx < state_stride; ++idx) {
                state[idx] *= decay;
            }

            for (int j = 0; j < head_v_dim; ++j) {
                float sum = 0.0f;
                const float* state_row = state + j * head_k_dim;
                for (int i = 0; i < head_k_dim; ++i) {
                    sum += state_row[i] * k_norm[i];
                }
                kv_mem[j] = sum;
            }

            for (int j = 0; j < head_v_dim; ++j) {
                delta[j] = (v_head[j] - kv_mem[j]) * beta_gate;
            }

            for (int j = 0; j < head_v_dim; ++j) {
                float* state_row = state + j * head_k_dim;
                const float dj = delta[j];
                for (int i = 0; i < head_k_dim; ++i) {
                    state_row[i] += dj * k_norm[i];
                }
            }

            for (int j = 0; j < head_v_dim; ++j) {
                float sum = 0.0f;
                const float* state_row = state + j * head_k_dim;
                for (int i = 0; i < head_k_dim; ++i) {
                    sum += state_row[i] * q_norm[i];
                }
                y_head[j] = sum;
            }

            float sum_sq = 0.0f;
            for (int j = 0; j < head_v_dim; ++j) {
                sum_sq += y_head[j] * y_head[j];
            }
            const float rms = std::sqrt(sum_sq / head_v_dim + ud->norm_eps);
            for (int j = 0; j < head_v_dim; ++j) {
                const float z_val = z_t[h * head_v_dim + j];
                const float gated = z_val / (1.0f + std::exp(-z_val));
                y_head[j] = (y_head[j] / rms) * ud->norm_weight[j] * gated;
            }

            if (ssm_debug && t == 0 && h == 0) {
                fprintf(stderr,
                        "[QWEN35_SSM] alpha=%.4f decay=%.4f beta=%.4f q0=%.4f k0=%.4f v0=%.4f z0=%.4f y0=%.4f\n", alpha,
                        decay, beta_gate, q_norm[0], k_norm[0], v_head[0], z_t[0], y_head[0]);
            }
        }

        std::memcpy(y_out + static_cast<ptrdiff_t>(t) * out_stride, y.data(),
                    static_cast<size_t>(ud->d_inner) * sizeof(float));
    }
}

struct ggml_tensor* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache, struct ggml_context* ctx_c,
                                          const BatchSpec& batch, bool embedding_mode, struct ggml_cgraph* gf,
                                          struct ggml_tensor** out_embd, struct ggml_tensor** out_pos) {
    // =========================================================================
    // STRATEGY PATTERN DISPATCH
    // =========================================================================
    // Try to use a registered GraphBuilder for this architecture.
    // This allows seamless support for ViT, MoE, and other variants without
    // cluttering the main inference loop.
    // =========================================================================
    {
        auto builder = densecore::TransformerGraphRegistry::Instance().GetBuilder(static_cast<int>(model->arch));

        if (builder) {
            if (IsVerboseGraphBuildLoggingEnabled()) {
                std::cerr << "[BuildTransformerGraph] Using strategy: " << builder->Name() << std::endl;
            }
            return builder->Build(model, cache, ctx_c, batch, embedding_mode, gf, out_embd, out_pos);
        }
    }

    // Fallback: Inline LLaMA/Default implementation
    if (IsVerboseGraphBuildLoggingEnabled()) {
        std::cerr << "[BuildTransformerGraph] No builder found for arch " << (int)model->arch
                  << ", using default inline LLaMA logic." << std::endl;
    }


    // ENSURE: ctx_c must be initialized with sufficient memory (e.g. 128MB+)
    // to hold the compute graph nodes, especially for deep models like Qwen.
    // This initialization happens in worker.cpp (InitGraphCache or temp
    // context).

    // N is batch size
    const int N = batch.tokens.size();
    const int n_embd = model->hparams.n_embd;
    const int n_head = model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    const int n_layer = model->hparams.n_layer;
    const int n_ctx = model->hparams.n_ctx;
    const DecodePagedAttentionPolicy decode_paged_policy = LoadDecodePagedAttentionPolicy();
    (void)n_embd;
    (void)n_head_kv;
    (void)n_ctx;

    // =========================================================================
    // 1. Token Embedding Lookup
    // =========================================================================
    struct ggml_tensor* embd_inp = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
    ggml_set_name(embd_inp, "embd_inp");
    if (embd_inp->data) {
        memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
    }
    if (out_embd) *out_embd = embd_inp;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, embd_inp);

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check embedding output
        auto cb_check_embd = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* ud) {
            if (ith != 0) return;
            static int cb_ct = 0;
            const bool emit = (cb_ct < 5);
            if (emit && src && src->data) {
                const float* d = (const float*)src->data;
                int n = ggml_nelements(src);
                int zero_ct = 0;
                float mn = d[0], mx = d[0];
                for (int i = 0; i < n; i++) {
                    if (d[i] == 0.0f)
                        zero_ct++;
                    else {
                        if (d[i] < mn) mn = d[i];
                        if (d[i] > mx) mx = d[i];
                    }
                }
                fprintf(stderr, "[EMBED #%d] shape=[%ld,%ld] total=%d zero=%d min=%.6f max=%.6f\n", cb_ct,
                        (long)src->ne[0], (long)src->ne[1], n, zero_ct, mn, mx);
                cb_ct++;
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_embd, 1, nullptr);
    }

    // Position tensor for RoPE
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
    ggml_set_name(pos, "pos");
    if (pos->data) {
        memcpy(pos->data, batch.pos.data(), N * sizeof(int));
    }
    if (out_pos) *out_pos = pos;

    const bool decode_only_batch_layout = IsDecodeOnlyBatchLayout(batch, N);

    auto apply_weighted_rms_norm = [&](struct ggml_tensor * src, struct ggml_tensor * norm_weight,
                                       const char* debug_name) -> struct ggml_tensor* {
        if (!src || !norm_weight) {
            return src;
        }

        const bool force_cpu_norm =
            IsMixedRoutingEnabled(&batch) && ResolvePreferredNormDevice(&batch) == densecore::DeviceType::CPU &&
            ResolvePreferredDevice(&batch) != densecore::DeviceType::CPU && src->type == GGML_TYPE_F32 &&
            norm_weight->type == GGML_TYPE_F32 && src->nb[0] == static_cast<int64_t>(sizeof(float));

        if (!force_cpu_norm) {
            struct ggml_tensor* out = ggml_rms_norm(ctx_c, src, model->hparams.f_norm_rms_eps);
            out = ggml_mul(ctx_c, out, norm_weight);
            if (debug_name) {
                ggml_set_name(out, debug_name);
            }
            return out;
        }

        AddRMSNormUserData* ud = GetAddRMSNormUserData();
        ud->residual = nullptr;
        ud->rms_weight = reinterpret_cast<const float*>(norm_weight->data);
        ud->n_embd = static_cast<int>(src->ne[0]);
        ud->n_tokens = static_cast<int>(src->ne[1]);
        ud->eps = model->hparams.f_norm_rms_eps;
        const int n_tasks = ResolveTaskCount(&batch, std::max(1, ud->n_tokens));
        struct ggml_tensor* out = ggml_map_custom1(ctx_c, src, cb_residual_rmsnorm_fused, n_tasks, ud);
        if (debug_name) {
            ggml_set_name(out, debug_name);
        }
        return out;
    };

    // Reuse causal mask tensor across layers for the same forward pass.
    // Prefill builds N>1 attention repeatedly; hoisting avoids O(L * N * K)
    // mask materialization and graph-build overhead.
    struct ggml_tensor* shared_prefill_flash_mask = nullptr;
    int shared_prefill_mask_n_total = -1;
    int shared_prefill_mask_n_padded = -1;
    int shared_prefill_mask_n = -1;
    int shared_prefill_mask_n_past = -1;
    if (N > 1 && !decode_only_batch_layout) {
        int prefill_n_past = 0;
        if (cache && batch.num_seqs > 0 && !batch.n_past.empty()) {
            prefill_n_past = *std::max_element(batch.n_past.begin(), batch.n_past.end());
        }
        const int prefill_n_total = prefill_n_past + N;
        const int prefill_n_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);

        shared_prefill_flash_mask = ggml_new_tensor_4d(ctx_c, GGML_TYPE_F32, prefill_n_total, prefill_n_padded, 1, 1);
        float* mask_data = reinterpret_cast<float*>(shared_prefill_flash_mask->data);
        for (int q = 0; q < prefill_n_padded; ++q) {
            for (int k = 0; k < prefill_n_total; ++k) {
                const int query_pos = prefill_n_past + q;
                const int key_pos = k;
                const int idx = k + q * prefill_n_total;
                mask_data[idx] = (q < N && key_pos <= query_pos) ? 0.0f : -INFINITY;
            }
        }

        shared_prefill_mask_n_total = prefill_n_total;
        shared_prefill_mask_n_padded = prefill_n_padded;
        shared_prefill_mask_n = N;
        shared_prefill_mask_n_past = prefill_n_past;
    }

    // =========================================================================
    // 2. Transformer Layers
    // =========================================================================
    int ssm_ordinal_counter = 0;  // Counts SSM layers for state indexing
    for (int il = 0; il < n_layer; ++il) {
        auto& layer = model->layers[il];
        auto* attn_norm = layer.Get(model_keys::kAttnNorm);
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        auto* wv = layer.Get(model_keys::kAttnVWeight);
        auto* wo = layer.Get(model_keys::kAttnOWeight);
        auto* q_a = layer.Get(model_keys::kAttnQAProj);
        auto* q_a_norm = layer.Get(model_keys::kAttnQANorm);
        auto* q_b = layer.Get(model_keys::kAttnQBProj);
        auto* kv_a = layer.Get(model_keys::kAttnKvAProj);
        auto* kv_a_norm = layer.Get(model_keys::kAttnKvANorm);
        auto* kv_b = layer.Get(model_keys::kAttnKvBProj);
        auto* indexer_wq_b = layer.Get(model_keys::kIndexerWqB);
        auto* indexer_wk = layer.Get(model_keys::kIndexerWk);
        auto* indexer_k_norm = layer.Get(model_keys::kIndexerKNorm);
        auto* indexer_weights_proj = layer.Get(model_keys::kIndexerWeightsProj);
        auto* bq = layer.Get(model_keys::kAttnQBias);
        auto* bk = layer.Get(model_keys::kAttnKBias);
        auto* bv = layer.Get(model_keys::kAttnVBias);
        auto* bo = layer.Get(model_keys::kAttnOBias);
        auto* q_norm = layer.Get(model_keys::kAttnQNorm);
        auto* k_norm = layer.Get(model_keys::kAttnKNorm);
        auto* ffn_norm = layer.Get(model_keys::kFfnNorm);
        auto* ffn_gate = layer.Get(model_keys::kFfnGate);
        auto* ffn_up = layer.Get(model_keys::kFfnUp);
        auto* ffn_down = layer.Get(model_keys::kFfnDown);
        auto* moe_gate = layer.Get(model_keys::kMoeGate);

        struct ggml_tensor* inpL = cur;

        // Attention Norm
        if (!attn_norm) {
            throw densecore::InvalidArgumentException("Missing attention_norm weight in TransformerLayer");
        }
        cur = apply_weighted_rms_norm(cur, attn_norm, "attn_norm");

        // SSM / Attention layer dispatch
        const bool is_ssm_layer = model->arch_flags.is_hybrid_ssm &&
                                  (il % model->ssm_full_attn_interval != model->ssm_full_attn_interval - 1);
        struct ggml_tensor* attn_out = nullptr;
        struct ggml_tensor* attn_post_residual = nullptr;

        if (is_ssm_layer) {
            // =================================================================
            // SSM/Mamba2 Layer Forward Path
            // =================================================================
            auto* attn_qkv = layer.Get(model_keys::kAttnQkvWeight);
            auto* ssm_conv1d_w = layer.Get(model_keys::kSSMConv1d);
            auto* ssm_a = layer.Get(model_keys::kSSMA);
            auto* ssm_alpha_w = layer.Get(model_keys::kSSMAlpha);
            auto* ssm_beta_w = layer.Get(model_keys::kSSMBeta);
            auto* ssm_dt_bias_t = layer.Get(model_keys::kSSMDtBias);
            auto* ssm_norm_w = layer.Get(model_keys::kSSMNorm);
            auto* attn_gate_w = layer.Get(model_keys::kAttnGate);
            auto* ssm_out_w = layer.Get(model_keys::kSSMOut);

            if (!attn_qkv || !ssm_conv1d_w || !ssm_a || !ssm_alpha_w || !ssm_beta_w || !ssm_dt_bias_t || !ssm_norm_w ||
                !attn_gate_w || !ssm_out_w) {
                throw densecore::InvalidArgumentException("Missing SSM weights in layer " + std::to_string(il));
            }

            const int ssm_ordinal = ssm_ordinal_counter++;
            auto& ssm_rt = model->ssm_layer_states[ssm_ordinal];

            const int d_inner = model->ssm_inner_size;
            const int num_v_heads = model->ssm_time_step_rank;
            const int head_dim_v = d_inner / num_v_heads;
            const int head_dim_k = model->ssm_state_size;
            const int n_groups = model->ssm_group_count;
            const int conv_channels = d_inner + 2 * n_groups * head_dim_k;
            const int conv_kernel = model->ssm_conv_kernel;

            // 1. qkv_mixed projection: normed input [n_embd, N] → [conv_channels, N]
            struct ggml_tensor* qkv_mixed = smart_mul_mat(ctx_c, attn_qkv, cur, model);

            // 2. Conv1D: updates conv_state ring buffer, outputs [conv_channels, N]
            SSMConv1DUserData* conv_ud = GetSSMConv1DUserData();
            conv_ud->conv_state = ssm_rt.conv_state.data();
            conv_ud->weight = !ssm_rt.conv1d_f32.empty() ? ssm_rt.conv1d_f32.data()
                                                         : reinterpret_cast<const float*>(ssm_conv1d_w->data);
            conv_ud->channels = conv_channels;
            conv_ud->kernel_size = conv_kernel;
            struct ggml_tensor* qkv_conv = ggml_map_custom1(ctx_c, qkv_mixed, cb_ssm_conv1d, 1, conv_ud);
            qkv_conv = ggml_silu(ctx_c, qkv_conv);

            // 3. z projection and recurrent Qwen3.5 delta-net block.
            struct ggml_tensor* z = smart_mul_mat(ctx_c, attn_gate_w, cur, model);
            SSMQwen35DeltaUserData* scan_ud = GetSSMQwen35DeltaUserData();
            scan_ud->alpha_weight = ssm_rt.alpha_f32.data();
            scan_ud->beta_weight = ssm_rt.beta_f32.data();
            scan_ud->dt_bias = !ssm_rt.dt_bias_f32.empty() ? ssm_rt.dt_bias_f32.data()
                                                           : reinterpret_cast<const float*>(ssm_dt_bias_t->data);
            scan_ud->ssm_a =
                !ssm_rt.ssm_a_f32.empty() ? ssm_rt.ssm_a_f32.data() : reinterpret_cast<const float*>(ssm_a->data);
            scan_ud->norm_weight =
                !ssm_rt.norm_f32.empty() ? ssm_rt.norm_f32.data() : reinterpret_cast<const float*>(ssm_norm_w->data);
            scan_ud->ssm_state = ssm_rt.ssm_state.data();
            scan_ud->n_embd = n_embd;
            scan_ud->d_inner = d_inner;
            scan_ud->n_heads = num_v_heads;
            scan_ud->head_dim_v = head_dim_v;
            scan_ud->head_dim_k = head_dim_k;
            scan_ud->n_groups = n_groups;
            scan_ud->norm_eps = model->hparams.f_norm_rms_eps;
            struct ggml_tensor* y_scratch = ggml_map_custom3(ctx_c, qkv_conv, z, cur, cb_ssm_qwen35_delta, 1, scan_ud);
            struct ggml_tensor* y = ggml_cont(ctx_c, ggml_view_2d(ctx_c, y_scratch, d_inner, N, y_scratch->nb[1], 0));

            // 4. Output projection: [d_inner, N] → [n_embd, N]
            cur = smart_mul_mat(ctx_c, ssm_out_w, y, model);

            // Residual connection
            attn_out = cur;
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
        } else {
            // =================================================================
            // Attention Layer Forward Path
            // =================================================================

            // Q/K/V Projections (using smart dispatcher for Parallel GEMV)
            const bool use_glm_dsa_mla =
                model->arch_flags.is_glm_dsa && q_a && q_a_norm && q_b && kv_a && kv_a_norm && kv_b;
            if (!use_glm_dsa_mla && (!wq || !wk || !wv)) {
                throw densecore::InvalidArgumentException("Missing Q/K/V weights in TransformerLayer");
            }
            struct ggml_tensor* Qcur = nullptr;
            struct ggml_tensor* Kcur = nullptr;
            struct ggml_tensor* Vcur = nullptr;
            struct ggml_tensor* glm_q_resid = nullptr;
            struct ggml_tensor* glm_index_query = nullptr;
            struct ggml_tensor* glm_index_weights = nullptr;
            struct ggml_tensor* glm_index_key = nullptr;
            const bool use_glm_dsa_sparse = use_glm_dsa_mla && cache && cache->has_index_cache &&
                                            cache->index_head_dim == model->glm_index_head_dim &&
                                            decode_only_batch_layout && indexer_wq_b && indexer_wk && indexer_k_norm &&
                                            indexer_weights_proj && model->glm_index_n_heads > 0 &&
                                            model->glm_index_head_dim > 0;

            // Optional fused QKV projection (single pass over input per token).
            // Falls back to per-projection matmul for non-F32/quantized weights.
            const bool fused_qkv_supported = !use_glm_dsa_mla && IsFusedQKVEnabled() && cur->type == GGML_TYPE_F32 &&
                                             wq->type == GGML_TYPE_F32 && wk->type == GGML_TYPE_F32 &&
                                             wv->type == GGML_TYPE_F32 && wq->data && wk->data && wv->data &&
                                             wq->ne[0] == cur->ne[0] && wk->ne[0] == cur->ne[0] &&
                                             wv->ne[0] == cur->ne[0] && wq->ne[1] > 0 && wk->ne[1] > 0 && wv->ne[1] > 0;

            if (use_glm_dsa_mla) {
                static bool logged_glm_dsa_path = false;
                if (!logged_glm_dsa_path) {
                    std::cerr << "[DenseCore] GLM-5 DSA path enabled"
                              << (use_glm_dsa_sparse ? " with sparse indexer cache."
                                                     : " without sparse indexer cache; using dense attention.")
                              << std::endl;
                    logged_glm_dsa_path = true;
                }
                const int qk_nope_head_dim = model->glm_qk_nope_head_dim;
                const int qk_rope_head_dim = model->glm_qk_rope_head_dim;
                const int v_head_dim = model->glm_v_head_dim;
                const int kv_lora_rank = model->glm_kv_lora_rank;
                const int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
                const int kv_proj_head_dim = qk_nope_head_dim + v_head_dim;

                if (qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 || v_head_dim <= 0 || kv_lora_rank <= 0) {
                    throw densecore::InvalidArgumentException("Incomplete GLM-5 DSA metadata for MLA projection path");
                }

                glm_q_resid = smart_mul_mat(ctx_c, q_a, cur, model);
                glm_q_resid = apply_weighted_rms_norm(glm_q_resid, q_a_norm, "glm_q_a_norm");
                struct ggml_tensor* q_raw = smart_mul_mat(ctx_c, q_b, glm_q_resid, model);

                struct ggml_tensor* kv_a_cur = smart_mul_mat(ctx_c, kv_a, cur, model);
                const int64_t kv_a_dim = kv_a_cur->ne[0];
                if (kv_a_dim < static_cast<int64_t>(kv_lora_rank + qk_rope_head_dim)) {
                    throw densecore::InvalidArgumentException(
                        "GLM-5 kv_a projection is smaller than kv_lora_rank + rope dim");
                }

                struct ggml_tensor* kv_comp = ggml_view_2d(ctx_c, kv_a_cur, kv_lora_rank, N, kv_a_cur->nb[1], 0);
                struct ggml_tensor* k_rope =
                    ggml_view_2d(ctx_c, kv_a_cur, qk_rope_head_dim, N, kv_a_cur->nb[1],
                                 static_cast<size_t>(kv_lora_rank) * ggml_element_size(kv_a_cur));
                kv_comp = ggml_cont(ctx_c, kv_comp);
                k_rope = ggml_cont(ctx_c, k_rope);
                kv_comp = apply_weighted_rms_norm(kv_comp, kv_a_norm, "glm_kv_a_norm");
                struct ggml_tensor* kv_raw = smart_mul_mat(ctx_c, kv_b, kv_comp, model);

                if (q_raw->ne[0] != static_cast<int64_t>(q_head_dim * n_head)) {
                    throw densecore::InvalidArgumentException("GLM-5 q_b projection shape mismatch");
                }
                if (kv_raw->ne[0] != static_cast<int64_t>(kv_proj_head_dim * n_head_kv)) {
                    throw densecore::InvalidArgumentException("GLM-5 kv_b projection shape mismatch");
                }

                GLMDSAPackUserData* q_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* k_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* v_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                if (!q_pack_ud || !k_pack_ud || !v_pack_ud) {
                    throw densecore::OutOfMemoryException("Failed to allocate GLM-5 DSA packing userdata");
                }
                *q_pack_ud = {n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *k_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *v_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};

                struct ggml_tensor* q_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head, N);
                struct ggml_tensor* k_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head_kv, N);
                struct ggml_tensor* v_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, v_head_dim * n_head_kv, N);

                // Repack [nope|rope] into [rope|nope] so the existing partial-RoPE path
                // can operate on the leading rope dimensions.
                Qcur = ggml_map_custom2(ctx_c, q_packed, q_raw, cb_pack_glm_dsa_q, 1, q_pack_ud);
                Kcur = ggml_map_custom3(ctx_c, k_packed, kv_raw, k_rope, cb_pack_glm_dsa_k, 1, k_pack_ud);
                Vcur = ggml_map_custom2(ctx_c, v_packed, kv_raw, cb_pack_glm_dsa_v, 1, v_pack_ud);

                if (use_glm_dsa_sparse) {
                    glm_index_query = smart_mul_mat(ctx_c, indexer_wq_b, glm_q_resid, model);
                    glm_index_weights = smart_mul_mat(ctx_c, indexer_weights_proj, cur, model);
                    glm_index_weights =
                        ggml_scale(ctx_c, glm_index_weights, 1.0f / std::sqrt((float)model->glm_index_n_heads));
                    glm_index_key = smart_mul_mat(ctx_c, indexer_wk, cur, model);
                    glm_index_key = apply_weighted_rms_norm(glm_index_key, indexer_k_norm, "glm_index_k_norm");
                }
            } else if (fused_qkv_supported) {
                const int dim_q_fused = static_cast<int>(wq->ne[1]);
                const int dim_k_fused = static_cast<int>(wk->ne[1]);
                const int dim_v_fused = static_cast<int>(wv->ne[1]);
                const int n_embd_fused = static_cast<int>(cur->ne[0]);
                const int merged_dim = dim_q_fused + dim_k_fused + dim_v_fused;

                struct ggml_tensor* qkv_input = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx_c, cur);
                struct ggml_tensor* qkv_merged = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, merged_dim, N);

                QKVUserData* qkv_ud = GetQKVUserData();
                qkv_ud->w_q = reinterpret_cast<const float*>(wq->data);
                qkv_ud->w_k = reinterpret_cast<const float*>(wk->data);
                qkv_ud->w_v = reinterpret_cast<const float*>(wv->data);
                qkv_ud->n_embd = n_embd_fused;
                qkv_ud->dim_q = dim_q_fused;
                qkv_ud->dim_k = dim_k_fused;
                qkv_ud->dim_v = dim_v_fused;

                const int n_tasks = ResolveTaskCount(&batch, std::max(1, merged_dim));
                qkv_merged = ggml_map_custom2(ctx_c, qkv_merged, qkv_input, cb_compute_qkv_map2, n_tasks, qkv_ud);

                const size_t k_offset = static_cast<size_t>(dim_q_fused) * sizeof(float);
                const size_t v_offset = static_cast<size_t>(dim_q_fused + dim_k_fused) * sizeof(float);
                Qcur = ggml_view_2d(ctx_c, qkv_merged, dim_q_fused, N, qkv_merged->nb[1], 0);
                Kcur = ggml_view_2d(ctx_c, qkv_merged, dim_k_fused, N, qkv_merged->nb[1], k_offset);
                Vcur = ggml_view_2d(ctx_c, qkv_merged, dim_v_fused, N, qkv_merged->nb[1], v_offset);
            } else {
                Qcur = smart_mul_mat(ctx_c, wq, cur, model);
                Kcur = smart_mul_mat(ctx_c, wk, cur, model);
                Vcur = smart_mul_mat(ctx_c, wv, cur, model);
            }

            // Apply Multi-LoRA
            char name_buf[64];
            auto apply_lora = [&](struct ggml_tensor* dst, struct ggml_tensor* src, const char* suffix) {
                if (batch.lora_map.empty()) {
                    return dst;
                }
                if (!ggml_is_contiguous(dst)) {
                    dst = ggml_cont(ctx_c, dst);
                }
                snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", il, suffix);
                ggml_set_name(dst, name_buf);
                return ggml_map_custom2(ctx_c, dst, src, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            };

            Qcur = apply_lora(Qcur, cur, "attn_q");
            Kcur = apply_lora(Kcur, cur, "attn_k");
            Vcur = apply_lora(Vcur, cur, "attn_v");

            // Add Bias if present (for Qwen2 and some other models)
            if (bq) {
                if (Qcur->ne[0] == bq->ne[0]) {
                    Qcur = ggml_add(ctx_c, Qcur, bq);
                } else {
                    // SKIP BIAS (Safe fallback)
                    // std::cerr << "Skipping BQ mismatch L" << il << std::endl;
                }
            }
            if (bk) {
                if (Kcur->ne[0] == bk->ne[0]) {
                    Kcur = ggml_add(ctx_c, Kcur, bk);
                } else {
                    // SKIP BIAS on mismatch to avoid graph complexity/hangs
                    // std::cerr << "Skipping BK mismatch" << std::endl;
                }
            }
            if (bv) {
                if (Vcur->ne[0] == bv->ne[0]) {
                    Vcur = ggml_add(ctx_c, Vcur, bv);
                } else {
                    // SKIP BIAS
                }
            }

            // ggml_reshape_3d requires contiguous inputs. Fused QKV views can be
            // strided, so materialize contiguous tensors before reshape.
            if (!ggml_is_contiguous(Qcur)) {
                Qcur = ggml_cont(ctx_c, Qcur);
            }
            if (!ggml_is_contiguous(Kcur)) {
                Kcur = ggml_cont(ctx_c, Kcur);
            }
            if (!ggml_is_contiguous(Vcur)) {
                Vcur = ggml_cont(ctx_c, Vcur);
            }

            // Dynamically infer head dimensions from the actual projected tensors
            int dim_q = Qcur->ne[0];
            int dim_k = Kcur->ne[0];
            int dim_v = Vcur->ne[0];

            int n_head_kv = model->hparams.n_head_kv;
            int head_dim_kv = (model->hparams.n_embd_head_k > 0) ? model->hparams.n_embd_head_k : (dim_k / n_head_kv);
            int head_dim_v = (model->hparams.n_embd_head_v > 0) ? model->hparams.n_embd_head_v : (dim_v / n_head_kv);
            struct ggml_tensor* attn_gate = nullptr;

            // Qwen3.5 hybrid attention packs [q | gate] per-head, not as two
            // contiguous global halves. Mirror llama.cpp's strided per-head views.
            if (model->arch_flags.is_hybrid_ssm && dim_q > head_dim_kv * n_head) {
                const int64_t q_attn_dim = static_cast<int64_t>(head_dim_kv) * n_head;
                const int64_t q_full_per_head = dim_q / n_head;
                const int64_t q_attn_per_head = head_dim_kv;
                if (q_full_per_head >= q_attn_per_head * 2) {
                    struct ggml_tensor* Qcur_full = Qcur;
                    const size_t elem = ggml_element_size(Qcur_full);
                    struct ggml_tensor* Qcur_head = ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N,
                                                                 elem * q_full_per_head, Qcur_full->nb[1], 0);
                    struct ggml_tensor* gate_head =
                        ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N, elem * q_full_per_head,
                                     Qcur_full->nb[1], elem * q_attn_per_head);
                    Qcur = ggml_cont_2d(ctx_c, Qcur_head, q_attn_dim, N);
                    attn_gate = ggml_cont_2d(ctx_c, gate_head, q_attn_dim, N);
                } else {
                    attn_gate = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1],
                                                              q_attn_dim * ggml_element_size(Qcur)));
                    Qcur = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1], 0));
                }
                dim_q = static_cast<int>(q_attn_dim);
            }

            int head_dim_q = dim_q / n_head;

            bool k_done = false;
            bool v_done = false;

            // Reshape Q (Standard)
            Qcur = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);

            // Reshape K/V
            if (!k_done) {
                Kcur = ggml_reshape_3d(ctx_c, Kcur, head_dim_kv, n_head_kv, N);
            }
            if (!v_done) {
                Vcur = ggml_reshape_3d(ctx_c, Vcur, head_dim_v, n_head_kv, N);
            }

            // =========================================================================
            // Per-Head QK-Norm (Architecture-flag based for Qwen3/Qwen2.5)
            // =========================================================================
            // Qwen3 requires RMS normalization applied per-head, not over the entire
            // embedding dimension. We reshape to [head_dim, n_heads * n_tokens] so that
            // ggml_rms_norm normalizes each head_dim vector independently.
            //
            // Using arch_flags for explicit requirement checking instead of implicit
            // null pointer guards. The tensor null check is kept for safety.
            //
            // Flow:
            //   1. Reshape Q from [head_dim, n_head, N] to [head_dim, n_head * N]
            //   2. Apply ggml_rms_norm (normalizes over ne[0] = head_dim)
            //   3. Multiply by weight [head_dim] (broadcasts across all head*token)
            //   4. Reshape back to [head_dim, n_head, N]
            // =========================================================================
            // Q Normalization (required for Qwen3, optional for others)
            if (model->arch_flags.requires_q_norm && q_norm) {
                const int64_t q_n_tokens = Qcur->ne[2];  // N (batch size)
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int qnorm_dbg = 0;
                    if (qnorm_dbg < 2 && q_norm->data && q_norm->type == GGML_TYPE_F32) {
                        const float* qn = reinterpret_cast<const float*>(q_norm->data);
                        fprintf(stderr, "[QNORM_L3 #%d] q_norm[0]=%.6f q_norm[1]=%.6f q_norm[2]=%.6f q_norm[3]=%.6f\n",
                                qnorm_dbg, qn[0], qn[1], qn[2], qn[3]);
                        qnorm_dbg++;
                    }
                }

                if (head_dim_q == q_norm->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head * n_tokens] for per-head norm
                    struct ggml_tensor* Q_2d = ggml_reshape_2d(ctx_c, Qcur, head_dim_q, n_head * q_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    if (Q_2d->type != GGML_TYPE_F32) {
                        fprintf(stderr, "CRITICAL: Layer %d Q_2d type is %d! Tensor name: %s\n", il, Q_2d->type,
                                Q_2d->name);
                    }
                    Q_2d = ggml_rms_norm(ctx_c, Q_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    Q_2d = ggml_mul(ctx_c, Q_2d, q_norm);

                    // Reshape back to original 3D: [head_dim, n_head, n_tokens]
                    Qcur = ggml_reshape_3d(ctx_c, Q_2d, head_dim_q, n_head, q_n_tokens);
                } else if (il == 0) {
                    static bool logged_q_mismatch = false;
                    if (!logged_q_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_q_norm dimension mismatch! "
                                  << "Qcur->ne[0]=" << Qcur->ne[0] << " vs norm->ne[0]=" << q_norm->ne[0]
                                  << ". Skipping Q normalization." << std::endl;
                        logged_q_mismatch = true;
                    }
                }
            }

            // K Normalization (required for Qwen3, optional for others)
            if (model->arch_flags.requires_k_norm && k_norm) {
                const int64_t k_n_tokens = Kcur->ne[2];  // N (batch size)
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int knorm_dbg = 0;
                    if (knorm_dbg < 2 && k_norm->data && k_norm->type == GGML_TYPE_F32) {
                        const float* kn = reinterpret_cast<const float*>(k_norm->data);
                        fprintf(stderr, "[KNORM_L3 #%d] k_norm[0]=%.6f k_norm[1]=%.6f k_norm[2]=%.6f k_norm[3]=%.6f\n",
                                knorm_dbg, kn[0], kn[1], kn[2], kn[3]);
                        knorm_dbg++;
                    }
                }

                if (head_dim_kv == k_norm->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head_kv * n_tokens] for per-head norm
                    struct ggml_tensor* K_2d = ggml_reshape_2d(ctx_c, Kcur, head_dim_kv, n_head_kv * k_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    if (K_2d->type != GGML_TYPE_F32) {
                        fprintf(stderr, "CRITICAL: Layer %d K_2d type is %d! Tensor name: %s\n", il, K_2d->type,
                                K_2d->name);
                    }
                    K_2d = ggml_rms_norm(ctx_c, K_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    K_2d = ggml_mul(ctx_c, K_2d, k_norm);

                    // Reshape back to original 3D: [head_dim, n_head_kv, n_tokens]
                    Kcur = ggml_reshape_3d(ctx_c, K_2d, head_dim_kv, n_head_kv, k_n_tokens);
                } else if (il == 0) {
                    static bool logged_k_mismatch = false;
                    if (!logged_k_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_k_norm dimension mismatch! "
                                  << "Kcur->ne[0]=" << Kcur->ne[0] << " vs norm->ne[0]=" << k_norm->ne[0]
                                  << ". Skipping K normalization." << std::endl;
                        logged_k_mismatch = true;
                    }
                }
            }

            // Apply RoPE
            // Use n_rot from model params if specified (e.g. for partial RoPE or
            // specific dim) Fallback to full head_dim_q if n_rot is 0
            int rope_dim = model->hparams.n_rot;
            if (rope_dim <= 0) {
                rope_dim = head_dim_q;
            }

            // Ensure rope_dim is valid (<= head_dim)
            if (rope_dim > head_dim_q) rope_dim = head_dim_q;

            // SKIP RoPE if we detected anomaly and sliced (k_done)
            // Layer 0 anomaly (80 dim) is unsafe for 128-dim RoPE/Kernel which expects
            // 128
            bool skip_rope = k_done;
            if (skip_rope) {
                // std::cerr << "[DenseCore] Skipping RoPE for anomalous Layer " << il <<
                // std::endl;
            }

            if (!skip_rope) {
                struct ggml_tensor* Q_rope_fast = nullptr;
                struct ggml_tensor* K_rope_fast = nullptr;
                if (IsPrecomputedRoPEEnabled()) {
                    Q_rope_fast = ggml_rope_precomputed_table(ctx_c, Qcur, pos, model, rope_dim, &batch);
                    K_rope_fast = ggml_rope_precomputed_table(ctx_c, Kcur, pos, model, rope_dim, &batch);
                }
                const bool use_mrope = model->hparams.rope_sections[0] > 0 && model->hparams.rope_sections[1] > 0;
                if (Q_rope_fast && K_rope_fast) {
                    Qcur = Q_rope_fast;
                    Kcur = K_rope_fast;
                } else {
                    if (use_mrope) {
                        int rope_sections[GGML_MROPE_SECTIONS] = {
                            model->hparams.rope_sections[0],
                            model->hparams.rope_sections[1],
                            model->hparams.rope_sections[2],
                            model->hparams.rope_sections[3],
                        };
                        Qcur = ggml_rope_multi(ctx_c, Qcur, pos, nullptr, rope_dim, rope_sections, GGML_ROPE_TYPE_MROPE,
                                               n_ctx, model->hparams.rope_freq_base, model->hparams.rope_freq_scale,
                                               0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur = ggml_rope_multi(ctx_c, Kcur, pos, nullptr, rope_dim, rope_sections, GGML_ROPE_TYPE_MROPE,
                                               n_ctx, model->hparams.rope_freq_base, model->hparams.rope_freq_scale,
                                               0.0f, 1.0f, 0.0f, 0.0f);
                    } else {
                        // Fallback to standard GGML RoPE when precomputed path is
                        // unavailable for this tensor/layout.
                        Qcur =
                            ggml_rope_ext(ctx_c, Qcur, pos, nullptr, rope_dim, 0, n_ctx, model->hparams.rope_freq_base,
                                          model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur =
                            ggml_rope_ext(ctx_c, Kcur, pos, nullptr, rope_dim, 0, n_ctx, model->hparams.rope_freq_base,
                                          model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    }
                }
            }

            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                const int index_n_heads = model->glm_index_n_heads;
                const int index_head_dim = model->glm_index_head_dim;
                const int index_rope_dim = std::min(model->glm_qk_rope_head_dim, index_head_dim);

                glm_index_query = ggml_cont(ctx_c, glm_index_query);
                glm_index_weights = ggml_cont(ctx_c, glm_index_weights);
                glm_index_key = ggml_cont(ctx_c, glm_index_key);

                glm_index_query = ggml_reshape_3d(ctx_c, glm_index_query, index_head_dim, index_n_heads, N);
                if (index_rope_dim > 0) {
                    glm_index_query = ggml_rope_ext(ctx_c, glm_index_query, pos, nullptr, index_rope_dim, 0, n_ctx,
                                                    model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                                    1.0f, 0.0f, 0.0f);
                    struct ggml_tensor* index_k_3d = ggml_reshape_3d(ctx_c, glm_index_key, index_head_dim, 1, N);
                    index_k_3d = ggml_rope_ext(ctx_c, index_k_3d, pos, nullptr, index_rope_dim, 0, n_ctx,
                                               model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                               1.0f, 0.0f, 0.0f);
                    glm_index_key = ggml_reshape_2d(ctx_c, index_k_3d, index_head_dim, N);
                }
            }

            struct ggml_tensor* KQV = nullptr;
            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                PagedAttentionUserData* ud = GetPagedAttentionUserData();
                ud->cache = cache;
                ud->layer = il;
                ud->head_dim = head_dim_q;
                ud->v_head_dim = head_dim_v;
                ud->n_head = n_head;
                ud->index_n_heads = model->glm_index_n_heads;
                ud->index_head_dim = model->glm_index_head_dim;
                ud->index_topk = model->glm_index_topk;
                ud->epoch_started.store(0, std::memory_order_relaxed);
                ud->epoch_done.store(0, std::memory_order_relaxed);
                ud->kv_writers_done.store(0, std::memory_order_relaxed);

                struct ggml_tensor* q_sparse = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
                struct ggml_tensor* k_sparse = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
                struct ggml_tensor* v_sparse = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);
                struct ggml_tensor* index_q_sparse =
                    ggml_is_contiguous(glm_index_query) ? glm_index_query : ggml_cont(ctx_c, glm_index_query);
                struct ggml_tensor* index_w_sparse =
                    ggml_is_contiguous(glm_index_weights) ? glm_index_weights : ggml_cont(ctx_c, glm_index_weights);
                struct ggml_tensor* index_k_sparse =
                    ggml_is_contiguous(glm_index_key) ? glm_index_key : ggml_cont(ctx_c, glm_index_key);

                KQV = ggml_glm_dsa_attention(ctx_c, q_sparse, k_sparse, v_sparse, index_q_sparse, index_w_sparse,
                                             index_k_sparse, ud);
            }

            // =========================================================================
            // KV CACHE INTEGRATION (Universal Paged Attention)
            // =========================================================================
            if (!KQV) {
                struct ggml_tensor* K_all = Kcur;  // Default to current K
                struct ggml_tensor* V_all = Vcur;  // Default to current V

                const bool use_cache = (cache != nullptr);
                int n_past_val = 0;
                if (use_cache && batch.num_seqs > 0 && batch.n_past.size() > 0) {
                    const KVRetentionPolicy& retention_policy = GetKVRetentionPolicy();
                    for (int n_past_i : batch.n_past) {
                        n_past_val =
                            std::max(n_past_val, ComputeKVRetentionSpan(n_past_i, retention_policy).history_kept);
                    }
                }
                const int n_total_tokens = n_past_val + N;
                const bool paged_decode_candidate =
                    !model->arch_flags.is_glm_dsa &&
                    IsPagedDecodeCandidate(cache, batch, N, n_head, n_head_kv, head_dim_q, head_dim_kv);
                const bool requested_paged_decode_attention =
                    !model->arch_flags.is_glm_dsa &&
                    ShouldUsePagedDecodeAttention(decode_paged_policy, cache, batch, N, n_head, n_head_kv, head_dim_q,
                                                  head_dim_kv);
                const bool decode_only_batch = decode_only_batch_layout;

                // Safety override: GGML's generic decode matmul path can become numerically
                // unstable for GQA decode (N=1, n_head != n_head_kv) on some CPU kernels.
                // Force the custom paged decode attention path for correctness in this case.
                const bool force_safe_gqa_decode =
                    IsForceSafeGqaDecodeEnabled() && use_cache && paged_decode_candidate && N == 1 && n_past_val > 0 &&
                    n_head_kv > 0 && n_head > n_head_kv && (n_head % n_head_kv == 0) && (head_dim_q == head_dim_kv) &&
                    batch.num_seqs == 1 && !batch.seq_id.empty();
                // For decode-only batched scheduling (N>1), force paged decode when the
                // batch layout is a valid paged candidate. The legacy non-paged batched
                // path is not sequence-isolated and can introduce cross-sequence drift.
                const bool force_batched_decode_path =
                    use_cache && decode_only_batch && N > 1 && paged_decode_candidate;

                bool use_paged_decode_attention = requested_paged_decode_attention;
                if (force_batched_decode_path) {
                    const bool was_enabled = use_paged_decode_attention;
                    use_paged_decode_attention = true;
                    if (!was_enabled) {
                        static bool logged_force_batched_decode = false;
                        if (!logged_force_batched_decode) {
                            std::cerr
                                << "[DenseCore] Forcing paged decode attention for decode-only batched scheduling "
                                << "for sequence-isolated correctness " << "(N=" << N << ")" << std::endl;
                            logged_force_batched_decode = true;
                        }
                    }
                }

                if (decode_paged_policy.debug_log && il == 0) {
                    const char* mode = "auto";
                    if (decode_paged_policy.mode == DecodePagedAttentionMode::Off) mode = "off";
                    if (decode_paged_policy.mode == DecodePagedAttentionMode::On) mode = "on";
                    const int cache_type = cache ? static_cast<int>(cache->cache_type) : -1;
                    const DecodeContextSummary context_summary = SummarizeDecodeContext(batch, N);
                    std::cerr << "[PagedDecode] mode=" << mode << " candidate=" << (paged_decode_candidate ? "1" : "0")
                              << " use=" << (use_paged_decode_attention ? "1" : "0") << " N=" << N
                              << " n_past=" << n_past_val << " n_head=" << n_head << " n_head_kv=" << n_head_kv
                              << " head_dim_q=" << head_dim_q << " cache_type=" << cache_type
                              << " ctx_min=" << (context_summary.valid ? context_summary.min_context : -1)
                              << " ctx_avg=" << (context_summary.valid ? context_summary.avg_context : -1)
                              << " ctx_max=" << (context_summary.valid ? context_summary.max_context : -1) << std::endl;
                }

                if (use_cache && !use_paged_decode_attention) {  // Re-enabled old KV cache approach
                    // Only need fancy logic if we have history.
                    // If n_past = 0 (Prefill), K_all == Kcur is mostly fine,
                    // BUT we still need to WRITE to cache.
                    // The 'ggml_pad' trick updates cache as side effect.
                    // So we act always if use_cache is true.

                    // Use ggml_pad to create a tensor of correct size (N + n_past)
                    // ggml_pad(ctx, a, pad_0, pad_1, pad_2, pad_3)
                    // We pad dimension 2 (sequence) by n_past_val.
                    // Result shape: [head_dim, n_head, N + n_past]
                    struct ggml_tensor* K_padded = Kcur;
                    struct ggml_tensor* V_padded = Vcur;

                    if (n_past_val > 0) {
                        K_padded = ggml_pad(ctx_c, Kcur, 0, 0, n_past_val, 0);
                        V_padded = ggml_pad(ctx_c, Vcur, 0, 0, n_past_val, 0);
                    }

                    KVCacheUserData* k_ud = GetKVCacheUserData(il, true);
                    *k_ud = {cache, il, head_dim_kv, true};  // batch accessed via GetCurrentBatch()
                    KVCacheUserData* v_ud = GetKVCacheUserData(il, false);
                    *v_ud = {cache, il, head_dim_v, false};  // batch accessed via GetCurrentBatch()

                    int kv_tasks = ResolveInferenceConfig(&batch).num_threads;
                    if (kv_tasks <= 0) {
                        kv_tasks = std::thread::hardware_concurrency();
                        if (kv_tasks <= 0) kv_tasks = 4;
                    }
                    int physical_cores = ResolveHardwareTopology(&batch).GetPhysicalCoreCount();
                    if (physical_cores > 0) {
                        kv_tasks = std::min(kv_tasks, physical_cores);
                    }
                    kv_tasks = std::max(1, kv_tasks);
                    kv_tasks = std::min(kv_tasks, std::max(1, n_head_kv));
                    {
                        // Escape hatch for platform-specific troubleshooting.
                        const char* env = std::getenv("DENSECORE_KV_CALLBACK_SINGLE_THREAD");
                        const bool force_single = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                        if (force_single) {
                            kv_tasks = 1;
                        }
                    }

                    K_all = ggml_map_custom1(ctx_c, K_padded, cb_kv_manage, kv_tasks, k_ud);
                    V_all = ggml_map_custom1(ctx_c, V_padded, cb_kv_manage, kv_tasks, v_ud);
                }

                // =========================================================================
                // NEW: Robust KV Cache Integration (Replaces ggml_pad approach)
                // =========================================================================
                // This approach explicitly:
                //   1. Allocates destination tensors with full size [head_dim, n_head_kv,
                //   n_total]
                //   2. Uses cb_kv_update_and_gather to write cache, read history, append
                //   current
                //   3. Does NOT rely on ggml_pad padding behavior which was causing
                //   NaN/hangs
                // =========================================================================
                if (false) {  // DISABLED: New approach has graph dependency bugs
                    // Step 1: Explicitly allocate K_all and V_all with full context size
                    // Shape: [head_dim_kv, n_head_kv, n_total_tokens]
                    struct ggml_tensor* K_all_tensor =
                        ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
                    struct ggml_tensor* V_all_tensor =
                        ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
                    ggml_set_name(K_all_tensor, "K_all");
                    ggml_set_name(V_all_tensor, "V_all");

                    // Step 2: Force Kcur/Vcur to be contiguous before passing data pointers
                    // This ensures src_data pointer is valid for memcpy in callback
                    struct ggml_tensor* Kcur_contig = ggml_cont(ctx_c, Kcur);
                    struct ggml_tensor* Vcur_contig = ggml_cont(ctx_c, Vcur);

                    // Step 3: Setup userdata for K with src_tensor pointer
                    // NOTE: src_tensor is set below after ggml_cont
                    //       The tensor pointer is stable; data is populated at graph
                    //       execution
                    KVUpdateGatherUserData* k_gather_ud = GetKVUpdateGatherUserData(il, true);
                    k_gather_ud->cache = cache;
                    k_gather_ud->batch = &batch;
                    k_gather_ud->layer = il;
                    k_gather_ud->head_dim = head_dim_kv;
                    k_gather_ud->n_head_kv = n_head_kv;
                    k_gather_ud->N = N;
                    k_gather_ud->n_past = n_past_val;
                    k_gather_ud->is_k = true;
                    k_gather_ud->src_tensor = nullptr;  // Set below

                    // Step 4: Setup userdata for V
                    KVUpdateGatherUserData* v_gather_ud = GetKVUpdateGatherUserData(il, false);
                    v_gather_ud->cache = cache;
                    v_gather_ud->batch = &batch;
                    v_gather_ud->layer = il;
                    v_gather_ud->head_dim = head_dim_kv;
                    v_gather_ud->n_head_kv = n_head_kv;
                    v_gather_ud->N = N;
                    v_gather_ud->n_past = n_past_val;
                    v_gather_ud->is_k = false;
                    v_gather_ud->src_tensor = nullptr;  // Set below

                    // Step 5: Create graph nodes that will execute the callbacks
                    // The src_data will be populated from the contiguous tensor's data
                    // pointer when the graph is executed (data is allocated by this point)
                    //
                    // WORKAROUND: ggml_map_custom1 passes its input tensor as 'src'.
                    // We need to pass BOTH the destination and source data.
                    // Solution: Store Kcur_contig as 'src' input, K_all_tensor->data is
                    // 'dst'
                    //
                    // The callback signature is: cb(dst, src, ith, nth, userdata)
                    // We set src_data = src->data in the callback if it's nullptr

                    // For K: Map from Kcur_contig, output shape matches K_all_tensor
                    // We need a custom callback wrapper that sets src_data from src tensor
                    // For now, we'll pass the contiguous tensor and handle in callback

                    // Actually, ggml_map_custom1(ctx, a, cb, n_tasks, userdata) creates:
                    //   result tensor with same shape as 'a'
                    //   callback receives: cb(result, a, ith, nth, userdata)
                    //
                    // So 'a' becomes 'src', and result is 'dst'
                    // We need result to have shape [head_dim_kv, n_head_kv, n_total_tokens]
                    // This means we should pass K_all_tensor as 'a', not Kcur!
                    //
                    // But then we need to access Kcur data via userdata.
                    // Since Kcur_contig->data is available at graph execution time,
                    // we can store its pointer now and it will be valid.

                    // CRITICAL FIX: The tensor data pointers are only valid AFTER
                    // ggml_backend allocates memory. At graph construction time, data may
                    // be nullptr. We need to access the data through the tensor pointer in
                    // the callback.

                    // Store tensor pointers in userdata (not raw data pointers)
                    // This requires modifying the struct to take ggml_tensor* instead of
                    // float* For now, we'll use a simpler workaround: pass Kcur_contig as
                    // input, create output tensor of correct size via ggml_new_tensor, then
                    // use ggml_cpy

                    // SIMPLER APPROACH: Use ggml_map_custom1 on K_all_tensor, pass Kcur as
                    // extra userdata Since Kcur_contig is built into the graph, its data
                    // pointer is stable
                    k_gather_ud->src_tensor = Kcur_contig;
                    v_gather_ud->src_tensor = Vcur_contig;

                    // Create the combined tensors via callback
                    K_all = ggml_map_custom1(ctx_c, K_all_tensor, cb_kv_update_and_gather, 1, k_gather_ud);
                    V_all = ggml_map_custom1(ctx_c, V_all_tensor, cb_kv_update_and_gather, 1, v_gather_ud);

                    // Mark as dependent on Kcur_contig and Vcur_contig for proper execution
                    // order
                    ggml_build_forward_expand(gf, Kcur_contig);
                    ggml_build_forward_expand(gf, Vcur_contig);
                }

                // After projection and reshape:

                // Q: [head_dim_q, n_head, N]
                // K_all: [head_dim_kv, n_head_kv, n_past + N]
                // V_all: [head_dim_kv, n_head_kv, n_past + N]

                // =========================================================================
                // GQA (Grouped Query Attention): LOGICAL BROADCASTING
                // =========================================================================
                // For models like Qwen3 where n_head != n_head_kv (e.g., 32 Q heads, 4 KV
                // heads):
                //
                // OLD APPROACH (REMOVED - caused segfaults and was inefficient):
                //   Used ggml_repeat to physically expand K/V from n_head_kv to n_head.
                //   This allocated 8x more memory and caused OOM/crashes.
                //
                // NEW APPROACH (Logical Broadcasting):
                //   Keep K/V at their original [head_dim, n_head_kv, seq] shape.
                //   The attention kernel computes: kv_head = query_head / (n_head /
                //   n_head_kv) This is zero-copy and memory-efficient.
                //
                // Both ggml_flash_attn_ext and our custom FlashAttentionGQA support this.
                // =========================================================================
                struct ggml_tensor* K = K_all;
                struct ggml_tensor* V = V_all;

                // Compute GQA repetition factor for attention dispatch
                const int n_rep = (n_head_kv > 0) ? (n_head / n_head_kv) : 1;
                (void)n_rep;  // Used in attention mask/kernel setup

                // =========================================================================
                // ATTENTION (llama.cpp style - corrected tensor layouts)
                // =========================================================================
                // Tensor shapes at this point:
                //   Q: [head_dim_q, n_head, N]
                //   K: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //   V: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //
                // For GQA: The attention kernel handles broadcasting internally.
                // Query heads [0, n_rep) all attend to KV head 0, etc.
                // =========================================================================

                // =========================================================================
                // ATTENTION DISPATCH (Runtime selection based on CPU capabilities)
                // - AVX-512+: Use Flash Attention (ggml_flash_attn_ext) for efficiency
                // - Other: Use standard Q*K^T -> softmax -> V for compatibility
                // =========================================================================
                // Note: Flash Attention still requires AVX-512-class x86 support for
                // correctness/perf in this path.
                const bool flash_attn_head_layout_supported =
                    !model->arch_flags.is_glm_dsa && (n_head_kv > 0) && (n_head % n_head_kv == 0) && (n_head % 8 == 0);
                const bool flash_attn_runtime_supported = flash_attn_head_layout_supported &&
                                                          densecore::OpsRegistry::IsInitialized() &&
                                                          IsFlashAttentionIsaSupported();
                const bool flash_attn_forced = IsFlashAttentionForced();
                const bool use_flash_attention = !IsFlashAttentionDisabled() && flash_attn_runtime_supported;
                const densecore::DeviceType preferred_attention_device = ResolvePreferredAttentionDevice(&batch);
                // HAL FlashAttention API currently exposes only `causal` + `n_head_kv`.
                // For decode (N == 1), no intra-query future tokens exist, so offset is
                // unnecessary even when n_past > 0. Prefill still requires zero offset.
                const bool hal_attention_offset_safe = (n_past_val == 0) || (N == 1);
                const bool use_hal_attention_dispatch = preferred_attention_device != densecore::DeviceType::CPU &&
                                                        !use_paged_decode_attention && hal_attention_offset_safe &&
                                                        !model->arch_flags.is_glm_dsa;
                const bool portable_cpu_flash_attention_supported =
                    flash_attn_head_layout_supported && densecore::OpsRegistry::IsInitialized() &&
                    (IsPortableCpuFlashAttentionEnabled() || flash_attn_forced);
                const bool prefer_portable_cpu_flash_safe_decode =
                    force_safe_gqa_decode && preferred_attention_device == densecore::DeviceType::CPU &&
                    hal_attention_offset_safe && portable_cpu_flash_attention_supported;
                if (!use_paged_decode_attention && force_safe_gqa_decode && !prefer_portable_cpu_flash_safe_decode) {
                    use_paged_decode_attention = true;
                    static bool logged_force_safe_decode = false;
                    if (!logged_force_safe_decode) {
                        std::cerr << "[DenseCore] Forcing paged decode attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_force_safe_decode = true;
                    }
                } else if (!use_paged_decode_attention && prefer_portable_cpu_flash_safe_decode) {
                    static bool logged_safe_decode_flash = false;
                    if (!logged_safe_decode_flash) {
                        std::cerr << "[DenseCore] Using portable CPU flash attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_safe_decode_flash = true;
                    }
                }
                const bool use_portable_cpu_flash_attention =
                    !IsFlashAttentionDisabled() && preferred_attention_device == densecore::DeviceType::CPU &&
                    !use_paged_decode_attention && hal_attention_offset_safe && portable_cpu_flash_attention_supported;
                if (flash_attn_forced && !flash_attn_runtime_supported && il == 0 &&
                    IsVerboseGraphBuildLoggingEnabled()) {
                    std::cerr
                        << "[DenseCore] DENSECORE_FORCE_FLASH_ATTN requested but native ggml flash is unavailable; "
                           "falling back to DenseCore portable flash attention or standard attention."
                        << std::endl;
                }

                if (decode_paged_policy.debug_log && il == 0 && decode_only_batch && N > 1) {
                    const char* path =
                        use_paged_decode_attention
                            ? "paged_decode"
                            : (use_hal_attention_dispatch         ? "hal_flash"
                               : use_portable_cpu_flash_attention ? "cpu_flash_hal"
                                                                  : (use_flash_attention ? "flash" : "standard"));
                    std::cerr << "[DecodeAttentionPath] N=" << N << " path=" << path << std::endl;
                }

                if (use_paged_decode_attention) {
                    // -----------------------------------------------------------------------
                    // DECODE PAGED ATTENTION FAST PATH (decode-only batches, one token/seq)
                    // -----------------------------------------------------------------------
                    // Avoids materializing [n_total] K/V tensors each token.
                    // Writes current K/V to paged cache and reads history directly from cache.
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q_decode = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
                    struct ggml_tensor* K_decode = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
                    struct ggml_tensor* V_decode = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);

                    PagedAttentionUserData* ud = GetPagedAttentionUserData();
                    ud->cache = cache;
                    ud->layer = il;
                    ud->head_dim = head_dim_q;
                    ud->v_head_dim = head_dim_v;
                    ud->n_head = n_head;
                    ud->epoch_started.store(0, std::memory_order_relaxed);
                    ud->epoch_done.store(0, std::memory_order_relaxed);
                    ud->kv_writers_done.store(0, std::memory_order_relaxed);
                    KQV = ggml_paged_attention_decode(ctx_c, Q_decode, K_decode, V_decode, ud);
                } else if (use_hal_attention_dispatch) {
                    // -----------------------------------------------------------------------
                    // HAL ATTENTION PATH (per-op-class mixed routing)
                    // -----------------------------------------------------------------------
                    // Dispatches attention to the preferred attention device while the rest
                    // of the graph can remain on a different primary backend.
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
                    struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                    struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
                    Q_hal = ggml_cont(ctx_c, Q_hal);
                    K_hal = ggml_cont(ctx_c, K_hal);
                    V_hal = ggml_cont(ctx_c, V_hal);

                    const float scale = 1.0f / sqrtf((float)head_dim_q);
                    // For decode (N==1), causal=false is correct because K already contains
                    // only historical + current keys (no future positions).
                    const bool hal_causal = (N > 1);
                    KQV = ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv,
                                                   preferred_attention_device);

                    // Convert [head_dim, N, n_head] -> [head_dim, n_head, N]
                    KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                } else if (use_portable_cpu_flash_attention) {
                    // -----------------------------------------------------------------------
                    // PORTABLE CPU FLASH ATTENTION PATH
                    // -----------------------------------------------------------------------
                    // On ARM/Apple CPU runtimes, route through DenseCore's backend-agnostic
                    // FlashAttention op instead of forcing the materialized standard path.
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
                    struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                    struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
                    Q_hal = ggml_cont(ctx_c, Q_hal);
                    K_hal = ggml_cont(ctx_c, K_hal);
                    V_hal = ggml_cont(ctx_c, V_hal);

                    const float scale = 1.0f / sqrtf((float)head_dim_q);
                    const bool hal_causal = (N > 1);
                    KQV = ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv,
                                                   densecore::DeviceType::CPU);

                    // Convert [head_dim, N, n_head] -> [head_dim, n_head, N]
                    KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                } else if (use_flash_attention) {
                    // -----------------------------------------------------------------------
                    // FLASH ATTENTION PATH (AVX-512 only)
                    // -----------------------------------------------------------------------
                    // ggml_flash_attn_ext natively supports GQA - it handles K/V with fewer
                    // heads than Q. The kernel internally computes: kv_head = query_head /
                    // n_rep
                    //
                    // Shapes: Q: [head_dim, N, n_head], K/V: [head_dim, n_total, n_head_kv]
                    // -----------------------------------------------------------------------
                    struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

                    // K/V: [head_dim, n_head_kv, n_total] -> [head_dim, n_total, n_head_kv]
                    struct ggml_tensor* K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                    struct ggml_tensor* V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

                    // Create mask [n_total, N_padded, 1, 1] as required by
                    // ggml_flash_attn_ext 0.0f = can attend, -INFINITY = cannot attend
                    // (masked)
                    int N_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);
                    struct ggml_tensor* KQ_mask = nullptr;
                    if (shared_prefill_flash_mask && shared_prefill_mask_n_total == n_total_tokens &&
                        shared_prefill_mask_n_padded == N_padded && shared_prefill_mask_n == N &&
                        shared_prefill_mask_n_past == n_past_val) {
                        KQ_mask = shared_prefill_flash_mask;
                    } else {
                        KQ_mask = ggml_new_tensor_4d(ctx_c, GGML_TYPE_F32, n_total_tokens, N_padded, 1, 1);

                        // Fill causal mask (column-major: element (k, q) is at k + q * n_kv)
                        float* mask_data = reinterpret_cast<float*>(KQ_mask->data);
                        for (int q = 0; q < N_padded; q++) {
                            for (int k = 0; k < n_total_tokens; k++) {
                                const int query_pos = n_past_val + q;
                                const int key_pos = k;
                                const int idx = k + q * n_total_tokens;

                                if (q >= N || key_pos <= query_pos) {
                                    mask_data[idx] = 0.0f;
                                } else {
                                    mask_data[idx] = -INFINITY;
                                }
                            }
                        }

                        if (N > 1 && !decode_only_batch) {
                            shared_prefill_flash_mask = KQ_mask;
                            shared_prefill_mask_n_total = n_total_tokens;
                            shared_prefill_mask_n_padded = N_padded;
                            shared_prefill_mask_n = N;
                            shared_prefill_mask_n_past = n_past_val;
                        }
                    }

                    // Ensure contiguity for Flash Attention
                    Q = ggml_cont(ctx_c, Q);
                    K_fa = ggml_cont(ctx_c, K_fa);
                    V_fa = ggml_cont(ctx_c, V_fa);

                    // Scale factor: 1/sqrt(head_dim)
                    float scale = 1.0f / sqrtf((float)head_dim_q);

                    // Flash Attention: fused Q*K^T, scale, mask, softmax, *V
                    // Result: [head_dim, N, n_head]
                    KQV = ggml_flash_attn_ext(ctx_c, Q, K_fa, V_fa, KQ_mask, scale, 0.0f, 0.0f);

                    // Permute to [head_dim, n_head, N] for projection
                    KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                } else {
                    // -----------------------------------------------------------------------
                    // STANDARD ATTENTION PATH (AVX2/Fallback) - Tiled GQA Implementation
                    // -----------------------------------------------------------------------
                    // For GQA models (n_head != n_head_kv), we use a TILED approach:
                    //   - Iterate over KV heads (h_kv = 0 to n_head_kv)
                    //   - For each KV head, process n_rep query heads together
                    //   - Use ggml_view to slice tensors without copying (O(1) memory)
                    //
                    // This avoids the massive memory bloat of ggml_repeat while maintaining
                    // correctness on all hardware (AVX2, SSE, etc.)
                    // -----------------------------------------------------------------------

                    // Scale factor for attention
                    float scale = 1.0f / sqrtf((float)head_dim_q);

                    // =================================================================
                    // UNIFIED ATTENTION PATH (GQA + MHA)
                    // =================================================================
                    // GGML's mul_mat natively supports GQA broadcasting:
                    // when K has n_head_kv heads and Q has n_head heads (where
                    // n_head % n_head_kv == 0), mul_mat broadcasts K across
                    // query head groups automatically. This matches llama.cpp.
                    // =================================================================
                    {
                        const bool skip_attn_cont = (N > 1) && IsPrefillAttentionSkipContEnabled();
                        // Q: [head_dim, N, n_head]
                        struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

                        // K/V: [head_dim, n_total, n_head_kv]
                        struct ggml_tensor* K_att = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                        struct ggml_tensor* V_att = ggml_permute(ctx_c, V, 0, 2, 1, 3);

                        // ggml_mul_mat requires lhs (a) to be non-transposed. Always
                        // materialize K_att.
                        K_att = ggml_cont(ctx_c, K_att);
                        if (!skip_attn_cont) {
                            // Fully contiguous fallback path.
                            Q = ggml_cont(ctx_c, Q);
                            V_att = ggml_cont(ctx_c, V_att);
                        }

                        // Q @ K^T -> [n_total, N, n_head] (broadcasts n_head_kv → n_head)
                        struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_att, Q);

                        // Scale
                        KQ = ggml_scale(ctx_c, KQ, scale);

                        // Causal mask (prefill only)
                        if (N > 1) {
                            KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
                        }

                        // Softmax
                        KQ = ggml_soft_max(ctx_c, KQ);

                        // KQ @ V -> [head_dim, N, n_head]
                        // V needs transpose: [n_total, head_dim, n_head_kv]
                        struct ggml_tensor* V_t = ggml_permute(ctx_c, V_att, 1, 0, 2, 3);
                        // ggml_mul_mat requires lhs (a) to be non-transposed.
                        V_t = ggml_cont(ctx_c, V_t);
                        KQV = ggml_mul_mat(ctx_c, V_t, KQ);

                        // Permute to [head_dim, n_head, N] for projection
                        KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
                    }
                }
            }

            // Must be contiguous before reshape
            struct ggml_tensor* KQV_merged = ggml_cont(ctx_c, KQV);
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_kqv = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[KQV%d #%d] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f "
                                "max=%.6f mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (int)src->type, (long)src->ne[0], (long)src->ne[1],
                                (long)src->ne[2], n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                KQV_merged = ggml_map_custom1(ctx_c, KQV_merged, cb_check_kqv, 1, &dbg_layers[il]);
            }

            const int attn_out_head_dim = static_cast<int>(KQV_merged->ne[0]);
            cur = ggml_reshape_2d(ctx_c, KQV_merged, attn_out_head_dim * n_head, N);
            if (attn_gate) {
                attn_gate = ggml_sigmoid(ctx_c, attn_gate);
                cur = ggml_mul(ctx_c, cur, attn_gate);
            }

            // Output Projection (using smart dispatcher for Parallel GEMV)
            struct ggml_tensor* cur_input_to_wo = cur;
            if (!wo) {
                throw densecore::InvalidArgumentException("Missing attn_output weight in TransformerLayer");
            }
            cur = smart_mul_mat(ctx_c, wo, cur, model);

            // Apply Multi-LoRA to Output Projection
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, cur_input_to_wo, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
            if (bo) cur = ggml_add(ctx_c, cur, bo);

            // Residual Connection
            attn_out = cur;
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_attn = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                        void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[ATTN%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f "
                                "mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct,
                                inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                cur = ggml_map_custom1(ctx_c, cur, cb_check_attn, 1, &dbg_layers[il]);
            }
            // Usually output projection expects n_embd input.
            // wo: [n_embd, n_embd] (or [n_embd, n_head*head_dim])
            // The standard transformer expects concatenation of all heads to be
            // n_embd. If n_head * head_dim_kv != n_embd, we have a mismatch. Qwen3
            // has n_head=16, head_dim_kv=128 => 2048 != 1024. This implies wo expects
            // 2048 input!

            // KQV = ggml_reshape_2d(ctx_c, KQV, n_head * head_dim_kv, N);

            // Output projection
            // cur = ggml_mul_mat(ctx_c, model->layers[il].wo, KQV);
            // if (model->layers[il].bo)
            //   cur = ggml_add(ctx_c, cur, model->layers[il].bo);

            // Residual connection
            // cur = ggml_add(ctx_c, cur, inpL);

        }  // end else (attention path)

        // =========================================================================
        // FFN (shared between SSM and attention layers)
        // =========================================================================
        struct ggml_tensor* inpFF = attn_post_residual;
        if (!ffn_norm) {
            throw densecore::InvalidArgumentException("Missing ffn_norm weight in TransformerLayer");
        }

        bool used_fused_pre_ffn_norm = false;
        if (IsFusedResidualRmsNormEnabled() && attn_out && inpL && attn_out->type == GGML_TYPE_F32 &&
            inpL->type == GGML_TYPE_F32 && ffn_norm->type == GGML_TYPE_F32 && attn_out->data && inpL->data &&
            ffn_norm->data) {
            AddRMSNormUserData* fused_ud = GetAddRMSNormUserData();
            fused_ud->residual = reinterpret_cast<const float*>(inpL->data);
            fused_ud->rms_weight = reinterpret_cast<const float*>(ffn_norm->data);
            fused_ud->n_embd = static_cast<int>(attn_out->ne[0]);
            fused_ud->n_tokens = static_cast<int>(attn_out->ne[1]);
            fused_ud->eps = model->hparams.f_norm_rms_eps;

            const int n_tasks = ResolveTaskCount(&batch, std::max<int>(1, fused_ud->n_tokens));
            cur = ggml_map_custom1(ctx_c, attn_out, cb_residual_rmsnorm_fused, n_tasks, fused_ud);
            used_fused_pre_ffn_norm = true;
        }

        if (!used_fused_pre_ffn_norm) {
            cur = apply_weighted_rms_norm(cur, ffn_norm, "ffn_norm");
        }

        if (model->layers[il].is_moe) {
            // =====================================================================
            // MOE PATH
            // =====================================================================
            struct ggml_tensor* moe_input = cur;

            // 1. Router: gate_logits = moe_gate * input
            if (!moe_gate) {
                throw densecore::InvalidArgumentException("Missing moe_gate weight in TransformerLayer");
            }
            struct ggml_tensor* gate_logits = smart_mul_mat(ctx_c, moe_gate, cur, model);

            // 2. Dispatch
            MoEUserData* moe_ud = AllocateMoEUserData(ctx_c);
            if (moe_ud) {
                moe_ud->model = model;
                moe_ud->layer = &model->layers[il];
                moe_ud->k = model->hparams.n_experts_used;
                moe_ud->batch = &batch;
                moe_ud->scheduler = batch.scheduler;
                // Resolve preferred device first, then guarantee CPU fallback for MoE.
                densecore::BackendRegistry& registry = ResolveBackendRegistry(&batch);
                densecore::ComputeBackend* preferred_backend = registry.Get(ResolvePreferredDevice(&batch));
                moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(preferred_backend);
                if (!moe_ud->backend) {
                    moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(registry.Get(densecore::DeviceType::CPU));
                }
                if (moe_ud->backend) {
                    auto experts = BuildExpertWeights(moe_ud->layer);
                    const int n_experts = static_cast<int>(experts.size());
                    moe_ud->backend->InitMoEProfiler(moe_ud->layer, n_experts);
                    moe_ud->backend->RegisterMoEExperts(moe_ud->layer, experts);
                    EnsureMoERebalanceThread(moe_ud->backend);
                }
            }

            // Use map_custom2: src0=cur, src1=gate_logits
            // Use 1 task (single thread dispatch, threaded inside backend)
            cur = ggml_map_custom2(ctx_c, cur, gate_logits, cb_moe_forward, 1, moe_ud);
            ggml_set_name(cur, "moe_forward");

            // GLM-style MoE keeps a dense shared-expert MLP in parallel with routed experts.
            if (model->moe_n_shared_experts > 0 && ffn_gate && ffn_up && ffn_down) {
                struct ggml_tensor* shared_gate = smart_mul_mat(ctx_c, ffn_gate, moe_input, model);
                struct ggml_tensor* shared_up = smart_mul_mat(ctx_c, ffn_up, moe_input, model);
                struct ggml_tensor* shared_ffn =
                    ggml_map_custom2(ctx_c, shared_gate, shared_up, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
                shared_ffn = smart_mul_mat(ctx_c, ffn_down, shared_ffn, model);
                cur = ggml_add(ctx_c, cur, shared_ffn);
            }
        } else {
            // =====================================================================
            // DENSE PATH
            // =====================================================================
            // SwiGLU FFN (using smart dispatcher for INT4 support)
            if (!ffn_gate || !ffn_up || !ffn_down) {
                throw densecore::InvalidArgumentException("Missing FFN weights in TransformerLayer");
            }
            struct ggml_tensor* w1 = smart_mul_mat(ctx_c, ffn_gate, cur, model);
            struct ggml_tensor* w3 = smart_mul_mat(ctx_c, ffn_up, cur, model);

            // Apply Multi-LoRA [FFN Gate/Up]
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate", il);
                ggml_set_name(w1, name_buf);
                w1 = ggml_map_custom2(ctx_c, w1, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());

                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up", il);
                ggml_set_name(w3, name_buf);
                w3 = ggml_map_custom2(ctx_c, w3, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }

            // Fused SiLU×Mul: silu(w1) * w3 in single pass (avoids intermediate tensor)
            // This saves ~50% memory bandwidth in FFN forward pass.
            cur = ggml_map_custom2(ctx_c, w1, w3, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
            ggml_set_name(cur, "ffn_silu_mul_fused");

            // Apply Multi-LoRA [FFN Down]
            struct ggml_tensor* ffn_input = cur;
            cur = smart_mul_mat(ctx_c, ffn_down, cur, model);
            {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, ffn_input, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
        }

        // Residual connection
        cur = ggml_add(ctx_c, cur, inpFF);

        if (IsDebugInferenceStatsEnabled()) {
            auto cb_check_layer = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                     void* ud) {
                (void)nth;
                if (ith != 0) return;
                const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                if (layer_idx < 0 || layer_idx >= 128) return;

                static int cb_ct[128] = {0};
                const float* d = reinterpret_cast<const float*>(src->data);
                const int n = ggml_nelements(src);
                if (!d || n <= 0) {
                    if (dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                    return;
                }

                int zero_ct = 0;
                int nan_ct = 0;
                int inf_ct = 0;
                float mn = std::numeric_limits<float>::infinity();
                float mx = -std::numeric_limits<float>::infinity();
                double sum = 0.0;
                double sum_sq = 0.0;
                int finite_ct = 0;
                for (int i = 0; i < n; i++) {
                    const float v = d[i];
                    if (std::isnan(v)) {
                        nan_ct++;
                        continue;
                    }
                    if (!std::isfinite(v)) {
                        inf_ct++;
                        continue;
                    }
                    if (v == 0.0f) zero_ct++;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    finite_ct++;
                }

                // Keep early-layer shape/stats visibility while always surfacing NaN/Inf.
                const bool emit_regular = (layer_idx <= 5) && (cb_ct[layer_idx] < 5);
                const bool emit_anomaly = (nan_ct > 0 || inf_ct > 0);
                if (emit_regular || emit_anomaly) {
                    if (!std::isfinite(mn)) mn = 0.0f;
                    if (!std::isfinite(mx)) mx = 0.0f;
                    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                    fprintf(stderr,
                            "[LAYER%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f "
                            "rms=%.6f\n",
                            layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct, inf_ct,
                            mn, mx, mean, rms);
                }

                if (dst->data && src->data) {
                    memcpy(dst->data, src->data, ggml_nbytes(src));
                }
                cb_ct[layer_idx]++;
            };
            static int layers[128] = {};
            for (int li = 0; li < 128; ++li) layers[li] = li;
            void* layer_ud = static_cast<void*>(&layers[il]);
            cur = ggml_map_custom1(ctx_c, cur, cb_check_layer, 1, layer_ud);
        }
    }

    // =========================================================================
    // 3. Final Layer Norm and LM Head
    // =========================================================================
    cur = apply_weighted_rms_norm(cur, model->output_norm, "output_norm");

    if (embedding_mode) {
        return cur;
    }

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check hidden state before lm_head
        auto cb_check_hidden = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* ud) {
            if (ith != 0) return;
            static int cb_ct = 0;
            const bool emit = (cb_ct < 5);
            if (emit && src && src->data) {
                const float* d = (const float*)src->data;
                int n = ggml_nelements(src);
                int zero_ct = 0;
                float mn = d[0], mx = d[0];
                for (int i = 0; i < n; i++) {
                    if (d[i] == 0.0f)
                        zero_ct++;
                    else {
                        if (d[i] < mn) mn = d[i];
                        if (d[i] > mx) mx = d[i];
                    }
                }
                fprintf(stderr, "[HIDDEN #%d] shape=[%ld,%ld] total=%d zero=%d min=%.6f max=%.6f\n", cb_ct,
                        (long)src->ne[0], (long)src->ne[1], n, zero_ct, mn, mx);
                cb_ct++;
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_hidden, 1, nullptr);
    }

    // LM Head projection: [n_embd, N] -> [n_vocab, N]
    cur = smart_mul_mat(ctx_c, model->output, cur, model);
    ggml_set_name(cur, "output");

    // Add to graph
    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }

    return cur;
}

// ============================================================================
// Grammar-Based Sampling Implementation
// ============================================================================

void InitGrammarConstraint(GrammarConstraint* grammar, const std::vector<std::string>& vocab) {
    if (!grammar) return;

    // Find token IDs for JSON special characters
    for (size_t i = 0; i < vocab.size(); i++) {
        const std::string& token = vocab[i];
        if (token == "{" || token == " {")
            grammar->token_lbrace = i;
        else if (token == "}" || token == " }")
            grammar->token_rbrace = i;
        else if (token == "[" || token == " [")
            grammar->token_lbracket = i;
        else if (token == "]" || token == " ]")
            grammar->token_rbracket = i;
        else if (token == "\"" || token == " \"")
            grammar->token_quote = i;
        else if (token == ":" || token == " :")
            grammar->token_colon = i;
        else if (token == "," || token == " ,")
            grammar->token_comma = i;
    }
}

void GrammarConstraint::UpdateState(const std::string& token_text) {
    if (!enabled || !is_json_mode) return;

    accumulated += token_text;

    // Trim leading whitespace for state transitions
    std::string trimmed = token_text;
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
        trimmed = trimmed.substr(start);
    }

    if (trimmed.empty()) return;

    char first_char = trimmed[0];

    switch (state) {
    case JSONState::EXPECT_OBJECT_START:
        if (first_char == '{') {
            state = JSONState::EXPECT_KEY_OR_END;
            brace_depth = 1;
        }
        break;

    case JSONState::EXPECT_KEY_OR_END:
        if (first_char == '"') {
            state = JSONState::IN_KEY;
        } else if (first_char == '}') {
            brace_depth--;
            if (brace_depth == 0) {
                state = JSONState::COMPLETED;
            }
        }
        break;

    case JSONState::IN_KEY:
        if (first_char == '"' && !in_escape) {
            state = JSONState::EXPECT_COLON;
        } else if (first_char == '\\') {
            in_escape = !in_escape;
        } else {
            in_escape = false;
        }
        break;

    case JSONState::EXPECT_COLON:
        if (first_char == ':') {
            state = JSONState::EXPECT_VALUE;
        }
        break;

    case JSONState::EXPECT_VALUE:
        if (first_char == '"') {
            state = JSONState::IN_STRING_VALUE;
        } else if (first_char == '{') {
            brace_depth++;
            state = JSONState::EXPECT_KEY_OR_END;
        } else if (first_char == '[') {
            bracket_depth++;
            state = JSONState::IN_ARRAY;
        } else if (isdigit(first_char) || first_char == '-') {
            state = JSONState::IN_NUMBER;
        } else if (trimmed.find("true") == 0 || trimmed.find("false") == 0 || trimmed.find("null") == 0) {
            state = JSONState::EXPECT_COMMA_OR_END;
        }
        break;

    case JSONState::IN_STRING_VALUE:
        if (first_char == '"' && !in_escape) {
            state = JSONState::EXPECT_COMMA_OR_END;
        } else if (first_char == '\\') {
            in_escape = !in_escape;
        } else {
            in_escape = false;
        }
        break;

    case JSONState::IN_NUMBER:
        if (first_char == ',' || first_char == '}' || first_char == ']') {
            state = JSONState::EXPECT_COMMA_OR_END;
            if (first_char == ',') {
                state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
            } else if (first_char == '}') {
                brace_depth--;
                if (brace_depth == 0) state = JSONState::COMPLETED;
            } else if (first_char == ']') {
                bracket_depth--;
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        }
        break;

    case JSONState::EXPECT_COMMA_OR_END:
        if (first_char == ',') {
            state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
        } else if (first_char == '}') {
            brace_depth--;
            if (brace_depth == 0) {
                state = JSONState::COMPLETED;
            }
        } else if (first_char == ']') {
            bracket_depth--;
            if (bracket_depth == 0) {
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        }
        break;

    case JSONState::IN_ARRAY:
        if (first_char == ']') {
            bracket_depth--;
            if (bracket_depth == 0) {
                state = JSONState::EXPECT_COMMA_OR_END;
            }
        } else if (first_char == ',') {
            // Stay in array
        } else if (first_char == '"') {
            state = JSONState::IN_STRING_VALUE;
        } else if (first_char == '{') {
            brace_depth++;
            state = JSONState::EXPECT_KEY_OR_END;
        }
        break;

    case JSONState::COMPLETED: break;
    }
}

bool IsDigitToken(const std::string& token) {
    if (token.empty()) return false;
    for (char c : token) {
        if (!isdigit(c) && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+' && c != ' ') return false;
    }
    return true;
}

bool IsWhitespaceToken(const std::string& token) {
    if (token.empty()) return false;
    for (char c : token) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return false;
    }
    return true;
}

bool ContainsChar(const std::string& token, char ch) {
    return token.find(ch) != std::string::npos;
}

void ApplyGrammarMask(float* logits, int n_vocab, const GrammarConstraint* grammar,
                      const std::vector<std::string>& vocab) {
    if (!grammar || !grammar->enabled || !grammar->is_json_mode) {
        return;
    }

    const float NEG_INF = -INFINITY;
    std::vector<bool> allowed(n_vocab, false);

    // Always allow whitespace
    for (int i = 0; i < n_vocab; i++) {
        if (IsWhitespaceToken(vocab[i])) {
            allowed[i] = true;
        }
    }

    switch (grammar->state) {
    case JSONState::EXPECT_OBJECT_START:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], '{')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_KEY_OR_END:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], '"') || ContainsChar(vocab[i], '}')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_KEY:
    case JSONState::IN_STRING_VALUE:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            bool has_control = false;
            for (char c : token) {
                if (c < 32 && c != '\t' && c != '\n') {
                    has_control = true;
                    break;
                }
            }
            if (!has_control) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_COLON:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], ':')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_VALUE:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                IsDigitToken(token) || token.find("true") != std::string::npos ||
                token.find("false") != std::string::npos || token.find("null") != std::string::npos) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_NUMBER:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (IsDigitToken(token) || ContainsChar(token, ',') || ContainsChar(token, '}') ||
                ContainsChar(token, ']')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::EXPECT_COMMA_OR_END:
        for (int i = 0; i < n_vocab; i++) {
            if (ContainsChar(vocab[i], ',') || ContainsChar(vocab[i], '}') || ContainsChar(vocab[i], ']')) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_ARRAY:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                ContainsChar(token, ']') || ContainsChar(token, ',') || IsDigitToken(token)) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::COMPLETED: break;
    }

    for (int i = 0; i < n_vocab; i++) {
        if (!allowed[i]) {
            logits[i] = NEG_INF;
        }
    }
}

// ============================================================================
// Token Sampling
// ============================================================================

int SampleToken(struct ggml_tensor* logits, int idx, const SamplingParams& params) {
    if (!logits || !logits->data) return 0;

    float* logits_data = (float*)logits->data;
    int n_vocab = logits->ne[0];
    if (n_vocab <= 0) return 0;

    const int n_cols = std::max<int>(1, (int)logits->ne[1]);
    if (idx < 0) idx = 0;
    if (idx >= n_cols) idx = n_cols - 1;
    const ptrdiff_t row_stride = static_cast<ptrdiff_t>(logits->nb[1] / sizeof(float));
    if (row_stride < n_vocab) {
        return 0;
    }
    float* last_logits = logits_data + static_cast<ptrdiff_t>(idx) * row_stride;

    int range_start = 0;
    int range_end = n_vocab;
    if (params.action_token_count > 0 && params.grammar == nullptr) {
        const int requested_start = std::max(0, params.action_token_start);
        if (requested_start < n_vocab) {
            const int64_t requested_end = static_cast<int64_t>(requested_start) + params.action_token_count;
            if (requested_end > requested_start) {
                range_start = requested_start;
                range_end = static_cast<int>(std::min<int64_t>(n_vocab, requested_end));
            }
        }
    }
    if (range_end <= range_start) {
        range_start = 0;
        range_end = n_vocab;
    }
    const int active_vocab = range_end - range_start;
    auto is_disallowed = [&](int token_id) -> bool {
        if (!params.disallowed_token_ids || params.disallowed_token_ids->empty()) {
            return false;
        }
        return std::binary_search(params.disallowed_token_ids->begin(), params.disallowed_token_ids->end(), token_id);
    };

    auto finite_argmax_raw = [last_logits, range_start, range_end, &is_disallowed]() -> int {
        int best_idx = range_start;
        float best_val = -INFINITY;
        bool found = false;
        for (int i = range_start; i < range_end; ++i) {
            const float v = last_logits[i];
            if (!std::isfinite(v) || is_disallowed(i)) continue;
            if (!found || v > best_val) {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        return found ? best_idx : range_start;
    };
    const bool debug_sample = (std::getenv("DENSECORE_DEBUG_SAMPLE") != nullptr);
    auto debug_log_sample = [&](int token_id) {
        if (!debug_sample) return;
        static int debug_sample_count = 0;
        if (debug_sample_count >= 8) return;
        const float logit = (token_id >= range_start && token_id < range_end) ? last_logits[token_id] : NAN;
        fprintf(stderr, "[SAMPLE_DBG #%d] idx=%d token=%d logit=%.6f range=[%d,%d)\n", debug_sample_count, idx,
                token_id, logit, range_start, range_end);
        if (params.vocab && debug_sample_count < 2) {
            std::vector<std::pair<float, int>> top;
            top.reserve(8);
            for (int i = range_start; i < range_end; ++i) {
                const float v = last_logits[i];
                if (!std::isfinite(v)) continue;
                if (top.size() < 8) {
                    top.emplace_back(v, i);
                    std::push_heap(top.begin(), top.end(),
                                   [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                    continue;
                }
                if (v > top.front().first) {
                    std::pop_heap(top.begin(), top.end(),
                                  [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                    top.back() = {v, i};
                    std::push_heap(top.begin(), top.end(),
                                   [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
                }
            }
            std::sort(top.begin(), top.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
            for (const auto& [score, top_id] : top) {
                std::string tok =
                    (top_id >= 0 && top_id < static_cast<int>(params.vocab->size())) ? (*params.vocab)[top_id] : "";
                for (char& ch : tok) {
                    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                }
                fprintf(stderr, "  [TOP] token=%d logit=%.6f text='%s'\n", top_id, score, tok.c_str());
            }
        }
        debug_sample_count++;
    };

    if (IsDebugInferenceStatsEnabled()) {
        // TEMP DEBUG: Check logits stats for first 5 calls
        static int dbg_cnt = 0;
        if (dbg_cnt < 5) {
            int nan_ct = 0;
            int zero_ct = 0;
            float mn = last_logits[range_start];
            float mx = last_logits[range_start];
            for (int i = range_start; i < range_end; i++) {
                if (std::isnan(last_logits[i]))
                    nan_ct++;
                else {
                    if (last_logits[i] == 0.0f) zero_ct++;
                    if (last_logits[i] < mn) mn = last_logits[i];
                    if (last_logits[i] > mx) mx = last_logits[i];
                }
            }
            fprintf(stderr, "[LOGITS #%d] idx=%d vocab=[%d,%d) min=%.4f max=%.4f nan=%d zero=%d\n", dbg_cnt, idx,
                    range_start, range_end, mn, mx, nan_ct, zero_ct);
            dbg_cnt++;
        }
    }

    const bool has_history = params.token_history && !params.token_history->empty();
    const bool has_penalty = has_history && (params.repetition_penalty != 1.0f || params.frequency_penalty != 0.0f ||
                                             params.presence_penalty != 0.0f);
    if (params.temperature <= 0.0f && params.grammar == nullptr && !has_penalty) {
        const int token = finite_argmax_raw();
        debug_log_sample(token);
        return token;
    }
    if (params.temperature == 1.0f && params.top_k <= 1 && params.top_p >= 1.0f && params.min_p <= 0.0f &&
        params.grammar == nullptr && !has_penalty) {
        const int token = finite_argmax_raw();
        debug_log_sample(token);
        return token;
    }

    thread_local std::vector<float> working_logits;
    working_logits.assign(last_logits + range_start, last_logits + range_end);
    if (params.disallowed_token_ids && !params.disallowed_token_ids->empty()) {
        for (int token_id : *params.disallowed_token_ids) {
            if (token_id >= range_start && token_id < range_end) {
                working_logits[token_id - range_start] = -INFINITY;
            }
        }
    }
    auto finite_argmax = [range_start]() -> int {
        int best_idx = 0;
        float best_val = -INFINITY;
        bool found = false;
        for (int i = 0; i < static_cast<int>(working_logits.size()); ++i) {
            const float v = working_logits[i];
            if (!std::isfinite(v)) continue;
            if (!found || v > best_val) {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        return found ? (range_start + best_idx) : range_start;
    };
    auto sanitize_logits = []() -> int {
        int finite_count = 0;
        for (float& v : working_logits) {
            if (std::isfinite(v)) {
                finite_count++;
            } else {
                v = -INFINITY;
            }
        }
        return finite_count;
    };

    if (params.grammar && params.vocab) {
        ApplyGrammarMask(working_logits.data(), active_vocab, params.grammar, *params.vocab);
    }

    if (params.repetition_penalty != 1.0f && params.token_history && !params.token_history->empty()) {
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                const int local_token = token - range_start;
                if (working_logits[local_token] < 0) {
                    working_logits[local_token] *= params.repetition_penalty;
                } else {
                    working_logits[local_token] /= params.repetition_penalty;
                }
            }
        }
    }

    if ((params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) && params.token_history &&
        !params.token_history->empty()) {
        std::unordered_map<int, int> token_counts;
        token_counts.reserve(params.token_history->size());
        for (int token : *params.token_history) {
            if (token >= range_start && token < range_end) {
                token_counts[token - range_start]++;
            }
        }

        for (auto& kv : token_counts) {
            const int token = kv.first;
            int count = kv.second;
            float penalty = params.frequency_penalty * count + params.presence_penalty * (count > 0 ? 1.0f : 0.0f);
            working_logits[token] -= penalty;
        }
    }

    if (params.temperature <= 0.0f) {
        const int token = finite_argmax();
        debug_log_sample(token);
        return token;
    }
    if (params.temperature != 1.0f && params.temperature > 0.0f) {
        for (int i = 0; i < active_vocab; i++) {
            working_logits[i] /= params.temperature;
        }
    }
    if (sanitize_logits() == 0) {
        const int token = finite_argmax_raw();
        debug_log_sample(token);
        return token;
    }

    float max_logit = -INFINITY;
    for (float v : working_logits) {
        if (std::isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!std::isfinite(max_logit)) {
        const int token = finite_argmax();
        debug_log_sample(token);
        return token;
    }

    int k = params.top_k;
    if (k <= 0 || k > active_vocab) {
        k = active_vocab;
    }

    thread_local std::vector<std::pair<float, int>> prob_idx;
    prob_idx.clear();
    prob_idx.reserve(static_cast<size_t>(active_vocab));
    float sum_exp = 0.0f;
    for (int i = 0; i < active_vocab; i++) {
        float prob = std::isfinite(working_logits[i]) ? std::exp(working_logits[i] - max_logit) : 0.0f;
        if (!std::isfinite(prob)) {
            prob = 0.0f;
        }
        sum_exp += prob;
        prob_idx.push_back({prob, range_start + i});
    }

    if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) {
        const int token = finite_argmax();
        debug_log_sample(token);
        return token;
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    for (auto& p : prob_idx) {
        p.first *= inv_sum_exp;
    }

    auto prob_desc = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first > b.first;
    };

    if (k < active_vocab) {
        std::nth_element(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(), prob_desc);
        prob_idx.resize(k);
    }
    std::sort(prob_idx.begin(), prob_idx.end(), prob_desc);

    if (params.min_p > 0.0f && !prob_idx.empty()) {
        float max_prob = prob_idx[0].first;
        float threshold = params.min_p * max_prob;
        auto it = std::remove_if(prob_idx.begin(), prob_idx.end(),
                                 [threshold](const auto& p) { return p.first < threshold; });
        prob_idx.erase(it, prob_idx.end());
    }

    if (params.top_p < 1.0f && !prob_idx.empty()) {
        float cumulative = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < prob_idx.size(); i++) {
            cumulative += prob_idx[i].first;
            cutoff = i + 1;
            if (cumulative >= params.top_p) {
                break;
            }
        }
        prob_idx.resize(cutoff);
    }

    if (prob_idx.empty()) {
        const int token = finite_argmax();
        debug_log_sample(token);
        return token;
    }

    float total = 0.0f;
    for (const auto& p : prob_idx) {
        total += p.first;
    }
    if (total <= 0.0f || !std::isfinite(total)) {
        const int token = prob_idx[0].second;
        debug_log_sample(token);
        return token;
    }
    const float inv_total = 1.0f / total;
    for (auto& p : prob_idx) {
        p.first *= inv_total;
    }

    thread_local std::unique_ptr<std::mt19937> rng;
    if (!rng) {
        rng = std::make_unique<std::mt19937>(std::random_device{}());
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_val = dist(*rng);
    float cumulative = 0.0f;
    for (const auto& p : prob_idx) {
        cumulative += p.first;
        if (random_val <= cumulative) {
            debug_log_sample(p.second);
            return p.second;
        }
    }

    debug_log_sample(prob_idx[0].second);
    return prob_idx[0].second;
}
