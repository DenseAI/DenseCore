#include <ggml-cpu.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "densecore/utils/logging.h"

#include "densecore/backend/hardware_topology.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/hal/op_registry.h"
#include "densecore/runtime/optimization_bridge.h"  // Runtime SIMD dispatch
#include "densecore/simd/simd_ops.h"
#include "ggml.h"
#include "llm/config/runtime_config.h"
#include "runtime/engine_internal.h"
#include "runtime/runtime_env.h"
#include "runtime/worker_internal.h"
#include "utils/raii_guards.h"

#include "densecore/arm_runtime.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/graph_executor.h"
#include "densecore/models/graph_registry.h"
#include "densecore/models/model_descriptor.h"
#include "densecore/runtime/inference.h"
#include "models/model_inference_policy.h"

namespace {

thread_local const densecore::llm::config::FastPathRuntimeConfig* tls_fast_path_runtime_config = nullptr;

const densecore::llm::config::FastPathRuntimeConfig& GetFastPathRuntimeConfig() {
    if (tls_fast_path_runtime_config) {
        return *tls_fast_path_runtime_config;
    }
    static const densecore::llm::config::FastPathRuntimeConfig config =
        densecore::llm::config::LoadFastPathRuntimeConfig();
    return config;
}

const densecore::llm::config::WorkerRuntimeConfig& GetWorkerRuntimeConfig() {
    return GetFastPathRuntimeConfig().worker;
}

class ScopedFastPathRuntimeConfigBinder {
public:
    explicit ScopedFastPathRuntimeConfigBinder(const densecore::llm::config::FastPathRuntimeConfig* config)
        : previous_(tls_fast_path_runtime_config) {
        tls_fast_path_runtime_config = config;
    }

    ~ScopedFastPathRuntimeConfigBinder() { tls_fast_path_runtime_config = previous_; }

private:
    const densecore::llm::config::FastPathRuntimeConfig* previous_ = nullptr;
};

bool IsMulGraphValidationEnabled() {
    return GetWorkerRuntimeConfig().validate_mul;
}

bool IsRuntimePathLoggingEnabled() {
    return GetWorkerRuntimeConfig().runtime_path_logging;
}

bool IsSchedulerStallDebugEnabled() {
    return GetWorkerRuntimeConfig().scheduler_stall_debug;
}

long GetEmptyScheduleWarnAfterMs() {
    static const long value =
        std::max<long>(0, densecore::env::ParseIntEnv("DENSECORE_EMPTY_SCHEDULE_WARN_AFTER_MS", 250));
    return value;
}

long GetEmptyScheduleFailAfterMs() {
    static const long value =
        std::max<long>(0, densecore::env::ParseIntEnv("DENSECORE_EMPTY_SCHEDULE_FAIL_AFTER_MS", 2000));
    return value;
}

bool IsMoEPathTraceDumpEnabled() {
    return GetWorkerRuntimeConfig().moe_trace_dump;
}

bool IsSamplerTraceDumpEnabled() {
    return GetWorkerRuntimeConfig().sampler_trace_dump;
}

bool IsDeterminismBoundaryDebugEnabled() {
    return GetWorkerRuntimeConfig().determinism_boundary_debug;
}

bool IsPrefixCacheReuseDisabled() {
    return GetWorkerRuntimeConfig().prefix_cache_reuse_disabled;
}

bool IsHybridSSMSnapshotRestoreDisabled() {
    return GetWorkerRuntimeConfig().hybrid_ssm_snapshot_restore_disabled;
}

bool IsQwen36PrefixCacheReuseEnabled() {
    return GetWorkerRuntimeConfig().qwen36_prefix_cache_reuse_enabled;
}

bool IsQwen36HybridSSMSnapshotRestoreEnabled() {
    return GetWorkerRuntimeConfig().qwen36_hybrid_ssm_snapshot_restore_enabled;
}

bool IsQwen36HybridSSMModel(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    return densecore::models::DescribeModel(model).variant == ModelVariant::QWEN36;
}

bool IsPrefixCacheAllowedForModel(const TransformerModel* model) {
    if (IsPrefixCacheReuseDisabled()) {
        return false;
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36PrefixCacheReuseEnabled()) {
        return false;
    }
    return true;
}

bool IsHybridSSMSnapshotRestoreAllowedForModel(const TransformerModel* model) {
    if (IsHybridSSMSnapshotRestoreDisabled()) {
        return false;
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36HybridSSMSnapshotRestoreEnabled()) {
        return false;
    }
    return true;
}

bool ResolveSamplingLogitsColumnForRequestImpl(int token_offset, int processed_count, int output_columns,
                                               bool sampled_from_prefill, int remaining_prompt_tokens,
                                               int n_past_before, int n_past_after, int* out_last_token_idx,
                                               std::string* error) {
    auto fail = [&](const char* message) {
        if (error) {
            *error = message ? message : "unknown logits-column validation failure";
        }
        return false;
    };
    if (!out_last_token_idx) {
        return fail("missing logits-column output pointer");
    }
    if (token_offset < 0) {
        return fail("negative token offset");
    }
    if (processed_count <= 0) {
        return fail("processed_count must be positive");
    }
    if (output_columns <= 0) {
        return fail("output tensor must expose at least one column");
    }
    const int last_token_idx = token_offset + processed_count - 1;
    if (last_token_idx < token_offset || last_token_idx >= output_columns) {
        return fail("sampling logits column is out of bounds for output tensor");
    }
    if (sampled_from_prefill) {
        if (remaining_prompt_tokens <= 0) {
            return fail("prefill sampling requires positive remaining prompt tokens");
        }
        if (processed_count != remaining_prompt_tokens) {
            return fail("prefill sampling must use the final prompt chunk only");
        }
        if (n_past_after != n_past_before + processed_count) {
            return fail("prefill sampling observed inconsistent n_past progression");
        }
    }
    *out_last_token_idx = last_token_idx;
    return true;
}

bool IsGraphCacheReuseDisabled() {
    return GetWorkerRuntimeConfig().graph_cache_reuse_disabled;
}

bool IsMoETracePlumbingDisabled() {
    return GetWorkerRuntimeConfig().moe_trace_plumbing_disabled;
}

bool IsMoEGraphSummaryEnabled() {
    return GetWorkerRuntimeConfig().moe_graph_summary;
}

void MaybeLogMoEGraphSummary(struct ggml_cgraph* gf, const TransformerModel* model, bool is_prefill_batch) {
    if (!IsMoEGraphSummaryEnabled() || !gf || !model || model->hparams.n_layer == 0 || model->hparams.n_experts == 0) {
        return;
    }
    static std::atomic<bool> logged_prefill{false};
    static std::atomic<bool> logged_decode{false};
    if (is_prefill_batch) {
        if (logged_prefill.exchange(true, std::memory_order_relaxed)) return;
    } else {
        if (logged_decode.exchange(true, std::memory_order_relaxed)) return;
    }

    const int n_nodes = ggml_graph_n_nodes(gf);
    uint64_t moe_gating_ops = 0;
    uint64_t moe_scatter_ops = 0;
    uint64_t moe_forward_ops = 0;
    uint64_t moe_gather_ops = 0;
    uint64_t dense_ffn_matmul_ops = 0;
    std::vector<uint64_t> layer_dense_ffn_ops(model->hparams.n_layer, 0);
    std::vector<uint64_t> layer_moe_forward_ops(model->hparams.n_layer, 0);
    std::vector<uint64_t> layer_moe_gate_ops(model->hparams.n_layer, 0);

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(gf, i);
        if (!node) continue;
        const char* name = node->name;
        if (!name || !name[0]) continue;

        int layer_idx = -1;
        if (std::sscanf(name, "blk.%d.", &layer_idx) != 1) {
            layer_idx = -1;
        }
        const bool layer_ok = layer_idx >= 0 && layer_idx < static_cast<int>(model->hparams.n_layer);

        if (std::strstr(name, ".moe_gate_logits")) {
            ++moe_gating_ops;
            if (layer_ok) {
                ++layer_moe_gate_ops[static_cast<size_t>(layer_idx)];
            }
        } else if (std::strstr(name, ".moe_forward")) {
            ++moe_forward_ops;
            if (layer_ok) {
                ++layer_moe_forward_ops[static_cast<size_t>(layer_idx)];
            }
        } else if (std::strstr(name, ".ffn_gate") || std::strstr(name, ".ffn_up") || std::strstr(name, ".ffn_down")) {
            ++dense_ffn_matmul_ops;
            if (layer_ok) {
                ++layer_dense_ffn_ops[static_cast<size_t>(layer_idx)];
            }
        }
    }

    std::fprintf(stderr,
                 "[MOE_GRAPH_SUMMARY] phase=%s nodes=%d ops{MoEGating:%llu,MoEScatter:%llu,MoEForward:%llu,"
                 "MoEGather:%llu,dense_ffn_matmul:%llu} layers=%u\n",
                 is_prefill_batch ? "prefill" : "decode", n_nodes, static_cast<unsigned long long>(moe_gating_ops),
                 static_cast<unsigned long long>(moe_scatter_ops), static_cast<unsigned long long>(moe_forward_ops),
                 static_cast<unsigned long long>(moe_gather_ops), static_cast<unsigned long long>(dense_ffn_matmul_ops),
                 model->hparams.n_layer);

    for (uint32_t layer = 0; layer < model->hparams.n_layer; ++layer) {
        const TransformerLayer& l = model->layers[layer];
        std::fprintf(stderr,
                     "[MOE_GRAPH_LAYER] phase=%s layer=%u is_moe=%d has_moe_gate=%d num_experts=%zu "
                     "ffn_ops{moe_gate:%llu,moe_forward:%llu,dense_ffn_matmul:%llu}\n",
                     is_prefill_batch ? "prefill" : "decode", layer, l.is_moe ? 1 : 0,
                     l.Get(model_keys::kMoeGate) ? 1 : 0, l.NumExperts(),
                     static_cast<unsigned long long>(layer_moe_gate_ops[layer]),
                     static_cast<unsigned long long>(layer_moe_forward_ops[layer]),
                     static_cast<unsigned long long>(layer_dense_ffn_ops[layer]));
    }
}

bool ShouldZeroFillPrefillInputBuffer() {
    return GetWorkerRuntimeConfig().zero_fill_prefill_input_buffer;
}

bool ShouldZeroFillPrefillGraphBuffer() {
    return GetWorkerRuntimeConfig().zero_fill_prefill_graph_buffer;
}

bool ShouldZeroFillPrefillKVBlocks() {
    return GetWorkerRuntimeConfig().zero_fill_prefill_kv_blocks;
}

int ResolveQwen36PrefillChunkTokensImpl(const TransformerModel* model, const Request* req) {
    if (!model || !req) {
        return -1;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::QWEN36 || !model->arch_flags.is_hybrid_ssm ||
        model->hparams.n_experts <= 0) {
        return -1;
    }
    const densecore::env::RuntimeToggleMode mode = densecore::env::ParseRuntimeToggleMode(
        "DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", densecore::env::RuntimeToggleMode::Auto);
    if (mode == densecore::env::RuntimeToggleMode::Off) {
        return -1;
    }
    const int explicit_tokens = densecore::env::ParsePositiveEnvInt("DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS", 0);
    if (explicit_tokens > 0) {
        return explicit_tokens;
    }
    if (mode == densecore::env::RuntimeToggleMode::On) {
        return 256;
    }
    const int prompt_tokens = !req->prompt_tokens_for_cache.empty()
                                  ? static_cast<int>(req->prompt_tokens_for_cache.size())
                                  : req->prompt_token_count;
    return prompt_tokens >= 512 ? 256 : -1;
}

int PrefillThreadOverride() {
    return GetWorkerRuntimeConfig().prefill_thread_override;
}

uint64_t Fnv1aInit() {
    return 1469598103934665603ull;
}

void Fnv1aMixU64(uint64_t* hash, uint64_t value) {
    if (!hash) {
        return;
    }
    *hash ^= value;
    *hash *= 1099511628211ull;
}

uint64_t HashIntVectorSummary(const std::vector<int>& values, size_t max_items = 32) {
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(values.size()));
    const size_t limit = std::min(max_items, values.size());
    for (size_t i = 0; i < limit; ++i) {
        Fnv1aMixU64(&hash, static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
    }
    if (values.size() > limit) {
        for (size_t i = values.size() - std::min<size_t>(4, values.size()); i < values.size(); ++i) {
            Fnv1aMixU64(&hash, static_cast<uint64_t>(static_cast<uint32_t>(values[i])));
        }
    }
    return hash;
}

uint64_t HashByteSpanSummary(const void* data, size_t bytes) {
    if (!data || bytes == 0) {
        return 0;
    }
    const auto* ptr = reinterpret_cast<const uint8_t*>(data);
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(bytes));
    const size_t window = std::min<size_t>(bytes, 4096);
    for (size_t i = 0; i < window; ++i) {
        Fnv1aMixU64(&hash, ptr[i]);
    }
    if (bytes > window) {
        const size_t mid = bytes / 2;
        for (size_t i = 0; i < std::min<size_t>(256, bytes - mid); ++i) {
            Fnv1aMixU64(&hash, ptr[mid + i]);
        }
        const size_t tail_start = bytes - std::min<size_t>(256, bytes);
        for (size_t i = tail_start; i < bytes; ++i) {
            Fnv1aMixU64(&hash, ptr[i]);
        }
    }
    return hash;
}

uint64_t HashHybridSSMStateSummary(const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
    uint64_t hash = Fnv1aInit();
    Fnv1aMixU64(&hash, static_cast<uint64_t>(states.size()));
    for (const auto& state : states) {
        Fnv1aMixU64(&hash, static_cast<uint64_t>(state.conv_state.size()));
        Fnv1aMixU64(&hash, static_cast<uint64_t>(state.ssm_state.size()));
        if (!state.conv_state.empty()) {
            const size_t mid = state.conv_state.size() / 2;
            uint32_t bits = 0;
            std::memcpy(&bits, &state.conv_state[0], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.conv_state[mid], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.conv_state.back(), sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
        }
        if (!state.ssm_state.empty()) {
            const size_t mid = state.ssm_state.size() / 2;
            uint32_t bits = 0;
            std::memcpy(&bits, &state.ssm_state[0], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.ssm_state[mid], sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
            std::memcpy(&bits, &state.ssm_state.back(), sizeof(uint32_t));
            Fnv1aMixU64(&hash, bits);
        }
    }
    return hash;
}

void LogPrefillStage(const char* stage, const Request* req, const TransformerModel* model,
                     const InferenceContext* inference_ctx, const void* input_buffer, size_t input_buffer_bytes,
                     const struct ggml_tensor* output, bool is_prefill_batch, int active_threads) {
    if (!IsDeterminismBoundaryDebugEnabled() || !req || !is_prefill_batch) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    const uint64_t token_hash = HashIntVectorSummary(req->tokens);
    const uint64_t block_hash = HashIntVectorSummary(req->block_table);
    const uint64_t ssm_hash = HashHybridSSMStateSummary(req->ssm_runtime_states);
    const uint64_t input_hash = HashByteSpanSummary(input_buffer, input_buffer_bytes);
    const uint64_t graph_hash =
        inference_ctx ? HashByteSpanSummary(inference_ctx->compute_buffer, inference_ctx->compute_buffer_size) : 0;
    uint64_t output_hash = 0;
    size_t output_bytes = 0;
    if (output && output->data) {
        output_bytes = ggml_nbytes(output);
        output_hash = HashByteSpanSummary(output->data, output_bytes);
    }
    std::cerr << "[PrefillStage] stage=" << (stage ? stage : "<unknown>") << " req=" << req->id
              << " variant=" << densecore::models::ModelVariantName(descriptor.variant) << " threads=" << active_threads
              << " prompt_tokens=" << req->tokens.size() << " n_past=" << req->n_past << " token_hash=0x" << std::hex
              << token_hash << " block_hash=0x" << block_hash << " ssm_hash=0x" << ssm_hash << " input_hash=0x"
              << input_hash << " graph_hash=0x" << graph_hash << " output_hash=0x" << output_hash << std::dec
              << " input_bytes=" << input_buffer_bytes << " output_bytes=" << output_bytes << std::endl;
}

void LogDeterminismBoundary(const char* stage, const Request* req, const TransformerModel* model,
                            bool prefix_cache_allowed, bool prefix_cache_hit, bool hybrid_restore_attempted,
                            bool hybrid_restore_applied, bool chunked_prefill, bool decode_cache_active,
                            bool decode_cache_reused, bool prefill_cache_active) {
    if (!IsDeterminismBoundaryDebugEnabled() || !req) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    const uint64_t token_hash = HashIntVectorSummary(req->tokens);
    const uint64_t block_hash = HashIntVectorSummary(req->block_table);
    const uint64_t ssm_hash = HashHybridSSMStateSummary(req->ssm_runtime_states);
    std::cerr << "[DeterminismBoundary] stage=" << (stage ? stage : "<unknown>") << " req=" << req->id
              << " variant=" << densecore::models::ModelVariantName(descriptor.variant)
              << " prompt_tokens=" << req->tokens.size() << " n_past=" << req->n_past
              << " generated=" << req->generated_count << " is_prefill=" << (req->is_prefill ? 1 : 0)
              << " prefix_cache_allowed=" << (prefix_cache_allowed ? 1 : 0)
              << " prefix_cache_hit=" << (prefix_cache_hit ? 1 : 0)
              << " hybrid_restore_attempted=" << (hybrid_restore_attempted ? 1 : 0)
              << " hybrid_restore_applied=" << (hybrid_restore_applied ? 1 : 0)
              << " chunked_prefill=" << (chunked_prefill ? 1 : 0)
              << " decode_cache_active=" << (decode_cache_active ? 1 : 0)
              << " decode_cache_reused=" << (decode_cache_reused ? 1 : 0)
              << " prefill_cache_active=" << (prefill_cache_active ? 1 : 0) << " token_hash=0x" << std::hex
              << token_hash << " block_hash=0x" << block_hash << " ssm_hash=0x" << ssm_hash << std::dec << std::endl;
}

bool IsValidGgmlType(enum ggml_type type) {
    const int value = static_cast<int>(type);
    return value >= 0 && value < static_cast<int>(GGML_TYPE_COUNT);
}

bool IsValidGgmlOp(enum ggml_op op) {
    const int value = static_cast<int>(op);
    return value >= 0 && value < static_cast<int>(GGML_OP_COUNT);
}

const char* SafeGgmlTypeName(enum ggml_type type) {
    return IsValidGgmlType(type) ? ggml_type_name(type) : "<invalid-type>";
}

const char* SafeGgmlOpName(enum ggml_op op) {
    return IsValidGgmlOp(op) ? ggml_op_name(op) : "<invalid-op>";
}

std::string TensorDebugSummary(const struct ggml_tensor* tensor) {
    if (!tensor) {
        return "<null>";
    }

    std::string summary;
    summary.reserve(256);
    summary += "ptr=" + std::to_string(reinterpret_cast<uintptr_t>(tensor));
    summary += " name=";
    summary += tensor->name[0] ? tensor->name : "<unnamed>";
    summary += " op=";
    summary += SafeGgmlOpName(tensor->op);
    summary += " type=";
    summary += SafeGgmlTypeName(tensor->type);
    summary += " ne=[";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d != 0) summary += ",";
        summary += std::to_string(static_cast<long long>(tensor->ne[d]));
    }
    summary += "] nb=[";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d != 0) summary += ",";
        summary += std::to_string(static_cast<long long>(tensor->nb[d]));
    }
    summary += "]";
    return summary;
}

bool IsHybridSSMSnapshotDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_HYBRID_SSM_SNAPSHOT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsGraphNodeDumpEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GRAPH_NODES");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

int GraphNodeDumpLimit() {
    const char* env = std::getenv("DENSECORE_DEBUG_GRAPH_NODE_LIMIT");
    if (!env || env[0] == '\0') {
        return 32;
    }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0 || parsed > 4096) {
        return 32;
    }
    return static_cast<int>(parsed);
}

void DebugDumpGraphNodes(const struct ggml_cgraph* graph, const char* stage, int max_nodes = 32) {
    if (!IsGraphNodeDumpEnabled() || !graph) {
        return;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<struct ggml_cgraph*>(graph));
    const int configured_limit = GraphNodeDumpLimit();
    const int limit = std::min(n_nodes, std::max(1, std::max(max_nodes, configured_limit)));
    std::cerr << "[GraphNodeDump] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        struct ggml_tensor* node = ggml_graph_node(const_cast<struct ggml_cgraph*>(graph), i);
        std::cerr << "  node[" << i << "] " << TensorDebugSummary(node) << std::endl;
        if (node) {
            std::cerr << "    src0: " << TensorDebugSummary(node->src[0]) << std::endl;
            std::cerr << "    src1: " << TensorDebugSummary(node->src[1]) << std::endl;
            std::cerr << "    src2: " << TensorDebugSummary(node->src[2]) << std::endl;
        }
    }
}

void DebugLogHybridSSMSnapshot(const char* stage, int req_id, int cached_tokens, int block_id,
                               const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
    if (!IsHybridSSMSnapshotDebugEnabled()) {
        return;
    }
    const size_t n_layers = states.size();
    float conv0 = 0.0f;
    float ssm0 = 0.0f;
    if (!states.empty()) {
        if (!states[0].conv_state.empty()) conv0 = states[0].conv_state[0];
        if (!states[0].ssm_state.empty()) ssm0 = states[0].ssm_state[0];
    }
    std::cerr << "[HybridSSMSnapshot] stage=" << (stage ? stage : "<unknown>") << " req=" << req_id
              << " cached_tokens=" << cached_tokens << " block_id=" << block_id << " layers=" << n_layers
              << " conv0=" << conv0 << " ssm0=" << ssm0 << std::endl;
}

void ValidateMulNodesOrThrow(struct ggml_cgraph* graph, const char* stage) {
    if (!IsMulGraphValidationEnabled() || !graph) {
        return;
    }

    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }

        const bool is_binary_shape_checked_op =
            node->op == GGML_OP_ADD || node->op == GGML_OP_SUB || node->op == GGML_OP_MUL || node->op == GGML_OP_DIV;
        if (!is_binary_shape_checked_op) {
            continue;
        }

        struct ggml_tensor* src0 = node->src[0];
        struct ggml_tensor* src1 = node->src[1];
        std::string issue;
        if (!IsValidGgmlType(node->type)) {
            issue = "dst has invalid type";
        } else if (!src0 || !src1) {
            issue = "missing MUL source";
        } else if (!IsValidGgmlType(src0->type)) {
            issue = "src0 has invalid type";
        } else if (!IsValidGgmlType(src1->type)) {
            issue = "src1 has invalid type";
        } else if (!ggml_can_repeat(src1, src0)) {
            issue = "src1 cannot broadcast to src0";
        } else if (!ggml_are_same_shape(src0, node)) {
            issue = "dst shape does not match src0";
        }

        if (!issue.empty()) {
            std::cerr << "[MulGraphValidation] stage=" << (stage ? stage : "<unknown>") << " node=" << i
                      << " op=" << SafeGgmlOpName(node->op) << " issue=" << issue << std::endl;
            std::cerr << "  dst:  " << TensorDebugSummary(node) << std::endl;
            std::cerr << "  src0: " << TensorDebugSummary(src0) << std::endl;
            std::cerr << "  src1: " << TensorDebugSummary(src1) << std::endl;
            throw densecore::InvalidArgumentException("GGML MUL graph validation failed at node " + std::to_string(i) +
                                                      " (" + issue + ")");
        }
    }
}

}  // namespace

bool ResolveSamplingLogitsColumnForRequest(int token_offset, int processed_count, int output_columns,
                                           bool sampled_from_prefill, int remaining_prompt_tokens, int n_past_before,
                                           int n_past_after, int* out_last_token_idx, std::string* error) {
    return ResolveSamplingLogitsColumnForRequestImpl(token_offset, processed_count, output_columns,
                                                     sampled_from_prefill, remaining_prompt_tokens, n_past_before,
                                                     n_past_after, out_last_token_idx, error);
}

int GetDebugSamplingLogitsOffset() {
    static const int value = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SAMPLING_LOGITS_OFFSET");
        if (!env || env[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env) {
            return 0;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

int GetDebugFirstTokenSamplingLogitsOffset() {
    static const int value = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_FIRST_TOKEN_SAMPLING_LOGITS_OFFSET");
        if (!env || env[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env) {
            return 0;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

int ResolveQwen36PrefillChunkTokens(const TransformerModel* model, const Request* req) {
    return ResolveQwen36PrefillChunkTokensImpl(model, req);
}

// Worker Loop (Continuous Batching) - Uses Scheduler for batch formation
void EngineLoop(EngineState* state) {
    struct WorkContextBinder {
        explicit WorkContextBinder(InferenceWorkContext* ctx) { SetCurrentWorkContext(ctx); }
        ~WorkContextBinder() { SetCurrentWorkContext(nullptr); }
    };

    try {
        auto work_ctx = std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)>(
            CreateInferenceWorkContext(), DestroyInferenceWorkContext);
        WorkContextBinder work_ctx_binder(work_ctx.get());

        // =========================================================================
        // STEP 0: Initialize SIMD dispatch table (must be first!)
        // =========================================================================
        // OpsRegistry selects optimal kernel implementations based on CPU caps.
        // Must be called before any inference operations that use SIMD kernels.
        // =========================================================================
        if (!densecore::OpsRegistry::IsInitialized()) {
            densecore::OpsRegistry::Init();
        }
        if (state->op_registry) {
            densecore::GetCpuBackend().SetOpRegistry(state->op_registry.get());
        }

        // =========================================================================
        // THREADING MODEL (DenseCore Unified Threading):
        // =========================================================================
        // DenseCore uses a two-level parallelism strategy WITHOUT OpenMP:
        //
        // 1. Request-level parallelism: std::thread workers in worker.cpp
        //    - Each worker processes requests from the scheduler queue
        //
        // 2. Compute-level parallelism: GGML's internal thread pool
        //    - Configured via ggml_backend_cpu_set_n_threads() below
        //    - All SIMD kernels in densecore/simd/simd_ops.h are single-threaded
        //    - GGML callbacks invoke kernels with (ith, nth) for work partitioning
        //
        // OpenMP is NOT used to avoid thread oversubscription (nested parallelism).
        // =========================================================================

        // Resolve model once at loop start (for performance)
        ModelEntry* model_entry = state->GetDefaultModel();

        if (!model_entry) {
            LOG_CRITICAL("FATAL: No model loaded. EngineLoop exiting.");
            return;
        }
        ScopedFastPathRuntimeConfigBinder fast_path_runtime_config_binder(&state->fast_path_config);
        TransformerModel* current_model = model_entry->model.get();
        PagedKVCache* current_kv_cache = model_entry->kv_cache.get();

        // =========================================================================
        // GGML BACKEND THREAD CONFIGURATION (Critical for performance)
        // =========================================================================
        // This is the key call that enables multi-threaded GGML compute!
        // Without this, GGML defaults to single-threaded execution.
        // GGML BACKEND THREAD CONFIGURATION (Critical for performance)
        // =========================================================================
        // This is the key call that enables multi-threaded GGML compute!
        // Without this, GGML defaults to single-threaded execution.
        int n_threads = state->n_threads;
        const ArmComputeAffinityPolicy arm_compute_affinity = ResolveArmComputeAffinityPolicy();
        if (n_threads <= 0) {
            if (!arm_compute_affinity.core_ids.empty()) {
                n_threads = static_cast<int>(arm_compute_affinity.core_ids.size());
            }
            // Use PHYSICAL cores by default to avoid Hyperthreading thrashing
            // on AVX workloads where execution units are shared.
            if (n_threads <= 0) {
                auto& topo = densecore::HardwareTopology::GetInstance();
                int num_physical = topo.GetPhysicalCoreCount();
                // Fallback if detection fails
                if (num_physical <= 0)
                    n_threads = densecore::simd::GetNumCores() / 2;
                else
                    n_threads = num_physical;
            }

            if (n_threads < 1) n_threads = 1;
        }

        // Persist resolved thread count for downstream paths.
        state->n_threads = n_threads;
        const int base_threads = n_threads;
        int physical_core_count = !arm_compute_affinity.core_ids.empty()
                                      ? static_cast<int>(arm_compute_affinity.core_ids.size())
                                      : densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
        if (physical_core_count <= 0) {
            physical_core_count = base_threads;
        }
        // Tracks the configured thread count for the CPU backend handle only.
        int last_set_threads = -1;
        if (!arm_compute_affinity.core_ids.empty()) {
            LOG_INFO("GGML backend configured: {} threads (ARM cluster policy: {})", n_threads,
                     arm_compute_affinity.label);
        } else {
            LOG_INFO("GGML backend configured: {} threads (Physical Cores prioritized)", n_threads);
        }


        // =========================================================================
        // THREAD PINNING POLICY (Critical for avoiding deadlocks)
        // =========================================================================
        // The orchestration/control thread (this thread) should NEVER be pinned!
        // Aggressive pinning was causing deadlocks on consumer hardware (e.g.,
        // i7-10870H) where the main thread and compute workers contended for Core
        // 0.
        //
        // Strategy:
        // - Orchestration thread: Let OS schedule freely (unpinned)
        // - Compute threads: Pin via SetupComputeThreadAffinity (done below)
        // =========================================================================
        {
            auto& topo = densecore::HardwareTopology::GetInstance();
            LOG_INFO("Orchestration thread unpinned (OS Scheduled). NUMA nodes detected: {}", topo.GetNumaNodeCount());
            // NOTE: Do NOT call PinCurrentThread() or PinCurrentThreadToNumaNode()
            // for this thread. Only compute worker threads should be pinned.
        }

        // Setup compute thread affinity for GGML workers
        // Use SCATTER policy to spread threads across physical cores
        {
            auto& topo = densecore::HardwareTopology::GetInstance();
            if (!arm_compute_affinity.core_ids.empty()) {
                topo.SetupComputeThreadAffinityFromCoreIds(arm_compute_affinity.core_ids, n_threads,
                                                           densecore::PinningPolicy::SCATTER);
            } else {
                int target_node = (state->numa_node_id >= 0) ? state->numa_node_id : 0;
                int physical_cores = topo.GetPhysicalCoreCount(target_node);

                if (n_threads > 0 && physical_cores > 0) {
                    // Use simple SCATTER policy - spread threads across all physical cores
                    topo.SetupComputeThreadAffinity(target_node, n_threads, densecore::PinningPolicy::SCATTER);
                }
            }
        }

        // Initialize persistent compute buffer (eliminates malloc/free per
        // iteration)
        state->InitComputeBuffer();

        // Mapping: scheduler seq_id -> Request*
        std::unordered_map<int, Request*> seq_to_request;
        struct DecodeGraphCacheKey {
            int batch_size = 0;
            int threads = 0;
            uintptr_t model_id = 0;
            int arch_id = 0;
            int cache_type_id = -1;
            uint64_t feature_flags = 0;

            bool operator==(const DecodeGraphCacheKey& other) const {
                return batch_size == other.batch_size && threads == other.threads && model_id == other.model_id &&
                       arch_id == other.arch_id && cache_type_id == other.cache_type_id &&
                       feature_flags == other.feature_flags;
            }
        };
        struct DecodeGraphCacheKeyHash {
            size_t operator()(const DecodeGraphCacheKey& key) const {
                size_t h = 1469598103934665603ull;
                auto mix = [&h](uint64_t v) {
                    h ^= static_cast<size_t>(v);
                    h *= static_cast<size_t>(1099511628211ull);
                };
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.batch_size)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.threads)));
                mix(static_cast<uint64_t>(key.model_id));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.arch_id)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.cache_type_id)));
                mix(key.feature_flags);
                return h;
            }
        };
        struct DecodeGraphCacheEntry {
            std::vector<uint8_t> ctx_buffer;
            struct ggml_context* ctx = nullptr;
            struct ggml_cgraph* graph = nullptr;
            struct ggml_tensor* output = nullptr;
            struct ggml_tensor* embd_inp = nullptr;
            struct ggml_tensor* pos = nullptr;
            bool verified_paged_decode_op = false;
            std::list<DecodeGraphCacheKey>::iterator lru_it;
        };
        struct PrefillGraphCacheKey {
            int batch_size = 0;
            int tokens = 0;
            int n_past = 0;
            int threads = 0;
            uintptr_t model_id = 0;
            int arch_id = 0;
            int cache_type_id = -1;
            uint64_t feature_flags = 0;

            bool operator==(const PrefillGraphCacheKey& other) const {
                return batch_size == other.batch_size && tokens == other.tokens && n_past == other.n_past &&
                       threads == other.threads && model_id == other.model_id && arch_id == other.arch_id &&
                       cache_type_id == other.cache_type_id && feature_flags == other.feature_flags;
            }
        };
        struct PrefillGraphCacheKeyHash {
            size_t operator()(const PrefillGraphCacheKey& key) const {
                size_t h = 1469598103934665603ull;
                auto mix = [&h](uint64_t v) {
                    h ^= static_cast<size_t>(v);
                    h *= static_cast<size_t>(1099511628211ull);
                };
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.batch_size)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.tokens)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.n_past)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.threads)));
                mix(static_cast<uint64_t>(key.model_id));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.arch_id)));
                mix(static_cast<uint64_t>(static_cast<uint32_t>(key.cache_type_id)));
                mix(key.feature_flags);
                return h;
            }
        };
        struct PrefillGraphCacheEntry {
            std::vector<uint8_t> ctx_buffer;
            struct ggml_context* ctx = nullptr;
            struct ggml_cgraph* graph = nullptr;
            struct ggml_tensor* output = nullptr;
            struct ggml_tensor* embd_inp = nullptr;
            struct ggml_tensor* pos = nullptr;
            size_t ctx_bytes = 0;
            std::list<PrefillGraphCacheKey>::iterator lru_it;
        };

        std::unordered_map<DecodeGraphCacheKey, DecodeGraphCacheEntry, DecodeGraphCacheKeyHash> decode_graph_cache;
        std::list<DecodeGraphCacheKey> decode_graph_lru;
        std::unordered_set<DecodeGraphCacheKey, DecodeGraphCacheKeyHash> decode_graph_uncacheable;
        std::list<DecodeGraphCacheKey> decode_graph_uncacheable_lru;
        std::unordered_map<PrefillGraphCacheKey, PrefillGraphCacheEntry, PrefillGraphCacheKeyHash> prefill_graph_cache;
        std::list<PrefillGraphCacheKey> prefill_graph_lru;
        size_t prefill_graph_cache_bytes = 0;

        auto free_decode_graph_entry = [](DecodeGraphCacheEntry* entry) {
            if (!entry) return;
            if (entry->ctx) {
                ggml_free(entry->ctx);
                entry->ctx = nullptr;
            }
            entry->graph = nullptr;
            entry->output = nullptr;
            entry->embd_inp = nullptr;
            entry->pos = nullptr;
            entry->ctx_buffer.clear();
        };
        auto clear_decode_graph_cache = [&]() {
            for (auto& kv : decode_graph_cache) {
                free_decode_graph_entry(&kv.second);
            }
            decode_graph_cache.clear();
            decode_graph_lru.clear();
            decode_graph_uncacheable.clear();
            decode_graph_uncacheable_lru.clear();
        };
        auto clear_prefill_graph_cache = [&]() {
            for (auto& kv : prefill_graph_cache) {
                if (kv.second.ctx) {
                    ggml_free(kv.second.ctx);
                    kv.second.ctx = nullptr;
                }
                kv.second.graph = nullptr;
                kv.second.output = nullptr;
                kv.second.embd_inp = nullptr;
                kv.second.pos = nullptr;
                kv.second.ctx_buffer.clear();
                kv.second.ctx_bytes = 0;
            }
            prefill_graph_cache.clear();
            prefill_graph_lru.clear();
            prefill_graph_cache_bytes = 0;
        };
        auto touch_decode_graph_entry = [&](DecodeGraphCacheEntry* entry) {
            if (!entry) return;
            decode_graph_lru.splice(decode_graph_lru.begin(), decode_graph_lru, entry->lru_it);
            entry->lru_it = decode_graph_lru.begin();
        };
        auto touch_prefill_graph_entry = [&](PrefillGraphCacheEntry* entry) {
            if (!entry) return;
            prefill_graph_lru.splice(prefill_graph_lru.begin(), prefill_graph_lru, entry->lru_it);
            entry->lru_it = prefill_graph_lru.begin();
        };
        auto reap_finished_requests = [&]() {
            std::vector<Request*> finished_requests;
            {
                std::lock_guard<std::mutex> lock(state->active_mu);
                size_t write_idx = 0;
                for (Request* req : state->active_requests) {
                    if (req && !req->finished) {
                        state->active_requests[write_idx++] = req;
                    } else if (req) {
                        finished_requests.push_back(req);
                    }
                }
                state->active_requests.resize(write_idx);
            }

            if (finished_requests.empty()) {
                return;
            }

            state->metrics.active_requests.fetch_sub(static_cast<int>(finished_requests.size()),
                                                     std::memory_order_relaxed);
            for (Request* req : finished_requests) {
                state->request_pool.Release(req);
            }

            std::lock_guard<std::mutex> cv_lock(state->cv_mu);
            state->queue_cv.notify_one();
        };
        auto reset_empty_schedule_watchdog = [&]() {
            {
                std::lock_guard<std::mutex> active_lock(state->active_mu);
                for (Request* req : state->active_requests) {
                    if (!req || req->finished) {
                        continue;
                    }
                    req->empty_schedule_stall_count = 0;
                }
            }
            std::lock_guard<std::mutex> watchdog_lock(state->scheduler_watchdog_mu);
            state->empty_schedule_watchdog.consecutive_loops = 0;
            state->empty_schedule_watchdog.first_seen = std::chrono::steady_clock::time_point();
            state->empty_schedule_watchdog.last_seen = std::chrono::steady_clock::time_point();
            state->empty_schedule_watchdog.last_log = std::chrono::steady_clock::time_point();
        };
        static constexpr auto kEmptyScheduleLogEvery = std::chrono::seconds(1);

        while (state->status != EngineStatus::STOPPED) {
            const bool global_bench_fast_path = IsBenchmarkFastPathEnabled();
            const bool global_direct_callback = ShouldUseDirectResultCallbacks(global_bench_fast_path);
            // 1. Fetch new requests and register with scheduler
            // =========================================================================
            // WAIT FOR WORK (Spin-then-CV for low-latency wakeup)
            // =========================================================================
            // Spin for ~50k iterations (~10-50us on modern CPUs) checking for work
            // before falling through to the condition variable. Uses iteration
            // count instead of steady_clock::now() to avoid VDSO/syscall overhead
            // in the hot loop. _mm_pause (x86) / yield (ARM) reduces power and
            // avoids starving sibling hyperthreads.
            // =========================================================================
            {
                static constexpr int kSpinIterations = 50000;
                bool found_work = false;
                for (int spin_i = 0; spin_i < kSpinIterations; ++spin_i) {
                    if (state->status != EngineStatus::RUNNING) {
                        found_work = true;  // Exit spin to handle DRAINING/STOPPED
                        break;
                    }
                    if (!state->pending_requests.Empty() ||
                        state->metrics.active_requests.load(std::memory_order_acquire) > 0) {
                        found_work = true;
                        break;
                    }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                    _mm_pause();
#elif defined(__aarch64__)
                    __asm__ volatile("yield");
#else
                    std::this_thread::yield();
#endif
                }

                // If spin didn't find work, fall through to CV wait (lost-wakeup safe)
                if (!found_work) {
                    std::unique_lock<std::mutex> lock(state->cv_mu);
                    state->queue_cv.wait(lock, [state]() {
                        return !state->pending_requests.Empty() ||
                               state->metrics.active_requests.load(std::memory_order_acquire) > 0 ||
                               state->status != EngineStatus::RUNNING;
                    });
                }
            }

            // std::cerr << "[DEBUG] EngineLoop: Woke up" << std::endl;

            // Check if draining and all work is done
            if (state->status == EngineStatus::DRAINING) {
                std::lock_guard<std::mutex> active_lock(state->active_mu);
                if (state->active_requests.empty() && state->pending_requests.Empty()) {
                    break;  // Exit: draining complete, no more work
                }
            }

            {
                // Note: std::mutex lock/unlock provides sufficient memory ordering.
                // No atomic fence needed - cv.wait() holding cv_mu provides acquire
                // semantics.

                while (state->status == EngineStatus::RUNNING) {
                    Request* req = state->pending_requests.Pop();
                    if (!req) break;  // Queue empty

                    state->RemovePendingRequest(req->id);

                    // std::cerr << "[DEBUG] EngineLoop: Popped request " << req->id
                    //           << std::endl;

                    if (state->ConsumePendingCancellation(req->id)) {
                        req->cancelled.store(true, std::memory_order_relaxed);
                    }

                    // Check if cancelled before even starting
                    if (req->cancelled.load(std::memory_order_relaxed)) {
                        LOG_INFO("Skipping cancelled request: {}", req->id);
                        state->request_pool.Release(req);
                        continue;
                    }


                    // =========================================================================
                    // UNIVERSAL GRAPH EXECUTION DISPATCH
                    // =========================================================================
                    if (req->is_graph_execution) {
                        LOG_INFO("Worker processing graph request: {}", req->graph_name);

                        try {
                            // 1. Get Graph Builder
                            auto builder = densecore::GraphRegistry::Instance().GetBuilder(req->graph_name);
                            if (!builder) {
                                throw std::runtime_error("Graph builder not found for architecture: " +
                                                         req->graph_name);
                            }

                            // 2. Wrap Inputs (Zero-Copy)
                            std::vector<densecore::Tensor> input_tensors;
                            input_tensors.reserve(req->graph_inputs.size());

                            for (const auto& in : req->graph_inputs) {
                                std::vector<int64_t> shape;
                                for (int i = 0; i < in.ndim; ++i) shape.push_back(in.shape[i]);

                                // Create tensor wrapping external data (zero-copy)
                                // Note: const_cast is safe here because:
                                // 1. Graph building only reads input tensor metadata (shape, dtype)
                                // 2. Graph execution reads input data but never writes to it
                                // 3. Tensor::Wrap is designed for both input (read-only) and output (mutable) buffers
                                // The underlying data is treated as immutable during the entire inference pipeline.
                                densecore::Tensor t = densecore::Tensor::Wrap(const_cast<void*>(in.data), shape,
                                                                              static_cast<densecore::DType>(in.dtype));
                                input_tensors.push_back(std::move(t));
                            }

                            // 3. Build Graph
                            auto graph = builder->Build(input_tensors, req->graph_name);
                            if (!graph) {
                                throw std::runtime_error("Graph build failed");
                            }

                            // 4. Execute Graph
                            densecore::DeviceType graph_device = densecore::OpRegistry::DetectDeviceType();
                            if (state->backend_selector) {
                                auto clamp_dim_to_int = [](int64_t value, int fallback) -> int {
                                    if (value <= 0) return fallback;
                                    if (value > std::numeric_limits<int>::max()) {
                                        return std::numeric_limits<int>::max();
                                    }
                                    return static_cast<int>(value);
                                };

                                int inferred_batch = 1;
                                int inferred_seq = 1;
                                if (!req->graph_inputs.empty()) {
                                    const auto& first_input = req->graph_inputs.front();
                                    if (first_input.ndim >= 1) {
                                        inferred_batch = clamp_dim_to_int(first_input.shape[0], 1);
                                    }
                                    if (first_input.ndim >= 2) {
                                        inferred_seq = clamp_dim_to_int(first_input.shape[1], 1);
                                    }
                                }

                                densecore::BatchContext sel_ctx = densecore::BatchContext::Prefill(
                                    std::max(1, inferred_batch), std::max(1, inferred_seq));
                                densecore::BackendCandidates candidates;
                                candidates.cpu_backend =
                                    current_model->cpu_backend ? current_model->cpu_backend : current_model->backend;
                                candidates.gpu_backend = current_model->metal_backend;
#ifdef __APPLE__
                                if (state->hybrid_ane_backend && current_model->metal_backend) {
                                    candidates.accelerator_backend = current_model->metal_backend;
                                }
#endif

                                const densecore::BackendSelection selection =
                                    state->backend_selector->Select(sel_ctx, candidates);
                                graph_device = selection.preferred_device;
                            }

                            densecore::GraphExecutor executor;
                            executor.Execute(*graph, graph_device);

                            // 5. Prepare Outputs
                            std::vector<DenseCoreTensorOutput> outputs;
                            const auto& output_indices = graph->GetOutputIndices();
                            auto& all_tensors = graph->MutableTensors();

                            outputs.reserve(output_indices.size());
                            for (size_t idx : output_indices) {
                                const auto& t = all_tensors[idx];
                                DenseCoreTensorOutput out;
                                out.name = "output";
                                out.data = t.data;
                                out.ndim = t.ndim;
                                out.dtype = static_cast<DenseCoreDType>(t.dtype);
                                std::copy(t.shape.begin(), t.shape.begin() + t.ndim, out.shape);
                                outputs.push_back(out);
                            }

                            // 6. Invoke Callback
                            if (req->graph_callback) {
                                req->graph_callback(outputs.data(), outputs.size(), req->user_data);
                            }

                        } catch (const std::exception& e) {
                            LOG_ERROR("Graph execution error: {}", e.what());
                        }

                        req->finished = true;
                        state->metrics.completed_requests++;
                        state->request_pool.Release(req);
                        continue;
                    }
                    req->start_time = std::chrono::steady_clock::now();
                    auto queue_wait_us =
                        std::chrono::duration_cast<std::chrono::microseconds>(req->start_time - req->arrival_time)
                            .count();
                    state->metrics.RecordQueueWait(queue_wait_us);

                    // Tokens are pre-tokenized in SubmitRequest - assert this
                    // invariant
                    if (req->tokens.empty()) {
                        LOG_CRITICAL("FATAL: Request {} has no tokens (should be pre-tokenized)", req->id);
                        req->finished = true;
                        req->decode_finish_cause = DecodeFinishCause::MissingTokens;
                        FinalizeDecodeSilentFinishReason(req);
                        LogRequestDecodeSummary(req, current_model);
                        state->metrics.failed_requests++;
                        // Push error event to callback queue (instead of direct callback)
                        EmitRequestResult(state, req, "Error: Missing tokens", -1, true, true, global_direct_callback);
                        state->request_pool.Release(req);
                        continue;
                    }
                    state->metrics.total_prompt_tokens += req->tokens.size();

                    // Hybrid SSM models must keep recurrent state per request, not
                    // globally on the shared model, otherwise batched execution
                    // cross-contaminates sequences and forces single-request mode.
                    EnsureRequestHybridSSMRuntimeState(current_model, req);
                    LogDeterminismBoundary("after_tokenization", req, current_model,
                                           /*prefix_cache_allowed=*/IsPrefixCacheAllowedForModel(current_model),
                                           /*prefix_cache_hit=*/false,
                                           /*hybrid_restore_attempted=*/false,
                                           /*hybrid_restore_applied=*/false,
                                           /*chunked_prefill=*/false,
                                           /*decode_cache_active=*/false,
                                           /*decode_cache_reused=*/false,
                                           /*prefill_cache_active=*/false);

                    // Register with scheduler (blocks allocated by scheduler)
                    const int qwen36_prefill_chunk_tokens = ResolveQwen36PrefillChunkTokens(current_model, req);
                    req->prefill_chunk_tokens_effective = std::max(0, qwen36_prefill_chunk_tokens);
                    int seq_id = state->scheduler->AddRequest(
                        req->id, req->tokens.size(), req->max_tokens, req->priority, &req->tokens,
                        /*allow_chunked_prefill=*/!req->is_embedding,
                        /*require_hybrid_ssm_prefix_snapshot=*/current_model && current_model->arch_flags.is_hybrid_ssm,
                        qwen36_prefill_chunk_tokens);

                    if (seq_id < 0) {
                        // Scheduler rejected (e.g., queue full or impossible non-chunked prefill)
                        LOG_WARN("Scheduler rejected request {}", req->id);
                        req->finished = true;
                        req->decode_finish_cause = DecodeFinishCause::SchedulerRejected;
                        FinalizeDecodeSilentFinishReason(req);
                        LogRequestDecodeSummary(req, current_model);
                        state->metrics.failed_requests++;
                        // Push error event to callback queue (instead of direct callback)
                        EmitRequestResult(state, req,
                                          "Error: Request cannot be scheduled (queue full or prefill budget exceeded)",
                                          -1, true, true, global_direct_callback);
                        state->request_pool.Release(req);
                        continue;
                    }

                    // Store seq_id in request for O(1) cleanup lookup
                    req->seq_id = seq_id;

                    // Store mapping and add to active list
                    seq_to_request[seq_id] = req;
                    {
                        std::lock_guard<std::mutex> active_lock(state->active_mu);
                        state->active_requests.push_back(req);
                    }
                    req->empty_schedule_stall_count = 0;
                    req->last_progress_time = req->start_time;
                    state->metrics.active_requests++;
                    state->metrics.total_requests++;
                    LOG_TRACE("Request {} moved to active", req->id);
                }
            }

            if (state->metrics.active_requests.load(std::memory_order_acquire) == 0) continue;

            // Check if we should exit (DRAINING and no more active work)
            {
                std::lock_guard<std::mutex> lock(state->active_mu);
                if (state->status == EngineStatus::DRAINING && state->active_requests.empty()) {
                    break;
                }
            }

            // 2. Process cancelled/finished requests first
            std::vector<int> cancelled_seq_ids;
            {
                std::lock_guard<std::mutex> lock(state->active_mu);
                auto it = state->active_requests.begin();
                while (it != state->active_requests.end()) {
                    Request* req = *it;
                    if (req->cancelled.load(std::memory_order_relaxed) && !req->finished) {
                        LOG_INFO("Cancelling active request: {}", req->id);
                        req->finished = true;
                        req->decode_finish_cause = DecodeFinishCause::RequestCanceled;
                        FinalizeDecodeSilentFinishReason(req);
                        LogRequestDecodeSummary(req, current_model);
                        state->metrics.failed_requests++;
                        EmitRequestResult(state, req, "Error: request canceled", -1, true, true,
                                          global_direct_callback);
                        // Defer scheduler removal to avoid lock-order inversion
                        if (req->seq_id >= 0) {
                            cancelled_seq_ids.push_back(req->seq_id);
                        }
                        // Free blocks if allocated
                        current_kv_cache->block_manager->Free(req->block_table);
                        req->block_table.clear();
                        {
                            std::lock_guard<std::mutex> lk(req->mu);
                            req->cv.notify_all();
                        }
                    }
                    ++it;
                }
            }
            for (int seq_id : cancelled_seq_ids) {
                state->scheduler->RemoveRequest(seq_id, false);
                seq_to_request.erase(seq_id);
            }

            // 3. Query Scheduler for next batch
            //    Single-request fast-path: for one active request with no queued or
            //    waiting peers, bypass scheduler iteration bookkeeping and build a
            //    direct schedule.
            densecore::SchedulerOutput sched_output;
            bool used_single_request_fast_path = false;
            auto flush_scheduler_progress = [&](const std::vector<Request*>& requests) {
                if (global_bench_fast_path || !state->scheduler) return;
                std::vector<std::pair<int, int>> updates;
                updates.reserve(requests.size());
                for (Request* request : requests) {
                    if (!request || request->seq_id < 0 || request->pending_scheduler_progress <= 0 ||
                        request->finished) {
                        continue;
                    }
                    updates.emplace_back(request->seq_id, request->pending_scheduler_progress);
                    request->pending_scheduler_progress = 0;
                }
                if (!updates.empty()) {
                    state->scheduler->UpdateProgressBatch(updates);
                }
            };
            const bool single_request_fast_path_enabled = IsSingleRequestFastPathEnabled();
            if ((global_bench_fast_path || single_request_fast_path_enabled) && state->pending_requests.Empty()) {
                Request* single_req = nullptr;
                {
                    std::lock_guard<std::mutex> lock(state->active_mu);
                    if (state->active_requests.size() == 1) {
                        single_req = state->active_requests.front();
                    }
                }

                const auto sched_stats =
                    state->scheduler ? state->scheduler->GetStats() : densecore::Scheduler::Stats{};
                const bool scheduler_single_tenant =
                    !state->scheduler || (sched_stats.waiting_count == 0 && sched_stats.swapped_count == 0 &&
                                          sched_stats.running_count <= 1);

                if (scheduler_single_tenant && single_req && !single_req->finished && !single_req->is_embedding &&
                    !single_req->is_swapped && !single_req->cancelled.load(std::memory_order_relaxed) &&
                    single_req->seq_id >= 0 && !single_req->tokens.empty() &&
                    !ShouldBypassSingleRequestFastPathForLongHybridSSM(current_model, single_req)) {
                    const int tokens_to_process =
                        single_req->is_prefill ? static_cast<int>(single_req->tokens.size()) : 1;
                    if (tokens_to_process > 0) {
                        const int blocks_needed =
                            (single_req->n_past + tokens_to_process + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        if (static_cast<int>(single_req->block_table.size()) < blocks_needed) {
                            const int missing_blocks = blocks_needed - static_cast<int>(single_req->block_table.size());
                            auto new_blocks = current_kv_cache->block_manager->Allocate(missing_blocks);
                            if (static_cast<int>(new_blocks.size()) == missing_blocks) {
                                single_req->block_table.insert(single_req->block_table.end(), new_blocks.begin(),
                                                               new_blocks.end());
                            }
                        }

                        if (static_cast<int>(single_req->block_table.size()) >= blocks_needed) {
                            if (single_req->is_prefill) {
                                sched_output.prefill_seq_ids.push_back(single_req->seq_id);
                                sched_output.num_prefill_tokens = tokens_to_process;
                            } else {
                                sched_output.decode_seq_ids.push_back(single_req->seq_id);
                                sched_output.num_decode_tokens = 1;
                            }
                            sched_output.total_tokens = tokens_to_process;
                            sched_output.batch_context_len = single_req->n_past;
                            used_single_request_fast_path = true;
                        }
                    }
                }
            }

            if (!used_single_request_fast_path) {
                std::vector<Request*> requests_to_flush;
                {
                    std::lock_guard<std::mutex> lock(state->active_mu);
                    requests_to_flush = state->active_requests;
                }
                flush_scheduler_progress(requests_to_flush);
                // std::cerr << "[DEBUG] EngineLoop: Calling Scheduler->Schedule()"
                //           << std::endl;
                sched_output = state->scheduler->Schedule();
            }
            // std::cerr << "[DEBUG] EngineLoop: Schedule returned "
            //           << sched_output.prefill_seq_ids.size() << " prefill, "
            //           << sched_output.decode_seq_ids.size() << " decode" <<
            //           std::endl;

            // 4. Handle scheduler output: block allocations
            if (!used_single_request_fast_path) {
                for (const auto& alloc : sched_output.new_block_allocations) {
                    int seq_id = alloc.first;
                    const std::vector<int>& block_ids = alloc.second;
                    auto req_it = seq_to_request.find(seq_id);
                    if (req_it != seq_to_request.end()) {
                        Request* req = req_it->second;
                        // Append new blocks to request's block table
                        req->block_table.insert(req->block_table.end(), block_ids.begin(), block_ids.end());
                    }
                }
            }

            const bool prefix_cache_allowed = IsPrefixCacheAllowedForModel(current_model);
            std::unordered_set<int> determinism_prefix_hit_reqs;
            std::unordered_set<int> determinism_hybrid_restore_attempt_reqs;
            std::unordered_set<int> determinism_hybrid_restore_applied_reqs;
            std::unordered_set<int> determinism_chunked_prefill_reqs;

            // 4a. Handle scheduler output: prefix cache hits
            if (!used_single_request_fast_path && prefix_cache_allowed) {
                for (const auto& hit : sched_output.prefix_cache_hits) {
                    auto req_it = seq_to_request.find(hit.seq_id);
                    if (req_it != seq_to_request.end()) {
                        Request* req = req_it->second;
                        determinism_prefix_hit_reqs.insert(req->id);
                        // Append shared blocks from prefix cache
                        req->block_table.insert(req->block_table.end(), hit.cached_block_ids.begin(),
                                                hit.cached_block_ids.end());

                        // Advance n_past and adjust tokens to process
                        req->n_past += hit.cached_tokens;
                        if (hit.cached_tokens > 0 && hit.cached_tokens <= (int)req->tokens.size()) {
                            req->tokens.erase(req->tokens.begin(), req->tokens.begin() + hit.cached_tokens);
                        }
                        req->registered_prefix_blocks =
                            std::max(req->registered_prefix_blocks, hit.cached_tokens / BLOCK_SIZE);

                        if (current_model && current_model->arch_flags.is_hybrid_ssm && !hit.cached_block_ids.empty() &&
                            !req->ssm_runtime_states.empty()) {
                            std::vector<TransformerModel::SSMSequenceRuntimeState> snapshot;
                            const int snapshot_block = hit.cached_block_ids.back();
                            const int expected_conv =
                                current_model->ssm_inner_size +
                                2 * current_model->ssm_group_count * current_model->ssm_state_size;
                            const int expected_head_dim =
                                current_model->ssm_inner_size / std::max(1, current_model->ssm_time_step_rank);
                            const size_t expected_conv_elems =
                                TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(
                                    expected_conv, current_model->ssm_conv_kernel);
                            const size_t expected_ssm_elems =
                                TransformerModel::SSMSequenceRuntimeState::ExpectedStateElements(
                                    current_model->ssm_time_step_rank, expected_head_dim,
                                    current_model->ssm_state_size);
                            auto snapshot_shape_ok =
                                [&](const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
                                    if (hit.cached_tokens <= 0 || hit.cached_tokens % BLOCK_SIZE != 0) {
                                        return false;
                                    }
                                    if (hit.cached_block_ids.empty() || snapshot_block != hit.cached_block_ids.back()) {
                                        return false;
                                    }
                                    if (states.size() != req->ssm_runtime_states.size()) {
                                        return false;
                                    }
                                    for (const auto& state : states) {
                                        if (state.conv_state.size() != expected_conv_elems ||
                                            state.ssm_state.size() != expected_ssm_elems) {
                                            return false;
                                        }
                                    }
                                    return true;
                                };
                            determinism_hybrid_restore_attempt_reqs.insert(req->id);
                            if (!IsHybridSSMSnapshotRestoreAllowedForModel(current_model)) {
                                if (IsHybridSSMSnapshotDebugEnabled()) {
                                    std::cerr << "[HybridSSMSnapshot] restore_disabled req=" << req->id
                                              << " cached_tokens=" << hit.cached_tokens
                                              << " block_id=" << snapshot_block << std::endl;
                                }
                            } else if (current_kv_cache->block_manager->LoadHybridSSMSnapshotForBlock(snapshot_block,
                                                                                                      &snapshot) &&
                                       snapshot_shape_ok(snapshot)) {
                                DebugLogHybridSSMSnapshot("restore_before", req->id, hit.cached_tokens, snapshot_block,
                                                          snapshot);
                                req->ssm_runtime_states = std::move(snapshot);
                                determinism_hybrid_restore_applied_reqs.insert(req->id);
                                DebugLogHybridSSMSnapshot("restore_after", req->id, hit.cached_tokens, snapshot_block,
                                                          req->ssm_runtime_states);
                            } else if (!snapshot.empty()) {
                                LOG_WARN("Discarding hybrid SSM snapshot for req {}: shape mismatch on block {}",
                                         req->id, snapshot_block);
                            } else if (IsHybridSSMSnapshotDebugEnabled()) {
                                std::cerr << "[HybridSSMSnapshot] restore_miss req=" << req->id
                                          << " cached_tokens=" << hit.cached_tokens << " block_id=" << snapshot_block
                                          << std::endl;
                            }
                        }
                        LogDeterminismBoundary(
                            "after_prefix_restore", req, current_model, prefix_cache_allowed,
                            /*prefix_cache_hit=*/true,
                            /*hybrid_restore_attempted=*/determinism_hybrid_restore_attempt_reqs.count(req->id) != 0,
                            /*hybrid_restore_applied=*/determinism_hybrid_restore_applied_reqs.count(req->id) != 0,
                            /*chunked_prefill=*/false,
                            /*decode_cache_active=*/false,
                            /*decode_cache_reused=*/false,
                            /*prefill_cache_active=*/false);
                        LOG_INFO("Prefix cache hit for req {}: skipped {} tokens.", req->id, hit.cached_tokens);
                    }
                }
            }

            // 4b. Restore swapped sequences before building the batch
            if (!used_single_request_fast_path) {
                for (int seq_id : sched_output.swap_in_seq_ids) {
                    auto req_it = seq_to_request.find(seq_id);
                    if (req_it == seq_to_request.end()) {
                        continue;
                    }
                    Request* req = req_it->second;
                    if (!req->swap_state.active) {
                        continue;
                    }

                    if ((int)req->block_table.size() != req->swap_state.num_blocks) {
                        LOG_ERROR("Swap-in failed for req {}: block count mismatch (have {}, expected {})", req->id,
                                  req->block_table.size(), req->swap_state.num_blocks);
                        current_kv_cache->block_manager->Free(req->block_table);
                        req->block_table.clear();
                        req->swap_state.Clear();
                        req->is_swapped = false;
                        req->finished = true;
                        state->metrics.failed_requests++;
                        EmitRequestResult(state, req, "Error: Swap-in failed", -1, true, true, global_direct_callback);
                        if (req->seq_id >= 0) {
                            state->scheduler->RemoveRequest(req->seq_id, false);
                            seq_to_request.erase(req->seq_id);
                        }
                        continue;
                    }

                    current_kv_cache->RestoreBlocksFromHost(req->block_table, req->swap_state.k_data,
                                                            req->swap_state.v_data);
                    req->swap_state.Clear();
                    req->is_swapped = false;
                }
            }

            // 5. Handle scheduler output: freed blocks
            if (!used_single_request_fast_path) {
                for (const auto& freed : sched_output.freed_blocks) {
                    int seq_id = freed.first;
                    const std::vector<int>& block_ids = freed.second;
                    (void)seq_id;  // Scheduler already handles via BlockManager
                    current_kv_cache->block_manager->Free(block_ids);
                }
            }

            // 6. Handle scheduler output: preempted sequences (swap out)
            if (!used_single_request_fast_path) {
                for (int preempted_seq_id : sched_output.preempted_seq_ids) {
                    auto req_it = seq_to_request.find(preempted_seq_id);
                    if (req_it != seq_to_request.end()) {
                        Request* req = req_it->second;
                        if (req->block_table.empty()) {
                            continue;
                        }

                        if (req->block_table.empty()) {
                            continue;
                        }

                        LOG_INFO("Preempting request {} (seq_id={})", req->id, preempted_seq_id);
                        req->swap_state.total_tokens = req->n_past;
                        req->swap_state.num_blocks = static_cast<int>(req->block_table.size());
                        current_kv_cache->CopyBlocksToHost(req->block_table, &req->swap_state.k_data,
                                                           &req->swap_state.v_data);
                        req->swap_state.active = true;
                        req->is_swapped = true;

                        current_kv_cache->block_manager->Free(req->block_table);
                        req->block_table.clear();
                    }
                }
            }

            // =========================================================================
            // 7. Handle Empty Scheduler Output (Latency-Aware Timed Wait)
            // =========================================================================
            // If scheduler returned empty batch but active_requests exist:
            //   - Memory fragmentation may prevent scheduling
            //   - Wait with SHORT timeout (100us) for fast response to freed memory
            //
            // Using wait_for with 100us timeout instead of yield():
            //   - Prevents 100% CPU usage from busy-looping
            //   - Enables sub-millisecond latency when work arrives
            //   - Balances power efficiency with responsiveness
            // =========================================================================
            if (sched_output.IsEmpty()) {
                bool has_active = false;
                uint64_t watchdog_loops = 0;
                long stall_ms = 0;
                long oldest_idle_ms = 0;
                bool should_log = false;
                bool should_fail = false;
                std::vector<Request*> stalled_requests;
                Request* unschedulable_request = nullptr;
                const auto now = std::chrono::steady_clock::now();
                const bool scheduler_unschedulable =
                    sched_output.unschedulable_reason != densecore::SchedulerUnschedulableReason::None;
                if (scheduler_unschedulable && sched_output.diagnostic_seq_id >= 0) {
                    auto req_it = seq_to_request.find(sched_output.diagnostic_seq_id);
                    if (req_it != seq_to_request.end()) {
                        unschedulable_request = req_it->second;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(state->active_mu);
                    for (Request* req : state->active_requests) {
                        if (!req || req->finished) {
                            continue;
                        }
                        has_active = true;
                        const auto last_progress = (req->last_progress_time == std::chrono::steady_clock::time_point())
                                                       ? req->start_time
                                                       : req->last_progress_time;
                        const long idle_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
                        oldest_idle_ms = std::max(oldest_idle_ms, idle_ms);
                        if (scheduler_unschedulable) {
                            continue;
                        }
                        if (idle_ms < GetEmptyScheduleWarnAfterMs()) {
                            continue;
                        }
                        req->empty_schedule_stall_count++;
                        stalled_requests.push_back(req);
                    }
                }

                if (has_active) {
                    if (scheduler_unschedulable) {
                        should_fail = true;
                    } else if (!stalled_requests.empty()) {
                        std::lock_guard<std::mutex> watchdog_lock(state->scheduler_watchdog_mu);
                        auto& watchdog = state->empty_schedule_watchdog;
                        if (watchdog.consecutive_loops == 0) {
                            watchdog.first_seen = now;
                        }
                        watchdog.last_seen = now;
                        watchdog.consecutive_loops++;
                        watchdog_loops = watchdog.consecutive_loops;
                        stall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(watchdog.last_seen -
                                                                                         watchdog.first_seen)
                                       .count();
                        if (stall_ms >= GetEmptyScheduleFailAfterMs()) {
                            should_fail = true;
                            watchdog.failure_count++;
                            watchdog.last_log = now;
                        } else if (stall_ms >= GetEmptyScheduleWarnAfterMs() &&
                                   (watchdog.last_log == std::chrono::steady_clock::time_point() ||
                                    now - watchdog.last_log >= kEmptyScheduleLogEvery)) {
                            should_log = true;
                            watchdog.last_log = now;
                        }
                    } else {
                        reset_empty_schedule_watchdog();
                    }

                    if (should_log || should_fail) {
                        const std::string scheduler_state = state->DescribeSchedulerState();
                        const std::string active_state = state->DescribeActiveRequests();
                        if (should_fail) {
                            if (scheduler_unschedulable) {
                                LOG_ERROR("Scheduler declared active request unschedulable (reason={}). {} {}",
                                          densecore::Scheduler::DescribeUnschedulableReason(sched_output),
                                          scheduler_state, active_state);
                            } else {
                                LOG_ERROR(
                                    "Broken scheduling state detected after {} ms (loops={}, oldest_idle_ms={}). {} {}",
                                    stall_ms, watchdog_loops, oldest_idle_ms, scheduler_state, active_state);
                            }
                            std::vector<Request*> failed_requests = stalled_requests;
                            if (scheduler_unschedulable) {
                                failed_requests.clear();
                                if (unschedulable_request != nullptr) {
                                    failed_requests.push_back(unschedulable_request);
                                }
                            }
                            for (Request* req : failed_requests) {
                                if (!req || req->finished) {
                                    continue;
                                }
                                const auto last_progress =
                                    (req->last_progress_time == std::chrono::steady_clock::time_point())
                                        ? req->start_time
                                        : req->last_progress_time;
                                const long idle_ms =
                                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
                                LOG_ERROR("Failing stalled request {} (seq_id={}, prefill={}, tokens={}, n_past={}, "
                                          "generated={}, empty_loops={}, idle_ms={})",
                                          req->id, req->seq_id, req->is_prefill, req->tokens.size(), req->n_past,
                                          req->generated_count, req->empty_schedule_stall_count, idle_ms);
                                req->finished = true;
                                req->decode_finish_cause = scheduler_unschedulable
                                                               ? DecodeFinishCause::SchedulerUnschedulable
                                                               : DecodeFinishCause::SchedulerEmptyBatchStall;
                                FinalizeDecodeSilentFinishReason(req);
                                LogRequestDecodeSummary(req, current_model);
                                state->metrics.failed_requests++;
                                const std::string error_text =
                                    scheduler_unschedulable
                                        ? ("Error: Request cannot make scheduler progress (" +
                                           densecore::Scheduler::DescribeUnschedulableReason(sched_output) + ")")
                                        : "Error: Scheduler stalled while request remained active";
                                EmitRequestResult(state, req, error_text, -1, true, true, global_direct_callback);
                                if (!req->block_table.empty()) {
                                    current_kv_cache->block_manager->Free(req->block_table);
                                    req->block_table.clear();
                                }
                                if (req->seq_id >= 0) {
                                    state->scheduler->RemoveRequest(req->seq_id, false);
                                    seq_to_request.erase(req->seq_id);
                                }
                                {
                                    std::lock_guard<std::mutex> lk(req->mu);
                                    req->cv.notify_all();
                                }
                            }
                            std::lock_guard<std::mutex> cv_lock(state->cv_mu);
                            state->queue_cv.notify_one();
                        } else if (!scheduler_unschedulable) {
                            LOG_WARN("Scheduler returned empty batch with stalled active requests for {} ms (loops={}, "
                                     "oldest_idle_ms={}). {} {}",
                                     stall_ms, watchdog_loops, oldest_idle_ms, scheduler_state, active_state);
                        }
                    }

                    if (should_fail) {
                        reap_finished_requests();
                        continue;
                    }

                    // Active requests exist but scheduler couldn't schedule them
                    // Wait with VERY short timeout for latency-sensitive operation
                    std::unique_lock<std::mutex> lock(state->cv_mu);
                    const auto wait_begin = std::chrono::steady_clock::now();
                    state->queue_cv.wait_for(lock, std::chrono::microseconds(100), [state]() {
                        // Wake up if: new pending requests OR engine stopping
                        return !state->pending_requests.Empty() || state->status != EngineStatus::RUNNING;
                    });
                    const auto wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                   std::chrono::steady_clock::now() - wait_begin)
                                                                   .count());
                    std::lock_guard<std::mutex> active_lock(state->active_mu);
                    for (Request* req : state->active_requests) {
                        if (req && !req->finished) {
                            req->scheduler_wait_ns += wait_ns;
                        }
                    }
                } else {
                    reap_finished_requests();
                    reset_empty_schedule_watchdog();
                    // No active requests and scheduler empty - truly nothing to do
                    // Use slightly longer wait to save power
                    std::unique_lock<std::mutex> lock(state->cv_mu);
                    state->queue_cv.wait_for(lock, std::chrono::milliseconds(1), [state]() {
                        return !state->pending_requests.Empty() || state->status != EngineStatus::RUNNING;
                    });
                }
                continue;
            }

            reset_empty_schedule_watchdog();

            // 8. Form BatchSpec from SchedulerOutput
            BatchSpec batch;
            InferenceDependencies deps;
            std::vector<Request*> batch_requests;
            std::vector<int> batch_token_counts;
            std::unordered_map<int, int> prefill_chunk_tokens;
            const auto batch_build_begin = std::chrono::steady_clock::now();
            bool is_embedding_batch = false;
            bool first_request = true;
            batch.scheduler = state->scheduler.get();
            batch.input_kind = BatchInputKind::Tokens;
            deps.config = &InferenceConfig::Instance();
            deps.hardware_topology = &densecore::HardwareTopology::GetInstance();
            deps.backend_registry = &densecore::BackendRegistry::Instance();
            deps.op_registry = state->op_registry ? state->op_registry.get() : nullptr;
            deps.fast_path_config = &state->fast_path_config;
            deps.transformer_execution_plan = &model_entry->transformer_execution_plan;
            deps.preferred_device = densecore::DeviceType::CPU;
            deps.preferred_matmul_device = densecore::DeviceType::CPU;
            deps.preferred_attention_device = densecore::DeviceType::CPU;
            deps.preferred_norm_device = densecore::DeviceType::CPU;
            deps.mixed_operation_routing = false;
            batch.deps = &deps;

            struct BatchBuildStats {
                int scheduled_prefill_count = 0;
                int scheduled_decode_count = 0;
                int built_batch_count = 0;
                int skipped_block_writable_count = 0;
                int skipped_finished_count = 0;
                int skipped_cancelled_count = 0;
                int skipped_mixed_embedding_count = 0;
                int skipped_missing_request_count = 0;
                std::vector<Request*> stalled_requests;
                std::vector<int> missing_seq_ids;

                int scheduled_total() const { return scheduled_prefill_count + scheduled_decode_count; }
            } batch_build_stats;

            auto note_batch_build_stall = [&](Request* req) {
                if (!req) {
                    return;
                }
                req->batch_build_stall_count++;
                batch_build_stats.stalled_requests.push_back(req);
            };
            auto reset_batch_build_stall = [&](Request* req) {
                if (!req) {
                    return;
                }
                req->batch_build_stall_count = 0;
            };

            for (const auto& chunk : sched_output.prefill_chunk_info) {
                if (chunk.seq_id >= 0 && chunk.chunk_tokens > 0) {
                    prefill_chunk_tokens[chunk.seq_id] = chunk.chunk_tokens;
                }
            }

            // Prefix-cache hits can make multiple sequences share the same
            // physical KV block. Before writing new tokens, force Copy-on-Write
            // on every target block to prevent cross-sequence overwrite.
            auto ensure_request_block_writable = [&](Request* req, int token_pos) -> bool {
                if (!req || !current_kv_cache || !current_kv_cache->block_manager) {
                    return true;
                }
                if (token_pos < 0) {
                    return false;
                }

                const int block_index = token_pos / BLOCK_SIZE;
                if (block_index < 0) {
                    return false;
                }
                if (block_index >= static_cast<int>(req->block_table.size())) {
                    // Block will be allocated later if needed.
                    return true;
                }

                const int old_block_id = req->block_table[static_cast<size_t>(block_index)];
                if (old_block_id < 0) {
                    return false;
                }

                if (!current_kv_cache->block_manager->IsShared(old_block_id)) {
                    return true;
                }

                // Worker-side block table is the source of truth for decode writes.
                // Avoid scheduler table dependency here because direct decode-side
                // block growth can temporarily diverge from scheduler bookkeeping.
                const int writable_block_id = current_kv_cache->block_manager->CopyOnWrite(
                    old_block_id, [&](int src, int dst) { current_kv_cache->CopyBlockData(src, dst); });
                if (writable_block_id < 0) {
                    return false;
                }

                req->block_table[static_cast<size_t>(block_index)] = writable_block_id;
                return true;
            };

            auto ensure_request_blocks_writable = [&](Request* req, int start_pos, int token_count) -> bool {
                if (token_count <= 0) {
                    return true;
                }
                if (!req || start_pos < 0) {
                    return false;
                }

                const int end_pos = start_pos + token_count - 1;
                if (end_pos < start_pos) {
                    return false;
                }

                const int start_block = start_pos / BLOCK_SIZE;
                const int end_block = end_pos / BLOCK_SIZE;
                for (int block_idx = start_block; block_idx <= end_block; ++block_idx) {
                    const int block_start_pos = block_idx * BLOCK_SIZE;
                    if (!ensure_request_block_writable(req, block_start_pos)) {
                        return false;
                    }
                }
                return true;
            };

            // Process prefill sequences
            for (int seq_id : sched_output.prefill_seq_ids) {
                batch_build_stats.scheduled_prefill_count++;
                auto req_it = seq_to_request.find(seq_id);
                if (req_it == seq_to_request.end()) {
                    batch_build_stats.skipped_missing_request_count++;
                    batch_build_stats.missing_seq_ids.push_back(seq_id);
                    continue;
                }

                Request* req = req_it->second;
                LOG_TRACE("Prefill loop: Found request {} for seq {} (finished={})", req->id, seq_id, req->finished);
                if (req->finished) {
                    batch_build_stats.skipped_finished_count++;
                    note_batch_build_stall(req);
                    continue;
                }
                if (req->cancelled.load(std::memory_order_relaxed)) {
                    batch_build_stats.skipped_cancelled_count++;
                    note_batch_build_stall(req);
                    continue;
                }

                // Enforce homogeneous batch type
                if (first_request) {
                    is_embedding_batch = req->is_embedding;
                    first_request = false;
                } else if (req->is_embedding != is_embedding_batch) {
                    batch_build_stats.skipped_mixed_embedding_count++;
                    note_batch_build_stall(req);
                    continue;  // Skip mixed types
                }

                int tokens_to_take = static_cast<int>(req->tokens.size());
                auto chunk_it = prefill_chunk_tokens.find(seq_id);
                if (chunk_it != prefill_chunk_tokens.end()) {
                    tokens_to_take = std::min(tokens_to_take, chunk_it->second);
                }
                if (tokens_to_take <= 0) continue;

                if (!ensure_request_blocks_writable(req, req->n_past, tokens_to_take)) {
                    batch_build_stats.skipped_block_writable_count++;
                    note_batch_build_stall(req);
                    LOG_WARN("Skipping req {} in prefill: failed to ensure writable KV blocks (seq_id={}, n_past={}, "
                             "tokens={})",
                             req->id, seq_id, req->n_past, tokens_to_take);
                    continue;
                }

                if (ShouldZeroFillPrefillKVBlocks() && current_kv_cache && !req->block_table.empty()) {
                    std::vector<uint8_t> zero_k;
                    std::vector<uint8_t> zero_v;
                    current_kv_cache->CopyBlocksToHost(req->block_table, &zero_k, &zero_v);
                    std::fill(zero_k.begin(), zero_k.end(), 0);
                    std::fill(zero_v.begin(), zero_v.end(), 0);
                    current_kv_cache->RestoreBlocksFromHost(req->block_table, zero_k, zero_v);
                }

                // Save original prompt tokens once for prefix cache registration.
                if (req->prompt_tokens_for_cache.empty()) {
                    req->prompt_tokens_for_cache = req->tokens;
                    req->prompt_token_count = static_cast<int>(req->tokens.size());
                }

                std::vector<int> tokens;
                tokens.reserve(tokens_to_take);
                tokens.insert(tokens.end(), req->tokens.begin(), req->tokens.begin() + tokens_to_take);

                std::vector<int> pos;
                for (int i = 0; i < tokens_to_take; ++i) {
                    pos.push_back(req->n_past + i);
                }

                int batch_seq_idx = batch_requests.size();
                batch.tokens.insert(batch.tokens.end(), tokens.begin(), tokens.end());
                batch.pos.insert(batch.pos.end(), pos.begin(), pos.end());
                for (size_t k = 0; k < tokens.size(); ++k) {
                    batch.seq_id.push_back(batch_seq_idx);
                }
                batch.block_tables.push_back(req->block_table);
                batch.n_past.push_back(req->n_past);
                batch.scheduler_seq_ids.push_back(req->seq_id);
                batch.hybrid_ssm_runtime_states.push_back(req->ssm_runtime_states.empty() ? nullptr
                                                                                          : &req->ssm_runtime_states);
                batch_requests.push_back(req);
                batch_token_counts.push_back(tokens_to_take);
                batch_build_stats.built_batch_count++;
                reset_batch_build_stall(req);

                GenericInput input;
                input.kind = BatchInputKind::Tokens;
                input.tokens = tokens;
                batch.inputs.push_back(std::move(input));
            }

            // Process decode sequences
            for (int seq_id : sched_output.decode_seq_ids) {
                batch_build_stats.scheduled_decode_count++;
                auto req_it = seq_to_request.find(seq_id);
                if (req_it == seq_to_request.end()) {
                    batch_build_stats.skipped_missing_request_count++;
                    batch_build_stats.missing_seq_ids.push_back(seq_id);
                    continue;
                }

                Request* req = req_it->second;
                LOG_TRACE("Decode loop: Found request {} for seq {} (finished={})", req->id, seq_id, req->finished);
                if (req->finished) {
                    batch_build_stats.skipped_finished_count++;
                    note_batch_build_stall(req);
                    continue;
                }
                if (req->cancelled.load(std::memory_order_relaxed)) {
                    batch_build_stats.skipped_cancelled_count++;
                    note_batch_build_stall(req);
                    continue;
                }

                // Enforce homogeneous batch type
                if (first_request) {
                    is_embedding_batch = req->is_embedding;
                    first_request = false;
                } else if (req->is_embedding != is_embedding_batch) {
                    batch_build_stats.skipped_mixed_embedding_count++;
                    note_batch_build_stall(req);
                    continue;
                }

                // Decode: single token
                if (!ensure_request_blocks_writable(req, req->n_past, 1)) {
                    batch_build_stats.skipped_block_writable_count++;
                    note_batch_build_stall(req);
                    LOG_WARN("Skipping req {} in decode: failed to ensure writable KV block (seq_id={}, n_past={})",
                             req->id, seq_id, req->n_past);
                    continue;
                }

                int batch_seq_idx = batch_requests.size();
                batch.tokens.push_back(req->tokens.back());
                batch.pos.push_back(req->n_past);
                batch.seq_id.push_back(batch_seq_idx);
                batch.block_tables.push_back(req->block_table);
                batch.n_past.push_back(req->n_past);
                batch.scheduler_seq_ids.push_back(req->seq_id);
                batch.hybrid_ssm_runtime_states.push_back(req->ssm_runtime_states.empty() ? nullptr
                                                                                          : &req->ssm_runtime_states);
                batch_requests.push_back(req);
                batch_token_counts.push_back(1);
                batch_build_stats.built_batch_count++;
                reset_batch_build_stall(req);

                GenericInput input;
                input.kind = BatchInputKind::Tokens;
                if (!req->tokens.empty()) {
                    input.tokens.push_back(req->tokens.back());
                }
                batch.inputs.push_back(std::move(input));
            }

            if (batch_requests.empty()) {
                std::sort(batch_build_stats.stalled_requests.begin(), batch_build_stats.stalled_requests.end());
                batch_build_stats.stalled_requests.erase(
                    std::unique(batch_build_stats.stalled_requests.begin(), batch_build_stats.stalled_requests.end()),
                    batch_build_stats.stalled_requests.end());
                const auto batch_build_ns =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - batch_build_begin)
                                              .count());
                for (Request* req : batch_build_stats.stalled_requests) {
                    if (req) {
                        req->batch_build_ns += batch_build_ns;
                    }
                }
                std::sort(batch_build_stats.missing_seq_ids.begin(), batch_build_stats.missing_seq_ids.end());
                batch_build_stats.missing_seq_ids.erase(
                    std::unique(batch_build_stats.missing_seq_ids.begin(), batch_build_stats.missing_seq_ids.end()),
                    batch_build_stats.missing_seq_ids.end());

                if (batch_build_stats.scheduled_total() > 0) {
                    const auto now = std::chrono::steady_clock::now();
                    long oldest_idle_ms = 0;
                    long max_build_stall_loops = 0;
                    for (Request* req : batch_build_stats.stalled_requests) {
                        if (!req || req->finished) {
                            continue;
                        }
                        const auto last_progress = (req->last_progress_time == std::chrono::steady_clock::time_point())
                                                       ? req->start_time
                                                       : req->last_progress_time;
                        const long idle_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
                        oldest_idle_ms = std::max(oldest_idle_ms, idle_ms);
                        max_build_stall_loops =
                            std::max<long>(max_build_stall_loops, static_cast<long>(req->batch_build_stall_count));
                    }

                    const bool should_fail = oldest_idle_ms >= GetEmptyScheduleFailAfterMs();
                    const bool should_log =
                        IsSchedulerStallDebugEnabled() || oldest_idle_ms >= GetEmptyScheduleWarnAfterMs();

                    if (should_log || should_fail) {
                        const std::string scheduler_state = state->DescribeSchedulerState();
                        const std::string active_state = state->DescribeActiveRequests();
                        std::ostringstream skip_summary_stream;
                        skip_summary_stream
                            << "scheduled[prefill=" << batch_build_stats.scheduled_prefill_count
                            << ",decode=" << batch_build_stats.scheduled_decode_count
                            << "] built=" << batch_build_stats.built_batch_count
                            << " skipped[block_writable=" << batch_build_stats.skipped_block_writable_count
                            << ",finished=" << batch_build_stats.skipped_finished_count
                            << ",cancelled=" << batch_build_stats.skipped_cancelled_count
                            << ",mixed_embedding=" << batch_build_stats.skipped_mixed_embedding_count
                            << ",missing_request=" << batch_build_stats.skipped_missing_request_count
                            << "] max_build_stall_loops=" << max_build_stall_loops;
                        const std::string skip_summary = skip_summary_stream.str();
                        if (should_fail) {
                            LOG_ERROR("Batch construction stalled after {} ms with {}. {} {}", oldest_idle_ms,
                                      skip_summary, scheduler_state, active_state);
                        } else {
                            LOG_WARN("Batch construction skipped all scheduled work after {} ms with {}. {} {}",
                                     oldest_idle_ms, skip_summary, scheduler_state, active_state);
                        }
                    }

                    if (should_fail) {
                        for (Request* req : batch_build_stats.stalled_requests) {
                            if (!req || req->finished) {
                                continue;
                            }
                            LOG_ERROR("Failing batch-build-stalled request {} (seq_id={}, prefill={}, tokens={}, "
                                      "n_past={}, generated={}, build_stall_loops={})",
                                      req->id, req->seq_id, req->is_prefill, req->tokens.size(), req->n_past,
                                      req->generated_count, req->batch_build_stall_count);
                            req->finished = true;
                            req->decode_finish_cause = DecodeFinishCause::BatchBuildStall;
                            FinalizeDecodeSilentFinishReason(req);
                            LogRequestDecodeSummary(req, current_model);
                            state->metrics.failed_requests++;
                            EmitRequestResult(state, req,
                                              "Error: Scheduler scheduled work but batch construction stalled", -1,
                                              true, true, global_direct_callback);
                            if (!req->block_table.empty()) {
                                current_kv_cache->block_manager->Free(req->block_table);
                                req->block_table.clear();
                            }
                            if (req->seq_id >= 0) {
                                state->scheduler->RemoveRequest(req->seq_id, false);
                                seq_to_request.erase(req->seq_id);
                            }
                            {
                                std::lock_guard<std::mutex> lk(req->mu);
                                req->cv.notify_all();
                            }
                        }
                        for (int missing_seq_id : batch_build_stats.missing_seq_ids) {
                            state->scheduler->RemoveRequest(missing_seq_id, false);
                            seq_to_request.erase(missing_seq_id);
                        }
                        reap_finished_requests();
                        continue;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            const auto batch_build_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                  std::chrono::steady_clock::now() - batch_build_begin)
                                                                  .count());
            for (Request* req : batch_requests) {
                if (req) {
                    req->batch_build_ns += batch_build_ns;
                }
            }

            batch.num_seqs = batch_requests.size();

            bool is_prefill_batch = false;
            for (Request* req : batch_requests) {
                if (req->is_prefill) {
                    is_prefill_batch = true;
                    break;
                }
            }

            InferenceConfig& infer_cfg = InferenceConfig::Instance();
            int active_threads = base_threads;
            const int decode_batch_size = static_cast<int>(batch_requests.size());
            int prefill_prompt_token_count = 0;
            if (is_prefill_batch && decode_batch_size == 1 && !batch_requests.empty()) {
                const Request* const req = batch_requests.front();
                if (req && !req->prompt_tokens_for_cache.empty()) {
                    prefill_prompt_token_count = static_cast<int>(req->prompt_tokens_for_cache.size());
                } else if (!batch_token_counts.empty()) {
                    prefill_prompt_token_count = batch_token_counts.front();
                }
            }
            const bool is_decode_batch = !is_prefill_batch && !is_embedding_batch;
            const auto simd_level = densecore::simd::DetectSimdLevel();
            const char* prefill_thread_policy = "prefill_base";
            const char* decode_thread_policy = "base";

            // DENSECORE_BENCH_RESPECT_THREADS=1 forces full thread count for all phases
            const bool bench_respect_threads = state->fast_path_config.bench_respect_threads;

            if (bench_respect_threads) {
                active_threads = base_threads;
                if (is_prefill_batch) {
                    prefill_thread_policy = "bench_respect";
                } else {
                    decode_thread_policy = "bench_respect";
                }
            } else if (!is_embedding_batch && infer_cfg.enable_split_thread_policy) {
                const int configured_threads = is_prefill_batch ? infer_cfg.prefill_threads : infer_cfg.decode_threads;
                if (configured_threads > 0) {
                    active_threads = configured_threads;
                    if (is_prefill_batch) {
                        prefill_thread_policy = "prefill_override";
                    } else {
                        decode_thread_policy = "decode_override";
                    }
                } else {
                    if (is_prefill_batch) {
                        const PrefillThreadPolicySelection selection = ResolvePrefillThreadPolicySelection(
                            current_model, decode_batch_size, prefill_prompt_token_count, physical_core_count,
                            base_threads, simd_level);
                        active_threads = selection.threads;
                        prefill_thread_policy = selection.label;
                    } else {
                        const int batch_override = DecodeThreadsBatchOverride(decode_batch_size);
                        if (batch_override > 0) {
                            active_threads = batch_override;
                            decode_thread_policy = "decode_batch_env";
                        } else if (UseLegacyDecodeThreadPolicy()) {
                            active_threads =
                                ResolveLegacyDecodeThreads(decode_batch_size, physical_core_count, base_threads);
                            decode_thread_policy = "decode_legacy_auto";
                        } else {
                            const DecodeThreadPolicySelection selection = ResolveDecodeThreadPolicySelection(
                                current_model, decode_batch_size, physical_core_count, base_threads, simd_level);
                            active_threads = selection.threads;
                            decode_thread_policy = selection.label;
                        }
                    }
                }
            }
            if (is_prefill_batch) {
                const int debug_prefill_threads = PrefillThreadOverride();
                if (debug_prefill_threads > 0) {
                    active_threads = debug_prefill_threads;
                    prefill_thread_policy = "prefill_debug_override";
                }
            }

            // Benchmark decode-batch fast path: when explicitly enabled, allow
            // batch>=4 decode to scale above the base thread budget.
            bool bench_decode_batch_thread_boost = false;
            const bool bench_decode_batch_fast_path = global_bench_fast_path && !bench_respect_threads &&
                                                      IsBenchmarkDecodeBatchFastPathEnabled() && !is_prefill_batch &&
                                                      !is_embedding_batch;
            if (bench_decode_batch_fast_path && physical_core_count > 0) {
                const int max_fast_batch = std::max(1, BenchmarkFastPathMaxBatch());
                if (decode_batch_size >= 4 && decode_batch_size <= max_fast_batch) {
                    const int decode_boost_target = std::max(base_threads, std::max(1, (base_threads * 3) / 2));
                    active_threads = std::max(active_threads, decode_boost_target);
                    bench_decode_batch_thread_boost = active_threads > base_threads;
                    if (bench_decode_batch_thread_boost) {
                        decode_thread_policy = "bench_fast_path_boost";
                    }
                }
            }

            // Keep decode within user-provided base thread budget unless
            // explicitly opting into oversubscription.
            if (!is_prefill_batch && base_threads > 0 && !AllowDecodeThreadsOverBase() &&
                !bench_decode_batch_thread_boost) {
                active_threads = std::min(active_threads, base_threads);
            }
            active_threads = std::max(1, std::min(active_threads, std::max(1, physical_core_count)));
            infer_cfg.num_threads = active_threads;
            if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                GetDecodeWorkerStats().last_threads_by_batch[decode_batch_size].store(active_threads,
                                                                                      std::memory_order_relaxed);
            }
            if (!is_embedding_batch) {
                for (Request* req : batch_requests) {
                    if (!req) {
                        continue;
                    }
                    if (is_prefill_batch) {
                        req->prefill_thread_count = active_threads;
                        req->prefill_thread_policy = prefill_thread_policy;
                    } else {
                        req->decode_thread_count = active_threads;
                        req->decode_thread_policy = decode_thread_policy;
                    }
                    req->active_thread_count = active_threads;
                }
            }
            if (IsDebugDecodeThreadsEnabled() && !is_embedding_batch) {
                static std::atomic<uint64_t> logged_decode_thread_events{0};
                const uint64_t event_idx = logged_decode_thread_events.fetch_add(1, std::memory_order_relaxed);
                if (event_idx < 64) {
                    std::cerr << "[ThreadPolicy] phase=" << (is_prefill_batch ? "prefill" : "decode")
                              << " batch=" << decode_batch_size << " selected=" << active_threads
                              << " base=" << base_threads << " physical=" << physical_core_count
                              << " policy=" << (is_prefill_batch ? prefill_thread_policy : decode_thread_policy)
                              << std::endl;
                }
            }

            // =========================================================================
            // 8a. Build Multi-LoRA Adapter Token Map (per-request adapters)
            // =========================================================================
            // We populate the BatchSpec directly with shared_ptrs to keep adapters active
            int batch_token_offset = 0;
            for (size_t req_idx = 0; req_idx < batch_requests.size(); ++req_idx) {
                Request* req = batch_requests[req_idx];
                const int token_count =
                    (req_idx < batch_token_counts.size()) ? batch_token_counts[req_idx] : (req->is_prefill ? 0 : 1);
                if (!req->lora_name.empty()) {
                    // Get shared_ptr to ensure safety
                    auto adapter = state->lora_storage.GetAdapter(req->lora_name);
                    if (adapter) {
                        auto& indices = batch.lora_map[adapter];
                        // If new entry, reserve space (heuristic)
                        if (indices.empty()) indices.reserve(batch.tokens.size());

                        for (int i = 0; i < token_count; ++i) {
                            indices.push_back(batch_token_offset + i);
                        }
                    }
                }
                batch_token_offset += token_count;
            }
            if (!batch.lora_map.empty()) {
                LOG_TRACE("Multi-LoRA batch: {} adapters", batch.lora_map.size());
            }

            // =========================================================================
            // 9. Run Inference
            // =========================================================================
            // Default strategy is still "Rebuild Graph, Reuse Memory" because most
            // decode/prefill layouts have changing topology. For stable decode-only
            // paged layout (one token per seq), we use a bounded graph cache.
            // =========================================================================
            struct ggml_cgraph* gf = nullptr;
            struct ggml_tensor* output = nullptr;
            struct ggml_tensor* embd_inp = nullptr;
            struct ggml_tensor* pos = nullptr;
            bool reused_decode_graph = false;
            bool using_cached_decode_graph = false;
            bool cached_graph_verified_paged_decode_op = false;
            const auto graph_build_rebind_begin = std::chrono::steady_clock::now();

            // =========================================================================
            // BACKEND SELECTION (Abstracted via BackendSelector interface)
            // =========================================================================
            // Platform-specific backend selection is delegated to BackendSelector.
            // On Apple: Uses HybridScheduler to choose CPU/Metal/ANE
            // On other platforms: Always returns CPU backend
            // =========================================================================
            const ggml_backend_t cpu_backend_handle =
                current_model->cpu_backend ? current_model->cpu_backend : current_model->backend;
            ggml_backend_t active_backend = current_model->backend;
            densecore::InferenceProfile requested_profile = densecore::InferenceProfile::LLMCore;
            if (is_embedding_batch) {
                requested_profile = densecore::InferenceProfile::Embedding;
            } else if (current_model->hparams.n_experts > 0) {
                requested_profile = densecore::InferenceProfile::LLMCoreMoE;
            }

            if (state->backend_selector) {
                // Build batch context for backend selection
                int max_n_past = 0;
                for (int n_past : batch.n_past) {
                    max_n_past = std::max(max_n_past, n_past);
                }
                int avg_tokens = batch.num_seqs > 0 ? static_cast<int>(batch.tokens.size() / batch.num_seqs) : 1;
                int seq_len = std::max(1, max_n_past + avg_tokens);

                densecore::BatchContext ctx;
                ctx.batch_size = batch.num_seqs;
                ctx.seq_len = seq_len;
                ctx.is_prefill = is_prefill_batch;
                ctx.n_past = max_n_past;

                densecore::BackendCandidates candidates;
                candidates.cpu_backend = cpu_backend_handle;
                candidates.gpu_backend = current_model->metal_backend;
                candidates.accelerator_backend = nullptr;
#ifdef __APPLE__
                if (state->hybrid_ane_backend && current_model->metal_backend) {
                    // GGML does not expose a dedicated ANE backend handle yet.
                    // Pass Metal as an accelerator proxy so selector can keep NPU
                    // intent while HAL mixed-routing handles per-op dispatch.
                    candidates.accelerator_backend = current_model->metal_backend;
                }
#endif

                const densecore::BackendSelection selection = state->backend_selector->Select(ctx, candidates);
                if (selection.backend) {
                    active_backend = selection.backend;
                }
                deps.preferred_device = selection.preferred_device;
                deps.preferred_matmul_device = selection.preferred_matmul_device;
                deps.preferred_attention_device = selection.preferred_attention_device;
                deps.preferred_norm_device = selection.preferred_norm_device;
                deps.mixed_operation_routing = selection.mixed_operation_routing;
            }

            // Profile admission on backend capabilities:
            // if the selected device cannot satisfy requested profile even with
            // fallback, force CPU to avoid late runtime failures.
            if (deps.backend_registry &&
                !deps.backend_registry->SupportsProfile(deps.preferred_device, requested_profile, true)) {
                LOG_WARN("Backend {} cannot satisfy profile {}, forcing CPU fallback",
                         DeviceTypeName(deps.preferred_device), InferenceProfileName(requested_profile));
                deps.preferred_device = densecore::DeviceType::CPU;
                active_backend = cpu_backend_handle;
            }
            const bool cpu_backend_active = (active_backend == cpu_backend_handle);
            auto maybe_set_cpu_threads = [&]() {
                // Thread config must target the actual CPU backend handle; when a
                // non-CPU backend is selected, touching CPU thread state is irrelevant
                // and can desynchronize bookkeeping.
                if (!cpu_backend_active) return;
                if (active_threads != last_set_threads) {
                    ggml_backend_cpu_set_n_threads(cpu_backend_handle, active_threads);
                    last_set_threads = active_threads;
                }
            };

            const int decode_graph_cache_max_batch = std::max(1, DecodeGraphCacheMaxBatch());
            const int decode_graph_cache_lru_size = std::max(1, DecodeGraphCacheLruSize());
            const bool decode_graph_cache_active =
                IsDecodeGraphCacheEnabled() && current_kv_cache != nullptr && decode_graph_cache_lru_size > 0;
            const bool force_disable_graph_cache = IsGraphCacheReuseDisabled();
            const auto& prefill_graph_cache_policy = state->fast_path_config.prefill_graph_cache;
            const int prefill_graph_cache_lru_size = std::max(1, prefill_graph_cache_policy.lru_size);
            const size_t prefill_graph_cache_max_bytes = prefill_graph_cache_policy.max_bytes;
            const bool prefill_graph_cache_active =
                prefill_graph_cache_policy.enabled && current_kv_cache != nullptr && prefill_graph_cache_lru_size > 0;
            const bool decode_graph_cache_active_effective = decode_graph_cache_active && !force_disable_graph_cache;
            const bool prefill_graph_cache_active_effective = prefill_graph_cache_active && !force_disable_graph_cache;
            auto compute_graph_context_estimate = [&]() {
                size_t seq_len_hint = 0;
                size_t chunk_token_hint = 0;
                for (size_t req_idx = 0; req_idx < batch_requests.size(); ++req_idx) {
                    const Request* batch_req = batch_requests[req_idx];
                    const int token_count =
                        (req_idx < batch_token_counts.size()) ? std::max(0, batch_token_counts[req_idx]) : 0;
                    const size_t prompt_hint = batch_req
                                                   ? static_cast<size_t>(std::max(batch_req->prompt_token_count,
                                                                                  batch_req->n_past + token_count))
                                                   : static_cast<size_t>(token_count);
                    seq_len_hint = std::max(seq_len_hint, prompt_hint);
                    chunk_token_hint = std::max(chunk_token_hint, static_cast<size_t>(token_count));
                }
                return EngineState::EstimateGraphContextSize(
                    current_model, seq_len_hint, static_cast<size_t>(std::max(1, batch.num_seqs)), chunk_token_hint);
            };
            const EngineState::GraphContextEstimate graph_ctx_estimate = compute_graph_context_estimate();
            const size_t prefill_graph_ctx_bytes =
                prefill_graph_cache_active_effective ? graph_ctx_estimate.total_bytes : 0;
            bool decode_single_token_layout = !is_embedding_batch && !is_prefill_batch && batch.num_seqs > 0 &&
                                              batch.num_seqs <= decode_graph_cache_max_batch &&
                                              batch_token_counts.size() == batch_requests.size() &&
                                              static_cast<int>(batch_token_counts.size()) == batch.num_seqs;
            if (decode_single_token_layout) {
                for (size_t i = 0; i < batch_token_counts.size(); ++i) {
                    if (batch_token_counts[i] != 1 || batch_requests[i]->is_prefill) {
                        decode_single_token_layout = false;
                        break;
                    }
                }
            }
            bool stable_paged_decode_topology = false;
            if (decode_single_token_layout) {
                const bool paged_decode_supported = densecore::models::SupportsPagedDecodeAttention(current_model);
                if (UseLegacyDecodeGraphCachePolicy()) {
                    const bool batched_decode_forced_paged = paged_decode_supported && batch.num_seqs > 1;
                    const bool single_decode_forced_paged = paged_decode_supported && batch.num_seqs == 1 &&
                                                            IsPagedDecodeModeForcedOn() &&
                                                            IsBatchedPagedDecodeEnabled();
                    stable_paged_decode_topology = batched_decode_forced_paged || single_decode_forced_paged;
                } else {
                    stable_paged_decode_topology =
                        IsStablePagedDecodeTopologyForCache(current_model, current_kv_cache, batch);
                }
            }
            // Graph reuse is safe when decode path is deterministically paged.
            const bool decode_topology_stable = stable_paged_decode_topology;
            // CPU-only cache admission: cached decode graphs may include paged
            // decode custom ops that are not portable across backend/device types.
            const bool decode_reuse_shape_eligible = decode_graph_cache_active_effective && decode_topology_stable &&
                                                     batch.lora_map.empty() &&
                                                     IsDecodeGraphCacheSafeForModel(current_model);
            const bool decode_reuse_candidate = decode_reuse_shape_eligible && cpu_backend_active;
            const auto decode_model_descriptor = densecore::models::DescribeModel(current_model);
            const size_t decode_model_variant_index =
                std::min(static_cast<size_t>(decode_model_descriptor.variant), kDecodeGraphCacheTrackedVariants - 1);
            const size_t decode_batch_bucket = static_cast<size_t>(
                std::clamp(batch.num_seqs, 1, static_cast<int>(kDecodeGraphCacheTrackedBatches - 1)));
            const size_t decode_graph_uncacheable_limit =
                static_cast<size_t>(std::max(1, decode_graph_cache_lru_size * 2));
            if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                DecodeWorkerStats& decode_stats = GetDecodeWorkerStats();
                if (!decode_graph_cache_active_effective) {
                    decode_stats.graph_cache_skip_disabled.fetch_add(1, std::memory_order_relaxed);
                } else if (!decode_topology_stable) {
                    decode_stats.graph_cache_skip_unstable.fetch_add(1, std::memory_order_relaxed);
                } else if (!batch.lora_map.empty()) {
                    decode_stats.graph_cache_skip_lora.fetch_add(1, std::memory_order_relaxed);
                } else if (!cpu_backend_active) {
                    decode_stats.graph_cache_skip_backend.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (decode_reuse_shape_eligible && !cpu_backend_active && IsDebugGraphLoggingEnabled()) {
                std::cerr << "[DecodeGraphCache][DEBUG] skip cache reuse due to backend/device mismatch"
                          << " (device=" << DeviceTypeName(deps.preferred_device)
                          << ", selected_backend=" << static_cast<const void*>(active_backend)
                          << ", cpu_backend=" << static_cast<const void*>(cpu_backend_handle) << ")" << std::endl;
            }

            DecodeGraphCacheKey decode_graph_key{};
            if (decode_reuse_candidate) {
                decode_graph_key.batch_size = batch.num_seqs;
                decode_graph_key.threads = active_threads;
                decode_graph_key.model_id = reinterpret_cast<uintptr_t>(current_model);
                decode_graph_key.arch_id = current_model ? static_cast<int>(current_model->arch) : -1;
                decode_graph_key.cache_type_id = current_kv_cache ? static_cast<int>(current_kv_cache->cache_type) : -1;
                decode_graph_key.feature_flags = BuildDecodeGraphFeatureFlags(current_model);
            }
            const bool decode_key_marked_uncacheable =
                decode_reuse_candidate &&
                decode_graph_uncacheable.find(decode_graph_key) != decode_graph_uncacheable.end();
            const bool decode_reuse_attempt_allowed = decode_reuse_candidate && !decode_key_marked_uncacheable;
            const bool prefill_graph_entry_cacheable = prefill_graph_cache_active_effective &&
                                                       prefill_graph_ctx_bytes > 0 &&
                                                       prefill_graph_ctx_bytes <= prefill_graph_cache_max_bytes;
            const bool prefill_reuse_shape_eligible =
                prefill_graph_cache_active_effective && is_prefill_batch && !is_embedding_batch && cpu_backend_active &&
                current_model && current_model->hparams.n_experts == 0 && !current_model->arch_flags.is_hybrid_ssm &&
                batch.lora_map.empty() && batch.num_seqs == 1 && batch_token_counts.size() == 1 &&
                batch_requests.size() == 1 && !batch_requests[0]->is_embedding && prefill_graph_entry_cacheable;
            const bool prefill_reuse_candidate = prefill_reuse_shape_eligible;

            PrefillGraphCacheKey prefill_graph_key{};
            if (prefill_reuse_candidate) {
                int n_past = 0;
                if (!batch.n_past.empty()) {
                    n_past = std::max(0, batch.n_past[0]);
                }
                prefill_graph_key.batch_size = batch.num_seqs;
                prefill_graph_key.tokens = static_cast<int>(batch.tokens.size());
                prefill_graph_key.n_past = n_past;
                prefill_graph_key.threads = active_threads;
                prefill_graph_key.model_id = reinterpret_cast<uintptr_t>(current_model);
                prefill_graph_key.arch_id = current_model ? static_cast<int>(current_model->arch) : -1;
                prefill_graph_key.cache_type_id =
                    current_kv_cache ? static_cast<int>(current_kv_cache->cache_type) : -1;
                prefill_graph_key.feature_flags = BuildDecodeGraphFeatureFlags(current_model);
            }

            std::chrono::steady_clock::time_point graph_build_begin, graph_build_end;
            bool graph_build_accounted = false;
            bool built_decode_graph_cache_entry = false;
            bool reused_prefill_graph = false;
            bool using_cached_prefill_graph = false;
            bool built_prefill_graph_cache_entry = false;
            if (decode_reuse_attempt_allowed) {
                if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                    GetDecodeWorkerStats().graph_cache_attempts.fetch_add(1, std::memory_order_relaxed);
                    GetDecodeWorkerStats()
                        .graph_cache_by_variant_batch[decode_model_variant_index][decode_batch_bucket]
                        .attempts.fetch_add(1, std::memory_order_relaxed);
                }
                auto it = decode_graph_cache.find(decode_graph_key);
                if (it != decode_graph_cache.end() && it->second.graph && it->second.output && it->second.embd_inp &&
                    it->second.pos) {
                    bool rebind_ok = true;
                    if (DoesDecodeGraphCacheRequireRuntimeRebind(current_model)) {
                        rebind_ok = RebindHybridSSMDecodeGraphRuntimeState(it->second.graph, batch);
                        if (!rebind_ok) {
                            if (IsDebugGraphLoggingEnabled()) {
                                std::cerr << "[DecodeGraphCache] evicting hybrid SSM cached graph after runtime "
                                             "rebind failure"
                                          << " (bs=" << decode_graph_key.batch_size
                                          << ", threads=" << decode_graph_key.threads << ")" << std::endl;
                            }
                            decode_graph_lru.erase(it->second.lru_it);
                            free_decode_graph_entry(&it->second);
                            decode_graph_cache.erase(it);
                        }
                    }
                    if (!rebind_ok) {
                        // Fall back to rebuilding below.
                    } else {
                        touch_decode_graph_entry(&it->second);
                        gf = it->second.graph;
                        output = it->second.output;
                        embd_inp = it->second.embd_inp;
                        pos = it->second.pos;
                        cached_graph_verified_paged_decode_op = it->second.verified_paged_decode_op;
                        reused_decode_graph = true;
                        using_cached_decode_graph = true;
                        for (Request* req : batch_requests) {
                            if (req) {
                                req->graph_cache_hit_count++;
                            }
                        }
                        if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                            GetDecodeWorkerStats().graph_cache_hits.fetch_add(1, std::memory_order_relaxed);
                            GetDecodeWorkerStats()
                                .graph_cache_by_variant_batch[decode_model_variant_index][decode_batch_bucket]
                                .hits.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }

            if (!reused_decode_graph && prefill_reuse_candidate) {
                auto it = prefill_graph_cache.find(prefill_graph_key);
                if (it != prefill_graph_cache.end() && it->second.graph && it->second.output && it->second.embd_inp &&
                    it->second.pos) {
                    touch_prefill_graph_entry(&it->second);
                    gf = it->second.graph;
                    output = it->second.output;
                    embd_inp = it->second.embd_inp;
                    pos = it->second.pos;
                    reused_prefill_graph = true;
                    using_cached_prefill_graph = true;
                    for (Request* req : batch_requests) {
                        if (req) {
                            req->graph_cache_hit_count++;
                        }
                    }
                }
            }

            if (!reused_decode_graph && !reused_prefill_graph) {
                if (decode_reuse_attempt_allowed) {
                    while (decode_graph_cache.size() >= static_cast<size_t>(decode_graph_cache_lru_size) &&
                           !decode_graph_lru.empty()) {
                        const DecodeGraphCacheKey evict_key = decode_graph_lru.back();
                        decode_graph_lru.pop_back();
                        auto evict_it = decode_graph_cache.find(evict_key);
                        if (evict_it != decode_graph_cache.end()) {
                            free_decode_graph_entry(&evict_it->second);
                            decode_graph_cache.erase(evict_it);
                        }
                    }

                    DecodeGraphCacheEntry candidate;
                    struct ggml_init_params decode_params = {
                        .mem_size = DecodeGraphCacheCtxBytes(),
                        .mem_buffer = nullptr,
                        .no_alloc = false,
                    };
                    candidate.ctx = ggml_init(decode_params);
                    if (candidate.ctx) {
                        candidate.graph = ggml_new_graph_custom(candidate.ctx, 32768, false);
                        if (candidate.graph) {
                            graph_build_begin = std::chrono::steady_clock::now();
                            candidate.output = BuildTransformerGraph(current_model, current_kv_cache, candidate.ctx,
                                                                     batch, is_embedding_batch, candidate.graph,
                                                                     &candidate.embd_inp, &candidate.pos);
                            graph_build_end = std::chrono::steady_clock::now();
                            if (candidate.output && candidate.embd_inp && candidate.pos) {
                                bool cache_entry_admissible = true;
                                const bool requires_paged_decode_verify =
                                    decode_single_token_layout && batch.num_seqs > 1;
                                candidate.verified_paged_decode_op =
                                    !requires_paged_decode_verify ||
                                    VerifyCachedDecodeGraphPagedOpOnBuild(candidate.graph, batch.num_seqs);
                                if (requires_paged_decode_verify && !candidate.verified_paged_decode_op) {
                                    cache_entry_admissible = false;
                                    const auto inserted_uncacheable = decode_graph_uncacheable.insert(decode_graph_key);
                                    if (inserted_uncacheable.second) {
                                        if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                                            GetDecodeWorkerStats().graph_cache_rejected_uncacheable.fetch_add(
                                                1, std::memory_order_relaxed);
                                            GetDecodeWorkerStats()
                                                .graph_cache_by_variant_batch[decode_model_variant_index]
                                                                             [decode_batch_bucket]
                                                .rejected_uncacheable.fetch_add(1, std::memory_order_relaxed);
                                        }
                                        decode_graph_uncacheable_lru.push_back(decode_graph_key);
                                        while (decode_graph_uncacheable.size() > decode_graph_uncacheable_limit &&
                                               !decode_graph_uncacheable_lru.empty()) {
                                            const DecodeGraphCacheKey evict_uncacheable_key =
                                                decode_graph_uncacheable_lru.front();
                                            decode_graph_uncacheable_lru.pop_front();
                                            decode_graph_uncacheable.erase(evict_uncacheable_key);
                                        }
                                        if (IsDebugGraphLoggingEnabled()) {
                                            std::cerr
                                                << "[DecodeGraphCache] key marked uncacheable (missing paged op), "
                                                   "skipping future cache attempts."
                                                << " (bs=" << decode_graph_key.batch_size
                                                << ", threads=" << decode_graph_key.threads << ")" << std::endl;
                                        }
                                    }
                                }
                                if (cache_entry_admissible && DoesDecodeGraphCacheRequireRuntimeRebind(current_model) &&
                                    !RebindHybridSSMDecodeGraphRuntimeState(candidate.graph, batch)) {
                                    cache_entry_admissible = false;
                                    if (IsDebugGraphLoggingEnabled()) {
                                        std::cerr << "[DecodeGraphCache] skip cache insert: hybrid SSM runtime "
                                                     "rebind probe failed"
                                                  << " (bs=" << decode_graph_key.batch_size
                                                  << ", threads=" << decode_graph_key.threads << ")" << std::endl;
                                    }
                                }
                                if (cache_entry_admissible) {
                                    decode_graph_lru.push_front(decode_graph_key);
                                    candidate.lru_it = decode_graph_lru.begin();
                                    auto inserted = decode_graph_cache.emplace(decode_graph_key, std::move(candidate));
                                    DecodeGraphCacheEntry& entry = inserted.first->second;
                                    gf = entry.graph;
                                    output = entry.output;
                                    embd_inp = entry.embd_inp;
                                    pos = entry.pos;
                                    cached_graph_verified_paged_decode_op = entry.verified_paged_decode_op;
                                    built_decode_graph_cache_entry = true;
                                    using_cached_decode_graph = true;
                                    for (Request* req : batch_requests) {
                                        if (req) {
                                            req->graph_cache_miss_count++;
                                        }
                                    }
                                    if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                                        GetDecodeWorkerStats().graph_cache_builds.fetch_add(1,
                                                                                            std::memory_order_relaxed);
                                        GetDecodeWorkerStats()
                                            .graph_cache_by_variant_batch[decode_model_variant_index]
                                                                         [decode_batch_bucket]
                                            .builds.fetch_add(1, std::memory_order_relaxed);
                                    }
                                }
                            }
                        }
                    }
                    if (!built_decode_graph_cache_entry) {
                        free_decode_graph_entry(&candidate);
                    }
                }

                if (!built_decode_graph_cache_entry && prefill_reuse_candidate) {
                    while (prefill_graph_cache.size() >= static_cast<size_t>(prefill_graph_cache_lru_size) &&
                           !prefill_graph_lru.empty()) {
                        const PrefillGraphCacheKey evict_key = prefill_graph_lru.back();
                        prefill_graph_lru.pop_back();
                        auto evict_it = prefill_graph_cache.find(evict_key);
                        if (evict_it != prefill_graph_cache.end()) {
                            prefill_graph_cache_bytes = (prefill_graph_cache_bytes > evict_it->second.ctx_bytes)
                                                            ? (prefill_graph_cache_bytes - evict_it->second.ctx_bytes)
                                                            : 0;
                            if (evict_it->second.ctx) {
                                ggml_free(evict_it->second.ctx);
                                evict_it->second.ctx = nullptr;
                            }
                            evict_it->second.graph = nullptr;
                            evict_it->second.output = nullptr;
                            evict_it->second.embd_inp = nullptr;
                            evict_it->second.pos = nullptr;
                            evict_it->second.ctx_buffer.clear();
                            prefill_graph_cache.erase(evict_it);
                        }
                    }

                    PrefillGraphCacheEntry candidate;
                    candidate.ctx_bytes = prefill_graph_ctx_bytes;
                    struct ggml_init_params prefill_params = {
                        .mem_size = candidate.ctx_bytes,
                        .mem_buffer = nullptr,
                        .no_alloc = false,
                    };
                    candidate.ctx = ggml_init(prefill_params);
                    if (candidate.ctx) {
                        candidate.graph = ggml_new_graph_custom(candidate.ctx, 32768, false);
                        if (candidate.graph) {
                            graph_build_begin = std::chrono::steady_clock::now();
                            candidate.output = BuildTransformerGraph(current_model, current_kv_cache, candidate.ctx,
                                                                     batch, is_embedding_batch, candidate.graph,
                                                                     &candidate.embd_inp, &candidate.pos);
                            graph_build_end = std::chrono::steady_clock::now();
                            if (candidate.output && candidate.embd_inp && candidate.pos) {
                                while (
                                    !prefill_graph_lru.empty() &&
                                    (prefill_graph_cache.size() >= static_cast<size_t>(prefill_graph_cache_lru_size) ||
                                     prefill_graph_cache_bytes + candidate.ctx_bytes > prefill_graph_cache_max_bytes)) {
                                    const PrefillGraphCacheKey budget_evict_key = prefill_graph_lru.back();
                                    prefill_graph_lru.pop_back();
                                    auto budget_evict_it = prefill_graph_cache.find(budget_evict_key);
                                    if (budget_evict_it == prefill_graph_cache.end()) {
                                        continue;
                                    }
                                    prefill_graph_cache_bytes =
                                        (prefill_graph_cache_bytes > budget_evict_it->second.ctx_bytes)
                                            ? (prefill_graph_cache_bytes - budget_evict_it->second.ctx_bytes)
                                            : 0;
                                    if (budget_evict_it->second.ctx) {
                                        ggml_free(budget_evict_it->second.ctx);
                                        budget_evict_it->second.ctx = nullptr;
                                    }
                                    budget_evict_it->second.graph = nullptr;
                                    budget_evict_it->second.output = nullptr;
                                    budget_evict_it->second.embd_inp = nullptr;
                                    budget_evict_it->second.pos = nullptr;
                                    budget_evict_it->second.ctx_buffer.clear();
                                    budget_evict_it->second.ctx_bytes = 0;
                                    prefill_graph_cache.erase(budget_evict_it);
                                }
                                prefill_graph_lru.push_front(prefill_graph_key);
                                candidate.lru_it = prefill_graph_lru.begin();
                                auto inserted = prefill_graph_cache.emplace(prefill_graph_key, std::move(candidate));
                                PrefillGraphCacheEntry& entry = inserted.first->second;
                                prefill_graph_cache_bytes += entry.ctx_bytes;
                                gf = entry.graph;
                                output = entry.output;
                                embd_inp = entry.embd_inp;
                                pos = entry.pos;
                                built_prefill_graph_cache_entry = true;
                                using_cached_prefill_graph = true;
                                for (Request* req : batch_requests) {
                                    if (req) {
                                        req->graph_cache_miss_count++;
                                    }
                                }
                            }
                        }
                    }
                    if (!built_prefill_graph_cache_entry && candidate.ctx) {
                        ggml_free(candidate.ctx);
                        candidate.ctx = nullptr;
                        candidate.graph = nullptr;
                        candidate.output = nullptr;
                        candidate.embd_inp = nullptr;
                        candidate.pos = nullptr;
                        candidate.ctx_buffer.clear();
                    }
                }

                if (!built_decode_graph_cache_entry && !built_prefill_graph_cache_entry) {
                    if (is_prefill_batch && ShouldZeroFillPrefillGraphBuffer() && state->inference_ctx.compute_buffer &&
                        state->inference_ctx.compute_buffer_size > 0) {
                        std::memset(state->inference_ctx.compute_buffer, 0, state->inference_ctx.compute_buffer_size);
                    }
                    const size_t ctx_size = graph_ctx_estimate.total_bytes;
                    if (!state->inference_ctx.IsInitialized() || state->inference_ctx.compute_buffer_size < ctx_size) {
                        if (state->inference_ctx.IsInitialized() &&
                            state->inference_ctx.compute_buffer_size < ctx_size) {
                            state->inference_ctx.Free();
                        }
                        const auto descriptor = densecore::models::DescribeModel(current_model);
                        std::cerr << "[DenseCore] GraphCtxEstimate variant="
                                  << densecore::models::ModelVariantName(descriptor.variant)
                                  << " seq_hint=" << graph_ctx_estimate.effective_seq_len
                                  << " num_seqs_hint=" << graph_ctx_estimate.effective_num_seqs
                                  << " chunk_tokens=" << graph_ctx_estimate.chunk_token_hint
                                  << " base_mb=" << (graph_ctx_estimate.base_graph_working_set_bytes / (1024 * 1024))
                                  << " hybrid_mb=" << (graph_ctx_estimate.hybrid_ssm_extra_bytes / (1024 * 1024))
                                  << " safety_mb=" << (graph_ctx_estimate.long_context_safety_pad_bytes / (1024 * 1024))
                                  << " env_extra_mb=" << (graph_ctx_estimate.env_extra_bytes / (1024 * 1024))
                                  << " total_mb=" << (ctx_size / (1024 * 1024)) << std::endl;
                        state->inference_ctx.Init(ctx_size);
                    }

                    if (Request* trace_req = !batch_requests.empty() ? batch_requests[0] : nullptr) {
                        LogPrefillStage("before_prefill_graph_build", trace_req, current_model, &state->inference_ctx,
                                        nullptr, 0, nullptr, is_prefill_batch, active_threads);
                    }

                    // Reset persistent context (O(1) - reuses existing memory buffer)
                    // This prepares a fresh context for graph building without malloc/free
                    state->inference_ctx.Reset();
                    struct ggml_context* ctx_nodes = state->inference_ctx.GetContext();

                    if (!ctx_nodes) {
                        LOG_CRITICAL("FATAL: InferenceContext not initialized!");
                        continue;
                    }

                    gf = ggml_new_graph_custom(ctx_nodes, 32768, false);
                    graph_build_begin = std::chrono::steady_clock::now();
                    output = BuildTransformerGraph(current_model, current_kv_cache, ctx_nodes, batch,
                                                   is_embedding_batch, gf, &embd_inp, &pos);
                    graph_build_end = std::chrono::steady_clock::now();
                    const auto graph_build_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(graph_build_end - graph_build_begin)
                            .count());
                    for (Request* req : batch_requests) {
                        if (req) {
                            req->graph_build_ns += graph_build_ns;
                            req->graph_cache_miss_count++;
                        }
                    }
                    graph_build_accounted = true;
                    if (Request* trace_req = !batch_requests.empty() ? batch_requests[0] : nullptr) {
                        LogPrefillStage("after_prefill_graph_build", trace_req, current_model, &state->inference_ctx,
                                        nullptr, 0, output, is_prefill_batch, active_threads);
                    }
                }
            }

            if (!output || !embd_inp || !pos) {
                LOG_ERROR("Fatal: Content creation failed or tensors missing");
                continue;  // Recover
            }
            const auto graph_build_rebind_ns =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - graph_build_rebind_begin)
                                          .count());
            if (!graph_build_accounted && graph_build_begin != std::chrono::steady_clock::time_point() &&
                graph_build_end != std::chrono::steady_clock::time_point() && graph_build_end >= graph_build_begin) {
                const auto graph_build_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(graph_build_end - graph_build_begin).count());
                for (Request* req : batch_requests) {
                    if (req) {
                        req->graph_build_ns += graph_build_ns;
                    }
                }
            }
            for (Request* req : batch_requests) {
                if (req) {
                    req->graph_rebind_ns += graph_build_rebind_ns;
                }
            }

            if (IsRuntimePathLoggingEnabled()) {
                const auto descriptor = densecore::models::DescribeModel(current_model);
                std::cerr << "[RuntimePath] variant=" << densecore::models::ModelVariantName(descriptor.variant)
                          << " batch_seqs=" << batch.num_seqs
                          << " decode_layout=" << (decode_single_token_layout ? "single_token" : "general")
                          << " decode_cache_reused=" << (reused_decode_graph ? "1" : "0")
                          << " decode_cache_active=" << (using_cached_decode_graph ? "1" : "0")
                          << " prefill_cache_active=" << (using_cached_prefill_graph ? "1" : "0")
                          << " decode_topology_stable=" << (decode_topology_stable ? "1" : "0")
                          << " paged_decode_supported="
                          << (densecore::models::SupportsPagedDecodeAttention(current_model) ? "1" : "0") << std::endl;
            }

            if (using_cached_decode_graph && decode_single_token_layout && batch.num_seqs > 1) {
                DebugVerifyCachedDecodeGraphReuseState(gf, batch.num_seqs, reused_decode_graph,
                                                       cached_graph_verified_paged_decode_op);
#ifndef NDEBUG
                if (current_model && current_model->arch_flags.is_gemma4 && cached_graph_verified_paged_decode_op) {
                    throw densecore::InvalidArgumentException(
                        "Decode graph cache invariant failed: Gemma4 must not reuse paged decode cached graphs.");
                }
#endif
                if (cached_graph_verified_paged_decode_op && !cpu_backend_active) {
                    if (IsDebugGraphLoggingEnabled()) {
                        std::cerr << "[DecodeGraphCache][DEBUG] cached paged decode graph rejected: selected backend "
                                     "does not support CPU custom op"
                                  << " (device=" << DeviceTypeName(deps.preferred_device)
                                  << ", selected_backend=" << static_cast<const void*>(active_backend)
                                  << ", cpu_backend=" << static_cast<const void*>(cpu_backend_handle) << ")"
                                  << std::endl;
                    }
                    throw densecore::InvalidArgumentException(
                        "Decode graph cache invariant failed: cached paged decode graph requires CPU backend.");
                }
            }

            // SETUP INPUTS (Manual Data Assignment)
            // We use state->compute_buffer for input data

            // Use offset 0 of compute buffer for inputs
            char* input_base = state->compute_buffer.get();
            size_t embd_size = ggml_nbytes(embd_inp);
            const size_t pos_size = ggml_nbytes(pos);
            const size_t input_span_bytes = embd_size + 256 + pos_size;

            embd_inp->data = input_base;
            pos->data = input_base + embd_size + 256;  // alignment padding

            if (is_prefill_batch && ShouldZeroFillPrefillInputBuffer()) {
                std::memset(input_base, 0, input_span_bytes);
            }
            memcpy(embd_inp->data, batch.tokens.data(), batch.tokens.size() * sizeof(int));
            PopulatePositionTensor(current_model, batch, pos);
            if (Request* trace_req = !batch_requests.empty() ? batch_requests[0] : nullptr) {
                LogPrefillStage("before_prefill_backend_execute", trace_req, current_model, &state->inference_ctx,
                                input_base, input_span_bytes, output, is_prefill_batch, active_threads);
            }

            // =======================================================================
            // MEMORY FENCE: Ensures visibility of input data to GGML worker threads.
            // =======================================================================
            // The memcpy operations above write input data to buffers that will be
            // read by GGML's internal thread pool workers. On modern CPUs with
            // out-of-order execution and store buffers, these writes may not be
            // immediately visible to other CPU cores.
            //
            // We use a release fence as the publishing side of the synchronization:
            // - std::memory_order_release guarantees that all preceding writes
            //   (the memcpy calls) are visible before any subsequent synchronization
            //   operation that has acquire semantics.
            //
            // GGML's thread pool internally uses proper synchronization (typically
            // condition variables or atomics) with acquire semantics when worker
            // threads start execution, completing the acquire-release synchronization
            // pair and ensuring they observe the input data correctly.
            //
            // This replaces the previous fragile sleep_for(100us) workaround which:
            // - Was not guaranteed to work (timing-based, system-dependent)
            // - Added unnecessary latency to every inference call
            // - Could fail under heavy load or on different CPU architectures
            // =======================================================================
            ResetInferenceWorkContext(work_ctx.get());
            ResetQwen36Profile(work_ctx.get());
            SetCurrentBatch(&batch);

            // Note: Memory ordering for GGML worker threads is handled internally
            // by GGML's thread pool (uses CV/mutex). No explicit fence needed.

            // Keep backend thread count and custom-op n_tasks synchronized.
            maybe_set_cpu_threads();

            static int decode_graph_regression_checked_steps = 0;
            const bool decode_graph_regression_single_seq_hybrid =
                batch.num_seqs == 1 && current_model && current_model->arch_flags.is_hybrid_ssm &&
                densecore::models::DescribeModel(current_model).variant == ModelVariant::QWEN36;
            const bool run_decode_graph_cache_regression_check =
                cpu_backend_active && current_kv_cache && using_cached_decode_graph && decode_single_token_layout &&
                (batch.num_seqs > 1 || decode_graph_regression_single_seq_hybrid) && batch.lora_map.empty() &&
                IsDecodeGraphCacheRegressionEnabled() &&
                decode_graph_regression_checked_steps < DecodeGraphCacheRegressionSteps();
            const bool run_batched_decode_correctness_check =
                cpu_backend_active && IsBatchedDecodeCorrectnessCheckEnabled() && !is_embedding_batch &&
                !is_prefill_batch && current_kv_cache && batch.num_seqs > 1 &&
                static_cast<int>(batch.tokens.size()) == batch.num_seqs &&
                static_cast<int>(batch.seq_id.size()) == batch.num_seqs &&
                static_cast<int>(batch.pos.size()) == batch.num_seqs &&
                static_cast<int>(batch.block_tables.size()) == batch.num_seqs &&
                static_cast<int>(batch.n_past.size()) == batch.num_seqs && batch.lora_map.empty();
            const bool need_decode_check_kv_snapshot =
                run_batched_decode_correctness_check || run_decode_graph_cache_regression_check;

            std::vector<int> decode_check_write_blocks;
            std::vector<uint8_t> decode_check_pre_k;
            std::vector<uint8_t> decode_check_pre_v;
            bool decode_check_ready = false;

            if (need_decode_check_kv_snapshot) {
                // Snapshot only blocks written in this decode step for decode
                // correctness and graph-cache regression checks.
                bool valid_layout = true;
                for (int i = 0; i < batch.num_seqs; ++i) {
                    const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
                    if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
                        valid_layout = false;
                        break;
                    }

                    const int pos_i = batch.pos[static_cast<size_t>(i)];
                    const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
                    if (pos_i < 0 || block_table.empty()) {
                        valid_layout = false;
                        break;
                    }

                    const int logical_block = pos_i / BLOCK_SIZE;
                    if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
                        valid_layout = false;
                        break;
                    }

                    const int block_id = block_table[static_cast<size_t>(logical_block)];
                    if (block_id < 0 || block_id >= current_kv_cache->max_blocks) {
                        valid_layout = false;
                        break;
                    }
                    decode_check_write_blocks.push_back(block_id);
                }

                if (valid_layout && !decode_check_write_blocks.empty()) {
                    std::sort(decode_check_write_blocks.begin(), decode_check_write_blocks.end());
                    decode_check_write_blocks.erase(
                        std::unique(decode_check_write_blocks.begin(), decode_check_write_blocks.end()),
                        decode_check_write_blocks.end());
                    current_kv_cache->CopyBlocksToHost(decode_check_write_blocks, &decode_check_pre_k,
                                                       &decode_check_pre_v);
                    decode_check_ready = true;
                }
            }

            if (run_decode_graph_cache_regression_check) {
                if (!decode_check_ready) {
                    std::cerr << "[DecodeGraphCacheCheck] skipped: unable to snapshot decode-step KV pre-state"
                              << std::endl;
                } else {
                    try {
                        // Both cached and uncached verification runs write decode-step
                        // KV blocks, so each must start from the same pre-step KV
                        // state; restore again before the real decode compute so this
                        // regression mode does not perturb generation.
                        MaybeLogMoEGraphSummary(gf, current_model, is_prefill_batch);
                        ggml_backend_graph_compute(active_backend, gf);

                        if (!output || !output->data) {
                            std::cerr << "[DecodeGraphCacheCheck] skipped: cached output missing" << std::endl;
                        } else {
                            int n_vocab = current_model ? static_cast<int>(current_model->hparams.n_vocab)
                                                        : static_cast<int>(output->ne[0]);
                            if (n_vocab <= 0) {
                                n_vocab = static_cast<int>(output->ne[0]);
                            }
                            const int cached_rows = static_cast<int>(output->ne[0]);
                            const int cached_cols = static_cast<int>(output->ne[1]);
                            const int expected_cols = batch.num_seqs;
                            if (cached_rows != n_vocab || cached_cols != expected_cols) {
                                std::cerr
                                    << "[DecodeGraphCacheCheck] skipped due to shape mismatch: cached logits shape "
                                    << "(" << cached_rows << "," << cached_cols << ")" << " expected=(" << n_vocab
                                    << "," << expected_cols << ")" << std::endl;
                            } else {
                                const ptrdiff_t cached_row_stride =
                                    static_cast<ptrdiff_t>(output->nb[1] / sizeof(float));
                                const float* cached_logits_base = reinterpret_cast<const float*>(output->data);
                                std::vector<float> cached_logits_snapshot;
                                cached_logits_snapshot.resize(static_cast<size_t>(batch.num_seqs) *
                                                              static_cast<size_t>(n_vocab));
                                for (int i = 0; i < batch.num_seqs; ++i) {
                                    const float* src_row =
                                        cached_logits_base + static_cast<ptrdiff_t>(i) * cached_row_stride;
                                    float* dst_row = cached_logits_snapshot.data() +
                                                     static_cast<size_t>(i) * static_cast<size_t>(n_vocab);
                                    std::memcpy(dst_row, src_row, static_cast<size_t>(n_vocab) * sizeof(float));
                                }

                                current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_pre_k,
                                                                        decode_check_pre_v);
                                ResetInferenceWorkContext(work_ctx.get());
                                SetCurrentBatch(&batch);

                                struct ggml_init_params uncached_params = {
                                    .mem_size = DecodeGraphCacheCtxBytes(),
                                    .mem_buffer = nullptr,
                                    .no_alloc = false,
                                };
                                struct ggml_context* uncached_ctx = ggml_init(uncached_params);
                                densecore::GGMLContextGuard uncached_ctx_guard(uncached_ctx);
                                if (!uncached_ctx) {
                                    std::cerr << "[DecodeGraphCacheCheck] skipped: failed to allocate uncached verify "
                                                 "ctx (bytes="
                                              << DecodeGraphCacheCtxBytes() << ")" << std::endl;
                                } else {
                                    struct ggml_cgraph* uncached_gf = ggml_new_graph_custom(uncached_ctx, 32768, false);
                                    if (!uncached_gf) {
                                        std::cerr
                                            << "[DecodeGraphCacheCheck] skipped: failed to create uncached verify graph"
                                            << std::endl;
                                    } else {
                                        struct ggml_tensor* uncached_output = nullptr;
                                        struct ggml_tensor* uncached_embd = nullptr;
                                        struct ggml_tensor* uncached_pos = nullptr;
                                        uncached_output = BuildTransformerGraph(
                                            current_model, current_kv_cache, uncached_ctx, batch, is_embedding_batch,
                                            uncached_gf, &uncached_embd, &uncached_pos);
                                        if (!uncached_output || !uncached_embd || !uncached_pos) {
                                            std::cerr << "[DecodeGraphCacheCheck] skipped: failed to build uncached "
                                                         "verify graph"
                                                      << std::endl;
                                        } else {
                                            const size_t uncached_embd_bytes = ggml_nbytes(uncached_embd);
                                            std::vector<uint8_t> uncached_inputs(uncached_embd_bytes +
                                                                                 ggml_nbytes(uncached_pos) + 256);
                                            uncached_embd->data = uncached_inputs.data();
                                            uncached_pos->data = uncached_inputs.data() + uncached_embd_bytes + 256;
                                            std::memcpy(uncached_embd->data, batch.tokens.data(),
                                                        batch.tokens.size() * sizeof(int));
                                            PopulatePositionTensor(current_model, batch, uncached_pos);

                                            maybe_set_cpu_threads();

                                            MaybeLogMoEGraphSummary(uncached_gf, current_model, is_prefill_batch);
                                            ggml_backend_graph_compute(active_backend, uncached_gf);
                                            if (!uncached_output->data) {
                                                std::cerr << "[DecodeGraphCacheCheck] skipped due to shape mismatch: "
                                                             "uncached "
                                                             "output missing"
                                                          << std::endl;
                                            } else {
                                                const int uncached_rows = static_cast<int>(uncached_output->ne[0]);
                                                const int uncached_cols = static_cast<int>(uncached_output->ne[1]);
                                                if (uncached_rows != n_vocab || uncached_cols != expected_cols) {
                                                    std::cerr << "[DecodeGraphCacheCheck] skipped due to shape "
                                                                 "mismatch: uncached "
                                                                 "logits shape ("
                                                              << uncached_rows << "," << uncached_cols << ")"
                                                              << " expected=(" << n_vocab << "," << expected_cols << ")"
                                                              << std::endl;
                                                } else {
                                                    const float tol = DecodeGraphCacheRegressionTolerance();
                                                    const ptrdiff_t uncached_row_stride =
                                                        static_cast<ptrdiff_t>(uncached_output->nb[1] / sizeof(float));
                                                    const float* uncached_logits_base =
                                                        reinterpret_cast<const float*>(uncached_output->data);

                                                    auto argmax_finite = [](const float* logits, int len) {
                                                        int best_idx = 0;
                                                        float best_val = -INFINITY;
                                                        bool found = false;
                                                        for (int i = 0; i < len; ++i) {
                                                            const float v = logits[i];
                                                            if (std::isfinite(v) && (!found || v > best_val)) {
                                                                best_val = v;
                                                                best_idx = i;
                                                                found = true;
                                                            }
                                                        }
                                                        return best_idx;
                                                    };

                                                    float worst_max_abs = 0.0f;
                                                    int worst_seq = -1;
                                                    int worst_cached_argmax = -1;
                                                    int worst_uncached_argmax = -1;
                                                    int argmax_diff_count = 0;
                                                    int mismatch_count = 0;
                                                    for (int i = 0; i < batch.num_seqs; ++i) {
                                                        const float* cached_row =
                                                            cached_logits_snapshot.data() +
                                                            static_cast<size_t>(i) * static_cast<size_t>(n_vocab);
                                                        const float* uncached_row =
                                                            uncached_logits_base +
                                                            static_cast<ptrdiff_t>(i) * uncached_row_stride;

                                                        float max_abs_diff = 0.0f;
                                                        for (int v = 0; v < n_vocab; ++v) {
                                                            const float diff =
                                                                std::fabs(cached_row[v] - uncached_row[v]);
                                                            if (diff > max_abs_diff) {
                                                                max_abs_diff = diff;
                                                            }
                                                        }
                                                        const int cached_argmax = argmax_finite(cached_row, n_vocab);
                                                        const int uncached_argmax =
                                                            argmax_finite(uncached_row, n_vocab);
                                                        if (cached_argmax != uncached_argmax) {
                                                            ++argmax_diff_count;
                                                        }
                                                        if (max_abs_diff > tol || cached_argmax != uncached_argmax) {
                                                            ++mismatch_count;
                                                        }
                                                        if (worst_seq < 0 || max_abs_diff >= worst_max_abs) {
                                                            worst_max_abs = max_abs_diff;
                                                            worst_seq = i;
                                                            worst_cached_argmax = cached_argmax;
                                                            worst_uncached_argmax = uncached_argmax;
                                                        }
                                                    }

                                                    const int step_idx = decode_graph_regression_checked_steps + 1;
                                                    const int target_steps = DecodeGraphCacheRegressionSteps();
                                                    std::cerr << "[DecodeGraphCacheCheck] step=" << step_idx << "/"
                                                              << target_steps << " max_abs_diff=" << worst_max_abs
                                                              << " worst_seq=" << worst_seq
                                                              << " argmax_diff_count=" << argmax_diff_count << "/"
                                                              << batch.num_seqs << std::endl;

                                                    if (mismatch_count > 0) {
                                                        throw densecore::InvalidArgumentException(
                                                            "Decode graph cache regression failed: cached vs uncached "
                                                            "mismatch "
                                                            "(seq=" +
                                                            std::to_string(worst_seq) +
                                                            ", max_abs_diff=" + std::to_string(worst_max_abs) +
                                                            ", argmax_diff_count=" + std::to_string(argmax_diff_count) +
                                                            "/" + std::to_string(batch.num_seqs) +
                                                            ", argmax(cached/uncached)=" +
                                                            std::to_string(worst_cached_argmax) + "/" +
                                                            std::to_string(worst_uncached_argmax) +
                                                            ", tol=" + std::to_string(tol) + ")");
                                                    }

                                                    ++decode_graph_regression_checked_steps;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } catch (const std::exception&) {
                        current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_pre_k,
                                                                decode_check_pre_v);
                        ResetInferenceWorkContext(work_ctx.get());
                        SetCurrentBatch(&batch);
                        throw;
                    }
                    current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_pre_k,
                                                            decode_check_pre_v);
                    ResetInferenceWorkContext(work_ctx.get());
                    SetCurrentBatch(&batch);
                }
            }

            const auto compute_begin = std::chrono::steady_clock::now();
            ValidateMulNodesOrThrow(gf, is_decode_batch ? "decode" : "prefill");
            MaybeLogMoEGraphSummary(gf, current_model, is_prefill_batch);
            DebugDumpGraphNodes(gf, is_decode_batch ? "decode" : "prefill");
            ResetMoEStrictFailure();
            ggml_backend_graph_compute(active_backend, gf);
            const auto compute_end = std::chrono::steady_clock::now();
            const auto graph_execute_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(compute_end - compute_begin).count());
            const Qwen36ProfileSnapshot qwen36_profile = GetQwen36ProfileSnapshot(work_ctx.get());
            for (Request* req : batch_requests) {
                if (req) {
                    req->graph_execute_ns += graph_execute_ns;
                    req->attention_ns += qwen36_profile.attention_ns;
                    req->paged_attention_ns += qwen36_profile.paged_attention_ns;
                    req->standard_attention_ns += qwen36_profile.standard_attention_ns;
                    req->portable_flash_attention_ns += qwen36_profile.portable_flash_attention_ns;
                    req->native_flash_attention_ns += qwen36_profile.native_flash_attention_ns;
                    req->hal_attention_ns += qwen36_profile.hal_attention_ns;
                    req->attention_repack_ns += qwen36_profile.attention_repack_ns;
                    req->moe_forward_ns += qwen36_profile.moe_forward_ns;
                    req->shared_expert_ns += qwen36_profile.shared_expert_ns;
                    req->quant_matmul_ns += qwen36_profile.quant_matmul_ns;
                    req->kv_update_ns += qwen36_profile.kv_update_ns;
                    req->sample_ns += qwen36_profile.sample_ns;
                    req->graph_cache_hit_count += qwen36_profile.graph_cache_hits;
                    req->graph_cache_miss_count += qwen36_profile.graph_cache_misses;
                    req->q4k_true_batched_used =
                        std::max(req->q4k_true_batched_used, qwen36_profile.q4k_true_batched_used);
                    req->arm_batched_quant_used =
                        std::max(req->arm_batched_quant_used, qwen36_profile.arm_batched_quant_used);
                    req->attention_path_paged =
                        std::max(req->attention_path_paged, qwen36_profile.attention_path_paged);
                    req->attention_path_standard =
                        std::max(req->attention_path_standard, qwen36_profile.attention_path_standard);
                    req->attention_path_portable_flash =
                        std::max(req->attention_path_portable_flash, qwen36_profile.attention_path_portable_flash);
                    req->attention_path_native_flash =
                        std::max(req->attention_path_native_flash, qwen36_profile.attention_path_native_flash);
                    req->attention_path_hal = std::max(req->attention_path_hal, qwen36_profile.attention_path_hal);
                    req->moe_task_count = std::max(req->moe_task_count, qwen36_profile.moe_task_count);
                    req->selected_expert_count =
                        std::max(req->selected_expert_count, qwen36_profile.selected_expert_count);
                }
            }
            std::string moe_strict_failure;
            if (ConsumeMoEStrictFailure(&moe_strict_failure)) {
                if (moe_strict_failure.empty()) {
                    moe_strict_failure = "MoE strict mode failure";
                }
                LOG_ERROR("Failing batch after MoE strict failure: {}", moe_strict_failure);
                for (Request* req : batch_requests) {
                    if (!req || req->finished) {
                        continue;
                    }
                    req->finished = true;
                    state->metrics.failed_requests++;
                    EmitRequestResult(state, req, "Error: " + moe_strict_failure, -1, true, true,
                                      global_direct_callback);
                    if (!req->block_table.empty()) {
                        current_kv_cache->block_manager->Free(req->block_table);
                        req->block_table.clear();
                    }
                    if (req->seq_id >= 0) {
                        state->scheduler->RemoveRequest(req->seq_id, false);
                        seq_to_request.erase(req->seq_id);
                    }
                    {
                        std::lock_guard<std::mutex> lk(req->mu);
                        req->cv.notify_all();
                    }
                }
                reap_finished_requests();
                continue;
            }
            if (Request* trace_req = !batch_requests.empty() ? batch_requests[0] : nullptr) {
                LogPrefillStage("after_prefill_backend_execute", trace_req, current_model, &state->inference_ctx,
                                input_base, input_span_bytes, output, is_prefill_batch, active_threads);
                LogPrefillStage("after_prefill_logits_extraction", trace_req, current_model, &state->inference_ctx,
                                input_base, input_span_bytes, output, is_prefill_batch, active_threads);
            }
            LOG_TRACE("Graph compute done for batch size {}", batch.num_seqs);
            if (is_decode_batch && decode_batch_size >= 1 && decode_batch_size <= 4) {
                GetDecodeWorkerStats().decode_batches.fetch_add(1, std::memory_order_relaxed);
                MaybeLogDecodeRuntimeStats();
            }

            if (run_batched_decode_correctness_check && decode_check_ready && output && output->data) {
                std::vector<uint8_t> decode_check_post_k;
                std::vector<uint8_t> decode_check_post_v;
                current_kv_cache->CopyBlocksToHost(decode_check_write_blocks, &decode_check_post_k,
                                                   &decode_check_post_v);
                try {
                    const int n_vocab = static_cast<int>(output->ne[0]);
                    const int n_cols = std::max(1, static_cast<int>(output->ne[1]));
                    const ptrdiff_t batched_row_stride = static_cast<ptrdiff_t>(output->nb[1] / sizeof(float));
                    const float* batched_logits_base = reinterpret_cast<const float*>(output->data);
                    std::vector<float> batched_logits_snapshot;
                    batched_logits_snapshot.resize(static_cast<size_t>(batch.num_seqs) * static_cast<size_t>(n_vocab));
                    for (int i = 0; i < batch.num_seqs; ++i) {
                        const int src_col = std::min(i, n_cols - 1);
                        const float* src_row =
                            batched_logits_base + static_cast<ptrdiff_t>(src_col) * batched_row_stride;
                        float* dst_row =
                            batched_logits_snapshot.data() + static_cast<size_t>(i) * static_cast<size_t>(n_vocab);
                        std::memcpy(dst_row, src_row, static_cast<size_t>(n_vocab) * sizeof(float));
                    }
                    const float tol = BatchedDecodeCorrectnessTolerance();
                    const size_t check_ctx_bytes = BatchedDecodeCorrectnessContextBytes();

                    auto argmax_finite = [](const float* logits, int len) {
                        int best_idx = 0;
                        float best_val = -INFINITY;
                        bool found = false;
                        for (int i = 0; i < len; ++i) {
                            const float v = logits[i];
                            if (std::isfinite(v) && (!found || v > best_val)) {
                                best_val = v;
                                best_idx = i;
                                found = true;
                            }
                        }
                        return best_idx;
                    };

                    int mismatch_count = 0;
                    bool check_completed = true;
                    float worst_max_abs = 0.0f;
                    int worst_seq = -1;
                    int worst_batch_argmax = -1;
                    int worst_ref_argmax = -1;

                    for (int i = 0; i < batch.num_seqs; ++i) {
                        current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_pre_k,
                                                                decode_check_pre_v);

                        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
                        if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
                            continue;
                        }

                        BatchSpec single_batch;
                        single_batch.input_kind = BatchInputKind::Tokens;
                        single_batch.tokens.push_back(batch.tokens[static_cast<size_t>(i)]);
                        single_batch.pos.push_back(batch.pos[static_cast<size_t>(i)]);
                        single_batch.seq_id.push_back(0);
                        single_batch.block_tables.push_back(batch.block_tables[static_cast<size_t>(seq_idx)]);
                        single_batch.n_past.push_back(batch.n_past[static_cast<size_t>(seq_idx)]);
                        if (seq_idx < static_cast<int>(batch.scheduler_seq_ids.size())) {
                            single_batch.scheduler_seq_ids.push_back(
                                batch.scheduler_seq_ids[static_cast<size_t>(seq_idx)]);
                        }
                        single_batch.scheduler = batch.scheduler;
                        single_batch.num_seqs = 1;
                        single_batch.deps = &deps;

                        GenericInput single_input;
                        single_input.kind = BatchInputKind::Tokens;
                        single_input.tokens.push_back(batch.tokens[static_cast<size_t>(i)]);
                        single_batch.inputs.push_back(std::move(single_input));

                        ResetInferenceWorkContext(work_ctx.get());
                        SetCurrentBatch(&single_batch);

                        struct ggml_init_params verify_params = {
                            .mem_size = check_ctx_bytes,
                            .mem_buffer = nullptr,
                            .no_alloc = false,
                        };
                        struct ggml_context* verify_ctx = ggml_init(verify_params);
                        densecore::GGMLContextGuard verify_ctx_guard(verify_ctx);
                        if (!verify_ctx) {
                            std::cerr << "[DecodeCorrectness] skipped: failed to allocate verify ctx (bytes="
                                      << check_ctx_bytes << ")" << std::endl;
                            check_completed = false;
                            break;
                        }

                        struct ggml_cgraph* verify_gf = ggml_new_graph_custom(verify_ctx, 32768, false);
                        if (!verify_gf) {
                            std::cerr << "[DecodeCorrectness] skipped: failed to create verify graph for seq " << i
                                      << std::endl;
                            check_completed = false;
                            break;
                        }

                        struct ggml_tensor* verify_output = nullptr;
                        struct ggml_tensor* verify_embd = nullptr;
                        struct ggml_tensor* verify_pos = nullptr;
                        verify_output = BuildTransformerGraph(current_model, current_kv_cache, verify_ctx, single_batch,
                                                              false, verify_gf, &verify_embd, &verify_pos);
                        if (!verify_output || !verify_embd || !verify_pos) {
                            std::cerr << "[DecodeCorrectness] skipped: failed to build verify graph for seq " << i
                                      << std::endl;
                            check_completed = false;
                            break;
                        }

                        const size_t verify_embd_bytes = ggml_nbytes(verify_embd);
                        const size_t verify_pos_bytes = ggml_nbytes(verify_pos);
                        std::vector<uint8_t> verify_input(verify_embd_bytes + verify_pos_bytes + 256);
                        verify_embd->data = verify_input.data();
                        verify_pos->data = verify_input.data() + verify_embd_bytes + 256;
                        std::memcpy(verify_embd->data, single_batch.tokens.data(), sizeof(int));
                        PopulatePositionTensor(current_model, single_batch, verify_pos);

                        maybe_set_cpu_threads();
                        MaybeLogMoEGraphSummary(verify_gf, current_model, is_prefill_batch);
                        ggml_backend_graph_compute(active_backend, verify_gf);

                        if (!verify_output->data || static_cast<int>(verify_output->ne[0]) != n_vocab) {
                            std::cerr << "[DecodeCorrectness] skipped: verify output shape mismatch for seq " << i
                                      << std::endl;
                            continue;
                        }

                        const float* verify_row = reinterpret_cast<const float*>(verify_output->data);
                        const float* batched_row =
                            batched_logits_snapshot.data() + static_cast<size_t>(i) * static_cast<size_t>(n_vocab);

                        float max_abs_diff = 0.0f;
                        for (int v = 0; v < n_vocab; ++v) {
                            const float diff = std::fabs(batched_row[v] - verify_row[v]);
                            if (diff > max_abs_diff) {
                                max_abs_diff = diff;
                            }
                        }

                        const int batch_argmax = argmax_finite(batched_row, n_vocab);
                        const int ref_argmax = argmax_finite(verify_row, n_vocab);
                        if (max_abs_diff > tol || batch_argmax != ref_argmax) {
                            mismatch_count++;
                            if (max_abs_diff >= worst_max_abs) {
                                worst_max_abs = max_abs_diff;
                                worst_seq = i;
                                worst_batch_argmax = batch_argmax;
                                worst_ref_argmax = ref_argmax;
                            }
                        }
                    }

                    current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_post_k,
                                                            decode_check_post_v);
                    ResetInferenceWorkContext(work_ctx.get());
                    SetCurrentBatch(&batch);

                    if (!check_completed) {
                        std::cerr << "[DecodeCorrectness] skipped: reference replay incomplete" << std::endl;
                    } else if (mismatch_count > 0) {
                        std::cerr << "[DecodeCorrectness] mismatches=" << mismatch_count << "/" << batch.num_seqs
                                  << " tol=" << tol << " worst_seq=" << worst_seq << " max_abs=" << worst_max_abs
                                  << " argmax(batch/ref)=" << worst_batch_argmax << "/" << worst_ref_argmax
                                  << std::endl;
                        if (IsBatchedDecodeCorrectnessAbortEnabled()) {
                            throw densecore::InvalidArgumentException(
                                "Batched decode correctness check failed (set DENSECORE_CHECK_BATCHED_DECODE_ABORT=0 "
                                "to continue).");
                        }
                    } else {
                        std::cerr << "[DecodeCorrectness] passed bs=" << batch.num_seqs << " tol=" << tol << std::endl;
                    }
                } catch (const std::exception& e) {
                    current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_post_k,
                                                            decode_check_post_v);
                    ResetInferenceWorkContext(work_ctx.get());
                    SetCurrentBatch(&batch);
                    std::cerr << "[DecodeCorrectness] skipped: " << e.what() << std::endl;
                }
            }

            // DENSECORE_PROFILE_DECODE=1: per-iteration graph build/compute timing
            if (IsDecodeProfileEnabled() && !is_prefill_batch && !is_embedding_batch) {
                static thread_local int profile_iter_count = 0;
                if (++profile_iter_count % 100 == 0) {
                    const double graph_compute_ms =
                        std::chrono::duration<double, std::milli>(compute_end - compute_begin).count();
                    // graph_build timing only available when graph was freshly built
                    if (!reused_decode_graph) {
                        const double graph_build_ms =
                            std::chrono::duration<double, std::milli>(graph_build_end - graph_build_begin).count();
                        const double total_ms = graph_build_ms + graph_compute_ms;
                        fprintf(stderr,
                                "[DecodeProfile] bs=%d threads=%d graph_build_ms=%.1f graph_compute_ms=%.1f "
                                "total_ms=%.1f\n",
                                batch.num_seqs, active_threads, graph_build_ms, graph_compute_ms, total_ms);
                    } else {
                        fprintf(stderr,
                                "[DecodeProfile] bs=%d threads=%d graph_build_ms=cached graph_compute_ms=%.1f\n",
                                batch.num_seqs, active_threads, graph_compute_ms);
                    }
                }
            }

            // Lightweight decode-batch telemetry for heterogeneous paged decode.
            // Enable with: DENSECORE_LOG_DECODE_BATCH_TPS=1
            // Recommended benchmark sweep: batch sizes 1,4,8,16 with mixed n_past.
            // Correctness check: replay the same prompts one-by-one with
            // deterministic sampling and compare generated text to batched mode.
            if (IsDecodeBatchPerfLoggingEnabled() && !is_embedding_batch && !is_prefill_batch && batch.num_seqs > 0 &&
                static_cast<int>(batch.tokens.size()) == batch.num_seqs &&
                static_cast<int>(batch.n_past.size()) == batch.num_seqs) {
                const double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::duration<double>>(compute_end - compute_begin).count();
                if (elapsed_sec > 0.0) {
                    int min_ctx = std::numeric_limits<int>::max();
                    int max_ctx = 0;
                    int64_t sum_ctx = 0;
                    for (int i = 0; i < batch.num_seqs; ++i) {
                        const int ctx_i = std::max(1, batch.n_past[static_cast<size_t>(i)] + 1);
                        min_ctx = std::min(min_ctx, ctx_i);
                        max_ctx = std::max(max_ctx, ctx_i);
                        sum_ctx += ctx_i;
                    }
                    const double avg_ctx = static_cast<double>(sum_ctx) / static_cast<double>(batch.num_seqs);
                    const double tps = static_cast<double>(batch.num_seqs) / elapsed_sec;
                    std::cerr << "[DecodeBatchPerf] bs=" << batch.num_seqs << " step_tps=" << tps
                              << " ctx_min=" << min_ctx << " ctx_avg=" << avg_ctx << " ctx_max=" << max_ctx
                              << std::endl;
                }
            }

            // Note: GGML's thread pool join provides acquire semantics.
            // No explicit fence needed to see results.

            // 10. Process Outputs
            int token_offset = 0;
            int n_embd = current_model->hparams.n_embd;

            for (int i = 0; i < batch.num_seqs; ++i) {
                Request* req = batch_requests[i];
                // std::cerr << "[DEBUG] EngineLoop: Processing output for request " << req->id << std::endl;
                int processed_count = (i < static_cast<int>(batch_token_counts.size())) ? batch_token_counts[i] : 1;

                int last_token_idx = token_offset + processed_count - 1;

                if (is_embedding_batch) {
                    float* data = (float*)output->data;
                    int seq_len = processed_count;
                    float* hidden_states = data + token_offset * n_embd;
                    std::vector<float> embedding(n_embd);

                    switch (req->pooling_type) {
                    case densecore::PoolingStrategy::MEAN:
                        densecore::simd::MeanPool(hidden_states, embedding.data(), seq_len, n_embd);
                        break;
                    case densecore::PoolingStrategy::CLS:
                        densecore::simd::ClsPool(hidden_states, embedding.data(), n_embd);
                        break;
                    case densecore::PoolingStrategy::LAST:
                        densecore::simd::LastPool(hidden_states, embedding.data(), seq_len, n_embd);
                        break;
                    case densecore::PoolingStrategy::MAX:
                        densecore::simd::MaxPool(hidden_states, embedding.data(), seq_len, n_embd);
                        break;
                    default: densecore::simd::MeanPool(hidden_states, embedding.data(), seq_len, n_embd); break;
                    }

                    if (req->normalize_embedding) {
                        densecore::simd::NormalizeL2(embedding.data(), n_embd);
                    }

                    // Push embedding result to callback queue (RAII-safe via std::move)
                    if (req->embedding_callback) {
                        // Move the embedding vector directly - no manual allocation needed
                        PushEmbeddingResultEvent(state, req->id, std::move(embedding), req->embedding_callback,
                                                 req->user_data);
                    }
                    req->finished = true;

                    // Cleanup and remove from scheduler
                    current_kv_cache->block_manager->Free(req->block_table);
                    req->block_table.clear();
                    // Cleanup using stored seq_id (O(1) instead of O(n))
                    if (req->seq_id >= 0) {
                        state->scheduler->RemoveRequest(req->seq_id, true);
                        seq_to_request.erase(req->seq_id);
                    }
                    {
                        std::lock_guard<std::mutex> lk(req->mu);
                        req->cv.notify_all();
                    }
                } else {
                    // Generation
                    const bool was_prefill_step = req->is_prefill;
                    const int n_past_before_step = req->n_past;
                    const int remaining_prompt_tokens_before_step =
                        was_prefill_step ? static_cast<int>(req->tokens.size()) : 0;
                    auto commit_prefill_scheduler_progress = [&](bool prefill_finished) {
                        if (!state->scheduler || req->seq_id < 0) {
                            req->pending_scheduler_progress = 0;
                            return;
                        }
                        state->scheduler->OnPrefillChunkComplete(req->seq_id, processed_count, prefill_finished);
                        req->pending_scheduler_progress = 0;
                    };
                    auto commit_decode_scheduler_progress = [&]() {
                        if (!state->scheduler || req->seq_id < 0) {
                            req->pending_scheduler_progress = 0;
                            return;
                        }
                        state->scheduler->OnDecodeStepComplete(req->seq_id, processed_count);
                        req->pending_scheduler_progress = 0;
                    };
                    if (req->is_prefill) {
                        const int remaining_prompt_tokens = static_cast<int>(req->tokens.size());
                        req->n_past += processed_count;
                        req->empty_schedule_stall_count = 0;
                        req->last_progress_time = std::chrono::steady_clock::now();

                        // Register any newly completed full blocks immediately after
                        // this prefill chunk so prefix reuse can restore the exact
                        // hybrid SSM boundary state for the latest completed block.
                        if (prefix_cache_allowed && !req->prompt_tokens_for_cache.empty() &&
                            !req->block_table.empty() && req->n_past > 0) {
                            const int* tokens_ptr = req->prompt_tokens_for_cache.data();
                            int total_tokens = static_cast<int>(req->prompt_tokens_for_cache.size());
                            const int completed_blocks =
                                std::min(req->n_past / BLOCK_SIZE, static_cast<int>(req->block_table.size()));

                            for (int blk_idx = req->registered_prefix_blocks; blk_idx < completed_blocks; ++blk_idx) {
                                int block_id = req->block_table[static_cast<size_t>(blk_idx)];
                                int start_token = blk_idx * BLOCK_SIZE;
                                int block_tokens = std::min(BLOCK_SIZE, total_tokens - start_token);
                                if (block_tokens != BLOCK_SIZE) {
                                    break;
                                }

                                uint64_t hash = BlockManager::ComputeTokenHash(tokens_ptr + start_token, block_tokens);
                                const bool attach_hybrid_snapshot = current_model &&
                                                                    current_model->arch_flags.is_hybrid_ssm &&
                                                                    ((blk_idx + 1) * BLOCK_SIZE == req->n_past);
                                if (attach_hybrid_snapshot) {
                                    DebugLogHybridSSMSnapshot("save", req->id, req->n_past, block_id,
                                                              req->ssm_runtime_states);
                                }
                                current_kv_cache->block_manager->RegisterPrefixBlockWithTokens(
                                    block_id, hash, tokens_ptr + start_token, block_tokens,
                                    attach_hybrid_snapshot ? &req->ssm_runtime_states : nullptr);
                            }
                            req->registered_prefix_blocks = std::max(req->registered_prefix_blocks, completed_blocks);
                        }

                        // Chunked prefill in progress: continue prefill without sampling.
                        if (processed_count < remaining_prompt_tokens) {
                            commit_prefill_scheduler_progress(/*prefill_finished=*/false);
                            determinism_chunked_prefill_reqs.insert(req->id);
                            req->tokens.erase(req->tokens.begin(), req->tokens.begin() + processed_count);
                            token_offset += processed_count;
                            continue;
                        }

                        commit_prefill_scheduler_progress(/*prefill_finished=*/true);
                        req->tokens.clear();
                        req->is_prefill = false;
                        // Clear after registration, or after skipping cache registration for hybrid SSM.
                        req->prompt_tokens_for_cache.clear();

                        req->first_token_time = std::chrono::steady_clock::now();
                        auto ttft_us = std::chrono::duration_cast<std::chrono::microseconds>(req->first_token_time -
                                                                                             req->start_time)
                                           .count();
                        state->metrics.RecordTTFT(ttft_us);
                        req->last_token_time = req->first_token_time;
                        LogDeterminismBoundary(
                            "after_prefill_complete", req, current_model, prefix_cache_allowed,
                            /*prefix_cache_hit=*/determinism_prefix_hit_reqs.count(req->id) != 0,
                            /*hybrid_restore_attempted=*/determinism_hybrid_restore_attempt_reqs.count(req->id) != 0,
                            /*hybrid_restore_applied=*/determinism_hybrid_restore_applied_reqs.count(req->id) != 0,
                            /*chunked_prefill=*/determinism_chunked_prefill_reqs.count(req->id) != 0,
                            /*decode_cache_active=*/using_cached_decode_graph,
                            /*decode_cache_reused=*/reused_decode_graph,
                            /*prefill_cache_active=*/using_cached_prefill_graph);
                    }

                    int sampling_logits_idx = last_token_idx;
                    std::string sampling_guard_error;
                    if (!ResolveSamplingLogitsColumnForRequest(
                            token_offset, processed_count, static_cast<int>(output->ne[1]), was_prefill_step,
                            remaining_prompt_tokens_before_step, n_past_before_step, req->n_past, &sampling_logits_idx,
                            &sampling_guard_error)) {
                        throw densecore::InvalidArgumentException("invalid sampling logits column for req " +
                                                                  std::to_string(req->id) + ": " +
                                                                  sampling_guard_error);
                    }
                    const int debug_sampling_logits_offset = GetDebugSamplingLogitsOffset();
                    if (debug_sampling_logits_offset != 0) {
                        const int adjusted = sampling_logits_idx + debug_sampling_logits_offset;
                        if (adjusted < 0 || adjusted >= static_cast<int>(output->ne[1])) {
                            throw densecore::InvalidArgumentException(
                                "debug sampling logits offset moved column out of bounds for req " +
                                std::to_string(req->id));
                        }
                        sampling_logits_idx = adjusted;
                    }
                    const int debug_first_token_sampling_logits_offset = GetDebugFirstTokenSamplingLogitsOffset();
                    if (req->generated_count == 0 && debug_first_token_sampling_logits_offset != 0) {
                        const int adjusted = sampling_logits_idx + debug_first_token_sampling_logits_offset;
                        if (adjusted < 0 || adjusted >= static_cast<int>(output->ne[1])) {
                            throw densecore::InvalidArgumentException(
                                "debug first-token sampling logits offset moved column out of bounds for req " +
                                std::to_string(req->id));
                        }
                        sampling_logits_idx = adjusted;
                    }

                    // Sample token
                    SamplingParams sampling_params = req->sampling_params;
                    sampling_params.token_history = &req->token_history;
                    sampling_params.disallowed_token_ids = &req->disallowed_token_ids;
                    sampling_params.request_id = req->id;
                    sampling_params.output_token_index = req->generated_count;
                    if (std::getenv("DENSECORE_DEBUG_SAMPLE") != nullptr ||
                        std::getenv("DENSECORE_DEBUG_SAMPLE_TOP") != nullptr) {
                        sampling_params.vocab = &current_model->vocab_tokens;
                    }
                    if (req->json_mode) {
                        sampling_params.grammar = &req->grammar;
                        sampling_params.vocab = &current_model->vocab_tokens;
                    }
                    if (req->generated_count == 0) {
                        LogDeterminismBoundary(
                            "pre_decode_token0_lm_head", req, current_model, prefix_cache_allowed,
                            /*prefix_cache_hit=*/determinism_prefix_hit_reqs.count(req->id) != 0,
                            /*hybrid_restore_attempted=*/determinism_hybrid_restore_attempt_reqs.count(req->id) != 0,
                            /*hybrid_restore_applied=*/determinism_hybrid_restore_applied_reqs.count(req->id) != 0,
                            /*chunked_prefill=*/determinism_chunked_prefill_reqs.count(req->id) != 0,
                            /*decode_cache_active=*/using_cached_decode_graph,
                            /*decode_cache_reused=*/reused_decode_graph,
                            /*prefill_cache_active=*/using_cached_prefill_graph);
                    }

                    const auto sample_begin = std::chrono::steady_clock::now();
                    int best_token = SampleToken(output, sampling_logits_idx, sampling_params);
                    const auto sample_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                     std::chrono::steady_clock::now() - sample_begin)
                                                                     .count());
                    req->sample_ns += sample_ns;
                    if (IsVerboseTokenTraceEnabled()) {
                        std::cerr << "[TRACE] Sampled token " << best_token << " for request " << req->id << std::endl;
                    }
                    const auto decode_sample_time = std::chrono::steady_clock::now();
                    NoteDecodeSampleProgress(req, decode_sample_time, best_token);
                    bool terminal_error_emitted = false;

                    req->tokens.clear();
                    req->tokens.push_back(best_token);
                    req->generated_count++;
                    req->token_history.push_back(best_token);
                    constexpr size_t kMaxTokenHistory = 2048;
                    if (req->token_history.size() > kMaxTokenHistory) {
                        const size_t drop = req->token_history.size() - kMaxTokenHistory;
                        req->token_history.erase(req->token_history.begin(),
                                                 req->token_history.begin() +
                                                     static_cast<std::vector<int>::difference_type>(drop));
                    }
                    if (ShouldTerminateRepetitiveLoop(current_model, req)) {
                        req->finished = true;
                        req->decode_finish_cause = DecodeFinishCause::LoopGuard;
                    }

                    std::string token_str;
                    const bool bench_fast_path_text_enabled = []() {
                        const char* env = std::getenv("DENSECORE_BENCH_QA_TEXT");
                        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                    }();
                    const bool req_bench_fast_path = global_bench_fast_path && !req->json_mode;
                    const bool should_detokenize =
                        !req_bench_fast_path || bench_fast_path_text_enabled || req->token_result_callback != nullptr;
                    if (should_detokenize) {
                        std::string token_piece = Tokenizer::Detokenize(current_model, best_token);
                        if (const char* env = std::getenv("DENSECORE_DEBUG_FIRST_TOKEN_RAW");
                            env && env[0] != '\0' && best_token >= 0 &&
                            best_token < static_cast<int>(current_model->vocab_tokens.size()) &&
                            req->generated_count == 1) {
                            const std::string& raw_token = current_model->vocab_tokens[static_cast<size_t>(best_token)];
                            std::cerr << "[FIRST_TOKEN_RAW] id=" << best_token << " raw=" << raw_token
                                      << " detok=" << token_piece << std::endl;
                        }
                        if (!token_piece.empty()) {
                            if (req->utf8_pending.empty()) {
                                const size_t emit_len = Utf8ValidPrefixLength(token_piece);
                                if (emit_len == token_piece.size()) {
                                    token_str = std::move(token_piece);
                                } else if (emit_len > 0) {
                                    token_str.assign(token_piece.data(), emit_len);
                                    req->utf8_pending.assign(token_piece.data() + emit_len,
                                                             token_piece.size() - emit_len);
                                } else {
                                    req->utf8_pending = std::move(token_piece);
                                }
                            } else {
                                req->utf8_pending.append(token_piece);
                                const size_t emit_len = Utf8ValidPrefixLength(req->utf8_pending);
                                if (emit_len > 0) {
                                    token_str.assign(req->utf8_pending.data(), emit_len);
                                    req->utf8_pending.erase(0, emit_len);
                                }
                            }
                        }
                    }
                    const bool had_visible_text_before_suppression = !token_str.empty();

                    // Hide model-internal reasoning/tool blocks from streamed
                    // user output by default. Disable with
                    // DENSECORE_SUPPRESS_REASONING_TAGS=0.
                    if (!req_bench_fast_path && !req->json_mode && !token_str.empty() && req->suppress_reasoning_tags &&
                        IsReasoningTagSuppressionEnabled()) {
                        const bool may_contain_tag =
                            req->in_think_block || req->in_tool_call_block || req->in_tool_response_block ||
                            !req->think_tag_pending.empty() || !req->tool_call_tag_pending.empty() ||
                            !req->tool_response_tag_pending.empty() || token_str.find('<') != std::string::npos;
                        if (may_contain_tag) {
                            SuppressTaggedBlock(&token_str, &req->in_think_block, &req->think_tag_pending, "<think>",
                                                "</think>");
                            SuppressTaggedBlock(&token_str, &req->in_tool_call_block, &req->tool_call_tag_pending,
                                                "<tool_call>", "</tool_call>");
                            SuppressTaggedBlock(&token_str, &req->in_tool_response_block,
                                                &req->tool_response_tag_pending, "<tool_response>", "</tool_response>");
                        }
                    }
                    if (had_visible_text_before_suppression && token_str.empty()) {
                        NoteSuppressedToken(req);
                    }

                    if (req->json_mode && !token_str.empty()) {
                        req->grammar.UpdateState(token_str);
                    }

                    // =========================================================================
                    // TOKEN STREAMING VIA CALLBACK QUEUE (GIL-Free Hot Path)
                    // =========================================================================
                    // Push token to callback queue instead of invoking callback directly.
                    // This prevents the worker thread from blocking on Python GIL.
                    // =========================================================================
                    if (req->callback || req->token_result_callback) {
                        if (req_bench_fast_path && !should_detokenize) {
                            // Benchmark fast-path: emit one callback per generated token
                            // without detokenization/string processing overhead.
                            EmitRequestResult(state, req, "", best_token, false, false, global_direct_callback);
                        } else if (!token_str.empty() || req->token_result_callback) {
                            if (IsVerboseTokenTraceEnabled()) {
                                std::cerr << "[TRACE] Pushing result for request " << req->id << std::endl;
                            }
                            if (std::getenv("DENSECORE_PARITY_DEBUG") != nullptr &&
                                req->parity_debug_output_tokens < 64) {
                                std::cerr << "[ParityOutput] req=" << req->id
                                          << " index=" << req->parity_debug_output_tokens << " token_id=" << best_token
                                          << " text=" << token_str << std::endl;
                                req->parity_debug_output_tokens++;
                            }
                            EmitRequestResult(state, req, token_str, best_token, false, false, global_direct_callback);
                            if (!token_str.empty()) {
                                NoteVisibleEmitProgress(req, decode_sample_time, best_token);
                            }
                        }
                    }

                    state->metrics.total_tokens_generated++;

                    // Decode steps consume exactly one input token per iteration.
                    // Do not advance n_past on the prefill->decode transition token:
                    // that token is only sampled here and will be written to KV on
                    // the next decode iteration.
                    if (!was_prefill_step) {
                        req->n_past += processed_count;
                        auto now = std::chrono::steady_clock::now();
                        req->empty_schedule_stall_count = 0;
                        req->last_progress_time = now;
                        commit_decode_scheduler_progress();
                        auto itl_us =
                            std::chrono::duration_cast<std::chrono::microseconds>(now - req->last_token_time).count();
                        state->metrics.RecordITL(itl_us);
                        req->last_token_time = now;
                    }

                    // Ensure we have enough blocks for the NEXT token
                    int blocks_needed = (req->n_past + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    while ((int)req->block_table.size() < blocks_needed) {
                        auto new_blocks = current_kv_cache->block_manager->Allocate(1);
                        if (new_blocks.empty()) {
                            std::cerr << "[DenseCore] OOM during generation for req " << req->id << std::endl;
                            req->finished = true;
                            req->decode_finish_cause = DecodeFinishCause::OutOfMemory;
                            state->metrics.oom_errors++;
                            state->metrics.failed_requests++;
                            // Push OOM error to callback queue (instead of direct callback)
                            if (req->callback || req->token_result_callback) {
                                EmitRequestResult(state, req, "Error: Out of memory", -1, true, true,
                                                  global_direct_callback);
                                terminal_error_emitted = true;
                            }
                            break;
                        }
                        req->block_table.push_back(new_blocks[0]);
                    }

                    // Check stop sequences (best-effort, suffix match)
                    if (!req_bench_fast_path && !token_str.empty() && !req->stop_sequences.empty() &&
                        req->stop_buffer_max > 0) {
                        req->stop_buffer.append(token_str);
                        if (req->stop_buffer.size() > req->stop_buffer_max) {
                            req->stop_buffer.erase(0, req->stop_buffer.size() - req->stop_buffer_max);
                        }
                        for (const auto& seq : req->stop_sequences) {
                            if (!seq.empty() && req->stop_buffer.size() >= seq.size() &&
                                req->stop_buffer.compare(req->stop_buffer.size() - seq.size(), seq.size(), seq) == 0) {
                                req->finished = true;
                                req->decode_finish_cause = DecodeFinishCause::StopSequence;
                                break;
                            }
                        }
                    }

                    if (!req->finished && !req_bench_fast_path &&
                        HasDecodeVisibleProgressStalled(req, decode_sample_time)) {
                        LOG_ERROR("Failing decode-stalled request {} (seq_id={}, generated={}, silent_steps={})",
                                  req->id, req->seq_id, req->generated_count, req->decode_no_output_steps);
                        req->finished = true;
                        req->decode_finish_cause = DecodeFinishCause::DecodeVisibleProgressTimeout;
                        state->metrics.failed_requests++;
                        if (req->callback || req->token_result_callback) {
                            EmitRequestResult(state, req, "Error: decode made no externally visible progress", -1, true,
                                              true, global_direct_callback);
                            terminal_error_emitted = true;
                        }
                    }

                    // Check finish conditions
                    if (req->finished || IsStopTokenId(current_model, best_token) ||
                        req->generated_count >= req->max_tokens) {
                        if (!req->finished) {
                            if (IsStopTokenId(current_model, best_token)) {
                                req->decode_finish_cause = DecodeFinishCause::StopToken;
                            } else if (req->generated_count >= req->max_tokens) {
                                req->decode_finish_cause = DecodeFinishCause::MaxTokens;
                            }
                        }
                        req->finished = true;
                        if (IsMoEPathTraceDumpEnabled() && !IsMoETracePlumbingDisabled()) {
                            const auto forward_calls =
                                densecore::GetTelemetryCpuBackend().GetMoEForwardInvocationCount();
                            const auto graph_wiring = GetMoEGraphWiringDebugCounter();
                            const auto traces = densecore::GetTelemetryCpuBackend().GetMoEPathTraceSnapshot();
                            std::cerr << "[MOE_TRACE_DUMP] request_id=" << req->id
                                      << " generated_tokens=" << req->generated_count
                                      << " moe_graph_wiring_count=" << graph_wiring
                                      << " moe_forward_invocations=" << forward_calls
                                      << " moe_path_entries=" << traces.size() << std::endl;
                            for (const auto& entry : traces) {
                                if (entry.decode_step >= 2) {
                                    continue;
                                }
                                std::cerr << "[MOE_TRACE_DUMP] request_id=" << req->id
                                          << " layer_idx=" << entry.layer_idx << " expert_id=" << entry.expert_id
                                          << " seq_id=" << entry.seq_id << " token_idx=" << entry.token_idx
                                          << " decode_step=" << entry.decode_step << " n_past=" << entry.n_past
                                          << " force_safe_reference=" << (entry.force_safe_reference ? 1 : 0)
                                          << " safe_reference_mode=" << (entry.safe_reference_mode ? 1 : 0)
                                          << " selected_path=" << static_cast<int>(entry.selected_path) << std::endl;
                            }
                        }
                        if (IsSamplerTraceDumpEnabled()) {
                            const auto sampling_trace = GetSamplingDebugTraceSnapshot();
                            for (const auto& entry : sampling_trace) {
                                if (entry.request_id != req->id || entry.output_token_index > 1) {
                                    continue;
                                }
                                std::cerr << "[SAMPLER_TRACE_DUMP] request_id=" << req->id
                                          << " output_token_index=" << entry.output_token_index
                                          << " sampled_token_id=" << entry.sampled_token_id << std::endl;
                                std::cerr << "[SAMPLER_TRACE_DUMP] request_id=" << req->id
                                          << " output_token_index=" << entry.output_token_index << " top_pre_penalty=";
                                for (const auto& c : entry.top_pre_penalty) {
                                    std::cerr << c.token_id << ":" << c.pre_penalty_logit << ":" << c.post_penalty_logit
                                              << " ";
                                }
                                std::cerr << std::endl;
                                std::cerr << "[SAMPLER_TRACE_DUMP] request_id=" << req->id
                                          << " output_token_index=" << entry.output_token_index << " top_post_penalty=";
                                for (const auto& c : entry.top_post_penalty) {
                                    std::cerr << c.token_id << ":" << c.pre_penalty_logit << ":" << c.post_penalty_logit
                                              << " ";
                                }
                                std::cerr << std::endl;
                            }
                        }
                        if (!terminal_error_emitted && !req_bench_fast_path && !req->utf8_pending.empty() &&
                            (req->callback || req->token_result_callback)) {
                            const size_t emit_len = Utf8ValidPrefixLength(req->utf8_pending);
                            if (emit_len > 0) {
                                std::string tail(req->utf8_pending.data(), emit_len);
                                if (!req->json_mode && req->suppress_reasoning_tags &&
                                    IsReasoningTagSuppressionEnabled()) {
                                    const bool may_contain_tag =
                                        req->in_think_block || req->in_tool_call_block || req->in_tool_response_block ||
                                        !req->think_tag_pending.empty() || !req->tool_call_tag_pending.empty() ||
                                        !req->tool_response_tag_pending.empty() || tail.find('<') != std::string::npos;
                                    if (may_contain_tag) {
                                        SuppressTaggedBlock(&tail, &req->in_think_block, &req->think_tag_pending,
                                                            "<think>", "</think>");
                                        SuppressTaggedBlock(&tail, &req->in_tool_call_block,
                                                            &req->tool_call_tag_pending, "<tool_call>", "</tool_call>");
                                        SuppressTaggedBlock(&tail, &req->in_tool_response_block,
                                                            &req->tool_response_tag_pending, "<tool_response>",
                                                            "</tool_response>");
                                    }
                                }
                                if (!tail.empty() || req->token_result_callback) {
                                    EmitRequestResult(state, req, tail, best_token, false, false,
                                                      global_direct_callback);
                                    if (!tail.empty()) {
                                        NoteVisibleEmitProgress(req, decode_sample_time, best_token);
                                    }
                                }
                            }
                        }
                        FinalizeDecodeSilentFinishReason(req);
                        LogRequestDecodeSummary(req, current_model);
                        req->utf8_pending.clear();
                        // Decode graph cache is keyed by batch shape/threading, not request ID.
                        if (!terminal_error_emitted) {
                            state->metrics.completed_requests++;
                        }
                        // Push finished signal to callback queue (instead of direct
                        // callback)
                        if (!terminal_error_emitted && (req->callback || req->token_result_callback)) {
                            EmitRequestResult(state, req, "", -1, true, false, global_direct_callback);
                        }
                        {
                            std::lock_guard<std::mutex> lk(req->mu);
                            req->cv.notify_all();
                        }
                        // Free blocks and remove from scheduler
                        current_kv_cache->block_manager->Free(req->block_table);
                        req->block_table.clear();
                        // Cleanup using stored seq_id (O(1) instead of O(n))
                        if (req->seq_id >= 0) {
                            state->scheduler->RemoveRequest(req->seq_id, true);
                            seq_to_request.erase(req->seq_id);
                        }
                    }
                }
                token_offset += processed_count;
            }

            flush_scheduler_progress(batch_requests);

            // RAII guard (ctx_temp_guard) automatically frees non-cached contexts

            // 11. Sync active_requests: remove finished requests
            // Also signal queue_cv when requests finish (frees memory for scheduler)
            reap_finished_requests();
        }
        clear_decode_graph_cache();
        clear_prefill_graph_cache();
    } catch (const densecore::DenseCoreException& e) {
        std::cerr << "[DenseCore] Worker thread exception (" << e.CodeInt() << "): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] Worker thread exception: " << e.what() << std::endl;
    }
}
