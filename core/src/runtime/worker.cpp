#include <ggml-cpu.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
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
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
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
#include "ggml-alloc.h"
#include "ggml.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/attention/exec.h"
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

void cb_gemv_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_gemv_batched_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_matmul_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_matmul_int4_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_matmul_fp8_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_paged_attention_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_flash_attention_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_kv_update_and_gather_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_rope_precomputed_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata);

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

bool IsRequestLifecycleTraceEnabled() {
    static const bool enabled =
        densecore::llm::config::ReadBoolEnv("DENSECORE_DEBUG_REQUEST_LIFECYCLE", false);
    return enabled;
}

void LogRequestLifecyclePhase(const char* phase, const Request* req, const TransformerModel* model, int seq_id = -1,
                              int batch_seqs = 0, int batch_tokens = 0, int active_threads = 0,
                              const char* detail = nullptr) {
    if (!IsRequestLifecycleTraceEnabled()) {
        return;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    std::cerr << "[RequestLifecycle] phase=" << (phase ? phase : "unknown")
              << " request_id=" << (req ? req->id : -1)
              << " seq_id=" << seq_id
              << " variant=" << densecore::models::ModelVariantName(descriptor.variant)
              << " is_prefill=" << (req && req->is_prefill ? 1 : 0)
              << " prompt_tokens=" << (req ? req->tokens.size() : 0)
              << " n_past=" << (req ? req->n_past : 0)
              << " generated=" << (req ? req->generated_count : 0)
              << " max_tokens=" << (req ? req->max_tokens : 0)
              << " batch_seqs=" << batch_seqs
              << " batch_tokens=" << batch_tokens
              << " active_threads=" << active_threads;
    if (detail && detail[0] != '\0') {
        std::cerr << " detail=" << detail;
    }
    std::cerr << std::endl;
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

class ScopedBatchWorkContext {
public:
    ScopedBatchWorkContext(InferenceWorkContext* ctx, const BatchSpec* batch, InferenceExecutionPhase phase,
                           bool reset_context = true, ModelVariant model_variant = ModelVariant::UNKNOWN,
                           bool graph_build_no_alloc = false)
        : previous_ctx_(GetCurrentWorkContext()),
          previous_batch_(GetCurrentBatch()),
          previous_phase_(GetCurrentExecutionPhase()) {
        if (!ctx || !batch) {
            return;
        }
        active_ctx_ = ctx;
        if (reset_context) {
            ResetInferenceWorkContext(ctx);
        }
        previous_active_graph_build_no_alloc_ = IsCurrentGraphBuildNoAlloc();
        SetInferenceWorkContextModelVariant(ctx, model_variant);
        SetInferenceWorkContextGraphBuildNoAlloc(ctx, graph_build_no_alloc);
        SetCurrentWorkContext(ctx);
        SetCurrentExecutionPhase(phase);
        SetCurrentBatch(batch);
        active_ = true;
    }

    ScopedBatchWorkContext(const ScopedBatchWorkContext&) = delete;
    ScopedBatchWorkContext& operator=(const ScopedBatchWorkContext&) = delete;

    ~ScopedBatchWorkContext() {
        if (!active_) {
            return;
        }
        SetCurrentWorkContext(active_ctx_);
        SetInferenceWorkContextGraphBuildNoAlloc(active_ctx_, previous_active_graph_build_no_alloc_);
        ClearCurrentBatch();
        SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);

        if (previous_ctx_) {
            SetCurrentWorkContext(previous_ctx_);
            SetCurrentExecutionPhase(previous_phase_);
            if (previous_batch_) {
                SetCurrentBatch(previous_batch_);
            } else {
                ClearCurrentBatch();
            }
        } else {
            ClearCurrentBatch();
            SetCurrentExecutionPhase(InferenceExecutionPhase::Unknown);
            SetCurrentWorkContext(nullptr);
        }
    }

private:
    InferenceWorkContext* active_ctx_ = nullptr;
    InferenceWorkContext* previous_ctx_ = nullptr;
    const BatchSpec* previous_batch_ = nullptr;
    InferenceExecutionPhase previous_phase_ = InferenceExecutionPhase::Unknown;
    bool previous_active_graph_build_no_alloc_ = false;
    bool active_ = false;
};

class ScopedGgmlGraphArena {
public:
    ~ScopedGgmlGraphArena() { Reset(); }

    bool Ensure(size_t metadata_bytes, ggml_backend_t backend) {
        if (!backend || metadata_bytes == 0) {
            return false;
        }
        if (ctx_ && gallocr_ && backend_ == backend && metadata_bytes <= metadata_bytes_) {
            ggml_reset(ctx_);
            return true;
        }
        return Init(metadata_bytes, backend);
    }

    bool Init(size_t metadata_bytes, ggml_backend_t backend) {
        Reset();
        if (!backend || metadata_bytes == 0) {
            return false;
        }
#if defined(_WIN32)
        metadata_buffer_ = _aligned_malloc(metadata_bytes, 64);
#else
        if (posix_memalign(&metadata_buffer_, 64, metadata_bytes) != 0) {
            metadata_buffer_ = nullptr;
        }
#endif
        if (!metadata_buffer_) {
            return false;
        }
        ggml_init_params params{
            .mem_size = metadata_bytes,
            .mem_buffer = metadata_buffer_,
            .no_alloc = true,
        };
        ctx_ = ggml_init(params);
        if (!ctx_) {
            Reset();
            return false;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        if (!buft) {
            Reset();
            return false;
        }
        gallocr_ = ggml_gallocr_new(buft);
        if (!gallocr_) {
            Reset();
            return false;
        }
        metadata_bytes_ = metadata_bytes;
        backend_ = backend;
        return true;
    }

    bool AllocGraph(ggml_cgraph* graph) { return gallocr_ && graph && ggml_gallocr_alloc_graph(gallocr_, graph); }

    ggml_context* ctx() const { return ctx_; }

    void Reset() {
        if (gallocr_) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        if (metadata_buffer_) {
#if defined(_WIN32)
            _aligned_free(metadata_buffer_);
#else
            std::free(metadata_buffer_);
#endif
            metadata_buffer_ = nullptr;
        }
        metadata_bytes_ = 0;
        backend_ = nullptr;
    }

private:
    void* metadata_buffer_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    size_t metadata_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
};

bool IsMulGraphValidationEnabled() {
    return GetWorkerRuntimeConfig().validate_mul;
}

bool HasNamePrefix(const ggml_tensor* tensor, const char* prefix) {
    return tensor && prefix && tensor->name[0] != '\0' && std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
}

size_t AlignUpBytes(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

size_t ParseSizeEnvMb(const char* name, size_t default_mb, size_t min_mb, size_t max_mb) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_mb;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        return default_mb;
    }
    return static_cast<size_t>(std::clamp<unsigned long long>(value, min_mb, max_mb));
}

bool ParseBoolEnvDefault(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    return !(std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0 ||
             std::strcmp(raw, "off") == 0 || std::strcmp(raw, "OFF") == 0);
}

bool IsFlexibleGraphPoolSizingEnabled(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    return ParseBoolEnvDefault("DENSECORE_FLEXIBLE_GRAPH_POOL", true);
}

size_t ReadAvailableMemoryBytesForRuntimePools() {
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
            return static_cast<size_t>(kb) * 1024ULL;
        }
    }
    std::fclose(file);
#endif
    return 0;
}

void AccumulateDryRunTensorBytes(const ggml_tensor* tensor, std::unordered_set<const ggml_tensor*>& seen,
                                 size_t& data_bytes) {
    if (!tensor || !seen.insert(tensor).second) {
        return;
    }
    if (tensor->view_src) {
        AccumulateDryRunTensorBytes(tensor->view_src, seen, data_bytes);
    } else if (!tensor->data) {
        data_bytes += ggml_nbytes_pad(tensor);
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        AccumulateDryRunTensorBytes(tensor->src[i], seen, data_bytes);
    }
}

struct FlexibleGraphPoolSizing {
    bool ok = false;
    size_t required_bytes = 0;
    size_t reserved_bytes = 0;
    size_t available_bytes = 0;
    size_t reservation_payload_bytes = 0;
    size_t reservation_slack_bytes = 0;
    size_t dry_context_bytes = 0;
    size_t dry_metadata_bytes = 0;
    size_t graph_tensor_bytes = 0;
    size_t margin_bytes = 0;
    int graph_nodes = 0;
};

struct RuntimeGraphPoolReservation {
    size_t total_bytes = 0;
    size_t payload_bytes = 0;
    size_t slack_bytes = 0;
};

size_t ApplyFlexibleGraphPoolGrowthReserve(size_t required_bytes,
                                           const EngineState::GraphContextEstimate& graph_estimate) {
    if (required_bytes == 0 || graph_estimate.effective_query_len <= 1) {
        return required_bytes;
    }
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t measured_margin = std::max(required_bytes / 8, graph_estimate.long_context_safety_pad_bytes);
    const size_t reserve_bytes = required_bytes + std::max<size_t>(measured_margin, 128ULL * MB);
    return AlignUpBytes(reserve_bytes, 512ULL * MB);
}

RuntimeGraphPoolReservation ClampRuntimeGraphPoolReservation(size_t requested_bytes, size_t available_bytes) {
    if (requested_bytes == 0 || available_bytes == 0) {
        return {requested_bytes, requested_bytes, 0};
    }
    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t max_total_bytes =
        available_bytes > 256ULL * MB ? available_bytes - 256ULL * MB : (available_bytes * 3) / 4;
    const size_t runtime_reserve_bytes = std::clamp<size_t>(available_bytes / 12, 512ULL * MB, 8192ULL * MB);
    const size_t allocator_slack_bytes = std::max<size_t>(available_bytes / 64, 128ULL * MB);
    const size_t reserved_bytes = runtime_reserve_bytes + allocator_slack_bytes;
    const size_t usable_bytes =
        available_bytes > reserved_bytes ? available_bytes - reserved_bytes : available_bytes / 2;
    const size_t payload_bytes = std::min(requested_bytes, std::max<size_t>(usable_bytes, 512ULL * MB));
    const size_t context_object_slack_bytes = std::clamp<size_t>(payload_bytes / 6, 128ULL * MB, 2048ULL * MB);
    // Dry-run sizing can still undercount ggml object-pool pressure when the
    // live request shape is not byte-identical to the calibration shape. Scale
    // that reserve from the measured payload instead of baking in a VM-specific
    // RAM number; larger hosts and larger graphs naturally get more cushion.
    const size_t object_pool_variance_bytes = std::max(payload_bytes / 16, requested_bytes / 32);
    size_t total_bytes =
        AlignUpBytes(payload_bytes + context_object_slack_bytes + object_pool_variance_bytes, 64ULL * MB);
    size_t capped_payload_bytes = payload_bytes;
    size_t capped_slack_bytes = context_object_slack_bytes;
    if (max_total_bytes > 0 && total_bytes > max_total_bytes) {
        total_bytes = std::max(512ULL * MB, (max_total_bytes / (64ULL * MB)) * (64ULL * MB));
        capped_slack_bytes = std::min(context_object_slack_bytes, std::max<size_t>(128ULL * MB, total_bytes / 16));
        capped_payload_bytes = total_bytes > capped_slack_bytes ? total_bytes - capped_slack_bytes : total_bytes;
        capped_payload_bytes = std::min(capped_payload_bytes, requested_bytes);
        capped_slack_bytes = total_bytes > capped_payload_bytes ? total_bytes - capped_payload_bytes : 0;
    }
    return {total_bytes, capped_payload_bytes, capped_slack_bytes};
}

FlexibleGraphPoolSizing MeasureFlexibleGraphPoolSize(TransformerModel* model, PagedKVCache* cache,
                                                     const BatchSpec& batch, bool embedding_mode,
                                                     const EngineState::GraphContextEstimate& fallback_estimate) {
    FlexibleGraphPoolSizing result{};
    if (!IsFlexibleGraphPoolSizingEnabled(model)) {
        return result;
    }

    constexpr size_t MB = 1024ULL * 1024ULL;
    const size_t fallback_mb = std::max<size_t>(512, fallback_estimate.total_bytes / MB);
    const size_t available_bytes = ReadAvailableMemoryBytesForRuntimePools();
    const size_t available_mb = available_bytes / MB;
    const size_t auto_dry_cap_mb =
        available_mb > 0 ? std::max<size_t>(512, (available_mb * 3) / 4) : std::max<size_t>(fallback_mb, 1024);
    const size_t dry_run_headroom_mb = fallback_estimate.effective_query_len > 1
                                           ? std::max<size_t>(512, fallback_mb / 8)
                                           : std::max<size_t>(128, fallback_mb / 16);
    const size_t default_dry_mb =
        std::min<size_t>(std::max<size_t>(fallback_mb + dry_run_headroom_mb, 1024), auto_dry_cap_mb);
    const size_t dry_mb = ParseSizeEnvMb("DENSECORE_GRAPH_DRY_RUN_CTX_MB", default_dry_mb, 512, auto_dry_cap_mb);
    const size_t dry_context_bytes = dry_mb * MB;

    void* dry_buffer = nullptr;
#if defined(_WIN32)
    dry_buffer = _aligned_malloc(dry_context_bytes, 64);
#else
    if (posix_memalign(&dry_buffer, 64, dry_context_bytes) != 0) {
        dry_buffer = nullptr;
    }
#endif
    if (!dry_buffer) {
        return result;
    }

    ggml_context* dry_ctx = nullptr;
    try {
        ggml_init_params params{
            .mem_size = dry_context_bytes,
            .mem_buffer = dry_buffer,
            .no_alloc = true,
        };
        dry_ctx = ggml_init(params);
        if (!dry_ctx) {
            throw densecore::OutOfMemoryException("flexible graph pool dry-run ggml_init failed");
        }
        ggml_cgraph* dry_graph = ggml_new_graph_custom(dry_ctx, 32768, false);
        ggml_tensor* dry_embd = nullptr;
        ggml_tensor* dry_pos = nullptr;
        auto dry_work_ctx = std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)>(
            CreateInferenceWorkContext(), DestroyInferenceWorkContext);
        const InferenceExecutionPhase dry_phase = batch.tokens.size() > static_cast<size_t>(batch.num_seqs)
                                                      ? InferenceExecutionPhase::Prefill
                                                      : InferenceExecutionPhase::Decode;
        ScopedBatchWorkContext dry_scope(dry_work_ctx.get(), &batch, dry_phase,
                                         /*reset_context=*/true, ModelVariant::UNKNOWN,
                                         /*graph_build_no_alloc=*/true);
        ggml_tensor* dry_output =
            BuildTransformerGraph(model, cache, dry_ctx, batch, embedding_mode, dry_graph, &dry_embd, &dry_pos);
        if (!dry_graph || !dry_output || !dry_embd || !dry_pos) {
            throw densecore::GraphBuildException("flexible graph pool dry-run graph build returned incomplete graph");
        }

        std::unordered_set<const ggml_tensor*> seen;
        size_t data_bytes = 0;
        const int n_nodes = ggml_graph_n_nodes(dry_graph);
        for (int i = 0; i < n_nodes; ++i) {
            AccumulateDryRunTensorBytes(ggml_graph_node(dry_graph, i), seen, data_bytes);
        }
        AccumulateDryRunTensorBytes(dry_embd, seen, data_bytes);
        AccumulateDryRunTensorBytes(dry_pos, seen, data_bytes);
        AccumulateDryRunTensorBytes(dry_output, seen, data_bytes);

        const size_t metadata_bytes = ggml_used_mem(dry_ctx);
        const size_t measured_bytes = metadata_bytes + data_bytes;
        const size_t percent_margin = measured_bytes / 8;
        const size_t min_margin = model->arch_flags.is_gemma4 ? 512ULL * MB : 128ULL * MB;
        const size_t margin_bytes = std::max(percent_margin, min_margin);
        const RuntimeGraphPoolReservation required_reservation =
            ClampRuntimeGraphPoolReservation(AlignUpBytes(measured_bytes + margin_bytes, 64ULL * MB), available_bytes);
        result.ok = true;
        result.required_bytes = required_reservation.total_bytes;
        const RuntimeGraphPoolReservation growth_reservation = ClampRuntimeGraphPoolReservation(
            ApplyFlexibleGraphPoolGrowthReserve(result.required_bytes, fallback_estimate), available_bytes);
        result.reserved_bytes = growth_reservation.total_bytes;
        result.available_bytes = available_bytes;
        result.reservation_payload_bytes = growth_reservation.payload_bytes;
        result.reservation_slack_bytes = growth_reservation.slack_bytes;
        result.dry_context_bytes = dry_context_bytes;
        result.dry_metadata_bytes = metadata_bytes;
        result.graph_tensor_bytes = data_bytes;
        result.margin_bytes = margin_bytes;
        result.graph_nodes = n_nodes;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] FlexibleGraphPool dry-run skipped: " << e.what() << std::endl;
    }

    if (dry_ctx) {
        ggml_free(dry_ctx);
    }
#if defined(_WIN32)
    _aligned_free(dry_buffer);
#else
    std::free(dry_buffer);
#endif
    return result;
}

void AccumulateQwen36SSMProjectionNodeTimes(InferenceWorkContext* work_ctx, ggml_cgraph* graph) {
    if (!work_ctx || !graph) {
        return;
    }
    uint64_t qkv_ns = 0;
    uint64_t gate_ns = 0;
    uint64_t out_ns = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (HasNamePrefix(node, "qwen36_ssm_qkv_proj") || HasNamePrefix(node, "qwen35_ssm_qkv_proj") ||
            HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj")) {
            qkv_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen36_ssm_gate_proj") || HasNamePrefix(node, "qwen35_ssm_gate_proj")) {
            gate_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen36_ssm_out_proj") || HasNamePrefix(node, "qwen35_ssm_out_proj")) {
            out_ns += elapsed_ns;
        }
    }
    if (qkv_ns || gate_ns || out_ns) {
        AddQwen36SSMProjectionWallProfile(work_ctx, qkv_ns, gate_ns, out_ns);
    }
}

struct Qwen36PrefillBreakdown {
    uint64_t ssm_projection_ns = 0;
    uint64_t ssm_delta_state_ns = 0;
    uint64_t attention_ns = 0;
    uint64_t mlp_or_moe_ns = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_ops;
};

struct HybridSSMGraphTimingBreakdown {
    uint64_t qkv_ns = 0;
    uint64_t gate_ns = 0;
    uint64_t out_ns = 0;
    uint64_t conv1d_ns = 0;
    uint64_t delta_ns = 0;
    uint64_t alpha_beta_qk_ns = 0;
    uint64_t total_ns = 0;
};

struct NativeMoEGraphTimingBreakdown {
    uint64_t route_ns = 0;
    uint64_t w1w3_ns = 0;
    uint64_t activation_ns = 0;
    uint64_t w2_ns = 0;
    uint64_t w1w3_fast_ns = 0;
    uint64_t w2_fast_ns = 0;
    uint64_t reduce_ns = 0;
    uint64_t total_ns = 0;
    uint64_t route_count = 0;
    uint64_t w1w3_count = 0;
    uint64_t activation_count = 0;
    uint64_t w2_count = 0;
    uint64_t w1w3_fast_count = 0;
    uint64_t w2_fast_count = 0;
    uint64_t reduce_count = 0;
    uint64_t native_node_count = 0;
    std::string node_hist;
    std::vector<MatmulShapeCensusEntry> top_slow_nodes;
};

struct DecodeGraphNodeTimingBreakdown {
    uint64_t measured_ns = 0;
    uint64_t custom_ns = 0;
    uint64_t mul_mat_ns = 0;
    uint64_t mul_mat_id_ns = 0;
    uint64_t norm_ns = 0;
    uint64_t view_copy_ns = 0;
    uint64_t elementwise_ns = 0;
    uint64_t attention_ns = 0;
    uint64_t other_ns = 0;
    uint64_t custom_count = 0;
    uint64_t mul_mat_count = 0;
    uint64_t mul_mat_id_count = 0;
    uint64_t norm_count = 0;
    uint64_t view_copy_count = 0;
    uint64_t elementwise_count = 0;
    uint64_t attention_count = 0;
    uint64_t other_count = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_nodes;
};

struct Gemma4PrefillAttentionTimingBreakdown {
    uint64_t attention_ns = 0;
    uint64_t portable_flash_ns = 0;
    uint64_t hal_ns = 0;
    uint64_t moe_or_mlp_ns = 0;
    uint64_t mul_mat_id_ns = 0;
    uint64_t mul_mat_ns = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_ops;
};

static const char* DecodeGraphNodeBucketName(const ggml_tensor* node) {
    if (!node) {
        return "other";
    }
    const char* name = node->name[0] ? node->name : "";
    if (std::strstr(name, "attn") || std::strstr(name, "paged") || std::strstr(name, "flash") ||
        std::strstr(name, "kv_") || std::strstr(name, "rope")) {
        return "attention";
    }
    switch (node->op) {
    case GGML_OP_CUSTOM: return "custom";
    case GGML_OP_MUL_MAT: return "mul_mat";
    case GGML_OP_MUL_MAT_ID: return "mul_mat_id";
    case GGML_OP_RMS_NORM:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM: return "norm";
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY: return "view_copy";
    case GGML_OP_ADD:
    case GGML_OP_ADD1:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_SCALE:
    case GGML_OP_UNARY:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_REPEAT:
    case GGML_OP_GET_ROWS: return "elementwise";
    default: return "other";
    }
}

const char* Gemma4WeightClass(const char* name);
const char* Gemma4CensusWeightType(ggml_type type);
bool Gemma4NodeHasCopyLikeInput(const ggml_tensor* node);
std::string Gemma4MatmulShapeBucket(const ggml_tensor* node, const ggml_tensor* weight);
std::string Gemma4CustomNodeClass(const ggml_tensor* node);

DecodeGraphNodeTimingBreakdown SummarizeDecodeGraphNodeTimes(const ggml_cgraph* graph,
                                                             bool collect_top_slow_nodes) {
    DecodeGraphNodeTimingBreakdown out;
    if (!graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const char* bucket = DecodeGraphNodeBucketName(node);
        if (std::strcmp(bucket, "custom") == 0) {
            out.custom_count += 1;
        } else if (std::strcmp(bucket, "mul_mat") == 0) {
            out.mul_mat_count += 1;
        } else if (std::strcmp(bucket, "mul_mat_id") == 0) {
            out.mul_mat_id_count += 1;
        } else if (std::strcmp(bucket, "norm") == 0) {
            out.norm_count += 1;
        } else if (std::strcmp(bucket, "view_copy") == 0) {
            out.view_copy_count += 1;
        } else if (std::strcmp(bucket, "elementwise") == 0) {
            out.elementwise_count += 1;
        } else if (std::strcmp(bucket, "attention") == 0) {
            out.attention_count += 1;
        } else {
            out.other_count += 1;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (std::strcmp(bucket, "custom") == 0) {
            out.custom_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "mul_mat") == 0) {
            out.mul_mat_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "mul_mat_id") == 0) {
            out.mul_mat_id_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "norm") == 0) {
            out.norm_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "view_copy") == 0) {
            out.view_copy_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "elementwise") == 0) {
            out.elementwise_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "attention") == 0) {
            out.attention_ns += elapsed_ns;
        } else {
            out.other_ns += elapsed_ns;
        }
        out.measured_ns += elapsed_ns;

        // The op-bucket totals above are cheap and always collected for decode so
        // every profiling run shows where graph-execute time lands. The per-node
        // census below allocates a string per node, so it stays behind the debug
        // flag to keep the hot decode path free of that overhead.
        if (!collect_top_slow_nodes) {
            continue;
        }
        MatmulShapeCensusEntry entry;
        entry.phase = "decode";
        entry.op_type = ggml_op_name(node->op);
        const std::string custom_class = Gemma4CustomNodeClass(node);
        entry.dispatch_path = custom_class.empty() ? bucket : custom_class;
        entry.weight_type = ggml_op_name(node->op);
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = node->name[0] ? node->name : "unnamed";
        entry.right_name = "elapsed_us";
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_nodes.push_back(std::move(entry));
    }
    std::sort(out.top_slow_nodes.begin(), out.top_slow_nodes.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_nodes.size() > kMatmulTopShapeCount) {
        out.top_slow_nodes.resize(kMatmulTopShapeCount);
    }
    return out;
}

Gemma4PrefillAttentionTimingBreakdown SummarizeGemma4PrefillAttentionNodeTimes(const TransformerModel* model,
                                                                               const ggml_cgraph* graph) {
    Gemma4PrefillAttentionTimingBreakdown out;
    if (!model || !model->arch_flags.is_gemma4 || !graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (custom_class == "flash_attention_hal") {
            out.attention_ns += elapsed_ns;
            out.portable_flash_ns += elapsed_ns;
            out.hal_ns += elapsed_ns;
        }
        if (node->op == GGML_OP_MUL_MAT_ID) {
            out.mul_mat_id_ns += elapsed_ns;
        } else if (node->op == GGML_OP_MUL_MAT) {
            out.mul_mat_ns += elapsed_ns;
        }
        const char* node_name = node->name[0] ? node->name : "";
        if (std::strstr(node_name, "moe") || std::strstr(node_name, "ffn") || std::strstr(node_name, "shared_ffn")) {
            out.moe_or_mlp_ns += elapsed_ns;
        }
        const bool include_node =
            node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT || custom_class == "flash_attention_hal";
        if (include_node) {
            const ggml_tensor* weight = node->src[0];
            const ggml_tensor* input = node->src[1];
            const char* weight_name = weight && weight->name[0] ? weight->name : "<src0>";
            const char* input_name = input && input->name[0] ? input->name : "<src1>";
            MatmulShapeCensusEntry entry;
            entry.phase = "prefill";
            if (node->op == GGML_OP_MUL_MAT_ID) {
                entry.op_type = "MUL_MAT_ID";
                entry.dispatch_path = "ggml_mul_mat_id";
            } else if (node->op == GGML_OP_MUL_MAT) {
                entry.op_type = "MUL_MAT";
                entry.dispatch_path = "ggml_mul_mat";
            } else {
                entry.op_type = "CUSTOM";
                entry.dispatch_path = custom_class.empty() ? "custom" : custom_class;
            }
            entry.weight_type = weight ? Gemma4CensusWeightType(weight->type) : "other";
            entry.weight_class = weight ? Gemma4WeightClass(weight_name) : "gemma4_other_dense";
            entry.shape_bucket = Gemma4MatmulShapeBucket(node, weight);
            entry.left_name = node_name[0] ? node_name : weight_name;
            entry.right_name = std::string("src0=") + weight_name + ",src1=" + input_name;
            entry.wall_ns = elapsed_ns;
            entry.ops = static_cast<uint64_t>(elapsed_us);
            entry.calls = 1;
            entry.active_threads = 0;
            entry.contiguous_or_copy_input = Gemma4NodeHasCopyLikeInput(node) ? 1 : 0;
            out.top_slow_ops.push_back(std::move(entry));
        }
    }
    std::sort(out.top_slow_ops.begin(), out.top_slow_ops.end(), [](const auto& a, const auto& b) {
        if (a.wall_ns != b.wall_ns) {
            return a.wall_ns > b.wall_ns;
        }
        return a.left_name < b.left_name;
    });
    constexpr std::size_t kGemma4PrefillTopSlowCount = 200;
    if (out.top_slow_ops.size() > kGemma4PrefillTopSlowCount) {
        out.top_slow_ops.resize(kGemma4PrefillTopSlowCount);
    }
    return out;
}

HybridSSMGraphTimingBreakdown SummarizeHybridSSMGraphNodeTimes(const TransformerModel* model,
                                                               const ggml_cgraph* graph) {
    HybridSSMGraphTimingBreakdown out;
    if (!model || !model->arch_flags.is_hybrid_ssm || !graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen35_ssm_qkv_proj") || HasNamePrefix(node, "qwen36_ssm_qkv_proj")) {
            out.qkv_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_gate_proj") || HasNamePrefix(node, "qwen36_ssm_gate_proj")) {
            out.gate_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_out_proj") || HasNamePrefix(node, "qwen36_ssm_out_proj")) {
            out.out_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_conv1d") || HasNamePrefix(node, "qwen36_ssm_conv1d")) {
            out.conv1d_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_delta") || HasNamePrefix(node, "qwen36_ssm_delta")) {
            out.delta_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_alpha_beta_qk") || HasNamePrefix(node, "qwen36_ssm_alpha_beta_qk")) {
            out.alpha_beta_qk_ns += elapsed_ns;
        }
    }
    out.total_ns = out.qkv_ns + out.gate_ns + out.out_ns + out.conv1d_ns + out.delta_ns + out.alpha_beta_qk_ns;
    return out;
}

bool IsQwenHybridSSMModel(const TransformerModel* model);

Qwen36PrefillBreakdown SummarizeQwen36PrefillNodeTimes(const TransformerModel* model, const ggml_cgraph* graph) {
    Qwen36PrefillBreakdown out;
    if (!IsQwenHybridSSMModel(model) || !graph) {
        return out;
    }
    if (model->variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0) {
        return out;
    }
    auto safe_op_name = [](enum ggml_op op) -> const char* {
        const int value = static_cast<int>(op);
        return value >= 0 && value < static_cast<int>(GGML_OP_COUNT) ? ggml_op_name(op) : "<invalid-op>";
    };
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        const char* name = node->name[0] ? node->name : "unnamed";
        const bool is_ssm_proj = HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_qkv_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_gate_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_out_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_qkv_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_gate_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_out_proj");
        if (is_ssm_proj) {
            out.ssm_projection_ns += elapsed_ns;
        } else if (std::strstr(name, "delta") || std::strstr(name, "ssm_scan") || std::strstr(name, "conv1d")) {
            out.ssm_delta_state_ns += elapsed_ns;
        } else if (std::strstr(name, "attn") || std::strstr(name, "flash") || std::strstr(name, "kv_")) {
            out.attention_ns += elapsed_ns;
        } else if (std::strstr(name, "moe") || std::strstr(name, "ffn") || std::strstr(name, "mlp")) {
            out.mlp_or_moe_ns += elapsed_ns;
        }
        MatmulShapeCensusEntry entry;
        entry.phase = "prefill";
        const char* op_name = safe_op_name(node->op);
        entry.op_type = op_name;
        entry.dispatch_path =
            is_ssm_proj ? "ssm_projection" : (std::strstr(name, "moe") ? "mlp_or_moe" : op_name);
        const ggml_tensor* src0 = node->src[0];
        const ggml_tensor* src1 = node->src[1];
        entry.weight_type = src0 ? ggml_type_name(src0->type) : "node";
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = name;
        const char* src0_name = src0 && src0->name[0] ? src0->name : "<src0>";
        const char* src1_name = src1 && src1->name[0] ? src1->name : "<src1>";
        entry.right_name = std::string("src0=") + src0_name + ",src1=" + src1_name;
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_ops.push_back(std::move(entry));
    }
    std::sort(out.top_slow_ops.begin(), out.top_slow_ops.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_ops.size() > kMatmulTopShapeCount) {
        out.top_slow_ops.resize(kMatmulTopShapeCount);
    }
    return out;
}

NativeMoEGraphTimingBreakdown SummarizeNativeQwenMoEGraphNodeTimes(const TransformerModel* model,
                                                                   const ggml_cgraph* graph) {
    NativeMoEGraphTimingBreakdown out;
    if (!model || (model->variant != ModelVariant::QWEN35 && model->variant != ModelVariant::QWEN36) || !graph) {
        return out;
    }
    struct Bucket {
        const char* name = "other";
        uint64_t ns = 0;
        uint64_t count = 0;
    };
    std::array<Bucket, 6> buckets = {
        {{"route", 0, 0}, {"w1w3", 0, 0}, {"activation", 0, 0}, {"w2", 0, 0}, {"reduce", 0, 0}, {"other", 0, 0}}};
    enum BucketIndex { Route = 0, W1W3 = 1, Activation = 2, W2 = 3, Reduce = 4, Other = 5 };
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    bool has_native_qwen_moe = false;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (node && node->name[0] && std::strstr(node->name, "qwen35_native_moe")) {
            has_native_qwen_moe = true;
            break;
        }
    }
    if (!has_native_qwen_moe) {
        return out;
    }
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const char* name = node->name[0] ? node->name : "unnamed";
        const bool native_moe_node = std::strstr(name, "qwen35_native_moe") || std::strstr(name, ".moe_gate_logits");
        if (!native_moe_node) {
            continue;
        }
        ++out.native_node_count;
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        BucketIndex bucket = Other;
        if (std::strstr(name, ".moe_gate_logits") || std::strstr(name, "_topk") || std::strstr(name, "_norm_weights") ||
            std::strstr(name, "_probs") || std::strstr(name, "_weights") || std::strstr(name, "_scaled_weights")) {
            bucket = Route;
        } else if (std::strstr(name, "_gate_up") || std::strstr(name, "_gate") || std::strstr(name, "_up")) {
            bucket = W1W3;
        } else if (std::strstr(name, "_swiglu")) {
            bucket = Activation;
        } else if (std::strstr(name, "_down")) {
            bucket = W2;
        } else if (std::strstr(name, "_expert_sum") || std::strstr(name, "_moe_out")) {
            bucket = Reduce;
        }
        const bool fast_w1w3_node = bucket == W1W3 &&
                                     (std::strstr(name, "_gateup_raw_q4k_swiglu") ||
                                      std::strstr(name, "_gateup_raw_qxk_swiglu"));
        const bool fast_w2_node = bucket == W2 && std::strstr(name, "_down_q5k_fast");
        buckets[static_cast<std::size_t>(bucket)].ns += elapsed_ns;
        buckets[static_cast<std::size_t>(bucket)].count += 1;
        if (fast_w1w3_node) {
            out.w1w3_fast_ns += elapsed_ns;
            out.w1w3_fast_count += 1;
        }
        if (fast_w2_node) {
            out.w2_fast_ns += elapsed_ns;
            out.w2_fast_count += 1;
        }
        out.total_ns += elapsed_ns;

        MatmulShapeCensusEntry entry;
        entry.phase = model->variant == ModelVariant::QWEN36 ? "qwen36" : "qwen35";
        entry.op_type = ggml_op_name(node->op);
        entry.dispatch_path = buckets[static_cast<std::size_t>(bucket)].name;
        entry.weight_type = ggml_op_name(node->op);
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = name;
        entry.right_name = "elapsed_us";
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_nodes.push_back(std::move(entry));
    }
    out.route_ns = buckets[Route].ns;
    out.w1w3_ns = buckets[W1W3].ns;
    out.activation_ns = buckets[Activation].ns;
    out.w2_ns = buckets[W2].ns;
    out.reduce_ns = buckets[Reduce].ns;
    out.route_count = buckets[Route].count;
    out.w1w3_count = buckets[W1W3].count;
    out.activation_count = buckets[Activation].count;
    out.w2_count = buckets[W2].count;
    out.reduce_count = buckets[Reduce].count;
    std::ostringstream hist;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        if (i != 0) hist << ",";
        hist << buckets[i].name << ":count=" << buckets[i].count
             << ":ms=" << (static_cast<double>(buckets[i].ns) / 1.0e6);
    }
    out.node_hist = hist.str();
    std::sort(out.top_slow_nodes.begin(), out.top_slow_nodes.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_nodes.size() > kMatmulTopShapeCount) {
        out.top_slow_nodes.resize(kMatmulTopShapeCount);
    }
    return out;
}

void SynthesizeDecodeGraphNodeTimingFromProfiles(DecodeGraphNodeTimingBreakdown* timing, uint64_t graph_execute_ns,
                                                 const NativeMoEGraphTimingBreakdown& native_moe,
                                                 const HybridSSMGraphTimingBreakdown& hybrid_ssm,
                                                 const Qwen36ProfileSnapshot& profile) {
    if (!timing || graph_execute_ns == 0 || timing->measured_ns != 0) {
        return;
    }
    uint64_t ssm_ns = hybrid_ssm.total_ns;
    if (ssm_ns == 0) {
        ssm_ns = profile.ssm_qkv_wall_ns + profile.ssm_out_wall_ns + profile.ssm_delta_wall_ns;
    }
    uint64_t custom_ns = native_moe.total_ns + ssm_ns;
    const uint64_t attention_ns = profile.attention_ns;
    if (custom_ns > graph_execute_ns) {
        custom_ns = graph_execute_ns;
    }
    const uint64_t after_custom = graph_execute_ns - custom_ns;
    const uint64_t bounded_attention_ns = std::min(attention_ns, after_custom);
    const uint64_t other_ns = graph_execute_ns - custom_ns - bounded_attention_ns;

    timing->custom_ns = custom_ns;
    timing->attention_ns = bounded_attention_ns;
    timing->other_ns = other_ns;
    timing->measured_ns = graph_execute_ns;

    if (timing->custom_count == 0) {
        timing->custom_count = native_moe.native_node_count + static_cast<uint64_t>(profile.ssm_conv1d_calls) +
                               static_cast<uint64_t>(profile.ssm_delta_calls);
    }
    if (timing->attention_count == 0 && bounded_attention_ns > 0) {
        timing->attention_count = 1;
    }
    if (timing->other_count == 0 && other_ns > 0) {
        timing->other_count = 1;
    }
}

bool IsGemma4NodeTimingDumpEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_GEMMA4_NODE_TIMES", false);
    return enabled;
}

bool IsLLMNodeTimingDumpEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_LLM_NODE_TIMES", false);
    return enabled;
}

int Gemma4NodeTimingDumpLimit() {
    return densecore::env::ParsePositiveEnvInt("DENSECORE_DEBUG_GEMMA4_NODE_TIMES_LIMIT", 24);
}

int LLMNodeTimingDumpLimit() {
    return densecore::env::ParsePositiveEnvInt("DENSECORE_DEBUG_LLM_NODE_TIMES_LIMIT", 24);
}

const char* Gemma4WeightClass(const char* name) {
    if (!name || name[0] == '\0') {
        return "unnamed_weight";
    }
    if (std::strcmp(name, "output.weight") == 0 || std::strcmp(name, "output") == 0 || std::strstr(name, "lm_head") ||
        std::strstr(name, "output.weight")) {
        return "gemma4_other_dense";
    }
    if (std::strstr(name, "attn_qkv")) {
        return "gemma4_qkv";
    }
    if (std::strstr(name, "attn_q.weight") || std::strstr(name, "attn_q")) {
        return "gemma4_qkv";
    }
    if (std::strstr(name, "attn_k.weight") || std::strstr(name, "attn_k")) {
        return "gemma4_qkv";
    }
    if (std::strstr(name, "attn_v.weight") || std::strstr(name, "attn_v")) {
        return "gemma4_qkv";
    }
    if (std::strstr(name, "attn_output.weight") || std::strstr(name, "attn_out") || std::strstr(name, "attn_o")) {
        return "gemma4_attn_o";
    }
    if (std::strstr(name, "ffn_gate_inp") || std::strstr(name, "moe_gate")) {
        return "gemma4_other_dense";
    }
    if (std::strstr(name, "ffn_gate_exps") || std::strstr(name, "ffn_gate_up_exps")) {
        return "gemma4_moe_gate_up";
    }
    if (std::strstr(name, "ffn_down_exps")) {
        return "gemma4_moe_down";
    }
    if (std::strstr(name, "shared_ffn_gate_up") || std::strstr(name, "ffn_gate_up.weight")) {
        return "gemma4_shared_gate";
    }
    if (std::strstr(name, "ffn_gate.weight")) {
        return "gemma4_shared_gate";
    }
    if (std::strstr(name, "ffn_up.weight")) {
        return "gemma4_shared_up";
    }
    if (std::strstr(name, "ffn_down.weight")) {
        return "gemma4_shared_down";
    }
    return "gemma4_other_dense";
}

const char* Gemma4CensusWeightType(ggml_type type) {
    switch (type) {
    case GGML_TYPE_Q4_K: return "q4_k";
    case GGML_TYPE_Q5_K: return "q5_k";
    case GGML_TYPE_Q6_K: return "q6_k";
    case GGML_TYPE_Q8_0: return "q8_0";
    case GGML_TYPE_F16: return "f16";
    case GGML_TYPE_F32: return "f32";
    default: return "other";
    }
}

bool Gemma4NodeHasCopyLikeInput(const ggml_tensor* node) {
    if (!node || !node->src[1]) {
        return false;
    }
    const ggml_tensor* input = node->src[1];
    switch (input->op) {
    case GGML_OP_CONT:
    case GGML_OP_CPY:
    case GGML_OP_DUP:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE: return true;
    default: return false;
    }
}

std::string Gemma4MatmulShapeBucket(const ggml_tensor* node, const ggml_tensor* weight) {
    const int64_t out_rows = node ? node->ne[0] : 0;
    const int64_t tokens = node ? std::max<int64_t>(node->ne[1], node->ne[2]) : 0;
    const int64_t k = weight ? weight->ne[0] : 0;
    const int64_t experts = weight && weight->ne[2] > 1 ? weight->ne[2] : 0;
    const int64_t active_experts = (node && node->op == GGML_OP_MUL_MAT_ID && node->src[2]) ? node->src[2]->ne[0] : 0;
    std::ostringstream shape;
    shape << "M=" << tokens << ",N=" << out_rows << ",K=" << k << ",tokens=" << tokens << ",experts=" << experts
          << ",active_experts=" << active_experts << ",copy_in=" << (Gemma4NodeHasCopyLikeInput(node) ? 1 : 0);
    return shape.str();
}

std::string Gemma4CustomNodeClass(const ggml_tensor* node) {
    if (!node || node->op != GGML_OP_CUSTOM) {
        return {};
    }
    struct CustomOpParamsRawView {
        std::uintptr_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(CustomOpParamsRawView) <= GGML_MAX_OP_PARAMS, "custom op params view too large");
    CustomOpParamsRawView params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    auto same_fun = [&](auto* fun) { return params.fun == reinterpret_cast<std::uintptr_t>(fun); };
    if (same_fun(cb_gemv_custom) || same_fun(cb_gemv_batched_custom)) {
        const ggml_tensor* weight = node->src[1];
        const char* weight_name = weight && weight->name[0] ? weight->name : "<unnamed_weight>";
        std::string label = same_fun(cb_gemv_custom) ? "gemv/" : "gemv_batched/";
        label += Gemma4WeightClass(weight_name);
        label += "/";
        label += weight_name;
        return label;
    }
    if (same_fun(cb_paged_attention_decode)) {
        return "paged_attention_decode";
    }
    if (same_fun(cb_flash_attention_hal_custom)) {
        return "flash_attention_hal";
    }
    if (same_fun(cb_kv_update_and_gather_custom)) {
        return "kv_update_and_gather";
    }
    if (same_fun(cb_rope_precomputed_custom)) {
        return "rope_precomputed";
    }
    if (same_fun(cb_matmul_hal_custom)) {
        return "matmul_hal_custom";
    }
    if (same_fun(cb_matmul_int4_custom)) {
        return "matmul_int4_custom";
    }
    if (same_fun(cb_matmul_fp8_custom)) {
        return "matmul_fp8_custom";
    }
    return "custom_other";
}

void DebugDumpGemma4NodeTimes(const TransformerModel* model, const ggml_cgraph* graph, const char* stage) {
    if (!model || !model->arch_flags.is_gemma4 || !graph || !IsGemma4NodeTimingDumpEnabled()) {
        return;
    }
    struct Entry {
        std::string key;
        uint64_t total_us = 0;
        int count = 0;
    };
    std::unordered_map<std::string, Entry> by_key;
    std::unordered_map<std::string, Entry> by_op;
    uint64_t measured_us = 0;
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        char fallback_name[256];
        const char* name = node->name[0] ? node->name : nullptr;
        if (!name || std::strncmp(name, "node_", 5) == 0) {
            const char* src0_name = (node->src[0] && node->src[0]->name[0]) ? node->src[0]->name : "<src0>";
            const char* src1_name = (node->src[1] && node->src[1]->name[0]) ? node->src[1]->name : "<src1>";
            std::snprintf(fallback_name, sizeof(fallback_name), "node_%d/src0=%s/src1=%s", i, src0_name, src1_name);
            name = fallback_name;
        }
        const char* op = ggml_op_name(node->op);
        std::string op_key = op ? op : "<op>";
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (!custom_class.empty()) {
            op_key += "/";
            op_key += custom_class;
        }
        std::string key = op_key + ":" + name;
        Entry& entry = by_key[key];
        entry.key = std::move(key);
        entry.total_us += static_cast<uint64_t>(elapsed_us);
        entry.count += 1;
        Entry& op_entry = by_op[op_key];
        op_entry.key = std::move(op_key);
        op_entry.total_us += static_cast<uint64_t>(elapsed_us);
        op_entry.count += 1;
        measured_us += static_cast<uint64_t>(elapsed_us);
    }
    auto make_sorted_entries = [](std::unordered_map<std::string, Entry>& values) {
        std::vector<Entry> entries;
        entries.reserve(values.size());
        for (auto& kv : values) {
            entries.push_back(std::move(kv.second));
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.total_us != b.total_us) {
                return a.total_us > b.total_us;
            }
            return a.key < b.key;
        });
        return entries;
    };
    std::vector<Entry> entries = make_sorted_entries(by_key);
    std::vector<Entry> op_entries = make_sorted_entries(by_op);
    const int op_limit = std::min<int>(8, op_entries.size());
    std::cerr << "[Gemma4NodeTimesByOp] stage=" << (stage ? stage : "<unknown>") << " showing=" << op_limit;
    for (int i = 0; i < op_limit; ++i) {
        const Entry& entry = op_entries[static_cast<size_t>(i)];
        std::cerr << " op" << (i + 1) << "=" << entry.key << ":" << (static_cast<double>(entry.total_us) / 1000.0)
                  << "ms/" << entry.count;
    }
    std::cerr << std::endl;
    const int limit = std::min<int>(Gemma4NodeTimingDumpLimit(), entries.size());
    std::cerr << "[Gemma4NodeTimes] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " measured_ms=" << (static_cast<double>(measured_us) / 1000.0) << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        const Entry& entry = entries[static_cast<size_t>(i)];
        std::cerr << "  rank=" << (i + 1) << " total_ms=" << (static_cast<double>(entry.total_us) / 1000.0)
                  << " count=" << entry.count << " key=" << entry.key << std::endl;
    }
}

void DebugDumpLLMNodeTimes(const TransformerModel* model, const ggml_cgraph* graph, const char* stage) {
    if (!model || !graph || !IsLLMNodeTimingDumpEnabled()) {
        return;
    }
    struct Entry {
        std::string key;
        uint64_t total_us = 0;
        int count = 0;
    };
    std::unordered_map<std::string, Entry> by_key;
    std::unordered_map<std::string, Entry> by_op;
    uint64_t measured_us = 0;
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        char fallback_name[256];
        const char* name = node->name[0] ? node->name : nullptr;
        if (!name || std::strncmp(name, "node_", 5) == 0) {
            const char* src0_name = (node->src[0] && node->src[0]->name[0]) ? node->src[0]->name : "<src0>";
            const char* src1_name = (node->src[1] && node->src[1]->name[0]) ? node->src[1]->name : "<src1>";
            std::snprintf(fallback_name, sizeof(fallback_name), "node_%d/src0=%s/src1=%s", i, src0_name, src1_name);
            name = fallback_name;
        }
        const char* op = ggml_op_name(node->op);
        std::string op_key = op ? op : "<op>";
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (!custom_class.empty()) {
            op_key += "/";
            op_key += custom_class;
        }
        std::string key = op_key + ":" + name;
        Entry& entry = by_key[key];
        entry.key = std::move(key);
        entry.total_us += static_cast<uint64_t>(elapsed_us);
        entry.count += 1;
        Entry& op_entry = by_op[op_key];
        op_entry.key = std::move(op_key);
        op_entry.total_us += static_cast<uint64_t>(elapsed_us);
        op_entry.count += 1;
        measured_us += static_cast<uint64_t>(elapsed_us);
    }
    auto make_sorted_entries = [](std::unordered_map<std::string, Entry>& values) {
        std::vector<Entry> entries;
        entries.reserve(values.size());
        for (auto& kv : values) {
            entries.push_back(std::move(kv.second));
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.total_us != b.total_us) {
                return a.total_us > b.total_us;
            }
            return a.key < b.key;
        });
        return entries;
    };
    std::vector<Entry> entries = make_sorted_entries(by_key);
    std::vector<Entry> op_entries = make_sorted_entries(by_op);
    const int op_limit = std::min<int>(12, op_entries.size());
    std::cerr << "[LLMNodeTimesByOp] stage=" << (stage ? stage : "<unknown>") << " showing=" << op_limit;
    for (int i = 0; i < op_limit; ++i) {
        const Entry& entry = op_entries[static_cast<size_t>(i)];
        std::cerr << " op" << (i + 1) << "=" << entry.key << ":" << (static_cast<double>(entry.total_us) / 1000.0)
                  << "ms/" << entry.count;
    }
    std::cerr << std::endl;
    const int limit = std::min<int>(LLMNodeTimingDumpLimit(), entries.size());
    std::cerr << "[LLMNodeTimes] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " measured_ms=" << (static_cast<double>(measured_us) / 1000.0) << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        const Entry& entry = entries[static_cast<size_t>(i)];
        std::cerr << "  rank=" << (i + 1) << " total_ms=" << (static_cast<double>(entry.total_us) / 1000.0)
                  << " count=" << entry.count << " key=" << entry.key << std::endl;
    }
}

void ResetPagedDecodeGraphExecutionState(ggml_cgraph* graph) {
    if (!graph) {
        return;
    }
    struct CustomOpParamsView {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(CustomOpParamsView) <= GGML_MAX_OP_PARAMS, "custom op params view too large");
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node || node->op != GGML_OP_CUSTOM) {
            continue;
        }
        CustomOpParamsView params{};
        std::memcpy(&params, node->op_params, sizeof(params));
        if (params.fun != cb_paged_attention_decode || !params.userdata) {
            continue;
        }
        auto* ud = static_cast<PagedAttentionUserData*>(params.userdata);
        ud->epoch_started.store(0, std::memory_order_relaxed);
        ud->epoch_done.store(0, std::memory_order_relaxed);
        ud->kv_writers_done.store(0, std::memory_order_relaxed);
        ud->shared_block_ptrs_ready.store(0, std::memory_order_relaxed);
    }
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

bool IsQwenHybridSSMModel(const TransformerModel* model) {
    if (!model || !model->arch_flags.is_hybrid_ssm) {
        return false;
    }
    const ModelVariant variant = densecore::models::DescribeModel(model).variant;
    return variant == ModelVariant::QWEN35 || variant == ModelVariant::QWEN36;
}

bool Qwen36HybridSSMProjectionWeightsAreNotQ4K(const TransformerModel* model) {
    if (!IsQwen36HybridSSMModel(model)) {
        return false;
    }
    bool saw_ssm_projection = false;
    for (const auto& layer : model->layers) {
        for (const char* key : {model_keys::kAttnQkvWeight, model_keys::kAttnGate, model_keys::kSSMOut}) {
            const ggml_tensor* tensor = layer.Get(key);
            if (!tensor) {
                continue;
            }
            saw_ssm_projection = true;
            if (tensor->type == GGML_TYPE_Q4_K) {
                return false;
            }
        }
    }
    return saw_ssm_projection;
}

struct Qwen36SSMProjectionTypeSummary {
    std::string actual_types;
    int quant_preserved = 0;
    int dequantized_count = 0;
};

Qwen36SSMProjectionTypeSummary SummarizeQwen36SSMProjectionTypes(const TransformerModel* model) {
    Qwen36SSMProjectionTypeSummary summary;
    if (!IsQwenHybridSSMModel(model)) {
        return summary;
    }
    std::map<std::string, int> type_counts;
    int saw = 0;
    int quantized = 0;
    int dense = 0;
    for (const auto& layer : model->layers) {
        for (const auto& item : {std::pair<const char*, const char*>("ssm_qkv", model_keys::kAttnQkvWeight),
                                 std::pair<const char*, const char*>("ssm_gate", model_keys::kAttnGate),
                                 std::pair<const char*, const char*>("ssm_out", model_keys::kSSMOut)}) {
            const ggml_tensor* tensor = layer.Get(item.second);
            if (!tensor) {
                continue;
            }
            ++saw;
            if (ggml_is_quantized(tensor->type)) {
                ++quantized;
            } else if (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_F32 ||
                       tensor->type == GGML_TYPE_BF16) {
                ++dense;
            }
            std::string key = std::string(item.first) + ":" + ggml_type_name(tensor->type);
            type_counts[key]++;
        }
    }
    std::ostringstream out;
    bool first = true;
    for (const auto& [key, count] : type_counts) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << key << ":" << count;
    }
    summary.actual_types = out.str();
    summary.quant_preserved = saw > 0 && quantized == saw ? 1 : 0;
    summary.dequantized_count = dense;
    return summary;
}

std::string SummarizeQwen36SSMQ8ProjectionCounts(const TransformerModel* model) {
    if (!IsQwen36HybridSSMModel(model)) {
        return "none";
    }
    uint64_t qkv = 0;
    uint64_t gate = 0;
    uint64_t out = 0;
    for (const auto& layer : model->layers) {
        const ggml_tensor* qkv_tensor = layer.Get(model_keys::kAttnQkvWeight);
        const ggml_tensor* gate_tensor = layer.Get(model_keys::kAttnGate);
        const ggml_tensor* out_tensor = layer.Get(model_keys::kSSMOut);
        if (qkv_tensor && qkv_tensor->type == GGML_TYPE_Q8_0) {
            ++qkv;
        }
        if (gate_tensor && gate_tensor->type == GGML_TYPE_Q8_0) {
            ++gate;
        }
        if (out_tensor && out_tensor->type == GGML_TYPE_Q8_0) {
            ++out;
        }
    }
    if (qkv == 0 && gate == 0 && out == 0) {
        return "none";
    }
    std::ostringstream counts;
    counts << "ssm_qkv:" << qkv << ",ssm_gate:" << gate << ",ssm_out:" << out;
    return counts.str();
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

const char* PrefixCacheSkipReasonForModel(const TransformerModel* model) {
    if (IsPrefixCacheReuseDisabled()) {
        return "disabled_by_env";
    }
    if (IsQwen36HybridSSMModel(model) && !IsQwen36PrefixCacheReuseEnabled()) {
        return "qwen36_prefix_cache_disabled";
    }
    return "none";
}

// Models whose layers carry per-sequence recurrent/conv state (hybrid SSM and
// LFM2 short-conv) cannot reconstruct that state from cached KV blocks alone.
// Prefix-cache reuse for them is only correct when the boundary state is
// snapshotted at block registration and restored on a cache hit. This predicate
// gates that snapshot/restore lifecycle (the machinery itself is state-agnostic;
// it copies SSMSequenceRuntimeState, which for LFM2 holds conv_state only).
bool ModelRequiresPrefixStateSnapshot(const TransformerModel* model) {
    return model && (model->arch_flags.is_hybrid_ssm || model->arch_flags.is_lfm2_shortconv);
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

BlockManager::HybridSSMSnapshotValidator BuildHybridSSMSnapshotValidatorForRequest(const TransformerModel* model,
                                                                                   const Request* req) {
    if (!ModelRequiresPrefixStateSnapshot(model) || !req || req->ssm_runtime_states.empty()) {
        return {};
    }

    const size_t expected_layers = req->ssm_runtime_states.size();
    size_t expected_conv_elems = 0;
    size_t expected_ssm_elems = 0;
    if (model->arch_flags.is_lfm2_shortconv) {
        // LFM2 short-conv layers keep only a conv-state ring (no SSM recurrent state).
        expected_conv_elems = TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(
            static_cast<int>(model->hparams.n_embd), model->lfm2_conv_kernel);
        expected_ssm_elems = 0;
    } else {
        const int expected_conv = model->ssm_inner_size + 2 * model->ssm_group_count * model->ssm_state_size;
        const int expected_head_dim = model->ssm_inner_size / std::max(1, model->ssm_time_step_rank);
        expected_conv_elems =
            TransformerModel::SSMSequenceRuntimeState::ExpectedConvStateElements(expected_conv, model->ssm_conv_kernel);
        expected_ssm_elems = TransformerModel::SSMSequenceRuntimeState::ExpectedStateElements(
            model->ssm_time_step_rank, expected_head_dim, model->ssm_state_size);
    }

    return [expected_layers, expected_conv_elems,
            expected_ssm_elems](const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
        if (states.size() != expected_layers) {
            return false;
        }
        for (const auto& state : states) {
            if (state.conv_state.size() != expected_conv_elems || state.ssm_state.size() != expected_ssm_elems) {
                return false;
            }
        }
        return true;
    };
}

void InitializeRequestPrefixCacheState(Request* req, const TransformerModel* model) {
    if (!req) {
        return;
    }
    if (req->original_prompt_tokens_for_cache.empty()) {
        req->original_prompt_tokens_for_cache = req->tokens;
    }
    if (req->prompt_tokens_for_cache.empty()) {
        req->prompt_tokens_for_cache = req->original_prompt_tokens_for_cache;
    }
    if (req->prompt_token_count <= 0) {
        req->prompt_token_count = static_cast<int>(req->original_prompt_tokens_for_cache.size());
    }
    req->prefix_cache_allowed = IsPrefixCacheAllowedForModel(model);
    if (!req->prefix_cache_allowed) {
        req->prefix_cache_skip_reason = PrefixCacheSkipReasonForModel(model);
    } else if (req->prefix_cache_skip_reason.empty()) {
        req->prefix_cache_skip_reason = "none";
    }
}

BlockManager::PrefixCacheMatch ProbeReusablePrefixCacheForRequest(PagedKVCache* kv_cache, const TransformerModel* model,
                                                                  const Request* req) {
    BlockManager::PrefixCacheMatch match;
    if (!kv_cache || !kv_cache->block_manager || !req || !req->prefix_cache_allowed ||
        req->original_prompt_tokens_for_cache.empty()) {
        return match;
    }
    const bool require_snapshot = ModelRequiresPrefixStateSnapshot(model);
    const auto snapshot_validator = BuildHybridSSMSnapshotValidatorForRequest(model, req);
    match = kv_cache->block_manager->FindLongestCachedPrefixWithVerification(
        req->original_prompt_tokens_for_cache.data(), static_cast<int>(req->original_prompt_tokens_for_cache.size()),
        require_snapshot, snapshot_validator);
    if (!match.cached_block_ids.empty()) {
        kv_cache->block_manager->Free(match.cached_block_ids);
    }
    return match;
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
    const int last_token_idx = token_offset + processed_count - 1;
    if (sampled_from_prefill && output_columns == 1) {
        *out_last_token_idx = 0;
        return true;
    }
    if (last_token_idx < token_offset || last_token_idx >= output_columns) {
        return fail("sampling logits column is out of bounds for output tensor");
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

int RequestPromptTokenCountForChunking(const Request* req) {
    if (!req) {
        return 0;
    }
    if (!req->original_prompt_tokens_for_cache.empty()) {
        return static_cast<int>(req->original_prompt_tokens_for_cache.size());
    }
    if (!req->prompt_tokens_for_cache.empty()) {
        return static_cast<int>(req->prompt_tokens_for_cache.size());
    }
    if (req->prompt_token_count > 0) {
        return req->prompt_token_count;
    }
    return static_cast<int>(req->tokens.size());
}

int ResolveQwen36PrefillChunkTokensImpl(const TransformerModel* model, const Request* req) {
    if (!model || !req) {
        return -1;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::QWEN35 && descriptor.variant != ModelVariant::QWEN36) {
        return -1;
    }
    const bool qwen35_dense = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts <= 0;
    const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
    const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
    const int hybrid_ssm_chunk_tokens =
#if defined(__aarch64__) || defined(_M_ARM64)
        192;
#else
        384;
#endif
    const int base_chunk_tokens =
        qwen35_dense ? 768 : ((qwen35_moe || qwen_hybrid_ssm) ? hybrid_ssm_chunk_tokens : 192);
    const int base_auto_min_tokens = qwen35_dense ? 1024 : (qwen35_moe ? 1024 : (qwen_hybrid_ssm ? 1280 : 1536));
    const char* chunk_env = "DENSECORE_QWEN36_PREFILL_CHUNK_TOKENS";
    const char* default_env = "DENSECORE_QWEN36_PREFILL_CHUNK_DEFAULT_TOKENS";
    const char* auto_min_env = "DENSECORE_QWEN36_PREFILL_CHUNK_AUTO_MIN_TOKENS";
    const int configured_default_chunk_tokens = densecore::env::ParsePositiveEnvInt(default_env, base_chunk_tokens);
    const int default_chunk_tokens = std::min(configured_default_chunk_tokens, base_chunk_tokens);

    const int explicit_tokens = densecore::env::ParsePositiveEnvInt(chunk_env, 0);
    if (explicit_tokens > 0) {
        return explicit_tokens;
    }

    const char* env_value = std::getenv(chunk_env);
    if (env_value && env_value[0] != '\0') {
        const std::string lowered = densecore::env::AsciiLowerCopy(env_value);
        if (lowered == "off" || lowered == "false" || lowered == "no") {
            const int prompt_tokens = RequestPromptTokenCountForChunking(req);
            const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, base_auto_min_tokens);
            if (qwen_hybrid_ssm && prompt_tokens >= auto_min_tokens) {
                return default_chunk_tokens;
            }
            return -1;
        }
        if (lowered == "on" || lowered == "true" || lowered == "yes" || lowered == "force") {
            return default_chunk_tokens;
        }
    }

    const int prompt_tokens = RequestPromptTokenCountForChunking(req);
    if (prompt_tokens <= 0) {
        return default_chunk_tokens;
    }
    const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, base_auto_min_tokens);
    return prompt_tokens >= auto_min_tokens ? default_chunk_tokens : -1;
}

int ResolveGemma4PrefillChunkTokensImpl(const TransformerModel* model, const Request* req) {
    if (!model || !req) {
        return -1;
    }
    const auto descriptor = densecore::models::DescribeModel(model);
    if (descriptor.variant != ModelVariant::GEMMA4 || model->hparams.n_experts <= 0) {
        return -1;
    }
    const char* chunk_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_TOKENS";
    const char* default_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_DEFAULT_TOKENS";
    const char* auto_min_env = "DENSECORE_GEMMA4_PREFILL_CHUNK_AUTO_MIN_TOKENS";
    const int explicit_tokens = densecore::env::ParsePositiveEnvInt(chunk_env, 0);
    if (explicit_tokens > 0) {
        return explicit_tokens;
    }

    const auto resolve_manual_default = [&]() { return densecore::env::ParsePositiveEnvInt(default_env, 128); };
    const char* env_value = std::getenv(chunk_env);
    bool explicit_auto = false;
    if (env_value && env_value[0] != '\0') {
        const std::string lowered = densecore::env::AsciiLowerCopy(env_value);
        if (lowered == "off" || lowered == "false" || lowered == "no") {
            return -1;
        }
        if (lowered == "on" || lowered == "true" || lowered == "yes" || lowered == "force") {
            return resolve_manual_default();
        }
        explicit_auto = (lowered == "0" || lowered == "auto");
    }
    if (!explicit_auto) {
        const densecore::env::RuntimeToggleMode mode =
            densecore::env::ParseRuntimeToggleMode(chunk_env, densecore::env::RuntimeToggleMode::Auto);
        if (mode == densecore::env::RuntimeToggleMode::Off) {
            return -1;
        }
        if (mode == densecore::env::RuntimeToggleMode::On) {
            return resolve_manual_default();
        }
    }

    const int prompt_tokens = RequestPromptTokenCountForChunking(req);
    if (prompt_tokens <= 0) {
        return resolve_manual_default();
    }
    const int auto_min_tokens = densecore::env::ParsePositiveEnvInt(auto_min_env, 1024);
    if (prompt_tokens < auto_min_tokens) {
        return -1;
    }
    if (densecore::env::ParsePositiveEnvInt(default_env, 0) > 0) {
        return resolve_manual_default();
    }

    // Gemma4 MoE prefill uses the transient ggml graph allocator, so the
    // default path can run the whole prompt without the KV history gathers
    // introduced by chunking.
    return -1;
}

int ResolveModelPrefillChunkTokens(const TransformerModel* model, const Request* req) {
    const int qwen36_tokens = ResolveQwen36PrefillChunkTokensImpl(model, req);
    if (qwen36_tokens != -1) {
        return qwen36_tokens;
    }
    return ResolveGemma4PrefillChunkTokensImpl(model, req);
}

size_t GraphContextSafetyMarginBytes() {
    return ParseSizeEnvMb("DENSECORE_GRAPH_CTX_SAFETY_MARGIN_MB", /*default_mb=*/512, /*min_mb=*/0,
                          /*max_mb=*/65536) *
           1024ULL * 1024ULL;
}

bool IsGraphContextAutoDowngradeEnabled() {
    return ParseBoolEnvDefault("DENSECORE_GRAPH_CTX_AUTO_DOWNGRADE", true);
}

bool IsGraphContextFailClosedEnabled() {
    return ParseBoolEnvDefault("DENSECORE_GRAPH_CTX_FAIL_CLOSED", true);
}

int ApplyGraphContextPrefillChunkDowngrade(const TransformerModel* model, Request* req, int chunk_tokens) {
    if (!model || !req || !IsGraphContextAutoDowngradeEnabled()) {
        return chunk_tokens;
    }
    const size_t available_bytes = ReadAvailableMemoryBytesForRuntimePools();
    if (available_bytes == 0) {
        return chunk_tokens;
    }
    const size_t safety_margin_bytes = GraphContextSafetyMarginBytes();
    const size_t prompt_tokens = static_cast<size_t>(std::max(1, RequestPromptTokenCountForChunking(req)));
    int effective_chunk = chunk_tokens;
    const bool originally_unchunked = effective_chunk <= 1;
    const auto initial_estimate = EngineState::EstimateGraphContextSize(
        model, prompt_tokens, /*num_seqs_hint=*/1, originally_unchunked ? 0 : static_cast<size_t>(effective_chunk));
    req->graph_ctx_requested_mb = initial_estimate.total_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_available_mb = available_bytes / (1024ULL * 1024ULL);
    req->graph_ctx_safety_margin_mb = safety_margin_bytes / (1024ULL * 1024ULL);
    if (initial_estimate.total_bytes + safety_margin_bytes <= available_bytes) {
        return chunk_tokens;
    }
    if (originally_unchunked) {
        const auto descriptor = densecore::models::DescribeModel(model);
        const bool qwen35_moe = descriptor.variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0;
        const bool qwen_hybrid_ssm = model->arch_flags.is_hybrid_ssm;
        effective_chunk = (qwen35_moe || qwen_hybrid_ssm) ? 384 : 512;
        effective_chunk = std::max(1, std::min<int>(effective_chunk, static_cast<int>(prompt_tokens)));
    }
    const int best_effort_pressure_chunk = std::max(1, effective_chunk);

    while (effective_chunk > 1) {
        const auto estimate = EngineState::EstimateGraphContextSize(model, prompt_tokens, /*num_seqs_hint=*/1,
                                                                    static_cast<size_t>(effective_chunk));
        if (estimate.total_bytes + safety_margin_bytes <= available_bytes) {
            req->graph_ctx_downgraded_chunk_tokens = effective_chunk;
            req->graph_ctx_fail_reason.clear();
            std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
                      << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
                      << " downgraded_chunk_tokens=" << effective_chunk
                      << " requested_mb=" << (initial_estimate.total_bytes / (1024ULL * 1024ULL))
                      << " downgraded_mb=" << (estimate.total_bytes / (1024ULL * 1024ULL))
                      << " available_mb=" << req->graph_ctx_available_mb
                      << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << std::endl;
            return effective_chunk;
        }
        effective_chunk = std::max(1, effective_chunk / 2);
    }

    effective_chunk = std::max(1, best_effort_pressure_chunk);
    req->graph_ctx_downgraded_chunk_tokens = effective_chunk;
    req->graph_ctx_fail_reason.clear();
    std::cerr << "[DenseCore] GraphCtxAutoDowngrade" << " req=" << req->id
              << " original_chunk_tokens=" << (originally_unchunked ? 0 : chunk_tokens)
              << " downgraded_chunk_tokens=" << effective_chunk
              << " requested_mb=" << (initial_estimate.total_bytes / (1024ULL * 1024ULL)) << " downgraded_mb=unknown"
              << " available_mb=" << req->graph_ctx_available_mb
              << " safety_margin_mb=" << req->graph_ctx_safety_margin_mb << " reason=best_effort_pressure_chunk"
              << std::endl;
    return effective_chunk;
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

int ResolveGemma4PrefillChunkTokens(const TransformerModel* model, const Request* req) {
    return ResolveGemma4PrefillChunkTokensImpl(model, req);
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
            using WorkContextPtr = std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)>;

            std::vector<uint8_t> ctx_buffer;
            struct ggml_context* ctx = nullptr;
            struct ggml_cgraph* graph = nullptr;
            struct ggml_tensor* output = nullptr;
            struct ggml_tensor* embd_inp = nullptr;
            struct ggml_tensor* pos = nullptr;
            WorkContextPtr work_ctx{nullptr, DestroyInferenceWorkContext};
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
            bool skip_output_logits = false;

            bool operator==(const PrefillGraphCacheKey& other) const {
                return batch_size == other.batch_size && tokens == other.tokens && n_past == other.n_past &&
                       threads == other.threads && model_id == other.model_id && arch_id == other.arch_id &&
                       cache_type_id == other.cache_type_id && feature_flags == other.feature_flags &&
                       skip_output_logits == other.skip_output_logits;
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
                mix(key.skip_output_logits ? 1ull : 0ull);
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
        struct PrefillArenaEntry {
            std::unique_ptr<ScopedGgmlGraphArena> arena;
            size_t metadata_bytes = 0;
            std::list<PrefillGraphCacheKey>::iterator lru_it;
        };

        std::unordered_map<DecodeGraphCacheKey, DecodeGraphCacheEntry, DecodeGraphCacheKeyHash> decode_graph_cache;
        std::list<DecodeGraphCacheKey> decode_graph_lru;
        std::unordered_set<DecodeGraphCacheKey, DecodeGraphCacheKeyHash> decode_graph_uncacheable;
        std::list<DecodeGraphCacheKey> decode_graph_uncacheable_lru;
        std::unordered_map<PrefillGraphCacheKey, PrefillGraphCacheEntry, PrefillGraphCacheKeyHash> prefill_graph_cache;
        std::list<PrefillGraphCacheKey> prefill_graph_lru;
        size_t prefill_graph_cache_bytes = 0;
        std::unordered_map<PrefillGraphCacheKey, PrefillArenaEntry, PrefillGraphCacheKeyHash> prefill_arena_pool;
        std::list<PrefillGraphCacheKey> prefill_arena_lru;
        std::unordered_map<std::string, FlexibleGraphPoolSizing> flexible_graph_pool_sizing_cache;
        size_t content_dependent_prefill_grown_until_bytes = 0;

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
            entry->work_ctx.reset();
            entry->ctx_buffer.clear();
        };
        auto reset_cached_graph_runtime_context = [](InferenceWorkContext* ctx) {
            ResetCachedDecodeGraphWorkContext(ctx);
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
        auto touch_prefill_arena_entry = [&](PrefillArenaEntry* entry) {
            if (!entry) return;
            prefill_arena_lru.splice(prefill_arena_lru.begin(), prefill_arena_lru, entry->lru_it);
            entry->lru_it = prefill_arena_lru.begin();
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
            const bool global_direct_callback = state->fast_path_config.worker.callback_mode ==
                                                densecore::llm::config::WorkerRuntimeConfig::CallbackMode::Direct;
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
                    LogRequestLifecyclePhase("engine_queue_pop", req, current_model);

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
                    EnsureRequestLFM2RuntimeState(current_model, req);
                    InitializeRequestPrefixCacheState(req, current_model);
                    LogDeterminismBoundary("after_tokenization", req, current_model,
                                           /*prefix_cache_allowed=*/req->prefix_cache_allowed,
                                           /*prefix_cache_hit=*/false,
                                           /*hybrid_restore_attempted=*/false,
                                           /*hybrid_restore_applied=*/false,
                                           /*chunked_prefill=*/false,
                                           /*decode_cache_active=*/false,
                                           /*decode_cache_reused=*/false,
                                           /*prefill_cache_active=*/false);

                    // Register with scheduler (blocks allocated by scheduler)
                    const int resolved_prefill_chunk_tokens = ResolveModelPrefillChunkTokens(current_model, req);
                    const int prefill_chunk_tokens =
                        ApplyGraphContextPrefillChunkDowngrade(current_model, req, resolved_prefill_chunk_tokens);
                    req->prefill_chunk_tokens_effective = std::max(0, prefill_chunk_tokens);
                    const std::vector<int>* scheduler_prefix_tokens =
                        req->prefix_cache_allowed ? &req->original_prompt_tokens_for_cache : nullptr;
                    const auto hybrid_ssm_snapshot_validator =
                        BuildHybridSSMSnapshotValidatorForRequest(current_model, req);
                    int seq_id = state->scheduler->AddRequest(
                        req->id, static_cast<int>(req->original_prompt_tokens_for_cache.size()), req->max_tokens,
                        req->priority, scheduler_prefix_tokens,
                        /*allow_chunked_prefill=*/!req->is_embedding,
                        /*require_hybrid_ssm_prefix_snapshot=*/ModelRequiresPrefixStateSnapshot(current_model),
                        prefill_chunk_tokens, hybrid_ssm_snapshot_validator);

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
                    LogRequestLifecyclePhase("scheduler_admit", req, current_model, seq_id);

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
                if (!state->scheduler) return;
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
            if (state->pending_requests.Empty()) {
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
                    const auto prefix_probe =
                        ProbeReusablePrefixCacheForRequest(current_kv_cache, current_model, single_req);
                    if (prefix_probe.cached_tokens > 0 && !prefix_probe.cached_block_ids.empty()) {
                        single_req->prefix_cache_skip_reason = "single_request_fast_path_bypassed_for_prefix_cache";
                        LOG_INFO("Single-request fast path skipped for req {}: prefix_cache_hit_possible "
                                 "cached_tokens={} cached_blocks={}",
                                 single_req->id, prefix_probe.cached_tokens, prefix_probe.cached_block_ids.size());
                    } else {
                        const int tokens_to_process =
                            single_req->is_prefill ? static_cast<int>(single_req->tokens.size()) : 1;
                        if (tokens_to_process > 0) {
                            const int blocks_needed =
                                (single_req->n_past + tokens_to_process + BLOCK_SIZE - 1) / BLOCK_SIZE;
                            if (static_cast<int>(single_req->block_table.size()) < blocks_needed) {
                                const int missing_blocks =
                                    blocks_needed - static_cast<int>(single_req->block_table.size());
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
                        req->prefix_cache_allowed = prefix_cache_allowed;
                        req->prefix_cache_hit = true;
                        req->prefix_cache_skipped_tokens += hit.cached_tokens;
                        req->prefix_cache_hit_blocks += static_cast<int>(hit.cached_block_ids.size());
                        req->prefix_cache_skip_reason = "none";
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

                        if (ModelRequiresPrefixStateSnapshot(current_model) && !hit.cached_block_ids.empty() &&
                            !req->ssm_runtime_states.empty()) {
                            std::vector<TransformerModel::SSMSequenceRuntimeState> snapshot;
                            const int snapshot_block = hit.cached_block_ids.back();
                            const auto snapshot_validator =
                                BuildHybridSSMSnapshotValidatorForRequest(current_model, req);
                            auto snapshot_shape_ok =
                                [&](const std::vector<TransformerModel::SSMSequenceRuntimeState>& states) {
                                    if (hit.cached_tokens <= 0 || hit.cached_tokens % BLOCK_SIZE != 0) {
                                        return false;
                                    }
                                    if (hit.cached_block_ids.empty() || snapshot_block != hit.cached_block_ids.back()) {
                                        return false;
                                    }
                                    return snapshot_validator && snapshot_validator(states);
                                };
                            determinism_hybrid_restore_attempt_reqs.insert(req->id);
                            req->hybrid_ssm_snapshot_restore_attempted = true;
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
                                req->hybrid_ssm_snapshot_restore_applied = true;
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

            size_t reserved_batch_tokens = sched_output.decode_seq_ids.size();
            for (int seq_id : sched_output.prefill_seq_ids) {
                auto req_it = seq_to_request.find(seq_id);
                if (req_it == seq_to_request.end() || !req_it->second) {
                    continue;
                }
                int tokens_to_take = static_cast<int>(req_it->second->tokens.size());
                auto chunk_it = prefill_chunk_tokens.find(seq_id);
                if (chunk_it != prefill_chunk_tokens.end()) {
                    tokens_to_take = std::min(tokens_to_take, chunk_it->second);
                }
                if (req_it->second && req_it->second->prefill_chunk_tokens_effective > 0) {
                    tokens_to_take = std::min(tokens_to_take, req_it->second->prefill_chunk_tokens_effective);
                }
                if (tokens_to_take > 0) {
                    reserved_batch_tokens += static_cast<size_t>(tokens_to_take);
                }
            }
            const size_t reserved_batch_requests =
                sched_output.prefill_seq_ids.size() + sched_output.decode_seq_ids.size();
            batch.tokens.reserve(reserved_batch_tokens);
            batch.pos.reserve(reserved_batch_tokens);
            batch.seq_id.reserve(reserved_batch_tokens);
            batch.inputs.reserve(reserved_batch_requests);
            batch.block_tables.reserve(reserved_batch_requests);
            batch.n_past.reserve(reserved_batch_requests);
            batch.scheduler_seq_ids.reserve(reserved_batch_requests);
            batch.hybrid_ssm_runtime_states.reserve(reserved_batch_requests);
            batch_requests.reserve(reserved_batch_requests);
            batch_token_counts.reserve(reserved_batch_requests);

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
                if (req->prefill_chunk_tokens_effective > 0) {
                    tokens_to_take = std::min(tokens_to_take, req->prefill_chunk_tokens_effective);
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

                // Save full original prompt tokens once for prefix cache registration.
                if (req->original_prompt_tokens_for_cache.empty()) {
                    req->original_prompt_tokens_for_cache = req->tokens;
                }
                if (req->prompt_tokens_for_cache.empty()) {
                    req->prompt_tokens_for_cache = req->original_prompt_tokens_for_cache;
                }
                if (req->prompt_token_count <= 0) {
                    req->prompt_token_count = static_cast<int>(req->original_prompt_tokens_for_cache.size());
                }

                int batch_seq_idx = batch_requests.size();
                const auto token_begin = req->tokens.begin();
                const auto token_end = req->tokens.begin() + tokens_to_take;
                batch.tokens.insert(batch.tokens.end(), token_begin, token_end);
                for (int i = 0; i < tokens_to_take; ++i) {
                    batch.pos.push_back(req->n_past + i);
                }
                batch.seq_id.insert(batch.seq_id.end(), static_cast<size_t>(tokens_to_take), batch_seq_idx);
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
                input.tokens.assign(token_begin, token_end);
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
            LogRequestLifecyclePhase("batch_ready", !batch_requests.empty() ? batch_requests.front() : nullptr,
                                     current_model,
                                     !batch_requests.empty() && batch_requests.front() ? batch_requests.front()->seq_id
                                                                                       : -1,
                                     batch.num_seqs, static_cast<int>(batch.tokens.size()));

            bool is_prefill_batch = false;
            for (Request* req : batch_requests) {
                if (req->is_prefill) {
                    is_prefill_batch = true;
                    break;
                }
            }
            const InferenceExecutionPhase batch_execution_phase =
                is_prefill_batch ? InferenceExecutionPhase::Prefill : InferenceExecutionPhase::Decode;
            const char* callback_mode_label = global_direct_callback ? "direct" : "async";
            for (Request* req : batch_requests) {
                if (req) {
                    req->callback_mode = callback_mode_label;
                }
            }
            bool main_work_context_bound_for_current_graph = false;
            std::unique_ptr<ScopedBatchWorkContext> main_work_context_scope;
            auto bind_main_work_context_for_graph = [&](const BatchSpec& graph_batch, InferenceExecutionPhase phase) {
                main_work_context_scope.reset();
                main_work_context_scope = std::make_unique<ScopedBatchWorkContext>(
                    work_ctx.get(), &graph_batch, phase, true,
                    current_model ? current_model->variant : ModelVariant::UNKNOWN);
                main_work_context_bound_for_current_graph = true;
                bool qwen36_amx_prepared = false;
                int qwen36_amx_mode = 0;
                for (const Request* req : batch_requests) {
                    if (req && req->qwen36_ssm_q8_prefill_amx_prepared != 0) {
                        qwen36_amx_prepared = true;
                        qwen36_amx_mode = req->qwen36_ssm_q8_prefill_amx_mode;
                        break;
                    }
                }
                if (qwen36_amx_prepared) {
                    RecordQwen36SSMQ8PrefillAMXPrepared(work_ctx.get(), qwen36_amx_mode);
                }
            };
            const bool prefill_intermediate_chunk =
                is_prefill_batch && !is_embedding_batch && batch_requests.size() == 1 &&
                batch_token_counts.size() == 1 && batch_requests.front() && batch_requests.front()->is_prefill &&
                batch_token_counts.front() < static_cast<int>(batch_requests.front()->tokens.size());
            batch.skip_output_logits = prefill_intermediate_chunk;
            InferenceConfig& infer_cfg = InferenceConfig::Instance();
            int active_threads = base_threads;
            const int decode_batch_size = static_cast<int>(batch_requests.size());
            int prefill_prompt_token_count = 0;
            if (is_prefill_batch && decode_batch_size == 1 && !batch_requests.empty()) {
                const Request* const req = batch_requests.front();
                if (req && !req->original_prompt_tokens_for_cache.empty()) {
                    prefill_prompt_token_count = static_cast<int>(req->original_prompt_tokens_for_cache.size());
                } else if (req && !req->prompt_tokens_for_cache.empty()) {
                    prefill_prompt_token_count = static_cast<int>(req->prompt_tokens_for_cache.size());
                } else if (!batch_token_counts.empty()) {
                    prefill_prompt_token_count = batch_token_counts.front();
                }
            }
            const bool is_decode_batch = !is_prefill_batch && !is_embedding_batch;
            if (is_decode_batch) {
                ClearQwen36SSMQ8PrefillAMXAliases(current_model);
            } else if (is_prefill_batch && current_model && current_model->variant == ModelVariant::QWEN36 &&
                       current_model->arch_flags.is_hybrid_ssm &&
                       state->fast_path_config.qwen36_ssm_q8_prefill_amx !=
                           densecore::llm::config::Qwen36SSMQ8PrefillAMXMode::Off &&
                       prefill_prompt_token_count >= state->fast_path_config.qwen36_ssm_q8_prefill_amx_min_tokens) {
                const bool prepared_qwen36_ssm_q8_prefill_amx =
                    PrepareQwen36SSMQ8PrefillAMXAliasesForExecution(current_model);
                if (prepared_qwen36_ssm_q8_prefill_amx) {
                    const std::string projection_counts = SummarizeQwen36SSMQ8ProjectionCounts(current_model);
                    for (Request* req : batch_requests) {
                        if (!req) {
                            continue;
                        }
                        req->qwen36_ssm_q8_prefill_amx_mode =
                            static_cast<int>(state->fast_path_config.qwen36_ssm_q8_prefill_amx);
                        req->qwen36_ssm_q8_prefill_amx_prepared = 1;
                        req->qwen36_ssm_q8_prefill_amx_prepared_projection_counts = projection_counts;
                    }
                }
            }
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

            // Keep decode within user-provided base thread budget unless
            // explicitly opting into oversubscription.
            if (!is_prefill_batch && base_threads > 0 && !AllowDecodeThreadsOverBase()) {
                active_threads = std::min(active_threads, base_threads);
            }
            int final_thread_cap = std::max(1, physical_core_count);
            if (current_model && base_threads >= 16) {
                const auto descriptor = densecore::models::DescribeModel(current_model);
                const bool qwen35_or_qwen36 =
                    descriptor.variant == ModelVariant::QWEN35 || descriptor.variant == ModelVariant::QWEN36;
                const char* benchmark_profile_env = std::getenv("DENSECORE_BENCHMARK_PROFILE");
                const bool benchmark_or_server_perf_profile =
                    state->fast_path_config.bench_respect_threads ||
                    (benchmark_profile_env && benchmark_profile_env[0] != '\0');
                if (qwen35_or_qwen36 && benchmark_or_server_perf_profile) {
                    final_thread_cap = std::max(final_thread_cap, base_threads);
                }
            }
            active_threads = std::max(1, std::min(active_threads, final_thread_cap));
            infer_cfg.num_threads = active_threads;
            if (is_decode_batch && decode_batch_size >= 1 &&
                decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
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
            bool using_transient_graph_arena = false;
            ScopedGgmlGraphArena* transient_graph_arena = nullptr;
            uint64_t graph_runtime_rebind_ns = 0;

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
                    const size_t request_key_hint =
                        batch_req ? static_cast<size_t>(std::max(1, batch_req->n_past + token_count))
                                  : static_cast<size_t>(std::max(1, token_count));
                    seq_len_hint = std::max(seq_len_hint, request_key_hint);
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
                    const bool single_decode_forced_paged = paged_decode_supported && batch.num_seqs == 1;
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
            auto note_decode_graph_cache_skip = [&](const char* reason) {
                DecodeWorkerStats& decode_stats = GetDecodeWorkerStats();
                const char* safe_reason = (reason && *reason) ? reason : "unknown";
                if (std::strcmp(safe_reason, "disabled") == 0) {
                    decode_stats.graph_cache_skip_disabled.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "layout") == 0) {
                    decode_stats.graph_cache_skip_layout.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "max_batch") == 0) {
                    decode_stats.graph_cache_skip_max_batch.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "unstable") == 0) {
                    decode_stats.graph_cache_skip_unstable.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "lora") == 0) {
                    decode_stats.graph_cache_skip_lora.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "model") == 0) {
                    decode_stats.graph_cache_skip_model.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "backend") == 0) {
                    decode_stats.graph_cache_skip_backend.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "key_uncacheable") == 0) {
                    decode_stats.graph_cache_skip_uncacheable.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "build_failure") == 0) {
                    decode_stats.graph_cache_skip_build_failure.fetch_add(1, std::memory_order_relaxed);
                } else if (std::strcmp(safe_reason, "rebind_failure") == 0) {
                    decode_stats.graph_cache_skip_rebind_failure.fetch_add(1, std::memory_order_relaxed);
                }
                for (Request* req : batch_requests) {
                    if (!req) {
                        continue;
                    }
                    req->graph_cache_skip_count++;
                    req->graph_cache_last_skip_reason = safe_reason;
                }
            };
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
            if (is_decode_batch && decode_batch_size >= 1 &&
                decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches) &&
                !decode_reuse_attempt_allowed) {
                if (!decode_graph_cache_active_effective) {
                    note_decode_graph_cache_skip("disabled");
                } else if (batch.num_seqs > decode_graph_cache_max_batch) {
                    note_decode_graph_cache_skip("max_batch");
                } else if (!decode_single_token_layout) {
                    note_decode_graph_cache_skip("layout");
                } else if (!decode_topology_stable) {
                    note_decode_graph_cache_skip("unstable");
                } else if (!batch.lora_map.empty()) {
                    note_decode_graph_cache_skip("lora");
                } else if (!IsDecodeGraphCacheSafeForModel(current_model)) {
                    note_decode_graph_cache_skip("model");
                } else if (!cpu_backend_active) {
                    note_decode_graph_cache_skip("backend");
                } else if (decode_key_marked_uncacheable) {
                    note_decode_graph_cache_skip("key_uncacheable");
                }
            }
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
                prefill_graph_key.skip_output_logits = batch.skip_output_logits;
            }

            std::chrono::steady_clock::time_point graph_build_begin, graph_build_end;
            bool graph_build_accounted = false;
            bool built_decode_graph_cache_entry = false;
            bool reused_prefill_graph = false;
            bool using_cached_prefill_graph = false;
            bool built_prefill_graph_cache_entry = false;
            if (decode_reuse_attempt_allowed) {
                if (is_decode_batch && decode_batch_size >= 1 &&
                    decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
                    GetDecodeWorkerStats().graph_cache_attempts.fetch_add(1, std::memory_order_relaxed);
                    GetDecodeWorkerStats()
                        .graph_cache_by_variant_batch[decode_model_variant_index][decode_batch_bucket]
                        .attempts.fetch_add(1, std::memory_order_relaxed);
                }
                auto it = decode_graph_cache.find(decode_graph_key);
                if (it != decode_graph_cache.end() && it->second.graph && it->second.output && it->second.embd_inp &&
                    it->second.pos && it->second.work_ctx) {
                    bool rebind_ok = true;
                    reset_cached_graph_runtime_context(it->second.work_ctx.get());
                    if (DoesDecodeGraphCacheRequireRuntimeRebind(current_model)) {
                        const auto rebind_begin = std::chrono::steady_clock::now();
                        rebind_ok = RebindDecodeGraphRuntimeStateForModel(current_model, it->second.graph, batch);
                        graph_runtime_rebind_ns +=
                            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      std::chrono::steady_clock::now() - rebind_begin)
                                                      .count());
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
                        if (is_decode_batch && decode_batch_size >= 1 &&
                            decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
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
                    GetDecodeWorkerStats().prefill_graph_reuse_hit.fetch_add(1, std::memory_order_relaxed);
                    for (Request* req : batch_requests) {
                        if (req) {
                            req->graph_cache_hit_count++;
                        }
                    }
                }
            }

            bool decode_cache_insert_skip_recorded = false;
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
                    candidate.work_ctx = DecodeGraphCacheEntry::WorkContextPtr(CreateInferenceWorkContext(),
                                                                               DestroyInferenceWorkContext);
                    struct ggml_init_params decode_params = {
                        .mem_size = DecodeGraphCacheCtxBytes(),
                        .mem_buffer = nullptr,
                        .no_alloc = false,
                    };
                    candidate.ctx = candidate.work_ctx ? ggml_init(decode_params) : nullptr;
                    if (candidate.ctx) {
                        candidate.graph = ggml_new_graph_custom(candidate.ctx, 32768, false);
                        if (candidate.graph) {
                            graph_build_begin = std::chrono::steady_clock::now();
                            {
                                ScopedBatchWorkContext cache_build_ctx(candidate.work_ctx.get(), &batch,
                                                                       InferenceExecutionPhase::Decode);
                                candidate.output = BuildTransformerGraph(current_model, current_kv_cache, candidate.ctx,
                                                                         batch, is_embedding_batch, candidate.graph,
                                                                         &candidate.embd_inp, &candidate.pos);
                            }
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
                                        if (is_decode_batch && decode_batch_size >= 1 &&
                                            decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
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
                                if (cache_entry_admissible && DoesDecodeGraphCacheRequireRuntimeRebind(current_model)) {
                                    const auto rebind_begin = std::chrono::steady_clock::now();
                                    const bool rebind_ok =
                                        RebindDecodeGraphRuntimeStateForModel(current_model, candidate.graph, batch);
                                    graph_runtime_rebind_ns +=
                                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                  std::chrono::steady_clock::now() - rebind_begin)
                                                                  .count());
                                    if (!rebind_ok) {
                                        cache_entry_admissible = false;
                                        note_decode_graph_cache_skip("rebind_failure");
                                        decode_cache_insert_skip_recorded = true;
                                        if (IsDebugGraphLoggingEnabled()) {
                                            std::cerr << "[DecodeGraphCache] skip cache insert: hybrid SSM runtime "
                                                         "rebind probe failed"
                                                      << " (bs=" << decode_graph_key.batch_size
                                                      << ", threads=" << decode_graph_key.threads << ")" << std::endl;
                                        }
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
                                    if (is_decode_batch && decode_batch_size >= 1 &&
                                        decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
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
                        if (!decode_cache_insert_skip_recorded) {
                            note_decode_graph_cache_skip("build_failure");
                        }
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
                            bind_main_work_context_for_graph(batch, batch_execution_phase);
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
                    const bool use_transient_prefill_arena =
                        is_prefill_batch && !is_embedding_batch && cpu_backend_active && current_model &&
                        current_model->arch_flags.is_gemma4 && batch.num_seqs == 1 && batch.lora_map.empty();
                    if (use_transient_prefill_arena) {
                        constexpr size_t MB = 1024ULL * 1024ULL;
                        const size_t metadata_bytes = 512ULL * MB;
                        PrefillGraphCacheKey arena_key{};
                        int n_past = 0;
                        if (!batch.n_past.empty()) {
                            n_past = std::max(0, batch.n_past[0]);
                        }
                        arena_key.batch_size = batch.num_seqs;
                        arena_key.tokens = static_cast<int>(batch.tokens.size());
                        arena_key.n_past = n_past;
                        arena_key.threads = active_threads;
                        arena_key.model_id = reinterpret_cast<uintptr_t>(current_model);
                        arena_key.arch_id = current_model ? static_cast<int>(current_model->arch) : -1;
                        arena_key.cache_type_id =
                            current_kv_cache ? static_cast<int>(current_kv_cache->cache_type) : -1;
                        arena_key.feature_flags = BuildDecodeGraphFeatureFlags(current_model);
                        arena_key.skip_output_logits = batch.skip_output_logits;

                        PrefillArenaEntry* arena_entry = nullptr;
                        auto arena_it = prefill_arena_pool.find(arena_key);
                        if (arena_it != prefill_arena_pool.end() && arena_it->second.arena) {
                            touch_prefill_arena_entry(&arena_it->second);
                            arena_entry = &arena_it->second;
                            GetDecodeWorkerStats().prefill_arena_reuse_hit.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            GetDecodeWorkerStats().prefill_arena_reuse_miss.fetch_add(1, std::memory_order_relaxed);
                            while (prefill_arena_pool.size() >= static_cast<size_t>(prefill_graph_cache_lru_size) &&
                                   !prefill_arena_lru.empty()) {
                                const PrefillGraphCacheKey evict_key = prefill_arena_lru.back();
                                prefill_arena_lru.pop_back();
                                prefill_arena_pool.erase(evict_key);
                            }
                            PrefillArenaEntry inserted_entry;
                            inserted_entry.arena = std::make_unique<ScopedGgmlGraphArena>();
                            inserted_entry.metadata_bytes = metadata_bytes;
                            prefill_arena_lru.push_front(arena_key);
                            inserted_entry.lru_it = prefill_arena_lru.begin();
                            auto inserted = prefill_arena_pool.emplace(arena_key, std::move(inserted_entry));
                            arena_entry = &inserted.first->second;
                        }
                        if (current_model && current_model->hparams.n_experts > 0) {
                            GetDecodeWorkerStats().prefill_graph_reuse_skip_model.fetch_add(1,
                                                                                            std::memory_order_relaxed);
                        }
                        transient_graph_arena = arena_entry ? arena_entry->arena.get() : nullptr;
                        if (!transient_graph_arena || !transient_graph_arena->Ensure(metadata_bytes, active_backend)) {
                            throw densecore::OutOfMemoryException(
                                "Gemma4 prefill graph arena metadata allocation failed");
                        }
                        struct ggml_context* ctx_nodes = transient_graph_arena->ctx();
                        gf = ggml_new_graph_custom(ctx_nodes, 32768, false);
                        bind_main_work_context_for_graph(batch, batch_execution_phase);
                        graph_build_begin = std::chrono::steady_clock::now();
                        output = BuildTransformerGraph(current_model, current_kv_cache, ctx_nodes, batch,
                                                       is_embedding_batch, gf, &embd_inp, &pos);
                        if (!output || !embd_inp || !pos || !transient_graph_arena->AllocGraph(gf)) {
                            throw densecore::OutOfMemoryException("Gemma4 prefill graph arena allocation failed");
                        }
                        graph_build_end = std::chrono::steady_clock::now();
                        using_transient_graph_arena = true;
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
                    } else if (is_prefill_batch && ShouldZeroFillPrefillGraphBuffer() &&
                               state->inference_ctx.compute_buffer && state->inference_ctx.compute_buffer_size > 0) {
                        std::memset(state->inference_ctx.compute_buffer, 0, state->inference_ctx.compute_buffer_size);
                    }
                    if (!using_transient_graph_arena) {
                        size_t ctx_size = graph_ctx_estimate.total_bytes;
                        FlexibleGraphPoolSizing flexible_sizing{};
                        bool used_flexible_sizing = false;
                        if (IsFlexibleGraphPoolSizingEnabled(current_model)) {
                            const bool content_dependent_prefill_graph =
                                is_prefill_batch && current_model &&
                                (current_model->arch_flags.is_hybrid_ssm || current_model->arch_flags.is_gemma4 ||
                                 current_model->hparams.n_experts > 0);
                            std::string sizing_key = std::to_string(reinterpret_cast<uintptr_t>(current_model));
                            sizing_key += ":" + std::to_string(static_cast<int>(is_embedding_batch));
                            sizing_key += ":" + std::to_string(graph_ctx_estimate.effective_seq_len);
                            sizing_key += ":" + std::to_string(graph_ctx_estimate.effective_query_len);
                            sizing_key += ":" + std::to_string(graph_ctx_estimate.effective_num_seqs);
                            sizing_key += ":" + std::to_string(graph_ctx_estimate.chunk_token_hint);
                            auto cached_sizing = content_dependent_prefill_graph
                                                     ? flexible_graph_pool_sizing_cache.end()
                                                     : flexible_graph_pool_sizing_cache.find(sizing_key);
                            if (cached_sizing == flexible_graph_pool_sizing_cache.end()) {
                                flexible_sizing = MeasureFlexibleGraphPoolSize(current_model, current_kv_cache, batch,
                                                                               is_embedding_batch, graph_ctx_estimate);
                                if (flexible_sizing.ok && !content_dependent_prefill_graph) {
                                    cached_sizing =
                                        flexible_graph_pool_sizing_cache.emplace(sizing_key, flexible_sizing).first;
                                }
                            }
                            if (content_dependent_prefill_graph && flexible_sizing.ok) {
                                used_flexible_sizing = true;
                            } else if (cached_sizing != flexible_graph_pool_sizing_cache.end() &&
                                       cached_sizing->second.ok) {
                                flexible_sizing = cached_sizing->second;
                                used_flexible_sizing = true;
                            }
                            if (used_flexible_sizing) {
                                ctx_size = flexible_sizing.required_bytes > 0 ? flexible_sizing.required_bytes
                                                                              : flexible_sizing.reserved_bytes;
                                bool content_dependent_grow_applied = false;
                                if (is_prefill_batch && graph_ctx_estimate.effective_query_len > 1) {
                                    ctx_size = std::max(ctx_size, flexible_sizing.reserved_bytes);
                                }
                                const size_t measured_floor =
                                    AlignUpBytes(flexible_sizing.dry_metadata_bytes +
                                                     flexible_sizing.graph_tensor_bytes + flexible_sizing.margin_bytes,
                                                 64ULL * 1024ULL * 1024ULL);
                                if (measured_floor > 0 && ctx_size < measured_floor) {
                                    const RuntimeGraphPoolReservation measured_floor_reservation =
                                        ClampRuntimeGraphPoolReservation(measured_floor,
                                                                         flexible_sizing.available_bytes);
                                    ctx_size = std::max(ctx_size, measured_floor_reservation.total_bytes);
                                }
                                if (content_dependent_prefill_graph && state->inference_ctx.IsInitialized() &&
                                    state->inference_ctx.compute_buffer_size > 0 &&
                                    state->inference_ctx.compute_buffer_size >= ctx_size &&
                                    state->inference_ctx.compute_buffer_size >
                                        content_dependent_prefill_grown_until_bytes) {
                                    const size_t current_pool_size = state->inference_ctx.compute_buffer_size;
                                    const size_t content_dependent_grow_ahead =
                                        std::max<size_t>(1, current_pool_size / 32);
                                    ctx_size = AlignUpBytes(current_pool_size + content_dependent_grow_ahead,
                                                            64ULL * 1024ULL * 1024ULL);
                                    content_dependent_prefill_grown_until_bytes = ctx_size;
                                    content_dependent_grow_applied = true;
                                }
                                if (state->inference_ctx.IsInitialized() &&
                                    state->inference_ctx.compute_buffer_size < ctx_size &&
                                    !content_dependent_grow_applied) {
                                    const size_t grow_ahead = std::max<size_t>(
                                        state->inference_ctx.compute_buffer_size / 2, 2048ULL * 1024ULL * 1024ULL);
                                    ctx_size = std::max<size_t>(
                                        ctx_size, AlignUpBytes(state->inference_ctx.compute_buffer_size + grow_ahead,
                                                               512ULL * 1024ULL * 1024ULL));
                                }
                            }
                        }
                        const bool graph_pool_shrink_allowed =
                            IsFlexibleGraphPoolSizingEnabled(current_model) && is_decode_batch && used_flexible_sizing;
                        const bool graph_pool_oversized =
                            graph_pool_shrink_allowed && state->inference_ctx.IsInitialized() &&
                            state->inference_ctx.compute_buffer_size > ctx_size * 2 &&
                            (state->inference_ctx.compute_buffer_size - ctx_size) > (2048ULL * 1024ULL * 1024ULL);
                        const size_t flexible_dry_context_floor =
                            used_flexible_sizing
                                ? AlignUpBytes(flexible_sizing.dry_context_bytes + 512ULL * 1024ULL * 1024ULL,
                                               64ULL * 1024ULL * 1024ULL)
                                : 0;
                        const auto fail_graph_ctx_requests = [&](const std::string& fail_reason,
                                                                 const char* error_message) {
                            for (Request* req : batch_requests) {
                                if (!req || req->finished) {
                                    continue;
                                }
                                req->graph_ctx_fail_reason = fail_reason;
                                req->finished = true;
                                req->decode_finish_cause = DecodeFinishCause::OutOfMemory;
                                FinalizeDecodeSilentFinishReason(req);
                                LogRequestDecodeSummary(req, current_model);
                                state->metrics.oom_errors++;
                                state->metrics.failed_requests++;
                                EmitRequestResult(state, req, error_message, -1, true, true, global_direct_callback);
                                if (!req->block_table.empty() && current_kv_cache && current_kv_cache->block_manager) {
                                    current_kv_cache->block_manager->Free(req->block_table);
                                    req->block_table.clear();
                                }
                                if (req->seq_id >= 0) {
                                    state->scheduler->RemoveRequest(req->seq_id, false);
                                    seq_to_request.erase(req->seq_id);
                                }
                                std::lock_guard<std::mutex> lk(req->mu);
                                req->cv.notify_all();
                            }
                        };
                        if (!state->inference_ctx.IsInitialized() ||
                            state->inference_ctx.compute_buffer_size < ctx_size || graph_pool_oversized) {
                            if (state->inference_ctx.IsInitialized() &&
                                (state->inference_ctx.compute_buffer_size < ctx_size || graph_pool_oversized)) {
                                state->inference_ctx.Free();
                                if (graph_pool_oversized && flexible_dry_context_floor > ctx_size) {
                                    const size_t available_after_free = ReadAvailableMemoryBytesForRuntimePools();
                                    constexpr size_t kPostShrinkReserveBytes = 128ULL * 1024ULL * 1024ULL;
                                    if (available_after_free == 0 ||
                                        flexible_dry_context_floor + kPostShrinkReserveBytes <= available_after_free) {
                                        ctx_size = flexible_dry_context_floor;
                                    }
                                }
                            }
                            if (is_prefill_batch && graph_ctx_estimate.effective_query_len > 1 &&
                                ctx_size >= 1024ULL * 1024ULL * 1024ULL) {
                                const auto q4k_cache_before = densecore::kernels::Q4KRepackedGemvCacheStatsSnapshot();
                                const size_t desired_headroom =
                                    std::max<size_t>(1024ULL * 1024ULL * 1024ULL, ctx_size / 4);
                                const size_t reserved_for_prefill = ctx_size + desired_headroom;
                                const size_t q4k_cache_limit =
                                    densecore::kernels::Q4KRepackedGemvRefreshRuntimeCacheBudget(reserved_for_prefill);
                                if (q4k_cache_before.resident_bytes > q4k_cache_limit) {
                                    const auto q4k_cache_after =
                                        densecore::kernels::Q4KRepackedGemvTrimCacheToBytes(q4k_cache_limit);
                                    if (q4k_cache_after.resident_bytes < q4k_cache_before.resident_bytes) {
                                        std::cerr << "[DenseCore] Q4KRepackedCacheTrim before_large_prefill"
                                                  << " before_mb=" << (q4k_cache_before.resident_bytes / (1024 * 1024))
                                                  << " after_mb=" << (q4k_cache_after.resident_bytes / (1024 * 1024))
                                                  << " ctx_mb=" << (ctx_size / (1024 * 1024))
                                                  << " reserve_mb=" << (reserved_for_prefill / (1024 * 1024))
                                                  << " cache_limit_mb=" << (q4k_cache_limit / (1024 * 1024))
                                                  << std::endl;
                                    }
                                } else {
                                    (void)q4k_cache_limit;
                                }
                            }
                            const auto descriptor = densecore::models::DescribeModel(current_model);
                            std::cerr << "[DenseCore] GraphCtxEstimate variant="
                                      << densecore::models::ModelVariantName(descriptor.variant)
                                      << " seq_hint=" << graph_ctx_estimate.effective_seq_len
                                      << " query_tokens=" << graph_ctx_estimate.effective_query_len
                                      << " num_seqs_hint=" << graph_ctx_estimate.effective_num_seqs
                                      << " chunk_tokens=" << graph_ctx_estimate.chunk_token_hint << " base_mb="
                                      << (graph_ctx_estimate.base_graph_working_set_bytes / (1024 * 1024))
                                      << " hybrid_mb=" << (graph_ctx_estimate.hybrid_ssm_extra_bytes / (1024 * 1024))
                                      << " safety_mb="
                                      << (graph_ctx_estimate.long_context_safety_pad_bytes / (1024 * 1024))
                                      << " env_extra_mb=" << (graph_ctx_estimate.env_extra_bytes / (1024 * 1024))
                                      << " total_mb=" << (ctx_size / (1024 * 1024));
                            if (used_flexible_sizing) {
                                std::cerr
                                    << " flexible=1 dry_ctx_mb=" << (flexible_sizing.dry_context_bytes / (1024 * 1024))
                                    << " dry_meta_mb=" << (flexible_sizing.dry_metadata_bytes / (1024 * 1024))
                                    << " graph_tensor_mb=" << (flexible_sizing.graph_tensor_bytes / (1024 * 1024))
                                    << " margin_mb=" << (flexible_sizing.margin_bytes / (1024 * 1024))
                                    << " required_mb=" << (flexible_sizing.required_bytes / (1024 * 1024))
                                    << " reserved_mb=" << (ctx_size / (1024 * 1024))
                                    << " available_mb=" << (flexible_sizing.available_bytes / (1024 * 1024))
                                    << " reservation_payload_mb="
                                    << (flexible_sizing.reservation_payload_bytes / (1024 * 1024))
                                    << " reservation_slack_mb="
                                    << (flexible_sizing.reservation_slack_bytes / (1024 * 1024))
                                    << " graph_nodes=" << flexible_sizing.graph_nodes
                                    << " shrink=" << (graph_pool_oversized ? 1 : 0);
                            }
                            std::cerr << std::endl;
                            const size_t graph_ctx_available_bytes = ReadAvailableMemoryBytesForRuntimePools();
                            const size_t graph_ctx_safety_margin_bytes = GraphContextSafetyMarginBytes();
                            for (Request* req : batch_requests) {
                                if (!req) {
                                    continue;
                                }
                                req->graph_ctx_requested_mb = ctx_size / (1024ULL * 1024ULL);
                                req->graph_ctx_available_mb = graph_ctx_available_bytes / (1024ULL * 1024ULL);
                                req->graph_ctx_safety_margin_mb = graph_ctx_safety_margin_bytes / (1024ULL * 1024ULL);
                            }
                            const bool graph_ctx_pressure =
                                graph_ctx_available_bytes > 0 &&
                                ctx_size + graph_ctx_safety_margin_bytes > graph_ctx_available_bytes;
                            if (graph_ctx_pressure) {
                                const std::string pressure_reason =
                                    is_prefill_batch ? "prefill_graph_ctx_pressure_after_chunk_downgrade"
                                                     : "graph_ctx_pressure";
                                std::cerr
                                    << "[DenseCore] GraphCtxPressure"
                                    << " requested_mb=" << (ctx_size / (1024ULL * 1024ULL))
                                    << " available_mb=" << (graph_ctx_available_bytes / (1024ULL * 1024ULL))
                                    << " safety_margin_mb=" << (graph_ctx_safety_margin_bytes / (1024ULL * 1024ULL))
                                    << " reason=" << pressure_reason
                                    << " fail_closed=" << (IsGraphContextFailClosedEnabled() ? 1 : 0) << std::endl;
                                if (IsGraphContextFailClosedEnabled()) {
                                    const std::string fail_reason =
                                        is_prefill_batch ? "prefill_graph_ctx_exceeds_available_after_chunk_downgrade"
                                                         : "graph_ctx_exceeds_available";
                                    fail_graph_ctx_requests(
                                        fail_reason, "Error: graph context reservation exceeds safe available memory");
                                    continue;
                                }
                            }
                            try {
                                state->inference_ctx.Init(ctx_size);
                                densecore::kernels::Q4KRepackedGemvRefreshRuntimeCacheBudget(0);
                            } catch (const densecore::OutOfMemoryException& e) {
                                std::cerr
                                    << "[DenseCore] GraphCtxAllocationFailed"
                                    << " requested_mb=" << (ctx_size / (1024ULL * 1024ULL))
                                    << " available_mb=" << (graph_ctx_available_bytes / (1024ULL * 1024ULL))
                                    << " safety_margin_mb=" << (graph_ctx_safety_margin_bytes / (1024ULL * 1024ULL))
                                    << " error=" << e.what() << std::endl;
                                fail_graph_ctx_requests("graph_ctx_allocation_failed",
                                                        "Error: graph context allocation failed");
                                continue;
                            }
                        }

                        const size_t steady_available_bytes = ReadAvailableMemoryBytesForRuntimePools();
                        const size_t steady_safety_margin_bytes = GraphContextSafetyMarginBytes();
                        if (is_prefill_batch && steady_available_bytes > 0 &&
                            steady_available_bytes < steady_safety_margin_bytes && IsGraphContextFailClosedEnabled()) {
                            for (Request* req : batch_requests) {
                                if (!req) {
                                    continue;
                                }
                                req->graph_ctx_requested_mb = ctx_size / (1024ULL * 1024ULL);
                                req->graph_ctx_available_mb = steady_available_bytes / (1024ULL * 1024ULL);
                                req->graph_ctx_safety_margin_mb = steady_safety_margin_bytes / (1024ULL * 1024ULL);
                            }
                            std::cerr << "[DenseCore] GraphCtxPressure"
                                      << " requested_mb=" << (ctx_size / (1024ULL * 1024ULL))
                                      << " available_mb=" << (steady_available_bytes / (1024ULL * 1024ULL))
                                      << " safety_margin_mb=" << (steady_safety_margin_bytes / (1024ULL * 1024ULL))
                                      << " reason=prefill_graph_ctx_headroom_exhausted fail_closed=1" << std::endl;
                            fail_graph_ctx_requests("prefill_graph_ctx_headroom_exhausted",
                                                    "Error: graph context headroom exhausted");
                            continue;
                        }

                        if (Request* trace_req = !batch_requests.empty() ? batch_requests[0] : nullptr) {
                            LogPrefillStage("before_prefill_graph_build", trace_req, current_model,
                                            &state->inference_ctx, nullptr, 0, nullptr, is_prefill_batch,
                                            active_threads);
                        }
                        LogRequestLifecyclePhase("graph_build_begin",
                                                 !batch_requests.empty() ? batch_requests.front() : nullptr,
                                                 current_model,
                                                 !batch_requests.empty() && batch_requests.front()
                                                     ? batch_requests.front()->seq_id
                                                     : -1,
                                                 batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads);

                        // Reset persistent context (O(1) - reuses existing memory buffer)
                        // This prepares a fresh context for graph building without malloc/free
                        state->inference_ctx.Reset();
                        struct ggml_context* ctx_nodes = state->inference_ctx.GetContext();

                        if (!ctx_nodes) {
                            LOG_CRITICAL("FATAL: InferenceContext not initialized!");
                            continue;
                        }

                        gf = ggml_new_graph_custom(ctx_nodes, 32768, false);
                        bind_main_work_context_for_graph(batch, batch_execution_phase);
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
                            LogPrefillStage("after_prefill_graph_build", trace_req, current_model,
                                            &state->inference_ctx, nullptr, 0, output, is_prefill_batch,
                                            active_threads);
                        }
                        LogRequestLifecyclePhase("graph_build_end",
                                                 !batch_requests.empty() ? batch_requests.front() : nullptr,
                                                 current_model,
                                                 !batch_requests.empty() && batch_requests.front()
                                                     ? batch_requests.front()->seq_id
                                                     : -1,
                                                 batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads);
                    }
                }
            }

            if (!output || !embd_inp || !pos) {
                LOG_ERROR("Fatal: Content creation failed or tensors missing");
                LogRequestLifecyclePhase("graph_build_failed",
                                         !batch_requests.empty() ? batch_requests.front() : nullptr, current_model,
                                         !batch_requests.empty() && batch_requests.front() ? batch_requests.front()->seq_id
                                                                                           : -1,
                                         batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads);
                continue;  // Recover
            }
            LogRequestLifecyclePhase(reused_decode_graph || reused_prefill_graph ? "graph_reuse_ready" : "graph_ready",
                                     !batch_requests.empty() ? batch_requests.front() : nullptr, current_model,
                                     !batch_requests.empty() && batch_requests.front() ? batch_requests.front()->seq_id
                                                                                       : -1,
                                     batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads);
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
                    req->graph_rebind_ns += graph_runtime_rebind_ns;
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

            char* input_base = nullptr;
            size_t embd_size = ggml_nbytes(embd_inp);
            const size_t pos_size = ggml_nbytes(pos);
            size_t input_span_bytes = embd_size + 256 + pos_size;

            if (using_transient_graph_arena) {
                if (!embd_inp->data || !pos->data) {
                    throw densecore::OutOfMemoryException("Gemma4 prefill graph arena did not allocate input tensors");
                }
                input_base = static_cast<char*>(embd_inp->data);
                input_span_bytes = embd_size;
            } else {
                // Use offset 0 of compute buffer for inputs
                input_base = state->compute_buffer.get();
                embd_inp->data = input_base;
                pos->data = input_base + embd_size + 256;  // alignment padding
            }

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
            if (!main_work_context_bound_for_current_graph) {
                bind_main_work_context_for_graph(batch, batch_execution_phase);
            } else {
                SetCurrentExecutionPhase(batch_execution_phase);
                SetCurrentBatch(&batch);
            }

            // Note: Memory ordering for GGML worker threads is handled internally
            // by GGML's thread pool (uses CV/mutex). No explicit fence needed.

            // Keep backend thread count and custom-op n_tasks synchronized.
            maybe_set_cpu_threads();

            static int decode_graph_regression_checked_steps = 0;
            const bool decode_graph_regression_single_seq_hybrid =
                batch.num_seqs == 1 && current_model && current_model->arch_flags.is_hybrid_ssm &&
                densecore::models::DescribeModel(current_model).variant == ModelVariant::QWEN36;
            const bool decode_graph_regression_single_seq_lfm2 =
                batch.num_seqs == 1 && current_model && current_model->arch_flags.is_lfm2_shortconv;
            const bool decode_graph_regression_single_seq_gemma4 =
                batch.num_seqs == 1 && current_model && current_model->arch_flags.is_gemma4;
            const bool run_decode_graph_cache_regression_check =
                cpu_backend_active && current_kv_cache && using_cached_decode_graph && decode_single_token_layout &&
                (batch.num_seqs > 1 || decode_graph_regression_single_seq_hybrid ||
                 decode_graph_regression_single_seq_lfm2 || decode_graph_regression_single_seq_gemma4) &&
                batch.lora_map.empty() && IsDecodeGraphCacheRegressionEnabled() &&
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
            std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>> decode_check_pre_ssm;
            bool decode_check_ready = false;
            bool decode_check_ssm_ready = false;

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

                if (ModelRequiresPrefixStateSnapshot(current_model)) {
                    decode_check_ssm_ready =
                        batch.hybrid_ssm_runtime_states.size() == static_cast<size_t>(batch.num_seqs);
                    if (decode_check_ssm_ready) {
                        decode_check_pre_ssm.reserve(batch.hybrid_ssm_runtime_states.size());
                        for (auto* states : batch.hybrid_ssm_runtime_states) {
                            if (!states) {
                                decode_check_ssm_ready = false;
                                decode_check_pre_ssm.clear();
                                break;
                            }
                            decode_check_pre_ssm.push_back(*states);
                        }
                    }
                } else {
                    decode_check_ssm_ready = true;
                }
            }
            const auto restore_decode_check_state = [&]() {
                current_kv_cache->RestoreBlocksFromHost(decode_check_write_blocks, decode_check_pre_k,
                                                        decode_check_pre_v);
                if (!decode_check_pre_ssm.empty() &&
                    decode_check_pre_ssm.size() == batch.hybrid_ssm_runtime_states.size()) {
                    for (size_t i = 0; i < decode_check_pre_ssm.size(); ++i) {
                        if (batch.hybrid_ssm_runtime_states[i]) {
                            *batch.hybrid_ssm_runtime_states[i] = decode_check_pre_ssm[i];
                        }
                    }
                }
                SetCurrentWorkContext(work_ctx.get());
                SetCurrentExecutionPhase(is_prefill_batch ? InferenceExecutionPhase::Prefill
                                                          : InferenceExecutionPhase::Decode);
                SetCurrentBatch(&batch);
            };

            if (run_decode_graph_cache_regression_check) {
                if (!decode_check_ready) {
                    std::cerr << "[DecodeGraphCacheCheck] skipped: unable to snapshot decode-step KV pre-state"
                              << std::endl;
                } else if (!decode_check_ssm_ready) {
                    std::cerr << "[DecodeGraphCacheCheck] skipped: unable to snapshot hybrid SSM pre-state"
                              << std::endl;
                } else {
                    try {
                        // Both cached and uncached verification runs write decode-step
                        // KV and hybrid-SSM recurrent state, so each must start from
                        // the same pre-step state. Restore again before the real decode
                        // compute so this regression mode does not perturb generation.
                        MaybeLogMoEGraphSummary(gf, current_model, is_prefill_batch);
                        ResetPagedDecodeGraphExecutionState(gf);
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

                                restore_decode_check_state();

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
                                        auto uncached_work_ctx =
                                            std::unique_ptr<InferenceWorkContext,
                                                            decltype(&DestroyInferenceWorkContext)>(
                                                CreateInferenceWorkContext(), DestroyInferenceWorkContext);
                                        ScopedBatchWorkContext uncached_scope(uncached_work_ctx.get(), &batch,
                                                                              batch_execution_phase);
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
                                            ResetPagedDecodeGraphExecutionState(uncached_gf);
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
                        restore_decode_check_state();
                        throw;
                    }
                    restore_decode_check_state();
                }
            }

            const densecore::kernels::Q4KRepackedGemvCacheStats q4k_cache_before =
                densecore::kernels::Q4KRepackedGemvCacheStatsSnapshot();
            const auto compute_begin = std::chrono::steady_clock::now();
            ValidateMulNodesOrThrow(gf, is_decode_batch ? "decode" : "prefill");
            MaybeLogMoEGraphSummary(gf, current_model, is_prefill_batch);
            DebugDumpGraphNodes(gf, is_decode_batch ? "decode" : "prefill");
            ResetMoEStrictFailure();
            ResetPagedDecodeGraphExecutionState(gf);
            LogRequestLifecyclePhase("graph_compute_begin", !batch_requests.empty() ? batch_requests.front() : nullptr,
                                     current_model,
                                     !batch_requests.empty() && batch_requests.front() ? batch_requests.front()->seq_id
                                                                                       : -1,
                                     batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads,
                                     is_decode_batch ? "decode" : "prefill");
            ggml_backend_graph_compute(active_backend, gf);
            const auto compute_end = std::chrono::steady_clock::now();
            LogRequestLifecyclePhase("graph_compute_end", !batch_requests.empty() ? batch_requests.front() : nullptr,
                                     current_model,
                                     !batch_requests.empty() && batch_requests.front() ? batch_requests.front()->seq_id
                                                                                       : -1,
                                     batch.num_seqs, static_cast<int>(batch.tokens.size()), active_threads,
                                     is_decode_batch ? "decode" : "prefill");
            AccumulateQwen36SSMProjectionNodeTimes(work_ctx.get(), gf);
            const Qwen36PrefillBreakdown qwen36_prefill_breakdown =
                is_prefill_batch ? SummarizeQwen36PrefillNodeTimes(current_model, gf) : Qwen36PrefillBreakdown{};
            const Gemma4PrefillAttentionTimingBreakdown gemma4_prefill_attention_breakdown =
                is_prefill_batch ? SummarizeGemma4PrefillAttentionNodeTimes(current_model, gf)
                                 : Gemma4PrefillAttentionTimingBreakdown{};
            const NativeMoEGraphTimingBreakdown native_moe_graph_timing =
                SummarizeNativeQwenMoEGraphNodeTimes(current_model, gf);
            const HybridSSMGraphTimingBreakdown hybrid_ssm_graph_timing =
                is_decode_batch ? SummarizeHybridSSMGraphNodeTimes(current_model, gf) : HybridSSMGraphTimingBreakdown{};
            // Always collect the cheap op-bucket totals for decode so the
            // graph-execute breakdown (custom/mul_mat/mul_mat_id/norm/attention/
            // other) is visible in every profiling run; only the per-node census
            // (top_slow_nodes) stays behind the debug-dump flag.
            const bool collect_decode_graph_node_timing = is_decode_batch;
            const bool collect_decode_graph_top_slow_nodes = is_decode_batch && IsLLMNodeTimingDumpEnabled();
            DecodeGraphNodeTimingBreakdown decode_graph_node_timing =
                collect_decode_graph_node_timing
                    ? SummarizeDecodeGraphNodeTimes(gf, collect_decode_graph_top_slow_nodes)
                    : DecodeGraphNodeTimingBreakdown{};
            DebugDumpLLMNodeTimes(current_model, gf, is_decode_batch ? "decode" : "prefill");
            DebugDumpGemma4NodeTimes(current_model, gf, is_decode_batch ? "decode" : "prefill");
            const auto graph_execute_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(compute_end - compute_begin).count());
            const Qwen36ProfileSnapshot qwen36_profile = GetQwen36ProfileSnapshot(work_ctx.get());
            if (collect_decode_graph_node_timing) {
                SynthesizeDecodeGraphNodeTimingFromProfiles(&decode_graph_node_timing, graph_execute_ns,
                                                            native_moe_graph_timing, hybrid_ssm_graph_timing,
                                                            qwen36_profile);
            }
            const densecore::kernels::Q4KRepackedGemvCacheStats q4k_cache_after =
                densecore::kernels::Q4KRepackedGemvCacheStatsSnapshot();
            const uint64_t q4k_cache_evictions_delta = q4k_cache_after.evictions >= q4k_cache_before.evictions
                                                           ? q4k_cache_after.evictions - q4k_cache_before.evictions
                                                           : 0;
            const uint64_t q4k_cache_evicted_bytes_delta =
                q4k_cache_after.evicted_bytes >= q4k_cache_before.evicted_bytes
                    ? q4k_cache_after.evicted_bytes - q4k_cache_before.evicted_bytes
                    : 0;
            const uint64_t q4k_repack_bytes_delta = q4k_cache_after.repack_bytes >= q4k_cache_before.repack_bytes
                                                        ? q4k_cache_after.repack_bytes - q4k_cache_before.repack_bytes
                                                        : 0;
            const uint64_t qwen36_prefill_graph_build_ns =
                (is_prefill_batch && graph_build_begin != std::chrono::steady_clock::time_point() &&
                 graph_build_end != std::chrono::steady_clock::time_point() && graph_build_end >= graph_build_begin)
                    ? static_cast<uint64_t>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(graph_build_end - graph_build_begin)
                              .count())
                    : 0;
            for (Request* req : batch_requests) {
                if (req) {
                    req->graph_execute_ns += graph_execute_ns;
                    if (!is_prefill_batch) {
                        req->decode_graph_execute_ns += graph_execute_ns;
                    }
                    if (collect_decode_graph_node_timing) {
                        req->decode_graph_node_measured_ns += decode_graph_node_timing.measured_ns;
                        req->decode_graph_node_custom_ns += decode_graph_node_timing.custom_ns;
                        req->decode_graph_node_mul_mat_ns += decode_graph_node_timing.mul_mat_ns;
                        req->decode_graph_node_mul_mat_id_ns += decode_graph_node_timing.mul_mat_id_ns;
                        req->decode_graph_node_norm_ns += decode_graph_node_timing.norm_ns;
                        req->decode_graph_node_view_copy_ns += decode_graph_node_timing.view_copy_ns;
                        req->decode_graph_node_elementwise_ns += decode_graph_node_timing.elementwise_ns;
                        req->decode_graph_node_attention_ns += decode_graph_node_timing.attention_ns;
                        req->decode_graph_node_other_ns += decode_graph_node_timing.other_ns;
                        req->decode_graph_node_custom_count += decode_graph_node_timing.custom_count;
                        req->decode_graph_node_mul_mat_count += decode_graph_node_timing.mul_mat_count;
                        req->decode_graph_node_mul_mat_id_count += decode_graph_node_timing.mul_mat_id_count;
                        req->decode_graph_node_norm_count += decode_graph_node_timing.norm_count;
                        req->decode_graph_node_view_copy_count += decode_graph_node_timing.view_copy_count;
                        req->decode_graph_node_elementwise_count += decode_graph_node_timing.elementwise_count;
                        req->decode_graph_node_attention_count += decode_graph_node_timing.attention_count;
                        req->decode_graph_node_other_count += decode_graph_node_timing.other_count;
                        req->decode_graph_top_slow_nodes.insert(req->decode_graph_top_slow_nodes.end(),
                                                                decode_graph_node_timing.top_slow_nodes.begin(),
                                                                decode_graph_node_timing.top_slow_nodes.end());
                        std::sort(req->decode_graph_top_slow_nodes.begin(), req->decode_graph_top_slow_nodes.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name;
                                  });
                        if (req->decode_graph_top_slow_nodes.size() > kMatmulTopShapeCount) {
                            req->decode_graph_top_slow_nodes.resize(kMatmulTopShapeCount);
                        }
                    }
                    if (is_prefill_batch && IsQwenHybridSSMModel(current_model)) {
                        req->qwen36_prefill_graph_build_ns += qwen36_prefill_graph_build_ns;
                        req->qwen36_prefill_graph_execute_ns += graph_execute_ns;
                        req->qwen36_prefill_total_ns += qwen36_prefill_graph_build_ns + graph_execute_ns;
                        req->qwen36_prefill_ssm_projection_ns += qwen36_prefill_breakdown.ssm_projection_ns;
                        req->qwen36_prefill_ssm_delta_state_ns += qwen36_prefill_breakdown.ssm_delta_state_ns;
                        req->qwen36_prefill_attention_ns += qwen36_prefill_breakdown.attention_ns;
                        req->qwen36_prefill_mlp_or_moe_ns += qwen36_prefill_breakdown.mlp_or_moe_ns;
                        auto merge_prefill_slow_entries = [](std::vector<MatmulShapeCensusEntry>& dst,
                                                             const std::vector<MatmulShapeCensusEntry>& src) {
                            for (const auto& entry : src) {
                                dst.push_back(entry);
                            }
                            std::sort(dst.begin(), dst.end(), [](const auto& a, const auto& b) {
                                return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name;
                            });
                            if (dst.size() > kMatmulTopShapeCount) {
                                dst.resize(kMatmulTopShapeCount);
                            }
                        };
                        merge_prefill_slow_entries(req->qwen36_prefill_top_slow_ops,
                                                   qwen36_prefill_breakdown.top_slow_ops);
                    }
                    if (is_prefill_batch && current_model && current_model->variant == ModelVariant::GEMMA4) {
                        req->attention_ns += gemma4_prefill_attention_breakdown.attention_ns;
                        req->portable_flash_attention_ns += gemma4_prefill_attention_breakdown.portable_flash_ns;
                        req->hal_attention_ns += gemma4_prefill_attention_breakdown.hal_ns;
                        req->gemma4_prefill_graph_build_ns += qwen36_prefill_graph_build_ns;
                        req->gemma4_prefill_graph_execute_ns += graph_execute_ns;
                        req->gemma4_prefill_total_ns += qwen36_prefill_graph_build_ns + graph_execute_ns;
                        req->gemma4_prefill_attention_ns += gemma4_prefill_attention_breakdown.attention_ns;
                        req->gemma4_prefill_moe_or_mlp_ns += gemma4_prefill_attention_breakdown.moe_or_mlp_ns;
                        req->gemma4_prefill_mul_mat_id_ns += gemma4_prefill_attention_breakdown.mul_mat_id_ns;
                        req->gemma4_prefill_mul_mat_ns += gemma4_prefill_attention_breakdown.mul_mat_ns;
                        req->gemma4_prefill_flash_attention_ns += gemma4_prefill_attention_breakdown.portable_flash_ns;
                        for (auto entry : gemma4_prefill_attention_breakdown.top_slow_ops) {
                            entry.active_threads = req->active_thread_count;
                            req->gemma4_prefill_top_slow_ops.push_back(std::move(entry));
                        }
                        std::sort(req->gemma4_prefill_top_slow_ops.begin(), req->gemma4_prefill_top_slow_ops.end(),
                                  [](const auto& a, const auto& b) {
                                      if (a.wall_ns != b.wall_ns) {
                                          return a.wall_ns > b.wall_ns;
                                      }
                                      return a.left_name < b.left_name;
                                  });
                        constexpr std::size_t kGemma4PrefillTopSlowCount = 200;
                        if (req->gemma4_prefill_top_slow_ops.size() > kGemma4PrefillTopSlowCount) {
                            req->gemma4_prefill_top_slow_ops.resize(kGemma4PrefillTopSlowCount);
                        }
                    }
                    if (native_moe_graph_timing.native_node_count > 0) {
                        req->native_moe_graph_ns += native_moe_graph_timing.total_ns;
                        if (!is_prefill_batch) {
                            req->decode_native_moe_graph_ns += native_moe_graph_timing.total_ns;
                            req->decode_moe_route_ns += native_moe_graph_timing.route_ns;
                            req->decode_moe_w1w3_ns += native_moe_graph_timing.w1w3_ns;
                            req->decode_moe_w2_ns += native_moe_graph_timing.w2_ns;
                            req->decode_moe_reduce_ns += native_moe_graph_timing.reduce_ns;
                            const uint64_t fast_w2_count =
                                std::min(native_moe_graph_timing.w2_fast_count, native_moe_graph_timing.w2_count);
                            const uint64_t fast_w2_ns =
                                std::min(native_moe_graph_timing.w2_fast_ns, native_moe_graph_timing.w2_ns);
                            const uint64_t fast_w1w3_count =
                                std::min(native_moe_graph_timing.w1w3_fast_count, native_moe_graph_timing.w1w3_count);
                            const uint64_t fast_w1w3_ns =
                                std::min(native_moe_graph_timing.w1w3_fast_ns, native_moe_graph_timing.w1w3_ns);
                            const uint64_t fallback_w1w3_count = native_moe_graph_timing.w1w3_count - fast_w1w3_count;
                            const uint64_t fallback_w1w3_ns = native_moe_graph_timing.w1w3_ns - fast_w1w3_ns;
                            const uint64_t fallback_w2_count = native_moe_graph_timing.w2_count - fast_w2_count;
                            const uint64_t fallback_w2_ns = native_moe_graph_timing.w2_ns - fast_w2_ns;
                            req->native_moe_fallback_w1w3_ns += fallback_w1w3_ns;
                            req->native_moe_fallback_w2_ns += fallback_w2_ns;
                            req->native_moe_fallback_w1w3_ops += fallback_w1w3_count;
                            req->native_moe_fallback_w2_ops += fallback_w2_count;
                            req->native_moe_fallback_ops += fallback_w1w3_count + fallback_w2_count;
                            if (fast_w1w3_count > 0) {
                                req->native_moe_fast_decode_candidate_ops += fast_w1w3_count;
                                req->native_moe_fast_decode_used_ops += fast_w1w3_count;
                                req->native_moe_fast_decode_w1w3_used_ops += fast_w1w3_count;
                                req->native_moe_fast_decode_ns += fast_w1w3_ns;
                                req->native_moe_fast_w1w3_ns += fast_w1w3_ns;
                                req->native_moe_fast_total_ns += fast_w1w3_ns;
                                req->native_moe_fast_w1w3_used_ops += fast_w1w3_count;
                                req->native_moe_fast_replaced_fallback_ops += fast_w1w3_count;
                            }
                            if (fast_w2_count > 0) {
                                req->native_moe_fast_decode_candidate_ops += fast_w2_count;
                                req->native_moe_fast_decode_used_ops += fast_w2_count;
                                req->native_moe_fast_decode_w2_used_ops += fast_w2_count;
                                req->native_moe_fast_decode_ns += fast_w2_ns;
                                req->native_moe_fast_w2_ns += fast_w2_ns;
                                req->native_moe_fast_total_ns += fast_w2_ns;
                                req->native_moe_fast_w2_used_ops += fast_w2_count;
                                req->native_moe_fast_w2_q5k_candidate_ops += fast_w2_count;
                                req->native_moe_fast_w2_q5k_used_ops += fast_w2_count;
                                req->native_moe_fast_w2_q5k_ns += fast_w2_ns;
                                req->native_moe_fast_replaced_fallback_ops += fast_w2_count;
                            }
                            const auto descriptor = densecore::models::DescribeModel(current_model);
                            if (descriptor.variant == ModelVariant::QWEN35 ||
                                descriptor.variant == ModelVariant::QWEN36) {
                                if (fast_w2_count == 0) {
                                    req->native_moe_fast_decode_candidate_ops += 1;
                                    req->native_moe_fast_decode_rejected_ops += 1;
                                    req->native_moe_fast_decode_last_reject_reason = "native_moe_w2_fast_node_missing";
                                }
                                if (fallback_w2_count > 0) {
                                    req->native_moe_fast_w2_q5k_candidate_ops += 1;
                                    req->native_moe_fast_w2_q5k_rejected_ops += 1;
                                    req->native_moe_fast_w2_q5k_last_reject_reason = "native_moe_w2_fast_node_missing";
                                }
                            }
                        } else if (current_model && current_model->variant == ModelVariant::QWEN36) {
                            const uint64_t fast_w2_count =
                                std::min(native_moe_graph_timing.w2_fast_count, native_moe_graph_timing.w2_count);
                            const uint64_t fast_w1w3_count =
                                std::min(native_moe_graph_timing.w1w3_fast_count, native_moe_graph_timing.w1w3_count);
                            const uint64_t candidate_count =
                                native_moe_graph_timing.w1w3_count + native_moe_graph_timing.w2_count;
                            const uint64_t used_count = fast_w1w3_count + fast_w2_count;
                            req->qwen36_prefill_native_moe_fast_candidate_ops += candidate_count;
                            req->qwen36_prefill_native_moe_fast_used_ops += used_count;
                            if (candidate_count > used_count) {
                                req->qwen36_prefill_native_moe_fast_rejected_ops += candidate_count - used_count;
                                req->qwen36_prefill_native_moe_fast_last_reject_reason = "native_moe_node_not_fast";
                            }
                        }
                        req->moe_route_ns += native_moe_graph_timing.route_ns;
                        req->moe_w1w3_ns += native_moe_graph_timing.w1w3_ns;
                        req->moe_w2_ns += native_moe_graph_timing.w2_ns;
                        req->moe_reduce_ns += native_moe_graph_timing.reduce_ns;
                        req->moe_expert_ns += native_moe_graph_timing.w1w3_ns + native_moe_graph_timing.activation_ns +
                                              native_moe_graph_timing.w2_ns;
                        req->native_moe_graph_node_hist = native_moe_graph_timing.node_hist;
                        req->native_moe_graph_top_slow_nodes.insert(req->native_moe_graph_top_slow_nodes.end(),
                                                                    native_moe_graph_timing.top_slow_nodes.begin(),
                                                                    native_moe_graph_timing.top_slow_nodes.end());
                        std::sort(req->native_moe_graph_top_slow_nodes.begin(),
                                  req->native_moe_graph_top_slow_nodes.end(), [](const auto& a, const auto& b) {
                                      return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name;
                                  });
                        if (req->native_moe_graph_top_slow_nodes.size() > kMatmulTopShapeCount) {
                            req->native_moe_graph_top_slow_nodes.resize(kMatmulTopShapeCount);
                        }
                        if (native_moe_graph_timing.total_ns == 0) {
                            req->native_moe_timing_missing = 1;
                        }
                    }
                    req->attention_ns += qwen36_profile.attention_ns;
                    req->paged_attention_ns += qwen36_profile.paged_attention_ns;
                    req->standard_attention_ns += qwen36_profile.standard_attention_ns;
                    req->portable_flash_attention_ns += qwen36_profile.portable_flash_attention_ns;
                    req->native_flash_attention_ns += qwen36_profile.native_flash_attention_ns;
                    req->hal_attention_ns += qwen36_profile.hal_attention_ns;
                    req->attention_repack_ns += qwen36_profile.attention_repack_ns;
                    req->moe_forward_ns += qwen36_profile.moe_forward_ns;
                    req->moe_route_ns += qwen36_profile.moe_route_ns;
                    req->moe_reorder_ns += qwen36_profile.moe_reorder_ns;
                    req->moe_expert_ns += qwen36_profile.moe_expert_ns;
                    req->moe_reduce_ns += qwen36_profile.moe_reduce_ns;
                    req->moe_w1w3_ns += qwen36_profile.moe_w1w3_ns;
                    req->moe_w2_ns += qwen36_profile.moe_w2_ns;
                    req->moe_rowblock_ns += qwen36_profile.moe_rowblock_ns;
                    req->moe_rowblock_w1w3_ns += qwen36_profile.moe_rowblock_w1w3_ns;
                    req->moe_rowblock_w2_ns += qwen36_profile.moe_rowblock_w2_ns;
                    req->shared_expert_ns += qwen36_profile.shared_expert_ns;
                    req->quant_matmul_ns += qwen36_profile.quant_matmul_ns;
                    req->ssm_qkv_wall_ns += qwen36_profile.ssm_qkv_wall_ns;
                    req->ssm_gate_wall_ns += qwen36_profile.ssm_gate_wall_ns;
                    req->ssm_delta_wall_ns += qwen36_profile.ssm_delta_wall_ns;
                    req->ssm_out_wall_ns += qwen36_profile.ssm_out_wall_ns;
                    req->ssm_conv1d_ns += qwen36_profile.ssm_conv1d_ns;
                    req->ssm_delta_ns += qwen36_profile.ssm_delta_ns;
                    req->kv_update_ns += qwen36_profile.kv_update_ns;
                    req->sample_ns += qwen36_profile.sample_ns;
                    if (!is_prefill_batch) {
                        req->decode_attention_ns += qwen36_profile.attention_ns;
                        req->decode_paged_attention_ns += qwen36_profile.paged_attention_ns;
                        if (hybrid_ssm_graph_timing.total_ns > 0) {
                            req->decode_ssm_qkv_wall_ns += hybrid_ssm_graph_timing.qkv_ns;
                            req->decode_ssm_out_wall_ns += hybrid_ssm_graph_timing.out_ns;
                            req->decode_ssm_delta_wall_ns +=
                                hybrid_ssm_graph_timing.delta_ns + hybrid_ssm_graph_timing.alpha_beta_qk_ns;
                            req->decode_ssm_conv1d_ns += hybrid_ssm_graph_timing.conv1d_ns;
                            req->decode_ssm_delta_ns +=
                                hybrid_ssm_graph_timing.delta_ns + hybrid_ssm_graph_timing.alpha_beta_qk_ns;
                        } else {
                            req->decode_ssm_qkv_wall_ns += qwen36_profile.ssm_qkv_wall_ns;
                            req->decode_ssm_delta_wall_ns += qwen36_profile.ssm_delta_wall_ns;
                            req->decode_ssm_conv1d_ns += qwen36_profile.ssm_conv1d_ns;
                            req->decode_ssm_delta_ns += qwen36_profile.ssm_delta_ns;
                        }
                        req->decode_sample_ns += qwen36_profile.sample_ns;
                    }
                    req->kleidiai_candidate_ops += qwen36_profile.kleidiai_candidate_ops;
                    req->kleidiai_allowed_ops += qwen36_profile.kleidiai_allowed_ops;
                    req->kleidiai_rejected_ops += qwen36_profile.kleidiai_rejected_ops;
                    req->graph_cache_hit_count += qwen36_profile.graph_cache_hits;
                    req->graph_cache_miss_count += qwen36_profile.graph_cache_misses;
                    req->q4k_repacked_gemv_cache_hits += qwen36_profile.q4k_repacked_gemv_cache_hits;
                    req->q4k_repacked_gemv_cache_waited_hits += qwen36_profile.q4k_repacked_gemv_cache_waited_hits;
                    req->q4k_repacked_gemv_cache_misses += qwen36_profile.q4k_repacked_gemv_cache_misses;
                    req->q4k_repacked_gemv_cache_evictions +=
                        std::max(qwen36_profile.q4k_repacked_gemv_cache_evictions, q4k_cache_evictions_delta);
                    req->q4k_repacked_gemv_cache_evicted_bytes +=
                        std::max(qwen36_profile.q4k_repacked_gemv_cache_evicted_bytes, q4k_cache_evicted_bytes_delta);
                    req->q4k_repacked_gemv_repack_bytes +=
                        std::max(qwen36_profile.q4k_repacked_gemv_repack_bytes, q4k_repack_bytes_delta);
                    req->q4k_repacked_gemv_probe_ns += qwen36_profile.q4k_repacked_gemv_probe_ns;
                    req->q4k_repacked_gemv_resident_bytes = std::max(
                        req->q4k_repacked_gemv_resident_bytes,
                        std::max(qwen36_profile.q4k_repacked_gemv_resident_bytes, q4k_cache_after.resident_bytes));
                    req->q4k_repacked_gemv_distinct_weights_seen =
                        std::max(req->q4k_repacked_gemv_distinct_weights_seen,
                                 qwen36_profile.q4k_repacked_gemv_distinct_weights_seen);
                    req->q4k_repacked_gemv_repeated_repack_count +=
                        qwen36_profile.q4k_repacked_gemv_repeated_repack_count;
                    req->q4k_copied_gemv_experiment_cache_hits += qwen36_profile.q4k_copied_gemv_experiment_cache_hits;
                    req->q4k_copied_gemv_experiment_cache_misses +=
                        qwen36_profile.q4k_copied_gemv_experiment_cache_misses;
                    req->qact_cache_hits += qwen36_profile.qact_cache_hits;
                    req->qact_cache_misses += qwen36_profile.qact_cache_misses;
                    req->qact_cache_reused_bytes += qwen36_profile.qact_cache_reused_bytes;
                    req->moe_decode_scratch_reused += qwen36_profile.moe_decode_scratch_reused;
                    req->moe_decode_allocations_avoided += qwen36_profile.moe_decode_allocations_avoided;
                    req->q4k_true_batched_used =
                        std::max(req->q4k_true_batched_used, qwen36_profile.q4k_true_batched_used);
                    req->qwen36_prefill_q4k_batched_mode = qwen36_profile.qwen36_prefill_q4k_batched_mode;
                    req->qwen36_prefill_q4k_batched_used =
                        std::max(req->qwen36_prefill_q4k_batched_used, qwen36_profile.qwen36_prefill_q4k_batched_used);
                    req->qwen36_prefill_q4k_batched_probe_pass =
                        std::max(req->qwen36_prefill_q4k_batched_probe_pass,
                                 qwen36_profile.qwen36_prefill_q4k_batched_probe_pass);
                    req->qwen36_prefill_q4k_batched_max_abs_error =
                        std::max(req->qwen36_prefill_q4k_batched_max_abs_error,
                                 qwen36_profile.qwen36_prefill_q4k_batched_max_abs_error);
                    req->qwen36_prefill_q4k_probe_participants += qwen36_profile.qwen36_prefill_q4k_probe_participants;
                    req->qwen36_prefill_q4k_probe_failures += qwen36_profile.qwen36_prefill_q4k_probe_failures;
                    req->qwen36_prefill_q4k_admission_downgraded +=
                        qwen36_profile.qwen36_prefill_q4k_admission_downgraded;
                    if (qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason != 0) {
                        req->qwen36_prefill_q4k_batched_reject_reason = Qwen36PrefillQ4KBatchedRejectReasonName(
                            qwen36_profile.qwen36_prefill_q4k_batched_last_reject_reason);
                    } else if (is_prefill_batch && req->qwen36_prefill_q4k_batched_used == 0 &&
                               (qwen36_profile.ssm_qkv_wall_ns > 0 || qwen36_profile.ssm_gate_wall_ns > 0 ||
                                qwen36_profile.ssm_out_wall_ns > 0) &&
                               Qwen36HybridSSMProjectionWeightsAreNotQ4K(current_model)) {
                        // smart_mul_mat runs while the graph is built, usually before a
                        // worker-local profile context exists. Preserve the important
                        // admission fact in the request summary: this GGUF's Qwen3.6
                        // SSM projection weights are Q8_0, so the Q4_K prefill probe is
                        // not an eligible candidate and native GGML fallback is expected.
                        req->qwen36_prefill_q4k_batched_reject_reason = "not_q4k";
                    }
                    if (qwen36_profile.qwen36_ssm_q8_prefill_amx_mode != 0) {
                        req->qwen36_ssm_q8_prefill_amx_mode = qwen36_profile.qwen36_ssm_q8_prefill_amx_mode;
                    }
                    req->qwen36_ssm_q8_prefill_amx_prepared = std::max(
                        req->qwen36_ssm_q8_prefill_amx_prepared, qwen36_profile.qwen36_ssm_q8_prefill_amx_prepared);
                    req->qwen36_ssm_q8_prefill_amx_used =
                        std::max(req->qwen36_ssm_q8_prefill_amx_used, qwen36_profile.qwen36_ssm_q8_prefill_amx_used);
                    if (qwen36_profile.qwen36_ssm_q8_prefill_amx_last_reject_reason != 0) {
                        req->qwen36_ssm_q8_prefill_amx_reject_reason = Qwen36SSMQ8PrefillAMXRejectReasonName(
                            qwen36_profile.qwen36_ssm_q8_prefill_amx_last_reject_reason);
                    }
                    if (qwen36_profile.qwen36_ssm_q8_decode_used_original_q8_path != 0) {
                        req->qwen36_ssm_q8_decode_used_original_q8_path = 1;
                    }
                    if (qwen36_profile.qwen36_ssm_q8_prefill_amx_qkv_count != 0 ||
                        qwen36_profile.qwen36_ssm_q8_prefill_amx_gate_count != 0 ||
                        qwen36_profile.qwen36_ssm_q8_prefill_amx_out_count != 0) {
                        std::ostringstream q8_counts;
                        q8_counts << "ssm_qkv:" << qwen36_profile.qwen36_ssm_q8_prefill_amx_qkv_count
                                  << ",ssm_gate:" << qwen36_profile.qwen36_ssm_q8_prefill_amx_gate_count
                                  << ",ssm_out:" << qwen36_profile.qwen36_ssm_q8_prefill_amx_out_count;
                        req->qwen36_ssm_q8_prefill_amx_projection_counts = q8_counts.str();
                    }
                    req->qwen36_ssm_q8_prefill_amx_candidate_ops +=
                        qwen36_profile.qwen36_ssm_q8_prefill_amx_candidate_ops;
                    req->qwen36_ssm_q8_prefill_amx_used_ops += qwen36_profile.qwen36_ssm_q8_prefill_amx_used_ops;
                    req->qwen36_ssm_q8_prefill_amx_rejected_ops +=
                        qwen36_profile.qwen36_ssm_q8_prefill_amx_rejected_ops;
                    for (std::size_t i = 0; i < req->qwen36_ssm_projection_weight_type_hist.size(); ++i) {
                        req->qwen36_ssm_projection_weight_type_hist[i] +=
                            qwen36_profile.qwen36_ssm_projection_weight_type_hist[i];
                    }
                    if (IsQwen36HybridSSMModel(current_model) &&
                        Qwen36HybridSSMProjectionWeightsAreNotQ4K(current_model)) {
                        const auto& fast_config = GetFastPathRuntimeConfig();
                        req->qwen36_ssm_q8_prefill_amx_mode = static_cast<int>(fast_config.qwen36_ssm_q8_prefill_amx);
                        if (req->sampled_token_count > 1 || req->generated_count > 1) {
                            req->qwen36_ssm_q8_decode_used_original_q8_path = 1;
                        }
                    }
                    if (is_prefill_batch && req->qwen36_ssm_projection_actual_types.empty() &&
                        IsQwenHybridSSMModel(current_model)) {
                        const Qwen36SSMProjectionTypeSummary type_summary =
                            SummarizeQwen36SSMProjectionTypes(current_model);
                        req->qwen36_ssm_projection_actual_types = type_summary.actual_types;
                        req->qwen36_ssm_projection_quant_preserved = type_summary.quant_preserved;
                        req->qwen36_ssm_projection_dequantized_count = type_summary.dequantized_count;
                    }
                    req->q4k_repacked_gemv_used =
                        std::max(req->q4k_repacked_gemv_used, qwen36_profile.q4k_repacked_gemv_used);
                    req->q4k_repacked_gemv_seen_ops += qwen36_profile.q4k_repacked_gemv_seen_ops;
                    req->q4k_repacked_gemv_candidate_ops += qwen36_profile.q4k_repacked_gemv_candidate_ops;
                    req->q4k_repacked_gemv_used_ops += qwen36_profile.q4k_repacked_gemv_used_ops;
                    req->q4k_repacked_gemv_rejected_ops += qwen36_profile.q4k_repacked_gemv_rejected_ops;
                    if (qwen36_profile.q4k_repacked_gemv_last_reject_reason != 0) {
                        req->q4k_repacked_gemv_last_reject_reason = qwen36_profile.q4k_repacked_gemv_last_reject_reason;
                        req->q4k_repacked_gemv_last_reject_reason_text =
                            Q4KRepackedGemvRejectReasonName(qwen36_profile.q4k_repacked_gemv_last_reject_reason);
                    }
                    if (qwen36_profile.q4k_repacked_gemv_primary_disable_reason != 0 &&
                        req->q4k_repacked_gemv_primary_disable_reason == 0) {
                        req->q4k_repacked_gemv_primary_disable_reason =
                            qwen36_profile.q4k_repacked_gemv_primary_disable_reason;
                        req->q4k_repacked_gemv_primary_disable_reason_text =
                            Q4KRepackedGemvRejectReasonName(qwen36_profile.q4k_repacked_gemv_primary_disable_reason);
                    }
                    req->q4k_repacked_gemv_cache_thrash_detected =
                        (req->q4k_repacked_gemv_repeated_repack_count > 0 ||
                         Q4KRepackedGemvRejectReasonIsCacheThrash(req->q4k_repacked_gemv_primary_disable_reason))
                            ? 1
                            : 0;
                    if (req->q4k_repacked_gemv_primary_disable_reason != 0) {
                        req->q4k_repacked_gemv_effective_state = "disabled";
                    } else if (req->q4k_repacked_gemv_used != 0 || req->q4k_repacked_gemv_used_ops != 0) {
                        req->q4k_repacked_gemv_effective_state = "used";
                    } else if (req->q4k_repacked_gemv_rejected_ops != 0) {
                        req->q4k_repacked_gemv_effective_state = "rejected";
                    } else {
                        req->q4k_repacked_gemv_effective_state = "unused";
                    }
                    req->gemv_custom_tasks_effective =
                        std::max(req->gemv_custom_tasks_effective, qwen36_profile.gemv_custom_tasks_effective);
                    req->gemv_custom_total_ops += qwen36_profile.gemv_custom_total_ops;
                    req->gemv_custom_decode_ops += qwen36_profile.gemv_custom_decode_ops;
                    req->gemv_custom_prefill_ops += qwen36_profile.gemv_custom_prefill_ops;
                    req->gemv_custom_q4k_seen_ops += qwen36_profile.gemv_custom_q4k_seen_ops;
                    req->gemv_custom_non_q4k_ops += qwen36_profile.gemv_custom_non_q4k_ops;
                    req->gemv_custom_quant_input_null_ops += qwen36_profile.gemv_custom_quant_input_null_ops;
                    req->gemv_custom_shape_reject_ops += qwen36_profile.gemv_custom_shape_reject_ops;
                    req->gemv_custom_phase_unknown_ops += qwen36_profile.gemv_custom_phase_unknown_ops;
                    req->gemv_custom_force_reference_ops += qwen36_profile.gemv_custom_force_reference_ops;
                    req->gemv_custom_dynamic_lora_ops += qwen36_profile.gemv_custom_dynamic_lora_ops;
                    req->lfm2_decode_lm_head_custom_gemv_used_ops +=
                        qwen36_profile.lfm2_decode_lm_head_custom_gemv_used_ops;
                    req->lfm2_decode_lm_head_custom_gemv_ns +=
                        qwen36_profile.lfm2_decode_lm_head_custom_gemv_ns;
                    for (std::size_t i = 0; i < req->gemv_custom_weight_type_hist.size(); ++i) {
                        req->gemv_custom_weight_type_hist[i] += qwen36_profile.gemv_custom_weight_type_hist[i];
                    }
                    for (std::size_t i = 0; i < req->gemv_custom_quant_input_type_hist.size(); ++i) {
                        req->gemv_custom_quant_input_type_hist[i] +=
                            qwen36_profile.gemv_custom_quant_input_type_hist[i];
                    }
                    if (qwen36_profile.gemv_custom_tasks_cap_reason != 0) {
                        req->gemv_custom_tasks_cap_reason =
                            GemvCustomTaskCapReasonName(qwen36_profile.gemv_custom_tasks_cap_reason);
                    }
                    req->decode_matmul_created_ops += qwen36_profile.decode_matmul_created_ops;
                    for (std::size_t i = 0; i < req->decode_matmul_weight_type_hist.size(); ++i) {
                        req->decode_matmul_weight_type_hist[i] += qwen36_profile.decode_matmul_weight_type_hist[i];
                    }
                    for (std::size_t i = 0; i < req->decode_matmul_path_hist.size(); ++i) {
                        req->decode_matmul_path_hist[i] += qwen36_profile.decode_matmul_path_hist[i];
                    }
                    for (std::size_t i = 0; i < req->prefill_matmul_weight_type_hist.size(); ++i) {
                        req->prefill_matmul_weight_type_hist[i] += qwen36_profile.prefill_matmul_weight_type_hist[i];
                    }
                    for (std::size_t i = 0; i < req->prefill_matmul_path_hist.size(); ++i) {
                        req->prefill_matmul_path_hist[i] += qwen36_profile.prefill_matmul_path_hist[i];
                    }
                    auto merge_shape_entries = [](std::vector<MatmulShapeCensusEntry>& dst,
                                                  const std::vector<MatmulShapeCensusEntry>& src) {
                        for (const auto& entry : src) {
                            auto it = std::find_if(dst.begin(), dst.end(), [&](const MatmulShapeCensusEntry& existing) {
                                return existing.phase == entry.phase && existing.dispatch_path == entry.dispatch_path &&
                                       existing.weight_type == entry.weight_type &&
                                       existing.shape_bucket == entry.shape_bucket &&
                                       existing.left_name == entry.left_name && existing.right_name == entry.right_name;
                            });
                            if (it == dst.end()) {
                                dst.push_back(entry);
                            } else {
                                it->ops += entry.ops;
                            }
                        }
                        std::sort(dst.begin(), dst.end(), [](const auto& a, const auto& b) {
                            return a.ops != b.ops ? a.ops > b.ops : a.shape_bucket < b.shape_bucket;
                        });
                        if (dst.size() > kMatmulTopShapeCount) {
                            dst.resize(kMatmulTopShapeCount);
                        }
                    };
                    merge_shape_entries(req->decode_matmul_top_shapes, qwen36_profile.decode_matmul_top_shapes);
                    merge_shape_entries(req->prefill_matmul_top_shapes, qwen36_profile.prefill_matmul_top_shapes);
                    merge_shape_entries(req->prefill_matmul_ggml_top_shapes,
                                        qwen36_profile.prefill_matmul_ggml_top_shapes);
                    req->q6k_gemv_seen_ops += qwen36_profile.q6k_gemv_seen_ops;
                    req->q6k_gemv_candidate_ops += qwen36_profile.q6k_gemv_candidate_ops;
                    req->q6k_gemv_used_ops += qwen36_profile.q6k_gemv_used_ops;
                    req->q6k_gemv_rejected_ops += qwen36_profile.q6k_gemv_rejected_ops;
                    req->q6k_gemv_reject_quant_input_null_ops += qwen36_profile.q6k_gemv_reject_quant_input_null_ops;
                    req->q6k_gemv_reject_unsupported_quant_input_ops +=
                        qwen36_profile.q6k_gemv_reject_unsupported_quant_input_ops;
                    req->q6k_gemv_reject_shape_ops += qwen36_profile.q6k_gemv_reject_shape_ops;
                    req->q6k_gemv_reject_phase_ops += qwen36_profile.q6k_gemv_reject_phase_ops;
                    req->q6k_gemv_reject_kernel_unavailable_ops +=
                        qwen36_profile.q6k_gemv_reject_kernel_unavailable_ops;
                    req->q6k_gemv_total_ns += qwen36_profile.q6k_gemv_total_ns;
                    if (qwen36_profile.q6k_gemv_last_reject_reason != 0) {
                        req->q6k_gemv_last_reject_reason =
                            Q6KGemvRejectReasonName(qwen36_profile.q6k_gemv_last_reject_reason);
                    }
                    if (!qwen36_profile.q6k_gemv_effective_phase.empty()) {
                        req->q6k_gemv_effective_phase = qwen36_profile.q6k_gemv_effective_phase;
                    }
                    if (!qwen36_profile.q6k_gemv_graph_phase.empty()) {
                        req->q6k_gemv_graph_phase = qwen36_profile.q6k_gemv_graph_phase;
                    }
                    if (!qwen36_profile.q6k_gemv_callback_phase.empty()) {
                        req->q6k_gemv_callback_phase = qwen36_profile.q6k_gemv_callback_phase;
                    }
                    merge_shape_entries(req->q6k_gemv_weight_shapes, qwen36_profile.q6k_gemv_weight_shapes);
                    req->moe_small_decode_parallel_candidate_ops +=
                        qwen36_profile.moe_small_decode_parallel_candidate_ops;
                    req->moe_small_decode_parallel_used_ops += qwen36_profile.moe_small_decode_parallel_used_ops;
                    req->moe_small_decode_parallel_rejected_ops +=
                        qwen36_profile.moe_small_decode_parallel_rejected_ops;
                    if (!qwen36_profile.moe_small_decode_parallel_last_reject_reason.empty()) {
                        req->moe_small_decode_parallel_last_reject_reason =
                            qwen36_profile.moe_small_decode_parallel_last_reject_reason;
                    }
                    for (std::size_t i = 0; i < req->moe_expert_matmul_weight_type_hist.size(); ++i) {
                        req->moe_expert_matmul_weight_type_hist[i] +=
                            qwen36_profile.moe_expert_matmul_weight_type_hist[i];
                    }
                    req->moe_q4k_repacked_candidate_ops += qwen36_profile.moe_q4k_repacked_candidate_ops;
                    req->moe_q4k_repacked_used_ops += qwen36_profile.moe_q4k_repacked_used_ops;
                    req->moe_q4k_repacked_rejected_ops += qwen36_profile.moe_q4k_repacked_rejected_ops;
                    if (!qwen36_profile.moe_q4k_repacked_last_reject_reason.empty()) {
                        req->moe_q4k_repacked_last_reject_reason = qwen36_profile.moe_q4k_repacked_last_reject_reason;
                    }
                    req->moe_q5k_repacked_candidate_ops += qwen36_profile.moe_q5k_repacked_candidate_ops;
                    req->moe_q5k_repacked_used_ops += qwen36_profile.moe_q5k_repacked_used_ops;
                    req->moe_q5k_repacked_rejected_ops += qwen36_profile.moe_q5k_repacked_rejected_ops;
                    if (!qwen36_profile.moe_q5k_repacked_last_reject_reason.empty()) {
                        req->moe_q5k_repacked_last_reject_reason = qwen36_profile.moe_q5k_repacked_last_reject_reason;
                    }
                    req->gemma4_moe_prefill_quant_batch_candidate_ops +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_candidate_ops;
                    req->gemma4_moe_prefill_quant_batch_used_ops +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_used_ops;
                    req->gemma4_moe_prefill_quant_batch_rejected_ops +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_rejected_ops;
                    req->gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_reject_gate_up_shape_or_type_ops;
                    req->gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_reject_down_shape_or_type_ops;
                    req->gemma4_moe_prefill_quant_batch_gate_up_used +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_gate_up_used;
                    req->gemma4_moe_prefill_quant_batch_down_used +=
                        qwen36_profile.gemma4_moe_prefill_quant_batch_down_used;
                    if (!qwen36_profile.gemma4_moe_prefill_quant_batch_last_reject_reason.empty()) {
                        req->gemma4_moe_prefill_quant_batch_last_reject_reason =
                            qwen36_profile.gemma4_moe_prefill_quant_batch_last_reject_reason;
                    }
                    req->gemma4_native_moe_prefill_candidate_layers +=
                        qwen36_profile.gemma4_native_moe_prefill_candidate_layers;
                    req->gemma4_native_moe_prefill_used_layers += qwen36_profile.gemma4_native_moe_prefill_used_layers;
                    req->gemma4_native_moe_prefill_rejected_layers +=
                        qwen36_profile.gemma4_native_moe_prefill_rejected_layers;
                    req->gemma4_native_moe_prefill_gate_up_ns += qwen36_profile.gemma4_native_moe_prefill_gate_up_ns;
                    req->gemma4_native_moe_prefill_down_ns += qwen36_profile.gemma4_native_moe_prefill_down_ns;
                    req->gemma4_native_moe_prefill_total_ns += qwen36_profile.gemma4_native_moe_prefill_total_ns;
                    req->gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops +=
                        qwen36_profile.gemma4_native_moe_prefill_replaced_ggml_mul_mat_id_ops;
                    req->gemma4_native_moe_prefill_duplicate_work_detected +=
                        qwen36_profile.gemma4_native_moe_prefill_duplicate_work_detected;
                    if (!qwen36_profile.gemma4_native_moe_prefill_last_reject_reason.empty()) {
                        req->gemma4_native_moe_prefill_last_reject_reason =
                            qwen36_profile.gemma4_native_moe_prefill_last_reject_reason;
                    }
                    req->gemma4_dense_prefill_native_candidate_ops +=
                        qwen36_profile.gemma4_dense_prefill_native_candidate_ops;
                    req->gemma4_dense_prefill_native_used_ops += qwen36_profile.gemma4_dense_prefill_native_used_ops;
                    req->gemma4_dense_prefill_native_rejected_ops +=
                        qwen36_profile.gemma4_dense_prefill_native_rejected_ops;
                    req->gemma4_dense_prefill_native_q4k_ops += qwen36_profile.gemma4_dense_prefill_native_q4k_ops;
                    req->gemma4_dense_prefill_native_q8_0_ops += qwen36_profile.gemma4_dense_prefill_native_q8_0_ops;
                    req->gemma4_dense_prefill_native_ns += qwen36_profile.gemma4_dense_prefill_native_ns;
                    req->gemma4_dense_prefill_replaced_ggml_mul_mat_ops +=
                        qwen36_profile.gemma4_dense_prefill_replaced_ggml_mul_mat_ops;
                    req->gemma4_dense_prefill_duplicate_work_detected +=
                        qwen36_profile.gemma4_dense_prefill_duplicate_work_detected;
                    if (!qwen36_profile.gemma4_dense_prefill_native_last_reject_reason.empty()) {
                        req->gemma4_dense_prefill_native_last_reject_reason =
                            qwen36_profile.gemma4_dense_prefill_native_last_reject_reason;
                    }
                    req->gemma4_fast_gelu_enabled += qwen36_profile.gemma4_fast_gelu_enabled;
                    req->gemma4_fast_gelu_used += qwen36_profile.gemma4_fast_gelu_used;
                    req->gemma4_fast_gelu_ns += qwen36_profile.gemma4_fast_gelu_ns;
                    req->gemma4_native_moe_prefill_gate_up_fast_gelu_ns +=
                        qwen36_profile.gemma4_native_moe_prefill_gate_up_fast_gelu_ns;
                    req->gemma4_decode_native_candidate_ops += qwen36_profile.gemma4_decode_native_candidate_ops;
                    req->gemma4_decode_native_used_ops += qwen36_profile.gemma4_decode_native_used_ops;
                    req->gemma4_decode_native_rejected_ops += qwen36_profile.gemma4_decode_native_rejected_ops;
                    req->gemma4_decode_native_moe_used_ops += qwen36_profile.gemma4_decode_native_moe_used_ops;
                    req->gemma4_decode_native_dense_used_ops += qwen36_profile.gemma4_decode_native_dense_used_ops;
                    req->gemma4_decode_native_lm_head_used_ops += qwen36_profile.gemma4_decode_native_lm_head_used_ops;
                    req->gemma4_decode_native_ns += qwen36_profile.gemma4_decode_native_ns;
                    req->gemma4_decode_replaced_ggml_mul_mat_ops +=
                        qwen36_profile.gemma4_decode_replaced_ggml_mul_mat_ops;
                    req->gemma4_decode_replaced_ggml_mul_mat_id_ops +=
                        qwen36_profile.gemma4_decode_replaced_ggml_mul_mat_id_ops;
                    req->gemma4_decode_duplicate_work_detected += qwen36_profile.gemma4_decode_duplicate_work_detected;
                    req->gemma4_native_int4_gemv_candidate_ops += qwen36_profile.gemma4_native_int4_gemv_candidate_ops;
                    req->gemma4_native_int4_gemv_used_ops += qwen36_profile.gemma4_native_int4_gemv_used_ops;
                    req->gemma4_native_int4_gemv_ns += qwen36_profile.gemma4_native_int4_gemv_ns;
                    req->gemma4_native_int4_repacked_weight_count +=
                        qwen36_profile.gemma4_native_int4_repacked_weight_count;
                    req->gemma4_native_int4_repacked_bytes += qwen36_profile.gemma4_native_int4_repacked_bytes;
                    req->gemma4_native_fused_gateup_used_ops += qwen36_profile.gemma4_native_fused_gateup_used_ops;
                    req->ggml_delegated_quant_gemv_ops += qwen36_profile.ggml_delegated_quant_gemv_ops;
                    req->gemma4_native_paged_attention_candidate_ops +=
                        qwen36_profile.gemma4_native_paged_attention_candidate_ops;
                    req->gemma4_native_paged_attention_used_ops +=
                        qwen36_profile.gemma4_native_paged_attention_used_ops;
                    req->gemma4_native_paged_attention_ns += qwen36_profile.gemma4_native_paged_attention_ns;
                    req->gemma4_ggml_attention_fallback_ops += qwen36_profile.gemma4_ggml_attention_fallback_ops;
                    req->gemma4_paged_attention_cache_type = qwen36_profile.gemma4_paged_attention_cache_type;
                    req->gemma4_paged_attention_context_len = std::max(
                        req->gemma4_paged_attention_context_len, qwen36_profile.gemma4_paged_attention_context_len);
                    if (qwen36_profile.gemma4_paged_attention_head_range != 0) {
                        req->gemma4_paged_attention_head_range = qwen36_profile.gemma4_paged_attention_head_range;
                    }
                    if (!qwen36_profile.gemma4_decode_native_last_reject_reason.empty()) {
                        req->gemma4_decode_native_last_reject_reason =
                            qwen36_profile.gemma4_decode_native_last_reject_reason;
                    }
                    req->native_moe_fast_decode_candidate_ops += qwen36_profile.native_moe_fast_decode_candidate_ops;
                    req->native_moe_fast_decode_used_ops += qwen36_profile.native_moe_fast_decode_used_ops;
                    req->native_moe_fast_decode_rejected_ops += qwen36_profile.native_moe_fast_decode_rejected_ops;
                    req->native_moe_fast_decode_w1w3_used_ops += qwen36_profile.native_moe_fast_decode_w1w3_used_ops;
                    req->native_moe_fast_decode_w2_used_ops += qwen36_profile.native_moe_fast_decode_w2_used_ops;
                    req->native_moe_fast_decode_ns += qwen36_profile.native_moe_fast_decode_ns;
                    req->native_moe_fast_w1w3_ns += qwen36_profile.native_moe_fast_w1w3_ns;
                    req->native_moe_fast_w2_ns += qwen36_profile.native_moe_fast_w2_ns;
                    req->native_moe_fast_reduce_ns += qwen36_profile.native_moe_fast_reduce_ns;
                    req->native_moe_fast_total_ns += qwen36_profile.native_moe_fast_total_ns;
                    req->native_moe_fast_w1w3_used_ops += qwen36_profile.native_moe_fast_w1w3_used_ops;
                    req->native_moe_fast_w2_used_ops += qwen36_profile.native_moe_fast_w2_used_ops;
                    req->native_moe_fast_w2_q5k_candidate_ops += qwen36_profile.native_moe_fast_w2_q5k_candidate_ops;
                    req->native_moe_fast_w2_q5k_used_ops += qwen36_profile.native_moe_fast_w2_q5k_used_ops;
                    req->native_moe_fast_w2_q5k_rejected_ops += qwen36_profile.native_moe_fast_w2_q5k_rejected_ops;
                    req->native_moe_fast_w2_q5k_ns += qwen36_profile.native_moe_fast_w2_q5k_ns;
                    if (!qwen36_profile.native_moe_fast_decode_last_reject_reason.empty()) {
                        req->native_moe_fast_decode_last_reject_reason =
                            qwen36_profile.native_moe_fast_decode_last_reject_reason;
                    }
                    if (!qwen36_profile.native_moe_fast_w2_q5k_last_reject_reason.empty()) {
                        req->native_moe_fast_w2_q5k_last_reject_reason =
                            qwen36_profile.native_moe_fast_w2_q5k_last_reject_reason;
                    }
                    // Mixed fast/fallback counts across different native-MoE nodes are expected when only a subset of
                    // layers or quant types can use the custom path. Same-node duplicate work must be reported by the
                    // executing node itself; global coexistence is not sufficient evidence.
                    if (req->native_moe_fast_decode_candidate_ops > 0 && req->native_moe_fast_decode_w2_used_ops == 0) {
                        req->native_moe_fast_missing_w2 = 1;
                    }
                    if (qwen36_profile.qwen35_moe_path != 0) {
                        req->qwen35_moe_path = Qwen35MoEPathName(qwen36_profile.qwen35_moe_path);
                    }
                    req->qwen35_moe_layers_seen += qwen36_profile.qwen35_moe_layers_seen;
                    req->qwen35_moe_forward_calls += qwen36_profile.qwen35_moe_forward_calls;
                    for (std::size_t i = 0; i < req->qwen35_moe_w1w3_weight_type_hist.size(); ++i) {
                        req->qwen35_moe_w1w3_weight_type_hist[i] += qwen36_profile.qwen35_moe_w1w3_weight_type_hist[i];
                        req->qwen35_moe_w2_weight_type_hist[i] += qwen36_profile.qwen35_moe_w2_weight_type_hist[i];
                    }
                    req->qwen35_moe_selected_expert_count = std::max(req->qwen35_moe_selected_expert_count,
                                                                     qwen36_profile.qwen35_moe_selected_expert_count);
                    req->qwen35_moe_top_k = std::max(req->qwen35_moe_top_k, qwen36_profile.qwen35_moe_top_k);
                    req->native_moe_selected_experts =
                        std::max(req->native_moe_selected_experts, qwen36_profile.qwen35_moe_selected_expert_count);
                    req->native_moe_fast_covered_experts =
                        req->native_moe_fast_decode_used_ops > 0 ? req->native_moe_selected_experts : 0;
                    if (req->native_moe_fast_decode_used_ops > 0 &&
                        req->native_moe_fast_covered_experts < req->native_moe_selected_experts) {
                        req->native_moe_fast_partial_expert_coverage = 1;
                    }
                    req->qwen35_moe_instrumentation_missing = std::max(
                        req->qwen35_moe_instrumentation_missing, qwen36_profile.qwen35_moe_instrumentation_missing);
                    req->moe_selected_expert_count =
                        std::max(req->moe_selected_expert_count, qwen36_profile.moe_selected_expert_count);
                    req->moe_top_k = std::max(req->moe_top_k, qwen36_profile.moe_top_k);
                    req->moe_expert_parallel_tasks =
                        std::max(req->moe_expert_parallel_tasks, qwen36_profile.moe_expert_parallel_tasks);
                    for (const auto& entry : qwen36_profile.matmul_dispatch_top_slow_entries) {
                        auto it = std::find_if(req->matmul_dispatch_top_slow_entries.begin(),
                                               req->matmul_dispatch_top_slow_entries.end(),
                                               [&](const MatmulDispatchCensusEntry& existing) {
                                                   return existing.model_family == entry.model_family &&
                                                          existing.phase == entry.phase &&
                                                          existing.dispatch_path == entry.dispatch_path &&
                                                          existing.weight_type == entry.weight_type &&
                                                          existing.shape_bucket == entry.shape_bucket;
                                               });
                        if (it == req->matmul_dispatch_top_slow_entries.end()) {
                            req->matmul_dispatch_top_slow_entries.push_back(entry);
                        } else {
                            it->wall_ns += entry.wall_ns;
                            it->ops += entry.ops;
                        }
                    }
                    std::sort(req->matmul_dispatch_top_slow_entries.begin(),
                              req->matmul_dispatch_top_slow_entries.end(), [](const auto& a, const auto& b) {
                                  return a.wall_ns != b.wall_ns ? a.wall_ns > b.wall_ns : a.ops > b.ops;
                              });
                    if (req->matmul_dispatch_top_slow_entries.size() > kMatmulDispatchTopSlowCount) {
                        req->matmul_dispatch_top_slow_entries.resize(kMatmulDispatchTopSlowCount);
                    }
                    req->qwen_target_ggml_compute_ops += qwen36_profile.qwen_target_ggml_compute_ops;
                    req->qwen_target_ggml_matmul_ops += qwen36_profile.qwen_target_ggml_matmul_ops;
                    req->qwen_target_ggml_matmul_id_ops += qwen36_profile.qwen_target_ggml_matmul_id_ops;
                    req->qwen_target_ggml_quant_vecdot_ops += qwen36_profile.qwen_target_ggml_quant_vecdot_ops;
                    req->qwen_target_ggml_quantize_kv_ops += qwen36_profile.qwen_target_ggml_quantize_kv_ops;
                    req->qwen_target_ggml_attention_ops += qwen36_profile.qwen_target_ggml_attention_ops;
                    if (!qwen36_profile.qwen_target_ggml_compute_last_reason.empty()) {
                        req->qwen_target_ggml_compute_last_reason = qwen36_profile.qwen_target_ggml_compute_last_reason;
                    }
                    if (!qwen36_profile.qwen_target_ggml_compute_last_op.empty()) {
                        req->qwen_target_ggml_compute_last_op = qwen36_profile.qwen_target_ggml_compute_last_op;
                    }
                    if (!qwen36_profile.qwen_target_ggml_compute_target.empty()) {
                        req->qwen_target_ggml_compute_target = qwen36_profile.qwen_target_ggml_compute_target;
                    }
                    merge_shape_entries(req->qwen36_prefill_top_slow_ops, qwen36_profile.qwen36_prefill_top_slow_ops);
                    req->qwen36_prefill_total_ns += qwen36_profile.qwen36_prefill_total_ns;
                    req->qwen36_prefill_ssm_projection_ns += qwen36_profile.qwen36_prefill_ssm_projection_ns;
                    req->qwen36_prefill_ssm_delta_state_ns += qwen36_profile.qwen36_prefill_ssm_delta_state_ns;
                    req->qwen36_prefill_attention_ns += qwen36_profile.qwen36_prefill_attention_ns;
                    req->qwen36_prefill_mlp_or_moe_ns += qwen36_profile.qwen36_prefill_mlp_or_moe_ns;
                    req->qwen36_prefill_graph_build_ns += qwen36_profile.qwen36_prefill_graph_build_ns;
                    req->qwen36_prefill_graph_execute_ns += qwen36_profile.qwen36_prefill_graph_execute_ns;
                    req->q4k_copied_gemv_experiment_used =
                        std::max(req->q4k_copied_gemv_experiment_used, qwen36_profile.q4k_copied_gemv_experiment_used);
                    if (qwen36_profile.q4k_copied_gemv_experiment_last_reject_reason != 0) {
                        req->q4k_copied_gemv_experiment_last_reject_reason =
                            qwen36_profile.q4k_copied_gemv_experiment_last_reject_reason;
                        req->q4k_copied_gemv_experiment_reject_reason = Q4KCopiedGemvExperimentRejectReasonName(
                            qwen36_profile.q4k_copied_gemv_experiment_last_reject_reason);
                    }
                    req->paged_attn_decode_head_tile_effective =
                        std::max(req->paged_attn_decode_head_tile_effective,
                                 qwen36_profile.paged_attn_decode_head_tile_effective);
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
                    req->flash_attention_headseq_prefill_calls += qwen36_profile.flash_attention_headseq_prefill_calls;
                    req->flash_attention_native_decode_calls += qwen36_profile.flash_attention_native_decode_calls;
                    req->flash_attention_reference_calls += qwen36_profile.flash_attention_reference_calls;
                    req->flash_attention_non_avx512_tiled_calls +=
                        qwen36_profile.flash_attention_non_avx512_tiled_calls;
                    req->flash_attention_avx512_tiled_calls += qwen36_profile.flash_attention_avx512_tiled_calls;
                    req->flash_attention_last_nth =
                        std::max(req->flash_attention_last_nth, qwen36_profile.flash_attention_last_nth);
                    req->flash_attention_last_active_threads = std::max(
                        req->flash_attention_last_active_threads, qwen36_profile.flash_attention_last_active_threads);
                    req->kleidiai_compiled_enabled =
                        std::max(req->kleidiai_compiled_enabled, qwen36_profile.kleidiai_compiled_enabled);
                    if (qwen36_profile.kleidiai_last_reject_reason != 0) {
                        req->kleidiai_last_reject_reason = qwen36_profile.kleidiai_last_reject_reason;
                    }
                    req->moe_task_count = std::max(req->moe_task_count, qwen36_profile.moe_task_count);
                    req->moe_rowblock_used = std::max(req->moe_rowblock_used, qwen36_profile.moe_rowblock_used);
                    req->moe_rowblock_tasks = std::max(req->moe_rowblock_tasks, qwen36_profile.moe_rowblock_tasks);
                    req->selected_expert_count =
                        std::max(req->selected_expert_count, qwen36_profile.selected_expert_count);
                    req->ssm_conv1d_calls = std::max(req->ssm_conv1d_calls, qwen36_profile.ssm_conv1d_calls);
                    req->ssm_delta_calls = std::max(req->ssm_delta_calls, qwen36_profile.ssm_delta_calls);
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
            if (is_decode_batch && decode_batch_size >= 1 &&
                decode_batch_size < static_cast<int>(kDecodeGraphCacheTrackedBatches)) {
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
                        auto verify_work_ctx =
                            std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)>(
                                CreateInferenceWorkContext(), DestroyInferenceWorkContext);
                        ScopedBatchWorkContext verify_scope(verify_work_ctx.get(), &single_batch,
                                                            InferenceExecutionPhase::Decode);
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
                    SetCurrentWorkContext(work_ctx.get());
                    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
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
                    SetCurrentWorkContext(work_ctx.get());
                    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
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
                        if (prefix_cache_allowed && !req->original_prompt_tokens_for_cache.empty() &&
                            !req->block_table.empty() && req->n_past > 0) {
                            const int* tokens_ptr = req->original_prompt_tokens_for_cache.data();
                            int total_tokens = static_cast<int>(req->original_prompt_tokens_for_cache.size());
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
                                const bool attach_hybrid_snapshot = ModelRequiresPrefixStateSnapshot(current_model) &&
                                                                    ((blk_idx + 1) * BLOCK_SIZE == req->n_past);
                                if (attach_hybrid_snapshot) {
                                    DebugLogHybridSSMSnapshot("save", req->id, req->n_past, block_id,
                                                              req->ssm_runtime_states);
                                }
                                current_kv_cache->block_manager->RegisterPrefixBlockWithTokens(
                                    block_id, hash, tokens_ptr + start_token, block_tokens,
                                    attach_hybrid_snapshot ? &req->ssm_runtime_states : nullptr);
                                req->prefix_cache_registered_blocks++;
                                if (blk_idx >= req->prefix_cache_hit_blocks) {
                                    req->prefix_cache_extended_blocks++;
                                }
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
                        ClearQwen36SSMQ8PrefillAMXAliases(current_model);
                        // Clear after registration, or after skipping cache registration for hybrid SSM.
                        req->prompt_tokens_for_cache.clear();
                        req->original_prompt_tokens_for_cache.clear();

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
                    if (current_model && current_model->arch_flags.is_gemma4 &&
                        current_model->gemma4_final_logit_softcapping > 0.0f &&
                        !densecore::models::IsGemma4FinalLogitSoftcapDisabled()) {
                        sampling_params.final_logit_softcap = current_model->gemma4_final_logit_softcapping;
                    }
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
                    {
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
                    if (!req->json_mode && !token_str.empty() && req->suppress_reasoning_tags &&
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
                    if (req->callback || req->callback_ex || req->token_result_callback) {
                        if (!token_str.empty() || req->token_result_callback) {
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
                            if (req->callback || req->callback_ex || req->token_result_callback) {
                                EmitRequestResult(state, req, "Error: Out of memory", -1, true, true,
                                                  global_direct_callback);
                                terminal_error_emitted = true;
                            }
                            break;
                        }
                        req->block_table.push_back(new_blocks[0]);
                    }

                    // Check stop sequences (best-effort, suffix match)
                    if (!token_str.empty() && !req->stop_sequences.empty() && req->stop_buffer_max > 0) {
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
                            auto format_sampling_candidate = [&](const SamplingDebugCandidate& c) {
                                std::string raw;
                                if (c.token_id >= 0 &&
                                    c.token_id < static_cast<int>(current_model->vocab_tokens.size())) {
                                    raw = current_model->vocab_tokens[static_cast<size_t>(c.token_id)];
                                    for (char& ch : raw) {
                                        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                                    }
                                }
                                std::cerr << c.token_id << ":" << c.pre_penalty_logit << ":" << c.post_penalty_logit;
                                if (!raw.empty()) {
                                    std::cerr << ":'" << raw << "'";
                                }
                                std::cerr << " ";
                            };
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
                                    format_sampling_candidate(c);
                                }
                                std::cerr << std::endl;
                                std::cerr << "[SAMPLER_TRACE_DUMP] request_id=" << req->id
                                          << " output_token_index=" << entry.output_token_index << " top_post_penalty=";
                                for (const auto& c : entry.top_post_penalty) {
                                    format_sampling_candidate(c);
                                }
                                std::cerr << std::endl;
                            }
                        }
                        if (!terminal_error_emitted && !req->utf8_pending.empty() &&
                            (req->callback || req->callback_ex || req->token_result_callback)) {
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

                            if (!req->finished && HasDecodeVisibleProgressStalled(req, decode_sample_time)) {
                                LOG_ERROR(
                                    "Failing decode-stalled request {} (seq_id={}, generated={}, silent_steps={})",
                                    req->id, req->seq_id, req->generated_count, req->decode_no_output_steps);
                                req->finished = true;
                                req->decode_finish_cause = DecodeFinishCause::DecodeVisibleProgressTimeout;
                                state->metrics.failed_requests++;
                                if (req->callback || req->callback_ex || req->token_result_callback) {
                                    EmitRequestResult(state, req, "Error: decode made no externally visible progress",
                                                      -1, true, true, global_direct_callback);
                                    terminal_error_emitted = true;
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
                        if (!terminal_error_emitted &&
                            (req->callback || req->callback_ex || req->token_result_callback)) {
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
        ClearQwen36SSMQ8PrefillAMXAliases(current_model);
        clear_decode_graph_cache();
        clear_prefill_graph_cache();
        prefill_arena_pool.clear();
        prefill_arena_lru.clear();
    } catch (const densecore::DenseCoreException& e) {
        std::cerr << "[DenseCore] Worker thread exception (" << e.CodeInt() << "): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DenseCore] Worker thread exception: " << e.what() << std::endl;
    }
}
