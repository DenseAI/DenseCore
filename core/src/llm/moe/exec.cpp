#include "llm/moe/exec.h"
#include "backend/cpu_moe_execution.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/models/lfm2_shortconv_math.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/runtime/inference.h"
#include "densecore/runtime/scheduler.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/graph/construction_ops.h"
#include "llm/runtime/cpu_execution.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/work_context.h"
#include "llm/ssm/internal.h"
#include "models/model_inference_policy.h"
#include "runtime/inference_types_internal.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace densecore::llm::graph::detail;
using densecore::env::ParseBoolEnv;
using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using namespace densecore::llm::runtime;

static void DebugLogTensorFiniteStats(const char* tag, const struct ggml_tensor* tensor) {
    if (!tag || !tensor || !tensor->data || tensor->type != GGML_TYPE_F32) {
        return;
    }
    const float* data = reinterpret_cast<const float*>(tensor->data);
    const int n = ggml_nelements(tensor);
    if (n <= 0) {
        return;
    }

    int nan_ct = 0;
    int inf_ct = 0;
    int zero_ct = 0;
    int finite_ct = 0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++nan_ct;
            continue;
        }
        if (!std::isfinite(v)) {
            ++inf_ct;
            continue;
        }
        if (v == 0.0f) {
            ++zero_ct;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        ++finite_ct;
    }

    if (!std::isfinite(mn)) mn = 0.0f;
    if (!std::isfinite(mx)) mx = 0.0f;
    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
    std::fprintf(
        stderr,
        "[%s] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n", tag,
        static_cast<int>(tensor->type), static_cast<long>(tensor->ne[0]), static_cast<long>(tensor->ne[1]),
        static_cast<long>(tensor->ne[2]), n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
}

MoEUserData* AllocateMoEUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx_c)) {
        thread_local MoEUserData dry_run_storage;
        dry_run_storage = MoEUserData{};
        return &dry_run_storage;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(MoEUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(MoEUserData));
    return reinterpret_cast<MoEUserData*>(storage->data);
}

std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeights(const TransformerLayer* layer,
                                                                     const TransformerModel* model) {
    using ExpertWeights = densecore::CpuBackend::ExpertWeights;
    using ExpertPackedInt4Weight = densecore::CpuBackend::ExpertPackedInt4Weight;

    std::vector<ExpertWeights> experts;
    if (!layer) {
        return experts;
    }

    const auto make_int4_binding = [model](const ggml_tensor* tensor, int expected_k,
                                           int expected_n) -> ExpertPackedInt4Weight {
        ExpertPackedInt4Weight packed{};
        if (!model || !tensor || expected_k <= 0 || expected_n <= 0) {
            return packed;
        }
        const auto pack_direct = [&](const TransformerModel::Int4WeightBinding& binding,
                                     const ggml_tensor* bound_tensor, int logical_n) -> ExpertPackedInt4Weight {
            if (!bound_tensor || !bound_tensor->data || !binding.packed || !binding.scales || !binding.zeros ||
                !binding.packed->data || !binding.scales->data || !binding.zeros->data || binding.group_size <= 0 ||
                binding.k != expected_k || binding.n != logical_n) {
                return {};
            }
            ExpertPackedInt4Weight resolved{};
            resolved.packed_weights = reinterpret_cast<const uint8_t*>(bound_tensor->data);
            resolved.scales = reinterpret_cast<const float*>(binding.scales->data);
            resolved.zeros = reinterpret_cast<const float*>(binding.zeros->data);
            resolved.group_size = binding.group_size;
            resolved.K = static_cast<int>(binding.k);
            resolved.N = static_cast<int>(binding.n);
            return resolved;
        };

        const auto it = model->int4_weight_bindings.find(tensor);
        if (it != model->int4_weight_bindings.end()) {
            packed = pack_direct(it->second, tensor, expected_n);
            if (packed.IsValid()) {
                return packed;
            }
        }

        const ggml_tensor* root = tensor->view_src;
        if (!root) {
            return {};
        }
        const auto it_root = model->int4_weight_bindings.find(root);
        if (it_root == model->int4_weight_bindings.end()) {
            return {};
        }
        const auto& binding = it_root->second;
        if (!binding.packed || !binding.scales || !binding.zeros || !binding.packed->data || !binding.scales->data ||
            !binding.zeros->data || binding.group_size <= 0 || binding.k != expected_k || binding.n < expected_n ||
            !tensor->data) {
            return {};
        }

        const size_t row_stride_bytes = static_cast<size_t>(root->nb[1]);
        if (row_stride_bytes == 0) {
            return {};
        }
        const size_t plane_stride_bytes = static_cast<size_t>(root->nb[2]);
        const size_t total_offs = tensor->view_offs;
        size_t expert_idx = 0;
        size_t row_offs_bytes = total_offs;
        if (plane_stride_bytes > 0 && root->ne[2] > 1) {
            expert_idx = total_offs / plane_stride_bytes;
            row_offs_bytes = total_offs % plane_stride_bytes;
        }
        if ((row_offs_bytes % row_stride_bytes) != 0) {
            return {};
        }

        const size_t row_start = row_offs_bytes / row_stride_bytes;
        if (row_start + static_cast<size_t>(expected_n) > static_cast<size_t>(binding.n)) {
            return {};
        }

        const int64_t groups_per_row = binding.k / binding.group_size;
        if (groups_per_row <= 0) {
            return {};
        }

        const size_t scales_row_stride = static_cast<size_t>(binding.scales->nb[1] / sizeof(float));
        const size_t zeros_row_stride = static_cast<size_t>(binding.zeros->nb[1] / sizeof(float));
        if (scales_row_stride == 0 || zeros_row_stride == 0) {
            return {};
        }

        size_t scales_off = row_start * scales_row_stride;
        size_t zeros_off = row_start * zeros_row_stride;
        if (expert_idx > 0) {
            if (binding.scales->ne[2] <= 1 || binding.zeros->ne[2] <= 1 || binding.scales->nb[2] == 0 ||
                binding.zeros->nb[2] == 0) {
                return {};
            }
            scales_off += expert_idx * static_cast<size_t>(binding.scales->nb[2] / sizeof(float));
            zeros_off += expert_idx * static_cast<size_t>(binding.zeros->nb[2] / sizeof(float));
        }

        packed.packed_weights = reinterpret_cast<const uint8_t*>(tensor->data);
        packed.scales = reinterpret_cast<const float*>(binding.scales->data) + scales_off;
        packed.zeros = reinterpret_cast<const float*>(binding.zeros->data) + zeros_off;
        packed.group_size = binding.group_size;
        packed.K = expected_k;
        packed.N = expected_n;
        return packed;
    };

    size_t n_experts = layer->NumExperts();
    experts.reserve(n_experts);

    const densecore::models::DecoderLayerSpec* layer_spec =
        densecore::models::ResolveDecoderLayerSpecForLayer(model, layer);

    for (size_t i = 0; i < n_experts; ++i) {
        ExpertWeights w;
        w.w1 = {nullptr, 0};
        w.w2 = {nullptr, 0};
        w.w3 = {nullptr, 0};
        w.hidden_dim = 0;
        w.intermediate_dim = 0;
        if (model && i < model->numa_expert_node_hints.size()) {
            w.numa_node_hint = model->numa_expert_node_hints[i];
        }
        w.use_gelu_activation =
            layer_spec ? layer_spec->ffn.activation == densecore::models::DecoderActivation::GeluPytorchTanh
                       : densecore::models::IsGemma4MoEModel(model, layer);
        w.force_safe_reference = false;
        w.gate_up_tensor = layer->GetExpert(i, model_keys::kGemma4PackedGateUpExpert);
        w.w2_scale_tensor = layer->GetExpert(i, model_keys::kGemma4PackedDownScale);
        w.uses_canonical_gemma4_packed_layout = (w.gate_up_tensor != nullptr);

        auto* gw1 = layer->GetExpert(i, model_keys::kFfnGate);
        if (gw1) {
            w.w1.ptr = gw1->data;
            w.w1.size = ggml_nbytes(gw1);
            w.hidden_dim = static_cast<int>(gw1->ne[0]);
            w.intermediate_dim = static_cast<int>(gw1->ne[1]);
            w.w1_type = static_cast<int>(gw1->type);
            w.w1_tensor = gw1;
            w.w1_int4 = make_int4_binding(gw1, w.hidden_dim, w.intermediate_dim);
        }

        auto* gw2 = layer->GetExpert(i, model_keys::kFfnDown);
        if (gw2) {
            w.w2.ptr = gw2->data;
            w.w2.size = ggml_nbytes(gw2);
            w.w2_type = static_cast<int>(gw2->type);
            w.w2_tensor = gw2;
            w.w2_int4 = make_int4_binding(gw2, w.intermediate_dim, w.hidden_dim);
        }

        auto* gw3 = layer->GetExpert(i, model_keys::kFfnUp);
        if (gw3) {
            w.w3.ptr = gw3->data;
            w.w3.size = ggml_nbytes(gw3);
            w.w3_type = static_cast<int>(gw3->type);
            w.w3_tensor = gw3;
            w.w3_int4 = make_int4_binding(gw3, w.hidden_dim, w.intermediate_dim);
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
    static const bool enabled = ParseBoolEnv("DENSECORE_MOE_ENABLE_PAGE_MIGRATION", false);
    return enabled;
}

static bool IsBenchmarkMode() {
    return ParseBoolEnv("DENSECORE_BENCH_MODE", false);
}


static std::atomic<uint64_t> g_moe_callback_entry_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_userdata_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_backend_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_experts_count{0};
static std::atomic<uint64_t> g_moe_callback_routing_failure_count{0};
static std::atomic<uint64_t> g_moe_callback_empty_routing_count{0};
static std::atomic<uint64_t> g_moe_callback_fail_closed_count{0};
static std::atomic<bool> g_moe_strict_failure_pending{false};
static std::mutex g_moe_strict_failure_mu;
static std::string g_moe_strict_failure_message;

void ResetMoECallbackEntryCount() {
    g_moe_callback_entry_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_userdata_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_backend_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_experts_count.store(0, std::memory_order_relaxed);
    g_moe_callback_routing_failure_count.store(0, std::memory_order_relaxed);
    g_moe_callback_empty_routing_count.store(0, std::memory_order_relaxed);
    g_moe_callback_fail_closed_count.store(0, std::memory_order_relaxed);
    g_moe_strict_failure_pending.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    g_moe_strict_failure_message.clear();
}

uint64_t GetMoECallbackEntryCount() {
    return g_moe_callback_entry_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackMissingUserdataCount() {
    return g_moe_callback_missing_userdata_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackMissingBackendCount() {
    return g_moe_callback_missing_backend_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackMissingExpertsCount() {
    return g_moe_callback_missing_experts_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackRoutingFailureCount() {
    return g_moe_callback_routing_failure_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackEmptyRoutingCount() {
    return g_moe_callback_empty_routing_count.load(std::memory_order_relaxed);
}

uint64_t GetMoECallbackFailClosedCount() {
    return g_moe_callback_fail_closed_count.load(std::memory_order_relaxed);
}

bool IsMoEDebugLoggingEnabled() {
    return densecore::env::ParseDiagnosticEnv(
        "DENSECORE_DEBUG_MOE", densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_MOE_CALLBACKS", false));
}

static bool IsMoEStrictModeEnabled() {
    return ParseBoolEnv("DENSECORE_MOE_STRICT", false);
}

static bool IsMoEStageTimingEnabled() {
    return densecore::env::ParseDiagnosticEnv(
        "DENSECORE_MOE_STAGE_TIMING", densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_MOE_STAGE_TIMING", false));
}

static bool IsMoEDetailedTraceEnabled() {
    return densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_MOE_TRACE", false);
}

void ResetMoEStrictFailureState() {
    g_moe_strict_failure_pending.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    g_moe_strict_failure_message.clear();
}

bool ConsumeMoEStrictFailureState(std::string* message) {
    if (!g_moe_strict_failure_pending.exchange(false, std::memory_order_acq_rel)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    if (message) {
        *message = g_moe_strict_failure_message;
    }
    g_moe_strict_failure_message.clear();
    return true;
}

static void RecordMoEStrictFailure(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
        g_moe_strict_failure_message = message;
    }
    g_moe_strict_failure_pending.store(true, std::memory_order_release);
}

static void ZeroFillTensor(struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
}

static void FailClosedMoECallback(struct ggml_tensor* dst, std::atomic<uint64_t>* counter, const std::string& message) {
    if (counter) {
        counter->fetch_add(1, std::memory_order_relaxed);
    }
    g_moe_callback_fail_closed_count.fetch_add(1, std::memory_order_relaxed);
    ZeroFillTensor(dst);
    std::fprintf(stderr, "[DenseCore][MoE] fail-closed: %s\n", message.c_str());
    if (IsMoEStrictModeEnabled()) {
        RecordMoEStrictFailure(message);
    }
}

static int GetMoEDebugSelectedLayer() {
    static const int layer = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_MOE_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    return layer;
}

static int GetMoEDebugSelectedToken() {
    static const int token = std::max(0, densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_MOE_TOKEN", 0));
    return token;
}

static bool ShouldTraceMoELayer(const MoEUserData* ud) {
    if (!IsMoEDetailedTraceEnabled() || !ud) {
        return false;
    }
    const int selected_layer = GetMoEDebugSelectedLayer();
    return selected_layer < 0 || ud->layer_idx == selected_layer;
}

static void DumpMoERouteTrace(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                              const densecore::moe::MoERouteResult& routing) {
    if (!ShouldTraceMoELayer(ud) || !gate_logits || !gate_logits->data || routing.batch_size <= 0) {
        return;
    }

    const int token_idx = std::min(GetMoEDebugSelectedToken(), routing.batch_size - 1);
    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    const float* row = logits + static_cast<size_t>(token_idx) * n_experts;

    std::fprintf(stderr,
                 "[MOE_TRACE_ROUTE] layer=%d token=%d batch=%d experts=%d top_k=%d shared=%d scale=%g norm_topk=%d\n",
                 ud->layer_idx, token_idx, routing.batch_size, n_experts, routing.top_k,
                 ud->model ? ud->model->moe_n_shared_experts : 0,
                 ud->model ? static_cast<double>(ud->model->moe_routed_scaling_factor) : 0.0,
                 (ud->model && ud->model->moe_norm_topk_prob) ? 1 : 0);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] logits=");
    for (int e = 0; e < n_experts; ++e) {
        if (e != 0) std::fputc(',', stderr);
        std::fprintf(stderr, "%g", static_cast<double>(row[e]));
    }
    std::fputc('\n', stderr);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] selected=");
    for (int k = 0; k < routing.top_k; ++k) {
        const size_t idx = static_cast<size_t>(token_idx * routing.top_k + k);
        std::fprintf(stderr, "%s(%d,%g)", k == 0 ? "" : ",", routing.expert_ids[idx],
                     static_cast<double>(routing.weights[idx]));
    }
    std::fputc('\n', stderr);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] token_indices=");
    for (int k = 0; k < routing.top_k; ++k) {
        const size_t idx = static_cast<size_t>(token_idx * routing.top_k + k);
        std::fprintf(stderr, "%s%d", k == 0 ? "" : ",", routing.token_indices[idx]);
    }
    std::fputc('\n', stderr);
}

static void DumpMoEOutputTrace(const struct ggml_tensor* dst, const MoEUserData* ud) {
    if (!ShouldTraceMoELayer(ud) || !dst || !dst->data) {
        return;
    }
    const int64_t elems = ggml_nelements(dst);
    const float* values = reinterpret_cast<const float*>(dst->data);
    double sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    for (int64_t i = 0; i < elems; ++i) {
        const double v = static_cast<double>(values[i]);
        sum += v;
        abs_sum += std::fabs(v);
        sq_sum += v * v;
    }
    std::fprintf(stderr, "[MOE_TRACE_OUT] layer=%d elems=%lld sum=%g abs_sum=%g sq_sum=%g\n", ud->layer_idx,
                 static_cast<long long>(elems), sum, abs_sum, sq_sum);
}

void EnsureMoERebalanceThread(densecore::CpuBackend* backend) {
    if (IsBenchmarkMode() || !backend || backend->IsRebalanceThreadRunning()) {
        return;
    }
    backend->StartRebalanceThread(GetMoERebalanceIntervalMs(), GetMoERebalanceTopK(), IsMoEPageMigrationEnabled());
}

static void UpdateSchedulerExperts(const MoEUserData* ud, const densecore::moe::MoERouteResult& routing) {
    if (!ud) {
        return;
    }
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch || !batch->scheduler || batch->seq_id.empty() || batch->scheduler_seq_ids.empty()) {
        return;
    }

    const int num_seqs = static_cast<int>(batch->scheduler_seq_ids.size());
    thread_local std::vector<std::vector<int>> per_seq_experts;
    per_seq_experts.resize(static_cast<size_t>(num_seqs));
    for (auto& experts : per_seq_experts) {
        experts.clear();
    }

    const bool has_token_indices = !routing.token_indices.empty();
    const int total_assignments = static_cast<int>(routing.expert_ids.size());
    const bool decode_single_token_per_seq = batch->seq_id.size() == batch->scheduler_seq_ids.size() &&
                                             static_cast<int>(batch->seq_id.size()) == routing.batch_size;

    if (decode_single_token_per_seq) {
        std::vector<int> experts;
        experts.reserve(static_cast<size_t>(std::max(1, routing.top_k)));

        const int token_count = static_cast<int>(batch->seq_id.size());
        for (int token_idx = 0; token_idx < token_count; ++token_idx) {
            const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
            if (seq_idx < 0 || seq_idx >= num_seqs) {
                continue;
            }

            experts.clear();
            for (int i = 0; i < total_assignments; ++i) {
                const int routed_token_idx =
                    has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / routing.top_k);
                if (routed_token_idx != token_idx) {
                    continue;
                }
                const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                if (expert_id < 0) {
                    continue;
                }
                if (std::find(experts.begin(), experts.end(), expert_id) == experts.end()) {
                    experts.push_back(expert_id);
                }
            }

            if (experts.empty()) {
                continue;
            }
            const int sched_seq_id = batch->scheduler_seq_ids[static_cast<size_t>(seq_idx)];
            if (sched_seq_id < 0) {
                continue;
            }
            batch->scheduler->SetPredictedExperts(sched_seq_id, experts);
        }
        return;
    }

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

        if (!decode_single_token_per_seq && experts.size() > 1) {
            std::sort(experts.begin(), experts.end());
            experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        }
        batch->scheduler->SetPredictedExperts(sched_seq_id, experts);
    }
}

static bool EnsureMoERouteStorage(densecore::moe::MoERouteResult* routing, int batch_size, int top_k) {
    if (!routing || batch_size <= 0 || top_k <= 0) {
        return false;
    }
    const size_t total = static_cast<size_t>(batch_size * top_k);
    routing->batch_size = batch_size;
    routing->top_k = top_k;
    routing->expert_ids.resize(total);
    routing->weights.resize(total);
    routing->token_indices.resize(total);
    std::fill(routing->expert_ids.begin(), routing->expert_ids.end(), -1);
    std::fill(routing->weights.begin(), routing->weights.end(), 0.0f);
    std::fill(routing->token_indices.begin(), routing->token_indices.end(), 0);
    return true;
}

bool RouteMoEGroupedSigmoid(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                            densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !routing) {
        return false;
    }
    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    const struct ggml_tensor* bias_t = ud->layer ? ud->layer->Get(model_keys::kMoeCorrectionBias) : nullptr;
    const float* correction_bias =
        (bias_t && bias_t->type == GGML_TYPE_F32) ? reinterpret_cast<const float*>(bias_t->data) : nullptr;

    const int n_group = std::max(1, ud->model->moe_n_group);
    const int group_size = std::max(1, n_experts / n_group);
    const int topk_group = std::max(1, std::min(ud->model->moe_topk_group, n_group));

    thread_local std::vector<float> probs;
    thread_local std::vector<float> choice_scores;
    thread_local std::vector<float> group_scores;
    thread_local std::vector<int> active_groups;
    thread_local std::vector<int> selected;
    thread_local std::vector<std::pair<float, int>> candidates;
    probs.assign(static_cast<size_t>(n_experts), 0.0f);
    choice_scores.assign(static_cast<size_t>(n_experts), 0.0f);
    group_scores.assign(static_cast<size_t>(n_group), -std::numeric_limits<float>::infinity());
    active_groups.resize(static_cast<size_t>(n_group));
    selected.assign(static_cast<size_t>(top_k), -1);
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

        candidates.clear();
        candidates.reserve(static_cast<size_t>(std::max(top_k, topk_group * group_size)));
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
            routing->expert_ids[static_cast<size_t>(token_idx * top_k + k)] = expert_id;
            routing->weights[static_cast<size_t>(token_idx * top_k + k)] = weight;
            routing->token_indices[static_cast<size_t>(token_idx * top_k + k)] = token_idx;
        }
    }

    return true;
}

bool RouteMoESoftmaxTopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                         densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !routing) {
        return false;
    }

    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    thread_local std::vector<uint8_t> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    route_workspace.resize(workspace_bytes + 64);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size(), batch_size,
                                                 n_experts, top_k)) {
        return false;
    }
    if (!densecore::moe::MoETopKRoute(logits, batch_size, n_experts, top_k, ud->model->moe_norm_topk_prob, routing,
                                      &ws)) {
        return false;
    }

    const float scale = ud->model->moe_routed_scaling_factor;
    for (float& weight : routing->weights) {
        weight *= scale;
    }
    return true;
}

bool RouteMoEGemma4TopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                        densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !ud->layer || !routing) {
        return false;
    }

    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    thread_local std::vector<uint8_t> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    route_workspace.resize(workspace_bytes + 64);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size(), batch_size,
                                                 n_experts, top_k)) {
        return false;
    }
    // Gemma4/HF router uses softmax probabilities followed by top-k
    // renormalization. The per-expert scale is represented as a down-projection
    // sidecar in GGUF and is applied in the expert projection path, not as a
    // second router-probability multiplier.
    if (!densecore::moe::MoETopKRoute(logits, batch_size, n_experts, top_k, /*normalize_weights=*/true, routing, &ws)) {
        return false;
    }

    for (size_t i = 0; i < routing->weights.size(); ++i) {
        routing->token_indices[i] = static_cast<int>(i / static_cast<size_t>(top_k));
    }
    return true;
}


void cb_moe_forward(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                    int nth, void* userdata) {
    (void)nth;
    const auto profile_begin = (ith == 0 && IsQwen36ProfilingEnabled()) ? std::chrono::steady_clock::now()
                                                                        : std::chrono::steady_clock::time_point{};
    if (ith == 0) {
        g_moe_callback_entry_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (IsMoEDebugLoggingEnabled() && ith == 0) {
        static std::atomic<int> moe_call_count{0};
        int call_id = moe_call_count.fetch_add(1);
        fprintf(stderr, "[DBG] cb_moe_forward #%d src0=[%lld,%lld] src1=[%lld,%lld]\n", call_id, (long long)src0->ne[0],
                (long long)src0->ne[1], (long long)src1->ne[0], (long long)src1->ne[1]);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_IN", src0);
            DebugLogTensorFiniteStats("MOE_GATE", src1);
        }
    }
    if (ith != 0) {
        return;
    }
    auto* ud = static_cast<MoEUserData*>(userdata);
    ScopedInferenceWorkContext context(ud ? ud->work_ctx : nullptr);
    if (!ud || !ud->layer) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_userdata_count, "MoE callback missing userdata/layer");
        return;
    }
    if (!ud->backend) {
        if (ud->work_ctx) {
            const BatchSpec* bound_batch = GetInferenceWorkContextBatch(ud->work_ctx);
            if (!bound_batch || !bound_batch->deps ||
                (!bound_batch->deps->cpu_backend && !bound_batch->deps->backend_registry)) {
                FailClosedMoECallback(dst, &g_moe_callback_missing_backend_count,
                                      "MoE graph context is missing CPU backend dependencies");
                return;
            }
            ud->backend = &densecore::llm::runtime::ResolveCpuBackend(bound_batch);
        } else {
            // Standalone legacy callbacks resolve the registry CPU first. They
            // must not bypass expert registrations on that backend instance.
            ud->backend = &densecore::GetTelemetryCpuBackend();
        }
    }
    if (!ud->backend) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_backend_count, "MoE callback missing CPU backend");
        return;
    }
    if ((!ud->experts_registered || !ud->experts || ud->n_experts <= 0) && ud->layer) {
        const densecore::CpuBackend::ExpertWeights* registered_experts = nullptr;
        int registered_count = 0;
        if (ud->backend->GetRegisteredExpertsView(ud->layer, &registered_experts, &registered_count) &&
            registered_experts && registered_count > 0) {
            ud->experts = registered_experts;
            ud->n_experts = registered_count;
            ud->experts_registered = true;
        }
    }
    if (!ud->experts_registered || !ud->experts || ud->n_experts <= 0) {
        FailClosedMoECallback(dst, &g_moe_callback_missing_experts_count, "MoE callback missing registered experts");
        return;
    }

    // Routing (src1 = gate_logits)
    const bool debug_stage_timing = IsMoEStageTimingEnabled();
    const bool collect_profile = ud->profile && profile_begin != std::chrono::steady_clock::time_point{};
    const auto route_begin = (debug_stage_timing || collect_profile) ? std::chrono::steady_clock::now()
                                                                     : std::chrono::steady_clock::time_point{};
    thread_local densecore::moe::MoERouteResult routing;
    const densecore::models::DecoderLayerSpec* layer_spec =
        densecore::models::ResolveDecoderLayerSpecForLayer(ud->model, ud->layer);
    densecore::models::DecoderMoERouter router =
        layer_spec ? layer_spec->ffn.router : densecore::models::DecoderMoERouter::SoftmaxTopK;
    const bool use_grouped_sigmoid_routing = router == densecore::models::DecoderMoERouter::GroupedSigmoidTopK;
    const bool routed = use_grouped_sigmoid_routing ? RouteMoEGroupedSigmoid(src1, ud, &routing)
                                                    : (router == densecore::models::DecoderMoERouter::Gemma4SoftmaxTopK
                                                           ? RouteMoEGemma4TopK(src1, ud, &routing)
                                                           : RouteMoESoftmaxTopK(src1, ud, &routing));
    if (!routed) {
        FailClosedMoECallback(dst, &g_moe_callback_routing_failure_count, "MoE routing failed");
        return;
    }
    if (ud->test_force_empty_routing) {
        routing.expert_ids.clear();
        routing.weights.clear();
        routing.token_indices.clear();
    }
    if (routing.expert_ids.empty()) {
        FailClosedMoECallback(dst, &g_moe_callback_empty_routing_count, "MoE routing produced no experts");
        return;
    }
    if (ith == 0) {
        DumpMoERouteTrace(src1, ud, routing);
    }
    const auto route_end = (debug_stage_timing || collect_profile) ? std::chrono::steady_clock::now()
                                                                   : std::chrono::steady_clock::time_point{};
    if (ith == 0) {
        UpdateSchedulerExperts(ud, routing);
    }
    const auto scheduler_end =
        debug_stage_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (ud->profile) {
        SetQwen36ProfileMax(ud->profile->moe_task_count, 1);
        int unique_experts = 0;
        for (size_t i = 0; i < routing.expert_ids.size(); ++i) {
            const int expert_id = routing.expert_ids[i];
            bool seen = false;
            for (size_t j = 0; j < i; ++j) {
                if (routing.expert_ids[j] == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                ++unique_experts;
            }
        }
        SetQwen36ProfileMax(ud->profile->selected_expert_count, unique_experts);
    }

    // Forward (src0 = input, dst = output)
    densecore::Tensor t_input = GgmlToRowMajorTensor(src0);
    densecore::Tensor t_output = GgmlToRowMajorTensor(dst);
    densecore::CpuBackend::MoEForwardProfile moe_profile;
    densecore::CpuBackend::MoEForwardProfile* moe_profile_ptr = collect_profile ? &moe_profile : nullptr;
    if (collect_profile && route_end >= route_begin) {
        moe_profile.route_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(route_end - route_begin).count());
    }
    const BatchSpec* active_batch = GetCurrentBatch();
    const auto execution_options = densecore::llm::runtime::ResolveCpuExecutionOptions(ud->work_ctx);
    static const std::vector<int> empty_trace;
    const densecore::CpuBatchTraceView trace{active_batch ? active_batch->seq_id : empty_trace,
                                             active_batch ? active_batch->n_past : empty_trace};
    ud->backend->ForwardMoE(execution_options, ud->model, ud->layer, ud->layer_idx, active_batch ? &trace : nullptr,
                            t_input, routing, ud->experts, ud->n_experts, &t_output, moe_profile_ptr);
    if (collect_profile) {
        AddQwen36ProfileNs(ud->profile->moe_forward_ns,
                           static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - profile_begin)
                                                     .count()));
        AddQwen36ProfileNs(ud->profile->moe_route_ns, moe_profile.route_ns);
        AddQwen36ProfileNs(ud->profile->moe_reorder_ns, moe_profile.reorder_ns);
        AddQwen36ProfileNs(ud->profile->moe_expert_ns, moe_profile.expert_ns);
        AddQwen36ProfileNs(ud->profile->moe_reduce_ns, moe_profile.reduce_ns);
        AddQwen36ProfileNs(ud->profile->moe_w1w3_ns, moe_profile.w1w3_ns);
        AddQwen36ProfileNs(ud->profile->moe_w2_ns, moe_profile.w2_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_ns, moe_profile.rowblock_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_w1w3_ns, moe_profile.rowblock_w1w3_ns);
        AddQwen36ProfileNs(ud->profile->moe_rowblock_w2_ns, moe_profile.rowblock_w2_ns);
        ud->profile->moe_decode_scratch_reused.fetch_add(moe_profile.decode_scratch_reused, std::memory_order_relaxed);
        ud->profile->moe_decode_allocations_avoided.fetch_add(moe_profile.decode_allocations_avoided,
                                                              std::memory_order_relaxed);
        SetQwen36ProfileMax(ud->profile->moe_rowblock_used, moe_profile.rowblock_used);
        SetQwen36ProfileMax(ud->profile->moe_rowblock_tasks, moe_profile.rowblock_tasks);
    }
    if (debug_stage_timing) {
        const auto backend_end = std::chrono::steady_clock::now();
        const auto route_us = std::chrono::duration_cast<std::chrono::microseconds>(route_end - route_begin).count();
        const auto scheduler_us =
            std::chrono::duration_cast<std::chrono::microseconds>(scheduler_end - route_end).count();
        const auto backend_us =
            std::chrono::duration_cast<std::chrono::microseconds>(backend_end - scheduler_end).count();
        std::fprintf(stderr,
                     "[MOE_STAGE] layer=%d batch=%d top_k=%d assignments=%zu route_us=%lld scheduler_us=%lld "
                     "backend_us=%lld\n",
                     ud->layer_idx, routing.batch_size, routing.top_k, routing.expert_ids.size(),
                     static_cast<long long>(route_us), static_cast<long long>(scheduler_us),
                     static_cast<long long>(backend_us));
    }
    DumpMoEOutputTrace(dst, ud);
    if (IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> moe_out_count{0};
        const int call_id = moe_out_count.fetch_add(1);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_OUT", dst);
        }
    }
}
