#include "densecore/runtime/inference.h"

#include "densecore/backend/flash_attention.h"
#include "densecore/backend/hardware_topology.h"  // For compute thread affinity
#include "densecore/backend/matmul_backend.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/transformer_graph_builder.h"  // Strategy Pattern for graph building
#include "densecore/runtime/optimization_bridge.h"       // Runtime SIMD dispatch
#include "ggml-cpu.h"                                    // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"                                        // Required for ggml_tensor definition
#include "models/model_inference_policy.h"
#include "runtime/batched_activation_pack_cache.h"
#include "runtime/inference_types_internal.h"  // Shared internal types
#include "runtime/qwen36_gateup_rowpair.h"

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
extern "C" {
void ggml_gemv_q4_K_8x4_q8_K(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);
}
#endif

#include "backend/cpu_moe_execution.h"
#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "densecore/memory/kv_cache.h"  // Added for KV cache
#include "densecore/memory/memory_pool.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_execution_contract.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/quantization/int4_types.h"  // For TensorInt4
#include "densecore/runtime/dtype_utils.h"      // For GgmlTypeToDType
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/scheduler.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/kernel_caps.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/attention/callback_ops.h"
#include "llm/attention/diagnostics.h"
#include "llm/attention/exec.h"
#include "llm/attention/internal.h"
#include "llm/config/runtime_config.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/planning.h"
#include "llm/matmul/arm_m4.h"
#include "llm/matmul/diagnostics.h"
#include "llm/matmul/execution_policy.h"
#include "llm/matmul/graph_ops.h"
#include "llm/matmul/kquant_batched_kernels.h"
#include "llm/matmul/q6_small_batch.h"
#include "llm/matmul/q8_small_batch.h"
#include "llm/models/common/family_internal.h"
#include "llm/runtime/cpu_execution.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/spin_wait.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/work_context.h"
#include "models/gemma4_packed_expert_layout.h"
#include "runtime/kernel_admission.h"
#include "runtime/runtime_env.h"

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

namespace densecore {
bool RunQ5KRepackedMoEFusedSwiGLURawProjection(CpuBackend* backend, const void* gate_weight_ptr,
                                               const void* up_weight_ptr, const float* input_data,
                                               const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                               int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                               bool allow_parallel);
bool RunQ4KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ6KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ6KRepackedMoEProjectionCached(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                       size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                       int64_t input_cols, int numa_node, bool allow_parallel,
                                       std::shared_ptr<void>* packed_cache);
bool RunQ4KRepackedMoEFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                            const float* input_data, const uint8_t* qinput_data,
                                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                            int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ4KPrepackedMoEFusedSwiGLUTileRange(const void* fused_weight_ptr, const uint8_t* qinput_data,
                                            float* output_data, int64_t cols, int64_t input_cols, int tile_start,
                                            int tile_end);
bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedProjection(CpuBackend* backend, int ggml_type_id, const void* weight_ptr,
                                      const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                      int64_t N, int64_t K, int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedFusedSwiGLU(CpuBackend* backend, int ggml_type_id, const void* gate_weight_ptr,
                                       const void* up_weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
                                       float* out_data, int64_t M, int64_t N, int64_t K, int numa_node,
                                       bool allow_parallel);
}  // namespace densecore


#include "llm/graph/construction_ops.h"
#include "llm/matmul/kquant_batched_kernels.h"
#include "llm/matmul/q8_kernels.h"
#include "llm/matmul/quant_cache.h"
#include "llm/moe/native.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"


#include "llm/moe/native_internal.h"

namespace densecore::llm::graph::detail {
using densecore::env::ParseBoolEnv;
using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using densecore::env::RuntimeToggleMode;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;

constexpr const char* kGemma4RouterPerExpertScaleKey = "gemma4.router.per_expert_scale";
constexpr const char* kGemma4PackedGateUpExpertsWeightKey = "ffn_gate_up_exps.weight";
constexpr const char* kGemma4PackedGateUpExpertsKey = "ffn_gate_up_exps";
constexpr const char* kGemma4PackedDownExpertsWeightKey = "ffn_down_exps.weight";
constexpr const char* kGemma4PackedDownExpertsKey = "ffn_down_exps";
constexpr const char* kGemma4PackedDownExpertsScaleKey = "ffn_down_exps.scale";

bool IsDebugGemma4NativeMoEPrefillEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_GEMMA4_NATIVE_MOE_PREFILL");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsGemma4NativeMoEGraphEnabled() {
    return true;
}

bool Gemma4NativeMoEPrefillKernelSupported() {
    // C4 validation showed both candidate native prefill variants are not
    // promotable yet: full gate/up+down passes QA but regresses prefill, and
    // gate/up-only also regresses the default server path. Keep this fail-closed
    // until the prefill work moves to a batched/AMX-grade implementation.
    // This is deliberately not controlled by a benchmark env knob; once a faster
    // path passes QA and throughput gates it should be promoted as the default.
    return false;
}

bool IsGemma4NativeMoEPrefillEnabled() {
    return Gemma4NativeMoEPrefillKernelSupported();
}

struct Gemma4PackedMoERoots {
    ggml_tensor* gate_up = nullptr;
    ggml_tensor* down = nullptr;
    ggml_tensor* down_scale = nullptr;
    densecore::gemma4::PackedExpertLayout layout{};
};

bool ResolveGemma4PackedMoERoots(TransformerLayer* layer, Gemma4PackedMoERoots* roots) {
    if (!layer || !roots) {
        return false;
    }
    roots->gate_up = GetLayerTensorAny(layer, {kGemma4PackedGateUpExpertsWeightKey, kGemma4PackedGateUpExpertsKey});
    roots->down = GetLayerTensorAny(layer, {kGemma4PackedDownExpertsWeightKey, kGemma4PackedDownExpertsKey});
    roots->down_scale = GetLayerTensorAny(layer, {kGemma4PackedDownExpertsScaleKey});
    if (!roots->gate_up || !roots->down || !roots->down_scale) {
        return false;
    }
    std::string reason;
    if (!densecore::gemma4::InferPackedExpertLayout(roots->gate_up, roots->down, &roots->layout, &reason)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoE] packed root rejected: %s\n", reason.c_str());
        }
        return false;
    }
    return roots->layout.num_experts > 0 && roots->layout.hidden_dim > 0 && roots->layout.intermediate_dim > 0;
}

ggml_tensor* BuildGemma4PackedGateOrUp3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots, bool up_projection) {
    ggml_tensor* root = roots.gate_up;
    const auto& layout = roots.layout;
    if (!ctx || !root) {
        return nullptr;
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::RowStacked3D) {
        const size_t offset =
            up_projection ? static_cast<size_t>(layout.intermediate_dim) * static_cast<size_t>(root->nb[1]) : 0;
        return ggml_view_3d(ctx, root, layout.hidden_dim, layout.intermediate_dim, layout.num_experts, root->nb[1],
                            root->nb[2], offset);
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::PlaneSeparated4D) {
        const size_t offset = up_projection ? static_cast<size_t>(root->nb[2]) : 0;
        return ggml_view_3d(ctx, root, layout.hidden_dim, layout.intermediate_dim, layout.num_experts, root->nb[1],
                            root->nb[3], offset);
    }
    return nullptr;
}

ggml_tensor* BuildGemma4PackedGateUpMerged3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* root = roots.gate_up;
    const auto& layout = roots.layout;
    if (!ctx || !root) {
        return nullptr;
    }
    const int64_t merged_rows = layout.intermediate_dim * 2;
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::RowStacked3D) {
        return ggml_view_3d(ctx, root, layout.hidden_dim, merged_rows, layout.num_experts, root->nb[1], root->nb[2], 0);
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::PlaneSeparated4D &&
        root->nb[2] == root->nb[1] * layout.intermediate_dim) {
        return ggml_view_3d(ctx, root, layout.hidden_dim, merged_rows, layout.num_experts, root->nb[1], root->nb[3], 0);
    }
    return nullptr;
}

ggml_tensor* BuildGemma4PackedDown3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* root = roots.down;
    const auto& layout = roots.layout;
    if (!ctx || !root || layout.down_expert_axis < 0) {
        return nullptr;
    }
    return ggml_view_3d(ctx, root, layout.intermediate_dim, layout.hidden_dim, layout.num_experts, root->nb[1],
                        root->nb[layout.down_expert_axis], 0);
}

ggml_tensor* BuildGemma4PackedDownScaleRows(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* scale = roots.down_scale;
    if (!ctx || !scale || scale->type != GGML_TYPE_F32 || scale->ne[0] != roots.layout.num_experts) {
        return nullptr;
    }
    return ggml_reshape_2d(ctx, scale, 1, roots.layout.num_experts);
}

struct Gemma4GateUpQ4KPrefillUserData {
    int64_t hidden_dim = 0;
    int64_t intermediate_dim = 0;
    int64_t top_k = 0;
    int64_t n_tokens = 0;
    int64_t n_experts = 0;
    int64_t max_assignments = 0;
    int64_t max_batches = 0;
    int32_t* expert_offsets = nullptr;
    int32_t* expert_cursors = nullptr;
    int32_t* assignment_tokens = nullptr;
    int32_t* assignment_slots = nullptr;
    int32_t* batch_experts = nullptr;
    int32_t* batch_starts = nullptr;
    densecore::CpuBackend* numa_backend = nullptr;
    const TransformerLayer* numa_layer = nullptr;
    bool numa_sticky_enabled = false;
    std::atomic<int64_t> total_batches{0};
    std::atomic<int64_t> next_batch{0};
    std::atomic<uint64_t> epoch{0};
    std::atomic<uint64_t> ready_epoch{0};
    std::atomic<int> failure_count{0};
    std::atomic<const char*> first_failure_reason{nullptr};
    std::atomic<bool> used_recorded{false};
    static constexpr int kMaxTasks = 128;
    uint64_t task_epoch[kMaxTasks] = {};
};

static void SetGemma4NativeMoENumaContext(Gemma4GateUpQ4KPrefillUserData* ud, densecore::CpuBackend* backend,
                                          const TransformerLayer* layer) {
    if (!ud) {
        return;
    }
    ud->numa_backend = backend;
    ud->numa_layer = layer;
    ud->numa_sticky_enabled = NativeMoEHasVerifiedNumaPlacement(backend, layer);
}

constexpr int kGemma4GateUpPrefillRowsPerBatch = 16;

static float Gemma4GeluTanh(float x) {
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

static inline void Gemma4ApplyGEGLU(float* out, const float* gate, const float* up, int cols) {
    for (int c = 0; c < cols; ++c) {
        const float value = Gemma4GeluTanh(gate[c]) * up[c];
        out[c] = std::isfinite(value) ? value : 0.0f;
    }
}

static bool Gemma4QuantizedRowDot(ggml_type weight_type, const void* weight_row, const uint8_t* quant_input,
                                  int64_t cols, float* out) {
    if (!weight_row || !quant_input || !out || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    *out = 0.0f;
    traits->vec_dot(static_cast<int>(cols), out, 0, weight_row, 0, quant_input, 0, 1);
    if (!std::isfinite(*out)) {
        *out = 0.0f;
    }
    return true;
}

static bool Gemma4QuantizedRowDotInputType(ggml_type weight_type, int64_t cols, ggml_type* input_quant_type,
                                           size_t* input_quant_row_bytes) {
    if (!input_quant_type || !input_quant_row_bytes || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    const ggml_type quant_type = traits->vec_dot_type;
    const ggml_type_traits_cpu* input_traits = ggml_get_type_traits_cpu(quant_type);
    const size_t row_bytes = ggml_row_size(quant_type, cols);
    if (!input_traits || !input_traits->from_float || row_bytes == 0) {
        return false;
    }
    *input_quant_type = quant_type;
    *input_quant_row_bytes = row_bytes;
    return true;
}

static bool Gemma4CanUseDequantizedF32RowDot(ggml_type weight_type, int64_t cols) {
    if (cols <= 0 || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(weight_type);
    return traits && traits->to_float && ggml_row_size(weight_type, cols) > 0;
}

static bool Gemma4DequantizedF32RowDot(ggml_type weight_type, const void* weight_row, const float* input, int64_t cols,
                                       float* out) {
    if (!weight_row || !input || !out || !Gemma4CanUseDequantizedF32RowDot(weight_type, cols)) {
        return false;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(weight_type);
    thread_local std::vector<float> weight_f32;
    weight_f32.resize(static_cast<size_t>(cols));
    traits->to_float(weight_row, weight_f32.data(), cols);
    double acc = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        const float w = weight_f32[static_cast<size_t>(i)];
        const float x = input[static_cast<size_t>(i)];
        if (std::isfinite(w) && std::isfinite(x)) {
            acc += static_cast<double>(w) * static_cast<double>(x);
        }
    }
    *out = std::isfinite(acc) ? static_cast<float>(acc) : 0.0f;
    return true;
}

static void ZeroGemma4PrefillF32Tensor(ggml_tensor* tensor) {
    if (!tensor || !tensor->data || tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0) {
        return;
    }
    char* base = static_cast<char*>(tensor->data);
    const size_t row_bytes = static_cast<size_t>(tensor->ne[0]) * sizeof(float);
    for (int64_t z = 0; z < tensor->ne[2]; ++z) {
        for (int64_t y = 0; y < tensor->ne[1]; ++y) {
            std::memset(base + static_cast<size_t>(y) * tensor->nb[1] + static_cast<size_t>(z) * tensor->nb[2], 0,
                        row_bytes);
        }
    }
}

static Gemma4GateUpQ4KPrefillUserData* AllocateGemma4GateUpQ4KPrefillUserData(ggml_context* ctx, int64_t hidden_dim,
                                                                              int64_t intermediate_dim, int64_t top_k,
                                                                              int64_t n_tokens, int64_t n_experts) {
    if (!ctx || hidden_dim <= 0 || intermediate_dim <= 0 || top_k <= 0 || n_tokens <= 0 || n_experts <= 0) {
        return nullptr;
    }
    const int64_t max_assignments = top_k * n_tokens;
    const int64_t max_batches = max_assignments;
    if (max_assignments <= 0 || max_assignments > std::numeric_limits<int32_t>::max() ||
        n_experts + 1 > std::numeric_limits<int32_t>::max()) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx)) {
        // ggml no-alloc graph construction stores the userdata pointer in the
        // custom op. A thread-local scratch object is unsafe because later graph
        // builds (for example decode n_tokens=1) overwrite it before the prefill
        // graph executes. The native prefill path is diagnostic/opt-in, so use a
        // stable process-lifetime sidecar here until this path is promoted into a
        // graph-owned allocation contract.
        auto* ud = new Gemma4GateUpQ4KPrefillUserData();
        ud->expert_offsets = new int32_t[static_cast<size_t>(n_experts + 1)]();
        ud->expert_cursors = new int32_t[static_cast<size_t>(n_experts)]();
        ud->assignment_tokens = new int32_t[static_cast<size_t>(max_assignments)]();
        ud->assignment_slots = new int32_t[static_cast<size_t>(max_assignments)]();
        ud->batch_experts = new int32_t[static_cast<size_t>(max_batches)]();
        ud->batch_starts = new int32_t[static_cast<size_t>(max_batches)]();
        ud->hidden_dim = hidden_dim;
        ud->intermediate_dim = intermediate_dim;
        ud->top_k = top_k;
        ud->n_tokens = n_tokens;
        ud->n_experts = n_experts;
        ud->max_assignments = max_assignments;
        ud->max_batches = max_batches;
        return ud;
    }

    ggml_tensor* ud_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Gemma4GateUpQ4KPrefillUserData));
    ggml_tensor* offsets_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_experts + 1);
    ggml_tensor* cursors_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_experts);
    ggml_tensor* tokens_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_assignments);
    ggml_tensor* slots_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_assignments);
    ggml_tensor* batch_experts_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_batches);
    ggml_tensor* batch_starts_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_batches);
    if (!ud_storage || !ud_storage->data || !offsets_storage || !offsets_storage->data || !cursors_storage ||
        !cursors_storage->data || !tokens_storage || !tokens_storage->data || !slots_storage || !slots_storage->data ||
        !batch_experts_storage || !batch_experts_storage->data || !batch_starts_storage ||
        !batch_starts_storage->data) {
        return nullptr;
    }

    auto* ud = new (ud_storage->data) Gemma4GateUpQ4KPrefillUserData();
    ud->hidden_dim = hidden_dim;
    ud->intermediate_dim = intermediate_dim;
    ud->top_k = top_k;
    ud->n_tokens = n_tokens;
    ud->n_experts = n_experts;
    ud->max_assignments = max_assignments;
    ud->max_batches = max_batches;
    ud->expert_offsets = static_cast<int32_t*>(offsets_storage->data);
    ud->expert_cursors = static_cast<int32_t*>(cursors_storage->data);
    ud->assignment_tokens = static_cast<int32_t*>(tokens_storage->data);
    ud->assignment_slots = static_cast<int32_t*>(slots_storage->data);
    ud->batch_experts = static_cast<int32_t*>(batch_experts_storage->data);
    ud->batch_starts = static_cast<int32_t*>(batch_starts_storage->data);
    return ud;
}

static void MarkGemma4NativeMoEPrefillFailure(Gemma4GateUpQ4KPrefillUserData* ud, const char* reason, const char* stage,
                                              int ith) {
    if (!ud) {
        return;
    }
    ud->failure_count.fetch_add(1, std::memory_order_relaxed);
    const char* expected = nullptr;
    ud->first_failure_reason.compare_exchange_strong(expected, reason, std::memory_order_relaxed);
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] %s failed reason=%s\n", stage ? stage : "native_prefill",
                     reason ? reason : "unknown");
    }
}

static bool Gemma4PrefillInputShapeOk(const ggml_tensor* input, int64_t hidden_dim, int64_t top_k, int64_t n_tokens) {
    if (!input || input->type != GGML_TYPE_F32 || input->ne[0] != hidden_dim || top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    if (input->ne[1] == n_tokens) {
        return true;
    }
    if (input->ne[2] == n_tokens && (input->ne[1] == top_k || input->ne[1] == 1)) {
        return true;
    }
    return false;
}

static bool Gemma4PrefillAssignmentShapeOk(const ggml_tensor* input, int64_t top_k, int64_t n_tokens) {
    if (!input || input->type != GGML_TYPE_F32 || top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    if (input->ne[1] == n_tokens) {
        return true;
    }
    if (input->ne[2] == n_tokens && (input->ne[1] == top_k || input->ne[1] == 1)) {
        return true;
    }
    return false;
}

static const float* Gemma4PrefillInputRow(const ggml_tensor* input, int32_t token, int32_t slot,
                                          const Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!input || !input->data || !ud || token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k ||
        !Gemma4PrefillInputShapeOk(input, ud->hidden_dim, ud->top_k, ud->n_tokens)) {
        return nullptr;
    }
    const char* base = static_cast<const char*>(input->data);
    if (input->ne[1] == ud->n_tokens) {
        return reinterpret_cast<const float*>(base + static_cast<size_t>(token) * input->nb[1]);
    }
    const int32_t source_slot = input->ne[1] == 1 ? 0 : slot;
    return reinterpret_cast<const float*>(base + static_cast<size_t>(source_slot) * input->nb[1] +
                                          static_cast<size_t>(token) * input->nb[2]);
}

static bool PrepareGemma4GateUpQ4KPrefillBatches(Gemma4GateUpQ4KPrefillUserData* ud, const ggml_tensor* input,
                                                 const ggml_tensor* selected_experts, int ith, int nth) {
    if (!ud || !input || !selected_experts || !input->data || !selected_experts->data || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || nth <= 0 || ith < 0 || ith >= nth ||
        nth > Gemma4GateUpQ4KPrefillUserData::kMaxTasks ||
        !Gemma4PrefillAssignmentShapeOk(input, ud->top_k, ud->n_tokens) || selected_experts->ne[0] != ud->top_k ||
        selected_experts->ne[1] != ud->n_tokens || !ud->expert_offsets || !ud->expert_cursors ||
        !ud->assignment_tokens || !ud->assignment_slots || !ud->batch_experts || !ud->batch_starts) {
        return false;
    }

    if (ith == 0) {
        const uint64_t epoch = ud->epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
        ud->task_epoch[0] = epoch;
        std::fill_n(ud->expert_offsets, static_cast<size_t>(ud->n_experts + 1), int32_t{0});
        for (int64_t token = 0; token < ud->n_tokens; ++token) {
            for (int64_t k = 0; k < ud->top_k; ++k) {
                const int32_t expert =
                    *reinterpret_cast<const int32_t*>(static_cast<const char*>(selected_experts->data) +
                                                      static_cast<size_t>(k) * selected_experts->nb[0] +
                                                      static_cast<size_t>(token) * selected_experts->nb[1]);
                if (expert >= 0 && expert < ud->n_experts) {
                    ++ud->expert_offsets[expert + 1];
                }
            }
        }
        for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
            ud->expert_offsets[expert + 1] += ud->expert_offsets[expert];
            ud->expert_cursors[expert] = ud->expert_offsets[expert];
        }
        for (int64_t token = 0; token < ud->n_tokens; ++token) {
            for (int64_t k = 0; k < ud->top_k; ++k) {
                const int32_t expert =
                    *reinterpret_cast<const int32_t*>(static_cast<const char*>(selected_experts->data) +
                                                      static_cast<size_t>(k) * selected_experts->nb[0] +
                                                      static_cast<size_t>(token) * selected_experts->nb[1]);
                if (expert >= 0 && expert < ud->n_experts) {
                    const int32_t pos = ud->expert_cursors[expert]++;
                    if (pos >= 0 && pos < ud->max_assignments) {
                        ud->assignment_tokens[pos] = static_cast<int32_t>(token);
                        ud->assignment_slots[pos] = static_cast<int32_t>(k);
                    }
                }
            }
        }
        int64_t batches = 0;
        for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
            for (int32_t start = ud->expert_offsets[expert]; start < ud->expert_offsets[expert + 1];
                 start += kGemma4GateUpPrefillRowsPerBatch) {
                if (batches >= ud->max_batches) {
                    break;
                }
                ud->batch_experts[batches] = static_cast<int32_t>(expert);
                ud->batch_starts[batches] = start;
                ++batches;
            }
        }
        ud->total_batches.store(batches, std::memory_order_release);
        ud->next_batch.store(0, std::memory_order_release);
        ud->ready_epoch.store(epoch, std::memory_order_release);
        if (IsDebugGemma4NativeMoEPrefillEnabled()) {
            int64_t assignments = 0;
            int64_t active_experts = 0;
            for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
                const int32_t count = ud->expert_offsets[expert + 1] - ud->expert_offsets[expert];
                assignments += count;
                active_experts += count > 0 ? 1 : 0;
            }
            std::fprintf(
                stderr,
                "[Gemma4NativeMoEPrefillDebug] prepare tokens=%lld top_k=%lld experts=%lld assignments=%lld "
                "active_experts=%lld batches=%lld selected_ne=[%lld,%lld,%lld] selected_nb=[%lld,%lld,%lld]\n",
                static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                static_cast<long long>(ud->n_experts), static_cast<long long>(assignments),
                static_cast<long long>(active_experts), static_cast<long long>(batches),
                static_cast<long long>(selected_experts->ne[0]), static_cast<long long>(selected_experts->ne[1]),
                static_cast<long long>(selected_experts->ne[2]), static_cast<long long>(selected_experts->nb[0]),
                static_cast<long long>(selected_experts->nb[1]), static_cast<long long>(selected_experts->nb[2]));
        }
        return true;
    }

    const uint64_t last_epoch = ud->task_epoch[ith];
    uint64_t epoch = ud->epoch.load(std::memory_order_acquire);
    while (epoch == last_epoch) {
        std::this_thread::yield();
        epoch = ud->epoch.load(std::memory_order_acquire);
    }
    while (ud->ready_epoch.load(std::memory_order_acquire) != epoch) {
        std::this_thread::yield();
    }
    ud->task_epoch[ith] = epoch;
    return true;
}

static bool RunGemma4GateUpQ4KPrefillFusedGEGLU(ggml_tensor* dst, const ggml_tensor* gate_up_exps,
                                                const ggml_tensor* input, const ggml_tensor* selected_experts, int ith,
                                                int nth, Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !gate_up_exps || !input || !selected_experts || !ud || !dst->data || !gate_up_exps->data ||
        !input->data || nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_missing_runtime_data", "gateup", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || gate_up_exps->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->intermediate_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || !Gemma4PrefillInputShapeOk(input, ud->hidden_dim, ud->top_k, ud->n_tokens) ||
        gate_up_exps->ne[0] != ud->hidden_dim || gate_up_exps->ne[1] != 2 * ud->intermediate_dim ||
        gate_up_exps->ne[2] != ud->n_experts || (ud->hidden_dim % QK_K) != 0 || (ud->intermediate_dim % 8) != 0 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] gateup shape_mismatch dst_ne=[%lld,%lld,%lld] "
                         "input_ne=[%lld,%lld,%lld] gate_ne=[%lld,%lld,%lld] selected_ne=[%lld,%lld,%lld] "
                         "ud=[hidden=%lld,intermediate=%lld,top_k=%lld,tokens=%lld,experts=%lld]\n",
                         static_cast<long long>(dst->ne[0]), static_cast<long long>(dst->ne[1]),
                         static_cast<long long>(dst->ne[2]), static_cast<long long>(input->ne[0]),
                         static_cast<long long>(input->ne[1]), static_cast<long long>(input->ne[2]),
                         static_cast<long long>(gate_up_exps->ne[0]), static_cast<long long>(gate_up_exps->ne[1]),
                         static_cast<long long>(gate_up_exps->ne[2]), static_cast<long long>(selected_experts->ne[0]),
                         static_cast<long long>(selected_experts->ne[1]),
                         static_cast<long long>(selected_experts->ne[2]), static_cast<long long>(ud->hidden_dim),
                         static_cast<long long>(ud->intermediate_dim), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->n_experts));
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_shape_or_type_mismatch", "gateup", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, input, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(
                stderr,
                "[Gemma4NativeMoEPrefillDebug] gateup prepare_failed dst_data=%p gate_data=%p input_data=%p "
                "selected_data=%p dst_ne=[%lld,%lld,%lld] gate_ne=[%lld,%lld,%lld] input_ne=[%lld,%lld,%lld] "
                "selected_ne=[%lld,%lld,%lld]\n",
                dst ? dst->data : nullptr, gate_up_exps ? gate_up_exps->data : nullptr, input ? input->data : nullptr,
                selected_experts ? selected_experts->data : nullptr, dst ? static_cast<long long>(dst->ne[0]) : -1LL,
                dst ? static_cast<long long>(dst->ne[1]) : -1LL, dst ? static_cast<long long>(dst->ne[2]) : -1LL,
                gate_up_exps ? static_cast<long long>(gate_up_exps->ne[0]) : -1LL,
                gate_up_exps ? static_cast<long long>(gate_up_exps->ne[1]) : -1LL,
                gate_up_exps ? static_cast<long long>(gate_up_exps->ne[2]) : -1LL,
                input ? static_cast<long long>(input->ne[0]) : -1LL,
                input ? static_cast<long long>(input->ne[1]) : -1LL,
                input ? static_cast<long long>(input->ne[2]) : -1LL,
                selected_experts ? static_cast<long long>(selected_experts->ne[0]) : -1LL,
                selected_experts ? static_cast<long long>(selected_experts->ne[1]) : -1LL,
                selected_experts ? static_cast<long long>(selected_experts->ne[2]) : -1LL);
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_prepare_failed", "gateup", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> gateup_debug_count{0};
        const int debug_idx = gateup_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] gateup_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int tile_count = static_cast<int>(ud->intermediate_dim / 8);
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(gate_up_exps->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, ud->hidden_dim);
    if (q4_row_bytes == 0 || static_cast<size_t>(gate_up_exps->nb[1]) < q4_row_bytes ||
        static_cast<size_t>(gate_up_exps->nb[2]) <
            static_cast<size_t>(2 * ud->intermediate_dim - 1) * static_cast<size_t>(gate_up_exps->nb[1]) +
                q4_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_row_stride_mismatch", "gateup", ith);
        return false;
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float || !q4_traits || q4_traits->vec_dot_type != GGML_TYPE_Q8_K) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_q4k_q8k_traits_unavailable", "gateup", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_tail_buf;
    q8_tail_buf.resize(ggml_row_size(GGML_TYPE_Q8_K, ud->hidden_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base = weight_base + static_cast<size_t>(expert) * static_cast<size_t>(gate_up_exps->nb[2]);

        bool batch_ok = true;
        const auto run_batch = [&] {
            std::array<float, 8> gate_tail{};
            std::array<float, 8> up_tail{};
            for (int r = 0; r < rows; ++r) {
                const int32_t token = ud->assignment_tokens[start + r];
                const int32_t slot = ud->assignment_slots[start + r];
                if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                    continue;
                }
                const float* src = Gemma4PrefillInputRow(input, token, slot, ud);
                if (!src) {
                    MarkGemma4NativeMoEPrefillFailure(ud, "gateup_input_row_unavailable", "gateup", ith);
                    batch_ok = false;
                    return;
                }
                q8_traits->from_float(src, q8_tail_buf.data(), ud->hidden_dim);
                for (int tile = 0; tile < tile_count; ++tile) {
                    for (int lane = 0; lane < 8; ++lane) {
                        const int64_t row = static_cast<int64_t>(tile) * 8 + lane;
                        const void* gate_row =
                            expert_base + static_cast<size_t>(row) * static_cast<size_t>(gate_up_exps->nb[1]);
                        const void* up_row = expert_base + static_cast<size_t>(ud->intermediate_dim + row) *
                                                               static_cast<size_t>(gate_up_exps->nb[1]);
                        if (!Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, gate_row, q8_tail_buf.data(), ud->hidden_dim,
                                                   &gate_tail[lane]) ||
                            !Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, up_row, q8_tail_buf.data(), ud->hidden_dim,
                                                   &up_tail[lane])) {
                            MarkGemma4NativeMoEPrefillFailure(ud, "gateup_row_dot_failed", "gateup", ith);
                            batch_ok = false;
                            return;
                        }
                    }
                    float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(tile) * 8 * dst->nb[0] +
                                                          static_cast<size_t>(slot) * dst->nb[1] +
                                                          static_cast<size_t>(token) * dst->nb[2]);
                    Gemma4ApplyGEGLU(out, gate_tail.data(), up_tail.data(), 8);
                    if (ith == 0 && batch == 0 && r == 0 && tile == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                        float max_abs = 0.0f;
                        for (float v : gate_tail) max_abs = std::max(max_abs, std::fabs(v));
                        for (float v : up_tail) max_abs = std::max(max_abs, std::fabs(v));
                        std::fprintf(stderr,
                                     "[Gemma4NativeMoEPrefillDebug] gateup_first token=%d slot=%d expert=%d max_abs=%g "
                                     "out0=%g out1=%g\n",
                                     token, slot, expert, max_abs, out[0], out[1]);
                    }
                }
            }
        };
        const int raw_node = ud->numa_sticky_enabled && ud->numa_backend && ud->numa_layer
                                 ? ud->numa_backend->GetExpertNumaNode(ud->numa_layer, expert)
                                 : -1;
        const int node = (raw_node >= 0 && raw_node < ud->numa_backend->GetNumaNodeCount()) ? raw_node : -1;
        RecordNativeMoENumaDispatch(node);
        if (node >= 0) {
            ud->numa_backend->RunOnNumaNode(node, run_batch);
        } else {
            run_batch();
        }
        if (!batch_ok) {
            return false;
        }
    }
    return true;
}

static void cb_gemma4_gateup_q4k_prefill_geglu(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4GateUpQ4KPrefillFusedGEGLU(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                                        dst ? dst->src[2] : nullptr, ith, nth, ud);
    if (ith == 0 && ud && ud->numa_backend && ud->numa_layer && dst && dst->src[2] && dst->src[2]->data) {
        const ggml_tensor* selected = dst->src[2];
        std::vector<int> expert_ids;
        expert_ids.reserve(static_cast<size_t>(std::max<int64_t>(0, selected->ne[0] * selected->ne[1])));
        for (int64_t token = 0; token < selected->ne[1]; ++token) {
            for (int64_t k = 0; k < selected->ne[0]; ++k) {
                const int expert = *reinterpret_cast<const int32_t*>(static_cast<const char*>(selected->data) +
                                                                     static_cast<size_t>(k) * selected->nb[0] +
                                                                     static_cast<size_t>(token) * selected->nb[1]);
                if (expert >= 0 && expert < ud->n_experts) {
                    expert_ids.push_back(expert);
                }
            }
        }
        if (!expert_ids.empty()) {
            ud->numa_backend->RecordExpertAccess(ud->numa_layer, expert_ids.data(),
                                                 static_cast<int>(expert_ids.size()));
        }
    }
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Prefill) {
        RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), ns, 0, ns);
        if (!ok && ud) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                                 ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
        }
    }
    if (ith == 0 && GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
        RecordGemma4NativeFusedGateUpUsed(GetCurrentWorkContext());
        RecordGemma4DecodeNativeDecision(GetCurrentWorkContext(), /*candidate=*/true, /*used=*/true, nullptr,
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, ns,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/1,
                                         /*duplicate_work_detected=*/false);
    }
}

static bool RunGemma4DownQuantPrefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                      const ggml_tensor* selected_experts, int ith, int nth,
                                      Gemma4GateUpQ4KPrefillUserData* ud, ggml_type weight_type,
                                      const char* debug_label) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data || !hidden->data ||
        nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_missing_runtime_data", debug_label, ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != weight_type || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_shape_or_type_mismatch", debug_label, ith);
        return false;
    }
    ggml_type input_quant_type = GGML_TYPE_COUNT;
    size_t input_quant_row_bytes = 0;
    const bool use_vec_dot =
        Gemma4QuantizedRowDotInputType(weight_type, ud->intermediate_dim, &input_quant_type, &input_quant_row_bytes);
    const bool use_dequant_dot = !use_vec_dot && Gemma4CanUseDequantizedF32RowDot(weight_type, ud->intermediate_dim);
    if (!use_vec_dot && !use_dequant_dot) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_dot_traits_unavailable", debug_label, ith);
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(weight_type, ud->intermediate_dim);
    if (weight_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < weight_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + weight_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_row_stride_mismatch", debug_label, ith);
        return false;
    }
    const ggml_type_traits_cpu* input_traits = use_vec_dot ? ggml_get_type_traits_cpu(input_quant_type) : nullptr;
    if (use_vec_dot && (!input_traits || !input_traits->from_float)) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_input_traits_unavailable", debug_label, ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] %s prepare_failed\n",
                         debug_label ? debug_label : "down_quant");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_prepare_failed", debug_label, ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_quant_debug_count{0};
        const int debug_idx = down_quant_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] %s_execute[%d] dst=%s type=%s input_quant=%s tokens=%lld "
                         "top_k=%lld total_batches=%lld nth=%d\n",
                         debug_label ? debug_label : "down_quant", debug_idx, dst->name, ggml_type_name(weight_type),
                         use_vec_dot ? ggml_type_name(input_quant_type) : "f32_dequant_row_dot",
                         static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    thread_local std::vector<uint8_t> qrow;
    if (use_vec_dot) {
        qrow.resize(input_quant_row_bytes);
    }

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base = weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(slot) * dst->nb[1] +
                                                  static_cast<size_t>(token) * dst->nb[2]);
            if (use_vec_dot) {
                input_traits->from_float(src, qrow.data(), ud->intermediate_dim);
            }
            for (int64_t row = 0; row < ud->hidden_dim; ++row) {
                const void* down_row = expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                if (use_vec_dot) {
                    if (!Gemma4QuantizedRowDot(weight_type, down_row, qrow.data(), ud->intermediate_dim, out + row)) {
                        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_row_dot_failed", debug_label, ith);
                        return false;
                    }
                } else if (!Gemma4DequantizedF32RowDot(weight_type, down_row, src, ud->intermediate_dim, out + row)) {
                    MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_dequant_row_dot_failed", debug_label, ith);
                    return false;
                }
            }
            if (ith == 0 && batch == 0 && r == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                std::fprintf(stderr,
                             "[Gemma4NativeMoEPrefillDebug] %s_first token=%d slot=%d expert=%d out0=%g out1=%g\n",
                             debug_label ? debug_label : "down_quant", token, slot, expert, out[0], out[1]);
            }
        }
    }
    return true;
}

static bool RunGemma4DownQ4KPrefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                    const ggml_tensor* selected_experts, int ith, int nth,
                                    Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data || !hidden->data ||
        nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_missing_runtime_data", "down_q4k", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != GGML_TYPE_Q4_K || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts || (ud->intermediate_dim % QK_K) != 0 ||
        (ud->hidden_dim % 8) != 0 || dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_shape_or_type_mismatch", "down_q4k", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] down_q4k prepare_failed\n");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_prepare_failed", "down_q4k", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_q4k_debug_count{0};
        const int debug_idx = down_q4k_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] down_q4k_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int tile_count = static_cast<int>(ud->hidden_dim / 8);
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, ud->intermediate_dim);
    if (q4_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < q4_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + q4_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_row_stride_mismatch", "down_q4k", ith);
        return false;
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float || !q4_traits || q4_traits->vec_dot_type != GGML_TYPE_Q8_K) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_q8k_traits_unavailable", "down_q4k", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_tail_buf;
    q8_tail_buf.resize(ggml_row_size(GGML_TYPE_Q8_K, ud->intermediate_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base = weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);

        std::array<float, 8> down_tail{};
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            q8_traits->from_float(src, q8_tail_buf.data(), ud->intermediate_dim);
            for (int tile = 0; tile < tile_count; ++tile) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int64_t row = static_cast<int64_t>(tile) * 8 + lane;
                    const void* down_row =
                        expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                    if (!Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, down_row, q8_tail_buf.data(), ud->intermediate_dim,
                                               &down_tail[lane])) {
                        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_row_dot_failed", "down_q4k", ith);
                        return false;
                    }
                }
                float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(tile) * 8 * dst->nb[0] +
                                                      static_cast<size_t>(slot) * dst->nb[1] +
                                                      static_cast<size_t>(token) * dst->nb[2]);
                std::memcpy(out, down_tail.data(), sizeof(float) * 8);
                if (ith == 0 && batch == 0 && r == 0 && tile == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                    std::fprintf(stderr,
                                 "[Gemma4NativeMoEPrefillDebug] down_q4k_first token=%d slot=%d expert=%d out0=%g "
                                 "out1=%g\n",
                                 token, slot, expert, out[0], out[1]);
                }
            }
        }
    }
    return true;
}

static void cb_gemma4_down_q4k_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQ4KPrefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                            dst ? dst->src[2] : nullptr, ith, nth, ud);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud) {
        if (ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                                 /*replaced_mul_mat_id_ops=*/2, false);
        } else if (!ok) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                                 ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
        }
    }
}

static bool RunGemma4DownQ8_0Prefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                     const ggml_tensor* selected_experts, int ith, int nth,
                                     Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data || !hidden->data ||
        nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_missing_runtime_data", "down_q8_0", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != GGML_TYPE_Q8_0 || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts ||
        (ud->intermediate_dim % QK8_0) != 0 || (ud->hidden_dim % 4) != 0 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_shape_or_type_mismatch", "down_q8_0", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] down_q8_0 prepare_failed\n");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_prepare_failed", "down_q8_0", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_q8_debug_count{0};
        const int debug_idx = down_q8_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] down_q8_0_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ud->intermediate_dim);
    if (q8_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < q8_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + q8_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_row_stride_mismatch", "down_q8_0", ith);
        return false;
    }
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    const auto* weight_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!weight_traits || !weight_traits->vec_dot) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_weight_traits_unavailable", "down_q8_0", ith);
        return false;
    }
    const ggml_type input_quant_type = weight_traits->vec_dot_type;
    const auto* input_traits = ggml_get_type_traits_cpu(input_quant_type);
    if (!input_traits || !input_traits->from_float) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_input_traits_unavailable", "down_q8_0", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_row;
    q8_row.resize(ggml_row_size(input_quant_type, ud->intermediate_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base = weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            input_traits->from_float(src, q8_row.data(), ud->intermediate_dim);
            float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(slot) * dst->nb[1] +
                                                  static_cast<size_t>(token) * dst->nb[2]);
            for (int64_t row = 0; row < ud->hidden_dim; ++row) {
                const void* down_row = expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                if (!Gemma4QuantizedRowDot(GGML_TYPE_Q8_0, down_row, q8_row.data(), ud->intermediate_dim, out + row)) {
                    MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_row_dot_failed", "down_q8_0", ith);
                    return false;
                }
            }
            if (ith == 0 && batch == 0 && r == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                std::fprintf(stderr,
                             "[Gemma4NativeMoEPrefillDebug] down_q8_0_first token=%d slot=%d expert=%d out0=%g "
                             "out1=%g\n",
                             token, slot, expert, out[0], out[1]);
            }
        }
    }
    return true;
}

static void cb_gemma4_down_q8_0_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQ8_0Prefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                             dst ? dst->src[2] : nullptr, ith, nth, ud);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud && ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                             /*replaced_mul_mat_id_ops=*/2, false);
    } else if (ud && !ok) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                             ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
    }
}

static void cb_gemma4_down_q5_1_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQuantPrefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                              dst ? dst->src[2] : nullptr, ith, nth, ud, GGML_TYPE_Q5_1, "down_q5_1");
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud && ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                             /*replaced_mul_mat_id_ops=*/2, false);
    } else if (ud && !ok) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                             ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
    }
}

static bool CanUseGemma4GateUpQ4KFusedGEGLU(const TransformerModel* model, const ggml_tensor* gate_up_exps,
                                            const ggml_tensor* input, const ggml_tensor* selected_experts,
                                            int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !gate_up_exps || !input || !selected_experts || !model->arch_flags.is_gemma4) return false;
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (gate_up_exps->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    if (n_tokens <= 0 || top_k <= 0 || !Gemma4PrefillInputShapeOk(input, gate_up_exps->ne[0], top_k, n_tokens) ||
        gate_up_exps->ne[1] != 2 * intermediate_dim || gate_up_exps->ne[2] != n_experts ||
        selected_experts->ne[0] <= 0) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, input->ne[0]);
    if (row_bytes == 0 || gate_up_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    if ((input->ne[0] % QK_K) != 0 || (intermediate_dim % 8) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQuantPrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                         const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                         int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts,
                                         ggml_type weight_type) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != weight_type || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim || down_exps->ne[2] != n_experts ||
        selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    ggml_type input_quant_type = GGML_TYPE_COUNT;
    size_t input_quant_row_bytes = 0;
    if (!Gemma4QuantizedRowDotInputType(weight_type, intermediate_dim, &input_quant_type, &input_quant_row_bytes) &&
        !Gemma4CanUseDequantizedF32RowDot(weight_type, intermediate_dim)) {
        return false;
    }
    (void)input_quant_type;
    (void)input_quant_row_bytes;
    const size_t row_bytes = ggml_row_size(weight_type, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQ4KPrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                       const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                       int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != GGML_TYPE_Q4_K || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim || down_exps->ne[2] != n_experts ||
        selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    if ((intermediate_dim % QK_K) != 0 || (hidden_dim % 8) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQ8_0Prefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                        const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                        int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != GGML_TYPE_Q8_0 || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim || down_exps->ne[2] != n_experts ||
        selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] != row_bytes) return false;
    if ((intermediate_dim % QK8_0) != 0 || (hidden_dim % 4) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownNativePrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                          const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                          int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    return CanUseGemma4DownQ4KPrefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                      n_experts) ||
           CanUseGemma4DownQ8_0Prefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                       n_experts) ||
           CanUseGemma4DownQuantPrefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                        n_experts, GGML_TYPE_Q5_1);
}

static bool CanUseGemma4DownWeightedSumDecode(const TransformerModel* model, const ggml_tensor* down_exps,
                                              const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                              const ggml_tensor* weights, const char** reject_reason) {
    auto reject = [&](const char* reason) {
        if (reject_reason) {
            *reject_reason = reason;
        }
        return false;
    };
    if (reject_reason) {
        *reject_reason = nullptr;
    }
    if (!model || !down_exps || !hidden || !selected_experts || !weights) return reject("null_ptr");
    if (!model->arch_flags.is_gemma4) return reject("wrong_variant");
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return reject("dynamic_lora");
    if (down_exps->type != GGML_TYPE_Q8_0) return reject("unsupported_down_quant");
    if (hidden->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 || weights->type != GGML_TYPE_F32) {
        return reject("bad_tensor_type");
    }
    if (selected_experts->ne[0] <= 0 || selected_experts->ne[1] != 1 ||
        selected_experts->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (hidden->ne[1] != selected_experts->ne[0] || hidden->ne[2] != selected_experts->ne[1]) {
        return reject("hidden_shape");
    }
    if (weights->ne[0] != 1 || weights->ne[1] != selected_experts->ne[0] || weights->ne[2] != selected_experts->ne[1]) {
        return reject("weights_shape");
    }
    if (down_exps->ne[0] != hidden->ne[0] || down_exps->ne[2] <= 0) return reject("down_shape");
    const int64_t q8_block = ggml_blck_size(GGML_TYPE_Q8_0);
    if (q8_block <= 0 || (down_exps->ne[0] % q8_block) != 0) return reject("bad_q8_0_alignment");
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, down_exps->ne[0]);
    if (q8_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) != q8_row_bytes) {
        return reject("bad_q8_0_stride");
    }
    return true;
}

ggml_tensor* TryBuildGemma4NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k, densecore::CpuBackend* numa_backend) {
    if (!ctx || !gf || !model || !layer || !routed_input || !gate_logits || !model->arch_flags.is_gemma4 ||
        !IsGemma4NativeMoEGraphEnabled()) {
        return nullptr;
    }
    Gemma4PackedMoERoots roots;
    if (!ResolveGemma4PackedMoERoots(layer, &roots)) {
        return nullptr;
    }
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) {
        ggml_tensor* gate_up_repack = UseCpuRepackAliasIfAvailable(model, roots.gate_up);
        if (!gate_up_repack || gate_up_repack->type == roots.gate_up->type) {
            roots.gate_up = gate_up_repack;
        }
    }
    std::string repack_reason;
    if (!densecore::gemma4::InferPackedExpertLayout(roots.gate_up, roots.down, &roots.layout, &repack_reason)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoE] CPU_REPACK alias rejected: %s\n", repack_reason.c_str());
        }
        return nullptr;
    }
    const int64_t n_tokens = routed_input->ne[1];
    const int64_t n_embd = roots.layout.hidden_dim;
    const int64_t n_expert_used = std::max<int64_t>(1, std::min<int64_t>(top_k, roots.layout.num_experts));
    if (n_tokens <= 0 || routed_input->ne[0] != n_embd || gate_logits->ne[0] != roots.layout.num_experts ||
        gate_logits->ne[1] != n_tokens) {
        return nullptr;
    }
    ggml_tensor* gate_up_exps = BuildGemma4PackedGateUpMerged3DView(ctx, roots);
    if (gate_up_exps && gate_up_exps->view_src && gate_up_exps->view_src->buffer && !gate_up_exps->buffer) {
        ggml_backend_view_init(gate_up_exps);
    }
    ggml_tensor* gate_exps = nullptr;
    ggml_tensor* up_exps = nullptr;
    if (!gate_up_exps) {
        gate_exps = BuildGemma4PackedGateOrUp3DView(ctx, roots, /*up_projection=*/false);
        up_exps = BuildGemma4PackedGateOrUp3DView(ctx, roots, /*up_projection=*/true);
        if (gate_exps && gate_exps->view_src && gate_exps->view_src->buffer && !gate_exps->buffer) {
            ggml_backend_view_init(gate_exps);
        }
        if (up_exps && up_exps->view_src && up_exps->view_src->buffer && !up_exps->buffer) {
            ggml_backend_view_init(up_exps);
        }
    }
    ggml_tensor* down_exps = BuildGemma4PackedDown3DView(ctx, roots);
    if (down_exps && down_exps->view_src && down_exps->view_src->buffer && !down_exps->buffer) {
        ggml_backend_view_init(down_exps);
    }
    ggml_tensor* scale_rows = BuildGemma4PackedDownScaleRows(ctx, roots);
    if ((!gate_up_exps && (!gate_exps || !up_exps)) || !down_exps || !scale_rows) {
        return nullptr;
    }

    ggml_tensor* selected_experts = ggml_argsort_top_k(ctx, gate_logits, static_cast<int>(n_expert_used));
    ggml_set_name(selected_experts, "gemma4_native_moe_topk");
    ggml_tensor* weights =
        BuildMoETopKWeightsFromLogits(ctx, gate_logits, selected_experts, "gemma4_native_moe_topk_weights");
    if (!weights) {
        return nullptr;
    }

    scale_rows = ggml_repeat_4d(ctx, scale_rows, 1, roots.layout.num_experts, n_tokens, 1);
    ggml_tensor* selected_scales = ggml_get_rows(ctx, scale_rows, selected_experts);
    weights = ggml_mul(ctx, weights, selected_scales);
    ggml_set_name(weights, "gemma4_native_moe_weights");
    ggml_build_forward_expand(gf, weights);

    ggml_tensor* hidden = nullptr;
    // Do not build the full native gate/up+down prefill graph yet. The C4
    // shape-correct run passed QA but regressed prefill because the native
    // down-projection still performs per-row scalar dot work. Keep the proven
    // gate/up custom op below and leave down/weighted-sum on the maintained
    // graph path until a batched down kernel is available.

    ggml_tensor* cur3 = ggml_reshape_3d(ctx, routed_input, n_embd, 1, n_tokens);
    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    if (gate_up_exps) {
        if (CanUseGemma4GateUpQ4KFusedGEGLU(model, gate_up_exps, routed_input, selected_experts,
                                            roots.layout.intermediate_dim, roots.layout.num_experts)) {
            Gemma4GateUpQ4KPrefillUserData* gateup_ud = AllocateGemma4GateUpQ4KPrefillUserData(
                ctx, n_embd, roots.layout.intermediate_dim, n_expert_used, n_tokens, roots.layout.num_experts);
            if (gateup_ud) {
                SetGemma4NativeMoENumaContext(gateup_ud, numa_backend, layer);
                ggml_tensor* args[] = {gate_up_exps, routed_input, selected_experts};
                const int gateup_tasks = gateup_ud->numa_sticky_enabled ? 1 : GGML_N_TASKS_MAX;
                hidden = ggml_custom_4d(ctx, GGML_TYPE_F32, roots.layout.intermediate_dim, n_expert_used, n_tokens, 1,
                                        args, 3, cb_gemma4_gateup_q4k_prefill_geglu, gateup_tasks, gateup_ud);
                ggml_set_name(hidden, "gemma4_native_moe_gateup_q4k_prefill_geglu");
            }
        }
        if (!hidden) {
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_up_exps, cur3, selected_experts);
            ggml_set_name(gate_up, "gemma4_native_moe_gate_up");
            gate = ggml_view_3d(ctx, gate_up, roots.layout.intermediate_dim, n_expert_used, n_tokens, gate_up->nb[1],
                                gate_up->nb[2], 0);
            ggml_set_name(gate, "gemma4_native_moe_gate");
            up = ggml_view_3d(ctx, gate_up, roots.layout.intermediate_dim, n_expert_used, n_tokens, gate_up->nb[1],
                              gate_up->nb[2],
                              static_cast<size_t>(roots.layout.intermediate_dim) * static_cast<size_t>(gate_up->nb[0]));
            ggml_set_name(up, "gemma4_native_moe_up");
        }
    } else {
        up = ggml_mul_mat_id(ctx, up_exps, cur3, selected_experts);
        ggml_set_name(up, "gemma4_native_moe_up");
        gate = ggml_mul_mat_id(ctx, gate_exps, cur3, selected_experts);
        ggml_set_name(gate, "gemma4_native_moe_gate");
    }
    if (!hidden) {
        hidden = ggml_geglu_split(ctx, gate, up);
        ggml_set_name(hidden, "gemma4_native_moe_geglu");
    }
    const char* weighted_down_reject = nullptr;
    if (CanUseGemma4DownWeightedSumDecode(model, down_exps, hidden, selected_experts, weights, &weighted_down_reject)) {
        Qwen35SharedQ8RowsUserData* hidden_q8_ud =
            AllocateQwen35SharedQ8RowsUserData(ctx, hidden, n_expert_used * n_tokens);
        if (hidden_q8_ud) {
            SetNativeMoENumaContext(hidden_q8_ud, numa_backend, layer, model->variant);
            const int native_moe_callback_tasks =
                NativeMoEHasVerifiedNumaPlacement(numa_backend, layer)
                    ? 1
                    : ResolveNativeMoEGraphCallbackTaskCount(model, GetCurrentBatch(), GetCurrentExecutionPhase(),
                                                             n_tokens, static_cast<int>(n_expert_used));
            SetNativeMoECallbackRequestedTaskCount(hidden_q8_ud, native_moe_callback_tasks);
            ggml_tensor* down_args[] = {down_exps, hidden, selected_experts, weights};
            ggml_tensor* out =
                ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, 1, 1, down_args, 4,
                               cb_gemma4_native_moe_down_weighted_sum, native_moe_callback_tasks, hidden_q8_ud);
            char native_name[80];
            std::snprintf(native_name, sizeof(native_name), "blk.%d.gemma4_native_moe_down_weighted_sum", layer_idx);
            ggml_set_name(out, native_name);
            return out;
        }
        weighted_down_reject = "userdata_allocation_failed";
    }
    if (n_tokens == 1) {
        RecordGemma4DecodeNativeDecision(GetCurrentWorkContext(), /*candidate=*/true, /*used=*/false,
                                         weighted_down_reject ? weighted_down_reject : "unsupported_down_weighted_sum",
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, 0,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/0,
                                         /*duplicate_work_detected=*/false);
    }
    ggml_tensor* experts = ggml_mul_mat_id(ctx, down_exps, hidden, selected_experts);
    experts = ggml_mul(ctx, experts, weights);
    ggml_set_name(experts, "gemma4_native_moe_weighted_down");

    ggml_tensor* expert_views[32] = {nullptr};
    if (n_expert_used > static_cast<int64_t>(std::size(expert_views))) {
        return nullptr;
    }
    for (int64_t i = 0; i < n_expert_used; ++i) {
        expert_views[i] = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2],
                                       static_cast<size_t>(i) * static_cast<size_t>(experts->nb[1]));
        ggml_build_forward_expand(gf, expert_views[i]);
    }
    ggml_tensor* out = expert_views[0];
    for (int64_t i = 1; i < n_expert_used; ++i) {
        out = ggml_add(ctx, out, expert_views[i]);
    }
    if (n_expert_used == 1) {
        out = ggml_cont(ctx, out);
    }
    char name[80];
    std::snprintf(name, sizeof(name), "blk.%d.gemma4_native_moe_out", layer_idx);
    ggml_set_name(out, name);
    return out;
}

// ============================================================================
// Qwen3.5 W2 Q5_K fast down projection — replaces ggml_mul_mat_id with a
// direct custom callback so all ggml threads participate in the GEMV without
// ggml_mul_mat_id dispatch overhead. The row dot and Q8_K activation
// quantization are DenseCore HWY kernels, not GGML quant vec_dot.
// ============================================================================


}  // namespace densecore::llm::graph::detail

#ifdef DENSECORE_TEST_BUILD
using namespace densecore::llm::graph::detail;
namespace densecore::testing {
using namespace densecore::llm::graph::detail;
float Gemma4GeluTanhExactForTest(float x) {
    return ::Gemma4GeluTanh(x);
}
}  // namespace densecore::testing
#endif
