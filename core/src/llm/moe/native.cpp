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
#include "llm/moe/native_internal.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"

#include "llm/graph/common_types.h"
#include "llm/graph/policy_types.h"
#include "llm/graph/reference_types.h"
#include "llm/graph/ssm_types.h"
#include "llm/matmul/work_state.h"
#include "llm/moe/q4_rowpair.h"

static bool ShouldUseQwen36Q4DownRowPair(ModelVariant variant, InferenceExecutionPhase phase, int64_t tokens) {
#if defined(DENSECORE_TARGET_C4A)
    return variant == ModelVariant::QWEN36 && phase == InferenceExecutionPhase::Decode && tokens == 4 &&
           Q4DownRowPairNativeAvailable();
#else
    (void)variant;
    (void)phase;
    (void)tokens;
    return false;
#endif
}

#ifdef DENSECORE_TEST_BUILD
static std::atomic<uint64_t> q4_down_native_pair_test_ops{0};
#endif


namespace densecore::llm::graph::detail {


bool IsDebugSharedExpertShapeEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_VALIDATE_MUL");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void DebugLogSharedExpertTensor(const char* stage, int layer_idx, const struct ggml_tensor* tensor) {
    if (!IsDebugSharedExpertShapeEnabled() || !tensor) {
        return;
    }
    std::fprintf(stderr,
                 "[SharedExpertShape] layer=%d stage=%s name=%s type=%d ne=[%lld,%lld,%lld,%lld] "
                 "nb=[%lld,%lld,%lld,%lld]\n",
                 layer_idx, stage ? stage : "<unknown>", tensor->name[0] ? tensor->name : "<unnamed>",
                 static_cast<int>(tensor->type), static_cast<long long>(tensor->ne[0]),
                 static_cast<long long>(tensor->ne[1]), static_cast<long long>(tensor->ne[2]),
                 static_cast<long long>(tensor->ne[3]), static_cast<long long>(tensor->nb[0]),
                 static_cast<long long>(tensor->nb[1]), static_cast<long long>(tensor->nb[2]),
                 static_cast<long long>(tensor->nb[3]));
}

bool ShouldRunMoESharedDenseBranch(const TransformerModel* model, const densecore::models::DecoderLayerSpec* layer_spec,
                                   bool is_gemma4_moe, const struct ggml_tensor* ffn_gate,
                                   const struct ggml_tensor* ffn_up, const struct ggml_tensor* ffn_down) {
    if (!model || !ffn_gate || !ffn_up || !ffn_down) {
        return false;
    }
    if (layer_spec) {
        return layer_spec->ffn.has_shared_dense_branch;
    }
    if (model->moe_n_shared_experts > 0) {
        return true;
    }
    // Gemma4-26B-A4B carries a regular dense MLP branch alongside the sparse
    // MoE branch. Some GGUF exports do not advertise it via n_shared_experts,
    // so the presence of the shared FFN tensors is the load-bearing signal.
    return is_gemma4_moe;
}

using densecore::env::ParseBoolEnv;
using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using densecore::env::RuntimeToggleMode;
using densecore::llm::config::DecodePagedAttentionMode;
using densecore::llm::config::DecodePagedAttentionPolicy;
using densecore::llm::config::KVRetentionPolicy;
using densecore::llm::config::KVRetentionSpan;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;


std::atomic<uint64_t> g_moe_graph_wiring_debug_counter{0};
constexpr int64_t kQwen36C4AmxGraphMaxTokens = 128;
constexpr int64_t kNativeMoEFastPathDefaultMaxDirectTokens = 4096;

bool IsQwen35NativeMoEDownQ5KDiagEnabled() {
    static const bool enabled = densecore::env::ParseDiagnosticEnv("DENSECORE_NATIVE_MOE_FAST_W2_Q5K_DIAG", false);
    return enabled;
}

bool ModelRequiresNativeMoEFastPath(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    return densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract);
}

int64_t NativeMoEFastPathMaxDirectTokens(const TransformerModel* model) {
    if (!model) {
        return kNativeMoEFastPathDefaultMaxDirectTokens;
    }
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    const int64_t contract_limit = densecore::models::ModelExecutionContractNativeMoEMaxDirectTokens(contract);
    return contract_limit > 0 ? contract_limit : kNativeMoEFastPathDefaultMaxDirectTokens;
}

bool ShouldEnableNativeMoEFastPathByDefault(const TransformerModel* model, InferenceExecutionPhase phase,
                                            densecore::env::RuntimeToggleMode mode) {
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return false;
    }
    return ModelRequiresNativeMoEFastPath(model) && mode != densecore::env::RuntimeToggleMode::Off;
}

struct Qwen35SharedQ8RowsUserData;

bool IsDebugLFM2NativeMoEReferenceEnabled() {
    static const bool enabled = densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_LFM2_NATIVE_MOE_REFERENCE", false);
    return enabled;
}

bool IsDebugQwen35NativeMoEReferenceEnabled() {
    static const bool enabled =
        densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE", false);
    return enabled;
}

inline float NativeMoEFastExp(float x) {
    if (x < -50.0f) x = -50.0f;
    if (x > 50.0f) x = 50.0f;

    constexpr float kLog2E = 1.4426950408889634f;
    const float y = x * kLog2E;
    const int32_t i = static_cast<int32_t>(std::floor(y));
    const float f = y - static_cast<float>(i);
    const float p = 1.0f + f * (0.6960656421638072f + f * (0.224494337302845f + f * 0.07944023841053369f));

    const int32_t exp_bits = (i + 127) << 23;
    float two_i = 0.0f;
    std::memcpy(&two_i, &exp_bits, sizeof(two_i));
    return two_i * p;
}

inline float NativeMoESiLU(float x) {
    return x / (1.0f + NativeMoEFastExp(-x));
}

bool ShouldRunQwen35NativeMoEReferenceProbe(int layer_idx, const char* stage, int64_t token_idx) {
    if (!IsDebugQwen35NativeMoEReferenceEnabled()) {
        return false;
    }
    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_LAYER", -1);
    static const int target_token =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    static const char* target_stage =
        densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_STAGE");
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_MAX_CALLS", 16)};
    if (target_layer >= 0 && layer_idx != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }
    if (target_stage && target_stage[0] != '\0' && stage && std::strcmp(target_stage, stage) != 0) {
        return false;
    }
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

float Qwen35NativeMoEReferenceTolerance() {
    static const float tol = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOL");
        if (!env || env[0] == '\0') {
            return 1e-2f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-2f;
        }
        return parsed;
    }();
    return tol;
}

bool Qwen35NativeMoEReferenceDotQXK(ggml_type weight_type, const void* weight_row, const uint8_t* qrow, int64_t cols,
                                    float* out_value) {
    if (!weight_row || !qrow || !out_value || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K ||
        (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    *out_value = 0.0f;
    traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qrow, 0, 1);
    return std::isfinite(*out_value);
}


// Head granularity for the Qwen hybrid-SSM delta scan. Single-sequence decode
// keeps the granularity it was tuned with; a batch carries one token per
// sequence through every head, so it needs a finer split to keep every thread
// fed.


// Operational A/B switch for the multi-sequence recurrent split below. Setting
// it restores the previous serial behaviour on the same binary, so the
// concurrency effect can be measured without maintaining two builds. This is a
// plain (non-diagnostic) accessor on purpose: diagnostic env vars compile out of
// release builds, which is exactly where the measurement has to run.


// The hybrid-SSM recurrent ops partition over disjoint slices of recurrent
// state: conv1d over channel ranges, the delta scan over v-head ranges. A task
// only ever touches its own slice of the state belonging to the sequence each
// token maps to, and it walks tokens in ascending order, so the per-sequence
// recurrence stays ordered. That makes a multi-sequence batch partition exactly
// like a single-sequence one. Gating these on num_seqs == 1 collapsed 30 of
// Qwen3.6's 40 layers onto one thread at the moment the batch carried 2-4x the
// recurrent work, which is what capped concurrent decode throughput.


bool IsMoEWiringDebugEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_MOE_WIRING_DEBUG");
        if (!env || env[0] == '\0') {
            env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_MOE_WIRING");
        }
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}


struct Qwen35SharedQ8RowsUserData;

ggml_tensor* GetLayerTensorAny(TransformerLayer* layer, std::initializer_list<const char*> keys) {
    if (!layer) {
        return nullptr;
    }
    for (const char* key : keys) {
        if (!key) {
            continue;
        }
        if (ggml_tensor* tensor = layer->Get(key)) {
            return tensor;
        }
    }
    return nullptr;
}


ggml_tensor* UseCpuRepackAliasIfAvailable(TransformerModel* model, ggml_tensor* tensor) {
    if (!model || !tensor) {
        return tensor;
    }
    auto it = model->prepared_weights.cpu_repack_aliases.find(tensor);
    return it == model->prepared_weights.cpu_repack_aliases.end() ? tensor : it->second;
}

bool CpuRepackAliasHasLayout(const TransformerModel* model, const ggml_tensor* tensor,
                             TransformerModel::CpuRepackAliasLayout layout) {
    if (!model || !tensor || layout == TransformerModel::CpuRepackAliasLayout::Unknown) {
        return false;
    }
    auto it = model->prepared_weights.cpu_repack_alias_layouts.find(tensor);
    if (it != model->prepared_weights.cpu_repack_alias_layouts.end()) {
        return it->second == layout;
    }
    const ggml_tensor* view_src = tensor->view_src;
    if (!view_src) {
        return false;
    }
    it = model->prepared_weights.cpu_repack_alias_layouts.find(view_src);
    return it != model->prepared_weights.cpu_repack_alias_layouts.end() && it->second == layout;
}

ggml_tensor* UseCpuRepackAliasForTokenCount(TransformerModel* model, ggml_tensor* tensor, int64_t n_tokens) {
    if (!model || !tensor) {
        return tensor;
    }
    auto it = model->prepared_weights.cpu_repack_aliases.find(tensor);
    if (it == model->prepared_weights.cpu_repack_aliases.end() || !it->second) {
        return tensor;
    }
    if (model->variant == ModelVariant::QWEN36 && n_tokens > 0 && n_tokens <= kQwen36C4AmxGraphMaxTokens &&
        model->prepared_weights.cpu_amx_aliases.find(it->second) != model->prepared_weights.cpu_amx_aliases.end()) {
        return tensor;
    }
    return it->second;
}

void cb_moe_expert_weighted_sum_with_weights(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->src[1] || !dst->data || !dst->src[0]->data || !dst->src[1]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* experts = dst->src[0];
    const ggml_tensor* weights = dst->src[1];
    if (dst->type != GGML_TYPE_F32 || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        experts->ne[0] != dst->ne[0] || experts->ne[2] != dst->ne[1] || experts->ne[1] <= 0 || weights->ne[0] != 1 ||
        weights->ne[1] != experts->ne[1] || weights->ne[2] != experts->ne[2]) {
        return;
    }

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t total = n_embd * n_tokens;
    const int64_t start = (total * ith) / nth;
    const int64_t end = (total * (ith + 1)) / nth;

    const char* src_base = static_cast<const char*>(experts->data);
    const char* weight_base = static_cast<const char*>(weights->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t dense_expert_stride = static_cast<size_t>(n_embd) * sizeof(float);
    const bool dense_strides =
        experts->nb[0] == static_cast<int64_t>(sizeof(float)) && experts->nb[1] == dense_expert_stride &&
        weights->nb[0] == static_cast<int64_t>(sizeof(float)) && dst->nb[0] == static_cast<int64_t>(sizeof(float));
    if (dense_strides) {
        int64_t idx = start;
        while (idx < end) {
            const int64_t tok = idx / n_embd;
            const int64_t embd0 = idx - tok * n_embd;
            const int64_t embd1 = std::min<int64_t>(n_embd, end - tok * n_embd);
            const int64_t count = embd1 - embd0;
            const float w0 = *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2]);
            const float safe_w0 = std::isfinite(w0) ? w0 : 0.0f;
            const float* src0 =
                reinterpret_cast<const float*>(src_base + tok * experts->nb[2] + embd0 * experts->nb[0]);
            float* dst_ptr = reinterpret_cast<float*>(dst_base + tok * dst->nb[1] + embd0 * dst->nb[0]);
            if (n_expert_used == 8) {
                const float* src1 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] + experts->nb[1] +
                                                                   embd0 * experts->nb[0]);
                const float* src2 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   2 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src3 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   3 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src4 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   4 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src5 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   5 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src6 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   6 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src7 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                   7 * experts->nb[1] + embd0 * experts->nb[0]);
                const float w1 = *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + weights->nb[1]);
                const float w2 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 2 * weights->nb[1]);
                const float w3 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 3 * weights->nb[1]);
                const float w4 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 4 * weights->nb[1]);
                const float w5 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 5 * weights->nb[1]);
                const float w6 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 6 * weights->nb[1]);
                const float w7 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 7 * weights->nb[1]);
                const float safe_w1 = std::isfinite(w1) ? w1 : 0.0f;
                const float safe_w2 = std::isfinite(w2) ? w2 : 0.0f;
                const float safe_w3 = std::isfinite(w3) ? w3 : 0.0f;
                const float safe_w4 = std::isfinite(w4) ? w4 : 0.0f;
                const float safe_w5 = std::isfinite(w5) ? w5 : 0.0f;
                const float safe_w6 = std::isfinite(w6) ? w6 : 0.0f;
                const float safe_w7 = std::isfinite(w7) ? w7 : 0.0f;
                for (int64_t i = 0; i < count; ++i) {
                    float sum = src0[i] * safe_w0;
                    sum += src1[i] * safe_w1;
                    sum += src2[i] * safe_w2;
                    sum += src3[i] * safe_w3;
                    sum += src4[i] * safe_w4;
                    sum += src5[i] * safe_w5;
                    sum += src6[i] * safe_w6;
                    sum += src7[i] * safe_w7;
                    dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
                }
                idx += count;
                continue;
            }
            for (int64_t i = 0; i < count; ++i) {
                const float sum = src0[i] * safe_w0;
                dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
            }
            for (int64_t expert = 1; expert < n_expert_used; ++expert) {
                const float weight =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + expert * weights->nb[1]);
                const float safe_weight = std::isfinite(weight) ? weight : 0.0f;
                const float* src_ptr = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] +
                                                                      expert * experts->nb[1] + embd0 * experts->nb[0]);
                for (int64_t i = 0; i < count; ++i) {
                    const float sum = dst_ptr[i] + src_ptr[i] * safe_weight;
                    dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
                }
            }
            idx += count;
        }
        return;
    }

    for (int64_t idx = start; idx < end; ++idx) {
        const int64_t embd = idx % n_embd;
        const int64_t tok = idx / n_embd;
        float sum = 0.0f;
        for (int64_t expert = 0; expert < n_expert_used; ++expert) {
            const char* src_ptr = src_base + embd * experts->nb[0] + expert * experts->nb[1] + tok * experts->nb[2];
            const char* weight_ptr = weight_base + expert * weights->nb[1] + tok * weights->nb[2];
            const float weight = *reinterpret_cast<const float*>(weight_ptr);
            const float weighted = *reinterpret_cast<const float*>(src_ptr) * (std::isfinite(weight) ? weight : 0.0f);
            sum += weighted;
        }
        char* dst_ptr = dst_base + embd * dst->nb[0] + tok * dst->nb[1];
        *reinterpret_cast<float*>(dst_ptr) = std::isfinite(sum) ? sum : 0.0f;
    }
}

ggml_tensor* BuildMoeExpertWeightedSumWithWeights(struct ggml_context* ctx, ggml_tensor* experts, ggml_tensor* weights,
                                                  int64_t n_embd, int64_t n_tokens, const char* name) {
    if (!ctx || !experts || !weights || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        n_embd <= 0 || n_tokens <= 0 || experts->ne[0] != n_embd || experts->ne[2] != n_tokens || experts->ne[1] <= 0 ||
        weights->ne[0] != 1 || weights->ne[1] != experts->ne[1] || weights->ne[2] != n_tokens) {
        return nullptr;
    }
    ggml_tensor* args[] = {experts, weights};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, 1, 1, args, 2,
                                      cb_moe_expert_weighted_sum_with_weights, GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(out, name);
    return out;
}

constexpr int64_t kQwenNativeMoEFusedRouterTopK = 8;

struct QwenNativeMoEFusedRouterState {
    int64_t n_experts = 0;
    int64_t n_tokens = 0;
    int64_t top_k = 0;
    float* normalized_weights = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
};

static bool QwenNativeMoERouterCandidatePrecedes(float lhs, int32_t lhs_id, float rhs, int32_t rhs_id) {
    const bool lhs_nan = std::isnan(lhs);
    const bool rhs_nan = std::isnan(rhs);
    if (lhs_nan != rhs_nan) {
        return !lhs_nan;
    }
    if (lhs > rhs) return true;
    if (lhs < rhs) return false;
    return lhs_id < rhs_id;
}

static bool ComputeQwenNativeMoEStableTopK(const float* logits, int64_t n_experts, int64_t top_k, int32_t* selected,
                                           float* normalized_weights) {
    if (!logits || !selected || !normalized_weights || n_experts <= 0 || top_k <= 0 || top_k > n_experts ||
        top_k > kQwenNativeMoEFusedRouterTopK) {
        return false;
    }

    int64_t selected_count = 0;
    for (int32_t expert = 0; expert < n_experts; ++expert) {
        const float score = logits[expert];
        int64_t insert_at = selected_count;
        while (insert_at > 0 && QwenNativeMoERouterCandidatePrecedes(score, expert, logits[selected[insert_at - 1]],
                                                                     selected[insert_at - 1])) {
            --insert_at;
        }
        if (insert_at >= top_k) {
            continue;
        }
        const int64_t move_end = std::min<int64_t>(selected_count, top_k - 1);
        for (int64_t pos = move_end; pos > insert_at; --pos) {
            selected[pos] = selected[pos - 1];
        }
        selected[insert_at] = expert;
        selected_count = std::min<int64_t>(selected_count + 1, top_k);
    }
    if (selected_count != top_k) {
        return false;
    }

    float max_logit = -INFINITY;
    bool all_finite = true;
    for (int64_t k = 0; k < top_k; ++k) {
        const float value = logits[selected[k]];
        all_finite = all_finite && std::isfinite(value);
        if (std::isfinite(value)) {
            max_logit = std::max(max_logit, value);
        }
    }

    float denominator = 0.0f;
    if (all_finite && std::isfinite(max_logit)) {
        for (int64_t k = 0; k < top_k; ++k) {
            normalized_weights[k] = std::exp(logits[selected[k]] - max_logit);
            denominator += normalized_weights[k];
        }
    }
    if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
        const float uniform = 1.0f / static_cast<float>(top_k);
        std::fill(normalized_weights, normalized_weights + top_k, uniform);
        return true;
    }
    for (int64_t k = 0; k < top_k; ++k) {
        normalized_weights[k] /= denominator;
    }
    return true;
}

static QwenNativeMoEFusedRouterState* AllocateQwenNativeMoEFusedRouterState(ggml_context* ctx, int64_t n_experts,
                                                                            int64_t n_tokens, int64_t top_k) {
    if (!ctx || n_experts <= 0 || n_tokens <= 0 || top_k != kQwenNativeMoEFusedRouterTopK || top_k > n_experts) {
        return nullptr;
    }
    const size_t weights_count = static_cast<size_t>(top_k) * static_cast<size_t>(n_tokens);
    if (ggml_get_no_alloc(ctx)) {
        thread_local QwenNativeMoEFusedRouterState dry_run_state;
        thread_local std::vector<float> dry_run_weights;
        dry_run_weights.resize(weights_count);
        dry_run_state.n_experts = n_experts;
        dry_run_state.n_tokens = n_tokens;
        dry_run_state.top_k = top_k;
        dry_run_state.normalized_weights = dry_run_weights.data();
        dry_run_state.work_ctx = GetCurrentWorkContext();
        return &dry_run_state;
    }
    ggml_tensor* state_storage =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(sizeof(QwenNativeMoEFusedRouterState)));
    ggml_tensor* weights_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(weights_count));
    if (!state_storage || !state_storage->data || !weights_storage || !weights_storage->data) {
        return nullptr;
    }
    auto* state = new (state_storage->data) QwenNativeMoEFusedRouterState();
    state->n_experts = n_experts;
    state->n_tokens = n_tokens;
    state->top_k = top_k;
    state->normalized_weights = static_cast<float*>(weights_storage->data);
    state->work_ctx = GetCurrentWorkContext();
    return state;
}

static void cb_qwen_native_moe_fused_router(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* state = static_cast<QwenNativeMoEFusedRouterState*>(userdata);
    const ggml_tensor* logits = dst ? dst->src[0] : nullptr;
    if (!dst || !state || !logits || !dst->data || !logits->data || nth <= 0 || dst->type != GGML_TYPE_I32 ||
        logits->type != GGML_TYPE_F32 || logits->ne[0] != state->n_experts || logits->ne[1] != state->n_tokens ||
        dst->ne[0] != state->top_k || dst->ne[1] != state->n_tokens || !state->normalized_weights) {
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    bool used = false;
    const int64_t token_begin = (state->n_tokens * ith) / nth;
    const int64_t token_end = (state->n_tokens * (ith + 1)) / nth;
    for (int64_t token = token_begin; token < token_end; ++token) {
        const float* token_logits = reinterpret_cast<const float*>(
            static_cast<const char*>(logits->data) + static_cast<size_t>(token) * static_cast<size_t>(logits->nb[1]));
        int32_t* token_selected = reinterpret_cast<int32_t*>(
            static_cast<char*>(dst->data) + static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
        float* token_weights = state->normalized_weights + static_cast<size_t>(token) * state->top_k;
        used = ComputeQwenNativeMoEStableTopK(token_logits, state->n_experts, state->top_k, token_selected,
                                              token_weights) ||
               used;
    }
    if (used && state->work_ctx) {
        const uint64_t elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
        RecordQwenNativeMoEFusedRouterUse(state->work_ctx, elapsed_ns);
    }
}

static ggml_tensor* BuildQwenNativeMoEFusedRouter(ggml_context* ctx, ggml_tensor* gate_logits, int64_t top_k,
                                                  QwenNativeMoEFusedRouterState** state_out) {
    if (state_out) *state_out = nullptr;
    if (!ctx || !gate_logits || gate_logits->type != GGML_TYPE_F32 || gate_logits->ne[0] < top_k ||
        gate_logits->ne[1] != 1 || top_k != kQwenNativeMoEFusedRouterTopK) {
        return nullptr;
    }
    QwenNativeMoEFusedRouterState* state =
        AllocateQwenNativeMoEFusedRouterState(ctx, gate_logits->ne[0], gate_logits->ne[1], top_k);
    if (!state) {
        return nullptr;
    }
    ggml_tensor* args[] = {gate_logits};
    ggml_tensor* selected = ggml_custom_4d(ctx, GGML_TYPE_I32, top_k, gate_logits->ne[1], 1, 1, args, 1,
                                           cb_qwen_native_moe_fused_router, 1, state);
    ggml_set_name(selected, "qwen_native_moe_fused_top8_router");
    if (state_out) *state_out = state;
    return selected;
}

static bool CopyQwenNativeMoEFusedRouterWeights(const QwenNativeMoEFusedRouterState* state, int64_t token,
                                                float* weights, int64_t weights_capacity) {
    if (!state || !state->normalized_weights || !weights || token < 0 || token >= state->n_tokens ||
        state->top_k <= 0 || state->top_k > weights_capacity) {
        return false;
    }
    const float* src = state->normalized_weights + static_cast<size_t>(token) * state->top_k;
    std::copy(src, src + state->top_k, weights);
    return true;
}

static void cb_qwen_native_moe_fused_router_weights(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* state = static_cast<const QwenNativeMoEFusedRouterState*>(userdata);
    if (!dst || !state || !dst->src[0] || !dst->data || nth <= 0 || dst->type != GGML_TYPE_F32 || dst->ne[0] != 1 ||
        dst->ne[1] != state->top_k || dst->ne[2] != state->n_tokens) {
        return;
    }
    const int64_t token_begin = (state->n_tokens * ith) / nth;
    const int64_t token_end = (state->n_tokens * (ith + 1)) / nth;
    for (int64_t token = token_begin; token < token_end; ++token) {
        for (int64_t k = 0; k < state->top_k; ++k) {
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) + static_cast<size_t>(k) * dst->nb[1] +
                                      static_cast<size_t>(token) * dst->nb[2]) =
                state->normalized_weights[static_cast<size_t>(token) * state->top_k + k];
        }
    }
}

static ggml_tensor* BuildQwenNativeMoEFusedRouterWeights(ggml_context* ctx, ggml_tensor* selected,
                                                         QwenNativeMoEFusedRouterState* state, const char* name) {
    if (!ctx || !selected || !state || selected->type != GGML_TYPE_I32 || selected->ne[0] != state->top_k ||
        selected->ne[1] != state->n_tokens) {
        return nullptr;
    }
    ggml_tensor* args[] = {selected};
    ggml_tensor* weights = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, state->top_k, state->n_tokens, 1, args, 1,
                                          cb_qwen_native_moe_fused_router_weights, 1, state);
    ggml_set_name(weights, name);
    return weights;
}

void cb_moe_topk_weights_from_logits(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->src[1] || !dst->data || !dst->src[0]->data || !dst->src[1]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* logits = dst->src[0];
    const ggml_tensor* selected = dst->src[1];
    if (dst->type != GGML_TYPE_F32 || logits->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
        logits->ne[0] <= 0 || logits->ne[1] <= 0 || selected->ne[0] <= 0 || selected->ne[1] != logits->ne[1] ||
        dst->ne[0] != 1 || dst->ne[1] != selected->ne[0] || dst->ne[2] != logits->ne[1]) {
        return;
    }

    const int64_t n_experts = logits->ne[0];
    const int64_t n_tokens = logits->ne[1];
    const int64_t top_k = selected->ne[0];
    const int64_t start = (n_tokens * ith) / nth;
    const int64_t end = (n_tokens * (ith + 1)) / nth;
    const char* logits_base = static_cast<const char*>(logits->data);
    const char* selected_base = static_cast<const char*>(selected->data);
    char* dst_base = static_cast<char*>(dst->data);

    for (int64_t tok = start; tok < end; ++tok) {
        float max_logit = -INFINITY;
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            if (expert >= 0 && expert < n_experts) {
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                if (std::isfinite(logit)) {
                    max_logit = std::max(max_logit, logit);
                }
            }
        }

        float denom = 0.0f;
        int valid_selected = 0;
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            if (expert >= 0 && expert < n_experts) {
                valid_selected++;
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                if (std::isfinite(logit) && std::isfinite(max_logit)) {
                    denom += std::exp(logit - max_logit);
                }
            }
        }
        const bool use_uniform = !(denom > 0.0f) || !std::isfinite(denom);
        denom = use_uniform ? 1.0f : std::max(denom, 6.103515625e-5f);
        const float uniform = valid_selected > 0 ? 1.0f / static_cast<float>(valid_selected) : 0.0f;

        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            float value = 0.0f;
            if (expert >= 0 && expert < n_experts) {
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                value = use_uniform || !std::isfinite(logit) || !std::isfinite(max_logit)
                            ? uniform
                            : std::exp(logit - max_logit) / denom;
            }
            *reinterpret_cast<float*>(dst_base + k * dst->nb[1] + tok * dst->nb[2]) = value;
        }
    }
}

ggml_tensor* BuildMoETopKWeightsFromLogits(struct ggml_context* ctx, ggml_tensor* logits, ggml_tensor* selected,
                                           const char* name) {
    if (!ctx || !logits || !selected || logits->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
        logits->ne[0] <= 0 || logits->ne[1] <= 0 || selected->ne[0] <= 0 || selected->ne[1] != logits->ne[1]) {
        return nullptr;
    }
    const int n_tasks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(GGML_N_TASKS_MAX, logits->ne[1])));
    ggml_tensor* args[] = {logits, selected};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, selected->ne[0], logits->ne[1], 1, args, 2,
                                      cb_moe_topk_weights_from_logits, n_tasks, nullptr);
    ggml_set_name(out, name);
    return out;
}

void cb_fused_gate_up_silu_mul(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->data || !dst->src[0]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* gate_up = dst->src[0];
    if (dst->type != GGML_TYPE_F32 || gate_up->type != GGML_TYPE_F32 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float)) || gate_up->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        gate_up->ne[0] != 2 * dst->ne[0] || gate_up->ne[1] != dst->ne[1]) {
        return;
    }

    const int64_t n_ff = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const char* src_base = static_cast<const char*>(gate_up->data);
    char* dst_base = static_cast<char*>(dst->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float* gate =
            reinterpret_cast<const float*>(src_base + static_cast<size_t>(token) * static_cast<size_t>(gate_up->nb[1]));
        const float* up = gate + n_ff;
        float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
        densecore::simd::SiLUMulParallel(out, gate, up, static_cast<size_t>(n_ff), ith, nth);
    }
}

ggml_tensor* BuildFusedGateUpSiluMul(struct ggml_context* ctx, ggml_tensor* gate_up, int64_t n_ff, int64_t n_tokens,
                                     const char* name) {
    if (!ctx || !gate_up || gate_up->type != GGML_TYPE_F32 || n_ff <= 0 || n_tokens <= 0 ||
        gate_up->ne[0] != 2 * n_ff || gate_up->ne[1] != n_tokens ||
        gate_up->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return nullptr;
    }
    ggml_tensor* args[] = {gate_up};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_ff, n_tokens, 1, 1, args, 1, cb_fused_gate_up_silu_mul,
                                      GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(out, name);
    return out;
}

constexpr int kQwen35SharedQ8MaxTasks = 128;
constexpr int64_t kQwen35NativeMoEBatchedQ4KMaxAssignments = 256;
constexpr int64_t kQwen35NativeMoEGateUpBatchTile = 128;
constexpr int kQwen35NativeMoEMaxGroupedExperts = 64;

struct Qwen35MoEAssignment {
    int32_t expert = -1;
    int32_t token = -1;
    int32_t topk_index = -1;
    float weight = 1.0f;
};

struct Qwen35SharedQ8RowsUserData {
    int64_t cols = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int debug_layer_idx = -1;
    size_t row_bytes = 0;
    uint8_t* rows = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    InferenceExecutionPhase execution_phase = InferenceExecutionPhase::Unknown;
    ModelVariant model_variant = ModelVariant::UNKNOWN;
    densecore::CpuBackend* numa_backend = nullptr;
    const TransformerLayer* numa_layer = nullptr;
    const ggml_tensor* prepacked_fused_gate_up_exps = nullptr;
    bool numa_sticky_enabled = false;
    std::atomic<uint64_t> epoch{0};
    std::atomic<uint64_t> ready_epoch{0};
    std::atomic<int> remaining_tasks{0};
    std::atomic<int> failed{0};
    bool prefer_q4k_repacked_swiglu = false;
    bool q5k_gateup_8x8_single_copy = false;
    bool q5k_gateup_8x8_single_copy_required = false;
    bool record_lfm2_w1w3_kernel = false;
    bool group_small_decode_gateup = false;
#ifdef DENSECORE_TEST_BUILD
    std::atomic<uint64_t> grouped_gateup_tiles{0};
#endif
    std::atomic<int> repacked_swiglu_failed{0};
    bool weighted_logits_lfm2_sigmoid = false;
    bool weighted_logits_norm_topk = true;
    float weighted_logits_scale = 1.0f;
    QwenNativeMoEFusedRouterState* fused_router_state = nullptr;
    int requested_task_count = 0;
    // Task width the native MoE callback is currently running at, published by
    // RemapNativeMoECallbackTask. Kernels dispatched from inside that region use
    // it to decide whether re-parallelizing onto a node thread pool is worth it:
    // when the outer region is already wide, it is not.
    std::atomic<int> outer_task_width{0};
    // 0 = unresolved, 1 = keep the validated Highway reference, 2 = use the
    // maintained DenseCore AVX2/scalar single-row Q5_K kernel. The verdict is
    // populated once per graph userdata before the row-parallel hot loop.
    std::atomic<int> q5k_single_row_admission{0};
    std::atomic<uint64_t> assignments_ready_epoch{0};
    std::atomic<int> assignments_failed{0};
    Qwen35MoEAssignment* assignments = nullptr;
    size_t assignment_capacity = 0;
    size_t assignment_count = 0;
    uint64_t task_epoch[kQwen35SharedQ8MaxTasks] = {};
    float* direct_expert_outputs = nullptr;
    size_t direct_expert_output_capacity = 0;
    std::atomic<uint64_t> direct_epoch{0};
    std::atomic<int> direct_remaining_tasks{0};
    std::atomic<int> direct_failed{0};
    uint64_t direct_task_epoch[kQwen35SharedQ8MaxTasks] = {};
    int outer_task_cores[kQwen35SharedQ8MaxTasks] = {};
    int outer_task_nodes[kQwen35SharedQ8MaxTasks] = {};
    int outer_task_mapping_count = 0;
};

Qwen35SharedQ8RowsUserData* AllocateQwen35SharedQ8RowsUserData(ggml_context* ctx, const ggml_tensor* src,
                                                               int64_t max_assignments, int64_t direct_output_rows) {
    if (!ctx || !src || src->type != GGML_TYPE_F32 || src->ne[0] <= 0 || src->ne[1] <= 0 || src->ne[2] <= 0) {
        return nullptr;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_K, src->ne[0]);
    const int64_t rows = src->ne[1] * src->ne[2];
    if (row_bytes == 0 || rows <= 0) {
        return nullptr;
    }
    const size_t rows_bytes = row_bytes * static_cast<size_t>(rows);
    const size_t assignment_capacity =
        max_assignments > 0 ? static_cast<size_t>(std::min<int64_t>(max_assignments, 4096 * 64)) : 0;
    const size_t assignments_bytes = assignment_capacity * sizeof(Qwen35MoEAssignment);
    const size_t direct_output_capacity = direct_output_rows > 0 && assignment_capacity > 0
                                              ? assignment_capacity * static_cast<size_t>(direct_output_rows)
                                              : 0;
    const size_t direct_outputs_bytes = direct_output_capacity * sizeof(float);
    if (ggml_get_no_alloc(ctx)) {
        // The custom op retains userdata after graph construction. Allocate a
        // stable sidecar per op; reusing one TLS object aliases different MoE
        // layers and can route them with the last-built layer's placement.
        auto* ud = new Qwen35SharedQ8RowsUserData();
        ud->cols = src->ne[0];
        ud->ne1 = src->ne[1];
        ud->ne2 = src->ne[2];
        ud->row_bytes = row_bytes;
        ud->rows = new uint8_t[rows_bytes]();
        ud->assignments = assignment_capacity > 0 ? new Qwen35MoEAssignment[assignment_capacity]() : nullptr;
        ud->assignment_capacity = assignment_capacity;
        ud->direct_expert_outputs = direct_output_capacity > 0 ? new float[direct_output_capacity]() : nullptr;
        ud->direct_expert_output_capacity = direct_output_capacity;
        return ud;
    }
    ggml_tensor* ud_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Qwen35SharedQ8RowsUserData));
    ggml_tensor* rows_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(rows_bytes));
    ggml_tensor* assignments_storage =
        assignments_bytes > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(assignments_bytes))
                              : nullptr;
    ggml_tensor* direct_outputs_storage =
        direct_outputs_bytes > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(direct_outputs_bytes))
                                 : nullptr;
    if (!ud_storage || !ud_storage->data || !rows_storage || !rows_storage->data) {
        return nullptr;
    }
    auto* ud = new (ud_storage->data) Qwen35SharedQ8RowsUserData();
    ud->cols = src->ne[0];
    ud->ne1 = src->ne[1];
    ud->ne2 = src->ne[2];
    ud->debug_layer_idx = -1;
    ud->row_bytes = row_bytes;
    ud->rows = static_cast<uint8_t*>(rows_storage->data);
    ud->numa_backend = nullptr;
    ud->numa_layer = nullptr;
    ud->numa_sticky_enabled = false;
    ud->prefer_q4k_repacked_swiglu = false;
    ud->q5k_gateup_8x8_single_copy = false;
    ud->q5k_gateup_8x8_single_copy_required = false;
    ud->record_lfm2_w1w3_kernel = false;
    ud->repacked_swiglu_failed.store(0, std::memory_order_relaxed);
    ud->weighted_logits_lfm2_sigmoid = false;
    ud->weighted_logits_norm_topk = true;
    ud->weighted_logits_scale = 1.0f;
    ud->fused_router_state = nullptr;
    ud->requested_task_count = 0;
    ud->assignments_ready_epoch.store(0, std::memory_order_relaxed);
    ud->assignments_failed.store(0, std::memory_order_relaxed);
    if (assignments_storage && assignments_storage->data) {
        ud->assignments = static_cast<Qwen35MoEAssignment*>(assignments_storage->data);
        ud->assignment_capacity = assignment_capacity;
    }
    if (direct_outputs_storage && direct_outputs_storage->data) {
        ud->direct_expert_outputs = static_cast<float*>(direct_outputs_storage->data);
        ud->direct_expert_output_capacity = direct_output_capacity;
    }
    ud->assignment_count = 0;
    return ud;
}

static bool RemapNativeMoECallbackTask(Qwen35SharedQ8RowsUserData* ud, int ith, int nth, int* effective_ith,
                                       int* effective_nth) {
    if (!effective_ith || !effective_nth || ith < 0 || nth <= 0 || ith >= nth) {
        return false;
    }
    int task_count = ud ? ud->requested_task_count : 0;
    if (task_count <= 0 || task_count >= nth) {
        *effective_ith = ith;
        *effective_nth = nth;
        if (ud) {
            // Every task publishes the same width, so a relaxed store is enough.
            ud->outer_task_width.store(nth, std::memory_order_relaxed);
        }
        return true;
    }
    task_count = std::max(1, task_count);
    if (ud) {
        ud->outer_task_width.store(task_count, std::memory_order_relaxed);
    }
    if (ith >= task_count) {
        return false;
    }
    *effective_ith = ith;
    *effective_nth = task_count;
    return true;
}

// True when the caller is the only task running the native MoE op, so a kernel
// may safely fan out onto a NUMA node's thread pool. Inside a wide outer region
// that fan-out is pure loss: the outer tasks already saturate the machine, and
// the node pool's workers would just oversubscribe the same cores.
static inline bool NativeMoEKernelMayFanOut(const Qwen35SharedQ8RowsUserData* ud, int numa_node) {
    if (numa_node < 0) {
        return false;
    }
    const int width = ud ? ud->outer_task_width.load(std::memory_order_relaxed) : 0;
    return width <= 1;
}

void SetNativeMoECallbackRequestedTaskCount(Qwen35SharedQ8RowsUserData* ud, int requested_task_count) {
    if (ud) {
        ud->requested_task_count = requested_task_count;
    }
}

// --------------------------------------------------------------------------
// Native MoE NUMA sticky-routing observability
// --------------------------------------------------------------------------
// The small-decode dispatcher announces itself with a log line, but the native
// MoE path below has historically been silent. That made an inactive sticky
// router indistinguishable from an active one in a server log, so these
// Counters record actual NUMA handoffs. Legacy paths count per expert, while
// grouped decode paths count once per active node group.

struct NativeMoENumaCounters {
    std::atomic<uint64_t> context_set_total{0};
    std::atomic<uint64_t> context_enabled_total{0};
    std::atomic<uint64_t> sticky_dispatch_ops{0};
    std::atomic<uint64_t> legacy_dispatch_ops{0};
    std::atomic<uint64_t> grouped_decode_used_ops{0};
    std::atomic<uint64_t> grouped_dispatch_node_tasks{0};
    std::atomic<uint64_t> grouped_dispatch_expert_items{0};
    std::atomic<uint64_t> direct_decode_used_ops{0};
    std::atomic<uint64_t> direct_decode_rejected_ops{0};
    std::array<std::atomic<uint64_t>, kNativeMoENumaMaxTrackedNodes> node_dispatch_ops{};
    std::atomic<uint64_t> node_dispatch_overflow_ops{0};
    std::atomic<int> last_state{static_cast<int>(NativeMoENumaStickyState::Unset)};
};

static NativeMoENumaCounters& GetNativeMoENumaCounters() {
    static NativeMoENumaCounters counters;
    return counters;
}

// A non-sticky callback can dispatch many experts on every GGML worker. Keep
// the same per-expert count without contending on the global atomic each time.
// Publish on scope exit as well, including when an expert callback throws.
class NativeMoELegacyDispatchBatch {
public:
    explicit NativeMoELegacyDispatchBatch(std::atomic<uint64_t>& counter) : counter_(counter) {}
    NativeMoELegacyDispatchBatch(const NativeMoELegacyDispatchBatch&) = delete;
    NativeMoELegacyDispatchBatch& operator=(const NativeMoELegacyDispatchBatch&) = delete;
    ~NativeMoELegacyDispatchBatch() { Flush(); }

    void Record() { ++pending_; }
    void Flush() {
        if (pending_ != 0) {
            counter_.fetch_add(pending_, std::memory_order_relaxed);
            pending_ = 0;
        }
    }

private:
    std::atomic<uint64_t>& counter_;
    uint64_t pending_ = 0;
};

// The facts behind the sticky-routing verdict. The state alone cannot explain a
// degenerate layout, so the expert count and the node they all landed on ride
// along for the one-shot warning in RecordNativeMoENumaContext.
struct NativeMoENumaPlacement {
    NativeMoENumaStickyState state = NativeMoENumaStickyState::Unset;
    int expert_count = 0;
    int node_count = 0;
    int sole_node = -1;  // set only for EnabledSingleNodeDegenerate
};

// Sticky dispatch arms for both armed states. The degenerate layout is reported
// differently but routed identically, which keeps this split purely
// observational: whether to fall back to core round-robin when every expert
// sits on one node is a policy question that needs real two-socket throughput
// numbers to answer, and spreading compute away from where the weights live can
// cost more in remote access than it wins in parallelism.
static constexpr bool NativeMoENumaStickyArmed(NativeMoENumaStickyState state) {
    return state == NativeMoENumaStickyState::Enabled || state == NativeMoENumaStickyState::EnabledSingleNodeDegenerate;
}

static bool NativeMoENumaStickyQualifiedModelVariant(ModelVariant variant) {
    // Qwen3.5-122B-A10B regressed decode on the two-socket N2 qualification
    // host. Keep QWEN35 on the maintained global graph until that family has a
    // positive fallback-free A/B result; QWEN36 retains its published result.
    return variant != ModelVariant::QWEN35;
}

static NativeMoENumaPlacement EvaluateNativeMoENumaPlacement(densecore::CpuBackend* backend,
                                                             const TransformerLayer* layer,
                                                             ModelVariant model_variant = ModelVariant::UNKNOWN) {
    NativeMoENumaPlacement placement;
    if (densecore::env::ParseDiagnosticEnv("DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY", false)) {
        placement.state = NativeMoENumaStickyState::DisabledByDebug;
        return placement;
    }
    if (!densecore::env::ParseBoolEnv("DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY", false)) {
        placement.state = NativeMoENumaStickyState::DisabledByPolicy;
        return placement;
    }
    if (!backend || !layer) {
        placement.state = NativeMoENumaStickyState::NoBackend;
        return placement;
    }
    placement.node_count = backend->GetNumaNodeCount();
    if (placement.node_count <= 1) {
        placement.state = NativeMoENumaStickyState::SingleNode;
        return placement;
    }
    std::vector<int> nodes;
    if (!backend->CopyExpertNumaNodes(layer, static_cast<int>(layer->NumExperts()), &nodes) || nodes.empty()) {
        placement.state = NativeMoENumaStickyState::PlacementUnavailable;
        return placement;
    }
    placement.expert_count = static_cast<int>(nodes.size());
    const int node_count = placement.node_count;
    const bool all_valid =
        std::all_of(nodes.begin(), nodes.end(), [node_count](int node) { return node >= 0 && node < node_count; });
    if (!all_valid) {
        placement.state = NativeMoENumaStickyState::PlacementInvalid;
        return placement;
    }
    const int first_node = nodes.front();
    const bool degenerate =
        std::all_of(nodes.begin(), nodes.end(), [first_node](int node) { return node == first_node; });
    const ModelVariant effective_variant =
        model_variant != ModelVariant::UNKNOWN ? model_variant : GetCurrentInferenceWorkContextModelVariant();
    if (!NativeMoENumaStickyQualifiedModelVariant(effective_variant)) {
        placement.state = NativeMoENumaStickyState::DisabledUnqualifiedModel;
        return placement;
    }
    placement.state =
        degenerate ? NativeMoENumaStickyState::EnabledSingleNodeDegenerate : NativeMoENumaStickyState::Enabled;
    placement.sole_node = degenerate ? first_node : -1;
    return placement;
}

// One-shot so a 60-layer model does not emit 60 identical lines per decode step.
// Reports the first observed state and, separately, the first time sticky
// routing actually arms — the transition operators care about.
static void RecordNativeMoENumaContext(const NativeMoENumaPlacement& placement, densecore::CpuBackend* backend) {
    auto& counters = GetNativeMoENumaCounters();
    counters.context_set_total.fetch_add(1, std::memory_order_relaxed);
    counters.last_state.store(static_cast<int>(placement.state), std::memory_order_relaxed);

    // Counts armed contexts, degenerate included: the router did arm, and the
    // state field alongside it says in which form.
    const bool armed = NativeMoENumaStickyArmed(placement.state);
    if (armed) {
        counters.context_enabled_total.fetch_add(1, std::memory_order_relaxed);
    }

    const bool degenerate = placement.state == NativeMoENumaStickyState::EnabledSingleNodeDegenerate;
    static std::atomic<bool> logged_any{false};
    static std::atomic<bool> logged_enabled{false};
    static std::atomic<bool> logged_degenerate{false};
    const bool first_any = !logged_any.exchange(true, std::memory_order_relaxed);
    const bool first_enabled = armed && !logged_enabled.exchange(true, std::memory_order_relaxed);
    // Its own one-shot: a degenerate layer arriving after a well-spread one
    // would otherwise never warn, since logged_enabled is already set.
    const bool first_degenerate = degenerate && !logged_degenerate.exchange(true, std::memory_order_relaxed);
    if (!first_any && !first_enabled && !first_degenerate) {
        return;
    }
    const char* state_name = GetNativeMoENumaStickyStateName(static_cast<int>(placement.state));
    const int node_count = backend ? backend->GetNumaNodeCount() : 0;
    if (degenerate) {
        std::fprintf(stderr,
                     "[NUMA] Native MoE sticky routing active but DEGENERATE (state=%s, nodes=%d) -- all %d "
                     "experts are on node %d, so MoE compute will not spread past it and the remaining "
                     "node(s) stay idle. Set DENSECORE_NUMA_WEIGHTS=round_robin to partition expert "
                     "weights across nodes.\n",
                     state_name, node_count, placement.expert_count, placement.sole_node);
        return;
    }
    std::fprintf(stderr, "[NUMA] Native MoE sticky routing %s (state=%s, nodes=%d)\n", armed ? "active" : "inactive",
                 state_name, node_count);
}

void RecordNativeMoENumaDispatch(int node) {
    auto& counters = GetNativeMoENumaCounters();
    if (node < 0) {
        counters.legacy_dispatch_ops.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    counters.sticky_dispatch_ops.fetch_add(1, std::memory_order_relaxed);
    if (static_cast<std::size_t>(node) < kNativeMoENumaMaxTrackedNodes) {
        counters.node_dispatch_ops[static_cast<std::size_t>(node)].fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.node_dispatch_overflow_ops.fetch_add(1, std::memory_order_relaxed);
    }
}

void SetNativeMoENumaContext(Qwen35SharedQ8RowsUserData* ud, densecore::CpuBackend* backend,
                             const TransformerLayer* layer, ModelVariant model_variant) {
    if (!ud) {
        return;
    }
    ud->numa_backend = backend;
    ud->numa_layer = layer;
    ud->execution_phase = GetCurrentExecutionPhase();
    ud->model_variant = model_variant;
    const NativeMoENumaPlacement placement = EvaluateNativeMoENumaPlacement(backend, layer, model_variant);
    ud->numa_sticky_enabled =
        ud->execution_phase == InferenceExecutionPhase::Decode && NativeMoENumaStickyArmed(placement.state);
    ud->outer_task_mapping_count = 0;
    if (ud->numa_sticky_enabled) {
        auto& topology = densecore::HardwareTopology::GetInstance();
        for (int task = 0; task < kQwen35SharedQ8MaxTasks; ++task) {
            const int core = topology.GetAssignedCore(task);
            const int node = topology.GetAssignedNumaNode(task);
            if (core < 0 || node < 0) {
                break;
            }
            ud->outer_task_cores[task] = core;
            ud->outer_task_nodes[task] = node;
            ++ud->outer_task_mapping_count;
        }
    }
    RecordNativeMoENumaContext(placement, backend);
}

static int ResolveNativeMoEExpertNumaNode(const Qwen35SharedQ8RowsUserData* ud, int expert_id) {
    if (!ud || !ud->numa_sticky_enabled || !ud->numa_backend || !ud->numa_layer || expert_id < 0) {
        return -1;
    }
    const int node = ud->numa_backend->GetExpertNumaNode(ud->numa_layer, expert_id);
    return node >= 0 && node < ud->numa_backend->GetNumaNodeCount() ? node : -1;
}

struct NativeMoENodeGroupPlan {
    std::array<int, kQwen35NativeMoEMaxGroupedExperts> item_nodes{};
    std::array<int, kQwen35NativeMoEMaxGroupedExperts> active_nodes{};
    int item_count = 0;
    int active_node_count = 0;
};

struct NativeMoEOuterTaskPlan {
    int task_node = -1;
    int node_task_rank = -1;
    int node_task_count = 0;
};

static InferenceExecutionPhase ResolveNativeMoEOuterTaskExecutionPhase(const Qwen35SharedQ8RowsUserData* ud) {
    // GGML secondary workers do not inherit the inference worker's TLS work
    // context. A wide custom op must make one shared admission decision, so use
    // the graph userdata's context instead of GetCurrentExecutionPhase().
    return ud && ud->execution_phase != InferenceExecutionPhase::Unknown ? ud->execution_phase
                                                                         : GetCurrentExecutionPhase();
}

static bool BuildNativeMoEOuterTaskPlan(const int* task_nodes, int task_count, int task_index, int node_count,
                                        NativeMoEOuterTaskPlan* plan) {
    if (!task_nodes || !plan || task_count <= 1 || task_count > kQwen35SharedQ8MaxTasks || task_index < 0 ||
        task_index >= task_count || node_count <= 1) {
        return false;
    }
    *plan = {};
    const int task_node = task_nodes[task_index];
    if (task_node < 0 || task_node >= node_count) {
        return false;
    }
    int rank = 0;
    int count = 0;
    for (int task = 0; task < task_count; ++task) {
        const int node = task_nodes[task];
        if (node < 0 || node >= node_count) {
            return false;
        }
        if (node == task_node) {
            rank += task < task_index ? 1 : 0;
            ++count;
        }
    }
    if (count <= 0 || rank < 0 || rank >= count) {
        return false;
    }
    plan->task_node = task_node;
    plan->node_task_rank = rank;
    plan->node_task_count = count;
    return true;
}

static bool ResolveNativeMoEOuterTaskPlan(Qwen35SharedQ8RowsUserData* ud, int ith, int nth,
                                          NativeMoEOuterTaskPlan* plan) {
    if (!ud || !ud->numa_backend || !ud->numa_sticky_enabled || !plan || nth <= 1 || nth > kQwen35SharedQ8MaxTasks) {
        return false;
    }
    if (ud->outer_task_mapping_count < nth || ith >= ud->outer_task_mapping_count) {
        return false;
    }
    const int assigned_core = ud->outer_task_cores[ith];
    if (assigned_core < 0) {
        return false;
    }
    // GGML clears only task 0's affinity after each graph. Re-pin task 0 once
    // per graph execution and persistent secondary workers only when their
    // assignment changes. Merely observing the assigned CPU is not proof of a
    // singleton affinity mask and would allow migration mid-kernel.
    thread_local int pinned_outer_core = -1;
    thread_local const InferenceWorkContext* pinned_outer_work_ctx = nullptr;
    thread_local uint64_t pinned_outer_generation = 0;
    const InferenceWorkContext* work_ctx = ud->work_ctx;
    const uint64_t execution_generation = GetInferenceWorkContextExecutionGeneration(work_ctx);
    const bool task_zero_graph_changed =
        ith == 0 && (!work_ctx || pinned_outer_work_ctx != work_ctx || pinned_outer_generation != execution_generation);
    if ((task_zero_graph_changed || pinned_outer_core != assigned_core) &&
        !densecore::HardwareTopology::PinCurrentThread(assigned_core)) {
        return false;
    }
    pinned_outer_core = assigned_core;
    pinned_outer_work_ctx = work_ctx;
    pinned_outer_generation = execution_generation;
    return BuildNativeMoEOuterTaskPlan(ud->outer_task_nodes, nth, ith, ud->numa_backend->GetNumaNodeCount(), plan);
}

static uint64_t BeginNativeMoEDirectOuterTask(Qwen35SharedQ8RowsUserData* ud, int ith, int nth, bool initialize_ok) {
    if (!ud || ith < 0 || ith >= nth || nth <= 1 || nth > kQwen35SharedQ8MaxTasks) {
        return 0;
    }
    uint64_t epoch = 0;
    if (ith == 0) {
        epoch = ud->direct_epoch.load(std::memory_order_relaxed) + 1;
        ud->direct_failed.store(initialize_ok ? 0 : 1, std::memory_order_relaxed);
        ud->direct_remaining_tasks.store(nth, std::memory_order_relaxed);
        ud->direct_epoch.store(epoch, std::memory_order_release);
    } else {
        const uint64_t last_epoch = ud->direct_task_epoch[ith];
        epoch = ud->direct_epoch.load(std::memory_order_acquire);
        while (epoch == last_epoch) {
            std::this_thread::yield();
            epoch = ud->direct_epoch.load(std::memory_order_acquire);
        }
    }
    return epoch;
}

static bool FinishNativeMoEDirectOuterTask(Qwen35SharedQ8RowsUserData* ud, int ith, uint64_t epoch, bool task_ok,
                                           bool* is_owner_task = nullptr) {
    if (!ud || epoch == 0) {
        return false;
    }
    const bool owner_task = ith == 0;
    if (is_owner_task) {
        *is_owner_task = owner_task;
    }
    if (!task_ok) {
        ud->direct_failed.store(1, std::memory_order_relaxed);
    }
    ud->direct_remaining_tasks.fetch_sub(1, std::memory_order_acq_rel);
    ud->direct_task_epoch[ith] = epoch;
    if (!owner_task) {
        // GGML already places a barrier after this custom node. Let secondary
        // workers reach it while task 0 waits for completion and publishes the
        // reduction (or the complete grouped fallback) for the node.
        return true;
    }
    while (ud->direct_remaining_tasks.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
    const bool all_ok = ud->direct_failed.load(std::memory_order_acquire) == 0;
    auto& counters = GetNativeMoENumaCounters();
    (all_ok ? counters.direct_decode_used_ops : counters.direct_decode_rejected_ops)
        .fetch_add(1, std::memory_order_relaxed);
    return all_ok;
}

static bool BuildNativeMoENodeGroupPlan(const int* item_nodes, int item_count, int node_count,
                                        NativeMoENodeGroupPlan* plan) {
    if (!item_nodes || !plan || item_count < 0 || item_count > kQwen35NativeMoEMaxGroupedExperts || node_count <= 0) {
        return false;
    }
    *plan = {};
    plan->item_count = item_count;
    for (int item = 0; item < item_count; ++item) {
        const int node = item_nodes[item];
        plan->item_nodes[static_cast<size_t>(item)] = node;
        if (node < 0) {
            continue;
        }
        if (node >= node_count) {
            return false;
        }
        bool seen = false;
        for (int active = 0; active < plan->active_node_count; ++active) {
            if (plan->active_nodes[static_cast<size_t>(active)] == node) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            plan->active_nodes[static_cast<size_t>(plan->active_node_count++)] = node;
        }
    }
    return true;
}

template <typename Fn>
static bool RunNativeMoEOnExpertNodeGroups(Qwen35SharedQ8RowsUserData* ud, const int* experts, int item_count,
                                           Fn&& run_node_group) {
    if (!ud || !ud->numa_sticky_enabled || !ud->numa_backend || !experts || item_count < 0 ||
        item_count > kQwen35NativeMoEMaxGroupedExperts) {
        return false;
    }
    const int node_count = ud->numa_backend->GetNumaNodeCount();
    if (node_count <= 1) {
        return false;
    }

    std::array<int, kQwen35NativeMoEMaxGroupedExperts> item_nodes{};
    for (int item = 0; item < item_count; ++item) {
        if (experts[item] < 0) {
            item_nodes[static_cast<size_t>(item)] = -1;
            continue;
        }
        const int node = ResolveNativeMoEExpertNumaNode(ud, experts[item]);
        if (node < 0) {
            return false;
        }
        item_nodes[static_cast<size_t>(item)] = node;
    }

    NativeMoENodeGroupPlan plan;
    if (!BuildNativeMoENodeGroupPlan(item_nodes.data(), item_count, node_count, &plan)) {
        return false;
    }
    if (plan.active_node_count == 0) {
        return true;
    }
    for (int active = 0; active < plan.active_node_count; ++active) {
        RecordNativeMoENumaDispatch(plan.active_nodes[static_cast<size_t>(active)]);
    }
    uint64_t routed_item_count = 0;
    for (int item = 0; item < plan.item_count; ++item) {
        routed_item_count += plan.item_nodes[static_cast<size_t>(item)] >= 0 ? 1 : 0;
    }
    auto& counters = GetNativeMoENumaCounters();
    counters.grouped_dispatch_node_tasks.fetch_add(static_cast<uint64_t>(plan.active_node_count),
                                                   std::memory_order_relaxed);
    counters.grouped_dispatch_expert_items.fetch_add(routed_item_count, std::memory_order_relaxed);

    const auto run_node = [&](int node) { run_node_group(node, plan); };
    if (plan.active_node_count == 1) {
        const int node = plan.active_nodes[0];
        ud->numa_backend->RunOnNumaNode(node, [&] { run_node(node); });
        return true;
    }
    ud->numa_backend->RunConcurrentNodeTasks(node_count, run_node);
    return true;
}

template <typename Fn> static void RunNativeMoEOnExpertNode(Qwen35SharedQ8RowsUserData* ud, int expert_id, Fn&& fn) {
    const int node = ResolveNativeMoEExpertNumaNode(ud, expert_id);
    RecordNativeMoENumaDispatch(node);
    if (node < 0) {
        fn(-1);
        return;
    }
    ud->numa_backend->RunOnNumaNode(node, [&] { fn(node); });
}

bool NativeMoEHasVerifiedNumaPlacement(densecore::CpuBackend* backend, const TransformerLayer* layer) {
    return NativeMoENumaStickyArmed(EvaluateNativeMoENumaPlacement(backend, layer).state);
}

static bool Qwen35NativeQuantizeRowQ8K(const float* src, uint8_t* dst, int64_t cols) {
    return densecore::hwy_kernels::QuantizeRowQ8K_Hwy(src, dst, cols);
}

static bool Qwen35QuantizeSharedQ8RowsRange(Qwen35SharedQ8RowsUserData* ud, const ggml_tensor* src, int ith, int nth) {
    const int64_t rows = ud->ne1 * ud->ne2;
    const int64_t row_start = (static_cast<int64_t>(ith) * rows) / nth;
    const int64_t row_end = (static_cast<int64_t>(ith + 1) * rows) / nth;
    bool ok = true;
    for (int64_t linear = row_start; linear < row_end; ++linear) {
        const int64_t row1 = linear % ud->ne1;
        const int64_t row2 = linear / ud->ne1;
        const float* src_row = reinterpret_cast<const float*>(
            static_cast<const char*>(src->data) + static_cast<size_t>(row1) * static_cast<size_t>(src->nb[1]) +
            static_cast<size_t>(row2) * static_cast<size_t>(src->nb[2]));
        if (!Qwen35NativeQuantizeRowQ8K(src_row, ud->rows + static_cast<size_t>(linear) * ud->row_bytes, ud->cols)) {
            ok = false;
            break;
        }
    }
    return ok;
}

static bool PrepareQwen35SharedQ8Rows(Qwen35SharedQ8RowsUserData* ud, const ggml_tensor* src, int ith, int nth) {
    if (!ud || !src || !src->data || src->type != GGML_TYPE_F32 || ith < 0 || ith >= nth || nth <= 0 ||
        nth > kQwen35SharedQ8MaxTasks || src->ne[0] != ud->cols || src->ne[1] != ud->ne1 || src->ne[2] != ud->ne2 ||
        !ud->rows) {
        return false;
    }
    uint64_t epoch = 0;
    if (ith == 0) {
        epoch = ud->epoch.load(std::memory_order_relaxed) + 1;
        ud->failed.store(0, std::memory_order_relaxed);
        ud->remaining_tasks.store(nth, std::memory_order_relaxed);
        ud->task_epoch[0] = epoch;
        ud->epoch.store(epoch, std::memory_order_release);
    } else {
        const uint64_t last_epoch = ud->task_epoch[ith];
        epoch = ud->epoch.load(std::memory_order_acquire);
        while (epoch == last_epoch) {
            std::this_thread::yield();
            epoch = ud->epoch.load(std::memory_order_acquire);
        }
    }

    const bool ok = Qwen35QuantizeSharedQ8RowsRange(ud, src, ith, nth);
    if (!ok) {
        ud->failed.store(1, std::memory_order_relaxed);
    }
    if (ud->remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        ud->ready_epoch.store(epoch, std::memory_order_release);
    }
    while (ud->ready_epoch.load(std::memory_order_acquire) != epoch) {
        std::this_thread::yield();
    }
    ud->task_epoch[ith] = epoch;
    return ud->failed.load(std::memory_order_acquire) == 0;
}

static const uint8_t* Qwen35SharedQ8RowPtr(const Qwen35SharedQ8RowsUserData* ud, int64_t row1, int64_t row2 = 0) {
    if (!ud || !ud->rows || row1 < 0 || row1 >= ud->ne1 || row2 < 0 || row2 >= ud->ne2) {
        return nullptr;
    }
    const int64_t linear = row2 * ud->ne1 + row1;
    return ud->rows + static_cast<size_t>(linear) * ud->row_bytes;
}

static void Qwen35NativeMoEZeroDst2D(ggml_tensor* dst, int64_t n_rows, int64_t n_tokens) {
    if (!dst || !dst->data) return;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t row = 0; row < n_rows; ++row) {
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
    }
}

static void Qwen35NativeMoEZeroDst2DRange(ggml_tensor* dst, int64_t row_start, int64_t row_end, int64_t n_tokens) {
    if (!dst || !dst->data) return;
    row_start = std::max<int64_t>(0, row_start);
    row_end = std::min<int64_t>(dst->ne[0], row_end);
    if (row_start >= row_end) return;
    for (int64_t token = 0; token < n_tokens; ++token) {
        char* dst_col = static_cast<char*>(dst->data) + static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]);
        for (int64_t row = row_start; row < row_end; ++row) {
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) = 0.0f;
        }
    }
}

static bool Qwen35BuildMoEAssignments(const ggml_tensor* selected_experts, int64_t n_experts,
                                      std::vector<Qwen35MoEAssignment>* assignments) {
    if (!selected_experts || !selected_experts->data || selected_experts->type != GGML_TYPE_I32 || n_experts <= 0 ||
        !assignments) {
        return false;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    assignments->clear();
    assignments->reserve(static_cast<size_t>(top_k) * static_cast<size_t>(n_tokens));
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert = *reinterpret_cast<const int32_t*>(
                selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
            if (expert < 0 || expert >= n_experts) {
                continue;
            }
            assignments->push_back({expert, static_cast<int32_t>(token), static_cast<int32_t>(k), 1.0f});
        }
    }
    std::sort(assignments->begin(), assignments->end(), [](const Qwen35MoEAssignment& a, const Qwen35MoEAssignment& b) {
        if (a.expert != b.expert) return a.expert < b.expert;
        if (a.token != b.token) return a.token < b.token;
        return a.topk_index < b.topk_index;
    });
    return !assignments->empty();
}

static bool Qwen35BuildMoEAssignmentsToBuffer(const ggml_tensor* selected_experts, int64_t n_experts,
                                              Qwen35MoEAssignment* assignments, size_t assignment_capacity,
                                              size_t* assignment_count_out) {
    if (assignment_count_out) {
        *assignment_count_out = 0;
    }
    if (!selected_experts || !selected_experts->data || selected_experts->type != GGML_TYPE_I32 || n_experts <= 0 ||
        !assignments || assignment_capacity == 0 || !assignment_count_out) {
        return false;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (top_k <= 0 || n_tokens <= 0 ||
        static_cast<uint64_t>(top_k) * static_cast<uint64_t>(n_tokens) > assignment_capacity) {
        return false;
    }
    size_t count = 0;
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert = *reinterpret_cast<const int32_t*>(
                selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
            if (expert < 0 || expert >= n_experts) {
                continue;
            }
            assignments[count++] = {expert, static_cast<int32_t>(token), static_cast<int32_t>(k), 1.0f};
        }
    }
    std::sort(assignments, assignments + count, [](const Qwen35MoEAssignment& a, const Qwen35MoEAssignment& b) {
        if (a.expert != b.expert) return a.expert < b.expert;
        if (a.token != b.token) return a.token < b.token;
        return a.topk_index < b.topk_index;
    });
    *assignment_count_out = count;
    return count > 0;
}

static const Qwen35MoEAssignment* PrepareQwen35SharedMoEAssignments(Qwen35SharedQ8RowsUserData* ud,
                                                                    const ggml_tensor* selected_experts,
                                                                    int64_t n_experts, int ith,
                                                                    size_t* assignment_count_out) {
    if (assignment_count_out) {
        *assignment_count_out = 0;
    }
    if (!ud || !selected_experts || ith < 0 || !assignment_count_out) {
        return nullptr;
    }
    const uint64_t epoch = ud->epoch.load(std::memory_order_acquire);
    if (epoch == 0) {
        return nullptr;
    }
    if (ith == 0) {
        size_t count = 0;
        const bool ok = Qwen35BuildMoEAssignmentsToBuffer(selected_experts, n_experts, ud->assignments,
                                                          ud->assignment_capacity, &count);
        ud->assignment_count = ok ? count : 0;
        ud->assignments_failed.store(ok ? 0 : 1, std::memory_order_relaxed);
        ud->assignments_ready_epoch.store(epoch, std::memory_order_release);
    } else {
        while (ud->assignments_ready_epoch.load(std::memory_order_acquire) != epoch) {
            std::this_thread::yield();
        }
    }
    if (ud->assignments_failed.load(std::memory_order_acquire) != 0) {
        return nullptr;
    }
    *assignment_count_out = ud->assignment_count;
    return ud->assignments;
}

static const float* Qwen35NativeMoEDownHiddenRowPtr(const ggml_tensor* hidden, int64_t token, int64_t topk_index) {
    if (!hidden || !hidden->data) return nullptr;
    return reinterpret_cast<const float*>(static_cast<const char*>(hidden->data) +
                                          static_cast<size_t>(topk_index) * static_cast<size_t>(hidden->nb[1]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(hidden->nb[2]));
}

// Phase 0: capability decisions now live in kernels/kernel_caps.h (single source
// of truth). These thin wrappers preserve the call-site names.
static bool PreferGgmlQ4KVecDotForNativeMoE() {
    return densecore::kernels::PreferGgmlKQuantVecDot();
}

static bool CanUseGgmlQ4KVecDotRowPairForNativeMoE() {
    return densecore::kernels::KQuantVecDotRowPairSupported();
}

static bool Qwen35NativeMoEDownQ5KQuantizeHidden(const ggml_tensor* hidden, int64_t token, int64_t topk_index,
                                                 std::vector<uint8_t>& qbuf) {
    if (!hidden || !hidden->data || hidden->type != GGML_TYPE_F32) return false;
    const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, topk_index);
    if (!hidden_row) return false;
    const int64_t cols = hidden->ne[0];
    const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    if (qrow_bytes == 0) return false;
    qbuf.resize(qrow_bytes);
    return Qwen35NativeQuantizeRowQ8K(hidden_row, qbuf.data(), cols);
}

static bool Qwen35NativeMoEDownQuantizeHiddenForWeight(const ggml_tensor* hidden, int64_t token, int64_t topk_index,
                                                       ggml_type weight_type, std::vector<uint8_t>& qbuf) {
    if (weight_type == GGML_TYPE_Q8_0) {
        if (!hidden || !hidden->data || hidden->type != GGML_TYPE_F32) return false;
        const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, topk_index);
        if (!hidden_row) return false;
        const int64_t cols = hidden->ne[0];
        const ggml_type_traits_cpu* q8_0_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_0, cols);
        if (!q8_0_traits || !q8_0_traits->from_float || qrow_bytes == 0) return false;
        qbuf.resize(qrow_bytes);
        q8_0_traits->from_float(hidden_row, qbuf.data(), cols);
        return true;
    }
    return Qwen35NativeMoEDownQ5KQuantizeHidden(hidden, token, topk_index, qbuf);
}

static bool Qwen35NativeMoEDownQ8_0ReferenceDotRowForExpertWithHidden(const ggml_tensor* down_exps, int32_t expert,
                                                                      int64_t row, const float* hidden_row,
                                                                      float* out_value) {
    if (!down_exps || !hidden_row || !out_value || down_exps->type != GGML_TYPE_Q8_0) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const ggml_type_traits* q8_0_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    if (!q8_0_traits || !q8_0_traits->to_float) return false;
    const int64_t cols = down_exps->ne[0];
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    thread_local std::vector<float> wbuf;
    wbuf.resize(static_cast<size_t>(cols));
    q8_0_traits->to_float(weight_row, wbuf.data(), cols);
    double acc = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        acc += static_cast<double>(hidden_row[static_cast<size_t>(i)]) * wbuf[static_cast<size_t>(i)];
    }
    *out_value = static_cast<float>(acc);
    return true;
}

static bool Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert, int64_t row,
                                                           const uint8_t* qbuf, float* out_value) {
    if (!down_exps || !qbuf || !out_value || down_exps->type != GGML_TYPE_Q8_0) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const int64_t cols = down_exps->ne[0];
    const ggml_type_traits_cpu* q8_0_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!q8_0_traits || !q8_0_traits->vec_dot || q8_0_traits->vec_dot_type != GGML_TYPE_Q8_0 ||
        (cols % ggml_blck_size(GGML_TYPE_Q8_0)) != 0) {
        return false;
    }
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    q8_0_traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
    return true;
}

static bool Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert, int64_t row,
                                                          const uint8_t* qbuf, const float* hidden_row,
                                                          float* out_value, bool use_densecore_q5k_single_row = false) {
    if (!down_exps || !out_value) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const int64_t cols = down_exps->ne[0];
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    (void)hidden_row;
    if (down_exps->type == GGML_TYPE_Q8_0) {
        return Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row, qbuf, out_value);
    }
    if (!qbuf) return false;
    if (down_exps->type == GGML_TYPE_Q5_K) {
        if (use_densecore_q5k_single_row) {
            return ComputeQ5KQ8KBatchedRow(weight_row, qbuf, ggml_row_size(GGML_TYPE_Q8_K, cols), 1,
                                           static_cast<int>(cols), out_value);
        }
        if (PreferGgmlQ4KVecDotForNativeMoE()) {
            const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q5_K);
            if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
                (cols % ggml_blck_size(GGML_TYPE_Q5_K)) == 0) {
                traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
                return true;
            }
        }
        return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, qbuf, cols, out_value);
    }
    if (down_exps->type == GGML_TYPE_Q4_K) {
        if (PreferGgmlQ4KVecDotForNativeMoE()) {
            const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
            if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
                (cols % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
                traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
                return true;
            }
        }
        return densecore::hwy_kernels::DotQ4KQ8K_Hwy(weight_row, qbuf, cols, out_value);
    }
    if (down_exps->type == GGML_TYPE_Q6_K) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
        if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
            return false;
        }
        traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
        return true;
    }
    return false;
}

static bool Qwen35NativeMoEDownQ5KReadExpert(const ggml_tensor* selected_experts, const ggml_tensor* down_exps,
                                             int64_t token, int64_t topk_index, int32_t* expert_out) {
    if (!selected_experts || !down_exps || !selected_experts->data || !expert_out) return false;
    const int32_t expert = *reinterpret_cast<const int32_t*>(
        static_cast<const char*>(selected_experts->data) +
        static_cast<size_t>(topk_index) * static_cast<size_t>(selected_experts->nb[0]) +
        static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
    if (expert < 0 || expert >= down_exps->ne[2]) return false;
    *expert_out = expert;
    return true;
}

static bool Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert, int64_t row,
                                                              const uint8_t* qbuf, const float* hidden_row, float* out0,
                                                              float* out1, bool use_densecore_q5k_single_row = false,
                                                              bool use_q4_native_pair = false,
                                                              bool* used_q4_native_pair = nullptr) {
    if (!down_exps || !out0 || !out1) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row + 1 >= down_exps->ne[1]) return false;
    (void)hidden_row;
    if (down_exps->type == GGML_TYPE_Q8_0) {
        if (!qbuf) return false;
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_0 &&
            (down_exps->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0)) == 0) {
            const char* weight_row = static_cast<const char*>(down_exps->data) +
                                     static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                                     static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
            float sums[32] = {};
            traits->vec_dot(static_cast<int>(down_exps->ne[0]), sums, 2, weight_row,
                            static_cast<size_t>(down_exps->nb[1]), qbuf, 0, 2);
            *out0 = sums[0];
            *out1 = sums[1];
            return true;
        }
        return Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row, qbuf, out0) &&
               Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row + 1, qbuf, out1);
    }
    if (!qbuf) return false;
    if (use_q4_native_pair && down_exps->type == GGML_TYPE_Q4_K) {
        const char* weight_row = static_cast<const char*>(down_exps->data) +
                                 static_cast<size_t>(expert) * down_exps->nb[2] +
                                 static_cast<size_t>(row) * down_exps->nb[1];
        float sums[2];
        if (ComputeQ4DownRowPairNative(weight_row, down_exps->nb[1], qbuf, down_exps->ne[0], sums)) {
            *out0 = sums[0];
            *out1 = sums[1];
            if (used_q4_native_pair) *used_q4_native_pair = true;
            return true;
        }
    }
    if ((down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K || down_exps->type == GGML_TYPE_Q6_K) &&
        CanUseGgmlQ4KVecDotRowPairForNativeMoE()) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(down_exps->type);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
            (down_exps->ne[0] % ggml_blck_size(down_exps->type)) == 0) {
            const char* weight_row = static_cast<const char*>(down_exps->data) +
                                     static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                                     static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
            float sums[32] = {};
            traits->vec_dot(static_cast<int>(down_exps->ne[0]), sums, 2, weight_row,
                            static_cast<size_t>(down_exps->nb[1]), qbuf, 0, 2);
            *out0 = sums[0];
            *out1 = sums[1];
            return true;
        }
    }
    return Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qbuf, hidden_row, out0,
                                                         use_densecore_q5k_single_row) &&
           Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row + 1, qbuf, hidden_row, out1,
                                                         use_densecore_q5k_single_row);
}

static bool ResolveQwen35NativeMoEDownQ5KSingleRowAdmission(Qwen35SharedQ8RowsUserData* shared_q8,
                                                            const ggml_tensor* down_exps, int32_t expert,
                                                            const uint8_t* qrow) {
    if (!shared_q8 || !down_exps || !qrow || down_exps->type != GGML_TYPE_Q5_K || expert < 0 ||
        expert >= down_exps->ne[2] || down_exps->ne[0] <= 0 || down_exps->ne[1] <= 0) {
        return false;
    }

    const int cached = shared_q8->q5k_single_row_admission.load(std::memory_order_acquire);
    if (cached != 0) {
        return cached == 2;
    }

    constexpr uint32_t kQwen35MoEDownQ5KSingleRowOp = 0x51354b31u;
    const uint64_t key = densecore::kernels::ParityGate::MakeKey(kQwen35MoEDownQ5KSingleRowOp, down_exps->data,
                                                                 down_exps->ne[1], down_exps->ne[0]);
    auto verdict = densecore::kernels::ParityGate::Check(key);
    if (verdict == densecore::kernels::ParityGate::Verdict::kProbe) {
        const int64_t cols = down_exps->ne[0];
        const char* weight_row = static_cast<const char*>(down_exps->data) +
                                 static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
        float reference = 0.0f;
        float candidate = 0.0f;
        const bool reference_ok = densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, qrow, cols, &reference);
        const bool candidate_ok = ComputeQ5KQ8KBatchedRow(weight_row, qrow, ggml_row_size(GGML_TYPE_Q8_K, cols), 1,
                                                          static_cast<int>(cols), &candidate);
        const float max_abs_error = reference_ok && candidate_ok && std::isfinite(candidate)
                                        ? std::fabs(candidate - reference)
                                        : std::numeric_limits<float>::infinity();
        densecore::kernels::ParityGate::Report(key, max_abs_error, std::fabs(reference), 1e-5f);
        verdict = densecore::kernels::ParityGate::Check(key);
    }

    const bool use_densecore = verdict == densecore::kernels::ParityGate::Verdict::kAllow;
    shared_q8->q5k_single_row_admission.store(use_densecore ? 2 : 1, std::memory_order_release);
    static std::atomic<bool> logged_allow{false};
    static std::atomic<bool> logged_deny{false};
    std::atomic<bool>& logged = use_densecore ? logged_allow : logged_deny;
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[Q5KSingleRow] admitted=%d path=%s\n", use_densecore ? 1 : 0,
                     use_densecore ? "densecore_avx2" : "highway_reference");
    }
    return use_densecore;
}

static bool Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
    ggml_tensor* dst, const ggml_tensor* down_exps, const uint8_t* qtile, size_t qrow_bytes,
    const std::vector<Qwen35MoEAssignment>& tile_assignments, int32_t expert, int64_t row_start, int64_t row_end,
    int numa_node, bool allow_parallel, InferenceExecutionPhase phase) {
    if (!dst || !down_exps || !qtile || !dst->data || !down_exps->data || tile_assignments.empty()) {
        return false;
    }
    const ggml_type wtype = down_exps->type;
    const auto* traits = ggml_get_type_traits_cpu(wtype);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
        return false;
    }
    const int64_t K = down_exps->ne[0];
    const int64_t n_embd = down_exps->ne[1];
    if (K <= 0 || n_embd <= 0 || row_start < 0 || row_end > n_embd || row_start > row_end ||
        (K % ggml_blck_size(wtype)) != 0 || qrow_bytes < ggml_row_size(GGML_TYPE_Q8_K, K)) {
        return false;
    }
    const size_t w_row_bytes = static_cast<size_t>(down_exps->nb[1]);
    const char* expert_base =
        static_cast<const char*>(down_exps->data) + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
    if (wtype == GGML_TYPE_Q5_K && row_start < row_end) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch());
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const auto raw_batched_begin = std::chrono::steady_clock::now();
        const bool projected = densecore::RunMoEKQuantRawBatchedProjection(
            densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
            static_cast<int>(wtype), weight_start, qtile, qrow_bytes, projection_tile.data(),
            static_cast<int64_t>(tile_assignments.size()), row_count, K, numa_node, allow_parallel);
        if (projected) {
            const auto elapsed = std::chrono::steady_clock::now() - raw_batched_begin;
            RecordMoEKQuantRawBatchedUse(
                GetCurrentWorkContext(), wtype,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                /*qwen_native_w2_q5k=*/true);
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col +
                                              static_cast<size_t>(row_start + row) * static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }
    if (wtype == GGML_TYPE_Q5_K && !PreferGgmlQ4KVecDotForNativeMoE()) {
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sum = 0.0f;
                if (!densecore::hwy_kernels::DotQ5KQ8K_Hwy(w_row, qrow, K, &sum)) {
                    return false;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sum * assignment.weight;
            }
        }
        return true;
    }

    // Secondary GGML/NUMA workers do not inherit the inference worker TLS.
    const bool prefill_rows = phase == InferenceExecutionPhase::Prefill;

    if (wtype == GGML_TYPE_Q4_K && row_start < row_end && (row_start % 8) == 0 && ((row_end - row_start) % 8) == 0) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch());
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const bool projected =
            prefill_rows ? densecore::RunMoEKQuantRawBatchedProjection(
                               densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                               static_cast<int>(wtype), weight_start, qtile, qrow_bytes, projection_tile.data(),
                               static_cast<int64_t>(tile_assignments.size()), row_count, K, numa_node, allow_parallel)
                         : densecore::RunQ4KRepackedMoEProjection(
                               densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                               weight_start, qtile, qrow_bytes, projection_tile.data(),
                               static_cast<int64_t>(tile_assignments.size()), row_count, K, numa_node, allow_parallel);
        if (projected) {
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col +
                                              static_cast<size_t>(row_start + row) * static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }
    if (wtype == GGML_TYPE_Q6_K && row_start < row_end && (row_start % 8) == 0 && ((row_end - row_start) % 8) == 0) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch());
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const bool lfm2_decode_q6k_raw_batched =
            GetCurrentInferenceWorkContextModelVariant() == ModelVariant::LFM2MOE &&
            phase == InferenceExecutionPhase::Decode;
        const bool projected =
            lfm2_decode_q6k_raw_batched
                ? densecore::RunMoEKQuantRawBatchedProjection(
                      densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                      static_cast<int>(wtype), weight_start, qtile, qrow_bytes, projection_tile.data(),
                      static_cast<int64_t>(tile_assignments.size()), row_count, K, numa_node, allow_parallel)
                : densecore::RunQ6KRepackedMoEProjection(
                      &backend, weight_start, qtile, qrow_bytes, projection_tile.data(),
                      static_cast<int64_t>(tile_assignments.size()), row_count, K, numa_node, allow_parallel);
        if (projected) {
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col +
                                              static_cast<size_t>(row_start + row) * static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }

    const bool kquant_row_pair = (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q5_K || wtype == GGML_TYPE_Q6_K) &&
                                 CanUseGgmlQ4KVecDotRowPairForNativeMoE();
    const bool supports_row_pair = (wtype == GGML_TYPE_Q8_0 && traits->nrows >= 2) || kquant_row_pair;
    const int64_t pair_start = supports_row_pair ? (row_start + 1) / 2 : 0;
    const int64_t pair_end = supports_row_pair ? (row_end / 2) : 0;
    if ((row_start & 1) != 0) {
        const int64_t row = row_start;
        const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
        for (size_t m = 0; m < tile_assignments.size(); ++m) {
            const Qwen35MoEAssignment& assignment = tile_assignments[m];
            const uint8_t* qrow = qtile + m * qrow_bytes;
            float sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
            char* dst_col =
                static_cast<char*>(dst->data) + static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                sum * assignment.weight;
        }
    }
    if (supports_row_pair) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sums[4] = {};
                traits->vec_dot(static_cast<int>(K), sums, 2, w_row, w_row_bytes, qrow, 0, 2);
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sums[0] * assignment.weight;
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0])) +=
                    sums[1] * assignment.weight;
            }
        }
    } else {
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sum = 0.0f;
                traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sum * assignment.weight;
            }
        }
    }
    if ((row_end & 1) != 0 && row_end - 1 >= row_start) {
        const int64_t row = row_end - 1;
        const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
        for (size_t m = 0; m < tile_assignments.size(); ++m) {
            const Qwen35MoEAssignment& assignment = tile_assignments[m];
            const uint8_t* qrow = qtile + m * qrow_bytes;
            float sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
            char* dst_col =
                static_cast<char*>(dst->data) + static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                sum * assignment.weight;
        }
    }
    return true;
}

static void RunQwen35NativeMoEDownQ5KFastPath(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                              const ggml_tensor* selected_experts, int ith, int nth,
                                              Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !dst->data || !down_exps->data || !hidden->data ||
        !selected_experts->data || nth <= 0)
        return;
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow = (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                                      ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                      : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) {
                const int64_t pair_count = n_embd / 2;
                const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
                const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
                for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t row = pair * 2;
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                          &value0, &value1)) {
                        *reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                            static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) = value0;
                        *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                  static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                                  static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                                  static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) =
                            value1;
                    }
                }
                if ((n_embd & 1) != 0 && ith == nth - 1) {
                    const int64_t row = n_embd - 1;
                    float value = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                      &value)) {
                        *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                  static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                  static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                                  static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) = value;
                    }
                }
            });
        }
    }
}

static const float* Qwen35NativeMoEDownWeightPtr(const ggml_tensor* weights, int64_t token, int64_t topk_index) {
    if (!weights || !weights->data || weights->type != GGML_TYPE_F32) return nullptr;
    return reinterpret_cast<const float*>(static_cast<const char*>(weights->data) +
                                          static_cast<size_t>(topk_index) * static_cast<size_t>(weights->nb[1]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(weights->nb[2]));
}

static bool Qwen35NativeMoEComputeTopKWeightsFromLogits(const ggml_tensor* gate_logits,
                                                        const ggml_tensor* selected_experts, int64_t token,
                                                        float* weights, int64_t weights_capacity) {
    if (!gate_logits || !selected_experts || !gate_logits->data || !selected_experts->data || !weights ||
        gate_logits->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 || token < 0 ||
        token >= gate_logits->ne[1] || selected_experts->ne[1] != gate_logits->ne[1]) {
        return false;
    }
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > weights_capacity) {
        return false;
    }

    const char* logits_base = static_cast<const char*>(gate_logits->data);
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    float max_logit = -INFINITY;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert = *reinterpret_cast<const int32_t*>(
            selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
            static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
        if (expert >= 0 && expert < n_experts) {
            const float logit = *reinterpret_cast<const float*>(
                logits_base + static_cast<size_t>(expert) * static_cast<size_t>(gate_logits->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(gate_logits->nb[1]));
            max_logit = std::max(max_logit, logit);
        }
    }
    if (!std::isfinite(max_logit)) {
        max_logit = 0.0f;
    }

    float denom = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert = *reinterpret_cast<const int32_t*>(
            selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
            static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
        float value = 0.0f;
        if (expert >= 0 && expert < n_experts) {
            const float logit = *reinterpret_cast<const float*>(
                logits_base + static_cast<size_t>(expert) * static_cast<size_t>(gate_logits->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(gate_logits->nb[1]));
            value = std::exp(logit - max_logit);
            denom += value;
        }
        weights[k] = value;
    }
    denom = std::max(denom, 6.103515625e-5f);
    for (int64_t k = 0; k < top_k; ++k) {
        weights[k] /= denom;
    }
    return true;
}

static bool ResolveQwen35NativeMoETopKWeights(const Qwen35SharedQ8RowsUserData* shared_q8,
                                              const ggml_tensor* gate_logits, const ggml_tensor* selected_experts,
                                              int64_t token, float* weights, int64_t weights_capacity) {
    if (shared_q8 &&
        CopyQwenNativeMoEFusedRouterWeights(shared_q8->fused_router_state, token, weights, weights_capacity)) {
        return true;
    }
    return Qwen35NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, weights, weights_capacity);
}

static bool LFM2NativeMoEComputeTopKWeightsFromLogits(const ggml_tensor* gate_logits,
                                                      const ggml_tensor* selected_experts, int64_t token,
                                                      float* weights, int64_t weights_capacity, bool normalize_weights,
                                                      float routed_scale) {
    if (!gate_logits || !selected_experts || !gate_logits->data || !selected_experts->data || !weights ||
        gate_logits->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 || token < 0 ||
        token >= gate_logits->ne[1] || selected_experts->ne[1] != gate_logits->ne[1]) {
        return false;
    }
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > weights_capacity) {
        return false;
    }

    const char* logits_base = static_cast<const char*>(gate_logits->data);
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    float denom = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert = *reinterpret_cast<const int32_t*>(
            selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
            static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
        float value = 0.0f;
        if (expert >= 0 && expert < n_experts) {
            const float logit = *reinterpret_cast<const float*>(
                logits_base + static_cast<size_t>(expert) * static_cast<size_t>(gate_logits->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(gate_logits->nb[1]));
            value = 1.0f / (1.0f + std::exp(-logit));
            denom += value;
        }
        weights[k] = value;
    }
    const float scale = (routed_scale != 0.0f && routed_scale != 1.0f) ? routed_scale : 1.0f;
    if (normalize_weights) {
        denom = std::max(denom, 6.103515625e-5f);
        for (int64_t k = 0; k < top_k; ++k) {
            weights[k] = (weights[k] / denom) * scale;
        }
    } else if (scale != 1.0f) {
        for (int64_t k = 0; k < top_k; ++k) {
            weights[k] *= scale;
        }
    }
    return true;
}

static void ProbeQwen35NativeMoEDownWeightedLogitsReference(const ggml_tensor* dst, const ggml_tensor* down_exps,
                                                            const ggml_tensor* hidden,
                                                            const ggml_tensor* selected_experts,
                                                            const ggml_tensor* gate_logits, int64_t row_start,
                                                            int64_t row_end,
                                                            const Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !gate_logits->data || row_start >= row_end) {
        return;
    }
    const int layer_idx = shared_q8 ? shared_q8->debug_layer_idx : -1;
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t target_token_env =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    const int64_t token = target_token_env >= 0 ? target_token_env : n_tokens - 1;
    if (token < 0 || token >= n_tokens ||
        !ShouldRunQwen35NativeMoEReferenceProbe(layer_idx, "down_weighted_logits", token)) {
        return;
    }
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > 64) {
        return;
    }
    float topk_weights[64];
    const bool weights_ok =
        shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
            ? LFM2NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64,
                                                        shared_q8->weighted_logits_norm_topk,
                                                        shared_q8->weighted_logits_scale)
            : ResolveQwen35NativeMoETopKWeights(shared_q8, gate_logits, selected_experts, token, topk_weights, 64);
    if (!weights_ok) {
        return;
    }
    const int64_t K = down_exps->ne[0];
    const size_t w_row_bytes = static_cast<size_t>(down_exps->nb[1]);
    thread_local std::vector<uint8_t> qbuf;
    float max_abs_diff = 0.0f;
    int64_t max_row = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    for (int64_t row = row_start; row < row_end; ++row) {
        float ref = 0.0f;
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) {
                continue;
            }
            const float gate_weight = topk_weights[k];
            if (gate_weight == 0.0f) {
                continue;
            }
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow =
                (down_exps->type != GGML_TYPE_Q8_0 && shared_q8) ? Qwen35SharedQ8RowPtr(shared_q8, k, token) : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            float sum = 0.0f;
            bool ok = false;
            if (down_exps->type == GGML_TYPE_Q8_0 && hidden_row) {
                ok =
                    Qwen35NativeMoEDownQ8_0ReferenceDotRowForExpertWithHidden(down_exps, expert, row, hidden_row, &sum);
            } else {
                const char* expert_base = static_cast<const char*>(down_exps->data) +
                                          static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
                const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
                ok = Qwen35NativeMoEReferenceDotQXK(down_exps->type, w_row, qrow, K, &sum);
            }
            if (ok) {
                ref += sum * gate_weight;
            }
        }
        const auto* actual_ptr = reinterpret_cast<const float*>(
            static_cast<const char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
            static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
        const float actual = *actual_ptr;
        const float diff = std::fabs(actual - ref);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_row = row;
            max_actual = actual;
            max_ref = ref;
        }
    }
    std::fprintf(stderr,
                 "[QWEN35_NATIVE_MOE_REF] layer=%d stage=down_weighted_logits token=%lld rows=[%lld,%lld) "
                 "max_abs_diff=%.8g row=%lld actual=%.8g ref=%.8g tol=%.8g\n",
                 layer_idx, static_cast<long long>(token), static_cast<long long>(row_start),
                 static_cast<long long>(row_end), max_abs_diff, static_cast<long long>(max_row), max_actual, max_ref,
                 Qwen35NativeMoEReferenceTolerance());
}

static void RunQwen35NativeMoEDownQ5KWeightedSumFastPath(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                         const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                                         const ggml_tensor* weights, int ith, int nth,
                                                         Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !weights || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !weights->data || nth <= 0) {
        return;
    }
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != n_embd || dst->ne[1] != n_tokens || hidden->ne[1] != top_k ||
        hidden->ne[2] != n_tokens || weights->type != GGML_TYPE_F32 || weights->ne[0] != 1 || weights->ne[1] != top_k ||
        weights->ne[2] != n_tokens) {
        return;
    }

    const int64_t pair_count = n_embd / 2;
    const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
    const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
    const bool owns_odd_tail = (n_embd & 1) != 0 && ith == nth - 1;
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    thread_local std::vector<Qwen35MoEAssignment> assignments;

    const bool batched_qxk_down =
        down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K || down_exps->type == GGML_TYPE_Q6_K;
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8) {
        size_t shared_assignment_count = 0;
        const Qwen35MoEAssignment* shared_assignments = PrepareQwen35SharedMoEAssignments(
            shared_q8, selected_experts, down_exps->ne[2], ith, &shared_assignment_count);
        if (shared_assignments && shared_assignment_count > 0) {
            assignments.assign(shared_assignments, shared_assignments + shared_assignment_count);
        } else if (!Qwen35BuildMoEAssignments(selected_experts, down_exps->ne[2], &assignments)) {
            assignments.clear();
        }
    }
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8 && !assignments.empty()) {
        for (Qwen35MoEAssignment& assignment : assignments) {
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, assignment.token, assignment.topk_index);
            assignment.weight = weight_ptr ? *weight_ptr : 0.0f;
        }
        const int64_t row_start = (static_cast<int64_t>(ith) * n_embd) / nth;
        const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_embd) / nth;
        if (row_start >= row_end) return;
        Qwen35NativeMoEZeroDst2DRange(dst, row_start, row_end, n_tokens);
        const size_t qrow_bytes = shared_q8->row_bytes;
        thread_local std::vector<uint8_t> qtile;
        thread_local std::vector<Qwen35MoEAssignment> tile_assignments;
        qtile.resize(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments) * qrow_bytes);
        tile_assignments.reserve(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments));
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (size_t tile_start = group_start; tile_start < group_end;) {
                tile_assignments.clear();
                while (tile_start < group_end &&
                       tile_assignments.size() < static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments)) {
                    const Qwen35MoEAssignment& assignment = assignments[tile_start++];
                    if (assignment.weight == 0.0f) {
                        continue;
                    }
                    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token);
                    if (!qrow) {
                        continue;
                    }
                    std::memcpy(qtile.data() + tile_assignments.size() * qrow_bytes, qrow, qrow_bytes);
                    tile_assignments.push_back(assignment);
                }
                if (tile_assignments.empty()) {
                    continue;
                }
                bool ok = false;
                // Bind the thread_local scratch on the CALLING thread. When sticky
                // routing is armed, RunNativeMoEOnExpertNode runs this body on a
                // NUMA-pinned helper thread, where `qtile`/`tile_assignments` resolve
                // to that thread's own empty instances. See W2 in NUMA_TIER1_FINDINGS.md.
                const uint8_t* qtile_ptr = qtile.data();
                const auto& tile_assignments_ref = tile_assignments;
                RunNativeMoEOnExpertNode(shared_q8, expert, [&](int numa_node) {
                    ok = Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
                        dst, down_exps, qtile_ptr, qrow_bytes, tile_assignments_ref, expert, row_start, row_end,
                        numa_node, NativeMoEKernelMayFanOut(shared_q8, numa_node),
                        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8));
                });
                if (!ok) {
                    return;
                }
            }
            group_start = group_end;
        }
        return;
    }
    if (n_tokens > 4 && !assignments.empty()) {
        for (Qwen35MoEAssignment& assignment : assignments) {
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, assignment.token, assignment.topk_index);
            assignment.weight = weight_ptr ? *weight_ptr : 0.0f;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            }
        }
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                for (size_t ai = group_start; ai < group_end; ++ai) {
                    const Qwen35MoEAssignment& assignment = assignments[ai];
                    if (assignment.weight == 0.0f) continue;
                    const float* hidden_row =
                        Qwen35NativeMoEDownHiddenRowPtr(hidden, assignment.token, assignment.topk_index);
                    const uint8_t* qrow = (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                                              ? Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token)
                                              : nullptr;
                    if (!qrow) {
                        if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, assignment.token, assignment.topk_index,
                                                                        down_exps->type, qbuf)) {
                            continue;
                        }
                        qrow = qbuf.data();
                    }
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                           &value0, &value1)) {
                        continue;
                    }
                    float* dst0 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]));
                    float* dst1 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]));
                    *dst0 += value0 * assignment.weight;
                    *dst1 += value1 * assignment.weight;
                }
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                for (size_t ai = group_start; ai < group_end; ++ai) {
                    const Qwen35MoEAssignment& assignment = assignments[ai];
                    if (assignment.weight == 0.0f) continue;
                    const float* hidden_row =
                        Qwen35NativeMoEDownHiddenRowPtr(hidden, assignment.token, assignment.topk_index);
                    const uint8_t* qrow = (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                                              ? Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token)
                                              : nullptr;
                    if (!qrow) {
                        if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, assignment.token, assignment.topk_index,
                                                                        down_exps->type, qbuf)) {
                            continue;
                        }
                        qrow = qbuf.data();
                    }
                    float value = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                      &value)) {
                        float* dst_ptr = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]));
                        *dst_ptr += value * assignment.weight;
                    }
                }
            }
            group_start = group_end;
        }
        return;
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
        if (owns_odd_tail) {
            const int64_t row = n_embd - 1;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }

        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, token, k);
            if (!weight_ptr) continue;
            const float gate_weight = *weight_ptr;
            if (gate_weight == 0.0f) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow = (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                                      ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                      : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) {
                for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t row = pair * 2;
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                           &value0, &value1)) {
                        continue;
                    }
                    float* dst0 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    float* dst1 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    *dst0 += value0 * gate_weight;
                    *dst1 += value1 * gate_weight;
                }
                if (owns_odd_tail) {
                    const int64_t row = n_embd - 1;
                    float value = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                      &value)) {
                        float* dst_ptr = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                        *dst_ptr += value * gate_weight;
                    }
                }
            });
        }
    }
}

static void ReduceNativeMoEExpertOutputsInRoutingOrder(float* dst_base, size_t dst_row_stride,
                                                       const float* expert_outputs, const int* experts,
                                                       const float* weights, int item_count, int64_t row_count) {
    if (!dst_base || !expert_outputs || !experts || !weights || item_count <= 0 || row_count <= 0) {
        return;
    }
    for (int item = 0; item < item_count; ++item) {
        if (experts[item] < 0 || weights[item] == 0.0f) {
            continue;
        }
        const float* item_output = expert_outputs + static_cast<size_t>(item) * static_cast<size_t>(row_count);
        for (int64_t row = 0; row < row_count; ++row) {
            float* dst_value =
                reinterpret_cast<float*>(reinterpret_cast<char*>(dst_base) + static_cast<size_t>(row) * dst_row_stride);
            *dst_value += item_output[row] * weights[item];
        }
    }
}

static bool TryRunQwen35NativeMoEDownWeightedLogitsGroupedDecode(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                                 const ggml_tensor* hidden,
                                                                 const ggml_tensor* selected_experts,
                                                                 const ggml_tensor* gate_logits,
                                                                 Qwen35SharedQ8RowsUserData* shared_q8);

static bool ShouldUseQwen35NativeMoEDownGlobalRowPartition(bool numa_sticky_enabled, bool weighted_logits_lfm2_sigmoid,
                                                           InferenceExecutionPhase phase, int64_t n_tokens, int nth) {
    return numa_sticky_enabled && !weighted_logits_lfm2_sigmoid && phase == InferenceExecutionPhase::Decode &&
           n_tokens == 1 && nth > 1;
}

static bool ShouldNarrowNativeMoECallbackTasksForStickyFallback(bool sticky_armed, InferenceExecutionPhase phase,
                                                                bool direct_outer_candidate) {
    return sticky_armed && phase == InferenceExecutionPhase::Decode && !direct_outer_candidate;
}

static bool TryRunQwen35NativeMoEDownWeightedLogitsDirectOuterTasks(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                                    const ggml_tensor* hidden,
                                                                    const ggml_tensor* selected_experts,
                                                                    const ggml_tensor* gate_logits, int ith, int nth,
                                                                    Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !shared_q8 || !shared_q8->numa_backend ||
        !shared_q8->numa_sticky_enabled || shared_q8->weighted_logits_lfm2_sigmoid ||
        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8) != InferenceExecutionPhase::Decode ||
        selected_experts->ne[1] != 1 || selected_experts->ne[0] <= 0 ||
        selected_experts->ne[0] > kQwen35NativeMoEMaxGroupedExperts ||
        (down_exps->type != GGML_TYPE_Q4_K && down_exps->type != GGML_TYPE_Q5_K && down_exps->type != GGML_TYPE_Q6_K) ||
        dst->ne[0] != down_exps->ne[1] || nth <= 1) {
        return false;
    }

    const int top_k = static_cast<int>(selected_experts->ne[0]);
    const int64_t n_embd = down_exps->ne[1];
    const size_t output_count = static_cast<size_t>(top_k) * static_cast<size_t>(n_embd);
    bool initialize_ok = shared_q8->direct_expert_outputs && shared_q8->direct_expert_output_capacity >= output_count;
    if (ith == 0 && initialize_ok) {
        std::memset(shared_q8->direct_expert_outputs, 0, output_count * sizeof(float));
    }

    NativeMoEOuterTaskPlan task_plan;
    bool task_ok = ResolveNativeMoEOuterTaskPlan(shared_q8, ith, nth, &task_plan) && initialize_ok;

    std::array<int, kQwen35NativeMoEMaxGroupedExperts> experts{};
    std::array<float, kQwen35NativeMoEMaxGroupedExperts> weights{};
    std::array<const uint8_t*, kQwen35NativeMoEMaxGroupedExperts> qrows{};
    std::array<const float*, kQwen35NativeMoEMaxGroupedExperts> hidden_rows{};
    experts.fill(-1);
    task_ok = task_ok &&
              ResolveQwen35NativeMoETopKWeights(shared_q8, gate_logits, selected_experts, 0, weights.data(), top_k);
    for (int item = 0; item < top_k && task_ok; ++item) {
        int32_t expert = -1;
        if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, 0, item, &expert)) {
            task_ok = false;
            break;
        }
        experts[static_cast<size_t>(item)] = expert;
        qrows[static_cast<size_t>(item)] = Qwen35SharedQ8RowPtr(shared_q8, item, 0);
        hidden_rows[static_cast<size_t>(item)] = Qwen35NativeMoEDownHiddenRowPtr(hidden, 0, item);
        const int expert_node = ResolveNativeMoEExpertNumaNode(shared_q8, expert);
        int node_task_count = 0;
        for (int task = 0; task < nth; ++task) {
            node_task_count += shared_q8->outer_task_nodes[task] == expert_node ? 1 : 0;
        }
        if (!qrows[static_cast<size_t>(item)] || !hidden_rows[static_cast<size_t>(item)] || expert_node < 0 ||
            node_task_count == 0) {
            task_ok = false;
        }
    }

    // ParityGate probing is single-owner. Concurrent probes return a temporary
    // deny verdict, so letting every GGML task resolve admission can race and
    // publish a nondeterministic final path. Task 0 resolves before publishing
    // the direct epoch; acquire readers then consume the stable cached verdict.
    if (ith == 0 && task_ok && down_exps->type == GGML_TYPE_Q5_K) {
        (void)ResolveQwen35NativeMoEDownQ5KSingleRowAdmission(shared_q8, down_exps, experts[0], qrows[0]);
    }
    const uint64_t epoch = BeginNativeMoEDirectOuterTask(shared_q8, ith, nth, task_ok);
    task_ok = task_ok && epoch != 0;
    const bool use_densecore_q5k_single_row = task_ok && down_exps->type == GGML_TYPE_Q5_K &&
                                              shared_q8->q5k_single_row_admission.load(std::memory_order_acquire) == 2;
    const int64_t pair_count = n_embd / 2;
    const int64_t pair_start = task_ok ? (pair_count * task_plan.node_task_rank) / task_plan.node_task_count : 0;
    const int64_t pair_end = task_ok ? (pair_count * (task_plan.node_task_rank + 1)) / task_plan.node_task_count : 0;
    for (int item = 0; item < top_k && task_ok; ++item) {
        const int expert = experts[static_cast<size_t>(item)];
        if (ResolveNativeMoEExpertNumaNode(shared_q8, expert) != task_plan.task_node) {
            continue;
        }
        float* item_output = shared_q8->direct_expert_outputs + static_cast<size_t>(item) * static_cast<size_t>(n_embd);
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(
                    down_exps, expert, row, qrows[static_cast<size_t>(item)], hidden_rows[static_cast<size_t>(item)],
                    &value0, &value1, use_densecore_q5k_single_row)) {
                task_ok = false;
                break;
            }
            item_output[row] = value0;
            item_output[row + 1] = value1;
        }
        if (task_ok && (n_embd & 1) != 0 && task_plan.node_task_rank == task_plan.node_task_count - 1) {
            float value = 0.0f;
            if (!Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(
                    down_exps, expert, n_embd - 1, qrows[static_cast<size_t>(item)],
                    hidden_rows[static_cast<size_t>(item)], &value, use_densecore_q5k_single_row)) {
                task_ok = false;
                break;
            }
            item_output[n_embd - 1] = value;
        }
    }

    bool owner_task = false;
    const bool all_ok = FinishNativeMoEDirectOuterTask(shared_q8, ith, epoch, task_ok, &owner_task);
    if (!owner_task) {
        return true;
    }
    if (!all_ok) {
        shared_q8->outer_task_width.store(1, std::memory_order_relaxed);
        return TryRunQwen35NativeMoEDownWeightedLogitsGroupedDecode(dst, down_exps, hidden, selected_experts,
                                                                    gate_logits, shared_q8);
    }
    Qwen35NativeMoEZeroDst2DRange(dst, 0, n_embd, 1);
    ReduceNativeMoEExpertOutputsInRoutingOrder(static_cast<float*>(dst->data), static_cast<size_t>(dst->nb[0]),
                                               shared_q8->direct_expert_outputs, experts.data(), weights.data(), top_k,
                                               n_embd);
    static std::atomic<bool> logged_direct_down{false};
    if (!logged_direct_down.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[NUMA] Qwen native MoE direct outer-task decode active "
                     "(stage=down, tasks=%d, experts=%d)\n",
                     nth, top_k);
    }
    return true;
}

static bool TryRunQwen35NativeMoEDownWeightedLogitsGroupedDecode(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                                 const ggml_tensor* hidden,
                                                                 const ggml_tensor* selected_experts,
                                                                 const ggml_tensor* gate_logits,
                                                                 Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !shared_q8 || !shared_q8->numa_backend ||
        !shared_q8->numa_sticky_enabled || shared_q8->weighted_logits_lfm2_sigmoid ||
        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8) != InferenceExecutionPhase::Decode ||
        selected_experts->ne[1] != 1 || selected_experts->ne[0] <= 0 ||
        selected_experts->ne[0] > kQwen35NativeMoEMaxGroupedExperts ||
        (down_exps->type != GGML_TYPE_Q4_K && down_exps->type != GGML_TYPE_Q5_K && down_exps->type != GGML_TYPE_Q6_K) ||
        dst->ne[0] != down_exps->ne[1]) {
        return false;
    }
    const int outer_width = shared_q8->outer_task_width.load(std::memory_order_relaxed);
    if (outer_width > 1) {
        return false;
    }

    const int top_k = static_cast<int>(selected_experts->ne[0]);
    const int64_t n_embd = down_exps->ne[1];
    std::array<int, kQwen35NativeMoEMaxGroupedExperts> experts{};
    std::array<float, kQwen35NativeMoEMaxGroupedExperts> weights{};
    std::array<const uint8_t*, kQwen35NativeMoEMaxGroupedExperts> qrows{};
    std::array<const float*, kQwen35NativeMoEMaxGroupedExperts> hidden_rows{};
    experts.fill(-1);
    if (!ResolveQwen35NativeMoETopKWeights(shared_q8, gate_logits, selected_experts, 0, weights.data(), top_k)) {
        return false;
    }
    for (int k = 0; k < top_k; ++k) {
        int32_t expert = -1;
        if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, 0, k, &expert)) {
            continue;
        }
        const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, k, 0);
        const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, 0, k);
        if (!qrow || !hidden_row) {
            return false;
        }
        experts[static_cast<size_t>(k)] = expert;
        qrows[static_cast<size_t>(k)] = qrow;
        hidden_rows[static_cast<size_t>(k)] = hidden_row;
    }
    bool use_densecore_q5k_single_row = false;
    if (down_exps->type == GGML_TYPE_Q5_K) {
        for (int k = 0; k < top_k; ++k) {
            if (experts[static_cast<size_t>(k)] >= 0 && qrows[static_cast<size_t>(k)]) {
                use_densecore_q5k_single_row = ResolveQwen35NativeMoEDownQ5KSingleRowAdmission(
                    shared_q8, down_exps, experts[static_cast<size_t>(k)], qrows[static_cast<size_t>(k)]);
                break;
            }
        }
    }

    thread_local std::vector<float> expert_outputs;
    expert_outputs.assign(static_cast<size_t>(top_k) * static_cast<size_t>(n_embd), 0.0f);
    float* expert_output_data = expert_outputs.data();
    std::atomic<bool> all_ok{true};
    const bool grouped = RunNativeMoEOnExpertNodeGroups(
        shared_q8, experts.data(), top_k, [&](int node, const NativeMoENodeGroupPlan& plan) {
            std::array<int, kQwen35NativeMoEMaxGroupedExperts> group_items{};
            int group_count = 0;
            for (int item = 0; item < plan.item_count; ++item) {
                if (plan.item_nodes[static_cast<size_t>(item)] == node) {
                    group_items[static_cast<size_t>(group_count++)] = item;
                }
            }
            if (group_count == 0) {
                return;
            }
            const int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount(node);
            const int worker_count = std::max(1, physical_cores > 0 ? physical_cores : group_count);
            const int splits_per_expert = std::max(1, (worker_count + group_count - 1) / group_count);
            const int total_units = group_count * splits_per_expert;
            shared_q8->numa_backend->ParallelFor(
                total_units,
                [&](int unit_start, int unit_end, int) {
                    for (int unit = unit_start; unit < unit_end && all_ok.load(std::memory_order_relaxed); ++unit) {
                        const int item = group_items[static_cast<size_t>(unit / splits_per_expert)];
                        const int split = unit % splits_per_expert;
                        const int64_t pair_count = n_embd / 2;
                        const int64_t pair_start = (pair_count * split) / splits_per_expert;
                        const int64_t pair_end = (pair_count * (split + 1)) / splits_per_expert;
                        float* item_output =
                            expert_output_data + static_cast<size_t>(item) * static_cast<size_t>(n_embd);
                        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                            const int64_t row = pair * 2;
                            float value0 = 0.0f;
                            float value1 = 0.0f;
                            if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(
                                    down_exps, experts[static_cast<size_t>(item)], row,
                                    qrows[static_cast<size_t>(item)], hidden_rows[static_cast<size_t>(item)], &value0,
                                    &value1, use_densecore_q5k_single_row)) {
                                all_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            item_output[row] = value0;
                            item_output[row + 1] = value1;
                        }
                        if ((n_embd & 1) != 0 && split == splits_per_expert - 1) {
                            float value = 0.0f;
                            if (!Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(
                                    down_exps, experts[static_cast<size_t>(item)], n_embd - 1,
                                    qrows[static_cast<size_t>(item)], hidden_rows[static_cast<size_t>(item)], &value,
                                    use_densecore_q5k_single_row)) {
                                all_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                            item_output[n_embd - 1] = value;
                        }
                    }
                },
                node);
        });
    if (!grouped || !all_ok.load(std::memory_order_relaxed)) {
        return false;
    }

    Qwen35NativeMoEZeroDst2DRange(dst, 0, n_embd, 1);
    ReduceNativeMoEExpertOutputsInRoutingOrder(static_cast<float*>(dst->data), static_cast<size_t>(dst->nb[0]),
                                               expert_output_data, experts.data(), weights.data(), top_k, n_embd);
    GetNativeMoENumaCounters().grouped_decode_used_ops.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<bool> logged_grouped_down{false};
    if (!logged_grouped_down.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[NUMA] Qwen native MoE grouped decode active (stage=down, experts=%d)\n", top_k);
    }
    return true;
}

static void RunQwen35NativeMoEDownQ5KWeightedLogitsFastPath(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                            const ggml_tensor* hidden,
                                                            const ggml_tensor* selected_experts,
                                                            const ggml_tensor* gate_logits, int ith, int nth,
                                                            Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !gate_logits->data || nth <= 0) {
        return;
    }
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != n_embd || dst->ne[1] != n_tokens || hidden->ne[1] != top_k ||
        hidden->ne[2] != n_tokens || gate_logits->type != GGML_TYPE_F32 || gate_logits->ne[1] != n_tokens ||
        gate_logits->ne[0] != down_exps->ne[2] || top_k > 64) {
        return;
    }

    thread_local std::vector<uint8_t> qbuf;
    const int64_t pair_count = n_embd / 2;
    const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
    const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
    const bool owns_odd_tail = (n_embd & 1) != 0 && ith == nth - 1;
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<Qwen35MoEAssignment> assignments;

    // Physical 2-socket A/B showed that expert-local W2 materialization and its
    // owner reduction cost more than locality saves. Keep sticky gate/up, but
    // let all outer tasks row-partition W2 and accumulate in routing order.
    const bool use_global_row_partition =
        shared_q8 && ShouldUseQwen35NativeMoEDownGlobalRowPartition(
                         shared_q8->numa_sticky_enabled, shared_q8->weighted_logits_lfm2_sigmoid,
                         ResolveNativeMoEOuterTaskExecutionPhase(shared_q8), n_tokens, nth);

    if (!use_global_row_partition && use_shared_q8 &&
        TryRunQwen35NativeMoEDownWeightedLogitsDirectOuterTasks(dst, down_exps, hidden, selected_experts, gate_logits,
                                                                ith, nth, shared_q8)) {
        if (ith == 0) {
            ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits, 0,
                                                            n_embd, shared_q8);
        }
        return;
    }
    if (ith == 0 && nth == 1 && use_shared_q8 &&
        TryRunQwen35NativeMoEDownWeightedLogitsGroupedDecode(dst, down_exps, hidden, selected_experts, gate_logits,
                                                             shared_q8)) {
        ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits, 0,
                                                        n_embd, shared_q8);
        return;
    }

    const bool batched_qxk_down =
        down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K || down_exps->type == GGML_TYPE_Q6_K;
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8) {
        size_t shared_assignment_count = 0;
        const Qwen35MoEAssignment* shared_assignments = PrepareQwen35SharedMoEAssignments(
            shared_q8, selected_experts, down_exps->ne[2], ith, &shared_assignment_count);
        if (shared_assignments && shared_assignment_count > 0) {
            assignments.assign(shared_assignments, shared_assignments + shared_assignment_count);
        } else if (!Qwen35BuildMoEAssignments(selected_experts, down_exps->ne[2], &assignments)) {
            assignments.clear();
        }
    }
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8 && !assignments.empty()) {
        const int64_t row_start = (static_cast<int64_t>(ith) * n_embd) / nth;
        const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_embd) / nth;
        if (row_start >= row_end) return;
        thread_local std::vector<float> topk_weight_matrix;
        topk_weight_matrix.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(top_k), 0.0f);
        for (int64_t token = 0; token < n_tokens; ++token) {
            float topk_weights[64];
            const bool weights_ok = shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
                                        ? LFM2NativeMoEComputeTopKWeightsFromLogits(
                                              gate_logits, selected_experts, token, topk_weights, 64,
                                              shared_q8->weighted_logits_norm_topk, shared_q8->weighted_logits_scale)
                                        : ResolveQwen35NativeMoETopKWeights(shared_q8, gate_logits, selected_experts,
                                                                            token, topk_weights, 64);
            if (!weights_ok) {
                continue;
            }
            float* token_weights = topk_weight_matrix.data() + static_cast<size_t>(token) * static_cast<size_t>(top_k);
            for (int64_t k = 0; k < top_k; ++k) {
                token_weights[static_cast<size_t>(k)] = topk_weights[k];
            }
        }
        for (Qwen35MoEAssignment& assignment : assignments) {
            assignment.weight = 0.0f;
            if (assignment.token >= 0 && assignment.token < n_tokens && assignment.topk_index >= 0 &&
                assignment.topk_index < top_k) {
                assignment.weight =
                    topk_weight_matrix[static_cast<size_t>(assignment.token) * static_cast<size_t>(top_k) +
                                       static_cast<size_t>(assignment.topk_index)];
            }
        }
        Qwen35NativeMoEZeroDst2DRange(dst, row_start, row_end, n_tokens);
        const size_t qrow_bytes = shared_q8->row_bytes;
        thread_local std::vector<uint8_t> qtile;
        thread_local std::vector<Qwen35MoEAssignment> tile_assignments;
        qtile.resize(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments) * qrow_bytes);
        tile_assignments.reserve(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments));
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (size_t tile_start = group_start; tile_start < group_end;) {
                tile_assignments.clear();
                while (tile_start < group_end &&
                       tile_assignments.size() < static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments)) {
                    const Qwen35MoEAssignment& assignment = assignments[tile_start++];
                    if (assignment.weight == 0.0f) {
                        continue;
                    }
                    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token);
                    if (!qrow) {
                        continue;
                    }
                    std::memcpy(qtile.data() + tile_assignments.size() * qrow_bytes, qrow, qrow_bytes);
                    tile_assignments.push_back(assignment);
                }
                if (tile_assignments.empty()) {
                    continue;
                }
                bool ok = false;
                // Bind the thread_local scratch on the CALLING thread. When sticky
                // routing is armed, RunNativeMoEOnExpertNode runs this body on a
                // NUMA-pinned helper thread, where `qtile`/`tile_assignments` resolve
                // to that thread's own empty instances. See W2 in NUMA_TIER1_FINDINGS.md.
                const uint8_t* qtile_ptr = qtile.data();
                const auto& tile_assignments_ref = tile_assignments;
                RunNativeMoEOnExpertNode(shared_q8, expert, [&](int numa_node) {
                    ok = Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
                        dst, down_exps, qtile_ptr, qrow_bytes, tile_assignments_ref, expert, row_start, row_end,
                        numa_node, NativeMoEKernelMayFanOut(shared_q8, numa_node),
                        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8));
                });
                if (!ok) {
                    return;
                }
            }
            group_start = group_end;
        }
        ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits,
                                                        row_start, row_end, shared_q8);
        return;
    }

    NativeMoELegacyDispatchBatch legacy_dispatches(GetNativeMoENumaCounters().legacy_dispatch_ops);
    const bool non_sticky_dispatch = !shared_q8 || !shared_q8->numa_sticky_enabled;
    const bool use_q4_native_pair =
        shared_q8 && ShouldUseQwen36Q4DownRowPair(shared_q8->model_variant,
                                                  ResolveNativeMoEOuterTaskExecutionPhase(shared_q8), n_tokens);
    bool used_q4_native_pair = false;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
        if (owns_odd_tail) {
            const int64_t row = n_embd - 1;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }

        float topk_weights[64];
        const bool weights_ok =
            shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
                ? LFM2NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64,
                                                            shared_q8->weighted_logits_norm_topk,
                                                            shared_q8->weighted_logits_scale)
                : ResolveQwen35NativeMoETopKWeights(shared_q8, gate_logits, selected_experts, token, topk_weights, 64);
        if (!weights_ok) {
            continue;
        }
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float gate_weight = topk_weights[k];
            if (gate_weight == 0.0f) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow = (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                                      ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                      : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            const auto accumulate_expert_rows = [&] {
                for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                    const int64_t row = pair * 2;
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                           &value0, &value1, false, use_q4_native_pair,
                                                                           &used_q4_native_pair)) {
                        continue;
                    }
                    float* dst0 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    float* dst1 = reinterpret_cast<float*>(
                        static_cast<char*>(dst->data) + static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                        static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    *dst0 += value0 * gate_weight;
                    *dst1 += value1 * gate_weight;
                }
                if (owns_odd_tail) {
                    const int64_t row = n_embd - 1;
                    float value = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                      &value)) {
                        float* dst_ptr = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                        *dst_ptr += value * gate_weight;
                    }
                }
            };
            if (use_global_row_partition) {
                accumulate_expert_rows();
            } else if (non_sticky_dispatch) {
                legacy_dispatches.Record();
                accumulate_expert_rows();
            } else {
                RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) { accumulate_expert_rows(); });
            }
        }
    }
    legacy_dispatches.Flush();
#ifdef DENSECORE_TEST_BUILD
    if (used_q4_native_pair) q4_down_native_pair_test_ops.fetch_add(1, std::memory_order_relaxed);
#endif
    const int64_t row_start = pair_start * 2;
    const int64_t row_end = pair_end * 2 + (owns_odd_tail ? 1 : 0);
    ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits, row_start,
                                                    std::min<int64_t>(row_end, n_embd), shared_q8);
}

static void cb_qwen35_native_moe_down_q5k(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(shared_q8 ? shared_q8->work_ctx : nullptr);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KFastPath(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                      dst ? dst->src[2] : nullptr, effective_ith, effective_nth, shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                          /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

static void cb_qwen35_native_moe_down_q5k_weighted_sum(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(shared_q8 ? shared_q8->work_ctx : nullptr);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] WEIGHTED CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KWeightedSumFastPath(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                                 dst ? dst->src[2] : nullptr, dst ? dst->src[3] : nullptr,
                                                 effective_ith, effective_nth, shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                          /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

void cb_gemma4_native_moe_down_weighted_sum(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(shared_q8 ? shared_q8->work_ctx : nullptr);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    RunQwen35NativeMoEDownQ5KWeightedSumFastPath(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                                 dst ? dst->src[2] : nullptr, dst ? dst->src[3] : nullptr,
                                                 effective_ith, effective_nth, shared_q8);
    if (ith == 0) {
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordGemma4DecodeNativeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, wall_ns,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/1,
                                         /*duplicate_work_detected=*/false);
    }
}

static void cb_qwen35_native_moe_down_q5k_weighted_logits(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(shared_q8 ? shared_q8->work_ctx : nullptr);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] WEIGHTED LOGITS CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KWeightedLogitsFastPath(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                                    dst ? dst->src[2] : nullptr, dst ? dst->src[3] : nullptr,
                                                    effective_ith, effective_nth, shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                          /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

static bool CanReplaceQwen35W2WithCustomCallback(const TransformerModel* model, const ggml_tensor* down_exps,
                                                 const ggml_tensor* hidden, const ggml_tensor* selected_experts) {
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] rejected: %s\n", reason);
        }
        return false;
    };
    if (!model || !down_exps || !hidden || !selected_experts) return reject("null_ptr");
    const bool qwen_native_moe = (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                                 model->arch_flags.is_hybrid_ssm;
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (!qwen_native_moe && !lfm2_native_moe) return reject("wrong_variant");
    if (model->hparams.n_experts <= 0) return reject("no_experts");
    const bool supported_down_quant = lfm2_native_moe
                                          ? (down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K ||
                                             down_exps->type == GGML_TYPE_Q6_K || down_exps->type == GGML_TYPE_Q8_0)
                                          : (down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K ||
                                             down_exps->type == GGML_TYPE_Q6_K || down_exps->type == GGML_TYPE_Q8_0);
    if (!supported_down_quant) {
        return reject("unsupported_w2_quant");
    }
    if (down_exps->type == GGML_TYPE_Q6_K) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
        if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
            return reject("q6k_vec_dot_unavailable");
        }
    }
    if (hidden->type != GGML_TYPE_F32) return reject("hidden_not_f32");
    if (selected_experts->type != GGML_TYPE_I32) return reject("experts_not_i32");
    if (down_exps->ne[0] % ggml_blck_size(down_exps->type) != 0) return reject("bad_alignment");
    if ((hidden->ne[0] % QK_K) != 0) return reject("bad_q8k_alignment");
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return reject("unsupported_phase");
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool fast_moe_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, phase, fast_config.native_moe_fast_decode);
    if (!fast_moe_enabled) {
        return reject("native_moe_fast_decode_disabled");
    }
    if (current_batch && !current_batch->lora_map.empty()) {
        return reject("dynamic_lora");
    }
    if (selected_experts->ne[1] <= 0 || selected_experts->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        std::fprintf(stderr, "[W2_Q5K_DIAG] ACCEPTED — replacing ggml_mul_mat_id with custom Q5K callback\n");
    }
    return true;
}

static bool CanFuseQwen35W2NormWeightsFromLogitsWithCustomCallback(const TransformerModel* model,
                                                                   const ggml_tensor* down_exps,
                                                                   const ggml_tensor* hidden,
                                                                   const ggml_tensor* selected_experts,
                                                                   const ggml_tensor* gate_logits) {
    if (!CanReplaceQwen35W2WithCustomCallback(model, down_exps, hidden, selected_experts)) {
        return false;
    }
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] weighted_logits rejected: %s\n", reason);
        }
        return false;
    };
    if (!gate_logits || gate_logits->type != GGML_TYPE_F32) return reject("logits_not_f32");
    if (gate_logits->ne[0] != down_exps->ne[2] || gate_logits->ne[1] != selected_experts->ne[1]) {
        return reject("logits_shape");
    }
    if (selected_experts->ne[0] <= 0 || selected_experts->ne[0] > 64) return reject("topk_too_large");
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (lfm2_native_moe) {
        return true;
    }
    if (!model->moe_norm_topk_prob) return reject("topk_not_normalized");
    if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
        return reject("scaled_weights");
    }
    return true;
}

static bool CanFuseQwen35W2WeightedSumWithCustomCallback(const TransformerModel* model, const ggml_tensor* down_exps,
                                                         const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                                         const ggml_tensor* weights) {
    if (!CanReplaceQwen35W2WithCustomCallback(model, down_exps, hidden, selected_experts)) {
        return false;
    }
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] weighted_sum rejected: %s\n", reason);
        }
        return false;
    };
    if (!weights || weights->type != GGML_TYPE_F32) return reject("weights_not_f32");
    if (weights->ne[0] != 1 || weights->ne[1] != selected_experts->ne[0] || weights->ne[2] != selected_experts->ne[1]) {
        return reject("weights_shape");
    }
    return true;
}

static const float* Qwen35NativeMoEGateUpInputRowPtr(const ggml_tensor* input, int64_t token) {
    if (!input || !input->data || input->type != GGML_TYPE_F32 || token < 0 || token >= input->ne[1]) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(static_cast<const char*>(input->data) +
                                          static_cast<size_t>(token) * static_cast<size_t>(input->nb[1]));
}

static bool Qwen35NativeMoEGateUpQuantizeInput(const ggml_tensor* input, int64_t token, std::vector<uint8_t>& qbuf) {
    const float* input_row = Qwen35NativeMoEGateUpInputRowPtr(input, token);
    if (!input_row) return false;
    const int64_t cols = input->ne[0];
    const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    if (qrow_bytes == 0) return false;
    qbuf.resize(qrow_bytes);
    return Qwen35NativeQuantizeRowQ8K(input_row, qbuf.data(), cols);
}

static bool Qwen35NativeMoEQ4KQ8KDotRow(const void* weight_row, const uint8_t* qrow, int64_t cols, float* out_value) {
    if (PreferGgmlQ4KVecDotForNativeMoE()) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
            (cols % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
            traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qrow, 0, 1);
            return true;
        }
    }
    return densecore::hwy_kernels::DotQ4KQ8K_Hwy(weight_row, qrow, cols, out_value);
}

static bool Qwen35NativeMoEKQ8KDotRow(ggml_type weight_type, const void* weight_row, const uint8_t* qrow, int64_t cols,
                                      float* out_value) {
    if (weight_type == GGML_TYPE_Q4_K) {
        return Qwen35NativeMoEQ4KQ8KDotRow(weight_row, qrow, cols, out_value);
    }
    if (weight_type == GGML_TYPE_Q5_K) {
        return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, qrow, cols, out_value);
    }
    return false;
}

static bool Qwen35NativeMoEKQ8KFusedSwiGLURows(InferenceWorkContext* work_ctx, ggml_type weight_type,
                                               const void* gate_row_start, const void* up_row_start,
                                               const uint8_t* qrow, int64_t cols, int64_t row_count, size_t row_bytes,
                                               float* out_start) {
    if (!gate_row_start || !up_row_start || !qrow || !out_start || cols <= 0 || row_count <= 0 || row_bytes == 0) {
        return false;
    }
    if (weight_type != GGML_TYPE_Q4_K && weight_type != GGML_TYPE_Q5_K) {
        return false;
    }
    if (weight_type == GGML_TYPE_Q5_K) {
        RecordLFM2NativeMoEW1W3Kernel(work_ctx, "q5k_hwy");
        const auto* gate_base = static_cast<const char*>(gate_row_start);
        const auto* up_base = static_cast<const char*>(up_row_start);
        for (int64_t row = 0; row < row_count; ++row) {
            const void* gate_row = gate_base + static_cast<size_t>(row) * row_bytes;
            const void* up_row = up_base + static_cast<size_t>(row) * row_bytes;
            float gate = 0.0f;
            float up = 0.0f;
            if (!densecore::hwy_kernels::DotQ5KQ8K_Hwy(gate_row, qrow, cols, &gate) ||
                !densecore::hwy_kernels::DotQ5KQ8K_Hwy(up_row, qrow, cols, &up)) {
                return false;
            }
            out_start[row] = NativeMoESiLU(gate) * up;
        }
        return true;
    }
    // LFM2 W1/W3 (Q4_K gate/up) decode: routing this fused-SwiGLU through ggml's
    // K-quant 2-row vec_dot (rowpair, nrc=2) produced long-form repetition and
    // garbage on ARM. The C4A final-sweep showed lfm2_w1w3_q4k_vecdot_rowpair
    // dominating while QA failed (repetition_detected;garbage_output_detected),
    // whereas C4 (x86) used the validated Highway kernel (lfm2_w1w3_q4k_hwy) and
    // passed. The row-pair K-quant vec_dot is not parity-verified for this
    // fused-SwiGLU shape/input-layout, so force the known-good Highway path for
    // Q4_K on every ISA. This matches x86, where PreferGgmlQ4KVecDotForNativeMoE()
    // is already false and the ggml vec_dot branch was never taken. Q5_K above
    // already uses Highway unconditionally. Re-enabling the row-pair fast path
    // requires wiring it through ParityGate (kernel_caps.h) so a divergent kernel
    // auto-falls back to this Highway reference. Qwen36 shared-expert tiles below
    // use that separate gate; this single-input path stays on Highway.
    RecordLFM2NativeMoEW1W3Kernel(work_ctx, "q4k_hwy");
    return densecore::hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(gate_row_start, up_row_start, qrow, cols, row_count,
                                                             row_bytes, out_start);
}

static bool TryRunQwen35NativeMoEGateUpGroupedDecode(ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                     const ggml_tensor* up_exps, const ggml_tensor* selected_experts,
                                                     int64_t row_start, int64_t row_end,
                                                     Qwen35SharedQ8RowsUserData* shared_q8);

static bool TryRunQwen35NativeMoEGateUpDirectOuterTasks(ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                        const ggml_tensor* up_exps, const ggml_tensor* selected_experts,
                                                        int ith, int nth, Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !selected_experts || !shared_q8 || !shared_q8->numa_backend ||
        !shared_q8->numa_sticky_enabled || shared_q8->record_lfm2_w1w3_kernel ||
        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8) != InferenceExecutionPhase::Decode ||
        selected_experts->ne[1] != 1 || selected_experts->ne[0] <= 0 ||
        selected_experts->ne[0] > kQwen35NativeMoEMaxGroupedExperts || gate_exps->type != GGML_TYPE_Q4_K ||
        up_exps->type != GGML_TYPE_Q4_K || dst->nb[0] != static_cast<int64_t>(sizeof(float)) || nth <= 1) {
        return false;
    }
    const ggml_tensor* prepacked = shared_q8->prepacked_fused_gate_up_exps;
    const int64_t row_count = dst->ne[0];
    if (!prepacked || !prepacked->data || prepacked->type != GGML_TYPE_Q4_K || prepacked->ne[0] != gate_exps->ne[0] ||
        prepacked->ne[1] != 2 * row_count || prepacked->ne[2] != gate_exps->ne[2] || (row_count % 8) != 0) {
        return false;
    }

    NativeMoEOuterTaskPlan task_plan;
    bool task_ok = ResolveNativeMoEOuterTaskPlan(shared_q8, ith, nth, &task_plan);
    const uint64_t epoch = BeginNativeMoEDirectOuterTask(shared_q8, ith, nth, /*initialize_ok=*/true);
    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, 0, 0);
    task_ok = task_ok && qrow != nullptr;
    const int tile_count = static_cast<int>(row_count / 8);
    const int tile_start = task_ok ? (tile_count * task_plan.node_task_rank) / task_plan.node_task_count : 0;
    const int tile_end = task_ok ? (tile_count * (task_plan.node_task_rank + 1)) / task_plan.node_task_count : 0;
    const int top_k = static_cast<int>(selected_experts->ne[0]);
    for (int item = 0; item < top_k && task_ok; ++item) {
        int32_t expert = -1;
        if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, 0, item, &expert)) {
            task_ok = false;
            break;
        }
        const int expert_node = ResolveNativeMoEExpertNumaNode(shared_q8, expert);
        int node_task_count = 0;
        for (int task = 0; task < nth; ++task) {
            node_task_count += shared_q8->outer_task_nodes[task] == expert_node ? 1 : 0;
        }
        if (expert_node < 0 || node_task_count == 0) {
            task_ok = false;
            break;
        }
        if (expert_node != task_plan.task_node || tile_end <= tile_start) {
            continue;
        }
        const void* fused_expert = static_cast<const char*>(prepacked->data) +
                                   static_cast<size_t>(expert) * static_cast<size_t>(prepacked->nb[2]);
        float* out = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                              static_cast<size_t>(item) * static_cast<size_t>(dst->nb[1]));
        task_ok = densecore::RunQ4KPrepackedMoEFusedSwiGLUTileRange(fused_expert, qrow, out, row_count,
                                                                    gate_exps->ne[0], tile_start, tile_end);
    }
    bool owner_task = false;
    const bool all_ok = FinishNativeMoEDirectOuterTask(shared_q8, ith, epoch, task_ok, &owner_task);
    if (!owner_task) {
        return true;
    }
    if (!all_ok) {
        shared_q8->outer_task_width.store(1, std::memory_order_relaxed);
        return TryRunQwen35NativeMoEGateUpGroupedDecode(dst, gate_exps, up_exps, selected_experts, 0, row_count,
                                                        shared_q8);
    }
    static std::atomic<bool> logged_direct_gateup{false};
    if (!logged_direct_gateup.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[NUMA] Qwen native MoE direct outer-task decode active "
                     "(stage=gateup, tasks=%d, experts=%d)\n",
                     nth, top_k);
    }
    return true;
}

static bool TryRunQwen35NativeMoEGateUpGroupedDecode(ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                     const ggml_tensor* up_exps, const ggml_tensor* selected_experts,
                                                     int64_t row_start, int64_t row_end,
                                                     Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !selected_experts || !shared_q8 || !shared_q8->numa_backend ||
        !shared_q8->numa_sticky_enabled || shared_q8->record_lfm2_w1w3_kernel ||
        GetCurrentExecutionPhase() != InferenceExecutionPhase::Decode || selected_experts->ne[1] != 1 ||
        selected_experts->ne[0] <= 0 || selected_experts->ne[0] > kQwen35NativeMoEMaxGroupedExperts ||
        gate_exps->type != GGML_TYPE_Q4_K || up_exps->type != GGML_TYPE_Q4_K ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float)) || row_start < 0 || row_end <= row_start) {
        return false;
    }
    const int outer_width = shared_q8->outer_task_width.load(std::memory_order_relaxed);
    if (outer_width > 1) {
        return false;
    }
    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, 0, 0);
    if (!qrow) {
        return false;
    }

    const int top_k = static_cast<int>(selected_experts->ne[0]);
    std::array<int, kQwen35NativeMoEMaxGroupedExperts> experts{};
    experts.fill(-1);
    for (int k = 0; k < top_k; ++k) {
        int32_t expert = -1;
        if (Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, 0, k, &expert)) {
            experts[static_cast<size_t>(k)] = expert;
        }
    }

    const int64_t row_count = row_end - row_start;
    const size_t weight_row_bytes = static_cast<size_t>(gate_exps->nb[1]);
    const ggml_tensor* prepacked = shared_q8->prepacked_fused_gate_up_exps;
    const bool use_prepacked = prepacked && prepacked->data && prepacked->type == GGML_TYPE_Q4_K && row_start == 0 &&
                               prepacked->ne[0] == gate_exps->ne[0] && prepacked->ne[1] == 2 * row_count &&
                               prepacked->ne[2] == gate_exps->ne[2] && (row_count % 8) == 0;
    std::atomic<bool> all_ok{true};
    const bool grouped = RunNativeMoEOnExpertNodeGroups(
        shared_q8, experts.data(), top_k, [&](int node, const NativeMoENodeGroupPlan& plan) {
            std::array<int, kQwen35NativeMoEMaxGroupedExperts> group_items{};
            int group_count = 0;
            for (int item = 0; item < plan.item_count; ++item) {
                if (plan.item_nodes[static_cast<size_t>(item)] == node) {
                    group_items[static_cast<size_t>(group_count++)] = item;
                }
            }
            if (group_count == 0) {
                return;
            }
            if (use_prepacked) {
                const int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount(node);
                const int worker_count = std::max(1, physical_cores > 0 ? physical_cores : group_count);
                const int splits_per_expert = std::max(1, (worker_count + group_count - 1) / group_count);
                const int total_units = group_count * splits_per_expert;
                const int tile_count = static_cast<int>(row_count / 8);
                shared_q8->numa_backend->ParallelFor(
                    total_units,
                    [&](int unit_start, int unit_end, int) {
                        for (int unit = unit_start; unit < unit_end && all_ok.load(std::memory_order_relaxed); ++unit) {
                            const int item = group_items[static_cast<size_t>(unit / splits_per_expert)];
                            const int split = unit % splits_per_expert;
                            const int tile_start = (tile_count * split) / splits_per_expert;
                            const int tile_end = (tile_count * (split + 1)) / splits_per_expert;
                            const int expert = experts[static_cast<size_t>(item)];
                            const void* fused_expert =
                                static_cast<const char*>(prepacked->data) +
                                static_cast<size_t>(expert) * static_cast<size_t>(prepacked->nb[2]);
                            float* out =
                                reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                         static_cast<size_t>(item) * static_cast<size_t>(dst->nb[1]));
                            if (!densecore::RunQ4KPrepackedMoEFusedSwiGLUTileRange(
                                    fused_expert, qrow, out, row_count, gate_exps->ne[0], tile_start, tile_end)) {
                                all_ok.store(false, std::memory_order_relaxed);
                            }
                        }
                    },
                    node);
                return;
            }
            const int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount(node);
            const int worker_count = std::max(1, physical_cores > 0 ? physical_cores : group_count);
            const int splits_per_expert = std::max(1, (worker_count + group_count - 1) / group_count);
            const int total_units = group_count * splits_per_expert;
            shared_q8->numa_backend->ParallelFor(
                total_units,
                [&](int unit_start, int unit_end, int) {
                    for (int unit = unit_start; unit < unit_end && all_ok.load(std::memory_order_relaxed); ++unit) {
                        const int item = group_items[static_cast<size_t>(unit / splits_per_expert)];
                        const int split = unit % splits_per_expert;
                        const int64_t split_start = (row_count * split) / splits_per_expert;
                        const int64_t split_end = (row_count * (split + 1)) / splits_per_expert;
                        if (split_end <= split_start) {
                            continue;
                        }
                        const int expert = experts[static_cast<size_t>(item)];
                        const char* gate_row = static_cast<const char*>(gate_exps->data) +
                                               static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]) +
                                               static_cast<size_t>(row_start + split_start) * weight_row_bytes;
                        const char* up_row = static_cast<const char*>(up_exps->data) +
                                             static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]) +
                                             static_cast<size_t>(row_start + split_start) * weight_row_bytes;
                        float* out = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) +
                            static_cast<size_t>(row_start + split_start) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(item) * static_cast<size_t>(dst->nb[1]));
                        if (!Qwen35NativeMoEKQ8KFusedSwiGLURows(nullptr, gate_exps->type, gate_row, up_row, qrow,
                                                                gate_exps->ne[0], split_end - split_start,
                                                                weight_row_bytes, out)) {
                            all_ok.store(false, std::memory_order_relaxed);
                        }
                    }
                },
                node);
        });
    if (!grouped || !all_ok.load(std::memory_order_relaxed)) {
        return false;
    }
    GetNativeMoENumaCounters().grouped_decode_used_ops.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<bool> logged_grouped_gateup{false};
    if (!logged_grouped_gateup.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[NUMA] Qwen native MoE grouped decode active (stage=gateup, experts=%d, kernel=%s)\n",
                     top_k, use_prepacked ? "q4k_8x8_prepacked" : "raw_q4k");
    }
    return true;
}

static void ProbeQwen35NativeMoEGateUpReference(const ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                const ggml_tensor* up_exps, const ggml_tensor* input,
                                                const ggml_tensor* selected_experts, int64_t row_start, int64_t row_end,
                                                const Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !input || !selected_experts || !dst->data || !gate_exps->data ||
        !up_exps->data || !input->data || !selected_experts->data || row_start >= row_end) {
        return;
    }
    const int layer_idx = shared_q8 ? shared_q8->debug_layer_idx : -1;
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t target_token_env =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    const int64_t token = target_token_env >= 0 ? target_token_env : n_tokens - 1;
    if (token < 0 || token >= n_tokens || !ShouldRunQwen35NativeMoEReferenceProbe(layer_idx, "gateup", token)) {
        return;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t cols = gate_exps->ne[0];
    const size_t row_bytes = static_cast<size_t>(gate_exps->nb[1]);
    thread_local std::vector<uint8_t> qbuf;
    const uint8_t* qrow = shared_q8 ? Qwen35SharedQ8RowPtr(shared_q8, token, 0) : nullptr;
    if (!qrow) {
        if (!Qwen35NativeMoEGateUpQuantizeInput(input, token, qbuf)) {
            return;
        }
        qrow = qbuf.data();
    }

    float max_abs_diff = 0.0f;
    int64_t max_topk = -1;
    int64_t max_row = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        int32_t expert = -1;
        if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, token, k, &expert)) {
            continue;
        }
        const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
        const char* up_base =
            static_cast<const char*>(up_exps->data) + static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* gate_row = gate_base + static_cast<size_t>(row) * row_bytes;
            const void* up_row = up_base + static_cast<size_t>(row) * static_cast<size_t>(up_exps->nb[1]);
            float gate_ref = 0.0f;
            float up_ref = 0.0f;
            if (!Qwen35NativeMoEReferenceDotQXK(gate_exps->type, gate_row, qrow, cols, &gate_ref) ||
                !Qwen35NativeMoEReferenceDotQXK(up_exps->type, up_row, qrow, cols, &up_ref)) {
                continue;
            }
            const float ref = NativeMoESiLU(gate_ref) * up_ref;
            const auto* actual_ptr = reinterpret_cast<const float*>(
                static_cast<const char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2]));
            const float actual = *actual_ptr;
            const float diff = std::fabs(actual - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_topk = k;
                max_row = row;
                max_actual = actual;
                max_ref = ref;
            }
        }
    }
    std::fprintf(stderr,
                 "[QWEN35_NATIVE_MOE_REF] layer=%d stage=gateup token=%lld rows=[%lld,%lld) max_abs_diff=%.8g "
                 "topk=%lld row=%lld actual=%.8g ref=%.8g tol=%.8g\n",
                 layer_idx, static_cast<long long>(token), static_cast<long long>(row_start),
                 static_cast<long long>(row_end), max_abs_diff, static_cast<long long>(max_topk),
                 static_cast<long long>(max_row), max_actual, max_ref, Qwen35NativeMoEReferenceTolerance());
}

static bool Qwen35GateUpQ5KRepackKernelAvailable() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

// C4A maintained path: enable the i8mm-optimized ggml q4_K_8x8 repacked MoE
// gate/up lane for small RAM-safe native-MoE models (LFM2). ARM-only; x86
// stays on the existing DenseCore-owned x86 admission.
static bool C4ALfm2Q4KMoERepackEnabled() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#else
    return false;
#endif
}

static bool Qwen35GateUpQ4KRepackKernelAvailable() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM: ggml's q4_K_8x8 gemv/gemm is i8mm/dotprod-optimized (llama's kernel
    // class) and Q4KRealPackedGemvKernelAvailable() is already true here. Qwen
    // prefill remains outside this LFM2 gate because it measured slower on the
    // repacked lane even with zero cache evictions.
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable() && C4ALfm2Q4KMoERepackEnabled();
#else
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable() && ggml_cpu_has_avx2();
#endif
}

static void RunQwen35NativeMoEGateUpRawQXKSwiGLU(ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                 const ggml_tensor* up_exps, const ggml_tensor* input,
                                                 const ggml_tensor* selected_experts, int ith, int nth,
                                                 Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !input || !selected_experts || !dst->data || nth <= 0) {
        return;
    }
    const int64_t n_ff = dst->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    const bool q4_gateup = gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K;
    const bool q5_gateup = gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K;
    const bool q5_single_copy_8x8 = q5_gateup && shared_q8 && shared_q8->q5k_gateup_8x8_single_copy;
    const bool q5_single_copy_required = q5_gateup && shared_q8 && shared_q8->q5k_gateup_8x8_single_copy_required;
    InferenceWorkContext* work_ctx = shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
    InferenceWorkContext* lfm2_w1w3_work_ctx = shared_q8 && shared_q8->record_lfm2_w1w3_kernel ? work_ctx : nullptr;
    if (dst->type != GGML_TYPE_F32 || (!q4_gateup && !q5_gateup) || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || gate_exps->ne[1] != n_ff || up_exps->ne[1] != n_ff ||
        dst->ne[1] != top_k || dst->ne[2] != n_tokens || gate_exps->ne[2] != up_exps->ne[2]) {
        return;
    }
    const int64_t block_size = ggml_blck_size(gate_exps->type);
    if (gate_exps->ne[0] != up_exps->ne[0] || block_size <= 0 || (gate_exps->ne[0] % block_size) != 0) {
        return;
    }

    const int64_t row_start = (static_cast<int64_t>(ith) * n_ff) / nth;
    const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_ff) / nth;
    const int64_t row_count = row_end - row_start;
    const bool compact_gateup_rows = gate_exps->nb[1] == ggml_row_size(gate_exps->type, gate_exps->ne[0]) &&
                                     up_exps->nb[1] == ggml_row_size(up_exps->type, up_exps->ne[0]);
    const bool use_shared_q8 = PrepareQwen35SharedQ8Rows(shared_q8, input, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    thread_local std::vector<Qwen35MoEAssignment> assignments;
    thread_local std::vector<uint8_t> gateup_qtile;
    thread_local std::vector<float> gateup_input_tile;
    thread_local std::vector<float> gateup_tile_out;
    if (ith == 0 && shared_q8) {
        shared_q8->repacked_swiglu_failed.store(0, std::memory_order_relaxed);
    }
    const bool lfm2_q4k_repacked_prefill = shared_q8 && shared_q8->record_lfm2_w1w3_kernel &&
                                           GetCurrentExecutionPhase() == InferenceExecutionPhase::Prefill &&
                                           C4ALfm2Q4KMoERepackEnabled();
    // Qwen Q5_K decode fast lane: run the loader-owned single-copy q5_K_8x8
    // layout through ggml's GEMV kernel. Do not build a runtime repack cache
    // here: keeping both raw Q5_K and q5_K_8x8 copies was the RAM wall for 35B
    // Q5. LFM2 Q5 does not use this Qwen-only layout and must fall through to
    // its validated raw vecdot lane instead of being rejected here.
    if (q5_gateup && (q5_single_copy_8x8 || q5_single_copy_required) && n_tokens == 1 && use_shared_q8 &&
        dst->nb[0] == static_cast<int64_t>(sizeof(float)) && (n_ff % 8) == 0) {
        if (!Qwen35GateUpQ5KRepackKernelAvailable()) {
            if (ith == 0) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false,
                                             "qwen35_gateup_q5k_repack_kernel_unavailable");
            }
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
            return;
        }
        if (!q5_single_copy_8x8) {
            if (ith == 0) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false,
                                             "qwen35_gateup_q5k_single_copy_missing");
            }
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
            return;
        }
        if (ith == 0) {
            RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr);
        }
        const int64_t hidden_dim = gate_exps->ne[0];
        const size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, hidden_dim);
        const int64_t tile_total = n_ff / 8;
        const int64_t tile_lo = (static_cast<int64_t>(ith) * tile_total) / nth;
        const int64_t tile_hi = (static_cast<int64_t>(ith + 1) * tile_total) / nth;
        const int64_t r0 = tile_lo * 8;
        const int64_t r1 = tile_hi * 8;
        if (r1 > r0) {
            const int nc = static_cast<int>(r1 - r0);
            const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, 0, 0);
            if (!qrow) {
                shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                return;
            }
            const size_t tile_byte_off = static_cast<size_t>(r0) * row_size;
            thread_local std::vector<float> q5k_gate_buf;
            thread_local std::vector<float> q5k_up_buf;
            if (static_cast<int>(q5k_gate_buf.size()) < nc) q5k_gate_buf.resize(static_cast<size_t>(nc));
            if (static_cast<int>(q5k_up_buf.size()) < nc) q5k_up_buf.resize(static_cast<size_t>(nc));
            for (int64_t k = 0; k < top_k; ++k) {
                int32_t expert = -1;
                if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, 0, k, &expert)) {
                    continue;
                }
                const uint8_t* gate_vx = static_cast<const uint8_t*>(gate_exps->data) +
                                         static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]) +
                                         tile_byte_off;
                const uint8_t* up_vx = static_cast<const uint8_t*>(up_exps->data) +
                                       static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]) +
                                       tile_byte_off;
                float* out_col = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                          static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]));
                // Bound on the calling thread: the body may run on a NUMA-pinned
                // helper thread with its own empty thread_local buffers (W2).
                float* q5k_gate_ptr = q5k_gate_buf.data();
                float* q5k_up_ptr = q5k_up_buf.data();
                RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) {
                    ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(hidden_dim), q5k_gate_ptr, 0, gate_vx, qrow, 1, nc);
                    ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(hidden_dim), q5k_up_ptr, 0, up_vx, qrow, 1, nc);
                    for (int i = 0; i < nc; ++i) {
                        out_col[r0 + i] =
                            NativeMoESiLU(q5k_gate_ptr[static_cast<size_t>(i)]) * q5k_up_ptr[static_cast<size_t>(i)];
                    }
                });
            }
        }
        return;
    }
    bool qwen36_decode_rowpair =
        q4_gateup && compact_gateup_rows && use_shared_q8 && shared_q8 && shared_q8->group_small_decode_gateup &&
        ResolveNativeMoEOuterTaskExecutionPhase(shared_q8) == InferenceExecutionPhase::Decode && n_tokens >= 2 &&
        n_tokens <= 4;
    size_t assignment_count = 0;
    const Qwen35MoEAssignment* assignment_data = nullptr;
    if ((q4_gateup || q5_gateup) && (n_tokens > 4 || qwen36_decode_rowpair) && use_shared_q8) {
        assignment_data =
            PrepareQwen35SharedMoEAssignments(shared_q8, selected_experts, gate_exps->ne[2], ith, &assignment_count);
        if (!assignment_data && Qwen35BuildMoEAssignments(selected_experts, gate_exps->ne[2], &assignments)) {
            assignment_data = assignments.data();
            assignment_count = assignments.size();
        }
    }
    size_t paired_assignments = 0;
    if (qwen36_decode_rowpair && assignment_data) {
        for (size_t begin = 0; begin < assignment_count;) {
            size_t end = begin + 1;
            while (end < assignment_count && assignment_data[end].expert == assignment_data[begin].expert) ++end;
            paired_assignments += ((end - begin) / 2) * 2;
            begin = end;
        }
        qwen36_decode_rowpair = paired_assignments > 0 && paired_assignments * 2 >= assignment_count;
    }
    if (ith == 0 && shared_q8 && shared_q8->debug_layer_idx == 0 && n_tokens <= 4 &&
        densecore::runtime::Qwen36GateUpRowPairProfiling()) {
        std::fprintf(stderr,
                     "[Q4PairAdmission] m=%lld flag=%d phase=%d compact=%d shared=%d paired=%zu total=%zu use=%d\n",
                     static_cast<long long>(n_tokens), shared_q8->group_small_decode_gateup ? 1 : 0,
                     static_cast<int>(ResolveNativeMoEOuterTaskExecutionPhase(shared_q8)), compact_gateup_rows ? 1 : 0,
                     use_shared_q8 ? 1 : 0, paired_assignments, assignment_count, qwen36_decode_rowpair ? 1 : 0);
    }
    if ((q4_gateup || q5_gateup) && (n_tokens > 4 || qwen36_decode_rowpair) && use_shared_q8 && assignment_data &&
        assignment_count > 0) {
        if (row_count <= 0) {
            return;
        }
        const size_t weight_row_bytes = q5_single_copy_8x8 ? ggml_row_size(GGML_TYPE_Q5_K, gate_exps->ne[0])
                                                           : static_cast<size_t>(gate_exps->nb[1]);
        for (size_t group_start = 0; group_start < assignment_count;) {
            const int32_t expert = assignment_data[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignment_count && assignment_data[group_end].expert == expert) {
                ++group_end;
            }
            const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                    static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
            const char* up_base = static_cast<const char*>(up_exps->data) +
                                  static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
            const void* gate_row_start = gate_base + static_cast<size_t>(row_start) * weight_row_bytes;
            const void* up_row_start = up_base + static_cast<size_t>(row_start) * weight_row_bytes;
            const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
            const bool q4_repacked_gateup =
                q4_gateup && shared_q8 && compact_gateup_rows && (row_count % 8) == 0 &&
                Qwen35GateUpQ4KRepackKernelAvailable() &&
                ((shared_q8->prefer_q4k_repacked_swiglu && phase == InferenceExecutionPhase::Decode) ||
                 lfm2_q4k_repacked_prefill);
            const bool repacked_gateup = q5_single_copy_8x8 || q4_repacked_gateup;
            const bool can_use_batched_gateup = (q4_gateup || q5_gateup) &&
                                                (compact_gateup_rows || q5_single_copy_8x8) && row_count > 0 &&
                                                group_end - group_start >= 2 && shared_q8 && shared_q8->row_bytes > 0 &&
                                                (!q5_single_copy_8x8 || ((row_start % 8) == 0 && (row_count % 8) == 0));
            if (can_use_batched_gateup) {
                densecore::CpuBackend& backend = densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch());
                const size_t qrow_bytes = shared_q8->row_bytes;
                gateup_qtile.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) * qrow_bytes);
                if (repacked_gateup) {
                    gateup_input_tile.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) *
                                             static_cast<size_t>(gate_exps->ne[0]));
                }
                gateup_tile_out.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) *
                                       static_cast<size_t>(row_count));
                for (size_t tile_start = group_start; tile_start < group_end;) {
                    size_t tile_count = 0;
                    while (tile_start < group_end &&
                           tile_count < static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile)) {
                        const Qwen35MoEAssignment& assignment = assignment_data[tile_start++];
                        const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.token, 0);
                        const float* input_row =
                            repacked_gateup ? Qwen35NativeMoEGateUpInputRowPtr(input, assignment.token) : nullptr;
                        if (!qrow || (repacked_gateup && !input_row)) {
                            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                            return;
                        }
                        std::memcpy(gateup_qtile.data() + tile_count * qrow_bytes, qrow, qrow_bytes);
                        if (repacked_gateup) {
                            std::memcpy(gateup_input_tile.data() + tile_count * static_cast<size_t>(gate_exps->ne[0]),
                                        input_row, static_cast<size_t>(gate_exps->ne[0]) * sizeof(float));
                        }
                        ++tile_count;
                    }
                    if (tile_count == 0) {
                        continue;
                    }
                    bool ok = false;
                    // ROOT CAUSE OF W2: these are thread_local buffers, and when sticky
                    // NUMA routing is armed RunNativeMoEOnExpertNode executes this body on
                    // a NUMA-pinned helper thread. There the thread_locals are that
                    // thread's own instances -- empty, so .data() was returning nullptr and
                    // every kernel below rejected on its null-argument guard. The reject
                    // left `dst` unwritten with no fallback, silently corrupting prefill.
                    // Bind the pointers here, on the thread that actually filled them.
                    const float* gateup_input_tile_ptr = gateup_input_tile.data();
                    const uint8_t* gateup_qtile_ptr = gateup_qtile.data();
                    float* gateup_tile_out_ptr = gateup_tile_out.data();
                    RunNativeMoEOnExpertNode(shared_q8, expert, [&](int numa_node) {
                        const bool allow_parallel = NativeMoEKernelMayFanOut(shared_q8, numa_node);
                        if (q5_single_copy_8x8) {
                            ok = densecore::RunQ5KRepackedMoEFusedSwiGLURawProjection(
                                densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                                gate_row_start, up_row_start, gateup_input_tile_ptr, gateup_qtile_ptr, qrow_bytes,
                                gateup_tile_out_ptr, static_cast<int64_t>(tile_count), row_count, gate_exps->ne[0],
                                numa_node, allow_parallel);
                        } else if (q4_repacked_gateup) {
                            RecordLFM2NativeMoEW1W3Kernel(lfm2_w1w3_work_ctx, "q4k_repacked");
                            ok = densecore::RunQ4KRepackedMoEFusedSwiGLUProjection(
                                densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                                gate_row_start, up_row_start, gateup_input_tile_ptr, gateup_qtile_ptr, qrow_bytes,
                                gateup_tile_out_ptr, static_cast<int64_t>(tile_count), row_count, gate_exps->ne[0],
                                numa_node, allow_parallel);
                        } else if (qwen36_decode_rowpair) {
                            if (tile_count >= 2) {
                                ok = densecore::runtime::Qwen36GateUpRowPair(
                                    gate_row_start, up_row_start, gateup_qtile_ptr, gate_exps->ne[0], row_count,
                                    static_cast<int>(tile_count), weight_row_bytes, qrow_bytes, gateup_tile_out_ptr);
                            } else {
                                ok = densecore::hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(
                                    gate_row_start, up_row_start, gateup_qtile_ptr, gate_exps->ne[0], row_count,
                                    weight_row_bytes, gateup_tile_out_ptr);
                            }
                        } else {
                            ok = densecore::RunMoEKQuantRawBatchedFusedSwiGLU(
                                densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()), &backend,
                                static_cast<int>(gate_exps->type), gate_row_start, up_row_start, gateup_qtile_ptr,
                                qrow_bytes, gateup_tile_out_ptr, static_cast<int64_t>(tile_count), row_count,
                                gate_exps->ne[0], numa_node, allow_parallel);
                        }
                    });
                    if (!ok) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                        return;
                    }
#ifdef DENSECORE_TEST_BUILD
                    if (qwen36_decode_rowpair && tile_count >= 2)
                        shared_q8->grouped_gateup_tiles.fetch_add(1, std::memory_order_relaxed);
#endif
                    const size_t emitted_start = tile_start - tile_count;
                    for (size_t tm = 0; tm < tile_count; ++tm) {
                        const Qwen35MoEAssignment& assignment = assignment_data[emitted_start + tm];
                        float* out = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) +
                            static_cast<size_t>(row_start) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(assignment.topk_index) * static_cast<size_t>(dst->nb[1]) +
                            static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[2]));
                        const float* src = gateup_tile_out.data() + tm * static_cast<size_t>(row_count);
                        if (dst->nb[0] == static_cast<int64_t>(sizeof(float))) {
                            std::memcpy(out, src, static_cast<size_t>(row_count) * sizeof(float));
                        } else {
                            for (int64_t row = 0; row < row_count; ++row) {
                                *reinterpret_cast<float*>(reinterpret_cast<char*>(out) +
                                                          static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) =
                                    src[row];
                            }
                        }
                    }
                }
                group_start = group_end;
                continue;
            }
            for (size_t ai = group_start; ai < group_end; ++ai) {
                const Qwen35MoEAssignment& assignment = assignment_data[ai];
                const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.token, 0);
                if (!qrow) {
                    if (shared_q8) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                    }
                    return;
                }
                float* out = reinterpret_cast<float*>(
                    static_cast<char*>(dst->data) + static_cast<size_t>(row_start) * static_cast<size_t>(dst->nb[0]) +
                    static_cast<size_t>(assignment.topk_index) * static_cast<size_t>(dst->nb[1]) +
                    static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[2]));
                const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
                const bool q4_repacked_gateup =
                    q4_gateup && shared_q8 && compact_gateup_rows && (row_count % 8) == 0 &&
                    Qwen35GateUpQ4KRepackKernelAvailable() &&
                    ((shared_q8->prefer_q4k_repacked_swiglu && phase == InferenceExecutionPhase::Decode) ||
                     lfm2_q4k_repacked_prefill);
                if (q5_single_copy_8x8 || q4_repacked_gateup) {
                    const float* input_row =
                        q5_single_copy_8x8 ? Qwen35NativeMoEGateUpInputRowPtr(input, assignment.token) : nullptr;
                    const bool q5_row_aligned = !q5_single_copy_8x8 || ((row_start % 8) == 0 && (row_count % 8) == 0);
                    bool ok = false;
                    RunNativeMoEOnExpertNode(shared_q8, expert, [&](int numa_node) {
                        const bool allow_parallel = NativeMoEKernelMayFanOut(shared_q8, numa_node);
                        ok = q5_single_copy_8x8
                                 ? (input_row && q5_row_aligned &&
                                    densecore::RunQ5KRepackedMoEFusedSwiGLURawProjection(
                                        densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()),
                                        &densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch()), gate_row_start,
                                        up_row_start, input_row, qrow, shared_q8->row_bytes, out, /*rows=*/1, row_count,
                                        gate_exps->ne[0], numa_node, allow_parallel))
                                 : densecore::RunQ4KRepackedMoEFusedSwiGLUProjection(
                                       densecore::llm::runtime::ResolveCpuExecutionOptions(GetCurrentWorkContext()),
                                       &densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch()), gate_row_start,
                                       up_row_start, nullptr, qrow, shared_q8->row_bytes, out, /*rows=*/1, row_count,
                                       gate_exps->ne[0], numa_node, allow_parallel);
                    });
                    if (ok && !q5_single_copy_8x8) {
                        RecordLFM2NativeMoEW1W3Kernel(lfm2_w1w3_work_ctx, "q4k_repacked");
                    }
                    if (!ok) {
                        if (shared_q8) {
                            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                        }
                        return;
                    }
                    continue;
                }
                bool ok = false;
                RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) {
                    ok = Qwen35NativeMoEKQ8KFusedSwiGLURows(lfm2_w1w3_work_ctx, gate_exps->type, gate_row_start,
                                                            up_row_start, qrow, gate_exps->ne[0], row_count,
                                                            weight_row_bytes, out);
                });
                if (!ok) {
                    if (shared_q8) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                    }
                    return;
                }
            }
            group_start = group_end;
        }
        if (!q5_single_copy_8x8) {
            ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, row_start, row_end,
                                                shared_q8);
        }
        return;
    }
    if (q5_single_copy_8x8) {
        if (shared_q8) {
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
        }
        return;
    }
    if (TryRunQwen35NativeMoEGateUpDirectOuterTasks(dst, gate_exps, up_exps, selected_experts, ith, nth, shared_q8)) {
        if (ith == 0 && shared_q8->repacked_swiglu_failed.load(std::memory_order_relaxed) == 0) {
            ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, 0, n_ff, shared_q8);
        }
        return;
    }
    if (TryRunQwen35NativeMoEGateUpGroupedDecode(dst, gate_exps, up_exps, selected_experts, row_start, row_end,
                                                 shared_q8)) {
        if (shared_q8->repacked_swiglu_failed.load(std::memory_order_relaxed) == 0) {
            ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, row_start, row_end,
                                                shared_q8);
        }
        return;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        const uint8_t* qrow = use_shared_q8 ? Qwen35SharedQ8RowPtr(shared_q8, token, 0) : nullptr;
        if (!qrow) {
            if (!Qwen35NativeMoEGateUpQuantizeInput(input, token, qbuf)) continue;
            qrow = qbuf.data();
        }
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, token, k, &expert)) continue;
            const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                    static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
            const char* up_base = static_cast<const char*>(up_exps->data) +
                                  static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
            if (row_count <= 0) {
                continue;
            }
            const char* gate_row_start =
                gate_base + static_cast<size_t>(row_start) * static_cast<size_t>(gate_exps->nb[1]);
            const char* up_row_start = up_base + static_cast<size_t>(row_start) * static_cast<size_t>(up_exps->nb[1]);
            float* out_start = reinterpret_cast<float*>(
                static_cast<char*>(dst->data) + static_cast<size_t>(row_start) * static_cast<size_t>(dst->nb[0]) +
                static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2]));
            bool expert_ok = false;
            RunNativeMoEOnExpertNode(shared_q8, expert, [&](int) {
                expert_ok = Qwen35NativeMoEKQ8KFusedSwiGLURows(lfm2_w1w3_work_ctx, gate_exps->type, gate_row_start,
                                                               up_row_start, qrow, gate_exps->ne[0], row_count,
                                                               static_cast<size_t>(gate_exps->nb[1]), out_start);
            });
            if (!expert_ok) {
                if (shared_q8) {
                    shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                }
                return;
            }
        }
    }
    ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, row_start, row_end,
                                        shared_q8);
}

static void cb_qwen35_native_moe_gateup_raw_qxk_swiglu(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    ScopedInferenceWorkContext callback_context(shared_q8 ? shared_q8->work_ctx : nullptr);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    RunQwen35NativeMoEGateUpRawQXKSwiGLU(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                         dst ? dst->src[2] : nullptr, dst ? dst->src[3] : nullptr, effective_ith,
                                         effective_nth, shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[3] : nullptr;
        if (shared_q8 && shared_q8->numa_backend && shared_q8->numa_layer && selected && selected->data) {
            std::vector<int> expert_ids;
            expert_ids.reserve(static_cast<size_t>(std::max<int64_t>(0, selected->ne[0] * selected->ne[1])));
            for (int64_t token = 0; token < selected->ne[1]; ++token) {
                for (int64_t k = 0; k < selected->ne[0]; ++k) {
                    int32_t expert = -1;
                    if (Qwen35NativeMoEDownQ5KReadExpert(selected, dst->src[0], token, k, &expert)) {
                        expert_ids.push_back(expert);
                    }
                }
            }
            if (!expert_ids.empty()) {
                shared_q8->numa_backend->RecordExpertAccess(shared_q8->numa_layer, expert_ids.data(),
                                                            static_cast<int>(expert_ids.size()));
            }
        }
        const int selected_experts = dst ? static_cast<int>(std::max<int64_t>(0, dst->ne[1])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        const bool swiglu_failed = shared_q8 && shared_q8->repacked_swiglu_failed.load(std::memory_order_relaxed) != 0;
        const char* reject_reason = nullptr;
        if (swiglu_failed) {
            reject_reason = shared_q8->q5k_gateup_8x8_single_copy_required ? "qwen35_gateup_q5k_single_copy_failed"
                            : shared_q8->record_lfm2_w1w3_kernel
                                ? (shared_q8->prefer_q4k_repacked_swiglu ? "lfm2_w1w3_repacked_swiglu_failed"
                                                                         : "lfm2_w1w3_range_swiglu_failed")
                                : "qwen35_gateup_q4k_swiglu_failed";
        }
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/!swiglu_failed, reject_reason,
                                          /*w1w3_used=*/!swiglu_failed, /*w2_used=*/false, swiglu_failed ? 0 : wall_ns);
    }
}

static bool CanUseQwen35NativeMoEGateUpRawQXKSwiGLU(const TransformerModel* model, const ggml_tensor* gate_exps,
                                                    const ggml_tensor* up_exps, const ggml_tensor* input,
                                                    const ggml_tensor* selected_experts) {
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W1W3_QXK_DIAG] gateup_raw rejected: %s\n", reason);
        }
        return false;
    };
    if (!model || !gate_exps || !up_exps || !input || !selected_experts) return reject("missing_arg");
    const bool qwen_native_moe = (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                                 model->arch_flags.is_hybrid_ssm;
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (!qwen_native_moe && !lfm2_native_moe) {
        return reject("unsupported_model");
    }
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return reject("unsupported_phase");
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool fast_moe_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, phase, fast_config.native_moe_fast_decode);
    if (!fast_moe_enabled) {
        return reject("native_moe_fast_decode_disabled");
    }
    if (current_batch && !current_batch->lora_map.empty()) return reject("dynamic_lora");
    const bool qwen_supported_gateup =
        qwen_native_moe && ((gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K) ||
                            (gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K));
    const bool lfm2_supported_gateup =
        lfm2_native_moe && ((gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K) ||
                            (gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K));
    if ((!qwen_supported_gateup && !lfm2_supported_gateup) || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32) {
        return reject("unsupported_type");
    }
    if (!gate_exps->data || !up_exps->data) return reject("missing_weight_data");
    if (gate_exps->view_src || up_exps->view_src) return reject("weight_view");
    if (input->ne[1] <= 0 || input->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (selected_experts->ne[1] != input->ne[1]) return reject("selected_expert_token_mismatch");
    if (gate_exps->ne[0] != input->ne[0] || up_exps->ne[0] != input->ne[0] || gate_exps->ne[1] != up_exps->ne[1] ||
        gate_exps->ne[2] != up_exps->ne[2]) {
        return reject("shape_mismatch");
    }
    const int64_t block_size = ggml_blck_size(gate_exps->type);
    if (block_size <= 0 || (gate_exps->ne[0] % block_size) != 0) return reject("bad_kquant_alignment");
    if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        std::fprintf(stderr, "[W1W3_QXK_DIAG] gateup_raw accepted type=%d\n", static_cast<int>(gate_exps->type));
    }
    return true;
}

struct QwenLikeNativeMoEGraphPlan {
    bool qwen_native_moe = false;
    bool lfm2_native_moe = false;
    bool target_requires_native_moe = false;
    InferenceExecutionPhase graph_phase = InferenceExecutionPhase::Unknown;
    bool lfm2_debug_reference = false;
    bool lfm2_fast_only_decode = false;
};

struct QwenLikeNativeMoEWeights {
    ggml_tensor* gate_exps = nullptr;
    ggml_tensor* up_exps = nullptr;
    ggml_tensor* down_exps = nullptr;
    ggml_tensor* gate_up_exps = nullptr;
    ggml_tensor* raw_gate_exps = nullptr;
    ggml_tensor* raw_up_exps = nullptr;
    ggml_tensor* raw_down_exps = nullptr;
    bool use_fused_gate_up = false;
    ggml_type w1w3_type = GGML_TYPE_COUNT;
    bool native_q5_gateup_single_copy = false;
};

static QwenLikeNativeMoEGraphPlan ResolveQwenLikeNativeMoEGraphPlan(const TransformerModel* model) {
    QwenLikeNativeMoEGraphPlan plan;
    plan.qwen_native_moe = model &&
                           (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                           model->arch_flags.is_hybrid_ssm;
    plan.lfm2_native_moe = model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    plan.target_requires_native_moe = ModelRequiresNativeMoEFastPath(model);
    plan.graph_phase = GetCurrentExecutionPhase();
    plan.lfm2_debug_reference = plan.lfm2_native_moe && IsDebugLFM2NativeMoEReferenceEnabled();
    plan.lfm2_fast_only_decode =
        plan.lfm2_native_moe && plan.graph_phase == InferenceExecutionPhase::Decode && !plan.lfm2_debug_reference;
    return plan;
}

static bool ShouldGroupQwenLikeSmallDecodeGateUp(const QwenLikeNativeMoEGraphPlan& plan, ModelVariant variant) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    return variant == ModelVariant::QWEN36 && plan.qwen_native_moe &&
           plan.graph_phase == InferenceExecutionPhase::Decode;
#else
    (void)plan;
    (void)variant;
    return false;
#endif
}

static bool ShouldUseQwenLikeGateUpQ4KRepackedSwiGLU(const QwenLikeNativeMoEGraphPlan& graph_plan, ggml_type w1w3_type,
                                                     InferenceExecutionPhase phase, bool kernel_available) {
    if (w1w3_type != GGML_TYPE_Q4_K || phase != InferenceExecutionPhase::Decode || !kernel_available) {
        return false;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM (C4A): the raw decode lane is dotprod per-row; the repacked
    // q4_K_8x8 gemv emits 8 output rows per i8mm call and amortizes the Q8_K
    // input read, so the x86 "repack regresses decode" verdict does not carry
    // over. Enable it for small RAM-safe native MoE (LFM2) only; keep the 35B
    // Qwen native MoE on the raw path (its gate/up thrashes the repack cache).
    // kernel_available only proves the kernel + some opt-in is active; re-check
    // the LFM2 opt-in here so a Qwen-prefill-only flag cannot leak into LFM2
    // decode.
    return graph_plan.lfm2_native_moe && C4ALfm2Q4KMoERepackEnabled();
#else
    // x86: C4 validation showed the single-token LFM2 Q4_K repacked SwiGLU path
    // regresses decode throughput versus the raw k-quant row path, so scope to
    // Qwen native MoE only.
    return graph_plan.qwen_native_moe;
#endif
}

static const char* ValidateQwenLikeNativeMoERouter(const QwenLikeNativeMoEGraphPlan& plan,
                                                   const densecore::models::DecoderLayerSpec* layer_spec) {
    if (layer_spec && plan.qwen_native_moe &&
        layer_spec->ffn.router != densecore::models::DecoderMoERouter::SoftmaxTopK) {
        return "router_not_softmax_topk";
    }
    if (layer_spec && plan.lfm2_native_moe &&
        layer_spec->ffn.router != densecore::models::DecoderMoERouter::GroupedSigmoidTopK) {
        return "router_not_grouped_sigmoid_topk";
    }
    return nullptr;
}

static bool ResolveQwenLikeNativeMoEWeights(TransformerModel* model, TransformerLayer* layer,
                                            const QwenLikeNativeMoEGraphPlan& plan, int64_t n_tokens,
                                            QwenLikeNativeMoEWeights* weights, const char** reject_reason) {
    if (!model || !layer || !weights) {
        if (reject_reason) *reject_reason = "missing_graph_inputs";
        return false;
    }
    weights->gate_exps = GetLayerTensorAny(layer, {"ffn_gate_exps.weight", "ffn_gate_exps"});
    weights->up_exps = GetLayerTensorAny(layer, {"ffn_up_exps.weight", "ffn_up_exps"});
    weights->down_exps = GetLayerTensorAny(layer, {"ffn_down_exps.weight", "ffn_down_exps"});
    weights->gate_up_exps = GetLayerTensorAny(layer, {"ffn_gate_up_exps.cpu_repack_fused"});
    if (!weights->gate_exps || !weights->up_exps || !weights->down_exps) {
        if (reject_reason) *reject_reason = "missing_moe_weight_tensor";
        return false;
    }

    weights->raw_gate_exps = weights->gate_exps;
    weights->raw_up_exps = weights->up_exps;
    weights->raw_down_exps = weights->down_exps;
    if (!plan.lfm2_native_moe) {
        weights->gate_exps = UseCpuRepackAliasForTokenCount(model, weights->gate_exps, n_tokens);
        weights->up_exps = UseCpuRepackAliasForTokenCount(model, weights->up_exps, n_tokens);
        weights->down_exps = UseCpuRepackAliasForTokenCount(model, weights->down_exps, n_tokens);
        if (weights->gate_up_exps) {
            weights->gate_up_exps = UseCpuRepackAliasForTokenCount(model, weights->gate_up_exps, n_tokens);
        }
    }
    weights->use_fused_gate_up = weights->gate_up_exps && weights->gate_up_exps->type == weights->gate_exps->type &&
                                 weights->gate_up_exps->ne[0] == weights->gate_exps->ne[0] &&
                                 weights->gate_up_exps->ne[1] == weights->gate_exps->ne[1] + weights->up_exps->ne[1] &&
                                 weights->gate_up_exps->ne[2] == weights->gate_exps->ne[2];
    weights->w1w3_type = weights->use_fused_gate_up ? weights->gate_up_exps->type : weights->gate_exps->type;
    weights->native_q5_gateup_single_copy =
        !weights->use_fused_gate_up && weights->raw_gate_exps && weights->raw_up_exps &&
        weights->raw_gate_exps->type == GGML_TYPE_Q5_K && weights->raw_up_exps->type == GGML_TYPE_Q5_K &&
        model->prepared_weights.q5k_8x8_repacked_tensors.find(weights->raw_gate_exps) !=
            model->prepared_weights.q5k_8x8_repacked_tensors.end() &&
        model->prepared_weights.q5k_8x8_repacked_tensors.find(weights->raw_up_exps) !=
            model->prepared_weights.q5k_8x8_repacked_tensors.end();
    if ((plan.qwen_native_moe || plan.lfm2_native_moe) && weights->w1w3_type == GGML_TYPE_Q5_K &&
        !weights->native_q5_gateup_single_copy) {
        if (reject_reason) {
            *reject_reason =
                plan.qwen_native_moe ? "qwen35_gateup_q5k_single_copy_missing" : "lfm2_gateup_q5k_single_copy_missing";
        }
        return false;
    }
    return true;
}

static bool ValidateQwenLikeNativeMoEShapes(const QwenLikeNativeMoEWeights& weights, ggml_tensor* routed_input,
                                            ggml_tensor* gate_logits, int64_t n_tokens, int64_t n_embd,
                                            int64_t n_experts) {
    return n_tokens > 0 && n_embd > 0 && n_experts > 0 && routed_input && gate_logits &&
           gate_logits->ne[1] == n_tokens && weights.gate_exps && weights.up_exps && weights.down_exps &&
           weights.gate_exps->ne[0] == n_embd && weights.up_exps->ne[0] == n_embd &&
           weights.gate_exps->ne[2] == n_experts && weights.up_exps->ne[2] == n_experts &&
           weights.down_exps->ne[2] == n_experts && weights.gate_exps->ne[1] == weights.up_exps->ne[1] &&
           weights.down_exps->ne[0] == weights.gate_exps->ne[1] && weights.down_exps->ne[1] == n_embd;
}

static const char* RecordQwenLikeNativeMoEFastDecodeAdmission(const TransformerModel* model,
                                                              const QwenLikeNativeMoEGraphPlan& plan,
                                                              const QwenLikeNativeMoEWeights& weights, int64_t n_tokens,
                                                              int64_t n_experts, int64_t n_expert_used) {
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        return nullptr;
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool dynamic_lora_active = current_batch && !current_batch->lora_map.empty();
    const bool supported_lfm2_w2_quant =
        weights.down_exps->type == GGML_TYPE_Q4_K || weights.down_exps->type == GGML_TYPE_Q5_K ||
        weights.down_exps->type == GGML_TYPE_Q6_K || weights.down_exps->type == GGML_TYPE_Q8_0;
    const bool supported_qwen_w2_quant =
        weights.down_exps->type == GGML_TYPE_Q4_K || weights.down_exps->type == GGML_TYPE_Q5_K ||
        weights.down_exps->type == GGML_TYPE_Q6_K || weights.down_exps->type == GGML_TYPE_Q8_0;
    const bool supported_w1w3_quant =
        plan.lfm2_native_moe ? (weights.w1w3_type == GGML_TYPE_Q4_K || weights.w1w3_type == GGML_TYPE_Q5_K)
                             : (weights.w1w3_type == GGML_TYPE_Q4_K || weights.w1w3_type == GGML_TYPE_Q5_K);
    const bool supported_w2_quant = plan.lfm2_native_moe ? supported_lfm2_w2_quant : supported_qwen_w2_quant;
    const bool supported_quant = supported_w1w3_quant && supported_w2_quant;
    const bool supported_shape =
        (plan.graph_phase == InferenceExecutionPhase::Decode || plan.graph_phase == InferenceExecutionPhase::Prefill) &&
        n_tokens > 0 && n_tokens <= NativeMoEFastPathMaxDirectTokens(model);
    const bool selected_experts_available = n_expert_used > 0 && n_expert_used <= n_experts;
    const bool fast_kernel_available = supported_quant;
    const bool fast_decode_candidate = plan.graph_phase == InferenceExecutionPhase::Decode && n_experts > 0 &&
                                       (plan.qwen_native_moe || plan.lfm2_native_moe);
    const bool fast_decode_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, plan.graph_phase, fast_config.native_moe_fast_decode);
    const bool fast_decode_supported = fast_decode_candidate && fast_decode_enabled && supported_shape &&
                                       selected_experts_available && supported_quant && !dynamic_lora_active &&
                                       fast_kernel_available;
    if (!fast_decode_candidate || fast_decode_supported) {
        return nullptr;
    }

    const char* reason = "none";
    if (!fast_decode_enabled) {
        reason = "native_moe_fast_decode_disabled";
    } else if (!supported_shape) {
        reason = plan.lfm2_native_moe ? "callback_shape_mismatch" : "unsupported_shape";
    } else if (!selected_experts_available) {
        reason = "missing_selected_experts";
    } else if (dynamic_lora_active) {
        reason = "dynamic_lora";
    } else if (!supported_w1w3_quant) {
        reason = "unsupported_w1w3_quant";
    } else if (!supported_w2_quant) {
        reason = "unsupported_w2_quant";
    } else if (!fast_kernel_available) {
        reason = "no_fast_kernel";
    }
    RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, reason,
                                      /*w1w3_used=*/false, /*w2_used=*/false);
    if (supported_w2_quant) {
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/false, reason);
    }
    return plan.lfm2_fast_only_decode ? reason : nullptr;
}

static void RecordQwenLikeNativeMoEGraphCensus(const QwenLikeNativeMoEGraphPlan& plan,
                                               const QwenLikeNativeMoEWeights& weights, ggml_tensor* routed_input,
                                               int64_t n_tokens, int64_t n_embd, int64_t n_expert_used,
                                               int native_moe_callback_tasks) {
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        return;
    }
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.w1w3_type, n_tokens,
                                 weights.use_fused_gate_up ? weights.gate_up_exps->ne[1] : weights.gate_exps->ne[1],
                                 n_embd,
                                 weights.use_fused_gate_up ? weights.gate_up_exps->name : weights.gate_exps->name,
                                 routed_input->name, n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    if (!weights.use_fused_gate_up) {
        RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.up_exps->type, n_tokens,
                                     weights.up_exps->ne[1], n_embd, weights.up_exps->name, routed_input->name,
                                     n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    }
    RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.down_exps->type, n_tokens, n_embd,
                                 weights.down_exps->ne[0], weights.down_exps->name, "qwen35_native_moe_swiglu",
                                 n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    if (plan.qwen_native_moe || plan.lfm2_native_moe) {
        RecordQwen35MoEGraphPath(work_ctx, "native_graph", static_cast<int>(n_expert_used),
                                 static_cast<int>(n_expert_used), native_moe_callback_tasks, weights.w1w3_type,
                                 weights.down_exps->type);
    }
}

ggml_tensor* TryBuildQwen35NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k,
                                          const densecore::models::DecoderLayerSpec* layer_spec,
                                          densecore::CpuBackend* numa_backend) {
    const QwenLikeNativeMoEGraphPlan graph_plan = ResolveQwenLikeNativeMoEGraphPlan(model);
    if (!ctx || !gf || !model || !layer || !routed_input || !gate_logits ||
        (!graph_plan.qwen_native_moe && !graph_plan.lfm2_native_moe)) {
        return nullptr;
    }
    auto reject_native_moe = [&](const char* reason) -> ggml_tensor* {
        const char* safe_reason = reason && reason[0] != '\0' ? reason : "unknown";
        if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
            RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, safe_reason,
                                              /*w1w3_used=*/false, /*w2_used=*/false);
            RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/false, safe_reason);
        }
        if (graph_plan.target_requires_native_moe || IsQwen35NativeMoEDownQ5KDiagEnabled() ||
            IsMoEWiringDebugEnabled()) {
            const int64_t input0 = routed_input ? routed_input->ne[0] : -1;
            const int64_t input1 = routed_input ? routed_input->ne[1] : -1;
            const int64_t logits0 = gate_logits ? gate_logits->ne[0] : -1;
            const int64_t logits1 = gate_logits ? gate_logits->ne[1] : -1;
            std::fprintf(stderr,
                         "[NativeMoEAdmission] rejected reason=%s layer=%d variant=%d phase=%d input=[%lld,%lld] "
                         "logits=[%lld,%lld] top_k=%d qwen=%d lfm2=%d\n",
                         safe_reason, layer_idx, static_cast<int>(model->variant),
                         static_cast<int>(graph_plan.graph_phase), static_cast<long long>(input0),
                         static_cast<long long>(input1), static_cast<long long>(logits0),
                         static_cast<long long>(logits1), top_k, graph_plan.qwen_native_moe ? 1 : 0,
                         graph_plan.lfm2_native_moe ? 1 : 0);
        }
        return nullptr;
    };
    if (const char* router_reject = ValidateQwenLikeNativeMoERouter(graph_plan, layer_spec)) {
        return reject_native_moe(router_reject);
    }

    const int64_t n_tokens = routed_input->ne[1];
    QwenLikeNativeMoEWeights moe_weights;
    const char* weight_reject = nullptr;
    if (!ResolveQwenLikeNativeMoEWeights(model, layer, graph_plan, n_tokens, &moe_weights, &weight_reject)) {
        const bool q5_single_copy_missing =
            weight_reject && (std::strcmp(weight_reject, "qwen35_gateup_q5k_single_copy_missing") == 0 ||
                              std::strcmp(weight_reject, "lfm2_gateup_q5k_single_copy_missing") == 0);
        if (q5_single_copy_missing) {
            if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false, weight_reject);
                RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, weight_reject,
                                                  /*w1w3_used=*/false, /*w2_used=*/false);
            }
        }
        return reject_native_moe(weight_reject);
    }
    ggml_tensor* gate_exps = moe_weights.gate_exps;
    ggml_tensor* up_exps = moe_weights.up_exps;
    ggml_tensor* down_exps = moe_weights.down_exps;
    ggml_tensor* raw_gate_exps = moe_weights.raw_gate_exps;
    ggml_tensor* raw_up_exps = moe_weights.raw_up_exps;
    ggml_tensor* raw_down_exps = moe_weights.raw_down_exps;

    const int64_t n_embd = routed_input->ne[0];
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t n_expert_used = std::max<int64_t>(1, std::min<int64_t>(top_k, n_experts));
    const NativeMoENumaPlacement build_placement = EvaluateNativeMoENumaPlacement(numa_backend, layer, model->variant);
    const bool build_sticky_armed = NativeMoENumaStickyArmed(build_placement.state);
    const bool direct_outer_candidate =
        build_sticky_armed && graph_plan.qwen_native_moe && graph_plan.graph_phase == InferenceExecutionPhase::Decode &&
        n_tokens == 1 && raw_gate_exps && raw_up_exps && raw_down_exps && raw_gate_exps->type == GGML_TYPE_Q4_K &&
        raw_up_exps->type == GGML_TYPE_Q4_K &&
        (raw_down_exps->type == GGML_TYPE_Q4_K || raw_down_exps->type == GGML_TYPE_Q5_K ||
         raw_down_exps->type == GGML_TYPE_Q6_K) &&
        moe_weights.use_fused_gate_up && moe_weights.gate_up_exps && moe_weights.gate_up_exps->data &&
        moe_weights.gate_up_exps->type == GGML_TYPE_Q4_K && ggml_cpu_has_avx2() &&
        std::strstr(moe_weights.gate_up_exps->name, "cpu_repack_fused");
    const int native_moe_callback_tasks =
        ShouldNarrowNativeMoECallbackTasksForStickyFallback(build_sticky_armed, graph_plan.graph_phase,
                                                            direct_outer_candidate)
            ? 1
            : (direct_outer_candidate
                   ? ResolveTaskCount(GetCurrentBatch(), 0)
                   : ResolveNativeMoEGraphCallbackTaskCount(model, GetCurrentBatch(), graph_plan.graph_phase, n_tokens,
                                                            static_cast<int>(n_expert_used)));
    // One-shot. Sticky routing arming at RUN time while the graph was built with
    // the wide task count is the combination that puts a per-expert dispatch
    // inside a wide parallel region, so the two facts have to be readable
    // together -- the run-time sticky state alone does not say which task count
    // the graph was wired with.
    {
        static std::atomic<bool> logged_build_tasks{false};
        if (!logged_build_tasks.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                         "[NUMA] Native MoE graph built with callback_tasks=%d (build-time sticky state=%s, "
                         "experts=%d, nodes=%d)\n",
                         native_moe_callback_tasks,
                         GetNativeMoENumaStickyStateName(static_cast<int>(build_placement.state)),
                         build_placement.expert_count, build_placement.node_count);
        }
    }
    if (!ValidateQwenLikeNativeMoEShapes(moe_weights, routed_input, gate_logits, n_tokens, n_embd, n_experts)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr,
                         "[Qwen35NativeMoE] rejected layer=%d input=[%lld,%lld] logits=[%lld,%lld] "
                         "gate=[%lld,%lld,%lld,%lld] up=[%lld,%lld,%lld,%lld] down=[%lld,%lld,%lld,%lld]\n",
                         layer_idx, static_cast<long long>(routed_input->ne[0]),
                         static_cast<long long>(routed_input->ne[1]), static_cast<long long>(gate_logits->ne[0]),
                         static_cast<long long>(gate_logits->ne[1]), static_cast<long long>(gate_exps->ne[0]),
                         static_cast<long long>(gate_exps->ne[1]), static_cast<long long>(gate_exps->ne[2]),
                         static_cast<long long>(gate_exps->ne[3]), static_cast<long long>(up_exps->ne[0]),
                         static_cast<long long>(up_exps->ne[1]), static_cast<long long>(up_exps->ne[2]),
                         static_cast<long long>(up_exps->ne[3]), static_cast<long long>(down_exps->ne[0]),
                         static_cast<long long>(down_exps->ne[1]), static_cast<long long>(down_exps->ne[2]),
                         static_cast<long long>(down_exps->ne[3]));
        }
        return reject_native_moe("shape_mismatch");
    }
    const ggml_type w1w3_type = moe_weights.w1w3_type;
    const bool native_q5_gateup_single_copy = moe_weights.native_q5_gateup_single_copy;
    if (const char* fast_decode_reject = RecordQwenLikeNativeMoEFastDecodeAdmission(
            model, graph_plan, moe_weights, n_tokens, n_experts, n_expert_used)) {
        return reject_native_moe(fast_decode_reject);
    }
    RecordQwenLikeNativeMoEGraphCensus(graph_plan, moe_weights, routed_input, n_tokens, n_embd, n_expert_used,
                                       native_moe_callback_tasks);

    ggml_tensor* routing_probs = gate_logits;
    ggml_tensor* selection_scores = gate_logits;
    if (graph_plan.lfm2_native_moe) {
        routing_probs = ggml_sigmoid(ctx, gate_logits);
        ggml_set_name(routing_probs, "lfm2_native_moe_sigmoid_probs");
        selection_scores = routing_probs;
        if (ggml_tensor* bias = layer->Get(model_keys::kMoeCorrectionBias);
            bias && bias->type == GGML_TYPE_F32 && bias->ne[0] == n_experts) {
            selection_scores = ggml_add(ctx, routing_probs, bias);
            ggml_set_name(selection_scores, "lfm2_native_moe_biased_selection_scores");
        }
    }

    QwenNativeMoEFusedRouterState* fused_router_state = nullptr;
    ggml_tensor* selected_experts = nullptr;
    if (graph_plan.qwen_native_moe && n_tokens == 1 && n_expert_used == kQwenNativeMoEFusedRouterTopK &&
        model->moe_norm_topk_prob) {
        selected_experts = BuildQwenNativeMoEFusedRouter(ctx, gate_logits, n_expert_used, &fused_router_state);
    }
    if (!selected_experts) {
        selected_experts = ggml_argsort_top_k(ctx, selection_scores, static_cast<int>(n_expert_used));
        ggml_set_name(selected_experts, "qwen35_native_moe_topk");
    }
    ggml_build_forward_expand(gf, selected_experts);

    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    ggml_tensor* hidden = nullptr;
    if (!graph_plan.lfm2_debug_reference &&
        CanUseQwen35NativeMoEGateUpRawQXKSwiGLU(model, raw_gate_exps, raw_up_exps, routed_input, selected_experts)) {
        Qwen35SharedQ8RowsUserData* gateup_q8_ud =
            AllocateQwen35SharedQ8RowsUserData(ctx, routed_input, n_expert_used * n_tokens);
        if (gateup_q8_ud) {
            SetNativeMoENumaContext(gateup_q8_ud, numa_backend, layer, model->variant);
            gateup_q8_ud->work_ctx = GetCurrentWorkContext();
            gateup_q8_ud->debug_layer_idx = layer_idx;
            gateup_q8_ud->requested_task_count = native_moe_callback_tasks;
            // Q4_K prefill is assignment-batched and stays on the raw-batched
            // k-quant path. Decode uses the validated repacked helper for
            // Qwen-like MoE variants when the kernel is available.
            gateup_q8_ud->prefer_q4k_repacked_swiglu = ShouldUseQwenLikeGateUpQ4KRepackedSwiGLU(
                graph_plan, w1w3_type, GetCurrentExecutionPhase(), Qwen35GateUpQ4KRepackKernelAvailable());
            gateup_q8_ud->prepacked_fused_gate_up_exps =
                graph_plan.qwen_native_moe && moe_weights.use_fused_gate_up &&
                        GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode &&
                        moe_weights.gate_up_exps->type == GGML_TYPE_Q4_K && ggml_cpu_has_avx2() &&
                        std::strstr(moe_weights.gate_up_exps->name, "cpu_repack_fused")
                    ? moe_weights.gate_up_exps
                    : nullptr;
            gateup_q8_ud->q5k_gateup_8x8_single_copy = native_q5_gateup_single_copy;
            gateup_q8_ud->q5k_gateup_8x8_single_copy_required =
                (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) && w1w3_type == GGML_TYPE_Q5_K;
            gateup_q8_ud->record_lfm2_w1w3_kernel = graph_plan.lfm2_native_moe;
            gateup_q8_ud->group_small_decode_gateup = ShouldGroupQwenLikeSmallDecodeGateUp(graph_plan, model->variant);
        }
        ggml_tensor* args[] = {raw_gate_exps, raw_up_exps, routed_input, selected_experts};
        hidden = ggml_custom_4d(ctx, GGML_TYPE_F32, raw_gate_exps->ne[1], n_expert_used, n_tokens, 1, args, 4,
                                cb_qwen35_native_moe_gateup_raw_qxk_swiglu, native_moe_callback_tasks, gateup_q8_ud);
        ggml_set_name(hidden, "qwen35_native_moe_gateup_raw_qxk_swiglu");
    } else if (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) {
        return reject_native_moe("w1w3_fast_callback_unavailable");
    } else {
        return reject_native_moe("unsupported_native_moe_graph");
    }
    if (!hidden) {
        hidden = ggml_swiglu_split(ctx, gate, up);
        ggml_set_name(hidden, "qwen35_native_moe_swiglu");
    }
    ggml_tensor* fused_out = nullptr;
    Qwen35SharedQ8RowsUserData* hidden_q8_ud = nullptr;
    ggml_tensor* fast_down_exps = graph_plan.qwen_native_moe ? raw_down_exps : down_exps;
    if (!graph_plan.lfm2_debug_reference && CanFuseQwen35W2NormWeightsFromLogitsWithCustomCallback(
                                                model, fast_down_exps, hidden, selected_experts, gate_logits)) {
        hidden_q8_ud = AllocateQwen35SharedQ8RowsUserData(
            ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1], fast_down_exps->ne[1]);
        if (hidden_q8_ud && graph_plan.lfm2_native_moe) {
            SetNativeMoENumaContext(hidden_q8_ud, numa_backend, layer, model->variant);
            hidden_q8_ud->work_ctx = GetCurrentWorkContext();
            hidden_q8_ud->debug_layer_idx = layer_idx;
            hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            hidden_q8_ud->weighted_logits_lfm2_sigmoid = true;
            hidden_q8_ud->weighted_logits_norm_topk = model->moe_norm_topk_prob;
            hidden_q8_ud->weighted_logits_scale = model->moe_routed_scaling_factor;
        } else if (hidden_q8_ud) {
            SetNativeMoENumaContext(hidden_q8_ud, numa_backend, layer, model->variant);
            hidden_q8_ud->work_ctx = GetCurrentWorkContext();
            hidden_q8_ud->debug_layer_idx = layer_idx;
            hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            hidden_q8_ud->fused_router_state = fused_router_state;
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts, gate_logits};
        fused_out =
            ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[1], 1, 1, args, 4,
                           cb_qwen35_native_moe_down_q5k_weighted_logits, native_moe_callback_tasks, hidden_q8_ud);
        ggml_set_name(fused_out, graph_plan.lfm2_native_moe ? "lfm2_native_moe_down_qxk_fast_weighted_logits"
                                                            : "qwen35_native_moe_down_q5k_fast_weighted_logits");
    }
    if (fused_out) {
        ggml_build_forward_expand(gf, fused_out);
        char name[80];
        std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_down_q5k_fast_out", layer_idx);
        ggml_set_name(fused_out, name);
        return fused_out;
    }

    ggml_tensor* weights = nullptr;
    if (graph_plan.lfm2_native_moe) {
        routing_probs = ggml_reshape_3d(ctx, routing_probs, 1, n_experts, n_tokens);
        weights = ggml_get_rows(ctx, routing_probs, selected_experts);
        ggml_set_name(weights, "lfm2_native_moe_weights");
        if (model->moe_norm_topk_prob) {
            weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            ggml_tensor* weight_sum = ggml_sum_rows(ctx, weights);
            weight_sum = ggml_clamp(ctx, weight_sum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, weight_sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
            ggml_set_name(weights, "lfm2_native_moe_norm_weights");
        }
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
            ggml_set_name(weights, "lfm2_native_moe_scaled_weights");
        }
    } else if (model->moe_norm_topk_prob) {
        weights = fused_router_state ? BuildQwenNativeMoEFusedRouterWeights(ctx, selected_experts, fused_router_state,
                                                                            "qwen35_native_moe_norm_weights")
                                     : BuildMoETopKWeightsFromLogits(ctx, gate_logits, selected_experts,
                                                                     "qwen35_native_moe_norm_weights");
        if (!weights) {
            return reject_native_moe("topk_weights_build_failed");
        }
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
            ggml_set_name(weights, "qwen35_native_moe_scaled_weights");
        }
    } else {
        ggml_tensor* probs = ggml_soft_max(ctx, gate_logits);
        ggml_set_name(probs, "qwen35_native_moe_probs");
        probs = ggml_reshape_3d(ctx, probs, 1, n_experts, n_tokens);
        weights = ggml_get_rows(ctx, probs, selected_experts);
        ggml_set_name(weights, "qwen35_native_moe_weights");
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
        }
    }
    ggml_build_forward_expand(gf, weights);

    if (!graph_plan.lfm2_debug_reference &&
        CanFuseQwen35W2WeightedSumWithCustomCallback(model, fast_down_exps, hidden, selected_experts, weights)) {
        if (!hidden_q8_ud) {
            hidden_q8_ud =
                AllocateQwen35SharedQ8RowsUserData(ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1]);
            if (hidden_q8_ud) {
                SetNativeMoENumaContext(hidden_q8_ud, numa_backend, layer, model->variant);
                hidden_q8_ud->work_ctx = GetCurrentWorkContext();
                hidden_q8_ud->debug_layer_idx = layer_idx;
                hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            }
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts, weights};
        fused_out = ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[1], 1, 1, args, 4,
                                   cb_qwen35_native_moe_down_q5k_weighted_sum, native_moe_callback_tasks, hidden_q8_ud);
        ggml_set_name(fused_out, "qwen35_native_moe_down_q5k_fast_weighted_sum");
    }
    if (fused_out) {
        ggml_build_forward_expand(gf, fused_out);
        char name[80];
        std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_down_q5k_fast_out", layer_idx);
        ggml_set_name(fused_out, name);
        return fused_out;
    }
    ggml_tensor* experts = nullptr;
    if (!graph_plan.lfm2_debug_reference &&
        CanReplaceQwen35W2WithCustomCallback(model, fast_down_exps, hidden, selected_experts)) {
        if (!hidden_q8_ud) {
            hidden_q8_ud =
                AllocateQwen35SharedQ8RowsUserData(ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1]);
            if (hidden_q8_ud) {
                SetNativeMoENumaContext(hidden_q8_ud, numa_backend, layer, model->variant);
                hidden_q8_ud->debug_layer_idx = layer_idx;
                hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            }
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts};
        experts =
            ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[0], selected_experts->ne[1],
                           1, args, 3, cb_qwen35_native_moe_down_q5k, native_moe_callback_tasks, hidden_q8_ud);
        ggml_set_name(experts, "qwen35_native_moe_down_q5k_fast");
    }
    if (!experts) {
        if (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) {
            return reject_native_moe("w2_fast_callback_unavailable");
        }
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] rejected: layer=%d w2 custom callback unavailable\n", layer_idx);
        }
        return nullptr;
    }

    ggml_tensor* out =
        BuildMoeExpertWeightedSumWithWeights(ctx, experts, weights, n_embd, n_tokens, "qwen35_native_moe_expert_sum");
    if (!out) {
        return reject_native_moe("expert_weighted_sum_build_failed");
    }
    ggml_build_forward_expand(gf, out);
    char name[80];
    std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_out", layer_idx);
    ggml_set_name(out, name);
    return out;
}


struct NativeMoEGraphCallbackTaskPlan {
    bool use_per_op_task_count = false;
    int work_items = 0;
};

static NativeMoEGraphCallbackTaskPlan ResolveNativeMoEGraphCallbackTaskPlan(const TransformerModel* model,
                                                                            InferenceExecutionPhase phase,
                                                                            int64_t n_tokens, int top_k) {
    NativeMoEGraphCallbackTaskPlan plan;
    if (model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv &&
        phase == InferenceExecutionPhase::Decode && n_tokens == 1 && top_k > 1) {
        plan.use_per_op_task_count = true;
        // LFM2 high-topK decode keeps all configured workers available; zero means no work-item cap.
        plan.work_items = 0;
    }
    return plan;
}

int ResolveNativeMoEGraphCallbackTaskCount(const TransformerModel* model, const BatchSpec* batch,
                                           InferenceExecutionPhase phase, int64_t n_tokens, int top_k) {
    const NativeMoEGraphCallbackTaskPlan plan = ResolveNativeMoEGraphCallbackTaskPlan(model, phase, n_tokens, top_k);
    if (plan.use_per_op_task_count) {
        return ResolveTaskCount(batch, plan.work_items);
    }
    return GGML_N_TASKS_MAX;
}
}  // namespace densecore::llm::graph::detail
using namespace densecore::llm::graph::detail;

NativeMoENumaStatsSnapshot GetNativeMoENumaStatsSnapshot() {
    const NativeMoENumaCounters& counters = GetNativeMoENumaCounters();
    NativeMoENumaStatsSnapshot snapshot;
    snapshot.context_set_total = counters.context_set_total.load(std::memory_order_relaxed);
    snapshot.context_enabled_total = counters.context_enabled_total.load(std::memory_order_relaxed);
    snapshot.sticky_dispatch_ops = counters.sticky_dispatch_ops.load(std::memory_order_relaxed);
    snapshot.legacy_dispatch_ops = counters.legacy_dispatch_ops.load(std::memory_order_relaxed);
    snapshot.grouped_decode_used_ops = counters.grouped_decode_used_ops.load(std::memory_order_relaxed);
    snapshot.grouped_dispatch_node_tasks = counters.grouped_dispatch_node_tasks.load(std::memory_order_relaxed);
    snapshot.grouped_dispatch_expert_items = counters.grouped_dispatch_expert_items.load(std::memory_order_relaxed);
    snapshot.direct_decode_used_ops = counters.direct_decode_used_ops.load(std::memory_order_relaxed);
    snapshot.direct_decode_rejected_ops = counters.direct_decode_rejected_ops.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < snapshot.node_dispatch_ops.size(); ++i) {
        snapshot.node_dispatch_ops[i] = counters.node_dispatch_ops[i].load(std::memory_order_relaxed);
    }
    snapshot.node_dispatch_overflow_ops = counters.node_dispatch_overflow_ops.load(std::memory_order_relaxed);
    snapshot.last_state = counters.last_state.load(std::memory_order_relaxed);
    return snapshot;
}

const char* GetNativeMoENumaStickyStateName(int state) {
    switch (static_cast<NativeMoENumaStickyState>(state)) {
    case NativeMoENumaStickyState::Unset: return "unset";
    case NativeMoENumaStickyState::NoBackend: return "no_backend";
    case NativeMoENumaStickyState::SingleNode: return "single_node";
    case NativeMoENumaStickyState::PlacementUnavailable: return "placement_unavailable";
    case NativeMoENumaStickyState::PlacementInvalid: return "placement_invalid";
    case NativeMoENumaStickyState::Enabled: return "enabled";
    case NativeMoENumaStickyState::EnabledSingleNodeDegenerate: return "enabled_degenerate";
    case NativeMoENumaStickyState::DisabledByDebug: return "disabled_by_debug";
    case NativeMoENumaStickyState::DisabledByPolicy: return "disabled_by_policy";
    case NativeMoENumaStickyState::DisabledUnqualifiedModel: return "disabled_unqualified_model";
    }
    return "unknown";
}

void RecordMoEGraphWiring() {
    densecore::llm::graph::detail::g_moe_graph_wiring_debug_counter.fetch_add(1, std::memory_order_relaxed);
}

void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}

uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}

#ifdef DENSECORE_TEST_BUILD
#include "llm/runtime/work_context_internal.h"
namespace densecore::testing {
bool QwenQ4DownPairAvailableForTest() {
    return ShouldUseQwen36Q4DownRowPair(ModelVariant::QWEN36, InferenceExecutionPhase::Decode, 4);
}

bool RunQwenQ4DownPairCallbackForTest(bool qwen36, int tokens, int phase, bool* matches, uint64_t* used) {
    if (!matches || !used || tokens < 1 || tokens > 4) return false;
    ggml_cpu_init();
    constexpr int cols = 512, rows = 37, experts = 2, threads = 16;
    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    if (!ctx) return false;
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ctx, ggml_free);
    auto* down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, cols, rows, experts);
    auto* hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, experts, tokens);
    auto* selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, experts, tokens);
    auto* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, tokens);
    auto* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    std::vector<float> values(cols);
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    for (int r = 0; r < rows * experts; ++r) {
        for (int c = 0; c < cols; ++c) values[c] = std::sin((r * 13 + c) * 0.03f);
        traits->from_float(values.data(), static_cast<char*>(down->data) + r * down->nb[1], cols);
    }
    for (int i = 0; i < cols * experts * tokens; ++i) static_cast<float*>(hidden->data)[i] = std::cos(i * 0.07f);
    for (int i = 0; i < experts * tokens; ++i) {
        static_cast<int32_t*>(selected->data)[i] = i % experts;
        static_cast<float*>(logits->data)[i] = 0.125f * i;
    }
    InferenceWorkContext work{};
    work.phase = static_cast<InferenceExecutionPhase>(phase);
    Qwen35SharedQ8RowsUserData shared{};
    shared.work_ctx = &work;
    shared.execution_phase = work.phase;
    const auto run = [&](ModelVariant variant) {
        work.model_variant = variant;
        shared.model_variant = variant;
        const auto before = GetNativeMoENumaCounters().legacy_dispatch_ops.load();
        std::vector<std::thread> workers;
        for (int ith = 0; ith < threads; ++ith)
            workers.emplace_back([&, ith] {
                RunQwen35NativeMoEDownQ5KWeightedLogitsFastPath(output, down, hidden, selected, logits, ith, threads,
                                                                &shared);
            });
        for (auto& worker : workers) worker.join();
        return GetNativeMoENumaCounters().legacy_dispatch_ops.load() - before == uint64_t(threads * experts * tokens);
    };
    if (!run(ModelVariant::QWEN35)) return false;
    const auto* data = static_cast<float*>(output->data);
    const std::vector<float> reference(data, data + rows * tokens);
    const auto before = q4_down_native_pair_test_ops.load();
    if (!run(qwen36 ? ModelVariant::QWEN36 : ModelVariant::QWEN35)) return false;
    *used = q4_down_native_pair_test_ops.load() - before;
    *matches = std::memcmp(reference.data(), output->data, reference.size() * sizeof(float)) == 0;
    for (int i = 0; i < rows * tokens; ++i) *matches = *matches && std::isfinite(data[i]);
    return true;
}

bool RunQwen36PrefillDownWithoutWorkerTLSForTest(bool* exact, uint64_t* repacked_bytes) {
    if (!exact || !repacked_bytes) return false;
    ggml_cpu_init();
    constexpr int cols = 512, rows = 16, tokens = 5;
    ggml_context* ctx = ggml_init({2 * 1024 * 1024, nullptr, false});
    if (!ctx) return false;
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ctx, ggml_free);
    auto* down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, cols, rows, 1);
    auto* reference = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    auto* actual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    std::vector<float> values(cols);
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) values[c] = std::sin(float(c + 3 * r) * 0.07f) * 0.05f;
        traits->from_float(values.data(), static_cast<char*>(down->data) + r * down->nb[1], cols);
    }
    const size_t qstride = ggml_row_size(GGML_TYPE_Q8_K, cols);
    std::vector<uint8_t> quantized(tokens * qstride);
    std::vector<Qwen35MoEAssignment> assignments;
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < cols; ++c) values[c] = std::cos(float(c + 7 * t) * 0.11f);
        if (!hwy_kernels::QuantizeRowQ8K_Hwy(values.data(), quantized.data() + t * qstride, cols)) return false;
        assignments.push_back({0, t, 0, 0.25f + t * 0.125f});
    }
    Qwen35SharedQ8RowsUserData ud{};
    ud.execution_phase = InferenceExecutionPhase::Prefill;
    const auto run = [&](ggml_tensor* dst, bool bind_tls) {
        bool ok = false;
        std::thread worker([&] {
            InferenceWorkContext context{};
            context.phase = InferenceExecutionPhase::Prefill;
            SetCurrentWorkContext(bind_tls ? &context : nullptr);
            std::fill_n(static_cast<float*>(dst->data), rows * tokens, 0.0f);
            ok = Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(dst, down, quantized.data(), qstride,
                                                                        assignments, 0, 0, rows, 0, false,
                                                                        ResolveNativeMoEOuterTaskExecutionPhase(&ud));
            SetCurrentWorkContext(nullptr);
        });
        worker.join();
        return ok;
    };
    if (!run(reference, true)) return false;
    const auto before = kernels::Q4KRepackedGemvCacheStatsSnapshot();
    if (!run(actual, false)) return false;
    const auto after = kernels::Q4KRepackedGemvCacheStatsSnapshot();
    *repacked_bytes = after.repack_bytes - before.repack_bytes;
    *exact = std::memcmp(reference->data, actual->data, rows * tokens * sizeof(float)) == 0;
    for (int i = 0; i < rows * tokens; ++i)
        if (!std::isfinite(static_cast<float*>(actual->data)[i])) return false;
    return true;
}

bool Qwen36SmallDecodeGateUpGroupingPolicyForTest(int variant, int phase, bool hybrid) {
    auto* previous_context = GetCurrentWorkContext();
    InferenceWorkContext context{};
    SetCurrentWorkContext(&context);
    SetCurrentExecutionPhase(static_cast<InferenceExecutionPhase>(phase));
    TransformerModel model{};
    model.variant = static_cast<ModelVariant>(variant);
    model.arch_flags.is_hybrid_ssm = hybrid;
    const auto plan = ResolveQwenLikeNativeMoEGraphPlan(&model);
    const bool enabled = ShouldGroupQwenLikeSmallDecodeGateUp(plan, model.variant);
    SetCurrentWorkContext(previous_context);
    return enabled;
}

bool RunQwen36SmallDecodeGateUpGroupingForTest(int type_id, int tokens, int overlap, float input_scale,
                                               float* max_error) {
    ggml_cpu_init();
    const auto type = static_cast<ggml_type>(type_id);
    if (!max_error || (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K) ||
        (tokens != 2 && tokens != 3 && tokens != 4) || overlap < 0 || overlap > 2)
        return false;
    constexpr int cols = 2048, rows = 64, experts = 32, top_k = 8, threads = 16;
    ggml_context* ctx = ggml_init({8 * 1024 * 1024, nullptr, false});
    if (!ctx) return false;
    struct Guard {
        ggml_context* ctx;
        ~Guard() { ggml_free(ctx); }
    } guard{ctx};
    auto* gate = ggml_new_tensor_3d(ctx, type, cols, rows, experts);
    auto* up = ggml_new_tensor_3d(ctx, type, cols, rows, experts);
    auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
    auto* selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, tokens);
    auto* reference = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, top_k, tokens);
    auto* candidate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, top_k, tokens);
    auto* reference_ud = AllocateQwen35SharedQ8RowsUserData(ctx, input, tokens * top_k);
    auto* candidate_ud = AllocateQwen35SharedQ8RowsUserData(ctx, input, tokens * top_k);
    if (!reference_ud || !candidate_ud) return false;
    candidate_ud->group_small_decode_gateup = true;
    const auto sample = [](uint32_t seed) {
        seed ^= seed >> 16;
        seed *= 0x7feb352du;
        seed ^= seed >> 15;
        seed *= 0x846ca68bu;
        seed ^= seed >> 16;
        return (static_cast<float>(seed & 0xffffu) / 32767.5f - 1.0f) * 1.7320508f;
    };
    std::vector<float> weights(cols);
    const auto* traits = ggml_get_type_traits_cpu(type);
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) weights[c] = 0.05f * sample(uint32_t((e * rows + r) * cols + c) + 1u);
            traits->from_float(weights.data(), static_cast<char*>(gate->data) + e * gate->nb[2] + r * gate->nb[1],
                               cols);
            for (int c = 0; c < cols; ++c) weights[c] = 0.05f * sample(uint32_t((e * rows + r) * cols + c) + 0x987654u);
            traits->from_float(weights.data(), static_cast<char*>(up->data) + e * up->nb[2] + r * up->nb[1], cols);
        }
    }
    *max_error = 0.0f;
    for (int round = 0; round < 3; ++round) {
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < cols; ++c)
                static_cast<float*>(input->data)[t * cols + c] =
                    input_scale * sample(uint32_t((round * tokens + t) * cols + c) + 0x12345u);
            for (int k = 0; k < top_k; ++k) {
                const int base = overlap == 0 ? t * top_k : overlap == 1 ? 0 : t * 4;
                // Reverse top-k order so expert sorting cannot masquerade as correct scatter.
                static_cast<int32_t*>(selected->data)[t * top_k + k] = (base + top_k - 1 - k + round) % experts;
            }
        }
        const auto run = [&](ggml_tensor* dst, Qwen35SharedQ8RowsUserData* ud) {
            InferenceWorkContext context{};
            context.phase = InferenceExecutionPhase::Decode;
            context.execution_generation = static_cast<uint64_t>(round + 1);
            ud->work_ctx = &context;
            ud->execution_phase = InferenceExecutionPhase::Decode;
            std::fill_n(static_cast<float*>(dst->data), rows * top_k * tokens, std::numeric_limits<float>::quiet_NaN());
            std::vector<std::thread> workers;
            for (int ith = 0; ith < threads; ++ith)
                workers.emplace_back([&, ith] {
                    SetCurrentWorkContext(round == 2 && ith != 0 ? nullptr : &context);
                    ud->outer_task_width.store(threads, std::memory_order_relaxed);
                    RunQwen35NativeMoEGateUpRawQXKSwiGLU(dst, gate, up, input, selected, ith, threads, ud);
                    SetCurrentWorkContext(nullptr);
                });
            for (auto& worker : workers) worker.join();
            return ud->repacked_swiglu_failed.load() == 0;
        };
        const uint64_t grouped_before = candidate_ud->grouped_gateup_tiles.load();
        if (!run(reference, reference_ud) || !run(candidate, candidate_ud) || reference_ud->assignment_count != 0 ||
            candidate_ud->assignment_count != (type == GGML_TYPE_Q4_K ? size_t(tokens * top_k) : 0))
            return false;
        const bool grouped_compute = candidate_ud->grouped_gateup_tiles.load() > grouped_before;
        if (grouped_compute != (type == GGML_TYPE_Q4_K && overlap != 0) ||
            reference_ud->grouped_gateup_tiles.load() != 0)
            return false;
        for (int i = 0; i < rows * top_k * tokens; ++i) {
            const float actual = static_cast<float*>(candidate->data)[i];
            const float expected = static_cast<float*>(reference->data)[i];
            if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
            *max_error = std::max(*max_error, std::fabs(actual - expected));
        }
    }
    return true;
}

bool NativeMoENumaStickyQualifiedModelVariantForTest(ModelVariant variant) {
    return ::NativeMoENumaStickyQualifiedModelVariant(variant);
}

void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}

uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}


bool RunQwen35NativeMoEQ4KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols, float* output) {
    return ::Qwen35NativeMoEQ4KQ8KDotRow(weight_row, static_cast<const uint8_t*>(q8_input), cols, output);
}

bool RunQwen35NativeMoEQ5KFusedSwiGLURowsForTest(const void* gate_rows, const void* up_rows, const void* q8_input,
                                                 int64_t cols, int64_t row_count, size_t row_bytes, float* output) {
    return ::Qwen35NativeMoEKQ8KFusedSwiGLURows(nullptr, GGML_TYPE_Q5_K, gate_rows, up_rows,
                                                static_cast<const uint8_t*>(q8_input), cols, row_count, row_bytes,
                                                output);
}

int64_t Qwen35NativeMoEMaxDirectTokensForTest() {
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = ModelVariant::QWEN36;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_experts = 1;
    model.hparams.n_experts_used = 1;
    model.layers.resize(1);
    model.layers[0].is_moe = true;
    return ::NativeMoEFastPathMaxDirectTokens(&model);
}

bool RunQwen35NativeQuantizeRowQ8KForTest(const float* input, void* q8_output, int64_t cols) {
    return ::Qwen35NativeQuantizeRowQ8K(input, static_cast<uint8_t*>(q8_output), cols);
}

int ResolveNativeMoEGraphCallbackTaskCountForTest(const TransformerModel* model, const BatchSpec* batch, int phase,
                                                  int64_t n_tokens, int top_k) {
    return ::ResolveNativeMoEGraphCallbackTaskCount(model, batch, static_cast<InferenceExecutionPhase>(phase), n_tokens,
                                                    top_k);
}

bool ComputeQwenNativeMoEStableTopKForTest(const std::vector<float>& logits, int top_k, std::vector<int32_t>* selected,
                                           std::vector<float>* normalized_weights) {
    if (!selected || !normalized_weights || logits.empty() || top_k <= 0) {
        return false;
    }
    selected->assign(static_cast<size_t>(top_k), -1);
    normalized_weights->assign(static_cast<size_t>(top_k), 0.0f);
    return ::ComputeQwenNativeMoEStableTopK(logits.data(), static_cast<int64_t>(logits.size()), top_k, selected->data(),
                                            normalized_weights->data());
}

bool RunQwenNativeMoEFusedRouterCustomNodeForTest(const std::vector<float>& logits, std::vector<int32_t>* selected,
                                                  std::vector<float>* normalized_weights) {
    if (!selected || !normalized_weights || logits.size() < static_cast<size_t>(kQwenNativeMoEFusedRouterTopK)) {
        return false;
    }
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor* logits_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int64_t>(logits.size()), 1);
    if (!logits_tensor || !logits_tensor->data) {
        ggml_free(ctx);
        return false;
    }
    std::memcpy(logits_tensor->data, logits.data(), logits.size() * sizeof(float));
    QwenNativeMoEFusedRouterState* state = nullptr;
    ggml_tensor* selected_tensor =
        BuildQwenNativeMoEFusedRouter(ctx, logits_tensor, kQwenNativeMoEFusedRouterTopK, &state);
    ggml_tensor* weights_tensor =
        selected_tensor && state
            ? BuildQwenNativeMoEFusedRouterWeights(ctx, selected_tensor, state, "qwen_router_weights_test")
            : nullptr;
    if (!selected_tensor || !selected_tensor->data || !weights_tensor || !weights_tensor->data) {
        ggml_free(ctx);
        return false;
    }
    cb_qwen_native_moe_fused_router(selected_tensor, 0, 1, state);
    cb_qwen_native_moe_fused_router_weights(weights_tensor, 0, 1, state);
    selected->assign(static_cast<const int32_t*>(selected_tensor->data),
                     static_cast<const int32_t*>(selected_tensor->data) + kQwenNativeMoEFusedRouterTopK);
    normalized_weights->resize(kQwenNativeMoEFusedRouterTopK);
    for (int64_t k = 0; k < kQwenNativeMoEFusedRouterTopK; ++k) {
        (*normalized_weights)[static_cast<size_t>(k)] = *reinterpret_cast<const float*>(
            static_cast<const char*>(weights_tensor->data) + static_cast<size_t>(k) * weights_tensor->nb[1]);
    }
    ggml_free(ctx);
    return true;
}

bool ShouldEnableNativeMoEFastPathByDefaultForTest(const TransformerModel* model, int phase, int mode) {
    return ::ShouldEnableNativeMoEFastPathByDefault(model, static_cast<InferenceExecutionPhase>(phase),
                                                    static_cast<densecore::env::RuntimeToggleMode>(mode));
}

bool CanUseQwenNativeMoEGateUpForTest(const TransformerModel* model, const ggml_tensor* gate_exps,
                                      const ggml_tensor* up_exps, const ggml_tensor* input,
                                      const ggml_tensor* selected_experts, int phase) {
    InferenceWorkContext* previous_ctx = GetCurrentWorkContext();
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> owned_ctx(nullptr,
                                                                                     DestroyInferenceWorkContext);
    if (!previous_ctx) {
        owned_ctx.reset(CreateInferenceWorkContext());
        SetCurrentWorkContext(owned_ctx.get());
    }
    const InferenceExecutionPhase previous = GetCurrentExecutionPhase();
    SetCurrentExecutionPhase(static_cast<InferenceExecutionPhase>(phase));
    const bool accepted = ::CanUseQwen35NativeMoEGateUpRawQXKSwiGLU(model, gate_exps, up_exps, input, selected_experts);
    SetCurrentExecutionPhase(previous);
    if (!previous_ctx) {
        SetCurrentWorkContext(nullptr);
    }
    return accepted;
}

bool CanUseQwenNativeMoEW2ForTest(const TransformerModel* model, const ggml_tensor* down_exps,
                                  const ggml_tensor* hidden, const ggml_tensor* selected_experts, int phase) {
    InferenceWorkContext* previous_ctx = GetCurrentWorkContext();
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> owned_ctx(nullptr,
                                                                                     DestroyInferenceWorkContext);
    if (!previous_ctx) {
        owned_ctx.reset(CreateInferenceWorkContext());
        SetCurrentWorkContext(owned_ctx.get());
    }
    const InferenceExecutionPhase previous = GetCurrentExecutionPhase();
    SetCurrentExecutionPhase(static_cast<InferenceExecutionPhase>(phase));
    const bool accepted = ::CanReplaceQwen35W2WithCustomCallback(model, down_exps, hidden, selected_experts);
    SetCurrentExecutionPhase(previous);
    if (!previous_ctx) {
        SetCurrentWorkContext(nullptr);
    }
    return accepted;
}

bool RemapNativeMoECallbackTaskForTest(int requested_task_count, int ith, int nth, int* effective_ith,
                                       int* effective_nth) {
    Qwen35SharedQ8RowsUserData ud{};
    ud.requested_task_count = requested_task_count;
    return ::RemapNativeMoECallbackTask(&ud, ith, nth, effective_ith, effective_nth);
}

bool BuildNativeMoENodeGroupPlanForTest(const std::vector<int>& item_nodes, int node_count,
                                        std::vector<int>* active_nodes) {
    if (!active_nodes) {
        return false;
    }
    NativeMoENodeGroupPlan plan;
    if (!::BuildNativeMoENodeGroupPlan(item_nodes.data(), static_cast<int>(item_nodes.size()), node_count, &plan)) {
        active_nodes->clear();
        return false;
    }
    active_nodes->assign(plan.active_nodes.begin(), plan.active_nodes.begin() + plan.active_node_count);
    return true;
}

bool BuildNativeMoEOuterTaskPlanForTest(const std::vector<int>& task_nodes, int task_index, int node_count,
                                        int* task_node, int* node_task_rank, int* node_task_count) {
    if (!task_node || !node_task_rank || !node_task_count) {
        return false;
    }
    NativeMoEOuterTaskPlan plan;
    if (!::BuildNativeMoEOuterTaskPlan(task_nodes.data(), static_cast<int>(task_nodes.size()), task_index, node_count,
                                       &plan)) {
        return false;
    }
    *task_node = plan.task_node;
    *node_task_rank = plan.node_task_rank;
    *node_task_count = plan.node_task_count;
    return true;
}

bool ShouldUseQwenNativeMoEDownGlobalRowPartitionForTest(bool numa_sticky_enabled, bool weighted_logits_lfm2_sigmoid,
                                                         int phase, int64_t n_tokens, int nth) {
    return ::ShouldUseQwen35NativeMoEDownGlobalRowPartition(numa_sticky_enabled, weighted_logits_lfm2_sigmoid,
                                                            static_cast<InferenceExecutionPhase>(phase), n_tokens, nth);
}

bool ShouldNarrowNativeMoECallbackTasksForStickyFallbackForTest(bool sticky_armed, int phase,
                                                                bool direct_outer_candidate) {
    return ::ShouldNarrowNativeMoECallbackTasksForStickyFallback(
        sticky_armed, static_cast<InferenceExecutionPhase>(phase), direct_outer_candidate);
}

bool RunNativeMoEDirectOuterTaskBarrierForTest(int task_count, int failing_task, int rounds) {
    if (task_count <= 1 || task_count > kQwen35SharedQ8MaxTasks || failing_task >= task_count || rounds <= 0) {
        return false;
    }
    Qwen35SharedQ8RowsUserData ud{};
    for (int round = 0; round < rounds; ++round) {
        std::atomic<int> matching_results{0};
        std::atomic<int> owner_results{0};
        std::vector<std::thread> tasks;
        tasks.reserve(static_cast<size_t>(task_count));
        for (int task = 0; task < task_count; ++task) {
            tasks.emplace_back([&, task] {
                const uint64_t epoch = ::BeginNativeMoEDirectOuterTask(&ud, task, task_count, true);
                bool owner_task = false;
                const bool result =
                    ::FinishNativeMoEDirectOuterTask(&ud, task, epoch, task != failing_task, &owner_task);
                const bool expected = failing_task < 0;
                if (owner_task) {
                    owner_results.fetch_add(1, std::memory_order_relaxed);
                    matching_results.fetch_add(result == expected ? 1 : 0, std::memory_order_relaxed);
                } else {
                    matching_results.fetch_add(result ? 1 : 0, std::memory_order_relaxed);
                }
            });
        }
        for (std::thread& task : tasks) {
            task.join();
        }
        if (matching_results.load(std::memory_order_relaxed) != task_count ||
            owner_results.load(std::memory_order_relaxed) != 1) {
            return false;
        }
    }
    return true;
}

std::vector<float> ReduceNativeMoEExpertOutputsInRoutingOrderForTest(const std::vector<float>& expert_outputs,
                                                                     const std::vector<int>& experts,
                                                                     const std::vector<float>& weights, int row_count) {
    if (row_count <= 0 || experts.size() != weights.size() ||
        expert_outputs.size() != experts.size() * static_cast<size_t>(row_count)) {
        return {};
    }
    std::vector<float> output(static_cast<size_t>(row_count), 0.0f);
    ::ReduceNativeMoEExpertOutputsInRoutingOrder(output.data(), sizeof(float), expert_outputs.data(), experts.data(),
                                                 weights.data(), static_cast<int>(experts.size()), row_count);
    return output;
}

bool QwenNativeMoENoAllocUserDataIsPerOpForTest() {
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor* src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, QK_K, 1, 1);
    auto* first = ::AllocateQwen35SharedQ8RowsUserData(ctx, src, 8, 64);
    auto* second = ::AllocateQwen35SharedQ8RowsUserData(ctx, src, 8, 64);
    const bool per_op = first && second && first != second && first->rows != second->rows &&
                        first->assignments != second->assignments &&
                        first->direct_expert_outputs != second->direct_expert_outputs &&
                        first->direct_expert_output_capacity == 8 * 64;
    const auto destroy = [](Qwen35SharedQ8RowsUserData* ud) {
        if (!ud) return;
        delete[] ud->rows;
        delete[] ud->assignments;
        delete[] ud->direct_expert_outputs;
        delete ud;
    };
    destroy(first);
    destroy(second);
    ggml_free(ctx);
    return per_op;
}

bool NativeMoELegacyDispatchBatchForTest(int exit_mode, int threads, int calls) {
    if (threads < 1 || calls < 1) return false;
    std::atomic<uint64_t> count{0};
    std::atomic<bool> correct{true};
    // Validate deferred publication and an idempotent explicit flush without
    // other writers; then exercise concurrent scope-exit publication.
    {
        NativeMoELegacyDispatchBatch batch(count);
        batch.Record();
        if (count.load() != 0) return false;
        batch.Flush();
        batch.Flush();
        if (count.load() != 1) return false;
    }
    count.store(0);
    const auto run = [&] {
        try {
            NativeMoELegacyDispatchBatch batch(count);
            for (int i = 0; i < calls; ++i) batch.Record();
            if (exit_mode == 1) return;
            if (exit_mode == 2) throw std::runtime_error("dispatch test");
            batch.Flush();
        } catch (const std::runtime_error&) {
            if (exit_mode != 2) correct.store(false);
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) workers.emplace_back(run);
    for (auto& worker : workers) worker.join();
    return correct.load() && count.load() == static_cast<uint64_t>(threads) * calls;
}
}  // namespace densecore::testing
#endif

namespace densecore::testing {
bool ShouldRunMoESharedDenseBranchForTest(const TransformerModel* model, bool is_gemma4_moe,
                                          const struct ggml_tensor* ffn_gate, const struct ggml_tensor* ffn_up,
                                          const struct ggml_tensor* ffn_down) {
    return ShouldRunMoESharedDenseBranch(model, nullptr, is_gemma4_moe, ffn_gate, ffn_up, ffn_down);
}

bool ShouldUseQwenLikeGateUpQ4KRepackedSwiGLUForTest(bool qwen_native_moe, bool lfm2_native_moe, int phase,
                                                     bool kernel_available) {
    QwenLikeNativeMoEGraphPlan graph_plan;
    graph_plan.qwen_native_moe = qwen_native_moe;
    graph_plan.lfm2_native_moe = lfm2_native_moe;
    graph_plan.graph_phase = static_cast<InferenceExecutionPhase>(phase);
    return ::ShouldUseQwenLikeGateUpQ4KRepackedSwiGLU(graph_plan, GGML_TYPE_Q4_K,
                                                      static_cast<InferenceExecutionPhase>(phase), kernel_available);
}
}  // namespace densecore::testing
