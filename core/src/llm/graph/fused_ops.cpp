#include "densecore/exceptions.h"
#include "densecore/runtime/dtype_utils.h"
#include "densecore/runtime/optimization_bridge.h"
#include "densecore/simd/simd_ops.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/support_internal.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/work_context.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace densecore::llm::graph::detail {}  // namespace densecore::llm::graph::detail

using namespace densecore::llm::graph::detail;

// ============================================================================
// Multi-LoRA Batching Callback
// ============================================================================
// Applies per-request LoRA adapters during inference graph execution.
// Uses thread-local batch context to access the adapter-to-token mapping.
// The tensor name (e.g., "blk.0.attn_q") identifies which layer weights to use.
// ============================================================================

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

    ScopedInferenceWorkContext callback_context(static_cast<InferenceWorkContext*>(userdata));
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
    densecore::CpuBackend& backend = densecore::llm::runtime::ResolveCpuBackend(GetCurrentBatch());
    backend.ApplyMultiLoRA(t_input, std::string(layer_name), batch->lora_map, &t_output);
}

// =============================================================================
// Parallel GEMV User Data + Buffers
// =============================================================================

/**
 * User data for parallel GEMV operation
 */


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
    if (!src || !dst || !src->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (src->nb[0] != static_cast<int64_t>(sizeof(float)) || dst->nb[0] != static_cast<int64_t>(sizeof(float))) return;

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    const float eps = ud->eps;
    const bool has_residual = ud->residual != nullptr;
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t residual_row_stride = ud->residual_row_stride > 0 ? ud->residual_row_stride : n_embd;
    const bool run_reference_probe = ShouldRunAddRmsNormReferenceProbe(ud->layer_idx);

    // Partition work across tokens
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens) return;

    // Process assigned tokens
    for (int t = t_start; t < t_end; t++) {
        const float* x_ptr = reinterpret_cast<const float*>(src->data) + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* out_ptr = reinterpret_cast<float*>(dst->data) + static_cast<ptrdiff_t>(t) * dst_row_stride;
        const float* res_ptr =
            has_residual ? (ud->residual + static_cast<ptrdiff_t>(t) * residual_row_stride) : nullptr;

        if (has_residual) {
            // Use unified AddRMSNorm dispatcher (Runtime AVX512/AVX2/Scalar)
            densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);
        } else {
            densecore::simd::RMSNorm(x_ptr, ud->rms_weight, out_ptr, static_cast<size_t>(n_embd), eps);
        }

        if (run_reference_probe) {
            static std::atomic<int> emitted{0};
            const int max_calls =
                densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_MAX_CALLS", 8);
            const int prior = emitted.load(std::memory_order_relaxed);
            if (prior < max_calls && emitted.fetch_add(1, std::memory_order_relaxed) < max_calls) {
                std::vector<float> summed(static_cast<size_t>(n_embd));
                double sum_sq = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    float val = x_ptr[i];
                    if (res_ptr) {
                        val += res_ptr[i];
                    }
                    summed[static_cast<size_t>(i)] = val;
                    sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }

                const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);
                float max_abs_diff = 0.0f;
                int first_bad_idx = -1;
                float first_actual = 0.0f;
                float first_ref = 0.0f;
                bool actual_nonfinite = false;
                bool ref_nonfinite = false;
                for (int i = 0; i < n_embd; ++i) {
                    const float ref = summed[static_cast<size_t>(i)] * inv_rms * ud->rms_weight[i];
                    const float actual = out_ptr[i];
                    actual_nonfinite = actual_nonfinite || !std::isfinite(actual);
                    ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
                    const float diff = std::fabs(actual - ref);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        first_bad_idx = i;
                        first_actual = actual;
                        first_ref = ref;
                    }
                }
                const int seq_id = ud->token_seq_ids ? ud->token_seq_ids[t] : -1;
                fprintf(stderr,
                        "[ADD_RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d has_residual=%d "
                        "max_abs_diff=%.9g first_idx=%d actual=%.9g ref=%.9g actual_nonfinite=%d "
                        "ref_nonfinite=%d\n",
                        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", t,
                        seq_id, has_residual ? 1 : 0, max_abs_diff, first_bad_idx, first_actual, first_ref,
                        actual_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0);
            }
        }
    }
}

void cb_residual_rmsnorm_fused2(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                const struct ggml_tensor* residual, int ith, int nth, void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->rms_weight) return;
    if (!src || !residual || !dst || !src->data || !residual->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (src->nb[0] != static_cast<int64_t>(sizeof(float)) || residual->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    if (n_embd <= 0 || n_tokens <= 0 || src->ne[0] != n_embd || residual->ne[0] != n_embd || dst->ne[0] != n_embd) {
        return;
    }

    const float eps = ud->eps;
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t residual_row_stride = static_cast<ptrdiff_t>(residual->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const bool run_reference_probe = ShouldRunAddRmsNormReferenceProbe(ud->layer_idx);
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);
    if (t_start >= n_tokens) return;

    for (int t = t_start; t < t_end; ++t) {
        const float* x_ptr = reinterpret_cast<const float*>(src->data) + static_cast<ptrdiff_t>(t) * src_row_stride;
        const float* res_ptr =
            reinterpret_cast<const float*>(residual->data) + static_cast<ptrdiff_t>(t) * residual_row_stride;
        float* out_ptr = reinterpret_cast<float*>(dst->data) + static_cast<ptrdiff_t>(t) * dst_row_stride;
        densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);

        if (run_reference_probe) {
            static std::atomic<int> emitted{0};
            const int max_calls =
                densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_MAX_CALLS", 8);
            const int prior = emitted.load(std::memory_order_relaxed);
            if (prior < max_calls && emitted.fetch_add(1, std::memory_order_relaxed) < max_calls) {
                double sum_sq = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    const float val = x_ptr[i] + res_ptr[i];
                    sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }

                const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);
                float max_abs_diff = 0.0f;
                int first_bad_idx = -1;
                float first_actual = 0.0f;
                float first_ref = 0.0f;
                bool actual_nonfinite = false;
                bool ref_nonfinite = false;
                for (int i = 0; i < n_embd; ++i) {
                    const float ref = (x_ptr[i] + res_ptr[i]) * inv_rms * ud->rms_weight[i];
                    const float actual = out_ptr[i];
                    actual_nonfinite = actual_nonfinite || !std::isfinite(actual);
                    ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
                    const float diff = std::fabs(actual - ref);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        first_bad_idx = i;
                        first_actual = actual;
                        first_ref = ref;
                    }
                }
                const int seq_id = ud->token_seq_ids ? ud->token_seq_ids[t] : -1;
                fprintf(stderr,
                        "[ADD_RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d has_residual=1 "
                        "max_abs_diff=%.9g first_idx=%d actual=%.9g ref=%.9g actual_nonfinite=%d "
                        "ref_nonfinite=%d\n",
                        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", t,
                        seq_id, max_abs_diff, first_bad_idx, first_actual, first_ref, actual_nonfinite ? 1 : 0,
                        ref_nonfinite ? 1 : 0);
            }
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

    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }
    if (a->nb[0] != static_cast<int64_t>(sizeof(float)) || b->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    // Total elements (flattened)
    const size_t size = ggml_nelements(a);

    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst)) {
        densecore::simd::SiLUMulParallel(out, gate, up, size, ith, nth);
        return;
    }

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    const auto offset_bytes = [](const ggml_tensor* tensor, size_t flat_idx) -> size_t {
        size_t rem = flat_idx;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            const int64_t extent = tensor->ne[dim] > 0 ? tensor->ne[dim] : 1;
            const size_t coord = rem % static_cast<size_t>(extent);
            rem /= static_cast<size_t>(extent);
            offset += coord * static_cast<size_t>(tensor->nb[dim]);
        }
        return offset;
    };
    for (size_t flat = begin; flat < end; ++flat) {
        const float g = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(a->data) + offset_bytes(a, flat));
        const float u = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(b->data) + offset_bytes(b, flat));
        float* dst_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + offset_bytes(dst, flat));
        *dst_ptr = (g / (1.0f + std::exp(-g))) * u;
    }
}

void cb_gelu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;
    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3] ||
        dst->ne[0] != a->ne[0] || dst->ne[1] != a->ne[1] || dst->ne[2] != a->ne[2] || dst->ne[3] != a->ne[3]) {
        return;
    }
    if (a->nb[0] != static_cast<int64_t>(sizeof(float)) || b->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    const size_t size = ggml_nelements(a);
    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst)) {
        const float* gate = reinterpret_cast<const float*>(a->data);
        const float* up = reinterpret_cast<const float*>(b->data);
        float* out = reinterpret_cast<float*>(dst->data);
        for (size_t i = begin; i < end; ++i) {
            const float x = gate[i];
            const float x3 = x * x * x;
            const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
            out[i] = gelu * up[i];
        }
        return;
    }

    const auto offset_bytes = [](const ggml_tensor* tensor, size_t flat_idx) -> size_t {
        size_t rem = flat_idx;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            const int64_t extent = tensor->ne[dim] > 0 ? tensor->ne[dim] : 1;
            const size_t coord = rem % static_cast<size_t>(extent);
            rem /= static_cast<size_t>(extent);
            offset += coord * static_cast<size_t>(tensor->nb[dim]);
        }
        return offset;
    };
    for (size_t flat = begin; flat < end; ++flat) {
        const float x = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(a->data) + offset_bytes(a, flat));
        const float up =
            *reinterpret_cast<const float*>(reinterpret_cast<const char*>(b->data) + offset_bytes(b, flat));
        const float x3 = x * x * x;
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
        float* out = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + offset_bytes(dst, flat));
        *out = gelu * up;
    }
}

void cb_gelu_tanh_unary(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!src || !dst || !src->data || !dst->data) return;

    const float* in = reinterpret_cast<const float*>(src->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const size_t size = ggml_nelements(src);

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    for (size_t i = begin; i < end; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        out[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
    }
}

void cb_apply_shared_scalar_gate(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                 const struct ggml_tensor* gate_logits_scalar, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !src || !gate_logits_scalar || !dst->data || !src->data || !gate_logits_scalar->data) {
        return;
    }

    const int hidden = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int scalar_elems = static_cast<int>(ggml_nelements(gate_logits_scalar));
    if (hidden <= 0 || tokens <= 0 || scalar_elems < tokens) {
        return;
    }

    const auto stable_sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };

    const float* src_base = reinterpret_cast<const float*>(src->data);
    float* dst_base = reinterpret_cast<float*>(dst->data);
    const float* gate_base = reinterpret_cast<const float*>(gate_logits_scalar->data);
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t gate_row_stride = static_cast<ptrdiff_t>(gate_logits_scalar->nb[1] / sizeof(float));

    const int token_begin = (tokens * ith) / nth;
    const int token_end = (tokens * (ith + 1)) / nth;
    for (int t = token_begin; t < token_end; ++t) {
        const float gate = stable_sigmoid(gate_base[static_cast<ptrdiff_t>(t) * gate_row_stride]);
        const float* src_row = src_base + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* dst_row = dst_base + static_cast<ptrdiff_t>(t) * dst_row_stride;
        densecore::simd::ScaleF32(dst_row, src_row, gate, static_cast<size_t>(hidden));
    }
}
