#include "densecore/backend/cpu_backend.h"
#include "densecore/backend/flash_attention.h"
#include "densecore/backend/hardware_topology.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "llm/attention/callback_ops.h"
#include "llm/attention/diagnostics.h"
#include "llm/attention/internal.h"
#include "llm/attention/kv_exec.h"
#include "llm/matmul/execution_policy.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/profile_types.h"
#include "llm/runtime/spin_wait.h"
#include "llm/runtime/work_context.h"
#include "runtime/runtime_env.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>

using densecore::llm::runtime::ResolveBackendRegistry;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;
using densecore::llm::runtime::ResolveHardwareTopology;
using densecore::llm::runtime::ResolveInferenceConfig;
using densecore::llm::runtime::SpinPause;

static bool IsPortableFlashParityCheckEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_CHECK_PORTABLE_FLASH_ATTN");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static float PortableFlashParityTolerance() {
    static const float tol = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_TOL");
        if (!env || env[0] == '\0') {
            return 1e-3f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-3f;
        }
        return parsed;
    }();
    return tol;
}

static bool ShouldRunPortableFlashParityCheck(int layer) {
    if (!IsPortableFlashParityCheckEnabled()) {
        return false;
    }

    static const int target_layer =
        densecore::env::ParseDiagnosticEnvInt("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_LAYER", -1);
    static std::atomic<int> remaining_budget{
        densecore::env::ParseDiagnosticPositiveEnvInt("DENSECORE_CHECK_PORTABLE_FLASH_ATTN_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
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

static bool IsPortableFlashReferenceFallbackForced() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_PORTABLE_FLASH_ATTN_FORCE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ShouldUsePortableFlashHeadSeqReferenceFallback(bool explicit_debug_reference) {
    return explicit_debug_reference;
}


static densecore::Tensor MakeHalAttentionTensorFromGgml(const struct ggml_tensor* tensor, int n_head, int seq_len,
                                                        int head_dim) {
    densecore::Tensor wrapped = densecore::Tensor::Make4D(const_cast<void*>(tensor->data), 1, n_head, seq_len, head_dim,
                                                          densecore::DType::F32, densecore::DeviceType::CPU);
    wrapped.stride[0] = static_cast<int64_t>(n_head) * seq_len * head_dim;
    wrapped.stride[1] = static_cast<int64_t>(tensor->nb[1] / sizeof(float));
    wrapped.stride[2] = static_cast<int64_t>(tensor->nb[2] / sizeof(float));
    wrapped.stride[3] = static_cast<int64_t>(tensor->nb[0] / sizeof(float));
    return wrapped;
}

void cb_flash_attention_hal_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    ScopedInferenceWorkContext callback_context(static_cast<InferenceWorkContext*>(userdata));
    if (nth <= 0 || !dst || !dst->src[0] || !dst->src[1] || !dst->src[2]) {
        return;
    }

    const auto* params = reinterpret_cast<const HalAttentionCustomParams*>(dst->op_params);
    if (!params) {
        return;
    }

    const struct ggml_tensor* q = dst->src[0];
    const struct ggml_tensor* k = dst->src[1];
    const struct ggml_tensor* v = dst->src[2];
    if (!q || !k || !v || !q->data || !k->data || !v->data || !dst->data) {
        return;
    }
    const bool direct_cpu_flash_path = params->data.preferred_device == densecore::DeviceType::CPU;
    if (!direct_cpu_flash_path && ith != 0) {
        return;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return;
    }
    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || k->ne[0] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
        v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0) {
        return;
    }
    if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2]) {
        return;
    }

    const HalAttentionTensorLayout layout = static_cast<HalAttentionTensorLayout>(params->data.layout);

    int head_dim = static_cast<int>(q->ne[0]);
    int seq_q = 0;
    int n_head = 0;
    int seq_kv = 0;
    int inferred_n_head_kv = 0;

    densecore::Tensor Q;
    densecore::Tensor K;
    densecore::Tensor V;
    densecore::Tensor O;

    if (layout == HalAttentionTensorLayout::HeadSeq) {
        if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || !ggml_is_contiguous(dst)) {
            return;
        }

        seq_q = static_cast<int>(q->ne[1]);
        n_head = static_cast<int>(q->ne[2]);
        seq_kv = static_cast<int>(k->ne[1]);
        inferred_n_head_kv = static_cast<int>(k->ne[2]);

        Q = densecore::Tensor::Make4D(const_cast<void*>(q->data), 1, n_head, seq_q, head_dim, densecore::DType::F32,
                                      densecore::DeviceType::CPU);
        K = densecore::Tensor::Make4D(const_cast<void*>(k->data), 1, inferred_n_head_kv, seq_kv, head_dim,
                                      densecore::DType::F32, densecore::DeviceType::CPU);
        V = densecore::Tensor::Make4D(const_cast<void*>(v->data), 1, inferred_n_head_kv, seq_kv, head_dim,
                                      densecore::DType::F32, densecore::DeviceType::CPU);
        O = densecore::Tensor::Make4D(dst->data, 1, n_head, seq_q, head_dim, densecore::DType::F32,
                                      densecore::DeviceType::CPU);
    } else {
        if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float) ||
            dst->nb[0] != sizeof(float)) {
            return;
        }

        n_head = static_cast<int>(q->ne[1]);
        seq_q = static_cast<int>(q->ne[2]);
        inferred_n_head_kv = static_cast<int>(k->ne[1]);
        seq_kv = static_cast<int>(k->ne[2]);
        if (seq_q != 1) {
            return;
        }

        Q = MakeHalAttentionTensorFromGgml(q, n_head, seq_q, head_dim);
        K = MakeHalAttentionTensorFromGgml(k, inferred_n_head_kv, seq_kv, head_dim);
        V = MakeHalAttentionTensorFromGgml(v, inferred_n_head_kv, seq_kv, head_dim);
        O = MakeHalAttentionTensorFromGgml(dst, n_head, seq_q, head_dim);
    }

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

    if (direct_cpu_flash_path) {
        densecore::FlashAttentionConfig config = densecore::AutoTuneFlashConfig(head_dim, seq_kv);
        config.scale = params->data.scale;
        config.logit_softcap = params->data.logit_softcap;
        config.causal = params->data.causal != 0;
        config.sliding_window = params->data.sliding_window;
        config.num_threads = std::max(1, nth);
        config.q_start_offset = std::max(0, params->data.q_start_offset);
        config.kv_start_offset = std::max(0, params->data.kv_start_offset);
        config.semantic_flags = params->data.semantic_flags;

        if (layout == HalAttentionTensorLayout::HeadSeq) {
            if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) ||
                !ggml_is_contiguous(dst)) {
                return;
            }

            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);

            const bool force_reference =
                ShouldUsePortableFlashHeadSeqReferenceFallback(IsPortableFlashReferenceFallbackForced());
            if (ith == 0) {
                if (auto* work_ctx = GetCurrentWorkContext()) {
                    auto& profile = (*GetInferenceWorkContextProfile(work_ctx));
                    profile.flash_attention_headseq_prefill_calls.fetch_add(1, std::memory_order_relaxed);
                    profile.flash_attention_last_nth.store(nth, std::memory_order_relaxed);
                    profile.flash_attention_last_active_threads.store(
                        force_reference ? 1 : std::max(1, std::min(nth, n_head)), std::memory_order_relaxed);
                    if (force_reference) {
                        profile.flash_attention_reference_calls.fetch_add(1, std::memory_order_relaxed);
                    } else if (CompiledWithX86Avx512ForFlashAttention()) {
                        profile.flash_attention_avx512_tiled_calls.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        profile.flash_attention_non_avx512_tiled_calls.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            if (force_reference) {
                if (ith != 0) {
                    return;
                }
                ComputeFlashAttentionReference(q_data, k_data, v_data, o_data, n_head, n_head_kv, seq_q, seq_kv,
                                               head_dim, params->data.scale, params->data.causal != 0,
                                               params->data.q_start_offset, params->data.kv_start_offset,
                                               params->data.sliding_window, params->data.logit_softcap);
            } else if (n_head == n_head_kv) {
                densecore::FlashAttentionBatched(q_data, k_data, v_data, o_data, 1, n_head, seq_q, seq_kv, head_dim,
                                                 config, ith, nth);
            } else {
                densecore::FlashAttentionGQA(q_data, k_data, v_data, o_data, 1, n_head, n_head_kv, seq_q, seq_kv,
                                             head_dim, config, ith, nth);
            }
        } else {
            if (seq_q != 1 || q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float) ||
                dst->nb[0] != sizeof(float)) {
                return;
            }

            static thread_local densecore::FlashAttentionScratch tl_scratch;
            tl_scratch.Resize(config.block_m, config.block_n, head_dim);

            const int total_work = n_head;
            const int work_per_thread = std::max(1, (total_work + nth - 1) / nth);
            const int work_start = ith * work_per_thread;
            const int work_end = std::min(total_work, work_start + work_per_thread);
            const int n_rep = n_head / n_head_kv;

            if (ith == 0) {
                if (auto* work_ctx = GetCurrentWorkContext()) {
                    auto& profile = (*GetInferenceWorkContextProfile(work_ctx));
                    profile.flash_attention_native_decode_calls.fetch_add(1, std::memory_order_relaxed);
                    profile.flash_attention_last_nth.store(nth, std::memory_order_relaxed);
                    profile.flash_attention_last_active_threads.store(std::max(1, std::min(nth, n_head)),
                                                                      std::memory_order_relaxed);
                }
            }

            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);

            const int64_t q_head_stride = static_cast<int64_t>(q->nb[1] / sizeof(float));
            const int64_t kv_head_stride = static_cast<int64_t>(k->nb[1] / sizeof(float));
            const int64_t kv_seq_stride = static_cast<int64_t>(k->nb[2] / sizeof(float));
            const int64_t v_head_stride = static_cast<int64_t>(v->nb[1] / sizeof(float));
            const int64_t v_seq_stride = static_cast<int64_t>(v->nb[2] / sizeof(float));
            const int64_t o_head_stride = static_cast<int64_t>(dst->nb[1] / sizeof(float));

            for (int h = work_start; h < work_end; ++h) {
                const int h_kv = h / n_rep;
                const float* q_ptr = q_data + static_cast<ptrdiff_t>(h) * q_head_stride;
                const float* k_ptr = k_data + static_cast<ptrdiff_t>(h_kv) * kv_head_stride;
                const float* v_ptr = v_data + static_cast<ptrdiff_t>(h_kv) * v_head_stride;
                float* o_ptr = o_data + static_cast<ptrdiff_t>(h) * o_head_stride;

                densecore::FlashAttentionSingleQueryStridedKV(q_ptr, k_ptr, kv_seq_stride, v_ptr, v_seq_stride, o_ptr,
                                                              seq_kv, head_dim, config, tl_scratch);
            }
        }
        if (layout == HalAttentionTensorLayout::HeadSeq && ShouldRunPortableFlashParityCheck(params->data.layer) &&
            ith == 0) {
            const float* q_data = reinterpret_cast<const float*>(q->data);
            const float* k_data = reinterpret_cast<const float*>(k->data);
            const float* v_data = reinterpret_cast<const float*>(v->data);
            float* o_data = reinterpret_cast<float*>(dst->data);
            std::vector<float> ref(static_cast<size_t>(n_head) * seq_q * head_dim, 0.0f);
            ComputeFlashAttentionReference(q_data, k_data, v_data, ref.data(), n_head, n_head_kv, seq_q, seq_kv,
                                           head_dim, params->data.scale, params->data.causal != 0,
                                           params->data.q_start_offset, params->data.kv_start_offset,
                                           params->data.sliding_window, params->data.logit_softcap);

            float max_abs = 0.0f;
            int max_idx = -1;
            for (size_t i = 0; i < ref.size(); ++i) {
                const float diff = std::fabs(o_data[i] - ref[i]);
                if (diff > max_abs) {
                    max_abs = diff;
                    max_idx = static_cast<int>(i);
                }
            }
            if (max_abs > PortableFlashParityTolerance()) {
                const float got_val = (max_idx >= 0) ? o_data[max_idx] : 0.0f;
                const float ref_val = (max_idx >= 0) ? ref[static_cast<size_t>(max_idx)] : 0.0f;
                fprintf(stderr,
                        "[PortableFlashParity] FAIL layer=%d seq_q=%d seq_kv=%d n_head=%d n_head_kv=%d head_dim=%d "
                        "max_abs=%.6f idx=%d got=%.6f ref=%.6f\n",
                        params->data.layer, seq_q, seq_kv, n_head, n_head_kv, head_dim, max_abs, max_idx, got_val,
                        ref_val);
            }
        }
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

    try {
        backend->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv,
                                params->data.sliding_window, params->data.logit_softcap, params->data.semantic_flags);
    } catch (...) {
        densecore::ComputeBackend* cpu = registry.Get(densecore::DeviceType::CPU);
        if (cpu && cpu != backend) {
            cpu->FlashAttention(Q, K, V, &O, params->data.scale, params->data.causal != 0, n_head_kv,
                                params->data.sliding_window, params->data.logit_softcap, params->data.semantic_flags);
        }
    }

    if (layout == HalAttentionTensorLayout::HeadSeq && ShouldRunPortableFlashParityCheck(params->data.layer)) {
        std::vector<float> ref(static_cast<size_t>(n_head) * seq_q * head_dim, 0.0f);
        ComputeFlashAttentionReference(reinterpret_cast<const float*>(q->data), reinterpret_cast<const float*>(k->data),
                                       reinterpret_cast<const float*>(v->data), ref.data(), n_head, n_head_kv, seq_q,
                                       seq_kv, head_dim, params->data.scale, params->data.causal != 0,
                                       params->data.q_start_offset, params->data.kv_start_offset,
                                       params->data.sliding_window, params->data.logit_softcap);

        const float* got = reinterpret_cast<const float*>(dst->data);
        float max_abs = 0.0f;
        int max_idx = -1;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float diff = std::fabs(got[i] - ref[i]);
            if (diff > max_abs) {
                max_abs = diff;
                max_idx = static_cast<int>(i);
            }
        }
        if (max_abs > PortableFlashParityTolerance()) {
            const float got_val = (max_idx >= 0) ? got[max_idx] : 0.0f;
            const float ref_val = (max_idx >= 0) ? ref[static_cast<size_t>(max_idx)] : 0.0f;
            fprintf(stderr,
                    "[PortableFlashParity] FAIL layer=%d seq_q=%d seq_kv=%d n_head=%d n_head_kv=%d head_dim=%d "
                    "max_abs=%.6f idx=%d got=%.6f ref=%.6f\n",
                    params->data.layer, seq_q, seq_kv, n_head, n_head_kv, head_dim, max_abs, max_idx, got_val, ref_val);
        }
    }
}
