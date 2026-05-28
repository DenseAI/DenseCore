#include "llm/attention/exec.h"

#include "ggml.h"
#include "llm/attention/internal.h"
#include "models/model_inference_policy.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif

namespace {

std::atomic<uint64_t> g_shared_prefill_flash_mask_builds{0};

struct AttentionTensorTraceUserData {
    int layer = -1;
    const char* stage = nullptr;
};

bool IsGemma4ParityTraceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_GEMMA4_PARITY_TRACE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void RecordGemma4GgmlAttentionFallback(TransformerModel* model) {
    if (!model || !model->arch_flags.is_gemma4 || !IsQwen36ProfilingEnabled()) return;
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) return;
    ::RecordGemma4GgmlAttentionFallback(work_ctx);
}

int Gemma4ParityTraceLayer() {
    static const int layer = []() {
        const char* env = std::getenv("DENSECORE_GEMMA4_PARITY_TRACE_LAYER");
        return (env && env[0] != '\0') ? std::atoi(env) : -1;
    }();
    return layer;
}

void cb_attention_tensor_trace(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                               void* userdata) {
    (void)nth;
    if (dst && src && dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (ith != 0 || !src || !src->data || src->type != GGML_TYPE_F32) {
        return;
    }
    const auto* ud = static_cast<const AttentionTensorTraceUserData*>(userdata);
    const float* data = static_cast<const float*>(src->data);
    const int64_t n = ggml_nelements(src);
    int64_t nan_ct = 0;
    int64_t inf_ct = 0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum_sq = 0.0;
    int64_t finite_ct = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++nan_ct;
            continue;
        }
        if (!std::isfinite(v)) {
            ++inf_ct;
            continue;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        ++finite_ct;
    }
    if (!std::isfinite(mn)) mn = 0.0f;
    if (!std::isfinite(mx)) mx = 0.0f;
    std::fprintf(stderr,
                 "[GEMMA4_ATTN_TRACE] layer=%d stage=%s shape=[%ld,%ld,%ld,%ld] total=%ld nan=%ld inf=%ld min=%.8g "
                 "max=%.8g rms=%.8g\n",
                 ud ? ud->layer : -1, ud && ud->stage ? ud->stage : "unknown", (long)src->ne[0], (long)src->ne[1],
                 (long)src->ne[2], (long)src->ne[3], (long)n, (long)nan_ct, (long)inf_ct, mn, mx,
                 finite_ct > 0 ? std::sqrt(sum_sq / static_cast<double>(finite_ct)) : 0.0);
}

struct ggml_tensor* MaybeTraceGemma4AttentionTensor(struct ggml_context* ctx_c, TransformerModel* model,
                                                    struct ggml_tensor* tensor, int layer, const char* stage) {
    if (!ctx_c || !tensor || !model || !model->arch_flags.is_gemma4 || !IsGemma4ParityTraceEnabled()) {
        return tensor;
    }
    const int target_layer = Gemma4ParityTraceLayer();
    if (target_layer >= 0 && target_layer != layer) {
        return tensor;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(AttentionTensorTraceUserData));
    if (!storage || !storage->data) {
        return tensor;
    }
    auto* ud = reinterpret_cast<AttentionTensorTraceUserData*>(storage->data);
    *ud = AttentionTensorTraceUserData{layer, stage};
    return ggml_map_custom1(ctx_c, tensor, cb_attention_tensor_trace, 1, ud);
}

float ResolveAttentionScaleForRuntime(int head_dim_q, bool use_explicit_attention_scale) {
    if (use_explicit_attention_scale) {
        return 1.0f;
    }
    return head_dim_q > 0 ? 1.0f / sqrtf(static_cast<float>(head_dim_q)) : 1.0f;
}

struct ggml_tensor* RepeatGroupedKvHeads(struct ggml_context* ctx_c, struct ggml_tensor* tensor, int n_head) {
    if (!tensor || n_head <= 0) {
        return tensor;
    }
    const int n_head_kv = static_cast<int>(tensor->ne[2]);
    if (n_head_kv <= 0 || n_head == n_head_kv) {
        return tensor;
    }
    if ((n_head % n_head_kv) != 0) {
        return tensor;
    }

    const int n_rep = n_head / n_head_kv;
    struct ggml_tensor* as_4d = ggml_reshape_4d(ctx_c, tensor, tensor->ne[0], tensor->ne[1], 1, tensor->ne[2]);
    struct ggml_tensor* repeated = ggml_repeat_4d(ctx_c, as_4d, tensor->ne[0], tensor->ne[1], n_rep, tensor->ne[2]);
    return ggml_reshape_3d(ctx_c, repeated, tensor->ne[0], tensor->ne[1], n_head);
}

struct ggml_tensor*
BuildOrReuseSharedPrefillFlashMask(struct ggml_context* ctx_c, int n_total_tokens, int n_tokens, int n_past_val,
                                   int attn_query_base_pos, int fast_attn_sliding_window, bool decode_only_batch,
                                   struct ggml_tensor** shared_prefill_flash_mask, int* shared_prefill_mask_n_total,
                                   int* shared_prefill_mask_n_padded, int* shared_prefill_mask_n,
                                   int* shared_prefill_mask_n_past, int* shared_prefill_mask_sliding_window) {
    const int n_padded = (n_tokens + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);
    if (*shared_prefill_flash_mask && *shared_prefill_mask_n_total == n_total_tokens &&
        *shared_prefill_mask_n_padded == n_padded && *shared_prefill_mask_n == n_tokens &&
        *shared_prefill_mask_n_past == n_past_val && *shared_prefill_mask_sliding_window == fast_attn_sliding_window) {
        return *shared_prefill_flash_mask;
    }

    struct ggml_tensor* mask = densecore::llm::attention::BuildAttentionMaskTensor(
        ctx_c, n_total_tokens, n_tokens, attn_query_base_pos, fast_attn_sliding_window, n_padded);
    if (n_tokens > 1 && !decode_only_batch) {
        *shared_prefill_flash_mask = mask;
        *shared_prefill_mask_n_total = n_total_tokens;
        *shared_prefill_mask_n_padded = n_padded;
        *shared_prefill_mask_n = n_tokens;
        *shared_prefill_mask_n_past = n_past_val;
        *shared_prefill_mask_sliding_window = fast_attn_sliding_window;
        g_shared_prefill_flash_mask_builds.fetch_add(1, std::memory_order_relaxed);
    }
    return mask;
}

}  // namespace

struct ggml_tensor* ggml_paged_attention_decode(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                                struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                                PagedAttentionUserData* userdata);
struct ggml_tensor* ggml_flash_attention_hal(struct ggml_context* ctx, struct ggml_tensor* Q, struct ggml_tensor* K,
                                             struct ggml_tensor* V, float scale, bool causal, int n_head_kv,
                                             int sliding_window = -1, float logit_softcap = 0.0f,
                                             uint32_t semantic_flags = 0u, int layer = -1,
                                             densecore::DeviceType preferred_device = densecore::DeviceType::CPU,
                                             HalAttentionTensorLayout layout = HalAttentionTensorLayout::HeadSeq,
                                             int q_start_offset = 0, int kv_start_offset = 0);

struct ggml_tensor* ExecutePagedDecodeAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                                    PagedKVCache* cache, struct ggml_tensor* Qcur,
                                                    struct ggml_tensor* Kcur, struct ggml_tensor* Vcur, int il,
                                                    int kv_cache_layer, bool gemma4_shared_kv_layer,
                                                    bool gemma4_shared_kv_source_layer, int head_dim_q, int head_dim_v,
                                                    int n_head, int n_head_kv, int n_total_tokens,
                                                    float fast_attn_logit_softcap, bool use_explicit_attention_scale) {
    struct ggml_tensor* Q_decode = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
    struct ggml_tensor* K_decode = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
    struct ggml_tensor* V_decode = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);
    const int paged_attn_sliding_window =
        (model->arch_flags.is_gemma4 && densecore::models::IsGemma4SlidingLayer(model, il) &&
         model->gemma4_sliding_window > 0)
            ? model->gemma4_sliding_window
            : -1;
    const float paged_attn_scale = ResolveAttentionScaleForRuntime(head_dim_q, use_explicit_attention_scale);

    PagedAttentionUserData* ud = GetPagedAttentionUserData();
    ud->cache = cache;
    ud->layer = il;
    ud->read_layer = kv_cache_layer;
    ud->write_layer = kv_cache_layer;
    ud->head_dim = head_dim_q;
    ud->v_head_dim = head_dim_v;
    ud->n_head = n_head;
    ud->n_head_kv = n_head_kv;
    ud->write_current_kv = !gemma4_shared_kv_layer;
    ud->force_full_history = gemma4_shared_kv_layer || gemma4_shared_kv_source_layer;
    ud->is_gemma4 = model && model->arch_flags.is_gemma4;
    ud->sliding_window = paged_attn_sliding_window;
    ud->attention_scale = paged_attn_scale;
    ud->logit_softcap = fast_attn_logit_softcap;
    ud->epoch_started.store(0, std::memory_order_relaxed);
    ud->epoch_done.store(0, std::memory_order_relaxed);
    ud->kv_writers_done.store(0, std::memory_order_relaxed);
    ud->shared_block_ptrs_ready.store(0, std::memory_order_relaxed);
    return ggml_paged_attention_decode(ctx_c, Q_decode, K_decode, V_decode, ud);
}

struct ggml_tensor* ExecuteHalAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                            struct ggml_tensor* Qcur, struct ggml_tensor* K, struct ggml_tensor* V,
                                            int il, int n_tokens, int head_dim_q, int n_head_kv,
                                            int fast_attn_sliding_window, float fast_attn_logit_softcap,
                                            uint32_t fast_attn_semantic_flags,
                                            densecore::DeviceType preferred_attention_device,
                                            bool use_explicit_attention_scale) {
    struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
    struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
    struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
    Q_hal = ggml_cont(ctx_c, Q_hal);
    K_hal = ggml_cont(ctx_c, K_hal);
    V_hal = ggml_cont(ctx_c, V_hal);

    const float scale = ResolveAttentionScaleForRuntime(head_dim_q, use_explicit_attention_scale);
    const bool hal_causal = (n_tokens > 1);
    struct ggml_tensor* KQV =
        ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv, fast_attn_sliding_window,
                                 fast_attn_logit_softcap, fast_attn_semantic_flags, il, preferred_attention_device);
    return ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
}

struct ggml_tensor* ExecutePortableCpuFlashAttentionPath(
    struct ggml_context* ctx_c, TransformerModel* model, struct ggml_tensor* Qcur, struct ggml_tensor* K,
    struct ggml_tensor* V, int il, int n_tokens, int head_dim_q, int head_dim_kv, int head_dim_v, int n_head_kv,
    int attn_query_base_pos, int attn_kv_start_pos, int fast_attn_sliding_window, float fast_attn_logit_softcap,
    uint32_t fast_attn_semantic_flags, bool use_explicit_attention_scale, bool native_decode_layout) {
    const float scale = ResolveAttentionScaleForRuntime(head_dim_q, use_explicit_attention_scale);
    const bool hal_causal = (n_tokens > 1);
    const int flash_q_start_offset = attn_query_base_pos;
    if (native_decode_layout) {
        return ggml_flash_attention_hal(ctx_c, Qcur, K, V, scale, false, n_head_kv, fast_attn_sliding_window,
                                        fast_attn_logit_softcap, fast_attn_semantic_flags, il,
                                        densecore::DeviceType::CPU, HalAttentionTensorLayout::GgmlDimHeadSeq,
                                        flash_q_start_offset, attn_kv_start_pos);
    }

    struct ggml_tensor* Q_hal = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
    struct ggml_tensor* K_hal = ggml_permute(ctx_c, K, 0, 2, 1, 3);
    struct ggml_tensor* V_hal = ggml_permute(ctx_c, V, 0, 2, 1, 3);
    Q_hal = ggml_cont(ctx_c, Q_hal);
    K_hal = ggml_cont(ctx_c, K_hal);
    V_hal = ggml_cont(ctx_c, V_hal);

    struct ggml_tensor* KQV =
        ggml_flash_attention_hal(ctx_c, Q_hal, K_hal, V_hal, scale, hal_causal, n_head_kv, fast_attn_sliding_window,
                                 fast_attn_logit_softcap, fast_attn_semantic_flags, il, densecore::DeviceType::CPU,
                                 HalAttentionTensorLayout::HeadSeq, flash_q_start_offset, attn_kv_start_pos);
    return ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
}

struct ggml_tensor* ExecuteNativeFlashAttentionPath(
    struct ggml_context* ctx_c, TransformerModel* model, struct ggml_tensor* Qcur, struct ggml_tensor* K,
    struct ggml_tensor* V, int n_tokens, int n_past_val, int n_total_tokens, int head_dim_q, int n_head_kv,
    int attn_query_base_pos, int fast_attn_sliding_window, bool decode_only_batch, bool use_explicit_attention_scale,
    struct ggml_tensor** shared_prefill_flash_mask, int* shared_prefill_mask_n_total, int* shared_prefill_mask_n_padded,
    int* shared_prefill_mask_n, int* shared_prefill_mask_n_past, int* shared_prefill_mask_sliding_window) {
    RecordGemma4GgmlAttentionFallback(model);
    struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
    struct ggml_tensor* K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
    struct ggml_tensor* V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

    struct ggml_tensor* KQ_mask = BuildOrReuseSharedPrefillFlashMask(
        ctx_c, n_total_tokens, n_tokens, n_past_val, attn_query_base_pos, fast_attn_sliding_window, decode_only_batch,
        shared_prefill_flash_mask, shared_prefill_mask_n_total, shared_prefill_mask_n_padded, shared_prefill_mask_n,
        shared_prefill_mask_n_past, shared_prefill_mask_sliding_window);

    Q = ggml_cont(ctx_c, Q);
    K_fa = ggml_cont(ctx_c, K_fa);
    V_fa = ggml_cont(ctx_c, V_fa);

    const float scale = ResolveAttentionScaleForRuntime(head_dim_q, use_explicit_attention_scale);
    struct ggml_tensor* KQV = ggml_flash_attn_ext(ctx_c, Q, K_fa, V_fa, KQ_mask, scale, 0.0f, 0.0f);
    return ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
}

struct ggml_tensor* ExecuteStandardAttentionPath(struct ggml_context* ctx_c, TransformerModel* model,
                                                 struct ggml_tensor* Qcur, struct ggml_tensor* K, struct ggml_tensor* V,
                                                 int il, int n_tokens, int n_past_val, int n_total_tokens, int n_head,
                                                 int n_head_kv, int head_dim_q, int fast_attn_sliding_window,
                                                 int attn_query_base_pos, bool use_explicit_attention_scale) {
    RecordGemma4GgmlAttentionFallback(model);
    const float scale = ResolveAttentionScaleForRuntime(head_dim_q, use_explicit_attention_scale);

    const bool skip_attn_cont = (n_tokens > 1) && densecore::llm::attention::IsPrefillAttentionSkipContEnabled();
    struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);
    struct ggml_tensor* K_att = ggml_permute(ctx_c, K, 0, 2, 1, 3);
    struct ggml_tensor* V_att = ggml_permute(ctx_c, V, 0, 2, 1, 3);

    K_att = ggml_cont(ctx_c, K_att);
    K_att = RepeatGroupedKvHeads(ctx_c, K_att, n_head);
    if (!skip_attn_cont) {
        Q = ggml_cont(ctx_c, Q);
        V_att = ggml_cont(ctx_c, V_att);
    }
    Q = MaybeTraceGemma4AttentionTensor(ctx_c, model, Q, il, "standard_q");
    K_att = MaybeTraceGemma4AttentionTensor(ctx_c, model, K_att, il, "standard_k");

    const float gemma4_attention_softcap = densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(model);
    if (model->arch_flags.is_gemma4 && gemma4_attention_softcap > 0.0f) {
        const float inv_softcap = 1.0f / gemma4_attention_softcap;
        struct ggml_tensor* Q_scaled = ggml_scale(ctx_c, Q, inv_softcap);
        Q_scaled = MaybeTraceGemma4AttentionTensor(ctx_c, model, Q_scaled, il, "standard_q_scaled");
        struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_att, Q_scaled);
        KQ = MaybeTraceGemma4AttentionTensor(ctx_c, model, KQ, il, "standard_kq_presoftcap");
        KQ = ggml_tanh(ctx_c, KQ);
        KQ = ggml_scale(ctx_c, KQ, gemma4_attention_softcap);
        KQ = ggml_scale(ctx_c, KQ, scale);
        KQ = MaybeTraceGemma4AttentionTensor(ctx_c, model, KQ, il, "standard_kq_softcapped");
        if (densecore::llm::attention::ShouldBuildExplicitStandardAttentionMask(n_tokens, fast_attn_sliding_window)) {
            if (fast_attn_sliding_window >= 0) {
                struct ggml_tensor* KQ_mask = densecore::llm::attention::BuildAttentionMaskTensor(
                    ctx_c, n_total_tokens, n_tokens, attn_query_base_pos, fast_attn_sliding_window);
                KQ = ggml_add(ctx_c, KQ, KQ_mask);
            } else {
                KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
            }
        }
        KQ = MaybeTraceGemma4AttentionTensor(ctx_c, model, KQ, il, "standard_kq_masked");
        KQ = ggml_soft_max(ctx_c, KQ);
        KQ = MaybeTraceGemma4AttentionTensor(ctx_c, model, KQ, il, "standard_kq_softmax");
        struct ggml_tensor* V_t = ggml_permute(ctx_c, V_att, 1, 0, 2, 3);
        V_t = ggml_cont(ctx_c, V_t);
        V_t = RepeatGroupedKvHeads(ctx_c, V_t, n_head);
        if (!ggml_is_contiguous(V_t)) {
            V_t = ggml_cont(ctx_c, V_t);
        }
        struct ggml_tensor* KQV = ggml_mul_mat(ctx_c, V_t, KQ);
        KQV = MaybeTraceGemma4AttentionTensor(ctx_c, model, KQV, il, "standard_kqv");
        return ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
    }

    struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_att, Q);
    KQ = ggml_scale(ctx_c, KQ, scale);

    if (densecore::llm::attention::ShouldBuildExplicitStandardAttentionMask(n_tokens, fast_attn_sliding_window)) {
        if (fast_attn_sliding_window >= 0) {
            struct ggml_tensor* KQ_mask = densecore::llm::attention::BuildAttentionMaskTensor(
                ctx_c, n_total_tokens, n_tokens, attn_query_base_pos, fast_attn_sliding_window);
            KQ = ggml_add(ctx_c, KQ, KQ_mask);
        } else {
            KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
        }
    }

    KQ = ggml_soft_max(ctx_c, KQ);
    struct ggml_tensor* V_t = ggml_permute(ctx_c, V_att, 1, 0, 2, 3);
    V_t = ggml_cont(ctx_c, V_t);
    V_t = RepeatGroupedKvHeads(ctx_c, V_t, n_head);
    if (!ggml_is_contiguous(V_t)) {
        V_t = ggml_cont(ctx_c, V_t);
    }
    struct ggml_tensor* KQV = ggml_mul_mat(ctx_c, V_t, KQ);
    return ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
}

#ifdef DENSECORE_TEST_BUILD
namespace densecore::llm::attention::testing {

void ResetSharedPrefillFlashMaskBuildsForTest() {
    g_shared_prefill_flash_mask_builds.store(0, std::memory_order_relaxed);
}

uint64_t GetSharedPrefillFlashMaskBuildsForTest() {
    return g_shared_prefill_flash_mask_builds.load(std::memory_order_relaxed);
}

void ExerciseSharedPrefillFlashMaskBuildForTest(bool native_flash_selected, int n_total_tokens, int n_tokens,
                                                int attn_query_base_pos, bool decode_only_batch,
                                                int fast_attn_sliding_window) {
    if (!native_flash_selected) {
        return;
    }

    struct ggml_init_params params {};
    params.mem_size = 256 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return;
    }

    struct ggml_tensor* shared_prefill_flash_mask = nullptr;
    int shared_prefill_mask_n_total = -1;
    int shared_prefill_mask_n_padded = -1;
    int shared_prefill_mask_n = -1;
    int shared_prefill_mask_n_past = -1;
    int shared_prefill_mask_sliding_window = -1;
    (void)BuildOrReuseSharedPrefillFlashMask(
        ctx, n_total_tokens, n_tokens, /*n_past_val=*/std::max(0, n_total_tokens - n_tokens), attn_query_base_pos,
        fast_attn_sliding_window, decode_only_batch, &shared_prefill_flash_mask, &shared_prefill_mask_n_total,
        &shared_prefill_mask_n_padded, &shared_prefill_mask_n, &shared_prefill_mask_n_past,
        &shared_prefill_mask_sliding_window);
    ggml_free(ctx);
}

}  // namespace densecore::llm::attention::testing
#endif
