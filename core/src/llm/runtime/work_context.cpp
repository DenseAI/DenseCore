#include "llm/runtime/work_context.h"

namespace {
thread_local InferenceWorkContext* current_work_context = nullptr;
thread_local InferenceWorkContext* dispatch_work_context = nullptr;
}  // namespace

void SetCurrentWorkContext(InferenceWorkContext* context) {
    current_work_context = context;
    dispatch_work_context = context;
}

InferenceWorkContext* GetCurrentWorkContext() {
    return current_work_context;
}

InferenceWorkContext* GetDispatchWorkContext() {
    return dispatch_work_context;
}

void SetCallbackWorkContext(InferenceWorkContext* context, InferenceWorkContext* dispatch_context) {
    current_work_context = context;
    dispatch_work_context = dispatch_context;
}

#include "densecore/exceptions.h"
#include "llm/graph/construction_ops.h"
#include "llm/matmul/execution_policy.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/work_context_internal.h"
#include <algorithm>
#include <cmath>
#include <limits>


Qwen36ProfileCounters* GetInferenceWorkContextProfile(InferenceWorkContext* ctx) {
    return ctx ? &ctx->qwen36_profile : nullptr;
}


MatmulWorkState& GetMatmulWorkState(InferenceWorkContext* ctx) {
    return ctx->matmul;
}
const MatmulWorkState& GetMatmulWorkState(const InferenceWorkContext* ctx) {
    return ctx->matmul;
}
const BatchSpec* GetInferenceWorkContextBatch(const InferenceWorkContext* ctx) {
    return ctx ? ctx->batch : nullptr;
}
void BindInferenceWorkContextBatch(InferenceWorkContext* context, const BatchSpec* batch) {
    if (context) context->batch = batch;
}
void SetInferenceWorkContextPhase(InferenceWorkContext* context, InferenceExecutionPhase phase) {
    if (context) context->phase = phase;
}
InferenceExecutionPhase GetInferenceWorkContextPhase(const InferenceWorkContext* ctx) {
    return ctx ? ctx->phase : InferenceExecutionPhase::Unknown;
}
ModelVariant GetInferenceWorkContextModelVariant(const InferenceWorkContext* ctx) {
    return ctx ? ctx->model_variant : ModelVariant::UNKNOWN;
}

uint64_t GetInferenceWorkContextExecutionGeneration(const InferenceWorkContext* ctx) {
    return ctx ? ctx->execution_generation : 0;
}

static void ResetQuantizedActivationCache(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->matmul.qact_generation = 0;
    ctx->matmul.qact_tensor = nullptr;
    ctx->matmul.qact_source = nullptr;
    ctx->matmul.qact_len = 0;
    ctx->matmul.qact_type = GGML_TYPE_COUNT;
    ctx->matmul.qact_bytes = 0;
    ctx->matmul.qact_slot_id = -1;
    ctx->matmul.qact_token_pos = std::numeric_limits<int64_t>::min();
    ctx->matmul.qact_buffer.clear();
    for (auto& slot : ctx->matmul.qact_extra_slots) {
        slot.generation = 0;
        slot.tensor = nullptr;
        slot.source = nullptr;
        slot.len = 0;
        slot.type = GGML_TYPE_COUNT;
        slot.bytes = 0;
        slot.slot_id = -1;
        slot.token_pos = std::numeric_limits<int64_t>::min();
        slot.buffer.clear();
    }
    ctx->matmul.qact_extra_next_slot = 0;
}


InferenceWorkContext* CreateInferenceWorkContext() {
    return new InferenceWorkContext();
}

densecore::CpuExecutionOptions densecore::llm::runtime::ResolveCpuExecutionOptions(InferenceWorkContext* context) {
    if (!context) return {};
    const auto* dispatch = GetDispatchWorkContext();
    if (dispatch != context) dispatch = nullptr;
    return {dispatch ? static_cast<densecore::CpuExecutionPhase>(dispatch->phase)
                     : densecore::CpuExecutionPhase::Unknown,
            dispatch != nullptr,
            dispatch && dispatch->model_variant == ModelVariant::LFM2MOE,
            &context->cpu_telemetry,
            true,
            std::this_thread::get_id()};
}

void DestroyInferenceWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) {
        return;
    }
    if (GetCurrentWorkContext() == ctx) {
        SetCurrentWorkContext(nullptr);
    }
    delete ctx;
}

void ResetInferenceWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) return;
    static std::atomic<uint64_t> generation_counter{1};
    ctx->batch = nullptr;
    ctx->phase = InferenceExecutionPhase::Unknown;
    ctx->model_variant = ModelVariant::UNKNOWN;
    ctx->graph_build_no_alloc = false;
    ctx->matmul.lfm2_greedy_lm_head_argmax_allowed = false;
    ctx->matmul.lfm2_greedy_lm_head_argmax_reject_reason = LFM2GreedyLMHeadArgmaxRejectReason::None;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repetition_penalty = 1.0f;
    ctx->matmul.lfm2_greedy_lm_head_argmax_final_logit_softcap = 0.0f;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.clear();
    ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.clear();
    ctx->matmul.lfm2_greedy_lm_head_argmax_generation = 0;
    ctx->matmul.lfm2_greedy_lm_head_argmax_token = -1;
    ctx->matmul.lfm2_greedy_lm_head_argmax_value = -std::numeric_limits<float>::infinity();
    ctx->matmul.lfm2_greedy_lm_head_argmax_vocab_size = 0;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_ptr = nullptr;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_vocab_size = 0;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token = -1;
    ctx->execution_generation = generation_counter.fetch_add(1, std::memory_order_relaxed);
    ResetQwen36Profile(ctx);
    ctx->gemma4_shared_kv_states.clear();
    ctx->qkv_index = 0;
    ctx->add_rmsnorm_index = 0;
    ctx->matmul.gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ResetQuantizedActivationCache(ctx);
    ctx->matmul.q8_gemm_packed_generation = 0;
    ctx->matmul.q8_gemm_packed_tensor = nullptr;
    ctx->matmul.q8_gemm_packed_source = nullptr;
    ctx->matmul.q8_gemm_packed_rows = 0;
    ctx->matmul.q8_gemm_packed_cols = 0;
    ctx->matmul.q8_gemm_packed_bytes = 0;
    ctx->matmul.q8_gemm_packed_token_pos = std::numeric_limits<int64_t>::min();
    ctx->matmul.q8_gemm_packed_buffer.clear();
    ctx->matmul.q8_gemm_packed_stamp.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(ctx->matmul.q4k_repacked_gemv_request_mutex);
        ctx->matmul.q4k_repacked_gemv_repack_counts.clear();
    }
    ctx->matmul.gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->matmul.gemv_userdata_index = 0;
    ctx->matmul.gemv_batched_userdata_index = 0;
    ctx->paged_attention_userdata_index = 0;
    ctx->paged_attention_shared_k_block_ptrs.clear();
    ctx->paged_attention_shared_v_block_ptrs.clear();
    ctx->ssm_conv1d_index = 0;
    ctx->lfm2_shortconv_index = 0;
    ctx->projection_reference_index = 0;
    ctx->rmsnorm_reference_index = 0;
    ctx->attention_core_reference_index = 0;
    ctx->ssm_qwen35_delta_index = 0;
}

void ResetCachedDecodeGraphWorkContext(InferenceWorkContext* ctx) {
    if (!ctx) return;
    static std::atomic<uint64_t> generation_counter{1000000000ull};
    ctx->phase = InferenceExecutionPhase::Decode;
    ctx->graph_build_no_alloc = false;
    ctx->matmul.lfm2_greedy_lm_head_argmax_allowed = false;
    ctx->matmul.lfm2_greedy_lm_head_argmax_reject_reason = LFM2GreedyLMHeadArgmaxRejectReason::None;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repetition_penalty = 1.0f;
    ctx->matmul.lfm2_greedy_lm_head_argmax_final_logit_softcap = 0.0f;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.clear();
    ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.clear();
    ctx->matmul.lfm2_greedy_lm_head_argmax_generation = 0;
    ctx->matmul.lfm2_greedy_lm_head_argmax_token = -1;
    ctx->matmul.lfm2_greedy_lm_head_argmax_value = -std::numeric_limits<float>::infinity();
    ctx->matmul.lfm2_greedy_lm_head_argmax_vocab_size = 0;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_ptr = nullptr;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_vocab_size = 0;
    ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token = -1;
    ctx->execution_generation = generation_counter.fetch_add(1, std::memory_order_relaxed);
    ResetQwen36Profile(ctx);
    ctx->paged_attention_shared_k_block_ptrs.clear();
    ctx->paged_attention_shared_v_block_ptrs.clear();
    ctx->matmul.gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ResetQuantizedActivationCache(ctx);
    ctx->matmul.q8_gemm_packed_generation = 0;
    ctx->matmul.q8_gemm_packed_tensor = nullptr;
    ctx->matmul.q8_gemm_packed_source = nullptr;
    ctx->matmul.q8_gemm_packed_rows = 0;
    ctx->matmul.q8_gemm_packed_cols = 0;
    ctx->matmul.q8_gemm_packed_bytes = 0;
    ctx->matmul.q8_gemm_packed_token_pos = std::numeric_limits<int64_t>::min();
    ctx->matmul.q8_gemm_packed_buffer.clear();
    ctx->matmul.q8_gemm_packed_stamp.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(ctx->matmul.q4k_repacked_gemv_request_mutex);
        ctx->matmul.q4k_repacked_gemv_repack_counts.clear();
    }
    ctx->matmul.gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
}

void ResetCachedPrefillGraphWorkContext(InferenceWorkContext* ctx) {
    ResetCachedDecodeGraphWorkContext(ctx);
    if (!ctx) return;
    ctx->phase = InferenceExecutionPhase::Prefill;
}

void SetInferenceWorkContextGraphBuildNoAlloc(InferenceWorkContext* ctx, bool no_alloc) {
    if (!ctx) {
        return;
    }
    ctx->graph_build_no_alloc = no_alloc;
}

bool IsCurrentGraphBuildNoAlloc() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    return ctx && ctx->graph_build_no_alloc;
}

void SetInferenceWorkContextModelVariant(InferenceWorkContext* ctx, ModelVariant variant) {
    if (!ctx) {
        return;
    }
    ctx->model_variant = variant;
}

ModelVariant GetCurrentInferenceWorkContextModelVariant() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    return ctx ? ctx->model_variant : ModelVariant::UNKNOWN;
}

void SetInferenceWorkContextLFM2GreedyLMHeadArgmaxSampling(InferenceWorkContext* ctx, bool allowed,
                                                           LFM2GreedyLMHeadArgmaxRejectReason reject_reason,
                                                           float repetition_penalty, float final_logit_softcap,
                                                           const std::vector<int>* token_history,
                                                           const std::vector<int>* disallowed_token_ids) {
    if (!ctx) {
        return;
    }
    ctx->matmul.lfm2_greedy_lm_head_argmax_allowed = allowed;
    ctx->matmul.lfm2_greedy_lm_head_argmax_reject_reason = reject_reason;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repetition_penalty = repetition_penalty;
    ctx->matmul.lfm2_greedy_lm_head_argmax_final_logit_softcap = final_logit_softcap;
    ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.clear();
    ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.clear();
    if (!allowed) {
        ctx->matmul.lfm2_greedy_lm_head_argmax_generation = 0;
        ctx->matmul.lfm2_greedy_lm_head_argmax_token = -1;
        ctx->matmul.lfm2_greedy_lm_head_argmax_value = -std::numeric_limits<float>::infinity();
        ctx->matmul.lfm2_greedy_lm_head_argmax_vocab_size = 0;
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_ptr = nullptr;
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_vocab_size = 0;
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token = -1;
        return;
    }
    if (allowed && token_history && !token_history->empty()) {
        ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.assign(token_history->begin(), token_history->end());
        std::sort(ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.begin(),
                  ctx->matmul.lfm2_greedy_lm_head_argmax_repeated_tokens.end());
    }
    if (allowed && disallowed_token_ids && !disallowed_token_ids->empty()) {
        ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.assign(disallowed_token_ids->begin(),
                                                                        disallowed_token_ids->end());
        std::sort(ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.begin(),
                  ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.end());
        ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.erase(
            std::unique(ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.begin(),
                        ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.end()),
            ctx->matmul.lfm2_greedy_lm_head_argmax_disallowed_tokens.end());
    }
}

void WriteInferenceWorkContextLFM2GreedyLMHeadSparseLogits(InferenceWorkContext* ctx, float* output, int vocab_size,
                                                           int token, float value) {
    if (!output || vocab_size <= 0) {
        return;
    }
    const bool can_reuse_sparse_buffer = ctx && ctx->matmul.lfm2_greedy_lm_head_sparse_logits_ptr == output &&
                                         ctx->matmul.lfm2_greedy_lm_head_sparse_logits_vocab_size == vocab_size &&
                                         ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token >= 0 &&
                                         ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token < vocab_size;
    if (can_reuse_sparse_buffer) {
        output[ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token] = -std::numeric_limits<float>::infinity();
    } else {
        std::fill_n(output, vocab_size, -std::numeric_limits<float>::infinity());
    }
    if (token >= 0 && token < vocab_size && std::isfinite(value)) {
        output[token] = value;
    }
    if (ctx) {
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_ptr = output;
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_vocab_size = vocab_size;
        ctx->matmul.lfm2_greedy_lm_head_sparse_logits_token = token;
    }
}

void RecordInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(InferenceWorkContext* ctx, uint64_t generation, int token,
                                                           float value, int vocab_size) {
    if (!ctx) {
        return;
    }
    ctx->matmul.lfm2_greedy_lm_head_argmax_generation = generation;
    ctx->matmul.lfm2_greedy_lm_head_argmax_token = token;
    ctx->matmul.lfm2_greedy_lm_head_argmax_value = value;
    ctx->matmul.lfm2_greedy_lm_head_argmax_vocab_size = vocab_size;
}

bool TryGetInferenceWorkContextLFM2GreedyLMHeadArgmaxToken(const InferenceWorkContext* ctx, int vocab_size, int* token,
                                                           float* value) {
    if (!ctx || !token || !value || vocab_size <= 0 || !ctx->matmul.lfm2_greedy_lm_head_argmax_allowed ||
        ctx->matmul.lfm2_greedy_lm_head_argmax_generation != ctx->execution_generation ||
        ctx->matmul.lfm2_greedy_lm_head_argmax_vocab_size != vocab_size ||
        ctx->matmul.lfm2_greedy_lm_head_argmax_token < 0 ||
        ctx->matmul.lfm2_greedy_lm_head_argmax_token >= vocab_size ||
        !std::isfinite(ctx->matmul.lfm2_greedy_lm_head_argmax_value)) {
        return false;
    }
    *token = ctx->matmul.lfm2_greedy_lm_head_argmax_token;
    *value = ctx->matmul.lfm2_greedy_lm_head_argmax_value;
    return true;
}

uint64_t GetInferenceWorkContextExecutionGenerationForTest(const InferenceWorkContext* ctx) {
    return ctx ? ctx->execution_generation : 0;
}

void SetCurrentExecutionPhase(InferenceExecutionPhase phase) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        return;
    }
    ctx->phase = phase;
}

InferenceExecutionPhase GetCurrentExecutionPhase() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    return ctx ? ctx->phase : InferenceExecutionPhase::Unknown;
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
}

const BatchSpec* GetCurrentBatch() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (ctx && ctx->batch) {
        return ctx->batch;
    }
    return nullptr;
}

void ClearCurrentBatch() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (ctx) {
        ctx->batch = nullptr;
    }
}

static Gemma4SharedKVState* GetGemma4SharedKVStateSlot(int layer) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetGemma4SharedKVStateSlot called without active InferenceWorkContext");
    }
    if (layer < 0) {
        throw densecore::InvalidArgumentException("GetGemma4SharedKVStateSlot called with negative layer");
    }
    if (ctx->gemma4_shared_kv_states.size() <= static_cast<size_t>(layer)) {
        ctx->gemma4_shared_kv_states.resize(static_cast<size_t>(layer) + 1);
    }
    return &ctx->gemma4_shared_kv_states[static_cast<size_t>(layer)];
}

void SetGemma4SharedKVState(int source_layer, ggml_tensor* k, ggml_tensor* v) {
    Gemma4SharedKVState* slot = GetGemma4SharedKVStateSlot(source_layer);
    slot->k = k;
    slot->v = v;
    slot->source_layer = source_layer;
}

const Gemma4SharedKVState* GetGemma4SharedKVState(int source_layer) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx || source_layer < 0 || static_cast<size_t>(source_layer) >= ctx->gemma4_shared_kv_states.size()) {
        return nullptr;
    }
    const Gemma4SharedKVState& slot = ctx->gemma4_shared_kv_states[static_cast<size_t>(source_layer)];
    if (!slot.k || !slot.v || slot.source_layer != source_layer) {
        return nullptr;
    }
    return &slot;
}

// NOTE: KVCacheUserData is defined in runtime/inference_types_internal.h
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
    ctx->kv_pool[idx].work_ctx = ctx;
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

QKVUserData* GetQKVUserData() {
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

PagedAttentionUserData* GetPagedAttentionUserData() {
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
    PagedAttentionUserData* ud = &ctx->paged_attention_userdata_pool[idx];
    ud->work_ctx = ctx;
    ud->shared_k_block_ptrs = &ctx->paged_attention_shared_k_block_ptrs;
    ud->shared_v_block_ptrs = &ctx->paged_attention_shared_v_block_ptrs;
    ud->shared_block_ptrs_ready.store(0, std::memory_order_relaxed);
    return ud;
}

SSMConv1DUserData* GetSSMConv1DUserData() {
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

LFM2ShortConvUserData* GetLFM2ShortConvUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetLFM2ShortConvUserData called without active InferenceWorkContext");
    }
    int idx = ctx->lfm2_shortconv_index++;
    if (idx >= 128) {
        ctx->lfm2_shortconv_index = 0;
        throw densecore::OutOfMemoryException("LFM2ShortConvUserData pool exhausted");
    }
    LFM2ShortConvUserData* ud = &ctx->lfm2_shortconv_pool[idx];
    ud->decode_in_proj_ready_stamp.store(0, std::memory_order_relaxed);
    ud->decode_in_proj_done.store(0, std::memory_order_relaxed);
    ud->decode_y_stamp.store(0, std::memory_order_relaxed);
    ud->decode_out_proj_stamp.store(0, std::memory_order_relaxed);
    ud->decode_in_proj_q4k_packed.reset();
    ud->decode_in_proj_weight_data = nullptr;
    ud->decode_in_proj_rows = 0;
    ud->decode_in_proj_cols = 0;
    ud->decode_out_proj_q4k_packed.reset();
    ud->decode_out_proj_weight_data = nullptr;
    ud->decode_out_proj_rows = 0;
    ud->decode_out_proj_cols = 0;
    return ud;
}

ProjectionReferenceUserData* GetProjectionReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetProjectionReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->projection_reference_index++;
    if (idx >= 384) {
        ctx->projection_reference_index = 0;
        idx = 0;
    }
    ctx->projection_reference_pool[idx] = ProjectionReferenceUserData{};
    return &ctx->projection_reference_pool[idx];
}

RmsNormReferenceUserData* GetRmsNormReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetRmsNormReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->rmsnorm_reference_index++;
    if (idx >= 256) {
        ctx->rmsnorm_reference_index = 0;
        idx = 0;
    }
    return &ctx->rmsnorm_reference_pool[idx];
}

SharedScalarGateReferenceUserData* GetSharedScalarGateReferenceUserData() {
    static thread_local SharedScalarGateReferenceUserData pool[64];
    static thread_local int index = 0;
    if (index >= 64) {
        index = 0;
    }
    return &pool[index++];
}

AttentionCoreReferenceUserData* GetAttentionCoreReferenceUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException(
            "GetAttentionCoreReferenceUserData called without active InferenceWorkContext");
    }
    int idx = ctx->attention_core_reference_index++;
    if (idx >= 128) {
        ctx->attention_core_reference_index = 0;
        idx = 0;
    }
    return &ctx->attention_core_reference_pool[idx];
}

SSMQwen35DeltaUserData* GetSSMQwen35DeltaUserData() {
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


GemvUserData* GetGemvUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvUserData called without active InferenceWorkContext");
    }
    int idx = ctx->matmul.gemv_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->matmul.gemv_userdata_index = 0;
        idx = 0;
    }
    GemvUserData* ud = &ctx->matmul.gemv_userdata_pool[idx];
    ud->slot_id = idx;
    ud->quant_input_shared = ctx->matmul.gemv_quant_input_shared.data();
    ud->quantized_stamp = &ctx->matmul.gemv_quantized_stamp;
    ud->lfm2_argmax_key = 0;
    ud->lfm2_argmax_done = 0;
    ud->lfm2_argmax_nth = 0;
    ud->lfm2_argmax_best_token = -1;
    ud->lfm2_argmax_best_value = -std::numeric_limits<float>::infinity();
    ud->lfm2_argmax_begin = {};
    return ud;
}

GemvBatchedUserData* GetGemvBatchedUserData() {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemvBatchedUserData called without active InferenceWorkContext");
    }
    int idx = ctx->matmul.gemv_batched_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        ctx->matmul.gemv_batched_userdata_index = 0;
        idx = 0;
    }
    GemvBatchedUserData* ud = &ctx->matmul.gemv_batched_userdata_pool[idx];
    ud->slot_id = idx;
    ud->input_quant_type = GGML_TYPE_F32;
    ud->quant_input_shared = ctx->matmul.gemv_batched_quant_input_shared.data();
    ud->quantized_stamp = &ctx->matmul.gemv_batched_quantized_stamp;
    ud->work_ctx = ctx;
    ud->qwen36_prefill_q4k_admission_key = 0;
    ud->qwen36_prefill_q4k_probe = false;
    ud->qwen36_prefill_q4k_admitted = false;
    ud->require_q4k_true_batched = false;
    ud->disable_quant_nrc_fast = false;
    ud->gemma4_dense_prefill_native = false;
    ud->gemma4_prefill_safe_batched = false;
    ud->lfm2_q8_repacked_batched = false;
    ud->qwen36_ssm_q8_repacked_batched = false;
    ud->qwen36_ssm_q8_direct_batched = false;
    ud->qwen38_q4k_8x8_weight = nullptr;
    ud->qwen38_q4k_8x8_rows = 0;
    ud->qwen38_q4k_8x8_cols = 0;
    ud->qwen36_prefill_q4k_probe_done.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_failures.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_internal_errors.store(0, std::memory_order_relaxed);
    ud->qwen36_prefill_q4k_probe_max_abs_error_bits.store(0, std::memory_order_relaxed);
    return ud;
}
