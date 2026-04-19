// ============================================================================
// EXPLICIT INFERENCE WORK CONTEXT (per-thread, no implicit TLS pools)
// ============================================================================
struct InferenceWorkContext {
    const BatchSpec* batch = nullptr;
    KVCacheUserData kv_pool[256];
    std::vector<Gemma4SharedKVState> gemma4_shared_kv_states;
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
    std::vector<const void*> paged_attention_shared_k_block_ptrs;
    std::vector<const void*> paged_attention_shared_v_block_ptrs;
    SSMConv1DUserData ssm_conv1d_pool[128];
    int ssm_conv1d_index = 0;
    ProjectionReferenceUserData projection_reference_pool[384];
    int projection_reference_index = 0;
    RmsNormReferenceUserData rmsnorm_reference_pool[256];
    int rmsnorm_reference_index = 0;
    AttentionCoreReferenceUserData attention_core_reference_pool[128];
    int attention_core_reference_index = 0;
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
    ctx->gemma4_shared_kv_states.clear();
    ctx->qkv_index = 0;
    ctx->add_rmsnorm_index = 0;
    ctx->gemv_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->gemv_batched_quantized_stamp.store(0, std::memory_order_relaxed);
    ctx->gemv_userdata_index = 0;
    ctx->gemv_batched_userdata_index = 0;
    ctx->paged_attention_userdata_index = 0;
    ctx->paged_attention_shared_k_block_ptrs.clear();
    ctx->paged_attention_shared_v_block_ptrs.clear();
    ctx->ssm_conv1d_index = 0;
    ctx->projection_reference_index = 0;
    ctx->rmsnorm_reference_index = 0;
    ctx->attention_core_reference_index = 0;
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

static Gemma4SharedKVState* GetGemma4SharedKVStateSlot(int layer) {
    InferenceWorkContext* ctx = GetCurrentWorkContext();
    if (!ctx) {
        throw densecore::InvalidArgumentException("GetGemma4SharedKVStateSlot called without active InferenceWorkContext");
    }
    if (layer < 0) {
        throw densecore::InvalidArgumentException("GetGemma4SharedKVStateSlot called with negative layer");
    }
    if (ctx->gemma4_shared_kv_states.size() <= static_cast<size_t>(layer)) {
        ctx->gemma4_shared_kv_states.resize(static_cast<size_t>(layer) + 1);
    }
    return &ctx->gemma4_shared_kv_states[static_cast<size_t>(layer)];
}

static void SetGemma4SharedKVState(int source_layer, ggml_tensor* k, ggml_tensor* v) {
    Gemma4SharedKVState* slot = GetGemma4SharedKVStateSlot(source_layer);
    slot->k = k;
    slot->v = v;
    slot->source_layer = source_layer;
}

static const Gemma4SharedKVState* GetGemma4SharedKVState(int source_layer) {
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
    PagedAttentionUserData* ud = &ctx->paged_attention_userdata_pool[idx];
    ud->shared_k_block_ptrs = &ctx->paged_attention_shared_k_block_ptrs;
    ud->shared_v_block_ptrs = &ctx->paged_attention_shared_v_block_ptrs;
    ud->shared_block_ptrs_ready.store(0, std::memory_order_relaxed);
    return ud;
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

inline ProjectionReferenceUserData* GetProjectionReferenceUserData() {
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
    return &ctx->projection_reference_pool[idx];
}

inline RmsNormReferenceUserData* GetRmsNormReferenceUserData() {
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

inline SharedScalarGateReferenceUserData* GetSharedScalarGateReferenceUserData() {
    static thread_local SharedScalarGateReferenceUserData pool[64];
    static thread_local int index = 0;
    if (index >= 64) {
        index = 0;
    }
    return &pool[index++];
}

inline AttentionCoreReferenceUserData* GetAttentionCoreReferenceUserData() {
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

static inline bool ModelUsesMRoPE(const TransformerModel* model);
static inline int PositionIdsPerToken(const TransformerModel* model);

inline struct ggml_tensor* ggml_rope_precomputed_table(struct ggml_context* ctx, struct ggml_tensor* input,
                                                       struct ggml_tensor* pos, const TransformerModel* model,
                                                       int rope_dim, const BatchSpec* batch) {
    if (!ctx || !input || !pos || !model) return nullptr;
    if (input->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) return nullptr;
    if (ModelUsesMRoPE(model)) return nullptr;
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

static inline bool IsTokenSpanDense(const struct ggml_tensor* t, int head_dim, int n_head_kv) {
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    return IsTokenHeadDense(t, head_dim) && t->nb[2] == head_block_bytes;
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

static inline bool ModelUsesMRoPE(const TransformerModel* model) {
    static const bool force_standard_qwen35_rope = ParseTruthyEnv("DENSECORE_QWEN35_FORCE_STANDARD_ROPE", false);
    if (force_standard_qwen35_rope && model && model->arch == ModelArch::QWEN35) {
        return false;
    }
    return model && model->hparams.rope_sections[0] > 0 && model->hparams.rope_sections[1] > 0;
}

static inline int ModelMRoPEMode(const TransformerModel* model) {
    if (!ModelUsesMRoPE(model)) {
        return GGML_ROPE_TYPE_NORMAL;
    }
    return model->hparams.rope_mrope_interleaved ? GGML_ROPE_TYPE_IMROPE : GGML_ROPE_TYPE_MROPE;
}

static inline int PositionIdsPerToken(const TransformerModel* model) {
    return ModelUsesMRoPE(model) ? GGML_MROPE_SECTIONS : 1;
}

bool PopulatePositionTensor(TransformerModel* model, const BatchSpec& batch, struct ggml_tensor* pos) {
    if (!model || !pos || !pos->data || pos->type != GGML_TYPE_I32) {
        return false;
    }

    const int n_tokens = static_cast<int>(batch.pos.size());
    if (n_tokens <= 0) {
        return false;
    }

    const int ids_per_token = PositionIdsPerToken(model);
    const int expected = n_tokens * ids_per_token;
    if (pos->ne[0] != expected) {
        return false;
    }

    int32_t* dst = reinterpret_cast<int32_t*>(pos->data);
    if (ids_per_token == 1) {
        std::memcpy(dst, batch.pos.data(), static_cast<size_t>(n_tokens) * sizeof(int32_t));
        return true;
    }

    for (int i = 0; i < n_tokens; ++i) {
        const int32_t p = batch.pos[static_cast<size_t>(i)];
        for (int j = 0; j < ids_per_token; ++j) {
            const size_t offset = static_cast<size_t>(j) * static_cast<size_t>(n_tokens) + static_cast<size_t>(i);
            dst[offset] = p;
        }
    }

    return true;
}

static inline void GatherTokenSpanHeadContiguous(const struct ggml_tensor* src, int token_idx, int token_count,
                                                 int head_dim, int n_head_kv, float* out) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    const char* token_base = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(token_idx) * src->nb[2];
    if (IsTokenSpanDense(src, head_dim, n_head_kv)) {
        std::memcpy(out, token_base, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }
    for (int i = 0; i < token_count; ++i) {
        GatherTokenHeadContiguous(src, token_idx + i, head_dim, n_head_kv,
                                  out + static_cast<size_t>(i) * static_cast<size_t>(head_dim) * n_head_kv);
    }
}

static inline void ScatterTokenSpanHeadContiguous(const float* in, struct ggml_tensor* dst, int token_idx,
                                                  int token_count, int head_dim, int n_head_kv) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        std::memcpy(token_base, in, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }
    for (int i = 0; i < token_count; ++i) {
        ScatterTokenHeadContiguous(in + static_cast<size_t>(i) * static_cast<size_t>(head_dim) * n_head_kv, dst,
                                   token_idx + i, head_dim, n_head_kv);
    }
}

static inline void ZeroTokenSpanHead(struct ggml_tensor* dst, int token_idx, int token_count, int head_dim,
                                     int n_head_kv) {
    if (token_count <= 0) {
        return;
    }
    const size_t head_block_bytes = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv) * sizeof(float);
    char* token_base = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(token_idx) * dst->nb[2];
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        std::memset(token_base, 0, static_cast<size_t>(token_count) * head_block_bytes);
        return;
    }

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    for (int i = 0; i < token_count; ++i) {
        char* token_ptr = token_base + static_cast<size_t>(i) * dst->nb[2];
        for (int h = 0; h < n_head_kv; ++h) {
            char* head_ptr = token_ptr + static_cast<size_t>(h) * nb1;
            for (int d = 0; d < head_dim; ++d) {
                *reinterpret_cast<float*>(head_ptr + static_cast<size_t>(d) * nb0) = 0.0f;
            }
        }
    }
}

static inline void WriteCurrentBatchKvToCache(const BatchSpec* batch, const struct ggml_tensor* src,
                                              PagedKVCache* cache, int layer, int head_dim, int n_head_kv, bool is_k,
                                              int token_begin, int token_end, int* writes_ok, int* writes_skipped) {
    if (!batch || !src || !cache || !src->data || head_dim <= 0 || n_head_kv <= 0) {
        return;
    }

    const int N = static_cast<int>(batch->tokens.size());
    if (N <= 0) {
        return;
    }

    const int begin = std::max(0, token_begin);
    const int end = (token_end < 0) ? N : std::min(N, token_end);
    if (begin >= end) {
        return;
    }

    const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    const bool src_dense = IsTokenSpanDense(src, head_dim, n_head_kv);
    std::vector<float> packed;
    if (!src_dense) {
        packed.reserve(static_cast<size_t>(std::min(end - begin, BLOCK_SIZE)) * head_block_size);
    }

    const bool debug_gemma4_decode_write = [&]() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0 && batch->tokens.size() == 1 &&
               (layer == 13 || layer == 14);
    }();

    for (int i = begin; i < end;) {
        if (i >= static_cast<int>(batch->seq_id.size()) || i >= static_cast<int>(batch->pos.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const int seq_id = batch->seq_id[i];
        const int pos = batch->pos[i];
        if (seq_id < 0 || seq_id >= static_cast<int>(batch->block_tables.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const auto& block_table = batch->block_tables[seq_id];
        const int logical_block = pos / BLOCK_SIZE;
        const int slot = pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            if (writes_skipped) {
                ++(*writes_skipped);
            }
            ++i;
            continue;
        }

        const int block_id = block_table[logical_block];
        int run = 1;
        while (i + run < end && i + run < static_cast<int>(batch->seq_id.size()) &&
               i + run < static_cast<int>(batch->pos.size())) {
            const int next_seq_id = batch->seq_id[i + run];
            const int next_pos = batch->pos[i + run];
            if (next_seq_id != seq_id || next_pos != pos + run) {
                break;
            }
            if ((next_pos / BLOCK_SIZE) != logical_block) {
                break;
            }
            ++run;
        }

        packed.resize(static_cast<size_t>(run) * head_block_size);
        const float* run_data = nullptr;
        if (src_dense) {
            run_data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                      static_cast<size_t>(i) * src->nb[2]);
        } else {
            GatherTokenSpanHeadContiguous(src, i, run, head_dim, n_head_kv, packed.data());
            run_data = packed.data();
        }
        if (is_k) {
            cache->WriteKSlots(block_id, layer, slot, run, run_data);
        } else {
            cache->WriteVSlots(block_id, layer, slot, run, run_data);
        }
        if (debug_gemma4_decode_write && run == 1) {
            std::vector<float> roundtrip(static_cast<size_t>(run) * head_block_size, 0.0f);
            if (is_k) {
                cache->ReadKSlots(block_id, layer, slot, run, roundtrip.data());
            } else {
                cache->ReadVSlots(block_id, layer, slot, run, roundtrip.data());
            }

            auto log_slice = [&](const char* stage, const float* values) {
                float min_v = std::numeric_limits<float>::infinity();
                float max_v = -std::numeric_limits<float>::infinity();
                float max_abs = 0.0f;
                double sum = 0.0;
                double sum_sq = 0.0;
                double checksum = 0.0;
                int finite_ct = 0;
                int nan_ct = 0;
                int inf_ct = 0;
                for (size_t vi = 0; vi < head_block_size; ++vi) {
                    const float v = values[vi];
                    if (std::isnan(v)) {
                        nan_ct++;
                        continue;
                    }
                    if (!std::isfinite(v)) {
                        inf_ct++;
                        continue;
                    }
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    max_abs = std::max(max_abs, std::fabs(v));
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    checksum += static_cast<double>(vi + 1) * static_cast<double>(v);
                    finite_ct++;
                }
                if (!std::isfinite(min_v)) min_v = 0.0f;
                if (!std::isfinite(max_v)) max_v = 0.0f;
                std::fprintf(stderr,
                             "[GEMMA4_SHARED_KV] action=%s kind=%s layer=%d source_layer=%d seq=%d pos=%d block=%d "
                             "slot=%d head_dim=%d heads=%d nan=%d inf=%d min=%.8g max=%.8g max_abs=%.8g "
                             "mean=%.8g rms=%.8g checksum=%.12g first8=",
                             stage, is_k ? "K" : "V", layer, layer, seq_id, pos, block_id, slot, head_dim, n_head_kv,
                             nan_ct, inf_ct, min_v, max_v, max_abs, finite_ct > 0 ? (sum / finite_ct) : 0.0,
                             finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0, checksum);
                const size_t preview = std::min<size_t>(8, head_block_size);
                for (size_t vi = 0; vi < preview; ++vi) {
                    std::fprintf(stderr, "%s%.8g", vi == 0 ? "" : ",", values[vi]);
                }
                std::fprintf(stderr, "\n");
            };

            log_slice("pre-cache-write", run_data);
            log_slice("post-cache-readback", roundtrip.data());
        }
        if (ShouldRunKvRoundTripProbe(layer, is_k)) {
            std::vector<float> roundtrip(static_cast<size_t>(run) * head_block_size, 0.0f);
            if (is_k) {
                cache->ReadKSlots(block_id, layer, slot, run, roundtrip.data());
            } else {
                cache->ReadVSlots(block_id, layer, slot, run, roundtrip.data());
            }

            float max_abs_diff = 0.0f;
            int max_idx = -1;
            for (size_t ri = 0; ri < roundtrip.size(); ++ri) {
                const float diff = std::fabs(roundtrip[ri] - run_data[ri]);
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    max_idx = static_cast<int>(ri);
                }
            }
            std::fprintf(stderr,
                         "[KV_ROUNDTRIP] layer=%d kind=%s seq=%d pos=%d block=%d slot=%d run=%d cache_type=%s "
                         "max_abs_diff=%g max_idx=%d src=%g dst=%g\n",
                         layer, is_k ? "K" : "V", seq_id, pos, block_id, slot, run, ggml_type_name(cache->cache_type),
                         static_cast<double>(max_abs_diff), max_idx,
                         max_idx >= 0 ? static_cast<double>(run_data[max_idx]) : 0.0,
                         max_idx >= 0 ? static_cast<double>(roundtrip[static_cast<size_t>(max_idx)]) : 0.0);
        }
        if (writes_ok) {
            *writes_ok += run;
        }
        i += run;
    }
}

static inline bool ReadSingleCurrentBatchKvFromCache(const BatchSpec* batch, PagedKVCache* cache, int layer,
                                                     int head_dim, int n_head_kv, bool is_k, int batch_token_idx,
                                                     float* out) {
    if (!batch || !cache || !out || batch_token_idx < 0 || batch_token_idx >= static_cast<int>(batch->tokens.size()) ||
        batch_token_idx >= static_cast<int>(batch->seq_id.size()) ||
        batch_token_idx >= static_cast<int>(batch->pos.size())) {
        return false;
    }

    const int seq_id = batch->seq_id[batch_token_idx];
    const int pos = batch->pos[batch_token_idx];
    if (seq_id < 0 || seq_id >= static_cast<int>(batch->block_tables.size()) || pos < 0) {
        return false;
    }

    const auto& block_table = batch->block_tables[static_cast<size_t>(seq_id)];
    const int logical_block = pos / BLOCK_SIZE;
    const int slot = pos % BLOCK_SIZE;
    if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
        return false;
    }

    const int block_id = block_table[static_cast<size_t>(logical_block)];
    if (block_id < 0) {
        return false;
    }

    if (is_k) {
        cache->ReadKSlots(block_id, layer, slot, 1, out);
    } else {
        cache->ReadVSlots(block_id, layer, slot, 1, out);
    }
    return true;
}

// Custom callback to load K/V history from cache and append current K/V.
// This implementation is stride-safe for both contiguous and view tensors.
void cb_kv_manage(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !src || !dst || !src->data || !dst->data) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    const int n_total = static_cast<int>(src->ne[2]);
    const int n_past = n_total - N;
    if (N < 0 || n_total < 0 || n_past < 0) return;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    int writes_ok = 0;
    int writes_skipped = 0;

    // 1) Write current tokens to cache (parallel over token ranges).
    if (N > 0) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                       &writes_ok, &writes_skipped);
        }
    }

    // 2) Read history [0, n_past) from cache into dst (parallel over tokens).
    if (n_past > 0) {
        const int seq_id = batch->seq_id.empty() ? -1 : batch->seq_id[0];
        const bool has_valid_seq = (seq_id >= 0 && seq_id < static_cast<int>(batch->block_tables.size()));
        const auto* block_table = has_valid_seq ? &batch->block_tables[seq_id] : nullptr;
        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
        if (ud->force_full_history) {
            retention_policy.enabled = false;
            retention_policy.sliding_window = -1;
            retention_policy.sink_tokens = 0;
        }
        KVRetentionSpan retained_history;
        if (has_valid_seq && seq_id < static_cast<int>(batch->n_past.size())) {
            const int seq_n_past = std::max(0, batch->n_past[static_cast<size_t>(seq_id)]);
            retained_history = ComputeKVRetentionSpan(seq_n_past, retention_policy);
        }

        const int tokens_per_thread = (n_past + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_past);

        std::vector<float> packed;
        packed.reserve(static_cast<size_t>(std::max(1, BLOCK_SIZE)) * head_block_size);
        for (int t = t_start; t < t_end;) {
            if (!has_valid_seq) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            if (!block_table || t >= retained_history.history_kept) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            const int token_pos = MapRetainedHistoryIndex(retained_history, t);
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table->size())) {
                ZeroTokenSpanHead(dst, t, 1, head_dim, n_head_kv);
                ++t;
                continue;
            }

            const int block_id = (*block_table)[logical_block];
            int run = 1;
            while (t + run < t_end && t + run < retained_history.history_kept) {
                const int next_token_pos = MapRetainedHistoryIndex(retained_history, t + run);
                if (next_token_pos != token_pos + run) {
                    break;
                }
                if ((next_token_pos / BLOCK_SIZE) != logical_block) {
                    break;
                }
                ++run;
            }
            packed.resize(static_cast<size_t>(run) * head_block_size);
            if (ud->is_k) {
                ud->cache->ReadKSlots(block_id, ud->layer, slot, run, packed.data());
            } else {
                ud->cache->ReadVSlots(block_id, ud->layer, slot, run, packed.data());
            }
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t, run, head_dim, n_head_kv);
            t += run;
        }
    }

    // 3) Append current tokens to dst [n_past, n_total) (parallel over tokens).
    if (N > 0) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);

        if (t_end > t_start) {
            std::vector<float> packed;
            packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
            GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
            ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, t_end - t_start, head_dim, n_head_kv);
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

void cb_kv_write_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !src || !dst || !src->data || !dst->data) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    if (N <= 0) {
        return;
    }

    const int tokens_per_thread = (N + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, N);

    if (t_end > t_start) {
        WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                   nullptr, nullptr);
    }

    const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    if (t_end > t_start) {
        if (IsTokenSpanDense(src, head_dim, n_head_kv) && IsTokenSpanDense(dst, head_dim, n_head_kv)) {
            const char* src_ptr = reinterpret_cast<const char*>(src->data) + static_cast<size_t>(t_start) * src->nb[2];
            char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(t_start) * dst->nb[2];
            std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(t_end - t_start) * head_block_size * sizeof(float));
        } else {
            std::vector<float> packed;
            packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
            GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t_start, t_end - t_start, head_dim, n_head_kv);
        }
    }
}

void cb_kv_update_and_gather_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<KVCacheUserData*>(userdata);
    if (!ud || !ud->cache || !dst || !dst->data || !dst->src[0] || !dst->src[0]->data || nth <= 0) return;

    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    const struct ggml_tensor* src = dst->src[0];
    const int head_dim = ud->head_dim_kv;
    const int n_head_kv = ud->cache->GetHeadCountForLayer(ud->layer);
    if (head_dim <= 0 || n_head_kv <= 0) return;

    const int N = static_cast<int>(batch->tokens.size());
    const int n_total = static_cast<int>(dst->ne[2]);
    const int n_past = n_total - N;
    if (N < 0 || n_total < 0 || n_past < 0) return;

    const bool read_only_shared_kv = ud->read_only_shared_kv;
    const int history_tokens = read_only_shared_kv ? n_total : n_past;
    const char* debug_shared_env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
    const bool debug_gemma4_decode = debug_shared_env && debug_shared_env[0] != '\0' &&
                                     std::strcmp(debug_shared_env, "0") != 0 && (ud->layer == 13 || ud->layer == 14);
    if (debug_gemma4_decode && ith == 0) {
        std::fprintf(stderr,
                     "[GEMMA4_SHARED_KV] action=callback-state kind=%s layer=%d source_layer=%d N=%d n_total=%d "
                     "n_past=%d read_only=%d force_full_history=%d seqs=%d pos0=%d\n",
                     ud->is_k ? "K" : "V", ud->layer, ud->layer, N, n_total, n_past, read_only_shared_kv ? 1 : 0,
                     ud->force_full_history ? 1 : 0, batch->num_seqs,
                     batch->pos.empty() ? -1 : batch->pos[0]);
    }

    if (N > 0 && !read_only_shared_kv) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            WriteCurrentBatchKvToCache(batch, src, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k, t_start, t_end,
                                       nullptr, nullptr);
        }
    }

    if (history_tokens > 0) {
        const int seq_id = batch->seq_id.empty() ? -1 : batch->seq_id[0];
        const bool has_valid_seq = (seq_id >= 0 && seq_id < static_cast<int>(batch->block_tables.size()));
        const auto* block_table = has_valid_seq ? &batch->block_tables[seq_id] : nullptr;
        KVRetentionPolicy retention_policy = GetKVRetentionPolicy();
        if (ud->force_full_history) {
            retention_policy.enabled = false;
            retention_policy.sliding_window = -1;
            retention_policy.sink_tokens = 0;
        }
        KVRetentionSpan retained_history;
        if (has_valid_seq && seq_id < static_cast<int>(batch->n_past.size())) {
            const int seq_n_past = read_only_shared_kv ? history_tokens
                                                       : std::max(0, batch->n_past[static_cast<size_t>(seq_id)]);
            retained_history = ComputeKVRetentionSpan(seq_n_past, retention_policy);
        }

        const int tokens_per_thread = (history_tokens + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, history_tokens);
        std::vector<float> packed;
        const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
        packed.reserve(static_cast<size_t>(std::max(1, BLOCK_SIZE)) * head_block_size);
        for (int t = t_start; t < t_end;) {
            if (!has_valid_seq) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            if (!block_table || t >= retained_history.history_kept) {
                ZeroTokenSpanHead(dst, t, t_end - t, head_dim, n_head_kv);
                break;
            }

            const int token_pos = MapRetainedHistoryIndex(retained_history, t);
            const int logical_block = token_pos / BLOCK_SIZE;
            const int slot = token_pos % BLOCK_SIZE;
            if (logical_block < 0 || logical_block >= static_cast<int>(block_table->size())) {
                ZeroTokenSpanHead(dst, t, 1, head_dim, n_head_kv);
                ++t;
                continue;
            }

            const int block_id = (*block_table)[logical_block];
            int run = 1;
            while (t + run < t_end && t + run < retained_history.history_kept) {
                const int next_token_pos = MapRetainedHistoryIndex(retained_history, t + run);
                if (next_token_pos != token_pos + run) {
                    break;
                }
                if ((next_token_pos / BLOCK_SIZE) != logical_block) {
                    break;
                }
                ++run;
            }
            packed.resize(static_cast<size_t>(run) * head_block_size);
            if (ud->is_k) {
                ud->cache->ReadKSlots(block_id, ud->layer, slot, run, packed.data());
            } else {
                ud->cache->ReadVSlots(block_id, ud->layer, slot, run, packed.data());
            }
            ScatterTokenSpanHeadContiguous(packed.data(), dst, t, run, head_dim, n_head_kv);
            t += run;
        }
    }

    if (N > 0 && !read_only_shared_kv) {
        const int tokens_per_thread = (N + nth - 1) / nth;
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, N);
        if (t_end > t_start) {
            const size_t head_block_size = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
            if (N == 1) {
                std::vector<float> packed(head_block_size, 0.0f);
                if (!ReadSingleCurrentBatchKvFromCache(batch, ud->cache, ud->layer, head_dim, n_head_kv, ud->is_k,
                                                       t_start, packed.data())) {
                    GatherTokenSpanHeadContiguous(src, t_start, 1, head_dim, n_head_kv, packed.data());
                }
                ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, 1, head_dim, n_head_kv);
            } else if (IsTokenSpanDense(src, head_dim, n_head_kv) && IsTokenSpanDense(dst, head_dim, n_head_kv)) {
                const char* src_ptr =
                    reinterpret_cast<const char*>(src->data) + static_cast<size_t>(t_start) * src->nb[2];
                char* dst_ptr = reinterpret_cast<char*>(dst->data) + static_cast<size_t>(n_past + t_start) * dst->nb[2];
                std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(t_end - t_start) * head_block_size * sizeof(float));
            } else {
                std::vector<float> packed;
                packed.resize(static_cast<size_t>(t_end - t_start) * head_block_size);
                GatherTokenSpanHeadContiguous(src, t_start, t_end - t_start, head_dim, n_head_kv, packed.data());
                ScatterTokenSpanHeadContiguous(packed.data(), dst, n_past + t_start, t_end - t_start, head_dim,
                                               n_head_kv);
            }
        }
    }
}

inline struct ggml_tensor* ggml_kv_update_and_gather(struct ggml_context* ctx, struct ggml_tensor* src, int n_total,
                                                     int n_tasks, KVCacheUserData* userdata) {
    if (!ctx || !src || n_total <= 0) {
        return src;
    }
    struct ggml_tensor* args[1] = {src};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, src->ne[0], src->ne[1], n_total, 1, args, 1,
                          cb_kv_update_and_gather_custom, n_tasks, userdata);
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
    for (int i = 0; i < N;) {
        if (i >= static_cast<int>(ud->batch->seq_id.size()) || i >= static_cast<int>(ud->batch->pos.size())) {
            ++i;
            continue;
        }

        const int seq_id = ud->batch->seq_id[i];
        const int pos = ud->batch->pos[i];
        if (seq_id < 0 || seq_id >= static_cast<int>(ud->batch->block_tables.size())) {
            ++i;
            continue;
        }

        const auto& block_table = ud->batch->block_tables[seq_id];
        const int logical_block = pos / BLOCK_SIZE;
        const int slot = pos % BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            ++i;
            continue;
        }

        int run = 1;
        while (i + run < N && i + run < static_cast<int>(ud->batch->seq_id.size()) &&
               i + run < static_cast<int>(ud->batch->pos.size())) {
            const int next_seq_id = ud->batch->seq_id[i + run];
            const int next_pos = ud->batch->pos[i + run];
            if (next_seq_id != seq_id || next_pos != pos + run) {
                break;
            }
            if ((next_pos / BLOCK_SIZE) != logical_block) {
                break;
            }
            ++run;
        }

        const int block_id = block_table[logical_block];
        const float* token_data = src_data + static_cast<size_t>(i) * head_block_size;
        if (is_k) {
            ud->cache->WriteKSlots(block_id, layer, slot, run, token_data);
        } else {
            ud->cache->WriteVSlots(block_id, layer, slot, run, token_data);
        }
        i += run;
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
            const bool dst_dense = IsTokenSpanDense(dst, head_dim, n_head_kv);
            std::vector<float> packed;
            packed.reserve(static_cast<size_t>(BLOCK_SIZE) * head_block_size);

            for (int i = 0; i < n_past;) {
                const int logical_block = i / BLOCK_SIZE;
                const int slot = i % BLOCK_SIZE;
                const int run = std::min(BLOCK_SIZE - slot, n_past - i);

                if (logical_block >= 0 && logical_block < static_cast<int>(block_table.size())) {
                    const int block_id = block_table[logical_block];
                    float* dst_slot =
                        dst_dense ? (reinterpret_cast<float*>(dst->data) + static_cast<size_t>(i) * head_block_size)
                                  : nullptr;
                    if (!dst_dense) {
                        packed.resize(static_cast<size_t>(run) * head_block_size);
                        dst_slot = packed.data();
                    }
                    if (is_k) {
                        ud->cache->ReadKSlots(block_id, layer, slot, run, dst_slot);
                    } else {
                        ud->cache->ReadVSlots(block_id, layer, slot, run, dst_slot);
                    }
                    if (!dst_dense) {
                        ScatterTokenSpanHeadContiguous(dst_slot, dst, i, run, head_dim, n_head_kv);
                    }
                } else {
                    ZeroTokenSpanHead(dst, i, run, head_dim, n_head_kv);
                }
                i += run;
            }
        } else {
            // Invalid sequence - zero-fill entire history section
            ZeroTokenSpanHead(dst, 0, n_past, head_dim, n_head_kv);
        }
    }

    // ===========================================================================
    // STEP C: Append current tokens to destination tensor [n_past, n_total)
    // ===========================================================================
    // Copy the current K/V data after the history section
    // ===========================================================================
    if (IsTokenSpanDense(dst, head_dim, n_head_kv)) {
        float* dst_current = reinterpret_cast<float*>(dst->data) + n_past * head_block_size;
        memcpy(dst_current, src_data, N * head_block_bytes);
    } else {
        ScatterTokenSpanHeadContiguous(src_data, dst, n_past, N, head_dim, n_head_kv);
    }
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
