#ifdef DENSECORE_TEST_BUILD
namespace densecore {
namespace testing {
namespace {
class ScopedAttentionCaptureGuard {
public:
    ScopedAttentionCaptureGuard(std::vector<float>* out, int layer)
        : previous_out_(g_test_capture_attention_out),
          previous_layer_(g_test_capture_attention_layer.load(std::memory_order_relaxed)) {
        g_test_capture_attention_out = out;
        g_test_capture_attention_layer.store(layer, std::memory_order_relaxed);
    }

    ~ScopedAttentionCaptureGuard() {
        g_test_capture_attention_layer.store(previous_layer_, std::memory_order_relaxed);
        g_test_capture_attention_out = previous_out_;
    }

private:
    std::vector<float>* previous_out_;
    int previous_layer_;
};
}  // namespace

bool ShouldUsePagedDecodeAttentionForBatchTest(const TransformerModel* model, const PagedKVCache* cache,
                                               const BatchSpec& batch) {
    if (!model) {
        return false;
    }
    const int n_tokens_in_batch = static_cast<int>(batch.tokens.size());
    const int n_head = static_cast<int>(model->hparams.n_head);
    const int n_head_kv = static_cast<int>(model->hparams.n_head_kv);
    if (n_head <= 0 || n_head_kv <= 0) {
        return false;
    }
    const int head_dim_q = static_cast<int>(model->hparams.n_embd) / n_head;
    const int head_dim_kv =
        model->hparams.n_embd_head_k > 0 ? static_cast<int>(model->hparams.n_embd_head_k) : head_dim_q;
    const BasePagedDecodeExecutionDecision decision =
        ::densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
            ::ResolveDecodePagedAttentionPolicy(&batch), model, cache, batch, n_tokens_in_batch, n_head, n_head_kv,
            head_dim_q, head_dim_kv);
    return decision.use_paged_decode_attention;
}

bool ShouldUsePrefillLastLogitsOnlyForTest(const TransformerModel* model, int num_seqs, int n_tokens) {
    BatchSpec batch{};
    batch.num_seqs = num_seqs;
    return ::ShouldUsePrefillLastLogitsOnly(model, batch, n_tokens);
}

std::vector<float> ComputeStandardAttentionOutputForTest(const std::vector<float>& q, const std::vector<float>& k,
                                                         const std::vector<float>& v, int n_head, int n_head_kv,
                                                         int head_dim_q, int head_dim_k, int head_dim_v, int n_queries,
                                                         int n_total_tokens, int n_past, int sliding_window,
                                                         float scale, float logit_softcap,
                                                         bool retained_history_layout) {
    if (n_head <= 0 || n_head_kv <= 0 || head_dim_q <= 0 || head_dim_k <= 0 || head_dim_v <= 0 || n_queries <= 0 ||
        n_total_tokens <= 0 || (n_head % n_head_kv) != 0) {
        return {};
    }
    if (static_cast<int>(q.size()) != n_queries * n_head * head_dim_q ||
        static_cast<int>(k.size()) != n_total_tokens * n_head_kv * head_dim_k ||
        static_cast<int>(v.size()) != n_total_tokens * n_head_kv * head_dim_v) {
        return {};
    }

    const bool use_explicit_mask =
        ::densecore::llm::attention::ShouldBuildExplicitStandardAttentionMask(n_queries, sliding_window);
    const int n_rep = n_head / n_head_kv;
    std::vector<float> out(static_cast<size_t>(n_queries) * static_cast<size_t>(n_head) * head_dim_v, 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_total_tokens), -std::numeric_limits<float>::infinity());

    KVRetentionPolicy retention_policy;
    retention_policy.enabled = (sliding_window >= 0);
    retention_policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    retention_policy.sink_tokens = GetKVRetentionPolicy().sink_tokens;
    const KVRetentionSpan retained = densecore::llm::config::ComputeKVRetentionSpan(n_past, retention_policy);

    for (int token_idx = 0; token_idx < n_queries; ++token_idx) {
        const int query_pos = n_past + token_idx;
        for (int h = 0; h < n_head; ++h) {
            const int kv_head = h / n_rep;
            const float* q_head = q.data() + (static_cast<size_t>(token_idx) * n_head + h) * head_dim_q;
            float* out_head = out.data() + (static_cast<size_t>(token_idx) * n_head + h) * head_dim_v;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k_idx = 0; k_idx < n_total_tokens; ++k_idx) {
                bool masked = false;
                if (use_explicit_mask) {
                    const int key_pos = retained_history_layout
                                            ? ((k_idx < retained.history_kept)
                                                   ? densecore::llm::config::MapRetainedHistoryIndex(retained, k_idx)
                                                   : (n_past + (k_idx - retained.history_kept)))
                                            : k_idx;
                    if (sliding_window >= 0 && key_pos < (query_pos - sliding_window)) {
                        masked = true;
                    }
                    if (n_queries > 1 && key_pos > query_pos) {
                        masked = true;
                    }
                }
                if (masked) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const float* k_head = k.data() + (static_cast<size_t>(k_idx) * n_head_kv + kv_head) * head_dim_k;
                float dot = 0.0f;
                for (int d = 0; d < head_dim_q; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                float score = dot;
                if (logit_softcap > 0.0f && std::isfinite(score)) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
                score *= scale;
                scores[static_cast<size_t>(k_idx)] = score;
                max_score = std::max(max_score, score);
            }

            if (!std::isfinite(max_score)) {
                continue;
            }

            float denom = 0.0f;
            for (int k_idx = 0; k_idx < n_total_tokens; ++k_idx) {
                const float score = scores[static_cast<size_t>(k_idx)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const float weight = std::exp(score - max_score);
                denom += weight;
                const float* v_head = v.data() + (static_cast<size_t>(k_idx) * n_head_kv + kv_head) * head_dim_v;
                for (int d = 0; d < head_dim_v; ++d) {
                    out_head[d] += weight * v_head[d];
                }
            }

            if (denom > 0.0f) {
                const float inv = 1.0f / denom;
                for (int d = 0; d < head_dim_v; ++d) {
                    out_head[d] *= inv;
                }
            }
        }
    }

    return out;
}
static std::vector<float> ExecuteTransformerGraphForTestImpl(TransformerModel* model, PagedKVCache* cache,
                                                             const BatchSpec& batch, int num_threads,
                                                             bool embedding_mode) {
    if (!model) {
        return {};
    }

    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> work_ctx(CreateInferenceWorkContext(),
                                                                                    DestroyInferenceWorkContext);
    if (!work_ctx) {
        return {};
    }

    SetCurrentWorkContext(work_ctx.get());
    ResetInferenceWorkContext(work_ctx.get());
    SetCurrentExecutionPhase(static_cast<int>(batch.tokens.size()) > std::max(1, batch.num_seqs)
                                 ? InferenceExecutionPhase::Prefill
                                 : InferenceExecutionPhase::Decode);
    SetCurrentBatch(&batch);

    struct ggml_init_params params = {
        /*.mem_size   =*/32ull * 1024ull * 1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        SetCurrentWorkContext(nullptr);
        return {};
    }

    std::vector<float> logits;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
    struct ggml_tensor* output = BuildTransformerGraph(model, cache, ctx, batch, embedding_mode, gf, nullptr, nullptr);
    if (output) {
        if (ggml_graph_n_nodes(gf) == 0) {
            ggml_build_forward_expand(gf, output);
        }
        ggml_graph_compute_with_ctx(ctx, gf, std::max(1, num_threads));
        if (output->data && output->type == GGML_TYPE_F32) {
            const int rows = static_cast<int>(output->ne[0]);
            const int cols = std::max(1, static_cast<int>(output->ne[1]));
            const ptrdiff_t row_stride = static_cast<ptrdiff_t>(output->nb[1] / sizeof(float));
            const float* src = reinterpret_cast<const float*>(output->data);
            logits.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            for (int col = 0; col < cols; ++col) {
                std::memcpy(logits.data() + static_cast<size_t>(col) * static_cast<size_t>(rows),
                            src + static_cast<ptrdiff_t>(col) * row_stride, static_cast<size_t>(rows) * sizeof(float));
            }
        }
    }

    ggml_free(ctx);
    ResetInferenceWorkContext(work_ctx.get());
    SetCurrentWorkContext(nullptr);
    return logits;
}
std::vector<float> ExecuteTransformerGraphForTest(TransformerModel* model, PagedKVCache* cache, const BatchSpec& batch,
                                                  int num_threads) {
    return ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/false);
}
std::vector<float> ExecuteTransformerGraphEmbeddingsForTest(TransformerModel* model, PagedKVCache* cache,
                                                            const BatchSpec& batch, int num_threads) {
    return ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/true);
}
std::vector<float> ExecuteTransformerAttentionForTest(TransformerModel* model, PagedKVCache* cache,
                                                      const BatchSpec& batch, int target_layer, int num_threads) {
    std::vector<float> captured;
    ScopedAttentionCaptureGuard capture_guard(&captured, target_layer);
    (void)ExecuteTransformerGraphForTestImpl(model, cache, batch, num_threads, /*embedding_mode=*/false);
    return captured;
}
void SetFlashAttentionDisabledForTest(bool disabled) {
    g_test_force_flash_attention_disabled.store(disabled ? 1 : 0, std::memory_order_relaxed);
}
void ComputeKVRetentionSpanForTest(int n_past, int sliding_window, int sink_tokens, int* history_kept, int* sink_kept,
                                   int* tail_start) {
    KVRetentionPolicy policy;
    policy.enabled = (sliding_window >= 0);
    policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    policy.sink_tokens = std::max(0, sink_tokens);

    const KVRetentionSpan span = densecore::llm::config::ComputeKVRetentionSpan(n_past, policy);
    if (history_kept) {
        *history_kept = span.history_kept;
    }
    if (sink_kept) {
        *sink_kept = span.sink_kept;
    }
    if (tail_start) {
        *tail_start = span.tail_start;
    }
}
int MapRetainedHistoryIndexForTest(int n_past, int sliding_window, int sink_tokens, int retained_index) {
    KVRetentionPolicy policy;
    policy.enabled = (sliding_window >= 0);
    policy.sliding_window = sliding_window >= 0 ? sliding_window : -1;
    policy.sink_tokens = std::max(0, sink_tokens);
    const KVRetentionSpan span = densecore::llm::config::ComputeKVRetentionSpan(n_past, policy);
    return densecore::llm::config::MapRetainedHistoryIndex(span, retained_index);
}
int GetArmQ4KNativeVecDotModeTest() {
    return static_cast<int>(::GetArmQ4KNativeVecDotMode());
}
void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}
uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}
void ResetHybridSSMQkvForceGgmlCache() {
    densecore::llm::models::ResetHybridSSMQkvForceGgmlCache();
}
bool ShouldForcePlainGgmlForHybridSSMQkvTest() {
    return densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv();
}
bool ShouldUseArmNativeQ4KVecDotValidatedTest(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                              const void* sample_row_ptr, const void* sample_quant_input,
                                              const float* sample_input_f32, int N) {
    return ::ShouldUseArmNativeQ4KVecDotValidated(weight_type, type_traits_cpu, nullptr, sample_row_ptr,
                                                  sample_quant_input, sample_input_f32, N);
}
std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeightsForTest(const TransformerLayer* layer,
                                                                            const TransformerModel* model) {
    return ::BuildExpertWeights(layer, model);
}
bool RouteMoESoftmaxTopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoESoftmaxTopK(gate_logits, &ud, routing);
}
bool RouteMoEGroupedSigmoidForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                                   const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoEGroupedSigmoid(gate_logits, &ud, routing);
}
bool RouteMoEGemma4TopKForTest(const struct ggml_tensor* gate_logits, const TransformerModel* model,
                               const TransformerLayer* layer, int top_k, densecore::moe::MoERouteResult* routing) {
    MoEUserData ud{};
    ud.model = model;
    ud.layer = layer;
    ud.k = top_k;
    return ::RouteMoEGemma4TopK(gate_logits, &ud, routing);
}
std::vector<float> ApplySharedScalarGateForTest(const std::vector<float>& shared_ffn_pre_gate,
                                                const std::vector<float>& shared_gate_logits_scalar, int tokens,
                                                int hidden_dim) {
    if (tokens <= 0 || hidden_dim <= 0 || static_cast<int>(shared_ffn_pre_gate.size()) != tokens * hidden_dim ||
        static_cast<int>(shared_gate_logits_scalar.size()) != tokens) {
        return {};
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.no_alloc = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return {};
    }

    struct ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, tokens);
    struct ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, tokens);
    struct ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, tokens);
    if (!src || !dst || !gate || !src->data || !dst->data || !gate->data) {
        ggml_free(ctx);
        return {};
    }

    std::memcpy(src->data, shared_ffn_pre_gate.data(), shared_ffn_pre_gate.size() * sizeof(float));
    std::memcpy(gate->data, shared_gate_logits_scalar.data(), shared_gate_logits_scalar.size() * sizeof(float));
    ::cb_apply_shared_scalar_gate(dst, src, gate, 0, 1, nullptr);

    std::vector<float> out(static_cast<size_t>(tokens * hidden_dim), 0.0f);
    std::memcpy(out.data(), dst->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}
std::vector<float> ComputeSharedExpertMergedOutputForTest(
    const std::vector<float>& moe_input, const std::vector<float>& routed_output, const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight, const std::vector<float>& down_weight,
    const std::vector<float>& shared_gate_logits_scalar, int tokens, int hidden_dim, int intermediate_dim) {
    if (tokens <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 ||
        static_cast<int>(moe_input.size()) != tokens * hidden_dim ||
        static_cast<int>(routed_output.size()) != tokens * hidden_dim ||
        static_cast<int>(gate_weight.size()) != intermediate_dim * hidden_dim ||
        static_cast<int>(up_weight.size()) != intermediate_dim * hidden_dim ||
        static_cast<int>(down_weight.size()) != hidden_dim * intermediate_dim ||
        static_cast<int>(shared_gate_logits_scalar.size()) != tokens) {
        return {};
    }

    auto matmul_trans_b = [](const std::vector<float>& input, const std::vector<float>& weight, int M, int K, int N) {
        std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += input[static_cast<size_t>(m * K + k)] * weight[static_cast<size_t>(n * K + k)];
                }
                out[static_cast<size_t>(m * N + n)] = sum;
            }
        }
        return out;
    };

    const std::vector<float> shared_gate = matmul_trans_b(moe_input, gate_weight, tokens, hidden_dim, intermediate_dim);
    const std::vector<float> shared_up = matmul_trans_b(moe_input, up_weight, tokens, hidden_dim, intermediate_dim);
    std::vector<float> shared_ffn_pre_gate(static_cast<size_t>(tokens * intermediate_dim), 0.0f);
    for (size_t i = 0; i < shared_ffn_pre_gate.size(); ++i) {
        const float g = shared_gate[i];
        shared_ffn_pre_gate[i] = (g / (1.0f + std::exp(-g))) * shared_up[i];
    }
    const std::vector<float> gated =
        ApplySharedScalarGateForTest(shared_ffn_pre_gate, shared_gate_logits_scalar, tokens, intermediate_dim);
    const std::vector<float> shared_down = matmul_trans_b(gated, down_weight, tokens, intermediate_dim, hidden_dim);
    std::vector<float> merged = routed_output;
    for (size_t i = 0; i < merged.size(); ++i) {
        merged[i] += shared_down[i];
    }
    return merged;
}
void CbSsmQwen35DeltaTest(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                          const struct ggml_tensor* c, int ith, int nth, void* userdata) {
    ::cb_ssm_qwen35_delta(dst, a, b, c, ith, nth, userdata);
}
struct ggml_tensor* SmartMulMatTest(struct ggml_context* ctx, struct ggml_tensor* weight, struct ggml_tensor* input,
                                    TransformerModel* model) {
    return ::smart_mul_mat(ctx, weight, input, model);
}
struct ggml_tensor* SmartMulMatWithPhaseTest(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input, TransformerModel* model, int phase) {
    InferenceWorkContext* previous_ctx = GetCurrentWorkContext();
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> owned_ctx(nullptr, DestroyInferenceWorkContext);
    if (!previous_ctx) {
        owned_ctx.reset(CreateInferenceWorkContext());
        SetCurrentWorkContext(owned_ctx.get());
    }
    const InferenceExecutionPhase previous_phase = GetCurrentExecutionPhase();
    SetCurrentExecutionPhase(static_cast<InferenceExecutionPhase>(phase));
    struct ggml_tensor* result = ::smart_mul_mat(ctx, weight, input, model);
    SetCurrentExecutionPhase(previous_phase);
    if (!previous_ctx) {
        SetCurrentWorkContext(nullptr);
    }
    return result;
}
int ResolveQuantBatchedTileColsForTest(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k) {
    return ::ResolveQuantBatchedTileCols(requested_cols, vec_dot_nrows, allow_true_batched_q4k);
}
bool ResolveQ4KTrueBatchedKernelPolicyForTest(int simd_level, bool compiled_with_sve) {
    return ::ResolveQ4KTrueBatchedKernelEnabledPolicy(static_cast<densecore::simd::SimdLevel>(simd_level),
                                                      compiled_with_sve);
}
float Gemma4GeluTanhExactForTest(float x) {
    return ::Gemma4GeluTanh(x);
}
bool ShouldUsePortableFlashHeadSeqReferenceFallbackForTest(bool explicit_debug_reference) {
    return ::ShouldUsePortableFlashHeadSeqReferenceFallback(explicit_debug_reference);
}
bool CompiledWithX86Avx512ForFlashAttentionForTest() {
    return ::CompiledWithX86Avx512ForFlashAttention();
}
int ResolvePagedAttentionDecodeHeadTileForTest(int n_head, int n_tokens, int n_tasks) {
    return ::ResolvePagedAttentionDecodeHeadTile(n_head, n_tokens, n_tasks);
}
uint64_t HashQwen36Q4KBatchedAdmissionKeyForTest(const TransformerModel* model, const ggml_tensor* weight,
                                                 const ggml_tensor* input, int M, int N, int K) {
    return ::HashQwen36Q4KBatchedAdmissionKey(model, weight, input, M, N, K);
}
void StoreQwen36Q4KBatchedAdmissionForTest(uint64_t key, bool pass, float max_abs_error, const char* reason) {
    const auto parsed_reason =
        (reason && std::strcmp(reason, "probe_mismatch") == 0)
            ? ::Qwen36PrefillQ4KBatchedRejectReason::ProbeMismatch
            : (pass ? ::Qwen36PrefillQ4KBatchedRejectReason::Admitted
                    : ::Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError);
    ::StoreQwen36Q4KBatchedAdmission(key,
                                     pass ? ::Qwen36Q4KBatchedAdmissionState::Pass
                                          : ::Qwen36Q4KBatchedAdmissionState::Reject,
                                     max_abs_error, parsed_reason);
}
int LookupQwen36Q4KBatchedAdmissionForTest(uint64_t key) {
    return static_cast<int>(::LookupQwen36Q4KBatchedAdmission(key).state);
}
void DowngradeQwen36Q4KBatchedAdmissionForTest(uint64_t key, float max_abs_error) {
    ::DowngradeQwen36Q4KBatchedAdmissionOnRuntimeFailure(
        key, max_abs_error, ::Qwen36PrefillQ4KBatchedRejectReason::ProbeInternalError, nullptr);
}
bool RunQwen35NativeMoEQ4KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols,
                                           float* output) {
    return ::Qwen35NativeMoEQ4KQ8KDotRow(weight_row, static_cast<const uint8_t*>(q8_input), cols, output);
}
bool RunQwen35NativeMoEQ5KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols,
                                           float* output) {
    return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, q8_input, cols, output);
}
bool RunQwen35NativeMoEQ5KFusedSwiGLURowsForTest(const void* gate_rows, const void* up_rows, const void* q8_input,
                                                 int64_t cols, int64_t row_count, size_t row_bytes,
                                                 float* output) {
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
bool Q4KRepackedGemvEnabledForTest(densecore::env::RuntimeToggleMode mode, int* reject_reason) {
    densecore::llm::config::FastPathRuntimeConfig config{};
    config.q4k_repacked_gemv = mode;
    Q4KRepackedGemvRejectReason reject = Q4KRepackedGemvRejectReason::None;
    const bool enabled = ::Q4KRepackedGemvEnabled(config, &reject);
    if (reject_reason) {
        *reject_reason = static_cast<int>(reject);
    }
    return enabled;
}
int ResolveQwen36PrefillQ4KBatchedReasonForTest(bool relevant, bool mode_off, bool lora_active,
                                                bool weight_is_q4k, bool shape_supported, bool kernel_available,
                                                bool has_vec_dot, bool candidate_ready, bool mode_on,
                                                bool mode_probe, int admission_state) {
    return static_cast<int>(::ResolveQwen36PrefillQ4KBatchedReason(
        relevant, mode_off, lora_active, weight_is_q4k, shape_supported, kernel_available, has_vec_dot,
        candidate_ready, mode_on, mode_probe, static_cast<::Qwen36Q4KBatchedAdmissionState>(admission_state)));
}
const char* Qwen36SSMQ8PrefillAMXRejectReasonNameForTest(int reason) {
    return ::Qwen36SSMQ8PrefillAMXRejectReasonName(reason);
}
int ResolveQwen36SSMQ8PrefillAMXReasonForTest(int mode, int phase, bool lora_active) {
    return static_cast<int>(::ResolveQwen36SSMQ8PrefillAMXReason(
        static_cast<densecore::llm::config::Qwen36SSMQ8PrefillAMXMode>(mode),
        static_cast<InferenceExecutionPhase>(phase), lora_active));
}
bool QActCacheSharedDataDifferentTensorMissesForTest() {
    InferenceWorkContext ctx{};
    ctx.execution_generation = 42;
    std::array<float, QK_K> values{};
    for (int i = 0; i < QK_K; ++i) {
        values[static_cast<size_t>(i)] = static_cast<float>(i) * 0.01f;
    }
    ggml_init_params params{16 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* t0 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    ggml_tensor* t1 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    if (!t0 || !t1) {
        ggml_free(ggml_ctx);
        return false;
    }
    t0->data = values.data();
    t1->data = values.data();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, QK_K);
    const uint8_t* first =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint64_t misses_after_first = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    const uint8_t* second =
        ::GetOrFillQuantizedActivationCache(&ctx, t1, t1->data, values.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint64_t misses_after_second = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    ggml_free(ggml_ctx);
    return first && second && misses_after_first == 1 && misses_after_second == 2;
}
bool QActCacheSameTensorDifferentTokenOrSlotMissesForTest(bool change_token_pos) {
    InferenceWorkContext ctx{};
    ctx.execution_generation = 42;
    std::array<float, QK_K> values{};
    for (int i = 0; i < QK_K; ++i) {
        values[static_cast<size_t>(i)] = static_cast<float>(i) * 0.02f;
    }
    ggml_init_params params{16 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* t0 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    if (!t0) {
        ggml_free(ggml_ctx);
        return false;
    }
    t0->data = values.data();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, QK_K);
    const uint8_t* first =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint64_t misses_after_first = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    const int second_slot = change_token_pos ? 1 : 2;
    const int64_t second_pos = change_token_pos ? 8 : 7;
    const uint8_t* second =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes,
                                            second_slot, second_pos, q8_traits);
    const uint64_t misses_after_second = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    ggml_free(ggml_ctx);
    return first && second && misses_after_first == 1 && misses_after_second == 2;
}
bool QActCacheResetAcrossCachedDecodeReuseForTest() {
    InferenceWorkContext ctx{};
    ctx.execution_generation = 42;
    std::array<float, QK_K> values{};
    ggml_init_params params{16 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* t0 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    if (!t0) {
        ggml_free(ggml_ctx);
        return false;
    }
    t0->data = values.data();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, QK_K);
    const uint8_t* first =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const bool filled = first && ctx.qact_tensor == t0 && ctx.qact_source == t0->data && !ctx.qact_buffer.empty();
    ::ResetCachedDecodeGraphWorkContext(&ctx);
    const bool reset = ctx.qact_tensor == nullptr && ctx.qact_source == nullptr && ctx.qact_buffer.empty() &&
                       ctx.qact_slot_id == -1;
    ggml_free(ggml_ctx);
    return filled && reset;
}

bool QActBatchedCacheReusesSameTensorForTest() {
    InferenceWorkContext ctx{};
    ctx.execution_generation = 42;
    constexpr int M = 4;
    constexpr int N = QK_K;
    std::vector<float> values(static_cast<size_t>(M) * static_cast<size_t>(N));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i % 127) * 0.003f;
    }
    std::vector<const float*> rows(static_cast<size_t>(M));
    for (int m = 0; m < M; ++m) {
        rows[static_cast<size_t>(m)] = values.data() + static_cast<size_t>(m) * static_cast<size_t>(N);
    }

    ggml_init_params params{64 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* t0 = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, N, M);
    if (!t0) {
        ggml_free(ggml_ctx);
        return false;
    }
    t0->data = values.data();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, N);
    const size_t q8_total_bytes = q8_row_bytes * static_cast<size_t>(M);
    const uint8_t* first = ::GetOrFillBatchedQuantizedActivationCache(
        &ctx, t0, t0->data, rows, M, N, GGML_TYPE_Q8_K, q8_row_bytes, q8_total_bytes, 7, q8_traits);
    const uint64_t misses_after_first = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    const uint8_t* second = ::GetOrFillBatchedQuantizedActivationCache(
        &ctx, t0, t0->data, rows, M, N, GGML_TYPE_Q8_K, q8_row_bytes, q8_total_bytes, 7, q8_traits);
    const uint64_t hits_after_second = ctx.qwen36_profile.qact_cache_hits.load(std::memory_order_relaxed);
    const uint64_t reused_after_second = ctx.qwen36_profile.qact_cache_reused_bytes.load(std::memory_order_relaxed);
    ggml_free(ggml_ctx);
    return first && second && first == second && misses_after_first == 1 && hits_after_second == 1 &&
           reused_after_second == q8_total_bytes;
}

bool QActCacheKeepsMultipleTensorEntriesForTest() {
    InferenceWorkContext ctx{};
    ctx.execution_generation = 42;
    std::array<float, QK_K> values0{};
    std::array<float, QK_K> values1{};
    for (int i = 0; i < QK_K; ++i) {
        values0[static_cast<size_t>(i)] = static_cast<float>(i) * 0.01f;
        values1[static_cast<size_t>(i)] = static_cast<float>(i) * -0.02f;
    }
    ggml_init_params params{16 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* t0 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    ggml_tensor* t1 = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, QK_K);
    if (!t0 || !t1) {
        ggml_free(ggml_ctx);
        return false;
    }
    t0->data = values0.data();
    t1->data = values1.data();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, QK_K);
    const uint8_t* first0 =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values0.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint8_t* first1 =
        ::GetOrFillQuantizedActivationCache(&ctx, t1, t1->data, values1.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint64_t misses_after_fill = ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    const uint8_t* second0 =
        ::GetOrFillQuantizedActivationCache(&ctx, t0, t0->data, values0.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint8_t* second1 =
        ::GetOrFillQuantizedActivationCache(&ctx, t1, t1->data, values1.data(), QK_K, GGML_TYPE_Q8_K, q8_bytes, 1, 7,
                                            q8_traits);
    const uint64_t hits_after_reuse = ctx.qwen36_profile.qact_cache_hits.load(std::memory_order_relaxed);
    ggml_free(ggml_ctx);
    return first0 && first1 && second0 && second1 && first0 == second0 && first1 == second1 &&
           misses_after_fill == 2 && hits_after_reuse == 2;
}

bool RunQwen36Q4KBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle,
                                      int* admission_state, int* reject_reason) {
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = false;
    if (admission_state) *admission_state = 0;
    if (reject_reason) *reject_reason = 0;
    constexpr int rows = 8;
    constexpr int cols = QK_K;
    constexpr int tokens = 2;
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!q4_traits || !q8_traits || !q4_traits->from_float || !q4_traits->vec_dot || !q8_traits->from_float) {
        return false;
    }
    ggml_init_params params{256 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor* weight = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_Q4_K, cols, rows);
    ggml_tensor* dst = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    if (!input || !weight || !dst || !input->data || !weight->data || !dst->data) {
        ggml_free(ggml_ctx);
        return false;
    }
    dst->src[0] = input;
    dst->src[1] = weight;
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.attn_qkv.weight");
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 31 + c) * 0.013f) * 0.75f;
        }
    }
    std::vector<float> weight_f32(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            weight_f32[static_cast<size_t>(r) * cols + c] =
                std::cos(static_cast<float>(r * 17 + c) * 0.021f) * 0.5f;
        }
        q4_traits->from_float(weight_f32.data() + static_cast<size_t>(r) * cols,
                              static_cast<uint8_t*>(weight->data) +
                                  static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q4_K, cols),
                              cols);
    }
    std::fill_n(reinterpret_cast<float*>(dst->data), rows * tokens, 12345.0f);
    constexpr uint64_t key = 0x9d360000ull;
    {
        std::lock_guard<std::mutex> lock(::Qwen36Q4KBatchedAdmissionMutex());
        ::Qwen36Q4KBatchedAdmissionMap().erase(key);
    }
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = cols;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = GGML_TYPE_Q4_K;
    ud.input_quant_type = GGML_TYPE_Q8_K;
    ud.quant_row_stride = densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_K, cols), static_cast<size_t>(64));
    ud.slot_id = -1;
    ud.qwen36_prefill_q4k_admission_key = key;
    ud.qwen36_prefill_q4k_probe = false;
    ud.qwen36_prefill_q4k_admitted = true;
    ud.require_q4k_true_batched = true;
    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    ud.work_ctx = &work_ctx;
    SetCurrentWorkContext(&work_ctx);
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    SetCurrentWorkContext(nullptr);
    std::vector<uint8_t> q8(static_cast<size_t>(tokens) * ud.quant_row_stride);
    for (int m = 0; m < tokens; ++m) {
        q8_traits->from_float(input_f32 + static_cast<size_t>(m) * cols,
                              q8.data() + static_cast<size_t>(m) * ud.quant_row_stride, cols);
    }
    bool matches = true;
    auto* out = reinterpret_cast<float*>(dst->data);
    for (int r = 0; r < rows; ++r) {
        const void* row_ptr = static_cast<const uint8_t*>(weight->data) +
                              static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q4_K, cols);
        for (int m = 0; m < tokens; ++m) {
            float ref = 0.0f;
            if (!densecore::hwy_kernels::DotQ4KQ8K_Hwy(row_ptr, q8.data() + static_cast<size_t>(m) * ud.quant_row_stride,
                                                       cols, &ref)) {
                matches = false;
                continue;
            }
            const float got = out[static_cast<size_t>(m) * rows + r];
            if (std::fabs(got - ref) > 1e-5f) {
                matches = false;
            }
        }
    }
    const auto value = ::LookupQwen36Q4KBatchedAdmission(key);
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = matches;
    if (admission_state) *admission_state = static_cast<int>(value.state);
    if (reject_reason) *reject_reason = static_cast<int>(value.reject_reason);
    ggml_free(ggml_ctx);
    return true;
}

static bool RunQwen36SSMQ8RepackedBatchedShapeForTest(int rows, int cols, int tokens, int nth,
                                                      bool exhaustive_oracle,
                                                      bool* output_matches_vecdot_oracle,
                                                      uint64_t* true_gemm_ops,
                                                      uint64_t* gemv_ops) {
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (rows <= 0 || cols <= 0 || tokens <= 0 || (cols % QK8_0) != 0 || (rows % 4) != 0 || !q8_traits ||
        !q8_traits->from_float || !q8_traits->vec_dot || q8_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }
    const size_t mem_size = std::max<size_t>(256 * 1024, ggml_row_size(GGML_TYPE_Q8_0, cols) *
                                                               static_cast<size_t>(rows) +
                                                               static_cast<size_t>(rows) *
                                                                   static_cast<size_t>(tokens) * sizeof(float) +
                                                               16 * 1024 * 1024);
    ggml_init_params params{mem_size, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor* weight = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_Q8_0, cols, rows);
    ggml_tensor* dst = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    if (!input || !weight || !dst || !input->data || !weight->data || !dst->data) {
        ggml_free(ggml_ctx);
        return false;
    }
    dst->src[0] = input;
    dst->src[1] = weight;
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.attn_qkv.weight");
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 29 + c) * 0.011f) * 0.5f;
        }
    }
    const auto weight_value = [](int row, int col) {
        return std::cos(static_cast<float>(row * 13 + col) * 0.017f) * 0.375f;
    };
    std::vector<float> weight_row(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            weight_row[static_cast<size_t>(c)] = weight_value(r, c);
        }
        q8_traits->from_float(weight_row.data(),
                              static_cast<uint8_t*>(weight->data) +
                                  static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q8_0, cols),
                              cols);
    }
    std::fill_n(reinterpret_cast<float*>(dst->data), rows * tokens, 12345.0f);
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = cols;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = GGML_TYPE_Q8_0;
    ud.input_quant_type = GGML_TYPE_Q8_0;
    ud.quant_row_stride = densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_0, cols), static_cast<size_t>(64));
    ud.slot_id = -1;
    ud.qwen36_ssm_q8_repacked_batched = true;
    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    ud.work_ctx = &work_ctx;
    SetCurrentWorkContext(&work_ctx);
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    SetCurrentWorkContext(nullptr);
    const auto snapshot = GetQwen36ProfileSnapshot(&work_ctx);
    if (true_gemm_ops) {
        *true_gemm_ops = snapshot.q8_batched_true_gemm_ops;
    }
    if (gemv_ops) {
        *gemv_ops = snapshot.q8_batched_gemv_ops;
    }

    std::vector<uint8_t> q8_input(static_cast<size_t>(tokens) * ud.quant_row_stride);
    for (int m = 0; m < tokens; ++m) {
        q8_traits->from_float(input_f32 + static_cast<size_t>(m) * cols,
                              q8_input.data() + static_cast<size_t>(m) * ud.quant_row_stride, cols);
    }
    bool matches = true;
    auto* out = reinterpret_cast<float*>(dst->data);
    std::vector<int> rows_to_check;
    if (exhaustive_oracle) {
        rows_to_check.reserve(static_cast<size_t>(rows));
        for (int r = 0; r < rows; ++r) {
            rows_to_check.push_back(r);
        }
    } else {
        const int samples[] = {0, 1, 127, rows / 2, rows - 2, rows - 1};
        for (int r : samples) {
            bool already_added = false;
            for (int existing : rows_to_check) {
                if (existing == r) {
                    already_added = true;
                    break;
                }
            }
            if (r >= 0 && r < rows && !already_added) {
                rows_to_check.push_back(r);
            }
        }
    }
    for (int r : rows_to_check) {
        const void* row_ptr = static_cast<const uint8_t*>(weight->data) +
                              static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q8_0, cols);
        for (int m = 0; m < tokens; ++m) {
            float ref = 0.0f;
            q8_traits->vec_dot(cols, &ref, 0, row_ptr, 0,
                               q8_input.data() + static_cast<size_t>(m) * ud.quant_row_stride, 0, 1);
            const float got = out[static_cast<size_t>(m) * rows + r];
            if (std::fabs(got - ref) > 1e-4f) {
                matches = false;
            }
        }
    }
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = matches;
    ggml_free(ggml_ctx);
    return true;
}

bool RunQwen36SSMQ8RepackedBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle) {
    return RunQwen36SSMQ8RepackedBatchedShapeForTest(
        128, QK8_0 * 4, 8, nth, /*exhaustive_oracle=*/true, output_matches_vecdot_oracle, nullptr, nullptr);
}

bool RunQwen36SSMQ8RepackedBatchedWideForTest(int nth, bool* output_matches_vecdot_oracle,
                                              uint64_t* true_gemm_ops, uint64_t* gemv_ops) {
    return RunQwen36SSMQ8RepackedBatchedShapeForTest(2048, QK8_0 * 384, 4, nth, /*exhaustive_oracle=*/false,
                                                     output_matches_vecdot_oracle, true_gemm_ops, gemv_ops);
}

int ResolveNativeMoEGraphCallbackTaskCountForTest(const TransformerModel* model, const BatchSpec* batch, int phase,
                                                  int64_t n_tokens, int top_k) {
    return ::ResolveNativeMoEGraphCallbackTaskCount(model, batch, static_cast<InferenceExecutionPhase>(phase), n_tokens,
                                                    top_k);
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
}  // namespace testing
}  // namespace densecore
#endif

void ResetMoEGraphWiringDebugCounter() {
    ::g_moe_graph_wiring_debug_counter.store(0, std::memory_order_relaxed);
}

uint64_t GetMoEGraphWiringDebugCounter() {
    return ::g_moe_graph_wiring_debug_counter.load(std::memory_order_relaxed);
}

void ResetMoECallbackEntryCounter() {
    ::ResetMoECallbackEntryCount();
}

uint64_t GetMoECallbackEntryCounter() {
    return ::GetMoECallbackEntryCount();
}

uint64_t GetMoECallbackMissingUserdataCounter() {
    return ::GetMoECallbackMissingUserdataCount();
}

uint64_t GetMoECallbackMissingBackendCounter() {
    return ::GetMoECallbackMissingBackendCount();
}

uint64_t GetMoECallbackMissingExpertsCounter() {
    return ::GetMoECallbackMissingExpertsCount();
}

uint64_t GetMoECallbackRoutingFailureCounter() {
    return ::GetMoECallbackRoutingFailureCount();
}

uint64_t GetMoECallbackEmptyRoutingCounter() {
    return ::GetMoECallbackEmptyRoutingCount();
}

uint64_t GetMoECallbackFailClosedCounter() {
    return ::GetMoECallbackFailClosedCount();
}

bool ConsumeMoEStrictFailure(std::string* message) {
    return ::ConsumeMoEStrictFailureState(message);
}

void ResetMoEStrictFailure() {
    ::ResetMoEStrictFailureState();
}

namespace densecore::testing {
namespace {
std::vector<float> ApplyWeightedRmsNormVectorForTest(const std::vector<float>& src, const std::vector<float>& weight,
                                                     float eps) {
    if (src.empty()) {
        return {};
    }
    std::vector<float> out(src.size(), 0.0f);
    float sum_sq = 0.0f;
    for (float v : src) {
        sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(src.size()) + eps);
    for (size_t i = 0; i < src.size(); ++i) {
        const float w = i < weight.size() ? weight[i] : 1.0f;
        out[i] = src[i] * inv_rms * w;
    }
    return out;
}
}  // namespace

Gemma4MoEBranchInputsSnapshot ComputeGemma4MoEBranchInputsForTest(const std::vector<float>& attn_post_residual,
                                                                  const std::vector<float>& inp_ff,
                                                                  const std::vector<float>& ffn_norm_weight,
                                                                  const std::vector<float>& pre_moe_norm_weight,
                                                                  float eps) {
    Gemma4MoEBranchInputsSnapshot snapshot;
    snapshot.shared_input = ApplyWeightedRmsNormVectorForTest(attn_post_residual, ffn_norm_weight, eps);
    snapshot.routed_input = pre_moe_norm_weight.empty()
                                ? snapshot.shared_input
                                : ApplyWeightedRmsNormVectorForTest(inp_ff, pre_moe_norm_weight, eps);
    return snapshot;
}

bool ShouldRunMoESharedDenseBranchForTest(const TransformerModel* model, bool is_gemma4_moe,
                                          const struct ggml_tensor* ffn_gate, const struct ggml_tensor* ffn_up,
                                          const struct ggml_tensor* ffn_down) {
    return ShouldRunMoESharedDenseBranch(model, nullptr, is_gemma4_moe, ffn_gate, ffn_up, ffn_down);
}
}  // namespace densecore::testing
