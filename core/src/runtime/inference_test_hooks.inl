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
                float score = dot * scale;
                if (logit_softcap > 0.0f && std::isfinite(score)) {
                    score = std::tanh(score / logit_softcap) * logit_softcap;
                }
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
int ResolveQuantBatchedTileColsForTest(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k) {
    return ::ResolveQuantBatchedTileCols(requested_cols, vec_dot_nrows, allow_true_batched_q4k);
}
bool ResolveQ4KTrueBatchedKernelPolicyForTest(int mode, int simd_level, bool compiled_with_sve) {
    return ::ResolveQ4KTrueBatchedKernelEnabledPolicy(
        static_cast<RuntimeToggleMode>(mode), static_cast<densecore::simd::SimdLevel>(simd_level), compiled_with_sve);
}
int ResolveQwen36MoECallbackTaskCountForTest(const TransformerModel* model, const BatchSpec* batch, int top_k) {
    return ::ResolveQwen36MoECallbackTaskCount(model, batch, top_k);
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
