#ifdef DENSECORE_TEST_BUILD
#include "llm/runtime/work_context_internal.h"
#include "llm/matmul/quant_cache.h"
#include "llm/matmul/q8_kernels.h"
namespace densecore {
namespace testing {

ggml_tensor* BuildHalMatmulForDependencyTest(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input) {
    return ggml_mul_mat_hal(ctx, weight, input, densecore::DeviceType::CPU);
}

ggml_tensor* BuildFp8MatmulForExtractionTest(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                                          const TransformerModel::FP8WeightBinding& binding) {
    return ggml_mul_mat_fp8(ctx, weight, input, binding);
}

int ModelVariantAfterWorkContextResetForTest(int kind) {
    const std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)> work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    SetInferenceWorkContextModelVariant(work.get(), ModelVariant::QWEN36);
    if (kind == 0) ResetQwen36Profile(work.get());
    else if (kind == 1) ResetCachedDecodeGraphWorkContext(work.get());
    else ResetInferenceWorkContext(work.get());
    return static_cast<int>(GetInferenceWorkContextModelVariant(work.get()));
}





bool NativeM4AvailableForTest(bool q6) {
    return q6 ? Q6KQ8KM4NativeAvailable() : Q8SmallBatchDot4Supported();
}

bool RunNativeM4CallbackForTest(bool q6, int tokens, int phase, bool reference_only,
                               int threads, bool* exact, uint64_t* used, bool qwen36, bool padded) {
    if (!exact || !used || tokens < 1 || tokens > 4 || threads < 1) return false;
    ggml_cpu_init();
    constexpr int reduction = 2048, rows = 37;
    const ggml_type weight_type = q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
    const auto* weights = ggml_get_type_traits_cpu(weight_type);
    const auto* inputs = ggml_get_type_traits_cpu(weights->vec_dot_type);
    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    if (!ctx) return false;
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ctx, ggml_free);
    auto* weight = ggml_new_tensor_2d(ctx, weight_type, reduction, rows);
    auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, reduction + (padded ? 16 : 0), tokens);
    input->ne[0] = reduction;
    auto* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows + (padded ? 7 : 0), tokens);
    output->ne[0] = rows;
    output->src[0] = input;
    output->src[1] = weight;
    std::vector<float> data(reduction);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < reduction; ++c) data[c] = std::sin((r * 17 + c) * 0.13f);
        weights->from_float(data.data(), static_cast<char*>(weight->data) + r * weight->nb[1], reduction);
    }
    const size_t input_stride = input->nb[1] / sizeof(float);
    const size_t output_stride = output->nb[1] / sizeof(float);
    std::fill_n(static_cast<float*>(output->data), output_stride * tokens, 12345.0f);
    for (int m = 0; m < tokens; ++m)
        for (int i = 0; i < reduction; ++i)
            static_cast<float*>(input->data)[m * input_stride + i] = std::cos((m * reduction + i) * 0.031f);
    InferenceWorkContext context{};
    context.model_variant = qwen36 ? ModelVariant::QWEN36 : ModelVariant::QWEN35;
    context.phase = static_cast<InferenceExecutionPhase>(phase);
    if (context.phase == InferenceExecutionPhase::Decode) ResetCachedDecodeGraphWorkContext(&context);
    else if (context.phase == InferenceExecutionPhase::Prefill) ResetCachedPrefillGraphWorkContext(&context);
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = reduction;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = weight_type;
    ud.input_quant_type = weights->vec_dot_type;
    ud.quant_row_stride = densecore::AlignUp(ggml_row_size(weights->vec_dot_type, reduction), size_t(64));
    ud.work_ctx = &context;
    ud.disable_quant_nrc_fast = reference_only;
    const auto before = BatchedDecodeNativeM4OpsForTest();
    // Real secondary workers deliberately have no inference TLS context.
    std::vector<std::thread> workers;
    for (int ith = 0; ith < threads; ++ith) {
        workers.emplace_back([&, ith] { cb_gemv_batched_custom(output, ith, threads, &ud); });
    }
    for (auto& worker : workers) worker.join();
    *used = BatchedDecodeNativeM4OpsForTest() - before;
    if (*used && context.qwen36_profile.q4k_true_batched_used.load() != 0) return false;
    *exact = true;
    std::vector<uint8_t> quantized(ud.quant_row_stride);
    for (int m = 0; m < tokens; ++m) {
        inputs->from_float(static_cast<float*>(input->data) + m * input_stride, quantized.data(), reduction);
        for (int r = 0; r < rows; ++r) {
            float expected = 0;
            weights->vec_dot(reduction, &expected, 0, static_cast<char*>(weight->data) + r * weight->nb[1],
                             0, quantized.data(), 0, 1);
            const float actual = static_cast<float*>(output->data)[m * output_stride + r];
            // The new AVX2 kernels preserve bitwise order. Existing Arm nrc=2
            // fallbacks have a different reduction order and retain their tolerance.
            const bool differs = NativeM4AvailableForTest(q6)
                ? std::memcmp(&expected, &actual, sizeof(float)) != 0
                : std::fabs(expected - actual) > 1e-4f + 2e-6f * std::fabs(expected);
            if (!std::isfinite(actual) || !std::isfinite(expected) || differs) *exact = false;
        }
    }
    for (int m = 0; m < tokens; ++m)
        for (size_t r = rows; r < output_stride; ++r)
            if (static_cast<float*>(output->data)[m * output_stride + r] != 12345.0f) *exact = false;
    return true;
}



// Compare the prefill worker and a GGML secondary worker with no TLS context.







bool RunQuantBatchedColumnsForTest(ggml_type type, int tokens) {
    ggml_cpu_init();
    constexpr int rows = 33, cols = QK_K;
    const auto* traits = ggml_get_type_traits_cpu(type);
    const auto* activation = ggml_get_type_traits_cpu(traits->vec_dot_type);
    ggml_context* ctx = ggml_init({512 * 1024, nullptr, false});
    if (!ctx) return false;
    auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
    auto* weight = ggml_new_tensor_2d(ctx, type, cols, rows);
    auto* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    std::vector<float> guarded_output(rows * tokens + 16, 12345.0f);
    output->data = guarded_output.data() + 8;
    output->src[0] = input;
    output->src[1] = weight;
    std::vector<float> row(cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) row[c] = std::cos((r * 13 + c) * 0.021f);
        traits->from_float(row.data(), static_cast<uint8_t*>(weight->data) + r * weight->nb[1], cols);
    }
    auto* values = static_cast<float*>(input->data);
    for (int i = 0; i < cols * tokens; ++i) values[i] = std::sin(i * 0.017f);
    std::fill_n(static_cast<float*>(output->data), rows * tokens, 12345.0f);
    auto work = std::make_unique<InferenceWorkContext>();
    SetInferenceWorkContextPhase(work.get(), InferenceExecutionPhase::Decode);
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = cols;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = type;
    ud.input_quant_type = traits->vec_dot_type;
    ud.work_ctx = work.get();
    std::thread first([&] { cb_gemv_batched_custom(output, 0, 2, &ud); });
    std::thread second([&] { cb_gemv_batched_custom(output, 1, 2, &ud); });
    first.join();
    second.join();
    std::vector<uint8_t> quantized(ggml_row_size(traits->vec_dot_type, cols));
    bool matches = true;
    for (int m = 0; m < tokens; ++m) {
        activation->from_float(values + m * cols, quantized.data(), cols);
        for (int r = 0; r < rows; ++r) {
            float reference = 0;
            traits->vec_dot(cols, &reference, 0, static_cast<uint8_t*>(weight->data) + r * weight->nb[1],
                            0, quantized.data(), 0, 1);
            const float got = static_cast<float*>(output->data)[m * rows + r];
            matches = matches && std::isfinite(got) && std::fabs(got - reference) <= 1e-4f;
        }
    }
    for (int i = 0; i < 8; ++i) {
        matches = matches && guarded_output[i] == 12345.0f &&
                  guarded_output[rows * tokens + 8 + i] == 12345.0f;
    }
    ggml_free(ctx);
    return matches;
}

static int RunQ4KSharedBatchedPackCaseForTest(int tokens, bool map3) {
    ggml_cpu_init();
    if (!densecore::kernels::Q4KRepackedGemvIsaSupported() ||
        !densecore::kernels::Q4KRealPackedGemvKernelAvailable()) return -1;
    constexpr int rows = 32, cols = QK_K;
    const auto* q4 = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ggml_init_params params{512 * 1024, nullptr, false};
    ggml_context* graph = ggml_init(params);
    if (!graph) return 0;
    auto* input = ggml_new_tensor_2d(graph, GGML_TYPE_F32, cols, tokens);
    auto* weight = ggml_new_tensor_2d(graph, GGML_TYPE_Q4_K, cols, rows);
    auto* output = ggml_new_tensor_2d(graph, GGML_TYPE_F32, rows, tokens);
    output->src[0] = input;
    output->src[1] = weight;
    std::vector<float> floats(cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) floats[c] = std::cos((r * 13 + c) * 0.021f) * 0.25f;
        q4->from_float(floats.data(), static_cast<uint8_t*>(weight->data) + r * weight->nb[1], cols);
    }
    auto work = std::make_unique<InferenceWorkContext>();
    densecore::llm::config::FastPathRuntimeConfig config{};
    config.q4k_repacked_gemv = densecore::env::RuntimeToggleMode::On;
    InferenceDependencies deps{};
    deps.fast_path_config = &config;
    BatchSpec batch{};
    batch.deps = &deps;
    // Distinct graph operations may consume the same scratch address in one execution.
    GemvBatchedUserData operations[3]{};
    for (auto& ud : operations) {
        ud.weight_tensor = weight;
        ud.N = cols;
        ud.K = rows;
        ud.M = tokens;
        ud.weight_type = GGML_TYPE_Q4_K;
        ud.input_quant_type = GGML_TYPE_Q8_K;
        ud.work_ctx = work.get();
    }
    const size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    std::vector<uint8_t> quantized(q8_bytes);
    bool matches = true;
    for (int execution = 0; execution < 3; ++execution) {
        if (!map3 || execution == 0) ResetInferenceWorkContext(work.get());
        auto& ud = operations[map3 ? execution : 0];
        SetInferenceWorkContextPhase(work.get(), InferenceExecutionPhase::Decode);
        BindInferenceWorkContextBatch(work.get(), &batch);
        auto* values = static_cast<float*>(input->data);
        for (int i = 0; i < cols * tokens; ++i) values[i] = std::sin((i + execution * 137) * 0.017f);
        std::fill_n(static_cast<float*>(output->data), rows * tokens, 12345.0f);
        const auto run_worker = [&](int ith) {
            if (map3) {
                cb_gemv_batched_custom_map3(output, output, input, weight, ith, 2, &ud);
            } else {
                cb_gemv_batched_custom(output, ith, 2, &ud);
            }
        };
        const auto used_before = (*GetInferenceWorkContextProfile(work.get())).q4k_repacked_gemv_used_ops.load();
        const auto fills_before = GetMatmulWorkState(work.get()).q8k_batched_pack_cache.SuccessfulFillCountForTest();
        std::thread first([&] { run_worker(0); });
        std::thread second([&] { run_worker(1); });
        first.join();
        second.join();
        matches = matches && (*GetInferenceWorkContextProfile(work.get())).q4k_repacked_gemv_used_ops.load() > used_before;
        // Every operation packs once for all workers. The next operation consumes
        // changed values at the same address and must produce fresh numeric output.
        matches = matches && GetMatmulWorkState(work.get()).q8k_batched_pack_cache.SuccessfulFillCountForTest() == fills_before + 1;
        for (int m = 0; m < tokens; ++m) {
            q8->from_float(values + m * cols, quantized.data(), cols);
            for (int r = 0; r < rows; ++r) {
                float reference = 0;
                q4->vec_dot(cols, &reference, 0, static_cast<uint8_t*>(weight->data) + r * weight->nb[1],
                            0, quantized.data(), 0, 1);
                const float got = static_cast<float*>(output->data)[m * rows + r];
                matches = matches && std::isfinite(got) && std::fabs(got - reference) <= 1e-4f;
            }
        }
    }
    ggml_free(graph);
    return matches ? 1 : 0;
}

int RunQ4KSharedBatchedPackForTest(int tokens) {
    return RunQ4KSharedBatchedPackCaseForTest(tokens, false);
}

int RunQ4KMap3SharedBatchedPackForTest(int tokens) {
    return RunQ4KSharedBatchedPackCaseForTest(tokens, true);
}



namespace {
class ScopedAttentionCaptureGuard {
public:
    ScopedAttentionCaptureGuard(std::vector<float>* out, int layer)
        : previous_(ExchangeAttentionCaptureForTest({out, layer})) {}
    ~ScopedAttentionCaptureGuard() { ExchangeAttentionCaptureForTest(previous_); }
private:
    AttentionCaptureState previous_;

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

bool ShouldUsePrefillPerSequenceLastLogitsForTest(const TransformerModel* model, int num_seqs, int n_tokens) {
    BatchSpec batch{};
    batch.num_seqs = num_seqs;
    for (int token = 0; token < n_tokens; ++token) {
        batch.seq_id.push_back(token % std::max(1, num_seqs));
    }
    return ::ShouldUsePrefillPerSequenceLastLogits(model, batch, n_tokens);
}

int ResolveSsmConvTaskCountForTest(int variant, int num_seqs, int n_tokens, int conv_channels, int num_threads,
                                   bool force_serial_multi_seq) {
    return ::ResolveSsmConvTaskCount(static_cast<ModelVariant>(variant), num_seqs, n_tokens, conv_channels,
                                     num_threads, force_serial_multi_seq);
}

int ResolveSsmDeltaTaskCountForTest(int variant, int num_seqs, int num_v_heads, int num_threads,
                                    bool force_serial_multi_seq) {
    return ::ResolveSsmDeltaTaskCount(static_cast<ModelVariant>(variant), num_seqs, num_v_heads, num_threads,
                                      force_serial_multi_seq);
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
    densecore::llm::graph::detail::SetFlashAttentionDisabledForGraphTest(disabled);
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
    return static_cast<int>(densecore::llm::config::LoadArmQ4KNativeVecDotMode());
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
    InferenceWorkContext* previous_ctx = GetCurrentWorkContext();
    std::unique_ptr<InferenceWorkContext, void (*)(InferenceWorkContext*)> owned_ctx(nullptr, DestroyInferenceWorkContext);
    if (!previous_ctx) {
        owned_ctx.reset(CreateInferenceWorkContext());
        SetCurrentWorkContext(owned_ctx.get());
    }
    struct ggml_tensor* result = ::smart_mul_mat(ctx, weight, input, model);
    if (!previous_ctx) {
        SetCurrentWorkContext(nullptr);
    }
    return result;
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

bool RunQwen35NativeMoEQ5KQ8KDotRowForTest(const void* weight_row, const void* q8_input, int64_t cols,
                                           float* output) {
    return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, q8_input, cols, output);
}



bool RunQ5KQ8KBatchedGemvRowForTest(const void* weight_row, const void* q8_input_base, size_t q8_row_stride, int M,
                                    int cols, float* output) {
    return ::ComputeQ5KQ8KBatchedRow(weight_row, static_cast<const uint8_t*>(q8_input_base), q8_row_stride, M, cols,
                                     output);
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
    const bool filled = first && ctx.matmul.qact_tensor == t0 && ctx.matmul.qact_source == t0->data && !ctx.matmul.qact_buffer.empty();
    ::ResetCachedDecodeGraphWorkContext(&ctx);
    const bool reset = ctx.matmul.qact_tensor == nullptr && ctx.matmul.qact_source == nullptr && ctx.matmul.qact_buffer.empty() &&
                       ctx.matmul.qact_slot_id == -1;
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

static bool RunQwen36Q4KBatchedDirectImplForTest(int nth, bool* output_matches_vecdot_oracle,
                                                 int* admission_state, int* reject_reason,
                                                 std::vector<MatmulDispatchCensusEntry>* dispatch_entries) {
    ggml_cpu_init();
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = false;
    if (admission_state) *admission_state = 0;
    if (reject_reason) *reject_reason = 0;
    if (dispatch_entries) dispatch_entries->clear();
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
    BatchSpec batch{};
    batch.num_seqs = tokens;
    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    work_ctx.batch = &batch;
    ud.work_ctx = &work_ctx;
    SetCurrentWorkContext(&work_ctx);
    const InferenceExecutionPhase previous_phase = GetCurrentExecutionPhase();
    SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    SetCurrentExecutionPhase(previous_phase);
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
    if (dispatch_entries) {
        *dispatch_entries = GetQwen36ProfileSnapshot(&work_ctx).matmul_dispatch_top_slow_entries;
    }
    ggml_free(ggml_ctx);
    return true;
}

bool RunQwen36Q4KBatchedDirectForTest(int nth, bool* output_matches_vecdot_oracle,
                                      int* admission_state, int* reject_reason) {
    return RunQwen36Q4KBatchedDirectImplForTest(nth, output_matches_vecdot_oracle, admission_state, reject_reason,
                                                nullptr);
}

bool RunQwen36Q4KBatchedDirectDispatchCensusForTest(int nth, std::vector<MatmulDispatchCensusEntry>* dispatch_entries,
                                                    bool* output_matches_vecdot_oracle) {
    return RunQwen36Q4KBatchedDirectImplForTest(nth, output_matches_vecdot_oracle, nullptr, nullptr,
                                                dispatch_entries);
}

bool RunGemma4SafeQ4KPrefillBatchedUsesQActCacheForTest(int nth, bool* output_matches_vecdot_oracle,
                                                        uint64_t* qact_hits, uint64_t* qact_misses) {
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = false;
    if (qact_hits) *qact_hits = 0;
    if (qact_misses) *qact_misses = 0;
    constexpr int rows = 8;
    constexpr int cols = QK_K;
    constexpr int tokens = 20;
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!q4_traits || !q8_traits || !q4_traits->from_float || !q8_traits->from_float) {
        return false;
    }
    ggml_init_params params{512 * 1024, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor* weight0 = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_Q4_K, cols, rows);
    ggml_tensor* weight1 = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_Q4_K, cols, rows);
    ggml_tensor* dst0 = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    ggml_tensor* dst1 = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    if (!input || !weight0 || !weight1 || !dst0 || !dst1 || !input->data || !weight0->data || !weight1->data ||
        !dst0->data || !dst1->data) {
        ggml_free(ggml_ctx);
        return false;
    }
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 23 + c) * 0.017f) * 0.5f;
        }
    }
    std::vector<float> weight_f32(static_cast<size_t>(rows) * cols);
    auto fill_weight = [&](ggml_tensor* weight, int seed) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                weight_f32[static_cast<size_t>(r) * cols + c] =
                    std::cos(static_cast<float>(seed + r * 19 + c) * 0.019f) * 0.375f;
            }
            q4_traits->from_float(weight_f32.data() + static_cast<size_t>(r) * cols,
                                  static_cast<uint8_t*>(weight->data) +
                                      static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q4_K, cols),
                                  cols);
        }
    };
    fill_weight(weight0, 3);
    fill_weight(weight1, 47);
    std::snprintf(weight0->name, sizeof(weight0->name), "blk.0.ffn_gate.weight");
    std::snprintf(weight1->name, sizeof(weight1->name), "blk.0.ffn_up.weight");
    dst0->src[0] = input;
    dst0->src[1] = weight0;
    dst1->src[0] = input;
    dst1->src[1] = weight1;
    std::fill_n(reinterpret_cast<float*>(dst0->data), rows * tokens, 12345.0f);
    std::fill_n(reinterpret_cast<float*>(dst1->data), rows * tokens, 12345.0f);

    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    SetCurrentWorkContext(&work_ctx);
    std::atomic<uint64_t> quantized_stamp{0};
    std::vector<uint8_t> shared_quant(static_cast<size_t>(tokens) *
                                      densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_K, cols), static_cast<size_t>(64)));
    auto run_weight = [&](ggml_tensor* weight, ggml_tensor* dst) {
        GemvBatchedUserData ud{};
        ud.weight_tensor = weight;
        ud.N = cols;
        ud.K = rows;
        ud.M = tokens;
        ud.weight_type = GGML_TYPE_Q4_K;
        ud.input_quant_type = GGML_TYPE_Q8_K;
        ud.quant_row_stride = densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_K, cols), static_cast<size_t>(64));
        ud.slot_id = 0;
        ud.quant_input_shared = shared_quant.data();
        ud.quantized_stamp = &quantized_stamp;
        ud.gemma4_prefill_safe_batched = true;
        ud.work_ctx = &work_ctx;
        for (int ith = 0; ith < std::max(1, nth); ++ith) {
            cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
        }
    };
    run_weight(weight0, dst0);
    run_weight(weight1, dst1);
    SetCurrentWorkContext(nullptr);

    std::vector<uint8_t> q8(static_cast<size_t>(tokens) *
                            densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_K, cols), static_cast<size_t>(64)));
    const size_t q8_stride = densecore::AlignUp(ggml_row_size(GGML_TYPE_Q8_K, cols), static_cast<size_t>(64));
    for (int m = 0; m < tokens; ++m) {
        q8_traits->from_float(input_f32 + static_cast<size_t>(m) * cols,
                              q8.data() + static_cast<size_t>(m) * q8_stride, cols);
    }
    auto matches_output = [&](const ggml_tensor* weight, const ggml_tensor* dst) {
        bool matches = true;
        const auto* out = reinterpret_cast<const float*>(dst->data);
        for (int r = 0; r < rows; ++r) {
            const void* row_ptr = static_cast<const uint8_t*>(weight->data) +
                                  static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q4_K, cols);
            for (int m = 0; m < tokens; ++m) {
                float ref = 0.0f;
                if (!densecore::hwy_kernels::DotQ4KQ8K_Hwy(row_ptr, q8.data() + static_cast<size_t>(m) * q8_stride,
                                                           cols, &ref)) {
                    matches = false;
                    continue;
                }
                const float got = out[static_cast<size_t>(m) * rows + r];
                matches = matches && std::fabs(got - ref) <= 1e-5f;
            }
        }
        return matches;
    };
    if (output_matches_vecdot_oracle) {
        *output_matches_vecdot_oracle = matches_output(weight0, dst0) && matches_output(weight1, dst1);
    }
    if (qact_hits) {
        *qact_hits = work_ctx.qwen36_profile.qact_cache_hits.load(std::memory_order_relaxed);
    }
    if (qact_misses) {
        *qact_misses = work_ctx.qwen36_profile.qact_cache_misses.load(std::memory_order_relaxed);
    }
    ggml_free(ggml_ctx);
    return true;
}

bool ArmQwen36LargeQ8PrefillAdmittedForTest(bool relevant, int tokens, int reduction, int rows) {
    return ShouldUseArmQwen36LargeQ8Prefill(relevant, tokens, reduction, rows);
}

static bool RunQwenQ8ProjectionGraphForTest(int tokens, bool fused, ModelVariant variant, InferenceExecutionPhase phase, bool* owned_callback,
                                 bool* output_matches, uint64_t* true_gemm_ops) {
    ggml_cpu_init();
    *owned_callback = false;
    *output_matches = false;
    *true_gemm_ops = 0;
    constexpr int reduction = 2048;
    const int rows = fused ? 12288 : 8192;
    if (tokens <= 0 || tokens > 390) return false;
    ggml_context* ctx = ggml_init({256ULL * 1024 * 1024, nullptr, false});
    if (!ctx) return false;
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> context_owner(ctx, ggml_free);
    auto* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, reduction, tokens);
    auto* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, reduction, rows);
    ggml_set_name(weight, fused ? "blk.0.ssm_qkv_gate.cpu_fused_2d" : "blk.0.attn_qkv.weight");
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::vector<float> row(reduction);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < reduction; ++c) row[c] = std::sin((r * 7 + c) * 0.031f) * 0.125f;
        traits->from_float(row.data(), static_cast<uint8_t*>(weight->data) + r * weight->nb[1], reduction);
    }
    auto* values = static_cast<float*>(input->data);
    for (int i = 0; i < tokens * reduction; ++i) values[i] = std::cos(i * 0.013f) * 0.25f;
    TransformerModel model{};
    model.arch = ModelArch::QWEN35;
    model.variant = variant;
    model.arch_flags.is_hybrid_ssm = true;
    model.hparams.n_embd = reduction;
    if (fused) {
        // The loader-created alias name does not encode an attn_qkv role.
        // Inherit the layer's explicit role, as the loader now supplies it.
        model.layers.resize(1);
        model.layers[0].Set("attn_qkv_gate.cpu_fused_decode", weight,
                            densecore::runtime::DenseCoreTensorRole::HybridSSMQkv);
        const auto requirement = densecore::models::ResolveModelTensorExecutionRequirement(
            &model, weight, weight->name, false, weight->type);
        if (requirement.tensor_role != densecore::runtime::DenseCoreTensorRole::HybridSSMQkv) {
            return false;
        }
    }
    InferenceConfig config{};
    config.num_threads = 16;
    config.prefill_threads = 16;
    InferenceDependencies deps{};
    deps.config = &config;
    BatchSpec batch{};
    batch.deps = &deps;
    batch.num_seqs = phase == InferenceExecutionPhase::Decode ? tokens : 1;
    batch.n_past.assign(batch.num_seqs, phase == InferenceExecutionPhase::Decode ? 32 : 0);
    batch.tokens.assign(tokens, 1);
    batch.seq_id.assign(tokens, 0);
    batch.pos.resize(tokens);
    for (int i = 0; i < tokens; ++i) {
        batch.pos[i] = phase == InferenceExecutionPhase::Decode ? 32 : i;
        batch.seq_id[i] = phase == InferenceExecutionPhase::Decode ? i : 0;
    }
    auto work = std::unique_ptr<InferenceWorkContext, decltype(&DestroyInferenceWorkContext)>(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    bool success = false;
    {
        struct RestoreWorkContext {
            InferenceWorkContext* previous;
            ~RestoreWorkContext() { SetCurrentWorkContext(previous); }
        } restore{GetCurrentWorkContext()};
        ResetInferenceWorkContext(work.get());
        BindInferenceWorkContextBatch(work.get(), &batch);
        SetInferenceWorkContextPhase(work.get(), phase);
        SetInferenceWorkContextModelVariant(work.get(), variant);
        SetCurrentWorkContext(work.get());
        auto* candidate = smart_mul_mat(ctx, weight, input, &model);
        auto* reference = ggml_mul_mat(ctx, weight, input);
        *owned_callback = candidate->op == GGML_OP_CUSTOM || candidate->op == GGML_OP_MAP_CUSTOM3;
        auto* reference_graph = ggml_new_graph_custom(ctx, 16, false);
        auto* candidate_graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(reference_graph, reference);
        ggml_build_forward_expand(candidate_graph, candidate);
        // Both execute from F32 input, so activation quantization/packing is
        // exercised inside the graph rather than supplied by the fixture.
        success = ggml_graph_compute_with_ctx(ctx, reference_graph, 16) == GGML_STATUS_SUCCESS &&
                  ggml_graph_compute_with_ctx(ctx, candidate_graph, 16) == GGML_STATUS_SUCCESS;
        bool matches = success;
        const auto* expected = static_cast<const float*>(reference->data);
        const auto* actual = static_cast<const float*>(candidate->data);
        for (int i = 0; i < tokens * rows; ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                std::fabs(actual[i] - expected[i]) > 1e-4f + 2e-6f * std::fabs(expected[i])) {
                matches = false;
                break;
            }
        }
        *output_matches = matches;
        *true_gemm_ops = (*GetInferenceWorkContextProfile(work.get())).q8_batched_true_gemm_ops.load();
    }
    return success;
}

bool RunQwenQ8PrefillGraphForTest(int tokens, bool fused, ModelVariant variant, bool* owned_callback,
                                 bool* output_matches, uint64_t* true_gemm_ops) {
    return RunQwenQ8ProjectionGraphForTest(tokens, fused, variant, InferenceExecutionPhase::Prefill,
                                         owned_callback, output_matches, true_gemm_ops);
}

bool BatchedDecodeRowMajorPolicyForTest(ModelVariant variant, int phase, int tokens, int override_value) {
    return ShouldUseBatchedDecodeRowMajor(variant, static_cast<InferenceExecutionPhase>(phase), tokens, override_value);
}

bool RunQwenRowMajorGraphForTest(int tokens, bool fused, ModelVariant variant, int phase,
                                bool* matches, uint64_t* row_major_ops) {
    const auto before = BatchedDecodeRowMajorOpsForTest();
    bool owned = false;
    uint64_t gemm_ops = 0;
    const bool ok = RunQwenQ8ProjectionGraphForTest(tokens, fused, variant,
        static_cast<InferenceExecutionPhase>(phase), &owned, matches, &gemm_ops);
    *row_major_ops = BatchedDecodeRowMajorOpsForTest() - before;
    return ok;
}

bool RunQwenNativeM4GraphForTest(bool fused, bool* matches, uint64_t* ops) {
    const auto before = BatchedDecodeNativeM4OpsForTest();
    bool owned = false;
    uint64_t gemm_ops = 0;
    const bool ok = RunQwenQ8ProjectionGraphForTest(4, fused, ModelVariant::QWEN36,
        InferenceExecutionPhase::Decode, &owned, matches, &gemm_ops);
    *ops = BatchedDecodeNativeM4OpsForTest() - before;
    return ok && owned;
}

static bool RunQwen36SSMQ8RepackedBatchedShapeForTest(int rows, int cols, int tokens, int nth,
                                                      bool exhaustive_oracle,
                                                      bool* output_matches_vecdot_oracle,
                                                      uint64_t* true_gemm_ops,
                                                      uint64_t* gemv_ops) {
    // Standalone filtered tests must initialize the CPU fp16 lookup tables used
    // by the vec-dot oracle, independently of earlier backend tests.
    ggml_cpu_init();
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
            if (!std::isfinite(got) || !std::isfinite(ref) || std::fabs(got - ref) > 1e-4f) {
                std::fprintf(stderr,
                             "Q8 batched oracle mismatch: rows=%d cols=%d tokens=%d nth=%d row=%d token=%d "
                             "actual=%.9g expected=%.9g abs_error=%.9g\n",
                             rows, cols, tokens, nth, r, m, got, ref, std::fabs(got - ref));
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

bool RunQwen36SSMQ8RepackedBatchedC4ProjectionForTest(int nth, bool* output_matches_vecdot_oracle,
                                                      uint64_t* true_gemm_ops, uint64_t* gemv_ops) {
    return RunQwen36SSMQ8RepackedBatchedShapeForTest(2048, 2048, 4, nth, /*exhaustive_oracle=*/false,
                                                     output_matches_vecdot_oracle, true_gemm_ops, gemv_ops);
}

bool RunDenseCoreQ8RepackedGemvForTest(bool* output_matches_vecdot_oracle) {
    ggml_cpu_init();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    constexpr int rows = 64;
    constexpr int cols = QK8_0 * 64;
    if (!q8_traits || !q8_traits->from_float || !q8_traits->vec_dot || q8_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }
    ggml_init_params params{8 << 20, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* weight = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_Q8_0, cols, rows);
    if (!weight || !weight->data) {
        ggml_free(ggml_ctx);
        return false;
    }

    std::vector<float> row_f32(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            row_f32[static_cast<size_t>(c)] =
                std::sin(static_cast<float>(r * 19 + c) * 0.013f) * 0.35f;
        }
        q8_traits->from_float(row_f32.data(),
                              static_cast<uint8_t*>(weight->data) +
                                  static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q8_0, cols),
                              cols);
    }

    std::vector<float> input_f32(static_cast<size_t>(cols));
    for (int c = 0; c < cols; ++c) {
        input_f32[static_cast<size_t>(c)] = std::cos(static_cast<float>(c) * 0.021f) * 0.45f;
    }
    std::vector<uint8_t> q8_input(ggml_row_size(GGML_TYPE_Q8_0, cols));
    q8_traits->from_float(input_f32.data(), q8_input.data(), cols);

    std::vector<uint8_t> packed;
    const bool packed_ok = DenseCorePackQ8_0RowsTo4x8(static_cast<const uint8_t*>(weight->data),
                                                      ggml_row_size(GGML_TYPE_Q8_0, cols), rows, cols, packed);
    if (!packed_ok) {
        ggml_free(ggml_ctx);
        return false;
    }

    std::vector<float> output(static_cast<size_t>(rows), 0.0f);
    DenseCoreGemvQ8_0_4x8Q8_0Generic(cols, output.data(), packed.data(), q8_input.data(), rows);
    std::vector<float> streamed_output(static_cast<size_t>(rows), std::numeric_limits<float>::quiet_NaN());
    DenseCoreForEachQ8_0_4x8Q8_0DotGeneric(
        cols, packed.data(), q8_input.data(), rows, 0,
        [&](int row, float value) {
            if (row >= 0 && row < rows) {
                streamed_output[static_cast<size_t>(row)] = value;
            }
        });

    bool matches = true;
    for (int r = 0; r < rows; ++r) {
        const void* row_ptr = static_cast<const uint8_t*>(weight->data) +
                              static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q8_0, cols);
        float ref = 0.0f;
        q8_traits->vec_dot(cols, &ref, 0, row_ptr, 0, q8_input.data(), 0, 1);
        matches = matches && std::fabs(output[static_cast<size_t>(r)] - ref) <= 1e-4f;
        matches = matches && std::fabs(streamed_output[static_cast<size_t>(r)] - ref) <= 1e-4f;
    }
    if (output_matches_vecdot_oracle) {
        *output_matches_vecdot_oracle = matches;
    }
    ggml_free(ggml_ctx);
    return true;
}

bool RunGemma4SafeF32BatchedForTest(int nth, bool* output_matches_oracle, bool* output_all_finite) {
    constexpr int rows = 17;
    constexpr int cols = 32;
    constexpr int tokens = 5;
    ggml_init_params params{1 << 20, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor* weight = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor* dst = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    if (!input || !weight || !dst || !input->data || !weight->data || !dst->data) {
        ggml_free(ggml_ctx);
        return false;
    }
    dst->src[0] = input;
    dst->src[1] = weight;
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.ffn_gate_inp.weight");
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    auto* weight_f32 = reinterpret_cast<float*>(weight->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 11 + c) * 0.021f);
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            weight_f32[static_cast<size_t>(r) * cols + c] =
                std::cos(static_cast<float>(r * 7 + c) * 0.017f) * 0.25f;
        }
    }
    input_f32[static_cast<size_t>(2) * cols + 3] = std::numeric_limits<float>::quiet_NaN();
    weight_f32[static_cast<size_t>(4) * cols + 9] = std::numeric_limits<float>::infinity();
    std::fill_n(reinterpret_cast<float*>(dst->data), rows * tokens, 12345.0f);
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = cols;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = GGML_TYPE_F32;
    ud.input_quant_type = GGML_TYPE_F32;
    ud.slot_id = -1;
    ud.gemma4_prefill_safe_batched = true;
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    const auto* out = reinterpret_cast<const float*>(dst->data);
    bool matches = true;
    bool finite = true;
    for (int m = 0; m < tokens; ++m) {
        for (int r = 0; r < rows; ++r) {
            float ref = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const float x = input_f32[static_cast<size_t>(m) * cols + c];
                const float w = weight_f32[static_cast<size_t>(r) * cols + c];
                if (!std::isfinite(x) || !std::isfinite(w)) {
                    continue;
                }
                ref += x * w;
            }
            const float got = out[static_cast<size_t>(m) * rows + r];
            finite = finite && std::isfinite(got);
            matches = matches && std::fabs(got - ref) <= 1e-5f;
        }
    }
    if (output_matches_oracle) *output_matches_oracle = matches;
    if (output_all_finite) *output_all_finite = finite;
    ggml_free(ggml_ctx);
    return true;
}

bool RunGemma4SafeF32MsplitBatchedForTest(int nth, bool* output_matches_oracle, bool* output_all_finite) {
    constexpr int rows = 128;
    constexpr int cols = 1024;
    constexpr int tokens = 128;
    ggml_init_params params{96 << 20, nullptr, false};
    ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, tokens);
    ggml_tensor* weight = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor* dst = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, rows, tokens);
    if (!input || !weight || !dst || !input->data || !weight->data || !dst->data) {
        ggml_free(ggml_ctx);
        return false;
    }
    dst->src[0] = input;
    dst->src[1] = weight;
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.ffn_gate_inp.weight");
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    auto* weight_f32 = reinterpret_cast<float*>(weight->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 7 + c) * 0.011f) * 0.125f;
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            weight_f32[static_cast<size_t>(r) * cols + c] =
                std::cos(static_cast<float>(r * 5 + c) * 0.013f) * 0.0625f;
        }
    }
    std::fill_n(reinterpret_cast<float*>(dst->data), rows * tokens, 12345.0f);
    GemvBatchedUserData ud{};
    ud.weight_tensor = weight;
    ud.N = cols;
    ud.K = rows;
    ud.M = tokens;
    ud.weight_type = GGML_TYPE_F32;
    ud.input_quant_type = GGML_TYPE_F32;
    ud.slot_id = -1;
    ud.gemma4_prefill_safe_batched = true;
    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    ud.work_ctx = &work_ctx;
    SetCurrentWorkContext(&work_ctx);
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    SetCurrentWorkContext(nullptr);
    const auto* out = reinterpret_cast<const float*>(dst->data);
    bool matches = true;
    bool finite = true;
    for (int m = 0; m < tokens; ++m) {
        for (int r = 0; r < rows; ++r) {
            double ref = 0.0;
            for (int c = 0; c < cols; ++c) {
                ref += static_cast<double>(input_f32[static_cast<size_t>(m) * cols + c]) *
                       static_cast<double>(weight_f32[static_cast<size_t>(r) * cols + c]);
            }
            const float got = out[static_cast<size_t>(m) * rows + r];
            finite = finite && std::isfinite(got);
            matches = matches && std::fabs(got - static_cast<float>(ref)) <= 2e-4f;
        }
    }
    if (output_matches_oracle) *output_matches_oracle = matches;
    if (output_all_finite) *output_all_finite = finite;
    ggml_free(ggml_ctx);
    return true;
}

bool RunGemma4SafeQ8BatchedForTest(int nth, bool* output_matches_vecdot_oracle, uint64_t* q8_batched_used_ops) {
    ggml_cpu_init();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    constexpr int rows = 64;
    constexpr int cols = QK8_0 * 4;
    constexpr int tokens = 5;
    if (!q8_traits || !q8_traits->from_float || !q8_traits->vec_dot || q8_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }
    ggml_init_params params{2 << 20, nullptr, false};
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
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.attn_q.weight");
    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 13 + c) * 0.019f) * 0.5f;
        }
    }
    std::vector<float> row_f32(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            row_f32[static_cast<size_t>(c)] = std::cos(static_cast<float>(r * 17 + c) * 0.013f) * 0.25f;
        }
        q8_traits->from_float(row_f32.data(),
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
    ud.gemma4_dense_prefill_native = true;
    ud.gemma4_prefill_safe_batched = true;
    InferenceWorkContext work_ctx{};
    ResetInferenceWorkContext(&work_ctx);
    ud.work_ctx = &work_ctx;
    SetCurrentWorkContext(&work_ctx);
    for (int ith = 0; ith < std::max(1, nth); ++ith) {
        cb_gemv_batched_custom(dst, ith, std::max(1, nth), &ud);
    }
    SetCurrentWorkContext(nullptr);
    const auto snapshot = GetQwen36ProfileSnapshot(&work_ctx);
    if (q8_batched_used_ops) {
        *q8_batched_used_ops = snapshot.q8_batched_used_ops;
    }

    std::vector<uint8_t> q8_input(static_cast<size_t>(tokens) * ud.quant_row_stride);
    for (int m = 0; m < tokens; ++m) {
        q8_traits->from_float(input_f32 + static_cast<size_t>(m) * cols,
                              q8_input.data() + static_cast<size_t>(m) * ud.quant_row_stride, cols);
    }
    const auto* out = reinterpret_cast<const float*>(dst->data);
    bool matches = true;
    for (int r = 0; r < rows; ++r) {
        const void* row_ptr = static_cast<const uint8_t*>(weight->data) +
                              static_cast<size_t>(r) * ggml_row_size(GGML_TYPE_Q8_0, cols);
        for (int m = 0; m < tokens; ++m) {
            float ref = 0.0f;
            q8_traits->vec_dot(cols, &ref, 0, row_ptr, 0,
                               q8_input.data() + static_cast<size_t>(m) * ud.quant_row_stride, 0, 1);
            const float got = out[static_cast<size_t>(m) * rows + r];
            matches = matches && std::fabs(got - ref) <= 1e-4f;
        }
    }
    if (output_matches_vecdot_oracle) *output_matches_vecdot_oracle = matches;
    ggml_free(ggml_ctx);
    return true;
}

bool RunDenseCoreQ8PackedTileForTest(int cols, int rows, int pattern) {
    if (cols <= 0 || cols % QK8_0 != 0 || rows <= 0 || rows % 4 != 0) return false;
    ggml_cpu_init();
    const int nb = cols / QK8_0;
    std::vector<block_q8_0> weight(static_cast<size_t>(rows) * nb);
    std::vector<block_q8_0> input(4 * nb);
    const auto fill = [pattern](std::vector<block_q8_0>& blocks, int salt) {
        for (size_t b = 0; b < blocks.size(); ++b) {
            blocks[b].d = ggml_fp32_to_fp16(0.001f * static_cast<float>(1 + (b + salt) % 11));
            for (int i = 0; i < QK8_0; ++i) {
                blocks[b].qs[i] = pattern == 0 ? 0
                                  : pattern == 1
                                      ? ((i + b + salt) % 2 ? 127 : -127)
                                      : static_cast<int8_t>(static_cast<int>((i * 17 + b * 31 + salt) % 255) - 127);
                if (pattern == 3 && salt == 3 && i % 3 == 0) blocks[b].qs[i] = -128;
            }
        }
    };
    fill(weight, 3);
    fill(input, 7);
    std::vector<uint8_t> packed_weight, packed_input;
    const size_t stride = static_cast<size_t>(nb) * sizeof(block_q8_0);
    if (!DenseCorePackQ8_0RowsTo4x8(reinterpret_cast<const uint8_t*>(weight.data()), stride, rows, cols,
                                    packed_weight) ||
        !DenseCorePackQ8_0RowsTo4x8(reinterpret_cast<const uint8_t*>(input.data()), stride, 4, cols, packed_input)) {
        return false;
    }
    const int output_stride = rows + 3;
    std::vector<float> output(4 * output_stride, 12345.0f);
    DenseCoreGemmQ8_0_4x8x4Q8_0Generic(cols, output.data(), output_stride, packed_weight.data(), packed_input.data(),
                                       rows);
    for (int m = 0; m < 4; ++m) {
        for (int r = 0; r < rows; ++r) {
            float expected = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const auto& w = weight[static_cast<size_t>(r) * nb + b];
                const auto& x = input[m * nb + b];
                int dot = 0;
                for (int k = 0; k < QK8_0; ++k) dot += static_cast<int>(w.qs[k]) * x.qs[k];
                expected += static_cast<float>(dot) * ggml_fp16_to_fp32(w.d) * ggml_fp16_to_fp32(x.d);
            }
            const float actual = output[m * output_stride + r];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-5f + 2e-6f * std::fabs(expected)) {
                return false;
            }
        }
        for (int r = rows; r < output_stride; ++r) {
            if (output[m * output_stride + r] != 12345.0f) return false;
        }
    }
    return true;
}

bool RunGemma4NativeQ8PrefillTrueGemmForTest(int nth, bool* output_matches_vecdot_oracle,
                                             uint64_t* true_gemm_ops, uint64_t* gemv_ops) {
    // CTest runs each test in its own process; initialize the vec-dot oracle's
    // CPU fp16 lookup tables even when no earlier backend test has run.
    ggml_cpu_init();
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    constexpr int rows = 128;
    constexpr int cols = 2048;
    constexpr int tokens = 4;
    if (!q8_traits || !q8_traits->from_float || !q8_traits->vec_dot || q8_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }
    ggml_init_params params{32 << 20, nullptr, false};
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
    std::snprintf(weight->name, sizeof(weight->name), "blk.0.attn_q.weight");

    auto* input_f32 = reinterpret_cast<float*>(input->data);
    for (int m = 0; m < tokens; ++m) {
        for (int c = 0; c < cols; ++c) {
            input_f32[static_cast<size_t>(m) * cols + c] =
                std::sin(static_cast<float>(m * 23 + c) * 0.007f) * 0.5f;
        }
    }
    std::vector<float> row_f32(static_cast<size_t>(cols));
    const size_t weight_row_size = ggml_row_size(GGML_TYPE_Q8_0, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            row_f32[static_cast<size_t>(c)] = std::cos(static_cast<float>(r * 19 + c) * 0.009f) * 0.25f;
        }
        q8_traits->from_float(row_f32.data(),
                              static_cast<uint8_t*>(weight->data) + static_cast<size_t>(r) * weight_row_size,
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
    ud.gemma4_dense_prefill_native = true;
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
    const auto* out = reinterpret_cast<const float*>(dst->data);
    for (int r = 0; r < rows; ++r) {
        const void* row_ptr = static_cast<const uint8_t*>(weight->data) + static_cast<size_t>(r) * weight_row_size;
        for (int m = 0; m < tokens; ++m) {
            float ref = 0.0f;
            q8_traits->vec_dot(cols, &ref, 0, row_ptr, 0,
                               q8_input.data() + static_cast<size_t>(m) * ud.quant_row_stride, 0, 1);
            const float got = out[static_cast<size_t>(m) * rows + r];
            matches = matches && std::fabs(got - ref) <= 1e-4f;
        }
    }
    if (output_matches_vecdot_oracle) {
        *output_matches_vecdot_oracle = matches;
    }
    ggml_free(ggml_ctx);
    return true;
}





























}  // namespace testing
}  // namespace densecore
#endif





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





bool ShouldUseQwenHybridSSMQ8RepackedBatchedForTest(bool relevant, int tokens) {
    return ::ShouldUseQwenHybridSSMQ8RepackedBatched(relevant, tokens);
}

bool ShouldUseQwenHybridSSMQ8DirectBatchedForTest(bool relevant, int tokens) {
    return ::ShouldUseQwenHybridSSMQ8DirectBatched(relevant, tokens);
}
}  // namespace densecore::testing
