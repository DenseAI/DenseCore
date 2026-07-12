static struct ggml_tensor* BertLayerNorm(struct ggml_context* ctx, struct ggml_tensor* src, struct ggml_tensor* weight,
                                         struct ggml_tensor* bias, float eps, const char* name) {
    if (!src || !weight) {
        throw densecore::InvalidArgumentException("BERT LayerNorm is missing input or weight tensor");
    }
    struct ggml_tensor* out = ggml_norm(ctx, src, eps);
    out = ggml_mul(ctx, out, weight);
    if (bias) {
        out = ggml_add(ctx, out, bias);
    }
    if (name) {
        ggml_set_name(out, name);
    }
    return out;
}

static struct ggml_tensor* BertLinear(struct ggml_context* ctx, TransformerModel* model, struct ggml_tensor* weight,
                                      struct ggml_tensor* bias, struct ggml_tensor* src, const char* name) {
    if (!weight || !src) {
        throw densecore::InvalidArgumentException("BERT linear projection is missing input or weight tensor");
    }
    struct ggml_tensor* out = smart_mul_mat(ctx, weight, src, model);
    if (bias && out->ne[0] == bias->ne[0]) {
        out = ggml_add(ctx, out, bias);
    }
    if (name) {
        ggml_set_name(out, name);
    }
    return out;
}

static struct ggml_tensor* BuildBertSelfAttention(struct ggml_context* ctx, TransformerModel* model,
                                                  const TransformerLayer& layer, struct ggml_tensor* src, int layer_idx,
                                                  int n_tokens) {
    const int n_embd = static_cast<int>(model->hparams.n_embd);
    const int n_head = static_cast<int>(model->hparams.n_head);
    if (n_embd <= 0 || n_head <= 0 || (n_embd % n_head) != 0) {
        throw densecore::InvalidArgumentException("BERT attention has invalid hidden/head dimensions");
    }
    const int head_dim = n_embd / n_head;

    struct ggml_tensor* q = BertLinear(ctx, model, layer.Get(model_keys::kAttnQWeight),
                                       layer.Get(model_keys::kAttnQBias), src, "bert_attn_q");
    struct ggml_tensor* k = BertLinear(ctx, model, layer.Get(model_keys::kAttnKWeight),
                                       layer.Get(model_keys::kAttnKBias), src, "bert_attn_k");
    struct ggml_tensor* v = BertLinear(ctx, model, layer.Get(model_keys::kAttnVWeight),
                                       layer.Get(model_keys::kAttnVBias), src, "bert_attn_v");

    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_head, n_tokens);
    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_head, n_tokens);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_head, n_tokens);

    struct ggml_tensor* q_att = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    struct ggml_tensor* k_att = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor* v_att = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    struct ggml_tensor* kq = ggml_mul_mat(ctx, k_att, q_att);
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_soft_max(ctx, kq);

    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_att, 1, 0, 2, 3));
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);

    struct ggml_tensor* merged = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);
    return BertLinear(ctx, model, layer.Get(model_keys::kAttnOWeight), layer.Get(model_keys::kAttnOBias), merged,
                      "bert_attn_output");
}

static struct ggml_tensor* BuildBertEncoderEmbeddingGraph(TransformerModel* model, struct ggml_context* ctx_c,
                                                          const BatchSpec& batch, struct ggml_cgraph* gf,
                                                          struct ggml_tensor** out_embd,
                                                          struct ggml_tensor** out_pos) {
    if (!model || !model->tok_embeddings || !model->position_embeddings) {
        throw densecore::InvalidArgumentException("BERT embedding graph requires token and position embeddings");
    }

    const int n_tokens = static_cast<int>(batch.tokens.size());
    if (n_tokens <= 0) {
        throw densecore::InvalidArgumentException("BERT embedding graph received an empty batch");
    }

    struct ggml_tensor* token_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids, "bert_token_ids");
    if (token_ids->data) {
        std::memcpy(token_ids->data, batch.tokens.data(), static_cast<size_t>(n_tokens) * sizeof(int));
    }
    if (out_embd) *out_embd = token_ids;

    struct ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos_ids, "bert_position_ids");
    if (pos_ids->data) {
        PopulatePositionTensor(model, batch, pos_ids);
    }
    if (out_pos) *out_pos = pos_ids;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, token_ids);
    cur = ggml_add(ctx_c, cur, ggml_get_rows(ctx_c, model->position_embeddings, pos_ids));

    if (model->token_type_embeddings) {
        struct ggml_tensor* token_type_ids = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, n_tokens);
        ggml_set_name(token_type_ids, "bert_token_type_ids");
        if (token_type_ids->data) {
            std::memset(token_type_ids->data, 0, static_cast<size_t>(n_tokens) * sizeof(int));
            cur = ggml_add(ctx_c, cur, ggml_get_rows(ctx_c, model->token_type_embeddings, token_type_ids));
        }
    }

    if (model->token_embd_norm) {
        cur = BertLayerNorm(ctx_c, cur, model->token_embd_norm, model->token_embd_norm_bias,
                            model->hparams.f_norm_rms_eps, "bert_embedding_norm");
    }

    for (int il = 0; il < static_cast<int>(model->hparams.n_layer); ++il) {
        const TransformerLayer& layer = model->layers[static_cast<size_t>(il)];
        struct ggml_tensor* attn_out = BuildBertSelfAttention(ctx_c, model, layer, cur, il, n_tokens);
        cur = ggml_add(ctx_c, cur, attn_out);
        cur = BertLayerNorm(ctx_c, cur, layer.Get(model_keys::kAttnOutputNorm),
                            layer.Get(model_keys::kAttnOutputNormBias), model->hparams.f_norm_rms_eps,
                            "bert_attn_output_norm");

        struct ggml_tensor* ffn = BertLinear(ctx_c, model, layer.Get(model_keys::kFfnUp),
                                             layer.Get(model_keys::kFfnUpBias), cur, "bert_ffn_up");
        ffn = ggml_gelu_erf(ctx_c, ffn);
        ffn = BertLinear(ctx_c, model, layer.Get(model_keys::kFfnDown), layer.Get(model_keys::kFfnDownBias), ffn,
                         "bert_ffn_down");
        cur = ggml_add(ctx_c, cur, ffn);
        cur = BertLayerNorm(ctx_c, cur, layer.Get(model_keys::kLayerOutputNorm),
                            layer.Get(model_keys::kLayerOutputNormBias), model->hparams.f_norm_rms_eps,
                            "bert_layer_output_norm");
    }

    ggml_set_name(cur, "bert_encoder_hidden_states");
    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }
    return cur;
}

static struct ggml_tensor* BuildTransformerGraphInlineImpl(TransformerModel* model, PagedKVCache* cache,
                                                           struct ggml_context* ctx_c, const BatchSpec& batch,
                                                           bool embedding_mode, struct ggml_cgraph* gf,
                                                           struct ggml_tensor** out_embd,
                                                           struct ggml_tensor** out_pos) {
    // ENSURE: ctx_c must be initialized with sufficient memory (e.g. 128MB+)
    // to hold the compute graph nodes, especially for deep models like Qwen.
    // This initialization happens in worker.cpp (InitGraphCache or temp
    // context).

    static constexpr int kSsmDeltaHeadsPerTask = 4;

    // N is batch size
    const int N = batch.tokens.size();
    const int n_embd = model->hparams.n_embd;
    const int n_head = model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    const int n_layer = model->hparams.n_layer;
    const int n_ctx = model->hparams.n_ctx;
    densecore::llm::decoder::DecoderSpecView decoder_spec_view(model);
    const densecore::models::DecoderModelSpec* decoder_spec = decoder_spec_view.get();
    const DecodePagedAttentionPolicy& decode_paged_policy = ResolveDecodePagedAttentionPolicy(&batch);
    (void)n_embd;
    (void)n_head_kv;
    (void)n_ctx;

    // =========================================================================
    // 1. Token Embedding Lookup
    // =========================================================================
    struct ggml_tensor* embd_inp = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
    ggml_set_name(embd_inp, "embd_inp");
    if (embd_inp->data) {
        memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
    }
    if (out_embd) *out_embd = embd_inp;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, embd_inp);
    if (const float embedding_scale = densecore::models::ResolveInputEmbeddingScale(model); embedding_scale != 1.0f) {
        cur = ggml_scale(ctx_c, cur, embedding_scale);
    }

    // Position tensor for RoPE
    const int pos_ids_per_token = PositionIdsPerToken(model);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, static_cast<int64_t>(N) * pos_ids_per_token);
    ggml_set_name(pos, "pos");
    if (pos->data) {
        PopulatePositionTensor(model, batch, pos);
    }
    if (out_pos) *out_pos = pos;

    const bool decode_only_batch_layout = IsDecodeOnlyBatchLayout(batch, N);
    const int debug_query_base_pos = ResolveAttentionQueryBasePosition(batch);
    const bool gemma4_decode_special_transforms_disabled =
        model->arch_flags.is_gemma4 && densecore::models::IsGemma4DecodeSpecialTransformDisabled() &&
        debug_query_base_pos > 0;

    auto requires_gemma_rms_weight_offset = [&]() -> bool {
        return densecore::models::RequiresUnitOffsetRmsNorm(model);
    };

    auto canonicalize_unit_offset_rms_weight = [&](struct ggml_tensor * norm_weight) -> void {
        if (!norm_weight || !requires_gemma_rms_weight_offset()) {
            return;
        }
        if (norm_weight->type != GGML_TYPE_F32 || !norm_weight->data) {
            return;
        }

        static std::mutex shifted_rms_mu;
        static std::unordered_map<const ggml_tensor*, const void*> shifted_rms_weights;
        {
            std::lock_guard<std::mutex> lock(shifted_rms_mu);
            auto it = shifted_rms_weights.find(norm_weight);
            if (it != shifted_rms_weights.end() && it->second == norm_weight->data) {
                return;
            }

            const int64_t n_weight = ggml_nelements(norm_weight);
            float* raw_weight = reinterpret_cast<float*>(norm_weight->data);
            for (int64_t i = 0; i < n_weight; ++i) {
                raw_weight[i] += 1.0f;
            }
            shifted_rms_weights[norm_weight] = norm_weight->data;
        }
    };

    auto effective_rms_weight = [&](struct ggml_tensor * norm_weight, const char* debug_name) -> struct ggml_tensor* {
        if (!norm_weight) {
            return norm_weight;
        }
        canonicalize_unit_offset_rms_weight(norm_weight);
        if (debug_name) {
            ggml_set_name(norm_weight, debug_name);
        }
        return norm_weight;
    };

    auto bind_add_rmsnorm_weight = [&](AddRMSNormUserData* ud, struct ggml_tensor* norm_weight,
                                       const char* weight_name) -> void {
        if (!ud || !norm_weight) {
            throw densecore::InvalidArgumentException("Missing RMSNorm weight tensor");
        }
        if (norm_weight->type != GGML_TYPE_F32 || !norm_weight->data) {
            throw densecore::InvalidArgumentException(std::string("RMSNorm weight ") + weight_name +
                                                      " must be materialized FP32");
        }
        const int64_t n_weight = ggml_nelements(norm_weight);
        if (n_weight <= 0) {
            throw densecore::InvalidArgumentException(std::string("RMSNorm weight ") + weight_name +
                                                      " has invalid element count");
        }
        canonicalize_unit_offset_rms_weight(norm_weight);
        ud->owned_rms_weight.clear();
        ud->rms_weight = reinterpret_cast<const float*>(norm_weight->data);
    };

    auto ensure_rms_norm_f32_input = [&](struct ggml_tensor * src, const char* name,
                                         int layer_idx) -> struct ggml_tensor* {
        if (!src || src->type == GGML_TYPE_F32) {
            return src;
        }
        static std::atomic<int> cast_log_budget{8};
        const int remaining = cast_log_budget.fetch_sub(1, std::memory_order_relaxed);
        if (remaining > 0) {
            std::cerr << "[DenseCore] INFO: casting " << (name ? name : "rms_norm_input") << " to F32 before RMSNorm"
                      << " layer=" << layer_idx << " src_type=" << static_cast<int>(src->type) << std::endl;
        }
        return ggml_cast(ctx_c, src, GGML_TYPE_F32);
    };

    auto apply_weighted_rms_norm = [&](struct ggml_tensor * src, struct ggml_tensor * norm_weight,
                                       const char* debug_name, int debug_layer_idx = -1) -> struct ggml_tensor* {
        if (!src || !norm_weight) {
            return src;
        }
        struct ggml_tensor* effective_norm_weight = effective_rms_weight(norm_weight, nullptr);

        const bool force_cpu_norm =
            IsMixedRoutingEnabled(&batch) && ResolvePreferredNormDevice(&batch) == densecore::DeviceType::CPU &&
            ResolvePreferredDevice(&batch) != densecore::DeviceType::CPU && src->type == GGML_TYPE_F32 &&
            effective_norm_weight->type == GGML_TYPE_F32 && src->nb[0] == static_cast<int64_t>(sizeof(float));
        const bool use_qwen_weighted_rms_norm =
            (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
            src->type == GGML_TYPE_F32 &&
            effective_norm_weight->type == GGML_TYPE_F32 && src->nb[0] == static_cast<int64_t>(sizeof(float));

        if (!force_cpu_norm && !use_qwen_weighted_rms_norm) {
            struct ggml_tensor* norm_src = ensure_rms_norm_f32_input(src, debug_name, debug_layer_idx);
            struct ggml_tensor* out = ggml_rms_norm(ctx_c, norm_src, model->hparams.f_norm_rms_eps);
            if (debug_name) {
                char rms_name[96];
                if (debug_layer_idx >= 0) {
                    std::snprintf(rms_name, sizeof(rms_name), "blk.%d.%s_rms", debug_layer_idx, debug_name);
                } else {
                    std::snprintf(rms_name, sizeof(rms_name), "%s_rms", debug_name);
                }
                ggml_set_name(out, rms_name);
            }
            out = ggml_mul(ctx_c, out, effective_norm_weight);
            if (debug_name) {
                ggml_set_name(out, debug_name);
            }
            if (debug_layer_idx >= 0 && ShouldRunRmsNormReferenceProbe(debug_layer_idx)) {
                auto* rms_ud = GetRmsNormReferenceUserData();
                rms_ud->input_tensor = norm_src;
                rms_ud->norm_weight = effective_norm_weight;
                rms_ud->layer_idx = debug_layer_idx;
                rms_ud->token_seq_ids = batch.seq_id.data();
                rms_ud->stage = "rms_norm";
                rms_ud->var_name = debug_name ? debug_name : "rms_norm";
                out = ggml_map_custom1(ctx_c, out, cb_rmsnorm_reference_probe, 1, rms_ud);
            }
            return out;
        }

        AddRMSNormUserData* ud = GetAddRMSNormUserData();
        ud->residual = nullptr;
        bind_add_rmsnorm_weight(ud, norm_weight, debug_name ? debug_name : "rms_norm");
        ud->n_embd = static_cast<int>(src->ne[0]);
        ud->n_tokens = static_cast<int>(src->ne[1]);
        ud->eps = model->hparams.f_norm_rms_eps;
        ud->residual_row_stride = 0;
        ud->layer_idx = debug_layer_idx;
        ud->token_seq_ids = batch.seq_id.data();
        ud->stage = "rms_norm";
        ud->var_name = debug_name ? debug_name : "rms_norm";
        const int n_tasks = ResolveTaskCount(&batch, std::max(1, ud->n_tokens));
        struct ggml_tensor* out = ggml_map_custom1(ctx_c, src, cb_residual_rmsnorm_fused, n_tasks, ud);
        if (debug_name) {
            ggml_set_name(out, debug_name);
        }
        return out;
    };

    struct ggml_tensor* gemma4_per_layer_inputs = nullptr;
    if (model->arch_flags.is_gemma4 && model->gemma4_hidden_size_per_layer_input > 0 &&
        model->gemma4_per_layer_model_projection && model->gemma4_per_layer_projection_norm &&
        model->gemma4_per_layer_token_embeddings) {
        const int hidden_per_layer = model->gemma4_hidden_size_per_layer_input;
        const int n_layer_i = static_cast<int>(model->hparams.n_layer);

        struct ggml_tensor* token_inputs = ggml_get_rows(ctx_c, model->gemma4_per_layer_token_embeddings, embd_inp);
        token_inputs = ggml_scale(ctx_c, token_inputs, std::sqrt(static_cast<float>(hidden_per_layer)));
        token_inputs = ggml_reshape_3d(ctx_c, token_inputs, hidden_per_layer, n_layer_i, N);

        struct ggml_tensor* projected_inputs =
            smart_mul_mat(ctx_c, model->gemma4_per_layer_model_projection, cur, model);
        projected_inputs =
            ggml_scale(ctx_c, projected_inputs, 1.0f / std::sqrt(static_cast<float>(model->hparams.n_embd)));
        projected_inputs = ggml_reshape_3d(ctx_c, projected_inputs, hidden_per_layer, n_layer_i, N);

        struct ggml_tensor* projected_inputs_2d =
            ggml_reshape_2d(ctx_c, projected_inputs, hidden_per_layer, n_layer_i * N);
        projected_inputs_2d = apply_weighted_rms_norm(projected_inputs_2d, model->gemma4_per_layer_projection_norm,
                                                      "gemma4_per_layer_projection_norm");
        gemma4_per_layer_inputs = ggml_reshape_3d(ctx_c, projected_inputs_2d, hidden_per_layer, n_layer_i, N);
        gemma4_per_layer_inputs = ggml_add(ctx_c, gemma4_per_layer_inputs, token_inputs);
        gemma4_per_layer_inputs = ggml_scale(ctx_c, gemma4_per_layer_inputs, std::pow(2.0f, -0.5f));
    }

    // Reuse causal mask tensor across layers for the same forward pass.
    // The mask is built lazily only if native flash attention is actually
    // selected. Portable CPU flash and standard attention do not need it.
    struct ggml_tensor* shared_prefill_flash_mask = nullptr;
    int shared_prefill_mask_n_total = -1;
    int shared_prefill_mask_n_padded = -1;
    int shared_prefill_mask_n = -1;
    int shared_prefill_mask_n_past = -1;
    int shared_prefill_mask_sliding_window = -1;

    // =========================================================================
    // 2. Transformer Layers
    // =========================================================================
    int ssm_ordinal_counter = 0;  // Counts SSM layers for state indexing
    std::array<uint64_t, 7> moe_wiring_reason_counts{};
    const bool moe_wiring_debug = IsMoEWiringDebugEnabled();
    if (moe_wiring_debug) {
        const auto resolution = densecore::models::ResolveGraphFamily(model);
        std::fprintf(stderr,
                     "[MOE_WIRING_MODEL] arch=%d variant=%d preferred_family=%s fail_closed=%d capabilities=%s "
                     "n_layer=%d n_experts=%u top_k=%u moe_first_k_dense_replace=%d N=%d decode_only=%d\n",
                     static_cast<int>(resolution.capabilities.arch), static_cast<int>(resolution.capabilities.variant),
                     densecore::models::GraphFamilyName(resolution.preferred_family), resolution.fail_closed ? 1 : 0,
                     densecore::models::FormatModelGraphCapabilities(resolution.capabilities).c_str(), n_layer,
                     model->hparams.n_experts, model->hparams.n_experts_used, model->moe_first_k_dense_replace, N,
                     decode_only_batch_layout ? 1 : 0);
    }
    for (int il = 0; il < n_layer; ++il) {
        auto& layer = model->layers[il];
        const densecore::models::DecoderLayerSpec* layer_spec = decoder_spec_view.Layer(il);
        auto* attn_norm = layer.Get(model_keys::kAttnNorm);
        auto* wq = layer.Get(model_keys::kAttnQWeight);
        auto* wk = layer.Get(model_keys::kAttnKWeight);
        auto* wv = layer.Get(model_keys::kAttnVWeight);
        auto* wo = layer.Get(model_keys::kAttnOWeight);
        auto* q_a = layer.Get(model_keys::kAttnQAProj);
        auto* q_a_norm = layer.Get(model_keys::kAttnQANorm);
        auto* q_b = layer.Get(model_keys::kAttnQBProj);
        auto* kv_a = layer.Get(model_keys::kAttnKvAProj);
        auto* kv_a_norm = layer.Get(model_keys::kAttnKvANorm);
        auto* kv_b = layer.Get(model_keys::kAttnKvBProj);
        auto* indexer_wq_b = layer.Get(model_keys::kIndexerWqB);
        auto* indexer_wk = layer.Get(model_keys::kIndexerWk);
        auto* indexer_k_norm = layer.Get(model_keys::kIndexerKNorm);
        auto* indexer_weights_proj = layer.Get(model_keys::kIndexerWeightsProj);
        auto* bq = layer.Get(model_keys::kAttnQBias);
        auto* bk = layer.Get(model_keys::kAttnKBias);
        auto* bv = layer.Get(model_keys::kAttnVBias);
        auto* bo = layer.Get(model_keys::kAttnOBias);
        auto* q_norm = layer.Get(model_keys::kAttnQNorm);
        auto* k_norm = layer.Get(model_keys::kAttnKNorm);
        auto* rope_freqs = layer.Get(model_keys::kAttnRopeFreqs);
        auto* ffn_norm = layer.Get(model_keys::kFfnNorm);
        auto* ffn_gate = layer.Get(model_keys::kFfnGate);
        auto* ffn_up = layer.Get(model_keys::kFfnUp);
        auto* ffn_down = layer.Get(model_keys::kFfnDown);
        auto* ffn_shared_gate = layer.Get(model_keys::kFfnSharedGate);
        auto* moe_gate = layer.Get(model_keys::kMoeGate);
        const int layer_num_experts = static_cast<int>(layer.NumExperts());
        const int dense_replace_cutoff = std::max(0, model->moe_first_k_dense_replace);
        const bool dense_replace_gate = il < dense_replace_cutoff;
        const bool layer_uses_moe = layer_spec ? layer_spec->ffn.is_moe : layer.is_moe;
        const bool model_has_moe = decoder_spec ? decoder_spec->has_moe : model->hparams.n_experts > 0;
        const bool has_moe_gate = moe_gate != nullptr;
        const bool has_experts = layer_num_experts > 0;
        const bool layer_moe_candidate = has_moe_gate && has_experts && !dense_replace_gate;
        if (moe_wiring_debug) {
            MoEWiringReasonCode reason = MoEWiringReasonCode::Wired;
            if (!layer_uses_moe) {
                if (!model_has_moe) {
                    reason = MoEWiringReasonCode::ModelHasNoMoE;
                } else if (!has_moe_gate) {
                    reason = MoEWiringReasonCode::MissingMoeGate;
                } else if (!has_experts) {
                    reason = MoEWiringReasonCode::NoExperts;
                } else if (dense_replace_gate) {
                    reason = MoEWiringReasonCode::DenseReplaceGate;
                } else if (layer_moe_candidate) {
                    reason = MoEWiringReasonCode::LayerFlagMismatch;
                } else {
                    reason = MoEWiringReasonCode::LayerFlagFalse;
                }
            }
            const int reason_code = static_cast<int>(reason);
            if (reason_code >= 0 && reason_code < static_cast<int>(moe_wiring_reason_counts.size())) {
                moe_wiring_reason_counts[static_cast<size_t>(reason_code)]++;
            }
            std::fprintf(stderr,
                         "[MOE_WIRING_LAYER] layer=%d is_moe_layer=%d has_moe_gate=%d num_experts=%d "
                         "dense_replace_gate=%d model_has_moe=%d attempted=%d reason_code=%d\n",
                         il, layer.is_moe ? 1 : 0, has_moe_gate ? 1 : 0, layer_num_experts, dense_replace_gate ? 1 : 0,
                         model_has_moe ? 1 : 0, layer_uses_moe ? 1 : 0, reason_code);
        }

        struct ggml_tensor* inpL = cur;
        if (ShouldRunHiddenSnapshotProbe(il, "layer_input")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_ids = batch.tokens.data();
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "layer_input";
                hidden_ud->var_name = "inpL";
                inpL = ggml_map_custom1(ctx_c, inpL, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        // Attention Norm
        if (!attn_norm) {
            throw densecore::InvalidArgumentException("Missing attention_norm weight in TransformerLayer");
        }
        cur = apply_weighted_rms_norm(cur, attn_norm, "attn_norm", il);

        // Short-conv / SSM / Attention layer dispatch
        const bool is_conv_layer = model->IsLFM2ConvLayer(il);
        const bool is_ssm_layer = model->IsHybridSSMLayer(il);
        struct ggml_tensor* attn_out = nullptr;
        struct ggml_tensor* attn_post_residual = nullptr;

        if (is_conv_layer) {
            // =================================================================
            // LFM2 / LFM2.5 double-gated short-conv mixer
            //   BCx = in_proj(x_norm); y = C * conv(B * x); out = out_proj(y)
            // =================================================================
            auto* in_proj = layer.Get(model_keys::kShortConvInProj);
            auto* out_proj = layer.Get(model_keys::kShortConvOutProj);
            if (!in_proj || !out_proj) {
                throw densecore::InvalidArgumentException("Missing LFM2 short-conv weights in layer " +
                                                          std::to_string(il));
            }
            const int conv_ordinal = model->LFM2ConvOrdinal(il);
            if (conv_ordinal < 0 || conv_ordinal >= static_cast<int>(model->lfm2_conv_weight_f32.size())) {
                throw densecore::InvalidArgumentException("LFM2 conv ordinal out of range in layer " +
                                                          std::to_string(il));
            }
            const int channels = static_cast<int>(in_proj->ne[1] / 3);

            LFM2ShortConvUserData* conv_ud = GetLFM2ShortConvUserData();
            InferenceWorkContext* current_work_ctx = GetCurrentWorkContext();
            conv_ud->conv_weight = model->lfm2_conv_weight_f32[static_cast<size_t>(conv_ordinal)].data();
            conv_ud->channels = channels;
            conv_ud->kernel = model->lfm2_conv_kernel;
            conv_ud->conv_ordinal = conv_ordinal;
            conv_ud->layer_idx = il;
            conv_ud->token_seq_ids = batch.seq_id.data();
            conv_ud->token_positions = batch.pos.data();
            conv_ud->runtime_states = &batch.hybrid_ssm_runtime_states;
            conv_ud->profile = &current_work_ctx->qwen36_profile;
            conv_ud->work_ctx = current_work_ctx;

            struct ggml_tensor* mixer_out = nullptr;
            const bool lfm2_decode_fused_inout =
                N == 1 && cur->type == GGML_TYPE_F32 && in_proj->type == GGML_TYPE_Q4_K &&
                out_proj->type == GGML_TYPE_Q4_K && in_proj->ne[1] >= 3 * channels &&
                in_proj->ne[0] == cur->ne[0] && out_proj->ne[0] == channels &&
                densecore::kernels::Q4KRepackedGemvIsaSupported() &&
                densecore::kernels::Q4KRealPackedGemvKernelAvailable();
            if (lfm2_decode_fused_inout) {
                ggml_tensor* args[] = {cur, in_proj, out_proj};
                mixer_out = ggml_custom_4d(ctx_c, GGML_TYPE_F32, out_proj->ne[1], 1, 1, 1, args, 3,
                                           cb_lfm2_shortconv_inout_q4k_decode,
                                           ResolveTaskCount(&batch, static_cast<int>(out_proj->ne[1])), conv_ud);
                ggml_set_name(mixer_out, "lfm2_shortconv_inout_q4k_decode");
            }

            // 1. in_proj: [n_embd, N] -> [3*channels, N] (rows: B, C, x)
            struct ggml_tensor* bcx = nullptr;
            if (!mixer_out) {
                bcx = smart_mul_mat(ctx_c, in_proj, cur, model);
                ggml_set_name(bcx, "lfm2_shortconv_bcx");
            }

            const bool lfm2_decode_fused_out =
                !mixer_out && N == 1 && bcx->type == GGML_TYPE_F32 && out_proj->type == GGML_TYPE_Q4_K &&
                out_proj->ne[0] == channels;
            if (lfm2_decode_fused_out) {
                const int64_t ne_out[4] = {out_proj->ne[1], 1, 1, 1};
                mixer_out = ggml_new_tensor(ctx_c, GGML_TYPE_F32, 4, ne_out);
                mixer_out->op = GGML_OP_CUSTOM;
                mixer_out->src[0] = bcx;
                mixer_out->src[1] = out_proj;
                struct {
                    ggml_custom_op_t fun;
                    int n_tasks;
                    void* userdata;
                } params = {cb_lfm2_shortconv_out_q4k_decode,
                            ResolveTaskCount(&batch, static_cast<int>(out_proj->ne[1])), conv_ud};
                static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
                std::memcpy(mixer_out->op_params, &params, sizeof(params));
            } else if (!mixer_out) {
                // 2. gated depthwise causal conv (updates per-seq conv state) -> [channels, N]
                struct ggml_tensor* y = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, channels, N);
                y = ggml_map_custom2(ctx_c, y, bcx, cb_lfm2_shortconv, 1, conv_ud);
                ggml_set_name(y, "lfm2_shortconv_y");
                // 3. out_proj: [channels, N] -> [n_embd, N], then residual add.
                mixer_out = smart_mul_mat(ctx_c, out_proj, y, model);
            }
            ggml_set_name(mixer_out, "lfm2_shortconv_out");
            attn_out = mixer_out;
            attn_post_residual = ggml_add(ctx_c, mixer_out, inpL);
            cur = attn_post_residual;
        } else if (is_ssm_layer) {
            // =================================================================
            // SSM/Mamba2 Layer Forward Path
            // =================================================================
            auto* attn_qkv = layer.Get(model_keys::kAttnQkvWeight);
            auto* ssm_conv1d_w = layer.Get(model_keys::kSSMConv1d);
            auto* ssm_a = layer.Get(model_keys::kSSMA);
            auto* ssm_alpha_w = layer.Get(model_keys::kSSMAlpha);
            auto* ssm_beta_w = layer.Get(model_keys::kSSMBeta);
            auto* ssm_dt_bias_t = layer.Get(model_keys::kSSMDtBias);
            auto* ssm_norm_w = layer.Get(model_keys::kSSMNorm);
            auto* attn_gate_w = layer.Get(model_keys::kAttnGate);
            auto* ssm_out_w = layer.Get(model_keys::kSSMOut);

            if (!attn_qkv || !ssm_conv1d_w || !ssm_a || !ssm_alpha_w || !ssm_beta_w || !ssm_dt_bias_t || !ssm_norm_w ||
                !attn_gate_w || !ssm_out_w) {
                throw densecore::InvalidArgumentException("Missing SSM weights in layer " + std::to_string(il));
            }

            const int ssm_ordinal = ssm_ordinal_counter++;
            auto& ssm_rt = model->ssm_layer_states[ssm_ordinal];

            const int d_inner = model->ssm_inner_size;
            const int num_v_heads = model->ssm_time_step_rank;
            const int head_dim_v = d_inner / num_v_heads;
            const int head_dim_k = model->ssm_state_size;
            const int n_groups = model->ssm_group_count;
            const int qk_total = head_dim_k * n_groups;
            const int conv_channels = d_inner + 2 * n_groups * head_dim_k;
            const int conv_kernel = model->ssm_conv_kernel;
            const size_t expected_conv_weights = static_cast<size_t>(conv_channels) * static_cast<size_t>(conv_kernel);
            const size_t expected_head_by_embd = static_cast<size_t>(num_v_heads) * static_cast<size_t>(n_embd);
            const size_t expected_per_head = static_cast<size_t>(num_v_heads);
            const size_t expected_norm_shared = static_cast<size_t>(head_dim_v);
            const size_t expected_norm_full = static_cast<size_t>(d_inner);

            if (ssm_rt.conv1d_f32.size() != expected_conv_weights || ssm_rt.alpha_f32.size() != expected_head_by_embd ||
                ssm_rt.beta_f32.size() != expected_head_by_embd || ssm_rt.dt_bias_f32.size() != expected_per_head ||
                ssm_rt.a_log_f32.size() != expected_per_head ||
                (ssm_rt.norm_layout == Qwen35SSMNormLayout::SHARED_HEAD_DIM &&
                 ssm_rt.norm_f32.size() != expected_norm_shared) ||
                (ssm_rt.norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER &&
                 ssm_rt.norm_f32.size() != expected_norm_full) ||
                ssm_rt.norm_layout == Qwen35SSMNormLayout::INVALID) {
                throw densecore::InvalidArgumentException("Missing canonical hybrid SSM runtime weights in layer " +
                                                          std::to_string(il));
            }

            if (IsSSMNonFiniteDebugEnabled()) {
                auto cb_check_ssm_input = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                             void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    const int* seq_ids = (GetCurrentWorkContext() && GetCurrentWorkContext()->batch)
                                             ? GetCurrentWorkContext()->batch->seq_id.data()
                                             : nullptr;
                    CheckSSMFiniteTensor(layer_idx, seq_ids, src, "ssm_input", "attn_norm_out");
                    if (dst->data && src->data) {
                        std::memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int ssm_input_layers[128] = {};
                for (int li = 0; li < 128; ++li) ssm_input_layers[li] = li;
                cur = ggml_map_custom1(ctx_c, cur, cb_check_ssm_input, 1, &ssm_input_layers[il]);
            }

            // 1. qkv_mixed projection: normed input [n_embd, N] → [conv_channels, N]
#if defined(__aarch64__) || defined(_M_ARM64)
            const bool qwen36_hybrid_ssm_q8_prefill_projection_set =
                model->variant == ModelVariant::QWEN36 && model->arch_flags.is_hybrid_ssm && N > 1 &&
                attn_qkv->type == GGML_TYPE_Q8_0 && attn_gate_w->type == GGML_TYPE_Q8_0 &&
                ssm_out_w->type == GGML_TYPE_Q8_0;
            const bool qwen35_hybrid_ssm_q6_projection =
                model->variant == ModelVariant::QWEN35 && model->arch_flags.is_hybrid_ssm &&
                (attn_qkv->type == GGML_TYPE_Q6_K || attn_gate_w->type == GGML_TYPE_Q6_K ||
                 ssm_out_w->type == GGML_TYPE_Q6_K);
            const bool prefer_plain_qwen_hybrid_matmul =
                model->arch_flags.is_hybrid_ssm &&
                ((model->variant == ModelVariant::QWEN35 && qwen35_hybrid_ssm_q6_projection) ||
                 (model->variant == ModelVariant::QWEN36 && !qwen36_hybrid_ssm_q8_prefill_projection_set));
#else
            const bool prefer_plain_qwen_hybrid_matmul = false;
#endif
            ggml_tensor* fused_qkv_gate = nullptr;
#if defined(__aarch64__) || defined(_M_ARM64)
            if ((model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                model->arch_flags.is_hybrid_ssm) {
                if (ggml_tensor* fused_w = layer.Get("attn_qkv_gate.cpu_fused_decode")) {
                    fused_qkv_gate = smart_mul_mat(ctx_c, fused_w, cur, model);
                    ggml_set_name(fused_qkv_gate, model->variant == ModelVariant::QWEN36
                                                     ? "qwen36_ssm_qkv_gate_fused_proj"
                                                     : "qwen35_ssm_qkv_gate_fused_proj");
                }
            }
#endif
#if !defined(__aarch64__) && !defined(_M_ARM64)
            // Keep Qwen3.5 on the QA-passing AMX-fused qkv+gate decode alias.
            // The split custom-gemv experiment was rejected on C4
            // (20260616 env A/B: 17.48 -> 17.16 tok/s decode).
            if (model->variant == ModelVariant::QWEN35 && model->arch_flags.is_hybrid_ssm &&
                model->hparams.n_experts > 0 && N == 1) {
                if (ggml_tensor* fused_w = layer.Get("attn_qkv_gate.amx_fused_decode")) {
                    fused_qkv_gate = ggml_mul_mat(ctx_c, fused_w, cur);
                    ggml_set_name(fused_qkv_gate, "qwen35_ssm_qkv_gate_fused_proj");
                }
            }
#endif
            struct ggml_tensor* qkv_mixed =
                fused_qkv_gate ? ggml_view_2d(ctx_c, fused_qkv_gate, conv_channels, N, fused_qkv_gate->nb[1], 0)
                               : (prefer_plain_qwen_hybrid_matmul ? ggml_mul_mat(ctx_c, attn_qkv, cur)
                                                                   : smart_mul_mat(ctx_c, attn_qkv, cur, model));
            if (model->variant == ModelVariant::QWEN35 && !fused_qkv_gate) {
                ggml_set_name(qkv_mixed, "qwen35_ssm_qkv_proj");
            } else if (model->variant == ModelVariant::QWEN36) {
                ggml_set_name(qkv_mixed, "qwen36_ssm_qkv_proj");
            }
            if (IsDebugSSMQkvReferenceEnabled() || IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* qkv_ref_ud = GetProjectionReferenceUserData();
                qkv_ref_ud->weight_tensor = attn_qkv;
                qkv_ref_ud->input_tensor = cur;
                qkv_ref_ud->layer_idx = il;
                qkv_ref_ud->token_seq_ids = batch.seq_id.data();
                qkv_ref_ud->stage = "qkv_proj";
                qkv_ref_ud->var_name = "qkv_mixed";
                struct ggml_tensor* qkv_probe = ggml_map_custom1(ctx_c, qkv_mixed, cb_projection_reference_probe, 1, qkv_ref_ud);
                if (gf) {
                    ggml_build_forward_expand(gf, qkv_probe);
                }
            }
            if (IsSSMNonFiniteDebugEnabled()) {
                auto cb_check_ssm_qkv = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                           void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    const int* seq_ids = (GetCurrentWorkContext() && GetCurrentWorkContext()->batch)
                                             ? GetCurrentWorkContext()->batch->seq_id.data()
                                             : nullptr;
                    CheckSSMFiniteTensor(layer_idx, seq_ids, src, "qkv_proj", "qkv_mixed");
                    if (dst->data && src->data) {
                        std::memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int ssm_qkv_layers[128] = {};
                for (int li = 0; li < 128; ++li) ssm_qkv_layers[li] = li;
                qkv_mixed = ggml_map_custom1(ctx_c, qkv_mixed, cb_check_ssm_qkv, 1, &ssm_qkv_layers[il]);
            }

            // 2. Conv1D: updates conv_state ring buffer, outputs [conv_channels, N]
            SSMConv1DUserData* conv_ud = GetSSMConv1DUserData();
            conv_ud->conv_state = nullptr;
            conv_ud->weight = ssm_rt.conv1d_f32.data();
            conv_ud->channels = conv_channels;
            conv_ud->kernel_size = conv_kernel;
            // Qwen3.5 and Qwen3.6 both apply SiLU after the SSM conv before the delta/recurrent block.
            conv_ud->apply_silu = (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36);
            conv_ud->layer_idx = il;
            conv_ud->ssm_ordinal = ssm_ordinal;
            conv_ud->token_seq_ids = batch.seq_id.data();
            conv_ud->runtime_states = &batch.hybrid_ssm_runtime_states;
            conv_ud->profile = &GetCurrentWorkContext()->qwen36_profile;
            const bool ssm_conv_channel_parallel =
                (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                batch.num_seqs == 1 && N > 1;
            const int ssm_conv_tasks =
                ssm_conv_channel_parallel
                    ? std::min(std::max(1, ResolveInferenceConfig(&batch).num_threads), std::max(1, conv_channels))
                    : 1;
            struct ggml_tensor* qkv_conv = ggml_map_custom1(ctx_c, qkv_mixed, cb_ssm_conv1d, ssm_conv_tasks, conv_ud);
            if (model->variant == ModelVariant::QWEN35) {
                ggml_set_name(qkv_conv, "qwen35_ssm_conv1d");
            } else if (model->variant == ModelVariant::QWEN36) {
                ggml_set_name(qkv_conv, "qwen36_ssm_conv1d");
            }
            if (ShouldRunHiddenSnapshotProbe(il, "ssm_conv1d_out")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "ssm_conv1d_out";
                    hidden_ud->var_name = "qkv_conv";
                    qkv_conv = ggml_map_custom1(ctx_c, qkv_conv, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }

            // 3. z projection and recurrent Qwen3.5 delta-net block.
            struct ggml_tensor* z =
                fused_qkv_gate
                    ? ggml_view_2d(ctx_c, fused_qkv_gate, d_inner, N, fused_qkv_gate->nb[1],
                                   static_cast<size_t>(conv_channels) * static_cast<size_t>(fused_qkv_gate->nb[0]))
                    : (prefer_plain_qwen_hybrid_matmul ? ggml_mul_mat(ctx_c, attn_gate_w, cur)
                                                       : smart_mul_mat(ctx_c, attn_gate_w, cur, model));
            if (model->variant == ModelVariant::QWEN35) {
                ggml_set_name(z, "qwen35_ssm_gate_proj");
            }
            if (model->variant == ModelVariant::QWEN36) {
                ggml_set_name(z, "qwen36_ssm_gate_proj");
            }
            if (IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* z_ref_ud = GetProjectionReferenceUserData();
                z_ref_ud->weight_tensor = attn_gate_w;
                z_ref_ud->input_tensor = cur;
                z_ref_ud->layer_idx = il;
                z_ref_ud->token_seq_ids = batch.seq_id.data();
                z_ref_ud->stage = "gate_proj";
                z_ref_ud->var_name = "z";
                struct ggml_tensor* z_probe = ggml_map_custom1(ctx_c, z, cb_projection_reference_probe, 1, z_ref_ud);
                if (gf) {
                    ggml_build_forward_expand(gf, z_probe);
                }
            }
            SSMQwen35DeltaUserData* scan_ud = GetSSMQwen35DeltaUserData();
            scan_ud->alpha_weight = ssm_rt.alpha_f32.data();
            scan_ud->beta_weight = ssm_rt.beta_f32.data();
            scan_ud->dt_bias = ssm_rt.dt_bias_f32.data();
            scan_ud->a_log = ssm_rt.a_log_f32.data();
            scan_ud->norm_weight = ssm_rt.norm_f32.data();
            scan_ud->ssm_state = nullptr;
            scan_ud->n_embd = n_embd;
            scan_ud->d_inner = d_inner;
            scan_ud->n_heads = num_v_heads;
            scan_ud->head_dim_v = head_dim_v;
            scan_ud->head_dim_k = head_dim_k;
            scan_ud->n_groups = n_groups;
            scan_ud->norm_layout = ssm_rt.norm_layout;
            scan_ud->norm_eps = model->hparams.f_norm_rms_eps;
            scan_ud->layer_idx = il;
            scan_ud->ssm_ordinal = ssm_ordinal;
            scan_ud->projection_profile = model->variant == ModelVariant::QWEN36
                                               ? Qwen35SSMQkvProjectionProfile::QWEN36_OFFICIAL
                                               : Qwen35SSMQkvProjectionProfile::QWEN35_OFFICIAL;
            scan_ud->token_seq_ids = batch.seq_id.data();
            scan_ud->runtime_states = &batch.hybrid_ssm_runtime_states;
            scan_ud->z_tensor = z;
            scan_ud->qkv_tensor = qkv_conv;
            scan_ud->input_tensor = cur;
            scan_ud->alpha_beta_tensor = nullptr;
            scan_ud->profile = &GetCurrentWorkContext()->qwen36_profile;
            const bool qwen_hybrid_ssm_model =
                (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                model->arch_flags.is_hybrid_ssm;
            scan_ud->fast_silu_gate = qwen_hybrid_ssm_model;
            struct ggml_tensor* alpha_beta = nullptr;
            const bool precompute_qwen_hybrid_ssm_scalars =
                qwen_hybrid_ssm_model && batch.num_seqs == 1 && N > 1;
            if (precompute_qwen_hybrid_ssm_scalars) {
                const int alpha_beta_rows = 2 * num_v_heads + 3 * n_groups;
                alpha_beta = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, alpha_beta_rows, N);
                const int alpha_beta_tasks =
                    ResolveTaskCount(&batch, std::max(1, N * (num_v_heads + n_groups)));
                alpha_beta = ggml_map_custom3(ctx_c, alpha_beta, cur, qkv_conv, cb_ssm_alpha_beta_qk_project_map3,
                                               alpha_beta_tasks, scan_ud);
                if (model->variant == ModelVariant::QWEN35) {
                    ggml_set_name(alpha_beta, "qwen35_ssm_alpha_beta_qk");
                } else if (model->variant == ModelVariant::QWEN36) {
                    ggml_set_name(alpha_beta, "qwen36_ssm_alpha_beta_qk");
                }
                scan_ud->alpha_beta_tensor = alpha_beta;
            }
            const bool ssm_delta_head_parallel =
                (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                batch.num_seqs == 1;
            const int ssm_delta_tasks =
                ssm_delta_head_parallel
                    ? std::min(std::max(1, ResolveInferenceConfig(&batch).num_threads),
                               std::max(1, num_v_heads / kSsmDeltaHeadsPerTask))
                    : 1;
            struct ggml_tensor* y = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, d_inner, N);
            y->op = GGML_OP_CUSTOM;
            y->src[0] = z;
            y->src[1] = qkv_conv;
            y->src[2] = cur;
            y->src[3] = alpha_beta;
            struct {
                ggml_custom_op_t fun;
                int n_tasks;
                void* userdata;
            } ssm_delta_params = {cb_ssm_qwen35_delta_custom, ssm_delta_tasks, scan_ud};
            static_assert(sizeof(ssm_delta_params) <= GGML_MAX_OP_PARAMS, "ssm_delta_params too large");
            std::memcpy(y->op_params, &ssm_delta_params, sizeof(ssm_delta_params));
            if (model->variant == ModelVariant::QWEN35) {
                ggml_set_name(y, "qwen35_ssm_delta");
            } else if (model->variant == ModelVariant::QWEN36) {
                ggml_set_name(y, "qwen36_ssm_delta");
            }
            if (ShouldRunHiddenSnapshotProbe(il, "ssm_delta_out")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "ssm_delta_out";
                    hidden_ud->var_name = "y";
                    struct ggml_tensor* y_probe = ggml_map_custom1(ctx_c, y, cb_hidden_snapshot_probe, 1, hidden_ud);
                    if (gf) {
                        ggml_build_forward_expand(gf, y_probe);
                    }
                }
            }

            // 4. Output projection: [d_inner, N] -> [n_embd, N]
            // Qwen x86 C4 uses smart dispatch for all hybrid-SSM projections so
            // qkv/gate/out can share the same Q4_K true-batched admission logic.
            // ARM remains conservative above because C4A validation historically
            // found silent SSM projection corruption outside the native path.
            cur = prefer_plain_qwen_hybrid_matmul ? ggml_mul_mat(ctx_c, ssm_out_w, y)
                                                  : smart_mul_mat(ctx_c, ssm_out_w, y, model);
            if (model->variant == ModelVariant::QWEN35) {
                ggml_set_name(cur, "qwen35_ssm_out_proj");
            }
            if (model->variant == ModelVariant::QWEN36) {
                ggml_set_name(cur, "qwen36_ssm_out_proj");
            }
            if (ShouldRunHiddenSnapshotProbe(il, "ssm_out_pre_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "ssm_out_pre_residual";
                    hidden_ud->var_name = "ssm_out";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }
            if (IsDebugSSMProjectionReferenceEnabled()) {
                ProjectionReferenceUserData* out_ref_ud = GetProjectionReferenceUserData();
                out_ref_ud->weight_tensor = ssm_out_w;
                out_ref_ud->input_tensor = y;
                out_ref_ud->layer_idx = il;
                out_ref_ud->token_seq_ids = batch.seq_id.data();
                out_ref_ud->stage = "ssm_out_proj";
                out_ref_ud->var_name = "ssm_out";
                struct ggml_tensor* out_probe = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, out_ref_ud);
                if (gf) {
                    ggml_build_forward_expand(gf, out_probe);
                }
            }

            // Residual connection
            attn_out = cur;
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
            if (ShouldRunHiddenSnapshotProbe(il, "after_attn_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "after_attn_residual";
                    hidden_ud->var_name = "attn_post_residual";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }
        } else {
            // =================================================================
            // Attention Layer Forward Path
            // =================================================================

            // Q/K/V Projections (using smart dispatcher for Parallel GEMV)
            const bool use_glm_dsa_mla =
                model->arch_flags.is_glm_dsa && q_a && q_a_norm && q_b && kv_a && kv_a_norm && kv_b;
            const bool gemma4_value_from_key = !use_glm_dsa_mla && !wv && wk && model->arch_flags.is_gemma4;
            if (!use_glm_dsa_mla && (!wq || !wk || !wv)) {
                if (gemma4_value_from_key) {
                    // Gemma4 alternative attention omits V projection on some layers.
                    // Those blocks use the raw K projection as the input to v_norm.
                    wv = wk;
                }
            }
            if (!use_glm_dsa_mla && (!wq || !wk || !wv)) {
                const std::string hint =
                    model->arch_flags.is_hybrid_ssm
                        ? " (hybrid SSM model: layer may be misclassified — check layer_types in GGUF)"
                        : "";
                throw densecore::InvalidArgumentException("Missing Q/K/V weights in TransformerLayer " +
                                                          std::to_string(il) + hint);
            }
            struct ggml_tensor* Qcur = nullptr;
            struct ggml_tensor* Kcur = nullptr;
            struct ggml_tensor* Vcur = nullptr;
            struct ggml_tensor* glm_q_resid = nullptr;
            struct ggml_tensor* glm_index_query = nullptr;
            struct ggml_tensor* glm_index_weights = nullptr;
            struct ggml_tensor* glm_index_key = nullptr;
            const bool use_glm_dsa_sparse = use_glm_dsa_mla && cache && cache->has_index_cache &&
                                            cache->index_head_dim == model->glm_index_head_dim &&
                                            decode_only_batch_layout && indexer_wq_b && indexer_wk && indexer_k_norm &&
                                            indexer_weights_proj && model->glm_index_n_heads > 0 &&
                                            model->glm_index_head_dim > 0;

            // Optional fused QKV projection (single pass over input per token).
            // Falls back to per-projection matmul for non-F32/quantized weights.
            struct ggml_tensor* gemma4_fused_qkv_repack = nullptr;
            if (model->arch_flags.is_gemma4 && IsFusedQKVEnabled() && cur->type == GGML_TYPE_F32) {
                gemma4_fused_qkv_repack = layer.Get("attn_qkv.cpu_repack_fused");
                if (gemma4_fused_qkv_repack &&
                    (gemma4_fused_qkv_repack->ne[0] != cur->ne[0] ||
                     gemma4_fused_qkv_repack->ne[1] != wq->ne[1] + wk->ne[1] + wv->ne[1])) {
                    gemma4_fused_qkv_repack = nullptr;
                }
            }
            const bool fused_qkv_supported = !use_glm_dsa_mla && IsFusedQKVEnabled() && cur->type == GGML_TYPE_F32 &&
                                             wq->type == GGML_TYPE_F32 && wk->type == GGML_TYPE_F32 &&
                                             wv->type == GGML_TYPE_F32 && wq->data && wk->data && wv->data &&
                                             wq->ne[0] == cur->ne[0] && wk->ne[0] == cur->ne[0] &&
                                             wv->ne[0] == cur->ne[0] && wq->ne[1] > 0 && wk->ne[1] > 0 && wv->ne[1] > 0;

            if (use_glm_dsa_mla) {
                static bool logged_glm_dsa_path = false;
                if (!logged_glm_dsa_path) {
                    std::cerr << "[DenseCore] GLM-5 DSA path enabled"
                              << (use_glm_dsa_sparse ? " with sparse indexer cache."
                                                     : " without sparse indexer cache; using dense attention.")
                              << std::endl;
                    logged_glm_dsa_path = true;
                }
                const int qk_nope_head_dim = model->glm_qk_nope_head_dim;
                const int qk_rope_head_dim = model->glm_qk_rope_head_dim;
                const int v_head_dim = model->glm_v_head_dim;
                const int kv_lora_rank = model->glm_kv_lora_rank;
                const int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
                const int kv_proj_head_dim = qk_nope_head_dim + v_head_dim;

                if (qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 || v_head_dim <= 0 || kv_lora_rank <= 0) {
                    throw densecore::InvalidArgumentException("Incomplete GLM-5 DSA metadata for MLA projection path");
                }

                glm_q_resid = smart_mul_mat(ctx_c, q_a, cur, model);
                glm_q_resid = apply_weighted_rms_norm(glm_q_resid, q_a_norm, "glm_q_a_norm", il);
                struct ggml_tensor* q_raw = smart_mul_mat(ctx_c, q_b, glm_q_resid, model);

                struct ggml_tensor* kv_a_cur = smart_mul_mat(ctx_c, kv_a, cur, model);
                const int64_t kv_a_dim = kv_a_cur->ne[0];
                if (kv_a_dim < static_cast<int64_t>(kv_lora_rank + qk_rope_head_dim)) {
                    throw densecore::InvalidArgumentException(
                        "GLM-5 kv_a projection is smaller than kv_lora_rank + rope dim");
                }

                struct ggml_tensor* kv_comp = ggml_view_2d(ctx_c, kv_a_cur, kv_lora_rank, N, kv_a_cur->nb[1], 0);
                struct ggml_tensor* k_rope =
                    ggml_view_2d(ctx_c, kv_a_cur, qk_rope_head_dim, N, kv_a_cur->nb[1],
                                 static_cast<size_t>(kv_lora_rank) * ggml_element_size(kv_a_cur));
                kv_comp = ggml_cont(ctx_c, kv_comp);
                k_rope = ggml_cont(ctx_c, k_rope);
                kv_comp = apply_weighted_rms_norm(kv_comp, kv_a_norm, "glm_kv_a_norm", il);
                struct ggml_tensor* kv_raw = smart_mul_mat(ctx_c, kv_b, kv_comp, model);

                if (q_raw->ne[0] != static_cast<int64_t>(q_head_dim * n_head)) {
                    throw densecore::InvalidArgumentException("GLM-5 q_b projection shape mismatch");
                }
                if (kv_raw->ne[0] != static_cast<int64_t>(kv_proj_head_dim * n_head_kv)) {
                    throw densecore::InvalidArgumentException("GLM-5 kv_b projection shape mismatch");
                }

                GLMDSAPackUserData* q_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* k_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                GLMDSAPackUserData* v_pack_ud = AllocateGLMDSAPackUserData(ctx_c);
                if (!q_pack_ud || !k_pack_ud || !v_pack_ud) {
                    throw densecore::OutOfMemoryException("Failed to allocate GLM-5 DSA packing userdata");
                }
                *q_pack_ud = {n_head, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *k_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
                *v_pack_ud = {n_head_kv, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};

                struct ggml_tensor* q_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head, N);
                struct ggml_tensor* k_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, q_head_dim * n_head_kv, N);
                struct ggml_tensor* v_packed = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, v_head_dim * n_head_kv, N);

                // Repack [nope|rope] into [rope|nope] so the existing partial-RoPE path
                // can operate on the leading rope dimensions.
                Qcur = ggml_map_custom2(ctx_c, q_packed, q_raw, cb_pack_glm_dsa_q, 1, q_pack_ud);
                Kcur = ggml_map_custom3(ctx_c, k_packed, kv_raw, k_rope, cb_pack_glm_dsa_k, 1, k_pack_ud);
                Vcur = ggml_map_custom2(ctx_c, v_packed, kv_raw, cb_pack_glm_dsa_v, 1, v_pack_ud);

                if (use_glm_dsa_sparse) {
                    glm_index_query = smart_mul_mat(ctx_c, indexer_wq_b, glm_q_resid, model);
                    glm_index_weights = smart_mul_mat(ctx_c, indexer_weights_proj, cur, model);
                    glm_index_weights =
                        ggml_scale(ctx_c, glm_index_weights, 1.0f / std::sqrt((float)model->glm_index_n_heads));
                    glm_index_key = smart_mul_mat(ctx_c, indexer_wk, cur, model);
                    glm_index_key = apply_weighted_rms_norm(glm_index_key, indexer_k_norm, "glm_index_k_norm", il);
                }
            } else if (gemma4_fused_qkv_repack) {
                const int dim_q_fused = static_cast<int>(wq->ne[1]);
                const int dim_k_fused = static_cast<int>(wk->ne[1]);
                const int dim_v_fused = static_cast<int>(wv->ne[1]);
                const int merged_dim = dim_q_fused + dim_k_fused + dim_v_fused;

                struct ggml_tensor* qkv_mixed = ggml_mul_mat(ctx_c, gemma4_fused_qkv_repack, cur);
                char qkv_name[96];
                std::snprintf(qkv_name, sizeof(qkv_name), "blk.%d.attn_qkv.cpu_repack_fused", il);
                ggml_set_name(qkv_mixed, qkv_name);

                const size_t k_offset = static_cast<size_t>(dim_q_fused) * sizeof(float);
                const size_t v_offset = static_cast<size_t>(dim_q_fused + dim_k_fused) * sizeof(float);
                Qcur = ggml_view_2d(ctx_c, qkv_mixed, dim_q_fused, N, qkv_mixed->nb[1], 0);
                Kcur = ggml_view_2d(ctx_c, qkv_mixed, dim_k_fused, N, qkv_mixed->nb[1], k_offset);
                Vcur = ggml_view_2d(ctx_c, qkv_mixed, dim_v_fused, N, qkv_mixed->nb[1], v_offset);
                (void)merged_dim;
            } else if (fused_qkv_supported) {
                const int dim_q_fused = static_cast<int>(wq->ne[1]);
                const int dim_k_fused = static_cast<int>(wk->ne[1]);
                const int dim_v_fused = static_cast<int>(wv->ne[1]);
                const int n_embd_fused = static_cast<int>(cur->ne[0]);
                const int merged_dim = dim_q_fused + dim_k_fused + dim_v_fused;

                struct ggml_tensor* qkv_input = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx_c, cur);
                struct ggml_tensor* qkv_merged = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, merged_dim, N);

                QKVUserData* qkv_ud = GetQKVUserData();
                qkv_ud->w_q = reinterpret_cast<const float*>(wq->data);
                qkv_ud->w_k = reinterpret_cast<const float*>(wk->data);
                qkv_ud->w_v = reinterpret_cast<const float*>(wv->data);
                qkv_ud->n_embd = n_embd_fused;
                qkv_ud->dim_q = dim_q_fused;
                qkv_ud->dim_k = dim_k_fused;
                qkv_ud->dim_v = dim_v_fused;

                const int n_tasks = ResolveTaskCount(&batch, std::max(1, merged_dim));
                qkv_merged = ggml_map_custom2(ctx_c, qkv_merged, qkv_input, cb_compute_qkv_map2, n_tasks, qkv_ud);

                const size_t k_offset = static_cast<size_t>(dim_q_fused) * sizeof(float);
                const size_t v_offset = static_cast<size_t>(dim_q_fused + dim_k_fused) * sizeof(float);
                Qcur = ggml_view_2d(ctx_c, qkv_merged, dim_q_fused, N, qkv_merged->nb[1], 0);
                Kcur = ggml_view_2d(ctx_c, qkv_merged, dim_k_fused, N, qkv_merged->nb[1], k_offset);
                Vcur = ggml_view_2d(ctx_c, qkv_merged, dim_v_fused, N, qkv_merged->nb[1], v_offset);
            } else {
#if defined(__aarch64__) || defined(_M_ARM64)
                const bool prefer_plain_attention_projections =
                    (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                    model->arch_flags.is_hybrid_ssm;
#else
                const bool prefer_plain_attention_projections = false;
#endif
                Qcur = prefer_plain_attention_projections ? ggml_mul_mat(ctx_c, wq, cur)
                                                          : smart_mul_mat(ctx_c, wq, cur, model);
                Kcur = prefer_plain_attention_projections ? ggml_mul_mat(ctx_c, wk, cur)
                                                          : smart_mul_mat(ctx_c, wk, cur, model);
                if (gemma4_value_from_key) {
                    Vcur = Kcur;
                } else {
                    Vcur = prefer_plain_attention_projections ? ggml_mul_mat(ctx_c, wv, cur)
                                                              : smart_mul_mat(ctx_c, wv, cur, model);
                }
                if (ShouldRunAttentionProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* q_ref_ud = GetProjectionReferenceUserData();
                    q_ref_ud->weight_tensor = wq;
                    q_ref_ud->input_tensor = cur;
                    q_ref_ud->layer_idx = il;
                    q_ref_ud->token_seq_ids = batch.seq_id.data();
                    q_ref_ud->stage = "attn_q_proj";
                    q_ref_ud->var_name = "Qcur";
                    Qcur = ggml_map_custom1(ctx_c, Qcur, cb_projection_reference_probe, 1, q_ref_ud);

                    ProjectionReferenceUserData* k_ref_ud = GetProjectionReferenceUserData();
                    k_ref_ud->weight_tensor = wk;
                    k_ref_ud->input_tensor = cur;
                    k_ref_ud->layer_idx = il;
                    k_ref_ud->token_seq_ids = batch.seq_id.data();
                    k_ref_ud->stage = "attn_k_proj";
                    k_ref_ud->var_name = "Kcur";
                    Kcur = ggml_map_custom1(ctx_c, Kcur, cb_projection_reference_probe, 1, k_ref_ud);

                    ProjectionReferenceUserData* v_ref_ud = GetProjectionReferenceUserData();
                    v_ref_ud->weight_tensor = wv;
                    v_ref_ud->input_tensor = cur;
                    v_ref_ud->layer_idx = il;
                    v_ref_ud->token_seq_ids = batch.seq_id.data();
                    v_ref_ud->stage = "attn_v_proj";
                    v_ref_ud->var_name = "Vcur";
                    Vcur = ggml_map_custom1(ctx_c, Vcur, cb_projection_reference_probe, 1, v_ref_ud);
                }
            }

            // Apply Multi-LoRA
            char name_buf[64];
            auto apply_lora = [&](struct ggml_tensor* dst, struct ggml_tensor* src, const char* suffix) {
                if (batch.lora_map.empty()) {
                    return dst;
                }
                if (!ggml_is_contiguous(dst)) {
                    dst = ggml_cont(ctx_c, dst);
                }
                snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", il, suffix);
                ggml_set_name(dst, name_buf);
                return ggml_map_custom2(ctx_c, dst, src, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            };

            Qcur = apply_lora(Qcur, cur, "attn_q");
            Kcur = apply_lora(Kcur, cur, "attn_k");
            Vcur = apply_lora(Vcur, cur, "attn_v");

            // Add Bias if present (for Qwen2 and some other models)
            if (bq) {
                if (Qcur->ne[0] == bq->ne[0]) {
                    Qcur = ggml_add(ctx_c, Qcur, bq);
                } else {
                    // SKIP BIAS (Safe fallback)
                    // std::cerr << "Skipping BQ mismatch L" << il << std::endl;
                }
            }
            if (bk) {
                if (Kcur->ne[0] == bk->ne[0]) {
                    Kcur = ggml_add(ctx_c, Kcur, bk);
                } else {
                    // SKIP BIAS on mismatch to avoid graph complexity/hangs
                    // std::cerr << "Skipping BK mismatch" << std::endl;
                }
            }
            if (bv) {
                if (Vcur->ne[0] == bv->ne[0]) {
                    Vcur = ggml_add(ctx_c, Vcur, bv);
                } else {
                    // SKIP BIAS
                }
            }

            // ggml_reshape_3d requires contiguous inputs. Fused QKV views can be
            // strided, so materialize contiguous tensors before reshape.
            if (!ggml_is_contiguous(Qcur)) {
                Qcur = ggml_cont(ctx_c, Qcur);
            }
            if (!ggml_is_contiguous(Kcur)) {
                Kcur = ggml_cont(ctx_c, Kcur);
            }
            if (!ggml_is_contiguous(Vcur)) {
                Vcur = ggml_cont(ctx_c, Vcur);
            }

            // Dynamically infer head dimensions from the actual projected tensors
            int dim_q = Qcur->ne[0];
            int dim_k = Kcur->ne[0];
            int dim_v = Vcur->ne[0];

            int n_head_kv = layer_spec ? layer_spec->attention.kv_head_count
                                       : densecore::models::ResolveLayerKVHeadCount(model, il);
            const bool use_runtime_kv_dims = densecore::models::UseRuntimeKVHeadDims(model);
            int head_dim_kv = (!use_runtime_kv_dims && model->hparams.n_embd_head_k > 0)
                                  ? model->hparams.n_embd_head_k
                                  : (n_head_kv > 0 ? (dim_k / n_head_kv) : 0);
            int head_dim_v = (!use_runtime_kv_dims && model->hparams.n_embd_head_v > 0)
                                 ? model->hparams.n_embd_head_v
                                 : (n_head_kv > 0 ? (dim_v / n_head_kv) : 0);
            struct ggml_tensor* attn_gate = nullptr;

            // Qwen3.5 hybrid attention packs [q | gate] per-head, not as two
            // contiguous global halves. Mirror llama.cpp's strided per-head views.
            if (model->arch_flags.is_hybrid_ssm && dim_q > head_dim_kv * n_head) {
                const int64_t q_attn_dim = static_cast<int64_t>(head_dim_kv) * n_head;
                const int64_t q_full_per_head = dim_q / n_head;
                const int64_t q_attn_per_head = head_dim_kv;
                if (q_full_per_head >= q_attn_per_head * 2) {
                    struct ggml_tensor* Qcur_full = Qcur;
                    const size_t elem = ggml_element_size(Qcur_full);
                    struct ggml_tensor* Qcur_head = ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N,
                                                                 elem * q_full_per_head, Qcur_full->nb[1], 0);
                    struct ggml_tensor* gate_head =
                        ggml_view_3d(ctx_c, Qcur_full, q_attn_per_head, n_head, N, elem * q_full_per_head,
                                     Qcur_full->nb[1], elem * q_attn_per_head);
                    Qcur = ggml_cont_2d(ctx_c, Qcur_head, q_attn_dim, N);
                    attn_gate = ggml_cont_2d(ctx_c, gate_head, q_attn_dim, N);
                } else {
                    attn_gate = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1],
                                                              q_attn_dim * ggml_element_size(Qcur)));
                    Qcur = ggml_cont(ctx_c, ggml_view_2d(ctx_c, Qcur, q_attn_dim, N, Qcur->nb[1], 0));
                }
                dim_q = static_cast<int>(q_attn_dim);
            }

            int head_dim_q = dim_q / n_head;

            const bool gemma4_shared_kv_layer =
                layer_spec ? layer_spec->attention.reads_shared_kv
                           : (model->arch_flags.is_gemma4 && densecore::models::Gemma4KVSourceLayer(model, il) != il);
            bool k_done = false;
            bool v_done = false;

            // Reshape Q (Standard)
            ValidateAttentionProjectionShape3D(Qcur, "Qcur", il, head_dim_q, n_head, N, dim_q, n_head);
            Qcur = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);

            // Reshape K/V
            if (!k_done) {
                ValidateAttentionProjectionShape3D(Kcur, "Kcur", il, head_dim_kv, n_head_kv, N, dim_k, n_head_kv);
                Kcur = ggml_reshape_3d(ctx_c, Kcur, head_dim_kv, n_head_kv, N);
            }
            if (!v_done) {
                ValidateAttentionProjectionShape3D(Vcur, "Vcur", il, head_dim_v, n_head_kv, N, dim_v, n_head_kv);
                Vcur = ggml_reshape_3d(ctx_c, Vcur, head_dim_v, n_head_kv, N);
            }

            // =========================================================================
            // Per-Head QK-Norm (Architecture-flag based for Qwen3/Qwen2.5)
            // =========================================================================
            // Qwen3 requires RMS normalization applied per-head, not over the entire
            // embedding dimension. We reshape to [head_dim, n_heads * n_tokens] so that
            // ggml_rms_norm normalizes each head_dim vector independently.
            //
            // Using arch_flags for explicit requirement checking instead of implicit
            // null pointer guards. The tensor null check is kept for safety.
            //
            // Flow:
            //   1. Reshape Q from [head_dim, n_head, N] to [head_dim, n_head * N]
            //   2. Apply ggml_rms_norm (normalizes over ne[0] = head_dim)
            //   3. Multiply by weight [head_dim] (broadcasts across all head*token)
            //   4. Reshape back to [head_dim, n_head, N]
            // =========================================================================
            // Q Normalization (required for Qwen3, optional for others)
            if (densecore::models::ShouldApplyQNorm(model, q_norm)) {
                const int64_t q_n_tokens = Qcur->ne[2];  // N (batch size)
                struct ggml_tensor* q_norm_effective = effective_rms_weight(q_norm, nullptr);
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int qnorm_dbg = 0;
                    if (qnorm_dbg < 2 && q_norm->data && q_norm->type == GGML_TYPE_F32) {
                        const float* qn_raw = reinterpret_cast<const float*>(q_norm->data);
                        fprintf(stderr,
                                "[QNORM_L3_RAW #%d] q_norm_raw[0]=%.6f q_norm_raw[1]=%.6f q_norm_raw[2]=%.6f "
                                "q_norm_raw[3]=%.6f\n",
                                qnorm_dbg, qn_raw[0], qn_raw[1], qn_raw[2], qn_raw[3]);
                    }
                    if (qnorm_dbg < 2 && q_norm_effective->data && q_norm_effective->type == GGML_TYPE_F32) {
                        const float* qn = reinterpret_cast<const float*>(q_norm_effective->data);
                        fprintf(
                            stderr,
                            "[QNORM_L3_EFFECTIVE #%d] q_norm[0]=%.6f q_norm[1]=%.6f q_norm[2]=%.6f q_norm[3]=%.6f\n",
                            qnorm_dbg, qn[0], qn[1], qn[2], qn[3]);
                        qnorm_dbg++;
                    }
                }

                if (head_dim_q == q_norm_effective->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head * n_tokens] for per-head norm
                    struct ggml_tensor* Q_2d = ggml_reshape_2d(ctx_c, Qcur, head_dim_q, n_head * q_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    Q_2d = ensure_rms_norm_f32_input(Q_2d, "Q_2d", il);
                    Q_2d = ggml_rms_norm(ctx_c, Q_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    Q_2d = ggml_mul(ctx_c, Q_2d, q_norm_effective);

                    // Reshape back to original 3D: [head_dim, n_head, n_tokens]
                    Qcur = ggml_reshape_3d(ctx_c, Q_2d, head_dim_q, n_head, q_n_tokens);
                } else if (il == 0) {
                    static bool logged_q_mismatch = false;
                    if (!logged_q_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_q_norm dimension mismatch! "
                                  << "Qcur->ne[0]=" << Qcur->ne[0] << " vs norm->ne[0]=" << q_norm->ne[0]
                                  << ". Skipping Q normalization." << std::endl;
                        logged_q_mismatch = true;
                    }
                }
            }

            // K Normalization (required for Qwen3, optional for others)
            if (densecore::models::ShouldApplyKNorm(model, k_norm, gemma4_shared_kv_layer)) {
                const int64_t k_n_tokens = Kcur->ne[2];  // N (batch size)
                struct ggml_tensor* k_norm_effective = effective_rms_weight(k_norm, nullptr);
                if (IsDebugInferenceStatsEnabled() && il == 3) {
                    static int knorm_dbg = 0;
                    if (knorm_dbg < 2 && k_norm->data && k_norm->type == GGML_TYPE_F32) {
                        const float* kn_raw = reinterpret_cast<const float*>(k_norm->data);
                        fprintf(stderr,
                                "[KNORM_L3_RAW #%d] k_norm_raw[0]=%.6f k_norm_raw[1]=%.6f k_norm_raw[2]=%.6f "
                                "k_norm_raw[3]=%.6f\n",
                                knorm_dbg, kn_raw[0], kn_raw[1], kn_raw[2], kn_raw[3]);
                    }
                    if (knorm_dbg < 2 && k_norm_effective->data && k_norm_effective->type == GGML_TYPE_F32) {
                        const float* kn = reinterpret_cast<const float*>(k_norm_effective->data);
                        fprintf(
                            stderr,
                            "[KNORM_L3_EFFECTIVE #%d] k_norm[0]=%.6f k_norm[1]=%.6f k_norm[2]=%.6f k_norm[3]=%.6f\n",
                            knorm_dbg, kn[0], kn[1], kn[2], kn[3]);
                        knorm_dbg++;
                    }
                }

                if (head_dim_kv == k_norm_effective->ne[0]) {
                    // Reshape to 2D: [head_dim, n_head_kv * n_tokens] for per-head norm
                    struct ggml_tensor* K_2d = ggml_reshape_2d(ctx_c, Kcur, head_dim_kv, n_head_kv * k_n_tokens);

                    // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                    K_2d = ensure_rms_norm_f32_input(K_2d, "K_2d", il);
                    K_2d = ggml_rms_norm(ctx_c, K_2d, model->hparams.f_norm_rms_eps);

                    // Multiply by weight [head_dim] - broadcasts across second dimension
                    K_2d = ggml_mul(ctx_c, K_2d, k_norm_effective);

                    // Reshape back to original 3D: [head_dim, n_head_kv, n_tokens]
                    Kcur = ggml_reshape_3d(ctx_c, K_2d, head_dim_kv, n_head_kv, k_n_tokens);
                } else if (il == 0) {
                    static bool logged_k_mismatch = false;
                    if (!logged_k_mismatch) {
                        std::cerr << "[DenseCore] WARN: attn_k_norm dimension mismatch! "
                                  << "Kcur->ne[0]=" << Kcur->ne[0] << " vs norm->ne[0]=" << k_norm->ne[0]
                                  << ". Skipping K normalization." << std::endl;
                        logged_k_mismatch = true;
                    }
                }
            }

            if (densecore::models::ShouldApplyVNorm(model, Vcur, gemma4_shared_kv_layer)) {
                struct ggml_tensor* v_norm = layer.Get(model_keys::kAttnVNorm);
                const int64_t v_n_tokens = Vcur->ne[2];
                if (head_dim_v > 0 && n_head_kv > 0) {
                    struct ggml_tensor* V_2d = ggml_reshape_2d(ctx_c, Vcur, head_dim_v, n_head_kv * v_n_tokens);
                    V_2d = ensure_rms_norm_f32_input(V_2d, "V_2d", il);
                    V_2d = ggml_rms_norm(ctx_c, V_2d, model->hparams.f_norm_rms_eps);
                    if (v_norm && v_norm->ne[0] == head_dim_v) {
                        V_2d = ggml_mul(ctx_c, V_2d, v_norm);
                    }
                    Vcur = ggml_reshape_3d(ctx_c, V_2d, head_dim_v, n_head_kv, v_n_tokens);
                }
            }

            // Apply RoPE
            // Use n_rot from model params if specified (e.g. for partial RoPE or
            // specific dim) Fallback to full head_dim_q if n_rot is 0
            int rope_dim = model->hparams.n_rot;
            float rope_freq_base = model->hparams.rope_freq_base;
            bool use_gemma4_proportional_rope = false;
            struct ggml_tensor* rope_freq_factors = nullptr;
            if (model->arch_flags.is_gemma4) {
                const bool is_sliding_layer = layer_spec ? layer_spec->attention.is_sliding_window
                                                         : densecore::models::IsGemma4SlidingLayer(model, il);
                if (is_sliding_layer) {
                    rope_dim = model->gemma4_rope_dim_swa > 0 ? model->gemma4_rope_dim_swa : rope_dim;
                } else {
                    // Gemma4 full attention: proportional RoPE via rope_freqs.weight.
                    // The freq_factors tensor has [1.0×64, 1e30×192] for 256 pairs:
                    // pairs 0-63 (freq=1.0) get normal rotation, pairs 64-255 (freq=1e30)
                    // get effectively zero rotation. The GGUF weights are permuted to
                    // match this NEOX-mode pairing convention (same as llama.cpp).
                    use_gemma4_proportional_rope = !densecore::models::IsGemma4FullRopeFreqsDisabled();
                    rope_freq_factors = use_gemma4_proportional_rope ? rope_freqs : nullptr;
                    rope_dim = use_gemma4_proportional_rope ? model->gemma4_rope_dim_full : 0;
                    if (rope_dim <= 0) {
                        rope_dim = static_cast<int>(std::lround(static_cast<float>(head_dim_q) *
                                                                model->gemma4_full_attention_partial_rotary_factor));
                        if (rope_dim <= 0) rope_dim = head_dim_q;
                    }
                    // RoPE pairs operate on an even count of dimensions.
                    rope_dim &= ~1;
                }
                rope_freq_base = is_sliding_layer && model->gemma4_rope_freq_base_swa > 0.0f
                                     ? model->gemma4_rope_freq_base_swa
                                     : model->gemma4_rope_freq_base_full;
            }
            if (rope_dim <= 0) {
                rope_dim = head_dim_q;
            }

            // Ensure rope_dim is valid (<= head_dim)
            if (rope_dim > head_dim_q) rope_dim = head_dim_q;

            // SKIP RoPE if we detected anomaly and sliced (k_done)
            // Layer 0 anomaly (80 dim) is unsafe for 128-dim RoPE/Kernel which expects
            // 128
            bool skip_rope = k_done;
            if (skip_rope) {
                // std::cerr << "[DenseCore] Skipping RoPE for anomalous Layer " << il <<
                // std::endl;
            }

            if (!skip_rope) {
                const int rope_mode = model->arch_flags.is_gemma4 ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
                struct ggml_tensor* Q_rope_fast = nullptr;
                struct ggml_tensor* K_rope_fast = nullptr;
                const bool can_use_precomputed_rope =
                    (DENSECORE_DEFAULT_PRECOMPUTED_ROPE != 0) &&
                    (!model->arch_flags.is_gemma4 ||
                     (use_gemma4_proportional_rope && rope_freq_factors == nullptr && !gemma4_shared_kv_layer));
                if (can_use_precomputed_rope) {
                    Q_rope_fast = ggml_rope_precomputed_table(ctx_c, Qcur, pos, model, rope_dim, &batch);
                    K_rope_fast = ggml_rope_precomputed_table(ctx_c, Kcur, pos, model, rope_dim, &batch);
                }
                const bool use_mrope = ModelUsesMRoPE(model);
                if (gemma4_shared_kv_layer) {
                    if (Q_rope_fast) {
                        Qcur = Q_rope_fast;
                    } else if (use_mrope) {
                        const int rope_mrope_mode = ModelMRoPEMode(model);
                        int rope_sections[GGML_MROPE_SECTIONS] = {
                            model->hparams.rope_sections[0],
                            model->hparams.rope_sections[1],
                            model->hparams.rope_sections[2],
                            model->hparams.rope_sections[3],
                        };
                        Qcur =
                            ggml_rope_multi(ctx_c, Qcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    } else {
                        Qcur = ggml_rope_ext(ctx_c, Qcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    }
                } else if (Q_rope_fast && K_rope_fast) {
                    Qcur = Q_rope_fast;
                    Kcur = K_rope_fast;
                } else {
                    if (use_mrope) {
                        const int rope_mrope_mode = ModelMRoPEMode(model);
                        int rope_sections[GGML_MROPE_SECTIONS] = {
                            model->hparams.rope_sections[0],
                            model->hparams.rope_sections[1],
                            model->hparams.rope_sections[2],
                            model->hparams.rope_sections[3],
                        };
                        Qcur =
                            ggml_rope_multi(ctx_c, Qcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur =
                            ggml_rope_multi(ctx_c, Kcur, pos, nullptr, rope_dim, rope_sections, rope_mrope_mode, n_ctx,
                                            rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    } else {
                        // Fallback to standard GGML RoPE when precomputed path is
                        // unavailable for this tensor/layout.
                        Qcur = ggml_rope_ext(ctx_c, Qcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                        Kcur = ggml_rope_ext(ctx_c, Kcur, pos, rope_freq_factors, rope_dim, rope_mode, n_ctx,
                                             rope_freq_base, model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                    }
                }
            }

            if (model->arch_flags.is_gemma4 && N == 1 &&
                densecore::llm::models::IsGemma4SharedKVSourceLayer(model, il)) {
                Kcur = MaybeAttachGemma4SharedKVProbe(ctx_c, Kcur, "pre-cache-write", "K", il, il);
                Vcur = MaybeAttachGemma4SharedKVProbe(ctx_c, Vcur, "pre-cache-write", "V", il, il);
            }

            const bool use_explicit_attention_scale = model->hparams.f_attention_scale > 0.0f;
            if (use_explicit_attention_scale) {
                Qcur = ggml_scale(ctx_c, Qcur, model->hparams.f_attention_scale);
            }

            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                const int index_n_heads = model->glm_index_n_heads;
                const int index_head_dim = model->glm_index_head_dim;
                const int index_rope_dim = std::min(model->glm_qk_rope_head_dim, index_head_dim);

                glm_index_query = ggml_cont(ctx_c, glm_index_query);
                glm_index_weights = ggml_cont(ctx_c, glm_index_weights);
                glm_index_key = ggml_cont(ctx_c, glm_index_key);

                glm_index_query = ggml_reshape_3d(ctx_c, glm_index_query, index_head_dim, index_n_heads, N);
                if (index_rope_dim > 0) {
                    glm_index_query = ggml_rope_ext(ctx_c, glm_index_query, pos, nullptr, index_rope_dim, 0, n_ctx,
                                                    model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                                    1.0f, 0.0f, 0.0f);
                    struct ggml_tensor* index_k_3d = ggml_reshape_3d(ctx_c, glm_index_key, index_head_dim, 1, N);
                    index_k_3d = ggml_rope_ext(ctx_c, index_k_3d, pos, nullptr, index_rope_dim, 0, n_ctx,
                                               model->hparams.rope_freq_base, model->hparams.rope_freq_scale, 0.0f,
                                               1.0f, 0.0f, 0.0f);
                    glm_index_key = ggml_reshape_2d(ctx_c, index_k_3d, index_head_dim, N);
                }
            }

            struct ggml_tensor* KQV = nullptr;
            struct ggml_tensor* attn_ref_k = nullptr;
            struct ggml_tensor* attn_ref_v = nullptr;
            int attn_ref_n_past = 0;
            bool attn_core_reference_eligible = false;
            if (use_glm_dsa_sparse && glm_index_query && glm_index_weights && glm_index_key) {
                PagedAttentionUserData* ud = GetPagedAttentionUserData();
                ud->cache = cache;
                ud->layer = il;
                ud->head_dim = head_dim_q;
                ud->v_head_dim = head_dim_v;
                ud->n_head = n_head;
                ud->index_n_heads = model->glm_index_n_heads;
                ud->index_head_dim = model->glm_index_head_dim;
                ud->index_topk = model->glm_index_topk;
                ud->epoch_started.store(0, std::memory_order_relaxed);
                ud->epoch_done.store(0, std::memory_order_relaxed);
                ud->kv_writers_done.store(0, std::memory_order_relaxed);

                struct ggml_tensor* q_sparse = ggml_is_contiguous(Qcur) ? Qcur : ggml_cont(ctx_c, Qcur);
                struct ggml_tensor* k_sparse = ggml_is_contiguous(Kcur) ? Kcur : ggml_cont(ctx_c, Kcur);
                struct ggml_tensor* v_sparse = ggml_is_contiguous(Vcur) ? Vcur : ggml_cont(ctx_c, Vcur);
                struct ggml_tensor* index_q_sparse =
                    ggml_is_contiguous(glm_index_query) ? glm_index_query : ggml_cont(ctx_c, glm_index_query);
                struct ggml_tensor* index_w_sparse =
                    ggml_is_contiguous(glm_index_weights) ? glm_index_weights : ggml_cont(ctx_c, glm_index_weights);
                struct ggml_tensor* index_k_sparse =
                    ggml_is_contiguous(glm_index_key) ? glm_index_key : ggml_cont(ctx_c, glm_index_key);

                KQV = ggml_glm_dsa_attention(ctx_c, q_sparse, k_sparse, v_sparse, index_q_sparse, index_w_sparse,
                                             index_k_sparse, ud);
            }

            // =========================================================================
            // KV CACHE INTEGRATION (Universal Paged Attention)
            // =========================================================================
            if (!KQV) {
                struct ggml_tensor* K_all = Kcur;  // Default to current K
                struct ggml_tensor* V_all = Vcur;  // Default to current V

                const bool use_cache = (cache != nullptr);
                const BasePagedDecodeExecutionDecision base_paged_decode =
                    densecore::llm::attention::ResolveBasePagedDecodeExecutionDecision(
                        decode_paged_policy, model, cache, batch, N, n_head, n_head_kv, head_dim_q, head_dim_kv);
                const int n_past_val = base_paged_decode.n_past_val;
                const int attn_query_base_pos = debug_query_base_pos;
                attn_ref_n_past = attn_query_base_pos;
                const int n_total_tokens = n_past_val + N;
                const DecodePagedDecision paged_decode_decision = base_paged_decode.paged_decode_decision;
                const bool paged_decode_candidate = paged_decode_decision.candidate;
                const bool requested_paged_decode_attention = paged_decode_decision.requested;
                const bool decode_only_batch = base_paged_decode.decode_only_batch;
                const int kv_cache_layer = layer_spec ? layer_spec->attention.kv_source_layer
                                                      : densecore::models::Gemma4KVSourceLayer(model, il);
                const bool gemma4_shared_kv_source_layer =
                    layer_spec ? layer_spec->attention.publishes_shared_kv
                               : densecore::llm::models::IsGemma4SharedKVSourceLayer(model, il);
                const bool gemma4_shared_kv_explicit_state_disabled =
                    model->arch_flags.is_gemma4 && densecore::models::IsGemma4SharedKVExplicitStateDisabled();

                // Safety override: GGML's generic decode matmul path can become numerically
                // unstable for GQA decode (N=1, n_head != n_head_kv) on some CPU kernels.
                // Force the custom paged decode attention path for correctness in this case.
                const bool paged_decode_supported = base_paged_decode.paged_decode_supported;
                const bool disable_paged_decode_for_model = !paged_decode_supported;
                const bool force_batched_decode_path = base_paged_decode.force_batched_decode_path;
                if (disable_paged_decode_for_model && requested_paged_decode_attention) {
                    static bool logged_gemma4_paged_decode_disable = false;
                    if (!logged_gemma4_paged_decode_disable) {
                        std::cerr << "[DenseCore] Gemma4 paged decode attention is disabled by support gate; "
                                     "falling back to the standard decode path."
                                  << std::endl;
                        logged_gemma4_paged_decode_disable = true;
                    }
                }
                const densecore::DeviceType preferred_attention_device = ResolvePreferredAttentionDevice(&batch);
                const auto attention_dispatch = densecore::llm::attention::ResolveDecodeAttentionDispatchDecision(
                    model, base_paged_decode, use_cache, il, N, n_past_val, n_head, n_head_kv, head_dim_q, head_dim_kv,
                    head_dim_v, preferred_attention_device, IsFlashAttentionDisabled(), false,
                    IsFlashAttentionIsaSupported(), true,
                    densecore::OpsRegistry::IsInitialized(), IsForceSafeGqaDecodeEnabled(), -1,
                    ggml_is_contiguous(Qcur), ggml_is_contiguous(K_all),
                    ggml_is_contiguous(V_all), batch.num_seqs == 1 && !batch.seq_id.empty());
                const bool use_paged_decode_attention = attention_dispatch.use_paged_decode_attention;
                if (force_batched_decode_path && !base_paged_decode.use_paged_decode_attention) {
                    static bool logged_force_batched_decode = false;
                    if (!logged_force_batched_decode) {
                        std::cerr << "[DenseCore] Forcing paged decode attention for decode-only batched scheduling "
                                  << "for sequence-isolated correctness " << "(N=" << N << ")" << std::endl;
                        logged_force_batched_decode = true;
                    }
                }
                if (!base_paged_decode.use_paged_decode_attention && attention_dispatch.force_safe_gqa_decode &&
                    !attention_dispatch.prefer_portable_cpu_flash_safe_decode) {
                    static bool logged_force_safe_decode = false;
                    if (!logged_force_safe_decode) {
                        std::cerr << "[DenseCore] Forcing paged decode attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_force_safe_decode = true;
                    }
                } else if (!base_paged_decode.use_paged_decode_attention &&
                           attention_dispatch.prefer_portable_cpu_flash_safe_decode) {
                    static bool logged_safe_decode_flash = false;
                    if (!logged_safe_decode_flash) {
                        std::cerr << "[DenseCore] Using portable CPU flash attention for GQA decode safety "
                                  << "(N=1, n_head=" << n_head << ", n_head_kv=" << n_head_kv << ")" << std::endl;
                        logged_safe_decode_flash = true;
                    }
                }

                if (use_cache && !use_paged_decode_attention) {
                    // Only need fancy logic if we have history.
                    // If n_past = 0 (Prefill), K_all == Kcur is mostly fine,
                    // BUT we still need to WRITE to cache.
                    // The 'ggml_pad' trick updates cache as side effect.
                    // So we act always if use_cache is true.
                    struct ggml_tensor* cache_k_src = Kcur;
                    struct ggml_tensor* cache_v_src = Vcur;
                    if (model->arch_flags.is_gemma4 && N == 1) {
                        // Single-token Gemma4 decode can reach this path with
                        // non-dense 3D views. Materialize the current K/V slice
                        // before cache write/gather so the source-layer publish
                        // path does not silently write zeros for the appended token.
                        cache_k_src = ggml_cont(ctx_c, Kcur);
                        cache_v_src = ggml_cont(ctx_c, Vcur);
                    }

                    KVCacheUserData* k_ud = GetKVCacheUserData(il, true);
                    *k_ud = {cache, kv_cache_layer, head_dim_kv, true};  // batch accessed via GetCurrentBatch()
                    k_ud->read_only_shared_kv = gemma4_shared_kv_layer;
                    k_ud->force_full_history = gemma4_shared_kv_source_layer;
                    KVCacheUserData* v_ud = GetKVCacheUserData(il, false);
                    *v_ud = {cache, kv_cache_layer, head_dim_v, false};  // batch accessed via GetCurrentBatch()
                    v_ud->read_only_shared_kv = gemma4_shared_kv_layer;
                    v_ud->force_full_history = gemma4_shared_kv_source_layer;

                    int kv_tasks = ResolveInferenceConfig(&batch).num_threads;
                    if (kv_tasks <= 0) {
                        kv_tasks = std::thread::hardware_concurrency();
                        if (kv_tasks <= 0) kv_tasks = 4;
                    }
                    int physical_cores = ResolveHardwareTopology(&batch).GetPhysicalCoreCount();
                    if (physical_cores > 0) {
                        kv_tasks = std::min(kv_tasks, physical_cores);
                    }
                    kv_tasks = std::max(1, kv_tasks);
                    kv_tasks = std::min(kv_tasks, std::max(1, n_head_kv));
                    {
                        // Escape hatch for platform-specific troubleshooting.
                        const char* env = std::getenv("DENSECORE_KV_CALLBACK_SINGLE_THREAD");
                        const bool force_single = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
                        if (force_single) {
                            kv_tasks = 1;
                        }
                    }

                    if (gemma4_shared_kv_layer) {
                        if (gemma4_shared_kv_explicit_state_disabled) {
                            K_all = ggml_kv_update_and_gather(ctx_c, Kcur, n_total_tokens, kv_tasks, k_ud);
                            V_all = ggml_kv_update_and_gather(ctx_c, Vcur, n_total_tokens, kv_tasks, v_ud);
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, K_all, "cache-read", "K", il, kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, V_all, "cache-read", "V", il, kv_cache_layer);
                        } else {
                            const Gemma4SharedKVState* shared_kv_state = GetGemma4SharedKVState(kv_cache_layer);
                            if (!shared_kv_state) {
                                throw densecore::InvalidArgumentException(
                                    "Gemma4 shared-KV layer is missing per-forward shared_kv_states for its source "
                                    "layer");
                            }
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->k, "explicit-read", "K", il,
                                                                   kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->v, "explicit-read", "V", il,
                                                                   kv_cache_layer);
                        }
                    } else if (n_past_val == 0) {
                        // Prefill first chunk fast path: avoid materializing an
                        // equivalent [history | current] tensor when history is empty.
                        K_all = ggml_map_custom1(ctx_c, cache_k_src, cb_kv_write_only, kv_tasks, k_ud);
                        V_all = ggml_map_custom1(ctx_c, cache_v_src, cb_kv_write_only, kv_tasks, v_ud);
                    } else {
                        // Prefill later-chunk fast path: write current tokens to KV cache,
                        // gather retained history directly into the final [head_dim, n_head_kv, n_total]
                        // tensor, and append current tokens without routing through ggml_pad.
                        K_all = ggml_kv_update_and_gather(ctx_c, cache_k_src, n_total_tokens, kv_tasks, k_ud);
                        V_all = ggml_kv_update_and_gather(ctx_c, cache_v_src, n_total_tokens, kv_tasks, v_ud);
                    }
                }

                if (model->arch_flags.is_gemma4) {
                    if (gemma4_shared_kv_layer) {
                        if (!use_cache && !gemma4_shared_kv_explicit_state_disabled) {
                            const Gemma4SharedKVState* shared_kv_state = GetGemma4SharedKVState(kv_cache_layer);
                            if (!shared_kv_state) {
                                throw densecore::InvalidArgumentException(
                                    "Gemma4 shared-KV layer is missing per-forward shared_kv_states for its source "
                                    "layer");
                            }
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->k, "explicit-read", "K", il,
                                                                   kv_cache_layer);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, shared_kv_state->v, "explicit-read", "V", il,
                                                                   kv_cache_layer);
                        }
                    } else if (gemma4_shared_kv_source_layer) {
                        if (!gemma4_shared_kv_explicit_state_disabled) {
                            K_all = MaybeAttachGemma4SharedKVProbe(ctx_c, K_all, "publish", "K", il, il);
                            V_all = MaybeAttachGemma4SharedKVProbe(ctx_c, V_all, "publish", "V", il, il);
                            SetGemma4SharedKVState(il, K_all, V_all);
                        }
                    }
                }

                // =========================================================================
                // NEW: Robust KV Cache Integration (Replaces ggml_pad approach)
                // =========================================================================
                // This approach explicitly:
                //   1. Allocates destination tensors with full size [head_dim, n_head_kv,
                //   n_total]
                //   2. Uses cb_kv_update_and_gather to write cache, read history, append
                //   current
                //   3. Does NOT rely on ggml_pad padding behavior which was causing
                //   NaN/hangs
                // =========================================================================
                // After projection and reshape:

                // Q: [head_dim_q, n_head, N]
                // K_all: [head_dim_kv, n_head_kv, n_past + N]
                // V_all: [head_dim_kv, n_head_kv, n_past + N]

                // =========================================================================
                // GQA (Grouped Query Attention): LOGICAL BROADCASTING
                // =========================================================================
                // For models like Qwen3 where n_head != n_head_kv (e.g., 32 Q heads, 4 KV
                // heads):
                //
                // OLD APPROACH (REMOVED - caused segfaults and was inefficient):
                //   Used ggml_repeat to physically expand K/V from n_head_kv to n_head.
                //   This allocated 8x more memory and caused OOM/crashes.
                //
                // NEW APPROACH (Logical Broadcasting):
                //   Keep K/V at their original [head_dim, n_head_kv, seq] shape.
                //   The attention kernel computes: kv_head = query_head / (n_head /
                //   n_head_kv) This is zero-copy and memory-efficient.
                //
                // Both ggml_flash_attn_ext and our custom FlashAttentionGQA support this.
                // =========================================================================
                struct ggml_tensor* K = K_all;
                struct ggml_tensor* V = V_all;
                int attn_kv_start_pos = 0;
                attn_ref_k = K;
                attn_ref_v = V;
                attn_core_reference_eligible = true;

                // Compute GQA repetition factor for attention dispatch
                const int n_rep = (n_head_kv > 0) ? (n_head / n_head_kv) : 1;
                (void)n_rep;  // Used in attention mask/kernel setup

                // =========================================================================
                // ATTENTION (llama.cpp style - corrected tensor layouts)
                // =========================================================================
                // Tensor shapes at this point:
                //   Q: [head_dim_q, n_head, N]
                //   K: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //   V: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
                //
                // For GQA: The attention kernel handles broadcasting internally.
                // Query heads [0, n_rep) all attend to KV head 0, etc.
                // =========================================================================

                // =========================================================================
                // ATTENTION DISPATCH (Runtime selection based on CPU capabilities)
                // - AVX-512+: Use Flash Attention (ggml_flash_attn_ext) for efficiency
                // - Other: Use standard Q*K^T -> softmax -> V for compatibility
                // =========================================================================
                // Note: Flash Attention still requires AVX-512-class x86 support for
                // correctness/perf in this path.
                const float fast_attn_logit_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
                const int fast_attn_sliding_window =
                    layer_spec ? layer_spec->attention.sliding_window
                               : ((model->arch_flags.is_gemma4 && densecore::models::IsGemma4SlidingLayer(model, il) &&
                                   model->gemma4_sliding_window > 0)
                                      ? model->gemma4_sliding_window
                                      : -1);
                const bool fast_attn_requires_extended_semantics = fast_attn_logit_softcap > 0.0f;
                const uint32_t fast_attn_semantic_flags =
                    fast_attn_requires_extended_semantics ? kFastAttentionSemanticLogitSoftcap : 0u;
                if (attention_dispatch.use_portable_cpu_flash_attention && fast_attn_sliding_window >= 0 &&
                    attn_query_base_pos > fast_attn_sliding_window && K && V && K->ne[2] == V->ne[2]) {
                    const int64_t kv_tokens = K->ne[2];
                    const int64_t drop_tokens =
                        std::min<int64_t>(kv_tokens - 1, attn_query_base_pos - fast_attn_sliding_window);
                    if (drop_tokens > 0 && drop_tokens < kv_tokens) {
                        const int64_t kept_tokens = kv_tokens - drop_tokens;
                        K = ggml_view_3d(ctx_c, K, K->ne[0], K->ne[1], kept_tokens, K->nb[1], K->nb[2],
                                         static_cast<size_t>(drop_tokens) * static_cast<size_t>(K->nb[2]));
                        V = ggml_view_3d(ctx_c, V, V->ne[0], V->ne[1], kept_tokens, V->nb[1], V->nb[2],
                                         static_cast<size_t>(drop_tokens) * static_cast<size_t>(V->nb[2]));
                        attn_kv_start_pos = static_cast<int>(drop_tokens);
                    }
                }

                if (decode_paged_policy.debug_log && il == 0 && (decode_only_batch || N == 1)) {
                    const char* path = use_paged_decode_attention
                                           ? "paged_decode"
                                           : (attention_dispatch.use_hal_attention_dispatch ? "hal_flash"
                                              : attention_dispatch.use_portable_cpu_flash_attention
                                                  ? "cpu_flash_hal"
                                                  : (attention_dispatch.use_flash_attention ? "flash" : "standard"));
                    std::cerr << "[DecodeAttentionPath] N=" << N << " path=" << path << std::endl;
                }

                if (!use_paged_decode_attention) {
                    RecordDecodePagedFallbackReason(paged_decode_decision.reason, il, N);
                }

                if (n_past_val > 0 || decode_only_batch || N == 1) {
                    RecordDecodeAttentionPath(attention_dispatch.attention_path_kind, il, N, n_past_val, n_head,
                                              n_head_kv, preferred_attention_device,
                                              attention_dispatch.use_portable_cpu_flash_native_decode_layout,
                                              paged_decode_candidate, use_paged_decode_attention,
                                              attention_dispatch.portable_cpu_flash_attention_supported,
                                              attention_dispatch.hal_attention_offset_safe);
                }
                if (IsQwen36ProfilingEnabled()) {
                    if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
                        if (use_paged_decode_attention) {
                            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_paged);
                        } else if (attention_dispatch.use_hal_attention_dispatch) {
                            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_hal);
                        } else if (attention_dispatch.use_portable_cpu_flash_attention) {
                            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_portable_flash);
                        } else if (attention_dispatch.use_flash_attention) {
                            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_native_flash);
                        } else {
                            MarkQwen36ProfileFlag(work_ctx->qwen36_profile.attention_path_standard);
                        }
                    }
                }

                if (use_paged_decode_attention) {
                    KQV = ExecutePagedDecodeAttentionPath(ctx_c, model, cache, Qcur, Kcur, Vcur, il, kv_cache_layer,
                                                          gemma4_shared_kv_layer, gemma4_shared_kv_source_layer,
                                                          head_dim_q, head_dim_v, n_head, n_head_kv, n_total_tokens,
                                                          fast_attn_sliding_window, fast_attn_logit_softcap,
                                                          use_explicit_attention_scale);
                } else if (attention_dispatch.use_hal_attention_dispatch) {
                    KQV = ExecuteHalAttentionPath(ctx_c, model, Qcur, K, V, il, N, head_dim_q, n_head_kv,
                                                  fast_attn_sliding_window, fast_attn_logit_softcap,
                                                  fast_attn_semantic_flags, preferred_attention_device,
                                                  use_explicit_attention_scale);
                } else if (attention_dispatch.use_portable_cpu_flash_attention) {
                    KQV = ExecutePortableCpuFlashAttentionPath(
                        ctx_c, model, Qcur, K, V, il, N, head_dim_q, head_dim_kv, head_dim_v, n_head_kv,
                        attn_query_base_pos, attn_kv_start_pos, fast_attn_sliding_window, fast_attn_logit_softcap,
                        fast_attn_semantic_flags, use_explicit_attention_scale,
                        attention_dispatch.use_portable_cpu_flash_native_decode_layout);
                } else if (attention_dispatch.use_flash_attention) {
                    KQV = ExecuteNativeFlashAttentionPath(
                        ctx_c, model, Qcur, K, V, N, n_past_val, n_total_tokens, head_dim_q, n_head_kv,
                        attn_query_base_pos, fast_attn_sliding_window, decode_only_batch, use_explicit_attention_scale,
                        &shared_prefill_flash_mask, &shared_prefill_mask_n_total, &shared_prefill_mask_n_padded,
                        &shared_prefill_mask_n, &shared_prefill_mask_n_past, &shared_prefill_mask_sliding_window);
                } else {
                    KQV = ExecuteStandardAttentionPath(ctx_c, model, Qcur, K, V, il, N, n_past_val, n_total_tokens,
                                                       n_head, n_head_kv, head_dim_q, fast_attn_sliding_window,
                                                       attn_query_base_pos, use_explicit_attention_scale);
                }
            }

            // Must be contiguous before reshape
            struct ggml_tensor* KQV_merged = ggml_cont(ctx_c, KQV);
            if (attn_core_reference_eligible && attn_ref_k && attn_ref_v && IsDebugAttentionCoreReferenceEnabled()) {
                AttentionCoreReferenceUserData* attn_ref_ud = GetAttentionCoreReferenceUserData();
                const int attn_ref_sliding_window =
                    layer_spec ? layer_spec->attention.sliding_window
                               : ((model->arch_flags.is_gemma4 && densecore::models::IsGemma4SlidingLayer(model, il) &&
                                   model->gemma4_sliding_window > 0)
                                      ? model->gemma4_sliding_window
                                      : -1);
                const float attn_ref_scale =
                    use_explicit_attention_scale ? 1.0f : 1.0f / sqrtf(static_cast<float>(head_dim_q));
                const float attn_ref_logit_softcap = ResolveGemma4AttentionLogitSoftcapRuntime(model);
                attn_ref_ud->value_tensor = attn_ref_v;
                attn_ref_ud->layer_idx = il;
                attn_ref_ud->n_head = n_head;
                attn_ref_ud->n_head_kv = n_head_kv;
                attn_ref_ud->head_dim_q = head_dim_q;
                attn_ref_ud->head_dim_k = head_dim_kv;
                attn_ref_ud->head_dim_v = head_dim_v;
                attn_ref_ud->n_past = attn_ref_n_past;
                attn_ref_ud->sliding_window = attn_ref_sliding_window;
                attn_ref_ud->attention_scale = attn_ref_scale;
                attn_ref_ud->logit_softcap = attn_ref_logit_softcap;
                attn_ref_ud->token_seq_ids = batch.seq_id.data();
                attn_ref_ud->stage = "attn_core";
                attn_ref_ud->var_name = "KQV_merged";
                KQV_merged = ggml_map_custom3(ctx_c, KQV_merged, Qcur, attn_ref_k, cb_attention_core_reference_probe, 1,
                                              attn_ref_ud);
            }
#ifdef DENSECORE_TEST_BUILD
            if (g_test_capture_attention_layer.load(std::memory_order_relaxed) == il) {
                KQV_merged = ggml_map_custom1(ctx_c, KQV_merged, cb_test_capture_attention_tensor, 1, nullptr);
            }
#endif
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_kqv = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[KQV%d #%d] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f "
                                "max=%.6f mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (int)src->type, (long)src->ne[0], (long)src->ne[1],
                                (long)src->ne[2], n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                KQV_merged = ggml_map_custom1(ctx_c, KQV_merged, cb_check_kqv, 1, &dbg_layers[il]);
            }

            const int attn_out_head_dim = static_cast<int>(KQV_merged->ne[0]);
            cur = ggml_reshape_2d(ctx_c, KQV_merged, attn_out_head_dim * n_head, N);
            if (attn_gate) {
                attn_gate = ggml_sigmoid(ctx_c, attn_gate);
                cur = ggml_mul(ctx_c, cur, attn_gate);
            }
            if (IsDebugAttentionPostReferenceEnabled()) {
                AttentionCoreReferenceUserData* post_ref_ud = GetAttentionCoreReferenceUserData();
                post_ref_ud->gate_tensor = attn_gate;
                post_ref_ud->layer_idx = il;
                post_ref_ud->n_head = n_head;
                post_ref_ud->head_dim_v = head_dim_v;
                post_ref_ud->token_seq_ids = batch.seq_id.data();
                post_ref_ud->stage = "attn_post";
                post_ref_ud->var_name = "cur_input_to_wo";
                cur = ggml_map_custom2(ctx_c, cur, KQV_merged, cb_attention_post_reference_probe, 1, post_ref_ud);
            }
            if (ShouldRunHiddenSnapshotProbe(il, "attn_pre_output_proj")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "attn_pre_output_proj";
                    hidden_ud->var_name = "cur_input_to_wo";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }

            // Output Projection (using smart dispatcher for Parallel GEMV)
            struct ggml_tensor* cur_input_to_wo = cur;
            if (!wo) {
                throw densecore::InvalidArgumentException("Missing attn_output weight in TransformerLayer");
            }
#if defined(__aarch64__) || defined(_M_ARM64)
            const bool prefer_plain_attn_output_matmul =
                (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                model->arch_flags.is_hybrid_ssm;
#else
            const bool prefer_plain_attn_output_matmul = false;
#endif
            cur = prefer_plain_attn_output_matmul ? ggml_mul_mat(ctx_c, wo, cur) : smart_mul_mat(ctx_c, wo, cur, model);
            if (ShouldRunAttentionProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* o_ref_ud = GetProjectionReferenceUserData();
                o_ref_ud->weight_tensor = wo;
                o_ref_ud->input_tensor = cur_input_to_wo;
                o_ref_ud->layer_idx = il;
                o_ref_ud->token_seq_ids = batch.seq_id.data();
                o_ref_ud->stage = "attn_o_proj";
                o_ref_ud->var_name = "attn_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, o_ref_ud);
            }

            // Apply Multi-LoRA to Output Projection
            if (!batch.lora_map.empty()) {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, cur_input_to_wo, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
            if (bo) cur = ggml_add(ctx_c, cur, bo);
            const bool qwen35_post_attn_norm_is_ffn_prenorm =
                model->variant == ModelVariant::QWEN35 && model->arch_flags.is_hybrid_ssm;
            if (auto* post_attn_norm = model->layers[il].Get(model_keys::kPostAttnNorm);
                post_attn_norm && post_attn_norm != ffn_norm && !qwen35_post_attn_norm_is_ffn_prenorm) {
                cur = apply_weighted_rms_norm(cur, post_attn_norm, "post_attention_norm", il);
            }

            if (il == static_cast<int>(model->layers.size()) - 1 && ShouldUsePrefillLastLogitsOnly(model, batch, N)) {
                const size_t last_token_offset = static_cast<size_t>(N - 1) * static_cast<size_t>(cur->nb[1]);
                cur = ggml_view_2d(ctx_c, cur, cur->ne[0], 1, cur->nb[1], last_token_offset);
                cur = ggml_cont(ctx_c, cur);

                const size_t residual_last_token_offset =
                    static_cast<size_t>(N - 1) * static_cast<size_t>(inpL->nb[1]);
                inpL = ggml_view_2d(ctx_c, inpL, inpL->ne[0], 1, inpL->nb[1], residual_last_token_offset);
                inpL = ggml_cont(ctx_c, inpL);
            }

            // Residual Connection
            attn_out = cur;
            if (ShouldRunHiddenSnapshotProbe(il, "attn_out_pre_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "attn_out_pre_residual";
                    hidden_ud->var_name = "attn_out";
                    attn_out = ggml_map_custom1(ctx_c, attn_out, cb_hidden_snapshot_probe, 1, hidden_ud);
                    cur = attn_out;
                }
            }
            attn_post_residual = ggml_add(ctx_c, cur, inpL);
            cur = attn_post_residual;
            if (ShouldRunHiddenSnapshotProbe(il, "after_attn_residual")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "after_attn_residual";
                    hidden_ud->var_name = "attn_post_residual";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }
            if ((il == 1 || il == 3 || il == 4) && IsDebugInferenceStatsEnabled()) {
                auto cb_check_attn = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                        void* ud) {
                    (void)nth;
                    if (ith != 0) return;
                    const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                    if (layer_idx < 0 || layer_idx >= 32) return;
                    static int cb_ct[32] = {0};
                    const bool emit = (cb_ct[layer_idx] < 5);
                    if (emit && src && src->data) {
                        const float* d = reinterpret_cast<const float*>(src->data);
                        const int n = ggml_nelements(src);
                        int zero_ct = 0;
                        int nan_ct = 0;
                        int inf_ct = 0;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        double sum = 0.0;
                        double sum_sq = 0.0;
                        int finite_ct = 0;
                        for (int i = 0; i < n; i++) {
                            const float v = d[i];
                            if (std::isnan(v)) {
                                nan_ct++;
                                continue;
                            }
                            if (!std::isfinite(v)) {
                                inf_ct++;
                                continue;
                            }
                            if (v == 0.0f) zero_ct++;
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                            sum += v;
                            sum_sq += static_cast<double>(v) * static_cast<double>(v);
                            finite_ct++;
                        }
                        if (!std::isfinite(mn)) mn = 0.0f;
                        if (!std::isfinite(mx)) mx = 0.0f;
                        const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                        const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                        fprintf(stderr,
                                "[ATTN%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f "
                                "mean=%.6f rms=%.6f\n",
                                layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct,
                                inf_ct, mn, mx, mean, rms);
                        cb_ct[layer_idx]++;
                    }
                    if (dst && src && dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                };
                static int dbg_layers[32] = {};
                for (int di = 0; di < 32; ++di) dbg_layers[di] = di;
                cur = ggml_map_custom1(ctx_c, cur, cb_check_attn, 1, &dbg_layers[il]);
            }
            // Usually output projection expects n_embd input.
            // wo: [n_embd, n_embd] (or [n_embd, n_head*head_dim])
            // The standard transformer expects concatenation of all heads to be
            // n_embd. If n_head * head_dim_kv != n_embd, we have a mismatch. Qwen3
            // has n_head=16, head_dim_kv=128 => 2048 != 1024. This implies wo expects
            // 2048 input!

            // KQV = ggml_reshape_2d(ctx_c, KQV, n_head * head_dim_kv, N);

            // Output projection
            // cur = ggml_mul_mat(ctx_c, model->layers[il].wo, KQV);
            // if (model->layers[il].bo)
            //   cur = ggml_add(ctx_c, cur, model->layers[il].bo);

            // Residual connection
            // cur = ggml_add(ctx_c, cur, inpL);

        }  // end else (attention path)

        // =========================================================================
        // FFN (shared between SSM and attention layers)
        // =========================================================================
        struct ggml_tensor* inpFF = attn_post_residual;
        if (!ffn_norm) {
            throw densecore::InvalidArgumentException("Missing ffn_norm weight in TransformerLayer");
        }
        if (ShouldRunHiddenSnapshotProbe(il, "pre_ffn_residual_input")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_ids = batch.tokens.data();
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "pre_ffn_residual_input";
                hidden_ud->var_name = "attn_post_residual";
                inpFF = ggml_map_custom1(ctx_c, inpFF, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        bool used_fused_pre_ffn_norm = false;
        const bool use_fused_pre_ffn_norm =
            model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36;
        if (use_fused_pre_ffn_norm && attn_out && inpL && attn_out->type == GGML_TYPE_F32 &&
            inpL->type == GGML_TYPE_F32 && ffn_norm->type == GGML_TYPE_F32 && ffn_norm->data) {
            AddRMSNormUserData* fused_ud = GetAddRMSNormUserData();
            fused_ud->residual = nullptr;
            bind_add_rmsnorm_weight(fused_ud, ffn_norm, "ffn_norm");
            fused_ud->n_embd = static_cast<int>(attn_out->ne[0]);
            fused_ud->n_tokens = static_cast<int>(attn_out->ne[1]);
            fused_ud->eps = model->hparams.f_norm_rms_eps;
            fused_ud->residual_row_stride = static_cast<ptrdiff_t>(inpL->nb[1] / sizeof(float));
            fused_ud->layer_idx = il;
            fused_ud->token_seq_ids = batch.seq_id.data();
            fused_ud->stage = "pre_ffn_add_rmsnorm";
            fused_ud->var_name = "ffn_norm";

            const int n_tasks = ResolveTaskCount(&batch, std::max<int>(1, fused_ud->n_tokens));
            cur = ggml_map_custom2(ctx_c, attn_out, inpL, cb_residual_rmsnorm_fused2, n_tasks, fused_ud);
            ggml_set_name(cur, "pre_ffn_add_rmsnorm_fused");
            used_fused_pre_ffn_norm = true;
        }

        if (!used_fused_pre_ffn_norm) {
            cur = apply_weighted_rms_norm(cur, ffn_norm, "ffn_norm", il);
        }
        if (ShouldRunHiddenSnapshotProbe(il, "after_ffn_norm")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_ids = batch.tokens.data();
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "after_ffn_norm";
                hidden_ud->var_name = "ffn_norm";
                cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        if (layer_uses_moe) {
            // =====================================================================
            // MOE PATH
            // =====================================================================
            const bool is_gemma4_moe =
                layer_spec ? layer_spec->ffn.router == densecore::models::DecoderMoERouter::Gemma4SoftmaxTopK
                           : densecore::models::IsGemma4MoEModel(model, &model->layers[il]);
            struct ggml_tensor* gemma_router_scale = model->layers[il].Get(kGemma4RouterScaleKey);
            struct ggml_tensor* gemma_pre_moe_norm = model->layers[il].Get(kGemma4PreMoeNormKey);
            struct ggml_tensor* gemma_post_shared_norm = model->layers[il].Get(kGemma4PostSharedNormKey);
            struct ggml_tensor* gemma_post_moe_norm = model->layers[il].Get(kGemma4PostMoeNormKey);
            struct ggml_tensor* shared_input = cur;
            struct ggml_tensor* routed_input = cur;
            if (is_gemma4_moe && gemma_pre_moe_norm) {
                routed_input =
                    apply_weighted_rms_norm(inpFF, gemma_pre_moe_norm, "gemma4_pre_feedforward_layernorm_2", il);
            }
            if (ShouldRunHiddenSnapshotProbe(il, "moe_routed_input")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "moe_routed_input";
                    hidden_ud->var_name = "ffn_norm_2";
                    routed_input = ggml_map_custom1(ctx_c, routed_input, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }
            struct ggml_tensor* router_input = cur;
            if (is_gemma4_moe) {
                struct ggml_tensor* router_norm_src =
                    ensure_rms_norm_f32_input(inpFF, "gemma4_router_rms_norm_input", il);
                router_input = ggml_rms_norm(ctx_c, router_norm_src, model->hparams.f_norm_rms_eps);
                ggml_set_name(router_input, "gemma4_router_rms_norm");
                if (gemma_router_scale) {
                    struct ggml_tensor* router_scale = ggml_repeat(ctx_c, gemma_router_scale, router_input);
                    router_input = ggml_mul(ctx_c, router_input, router_scale);
                }
                router_input =
                    ggml_scale(ctx_c, router_input, 1.0f / std::sqrt(static_cast<float>(model->hparams.n_embd)));
            }

            // 1. Router: gate_logits = moe_gate * input
            if (!moe_gate) {
                throw densecore::InvalidArgumentException("Missing moe_gate weight in TransformerLayer");
            }
            struct ggml_tensor* gate_logits = smart_mul_mat(ctx_c, moe_gate, router_input, model);
            {
                char gate_name[64];
                std::snprintf(gate_name, sizeof(gate_name), "blk.%d.moe_gate_logits", il);
                ggml_set_name(gate_logits, gate_name);
            }
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* gate_ref_ud = GetProjectionReferenceUserData();
                gate_ref_ud->weight_tensor = moe_gate;
                gate_ref_ud->input_tensor = router_input;
                gate_ref_ud->layer_idx = il;
                gate_ref_ud->token_seq_ids = batch.seq_id.data();
                gate_ref_ud->stage = "moe_router_proj";
                gate_ref_ud->var_name = "gate_logits";
                if (const auto it = model->int4_weight_bindings.find(moe_gate);
                    it != model->int4_weight_bindings.end()) {
                    gate_ref_ud->int4_packed = reinterpret_cast<const uint8_t*>(it->second.packed->data);
                    gate_ref_ud->int4_scales = reinterpret_cast<const float*>(it->second.scales->data);
                    gate_ref_ud->int4_zeros = reinterpret_cast<const float*>(it->second.zeros->data);
                    gate_ref_ud->int4_group_size = it->second.group_size;
                    gate_ref_ud->int4_k = static_cast<int>(it->second.k);
                    gate_ref_ud->int4_n = static_cast<int>(it->second.n);
                } else if (const auto it = model->fp8_weight_bindings.find(moe_gate);
                           it != model->fp8_weight_bindings.end()) {
                    gate_ref_ud->fp8_packed = reinterpret_cast<const uint8_t*>(it->second.packed->data);
                    gate_ref_ud->fp8_format = it->second.format;
                    gate_ref_ud->fp8_k = static_cast<int>(it->second.k);
                    gate_ref_ud->fp8_n = static_cast<int>(it->second.n);
                }
                gate_logits = ggml_map_custom1(ctx_c, gate_logits, cb_projection_reference_probe, 1, gate_ref_ud);
            }

            // 2. Dispatch
            int moe_top_k = layer_spec ? layer_spec->ffn.top_k : static_cast<int>(model->hparams.n_experts_used);
            densecore::BackendRegistry& moe_backend_registry = ResolveBackendRegistry(&batch);
            densecore::ComputeBackend* moe_preferred_backend =
                moe_backend_registry.Get(ResolvePreferredDevice(&batch));
            auto* native_moe_backend = dynamic_cast<densecore::CpuBackend*>(moe_preferred_backend);
            if (!native_moe_backend) {
                native_moe_backend =
                    dynamic_cast<densecore::CpuBackend*>(moe_backend_registry.Get(densecore::DeviceType::CPU));
            }
            if (!native_moe_backend) {
                native_moe_backend = &densecore::GetTelemetryCpuBackend();
            }
            if (native_moe_backend) {
                const densecore::CpuBackend::ExpertWeights* registered_experts = nullptr;
                int registered_count = 0;
                if (!native_moe_backend->GetRegisteredExpertsView(&model->layers[il], &registered_experts,
                                                                  &registered_count) ||
                    !registered_experts || registered_count <= 0) {
                    auto expert_weights = BuildExpertWeights(&model->layers[il], model);
                    const int n_experts = static_cast<int>(expert_weights.size());
                    native_moe_backend->InitMoEProfiler(&model->layers[il], n_experts);
                    native_moe_backend->RegisterMoEExperts(&model->layers[il], expert_weights);
                }
                EnsureMoERebalanceThread(native_moe_backend);
            }
            struct ggml_tensor* native_moe = TryBuildGemma4NativeMoEGraph(
                ctx_c, gf, model, &model->layers[il], il, routed_input, gate_logits, moe_top_k, native_moe_backend);
            if (!native_moe) {
                native_moe = TryBuildQwen35NativeMoEGraph(ctx_c, gf, model, &model->layers[il], il, routed_input,
                                                          gate_logits, moe_top_k, layer_spec, native_moe_backend);
            }
            if (native_moe) {
                cur = native_moe;
            } else {
                if (is_gemma4_moe) {
                    throw densecore::InvalidArgumentException(
                        "Gemma4 MoE native graph construction failed; refusing slow CPU backend MoE fallback");
                }
                const bool qwen_native_moe = (model->variant == ModelVariant::QWEN35 ||
                                              model->variant == ModelVariant::QWEN36) &&
                                             model->arch_flags.is_hybrid_ssm;
                const bool lfm2_native_moe =
                    model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
                if (qwen_native_moe || lfm2_native_moe) {
                    throw densecore::InvalidArgumentException(
                        "Native MoE fast path unavailable; refusing slow CPU backend MoE fallback");
                }
                MoEUserData* moe_ud = AllocateMoEUserData(ctx_c);
                if (moe_ud) {
                    moe_ud->model = model;
                    moe_ud->layer = &model->layers[il];
                    moe_ud->layer_idx = il;
                    moe_ud->k = moe_top_k;
                    if (IsQwen36ProfilingEnabled()) {
                        if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
                            moe_ud->profile = &work_ctx->qwen36_profile;
                            SetQwen36ProfileMax(work_ctx->qwen36_profile.moe_task_count, 1);
                        }
                    }
                    // Resolve preferred device first, then guarantee CPU fallback for MoE.
                    densecore::BackendRegistry& registry = ResolveBackendRegistry(&batch);
                    densecore::ComputeBackend* preferred_backend = registry.Get(ResolvePreferredDevice(&batch));
                    moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(preferred_backend);
                    if (!moe_ud->backend) {
                        moe_ud->backend = dynamic_cast<densecore::CpuBackend*>(registry.Get(densecore::DeviceType::CPU));
                    }
                    if (!moe_ud->backend) {
                        moe_ud->backend = &densecore::GetTelemetryCpuBackend();
                    }
                    if (moe_ud->backend) {
                        // Expert registration is CPU-side metadata, not ggml arena allocation.
                        // Prefill callback graphs are built under no-alloc planning and still
                        // need registered experts before graph execution.
                        const densecore::CpuBackend::ExpertWeights* registered_experts = nullptr;
                        int registered_count = 0;
                        if (!moe_ud->backend->GetRegisteredExpertsView(moe_ud->layer, &registered_experts,
                                                                       &registered_count) ||
                            !registered_experts || registered_count <= 0) {
                            auto experts = BuildExpertWeights(moe_ud->layer, model);
                            const int n_experts = static_cast<int>(experts.size());
                            moe_ud->backend->InitMoEProfiler(moe_ud->layer, n_experts);
                            moe_ud->backend->RegisterMoEExperts(moe_ud->layer, experts);
                            moe_ud->backend->GetRegisteredExpertsView(moe_ud->layer, &registered_experts,
                                                                      &registered_count);
                        }
                        EnsureMoERebalanceThread(moe_ud->backend);
                        if (registered_experts && registered_count > 0) {
                            moe_ud->experts = registered_experts;
                            moe_ud->n_experts = registered_count;
                            moe_ud->experts_registered = true;
                        }
                    }
                }

                // Use map_custom2: src0=cur, src1=gate_logits
                g_moe_graph_wiring_debug_counter.fetch_add(1, std::memory_order_relaxed);
                if (model->variant == ModelVariant::QWEN35) {
                    ggml_tensor* gate_exps = GetLayerTensorAny(&model->layers[il], {"ffn_gate_exps.weight", "ffn_gate_exps"});
                    ggml_tensor* down_exps =
                        GetLayerTensorAny(&model->layers[il], {"ffn_down_exps.weight", "ffn_down_exps"});
                    RecordQwen35MoEGraphPath(GetCurrentWorkContext(), "cb_moe_forward", moe_top_k, moe_top_k, 0,
                                             gate_exps ? gate_exps->type : GGML_TYPE_COUNT,
                                             down_exps ? down_exps->type : GGML_TYPE_COUNT);
                }
                cur = ggml_map_custom2(ctx_c, routed_input, gate_logits, cb_moe_forward, 1, moe_ud);
                {
                    char moe_name[64];
                    std::snprintf(moe_name, sizeof(moe_name), "blk.%d.moe_forward", il);
                    ggml_set_name(cur, moe_name);
                }
            }
            if (ShouldRunHiddenSnapshotProbe(il, "moe_routed_output")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "moe_routed_output";
                    hidden_ud->var_name = "moe_forward";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }

            // Shared-expert branch runs in parallel with routed experts.
            if (ShouldRunMoESharedDenseBranch(model, layer_spec, is_gemma4_moe, ffn_gate, ffn_up, ffn_down)) {
                if (!is_gemma4_moe && !ffn_shared_gate) {
                    throw densecore::InvalidArgumentException("Missing shared expert gate weight in TransformerLayer");
                }
                DebugLogSharedExpertTensor("shared_input", il, shared_input);
                DebugLogSharedExpertTensor("ffn_gate_w", il, ffn_gate);
                DebugLogSharedExpertTensor("ffn_up_w", il, ffn_up);
                DebugLogSharedExpertTensor("ffn_down_w", il, ffn_down);
                DebugLogSharedExpertTensor("ffn_shared_gate_w", il, ffn_shared_gate);
#if defined(__aarch64__) || defined(_M_ARM64)
                const bool prefer_plain_shared_expert_matmul =
                    (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                    model->arch_flags.is_hybrid_ssm;
#else
                const bool prefer_plain_shared_expert_matmul = false;
#endif
                struct ggml_tensor* shared_gate = nullptr;
                struct ggml_tensor* shared_up = nullptr;
                struct ggml_tensor* shared_gate_up = nullptr;
                const bool lfm2_shortconv_moe =
                    model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
                struct ggml_tensor* shared_gate_up_fused =
                    ((is_gemma4_moe || lfm2_shortconv_moe || model->variant == ModelVariant::QWEN35 ||
                      model->variant == ModelVariant::QWEN36) &&
                     !prefer_plain_shared_expert_matmul && shared_input->type == GGML_TYPE_F32)
                        ? layer.Get("ffn_gate_up.cpu_repack_fused")
                        : nullptr;
                if (shared_gate_up_fused &&
                    (shared_gate_up_fused->ne[0] != shared_input->ne[0] ||
                     shared_gate_up_fused->ne[1] != ffn_gate->ne[1] + ffn_up->ne[1])) {
                    shared_gate_up_fused = nullptr;
                }
                if (shared_gate_up_fused) {
                    shared_gate_up = ggml_mul_mat(ctx_c, shared_gate_up_fused, shared_input);
                    char fused_name[96];
                    std::snprintf(fused_name, sizeof(fused_name), "blk.%d.shared_ffn_gate_up.cpu_repack_fused", il);
                    ggml_set_name(shared_gate_up, fused_name);
                    const size_t up_offset = static_cast<size_t>(ffn_gate->ne[1]) * sizeof(float);
                    shared_gate = ggml_view_2d(ctx_c, shared_gate_up, ffn_gate->ne[1], shared_input->ne[1],
                                               shared_gate_up->nb[1], 0);
                    shared_up = ggml_view_2d(ctx_c, shared_gate_up, ffn_up->ne[1], shared_input->ne[1],
                                             shared_gate_up->nb[1], up_offset);
                } else {
                    shared_gate = prefer_plain_shared_expert_matmul
                                      ? ggml_mul_mat(ctx_c, ffn_gate, shared_input)
                                      : smart_mul_mat(ctx_c, ffn_gate, shared_input, model);
                    shared_up = prefer_plain_shared_expert_matmul
                                    ? ggml_mul_mat(ctx_c, ffn_up, shared_input)
                                    : smart_mul_mat(ctx_c, ffn_up, shared_input, model);
                    char shared_proj_name[96];
                    std::snprintf(shared_proj_name, sizeof(shared_proj_name), "blk.%d.shared_ffn_gate_proj", il);
                    ggml_set_name(shared_gate, shared_proj_name);
                    std::snprintf(shared_proj_name, sizeof(shared_proj_name), "blk.%d.shared_ffn_up_proj", il);
                    ggml_set_name(shared_up, shared_proj_name);
                }
                DebugLogSharedExpertTensor("shared_gate_proj", il, shared_gate);
                DebugLogSharedExpertTensor("shared_up_proj", il, shared_up);
                if (ShouldRunFfnProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* shared_gate_ref_ud = GetProjectionReferenceUserData();
                    shared_gate_ref_ud->weight_tensor = ffn_gate;
                    shared_gate_ref_ud->input_tensor = shared_input;
                    shared_gate_ref_ud->layer_idx = il;
                    shared_gate_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_gate_ref_ud->stage = "shared_expert_gate_proj";
                    shared_gate_ref_ud->var_name = "shared_gate";
                    shared_gate =
                        ggml_map_custom1(ctx_c, shared_gate, cb_projection_reference_probe, 1, shared_gate_ref_ud);

                    ProjectionReferenceUserData* shared_up_ref_ud = GetProjectionReferenceUserData();
                    shared_up_ref_ud->weight_tensor = ffn_up;
                    shared_up_ref_ud->input_tensor = shared_input;
                    shared_up_ref_ud->layer_idx = il;
                    shared_up_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_up_ref_ud->stage = "shared_expert_up_proj";
                    shared_up_ref_ud->var_name = "shared_up";
                    shared_up = ggml_map_custom1(ctx_c, shared_up, cb_projection_reference_probe, 1, shared_up_ref_ud);
                }
                struct ggml_tensor* shared_ffn = nullptr;
                const bool prefer_native_silu_mul = prefer_plain_shared_expert_matmul;
                if (is_gemma4_moe && GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
                    shared_ffn =
                        ggml_map_custom2(ctx_c, shared_gate, shared_up, cb_gelu_mul_fused, GGML_N_TASKS_MAX, nullptr);
                } else if (is_gemma4_moe) {
                    shared_ffn = ggml_geglu_split(ctx_c, shared_gate, shared_up);
                } else if (prefer_native_silu_mul) {
                    shared_ffn = ggml_mul(ctx_c, ggml_silu(ctx_c, shared_gate), shared_up);
                } else {
                    shared_ffn =
                        ggml_map_custom2(ctx_c, shared_gate, shared_up, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
                }
                DebugLogSharedExpertTensor("shared_ffn_pre_down", il, shared_ffn);
                struct ggml_tensor* shared_ffn_input = shared_ffn;
                shared_ffn = prefer_plain_shared_expert_matmul ? ggml_mul_mat(ctx_c, ffn_down, shared_ffn)
                                                               : smart_mul_mat(ctx_c, ffn_down, shared_ffn, model);
                char shared_down_name[96];
                std::snprintf(shared_down_name, sizeof(shared_down_name), "blk.%d.shared_ffn_down_proj", il);
                ggml_set_name(shared_ffn, shared_down_name);
                DebugLogSharedExpertTensor("shared_ffn_post_down", il, shared_ffn);
                if (ShouldRunFfnProjectionReferenceProbe(il)) {
                    ProjectionReferenceUserData* shared_down_ref_ud = GetProjectionReferenceUserData();
                    shared_down_ref_ud->weight_tensor = ffn_down;
                    shared_down_ref_ud->input_tensor = shared_ffn_input;
                    shared_down_ref_ud->layer_idx = il;
                    shared_down_ref_ud->token_seq_ids = batch.seq_id.data();
                    shared_down_ref_ud->stage = "shared_expert_down_proj";
                    shared_down_ref_ud->var_name = "shared_ffn";
                    shared_ffn =
                        ggml_map_custom1(ctx_c, shared_ffn, cb_projection_reference_probe, 1, shared_down_ref_ud);
                }
                if (is_gemma4_moe) {
                    if (gemma_post_shared_norm) {
                        shared_ffn = apply_weighted_rms_norm(shared_ffn, gemma_post_shared_norm,
                                                             "gemma4_post_feedforward_layernorm_1", il);
                    }
                    if (gemma_post_moe_norm) {
                        cur = apply_weighted_rms_norm(cur, gemma_post_moe_norm, "gemma4_post_feedforward_layernorm_2",
                                                      il);
                    }
                } else {
                    struct ggml_tensor* shared_gate_logits = ggml_mul_mat(ctx_c, ffn_shared_gate, shared_input);
                    char shared_scalar_name[96];
                    std::snprintf(shared_scalar_name, sizeof(shared_scalar_name), "blk.%d.shared_ffn_scalar_gate", il);
                    ggml_set_name(shared_gate_logits, shared_scalar_name);
                    struct ggml_tensor* shared_gate_logits_scalar = shared_gate_logits;
                    DebugLogSharedExpertTensor("shared_scalar_gate_proj", il, shared_gate_logits);
                    if (ShouldRunFfnProjectionReferenceProbe(il)) {
                        ProjectionReferenceUserData* shared_scalar_gate_ref_ud = GetProjectionReferenceUserData();
                        shared_scalar_gate_ref_ud->weight_tensor = ffn_shared_gate;
                        shared_scalar_gate_ref_ud->input_tensor = shared_input;
                        shared_scalar_gate_ref_ud->layer_idx = il;
                        shared_scalar_gate_ref_ud->token_seq_ids = batch.seq_id.data();
                        shared_scalar_gate_ref_ud->stage = "shared_expert_scalar_gate_proj";
                        shared_scalar_gate_ref_ud->var_name = "shared_gate_logits";
                        if (const auto it = model->int4_weight_bindings.find(ffn_shared_gate);
                            it != model->int4_weight_bindings.end()) {
                            shared_scalar_gate_ref_ud->int4_packed =
                                reinterpret_cast<const uint8_t*>(it->second.packed->data);
                            shared_scalar_gate_ref_ud->int4_scales =
                                reinterpret_cast<const float*>(it->second.scales->data);
                            shared_scalar_gate_ref_ud->int4_zeros =
                                reinterpret_cast<const float*>(it->second.zeros->data);
                            shared_scalar_gate_ref_ud->int4_group_size = it->second.group_size;
                            shared_scalar_gate_ref_ud->int4_k = static_cast<int>(it->second.k);
                            shared_scalar_gate_ref_ud->int4_n = static_cast<int>(it->second.n);
                        } else if (const auto it = model->fp8_weight_bindings.find(ffn_shared_gate);
                                   it != model->fp8_weight_bindings.end()) {
                            shared_scalar_gate_ref_ud->fp8_packed =
                                reinterpret_cast<const uint8_t*>(it->second.packed->data);
                            shared_scalar_gate_ref_ud->fp8_format = it->second.format;
                            shared_scalar_gate_ref_ud->fp8_k = static_cast<int>(it->second.k);
                            shared_scalar_gate_ref_ud->fp8_n = static_cast<int>(it->second.n);
                        }
                        shared_gate_logits = ggml_map_custom1(ctx_c, shared_gate_logits, cb_projection_reference_probe,
                                                              1, shared_scalar_gate_ref_ud);
                    }
                    struct ggml_tensor* shared_ffn_pre_scalar_gate = shared_ffn;
                    const int shared_gate_tasks = ResolveTaskCount(
                        &batch, std::max<int>(1, static_cast<int>(std::max<int64_t>(1, shared_ffn->ne[1]))));
                    shared_ffn = ggml_map_custom2(ctx_c, shared_ffn, shared_gate_logits_scalar,
                                                  cb_apply_shared_scalar_gate, shared_gate_tasks, nullptr);
                    DebugLogSharedExpertTensor("shared_ffn_post_scalar_gate", il, shared_ffn);
                    if (ShouldRunSharedScalarGateReferenceProbe(il)) {
                        SharedScalarGateReferenceUserData* shared_gate_runtime_ud =
                            GetSharedScalarGateReferenceUserData();
                        shared_gate_runtime_ud->shared_ffn_pre_gate = shared_ffn_pre_scalar_gate;
                        shared_gate_runtime_ud->shared_gate_logits_scalar = shared_gate_logits_scalar;
                        shared_gate_runtime_ud->layer_idx = il;
                        shared_gate_runtime_ud->token_seq_ids = batch.seq_id.data();
                        shared_gate_runtime_ud->stage = "shared_expert_branch";
                        shared_gate_runtime_ud->var_name = "shared_ffn_after_scalar_gate";
                        shared_ffn = ggml_map_custom1(ctx_c, shared_ffn, cb_shared_scalar_gate_reference_probe, 1,
                                                      shared_gate_runtime_ud);
                    }
                }
                cur = ggml_add(ctx_c, cur, shared_ffn);
                if (ShouldRunHiddenSnapshotProbe(il, "moe_after_shared")) {
                    auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                    if (hidden_ud) {
                        hidden_ud->layer_idx = il;
                        hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                        hidden_ud->token_ids = batch.tokens.data();
                        hidden_ud->token_seq_ids = batch.seq_id.data();
                        hidden_ud->stage = "moe_after_shared";
                        hidden_ud->var_name = "moe_plus_shared";
                        cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                    }
                }
            }
        } else {
            // =====================================================================
            // DENSE PATH
            // =====================================================================
            // SwiGLU FFN (using smart dispatcher for INT4 support)
            if (!ffn_gate || !ffn_up || !ffn_down) {
                throw densecore::InvalidArgumentException("Missing FFN weights in TransformerLayer");
            }
#if defined(__aarch64__) || defined(_M_ARM64)
            const bool qwen35_hybrid_ffn =
                model->variant == ModelVariant::QWEN35 && model->arch_flags.is_hybrid_ssm;
            const bool prefer_separate_qwen35_hybrid_ffn_gate_up = qwen35_hybrid_ffn && cur->ne[1] > 1;
#else
            const bool prefer_separate_qwen35_hybrid_ffn_gate_up = false;
#endif
            struct ggml_tensor* w1 = nullptr;
            struct ggml_tensor* w3 = nullptr;
            struct ggml_tensor* fused_gate_up_swiglu = nullptr;
            const bool lfm2_shortconv_moe =
                model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
            struct ggml_tensor* dense_gate_up_fused =
                ((model->arch_flags.is_gemma4 || lfm2_shortconv_moe || model->variant == ModelVariant::QWEN35 ||
                  model->variant == ModelVariant::QWEN36) &&
                 !prefer_separate_qwen35_hybrid_ffn_gate_up && cur->type == GGML_TYPE_F32)
                    ? layer.Get("ffn_gate_up.cpu_repack_fused")
                    : nullptr;
            if (model->variant == ModelVariant::QWEN35 && cur->ne[1] <= 32) {
                if (struct ggml_tensor* decode_fused = layer.Get("ffn_gate_up.cpu_repack_fused_decode")) {
                    dense_gate_up_fused = decode_fused;
                }
            }
            if (dense_gate_up_fused &&
                (dense_gate_up_fused->ne[0] != cur->ne[0] ||
                 dense_gate_up_fused->ne[1] != ffn_gate->ne[1] + ffn_up->ne[1])) {
                dense_gate_up_fused = nullptr;
            }
            if (dense_gate_up_fused) {
                struct ggml_tensor* gate_up = ggml_mul_mat(ctx_c, dense_gate_up_fused, cur);
                char fused_name[96];
                std::snprintf(fused_name, sizeof(fused_name), "blk.%d.ffn_gate_up.cpu_repack_fused", il);
                ggml_set_name(gate_up, fused_name);
                const bool can_fuse_gate_up_activation =
                    !model->arch_flags.is_gemma4 && batch.lora_map.empty() && !ShouldRunFfnProjectionReferenceProbe(il);
                fused_gate_up_swiglu =
                    can_fuse_gate_up_activation
                        ? BuildFusedGateUpSiluMul(ctx_c, gate_up, ffn_gate->ne[1], cur->ne[1], "ffn_silu_mul_fused")
                        : nullptr;
                if (!fused_gate_up_swiglu) {
                    const size_t up_offset = static_cast<size_t>(ffn_gate->ne[1]) * sizeof(float);
                    w1 = ggml_view_2d(ctx_c, gate_up, ffn_gate->ne[1], cur->ne[1], gate_up->nb[1], 0);
                    w3 = ggml_view_2d(ctx_c, gate_up, ffn_up->ne[1], cur->ne[1], gate_up->nb[1], up_offset);
                }
            } else {
                w1 = smart_mul_mat(ctx_c, ffn_gate, cur, model);
                w3 = smart_mul_mat(ctx_c, ffn_up, cur, model);
            }
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* w1_ref_ud = GetProjectionReferenceUserData();
                w1_ref_ud->weight_tensor = ffn_gate;
                w1_ref_ud->input_tensor = cur;
                w1_ref_ud->layer_idx = il;
                w1_ref_ud->token_seq_ids = batch.seq_id.data();
                w1_ref_ud->stage = "ffn_gate_proj";
                w1_ref_ud->var_name = "w1";
                w1 = ggml_map_custom1(ctx_c, w1, cb_projection_reference_probe, 1, w1_ref_ud);

                ProjectionReferenceUserData* w3_ref_ud = GetProjectionReferenceUserData();
                w3_ref_ud->weight_tensor = ffn_up;
                w3_ref_ud->input_tensor = cur;
                w3_ref_ud->layer_idx = il;
                w3_ref_ud->token_seq_ids = batch.seq_id.data();
                w3_ref_ud->stage = "ffn_up_proj";
                w3_ref_ud->var_name = "w3";
                w3 = ggml_map_custom1(ctx_c, w3, cb_projection_reference_probe, 1, w3_ref_ud);
            }

            // Apply Multi-LoRA [FFN Gate/Up]
            if (!fused_gate_up_swiglu && !batch.lora_map.empty()) {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate", il);
                ggml_set_name(w1, name_buf);
                w1 = ggml_map_custom2(ctx_c, w1, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());

                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up", il);
                ggml_set_name(w3, name_buf);
                w3 = ggml_map_custom2(ctx_c, w3, cur, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }

            // Fused SiLU×Mul: silu(w1) * w3 in single pass (avoids intermediate tensor)
            // This saves ~50% memory bandwidth in FFN forward pass.
            if (fused_gate_up_swiglu) {
                cur = fused_gate_up_swiglu;
            } else if (model->arch_flags.is_gemma4 &&
                       GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
                cur = ggml_map_custom2(ctx_c, w1, w3, cb_gelu_mul_fused, GGML_N_TASKS_MAX, nullptr);
            } else if (model->arch_flags.is_gemma4) {
                cur = ggml_geglu_split(ctx_c, w1, w3);
            } else {
                if ((model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                    model->arch_flags.is_hybrid_ssm) {
                    cur = ggml_mul(ctx_c, ggml_silu(ctx_c, w1), w3);
                } else {
                    cur = ggml_map_custom2(ctx_c, w1, w3, cb_silu_mul_fused, GGML_N_TASKS_MAX, nullptr);
                }
            }
            ggml_set_name(cur, "ffn_silu_mul_fused");
            if (ShouldRunHiddenSnapshotProbe(il, "ffn_pre_down")) {
                auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                if (hidden_ud) {
                    hidden_ud->layer_idx = il;
                    hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                    hidden_ud->token_ids = batch.tokens.data();
                    hidden_ud->token_seq_ids = batch.seq_id.data();
                    hidden_ud->stage = "ffn_pre_down";
                    hidden_ud->var_name = "ffn_swiglu";
                    cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                }
            }

            // Apply Multi-LoRA [FFN Down]
            struct ggml_tensor* ffn_input = cur;
            cur = smart_mul_mat(ctx_c, ffn_down, cur, model);
            if (ShouldRunFfnProjectionReferenceProbe(il)) {
                ProjectionReferenceUserData* down_ref_ud = GetProjectionReferenceUserData();
                down_ref_ud->weight_tensor = ffn_down;
                down_ref_ud->input_tensor = ffn_input;
                down_ref_ud->layer_idx = il;
                down_ref_ud->token_seq_ids = batch.seq_id.data();
                down_ref_ud->stage = "ffn_down_proj";
                down_ref_ud->var_name = "ffn_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, down_ref_ud);
            }
            if (!batch.lora_map.empty()) {
                char name_buf[64];
                snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down", il);
                ggml_set_name(cur, name_buf);
                cur = ggml_map_custom2(ctx_c, cur, ffn_input, cb_apply_multi_lora, 1, GetCurrentWorkContext());
            }
        }

        if (auto* gemma_post_ffn_norm = model->layers[il].Get(kGemma4PostFfnNormKey)) {
            if (model->arch_flags.is_gemma4) {
                cur = ensure_rms_norm_f32_input(cur, "gemma4_post_feedforward_layernorm_input", il);
                cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
                cur = ggml_mul(ctx_c, cur,
                               effective_rms_weight(gemma_post_ffn_norm, "gemma4_post_feedforward_layernorm_weight"));
                ggml_set_name(cur, "gemma4_post_feedforward_layernorm");
            } else {
                cur = apply_weighted_rms_norm(cur, gemma_post_ffn_norm, "gemma4_post_feedforward_layernorm", il);
            }
        }
        if (ShouldRunHiddenSnapshotProbe(il, "ffn_out_pre_residual")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_ids = batch.tokens.data();
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "ffn_out_pre_residual";
                hidden_ud->var_name = "ffn_out";
                cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        // Residual connection
        cur = ggml_add(ctx_c, cur, inpFF);
        if (ShouldRunHiddenSnapshotProbe(il, "after_ffn_residual")) {
            auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
            if (hidden_ud) {
                hidden_ud->layer_idx = il;
                hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                hidden_ud->token_ids = batch.tokens.data();
                hidden_ud->token_seq_ids = batch.seq_id.data();
                hidden_ud->stage = "after_ffn_residual";
                hidden_ud->var_name = "layer_output";
                cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
            }
        }

        if (model->arch_flags.is_gemma4 && !gemma4_decode_special_transforms_disabled &&
            !densecore::models::IsGemma4PerLayerInputDisabled() && gemma4_per_layer_inputs &&
            model->gemma4_hidden_size_per_layer_input > 0) {
            auto* per_layer_gate = model->layers[il].Get(model_keys::kGemma4PerLayerInputGate);
            auto* per_layer_proj = model->layers[il].Get(model_keys::kGemma4PerLayerProjection);
            auto* post_per_layer_norm = model->layers[il].Get(model_keys::kGemma4PostPerLayerInputNorm);
            if (per_layer_gate && per_layer_proj && post_per_layer_norm) {
                const int hidden_per_layer = model->gemma4_hidden_size_per_layer_input;
                struct ggml_tensor* per_layer_input =
                    ggml_view_2d(ctx_c, gemma4_per_layer_inputs, hidden_per_layer, N, gemma4_per_layer_inputs->nb[2],
                                 static_cast<size_t>(il) * gemma4_per_layer_inputs->nb[1]);
                struct ggml_tensor* per_layer_delta = smart_mul_mat(ctx_c, per_layer_gate, cur, model);
                per_layer_delta = ggml_gelu(ctx_c, per_layer_delta);
                per_layer_delta = ggml_mul(ctx_c, per_layer_delta, per_layer_input);
                per_layer_delta = smart_mul_mat(ctx_c, per_layer_proj, per_layer_delta, model);
                per_layer_delta = apply_weighted_rms_norm(per_layer_delta, post_per_layer_norm,
                                                          "gemma4_post_per_layer_input_norm", il);
                cur = ggml_add(ctx_c, cur, per_layer_delta);
                if (ShouldRunHiddenSnapshotProbe(il, "after_per_layer_input")) {
                    auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                    if (hidden_ud) {
                        hidden_ud->layer_idx = il;
                        hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                        hidden_ud->token_ids = batch.tokens.data();
                        hidden_ud->token_seq_ids = batch.seq_id.data();
                        hidden_ud->stage = "after_per_layer_input";
                        hidden_ud->var_name = "layer_output";
                        cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                    }
                }
            }
        }
        if (model->arch_flags.is_gemma4 && !gemma4_decode_special_transforms_disabled &&
            !densecore::models::IsGemma4LayerOutputScaleDisabled()) {
            if (auto* layer_output_scale = model->layers[il].Get(model_keys::kGemma4LayerOutputScale)) {
                // Gemma4 checkpoint scalar: scale the full layer output (same as llama.cpp).
                // This matches the training-time behavior where subsequent layers see
                // the scaled hidden state as their input.
                struct ggml_tensor* scale = ggml_repeat(ctx_c, layer_output_scale, cur);
                cur = ggml_mul(ctx_c, cur, scale);
                if (ShouldRunHiddenSnapshotProbe(il, "after_layer_output_scale")) {
                    auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
                    if (hidden_ud) {
                        hidden_ud->layer_idx = il;
                        hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
                        hidden_ud->token_ids = batch.tokens.data();
                        hidden_ud->token_seq_ids = batch.seq_id.data();
                        hidden_ud->stage = "after_layer_output_scale";
                        hidden_ud->var_name = "layer_output";
                        cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
                    }
                }
            }
        }
        if (IsDebugInferenceStatsEnabled()) {
            auto cb_check_layer = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                     void* ud) {
                (void)nth;
                if (ith != 0) return;
                const int layer_idx = ud ? *reinterpret_cast<const int*>(ud) : -1;
                if (layer_idx < 0 || layer_idx >= 128) return;

                static int cb_ct[128] = {0};
                const float* d = reinterpret_cast<const float*>(src->data);
                const int n = ggml_nelements(src);
                if (!d || n <= 0) {
                    if (dst->data && src->data) {
                        memcpy(dst->data, src->data, ggml_nbytes(src));
                    }
                    return;
                }

                int zero_ct = 0;
                int nan_ct = 0;
                int inf_ct = 0;
                float mn = std::numeric_limits<float>::infinity();
                float mx = -std::numeric_limits<float>::infinity();
                double sum = 0.0;
                double sum_sq = 0.0;
                int finite_ct = 0;
                for (int i = 0; i < n; i++) {
                    const float v = d[i];
                    if (std::isnan(v)) {
                        nan_ct++;
                        continue;
                    }
                    if (!std::isfinite(v)) {
                        inf_ct++;
                        continue;
                    }
                    if (v == 0.0f) zero_ct++;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                    sum += v;
                    sum_sq += static_cast<double>(v) * static_cast<double>(v);
                    finite_ct++;
                }

                // Keep early-layer shape/stats visibility while always surfacing NaN/Inf.
                const bool emit_regular = (cb_ct[layer_idx] < 2);
                const bool emit_anomaly = (nan_ct > 0 || inf_ct > 0);
                if (emit_regular || emit_anomaly) {
                    if (!std::isfinite(mn)) mn = 0.0f;
                    if (!std::isfinite(mx)) mx = 0.0f;
                    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
                    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
                    fprintf(stderr,
                            "[LAYER%d #%d] shape=[%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f "
                            "rms=%.6f\n",
                            layer_idx, cb_ct[layer_idx], (long)src->ne[0], (long)src->ne[1], n, zero_ct, nan_ct, inf_ct,
                            mn, mx, mean, rms);
                }

                if (dst->data && src->data) {
                    memcpy(dst->data, src->data, ggml_nbytes(src));
                }
                cb_ct[layer_idx]++;
            };
            static int layers[128] = {};
            for (int li = 0; li < 128; ++li) layers[li] = li;
            void* layer_ud = static_cast<void*>(&layers[il]);
            cur = ggml_map_custom1(ctx_c, cur, cb_check_layer, 1, layer_ud);
        }
    }
    if (moe_wiring_debug) {
        std::fprintf(stderr,
                     "[MOE_WIRING_SUMMARY] reason_counts={wired:%llu,model_no_moe:%llu,layer_flag_false:%llu,"
                     "missing_moe_gate:%llu,no_experts:%llu,dense_replace_gate:%llu,layer_flag_mismatch:%llu}\n",
                     static_cast<unsigned long long>(moe_wiring_reason_counts[0]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[1]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[2]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[3]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[4]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[5]),
                     static_cast<unsigned long long>(moe_wiring_reason_counts[6]));
    }

    // =========================================================================
    // 3. Final Layer Norm and LM Head
    // =========================================================================
    if (ShouldRunHiddenSnapshotProbe(static_cast<int>(model->layers.size()), "pre_output_norm_hidden")) {
        auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
        if (hidden_ud) {
            hidden_ud->layer_idx = static_cast<int>(model->layers.size());
            hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
            hidden_ud->token_ids = batch.tokens.data();
            hidden_ud->token_seq_ids = batch.seq_id.data();
            hidden_ud->stage = "pre_output_norm_hidden";
            hidden_ud->var_name = "cur";
            cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
        }
    }
    if (batch.skip_output_logits && !embedding_mode) {
        ggml_set_name(cur, "prefill_no_logits_output");
        if (gf) {
            ggml_build_forward_expand(gf, cur);
        }
        return cur;
    }

    cur = apply_weighted_rms_norm(cur, model->output_norm, "output_norm", static_cast<int>(model->layers.size()));

    if (embedding_mode) {
        return cur;
    }

    if (ShouldRunHiddenSnapshotProbe(static_cast<int>(model->layers.size()), "pre_lm_head_hidden")) {
        auto* hidden_ud = AllocateHiddenSnapshotUserData(ctx_c);
        if (hidden_ud) {
            hidden_ud->layer_idx = static_cast<int>(model->layers.size());
            hidden_ud->token_idx = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_TOKEN", -1);
            hidden_ud->token_ids = batch.tokens.data();
            hidden_ud->token_seq_ids = batch.seq_id.data();
            hidden_ud->stage = "pre_lm_head_hidden";
            hidden_ud->var_name = "cur";
            cur = ggml_map_custom1(ctx_c, cur, cb_hidden_snapshot_probe, 1, hidden_ud);
        }
    }

    // LM Head projection: [n_embd, N] -> [n_vocab, N]
    struct ggml_tensor* cur_input_to_lm_head = cur;
    if (cur->ne[1] > 1 && ShouldUsePrefillLastLogitsOnly(model, batch, N)) {
        const int64_t actual_tokens = cur->ne[1];
        const size_t last_token_offset = static_cast<size_t>(actual_tokens - 1) * static_cast<size_t>(cur->nb[1]);
        cur_input_to_lm_head = ggml_view_2d(ctx_c, cur, cur->ne[0], 1, cur->nb[1], last_token_offset);
        cur_input_to_lm_head = ggml_cont(ctx_c, cur_input_to_lm_head);
    }
    cur = smart_mul_mat(ctx_c, model->output, cur_input_to_lm_head, model);
    const bool debug_lm_head = IsDebugInferenceStatsEnabled() || std::getenv("DENSECORE_DEBUG_LM_HEAD_TOP") != nullptr;
    if (debug_lm_head) {
        auto cb_check_lm_head_out = [](struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* ud) {
            (void)nth;
            (void)ud;
            if (ith != 0) return;
            static int cb_lm_ct = 0;
            if (src && src->data) {
                const int n_vocab = (int)src->ne[0];
                const int n_tok = (int)src->ne[1];
                const ptrdiff_t row_stride = (ptrdiff_t)(src->nb[1] / sizeof(float));
                const float* d = (const float*)src->data;
                const bool debug_stats = IsDebugInferenceStatsEnabled();
                const char* top_env = std::getenv("DENSECORE_DEBUG_LM_HEAD_TOP");
                int top_n = 0;
                if (top_env && *top_env) {
                    char* end = nullptr;
                    long parsed = std::strtol(top_env, &end, 10);
                    top_n =
                        (end == top_env || (end && *end != '\0') || parsed <= 0) ? 8 : (int)std::min<long>(parsed, 32);
                }
                if (debug_stats && cb_lm_ct < 3) {
                    for (int ti : {0, n_tok - 1}) {
                        if (ti < 0 || ti >= n_tok) continue;
                        const float* row = d + (ptrdiff_t)ti * row_stride;
                        float mn = row[0], mx = row[0];
                        int pos_ct = 0;
                        for (int i = 0; i < n_vocab; i++) {
                            if (row[i] < mn) mn = row[i];
                            if (row[i] > mx) mx = row[i];
                            if (row[i] > 0) pos_ct++;
                        }
                        fprintf(stderr, "[LM_HEAD_OUT #%d] tok=%d/%d vocab=%d min=%.4f max=%.4f pos_ct=%d stride=%ld\n",
                                cb_lm_ct, ti, n_tok, n_vocab, mn, mx, pos_ct, (long)row_stride);
                    }
                    cb_lm_ct++;
                }
                if (top_n > 0 && n_tok > 0) {
                    const float* row = d + (ptrdiff_t)(n_tok - 1) * row_stride;
                    std::vector<std::pair<float, int>> top;
                    top.reserve((size_t)top_n);
                    auto worse_first = [](const auto& a, const auto& b) {
                        if (a.first == b.first) return a.second < b.second;
                        return a.first > b.first;
                    };
                    for (int i = 0; i < n_vocab; ++i) {
                        const float v = row[i];
                        if (!std::isfinite(v)) continue;
                        if ((int)top.size() < top_n) {
                            top.emplace_back(v, i);
                            std::push_heap(top.begin(), top.end(), worse_first);
                        } else if (v > top.front().first || (v == top.front().first && i < top.front().second)) {
                            std::pop_heap(top.begin(), top.end(), worse_first);
                            top.back() = {v, i};
                            std::push_heap(top.begin(), top.end(), worse_first);
                        }
                    }
                    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
                        if (a.first == b.first) return a.second < b.second;
                        return a.first > b.first;
                    });
                    fprintf(stderr, "[LM_HEAD_TOP] tok=%d/%d top_n=%d\n", n_tok - 1, n_tok, top_n);
                    for (const auto& [score, token_id] : top) {
                        fprintf(stderr, "  [TOP] token=%d logit=%.6f\n", token_id, score);
                    }
                    const char* watch_env = std::getenv("DENSECORE_DEBUG_LM_HEAD_WATCH_IDS");
                    if (watch_env && *watch_env) {
                        std::string spec(watch_env);
                        size_t start = 0;
                        while (start < spec.size()) {
                            size_t end = spec.find(',', start);
                            if (end == std::string::npos) end = spec.size();
                            std::string piece = spec.substr(start, end - start);
                            char* parse_end = nullptr;
                            long parsed = std::strtol(piece.c_str(), &parse_end, 10);
                            if (parse_end != piece.c_str() && (!parse_end || *parse_end == '\0') && parsed >= 0 &&
                                parsed < n_vocab) {
                                fprintf(stderr, "  [WATCH] token=%ld logit=%.6f\n", parsed, row[parsed]);
                            }
                            start = end + 1;
                        }
                    }
                }
            }
            if (dst && src && dst->data && src->data) {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
        };
        cur = ggml_map_custom1(ctx_c, cur, cb_check_lm_head_out, 1, nullptr);
    }
    if (ShouldRunFinalProjectionReferenceProbe()) {
        ProjectionReferenceUserData* final_ref_ud = GetProjectionReferenceUserData();
        final_ref_ud->weight_tensor = model->output;
        final_ref_ud->input_tensor = cur_input_to_lm_head;
        final_ref_ud->layer_idx = static_cast<int>(model->layers.size());
        final_ref_ud->token_seq_ids = batch.seq_id.data();
        final_ref_ud->stage = "lm_head";
        final_ref_ud->var_name = "logits";
        cur = ggml_map_custom1(ctx_c, cur, cb_projection_reference_probe, 1, final_ref_ud);
    }
    ggml_set_name(cur, "output");

    // Add to graph
    if (gf) {
        ggml_build_forward_expand(gf, cur);
    }

    return cur;
}
